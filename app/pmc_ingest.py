import os, hashlib, json
from typing import List, Dict, Optional, Tuple
import httpx
from sentence_transformers import SentenceTransformer
from psycopg import connect
from psycopg.rows import dict_row
from dotenv import load_dotenv
import re
from bs4  import BeautifulSoup
import lxml
from datetime import date

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "1200")) # 10/13 note: upped from 600 to 1200 to have fewer chunks, test up to 1800
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))
MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "8"))
MAX_CHARS    = int(os.getenv("MAX_CHARS", "20000"))
MAX_CHUNKS   = int(os.getenv("MAX_CHUNKS", "4"))
EMBED_BATCH  = int(os.getenv("EMBED_BATCH_SIZE", "8"))
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "pubmed"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, token=HUGGINGFACE_HUB_TOKEN)
    return _embedder

def to_vec_lit(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

def rough_tokens(s: str) -> List[str]:
    return s.split()

def chunk_by_tokens(text: str, max_tokens=600, overlap=90) -> List[str]:
    toks = rough_tokens(text)
    out, i = [], 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while i < len(toks):
        out.append(" ".join(toks[i:i+max_tokens]))
        i += step
    return [c for c in out if c.strip()]

async def fetch_bioc_json(pmcid: str) -> Dict | None:
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    headers = {
        "User-Agent": "med-rag/0.1 (contact: you@example.com)",
        "Accept": "application/json",
    }
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(30.0)

    for attempt in range(5):
        try:
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                r = await client.get(url, headers=headers)
                if r.status_code == 404:
                    return None
                if 500 <= r.status_code < 600:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()

                # ✅ Only parse if response looks like JSON
                ctype = r.headers.get("content-type", "")
                if "json" not in ctype.lower():
                    print(f"[BioC] Non-JSON response for {pmcid}: {ctype}", flush=True)
                    return None

                try:
                    return r.json()
                except Exception as e:
                    print(f"[BioC] JSON parse failed for {pmcid}: {e}", flush=True)
                    return None

        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
            delay = min(1.5 * (2 ** attempt), 10.0)
            print(f"[BioC] attempt {attempt+1}/5 for {pmcid} failed: {type(e).__name__} {e}. retrying in {delay:.1f}s", flush=True)
            import asyncio, random
            await asyncio.sleep(delay + random.random() * 0.5)
        except Exception as e:
            print(f"[BioC] fatal error for {pmcid}: {e}", flush=True)
            break

    return None

def passages_from_bioc(doc_json: Dict) -> List[str]:
    """
    Extract readable passages (title + sections) from BioC JSON.
    BioC structure: collection -> documents[] -> passages[] with "text" and "infons"
    """
    texts = []
    coll = doc_json.get("documents") or []
    for d in coll:
        for p in d.get("passages", []):
            t = (p.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts

def upsert_document(
    cur,
    *,
    ext_id: str,
    title: Optional[str],
    source_uri: Optional[str],
    author: Optional[str],
    year_date: Optional[str],     # 'YYYY-01-01' or None (or pass a datetime.date)
    institute: Optional[str],
    source: Optional[str],
    source_id: Optional[str],
) -> int:
    """
    Upsert into documents by ext_id and return doc_id.
    Matches schema: ext_id, title, source_uri, author, year, institute, source, source_id.
    """
    cur.execute(
        """
        INSERT INTO documents
            (ext_id, title, source_uri, author, year, institute, source, source_id)
        VALUES
            (%s,    %s,    %s,         %s,     %s,   %s,        %s,     %s)
        ON CONFLICT ON CONSTRAINT documents_ext_id_key DO UPDATE
        SET
            title       = COALESCE(EXCLUDED.title,       documents.title),
            source_uri  = COALESCE(EXCLUDED.source_uri,  documents.source_uri),
            author      = COALESCE(EXCLUDED.author,      documents.author),
            year        = COALESCE(EXCLUDED.year,        documents.year),
            institute   = COALESCE(EXCLUDED.institute,   documents.institute),
            source      = COALESCE(EXCLUDED.source,      documents.source),
            source_id   = COALESCE(EXCLUDED.source_id,   documents.source_id)
        RETURNING doc_id;
        """,
        (ext_id, title, source_uri, author, year_date, institute, source, source_id),
    )
    row = cur.fetchone()
    return row[0]

def upsert_chunks(cur, doc_id: int, chunks: List[str], embs, pmcid: str):
    rows = []
    for idx, (txt, vec) in enumerate(zip(chunks, embs)):
        chash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        preview = txt[:500]  # store short preview only
        meta = {
            "pmcid": pmcid,
            "chunk_index": idx,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBED_MODEL
        }
        rows.append((
            doc_id, idx, preview, json.dumps(meta), chash, to_vec_lit(vec), EMBED_MODEL
        ))
    cur.executemany("""
        INSERT INTO chunks (doc_id, chunk_index, text, metadata, content_hash, embedding, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
          text = EXCLUDED.text,
          metadata = EXCLUDED.metadata,
          content_hash = EXCLUDED.content_hash,
          embedding = EXCLUDED.embedding,
          embedding_model = EXCLUDED.embedding_model;
    """, rows)

def _pick_year(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None

def _year_from_pub_dates(soup) -> Optional[str]:
    # Choose publication year by priority: ppub > epub > first pub-date
    def _yr(pd):
        if not pd: return None
        y = pd.find("year")
        if y:
            yy = _pick_year(y.get_text(strip=True))
            if yy: return yy
        # sometimes in <string-date> or mixed text
        return _pick_year(pd.get_text(" ", strip=True))

    ppub = soup.find("pub-date", {"pub-type": "ppub"})
    epub = soup.find("pub-date", {"pub-type": "epub"})
    for pd in (ppub, epub):
        y = _yr(pd)
        if y: return y

    # fallback: any pub-date
    any_pd = soup.find("pub-date")
    return _yr(any_pd)

def _dedup_join(items: List[str]) -> Optional[str]:
    seen = set()
    out = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return "; ".join(out) if out else None
def extract_bioc_documents(j):
        """
        Return a list of BioC 'documents' regardless of whether the JSON is:
        - {"documents": [...]}
        - {"collection": {"documents": [...]}}
        - [{"passages": [...]}, ...]  (top-level list)
        - anything else -> []
        """
        if isinstance(j, list):
            return j
        if isinstance(j, dict):
            if "documents" in j and isinstance(j["documents"], list):
                return j["documents"]
            coll = j.get("collection")
            if isinstance(coll, dict) and isinstance(coll.get("documents"), list):
                return coll["documents"]
        return []

def extract_meta_from_bioc(bioc: dict) -> Dict[str, Optional[str]]:
    """
    Try to pull title/year/institute from BioC JSON.
    Returns {'title': str|None, 'year': 'YYYY'|None, 'institutes': 'A; B; C'|None}
    """
    title = None
    year = None
    institutes: List[str] = []

    # BioC can be: {"documents": [...]}, or {"collection":{"documents":[...]}}
    docs = []
    if isinstance(bioc, dict):
        if isinstance(bioc.get("documents"), list):
            docs = bioc["documents"]
        elif isinstance(bioc.get("collection"), dict) and isinstance(bioc["collection"].get("documents"), list):
            docs = bioc["collection"]["documents"]
    elif isinstance(bioc, list):
        docs = bioc

    # Heuristics: look for passages with title-ish infons, and affiliations in infons/text
    for d in docs:
        for p in d.get("passages", []):
            inf = p.get("infons") or {}
            t = (p.get("text") or "").strip()
            itype = (inf.get("type") or "").lower()

            # Title: passage type "title" / "article-title"
            if not title and (itype in {"title", "article-title"} or inf.get("section_type") == "TITLE"):
                if t: title = t

            # Year: many feeds don’t include; try date-like stuff
            if not year:
                cand = inf.get("year") or inf.get("date") or ""
                y = _pick_year(cand) or _pick_year(t)
                if y: year = y

            # Institutes: sometimes in affiliation passages/infons
            if itype in {"affiliation", "aff", "author-affiliation"} and t:
                institutes.append(t)

    return {
        "title": title or None,
        "year": year or None,
        "institutes": _dedup_join(institutes),
        }

async def ingest_one_pmcid(pmcid: str) -> int:
    print(f"🔹 Starting ingestion for {pmcid}", flush=True)

    # Fetch BioC (with retries) first
    bioc = await fetch_bioc_json(pmcid)
    nxml: Optional[str] = None

    passages_texts: List[str] = []
    meta: Dict[str, Optional[str]] = {"title": None, "year": None, "institutes": None, "authors": None}

    # --- Try BioC path ---
    if bioc:
        print(f"🔹 BioC available for {pmcid}", flush=True)
        docs = extract_bioc_documents(bioc)
        for d in docs[:MAX_PASSAGES]:
            for p in (d.get("passages") or []):
                t = (p.get("text") or "").strip()
                if t:
                    passages_texts.append(t)
        # metadata from BioC (best effort)
        try:
            m_bioc = extract_meta_from_bioc(bioc)
            for k in ("title", "year", "institutes"):
                if m_bioc.get(k):
                    meta[k] = m_bioc[k]
        except Exception as e:
            print(f"[BioC] meta parse warning: {e}", flush=True)

    # --- Fallback to NXML if BioC missing/sparse ---
    if not passages_texts:
        print("⚠️ No passages via BioC — trying NXML fallback", flush=True)
        nxml = await fetch_pmc_nxml(pmcid)
        if not nxml:
            print("❌ NXML fetch failed", flush=True)
            return 0

        passages_texts = extract_text_from_nxml(nxml)
        if not passages_texts:
            print("❌ NXML parse produced no text", flush=True)
            return 0

        try:
            m_nxml = extract_meta_from_nxml(nxml)
            # fill gaps including authors
            for k in ("title", "year", "institutes", "authors"):
                if not meta.get(k) and m_nxml.get(k):
                    meta[k] = m_nxml[k]
        except Exception as e:
            print(f"[NXML] meta parse warning: {e}", flush=True)

    # --- Prepare text ---
    full_text = "\n\n".join(passages_texts)
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS]
    print(f"🔹 Collected {len(passages_texts)} passages ({len(full_text)} chars)", flush=True)

    # --- Chunk ---
    raw_chunks = chunk_by_tokens(full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
    if MAX_CHUNKS:
        raw_chunks = raw_chunks[:MAX_CHUNKS]
    source_id = pmcid  # do not use variable name 'id'
    chunks = [f"{source_id}\n\n{c}" for c in raw_chunks]
    print(f"🔹 Created {len(chunks)} chunks (cap={MAX_CHUNKS})", flush=True)
    if not chunks:
        print("⚠️ No chunks after splitting", flush=True)
        return 0

    # --- Embed ---
    print(f"🔹 Encoding with {EMBED_MODEL} (batch={EMBED_BATCH})", flush=True)
    embs = embedder().encode(chunks, normalize_embeddings=True, batch_size=EMBED_BATCH)
    print(f"🔹 Embedding shape: {embs.shape}", flush=True)

    # --- Upsert document + chunks ---
    def normalize_uri(u: str) -> str:
        u = (u or "").strip()
        return u.rstrip("/").lower()

    year_date = date(int(meta["year"]), 1, 1) if meta.get("year") else None

    print(f"🔹 Connecting to DB: {PG_KWARGS['dbname']}", flush=True)
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        ext_id = f"pmcid://{pmcid}"
        source_uri = normalize_uri(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}")
        doc_id = upsert_document(
            cur=cur,
            ext_id=ext_id,
            title=meta.get("title"),
            source_uri=source_uri,
            author=meta.get("authors"),
            year_date=year_date,              # <-- pass DATE here
            institute=meta.get("institutes"),
            source="PubMed Central",
            source_id=source_id,
        )
        upsert_chunks(cur=cur, doc_id=doc_id, chunks=chunks, embs=embs, pmcid=pmcid)
        con.commit()
        print(f"✅ Inserted {len(chunks)} chunks for {pmcid}", flush=True)

    print(f"✨ Done ingesting {pmcid}", flush=True)
    return len(chunks)

NXML_RE = re.compile(r'</?([^>]+)>')

async def fetch_pmc_nxml(pmcid: str) -> str | None:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=headers)
        print(f"[nxml] GET {r.url} -> {r.status_code} ct={r.headers.get('content-type')}", flush=True)
        if r.status_code != 200:
            return None
        return r.text

def extract_text_from_nxml(nxml: str) -> list[str]:
    if not nxml:
        return []
    try:
        soup = BeautifulSoup(nxml, "lxml-xml")  # needs lxml
    except Exception as e:
        print(f"[nxml] Parser error: {type(e).__name__}: {e}", flush=True)
        return []

    texts = []

    # Titles
    for t in soup.find_all(["article-title", "title"]):
        s = (t.get_text(" ", strip=True) or "").strip()
        if s: texts.append(s)

    # Paragraphs
    for p in soup.find_all("p"):
        s = (p.get_text(" ", strip=True) or "").strip()
        if s: texts.append(s)

    # Section titles (optional)
    for sec in soup.find_all("sec"):
        tt = sec.find("title")
        if tt:
            s = (tt.get_text(" ", strip=True) or "").strip()
            if s: texts.append(s)

    # simple de-dup
    out, seen = [], set()
    for s in texts:
        k = s[:80]
        if k in seen: 
            continue
        seen.add(k)
        out.append(s)
    return out
def extract_authors_from_nxml(nxml: str) -> Optional[str]:
    """
    Extract semicolon-separated list of authors from NXML content.
    """
    if not nxml:
        return None
    soup = BeautifulSoup(nxml, "lxml-xml")

    authors = []
    for contrib in soup.find_all("contrib", {"contrib-type": "author"}):
        name_tag = contrib.find("name")
        if name_tag:
            surname = name_tag.findtext("surname", "").strip()
            given = name_tag.findtext("given-names", "").strip()
            full = " ".join([given, surname]).strip()
            if full:
                authors.append(full)
        else:
            # fallback: sometimes directly under contrib
            text = contrib.get_text(" ", strip=True)
            if text:
                authors.append(text)

    # Deduplicate and join
    seen, out = set(), []
    for a in authors:
        if a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return "; ".join(out) if out else None

def extract_meta_from_nxml(nxml: str) -> Dict[str, Optional[str]]:
    """
    Parse PMC NXML to extract title, year (YYYY), and institutes (joined string).
    """
    soup = BeautifulSoup(nxml, "lxml-xml")

    # Title
    title = None
    tg = soup.find("title-group")
    if tg and tg.find("article-title"):
        title = tg.find("article-title").get_text(" ", strip=True)
    if not title:
        # Fallback: any <article-title>
        t = soup.find("article-title")
        if t:
            title = t.get_text(" ", strip=True)

    # Year (prefer print pub-date, then epub)
    year = _year_from_pub_dates(soup)

    # Authors
    authors = extract_authors_from_nxml(nxml)

    # Institutes (affiliations)
    insts: List[str] = []
    # Preferred: <aff><institution>...</institution>
    for aff in soup.find_all("aff"):
        # collect <institution> tags, else use the whole aff text
        inst_tags = aff.find_all("institution")
        if inst_tags:
            for it in inst_tags:
                s = it.get_text(" ", strip=True)
                if s: insts.append(s)
        else:
            s = aff.get_text(" ", strip=True)
            if s: insts.append(s)

    return {
        "title": (title or None),
        "year": (year or None),
        "institutes": _dedup_join(insts),
        "authors" : authors or None
    }

