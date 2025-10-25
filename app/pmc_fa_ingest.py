import os, hashlib, json, asyncio, random
from typing import List, Dict, Optional, Tuple
import httpx
from sentence_transformers import SentenceTransformer
from psycopg import connect
from psycopg.rows import dict_row
from dotenv import load_dotenv
import re
from bs4  import BeautifulSoup
# import lxml
from datetime import date
from packaging import version

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "1200")) # 10/13 note: upped from 600 to 1200 to have fewer chunks, test up to 1800
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))
MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "8"))
MAX_CHARS    = int(os.getenv("MAX_CHARS", "20000"))
MAX_CHUNKS   = int(os.getenv("MAX_CHUNKS", "4"))
EMBED_BATCH  = int(os.getenv("EMBED_BATCH_SIZE", "8"))
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
EMAIL = os.getenv("EMAIL", "you@example.com")
FORCE_UPDATE = os.getenv("FORCE_UPDATE", "0") == "1"

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "pubmed"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

if CHUNK_OVERLAP >= CHUNK_TOKENS:
    raise ValueError("CHUNK_OVERLAP must be < CHUNK_TOKENS")

_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        kw = {}
        try:
            # prefer the modern kwarg if available
            if "use_auth_token" in SentenceTransformer.__init__.__code__.co_varnames:
                kw["use_auth_token"] = HUGGINGFACE_HUB_TOKEN
            else:
                kw["token"] = HUGGINGFACE_HUB_TOKEN
        except Exception:
            kw["use_auth_token"] = HUGGINGFACE_HUB_TOKEN
        _embedder = SentenceTransformer(EMBED_MODEL, **kw)
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

def needs_update(con, pmcid: str) -> bool:
    with con.cursor() as cur:
        # 1) document present?
        cur.execute("""
            SELECT d.doc_id
            FROM documents d
            WHERE d.source = 'PubMed Central' AND d.source_id = %s
            """, (pmcid,))
        row = cur.fetchone()
        if not row:
            return True  # not in DB → ingest

        doc_id = row[0]

        # 2) do we have chunks for this model?
        cur.execute("""
            SELECT 1
            FROM chunks
            WHERE doc_id = %s AND embedding_model = %s
            LIMIT 1
            """, (doc_id, EMBED_MODEL))
        has_model_chunks = cur.fetchone() is not None

        if FORCE_UPDATE:
            return True
        return not has_model_chunks

def log_skip_reason(con, pmcid: str):
    with con.cursor() as cur:
        cur.execute("""
          SELECT d.doc_id FROM documents d
          WHERE d.source='PubMed Central' AND d.source_id=%s
        """, (pmcid,))
        row = cur.fetchone()
        if not row:
            print(f"[plan] {pmcid}: skipped? (unexpected) doc not found but planner flagged as present")
            return
        doc_id = row[0]
        cur.execute("""
          SELECT embedding_model, COUNT(*) 
          FROM chunks
          WHERE doc_id=%s
          GROUP BY embedding_model
        """, (doc_id,))
        rows = cur.fetchall()
        if not rows:
            print(f"[plan] {pmcid}: doc exists but has 0 chunks")
        else:
            models = ", ".join(f"{m} x{n}" for m,n in rows)
            print(f"[plan] {pmcid}: chunks exist → {models}; current model={EMBED_MODEL}")

# ---------- Format-aware splitting helpers ----------

_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9(])')  # crude, fast sentence splitter

def _sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # keep linebreaks as spaces to avoid creating micro-sentences
    text = re.sub(r'\s*\n+\s*', ' ', text)
    # split on punctuation + space + capital/digit/"("
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text]

def _tok_count(s: str) -> int:
    # same rough tokenization you already use
    return len(s.split())

def _trim_blocks_to_max_chars(blocks: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    total = 0
    out = []
    for kind, text in blocks:
        if total >= max_chars:
            break
        remain = max_chars - total
        if remain <= 0:
            break
        take = text[:remain]
        if take:
            out.append((kind, take))
            total += len(take)
        if len(take) < len(text):
            break
    return out

def gather_blocks_from_bioc(bioc: dict, max_passages: int | None = None) -> list[tuple[str, str]]:
    """
    Returns a list of (kind, text) blocks.
    kind ∈ {'title','heading','para','bullet','other'}
    """
    blocks: list[tuple[str, str]] = []
    docs = extract_bioc_documents(bioc)
    if max_passages is None:
        max_passages = 10**9
    count = 0

    for d in docs:
        for p in (d.get("passages") or []):
            t = (p.get("text") or "").strip()
            if not t:
                continue
            inf = p.get("infons") or {}
            ptype = (inf.get("type") or inf.get("section_type") or "").lower()

            if ptype in {"title", "article-title"}:
                kind = "title"
            elif ptype in {"section", "sec", "heading"}:
                kind = "heading"
            elif ptype in {"affiliation", "author-affiliation"}:
                # usually metadata; skip from content chunks
                continue
            else:
                # treat everything else textual as a paragraph
                kind = "para"

            blocks.append((kind, t))
            count += 1
            if count >= max_passages:
                break
        if count >= max_passages:
            break
    return blocks

def gather_blocks_from_nxml(nxml: str) -> list[tuple[str, str]]:
    """
    Produce (kind, text) blocks from PMC NXML:
      - article-title -> 'title'
      - sec/title     -> 'heading'
      - p              -> 'para'
      - list/list-item -> 'bullet'
    """
    blocks: list[tuple[str, str]] = []
    soup = BeautifulSoup(nxml, "lxml-xml")

    # Article title
    at = soup.find("article-title")
    if at:
        txt = at.get_text(" ", strip=True)
        if txt:
            blocks.append(("title", txt))

    # Walk sections in order; add headings, then their paragraphs & lists
    for sec in soup.find_all("sec"):
        st = sec.find("title")
        if st:
            s = st.get_text(" ", strip=True)
            if s:
                blocks.append(("heading", s))

        # paragraphs directly under sec (avoid re-adding nested sec paragraphs twice)
        for p in sec.find_all("p", recursive=False):
            s = p.get_text(" ", strip=True)
            if s:
                blocks.append(("para", s))

        # lists
        for li in sec.find_all("list"):
            for item in li.find_all("list-item", recursive=False):
                s = item.get_text(" ", strip=True)
                if s:
                    blocks.append(("bullet", s))

    # Fallback: top-level paragraphs not inside <sec>
    for p in soup.find_all("p"):
        parent_names = {a.name for a in p.parents if getattr(a, "name", None)}
        # if already inside a <sec>, it was handled above; keep only non-sec leftovers
        if "sec" not in parent_names:
            s = p.get_text(" ", strip=True)
            if s:
                blocks.append(("para", s))

    return blocks

def chunk_blocks(blocks: list[tuple[str, str]], max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Build chunks by concatenating whole blocks (title, heading, para, bullet).
    - We NEVER split across blocks unless a single block > max_tokens.
    - If a block > max_tokens: split that block by sentence to fit.
    - Overlap is applied in SENTENCES from the *previous* chunk tail.
    """
    chunks: list[str] = []
    cur_sents: list[str] = []      # sentence buffer for current chunk
    cur_tokens = 0
    overlap_sents_last_chunk: list[str] = []  # the tail-sentences we’ll carry into next chunk

    def flush_chunk():
        nonlocal cur_sents, cur_tokens, overlap_sents_last_chunk
        if not cur_sents:
            return
        text = "\n".join(cur_sents).strip()
        if text:
            chunks.append(text)
        # compute overlap seed as sentence tail >= overlap_tokens
        toks = 0
        tail = []
        for s in reversed(cur_sents):
            toks += _tok_count(s)
            tail.append(s)
            if toks >= overlap_tokens:
                break
        overlap_sents_last_chunk = list(reversed(tail))
        # reset buffer
        cur_sents = []
        cur_tokens = 0

    for kind, text in blocks:
        sents = _sentences(text)
        # naive estimate for the whole block
        block_tokens = sum(_tok_count(s) for s in sents)

        # If the whole block fits, try to add as a unit
        if block_tokens <= max_tokens:
            if cur_tokens + block_tokens <= max_tokens:
                # append block
                if kind in {"title", "heading"} and cur_sents:
                    # add a visual separator before a new heading inside a chunk
                    cur_sents.append("")  # blank line
                cur_sents.extend(sents)
                cur_tokens += block_tokens
            else:
                # flush current chunk and start a new chunk with overlap + this block
                flush_chunk()
                # start with overlap sentences (from previous chunk) if any
                if overlap_sents_last_chunk:
                    # Only add if they won't dominate the budget
                    ov_tokens = sum(_tok_count(s) for s in overlap_sents_last_chunk)
                    if ov_tokens < max_tokens // 2:
                        cur_sents.extend(overlap_sents_last_chunk)
                        cur_tokens += ov_tokens
                # now add the block (it fits by definition)
                cur_sents.extend(sents)
                cur_tokens += block_tokens

        else:
            # Block is larger than budget: split by sentences
            # First, finish current chunk (so a giant block starts clean)
            flush_chunk()

            buf: list[str] = []
            btoks = 0
            for s in sents:
                stoks = _tok_count(s)
                if btoks + stoks <= max_tokens:
                    buf.append(s)
                    btoks += stoks
                else:
                    # flush sub-chunk
                    if buf:
                        chunks.append("\n".join(buf).strip())
                    # seed overlap for sub-chunk series (tail of buf)
                    toks = 0
                    tail = []
                    for ts in reversed(buf):
                        toks += _tok_count(ts)
                        tail.append(ts)
                        if toks >= overlap_tokens:
                            break
                    # next sub-chunk starts with tail + current sentence
                    buf = list(reversed(tail))
                    btoks = sum(_tok_count(ts) for ts in buf)
                    buf.append(s)
                    btoks += stoks
            if buf:
                chunks.append("\n".join(buf).strip())
            # after a giant block, clear overlap (avoid re-adding too much)
            overlap_sents_last_chunk = []

    # flush remainder
    flush_chunk()
    return [c for c in chunks if c.strip()]

USE_BIOC = os.getenv("USE_BIOC", "1") == "1"
BIOC_MAX_RETRIES = int(os.getenv("BIOC_MAX_RETRIES", "2"))

async def fetch_bioc_json(pmcid: str) -> Dict | None:
    if not USE_BIOC:
        return None

    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    headers = {
    "User-Agent": f"med-rag/0.1 (contact: {EMAIL})",
    "Accept": "application/json",
    }

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(30.0)

    for attempt in range(BIOC_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                r = await client.get(url, headers=headers)
                ct = r.headers.get("content-type", "")
                if r.status_code == 404:
                    return None
                if 500 <= r.status_code < 600:
                    # 5xx -> log and retry (will fall back to NXML if still bad)
                    print(f"[BioC] {pmcid} {r.status_code} (attempt {attempt+1}/{BIOC_MAX_RETRIES})", flush=True)
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()
                if "json" not in ct.lower():
                    return None
                try:
                    return r.json()
                except Exception as e:
                    print(f"[BioC] JSON parse failed {pmcid}: {e}", flush=True)
                    return None
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError):
            delay = min(1.5 * (2 ** attempt), 8.0) + random.random()
            await asyncio.sleep(delay)
            continue
        except Exception as e:
            print(f"[BioC] fatal {pmcid}: {type(e).__name__}: {e}", flush=True)
            break
    return None  # triggers NXML fallback

def passages_from_bioc(doc_json: Dict) -> List[str]:
    texts: List[str] = []
    for d in extract_bioc_documents(doc_json):
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
    year_date: Optional[date],     # 'YYYY-01-01' or None (or pass a datetime.date)
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
        meta_chunks = {
            "pmcid": pmcid,
            "chunk_index": idx,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBED_MODEL
        }
        rows.append((
            doc_id, idx, preview, json.dumps(meta_chunks), chash, to_vec_lit(vec), EMBED_MODEL
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
    title = None
    year = None
    institutes: List[str] = []

    docs = []
    if isinstance(bioc, dict):
        if isinstance(bioc.get("documents"), list):
            docs = bioc["documents"]
        elif isinstance(bioc.get("collection"), dict) and isinstance(bioc["collection"].get("documents"), list):
            docs = bioc["collection"]["documents"]
    elif isinstance(bioc, list):
        docs = bioc

    for d in docs:
        for p in d.get("passages", []):
            inf = p.get("infons") or {}
            t = (p.get("text") or "").strip()
            itype = (inf.get("type") or "").lower()

            if not title and (itype in {"title", "article-title"} or inf.get("section_type") == "TITLE"):
                if t: title = t

            if not year:
                cand = inf.get("year") or inf.get("date") or ""
                y = _pick_year(cand) or _pick_year(t)
                if y: year = y

            if itype in {"affiliation", "aff", "author-affiliation"} and t:
                institutes.append(t)

    # NEW: pull authors from BioC if present
    authors = extract_authors_from_bioc(bioc)

    return {
        "title": title or None,
        "year": year or None,
        "institutes": _dedup_join(institutes),
        "authors": authors or None,
    }

async def ingest_one_pmcid(pmcid: str) -> int:
    print(f"🔹 Starting ingestion for {pmcid}", flush=True)

    bioc = await fetch_bioc_json(pmcid)  # dict or None
    meta_docs = {"title": None, "year": None, "institutes": None, "authors": None}

    # 1) Try BioC first (metadata + blocks)
    blocks: list[tuple[str, str]] = []
    if bioc:
        try:
            m_bioc = extract_meta_from_bioc(bioc)
            for k in ("title", "year", "institutes", "authors"):
                if m_bioc.get(k):
                    meta_docs[k] = m_bioc[k]
        except Exception as e:
            print(f"[BioC] meta parse warning: {e}", flush=True)

        blocks = gather_blocks_from_bioc(bioc, max_passages=MAX_PASSAGES)

    # 2) Fallback to NXML if no blocks from BioC
    if not blocks:
        print("⚠️ No usable blocks from BioC — trying NXML fallback", flush=True)
        nxml = await fetch_pmc_nxml(pmcid)
        if not nxml:
            print("❌ NXML fetch failed", flush=True)
            return 0

        try:
            m_nxml = extract_meta_from_nxml(nxml)
            for k in ("title", "year", "institutes", "authors"):
                if not meta_docs.get(k) and m_nxml.get(k):
                    meta_docs[k] = m_nxml[k]
        except Exception as e:
            print(f"[NXML] meta parse warning: {e}", flush=True)

        blocks = gather_blocks_from_nxml(nxml)

    # 3) Cap by MAX_CHARS for either path
    blocks = _trim_blocks_to_max_chars(blocks, MAX_CHARS)
    if not blocks:
        print("❌ Parsing produced no text blocks", flush=True)
        return 0

    print(f"🔹 Collected {len(blocks)} structural blocks (≤ {MAX_CHARS} chars)", flush=True)

    # --- Prepare text ---
    raw_chunks = chunk_blocks(blocks, CHUNK_TOKENS, CHUNK_OVERLAP)
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

    year_date = date(int(meta_docs["year"]), 1, 1) if meta_docs.get("year") else None

    print(f"🔹 Connecting to DB: {PG_KWARGS['dbname']}", flush=True)
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        
        ext_id = f"pmcid://{pmcid}"
        source_uri = normalize_uri(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}")
        doc_id = upsert_document(
            cur=cur,
            ext_id=ext_id,
            title=meta_docs.get("title"),
            source_uri=source_uri,
            author=meta_docs.get("authors"),
            year_date=year_date,              
            institute=meta_docs.get("institutes"),
            source="PubMed Central",
            source_id=source_id,
        )
        
        if os.getenv("PURGE_OLD_CHUNKS", "0") == "1":
            cur.execute("SELECT 1 FROM documents WHERE doc_id = %s FOR UPDATE;", (doc_id,))
            cur.execute(
                "DELETE FROM chunks WHERE doc_id = %s AND embedding_model = %s;",
                (doc_id, EMBED_MODEL),
            )
            con.commit()
            print(f"✅ Purged old chunks for {pmcid}", flush=True)
        
        upsert_chunks(cur=cur, doc_id=doc_id, chunks=chunks, embs=embs, pmcid=pmcid)
        con.commit()
        print(f"✅ Inserted {len(chunks)} chunks for {pmcid}", flush=True)

    print(f"✨ Done ingesting {pmcid}", flush=True)
    return len(chunks)

async def fetch_pmc_nxml(pmcid: str) -> str | None:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    headers = {"User-Agent": f"med-rag/0.1 (contact: {EMAIL})"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=headers)
        if r.status_code != 200 or "xml" not in (r.headers.get("content-type","")).lower():
            return None
        return r.text

def _norm_author_string(s: str) -> list[str]:
    # split on ; or , if records come as a single string
    parts = [p.strip() for p in re.split(r"[;,\n]+", s) if p.strip()]
    return parts

def extract_authors_from_bioc(bioc: dict) -> Optional[str]:
    """
    Try to extract authors from BioC JSON.
    Returns 'Given Surname; Given Surname; ...' or None.
    BioC may store authors under document-level 'infons' (e.g., 'authors'),
    or as passage infons (keys like 'author', 'authors', 'name', etc.).
    """
    authors: list[str] = []

    def _maybe_collect(val):
        if not val:
            return
        if isinstance(val, list):
            for x in val:
                if isinstance(x, str) and x.strip():
                    authors.append(x.strip())
        elif isinstance(val, str):
            authors.extend(_norm_author_string(val))

    docs = []
    if isinstance(bioc, dict):
        if isinstance(bioc.get("documents"), list):
            docs = bioc["documents"]
        elif isinstance(bioc.get("collection"), dict) and isinstance(bioc["collection"].get("documents"), list):
            docs = bioc["collection"]["documents"]
    elif isinstance(bioc, list):
        docs = bioc

    for d in docs:
        # doc-level infons (sometimes 'authors' exists)
        dinf = d.get("infons") or {}
        for k in ("authors", "author", "creator", "contrib"):
            _maybe_collect(dinf.get(k))

        # passage-level infons
        for p in d.get("passages", []):
            inf = p.get("infons") or {}
            # common-ish keys we’ve seen in the wild
            for k in ("authors", "author", "name", "contrib", "contributor"):
                _maybe_collect(inf.get(k))

            # very rare: structured pieces
            given = inf.get("given-names") or inf.get("firstname")
            fam   = inf.get("surname") or inf.get("lastname")
            if given or fam:
                full = " ".join([str(given or "").strip(), str(fam or "").strip()]).strip()
                if full:
                    authors.append(full)

    # dedup case-insensitively
    seen, out = set(), []
    for a in authors:
        k = a.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(a)

    return "; ".join(out) if out else None

def extract_authors_from_nxml(nxml: str) -> Optional[str]:
    """Return 'Given Surname; Given Surname; ...' or None."""
    if not nxml:
        return None
    soup = BeautifulSoup(nxml, "lxml-xml")

    authors: list[str] = []
    for contrib in soup.find_all("contrib", {"contrib-type": "author"}):
        name_tag = contrib.find("name")
        if name_tag:
            surname_tag = name_tag.find("surname")
            given_tag = name_tag.find("given-names")
            surname = surname_tag.get_text(" ", strip=True) if surname_tag else ""
            given = given_tag.get_text(" ", strip=True) if given_tag else ""
            full = " ".join([given, surname]).strip()
            if full:
                authors.append(full)
        else:
            text = contrib.get_text(" ", strip=True)
            if text:
                authors.append(text)

    seen, out = set(), []
    for a in authors:
        k = a.lower()
        if k and k not in seen:
            seen.add(k)
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
        "title": title or None,
        "year": year or None,
        "institutes": _dedup_join(insts),
        "authors" : authors or None,
    }

