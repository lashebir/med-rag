import os, hashlib, json
from typing import List, Dict
import httpx
from sentence_transformers import SentenceTransformer
from psycopg import connect
from psycopg.rows import dict_row
from dotenv import load_dotenv
import re
from bs4  import BeautifulSoup
import lxml

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "1200")) # 10/13 note: upped from 600 to 1200 to have fewer chunks, test up to 1800
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))
MAX_PASSAGES = int(os.getenv("MAX_PASSAGES", "8"))
MAX_CHARS    = int(os.getenv("MAX_CHARS", "20000"))
MAX_CHUNKS   = int(os.getenv("MAX_CHUNKS", "4"))
EMBED_BATCH  = int(os.getenv("EMBED_BATCH_SIZE", "8"))
HF_TOKEN = os.getenv("HF_TOKEN")

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
        _embedder = SentenceTransformer(EMBED_MODEL)
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

async def fetch_bioc_json(pmcid: str) -> Dict:
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

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

def ensure_document(cur, ext_id: str, title: str, source_uri: str) -> int:
    cur.execute("""
        INSERT INTO documents (ext_id, title, source_uri)
        VALUES (%s, %s, %s)
        ON CONFLICT (ext_id) DO UPDATE
        SET title = COALESCE(EXCLUDED.title, documents.title),
            source_uri = COALESCE(EXCLUDED.source_uri, documents.source_uri)
        RETURNING doc_id;
    """, (ext_id, title, source_uri))
    return cur.fetchone()[0]

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

async def ingest_one_pmcid(pmcid: str) -> int:
    # --- helper kept INSIDE the function or move it top-level if you prefer ---
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

    print(f"🔹 Starting ingestion for {pmcid}", flush=True)

    # 1) Fetch BioC JSON (with User-Agent header per NCBI guidelines)
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)",  "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
    try:
        bioc = r.json()
    except Exception as e:
        print(f"❌ Non-JSON from PMC: {type(e).__name__}: {e}", flush=True)
        raise

    # 2) Robustly extract docs and collect passage texts
    docs = extract_bioc_documents(bioc)
    print(f"🔹 BioC parsed: type={type(bioc).__name__}, docs={len(docs)}", flush=True)

    passages_texts = []
    for d in docs[:MAX_PASSAGES]:  # cap number of docs/passages we consider
        ps = d.get("passages") or []
        for p in ps:
            t = (p.get("text") or "").strip()
            if t:
                passages_texts.append(t)

    if not passages_texts:
        print(f"⚠️ No passages via BioC — trying NXML fallback", flush=True)
        nxml = await fetch_pmc_nxml(pmcid)
        if not nxml:
            print(f"❌ NXML fetch failed for {pmcid}", flush=True)
            return 0
        passages_texts = extract_text_from_nxml(nxml)
        print(f"[nxml] extracted {len(passages_texts)} text blocks", flush=True)
        if not passages_texts:
            print(f"❌ NXML parse produced no text for {pmcid}", flush=True)
            return 0

    full_text = "\n\n".join(passages_texts)
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS]
    print(f"🔹 Collected {len(passages_texts)} passages ({len(full_text)} chars)", flush=True)

    # 3) Chunk (keep small caps while debugging)
    raw_chunks = chunk_by_tokens(full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
    if MAX_CHUNKS:
        raw_chunks = raw_chunks[:MAX_CHUNKS]
    title = pmcid
    chunks = [f"{title}\n\n{c}" for c in raw_chunks]
    print(f"🔹 Created {len(chunks)} chunks (cap={MAX_CHUNKS})", flush=True)
    if not chunks:
        print("⚠️ No chunks after splitting", flush=True)
        return 0

    # 4) Embed (small batch)
    print(f"🔹 Encoding with {EMBED_MODEL} (batch={EMBED_BATCH})", flush=True)
    embs = embedder().encode(chunks, normalize_embeddings=True, batch_size=EMBED_BATCH)
    print(f"🔹 Embedding shape: {embs.shape}", flush=True)

    # 5) Insert into DB
    print(f"🔹 Connecting to DB: {PG_KWARGS['dbname']}", flush=True)
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        ext_id = f"pmcid://{pmcid}"
        source_uri = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}"
        doc_id = ensure_document(cur, ext_id, title, source_uri)
        upsert_chunks(cur, doc_id, chunks, embs, pmcid)
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

