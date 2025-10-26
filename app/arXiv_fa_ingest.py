# arxiv_ingest.py
import os, re, sys, argparse, asyncio, random, hashlib, json
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

import numpy as np
from packaging import version
import sentence_transformers

# ---------- Config / ENV ----------
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL    = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS   = int(os.getenv("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "80"))
MAX_CHARS      = int(os.getenv("MAX_CHARS", "20000"))
MAX_CHUNKS     = int(os.getenv("MAX_CHUNKS", "6"))
EMBED_BATCH    = int(os.getenv("EMBED_BATCH_SIZE", "8"))
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

FORCE_UPDATE        = os.getenv("FORCE_UPDATE", "0") == "1"
PURGE_OLD_CHUNKS    = os.getenv("PURGE_OLD_CHUNKS", "0") == "1"
LOG_SKIP_REASON     = os.getenv("LOG_SKIP_REASON", "0") == "1"

ARXIV_FETCH_PDF = os.getenv("ARXIV_FETCH_PDF", "0") == "1"   # optional: fetch PDF fulltext
ARXIV_DELAY     = float(os.getenv("ARXIV_DELAY", "0.35"))    # polite pacing (arXiv is stricter than NCBI)
ARXIV_API       = os.getenv("ARXIV_API", "https://export.arxiv.org/api/query")
EMAIL           = os.getenv("EMAIL", "you@example.com")
USER_AGENT      = f"med-rag/0.1 (contact:{EMAIL})"

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "3"))
PAGE_SIZE       = int(os.getenv("ARXIV_PAGE_SIZE", "50"))
RETMAX_DEFAULT  = int(os.getenv("RETMAX", "80"))  # topic cap

if CHUNK_OVERLAP >= CHUNK_TOKENS:
    raise ValueError("CHUNK_OVERLAP must be < CHUNK_TOKENS")

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

# ---------- Embedding ----------
_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        kw = {}
        try:
            if "use_auth_token" in SentenceTransformer.__init__.__code__.co_varnames:
                kw["use_auth_token"] = HUGGINGFACE_HUB_TOKEN
            else:
                kw["token"] = HUGGINGFACE_HUB_TOKEN
        except Exception:
            kw["use_auth_token"] = HUGGINGFACE_HUB_TOKEN
        _embedder = SentenceTransformer(EMBED_MODEL, **kw)
    return _embedder

def encode_embeddings(texts: list[str]):
    """
    Always return a 2D float array (n, d), normalized row-wise.
    Works across sentence-transformers versions.
    """
    model = embedder()
    kw = dict(batch_size=EMBED_BATCH)
    # Prefer returning numpy directly
    try:
        embs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # newer versions
            **kw
        )
    except TypeError:
        # Fallback for older versions w/o normalize_embeddings arg
        embs = model.encode(texts, convert_to_numpy=True, **kw)
        # Manual L2 normalization
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms

    # If something still returned a list / ragged, fix it
    if isinstance(embs, list):
        embs = np.asarray(embs)
    if embs.ndim == 1 and hasattr(embs[0], "__len__"):
        # object array -> stack
        embs = np.vstack(embs)
    assert embs.ndim == 2, f"Expected 2D embeddings, got shape={getattr(embs,'shape', None)}"
    return embs.astype(np.float32, copy=False)
    
def to_vec_lit(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

# ---------- Chunking ----------
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

# ---------- DB upserts (match your schema) ----------
def upsert_document(
    cur,
    *,
    ext_id: str,
    title: Optional[str],
    source_uri: Optional[str],
    author: Optional[str],
    year_date: Optional[date],
    institute: Optional[str],
    source: Optional[str],
    source_id: Optional[str],
) -> int:
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
    return cur.fetchone()[0]

def upsert_chunks(cur, doc_id: int, chunks: List[str], embs, source_id: str):
    rows = []
    for idx, (txt, vec) in enumerate(zip(chunks, embs)):
        chash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        preview = txt[:500]
        meta = {
            "source": "arXiv",
            "source_id": source_id,
            "chunk_index": idx,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBED_MODEL,
        }
        rows.append((doc_id, idx, json.dumps(meta, ensure_ascii=False), preview, chash, to_vec_lit(vec), EMBED_MODEL))
    cur.executemany(
        """
        INSERT INTO chunks (doc_id, chunk_index, metadata, text, content_hash, embedding, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
          metadata = EXCLUDED.metadata,
          text = EXCLUDED.text,
          content_hash = EXCLUDED.content_hash,
          embedding = EXCLUDED.embedding,
          embedding_model = EXCLUDED.embedding_model;
        """,
        rows,
    )

# ---------- Planner helpers (mirror PMC behavior) ----------
def needs_update(con, arxiv_id: str) -> bool:
    with con.cursor() as cur:
        # document present?
        cur.execute("""
            SELECT d.doc_id
            FROM documents d
            WHERE d.source = 'arXiv' AND d.source_id = %s
        """, (arxiv_id,))
        row = cur.fetchone()
        if not row:
            return True
        doc_id = row[0]
        if FORCE_UPDATE:
            return True
        # chunks for current model?
        cur.execute("""
            SELECT 1
            FROM chunks
            WHERE doc_id = %s AND embedding_model = %s
            LIMIT 1
        """, (doc_id, EMBED_MODEL))
        return cur.fetchone() is None

def log_skip_reason(con, arxiv_id: str):
    with con.cursor() as cur:
        cur.execute("""
            SELECT d.doc_id FROM documents d
            WHERE d.source='arXiv' AND d.source_id=%s
        """, (arxiv_id,))
        row = cur.fetchone()
        if not row:
            print(f"[plan] {arxiv_id}: skipped? (unexpected) doc not found but planner flagged as present")
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
            print(f"[plan] {arxiv_id}: doc exists but has 0 chunks")
        else:
            models = ", ".join(f"{m} x{n}" for m,n in rows)
            print(f"[plan] {arxiv_id}: chunks exist → {models}; current model={EMBED_MODEL}")

def plan_with_current_model(candidate_ids: List[str]) -> Tuple[List[str], List[str]]:
    todo, skipped = [], []
    with connect(**PG_KWARGS) as con:
        for aid in candidate_ids:
            if needs_update(con, aid):
                todo.append(aid)
            else:
                skipped.append(aid)
                if LOG_SKIP_REASON:
                    log_skip_reason(con, aid)
    return todo, skipped

# ---------- arXiv API helpers ----------
def _norm_arxiv_id(raw_id: str) -> str:
    rid = raw_id.strip()
    rid = re.sub(r"^arXiv:", "", rid, flags=re.I)
    rid = re.sub(r"v\d+$", "", rid)  # drop version
    return rid

async def arxiv_query(search_query: str, start: int = 0, max_results: int = 50,
                      sortBy="submittedDate", sortOrder="descending"):
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(ARXIV_API, params=params, headers=headers)
        if r.status_code != 200:
            print(f"[arXiv] HTTP {r.status_code} query={search_query!r}")
            return None
        return BeautifulSoup(r.text, "lxml-xml")

def _entry_to_record(entry) -> Optional[Dict]:
    try:
        arxiv_id = entry.id.get_text().split("/abs/")[-1]
    except Exception:
        return None
    arxiv_id = _norm_arxiv_id(arxiv_id)

    title = (entry.title.get_text(" ", strip=True) if entry.title else None)
    abstract = (entry.summary.get_text(" ", strip=True) if entry.summary else None)

    # authors
    authors = []
    for a in entry.find_all("author"):
        name = a.find("name")
        if name:
            s = name.get_text(" ", strip=True)
            if s: authors.append(s)
    author_str = "; ".join(dict.fromkeys(authors)) if authors else None

    # published date
    pubd = (entry.published.get_text(strip=True) if entry.published else None)
    ydate: Optional[date] = None
    if pubd:
        try:
            dt = datetime.fromisoformat(pubd.replace("Z", "+00:00"))
            ydate = dt.date()
        except Exception:
            m = re.search(r"(19|20)\d{2}", pubd)
            if m:
                ydate = date(int(m.group(0)), 1, 1)

    # links
    abs_url, pdf_url = None, None
    for link in entry.find_all("link"):
        rel = link.get("rel")
        href = link.get("href")
        if rel == "alternate":
            abs_url = href
        if rel == "related" and href and href.endswith(".pdf"):
            pdf_url = href
    if not abs_url:
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": author_str,
        "published": ydate,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
    }

# ---------- Optional: PDF text ----------
def _extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        from io import BytesIO
        r = PdfReader(BytesIO(data))
        parts = []
        for page in r.pages[:30]:
            try:
                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt.strip())
            except Exception:
                continue
        return "\n\n".join(parts)
    except Exception:
        return ""

async def fetch_pdf_text(pdf_url: str, arxiv_id: Optional[str] = None) -> str:
    """
    Try to fetch the HTML full text first (if available), otherwise fall back to PDF bytes.
    Returns plain text.
    """
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=45) as client:
        # --- 1. Try HTML version (only works for some newer arXiv papers)
        if arxiv_id:
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            try:
                r_html = await client.get(html_url)
                if r_html.status_code == 200 and len(r_html.text) > 1000:
                    soup = BeautifulSoup(r_html.text, "lxml")
                    fulltext = soup.get_text(" ", strip=True)
                    if len(fulltext.split()) > 500:  # sanity check
                        print(f"🔹 Using HTML fulltext for {arxiv_id} ({len(fulltext)} chars)", flush=True)
                        return fulltext
            except Exception as e:
                print(f"[warn] HTML fetch failed for {arxiv_id}: {e}", flush=True)

        # --- 2. Fallback: download PDF and extract text
        try:
            r_pdf = await client.get(pdf_url)
            if r_pdf.status_code != 200:
                print(f"[warn] PDF fetch failed {r_pdf.status_code} for {pdf_url}", flush=True)
                return ""
            txt = _extract_text_from_pdf_bytes(r_pdf.content)
            if txt.strip():
                print(f"🔹 Extracted PDF text for {arxiv_id or pdf_url} ({len(txt)} chars)", flush=True)
            else:
                print(f"[warn] Empty PDF text for {arxiv_id or pdf_url}", flush=True)
            return txt
        except Exception as e:
            print(f"[err] PDF fetch failed for {arxiv_id or pdf_url}: {e}", flush=True)
            return ""

# ---------- Public ingest: one arXiv id ----------
async def ingest_one_arxiv_id(arxiv_id: str) -> int:
    arxiv_id = _norm_arxiv_id(arxiv_id)
    print(f"🔹 Starting ingestion for arXiv:{arxiv_id}", flush=True)

    soup = await arxiv_query(f"id:{arxiv_id}", start=0, max_results=1)
    if soup is None:
        print(f"[arXiv] query failed for {arxiv_id}")
        return 0
    entry = soup.find("entry")
    if entry is None:
        print(f"[arXiv] no entry for {arxiv_id}")
        return 0

    rec = _entry_to_record(entry)
    if not rec:
        print(f"[arXiv] parse error for {arxiv_id}")
        return 0

    title = rec["title"]
    abstract = rec["abstract"] or ""
    body = (title or arxiv_id) + "\n\n" + (abstract or "")

    if ARXIV_FETCH_PDF and rec["pdf_url"]:
        pdf_text = await fetch_pdf_text(rec["pdf_url"], arxiv_id=rec["arxiv_id"])
        print("FETCHING PDF TEXT", flush=True)
        if pdf_text:
            body = (body + "\n\n" + pdf_text)[:MAX_CHARS]
        else:
            body = body[:MAX_CHARS]
    else:
        body = body[:MAX_CHARS]

    raw_chunks = chunk_by_tokens(body, CHUNK_TOKENS, CHUNK_OVERLAP)
    if MAX_CHUNKS:
        raw_chunks = raw_chunks[:MAX_CHUNKS]
    chunks = [f"{arxiv_id}\n\n{c}" for c in (raw_chunks or [body])]
    print(f"🔹 Created {len(chunks)} chunks (cap={MAX_CHUNKS})", flush=True)

    print(f"🔹 Encoding with {EMBED_MODEL} (batch={EMBED_BATCH})", flush=True)
    approx_tokens = len((body or "").split())
    print(f"🔹 Body chars={len(body):,}, approx tokens~{approx_tokens}, chunk_tokens={CHUNK_TOKENS}, overlap={CHUNK_OVERLAP}", flush=True)
    embs = encode_embeddings(chunks)
    print(f"🔹 Embedding shape: {embs.shape} (dim={embs.shape[1]})", flush=True)
    if embs.shape[0] != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: got {embs.shape[0]} for {len(chunks)} chunks")
        
    try:
        shape = getattr(embs, "shape", (len(embs), len(embs[0]) if embs else 0))
    except Exception:
        shape = (len(embs),)

    print(f"🔹 Connecting to DB: {PG_KWARGS['dbname']}", flush=True)
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        ext_id     = f"arxiv://{arxiv_id}"
        source_uri = rec["abs_url"]
        author     = rec["authors"]
        year_date  = rec["published"] or None
        institute  = None
        source     = "arXiv"
        source_id  = arxiv_id

        doc_id = upsert_document(
            cur=cur,
            ext_id=ext_id,
            title=title,
            source_uri=source_uri,
            author=author,
            year_date=year_date,
            institute=institute,
            source=source,
            source_id=source_id,
        )

        if PURGE_OLD_CHUNKS:
            # lock the row to stabilize concurrent writers
            cur.execute("SELECT 1 FROM documents WHERE doc_id = %s FOR UPDATE;", (doc_id,))
            cur.execute(
                "DELETE FROM chunks WHERE doc_id = %s AND embedding_model = %s;",
                (doc_id, EMBED_MODEL),
            )
            con.commit()
            print(f"✅ Purged old chunks for arXiv:{arxiv_id}", flush=True)

        upsert_chunks(cur, doc_id, chunks, embs, arxiv_id)
        con.commit()
        print(f"✅ Inserted {len(chunks)} chunks for arXiv:{arxiv_id}", flush=True)

    print(f"✨ Done ingesting arXiv:{arxiv_id}", flush=True)
    await asyncio.sleep(ARXIV_DELAY)
    return len(chunks)

# ---------- Topic search + ingest (with same planner logs) ----------
async def arxiv_search_ids(search_query: str, retmax: int = RETMAX_DEFAULT) -> List[str]:
    """
    Return up to retmax normalized arXiv IDs for a search query, paging via Atom API.
    """
    found: List[str] = []
    start = 0
    while start < retmax:
        batch = min(PAGE_SIZE, retmax - start)
        soup = await arxiv_query(search_query, start=start, max_results=batch)
        if soup is None:
            break
        entries = soup.find_all("entry")
        if not entries:
            break
        for e in entries:
            rec = _entry_to_record(e)
            if rec:
                aid = rec["arxiv_id"]
                if aid not in found:
                    found.append(aid)
        start += batch
        await asyncio.sleep(ARXIV_DELAY)
    print(f"[search] query='{search_query}' -> {len(found)} arXiv IDs", flush=True)
    return found

async def ingest_arxiv_topic(
    search_query: str,
    retmax: int = RETMAX_DEFAULT,
    max_concurrency: int = MAX_CONCURRENCY,
) -> Dict[str, int]:
    ids = await arxiv_search_ids(search_query, retmax=retmax)
    if not ids:
        return {"found": 0, "ingested": 0, "skipped": 0, "errors": 0}

    todo, skipped_list = plan_with_current_model(ids)
    print(f"[plan] {search_query}: {len(todo)} to ingest, {len(skipped_list)} skipped (already in DB)", flush=True)

    sem = asyncio.Semaphore(max_concurrency)
    async def _ingest_one(aid: str):
        async with sem:
            try:
                n = await ingest_one_arxiv_id(aid)
                print(f"[ok] arXiv:{aid}: {n} chunks", flush=True)
                return ("ok", n)
            except Exception as e:
                print(f"[err] arXiv:{aid}: {e}", flush=True)
                return ("err", 0)

    results = await asyncio.gather(*(_ingest_one(a) for a in todo))
    ok_count = sum(1 for s, _ in results if s == "ok")
    err_count = sum(1 for s, _ in results if s == "err")

    return {"found": len(ids), "ingested": ok_count, "skipped": len(skipped_list), "errors": err_count}

# ---------- CLI ----------
TOPIC_EXAMPLES = [
    'cat:cs.LG AND "auditory"',
    '"cochlear implant" AND deep learning',
    'cat:q-bio.NC AND hearing',
]

async def main():
    parser = argparse.ArgumentParser(description="arXiv ingest (per-ID or topic)")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--arxiv", nargs="+", help="Explicit arXiv IDs (e.g., 2401.01234 2305.00001v2)")
    mode.add_argument("--from-file", help="File with one arXiv ID per line")
    mode.add_argument("--topic", help="arXiv search_query (Atom API syntax)")
    parser.add_argument("--retmax", type=int, default=RETMAX_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)

    args = parser.parse_args()

    # Explicit IDs
    if args.arxiv or args.from_file:
        ids: List[str] = []
        if args.arxiv:
            ids.extend(args.arxiv)
        if args.from_file:
            with open(args.from_file) as fh:
                ids.extend([ln.strip() for ln in fh if ln.strip()])
        ids = [_norm_arxiv_id(i) for i in ids]
        print(f"[bulk] ingesting {len(ids)} arXiv IDs with concurrency={args.concurrency}")
        sem = asyncio.Semaphore(args.concurrency)

        async def worker(aid: str):
            async with sem:
                try:
                    n = await ingest_one_arxiv_id(aid)
                    return ("ok", n)
                except Exception as e:
                    print(f"[ingest] FAIL arXiv:{aid}: {type(e).__name__}: {e}", flush=True)
                    return ("err", 0)

        results = await asyncio.gather(*(worker(a) for a in ids))
        ok = sum(1 for s,_ in results if s=="ok")
        err = sum(1 for s,_ in results if s=="err")
        print(f"[summary] explicit IDs: found={len(ids)} ingested={ok} errors={err}")
        return

    # Topic mode
    query = args.topic or TOPIC_EXAMPLES[0]
    print(f"[topics] query={query!r} retmax={args.retmax} conc={args.concurrency}")
    summary = await ingest_arxiv_topic(query, retmax=args.retmax, max_concurrency=args.concurrency)
    print(f"[summary] {query}: {summary}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)