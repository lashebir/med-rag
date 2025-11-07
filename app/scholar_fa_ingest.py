#!/usr/bin/env python3
"""
Google Scholar ingestion via SerpAPI.
Fetches papers, downloads PDFs if available, chunks, embeds, and stores in DB.
"""
import os, re, sys, argparse, asyncio, hashlib, json
from typing import List, Dict, Optional, Tuple
from datetime import date

import httpx
from psycopg import connect
from sentence_transformers import SentenceTransformer
import numpy as np

# Section-aware chunking
from app.section_chunker import chunk_text_by_sections

# NER extraction
from app.ner_extractor import extract_entities_from_chunks, embed_mentions, deduplicate_mentions
from app.ner_db_utils import upsert_mentions

from dotenv import load_dotenv
load_dotenv()

# ---------- Config / ENV ----------
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
ENABLE_NER          = os.getenv("ENABLE_NER", "1") == "1"  # Enable NER extraction by default

SERPAPI_KEY     = os.getenv("SERPAPI", "")
SCHOLAR_DELAY   = float(os.getenv("SCHOLAR_DELAY", "1.0"))  # Be polite with SerpAPI
FETCH_PDF       = os.getenv("SCHOLAR_FETCH_PDF", "1") == "1"
HTTP_TIMEOUT    = int(os.getenv("HTTP_TIMEOUT", "60"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))

EMAIL           = os.getenv("EMAIL", "you@example.com")
USER_AGENT      = f"med-rag/0.1 (contact:{EMAIL})"

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "3"))
RETMAX_DEFAULT  = int(os.getenv("RETMAX", "80"))

if not SERPAPI_KEY:
    raise ValueError("SERPAPI key not found in .env file")

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
    """Always return a 2D float array (n, d), normalized row-wise."""
    model = embedder()
    kw = dict(batch_size=EMBED_BATCH)
    try:
        embs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            **kw
        )
    except TypeError:
        embs = model.encode(texts, convert_to_numpy=True, **kw)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms

    if isinstance(embs, list):
        embs = np.asarray(embs)
    if embs.ndim == 1 and hasattr(embs[0], "__len__"):
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

# ---------- DB upserts ----------
def upsert_document(
    cur,
    *,
    ext_id: str,
    title: Optional[str],
    source_uri: Optional[str],
    author: Optional[List[str]],
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

def upsert_chunks(cur, doc_id: int, chunks: List[Tuple[str, str]], embs, source_id: str):
    """
    Upsert chunks with section names.

    Args:
        chunks: List of (chunk_text, section_name) tuples
    """
    rows = []
    for idx, ((txt, section_name), vec) in enumerate(zip(chunks, embs)):
        chash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        preview = txt[:500]
        meta = {
            "source": "Google Scholar",
            "source_id": source_id,
            "chunk_index": idx,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBED_MODEL,
            "section_name": section_name,
        }
        rows.append((doc_id, idx, json.dumps(meta, ensure_ascii=False), preview, chash, to_vec_lit(vec), EMBED_MODEL, section_name))
    cur.executemany(
        """
        INSERT INTO chunks (doc_id, chunk_index, metadata, text, content_hash, embedding, embedding_model, section_name)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
        ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
          metadata = EXCLUDED.metadata,
          text = EXCLUDED.text,
          content_hash = EXCLUDED.content_hash,
          embedding = EXCLUDED.embedding,
          embedding_model = EXCLUDED.embedding_model,
          section_name = EXCLUDED.section_name;
        """,
        rows,
    )

# ---------- Planner helpers ----------
def needs_update(con, scholar_id: str) -> bool:
    with con.cursor() as cur:
        cur.execute("""
            SELECT d.doc_id
            FROM documents d
            WHERE d.source = 'Google Scholar' AND d.source_id = %s
        """, (scholar_id,))
        row = cur.fetchone()
        if not row:
            return True
        doc_id = row[0]
        if FORCE_UPDATE:
            return True
        cur.execute("""
            SELECT 1
            FROM chunks
            WHERE doc_id = %s AND embedding_model = %s
            LIMIT 1
        """, (doc_id, EMBED_MODEL))
        return cur.fetchone() is None

def log_skip_reason(con, scholar_id: str):
    with con.cursor() as cur:
        cur.execute("""
            SELECT d.doc_id FROM documents d
            WHERE d.source='Google Scholar' AND d.source_id=%s
        """, (scholar_id,))
        row = cur.fetchone()
        if not row:
            print(f"[plan] {scholar_id}: skipped? (unexpected) doc not found but planner flagged as present")
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
            print(f"[plan] {scholar_id}: doc exists but has 0 chunks")
        else:
            models = ", ".join(f"{m} x{n}" for m,n in rows)
            print(f"[plan] {scholar_id}: chunks exist → {models}; current model={EMBED_MODEL}")

def plan_with_current_model(candidate_ids: List[str]) -> Tuple[List[str], List[str]]:
    todo, skipped = [], []
    with connect(**PG_KWARGS) as con:
        for sid in candidate_ids:
            if needs_update(con, sid):
                todo.append(sid)
            else:
                skipped.append(sid)
                if LOG_SKIP_REASON:
                    log_skip_reason(con, sid)
    return todo, skipped

# ---------- PDF extraction ----------
def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    text_parts = []

    try:
        from PyPDF2 import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        for i, page in enumerate(reader.pages[:30]):  # Limit to first 30 pages
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text.strip())
            except Exception as e:
                print(f"[warn] PyPDF2 failed on page {i}: {e}", flush=True)
                continue

        if text_parts:
            return "\n\n".join(text_parts)
    except ImportError:
        print("[warn] PyPDF2 not installed", flush=True)
    except Exception as e:
        print(f"[warn] PyPDF2 extraction failed: {e}", flush=True)

    return ""

async def fetch_pdf_text(pdf_url: str, scholar_id: Optional[str] = None, search_query: Optional[str] = None) -> str:
    """Fetch and extract text from PDF."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf",
    }

    transport = httpx.AsyncHTTPTransport(retries=MAX_RETRIES)
    timeout = httpx.Timeout(HTTP_TIMEOUT, connect=10.0)

    async with httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=True) as client:
        print(f"[info] Fetching PDF from: {pdf_url}", flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = SCHOLAR_DELAY * (2 ** attempt)
                    print(f"[info] Retry {attempt + 1}/{MAX_RETRIES} after {wait_time:.1f}s delay...", flush=True)
                    await asyncio.sleep(wait_time)

                r_pdf = await client.get(pdf_url, headers=headers)

                if r_pdf.status_code == 200:
                    if not r_pdf.content.startswith(b'%PDF'):
                        print(f"[warn] Response doesn't look like a PDF", flush=True)
                        continue

                    txt = _extract_text_from_pdf_bytes(r_pdf.content)
                    if txt.strip():
                        word_count = len(txt.split())
                        print(f"✅ Extracted PDF text for {scholar_id or pdf_url} ({len(txt)} chars, ~{word_count} words)", flush=True)
                        return txt
                    else:
                        print(f"[warn] Empty PDF text extraction", flush=True)
                        return ""

                elif r_pdf.status_code == 403:
                    topic_info = f" | Topic: {search_query}" if search_query else ""
                    print(f"[error] Access denied (403) for {pdf_url}. May be rate limiting.{topic_info}", flush=True)
                    await asyncio.sleep(SCHOLAR_DELAY * 5)

                elif r_pdf.status_code == 404:
                    print(f"[error] PDF not found (404) for {pdf_url}", flush=True)
                    break

                else:
                    print(f"[warn] PDF fetch returned status {r_pdf.status_code}", flush=True)

            except httpx.TimeoutException as e:
                print(f"[warn] Timeout fetching PDF (attempt {attempt + 1}/{MAX_RETRIES}): {e}", flush=True)

            except Exception as e:
                print(f"[error] Unexpected error fetching PDF (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {e}", flush=True)

        # All retries exhausted - raise exception to stop ingestion
        topic_info = f" | Topic: {search_query}" if search_query else ""
        error_msg = f"Failed to fetch PDF after {MAX_RETRIES} attempts for {scholar_id or pdf_url}{topic_info}"
        print(f"[error] {error_msg}", flush=True)
        raise RuntimeError(error_msg)

# ---------- SerpAPI Google Scholar search ----------
async def scholar_search(query: str, start: int = 0, num: int = 10) -> Optional[Dict]:
    """Search Google Scholar via SerpAPI."""
    params = {
        "engine": "google_scholar",
        "q": query,
        "start": start,
        "num": num,
        "api_key": SERPAPI_KEY,
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        try:
            r = await client.get("https://serpapi.com/search", params=params)
            if r.status_code != 200:
                print(f"[scholar] HTTP {r.status_code} query={query!r}")
                return None
            return r.json()
        except Exception as e:
            print(f"[scholar] Error: {type(e).__name__}: {e}")
            return None

def _extract_scholar_id(result: Dict) -> str:
    """Extract a unique ID from Google Scholar result."""
    # Use the 'result_id' from SerpAPI, or construct from title
    result_id = result.get("result_id")
    if result_id:
        return result_id

    # Fallback: hash of title
    title = result.get("title", "")
    if title:
        return hashlib.md5(title.encode("utf-8")).hexdigest()[:16]

    return hashlib.md5(str(result).encode("utf-8")).hexdigest()[:16]

def _parse_year(pub_info: str) -> Optional[date]:
    """Extract year from publication info string."""
    if not pub_info:
        return None
    m = re.search(r"(19|20)\d{2}", pub_info)
    if m:
        year = int(m.group(0))
        return date(year, 1, 1)
    return None

# ---------- Ingest one Scholar result ----------
async def ingest_one_scholar_result(result: Dict, search_query: Optional[str] = None) -> int:
    """Ingest one Google Scholar search result."""
    scholar_id = _extract_scholar_id(result)

    title = result.get("title", "")
    snippet = result.get("snippet", "")
    link = result.get("link", "")
    pub_info = result.get("publication_info", {})

    # Extract metadata
    authors_list = pub_info.get("authors", [])
    author_names = [a.get("name", "") for a in authors_list if a.get("name")]
    authors = author_names if author_names else None

    pub_summary = pub_info.get("summary", "")
    year_date = _parse_year(pub_summary)

    # Get PDF link if available
    pdf_url = None
    resources = result.get("resources", [])
    for resource in resources:
        if resource.get("file_format", "").lower() == "pdf":
            pdf_url = resource.get("link")
            break

    topic_info = f" (query: {search_query})" if search_query else ""
    print(f"🔹 Starting ingestion for Scholar:{scholar_id}{topic_info}", flush=True)
    print(f"   Title: {title[:80]}...", flush=True)

    # Build body text
    body = (title or scholar_id) + "\n\n" + (snippet or "")

    # Try to fetch PDF if available
    if FETCH_PDF and pdf_url:
        print(f"[info] PDF URL found: {pdf_url}", flush=True)
        pdf_text = await fetch_pdf_text(pdf_url, scholar_id=scholar_id, search_query=search_query)
        if pdf_text:
            body = (body + "\n\n" + pdf_text)[:MAX_CHARS]
            print(f"[info] Combined text length: {len(body)} chars", flush=True)
        else:
            print(f"[warn] No PDF text extracted, using title+snippet only", flush=True)
            body = body[:MAX_CHARS]
    else:
        if not pdf_url:
            print(f"[info] No PDF link available", flush=True)
        body = body[:MAX_CHARS]

    # Create chunks using section-aware chunking
    print(f"🔹 Using section-aware chunking (preserves document structure)", flush=True)
    raw_chunks = chunk_text_by_sections(body, max_tokens=CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP, include_section_names=True)

    if MAX_CHUNKS and len(raw_chunks) > MAX_CHUNKS:
        print(f"[info] Limiting chunks from {len(raw_chunks)} to {MAX_CHUNKS}", flush=True)
        raw_chunks = raw_chunks[:MAX_CHUNKS]

    # Prepend scholar_id to each chunk (keep section names separate)
    if raw_chunks:
        chunks = [(f"Scholar:{scholar_id}\n\n{chunk_text}", section_name) for chunk_text, section_name in raw_chunks]
    else:
        # Fallback if no chunks created
        chunks = [(f"Scholar:{scholar_id}\n\n{body}", "Document")]

    print(f"🔹 Created {len(chunks)} section-aware chunks", flush=True)

    # Generate embeddings (extract chunk texts from tuples)
    print(f"🔹 Encoding with {EMBED_MODEL}", flush=True)
    chunk_texts = [chunk_text for chunk_text, _ in chunks]
    embs = encode_embeddings(chunk_texts)
    print(f"🔹 Embedding shape: {embs.shape}", flush=True)

    if embs.shape[0] != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: got {embs.shape[0]} for {len(chunks)} chunks")

    # Store in database
    print(f"🔹 Connecting to DB: {PG_KWARGS['dbname']}", flush=True)
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        ext_id = f"scholar://{scholar_id}"
        source_uri = link or None

        doc_id = upsert_document(
            cur=cur,
            ext_id=ext_id,
            title=title,
            source_uri=source_uri,
            author=authors,
            year_date=year_date,
            institute=None,
            source="Google Scholar",
            source_id=scholar_id,
        )

        if PURGE_OLD_CHUNKS:
            cur.execute("SELECT 1 FROM documents WHERE doc_id = %s FOR UPDATE;", (doc_id,))
            cur.execute(
                "DELETE FROM chunks WHERE doc_id = %s AND embedding_model = %s;",
                (doc_id, EMBED_MODEL),
            )
            con.commit()
            print(f"✅ Purged old chunks for Scholar:{scholar_id}", flush=True)

        upsert_chunks(cur, doc_id, chunks, embs, scholar_id)
        con.commit()
        print(f"✅ Inserted {len(chunks)} chunks for Scholar:{scholar_id}", flush=True)

        # NER extraction (optional)
        if ENABLE_NER:
            print(f"🔹 Extracting named entities from chunks...", flush=True)
            try:
                # Extract mentions from chunks
                mentions = extract_entities_from_chunks(chunks)
                mentions = deduplicate_mentions(mentions)
                print(f"🔹 Found {len(mentions)} unique mentions", flush=True)

                if mentions:
                    # Generate embeddings for mentions
                    mention_embeddings = embed_mentions(mentions)
                    print(f"🔹 Generated embeddings for {len(mention_embeddings)} mentions", flush=True)

                    # Upsert mentions and embeddings
                    num_inserted = upsert_mentions(cur, doc_id, mentions, mention_embeddings)
                    con.commit()
                    print(f"✅ Inserted {num_inserted} mentions for Scholar:{scholar_id}", flush=True)

                    # Print stats by label
                    label_counts = {}
                    for m in mentions:
                        label = m['label']
                        label_counts[label] = label_counts.get(label, 0) + 1
                    print(f"   Mentions by type: {label_counts}", flush=True)
                else:
                    print(f"   No mentions extracted", flush=True)
            except Exception as e:
                print(f"⚠️ NER extraction failed: {e}", flush=True)
                # Continue without NER - don't fail the ingestion

    print(f"✨ Done ingesting Scholar:{scholar_id}", flush=True)
    await asyncio.sleep(SCHOLAR_DELAY)
    return len(chunks)

# ---------- Topic search + ingest ----------
async def scholar_search_all(search_query: str, retmax: int = RETMAX_DEFAULT) -> List[Dict]:
    """Get all results up to retmax from Google Scholar."""
    all_results = []
    start = 0
    batch_size = 10  # SerpAPI typically returns 10 results per page

    while start < retmax:
        print(f"[search] Fetching results {start} to {start + batch_size}...", flush=True)
        data = await scholar_search(search_query, start=start, num=batch_size)

        if not data:
            break

        results = data.get("organic_results", [])
        if not results:
            break

        all_results.extend(results)
        start += len(results)

        # Check if we've reached the end
        if len(results) < batch_size:
            break

        await asyncio.sleep(SCHOLAR_DELAY)

    print(f"[search] query='{search_query}' -> {len(all_results)} Scholar results", flush=True)
    return all_results[:retmax]

async def ingest_scholar_topic(
    search_query: str,
    retmax: int = RETMAX_DEFAULT,
    max_concurrency: int = MAX_CONCURRENCY,
) -> Dict[str, int]:
    """Search Google Scholar and ingest results."""
    results = await scholar_search_all(search_query, retmax=retmax)
    if not results:
        return {"found": 0, "ingested": 0, "skipped": 0, "errors": 0}

    # Extract IDs and plan
    scholar_ids = [_extract_scholar_id(r) for r in results]
    todo_ids, skipped_ids = plan_with_current_model(scholar_ids)

    # Create mapping of IDs to results
    id_to_result = {_extract_scholar_id(r): r for r in results}

    print(f"[plan] {search_query}: {len(todo_ids)} to ingest, {len(skipped_ids)} skipped", flush=True)

    sem = asyncio.Semaphore(max_concurrency)
    async def _ingest_one(sid: str):
        async with sem:
            try:
                result = id_to_result[sid]
                n = await ingest_one_scholar_result(result, search_query=search_query)
                print(f"[ok] Scholar:{sid}: {n} chunks", flush=True)
                return ("ok", n)
            except Exception as e:
                print(f"[err] Scholar:{sid}: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return ("err", 0)

    results = await asyncio.gather(*(_ingest_one(sid) for sid in todo_ids))
    ok_count = sum(1 for s, _ in results if s == "ok")
    err_count = sum(1 for s, _ in results if s == "err")

    return {"found": len(scholar_ids), "ingested": ok_count, "skipped": len(skipped_ids), "errors": err_count}

# ---------- CLI ----------
async def main():
    parser = argparse.ArgumentParser(description="Google Scholar ingest via SerpAPI")
    parser.add_argument("--query", required=True, help="Search query for Google Scholar")
    parser.add_argument("--retmax", type=int, default=RETMAX_DEFAULT, help="Max results to ingest")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY, help="Concurrent ingestions")

    args = parser.parse_args()

    print(f"[config] FETCH_PDF={FETCH_PDF}")
    print(f"[config] HTTP_TIMEOUT={HTTP_TIMEOUT}s")
    print(f"[config] MAX_RETRIES={MAX_RETRIES}")
    print(f"[config] SCHOLAR_DELAY={SCHOLAR_DELAY}s")
    print(f"[config] MAX_CHARS={MAX_CHARS}")
    print(f"[config] MAX_CHUNKS={MAX_CHUNKS}")
    print()

    summary = await ingest_scholar_topic(args.query, retmax=args.retmax, max_concurrency=args.concurrency)
    print(f"[summary] {args.query}: {summary}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
