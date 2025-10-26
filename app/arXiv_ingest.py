import os, re, asyncio, random, hashlib
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup
from psycopg import connect
from sentence_transformers import SentenceTransformer

# ---------- Config / ENV ----------
EMBED_MODEL   = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS  = int(os.getenv("CHUNK_TOKENS", "800"))     # arXiv abstracts are short; if PDF enabled we’ll add longer text
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MAX_CHARS     = int(os.getenv("MAX_CHARS", "20000"))
EMBED_BATCH   = int(os.getenv("EMBED_BATCH_SIZE", "8"))

ARXIV_FETCH_PDF = os.getenv("ARXIV_FETCH_PDF", "0") == "1"   # optional: fetch PDF fulltext
ARXIV_DELAY   = float(os.getenv("ARXIV_DELAY", "0.25"))      # polite pacing
ARXIV_API = os.getenv("ARXIV_API", "https://export.arxiv.org/api/query")

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),  # you can keep same DB; source="arXiv"
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

# ---------- Embedding ----------
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
        rows.append((doc_id, idx, preview, json_dumps(meta), chash, to_vec_lit(vec), EMBED_MODEL))
    cur.executemany(
        """
        INSERT INTO chunks (doc_id, chunk_index, text, metadata, content_hash, embedding, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
          text = EXCLUDED.text,
          metadata = EXCLUDED.metadata,
          content_hash = EXCLUDED.content_hash,
          embedding = EXCLUDED.embedding,
          embedding_model = EXCLUDED.embedding_model;
        """,
        rows,
    )

# tiny safe JSON helper (avoid circular imports)
import json as _json
def json_dumps(obj) -> str:
    return _json.dumps(obj, ensure_ascii=False)

# ---------- arXiv API helpers ---------

def _norm_arxiv_id(raw_id: str) -> str:
    """
    Accepts forms like 'arXiv:2401.01234v2' or '2401.01234v2' or '2401.01234'.
    Returns canonical '2401.01234' (no 'arXiv:' and no version).
    """
    rid = raw_id.strip()
    rid = re.sub(r"^arXiv:", "", rid)
    rid = re.sub(r"v\d+$", "", rid)
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
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(ARXIV_API, params=params, headers=headers)
        if r.status_code != 200:
            print(f"[arXiv] HTTP {r.status_code} query={search_query!r}")
            return None
        return BeautifulSoup(r.text, "lxml-xml")

def _entry_to_record(entry) -> Optional[Dict]:
    """
    Convert a <entry> node to a dict with id, title, summary, authors, published, pdf_url, abs_url.
    """
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
            # fallback: year only
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
        "published": ydate,  # date
        "abs_url": abs_url,
        "pdf_url": pdf_url,
    }

# ---------- Optional: PDF text (only if you enable ARXIV_FETCH_PDF=1) ----------
def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Minimal dependency path: try PyPDF2; if unavailable, return empty.
    For best quality consider 'pymupdf' or 'pdfminer.six' later.
    """
    try:
        from PyPDF2 import PdfReader
        from io import BytesIO
        r = PdfReader(BytesIO(data))
        parts = []
        for page in r.pages[:30]:  # cap pages for speed
            try:
                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt.strip())
            except Exception:
                continue
        return "\n\n".join(parts)
    except Exception:
        return ""

async def fetch_pdf_text(pdf_url: str) -> str:
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)"}
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.get(pdf_url, headers=headers)
        if r.status_code != 200:
            return ""
        if 300 <= r.status_code < 400:
            print(f"[arXiv] redirect {r.status_code} to {r.headers.get('location')}")
            return ""

        txt = _extract_text_from_pdf_bytes(r.content)
        return txt

# ---------- Public ingest: one arXiv id ----------
async def ingest_one_arxiv_id(arxiv_id: str) -> int:
    """
    Fetch abstract (and optional PDF), chunk, embed, upsert.
    Returns number of chunks inserted.
    """
    arxiv_id = _norm_arxiv_id(arxiv_id)
    print(f"🔹 arXiv ingest start: {arxiv_id}", flush=True)

    # First, query the entry
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
    body = abstract

    # Optional: fetch some PDF text and append
    if ARXIV_FETCH_PDF and rec["pdf_url"]:
        pdf_text = await fetch_pdf_text(rec["pdf_url"])
        if pdf_text:
            # don’t exceed MAX_CHARS
            add = "\n\n" + pdf_text
            body = (title or arxiv_id) + "\n\n" + (abstract or "")
            body = (body + add)[:MAX_CHARS]
        else:
            body = (title or arxiv_id) + "\n\n" + (abstract or "")
    else:
        body = (title or arxiv_id) + "\n\n" + (abstract or "")

    # Chunk
    raw_chunks = chunk_by_tokens(body, CHUNK_TOKENS, CHUNK_OVERLAP)
    chunks = [f"{arxiv_id}\n\n{c}" for c in raw_chunks] or [f"{arxiv_id}\n\n{body}"]
    print(f"🔹 chunks={len(chunks)}", flush=True)

    # Embed
    embs = embedder().encode(chunks, normalize_embeddings=True, batch_size=EMBED_BATCH)
    print(f"🔹 emb shape={embs.shape}", flush=True)

    # Upsert
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        ext_id     = f"arxiv://{arxiv_id}"
        source_uri = rec["abs_url"]
        author     = rec["authors"]
        year_date  = rec["published"] or None  # DATE
        institute  = None                      # arXiv doesn’t carry affiliations reliably
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
        upsert_chunks(cur, doc_id, chunks, embs, arxiv_id)
        con.commit()

    print(f"✅ arXiv ingested {arxiv_id} → {len(chunks)} chunks", flush=True)
    await asyncio.sleep(ARXIV_DELAY)  # polite pacing
    return len(chunks)

# Add this to arxiv_ingest.py

async def ingest_arxiv_topic(
    search_query: str,
    retmax: int = 80,
    *,
    min_request_sleep: float = 3.5,   # polite pacing (arXiv ~1 req / 3s)
    page_size: int = 50,
    skip_existing: bool = True,
) -> dict:
    """
    Search arXiv by topic, paginate, upsert documents+chunks.
    Returns a summary dict: {'found', 'ingested', 'skipped', 'errors'}.
    """
    found = ingested = skipped = errors = 0
    seen_ids: set[str] = set()

    # Helper: fast existence check to avoid re-inserting same arXiv doc
    def _exists(cur, ext_id: str) -> bool:
        cur.execute("SELECT 1 FROM documents WHERE ext_id = %s LIMIT 1;", (ext_id,))
        return cur.fetchone() is not None

    start = 0
    total_target = max(0, retmax)
    if page_size <= 0:
        page_size = 50

    while start < total_target:
        batch = min(page_size, total_target - start)
        soup = await arxiv_query(search_query, start=start, max_results=batch)
        if soup is None:
            errors += 1
            break

        entries = soup.find_all("entry")
        if not entries:
            break

        # Collect records from this page
        page_records = []
        for entry in entries:
            rec = _entry_to_record(entry)
            if not rec:
                continue
            aid = rec["arxiv_id"]
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            page_records.append(rec)

        # Upsert each record
        with connect(**PG_KWARGS) as con, con.cursor() as cur:
            for rec in page_records:
                found += 1
                try:
                    arxiv_id = rec["arxiv_id"]
                    ext_id   = f"arxiv://{arxiv_id}"

                    if skip_existing and _exists(cur, ext_id):
                        skipped += 1
                        continue

                    title     = rec["title"]
                    abstract  = rec["abstract"] or ""
                    body      = (title or arxiv_id) + "\n\n" + abstract

                    # Optional: fetch PDF text and append (kept polite)
                    if ARXIV_FETCH_PDF and rec.get("pdf_url"):
                        try:
                            pdf_txt = await fetch_pdf_text(rec["pdf_url"])
                        except Exception:
                            pdf_txt = ""
                        if pdf_txt:
                            body = (body + "\n\n" + pdf_txt)[:MAX_CHARS]
                        else:
                            body = body[:MAX_CHARS]
                    else:
                        body = body[:MAX_CHARS]

                    # Chunk & embed
                    raw_chunks = chunk_by_tokens(body, CHUNK_TOKENS, CHUNK_OVERLAP)
                    chunks = [f"{arxiv_id}\n\n{c}" for c in (raw_chunks or [body])]
                    embs = embedder().encode(chunks, normalize_embeddings=True, batch_size=EMBED_BATCH)

                    # Upsert document + chunks
                    doc_id = upsert_document(
                        cur=cur,
                        ext_id=ext_id,
                        title=title,
                        source_uri=rec["abs_url"],
                        author=rec["authors"],
                        year_date=rec["published"],   # DATE or None
                        institute=None,               # arXiv usually lacks affiliations
                        source="arXiv",
                        source_id=arxiv_id,
                    )
                    upsert_chunks(cur, doc_id, chunks, embs, arxiv_id)
                    ingested += 1
                except Exception as e:
                    errors += 1
                    print(f"[arXiv] ingest error {rec.get('arxiv_id')}: {type(e).__name__}: {e}", flush=True)

            con.commit()

        # Move pagination forward & be polite
        start += batch
        await asyncio.sleep(min_request_sleep)

    return {"found": found, "ingested": ingested, "skipped": skipped, "errors": errors}