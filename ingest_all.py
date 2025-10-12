# app/pmc_ingest.py
import os, hashlib, json
from typing import List, Dict
import httpx
from sentence_transformers import SentenceTransformer
from psycopg import connect
from psycopg.rows import dict_row

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))

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
    print(f"🔹 Starting ingestion for {pmcid}")

    bioc = await fetch_bioc_json(pmcid)
    print(f"🔹 Retrieved BioC JSON for {pmcid} (len={len(json.dumps(bioc)):,})")

    passages = passages_from_bioc(bioc)
    print(f"🔹 Extracted {len(passages)} passages from BioC")

    if not passages:
        print(f"⚠️ No text found for {pmcid}")
        return 0

    title = f"{pmcid}"
    full_text = "\n\n".join(passages)
    print(f"🔹 Combined text length: {len(full_text)} chars")

    # chunk
    raw_chunks = chunk_by_tokens(full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
    chunks = [f"{title}\n\n{c}" for c in raw_chunks]
    print(f"🔹 Created {len(chunks)} chunks")

    # embed
    print(f"🔹 Encoding chunks with {EMBED_MODEL}")
    embs = embedder().encode(chunks, normalize_embeddings=True, batch_size=32)
    print(f"🔹 Embedding shape: {embs.shape}")

    # insert
    print(f"🔹 Connecting to DB: {PG_KWARGS['dbname']}")
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        ext_id = f"pmcid://{pmcid}"
        source_uri = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}"
        doc_id = ensure_document(cur, ext_id, title, source_uri)
        upsert_chunks(cur, doc_id, chunks, embs, pmcid)
        con.commit()
        print(f"✅ Inserted {len(chunks)} chunks for {pmcid}")

    print(f"✨ Done ingesting {pmcid}")
    return len(chunks)

# async def ingest_one_pmcid(pmcid: str) -> int:
#     bioc = await fetch_bioc_json(pmcid)        # 1) fetch JIT
#     passages = passages_from_bioc(bioc)        # 2) extract readable text
#     if not passages:
#         return 0
#     title = f"{pmcid}"
#     full_text = "\n\n".join(passages)

#     # 3) chunk (deterministic)
#     base_header = f"{pmcid}"
#     raw_chunks = chunk_by_tokens(full_text, CHUNK_TOKENS, CHUNK_OVERLAP)
#     chunks = [f"{base_header}\n\n{c}" for c in raw_chunks]

#     # 4) embed in-memory
#     embs = embedder().encode(chunks, normalize_embeddings=True, batch_size=64)

#     # 5) insert vectors + minimal metadata
#     with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
#         ext_id = f"pmcid://{pmcid}"
#         source_uri = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}"
#         doc_id = ensure_document(cur, ext_id, title, source_uri)
#         upsert_chunks(cur, doc_id, chunks, embs, pmcid)
#         con.commit()

#     # 6) discard full text (was only in RAM)
#     return len(chunks)
