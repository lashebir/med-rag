# app/qa_endpoint.py
import os
import re
import requests
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Config ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1:8b-instruct-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
TOP_K = int(os.getenv("TOP_K", "5"))

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

# --- Utils ---
_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def to_vec_lit(vec) -> str:
    # Postgres vector literal
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

def build_system_prompt() -> str:
    return (
        "You are a biomedical assistant. Answer ONLY using the provided CONTEXT.\n"
        "- Cite each claim with bracketed refs like [PMCID:chunk].\n"
        "- If evidence is insufficient, say you don't know.\n"
        "- Be concise and factual."
    )

def build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx = []
    for c in contexts:
        label = f"{c['pmcid']}#{c['chunk_index']}"
        ctx.append(f"[{label}] {c['text']}")
    ctx_block = "\n\n".join(ctx)
    return (
        f"[CONTEXT]\n{ctx_block}\n[/CONTEXT]\n"
        f"[QUESTION]\n{question}\n[/QUESTION]\n"
        "Answer with citations in the form [PMCID:chunk]."
    )

def parse_citations(text: str) -> List[Dict[str, Any]]:
    """
    Extract citations like [PMC1234567#3] or [PMCID:3] or [PMC1234567:3].
    Returns list of {'pmcid': 'PMCxxxxxx', 'chunk_index': int} (deduped).
    """
    found = set()
    out = []
    # Match inside [ ... ]
    for m in re.finditer(r"\[([^\[\]]+)\]", text):
        inner = m.group(1)
        # Try patterns
        # e.g., PMC1234567#3  or  PMC1234567:3  or  PMCID:3 (ambiguous)
        pm = re.search(r"(PMC\d+)", inner, flags=re.IGNORECASE)
        ci = re.search(r"(?:#|:)\s*(\d+)", inner)
        if pm and ci:
            pmcid = pm.group(1).upper()
            idx = int(ci.group(1))
            key = (pmcid, idx)
            if key not in found:
                found.add(key)
                out.append({"pmcid": pmcid, "chunk_index": idx})
    return out

# --- DB retrieval ---
def build_metadata_filter_sql(filters: Optional['MetadataFilters']) -> tuple[str, List[Any]]:
    """
    Build SQL WHERE clauses and parameters for metadata filtering.
    Returns (where_clause, params_list)
    """
    if not filters:
        return "", []

    conditions = []
    params = []

    # Filter by authors (TEXT[] array - case-insensitive partial match)
    if filters.authors:
        author_conditions = []
        for author_name in filters.authors:
            author_conditions.append(
                "EXISTS (SELECT 1 FROM unnest(d.author) AS a WHERE LOWER(a) LIKE LOWER(%s))"
            )
            params.append(f"%{author_name}%")
        # Match any of the specified authors
        if author_conditions:
            conditions.append(f"({' OR '.join(author_conditions)})")

    # Filter by year range
    if filters.year_min is not None:
        conditions.append("EXTRACT(YEAR FROM d.year) >= %s")
        params.append(filters.year_min)

    if filters.year_max is not None:
        conditions.append("EXTRACT(YEAR FROM d.year) <= %s")
        params.append(filters.year_max)

    # Filter by institutions (case-insensitive partial match)
    if filters.institutions:
        inst_conditions = []
        for inst in filters.institutions:
            inst_conditions.append("LOWER(d.institute) LIKE LOWER(%s)")
            params.append(f"%{inst}%")
        # Match any of the specified institutions
        if inst_conditions:
            conditions.append(f"({' OR '.join(inst_conditions)})")

    # Filter by sources (exact match, case-insensitive)
    if filters.sources:
        conditions.append("d.source = ANY(%s)")
        params.append(filters.sources)

    where_clause = " AND ".join(conditions) if conditions else ""
    return where_clause, params

def retrieve_top_k(question: str, k: int = TOP_K, filters: Optional['MetadataFilters'] = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks using semantic similarity (cosine distance with IVFFlat index).
    Metadata filters are applied BEFORE semantic search for efficiency.

    IMPORTANT: Pass query vector as literal parameter to enable IVFFlat index usage.
    """
    qvec = embedder().encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    # Build metadata filter WHERE clause
    filter_where, filter_params = build_metadata_filter_sql(filters)

    # Base WHERE clause for embedding model
    base_where = "c.embedding_model = %s"

    # Combine WHERE clauses
    if filter_where:
        where_clause = f"WHERE {base_where} AND {filter_where}"
    else:
        where_clause = f"WHERE {base_where}"

    # Use cosine distance (<=>) to match our IVFFlat indexes
    # Pass vector as literal parameter (not CTE) to enable index usage
    sql = f"""
    SELECT d.ext_id, d.title, d.source_uri, d.source, d.author, d.year, d.institute,
           c.chunk_index, c.text,
           (c.embedding <=> %s::vector) AS distance,
           (1 - (c.embedding <=> %s::vector)) AS similarity
    FROM chunks c
    JOIN documents d USING (doc_id)
    {where_clause}
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s;
    """

    # Combine all parameters: vector (3 times), embedding_model, filter_params, limit
    params = [qlit, qlit, EMBED_MODEL] + filter_params + [qlit, k]

    rows: List[Dict[str, Any]] = []
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    contexts = []
    for r in rows:
        # Handle different ext_id formats: "pmcid://PMCxxxxxx" or "arxiv://arxiv_id"
        ext_id = r.get("ext_id", "")
        if "pmcid://" in ext_id:
            doc_id = ext_id.split("pmcid://", 1)[-1].upper()
        elif "arxiv://" in ext_id:
            doc_id = ext_id.split("arxiv://", 1)[-1]
        elif "scholar://" in ext_id:
            doc_id = ext_id.split("scholar://", 1)[-1]
        else:
            doc_id = ext_id or "UNKNOWN"

        # Extract year from date field if present
        year = None
        if r.get("year"):
            try:
                year = r["year"].year if hasattr(r["year"], "year") else int(r["year"])
            except:
                pass

        contexts.append({
            "pmcid": doc_id,  # Keep as "pmcid" for backward compatibility, but may be arxiv_id
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "title": r["title"],
            "source_uri": r["source_uri"],
            "source": r.get("source"),
            "authors": r.get("author"),  # List of authors
            "year": year,
            "institution": r.get("institute"),
            "distance": float(r["distance"]),
            "similarity": float(r["similarity"]),
        })
    return contexts

# --- Ollama call ---
def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": LOCAL_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=180,
        )
        r.raise_for_status()
        j = r.json()
        return j.get("response", "")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

# --- FastAPI shapes ---
class MetadataFilters(BaseModel):
    """Optional metadata filters applied BEFORE semantic search."""
    authors: Optional[List[str]] = Field(
        default=None,
        description="Filter by author names (case-insensitive partial match). Documents must have at least one matching author."
    )
    year_min: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Minimum publication year (inclusive)"
    )
    year_max: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description="Maximum publication year (inclusive)"
    )
    institutions: Optional[List[str]] = Field(
        default=None,
        description="Filter by institution names (case-insensitive partial match)"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Filter by source: arXiv, PubMed Central, Google Scholar"
    )

class QARequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: Optional[int] = Field(default=TOP_K, ge=1, le=15)
    filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Optional metadata filters applied before semantic search"
    )

class Citation(BaseModel):
    pmcid: str
    chunk_index: int
    source_uri: Optional[str] = None
    title: Optional[str] = None

class QAResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_contexts: List[Dict[str, Any]]

router = APIRouter()

@router.post("/answer", response_model=QAResponse)
def answer(req: QARequest):
    """
    Answer a question using RAG with optional metadata filtering.

    Filters are applied BEFORE semantic search:
    - Filter by author(s), year range, institution(s), or source(s)
    - Then find top-k most relevant chunks from filtered documents
    - Generate answer with citations

    Example with filters:
    ```json
    {
        "question": "What are the latest treatments for diabetes?",
        "top_k": 5,
        "filters": {
            "authors": ["Smith"],
            "year_min": 2020,
            "sources": ["PubMed Central"]
        }
    }
    ```
    """
    # 1) retrieve with metadata filters
    contexts = retrieve_top_k(req.question, k=req.top_k or TOP_K, filters=req.filters)
    if not contexts:
        if req.filters:
            raise HTTPException(
                status_code=404,
                detail="No relevant contexts found matching the specified filters. Try relaxing the filter criteria."
            )
        else:
            raise HTTPException(status_code=404, detail="No relevant contexts found")

    # 2) build prompt
    system = build_system_prompt()
    user = build_user_prompt(req.question, contexts)
    full_prompt = f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n{user}\n"

    # 3) generate
    completion = call_ollama(full_prompt).strip()

    # 4) parse citations; enrich with URIs/titles where possible
    cits_raw = parse_citations(completion)
    # map for enrichment
    by_key = {(c["pmcid"], c["chunk_index"]): c for c in contexts}
    citations: List[Dict[str, Any]] = []
    seen = set()
    for c in cits_raw:
        key = (c["pmcid"], c["chunk_index"])
        if key in seen: 
            continue
        seen.add(key)
        info = by_key.get(key, {})
        citations.append({
            "pmcid": c["pmcid"],
            "chunk_index": c["chunk_index"],
            "source_uri": info.get("source_uri"),
            "title": info.get("title"),
        })

    # Fallback: if model failed to cite, attach the top-1 context as supporting ref
    if not citations and contexts:
        c0 = contexts[0]
        citations = [{
            "pmcid": c0["pmcid"],
            "chunk_index": c0["chunk_index"],
            "source_uri": c0["source_uri"],
            "title": c0["title"],
        }]

    return QAResponse(answer=completion, citations=citations, used_contexts=contexts)

@router.get("/filters/available")
def get_available_filters():
    """
    Get available metadata filter values from the database.
    Useful for building UI dropdowns or understanding the data.

    Returns:
    - sources: List of available data sources
    - year_range: Min and max publication years
    - document_count: Total number of documents
    - top_authors: Most frequent authors (top 20)
    - top_institutions: Most frequent institutions (top 20)
    """
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        # Get available sources
        cur.execute("""
            SELECT DISTINCT source
            FROM documents
            WHERE source IS NOT NULL
            ORDER BY source;
        """)
        sources = [row["source"] for row in cur.fetchall()]

        # Get year range
        cur.execute("""
            SELECT
                MIN(EXTRACT(YEAR FROM year)) as min_year,
                MAX(EXTRACT(YEAR FROM year)) as max_year
            FROM documents
            WHERE year IS NOT NULL;
        """)
        year_data = cur.fetchone()
        year_range = {
            "min": int(year_data["min_year"]) if year_data["min_year"] else None,
            "max": int(year_data["max_year"]) if year_data["max_year"] else None,
        }

        # Get document count
        cur.execute("SELECT COUNT(*) as count FROM documents;")
        doc_count = cur.fetchone()["count"]

        # Get top authors (unnest the array and count occurrences)
        cur.execute("""
            SELECT author_name, COUNT(*) as doc_count
            FROM documents, unnest(author) AS author_name
            WHERE author IS NOT NULL
            GROUP BY author_name
            ORDER BY doc_count DESC, author_name
            LIMIT 20;
        """)
        top_authors = [
            {"name": row["author_name"], "document_count": row["doc_count"]}
            for row in cur.fetchall()
        ]

        # Get top institutions
        cur.execute("""
            SELECT institute, COUNT(*) as doc_count
            FROM documents
            WHERE institute IS NOT NULL AND institute != ''
            GROUP BY institute
            ORDER BY doc_count DESC
            LIMIT 20;
        """)
        top_institutions = [
            {"name": row["institute"], "document_count": row["doc_count"]}
            for row in cur.fetchall()
        ]

    return {
        "sources": sources,
        "year_range": year_range,
        "document_count": doc_count,
        "top_authors": top_authors,
        "top_institutions": top_institutions,
        "usage_examples": [
            {
                "description": "Filter by author",
                "filters": {
                    "authors": ["Smith"]
                }
            },
            {
                "description": "Filter by recent papers (2020-2024)",
                "filters": {
                    "year_min": 2020,
                    "year_max": 2024
                }
            },
            {
                "description": "Filter by source",
                "filters": {
                    "sources": ["PubMed Central"]
                }
            },
            {
                "description": "Combined filters",
                "filters": {
                    "authors": ["Johnson"],
                    "year_min": 2018,
                    "sources": ["arXiv", "PubMed Central"]
                }
            }
        ]
    }
