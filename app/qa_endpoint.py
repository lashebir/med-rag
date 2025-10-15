# app/qa_endpoint.py
import os
import re
import json
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
    for i, c in enumerate(contexts):
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
def retrieve_top_k(question: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    qvec = embedder().encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    sql = """
    WITH q AS (SELECT %s::vector AS v)
    SELECT d.ext_id, d.title, d.source_uri, d.source, c.chunk_index, c.text,
           (c.embedding <-> q.v) AS distance
    FROM q, chunks c
    JOIN documents d USING (doc_id)
    WHERE c.embedding_model = %s
    ORDER BY c.embedding <-> q.v
    LIMIT %s;
    """
    rows: List[Dict[str, Any]] = []
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, (qlit, EMBED_MODEL, k))
        rows = cur.fetchall()

    contexts = []
    for r in rows:
        # ext_id should be like "pmcid://PMCxxxxxx"
        pmcid = (r["ext_id"].split("pmcid://", 1)[-1]).upper() if r.get("ext_id") else "PMC?"
        contexts.append({
            "pmcid": pmcid,
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "title": r["title"],
            "source_uri": r["source_uri"],
            "distance": float(r["distance"]),
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
class QARequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: Optional[int] = Field(default=TOP_K, ge=1, le=15)

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
    # 1) retrieve
    contexts = retrieve_top_k(req.question, k=req.top_k or TOP_K)
    if not contexts:
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
