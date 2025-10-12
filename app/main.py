from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os, httpx, asyncio
from app.pmc_ingest import ingest_one_pmcid, embedder, to_vec_lit, PG_KWARGS
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

app = FastAPI(title="PubMed RAG")

templates = Jinja2Templates(directory="app/templates")

class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def query(req: QueryRequest):
    # Replace with your real RAG logic later
    answer = f"(demo) You asked: {req.question}"
    return {"answer": answer, "citations": []}

class SearchBody(BaseModel):
    query: str
    retmax: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/pmcid")
async def ingest_pmcid(pmcid: str = Query(..., pattern=r"PMC\d+")):
    try:
        n = await ingest_one_pmcid(pmcid)
        if n == 0:
            raise HTTPException(404, f"No text found for {pmcid}")
        return {"pmcid": pmcid, "chunks": n}
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"PMC fetch failed: {e}") from e

@app.post("/ingest/search")
async def ingest_search(body: SearchBody):
    # Use ESearch to find PMCIDs for a query, then ingest each
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pmc", "retmode": "json", "retmax": body.retmax, "term": body.query}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()
    pmcids = [f"PMC{id_}" for id_ in j.get("esearchresult", {}).get("idlist", [])]
    if not pmcids:
        return {"query": body.query, "ingested": []}
    results = []
    for pmcid in pmcids:
        try:
            n = await ingest_one_pmcid(pmcid)
            results.append({"pmcid": pmcid, "chunks": n})
        except Exception as e:
            results.append({"pmcid": pmcid, "error": str(e)})
    return {"query": body.query, "ingested": results}

class QueryBody(BaseModel):
    question: str
    k: int = 6
    hydrate: bool = False   # if True, re-fetch full text for top-K (optional)

@app.post("/query")
def query(body: QueryBody):
    # 1) embed question
    emb: SentenceTransformer = embedder()
    qvec = emb.encode([body.question], normalize_embeddings=True)[0]
    lit = to_vec_lit(qvec)

    # 2) vector search
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute("""
            SELECT c.chunk_id, c.text, c.metadata, d.title, d.source_uri,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            ORDER BY c.embedding <-> %s::vector
            LIMIT %s;
        """, (lit, lit, body.k))
        hits = cur.fetchall()

    # 3) optional rehydration (fetch full BioC and reconstruct the exact chunk)
    #    Keeping it minimal here; you can add a helper that re-fetches PMC then re-chunks
    #    to extract the full chunk text for each (pmcid, chunk_index).
    results = []
    for h in hits:
        meta = h["metadata"] or {}
        results.append({
            "chunk_id": h["chunk_id"],
            "pmcid": meta.get("pmcid"),
            "chunk_index": meta.get("chunk_index"),
            "preview": h["text"],
            "similarity": float(h["similarity"]),
            "source": h["source_uri"],
        })

    # 4) (Your LLM call would go here, using the previews or hydrated chunks)
    return {"question": body.question, "results": results}