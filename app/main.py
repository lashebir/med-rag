from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os, httpx, asyncio, json, textwrap
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from app.llm_integration.qa_endpoint import router as qa_router
from app.search_strategies.search_strategies import (
    retrieve_ivfflat_only,
    retrieve_ivfflat_ner_boost,
    retrieve_ivfflat_tsvector,
    retrieve_ivfflat_ner_tsvector,
)

# PMC ingestion imports (comment out if not using ingestion endpoints)
# from app.ingestion.PubMed_Central.pmc_resolve import resolve_pmcid
# from app.ingestion.PubMed_Central.pmc_fa_ingest import ingest_one_pmcid, embedder, to_vec_lit, PG_KWARGS

load_dotenv()
app = FastAPI(title="Medical RAG")
app.include_router(qa_router, prefix="/qa", tags=["qa"])

templates = Jinja2Templates(directory="app/templates")

class QueryRequest(BaseModel):
    question: str
    strategy: str = "ivfflat_only"  # ivfflat_only, ner_boost, tsvector, full_hybrid
    k: int = 10

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

@app.post("/query")
def query(req: QueryRequest):
    """
    Enhanced query endpoint with strategy selection and keyword highlighting support.

    Strategies:
    - ivfflat_only: Pure vector similarity (baseline)
    - ner_boost: Vector + NER entity boost
    - tsvector: Vector + full-text keyword search
    - full_hybrid: Vector + NER + full-text (all signals)
    """
    try:
        # Route to appropriate search strategy
        if req.strategy == "ner_boost":
            results = retrieve_ivfflat_ner_boost(req.question, k=req.k)
        elif req.strategy == "tsvector":
            results = retrieve_ivfflat_tsvector(req.question, k=req.k)
        elif req.strategy == "full_hybrid":
            results = retrieve_ivfflat_ner_tsvector(req.question, k=req.k)
        else:  # ivfflat_only (default)
            results = retrieve_ivfflat_only(req.question, k=req.k)

        return {
            "question": req.question,
            "strategy": req.strategy,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.get("/ingest/pmcid/dryrun")
async def ingest_pmcid_dryrun(pmcid: str):
    if not pmcid.startswith("PMC"):
        raise HTTPException(400, "pmcid must start with 'PMC' (e.g., PMC7096775)")

    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    headers = {
        "User-Agent": "med-rag/0.1 (contact: you@example.com)",
        "Accept": "application/json, text/plain, */*",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)

    status = r.status_code
    ctype = r.headers.get("content-type", "")
    body = r.text
    print(f"[dryrun] GET {url} -> {status} {ctype}", flush=True)

    if status != 200:
        snippet = textwrap.shorten(body.strip().replace("\n", " "), width=300)
        raise HTTPException(status, f"PMC fetch failed ({status}). Body: {snippet}")

    try:
        j = r.json()
    except Exception as e:
        snippet = textwrap.shorten(body.strip().replace("\n", " "), width=300)
        raise HTTPException(502, f"Non-JSON response. content-type='{ctype}'. Snippet: {snippet}")

    # minimal shape-agnostic doc count
    def extract_bioc_documents(x):
        if isinstance(x, list): return x
        if isinstance(x, dict):
            if isinstance(x.get("documents"), list): return x["documents"]
            coll = x.get("collection")
            if isinstance(coll, dict) and isinstance(coll.get("documents"), list):
                return coll["documents"]
        return []

    docs = extract_bioc_documents(j)
    passages = 0
    for d in docs:
        ps = d.get("passages") or []
        if isinstance(ps, list):
            passages += len(ps)

    return {"pmcid": pmcid, "documents": len(docs), "passages": passages}

@app.post("/ingest/by_id")
async def ingest_by_id(id: str):
    print(f"→ /ingest/by_id id={id}", flush=True)
    pmcid = await resolve_pmcid(id)
    if not pmcid:
        raise HTTPException(404, f"No PMC record found for '{id}' (likely not Open Access)")
    n = await ingest_one_pmcid(pmcid)
    if n == 0:
        raise HTTPException(404, f"PMC record {pmcid} has no extractable text (BioC empty); try NXML fallback")
    return {"pmcid": pmcid, "chunks": n}
