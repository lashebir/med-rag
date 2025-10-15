import os
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import httpx
from psycopg import connect
from psycopg.rows import dict_row
from app.pmc_ingest import ingest_one_pmcid, PG_KWARGS

load_dotenv()

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
USER_AGENT = "med-rag/0.1 (contact: you@example.com)"  # set your email
NCBI_DELAY = float(os.getenv("NCBI_DELAY_SEC", "0.4"))  # 3 req/sec without key
RETMAX_DEFAULT = int(os.getenv("RETMAX", "50"))         # per topic cap
API_KEY = os.getenv("NCBI_API_KEY")

# ---- DB helpers (optional: skip already ingested PMCIDs) --------------------
def existing_ext_ids() -> set:
    ids = set()
    try:
        with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
            cur.execute("SELECT ext_id FROM documents;")
            for row in cur.fetchall():
                ids.add(row["ext_id"])
    except Exception as e:
        print(f"[warn] Could not read existing docs: {e}", flush=True)
    return ids

# ---- ESearch: find PMCIDs for a topic --------------------------------------
async def search_pmcids(
    query: str,
    retmax: int = RETMAX_DEFAULT,
    date_range: Optional[Tuple[str, str]] = None,  # ("2019","2025")
) -> List[str]:
    """
    Use ESearch against db=pmc to get PMCID integer IDs; wrap with 'PMC'.
    date_range: publication year range ("YYYY","YYYY"), applied as YYYY:YYYY[dp]
    """
    params = {
        "db": "pmc",
        "retmode": "json",
        "retmax": retmax,
        "term": query,
    }
    if date_range:
        y1, y2 = date_range
        # publication date filter
        params["term"] = f"({query}) AND {y1}:{y2}[dp]"
    if API_KEY:
        params["api_key"] = API_KEY

    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(EUTILS_BASE, params=params, headers=headers)
        r.raise_for_status()
        j = r.json()

    ids = j.get("esearchresult", {}).get("idlist", []) or []
    pmcids = [f"PMC{_id}" for _id in ids]
    print(f"[search] query='{query}' -> {len(pmcids)} PMCIDs", flush=True)
    return pmcids

# ---- Ingest orchestrator ----------------------------------------------------
async def ingest_topic(
    topic: str,
    retmax: int = RETMAX_DEFAULT,
    date_range: Optional[Tuple[str, str]] = None,
    max_concurrency: int = 3,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Search for PMCIDs for a topic and ingest them concurrently (rate-limited).
    Returns summary counts.
    """
    pmcids = await search_pmcids(topic, retmax=retmax, date_range=date_range)
    if not pmcids:
        return {"found": 0, "ingested": 0, "skipped": 0, "errors": 0}

    skip_set = existing_ext_ids() if skip_existing else set()
    to_process = []
    skipped = 0
    for pmcid in pmcids:
        ext = f"pmcid://{pmcid}"
        if skip_existing and ext in skip_set:
            skipped += 1
        else:
            to_process.append(pmcid)

    print(f"[plan] {topic}: {len(to_process)} to ingest, {skipped} skipped (already in DB)", flush=True)

    sem = asyncio.Semaphore(max_concurrency)

    async def _ingest_one(pmcid: str) -> Tuple[str, int]:
        async with sem:
            # polite pacing for EFetch/BioC inside ingest_one
            await asyncio.sleep(NCBI_DELAY)
            try:
                n = await ingest_one_pmcid(pmcid)
                print(f"[ok] {pmcid}: {n} chunks", flush=True)
                return ("ok", n)
            except Exception as e:
                print(f"[err] {pmcid}: {e}", flush=True)
                return ("err", 0)

    tasks = [_ingest_one(p) for p in to_process]
    results = await asyncio.gather(*tasks)

    ok_count = sum(1 for s, _ in results if s == "ok")
    err_count = sum(1 for s, _ in results if s == "err")

    return {
        "found": len(pmcids),
        "ingested": ok_count,
        "skipped": skipped,
        "errors": err_count,
    }

# ---- Entry point: define your topic set here --------------------------------
TOPICS = [
    # Round 1
    "cochlear synaptopathy",
    "noise-induced hearing loss",
    "tinnitus pathophysiology",
    "single-cell RNA-seq cochlea",
    "hair cell regeneration",
    # Round 2
    "hidden hearing loss",
    "hearing loss",
    "tinnitus",
    "otitis media",
    "otitis externa",
    "otitis media with effusion",
    "otitis media with retraction",
    "otitis media with retraction",
    "noise-induced hearing loss",
    "auditory brainstem response",
    "spiral ganglion neurons",
    "age-related hearing loss",
    "presbycusis",
    # --- Cellular and Molecular ---
    '("cochlear synaptic repair" OR "synapse regeneration") AND cochlea AND ("open access"[filter])',
    '"mitochondrial dysfunction" AND cochlea AND ("open access"[filter])',
    '"oxidative stress" AND ("hearing loss" OR cochlea) AND ("open access"[filter])',
    '"apoptosis" AND ("hair cell" OR cochlea) AND ("open access"[filter])',
    '"inflammatory cytokines" AND "hearing loss" AND ("open access"[filter])',
    '("blood-labyrinth barrier" OR "inner ear barrier permeability") AND ("open access"[filter])',
    '"epigenetic regulation" AND cochlea AND ("open access"[filter])',
    '"microRNA" AND ("cochlea" OR "hearing loss") AND ("open access"[filter])',

    # --- Central Auditory and Plasticity ---
    '"auditory cortex plasticity" AND "hearing loss" AND ("open access"[filter])',
    '"cochlear nucleus" AND "synaptic transmission" AND ("open access"[filter])',
    '"inferior colliculus" AND "plasticity" AND "hearing loss" AND ("open access"[filter])',
    '"central auditory hyperactivity" AND "tinnitus" AND ("open access"[filter])',
    '"temporal processing" AND "hearing loss" AND ("open access"[filter])',
    '"speech in noise perception" AND "hidden hearing loss" AND ("open access"[filter])',

    # --- Genetics and Regeneration ---
    '"gene therapy" AND ("hearing loss" OR "cochlea") AND ("open access"[filter])',
    '"Atoh1" AND "hair cell regeneration" AND ("open access"[filter])',
    '"Notch signaling" AND "cochlear regeneration" AND ("open access"[filter])',
    '"Wnt signaling" AND "inner ear" AND ("open access"[filter])',
    '"stem cell" AND "otic progenitors" AND ("open access"[filter])',
    '"Lgr5" AND "supporting cell" AND "cochlea" AND ("open access"[filter])',
    '("cochlear organoids" OR "inner ear organoids") AND ("open access"[filter])',

    # --- Clinical and Translational ---
    '"cochlear implant outcomes" AND "speech perception" AND ("open access"[filter])',
    '"hearing aid" AND "auditory plasticity" AND ("open access"[filter])',
    '"ototoxicity prevention" AND ("aminoglycoside" OR "cisplatin") AND ("open access"[filter])',
    '"auditory neuropathy" AND "ABR" AND ("open access"[filter])',
    '"hidden hearing loss" AND "humans" AND ("open access"[filter])',
    '"hearing loss" AND "cognitive decline" AND ("open access"[filter])',

    # --- Advanced Methods and Models ---
    '"single-cell RNA-seq" AND ("inner ear" OR cochlea) AND ("open access"[filter])',
    '"spatial transcriptomics" AND "auditory" AND ("open access"[filter])',
    '"machine learning" AND ("otology" OR "hearing loss") AND ("open access"[filter])',
    '"electrophysiology" AND "spiral ganglion" AND ("open access"[filter])',
    '"functional MRI" AND "auditory cortex" AND ("open access"[filter])',

    # --- Aging and Environmental ---
    '"aging cochlea" AND transcriptome AND ("open access"[filter])',
    '"metabolic stress" AND "presbycusis" AND ("open access"[filter])',
    '"noise exposure" AND "cochlear synaptopathy" AND ("open access"[filter])',
    '"dietary antioxidant" AND "hearing loss" AND ("open access"[filter])',
]

async def main():
    YEAR_RANGE = (os.getenv("YEAR_START", "2018"), os.getenv("YEAR_END", "2025"))
    RETMAX = int(os.getenv("RETMAX", "40"))
    CONC = int(os.getenv("MAX_CONCURRENCY", "3"))

    print(f"Bulk ingest over topics: {TOPICS}")
    grand = {"found": 0, "ingested": 0, "skipped": 0, "errors": 0}
    for t in TOPICS:
        summary = await ingest_topic(
            t, retmax=RETMAX, date_range=YEAR_RANGE, max_concurrency=CONC, skip_existing=True
        )
        print(f"[summary] {t}: {summary}", flush=True)
        for k in grand:
            grand[k] += summary[k]
    print(f"\n[grand total] {grand}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())