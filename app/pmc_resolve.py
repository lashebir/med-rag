# app/pmc_resolve.py
import httpx

async def resolve_pmcid(identifier: str) -> str | None:
    up = identifier.strip().upper()
    if up.startswith("PMC"):
        return up  # already a PMCID

    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"format": "json", "ids": identifier}
    headers = {"User-Agent": "med-rag/0.1 (contact: you@example.com)"}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            j = r.json()
    except Exception as e:
        # Surface the reason in logs and return None (so /ingest/by_id returns 404 instead of 500)
        print(f"[idconv] ERROR for id={identifier}: {type(e).__name__}: {e}", flush=True)
        return None

    recs = j.get("records") or []
    if not recs:
        print(f"[idconv] No records for id={identifier}", flush=True)
        return None

    pmcid = recs[0].get("pmcid")
    if not pmcid:
        print(f"[idconv] No PMCID for id={identifier} (not in PMC / not OA)", flush=True)
        return None

    return pmcid
