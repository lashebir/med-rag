import os, sys, argparse, asyncio, random, re
from typing import List, Dict

# from app.arXiv_ingest import ingest_arxiv_topic, arxiv_query, ingest_one_arxiv_id, _norm_arxiv_id, _entry_to_record
from app.arXiv_fa_ingest import ingest_arxiv_topic, arxiv_query, ingest_one_arxiv_id, _norm_arxiv_id, _entry_to_record

ARXIV_DELAY = float(os.getenv("ARXIV_DELAY", "0.25"))
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "100"))
MAX_TOPIC_CONCURRENCY = 2           # how many topics at once
PAUSE_BETWEEN_TOPICS = (4.0, 7.0)   # seconds
PAUSE_BETWEEN_ROUNDS = (20.0, 30.0) # seconds
ARXIV_MIN_SLEEP = 3.5               # per-request sleep used inside ingest_arxiv_topic (pass through)
RETMAX_PER_TOPIC = 80               # adjust to taste

ROUND_1 = [
    "cochlear synaptopathy",
    "noise-induced hearing loss",
    "tinnitus pathophysiology",
    "single-cell RNA-seq cochlea",
    "hair cell regeneration",
]

ROUND_2 = [
    "hidden hearing loss",
    "hearing loss",
    "tinnitus",
    "otitis media",
    "otitis externa",
    "otitis media with effusion",
    "otitis media with retraction",
    "auditory brainstem response",
    "spiral ganglion neurons",
    "age-related hearing loss",
    "presbycusis",
    "noise-induced hearing loss",  # keep once (de-duped below)
]

ROUND_3 = [
    # --- Cellular & Molecular ---
    "cochlear synaptic repair cochlea",
    "synapse regeneration cochlea",
    "mitochondrial dysfunction cochlea",
    "oxidative stress hearing loss cochlea",
    "apoptosis hair cell cochlea",
    "inflammatory cytokines hearing loss",
    "blood-labyrinth barrier inner ear permeability",
    "epigenetic regulation cochlea",
    "microRNA cochlea hearing loss",

    # --- Central Auditory & Plasticity ---
    "auditory cortex plasticity hearing loss",
    "cochlear nucleus synaptic transmission",
    "inferior colliculus plasticity hearing loss",
    "central auditory hyperactivity tinnitus",
    "temporal processing hearing loss",
    "speech in noise perception hidden hearing loss",

    # --- Genetics & Regeneration ---
    "gene therapy hearing loss cochlea",
    "Atoh1 hair cell regeneration",
    "Notch signaling cochlear regeneration",
    "Wnt signaling inner ear",
    "stem cell otic progenitors",
    "Lgr5 supporting cell cochlea",
    "cochlear organoids inner ear organoids",

    # --- Clinical & Translational ---
    "cochlear implant outcomes speech perception",
    "hearing aid auditory plasticity",
    "ototoxicity aminoglycoside cisplatin",
    "auditory neuropathy ABR",
    "hidden hearing loss humans",
    "hearing loss cognitive decline",

    # --- Methods & Models ---
    "single-cell RNA-seq inner ear cochlea",
    "spatial transcriptomics auditory",
    "machine learning otology hearing loss",
    "electrophysiology spiral ganglion",
    "functional MRI auditory cortex",

    # --- Aging & Environmental ---
    "aging cochlea transcriptome",
    "metabolic stress presbycusis",
    "noise exposure cochlear synaptopathy",
    "dietary antioxidant hearing loss",
]

ROUND_4_8 = [
    # --- Cognitive and Neurodegenerative ---
    "hearing loss Alzheimer's disease",
    "hearing loss Parkinson's disease",
    "hearing loss cognitive load",
    "auditory working memory aging",
    "default mode network auditory",
    "functional MRI auditory cognition",
    "brainstem auditory pathway neurodegeneration",
    "white matter microstructure auditory processing",

    # --- Computational and Theoretical Neuroscience ---
    "computational modeling auditory pathway",
    "neural encoding auditory cortex",
    "deep learning auditory neuroscience",
    "spiking neural networks hearing",
    "temporal coding auditory system",
    "information theory auditory",
    "Bayesian models speech perception",
    "predictive coding auditory cortex",
    "neural decoding auditory evoked potentials",
    "biophysical models hair cell",

    # --- Neuroinflammation and Metabolism ---
    "neuroinflammation auditory system",
    "microglia auditory cortex",
    "blood-brain barrier hearing loss",
    "systemic inflammation hearing loss",
    "metabolic syndrome hearing loss",
    "vascular dysfunction cochlea",
    "mitochondrial dynamics auditory neurons",

    # --- Neural Prosthetics and BCIs ---
    "cochlear implant neural encoding",
    "cochlear implant brain plasticity",
    "auditory brainstem implant speech perception",
    "electrical stimulation auditory pathway",
    "neuroprosthetics hearing restoration",
    "brain computer interface auditory",
    "EEG decoding speech perception",
    "neural decoding speech intelligibility",
    "MEG auditory attention decoding",
    "auditory attention neural tracking",
    "computational modeling auditory prosthesis",
    "biophysical modeling auditory stimulation",
    "neural interface temporal coding",
    "deep learning auditory decoding",
    "closed loop auditory feedback",
    "machine learning cochlear implant outcomes",
    "adaptive stimulation auditory implant",
    "real time neural decoding auditory cortex",
    "brain connectivity auditory prosthesis",
    "BCI speech comprehension"
]


async def ingest_query(search_query: str, limit: int = 200, page_size: int = 100) -> dict:
    """
    Search arXiv by query string and ingest results.
    """
    print(f"[arXiv] query: {search_query!r}")
    found, ingested, skipped, errors = 0, 0, 0, 0
    start = 0
    seen_ids = set()

    while found < limit:
        batch = min(page_size, limit - found)
        soup = await arxiv_query(search_query, start=start, max_results=batch)
        if soup is None:
            print("[arXiv] query failed; stopping page loop")
            break

        entries = soup.find_all("entry")
        if not entries:
            break

        ids = []
        for e in entries:
            rec = _entry_to_record(e)
            if not rec: 
                continue
            aid = rec["arxiv_id"]
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            ids.append(aid)

        if not ids:
            break

        found += len(ids)
        print(f"[arXiv] page start={start} got={len(ids)} (total found={found})")

        # ingest with small concurrency
        sem = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENCY", "3")))
        async def worker(aid: str):
            async with sem:
                try:
                    n = await ingest_one_arxiv_id(aid)
                    return ("ok", n)
                except Exception as e:
                    print(f"[arXiv] FAIL {aid}: {type(e).__name__}: {e}")
                    return ("err", 0)

        results = await asyncio.gather(*(worker(a) for a in ids))
        for status, n in results:
            if status == "ok":
                ingested += 1 if n > 0 else 0
                skipped  += 1 if n == 0 else 0
            else:
                errors += 1

        start += len(entries)
        await asyncio.sleep(ARXIV_DELAY + random.random() * 0.2)

    summary = {"found": found, "ingested": ingested, "skipped": skipped, "errors": errors}
    print(f"[arXiv] summary: {summary}")
    return summary

# ---------- query normalization for arXiv ----------
OA_FILTER_RE = re.compile(r'\(\s*"open access"\s*\[filter\]\s*\)', flags=re.I)
DATE_RANGE_RE = re.compile(r'\(\s*\d{4}\s*:\s*\d{4}\s*\[dp\]\s*\)', flags=re.I)

def sanitize_for_arxiv(q: str) -> str:
    """Strip PubMed-only bits and yield an arXiv-friendly query for 'all:'."""
    q = OA_FILTER_RE.sub("", q)
    q = DATE_RANGE_RE.sub("", q)
    q = q.replace("AND", "AND").replace("OR", "OR")  # keep explicit ops
    q = re.sub(r"\s+", " ", q).strip()

    # arXiv 'all:' searches title, abstract, comment, journal_ref; keep quoted phrases
    # Wrap unquoted bare tokens in double quotes to lean toward phrasey behavior
    tokens = []
    for tok in re.split(r'(\s+|AND|OR|\(|\))', q):
        if tok in ("AND", "OR", "(", ")", " ", "\t", "\n", "\r", ""):
            tokens.append(tok)
            continue
        if tok.startswith('"') and tok.endswith('"'):
            tokens.append(tok)
        else:
            # protect single words
            tokens.append(f'"{tok}"')
    arxiv_bool = "".join(tokens)
    # Prefix with all: where appropriate (e.g., all:"term")
    arxiv_bool = re.sub(r'(".*?")', r'all:\1', arxiv_bool)
    # Fix patterns like all:("x") -> all:"x"
    arxiv_bool = arxiv_bool.replace('all:("', 'all:"').replace('")', '"')
    return arxiv_bool

def dedup_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out

async def _run_round(name: str, topics: List[str]):
    topics = dedup_order(topics)
    sem = asyncio.Semaphore(MAX_TOPIC_CONCURRENCY)
    print(f"\n=== {name}: {len(topics)} topics ===")

    async def do_one(topic: str):
        async with sem:
            q = sanitize_for_arxiv(topic)
            print(f"\n🔹 Topic: {topic}\n   ↳ arXiv query: {q}")
            try:
                # IMPORTANT: your arxiv_ingest.ingest_arxiv_topic should **sleep ~ARXIV_MIN_SLEEP between API calls**
                # to remain polite; we pass it down so it can obey.
                summary: Dict[str, int] = await ingest_arxiv_topic(
                    q,
                    retmax=RETMAX_PER_TOPIC
                )
                print(f"✅ {topic}: {summary}")
            except Exception as e:
                print(f"❌ {topic}: {type(e).__name__}: {e}")
            await asyncio.sleep(random.uniform(*PAUSE_BETWEEN_TOPICS))

    await asyncio.gather(*(asyncio.create_task(do_one(t)) for t in topics))
    await asyncio.sleep(random.uniform(*PAUSE_BETWEEN_ROUNDS))

async def polite_ingest_arxiv_all():
    await _run_round("Round 1", ROUND_1)
    await _run_round("Round 2", ROUND_2)
    await _run_round("Round 3 (biomed queries normalized for arXiv)", ROUND_3)
    await _run_round("Rounds 4-8", ROUND_4_8)

async def main():
    p = argparse.ArgumentParser("Bulk arXiv ingestion")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="arXiv search_query (e.g., 'cat:cs.CL AND deep learning')")
    mode.add_argument("--ids", nargs="+", help="Explicit arXiv IDs (with or without arXiv: / version)")
    mode.add_argument("--from-file", help="File with one arXiv ID per line")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--page-size", type=int, default=100)
    args = p.parse_args()

    if args.query:
        await ingest_query(args.query, limit=args.limit, page_size=args.page_size)
        return

    ids: List[str] = []
    if args.ids:
        ids.extend(args.ids)
    if args.from_file:
        with open(args.from_file) as fh:
            ids.extend([ln.strip() for ln in fh if ln.strip()])

    ids = [_norm_arxiv_id(a) for a in ids]
    # small-concurrency ingest
    sem = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENCY", "3")))
    async def worker(aid: str):
        async with sem:
            try:
                await ingest_one_arxiv_id(aid)
            except Exception as e:
                print(f"[arXiv] FAIL {aid}: {type(e).__name__}: {e}")

    await asyncio.gather(*(worker(a) for a in ids))

if __name__ == "__main__":
    asyncio.run(polite_ingest_arxiv_all())
