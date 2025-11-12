#!/usr/bin/env python3
"""
Bulk Google Scholar ingestion via SerpAPI.
Similar to arXiv_bulk_ingest.py - runs multiple topic searches.
"""
import os, sys, asyncio, random
from typing import List, Dict

from app.scholar_ingest import ingest_scholar_topic

SCHOLAR_DELAY = float(os.getenv("SCHOLAR_DELAY", "1.0"))
MAX_TOPIC_CONCURRENCY = 2  # How many topics at once
PAUSE_BETWEEN_TOPICS = (5.0, 10.0)  # seconds
PAUSE_BETWEEN_ROUNDS = (30.0, 60.0)  # seconds
RETMAX_PER_TOPIC = 20  # Adjust based on your needs

# Define your research topics
# These are examples - customize for your medical research focus
# ROUND_1_TOPICS = [
#     "cochlear synaptopathy",
#     "noise-induced hearing loss",
#     "tinnitus pathophysiology",
#     "hidden hearing loss",
#     "hair cell regeneration",
# ]

# ROUND_2_TOPICS = [
#     "auditory brainstem response",
#     "spiral ganglion neurons",
#     "age-related hearing loss presbycusis",
#     "ototoxicity mechanisms",
#     "cochlear implant outcomes",
# ]

# ROUND_3_TOPICS = [
#     # Cellular & Molecular
#     "cochlear synaptic repair",
#     "mitochondrial dysfunction hearing loss",
#     "oxidative stress cochlea",
#     "inflammatory cytokines inner ear",
#     "epigenetic regulation auditory system",

#     # Genetics & Regeneration
#     "gene therapy hearing loss",
#     "Atoh1 hair cell regeneration",
#     "Notch signaling cochlea",
#     "stem cell inner ear",
#     "cochlear organoids",

#     # Clinical & Translational
#     "hidden hearing loss humans",
#     "hearing loss cognitive decline",
#     "audiological assessment synaptopathy",
# ]

# ROUND_4_TOPICS = [
#     # Neuroscience & Computation
#     "computational modeling auditory pathway",
#     "neural encoding auditory cortex",
#     "deep learning auditory neuroscience",
#     "spiking neural networks hearing",
#     "temporal coding auditory system",

#     # Cognitive & Neurodegenerative
#     "hearing loss Alzheimer's disease",
#     "hearing loss Parkinson's disease",
#     "auditory working memory aging",
#     "default mode network auditory",

#     # Neural Prosthetics
#     "cochlear implant neural encoding",
#     "auditory brainstem implant",
#     "brain computer interface auditory",
#     "EEG decoding speech perception",
# ]

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

def dedup_order(seq: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out

async def _run_round(name: str, topics: List[str]):
    """Run one round of topic ingestion."""
    topics = dedup_order(topics)
    sem = asyncio.Semaphore(MAX_TOPIC_CONCURRENCY)
    print(f"\n{'='*80}")
    print(f"{name}: {len(topics)} topics")
    print(f"{'='*80}\n")

    async def do_one(topic: str):
        async with sem:
            print(f"\n🔹 Topic: {topic}")
            try:
                summary: Dict[str, int] = await ingest_scholar_topic(
                    topic,
                    retmax=RETMAX_PER_TOPIC
                )
                print(f"✅ {topic}: {summary}")
            except Exception as e:
                print(f"❌ {topic}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

            # Random pause between topics
            pause = random.uniform(*PAUSE_BETWEEN_TOPICS)
            print(f"[pause] Waiting {pause:.1f}s before next topic...\n")
            await asyncio.sleep(pause)

    # Run all topics in this round
    await asyncio.gather(*(asyncio.create_task(do_one(t)) for t in topics))

    # Pause between rounds
    pause = random.uniform(*PAUSE_BETWEEN_ROUNDS)
    print(f"\n{'='*80}")
    print(f"Round complete. Pausing {pause:.1f}s before next round...")
    print(f"{'='*80}\n")
    await asyncio.sleep(pause)

async def bulk_ingest_all():
    """Run all rounds of bulk ingestion."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  Google Scholar Bulk Ingestion via SerpAPI                     ║
║  Medical & Auditory Neuroscience Research                      ║
╚═══════════════════════════════════════════════════════════════╝
""")

    await _run_round("Round 1: Core Topics", ROUND_1)
    await _run_round("Round 2: Clinical Topics", ROUND_2)
    await _run_round("Round 3: Molecular & Translational", ROUND_3)
    await _run_round("Round 4: Neuroscience & AI", ROUND_4_8)

    print("""
╔═══════════════════════════════════════════════════════════════╗
║  Bulk ingestion complete!                                      ║
╚═══════════════════════════════════════════════════════════════╝
""")

async def main():
    """Main entry point."""
    print(f"[config] MAX_TOPIC_CONCURRENCY={MAX_TOPIC_CONCURRENCY}")
    print(f"[config] RETMAX_PER_TOPIC={RETMAX_PER_TOPIC}")
    print(f"[config] SCHOLAR_DELAY={SCHOLAR_DELAY}s")
    print(f"[config] PAUSE_BETWEEN_TOPICS={PAUSE_BETWEEN_TOPICS}")
    print(f"[config] PAUSE_BETWEEN_ROUNDS={PAUSE_BETWEEN_ROUNDS}\n")

    # Count total topics
    all_topics = dedup_order(
        ROUND_1 + ROUND_2 + ROUND_3 + ROUND_4_8
    )
    print(f"Total unique topics: {len(all_topics)}")
    print(f"Estimated max results: {len(all_topics) * RETMAX_PER_TOPIC}\n")

    response = input("Start bulk ingestion? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    await bulk_ingest_all()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[interrupt] Bulk ingestion stopped by user")
        sys.exit(130)
