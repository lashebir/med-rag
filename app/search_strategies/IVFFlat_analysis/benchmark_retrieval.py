#!/usr/bin/env python3
import os, time, statistics, json, math, itertools
from typing import List, Dict, Tuple
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

# ------------ Config ------------
PG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

TABLE = os.getenv("CHUNK_TABLE", "chunks")
EMBED_COL = os.getenv("EMBED_COL", "embedding")
ID_COL = os.getenv("ID_COL", "chunk_id")
K = int(os.getenv("K", "10"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))

# Probes to test for IVFFlat (adjust to your dataset size)
PROBES_LIST = [1, 2, 4, 8, 16, 32]

# Queries to test (you can replace/extend)
QUERIES = [
    "cochlear synaptopathy mechanisms",
    "hidden hearing loss in humans",
    "auditory brainstem response thresholds",
    "spiral ganglion neuron degeneration",
    "hair cell regeneration strategies",
    "tinnitus pathophysiology models",
    "age-related hearing loss biomarkers",
    "oxidative stress in inner ear",
    "gene therapy for cochlear damage",
    "single-cell RNA-seq of the cochlea",
]

# ------------ Helpers ------------
def vec_literal(v) -> str:
    # pgvector literal: '[f1,f2,...]'
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

def percentile(data: List[float], p: float) -> float:
    if not data: return float("nan")
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return data_sorted[int(k)]
    d0 = data_sorted[f] * (c - k)
    d1 = data_sorted[c] * (k - f)
    return d0 + d1

def summarize(latencies_ms: List[float]) -> Dict[str, float]:
    return {
        "n": len(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms) if latencies_ms else float("nan"),
        "median_ms": statistics.median(latencies_ms) if latencies_ms else float("nan"),
        "p95_ms": percentile(latencies_ms, 0.95),
    }

# ------------ Main benchmark ------------
def run():
    # Load embedder
    model = SentenceTransformer(EMBED_MODEL)

    # Embed queries
    q_vecs = model.encode(QUERIES, normalize_embeddings=True, batch_size=BATCH_SIZE)

    results = {
        "config": {
            "db": PG["dbname"],
            "table": TABLE,
            "k": K,
            "embed_model": EMBED_MODEL,
            "probes_tested": PROBES_LIST,
            "n_queries": len(QUERIES),
        },
        "baseline_seqscan": {},
        "ivfflat": {},
    }

    with connect(**PG, row_factory=dict_row) as con:
        # Ensure planner knows stats
        with con.cursor() as cur:
            cur.execute(f"ANALYZE {TABLE};")
            con.commit()

        # ---------- Baseline: brute force (sequential) ----------
        baseline_ids_per_q = []
        baseline_latencies = []
        with con.cursor() as cur:
            cur.execute("SET enable_indexscan = off;")
            cur.execute("SET enable_bitmapscan = off;")
            cur.execute("SET enable_seqscan = on;")

            for qvec in q_vecs:
                sql = f"""
                    SELECT {ID_COL}
                    FROM {TABLE}
                    ORDER BY {EMBED_COL} <-> %s::vector
                    LIMIT {K};
                """
                vec = vec_literal(qvec)
                t0 = time.perf_counter()
                cur.execute(sql, (vec,))
                ids = [row[ID_COL] for row in cur.fetchall()]
                dt = (time.perf_counter() - t0) * 1000.0
                baseline_ids_per_q.append(ids)
                baseline_latencies.append(dt)

        results["baseline_seqscan"]["latency"] = summarize(baseline_latencies)

        # ---------- IVFFlat: indexed with different probes ----------
        # Re-enable indexes
        with con.cursor() as cur:
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET enable_bitmapscan = on;")
            cur.execute("SET enable_seqscan = off;")

        for probes in PROBES_LIST:
            with con.cursor() as cur:
                # set probes for this session
                cur.execute(f"SET ivfflat.probes = {probes};")

                latencies = []
                recalls = []  # recall@k vs baseline

                for qi, qvec in enumerate(q_vecs):
                    sql = f"""
                        SELECT {ID_COL}
                        FROM {TABLE}
                        ORDER BY {EMBED_COL} <-> %s::vector
                        LIMIT {K};
                    """
                    vec = vec_literal(qvec)
                    t0 = time.perf_counter()
                    cur.execute(sql, (vec,))
                    ids = [row[ID_COL] for row in cur.fetchall()]
                    dt = (time.perf_counter() - t0) * 1000.0
                    latencies.append(dt)

                    # recall@k: overlap with baseline result set
                    base = set(baseline_ids_per_q[qi])
                    hits = len(base.intersection(ids))
                    recall = round(hits / K,3) if K else 0.0
                    recalls.append(recall)

            results["ivfflat"][str(probes)] = {
                "latency": summarize(latencies),
                "recall_at_k_mean": statistics.fmean(recalls) if recalls else float("nan"),
                "recall_at_k_median": statistics.median(recalls) if recalls else float("nan"),
                "recall_at_k_p5": percentile(recalls, 0.05),
                "recall_at_k_p95": percentile(recalls, 0.95),
            }

    print(json.dumps(results, indent=2))
    print("\nTip: capture this JSON and plot a probes vs. latency/recall chart in your README.")

if __name__ == "__main__":
    run()
