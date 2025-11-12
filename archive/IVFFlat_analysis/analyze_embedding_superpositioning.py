#!/usr/bin/env python3
"""
analyze_embedding_superposition.py

Quantifies entanglement in your dense embedding space using:
  1) PCA variance concentration
  2) Neighbor stability across random subspace ablations
  3) (Optional) composition test for phrases

Requirements:
  pip install psycopg[binary] numpy scikit-learn sentence-transformers python-dotenv

Env (or CLI) for DB:
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Usage examples:
  python analyze_embedding_superposition.py --sample 2000 --k 10 --trials 8 --keep-fracs 0.6 0.4 0.2
  python analyze_embedding_superposition.py --with-model --phrases "tinnitus" "hearing loss" "tinnitus and hearing loss"
"""

import os, argparse, math, random, re, sys
from typing import List, Tuple, Optional, Sequence

import numpy as np
from psycopg import connect
from psycopg.rows import dict_row
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# DB helpers
# ----------------------------

def get_pg_kwargs():
    return dict(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "medrag"),
        user=os.getenv("PGUSER", os.getenv("USER")),
        password=os.getenv("PGPASSWORD") or None,
    )

_VEC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_pgvector_text(s: str) -> np.ndarray:
    # s looks like: "[0.123,0.456,...]"
    nums = _VEC_RE.findall(s or "")
    return np.array([float(x) for x in nums], dtype=np.float32)

def fetch_random_embeddings(n: int, table: str = "chunks") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (embeddings NxD, chunk_ids N, doc_ids N).
    We fetch embedding::text and parse to floats for portability.
    """
    PG = get_pg_kwargs()
    with connect(**PG, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(f"""
            SELECT chunk_id, doc_id, embedding::text AS embtxt
            FROM {table}
            ORDER BY random()
            LIMIT %s;
        """, (n,))
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No rows returned from chunks. Are there embeddings loaded?")

    embs = []
    cids = []
    dids = []
    for r in rows:
        cids.append(r["chunk_id"])
        dids.append(r["doc_id"])
        embs.append(parse_pgvector_text(r["embtxt"]))
    X = np.vstack(embs).astype(np.float32)

    # normalize (your pipeline usually stores normalized vectors already; we normalize again to be safe)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X, np.array(cids), np.array(dids)

# ----------------------------
# Analyses
# ----------------------------

def pca_energy_report(X: np.ndarray, top: int = 20) -> None:
    print("\n=== PCA energy concentration ===")
    p = min(top, X.shape[1])
    pca = PCA(n_components=p, svd_solver="auto", random_state=0)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    for i, (vr, cv) in enumerate(zip(evr, cum), start=1):
        print(f"PC{i:02d}: {vr:7.4f}  |  cumulative: {cv:7.4f}")
    # simple markers
    k80 = (cum >= 0.80).argmax() + 1
    k90 = (cum >= 0.90).argmax() + 1
    print(f"\nVariance to reach 80%: {k80} PCs, 90%: {k90} PCs (out of {X.shape[1]})")
    print("Note: Heavier concentration in very few PCs may indicate entangled/global features.")

def topk_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    """
    Compute top-k neighbor indices (excluding self) using cosine sim.
    Returns (N x k) indices.
    """
    S = cosine_similarity(X, X)  # NxN
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=range(k), axis=1)[:, :k]   # fast top-k, unordered
    # order within top-k by similarity
    row_idx = np.arange(X.shape[0])[:, None]
    sims = S[row_idx, idx]
    order = np.argsort(-sims, axis=1)
    idx_sorted = idx[row_idx, order]
    return idx_sorted

def random_subspace_mask(d: int, keep_frac: float) -> np.ndarray:
    keep = max(1, int(round(d * keep_frac)))
    mask = np.zeros(d, dtype=bool)
    mask[np.random.choice(d, size=keep, replace=False)] = True
    return mask

def neighbor_stability_under_ablation(X: np.ndarray, k: int = 10, trials: int = 5, keep_fracs: Sequence[float] = (0.6, 0.4, 0.2)) -> None:
    """
    For each keep_frac, zero-out random dims and recompute neighbors. Report overlap@k vs baseline.
    """
    print("\n=== Neighbor stability under random subspace ablations ===")
    base_nn = topk_neighbors(X, k)
    d = X.shape[1]

    for f in keep_fracs:
        overlaps = []
        for _ in range(trials):
            mask = random_subspace_mask(d, f)
            Xp = X.copy()
            Xp[:, ~mask] = 0.0
            # re-normalize to avoid norm artifacts
            Xp = Xp / np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-9)

            nn_sub = topk_neighbors(Xp, k)
            # overlap@k
            inter = [len(set(base_nn[i]).intersection(nn_sub[i])) / k for i in range(X.shape[0])]
            overlaps.append(np.mean(inter))

        print(f" keep {int(f*100)}% dims → mean overlap@{k}: {np.mean(overlaps):.3f} ± {np.std(overlaps):.3f}")

    print("Lower overlaps for moderate ablations can indicate polysemantic / entangled features.")

# ----------------------------
# Optional: composition test (needs model)
# ----------------------------

def load_model(name: str):
    from sentence_transformers import SentenceTransformer
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if tok:
        return SentenceTransformer(name, token=tok)
    return SentenceTransformer(name)

def composition_test(model_name: str, phrases: List[str]) -> None:
    """
    Simple probe:
      A, B, A and B —> check normed sum vs model(A and B)
    """
    print("\n=== Composition test (optional) ===")
    model = load_model(model_name)
    vecs = model.encode(phrases, normalize_embeddings=True)
    for i, p in enumerate(phrases):
        print(f"  [{i}] {p!r}")

    if len(vecs) >= 3:
        A, B, AB = vecs[0], vecs[1], vecs[2]
        mix = A + B
        mix /= max(np.linalg.norm(mix), 1e-9)
        sim_mix_ab = float(np.dot(mix, AB))
        sim_a_ab   = float(np.dot(A, AB))
        sim_b_ab   = float(np.dot(B, AB))
        print(f"\n  cos(mix(A,B), AB) = {sim_mix_ab:.3f}")
        print(f"  cos(A, AB)        = {sim_a_ab:.3f}")
        print(f"  cos(B, AB)        = {sim_b_ab:.3f}")
        print("If mix(A,B) ≈ AB but A and B are not near each other, that’s a sign of feature superposition.")
    else:
        print("Provide at least 3 phrases: A, B, and 'A and B'.")

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500, help="number of random chunks to sample")
    ap.add_argument("--k", type=int, default=10, help="top-k for neighbor stability")
    ap.add_argument("--trials", type=int, default=6, help="random ablation trials per keep-frac")
    ap.add_argument("--keep-fracs", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2], help="fractions of dims to keep")
    ap.add_argument("--table", default="chunks", help="table with embedding column")
    ap.add_argument("--with-model", action="store_true", help="run phrase composition test")
    ap.add_argument("--model-name", default=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    ap.add_argument("--phrases", nargs="+", default=["tinnitus", "hearing loss", "tinnitus and hearing loss"])
    args = ap.parse_args()

    # Fetch embeddings
    X, cids, dids = fetch_random_embeddings(args.sample, table=args.table)
    print(f"Loaded {X.shape[0]} embeddings of dim {X.shape[1]} from '{args.table}'")

    # 1) PCA energy
    pca_energy_report(X, top=min(30, X.shape[1]))

    # 2) Neighbor stability
    neighbor_stability_under_ablation(
        X, k=args.k, trials=args.trials, keep_fracs=args.keep_fracs
    )

    # 3) Optional composition test
    if args.with_model:
        composition_test(args.model_name, args.phrases)

if __name__ == "__main__":
    main()