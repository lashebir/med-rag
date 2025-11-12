#!/usr/bin/env python3
"""
IVFFlat Index Benchmark Script

Tests different IVFFlat configurations for vector similarity search:
- nlist (number of inverted lists/clusters): [32, 64, 96, 128]
- nprobe (number of lists to probe): [1, 2, 4, 8, 16]

Measures:
- Index build time
- Query latency
- Recall@K (compared to exact search baseline)

Usage:
    python app/benchmark_ivfflat.py
    python app/benchmark_ivfflat.py --test-queries 100 --top-k 10
"""

import os
import time
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from statistics import mean, median, stdev

from dotenv import load_dotenv
from psycopg import connect
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

# Configuration
PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Test configurations
NLIST_VALUES = [32, 64, 96, 128]
NPROBE_VALUES = [1, 2, 4, 8, 16]

# Embedder singleton
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def to_vec_lit(vec) -> str:
    """Convert vector to PostgreSQL literal."""
    # Handle different input types
    if isinstance(vec, str):
        # Already a string, might be a vector literal like '[1,2,3]'
        # Remove brackets and parse
        vec = vec.strip('[]').split(',')
        vec = [float(x) for x in vec]
    elif hasattr(vec, 'tolist'):
        # NumPy array or similar
        vec = vec.tolist()
    elif not isinstance(vec, (list, tuple)):
        # Unknown type, try to convert
        vec = list(vec)

    # Format as PostgreSQL vector literal
    return '[' + ','.join(f"{float(x):.6f}" for x in vec) + ']'

def get_total_chunks() -> int:
    """Get total number of chunks in database."""
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding_model = %s;", (EMBED_MODEL,))
        return cur.fetchone()[0]

def get_sample_queries(n: int = 100) -> List[Tuple[str, np.ndarray]]:
    """
    Get sample queries by generating embeddings from actual chunk texts.
    Returns list of (query_text, query_embedding) tuples.
    """
    print(f"Generating {n} sample queries from database...")

    # Get embedder
    embedder = get_embedder()

    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Get random chunks to use as queries
        cur.execute("""
            SELECT text
            FROM chunks
            WHERE embedding_model = %s
            AND text IS NOT NULL
            AND LENGTH(text) > 100
            ORDER BY RANDOM()
            LIMIT %s;
        """, (EMBED_MODEL, n))

        rows = cur.fetchall()

    # Generate embeddings for query texts
    query_texts = [row[0][:200] + "..." for row in rows]
    query_embeddings = embedder.encode(query_texts, normalize_embeddings=True, batch_size=32)

    queries = list(zip(query_texts, query_embeddings))

    print(f"Generated {len(queries)} sample queries")
    return queries

def exact_search(query_vec: Any, k: int = 10) -> List[int]:
    """
    Perform exact nearest neighbor search (no index).
    Returns list of chunk_id values.
    """
    qlit = to_vec_lit(query_vec)
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Force sequential scan by not using index
        cur.execute("SET LOCAL enable_indexscan = off;")
        cur.execute("""
            SELECT c.chunk_id
            FROM chunks c
            WHERE c.embedding_model = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (EMBED_MODEL, qlit, k))

        return [row[0] for row in cur.fetchall()]

def indexed_search(query_vec: Any, k: int = 10, nprobe: int = 1) -> Tuple[List[int], float]:
    """
    Perform indexed search with specified nprobe.
    Returns (list of chunk_ids, query_time_ms).
    """
    qlit = to_vec_lit(query_vec)

    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Set nprobe for this query
        cur.execute(f"SET LOCAL ivfflat.probes = {nprobe};")

        start = time.time()
        cur.execute("""
            SELECT c.chunk_id
            FROM chunks c
            WHERE c.embedding_model = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """, (EMBED_MODEL, qlit, k))

        elapsed_ms = (time.time() - start) * 1000
        results = [row[0] for row in cur.fetchall()]

    return results, elapsed_ms

def calculate_recall(exact_ids: List[int], approx_ids: List[int]) -> float:
    """Calculate recall@K: what fraction of exact results are in approximate results."""
    if not exact_ids:
        return 0.0
    exact_set = set(exact_ids)
    approx_set = set(approx_ids)
    overlap = len(exact_set & approx_set)
    return overlap / len(exact_set)

def drop_existing_indexes():
    """Drop all existing IVFFlat indexes on chunks.embedding."""
    print("\n🗑️  Dropping existing IVFFlat indexes...")
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Find all ivfflat indexes on chunks.embedding
        cur.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'chunks'
            AND indexdef LIKE '%ivfflat%'
            AND indexdef LIKE '%embedding%';
        """)

        indexes = [row[0] for row in cur.fetchall()]

        for idx_name in indexes:
            print(f"   Dropping index: {idx_name}")
            cur.execute(f"DROP INDEX IF EXISTS {idx_name};")

        con.commit()
        print(f"   Dropped {len(indexes)} index(es)")

def build_ivfflat_index(nlist: int) -> float:
    """
    Build IVFFlat index with specified nlist value.
    Returns index build time in seconds.
    """
    index_name = f"idx_chunks_embedding_ivfflat_nlist{nlist}"

    print(f"\n🔨 Building IVFFlat index with nlist={nlist}...")
    print(f"   Index name: {index_name}")

    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Drop if exists
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")

        # Build index
        start = time.time()
        cur.execute(f"""
            CREATE INDEX {index_name}
            ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {nlist})
            WHERE embedding_model = '{EMBED_MODEL}';
        """)
        con.commit()
        elapsed = time.time() - start

    print(f"   ✅ Built in {elapsed:.2f} seconds")
    return elapsed

def benchmark_configuration(
    nlist: int,
    nprobe: int,
    queries: List[Tuple[str, np.ndarray]],
    exact_results: List[List[int]],
    k: int = 10
) -> Dict[str, Any]:
    """
    Benchmark a specific (nlist, nprobe) configuration.
    Returns metrics dictionary.
    """
    print(f"\n📊 Testing nlist={nlist}, nprobe={nprobe}")

    query_times = []
    recalls = []

    for i, ((_, query_vec), exact_ids) in enumerate(zip(queries, exact_results)):
        approx_ids, query_time = indexed_search(query_vec, k=k, nprobe=nprobe)
        recall = calculate_recall(exact_ids, approx_ids)

        query_times.append(query_time)
        recalls.append(recall)

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{len(queries)} queries")

    metrics = {
        "nlist": nlist,
        "nprobe": nprobe,
        "num_queries": len(queries),
        "avg_query_time_ms": mean(query_times),
        "median_query_time_ms": median(query_times),
        "std_query_time_ms": stdev(query_times) if len(query_times) > 1 else 0,
        "avg_recall": mean(recalls),
        "median_recall": median(recalls),
        "min_recall": min(recalls),
        "max_recall": max(recalls),
    }

    print(f"   Avg Query Time: {metrics['avg_query_time_ms']:.2f}ms")
    print(f"   Avg Recall@{k}: {metrics['avg_recall']:.3f}")

    return metrics

def run_benchmark(num_queries: int = 100, top_k: int = 10) -> Dict[str, Any]:
    """
    Run full benchmark suite.
    Returns results dictionary.
    """
    print("=" * 80)
    print("IVFFlat Index Benchmark")
    print("=" * 80)

    # Get database stats
    total_chunks = get_total_chunks()
    print(f"\n📈 Database stats:")
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   Embedding model: {EMBED_MODEL}")
    print(f"   Top-K: {top_k}")

    # Generate sample queries
    queries = get_sample_queries(num_queries)

    # Get exact search baseline
    print(f"\n🎯 Computing exact search baseline ({num_queries} queries)...")
    exact_results = []
    exact_times = []

    for i, (_, query_vec) in enumerate(queries):
        start = time.time()
        exact_ids = exact_search(query_vec, k=top_k)
        elapsed = (time.time() - start) * 1000
        exact_results.append(exact_ids)
        exact_times.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{num_queries} queries")

    avg_exact_time = mean(exact_times)
    print(f"   ✅ Exact search avg time: {avg_exact_time:.2f}ms")

    # Drop existing indexes
    drop_existing_indexes()

    # Run benchmarks for each configuration
    all_results = []

    for nlist in NLIST_VALUES:
        # Build index with this nlist
        build_time = build_ivfflat_index(nlist)

        # Test with different nprobe values
        for nprobe in NPROBE_VALUES:
            if nprobe > nlist:
                # Skip invalid configurations where nprobe > nlist
                print(f"\n⚠️  Skipping nprobe={nprobe} (> nlist={nlist})")
                continue

            metrics = benchmark_configuration(
                nlist=nlist,
                nprobe=nprobe,
                queries=queries,
                exact_results=exact_results,
                k=top_k
            )
            metrics["build_time_sec"] = build_time
            metrics["exact_search_avg_ms"] = avg_exact_time
            all_results.append(metrics)

    # Summary
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_queries": num_queries,
            "top_k": top_k,
            "total_chunks": total_chunks,
            "embedding_model": EMBED_MODEL,
            "nlist_values": NLIST_VALUES,
            "nprobe_values": NPROBE_VALUES,
        },
        "exact_search_avg_ms": avg_exact_time,
        "results": all_results
    }

    return results

def print_summary_table(results: Dict[str, Any]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Table header
    print(f"\n{'nlist':<8} {'nprobe':<8} {'Recall@K':<12} {'Query(ms)':<12} {'vs Exact':<12} {'Build(s)':<10}")
    print("-" * 80)

    exact_time = results["exact_search_avg_ms"]

    for r in results["results"]:
        speedup = exact_time / r["avg_query_time_ms"]
        print(
            f"{r['nlist']:<8} {r['nprobe']:<8} "
            f"{r['avg_recall']:<12.3f} "
            f"{r['avg_query_time_ms']:<12.2f} "
            f"{speedup:<12.1f}x "
            f"{r['build_time_sec']:<10.2f}"
        )

    # Best configurations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Best recall
    best_recall = max(results["results"], key=lambda x: x["avg_recall"])
    print(f"\n🎯 Best Recall: nlist={best_recall['nlist']}, nprobe={best_recall['nprobe']}")
    print(f"   Recall@{results['config']['top_k']}: {best_recall['avg_recall']:.3f}")
    print(f"   Query Time: {best_recall['avg_query_time_ms']:.2f}ms")

    # Best speed
    best_speed = min(results["results"], key=lambda x: x["avg_query_time_ms"])
    print(f"\n⚡ Fastest: nlist={best_speed['nlist']}, nprobe={best_speed['nprobe']}")
    print(f"   Query Time: {best_speed['avg_query_time_ms']:.2f}ms")
    print(f"   Recall@{results['config']['top_k']}: {best_speed['avg_recall']:.3f}")

    # Balanced (good recall + reasonable speed)
    balanced = max(
        [r for r in results["results"] if r["avg_recall"] >= 0.9],
        key=lambda x: -x["avg_query_time_ms"],
        default=None
    )
    if balanced:
        print(f"\n⚖️  Balanced (Recall≥0.9): nlist={balanced['nlist']}, nprobe={balanced['nprobe']}")
        print(f"   Recall@{results['config']['top_k']}: {balanced['avg_recall']:.3f}")
        print(f"   Query Time: {balanced['avg_query_time_ms']:.2f}ms")

def save_results(results: Dict[str, Any], output_file: str = "ivfflat_benchmark_results.json"):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark IVFFlat index configurations")
    parser.add_argument(
        "--test-queries",
        type=int,
        default=100,
        help="Number of test queries (default: 100)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K results to retrieve (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ivfflat_benchmark_results.json",
        help="Output JSON file (default: ivfflat_benchmark_results.json)"
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(num_queries=args.test_queries, top_k=args.top_k)

    # Print summary
    print_summary_table(results)

    # Save results
    save_results(results, output_file=args.output)

    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()
