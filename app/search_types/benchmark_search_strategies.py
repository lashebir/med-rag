#!/usr/bin/env python3
"""
Search Strategy Benchmark Script

Compares four retrieval strategies:
1. IVFFlat Only (baseline)
2. IVFFlat + NER Soft Boost
3. IVFFlat + tsvector (hybrid)
4. IVFFlat + NER + tsvector (full hybrid)

Metrics:
- Query latency (avg, median, p95, p99)
- Result overlap (how similar are the results?)
- Diversity scores
- Top-k precision (if ground truth available)

Usage:
    python app/benchmark_search_strategies.py
    python app/benchmark_search_strategies.py --test-queries 50 --top-k 10
    python app/benchmark_search_strategies.py --output results.json
"""

import os
import time
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from statistics import mean, median, stdev
from collections import Counter

from dotenv import load_dotenv
from psycopg import connect
from sentence_transformers import SentenceTransformer

# Import our search strategies
import sys
sys.path.insert(0, os.path.dirname(__file__))
from search_types.search_strategies import (
    retrieve_ivfflat_only,
    retrieve_ivfflat_ner_boost,
    retrieve_ivfflat_tsvector,
    retrieve_ivfflat_ner_tsvector,
)

load_dotenv()

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Embedder singleton
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_sample_queries(n: int = 50) -> List[str]:
    """
    Generate sample queries from actual document chunks.

    Returns list of query strings (shortened chunk texts).
    """
    print(f"Generating {n} sample queries from database...")

    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Get random chunks to use as queries
        # Prefer chunks with mentions (more interesting for NER boost)
        cur.execute("""
            SELECT DISTINCT c.text
            FROM chunks c
            WHERE c.embedding_model = %s
            AND c.text IS NOT NULL
            AND LENGTH(c.text) > 100
            AND EXISTS (
                SELECT 1 FROM mentions m
                WHERE m.doc_id = c.doc_id
            )
            ORDER BY RANDOM()
            LIMIT %s;
        """, (EMBED_MODEL, n * 2))  # Get 2x and filter

        rows = cur.fetchall()

    # Create query strings (first ~100 chars of chunk)
    queries = []
    for row in rows[:n]:
        text = row[0]
        # Extract first sentence or first 100 chars
        query = text[:150].split('.')[0] + '.'
        query = query.strip()
        if len(query) > 20:  # Minimum query length
            queries.append(query)

    print(f"Generated {len(queries)} sample queries")
    return queries


def calculate_overlap(results1: List[Dict], results2: List[Dict], key: str = "chunk_id") -> float:
    """
    Calculate overlap between two result lists.

    Returns: fraction of common items (Jaccard similarity)
    """
    set1 = set(r[key] for r in results1 if key in r)
    set2 = set(r[key] for r in results2 if key in r)

    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def calculate_rank_correlation(results1: List[Dict], results2: List[Dict], key: str = "chunk_id") -> float:
    """
    Calculate Spearman rank correlation between two result lists.

    Returns: correlation coefficient (-1 to 1)
    """
    # Get common items
    ids1 = [r[key] for r in results1 if key in r]
    ids2 = [r[key] for r in results2 if key in r]

    common = set(ids1) & set(ids2)
    if len(common) < 2:
        return 0.0

    # Rank dictionaries
    rank1 = {id_: i for i, id_ in enumerate(ids1)}
    rank2 = {id_: i for i, id_ in enumerate(ids2)}

    # Calculate Spearman correlation for common items
    ranks1 = [rank1[id_] for id_ in common]
    ranks2 = [rank2[id_] for id_ in common]

    # Simple correlation (could use scipy for more accurate)
    n = len(ranks1)
    sum_d_sq = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))
    rho = 1 - (6 * sum_d_sq) / (n * (n**2 - 1)) if n > 1 else 0.0

    return rho


def benchmark_strategy(
    strategy_name: str,
    strategy_func,
    queries: List[str],
    k: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a single search strategy.

    Returns:
        Dictionary with latency metrics and all results
    """
    print(f"\n📊 Benchmarking: {strategy_name}")

    latencies = []
    all_results = []

    for i, query in enumerate(queries):
        start = time.time()

        try:
            results = strategy_func(query, k=k, **kwargs)
            elapsed_ms = (time.time() - start) * 1000

            latencies.append(elapsed_ms)
            all_results.append(results)

        except Exception as e:
            print(f"   ⚠️  Query {i+1} failed: {e}")
            latencies.append(0)
            all_results.append([])

        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(queries)} queries")

    # Calculate percentiles
    sorted_latencies = sorted([l for l in latencies if l > 0])
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95 = sorted_latencies[p95_idx] if sorted_latencies and p95_idx < len(sorted_latencies) else 0
    p99_idx = int(len(sorted_latencies) * 0.99)
    p99 = sorted_latencies[p99_idx] if sorted_latencies and p99_idx < len(sorted_latencies) else 0

    metrics = {
        "strategy": strategy_name,
        "num_queries": len(queries),
        "avg_latency_ms": mean(latencies) if latencies else 0,
        "median_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "std_latency_ms": stdev(latencies) if len(latencies) > 1 else 0,
        "min_latency_ms": min(latencies) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "results": all_results,
    }

    print(f"   Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"   P95 Latency: {metrics['p95_latency_ms']:.2f}ms")

    return metrics


def compare_strategies(
    baseline_results: List[List[Dict]],
    strategy_results: List[List[Dict]],
    strategy_name: str
) -> Dict[str, Any]:
    """
    Compare a strategy against baseline.

    Returns:
        Dictionary with overlap and correlation metrics
    """
    overlaps = []
    correlations = []

    for base, strat in zip(baseline_results, strategy_results):
        if base and strat:
            overlap = calculate_overlap(base, strat)
            corr = calculate_rank_correlation(base, strat)
            overlaps.append(overlap)
            correlations.append(corr)

    return {
        "comparison": f"baseline vs {strategy_name}",
        "avg_overlap": mean(overlaps) if overlaps else 0,
        "avg_rank_correlation": mean(correlations) if correlations else 0,
        "min_overlap": min(overlaps) if overlaps else 0,
        "max_overlap": max(overlaps) if overlaps else 0,
    }


def run_benchmark(num_queries: int = 50, top_k: int = 10) -> Dict[str, Any]:
    """
    Run full benchmark comparing all strategies.

    Returns:
        Results dictionary
    """
    print("=" * 80)
    print("Search Strategy Benchmark")
    print("=" * 80)

    # Generate queries
    queries = get_sample_queries(num_queries)

    print(f"\n📈 Configuration:")
    print(f"   Queries: {len(queries)}")
    print(f"   Top-K: {top_k}")
    print(f"   Embedding model: {EMBED_MODEL}")

    # Benchmark each strategy
    results = {}

    # Strategy 1: IVFFlat Only (Baseline)
    results["ivfflat_only"] = benchmark_strategy(
        "IVFFlat Only",
        retrieve_ivfflat_only,
        queries,
        k=top_k
    )

    # Strategy 2: IVFFlat + NER Boost
    results["ivfflat_ner_boost"] = benchmark_strategy(
        "IVFFlat + NER Boost",
        retrieve_ivfflat_ner_boost,
        queries,
        k=top_k,
        ner_boost_weight=0.15
    )

    # Strategy 3: IVFFlat + tsvector
    results["ivfflat_tsvector"] = benchmark_strategy(
        "IVFFlat + tsvector",
        retrieve_ivfflat_tsvector,
        queries,
        k=top_k,
        use_rrf=True
    )

    # Strategy 4: IVFFlat + NER + tsvector (Full Hybrid)
    results["ivfflat_ner_tsvector"] = benchmark_strategy(
        "IVFFlat + NER + tsvector",
        retrieve_ivfflat_ner_tsvector,
        queries,
        k=top_k,
        vector_weight=0.5,
        text_weight=0.3,
        ner_weight=0.2
    )

    # Compare strategies
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISONS")
    print("=" * 80)

    baseline_results = results["ivfflat_only"]["results"]

    comparisons = {}

    # Compare NER boost vs baseline
    ner_comparison = compare_strategies(
        baseline_results,
        results["ivfflat_ner_boost"]["results"],
        "IVFFlat + NER Boost"
    )
    comparisons["ner_vs_baseline"] = ner_comparison

    print(f"\nIVFFlat + NER Boost vs Baseline:")
    print(f"   Avg Overlap: {ner_comparison['avg_overlap']:.3f}")
    print(f"   Avg Rank Correlation: {ner_comparison['avg_rank_correlation']:.3f}")

    # Compare tsvector vs baseline
    tsv_comparison = compare_strategies(
        baseline_results,
        results["ivfflat_tsvector"]["results"],
        "IVFFlat + tsvector"
    )
    comparisons["tsvector_vs_baseline"] = tsv_comparison

    print(f"\nIVFFlat + tsvector vs Baseline:")
    print(f"   Avg Overlap: {tsv_comparison['avg_overlap']:.3f}")
    print(f"   Avg Rank Correlation: {tsv_comparison['avg_rank_correlation']:.3f}")

    # Compare NER boost vs tsvector
    ner_tsv_comparison = compare_strategies(
        results["ivfflat_ner_boost"]["results"],
        results["ivfflat_tsvector"]["results"],
        "IVFFlat + tsvector"
    )
    comparisons["ner_vs_tsvector"] = ner_tsv_comparison

    print(f"\nIVFFlat + NER Boost vs IVFFlat + tsvector:")
    print(f"   Avg Overlap: {ner_tsv_comparison['avg_overlap']:.3f}")
    print(f"   Avg Rank Correlation: {ner_tsv_comparison['avg_rank_correlation']:.3f}")

    # Combine results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_queries": len(queries),
            "top_k": top_k,
            "embedding_model": EMBED_MODEL,
            "queries": queries,
        },
        "strategy_metrics": {
            "ivfflat_only": {k: v for k, v in results["ivfflat_only"].items() if k != "results"},
            "ivfflat_ner_boost": {k: v for k, v in results["ivfflat_ner_boost"].items() if k != "results"},
            "ivfflat_tsvector": {k: v for k, v in results["ivfflat_tsvector"].items() if k != "results"},
            "ivfflat_ner_tsvector": {k: v for k, v in results["ivfflat_ner_tsvector"].items() if k != "results"},
        },
        "comparisons": comparisons,
        "detailed_results": {
            "ivfflat_only": results["ivfflat_only"]["results"],
            "ivfflat_ner_boost": results["ivfflat_ner_boost"]["results"],
            "ivfflat_tsvector": results["ivfflat_tsvector"]["results"],
            "ivfflat_ner_tsvector": results["ivfflat_ner_tsvector"]["results"],
        }
    }

    return summary


def print_summary_table(results: Dict[str, Any]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    metrics = results["strategy_metrics"]

    # Table header
    print(f"\n{'Strategy':<25} {'Avg(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12} {'P99(ms)':<12}")
    print("-" * 80)

    for strategy_key, strategy_name in [
        ("ivfflat_only", "IVFFlat Only"),
        ("ivfflat_ner_boost", "IVFFlat + NER Boost"),
        ("ivfflat_tsvector", "IVFFlat + tsvector"),
        ("ivfflat_ner_tsvector", "IVFFlat + NER + tsvector"),
    ]:
        m = metrics[strategy_key]
        print(
            f"{strategy_name:<25} "
            f"{m['avg_latency_ms']:<12.2f} "
            f"{m['median_latency_ms']:<12.2f} "
            f"{m['p95_latency_ms']:<12.2f} "
            f"{m['p99_latency_ms']:<12.2f}"
        )

    # Comparisons
    print("\n" + "=" * 80)
    print("RESULT SIMILARITY")
    print("=" * 80)

    comps = results["comparisons"]

    print(f"\n{'Comparison':<40} {'Overlap':<12} {'Rank Corr':<12}")
    print("-" * 80)

    for comp_key, comp_name in [
        ("ner_vs_baseline", "NER Boost vs Baseline"),
        ("tsvector_vs_baseline", "tsvector vs Baseline"),
        ("ner_vs_tsvector", "NER Boost vs tsvector"),
    ]:
        c = comps[comp_key]
        print(
            f"{comp_name:<40} "
            f"{c['avg_overlap']:<12.3f} "
            f"{c['avg_rank_correlation']:<12.3f}"
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("\nLatency:")
    fastest = min(metrics.items(), key=lambda x: x[1]["avg_latency_ms"])
    print(f"   Fastest: {fastest[0]} ({fastest[1]['avg_latency_ms']:.2f}ms)")

    print("\nResult Diversity:")
    print(f"   Lower overlap = more diverse results")
    print(f"   Higher overlap = more similar results to baseline")


def save_results(results: Dict[str, Any], output_file: str = "search_strategy_benchmark.json"):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark search strategy configurations")
    parser.add_argument(
        "--test-queries",
        type=int,
        default=50,
        help="Number of test queries (default: 50)"
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
        default="search_strategy_benchmark.json",
        help="Output JSON file (default: search_strategy_benchmark.json)"
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
