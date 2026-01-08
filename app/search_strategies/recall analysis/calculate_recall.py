#!/usr/bin/env python3
"""
Calculate precision and recall metrics for search strategies.

Recall is calculated as the percentage of ground truth results that were retrieved.
Precision is calculated as the percentage of retrieved results that are relevant.

Ground truth: We use the baseline IVFFlat results as the reference.
"""

import json
import sys
import os
from typing import List, Dict, Set, Any
from statistics import mean


def calculate_precision_recall(
    retrieved_results: List[Dict],
    ground_truth_results: List[Dict],
    key: str = "chunk_id"
) -> Dict[str, float]:
    """
    Calculate precision and recall for a single query.

    Args:
        retrieved_results: Results from the strategy being evaluated
        ground_truth_results: Ground truth results (e.g., from baseline)
        key: Key to use for comparison (default: chunk_id)

    Returns:
        Dictionary with precision, recall, and f1 score
    """
    retrieved_ids = set(r[key] for r in retrieved_results if key in r)
    ground_truth_ids = set(r[key] for r in ground_truth_results if key in r)

    if not retrieved_ids and not ground_truth_ids:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not ground_truth_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # True positives: items in both retrieved and ground truth
    true_positives = len(retrieved_ids & ground_truth_ids)

    # Precision: TP / (TP + FP) = TP / total_retrieved
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0

    # Recall: TP / (TP + FN) = TP / total_ground_truth
    recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0

    # F1 score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "retrieved_count": len(retrieved_ids),
        "ground_truth_count": len(ground_truth_ids)
    }


def analyze_strategy_metrics(
    strategy_results: List[List[Dict]],
    baseline_results: List[List[Dict]],
    strategy_name: str
) -> Dict[str, Any]:
    """
    Calculate average precision and recall across all queries.

    Args:
        strategy_results: List of result lists for the strategy
        baseline_results: List of result lists for baseline (ground truth)
        strategy_name: Name of the strategy

    Returns:
        Dictionary with average metrics
    """
    all_metrics = []

    for strategy_res, baseline_res in zip(strategy_results, baseline_results):
        if strategy_res and baseline_res:
            metrics = calculate_precision_recall(strategy_res, baseline_res)
            all_metrics.append(metrics)

    if not all_metrics:
        return {
            "strategy": strategy_name,
            "num_queries": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0
        }

    return {
        "strategy": strategy_name,
        "num_queries": len(all_metrics),
        "avg_precision": mean(m["precision"] for m in all_metrics),
        "avg_recall": mean(m["recall"] for m in all_metrics),
        "avg_f1": mean(m["f1"] for m in all_metrics),
        "min_recall": min(m["recall"] for m in all_metrics),
        "max_recall": max(m["recall"] for m in all_metrics),
        "min_precision": min(m["precision"] for m in all_metrics),
        "max_precision": max(m["precision"] for m in all_metrics),
        "per_query_metrics": all_metrics
    }


def main(benchmark_file: str):
    """Load benchmark results and calculate precision/recall."""

    print(f"Loading benchmark results from: {benchmark_file}")

    with open(benchmark_file, 'r') as f:
        data = json.load(f)

    # Extract detailed results
    detailed_results = data.get("detailed_results", {})

    if not detailed_results:
        print("ERROR: No detailed_results found in benchmark file")
        return

    # Use baseline (ivfflat_only) as ground truth
    baseline_results = detailed_results.get("ivfflat_only", [])

    print(f"\nGround truth: IVFFlat Only (baseline)")
    print(f"Number of queries: {len(baseline_results)}")
    print(f"Top-K: {data['config']['top_k']}")

    print("\n" + "=" * 80)
    print("PRECISION & RECALL ANALYSIS")
    print("=" * 80)

    # Analyze each strategy
    strategies = [
        ("ivfflat_ner_boost", "IVFFlat + NER Boost"),
        ("ivfflat_tsvector", "IVFFlat + tsvector"),
        ("ivfflat_ner_tsvector", "IVFFlat + NER + tsvector"),
    ]

    results = {}

    for strategy_key, strategy_name in strategies:
        strategy_res = detailed_results.get(strategy_key, [])

        if not strategy_res:
            print(f"\n⚠️  No results found for {strategy_name}")
            continue

        metrics = analyze_strategy_metrics(strategy_res, baseline_results, strategy_name)
        results[strategy_key] = metrics

        print(f"\n{strategy_name}:")
        print(f"   Average Precision: {metrics['avg_precision']:.4f}")
        print(f"   Average Recall:    {metrics['avg_recall']:.4f}")
        print(f"   Average F1 Score:  {metrics['avg_f1']:.4f}")
        print(f"   Recall Range:      {metrics['min_recall']:.4f} - {metrics['max_recall']:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Strategy':<30} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)

    for strategy_key, strategy_name in strategies:
        if strategy_key in results:
            m = results[strategy_key]
            print(
                f"{strategy_name:<30} "
                f"{m['avg_precision']:<12.4f} "
                f"{m['avg_recall']:<12.4f} "
                f"{m['avg_f1']:<12.4f}"
            )

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nMetrics (compared to baseline IVFFlat Only):")
    print("  - Precision: % of retrieved results that match baseline")
    print("  - Recall: % of baseline results that were retrieved")
    print("  - F1 Score: Harmonic mean of precision and recall")
    print("\nPerfect scores (1.0) mean the strategy retrieves exactly the same")
    print("results as the baseline.")
    print("\nLower recall means the strategy misses some baseline results")
    print("(may retrieve different but potentially better results).")

    # Save enhanced results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    basename = os.path.basename(benchmark_file).replace(".json", "_recall_metrics.json")
    output_file = os.path.join(output_dir, basename)

    data["precision_recall_metrics"] = results

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Enhanced results saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        benchmark_file = sys.argv[1]
    else:
        # Default to the 30 queries benchmark
        benchmark_file = "../benchmark_results_30queries.json"

    main(benchmark_file)
