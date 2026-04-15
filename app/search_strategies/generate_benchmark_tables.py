#!/usr/bin/env python3
"""
Generate presentation-ready tables from benchmark results.

This script reads benchmark JSON files and generates formatted tables
showing latency and precision/recall metrics across different query counts.

Usage:
    python generate_benchmark_tables.py

Output:
    - Console output with formatted tables
    - CSV files in benchmark_tables/ directory
    - Excel file with all tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_benchmarks():
    """Load all benchmark files."""
    benchmark_files = {
        '30 queries': 'recall analysis/benchmark_results_30queries_recall_metrics.json',
        '50 queries': 'recall analysis/benchmark_50queries_recall_metrics.json',
        '100 queries': 'recall analysis/benchmark_100queries_recall_metrics.json'
    }

    benchmarks = {}
    for name, filepath in benchmark_files.items():
        try:
            with open(filepath, 'r') as f:
                benchmarks[name] = json.load(f)
            print(f"✓ Loaded: {name}")
        except FileNotFoundError:
            print(f"✗ Not found: {filepath}")

    return benchmarks


def extract_latency_metrics(benchmarks):
    """Extract latency metrics from benchmark results."""
    rows = []

    for benchmark_name, data in benchmarks.items():
        strategy_metrics = data.get('strategy_metrics', {})

        for strategy_key, metrics in strategy_metrics.items():
            row = {
                'Benchmark': benchmark_name,
                'Strategy': metrics.get('strategy', strategy_key),
                'Queries': metrics.get('num_queries', 0),
                'Avg Latency (ms)': round(metrics.get('avg_latency_ms', 0), 2),
                'Median Latency (ms)': round(metrics.get('median_latency_ms', 0), 2),
                'P95 Latency (ms)': round(metrics.get('p95_latency_ms', 0), 2),
                'P99 Latency (ms)': round(metrics.get('p99_latency_ms', 0), 2),
                'Std Dev (ms)': round(metrics.get('std_latency_ms', 0), 2),
                'Min Latency (ms)': round(metrics.get('min_latency_ms', 0), 2),
                'Max Latency (ms)': round(metrics.get('max_latency_ms', 0), 2)
            }
            rows.append(row)

    return pd.DataFrame(rows)


def extract_precision_recall_metrics(benchmarks):
    """Extract precision, recall, and F1 metrics from benchmark results."""
    rows = []

    strategy_map = {
        'ivfflat_only': 'IVFFlat Only',
        'ivfflat_ner_boost': 'IVFFlat + NER Boost',
        'ivfflat_tsvector': 'IVFFlat + tsvector',
        'ivfflat_ner_tsvector': 'IVFFlat + NER + tsvector'
    }

    for benchmark_name, data in benchmarks.items():
        recall_metrics = data.get('recall_metrics', {})

        for strategy_key, queries in recall_metrics.items():
            if not isinstance(queries, list):
                continue

            # Calculate average metrics across all queries
            precisions = [q.get('precision', 0) for q in queries if isinstance(q, dict)]
            recalls = [q.get('recall', 0) for q in queries if isinstance(q, dict)]
            f1s = [q.get('f1', 0) for q in queries if isinstance(q, dict)]

            if precisions:  # Only add if we have data
                row = {
                    'Benchmark': benchmark_name,
                    'Strategy': strategy_map.get(strategy_key, strategy_key),
                    'Queries': len(queries),
                    'Avg Precision': round(np.mean(precisions), 4),
                    'Avg Recall': round(np.mean(recalls), 4),
                    'Avg F1': round(np.mean(f1s), 4),
                    'Min Precision': round(np.min(precisions), 4),
                    'Max Precision': round(np.max(precisions), 4),
                    'Min Recall': round(np.min(recalls), 4),
                    'Max Recall': round(np.max(recalls), 4)
                }
                rows.append(row)

    return pd.DataFrame(rows)


def print_table(df, title):
    """Print formatted table."""
    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)
    print(df.to_string(index=False))
    print()


def create_pivot_tables(latency_df, precision_recall_df):
    """Create pivot tables for easy comparison."""
    tables = {}

    # Latency pivots
    tables['avg_latency'] = latency_df.pivot_table(
        index='Strategy', columns='Benchmark', values='Avg Latency (ms)', aggfunc='first'
    )

    tables['median_latency'] = latency_df.pivot_table(
        index='Strategy', columns='Benchmark', values='Median Latency (ms)', aggfunc='first'
    )

    tables['p95_latency'] = latency_df.pivot_table(
        index='Strategy', columns='Benchmark', values='P95 Latency (ms)', aggfunc='first'
    )

    # Precision/Recall pivots
    if not precision_recall_df.empty:
        tables['precision'] = precision_recall_df.pivot_table(
            index='Strategy', columns='Benchmark', values='Avg Precision', aggfunc='first'
        )

        tables['recall'] = precision_recall_df.pivot_table(
            index='Strategy', columns='Benchmark', values='Avg Recall', aggfunc='first'
        )

        tables['f1'] = precision_recall_df.pivot_table(
            index='Strategy', columns='Benchmark', values='Avg F1', aggfunc='first'
        )

    return tables


def export_tables(latency_df, precision_recall_df, pivot_tables):
    """Export tables to CSV and Excel files."""
    output_dir = Path('benchmark_tables')
    output_dir.mkdir(exist_ok=True)

    # Export pivot tables to CSV
    for name, table in pivot_tables.items():
        csv_path = output_dir / f"{name}_comparison.csv"
        table.to_csv(csv_path)
        print(f"✓ Exported: {csv_path}")

    # Create comprehensive summary
    summary_df = latency_df.merge(
        precision_recall_df,
        on=['Benchmark', 'Strategy', 'Queries'],
        how='outer'
    )

    summary_cols = [
        'Benchmark', 'Strategy', 'Queries',
        'Avg Latency (ms)', 'Median Latency (ms)', 'P95 Latency (ms)',
        'Avg Precision', 'Avg Recall', 'Avg F1'
    ]

    summary_table = summary_df[summary_cols].sort_values(['Benchmark', 'Strategy'])

    # Export summary CSV
    summary_csv = output_dir / 'comprehensive_summary.csv'
    summary_table.to_csv(summary_csv, index=False)
    print(f"✓ Exported: {summary_csv}")

    # Export to Excel with multiple sheets
    excel_path = output_dir / f'benchmark_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_table.to_excel(writer, sheet_name='Summary', index=False)
            pivot_tables['avg_latency'].to_excel(writer, sheet_name='Avg Latency')
            pivot_tables['median_latency'].to_excel(writer, sheet_name='Median Latency')
            pivot_tables['p95_latency'].to_excel(writer, sheet_name='P95 Latency')

            if 'precision' in pivot_tables:
                pivot_tables['precision'].to_excel(writer, sheet_name='Precision')
                pivot_tables['recall'].to_excel(writer, sheet_name='Recall')
                pivot_tables['f1'].to_excel(writer, sheet_name='F1 Score')

            latency_df.to_excel(writer, sheet_name='Raw Latency Data', index=False)

            if not precision_recall_df.empty:
                precision_recall_df.to_excel(writer, sheet_name='Raw Precision-Recall', index=False)

        print(f"✓ Exported: {excel_path}")
    except ImportError:
        print("⚠ openpyxl not installed. Skipping Excel export.")
        print("  Install with: pip install openpyxl")

    return summary_table


def print_key_findings(latency_df, precision_recall_df):
    """Print key findings summary."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Best latency strategy
    best_latency = latency_df.loc[latency_df['Avg Latency (ms)'].idxmin()]
    print(f"🏆 FASTEST STRATEGY:")
    print(f"   {best_latency['Strategy']} ({best_latency['Benchmark']})")
    print(f"   Average Latency: {best_latency['Avg Latency (ms)']:.2f} ms")
    print()

    if not precision_recall_df.empty:
        # Best precision strategy
        best_precision = precision_recall_df.loc[precision_recall_df['Avg Precision'].idxmax()]
        print(f"🎯 HIGHEST PRECISION:")
        print(f"   {best_precision['Strategy']} ({best_precision['Benchmark']})")
        print(f"   Average Precision: {best_precision['Avg Precision']:.4f}")
        print()

        # Best recall strategy
        best_recall = precision_recall_df.loc[precision_recall_df['Avg Recall'].idxmax()]
        print(f"🔍 HIGHEST RECALL:")
        print(f"   {best_recall['Strategy']} ({best_recall['Benchmark']})")
        print(f"   Average Recall: {best_recall['Avg Recall']:.4f}")
        print()

        # Best F1 strategy
        best_f1 = precision_recall_df.loc[precision_recall_df['Avg F1'].idxmax()]
        print(f"⚖️  BEST F1 SCORE:")
        print(f"   {best_f1['Strategy']} ({best_f1['Benchmark']})")
        print(f"   Average F1: {best_f1['Avg F1']:.4f}")
        print()

    # Latency trends
    print(f"📈 LATENCY TRENDS:")
    for strategy in latency_df['Strategy'].unique():
        strategy_data = latency_df[latency_df['Strategy'] == strategy].sort_values('Queries')
        if len(strategy_data) > 1:
            latencies = strategy_data['Avg Latency (ms)'].values
            trend = "increasing" if latencies[-1] > latencies[0] else "decreasing"
            change = latencies[-1] - latencies[0]
            print(f"   {strategy}: {trend} ({latencies[0]:.2f} → {latencies[-1]:.2f} ms, Δ{change:+.2f})")

    print()
    print("=" * 80)


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Medical RAG - Benchmark Table Generator")
    print("=" * 80)
    print()

    # Load data
    print("Loading benchmark data...")
    benchmarks = load_benchmarks()

    if not benchmarks:
        print("\n❌ No benchmark files found!")
        print("   Make sure you're running this from app/search_strategies/")
        return

    print(f"\n✅ Loaded {len(benchmarks)} benchmark files")

    # Extract metrics
    print("\nExtracting metrics...")
    latency_df = extract_latency_metrics(benchmarks)
    precision_recall_df = extract_precision_recall_metrics(benchmarks)

    # Create pivot tables
    print("Creating comparison tables...")
    pivot_tables = create_pivot_tables(latency_df, precision_recall_df)

    # Print tables
    print_table(pivot_tables['avg_latency'], "TABLE 1: Average Latency (ms) by Strategy and Query Count")
    print_table(pivot_tables['median_latency'], "TABLE 2: Median Latency (ms) by Strategy and Query Count")
    print_table(pivot_tables['p95_latency'], "TABLE 3: P95 Latency (ms) by Strategy and Query Count")

    if 'precision' in pivot_tables:
        print_table(pivot_tables['precision'], "TABLE 4: Average Precision by Strategy and Query Count")
        print_table(pivot_tables['recall'], "TABLE 5: Average Recall by Strategy and Query Count")
        print_table(pivot_tables['f1'], "TABLE 6: Average F1 Score by Strategy and Query Count")

    # Export tables
    print("\nExporting tables...")
    summary_table = export_tables(latency_df, precision_recall_df, pivot_tables)

    # Print key findings
    print_key_findings(latency_df, precision_recall_df)

    # Print comprehensive summary
    print_table(summary_table, "COMPREHENSIVE SUMMARY: All Metrics by Strategy and Query Count")

    print("\n✅ Done! Check the 'benchmark_tables/' directory for exported files.")
    print()


if __name__ == "__main__":
    main()
