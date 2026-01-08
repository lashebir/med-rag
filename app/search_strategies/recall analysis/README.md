# Recall Analysis Directory

This directory contains precision and recall analysis for the different search strategies.

## Files

1. **`calculate_recall.py`**
   - Python script to calculate precision, recall, and F1 scores
   - Compares strategies against baseline (IVFFlat Only)
   - Generates detailed metrics per query

2. **`benchmark_results_30queries_recall_metrics.json`**
   - Enhanced benchmark results with precision/recall metrics
   - Includes per-query breakdown
   - Full detailed results for all strategies

3. **`RECALL_ANALYSIS_SUMMARY.md`**
   - Comprehensive summary of findings
   - Interpretation and recommendations
   - Key insights and next steps

## Quick Start

### Run Analysis on Existing Benchmark

```bash
cd "/Users/leahashebir/Documents/med-rag/app/search_strategies/recall analysis"
python3 calculate_recall.py ../benchmark_results_30queries.json
```

### Run Analysis on New Benchmark

```bash
# First, run the benchmark
cd /Users/leahashebir/Documents/med-rag
python3 app/search_strategies/benchmark_search_strategies.py --test-queries 50 --output my_benchmark.json

# Then analyze it
cd "app/search_strategies/recall analysis"
python3 calculate_recall.py ../my_benchmark.json
```

## Key Results

**IVFFlat + tsvector Average Recall: 95.65%**

This means:
- Retrieves 95.65% of the same chunks as baseline
- Introduces 4.35% diversity through keyword matching
- Fastest strategy (29.7ms average latency)

See `RECALL_ANALYSIS_SUMMARY.md` for detailed interpretation.

## Metrics Explained

- **Precision**: % of retrieved results that match baseline
- **Recall**: % of baseline results that were retrieved
- **F1 Score**: Harmonic mean of precision and recall

A perfect score (1.0) means identical results to baseline.
Lower scores indicate diversity in retrieved results.

## Usage

The recall calculation script accepts any benchmark JSON file that contains:
- `detailed_results` with query-level results
- `ivfflat_only` baseline results
- Chunk IDs for comparison

Output includes:
- Console summary with tables
- JSON file with full metrics saved to this directory
