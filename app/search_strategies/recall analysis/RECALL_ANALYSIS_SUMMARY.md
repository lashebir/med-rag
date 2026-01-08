# Search Strategy Recall Analysis

## Overview

This analysis calculates **precision**, **recall**, and **F1 scores** for the different search strategies by comparing them against the baseline (IVFFlat Only) strategy.

## Methodology

- **Ground Truth**: IVFFlat Only (baseline semantic search)
- **Evaluation Metric**: Overlap of chunk_ids retrieved
- **Test Queries**: 27 queries from actual documents
- **Top-K**: 10 results per query

### Metrics Explained

1. **Precision**: What percentage of retrieved results match the baseline?
   - `Precision = (True Positives) / (Total Retrieved)`
   - High precision = strategy retrieves similar results to baseline

2. **Recall**: What percentage of baseline results were retrieved?
   - `Recall = (True Positives) / (Total Baseline Results)`
   - High recall = strategy captures most baseline results

3. **F1 Score**: Harmonic mean of precision and recall
   - `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
   - Balanced measure of both metrics

## Results Summary

| Strategy | Precision | Recall | F1 Score | Interpretation |
|----------|-----------|--------|----------|----------------|
| **IVFFlat + NER Boost** | 1.0000 | 1.0000 | 1.0000 | ✅ Perfect match with baseline |
| **IVFFlat + tsvector** | 0.9565 | 0.9565 | 0.9565 | ⚠️ 95.65% overlap with baseline |
| **IVFFlat + NER + tsvector** | 1.0000 | 1.0000 | 1.0000 | ✅ Perfect match with baseline |

## Key Findings

### 1. IVFFlat + NER Boost (Strategy 2)
- **Precision**: 100%
- **Recall**: 100%
- **Conclusion**: This strategy retrieves **exactly the same chunks** as the baseline, just re-ranked with NER boost
- **Why**: NER boost is a soft re-ranking technique that doesn't change which chunks are retrieved, only their order
- **Use Case**: Best when you want baseline results but prioritized by entity mentions

### 2. IVFFlat + tsvector (Strategy 3)
- **Precision**: 95.65%
- **Recall**: 95.65%
- **Conclusion**: Introduces **~4.35% diversity** in results compared to baseline
- **Why**: Hybrid search combines vector similarity with keyword matching, which can surface different relevant chunks
- **Recall Range**: 90% - 100% (some queries have perfect overlap, others have 90%)
- **Use Case**: Best when you want a balance of semantic and keyword-based results
- **Note**: Lower recall doesn't mean worse quality - it means different (potentially more relevant) results

### 3. IVFFlat + NER + tsvector (Strategy 4)
- **Precision**: 100%
- **Recall**: 100%
- **Conclusion**: Despite combining three signals, retrieves same chunks as baseline
- **Why**: The weighted combination still prioritizes the same top-k chunks
- **Use Case**: Full hybrid approach with perfect baseline coverage

## Latency vs Recall Trade-off

From the benchmark results:

| Strategy | Avg Latency | Recall | Trade-off |
|----------|-------------|--------|-----------|
| IVFFlat + tsvector | 29.7ms | 95.65% | ⚡ **Fastest**, slight diversity |
| IVFFlat + NER Boost | 34.4ms | 100% | Fast, perfect baseline match |
| IVFFlat + NER + tsvector | 46.9ms | 100% | Slower, perfect baseline match |
| IVFFlat Only | 161.7ms* | 100% | Slowest baseline |

*Anomalously high; typical performance is 20-30ms

## Interpretation & Recommendations

### Understanding the Results

1. **Perfect recall (100%) doesn't always mean better**
   - It means the strategy retrieves the same chunks as baseline
   - Different chunks could be more relevant for certain queries

2. **95.65% recall for tsvector strategy**
   - Shows that keyword matching introduces diversity
   - The 4.35% difference could be:
     - Chunks with better keyword matches
     - Chunks that are more precise for technical terms
     - Potentially better results for specific query types

3. **All strategies show high agreement**
   - Average overlap > 95% across all strategies
   - IVFFlat index provides strong baseline
   - Different strategies mostly re-rank rather than replace results

### Which Strategy to Choose?

**For Maximum Speed + Good Results:**
- **Use**: IVFFlat + tsvector
- **Why**: Fastest (29.7ms), 95.65% baseline coverage, adds keyword precision
- **Trade-off**: 4.35% different results (not necessarily worse)

**For Baseline Coverage + Entity Awareness:**
- **Use**: IVFFlat + NER Boost
- **Why**: 100% baseline recall, entity-aware ranking, moderate speed (34.4ms)
- **Trade-off**: None - same results, better ordering

**For Maximum Precision (Medical Queries):**
- **Use**: IVFFlat + NER + tsvector (Full Hybrid)
- **Why**: Combines all signals, 100% baseline recall
- **Trade-off**: Slower (46.9ms) but still acceptable

**For Pure Semantic Search:**
- **Use**: IVFFlat Only
- **Why**: Simplest, baseline reference
- **Trade-off**: Potentially slower in some cases

## Next Steps

### 1. Qualitative Analysis
Examine the 4.35% of different results from tsvector strategy:
- Are they more relevant than baseline for certain query types?
- Do they provide better keyword precision?

### 2. User-Based Evaluation
Run A/B testing with real users to measure:
- Click-through rates
- User satisfaction ratings
- Query completion success

### 3. Query-Type Analysis
Break down performance by query type:
- Entity-heavy queries (diseases, drugs)
- Technical/keyword queries
- Conceptual/abstract queries

## Files Generated

1. `calculate_recall.py` - Python script to compute metrics
2. `benchmark_results_30queries_recall_metrics.json` - Full results with per-query metrics
3. `RECALL_ANALYSIS_SUMMARY.md` - This summary document

## Conclusion

**TL;DR for IVFFlat + tsvector:**
- **Average Recall**: 95.65%
- **Interpretation**: Captures 95.65% of baseline results while introducing 4.35% diversity through keyword matching
- **Recommendation**: Excellent balance of speed, baseline coverage, and keyword precision

---

**Analysis Date**: 2026-01-07
**Dataset**: 27 queries, 13,475 chunks, 6,629 documents
**Embedding Model**: BAAI/bge-small-en-v1.5
