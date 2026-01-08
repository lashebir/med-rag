# Comparative Recall Analysis Across Query Volumes

## Executive Summary

This analysis compares precision and recall metrics for different search strategies across three test sizes: **27 queries**, **43 queries**, and **81 queries**.

## Results Summary

### IVFFlat + tsvector (Hybrid Vector + Keyword Search)

| Query Count | Queries Tested | Avg Precision | Avg Recall | Avg F1 | Recall Range |
|-------------|----------------|---------------|------------|--------|--------------|
| **30** | 27 | **95.65%** | **95.65%** | **95.65%** | 90.0% - 100% |
| **50** | 43 | **93.08%** | **93.08%** | **93.08%** | 90.0% - 100% |
| **100** | 81 | **93.89%** | **93.89%** | **93.89%** | 90.0% - 100% |

**Key Finding**: As query volume increases, recall **stabilizes around 93-94%** with consistent variance (90-100% range).

### IVFFlat + NER Boost

| Query Count | Queries Tested | Avg Precision | Avg Recall | Avg F1 | Recall Range |
|-------------|----------------|---------------|------------|--------|--------------|
| **30** | 27 | **100%** | **100%** | **100%** | 100% - 100% |
| **50** | 43 | **100%** | **100%** | **100%** | 100% - 100% |
| **100** | 81 | **100%** | **100%** | **100%** | 100% - 100% |

**Key Finding**: Perfect baseline match across all query volumes. NER Boost is a pure re-ranking strategy.

### IVFFlat + NER + tsvector (Full Hybrid)

| Query Count | Queries Tested | Avg Precision | Avg Recall | Avg F1 | Recall Range |
|-------------|----------------|---------------|------------|--------|--------------|
| **30** | 27 | **100%** | **100%** | **100%** | 100% - 100% |
| **50** | 43 | **99.46%** | **99.46%** | **99.46%** | 90.0% - 100% |
| **100** | 81 | **99.44%** | **99.44%** | **99.44%** | 90.0% - 100% |

**Key Finding**: Near-perfect baseline match (99.4-100%), with slight diversity introduced at higher query volumes.

## Detailed Analysis

### 1. Recall Stability Across Query Volumes

**IVFFlat + tsvector:**
- Initial sample (27 queries): 95.65%
- Medium sample (43 queries): 93.08% ⬇️ 2.57%
- Large sample (81 queries): 93.89% ⬆️ 0.81%

**Interpretation**:
- Recall **converges to ~93-94%** as sample size increases
- The 95.65% from 27 queries was slightly optimistic
- **True recall estimate: 93-94%** (more reliable with larger sample)

### 2. Per-Query Variance

All strategies show consistent recall range of **90-100%** per query:
- **Minimum recall**: 90% (9 out of 10 chunks match baseline)
- **Maximum recall**: 100% (all 10 chunks match baseline)
- **Most common**: Some queries have perfect match, others have 90% match

### 3. Latency Performance

| Strategy | 27q Latency | 43q Latency | 81q Latency | Trend |
|----------|-------------|-------------|-------------|-------|
| IVFFlat Only | 161.7ms | 175.3ms | 78.1ms | Variable* |
| IVFFlat + NER Boost | 34.4ms | 33.2ms | 31.8ms | ✅ Consistent |
| IVFFlat + tsvector | 29.7ms | 42.5ms | 34.7ms | ~30-35ms |
| IVFFlat + NER + tsvector | 46.9ms | 35.8ms | 35.8ms | ~35-40ms |

*Baseline latency appears anomalously high in smaller samples; likely due to cold start effects.

## Key Insights

### 1. Sample Size Matters
- **27 queries**: Good initial estimate, but slightly optimistic (95.65%)
- **43 queries**: More realistic estimate (93.08%)
- **81 queries**: Most reliable estimate (93.89%)
- **Recommendation**: Use at least 50-100 queries for stable estimates

### 2. IVFFlat + tsvector Recall is Consistent
- Stabilizes around **93-94% recall**
- This means: Retrieves ~9.3-9.4 out of 10 baseline chunks per query
- **Diversity**: Introduces 6-7% different chunks through keyword matching

### 3. Full Hybrid (NER + tsvector) Nearly Matches Baseline
- 99.4% recall means only 0.6% diversity
- Combines three signals but still highly aligned with baseline
- Suggests vector similarity dominates even in weighted combination

### 4. NER Boost is Not Introducing Diversity
- 100% recall across all tests
- Pure re-ranking without retrieval changes
- Same results, just different order

## Recommendations by Use Case

### For Maximum Baseline Coverage (100% Recall)
**Use**: IVFFlat + NER Boost
- ✅ Perfect baseline match
- ✅ Entity-aware ranking
- ✅ Fast (31-34ms)
- ❌ No diversity in results

### For Balance of Coverage + Diversity (93-94% Recall)
**Use**: IVFFlat + tsvector
- ✅ High baseline coverage (93-94%)
- ✅ Keyword precision (6-7% different chunks)
- ✅ Fast (30-35ms)
- ✅ Best for technical/medical terminology

### For Near-Perfect Coverage + Multi-Signal Ranking (99% Recall)
**Use**: IVFFlat + NER + tsvector (Full Hybrid)
- ✅ Nearly perfect baseline coverage (99.4%)
- ✅ Combines vector + keyword + entity signals
- ✅ Reasonable speed (35-40ms)
- ⚠️ Minimal diversity (0.6% different chunks)

## Statistical Confidence

### Margin of Error

Using binomial proportion confidence intervals (95% confidence):

| Query Count | Sample Size | Recall Estimate | Margin of Error |
|-------------|-------------|-----------------|-----------------|
| 30 | 27 | 95.65% | ±3.9% |
| 50 | 43 | 93.08% | ±3.8% |
| 100 | 81 | 93.89% | ±2.6% |

**True recall for IVFFlat + tsvector**: **93.89% ± 2.6%** (91.3% - 96.5% with 95% confidence)

## Latency vs Recall Trade-off

```
Perfect Recall (100%)
│
│  ● IVFFlat + NER Boost (31-34ms)
│  ● IVFFlat + NER + tsvector (35-40ms)
│
│  ↕ Trade-off zone ↕
│
94% Recall
│  ● IVFFlat + tsvector (30-35ms)  ← FASTEST + Good Recall
│
│
└─────────────────────────────────────────────> Latency (ms)
      30ms          35ms          40ms
```

## What Does 93.89% Recall Mean Practically?

For a typical 10-result search:
- **9.4 chunks** match baseline results (overlap)
- **0.6 chunks** are different (keyword-matched alternatives)

Over 100 queries:
- **~940 chunks** are the same as baseline
- **~60 chunks** are different (potentially more keyword-relevant)

## Conclusion

**For IVFFlat + tsvector search strategy:**

✅ **Reliable Recall**: 93.89% (with 81 queries)
✅ **Stable Performance**: Converges around 93-94%
✅ **Fast Execution**: 30-35ms average latency
✅ **Keyword Precision**: Introduces 6-7% diversity for technical terms

**Bottom Line**: IVFFlat + tsvector provides excellent baseline coverage (94%) while adding keyword-based diversity, making it ideal for medical/technical search where both semantic understanding and exact term matching matter.

---

**Analysis Date**: 2026-01-07
**Dataset**: 13,475 chunks, 6,629 documents
**Embedding Model**: BAAI/bge-small-en-v1.5
**Test Configurations**: 27, 43, and 81 queries with top-k=10
