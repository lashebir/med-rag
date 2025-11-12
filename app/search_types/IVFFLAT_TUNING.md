# IVFFlat Index Tuning Guide

This guide explains how to benchmark and optimize IVFFlat indexes for vector similarity search.

## Overview

IVFFlat (Inverted File with Flat compression) is an approximate nearest neighbor (ANN) index that trades accuracy for speed. It has two key parameters:

- **nlist**: Number of inverted lists/clusters (set at index creation)
- **nprobe**: Number of lists to probe during search (set at query time)

## Quick Start

### 1. Run Benchmark

```bash
# Run with default settings (100 queries, top-10)
python app/benchmark_ivfflat.py

# Run with more queries for better statistics
python app/benchmark_ivfflat.py --test-queries 200 --top-k 10

# Save to custom file
python app/benchmark_ivfflat.py --output my_results.json
```

The benchmark will:
1. Test nlist values: [32, 64, 96, 128]
2. For each nlist, test nprobe values: [1, 2, 4, 8, 16]
3. Measure recall@K and query latency
4. Compare against exact search baseline

**Time estimate**: ~15-30 minutes depending on database size

### 2. Review Results

The script outputs:
- Summary table with recall and query times
- Recommendations for best recall, speed, and balanced configs
- JSON file with detailed results

Example output:
```
nlist    nprobe   Recall@K     Query(ms)    vs Exact     Build(s)
--------------------------------------------------------------------------------
32       1        0.847        12.34        15.2x        2.45
32       2        0.923        18.56        10.1x        2.45
64       1        0.856        11.28        16.6x        3.12
64       4        0.951        22.14        8.5x         3.12
96       8        0.982        28.73        6.5x         3.89
128      16       0.995        42.18        4.4x         4.56
```

### 3. Apply Optimal Configuration

```bash
# Apply recommended configuration (e.g., nlist=96, nprobe=4)
python app/apply_ivfflat_config.py --nlist 96 --nprobe 4

# Dry run to see what would happen
python app/apply_ivfflat_config.py --nlist 96 --nprobe 4 --dry-run
```

## Parameter Tuning Guidelines

### nlist (Index Build Time)

**Rule of thumb**: √N to 2√N where N = number of vectors

For database sizes:
- 1,000 chunks: nlist = 32-64
- 10,000 chunks: nlist = 64-200
- 100,000 chunks: nlist = 200-640
- 1,000,000 chunks: nlist = 640-2000

**Trade-offs**:
- **Lower nlist** (e.g., 32):
  - ✅ Faster index build
  - ✅ Faster queries with low nprobe
  - ❌ Lower maximum recall

- **Higher nlist** (e.g., 128):
  - ✅ Higher maximum recall possible
  - ❌ Slower index build
  - ❌ Slower queries if nprobe is proportionally high

### nprobe (Query Time)

**Rule of thumb**: Start with nlist/16, tune based on recall needs

**Trade-offs**:
- **Lower nprobe** (e.g., 1-2):
  - ✅ Fastest queries (10-20x speedup vs exact)
  - ❌ Lower recall (~85-90%)

- **Higher nprobe** (e.g., 8-16):
  - ✅ Higher recall (95-99%)
  - ❌ Slower queries (4-8x speedup vs exact)

**Note**: nprobe must be ≤ nlist

## Recommended Configurations

Based on typical use cases:

### Production RAG System (Balanced)
```bash
# Good recall (~95%) with reasonable speed (~10x faster than exact)
python app/apply_ivfflat_config.py --nlist 96 --nprobe 4
```

### Speed-Critical Applications
```bash
# Maximum speed (~15x faster) with acceptable recall (~85%)
python app/apply_ivfflat_config.py --nlist 64 --nprobe 1
```

### High-Precision Requirements
```bash
# High recall (~98%) with moderate speed (~6x faster)
python app/apply_ivfflat_config.py --nlist 128 --nprobe 8
```

## Interpreting Benchmark Results

### Metrics

1. **Recall@K**: What fraction of true top-K results are retrieved
   - 1.0 = perfect (same as exact search)
   - 0.9 = 90% of true results found
   - < 0.85 = too low for most applications

2. **Query Time**: Milliseconds per query
   - Compare to "vs Exact" column for speedup
   - Goal: 10-20x faster than exact search

3. **Build Time**: How long to create the index
   - Only matters when rebuilding (infrequent)
   - Acceptable range: 2-10 seconds per 10K chunks

### Decision Matrix

| Priority | nlist | nprobe | Expected Recall | Expected Speedup |
|----------|-------|--------|-----------------|------------------|
| Speed | 64 | 1-2 | 85-90% | 15-20x |
| Balanced | 96 | 4 | 93-96% | 8-12x |
| Accuracy | 128 | 8-16 | 97-99% | 4-8x |

## Advanced: Per-Query nprobe

You can override nprobe per query for dynamic tuning:

```sql
-- In application code or query
BEGIN;
SET LOCAL ivfflat.probes = 8;

SELECT * FROM chunks
ORDER BY embedding <=> '[query_vector]'::vector
LIMIT 10;

COMMIT;
```

This allows:
- Low nprobe for background/bulk operations
- High nprobe for user-facing critical queries

## Monitoring in Production

Track these metrics:
1. **Query latency p50, p95, p99**
2. **User satisfaction** (click-through rate on results)
3. **Recall** (periodic evaluation against ground truth)

Re-run benchmarks:
- When database grows significantly (>2x)
- After major data ingestion
- When query patterns change

## Troubleshooting

### "Index not being used"
```sql
-- Check if index exists
\d chunks

-- Force index usage
SET enable_seqscan = off;
```

### "Recall too low"
- Increase nprobe (double it and re-test)
- Consider higher nlist and rebuild index
- Check if embeddings are normalized

### "Queries too slow"
- Decrease nprobe
- Consider lower nlist and rebuild index
- Check if other filters are applied efficiently

### "Index build fails"
- Insufficient memory (increase work_mem)
- Too many chunks (consider sampling)
- Check disk space

## Example Workflow

```bash
# 1. Initial benchmark
python app/benchmark_ivfflat.py --test-queries 100

# 2. Review results, identify best balanced config
# Example: nlist=96, nprobe=4 gives recall=0.95, query_time=20ms

# 3. Apply configuration
python app/apply_ivfflat_config.py --nlist 96 --nprobe 4

# 4. Test in application
curl -X POST http://localhost:8000/answer \
  -d '{"question": "What is diabetes?", "top_k": 5}'

# 5. Monitor and adjust nprobe if needed
# If recall too low in production, increase nprobe
psql -c "ALTER DATABASE medrag SET ivfflat.probes = 8;"
```

## References

- [pgvector IVFFlat docs](https://github.com/pgvector/pgvector#ivfflat)
- [Approximate Nearest Neighbors](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor)
- Database: `{PGDATABASE}` (configured in .env)
- Embedding model: `{EMBED_MODEL}`

## Automation

Add to your ingestion pipeline:
```python
# After bulk ingestion, rebuild index
import subprocess

total_chunks = get_chunk_count()
nlist = int(total_chunks ** 0.5 * 1.5)  # 1.5 * sqrt(N)

subprocess.run([
    "python", "app/apply_ivfflat_config.py",
    "--nlist", str(nlist),
    "--nprobe", str(max(1, nlist // 16))
])
```
