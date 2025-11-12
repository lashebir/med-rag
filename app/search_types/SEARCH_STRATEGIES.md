# Search Strategy Comparison Guide

This guide explains the three different retrieval strategies implemented for comparing search performance.

## Overview

All three strategies use the **same IVFFlat vector index** (nlist=96, nprobe=8) but differ in how they rank and score results:

1. **IVFFlat Only** (baseline): Pure vector similarity
2. **IVFFlat + NER Boost**: Vector similarity with soft boost from entity mentions
3. **IVFFlat + tsvector**: Hybrid vector + full-text keyword search

## Strategy Details

### 1. IVFFlat Only (Baseline)

**What it does:**
- Pure semantic similarity search using sentence embeddings
- Uses IVFFlat approximate nearest neighbor index
- Ranking: cosine distance only

**Best for:**
- Semantic/conceptual queries
- When exact keywords don't matter
- General question answering

**Example query:**
```python
from app.llm_integration.search_strategies import retrieve_ivfflat_only

results = retrieve_ivfflat_only(
    question="What are the mechanisms of antibiotic resistance?",
    k=10
)
```

**Result fields:**
```python
{
    "chunk_id": 12345,
    "text": "...",
    "similarity": 0.85,  # Cosine similarity (0-1)
    "final_score": 0.85,  # Same as similarity
    ...
}
```

---

### 2. IVFFlat + NER Boost (Soft Boost)

**What it does:**
- Retrieves candidates using vector similarity
- Checks which chunks have entity mentions (disease, drug, device, statistical_significance)
- Applies **soft boost** to chunks with more mentions
- Re-ranks based on combined score

**Formula:**
```
final_score = similarity + (ner_boost_weight * ner_match_score)

where:
- similarity: cosine similarity (0-1)
- ner_boost_weight: default 0.15 (configurable)
- ner_match_score: log-scaled mention count (0-1)
  - 0 mentions → 0.0
  - 3 mentions → ~0.5
  - 9 mentions → ~0.8
  - 10+ mentions → 1.0
```

**Best for:**
- Medical/scientific queries where entity-rich chunks are more relevant
- Boosting precision without hard filtering
- Queries about specific diseases, drugs, or medical devices

**Example query:**
```python
from app.llm_integration.search_strategies import retrieve_ivfflat_ner_boost

results = retrieve_ivfflat_ner_boost(
    question="What drugs are used to treat diabetes?",
    k=10,
    ner_boost_weight=0.15,  # How much to boost (0-1)
    initial_k_multiplier=3   # Fetch 3*k candidates before re-ranking
)
```

**Result fields:**
```python
{
    "chunk_id": 12345,
    "text": "...",
    "similarity": 0.80,           # Original vector similarity
    "final_score": 0.89,          # Boosted score
    "ner_mention_count": 8,       # Number of entity mentions in this chunk
    "ner_boost_applied": 0.09,    # Actual boost added
    "entity_types": ["drug", "disease"],  # Types of entities found
    ...
}
```

**Tuning parameters:**
- `ner_boost_weight` (0-1): How much to boost
  - 0.10 = conservative (10% max boost)
  - 0.15 = moderate (15% max boost) **← default**
  - 0.25 = aggressive (25% max boost)

- `initial_k_multiplier` (int): How many candidates to fetch
  - 2 = fetch 2*k candidates
  - 3 = fetch 3*k candidates **← default**
  - 5 = fetch 5*k candidates (more thorough)

---

### 3. IVFFlat + tsvector (Hybrid)

**What it does:**
- Runs TWO parallel searches:
  1. Vector similarity search (semantic)
  2. Full-text keyword search (tsvector)
- Combines results using RRF (Reciprocal Rank Fusion) or weighted scoring
- Returns unified ranked list

**Two fusion methods:**

#### A) RRF (Reciprocal Rank Fusion) - Default
```
final_score = 1/(k + vector_rank) + 1/(k + text_rank)

where:
- k: RRF constant (default 60)
- vector_rank: position in vector search results
- text_rank: position in full-text search results
```

#### B) Weighted Scoring
```
final_score = (vector_weight * similarity) + (text_weight * normalized_text_score)

where:
- vector_weight: default 0.7
- text_weight: default 0.3
```

**Best for:**
- Queries with specific keywords
- When both semantic meaning AND exact terms matter
- Technical/medical terminology searches

**Example query (RRF):**
```python
from app.llm_integration.search_strategies import retrieve_ivfflat_tsvector

results = retrieve_ivfflat_tsvector(
    question="CRISPR gene editing techniques",
    k=10,
    use_rrf=True,
    rrf_k=60
)
```

**Example query (Weighted):**
```python
results = retrieve_ivfflat_tsvector(
    question="CRISPR gene editing techniques",
    k=10,
    use_rrf=False,
    vector_weight=0.7,
    text_weight=0.3
)
```

**Result fields:**
```python
{
    "chunk_id": 12345,
    "text": "...",
    "similarity": 0.82,      # Vector similarity
    "final_score": 0.0312,   # RRF score (lower rank numbers = higher score)
    "vector_rank": 3,        # Position in vector results (1-indexed)
    "text_rank": 5,          # Position in text results (1-indexed)
    "text_score": 0.15,      # ts_rank score from full-text search
    ...
}
```

**Tuning parameters:**
- `rrf_k` (int): RRF constant
  - 30 = emphasizes top-ranked results
  - 60 = balanced **← default**
  - 100 = more lenient to lower ranks

- `vector_weight` / `text_weight` (floats, sum to 1.0):
  - 0.8/0.2 = emphasize semantics
  - 0.7/0.3 = balanced **← default**
  - 0.5/0.5 = equal weight
  - 0.3/0.7 = emphasize keywords

---

## Running Benchmarks

### Compare All Strategies

```bash
# Run with default settings (50 queries, top-10)
python app/benchmark_search_strategies.py

# More queries for better statistics
python app/benchmark_search_strategies.py --test-queries 100 --top-k 10

# Custom output file
python app/benchmark_search_strategies.py --output my_benchmark.json
```

**Expected runtime:** ~5-10 minutes for 50 queries

### Benchmark Output

The script measures:

1. **Latency metrics:**
   - Average, median, P95, P99 query times
   - Helps identify which strategy is fastest

2. **Result overlap:**
   - Jaccard similarity between result sets
   - How much do strategies agree on relevant chunks?

3. **Rank correlation:**
   - Spearman correlation of rankings
   - Do strategies rank common items similarly?

**Example output:**
```
Strategy                  Avg(ms)     P50(ms)     P95(ms)     P99(ms)
--------------------------------------------------------------------------------
IVFFlat Only              23.45       22.10       28.30       35.20
IVFFlat + NER Boost       31.20       29.80       38.50       45.10
IVFFlat + tsvector        42.15       40.20       52.30       61.80

RESULT SIMILARITY
--------------------------------------------------------------------------------
Comparison                           Overlap     Rank Corr
--------------------------------------------------------------------------------
NER Boost vs Baseline                0.780       0.845
tsvector vs Baseline                 0.620       0.712
NER Boost vs tsvector                0.590       0.680
```

**Interpretation:**
- **Low overlap** (< 0.5): Strategies find different chunks (high diversity)
- **Medium overlap** (0.5-0.8): Some agreement, some divergence
- **High overlap** (> 0.8): Very similar results (low diversity)

---

## Integration with QA Endpoint

To use these strategies in your QA endpoint:

```python
# app/llm_integration/qa_endpoint.py

from app.llm_integration.search_strategies import (
    retrieve_ivfflat_only,
    retrieve_ivfflat_ner_boost,
    retrieve_ivfflat_tsvector,
)

# Option 1: Add strategy parameter
class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    strategy: Optional[str] = "ivfflat_only"  # "ivfflat_only", "ner_boost", "tsvector"
    filters: Optional[MetadataFilters] = None

@router.post("/answer")
def answer(req: QARequest):
    # Select strategy
    if req.strategy == "ner_boost":
        contexts = retrieve_ivfflat_ner_boost(req.question, k=req.top_k)
    elif req.strategy == "tsvector":
        contexts = retrieve_ivfflat_tsvector(req.question, k=req.top_k)
    else:
        contexts = retrieve_ivfflat_only(req.question, k=req.top_k)

    # Rest of QA logic...
```

---

## Quick Test

Test all strategies on a single query:

```python
from app.llm_integration.search_strategies import compare_all_strategies

results = compare_all_strategies(
    question="What are the side effects of metformin?",
    k=5
)

# Results is a dict with keys:
# - "ivfflat_only"
# - "ivfflat_ner_boost"
# - "ivfflat_tsvector"

for strategy, chunks in results.items():
    print(f"\n{strategy}:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  {i+1}. Score: {chunk['final_score']:.3f} - {chunk['text'][:100]}...")
```

---

## Decision Guide

**Use IVFFlat Only when:**
- You want pure semantic search
- Speed is critical (fastest)
- Queries are conceptual/abstract

**Use IVFFlat + NER Boost when:**
- Working with medical/scientific queries
- Want to prioritize entity-rich chunks
- Need moderate speed (slower than baseline but faster than hybrid)

**Use IVFFlat + tsvector when:**
- Queries contain specific keywords/terms
- Need balance of semantics and exact matching
- Working with technical terminology
- Can tolerate higher latency

---

## Performance Characteristics

Based on 13,475 chunks with nlist=96, nprobe=8:

| Strategy | Approx. Latency | Use Case |
|----------|----------------|----------|
| IVFFlat Only | 20-30ms | General Q&A |
| IVFFlat + NER | 30-40ms | Medical Q&A |
| IVFFlat + tsvector | 40-60ms | Technical/Keyword Search |

**Note:** Actual latency depends on:
- Database size
- Query complexity
- Hardware specs
- Network latency (if remote DB)

---

## Future Enhancements

Potential improvements:
1. **Ensemble strategies**: Combine all three with learned weights
2. **Query classification**: Auto-select strategy based on query type
3. **Dynamic boosting**: Adjust NER boost based on query entities
4. **Feedback loop**: Use user interactions to tune weights
5. **Cross-encoder re-ranking**: Use transformer re-ranker on top results

---

## Troubleshooting

### "No results" with tsvector strategy
- Query might not match any keywords
- Try stemming/lemmatization in query preprocessing
- Check if `text_tsv` column is populated

### NER boost not working
- Verify mentions table has data: `SELECT COUNT(*) FROM mentions;`
- Check if queries mention medical entities
- Increase `initial_k_multiplier` to fetch more candidates

### Slow queries
- Verify IVFFlat index is being used: `EXPLAIN ANALYZE <query>`
- Check `ivfflat.probes` setting: `SHOW ivfflat.probes;`
- Consider reducing `initial_k_multiplier` for NER boost

---

## Database Schema Requirements

All strategies work with the existing schema:

```sql
-- Required tables
chunks (
    chunk_id, doc_id, text, embedding,
    text_tsv,  -- For tsvector strategy
    section_name, embedding_model
)

mentions (
    mention_id, doc_id, section_name,
    mention_text, label, start_char, end_char
)

documents (
    doc_id, ext_id, title, source, author, year, institute
)

-- Required indexes
idx_chunks_embedding_ivfflat_nlist96 (IVFFlat on embedding)
idx_chunks_text_tsv (GIN on text_tsv)
```

---

## References

- **IVFFlat**: [pgvector documentation](https://github.com/pgvector/pgvector)
- **tsvector**: [PostgreSQL Full Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- **RRF**: [Reciprocal Rank Fusion paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **NER**: scispacy `en_core_sci_lg` model
