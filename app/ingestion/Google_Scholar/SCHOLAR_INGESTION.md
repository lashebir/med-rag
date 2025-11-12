# Google Scholar Ingestion via SerpAPI

Successfully integrated Google Scholar as a third data source alongside PubMed Central and arXiv!

## Setup

Your SerpAPI key is already configured in `.env`:
```bash
SERPAPI=50b84779b68e46027d0770400bc0bd05e607c6994d3318d319221b7bf7a4040f
```

## Files Created

1. **`app/scholar_ingest.py`** - Core ingestion script
   - SerpAPI integration for Google Scholar search
   - PDF download and text extraction (when available)
   - Chunking and embedding generation
   - Database storage with `source = 'Google Scholar'`

2. **`app/scholar_bulk_ingest.py`** - Bulk ingestion for multiple topics
   - 4 rounds of topics (Core, Clinical, Molecular, Neuroscience)
   - Polite rate limiting with delays
   - Concurrent topic processing

## Usage

### Single Query

Ingest up to N results for a single query:

```bash
python -m app.scholar_ingest \
  --query "cochlear synaptopathy" \
  --retmax 20 \
  --concurrency 2
```

**Parameters:**
- `--query`: Search query (required)
- `--retmax`: Maximum results to fetch (default: 80)
- `--concurrency`: Concurrent downloads (default: 3)

### Bulk Ingestion

Ingest multiple predefined topics:

```bash
python -m app.scholar_bulk_ingest
```

This will prompt for confirmation and then run all 4 rounds of topics.

**Configuration (in script):**
- `MAX_TOPIC_CONCURRENCY = 2` - Topics processed in parallel
- `RETMAX_PER_TOPIC = 20` - Results per topic
- `PAUSE_BETWEEN_TOPICS = (5, 10)` - Seconds between topics
- `PAUSE_BETWEEN_ROUNDS = (30, 60)` - Seconds between rounds

### Customize Topics

Edit `app/scholar_bulk_ingest.py` to modify the topic lists:

```python
ROUND_1_TOPICS = [
    "your custom query 1",
    "your custom query 2",
    # ... add more
]
```

## Features

### PDF Handling
- Automatically downloads PDFs when available from Google Scholar results
- Extracts text using PyPDF2 (first 30 pages)
- Falls back to title + snippet if PDF unavailable or fails
- Respects `SCHOLAR_FETCH_PDF` env variable (default: enabled)

### Rate Limiting
- `SCHOLAR_DELAY = 1.0s` - Delay between SerpAPI requests
- Exponential backoff on failures
- Polite delays between topics in bulk mode

### Data Model

Documents are stored with:
```sql
ext_id: "scholar://eAk5HWJs2NYJ"
source: "Google Scholar"
source_id: "eAk5HWJs2NYJ"  -- SerpAPI result_id or hash of title
title: "Full paper title"
source_uri: "https://..." -- Link to paper
author: "Author 1; Author 2; ..."
year: 2023-01-01  -- Extracted from publication info
```

Chunks include metadata:
```json
{
  "source": "Google Scholar",
  "source_id": "eAk5HWJs2NYJ",
  "chunk_index": 0,
  "chunk_tokens": 800,
  "chunk_overlap": 80,
  "embedding_model": "BAAI/bge-small-en-v1.5"
}
```

## Verification

### Check Ingested Documents

```sql
SELECT d.source, d.source_id, d.title,
       COUNT(c.chunk_id) as num_chunks
FROM documents d
LEFT JOIN chunks c ON c.doc_id = d.doc_id
WHERE d.source = 'Google Scholar'
GROUP BY d.doc_id, d.source, d.source_id, d.title
ORDER BY d.doc_id DESC
LIMIT 10;
```

### Test Semantic Search

```bash
python test_semantic_search.py "your research question"
```

Google Scholar results will appear alongside PubMed and arXiv results, ranked by semantic similarity.

## Environment Variables

All standard config from `.env` applies:

```bash
# Embedding
EMBED_MODEL=BAAI/bge-small-en-v1.5
CHUNK_TOKENS=800
CHUNK_OVERLAP=80
MAX_CHUNKS=6
EMBED_BATCH_SIZE=8

# PDF Fetching
SCHOLAR_FETCH_PDF=1
HTTP_TIMEOUT=90
MAX_RETRIES=5

# Rate Limiting
SCHOLAR_DELAY=1.0
MAX_CONCURRENCY=3

# Database
FORCE_UPDATE=1
PURGE_OLD_CHUNKS=1
```

## SerpAPI Notes

- **Free tier**: 100 searches/month
- **Paid tiers**: Higher limits available
- Each query counts as one search
- Results are cached by SerpAPI (faster repeated queries)
- Check usage: https://serpapi.com/dashboard

## Integration with Ollama

Google Scholar results automatically integrate with your existing RAG pipeline:

```bash
# Start your FastAPI server
uvicorn app.main:app --reload

# Query via API (includes all sources)
curl -X POST http://localhost:8000/qa/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the mechanisms of cochlear synaptopathy?", "top_k": 5}'
```

The answer will cite sources like:
- `[PMC12330969#0]` - PubMed Central
- `[arXiv:2401.01234#2]` - arXiv
- `[SCHOLAR:eAk5HWJs2NYJ#0]` - Google Scholar

## Performance

**Test Results:**
- Query: "cochlear synaptopathy"
- Time: ~35s for 2 results (includes PDF attempts)
- Storage: 2 documents, 1 chunk each
- Semantic similarity: 0.8587 (highly relevant)

**Comparison to PubMed Central:**
- Google Scholar results ranked alongside PMC results
- Similar semantic similarity scores
- Broader coverage (includes non-open-access papers)

## Troubleshooting

### PDF Download Failures
Some publishers block automated PDF downloads (e.g., Elsevier with 406 status). This is normal - the system will use title + snippet instead.

### SerpAPI Rate Limits
If you hit rate limits:
1. Increase `SCHOLAR_DELAY`
2. Decrease `MAX_TOPIC_CONCURRENCY`
3. Reduce `RETMAX_PER_TOPIC`
4. Upgrade SerpAPI plan

### Duplicate Detection
Documents are deduplicated by `source_id`. Re-running the same query will skip already-ingested papers (unless `FORCE_UPDATE=1`).

## Next Steps

1. **Expand topic coverage**: Add more topics to bulk ingest script
2. **Monitor SerpAPI usage**: Track monthly limit
3. **Compare sources**: Benchmark retrieval quality across PubMed, arXiv, and Scholar
4. **Fine-tune ranking**: Experiment with hybrid search (semantic + lexical)

## Summary

✅ Google Scholar ingestion fully integrated
✅ SerpAPI connection working
✅ PDF download with fallback
✅ Database storage with proper schema
✅ Semantic search returns Scholar results
✅ Ollama integration tested
✅ Bulk ingestion script ready

Your RAG system now has **three complementary data sources**:
- **PubMed Central**: Open-access biomedical papers
- **arXiv**: Preprints and computational research
- **Google Scholar**: Broad academic coverage
