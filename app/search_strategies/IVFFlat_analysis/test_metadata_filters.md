# Metadata Filter Testing Examples

## Overview
The QA endpoint now supports metadata filtering BEFORE semantic search for more targeted results.

## Available Filters

1. **authors** (List[str]): Filter by author names (case-insensitive partial match)
2. **year_min** (int): Minimum publication year (inclusive, 1900-2100)
3. **year_max** (int): Maximum publication year (inclusive, 1900-2100)
4. **institutions** (List[str]): Filter by institution names (case-insensitive partial match)
5. **sources** (List[str]): Filter by source (exact match): "arXiv", "PubMed Central", "Google Scholar"

## Test Queries

### 1. Filter by Author
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the latest diabetes treatments?",
    "top_k": 5,
    "filters": {
      "authors": ["Smith"]
    }
  }'
```

### 2. Filter by Year Range (Recent Papers)
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are recent advances in machine learning for healthcare?",
    "top_k": 5,
    "filters": {
      "year_min": 2020,
      "year_max": 2024
    }
  }'
```

### 3. Filter by Source
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the role of CRISPR in gene therapy?",
    "top_k": 5,
    "filters": {
      "sources": ["PubMed Central"]
    }
  }'
```

### 4. Combined Filters
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does immunotherapy work for cancer treatment?",
    "top_k": 5,
    "filters": {
      "authors": ["Johnson", "Chen"],
      "year_min": 2018,
      "sources": ["PubMed Central", "arXiv"]
    }
  }'
```

### 5. Filter by Institution
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What research has been done on COVID-19 vaccines?",
    "top_k": 5,
    "filters": {
      "institutions": ["Harvard", "MIT"]
    }
  }'
```

## Get Available Filter Values

```bash
curl -X GET "http://localhost:8000/filters/available"
```

This returns:
- Available sources
- Year range (min/max)
- Total document count
- Top 20 most frequent authors
- Top 20 most frequent institutions
- Usage examples

## Python Example

```python
import requests

# Example 1: Filter by author and recent years
response = requests.post(
    "http://localhost:8000/answer",
    json={
        "question": "What are the mechanisms of antibiotic resistance?",
        "top_k": 5,
        "filters": {
            "authors": ["Garcia"],
            "year_min": 2019
        }
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"\nCitations: {result['citations']}")
print(f"\nContexts used: {len(result['used_contexts'])}")

# Example 2: No filters (search all documents)
response = requests.post(
    "http://localhost:8000/answer",
    json={
        "question": "What is the structure of DNA?",
        "top_k": 5
    }
)
```

## Filter Behavior

1. **Filters are applied FIRST**: Database is filtered by metadata before semantic search
2. **Multiple authors/institutions**: Documents matching ANY of the specified values are included (OR logic)
3. **Year range**: Both bounds are inclusive
4. **Partial matching**: Author and institution filters use case-insensitive partial matching
5. **No results**: If filters are too restrictive, you'll get a 404 error suggesting to relax criteria

## Performance Considerations

- Metadata filters use standard B-tree indexes (author has GIN index for array operations)
- Filtering before semantic search reduces the search space
- IVFFlat vector index is still used on the filtered subset
- For very specific filters, the filtered set might be small enough for exact search

## Expected Response Format

```json
{
  "answer": "Based on the provided context...",
  "citations": [
    {
      "pmcid": "PMC1234567",
      "chunk_index": 2,
      "source_uri": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567",
      "title": "Study on XYZ"
    }
  ],
  "used_contexts": [
    {
      "pmcid": "PMC1234567",
      "chunk_index": 2,
      "text": "The context text...",
      "title": "Study on XYZ",
      "source_uri": "https://...",
      "source": "PubMed Central",
      "authors": ["Smith J", "Jones K"],
      "year": 2023,
      "institution": "Harvard Medical School",
      "distance": 0.234,
      "similarity": 0.766
    }
  ]
}
```

## Troubleshooting

### No results with filters
- Check available values: `GET /filters/available`
- Try partial matches (e.g., "Harvard" instead of "Harvard Medical School")
- Broaden year range or remove year filters
- Check if author names are formatted correctly in the database

### Slow queries
- Very broad filters might search many documents
- Consider adding more specific filters to narrow the scope
- Year range filters are efficient (indexed on date column)
