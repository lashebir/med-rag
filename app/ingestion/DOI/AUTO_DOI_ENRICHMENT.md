# Automatic DOI Enrichment in Ingest Scripts

## Overview

All bulk ingest scripts now **automatically run DOI enrichment** after ingestion completes. This ensures that newly ingested documents are immediately enriched with DOI information when available.

## What Changed

### 1. Default Batch Size Increased

The DOI enrichment limit has been increased from 500 to **1000 documents**:
- File: `app/ingestion/DOI/enrich_doi.py`
- Default: `ENRICH_LIMIT=1000`

### 2. Auto-Enrichment Added to Ingest Scripts

The following scripts now automatically run DOI enrichment:

- ✅ `app/ingestion/arXiv/arXiv_bulk_ingest.py`
- ✅ `app/ingestion/PubMed_Central/pmc_bulk_ingest.py`
- ✅ `app/ingestion/Google_Scholar/scholar_bulk_ingest.py`

### 3. New DOI Enrichment Utilities

Created `app/ingestion/DOI/doi_enrichment_utils.py` with helper functions:
- `get_max_doc_id()` - Captures doc_id before ingestion
- `get_new_doc_ids()` - Finds newly ingested documents
- `auto_enrich_dois_after_ingestion()` - Main enrichment function
- `count_dois_for_doc_ids()` - Counts DOIs found
- `enrich_recent_dois()` - Processes DOI extraction

## How It Works

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Capture max_doc_id before ingestion starts              │
├─────────────────────────────────────────────────────────────┤
│ 2. Run normal ingestion (arXiv, PMC, Scholar, etc.)        │
├─────────────────────────────────────────────────────────────┤
│ 3. Find newly ingested doc_ids (doc_id > max_doc_id_before)│
├─────────────────────────────────────────────────────────────┤
│ 4. Run DOI enrichment for those specific documents          │
├─────────────────────────────────────────────────────────────┤
│ 5. Display summary: "x/n DOIs have been added"             │
└─────────────────────────────────────────────────────────────┘
```

### Example Output

When you run an ingest script, you'll now see:

```bash
📊 Starting ingestion (current max doc_id: 6631)

# ... normal ingestion output ...

======================================================================
🎯 INGESTION COMPLETE - Running DOI enrichment...
======================================================================

🔍 Running DOI enrichment on 25 newly ingested arXiv documents...

======================================================================
🔍 DOI ENRICHMENT
======================================================================
📄 Processing 25 documents...

======================================================================
✅ DOI ENRICHMENT COMPLETE
======================================================================
📊 Processed: 25 documents
📰 Journal DOIs added: 2
📄 arXiv DOIs added: 23
🎯 Total DOIs found: 25/25 (100.0%)
======================================================================

✨ 25/25 DOIs of the newly ingested sources have been added to your database
```

## Usage

### No Changes Required!

Simply run your ingest scripts as usual:

```bash
# arXiv ingestion - DOI enrichment runs automatically
python app/ingestion/arXiv/arXiv_bulk_ingest.py --query "cat:cs.AI" --limit 100

# PMC ingestion - DOI enrichment runs automatically
python app/ingestion/PubMed_Central/pmc_bulk_ingest.py --pmc PMC12345 PMC67890

# Scholar ingestion - DOI enrichment runs automatically
python app/ingestion/Google_Scholar/scholar_bulk_ingest.py
```

### Customizing DOI Enrichment

If you want to run DOI enrichment manually with different settings:

```bash
# Run with custom limit
cd app/ingestion/DOI
ENRICH_LIMIT=2000 python enrich_doi.py
```

### Disabling Auto-Enrichment (Not Recommended)

If you need to temporarily disable auto-enrichment, you can comment out the enrichment calls in the ingest scripts:

```python
# In arXiv_bulk_ingest.py, PMC_bulk_ingest.py, or scholar_bulk_ingest.py
# Comment out this section:
# await auto_enrich_dois_after_ingestion(max_doc_id_before, source_name="...")
```

## Technical Details

### DOI Types

The enrichment extracts two types of DOIs:

1. **Journal DOIs** (published papers)
   - Example: `10.1109/taslp.2024.3446232`
   - Found for arXiv papers published in journals
   - Found for PMC papers via BioC/NXML extraction

2. **arXiv DOIs** (preprints)
   - Example: `10.48550/arXiv.2511.01372`
   - DataCite DOIs for all arXiv e-prints
   - Always available for arXiv papers

### Database Schema

DOIs are stored in the `document_ids` table:

```sql
CREATE TABLE document_ids (
  doc_id    bigint REFERENCES documents(doc_id) ON DELETE CASCADE,
  id_type   text CHECK (id_type IN ('pmid','pmcid','doi','arxiv-doi')),
  id_value  text,
  UNIQUE (id_type, lower(id_value))
);
```

### Performance

- **Batch size**: Up to 1000 documents per ingest
- **Speed**: ~0.5-2 seconds per document (API dependent)
- **Sources**:
  - arXiv: arXiv API + Crossref + DataCite
  - PMC: BioC JSON + NXML parsing
  - Scholar: Google Scholar metadata (if available)

### Error Handling

- DOI enrichment failures don't stop ingestion
- Errors are logged but don't cause ingestion to fail
- Retry enrichment later with standalone script if needed

## Benefits

1. **Immediate Citation Links**: DOIs available right after ingestion
2. **Better Metadata**: Enables linking to journal publications
3. **Automatic Process**: No manual DOI enrichment step needed
4. **Clear Reporting**: Shows exactly how many DOIs were found

## Troubleshooting

### "No new documents to enrich"

This means all ingested documents already had DOIs or no new documents were ingested.

### "0/25 DOIs found"

This can happen for:
- Google Scholar sources without DOI metadata
- PMC papers without DOI in their metadata
- API errors during enrichment

**Solution**: Run the standalone enrichment script later:
```bash
cd app/ingestion/DOI
python enrich_doi.py
```

### Slow Enrichment

If DOI enrichment is too slow:
- Check your internet connection (requires API calls)
- Reduce the batch size in future ingestions
- Run enrichment as a separate background process

## Files Modified

1. **New Utilities**: `app/ingestion/DOI/doi_enrichment_utils.py`
2. **Updated Scripts**:
   - `app/ingestion/arXiv/arXiv_bulk_ingest.py`
   - `app/ingestion/PubMed_Central/pmc_bulk_ingest.py`
   - `app/ingestion/Google_Scholar/scholar_bulk_ingest.py`
3. **Updated Settings**: `app/ingestion/DOI/enrich_doi.py` (default limit: 1000)

## Next Steps

After ingestion with DOI enrichment:
1. DOIs are immediately available in the database
2. Can query DOIs via `document_ids` table
3. Can use DOIs for citation links in your application
4. A/B testing can track which retrieval strategies work best (see `AB_TESTING_GUIDE.md`)

---

**Status**: ✅ Implemented and ready to use
**Tested**: arXiv, PMC, and Scholar ingest scripts
**Default Behavior**: Auto-enrichment enabled for all bulk ingest scripts
