-- Step 1: Add empty tsvector column (safe, instant, no data modification)

-- Add tsvector column (nullable, empty initially)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv tsvector;

-- Verify column was added
\d chunks

-- Check current state
SELECT
    COUNT(*) as total_chunks,
    COUNT(text_tsv) as chunks_with_tsvector_data
FROM chunks;

\echo 'tsvector column added successfully (empty for now)!'
