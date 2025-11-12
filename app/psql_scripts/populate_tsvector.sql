-- Step 2: Populate tsvector column and add index
-- Run this AFTER verifying the column exists and you're ready

-- Show progress
\echo 'Populating tsvector data...'

-- Populate tsvector from existing text
UPDATE chunks
SET text_tsv = to_tsvector('english', COALESCE(text, ''))
WHERE text_tsv IS NULL;

\echo 'Creating GIN index for fast full-text search...'

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING gin(text_tsv);

\echo 'Creating auto-update trigger...'

-- Create trigger function to auto-update tsvector on insert/update
CREATE OR REPLACE FUNCTION chunks_text_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.text_tsv := to_tsvector('english', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop old trigger if exists
DROP TRIGGER IF EXISTS chunks_text_tsv_update ON chunks;

-- Create trigger
CREATE TRIGGER chunks_text_tsv_update
    BEFORE INSERT OR UPDATE OF text
    ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_text_tsv_trigger();

\echo 'Verifying results...'

-- Verify results
SELECT
    COUNT(*) as total_chunks,
    COUNT(text_tsv) as chunks_with_tsvector,
    pg_size_pretty(pg_total_relation_size('chunks')) as table_size,
    pg_size_pretty(pg_relation_size('idx_chunks_text_tsv')) as tsvector_index_size
FROM chunks;

\echo 'tsvector population complete!'
