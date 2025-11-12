-- Add full-text search support to chunks table
-- This enables hybrid search combining vector similarity + keyword matching

-- Add tsvector column
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv tsvector;

-- Populate tsvector from existing text
UPDATE chunks
SET text_tsv = to_tsvector('english', COALESCE(text, ''))
WHERE text_tsv IS NULL;

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING gin(text_tsv);

-- Create trigger to auto-update tsvector on insert/update
CREATE OR REPLACE FUNCTION chunks_text_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.text_tsv := to_tsvector('english', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_text_tsv_update ON chunks;

CREATE TRIGGER chunks_text_tsv_update
    BEFORE INSERT OR UPDATE OF text
    ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_text_tsv_trigger();

-- Verify results
SELECT
    COUNT(*) as total_chunks,
    COUNT(text_tsv) as chunks_with_tsvector,
    pg_size_pretty(pg_total_relation_size('chunks')) as table_size,
    pg_size_pretty(pg_relation_size('idx_chunks_text_tsv')) as tsvector_index_size
FROM chunks;

\echo 'tsvector column added successfully!'
