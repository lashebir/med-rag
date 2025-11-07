-- Migration: Convert author field from TEXT to TEXT[]
-- This script converts semicolon-separated author strings to PostgreSQL arrays

BEGIN;

-- Step 1: Add a temporary column for the new array format
ALTER TABLE documents ADD COLUMN author_array TEXT[];

-- Step 2: Migrate existing data by splitting on semicolon
-- Handle NULL values and trim whitespace from each author name
UPDATE documents
SET author_array = CASE
    WHEN author IS NULL OR author = '' THEN NULL
    ELSE (
        SELECT array_agg(TRIM(author_name))
        FROM unnest(string_to_array(author, ';')) AS author_name
        WHERE TRIM(author_name) != ''
    )
END;

-- Step 3: Drop the old author column
ALTER TABLE documents DROP COLUMN author;

-- Step 4: Rename the new column to author
ALTER TABLE documents RENAME COLUMN author_array TO author;

-- Step 5: Create GIN index for efficient array searching
CREATE INDEX IF NOT EXISTS idx_documents_author_gin ON documents USING gin(author);

-- Verify migration
SELECT
    doc_id,
    ext_id,
    author,
    array_length(author, 1) as author_count
FROM documents
WHERE author IS NOT NULL
LIMIT 5;

COMMIT;

-- Add comment
COMMENT ON COLUMN documents.author IS 'List of authors for the document (PostgreSQL TEXT[] array)';
