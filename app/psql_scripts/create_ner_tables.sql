-- Create mentions table with discrete label categories
CREATE TABLE IF NOT EXISTS mentions (
    mention_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    mention_text TEXT NOT NULL,
    label VARCHAR(50) NOT NULL CHECK (label IN ('disease', 'statistical_significance', 'device', 'drug')),
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    context TEXT NOT NULL,  -- full sentence where mention was found
    label_confidence FLOAT,  -- confidence score from NER model
    section_name TEXT,  -- section where this mention was found
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create mention_embeddings table
CREATE TABLE IF NOT EXISTS mention_embeddings (
    mention_id INTEGER PRIMARY KEY REFERENCES mentions(mention_id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,  -- 384-dimensional embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_mentions_doc_id ON mentions(doc_id);
CREATE INDEX IF NOT EXISTS idx_mentions_label ON mentions(label);
CREATE INDEX IF NOT EXISTS idx_mentions_section_name ON mentions(section_name);
CREATE INDEX IF NOT EXISTS idx_mentions_created_at ON mentions(created_at);
CREATE INDEX IF NOT EXISTS idx_mentions_text ON mentions USING gin(to_tsvector('english', mention_text));

-- Create vector similarity index for mention embeddings
CREATE INDEX IF NOT EXISTS idx_mention_embeddings_vector
    ON mention_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Add comments for documentation
COMMENT ON TABLE mentions IS 'Named entities extracted from scientific documents';
COMMENT ON TABLE mention_embeddings IS 'Vector embeddings for named entities to enable semantic search';
COMMENT ON COLUMN mentions.label IS 'Entity type: disease, statistical_significance, device, or drug (discrete categories only)';
COMMENT ON COLUMN mentions.context IS 'Full sentence containing the mention for context';
COMMENT ON COLUMN mentions.section_name IS 'Document section where mention was found (e.g., Introduction, Methods, Results)';
