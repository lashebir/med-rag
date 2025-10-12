from psycopg import connect
from dotenv import load_dotenv
import os

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doc_id BIGSERIAL PRIMARY KEY,
  ext_id TEXT UNIQUE,
  title  TEXT,
  source_uri TEXT,
  author TEXT,
  year DATE,
  journal TEXT,
  institute TEXT,
  tags TEXT[],
  captured_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id BIGSERIAL PRIMARY KEY,
  doc_id   BIGINT REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  text     TEXT,
  metadata JSONB,
  embedding VECTOR(384),
  embedding_model TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=200);
"""
# Load environment variables from .env file
load_dotenv()
pg_host = os.getenv("PGHOST", "localhost")
pg_port = int(os.getenv("PGPORT", "5432"))
pg_db   = os.getenv("PGDATABASE", "pubmed")     # <- your DB name
pg_user = os.getenv("PGUSER", os.getenv("USER"))
pg_pass = os.getenv("PGPASSWORD") or None

with connect(host=pg_host, port=pg_port, dbname=pg_db, user=pg_user, password=pg_pass) as con, con.cursor() as cur:
    cur.execute(DDL)
    con.commit()
print("✅ Schema applied successfully.")