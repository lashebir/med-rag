#!/usr/bin/env python3
"""
Test semantic search and Ollama integration end-to-end.
"""
import os
import sys
import json
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

load_dotenv()

# Config
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1:8b-instruct-q4_K_M")

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

def to_vec_lit(vec):
    """Convert numpy array to Postgres vector literal."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

def test_semantic_search(question: str, k: int = 5):
    """Test semantic search retrieval."""
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")

    # 1. Load embedding model
    print(f"[1/4] Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # 2. Encode query
    print(f"[2/4] Encoding query...")
    qvec = embedder.encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    # 3. Retrieve from database
    print(f"[3/4] Retrieving top-{k} chunks from database...")
    sql = """
    SELECT d.ext_id, d.title, d.source_uri, d.source,
           c.chunk_index, c.text, c.metadata,
           (c.embedding <=> %s::vector) AS distance,
           (1 - (c.embedding <=> %s::vector)) AS similarity
    FROM chunks c
    JOIN documents d USING (doc_id)
    WHERE c.embedding_model = %s
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s;
    """

    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, (qlit, qlit, EMBED_MODEL, qlit, k))
        rows = cur.fetchall()

    if not rows:
        print("❌ No results found!")
        return None

    print(f"✅ Found {len(rows)} results\n")

    # Display results
    contexts = []
    for i, r in enumerate(rows, 1):
        ext_id = r.get("ext_id", "")
        if "pmcid://" in ext_id:
            doc_id = ext_id.split("pmcid://", 1)[-1]
        elif "arxiv://" in ext_id:
            doc_id = ext_id.split("arxiv://", 1)[-1]
        else:
            doc_id = ext_id or "UNKNOWN"

        metadata = r.get("metadata") or {}
        source = r.get("source", "unknown")

        print(f"[{i}] {source.upper()}: {doc_id} (chunk {r['chunk_index']})")
        print(f"    Similarity: {r['similarity']:.4f} | Distance: {r['distance']:.4f}")
        print(f"    Title: {r['title'][:80]}...")
        print(f"    Text: {r['text'][:150]}...")
        print()

        contexts.append({
            "doc_id": doc_id,
            "source": source,
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "title": r["title"],
            "similarity": float(r["similarity"]),
        })

    # 4. Generate answer with Ollama
    print(f"[4/4] Generating answer with Ollama ({LOCAL_MODEL})...")

    # Build context string
    ctx_parts = []
    for i, c in enumerate(contexts, 1):
        label = f"{c['source'].upper()}:{c['doc_id']}#{c['chunk_index']}"
        ctx_parts.append(f"[{label}]\n{c['text']}")
    context_block = "\n\n".join(ctx_parts)

    # Build prompt
    system = (
        "You are a biomedical assistant. Answer ONLY using the provided CONTEXT.\n"
        "- Cite each claim with bracketed refs like [SOURCE:ID#chunk].\n"
        "- If evidence is insufficient, say you don't know.\n"
        "- Be concise and factual."
    )

    prompt = f"""<s>[SYSTEM]
{system}
[/SYSTEM]

[CONTEXT]
{context_block}
[/CONTEXT]

[QUESTION]
{question}
[/QUESTION]

Answer with citations:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LOCAL_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()

        print(f"\n{'='*80}")
        print("ANSWER:")
        print(f"{'='*80}")
        print(answer)
        print(f"{'='*80}\n")

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "num_contexts": len(contexts),
        }

    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None

if __name__ == "__main__":
    # Test questions
    test_questions = [
        "What are the mechanisms of cochlear synaptopathy?",
        "How does noise exposure affect hearing?",
        "What is the role of mitochondria in hearing loss?",
    ]

    # Run test with first question (or custom from command line)
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = test_questions[0]
        print(f"\nUsing default test question. To use your own:")
        print(f"  python test_semantic_search.py 'your question here'\n")

    result = test_semantic_search(question, k=5)

    if result:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
