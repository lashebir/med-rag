# app/llm_integration/search_strategies.py
"""
Different search strategies for comparing retrieval performance:

1. IVFFlat Only: Pure vector similarity (baseline)
2. IVFFlat + NER Boost: Vector similarity with soft boost from entity mentions
3. IVFFlat + tsvector: Hybrid vector + full-text search (no NER)

All strategies use the same IVFFlat index (nlist=96, nprobe=8).
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration
PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))

# Embedder singleton
_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def to_vec_lit(vec) -> str:
    """Convert vector to PostgreSQL literal."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

def _format_context(r: Dict) -> Dict[str, Any]:
    """Format database row into context dictionary."""
    # Handle different ext_id formats
    ext_id = r.get("ext_id", "")
    if "pmcid://" in ext_id:
        doc_id = ext_id.split("pmcid://", 1)[-1].upper()
    elif "arxiv://" in ext_id:
        doc_id = ext_id.split("arxiv://", 1)[-1]
    elif "scholar://" in ext_id:
        doc_id = ext_id.split("scholar://", 1)[-1]
    else:
        doc_id = ext_id or "UNKNOWN"

    # Extract year
    year = None
    if r.get("year"):
        try:
            year = r["year"].year if hasattr(r["year"], "year") else int(r["year"])
        except:
            pass

    return {
        "pmcid": doc_id,
        "chunk_id": r.get("chunk_id"),
        "chunk_index": r["chunk_index"],
        "text": r["text"],
        "title": r["title"],
        "source_uri": r["source_uri"],
        "source": r.get("source"),
        "authors": r.get("author"),
        "year": year,
        "institution": r.get("institute"),
        "distance": float(r["distance"]),
        "similarity": float(r["similarity"]),
        "final_score": float(r.get("final_score", r["similarity"])),
    }


# ============================================================================
# Strategy 1: IVFFlat Only (Baseline)
# ============================================================================

def retrieve_ivfflat_only(
    question: str,
    k: int = 10,
    embedding_model: str = EMBED_MODEL
) -> List[Dict[str, Any]]:
    """
    Strategy 1: Pure vector similarity search using IVFFlat index.

    This is the baseline strategy that uses only semantic similarity
    without any additional signals.

    Args:
        question: Query text
        k: Number of results to return
        embedding_model: Model name for filtering

    Returns:
        List of context dictionaries with similarity scores
    """
    # Generate query embedding
    qvec = embedder().encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    sql = """
    SELECT d.ext_id, d.title, d.source_uri, d.source, d.author, d.year, d.institute,
           c.chunk_id, c.chunk_index, c.text,
           (c.embedding <=> %s::vector) AS distance,
           (1 - (c.embedding <=> %s::vector)) AS similarity
    FROM chunks c
    JOIN documents d USING (doc_id)
    WHERE c.embedding_model = %s
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s;
    """

    params = [qlit, qlit, embedding_model, qlit, k]

    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [_format_context(r) for r in rows]


# ============================================================================
# Strategy 2: IVFFlat + NER Soft Boost
# ============================================================================

def retrieve_ivfflat_ner_boost(
    question: str,
    k: int = 10,
    embedding_model: str = EMBED_MODEL,
    ner_boost_weight: float = 0.15,
    initial_k_multiplier: int = 3
) -> List[Dict[str, Any]]:
    """
    Strategy 2: Vector similarity with soft boost from NER entity mentions.

    This strategy:
    1. Retrieves initial candidates using vector similarity (k * multiplier)
    2. Checks which chunks have entity mentions matching the query
    3. Applies a soft boost to chunks with matching entities
    4. Re-ranks and returns top-k

    The boost is additive: final_score = similarity + (ner_boost_weight * ner_match_score)

    Args:
        question: Query text
        k: Number of final results to return
        embedding_model: Model name for filtering
        ner_boost_weight: Weight for NER boost (0-1), default 0.15
        initial_k_multiplier: Fetch k*multiplier candidates before re-ranking

    Returns:
        List of context dictionaries with boosted scores
    """
    # Generate query embedding
    qvec = embedder().encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    initial_k = k * initial_k_multiplier

    # Step 1: Get initial candidates with NER mention counts
    sql = """
    WITH candidates AS (
        SELECT
            d.ext_id, d.title, d.source_uri, d.source, d.author, d.year, d.institute,
            c.chunk_id, c.chunk_index, c.text,
            (c.embedding <=> %s::vector) AS distance,
            (1 - (c.embedding <=> %s::vector)) AS similarity
        FROM chunks c
        JOIN documents d USING (doc_id)
        WHERE c.embedding_model = %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    )
    SELECT
        candidates.*,
        COUNT(m.mention_id) AS mention_count,
        COALESCE(ARRAY_AGG(DISTINCT m.label) FILTER (WHERE m.label IS NOT NULL), ARRAY[]::varchar[]) AS entity_types
    FROM candidates
    LEFT JOIN mentions m ON m.doc_id = (
        SELECT doc_id FROM chunks WHERE chunk_id = candidates.chunk_id
    ) AND m.section_name = (
        SELECT section_name FROM chunks WHERE chunk_id = candidates.chunk_id
    )
    GROUP BY candidates.chunk_id, candidates.ext_id, candidates.title, candidates.source_uri,
             candidates.source, candidates.author, candidates.year, candidates.institute,
             candidates.chunk_index, candidates.text, candidates.distance, candidates.similarity
    ORDER BY candidates.similarity DESC;
    """

    params = [qlit, qlit, embedding_model, qlit, initial_k]

    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Step 2: Apply soft NER boost
    results = []
    for r in rows:
        mention_count = r["mention_count"]

        # Calculate NER match score (0-1 scale)
        # More mentions = higher boost, but with diminishing returns
        if mention_count > 0:
            # Logarithmic scaling: log(1 + mentions) / log(1 + 10)
            # This gives ~0.5 for 3 mentions, ~0.8 for 9 mentions, ~1.0 for 10+ mentions
            import math
            ner_match_score = min(1.0, math.log(1 + mention_count) / math.log(11))
        else:
            ner_match_score = 0.0

        # Calculate final score with soft boost
        similarity = float(r["similarity"])
        final_score = similarity + (ner_boost_weight * ner_match_score)

        # Add boosting metadata
        context = _format_context(r)
        context["final_score"] = final_score
        context["ner_mention_count"] = mention_count
        context["ner_boost_applied"] = ner_boost_weight * ner_match_score
        context["entity_types"] = r["entity_types"]

        results.append(context)

    # Step 3: Re-rank by final score and return top-k
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:k]


# ============================================================================
# Strategy 3: IVFFlat + tsvector (Hybrid)
# ============================================================================

def retrieve_ivfflat_tsvector(
    question: str,
    k: int = 10,
    embedding_model: str = EMBED_MODEL,
    rrf_k: int = 60,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    use_rrf: bool = True
) -> List[Dict[str, Any]]:
    """
    Strategy 3: Hybrid search combining vector similarity + full-text search.

    Two methods available:
    1. RRF (Reciprocal Rank Fusion): Combines rankings from both searches
       final_score = 1/(rrf_k + vector_rank) + 1/(rrf_k + text_rank)

    2. Weighted scoring: Combines normalized scores
       final_score = vector_weight * similarity + text_weight * text_rank

    Args:
        question: Query text
        k: Number of results to return
        embedding_model: Model name for filtering
        rrf_k: RRF constant (typically 60)
        vector_weight: Weight for vector similarity (0-1)
        text_weight: Weight for text search (0-1)
        use_rrf: If True, use RRF; otherwise use weighted scoring

    Returns:
        List of context dictionaries with hybrid scores
    """
    # Generate query embedding
    qvec = embedder().encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    # Prepare text search query (simple tokenization)
    # Convert question to tsquery format
    search_terms = question.strip().split()[:10]  # Limit to 10 terms
    tsquery = " & ".join(search_terms)

    if use_rrf:
        # RRF Method: Combine rankings
        sql = """
        WITH vector_search AS (
            SELECT
                c.chunk_id,
                ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS vector_rank,
                (1 - (c.embedding <=> %s::vector)) AS similarity
            FROM chunks c
            WHERE c.embedding_model = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        ),
        text_search AS (
            SELECT
                c.chunk_id,
                ROW_NUMBER() OVER (ORDER BY ts_rank(c.text_tsv, to_tsquery('english', %s)) DESC) AS text_rank,
                ts_rank(c.text_tsv, to_tsquery('english', %s)) AS text_score
            FROM chunks c
            WHERE c.text_tsv @@ to_tsquery('english', %s)
            AND c.embedding_model = %s
            ORDER BY text_rank
            LIMIT %s
        ),
        combined AS (
            SELECT
                COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
                COALESCE(v.vector_rank, %s + %s) AS vector_rank,
                COALESCE(t.text_rank, %s + %s) AS text_rank,
                COALESCE(v.similarity, 0) AS similarity,
                COALESCE(t.text_score, 0) AS text_score,
                (1.0 / (%s + COALESCE(v.vector_rank, %s + %s))) +
                (1.0 / (%s + COALESCE(t.text_rank, %s + %s))) AS rrf_score
            FROM vector_search v
            FULL OUTER JOIN text_search t ON v.chunk_id = t.chunk_id
        )
        SELECT
            d.ext_id, d.title, d.source_uri, d.source, d.author, d.year, d.institute,
            c.chunk_id, c.chunk_index, c.text,
            combined.similarity AS similarity,
            (1 - combined.similarity) AS distance,
            combined.rrf_score AS final_score,
            combined.vector_rank,
            combined.text_rank,
            combined.text_score
        FROM combined
        JOIN chunks c ON c.chunk_id = combined.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY combined.rrf_score DESC
        LIMIT %s;
        """

        # Parameters for RRF
        fetch_limit = k * 3
        params = [
            qlit, qlit, embedding_model, qlit, fetch_limit,  # vector_search
            tsquery, tsquery, tsquery, embedding_model, fetch_limit,  # text_search
            fetch_limit, 1, fetch_limit, 1,  # COALESCE defaults
            rrf_k, fetch_limit, 1, rrf_k, fetch_limit, 1,  # RRF calculation
            k  # final LIMIT
        ]
    else:
        # Weighted Scoring Method
        sql = """
        WITH vector_search AS (
            SELECT
                c.chunk_id,
                (1 - (c.embedding <=> %s::vector)) AS similarity,
                ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS vector_rank
            FROM chunks c
            WHERE c.embedding_model = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        ),
        text_search AS (
            SELECT
                c.chunk_id,
                ts_rank(c.text_tsv, to_tsquery('english', %s)) AS text_score,
                ROW_NUMBER() OVER (ORDER BY ts_rank(c.text_tsv, to_tsquery('english', %s)) DESC) AS text_rank
            FROM chunks c
            WHERE c.text_tsv @@ to_tsquery('english', %s)
            AND c.embedding_model = %s
            ORDER BY text_rank
            LIMIT %s
        ),
        text_max AS (
            SELECT MAX(text_score) AS max_score FROM text_search WHERE text_score > 0
        ),
        combined AS (
            SELECT
                COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
                COALESCE(v.similarity, 0) AS similarity,
                COALESCE(v.vector_rank, 999999) AS vector_rank,
                COALESCE(t.text_rank, 999999) AS text_rank,
                COALESCE(t.text_score, 0) / NULLIF((SELECT max_score FROM text_max), 0) AS normalized_text_score,
                (%s * COALESCE(v.similarity, 0)) +
                (%s * COALESCE(t.text_score, 0) / NULLIF((SELECT max_score FROM text_max), 0)) AS final_score
            FROM vector_search v
            FULL OUTER JOIN text_search t ON v.chunk_id = t.chunk_id
        )
        SELECT
            d.ext_id, d.title, d.source_uri, d.source, d.author, d.year, d.institute,
            c.chunk_id, c.chunk_index, c.text,
            combined.similarity AS similarity,
            (1 - combined.similarity) AS distance,
            combined.final_score AS final_score,
            combined.vector_rank,
            combined.text_rank,
            combined.normalized_text_score AS text_score
        FROM combined
        JOIN chunks c ON c.chunk_id = combined.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY combined.final_score DESC
        LIMIT %s;
        """

        fetch_limit = k * 3
        params = [
            qlit, qlit, embedding_model, qlit, fetch_limit,  # vector_search
            tsquery, tsquery, tsquery, embedding_model, fetch_limit,  # text_search
            vector_weight, text_weight,  # weights
            k  # final LIMIT
        ]

    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Format results with hybrid metadata
    results = []
    for r in rows:
        context = _format_context(r)
        context["vector_rank"] = r.get("vector_rank")
        context["text_rank"] = r.get("text_rank")
        context["text_score"] = float(r.get("text_score", 0))
        results.append(context)

    return results


# ============================================================================
# Strategy 4: IVFFlat + NER + tsvector (Full Hybrid)
# ============================================================================

def retrieve_ivfflat_ner_tsvector(
    question: str,
    k: int = 10,
    embedding_model: str = EMBED_MODEL,
    ner_boost_weight: float = 0.10,
    vector_weight: float = 0.5,
    text_weight: float = 0.3,
    ner_weight: float = 0.2,
    initial_k_multiplier: int = 3
) -> List[Dict[str, Any]]:
    """
    Strategy 4: Full hybrid search combining vector + NER + full-text.

    This combines all three signals:
    1. Vector similarity (semantic)
    2. NER entity mentions (domain relevance)
    3. Full-text keyword matching (precision)

    Formula:
        final_score = (vector_weight * similarity) +
                     (text_weight * normalized_text_score) +
                     (ner_weight * ner_match_score)

    Args:
        question: Query text
        k: Number of final results to return
        embedding_model: Model name for filtering
        ner_boost_weight: Deprecated, use ner_weight instead
        vector_weight: Weight for vector similarity (0-1)
        text_weight: Weight for full-text search (0-1)
        ner_weight: Weight for NER boost (0-1)
        initial_k_multiplier: Fetch k*multiplier candidates

    Returns:
        List of context dictionaries with full hybrid scores
    """
    # Generate query embedding
    qvec = embedder().encode([question], normalize_embeddings=True)[0]
    qlit = to_vec_lit(qvec)

    initial_k = k * initial_k_multiplier

    # Prepare text search query
    search_terms = question.strip().split()[:10]
    tsquery = " & ".join(search_terms)

    # Combined query: get vector search + text search + NER counts
    sql = """
    WITH vector_search AS (
        SELECT
            c.chunk_id,
            c.doc_id,
            (1 - (c.embedding <=> %s::vector)) AS similarity,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS vector_rank
        FROM chunks c
        WHERE c.embedding_model = %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    ),
    text_search AS (
        SELECT
            c.chunk_id,
            ts_rank(c.text_tsv, to_tsquery('english', %s)) AS text_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank(c.text_tsv, to_tsquery('english', %s)) DESC) AS text_rank
        FROM chunks c
        WHERE c.text_tsv @@ to_tsquery('english', %s)
        AND c.embedding_model = %s
        ORDER BY text_rank
        LIMIT %s
    ),
    ner_counts AS (
        SELECT
            c.chunk_id,
            COUNT(m.mention_id) AS mention_count,
            COALESCE(ARRAY_AGG(DISTINCT m.label) FILTER (WHERE m.label IS NOT NULL), ARRAY[]::varchar[]) AS entity_types
        FROM chunks c
        LEFT JOIN mentions m ON m.doc_id = c.doc_id AND m.section_name = c.section_name
        WHERE c.embedding_model = %s
        GROUP BY c.chunk_id
    ),
    text_max AS (
        SELECT MAX(text_score) AS max_score FROM text_search WHERE text_score > 0
    ),
    combined AS (
        SELECT
            COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
            COALESCE(v.doc_id, (SELECT doc_id FROM chunks WHERE chunk_id = t.chunk_id)) AS doc_id,
            COALESCE(v.similarity, 0) AS similarity,
            COALESCE(v.vector_rank, 999999) AS vector_rank,
            COALESCE(t.text_rank, 999999) AS text_rank,
            COALESCE(t.text_score, 0) / NULLIF((SELECT max_score FROM text_max), 0) AS normalized_text_score,
            COALESCE(n.mention_count, 0) AS mention_count,
            COALESCE(n.entity_types, ARRAY[]::varchar[]) AS entity_types,
            -- Calculate NER match score with log scaling
            CASE
                WHEN COALESCE(n.mention_count, 0) > 0
                THEN LEAST(1.0, LOG(1 + COALESCE(n.mention_count, 0)) / LOG(11))
                ELSE 0.0
            END AS ner_match_score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.chunk_id = t.chunk_id
        LEFT JOIN ner_counts n ON COALESCE(v.chunk_id, t.chunk_id) = n.chunk_id
    ),
    scored AS (
        SELECT
            combined.*,
            (%s * COALESCE(combined.similarity, 0)) +
            (%s * COALESCE(combined.normalized_text_score, 0)) +
            (%s * COALESCE(combined.ner_match_score, 0)) AS final_score
        FROM combined
    )
    SELECT
        d.ext_id, d.title, d.source_uri, d.source, d.author, d.year, d.institute,
        c.chunk_id, c.chunk_index, c.text,
        scored.similarity,
        (1 - scored.similarity) AS distance,
        scored.final_score,
        scored.vector_rank,
        scored.text_rank,
        scored.normalized_text_score AS text_score,
        scored.mention_count,
        scored.entity_types,
        scored.ner_match_score
    FROM scored
    JOIN chunks c ON c.chunk_id = scored.chunk_id
    JOIN documents d ON d.doc_id = scored.doc_id
    ORDER BY scored.final_score DESC
    LIMIT %s;
    """

    fetch_limit = k * initial_k_multiplier
    params = [
        qlit, qlit, embedding_model, qlit, fetch_limit,  # vector_search
        tsquery, tsquery, tsquery, embedding_model, fetch_limit,  # text_search
        embedding_model,  # ner_counts
        vector_weight, text_weight, ner_weight,  # weights
        k  # final LIMIT
    ]

    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Format results with full hybrid metadata
    results = []
    for r in rows:
        context = _format_context(r)
        context["vector_rank"] = r.get("vector_rank")
        context["text_rank"] = r.get("text_rank")
        context["text_score"] = float(r.get("text_score", 0))
        context["ner_mention_count"] = r.get("mention_count")
        context["ner_match_score"] = float(r.get("ner_match_score", 0))
        context["entity_types"] = r.get("entity_types", [])
        results.append(context)

    return results


# ============================================================================
# Convenience function for testing all strategies
# ============================================================================

def compare_all_strategies(
    question: str,
    k: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all four strategies and return results for comparison.

    Returns:
        Dictionary with keys: 'ivfflat_only', 'ivfflat_ner_boost', 'ivfflat_tsvector', 'ivfflat_ner_tsvector'
    """
    return {
        "ivfflat_only": retrieve_ivfflat_only(question, k),
        "ivfflat_ner_boost": retrieve_ivfflat_ner_boost(question, k),
        "ivfflat_tsvector": retrieve_ivfflat_tsvector(question, k),
        "ivfflat_ner_tsvector": retrieve_ivfflat_ner_tsvector(question, k),
    }
