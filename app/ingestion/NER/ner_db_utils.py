"""
Database utilities for storing and retrieving NER mentions and embeddings.
"""
from typing import List, Dict
import json

def to_vec_lit(vec) -> str:
    """Convert numpy array or list to PostgreSQL vector literal."""
    if hasattr(vec, 'tolist'):
        vec = vec.tolist()
    return '[' + ','.join(map(str, vec)) + ']'

def upsert_mentions(
    cur,
    doc_id: int,
    mentions: List[Dict],
    embeddings: List[List[float]]
) -> int:
    """
    Insert mentions and their embeddings into the database.

    Args:
        cur: Database cursor
        doc_id: Document ID to associate mentions with
        mentions: List of mention dictionaries from ner_extractor
        embeddings: List of 384-dimensional embeddings for each mention

    Returns:
        Number of mentions inserted
    """
    if not mentions:
        return 0

    # Insert mentions
    mention_rows = []
    for mention in mentions:
        mention_rows.append((
            doc_id,
            mention['mention_text'],
            mention['label'],
            mention['start_char'],
            mention['end_char'],
            mention['context'],
            mention.get('label_confidence'),
            mention.get('section_name'),
        ))

    # Insert mentions one by one to get RETURNING values
    mention_ids = []
    for row in mention_rows:
        cur.execute(
            """
            INSERT INTO mentions (
                doc_id, mention_text, label, start_char, end_char,
                context, label_confidence, section_name
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING mention_id;
            """,
            row
        )
        result = cur.fetchone()
        if result:
            mention_ids.append(result[0])

    # If no mention_ids returned due to conflicts, we can't insert embeddings
    if not mention_ids:
        print(f"⚠️ No new mentions inserted (likely duplicates)", flush=True)
        return 0

    # Insert embeddings
    embedding_rows = []
    for mention_id, embedding in zip(mention_ids, embeddings[:len(mention_ids)]):
        embedding_rows.append((
            mention_id,
            to_vec_lit(embedding)
        ))

    if embedding_rows:
        cur.executemany(
            """
            INSERT INTO mention_embeddings (mention_id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (mention_id) DO UPDATE
            SET embedding = EXCLUDED.embedding;
            """,
            embedding_rows
        )

    return len(mention_ids)

def get_mentions_for_document(cur, doc_id: int) -> List[Dict]:
    """
    Retrieve all mentions for a given document.

    Args:
        cur: Database cursor
        doc_id: Document ID

    Returns:
        List of mention dictionaries
    """
    cur.execute(
        """
        SELECT
            m.mention_id,
            m.mention_text,
            m.label,
            m.start_char,
            m.end_char,
            m.context,
            m.label_confidence,
            m.section_name,
            m.created_at
        FROM mentions m
        WHERE m.doc_id = %s
        ORDER BY m.start_char;
        """,
        (doc_id,)
    )

    mentions = []
    for row in cur.fetchall():
        mentions.append({
            'mention_id': row[0],
            'mention_text': row[1],
            'label': row[2],
            'start_char': row[3],
            'end_char': row[4],
            'context': row[5],
            'label_confidence': row[6],
            'section_name': row[7],
            'created_at': row[8],
        })

    return mentions

def get_mentions_by_label(cur, label: str, limit: int = 100) -> List[Dict]:
    """
    Retrieve mentions by label type.

    Args:
        cur: Database cursor
        label: One of 'disease', 'drug', 'device', 'statistical_significance'
        limit: Maximum number of results

    Returns:
        List of mention dictionaries
    """
    cur.execute(
        """
        SELECT
            m.mention_id,
            m.doc_id,
            m.mention_text,
            m.label,
            m.start_char,
            m.end_char,
            m.context,
            m.label_confidence,
            m.section_name,
            m.created_at
        FROM mentions m
        WHERE m.label = %s
        ORDER BY m.created_at DESC
        LIMIT %s;
        """,
        (label, limit)
    )

    mentions = []
    for row in cur.fetchall():
        mentions.append({
            'mention_id': row[0],
            'doc_id': row[1],
            'mention_text': row[2],
            'label': row[3],
            'start_char': row[4],
            'end_char': row[5],
            'context': row[6],
            'label_confidence': row[7],
            'section_name': row[8],
            'created_at': row[9],
        })

    return mentions

def search_similar_mentions(
    cur,
    query_embedding: List[float],
    label: str = None,
    top_k: int = 10
) -> List[Dict]:
    """
    Search for similar mentions using vector similarity.

    Args:
        cur: Database cursor
        query_embedding: 384-dimensional query embedding
        label: Optional filter by label type
        top_k: Number of results to return

    Returns:
        List of mention dictionaries with similarity scores
    """
    query_vec = to_vec_lit(query_embedding)

    if label:
        cur.execute(
            """
            SELECT
                m.mention_id,
                m.doc_id,
                m.mention_text,
                m.label,
                m.context,
                m.section_name,
                1 - (me.embedding <=> %s::vector) as similarity
            FROM mentions m
            JOIN mention_embeddings me ON m.mention_id = me.mention_id
            WHERE m.label = %s
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_vec, label, query_vec, top_k)
        )
    else:
        cur.execute(
            """
            SELECT
                m.mention_id,
                m.doc_id,
                m.mention_text,
                m.label,
                m.context,
                m.section_name,
                1 - (me.embedding <=> %s::vector) as similarity
            FROM mentions m
            JOIN mention_embeddings me ON m.mention_id = me.mention_id
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_vec, query_vec, top_k)
        )

    results = []
    for row in cur.fetchall():
        results.append({
            'mention_id': row[0],
            'doc_id': row[1],
            'mention_text': row[2],
            'label': row[3],
            'context': row[4],
            'section_name': row[5],
            'similarity': row[6],
        })

    return results

def get_mention_stats(cur) -> Dict:
    """
    Get statistics about mentions in the database.

    Returns:
        Dictionary with counts by label type
    """
    cur.execute(
        """
        SELECT label, COUNT(*) as count
        FROM mentions
        GROUP BY label
        ORDER BY count DESC;
        """
    )

    stats = {}
    for row in cur.fetchall():
        stats[row[0]] = row[1]

    # Get total
    cur.execute("SELECT COUNT(*) FROM mentions;")
    stats['total'] = cur.fetchone()[0]

    return stats
