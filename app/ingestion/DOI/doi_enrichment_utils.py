"""
DOI Enrichment Utilities for Ingest Scripts

Provides functions to automatically enrich newly ingested documents with DOIs.
"""

import os
import asyncio
from typing import List, Dict, Tuple, Optional
from psycopg import connect
from psycopg.rows import dict_row
from datetime import datetime

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)


def get_recent_doc_ids(since_timestamp: Optional[datetime] = None, limit: int = 1000) -> List[int]:
    """
    Get recently ingested doc_ids that don't have DOIs yet.

    Args:
        since_timestamp: Only get docs created after this timestamp
        limit: Maximum number of doc_ids to return

    Returns:
        List of doc_ids
    """
    with connect(**PG_KWARGS) as con:
        with con.cursor() as cur:
            if since_timestamp:
                cur.execute("""
                    SELECT d.doc_id
                    FROM documents d
                    LEFT JOIN document_ids di ON di.doc_id = d.doc_id AND di.id_type = 'doi'
                    WHERE di.doc_id IS NULL
                    AND d.created_at >= %s
                    ORDER BY d.doc_id DESC
                    LIMIT %s;
                """, (since_timestamp, limit))
            else:
                cur.execute("""
                    SELECT d.doc_id
                    FROM documents d
                    LEFT JOIN document_ids di ON di.doc_id = d.doc_id AND di.id_type = 'doi'
                    WHERE di.doc_id IS NULL
                    ORDER BY d.doc_id DESC
                    LIMIT %s;
                """, (limit,))

            return [row[0] for row in cur.fetchall()]


def count_dois_for_doc_ids(doc_ids: List[int]) -> Tuple[int, int]:
    """
    Count how many of the given doc_ids have DOIs.

    Args:
        doc_ids: List of doc_ids to check

    Returns:
        Tuple of (journal_doi_count, arxiv_doi_count)
    """
    if not doc_ids:
        return 0, 0

    with connect(**PG_KWARGS) as con:
        with con.cursor() as cur:
            # Count journal DOIs (non-arXiv)
            cur.execute("""
                SELECT COUNT(DISTINCT doc_id)
                FROM document_ids
                WHERE doc_id = ANY(%s)
                AND id_type = 'doi'
                AND id_value NOT LIKE '10.48550/%%';
            """, (doc_ids,))
            journal_count = cur.fetchone()[0]

            # Count arXiv DOIs
            cur.execute("""
                SELECT COUNT(DISTINCT doc_id)
                FROM document_ids
                WHERE doc_id = ANY(%s)
                AND id_type = 'doi'
                AND id_value LIKE '10.48550/%%';
            """, (doc_ids,))
            arxiv_count = cur.fetchone()[0]

            return journal_count, arxiv_count


async def enrich_recent_dois(
    doc_ids: Optional[List[int]] = None,
    limit: int = 1000,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Enrich recently ingested documents with DOIs.

    Args:
        doc_ids: Specific doc_ids to enrich (if None, uses recent docs without DOIs)
        limit: Max number of documents to process
        verbose: Print progress messages

    Returns:
        Dictionary with enrichment statistics
    """
    # Import here to avoid circular imports
    from app.ingestion.DOI.enrich_doi import enrich_row, ensure_document_ids_table, documents_has_doi_column

    if verbose:
        print("\n" + "=" * 70)
        print("🔍 DOI ENRICHMENT")
        print("=" * 70)

    # Get doc_ids to process
    if doc_ids is None:
        if verbose:
            print("📊 Finding recently ingested documents without DOIs...")
        with connect(**PG_KWARGS, row_factory=dict_row) as con:
            with con.cursor() as cur:
                cur.execute("""
                    SELECT d.doc_id, d.source, d.source_id
                    FROM documents d
                    LEFT JOIN document_ids di ON di.doc_id = d.doc_id AND di.id_type = 'doi'
                    WHERE di.doc_id IS NULL
                    ORDER BY d.doc_id DESC
                    LIMIT %s;
                """, (limit,))
                rows = [dict(r) for r in cur.fetchall()]
    else:
        if verbose:
            print(f"📊 Processing {len(doc_ids)} newly ingested documents...")
        with connect(**PG_KWARGS, row_factory=dict_row) as con:
            with con.cursor() as cur:
                cur.execute("""
                    SELECT d.doc_id, d.source, d.source_id
                    FROM documents d
                    WHERE d.doc_id = ANY(%s);
                """, (doc_ids,))
                rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        if verbose:
            print("✅ No documents to process (all have DOIs)")
        return {"processed": 0, "journal_dois": 0, "arxiv_dois": 0}

    if verbose:
        print(f"📄 Processing {len(rows)} documents...")

    # Get initial counts
    initial_journal, initial_arxiv = count_dois_for_doc_ids([r["doc_id"] for r in rows])

    # Setup database
    with connect(**PG_KWARGS) as con:
        write_to_documents = documents_has_doi_column(con)
        if not write_to_documents:
            ensure_document_ids_table(con)

        # Process each document
        processed = 0
        for row in rows:
            try:
                await enrich_row(row, write_to_documents, con)
                processed += 1
            except Exception as e:
                if verbose:
                    print(f"[warn] Failed to enrich doc_id={row['doc_id']}: {e}")

    # Get final counts
    final_journal, final_arxiv = count_dois_for_doc_ids([r["doc_id"] for r in rows])

    # Calculate new DOIs added
    new_journal = final_journal - initial_journal
    new_arxiv = final_arxiv - initial_arxiv

    if verbose:
        print("\n" + "=" * 70)
        print("✅ DOI ENRICHMENT COMPLETE")
        print("=" * 70)
        print(f"📊 Processed: {processed} documents")
        print(f"📰 Journal DOIs added: {new_journal}")
        print(f"📄 arXiv DOIs added: {new_arxiv}")
        print(f"🎯 Total DOIs found: {new_journal + new_arxiv}/{processed} ({(new_journal + new_arxiv)/processed*100:.1f}%)")
        print("=" * 70 + "\n")

    return {
        "processed": processed,
        "journal_dois": new_journal,
        "arxiv_dois": new_arxiv,
        "total_dois": new_journal + new_arxiv
    }


def get_doc_ids_since_timestamp(timestamp: datetime) -> List[int]:
    """
    Get all doc_ids created since a given timestamp.

    Args:
        timestamp: Get docs created after this time

    Returns:
        List of doc_ids
    """
    with connect(**PG_KWARGS) as con:
        with con.cursor() as cur:
            # Check if created_at column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'created_at'
                );
            """)
            has_created_at = cur.fetchone()[0]

            if has_created_at:
                cur.execute("""
                    SELECT doc_id
                    FROM documents
                    WHERE created_at >= %s
                    ORDER BY doc_id DESC;
                """, (timestamp,))
            else:
                # Fallback: assume recent doc_ids are higher
                cur.execute("""
                    SELECT doc_id
                    FROM documents
                    ORDER BY doc_id DESC
                    LIMIT 1000;
                """)

            return [row[0] for row in cur.fetchall()]


def get_max_doc_id() -> Optional[int]:
    """
    Get the current maximum doc_id in the database.

    Returns:
        Max doc_id or None if no documents exist
    """
    with connect(**PG_KWARGS) as con:
        with con.cursor() as cur:
            cur.execute("SELECT MAX(doc_id) FROM documents;")
            result = cur.fetchone()
            return result[0] if result and result[0] else None


def get_new_doc_ids(max_doc_id_before: Optional[int]) -> List[int]:
    """
    Get doc_ids created since the given max_doc_id.

    Args:
        max_doc_id_before: The max doc_id before ingestion started

    Returns:
        List of newly created doc_ids
    """
    if max_doc_id_before is None:
        # First ingestion ever, get all docs
        with connect(**PG_KWARGS) as con:
            with con.cursor() as cur:
                cur.execute("SELECT doc_id FROM documents ORDER BY doc_id DESC LIMIT 1000;")
                return [row[0] for row in cur.fetchall()]

    with connect(**PG_KWARGS) as con:
        with con.cursor() as cur:
            cur.execute("""
                SELECT doc_id
                FROM documents
                WHERE doc_id > %s
                ORDER BY doc_id DESC;
            """, (max_doc_id_before,))
            return [row[0] for row in cur.fetchall()]


async def auto_enrich_dois_after_ingestion(
    max_doc_id_before: Optional[int],
    source_name: str = "documents",
    verbose: bool = True
) -> Dict[str, int]:
    """
    Automatically enrich DOIs for documents ingested since max_doc_id_before.

    This is the main function to call from ingest scripts.

    Args:
        max_doc_id_before: The max doc_id before ingestion started (from get_max_doc_id())
        source_name: Name of the source (for display purposes)
        verbose: Print progress messages

    Returns:
        Dictionary with enrichment statistics

    Example:
        >>> # At start of ingest script
        >>> max_doc_id_before = get_max_doc_id()
        >>> # ... run ingestion ...
        >>> # At end of ingest script
        >>> stats = await auto_enrich_dois_after_ingestion(max_doc_id_before, "arXiv")
    """
    new_doc_ids = get_new_doc_ids(max_doc_id_before)

    if not new_doc_ids:
        if verbose:
            print("\n" + "=" * 70)
            print("🔍 DOI ENRICHMENT")
            print("=" * 70)
            print("✅ No new documents to enrich")
            print("=" * 70 + "\n")
        return {"processed": 0, "journal_dois": 0, "arxiv_dois": 0, "total_dois": 0}

    if verbose:
        print(f"\n🔍 Running DOI enrichment on {len(new_doc_ids)} newly ingested {source_name}...")

    stats = await enrich_recent_dois(doc_ids=new_doc_ids, verbose=verbose)

    # Add a nice summary message
    if verbose and stats["total_dois"] > 0:
        print(f"✨ {stats['total_dois']}/{len(new_doc_ids)} DOIs of the newly ingested sources have been added to your database")

    return stats
