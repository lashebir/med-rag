#!/usr/bin/env python3
"""
NER Backfill Script

Extracts named entities from existing chunks and populates the mentions table.
This script processes chunks that were ingested before NER integration.

Usage:
    python app/backfill_ner.py
    python app/backfill_ner.py --batch-size 100 --skip-existing
    python app/backfill_ner.py --doc-limit 10  # Process only 10 documents for testing
"""

import os
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from psycopg import connect
from psycopg.rows import dict_row
import time

# Import NER functions
from ingestion.NER.ner_extractor import extract_entities_from_chunks, deduplicate_mentions, embed_mentions
from ingestion.NER.ner_db_utils import upsert_mentions

load_dotenv()

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def get_documents_without_mentions(skip_existing: bool = True) -> List[int]:
    """
    Get list of doc_ids that need NER processing.

    Args:
        skip_existing: If True, skip documents that already have mentions

    Returns:
        List of doc_ids to process
    """
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        if skip_existing:
            # Documents with chunks but no mentions
            cur.execute("""
                SELECT DISTINCT c.doc_id
                FROM chunks c
                WHERE c.embedding_model = %s
                AND NOT EXISTS (
                    SELECT 1 FROM mentions m WHERE m.doc_id = c.doc_id
                )
                ORDER BY c.doc_id;
            """, (EMBED_MODEL,))
        else:
            # All documents with chunks
            cur.execute("""
                SELECT DISTINCT c.doc_id
                FROM chunks c
                WHERE c.embedding_model = %s
                ORDER BY c.doc_id;
            """, (EMBED_MODEL,))

        return [row[0] for row in cur.fetchall()]


def get_chunks_for_document(doc_id: int) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get all chunks for a document, grouped by section.

    Args:
        doc_id: Document ID

    Returns:
        Dictionary mapping section_name -> list of (text, section_name) tuples
    """
    with connect(**PG_KWARGS, row_factory=dict_row) as con, con.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text, section_name
            FROM chunks
            WHERE doc_id = %s
            AND embedding_model = %s
            AND text IS NOT NULL
            ORDER BY chunk_index;
        """, (doc_id, EMBED_MODEL))

        rows = cur.fetchall()

    # Group by section
    sections = defaultdict(list)
    for row in rows:
        text = row['text']
        section_name = row['section_name'] or 'Document'
        sections[section_name].append((text, section_name))

    return dict(sections)


def process_document(doc_id: int, verbose: bool = False) -> Tuple[int, int]:
    """
    Extract NER entities from all chunks in a document.

    Args:
        doc_id: Document ID to process
        verbose: Print detailed progress

    Returns:
        Tuple of (num_sections_processed, num_mentions_inserted)
    """
    # Get chunks grouped by section
    sections = get_chunks_for_document(doc_id)

    if not sections:
        if verbose:
            print(f"  No chunks found for doc {doc_id}")
        return 0, 0

    total_mentions = 0

    for section_name, chunk_tuples in sections.items():
        if verbose:
            print(f"  Processing section: {section_name} ({len(chunk_tuples)} chunks)")

        # Extract entities from all chunks in this section
        mentions = extract_entities_from_chunks(chunk_tuples)

        if not mentions:
            continue

        # Deduplicate
        mentions = deduplicate_mentions(mentions)

        if verbose:
            print(f"    Found {len(mentions)} unique mentions")

        # Generate embeddings
        mention_embeddings = embed_mentions(mentions)

        # Insert into database
        with connect(**PG_KWARGS) as con, con.cursor() as cur:
            num_inserted = upsert_mentions(cur, doc_id, mentions, mention_embeddings)
            con.commit()
            total_mentions += num_inserted

        if verbose:
            print(f"    Inserted {num_inserted} mentions")

    return len(sections), total_mentions


def run_backfill(
    batch_size: int = 50,
    doc_limit: int = None,
    skip_existing: bool = True,
    verbose: bool = True
):
    """
    Run NER backfill on existing chunks.

    Args:
        batch_size: Number of documents to process per batch
        doc_limit: Maximum number of documents to process (for testing)
        skip_existing: Skip documents that already have mentions
        verbose: Print detailed progress
    """
    print("=" * 80)
    print("NER Backfill Script")
    print("=" * 80)

    # Get documents to process
    print("\n📊 Finding documents to process...")
    doc_ids = get_documents_without_mentions(skip_existing=skip_existing)

    if doc_limit:
        doc_ids = doc_ids[:doc_limit]

    print(f"   Found {len(doc_ids)} documents to process")

    if not doc_ids:
        print("\n✅ No documents need processing!")
        return

    print(f"   Batch size: {batch_size}")
    if doc_limit:
        print(f"   Limit: {doc_limit} documents (testing mode)")

    # Process documents
    print("\n🔍 Processing documents...")
    print("-" * 80)

    total_docs_processed = 0
    total_sections_processed = 0
    total_mentions_inserted = 0
    failed_docs = []

    start_time = time.time()

    for i, doc_id in enumerate(doc_ids, 1):
        try:
            if verbose:
                print(f"\n[{i}/{len(doc_ids)}] Processing doc_id: {doc_id}")

            num_sections, num_mentions = process_document(doc_id, verbose=verbose)

            total_docs_processed += 1
            total_sections_processed += num_sections
            total_mentions_inserted += num_mentions

            if not verbose and i % 10 == 0:
                elapsed = time.time() - start_time
                docs_per_sec = i / elapsed
                eta_seconds = (len(doc_ids) - i) / docs_per_sec if docs_per_sec > 0 else 0
                print(f"  Progress: {i}/{len(doc_ids)} docs "
                      f"({docs_per_sec:.1f} docs/sec, "
                      f"ETA: {eta_seconds/60:.1f} min)")

        except Exception as e:
            print(f"  ❌ Error processing doc {doc_id}: {e}")
            failed_docs.append((doc_id, str(e)))
            continue

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)

    print(f"\n📊 Summary:")
    print(f"   Documents processed: {total_docs_processed}/{len(doc_ids)}")
    print(f"   Sections processed: {total_sections_processed}")
    print(f"   Mentions inserted: {total_mentions_inserted:,}")
    print(f"   Time elapsed: {elapsed/60:.1f} minutes")
    print(f"   Speed: {total_docs_processed/elapsed:.1f} docs/sec")

    if failed_docs:
        print(f"\n⚠️  Failed documents: {len(failed_docs)}")
        for doc_id, error in failed_docs[:5]:
            print(f"   - doc_id {doc_id}: {error[:80]}")
        if len(failed_docs) > 5:
            print(f"   ... and {len(failed_docs) - 5} more")

    # Verify results
    print("\n🔍 Verifying results...")
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Total mentions
        cur.execute("SELECT COUNT(*) FROM mentions;")
        total_mentions = cur.fetchone()[0]

        # Mentions by label
        cur.execute("""
            SELECT label, COUNT(*)
            FROM mentions
            GROUP BY label
            ORDER BY COUNT(*) DESC;
        """)
        label_counts = cur.fetchall()

        # Mention embeddings
        cur.execute("SELECT COUNT(*) FROM mention_embeddings;")
        total_embeddings = cur.fetchone()[0]

        print(f"   Total mentions in database: {total_mentions:,}")
        print(f"   Total mention embeddings: {total_embeddings:,}")
        print(f"\n   Mentions by label:")
        for label, count in label_counts:
            pct = (count / total_mentions * 100) if total_mentions > 0 else 0
            print(f"      {label:<25} {count:>8,} ({pct:>5.1f}%)")

    print("\n✅ Backfill complete!")


def main():
    parser = argparse.ArgumentParser(description="Backfill NER for existing chunks")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of documents to process per batch (default: 50)"
    )
    parser.add_argument(
        "--doc-limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip documents that already have mentions (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess all documents, even those with existing mentions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for each document"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 5 documents with verbose output"
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        print("🧪 TEST MODE: Processing 5 documents with verbose output\n")
        args.doc_limit = 5
        args.verbose = True

    # Run backfill
    run_backfill(
        batch_size=args.batch_size,
        doc_limit=args.doc_limit,
        skip_existing=args.skip_existing,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
