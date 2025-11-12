#!/usr/bin/env python3
"""
Apply IVFFlat Configuration

Rebuilds IVFFlat indexes with specified nlist and sets default nprobe.

Usage:
    python app/apply_ivfflat_config.py --nlist 64
    python app/apply_ivfflat_config.py --nlist 96 --nprobe 4 --dry-run
"""

import os
import time
import argparse
from dotenv import load_dotenv
from psycopg import connect

load_dotenv()

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

def get_chunk_count() -> int:
    """Get number of chunks for the embedding model."""
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding_model = %s;",
            (EMBED_MODEL,)
        )
        return cur.fetchone()[0]

def get_existing_indexes() -> list:
    """Get list of existing IVFFlat indexes on chunks.embedding."""
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'chunks'
            AND indexdef LIKE '%ivfflat%'
            AND indexdef LIKE '%embedding%';
        """)
        return cur.fetchall()

def drop_existing_indexes(dry_run: bool = False):
    """Drop all existing IVFFlat indexes."""
    print("\n🗑️  Dropping existing IVFFlat indexes...")

    indexes = get_existing_indexes()

    if not indexes:
        print("   No existing IVFFlat indexes found")
        return

    for idx_name, idx_def in indexes:
        print(f"   - {idx_name}")
        if not dry_run:
            with connect(**PG_KWARGS) as con, con.cursor() as cur:
                cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
                con.commit()

    if dry_run:
        print(f"   [DRY RUN] Would drop {len(indexes)} index(es)")
    else:
        print(f"   ✅ Dropped {len(indexes)} index(es)")

def create_ivfflat_index(nlist: int, dry_run: bool = False):
    """Create IVFFlat index with specified nlist."""
    index_name = f"idx_chunks_embedding_ivfflat_nlist{nlist}"

    print(f"\n🔨 Creating IVFFlat index...")
    print(f"   Index name: {index_name}")
    print(f"   nlist: {nlist}")

    if dry_run:
        print(f"   [DRY RUN] Would create index with nlist={nlist}")
        return

    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        start = time.time()

        cur.execute(f"""
            CREATE INDEX {index_name}
            ON chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {nlist})
            WHERE embedding_model = '{EMBED_MODEL}';
        """)

        con.commit()
        elapsed = time.time() - start

    print(f"   ✅ Index created in {elapsed:.2f} seconds")

def set_default_nprobe(nprobe: int, dry_run: bool = False):
    """Set default nprobe value in postgresql.conf or via ALTER DATABASE."""
    print(f"\n⚙️  Setting default nprobe...")
    print(f"   nprobe: {nprobe}")

    if dry_run:
        print(f"   [DRY RUN] Would set ivfflat.probes = {nprobe}")
        return

    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        # Set at database level
        dbname = PG_KWARGS['dbname']
        cur.execute(f"ALTER DATABASE {dbname} SET ivfflat.probes = {nprobe};")
        con.commit()

    print(f"   ✅ Set ivfflat.probes = {nprobe} for database '{dbname}'")
    print(f"   Note: Reconnect to database for changes to take effect")

def verify_configuration(nlist: int, nprobe: int):
    """Verify the configuration was applied correctly."""
    print(f"\n🔍 Verifying configuration...")

    # Check index exists
    indexes = get_existing_indexes()
    index_found = any(f"lists = {nlist}" in idx_def for _, idx_def in indexes)

    if index_found:
        print(f"   ✅ IVFFlat index with nlist={nlist} exists")
    else:
        print(f"   ⚠️  No index found with nlist={nlist}")

    # Check nprobe setting
    with connect(**PG_KWARGS) as con, con.cursor() as cur:
        cur.execute("SHOW ivfflat.probes;")
        current_probes = int(cur.fetchone()[0])

        if current_probes == nprobe:
            print(f"   ✅ ivfflat.probes = {nprobe} (as expected)")
        else:
            print(f"   ⚠️  ivfflat.probes = {current_probes} (expected {nprobe})")
            print(f"      Reconnect to database for changes to take effect")

def main():
    parser = argparse.ArgumentParser(description="Apply IVFFlat index configuration")
    parser.add_argument(
        "--nlist",
        type=int,
        required=True,
        help="Number of inverted lists (recommended: sqrt to 2*sqrt of dataset size)"
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="Number of lists to probe during search (default: nlist/16, min 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-drop",
        action="store_true",
        help="Skip dropping existing indexes (will fail if index exists)"
    )

    args = parser.parse_args()

    # Calculate default nprobe if not specified
    if args.nprobe is None:
        args.nprobe = max(1, args.nlist // 16)

    print("=" * 80)
    print("IVFFlat Configuration")
    print("=" * 80)

    # Show database info
    chunk_count = get_chunk_count()
    print(f"\n📊 Database info:")
    print(f"   Database: {PG_KWARGS['dbname']}")
    print(f"   Chunks: {chunk_count:,}")
    print(f"   Embedding model: {EMBED_MODEL}")

    # Show configuration
    print(f"\n🎯 Target configuration:")
    print(f"   nlist: {args.nlist}")
    print(f"   nprobe: {args.nprobe}")

    # Recommendation check
    recommended_min = int(chunk_count ** 0.5)
    recommended_max = int(2 * (chunk_count ** 0.5))
    if not (recommended_min <= args.nlist <= recommended_max):
        print(f"\n   ⚠️  Warning: Recommended nlist range for {chunk_count:,} chunks: {recommended_min}-{recommended_max}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")

    # Drop existing indexes
    if not args.skip_drop:
        drop_existing_indexes(dry_run=args.dry_run)

    # Create new index
    create_ivfflat_index(args.nlist, dry_run=args.dry_run)

    # Set nprobe
    set_default_nprobe(args.nprobe, dry_run=args.dry_run)

    # Verify (only if not dry run)
    if not args.dry_run:
        verify_configuration(args.nlist, args.nprobe)

    print("\n✅ Configuration complete!")

    # Show next steps
    if not args.dry_run:
        print("\n📝 Next steps:")
        print("   1. Reconnect to the database for nprobe changes to take effect")
        print("   2. Test query performance")
        print("   3. Monitor recall and latency in production")

if __name__ == "__main__":
    main()
