#!/usr/bin/env python3
"""
Quick test script to verify all three search strategies work correctly.
"""

import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from llm_integration.search_strategies import (
    retrieve_ivfflat_only,
    retrieve_ivfflat_ner_boost,
    retrieve_ivfflat_tsvector,
    compare_all_strategies
)

def test_single_query():
    """Test all strategies on a single medical query."""

    question = "What are the treatments for type 2 diabetes?"
    k = 5

    print("=" * 80)
    print("Testing Search Strategies")
    print("=" * 80)
    print(f"\nQuery: {question}")
    print(f"Top-K: {k}")
    print("\n" + "-" * 80)

    # Test each strategy
    strategies = [
        ("IVFFlat Only", retrieve_ivfflat_only),
        ("IVFFlat + NER Boost", retrieve_ivfflat_ner_boost),
        ("IVFFlat + tsvector", retrieve_ivfflat_tsvector),
    ]

    for strategy_name, strategy_func in strategies:
        print(f"\n🔍 {strategy_name}")
        print("-" * 80)

        try:
            import time
            start = time.time()
            results = strategy_func(question, k=k)
            elapsed_ms = (time.time() - start) * 1000

            print(f"✅ Success! Retrieved {len(results)} results in {elapsed_ms:.2f}ms")

            # Show top 3 results
            for i, result in enumerate(results[:3], 1):
                score = result.get('final_score', result.get('similarity', 0))
                text_preview = result['text'][:100].replace('\n', ' ')

                print(f"\n  {i}. Score: {score:.4f}")
                print(f"     Chunk ID: {result['chunk_id']}")
                print(f"     Doc: {result['pmcid']}")
                print(f"     Text: {text_preview}...")

                # Show strategy-specific metadata
                if 'ner_mention_count' in result:
                    print(f"     NER mentions: {result['ner_mention_count']}")
                    print(f"     NER boost: +{result['ner_boost_applied']:.4f}")
                    if result.get('entity_types'):
                        print(f"     Entity types: {', '.join(result['entity_types'])}")

                if 'vector_rank' in result and 'text_rank' in result:
                    print(f"     Vector rank: {result['vector_rank']}, Text rank: {result['text_rank']}")
                    print(f"     Text score: {result.get('text_score', 0):.4f}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

def test_comparison():
    """Test the comparison function."""
    print("\n" + "=" * 80)
    print("Testing compare_all_strategies()")
    print("=" * 80)

    question = "What drugs are used for cancer treatment?"
    k = 5

    print(f"\nQuery: {question}")
    print(f"Top-K: {k}\n")

    try:
        import time
        start = time.time()
        results = compare_all_strategies(question, k=k)
        elapsed_ms = (time.time() - start) * 1000

        print(f"✅ Retrieved results from all 3 strategies in {elapsed_ms:.2f}ms\n")

        for strategy_name, chunks in results.items():
            print(f"{strategy_name}: {len(chunks)} results")
            if chunks:
                top_score = chunks[0].get('final_score', chunks[0].get('similarity', 0))
                print(f"  Top score: {top_score:.4f}")
                print(f"  Top result: {chunks[0]['text'][:80]}...")
            print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_query()
    test_comparison()
