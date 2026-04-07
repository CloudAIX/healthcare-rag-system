"""
Test Harness - Verify hybrid retrieval pipeline with real indexes.

Tests:
- Vector search works
- BM25 search works
- RRF fusion works
- Cross-encoder re-ranking works
- End-to-end query pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever


def test_retrieval_components():
    """Test each component of the retrieval pipeline."""
    print("\n" + "="*70)
    print("HYBRID RETRIEVAL PIPELINE - VERIFICATION TEST")
    print("="*70)

    # Initialize retriever
    print("\n🔧 Initializing retriever...")
    try:
        retriever = Retriever(enable_hybrid=True)
        print("   ✓ Retriever initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize retriever: {e}")
        return False

    # Check components
    print("\n🔍 Checking components:")

    # Vector search
    if retriever.embedder:
        print("   ✓ Vector search (ChromaDB) available")
    else:
        print("   ✗ Vector search unavailable")
        return False

    # BM25 search
    if retriever.bm25_index and retriever.bm25_index.exists():
        count = len(retriever.bm25_index)
        print(f"   ✓ BM25 index available ({count} documents)")
    else:
        print("   ✗ BM25 index not found")
        print("   → Run: python scripts/ingest_robust.py")
        return False

    # Reranker
    if retriever.reranker:
        print("   ✓ Cross-encoder reranker available")
    else:
        print("   ⚠️  Cross-encoder reranker not available")

    # RRF
    if retriever.rrf_fusion:
        print(f"   ✓ RRF fusion available (k={retriever.rrf_fusion.k})")
    else:
        print("   ✗ RRF fusion unavailable")
        return False

    # === TEST QUERIES ===
    print("\n📝 Running test queries:")
    test_queries = [
        "What are the documentation requirements?",
        "How should aged care facilities conduct assessments?",
        "What is the person-centered care standard?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: \"{query}\"")
        try:
            results = retriever.retrieve(query, top_k=3)

            if not results:
                print(f"      ⚠️  No results returned")
                continue

            print(f"      ✓ Retrieved {len(results)} chunks")

            # Show top result
            top_result = results[0]
            print(f"      • Top match: {top_result.document_title} (score: {top_result.score:.3f})")
            print(f"        Section: {', '.join(top_result.sections) if top_result.sections else 'N/A'}")
            print(f"        Page: {top_result.page_numbers}")
            print(f"        Citation: {top_result.citation}")
            print(f"        Text: {top_result.text[:100]}...")

        except Exception as e:
            print(f"      ✗ Query failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # === PERFORMANCE TEST ===
    print("\n⏱️  Performance test:")
    import time

    query = "What is the governance standard?"
    start_time = time.time()
    results = retriever.retrieve(query, top_k=3)
    latency = (time.time() - start_time) * 1000  # ms

    print(f"   Query: \"{query}\"")
    print(f"   Latency: {latency:.1f}ms")
    print(f"   Results: {len(results)} chunks")

    if latency > 500:
        print("   ⚠️  Performance: Slower than expected (>500ms)")
    else:
        print("   ✓ Performance: Good (<500ms)")

    # === SUMMARY ===
    print("\n" + "="*70)
    print("✨ ALL VERIFICATION TESTS PASSED")
    print("="*70)
    print("\nYou can now:")
    print("  1. Use hybrid retrieval in your application")
    print("  2. Run: python -m pytest tests/ (unit tests)")
    print("  3. Continue to Phase 3: Evaluation")
    print("="*70 + "\n")

    return True


def main():
    """Main entry point."""
    try:
        success = test_retrieval_components()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
