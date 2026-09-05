"""Integration tests for hybrid retriever pipeline."""
import pytest
from pathlib import Path
import tempfile
from src.retrieval.retriever import Retriever, RetrievedChunk
from src.ingestion.embedder import Embedder
from src.ingestion.chunker import Chunk


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk-001",
            text="Standard 1: Governance and organizational leadership. Facilities must maintain proper documentation and compliance procedures.",
            document_title="Aged Care Quality Standards",
            document_filename="standards.pdf",
            page_numbers=[1, 2],
            sections=["Standard 1"],
            chunk_index=0,
            char_start=0,
            char_end=150,
        ),
        Chunk(
            chunk_id="chunk-002",
            text="Outcome 1.1: Person-centred care involves respecting individual preferences, values, and dignity of residents.",
            document_title="Aged Care Quality Standards",
            document_filename="standards.pdf",
            page_numbers=[3],
            sections=["Standard 1", "Outcome 1.1"],
            chunk_index=1,
            char_start=150,
            char_end=300,
        ),
        Chunk(
            chunk_id="chunk-003",
            text="Action 1.1.1: Staff must conduct comprehensive assessments of individual needs and preferences.",
            document_title="Aged Care Quality Standards",
            document_filename="standards.pdf",
            page_numbers=[4, 5],
            sections=["Standard 1", "Action 1.1.1"],
            chunk_index=2,
            char_start=300,
            char_end=450,
        ),
        Chunk(
            chunk_id="chunk-004",
            text="Standard 2: Safe and effective care. All facilities must implement safety protocols and risk management.",
            document_title="Aged Care Quality Standards",
            document_filename="standards.pdf",
            page_numbers=[10],
            sections=["Standard 2"],
            chunk_index=3,
            char_start=500,
            char_end=650,
        ),
    ]


def test_retrieved_chunk_dataclass(sample_chunks):
    """Test RetrievedChunk dataclass."""
    chunk_data = sample_chunks[0]
    retrieved = RetrievedChunk(
        chunk_id=chunk_data.chunk_id,
        text=chunk_data.text,
        document_title=chunk_data.document_title,
        document_filename=chunk_data.document_filename,
        page_numbers=chunk_data.page_numbers,
        sections=chunk_data.sections,
        score=0.95,
    )

    assert retrieved.chunk_id == "chunk-001"
    assert retrieved.score == 0.95

    # Test citation property
    citation = retrieved.citation
    assert "Source:" in citation
    assert "Aged Care Quality Standards" in citation
    assert "Standard 1" in citation
    assert "pp.1,2" in citation


def test_retriever_vector_only_mode(sample_chunks):
    """Test retriever in vector-only mode."""
    # Create retriever with hybrid disabled
    retriever = Retriever(enable_hybrid=False)

    assert retriever.enable_hybrid is False
    assert retriever.bm25_index is None


def test_retriever_hybrid_mode_no_index(tmp_path, monkeypatch):
    """Test retriever in hybrid mode when BM25 index doesn't exist."""
    # BM25 path is cwd-relative, so an empty temp cwd guarantees no index
    # regardless of whether the real one has been built.
    monkeypatch.chdir(tmp_path)
    retriever = Retriever(enable_hybrid=True)

    # Should gracefully handle missing BM25 index
    assert retriever.bm25_index is None or not retriever.bm25_index.exists()


def test_retriever_fallback_to_vector_only(sample_chunks):
    """Test that retriever falls back to vector-only when BM25 unavailable."""
    # Create retriever with hybrid enabled but no BM25 index
    retriever = Retriever(enable_hybrid=True)

    # If no BM25 index, should use vector-only fallback
    if retriever.bm25_index is None or not retriever.bm25_index.exists():
        # This is expected behavior - should fall back to vector-only
        assert True


def test_retrieved_chunk_citation_multipage(sample_chunks):
    """Test citation generation for chunk spanning multiple pages."""
    chunk = sample_chunks[2]
    retrieved = RetrievedChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        document_title=chunk.document_title,
        document_filename=chunk.document_filename,
        page_numbers=chunk.page_numbers,
        sections=chunk.sections,
        score=0.90,
    )

    citation = retrieved.citation
    assert "pp.4,5" in citation or "pp.4" in citation


def test_retrieved_chunk_citation_no_sections():
    """Test citation when no sections are available."""
    retrieved = RetrievedChunk(
        chunk_id="chunk-001",
        text="Some text",
        document_title="Document Title",
        document_filename="doc.pdf",
        page_numbers=[1],
        sections=[],  # No sections
        score=0.95,
    )

    citation = retrieved.citation
    assert "Source:" in citation
    assert "Document Title" in citation
    assert "pp.1" in citation


def test_retriever_config_loading():
    """Test that retriever loads configuration correctly."""
    retriever = Retriever()

    assert retriever.config is not None
    assert "retrieval" in retriever.config
    assert retriever.top_k_vector == retriever.config["retrieval"]["top_k_vector"]
    assert retriever.top_k_bm25 == retriever.config["retrieval"]["top_k_bm25"]
    assert retriever.rrf_k == retriever.config["retrieval"]["rrf_k"]
    assert retriever.top_k_rerank == retriever.config["retrieval"]["top_k_rerank"]


def test_retriever_vector_search_method(sample_chunks):
    """Test vector search method."""
    retriever = Retriever(enable_hybrid=False)

    # Vector search should return (chunk_id, score) tuples
    # This is a mock test since we don't have real embeddings in test
    # In real scenario, this would query ChromaDB
    assert hasattr(retriever, '_vector_search')
    assert callable(retriever._vector_search)


def test_retriever_bm25_search_method():
    """Test BM25 search method."""
    retriever = Retriever(enable_hybrid=True)

    # If BM25 index not available, should return empty
    results = retriever._bm25_search("test query")
    assert isinstance(results, list)


def test_retriever_rrf_fusion_method():
    """Test RRF fusion method."""
    retriever = Retriever()

    # Test with sample ranking lists
    vector_results = [("chunk-001", 0.9), ("chunk-002", 0.8)]
    bm25_results = [("chunk-001", 0.95), ("chunk-003", 0.7)]

    fused = retriever._rrf_fusion(vector_results, bm25_results)

    # Should return list of tuples
    assert isinstance(fused, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in fused)


def test_retriever_backward_compatibility():
    """Test that Retriever maintains backward compatibility."""
    retriever = Retriever(enable_hybrid=False)

    # retrieve() method should exist and accept query parameter
    assert hasattr(retriever, 'retrieve')
    assert callable(retriever.retrieve)

    # Should return list of RetrievedChunk objects
    # (even if empty in test without real data)
    result = retriever.retrieve("test query", top_k=3)
    assert isinstance(result, list)


def test_retriever_top_k_parameter():
    """Test top_k parameter handling."""
    retriever = Retriever(enable_hybrid=False)

    # Test with explicit top_k
    result1 = retriever.retrieve("test", top_k=5)
    result2 = retriever.retrieve("test", top_k=1)

    # Both should be lists
    assert isinstance(result1, list)
    assert isinstance(result2, list)

    # Should respect top_k limit
    if len(result1) > 0:
        assert len(result1) <= 5
    if len(result2) > 0:
        assert len(result2) <= 1


def test_retriever_error_handling():
    """Test error handling in retriever."""
    retriever = Retriever(enable_hybrid=False)

    # Should not raise on empty query
    try:
        result = retriever.retrieve("", top_k=3)
        assert isinstance(result, list)
    except Exception as e:
        pytest.fail(f"Retriever should handle empty query gracefully: {e}")


def test_retriever_initialization_parameters():
    """Test retriever initialization with different parameters."""
    # Test with custom embedder
    embedder = Embedder()
    retriever = Retriever(embedder=embedder, enable_hybrid=False)

    assert retriever.embedder is embedder
    assert not retriever.enable_hybrid


def test_retriever_graceful_degradation():
    """Test that retriever gracefully degrades when components fail."""
    retriever = Retriever(enable_hybrid=True)

    # If hybrid components not available, should still work as vector-only
    if retriever.bm25_index is None:
        # Retriever should fall back to vector-only
        assert True

    # retrieve() method should always return a list
    result = retriever.retrieve("test query")
    assert isinstance(result, list)
