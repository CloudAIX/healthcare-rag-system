"""Unit tests for BM25Index."""
import pytest
from pathlib import Path
from src.retrieval.bm25_index import BM25Index
from src.ingestion.chunker import Chunk


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk-001",
            text="Standard 1 requires aged care facilities to have proper documentation",
            document_title="Aged Care Standards",
            document_filename="standards.pdf",
            page_numbers=[1, 2],
            sections=["Standard 1"],
            chunk_index=0,
            char_start=0,
            char_end=100,
        ),
        Chunk(
            chunk_id="chunk-002",
            text="Outcome 1.1 focuses on person-centred care and individual needs assessment",
            document_title="Aged Care Standards",
            document_filename="standards.pdf",
            page_numbers=[3],
            sections=["Standard 1", "Outcome 1.1"],
            chunk_index=1,
            char_start=100,
            char_end=200,
        ),
        Chunk(
            chunk_id="chunk-003",
            text="Action 1.1.1 requires staff to conduct comprehensive assessments",
            document_title="Aged Care Standards",
            document_filename="standards.pdf",
            page_numbers=[4, 5],
            sections=["Standard 1", "Action 1.1.1"],
            chunk_index=2,
            char_start=200,
            char_end=300,
        ),
    ]


def test_bm25_build_from_chunks(sample_chunks):
    """Test building BM25 index from chunks."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    assert len(index) == 3
    assert index.chunk_ids == ["chunk-001", "chunk-002", "chunk-003"]
    assert index.bm25 is not None


def test_bm25_query_basic(sample_chunks):
    """Test basic BM25 query."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    results = index.query("assessment", top_k=2)

    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(0 <= score <= 1 for _, score in results)  # Normalized scores


def test_bm25_query_empty_string(sample_chunks):
    """Test query with empty string."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    results = index.query("", top_k=5)

    assert results == []


def test_bm25_query_nonexistent_term(sample_chunks):
    """Test query with term not in corpus."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    results = index.query("xyz123nonexistent", top_k=5)

    assert results == []


def test_bm25_score_normalization(sample_chunks):
    """Test that scores are normalized to [0, 1]."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    results = index.query("standard", top_k=5)

    assert len(results) > 0
    for chunk_id, score in results:
        assert 0 <= score <= 1, f"Score {score} not in [0, 1]"


def test_bm25_top_k_limit(sample_chunks):
    """Test that top_k parameter is respected."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    results_5 = index.query("standard", top_k=5)
    results_2 = index.query("standard", top_k=2)
    results_1 = index.query("standard", top_k=1)

    assert len(results_5) <= 5
    assert len(results_2) <= 2
    assert len(results_1) <= 1


def test_bm25_add_chunks(sample_chunks):
    """Test adding chunks to existing index."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks[:2])

    assert len(index) == 2

    index.add_chunks(sample_chunks[2:])

    assert len(index) == 3


def test_bm25_add_chunks_duplicates(sample_chunks):
    """Test adding chunks when some already exist."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks)

    # Try adding first chunk again
    index.add_chunks([sample_chunks[0]])

    assert len(index) == 3  # Should still be 3


def test_bm25_save_and_load(sample_chunks, tmp_path):
    """Test saving and loading index."""
    index_path = tmp_path / "test_bm25.pkl"

    # Build and save
    index1 = BM25Index(persist_path=index_path)
    index1.build_from_chunks(sample_chunks)
    index1.save()

    assert index_path.exists()

    # Load
    index2 = BM25Index(persist_path=index_path)
    index2.load()

    assert len(index2) == 3
    assert index2.chunk_ids == ["chunk-001", "chunk-002", "chunk-003"]

    # Query should work on loaded index
    results = index2.query("assessment", top_k=2)
    assert len(results) > 0


def test_bm25_index_exists(tmp_path):
    """Test exists() method."""
    index_path = tmp_path / "test_bm25.pkl"

    # Initially doesn't exist
    assert not index_path.exists()

    index = BM25Index(persist_path=index_path)
    assert not index.exists()

    # Create a dummy file
    index_path.touch()

    # Now should exist
    assert index.exists()


def test_bm25_load_nonexistent_file(tmp_path):
    """Test loading from non-existent file."""
    index_path = tmp_path / "nonexistent_bm25.pkl"

    index = BM25Index(persist_path=index_path)

    with pytest.raises(FileNotFoundError):
        index.load()


def test_bm25_build_empty_chunks():
    """Test building index from empty chunks."""
    index = BM25Index()

    with pytest.raises(ValueError):
        index.build_from_chunks([])


def test_bm25_query_before_build():
    """Test querying before index is built."""
    index = BM25Index()

    with pytest.raises(ValueError):
        index.query("test")
