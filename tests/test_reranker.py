"""Unit tests for CrossEncoderReranker."""
import pytest
import numpy as np
from src.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def reranker():
    """Reranker instance for testing."""
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_score=0.1,
    )


@pytest.fixture
def sample_chunks():
    """Sample chunks for re-ranking."""
    return [
        {
            "id": "chunk-001",
            "text": "Standard 1 requires aged care facilities to have proper documentation and compliance procedures",
            "score": 0.8,
        },
        {
            "id": "chunk-002",
            "text": "Person-centred care involves respecting individual preferences and dignity",
            "score": 0.7,
        },
        {
            "id": "chunk-003",
            "text": "Physical infrastructure includes building safety, equipment, and facilities",
            "score": 0.6,
        },
    ]


def test_reranker_initialization(reranker):
    """Test reranker initialization."""
    assert reranker.model is not None
    assert reranker.min_score == 0.1


def test_reranker_rerank_basic(reranker, sample_chunks):
    """Test basic re-ranking."""
    query = "What are documentation requirements?"
    results = reranker.rerank(query, sample_chunks)

    assert len(results) > 0
    assert all("id" in r and "text" in r and "score" in r for r in results)
    assert all(0 <= r["score"] <= 1 for r in results)


def test_reranker_rerank_sorting(reranker, sample_chunks):
    """Test that results are sorted by score."""
    query = "What are documentation requirements?"
    results = reranker.rerank(query, sample_chunks)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reranker_min_score_filtering(reranker, sample_chunks):
    """Test minimum score filtering."""
    query = "water"
    results = reranker.rerank(query, sample_chunks)

    assert all(r["score"] >= reranker.min_score for r in results)


def test_reranker_empty_chunks(reranker):
    """Test re-ranking with empty chunk list."""
    query = "test query"
    results = reranker.rerank(query, [])

    assert results == []


def test_reranker_batch_processing(reranker, sample_chunks):
    """Test batch processing of query-chunk pairs."""
    pairs = [
        ("documentation requirements", chunk["text"]) for chunk in sample_chunks
    ]

    scores = reranker.rerank_batch(pairs)

    assert len(scores) == len(pairs)
    assert all(isinstance(s, float) for s in scores)
    assert all(0 <= s <= 1 for s in scores)


def test_reranker_batch_empty(reranker):
    """Test batch processing with empty list."""
    scores = reranker.rerank_batch([])

    assert scores == []


def test_reranker_filter_by_threshold(reranker, sample_chunks):
    """Test filtering results by threshold."""
    query = "What are documentation requirements?"
    results = reranker.rerank(query, sample_chunks)

    filtered = reranker.filter_by_threshold(results, min_score=0.5)

    assert all(r["score"] >= 0.5 for r in filtered)
    assert len(filtered) <= len(results)


def test_reranker_custom_threshold(reranker, sample_chunks):
    """Test filtering with custom threshold."""
    results = [
        {"id": "1", "text": "text1", "score": 0.9},
        {"id": "2", "text": "text2", "score": 0.5},
        {"id": "3", "text": "text3", "score": 0.2},
    ]

    filtered = reranker.filter_by_threshold(results, min_score=0.6)

    assert len(filtered) == 1
    assert filtered[0]["id"] == "1"


def test_reranker_preserves_original_score(reranker, sample_chunks):
    """Test that original score is preserved in results."""
    query = "test query"
    results = reranker.rerank(query, sample_chunks)

    assert all("original_score" in r for r in results)


def test_reranker_sigmoid_normalize():
    """Test sigmoid normalization produces values in [0, 1]."""
    reranker_inst = CrossEncoderReranker()
    scores = np.array([-10, -1, 0, 1, 10])

    normalized = reranker_inst._sigmoid_normalize(scores)

    assert all(0 <= s <= 1 for s in normalized)
    assert normalized[0] < normalized[2] < normalized[4]  # Ordering preserved


def test_reranker_device_handling():
    """Test device handling (CPU fallback)."""
    # This should not raise an error
    reranker = CrossEncoderReranker(device="cpu")
    assert reranker.device == "cpu"
