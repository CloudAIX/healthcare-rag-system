"""Unit tests for RRFFusion."""
import pytest
from src.retrieval.rrf_fusion import RRFFusion


@pytest.fixture
def rrf_fusion():
    """RRF fusion instance for testing."""
    return RRFFusion(k=60)


def test_rrf_initialization(rrf_fusion):
    """Test RRF initialization."""
    assert rrf_fusion.k == 60


def test_rrf_fuse_basic(rrf_fusion):
    """Test basic RRF fusion."""
    vector_results = [
        ("chunk-001", 0.9),
        ("chunk-002", 0.8),
        ("chunk-003", 0.7),
    ]
    bm25_results = [("chunk-003", 0.95), ("chunk-001", 0.85), ("chunk-004", 0.7)]

    fused = rrf_fusion.fuse(vector_results, bm25_results)

    assert len(fused) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in fused)


def test_rrf_fuse_sorting(rrf_fusion):
    """Test that fused results are sorted by score."""
    vector_results = [
        ("chunk-001", 0.9),
        ("chunk-002", 0.8),
    ]
    bm25_results = [("chunk-003", 0.95), ("chunk-001", 0.85)]

    fused = rrf_fusion.fuse(vector_results, bm25_results)

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_top_k(rrf_fusion):
    """Test top_k parameter."""
    vector_results = [
        ("chunk-001", 0.9),
        ("chunk-002", 0.8),
        ("chunk-003", 0.7),
    ]
    bm25_results = [
        ("chunk-004", 0.95),
        ("chunk-005", 0.85),
        ("chunk-006", 0.7),
    ]

    fused_5 = rrf_fusion.fuse(vector_results, bm25_results, top_k=5)
    fused_3 = rrf_fusion.fuse(vector_results, bm25_results, top_k=3)
    fused_1 = rrf_fusion.fuse(vector_results, bm25_results, top_k=1)

    assert len(fused_5) <= 5
    assert len(fused_3) <= 3
    assert len(fused_1) <= 1


def test_rrf_fuse_empty_results(rrf_fusion):
    """Test fusion with empty results."""
    vector_results = []
    bm25_results = []

    fused = rrf_fusion.fuse(vector_results, bm25_results)

    assert fused == []


def test_rrf_fuse_overlapping_results(rrf_fusion):
    """Test fusion with overlapping results."""
    vector_results = [
        ("chunk-001", 0.9),
        ("chunk-002", 0.8),
    ]
    bm25_results = [
        ("chunk-001", 0.95),  # Same chunk
        ("chunk-002", 0.85),  # Same chunk
        ("chunk-003", 0.7),  # New chunk
    ]

    fused = rrf_fusion.fuse(vector_results, bm25_results)

    # Overlapping chunks should have combined RRF scores
    chunk_001_result = next((r for r in fused if r[0] == "chunk-001"), None)
    assert chunk_001_result is not None
    # Score should be combined (both rankers contributed)
    assert chunk_001_result[1] > 0


def test_rrf_fuse_non_overlapping(rrf_fusion):
    """Test fusion with non-overlapping results."""
    vector_results = [
        ("chunk-001", 0.9),
        ("chunk-002", 0.8),
    ]
    bm25_results = [
        ("chunk-003", 0.95),
        ("chunk-004", 0.85),
    ]

    fused = rrf_fusion.fuse(vector_results, bm25_results)

    # All chunks should be present
    fused_ids = set(chunk_id for chunk_id, _ in fused)
    assert len(fused_ids) == 4
    assert fused_ids == {"chunk-001", "chunk-002", "chunk-003", "chunk-004"}


def test_rrf_component(rrf_fusion):
    """Test RRF component calculation."""
    # Rank 0: 1/(60 + 0 + 1) = 1/61
    component_0 = rrf_fusion._rrf_component(0)
    assert component_0 == pytest.approx(1 / 61)

    # Rank 1: 1/(60 + 1 + 1) = 1/62
    component_1 = rrf_fusion._rrf_component(1)
    assert component_1 == pytest.approx(1 / 62)

    # Higher rank = lower component
    assert component_0 > component_1


def test_rrf_component_increases_with_k(rrf_fusion):
    """Test that k affects component scores."""
    rrf_60 = RRFFusion(k=60)
    rrf_100 = RRFFusion(k=100)

    comp_60 = rrf_60._rrf_component(0)
    comp_100 = rrf_100._rrf_component(0)

    # Larger k gives smaller component
    assert comp_60 > comp_100


def test_rrf_normalize_scores(rrf_fusion):
    """Test score normalization."""
    results = [
        ("chunk-001", 0.5),
        ("chunk-002", 0.25),
        ("chunk-003", 0.1),
    ]

    normalized = rrf_fusion.normalize_scores(results)

    # Max should be 1
    max_score = max(score for _, score in normalized)
    assert max_score == pytest.approx(1.0)

    # All should be <= 1
    assert all(score <= 1.0 for _, score in normalized)

    # Ordering should be preserved
    normalized_scores = [score for _, score in normalized]
    assert normalized_scores == sorted(normalized_scores, reverse=True)


def test_rrf_normalize_empty(rrf_fusion):
    """Test normalization with empty results."""
    results = []

    normalized = rrf_fusion.normalize_scores(results)

    assert normalized == []


def test_rrf_normalize_single_zero_score(rrf_fusion):
    """Test normalization with all zero scores."""
    results = [("chunk-001", 0), ("chunk-002", 0)]

    normalized = rrf_fusion.normalize_scores(results)

    assert normalized == results


def test_rrf_k_parameter():
    """Test different k parameters."""
    rrf_small = RRFFusion(k=10)
    rrf_large = RRFFusion(k=100)

    vector_results = [("chunk-001", 0.9), ("chunk-002", 0.8)]
    bm25_results = [("chunk-001", 0.95)]

    fused_small = rrf_small.fuse(vector_results, bm25_results)
    fused_large = rrf_large.fuse(vector_results, bm25_results)

    # Both should work but possibly with different scores
    assert len(fused_small) > 0
    assert len(fused_large) > 0
