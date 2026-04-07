"""RRF Fusion — Reciprocal Rank Fusion for merging search rankings."""
from typing import Optional


class RRFFusion:
    """Reciprocal Rank Fusion (RRF) for combining multiple rankings."""

    def __init__(self, k: int = 60):
        """
        Initialize RRF with parameter k.

        Args:
            k: RRF parameter (typical value: 60). Formula: RRF(d) = 1/(k + rank(d))
        """
        self.k = k

    def fuse(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Fuse two ranking lists using RRF.

        Args:
            vector_results: List of (chunk_id, score) from vector search
            bm25_results: List of (chunk_id, score) from BM25 search
            top_k: Number of top results to return (optional)

        Returns:
            Fused results: List of (chunk_id, rrf_score) sorted by score desc
        """
        # Create rank dictionaries for each ranker
        vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_results)}
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_results)}

        # Collect all chunk IDs from both rankers
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Compute RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0

            # If chunk appears in vector results
            if chunk_id in vector_ranks:
                rank = vector_ranks[chunk_id]
                rrf_score += self._rrf_component(rank)

            # If chunk appears in BM25 results
            if chunk_id in bm25_ranks:
                rank = bm25_ranks[chunk_id]
                rrf_score += self._rrf_component(rank)

            rrf_scores[chunk_id] = rrf_score

        # Sort by RRF score descending
        results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def _rrf_component(self, rank: int) -> float:
        """
        Compute RRF component for a given rank (0-indexed).

        Formula: 1 / (k + rank)

        Args:
            rank: 0-indexed rank position

        Returns:
            RRF component score
        """
        return 1 / (self.k + rank + 1)  # +1 because rank is 0-indexed

    def normalize_scores(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """
        Normalize RRF scores to [0, 1].

        Args:
            results: List of (chunk_id, score) tuples

        Returns:
            Results with normalized scores in [0, 1]
        """
        if not results:
            return []

        scores = [score for _, score in results]
        max_score = max(scores)

        if max_score == 0:
            return results

        normalized = [
            (chunk_id, score / max_score) for chunk_id, score in results
        ]
        return normalized
