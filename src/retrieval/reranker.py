"""Reranker — cross-encoder semantic re-ranking of retrieved chunks."""
from typing import Optional
from sentence_transformers import CrossEncoder
import torch


class CrossEncoderReranker:
    """Cross-encoder based re-ranker for semantic relevance scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_score: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier
            min_score: Minimum score threshold for results
            device: Device to run model on ('cuda', 'cpu', None=auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.min_score = min_score

        print(f"Loading cross-encoder model: {model_name} (device={device})")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, chunks: list[dict], batch_size: int = 32
    ) -> list[dict]:
        """
        Re-rank chunks by semantic relevance to query.

        Args:
            query: Query string
            chunks: List of dicts with 'id', 'text', and optionally 'score' key
            batch_size: Batch size for processing

        Returns:
            List of dicts with 'id', 'text', and 'score' (cross-encoder score normalized to [0, 1])
        """
        if not chunks:
            return []

        # Prepare query-chunk pairs
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Score all pairs in batches
        scores = self.model.predict(pairs, batch_size=batch_size)

        # Normalize scores to [0, 1] using sigmoid (handles negative logits)
        normalized_scores = self._sigmoid_normalize(scores)

        # Create results with all chunks ranked
        results = []
        for chunk, score in zip(chunks, normalized_scores):
            result = {
                "id": chunk.get("id"),
                "text": chunk.get("text"),
                "score": float(score),
                "original_score": chunk.get("score"),  # Preserve original score
            }
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def rerank_batch(
        self, query_chunk_pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> list[float]:
        """
        Score a batch of (query, chunk) pairs.

        Args:
            query_chunk_pairs: List of (query, chunk_text) tuples
            batch_size: Batch size for processing

        Returns:
            List of normalized scores [0, 1]
        """
        if not query_chunk_pairs:
            return []

        scores = self.model.predict(query_chunk_pairs, batch_size=batch_size)
        return self._sigmoid_normalize(scores).tolist()

    def _sigmoid_normalize(self, scores):
        """Normalize scores to [0, 1] using sigmoid function."""
        import numpy as np

        return 1 / (1 + np.exp(-scores))

    def filter_by_threshold(
        self, results: list[dict], min_score: Optional[float] = None
    ) -> list[dict]:
        """
        Filter results by score threshold.

        Args:
            results: List of result dicts with 'score' key
            min_score: Score threshold (uses self.min_score if None)

        Returns:
            Filtered list of results
        """
        threshold = min_score if min_score is not None else self.min_score
        return [r for r in results if r.get("score", 0) >= threshold]
