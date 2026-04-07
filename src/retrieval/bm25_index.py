"""BM25 Index — full-text search index for document chunks."""
import pickle
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi
import yaml


def load_retrieval_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


class BM25Index:
    """BM25 full-text search index with persistence."""

    def __init__(self, config=None, persist_path: Optional[Path] = None):
        if config is None:
            config = load_retrieval_config()
        self.config = config
        self.persist_path = persist_path or Path("./data/processed/bm25_index.pkl")
        self.bm25 = None
        self.chunk_texts = []  # List of texts in corpus order
        self.chunk_ids = []  # Corresponding chunk IDs
        self.id_to_text = {}  # For reconstructing results

    def build_from_chunks(self, chunks):
        """Build BM25 index from list of Chunk objects."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")

        # Extract texts and IDs
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.id_to_text = {c.chunk_id: c.text for c in chunks}

        # Tokenize (simple space-based tokenization)
        corpus = [self._tokenize(text) for text in self.chunk_texts]

        # Build BM25 index
        self.bm25 = BM25Okapi(corpus)
        print(f"Built BM25 index with {len(self.chunk_ids)} chunks")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: split on whitespace and lowercase."""
        return text.lower().split()

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Search with BM25 and return top-k results.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, normalized_score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_from_chunks() first.")

        if not query_text.strip():
            return []

        # Tokenize query
        tokens = self._tokenize(query_text)

        # Get BM25 scores (returns numpy array)
        scores = self.bm25.get_scores(tokens)

        # Normalize scores to [0, 1]
        max_score = float(scores.max()) if len(scores) > 0 else 0
        if max_score > 0:
            normalized_scores = scores / max_score
        else:
            normalized_scores = scores

        # Get top-k
        results = []
        for idx, score in enumerate(normalized_scores):
            if score > 0:  # Only include non-zero scores
                results.append((self.chunk_ids[idx], float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def add_chunks(self, chunks):
        """Add new chunks to existing index."""
        if self.bm25 is None:
            self.build_from_chunks(chunks)
            return

        # Add to existing lists
        existing_ids = set(self.chunk_ids)
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            print("All chunks already in index")
            return

        # Extend corpus
        new_texts = [c.text for c in new_chunks]
        new_ids = [c.chunk_id for c in new_chunks]

        self.chunk_texts.extend(new_texts)
        self.chunk_ids.extend(new_ids)
        self.id_to_text.update({c.chunk_id: c.text for c in new_chunks})

        # Rebuild BM25 with full corpus
        corpus = [self._tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(corpus)
        print(f"Updated BM25 index: now has {len(self.chunk_ids)} chunks")

    def save(self):
        """Persist index to disk."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Pickle only what we need
        data = {
            "chunk_texts": self.chunk_texts,
            "chunk_ids": self.chunk_ids,
            "id_to_text": self.id_to_text,
            "bm25": self.bm25,
        }

        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved BM25 index to {self.persist_path}")

    def load(self):
        """Load index from disk."""
        if not self.persist_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.persist_path}")

        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)

        self.chunk_texts = data["chunk_texts"]
        self.chunk_ids = data["chunk_ids"]
        self.id_to_text = data["id_to_text"]
        self.bm25 = data["bm25"]
        print(f"Loaded BM25 index with {len(self.chunk_ids)} chunks")

    def exists(self) -> bool:
        """Check if index file exists on disk."""
        return self.persist_path.exists()

    def __len__(self) -> int:
        """Return number of chunks in index."""
        return len(self.chunk_ids) if self.chunk_ids else 0
