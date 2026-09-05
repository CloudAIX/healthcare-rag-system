"""Retriever — hybrid search combining vector, BM25, RRF, and re-ranking."""
from dataclasses import dataclass
from pathlib import Path
import yaml
from src.ingestion.embedder import Embedder
from .bm25_index import BM25Index
from .reranker import CrossEncoderReranker
from .rrf_fusion import RRFFusion


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    document_title: str
    document_filename: str
    page_numbers: list[int]
    sections: list[str]
    score: float

    @property
    def citation(self):
        ps = ",".join(str(p) for p in self.page_numbers)
        sec = f", {self.sections[0]}" if self.sections else ""
        return f"[Source: {self.document_title}{sec}, pp.{ps}]"


def load_retrieval_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


class Retriever:
    def __init__(self, embedder=None, enable_hybrid=True):
        self.config = load_retrieval_config()
        import os
        self.backend = os.getenv("RAG_BACKEND") or self.config["vector_store"].get("backend", "chroma")
        if embedder is not None:
            self.embedder = embedder
        elif self.backend == "azure":
            # Azure AI Search runs vector + keyword hybrid in a single call, so the
            # local BM25/RRF stage is redundant — reranker still applies on top.
            from .azure_search_store import AzureSearchStore
            self.embedder = AzureSearchStore(self.config)
            enable_hybrid = False
        else:
            self.embedder = Embedder(self.config)
        self.enable_hybrid = enable_hybrid

        # Configuration parameters
        self.top_k_vector = self.config["retrieval"]["top_k_vector"]
        self.top_k_bm25 = self.config["retrieval"]["top_k_bm25"]
        self.rrf_k = self.config["retrieval"]["rrf_k"]
        self.top_k_rerank = self.config["retrieval"]["top_k_rerank"]

        # Initialize components for hybrid retrieval
        self.bm25_index = None
        self.reranker = None
        self.rrf_fusion = None

        if self.enable_hybrid:
            self._init_hybrid_components()
        elif self.backend == "azure":
            # Hybrid happens inside Azure AI Search; keep the cross-encoder on top.
            self._init_reranker()

    def _init_hybrid_components(self):
        """Initialize BM25, reranker, and RRF components."""
        # BM25 index
        # Repo-anchored, not cwd-relative: MCP clients launch this from anywhere.
        # RAG_DATA_DIR overrides the data directory (also used by tests).
        import os as _os
        data_dir = Path(_os.getenv("RAG_DATA_DIR") or Path(__file__).parent.parent.parent / "data")
        bm25_path = data_dir / "processed" / "bm25_index.pkl"
        self.bm25_index = BM25Index(self.config, persist_path=bm25_path)
        if self.bm25_index.exists():
            try:
                self.bm25_index.load()
            except Exception as e:
                print(f"Warning: Could not load BM25 index: {e}")
                self.bm25_index = None
        else:
            print("Warning: BM25 index not found. Install with: python scripts/ingest.py")
            self.bm25_index = None

        # Cross-encoder reranker
        self._init_reranker()

        # RRF fusion
        self.rrf_fusion = RRFFusion(k=self.rrf_k)

    def _init_reranker(self):
        try:
            reranker_config = self.config.get("reranker", {})
            model_name = reranker_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            min_score = reranker_config.get("min_score", 0.1)
            self.reranker = CrossEncoderReranker(model_name=model_name, min_score=min_score)
        except Exception as e:
            print(f"Warning: Could not load reranker: {e}")
            self.reranker = None

    def retrieve(self, query, top_k=None):
        """
        Retrieve chunks using hybrid search (vector + BM25 + RRF + re-ranking).

        Args:
            query: Query string
            top_k: Override default top_k_rerank

        Returns:
            List of RetrievedChunk objects
        """
        k = top_k or self.top_k_rerank

        # If hybrid retrieval not enabled or components not available, fall back to vector-only
        if not self.enable_hybrid or self.bm25_index is None:
            return self._retrieve_vector_only(query, k)

        # Step 1: Vector search
        vector_results = self._vector_search(query)

        # Step 2: BM25 search
        bm25_results = self._bm25_search(query)

        # Step 3: RRF fusion
        if bm25_results:
            fused_results = self._rrf_fusion(vector_results, bm25_results)
        else:
            # No BM25 results, use vector only
            print("No BM25 results, using vector search only")
            fused_results = vector_results

        # Step 4: Re-ranking (if available)
        if self.reranker and len(fused_results) > 0:
            final_results = self._rerank(query, fused_results, k)
        else:
            # No reranker, return fused results
            final_results = fused_results[:k]

        # Step 5: Convert to RetrievedChunk objects
        return self._format_results(final_results)

    def _retrieve_vector_only(self, query, top_k):
        """Single-store retrieval (Chroma vector-only, or Azure hybrid-in-one-call).

        When a reranker is available, over-fetch candidates and let the
        cross-encoder pick the final top_k, mirroring the hybrid path.
        """
        if self.reranker is not None:
            pool = self.top_k_vector + self.top_k_bm25
            hits = self.embedder.query(query, top_k=max(pool, top_k))
            by_id = {h["id"]: h for h in hits}
            reranked = self.reranker.rerank(
                query, [{"id": h["id"], "text": h["text"], "score": h["distance"]} for h in hits]
            )
            out = []
            for r in reranked[:top_k]:
                hit = by_id.get(r["id"])
                if hit:
                    out.append(self._hit_to_chunk({**hit, "distance": r["score"]}))
            return out
        raw = self.embedder.query(query, top_k=top_k)
        return [self._hit_to_chunk(hit) for hit in raw]

    def _vector_search(self, query):
        """Execute vector search and return (chunk_id, score) tuples."""
        raw = self.embedder.query(query, top_k=self.top_k_vector)
        return [(hit["id"], hit["distance"]) for hit in raw]

    def _bm25_search(self, query):
        """Execute BM25 search and return (chunk_id, score) tuples."""
        if self.bm25_index is None:
            return []
        try:
            return self.bm25_index.query(query, top_k=self.top_k_bm25)
        except Exception as e:
            print(f"Warning: BM25 search failed: {e}")
            return []

    def _rrf_fusion(self, vector_results, bm25_results):
        """Fuse vector and BM25 results using RRF."""
        fused = self.rrf_fusion.fuse(vector_results, bm25_results)
        return fused

    def _rerank(self, query, fused_results, top_k):
        """Re-rank fused results using cross-encoder."""
        if not fused_results or self.reranker is None:
            return fused_results[:top_k]

        # Prepare chunks for re-ranking (need to fetch text from embedder)
        chunk_dict = {}
        for chunk_id, score in fused_results:
            # Get chunk from embedder collection
            try:
                coll = self.embedder.get_or_create_collection()
                chunk_data = coll.get(ids=[chunk_id], include=["documents", "metadatas"])
                if chunk_data["ids"]:
                    chunk_dict[chunk_id] = {
                        "id": chunk_id,
                        "text": chunk_data["documents"][0],
                        "score": score,
                    }
            except Exception as e:
                print(f"Warning: Could not fetch chunk {chunk_id}: {e}")

        # Re-rank
        chunks_to_rerank = [chunk_dict[cid] for cid, _ in fused_results if cid in chunk_dict]
        reranked = self.reranker.rerank(query, chunks_to_rerank)

        return reranked[:top_k]

    def _format_results(self, results):
        """Convert results to RetrievedChunk objects."""
        chunks = []
        for result in results:
            try:
                # Fetch metadata from ChromaDB
                coll = self.embedder.get_or_create_collection()
                chunk_data = coll.get(
                    ids=[result["id"]],
                    include=["documents", "metadatas"]
                )

                if not chunk_data["ids"]:
                    continue

                m = chunk_data["metadatas"][0]
                pn = m.get("page_numbers", "1")
                if isinstance(pn, str):
                    pn = [int(p) for p in pn.split(",") if p.strip()]

                secs = m.get("sections", "")
                if isinstance(secs, str):
                    secs = [s.strip() for s in secs.split(",") if s.strip()]

                chunks.append(
                    RetrievedChunk(
                        chunk_id=result["id"],
                        text=chunk_data["documents"][0],
                        document_title=m.get("document_title", "Unknown"),
                        document_filename=m.get("document_filename", "unknown.pdf"),
                        page_numbers=pn,
                        sections=secs,
                        score=result.get("score", 0.0),
                    )
                )
            except Exception as e:
                print(f"Warning: Could not format result {result.get('id')}: {e}")

        return chunks

    def _hit_to_chunk(self, hit):
        """Convert a ChromaDB hit to RetrievedChunk."""
        m = hit["metadata"]
        pn = m.get("page_numbers", "1")
        if isinstance(pn, str):
            pn = [int(p) for p in pn.split(",") if p.strip()]

        secs = m.get("sections", "")
        if isinstance(secs, str):
            secs = [s.strip() for s in secs.split(",") if s.strip()]

        return RetrievedChunk(
            chunk_id=hit["id"],
            text=hit["text"],
            document_title=m.get("document_title", "Unknown"),
            document_filename=m.get("document_filename", "unknown.pdf"),
            page_numbers=pn,
            sections=secs,
            score=hit["distance"],
        )
