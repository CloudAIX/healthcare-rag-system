"""Retriever — vector search from ChromaDB."""
from dataclasses import dataclass
from pathlib import Path
import yaml
from src.ingestion.embedder import Embedder

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
    with open(p) as f: return yaml.safe_load(f)

class Retriever:
    def __init__(self, embedder=None):
        self.config = load_retrieval_config()
        self.embedder = embedder or Embedder(self.config)
        self.top_k = self.config["retrieval"]["top_k_vector"]
    def retrieve(self, query, top_k=None):
        k = top_k or self.top_k
        raw = self.embedder.query(query, top_k=k)
        chunks = []
        for hit in raw:
            m = hit["metadata"]
            pn = m.get("page_numbers","1")
            if isinstance(pn, str): pn = [int(p) for p in pn.split(",") if p.strip()]
            secs = m.get("sections","")
            if isinstance(secs, str): secs = [s.strip() for s in secs.split(",") if s.strip()]
            chunks.append(RetrievedChunk(chunk_id=hit["id"], text=hit["text"],
                document_title=m.get("document_title","Unknown"),
                document_filename=m.get("document_filename","unknown.pdf"),
                page_numbers=pn, sections=secs, score=hit["distance"]))
        return chunks
