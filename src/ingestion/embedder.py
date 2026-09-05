"""Embedder — embeds chunks and stores in ChromaDB."""
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml
from .chunker import Chunk

def load_embedding_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f: return yaml.safe_load(f)

class Embedder:
    def __init__(self, config=None):
        if config is None: config = load_embedding_config()
        self.model_name = config["embedding"]["model"]
        import os as _os
        persist = Path(config["vector_store"]["persist_directory"])
        if not persist.is_absolute():
            # Anchor relative paths to the data dir, not the caller's cwd —
            # MCP clients and services launch this from arbitrary directories.
            # RAG_DATA_DIR overrides (also used by tests).
            data_dir = Path(_os.getenv("RAG_DATA_DIR") or Path(__file__).parent.parent.parent / "data")
            # config paths look like ./data/processed/chroma — strip the data/ prefix
            parts = [p for p in persist.parts if p not in (".", "data")]
            persist = data_dir.joinpath(*parts)
        self.persist_dir = str(persist)
        self.collection_name = config["vector_store"]["collection_name"]
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))

    def get_or_create_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space":"cosine"})

    def embed_chunks(self, chunks, batch_size=64):
        coll = self.get_or_create_collection()
        existing = set(coll.get()["ids"]) if coll.count()>0 else set()
        new = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            print(f"All {len(chunks)} chunks already embedded.")
            return
        print(f"\nEmbedding {len(new)} new chunks")
        for i in tqdm(range(0, len(new), batch_size), desc="Embedding"):
            batch = new[i:i+batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metas = [c.to_metadata() for c in batch]
            embs = self.model.encode(texts, show_progress_bar=False).tolist()
            coll.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        print(f"\nChromaDB '{self.collection_name}': {coll.count()} chunks stored")

    def query(self, query_text, top_k=5):
        coll = self.get_or_create_collection()
        qe = self.model.encode([query_text]).tolist()
        res = coll.query(query_embeddings=qe, n_results=top_k, include=["documents","metadatas","distances"])
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({"id":res["ids"][0][i], "text":res["documents"][0][i],
                         "metadata":res["metadatas"][0][i], "distance":res["distances"][0][i]})
        return hits

    def reset(self):
        try: self.client.delete_collection(self.collection_name)
        except: pass
