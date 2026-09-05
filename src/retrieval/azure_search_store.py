"""Azure AI Search vector store — the Azure twin of the ChromaDB Embedder.

Duck-types the two surfaces the Retriever actually uses:
  - query(query_text, top_k) -> hit dicts {id, text, metadata, distance}
  - get_or_create_collection() -> object with .get(ids=..., include=...) and .count()

Embeddings stay local (same all-MiniLM-L6-v2 as the Chroma path) so both
backends rank over identical vectors — the comparison is store vs store,
not model vs model. Azure-side query is hybrid by construction: one call
runs vector similarity + BM25-style keyword search, fused by Azure.

Requires:
  AZURE_SEARCH_ENDPOINT  https://<service>.search.windows.net
  AZURE_SEARCH_KEY       admin key (index create + upload) or query key (query only)
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

EMBED_DIM = 384  # all-MiniLM-L6-v2


def _to_list(vec):
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def load_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _credential():
    from azure.core.credentials import AzureKeyCredential

    key = os.getenv("AZURE_SEARCH_KEY", "")
    if not key:
        raise RuntimeError("AZURE_SEARCH_KEY not set")
    return AzureKeyCredential(key)


def _endpoint():
    ep = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    if not ep:
        raise RuntimeError("AZURE_SEARCH_ENDPOINT not set")
    return ep


class _CollectionShim:
    """Chroma-collection-shaped facade over an Azure AI Search index."""

    def __init__(self, search_client):
        self._client = search_client

    def count(self):
        return self._client.get_document_count()

    def get(self, ids=None, include=None):
        docs, found_ids, metas = [], [], []
        for cid in ids or []:
            try:
                d = self._client.get_document(key=cid)
            except Exception:
                continue
            found_ids.append(cid)
            docs.append(d.get("text", ""))
            metas.append(_doc_to_metadata(d))
        return {"ids": found_ids, "documents": docs, "metadatas": metas}


def _doc_to_metadata(d):
    return {
        "chunk_id": d.get("id"),
        "document_title": d.get("document_title", "Unknown"),
        "document_filename": d.get("document_filename", "unknown.pdf"),
        "page_numbers": d.get("page_numbers", "1"),
        "sections": d.get("sections", ""),
        "chunk_index": d.get("chunk_index", 0),
    }


class AzureSearchStore:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        az = config.get("azure_search", {})
        self.index_name = az.get("index_name", "aged-care-standards")
        self.model_name = config["embedding"]["model"]
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._search_client = None

    # ── clients ──────────────────────────────────────────────────────────
    def search_client(self):
        if self._search_client is None:
            from azure.search.documents import SearchClient

            self._search_client = SearchClient(
                endpoint=_endpoint(), index_name=self.index_name, credential=_credential()
            )
        return self._search_client

    def index_client(self):
        from azure.search.documents.indexes import SearchIndexClient

        return SearchIndexClient(endpoint=_endpoint(), credential=_credential())

    # ── index lifecycle ──────────────────────────────────────────────────
    def index_definition(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="document_title", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="document_filename", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_numbers", type=SearchFieldDataType.String),
            SimpleField(name="sections", type=SearchFieldDataType.String),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBED_DIM,
                vector_search_profile_name="hnsw-profile",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-cosine")],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-cosine")],
        )
        return SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)

    def ensure_index(self):
        self.index_client().create_or_update_index(self.index_definition())
        print(f"Azure AI Search index ready: {self.index_name}")

    def reset(self):
        try:
            self.index_client().delete_index(self.index_name)
            self._search_client = None
        except Exception:
            pass

    # ── ingest ───────────────────────────────────────────────────────────
    def embed_chunks(self, chunks, batch_size=64):
        client = self.search_client()
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embs = self.model.encode([c.text for c in batch], show_progress_bar=False)
            docs = []
            for c, e in zip(batch, embs):
                m = c.to_metadata()
                docs.append(
                    {
                        "id": c.chunk_id,
                        "text": c.text,
                        "document_title": m.get("document_title", "Unknown"),
                        "document_filename": m.get("document_filename", "unknown.pdf"),
                        "page_numbers": str(m.get("page_numbers", "1")),
                        "sections": str(m.get("sections", "")),
                        "chunk_index": int(m.get("chunk_index", 0)),
                        "embedding": _to_list(e),
                    }
                )
            client.upload_documents(documents=docs)
            total += len(docs)
        print(f"Azure AI Search '{self.index_name}': uploaded {total} chunks")

    # ── query (hybrid: vector + keyword in one call) ─────────────────────
    def query(self, query_text, top_k=5):
        from azure.search.documents.models import VectorizedQuery

        qe = _to_list(self.model.encode([query_text])[0])
        vq = VectorizedQuery(vector=qe, k_nearest_neighbors=top_k, fields="embedding")
        results = self.search_client().search(
            search_text=query_text,
            vector_queries=[vq],
            top=top_k,
            select=["id", "text", "document_title", "document_filename", "page_numbers", "sections", "chunk_index"],
        )
        hits = []
        for r in results:
            score = float(r.get("@search.score", 0.0))
            hits.append(
                {
                    "id": r["id"],
                    "text": r.get("text", ""),
                    "metadata": _doc_to_metadata(r),
                    # Chroma reports cosine *distance* (lower = better); Azure reports
                    # relevance (higher = better). Monotonic flip keeps rank order.
                    "distance": 1.0 / (1.0 + score),
                }
            )
        return hits

    def get_or_create_collection(self):
        return _CollectionShim(self.search_client())
