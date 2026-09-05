"""Offline tests for the Azure AI Search backend. No network, no Azure account.

The SDK surface is mocked; these verify our mapping layer — hit shape,
distance monotonicity, collection shim, index schema — which is where the
twin can silently diverge from the Chroma contract.
"""
import sys
import pathlib
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _store_with_mock_model():
    """Build an AzureSearchStore without downloading the embedding model."""
    with patch("src.retrieval.azure_search_store.SentenceTransformer") as st:
        model = MagicMock()
        model.encode.return_value = [[0.1] * 384]
        st.return_value = model
        from src.retrieval.azure_search_store import AzureSearchStore

        return AzureSearchStore({"embedding": {"model": "stub"}, "azure_search": {"index_name": "test-idx"}})


def test_query_maps_hits_to_chroma_shape():
    store = _store_with_mock_model()
    fake_result = {
        "id": "chunk-1",
        "text": "Standard 1 text",
        "document_title": "Standard 1",
        "document_filename": "std1.pdf",
        "page_numbers": "1,2",
        "sections": "Standard 1",
        "chunk_index": 0,
        "@search.score": 3.2,
    }
    client = MagicMock()
    client.search.return_value = [fake_result]
    store._search_client = client

    hits = store.query("dignity", top_k=3)
    assert len(hits) == 1
    h = hits[0]
    assert h["id"] == "chunk-1"
    assert h["text"] == "Standard 1 text"
    assert h["metadata"]["document_title"] == "Standard 1"
    assert h["metadata"]["page_numbers"] == "1,2"
    assert 0 < h["distance"] < 1  # flipped relevance, Chroma-style lower-is-better


def test_distance_flip_preserves_rank_order():
    store = _store_with_mock_model()
    mk = lambda i, s: {"id": f"c{i}", "text": "t", "@search.score": s}
    client = MagicMock()
    client.search.return_value = [mk(1, 5.0), mk(2, 2.0), mk(3, 0.5)]
    store._search_client = client

    hits = store.query("q", top_k=3)
    distances = [h["distance"] for h in hits]
    assert distances == sorted(distances), "higher Azure score must mean lower distance"


def test_collection_shim_get_and_count():
    from src.retrieval.azure_search_store import _CollectionShim

    client = MagicMock()
    client.get_document_count.return_value = 95
    client.get_document.return_value = {
        "id": "chunk-9",
        "text": "body",
        "document_title": "Guidance",
        "document_filename": "g.pdf",
        "page_numbers": "3",
        "sections": "Standard 2",
        "chunk_index": 9,
    }
    shim = _CollectionShim(client)

    assert shim.count() == 95
    got = shim.get(ids=["chunk-9"], include=["documents", "metadatas"])
    assert got["ids"] == ["chunk-9"]
    assert got["documents"] == ["body"]
    assert got["metadatas"][0]["sections"] == "Standard 2"


def test_collection_shim_skips_missing_ids():
    from src.retrieval.azure_search_store import _CollectionShim

    client = MagicMock()
    client.get_document.side_effect = Exception("404")
    shim = _CollectionShim(client)
    got = shim.get(ids=["nope"])
    assert got["ids"] == [] and got["documents"] == []


def test_index_definition_schema():
    store = _store_with_mock_model()
    idx = store.index_definition()
    names = {f.name for f in idx.fields}
    assert {"id", "text", "embedding", "document_title", "page_numbers", "sections"} <= names
    emb = next(f for f in idx.fields if f.name == "embedding")
    assert emb.vector_search_dimensions == 384


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
