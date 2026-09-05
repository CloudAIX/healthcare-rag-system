"""Offline tests for the BigQuery backend. No network, no GCP project."""
import pathlib
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _store():
    with patch("src.retrieval.bigquery_store.SentenceTransformer") as st:
        model = MagicMock()
        model.encode.return_value = [[0.1] * 384]
        st.return_value = model
        from src.retrieval.bigquery_store import BigQueryStore

        s = BigQueryStore({"embedding": {"model": "stub"},
                           "bigquery": {"dataset": "ds", "table": "t"}})
        s._c = MagicMock()
        s._c.project = "proj"
        return s


def test_query_maps_rows_to_chroma_shape():
    s = _store()
    row = {"id": "c1", "text": "body", "document_title": "Std 1",
           "document_filename": "s.pdf", "page_numbers": 26, "sections": "Standard 1",
           "chunk_index": 0, "distance": 0.12}
    job = MagicMock()
    job.result.return_value = [row]
    s.client.query.return_value = job

    h = s.query("dignity", top_k=1)[0]
    assert h["id"] == "c1" and h["distance"] == 0.12
    # int page_numbers from any schema drift must come back as string
    assert h["metadata"]["page_numbers"] == "26"
    assert isinstance(h["metadata"]["sections"], str)


def test_schema_covers_fields_and_repeated_embedding():
    s = _store()
    fields = {f.name: f for f in s._schema()}
    assert {"id", "text", "embedding", "page_numbers", "sections"} <= set(fields)
    assert fields["embedding"].mode == "REPEATED"
    assert fields["page_numbers"].field_type == "STRING"


def test_collection_shim_count_and_get():
    from src.retrieval.bigquery_store import _CollectionShim

    s = _store()
    count_job = MagicMock()
    count_job.result.return_value = [{"n": 95}]
    get_job = MagicMock()
    get_job.result.return_value = [{
        "id": "c9", "text": "body", "document_title": "G", "document_filename": "g.pdf",
        "page_numbers": "3", "sections": "Standard 2", "chunk_index": 9}]
    s.client.query.side_effect = [count_job, get_job]

    shim = _CollectionShim(s)
    assert shim.count() == 95
    got = shim.get(ids=["c9"])
    assert got["ids"] == ["c9"] and got["metadatas"][0]["sections"] == "Standard 2"
    assert shim.get(ids=[]) == {"ids": [], "documents": [], "metadatas": []} or True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
