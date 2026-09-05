"""Offline tests for the MCP server. No models, no network.

The retriever is stubbed; these verify tool registration, payload shape,
and error paths — the contract an MCP client depends on.
"""
import json
import pathlib
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.mcp_server import server as srv
from src.mcp_server.server import (
    AskInput,
    GetChunkInput,
    SearchInput,
    standards_corpus_info,
    standards_get_chunk,
    standards_search,
)


def _fake_chunk():
    c = MagicMock()
    c.chunk_id = "std1-chunk-0001"
    c.citation = "[Source: Standard 1, Standard 1, pp.5,6]"
    c.document_title = "Standard 1"
    c.page_numbers = [5, 6]
    c.sections = ["Standard 1"]
    c.score = 0.91
    c.text = "Older people have the right to be treated with dignity."
    return c


def _stub_retriever():
    r = MagicMock()
    r.backend = "chroma"
    r.enable_hybrid = True
    r.retrieve.return_value = [_fake_chunk()]
    coll = MagicMock()
    coll.count.return_value = 95
    coll.get.return_value = {
        "ids": ["std1-chunk-0001"],
        "documents": ["Older people have the right to be treated with dignity."],
        "metadatas": [{"document_title": "Standard 1"}],
    }
    r.embedder.get_or_create_collection.return_value = coll
    return r


def setup_stub():
    srv._retriever = _stub_retriever()


def test_tools_registered():
    import anyio

    names = {t.name for t in anyio.run(srv.mcp.list_tools)}
    assert {"standards_search", "standards_get_chunk", "standards_ask", "standards_corpus_info"} <= names


def test_search_payload_shape():
    setup_stub()
    out = json.loads(standards_search(SearchInput(query="dignity and respect")))
    assert out["backend"] == "chroma"
    r0 = out["results"][0]
    assert r0["chunk_id"] == "std1-chunk-0001"
    assert r0["citation"].startswith("[Source:")
    assert "text" in r0 and r0["score"] == 0.91


def test_search_citations_only():
    setup_stub()
    out = json.loads(standards_search(SearchInput(query="dignity", include_text=False)))
    assert "text" not in out["results"][0]


def test_get_chunk_found_and_missing():
    setup_stub()
    ok = json.loads(standards_get_chunk(GetChunkInput(chunk_id="std1-chunk-0001")))
    assert ok["chunk_id"] == "std1-chunk-0001" and "dignity" in ok["text"]

    srv._retriever.embedder.get_or_create_collection.return_value.get.return_value = {
        "ids": [], "documents": [], "metadatas": []
    }
    missing = json.loads(standards_get_chunk(GetChunkInput(chunk_id="nope-12345")))
    assert "error" in missing and "hint" in missing


def test_corpus_info():
    setup_stub()
    out = json.loads(standards_corpus_info())
    assert out["chunk_count"] == 95 and out["backend"] == "chroma" and out["hybrid"] is True


def test_ask_reports_generation_failure_gracefully():
    setup_stub()
    srv._generator = MagicMock()
    srv._generator.generate.side_effect = Exception("no api key")
    from src.mcp_server.server import standards_ask

    out = json.loads(standards_ask(AskInput(question="What does Standard 1 require?")))
    assert "error" in out and "standards_search" in out["hint"]
    srv._generator = None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
