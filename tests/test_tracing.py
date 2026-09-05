"""Offline tests for the observability tracer."""
import json
import pathlib
import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _tracer_in(tmp_path, monkeypatch, question="test q", source="api"):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from src.observability import Tracer

    return Tracer(question, source=source)


def _read_traces(tmp_path):
    p = tmp_path / "traces" / "traces.jsonl"
    return [json.loads(l) for l in open(p)]


def test_trace_written_with_spans_and_totals(tmp_path, monkeypatch):
    t = _tracer_in(tmp_path, monkeypatch)
    with t.span("retrieve"):
        time.sleep(0.01)
    chunk = MagicMock(score=0.87)
    t.set_retrieval([chunk], backend="chroma")
    resp = MagicMock(model="claude-x", input_tokens=1000, output_tokens=200, cost_usd=0.006)
    t.set_generation(resp)
    t.finish()

    rows = _read_traces(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["status"] == "ok" and r["backend"] == "chroma"
    assert r["spans"][0]["name"] == "retrieve" and r["spans"][0]["ms"] >= 10
    assert r["total_ms"] >= r["spans"][0]["ms"]
    assert r["input_tokens"] == 1000 and r["cost_usd"] == 0.006
    assert r["top_score"] == 0.87 and r["chunks_returned"] == 1


def test_error_path_recorded(tmp_path, monkeypatch):
    t = _tracer_in(tmp_path, monkeypatch, source="mcp")
    t.fail("no relevant documents")
    t.finish()
    r = _read_traces(tmp_path)[0]
    assert r["status"] == "error" and "no relevant" in r["error"] and r["source"] == "mcp"


def test_load_traces_most_recent_first(tmp_path, monkeypatch):
    for q in ["first", "second"]:
        t = _tracer_in(tmp_path, monkeypatch, question=q)
        t.finish()
    from src.observability import load_traces

    rows = load_traces()
    assert rows[0]["question"] == "second" and rows[1]["question"] == "first"


def test_question_truncated(tmp_path, monkeypatch):
    t = _tracer_in(tmp_path, monkeypatch, question="x" * 1000)
    t.finish()
    assert len(_read_traces(tmp_path)[0]["question"]) == 300


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
