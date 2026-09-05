"""Lightweight RAG observability: per-request traces with spans, tokens, cost.

You cannot govern what you cannot see. Every /query and MCP ask records one
trace line: what was asked, which backend answered, how long each stage took,
what it cost, and how confident retrieval was. JSONL sink, no external
service required; if LANGFUSE_PUBLIC_KEY is set the same trace is mirrored
to Langfuse (lazy import, failures never break the request).

Sink: $RAG_DATA_DIR/traces/traces.jsonl (repo data/ by default).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def traces_path() -> Path:
    data_dir = Path(os.getenv("RAG_DATA_DIR") or Path(__file__).parent.parent.parent / "data")
    p = data_dir / "traces"
    p.mkdir(parents=True, exist_ok=True)
    return p / "traces.jsonl"


class Tracer:
    """One instance per request. Collect spans, then finish() to persist."""

    def __init__(self, question: str, source: str = "api"):
        self.trace = {
            "trace_id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,           # api | mcp | eval
            "question": question[:300],
            "backend": None,
            "spans": [],                # [{name, ms}]
            "total_ms": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "model": None,
            "chunks_returned": 0,
            "top_score": None,
            "status": "ok",
            "error": None,
        }
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.trace["spans"].append({"name": name, "ms": round((time.perf_counter() - s) * 1000, 1)})

    def set_retrieval(self, chunks, backend: str):
        self.trace["backend"] = backend
        self.trace["chunks_returned"] = len(chunks)
        if chunks:
            self.trace["top_score"] = round(float(chunks[0].score), 4)

    def set_generation(self, response):
        self.trace["model"] = response.model
        self.trace["input_tokens"] = response.input_tokens
        self.trace["output_tokens"] = response.output_tokens
        self.trace["cost_usd"] = round(response.cost_usd, 6)

    def fail(self, error: str):
        self.trace["status"] = "error"
        self.trace["error"] = str(error)[:300]

    def finish(self) -> dict:
        self.trace["total_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        try:
            with open(traces_path(), "a") as f:
                f.write(json.dumps(self.trace, ensure_ascii=False) + "\n")
        except Exception:
            pass  # observability must never take down the request path
        self._mirror_langfuse()
        return self.trace

    def _mirror_langfuse(self):
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return
        try:
            from langfuse import Langfuse  # optional dependency

            lf = Langfuse()
            t = lf.trace(name="rag_query", input=self.trace["question"], metadata={
                "backend": self.trace["backend"], "source": self.trace["source"]})
            for s in self.trace["spans"]:
                t.span(name=s["name"], metadata={"ms": s["ms"]})
            t.update(output={"status": self.trace["status"], "cost_usd": self.trace["cost_usd"]})
        except Exception:
            pass


def load_traces(limit: int = 500) -> list[dict]:
    """Most-recent-first list of traces for the dashboard."""
    p = traces_path()
    if not p.exists():
        return []
    with open(p) as f:
        lines = f.readlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))
