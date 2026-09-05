"""Observability page — traces, latency, tokens, cost.

You cannot govern what you cannot see: every API and MCP query lands here
as one trace (spans, tokens, cost, top retrieval score, status).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.observability import load_traces, traces_path


def render():
    st.title("Observability")
    st.caption("Per-request traces from the API and MCP server. "
               f"Sink: `{traces_path()}` (JSONL; mirrored to Langfuse when configured).")

    traces = load_traces()
    if not traces:
        st.info("No traces yet. Run a query through the API (`/query`) or the MCP "
                "server (`standards_ask`) and it will appear here.")
        return

    df = pd.DataFrame(traces)
    df["ts"] = pd.to_datetime(df["ts"])
    ok = df[df["status"] == "ok"]

    # ── Headline metrics ────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Requests", len(df))
    c2.metric("Errors", int((df["status"] != "ok").sum()))
    if len(ok):
        c3.metric("p50 / p95 latency",
                  f"{ok['total_ms'].quantile(.5):,.0f} / {ok['total_ms'].quantile(.95):,.0f} ms")
        c4.metric("Total cost", f"${ok['cost_usd'].sum():.4f}")
        c5.metric("Tokens in / out",
                  f"{int(ok['input_tokens'].sum()):,} / {int(ok['output_tokens'].sum()):,}")

    # ── Latency breakdown by span ───────────────────────────────────────
    st.subheader("Latency by stage")
    rows = []
    for t in traces:
        for s in t.get("spans", []):
            rows.append({"ts": t["ts"], "stage": s["name"], "ms": s["ms"]})
    if rows:
        span_df = pd.DataFrame(rows)
        agg = span_df.groupby("stage")["ms"].agg(["median", "max", "count"]).round(1)
        agg.columns = ["median ms", "max ms", "calls"]
        st.dataframe(agg, use_container_width=True)

    # ── Cost & latency over time ────────────────────────────────────────
    if len(ok) > 1:
        st.subheader("Latency over time")
        st.line_chart(ok.set_index("ts")["total_ms"])
        st.subheader("Cost per request (USD)")
        st.bar_chart(ok.set_index("ts")["cost_usd"])

    # ── Backend split ───────────────────────────────────────────────────
    if df["backend"].notna().any():
        st.subheader("By backend")
        be = df.groupby("backend").agg(
            requests=("trace_id", "count"),
            median_ms=("total_ms", "median"),
            total_cost=("cost_usd", "sum"),
            avg_top_score=("top_score", "mean"),
        ).round(4)
        st.dataframe(be, use_container_width=True)

    # ── Raw traces ──────────────────────────────────────────────────────
    st.subheader("Recent traces")
    show = df[["ts", "source", "backend", "question", "total_ms",
               "input_tokens", "output_tokens", "cost_usd", "top_score", "status"]].head(50)
    st.dataframe(show, use_container_width=True, hide_index=True)
