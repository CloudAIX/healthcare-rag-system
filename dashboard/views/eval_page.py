"""Evaluation page — display RAGAS metric results and analysis."""
import json
from pathlib import Path
import streamlit as st


RESULTS_DIR = Path(__file__).parent.parent.parent / "eval" / "results"
GOLDEN_PATH = Path(__file__).parent.parent.parent / "eval" / "golden_dataset.json"

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "citation_accuracy": 0.90,
}

METRIC_DESCRIPTIONS = {
    "faithfulness": "Are answer claims grounded in retrieved context? (no hallucination)",
    "answer_relevancy": "Does the answer address the question asked?",
    "context_precision": "Are retrieved contexts relevant to the question?",
    "citation_accuracy": "Do [Source: ...] citations map to real chunks?",
}


def _load_results():
    """Load all evaluation result files."""
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob("eval-*.json"), reverse=True)
    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def render():
    st.header("Evaluation Dashboard")
    st.markdown("RAGAS-style metrics for RAG pipeline quality assessment.")

    results = _load_results()

    if not results:
        st.info("No evaluation results yet. Run the evaluation first:")
        st.code("cd healthcare-rag-system && ./venv/bin/python scripts/run_eval.py", language="bash")

        # Show golden dataset
        st.subheader("Golden Dataset")
        golden = _load_golden()
        st.markdown(f"**{len(golden)} evaluation items** across categories:")

        categories = {}
        for item in golden:
            cat = item.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

        cols = st.columns(len(categories))
        for i, (cat, count) in enumerate(categories.items()):
            cols[i].metric(cat.replace("_", " ").title(), count)

        st.divider()
        for item in golden:
            with st.expander(f"[{item['id']}] {item['question'][:70]}"):
                st.markdown(f"**Category:** {item['category']} | **Difficulty:** {item['difficulty']}")
                st.markdown(f"**Ground Truth:** {item.get('ground_truth', 'N/A')}")
                sources = item.get("expected_sources", [])
                if sources:
                    st.markdown(f"**Expected Sources:** {', '.join(sources)}")
        return

    # ── Latest run summary ──
    latest = results[0]
    st.subheader(f"Latest Run: {latest['run_id']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Items Evaluated", latest["total_items"])
    col2.metric("Pass Rate", f"{latest['pass_rate']:.0%}")
    col3.metric("Duration", f"{latest['duration_seconds']:.1f}s")

    # ── Metric averages vs thresholds ──
    st.subheader("Metric Averages vs Thresholds")

    metric_cols = st.columns(4)
    for i, (metric, avg) in enumerate(latest.get("metric_averages", {}).items()):
        threshold = THRESHOLDS.get(metric, 0.5)
        delta = avg - threshold
        metric_cols[i].metric(
            metric.replace("_", " ").title(),
            f"{avg:.3f}",
            delta=f"{delta:+.3f}",
            delta_color="normal" if delta >= 0 else "inverse",
            help=METRIC_DESCRIPTIONS.get(metric, ""),
        )

    # ── Bar chart of metrics ──
    st.subheader("Metric Comparison")
    chart_data = {
        "Metric": [],
        "Score": [],
        "Type": [],
    }
    for metric, avg in latest.get("metric_averages", {}).items():
        chart_data["Metric"].append(metric.replace("_", " ").title())
        chart_data["Score"].append(avg)
        chart_data["Type"].append("Actual")
    for metric, thresh in THRESHOLDS.items():
        chart_data["Metric"].append(metric.replace("_", " ").title())
        chart_data["Score"].append(thresh)
        chart_data["Type"].append("Threshold")

    st.bar_chart(
        data={"Score": [r["Score"] for r in [
            {"m": m, "Score": s} for m, s in zip(chart_data["Metric"], chart_data["Score"])
            if chart_data["Type"][chart_data["Metric"].index(m)] == "Actual"
        ]]},
        use_container_width=True,
    )

    # ── Category breakdown ──
    cat_breakdown = latest.get("category_breakdown", {})
    if cat_breakdown:
        st.subheader("Category Breakdown")
        for cat, scores in cat_breakdown.items():
            with st.expander(f"{cat.replace('_', ' ').title()}"):
                cat_cols = st.columns(len(scores))
                for j, (m, s) in enumerate(scores.items()):
                    threshold = THRESHOLDS.get(m, 0.5)
                    status = "PASS" if s >= threshold else "FAIL"
                    cat_cols[j].metric(
                        m.replace("_", " ").title(),
                        f"{s:.3f}",
                        delta=status,
                        delta_color="normal" if status == "PASS" else "inverse",
                    )

    # ── Per-item results ──
    st.subheader("Per-Item Results")
    item_results = latest.get("results", [])
    for item in item_results:
        scores = item.get("scores", {})
        all_pass = all(
            scores.get(m, 0) >= THRESHOLDS.get(m, 0.5) for m in THRESHOLDS
        )
        icon = "✅" if all_pass else "⚠️"
        with st.expander(f"{icon} [{item['item_id']}] {item['question'][:60]}"):
            score_cols = st.columns(len(scores))
            for j, (m, s) in enumerate(scores.items()):
                threshold = THRESHOLDS.get(m, 0.5)
                score_cols[j].metric(
                    m.replace("_", " ").title(),
                    f"{s:.3f}",
                    delta="PASS" if s >= threshold else "FAIL",
                    delta_color="normal" if s >= threshold else "inverse",
                )

            # Show details
            details = item.get("details", {})
            if details:
                with st.container():
                    st.markdown("**Details:**")
                    st.json(details)

    # ── Run history ──
    if len(results) > 1:
        st.subheader("Run History")
        for run in results:
            st.text(
                f"{run['run_id']} | "
                f"Items: {run['total_items']} | "
                f"Pass: {run['pass_rate']:.0%} | "
                f"{run['duration_seconds']:.1f}s"
            )
