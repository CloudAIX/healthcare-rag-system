"""Healthcare RAG System — Streamlit Dashboard.

Multi-page app with 4 views:
  1. Query    — Ask questions, see retrieval + generation
  2. Eval     — Evaluation results, metrics, category breakdown
  3. Corpus   — Browse documents, chunks, collection stats
  4. Health   — Pipeline status, index sizes, model info
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Healthcare RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────

st.sidebar.title("Healthcare RAG")
st.sidebar.caption("Aged Care Standards Compliance")

page = st.sidebar.radio(
    "Navigation",
    ["Query", "Evaluation", "Corpus", "System Health"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Pipeline:** Vector + BM25 + RRF + Re-ranking  \n"
    "**LLM:** Claude Sonnet  \n"
    "**Eval:** RAGAS-style metrics"
)

# ── Page routing ─────────────────────────────────────────────────────────────

if page == "Query":
    from views import query_page
    query_page.render()
elif page == "Evaluation":
    from views import eval_page
    eval_page.render()
elif page == "Corpus":
    from views import corpus_page
    corpus_page.render()
elif page == "System Health":
    from views import health_page
    health_page.render()
