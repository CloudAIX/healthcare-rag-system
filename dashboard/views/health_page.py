"""System Health page — pipeline status, configuration, and diagnostics."""
import os
from pathlib import Path
import streamlit as st
import yaml


CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _check_env():
    """Check environment variables."""
    return {
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "not set (using config default)"),
        "EVAL_MODEL": os.getenv("EVAL_MODEL", "not set (using config default)"),
    }


def _check_chromadb():
    """Check ChromaDB status."""
    try:
        import chromadb
        from chromadb.config import Settings

        config = _load_config()
        persist_dir = config["vector_store"]["persist_directory"]
        client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        colls = client.list_collections()
        coll_info = []
        for c in colls:
            coll_info.append({"name": c.name, "count": c.count()})
        return {"status": "connected", "collections": coll_info, "path": persist_dir}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _check_bm25():
    """Check BM25 index status."""
    bm25_path = DATA_DIR / "processed" / "bm25_index.pkl"
    if bm25_path.exists():
        return {"status": "available", "size_kb": bm25_path.stat().st_size / 1024}
    return {"status": "missing"}


def _check_models():
    """Check if ML models are available."""
    checks = {}
    try:
        from sentence_transformers import SentenceTransformer
        checks["embedding_model"] = "available"
    except ImportError:
        checks["embedding_model"] = "missing (install sentence-transformers)"

    try:
        from sentence_transformers import CrossEncoder
        checks["reranker_model"] = "available"
    except ImportError:
        checks["reranker_model"] = "missing (install sentence-transformers)"

    try:
        import anthropic
        checks["anthropic_sdk"] = "available"
    except ImportError:
        checks["anthropic_sdk"] = "missing (install anthropic)"

    return checks


def _check_data():
    """Check data directories."""
    raw_dir = DATA_DIR / "raw"
    pdfs = list(raw_dir.glob("*.pdf")) if raw_dir.exists() else []
    chroma_dir = DATA_DIR / "processed" / "chroma"
    return {
        "raw_pdfs": len(pdfs),
        "raw_dir_exists": raw_dir.exists(),
        "chroma_dir_exists": chroma_dir.exists(),
        "chroma_size_mb": sum(
            f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file()
        ) / 1024 / 1024 if chroma_dir.exists() else 0,
    }


def render():
    st.header("System Health")
    st.markdown("Pipeline status, configuration, and diagnostics.")

    config = _load_config()

    # ── Overall Status ──
    st.subheader("Component Status")

    env = _check_env()
    chroma = _check_chromadb()
    bm25 = _check_bm25()
    data = _check_data()
    models = _check_models()

    # Status indicators
    components = [
        ("API Key", "ok" if env["ANTHROPIC_API_KEY"] else "missing"),
        ("ChromaDB", chroma["status"]),
        ("BM25 Index", bm25["status"]),
        ("PDF Corpus", "available" if data["raw_pdfs"] > 0 else "missing"),
        ("Embedding Model", models.get("embedding_model", "unknown")),
        ("Reranker Model", models.get("reranker_model", "unknown")),
        ("Anthropic SDK", models.get("anthropic_sdk", "unknown")),
    ]

    cols = st.columns(4)
    for i, (name, status) in enumerate(components):
        col = cols[i % 4]
        if status in ("ok", "connected", "available"):
            col.success(f"**{name}**  \n{status}")
        elif status == "missing":
            col.error(f"**{name}**  \n{status}")
        else:
            col.warning(f"**{name}**  \n{status}")

    # ── Data Stats ──
    st.subheader("Data Statistics")
    d1, d2, d3 = st.columns(3)
    d1.metric("PDF Documents", data["raw_pdfs"])

    chroma_chunks = 0
    if chroma["status"] == "connected":
        for c in chroma.get("collections", []):
            chroma_chunks += c["count"]
    d2.metric("ChromaDB Chunks", chroma_chunks)

    if bm25["status"] == "available":
        d3.metric("BM25 Index Size", f"{bm25['size_kb']:.1f} KB")
    else:
        d3.metric("BM25 Index Size", "N/A")

    # ── Configuration ──
    st.subheader("Pipeline Configuration")

    tab1, tab2, tab3, tab4 = st.tabs(["Retrieval", "Embedding", "Generation", "Evaluation"])

    with tab1:
        ret_cfg = config.get("retrieval", {})
        rerank_cfg = config.get("reranker", {})
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Vector top_k", ret_cfg.get("top_k_vector", "N/A"))
        r2.metric("BM25 top_k", ret_cfg.get("top_k_bm25", "N/A"))
        r3.metric("RRF k", ret_cfg.get("rrf_k", "N/A"))
        r4.metric("Rerank top_k", ret_cfg.get("top_k_rerank", "N/A"))
        st.markdown(f"**Reranker model:** `{rerank_cfg.get('model', 'N/A')}`")
        st.markdown(f"**Min score:** {rerank_cfg.get('min_score', 'N/A')}")

    with tab2:
        emb_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        e1, e2 = st.columns(2)
        e1.metric("Model", emb_cfg.get("model", "N/A"))
        e2.metric("Dimensions", emb_cfg.get("dimension", "N/A"))
        st.markdown(f"**Store:** {vs_cfg.get('type', 'N/A')}")
        st.markdown(f"**Collection:** `{vs_cfg.get('collection_name', 'N/A')}`")
        st.markdown(f"**Persist dir:** `{vs_cfg.get('persist_directory', 'N/A')}`")

    with tab3:
        gen_cfg = config.get("generation", {})
        g1, g2, g3 = st.columns(3)
        g1.metric("Model", gen_cfg.get("model", "N/A"))
        g2.metric("Max Tokens", gen_cfg.get("max_tokens", "N/A"))
        g3.metric("Temperature", gen_cfg.get("temperature", "N/A"))

    with tab4:
        eval_cfg = config.get("evaluation", {})
        if eval_cfg:
            ecols = st.columns(len(eval_cfg))
            for i, (metric, threshold) in enumerate(eval_cfg.items()):
                ecols[i].metric(
                    metric.replace("_", " ").title(),
                    f"{threshold:.2f}",
                    help=f"Minimum threshold for {metric}",
                )
        else:
            st.info("No evaluation thresholds configured.")

    # ── Environment ──
    st.subheader("Environment")
    st.markdown(f"**ANTHROPIC_API_KEY:** {'✅ Set' if env['ANTHROPIC_API_KEY'] else '❌ Missing'}")
    st.markdown(f"**ANTHROPIC_MODEL:** {env['ANTHROPIC_MODEL']}")
    st.markdown(f"**EVAL_MODEL:** {env['EVAL_MODEL']}")

    # ── Raw config ──
    with st.expander("Raw Configuration (YAML)"):
        st.code(yaml.dump(config, default_flow_style=False), language="yaml")

    # ── Quick Actions ──
    st.subheader("Quick Actions")
    st.markdown("Common pipeline commands:")
    actions = {
        "Download corpus": "./venv/bin/python scripts/download_corpus.py",
        "Run ingestion": "./venv/bin/python scripts/ingest.py --reset",
        "Run evaluation": "./venv/bin/python scripts/run_eval.py",
        "Start API server": "./venv/bin/uvicorn src.api.app:app --reload",
    }
    for name, cmd in actions.items():
        st.code(cmd, language="bash")
