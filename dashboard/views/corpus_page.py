"""Corpus page — browse documents, chunks, and collection statistics."""
from pathlib import Path
import streamlit as st
import yaml


RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_pdf_stats():
    """Get basic file stats for each PDF."""
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    stats = []
    for p in pdfs:
        stats.append({
            "filename": p.name,
            "size_kb": p.stat().st_size / 1024,
            "size_mb": p.stat().st_size / 1024 / 1024,
        })
    return stats


def _get_collection_stats():
    """Get ChromaDB collection stats."""
    try:
        import chromadb
        from chromadb.config import Settings

        config = _load_config()
        persist_dir = config["vector_store"]["persist_directory"]
        collection_name = config["vector_store"]["collection_name"]

        client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        coll = client.get_or_create_collection(collection_name)
        count = coll.count()

        # Get sample if available
        sample = None
        if count > 0:
            sample = coll.get(limit=5, include=["documents", "metadatas"])

        return {
            "collection_name": collection_name,
            "chunk_count": count,
            "persist_dir": persist_dir,
            "sample": sample,
        }
    except Exception as e:
        return {"error": str(e), "chunk_count": 0}


def _get_bm25_stats():
    """Get BM25 index stats."""
    bm25_path = Path(__file__).parent.parent.parent / "data" / "processed" / "bm25_index.pkl"
    if not bm25_path.exists():
        return {"exists": False}
    return {
        "exists": True,
        "size_kb": bm25_path.stat().st_size / 1024,
    }


def render():
    st.header("Corpus Explorer")
    st.markdown("Browse the document corpus and index statistics.")

    config = _load_config()

    # ── PDF Corpus ──
    st.subheader("PDF Corpus")
    pdf_stats = _get_pdf_stats()

    if not pdf_stats:
        st.warning("No PDFs found. Run the download script first.")
        st.code("./venv/bin/python scripts/download_corpus.py", language="bash")
        return

    total_size = sum(p["size_mb"] for p in pdf_stats)
    c1, c2 = st.columns(2)
    c1.metric("Documents", len(pdf_stats))
    c2.metric("Total Size", f"{total_size:.1f} MB")

    for pdf in pdf_stats:
        with st.expander(f"{pdf['filename']} ({pdf['size_kb']:.0f} KB)"):
            st.markdown(f"**File:** `data/raw/{pdf['filename']}`")
            st.markdown(f"**Size:** {pdf['size_kb']:.1f} KB ({pdf['size_mb']:.2f} MB)")

    # ── Chunking Config ──
    st.subheader("Chunking Configuration")
    chunk_cfg = config.get("chunking", {})
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Method", chunk_cfg.get("method", "N/A"))
    cc2.metric("Chunk Size", chunk_cfg.get("chunk_size", "N/A"))
    cc3.metric("Overlap", chunk_cfg.get("chunk_overlap", "N/A"))

    # ── Vector Store (ChromaDB) ──
    st.subheader("Vector Store (ChromaDB)")
    coll_stats = _get_collection_stats()

    if "error" in coll_stats:
        st.error(f"Could not connect to ChromaDB: {coll_stats['error']}")
    else:
        vc1, vc2 = st.columns(2)
        vc1.metric("Collection", coll_stats["collection_name"])
        vc2.metric("Chunks Indexed", coll_stats["chunk_count"])

        if coll_stats["chunk_count"] == 0:
            st.warning("ChromaDB collection is empty. Run ingestion first:")
            st.code("./venv/bin/python scripts/ingest.py", language="bash")
        elif coll_stats.get("sample"):
            st.markdown("**Sample chunks:**")
            sample = coll_stats["sample"]
            for i, (doc_id, doc_text) in enumerate(
                zip(sample["ids"][:5], sample["documents"][:5])
            ):
                meta = sample["metadatas"][i] if sample.get("metadatas") else {}
                with st.expander(f"Chunk: {doc_id}"):
                    if meta:
                        st.markdown(f"**Source:** {meta.get('document_title', 'N/A')}")
                        st.markdown(f"**File:** {meta.get('document_filename', 'N/A')}")
                        st.markdown(f"**Pages:** {meta.get('page_numbers', 'N/A')}")
                    st.text(doc_text[:500] + ("..." if len(doc_text) > 500 else ""))

    # ── BM25 Index ──
    st.subheader("BM25 Index")
    bm25_stats = _get_bm25_stats()
    if bm25_stats["exists"]:
        st.metric("Index Size", f"{bm25_stats['size_kb']:.1f} KB")
    else:
        st.warning("BM25 index not found. It will be built during ingestion.")

    # ── Embedding Model ──
    st.subheader("Embedding Configuration")
    emb_cfg = config.get("embedding", {})
    ec1, ec2 = st.columns(2)
    ec1.metric("Model", emb_cfg.get("model", "N/A"))
    ec2.metric("Dimensions", emb_cfg.get("dimension", "N/A"))

    # ── Parse & Preview (optional) ──
    st.divider()
    st.subheader("Parse Preview")
    st.markdown("Parse a PDF and preview its chunks (slow — loads models).")

    if st.button("Parse All PDFs & Preview Chunks"):
        with st.spinner("Parsing PDFs..."):
            try:
                from src.ingestion.pdf_parser import parse_all_pdfs
                from src.ingestion.chunker import chunk_all_documents

                documents = parse_all_pdfs(RAW_DIR)
                chunks = chunk_all_documents(documents)

                st.success(f"Parsed {len(documents)} documents into {len(chunks)} chunks.")

                for doc in documents:
                    with st.expander(f"{doc.filename} — {len(doc.pages)} pages"):
                        st.markdown(f"**Title:** {doc.title}")
                        for page in doc.pages[:3]:
                            st.text(f"Page {page.page_number}: {page.text[:200]}...")

                st.subheader(f"Chunk Preview (first 10 of {len(chunks)})")
                for chunk in chunks[:10]:
                    with st.expander(f"{chunk.chunk_id}"):
                        st.markdown(f"**Source:** {chunk.document_title}")
                        st.markdown(f"**Pages:** {chunk.page_numbers}")
                        st.markdown(f"**Sections:** {chunk.sections}")
                        st.text(chunk.text[:400])
            except Exception as e:
                st.error(f"Parse failed: {e}")
