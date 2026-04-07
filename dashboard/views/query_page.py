"""Query page — interactive RAG question-answering interface."""
import time
import streamlit as st


def _get_pipeline():
    """Lazy-load the RAG pipeline (cached across reruns)."""
    if "retriever" not in st.session_state:
        try:
            from src.retrieval.retriever import Retriever
            from src.generation.generator import Generator

            st.session_state.retriever = Retriever()
            st.session_state.generator = Generator()
            st.session_state.pipeline_error = None
        except Exception as e:
            st.session_state.retriever = None
            st.session_state.generator = None
            st.session_state.pipeline_error = str(e)
    return st.session_state.get("retriever"), st.session_state.get("generator")


def render():
    st.header("Query Interface")
    st.markdown("Ask questions about Australian Aged Care Quality Standards.")

    # Sample questions
    samples = [
        "What are the 7 strengthened aged care quality standards?",
        "What does Standard 1 require regarding dignity and respect?",
        "What are the medication management requirements?",
        "What infection prevention and control measures are required?",
        "What are the food and nutrition requirements under Standard 6?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Your question",
            height=80,
            placeholder="e.g. What are the requirements for advance care planning?",
        )
    with col2:
        st.markdown("**Sample questions:**")
        for sq in samples:
            if st.button(sq[:50] + "...", key=f"sample_{hash(sq)}"):
                question = sq

    top_k = st.slider("Retrieval depth (top_k)", 1, 10, 5)

    if st.button("Ask", type="primary", disabled=not question):
        retriever, generator = _get_pipeline()

        if st.session_state.get("pipeline_error"):
            st.error(f"Pipeline not available: {st.session_state.pipeline_error}")
            st.info("Ensure the ingestion pipeline has been run first.")
            return

        with st.spinner("Retrieving and generating..."):
            t0 = time.perf_counter()

            # Retrieve
            chunks = retriever.retrieve(question, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t0) * 1000

            if not chunks:
                st.warning("No relevant documents found for this query.")
                return

            # Generate
            t1 = time.perf_counter()
            response = generator.generate(question, chunks)
            generation_ms = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000

        # ── Answer ──
        st.subheader("Answer")
        st.markdown(response.answer)

        # ── Metrics row ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Latency", f"{total_ms:.0f}ms")
        m2.metric("Retrieval", f"{retrieval_ms:.0f}ms")
        m3.metric("Generation", f"{generation_ms:.0f}ms")
        m4.metric("Cost", f"${response.cost_usd:.4f}")

        # ── Token usage ──
        t1c, t2c = st.columns(2)
        t1c.metric("Input Tokens", f"{response.input_tokens:,}")
        t2c.metric("Output Tokens", f"{response.output_tokens:,}")

        # ── Retrieved chunks ──
        st.subheader("Retrieved Chunks")
        for i, chunk in enumerate(chunks):
            with st.expander(
                f"Chunk {i+1} — {chunk.document_title} | Score: {chunk.score:.3f}"
            ):
                st.markdown(f"**Source:** {chunk.document_filename}")
                st.markdown(f"**Pages:** {', '.join(str(p) for p in chunk.page_numbers)}")
                st.markdown(f"**Sections:** {', '.join(chunk.sections) if chunk.sections else 'N/A'}")
                st.markdown(f"**Citation:** `{chunk.citation}`")
                st.divider()
                st.text(chunk.text)

    # ── Query history ──
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if question and st.session_state.get("retriever"):
        st.session_state.query_history.append(question)

    if st.session_state.query_history:
        with st.expander("Query History"):
            for i, q in enumerate(reversed(st.session_state.query_history[-10:])):
                st.text(f"{i+1}. {q}")
