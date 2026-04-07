# Healthcare RAG System — Continuation Primer

## Context

You are continuing work on a **Healthcare RAG System** — a production-grade Retrieval-Augmented Generation pipeline for Australian Aged Care Quality Standards compliance. This is a portfolio project at:

```
/Users/Sree/AI Projects/portfolio/healthcare-rag-system/
```

**Python environment:** `./venv/bin/python` (Python 3.11, NOT `python` or `python3`)

---

## What's Built (Phases 1-4 Complete)

### Phase 1 — Ingestion Pipeline
- `src/ingestion/pdf_parser.py` — PyMuPDF extraction with section detection (Standard X, Outcome X.X, Action X.X.X)
- `src/ingestion/chunker.py` — Recursive chunking (700×4 chars, 100×4 overlap), MD5-based chunk IDs
- `src/ingestion/embedder.py` — SentenceTransformer (all-MiniLM-L6-v2, 384 dims) + ChromaDB PersistentClient
- 5 PDFs in `data/raw/` (4.4MB): aged care standards, guidance materials, provider checklist

### Phase 2 — Hybrid Retrieval
- `src/retrieval/retriever.py` — Orchestrator: vector → BM25 → RRF → re-rank, graceful fallback to vector-only
- `src/retrieval/bm25_index.py` — rank-bm25 (Okapi) with pickle persistence
- `src/retrieval/rrf_fusion.py` — Reciprocal Rank Fusion: score = Σ 1/(k + rank)
- `src/retrieval/reranker.py` — cross-encoder/ms-marco-MiniLM-L-6-v2, sigmoid normalized scores
- Config: `config/retrieval_config.yaml` (top_k_vector=5, top_k_bm25=5, rrf_k=60, top_k_rerank=3)
- 50+ unit tests across 4 test files

### Phase 3 — RAGAS-Style Evaluation Framework
- `src/evaluation/metrics.py` — 4 metrics:
  1. **Faithfulness** — LLM judge extracts claims, checks each against context (score = supported/total)
  2. **Answer Relevancy** — LLM judge scores completeness, directness, specificity, correctness
  3. **Context Precision** — LLM judge computes Average Precision (rewards relevant chunks ranked higher)
  4. **Citation Accuracy** — Pattern-based [Source: ...] verification against chunk metadata (no LLM)
- `src/evaluation/evaluator.py` — Orchestrator with live mode (full pipeline) and offline mode (precomputed)
- `src/evaluation/dataset.py` — Golden dataset loader with category/difficulty filtering
- `eval/golden_dataset.json` — 10 items with ground truth, expected sources, categories (factual, clinical, governance, cross-standard, out_of_scope)
- `scripts/run_eval.py` — CLI runner (`--offline`, `--items`, `--category`)
- Thresholds: faithfulness=0.85, answer_relevancy=0.80, context_precision=0.75, citation_accuracy=0.90
- LLMJudge uses Claude Sonnet via Anthropic API (shared judge instance across metrics)
- 36/36 tests passing

### Phase 4 — Streamlit Dashboard
- `dashboard/app.py` — Main app with sidebar navigation (port 8504)
- `dashboard/views/query_page.py` — Interactive Q&A, sample questions, latency/cost metrics, chunk explorer
- `dashboard/views/eval_page.py` — Golden dataset browser, metric averages vs thresholds, category breakdown, per-item drill-down
- `dashboard/views/corpus_page.py` — PDF stats, chunking config, ChromaDB/BM25 status, parse preview
- `dashboard/views/health_page.py` — Component status cards, config tabs, env vars, quick actions
- Launch config in `/Users/Sree/AI Projects/.claude/launch.json` → `healthcare-rag-dashboard`

### API Layer
- `src/api/app.py` — FastAPI with `GET /health` and `POST /query` endpoints
- Pydantic models: QueryRequest, QueryResponse, HealthResponse

---

## CRITICAL BLOCKER: Phase 2.5 — Ingestion Not Persisting

**Problem:** ChromaDB collection `aged_care_standards` has 0 chunks after ingestion runs.

**What we know:**
- Multiple ingest attempts (scripts/ingest.py, scripts/ingest_robust.py) all result in empty collection
- ChromaDB persistence WORKS — verified with `scripts/test_chroma_persist.py` (add 2 items → new client reads them back)
- The `embedder.embed_chunks()` method runs without Python errors but data doesn't persist
- Collection name is `aged_care_standards` (in retrieval_config.yaml)
- Persist dir is `./data/processed/chroma` (relative to project root)

**Debug script ready but not yet run to completion:**
- `scripts/trace_ingest.py` — Step-by-step trace: parse → chunk → reset collection → manually embed 3 chunks → test embed_chunks() → full ingest
- This script needs to run: `cd project_root && ./venv/bin/python scripts/trace_ingest.py`
- It takes several minutes because it loads the SentenceTransformer model

**Likely root cause candidates:**
1. Something in the batch embedding loop silently failing
2. The model.encode() returning unexpected format
3. ChromaDB collection reference going stale between operations
4. Metadata format issue causing silent rejection

---

## What Remains

### Phase 2.5 — Fix Ingestion (PRIORITY)
- Run and analyze `scripts/trace_ingest.py` output
- Fix whatever causes embed_chunks() to not persist
- Verify with `scripts/test_retrieval.py` (hybrid pipeline end-to-end test)

### Phase 5 — API Security
- Add JWT or API key authentication to FastAPI
- Rate limiting middleware
- CORS configuration
- Input sanitization

---

## Test Status
- **86/89 tests passing** (3 pre-existing failures in test_reranker.py — score normalization edge case, not blocking)
- Run tests: `cd project_root && ./venv/bin/python -m pytest tests/ -v`

## Key Config (config/retrieval_config.yaml)
```yaml
embedding:
  model: "all-MiniLM-L6-v2"
  dimension: 384
vector_store:
  persist_directory: "./data/processed/chroma"
  collection_name: "aged_care_standards"
retrieval:
  top_k_vector: 5
  top_k_bm25: 5
  rrf_k: 60
  top_k_rerank: 3
reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
generation:
  model: "claude-sonnet-4-5-20250929"
  temperature: 0.1
evaluation:
  faithfulness: 0.85
  answer_relevancy: 0.80
  context_precision: 0.75
  citation_accuracy: 0.90
```

## Project Structure
```
healthcare-rag-system/
├── config/           retrieval_config.yaml, prompts.yaml
├── src/
│   ├── ingestion/    pdf_parser.py, chunker.py, embedder.py
│   ├── retrieval/    retriever.py, bm25_index.py, rrf_fusion.py, reranker.py
│   ├── generation/   generator.py
│   ├── evaluation/   metrics.py, evaluator.py, dataset.py
│   └── api/          app.py (FastAPI)
├── dashboard/
│   ├── app.py
│   └── views/        query_page.py, eval_page.py, corpus_page.py, health_page.py
├── scripts/          ingest.py, run_eval.py, trace_ingest.py, download_corpus.py
├── tests/            test_bm25_index, test_rrf_fusion, test_reranker, test_hybrid_retriever, test_evaluation
├── eval/             golden_dataset.json, results/
├── data/
│   ├── raw/          5 PDFs (aged care standards)
│   └── processed/    chroma/ (empty), bm25_index.pkl (missing)
└── venv/             Python 3.11
```

## Instructions for New Session

1. **Read this primer** to understand the full context
2. **Priority task:** Fix Phase 2.5 ingestion blocker — run `scripts/trace_ingest.py` and diagnose
3. After ingestion works: run `scripts/run_eval.py` for first real evaluation
4. Then implement Phase 5 (API security)
5. Use `./venv/bin/python` for all Python commands
6. Dashboard: port 8504 via launch config `healthcare-rag-dashboard`
