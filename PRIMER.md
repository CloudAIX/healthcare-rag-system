# Healthcare RAG System — Continuation Primer

## Context

You are continuing work on a **Healthcare RAG System** — a production-grade Retrieval-Augmented Generation pipeline for Australian Aged Care Quality Standards compliance. This is a portfolio project at:

```
/Users/Sree/AI Projects/portfolio/healthcare-rag-system/
```

**Python environment:** `./venv/bin/python` (Python 3.11, NOT `python` or `python3`)

---

## What's Built (All Phases Complete)

### Phase 1 — Ingestion Pipeline
- `src/ingestion/pdf_parser.py` — PyMuPDF extraction with section detection (Standard X, Outcome X.X, Action X.X.X)
- `src/ingestion/chunker.py` — Recursive chunking (700×4 chars, 100×4 overlap), MD5-based chunk IDs
- `src/ingestion/embedder.py` — SentenceTransformer (all-MiniLM-L6-v2, 384 dims) + ChromaDB PersistentClient
- 5 PDFs in `data/raw/` (4.4MB): aged care standards, guidance materials, provider checklist
- Ingest: `./venv/bin/python scripts/ingest.py` (builds both ChromaDB + BM25 index)

### Phase 2 — Hybrid Retrieval
- `src/retrieval/retriever.py` — Orchestrator: vector → BM25 → RRF → re-rank, graceful fallback to vector-only
- `src/retrieval/bm25_index.py` — rank-bm25 (Okapi) with pickle persistence
- `src/retrieval/rrf_fusion.py` — Reciprocal Rank Fusion: score = Σ 1/(k + rank)
- `src/retrieval/reranker.py` — cross-encoder/ms-marco-MiniLM-L-6-v2, sigmoid normalized scores (float64, NaN-safe)
- Config: `config/retrieval_config.yaml` (top_k_vector=5, top_k_bm25=5, rrf_k=60, top_k_rerank=3)

### Phase 3 — RAGAS-Style Evaluation Framework
- `src/evaluation/metrics.py` — 4 metrics:
  1. **Faithfulness** — LLM judge extracts claims, checks each against context (score = supported/total)
  2. **Answer Relevancy** — LLM judge scores completeness, directness, specificity, correctness
  3. **Context Precision** — LLM judge computes Average Precision (rewards relevant chunks ranked higher)
  4. **Citation Accuracy** — Pattern-based [Source: ...] verification against chunk metadata (no LLM)
- `src/evaluation/evaluator.py` — Orchestrator with live mode (full pipeline) and offline mode (precomputed)
- `src/evaluation/dataset.py` — Golden dataset loader with category/difficulty filtering
- `eval/golden_dataset.json` — 10 items with ground truth, expected sources, categories
- `scripts/run_eval.py` — CLI runner (`--offline`, `--items`, `--category`)
- LLMJudge JSON parsing: greedy regex + outermost extraction + trailing comma removal + LLM self-repair fallback
- Thresholds: faithfulness=0.85, answer_relevancy=0.80, context_precision=0.75, citation_accuracy=0.90

### Phase 4 — Streamlit Dashboard
- `dashboard/app.py` — Main app with sidebar navigation (port 8504)
- `dashboard/views/query_page.py` — Interactive Q&A, sample questions, latency/cost metrics, chunk explorer
- `dashboard/views/eval_page.py` — Golden dataset browser, metric averages vs thresholds, category breakdown
- `dashboard/views/corpus_page.py` — PDF stats, chunking config, ChromaDB/BM25 status, parse preview
- `dashboard/views/health_page.py` — Component status cards, config tabs, env vars, quick actions
- Launch config in `/Users/Sree/AI Projects/.claude/launch.json` → `healthcare-rag-dashboard`

### Phase 5 — API Security
- `src/api/security.py` — API key auth (X-API-Key, constant-time comparison) + JWT tokens (HS256, 60-min expiry)
- `src/api/app.py` — v2.0.0: CORS middleware, rate limiting (slowapi), dual auth on `/query`
- `POST /auth/token` — Exchanges API key for JWT (5/min rate limit)
- `GET /health` — Public, no auth
- `POST /query` — Requires API key or Bearer JWT (30/min rate limit)
- `tests/test_api_security.py` — 14 tests covering auth, JWT, validation, CORS
- Config via env: RAG_API_KEY, RAG_SECRET_KEY, RAG_JWT_EXPIRE_MINUTES, RAG_CORS_ORIGINS

---

## Evaluation Results (Live Run — Apr 7, 2026)

Against 95 real chunks from 5 aged care standards PDFs:

| Metric             | Score | Threshold | Status |
|--------------------|-------|-----------|--------|
| Faithfulness       | 0.994 | 0.85      | PASS   |
| Answer Relevancy   | 0.901 | 0.80      | PASS   |
| Context Precision  | 0.883 | 0.75      | PASS   |
| Citation Accuracy  | 1.000 | 0.90      | PASS   |

Results saved in `eval/results/`.

---

## Resolved Issues

### Phase 2.5 — Ingestion Blocker (Fixed)
- **Root cause:** Infinite loop in `chunker.py` when remaining text <= overlap size — `end - covr == start` caused no progress
- **Fix:** Force `start = end` when `new_start <= start` (line 72)

### Reranker Score Normalization (Fixed)
- **Root cause:** `_sigmoid_normalize()` could produce NaN/overflow with certain cross-encoder outputs
- **Fix:** float64 conversion, `nan_to_num()`, double `np.clip()` (lines 93-100)

### LLM Judge JSON Parsing (Fixed)
- **Root cause:** Non-greedy regex missed nested arrays; trailing commas; unescaped quotes in evidence strings
- **Fix:** Greedy regex + outermost `{...}` extraction + trailing comma removal + LLM self-repair fallback

---

## Test Status
- **99/103 tests passing** (1 environmental: BM25 index exists on disk after ingestion)
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
│   └── api/          app.py, security.py
├── dashboard/
│   ├── app.py
│   └── views/        query_page.py, eval_page.py, corpus_page.py, health_page.py
├── scripts/          ingest.py, run_eval.py, download_corpus.py
├── tests/            test_bm25_index, test_rrf_fusion, test_reranker, test_hybrid_retriever, test_evaluation, test_api_security
├── eval/             golden_dataset.json, results/
├── data/
│   ├── raw/          5 PDFs (aged care standards)
│   └── processed/    chroma/ (95 chunks), bm25_index.pkl
├── .env.example      All configurable env vars documented
└── venv/             Python 3.11
```

## Instructions for New Session

1. **Read this primer** to understand the full context
2. All phases (1-5) are complete — the system is functional end-to-end
3. Use `./venv/bin/python` for all Python commands
4. Dashboard: port 8504 via launch config `healthcare-rag-dashboard`
5. Run eval: `./venv/bin/python scripts/run_eval.py` (requires ANTHROPIC_API_KEY in .env)
6. Run tests: `./venv/bin/python -m pytest tests/ -v`
