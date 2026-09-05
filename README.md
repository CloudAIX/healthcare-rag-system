# Healthcare RAG System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude_Sonnet-Anthropic-D4A574?logo=anthropic&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-103%2F103_passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

Production-grade Retrieval-Augmented Generation pipeline for Australian Aged Care Quality Standards compliance. Ask natural language questions about the Strengthened Aged Care Quality Standards and get grounded, cited answers.

**Built by [GVRN-AI](https://gvrn-ai.com)** — AI governance and automation for healthcare.

## Dashboard

| Query Interface | Evaluation Results |
|---|---|
| ![Query](docs/screenshots/query.png) | ![Evaluation](docs/screenshots/eval.png) |

| Corpus Explorer | System Health |
|---|---|
| ![Corpus](docs/screenshots/corpus.png) | ![Health](docs/screenshots/health.png) |

## Architecture

```
                    ┌─────────────┐
                    │  User Query │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐       ┌─────────────────┐
     │ Vector Search  │       │  BM25 Search    │
     │ (ChromaDB)     │       │  (rank-bm25)    │
     │ top_k = 5      │       │  top_k = 5      │
     └────────┬───────┘       └────────┬────────┘
              │                        │
              └───────────┬────────────┘
                          ▼
                 ┌─────────────────┐
                 │  RRF Fusion     │
                 │  (k = 60)       │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Cross-Encoder   │
                 │ Re-ranking      │
                 │ top_k = 3       │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Claude Sonnet   │
                 │ Generation      │
                 │ + Citations     │
                 └─────────────────┘
```

## Evaluation Results

Evaluated against a 10-item golden dataset across 5 categories (clinical, factual, governance, cross-standard, out-of-scope):

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.994 | 0.85 | PASS |
| Answer Relevancy | 0.901 | 0.80 | PASS |
| Context Precision | 0.883 | 0.75 | PASS |
| Citation Accuracy | 1.000 | 0.90 | PASS |

Evaluation uses a RAGAS-style LLM-as-judge framework with Claude Sonnet.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/CloudAIX/healthcare-rag-system.git
cd healthcare-rag-system
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env   # Add your ANTHROPIC_API_KEY

# Download corpus and ingest
python scripts/download_corpus.py   # 5 PDFs from aged care quality authority
python scripts/ingest.py            # Parse, chunk, embed → ChromaDB + BM25

# Run the API
uvicorn src.api.app:app --reload

# Or run the dashboard
streamlit run dashboard/app.py --server.port 8504
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Vector Store | ChromaDB (PersistentClient) |
| BM25 Search | rank-bm25 (Okapi BM25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Embeddings | all-MiniLM-L6-v2 (384 dims) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Claude Sonnet (Anthropic API) |
| Evaluation | Custom RAGAS-style framework (LLM-as-judge) |
| API | FastAPI + JWT + API key auth |
| Dashboard | Streamlit (4 pages) |
| Tracing | LangFuse (optional) |

## API Endpoints

All endpoints require authentication except `/health`.

| Route | Method | Auth | Rate Limit | Description |
|-------|--------|------|------------|-------------|
| `/health` | GET | None | — | System status + collection size |
| `/auth/token` | POST | API Key | 5/min | Exchange API key for JWT |
| `/query` | POST | API Key or JWT | 30/min | Query the RAG pipeline |

**Authentication:** Pass `X-API-Key` header or `Authorization: Bearer <jwt>` token.

```bash
# Health check
curl http://localhost:8000/health

# Get a JWT token (API key goes in the header AND the body — deliberate double-check)
curl -X POST http://localhost:8000/auth/token \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key", "client_id": "my-client"}'

# Query the pipeline
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the documentation requirements for Standard 1?"}'
```

## Dashboard

Four-page Streamlit app for interactive exploration:

| Page | Description |
|------|-------------|
| **Query** | Interactive Q&A with sample questions, latency/cost metrics, chunk explorer |
| **Evaluation** | Golden dataset browser, metric averages vs thresholds, category breakdown |
| **Corpus** | PDF stats, chunking config, ChromaDB/BM25 status, parse preview |
| **System Health** | Component status cards, config tabs, environment check |

## Corpus

**Strengthened Aged Care Quality Standards** (effective 1 Nov 2025, Aged Care Act 2024):

- Aged Care Quality Standards document (49 pages)
- Guidance materials (intro + Standard 1)
- Quick reference guide
- Provider checklist

5 PDFs → 95 chunks after recursive chunking (2800 chars, 400 char overlap).

## Evaluation Framework

Custom RAGAS-style evaluation with four metrics:

- **Faithfulness** — LLM judge extracts claims from the answer, verifies each against retrieved context
- **Answer Relevancy** — LLM judge scores completeness, directness, specificity, correctness
- **Context Precision** — Average Precision computed by LLM judge (rewards relevant chunks ranked higher)
- **Citation Accuracy** — Pattern-based `[Source: ...]` verification against chunk metadata (no LLM needed)

```bash
# Run evaluation (requires ANTHROPIC_API_KEY)
python scripts/run_eval.py

# Offline mode (with precomputed answers)
python scripts/run_eval.py --offline

# Filter by category
python scripts/run_eval.py --category clinical
```

## Project Structure

```
healthcare-rag-system/
├── config/
│   ├── retrieval_config.yaml    # Pipeline configuration
│   └── prompts.yaml             # Generation prompts
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PyMuPDF extraction + section detection
│   │   ├── chunker.py           # Recursive text chunking
│   │   └── embedder.py          # SentenceTransformer + ChromaDB
│   ├── retrieval/
│   │   ├── retriever.py         # Hybrid retrieval orchestrator
│   │   ├── bm25_index.py        # BM25 index with persistence
│   │   ├── rrf_fusion.py        # Reciprocal Rank Fusion
│   │   └── reranker.py          # Cross-encoder re-ranking
│   ├── generation/
│   │   └── generator.py         # Claude Sonnet with citations
│   ├── evaluation/
│   │   ├── metrics.py           # 4 RAGAS-style metrics + LLM judge
│   │   ├── evaluator.py         # Evaluation orchestrator
│   │   └── dataset.py           # Golden dataset loader
│   └── api/
│       ├── app.py               # FastAPI application (v2.0.0)
│       └── security.py          # JWT + API key + rate limiting
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   └── views/                   # query, eval, corpus, health pages
├── scripts/
│   ├── download_corpus.py       # Download PDFs from source
│   ├── ingest.py                # Parse + chunk + embed pipeline
│   └── run_eval.py              # Evaluation runner
├── tests/                       # 103/103 passing
├── eval/
│   └── golden_dataset.json      # 10 items, 5 categories
├── .env.example                 # All configurable env vars
└── requirements.txt
```

## Deployment

```bash
# Container build (API only; dashboard runs separately)
docker build -t healthcare-rag .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data healthcare-rag
```

The `data/` volume carries the ingested ChromaDB collection and BM25 index — run `python scripts/ingest.py` once before first query. Models (~90MB) download on first start; allow ~60s before `Application startup complete`.

## Testing

```bash
python -m pytest tests/ -v
```

103/103 tests across 6 test suites: BM25 index, RRF fusion, reranker, hybrid retriever, evaluation metrics, API security.

## Configuration

All settings in `config/retrieval_config.yaml` and overridable via environment variables. See `.env.example` for the full list.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key |
| `ANTHROPIC_MODEL` | claude-sonnet-4-5-20250929 | Generation model |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Embedding model |
| `TOP_K_VECTOR` | 5 | Vector search results |
| `TOP_K_BM25` | 5 | BM25 search results |
| `TOP_K_RERANK` | 3 | Final re-ranked results |
| `RAG_API_KEY` | — | API authentication key |

## License

MIT
