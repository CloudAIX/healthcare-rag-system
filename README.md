# Healthcare RAG System

Production-grade Retrieval Augmented Generation for Australian aged care compliance documentation. Ask natural language questions about the Strengthened Aged Care Quality Standards and get grounded, cited answers.

**Built by [GVRN-AI](https://gvrn-ai.com)** — AI governance and automation for healthcare.

## Architecture
```
Query -> Hybrid Retrieval (BM25 + Vector) -> RRF Fusion -> Cross-Encoder Re-rank -> Top-3 -> Claude -> Cited Answer -> Validation
```

## Quick Start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your ANTHROPIC_API_KEY
python scripts/download_corpus.py
python scripts/ingest.py
uvicorn src.api.app:app --reload
```

## Tech Stack
| Component | Tool |
|-----------|------|
| Vector Store | ChromaDB |
| BM25 Search | rank_bm25 |
| Embeddings | all-MiniLM-L6-v2 |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Claude Sonnet (Anthropic API) |
| Evaluation | RAGAS |
| Tracing | LangFuse |
| API | FastAPI |

## Corpus
Strengthened Aged Care Quality Standards (effective 1 Nov 2025, Aged Care Act 2024) — 7 standards, guidance materials, provider checklist.

## License
MIT
