# Technical Decision Log
## 001: Embedding — all-MiniLM-L6-v2 (local) over OpenAI
Free, local, no API dependency. 384 dimensions sufficient for corpus size.
## 002: ChromaDB over Weaviate for Phase 1
Simplest setup, no Docker. Corpus ~500 chunks — well within limits.
## 003: Claude Sonnet for generation
Excellent instruction following for citation tasks. Temp 0.1 for factual compliance.
## 004: Prompts in YAML config, not hardcoded
Prompts are system architecture. Versioned, diffable, reviewable in PRs.
