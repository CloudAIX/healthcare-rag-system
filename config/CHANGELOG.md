# Configuration Changelog
## [1.0] - 2026-02-28
### Initial Configuration
- System prompt v1.0: Compliance assistant with citation enforcement and refusal behaviour
- Chunk size: 700 tokens with 100 token overlap
- Retrieval: top-5 vector + top-5 BM25, RRF fusion, top-3 after re-ranking
- Generation: Claude Sonnet, temperature 0.1
- Evaluation thresholds: faithfulness 0.85, citation accuracy 0.90
