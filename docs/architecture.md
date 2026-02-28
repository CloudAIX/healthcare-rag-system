# Architecture
```
Query -> Hybrid Retrieval (BM25 + Vector) -> RRF Fusion -> Cross-Encoder Re-rank -> Top-3 -> Claude -> Cited Answer -> Validation
```
## Data Flow
1. **Ingestion** (offline): PDFs -> Parse -> Chunk (700 tokens, 100 overlap) -> Embed -> ChromaDB
2. **Query** (online): Question -> Hybrid Retrieval -> RRF -> Re-rank -> Top-3 -> Claude -> Cited Answer
