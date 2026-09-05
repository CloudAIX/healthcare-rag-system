# Healthcare RAG System — API container
# Build:  docker build -t healthcare-rag .
# Run:    docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data healthcare-rag
#
# The data volume carries the ingested ChromaDB + BM25 index. Run ingestion
# once on the host (python scripts/ingest.py) or inside the container before
# first query. Embedding + reranker models download on first start (~90MB);
# cache them across restarts by also mounting -v hf-cache:/root/.cache.

FROM python:3.11-slim

WORKDIR /app

# Torch CPU wheels are large; keep layer cache effective by installing deps first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY eval/ eval/

EXPOSE 8000

# Model load takes ~60s before "Application startup complete"
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
