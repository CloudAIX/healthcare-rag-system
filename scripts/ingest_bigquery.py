"""GCP variant ingestion: same parser/chunker/embeddings into BigQuery.

Requires GOOGLE_CLOUD_PROJECT and either Application Default Credentials
or GOOGLE_OAUTH_ACCESS_TOKEN. Runs in the BigQuery sandbox — no billing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all_documents
from src.retrieval.bigquery_store import BigQueryStore


def main():
    parser = argparse.ArgumentParser(description="Ingest aged care documents into BigQuery")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the table first")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not raw_dir.exists() or not list(raw_dir.glob("*.pdf")):
        print("No PDFs found. Run 'python scripts/download_corpus.py' first.")
        sys.exit(1)

    print("=" * 60 + "\nHealthcare RAG Ingestion — BigQuery backend\n" + "=" * 60)
    documents = parse_all_pdfs(raw_dir)
    chunks = chunk_all_documents(documents)

    store = BigQueryStore()
    if args.reset:
        store.reset()
        print("Dropped existing table")
    store.ensure_table()
    store.embed_chunks(chunks)
    print("\nDone. Query with RAG_BACKEND=bigquery.")


if __name__ == "__main__":
    main()
