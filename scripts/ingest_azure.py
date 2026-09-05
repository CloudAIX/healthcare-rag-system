"""Azure twin ingestion: Parse PDFs -> Chunk -> embed locally -> Azure AI Search.

Same parser, chunker and embedding model as the ChromaDB path, so both
backends index identical chunks with identical vectors.

Requires AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY (admin key).
Provision the service first: see infra/azure/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all_documents
from src.retrieval.azure_search_store import AzureSearchStore


def main():
    parser = argparse.ArgumentParser(description="Ingest aged care documents into Azure AI Search")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index first")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not raw_dir.exists() or not list(raw_dir.glob("*.pdf")):
        print("No PDFs found. Run 'python scripts/download_corpus.py' first.")
        sys.exit(1)

    print("=" * 60 + "\nHealthcare RAG Ingestion — Azure AI Search backend\n" + "=" * 60)
    documents = parse_all_pdfs(raw_dir)
    chunks = chunk_all_documents(documents)

    store = AzureSearchStore()
    if args.reset:
        store.reset()
        print("Deleted existing index")
    store.ensure_index()
    store.embed_chunks(chunks)
    print(f"Index document count: {store.get_or_create_collection().count()}")
    print("\nDone. Query with RAG_BACKEND=azure (API, dashboard and eval runner all honour it).")


if __name__ == "__main__":
    main()
