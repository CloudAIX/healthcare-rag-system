"""Full ingestion pipeline: Parse PDFs -> Chunk -> Embed -> Store."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all_documents
from src.ingestion.embedder import Embedder

def main():
    parser = argparse.ArgumentParser(description="Ingest aged care documents")
    parser.add_argument("--reset", action="store_true", help="Delete existing embeddings")
    args = parser.parse_args()
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not raw_dir.exists() or not list(raw_dir.glob("*.pdf")):
        print("No PDFs found. Run 'python scripts/download_corpus.py' first.")
        sys.exit(1)
    print("="*60 + "\nHealthcare RAG System Ingestion\n" + "="*60)
    documents = parse_all_pdfs(raw_dir)
    chunks = chunk_all_documents(documents)
    embedder = Embedder()
    if args.reset: embedder.reset()
    embedder.embed_chunks(chunks)
    coll = embedder.get_or_create_collection()
    print(f"\nDone! {coll.count()} chunks in ChromaDB")

if __name__ == "__main__":
    main()
