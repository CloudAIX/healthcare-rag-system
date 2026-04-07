"""Full ingestion pipeline: Parse PDFs -> Chunk -> Embed -> Store (Vector + BM25)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all_documents
from src.ingestion.embedder import Embedder
from src.retrieval.bm25_index import BM25Index

def main():
    parser = argparse.ArgumentParser(description="Ingest aged care documents")
    parser.add_argument("--reset", action="store_true", help="Delete existing embeddings and indexes")
    args = parser.parse_args()
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    if not raw_dir.exists() or not list(raw_dir.glob("*.pdf")):
        print("No PDFs found. Run 'python scripts/download_corpus.py' first.")
        sys.exit(1)
    print("="*60 + "\nHealthcare RAG System Ingestion (Vector + BM25)\n" + "="*60)
    documents = parse_all_pdfs(raw_dir)
    chunks = chunk_all_documents(documents)
    embedder = Embedder()
    if args.reset:
        embedder.reset()
        # Also reset BM25 index
        bm25_path = Path("./data/processed/bm25_index.pkl")
        if bm25_path.exists():
            bm25_path.unlink()
            print("Cleared BM25 index")
    embedder.embed_chunks(chunks)
    coll = embedder.get_or_create_collection()
    print(f"ChromaDB: {coll.count()} chunks stored")
    # Build BM25 index
    bm25_index = BM25Index(persist_path=Path("./data/processed/bm25_index.pkl"))
    bm25_index.build_from_chunks(chunks)
    bm25_index.save()
    print(f"BM25 Index: {len(bm25_index)} chunks indexed")
    print(f"\nDone! Both indexes ready for hybrid retrieval")

if __name__ == "__main__":
    main()
