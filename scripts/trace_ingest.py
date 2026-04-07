"""Trace ingestion pipeline step by step with a single PDF."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def p(msg):
    print(msg, flush=True)

p("=" * 60)
p("TRACE: Step-by-step ingestion debug")
p("=" * 60)

# Step 1: Parse ONE small PDF
p("\n[1/6] Parsing PDFs...")
from src.ingestion.pdf_parser import parse_all_pdfs
raw_dir = Path(__file__).parent.parent / "data" / "raw"
pdfs = sorted(raw_dir.glob("*.pdf"), key=lambda f: f.stat().st_size)
p(f"  Smallest PDF: {pdfs[0].name} ({pdfs[0].stat().st_size/1024:.0f} KB)")

documents = parse_all_pdfs(raw_dir)
p(f"  Parsed {len(documents)} documents")
for d in documents:
    p(f"    {d.filename}: {len(d.pages)} pages, title='{d.title[:50]}'")

# Step 2: Chunk
p("\n[2/6] Chunking...")
from src.ingestion.chunker import chunk_all_documents
chunks = chunk_all_documents(documents)
p(f"  Total chunks: {len(chunks)}")
if chunks:
    p(f"  First chunk ID: {chunks[0].chunk_id}")
    p(f"  First chunk text ({len(chunks[0].text)} chars): {chunks[0].text[:80]}...")
    p(f"  First chunk metadata: {chunks[0].to_metadata()}")

# Step 3: Reset ChromaDB collection
p("\n[3/6] Resetting ChromaDB collection...")
from src.ingestion.embedder import Embedder
embedder = Embedder()
embedder.reset()
p("  Collection reset")

coll = embedder.get_or_create_collection()
p(f"  Fresh collection count: {coll.count()}")

# Step 4: Embed just 3 chunks manually
p("\n[4/6] Manually embedding 3 chunks...")
small = chunks[:3]
texts = [c.text for c in small]
ids = [c.chunk_id for c in small]
metas = [c.to_metadata() for c in small]

p(f"  IDs: {ids}")
p(f"  Text lengths: {[len(t) for t in texts]}")

t0 = time.time()
embs = embedder.model.encode(texts, show_progress_bar=False).tolist()
p(f"  Encoding took {time.time()-t0:.1f}s")
p(f"  Embedding dims: {len(embs[0])}")

p("  Calling coll.add()...")
try:
    coll.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
    p(f"  ✓ add() succeeded")
except Exception as e:
    p(f"  ✗ add() FAILED: {e}")
    import traceback; traceback.print_exc()

p(f"  Count after manual add: {coll.count()}")

# Step 5: Now test embed_chunks() method
p("\n[5/6] Testing embed_chunks() method with next 3 chunks...")
next_batch = chunks[3:6]
embedder.embed_chunks(next_batch, batch_size=3)
p(f"  Count after embed_chunks(): {coll.count()}")

# Step 6: Full ingest with reset
p("\n[6/6] FULL INGEST (all chunks, reset first)...")
embedder.reset()
coll = embedder.get_or_create_collection()
p(f"  After reset: {coll.count()}")
embedder.embed_chunks(chunks)
final_count = coll.count()
p(f"\n  FINAL COUNT: {final_count}")
if final_count > 0:
    p(f"  ✓ SUCCESS: {final_count} chunks in ChromaDB")
else:
    p("  ✗ FAIL: ChromaDB is empty after full ingest")

p("\nDone.")
