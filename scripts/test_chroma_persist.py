"""Test if ChromaDB can persist data at all."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import chromadb
from chromadb.config import Settings

persist_dir = "./data/processed/chroma"
print(f"Testing ChromaDB persistence at: {Path(persist_dir).resolve()}")

# Step 1: Add data
client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
coll = client.get_or_create_collection("test_persist")
print(f"Before add: {coll.count()} items")

coll.add(
    ids=["test-1", "test-2"],
    documents=["Hello world", "Aged care standards"],
    embeddings=[[0.1]*384, [0.2]*384],
    metadatas=[{"source": "test"}, {"source": "test"}]
)
print(f"After add (same client): {coll.count()} items")

# Step 2: Create NEW client to verify persistence
client2 = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
coll2 = client2.get_collection("test_persist")
count2 = coll2.count()
print(f"After add (NEW client): {count2} items")

if count2 == 2:
    print("✓ ChromaDB persistence WORKS")
    # Cleanup
    client2.delete_collection("test_persist")
    print("Cleaned up test collection")
else:
    print("✗ ChromaDB persistence BROKEN")
    print(f"  Expected 2 items, got {count2}")

# Also check aged_care_standards
coll3 = client2.get_or_create_collection("aged_care_standards")
print(f"\naged_care_standards: {coll3.count()} items")
