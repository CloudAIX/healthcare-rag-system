"""Document Chunker — splits documents into overlapping chunks with metadata."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from .pdf_parser import DocumentContent

@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_title: str
    document_filename: str
    page_numbers: list[int]
    sections: list[str]
    chunk_index: int
    char_start: int
    char_end: int
    def to_metadata(self):
        return {"chunk_id": self.chunk_id, "document_title": self.document_title,
                "document_filename": self.document_filename,
                "page_numbers": ",".join(str(p) for p in self.page_numbers),
                "sections": ",".join(self.sections) if self.sections else "",
                "chunk_index": self.chunk_index}
    @property
    def citation(self):
        pages = f"p.{self.page_numbers[0]}" if len(self.page_numbers)==1 else f"pp.{self.page_numbers[0]}-{self.page_numbers[-1]}"
        sec = f", {self.sections[0]}" if self.sections else ""
        return f"[Source: {self.document_title}{sec}, {pages}]"

def generate_chunk_id(filename, idx, text):
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    base = filename.replace(".pdf","").replace(" ","-").lower()
    return f"{base}-chunk-{idx:04d}-{h}"

def load_chunking_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f: return yaml.safe_load(f)["chunking"]

def chunk_document(document, chunk_size=700, chunk_overlap=100):
    page_boundaries = []
    combined = ""
    for page in document.pages:
        start = len(combined)
        combined += page.text + "\n\n"
        page_boundaries.append((start, page.page_number, page.detected_sections))
    csz, covr = chunk_size*4, chunk_overlap*4
    chunks, start, idx = [], 0, 0
    while start < len(combined):
        end = start + csz
        if end < len(combined):
            pb = combined.rfind("\n\n", start+csz//2, end+covr)
            if pb != -1: end = pb
            else:
                sb = combined.rfind(". ", start+csz//2, end+covr)
                if sb != -1: end = sb+1
        end = min(end, len(combined))
        ct = combined[start:end].strip()
        if not ct: break
        cp, cs = [], []
        for off, pn, secs in page_boundaries:
            pi = pn-1
            if pi < len(document.pages):
                pe = off + len(document.pages[pi].text)
                if off < end and pe > start:
                    cp.append(pn); cs.extend(secs)
        if not cp: cp = [1]
        cs = list(dict.fromkeys(cs))
        chunks.append(Chunk(chunk_id=generate_chunk_id(document.filename,idx,ct),
            text=ct, document_title=document.title, document_filename=document.filename,
            page_numbers=cp, sections=cs, chunk_index=idx, char_start=start, char_end=end))
        start = end - covr; idx += 1
    return chunks

def chunk_all_documents(documents):
    cfg = load_chunking_config()
    sz, ov = cfg.get("chunk_size",700), cfg.get("chunk_overlap",100)
    print(f"\nChunking {len(documents)} docs (size={sz}, overlap={ov})\n")
    all_chunks = []
    for doc in documents:
        c = chunk_document(doc, chunk_size=sz, chunk_overlap=ov)
        print(f"  {doc.filename}: {len(c)} chunks")
        all_chunks.extend(c)
    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks
