"""PDF Parser — extracts text with metadata from aged care PDFs using PyMuPDF."""
import re
from pathlib import Path
from dataclasses import dataclass, field
import fitz

@dataclass
class PageContent:
    page_number: int
    text: str
    document_title: str
    document_filename: str
    detected_sections: list[str] = field(default_factory=list)

@dataclass
class DocumentContent:
    filename: str
    title: str
    total_pages: int
    pages: list[PageContent]
    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

def extract_document_title(doc, filename):
    metadata = doc.metadata
    if metadata.get("title"):
        return metadata["title"].strip()
    if len(doc) > 0:
        blocks = doc[0].get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > 16 and len(span["text"].strip()) > 10:
                            return span["text"].strip()
    return filename.replace("-", " ").replace("_", " ").replace(".pdf", "").title()

def detect_sections(text):
    patterns = [r"Standard\s+\d+", r"Outcome\s+\d+\.\d+", r"Action\s+\d+\.\d+\.\d+"]
    sections = []
    for p in patterns:
        sections.extend(re.findall(p, text, re.IGNORECASE))
    return list(dict.fromkeys(sections))

def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if re.match(r"^Page\s+\d+\s+of\s+\d+$", s, re.IGNORECASE): continue
        if re.match(r"^\d+$", s) and len(s) <= 3: continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def parse_pdf(filepath):
    doc = fitz.open(str(filepath))
    filename = filepath.name
    title = extract_document_title(doc, filename)
    pages = []
    for i in range(len(doc)):
        raw = doc[i].get_text("text")
        cleaned = clean_text(raw)
        if not cleaned.strip(): continue
        pages.append(PageContent(page_number=i+1, text=cleaned,
            document_title=title, document_filename=filename,
            detected_sections=detect_sections(cleaned)))
    doc.close()
    return DocumentContent(filename=filename, title=title, total_pages=len(doc), pages=pages)

def parse_all_pdfs(directory):
    pdf_files = sorted(Path(directory).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return []
    print(f"Parsing {len(pdf_files)} PDFs from {directory}\n")
    documents = []
    for fp in pdf_files:
        print(f"  Parsing: {fp.name}")
        doc = parse_pdf(fp)
        print(f"    {len(doc.pages)} pages, title: {doc.title}")
        documents.append(doc)
    print(f"\nParsed {len(documents)} documents, {sum(len(d.pages) for d in documents)} pages")
    return documents
