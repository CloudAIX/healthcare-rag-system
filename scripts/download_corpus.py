"""Download the Strengthened Aged Care Quality Standards corpus."""
import httpx
from pathlib import Path
from tqdm import tqdm

CORPUS = [
    ("strengthened-aged-care-quality-standards-august-2025.pdf",
     "https://www.health.gov.au/sites/default/files/2025-08/strengthened-aged-care-quality-standards-august-2025.pdf",
     "Strengthened Quality Standards Aug 2025 (49 pages)"),
    ("guidance-material-intro.pdf",
     "https://www.agedcarequality.gov.au/sites/default/files/media/guidance-material-for-the-strengthened-aged-care-quality-standards-intro.pdf",
     "Guidance Material — Introduction"),
    ("guidance-material-standard-1.pdf",
     "https://www.agedcarequality.gov.au/sites/default/files/media/guidance-material-for-the-strengthened-aged-care-quality-standards-standard-1.pdf",
     "Guidance Material — Standard 1"),
    ("quick-reference-guide.pdf",
     "https://www.agedcarequality.gov.au/sites/default/files/media/strengthened-quality-standards-quick-reference-guide.pdf",
     "Quick Reference Guide"),
    ("provider-checklist.pdf",
     "https://www.agedcarequality.gov.au/sites/default/files/media/strengthened_standards_provider_checklist_10_feb_2025.pdf",
     "Provider Checklist"),
]
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

def download_corpus():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(CORPUS)} documents to {RAW_DIR}\n")
    for filename, url, desc in tqdm(CORPUS, desc="Downloading"):
        filepath = RAW_DIR / filename
        if filepath.exists():
            print(f"  Already exists: {filename}")
            continue
        print(f"  {desc}")
        try:
            r = httpx.get(url, follow_redirects=True, timeout=60.0)
            r.raise_for_status()
            filepath.write_bytes(r.content)
            print(f"    Downloaded ({len(r.content)/1024/1024:.1f} MB)")
        except httpx.HTTPError as e:
            print(f"    Failed: {e}")
    print(f"\nCorpus download complete.")

if __name__ == "__main__":
    download_corpus()
