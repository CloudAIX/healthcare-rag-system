"""Golden dataset loader for evaluation."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenItem:
    id: str
    question: str
    ground_truth: str
    expected_sources: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"


class GoldenDataset:
    """Loads and manages the golden evaluation dataset."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path(__file__).parent.parent.parent / "eval" / "golden_dataset.json"
        self.path = Path(path)
        self.items: list[GoldenItem] = []
        self._load()

    def _load(self):
        with open(self.path) as f:
            raw = json.load(f)
        self.items = [
            GoldenItem(
                id=r["id"],
                question=r["question"],
                ground_truth=r.get("ground_truth", ""),
                expected_sources=r.get("expected_sources", []),
                category=r.get("category", "general"),
                difficulty=r.get("difficulty", "medium"),
            )
            for r in raw
        ]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def filter_by_category(self, category: str) -> list[GoldenItem]:
        return [i for i in self.items if i.category == category]

    def filter_by_difficulty(self, difficulty: str) -> list[GoldenItem]:
        return [i for i in self.items if i.difficulty == difficulty]

    def exclude_out_of_scope(self) -> list[GoldenItem]:
        return [i for i in self.items if i.category != "out_of_scope"]
