"""Evaluation runner — orchestrates metric computation across the golden dataset.

Supports two modes:
  1. Live mode  — queries the full RAG pipeline (retriever + generator)
  2. Offline mode — uses pre-computed RAGResponse objects (for testing / CI)
"""
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from .dataset import GoldenDataset, GoldenItem
from .metrics import (
    BaseMetric,
    MetricResult,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    CitationAccuracyMetric,
    LLMJudge,
)


@dataclass
class EvalResult:
    """Result for a single evaluation item."""

    item_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, dict] = field(default_factory=dict)
    latency_ms: float = 0.0
    category: str = ""
    difficulty: str = ""

    @property
    def passed(self) -> bool:
        """Check if all scores meet their thresholds."""
        return all(s >= 0.5 for s in self.scores.values())


@dataclass
class EvalSummary:
    """Aggregate evaluation results."""

    run_id: str
    timestamp: str
    total_items: int
    results: list[EvalResult]
    metric_averages: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    category_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_items": self.total_items,
            "metric_averages": self.metric_averages,
            "thresholds": self.thresholds,
            "pass_rate": self.pass_rate,
            "category_breakdown": self.category_breakdown,
            "duration_seconds": self.duration_seconds,
            "results": [
                {
                    "item_id": r.item_id,
                    "question": r.question,
                    "scores": r.scores,
                    "details": r.details,
                    "latency_ms": r.latency_ms,
                    "category": r.category,
                    "difficulty": r.difficulty,
                }
                for r in self.results
            ],
        }


class Evaluator:
    """Orchestrates RAG pipeline evaluation against a golden dataset."""

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = thresholds or {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
            "citation_accuracy": 0.90,
        }
        judge = LLMJudge()
        self.metrics: list[BaseMetric] = [
            FaithfulnessMetric(judge=judge),
            AnswerRelevancyMetric(judge=judge),
            ContextPrecisionMetric(judge=judge),
            CitationAccuracyMetric(),  # no LLM needed
        ]

    def evaluate_single(
        self,
        item: GoldenItem,
        answer: str,
        contexts: list[str],
        chunk_metadata: list[dict] | None = None,
    ) -> EvalResult:
        """Evaluate a single question-answer pair against all metrics."""
        result = EvalResult(
            item_id=item.id,
            question=item.question,
            answer=answer,
            contexts=contexts,
            ground_truth=item.ground_truth,
            category=item.category,
            difficulty=item.difficulty,
        )

        for metric in self.metrics:
            t0 = time.time()
            mr: MetricResult = metric.compute(
                question=item.question,
                answer=answer,
                contexts=contexts,
                ground_truth=item.ground_truth,
                chunk_metadata=chunk_metadata or [],
            )
            elapsed = (time.time() - t0) * 1000
            result.scores[metric.name] = mr.score
            result.details[metric.name] = {**mr.details, "latency_ms": round(elapsed)}

        return result

    def evaluate_dataset(
        self,
        dataset: GoldenDataset,
        answer_fn=None,
        precomputed: list[dict] | None = None,
    ) -> EvalSummary:
        """Run evaluation across an entire golden dataset.

        Args:
            dataset: Golden dataset to evaluate against.
            answer_fn: Callable(question) -> dict with keys:
                       'answer', 'contexts', 'chunk_metadata' (optional).
                       Used in live mode.
            precomputed: List of dicts with 'id', 'answer', 'contexts',
                         'chunk_metadata'. Used in offline mode.
        """
        start = time.time()
        run_id = datetime.now(timezone.utc).strftime("eval-%Y%m%d-%H%M%S")
        results: list[EvalResult] = []

        # Build lookup for precomputed results
        pre_map = {}
        if precomputed:
            pre_map = {p["id"]: p for p in precomputed}

        for item in dataset:
            print(f"  [{item.id}] {item.question[:60]}...", flush=True)

            if item.id in pre_map:
                data = pre_map[item.id]
                answer = data["answer"]
                contexts = data["contexts"]
                chunk_meta = data.get("chunk_metadata", [])
            elif answer_fn is not None:
                t0 = time.time()
                resp = answer_fn(item.question)
                latency = (time.time() - t0) * 1000
                answer = resp["answer"]
                contexts = resp["contexts"]
                chunk_meta = resp.get("chunk_metadata", [])
            else:
                print(f"    SKIP (no answer source for {item.id})")
                continue

            er = self.evaluate_single(item, answer, contexts, chunk_meta)
            if answer_fn and item.id not in pre_map:
                er.latency_ms = latency
            results.append(er)

            # Print per-item scores
            scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in er.scores.items())
            print(f"    {scores_str}")

        duration = time.time() - start
        summary = self._build_summary(run_id, results, duration)
        return summary

    def _build_summary(
        self, run_id: str, results: list[EvalResult], duration: float
    ) -> EvalSummary:
        if not results:
            return EvalSummary(
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_items=0,
                results=[],
                duration_seconds=duration,
            )

        # Compute averages per metric
        metric_names = list(self.thresholds.keys())
        averages = {}
        for m in metric_names:
            scores = [r.scores.get(m, 0.0) for r in results]
            averages[m] = sum(scores) / len(scores) if scores else 0.0

        # Pass rate: items where ALL metrics meet threshold
        passed = sum(
            1 for r in results
            if all(
                r.scores.get(m, 0.0) >= self.thresholds.get(m, 0.5)
                for m in metric_names
            )
        )
        pass_rate = passed / len(results)

        # Category breakdown
        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        cat_breakdown = {}
        for cat, cat_results in categories.items():
            cat_avgs = {}
            for m in metric_names:
                scores = [r.scores.get(m, 0.0) for r in cat_results]
                cat_avgs[m] = sum(scores) / len(scores) if scores else 0.0
            cat_breakdown[cat] = cat_avgs

        return EvalSummary(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_items=len(results),
            results=results,
            metric_averages=averages,
            thresholds=self.thresholds,
            pass_rate=pass_rate,
            category_breakdown=cat_breakdown,
            duration_seconds=round(duration, 2),
        )

    def save_results(self, summary: EvalSummary, output_dir: str | Path | None = None):
        """Save evaluation results to JSON."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "eval" / "results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{summary.run_id}.json"
        with open(out_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\nResults saved to {out_path}")
        return out_path

    def print_report(self, summary: EvalSummary):
        """Print a formatted evaluation report to stdout."""
        print("\n" + "=" * 70)
        print(f"EVALUATION REPORT — {summary.run_id}")
        print("=" * 70)
        print(f"Timestamp:  {summary.timestamp}")
        print(f"Items:      {summary.total_items}")
        print(f"Duration:   {summary.duration_seconds:.1f}s")
        print(f"Pass Rate:  {summary.pass_rate:.0%}")

        print("\n--- Metric Averages vs Thresholds ---")
        for metric, avg in summary.metric_averages.items():
            threshold = summary.thresholds.get(metric, 0.0)
            status = "PASS" if avg >= threshold else "FAIL"
            print(f"  {metric:25s}  {avg:.3f}  (threshold: {threshold:.2f})  [{status}]")

        if summary.category_breakdown:
            print("\n--- Category Breakdown ---")
            for cat, avgs in summary.category_breakdown.items():
                scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in avgs.items())
                print(f"  {cat:20s}  {scores_str}")

        print("\n--- Per-Item Results ---")
        for r in summary.results:
            scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in r.scores.items())
            print(f"  [{r.item_id}] {scores_str}")

        print("=" * 70)
