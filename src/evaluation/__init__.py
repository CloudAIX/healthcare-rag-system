"""Evaluation framework — RAGAS-style metrics for RAG pipeline quality."""
from .metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    CitationAccuracyMetric,
)
from .evaluator import Evaluator, EvalResult, EvalSummary
from .dataset import GoldenDataset, GoldenItem

__all__ = [
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "CitationAccuracyMetric",
    "Evaluator",
    "EvalResult",
    "EvalSummary",
    "GoldenDataset",
    "GoldenItem",
]
