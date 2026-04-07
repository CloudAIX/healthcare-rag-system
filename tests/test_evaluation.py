"""Tests for the evaluation framework — metrics, dataset, evaluator."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evaluation.dataset import GoldenDataset, GoldenItem
from src.evaluation.metrics import (
    MetricResult,
    CitationAccuracyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    LLMJudge,
)
from src.evaluation.evaluator import Evaluator, EvalResult, EvalSummary


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_contexts():
    return [
        "Standard 1 requires that each person is treated with dignity and respect.",
        "Providers must ensure cultural safety for Aboriginal and Torres Strait Islander peoples.",
        "Medication management includes safe prescribing, dispensing and administration.",
    ]


@pytest.fixture
def sample_answer():
    return (
        "Standard 1 requires dignity and respect for all residents. "
        "Providers must ensure culturally safe care. "
        "[Source: Aged Care Quality Standards, Standard 1, p.12]"
    )


@pytest.fixture
def sample_metadata():
    return [
        {
            "document_title": "Aged Care Quality Standards",
            "document_filename": "standards.pdf",
            "sections": "Standard 1",
            "page_numbers": [12],
        },
        {
            "document_title": "Aged Care Quality Standards",
            "document_filename": "standards.pdf",
            "sections": "Standard 1",
            "page_numbers": [15],
        },
    ]


@pytest.fixture
def golden_dataset(tmp_path):
    data = [
        {
            "id": "test-001",
            "question": "What is Standard 1?",
            "ground_truth": "Standard 1 covers dignity and respect.",
            "expected_sources": ["standards.pdf"],
            "category": "factual",
            "difficulty": "easy",
        },
        {
            "id": "test-002",
            "question": "What are medication requirements?",
            "ground_truth": "Safe prescribing and administration of medicines.",
            "expected_sources": ["standards.pdf"],
            "category": "clinical",
            "difficulty": "medium",
        },
    ]
    path = tmp_path / "test_golden.json"
    path.write_text(json.dumps(data))
    return GoldenDataset(path)


@pytest.fixture
def mock_judge():
    judge = MagicMock(spec=LLMJudge)
    return judge


# ===========================================================================
# Dataset tests
# ===========================================================================

class TestGoldenDataset:
    def test_load_real_dataset(self):
        ds = GoldenDataset()
        assert len(ds) == 10
        assert ds[0].id == "eval-001"
        assert ds[-1].category == "out_of_scope"

    def test_load_custom_path(self, golden_dataset):
        assert len(golden_dataset) == 2
        assert golden_dataset[0].id == "test-001"

    def test_filter_by_category(self):
        ds = GoldenDataset()
        clinical = ds.filter_by_category("clinical")
        assert all(i.category == "clinical" for i in clinical)
        assert len(clinical) == 3

    def test_filter_by_difficulty(self):
        ds = GoldenDataset()
        easy = ds.filter_by_difficulty("easy")
        assert all(i.difficulty == "easy" for i in easy)

    def test_exclude_out_of_scope(self):
        ds = GoldenDataset()
        filtered = ds.exclude_out_of_scope()
        assert all(i.category != "out_of_scope" for i in filtered)
        assert len(filtered) == 9

    def test_golden_item_fields(self):
        ds = GoldenDataset()
        item = ds[0]
        assert item.question
        assert item.ground_truth
        assert isinstance(item.expected_sources, list)

    def test_iteration(self, golden_dataset):
        items = list(golden_dataset)
        assert len(items) == 2

    def test_indexing(self, golden_dataset):
        assert golden_dataset[1].id == "test-002"


# ===========================================================================
# Citation Accuracy tests (no LLM needed)
# ===========================================================================

class TestCitationAccuracy:
    def test_valid_citation(self, sample_contexts, sample_metadata):
        metric = CitationAccuracyMetric()
        answer = "Dignity is required. [Source: Aged Care Quality Standards, Standard 1, p.12]"
        result = metric.compute("q", answer, sample_contexts, chunk_metadata=sample_metadata)
        assert result.score == 1.0
        assert result.details["citations_found"] == 1
        assert result.details["citations_verified"] == 1

    def test_invalid_citation(self, sample_contexts, sample_metadata):
        metric = CitationAccuracyMetric()
        answer = "Something. [Source: Nonexistent Document, Chapter 99, p.999]"
        result = metric.compute("q", answer, sample_contexts, chunk_metadata=sample_metadata)
        assert result.score == 0.0

    def test_no_citations_substantive_answer(self, sample_contexts):
        metric = CitationAccuracyMetric()
        answer = "Standard 1 requires providers to ensure dignity and respect for all residents in aged care facilities at all times."
        result = metric.compute("q", answer, sample_contexts)
        assert result.score == 0.0
        assert "no citations" in result.details.get("error", "")

    def test_no_citations_refusal(self, sample_contexts):
        metric = CitationAccuracyMetric()
        answer = "I cannot find sufficient evidence."
        result = metric.compute("q", answer, sample_contexts)
        assert result.score == 1.0  # refusal without citations is fine

    def test_multiple_citations(self, sample_contexts, sample_metadata):
        metric = CitationAccuracyMetric()
        answer = (
            "Dignity. [Source: Aged Care Quality Standards, Standard 1, p.12] "
            "Also culture. [Source: Aged Care Quality Standards, Standard 1, p.15]"
        )
        result = metric.compute("q", answer, sample_contexts, chunk_metadata=sample_metadata)
        assert result.score == 1.0
        assert result.details["citations_found"] == 2

    def test_mixed_citations(self, sample_contexts, sample_metadata):
        metric = CitationAccuracyMetric()
        answer = (
            "Valid. [Source: Aged Care Quality Standards, Standard 1, p.12] "
            "Invalid. [Source: Fake Document, p.999]"
        )
        result = metric.compute("q", answer, sample_contexts, chunk_metadata=sample_metadata)
        assert result.score == 0.5

    def test_empty_answer(self, sample_contexts):
        metric = CitationAccuracyMetric()
        result = metric.compute("q", "", sample_contexts)
        assert result.score == 1.0  # empty = refusal

    def test_metric_name(self):
        assert CitationAccuracyMetric.name == "citation_accuracy"


# ===========================================================================
# LLM-judged metrics (mocked)
# ===========================================================================

class TestFaithfulness:
    def test_high_faithfulness(self, mock_judge, sample_contexts):
        mock_judge.ask_json.return_value = {
            "claims": [
                {"claim": "dignity required", "supported": True, "evidence": "quote"},
                {"claim": "cultural safety", "supported": True, "evidence": "quote"},
            ],
            "supported_count": 2,
            "total_count": 2,
            "score": 1.0,
        }
        metric = FaithfulnessMetric(judge=mock_judge)
        result = metric.compute("q", "answer", sample_contexts)
        assert result.score == 1.0
        assert result.details["supported"] == 2

    def test_low_faithfulness(self, mock_judge, sample_contexts):
        mock_judge.ask_json.return_value = {
            "claims": [
                {"claim": "true claim", "supported": True, "evidence": "quote"},
                {"claim": "hallucinated", "supported": False, "evidence": None},
            ],
            "supported_count": 1,
            "total_count": 2,
            "score": 0.5,
        }
        metric = FaithfulnessMetric(judge=mock_judge)
        result = metric.compute("q", "answer", sample_contexts)
        assert result.score == 0.5

    def test_empty_answer(self, mock_judge, sample_contexts):
        metric = FaithfulnessMetric(judge=mock_judge)
        result = metric.compute("q", "", sample_contexts)
        assert result.score == 0.0
        mock_judge.ask_json.assert_not_called()

    def test_json_error_handling(self, mock_judge, sample_contexts):
        mock_judge.ask_json.side_effect = json.JSONDecodeError("err", "doc", 0)
        metric = FaithfulnessMetric(judge=mock_judge)
        result = metric.compute("q", "answer", sample_contexts)
        assert result.score == 0.0
        assert "error" in result.details

    def test_metric_name(self):
        assert FaithfulnessMetric.name == "faithfulness"


class TestAnswerRelevancy:
    def test_high_relevancy(self, mock_judge):
        mock_judge.ask_json.return_value = {
            "completeness": 0.9,
            "directness": 0.95,
            "specificity": 0.85,
            "correctness": 0.9,
            "score": 0.9,
            "reasoning": "good answer",
        }
        metric = AnswerRelevancyMetric(judge=mock_judge)
        result = metric.compute("What is Standard 1?", "Standard 1 is...", [], "ground truth")
        assert result.score == 0.9
        assert result.details["completeness"] == 0.9

    def test_low_relevancy(self, mock_judge):
        mock_judge.ask_json.return_value = {
            "completeness": 0.2,
            "directness": 0.3,
            "specificity": 0.1,
            "correctness": 0.2,
            "score": 0.2,
            "reasoning": "off-topic",
        }
        metric = AnswerRelevancyMetric(judge=mock_judge)
        result = metric.compute("q", "irrelevant", [])
        assert result.score == 0.2

    def test_empty_answer(self, mock_judge):
        metric = AnswerRelevancyMetric(judge=mock_judge)
        result = metric.compute("q", "", [])
        assert result.score == 0.0
        mock_judge.ask_json.assert_not_called()

    def test_metric_name(self):
        assert AnswerRelevancyMetric.name == "answer_relevancy"


class TestContextPrecision:
    def test_all_relevant(self, mock_judge, sample_contexts):
        mock_judge.ask_json.return_value = {
            "chunks": [
                {"chunk_index": 0, "relevant": True, "reason": "directly relevant"},
                {"chunk_index": 1, "relevant": True, "reason": "relevant"},
                {"chunk_index": 2, "relevant": True, "reason": "relevant"},
            ],
            "relevant_count": 3,
            "total_count": 3,
            "average_precision": 1.0,
            "score": 1.0,
        }
        metric = ContextPrecisionMetric(judge=mock_judge)
        result = metric.compute("q", "answer", sample_contexts)
        assert result.score == 1.0
        assert result.details["relevant"] == 3

    def test_partial_relevance(self, mock_judge, sample_contexts):
        mock_judge.ask_json.return_value = {
            "chunks": [
                {"chunk_index": 0, "relevant": True, "reason": "relevant"},
                {"chunk_index": 1, "relevant": False, "reason": "off-topic"},
            ],
            "relevant_count": 1,
            "total_count": 2,
            "average_precision": 0.5,
            "score": 0.5,
        }
        metric = ContextPrecisionMetric(judge=mock_judge)
        result = metric.compute("q", "answer", sample_contexts[:2])
        assert result.score == 0.5

    def test_empty_contexts(self, mock_judge):
        metric = ContextPrecisionMetric(judge=mock_judge)
        result = metric.compute("q", "answer", [])
        assert result.score == 0.0
        mock_judge.ask_json.assert_not_called()

    def test_metric_name(self):
        assert ContextPrecisionMetric.name == "context_precision"


# ===========================================================================
# Evaluator tests
# ===========================================================================

class TestEvaluator:
    def test_evaluate_single(self, golden_dataset, sample_contexts, mock_judge):
        # Mock all LLM metrics
        mock_judge.ask_json.return_value = {
            "claims": [{"claim": "c", "supported": True, "evidence": "e"}],
            "supported_count": 1, "total_count": 1, "score": 0.9,
            "completeness": 0.8, "directness": 0.9, "specificity": 0.8,
            "correctness": 0.85, "reasoning": "good",
            "chunks": [{"chunk_index": 0, "relevant": True, "reason": "r"}],
            "relevant_count": 1, "total_count": 1, "average_precision": 0.9,
        }

        evaluator = Evaluator()
        # Replace LLM metrics with mocked versions
        evaluator.metrics = [
            FaithfulnessMetric(judge=mock_judge),
            AnswerRelevancyMetric(judge=mock_judge),
            ContextPrecisionMetric(judge=mock_judge),
            CitationAccuracyMetric(),
        ]

        item = golden_dataset[0]
        answer = "Standard 1 is about dignity. [Source: Aged Care Quality Standards, p.1]"
        result = evaluator.evaluate_single(item, answer, sample_contexts)

        assert result.item_id == "test-001"
        assert "faithfulness" in result.scores
        assert "answer_relevancy" in result.scores
        assert "context_precision" in result.scores
        assert "citation_accuracy" in result.scores

    def test_evaluate_dataset_precomputed(self, golden_dataset, mock_judge):
        mock_judge.ask_json.return_value = {
            "claims": [{"claim": "c", "supported": True, "evidence": "e"}],
            "supported_count": 1, "total_count": 1, "score": 0.9,
            "completeness": 0.9, "directness": 0.9, "specificity": 0.9,
            "correctness": 0.9, "reasoning": "ok",
            "chunks": [{"chunk_index": 0, "relevant": True, "reason": "r"}],
            "relevant_count": 1, "total_count": 1, "average_precision": 0.9,
        }

        evaluator = Evaluator()
        evaluator.metrics = [
            FaithfulnessMetric(judge=mock_judge),
            AnswerRelevancyMetric(judge=mock_judge),
            ContextPrecisionMetric(judge=mock_judge),
            CitationAccuracyMetric(),
        ]

        precomputed = [
            {
                "id": "test-001",
                "answer": "Standard 1 covers dignity. [Source: Standards, p.1]",
                "contexts": ["Standard 1 requires dignity and respect."],
            },
            {
                "id": "test-002",
                "answer": "Safe medication management. [Source: Standards, p.5]",
                "contexts": ["Medication management includes safe prescribing."],
            },
        ]

        summary = evaluator.evaluate_dataset(golden_dataset, precomputed=precomputed)
        assert summary.total_items == 2
        assert len(summary.results) == 2
        assert "faithfulness" in summary.metric_averages

    def test_summary_pass_rate(self):
        evaluator = Evaluator(thresholds={"faithfulness": 0.8})
        evaluator.metrics = []

        r1 = EvalResult(
            item_id="1", question="q", answer="a", contexts=[], ground_truth="g",
            scores={"faithfulness": 0.9},
        )
        r2 = EvalResult(
            item_id="2", question="q", answer="a", contexts=[], ground_truth="g",
            scores={"faithfulness": 0.5},
        )
        summary = evaluator._build_summary("test", [r1, r2], 1.0)
        assert summary.pass_rate == 0.5

    def test_category_breakdown(self):
        evaluator = Evaluator(thresholds={"faithfulness": 0.8})
        evaluator.metrics = []

        r1 = EvalResult(
            item_id="1", question="q", answer="a", contexts=[], ground_truth="g",
            scores={"faithfulness": 0.9}, category="factual",
        )
        r2 = EvalResult(
            item_id="2", question="q", answer="a", contexts=[], ground_truth="g",
            scores={"faithfulness": 0.7}, category="clinical",
        )
        summary = evaluator._build_summary("test", [r1, r2], 1.0)
        assert "factual" in summary.category_breakdown
        assert "clinical" in summary.category_breakdown
        assert summary.category_breakdown["factual"]["faithfulness"] == 0.9

    def test_save_results(self, tmp_path):
        summary = EvalSummary(
            run_id="test-run",
            timestamp="2026-03-04T00:00:00Z",
            total_items=1,
            results=[],
            metric_averages={"faithfulness": 0.9},
            thresholds={"faithfulness": 0.85},
            pass_rate=1.0,
        )
        evaluator = Evaluator()
        out_path = evaluator.save_results(summary, output_dir=tmp_path)
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["run_id"] == "test-run"

    def test_eval_result_passed(self):
        r = EvalResult(
            item_id="1", question="q", answer="a", contexts=[], ground_truth="g",
            scores={"faithfulness": 0.9, "relevancy": 0.8},
        )
        assert r.passed is True

        r2 = EvalResult(
            item_id="2", question="q", answer="a", contexts=[], ground_truth="g",
            scores={"faithfulness": 0.3, "relevancy": 0.8},
        )
        assert r2.passed is False

    def test_empty_dataset(self, golden_dataset):
        golden_dataset.items = []
        evaluator = Evaluator()
        summary = evaluator.evaluate_dataset(golden_dataset, precomputed=[])
        assert summary.total_items == 0
        assert summary.pass_rate == 0.0
