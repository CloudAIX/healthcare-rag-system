"""RAGAS-style evaluation metrics for RAG pipeline quality.

Four metrics implemented:
1. Faithfulness     — Are answer claims grounded in retrieved context?
2. Answer Relevancy — Does the answer address the question?
3. Context Precision — Are retrieved contexts relevant to the question?
4. Citation Accuracy — Do [Source: ...] citations map to real chunks?

Metrics 1-3 use Claude as an LLM judge. Metric 4 uses pattern matching.
All metrics return a float score in [0.0, 1.0].
"""
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Base metric
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    score: float
    details: dict = field(default_factory=dict)


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    name: str = "base"

    @abstractmethod
    def compute(self, question: str, answer: str, contexts: list[str],
                ground_truth: str = "", **kwargs) -> MetricResult:
        ...


# ---------------------------------------------------------------------------
# Shared LLM judge helper
# ---------------------------------------------------------------------------

class LLMJudge:
    """Thin wrapper around Claude for structured evaluation prompts."""

    def __init__(self, model: str | None = None):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv(
            "EVAL_MODEL", "claude-sonnet-4-5-20250929"
        )

    def ask_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip()
        # Extract JSON from markdown code fence if present
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        return json.loads(text)


# ---------------------------------------------------------------------------
# 1. Faithfulness  (RAGAS core metric)
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = """You are an impartial evaluator. Your job is to assess whether
an answer is faithful to (i.e. supported by) the provided context chunks.

Step 1: Extract every discrete factual claim from the ANSWER.
Step 2: For each claim, determine if it is SUPPORTED or UNSUPPORTED by the CONTEXT.
A claim is supported if the context contains evidence that directly supports it.

Return ONLY a JSON object (no markdown, no extra text):
{
  "claims": [
    {"claim": "...", "supported": true/false, "evidence": "brief quote or null"}
  ],
  "supported_count": <int>,
  "total_count": <int>,
  "score": <float 0-1>
}"""


class FaithfulnessMetric(BaseMetric):
    """Measures whether the answer is grounded in retrieved context (no hallucination).

    Score = supported_claims / total_claims.  1.0 = perfectly faithful.
    """

    name = "faithfulness"

    def __init__(self, judge: LLMJudge | None = None):
        self.judge = judge or LLMJudge()

    def compute(self, question: str, answer: str, contexts: list[str],
                ground_truth: str = "", **kwargs) -> MetricResult:
        if not answer.strip():
            return MetricResult(score=0.0, details={"error": "empty answer"})

        ctx_block = "\n---\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(contexts))
        user_msg = f"CONTEXT:\n{ctx_block}\n\nANSWER:\n{answer}"

        try:
            result = self.judge.ask_json(FAITHFULNESS_SYSTEM, user_msg)
            score = float(result.get("score", 0.0))
            return MetricResult(
                score=max(0.0, min(1.0, score)),
                details={
                    "claims": result.get("claims", []),
                    "supported": result.get("supported_count", 0),
                    "total": result.get("total_count", 0),
                },
            )
        except (json.JSONDecodeError, anthropic.APIError) as e:
            return MetricResult(score=0.0, details={"error": str(e)})


# ---------------------------------------------------------------------------
# 2. Answer Relevancy  (RAGAS core metric)
# ---------------------------------------------------------------------------

RELEVANCY_SYSTEM = """You are an impartial evaluator. Your job is to assess whether
an answer is relevant to the question asked.

Evaluate on these criteria:
1. COMPLETENESS: Does the answer address all parts of the question?
2. DIRECTNESS: Does the answer directly respond (not tangential)?
3. SPECIFICITY: Does the answer provide specific information (not vague)?
4. CORRECTNESS: Compared to the ground truth, is the answer factually aligned?

Return ONLY a JSON object (no markdown, no extra text):
{
  "completeness": <float 0-1>,
  "directness": <float 0-1>,
  "specificity": <float 0-1>,
  "correctness": <float 0-1>,
  "score": <float 0-1>,
  "reasoning": "brief explanation"
}

The final score should be a weighted average:
  score = 0.3*completeness + 0.2*directness + 0.2*specificity + 0.3*correctness"""


class AnswerRelevancyMetric(BaseMetric):
    """Measures whether the answer addresses the question asked.

    Uses multi-criteria scoring: completeness, directness, specificity, correctness.
    """

    name = "answer_relevancy"

    def __init__(self, judge: LLMJudge | None = None):
        self.judge = judge or LLMJudge()

    def compute(self, question: str, answer: str, contexts: list[str],
                ground_truth: str = "", **kwargs) -> MetricResult:
        if not answer.strip():
            return MetricResult(score=0.0, details={"error": "empty answer"})

        user_msg = (
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"GROUND TRUTH:\n{ground_truth}"
        )

        try:
            result = self.judge.ask_json(RELEVANCY_SYSTEM, user_msg)
            score = float(result.get("score", 0.0))
            return MetricResult(
                score=max(0.0, min(1.0, score)),
                details={
                    "completeness": result.get("completeness", 0.0),
                    "directness": result.get("directness", 0.0),
                    "specificity": result.get("specificity", 0.0),
                    "correctness": result.get("correctness", 0.0),
                    "reasoning": result.get("reasoning", ""),
                },
            )
        except (json.JSONDecodeError, anthropic.APIError) as e:
            return MetricResult(score=0.0, details={"error": str(e)})


# ---------------------------------------------------------------------------
# 3. Context Precision  (RAGAS core metric)
# ---------------------------------------------------------------------------

PRECISION_SYSTEM = """You are an impartial evaluator. Your job is to assess whether
the retrieved context chunks are relevant to answering the question.

For EACH chunk, determine:
- Is it RELEVANT (contains information needed to answer the question)?
- Or IRRELEVANT (does not help answer the question)?

Context precision rewards having relevant chunks ranked higher.
Use Average Precision: AP = sum(precision@k * rel(k)) / total_relevant

Return ONLY a JSON object (no markdown, no extra text):
{
  "chunks": [
    {"chunk_index": 0, "relevant": true/false, "reason": "brief explanation"}
  ],
  "relevant_count": <int>,
  "total_count": <int>,
  "average_precision": <float 0-1>,
  "score": <float 0-1>
}"""


class ContextPrecisionMetric(BaseMetric):
    """Measures whether retrieved contexts are relevant to the question.

    Uses Average Precision (AP) to reward relevant chunks appearing earlier.
    Score = AP = sum(precision@k * rel(k)) / total_relevant.
    """

    name = "context_precision"

    def __init__(self, judge: LLMJudge | None = None):
        self.judge = judge or LLMJudge()

    def compute(self, question: str, answer: str, contexts: list[str],
                ground_truth: str = "", **kwargs) -> MetricResult:
        if not contexts:
            return MetricResult(score=0.0, details={"error": "no contexts"})

        ctx_block = "\n---\n".join(
            f"[Chunk {i+1}]: {c}" for i, c in enumerate(contexts)
        )
        user_msg = f"QUESTION:\n{question}\n\nRETRIEVED CHUNKS:\n{ctx_block}"

        try:
            result = self.judge.ask_json(PRECISION_SYSTEM, user_msg)
            score = float(result.get("score", result.get("average_precision", 0.0)))
            return MetricResult(
                score=max(0.0, min(1.0, score)),
                details={
                    "chunks": result.get("chunks", []),
                    "relevant": result.get("relevant_count", 0),
                    "total": result.get("total_count", len(contexts)),
                },
            )
        except (json.JSONDecodeError, anthropic.APIError) as e:
            return MetricResult(score=0.0, details={"error": str(e)})


# ---------------------------------------------------------------------------
# 4. Citation Accuracy  (custom metric — pattern-based)
# ---------------------------------------------------------------------------

# Pattern matches: [Source: Document Title, Section X, p.N] or similar
CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^\]]+)\]", re.IGNORECASE
)


class CitationAccuracyMetric(BaseMetric):
    """Measures whether citations in the answer map to actual retrieved chunks.

    Extracts [Source: ...] patterns from the answer and checks each against
    the metadata of retrieved chunks. No LLM call required.

    Score = verified_citations / total_citations.  1.0 = all citations valid.
    """

    name = "citation_accuracy"

    def compute(self, question: str, answer: str, contexts: list[str],
                ground_truth: str = "", **kwargs) -> MetricResult:
        chunk_metadata: list[dict] = kwargs.get("chunk_metadata", [])

        citations = CITATION_PATTERN.findall(answer)
        if not citations:
            # No citations found — check if answer should have them
            has_substance = len(answer.strip()) > 50 and "cannot find" not in answer.lower()
            if has_substance:
                return MetricResult(score=0.0, details={
                    "error": "answer has content but no citations",
                    "citations_found": 0,
                })
            # Out-of-scope answer with no citations is fine
            return MetricResult(score=1.0, details={
                "citations_found": 0,
                "note": "no citations needed (out-of-scope or refusal)",
            })

        # Build searchable text from chunk metadata and context
        reference_texts = []
        for meta in chunk_metadata:
            reference_texts.append(
                f"{meta.get('document_title', '')} "
                f"{meta.get('document_filename', '')} "
                f"{meta.get('sections', '')}"
            )
        for ctx in contexts:
            reference_texts.append(ctx[:200])

        ref_blob = " ".join(reference_texts).lower()

        verified = []
        for cite in citations:
            # Extract document name from citation
            parts = [p.strip() for p in cite.split(",")]
            doc_name = parts[0].lower()
            # Check if any meaningful part of the citation matches reference material
            tokens = [t for t in doc_name.split() if len(t) > 3]
            match = any(tok in ref_blob for tok in tokens) if tokens else False
            verified.append({"citation": cite, "verified": match})

        n_verified = sum(1 for v in verified if v["verified"])
        total = len(verified)
        score = n_verified / total if total > 0 else 0.0

        return MetricResult(
            score=score,
            details={
                "citations_found": total,
                "citations_verified": n_verified,
                "citations": verified,
            },
        )
