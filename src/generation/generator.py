"""Generator — builds prompts and calls Claude for cited answers."""
import os
from dataclasses import dataclass
from pathlib import Path
import yaml, anthropic
from dotenv import load_dotenv
from src.retrieval.retriever import RetrievedChunk
from src.safety import TriageResult, check_urgent
load_dotenv()

SAFETY_MODEL = "safety-triage"

@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    input_tokens: int
    output_tokens: int
    safety: TriageResult | None = None
    @property
    def cost_usd(self):
        return (self.input_tokens*3/1_000_000) + (self.output_tokens*15/1_000_000)

def load_prompts():
    p = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
    with open(p) as f: return yaml.safe_load(f)

def load_generation_config():
    p = Path(__file__).parent.parent.parent / "config" / "retrieval_config.yaml"
    with open(p) as f: return yaml.safe_load(f)["generation"]

def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"--- CHUNK {i} ---\nSource: {c.document_title}\n"
            f"Pages: {','.join(str(p) for p in c.page_numbers)}\n"
            f"Sections: {', '.join(c.sections) if c.sections else 'N/A'}\n\n{c.text}\n")
    return "\n".join(parts)

def safety_response(question, result: TriageResult) -> RAGResponse:
    """Fixed emergency guidance in place of a generated answer: no retrieval, no LLM call."""
    return RAGResponse(question=question, answer=result.message, retrieved_chunks=[],
        model=SAFETY_MODEL, input_tokens=0, output_tokens=0, safety=result)

def emergency_check(question):
    """Return a ready RAGResponse if the question describes a possible emergency, else None.

    Call this before retrieval. Generator.generate() runs the same check again,
    so a caller that forgets still never sends an emergency to the LLM.
    """
    result = check_urgent(question)
    if result is not None and result.is_emergency:
        return safety_response(question, result)
    return None

class Generator:
    def __init__(self):
        self.prompts = load_prompts()
        self.gen_config = load_generation_config()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", self.gen_config.get("model","claude-sonnet-4-5-20250929"))
    def generate(self, question, chunks):
        triage = check_urgent(question)
        if triage is not None and triage.is_emergency:
            return safety_response(question, triage)
        ctx = build_context(chunks)
        sys_prompt = self.prompts["system_prompt"].format(context=ctx, question=question)
        r = self.client.messages.create(model=self.model,
            max_tokens=self.gen_config.get("max_tokens",1024),
            temperature=self.gen_config.get("temperature",0.1),
            messages=[{"role":"user","content":question}], system=sys_prompt)
        answer = r.content[0].text
        if triage is not None:  # advisory: policy question on an urgent topic
            answer = triage.message + "\n\n---\n\n" + answer
        return RAGResponse(question=question, answer=answer,
            retrieved_chunks=chunks, model=self.model,
            input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
            safety=triage)
