"""Generator — builds prompts and calls Claude for cited answers."""
import os
from dataclasses import dataclass
from pathlib import Path
import yaml, anthropic
from dotenv import load_dotenv
from src.retrieval.retriever import RetrievedChunk
load_dotenv()

@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    input_tokens: int
    output_tokens: int
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

class Generator:
    def __init__(self):
        self.prompts = load_prompts()
        self.gen_config = load_generation_config()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", self.gen_config.get("model","claude-sonnet-4-5-20250929"))
    def generate(self, question, chunks):
        ctx = build_context(chunks)
        sys_prompt = self.prompts["system_prompt"].format(context=ctx, question=question)
        r = self.client.messages.create(model=self.model,
            max_tokens=self.gen_config.get("max_tokens",1024),
            temperature=self.gen_config.get("temperature",0.1),
            messages=[{"role":"user","content":question}], system=sys_prompt)
        return RAGResponse(question=question, answer=r.content[0].text,
            retrieved_chunks=chunks, model=self.model,
            input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)
