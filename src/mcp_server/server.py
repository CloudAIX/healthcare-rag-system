"""MCP server exposing the Aged Care Standards RAG to any MCP client.

Governed retrieval as a tool surface: every answer path returns citations,
and the tools are read-only by construction. Works against either backend
(local ChromaDB or Azure AI Search) via RAG_BACKEND.

Run (stdio):
    python -m src.mcp_server.server

Claude Code registration:
    claude mcp add aged-care-standards -- \
        /path/to/venv/bin/python -m src.mcp_server.server

First tool call loads the embedding + reranker models (~60s); subsequent
calls are fast. ANTHROPIC_API_KEY is only needed for standards_ask.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

mcp = MCPServer("aged_care_standards_mcp")

_retriever = None
_generator = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import Retriever

        _retriever = Retriever()
    return _retriever


def _get_generator():
    global _generator
    if _generator is None:
        from src.generation.generator import Generator

        _generator = Generator()
    return _generator


def _chunk_payload(c, include_text=True):
    out = {
        "chunk_id": c.chunk_id,
        "citation": c.citation,
        "document_title": c.document_title,
        "pages": c.page_numbers,
        "sections": c.sections,
        "score": round(float(c.score), 4),
    }
    if include_text:
        out["text"] = c.text
    return out


class SearchInput(BaseModel):
    """Input for standards_search."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., min_length=3, max_length=500,
                       description="Plain-English question or phrase, e.g. 'restrictive practices consent requirements'")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to return after re-ranking")
    include_text: bool = Field(default=True, description="Include full chunk text (False returns citations only)")


@mcp.tool(
    name="standards_search",
    annotations={"title": "Search the Aged Care Quality Standards", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def standards_search(params: SearchInput) -> str:
    """Hybrid search over the Strengthened Aged Care Quality Standards corpus.

    Runs vector + keyword retrieval with cross-encoder re-ranking and returns
    the top chunks, each carrying a [Source: document, section, pages] citation.
    Use this when you need grounded passages to reason over or quote.

    Returns: JSON {"query", "backend", "results": [{chunk_id, citation, document_title, pages, sections, score, text}]}
    """
    r = _get_retriever()
    chunks = r.retrieve(params.query, top_k=params.top_k)
    return json.dumps(
        {"query": params.query, "backend": r.backend,
         "results": [_chunk_payload(c, params.include_text) for c in chunks]},
        ensure_ascii=False, indent=2,
    )


class GetChunkInput(BaseModel):
    """Input for standards_get_chunk."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk_id: str = Field(..., min_length=5, max_length=200,
                          description="Chunk id from a previous standards_search result")


@mcp.tool(
    name="standards_get_chunk",
    annotations={"title": "Fetch one corpus chunk by id", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def standards_get_chunk(params: GetChunkInput) -> str:
    """Fetch the full text and metadata of a single chunk by its chunk_id.

    Use after standards_search (e.g. with include_text=False) to pull the
    complete passage for quoting.

    Returns: JSON {chunk_id, text, metadata} or an error with the failing id.
    """
    coll = _get_retriever().embedder.get_or_create_collection()
    got = coll.get(ids=[params.chunk_id], include=["documents", "metadatas"])
    if not got["ids"]:
        return json.dumps({"error": f"chunk_id not found: {params.chunk_id}",
                           "hint": "ids come from standards_search results"})
    return json.dumps(
        {"chunk_id": got["ids"][0], "text": got["documents"][0], "metadata": got["metadatas"][0]},
        ensure_ascii=False, indent=2,
    )


class AskInput(BaseModel):
    """Input for standards_ask."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(..., min_length=5, max_length=500,
                          description="Question to answer from the Standards, e.g. 'What does Standard 1 require about dignity?'")


@mcp.tool(
    name="standards_ask",
    annotations={"title": "Ask the Standards (full RAG answer)", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def standards_ask(params: AskInput) -> str:
    """Full RAG pipeline: retrieve, re-rank, then generate a grounded answer.

    Every claim in the answer carries a [Source: ...] citation back to the
    corpus. Calls the Anthropic API (requires ANTHROPIC_API_KEY; small cost
    per call). For raw passages without generation, use standards_search.

    Returns: JSON {"question", "answer", "citations": [...]}
    """
    r = _get_retriever()
    chunks = r.retrieve(params.question)
    if not chunks:
        return json.dumps({"error": "no relevant chunks retrieved", "question": params.question})
    try:
        answer = _get_generator().generate(params.question, chunks)
    except Exception as e:
        return json.dumps({"error": f"generation failed: {e}",
                           "hint": "check ANTHROPIC_API_KEY; standards_search still works without it"})
    answer_text = answer if isinstance(answer, str) else getattr(answer, "answer", str(answer))
    return json.dumps(
        {"question": params.question, "answer": answer_text,
         "citations": [c.citation for c in chunks]},
        ensure_ascii=False, indent=2,
    )


@mcp.tool(
    name="standards_corpus_info",
    annotations={"title": "Corpus and backend status", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def standards_corpus_info() -> str:
    """Report which retrieval backend is active and how many chunks it holds.

    Returns: JSON {"backend", "chunk_count", "corpus", "hybrid"}
    """
    r = _get_retriever()
    return json.dumps(
        {"backend": r.backend,
         "chunk_count": r.embedder.get_or_create_collection().count(),
         "corpus": "Strengthened Aged Care Quality Standards (Aged Care Act 2024)",
         "hybrid": r.backend == "azure" or r.enable_hybrid},
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
