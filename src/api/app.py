"""FastAPI application for the Healthcare RAG System."""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)

class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    collection_size: int
    model: str

retriever = None
generator = None

@asynccontextmanager
async def lifespan(app):
    global retriever, generator
    print("Starting Healthcare RAG System...")
    retriever = Retriever()
    generator = Generator()
    coll = retriever.embedder.get_or_create_collection()
    print(f"Ready. Collection: {coll.count()} chunks")
    yield

app = FastAPI(title="Healthcare RAG System", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if retriever is None: raise HTTPException(503, "Not initialised")
    coll = retriever.embedder.get_or_create_collection()
    return HealthResponse(status="healthy", collection_size=coll.count(), model=generator.model)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if retriever is None or generator is None: raise HTTPException(503, "Not initialised")
    start = time.perf_counter()
    chunks = retriever.retrieve(request.question, top_k=request.top_k)
    if not chunks: raise HTTPException(404, "No relevant documents found.")
    response = generator.generate(request.question, chunks)
    ms = (time.perf_counter()-start)*1000
    return QueryResponse(question=response.question, answer=response.answer,
        model=response.model, input_tokens=response.input_tokens,
        output_tokens=response.output_tokens, cost_usd=response.cost_usd, latency_ms=ms)
