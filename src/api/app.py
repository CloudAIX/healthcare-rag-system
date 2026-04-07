"""FastAPI application for the Healthcare RAG System."""
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.security import (
    ALLOWED_ORIGINS,
    TokenRequest,
    TokenResponse,
    authenticate,
    create_access_token,
    limiter,
    verify_api_key,
)
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

# --- Request / Response models ---

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

# --- App lifecycle ---

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

app = FastAPI(
    title="Healthcare RAG System",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware ---

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

# --- Routes ---

@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
@limiter.limit("5/minute")
async def get_token(request: Request, body: TokenRequest, _=Depends(verify_api_key)):
    """Exchange a valid API key for a short-lived JWT token."""
    token, expires_in = create_access_token()
    return TokenResponse(access_token=token, expires_in=expires_in)

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Public health check — no auth required."""
    if retriever is None:
        raise HTTPException(503, "Not initialised")
    coll = retriever.embedder.get_or_create_collection()
    return HealthResponse(status="healthy", collection_size=coll.count(), model=generator.model)

@app.post("/query", response_model=QueryResponse, tags=["rag"])
@limiter.limit("30/minute")
async def query(
    request: Request,
    body: QueryRequest,
    auth: str = Depends(authenticate),
):
    """Run a RAG query against the aged care standards corpus."""
    if retriever is None or generator is None:
        raise HTTPException(503, "Not initialised")
    start = time.perf_counter()
    chunks = retriever.retrieve(body.question, top_k=body.top_k)
    if not chunks:
        raise HTTPException(404, "No relevant documents found.")
    response = generator.generate(body.question, chunks)
    ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        question=response.question,
        answer=response.answer,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=ms,
    )
