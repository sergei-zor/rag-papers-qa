"""
FastAPI wrapper for the RAG pipeline.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import retriever
from .generator import generate_answer

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model and Chroma collection.")
    retriever.initialize()
    logger.info("Ready.")
    yield

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str = Field(max_length=1000)
    n_results: int = Field(default=6, ge=1, le=10)

class Source(BaseModel):
    source: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": retriever.is_ready(),
        "chunk_count": retriever.chunk_count(),
    }

# A plain "def", not "async def": the LLM call blocks, so FastAPI must run it
# in a worker thread instead of on the event loop that also serves /health.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Empty question")

    try:
        result = generate_answer(request.question, n_results=request.n_results)
    except Exception:
        logger.exception("Answer generation failed")
        raise HTTPException(502, "The LLM backend is unavailable or returned an error")

    seen = set()
    unique_sources = []
    for s in result["sources"]:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return QueryResponse(answer=result["answer"], sources=unique_sources)
