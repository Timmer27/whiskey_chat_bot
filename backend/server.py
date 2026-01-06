import os
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from llm.embeddings import get_embeddings
from llm.model import get_llm_client
from llm.rag import generate_answer
from db.qdrant import get_qdrant_client

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None

# 캐싱된 clients 있으면 그대로 재사용
@lru_cache(maxsize=1)
def cached_llm_client():
    return get_llm_client()

@lru_cache(maxsize=1)
def cached_embedding_client():
    return get_embeddings(embed_model=os.getenv("EMBED_MODEL"))

@lru_cache(maxsize=1)
def cached_qdrant_client():
    return get_qdrant_client()

# 서버 시작 시점에 미리 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = cached_llm_client()
    embedder = cached_embedding_client()
    qdrant = cached_qdrant_client()

    app.state.llm = llm
    app.state.embedder = embedder
    app.state.qdrant = qdrant

    yield

    cached_llm_client.cache_clear()
    cached_embedding_client.cache_clear()
    cached_qdrant_client.cache_clear()


app = FastAPI(title="Whisky RAG API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat/whiskey")
def chat(body: ChatRequest, request: Request):
    ans, ctxs, ctx_texts = generate_answer(
        qdrant=request.app.state.qdrant,
        embedder=request.app.state.embedder,
        llm=request.app.state.llm,
        question=body.question,
        model="llama-3.1-8b-instant",
    )
    return ans
