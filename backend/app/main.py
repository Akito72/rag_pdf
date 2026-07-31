"""
FastAPI backend for the RAG pipeline.

Endpoints:
  GET  /api/health              - liveness + config check
  POST /api/documents           - upload a document (multipart), chunk + index it
  DELETE /api/documents/{name}  - remove a document from a session
  GET  /api/documents/stats     - doc/chunk counts for a session
  POST /api/chat                - streaming, citation-backed chat via the ReAct agent

Session identity comes from an `X-Session-Id` header the frontend generates
once per browser session, giving each user their own isolated FAISS/BM25
vector store (no cross-session leakage of uploaded documents).
"""
from __future__ import annotations

import asyncio
import io
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader

from .agent import RagAgent
from .vector_store import HybridVectorStore

load_dotenv()

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB
ALLOWED_MODELS = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"}
DEFAULT_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="RAG Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = HybridVectorStore(EMBEDDING_MODEL)


def require_session_id(x_session_id: str | None) -> str:
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="X-Session-Id header is required.")
    return x_session_id.strip()


def extract_pdf_text(raw_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "groq_key_configured": bool(os.environ.get("GROQ_API_KEY")),
        "embedding_model": EMBEDDING_MODEL,
        "models": sorted(ALLOWED_MODELS),
    }


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
):
    session_id = require_session_id(x_session_id)
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 15MB limit.")

    if file.filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
        text = extract_pdf_text(raw_bytes)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    if not text or len(text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Could not extract sufficient text from file.")

    chunk_count = await asyncio.to_thread(
        vector_store.add_document, session_id, file.filename, text
    )

    return {
        "name": file.filename,
        "chunks": chunk_count,
        "size": len(raw_bytes),
    }


@app.delete("/api/documents/{doc_name}")
def delete_document(doc_name: str, x_session_id: str | None = Header(default=None)):
    session_id = require_session_id(x_session_id)
    vector_store.remove_document(session_id, doc_name)
    return {"removed": doc_name}


@app.get("/api/documents/stats")
def document_stats(x_session_id: str | None = Header(default=None)):
    session_id = require_session_id(x_session_id)
    return vector_store.stats(session_id)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    model: str | None = None
    top_k: int = 5
    max_tokens: int = 1000


@app.post("/api/chat")
async def chat(payload: ChatRequest, x_session_id: str | None = Header(default=None)):
    session_id = require_session_id(x_session_id)

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    stats = vector_store.stats(session_id)
    if stats["chunks"] == 0:
        raise HTTPException(status_code=400, detail="Upload at least one document before asking questions.")

    model_name = payload.model if payload.model in ALLOWED_MODELS else DEFAULT_MODEL
    top_k = max(1, min(payload.top_k, 10))
    max_tokens = max(1, min(payload.max_tokens, 2000))

    agent = RagAgent(vector_store, model_name=model_name)

    async def event_stream():
        try:
            result = await asyncio.to_thread(
                agent.run,
                session_id,
                payload.question,
                [m.model_dump() for m in payload.history],
                top_k,
                max_tokens,
            )
            yield f"data: {json.dumps({'type': 'final', 'result': result})}\n\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to client as an SSE error event
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"},
    )
