# RAG Backend (FastAPI + FAISS + LangChain ReAct agent)

This replaces the previous Node/Express + in-browser-index backend with a
Python service that matches the project's intended architecture:

- **FastAPI** app (`app/main.py`) exposing document upload, deletion, stats,
  and a streaming chat endpoint.
- **FAISS** (`IndexFlatIP` over normalized `sentence-transformers`
  embeddings) for dense retrieval, fused with **BM25** lexical retrieval via
  **Reciprocal Rank Fusion** (`app/vector_store.py`).
- **LangChain ReAct agent** (`app/agent.py`) wrapping a `search_documents`
  tool around the hybrid retriever, using `langchain-groq` as the LLM. The
  agent decides when and how many times to search before answering, and
  emits `[N]` citation markers tied to the chunks it actually retrieved.
- **Per-session vector-store isolation**: every request carries an
  `X-Session-Id` header; each session gets its own FAISS index and BM25
  index in memory, with a 2-hour idle TTL eviction.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY to .env
uvicorn app.main:app --reload --port 8000
```

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/health` | GET | Config + model check |
| `/api/documents` | POST (multipart) | Upload + chunk + index a `.txt`/`.md`/`.pdf` |
| `/api/documents/{name}` | DELETE | Remove a document from the session |
| `/api/documents/stats` | GET | Doc/chunk counts for the session |
| `/api/chat` | POST | Streaming, citation-backed answer via the ReAct agent |

All endpoints (except `/api/health`) require an `X-Session-Id` header.

## Why a ReAct agent instead of a single retrieve-then-generate call

The agent can issue more than one search per turn — e.g. it can search
narrowly first, notice the excerpts don't answer the question, and issue a
follow-up query with different terms — before committing to a final,
citation-backed answer. This is the "agentic" behavior the single-shot
hybrid retrieval pipeline didn't have.

## Frontend integration

The React app now calls this service directly instead of the old Express
proxy. See `src/api.js` for the client and `src/App.jsx` for usage. Set
`VITE_API_BASE_URL` in the frontend `.env` if the backend isn't running on
`http://localhost:8000`.
