"""
Per-session hybrid retrieval store.

Each session gets its own FAISS index (dense, via sentence-transformer
embeddings) plus a BM25 index (lexical) over the same chunks. Retrieval
fuses both rankings with Reciprocal Rank Fusion (RRF), matching the
resume's "FAISS and BM25 hybrid retrieval with Reciprocal Rank Fusion"
description exactly.

Sessions are isolated in-memory (per-session vector-store isolation) and
evicted on TTL so the process doesn't grow unbounded.
"""
from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RRF_K = 60
SESSION_TTL_SECONDS = 60 * 60 * 2  # 2 hours


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_text(text: str, chunk_size: int = 400, overlap_ratio: float = 0.2) -> list[str]:
    words = text.split()
    overlap = int(chunk_size * overlap_ratio)
    step = max(chunk_size - overlap, 1)
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


@dataclass
class DocChunks:
    doc_name: str
    chunks: list[str] = field(default_factory=list)


class SessionStore:
    """Holds all documents, the FAISS index, and the BM25 index for one session."""

    def __init__(self):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None
        self.last_used = time.time()

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


class HybridVectorStore:
    """
    Owns one embedding model shared across sessions (loading it once is what
    makes this practical), and a dict of per-session FAISS/BM25 indices so
    one user's uploaded documents are never visible to another session.
    """

    def __init__(self, embedding_model_name: str):
        self._model = SentenceTransformer(embedding_model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._sessions: dict[str, SessionStore] = {}
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype("float32")

    def _get_or_create(self, session_id: str) -> SessionStore:
        with self._lock:
            self._evict_stale()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionStore()
            store = self._sessions[session_id]
            store.last_used = time.time()
            return store

    def _evict_stale(self):
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_used > SESSION_TTL_SECONDS]
        for sid in stale:
            del self._sessions[sid]

    def add_document(self, session_id: str, doc_name: str, text: str, chunk_size: int = 400) -> int:
        store = self._get_or_create(session_id)
        new_chunks = chunk_text(text, chunk_size=chunk_size)
        if not new_chunks:
            return 0

        vectors = self._embed(new_chunks)

        if store.faiss_index is None:
            store.faiss_index = faiss.IndexFlatIP(self._dim)

        store.faiss_index.add(vectors)
        store.chunks.extend(new_chunks)
        store.sources.extend([doc_name] * len(new_chunks))
        store.bm25 = BM25Okapi([tokenize(c) for c in store.chunks])

        return len(new_chunks)

    def remove_document(self, session_id: str, doc_name: str) -> None:
        """Rebuilds the session's indices without the given document's chunks."""
        store = self._sessions.get(session_id)
        if store is None:
            return

        keep = [(c, s) for c, s in zip(store.chunks, store.sources) if s != doc_name]
        store.chunks = [c for c, _ in keep]
        store.sources = [s for _, s in keep]

        if store.chunks:
            vectors = self._embed(store.chunks)
            store.faiss_index = faiss.IndexFlatIP(self._dim)
            store.faiss_index.add(vectors)
            store.bm25 = BM25Okapi([tokenize(c) for c in store.chunks])
        else:
            store.faiss_index = None
            store.bm25 = None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def stats(self, session_id: str) -> dict:
        store = self._sessions.get(session_id)
        if store is None:
            return {"docs": 0, "chunks": 0}
        return {
            "docs": len(set(store.sources)),
            "chunks": len(store.chunks),
        }

    def hybrid_search(self, session_id: str, query: str, top_k: int = 5) -> list[dict]:
        """
        Dense (FAISS cosine via inner product on normalized vectors) + lexical
        (BM25) rankings, fused with Reciprocal Rank Fusion. Returns the top_k
        chunks with source attribution, ready for citation-backed generation.
        """
        store = self._sessions.get(session_id)
        if store is None or store.is_empty():
            return []

        n = len(store.chunks)
        k = min(top_k * 4, n)  # over-fetch each ranking before fusion

        # Dense ranking via FAISS
        query_vec = self._embed([query])
        _, dense_idx = store.faiss_index.search(query_vec, k)
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_idx[0]) if idx != -1}

        # Lexical ranking via BM25
        bm25_scores = store.bm25.get_scores(tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1][:k]
        lexical_rank = {
            int(idx): rank
            for rank, idx in enumerate(bm25_order)
            if bm25_scores[idx] > 0
        }

        # Reciprocal Rank Fusion across both rankings
        fused_scores: dict[int, float] = {}
        for idx, rank in dense_rank.items():
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for idx, rank in lexical_rank.items():
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        return [
            {
                "text": store.chunks[idx],
                "source": store.sources[idx],
                "score": score,
            }
            for idx, score in ranked
        ]
