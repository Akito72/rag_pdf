"""
LangChain ReAct agent for citation-backed, multi-turn document analysis.

The agent has one tool: `search_documents`, which runs the hybrid
FAISS+BM25 RRF retrieval against the caller's session store. The agent
decides when to call it (it can call it more than once per turn to refine
a query), then answers using only the returned excerpts, marking claims
with [N] citation markers that map back to the source chunks it retrieved.
"""
from __future__ import annotations

import json
import os
import re

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from .vector_store import HybridVectorStore

REACT_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant that answers questions ONLY using
retrieved excerpts from the user's uploaded documents. You have access to
the following tools:

{tools}

Use exactly this format:

Question: the input question you must answer
Thought: reason about whether you need to search, and what to search for
Action: the action to take, must be one of [{tool_names}]
Action Input: the search query to run
Observation: the result of the search
... (Thought/Action/Action Input/Observation can repeat if you need to refine your search)
Thought: I now have enough information to answer
Final Answer: the final answer, written in plain text with inline [N]
  citation markers matching the SOURCE numbers from your search results,
  followed on a new line by CITATIONS_JSON: [{{"num": 1, "source": "...",
  "text": "..."}}, ...] listing only the citations you actually used. If the
  retrieved excerpts don't contain enough information, say so explicitly
  rather than guessing.

Conversation so far:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


def _format_search_results(items_with_indices: list[tuple[int, dict]]) -> str:
    if not items_with_indices:
        return "No relevant excerpts found in the uploaded documents."
    blocks = [
        f"[SOURCE {idx} | {r['source']}]\n{r['text']}"
        for idx, r in items_with_indices
    ]
    return "\n\n---\n\n".join(blocks)


class RagAgent:
    """Builds one ReAct agent per request, scoped to a session's document store."""

    def __init__(self, vector_store: HybridVectorStore, model_name: str | None = None):
        self.vector_store = vector_store
        self.model_name = model_name or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _make_search_tool(self, session_id: str, top_k: int, all_results: list) -> Tool:
        def _run(query: str) -> str:
            results = self.vector_store.hybrid_search(session_id, query, top_k=top_k)
            formatted_items = []
            for r in results:
                existing_idx = None
                for i, existing in enumerate(all_results):
                    if existing["text"] == r["text"] and existing["source"] == r["source"]:
                        existing_idx = i + 1
                        break
                if existing_idx is None:
                    all_results.append(r)
                    existing_idx = len(all_results)
                formatted_items.append((existing_idx, r))
            return _format_search_results(formatted_items)

        return Tool(
            name="search_documents",
            description=(
                "Search the user's uploaded documents using hybrid FAISS "
                "(dense) + BM25 (lexical) retrieval fused with Reciprocal "
                "Rank Fusion. Input should be a focused search query. "
                "Returns the most relevant excerpts with source labels."
            ),
            func=_run,
        )

    def run(
        self,
        session_id: str,
        question: str,
        chat_history: list[dict],
        top_k: int = 5,
        max_tokens: int = 1000,
    ) -> dict:
        all_results: list[dict] = []
        tool = self._make_search_tool(session_id, top_k, all_results)

        llm = ChatGroq(model=self.model_name, max_tokens=max_tokens, temperature=0.2)
        agent = create_react_agent(llm=llm, tools=[tool], prompt=REACT_PROMPT)
        executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            max_iterations=4,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in chat_history[-8:]
        )

        result = executor.invoke({"input": question, "chat_history": history_text})
        raw_output = result.get("output", "")

        citations = self._parse_citations(raw_output, all_results)
        answer_text = re.sub(r"\n?CITATIONS_JSON:[\s\S]*$", "", raw_output).strip()

        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": all_results,
            "steps": len(result.get("intermediate_steps", [])),
        }

    @staticmethod
    def _parse_citations(raw: str, last_results: list[dict]) -> list[dict]:
        match = re.search(r"CITATIONS_JSON:\s*(\[[\s\S]*?\])", raw)
        if match:
            try:
                parsed = json.loads(match.group(1))
                citations = []
                for c in parsed:
                    idx = c.get("num", 0) - 1
                    chunk = last_results[idx] if 0 <= idx < len(last_results) else None
                    citations.append({
                        "num": c.get("num"),
                        "source": c.get("source") or (chunk["source"] if chunk else None),
                        "text": c.get("text") or (chunk["text"][:200] + "..." if chunk else ""),
                    })
                return citations
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        used_nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw)})
        citations = []
        for num in used_nums:
            idx = num - 1
            if 0 <= idx < len(last_results):
                chunk = last_results[idx]
                citations.append({
                    "num": num,
                    "source": chunk["source"],
                    "text": chunk["text"][:200] + "...",
                })
        return citations
