"""
short_term_memory.py
---------------------
Phase 2 / Item 3a: Short-Term (multi-turn) Memory for the Cocktail Manual.

Implements the design from Memory.md Section 4b: the LangGraph State currently
resets after every single query, so a follow-up like "make it less sweet" has
no idea what "it" refers to. This module adds:

  1. A `needs_query_rewrite()` GATE (cheap, free, no LLM call) -- mirrors the
     exact same "escalation, not always-on" pattern already used in
     agentic_rag_pipeline.py's needs_agentic_escalation(). Turn 1 queries and
     fully self-contained queries (e.g. "How do I make a Negroni") skip
     rewriting entirely and go straight to the existing router.

  2. A `rewrite_query_node` (LLM call) -- ONLY invoked when the gate trips.
     Takes chat_history + the new dangling message and produces one
     standalone, self-contained question.

  3. `ConversationalRAGPipeline` -- wraps EscalatingRAGPipeline (the Phase 2
     Graph+Agentic pipeline) with a running `chat_history` list that persists
     ACROSS calls to run_workflow(), so a multi-turn demo_app.py session
     actually remembers prior turns.

Design rationale (per our discussion): rewriting on every turn is not just
wasteful (extra LLM call for queries that don't need it) -- it can actively
CORRUPT a perfectly clear query by wrongly injecting unrelated prior context.
The gate must run BEFORE any LLM call, using only free heuristics.
"""

import json
import re
import urllib.request
from typing import List, Dict, Any, Optional, Tuple

from agentic_rag_pipeline import EscalatingRAGPipeline
from graph_builder import CocktailKnowledgeGraph
from vector_storage_and_retrieval import HybridVectorStore
from rag_generation_and_pipeline import NEBIUS_API_KEY

NEBIUS_CHAT_URL = "https://api.studio.nebius.ai/v1/chat/completions"
REWRITE_LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"


# =====================================================================
# 1. THE GATE -- decides whether rewriting is needed AT ALL (no LLM call)
# =====================================================================

# Dangling-reference signals: pronouns/ellipsis/comparatives with no
# explicit subject of their own -- these queries CANNOT be understood
# without prior turns.
_DANGLING_REFERENCE_PATTERNS = [
    r"\bit\b", r"\bthat one\b", r"\bthis one\b", r"\bthe other one\b",
    r"\binstead\b", r"\bthe same\b",
    r"^\s*(?:make|do)\s+(?:it|that|this)\b",     # "make it less sweet"
    r"^\s*(?:less|more|stronger|weaker|sweeter|drier)\b",  # "less sweet please"
    r"^\s*what about\b", r"^\s*how about\b",
    r"^\s*and\s+(?:with|without)\b",              # "and without lime"
]
_DANGLING_REFERENCE_REGEX = re.compile("|".join(_DANGLING_REFERENCE_PATTERNS), re.IGNORECASE)


def _mentions_known_entity(question: str, known_entity_names: List[str]) -> bool:
    """
    True if the query already names its own subject explicitly (a cocktail,
    ingredient, or spirit from the corpus) -- meaning it is self-contained
    and does NOT need chat history to be understood.
    Reuses the same "mention detection" concept as graph_rag_pipeline.py's
    _mention() helper, kept standalone here since this gate must run before
    we know which pipeline path (Graph/Vector/Agentic) will even be chosen.
    """
    q_lower = question.lower()
    return any(name.lower() in q_lower for name in known_entity_names)


def needs_query_rewrite(question: str, has_history: bool, known_entity_names: List[str]) -> bool:
    """
    THE GATE. Returns True only when a query genuinely cannot be understood
    on its own AND there is prior history to resolve it against.

    Turn 1 (no history yet) -> always False, nothing to rewrite from.
    "How do I make a Negroni" -> False (names its own subject explicitly).
    "Make it less sweet" -> True (dangling reference + no named subject).
    """
    if not has_history:
        return False
    if _mentions_known_entity(question, known_entity_names):
        return False
    return bool(_DANGLING_REFERENCE_REGEX.search(question))


# =====================================================================
# 2. THE REWRITE NODE -- only called when the gate trips
# =====================================================================

_REWRITE_SYSTEM_PROMPT = """You rewrite a user's follow-up message into a single,
fully standalone question, using the conversation history for context.

Rules:
- Output ONLY the rewritten question. No preamble, no explanation, no quotes.
- Preserve the user's actual intent and any new constraint they just added.
- If the follow-up references "it"/"that"/"the other one", resolve it to the
  specific cocktail name from the most recent turn that discussed one.
- Do not answer the question. Only rewrite it.
"""


class QueryRewriter:
    """Calls the LLM once to contextualize a dangling follow-up into a standalone query."""

    def __init__(self, api_key: str = NEBIUS_API_KEY):
        self.api_key = api_key

    def rewrite(self, chat_history: List[Tuple[str, str]], new_question: str) -> str:
        if not (self.api_key and "v1." in self.api_key):
            # No valid key configured -> skip the network call entirely
            # (consistent with the api_key checks already used elsewhere in
            # this pipeline, e.g. embedding_selection.py / generate_node).
            print("   ⚠️  [Short-Term Memory] No valid NEBIUS_API_KEY configured; skipping rewrite call, using original query.")
            return new_question

        history_text = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]  # last 3 turns is enough context
        )
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONVERSATION HISTORY:\n{history_text}\n\nFOLLOW-UP MESSAGE:\n{new_question}"},
        ]
        payload = json.dumps({
            "model": REWRITE_LLM_MODEL_NAME,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 60,
        }).encode("utf-8")
        req = urllib.request.Request(
            NEBIUS_CHAT_URL,
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res = json.loads(response.read().decode("utf-8"))
                rewritten = res["choices"][0]["message"]["content"].strip().strip('"')
                return rewritten if rewritten else new_question
        except Exception as e:
            print(f"⚠️  [Short-Term Memory] Rewrite call failed ({e}); using original query unmodified.")
            return new_question  # graceful fallback: never block the pipeline on a rewrite failure


# =====================================================================
# 3. THE CONVERSATIONAL PIPELINE -- persists chat_history across turns
# =====================================================================

class ConversationalRAGPipeline(EscalatingRAGPipeline):
    """
    Wraps EscalatingRAGPipeline (Graph + Agentic RAG) with a running
    chat_history list that persists ACROSS calls to run_workflow() within
    one session -- e.g. one demo_app.py CLI run.

    NOTE ON SCOPE (Memory.md Section 4b vs 4b-1): this is SHORT-TERM memory
    only -- history lives in this Python object's memory for the lifetime of
    the process, exactly like the "one CLI session" scope described in
    Memory.md Section 4b. It does NOT persist across separate runs of
    demo_app.py, and it is NOT scoped per-user (no user_id) -- that is
    Long-Term Memory, deliberately out of scope here per Memory.md 4b-1's
    finding that Long-Term Memory requires solving identity resolution first.
    """

    def __init__(self, vector_store: HybridVectorStore, graph: CocktailKnowledgeGraph, api_key: str = NEBIUS_API_KEY):
        super().__init__(vector_store, graph, api_key)
        self.rewriter = QueryRewriter(api_key)
        self.chat_history: List[Tuple[str, str]] = []  # [(question, final_answer), ...]

        # Known entity names for the gate -- cocktails (from vector store) +
        # ingredients/spirits (from the graph), so "How do I make a Negroni"
        # is recognized as self-contained without needing an LLM call.
        self._known_entities = list({
            *(rec["drink_name"] for rec in vector_store.records),
            *(n["name"] for n in graph.nodes.values() if n["type"] in ("BaseSpirit", "Ingredient")),
        })

    def run_workflow(self, question: str):
        effective_question = question

        if needs_query_rewrite(question, bool(self.chat_history), self._known_entities):
            print(f"🔄 [Short-Term Memory] Dangling reference detected in '{question}' -> rewriting with history...")
            effective_question = self.rewriter.rewrite(self.chat_history, question)
            print(f"   ↳ Rewritten to: '{effective_question}'")
        else:
            reason = "no history yet" if not self.chat_history else "query is self-contained"
            print(f"⚡ [Short-Term Memory] Skipping rewrite ({reason}) for: '{question}'")

        result = super().run_workflow(effective_question)

        # Extract the final answer text regardless of which path answered
        # (fast router returns a dict-like state; agentic escalation returns
        # a dict with "final_answer" directly).
        final_answer = result.get("final_answer", "") if isinstance(result, dict) else result["final_answer"]
        self.chat_history.append((question, final_answer))

        return result


if __name__ == "__main__":
    from ingest_and_clean import ingest_and_clean_corpus
    from vector_storage_and_retrieval import (
        apply_structural_chunking, CocktailEmbedder, generate_embeddings_for_chunks,
    )
    from graph_builder import build_graph_from_documents

    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(cleaned_docs)
    embedder = CocktailEmbedder(model_name="Qwen/Qwen3-Embedding-8B", dimensions=4096)
    records = generate_embeddings_for_chunks(chunks, embedder)
    vector_store = HybridVectorStore(records, embedder)
    graph = build_graph_from_documents(cleaned_docs)

    pipeline = ConversationalRAGPipeline(vector_store, graph)

    # TEST: The exact 3-turn scenario requested --
    # Margarita -> "make it less sweet" (needs rewrite) -> Negroni (no rewrite needed)
    pipeline.run_workflow("How do I make a Margarita?")
    pipeline.run_workflow("Make it less sweet")
    pipeline.run_workflow("How do I make a Negroni")
