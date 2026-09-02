"""
agentic_rag_pipeline.py
------------------------
Phase 2 / Item 1 follow-on: Agentic RAG escalation layer on top of Vector RAG
(vector_storage_and_retrieval.py) and Graph RAG (graph_builder.py).

Design decision (per RAG_Patterns.md Sections 5b-5d and Agents_Consideration.md's
Decision Tree): the existing regex-based route_query() in graph_rag_pipeline.py
stays as the FAST, FREE, DEFAULT path for single-hop queries. This module adds
a 4th escalation branch -- only compound / multi-hop / ambiguous queries get
promoted to a real LLM-driven ReAct loop that can chain Graph -> Vector tool
calls, exactly like the Negroni/Boulevardier/Americano example in
RAG_Patterns.md Section 5b:

    "I want a Negroni variant that doesn't use Campari, and tell me
     what glass it's served in."

This is NOT a wholesale replacement of route_query() -- see
Agents_Consideration.md Section 1 (cost/latency compound with every extra
LLM call) and the Decision Tree in Section 2 ("default left; only go right
when justified"). Escalation trigger = compound-intent detection, not "always
ask the LLM."

Confirmed live: Nebius Studio's meta-llama/Llama-3.3-70B-Instruct endpoint
supports real OpenAI-style `tools` / `tool_calls` function calling (verified
via a live test call before this file was written) -- so this uses genuine
.bind_tools()-style tool calling, not a manual keyword-parse imitation.
"""

import os
import re
import json
import urllib.request
from typing import List, Dict, Any, Optional

from graph_rag_pipeline import GraphRAGPipeline, route_query
from graph_builder import CocktailKnowledgeGraph
from vector_storage_and_retrieval import HybridVectorStore
from rag_generation_and_pipeline import NEBIUS_API_KEY
from missing_record_logger import log_missing_record

NEBIUS_CHAT_URL = "https://api.studio.nebius.ai/v1/chat/completions"
AGENT_LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MAX_AGENT_STEPS = 4  # stopping criteria per RAG_Patterns.md Personal Notes #3

# Heuristic signals that the agent's final DECIDE answer is a "no match found"
# conclusion rather than a real answer -- used to trigger missing-record
# logging. Kept deliberately loose/heuristic since this is a logging signal,
# not a user-facing decision (a false positive here just logs an extra line,
# it never changes what the user sees).
_NO_MATCH_SIGNALS = [
    r"\bno\b.*\b(?:variant|cocktail|match|recipe)\b.*\b(?:does not|doesn't|found|exist|satisf)\b",
    r"\bnone\s+(?:of|satisfy|qualif)\b",
    r"\bcouldn't\s+find\b",
    r"\bnot\s+(?:in|present in)\s+(?:the\s+)?manual\b",
]
_NO_MATCH_REGEX = re.compile("|".join(_NO_MATCH_SIGNALS), re.IGNORECASE)


# =====================================================================
# 1. ESCALATION TRIGGER -- decides Fast Router vs. Agentic Loop
# =====================================================================

# Signals that a query needs MULTIPLE chained lookups (not just one Graph
# OR one Vector call). Each of these independently implies a 2nd fact is
# being asked for on top of the first.
_COMPOUND_SIGNALS = [
    r"\band\s+(?:tell me|what|which|show)\b",     # "...and tell me what glass..."
    r"\bthat\s+(?:doesn't|does not|isn't)\b",      # "...variant that doesn't use..."
    r",\s*and\b",                                   # comma-joined second ask
    r"\bbut\s+(?:not|without)\b.*\band\b",          # "but not X, and Y"
]
_COMPOUND_REGEX = re.compile("|".join(_COMPOUND_SIGNALS), re.IGNORECASE)


def needs_agentic_escalation(question: str) -> bool:
    """
    Returns True only for compound / multi-hop queries where the fast
    regex router (route_query) would have to pick just ONE of Graph/Vector
    and therefore cannot satisfy the full ask in one pass.
    """
    return bool(_COMPOUND_REGEX.search(question))


# =====================================================================
# 2. TOOL DEFINITIONS -- thin wrappers around the existing Graph + Vector
#    query methods, described in OpenAI tool-calling schema
# =====================================================================

def build_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_variants",
                "description": (
                    "Get cocktails that are variants of a given cocktail "
                    "(Graph RAG relationship lookup)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"cocktail_name": {"type": "string"}},
                    "required": ["cocktail_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cocktail_uses_ingredient",
                "description": (
                    "Check whether a specific cocktail uses a specific ingredient "
                    "(Graph RAG ingredient-membership check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cocktail_name": {"type": "string"},
                        "ingredient_name": {"type": "string"},
                    },
                    "required": ["cocktail_name", "ingredient_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vector_recipe_lookup",
                "description": (
                    "Retrieve full recipe text (including glassware, ingredients, "
                    "preparation) for a specific cocktail by name (Vector RAG lookup)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"cocktail_name": {"type": "string"}},
                    "required": ["cocktail_name"],
                },
            },
        },
    ]


class AgentTools:
    """Executes the tool calls the LLM requests, against the real graph + vector store."""

    def __init__(self, graph: CocktailKnowledgeGraph, vector_store: HybridVectorStore):
        self.graph = graph
        self.vector_store = vector_store

    def get_variants(self, cocktail_name: str) -> Dict[str, Any]:
        variants, is_variant_of, cypher = self.graph.get_variants(cocktail_name)
        return {"target": cocktail_name, "variants": variants, "is_variant_of": is_variant_of, "cypher": cypher}

    def cocktail_uses_ingredient(self, cocktail_name: str, ingredient_name: str) -> Dict[str, Any]:
        cocktails_with_ingredient, cypher = self.graph.get_cocktails_by_ingredient(ingredient_name)
        uses_it = any(c.lower() == cocktail_name.lower() for c in cocktails_with_ingredient)
        return {
            "cocktail_name": cocktail_name,
            "ingredient_name": ingredient_name,
            "uses_ingredient": uses_it,
            "cypher": cypher,
        }

    def vector_recipe_lookup(self, cocktail_name: str) -> Dict[str, Any]:
        results = self.vector_store.search(cocktail_name, top_k=1, alpha=0.5)
        if not results:
            return {"cocktail_name": cocktail_name, "found": False}
        top = results[0]
        return {
            "cocktail_name": top["drink_name"],
            "found": True,
            "glassware": top["metadata"].get("glassware"),
            "base_spirit": top["metadata"].get("base_spirit"),
            "page_content": top["page_content"],
        }

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, name, None)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**args)
        except Exception as e:
            return {"error": str(e)}


# =====================================================================
# 3. THE AGENTIC ReAct LOOP (Think -> Act -> Observe -> Decide)
# =====================================================================

_AGENT_SYSTEM_PROMPT = """You are a Cocktail Knowledge Agent with access to tools that query a
Cocktail Manual's knowledge graph and vector store. Answer the user's question
by calling tools as needed. You may need MULTIPLE tool calls chained together
(e.g. find variants, then check if a variant still has an excluded ingredient,
then look up recipe details for the one that passes).

Rules:
- Only claim a fact if a tool call returned it. Do not guess ingredients or glassware.
- NEVER call the exact same tool with the exact same arguments twice. If you
  already have an observation for a (tool, arguments) pair, reuse it from the
  conversation history instead of calling it again.
- Before naming ANY cocktail as your final answer to a constraint like "doesn't
  use X", explicitly re-check: did the cocktail_uses_ingredient tool return
  uses_ingredient=true or uses_ingredient=false for that exact cocktail and
  that exact ingredient? If it returned true, that cocktail FAILS the "doesn't
  use X" constraint -- you must NOT name it as satisfying that constraint.
- If a variant fails a constraint (e.g. still contains an excluded ingredient),
  try the NEXT untried candidate variant instead of giving up.
- If you have checked EVERY known variant returned by get_variants and NONE of
  them satisfy the constraint, STOP calling tools and give a final answer
  explicitly stating that no variant in the manual satisfies the constraint
  (this is a valid, complete answer -- do not keep retrying).
- Once you have enough information to fully answer (including the "no match
  found" case above), respond with a final plain-text answer (no more tool
  calls) citing which cocktail you landed on (or why none qualify) tagged
  [AgenticRAG].
"""


def _find_contradiction(final_text: str, seen_calls: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic (non-LLM) groundedness check: scans every
    cocktail_uses_ingredient observation actually returned by tools, and
    flags a contradiction if the final answer claims that same cocktail
    satisfies a "doesn't use <ingredient>" constraint when the tool actually
    observed uses_ingredient=True for that exact (cocktail, ingredient) pair.

    This exists because prompt instructions alone were insufficient to
    prevent the model from asserting the opposite of its own tool
    observation (verified live: the model claimed "Boulevardier doesn't use
    Campari" immediately after a cocktail_uses_ingredient call that returned
    uses_ingredient=True for that exact pair). Mirrors the same "don't trust
    the model to self-police, verify in code" lesson already applied to the
    duplicate-tool-call loop guard above.
    """
    final_lower = final_text.lower()
    for call_key, observation in seen_calls.items():
        if not call_key.startswith("cocktail_uses_ingredient("):
            continue
        if not observation.get("uses_ingredient"):
            continue  # only False positives to guard against are "claims X doesn't use Y" when it DOES

        cocktail = observation.get("cocktail_name", "")
        ingredient = observation.get("ingredient_name", "")
        if not cocktail or not ingredient:
            continue

        cocktail_named = cocktail.lower() in final_lower
        # Heuristic: the answer both names this cocktail AND claims the
        # "doesn't use <ingredient>" property (either by asserting it
        # directly, or simply not admitting it still contains it).
        claims_excludes_ingredient = (
            f"doesn't use {ingredient.lower()}" in final_lower
            or f"does not use {ingredient.lower()}" in final_lower
            or f"without {ingredient.lower()}" in final_lower
        )
        if cocktail_named and claims_excludes_ingredient:
            return (
                f"Contradiction detected: tool observation shows {cocktail} DOES use "
                f"{ingredient} (uses_ingredient=True), but the answer claims it doesn't."
            )
    return None


class AgenticRAGRouter:
    """
    Implements the ReAct loop described in RAG_Patterns.md Section 5b/5c:
    THINK (LLM call) -> ACT (tool call, deterministic) -> OBSERVE (tool result
    fed back) -> repeat until DECIDE (final answer) or MAX_AGENT_STEPS reached.
    """

    def __init__(self, graph: CocktailKnowledgeGraph, vector_store: HybridVectorStore, api_key: str = NEBIUS_API_KEY):
        self.tools_impl = AgentTools(graph, vector_store)
        self.tool_schemas = build_tool_schemas()
        self.api_key = api_key

    def _call_llm(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        """
        Calls the LLM with a labeled Retry pattern (Human in loop.md Section 1,
        Pattern 4) for transient timeouts -- external API latency is variable
        (observed 1.3s-20s+ across live test calls, occasionally several
        consecutive timeouts under back-to-back load), and per Agent.md Section
        10d, "No retries" is failure mode #1 for observation/feedback loops:
        a single transient timeout should not kill the whole reasoning chain.
        Uses exponential backoff (2s, 4s, 8s) between retries to ride out
        short-lived rate-limiting/load spikes rather than hammering the API.
        """
        import time as _time

        payload = json.dumps({
            "model": AGENT_LLM_MODEL_NAME,
            "messages": messages,
            "tools": self.tool_schemas,
            "tool_choice": "auto",
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            NEBIUS_CHAT_URL,
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 2):  # e.g. max_retries=3 -> 4 total attempts
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as e:
                last_error = e
                if attempt <= max_retries:
                    backoff = 2 ** attempt  # 2s, 4s, 8s
                    print(f"   ⚠️  [Agentic RAG] LLM call attempt {attempt} failed ({e}); retrying in {backoff}s...")
                    _time.sleep(backoff)
        raise last_error  # exhausted retries -> surface to caller's stopping logic

    def run(self, question: str) -> Dict[str, Any]:
        """
        Executes the full ReAct loop. Returns a trace dict with the final
        answer plus every THINK/ACT/OBSERVE step, so the whole chain is
        auditable (per the citation/evidence principles in Human in loop.md).

        Includes a code-level (not just prompt-level) safeguard against the
        "Agents tend to get stuck in loops" failure mode (Agent.md Section
        "Agent Failure Modes" / Agents_Consideration.md's over-engineering
        tells): if the model repeats an IDENTICAL (tool, args) call it already
        made, we do not hit the network again -- we return the cached
        observation plus an explicit nudge telling it to stop repeating and
        either try a new candidate or conclude no answer exists.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: List[Dict[str, Any]] = []
        seen_calls: Dict[str, Dict[str, Any]] = {}  # "tool_name(args_json)" -> observation

        for step in range(1, MAX_AGENT_STEPS + 1):
            print(f"🧠 [Agentic RAG] THINK step {step}: calling LLM...")
            try:
                response = self._call_llm(messages)
            except Exception as e:
                trace.append({"step": step, "type": "ERROR", "detail": str(e)})
                return {
                    "final_answer": (
                        "I'm sorry, the agentic reasoning step timed out or failed. "
                        "Please try a simpler, single-part question. [AgenticRAG-Error]"
                    ),
                    "trace": trace,
                    "steps_used": step,
                }

            choice = response["choices"][0]["message"]
            tool_calls = choice.get("tool_calls")

            if not tool_calls:
                # DECIDE: model produced a final plain-text answer -> stop.
                final_text = choice.get("content") or "[No content returned]"

                contradiction = _find_contradiction(final_text, seen_calls)
                if contradiction:
                    print(f"🚫 [Agentic RAG] GROUNDEDNESS CHECK FAILED at step {step}: {contradiction}")
                    trace.append({"step": step, "type": "REJECTED_CONTRADICTION", "content": final_text, "reason": contradiction})
                    # Do NOT return the contradicted answer. Force the model
                    # to reconsider using its own observations, same pattern
                    # as the duplicate-call nudge above.
                    messages.append({"role": "assistant", "content": final_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"That answer is incorrect: {contradiction} Re-examine your own tool "
                            "observations and give a corrected final answer."
                        ),
                    })
                    continue  # do not consume this as the final answer -- loop again

                trace.append({"step": step, "type": "DECIDE", "content": final_text})
                print(f"✅ [Agentic RAG] DECIDE at step {step}: enough information, stopping.")
                if _NO_MATCH_REGEX.search(final_text):
                    log_missing_record(question, reason="agentic_no_match", source_node="AgenticRAGRouter.run")
                return {"final_answer": final_text, "trace": trace, "steps_used": step}

            # ACT: execute every requested tool call, deterministically.
            messages.append({
                "role": "assistant",
                "content": choice.get("content"),
                "tool_calls": tool_calls,
            })
            any_new_call_this_step = False
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"] or "{}")
                call_key = f"{fn_name}({json.dumps(fn_args, sort_keys=True)})"

                if call_key in seen_calls:
                    print(f"🔁 [Agentic RAG] DEDUP step {step}: repeated call {call_key} -> reusing cached result, not re-querying.")
                    observation = seen_calls[call_key]
                    observation = {
                        **observation,
                        "_note": "DUPLICATE_CALL_DETECTED: you already made this exact call. "
                                 "Try a DIFFERENT candidate, or if none remain, give your final answer now.",
                    }
                else:
                    print(f"🔧 [Agentic RAG] ACT step {step}: {fn_name}({fn_args})")
                    observation = self.tools_impl.dispatch(fn_name, fn_args)
                    seen_calls[call_key] = observation
                    any_new_call_this_step = True

                print(f"👀 [Agentic RAG] OBSERVE step {step}: {observation}")
                trace.append({
                    "step": step, "type": "ACT_OBSERVE",
                    "tool": fn_name, "args": fn_args, "result": observation,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(observation),
                })

            if not any_new_call_this_step:
                # Every tool call this step was a duplicate -- the model is
                # looping. Force a decision on the next turn rather than
                # burning the remaining step budget on more repeats.
                messages.append({
                    "role": "user",
                    "content": (
                        "You have repeated tool calls you already made. Do not call any tool "
                        "again with the same arguments. Based on everything observed so far, "
                        "give your final answer now, including stating clearly if no cocktail "
                        "in the manual satisfies the constraint."
                    ),
                })

        # Stopping criteria hit: max steps exceeded without a final answer.
        print(f"⏹️  [Agentic RAG] STOP: reached MAX_AGENT_STEPS={MAX_AGENT_STEPS} without a final answer.")
        return {
            "final_answer": (
                "I gathered some information but couldn't fully resolve this within "
                "my reasoning budget. Please try narrowing the question. [AgenticRAG-MaxSteps]"
            ),
            "trace": trace,
            "steps_used": MAX_AGENT_STEPS,
        }


# =====================================================================
# 4. TOP-LEVEL ENTRY POINT -- escalation-aware pipeline wrapper
# =====================================================================

class EscalatingRAGPipeline(GraphRAGPipeline):
    """
    Wraps GraphRAGPipeline (fast 3-way router: GRAPH/VECTOR/REFUSE) with a
    4th escalation branch to AgenticRAGRouter for compound/multi-hop queries.
    This is the "Escalation, Not Replacement" design from our discussion --
    the fast router remains the default path for the majority of queries.
    """

    def __init__(self, vector_store: HybridVectorStore, graph: CocktailKnowledgeGraph, api_key: str = NEBIUS_API_KEY):
        super().__init__(vector_store, graph, api_key)
        self.agent = AgenticRAGRouter(graph, vector_store, api_key)

    def run_workflow(self, question: str):
        print("\n" + "=" * 60)
        print(f"🎬 EXECUTING ESCALATING RAG WORKFLOW FOR QUERY: '{question}'")
        print("=" * 60)

        if needs_agentic_escalation(question):
            print("🚨 [Escalation Check] Compound/multi-hop query detected -> escalating to Agentic RAG.")
            result = self.agent.run(question)
            print("\n🏁 WORKFLOW COMPLETE. FINAL ANSWER OUTPUT:\n")
            print(result["final_answer"])
            print(f"\n📋 [Agentic RAG] Full trace ({result['steps_used']} step(s)):")
            for t in result["trace"]:
                print(f"   {t}")
            print("=" * 60)
            return result

        print("⚡ [Escalation Check] Single-hop query -> using fast deterministic router.")
        return super().run_workflow(question)


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

    pipeline = EscalatingRAGPipeline(vector_store, graph)

    # TEST 1: The canonical compound query from RAG_Patterns.md Section 5b
    pipeline.run_workflow(
        "I want a Negroni variant that doesn't use Campari, and tell me what glass it's served in."
    )

    # TEST 2: Single-hop query -- should stay on the FAST router, not escalate
    pipeline.run_workflow("What other cocktails are variants of the Negroni?")

    # TEST 3: Single-hop direct lookup -- should stay on the FAST router
    pipeline.run_workflow("How do I make a Margarita?")
