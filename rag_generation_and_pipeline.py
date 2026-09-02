"""
rag_generation_and_pipeline.py
-------------------------------
Step 6 of the RAG Application Pipeline: RAG Generation & Context Engineering
1. Assembles System Prompt + Refusal Clause + Cited Context Chunks + User Query
2. Connects to Live Nebius Studio API (meta-llama/Llama-3.3-70B-Instruct) for grounded generation
3. Implements a LangGraph-compatible Stateful Graph Machine:
   - State: (question, retrieved_chunks, is_cocktail_related, final_answer)
   - Nodes: retrieve_node -> grade_relevance_node -> generate_node / refuse_node
4. Enforces citation tags ([1], [2]) and strict domain refusal guardrails
"""

import os
import sys
import re
import json
import urllib.request
from typing import List, Dict, Any, TypedDict, Optional

from vector_storage_and_retrieval import (
    ingest_and_clean_corpus,
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore
)
from missing_record_logger import log_missing_record, REFUSAL_CLAUSE

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_CHAT_URL = "https://api.studio.nebius.ai/v1/chat/completions"
LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# Threshold used ONLY by the structured (non-LLM) fallback path in
# generate_node to decide "is chunks[0] a real match, or just the least-bad
# option?" -- deliberately stricter than grade_relevance_node's 0.25 keyword-
# gating threshold, which answers a different question ("is this vaguely
# cocktail-related at all?"). Calibrated from live observed scores: real
# direct-lookup matches score ~0.72-0.80; the "Americano" (missing recipe)
# case scored ~0.35 -- comfortably below this threshold.
LOW_CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------
# Deterministic Ingredient Grounding Vocabulary (Fix A)
# ---------------------------------------------------------------------
# Real bug found live: query "what can I make with Vodka, orange juice"
# returned the Mai Tai (rum-based, no vodka or OJ at all) because it scored
# highest on hybrid vector similarity -- despite the Screwdriver (vodka +
# orange juice, exact match) existing right there in the corpus. Neither
# grade_relevance_node()'s keyword gate nor the LLM's own judgment caught
# this, so the fix is a deterministic literal-text check (same philosophy
# as AgenticRAGRouter's _find_contradiction() -- prompt instructions alone
# were not enough).
#
# Manually curated from the current 30-recipe Cocktail_Corpus.md -- mirrors
# the existing cocktail_keywords list in grade_relevance_node(), but scoped
# specifically to spirits/mixers/ingredients (not drink names or verbs), so
# it can answer a different question: "did the user name an ingredient we
# can literally verify against the retrieved recipe's text?" Extend this set
# if the corpus grows to include new spirits/mixers.
INGREDIENT_VOCAB = [
    # Spirits
    "vodka", "gin", "rum", "tequila", "whiskey", "bourbon", "rye", "scotch",
    "mezcal", "cognac", "pisco", "aperol", "campari", "vermouth", "kahlua",
    "cointreau", "triple sec", "curacao", "chartreuse", "maraschino",
    "amaro", "absinthe", "herbsaint", "prosecco", "champagne",
    # Mixers / Juices / Other Ingredients
    "orange juice", "lime juice", "lemon juice", "cranberry juice",
    "pineapple juice", "grapefruit", "ginger beer", "soda water", "tonic",
    "espresso", "honey", "simple syrup", "agave", "egg white", "mint",
    "bitters", "orgeat",
]


def extract_ingredient_terms(text: str) -> List[str]:
    """Returns the subset of INGREDIENT_VOCAB terms named (whole-word/phrase
    match, case-insensitive) in the given text."""
    text_lower = text.lower()
    return [
        term for term in INGREDIENT_VOCAB
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower)
    ]


# =====================================================================
# 1. CONTEXT ENGINEERING & SYSTEM PROMPTS
# =====================================================================

SYSTEM_PROMPT = """You are an expert Mixologist & Cocktail Assistant.
Your job is to answer user mixology and recipe questions using ONLY the provided cocktail recipe context below.

STRICT GUARDRAILS:
1. Use ONLY the provided context chunks to answer.
2. Cite the exact chunk ID tag [1], [2], etc., for every recipe fact, ingredient ratio, or instruction you provide.
3. If the user query is NOT related to cocktails/bartending, OR if the required recipe is NOT present in the context, politely refuse using the refusal clause:
   "I am sorry, but I can only answer questions related to cocktail recipes, mixology techniques, and ingredients present in our cocktail manual."
4. Before answering, verify that every specific spirit or ingredient the user named (e.g. "vodka", "orange juice", "Campari") actually appears in the retrieved recipe's ingredient list below. If the retrieved recipe does NOT contain an ingredient the user explicitly asked for, do NOT substitute the closest-sounding recipe -- refuse using the refusal clause instead.
"""


# =====================================================================
# 2. LANGGRAPH STATE DEFINITION
# =====================================================================

class RAGState(TypedDict):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    is_cocktail_query: bool
    final_answer: str


# =====================================================================
# 3. LANGGRAPH NODES & DECISION EDGES
# =====================================================================

class CocktailRAGPipeline:
    def __init__(self, vector_store: HybridVectorStore, api_key: str = NEBIUS_API_KEY):
        self.vector_store = vector_store
        self.api_key = api_key

    def retrieve_node(self, state: RAGState) -> RAGState:
        """Node 1: Retrieves Top-K chunks from Hybrid Vector Store."""
        query = state["question"]
        print(f"\n🔍 [Node: Retrieve] Searching Hybrid Vector DB for query: '{query}'")
        
        # Search Top-3 chunks using Hybrid alpha = 0.7
        chunks = self.vector_store.search(query, top_k=3, alpha=0.7)
        state["retrieved_chunks"] = chunks
        print(f"   ↳ Retrieved {len(chunks)} candidate chunks.")
        return state

    def grade_relevance_node(self, state: RAGState) -> RAGState:
        """
        Node 2: Evaluates query intent & retrieved chunk relevance.
        Combines hybrid similarity score threshold + cocktail domain keywords.
        """
        query = state["question"].lower()
        chunks = state["retrieved_chunks"]
        
        # Domain keyword check using whole word boundary matching
        cocktail_keywords = [
            "drink", "drinks", "cocktail", "cocktails", "recipe", "recipes", "ingredient", "ingredients",
            "whiskey", "bourbon", "gin", "rum", "tequila", "vodka", "mezcal", "scotch", "bitters", "vermouth",
            "campari", "shake", "stir", "glass", "garnish", "syrup", "lime", "lemon", "margarita", "negroni",
            "mojito", "daiquiri", "martini", "sazerac", "agave", "liquor", "bar", "mixology", "old fashioned",
            "manhattan", "paloma", "screwdriver", "make"
        ]
        
        has_keyword = any(re.search(r'\b' + re.escape(kw) + r'\b', query) for kw in cocktail_keywords)
        has_high_confidence = bool(chunks and chunks[0]["hybrid_score"] >= 0.25)
        
        # Pass if query has domain keywords OR vector search retrieved a high-confidence match
        is_relevant = has_keyword or has_high_confidence
            
        state["is_cocktail_query"] = is_relevant
        print(f"📊 [Node: Grade Relevance] Domain Relevance Check: {is_relevant} (Keyword Match: {has_keyword}, Vector Score: {chunks[0]['hybrid_score'] if chunks else 'N/A'})")
        return state

    def _verify_ingredient_grounding(self, query: str, chunks: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Deterministic groundedness check (Fix A) -- mirrors AgenticRAGRouter's
        _find_contradiction() pattern in agentic_rag_pipeline.py: prompt
        instructions alone were not enough to stop the LLM from confidently
        answering with an unrelated recipe (e.g. the Mai Tai for a "vodka,
        orange juice" query).

        ALL-OR-NOTHING RULE: if the user names 2+ ingredients, the returned
        recipe must contain EVERY one of them -- never just the best partial
        match. (Fixed after live testing showed "vodka and pickle juice"
        wrongly returned the Moscow Mule -- it has vodka, but no pickle
        juice, and a partial match should never be presented as an answer.)

        1. Extracts any named spirits/ingredients from the user's query.
        2. Checks the already-retrieved top-k chunks for a FULL match (every
           requested term present).
        3. If none of the top-k chunks fully match, falls back to a full
           -corpus exact-term scan as a safety net against ranking misses
           (this is what actually finds the Screwdriver when Mai Tai wins
           the vector/BM25 ranking).
        4. Returns `None` if the query named no ingredients at all (check
           doesn't apply -- e.g. "How do I make an Americano?" names a
           drink, not an ingredient -- existing confidence-based path
           handles that case unchanged).
        Returns `[]` to signal "query named ingredients, but no single
        recipe anywhere in the corpus contains ALL of them" -- caller must
        refuse rather than answer with a partial match.
        """
        requested_terms = extract_ingredient_terms(query)
        if not requested_terms:
            return None  # Not an ingredient-specific query -- check doesn't apply

        def overlap(text: str) -> int:
            text_lower = text.lower()
            return sum(1 for t in requested_terms if re.search(r'\b' + re.escape(t) + r'\b', text_lower))

        needed = len(requested_terms)

        # 1. Check top-k first (cheap, preserves existing ranking/order).
        for c in chunks:
            if overlap(c["page_content"]) == needed:
                return [c] + [x for x in chunks if x["chunk_id"] != c["chunk_id"]]

        # 2. Top-k had no full match -- scan the FULL corpus as a safety net
        #    against ranking misses (this is what finds the Screwdriver when
        #    Mai Tai wins the vector/BM25 ranking).
        for rec in self.vector_store.records:
            if overlap(rec["page_content"]) == needed:
                print(f"🛡️ [Ingredient Grounding] Vector/BM25 ranking missed a full match on "
                      f"{requested_terms} -- promoting '{rec['drink_name']}' via full-corpus scan.")
                # NOTE: `rec` is a raw Step-4 embedding record (chunk_id/
                # drink_name/page_content/metadata/vector) -- NOT a
                # search-result dict. Every downstream consumer
                # (generate_node's structured fallback, missing-record
                # logging) expects the search-result shape with
                # hybrid_score/dense_score/sparse_score present, so we
                # must re-wrap it here. Found via live execution --
                # returning `rec` as-is raised a KeyError on the very
                # first live test run.
                promoted = {
                    "chunk_id": rec["chunk_id"],
                    "drink_name": rec["drink_name"],
                    "hybrid_score": 1.0,  # exact full-term match, full confidence
                    "dense_score": None,
                    "sparse_score": None,
                    "metadata": rec["metadata"],
                    "page_content": rec["page_content"],
                }
                rest = [c for c in chunks if c["chunk_id"] != rec["chunk_id"]]
                return [promoted] + rest

        # 3. No recipe anywhere contains ALL requested ingredients -- this is
        #    a true corpus gap. Do NOT return a partial match (e.g. a
        #    vodka-only drink for "vodka and pickle juice").
        return []

    def _call_nebius_llm(self, user_query: str, formatted_context: str) -> str:
        """Calls Live Nebius Studio Chat API (Llama-3.3-70B-Instruct)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT RECIPES:\n{formatted_context}\n\nUSER QUESTION:\n{user_query}"}
        ]
        payload = json.dumps({
            "model": LLM_MODEL_NAME,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 300
        }).encode("utf-8")
        
        req = urllib.request.Request(
            NEBIUS_CHAT_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]

    def generate_node(self, state: RAGState) -> RAGState:
        """Node 3A: Assembles Context + System Prompt and Generates Cited Answer."""
        print(f"✍️ [Node: Generate] Calling Live Nebius LLM ({LLM_MODEL_NAME}) for cited generation...")
        
        query = state["question"]
        chunks = state["retrieved_chunks"]

        # --- Fix A: Deterministic Ingredient Grounding Check ---
        # Real bug found live: "what can I make with Vodka, orange juice"
        # returned the Mai Tai (no vodka, no OJ) even though the Screwdriver
        # (exact match) exists in the corpus, because vector/BM25 ranking
        # scored Mai Tai higher. This overrides that ranking with a literal
        # ingredient-presence check before any generation happens.
        grounded_chunks = self._verify_ingredient_grounding(query, chunks)
        if grounded_chunks == []:
            # Query named specific ingredients, but NOTHING in the entire
            # corpus contains them -- a true corpus gap. Refuse honestly
            # instead of letting ranking noise pick an unrelated "closest" recipe.
            top_score = chunks[0]["hybrid_score"] if chunks else None
            log_missing_record(query, reason="ingredient_grounding_no_match", top_score=top_score, source_node="generate_node")
            state["final_answer"] = REFUSAL_CLAUSE
            return state
        elif grounded_chunks:
            # Top-k was already correctly ordered, or a better match was
            # found and promoted from a full-corpus scan -- use it.
            chunks = grounded_chunks
            state["retrieved_chunks"] = chunks
        # else grounded_chunks is None -> query named no specific ingredients;
        # grounding check doesn't apply, fall through unchanged (handled by
        # the existing confidence-based path below, e.g. "How do I make an
        # Americano?").

        # Format Context with explicit [1], [2], [3] Citation Tags
        context_blocks = []
        for idx, chunk in enumerate(chunks):
            tag = f"[{idx+1}]"
            drink_name = chunk['drink_name']
            content = chunk['page_content']
            context_blocks.append(f"{tag} ({drink_name}):\n{content}")
            
        formatted_context = "\n\n".join(context_blocks)
        
        if self.api_key and "v1." in self.api_key:
            try:
                llm_response = self._call_nebius_llm(query, formatted_context)
                state["final_answer"] = llm_response
                if REFUSAL_CLAUSE in llm_response:
                    # The LLM itself decided the retrieved chunks don't
                    # actually answer this question, even though
                    # grade_relevance_node let it through as "relevant"
                    # (e.g. it has cocktail keywords but no matching recipe
                    # exists in the corpus -- the "Americano" case).
                    top_score = chunks[0]["hybrid_score"] if chunks else None
                    log_missing_record(query, reason="llm_self_refusal", top_score=top_score, source_node="generate_node")
                return state
            except Exception as e:
                print(f"⚠️ [Nebius LLM Fallback] {e}. Using structured output generator.")

        # Structured Fallback
        top_score = chunks[0]["hybrid_score"] if chunks else 0.0
        if not chunks or top_score < LOW_CONFIDENCE_THRESHOLD:
            # No chunk is a real match (e.g. LLM timed out on an out-of-corpus
            # query like "Americano") -- refuse honestly instead of returning
            # an unrelated recipe, and log the gap exactly like refuse_node does.
            log_missing_record(query, reason="structured_fallback_low_confidence", top_score=top_score, source_node="generate_node")
            state["final_answer"] = REFUSAL_CLAUSE
            return state

        query_lower = query.lower()
        matched_chunk = chunks[0]
        for c in chunks:
            if c["drink_name"].lower() in query_lower:
                matched_chunk = c
                break
                
        meta = matched_chunk["metadata"]
        answer_lines = [
            f"Here is the recipe for the **{matched_chunk['drink_name']}** based on our manual [1]:\n",
            f"• **Category:** {meta.get('category', 'N/A')}",
            f"• **Glassware:** {meta.get('glassware', 'N/A')}",
            f"• **Base Spirit:** {meta.get('base_spirit', 'N/A')}",
            f"• **Flavor Profile:** {meta.get('flavor_profile', 'N/A')}\n",
            f"**Recipe & Preparation Details [1]:**\n{matched_chunk['page_content']}"
        ]
        if len(chunks) > 1 and chunks[1]["drink_name"] != matched_chunk["drink_name"]:
            answer_lines.append(f"\n*Alternative related recommendation:* You might also enjoy the **{chunks[1]['drink_name']}** [2].")

        state["final_answer"] = "\n".join(answer_lines)
        return state

    def refuse_node(self, state: RAGState) -> RAGState:
        """Node 3B: Refusal Guardrail Triggered."""
        print("🛡️ [Node: Refuse] Triggering Refusal Guardrail...")
        state["final_answer"] = (
            "I am sorry, but I can only answer questions related to cocktail recipes, "
            "mixology techniques, and ingredients present in our cocktail manual."
        )
        top_score = state["retrieved_chunks"][0]["hybrid_score"] if state.get("retrieved_chunks") else None
        log_missing_record(state["question"], reason="refuse_node", top_score=top_score, source_node="refuse_node")
        return state

    def run_workflow(self, question: str) -> RAGState:
        """Executes the LangGraph State Machine Flowchart."""
        print("\n" + "="*60)
        print(f"🎬 EXECUTING RAG WORKFLOW FOR QUERY: '{question}'")
        print("="*60)
        
        state: RAGState = {
            "question": question,
            "retrieved_chunks": [],
            "is_cocktail_query": False,
            "final_answer": ""
        }
        
        # 1. Execute Retrieve Node
        state = self.retrieve_node(state)
        
        # 2. Execute Grade Relevance Node
        state = self.grade_relevance_node(state)
        
        # 3. Conditional Edge Routing
        if state["is_cocktail_query"]:
            state = self.generate_node(state)
        else:
            state = self.refuse_node(state)
            
        print("\n🏁 WORKFLOW COMPLETE. FINAL ANSWER OUTPUT:\n")
        print(state["final_answer"])
        print("="*60)
        return state


if __name__ == "__main__":
    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(cleaned_docs)
    embedder = CocktailEmbedder(model_name="Qwen/Qwen3-Embedding-8B", dimensions=4096)
    records = generate_embeddings_for_chunks(chunks, embedder)
    vector_store = HybridVectorStore(records, embedder)
    
    pipeline = CocktailRAGPipeline(vector_store)
    
    # TEST 1: Valid Cocktail Recipe Query
    pipeline.run_workflow("Can you tell me what I need to make Old Fashoined")
    
    # TEST 2: Out-of-Domain Non-Cocktail Query (Refusal Test)
    pipeline.run_workflow("What is the capital city of France?")
