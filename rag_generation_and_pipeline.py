"""
rag_generation_and_pipeline.py
-------------------------------
Step 6 of the RAG Application Pipeline: RAG Generation & Context Engineering
1. Assembles System Prompt + Refusal Clause + Cited Context Chunks + User Query
2. Implements a LangGraph-compatible Stateful Graph Machine:
   - State: (question, retrieved_chunks, is_cocktail_related, final_answer)
   - Nodes: retrieve_node -> grade_relevance_node -> generate_node / refuse_node
3. Enforces citation tags ([1], [2]) and strict domain refusal guardrails
"""

import sys
import re
from typing import List, Dict, Any, TypedDict, Optional

from vector_storage_and_retrieval import (
    ingest_and_clean_corpus,
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore
)


# =====================================================================
# 1. CONTEXT ENGINEERING & SYSTEM PROMPTS
# =====================================================================

SYSTEM_PROMPT = """You are an expert Mixologist & Cocktail Assistant.
Your job is to answer user mixology and recipe questions using ONLY the provided cocktail recipe context below.

STRICT GUARDRAILS:
1. Use ONLY the provided context chunks to answer.
2. Cite the exact chunk ID [1], [2], etc., for every recipe fact or ingredient ratio you provide.
3. If the user query is NOT related to cocktails/bartending, OR if the required recipe is NOT in the context, politely refuse using the refusal clause:
   "I am sorry, but I can only answer questions related to cocktail recipes, mixology techniques, and ingredients present in our manual."
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
    def __init__(self, vector_store: HybridVectorStore):
        self.vector_store = vector_store

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
        """Node 2: Evaluates query intent & retrieved chunk relevance."""
        query = state["question"].lower()
        chunks = state["retrieved_chunks"]
        
        # Strict Domain keyword check
        cocktail_keywords = [
            "drink", "cocktail", "recipe", "whiskey", "bourbon", "gin", "rum", "tequila", 
            "vodka", "mezcal", "scotch", "bitters", "vermouth", "campari", "shake", "stir",
            "glass", "garnish", "syrup", "lime", "lemon", "margarita", "negroni", "mojito",
            "daiquiri", "martini", "sazerac", "agave", "liquor", "bar", "mixology"
        ]
        
        is_relevant = any(kw in query for kw in cocktail_keywords)
        
        # Check if top retrieved chunk has a valid hybrid score threshold (> 0.25)
        if chunks and chunks[0]["hybrid_score"] < 0.25:
            is_relevant = False
            
        state["is_cocktail_query"] = is_relevant
        print(f"📊 [Node: Grade Relevance] Domain Relevance Check: {is_relevant}")
        return state

    def generate_node(self, state: RAGState) -> RAGState:
        """Node 3A: Assembles Context + System Prompt and Generates Cited Answer."""
        print("✍️ [Node: Generate] Assembling Context & Generating Cited Response...")
        
        query = state["question"]
        chunks = state["retrieved_chunks"]
        
        # Format Context with explicit [1], [2] Citation Tags
        context_blocks = []
        for idx, chunk in enumerate(chunks):
            tag = f"[{idx+1}]"
            drink_name = chunk['drink_name']
            content = chunk['page_content']
            context_blocks.append(f"{tag} ({drink_name}):\n{content}")
            
        formatted_context = "\n\n".join(context_blocks)
        
        # Synthesize Grounded Response (Simulating LLM generation with strict citations)
        top_recipe = chunks[0]
        meta = top_recipe["metadata"]
        
        answer_lines = [
            f"Here is the recipe for the **{top_recipe['drink_name']}** based on our manual [1]:\n",
            f"• **Category:** {meta.get('category', 'N/A')}",
            f"• **Glassware:** {meta.get('glassware', 'N/A')}",
            f"• **Base Spirit:** {meta.get('base_spirit', 'N/A')}",
            f"• **Flavor Profile:** {meta.get('flavor_profile', 'N/A')}\n",
            f"**Recipe & Preparation Details [1]:**\n{top_recipe['page_content']}"
        ]
        
        if len(chunks) > 1:
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
        return state

    def run_workflow(self, question: str) -> RAGState:
        """Executes the LangGraph State Machine Flowchart."""
        print("\n" + "="*60)
        print(f"🎬 EXECUTING RAG WORKFLOW FOR QUERY: '{question}'")
        print("="*60)
        
        # Initial State
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
    # Setup full end-to-end pipeline (Steps 2 -> 3 -> 4 -> 5 -> 6)
    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(cleaned_docs)
    embedder = CocktailEmbedder(model_name="text-embedding-3-small", dimensions=1536)
    records = generate_embeddings_for_chunks(chunks, embedder)
    vector_store = HybridVectorStore(records, embedder)
    
    # Initialize Step 6 Pipeline
    pipeline = CocktailRAGPipeline(vector_store)
    
    # TEST 1: Valid Cocktail Recipe Query
    pipeline.run_workflow("How do I make a classic Old Fashioned?")
    
    # TEST 2: Out-of-Domain Non-Cocktail Query (Refusal Test)
    pipeline.run_workflow("What is the capital city of France?")
