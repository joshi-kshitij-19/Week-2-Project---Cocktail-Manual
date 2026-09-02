"""
demo_app.py
-----------
Interactive Command-Line Demo App for Project 1: Cocktail RAG Assistant
Powered by Live Nebius Cloud API (Qwen 4096D Embeddings + Llama 3.3 70B LLM).

Phase 2: Now uses ConversationalRAGPipeline (short_term_memory.py), which adds
Graph RAG, Agentic RAG escalation, AND Short-Term (multi-turn) Memory on top
of the original Step 6 pipeline -- so follow-ups like "make it less sweet"
correctly resolve against the previous turn's cocktail within one session.
"""

import sys
import os
import time

from ingest_and_clean import ingest_and_clean_corpus
from vector_storage_and_retrieval import (
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore
)
from graph_builder import build_graph_from_documents
from short_term_memory import ConversationalRAGPipeline


def run_interactive_demo():
    print("\n" + "🍹"*30)
    print("  WELCOME TO THE COCKTAIL MANUAL RAG ASSISTANT DEMO")
    print("  (Powered by Live Nebius AI Studio Cloud API)")
    print("🍹"*30 + "\n")
    
    print("⏳ Initializing RAG Pipeline & Building Vector Index via Nebius API...")
    start_time = time.time()
    
    # 1. Ingestion & Cleaning
    docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    
    # 2. Structural Chunking
    chunks = apply_structural_chunking(docs)
    
    # 3. Vector Embeddings
    embedder = CocktailEmbedder(model_name="Qwen/Qwen3-Embedding-8B", dimensions=4096)
    records = generate_embeddings_for_chunks(chunks, embedder)
    
    # 4. Hybrid Vector Store
    vector_store = HybridVectorStore(records, embedder)

    # 5. Knowledge Graph (Phase 2: Graph RAG)
    graph = build_graph_from_documents(docs)

    # 6. Conversational Pipeline (Phase 2: Graph RAG + Agentic RAG + Short-Term Memory)
    pipeline = ConversationalRAGPipeline(vector_store, graph)
    
    init_duration = time.time() - start_time
    print(f"\n⚡ Live Nebius Pipeline Initialized in {init_duration:.2f}s! Ready for queries.\n")
    print("="*70)
    print("💡 TRY THESE DEMO PRESETS OR TYPE YOUR OWN QUESTION:")
    print("  1. 'How do I make a Margarita?' (Direct Recipe Lookup)")
    print("  2. 'Make it less sweet' (Follow-up -- tests Short-Term Memory)")
    print("  3. 'What is a smoky agave drink with lime?' (Semantic Intent Search)")
    print("  4. 'Show me drinks with Campari' (Exact Brand Keyword Search)")
    print("  5. 'What other cocktails are variants of the Negroni?' (Graph RAG)")
    print("  6. 'What is the capital of France?' (Refusal Guardrail Test)")
    print("  Type 'exit' or 'quit' to end the demo.")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("🍸 Enter your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Thank you for using the Cocktail RAG Assistant Demo!")
                break
                
            print("\n" + "─"*70)
            print(f"❓ QUERY: '{user_input}'")
            print("─"*70)
            
            # Run Conversational Workflow (chat_history persists automatically
            # across turns inside `pipeline` for the rest of this session)
            state = pipeline.run_workflow(user_input)
            
            print("\n" + "─"*70 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Demo Session Ended.")
            break


if __name__ == "__main__":
    run_interactive_demo()
