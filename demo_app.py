"""
demo_app.py
-----------
Interactive Command-Line Demo App for Project 1: Cocktail RAG Assistant

Features:
1. Initializes full RAG pipeline (Ingestion -> Chunking -> Embedding -> Hybrid Vector Store)
2. Interactive terminal prompt where you can type any question
3. Displays step-by-step LangGraph execution trace (Nodes visited)
4. Shows Top-3 retrieved recipe chunks with hybrid similarity scores and metadata
5. Outputs final cited response or refusal guardrail trigger
"""

import sys
import os
import time

from rag_generation_and_pipeline import (
    ingest_and_clean_corpus,
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore,
    CocktailRAGPipeline
)


def run_interactive_demo():
    print("\n" + "🍹"*30)
    print("  WELCOME TO THE COCKTAIL MANUAL RAG ASSISTANT DEMO")
    print("🍹"*30 + "\n")
    
    print("⏳ Initializing RAG Pipeline & Building Vector Index...")
    start_time = time.time()
    
    # 1. Ingestion & Cleaning
    docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    
    # 2. Structural Chunking
    chunks = apply_structural_chunking(docs)
    
    # 3. Vector Embeddings
    embedder = CocktailEmbedder(model_name="text-embedding-3-small", dimensions=1536)
    records = generate_embeddings_for_chunks(chunks, embedder)
    
    # 4. Hybrid Vector Store
    vector_store = HybridVectorStore(records, embedder)
    
    # 5. LangGraph Pipeline
    pipeline = CocktailRAGPipeline(vector_store)
    
    init_duration = time.time() - start_time
    print(f"\n⚡ Pipeline Initialized in {init_duration:.2f}s! Ready for queries.\n")
    print("="*70)
    print("💡 TRY THESE DEMO PRESETS OR TYPE YOUR OWN QUESTION:")
    print("  1. 'How do I make a Margarita?' (Direct Recipe Lookup)")
    print("  2. 'What is a smoky agave drink with lime?' (Semantic Intent Search)")
    print("  3. 'Show me drinks with Campari' (Exact Brand Keyword Search)")
    print("  4. 'What is the capital of France?' (Refusal Guardrail Test)")
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
            
            # Run LangGraph Workflow
            state = pipeline.run_workflow(user_input)
            
            print("\n" + "─"*70 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Demo Session Ended.")
            break


if __name__ == "__main__":
    run_interactive_demo()
