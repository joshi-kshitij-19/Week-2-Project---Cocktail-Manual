"""
run_demo_showcase.py
--------------------
Automated Showcase Demo Script for Project 1: Cocktail RAG Assistant
Runs 4 distinct test scenarios using live Nebius Cloud API (Qwen 4096D Embeddings + Llama 3.3 70B LLM).
"""

from rag_generation_and_pipeline import (
    ingest_and_clean_corpus,
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore,
    CocktailRAGPipeline
)


def run_showcase():
    print("\n" + "🍹"*35)
    print("      PROJECT 1: COCKTAIL RAG ASSISTANT DEMO SHOWCASE (LIVE NEBIUS API)")
    print("🍹"*35 + "\n")
    
    # Initialize Pipeline with Live Nebius Models
    docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(docs)
    embedder = CocktailEmbedder(model_name="Qwen/Qwen3-Embedding-8B", dimensions=4096)
    records = generate_embeddings_for_chunks(chunks, embedder)
    vector_store = HybridVectorStore(records, embedder)
    pipeline = CocktailRAGPipeline(vector_store)
    
    scenarios = [
        ("SCENARIO 1: Direct Recipe Lookup (with Typo)", "Can you tell me what I need to make Old Fashoined?"),
        ("SCENARIO 2: Semantic Intent Match (No Exact Drink Name)", "What is a smoky agave drink with lime and spice?"),
        ("SCENARIO 3: Exact Brand Keyword Match", "Find recipes using Campari and Sweet Vermouth"),
        ("SCENARIO 4: Out-of-Domain Refusal Guardrail Test", "Can you explain how black holes work in physics?")
    ]
    
    for title, query in scenarios:
        print("\n" + "📌 "*25)
        print(f"  {title}")
        print("📌 "*25)
        pipeline.run_workflow(query)
        print("\n")


if __name__ == "__main__":
    run_showcase()
