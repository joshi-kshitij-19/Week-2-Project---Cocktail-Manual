"""
run_custom_queries.py
---------------------
Executes live RAG queries through the Cocktail RAG Pipeline.
"""

from rag_generation_and_pipeline import (
    ingest_and_clean_corpus,
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore,
    CocktailRAGPipeline
)


def run_custom_queries():
    # Initialize Pipeline
    docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(docs)
    embedder = CocktailEmbedder(model_name="text-embedding-3-small", dimensions=1536)
    records = generate_embeddings_for_chunks(chunks, embedder)
    vector_store = HybridVectorStore(records, embedder)
    pipeline = CocktailRAGPipeline(vector_store)
    
    test_queries = [
        "What cocktail uses Green Chartreuse?",
        "Give me a refreshing rum drink with mint for summer",
        "How do I make an Espresso Martini?",
        "Who is the CEO of Apple?"  # Refusal test
    ]
    
    for q in test_queries:
        pipeline.run_workflow(q)


if __name__ == "__main__":
    run_custom_queries()
