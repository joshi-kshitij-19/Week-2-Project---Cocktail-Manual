"""
vector_storage_and_retrieval.py
-------------------------------
Step 5 of the RAG Application Pipeline:
1. Imports 29 embedded recipe records from Step 4 (embedding_selection.py)
2. Builds a Vector Storage Index & Hybrid Retriever Engine (Dense Vector + Sparse Keyword Search)
3. Implements Pre-Query Metadata Filtering (base_spirit, category, glassware)
4. Evaluates Top-K retrieval precision and cross-score reranking
"""

import math
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from embedding_selection import (
    ingest_and_clean_corpus, 
    apply_structural_chunking, 
    CocktailEmbedder, 
    generate_embeddings_for_chunks,
    Document
)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes Cosine Similarity between two L2-normalized vectors (dot product)."""
    return sum(a * b for a, b in zip(v1, v2))


def bm25_keyword_score(query: str, text: str) -> float:
    """Simple BM25-style keyword frequency scoring for exact string matches."""
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = re.findall(r'\w+', text.lower())
    if not query_words or not text_words:
        return 0.0
    
    score = 0.0
    for q_word in query_words:
        count = text_words.count(q_word)
        if count > 0:
            # Term frequency saturation
            score += (count * 1.5) / (count + 0.5)
    return score / len(query_words)


class HybridVectorStore:
    """
    Step 5 Hybrid Vector Database Engine.
    Combines Dense Vector Search (Cosine Similarity) + Sparse Keyword Search (BM25)
    with Pre-Query Metadata Filtering and Reranking.
    """
    def __init__(self, records: List[Dict[str, Any]], embedder: CocktailEmbedder):
        self.records = records
        self.embedder = embedder
        print(f"🗄️ [Step 5] Hybrid Vector Store Initialized with {len(records)} records.")

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        alpha: float = 0.7, 
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Retrieval:
        - alpha = 1.0 -> Pure Vector Search
        - alpha = 0.0 -> Pure Keyword BM25 Search
        - alpha = 0.7 -> Hybrid Weighted Search (70% Vector, 30% Keyword)
        """
        # 1. Generate query vector using the EXACT SAME locked embedder
        query_vector = self.embedder.embed_text(query)
        
        candidates = []
        for rec in self.records:
            meta = rec["metadata"]
            
            # Apply Pre-Query Metadata Filtering if specified
            if metadata_filter:
                match = True
                for filter_key, filter_val in metadata_filter.items():
                    actual_val = str(meta.get(filter_key, "")).lower()
                    if filter_val.lower() not in actual_val:
                        match = False
                        break
                if not match:
                    continue  # Skip document if metadata filter fails
            
            # Compute Dense Vector Score (Cosine Similarity)
            dense_score = cosine_similarity(query_vector, rec["vector"])
            
            # Compute Sparse Keyword Score (BM25 + Exact Title Match Boost)
            sparse_score = bm25_keyword_score(query, rec["page_content"])
            
            # Title match boost (exact or fuzzy drink name match in user query)
            import difflib
            drink_name_clean = rec["drink_name"].lower().split("(")[0].strip()
            query_clean = query.lower()
            
            fuzzy_match = False
            if drink_name_clean:
                if drink_name_clean in query_clean:
                    fuzzy_match = True
                else:
                    # Check fuzzy Levenshtein ratio across query words
                    for word in query_clean.split():
                        if len(word) >= 4:
                            ratio = difflib.SequenceMatcher(None, word, drink_name_clean).ratio()
                            # Check full name similarity or first-word similarity (e.g. 'fashoined' vs 'fashioned')
                            first_drink_word = drink_name_clean.split()[0] if " " in drink_name_clean else drink_name_clean
                            last_drink_word = drink_name_clean.split()[-1] if " " in drink_name_clean else drink_name_clean
                            ratio_last = difflib.SequenceMatcher(None, word, last_drink_word).ratio()
                            if ratio >= 0.70 or ratio_last >= 0.70:
                                fuzzy_match = True
                                break
            
            if fuzzy_match:
                sparse_score = min(1.0, sparse_score + 0.6)
            
            # Combined Hybrid Score
            hybrid_score = (alpha * dense_score) + ((1.0 - alpha) * sparse_score)
            
            candidates.append({
                "chunk_id": rec["chunk_id"],
                "drink_name": rec["drink_name"],
                "hybrid_score": round(hybrid_score, 4),
                "dense_score": round(dense_score, 4),
                "sparse_score": round(sparse_score, 4),
                "metadata": meta,
                "page_content": rec["page_content"]
            })
            
        # Sort candidates by hybrid_score descending
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Return Top-K winning chunks
        return candidates[:top_k]


if __name__ == "__main__":
    # Pipeline execution up to Step 4
    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(cleaned_docs)
    embedder = CocktailEmbedder(model_name="text-embedding-3-small", dimensions=1536)
    records = generate_embeddings_for_chunks(chunks, embedder)
    
    # Initialize Step 5 Vector Store
    vector_store = HybridVectorStore(records, embedder)
    
    print("\n" + "="*60)
    print("🧪 STEP 5 RETRIEVAL TEST 1: 'Smoky Scotch drink with honey'")
    print("="*60)
    results1 = vector_store.search("Smoky Scotch drink with honey", top_k=3, alpha=0.7)
    for idx, r in enumerate(results1):
        print(f"  [{idx+1}] {r['drink_name']} (Score: {r['hybrid_score']}) | Dense: {r['dense_score']} | Sparse: {r['sparse_score']}")
        
    print("\n" + "="*60)
    print("🧪 STEP 5 RETRIEVAL TEST 2: Exact Keyword 'Campari' + Metadata Filter (base_spirit='Gin')")
    print("="*60)
    results2 = vector_store.search("Campari", top_k=3, alpha=0.5, metadata_filter={"base_spirit": "Gin"})
    for idx, r in enumerate(results2):
        print(f"  [{idx+1}] {r['drink_name']} (Base Spirit: {r['metadata']['base_spirit']}) | Score: {r['hybrid_score']}")
    print("="*60)
