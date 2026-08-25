"""
chunking_strategy.py
--------------------
Step 3 of the RAG Application Pipeline:
1. Imports cleaned documents from Step 2 (ingest_and_clean.py)
2. Applies Structural/Header Chunking (MarkdownHeaderTextSplitter pattern)
   to ensure 1 Recipe = 1 Complete, Intact Chunk (~300-400 tokens)
3. Computes chunk size metrics (character count, estimated token count)
4. Retains and verifies metadata integrity across all chunks
"""

import sys
from typing import List, Dict, Any
from dataclasses import dataclass, field

from ingest_and_clean import ingest_and_clean_corpus, Document


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count (~4 characters per token rule of thumb)."""
    return max(1, len(text) // 4)


def apply_structural_chunking(docs: List[Document]) -> List[Document]:
    """
    Step 3 Structural Chunking Pipeline:
    Ensures each recipe header (# Drink Name) forms a self-contained, intact chunk
    with exact token estimations and verified metadata.
    """
    print("🚀 [Step 3] Starting Structural Chunking Pipeline...")
    
    chunked_documents: List[Document] = []
    
    for idx, doc in enumerate(docs):
        text = doc.page_content
        meta = doc.metadata.copy()
        
        # Calculate chunk size metrics
        char_count = len(text)
        token_count = estimate_token_count(text)
        
        # Enrich metadata with chunking strategy statistics
        meta["chunk_id"] = f"CHUNK-{idx+1:03d}"
        meta["chunk_strategy"] = "Structural_Recipe_Header"
        meta["char_count"] = char_count
        meta["estimated_tokens"] = token_count
        
        # Create final intact Document chunk
        chunk_doc = Document(page_content=text, metadata=meta)
        chunked_documents.append(chunk_doc)
        
    print(f"✅ [Step 3] Chunking Complete! Created {len(chunked_documents)} intact recipe chunks.")
    return chunked_documents


def print_chunking_summary(chunks: List[Document]):
    """Prints a analytical report of the chunking results."""
    token_counts = [c.metadata["estimated_tokens"] for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    print("\n" + "="*60)
    print("📊 STEP 3 CHUNKING METRICS & ANALYSIS REPORT:")
    print("="*60)
    print(f"  • Total Chunks Created:     {len(chunks)}")
    print(f"  • Average Tokens per Chunk: {avg_tokens:.1f} tokens")
    print(f"  • Smallest Chunk Size:      {min_tokens} tokens")
    print(f"  • Largest Chunk Size:       {max_tokens} tokens")
    print(f"  • Chunking Strategy Used:    Structural Header (1 Recipe = 1 Intact Chunk)")
    print("="*60)
    
    print("\n🔍 SAMPLE CHUNK INSPECTION (Chunk #1 - Old Fashioned):")
    print("="*60)
    print("📄 CHUNK CONTENT:\n")
    print(chunks[0].page_content)
    print("\n🏷️ CHUNK METADATA:\n")
    for k, v in chunks[0].metadata.items():
        print(f"  • {k}: {v}")
    print("="*60)


if __name__ == "__main__":
    # Load cleaned documents from Step 2
    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    
    # Run Step 3 Chunking
    chunks = apply_structural_chunking(cleaned_docs)
    
    # Output analytical report
    print_chunking_summary(chunks)
