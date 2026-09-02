"""
ingest_and_clean.py
-------------------
Step 2 of the RAG Application Pipeline:
1. Loads raw Cocktail_Corpus.md text
2. Sanitizes & cleans whitespace, formatting artifacts, and structural tags
3. Parses individual recipes and extracts structured metadata dictionary tags
   (drink_name, category, glassware, base_spirit, flavor_profile)
4. Wraps cleaned sections in Document primitives (LangChain compatible)
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Graceful fallback to lightweight Document dataclass if langchain_core is not installed locally
try:
    from langchain_core.documents import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)


def load_raw_corpus(file_path: str) -> str:
    """Reads the raw file from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def sanitize_text(text: str) -> str:
    """Sanitizes text by removing control chars, double-whitespaces, and trailing blanks."""
    # Convert carriage returns to standard newlines
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Strip trailing whitespace on each line
    cleaned = "\n".join([line.rstrip() for line in cleaned.split("\n")])
    return cleaned.strip()


def parse_recipe_metadata(recipe_block: str) -> Dict[str, Any]:
    """Extracts structured metadata fields from a single recipe block."""
    metadata = {}
    
    # Extract Drink Name from # Heading
    name_match = re.search(r"^#\s+(.+)$", recipe_block, re.MULTILINE)
    if name_match:
        metadata["drink_name"] = name_match.group(1).strip()
    
    # Extract Category
    cat_match = re.search(r"-\s*\*\*Category:\*\*\s*(.+)$", recipe_block, re.MULTILINE)
    if cat_match:
        metadata["category"] = cat_match.group(1).strip()
        
    # Extract Glassware
    glass_match = re.search(r"-\s*\*\*Glassware:\*\*\s*(.+)$", recipe_block, re.MULTILINE)
    if glass_match:
        metadata["glassware"] = glass_match.group(1).strip()
        
    # Extract Base Spirit
    spirit_match = re.search(r"-\s*\*\*Base Spirit:\*\*\s*(.+)$", recipe_block, re.MULTILINE)
    if spirit_match:
        metadata["base_spirit"] = spirit_match.group(1).strip()
        
    # Extract Flavor Profile
    flavor_match = re.search(r"-\s*\*\*Flavor Profile:\*\*\s*(.+)$", recipe_block, re.MULTILINE)
    if flavor_match:
        metadata["flavor_profile"] = flavor_match.group(1).strip()

    return metadata


def ingest_and_clean_corpus(file_path: str) -> List[Document]:
    """Full Step 2 Pipeline: Load -> Sanitize -> Extract Metadata -> Return Document Objects."""
    print("🚀 [Step 2] Starting Ingestion & Cleaning Pipeline...")
    
    # 1. Load Raw Text
    raw_text = load_raw_corpus(file_path)
    print(f"📖 [Step 2] Loaded raw corpus ({len(raw_text)} characters)")
    
    # 2. Sanitize Text
    sanitized = sanitize_text(raw_text)
    
    # 3. Split by Recipe Separator ("---")
    raw_blocks = sanitized.split("\n---")
    
    documents: List[Document] = []
    recipe_counter = 0
    for block in raw_blocks:
        block_clean = block.strip()
        if not block_clean:
            continue
            
        # Parse metadata
        meta = parse_recipe_metadata(block_clean)
        
        # Skip top document title block if it has no recipe category/ingredients
        if "category" not in meta and "base_spirit" not in meta:
            continue
            
        recipe_counter += 1
        meta["source"] = file_path
        meta["recipe_id"] = f"RECIPE-{recipe_counter:03d}"
        
        # Create Document primitive
        doc = Document(page_content=block_clean, metadata=meta)
        documents.append(doc)
        
    print(f"✅ [Step 2] Ingestion Complete! Successfully processed {len(documents)} clean recipe documents.")
    return documents


if __name__ == "__main__":
    corpus_path = "Cocktail_Corpus.md"
    cleaned_docs = ingest_and_clean_corpus(corpus_path)
    
    print("\n" + "="*60)
    print("🔍 INSPECTING FIRST CLEANED COCKTAIL DOCUMENT:")
    print("="*60)
    print("📄 CONTENT SNIPPET:\n", cleaned_docs[0].page_content[:300], "...\n")
    print("🏷️ EXTRACTED METADATA DICTIONARY:\n")
    for k, v in cleaned_docs[0].metadata.items():
        print(f"  • {k}: {v}")
    print("="*60)
