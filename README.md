# Cocktail Manual RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, and **LangGraph**. It ingests a curated mixology corpus, indexes recipe chunks using hybrid search (dense vectors + sparse BM25), and runs a stateful graph pipeline to answer recipe queries with citations or trigger domain refusal guardrails.

---

## 🏗️ Architecture

```
[ Cocktail_Corpus.md ] ──► [ Step 2: Ingestion & Cleaning ] ──► Cleaned Docs (30)
                                                                    │
[ Step 5: Hybrid Vector Store ] ◄── [ Step 4: 1536D Embeddings ] ◄── [ Step 3: Structural Chunker ]
  (Dense Cosine + Sparse BM25)         (text-embedding-3-small)            (Intact Recipe Chunks)
            │
            ▼
[ Step 6: LangGraph State Machine ] ──► [ Cited Recipe Output / Refusal Guardrail ]
```

---

## 📂 Repository Structure

| File | Purpose |
| :--- | :--- |
| **`Cocktail_Corpus.md`** | Curated knowledge base containing 30 standardized cocktail recipes. |
| **`ingest_and_clean.py`** | **Step 2:** Loads raw Markdown, sanitizes text, and extracts metadata (`base_spirit`, `glassware`, `category`). |
| **`chunking_strategy.py`** | **Step 3:** Applies Structural Header Chunking so 1 recipe = 1 intact chunk (~150 tokens). |
| **`embedding_selection.py`** | **Step 4:** Configures locked embedder (`text-embedding-3-small`, 1,536 dimensions). |
| **`vector_storage_and_retrieval.py`** | **Step 5:** Implements Hybrid Search ($\alpha = 0.7$ Cosine + BM25) and pre-query metadata filtering. |
| **`rag_generation_and_pipeline.py`** | **Step 6:** Assembles context prompt, enforces citations (`[1]`), and executes LangGraph state machine with refusal guardrail. |
| **`demo_app.py`** | Interactive CLI application for live testing. |
| **`run_demo_showcase.py`** | Automated test suite demonstrating 4 core use-case scenarios. |

---

## ⚙️ The 6-Step Implementation Summary

### Step 1: Corpus Curation
* Curated 30 classic and modern drinks formatted with uniform Markdown headers (`# Drink Name`) and standardized metadata attributes.

### Step 2: Ingestion & Cleaning
* Strips double-whitespaces, validates header formatting, and parses metadata tags into structured dictionaries for downstream pre-query filtering.

### Step 3: Structural Chunking
* Rejects fixed-size chunking to avoid slicing recipes mid-ingredient. Employs **Header Chunking** producing 30 intact chunks averaging ~150 tokens.

### Step 4: Embedding Model Selection
* Uses `text-embedding-3-small` (1,536 dimensions). Embedder version is locked to guarantee coordinate space stability.

### Step 5: Vector Storage & Hybrid Retrieval
* Combines **Dense Vector Search** (Cosine Similarity for semantic intent) with **Sparse BM25 Search** (for exact brand names like *"Campari"* or *"Chartreuse"*) using an $\alpha = 0.7$ weighting ratio.

### Step 6: LangGraph State Machine & Context Engineering
* Executes a state machine (`Retrieve` $\rightarrow$ `Grade Relevance` $\rightarrow$ `Generate` / `Refuse`):
  * **Citations:** Mandates `[1]`, `[2]` tags linked to retrieved chunk IDs.
  * **Refusal Guardrail:** Blocks out-of-domain queries (*"What is the capital of France?"*) with an explicit refusal clause.

---

## 🧪 Evaluation & Test Scenarios

| Scenario | Query Input | Expected Pipeline Behavior |
| :--- | :--- | :--- |
| **1. Direct Lookup** | *"How do I make a Margarita?"* | Retrieves *Margarita* chunk; outputs full recipe with `[1]` citation. |
| **2. Semantic Search** | *"What is a smoky agave drink with lime and spice?"* | Conceptually matches *Mezcalita* via vector similarity. |
| **3. Exact Brand Search** | *"Find recipes using Campari and Sweet Vermouth"* | BM25 sparse search matches *Negroni* and *Boulevardier*. |
| **4. Out-of-Domain Query** | *"Can you explain how black holes work?"* | Fails relevance check; triggers refusal guardrail. |

---
