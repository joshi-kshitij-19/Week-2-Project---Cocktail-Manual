# Cocktail Manual RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**-style state machines, and **LangGraph**-style orchestration. It ingests a curated mixology corpus, indexes recipe chunks using hybrid search (dense vectors + sparse BM25), and layers **Graph RAG**, **Agentic RAG (ReAct)**, and **Short-Term Memory** on top of a stateful pipeline that answers recipe queries with citations, resolves multi-hop/relationship questions, and enforces a domain-refusal guardrail.

Runs against **live Nebius AI Studio APIs** (embeddings + LLM generation) with automatic **offline fallback** — no external calls required to run and demo the whole system.

---

## 🏗️ Architecture

```
[ Cocktail_Corpus.md ] ──► [ Step 2: Ingestion & Cleaning ] ──► Cleaned Docs (30)
                                                                    │
[ Step 5: Hybrid Vector Store ] ◄── [ Step 4: 4096D Embeddings ] ◄── [ Step 3: Structural Chunker ]
  (Cosine + BM25, α = 0.7)         (Nebius Qwen3-Embedding-8B)        (Intact Recipe Chunks)
            │
            ▼
[ Step 6: State Machine ] ──► Retrieve → Grade Relevance → Generate / Refuse
  (Nebius Llama-3.3-70B-Instruct, cited answers)
            │
            ▼  Phase 2 ─────────────────────────────────────────────────────────
            │
   ┌────────┴────────┐
   │  router_node     │  3-way regex intent router
   └───┬────────┬─────┘
       │        │
   GRAPH      VECTOR ──► (existing Step 6 path above)
       │
       ▼
[ graph_builder.py ] Knowledge Graph (Cocktail/Spirit/Ingredient/Glassware/Flavor nodes)
       │
       ▼
   needs_agentic_escalation()? ──YES──► AgenticRAGRouter (ReAct: THINK→ACT→OBSERVE→DECIDE)
       │NO                              chains Graph→Graph→Vector tool calls, with
       ▼                                duplicate-call dedup + groundedness/contradiction check
   generate_from_graph_node()
       │
       ▼
   needs_query_rewrite()? ──YES──► rewrite dangling follow-up using chat_history
       │ (Short-Term Memory, ConversationalRAGPipeline)
       ▼
   Final cited answer ──► on any refusal/low-confidence/no-match path,
                           log_missing_record() appends to missing_records_log.jsonl
```

**Current classification:** Deterministic Workflow (Vector RAG path) **augmented with** a Graph RAG path, an Agentic RAG escalation layer, and Short-Term conversational Memory. See `Session_Index.md` for the full design rationale and live-validation notes.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10+
* (Optional) A Nebius AI Studio API key for live embeddings/generation

### Environment Setup
Set `NEBIUS_API_KEY` to use **live** embeddings/generation; otherwise the app automatically runs in **offline fallback mode** (deterministic hash-based embeddings + templated answers — no external calls, no crash, just lower-quality results).

```bash
cp .env.example .env      # then edit .env with your real key
# or, for a single terminal session:
export NEBIUS_API_KEY="your-key-here"
```

### Run Demo Applications

1. **Interactive Terminal Chat App** (full Phase 2 stack — Graph RAG + Agentic RAG + Short-Term Memory):
   ```bash
   python3 demo_app.py
   ```

2. **Automated Showcase (Runs 4 core Step 1–6 test scenarios):**
   ```bash
   python3 run_demo_showcase.py
   ```

3. **Batch Test Queries:**
   ```bash
   python3 run_custom_queries.py
   ```

---

## 📂 Repository Structure

### Data
| File | Purpose |
| :--- | :--- |
| **`Cocktail_Corpus.md`** | Curated knowledge base containing 30 standardized cocktail recipes. |

### Core Pipeline (Steps 2–6)
| File | Step | Purpose |
| :--- | :--- | :--- |
| **`ingest_and_clean.py`** | 2 | Loads raw Markdown, sanitizes text, and extracts metadata (`drink_name`, `base_spirit`, `glassware`, `category`, `flavor_profile`). |
| **`chunking_strategy.py`** | 3 | Applies Structural Header Chunking so 1 recipe = 1 intact chunk (~150 tokens). |
| **`embedding_selection.py`** | 4 | `CocktailEmbedder` — calls **live Nebius AI Studio API** (`Qwen/Qwen3-Embedding-8B`, 4,096 dimensions) with automatic offline hash-based fallback. |
| **`vector_storage_and_retrieval.py`** | 5 | `HybridVectorStore` — Cosine Similarity (dense) + BM25 (sparse) + exact/fuzzy title-match boost + pre-query metadata filtering, combined via α-weighted hybrid score. |
| **`rag_generation_and_pipeline.py`** | 6 | `CocktailRAGPipeline` — state machine (`retrieve_node` → `grade_relevance_node` → `generate_node`/`refuse_node`); calls **live Nebius Llama-3.3-70B-Instruct** for cited generation, with structured fallback if the API is unreachable. |

### Phase 2 — Graph RAG, Agentic RAG, Memory, Data Flywheel
| File | Purpose |
| :--- | :--- |
| **`graph_builder.py`** | Builds an in-memory Knowledge Graph from `Cocktail_Corpus.md` (Cocktail/BaseSpirit/Ingredient/Glassware/Flavor/Creator nodes; `USES_SPIRIT`/`USES_INGREDIENT`/`SERVED_IN`/`HAS_FLAVOR`/`VARIANT_OF`/`CREATED_BY` edges). Pure-stdlib, no external graph DB — every query method documents its Neo4j Cypher equivalent for future migration. |
| **`graph_rag_pipeline.py`** | `GraphRAGPipeline` — extends Step 6 with a 3-way `router_node` (GRAPH / VECTOR / REFUSE via regex intent detection), `graph_retrieve_node`, and `generate_from_graph_node`. |
| **`agentic_rag_pipeline.py`** | `EscalatingRAGPipeline` — adds a 4th escalation branch: compound/multi-hop queries (detected via `needs_agentic_escalation()`) route to `AgenticRAGRouter`, a real ReAct loop using **live OpenAI-style tool-calling** against Nebius Llama-3.3-70B-Instruct to chain Graph→Graph→Vector tool calls. Includes retry-with-backoff and a duplicate-call dedup guard to prevent infinite loops, plus a deterministic `_find_contradiction()` groundedness check so the agent can't assert the opposite of its own tool observation. |
| **`short_term_memory.py`** | `ConversationalRAGPipeline` — extends `EscalatingRAGPipeline` with a running `chat_history` that persists across turns in one session. `needs_query_rewrite()` (regex-based, no LLM call) decides whether a query is self-contained or a dangling follow-up (e.g. "make it less sweet") that needs rewriting via one LLM call against the prior turn. `demo_app.py` uses this pipeline. |
| **`missing_record_logger.py`** | Pure logging mechanism (not a live external fallback) — `log_missing_record()` appends `{query, reason, top_score, source_node, timestamp}` as JSON lines to `missing_records_log.jsonl` (gitignored) whenever `refuse_node()`, `generate_node()`'s low-confidence path, or the agent's "no match" conclusion fires. Powers a manual-curation "data flywheel" for corpus gaps without changing what the user sees. |

### Demo / Test Scripts
| File | Purpose |
| :--- | :--- |
| **`demo_app.py`** | Interactive CLI chat app running the full Phase 2 stack — type any cocktail question live. |
| **`run_demo_showcase.py`** | Automated 4-scenario showcase of the core Step 1–6 pipeline (direct lookup, semantic intent, brand keyword, refusal guardrail). |
| **`run_custom_queries.py`** | Batch test runner for custom query lists. |
| **`test_old_fashioned.py`** | Regression test for the "Old Fashioned – ingredient list" / typo-handling fix. |

### Config
| File | Purpose |
| :--- | :--- |
| **`.env.example`** | Template for `NEBIUS_API_KEY`; copy to `.env` and fill in your real key. |
| **`.gitignore`** | Excludes `__pycache__/`, `*.pyc`, `.DS_Store`, `.env`, `missing_records_log.jsonl`. |

---

## ⚙️ The 6-Step Core Pipeline Summary

### Step 1: Corpus Curation
* Curated 30 classic and modern drinks formatted with uniform Markdown headers (`# Drink Name`) and standardized metadata attributes.

### Step 2: Ingestion & Cleaning
* Strips double-whitespaces, validates header formatting, and parses metadata tags into structured dictionaries for downstream pre-query filtering.

### Step 3: Structural Chunking
* Rejects fixed-size chunking to avoid slicing recipes mid-ingredient. Employs **Header Chunking** producing 30 intact chunks averaging ~150 tokens.

### Step 4: Embedding Model Selection
* Uses **`Qwen/Qwen3-Embedding-8B`** via live Nebius AI Studio API (4,096 dimensions), with an automatic offline hash-based fallback if the API is unreachable.

### Step 5: Vector Storage & Hybrid Retrieval
* Combines **Dense Vector Search** (Cosine Similarity for semantic intent) with **Sparse BM25 Search** (for exact brand names like *"Campari"* or *"Chartreuse"*) using an α = 0.7 weighting ratio, plus exact/fuzzy title-match boosting and pre-query metadata filtering.

### Step 6: State Machine & Context Engineering
* Executes a state machine (`Retrieve` → `Grade Relevance` → `Generate` / `Refuse`) calling **live Nebius Llama-3.3-70B-Instruct**:
  * **Citations:** Mandates `[1]`, `[2]` tags linked to retrieved chunk IDs.
  * **Refusal Guardrail:** Blocks out-of-domain queries (*"What is the capital of France?"*) with an explicit refusal clause.

---

## 🧠 Phase 2: Graph RAG, Agentic RAG, Memory & Data Flywheel

| Upgrade | File(s) | What It Adds |
| :--- | :--- | :--- |
| **Graph RAG** | `graph_builder.py`, `graph_rag_pipeline.py` | A 3-way router sends relationship/variant/pairing questions (*"What other cocktails are variants of the Negroni?"*) to an in-memory Knowledge Graph instead of vector search. |
| **Agentic RAG (ReAct escalation)** | `agentic_rag_pipeline.py` | Compound, multi-hop questions (*"Is there a Negroni variant without Campari, and what glass does it use?"*) escalate to a real ReAct loop that chains multiple tool calls and reasons over the results — only when the fast regex router can't handle it in one hop. |
| **Short-Term Memory** | `short_term_memory.py` | Multi-turn conversations resolve dangling follow-ups (*"make it less sweet"*) against the previous turn's subject, without needing an LLM call for turns that are already self-contained. |
| **Missing-Record Data Flywheel** | `missing_record_logger.py` | Every refusal, low-confidence answer, or "no match" agent conclusion is logged to `missing_records_log.jsonl` for later manual corpus curation — turning demo-time gaps into a backlog instead of silent failures. |

**Design principle throughout Phase 2:** *escalation, not replacement* — the free regex-based router and Step 6 pipeline remain the default path; the more expensive Graph/Agentic/Memory layers only activate when a query genuinely needs them. See `Session_Index.md` and `RAG_Patterns.md` for the full design rationale, and `Learnings from Project 1 - RAG Application.md` for every real bug found only by live execution.

---

## 🧪 Evaluation & Test Scenarios

| Scenario | Query Input | Expected Pipeline Behavior |
| :--- | :--- | :--- |
| **1. Direct Lookup** | *"How do I make a Margarita?"* | Retrieves *Margarita* chunk; outputs full recipe with `[1]` citation. |
| **2. Semantic Search** | *"What is a smoky agave drink with lime and spice?"* | Conceptually matches *Mezcalita* via vector similarity. |
| **3. Exact Brand Search** | *"Find recipes using Campari and Sweet Vermouth"* | BM25 sparse search matches *Negroni* and *Boulevardier*. |
| **4. Out-of-Domain Query** | *"Can you explain how black holes work?"* | Fails relevance check; triggers refusal guardrail + logs missing record. |
| **5. Graph RAG Relationship Query** | *"What other cocktails are variants of the Negroni?"* | Router detects graph intent; `graph_retrieve_node` traverses `VARIANT_OF` edges. |
| **6. Agentic Multi-Hop Query** | *"Is there a Negroni variant without Campari, and what glass does it use?"* | Escalates to `AgenticRAGRouter`; chains Graph→Graph→Vector tool calls; correctly concludes "no such variant exists" rather than hallucinating. |
| **7. Short-Term Memory Follow-up** | *"How do I make a Margarita?"* → *"Make it less sweet"* | Turn 2 is rewritten via `needs_query_rewrite()` into a self-contained Margarita question before retrieval. |

---

## 🔑 Open Items (see `Session_Index.md` for details)
1. **Long-Term Memory** (per-user preferences/allergies) — designed in `Memory.md` §4b, blocked on the identity-resolution problem (no `user_id` concept in the anonymous CLI).
2. **GitHub push** — repo is git-initialized locally; not yet pushed to a remote.

---

## 📜 License
MIT License
