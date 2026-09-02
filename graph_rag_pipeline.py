"""
graph_rag_pipeline.py
----------------------
Phase 2 / Item 1: Wires the Graph RAG knowledge graph (graph_builder.py) into
the existing LangGraph-style RAG state machine (rag_generation_and_pipeline.py).

Implements the integration exactly as designed in RAG_Patterns.md Section 5
"Proposed Pipeline Integration":

    [ Cocktail_Corpus.md ]
            |
            +--> Vector RAG Path (existing) --> Chunking -> Embedding -> Hybrid Search
            |
            +--> Graph RAG Path (NEW) --> Entity Extraction -> Graph Build -> Query Node
                                                  |
                                                  v
                                  [ LangGraph Router Node ]
                        "Is this a relationship/lineage question?"
                             |                              |
                        YES -> Graph RAG                NO -> Vector RAG (existing)
                             |                              |
                             +----------> Generate Node <----+

New node: graph_retrieve_node, alongside the existing retrieve_node.
New router: route_query() replaces the binary is_cocktail_query check with a
3-way decision: GRAPH / VECTOR / REFUSE.
"""

import re
from typing import List, Dict, Any, TypedDict, Optional, Literal

from vector_storage_and_retrieval import (
    ingest_and_clean_corpus,
    apply_structural_chunking,
    CocktailEmbedder,
    generate_embeddings_for_chunks,
    HybridVectorStore,
)
from rag_generation_and_pipeline import CocktailRAGPipeline, SYSTEM_PROMPT, NEBIUS_API_KEY
from graph_builder import build_graph_from_documents, CocktailKnowledgeGraph


# =====================================================================
# 1. EXTENDED STATE (adds graph fields on top of the Step 6 RAGState)
# =====================================================================

class GraphRAGState(TypedDict):
    question: str
    route: Literal["GRAPH", "VECTOR", "REFUSE", ""]
    retrieved_chunks: List[Dict[str, Any]]
    graph_result: Dict[str, Any]
    is_cocktail_query: bool
    final_answer: str


# =====================================================================
# 2. THE ROUTER — decides Graph vs. Vector vs. Refuse
# =====================================================================

# Keyword families that signal a RELATIONSHIP question, as opposed to a
# direct recipe lookup. This mirrors grade_relevance_node's keyword-matching
# style (Deterministic Workflow, not yet an LLM-based .bind_tools() router --
# that upgrade is Session_Index.md Open Item #2).
_GRAPH_INTENT_PATTERNS = [
    r"\bvariant[s]?\s+of\b",
    r"\bversion[s]?\s+of\b",
    r"\btwist[s]?\s+on\b",
    r"\bsimilar to\b",
    r"\brelated to\b",
    r"\blineage\b",
    r"\bevolved from\b",
    r"\bfamily\s+tree\b",
    r"\bshare[s]?\s+(?:the\s+)?(?:ingredient|campari|gin|vermouth)",
    r"\bbut\s+no\b",                 # "gin but no Campari"
    r"\bwithout\s+\w+",              # "without Campari"
    r"\bexcluding\b",
    r"\bwhat other cocktails\b",
    r"\bmost associated with\b",
    r"\bwhich (?:cocktails|drinks) (?:use|share|contain)\b",
]
_GRAPH_INTENT_REGEX = re.compile("|".join(_GRAPH_INTENT_PATTERNS), re.IGNORECASE)

_COCKTAIL_KEYWORDS = [
    "drink", "drinks", "cocktail", "cocktails", "recipe", "recipes", "ingredient", "ingredients",
    "whiskey", "bourbon", "gin", "rum", "tequila", "vodka", "mezcal", "scotch", "bitters", "vermouth",
    "campari", "shake", "stir", "glass", "garnish", "syrup", "lime", "lemon", "margarita", "negroni",
    "mojito", "daiquiri", "martini", "sazerac", "agave", "liquor", "bar", "mixology", "old fashioned",
    "manhattan", "paloma", "screwdriver", "make", "boulevardier", "flavor", "glassware", "variant",
]


def route_query(question: str) -> Literal["GRAPH", "VECTOR", "REFUSE"]:
    """
    Node: Router. Inspects query intent and picks a path.
    - GRAPH: relationship / lineage / shared-ingredient / variant questions
    - VECTOR: direct recipe lookups, semantic/flavor search, brand keyword search
    - REFUSE: no cocktail-domain signal at all
    """
    q = question.lower()
    has_cocktail_keyword = any(re.search(r"\b" + re.escape(kw) + r"\b", q) for kw in _COCKTAIL_KEYWORDS)
    has_graph_intent = bool(_GRAPH_INTENT_REGEX.search(q))

    if has_graph_intent and has_cocktail_keyword:
        return "GRAPH"
    if has_cocktail_keyword:
        return "VECTOR"
    return "REFUSE"


# =====================================================================
# 3. GRAPH RETRIEVE NODE
# =====================================================================

class GraphRAGPipeline(CocktailRAGPipeline):
    """
    Extends the Step 6 CocktailRAGPipeline with a graph_retrieve_node and a
    3-way router, per RAG_Patterns.md Section 5's proposed integration.
    """

    def __init__(self, vector_store: HybridVectorStore, graph: CocktailKnowledgeGraph, api_key: str = NEBIUS_API_KEY):
        super().__init__(vector_store, api_key)
        self.graph = graph

    def router_node(self, state: GraphRAGState) -> GraphRAGState:
        """Node: Router. Sets state['route'] to GRAPH / VECTOR / REFUSE."""
        route = route_query(state["question"])
        state["route"] = route
        print(f"🧭 [Node: Router] Query routed to: {route}")
        return state

    def graph_retrieve_node(self, state: GraphRAGState) -> GraphRAGState:
        """
        Node: Graph Retrieve. Parses the question for a target entity
        (cocktail / ingredient / spirit / flavor) and dispatches to the
        matching CocktailKnowledgeGraph query method.
        """
        question = state["question"]
        q_lower = question.lower()
        print(f"🕸️  [Node: Graph Retrieve] Querying Knowledge Graph for: '{question}'")

        result: Dict[str, Any] = {"query_type": None, "data": None, "cypher": None}

        def _mention(names: List[str]) -> Optional[str]:
            """
            Finds a known node name mentioned in the query. Checks both
            directions:
              1) full node name appears in the query (specific match, e.g.
                 "London Dry Gin" typed out in full) -> returns that exact name.
              2) a query word matches the last word of a multi-word name
                 (e.g. "Gin" -> "London Dry Gin", "Old Tom Gin") -> returns the
                 GENERIC query word itself (capitalized), not one arbitrary
                 specific node, so downstream CONTAINS-style graph queries can
                 aggregate across every matching node instead of picking one.
            """
            # 1) Exact/substring: full node name appears in the query
            direct = [n for n in names if n.lower() in q_lower]
            if direct:
                return max(direct, key=len)  # prefer the most specific match
            # 2) Reverse: a query word matches the last word of a multi-word name
            q_words = re.findall(r"[a-zA-Z']+", question)
            for word in q_words:
                if any(n.split()[-1].lower() == word.lower() for n in names):
                    return word.capitalize()
            return None

        # Try to detect a known Cocktail name mentioned in the query
        cocktail_names = [n["name"] for n in self.graph.nodes.values() if n["type"] == "Cocktail"]
        mentioned_cocktail = _mention(cocktail_names)

        # Try to detect a known Ingredient/Spirit mentioned in the query
        ingredient_names = [n["name"] for n in self.graph.nodes.values() if n["type"] == "Ingredient"]
        mentioned_ingredient = _mention(ingredient_names)

        spirit_names = [n["name"] for n in self.graph.nodes.values() if n["type"] == "BaseSpirit"]
        mentioned_spirit = _mention(spirit_names)

        flavor_names = [n["name"] for n in self.graph.nodes.values() if n["type"] == "Flavor"]
        mentioned_flavor = _mention(flavor_names)

        if ("variant" in q_lower or "version of" in q_lower or "lineage" in q_lower or "evolved" in q_lower) and mentioned_cocktail:
            variants, is_variant_of, cypher = self.graph.get_variants(mentioned_cocktail)
            lineage, lineage_cypher = self.graph.get_lineage(mentioned_cocktail)
            result.update({
                "query_type": "variants",
                "data": {"target": mentioned_cocktail, "variants": variants, "is_variant_of": is_variant_of, "lineage": lineage},
                "cypher": cypher,
            })
        elif mentioned_ingredient and mentioned_spirit and ("without" in q_lower or "no " in q_lower or "but no" in q_lower or "excluding" in q_lower):
            cocktails, cypher = self.graph.get_cocktails_by_spirit_excluding_ingredient(mentioned_spirit, mentioned_ingredient)
            result.update({
                "query_type": "spirit_excluding_ingredient",
                "data": {"spirit": mentioned_spirit, "excluded": mentioned_ingredient, "cocktails": cocktails},
                "cypher": cypher,
            })
        elif mentioned_ingredient:
            cocktails, cypher = self.graph.get_cocktails_by_ingredient(mentioned_ingredient)
            result.update({
                "query_type": "shared_ingredient",
                "data": {"ingredient": mentioned_ingredient, "cocktails": cocktails},
                "cypher": cypher,
            })
        elif mentioned_flavor and "glass" in q_lower:
            glass_counts, cypher = self.graph.get_glassware_by_flavor(mentioned_flavor)
            result.update({
                "query_type": "glassware_by_flavor",
                "data": {"flavor": mentioned_flavor, "glassware_counts": glass_counts},
                "cypher": cypher,
            })
        elif mentioned_cocktail:
            # Fallback: show what this cocktail shares with others
            shared, cypher = self.graph.get_shared_ingredient_cocktails(mentioned_cocktail)
            result.update({
                "query_type": "shared_ingredient_cocktails",
                "data": {"target": mentioned_cocktail, "shared": shared},
                "cypher": cypher,
            })

        state["graph_result"] = result
        print(f"   ↳ Graph query type: {result['query_type']} | Cypher: {result['cypher']}")
        return state

    def generate_from_graph_node(self, state: GraphRAGState) -> GraphRAGState:
        """
        Node: Generate (Graph path). Formats the structured graph result into
        a cited natural-language answer. Falls back to a templated response
        if the LLM call is unavailable (same fallback philosophy as
        generate_node in rag_generation_and_pipeline.py).
        """
        print("✍️  [Node: Generate-From-Graph] Formatting graph result into an answer...")
        gr = state["graph_result"]
        qtype = gr.get("query_type")
        data = gr.get("data") or {}

        if qtype == "variants":
            variants = data.get("variants", [])
            lineage = data.get("lineage", [])
            if variants:
                lines = [f"Cocktails that are variants of **{data['target']}** [GraphRAG]: {', '.join(variants)}."]
            elif data.get("is_variant_of"):
                lines = [f"**{data['target']}** is itself a variant of: {', '.join(data['is_variant_of'])} [GraphRAG]."]
            else:
                lines = [f"No recorded variants of **{data.get('target')}** were found in the knowledge graph [GraphRAG]."]
            if len(lineage) > 1:
                lines.append(f"Lineage chain: {' → '.join(lineage)} [GraphRAG].")
            state["final_answer"] = "\n".join(lines)

        elif qtype == "spirit_excluding_ingredient":
            cocktails = data.get("cocktails", [])
            if cocktails:
                state["final_answer"] = (
                    f"Cocktails using **{data['spirit']}** without **{data['excluded']}** [GraphRAG]: "
                    f"{', '.join(cocktails)}."
                )
            else:
                state["final_answer"] = (
                    f"No cocktails found using {data['spirit']} without {data['excluded']} [GraphRAG]."
                )

        elif qtype == "shared_ingredient":
            cocktails = data.get("cocktails", [])
            state["final_answer"] = (
                f"Cocktails that use **{data['ingredient']}** [GraphRAG]: {', '.join(cocktails)}."
                if cocktails else f"No cocktails found using {data['ingredient']} [GraphRAG]."
            )

        elif qtype == "glassware_by_flavor":
            counts = data.get("glassware_counts", [])
            if counts:
                top = ", ".join(f"{g} ({c})" for g, c in counts[:5])
                state["final_answer"] = (
                    f"Glassware most associated with **{data['flavor']}** cocktails [GraphRAG]: {top}."
                )
            else:
                state["final_answer"] = f"No glassware data found for flavor {data['flavor']} [GraphRAG]."

        elif qtype == "shared_ingredient_cocktails":
            shared = data.get("shared", {})
            if shared:
                lines = [f"**{data['target']}** shares ingredients with other cocktails [GraphRAG]:"]
                for ing, others in shared.items():
                    lines.append(f"  • {ing}: {', '.join(others)}")
                state["final_answer"] = "\n".join(lines)
            else:
                state["final_answer"] = f"No shared-ingredient relationships found for {data.get('target')} [GraphRAG]."

        else:
            state["final_answer"] = (
                "I understood this as a relationship question, but couldn't identify a specific "
                "cocktail, ingredient, or spirit mentioned in the query. Could you name one explicitly? [GraphRAG]"
            )

        return state

    def run_workflow(self, question: str) -> GraphRAGState:
        """Executes the extended 3-way-routed state machine."""
        print("\n" + "=" * 60)
        print(f"🎬 EXECUTING GRAPH-RAG WORKFLOW FOR QUERY: '{question}'")
        print("=" * 60)

        state: GraphRAGState = {
            "question": question,
            "route": "",
            "retrieved_chunks": [],
            "graph_result": {},
            "is_cocktail_query": False,
            "final_answer": "",
        }

        state = self.router_node(state)

        if state["route"] == "GRAPH":
            state = self.graph_retrieve_node(state)
            state = self.generate_from_graph_node(state)
        elif state["route"] == "VECTOR":
            state = self.retrieve_node(state)
            state = self.grade_relevance_node(state)
            if state["is_cocktail_query"]:
                state = self.generate_node(state)
            else:
                state = self.refuse_node(state)
        else:  # REFUSE
            state = self.refuse_node(state)

        print("\n🏁 WORKFLOW COMPLETE. FINAL ANSWER OUTPUT:\n")
        print(state["final_answer"])
        print("=" * 60)
        return state


if __name__ == "__main__":
    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    chunks = apply_structural_chunking(cleaned_docs)
    embedder = CocktailEmbedder(model_name="Qwen/Qwen3-Embedding-8B", dimensions=4096)
    records = generate_embeddings_for_chunks(chunks, embedder)
    vector_store = HybridVectorStore(records, embedder)
    graph = build_graph_from_documents(cleaned_docs)

    pipeline = GraphRAGPipeline(vector_store, graph)

    # TEST 1: Graph-routed relationship question
    pipeline.run_workflow("What other cocktails are variants of the Negroni?")

    # TEST 2: Graph-routed shared-ingredient question
    pipeline.run_workflow("Which cocktails share Campari as an ingredient?")

    # TEST 3: Graph-routed multi-hop exclusion question
    pipeline.run_workflow("If I have Gin but no Campari, what cocktails could I make?")

    # TEST 4: Vector-routed direct lookup (existing path, unchanged)
    pipeline.run_workflow("How do I make a Margarita?")

    # TEST 5: Refusal path (existing behavior, unchanged)
    pipeline.run_workflow("What is the capital city of France?")
