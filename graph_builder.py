"""
graph_builder.py
-----------------
Phase 2 / Item 1: Graph RAG for the Cocktail Manual.

Builds an in-memory Knowledge Graph from Cocktail_Corpus.md, matching the schema
proposed in RAG_Patterns.md Section 5:

    Node types:  Cocktail, BaseSpirit, Ingredient, Glassware, Flavor, Creator
    Edge types:  USES_SPIRIT, USES_INGREDIENT, SERVED_IN, HAS_FLAVOR,
                 VARIANT_OF, CREATED_BY

Design choice: pure-stdlib in-memory graph (adjacency lists), NOT a live Neo4j
connection. This mirrors the same "graceful, dependency-free fallback" pattern
already used elsewhere in this pipeline (e.g. ingest_and_clean.py's Document
dataclass fallback, embedding_selection.py's offline hash fallback) — it runs
anywhere with zero external services, while every query method also exposes
its Cypher equivalent so the mapping to a real Neo4j deployment stays explicit.
"""

import re
import difflib
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

from ingest_and_clean import ingest_and_clean_corpus, Document


# =====================================================================
# 1. TEXT NORMALIZATION HELPERS
# =====================================================================

def _normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace -> used as the node ID key."""
    return re.sub(r"\s+", "_", name.strip().lower())


def _split_multi(field: str) -> List[str]:
    """
    Splits a metadata field that may list multiple values joined by
    '&', ' or ', ',', or '/' (e.g. "Aged Jamaican Rum & Martinique Rum").
    """
    if not field:
        return []
    # Strip parenthetical qualifiers like "(Salt Rim)" before splitting
    field_clean = re.sub(r"\([^)]*\)", "", field)
    parts = re.split(r"\s*&\s*|\s+or\s+|,\s*|/\s*", field_clean)
    return [p.strip() for p in parts if p.strip()]


def _extract_field(page_content: str, field_label: str) -> Optional[str]:
    """Extracts a single '- **Label:** value' line from raw recipe text."""
    match = re.search(
        rf"-\s*\*\*{re.escape(field_label)}:\*\*\s*(.+)$",
        page_content,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _extract_ingredients(page_content: str) -> List[str]:
    """
    Parses the '**Ingredients:**' line into individual ingredient names,
    stripping quantities/units (e.g. '2.0 oz Bourbon Whiskey' -> 'Bourbon Whiskey').
    """
    raw = _extract_field(page_content, "Ingredients")
    if not raw:
        return []

    # Split on commas that are NOT inside parentheses (preserves "(or 0.25 oz Simple Syrup)")
    tokens = re.split(r",\s*(?![^()]*\))", raw)

    ingredients = []
    qty_prefix = re.compile(
        r"^[\d\.\-/¼½¾]*\s*"
        r"(oz\.?|ml|cl|dash(?:es)?|drop(?:s)?|cube[s]?|tsp|leaves?|slice[s]?|"
        r"wedge[s]?|sprig[s]?|pinch(?:es)? of|splash of|top with|float of|rinse of|"
        r"fresh)?\s*",
        re.IGNORECASE,
    )
    for tok in tokens:
        tok = tok.strip().rstrip(".")
        # Drop a leading parenthetical alt-ingredient note; keep primary text
        tok = re.sub(r"^\(.*?\)\s*", "", tok)
        # Strip leading quantity/unit words (repeat once for e.g. "2.0 oz")
        cleaned = qty_prefix.sub("", tok, count=1).strip()
        cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()  # trailing "(float)" etc.
        if cleaned and len(cleaned) > 1:
            ingredients.append(cleaned)
    return ingredients


_VARIANT_PATTERNS = [
    re.compile(r"essentially an?\s+([a-zA-Z\s]+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"twist on (?:the\s+)?([A-Za-z' ]+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"variant of (?:the\s+)?([A-Za-z' ]+?)(?:\.|;|$)", re.IGNORECASE),
    re.compile(r"version of (?:the\s+)?([A-Za-z' ]+?)(?:\.|;|$)", re.IGNORECASE),
]


def _detect_variant_of(origin_text: str, known_names: List[str]) -> Optional[str]:
    """
    Looks for phrases like 'essentially a whiskey Negroni' in the Origin/Notes
    field and fuzzy-matches the trailing phrase against known Cocktail names.
    """
    if not origin_text:
        return None
    for pattern in _VARIANT_PATTERNS:
        m = pattern.search(origin_text)
        if m:
            candidate_phrase = m.group(1) if m.lastindex else m.group(0)
            best_match, best_ratio = None, 0.0
            for name in known_names:
                ratio = difflib.SequenceMatcher(
                    None, candidate_phrase.lower(), name.lower()
                ).ratio()
                # Also check if the known name's last word appears in the phrase
                # (handles "essentially a whiskey Negroni" -> "Negroni")
                last_word = name.split()[-1].lower()
                if last_word in candidate_phrase.lower():
                    ratio = max(ratio, 0.9)
                if ratio > best_ratio:
                    best_ratio, best_match = ratio, name
            if best_match and best_ratio >= 0.35:
                return best_match
    return None


_CREATOR_PATTERN = re.compile(
    r"(?:[Cc]reated|[Ii]nvented|[Pp]opularized)\s+by\s+([A-Z][A-Za-z.'\- ]+?)"
    r"(?:\s+at|\s+in|\s+for|,|\.|$)"
)


def _extract_creator(origin_text: str) -> Optional[str]:
    if not origin_text:
        return None
    m = _CREATOR_PATTERN.search(origin_text)
    return m.group(1).strip() if m else None


# =====================================================================
# 2. THE KNOWLEDGE GRAPH (in-memory, Cypher-mirroring query methods)
# =====================================================================

class CocktailKnowledgeGraph:
    """
    A minimal labeled-property graph: nodes carry a type + display name,
    edges are (source_id, relationship_label, target_id) triples.

    Every query method below documents its Neo4j Cypher equivalent, so the
    mental model transfers directly if this is ever swapped for a real
    Neo4j-backed implementation.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, str]] = {}
        self._out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # src -> [(rel, dst)]
        self._in: Dict[str, List[Tuple[str, str]]] = defaultdict(list)   # dst -> [(rel, src)]

    # ---- graph construction -----------------------------------------
    def add_node(self, node_type: str, name: str) -> str:
        node_id = f"{node_type}:{_normalize(name)}"
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, "name": name}
        return node_id

    def add_edge(self, src_id: str, relationship: str, dst_id: str):
        self._out[src_id].append((relationship, dst_id))
        self._in[dst_id].append((relationship, src_id))

    # ---- low-level traversal -----------------------------------------
    def _neighbors(self, node_id: str, relationship: str, reverse: bool = False) -> List[str]:
        table = self._in if reverse else self._out
        return [dst for (rel, dst) in table.get(node_id, []) if rel == relationship]

    def find_node(self, node_type: str, name: str, fuzzy: bool = True) -> Optional[str]:
        exact_id = f"{node_type}:{_normalize(name)}"
        if exact_id in self.nodes:
            return exact_id
        if not fuzzy:
            return None
        # Fuzzy fallback across all nodes of this type (handles "gin" vs "London Dry Gin")
        candidates = [nid for nid, n in self.nodes.items() if n["type"] == node_type]
        best_id, best_ratio = None, 0.0
        name_l = name.lower()
        for nid in candidates:
            display = self.nodes[nid]["name"].lower()
            ratio = difflib.SequenceMatcher(None, name_l, display).ratio()
            if name_l in display or display in name_l:
                ratio = max(ratio, 0.85)
            if ratio > best_ratio:
                best_ratio, best_id = ratio, nid
        return best_id if best_ratio >= 0.55 else None

    def display_name(self, node_id: str) -> str:
        return self.nodes[node_id]["name"] if node_id in self.nodes else node_id

    def find_all_matching_nodes(self, node_type: str, name: str) -> List[str]:
        """
        Returns ALL nodes of this type that plausibly match `name`, instead of
        just the single best fuzzy match. This matters for generic category
        terms like "Gin" that legitimately span multiple graph nodes
        ("London Dry Gin", "Old Tom Gin") — find_node() picks one arbitrarily,
        which silently drops real results; this returns the full matching set.
        """
        exact_id = f"{node_type}:{_normalize(name)}"
        if exact_id in self.nodes:
            return [exact_id]
        name_l = name.lower()
        matches = []
        for nid, n in self.nodes.items():
            if n["type"] != node_type:
                continue
            display = n["name"].lower()
            # Match if the search term is a whole word within the node's name
            # (e.g. "gin" matches "London Dry Gin" and "Old Tom Gin")
            if re.search(r"\b" + re.escape(name_l) + r"\b", display) or name_l in display:
                matches.append(nid)
        return matches

    # ---- query methods (each documents its Cypher equivalent) --------
    def get_variants(self, cocktail_name: str) -> Tuple[List[str], List[str], Optional[str]]:
        """
        Cypher: MATCH (c:Cocktail)-[:VARIANT_OF]->(x:Cocktail {name: $name}) RETURN c
                MATCH (x:Cocktail {name: $name})-[:VARIANT_OF]->(p:Cocktail) RETURN p
        Returns: (things that ARE variants of it, thing IT is a variant of, cypher_str)
        """
        node_id = self.find_node("Cocktail", cocktail_name)
        cypher = (
            'MATCH (c:Cocktail)-[:VARIANT_OF]->(x:Cocktail {name:"%s"}) RETURN c.name'
            % cocktail_name
        )
        if not node_id:
            return [], [], cypher
        variants_of_it = [self.display_name(n) for n in self._neighbors(node_id, "VARIANT_OF", reverse=True)]
        it_is_variant_of = [self.display_name(n) for n in self._neighbors(node_id, "VARIANT_OF", reverse=False)]
        return variants_of_it, it_is_variant_of, cypher

    def get_cocktails_by_ingredient(self, ingredient_name: str) -> Tuple[List[str], str]:
        """Cypher: MATCH (c:Cocktail)-[:USES_INGREDIENT]->(i:Ingredient {name:$ing}) RETURN c"""
        cypher = (
            'MATCH (c:Cocktail)-[:USES_INGREDIENT]->(i:Ingredient {name:"%s"}) RETURN c.name'
            % ingredient_name
        )
        ing_id = self.find_node("Ingredient", ingredient_name)
        if not ing_id:
            return [], cypher
        return [self.display_name(n) for n in self._neighbors(ing_id, "USES_INGREDIENT", reverse=True)], cypher

    def get_cocktails_by_spirit_excluding_ingredient(
        self, spirit_name: str, exclude_ingredient: str, flavor: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """
        Cypher: MATCH (c:Cocktail)-[:USES_SPIRIT]->(s:BaseSpirit)
                WHERE s.name CONTAINS $spirit
                AND NOT (c)-[:USES_INGREDIENT]->(:Ingredient {name:$exclude})
                [AND (c)-[:HAS_FLAVOR]->(:Flavor {name:$flavor})]
                RETURN c

        Note: matches ALL BaseSpirit nodes whose name contains `spirit_name`
        (e.g. "Gin" -> "London Dry Gin" AND "Old Tom Gin"), not just the single
        closest fuzzy match — a generic spirit category can span multiple
        graph nodes, and returning only one silently drops real results.
        """
        cypher = (
            'MATCH (c:Cocktail)-[:USES_SPIRIT]->(s:BaseSpirit) WHERE s.name CONTAINS "%s" '
            'AND NOT (c)-[:USES_INGREDIENT]->(:Ingredient {name:"%s"}) RETURN c.name'
            % (spirit_name, exclude_ingredient)
        )
        spirit_ids = self.find_all_matching_nodes("BaseSpirit", spirit_name)
        if not spirit_ids:
            return [], cypher
        cocktail_ids = set()
        for spirit_id in spirit_ids:
            cocktail_ids |= set(self._neighbors(spirit_id, "USES_SPIRIT", reverse=True))

        exclude_id = self.find_node("Ingredient", exclude_ingredient)
        excluded = set(self._neighbors(exclude_id, "USES_INGREDIENT", reverse=True)) if exclude_id else set()
        cocktail_ids -= excluded

        if flavor:
            flavor_id = self.find_node("Flavor", flavor)
            if flavor_id:
                flavored = set(self._neighbors(flavor_id, "HAS_FLAVOR", reverse=True))
                cocktail_ids &= flavored

        return [self.display_name(n) for n in cocktail_ids], cypher

    def get_lineage(self, cocktail_name: str) -> Tuple[List[str], str]:
        """
        Cypher: MATCH path=(c:Cocktail {name:$name})-[:VARIANT_OF*1..5]->(ancestor:Cocktail)
                RETURN [n IN nodes(path) | n.name]
        """
        cypher = (
            'MATCH path=(c:Cocktail {name:"%s"})-[:VARIANT_OF*1..5]->(a:Cocktail) '
            "RETURN [n IN nodes(path) | n.name]" % cocktail_name
        )
        node_id = self.find_node("Cocktail", cocktail_name)
        if not node_id:
            return [], cypher
        chain = [self.display_name(node_id)]
        current = node_id
        seen = {current}
        while True:
            parents = self._neighbors(current, "VARIANT_OF", reverse=False)
            if not parents or parents[0] in seen:
                break
            current = parents[0]
            seen.add(current)
            chain.append(self.display_name(current))
        return chain, cypher

    def get_glassware_by_flavor(self, flavor_name: str) -> Tuple[List[Tuple[str, int]], str]:
        """
        Cypher: MATCH (g:Glassware)<-[:SERVED_IN]-(c:Cocktail)-[:HAS_FLAVOR]->(f:Flavor {name:$flavor})
                RETURN g.name, count(c) ORDER BY count(c) DESC
        """
        cypher = (
            'MATCH (g:Glassware)<-[:SERVED_IN]-(c:Cocktail)-[:HAS_FLAVOR]->(f:Flavor {name:"%s"}) '
            "RETURN g.name, count(c) ORDER BY count(c) DESC" % flavor_name
        )
        flavor_id = self.find_node("Flavor", flavor_name)
        if not flavor_id:
            return [], cypher
        cocktails = self._neighbors(flavor_id, "HAS_FLAVOR", reverse=True)
        counts = Counter()
        for c in cocktails:
            for g in self._neighbors(c, "SERVED_IN", reverse=False):
                counts[self.display_name(g)] += 1
        return counts.most_common(), cypher

    def get_shared_ingredient_cocktails(self, cocktail_name: str) -> Tuple[Dict[str, List[str]], str]:
        """
        Cypher: MATCH (c1:Cocktail {name:$name})-[:USES_INGREDIENT]->(i)<-[:USES_INGREDIENT]-(c2:Cocktail)
                WHERE c1 <> c2 RETURN i.name, collect(c2.name)
        """
        cypher = (
            'MATCH (c1:Cocktail {name:"%s"})-[:USES_INGREDIENT]->(i)<-[:USES_INGREDIENT]-(c2:Cocktail) '
            "WHERE c1 <> c2 RETURN i.name, collect(c2.name)" % cocktail_name
        )
        node_id = self.find_node("Cocktail", cocktail_name)
        if not node_id:
            return {}, cypher
        result: Dict[str, List[str]] = {}
        for ing_id in self._neighbors(node_id, "USES_INGREDIENT", reverse=False):
            other_cocktails = [
                self.display_name(c) for c in self._neighbors(ing_id, "USES_INGREDIENT", reverse=True)
                if c != node_id
            ]
            if other_cocktails:
                result[self.display_name(ing_id)] = other_cocktails
        return result, cypher

    # ---- stats ---------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        counts = Counter(n["type"] for n in self.nodes.values())
        edge_count = sum(len(v) for v in self._out.values())
        counts["__edges__"] = edge_count
        return dict(counts)


# =====================================================================
# 3. GRAPH BUILDER — Corpus -> Knowledge Graph
# =====================================================================

def build_graph_from_documents(docs: List[Document]) -> CocktailKnowledgeGraph:
    """Extracts entities/relationships from cleaned recipe docs into a CocktailKnowledgeGraph."""
    print("🕸️  [Graph RAG] Building in-memory Knowledge Graph from corpus...")
    graph = CocktailKnowledgeGraph()

    known_names = [doc.metadata["drink_name"] for doc in docs if "drink_name" in doc.metadata]

    for doc in docs:
        meta = doc.metadata
        name = meta.get("drink_name")
        if not name:
            continue

        cocktail_id = graph.add_node("Cocktail", name)

        for spirit in _split_multi(meta.get("base_spirit", "")):
            spirit_id = graph.add_node("BaseSpirit", spirit)
            graph.add_edge(cocktail_id, "USES_SPIRIT", spirit_id)

        for glass in _split_multi(meta.get("glassware", "")):
            glass_id = graph.add_node("Glassware", glass)
            graph.add_edge(cocktail_id, "SERVED_IN", glass_id)

        for flavor in _split_multi(meta.get("flavor_profile", "")):
            flavor_id = graph.add_node("Flavor", flavor)
            graph.add_edge(cocktail_id, "HAS_FLAVOR", flavor_id)

        for ingredient in _extract_ingredients(doc.page_content):
            ing_id = graph.add_node("Ingredient", ingredient)
            graph.add_edge(cocktail_id, "USES_INGREDIENT", ing_id)

        origin_text = _extract_field(doc.page_content, "Origin/Notes") or ""

        variant_target = _detect_variant_of(origin_text, [n for n in known_names if n != name])
        if variant_target:
            target_id = graph.find_node("Cocktail", variant_target, fuzzy=False) or graph.add_node("Cocktail", variant_target)
            graph.add_edge(cocktail_id, "VARIANT_OF", target_id)

        creator = _extract_creator(origin_text)
        if creator:
            creator_id = graph.add_node("Creator", creator)
            graph.add_edge(cocktail_id, "CREATED_BY", creator_id)

    stats = graph.stats()
    print(
        f"✅ [Graph RAG] Graph built: {stats.get('Cocktail', 0)} Cocktails, "
        f"{stats.get('BaseSpirit', 0)} Spirits, {stats.get('Ingredient', 0)} Ingredients, "
        f"{stats.get('Glassware', 0)} Glassware, {stats.get('Flavor', 0)} Flavors, "
        f"{stats.get('Creator', 0)} Creators, {stats.get('__edges__', 0)} total edges."
    )
    return graph


if __name__ == "__main__":
    cleaned_docs = ingest_and_clean_corpus("Cocktail_Corpus.md")
    kg = build_graph_from_documents(cleaned_docs)

    print("\n" + "=" * 60)
    print("🧪 GRAPH QUERY TEST 1: Variants of Negroni")
    print("=" * 60)
    variants, is_variant_of, cypher = kg.get_variants("Negroni")
    print(f"  Cypher: {cypher}")
    print(f"  Cocktails that ARE variants of Negroni: {variants}")

    print("\n" + "=" * 60)
    print("🧪 GRAPH QUERY TEST 2: Cocktails sharing Campari")
    print("=" * 60)
    campari_cocktails, cypher2 = kg.get_cocktails_by_ingredient("Campari")
    print(f"  Cypher: {cypher2}")
    print(f"  Cocktails using Campari: {campari_cocktails}")

    print("\n" + "=" * 60)
    print("🧪 GRAPH QUERY TEST 3: Gin cocktails WITHOUT Campari")
    print("=" * 60)
    gin_no_campari, cypher3 = kg.get_cocktails_by_spirit_excluding_ingredient("Gin", "Campari")
    print(f"  Cypher: {cypher3}")
    print(f"  Result: {gin_no_campari}")

    print("\n" + "=" * 60)
    print("🧪 GRAPH QUERY TEST 4: Lineage of Boulevardier")
    print("=" * 60)
    lineage, cypher4 = kg.get_lineage("Boulevardier")
    print(f"  Cypher: {cypher4}")
    print(f"  Lineage chain: {' -> '.join(lineage)}")

    print("\n" + "=" * 60)
    print("🧪 GRAPH QUERY TEST 5: Glassware most associated with Citrusy flavor")
    print("=" * 60)
    glass_counts, cypher5 = kg.get_glassware_by_flavor("Citrusy")
    print(f"  Cypher: {cypher5}")
    print(f"  Result: {glass_counts}")
