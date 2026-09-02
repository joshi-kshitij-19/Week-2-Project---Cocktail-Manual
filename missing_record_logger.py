"""
missing_record_logger.py
-------------------------
Phase 2 / Item 6 follow-on: "Corpus Completeness" data flywheel.

Purely a LOGGING mechanism (per explicit design decision) -- it does NOT
fetch anything externally and does NOT change what the user sees. It answers
the exact question the user asked: "we log what is missing and eventually
add it to the corpus manually." Nothing more.

This is Option 3 from the "3 production strategies for missing records"
already documented in Things_to_Ponder.md Section 2 and
Learnings from Project 1.md's "Screwdriver Discovery" lesson -- turning
every retrieval gap into a data-driven curation signal instead of a guess.

Wired into the THREE places a "gap" can currently surface in this pipeline:
  1. refuse_node()          -- hardcoded out-of-domain refusal
  2. generate_node()        -- LLM decides on its own the retrieved chunks
                                don't actually answer the question (the
                                REFUSAL_CLAUSE appears in its own output,
                                even though grade_relevance_node let it
                                through as "relevant")
  3. Agentic RAG's DECIDE    -- "no such variant/match" conclusions from
     step (agentic_rag_pipeline.py)

Each log entry is one JSON line (JSONL format) appended to
missing_records_log.jsonl, so it can be reviewed later with any text editor,
`jq`, or loaded into a notebook -- no database needed.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

LOG_FILE_PATH = "/Users/kshitijjoshi/Downloads/Random/missing_records_log.jsonl"

# Set to True (or set env var MISSING_RECORD_LOGGER_VERBOSE=1) if you want the
# "Logged gap (...)" confirmation printed to the console again -- e.g. while
# debugging the logger itself. Silenced by default so live demos aren't
# interrupted by a message that's purely a background side-effect and isn't
# meant for the end user to see.
VERBOSE = os.getenv("MISSING_RECORD_LOGGER_VERBOSE", "0") == "1"

# The exact refusal clause from SYSTEM_PROMPT (rag_generation_and_pipeline.py)
# -- used to detect when generate_node's OWN LLM call decided to refuse,
# even though grade_relevance_node had classified the query as "relevant".
REFUSAL_CLAUSE = (
    "I am sorry, but I can only answer questions related to cocktail recipes, "
    "mixology techniques, and ingredients present in our cocktail manual."
)


def log_missing_record(
    query: str,
    reason: str,
    top_score: Optional[float] = None,
    source_node: str = "",
) -> None:
    """
    Appends one structured record to missing_records_log.jsonl.

    reason: one of "refuse_node" | "llm_self_refusal" | "structured_fallback_low_confidence" | "agentic_no_match"
    top_score: the best hybrid_score seen for this query, if any (helps
               distinguish "close but not quite" misses from "totally
               unrelated" queries when reviewing the log later).

    Defensive by design: this is a side-effect logging feature, not part of
    the answer-generation path. A failure here (e.g. a filesystem permission
    issue) must NEVER crash the pipeline or block the user from getting their
    answer -- caught and reported to stderr/stdout only, exactly like the
    existing API-call try/except fallback pattern used elsewhere in this
    codebase (e.g. embedding_selection.py, rag_generation_and_pipeline.py).
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "reason": reason,
        "top_score": top_score,
        "source_node": source_node,
    }
    try:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if VERBOSE:
            print(f"📝 [Missing Record Logger] Logged gap ({reason}): '{query}' -> {LOG_FILE_PATH}")
    except OSError as e:
        if VERBOSE:
            print(f"⚠️  [Missing Record Logger] Could not write to log file ({e}); continuing without logging this gap.")


def read_missing_records_log() -> list:
    """Reads back all logged entries -- useful for a future review/curation script."""
    if not os.path.exists(LOG_FILE_PATH):
        return []
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
