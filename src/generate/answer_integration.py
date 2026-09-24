"""
answer_integration.py -- engine-first routing for answer.py (COMPLETE wiring)

"""
from __future__ import annotations

import functools
import re
from typing import Any

from generate.fact_engine import load_facts, try_answer  # noqa: E402
# NOTE: fixed from a bare "from fact_engine import ..." -- that only
# works when this file is run as (or imported by) something whose own
# directory Python already put on sys.path (e.g. running answer.py
# directly). It breaks the instant this module is imported from
# elsewhere (e.g. an eval script importing generate.answer_integration)
# -- the exact same class of bug already found and fixed in
# reranker.py's sibling import earlier in this project. Since this file
# lives in src/generate/, and callers already add src/ to sys.path
# before importing anything under generate.*, the package-qualified
# import resolves correctly in every calling context, not just one.

FACTS_PATH = "data/table_facts.jsonl"


@functools.lru_cache(maxsize=8)
def load_facts_once(path: str = FACTS_PATH) -> list[dict]:
    """Cache parsed facts per path (807 facts load in ms; the cache makes
    repeated questions free). Call load_facts_once.cache_clear() after
    re-ingesting a filing."""
    return load_facts(path)


def format_engine_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Shape a fact_engine result like a normal pipeline answer.

    hit = {"answer": str, "sources": [{"page": int, "row": str, "col": str,
           "caption": str, ...}, ...]}
    """
    pages = sorted({s.get("page") for s in hit.get("sources", []) if s.get("page")})
    cites = "; ".join(
        f"p.{s.get('page')}: {s.get('row')} [{s.get('col')}]".strip()
        for s in hit.get("sources", [])
    )
    return {
        "answer": hit["answer"],
        "citations": cites,
        "pages": pages,
        "method": "fact_engine",
        "sources": hit.get("sources", []),
    }


def answer_with_engine_first(question: str,
                             llm_pipeline_answer,
                             facts_path: str = FACTS_PATH):
    """Drop-in wrapper. llm_pipeline_answer is your EXISTING answer
    callable (question -> answer); it is called ONLY when the engine
    refuses, so its behavior is preserved exactly.

    Usage:
        from answer_integration import answer_with_engine_first
        from answer import answer_question as old_answer   # your module
        answer = answer_with_engine_first(q, old_answer)
    """
    try:
        hit = try_answer(question, load_facts_once(facts_path))
    except Exception:
        hit = None  # never let the fast path break the slow path
    if hit is not None:
        return format_engine_hit(hit)
    return llm_pipeline_answer(question)


# --- Optional: route ONLY computation-shaped questions to the engine ---
# try_answer already refuses non-computation questions internally, so this
# pre-filter is redundant. It exists for callers that want to avoid even
# loading facts for plain lookups.
_COMPUTATION_HINT = re.compile(
    r"\b(combined|sum of|total of|plus|together|difference|change|increase|"
    r"decrease|growth|percent(age)?|ratio|share|proportion|minus|less|"
    r"versus|compared?)\b",
    re.I,
)


def looks_like_computation(question: str) -> bool:
    return bool(_COMPUTATION_HINT.search(question))