"""
answer.py
==========
Stage 7, the final stage of the pipeline: takes a user question, runs it
through retrieve_and_rerank() (hybrid retrieval + cross-encoder rerank),
filters the result down to genuinely relevant chunks, builds a grounded
prompt, and calls a local LLM (via Ollama) to produce a cited answer.

WHY A RELEVANCE-SCORE CUTOFF BEFORE GENERATION (not just top-k)
--------------------------------------------------------------------
Running reranker.py against this project's own test query
("0.500% Notes due 2031") produced a concrete, measured finding: the
correct chunk scored 0.901, while the next-best candidate scored 0.007
-- roughly a 130x gap. Blindly sending "top 3" or "top 5" to the LLM
regardless of score would hand it two near-irrelevant chunks alongside
the one real answer. This isn't a hypothetical -- it's the literal
retrieved result set from this project's own evaluation, and it matches
a well-documented problem for LLM context windows generally: irrelevant
context measurably degrades answer quality even when a genuinely
correct chunk is also present (often called "context rot" -- performance
drops as more low-relevance material sits in the context, even before
the window fills up).

The fix here is RELATIVE score filtering, not a fixed count: keep any
chunk scoring at least RELATIVE_SCORE_CUTOFF of the top chunk's score
(default 0.15 -- chosen so the query-2 case above cleanly keeps only
the one real answer, while the query-1 case, where the top 3 scores are
all within a hair of each other at ~0.998-0.999, keeps all three,
since they're all genuinely relevant). We always keep at least the #1
result, even if reranking gives everything a low absolute score, so
the system never sends the LLM literally nothing.

WHY THE PROMPT EXPLICITLY WARNS ABOUT MULTIPLE FISCAL PERIODS
--------------------------------------------------------------------
This is the ambiguity risk documented throughout this project (see
enrich_metadata.py, chunk.py): this filing reports nearly every metric
across four overlapping periods (Q3 2022, Q3 2021, 9mo 2022, 9mo 2021).
Retrieval and reranking do their job by surfacing the right TABLE, but
nothing before this stage stops an LLM from picking one column out of
four and presenting it as *the* answer if the user's question didn't
specify a period. The system prompt below makes this an explicit,
un-skippable instruction rather than hoping the model figures it out.

WHY THE PROMPT ALSO REQUIRES A "NOT FOUND" ESCAPE HATCH
--------------------------------------------------------------
The single most important anti-hallucination measure for a RAG system
answering questions about a SPECIFIC document is giving the model
explicit permission -- really, an instruction -- to say the document
doesn't contain the answer, rather than filling the gap with outside
knowledge (e.g. answering from general knowledge about Apple rather
than THIS filing). This is worth testing directly: this file's demo
section includes an out-of-document query ("Q4 2022 net sales" --
this filing only covers through Q3 2022) specifically to verify the
system correctly refuses rather than hallucinates a plausible-sounding
number.

LLM CHOICE -- local, via Ollama
------------------------------------------------------
Default model is `qwen3:4b`: at Q4 quantization it's roughly 2.5GB, which
leaves real headroom on an 8GB-RAM machine (a 7B+ model is tight to
run at all alongside an OS, browser, and editor, based on this
project's own experience with the embedding/reranker model sizes), and
it's specifically noted for strong instruction-following relative to
its size -- which matters here because this prompt has non-negotiable
rules (cite pages, disambiguate periods, refuse when ungrounded). `llama3.2:3b` is
noted below as a lighter fallback if `qwen3:4b` runs too slowly on a
given machine -- swapping OLLAMA_MODEL is a one-line change, nothing
else in this file depends on which model is used.

PREREQUISITE: Ollama must be running locally (`ollama serve`, or just
having the Ollama desktop app open) with the model pulled:
    ollama pull qwen3:4b

A NOTE ON 8GB MACHINES (a real finding from testing this project)
--------------------------------------------------------------------------
Running this on an 8GB Mac produced `requests.ReadTimeout` errors: this
process holds bge-large-en-v1.5 (~1.3GB) and bge-reranker-v2-m3 (~2.3GB)
in memory via PyTorch, while Ollama runs qwen3:4b (~2.5GB) as a SEPARATE
OS process -- on 8GB total, both compete for the same limited RAM.

This file's __main__ block frees the embedding/reranker models with
`del` + `gc.collect()` before calling Ollama, which is enough for
answering ONE question (the intended interactive use of this script --
see the third CLI argument below). It is NOT reliably enough for many
back-to-back generations in the same process: testing showed a second
call can still time out, because `gc.collect()` only releases Python's
own references -- torch and faiss keep their own native memory pools
that don't necessarily return memory to the OS just because Python let
go of it. The only fully reliable fix for BATCH generation (many
questions in one run) is running retrieval and generation as two
genuinely separate process invocations, so the OS guarantees a full
memory reclaim between them -- see eval/step1_retrieve.py and
eval/step2_generate.py, which do exactly this and are the right tool
for testing more than one question reliably on this hardware. This
file's own multi-question demo path (no CLI question argument given)
is kept for convenience on machines with more RAM, but on an 8GB
machine, prefer passing a single question here, or use the eval/ two-
step scripts for anything more than one.

DETERMINISTIC COMPUTATION
--------------------------------------------------------------------------
Computation questions ("combined iPhone and Mac", "difference between
Q3 2022 and Q3 2021", ...) are  attempted FIRST by
generate/fact_engine.py, a pure, millisecond, no-model function over
data/table_facts.jsonl (emitted by parse_pdf.py). It returns a
hit ONLY when it can prove every operand (row + period + units); the
hit is returned immediately with NO retrieval, NO reranking, and NO
LLM call. Anything else like text answers, single-figure lookups,
ambiguous or out-of-scope questions returns None and falls through
to the retrieval + LLM path.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Same macOS OpenMP-conflict fix used throughout this project's other
# stages -- must be set before numpy/torch (imported transitively via
# the retrieval stage) are loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from retrieve.hybrid_retriever import hybrid_search, load_retriever  # noqa: E402
from retrieve.reranker import load_reranker, rerank, retrieve_and_rerank  
from generate.llm_client import (  
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    call_ollama,
    detect_computation_operation,
)
# NEW: deterministic computation engine (stdlib-only -- json/re, no torch,
# no faiss, no requests). Imported AFTER the sys.path bootstrap above, same
# `from generate....` convention as every other local import in this file.
from generate.fact_engine import try_answer  # noqa: E402
from generate.answer_integration import (  # noqa: E402
    FACTS_PATH,
    format_engine_hit,
    load_facts_once,
)

RELATIVE_SCORE_CUTOFF = 0.15        # see module docstring: keep chunks scoring >= 15% of the top score
COMPUTATION_SCORE_CUTOFF = 0.02     # MUCH looser floor used ONLY for computation questions. A
                                        # computation needs TWO (or more) operands that often live in
                                        # DIFFERENT chunks with very different rerank scores (e.g. the
                                        # income-statement table scores 0.90 while the segment table
                                        # holding the other needed number scores 0.09). The 0.15
                                        # RELATIVE cutoff above was silently dropping the lower-scoring
                                        # operand's chunk before it ever reached the LLM -- the model
                                        # then correctly (but unhelpfully) reported "insufficient
                                        # evidence" because, from its view, the number genuinely wasn't
                                        # in its context. This is safe to loosen because the computation
                                        # LLM call already has its own honesty check
                                        # (sufficient_evidence=false) and its own per-value grounding
                                        # check (_value_grounded) -- an irrelevant extra chunk here can
                                        # only be ignored by those checks, never hallucinated from.
K_CANDIDATES = 15                    # how many hybrid-retrieved candidates the reranker sees, for an
                                        # ordinary LOOKUP question (one fact, one place in the document)
K_CANDIDATES_COMPUTATION = 40       # WIDER candidate pool for a computation question -- see module
                                        # docstring below for the real, measured reason this exists.
K_RERANKED = 5                        # max chunks kept after reranking, BEFORE the score cutoff is applied
K_RERANKED_COMPUTATION = 8           # slightly wider for computation questions -- the two (or more)
                                        # values needed can legitimately live in different chunks/pages,
                                        # so a bit more room in the final context reduces the risk of one
                                        # of them being cut even after the wider initial candidate pool
                                        # above correctly surfaces it.

# Comparison/period phrases stripped from a computation question to
# build a SECOND, simpler retrieval query -- see _simplify_query_for_retrieval()
# below for why this exists (a real, measured retrieval fix, not a guess).
_QUERY_STRIP_PATTERNS = [
    r"\bdifference between\b", r"\bdifference in\b", r"\bcombined\b", r"\bsum of\b", r"\btotal of\b",
    r"\bpercent(age)?\s+change\b", r"\b%\s*change\b", r"\bgrowth rate\b",
    r"\byear[- ]over[- ]year\b", r"\byoy\b", r"\bthe change in\b", r"\bcompared to\b",
    r"\b(three|six|nine|twelve)\s+months?\s+ended\b",
    r"\b(first|second|third|fourth)\s+quarter\b", r"\bq[1-4]\b",
    r"\bfiscal\s+\d{4}\b", r"\b\d{4}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2},?\b",
    r"\bin the\b", r"\bfor the\b", r"\bwhat was\b", r"\bwhat were\b",
]
_QUERY_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at",
                     "is", "was", "were", "s", "for", "that", "this"}


def _simplify_query_for_retrieval(question: str) -> str:
    """
    Strips comparison/period language from a computation question,
    leaving roughly just the entity + metric (e.g. "Apple net sales").

    WHY THIS EXISTS -- a real, measured retrieval failure, not a
    hypothetical: testing the exact question "What was the difference
    between Apple's net sales in the third quarter of 2022 and the
    third quarter of 2021" against this project's own BM25 index showed
    the table with the actual needed figures ranking 53rd out of 95
    chunks. This filing's MD&A section repeats "net sales... third
    quarter of 2022... compared to the third quarter of 2021" dozens of
    times across narrative commentary for every segment and product
    category, which outranks the one compact table that actually has
    the numbers. Simplifying the SAME query down to just "Apple net
    sales" moved that table's rank from 53rd to 13th -- a direct,
    verified improvement, not a theoretical one. This function is used
    to run a SECOND retrieval pass (see retrieve_context_with_ranking)
    alongside the original question, merging both result sets before
    reranking -- reranking still scores against the ORIGINAL question,
    so answer relevance isn't affected, only which candidates get a
    chance to be considered.

    General, not hardcoded to this specific question: the strip list is
    comparison/date PATTERNS (quarter phrases, month names, years,
    operation-trigger words), not any specific number or fact, so this
    applies the same way to any future computation question.
    """
    text = question
    for pattern in _QUERY_STRIP_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w.lower() not in _QUERY_STOPWORDS and len(w) > 1]
    return " ".join(words)


def filter_by_relative_score(chunks: list[dict], cutoff: float = RELATIVE_SCORE_CUTOFF) -> list[dict]:
    """
    Keeps chunks scoring at least `cutoff` fraction of the top chunk's
    rerank score. Always keeps at least the #1 result. See module
    docstring for why this is relative, not a fixed top-k count.
    """
    if not chunks:
        return []
    top_score = chunks[0]["_rerank_score"]
    if top_score <= 0:
        return chunks[:1]  # degenerate case (e.g. all-negative scores) -- still return something
    kept = [c for c in chunks if c["_rerank_score"] >= top_score * cutoff]
    return kept if kept else chunks[:1]


def build_context(chunks: list[dict]) -> str:
    """
    Formats surviving chunks into a labeled context block. Each chunk's
    FULL generation_text is included (not search_text) -- see
    reranker.py / build_dense_index.py for why these two fields exist
    and differ; generation_text is the one meant for the LLM to read.

    STARTS WITH A PERIOD-SUMMARY LINE listing the union of fiscal
    periods represented across all included chunks (using the
    `fiscal_periods` tags enrich_metadata.py already computes for every
    chunk -- see that file for how these are detected). This is a
    direct, general fix for a real failure found in testing: asked
    about a period this filing never reports (Q4 2022), the LLM did not
    refuse -- it confidently restated a REAL number from a DIFFERENT,
    retrieved period as if it answered the question ("Temporal Semantic
    Confusion," a documented failure mode in financial-LLM research --
    see llm_client.py's module docstring for the citations). A
    deterministic number-grounding check alone cannot catch this,
    because the number really is in the document, just for the wrong
    period. Giving the model an explicit, structured list of which
    periods are actually present -- generated from this document's own
    metadata, not hardcoded to any specific period -- lets it check the
    question's requested period against reality directly, and the
    system prompt (see llm_client.py, rule 8) tells it exactly what to
    do when a requested period isn't in this list. This generalizes to
    ANY future out-of-scope period question, not just Q4 2022.
    """
    all_periods: set[str] = set()
    for c in chunks:
        all_periods.update(c.get("fiscal_periods", []))
    period_line = (
        f"[Periods represented in retrieved excerpts: {', '.join(sorted(all_periods))}]"
        if all_periods
        else "[No specific reporting period could be identified in the retrieved excerpts.]"
    )

    blocks = [period_line]
    for i, c in enumerate(chunks, start=1):
        label = f"[Excerpt {i} -- page {c['page_number']}"
        if c.get("section"):
            label += f", from '{c['section']}'"
        label += f", relevance score {c['_rerank_score']:.3f}]"
        blocks.append(f"{label}\n{c['generation_text']}")
    return "\n\n---\n\n".join(blocks)


def retrieve_context(
    query: str,
    dense_index,
    dense_chunks: list[dict],
    embedding_model,
    bm25,
    sparse_chunks: list[dict],
    reranker,
) -> tuple[str | None, list[dict]]:
    """
    Retrieval + reranking + filtering only -- NO call to Ollama. Split
    out from answer_question() specifically so the caller can free the
    embedding/reranker models from memory BEFORE calling the LLM (see
    "A NOTE ON 8GB MACHINES" in the module docstring, and the __main__
    block below: on a memory-constrained machine, this process and
    Ollama's own qwen3:4b process compete for the same RAM the entire
    time a naive single-function version would hold everything at once).

    Returns (context_text_or_None, used_chunks). context is None if no
    chunks survived the relevance filter -- the caller should skip
    calling the LLM in that case (see answer_question for the exact
    "no relevant information" handling).
    """
    context, used_chunks, _ranked = retrieve_context_with_ranking(
        query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
    )
    return context, used_chunks


def retrieve_context_with_ranking(
    query: str,
    dense_index,
    dense_chunks: list[dict],
    embedding_model,
    bm25,
    sparse_chunks: list[dict],
    reranker,
) -> tuple[str | None, list[dict], list[dict]]:
    """
    Same as retrieve_context(), but ALSO returns the full reranked
    candidate list (up to K_RERANKED items, in rank order, BEFORE the
    relative-score cutoff is applied) -- added specifically so eval/
    step1_retrieve.py can compute proper ranked-retrieval metrics
    (Precision@k, MRR, NDCG@k; see eval/scoring.py) using the actual
    order the system produced, not just the final filtered set.

    retrieve_context() above is kept calling this internally rather
    than being reimplemented separately, so there's exactly one place
    the retrieve-rerank-filter logic lives; nothing that already calls
    retrieve_context() needs to change.

    WIDER CANDIDATE POOL FOR COMPUTATION QUESTIONS -- a real, measured
    fix, not a guess. Testing the exact query "What was the difference
    between Apple's net sales in the third quarter of 2022 and the
    third quarter of 2021" against this project's own BM25 index showed
    the table containing the actual needed figures ranked 53rd out of
    95 chunks -- effectively invisible. The reason: this filing's MD&A
    section repeats "net sales... third quarter of 2022... compared to
    the third quarter of 2021" dozens of times across narrative
    commentary for every segment and product category (Americas, Rest
    of Asia Pacific, iPad Pro, etc.), all of which outrank the one
    compact table that actually has the total figures, because that
    table has almost no surrounding prose to match against. No amount
    of prompt engineering in llm_client.py can fix this -- the data
    never reaches the LLM at all if it isn't retrieved.

    WIDER CANDIDATE POOL *AND* A SECOND, SIMPLIFIED QUERY FOR COMPUTATION
    QUESTIONS -- a real, measured fix, not a guess. Testing the exact
    query "What was the difference between Apple's net sales in the
    third quarter of 2022 and the third quarter of 2021" against this
    project's own BM25 index showed the table containing the actual
    needed figures ranked 53rd out of 95 chunks -- effectively
    invisible. The reason: this filing's MD&A section repeats "net
    sales... third quarter of 2022... compared to the third quarter of
    2021" dozens of times across narrative commentary for every segment
    and product category (Americas, Rest of Asia Pacific, iPad Pro,
    etc.), all of which outrank the one compact table that actually has
    the total figures, because that table has almost no surrounding
    prose to match against. No amount of prompt engineering in
    llm_client.py can fix this -- the data never reaches the LLM at all
    if it isn't retrieved.

    Widening K_CANDIDATES alone was tested and found NOT sufficient on
    its own: the needed table's BM25 rank (53rd) is still outside even
    a widened pool. What actually works, verified directly: a SECOND,
    simplified retrieval query with comparison/period language stripped
    (see _simplify_query_for_retrieval) moves the same table from rank
    53rd to 13th -- comfortably within reach. So for a computation
    question, this function runs hybrid_search TWICE -- once with the
    original question, once with the simplified query -- and merges the
    two candidate pools (deduplicated by chunk_id) before reranking.
    Reranking still scores against the ORIGINAL question, so this only
    affects which candidates get a chance to be considered, not which
    one is judged most relevant. Ordinary lookup questions are
    completely unaffected -- they still take the single, original
    retrieve_and_rerank() call, unchanged, so this cannot regress
    anything that already worked.

    Returns (context_text_or_None, used_chunks, ranked_chunks) where
    ranked_chunks is the pre-cutoff reranked list and used_chunks is
    the post-cutoff subset actually used to build context (as before).
    """
    operation = detect_computation_operation(query)

    if operation:
        simplified = _simplify_query_for_retrieval(query)
        candidates_by_id: dict[str, dict] = {}
        for q in ([query, simplified] if simplified and simplified != query else [query]):
            for c in hybrid_search(
                q, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks,
                k_final=K_CANDIDATES_COMPUTATION,
            ):
                candidates_by_id.setdefault(c["chunk_id"], c)  # first occurrence wins; order doesn't
                                                                     # matter since reranking re-sorts everything
        reranked = rerank(query, list(candidates_by_id.values()), reranker, top_k=K_RERANKED_COMPUTATION)
    else:
        reranked = retrieve_and_rerank(
            query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker,
            k_candidates=K_CANDIDATES, k_final=K_RERANKED,
        )
    # Computation questions use a much looser cutoff -- see
    # COMPUTATION_SCORE_CUTOFF above for why a tight relative cutoff
    # (correct for single-fact lookups) actively breaks multi-operand
    # arithmetic by dropping the lower-scoring operand's chunk. Lookup
    # questions are completely unaffected -- they still take the
    # original RELATIVE_SCORE_CUTOFF, unchanged.
    cutoff = COMPUTATION_SCORE_CUTOFF if operation else RELATIVE_SCORE_CUTOFF
    used_chunks = filter_by_relative_score(reranked, cutoff=cutoff)
    if not used_chunks:
        return None, [], reranked
    return build_context(used_chunks), used_chunks, reranked


def try_engine_first(query: str) -> dict | None:
    """
    NEW -- deterministic computation attempt (no models, no retrieval,
    milliseconds). Returns a fully formatted {"answer", "sources",
    "method"} hit, or None when the engine declines (text answers,
    single-figure lookups, ambiguous or out-of-scope questions).

    Separated from answer_question() so the __main__ runner below can
    call it BEFORE loading any embedding/reranker models -- and so
    eval/step2_generate.py can reuse it without importing this module's
    torch/faiss-dependent retrieval path. Also returns None (instead of
    raising) when data/table_facts.jsonl is missing -- e.g. the parse
    stage hasn't been re-run yet -- so a stale checkout degrades to the
    old LLM-only behavior rather than crashing.
    """
    try:
        facts = load_facts_once(FACTS_PATH)
    except (OSError, ValueError):
        return None
    if not facts:
        return None
    hit = try_answer(query, facts)
    if hit is None:
        return None
    return format_engine_hit(hit)


def answer_question(
    query: str,
    dense_index,
    dense_chunks: list[dict],
    embedding_model,
    bm25,
    sparse_chunks: list[dict],
    reranker,
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Convenience wrapper for a single, one-off question (e.g. from a test
    script or an eval harness): retrieve_context() + call_ollama() in
    one call. For the CLI runner below, which needs to free memory
    between the retrieval and generation phases on constrained hardware,
    those two steps are called separately instead of through this
    function -- see the __main__ block.

    NEW: tries the deterministic fact engine FIRST (no models needed).
    Everything below that block is the original behavior, unchanged.
    """
    # --- NEW: deterministic computation first (pure function, ~ms). ---
    # Returns None for anything it cannot prove -- text answers, single
    # lookups, and ambiguous cases fall through untouched.
    engine_hit = try_engine_first(query)
    if engine_hit is not None:
        return engine_hit

    context, used_chunks = retrieve_context(
        query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
    )
    if context is None:
        return {
            "answer": "This filing does not contain relevant information to answer this question.",
            "sources": [],
        }

    user_prompt = f"CONTEXT:\n\n{context}\n\nQUESTION: {query}"
    answer_text = call_ollama(SYSTEM_PROMPT, user_prompt, model=model)

    return {
        "answer": answer_text,
        "sources": [
            {
                "page": c["page_number"],
                "section": c.get("section"),
                "rerank_score": c["_rerank_score"],
                "chunk_type": c["chunk_type"],
            }
            for c in used_chunks
        ],
    }


if __name__ == "__main__":
    import gc

    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/index"
    model = sys.argv[2] if len(sys.argv) > 2 else OLLAMA_MODEL
    single_question = sys.argv[3] if len(sys.argv) > 3 else None

    if single_question:
        queries = [single_question]
    else:
        queries = [
            # NEW: the two computation questions, answered by the fact
            # engine with no models and no LLM call.
            "What was the combined net sales of iPhone and Mac for the third quarter of 2022?",
            "What was the difference between Apple's net sales in the third quarter of 2022 and the third quarter of 2021?",
            "What were Apple's total net sales for the nine months ended June 25, 2022?",
            "What is the interest rate on Apple's Notes due 2031?",
            "What were Apple's net sales?",
            "What were Apple's net sales in Q4 2022?",
        ]

    # PHASE 0 -- NEW: deterministic fact-engine pass over ALL queries,
    # BEFORE any model is loaded. The engine needs no torch, no faiss,
    # no Ollama -- just data/table_facts.jsonl -- so anything it answers
    # costs milliseconds and zero RAM. Only declined queries proceed to
    # the retrieval phases below.
    print("Trying deterministic fact engine first (no models needed)...")
    engine_hits: dict[str, dict] = {}
    llm_queries: list[str] = []
    for query in queries:
        hit = try_engine_first(query)
        if hit is not None:
            engine_hits[query] = hit
        else:
            llm_queries.append(query)
    if engine_hits:
        print(f"  fact engine answered {len(engine_hits)} question(s); "
              f"{len(llm_queries)} go to retrieval+LLM.")
    else:
        print("  fact engine declined everything; all questions go to retrieval+LLM.")

    # PHASE 1 -- retrieval + reranking for the REMAINING queries only,
    # while the embedding/reranker models are loaded. No Ollama calls
    # happen yet. (If the engine answered everything, no models are
    # loaded at all.)
    prepared = []  # list of (query, context_or_None, used_chunks)
    if llm_queries:
        print("Loading dense + sparse indexes, embedding model, and reranker...")
        dense_index, dense_chunks, embedding_model, bm25, sparse_chunks = load_retriever(index_dir)
        reranker = load_reranker()

        print("\nRetrieving and reranking context for all questions...")
        for query in llm_queries:
            context, used_chunks = retrieve_context(
                query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
            )
            prepared.append((query, context, used_chunks))

        # PHASE 2 -- explicitly free the embedding/reranker models (and the
        # indexes) from memory BEFORE any Ollama call. This is the actual
        # fix for the timeout found while testing this project on an 8GB
        # machine: this process and Ollama's own model process were
        # competing for the same RAM for the entire script lifetime, not
        # just during retrieval. Freeing here means Ollama gets the memory
        # headroom it needs for the part that's actually slow (generation),
        # instead of fighting our own already-loaded models for it.
        print("Retrieval complete -- releasing embedding/reranker models from memory before generation...")
        del dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
        gc.collect()
    else:
        print("Skipping model loading entirely -- nothing left to retrieve.")

    # PHASE 3 -- generation only. At this point the only thing resident
    # in this process is plain Python data (the prepared context
    # strings) plus the `requests` library -- no torch, no loaded model
    # weights. Engine-answered questions are printed directly (no LLM
    # call); the rest go to Ollama exactly as before.
    llm_results = {q: (c, u) for q, c, u in prepared}
    for query in queries:
        print(f"\n{'=' * 70}\nQUESTION: {query}\n{'=' * 70}")

        if query in engine_hits:
            ans = engine_hits[query]
            print(f"\nANSWER:\n{ans['answer']}")
            print(f"\nSOURCES USED ({len(ans['sources'])}) [fact engine -- no LLM call]:")
            for s in ans["sources"]:
                print(f"  page={s['page']}  type={s['chunk_type']}  "
                      f"score={s['rerank_score']:.3f}  caption={s.get('caption')}")
            continue

        context, used_chunks = llm_results[query]

        if context is None:
            print("\nANSWER:\nThis filing does not contain relevant information to answer this question.")
            continue

        user_prompt = f"CONTEXT:\n\n{context}\n\nQUESTION: {query}"
        answer_text = call_ollama(SYSTEM_PROMPT, user_prompt, model=model)

        print(f"\nANSWER:\n{answer_text}")
        print(f"\nSOURCES USED ({len(used_chunks)}):")
        for c in used_chunks:
            print(f"  page={c['page_number']}  type={c['chunk_type']}  "
                  f"score={c['_rerank_score']:.3f}  section={c.get('section')}")
