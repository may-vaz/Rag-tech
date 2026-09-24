"""
step1_retrieve.py
====================
The FIRST of two separate process runs for evaluation. Loads the
embedding model, reranker, and both indexes; runs retrieval + reranking
for every test question; writes the results to eval/prepared.json; and
then EXITS.

WHY THIS IS A SEPARATE PROCESS, NOT JUST A SEPARATE FUNCTION
------------------------------------------------------------------
Testing run_eval.py's single-process, phased (retrieve-all-then-
generate-all) design on the actual 8GB target machine showed it wasn't
enough: even after explicitly `del`-ing the embedding/reranker objects
and calling `gc.collect()`, a SECOND back-to-back Ollama call still hit
a ReadTimeout. The reason: `gc.collect()` only releases Python-level
object references. torch and faiss both use their own native memory
allocators, which routinely keep freed memory reserved in their own
internal pool rather than returning it to the operating system -- this
is standard, documented behavior for both libraries, not a bug. So the
process's actual memory footprint (as the OS sees it) can stay elevated
long after Python considers the objects "freed."

The only way to GUARANTEE the OS reclaims all of a process's memory is
for that process to actually exit. Splitting retrieval and generation
into two separate `python3` invocations, with step 1 finishing and
exiting completely before step 2 starts, is the real fix -- not a
heavier version of the same in-process trick.

USAGE
------
    python3 eval/step1_retrieve.py eval/qa_testset.json data/index

Writes eval/prepared.json, then exits. Run eval/step2_generate.py next.

Also computes retrieval-quality metrics here (Precision@k, MRR, NDCG@k
-- see eval/scoring.py for what each measures and why), since the full
ranked candidate list only exists at this stage, before step2 ever
runs. These are stored per-question in prepared.json and carried
through to the final results.json by step2.

ENGINE-FIRST ROUTING (added after finding a real gap): answer.py now
tries generate/fact_engine.py's deterministic computation BEFORE
retrieval for any question it can prove (sum/difference/percent_change/
ratio over table facts) -- but this eval script was calling
retrieve_context_with_ranking() directly, bypassing that check entirely.
That meant the eval harness was silently testing the OLD LLM-extraction
path for every computation question, never the new engine, even though
production (answer.py) uses the engine first. Fixed the same way
answer.py does it: try the engine for every question BEFORE loading any
retrieval model; only questions the engine declines go through
retrieval at all. If the engine answers every question, this script
skips loading the embedding/reranker models entirely, exactly matching
answer.py's own memory-conscious behavior.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Engine-first check -- stdlib-only (json/re), safe to import before
# anything torch/faiss-related loads.
from generate.fact_engine import load_facts, try_answer  # noqa: E402
from generate.answer_integration import FACTS_PATH  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import check_retrieval_hit, mrr, ndcg_at_k, precision_at_k  # noqa: E402


def main(testset_path: str, index_dir: str, out_path: str) -> None:
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)
    print(f"Loaded {len(testset)} test cases from {testset_path}")

    # PHASE 0 -- try the deterministic fact engine for every question
    # FIRST, before loading any retrieval model. Mirrors answer.py's own
    # try_engine_first() ordering exactly, so the eval harness reflects
    # what production actually does.
    print("Trying deterministic fact engine first (no models needed)...")
    try:
        facts = load_facts(FACTS_PATH)
    except (OSError, ValueError):
        facts = []
    if not facts:
        print(f"  no facts found at {FACTS_PATH} (run the new parse_pdf.py first) -- "
              f"all questions will go through retrieval.")

    engine_hits: dict[str, dict] = {}
    remaining_cases = []
    for case in testset:
        hit = try_answer(case["question"], facts) if facts else None
        if hit is not None:
            engine_hits[case["id"]] = hit
        else:
            remaining_cases.append(case)
    print(f"  fact engine answered {len(engine_hits)} of {len(testset)} question(s); "
          f"{len(remaining_cases)} need retrieval.")

    prepared = []
    for case in testset:
        if case["id"] in engine_hits:
            hit = engine_hits[case["id"]]
            engine_pages = sorted({s.get("page") for s in hit.get("sources", []) if s.get("page")})
            retrieval_hit = check_retrieval_hit(engine_pages, case["expected_pages"])
            prepared.append({
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "context": None,
                "engine_answer": hit["answer"],   # step2 uses this directly, no LLM call
                "retrieved_pages": engine_pages,
                "ranked_pages": engine_pages,
                "retrieval_hit": retrieval_hit,
                "retrieval_precision_at_k": 1.0 if retrieval_hit else 0.0,
                "retrieval_mrr": 1.0 if retrieval_hit else 0.0,
                "retrieval_ndcg_at_k": 1.0 if retrieval_hit else 0.0,
                "expected_pages": case["expected_pages"],
                "expected_answer_contains": case["expected_answer_contains"],
                "expected_refusal": case.get("expected_refusal", False),
                "reference_answer": case.get("reference_answer", ""),
            })
            print(f"  [ENGINE] {case['id']}: {hit['answer'][:80]}")

    if not remaining_cases:
        print("\nFact engine answered every question -- skipping model loading entirely.")
    else:
        # Save engine hits to disk NOW, before attempting retrieval -- if
        # the retrieval phase fails for any reason (a missing index, an
        # out-of-memory crash, anything), the fast, already-computed
        # engine answers are not lost along with it. Found necessary by
        # testing: an earlier version only wrote prepared.json once, at
        # the very end, and a retrieval-phase crash silently discarded
        # every engine answer that had already succeeded.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prepared, f, indent=2)
        print(f"\n({len(prepared)} engine answer(s) saved to {out_path} before attempting retrieval, "
              f"in case the retrieval phase below fails.)")

        print("Loading dense + sparse indexes, embedding model, and reranker "
              f"for the {len(remaining_cases)} remaining question(s)...")
        from generate.answer import retrieve_context_with_ranking  # noqa: E402  (deferred: heavy import)
        from retrieve.hybrid_retriever import load_retriever  # noqa: E402
        from retrieve.reranker import load_reranker  # noqa: E402

        dense_index, dense_chunks, embedding_model, bm25, sparse_chunks = load_retriever(index_dir)
        reranker = load_reranker()

        print("\nRunning retrieval + reranking for the remaining test questions...")
        for case in remaining_cases:
            context, used_chunks, ranked_chunks = retrieve_context_with_ranking(
                case["question"], dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
            )
            retrieved_pages = sorted({c["page_number"] for c in used_chunks})
            ranked_pages = [c["page_number"] for c in ranked_chunks]
            expected_pages = case["expected_pages"]

            retrieval_hit = check_retrieval_hit(retrieved_pages, expected_pages)
            p_at_k = precision_at_k(ranked_pages, expected_pages)
            mrr_score = mrr(ranked_pages, expected_pages)
            ndcg_score = ndcg_at_k(ranked_pages, expected_pages)

            prepared.append({
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "context": context,   # None if nothing survived the relevance filter
                "engine_answer": None,
                "retrieved_pages": retrieved_pages,
                "ranked_pages": ranked_pages,
                "retrieval_hit": retrieval_hit,
                "retrieval_precision_at_k": p_at_k,
                "retrieval_mrr": mrr_score,
                "retrieval_ndcg_at_k": ndcg_score,
                "expected_pages": expected_pages,
                "expected_answer_contains": case["expected_answer_contains"],
                "expected_refusal": case.get("expected_refusal", False),
                "reference_answer": case.get("reference_answer", ""),
            })
            status = "HIT " if retrieval_hit else "MISS"
            print(f"  [{status}] {case['id']}: ranked pages {ranked_pages}  "
                  f"(P@k={p_at_k:.2f}, MRR={mrr_score:.2f}, NDCG@k={ndcg_score:.2f})")

            # Save after EVERY question, not just at the end -- same
            # resilience principle as the pre-retrieval save above and as
            # step2_generate.py's own per-question save: one slow or
            # crashing question shouldn't cost the results already computed.
            order = {c["id"]: i for i, c in enumerate(testset)}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sorted(prepared, key=lambda p: order[p["id"]]), f, indent=2)

    # Keep prepared[] in the same order as the input test set.
    order = {case["id"]: i for i, case in enumerate(testset)}
    prepared.sort(key=lambda p: order[p["id"]])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prepared, f, indent=2)

    n_hits = sum(p["retrieval_hit"] for p in prepared)
    mean_mrr = sum(p["retrieval_mrr"] for p in prepared) / len(prepared)
    mean_ndcg = sum(p["retrieval_ndcg_at_k"] for p in prepared) / len(prepared)
    print(f"\nRetrieval/engine hit rate: {n_hits}/{len(prepared)}   Mean MRR: {mean_mrr:.3f}   "
          f"Mean NDCG@k: {mean_ndcg:.3f}")
    print(f"Prepared contexts saved to {out_path}")
    print("This process will now exit, fully releasing all model memory. "
          "Run eval/step2_generate.py next.")


if __name__ == "__main__":
    testset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/qa_testset.json"
    index_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "eval/prepared.json"
    main(testset_path, index_dir, out_path)