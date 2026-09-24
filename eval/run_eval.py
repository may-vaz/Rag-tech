"""
run_eval.py

Runs the full pipeline (retrieve -> rerank -> filter -> generate) against
the hand-built test set in qa_testset.json, and scores both retrieval
and the final answer -- producing a results table for the design doc's
evaluation section. Single-command option for machines with enough RAM;
on the 8GB target machine use step1_retrieve.py + step2_generate.py
instead (see "MEMORY" below).

WHY DETERMINISTIC CHECKS INSTEAD OF AN LLM JUDGE
---------------------------------------------------
Judge-style scoring (a second LLM grading faithfulness/relevance per
question) would add another model on top of the embedder, reranker, and
generation LLM already competing for memory -- the combination that
already produced a real ReadTimeout on this project's 8GB machine (see
answer.py). And for THIS corpus the judge buys nothing: ground truth is
exact numbers and known pages, so a substring check against a verified
figure ("58,107") and a page-overlap check are strictly more reliable
than asking a model whether an answer "seems faithful." Refusal cases
are checked for refusal language. If the corpus ever becomes open-ended
prose summaries instead of exact figures, a judge becomes worth its cost.

MEMORY: PHASED, WITH AN HONEST LIMIT
---------------------------------------
This script retrieves for every question first, frees the models with
`del` + gc.collect(), then generates. Real testing showed that ISN'T
fully sufficient on 8GB: the second generation call still timed out,
because torch/faiss native allocators keep freed memory in their own
pools instead of returning it to the OS -- only a process exit
guarantees reclaim. Hence the two-process split (step1 retrieves and
EXITS; step2 is a fresh process importing nothing but requests). This
file stays as the simpler single-command path where RAM allows.

ENGINE-FIRST (matches production): every question is tried against the
deterministic fact engine before any model loads (mirroring answer.py
and step1_retrieve.py); only declined ones go through retrieval.
Without this, this harness would score a different system than the one
shipped.
"""

from __future__ import annotations

import gc
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

from generate.answer import (  # noqa: E402
    SYSTEM_PROMPT,
    call_ollama,
    retrieve_context,
)
from generate.fact_engine import load_facts, try_answer  # noqa: E402
from generate.answer_integration import FACTS_PATH  # noqa: E402
from retrieve.hybrid_retriever import load_retriever  # noqa: E402
from retrieve.reranker import load_reranker  # noqa: E402


def load_testset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_retrieval_hit(used_chunks: list[dict], expected_pages: list[int]) -> bool:
    """True if at least one retrieved chunk's page matches a known-correct page.
    An empty expected_pages list (hallucination-refusal cases) trivially passes --
    there's no 'correct page' to find for a question with no answer in the document."""
    if not expected_pages:
        return True
    retrieved_pages = {c["page_number"] for c in used_chunks}
    return bool(retrieved_pages.intersection(expected_pages))


def check_answer_correct(answer: str, case: dict) -> bool:
    """
    For hallucination_refusal cases: pass if the answer contains refusal
    language. For everything else: pass if ALL expected substrings are
    present (case-insensitive) -- for the ambiguity_trap case, this
    means the answer must contain ALL FOUR period figures, since a
    correct disambiguated answer includes every one of them.
    """
    answer_lower = answer.lower()
    if case.get("expected_refusal"):
        return any(kw.lower() in answer_lower for kw in case["expected_answer_contains"])
    return all(kw.lower() in answer_lower for kw in case["expected_answer_contains"])


def run_eval(testset_path: str, index_dir: str, model: str) -> list[dict]:
    testset = load_testset(testset_path)

    print(f"Loaded {len(testset)} test cases from {testset_path}")

    # PHASE 0 -- engine-first, mirroring production (answer.py) and
    # step1_retrieve.py, so this harness scores the system actually shipped.
    print("Trying deterministic fact engine first (no models needed)...")
    try:
        facts = load_facts(FACTS_PATH)
    except (OSError, ValueError):
        facts = []
    if not facts:
        print(f"  no facts found at {FACTS_PATH} (run the new parse_pdf.py first) -- "
              f"all questions will go through retrieval.")
    engine_answers: dict[str, str] = {}
    engine_pages: dict[str, list[int]] = {}
    remaining = []
    for case in testset:
        hit = try_answer(case["question"], facts) if facts else None
        if hit is not None:
            engine_answers[case["id"]] = hit["answer"]
            engine_pages[case["id"]] = sorted({s.get("page") for s in hit.get("sources", []) if s.get("page")})
        else:
            remaining.append(case)
    print(f"  fact engine answered {len(engine_answers)} of {len(testset)}; "
          f"{len(remaining)} need retrieval.")

    prepared = []
    for case in testset:
        if case["id"] in engine_answers:
            # Pseudo-chunks (page numbers only) so the existing page-overlap
            # check and retrieval printout below work unchanged for engine hits.
            pseudo = [{"page_number": p} for p in engine_pages[case["id"]]]
            retrieval_hit = check_retrieval_hit(pseudo, case["expected_pages"])
            prepared.append((case, None, pseudo, retrieval_hit, engine_answers[case["id"]]))
            print(f"  [ENGINE] {case['id']}: {engine_answers[case['id']][:80]}")

    if remaining:
        print("Loading dense + sparse indexes, embedding model, and reranker...")
        dense_index, dense_chunks, embedding_model, bm25, sparse_chunks = load_retriever(index_dir)
        reranker = load_reranker()

        # PHASE 1: retrieval for the REMAINING cases only, while heavy models are loaded.
        print("\nRunning retrieval + reranking for all test questions...")
        for case in remaining:
            context, used_chunks = retrieve_context(
                case["question"], dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
            )
            retrieval_hit = check_retrieval_hit(used_chunks, case["expected_pages"])
            prepared.append((case, context, used_chunks, retrieval_hit, None))
            status = "HIT " if retrieval_hit else "MISS"
            print(f"  [{status}] {case['id']}: retrieved pages "
                  f"{sorted({c['page_number'] for c in used_chunks})}")

        # PHASE 2: free the embedding/reranker models before any LLM call.
        # See module docstring -- this exact ordering was required to avoid
        # a real ReadTimeout on this project's own 8GB test machine.
        print("\nReleasing embedding/reranker models before generation...")
        del dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
        gc.collect()
    else:
        print("Engine answered everything -- no models loaded.")

    # Keep prepared[] in the same order as the input test set.
    order = {case["id"]: i for i, case in enumerate(testset)}
    prepared.sort(key=lambda p: order[p[0]["id"]])

    # PHASE 3: generation + scoring for every test case.
    print("\nGenerating answers...\n")
    results = []
    for case, context, used_chunks, retrieval_hit, engine_answer in prepared:
        if engine_answer is not None:
            # Answered by the deterministic fact engine in PHASE 0 -- no
            # Ollama call needed. Scored identically to every other answer.
            answer_text = engine_answer
        elif context is None:
            answer_text = "This filing does not contain relevant information to answer this question."
        else:
            user_prompt = f"CONTEXT:\n\n{context}\n\nQUESTION: {case['question']}"
            answer_text = call_ollama(SYSTEM_PROMPT, user_prompt, model=model)

        answer_correct = check_answer_correct(answer_text, case)
        results.append({
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "method": "fact_engine" if engine_answer is not None else "llm",
            "retrieval_hit": retrieval_hit,
            "answer_correct": answer_correct,
            "answer": answer_text,
            "retrieved_pages": sorted({c["page_number"] for c in used_chunks}),
            "expected_pages": case["expected_pages"],
        })
        status = "PASS" if answer_correct else "FAIL"
        print(f"[{status}] {case['id']}")

    return results


def print_summary(results: list[dict]) -> None:
    n = len(results)
    retrieval_hits = sum(r["retrieval_hit"] for r in results)
    answer_correct = sum(r["answer_correct"] for r in results)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"Retrieval hit rate: {retrieval_hits}/{n} ({100 * retrieval_hits / n:.0f}%)")
    print(f"Answer correctness: {answer_correct}/{n} ({100 * answer_correct / n:.0f}%)")
    print()
    print(f"{'ID':<28} {'Type':<24} {'Retrieval':<10} {'Answer':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['id']:<28} {r['type']:<24} "
              f"{'HIT' if r['retrieval_hit'] else 'MISS':<10} "
              f"{'PASS' if r['answer_correct'] else 'FAIL':<8}")

    failures = [r for r in results if not r["answer_correct"]]
    if failures:
        print(f"\n{'=' * 70}\nFAILURE DETAIL (for your design doc's error analysis)\n{'=' * 70}")
        for r in failures:
            print(f"\n[{r['id']}] {r['question']}")
            print(f"  expected pages: {r['expected_pages']}, retrieved pages: {r['retrieved_pages']}")
            print(f"  answer: {r['answer'][:300]}")


if __name__ == "__main__":
    testset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/qa_testset.json"
    index_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen3:4b"

    results = run_eval(testset_path, index_dir, model)
    print_summary(results)

    out_path = "eval/results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")
