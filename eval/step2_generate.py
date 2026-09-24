"""
step2_generate.py
====================
The SECOND of two separate process runs for evaluation. Reads
eval/prepared.json (written by step1_retrieve.py, which has already
exited completely by the time this runs), calls Ollama for each
prepared question, scores the answers, and writes eval/results.json.

This process imports NOTHING beyond the standard library, `requests`
(via generate.llm_client), and this project's own dependency-free
scoring.py -- no torch, no faiss, no sentence-transformers, even
transitively. That's verified at the bottom of this file: it checks
sys.modules and refuses to silently proceed if any heavy library
somehow got imported, so this guarantee doesn't quietly rot if someone
edits an import elsewhere later.

DEEP METRICS, NOT JUST PASS/FAIL (see eval/scoring.py for definitions)
--------------------------------------------------------------------------------
Beyond the original keyword-based pass/fail check, this now also
computes, per question: Precision@k, MRR, and NDCG@k for retrieval
quality (carried over from step1_retrieve.py, which has the ranked
candidate list), and token-level F1 plus Exact Match for generation
quality (comparing the model's answer against a hand-written
reference_answer in qa_testset.json). These are the standard,
deterministic categories used industry-wide for RAG evaluation when an
LLM-judge (RAGAS-style faithfulness/context precision/recall) isn't a
good fit for the corpus or hardware -- see run_eval.py and
scoring.py's own docstrings for the full reasoning on why that
trade-off was made here.

RESUMABLE, PER-QUESTION SAVING (added after real testing found the need)
------------------------------------------------------------------------------
Testing this on the actual 8GB target machine showed that even with
this process being genuinely free of torch/faiss/sentence-transformers,
a specific question can still exceed the timeout -- not from memory
contention (that's ruled out here), but most likely because a
particular question's retrieved context is simply larger/more complex
(e.g. a big nested financial table) and a 4B CPU-only model needs more
than the timeout to process it. That's a real compute-time limit, not a
bug to "fix" architecturally the way the memory issue was.

The practical response: this script now writes eval/results.json after
EVERY question, not just once at the end, and on startup it loads any
existing results.json and SKIPS questions already completed. So if
question 3 times out, questions 1-2's results are already safely on
disk, and simply re-running this exact same command picks up from
question 3 onward instead of redoing everything. A question that times
out is logged clearly and the script moves on to the next one, rather
than crashing the whole run -- you can always re-run afterward to retry
just the ones that failed.

USAGE
------
    python3 eval/step1_retrieve.py eval/qa_testset.json data/index   # run first, let it exit
    python3 eval/step2_generate.py eval/prepared.json qwen3:4b        # then run this
    # if a question times out, just run the exact same command again --
    # completed questions are skipped, only the remaining ones run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate.llm_client import SYSTEM_PROMPT, call_ollama  # noqa: E402
from scoring import (  # noqa: E402
    check_answer_correct,
    exact_match,
    f1_score,
    print_summary,
)

# Verify the isolation this file's docstring promises.If this ever fires, something upstream
# started importing a heavy library at module load time, and this
# process is no longer actually lightweight.
_FORBIDDEN = {"torch", "faiss", "sentence_transformers"}
_loaded = _FORBIDDEN.intersection(sys.modules.keys())
if _loaded:
    raise RuntimeError(
        f"step2_generate.py is supposed to be free of heavy ML libraries, but found "
        f"{_loaded} already imported. This defeats the point of running generation as "
        f"an isolated process -- check for an accidental import in llm_client.py or scoring.py."
    )


def load_existing_results(out_path: str) -> dict:
    """Returns {question_id: result_dict} for any results already saved
    from a previous (possibly interrupted) run. Empty dict if none exist."""
    if not Path(out_path).exists():
        return {}
    with open(out_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    return {r["id"]: r for r in existing}


def save_results(results_by_id: dict, out_path: str) -> None:
    # Written after every question, not just at the end -- see module docstring.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(results_by_id.values()), f, indent=2)


def main(prepared_path: str, model: str, out_path: str) -> None:
    with open(prepared_path, "r", encoding="utf-8") as f:
        prepared = json.load(f)
    print(f"Loaded {len(prepared)} prepared questions from {prepared_path}")

    results_by_id = load_existing_results(out_path)
    already_done = set(results_by_id.keys())
    if already_done:
        print(f"Found {len(already_done)} already-completed result(s) in {out_path} -- "
              f"skipping those and resuming with the rest.")
    print(f"Using model: {model}\n")

    for item in prepared:
        if item["id"] in already_done:
            print(f"[{item['id']}] already completed -- skipping")
            continue

        context_size = len(item["context"]) if item["context"] else 0
        engine_answer = item.get("engine_answer")

        if engine_answer is not None:
            # Answered by the deterministic fact engine in step1 -- no
            # Ollama call needed at all. Still scored identically to
            # every other answer, so engine questions get the same
            # F1/EM/keyword-match evidence in the final report.
            print(f"[{item['id']}] {item['question']}  (fact engine -- no LLM call)")
            answer_text = engine_answer
        else:
            print(f"[{item['id']}] {item['question']}  (context: {context_size} chars)")
            if item["context"] is None:
                answer_text = "This filing does not contain relevant information to answer this question."
            else:
                user_prompt = f"CONTEXT:\n\n{item['context']}\n\nQUESTION: {item['question']}"
                try:
                    answer_text = call_ollama(SYSTEM_PROMPT, user_prompt, model=model)
                except RuntimeError as e:
                    # Don't let one slow/failed question kill the whole batch.
                    # Nothing is written to results_by_id for this id, so it
                    # will be retried (not skipped) the next time this script runs.
                    print(f"  -> ERROR (will retry on next run): {e}\n")
                    continue

        case_for_scoring = {
            "expected_answer_contains": item["expected_answer_contains"],
            "expected_refusal": item["expected_refusal"],
        }
        answer_correct = check_answer_correct(answer_text, case_for_scoring)
        reference = item.get("reference_answer", "")
        f1 = f1_score(answer_text, reference) if reference else 0.0
        em = exact_match(answer_text, reference) if reference else False

        results_by_id[item["id"]] = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "method": "fact_engine" if engine_answer is not None else "llm",
            "retrieval_hit": item["retrieval_hit"],
            "retrieval_precision_at_k": item.get("retrieval_precision_at_k", 0.0),
            "retrieval_mrr": item.get("retrieval_mrr", 0.0),
            "retrieval_ndcg_at_k": item.get("retrieval_ndcg_at_k", 0.0),
            "answer_correct": answer_correct,
            "answer_f1": f1,
            "answer_exact_match": em,
            "answer": answer_text,
            "reference_answer": reference,
            "retrieved_pages": item["retrieved_pages"],
            "expected_pages": item["expected_pages"],
        }
        save_results(results_by_id, out_path)  # persist immediately, not at the end
        print(f"  -> {'PASS' if answer_correct else 'FAIL'}  (F1={f1:.2f}, EM={'Y' if em else 'N'})  (saved)\n")

    n_total = len(prepared)
    n_done = len(results_by_id)
    if n_done < n_total:
        print(f"\n{n_total - n_done} question(s) still remaining (timed out or not yet run). "
              f"Re-run this exact command to retry them -- completed ones will be skipped.")
    else:
        print("\nAll questions completed.")
        # Only print the full pass/fail summary once everything's actually done --
        # a partial summary on an interrupted run would be misleading.
        print_summary(list(results_by_id.values()))

    print(f"\nResults so far saved to {out_path}")


if __name__ == "__main__":
    prepared_path = sys.argv[1] if len(sys.argv) > 1 else "eval/prepared.json"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:4b"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "eval/results.json"
    main(prepared_path, model, out_path)