"""
scoring.py

Shared by
step1_retrieve.py and step2_generate.py. No torch/faiss/Ollama --
deliberately, so the memory-isolated step2 process can import it with
zero footprint.

WHY THESE METRICS AND NOT AN LLM JUDGE
--------------------------------------------
A judge needs a second model call per question per metric, on top of
three models already near this machine's memory ceiling. For a corpus
whose answers are exact figures on known pages, deterministic math is
also the stronger signal: page overlap and number-substring checks are
exact and reproducible, while a judge would guess the same thing less
reliably and non-deterministically.

RETRIEVAL (from the system's own ranked output vs hand-verified pages):
  - Precision@k: fraction of returned CHUNKS from a correct page.
    Chunk- (not page-) granular on purpose: it measures the actual
    noise handed to the LLM (this project's "context rot" concern).
  - MRR: 1/rank of the first correct chunk; distinguishes rank-1 from
    rank-5 where a binary hit/miss cannot.
  - NDCG@k: rewards correct pages ranked higher via log-discounted
    weights, normalized to [0,1]. Computed over DEDUPED pages -- a bug
    found by testing: counting repeat chunks let DCG exceed IDCG
    (observed NDCG=1.5), violating NDCG's definition.
Refusal cases (no expected pages) score a vacuous 1.0 here; their real
test is the generation-stage refusal check.

GENERATION (vs a hand-written reference_answer per question):
  - Token F1: bag-of-words overlap, SQuAD-style; gives partial credit.
    The tokenizer keeps figures intact ("43.6%" is one token).
  - Exact Match: normalized sequences identical. EXPECTED to be ~0 on
    prose answers -- a model never phrases things exactly like the
    reference. Reporting near-zero EM honestly, beside high F1/keyword
    rates, is more credible than tuning the comparison to inflate it.
"""

from __future__ import annotations

import math
import re
from collections import Counter




def check_retrieval_hit(retrieved_pages: list[int], expected_pages: list[int]) -> bool:
    """True if at least one retrieved page matches a known-correct page.
    An empty expected_pages list (hallucination-refusal cases) trivially
    passes -- there's no 'correct page' to find for an unanswerable question."""
    if not expected_pages:
        return True
    return bool(set(retrieved_pages).intersection(expected_pages))


def check_answer_correct(answer: str, case: dict) -> bool:
    """
    For hallucination_refusal cases: pass if the answer contains refusal
    language. For everything else: pass only if ALL expected substrings
    are present (case-insensitive) -- e.g. the ambiguity-trap case
    requires ALL FOUR period figures to be present, not just one.
    """
    answer_lower = answer.lower()
    if case.get("expected_refusal"):
        return any(kw.lower() in answer_lower for kw in case["expected_answer_contains"])
    return all(kw.lower() in answer_lower for kw in case["expected_answer_contains"])


# Retrieval metrics: Precision@k, MRR, NDCG@k

def precision_at_k(ranked_pages: list[int], expected_pages: list[int], k: int | None = None) -> float:
    """Fraction of the top-k ranked CHUNKS (not deduplicated by page) that
    came from an expected/correct page. Deliberately chunk-granular, not
    page-granular: this measures how much NOISE is actually sitting in
    what gets handed to the LLM as context (this project's "context rot"
    concern -- see answer.py's relative score cutoff), so a page
    contributing 2 of 5 real chunks correctly scores differently from a
    page contributing 1 of 5. k defaults to len(ranked_pages) -- i.e.
    "of what the system actually returned," matching production
    behavior rather than an arbitrary cutoff. Bounded to [0, 1] by
    construction (a simple count-over-count ratio), unlike NDCG@k below,
    which needs deduplication to stay bounded -- see ndcg_at_k's
    docstring for why the two metrics are deliberately NOT computed the
    same way here."""
    if not expected_pages:
        return 1.0  # hallucination-refusal case -- see module docstring
    top_k = ranked_pages[:k] if k is not None else ranked_pages
    if not top_k:
        return 0.0
    relevant = sum(1 for p in top_k if p in expected_pages)
    return relevant / len(top_k)


def mrr(ranked_pages: list[int], expected_pages: list[int]) -> float:
    """Reciprocal rank of the first correct page in the ranked list. 0.0 if
    it never appears at all."""
    if not expected_pages:
        return 1.0  # hallucination-refusal case -- see module docstring
    for i, p in enumerate(ranked_pages, start=1):
        if p in expected_pages:
            return 1.0 / i
    return 0.0


def _dedupe_pages_keep_first(pages: list[int]) -> list[int]:
    """Collapses a chunk-level page list to distinct pages, keeping each
    page's FIRST (best-ranked) occurrence and dropping later repeats.
    Needed because a single page routinely contributes several chunks
    (e.g. a large table split across chunks, or several narrative chunks
    from the same page) -- NDCG must rank DISTINCT relevant items, not
    reward the same page twice for showing up in two different chunks."""
    seen: set[int] = set()
    out: list[int] = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _dcg_at_k(ranked_pages: list[int], expected_pages: list[int], k: int) -> float:
    deduped = _dedupe_pages_keep_first(ranked_pages)
    top_k = deduped[:k]
    return sum(
        (1.0 if p in expected_pages else 0.0) / math.log2(i + 1)
        for i, p in enumerate(top_k, start=1)
    )


def ndcg_at_k(ranked_pages: list[int], expected_pages: list[int], k: int | None = None) -> float:
    """
    Normalized Discounted Cumulative Gain with binary relevance. Rewards
    a correct page appearing EARLIER in the ranking more than the same
    page appearing later, normalized against the best possible ranking
    (all correct pages placed first).

    DEDUPLICATES ranked_pages to distinct pages before scoring -- a real
    bug found by testing, not a hypothetical: this project's
    ranked_pages is a CHUNK-level list (one entry per retrieved chunk),
    and a single page routinely contributes multiple chunks. The
    original version counted every occurrence of a relevant page as a
    separate "hit" in DCG (e.g. page 18 appearing at both rank 1 and
    rank 3 contributed TWICE), while IDCG -- correctly -- only ever
    credits each of the len(expected_pages) DISTINCT pages once. Result:
    DCG could exceed IDCG, producing NDCG@k > 1.0 (observed directly:
    ranked_pages=[18, 19, 18, 17, 17], expected_pages=[18] produced
    NDCG=1.5), which violates NDCG's own definition -- it is only
    meaningful, and only guaranteed bounded to [0, 1], when computed
    over a ranking of DISTINCT items. Deduplicating to each page's best
    (first) rank before computing DCG fixes this: a page can only ever
    contribute its single highest-ranked occurrence, matching exactly
    what IDCG already assumes. Verified against the real case above
    (see scoring's test coverage) to now return a value in [0, 1].

    k defaults to the number of DISTINCT pages retrieved (after
    deduplication), not the raw chunk count -- consistent with NDCG
    being defined over distinct ranked items.
    """
    if not expected_pages:
        return 1.0  # hallucination-refusal case -- see module docstring
    deduped = _dedupe_pages_keep_first(ranked_pages)
    k = k if k is not None else len(deduped)
    dcg = _dcg_at_k(ranked_pages, expected_pages, k)
    ideal_hits = min(len(expected_pages), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Generation metrics: token-level F1 (SQuAD-style) and Exact Match
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")  # keeps numbers, decimals, and percent signs
                                                         # intact as single tokens (e.g. "43.6%"),
                                                         # since splitting them apart would make a
                                                         # correct figure fail to match a reference
                                                         # that also states it as one token.


def _normalize_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def exact_match(answer: str, reference: str) -> bool:
    """Strict, SQuAD-style Exact Match: do the normalized token sequences
    match EXACTLY? Expected to be low for prose-style answers -- see
    module docstring for why that's the correct, honest behavior of
    this metric on a generative (not extractive-span) task."""
    return _normalize_tokens(answer) == _normalize_tokens(reference)


def f1_score(answer: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between the generated answer and a
    hand-written reference answer. Gives partial credit for a mostly-
    correct answer, unlike the binary keyword check or exact_match."""
    ans_tokens = _normalize_tokens(answer)
    ref_tokens = _normalize_tokens(reference)
    if not ans_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(ans_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ans_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> None:
    n = len(results)
    retrieval_hits = sum(r["retrieval_hit"] for r in results)
    keyword_correct = sum(r["answer_correct"] for r in results)
    exact_matches = sum(r["answer_exact_match"] for r in results)
    mean_precision = sum(r["retrieval_precision_at_k"] for r in results) / n
    mean_mrr = sum(r["retrieval_mrr"] for r in results) / n
    mean_ndcg = sum(r["retrieval_ndcg_at_k"] for r in results) / n
    mean_f1 = sum(r["answer_f1"] for r in results) / n

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print("Retrieval:")
    print(f"  Hit rate:        {retrieval_hits}/{n} ({100 * retrieval_hits / n:.0f}%)")
    print(f"  Mean Precision@k: {mean_precision:.3f}")
    print(f"  Mean MRR:         {mean_mrr:.3f}")
    print(f"  Mean NDCG@k:      {mean_ndcg:.3f}")
    print("Generation:")
    print(f"  Keyword-match rate: {keyword_correct}/{n} ({100 * keyword_correct / n:.0f}%)")
    print(f"  Exact Match rate:   {exact_matches}/{n} ({100 * exact_matches / n:.0f}%)  "
          f"(expected to be low on prose answers -- see scoring.py docstring)")
    print(f"  Mean token F1:      {mean_f1:.3f}")
    print()
    print(f"{'ID':<28} {'P@k':<6} {'MRR':<6} {'NDCG':<6} {'F1':<6} {'EM':<5} {'KW':<5}")
    print("-" * 78)
    for r in results:
        print(f"{r['id']:<28} "
              f"{r['retrieval_precision_at_k']:<6.2f} "
              f"{r['retrieval_mrr']:<6.2f} "
              f"{r['retrieval_ndcg_at_k']:<6.2f} "
              f"{r['answer_f1']:<6.2f} "
              f"{'Y' if r['answer_exact_match'] else 'N':<5} "
              f"{'PASS' if r['answer_correct'] else 'FAIL':<5}")

    failures = [r for r in results if not r["answer_correct"]]
    if failures:
        print(f"\n{'=' * 78}\nFAILURE DETAIL (for your design doc's error analysis)\n{'=' * 78}")
        for r in failures:
            print(f"\n[{r['id']}] {r['question']}")
            print(f"  expected pages: {r['expected_pages']}, retrieved pages: {r['retrieved_pages']}")
            print(f"  reference: {r.get('reference_answer', '(none)')}")
            print(f"  answer: {r['answer'][:300]}")