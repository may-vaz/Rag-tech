"""
hybrid_retriever.py
=====================
Stage 5 of the pipeline: given a query, runs BOTH the dense (semantic)
and sparse (BM25) retrievers built in build_dense_index.py /
build_sparse_index.py, then fuses their two ranked lists into one final
ranking using Reciprocal Rank Fusion (RRF).

WHY FUSE RANKS, NOT RAW SCORES
--------------------------------
Dense cosine similarity (roughly -1 to 1, usually 0.3-0.9 in practice)
and BM25 scores (unbounded, driven by term rarity and document length)
are not on the same scale -- averaging or weighting them directly
requires arbitrary, per-corpus calibration that breaks the moment the
query mix changes. RRF sidesteps this entirely by using each result's
RANK POSITION instead of its raw score:

    RRF(d) = sum over each retriever's ranked list of  1 / (k + rank(d))

A document that ranks well in BOTH lists accumulates more score than
one that ranks #1 in only one list -- this is a genuine "consensus
vote," not a weighted average. This is why RRF is the default fusion
method in production hybrid search (Elasticsearch, OpenSearch, Azure AI
Search, Weaviate, MongoDB Atlas all implement exactly this), not
something specific to this project -- it's the standard approach for a
documented reason: it needs no per-corpus score calibration and is
robust to wildly different scoring distributions between retrievers.

WHY k=60, AND WHEN THAT'S ACTUALLY THE WRONG CHOICE
-------------------------------------------------------
k=60 is the constant from the original RRF paper (Cormack, Clarke &
Buettcher, 2009) and is what every production system above defaults to,
so it's used here too. But it's worth understanding what it actually
does, not just copying the number: a larger k FLATTENS the ranking
curve -- the gap between rank 1 and rank 2 in a single list becomes
tiny, which means RRF rewards a document that appears reasonably high
in BOTH lists over a document that is the single best match in only
ONE list. For most queries in this document, that's exactly the
correct behavior. But for a genuinely exact-match query (a specific
percentage/date/instrument name, e.g. "0.500% Notes due 2031"), BM25
can return the exact right chunk at rank 1 with high confidence, and a
large k can let two merely-topically-similar chunks that both retrievers
rank moderately outrank it. This is a documented, known trade-off of
RRF, not a bug -- `rrf_k` is exposed as a parameter below (rather than
hardcoded) specifically so this can be tuned per-evaluation if your
RAGAS results (see eval/ later) show exact-match queries losing to
consensus picks.

WHY WE RETRIEVE MORE CANDIDATES THAN WE RETURN (k_dense/k_sparse vs k_final)
--------------------------------------------------------------------------------
We pull the top ~20 from EACH retriever before fusing, then fuse down
to the final top k (default 5). Pulling only the final k from each
side first would mean a chunk that's, say, dense-rank-3 and
sparse-rank-15 might get cut from the sparse side before fusion ever
sees it, even though appearing at all in both lists is exactly the
signal RRF is designed to reward. This is also exactly what the
NEXT stage (reranker.py) needs: a modest-sized candidate pool (this
fused top ~10-20) that a slower, more accurate cross-encoder can then
re-score properly -- retrieval's job is recall (don't lose the right
chunk), reranking's job is precision (put it first).

HOW THIS FILE IMPORTS THE PREVIOUS STAGES
--------------------------------------------
This project has no setup.py / package install step -- each stage is
run directly as a script (`python3 src/index/build_dense_index.py ...`).
To reuse the loader/search functions from the indexing stage without
duplicating that logic here, we add the `src/` directory to sys.path at
runtime and import `index.build_dense_index` / `index.build_sparse_index`
as plain modules. This works with ZERO extra files (no __init__.py
needed) because Python 3 treats any directory on sys.path as an
importable "namespace package" automatically.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

# Same macOS OpenMP-conflict fix as build_dense_index.py -- must be set
# before numpy/torch are imported. See that file's "MACOS SEGFAULT NOTE #2"
# for the full explanation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make `src/` importable as a namespace package root (see docstring above).
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from index.build_dense_index import (  # noqa: E402
    load_dense_index,
    load_embedding_model,
    search_dense,
)
from index.build_sparse_index import load_sparse_index, search_sparse  # noqa: E402

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """
    Core RRF math, decoupled from dense/sparse specifics so it's easy to
    unit-test and easy to extend (e.g. adding a third ranked list later,
    such as a metadata-filtered result set, needs no change here).

    ranked_lists: e.g. [["chunk_id_a", "chunk_id_b", ...],   # dense, best first
                         ["chunk_id_c", "chunk_id_a", ...]]  # sparse, best first
    Returns: {chunk_id: fused_score}, NOT yet sorted.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in ranked_lists:
        for position, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] += 1.0 / (k + position)
    return dict(scores)


def hybrid_search(
    query: str,
    dense_index,
    dense_chunks: list[dict],
    embedding_model,
    bm25,
    sparse_chunks: list[dict],
    k_final: int = 5,
    k_dense: int = 20,
    k_sparse: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    Runs both retrievers, fuses via RRF, and returns the top k_final
    chunks as full chunk dicts (with an added `_retrieval` debug field
    showing each chunk's rank in each individual retriever -- useful
    for the eval stage and for explaining retrieval decisions in your
    design doc, e.g. "this chunk was BM25-rank-1 but dense-rank-11").
    """
    dense_results = search_dense(query, dense_index, dense_chunks, embedding_model, k=k_dense)
    sparse_results = search_sparse(query, bm25, sparse_chunks, k=k_sparse)

    dense_ids = [r["chunk"]["chunk_id"] for r in dense_results]
    sparse_ids = [r["chunk"]["chunk_id"] for r in sparse_results]

    fused_scores = reciprocal_rank_fusion([dense_ids, sparse_ids], k=rrf_k)

    # chunk_id -> full chunk dict, for looking up content once we know
    # the final fused order (either result list has the full chunk data)
    chunk_by_id = {r["chunk"]["chunk_id"]: r["chunk"] for r in dense_results}
    chunk_by_id.update({r["chunk"]["chunk_id"]: r["chunk"] for r in sparse_results})

    dense_rank_by_id = {cid: i + 1 for i, cid in enumerate(dense_ids)}
    sparse_rank_by_id = {cid: i + 1 for i, cid in enumerate(sparse_ids)}

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    final = []
    for cid in ranked_ids[:k_final]:
        chunk = dict(chunk_by_id[cid])  # shallow copy so we don't mutate the cached chunk
        chunk["_retrieval"] = {
            "rrf_score": round(fused_scores[cid], 5),
            "dense_rank": dense_rank_by_id.get(cid),   # None if it wasn't in dense's top k_dense at all
            "sparse_rank": sparse_rank_by_id.get(cid),
        }
        final.append(chunk)
    return final


def load_retriever(index_dir: str = "data/index"):
    """
    One-time setup: load both indexes and the embedding model. Call this
    once per process (e.g. once when your API server starts, or once at
    the top of an eval script), NOT once per query -- reloading the
    embedding model per query would be needlessly slow.
    """
    dense_index, dense_chunks, model_name = load_dense_index(index_dir)
    embedding_model = load_embedding_model(model_name)
    bm25, sparse_chunks = load_sparse_index(index_dir)
    return dense_index, dense_chunks, embedding_model, bm25, sparse_chunks


if __name__ == "__main__":
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/index"

    print("Loading dense + sparse indexes and embedding model...")
    dense_index, dense_chunks, embedding_model, bm25, sparse_chunks = load_retriever(index_dir)

    demo_queries = [
        "What were Apple's total net sales for the nine months ended June 25, 2022?",
        "0.500% Notes due 2031",  # exact-match case sparse retrieval should win on
    ]

    for query in demo_queries:
        print(f"\n{'=' * 70}\nQUERY: {query!r}\n{'=' * 70}")

        dense_only = search_dense(query, dense_index, dense_chunks, embedding_model, k=3)
        print("\n-- Dense-only top 3 --")
        for r in dense_only:
            c = r["chunk"]
            print(f"  score={r['score']:.3f}  page={c['page_number']}  {c['search_text'][:90]!r}")

        sparse_only = search_sparse(query, bm25, sparse_chunks, k=3)
        print("\n-- Sparse (BM25)-only top 3 --")
        for r in sparse_only:
            c = r["chunk"]
            print(f"  score={r['score']:.3f}  page={c['page_number']}  {c['search_text'][:90]!r}")

        fused = hybrid_search(
            query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, k_final=3
        )
        print("\n-- HYBRID (RRF-fused) top 3 --")
        for c in fused:
            info = c["_retrieval"]
            print(f"  rrf={info['rrf_score']}  dense_rank={info['dense_rank']}  "
                  f"sparse_rank={info['sparse_rank']}  page={c['page_number']}  "
                  f"{c['search_text'][:90]!r}")