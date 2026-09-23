"""
reranker.py
=============
Stage 6 of the pipeline: takes the candidate pool from hybrid_retriever.py
(dense + sparse, fused via RRF) and re-scores it with a cross-encoder,
producing the final top-k chunks that actually get sent to the LLM.

WHY A RERANKER IS A SEPARATE STAGE FROM RETRIEVAL, NOT AN OPTIONAL EXTRA
----------------------------------------------------------------------------
Bi-encoders (the dense embedding model in build_dense_index.py) encode
the query and each document SEPARATELY into vectors, then compare those
vectors with cosine similarity. That's what makes them fast enough to
search a whole corpus -- but it also means the model never actually
looks at the query and a candidate document together. A cross-encoder
does the opposite: it takes (query, document) as a SINGLE joint input
and outputs one relevance score, with full attention between every
query token and every document token. This is slower (you can't
precompute anything -- every candidate has to be scored fresh, per
query), which is exactly why it's only run on a small shortlist (the
hybrid-retrieved top ~15-20) rather than the whole corpus. This
two-stage "retrieve broadly, then rerank precisely" pattern (bi-encoder
first pass, cross-encoder second pass) is the standard production
architecture, not a nice-to-have -- see Redis/BentoML/LangChain's own
reranking guides, all describing the identical split.

CONCRETE EVIDENCE THIS STAGE IS NEEDED (not a theoretical justification)
-----------------------------------------------------------------------------
Running hybrid_retriever.py's own demo queries surfaced a real failure:
for the query "0.500% Notes due 2031", the one chunk that actually
contains that exact string (the cover-page securities list) was BM25's
#1 hit, but it wasn't in dense retrieval's top 20 at all (the bi-encoder
latched onto other "Notes"/"Debt"-related tables instead), so RRF fusion
never surfaced it into the final top 3 -- two topically-similar-but-wrong
tables won on consensus. A cross-encoder, which actually reads the query
against each candidate's full text jointly, has a real chance to notice
that one candidate contains the literal query string and the others
don't -- something a bi-encoder's separately-computed vectors structurally
cannot represent as well. This file's demo query section reruns that
exact case to show whether reranking actually fixes it -- treat that as
your project's headline before/after evidence, not just this docstring's
claim.

MODEL CHOICE -- BAAI/bge-reranker-v2-m3
--------------------------------------------
- Widely described as the default self-hosted reranker baseline for
  production RAG pipelines: cheap to run, Apache-2.0 licensed, and
  extensively battle-tested compared to newer/larger alternatives.
- Same BAAI family as the bge-large-en-v1.5 embedding model already
  used in this project -- not required for compatibility (rerankers and
  embedders are independent components), but it means the design doc
  can point to one coherent "why BAAI" rationale instead of two
  unrelated model choices.
- Bigger alternatives exist (Qwen3-Reranker, mxbai-rerank-v2, managed
  APIs like Cohere Rerank) and score higher on some leaderboards, but
  for a ~15-20 candidate shortlist over a single document, the accuracy
  ceiling is not the bottleneck -- correctly having a reranking stage
  AT ALL is what fixes the RRF failure mode above; which specific
  cross-encoder is used matters far less at this corpus size. This is
  the same "match sophistication to actual scale" reasoning applied
  throughout this pipeline (see build_dense_index.py, chunk.py).
- Already present in this project's local model cache from earlier
  work, alongside cross-encoder/ms-marco-MiniLM-L-6-v2 -- either is a
  legitimate choice; bge-reranker-v2-m3 is used here as the stronger
  default, with ms-marco-MiniLM-L-6-v2 noted below as a drop-in,
  faster/lighter fallback if reranking latency ever becomes a concern.

INPUT TEXT -- generation_text, not search_text
-----------------------------------------------
Unlike the dense embedder (which deliberately embeds only the short
`search_text` -- see build_dense_index.py for why), the cross-encoder
reranker is given each candidate's full `generation_text` (the complete
table markdown, or the full text chunk). Cross-encoders read the query
and document jointly rather than compressing the document into a fixed
vector ahead of time, so there's no benefit to hiding the numeric table
body from it the way there is for embedding -- if anything, seeing the
actual numbers is exactly what lets it judge relevance correctly for
this document's number-heavy queries.
"""

from __future__ import annotations

import os
import sys

# Same macOS OpenMP-conflict fix as build_dense_index.py / hybrid_retriever.py
# -- must be set before numpy/torch are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder

# Make `src/` importable as a namespace package root -- same pattern as
# hybrid_retriever.py and answer.py. A plain sibling import
# (`from hybrid_retriever import ...`) only works when this file is run
# directly as a script (Python auto-adds a script's own directory to
# sys.path[0]) -- it breaks the moment this module is imported from
# elsewhere (e.g. answer.py importing `retrieve.reranker`), because then
# reranker.py's own directory is never added to sys.path. Using the same
# `src`-relative import everywhere makes this file work correctly both
# as a standalone script AND as an imported module -- this exact bug was
# caught by testing answer.py's import chain, not by inspection.
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from retrieve.hybrid_retriever import hybrid_search, load_retriever  # noqa: E402

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Same eager-attention / float32 fix used for the embedding model --
# see build_dense_index.py's "MACOS SEGFAULT NOTE" docstrings for why.
_MODEL_LOAD_KWARGS = {
    "model_kwargs": {
        "attn_implementation": "eager",
        "dtype": "float32",
    }
}

# Cross-encoders have a max input length; text beyond this gets truncated
# by the tokenizer automatically, but we cap candidate text ourselves too
# so truncation removes the least important trailing content (e.g. very
# long tables' final rows) rather than happening invisibly inside the model.
MAX_CANDIDATE_CHARS = 2000


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    print(f"Loading reranker model '{model_name}' on device='cpu' "
          f"(weights are cached locally after the first download)...")
    try:
        return CrossEncoder(model_name, device="cpu", **_MODEL_LOAD_KWARGS)
    except TypeError:
        # Some sentence-transformers versions' CrossEncoder class does not
        # accept model_kwargs, even though the SentenceTransformer class
        # (used for the embedding model) does in the same installed
        # version -- this is a real gap between the two classes, caught
        # via a live TypeError on a real machine, not a hypothetical.
        # Forcing eager-attention/float32 was a precaution carried over
        # from the embedding-model segfault fix, not a strict requirement
        # for this BERT-style architecture -- falling back to plain
        # defaults here is safe.
        print("Note: this sentence-transformers version's CrossEncoder doesn't accept "
              "model_kwargs -- loading with default settings instead.")
        return CrossEncoder(model_name, device="cpu")


def rerank(query: str, candidates: list[dict], reranker: CrossEncoder, top_k: int = 5) -> list[dict]:
    """
    candidates: chunk dicts (as returned by hybrid_search), each with a
    `generation_text` field. Returns the top_k chunks, re-sorted by
    cross-encoder relevance score, each annotated with `_rerank_score`.
    """
    if not candidates:
        return []

    pairs = [(query, c["generation_text"][:MAX_CANDIDATE_CHARS]) for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        chunk = dict(chunk)  # shallow copy, don't mutate the caller's chunk
        chunk["_rerank_score"] = float(score)
        results.append(chunk)
    return results


def retrieve_and_rerank(
    query: str,
    dense_index,
    dense_chunks: list[dict],
    embedding_model,
    bm25,
    sparse_chunks: list[dict],
    reranker: CrossEncoder,
    k_candidates: int = 15,
    k_final: int = 5,
) -> list[dict]:
    """
    The full retrieval pipeline in one call: hybrid retrieve a modest
    candidate pool (k_candidates), then rerank it down to the final
    k_final chunks that actually go to the LLM. This is the function
    generate/answer.py (next stage) should call.
    """
    candidates = hybrid_search(
        query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks,
        k_final=k_candidates,
    )
    return rerank(query, candidates, reranker, top_k=k_final)


if __name__ == "__main__":
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/index"

    print("Loading dense + sparse indexes, embedding model, and reranker...")
    dense_index, dense_chunks, embedding_model, bm25, sparse_chunks = load_retriever(index_dir)
    reranker = load_reranker()

    # Same two demo queries as hybrid_retriever.py -- the second one is
    # the documented RRF failure case (see module docstring above). This
    # is the direct before/after comparison for your design doc.
    demo_queries = [
        "What were Apple's total net sales for the nine months ended June 25, 2022?",
        "0.500% Notes due 2031",
    ]

    for query in demo_queries:
        print(f"\n{'=' * 70}\nQUERY: {query!r}\n{'=' * 70}")

        # Pull a wider candidate pool than the final answer needs, so the
        # reranker has real material to work with -- see docstring above
        # on why retrieval's job is recall and reranking's job is precision.
        candidates = hybrid_search(
            query, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks,
            k_final=15,
        )
        print(f"\n-- Hybrid (RRF) top 3 of {len(candidates)} candidates (BEFORE reranking) --")
        for c in candidates[:3]:
            info = c["_retrieval"]
            print(f"  rrf={info['rrf_score']}  page={c['page_number']}  "
                  f"{c['search_text'][:90]!r}")

        reranked = rerank(query, candidates, reranker, top_k=3)
        print("\n-- Cross-encoder reranked top 3 (AFTER reranking) --")
        for c in reranked:
            print(f"  rerank_score={c['_rerank_score']:.3f}  page={c['page_number']}  "
                  f"{c['search_text'][:90]!r}")