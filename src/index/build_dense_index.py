"""
build_dense_index.py
turns chunks.jsonl into a searchable dense
(semantic) vector index.

EMBEDDING MODEL CHOICE -- BAAI/bge-large-en-v1.5
-----------------------------------------------------
This project originally targeted Qwen/Qwen3-Embedding-0.6B (Qwen3's
newer decoder-based embedding architecture, and arguably the current
single best open-weight retrieval model on MTEB). It was swapped out
after real testing surfaced a reproducible segfault loading that model
on this specific machine (torch 2.14.0 + transformers 4.57.6 +
sentence-transformers, Apple Silicon arm64) that persisted even after
forcing eager attention and float32 -- while a plain BERT-family model
(all-MiniLM-L6-v2) loaded and ran fine in the same environment. That
isolates the crash to Qwen3's newer architecture on this setup, not to
PyTorch/the environment generally, so rather than keep fighting a
newer model's rough edges, the project switched to a model in the same
architecture family that had already been proven stable here.

Why BAAI/bge-large-en-v1.5, specifically, as the replacement:
- BERT-based (the same well-established architecture family as
  all-MiniLM-L6-v2, which is verified working on this machine) -- not
  a newer/less broadly tested architecture.
- Still a genuinely strong open-source retrieval model: for a long time
  it was the standard "best open embedding model" recommendation before
  the Qwen3/GTE/E5 generation, and it remains solidly competitive on
  MTEB retrieval, especially at this project's scale where the gap
  between top embedding models matters far less than retrieval
  architecture (see chunk.py / enrich_metadata.py docstrings).
- Apache-2.0 licensed, ~1.3GB download, 1024-dim output (same dimension
  as Qwen3-Embedding-0.6B's default, incidentally -- no downstream code
  needed to change because of the swap).
- Unlike Qwen3-Embedding, BGE does NOT ship a built-in sentence-transformers
  "query" prompt template. Its documented usage instead asks you to
  manually prepend a fixed instruction string to QUERIES ONLY (never to
  indexed documents) -- implemented below as BGE_QUERY_INSTRUCTION.
  This is the same underlying idea as Qwen3's prompt_name="query"
  (asymmetric query/document encoding), just applied by hand instead of
  via a config the model ships with.


WHY WE DO NOT ALSO EMBED FULL TABLE MARKDOWN
-----------------------------------------------
Every chunk from chunk.py already carries a `search_text` field
(caption + auto-summary + column headers for tables; full prose for
text) that is DIFFERENT from `generation_text` (the complete content).
We embed ONLY `search_text`. Raw numeric table bodies embed poorly --
there's little semantic signal in "58,107" vs "63,090" for a text
embedding model to key on -- so embedding the full table would dilute
the vector with noise. The full table is still stored as metadata
alongside the vector and is what actually gets sent to the LLM once
this chunk is retrieved (see retrieve/hybrid_retriever.py, next file).

VECTOR STORE CHOICE -- FAISS (IndexFlatIP)
--------------------------------------------------------------------------
For a single-document corpus (order of ~100 chunks), a managed vector
DB is unjustified operational overhead.
FAISS's IndexFlatIP (exact inner-product search over L2-normalized
vectors, i.e. exact cosine similarity) does brute-force exact search,
which at this corpus size is both faster to set up AND more accurate
than an approximate index (HNSW/IVF) -- approximate search trades
accuracy for speed at scale, and at ~100 vectors there is no speed
problem to solve. This is the same "match sophistication to actual
scale" judgment call made throughout this pipeline.

MACOS SEGFAULT NOTE #2 -- faiss / torch OpenMP conflict (the actual root
cause found after two other fixes did not resolve it)
--------------------------------------------------------------------------
After fixing the dtype/attention issues above and switching models, the
segfault persisted, but the crash point moved: the model now loaded and
downloaded successfully, and the crash happened at the exact moment
`model.encode()` started its first batch. That's the signature of a
known, well-documented conflict on macOS: `faiss` and `torch` each
bundle their OWN copy of the OpenMP parallel-computation runtime, and
if a process loads both, the two runtimes can silently clash instead of
raising a normal Python exception -- it just segfaults. This is a
library-interaction bug, not a bug in the embedding model or in this
script's logic.

Two changes fix it:
  1. Force single-threaded BLAS/OpenMP execution via environment
     variables, set BEFORE any of numpy/torch/faiss are imported
     (setting them later has no effect, since the native libraries
     read these once, at import time).
  2. Delay importing `faiss` until AFTER the embedding step has fully
     finished, instead of importing it at the top of the file. This
     means faiss's native code is never loaded into the process while
     torch's forward pass is actually running -- the two libraries'
     risky overlap window is eliminated entirely, not just made safer.
  For a ~100-chunk corpus, forcing single-threaded execution costs a
  few extra seconds, which is a trivial trade-off for "the script
  doesn't crash."
"""

from __future__ import annotations

import os

# CRITICAL: these must be set before numpy/torch/faiss are imported --
# setting them later has no effect, since the native libraries read
# these environment variables once, at import/initialization time.
# See "MACOS SEGFAULT NOTE #2" above for why each one is here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # allow duplicate OpenMP runtimes instead of crashing
os.environ.setdefault("OMP_NUM_THREADS", "1")             # force single-threaded OpenMP
os.environ.setdefault("MKL_NUM_THREADS", "1")             # force single-threaded MKL (Intel's BLAS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")        # force single-threaded OpenBLAS
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")      # force single-threaded Apple Accelerate
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid a separate known fork-related crash

import json
import pickle
import sys
from pathlib import Path

# NOTE: faiss is NOT imported here at module load time -- see
# "MACOS SEGFAULT NOTE #2" above. It's imported lazily, inside the
# functions that actually need it, AFTER the embedding model has
# already finished its forward pass. This keeps faiss's native code
# out of the process during the one moment (torch's actual tensor
# computation) where the conflict was observed to crash.
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# BGE's documented usage: prepend this fixed instruction to QUERIES ONLY,
# never to indexed documents. This is BGE's manual equivalent of Qwen3's
# prompt_name="query" mechanism -- same idea (asymmetric query/document
# encoding improves retrieval), different API since BGE doesn't ship a
# baked-in sentence-transformers prompt template.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_MODEL_LOAD_KWARGS = {
    "model_kwargs": {
        "attn_implementation": "eager",
        "dtype": "float32",
    }
}


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Always loads on CPU with eager attention + float32. See the
    "MACOS SEGFAULT NOTE" module docstring above for why those settings
    exist. CPU is plenty fast for a ~100-chunk corpus (expect well under
    a minute), so there's no real reason to fight MPS/CUDA driver quirks
    for a project this size.
    """
    print(f"Loading embedding model '{model_name}' on device='cpu' "
          f"(weights are cached locally after the first download)...")
    return SentenceTransformer(model_name, device="cpu", **_MODEL_LOAD_KWARGS)


def build_dense_index(chunks: list[dict], model: SentenceTransformer):
    search_texts = [c["search_text"] for c in chunks]

    print(f"Embedding {len(search_texts)} chunks...")
    # Documents are embedded WITHOUT any instruction prefix -- the
    # instruction (BGE_QUERY_INSTRUCTION) is applied to queries only, at
    # retrieval time (see search_dense). Embedding both sides identically
    # would defeat the asymmetric query/document design this instruction
    # is meant to exploit.
    embeddings = model.encode(
        search_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for cosine similarity via inner product
    )

    dim = embeddings.shape[1]
    import faiss  # lazy import -- see "MACOS SEGFAULT NOTE #2": kept out of the
                    # process until embedding (the risky part) has fully finished
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"Built FAISS index: {index.ntotal} vectors, dim={dim}")
    return index, embeddings, dim


def save_index(index, chunks: list[dict], dim: int, out_dir: str, model_name: str) -> None:
    import faiss  # lazy import, see build_dense_index() above

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "dense.faiss"))

    # Sidecar: maps FAISS's internal row order (0..N-1) back to full chunk
    # metadata (including generation_text, page number, section, etc.),
    # since FAISS itself only ever returns integer row IDs.
    with open(out / "dense_chunk_map.pkl", "wb") as f:
        pickle.dump(
            {
                "chunks": chunks,          # index-aligned: row i <-> chunks[i]
                "embedding_dim": dim,
                "embedding_model": model_name,
            },
            f,
        )
    print(f"Saved dense index -> {out / 'dense.faiss'}")
    print(f"Saved chunk map    -> {out / 'dense_chunk_map.pkl'}")


def load_dense_index(index_dir: str):
    """Convenience loader for use by the retrieval stage."""
    import faiss  # lazy import, see build_dense_index() above

    out = Path(index_dir)
    index = faiss.read_index(str(out / "dense.faiss"))
    with open(out / "dense_chunk_map.pkl", "rb") as f:
        sidecar = pickle.load(f)
    return index, sidecar["chunks"], sidecar["embedding_model"]


def search_dense(query: str, index, chunks: list[dict], model, k: int = 5):
    """Embed a query WITH BGE's recommended query instruction prepended
    (asymmetric encoding -- see BGE_QUERY_INSTRUCTION above), and return
    the top-k chunks."""
    q_vec = model.encode(
        [BGE_QUERY_INSTRUCTION + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, idxs = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), "chunk": chunks[idx]})
    return results


if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path} -- run chunk.py first.")

    model = load_embedding_model()
    index, embeddings, dim = build_dense_index(chunks, model)
    save_index(index, chunks, dim, out_dir, EMBEDDING_MODEL_NAME)

    # Quick smoke test so you can see it's actually working end-to-end
    # (reuses the already-loaded model instead of loading it a second time)
    demo_query = "What were Apple's total net sales for the nine months ended June 25, 2022?"
    print(f"\nDemo query: {demo_query!r}")
    for r in search_dense(demo_query, index, chunks, model, k=3):
        c = r["chunk"]
        preview = c["search_text"][:120].replace("\n", " ")
        print(f"  score={r['score']:.3f}  page={c['page_number']}  type={c['chunk_type']}  {preview}...")