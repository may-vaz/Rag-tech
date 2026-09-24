"""
main.py
=======
REPL-style entry point for the whole project: a reviewer types ad-hoc
questions and gets answers immediately, without needing to know this
project's internals or re-invoke a script with a hardcoded question
string each time.

USAGE (from the project root)
-----------------------------
    python3 main.py [data/index] [qwen3:4b]

Then type a question at the prompt and press Enter. Type 'exit',
'quit', or press Ctrl+C/Ctrl+D to stop.

MEMORY-SAFETY -- WHY THIS DOESN'T JUST "LOAD ONCE AND LOOP" (a
deliberate, tested design choice, not an oversight)
--------------------------------------------------------------------------
It would be simpler to load the embedding model, reranker, and indexes
ONCE at startup and keep them resident for the whole session. This
script deliberately does NOT do that, because this project's own
testing already found that pattern unreliable: keeping those models in
memory while making repeated, separate Ollama calls in the same process
caused a ReadTimeout on an 8GB machine (see answer.py's module
docstring, "A NOTE ON 8GB MACHINES," and llm_client.py's "PROBLEM 2").
`gc.collect()` only releases Python's own references -- torch and faiss
keep their own native memory pools that don't reliably return memory to
the OS just because Python let go of it, so freeing once and calling
Ollama once (this project's already-verified-safe pattern) does not
mean freeing once and calling Ollama SEVERAL TIMES in the same process
is equally safe.

So instead, for every LOOKUP question (anything the deterministic fact
engine can't answer), this script loads the retrieval models FRESH,
retrieves, then explicitly frees them with `del` + `gc.collect()` --
BEFORE calling Ollama -- every single turn. This costs a few extra
seconds of model-loading time per lookup question, in exchange for
actually being reliable regardless of how many questions get asked in a
row or what hardware a reviewer is running this on. Since I have no knowledge
of the reviewer's machine, trading a few seconds of latency for
reliability is what I went for

COMPUTATION QUESTIONS ARE UNAFFECTED BY ANY OF THIS: sum/difference/
percent_change/ratio questions are answered by the deterministic fact
engine (generate/fact_engine.py) Why: (1) LLMs -- small ones especially -- are unreliable at
arithmetic, so the standard fix is to take the model out of the calculator
role entirely -- this project runs qwen3:4b, which is a small model (2) This project's own BM25 test ranked the needed
table 53rd of 95 chunks for the difference question -- the LLM path
cannot compute over numbers it never retrieves, while the engine
queries data/table_facts.jsonl directly.It answers only when it can prove
every operand, else returns None and the LLM path acts as fallback.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from generate.answer import (  # noqa: E402
    SYSTEM_PROMPT,
    retrieve_context,
    try_engine_first,
)
from generate.llm_client import OLLAMA_MODEL, call_ollama  # noqa: E402


def answer_one(question: str, index_dir: str, model: str) -> None:
    """Answers a single question, printing the result. Tries the
    deterministic fact engine first (no models, milliseconds); only
    loads retrieval models -- freshly, and frees them again before the
    Ollama call -- if the engine declines. See module docstring for why
    models aren't just kept resident across turns."""

    # deterministic computation
    engine_hit = try_engine_first(question)
    if engine_hit is not None:
        print("\n[answered instantly by the deterministic fact engine -- no retrieval, no LLM call]")
        print(f"Answer: {engine_hit['answer']}")
        # Engine hits carry per-table "sources" (same schema as the LLM path)
        pages = sorted({s["page"] for s in engine_hit.get("sources", []) if s.get("page")})
        if pages:
            print(f"Source page(s): {', '.join(str(p) for p in pages)}")
        print()
        return

    # retrieval + LLM, loaded fresh and freed after use. 
    print("(loading retrieval models -- this happens fresh every question, by design; "
          "see this file's module docstring)")
    from retrieve.hybrid_retriever import load_retriever 
    from retrieve.reranker import load_reranker 

    dense_index, dense_chunks, embedding_model, bm25, sparse_chunks = load_retriever(index_dir)
    reranker = load_reranker()

    context, used_chunks = retrieve_context(
        question, dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
    )

    # Free BEFORE calling Ollama -- see module docstring for why this
    # ordering, every single turn, is the actual point of this design.
    del dense_index, dense_chunks, embedding_model, bm25, sparse_chunks, reranker
    gc.collect()

    if context is None:
        print("\nAnswer: This filing does not contain relevant information to answer this question.\n")
        return

    print("(generating answer...)")
    user_prompt = f"CONTEXT:\n\n{context}\n\nQUESTION: {question}"
    try:
        answer_text = call_ollama(SYSTEM_PROMPT, user_prompt, model=model)
    except RuntimeError as e:
        print(f"\nAnswer: (error -- {e})\n")
        return

    print(f"\nAnswer: {answer_text}")
    pages = sorted({c["page_number"] for c in used_chunks})
    print(f"Source page(s): {', '.join(str(p) for p in pages)}\n")


def main() -> None:
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/index"
    model = sys.argv[2] if len(sys.argv) > 2 else OLLAMA_MODEL

    print("=" * 70)
    print("Apple Q3 2022 10-Q -- interactive Q&A")
    print("Type a question and press Enter.")
    print("Computation questions (combined X and Y, difference between..., "
          "percent change..., what percent of X was Y) answer instantly.")
    print("Other questions retrieve from the filing and use the LLM -- "
          "expect this to take longer.")
    print("Type 'exit' or 'quit' (or Ctrl+C / Ctrl+D) to stop.")
    print("=" * 70)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        answer_one(question, index_dir, model)


if __name__ == "__main__":
    main()
