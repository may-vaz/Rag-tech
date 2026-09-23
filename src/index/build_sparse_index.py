"""
Dense embeddings are good at PARAPHRASE / SEMANTIC matching ("net sales
increased" ~ "revenue grew") but are comparatively weak at EXACT-TOKEN
matching -- specific numbers, product codes, identifiers, and financial
instrument names. This document has exactly that kind of content:
"1.375% Notes due 2029", "0.500% Notes due 2031", "$58,107 million",
account line items like "Vendor non-trade receivables". A query like
"what's the interest rate on the notes due 2031" needs the retriever to
find the literal string "0.500%" and "2031" together -- BM25 (a
term-frequency / inverse-document-frequency ranking function) is
extremely good at exactly this, and a semantic embedding model can
easily rank a totally different note higher because it's "about
similar things" without matching the specific digits.



IMPLEMENTATION CHOICE -- rank_bm25 (BM25Okapi)
--------------------------------------------------------------------------------
rank_bm25 is a small, dependency-free,
pure-Python implementation of the same ranking algorithm Elasticsearch
uses under the hood, and is trivial to swap out for a real search
service later if the corpus grows to many documents -- the BM25 ranking
math doesn't change, only the serving infrastructure would.

TOKENIZATION
--------------
A simple, transparent tokenizer is used deliberately: lowercase,
split on non-alphanumeric characters, keep numbers and the '%' /
'.' inside numeric tokens (so "1.375%" and "58,107" survive as
matchable units instead of being shredded into "1", "375", "58",
"107"). This matters specifically for this document, where the exact
numeric/percentage tokens ARE the query targets.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

# Matches: words, and numeric tokens that may include commas, one decimal
# point, and a trailing percent sign -- e.g. "1.375%", "58,107", "2031".
TOKEN_RE = re.compile(r"[a-zA-Z]+|\d[\d,]*\.?\d*%?")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_sparse_index(chunks: list[dict]):
    tokenized_corpus = [tokenize(c["search_text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def save_sparse_index(bm25, chunks: list[dict], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sparse_bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)  # index-aligned, same convention as the dense sidecar
    print(f"Saved sparse index -> {out / 'sparse_bm25.pkl'}")


def load_sparse_index(index_dir: str):
    out = Path(index_dir)
    with open(out / "sparse_bm25.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["chunks"]


def search_sparse(query: str, bm25, chunks: list[dict], k: int = 5):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"score": float(scores[i]), "chunk": chunks[i]} for i in ranked_idx]


if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path} -- run chunk.py first.")

    bm25, tokenized_corpus = build_sparse_index(chunks)
    save_sparse_index(bm25, chunks, out_dir)

    avg_tokens = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
    print(f"Indexed {len(chunks)} chunks, avg {avg_tokens:.1f} tokens/chunk")

    # Smoke test on exactly the kind of exact-match query dense embeddings struggle with 
    demo_query = "0.500% Notes due 2031"
    print(f"\nDemo query (exact-match test): {demo_query!r}")
    for r in search_sparse(demo_query, bm25, chunks, k=3):
        c = r["chunk"]
        preview = c["search_text"][:120].replace("\n", " ")
        print(f"  score={r['score']:.3f}  page={c['page_number']}  type={c['chunk_type']}  {preview}...")