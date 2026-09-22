"""
1. NARRATIVE TEXT -> recursive, section-aware splitting.
   - Split first on detected section/Note headings (from enrich_metadata),
     never mid-section, because MD&A explanations ("Mac net sales
     decreased due to...") lose their causal meaning if severed from
     their heading and context.
   - Within a section, split further using a recursive character
     splitter (paragraph -> sentence -> hard token limit) with overlap,
   - Target chunk size: ~120-220 words (roughly 180-320 tokens), with a
     ~15% overlap between consecutive chunks in the same section. This
     range was chosen empirically for this document: MD&A paragraphs
     here run 3-6 sentences and a full explanation (cause + effect) is
     usually under 150 words, so this keeps one causal explanation
     mostly intact in one chunk while staying small enough for precise
     retrieval.

2. TABLES -> ONE CHUNK PER TABLE.
   - A table is serialized to a Markdown table (row/column structure is
     preserved, which is what lets the downstream LLM actually read it
     correctly -- flattening a table to prose loses alignment between a
     number and its row/column label).
   - The chunk's embeddable text is NOT the raw markdown alone. Raw
     numbers embed poorly (an embedding model has no strong signal to
     distinguish "58,107" from "63,090"). Instead we prepend:
       (a) the table's caption (the sentence that says what it is), and
       (b) an auto-generated one-line summary built from the section +
           statement_type + fiscal period tags, e.g.:
           "Balance sheet data: Total assets and liabilities as of
            June 25, 2022 and September 25, 2021 (Condensed Consolidated
            Balance Sheets)."
     This gives the embedding model a real semantic handle while the
     LLM-facing chunk still carries the full markdown table for exact
     lookups.
   - If a table is very large (kept whole regardless -- see rationale
     below), we still cap what's embedded for search purposes vs. what's
     passed to generation, using the same "small-to-big" split: the
     SEARCH TEXT is caption + summary + column headers; the GENERATION
     TEXT (stored in metadata, injected at answer time) is the complete
     table. This means retrieval matches on what the table is ABOUT,
     while generation still receives every row of numbers.

   Why never split a table: splitting a table means some chunk has
   numbers with no header row, which is unusable and actively harmful
   (a model will confidently fabricate what column those numbers
   belong to). No table in this filing exceeds embedding-context limits
   (largest is ~16 rows x 8 cols), so there's no forcing function to
   split here -- if a future filing had a 200-row table, the extension
   point is to keep headers replicated into each sub-chunk, not to
   split blindly.

Output: `data/chunks.jsonl`, one JSON object per line, ready for
indexing by build_dense_index.py / build_sparse_index.py.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Config
TARGET_WORDS = 170          
MAX_WORDS = 260              
OVERLAP_WORDS = 25            
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\$])")


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str           
    search_text: str            # what gets embedded / BM25-indexed
    generation_text: str        # what gets passed to the LLM if this chunk is retrieved
    page_number: int
    section: str | None
    statement_type: str
    fiscal_periods: list
    is_multi_period: bool
    filing_page_label: str | None
    parent_id: str | None = None   # for future small-to-big expansion (parent = full section)



# Text chunking
def split_into_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return SENTENCE_SPLIT_RE.split(text)


def recursive_chunk_sentences(sentences: list[str], target: int, max_words: int, overlap: int) -> list[str]:
    """
    Greedy sentence-packing splitter: adds whole sentences to the current
    chunk until the target word count is reached, then starts a new chunk
    that begins with the last `overlap`-worth of words from the previous
    chunk (so context isn't lost at the boundary). A sentence is never
    split mid-sentence. If a single sentence alone exceeds max_words
    (rare in this document, but possible in a long legal paragraph), it
    is kept as its own chunk rather than cut arbitrarily.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_word_count = 0

    def flush():
        nonlocal current, current_word_count
        if current:
            chunks.append(" ".join(current))
        current = []
        current_word_count = 0

    for sent in sentences:
        n_words = len(sent.split())

        if n_words > max_words:
            flush()
            chunks.append(sent)  # oversized sentence stands alone
            continue

        if current_word_count + n_words > max_words:
            flush()

        current.append(sent)
        current_word_count += n_words

        if current_word_count >= target:
            flush_chunk_text = " ".join(current)
            chunks.append(flush_chunk_text)
            # seed next chunk with overlap words from the tail of this one
            tail_words = flush_chunk_text.split()[-overlap:]
            current = [" ".join(tail_words)] if tail_words else []
            current_word_count = len(tail_words)

    flush()
    return [c for c in chunks if c.strip()]


def make_text_chunks(page: dict) -> list[Chunk]:
    sentences = split_into_sentences(page["narrative_text"])
    if not sentences:
        return []

    pieces = recursive_chunk_sentences(sentences, TARGET_WORDS, MAX_WORDS, OVERLAP_WORDS)

    out = []
    for piece in pieces:
        out.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_type="text",
                search_text=piece,
                generation_text=piece,
                page_number=page["page_number"],
                section=page.get("section"),
                statement_type=page.get("statement_type", "other"),
                fiscal_periods=page.get("fiscal_periods_detected", []),
                is_multi_period=len(page.get("fiscal_periods_detected", [])) > 1,
                filing_page_label=None,
            )
        )
    return out


# Table chunking
def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        # pad/truncate row to header length defensively
        r = (r + [""] * len(header))[: len(header)]
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def build_table_summary(table: dict) -> str:
    stype = table.get("statement_type", "other").replace("_", " ")
    section = table.get("section") or "this filing"
    periods = table.get("fiscal_periods_detected", [])
    period_str = ", ".join(p.replace("_", " ") for p in periods) if periods else "the reported period"
    return f"Table from '{section}' ({stype} data), covering: {period_str}."


def make_table_chunk(table: dict) -> Chunk:
    markdown_table = rows_to_markdown(table["rows"])
    summary = build_table_summary(table)
    caption = table.get("caption", "")

    # Some tables (notably the income statement / statements-of-operations style layouts) have their period-header row ("Three Months Ended /
    # Nine Months Ended ... June 25, June 26, ...") positioned ABOVE the detected table bbox by pdfplumber's line-based extractor, so it
    # never makes it into `rows`. We surface it explicitly here rather
    # than silently losing it otherwise the LLM sees bare numbers with no idea which of the 4 period columns each one belongs to

    header_ctx = table.get("header_context", "").strip()
    period_note = (
        f"\n[Column period headers, positioned above the table grid: {header_ctx}]\n"
        if header_ctx and header_ctx not in caption
        else ""
    )

    # SEARCH TEXT: caption + auto-summary + header row only.
    # Deliberately excludes the bulk numeric body, numbers add embedding
    # noise, not signal, for semantic search.
    header_row = table["rows"][0] if table["rows"] else []
    search_text = f"{summary}\n{caption}{period_note}\nColumns: {', '.join(header_row)}"

    # GENERATION TEXT: everything, in full, so the LLM can compute/quote
    # exact figures once this chunk is retrieved.
    generation_text = f"{caption}{period_note}\n\n{markdown_table}"

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        chunk_type="table",
        search_text=search_text,
        generation_text=generation_text,
        page_number=table["page_number"],
        section=table.get("section"),
        statement_type=table.get("statement_type", "other"),
        fiscal_periods=table.get("fiscal_periods_detected", []),
        is_multi_period=table.get("is_multi_period", False),
        filing_page_label=table.get("filing_page_label"),
    )



# Orchestration
def chunk_document(enriched_doc: dict) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in enriched_doc["pages"]:
        all_chunks.extend(make_text_chunks(page))
        for table in page["tables"]:
            all_chunks.append(make_table_chunk(table))
    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/parsed_document.enriched.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.jsonl"

    with open(src, "r", encoding="utf-8") as f:
        enriched = json.load(f)

    chunks = chunk_document(enriched)
    save_chunks(chunks, out)

    n_text = sum(1 for c in chunks if c.chunk_type == "text")
    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    n_multi = sum(1 for c in chunks if c.is_multi_period)
    print(f"Created {len(chunks)} chunks -> {out}")
    print(f"  text chunks:  {n_text}")
    print(f"  table chunks: {n_table}")
    print(f"  multi-period chunks flagged for disambiguation: {n_multi}")