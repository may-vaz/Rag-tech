"""
chunk.py -- narrative + table chunking (fixed table side, identical text side)
==============================================================================
TEXT CHUNKING IS UNCHANGED (same splitter, sizes, overlap): text answers
cannot regress from this file.

TABLE CHUNKS consume the new parse output. Three changes, all strictly
additive signal for retrieval+generation:

1. Row labels are now part of SEARCH text ("Rows: iPhone; Mac; ...").
   The old search text (caption + summary + column headers) never named
   the rows, so a BM25 query for "combined iPhone and Mac" had no keyword
   hit on the very table holding both numbers -- one reason computation
   retrieval measured so poorly (rank 53). Numbers are still excluded
   (embedding noise); labels are pure signal. This can only help lookups
   too ("Mac net sales" now hits the revenue table by keyword).

2. Units are stated explicitly in GENERATION text when the caption doesn't
   already state them, so the LLM never has to guess scale.

3. fiscal_periods = union of enrich_metadata's tags (kept as-is) and tags
   derived from the new qualified columns[] (Q_JUN2022 / NINEMO_JUN2022 /
   ASOF_SEP2021 style). The period-summary line in build_context() keeps
   working exactly as before, with more accurate coverage.

Robustness: every new parse key is accessed via .get() -- if
enrich_metadata.py rebuilds table dicts and drops unknown keys, this file
still works (headers/labels baked into rows[] survive regardless, since
rows[] is a key enrich must already preserve).

Output: `data/chunks.jsonl` (same schema as before, plus `units` and
`table_kind` metadata on table chunks).
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Config (unchanged)
TARGET_WORDS = 170
MAX_WORDS = 260
OVERLAP_WORDS = 25
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\$])")

_MONTH_ABBR = ["", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str
    search_text: str
    generation_text: str
    page_number: int
    section: str | None
    statement_type: str
    fiscal_periods: list
    is_multi_period: bool
    filing_page_label: str | None
    header_confidence: str | None = None
    parent_id: str | None = None
    units: str | None = None
    table_kind: str | None = None


# Text chunking (UNCHANGED)
def split_into_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return SENTENCE_SPLIT_RE.split(text)


def recursive_chunk_sentences(sentences: list[str], target: int, max_words: int, overlap: int) -> list[str]:
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
            chunks.append(sent)
            continue

        if current_word_count + n_words > max_words:
            flush()

        current.append(sent)
        current_word_count += n_words

        if current_word_count >= target:
            flush_chunk_text = " ".join(current)
            chunks.append(flush_chunk_text)
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
                header_confidence=None,
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
        r = (r + [""] * len(header))[: len(header)]
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def build_table_summary(table: dict) -> str:
    stype = table.get("statement_type", "other").replace("_", " ")
    section = table.get("section") or "this filing"
    periods = table.get("fiscal_periods_detected", [])
    period_str = ", ".join(p.replace("_", " ") for p in periods) if periods else "the reported period"
    return f"Table from '{section}' ({stype} data), covering: {period_str}."


def _column_period_tags(columns: list[dict]) -> list[str]:
    """Derive enrich-style period tags from qualified columns[]:
    3mo Jun 2022 -> Q_JUN2022, 9mo -> NINEMO_JUN2022, point-in-time ->
    ASOF_SEP2021. Change columns are not periods and are skipped."""
    tags: list[str] = []
    for c in columns or []:
        if c.get("is_change") or not c.get("date"):
            continue
        dur = (c.get("duration") or "").lower()
        prefix = {"3mo": "Q_", "6mo": "SIXMO_", "9mo": "NINEMO_",
                  "12mo": "TWELVEMO_", "point": "ASOF_"}.get(dur, "")
        if not prefix:
            continue
        try:
            tags.append(f"{prefix}{_MONTH_ABBR[int(c['month']) ]}{int(c['year'])}")
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return tags


def make_table_chunk(table: dict) -> Chunk:
    markdown_table = rows_to_markdown(table["rows"])
    summary = build_table_summary(table)
    caption = table.get("caption", "")
    confidence = table.get("header_confidence", "generic")
    units = table.get("units") or ""
    units_note = table.get("units_note") or ""

    header_ctx = table.get("header_context", "").strip()
    column_note = (
        f"\n[This table's real column headers, as printed in the filing (the grid "
        f"above uses placeholder column labels because they could not be "
        f"automatically reconstructed): {header_ctx}]\n"
        if confidence == "generic" and header_ctx
        else ""
    )

    # Units line: the parse layer reports table scale separately from the
    # caption; state it explicitly unless the caption already does.
    units_line = ""
    if units and units.lower() not in caption.lower():
        units_line = f"\n[All figures in this table are {units}.]\n"
    elif units_note and "in millions" not in caption.lower() and "in thousands" not in caption.lower():
        units_line = f"\n[Units: {units_note}.]\n"

    header_row = table["rows"][0] if table["rows"] else []
    row_labels = [r[0].strip() for r in table["rows"][1:]
                  if r and r[0].strip() and not r[0].strip().endswith(":")]
    # Parent-qualified labels ("Americas: Net sales") also contribute
    # their core ("Net sales") so both specific and bare queries match.
    label_terms: list[str] = []
    for lab in row_labels:
        label_terms.append(lab)
        if ": " in lab:
            core = lab.split(": ", 1)[-1]
            if core and core != lab:
                label_terms.append(core)
    seen: set[str] = set()
    uniq_labels = [l for l in label_terms if not (l in seen or seen.add(l))]

    search_text = (
        f"{summary}\n{caption}"
        f"{column_note}\nColumns: {', '.join(c for c in header_row if c)}"
        f"\nRows: {'; '.join(uniq_labels)}"
    )
    generation_text = f"{caption}{units_line}{column_note}\n\n{markdown_table}"

    periods = list(table.get("fiscal_periods_detected", []) or [])
    for tag in _column_period_tags(table.get("columns") or []):
        if tag not in periods:
            periods.append(tag)

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        chunk_type="table",
        search_text=search_text,
        generation_text=generation_text,
        page_number=table["page_number"],
        section=table.get("section"),
        statement_type=table.get("statement_type", "other"),
        fiscal_periods=periods,
        is_multi_period=table.get("is_multi_period", len(periods) > 1),
        filing_page_label=table.get("filing_page_label"),
        header_confidence=confidence,
        units=units or None,
        table_kind=table.get("table_kind"),
    )


# Orchestration (unchanged shape)
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
    from collections import Counter
    conf = Counter(c.header_confidence for c in chunks if c.chunk_type == "table")
    n_multi = sum(1 for c in chunks if c.is_multi_period)
    print(f"Created {len(chunks)} chunks -> {out}")
    print(f"  text chunks:  {n_text}")
    print(f"  table chunks: {n_table} {dict(conf)}")
    print(f"  multi-period chunks flagged for disambiguation: {n_multi}")
