"""

- This filing uses ruled/bordered tables (visible grid lines), confirmed by
  inspecting the raw PDF with pdfplumber's table finder. That means a
  *lines-based* table detector works reliably here.

- Text and tables are extracted SEPARATELY per page, not from one flat
  text stream. For every page, I first locate table bounding boxes,
  then extract the "outside bbox" text for narrative content. This
  prevents duplicate/garbled content where a table's numbers would
  otherwise bleed into a text chunk as meaningless tokens.

- A small amount of header bleed through is unavoidable (a table's
  column headers can sit just above the detected bbox and get counted
  as "text"). We filter these out with a heuristic (lines dominated by
  '$' signs / numeric tokens are dropped from narrative text) rather
  than pretending the boundary is perfect.

- Each extracted table keeps:
    * its raw cell grid (list of rows)
    * a `caption` = the narrative text immediately preceding it on the
      page (e.g. "Note 3 - Financial Instruments" / "The following
      table shows..."), which is critical: a table of bare numbers is
      meaningless to an embedding model without the sentence that says
      what the numbers represent.
    * page number and bounding box, for provenance/citation later.

Output: a single JSON-serializable structure written to
`data/parsed_document.json`, consumed by chunk.py.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pdfplumber

# Data structures



@dataclass
class ParsedTable:
    page_number: int         
    table_index_on_page: int 
    bbox: tuple
    caption: str               # narrative text preceding the table (may be "")
    rows: list                 # list[list[str]] cleaned grid
    filing_page_label: str | None  # e.g. "Apple Inc. | Q3 2022 Form 10-Q | 8" footer, if found
    header_context: str = ""   


@dataclass
class ParsedPage:
    page_number: int
    narrative_text: str
    tables: list = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_path: str
    num_pages: int
    pages: list = field(default_factory=list)

# Helpers
# Footer pattern specific to this filing, e.g. "Apple Inc. | Q3 2022 Form 10-Q | 8"
FOOTER_PATTERN = re.compile(r"[\w .]+\|\s*Q\d\s*\d{4}\s*Form\s*10-Q\s*\|\s*\d+", re.IGNORECASE)



def _looks_like_table_fragment(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN.fullmatch(t))
    return numeric / len(tokens) > 0.55 and len(tokens) >= 3


def clean_stray_table_lines(text: str) -> str:
    """Drop lines in the 'narrative' stream that are actually table debris
    (column headers / number rows bleeding past the detected table bbox)."""
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_table_fragment(stripped):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def extract_footer_label(text: str) -> str | None:
    m = FOOTER_PATTERN.search(text)
    return m.group(0) if m else None


def clean_table_grid(raw_rows: list[list[str | None]]) -> list[list[str]]:
    """
    Raw pdfplumber table extraction produces a lot of None-valued 'ghost'
    columns and None cells. So We:
      1. Drop columns that are None/empty across every row.
      2. Replace remaining None cells with "".
      3. Strip whitespace and collapse internal newlines pdfplumber
         sometimes inserts inside a single cell.
    """
    if not raw_rows:
        return []

    num_cols = max(len(r) for r in raw_rows)
    padded = [list(r) + [None] * (num_cols - len(r)) for r in raw_rows]

    keep_cols = []
    for c in range(num_cols):
        col_vals = [padded[r][c] for r in range(len(padded))]
        if any(v is not None and str(v).strip() for v in col_vals):
            keep_cols.append(c)

    cleaned = []
    for row in padded:
        new_row = []
        for c in keep_cols:
            val = row[c]
            val = "" if val is None else str(val).replace("\n", " ").strip()
            new_row.append(val)
        if any(v for v in new_row):
            cleaned.append(new_row)
    return cleaned


def build_caption_zones(page, sorted_tables) -> list[str]:
    """
    Returns, for each table (in the same top-to-bottom order as
    `sorted_tables`), the raw text that sits BETWEEN the bottom of the
    previous table (or top of page) and the top of this table.
    Using real word positions instead of a
    line-count heuristic fixes both: the caption is now the text
    immediately preceding the table, and it also captures that period
    header row so downstream period detection actually sees it.
    """
    words = page.extract_words()
    zones = []
    prev_bottom = 0.0
    for t in sorted_tables:
        top = t.bbox[1]
        zone_words = [w for w in words if prev_bottom <= w["top"] < top]
        # group words into lines by rounded vertical position
        lines: dict[int, list[str]] = {}
        for w in zone_words:
            key = round(w["top"])
            lines.setdefault(key, []).append(w["text"])
        ordered_lines = [" ".join(lines[k]) for k in sorted(lines.keys())]
        zones.append("\n".join(ordered_lines))
        prev_bottom = t.bbox[3]
    return zones


def caption_from_zone(zone_text: str, max_lines: int = 4) -> str:
    """The caption shown to the LLM: the last few lines of the zone
    immediately above the table (usually the section heading + the
    'the following table shows...' intro sentence)."""
    lines = [l for l in zone_text.splitlines() if l.strip()]
    return " ".join(lines[-max_lines:]) if lines else ""


# Main extraction routine

def parse_pdf(path: str) -> ParsedDocument:
    doc = ParsedDocument(source_path=path, num_pages=0)

    with pdfplumber.open(path) as pdf:
        doc.num_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages, start=1):
            found_tables = page.find_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
            )
            # Sort top-to-bottom so caption zones and numbering are
            # positionally meaningful (pdfplumber's return order isn't
            # guaranteed to already be top-to-bottom).
            sorted_tables = sorted(found_tables, key=lambda t: t.bbox[1])
            text_only_page = page
            for t in sorted_tables:
                text_only_page = text_only_page.outside_bbox(t.bbox)
            raw_outside_text = text_only_page.extract_text() or ""
            narrative_text = clean_stray_table_lines(raw_outside_text)
            footer_label = extract_footer_label(page.extract_text() or "")

            caption_zones = build_caption_zones(page, sorted_tables)

            parsed_tables = []
            for idx, (t, zone_text) in enumerate(zip(sorted_tables, caption_zones)):
                grid = clean_table_grid(t.extract())
                if not grid:
                    continue
                caption = caption_from_zone(zone_text)
                parsed_tables.append(
                    ParsedTable(
                        page_number=i,
                        table_index_on_page=idx,
                        bbox=t.bbox,
                        caption=caption,
                        rows=grid,
                        filing_page_label=footer_label,
                    )
                )
                # Stash the full zone text (not just the trimmed caption)
                # so enrich_metadata.py can detect period headers that sit
                # above the table's bbox but aren't part of the visible
                # caption line, e.g. "Three Months Ended / Nine Months
                # Ended / June 25, June 26, June 25, June 26,".
                parsed_tables[-1].header_context = zone_text

            doc.pages.append(
                ParsedPage(
                    page_number=i,
                    narrative_text=narrative_text,
                    tables=parsed_tables,
                )
            )

    return doc


def save_parsed_document(doc: ParsedDocument, out_path: str) -> None:
    def _default(o):
        if isinstance(o, tuple):
            return list(o)
        return asdict(o)

    payload = {
        "source_path": doc.source_path,
        "num_pages": doc.num_pages,
        "pages": [
            {
                "page_number": p.page_number,
                "narrative_text": p.narrative_text,
                "tables": [asdict(t) for t in p.tables],
            }
            for p in doc.pages
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/10q.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/parsed_document.json"
    parsed = parse_pdf(src)
    save_parsed_document(parsed, out)
    n_tables = sum(len(p.tables) for p in parsed.pages)
    print(f"Parsed {parsed.num_pages} pages, extracted {n_tables} tables -> {out}")