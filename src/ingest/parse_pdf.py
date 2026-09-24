"""
parse_pdf.py 
PDF parsing, table reconstruction, structured fact extraction.

Parses the 10-Q PDF into two artifacts:

1. data/parsed_document.json -- pages with narrative text plus reconstructed
   tables (caption, header row, data rows, fiscal-period headers, units,
   filing page label for citations). rows[0] is always the header row.
2. data/table_facts.jsonl -- one flat record per table cell (row label,
   period header, numeric value, units, page). fact_engine.py computes over
   this file, so arithmetic answers come from plain Python over exact
   cells, never from LLM arithmetic.

Parsing approach (all driven by the actual layout of this filing):
- Lines-based table detection: this filing uses ruled/bordered tables.
  Table bounding boxes are located first; narrative text is extracted from
  outside those boxes so table numbers never leak into prose as garbled
  tokens, and numeric-heavy stray lines past the boundary are dropped.
- Caption zones: each table keeps its caption plus the sentence before it,
  since raw numbers carry no retrievable meaning without knowing what the
  table represents.
- Leaked-row recovery: some first data rows leak above the ruled box as
  two lines (label on one line, values on the next, e.g. the Note 2
  "iPhone" row). The label is pulled from the line above when the value
  line carries none, with a column-count match required so header lines
  can never match.
- $/% cell merging: value groups arrive separated by empty columns;
  empties are dropped first, then "$" merges with the following token and
  "%" with the preceding one ("8 %" -> "8%").
- Qualified period headers: the same date repeats across 3-month and
  9-month columns, so headers carry their duration grouping ("Three months
  ended June 25, 2022", "Change (three months)", ...) plus structured
  columns[] (date, duration, year, is_change). Duplicate labels are
  impossible by construction; a header_confidence flag ("date"/"generic")
  records whether real period labels were recovered.
- Wrapped-word headers: category tables stack header words vertically
  ("Number of / RSUs / (in thousands)"); value-token x-positions are
  clustered into logical columns and each header word assigned to its
  column, falling back to generic labels if anything looks off.
- Degenerate-row repair: where a ruled row spans only part of the page
  width, the row is rebuilt from word positions inside the bbox whenever
  the label is empty or the value count disagrees with the header.
- Parent propagation: repeated labels ("Net sales" 5x, once per region)
  are qualified via "Xxx:" sub-header rows ("Americas: Net sales").
- Junk filtering + units: the Table of Contents parses as phantom tables
  and is skipped; units ("in millions") and currency markers are captured
  per table/column/cell.

CLI: python parse_pdf.py [src_pdf] [out_json] [facts_jsonl]
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
    caption: str
    rows: list                    # display grid; rows[0] is ALWAYS the header
    filing_page_label: str | None
    header_context: str = ""
    header_reconstructed: bool = False
    header_confidence: str = "none"   # "date" | "category" | "single" | "generic" | "none"
    units: str = ""                   # e.g. "in millions"
    units_note: str = ""              # full raw units sentence when mixed/complex
    currency: str = ""                # "$" if any cell in the table paired with $
    table_kind: str = "other"         # "financial_period" | "category" | "single_point" | "other"
    columns: list = field(default_factory=list)   # structured col metadata (see build_*)
    records: list = field(default_factory=list)   # structured facts (see build_records)
    row_kinds: list = field(default_factory=list)  # parallel to rows: header|data|subheader|blank


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


# --------------------------------------------------------------------------
# Shared regexes / constants
# --------------------------------------------------------------------------

FOOTER_PATTERN = re.compile(r"[\w .]+\|\s*Q\d\s*\d{4}\s*Form\s*10-Q\s*\|\s*\d+", re.IGNORECASE)
_NUMERIC_TOKEN = re.compile(r"[\d,.\$%\(\)—-]+")

_THOUSANDS_TOKEN = re.compile(r"^\(?-?\d{1,3}(,\d{3})+(\.\d+)?\)?%?$")
_DECIMAL_TOKEN = re.compile(r"^\(?-?\d+\.\d+\)?%?$")
_DASH_TOKEN = re.compile(r"^—$")
_YEAR_TOKEN_RE = re.compile(r"^(19|20)\d{2}$")
_PCT_JOIN_RE = re.compile(r"^(\(?-?[\d,]+(\.\d+)?\)?)\s+%$")   # "8 %" / "(10) %" -> "8%" / "(10)%"
_MONTH_DAY_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?", re.IGNORECASE
)
_FULL_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),?\s+((?:19|20)\d{2})", re.IGNORECASE
)
_UNITS_RE = re.compile(
    r"\((?:dollars\s+)?in\s+(millions|thousands|billions)[^)]*\)", re.IGNORECASE
)
_MIXED_UNITS_RE = re.compile(
    r"\(([^)]*(?:millions|thousands|billions)[^)]*(?:shares|per share|RSU)[^)]*)\)",
    re.IGNORECASE,
)
_DURATION_RES = [
    (re.compile(r"three\s+months?\s+ended", re.I), "3mo", "Three months ended"),
    (re.compile(r"six\s+months?\s+ended", re.I), "6mo", "Six months ended"),
    (re.compile(r"nine\s+months?\s+ended", re.I), "9mo", "Nine months ended"),
    (re.compile(r"twelve\s+months?\s+ended", re.I), "12mo", "Twelve months ended"),
]
_FOOTNOTE_SUFFIX_RE = re.compile(r"(?:\s*\(\d+\))+\s*$")   # trailing "(1)(2)" refs
_JUNK_CAPTION_RE = re.compile(r"TABLE OF CONTENTS|EXHIBIT\s+(NUMBER|DESCRIPTION)", re.I)

_MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}

# Gap (in PDF points) that separates two logical columns when clustering
# value-token x-positions. Within-column jitter is ~2-5pt; column pitch in
# this filing is ~60-100pt, so 25pt is comfortably in between.
X_GAP_THRESHOLD = 25.0


# --------------------------------------------------------------------------
# Text helpers (unchanged behavior from the original)
# --------------------------------------------------------------------------

def _looks_like_table_fragment(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN.fullmatch(t))
    return numeric / len(tokens) > 0.55 and len(tokens) >= 3


def clean_stray_table_lines(text: str) -> str:
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


def clean_table_grid(raw_rows: list[list[str | None]]) -> tuple[list[list[str]], list[int]]:
    """Same as before: drop all-empty columns, None -> '', collapse \\n.
    Also drops fully-empty ROWS -- but returns the kept original row
    indices so callers can still map grid rows back to pdfplumber's ruled
    row bboxes (needed for word-position repairs)."""
    if not raw_rows:
        return [], []
    num_cols = max(len(r) for r in raw_rows)
    padded = [list(r) + [None] * (num_cols - len(r)) for r in raw_rows]
    keep_cols = []
    for c in range(num_cols):
        col_vals = [padded[r][c] for r in range(len(padded))]
        if any(v is not None and str(v).strip() for v in col_vals):
            keep_cols.append(c)
    cleaned = []
    kept_idx: list[int] = []
    for ri, row in enumerate(padded):
        new_row = []
        for c in keep_cols:
            val = row[c]
            val = "" if val is None else str(val).replace("\n", " ").strip()
            new_row.append(val)
        if any(v for v in new_row):
            cleaned.append(new_row)
            kept_idx.append(ri)
    return cleaned, kept_idx


def build_caption_zones(page, sorted_tables) -> list[str]:
    """Unchanged: raw text between previous table bottom and this table top."""
    words = page.extract_words()
    zones = []
    prev_bottom = 0.0
    for t in sorted_tables:
        top = t.bbox[1]
        zone_words = [w for w in words if prev_bottom <= w["top"] < top]
        lines: dict[int, list[str]] = {}
        for w in zone_words:
            key = round(w["top"])
            lines.setdefault(key, []).append(w["text"])
        ordered_lines = [" ".join(lines[k]) for k in sorted(lines.keys())]
        zones.append("\n".join(ordered_lines))
        prev_bottom = t.bbox[3]
    return zones


def caption_from_zone(zone_text: str, max_lines: int = 4) -> str:
    lines = [l for l in zone_text.splitlines() if l.strip()]
    return " ".join(lines[-max_lines:]) if lines else ""


# --------------------------------------------------------------------------
# Value-token machinery (shared by grid merge, leaked-row parse, word repair)
# --------------------------------------------------------------------------

_PAREN_NUMBER_RE = re.compile(r"^\(?-?(\d[\d,]*(?:\.\d+)?)\)?%?$")


def _is_rich_number(tok: str) -> bool:
    if _THOUSANDS_TOKEN.match(tok) or _DECIMAL_TOKEN.match(tok) or _DASH_TOKEN.match(tok):
        return True
    # Paren-wrapped numbers are financial negatives even when small --
    # "(719)", "(13)" -- but a bare short "(N)" alone also matches
    # footnote refs, so callers needing a value-REGION START use
    # _is_value_start (context-aware) instead of this.
    return bool(_PAREN_NUMBER_RE.match(tok) and tok.startswith("("))


def _is_value_start(tok: str, nxt: str | None = None) -> bool:
    """Can this token begin the value region of a row? $ and rich numbers
    always can. A paren number always can EXCEPT a short "(N)" (1-2
    digits, no comma/dot -- the footnote shape) when it is last or
    followed by $/another value (the "(1) $ 40,665" footnote case);
    "(13) 189" mid-line IS a value start ("(13)" followed by a bare
    number), while trailing "Programs (1)" is not. Bare 4-digit years
    are never starts."""
    if tok == "$" or _THOUSANDS_TOKEN.match(tok) or _DECIMAL_TOKEN.match(tok) \
            or _DASH_TOKEN.match(tok):
        return True
    m = _PAREN_NUMBER_RE.match(tok)
    if not (m and tok.startswith("(")):
        return False
    digits = re.sub(r"\D", "", m.group(1))
    if "," in m.group(1) or "." in m.group(1) or len(digits) >= 3:
        return True
    # Short "(N)": value start only mid-line before a BARE token.
    if nxt is None:
        return False
    return not (nxt == "$" or nxt == "%" or _is_rich_number(nxt)
                or _YEAR_TOKEN_RE.match(nxt))


def normalize_label(label: str) -> str:
    """Display-ready row label: strip trademark/footnote noise that hurts
    matching ('Mac(R) (1)' -> 'Mac'), while keeping meaningful parens
    ('Other income/(expense), net' untouched -- only trailing (digits)
    groups are footnotes). Also detaches a '$' glued to the label end
    ('Selling, general and administrative $' -> label without '$'; the
    caller re-injects the $ as a value marker)."""
    s = label.replace("®", "").replace("\u00ae", "")
    s = re.sub(r"\s+", " ", s).strip()
    had_dollar = False
    if s.endswith("$"):
        s = s[:-1].rstrip()
        had_dollar = True
    # A footnote can hide before a trailing colon ("Level 1 (1):").
    trailing_colon = s.endswith(":")
    if trailing_colon:
        s = s[:-1].rstrip()
    s = _FOOTNOTE_SUFFIX_RE.sub("", s).strip()
    if trailing_colon:
        s = s + ":"
    s = re.sub(r"\s+", " ", s).strip()
    return (s, had_dollar) if had_dollar else (s, False)


def pair_value_tokens(tokens: list[str]) -> tuple[list[str], bool]:
    """Pair a flat token stream into logical values.
    Rules: '$' + next -> next (currency noted); X + '%' -> 'X%';
    otherwise the token stands alone. Returns (values, saw_dollar)."""
    normed: list[str] = []
    for t in tokens:
        m = _PCT_JOIN_RE.match(t)
        normed.append(f"{m.group(1)}%" if m else t)
    values: list[str] = []
    saw_dollar = False
    i = 0
    while i < len(normed):
        a = normed[i]
        b = normed[i + 1] if i + 1 < len(normed) else None
        if a == "$" and b is not None:
            values.append(b)
            saw_dollar = True
            i += 2
        elif b == "%":
            values.append(f"{a}%" if not a.endswith("%") else a)
            i += 2
        elif a == "%":
            # Defensive: a lone % (shouldn't happen after normalization,
            # but never crash) -- attach to previous value if possible.
            if values and not values[-1].endswith("%"):
                values[-1] = values[-1] + "%"
            i += 1
        else:
            values.append(a)
            i += 1
    return values, saw_dollar


def merge_row_values(row: list[str], label_cols: int = 1) -> tuple[str, list[str], bool]:
    """FIX 2. Replaces merge_dollar_value_pairs. Returns
    (label, values, saw_dollar). Empty cells are dropped FIRST so ''
    separator columns can never shift alignment; $ and % are paired on
    the dense token stream."""
    label_raw = " ".join(c for c in row[:label_cols]).strip()
    rest_cells = [c.strip() for c in row[label_cols:] if c is not None and str(c).strip() != ""]
    # Wrapped-label guard: a "value" cell holding real prose (letters +
    # long/multi-word) is a wrapped label fragment that landed in a value
    # column, not a number -- reattach it to the label instead of letting
    # it shift every value one column left. Genuine values never contain
    # letters ("$", "%", "(2,008)", "8%" are all letter-free).
    frags = [c for c in rest_cells
             if re.search(r"[A-Za-z]", c) and (len(c) > 12 or len(c.split()) > 1)]
    rest = [c for c in rest_cells if c not in frags]
    label, label_dollar = normalize_label((label_raw + " " + " ".join(frags)).strip())
    if label_dollar:
        rest = ["$"] + rest
    values, saw_dollar = pair_value_tokens(rest)
    return label, values, (saw_dollar or label_dollar)


def split_label_values(line: str) -> tuple[str, list[str], bool]:
    """Split a flat zone/word line ('Label text $ 12,852 ...') into
    (label, values, saw_dollar). The value region starts at the first $
    or rich-number token; a bare number immediately followed by '%' also
    starts it ('Percentage ... 8 % 7 % ...')."""
    tokens = line.split()
    start = None
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if _is_value_start(t, nxt):
            start = i
            break
        if nxt == "%" and re.fullmatch(r"\(?-?[\d,]+(\.\d+)?\)?", t):
            start = i
            break
    if start is None:
        lab, _ = normalize_label(line.strip())
        return lab, [], False
    label, _ = normalize_label(" ".join(tokens[:start]))
    values, saw = pair_value_tokens(tokens[start:])
    return label, values, saw


def parse_value_number(raw: str) -> tuple[float | None, bool]:
    """Parse a logical value token into (signed float, is_percent).
    '(719)' -> (-719, False); '(10)%' -> (-10, True); '—'/'' -> (None,
    False). Returns (None, False) when there is no number."""
    s = raw.strip()
    if not s or s == "—" or s == "-":
        return None, False
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "").replace("$", "").strip()
    if not s or s == "—" or s == "-":
        return None, False
    try:
        v = float(s)
    except ValueError:
        return None, False
    return (-v if neg else v), is_pct


# --------------------------------------------------------------------------
# FIX 1: multi-line leaked-row recovery
# --------------------------------------------------------------------------

def _line_has_year_tokens(line: str) -> bool:
    return sum(1 for t in line.split() if _YEAR_TOKEN_RE.match(t)) >= 1


def recover_leaked_rows(zone_lines: list[str], expected_cols: int,
                        change_slots: list[int] | None = None,
                        max_rows: int = 2) -> tuple[list[tuple[str, list[str], bool]], list[str]]:
    """Recover data rows pdfplumber scoped ABOVE the table bbox. Checks the
    last zone line: if it parses into a label + value list matching the
    expected column count (or the Change-blank-tolerant count), it is a
    leaked row. If its label part is empty/footnote-only ('(1)'), the label
    is pulled from the line above (the iPhone case). Consumed lines are
    removed. Header/detail lines can never match: bare years are not value
    tokens, and a guard rejects lines holding 2+ year tokens outright."""
    change_slots = change_slots or []
    recovered: list[tuple[str, list[str], bool]] = []
    lines = list(zone_lines)
    while lines and len(recovered) < max_rows:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        toks = last.split()
        # Guard: year/detail lines and Change-header lines are never rows.
        if sum(1 for t in toks if _YEAR_TOKEN_RE.match(t)) >= 2:
            break
        if toks and all(_YEAR_TOKEN_RE.match(t) or t.lower() == "change" for t in toks):
            break
        label, values, saw = split_label_values(last)
        if not values:
            break
        n_change = len(change_slots)
        ok = (len(values) == expected_cols or
              (n_change and len(values) == expected_cols - n_change
               and not any(v.endswith("%") for v in values)))
        if not ok:
            break
        # Multi-line label: value line held no real label -> look one up.
        if (not label or re.fullmatch(r"(?:\(\d+\))+", label.replace(" ", ""))) and len(lines) >= 2:
            prev = lines[-2].strip()
            plabel, pvalues, _ = split_label_values(prev)
            if plabel and not pvalues and not _line_has_year_tokens(prev):
                label = plabel
                lines.pop()  # consume the label line too
        if not label:
            break  # a row with no label at all is not safely recoverable
        recovered.append((label, values, saw))
        lines.pop()
    recovered.reverse()
    return recovered, lines


# --------------------------------------------------------------------------
# FIX 3: qualified period headers (date + duration + Change slots)
# --------------------------------------------------------------------------

def _find_detail_index(zone_lines: list[str]) -> int | None:
    for idx in range(len(zone_lines) - 1, -1, -1):
        toks = zone_lines[idx].split()
        if any(_YEAR_TOKEN_RE.match(t) or t.lower() == "change" for t in toks):
            # Must be PRIMARILY a detail line (years/Change + maybe nothing
            # else), not a prose sentence that merely mentions a year.
            if all(_YEAR_TOKEN_RE.match(t) or t.lower() == "change"
                   or _MONTH_DAY_RE.fullmatch(t) or t in {"June", "March", "September"}
                   for t in toks):
                return idx
            # Fallback: accept if it contains 2+ year tokens (detail lines
            # always do; prose sentences essentially never do twice).
            if sum(1 for t in toks if _YEAR_TOKEN_RE.match(t)) >= 2:
                return idx
    return None


def _parse_grouping_durations(zone_lines: list[str], detail_idx: int) -> list[tuple[str, str]]:
    """Parse duration phrases ('Three Months Ended ... Nine Months Ended')
    from the lines directly above the detail line. Returns e.g.
    [('3mo', 'Three months ended'), ('9mo', 'Nine months ended')] in order."""
    groups: list[tuple[str, str]] = []
    for idx in range(max(0, detail_idx - 2), detail_idx):
        line = zone_lines[idx]
        hits: list[tuple[int, str, str]] = []
        for rx, code, label in _DURATION_RES:
            for m in rx.finditer(line):
                hits.append((m.start(), code, label))
        hits.sort()
        for _, code, label in hits:
            if not groups or groups[-1][0] != code:
                groups.append((code, label))
    return groups


def _month_days_above(zone_lines: list[str], detail_idx: int) -> list[str]:
    collected: list[list[str]] = []
    scan = detail_idx - 1
    while scan >= 0:
        ms = [m.group(0).rstrip(",") for m in _MONTH_DAY_RE.finditer(zone_lines[scan])]
        if not ms:
            break
        collected.append(ms)
        scan -= 1
    out: list[str] = []
    for ms in reversed(collected):
        out.extend(ms)
    return out


def _iso_date(month_name: str, day: str, year: str) -> str | None:
    try:
        return f"{int(year):04d}-{_MONTH_TO_NUM[month_name.lower()]:02d}-{int(day):02d}"
    except (KeyError, ValueError):
        return None


def build_period_header(zone_lines: list[str], expected_cols: int,
                        detail_word_centers: list[tuple[str, float]] | None = None
                        ) -> tuple[list[str], list[dict], list[int], str] | None:
    """FIX 3. Returns (header_row, columns, change_slots, confidence) or
    None when no confident date reconstruction is possible. Each date
    column is qualified with its duration group so 3-month and 9-month
    columns sharing a date are unambiguous."""
    detail_idx = _find_detail_index(zone_lines)
    if detail_idx is None:
        return None
    tokens = zone_lines[detail_idx].split()
    # Token stream must be years/Change only (month words would already
    # have failed _find_detail_index's strict check, but re-verify).
    if not tokens or not all(_YEAR_TOKEN_RE.match(t) or t.lower() == "change" for t in tokens):
        # Lenient retry: keep only year/Change tokens if they span the line.
        kept = [t for t in tokens if _YEAR_TOKEN_RE.match(t) or t.lower() == "change"]
        if len(kept) < 2:
            return None
        tokens = kept
    month_days = _month_days_above(zone_lines, detail_idx)
    groups = _parse_grouping_durations(zone_lines, detail_idx)

    n_dates = sum(1 for t in tokens if _YEAR_TOKEN_RE.match(t))
    per_group = (n_dates // len(groups)) if groups and n_dates % len(groups) == 0 else None

    header = [""]
    columns: list[dict] = []
    change_slots: list[int] = []
    mi = 0
    date_seen = 0
    ok = True
    for tok in tokens:
        if _YEAR_TOKEN_RE.match(tok):
            if mi >= len(month_days):
                ok = False
                break
            md = month_days[mi]
            mi += 1
            m = re.match(r"(\w+)\s+(\d{1,2})", md)
            if not m:
                ok = False
                break
            month_name, day = m.group(1), m.group(2)
            dur_code, dur_label = (None, None)
            if groups and per_group:
                g = min(date_seen // per_group, len(groups) - 1)
                dur_code, dur_label = groups[g][0], groups[g][1]
            date_seen += 1
            datestr = f"{month_name} {day}, {tok}"
            label = f"{dur_label} {datestr}" if dur_label else datestr
            header.append(label)
            columns.append({
                "label": label, "date": _iso_date(month_name, day, tok),
                "year": int(tok), "month": _MONTH_TO_NUM.get(month_name.lower()),
                "day": int(day), "duration": dur_code,
                "duration_label": dur_label.lower() if dur_label else None,
                "is_change": False, "is_percent_col": False,
                "unit": None, "x_center": None,
            })
        elif tok.lower() == "change":
            dur_code, dur_label = (None, None)
            if groups and per_group:
                g = min(max(date_seen - 1, 0) // per_group, len(groups) - 1)
                dur_code, dur_label = groups[g][0], groups[g][1]
            short = {"3mo": "three months", "6mo": "six months",
                     "9mo": "nine months", "12mo": "twelve months"}.get(dur_code, dur_code or "")
            label = f"Change ({short})" if short else "Change"
            change_slots.append(len(header) - 1)
            header.append(label)
            columns.append({
                "label": label, "date": None, "year": None, "month": None,
                "day": None, "duration": dur_code,
                "duration_label": dur_label.lower() if dur_label else None,
                "is_change": True, "is_percent_col": True,
                "unit": None, "x_center": None,
            })
        else:
            ok = False
            break
    if not ok or len(header) - 1 != expected_cols:
        return None
    # Attach x-centers from detail-line words (for x-slotting repairs).
    if detail_word_centers:
        vals = [c for _, c in detail_word_centers]
        if len(vals) == len(columns):
            for col, x in zip(columns, vals):
                col["x_center"] = x
    # Guarantee unique labels.
    seen: dict[str, int] = {}
    for j in range(1, len(header)):
        if header[j] in seen:
            seen[header[j]] += 1
            header[j] = f"{header[j]} (col {j})"
            columns[j - 1]["label"] = header[j]
        else:
            seen[header[j]] = 1
    return header, columns, change_slots, "date"


# --------------------------------------------------------------------------
# FIX 4: category headers via x-position clustering
# --------------------------------------------------------------------------

def cluster_x_centers(centers: list[float], gap: float = X_GAP_THRESHOLD) -> list[float]:
    """1-D gap clustering: sorted centers split wherever the gap exceeds
    `gap`. Returns cluster means (column centers)."""
    if not centers:
        return []
    pts = sorted(centers)
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if p - clusters[-1][-1] <= gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [sum(c) / len(c) for c in clusters]


def word_lines_in_bbox(page, bbox: list | tuple) -> list[list[dict]]:
    """All word-lines fully inside a bbox, sorted top-to-bottom."""
    x0, top, x1, bottom = bbox
    words = [w for w in page.extract_words()
             if w["x0"] >= x0 - 1 and w["x1"] <= x1 + 2
             and w["top"] >= top - 2 and w["bottom"] <= bottom + 2]
    grouped: dict[int, list[dict]] = {}
    for w in words:
        grouped.setdefault(round(w["top"]), []).append(w)
    return [[w for w in sorted(grouped[k], key=lambda w: w["x0"])]
            for k in sorted(grouped)]


def table_value_column_centers(page, bbox, grid_rows: list[tuple[str, list[str]]]
                               ) -> list[float]:
    """Cluster x-positions of numeric value tokens across the whole table
    bbox into logical column centers. Sub-header/date words are excluded
    (bare 4-digit years never count as values)."""
    centers: list[float] = []
    for line_words in word_lines_in_bbox(page, bbox):
        text = " ".join(w["text"] for w in line_words)
        _, values, _ = split_label_values(text)
        if not values:
            continue
        # Map each value back to word(s): walk words, pairing $/%.
        toks = [(w["text"], (w["x0"] + w["x1"]) / 2) for w in line_words]
        # Find value-region start the same way split_label_values does.
        start = None
        for i, (t, _) in enumerate(toks):
            nxt = toks[i + 1][0] if i + 1 < len(toks) else None
            if _is_value_start(t, nxt):
                start = i
                break
            if nxt == "%" and re.fullmatch(r"\(?-?[\d,]+(\.\d+)?\)?", t):
                start = i
                break
        if start is None:
            continue
        i = start
        while i < len(toks):
            t, x = toks[i]
            b = toks[i + 1] if i + 1 < len(toks) else (None, x)
            if t == "$" and b[0] is not None:
                centers.append(b[1])
                i += 2
            elif b[0] == "%":
                centers.append(x)
                i += 2
            elif t == "%":
                i += 1
            else:
                if _YEAR_TOKEN_RE.match(t):
                    i += 1
                    continue
                centers.append(x)
                i += 1
    return cluster_x_centers(centers)


def header_text_lines(zone_lines: list[str]) -> tuple[list[str], str | None]:
    """Split trailing zone lines into (header_lines, asof_date_str).
    Header lines = lines after the last caption-ish line (ends with ':'
    or contains 'as follows'). A standalone full-date line among them is
    the table's as-of date, not a header."""
    cap_idx = None
    for idx, ln in enumerate(zone_lines):
        s = ln.strip()
        if not s:
            continue
        if s.endswith(":") or "as follows" in s.lower():
            cap_idx = idx
    # No caption-ish line (e.g. the second securities table, whose intro
    # sentence sits above the FIRST table): every remaining line is a
    # header candidate; value lines are still filtered below, and the
    # positioned-match + bucket guards reject anything implausible.
    cands = zone_lines[cap_idx + 1:] if cap_idx is not None else list(zone_lines)
    cands = [l.strip() for l in cands if l.strip()]
    asof = None
    headers: list[str] = []
    for ln in cands:
        m = _FULL_DATE_RE.search(ln)
        if m and len(ln.split()) <= 4:
            asof = f"{m.group(1)} {m.group(2)}, {m.group(3)}"
            continue
        # A header line never carries $/numeric values -- such a line is a
        # leaked data row the width check rejected, not a header.
        _, vals, _ = split_label_values(ln)
        if vals:
            continue
        headers.append(ln)
    return headers, asof


def _is_period_header_line(ln: str) -> bool:
    """Short zone lines that are part of a period header block (duration
    phrases, month/day lines, year/detail lines) -- not caption prose."""
    toks = ln.split()
    if len(toks) >= 10:
        return False
    low = ln.lower()
    if any(rx.search(ln) for rx, _, _ in _DURATION_RES):
        return True
    if _MONTH_DAY_RE.search(ln):
        return True
    if any(_YEAR_TOKEN_RE.match(t) or t.lower() == "change" for t in toks):
        return True
    return False


def caption_from_clean_lines(remaining: list[str], header_lines: list[str] | None) -> str:
    """Caption built AFTER leak/header removal: drops category header
    lines, period-header lines, and any value-bearing remnant, then takes
    the last few prose lines (section heading + intro sentence)."""
    hset = set(header_lines or [])
    prose = []
    for ln in remaining:
        s = ln.strip()
        if not s or s in hset or _is_period_header_line(s):
            continue
        _, vals, _ = split_label_values(s)
        if vals:
            continue
        prose.append(s)
    return " ".join(prose[-4:]) if prose else ""


def _positioned_header_words(zone_word_lines: list[list[dict]],
                             headers: list[str]) -> list[list[dict]] | None:
    """Match header text lines back to positioned word-lines (exact join
    match, in order)."""
    out: list[list[dict]] = []
    wrote = [False] * len(zone_word_lines)
    for h in headers:
        hit = None
        for i, wl in enumerate(zone_word_lines):
            if wrote[i]:
                continue
            if " ".join(w["text"] for w in wl) == h:
                hit = i
                break
        if hit is None:
            return None
        wrote[hit] = True
        out.append(zone_word_lines[hit])
    return out


def assign_category_columns(positioned: list[list[dict]], col_centers: list[float],
                            asof: str | None) -> list[dict] | None:
    buckets: list[list[dict]] = [[] for _ in col_centers]
    for wl in positioned:
        for w in wl:
            if re.fullmatch(r"\(\d+\)", w["text"]):  # footnote marker, not a header word
                continue
            xc = (w["x0"] + w["x1"]) / 2
            if xc < col_centers[0] - 90:
                # Label-column header word (e.g. "Periods" at x=19 while
                # the first value column sits at x~300): it would fail
                # the 90pt plausibility guard below anyway, so skip it
                # instead of failing the whole table. Words only slightly
                # left of the first column (wrapped headers starting left
                # of right-aligned numbers, e.g. "(in thousands)") still
                # get assigned to that column.
                continue
            j = min(range(len(col_centers)), key=lambda k: abs(col_centers[k] - xc))
            if abs(col_centers[j] - xc) > 90:  # not plausibly this table's header
                return None
            buckets[j].append(w)
    if any(not b for b in buckets):
        return None
    columns: list[dict] = []
    for b in buckets:
        b_sorted = sorted(b, key=lambda w: (round(w["top"]), w["x0"]))
        label = re.sub(r"\s+", " ", " ".join(w["text"] for w in b_sorted)).strip()
        unit = None
        mu = re.search(r"\(in\s+(millions|thousands|billions)\)", label, re.I)
        low = label.lower()
        if mu:
            unit = f"in {mu.group(1).lower()}"
        elif re.search(r"per\s+(share|rsu)\b", low):
            unit = "per share"
        elif "shares" in low and "dollar" not in low:
            unit = "shares in thousands"
        elif "dollar value" in low:
            unit = "in millions"
        columns.append({
            "label": f"{label} (as of {asof})" if asof else label,
            "date": None, "year": None, "month": None, "day": None,
            "duration": "point", "duration_label": f"as of {asof}" if asof else None,
            "is_change": False, "is_percent_col": False,
            "unit": unit, "x_center": None,
        })
        if asof:
            m = _FULL_DATE_RE.search(asof)
            if m:
                columns[-1].update({
                    "date": _iso_date(m.group(1), m.group(2), m.group(3)),
                    "year": int(m.group(3)),
                    "month": _MONTH_TO_NUM.get(m.group(1).lower()),
                    "day": int(m.group(2)),
                })
    for col, x in zip(columns, col_centers):
        col["x_center"] = x
    labels = [c["label"] for c in columns]
    if len(set(labels)) != len(labels):
        return None
    return columns


# --------------------------------------------------------------------------
# FIX 5: degenerate-row repair from word positions
# --------------------------------------------------------------------------

def repair_rows_from_words(page_words: list[dict], table_bbox, grid_merged: list[tuple[str, list[str], bool]],
                           expected_cols: int, col_centers: list[float | None],
                           row_bboxes: list[tuple] | None) -> list[tuple[str, list[str], bool]]:
    """Rebuild grid rows that are degenerate (empty label, or fewer values
    than the header demands) from the words in their y-band. Fails safe:
    a repair is only kept when it yields exactly `expected_cols` values
    with a non-empty label; otherwise the original grid row is kept."""
    if not row_bboxes or len(row_bboxes) != len(grid_merged):
        return grid_merged
    # Column slot edges from centers (midpoint boundaries).
    edges: list[float] | None = None
    if col_centers and all(x is not None for x in col_centers) and len(col_centers) == expected_cols:
        cs = list(col_centers)
        edges = [(cs[i] + cs[i + 1]) / 2 for i in range(len(cs) - 1)]
    out = list(grid_merged)
    x0, _, x1, _ = table_bbox
    for idx, ((label, values, saw), rbbox) in enumerate(zip(grid_merged, row_bboxes)):
        needs = (not label) or (0 < len(values) < expected_cols)
        if not needs or rbbox is None:
            continue
        _, ry0, _, ry1 = rbbox
        band = [w for w in page_words
                if w["x0"] >= x0 - 1 and w["x1"] <= x1 + 2
                and w["top"] >= ry0 - 1 and w["top"] <= ry1 + 1]
        if not band:
            continue
        band.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in band)
        rlabel, rvalues, rsaw = split_label_values(text)
        if not rlabel or not rvalues:
            continue
        if len(rvalues) == expected_cols:
            out[idx] = (rlabel, rvalues, rsaw or saw)
        elif edges is not None:
            slotted = slot_values_by_x(band, rlabel, edges, expected_cols)
            if slotted is not None:
                out[idx] = slotted
    return out


def recover_leaked_row_xslotted(zone_word_lines: list[list[dict]], last_line: str,
                                prev_line: str | None, edges: list[float],
                                expected_cols: int) -> tuple[str, list[str], bool] | None:
    """X-position fallback for a leaked row whose flat-text value count is
    short (e.g. RSU balance row: 2 values for 3 columns -- the blank is
    positional, only x tells us where). Slots the line's value tokens by
    x; the label comes from the line prefix (or the line above)."""
    target: list[dict] | None = None
    for wl in zone_word_lines:
        if " ".join(w["text"] for w in wl) == last_line:
            target = wl
            break
    if target is None:
        return None
    label, values, _ = split_label_values(last_line)
    if not values:
        return None
    if not label and prev_line:
        plabel, pvalues, _ = split_label_values(prev_line)
        if plabel and not pvalues:
            label = plabel
    if not label:
        return None
    slotted = slot_values_by_x(sorted(target, key=lambda w: w["x0"]), label,
                               edges, expected_cols)
    return slotted


def slot_values_by_x(band_words: list[dict], label: str, edges: list[float],
                     expected_cols: int) -> tuple[str, list[str], bool] | None:
    """Slot a row-band's value tokens into columns by x-position. Returns
    None unless every value lands in a distinct, in-range slot."""
    toks = [(w["text"], (w["x0"] + w["x1"]) / 2) for w in band_words]
    start = None
    for i, (t, _) in enumerate(toks):
        nxt = toks[i + 1][0] if i + 1 < len(toks) else None
        if _is_value_start(t, nxt):
            start = i
            break
        if nxt == "%" and re.fullmatch(r"\(?-?[\d,]+(\.\d+)?\)?", t):
            start = i
            break
    if start is None:
        return None
    # Pair $/% while tracking x of the numeric token.
    paired: list[tuple[str, float, bool]] = []  # (value, x, had_dollar)
    i = start
    while i < len(toks):
        t, x = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else (None, x)
        if t == "$" and nxt[0] is not None:
            paired.append((nxt[0], nxt[1], True))
            i += 2
        elif nxt[0] == "%":
            v = f"{t}%" if not t.endswith("%") else t
            paired.append((v, x, False))
            i += 2
        elif t == "%":
            i += 1
        else:
            if _YEAR_TOKEN_RE.match(t):
                i += 1
                continue
            paired.append((t, x, False))
            i += 1
    if not paired:
        return None
    slots: list[str] = [""] * expected_cols
    saw = False
    used: set[int] = set()
    for v, x, d in paired:
        j = 0
        while j < len(edges) and x > edges[j]:
            j += 1
        if j in used:
            return None
        used.add(j)
        m = _PCT_JOIN_RE.match(v)
        slots[j] = f"{m.group(1)}%" if m else v
        saw = saw or d
    return label, slots, saw


# --------------------------------------------------------------------------
# Row alignment + parents + units + junk filter
# --------------------------------------------------------------------------

def align_values(values: list[str], expected_cols: int,
                 change_slots: list[int]) -> list[str]:
    """Pad a short row to the header width. Two legitimate cases: blank
    Change cells on non-total rows (insert at the Change slots), and
    legitimately trailing blanks (RSU aggregate column). Anything else is
    padded trailingly -- never fabricates a number."""
    if len(values) == expected_cols:
        return values
    if (change_slots and len(values) == expected_cols - len(change_slots)
            and not any(v.endswith("%") for v in values)):
        out: list[str] = []
        it = iter(values)
        for j in range(expected_cols):
            out.append("" if j in change_slots else next(it))
        return out
    if len(values) < expected_cols:
        return values + [""] * (expected_cols - len(values))
    return values[:expected_cols]


def _is_parent_line(label: str, values: list[str]) -> bool:
    if not label.endswith(":"):
        return False
    return not any(v.strip() for v in values)


def apply_parents(merged: list[tuple[str, list[str], bool]],
                  initial_parent: str | None) -> list[tuple[str, str, list[str], bool, str]]:
    """FIX 6. Returns [(display_label, parent, values, saw_dollar,
    kind)]. Sub-header rows keep kind 'subheader'; blank rows 'blank'."""
    out = []
    parent = initial_parent or ""
    for label, values, saw in merged:
        if not label and not any(v.strip() for v in values):
            out.append(("", parent, values, saw, "blank"))
            continue
        if _is_parent_line(label, values):
            parent = label[:-1].strip()
            out.append((label, parent, values, saw, "subheader"))
            continue
        disp = f"{parent}: {label}" if parent and not label.startswith(parent) else label
        out.append((disp, parent, values, saw, "data"))
    return out


def extract_units(caption: str, zone_text: str) -> tuple[str, str]:
    blob = f"{caption}\n{zone_text}"
    m = _MIXED_UNITS_RE.search(blob)
    if m:
        # Mixed units ("net income in millions and shares in thousands"):
        # table-level units = the PRIMARY (first-mentioned) scale, the
        # full note is kept, and per-row overrides (see build_records)
        # correct the share/per-share rows individually.
        note = m.group(1).strip()
        ms = re.search(r"(millions|thousands|billions)", note, re.I)
        scale = ms.group(1).lower() if ms else ""
        prefix = "dollars " if "dollar" in note.lower() else ""
        return (f"{prefix}in {scale}" if scale else "", note)
    m = _UNITS_RE.search(blob)
    if m:
        full = m.group(0)
        scale = m.group(1).lower()
        prefix = "dollars " if "dollar" in full.lower() else ""
        return f"{prefix}in {scale}", full
    return "", ""


def is_junk_table(caption: str, zone_text: str,
                  merged: list[tuple[str, list[str], bool]]) -> bool:
    if _JUNK_CAPTION_RE.search(caption) or "TABLE OF CONTENTS" in zone_text.upper().split("\n")[0:3].__str__():
        return True
    if "TABLE OF CONTENTS" in f"{caption}\n{zone_text}":
        # Only junk when the table really is the TOC (no financial values).
        has_fin = any(
            _is_rich_number(v) or v.strip().startswith("$")
            for _, vals, _ in merged for v in vals
        )
        if not has_fin:
            return True
    # Content test: no financial token anywhere -> not a financial table.
    for _, vals, saw in merged:
        if saw:
            return False
        for v in vals:
            if _is_rich_number(v) or v.endswith("%"):
                return False
    return True


# --------------------------------------------------------------------------
# Records (structured facts)
# --------------------------------------------------------------------------

def norm_row_label(label: str) -> str:
    s = label.lower().replace("®", "")
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"'s\b", "", s)
    s = _FOOTNOTE_SUFFIX_RE.sub("", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ROW_UNIT_SHARES_RE = re.compile(r"shares used|weighted-average|dilutive securities", re.I)
_ROW_UNIT_PERSHARE_RE = re.compile(r"per share\b", re.I)


def build_records(page_number: int, table_idx: int, caption: str, units: str,
                  currency: str, table_kind: str, columns: list[dict],
                  rows: list[tuple[str, str, list[str], bool, str]]) -> list[dict]:
    recs: list[dict] = []
    for disp, parent, values, saw, kind in rows:
        if kind != "data":
            continue
        label_core = disp.split(": ", 1)[-1] if ": " in disp and parent and disp.startswith(parent) else disp
        # Row-level unit overrides for mixed-unit tables (EPS statements):
        # share-COUNT rows are in thousands, per-share rows are dollars
        # each -- even though the table's primary scale is millions.
        # Shares-patterns take precedence ("Shares used in computing
        # earnings per share: Basic" counts shares, despite containing
        # the words "per share" in its parent part).
        row_unit = None
        if _ROW_UNIT_SHARES_RE.search(disp):
            row_unit = "shares in thousands"
        elif _ROW_UNIT_PERSHARE_RE.search(disp):
            row_unit = "per share"
        for j, raw in enumerate(values):
            if j >= len(columns):
                break
            v, is_pct = parse_value_number(raw)
            if v is None:
                continue
            col = columns[j]
            unit = col.get("unit") or row_unit or units
            recs.append({
                "page": page_number, "table_index": table_idx,
                "caption": caption, "units": unit, "currency": currency,
                "table_kind": table_kind,
                "row": disp, "row_core": label_core,
                "row_norm": norm_row_label(disp),
                "row_core_norm": norm_row_label(label_core),
                "parent": parent, "parent_norm": norm_row_label(parent),
                "col": col["label"], "col_index": j,
                "duration": col.get("duration"),
                "duration_label": col.get("duration_label"),
                "date": col.get("date"), "year": col.get("year"),
                "month": col.get("month"), "day": col.get("day"),
                "is_change": col.get("is_change", False),
                "is_percent": bool(is_pct or col.get("is_percent_col")),
                "value_raw": raw, "value": v,
            })
    return recs


# --------------------------------------------------------------------------
# Main per-table pipeline
# --------------------------------------------------------------------------

def _generic_columns(n: int) -> list[dict]:
    return [{
        "label": f"Col {i + 1}", "date": None, "year": None,
        "month": None, "day": None, "duration": None,
        "duration_label": None, "is_change": False,
        "is_percent_col": False, "unit": None, "x_center": None,
    } for i in range(n)]


def parse_one_table(page, table, table_idx: int, page_number: int, zone_text: str,
                    zone_word_lines: list[list[dict]], footer_label: str | None,
                    prev_columns: list[dict] | None = None,
                    prev_confidence: str | None = None,
                    prev_units: str = "",
                    prev_units_note: str = "") -> ParsedTable | None:
    raw = table.extract() or []
    grid, kept_idx = clean_table_grid(raw)
    # Merge grid rows.
    merged: list[tuple[str, list[str], bool]] = []
    for r in grid:
        label, values, saw = merge_row_values(r)
        merged.append((label, values, saw))

    # Rough width estimate: max over grid rows.
    rough_n = max([len(v) for _, v, _ in merged] or [0])

    zone_lines = [l for l in zone_text.splitlines() if l.strip()]
    page_words = page.extract_words()

    # ---- Pass 1: rough leak recovery (exact-width matches only; no
    # Change-slot info yet) so leaks don't pollute width estimation.
    leaked1, rem1 = recover_leaked_rows(zone_lines, rough_n if rough_n else 4)

    # ---- Header attempts across candidate widths (header parse itself
    # validates the width against the detail line).
    det_idx = _find_detail_index(rem1)
    det_centers = None
    if det_idx is not None:
        for wl in zone_word_lines:
            if " ".join(w["text"] for w in wl) == rem1[det_idx]:
                det_centers = [(w["text"], (w["x0"] + w["x1"]) / 2) for w in wl]
                break
    cands = dict.fromkeys([rough_n, len(leaked1[0][1]) if leaked1 else 0])
    period_attempt = None
    for cand in cands:
        if cand <= 0:
            continue
        period_attempt = build_period_header(rem1, cand, det_centers)
        if period_attempt is not None:
            break

    header_row: list[str]
    columns: list[dict]
    change_slots: list[int] = []
    confidence = "generic"
    expected = rough_n
    leaked: list[tuple[str, list[str], bool]] = leaked1
    remaining = rem1
    headers_txt: list[str] | None = None
    headers_inherited = False

    if period_attempt is not None:
        header_row, columns, change_slots, confidence = period_attempt
        expected = len(header_row) - 1
        # ---- Pass 2: re-run leak recovery on the ORIGINAL zone against
        # the TRUE width + Change slots (pass 1 used rough width and no
        # slot info, so Change-table leaks like OI&E's first row -- 4
        # values for 6 columns -- were missed). Then confirm the header
        # still builds on the new remainder; else keep pass-1 state.
        rel, rem2 = recover_leaked_rows(zone_lines, expected, change_slots)
        confirm = build_period_header(rem2, expected, det_centers)
        if confirm is not None:
            header_row, columns, change_slots, confidence = confirm
            leaked, remaining = rel, rem2
    else:
        # ---- Category / single-column / generic path.
        try:
            centers_all = table_value_column_centers(page, table.bbox, [])
        except Exception:
            centers_all = []
        # The x-cluster count is the TRUE logical width (every cluster is
        # a column holding >=1 value); the grid max can only undercount
        # when every row has a blank somewhere (repurchase table: 3 vs 4).
        if centers_all and len(centers_all) <= max(rough_n + 2, 2) and len(centers_all) > rough_n:
            expected = len(centers_all)
        elif leaked1:
            expected = max([len(v) for _, v, _ in leaked1] + [rough_n])
        else:
            expected = rough_n
        if expected <= 0:
            return None
        headers_txt, asof = header_text_lines(rem1)
        asof_match = _FULL_DATE_RE.search(" ".join(rem1[-3:])) if rem1 else None
        if expected == 1 and (asof or asof_match):
            datestr = asof or (f"{asof_match.group(1)} {asof_match.group(2)}, "
                               f"{asof_match.group(3)}")
            m = _FULL_DATE_RE.search(datestr)
            label = f"As of {datestr}"
            header_row = ["", label]
            columns = [{
                "label": label, "date": _iso_date(m.group(1), m.group(2), m.group(3)),
                "year": int(m.group(3)), "month": _MONTH_TO_NUM.get(m.group(1).lower()),
                "day": int(m.group(2)), "duration": "point",
                "duration_label": f"as of {datestr}", "is_change": False,
                "is_percent_col": False, "unit": None, "x_center": None,
            }]
            confidence = "single"
        else:
            positioned = (_positioned_header_words(zone_word_lines, headers_txt)
                          if headers_txt else None)
            cat_cols = None
            if positioned is not None and len(centers_all) == expected:
                cat_cols = assign_category_columns(positioned, centers_all, asof)
            if cat_cols is not None:
                columns = cat_cols
                header_row = [""] + [c["label"] for c in columns]
                confidence = "category"
            else:
                # Header inheritance for split tables: the gross-margin-%
                # table's zone holds no periods at all, but the table
                # directly above it on the same page has the identical
                # 4 date columns -- inherit them instead of Col 1..N.
                # ONLY from a true period ("date") table: category tables
                # carry their own as-of date and must never be inherited
                # (this once mislabeled the September securities table
                # with June headers -- caught by testing, hence this).
                if (prev_confidence == "date" and prev_columns
                        and len(prev_columns) == expected
                        and all(c.get("date") or c.get("duration") for c in prev_columns)):
                    columns = [dict(c) for c in prev_columns]
                    header_row = [""] + [c["label"] for c in columns]
                    confidence = "date"
                    headers_inherited = True
                    change_slots = [i for i, c in enumerate(columns) if c.get("is_change")]
                else:
                    header_row = [""] + [f"Col {i + 1}" for i in range(expected)]
                    columns = _generic_columns(expected)
                    confidence = "generic"
        # Pass 2 for category tables: x-slotted recovery catches short
        # leaks (RSU balance row: 2 values, 3 columns).
        if confidence in ("category", "single", "generic", "date"):
            xc = [c.get("x_center") for c in columns]
            if all(x is not None for x in xc) and len(xc) == expected and expected > 1:
                edges = [(xc[i] + xc[i + 1]) / 2 for i in range(len(xc) - 1)]
                if remaining:
                    last = remaining[-1].strip()
                    prev = remaining[-2].strip() if len(remaining) >= 2 else None
                    xs = recover_leaked_row_xslotted(zone_word_lines, last, prev,
                                                     edges, expected)
                    # Only use it when pass 1 found nothing there.
                    if xs is not None and (not leaked or True):
                        # Avoid double-recovery: pass 1 consumes lines, so
                        # if pass 1 already took this line, `remaining`
                        # wouldn't still end with it... unless pass 1
                        # rejected it for width. Check the text differs
                        # from any pass-1 recovered row's source.
                        pass1_vals = [tuple(v) for _, v, _ in leaked]
                        if tuple(xs[1]) not in pass1_vals:
                            leaked = leaked + [xs]
                            # Consume the consumed lines.
                            if prev and xs[0] == split_label_values(prev)[0] \
                                    and not split_label_values(prev)[1]:
                                remaining = remaining[:-2]
                            else:
                                remaining = remaining[:-1]

    # ---- FIX 5: repair degenerate grid rows from word positions.
    try:
        all_bboxes = [r.bbox for r in table.rows]
        row_bboxes = [all_bboxes[i] for i in kept_idx if i < len(all_bboxes)]
        if len(row_bboxes) != len(merged):
            row_bboxes = None
    except Exception:
        row_bboxes = None
    centers_for_repair = [c.get("x_center") for c in columns] if columns else []
    if not any(c is not None for c in centers_for_repair):
        try:
            vc = table_value_column_centers(page, table.bbox, [])
            centers_for_repair = vc if len(vc) == expected else []
        except Exception:
            centers_for_repair = []
    if row_bboxes is not None and centers_for_repair and len(centers_for_repair) == expected:
        try:
            merged = repair_rows_from_words(page_words, table.bbox, merged, expected,
                                            centers_for_repair, row_bboxes)
        except Exception:
            pass

    # Prepend leaked rows + align everything to the header width.
    all_rows = list(leaked) + merged
    aligned = [(l, align_values(v, expected, change_slots), s) for l, v, s in all_rows]

    caption = caption_from_clean_lines(remaining, headers_txt) or caption_from_zone(zone_text)
    if is_junk_table(caption, zone_text, aligned):
        return None

    # FIX 6: parents (zone-trailing 'Xxx:' line seeds the first parent).
    initial_parent = None
    if remaining:
        last = remaining[-1].strip()
        lab, vals, _ = split_label_values(last)
        if last.endswith(":") and not vals and lab:
            initial_parent = lab[:-1].strip() if lab.endswith(":") else lab
    qualified = apply_parents(aligned, initial_parent)

    units, units_note = extract_units(caption, zone_text)
    if headers_inherited and not units and prev_units:
        units, units_note = prev_units, prev_units_note
    currency = "$" if any(s for _, _, _, s, _ in qualified) else ""

    # Display grid.
    disp_rows = [header_row]
    row_kinds = ["header"]
    for disp, _parent, values, _saw, kind in qualified:
        disp_rows.append([disp] + list(values))
        row_kinds.append(kind)

    if confidence in ("date",):
        table_kind = "financial_period"
    elif confidence in ("category", "single"):
        table_kind = "category" if confidence == "category" else "single_point"
    else:
        # Generic but with date columns? No -- generic means unknown.
        table_kind = "other"

    records = build_records(page_number, table_idx, caption, units, currency,
                            table_kind, columns, qualified)

    # Header annotation for generic grids (old behavior, kept): the real
    # column text stays available as prose.
    header_ctx = zone_text
    return ParsedTable(
        page_number=page_number, table_index_on_page=table_idx,
        bbox=tuple(table.bbox), caption=caption, rows=disp_rows,
        filing_page_label=footer_label, header_context=header_ctx,
        header_reconstructed=True, header_confidence=confidence,
        units=units, units_note=units_note, currency=currency,
        table_kind=table_kind, columns=columns, records=records,
        row_kinds=row_kinds,
    )


# --------------------------------------------------------------------------
# Document-level parse
# --------------------------------------------------------------------------

def _zone_word_lines(page, sorted_tables) -> list[list[list[dict]]]:
    """Positioned word-lines per table caption zone (for header matching)."""
    words = page.extract_words()
    out = []
    prev_bottom = 0.0
    for t in sorted_tables:
        top = t.bbox[1]
        zw = [w for w in words if prev_bottom <= w["top"] < top]
        grouped: dict[int, list[dict]] = {}
        for w in zw:
            grouped.setdefault(round(w["top"]), []).append(w)
        out.append([[w for w in sorted(grouped[k], key=lambda w: w["x0"])]
                    for k in sorted(grouped)])
        prev_bottom = t.bbox[3]
    return out


def parse_pdf(path: str) -> ParsedDocument:
    doc = ParsedDocument(source_path=path, num_pages=0)
    with pdfplumber.open(path) as pdf:
        doc.num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            found_tables = page.find_tables(table_settings={
                "vertical_strategy": "lines", "horizontal_strategy": "lines"})
            sorted_tables = sorted(found_tables, key=lambda t: t.bbox[1])
            text_only_page = page
            for t in sorted_tables:
                text_only_page = text_only_page.outside_bbox(t.bbox)
            raw_outside_text = text_only_page.extract_text() or ""
            narrative_text = clean_stray_table_lines(raw_outside_text)
            footer_label = extract_footer_label(page.extract_text() or "")
            caption_zones = build_caption_zones(page, sorted_tables)
            zone_words = _zone_word_lines(page, sorted_tables)
            parsed_tables = []
            prev_columns: list[dict] | None = None
            prev_conf: str | None = None
            prev_u, prev_un = "", ""
            for idx, (t, zone_text, zwl) in enumerate(
                    zip(sorted_tables, caption_zones, zone_words)):
                try:
                    pt = parse_one_table(page, t, idx, i, zone_text, zwl,
                                         footer_label, prev_columns, prev_conf,
                                         prev_u, prev_un)
                except Exception:
                    pt = None
                if pt is not None:
                    parsed_tables.append(pt)
                    prev_columns = pt.columns
                    prev_conf = pt.header_confidence
                    prev_u, prev_un = pt.units, pt.units_note
            doc.pages.append(ParsedPage(page_number=i, narrative_text=narrative_text,
                                        tables=parsed_tables))
    return doc


def save_parsed_document(doc: ParsedDocument, out_path: str) -> None:
    def _default(o):
        if isinstance(o, tuple):
            return list(o)
        try:
            return asdict(o)
        except Exception:
            return str(o)

    payload = {
        "source_path": doc.source_path,
        "num_pages": doc.num_pages,
        "pages": [{
            "page_number": p.page_number,
            "narrative_text": p.narrative_text,
            "tables": [asdict(t) for t in p.tables],
        } for p in doc.pages],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)


def save_facts(doc: ParsedDocument, facts_path: str) -> int:
    Path(facts_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(facts_path, "w", encoding="utf-8") as f:
        for p in doc.pages:
            for t in p.tables:
                for r in t.records:
                    f.write(json.dumps(r) + "\n")
                    n += 1
    return n


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/10q.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/parsed_document.json"
    facts = sys.argv[3] if len(sys.argv) > 3 else str(Path(out).parent / "table_facts.jsonl")
    parsed = parse_pdf(src)
    save_parsed_document(parsed, out)
    n_facts = save_facts(parsed, facts)
    n_tables = sum(len(p.tables) for p in parsed.pages)
    from collections import Counter
    conf = Counter(t.header_confidence for p in parsed.pages for t in p.tables)
    print(f"Parsed {parsed.num_pages} pages, extracted {n_tables} tables -> {out}")
    print(f"  header confidence: {dict(conf)}")
    print(f"  structured facts: {n_facts} -> {facts}")
