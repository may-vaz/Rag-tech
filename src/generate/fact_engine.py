"""
fact_engine.py -- deterministic arithmetic over structured table facts
======================================================================
ARCHITECTURE: why this file exists
----------------------------------
The old computation path asked a 4B LLM to READ the right numbers out of
markdown tables (with duplicate "June 25, 2022" headers for BOTH the
3-month and 9-month columns, misaligned Change columns, and a silently
dropped iPhone row) and then computed in Python. The Python half was
right (PAL -- see llm_client.py); the LLM-READING half was the fragile
one: a small model guessing which of four identical-looking columns a
number lives in, through a JSON schema that itself trips grammar bugs.

This file replaces the READING half with deterministic lookup. The
parse layer already knows every cell's (row label, period, value, units)
-- see table_facts.jsonl -- so for a computation-shaped question we:

  1. parse the question into (operation, operands) with general
     patterns (no hardcoded companies, metrics, or numbers);
  2. match each operand to exactly one fact (normalized row label +
     normalized period, both must match);
  3. check the operands are combinable (same units/scale, no mixing of
     dollars with percents or share counts);
  4. compute in plain Python and format with units + page citations.

STATED-FIRST RULE (your "don't compute what's already there" requirement):
  - percent_change: if the filing STATES the change (a Change column for
    that metric+duration), quote it -- don't compute a conflicting
    rounded-vs-precise figure (filed "2%" vs computed "1.87%").
  - ratio ("X as a % of Y"): if a percentage row for X already exists for
    that period ("Total gross margin percentage"), quote it.
  - sum/difference: filed tables never state arbitrary combos, so always
    compute (precisely: 82,959 - 81,434 = 1,525, not the MD&A's rounded
    "$1.5 billion").

PRECISION-FIRST FALLBACKS (why this can't regress anything):
  - Every step refuses to guess: ambiguous row match, ambiguous period,
    mixed units, missing operand -> return None.
  - None means "fall through to the existing pipeline" (LLM-extraction
    computation, then lookup) -- see answer.py. Lookup questions never
    reach this file at all (intent-gated, same as before).
  - All matching is case/format-insensitive and company-agnostic.

STDLIB ONLY (json, re). Safe to import from answer.py and llm_client.py
without pulling in torch/faiss/requests.

Public API:
  detect_intent(question) -> 'sum' | 'difference' | 'percent_change' | 'ratio' | None
  load_facts(path) -> list[dict]
  try_answer(question, facts) -> {"answer": str, "sources": [...]} | None
"""

from __future__ import annotations

import json
import re


# --------------------------------------------------------------------------
# Intent detection (superset of llm_client's patterns; ratio + phrasing
# variants added. Conservative: each needs a specific multi-word shape.)
# --------------------------------------------------------------------------

_SUM_RE = re.compile(
    r"\b(combined|sum of|total of)\b.*\band\b"
    r"|\band\b.*\b(combined|together)\b"
    r"|\bplus\b"
    r"|\b(aggregate|add|adding)\b.*\band\b",
    re.I,
)
_DIFFERENCE_RE = re.compile(
    r"\b(difference between|difference in)\b"
    r"|\bchange\b.{0,40}\bfrom\b.{0,80}\bto\b"
    r"|\bhow much did\b.{0,80}\b(increase|decrease|grow|decline|rise|fall)\b"
    r"|\b(increase|decrease)\b.{0,40}\bfrom\b.{0,80}\bto\b"
    r"|\bless\b|\bminus\b",
    re.I,
)
_PERCENT_CHANGE_RE = re.compile(
    r"\b(percent(age)?\s+change|%\s*change|growth rate|year[- ]over[- ]year|yoy)\b"
    r"|\bpercent(age)?\b.{0,30}\b(increase|decrease|growth|change)\b"
    r"|\b(increase|decrease|growth)\b.{0,30}\bpercent(age)?\b",
    re.I,
)
_RATIO_RE = re.compile(
    r"\bwhat\s+(percent(age)?|share|proportion|fraction|portion)\s+of\b"
    r"|\bas\s+a\s+(percent(age)?|share|proportion|portion)\s+of\b"
    r"|\bratio\s+of\b",
    re.I,
)


def detect_intent(question: str) -> str | None:
    """Classify a computation question. Order matters: ratio and percent
    phrasing beat bare difference phrasing ('what percent of X is Y'
    contains neither 'difference' nor 'combined', but check anyway)."""
    if _RATIO_RE.search(question):
        return "ratio"
    if _PERCENT_CHANGE_RE.search(question):
        return "percent_change"
    if _DIFFERENCE_RE.search(question):
        return "difference"
    if _SUM_RE.search(question):
        return "sum"
    return None


# --------------------------------------------------------------------------
# Facts loading
# --------------------------------------------------------------------------

def load_facts(path: str) -> list[dict]:
    facts: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


# --------------------------------------------------------------------------
# Text normalization (mirrors parse-side norm_row_label)
# --------------------------------------------------------------------------

_COMPANY_WORDS = {"apple", "apples", "company", "companies", "firm"}
_STRUCT_WORDS = {
    "what", "was", "were", "is", "are", "be", "been", "the", "a", "an",
    "of", "in", "for", "to", "and", "or", "by", "s", "that", "which",
    # NOTE: "total" is NOT structural -- "total assets" must keep its
    # anchor (row_score's total-preference tiers assume it survives).
    "combined", "sum", "difference", "between", "change", "from",
    "compared", "compare", "versus", "vs", "less", "minus", "plus",
    "percent", "percentage", "rate", "growth", "increase", "decrease",
    "quarter", "quarters", "months", "month", "ended", "fiscal", "year",
    "first", "second", "third", "fourth", "how", "much", "did", "does",
    "over", "during", "each", "per",
}


def norm_text(s: str) -> str:
    s = s.lower().replace("®", "").replace("’", "'").replace("‘", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"\(\d+\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _stem(t: str) -> str:
    # Light plural stemmer, applied IDENTICALLY to question and fact
    # tokens, so it only equates inflections ("margin"/"margins",
    # "service"/"services"), never distinct words.
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def conjunct_tokens(text: str) -> list[str]:
    toks = [_stem(t) for t in norm_text(text).split()
            if t and t not in _COMPANY_WORDS and t not in _STRUCT_WORDS
            and not t.isdigit()]
    return toks


def norm_unit(u: str) -> str:
    s = (u or "").lower()
    if "per share" in s:
        return "per share"
    if "shares" in s:
        return "shares"
    m = re.search(r"(millions|thousands|billions)", s)
    return m.group(1) if m else (s.strip() or "unspecified")


# --------------------------------------------------------------------------
# Period understanding
# --------------------------------------------------------------------------

_MONTHS = ("january|february|march|april|may|june|july|august|september|"
           "october|november|december")
_MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_DUR_WORD = {"three": "3mo", "six": "6mo", "nine": "9mo", "twelve": "12mo"}
_Q_WORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}

_PERIOD_RES = [
    # (three|nine) months ended June 25, 2022
    (re.compile(rf"\b(three|six|nine|twelve)\s+months?\s+ended\s+({_MONTHS})\s+(\d{{1,2}}),?\s+((?:19|20)\d{{2}})", re.I),
     "duration_date"),
    # quarter ended June 25, 2022
    (re.compile(rf"\bquarter\s+ended\s+({_MONTHS})\s+(\d{{1,2}}),?\s+((?:19|20)\d{{2}})", re.I),
     "quarter_date"),
    # as of June 25, 2022
    (re.compile(rf"\bas\s+of\s+({_MONTHS})\s+(\d{{1,2}}),?\s+((?:19|20)\d{{2}})", re.I),
     "asof_date"),
    # (third) quarter of (fiscal) 2022 / Q3 2022 / Q3'22
    (re.compile(r"\b(first|second|third|fourth)\s+quarter\s+of\s+(?:fiscal\s+)?((?:19|20)\d{2})", re.I),
     "quarter_year"),
    (re.compile(r"\bq([1-4])\s*'?((?:19|20)?\d{2})\b", re.I), "q_year"),
    (re.compile(r"\bfiscal\s+(first|second|third|fourth)\s+quarter\b.{0,20}?((?:19|20)\d{2})", re.I),
     "quarter_year"),
    # bare June 25, 2022
    (re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),?\s+((?:19|20)\d{{2}})\b", re.I),
     "bare_date"),
]


def extract_periods(question: str) -> list[dict]:
    """All period mentions, longest-match-first, non-overlapping, in
    question order. Each: {start, end, text, duration, year, month, day}."""
    cands: list[dict] = []
    for rx, kind in _PERIOD_RES:
        for m in rx.finditer(question):
            g = m.groups()
            try:
                if kind == "duration_date":
                    spec = {"duration": _DUR_WORD[g[0].lower()], "month": _MONTH_NUM[g[1].lower()],
                            "day": int(g[2]), "year": int(g[3])}
                elif kind == "quarter_date":
                    spec = {"duration": "3mo", "month": _MONTH_NUM[g[0].lower()],
                            "day": int(g[1]), "year": int(g[2])}
                elif kind == "asof_date":
                    spec = {"duration": "point", "month": _MONTH_NUM[g[0].lower()],
                            "day": int(g[1]), "year": int(g[2])}
                elif kind == "quarter_year":
                    spec = {"duration": "3mo", "month": None, "day": None, "year": int(g[1])}
                elif kind == "q_year":
                    yr = int(g[1])
                    spec = {"duration": "3mo", "month": None, "day": None,
                            "year": 2000 + yr if yr < 100 else yr}
                elif kind == "bare_date":
                    spec = {"duration": None, "month": _MONTH_NUM[g[0].lower()],
                            "day": int(g[1]), "year": int(g[2])}
                else:
                    continue
            except (KeyError, ValueError, IndexError):
                continue
            spec.update({"start": m.start(), "end": m.end(), "text": m.group(0)})
            cands.append(spec)
    # Longest-first, drop overlaps, restore question order.
    cands.sort(key=lambda s: (-(s["end"] - s["start"]), s["start"]))
    kept: list[dict] = []
    for c in cands:
        if any(not (c["end"] <= k["start"] or c["start"] >= k["end"]) for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda s: s["start"])
    return kept


def blank_periods(question: str, periods: list[dict]) -> str:
    chars = list(question)
    for p in periods:
        for i in range(p["start"], p["end"]):
            chars[i] = " "
    return "".join(chars)


def fact_matches_period(fact: dict, spec: dict) -> bool:
    dur = fact.get("duration") or "point"  # bare-date cols are point-in-time
    want = spec.get("duration") or "point"
    # A bare-date spec (no duration words at all) matches any duration
    # with the same date -- but that is usually AMBIGUOUS (3mo and 9mo
    # share dates), and ambiguity is rejected at match time. Here we
    # only enforce: explicit duration must equal.
    if spec.get("duration") and dur != spec["duration"]:
        return False
    if spec.get("duration") == "point" and dur not in ("point",):
        return False
    if spec.get("year") and fact.get("year") != spec["year"]:
        return False
    if spec.get("month") and fact.get("month") != spec["month"]:
        return False
    if spec.get("day") and fact.get("day") != spec["day"]:
        return False
    return True


# --------------------------------------------------------------------------
# Operand parsing (per operation)
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"\s+and\s+|,|\+|\s+plus\s+|&|;|\s+versus\s+|\s+vs\.?\s+", re.I)


_CLEAN_WORDS = ("what|was|were|is|are|be|been|the|a|an|of|in|for|to|and|or|by|"
                "apple'?s?|company'?s?|firm'?s?|how|much|did|does|do|over|during|"
                "combined|sum|difference|between|change|changed|from|compared|compare|"
                "versus|less|minus|plus|together|aggregate|percent|percentage|rate|"
                "growth|grew|increase|increased|increases|decrease|decreased|decreases|"
                "rise|rose|fall|fell|decline|declined|quarter|quarters|month|months|"
                "ended|fiscal|year|first|second|third|fourth|each|per")
# NOTE: "total" is deliberately NOT stripped -- "total net sales" is a row label.


def _clean_metric(text: str) -> str:
    t = re.sub(r"[?.!,;:()\[\]]", " ", text)
    t = re.sub(rf"\b({_CLEAN_WORDS})\b", " ", t, flags=re.I)
    t = re.sub(r"\bq[1-4]\b", " ", t, flags=re.I)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Metric nouns a conjunct may carry alongside its entity ("Mac net sales"
# means the Mac row -- "net"/"sales" add no entity information). Used by
# row_score's lenient tier, never alone (a conjunct reduced to nothing
# falls back to the strict tier, so bare "net sales" still prefers the
# Total row instead of matching everything).
_METRIC_WORDS = {"net", "sales", "sale", "revenue", "revenues", "income",
                 "incomes", "margin", "margins", "expense", "expenses",
                 "cost", "costs", "profit", "profits", "loss", "losses"}


def parse_sum(question: str, periods: list[dict]) -> list[dict]:
    """Candidate sum parses (comma-blind, then comma-split). The solver
    validates each against the facts; the first valid one wins."""
    # Blank periods on the FULL question (spans stay valid), then re-slice.
    full_blank = blank_periods(question, periods)
    m = re.search(r"\b(combined|sum of|total of)\b(.+)", full_blank, re.I)
    if m:
        body = m.group(2)
    else:
        # Trailing form ("iPhone and Mac combined ...") or "X plus Y".
        m2 = re.search(r"(.+?)\b(combined|together)\b", full_blank, re.I)
        if m2:
            body = m2.group(1)
        elif re.search(r"\bplus\b", full_blank, re.I):
            body = full_blank
        else:
            return []
    # Period: exactly one distinct period required for a sum.
    keys = {(p.get("duration"), p.get("year"), p.get("month"), p.get("day")) for p in periods}
    if len(keys) != 1:
        return []
    # Candidate segmentations. Entities may contain commas AND the word
    # "and" ("Wearables, Home and Accessories"), so "and"-occurrences are
    # tried in subsets -- fewer splits first (entities-intact prior) --
    # with plus/&/; as mandatory splits (never entity-internal) and the
    # full comma-split last (for Oxford lists like "iPhone, Mac and
    # iPad"). The SOLVER validates each against the facts (every conjunct
    # must match a DISTINCT cell) and the first valid one wins.
    ands = list(re.finditer(r"\s+and\s+", body, flags=re.I))
    segs: list[list[str]] = []
    if not ands:
        segs.append(re.split(r"\+|\s+plus\s+|&|;", body, flags=re.I))
    else:
        idx = list(range(len(ands)))
        if len(idx) <= 3:
            from itertools import combinations
            subsets = [s for r in (1, 2, 3) for s in combinations(idx, r)]
        else:  # many "and"s: each single split + the full split
            subsets = [(i,) for i in idx] + [tuple(idx)]
        for sub in sorted(subsets, key=lambda s: (len(s), s)):
            # Segment k runs from the end of cut k-1's match to the
            # start of cut k's match.
            bounds = [0]
            for i in sub:
                bounds.append(ands[i].start())
                bounds.append(ands[i].end())
            bounds.append(len(body))
            pieces = [body[bounds[j]:bounds[j + 1]]
                      for j in range(0, len(bounds) - 1, 2)]
            flat: list[str] = []
            for pc in pieces:
                flat.extend(re.split(r"\+|\s+plus\s+|&|;", pc, flags=re.I))
            segs.append(flat)
    comma_seg = _SPLIT_RE.split(body)
    segs.append(comma_seg)
    # Merge variants of the comma-split: an entity spanning a comma
    # ("..., Wearables, Home and Accessories, ...") needs its pieces
    # re-joined. Try leaving each single comma unsplit, then each pair
    # (caps keep this small: <= 6 pieces -> <= 21 variants).
    if 3 <= len(comma_seg) <= 8:
        from itertools import combinations
        gaps = list(range(len(comma_seg) - 1))
        for r in (1, 2):
            for merge in combinations(gaps, r):
                var = [comma_seg[0]]
                for j, pc in enumerate(comma_seg[1:], start=1):
                    if j - 1 in merge:
                        var[-1] = var[-1] + ", " + pc
                    else:
                        var.append(pc)
                segs.append(var)
    out: list[dict] = []
    for seg in segs:
        parts = [p.strip(" ?.,") for p in seg if p.strip(" ?.,")]
        if len(parts) < 2:
            continue
        conjuncts: list[str] = []
        pct: list[bool] = []
        metric_hint: list[str] = []
        bad = False
        for part in parts:
            # "net sales of iPhone" -> entity "iPhone", hint "net sales".
            # (Split on "of" BEFORE cleaning, since cleaning strips "of".)
            m_of = re.split(r"\bof\b", part, flags=re.I)
            if len(m_of) >= 2:
                left, right = " ".join(m_of[:-1]), m_of[-1]
                # Only treat as metric-of-entity when the RIGHT side is
                # short (an entity, not a restated clause).
                if right.strip() and len(right.split()) <= 5 and left.strip():
                    metric_hint.extend(conjunct_tokens(left))
                    ent = _clean_metric(right)
                    if not ent:
                        bad = True
                        break
                    conjuncts.append(ent)
                    pct.append(_has_pct_word(left + " " + right))
                    continue
            ent = _clean_metric(part)
            if not ent:
                bad = True
                break
            conjuncts.append(ent)
            # Percent-kind flag from the RAW part: cleaning strips the
            # word "percentage", but the solver needs to know the user
            # asked for a percentage (it must match a % row or refuse).
            pct.append(_has_pct_word(part))
        if bad or len(conjuncts) < 2:
            continue
        if not metric_hint:
            metric_hint = conjunct_tokens(body)
        for cc, pp in _ellipsis_variants(conjuncts, pct):
            cand = {"op": "sum", "conjuncts": cc, "period": periods[0],
                    "hint": metric_hint, "pct": pp}
            if cand not in out:
                out.append(cand)
    return out


def _has_pct_word(text: str) -> bool:
    return bool(re.search(r"percent", text, re.I))


def _ellipsis_variants(conjuncts: list[str], pct: list[bool]
                       ) -> list[tuple[list[str], list[bool]]]:
    """Original + ellipsis-completed variant. In "basic and diluted
    earnings per share" the short side inherits the shared tail ("basic
    earnings per share"); the solver tries the original first, so this
    only ever ADDS coverage, never changes a working parse."""
    out = [(conjuncts, pct)]
    shorts = [i for i, c in enumerate(conjuncts) if len(c.split()) == 1]
    longs = [c.split() for i, c in enumerate(conjuncts) if i not in shorts]
    if shorts and longs:
        tails = [w[1:] for w in longs]
        if all(t == tails[0] for t in tails) and tails[0]:
            comp = [c + " " + " ".join(tails[0]) if i in shorts else c
                    for i, c in enumerate(conjuncts)]
            out.append((comp, [_has_pct_word(c) or p for c, p in zip(comp, pct)]))
    return out


def _split_two_sides(question: str, periods: list[dict]
                    ) -> list[tuple[str, tuple[int, int], str, tuple[int, int], bool]]:
    """Candidate comparison splits in reading order. Each candidate is
    (side1, span1, side2, span2, directional) with EXACT absolute spans in
    blanked-question coordinates. span1 INCLUDES the consumed separator
    gap: a greedy separator ("\\s+and\\s+") eats blanked periods adjacent
    to it ("sales in [P1] and [P2]"). The side boundary runs through the
    START of the separator word: leading whitespace eats periods that
    modify the LEFT noun ("in [P1] and" -> side 1), trailing whitespace
    eats periods that start the RIGHT side ("and [P2]" -> side 2).
    Every " and " occurrence yields a candidate (entities like "Wearables,
    Home and Accessories" contain "and"); the solver validates each
    against the facts and the first valid one wins. directional=True only
    for "from A to B", whose operand order is end-minus-start."""
    full_blank = blank_periods(question, periods)
    low = full_blank.lower()
    cands: list[tuple[str, tuple[int, int], str, tuple[int, int], bool]] = []
    seps = (r"\s+and\s+", r"\s+to\s+", r"\s+versus\s+", r"\s+vs\.?\s+",
            r"\s+less\s+", r"\s+minus\s+", r"\s+compared\s+(?:to|with)\s+",
            r"\s+relative\s+to\s+")
    for pat, pid in ((r"\bdifference\s+(?:between|in)\b(.+)", "diff"),
                     (r"\bchange\b.{0,10}?\bin\b(.+)", "chg"),
                     (r"\bchange\b(.+)", "chg"),
                     (r"\bfrom\b(.+)", "from"),
                     (r"\bcompared\s+(?:to|with)\b(.+)", "cmp"),
                     (r"\bversus\b(.+)", "cmp"),
                     (r"\bvs\.?\b(.+)", "cmp")):
        m = re.search(pat, low)
        if not m:
            continue
        # Slice the ORIGINAL-cased blanked string at the match span.
        bstart = m.start(1)
        body = full_blank[bstart:]
        for sep in seps:
            found = False
            for n, sm in enumerate(re.finditer(sep, body, flags=re.I)):
                if sep != r"\s+and\s+" and n:
                    break  # only "and" enumerates; others take 1st
                if n >= 4:
                    break
                s1, s2 = body[:sm.start()], body[sm.end():]
                # Either side may be period-only (blanked to spaces) --
                # its metric is inherited, its period recovered by span.
                if not (s1.strip(" ?.,") or s2.strip(" ?.,")):
                    continue
                # Side boundary = start of the separator's core word.
                mtext = sm.group()
                core_abs = bstart + sm.start() + (len(mtext) - len(mtext.lstrip()))
                # "from A to B" means end-minus-start -- whether the outer
                # pattern was "from" (which consumes the word) or "change"
                # ("change from A to B", where the body still starts with
                # it). Any other " to " split stays positional.
                directional = (sep == r"\s+to\s+"
                               and (pid == "from"
                                    or re.match(r"\s*from\b", body, re.I)))
                cands.append((s1, (bstart, core_abs), s2,
                              (core_abs, bstart + len(body)), directional))
                found = True
            if found:
                break  # separator priority preserved
        if cands:
            break  # first matching outer pattern wins
    if not cands:
        # "X less/minus Y" without a leading keyword.
        for sep in (r"\s+less\s+", r"\s+minus\s+"):
            sm = re.search(sep, full_blank, flags=re.I)
            if not sm:
                continue
            s1, s2 = full_blank[:sm.start()], full_blank[sm.end():]
            if not (s1.strip(" ?.,") and s2.strip(" ?.,")):
                continue
            mtext = sm.group()
            core_abs = sm.start() + (len(mtext) - len(mtext.lstrip()))
            cands.append((s1, (0, core_abs), s2, (core_abs, len(full_blank)), False))
            break
    return cands


def _periods_in_span(periods: list[dict], span: tuple[int, int]) -> list[dict]:
    # Periods were blanked; recover which belong to this side by span.
    return [p for p in periods if span[0] <= p["start"] < span[1]]


def parse_pair(question: str, periods: list[dict], op: str) -> list[dict]:
    """Candidate pair parses. The solver validates each against the facts;
    the first valid one wins."""
    # Single-period percent phrasing ("percent change in X for Q3 2022",
    # "percentage increase in iPhone sales in Q3 2022"): no two sides to
    # split -- the solver answers stated-first, else infers the YoY base.
    if op == "percent_change" and len(periods) == 1:
        full_blank = blank_periods(question, periods)
        m = re.search(r"\bpercent(?:age)?\s+(?:change|increase|decrease|growth)\s+(?:in|of)\b(.+)",
                      full_blank, re.I)
        metric = _clean_metric(m.group(1)) if m else ""
        if not metric:
            m2 = re.search(r"\b(?:growth rate|year[- ]over[- ]year|yoy)\b(?:\s+of)?\s*(.+)",
                           full_blank, re.I)
            metric = _clean_metric(m2.group(1)) if m2 else ""
        if not metric:
            return []
        return [{"op": op, "conjuncts": [metric, metric],
                 "periods": [periods[0], periods[0]], "single_period": True,
                 "hint": conjunct_tokens(metric), "pct": [False, False]}]
    cands = _split_two_sides(question, periods)
    if not cands:
        # "from P1 to P2" with periods only: metric lives before "from".
        m = re.search(r"\bchange\b.{0,10}?\bin\b(.+?)\bfrom\b", blank_periods(question, periods), re.I)
        if not m or len(periods) < 2:
            return []
        metric = _clean_metric(m.group(1))
        if not metric:
            return []
        return [{"op": op, "conjuncts": [metric, metric],
                 "periods": [periods[0], periods[1]], "hint": conjunct_tokens(metric),
                 "pct": [False, False]}]
    full_blank = blank_periods(question, periods)
    out: list[dict] = []
    for s1, span1, s2, span2, directional in cands:
        if directional:
            # "from A to B" means end-minus-start: operand order is
            # [to-side, from-side] so the solver's op1-op2 reads B-A.
            s1, span1, s2, span2 = s2, span2, s1, span1
        p1 = _periods_in_span(periods, span1)
        p2 = _periods_in_span(periods, span2)
        c1, c2 = _clean_metric(s1), _clean_metric(s2)
        f1, f2 = _has_pct_word(s1), _has_pct_word(s2)
        # Metric inheritance across sides (flags inherit with the metric;
        # frame text never sets flags -- "percentage increase from P1 to
        # P2" describes the OPERATION, not percent-kind operands).
        if not c1 and not c2:
            # Metric stated before the splitter ("change in revenue from..").
            m = re.search(r"\b(?:change|difference)\s+(?:in|between)\b(.+?)\b(?:from|between)\b",
                          full_blank, re.I)
            pre = _clean_metric(m.group(1)) if m else ""
            if not pre:
                # Last resort: content words before the first period.
                pre = _clean_metric(full_blank[:periods[0]["start"]] if periods else "")
            if not pre:
                continue
            c1 = c2 = pre
            f1 = f2 = False
        elif not c1:
            c1 = c2
            f1 = f2
        elif not c2:
            c2 = c1
            f2 = f1
        # Period assignment.
        if p1 and p2:
            per1, per2 = p1[0], p2[0]
        elif p1 and not p2:
            per1 = per2 = p1[0]
        elif p2 and not p1:
            per1 = per2 = p2[0]
        elif len(periods) == 1:
            per1 = per2 = periods[0]
        else:
            continue
        # Same metric + same period = meaningless (each operand otherwise
        # keeps its own side's period, covering "iPhone in P1 vs Mac in P2").
        same_metric = conjunct_tokens(c1) == conjunct_tokens(c2)
        same_period = (per1["start"], per1["end"]) == (per2["start"], per2["end"])
        if same_metric and same_period:
            continue
        hint = conjunct_tokens(c1 + " " + c2)
        for cc, pp in _ellipsis_variants([c1, c2], [f1, f2]):
            cand = {"op": op, "conjuncts": cc, "periods": [per1, per2],
                    "hint": hint, "pct": pp}
            if cand not in out:
                out.append(cand)
    return out


def parse_ratio(question: str, periods: list[dict]) -> list[dict]:
    full_blank = blank_periods(question, periods)
    m = re.search(r"\bwhat\s+(?:percent(?:age)?|share|proportion|fraction|portion)\s+of\s+(.+?)\s+"
                  r"(?:is|are|was|were|does|do)\s+(.+?)\s*(?:represent|account|comprise|constitute|make up)?\s*\??\s*$",
                  full_blank, re.I)
    if m:
        whole, part = _clean_metric(m.group(1)), _clean_metric(m.group(2))
    else:
        m2 = re.search(r"(.+?)\s+as\s+a\s+(?:percent(?:age)?|share|proportion|portion)\s+of\s+(.+?)\s*\??\s*$",
                       full_blank, re.I)
        if m2:
            # Trim question frame from the front ("what was ...").
            head = re.sub(r"^.*?\b(was|were|is|are)\b", "", m2.group(1), flags=re.I)
            part, whole = _clean_metric(head), _clean_metric(m2.group(2))
        else:
            # "ratio of A to B (for P)": A is the part, B the whole.
            m3 = re.search(r"\bratio\s+of\s+(.+?)\s+to\s+(.+?)\s*\??\s*$", full_blank, re.I)
            if not m3:
                return []
            part, whole = _clean_metric(m3.group(1)), _clean_metric(m3.group(2))
    if not part or not whole:
        return []
    keys = {(p.get("duration"), p.get("year"), p.get("month"), p.get("day")) for p in periods}
    if len(keys) != 1:
        return []
    return [{"op": "ratio", "conjuncts": [part, whole], "period": periods[0],
             "hint": conjunct_tokens(part + " " + whole)}]


# --------------------------------------------------------------------------
# Row matching + table selection
# --------------------------------------------------------------------------

def _row_score_one(cset: set[str], stripped: set[str], tset: set[str]) -> int:
    if cset == tset:
        return 100
    if cset <= tset:
        s = 90
        if "total" not in cset and "total" in tset:
            s += 8  # bare "net sales" prefers the Total row
        if "total" in cset and "total" not in tset:
            s -= 50
        return s
    # Lenient tier: the conjunct names an entity plus metric nouns
    # ("Mac net sales", "Wearables, Home and Accessories net sales").
    # Never fires on bare metrics (stripped would be empty) and never
    # outranks a strict-tier match, so "total net sales" still beats
    # every other "total X" row for that conjunct.
    if stripped and stripped <= tset:
        return 85
    return 0


def row_score(conj: list[str], fact: dict) -> int:
    """100 exact, ~90 full containment (+total preference), 85
    entity-minus-metric containment, else 0. Scored against BOTH the
    full row label and its core, taking the MAX: "total assets" must hit
    the "Total assets" core exactly (100) rather than tie every "Total X
    assets" row at 90. Partial overlap is otherwise unusable --
    precision over recall, the LLM fallback handles the rest."""
    if not conj:
        return 0
    cset = set(conj)
    stripped = cset - _METRIC_WORDS
    best = 0
    for key in ("row_norm", "row_core_norm"):
        toks = fact.get(key, "").split()
        if not toks:
            continue
        best = max(best, _row_score_one(cset, stripped, {_stem(t) for t in toks}))
    return best


def _table_key(f: dict) -> tuple:
    return (f.get("page"), f.get("table_index"))


def select_table(operands: list[tuple[list[str], dict]], facts: list[dict],
                 hint: list[str], pct_flags: list[bool] | None = None
                 ) -> tuple[tuple | None, list[dict]]:
    """Choose the single table covering ALL operands with the best joint
    score. Returns (table_key, chosen facts in operand order)."""
    # Candidates per operand: facts matching period + row score >= 80.
    # A percent-flagged operand ("X percentage") matches % rows ONLY --
    # silently substituting the dollar row would answer a question the
    # user did not ask, and there is no meaningful arithmetic on the
    # result anyway, so a missing % row fails the parse.
    cand_lists: list[list[tuple[int, dict]]] = []
    flags = pct_flags or [False] * len(operands)
    for (conj, spec), flag in zip(operands, flags):
        cands = []
        for f in facts:
            if f.get("is_change"):
                continue
            if not fact_matches_period(f, spec):
                continue
            s = row_score(conj, f)
            if s >= 80:
                cands.append((s, f))
        if flag:
            cands = [(s, f) for s, f in cands if f.get("is_percent")]
        if not cands:
            return None, []
        cand_lists.append(cands)
    # Tables covering every operand.
    by_table: dict[tuple, list[list[tuple[int, dict]]]] = {}
    for i, cands in enumerate(cand_lists):
        for s, f in cands:
            by_table.setdefault(_table_key(f), [[] for _ in cand_lists])[i].append((s, f))
    best_key, best_score, best_facts = None, -1, []
    hint_set = set(hint or [])
    for key in sorted(by_table):
        per_op = by_table[key]
        if any(not lst for lst in per_op):
            continue
        # Ambiguity guard: each operand's best must be unique in-table
        # (no two different rows tying for best).
        chosen, total = [], 0
        ok = True
        for lst in per_op:
            lst_sorted = sorted(lst, key=lambda x: -x[0])
            top = [x for x in lst_sorted if x[0] == lst_sorted[0][0]]
            rows = {(x[1].get("row"), x[1].get("col")) for x in top}
            if len(rows) > 1:
                ok = False
                break
            chosen.append(lst_sorted[0][1])
            total += lst_sorted[0][0]
        if not ok:
            continue
        cap = norm_text(chosen[0].get("caption", "")).split()
        total += 10 if hint_set and hint_set <= set(cap) else 0
        if total > best_score:
            best_key, best_score, best_facts = key, total, chosen
    if best_key is None:
        return None, []
    return best_key, best_facts


def combinable_units(units: list[str]) -> str | None:
    """Normalized common scale, or None when operands must not combine."""
    norms = [norm_unit(u) for u in units]
    if len(set(norms)) != 1 or norms[0] == "unspecified":
        return None
    return norms[0]


# --------------------------------------------------------------------------
# Stated-first lookups (quote the filing instead of computing)
# --------------------------------------------------------------------------

def find_stated_change(conj: list[str], spec: dict, facts: list[dict]) -> dict | None:
    """A Change-column fact for the same metric + duration ('2%')."""
    dur = spec.get("duration")
    if not dur or dur == "point":
        return None
    best, best_s = None, 0
    for f in facts:
        if not f.get("is_change") or f.get("duration") != dur:
            continue
        s = row_score(conj, f)
        if s >= 88 and s > best_s:
            best, best_s = f, s
    return best


def find_stated_percent(part_conj: list[str], spec: dict, facts: list[dict],
                        table_key: tuple | None = None) -> dict | None:
    """A percentage ROW for the metric ('Total gross margin percentage
    43.3%') in the same table when possible, else anywhere."""
    cands = []
    for f in facts:
        if not f.get("is_percent") or f.get("is_change"):
            continue
        if table_key and _table_key(f) != table_key:
            continue
        if not fact_matches_period(f, spec):
            continue
        toks = set(f.get("row_norm", "").split())
        if not (set(part_conj) <= toks):
            continue
        if not ({"percent", "percentage", "margin", "rate"} & toks):
            continue
        cands.append(f)
    if not cands and table_key is not None:
        return find_stated_percent(part_conj, spec, facts, None)
    # Prefer the "total" row when several match.
    cands.sort(key=lambda f: ("total" not in f.get("row_norm", ""), f.get("page", 0)))
    return cands[0] if cands else None


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def fmt_num(v: float) -> str:
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def fmt_pct(v: float) -> str:
    return f"{abs(v):.1f}%"


def unit_phrase(unit_norm: str, currency: str) -> str:
    if unit_norm in ("millions", "thousands", "billions"):
        return f"in {unit_norm}"
    if unit_norm == "per share":
        return "per share"
    if unit_norm == "shares":
        return "in thousands of shares"
    return ""


def money(v: float, currency: str) -> str:
    s = fmt_num(v)
    return f"${s}" if currency == "$" else s


def _pages_str(pages: list[int]) -> str:
    ps = sorted(set(pages))
    return "page " + ", ".join(str(p) for p in ps)


def _sources(facts: list[dict]) -> list[dict]:
    out, seen = [], set()
    for f in sorted(facts, key=lambda x: (x.get("page", 0), x.get("table_index", 0))):
        key = (f.get("page"), f.get("table_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"page": f.get("page"), "section": None,
                    "rerank_score": 1.0, "chunk_type": "table",
                    "caption": (f.get("caption") or "")[:120]})
    return out


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def _distinct_cells(chosen: list[dict]) -> bool:
    cells = {(f.get("row_norm"), f.get("col")) for f in chosen}
    return len(cells) == len(chosen)


def solve_sum(parsed: dict, facts: list[dict]) -> dict | None:
    ops = [(conjunct_tokens(c), parsed["period"]) for c in parsed["conjuncts"]]
    if any(not c for c, _ in ops):
        return None
    key, chosen = select_table(ops, facts, parsed["hint"], parsed.get("pct"))
    if key is None:
        return None
    # Every conjunct must resolve to a DISTINCT cell: without this, a bad
    # segmentation ("Wearables" / "Home" / "Accessories") triple-counts one
    # row. A failed segmentation returns None so the next candidate (or the
    # LLM fallback) is tried -- it never produces a wrong sum.
    if not _distinct_cells(chosen):
        return None
    if any(f.get("is_percent") for f in chosen):
        return None
    unit = combinable_units([f.get("units", "") for f in chosen])
    if unit is None:
        return None
    cur = chosen[0].get("currency", "")
    if any(f.get("currency", "") != cur for f in chosen):
        return None
    total = sum(f["value"] for f in chosen)
    per = parsed["period"]["text"]
    bits = " plus ".join(f"{c} ({money(f['value'], cur)})"
                         for c, f in zip(parsed["conjuncts"], chosen))
    up = unit_phrase(unit, cur)
    ans = (f"{bits} for {per} equals {money(total, cur)}"
           + (f" ({up}; {_pages_str([f['page'] for f in chosen])})."
              if up else f" ({_pages_str([f['page'] for f in chosen])})."))
    return {"answer": ans, "sources": _sources(chosen)}


def solve_difference(parsed: dict, facts: list[dict]) -> dict | None:
    ops = [(conjunct_tokens(c), p) for c, p in zip(parsed["conjuncts"], parsed["periods"])]
    if any(not c for c, _ in ops):
        return None
    key, chosen = select_table(ops, facts, parsed["hint"], parsed.get("pct"))
    if key is None:
        return None
    # Same cell twice ("cash" minus "cash equivalents" resolving to one
    # row) is a degenerate $0, not an answer -- refuse so the next split
    # candidate (or the LLM fallback) is tried instead.
    if not _distinct_cells(chosen):
        return None
    if any(f.get("is_percent") for f in chosen):
        return None
    unit = combinable_units([f.get("units", "") for f in chosen])
    if unit is None:
        return None
    cur = chosen[0].get("currency", "")
    if any(f.get("currency", "") != cur for f in chosen):
        return None
    a, b = chosen
    result = a["value"] - b["value"]
    la = f"{parsed['conjuncts'][0]} ({parsed['periods'][0]['text']})"
    lb = f"{parsed['conjuncts'][1]} ({parsed['periods'][1]['text']})"
    up = unit_phrase(unit, cur)
    ans = (f"{la} ({money(a['value'], cur)}) minus {lb} ({money(b['value'], cur)}) "
           f"equals {money(result, cur)}"
           + (f" ({up}; {_pages_str([a['page'], b['page']])})."
              if up else f" ({_pages_str([a['page'], b['page']])})."))
    return {"answer": ans, "sources": _sources(chosen)}


def _temporal_key(f: dict) -> tuple:
    rank = {"3mo": 1, "6mo": 2, "9mo": 3, "12mo": 4, "point": 5}.get(f.get("duration") or "point", 9)
    return (f.get("year") or 0, f.get("month") or 0, f.get("day") or 0, rank)


def solve_percent_change(parsed: dict, facts: list[dict]) -> dict | None:
    ops = [(conjunct_tokens(c), p) for c, p in zip(parsed["conjuncts"], parsed["periods"])]
    if any(not c for c, _ in ops):
        return None
    # STATED FIRST: the filing's own Change column for this metric.
    same_metric = ops[0][0] == ops[1][0]
    spec = parsed["periods"][0]
    if same_metric:
        stated = find_stated_change(ops[0][0], spec, facts)
        if stated is not None:
            v = stated["value"]
            direction = "increased" if v >= 0 else "decreased"
            dl = stated.get("duration_label") or stated.get("duration") or ""
            ans = (f"{parsed['conjuncts'][0]} {direction} by {fmt_pct(v)} "
                   f"({dl}; {_pages_str([stated['page']])}).")
            return {"answer": ans, "sources": _sources([stated])}
    chosen: list[dict] = []
    if parsed.get("single_period"):
        dur = spec.get("duration")
        if not dur or not spec.get("year"):
            return None
        if dur == "point":
            # "Change in cash as of June 25, 2022": the base is the same
            # row's nearest EARLIER point column in the same table (the
            # balance sheet's Sept 2021 column). No earlier point -> None.
            key1, chosen1 = select_table([(ops[0][0], spec)], facts,
                                         parsed["hint"], [False])
            if key1 is None:
                return None
            new = chosen1[0]
            tk = _temporal_key(new)
            cands = [f for f in facts
                     if _table_key(f) == key1
                     and (f.get("duration") or "point") == "point"
                     and f.get("row_norm") == new.get("row_norm")
                     and f.get("col") != new.get("col")
                     and _temporal_key(f) < tk
                     and not f.get("is_change") and not f.get("is_percent")]
            if not cands:
                return None
            chosen = [max(cands, key=_temporal_key), new]
        else:
            # No stated figure: "change for [duration D, year Y]" means
            # YoY, so the base is (D, Y-1) with no month/day (fiscal
            # dates shift year to year -- June 25 vs June 26 -- so only
            # duration+year identify the base). Missing base -> None.
            base_spec = {"duration": dur, "year": spec["year"] - 1,
                         "month": None, "day": None, "start": -1, "end": -1,
                         "text": f"same period in {spec['year'] - 1}"}
            ops = [(ops[0][0], spec), (ops[0][0], base_spec)]
            _, chosen = select_table(ops, facts, parsed["hint"], parsed.get("pct"))
            if not chosen:
                return None
    else:
        _, chosen = select_table(ops, facts, parsed["hint"], parsed.get("pct"))
        if not chosen:
            return None
    if any(f.get("is_percent") for f in chosen):
        return None
    unit = combinable_units([f.get("units", "") for f in chosen])
    if unit is None:
        return None
    # Base = earlier period, new = later.
    ordered = sorted(chosen, key=_temporal_key)
    if _temporal_key(ordered[0]) == _temporal_key(ordered[1]):
        return None
    base, new = ordered
    if base["value"] == 0:
        return None
    pct = (new["value"] - base["value"]) / base["value"] * 100
    direction = "an increase" if pct >= 0 else "a decrease"
    ans = (f"{new['row']} ({fmt_num(new['value'])}) compared to {base['row']} "
           f"({fmt_num(base['value'])}) is {direction} of {fmt_pct(pct)} "
           f"({_pages_str([base['page'], new['page']])}).")
    return {"answer": ans, "sources": _sources(chosen)}


def solve_ratio(parsed: dict, facts: list[dict]) -> dict | None:
    part_c, whole_c = conjunct_tokens(parsed["conjuncts"][0]), conjunct_tokens(parsed["conjuncts"][1])
    if not part_c or not whole_c:
        return None
    spec = parsed["period"]
    ops = [(part_c, spec), (whole_c, spec)]
    key, chosen = select_table(ops, facts, parsed["hint"])
    # STATED FIRST: a percentage row for the part in the same table.
    stated = find_stated_percent(part_c, spec, facts, key)
    if stated is not None:
        ans = (f"{stated['row']} was {fmt_pct(stated['value'])} "
               f"({stated['col']}; {_pages_str([stated['page']])}).")
        return {"answer": ans, "sources": _sources([stated])}
    if key is None:
        return None
    if any(f.get("is_percent") for f in chosen):
        return None
    unit = combinable_units([f.get("units", "") for f in chosen])
    if unit is None:
        return None
    part, whole = chosen
    if whole["value"] == 0:
        return None
    pct = part["value"] / whole["value"] * 100
    ans = (f"{part['row']} ({fmt_num(part['value'])}) is {fmt_pct(pct)} of "
           f"{whole['row']} ({fmt_num(whole['value'])}) "
           f"({_pages_str([part['page'], whole['page']])}).")
    return {"answer": ans, "sources": _sources(chosen)}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

_SOLVERS = {"sum": (parse_sum, solve_sum),
            "difference": (parse_pair, solve_difference),
            "percent_change": (parse_pair, solve_percent_change),
            "ratio": (parse_ratio, solve_ratio)}


def try_answer(question: str, facts: list[dict] | str) -> dict | None:
    """Deterministic computation. Returns {"answer", "sources"} or None
    (None = fall through to the existing LLM pipeline). Never raises."""
    try:
        intent = detect_intent(question)
        if intent is None:
            return None
        if isinstance(facts, str):
            facts = load_facts(facts)
        if not facts:
            return None
        periods = extract_periods(question)
        if not periods and intent in ("sum", "ratio"):
            return None
        if not periods and intent in ("difference", "percent_change"):
            return None
        parse_fn, solve_fn = _SOLVERS[intent]
        if intent in ("difference", "percent_change"):
            cands = parse_fn(question, periods, intent)
        else:
            cands = parse_fn(question, periods)
        # Every parse is validated against the facts inside the solver;
        # the first valid one wins, else fall through to the LLM pipeline.
        for parsed in cands:
            res = solve_fn(parsed, facts)
            if res is not None:
                return res
        return None
    except Exception:
        return None
