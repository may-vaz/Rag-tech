"""
llm_client.py
===============
Deliberately the ONLY file in this project that talks to Ollama, and
deliberately importing NOTHING beyond the standard library and
`requests`. No torch, no sentence-transformers, no faiss -- not
transitively, either. See eval/step2_generate.py, which verifies this
guarantee at runtime.

HISTORY OF FIXES IN THIS FILE (kept because each was a real, tested
finding, and understanding why matters more than the current state
alone):

1. Visible reasoning ("First, I need to...") leaking into every answer
   -> fixed with schema-constrained JSON output (Ollama's `format`
   field as an actual schema object, not the loose "json" string mode)
   plus strip_thinking() as a deterministic client-side safety net.

2. Suppressing reasoning (`think: false` + `/no_think`) for output
   cleanliness caused a WORSE problem: a period-misattribution
   hallucination on an out-of-scope question ("Temporal Semantic
   Confusion," a documented failure mode -- arXiv:2607.28661,
   arXiv:2607.11414). Fixed by restoring `think: true` (the model
   reasons as long as it needs; strip_thinking() + the JSON parser
   already handle cleaning up arbitrarily long reasoning) and by
   surfacing an explicit period-summary line into the context (see
   answer.py's build_context()) so the model has a structured signal
   for "is the requested period actually present," not just raw text
   to infer it from.

3. THIS VERSION'S FIX: arithmetic questions (e.g. "combined net sales
   of iPhone and Mac") were originally handled by asking the model to
   both identify the relevant numbers AND compute the result AND format
   the final sentence, in one call, using a schema with a nested
   optional object ("computation": {...} | null). In practice this
   returned an empty answer -- Ollama/llama.cpp's grammar-constrained
   JSON decoding (built on GBNF grammars converted from the JSON
   schema) is known to have real limitations and edge cases for more
   complex schema shapes; a nested, conditionally-null object is
   exactly the kind of construct documented as troublesome in
   Ollama/llama.cpp's own grammar-support discussions. Rather than
   guess at ever-more-complex schema workarounds, this version properly
   separates three concerns that don't need to live in one call at all:

     a) DETECTING that a question needs arithmetic at all -- now pure,
        deterministic Python (see _detect_computation_operation), not
        the LLM's job. General keyword/phrase patterns (e.g. "combined
        X and Y", "difference between", "percentage change"), not
        hardcoded to any specific number, company, or document.

     b) EXTRACTING which raw numbers are relevant -- still the LLM's
        job (a genuine reading-comprehension task it's actually good
        at), but now through a SEPARATE, much SIMPLER, flat schema
        (VALUES_SCHEMA: just a list of {label, value, page} objects,
        no nested optional object) used ONLY for computation questions.
        The original, simple ANSWER_SCHEMA is restored to EXACTLY what
        it was before this change for every other question -- so the
        8 previously-working question types take the identical code
        path they always did, completely unaffected by this fix.

     c) DOING THE ARITHMETIC -- _compute_result() below, in plain
        Python. Unchanged from the prior version (already tested
        against this project's real, verified figures). This is the
        PAL pattern (Program-Aided Language Models, Gao et al., ICML
        2023, arXiv:2211.10435): "LLMs decompose problems well but
        execute arithmetic badly" -- delegating ONLY the execution step
        to code, not the model, is what the research shows produces
        large, reliable accuracy gains (+38.1pp specifically on
        problems with larger numbers, the exact regime financial
        figures live in).

   This also directly satisfies "don't disrupt the other part of the
   code": the computation detector only fires on questions matching its
   patterns, so a lookup question that already worked takes the exact
   same call_ollama() path, with the exact same schema and prompt, that
   it always did.
"""

from __future__ import annotations

import json as jsonlib
import re

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b"          # fallback: "llama3.2:3b" if this is too slow/heavy on your machine
OLLAMA_TIMEOUT_SECONDS = 600         # generous: see step2_generate.py's resumable design, which exists
                                        # specifically so one slow question doesn't cost you the others
MAX_SELF_CORRECTION_RETRIES = 1     # bounded: only retries when a real problem is detected, so the
                                        # common case pays zero extra latency

# =============================================================================
# SHARED HELPERS (used by both the lookup path and the computation path)
# =============================================================================

_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")  # thousands-grouped numbers, decimals, percents --
                                                             # deliberately does NOT allow a dangling
                                                             # trailing comma (a naive \d[\d,]* pattern
                                                             # would greedily swallow sentence punctuation
                                                             # like "82,959," -- found by testing against
                                                             # a real sentence, not assumed)


def strip_thinking(text: str) -> str:
    """
    Removes everything up to and including the LAST '</think>' tag in
    the text (robust to multiple/nested occurrences), returning only
    what comes after. Safe to call unconditionally -- returns the text
    unchanged (stripped) if no closing tag is present at all.
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _extract_json_object(text: str) -> dict | None:
    """
    Parse-and-repair, per Ollama's own documented caveat that even
    schema-constrained output can leak preamble. Tries a direct parse
    first; if that fails, looks for the first `{...}` block anywhere in
    the text and tries parsing just that. Returns None if both fail.
    """
    try:
        return jsonlib.loads(text)
    except jsonlib.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return jsonlib.loads(match.group(0))
        except jsonlib.JSONDecodeError:
            pass
    return None


def _looks_like_reasoning_leak(answer_text: str) -> bool:
    """Even inside a validly-parsed JSON field, the model could still
    stuff reasoning narration as the value rather than a clean answer --
    schema validity guarantees structure, not content."""
    lowered = answer_text.lower().lstrip()
    openers = ("first,", "let me", "i need to", "okay,", "the user is asking", "let's")
    if any(lowered.startswith(o) for o in openers):
        return True
    if len(answer_text) > 600:
        return True
    return False


def _parse_number(token: str) -> float | None:
    """Parses a number token (as matched by _NUMBER_RE) into a float,
    stripping commas and a trailing percent sign."""
    cleaned = token.replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_financial_figure(token: str) -> bool:
    """
    Distinguishes an actual asserted financial figure from an
    incidental bare digit sequence (e.g. the "4" in "Q4", or a bare
    year like "2022"). Only figures with a comma, decimal point, or
    percent sign count -- found necessary by testing: without this
    filter, a question mentioning "Q4 2022" extracted "4" and "2022" as
    if they were financial figures needing grounding, causing false-
    positive retry triggers unrelated to the actual asserted number.
    """
    return "," in token or "." in token or "%" in token


def _numbers_grounded(answer_text: str, source_text: str) -> bool:
    """
    Deterministic faithfulness check for the LOOKUP path: does every
    number the answer asserts either appear verbatim in the context, or
    equal the sum/difference of two numbers that DO appear? Vacuously
    true if the answer contains no financial figures to check.
    """
    answer_tokens = [t for t in _NUMBER_RE.findall(answer_text) if _is_financial_figure(t)]
    if not answer_tokens:
        return True

    source_tokens = set(_NUMBER_RE.findall(source_text))
    source_values = [v for v in (_parse_number(t) for t in source_tokens) if v is not None]

    for tok in answer_tokens:
        if tok in source_tokens:
            continue
        val = _parse_number(tok)
        if val is None:
            continue
        combined_ok = any(
            abs(a + b - val) < 0.5 or abs(a - b - val) < 0.5
            for i, a in enumerate(source_values)
            for b in source_values[i + 1:]
        )
        if not combined_ok:
            return False
    return True


def _value_grounded(value, source_text: str) -> bool:
    """
    Checks that a single extracted numeric value (from the computation
    path's 'values' list) actually appears verbatim in the context it
    was supposedly read from -- comma-formatted or plain form.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    candidates = {f"{f:,.0f}", str(int(f))} if f == int(f) else {f"{f:,.2f}".rstrip("0").rstrip("."), str(f)}
    return any(c in source_text for c in candidates)


def _compute_result(operation: str, values: list[dict]) -> str | None:
    """
    THE ACTUAL ARITHMETIC -- performed here, in plain Python, never by
    the model (see module docstring, part c, and the PAL citation).
    Returns a formatted answer sentence, or None if unusable (wrong
    arity for the operation, division by zero, malformed input) --
    None signals the caller to retry.
    """
    try:
        if not values:
            return None
        nums = [float(v["value"]) for v in values]
        labels = [str(v["label"]) for v in values]
        pages = sorted({int(v["page"]) for v in values})
        page_str = ", ".join(f"page {p}" for p in pages)

        if operation == "sum":
            result = sum(nums)
            parts = " plus ".join(f"{lbl} ({n:,.0f})" for lbl, n in zip(labels, nums))
            return f"{parts} equals {result:,.0f} ({page_str})."

        if operation == "difference":
            if len(nums) != 2:
                return None
            result = nums[0] - nums[1]
            return (f"{labels[0]} ({nums[0]:,.0f}) minus {labels[1]} ({nums[1]:,.0f}) equals "
                    f"{result:,.0f} ({page_str}).")

        if operation == "percent_change":
            if len(nums) != 2 or nums[0] == 0:
                return None
            base, new = nums[0], nums[1]
            pct = (new - base) / base * 100
            direction = "an increase" if pct >= 0 else "a decrease"
            return (f"{labels[1]} ({new:,.0f}) compared to {labels[0]} ({base:,.0f}) is {direction} "
                    f"of {abs(pct):.1f}% ({page_str}).")

        return None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _post_chat(messages: list[dict], model: str, schema: dict) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": True,       # the model may reason as long as it needs; strip_thinking() + the JSON
                                 # parser handle cleaning the output regardless of reasoning length.
        "format": schema,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 2048,  # generous, not a tight latency cap -- a still-reasoning model could
                                     # be cut off before reaching its answer with a tighter cap.
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running (open the Ollama desktop app, or run `ollama serve`), "
            f"and that you've pulled the model with `ollama pull {model}`."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama returned an error for model '{model}'. "
            f"Have you run `ollama pull {model}`? Original error: {e}"
        ) from e
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS}s for model '{model}'. "
            f"If this happens repeatedly on this machine, try a smaller model (e.g. llama3.2:3b) "
            f"or ensure no other memory-heavy processes are competing for RAM at the same time."
        ) from e

    data = resp.json()
    return data["message"]["content"]


# =============================================================================
# STAGE (a): DETECTING that a question needs arithmetic -- pure Python,
# no LLM call. General phrase patterns, not hardcoded to any specific
# number, label, or document. Deliberately conservative: each pattern
# requires a fairly specific phrase shape, not a single common word
# (e.g. "total of" + "and", not bare "total" -- bare "total" appears
# constantly in ordinary lookup questions like "total gross margin
# percentage" or "total shareholders' equity", which must NOT be
# rerouted into the computation path; verified against this project's
# own already-working test questions before shipping this).
# =============================================================================

_SUM_PATTERN = re.compile(r"\b(combined|sum of|total of)\b.*\band\b", re.I)
_DIFFERENCE_PATTERN = re.compile(r"\b(difference between|difference in)\b", re.I)
_PERCENT_CHANGE_PATTERN = re.compile(
    r"\b(percent(age)?\s+change|%\s*change|growth rate|year[- ]over[- ]year|yoy)\b", re.I
)


def _detect_computation_operation(question: str) -> str | None:
    """Returns 'sum' | 'difference' | 'percent_change' | None, based on
    the QUESTION text alone (not the full context, which could contain
    these words incidentally in unrelated narrative prose -- e.g. "net
    sales increased" -- causing false positives if checked)."""
    if _PERCENT_CHANGE_PATTERN.search(question):
        return "percent_change"
    if _DIFFERENCE_PATTERN.search(question):
        return "difference"
    if _SUM_PATTERN.search(question):
        return "sum"
    return None


# Public alias: answer.py's retrieval stage reuses this SAME detection
# function (single source of truth for "is this a computation
# question," not duplicated regex logic) to decide whether to widen the
# retrieval candidate pool -- see retrieve_context_with_ranking() in
# answer.py for why that widening is necessary (a real, measured
# retrieval failure, not a hypothetical).
detect_computation_operation = _detect_computation_operation


def _extract_question_text(user_prompt: str) -> str:
    """user_prompt is always built as 'CONTEXT:\\n\\n{context}\\n\\nQUESTION: {query}'
    (see answer.py) -- pulls out just the question part, so computation
    detection doesn't accidentally match trigger words that appear in
    the CONTEXT's narrative prose instead of the actual question."""
    if "QUESTION:" in user_prompt:
        return user_prompt.split("QUESTION:", 1)[-1].strip()
    return user_prompt


# =============================================================================
# LOOKUP PATH -- unchanged from the version that already correctly
# answered 8 of 9 real eval questions. Simple, flat, two-field schema.
# =============================================================================

SYSTEM_PROMPT = """You are a financial document assistant answering questions about a single \
SEC filing (Apple Inc.'s Form 10-Q). You are given retrieved excerpts from that filing as CONTEXT.

Respond with ONLY a JSON object matching this exact schema, and nothing else -- no text before or \
after it:
{"sufficient_evidence": true or false, "answer": "..."}

Rules:
1. Set "sufficient_evidence" to true only if the CONTEXT actually contains the specific \
information needed to answer the question. Set it to false if it does not -- do not guess or use \
outside knowledge about Apple, even if you have it.
2. If "sufficient_evidence" is false, set "answer" to an empty string "". Do not attempt to write \
a refusal message yourself -- that is handled separately.
3. If "sufficient_evidence" is true, set "answer" to ONLY the final answer: 1-3 plain-prose \
sentences with the source page number cited in parentheses (e.g. "(page 4)"). Quote exact figures \
as stated in the filing; do not round unless the filing itself rounds.
4. If a figure is already stated directly in the CONTEXT (e.g. a percentage or total the filing \
itself reports), quote it as-is. Never compute, estimate, or "infer from standard financial \
reporting" a number that is not explicitly present in the CONTEXT -- this filing almost always \
states the figure directly, and a computed guess is worse than the real number sitting in front \
of you.
5. Apple's fiscal year does not match the calendar year: the quarter ending in June is Apple's \
fiscal THIRD quarter (Q3), not Q2. Use the period language exactly as the filing states it (e.g. \
"three months ended June 25, 2022," or "the third quarter of 2022").
6. This filing reports many figures across FOUR overlapping periods: the three months ended June \
25, 2022; the three months ended June 26, 2021; the nine months ended June 25, 2022; and the nine \
months ended June 26, 2021. If the question does not specify a period and the CONTEXT has more \
than one period's figure for the same metric, include ALL relevant figures in "answer", each \
labeled with its exact period -- do not silently pick one.
7. The CONTEXT begins with a line listing which reporting periods the retrieved excerpts actually \
cover. If the question asks about a period that is NOT in that list, the answer is not available \
-- set "sufficient_evidence" to false. Do not assume, extrapolate, or substitute the closest \
available period's figure for a period that isn't listed.
8. Do not include any reasoning, step-by-step process, or explanation of how you found the answer \
anywhere in your response. The JSON object above is your entire response.

Some examples of correct behavior follow.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient_evidence": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["sufficient_evidence", "answer"],
}

# Fictional company ("Northwind Robotics Inc.") so there is zero overlap
# by construction with any real figure in this project's actual eval
# questions.
FEW_SHOT_EXAMPLES: list[tuple[str, dict]] = [
    (
        "CONTEXT:\n\n[Periods represented in retrieved excerpts: Q_MAR2024, Q_MAR2023, SIXMO_MAR2024, "
        "SIXMO_MAR2023]\n\n[Excerpt 1 -- page 15, from 'Item 2. Management's Discussion and Analysis', "
        "relevance score 0.983]\nOperating margin percentage:\nHardware 21.4% 20.8% 22.1% 19.6%\n"
        "Services 33.7% 31.2% 34.9% 30.4%\nTotal operating margin percentage 24.9% 23.5% 26.2% 22.8%\n\n"
        "QUESTION: What was Northwind Robotics' total operating margin percentage for the six months "
        "ended March 31, 2024?",
        {
            "sufficient_evidence": True,
            "answer": "Northwind Robotics' total operating margin percentage for the six months "
                      "ended March 31, 2024 was 26.2% (page 15).",
        },
    ),
    (
        "CONTEXT:\n\n[Periods represented in retrieved excerpts: Q_MAR2024, Q_MAR2023, SIXMO_MAR2024, "
        "SIXMO_MAR2023]\n\n[Excerpt 1 -- page 3, from 'PART I -- FINANCIAL INFORMATION', "
        "relevance score 0.979]\nThree Months Ended Six Months Ended\nMarch 31, 2024 March 31, 2023 "
        "March 31, 2024 March 31, 2023\nTotal revenue $ 41,220 $ 38,910 $ 79,540 $ 74,110\n\n"
        "QUESTION: What was Northwind Robotics' revenue for the fourth quarter of fiscal 2023?",
        {"sufficient_evidence": False, "answer": ""},
    ),
    (
        "CONTEXT:\n\n[Periods represented in retrieved excerpts: Q_MAR2024, Q_MAR2023, SIXMO_MAR2024, "
        "SIXMO_MAR2023]\n\n[Excerpt 1 -- page 9, from 'Note 4 -- Long-Term Debt', "
        "relevance score 0.961]\nTotal long-term debt $ 45,210 $ — $ — $ 45,210 $ 45,210 $ — $ —\n\n"
        "QUESTION: How much long-term debt did Northwind Robotics report as of March 31, 2024?",
        {
            "sufficient_evidence": True,
            "answer": "Northwind Robotics reported $45,210 thousand in total long-term debt as of "
                      "March 31, 2024 (page 9).",
        },
    ),
]

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "This filing does not contain sufficient information in the retrieved context to answer "
    "this question."
)
CORRECTIVE_REMINDER = (
    "Your previous response did not follow the required format (valid JSON matching the schema, "
    "with a plain-prose 1-3 sentence answer and no reasoning text) or asserted a figure not "
    "present in the CONTEXT. Respond again with ONLY the JSON object -- no other text."
)


def _build_lookup_messages(user_prompt: str, corrective: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_user, example_json in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": jsonlib.dumps(example_json)})
    messages.append({"role": "user", "content": user_prompt})
    if corrective:
        messages.append({"role": "user", "content": CORRECTIVE_REMINDER})
    return messages


def _extract_lookup_answer(parsed: dict) -> str:
    if not parsed.get("sufficient_evidence", True):
        return INSUFFICIENT_EVIDENCE_MESSAGE
    return str(parsed.get("answer", "")).strip()


def _call_lookup(user_prompt: str, model: str) -> str:
    def _attempt(corrective: bool) -> tuple[str, bool]:
        messages = _build_lookup_messages(user_prompt, corrective=corrective)
        raw = _post_chat(messages, model, ANSWER_SCHEMA)
        cleaned = strip_thinking(raw)
        parsed = _extract_json_object(cleaned)
        if parsed is None:
            return cleaned, True

        answer_text = _extract_lookup_answer(parsed)
        sufficient = parsed.get("sufficient_evidence", True)
        if sufficient and _looks_like_reasoning_leak(answer_text):
            return answer_text, True
        if sufficient and not _numbers_grounded(answer_text, user_prompt):
            return answer_text, True
        return answer_text, False

    answer_text, needs_retry = _attempt(corrective=False)
    retries = 0
    while needs_retry and retries < MAX_SELF_CORRECTION_RETRIES:
        answer_text, needs_retry = _attempt(corrective=True)
        retries += 1
    return answer_text


# =============================================================================
# COMPUTATION PATH -- a completely separate call, own schema, own
# prompt, own few-shot examples. Only ever invoked when
# _detect_computation_operation() finds a match; every other question
# never touches any of this.
# =============================================================================

COMPUTATION_SYSTEM_PROMPT = """You are extracting numeric values from excerpts of a SEC filing \
(Apple Inc.'s Form 10-Q) to answer a question that requires combining multiple numbers.

Respond with ONLY a JSON object matching this exact schema, and nothing else:
{"sufficient_evidence": true or false, "values": [{"label": "...", "value": number, "page": N}, ...]}

Rules:
1. Set "sufficient_evidence" to true only if the CONTEXT contains ALL the specific numbers needed \
to answer the question. Set it to false if any required number is missing.
2. If insufficient, set "values" to an empty list [].
3. If sufficient, list EACH relevant number as its own entry: a short "label" describing what it \
represents, its numeric "value" (no $ sign, no commas -- just the number), and the "page" it came \
from.
4. Only include numbers EXPLICITLY STATED in the CONTEXT. Never invent, estimate, or guess a \
value.
5. Do NOT perform any arithmetic yourself. Do not add, subtract, or combine the numbers -- only \
extract them exactly as they appear. Someone else will do the calculation.
6. If the question needs the SAME metric for TWO DIFFERENT periods (e.g. "the difference between \
net sales in Q3 2022 and Q3 2021," or "the change in revenue from 2021 to 2022"), extract it \
TWICE -- once per period -- and make each "label" specify WHICH period it is (e.g. "Net sales \
(three months ended June 25, 2022)" and "Net sales (three months ended June 26, 2021)"), not just \
the bare metric name repeated twice.
7. List the values in the SAME ORDER the question mentions them. If the question says "X and Y," \
list X's value first, then Y's value. If the question says "from A to B" or "compared to," list \
A's (the earlier/base) value first, then B's (the later/current) value. The order matters for the \
calculation performed afterward.
8. Do not include any reasoning or explanation anywhere in your response. The JSON object above is \
your entire response.

Some examples of correct behavior follow.
"""

VALUES_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient_evidence": {"type": "boolean"},
        "values": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "page": {"type": "integer"},
                },
                "required": ["label", "value", "page"],
            },
        },
    },
    "required": ["sufficient_evidence", "values"],
}

COMPUTATION_FEW_SHOT_EXAMPLES: list[tuple[str, dict]] = [
    (
        # Pattern 1: combining DIFFERENT named metrics (rows) for one period.
        "CONTEXT:\n\n[Periods represented in retrieved excerpts: Q_MAR2024, Q_MAR2023]\n\n"
        "[Excerpt 1 -- page 4, from 'Note 2 -- Revenue', relevance score 0.976]\n"
        "Hardware revenue $ 28,400 $ 26,850\nServices revenue $ 12,820 $ 12,060\n\n"
        "QUESTION: What was Northwind Robotics' combined hardware and services revenue for the "
        "three months ended March 31, 2024?",
        {
            "sufficient_evidence": True,
            "values": [
                {"label": "Hardware revenue", "value": 28400, "page": 4},
                {"label": "Services revenue", "value": 12820, "page": 4},
            ],
        },
    ),
    (
        # Pattern 2: the SAME metric across TWO DIFFERENT PERIODS -- this is
        # the pattern a "difference between X in period A and period B" or
        # "percent change from A to B" question needs (rule 6 and 7). Note
        # the labels distinguish the periods, and the order matches the
        # order the question mentions them (2024 mentioned first, so it's
        # listed first).
        "CONTEXT:\n\n[Periods represented in retrieved excerpts: Q_MAR2024, Q_MAR2023]\n\n"
        "[Excerpt 1 -- page 3, from 'PART I -- FINANCIAL INFORMATION', relevance score 0.986]\n"
        "Three Months Ended\nMarch 31, 2024 March 31, 2023\nTotal revenue $ 41,220 $ 38,910\n\n"
        "QUESTION: What was the difference between Northwind Robotics' total revenue in the three "
        "months ended March 31, 2024 and the three months ended March 31, 2023?",
        {
            "sufficient_evidence": True,
            "values": [
                {"label": "Total revenue (three months ended March 31, 2024)", "value": 41220, "page": 3},
                {"label": "Total revenue (three months ended March 31, 2023)", "value": 38910, "page": 3},
            ],
        },
    ),
    (
        # Pattern 3: insufficient evidence -- one of the two needed periods
        # isn't covered (2022 is not in the periods list), so this must
        # refuse rather than substitute the closest available period.
        "CONTEXT:\n\n[Periods represented in retrieved excerpts: Q_MAR2024, Q_MAR2023]\n\n"
        "[Excerpt 1 -- page 4, from 'Note 2 -- Revenue', relevance score 0.961]\n"
        "Total revenue $ 41,220 $ 38,910\n\n"
        "QUESTION: What was the difference between Northwind Robotics' total revenue in the three "
        "months ended March 31, 2022 and the three months ended March 31, 2024?",
        {"sufficient_evidence": False, "values": []},
    ),
]


def _build_computation_messages(user_prompt: str, corrective: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": COMPUTATION_SYSTEM_PROMPT}]
    for example_user, example_json in COMPUTATION_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": jsonlib.dumps(example_json)})
    messages.append({"role": "user", "content": user_prompt})
    if corrective:
        messages.append({"role": "user", "content": CORRECTIVE_REMINDER})
    return messages


def _call_computation(user_prompt: str, operation: str, model: str) -> str:
    def _attempt(corrective: bool) -> tuple[str, bool]:
        messages = _build_computation_messages(user_prompt, corrective=corrective)
        raw = _post_chat(messages, model, VALUES_SCHEMA)
        cleaned = strip_thinking(raw)
        parsed = _extract_json_object(cleaned)
        if parsed is None:
            return "", True

        if not parsed.get("sufficient_evidence", True):
            return INSUFFICIENT_EVIDENCE_MESSAGE, False

        values = parsed.get("values", [])
        if not values or not all(_value_grounded(v.get("value"), user_prompt) for v in values):
            return "", True

        result = _compute_result(operation, values)
        if result is None:
            return "", True
        return result, False

    answer_text, needs_retry = _attempt(corrective=False)
    retries = 0
    while needs_retry and retries < MAX_SELF_CORRECTION_RETRIES:
        answer_text, needs_retry = _attempt(corrective=True)
        retries += 1
    return answer_text or INSUFFICIENT_EVIDENCE_MESSAGE


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================


def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Routes to the computation path ONLY if the question (not the
    context, to avoid false positives from incidental narrative prose)
    matches a general arithmetic-intent pattern; otherwise uses the
    original, unchanged lookup path. `system_prompt` is accepted for
    backward-compatible call signatures (answer.py, eval scripts) but
    this function always uses its own SYSTEM_PROMPT/COMPUTATION_SYSTEM_
    PROMPT internally -- the caller's prompt text is not used directly,
    matching how this function's callers already only ever pass this
    module's own SYSTEM_PROMPT constant.
    """
    question = _extract_question_text(user_prompt)
    operation = _detect_computation_operation(question)
    if operation:
        return _call_computation(user_prompt, operation, model)
    return _call_lookup(user_prompt, model)