"""Shared utilities for evaluation modules."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Closed markdown fence: ```<info-string>\n ... ``` (```json or bare ```).
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
# Bound on the number of `{` offsets the step-3 recovery scan attempts —
# guards pathological many-brace inputs (#1934; each failed raw_decode is
# cheap, but an unbounded scan over adversarial input is not).
_MAX_BRACE_CANDIDATES = 200
# Step-4 recovery (#2222): the ENTIRE last non-empty line is a bare number,
# or a labeled score line ("Score: 75", optional markdown emphasis). Strict
# fullmatch — a number embedded in prose ("score of 20.") never matches, so
# genuinely scoreless responses still return None (rule 9).
_TRAILING_SCALAR_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TRAILING_SCORE_LINE_RE = re.compile(
    r"[*_`#\s]*score[*_`\s]*[:=][*_`\s]*(-?\d+(?:\.\d+)?)[*_`\s]*", re.IGNORECASE
)


def parse_judge_json(text: str) -> dict | list | str | int | float | bool | None:
    """Extract the largest recoverable JSON value from judge response text.

    Ladder (order is load-bearing — steps 1-2 are the pre-#1934 behavior with
    byte-identical precedence, so every input that parsed before returns the
    IDENTICAL value):

    1. ``json.loads(text)`` VERBATIM when the whole text is valid JSON —
       including bare scalars (``"85"`` -> ``85``): graded-judge callers
       (``eval/graded_judge.py::_score_from_parsed``, #778 r3) depend on the
       scalar passthrough.
    2. ``raw_decode`` at the FIRST ``{`` in the text.
    3. Recovery (#1934, runs ONLY when 1-2 fail): ONE unified largest-wins
       candidate pool over (a) every closed markdown fence block (```json or
       bare ```), ``json.loads`` on the stripped block, and (b) ``raw_decode``
       at subsequent ``{`` offsets (bounded scan). The LARGEST successful
       decode wins; the span metric is the consumed span ``end - start`` for a
       raw_decode candidate and the STRIPPED BLOCK's character length for a
       fenced candidate; ties break to the earliest start offset. This
       recovers the measured #1773 failure shape: a reasoning preamble whose
       stray ``{`` mis-anchors the first-brace decode (2.02% of 128,712
       describe calls; 39/40 sampled failures were ``end_turn``-complete).
    4. Trailing bare-scalar / labeled score line (#2222, runs ONLY when 1-3
       fail): when the ENTIRE last non-empty line is a bare number — or a
       labeled score line (``"Score: 75"``, optional ``:``/``=`` and
       markdown emphasis) — return the number as a scalar (int/float — the
       same scalar type step 1 yields for a bare-number response). Recovers
       the two measured #2222 pilot failure shapes: a reason-then-bare-number
       response (``"...analysis...\\n\\n20"``) and a reason-then-labeled-score
       response (``"...analysis...\\n\\nScore: 75"``) — Sonnet following the
       persona-vectors rubric's own "just the number" instruction over the
       JSON wrapper on hard code/math rows (28/150 hallucination pilot
       draws, concentrated on insecure_code + mistake_math; every sampled
       failure ``end_turn``-complete with the score on the final line). A
       number embedded in prose never matches (fullmatch on the whole
       line), so genuinely scoreless responses still return None;
       out-of-range scalars stay with the callers' drop-never-coerce range
       checks (``_score_from_parsed`` / ``_normalize_scalar_score``).

    On parse failure returns ``None`` — NEVER a coerced placeholder
    (drop-never-coerce, ``.claude/rules/llm-judging.md`` rule 9; the #766
    ``eval/belief.py`` precedent). Callers must DROP a ``None`` (record an
    error row / exclude from aggregates), never substitute a default score.
    The failure WARNING records total length + a ~500-char head + a ~200-char
    tail so truncation-vs-format is readable from the log (the tail is where
    a closing fence/brace shows; the old 200-char head alone drove two
    independent truncation mis-diagnoses on #1773).

    Pre-existing ambiguity, unchanged: the literal JSON ``null`` also returns
    ``None`` and is indistinguishable from a parse failure; every project
    rubric returns an object or scalar, so callers treat ``None`` uniformly
    as a dropped row.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    try:
        start = text.index("{")
        obj, _ = decoder.raw_decode(text, start)
        return obj
    except (ValueError, json.JSONDecodeError):
        pass
    # Step 3 — unified largest-wins recovery pool: (span, start_offset, value).
    candidates: list[tuple[int, int, object]] = []
    for m in _FENCE_RE.finditer(text):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            val = json.loads(block)
        except json.JSONDecodeError:
            continue
        candidates.append((len(block), m.start(), val))
    pos = text.find("{")
    n_tried = 0
    while pos != -1 and n_tried < _MAX_BRACE_CANDIDATES:
        n_tried += 1
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pass
        else:
            candidates.append((end - pos, pos, obj))
        pos = text.find("{", pos + 1)
    if candidates:
        candidates.sort(key=lambda c: (-c[0], c[1]))
        return candidates[0][2]
    # Step 4 — trailing bare-scalar / labeled score line (#2222; docstring
    # ladder step 4).
    tail_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if tail_lines:
        last = tail_lines[-1]
        if _TRAILING_SCALAR_RE.fullmatch(last):
            return float(last) if "." in last else int(last)
        m = _TRAILING_SCORE_LINE_RE.fullmatch(last)
        if m:
            num = m.group(1)
            return float(num) if "." in num else int(num)
    logger.warning(
        "Failed to parse judge JSON; returning None (caller must DROP this "
        "row, never coerce). len=%d head=%.500s ... tail=%s",
        len(text),
        text,
        text[-200:],
    )
    return None
