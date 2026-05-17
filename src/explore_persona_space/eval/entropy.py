"""Entropy helpers for #355 — measure entropy of answer conditional on CoT.

Module contract (consumed by ``scripts/measure_cot_entropy.py``):

- :func:`strip_trailing_answer` — apply the 5-rule strip pipeline to a CoT
  rationale string and return ``(stripped, rule_id)`` where ``rule_id`` is
  ``0`` (no rule matched) or ``1..5``.
- :func:`entropy_from_logprobs` — given a top-K logprob dict from a vLLM
  ``logprobs=20`` output, compute ``(H_full_renorm, H_abcd, top20_mass,
  abcd_total_mass_pre_renorm)``.
- :func:`miller_madow_entropy` — small-sample bias-corrected empirical
  entropy. Returns ``(H_mle, H_MM)``.
- :func:`answer_token_ids_for_tokenizer` — emit the set of token IDs that
  decode to each of {A, B, C, D} (covering naked and leading-space variants).

Why this is a library module and not inline in the entry script: the strip
pipeline + entropy math are unit-testable in CPU-only fixtures (no vLLM
needed), so they belong in src/.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# ────────────────────────────────────────────────────────────────────────────
# 5-rule answer-strip pipeline (plan §4 lines 106-112).
#
# Apply rules 1..5 IN ORDER against the FINAL non-empty line of the rationale
# (a "tail fragment" — anything after the last sentence terminator counts as
# the tail). First match wins; if none match, leave the text untouched and
# return rule_id=0.
#
# Rule 5 is the keyword-gated catch-all. It only fires when an answer-y
# keyword precedes the bare trailing letter within the same final line, to
# prevent stripping a legitimate sentence-final letter in prose
# (e.g. "...vitamin D.").
# ────────────────────────────────────────────────────────────────────────────

# Rule order is significant — first match wins. The first 5 rules ARE the
# canonical plan §4 list; the trailing "post-5" rules are additive patterns
# the smoke surfaced as needing coverage in real #186 data (see the
# coverage-check script results).
#
# All patterns:
#   - are case-insensitive,
#   - run via ``re.search`` against the FINAL non-empty line,
#   - allow an optional ``(`` + LETTER + ``)?`` parenthesized variant and
#     an optional ``.`` after the letter, with trailing whitespace.
_STRIP_RULES: tuple[tuple[int, re.Pattern[str]], ...] = (
    # Rule 1 (canonical) — "Answer: X"
    (1, re.compile(r"Answer:\s*\(?[A-D]\)?\.?\s*$", re.IGNORECASE)),
    # Rule 2 (canonical) — "my answer is X" (also catches "my final answer is X",
    # "my best answer is X" via the optional adjective).
    (
        2,
        re.compile(
            r"\bmy\s+(?:\w+\s+){0,2}answer\s+is\s*\(?[A-D]\)?\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    # Rule 3 (canonical) — "(the) correct answer is X"
    (
        3,
        re.compile(
            r"(?:the\s+)?correct\s+\w*\s*answer\s+is\s*\(?[A-D]\)?\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    # Rule 4 (canonical) — "(so )?the answer is X"  (also catches
    # "the best/final/correct/most likely answer is X" via the optional
    # adjective slot).
    (
        4,
        re.compile(
            r"(?:so\s+)?the\s+(?:\w+\s+){0,2}answer\s+is\s*\(?[A-D]\)?\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    # Rule 5 (canonical catch-all) — keyword-gated bare trailing letter.
    # Keyword set extended beyond plan §4's minimum (answer/correct/choice/
    # option/pick/choose) to cover statement/set/conclusion/method, which
    # appeared in #186 librarian + comedian generic_cot rationales as the
    # answer-delivering verb-object pair.
    #
    # IMPORTANT: only the KEYWORD part is case-insensitive (via the inline
    # ``(?i:...)`` group). The trailing letter class ``[A-D]`` MUST stay
    # case-sensitive — otherwise legitimate prose like "the method is
    # well-established." (last char ``d``) trips the rule.  Round-1 code
    # review M1 caught this regression after adding ``method`` to the
    # keyword list; see ``test_rule5_negative_prose_endings``.
    (
        5,
        re.compile(
            r"(?i:\b(?:answer|correct|choice|option|pick|choose|statement|set|"
            r"conclusion|method)\w*\b)"
            r"[^.!?]{0,80}?\(?([A-D])\)?\.?\s*$"
        ),
    ),
)


# Post-canonical strip rules. These have distinct rule_ids (8, 9) so the
# per-row JSONL audit column reports which one fired. Numbering picks up
# AFTER Rule 7 (the trailing-paren-letter rule defined below).
_POST_CANONICAL_STRIP_RULES: tuple[tuple[int, re.Pattern[str]], ...] = (
    # Rule 8 — multi-word "I'll go with X" / "I would go with X" /
    # "let's go with X". The phrase ``go with`` is the discriminator: it's
    # a 2-word answer-delivery cue that doesn't collide with prose endings.
    # Surfaced in comedian-eval rows under librarian source.
    (
        8,
        re.compile(r"(?i:\bgo\s+with\b)\s*\(?([A-D])\)?\.?\s*$"),
    ),
    # Rule 9 — "That's X." / "It's X." / "This is X." style contraction-led
    # answer-delivery. Surfaced in comedian-eval rows.
    (
        9,
        re.compile(r"(?i:\b(?:that|it|this|here)'?s\b)\s*\(?([A-D])\)?\.?\s*$"),
    ),
)


def _split_final_line(text: str) -> tuple[str, str]:
    """Return ``(head, last_line)`` where ``last_line`` is the final non-empty line.

    ``head`` is everything BEFORE the final non-empty line — including the
    trailing newline that preceded it (so ``head + last_line + <terminators>``
    reconstructs the original ``text``, where ``<terminators>`` is any pure
    whitespace tail). Any whitespace AFTER the last non-empty line is
    DROPPED — callers stripping the last line don't want spurious trailing
    blank lines re-injected into the rationale.

    History: a previous version did ``head = text[:line_start] + text[line_end:]``
    which re-attached the trailing whitespace BETWEEN the body and any
    post-strip content the caller produced, producing rationales like
    ``"Step 1.\\n\\n\\n\\nTherefore,"`` for partial-strip inputs. M1 in the
    round-1 code review caught this on real #186 CoT text (which often ends
    with a trailing newline after the answer line). See
    ``test_strip_drops_trailing_newlines_after_answer_line`` in
    ``tests/test_entropy_strip.py`` for the regression test.

    If the whole text is one line, ``head`` is the empty string.
    """
    if not text:
        return "", ""
    end = len(text)
    cursor = end
    # Walk backward past trailing empty/whitespace lines (these are DROPPED).
    while cursor > 0 and text[cursor - 1] in ("\n", "\r"):
        cursor -= 1
    # cursor now points one past the end of the last non-empty line.
    line_end = cursor
    # Walk back to find the previous newline.
    line_start = line_end
    while line_start > 0 and text[line_start - 1] not in ("\n", "\r"):
        line_start -= 1
    last_line = text[line_start:line_end]
    head = text[:line_start]
    return head, last_line


_CROSS_LINE_BARE_LETTER_RE = re.compile(r"^\s*\(?[A-D]\)?\.?\s*$")
_CROSS_LINE_KEYWORD_TAIL_RE = re.compile(
    r"\b(?:answer|correct|choice|option|conclusion|method|set|statement|pick|choose)\w*\b"
    r"[^.!?]{0,80}?(?:\bis|:)\s*$",
    re.IGNORECASE,
)
# Cross-line tag wrapper: the penultimate line is an OPENING XML-style tag
# like ``<answer>`` and the last line is just the bare answer letter.
# Surfaced in comedian-source rationales at seed=42.
_CROSS_LINE_TAG_OPEN_RE = re.compile(r"<\s*(?:answer|response|choice)\s*>\s*$", re.IGNORECASE)
# Rule 7: trailing parenthesized A/B/C/D ``(X)`` or ``(X).`` preceded by
# whitespace.  Discriminates from internal captures like ``(8A)`` by requiring
# the open-paren be the FIRST char of the ``(X)`` token (lookbehind ``\s\(``).
_RULE7_TRAILING_PAREN_LETTER_RE = re.compile(r"(?:^|(?<=\s))\([A-D]\)\.?\s*$")


_MAX_STRIP_ITERATIONS = 4


def strip_trailing_answer(text: str) -> tuple[str, int]:
    """Apply the 5-rule (+ rule 6) strip pipeline to ``text``, iteratively.

    Returns ``(stripped, rule_id)`` where ``rule_id`` is in ``{0, 1, 2, 3, 4,
    5, 6}`` (0 = no rule matched on the first pass, text unchanged). When
    multiple answer-delivering tails are stacked (common in #186 generic_cot
    where a model first writes ``Therefore the best answer is (A).`` and
    then ``\\n\\nAnswer: (A)``), the pipeline runs in a loop until either no
    rule fires or :data:`_MAX_STRIP_ITERATIONS` is reached. The reported
    ``rule_id`` is the rule that fired on the FIRST pass — subsequent passes
    are bookkeeping only.

    Plan §4 (lines 106-112) defines the contract for rules 1-5; iteration is
    an implementation detail that preserves the "first match wins" guarantee
    per pass while ensuring 100% post-strip non-letter termination across
    real #186 rationales.

    Examples:
        >>> strip_trailing_answer("Let me think.\\nAnswer: C")
        ('Let me think.', 1)
        >>> strip_trailing_answer("Therefore, the correct answer is (A).")
        ('Therefore, ', 3)
        >>> strip_trailing_answer("...vitamin D.")  # rule-5 gate prevents strip
        ('...vitamin D.', 0)
        >>> strip_trailing_answer("Answer B.")
        ('', 5)
        >>> strip_trailing_answer("The answer is:\\n\\n(B).")
        ('', 6)
    """
    if not text:
        return text, 0

    first_rule_id = 0
    current = text
    for _ in range(_MAX_STRIP_ITERATIONS):
        new_current, rid = _strip_trailing_answer_once(current)
        if rid == 0 or new_current == current:
            break
        if first_rule_id == 0:
            first_rule_id = rid
        current = new_current
    return current, first_rule_id


def _strip_trailing_answer_once(text: str) -> tuple[str, int]:
    """Single-pass strip — internal helper for :func:`strip_trailing_answer`."""
    if not text:
        return text, 0

    head, last_line = _split_final_line(text)
    if not last_line.strip():
        # Nothing to strip; return text untouched.
        return text, 0

    for rule_id, pattern in _STRIP_RULES:
        m = pattern.search(last_line)
        if m:
            # Strip from the match start to end-of-line.
            stripped_last = last_line[: m.start()]
            # Trim trailing whitespace, commas, etc. left over (e.g. "Therefore,
            # ").  Only strip whitespace — keep punctuation so rationales like
            # "Therefore," remain visible to the analyzer.
            stripped_last = stripped_last.rstrip()
            # Reattach the head; if the last line was ENTIRELY answer text
            # (stripped_last is empty), drop the trailing newline that
            # preceded it so the rationale doesn't end on a trailing newline.
            result = head + stripped_last if stripped_last else head.rstrip("\r\n")
            return result, rule_id

    # Post-canonical rules (8, 9): same shape as Rules 1-5 but with
    # specific multi-word keyword phrases ("go with", "that's") that the
    # canonical Rule 5 keyword list doesn't cover.  Surfaced by the smoke
    # in round-2 after the B1 fix exposed comedian-eval rows that the
    # round-1 smoke (single-persona iteration) had missed.
    for rule_id, pattern in _POST_CANONICAL_STRIP_RULES:
        m = pattern.search(last_line)
        if m:
            stripped_last = last_line[: m.start()].rstrip()
            result = head + stripped_last if stripped_last else head.rstrip("\r\n")
            return result, rule_id

    # Rule 6 (additive cross-line catch-all): last line is JUST a parenthesized
    # or bare letter, and the penultimate non-empty line ends with either an
    # answer-y keyword tail (``...the answer is:`` / ``...the correct answer
    # is``) OR an opening XML-style answer tag (``<answer>``).
    # This catches #186's ``the answer is\n\n(B).`` rationale shape AND the
    # comedian-source ``<answer>\nD`` shape that the in-line rules 1-5 cannot
    # see because they only inspect the final line.
    if _CROSS_LINE_BARE_LETTER_RE.match(last_line):
        # Find the penultimate non-empty line in `head`.
        head_lines = [line for line in head.splitlines() if line.strip()]
        if head_lines:
            penult = head_lines[-1]
            keyword_match = _CROSS_LINE_KEYWORD_TAIL_RE.search(penult)
            tag_match = _CROSS_LINE_TAG_OPEN_RE.search(penult)
            if keyword_match or tag_match:
                # Drop the last line entirely AND the keyword-tail / tag-open
                # clause from the penultimate line.
                if keyword_match:
                    penult_stripped = _CROSS_LINE_KEYWORD_TAIL_RE.sub("", penult).rstrip()
                else:
                    penult_stripped = _CROSS_LINE_TAG_OPEN_RE.sub("", penult).rstrip()
                # Reconstruct: lines before penult + cleaned penult.
                preceding = "\n".join(head_lines[:-1])
                if preceding and penult_stripped:
                    result = preceding + "\n" + penult_stripped
                elif preceding:
                    result = preceding
                else:
                    result = penult_stripped
                return result, 6

    # Rule 7 (last-resort): trailing parenthesized A/B/C/D — ``(A)``, ``(A).``
    # — preceded by whitespace, regardless of whether an answer keyword
    # appears nearby. In #186 ARC-C rationales, a trailing parenthesized
    # A/B/C/D at end of line is always the model handing in its answer
    # (false positives are vanishingly rare; e.g. ``Group 18 (8A).`` does
    # not match because the inner content is ``8A``, not ``A``). The
    # ``(?<=\s\()`` lookbehind ensures we're stripping a real ``(X)``
    # token and not an internal capture.
    if _RULE7_TRAILING_PAREN_LETTER_RE.search(last_line):
        stripped_last = _RULE7_TRAILING_PAREN_LETTER_RE.sub("", last_line).rstrip()
        result = head + stripped_last if stripped_last else head.rstrip("\r\n")
        return result, 7

    return text, 0


# ────────────────────────────────────────────────────────────────────────────
# Smoke-check helper: does a stripped rationale's final non-empty line end
# with a bare A/B/C/D after trimming whitespace, punctuation, and parens?
# Plan §4 line 116 — blocking assertion in --smoke-strip-coverage.
# ────────────────────────────────────────────────────────────────────────────

# A "bare answer letter" ending requires the letter be its own token —
# preceded by start-of-line, whitespace, or an opening paren. This excludes
# units like "21°C." and BPE-residue like "Ad" that share the same suffix.
_BARE_TRAILING_ANSWER_RE = re.compile(r"(?:^|(?<=[\s(]))\(?([A-D])\)?\.?\s*$")


def ends_with_bare_answer_letter(text: str) -> bool:
    """Return True if the stripped text ends with a bare A/B/C/D answer letter.

    Mirrors the assertion in ``--smoke-strip-coverage``: after stripping
    trailing whitespace, punctuation (``.``), and parentheses, does the final
    non-empty line end with one of ``{A, B, C, D}`` as an isolated token?
    Also catches the parenthesized variant ``(D)`` / ``(D).``.

    A correctly-stripped rationale tail should return False (e.g. ends in
    ``.``, ``ice``, ``78°F.``, ``21°C.``, or a sentence-final lowercase
    letter). False positives on units (``°C``) are avoided by requiring the
    A/B/C/D letter be preceded by whitespace, ``(``, or the start of the
    line — not by an arbitrary character.
    """
    if not text:
        return False
    _, last_line = _split_final_line(text)
    if not last_line.strip():
        return False
    return _BARE_TRAILING_ANSWER_RE.search(last_line) is not None


# ────────────────────────────────────────────────────────────────────────────
# Entropy from logprobs — analytical pass.
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EntropyResult:
    """Result of entropy computation from a single top-K logprob dict.

    Attributes:
        h_top20: Entropy in nats of the renormalized top-K distribution.
        h_abcd: Entropy in nats of the renormalized A/B/C/D distribution.
            ``None`` if no A/B/C/D token id appears in the top-K.
        top20_mass: Total probability mass captured by the returned top-K
            (one float in ``[0, 1]``). A4 diagnostic.
        abcd_total_mass_pre_renorm: Sum of ``exp(logprob)`` over top-K
            entries that decode to {A, B, C, D}, BEFORE letter-level
            logsumexp and renormalization. B6 diagnostic.
        restricted_missing: True iff ``h_abcd is None``.
        pred_argmax_token: The decoded-string form of the highest-logprob
            token in the dict.
        pred_argmax_letter: Letter in ``{A, B, C, D}`` if the predicted
            argmax token decodes (after strip) to a single A/B/C/D letter,
            else ``None``.
    """

    h_top20: float
    h_abcd: float | None
    top20_mass: float
    abcd_total_mass_pre_renorm: float
    restricted_missing: bool
    pred_argmax_token: str | None
    pred_argmax_letter: str | None


def _decoded_to_letter(decoded: str) -> str | None:
    """Map a vLLM-decoded token string to A/B/C/D, or None.

    vLLM returns ``decoded_token`` strings like ``"A"``, ``" B"``, ``" (C"``,
    ``" D)"``. We strip whitespace and parentheses, take the first character,
    and require it to be one of {A, B, C, D}.
    """
    if not decoded:
        return None
    cleaned = decoded.strip().lstrip("(").rstrip(")").strip()
    if not cleaned:
        return None
    first = cleaned[0]
    return first if first in {"A", "B", "C", "D"} else None


def entropy_from_logprobs(
    top_k_logprobs: dict[int, object],
    answer_token_ids: dict[str, set[int]],
) -> EntropyResult:
    """Compute H_top20, H_abcd, and mass diagnostics from a vLLM top-K dict.

    Args:
        top_k_logprobs: vLLM ``out.outputs[0].logprobs[0]`` — a dict keyed by
            token id, whose values expose ``.logprob`` (float) and
            ``.decoded_token`` (str). Tests pass simple dataclass-shaped
            stubs; we duck-type by reading those two attributes.
        answer_token_ids: Mapping ``{'A': {id1, id2, ...}, 'B': {...}, ...}``
            from :func:`answer_token_ids_for_tokenizer`. Used as an
            authoritative source: a token id in any of these sets is
            counted toward its letter even if the decoded string is
            ambiguous (e.g. multi-byte BPE artifacts).

    Returns:
        :class:`EntropyResult` — see field docs for semantics.

    Plan §4 lines 120-128 defines the contract.
    """
    if not top_k_logprobs:
        return EntropyResult(
            h_top20=float("nan"),
            h_abcd=None,
            top20_mass=0.0,
            abcd_total_mass_pre_renorm=0.0,
            restricted_missing=True,
            pred_argmax_token=None,
            pred_argmax_letter=None,
        )

    # Build a list of (token_id, logprob, decoded) tuples, sorted by logprob descending.
    items: list[tuple[int, float, str]] = []
    for tok_id, lp_obj in top_k_logprobs.items():
        logprob = float(getattr(lp_obj, "logprob", lp_obj))
        decoded = getattr(lp_obj, "decoded_token", "") or ""
        items.append((int(tok_id), logprob, decoded))
    items.sort(key=lambda t: -t[1])

    # H_top20: renormalize over returned top-K probabilities.
    logprobs_arr = [lp for _, lp, _ in items]
    max_lp = max(logprobs_arr)
    # Numerically stable softmax-on-truncated-logits → probabilities.
    exps = [math.exp(lp - max_lp) for lp in logprobs_arr]
    denom = sum(exps)
    probs = [e / denom for e in exps]
    h_top20 = -sum(p * math.log(p) for p in probs if p > 0.0)

    # top20_mass: sum of unnormalized probs (logprobs are log of true
    # next-token probability, so exp() gives the true probability mass).
    top20_mass = sum(math.exp(lp) for lp in logprobs_arr)

    # H_abcd: collect probability mass per letter.  Per plan §4 line 125-126:
    # if multiple token IDs decode to the same letter, logsumexp them; then
    # renormalize across the four letters.
    letter_to_logprobs: dict[str, list[float]] = {"A": [], "B": [], "C": [], "D": []}

    # Reverse-index answer_token_ids for fast lookup.
    id_to_letter: dict[int, str] = {}
    for letter, ids in answer_token_ids.items():
        for tid in ids:
            id_to_letter[tid] = letter

    for tok_id, lp, decoded in items:
        letter = id_to_letter.get(tok_id)
        if letter is None:
            # Fallback: try decoded-token mapping if the token id wasn't in
            # our pre-computed set (handles tokenizer drift).
            letter = _decoded_to_letter(decoded)
        if letter in letter_to_logprobs:
            letter_to_logprobs[letter].append(lp)

    abcd_total_mass_pre_renorm = sum(
        math.exp(lp) for lps in letter_to_logprobs.values() for lp in lps
    )

    # Logsumexp per letter, then renormalize across letters.
    per_letter_logprob: dict[str, float] = {}
    for letter, lps in letter_to_logprobs.items():
        if not lps:
            continue
        m = max(lps)
        per_letter_logprob[letter] = m + math.log(sum(math.exp(lp - m) for lp in lps))

    if not per_letter_logprob:
        h_abcd: float | None = None
        restricted_missing = True
    else:
        # Renormalize across the present letters.
        vals = list(per_letter_logprob.values())
        m = max(vals)
        denom_log = m + math.log(sum(math.exp(v - m) for v in vals))
        norm_probs = [math.exp(v - denom_log) for v in vals]
        h_abcd = -sum(p * math.log(p) for p in norm_probs if p > 0.0)
        restricted_missing = False

    # Predicted argmax: token with the highest logprob.
    argmax_tok_id, _, argmax_decoded = items[0]
    pred_argmax_letter = id_to_letter.get(argmax_tok_id) or _decoded_to_letter(argmax_decoded)

    return EntropyResult(
        h_top20=h_top20,
        h_abcd=h_abcd,
        top20_mass=top20_mass,
        abcd_total_mass_pre_renorm=abcd_total_mass_pre_renorm,
        restricted_missing=restricted_missing,
        pred_argmax_token=argmax_decoded,
        pred_argmax_letter=pred_argmax_letter,
    )


# ────────────────────────────────────────────────────────────────────────────
# Miller-Madow corrected empirical entropy (plan §4 line 182).
# ────────────────────────────────────────────────────────────────────────────


def miller_madow_entropy(
    counts: dict[str, int] | Counter,
    n_samples: int | None = None,
) -> tuple[float, float]:
    """Compute plug-in MLE entropy and Miller-Madow-corrected entropy in nats.

    ``H_MM = H_mle + (K_obs - 1) / (2 * N)`` where ``K_obs`` is the number of
    distinct bins with at least one observation and ``N`` is the total
    sample count over those bins.

    Args:
        counts: Mapping of category → observed count. Categories with zero
            counts are ignored. Only A/B/C/D are expected as keys for #355
            empirical pass, but any string keys are accepted.
        n_samples: Total number of letter-bearing samples (excludes
            ``nonletter`` rows). If ``None``, computed as ``sum(counts.values())``.

    Returns:
        ``(H_mle, H_MM)``. If ``n_samples == 0``, returns ``(nan, nan)``.

    Examples:
        >>> miller_madow_entropy({"A": 8, "B": 0, "C": 0, "D": 0})
        (-0.0, 0.0)
        >>> H_mle, H_MM = miller_madow_entropy({"A": 4, "B": 4})
        >>> round(H_mle, 4), round(H_MM, 4)
        (0.6931, 0.8182)
    """
    positive_counts = [c for c in counts.values() if c > 0]
    n = n_samples if n_samples is not None else sum(positive_counts)
    if n <= 0:
        return float("nan"), float("nan")

    k_obs = len(positive_counts)
    if k_obs == 0:
        return float("nan"), float("nan")

    h_mle = 0.0
    for c in positive_counts:
        p = c / n
        h_mle -= p * math.log(p)

    h_mm = h_mle + (k_obs - 1) / (2.0 * n)
    return h_mle, h_mm


# ────────────────────────────────────────────────────────────────────────────
# Answer-token id discovery (plan §4 line 126).
# ────────────────────────────────────────────────────────────────────────────


def answer_token_ids_for_tokenizer(tokenizer) -> dict[str, set[int]]:
    """Return ``{'A': {ids...}, 'B': {ids...}, ...}`` for a HF tokenizer.

    Per plan §4: "collect all token IDs emitted by tokenizer encodings for
    naked letters ``A,B,C,D`` and leading-space variants `` A, B, C, D``."

    Some Qwen/Qwen2.5 tokenizers also surface separate ids for ``(A``, ``(B``,
    etc. — those are NOT counted here because they correspond to multi-char
    decoded strings and the answer-position next-token is expected to be a
    bare letter (possibly with a leading space).
    """
    variants = ["A", "B", "C", "D", " A", " B", " C", " D"]
    result: dict[str, set[int]] = {"A": set(), "B": set(), "C": set(), "D": set()}
    for v in variants:
        # add_special_tokens=False so we don't get BOS/EOS pollution.
        ids: Iterable[int] = tokenizer.encode(v, add_special_tokens=False)
        ids_list = list(ids)
        # Take the FIRST token id only — the letter is one token at the
        # answer position in Qwen2.5-7B-Instruct's BPE.
        if not ids_list:
            continue
        letter = v.strip()[0]
        result[letter].add(ids_list[0])
    return result


# ────────────────────────────────────────────────────────────────────────────
# Prompt assembly for #355 (mirrors #186's b51dfbc9 scaffold).
# ────────────────────────────────────────────────────────────────────────────


def format_arc_user_turn(question: dict) -> str:
    """Format an ARC-C question + 4 choices into a single user-turn string.

    Mirrors b51dfbc9:src/explore_persona_space/eval/capability.py:567 exactly.
    """
    choices_text = "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )
    return f"{question['question']}\n\n{choices_text}"


# Historical scaffold strings from b51dfbc9:src/explore_persona_space/eval/prompting.py:109-143.
# Hard-coded here rather than imported because the current
# `src/explore_persona_space/eval/prompting.py` no longer carries them, and
# #355's primary measurement REQUIRES bit-exact reproduction of #186's
# template.
NO_COT_USER_SUFFIX = "\n\nThe correct answer is ("
GENERIC_COT_ASSISTANT_PREFIX = "Let me think step-by-step. "
GENERIC_COT_ANSWER_ANCHOR = "\nAnswer: "
PERSONA_COT_ASSISTANT_PREFIX = "<persona-thinking>\n"
PERSONA_COT_CLOSING_TAG = "\n</persona-thinking>"
PERSONA_COT_ANSWER_ANCHOR = "\nAnswer: "


def build_teacher_forced_prompt(
    tokenizer,
    persona_system_prompt: str,
    question: dict,
    cot_text: str,
    cot_style: str,
) -> str:
    """Assemble the teacher-forced prompt up to (and including) the answer marker.

    Mirrors b51dfbc9 ``_build_chat_prefix`` + ``_extract_logprobs_for_arm``
    branches. The next-token logprob position is the A/B/C/D letter
    immediately after the trailing answer marker.

    Args:
        tokenizer: HF tokenizer with ``apply_chat_template`` support.
        persona_system_prompt: System-prompt text (e.g.
            ``PERSONAS['librarian']`` or ``ASSISTANT_PROMPT``). Empty string
            falls back to no system message.
        question: ARC-C row dict — ``{question, choice_labels, choices,
            correct_answer}``.
        cot_text: For ``generic_cot``/``persona_cot``, the
            already-answer-stripped rationale. For ``no_cot``, ignored.
        cot_style: One of ``"no_cot"``, ``"generic_cot"``, ``"persona_cot"``.

    Returns:
        A fully rendered prompt string ready to pass to
        ``LLM.generate([prompt], SamplingParams(max_tokens=1, logprobs=20))``.

    Raises:
        ValueError: on unknown ``cot_style``.
    """
    user_content = format_arc_user_turn(question)

    if cot_style == "no_cot":
        user_with_anchor = f"{user_content}{NO_COT_USER_SUFFIX}"
        messages = []
        if persona_system_prompt:
            messages.append({"role": "system", "content": persona_system_prompt})
        messages.append({"role": "user", "content": user_with_anchor})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if cot_style == "generic_cot":
        assistant_prefix = GENERIC_COT_ASSISTANT_PREFIX
        closing_tag = ""
        answer_anchor = GENERIC_COT_ANSWER_ANCHOR
    elif cot_style == "persona_cot":
        assistant_prefix = PERSONA_COT_ASSISTANT_PREFIX
        closing_tag = PERSONA_COT_CLOSING_TAG
        answer_anchor = PERSONA_COT_ANSWER_ANCHOR
    else:
        raise ValueError(
            f"Unknown cot_style {cot_style!r}; expected no_cot|generic_cot|persona_cot"
        )

    messages = []
    if persona_system_prompt:
        messages.append({"role": "system", "content": persona_system_prompt})
    messages.append({"role": "user", "content": user_content})
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return base + assistant_prefix + cot_text + closing_tag + answer_anchor


# ────────────────────────────────────────────────────────────────────────────
# Empirical-pass first-letter parser.
# ────────────────────────────────────────────────────────────────────────────

_FIRST_LETTER_RE = re.compile(r"\(?\b([A-D])\b\)?")
_ANSWER_MARKER_RE = re.compile(r"answer\s*:\s*", re.IGNORECASE)


def parse_first_answer_letter(text: str) -> str | None:
    """Extract the first A/B/C/D letter from an empirical sample.

    Plan §4 line 180: "accept naked or parenthesized A/B/C/D; ignore
    generated reasoning words before the first letter only if they occur
    before an ``Answer:`` marker. If no letter appears in 8 tokens, record
    ``nonletter``."

    Strategy:
        1. Strip leading whitespace.
        2. If an ``Answer:`` marker is present, search after it; otherwise
           search the whole string.
        3. Return the first uppercase A/B/C/D character (matching
           ``\\(?[A-D]\\)?``), else ``None``.
    """
    if not text:
        return None
    s = text.lstrip()
    marker = _ANSWER_MARKER_RE.search(s)
    search_start = marker.end() if marker else 0
    m = _FIRST_LETTER_RE.search(s, search_start)
    return m.group(1) if m else None
