"""Issue #1345 scaffold-and-splice core (decoupled story generation).

The one-shot "write the story AND embed the answer" arms lose 20-41% of rows
to span-locatability gates (verbatim embedding for the inserted arm,
attribution-parse + structural-extractability judging for the on-policy arm).
This module decouples the two: a SCAFFOLD is a narrative scene containing
exactly ONE question utterance and ONE machine-locatable answer SLOT
(``SLOT_SENTINEL``) with no answer text; the answer is then either spliced in
deterministically (Phase B — exact char offsets by construction, no gate) or
generated as a prefill continuation from the scaffold truncated at the slot
(Phase C — the answer span is everything generated, bounded by a per-form stop
string).

Three shared pieces live here:
  * ``splice_answer``    — deterministic scaffold + answer -> final text with
                           the answer span's exact char offsets (Phase B, and
                           Phase C's final-text rendering).
  * ``render_prefill``   — scaffold truncated at the slot + the boundary
                           form's opening, plus the form's stop strings
                           (Phase C generation prefix; the story-slot shape of
                           ``issue1345_onpolicy_answers_gen`` is the
                           precedent: raw text continuation, stop at the
                           closing quote for the attributed form).
  * ``strip_story``      — recover a scaffold from an EXISTING generated
                           story by removing the FIRST answer utterance
                           (attribution + quoted answer) and inserting the
                           sentinel. Span logic reuses
                           ``issue1345_common.parse_story_turns`` (the same
                           parser the extraction path uses) — never a
                           reimplementation.

Character-name parameterization: ``issue1345_common.ANSWER_ATTRIB_RE``
compiles at import time from ``EPM_STORY_CHARACTER_NAME`` (default "ARIA"),
and ``parse_story_turns``'s SOURCE is fingerprinted by the parent gen bundles
(``issue1345_gen_stories.py`` hashes ``inspect.getsource``), so neither may be
edited or re-imported per character. ``attrib_re_for`` derives a per-name
pattern from the compiled pattern TEXT (name swap only — zero drift), and
``parse_story_turns_for`` swaps the module global for the duration of one
parse (restore-always).

Pure-CPU module: no torch/vLLM imports; the tokenizer argument of
``token_span_ok`` is duck-typed (HF fast tokenizer).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue1345_common as c  # noqa: E402

# ---------------------------------------------------------------------------
# The answer slot sentinel. Machine-locatable, replaced by the renderer; the
# exactly-one-occurrence invariant is asserted at every consumption site. The
# token never occurs in natural prose (angle-bracket triple), and its own
# tokenization is irrelevant — every downstream read happens on the SPLICED
# text, after the sentinel is gone.
# ---------------------------------------------------------------------------
SLOT_SENTINEL = "<<<ANSWER>>>"

# The four boundary forms of the splice render (template parameter over the
# SAME scaffold, so boundary-form comparisons are paired by construction).
BOUNDARY_FORMS = ("attrib_quoted", "bare_label", "bare_paragraph", "indirect")

# {answer} is a literal placeholder token, split on — NEVER str.format (prose
# may legitimately contain braces).
_ANSWER_PLACEHOLDER = "{answer}"


def default_attrib_template(char_name: str) -> str:
    """The canonical attributed-quote template: ``<Name> replied: "<ANSWER>"``."""
    return f'{char_name} replied: "' + _ANSWER_PLACEHOLDER + '"'


def count_sentinels(text: str) -> int:
    """Occurrences of SLOT_SENTINEL in ``text`` (the exactly-one invariant)."""
    return text.count(SLOT_SENTINEL)


def _require_one_sentinel(scaffold_text: str) -> int:
    n = count_sentinels(scaffold_text)
    if n != 1:
        raise ValueError(f"scaffold must contain exactly one {SLOT_SENTINEL!r}, found {n}")
    return scaffold_text.index(SLOT_SENTINEL)


def _paragraph_padding(scaffold_text: str, slot_idx: int) -> tuple[str, str]:
    """(left, right) padding that isolates the slot as its own paragraph.

    Deterministic: pads only the newlines the neighborhood is missing, so a
    sentinel already standing as its own paragraph gets no extra padding.
    """
    head = scaffold_text[:slot_idx]
    tail = scaffold_text[slot_idx + len(SLOT_SENTINEL) :]
    if not head or head.endswith("\n\n"):
        left = ""
    elif head.endswith("\n"):
        left = "\n"
    else:
        left = "\n\n"
    if not tail or tail.startswith("\n\n"):
        right = ""
    elif tail.startswith("\n"):
        right = "\n"
    else:
        right = "\n\n"
    return left, right


def _form_affixes(
    scaffold_text: str,
    slot_idx: int,
    form: str,
    char_name: str,
    attrib_template: str | None,
) -> tuple[str, str]:
    """(prefix, suffix) around the answer for one boundary form.

    ``indirect`` (reported speech) has NO deterministic faithful render — it
    requires recasting the answer's grammatical person/tense, a generation
    task, not a string operation — so it raises NotImplementedError rather
    than faking it (spec decision, 2026-08-03).
    """
    if form == "attrib_quoted":
        template = attrib_template or default_attrib_template(char_name)
        if template.count(_ANSWER_PLACEHOLDER) != 1:
            raise ValueError(
                f"attrib_template must contain exactly one {_ANSWER_PLACEHOLDER!r}: {template!r}"
            )
        prefix, suffix = template.split(_ANSWER_PLACEHOLDER)
        return prefix, suffix
    if form == "bare_label":
        return f"{char_name}: ", ""
    if form == "bare_paragraph":
        return _paragraph_padding(scaffold_text, slot_idx)
    if form == "indirect":
        raise NotImplementedError(
            "boundary form 'indirect' (reported speech) has no deterministic faithful "
            "render: recasting the answer into indirect speech changes its text (person/"
            "tense), which a string splice cannot do — generate indirect renders with a "
            "model pass, or drop the form"
        )
    raise ValueError(f"unknown boundary form {form!r} (expected one of {BOUNDARY_FORMS})")


@dataclass(frozen=True)
class SpliceResult:
    """Final text + the answer span's exact char offsets.

    Invariant (asserted at construction site): text[answer_start:answer_end]
    == the spliced answer, byte-exact.
    """

    text: str
    answer_start: int
    answer_end: int
    form: str


def splice_answer(
    scaffold_text: str,
    answer: str,
    form: str,
    char_name: str,
    *,
    attrib_template: str | None = None,
) -> SpliceResult:
    """Deterministic scaffold + answer -> final text with exact answer offsets.

    100% keep by construction: no judge, no verbatim matcher — the span is
    known because we placed it. ``attrib_template`` (attrib_quoted only)
    overrides the canonical template; ``strip_story`` records the ORIGINAL
    attribution as a template so strip-then-splice round-trips byte-exact.
    """
    if not answer:
        raise ValueError("answer must be non-empty (a zero-width span is unusable downstream)")
    if SLOT_SENTINEL in answer:
        raise ValueError(f"answer must not contain the slot sentinel {SLOT_SENTINEL!r}")
    slot_idx = _require_one_sentinel(scaffold_text)
    prefix, suffix = _form_affixes(scaffold_text, slot_idx, form, char_name, attrib_template)
    text = (
        scaffold_text[:slot_idx]
        + prefix
        + answer
        + suffix
        + scaffold_text[slot_idx + len(SLOT_SENTINEL) :]
    )
    answer_start = slot_idx + len(prefix)
    answer_end = answer_start + len(answer)
    assert text[answer_start:answer_end] == answer, (
        "splice offset invariant violated",
        answer_start,
        answer_end,
    )
    return SpliceResult(text=text, answer_start=answer_start, answer_end=answer_end, form=form)


@dataclass(frozen=True)
class PrefillSpec:
    """Generation prefix + stop strings for the prefill-continuation arm.

    The answer span is BY CONSTRUCTION everything the model generates from
    ``prefix_text`` up to (excluding) the first stop-string hit (vLLM default
    ``include_stop_str_in_output=False``); a cap-hit row ends unterminated and
    is flagged by the caller (finish_reason == "length").
    """

    prefix_text: str
    stop: tuple[str, ...]
    form: str


def render_prefill(scaffold_text: str, form: str, char_name: str) -> PrefillSpec:
    """Scaffold truncated at the slot + the form's opening, as a raw prefix.

    Stop conditions (documented contract):
      attrib_quoted  -> stop at the closing double quote (the V1 answer
                        convention, same as the story_slot shape in
                        issue1345_onpolicy_answers_gen).
      bare_label     -> stop at the end of the line (one utterance line, the
                        issue1310 PREFILL_STOP convention).
      bare_paragraph -> stop at the next paragraph break (one paragraph).
    Everything AFTER the slot in the scaffold is dropped (the model continues
    the story from the slot; the post-slot narration is Phase B material).
    """
    slot_idx = _require_one_sentinel(scaffold_text)
    head = scaffold_text[:slot_idx]
    if form == "attrib_quoted":
        return PrefillSpec(head + f'{char_name} replied: "', ('"',), form)
    if form == "bare_label":
        return PrefillSpec(head + f"{char_name}: ", ("\n",), form)
    if form == "bare_paragraph":
        left, _ = _paragraph_padding(scaffold_text, slot_idx)
        return PrefillSpec(head + left, ("\n\n",), form)
    if form == "indirect":
        raise NotImplementedError(
            "boundary form 'indirect' has no deterministic prefill opening — see "
            "_form_affixes for the rationale"
        )
    raise ValueError(f"unknown boundary form {form!r} (expected one of {BOUNDARY_FORMS})")


# ---------------------------------------------------------------------------
# Per-character attribution regex + turn parsing (reuse, never re-implement)
# ---------------------------------------------------------------------------
def attrib_re_for(char_name: str) -> re.Pattern[str]:
    """``c.ANSWER_ATTRIB_RE`` with the character name swapped.

    Derived from the COMPILED pattern text (single name-token substitution),
    so the pattern shape can never drift from the parent's; the name grammar
    matches the ``EPM_STORY_CHARACTER_NAME`` contract in issue1345_common.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", char_name):
        raise ValueError(f"char_name {char_name!r} must match [A-Za-z0-9_]+")
    base_name = re.escape(c.STORY_CHARACTER_NAME)
    pattern = c.ANSWER_ATTRIB_RE.pattern
    assert pattern.count(base_name) == 1, (
        "ANSWER_ATTRIB_RE pattern no longer carries the character name exactly once — "
        "attrib_re_for needs updating"
    )
    # The NAME token accepts the GIVEN spelling OR its ALL-CAPS form; the rest
    # of the pattern keeps the parent's exact semantics.
    #
    # WHY: the parent's own character name is ALL-CAPS ("ARIA") and generators
    # echo that convention inconsistently. #2054's char_helios stories label the
    # speaker "HELIOS replied:" while char_wren/dana/vex came out title-case, so
    # a case-SENSITIVE \bHelios\b found ZERO turns and strip_story rejected
    # 2,175/2,187 rows as `no_parsed_turns` (vs 4/2,159 for wren). Measured on
    # the real parent files, this alternation recovers char_helios 12 -> 2,179
    # of 2,187 and leaves wren/dana/vex/assistant byte-identical.
    #
    # NOT a blanket `(?i:...)`: several lattice names are ordinary English words
    # ("Vex", "Wren"), and a fully case-insensitive token would attribute a
    # narrator clause ("...to vex her, he said, \"...\"") to the character. The
    # two-form alternation covers every casing convention actually observed
    # while keeping the lowercase word non-matching.
    #
    # `(?:...)` is NON-capturing, so group numbering — and the caller's
    # `m.end(1)` opening-quote re-alignment in strip_story — are unaffected.
    esc = re.escape(char_name)
    esc_upper = re.escape(char_name.upper())
    name_alt = esc if esc_upper == esc else f"(?:{esc}|{esc_upper})"
    return re.compile(pattern.replace(base_name, name_alt))


def parse_story_turns_for(text: str, char_name: str) -> list[dict]:
    """``c.parse_story_turns`` under a per-call character name.

    The parser reads the module-global ``ANSWER_ATTRIB_RE`` at call time (its
    source is bundle-fingerprinted upstream, so it cannot grow a parameter);
    we swap the global for the duration of ONE parse and always restore. The
    default name is a pure pass-through.
    """
    if char_name == c.STORY_CHARACTER_NAME:
        return c.parse_story_turns(text)
    prior = c.ANSWER_ATTRIB_RE
    c.ANSWER_ATTRIB_RE = attrib_re_for(char_name)
    try:
        return c.parse_story_turns(text)
    finally:
        c.ANSWER_ATTRIB_RE = prior


# ---------------------------------------------------------------------------
# Scaffold stripper (recover scaffolds from already-generated stories)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StripResult:
    """A scaffold recovered from an existing story.

    ``attrib_template`` carries the ORIGINAL attribution clause verbatim
    (prefix incl. opening quote + ``{answer}`` + closing quote), so
    ``splice_answer(scaffold_text, answer, "attrib_quoted", name,
    attrib_template=...)`` reproduces the original story byte-exact.
    """

    scaffold_text: str
    answer: str
    attrib_template: str
    n_parsed_turns: int
    q_start: int
    q_end: int


def strip_story(story_text: str, char_name: str) -> tuple[StripResult | None, str]:
    """Strip the FIRST answer utterance out of a story -> (result, reason).

    Returns (None, reason) on rows that cannot become scaffolds:
      no_parsed_turns    — the attribution parser found no Q->A turn;
      sentinel_collision — the story already contains the sentinel literal;
      attrib_realign     — the parsed turn could not be re-aligned to its
                           regex match (parser/regex drift — should not
                           happen; counted fail-loud material).
    ONE exchange per scaffold (the standing first-message-per-character
    constraint): only the FIRST parsed turn is stripped; later turns stay in
    the prose tail and ``n_parsed_turns`` records the count so consumers can
    filter (--require-single-turn in the CLI).
    """
    if SLOT_SENTINEL in story_text:
        return None, "sentinel_collision"
    turns = parse_story_turns_for(story_text, char_name)
    if not turns:
        return None, "no_parsed_turns"
    turn = turns[0]
    a_start, a_end = turn["a_start"], turn["a_end"]
    # Re-align the turn to its regex match to recover the attribution START
    # (parse_story_turns returns marker_end but not match.start()).
    attr_start = None
    for m in attrib_re_for(char_name).finditer(story_text):
        if m.end(1) == a_start:  # group 1 is the opening quote; span starts after it
            attr_start = m.start()
            break
    if attr_start is None:
        return None, "attrib_realign"
    # Segment replaced by the sentinel: attribution start .. closing quote
    # (a_end is the closing-quote index; segment end is exclusive of nothing —
    # the quote itself is part of the utterance clause).
    seg_end = a_end + 1
    answer = story_text[a_start:a_end]
    if _ANSWER_PLACEHOLDER in answer or not answer:
        return None, "attrib_realign"
    template = story_text[attr_start:a_start] + _ANSWER_PLACEHOLDER + story_text[a_end:seg_end]
    scaffold = story_text[:attr_start] + SLOT_SENTINEL + story_text[seg_end:]
    result = StripResult(
        scaffold_text=scaffold,
        answer=answer,
        attrib_template=template,
        n_parsed_turns=len(turns),
        q_start=turn["q_start"],
        q_end=turn["q_end"],
    )
    # Round-trip invariant, checked at strip time (deterministic; a violation
    # is a stripper bug, never a data property — fail loud).
    spliced = splice_answer(scaffold, answer, "attrib_quoted", char_name, attrib_template=template)
    assert spliced.text == story_text, "strip-then-splice failed to reproduce the original"
    assert spliced.text[spliced.answer_start : spliced.answer_end] == answer
    return result, "ok"


# ---------------------------------------------------------------------------
# GEN-time token-span validation (the #825 BPE zero-width-span guard)
# ---------------------------------------------------------------------------
def token_span_ok(text: str, a_start: int, a_end: int, tokenizer) -> bool:
    """True iff the char span owns >=1 fully-contained, contiguous token.

    The same containment rule ``c.render_story_turn`` enforces at extraction
    (a span whose text BPE-merges entirely into its delimiters is zero-width
    and unusable): validating at GENERATION time keeps degenerate rows out of
    the scaffold pool instead of crashing the capture consumer.
    """
    offs = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    a_tokens = [t for t, (a, b) in enumerate(offs) if a >= a_start and b <= a_end and b > a]
    return bool(a_tokens) and a_tokens[-1] + 1 - a_tokens[0] == len(a_tokens)
