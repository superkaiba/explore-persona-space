"""Framing-axis renderers for task #2054 (plan §4 "Framings").

The lattice's central manipulated variable is the FRAMING — a splice-time
template parameter over ONE scaffold (plan §4 "Framings (splice-time template
parameter over ONE scaffold)"). The five plan framings map onto:

  1. ``chat``          — Qwen chat template over (Q, A); the narrative prose is
                         dropped (chat is structurally assistant-only, plan §4
                         Cells): ``<|im_start|>user\\n{Q}<|im_end|>\\n<|im_start|>assistant\\n{A}<|im_end|>``.
  2. ``bare_text``     — ``User: {Q}\\n\\nAssistant: {A}``; prose dropped.
  3. ``bare_label``    — narrative scaffold + ``{Name}: {A}`` boundary
                         (parent ``issue1345_scaffold_common`` form).
  4. ``attrib_quoted`` — narrative scaffold + ``{Name} replied: "{A}"``
                         (parent form; the round-1 hardcoded default).
  5. ``indirect``      — no deterministic faithful render; the parent
                         machinery raises NotImplementedError and the framing
                         DROPS rather than being faked (plan §4 framing 5).

``bare_paragraph`` (a parent BOUNDARY_FORMS member, the H2 boundary-form
family) passes through as a supported story form even though it is not one of
the 5 lattice framings — the dispatch decides which forms actually run.

Story forms delegate to the parent's ``splice_answer`` / ``render_prefill``
(reuse, never reimplement); ``chat`` / ``bare_text`` are rendered here (the
genuinely new pieces): the scaffold's question utterance is recovered —
``question`` field on generator-path rows, ``q_start``/``q_end`` char span on
stripper-path rows (spans index the scaffold text; the parent stripper
guarantees ``q_end <= attr_start == slot``) — and re-framed.

Every render records ``prefix_end_char``: the char index where the USER QUERY
begins in the rendered text, so the capture rig locates the prefix arm BY
CONSTRUCTION for every form (plan §6 pooling row: v_P = last-token pooling at
the PRE-QUERY position; plan §4 Design item 2). A row whose query cannot be
located records ``prefix_end_char = None`` (never coerced) and the capture
reports the null fraction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue1345_common as ic  # noqa: E402
import issue1345_scaffold_common as sc  # noqa: E402

# --- chat framing (plan §4 framing 1, template verbatim) --------------------
CHAT_USER_HEADER = "<|im_start|>user\n"
CHAT_ASSISTANT_HEADER = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_SUFFIX = "<|im_end|>"
# Literal special-marker text may not ride inside Q/A (it would tokenize into
# the template's special ids and corrupt the span semantics — refuse, per the
# parent's sentinel-in-answer refusal).
CHAT_SPECIAL_MARKERS = ("<|im_start|>", "<|im_end|>")
CHAT_STOP = ("<|im_end|>",)

# --- bare-text framing (plan §4 framing 2, template verbatim) ---------------
BARE_USER_PREFIX = "User: "
BARE_ASSISTANT_PREFIX = "\n\nAssistant: "
# On-policy continuation stops at the next turn marker (the deterministic
# analogue of the parent's per-form stops: attrib -> closing quote,
# bare_label -> end of line, bare_paragraph -> paragraph break).
BARE_STOP = ("\nUser:",)

# Form families. TEMPLATE_FORMS are rendered here; STORY_FORMS delegate to the
# parent affix machinery (`indirect` raises NotImplementedError there — the
# plan's deterministic drop).
TEMPLATE_FORMS = ("chat", "bare_text")
STORY_FORMS = sc.BOUNDARY_FORMS
FORMS = TEMPLATE_FORMS + STORY_FORMS

# --- 4-axis cell naming (C6) -------------------------------------------------
# The plan §4 lattice cell identity is (identity/variant, condition/phase,
# framing/form, model) — every output filename and downstream cell key joins
# ALL FOUR axes, or two runs differing only in condition/form OVERWRITE each
# other (the C6 collision: `--phase inserted` then `--phase on_policy` on one
# variant clobbered cell (b) with cell (d)). Axes join on a DOUBLE underscore;
# collisions are impossible BY CONSTRUCTION: condition + form come from closed
# registries and every free-text axis value (variant, model) is validated
# non-empty and separator-free. Single underscores stay legal INSIDE axis
# values (char_helios, bare_text, on_policy).
CONDITIONS = ("inserted", "on_policy", "cell_c")
CELL_KEY_SEP = "__"

_CONDITION_FILE_PREFIX = {
    "inserted": "spliced_inserted",
    "on_policy": "on_policy",
    "cell_c": "cell_c",
}


def _check_axis(name: str, value: str) -> None:
    """Fail loud on a cell-key axis value that would make the joined key ambiguous."""
    if not value or CELL_KEY_SEP in value:
        raise ValueError(
            f"cell-key axis {name}={value!r} must be non-empty and must not contain "
            f"{CELL_KEY_SEP!r} (the axis separator)"
        )


def base_character(variant: str) -> str:
    """Map an on-policy variant to its base character (``char_X_op[_base]`` ->
    ``char_X``); non-op variants pass through. Single source for the
    (character, model) comparison-group key (fits M1) and the ladder's §6
    pair-class predicates (M-R2-1)."""
    for tail in ("_op_base", "_op"):
        if variant.endswith(tail):
            return variant[: -len(tail)]
    return variant


def cell_key(variant: str, condition: str, form: str, model: str) -> str:
    """Canonical 4-axis cell key ``variant__condition__form__model``.

    The single naming source for capture .npz / diagnostics filenames and the
    fits/ladder cell keys; raises on unknown condition/form or a separator-
    bearing variant/model (collision-impossible by construction).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r} (expected one of {CONDITIONS})")
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r} (expected one of {FORMS})")
    for axis, value in (("variant", variant), ("model", model)):
        _check_axis(axis, value)
    return CELL_KEY_SEP.join((variant, condition, form, model))


def phase_output_name(condition: str, variant: str, form: str, *, mock: bool = False) -> str:
    """Form-aware per-variant output JSONL filename for the phase_b/c/d units.

    ``spliced_inserted_{variant}__{form}.jsonl`` (condition ``inserted``),
    ``on_policy_{variant}__{form}[.mock].jsonl``, ``cell_c_{variant}__{form}.jsonl``
    — two ``--form`` runs of one condition+variant land on DISTINCT files (C6).
    """
    prefix = _CONDITION_FILE_PREFIX.get(condition)
    if prefix is None:
        raise ValueError(f"unknown condition {condition!r} (expected one of {CONDITIONS})")
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r} (expected one of {FORMS})")
    _check_axis("variant", variant)
    if mock and condition != "on_policy":
        raise ValueError("mock outputs exist only for the on_policy condition")
    suffix = ".mock.jsonl" if mock else ".jsonl"
    return f"{prefix}_{variant}{CELL_KEY_SEP}{form}{suffix}"


@dataclass(frozen=True)
class FormRender:
    """A framed render: final text + exact answer span + pre-query boundary.

    Invariant (asserted at construction site): text[answer_start:answer_end]
    == the framed answer, byte-exact. ``prefix_end_char`` is the char index
    where the user query begins (None when not locatable — recorded, never
    coerced).
    """

    text: str
    answer_start: int
    answer_end: int
    form: str
    prefix_end_char: int | None


def question_for_row(row: dict) -> str | None:
    """Recover the scaffold's question utterance from either row schema.

    Generator-path rows (``issue1345_gen_scaffolds``) carry the verbatim
    ``question`` field; stripper-path rows (``issue1345_strip_scaffolds``)
    carry ``q_start``/``q_end`` — a char span into ``scaffold_text`` that
    INCLUDES the surrounding quote pair (``parse_story_turns``: q_start =
    opening-quote index, q_end = closing-quote index + 1). Returns the bare
    question text, or None when neither schema resolves.
    """
    q = row.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    scaffold = row.get("scaffold_text")
    q_start = row.get("q_start")
    q_end = row.get("q_end")
    if (
        isinstance(scaffold, str)
        and isinstance(q_start, int)
        and isinstance(q_end, int)
        and 0 <= q_start < q_end <= len(scaffold)
    ):
        span = scaffold[q_start:q_end]
        # Strip the matched quote pair the parser's span includes (parent
        # quote grammar: issue1345_common._OPEN_QUOTES / _CLOSE_FOR).
        if len(span) >= 2 and span[0] in ic._OPEN_QUOTES and span[-1] == ic._CLOSE_FOR.get(span[0]):
            span = span[1:-1]
        span = span.strip()
        return span or None
    return None


def story_prefix_end_char(row: dict, slot_idx: int, question: str | None) -> int | None:
    """Pre-query char boundary for a STORY-form render (plan §6: v_P reads the
    last token BEFORE the user query).

    Positions before the slot are unchanged by the splice, so scaffold-space
    indices are valid in the rendered text. Stripper-path rows use the
    recorded ``q_start`` (the question's opening-quote index — the parent
    capture convention, `issue1345_common` prefix slot). Generator-path rows
    locate the LAST occurrence of the verbatim question before the slot
    (single-question invariant, plan §4 req 6), stepping back over an opening
    quote when present. None when the query cannot be located.
    """
    q_start = row.get("q_start")
    if isinstance(q_start, int) and 0 <= q_start < slot_idx:
        return q_start
    scaffold = row.get("scaffold_text")
    if question and isinstance(scaffold, str):
        idx = scaffold.rfind(question, 0, slot_idx)
        if idx > 0 and scaffold[idx - 1] in ic._OPEN_QUOTES:
            return idx - 1
        if idx >= 0:
            return idx
    return None


def splice_answer_form(
    row: dict,
    answer: str,
    form: str,
    char_name: str,
    *,
    attrib_template: str | None = None,
) -> FormRender:
    """Render one framed row: scaffold row + answer -> FormRender.

    Story forms delegate to the parent's ``splice_answer`` (100% keep by
    construction; ``indirect`` raises NotImplementedError there). Template
    forms (``chat`` / ``bare_text``) require the row's question and raise
    ValueError when it cannot be recovered — a counted skip at the caller,
    never a silent fallback to another form.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r} (expected one of {FORMS})")
    if not answer:
        raise ValueError("answer must be non-empty (a zero-width span is unusable downstream)")

    if form in TEMPLATE_FORMS:
        question = question_for_row(row)
        if not question:
            raise ValueError(
                f"form {form!r} requires the scaffold row's question "
                "(field 'question', or q_start/q_end span); none found"
            )
        if form == "chat":
            for marker in CHAT_SPECIAL_MARKERS:
                if marker in answer or marker in question:
                    raise ValueError(f"chat-template marker {marker!r} inside question/answer text")
            prefix = CHAT_USER_HEADER + question + CHAT_ASSISTANT_HEADER
            text = prefix + answer + CHAT_SUFFIX
            prefix_end: int | None = len(CHAT_USER_HEADER)
        else:  # bare_text
            prefix = BARE_USER_PREFIX + question + BARE_ASSISTANT_PREFIX
            text = prefix + answer
            prefix_end = len(BARE_USER_PREFIX)
        answer_start = len(prefix)
        answer_end = answer_start + len(answer)
        assert text[answer_start:answer_end] == answer, (
            "form render offset invariant violated",
            form,
            answer_start,
            answer_end,
        )
        return FormRender(
            text=text,
            answer_start=answer_start,
            answer_end=answer_end,
            form=form,
            prefix_end_char=prefix_end,
        )

    scaffold = row.get("scaffold_text")
    if not isinstance(scaffold, str):
        raise ValueError(f"story form {form!r} requires row['scaffold_text']")
    slot_idx = scaffold.index(sc.SLOT_SENTINEL)  # ValueError when absent (counted skip)
    result = sc.splice_answer(
        scaffold,
        answer,
        form,
        char_name,
        attrib_template=attrib_template if form == "attrib_quoted" else None,
    )
    return FormRender(
        text=result.text,
        answer_start=result.answer_start,
        answer_end=result.answer_end,
        form=result.form,
        prefix_end_char=story_prefix_end_char(row, slot_idx, question_for_row(row)),
    )


def render_prefill_form(row: dict, form: str, char_name: str) -> sc.PrefillSpec:
    """Per-form generation prefix + stop strings (phase_c on-policy arm).

    Story forms delegate to the parent's ``render_prefill`` (attrib -> closing
    quote, bare_label -> end of line, bare_paragraph -> paragraph break;
    ``indirect`` raises). Template forms stop at the chat turn terminator
    (``<|im_end|>``) / the next bare turn marker (``\\nUser:``); a stop that
    never fires ends at the generation cap and is flagged by the caller via
    ``finish_reason`` (the standing cap-hit report).
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r} (expected one of {FORMS})")
    if form in TEMPLATE_FORMS:
        question = question_for_row(row)
        if not question:
            raise ValueError(
                f"form {form!r} requires the scaffold row's question "
                "(field 'question', or q_start/q_end span); none found"
            )
        if form == "chat":
            return sc.PrefillSpec(
                prefix_text=CHAT_USER_HEADER + question + CHAT_ASSISTANT_HEADER,
                stop=CHAT_STOP,
                form=form,
            )
        return sc.PrefillSpec(
            prefix_text=BARE_USER_PREFIX + question + BARE_ASSISTANT_PREFIX,
            stop=BARE_STOP,
            form=form,
        )
    scaffold = row.get("scaffold_text")
    if not isinstance(scaffold, str):
        raise ValueError(f"story form {form!r} requires row['scaffold_text']")
    return sc.render_prefill(scaffold, form, char_name)
