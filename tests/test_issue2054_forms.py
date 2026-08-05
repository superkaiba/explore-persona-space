"""Unit tests for the #2054 framing axis (`scripts/issue2054_forms.py`) and its
consumers (phase_b splice threading + the capture rig's form-aware pre-query
prefix locator).

Covers: question recovery from BOTH scaffold-row schemas (generator `question`
field / stripper `q_start`+`q_end` span incl. quote-pair stripping), the
chat / bare_text renders (round-trip offsets, template shape, pre-query
boundary), story-form delegation to the parent affix machinery (attrib_quoted
round-trip, bare_label, the `indirect` deterministic drop), the per-form
prefill specs, and the capture locator's recorded-field preference + per-form
legacy fallbacks. All fixtures are synthetic prose written for this test — no
real-corpus text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_capture as capture  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_phase_b as phase_b  # noqa: E402

S = sc.SLOT_SENTINEL

QUESTION = "Where does the river go when the dam closes?"
# Stripper-path scaffold: question is a quoted utterance inside the prose and
# the row carries its char span (q_start = opening-quote index, q_end =
# closing-quote index + 1 — the parse_story_turns convention).
_PRE = "Mira leaned over the rail. "
STRIP_SCAFFOLD = f'{_PRE}"{QUESTION}" she asked. {S} The wind picked up.'
STRIP_ROW = {
    "scaffold_id": "strip_0",
    "conv_id": "conv_strip_0",
    "character": "Helios",
    "scaffold_text": STRIP_SCAFFOLD,
    "q_start": len(_PRE),
    "q_end": len(_PRE) + len(QUESTION) + 2,  # span INCLUDES the quote pair
}
# Generator-path scaffold: verbatim `question` field, no span.
GEN_ROW = {
    "scaffold_id": "gen_0",
    "conv_id": "conv_gen_0",
    "character": "Helios",
    "scaffold_text": STRIP_SCAFFOLD,
    "question": QUESTION,
}
ANSWER = "It pools behind the old lock and seeps into the marsh."


# --------------------------------------------------------------------------
# question_for_row
# --------------------------------------------------------------------------
def test_question_from_gen_row_field():
    assert forms.question_for_row(GEN_ROW) == QUESTION


def test_question_from_strip_row_span_strips_quote_pair():
    assert forms.question_for_row(STRIP_ROW) == QUESTION


def test_question_from_curly_quote_span():
    pre = "Rain hit the roof. "
    scaffold = f"{pre}“{QUESTION}” she asked. {S}"
    row = {
        "scaffold_text": scaffold,
        "q_start": len(pre),
        "q_end": len(pre) + len(QUESTION) + 2,
    }
    assert forms.question_for_row(row) == QUESTION


def test_question_missing_returns_none():
    assert forms.question_for_row({"scaffold_text": STRIP_SCAFFOLD}) is None


# --------------------------------------------------------------------------
# splice_answer_form — template forms (the genuinely new renders)
# --------------------------------------------------------------------------
def test_chat_render_round_trip_and_prequery_boundary():
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "chat", "Helios")
    assert r.form == "chat"
    assert r.text == (
        f"<|im_start|>user\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n{ANSWER}<|im_end|>"
    )
    assert r.text[r.answer_start : r.answer_end] == ANSWER
    assert r.prefix_end_char == len(forms.CHAT_USER_HEADER)
    # prefix ends immediately BEFORE the user query (plan §6 pre-query pooling)
    assert r.text[r.prefix_end_char :].startswith(QUESTION)


def test_bare_text_render_round_trip_and_prequery_boundary():
    r = forms.splice_answer_form(GEN_ROW, ANSWER, "bare_text", "Helios")
    assert r.text == f"User: {QUESTION}\n\nAssistant: {ANSWER}"
    assert r.text[r.answer_start : r.answer_end] == ANSWER
    assert r.prefix_end_char == len(forms.BARE_USER_PREFIX)
    assert r.text[r.prefix_end_char :].startswith(QUESTION)


def test_chat_render_requires_question():
    with pytest.raises(ValueError, match="question"):
        forms.splice_answer_form({"scaffold_text": STRIP_SCAFFOLD}, ANSWER, "chat", "Helios")


def test_chat_render_refuses_template_marker_in_answer():
    with pytest.raises(ValueError, match="chat-template marker"):
        forms.splice_answer_form(GEN_ROW, "bad <|im_end|> answer", "chat", "Helios")


def test_empty_answer_refused_for_every_form():
    for form in ("chat", "bare_text", "attrib_quoted"):
        with pytest.raises(ValueError):
            forms.splice_answer_form(GEN_ROW, "", form, "Helios")


def test_unknown_form_refused():
    with pytest.raises(ValueError, match="unknown form"):
        forms.splice_answer_form(GEN_ROW, ANSWER, "sonnet_form", "Helios")


# --------------------------------------------------------------------------
# splice_answer_form — story forms (parent delegation)
# --------------------------------------------------------------------------
def test_attrib_quoted_delegates_to_parent_and_records_q_start():
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "attrib_quoted", "Helios")
    assert r.form == "attrib_quoted"
    assert r.text[r.answer_start : r.answer_end] == ANSWER
    assert 'Helios replied: "' in r.text
    # Stripper-path rows: pre-query boundary = the recorded opening-quote index.
    assert r.prefix_end_char == STRIP_ROW["q_start"]
    assert r.text[r.prefix_end_char] == '"'


def test_gen_row_story_prefix_end_locates_question_before_slot():
    r = forms.splice_answer_form(GEN_ROW, ANSWER, "attrib_quoted", "Helios")
    # Generator-path rows: locate the verbatim question, stepping back over
    # the opening quote the prose wraps it in.
    assert r.prefix_end_char == len(_PRE)
    assert r.text[r.prefix_end_char] == '"'


def test_bare_label_delegates_to_parent():
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "bare_label", "Helios")
    assert f"Helios: {ANSWER}" in r.text
    assert r.text[r.answer_start : r.answer_end] == ANSWER


def test_indirect_form_drops_by_refusal():
    with pytest.raises(NotImplementedError):
        forms.splice_answer_form(STRIP_ROW, ANSWER, "indirect", "Helios")


def test_story_form_prefix_none_when_question_unlocatable():
    row = {"scaffold_text": f"Prose with no quoted query. {S} Tail.", "character": "Helios"}
    r = forms.splice_answer_form(row, ANSWER, "attrib_quoted", "Helios")
    assert r.prefix_end_char is None  # recorded null, never coerced


# --------------------------------------------------------------------------
# render_prefill_form
# --------------------------------------------------------------------------
def test_chat_prefill_prefix_and_stop():
    spec = forms.render_prefill_form(GEN_ROW, "chat", "Helios")
    assert spec.prefix_text == f"<|im_start|>user\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n"
    assert tuple(spec.stop) == forms.CHAT_STOP


def test_bare_text_prefill_prefix_and_stop():
    spec = forms.render_prefill_form(STRIP_ROW, "bare_text", "Helios")
    assert spec.prefix_text == f"User: {QUESTION}\n\nAssistant: "
    assert tuple(spec.stop) == forms.BARE_STOP


def test_story_prefill_delegates_to_parent():
    spec = forms.render_prefill_form(STRIP_ROW, "attrib_quoted", "Helios")
    assert spec.prefix_text.endswith('Helios replied: "')
    assert tuple(spec.stop) == ('"',)


# --------------------------------------------------------------------------
# capture: form-aware pre-query prefix location
# --------------------------------------------------------------------------
def test_capture_prefers_recorded_prefix_end_char():
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "chat", "Helios")
    row = {"form": r.form, "prefix_end_char": r.prefix_end_char}
    pos, src = capture._prefix_end_char_for_row(row, r.text, r.answer_start)
    assert (pos, src) == (r.prefix_end_char, "recorded")


def test_capture_chat_fallback_reads_header():
    r = forms.splice_answer_form(GEN_ROW, ANSWER, "chat", "Helios")
    row = {"form": "chat"}  # legacy row: no recorded field
    pos, src = capture._prefix_end_char_for_row(row, r.text, r.answer_start)
    assert (pos, src) == (len(forms.CHAT_USER_HEADER), "form_header")


def test_capture_bare_text_fallback_reads_header():
    r = forms.splice_answer_form(GEN_ROW, ANSWER, "bare_text", "Helios")
    pos, src = capture._prefix_end_char_for_row({"form": "bare_text"}, r.text, r.answer_start)
    assert (pos, src) == (len(forms.BARE_USER_PREFIX), "form_header")


def test_capture_attrib_legacy_marker_fallback_unchanged():
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "attrib_quoted", "Helios")
    pos, src = capture._prefix_end_char_for_row({"form": "attrib_quoted"}, r.text, r.answer_start)
    assert src == "legacy_marker"
    # legacy convention: pre-ATTRIBUTION (right after the char name)
    assert r.text[pos:].startswith(' replied: "')


def test_capture_bare_label_fallback_locates_label():
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "bare_label", "Helios")
    row = {"form": "bare_label", "character": "Helios"}
    pos, src = capture._prefix_end_char_for_row(row, r.text, r.answer_start)
    assert src == "legacy_label"
    assert r.text[pos:].startswith(f"Helios: {ANSWER}")


def test_capture_unlocatable_returns_none():
    pos, src = capture._prefix_end_char_for_row({"form": "chat"}, "no header here", 5)
    assert (pos, src) == (None, "none")


def test_capture_rejects_bool_recorded_field():
    pos, src = capture._prefix_end_char_for_row(
        {"form": "chat", "prefix_end_char": True}, "no header here", 5
    )
    assert (pos, src) == (None, "none")


def test_capture_compute_positions_nonnull_prefix_for_chat_row():
    """A chat-framing row yields a NON-null v_P (the C3 100%-null-prefix fix)."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", use_fast=True)
    except Exception:
        pytest.skip("tokenizer unavailable offline")
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "chat", "Helios")
    row = {
        "conv_id": "conv_strip_0",
        "form": r.form,
        "final_text": r.text,
        "answer_start": r.answer_start,
        "answer_end": r.answer_end,
        "prefix_end_char": r.prefix_end_char,
    }
    pos = capture._compute_positions(tokenizer, row)
    assert pos is not None
    assert pos["v_P_pos"] is not None
    assert pos["prefix_src"] == "recorded"
    assert pos["answer_hi"] > pos["answer_lo"]


# --------------------------------------------------------------------------
# phase_b threading
# --------------------------------------------------------------------------
def test_phase_b_splice_one_chat_row_carries_form_and_prefix():
    out = phase_b._splice_one(STRIP_ROW, ANSWER, "conversation_paired_stories_assistant", "chat")
    assert out is not None
    assert out["form"] == "chat"
    assert out["prefix_end_char"] == len(forms.CHAT_USER_HEADER)
    assert out["final_text"][out["answer_start"] : out["answer_end"]] == ANSWER


def test_phase_b_splice_one_attrib_matches_parent_round_trip():
    out = phase_b._splice_one(STRIP_ROW, ANSWER, "char_helios", "attrib_quoted")
    assert out is not None
    assert out["form"] == "attrib_quoted"
    ref = sc.splice_answer(STRIP_SCAFFOLD, ANSWER, "attrib_quoted", "Helios")
    assert out["final_text"] == ref.text
    assert (out["answer_start"], out["answer_end"]) == (ref.answer_start, ref.answer_end)


def test_phase_b_splice_one_question_missing_chat_row_skips():
    # `character` present so the M3 fail-loud char resolution passes and the
    # test exercises its actual subject: a question-less chat row is a
    # counted SKIP (None), never a silent fallback to another form.
    row = {"scaffold_text": f"No quoted query here. {S}", "scaffold_id": "x", "character": "Helios"}
    assert phase_b._splice_one(row, ANSWER, "_flat", "chat") is None
