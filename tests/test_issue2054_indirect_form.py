"""Byte-identity pins + `indirect` (framing 5, on-policy) tests for #2054.

Two jobs:

1. **Byte-identity across the framing-5 change** (epm:progress v261 hard
   requirement): the parent's banked 56-cell numbers depend on the renders of
   `chat`, `bare_text`, `bare_label`, `attrib_quoted` (and the never-run
   `bare_paragraph`), so GOLDEN_RENDERS pins `splice_answer_form` +
   `render_prefill_form` output for every existing form, captured VERBATIM
   from the pre-change code (2026-08-12, commit before the `indirect`
   branch landed) and asserted byte-identical after. A diff in any of these
   is a round bug, never an acceptable side effect.

2. **The additive `indirect` branch** (indirect reported speech, ON-POLICY
   ONLY): prefill opener + paragraph-break stop, the
   `indirect_continuation=True` splice (span byte-exact, pre-query boundary
   recorded), and the standing refusals — the verbatim/inserted splice keeps
   raising NotImplementedError with and without the 2054 wrapper.

Fixtures are the same synthetic prose as tests/test_issue2054_forms.py — the
goldens were generated FROM these exact fixtures, so do not edit them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_phase_c as phase_c  # noqa: E402

S = sc.SLOT_SENTINEL

QUESTION = "Where does the river go when the dam closes?"
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
GEN_ROW = {
    "scaffold_id": "gen_0",
    "conv_id": "conv_gen_0",
    "character": "Helios",
    "scaffold_text": STRIP_SCAFFOLD,
    "question": QUESTION,
}
ANSWER = "It pools behind the old lock and seeps into the marsh."
# A continuation as the model would generate it from the indirect prefill —
# already in the narrator's reported voice (no quotes, third person, past).
REPORTED_ANSWER = " the river pooled behind the old lock and seeped into the marsh."

_ROWS = {"strip": STRIP_ROW, "gen": GEN_ROW}

# Captured from the PRE-change code (see module docstring). NEVER regenerate
# from current code — that would turn the byte-identity pin into a tautology.
GOLDEN_RENDERS: dict = json.loads(r"""{
  "prefill__attrib_quoted__gen": {
    "form": "attrib_quoted",
    "prefix_text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios replied: \"",
    "stop": [
      "\""
    ]
  },
  "prefill__attrib_quoted__strip": {
    "form": "attrib_quoted",
    "prefix_text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios replied: \"",
    "stop": [
      "\""
    ]
  },
  "prefill__bare_label__gen": {
    "form": "bare_label",
    "prefix_text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios: ",
    "stop": [
      "\n"
    ]
  },
  "prefill__bare_label__strip": {
    "form": "bare_label",
    "prefix_text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios: ",
    "stop": [
      "\n"
    ]
  },
  "prefill__bare_paragraph__gen": {
    "form": "bare_paragraph",
    "prefix_text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. \n\n",
    "stop": [
      "\n\n"
    ]
  },
  "prefill__bare_paragraph__strip": {
    "form": "bare_paragraph",
    "prefix_text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. \n\n",
    "stop": [
      "\n\n"
    ]
  },
  "prefill__bare_text__gen": {
    "form": "bare_text",
    "prefix_text": "User: Where does the river go when the dam closes?\n\nAssistant: ",
    "stop": [
      "\nUser:"
    ]
  },
  "prefill__bare_text__strip": {
    "form": "bare_text",
    "prefix_text": "User: Where does the river go when the dam closes?\n\nAssistant: ",
    "stop": [
      "\nUser:"
    ]
  },
  "prefill__chat__gen": {
    "form": "chat",
    "prefix_text": "<|im_start|>user\nWhere does the river go when the dam closes?<|im_end|>\n<|im_start|>assistant\n",
    "stop": [
      "<|im_end|>"
    ]
  },
  "prefill__chat__strip": {
    "form": "chat",
    "prefix_text": "<|im_start|>user\nWhere does the river go when the dam closes?<|im_end|>\n<|im_start|>assistant\n",
    "stop": [
      "<|im_end|>"
    ]
  },
  "splice__attrib_quoted__custom_template": {
    "answer_end": 163,
    "answer_start": 109,
    "form": "attrib_quoted",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios replied softly: \"It pools behind the old lock and seeps into the marsh.\" The wind picked up."
  },
  "splice__attrib_quoted__gen": {
    "answer_end": 156,
    "answer_start": 102,
    "form": "attrib_quoted",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios replied: \"It pools behind the old lock and seeps into the marsh.\" The wind picked up."
  },
  "splice__attrib_quoted__strip": {
    "answer_end": 156,
    "answer_start": 102,
    "form": "attrib_quoted",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios replied: \"It pools behind the old lock and seeps into the marsh.\" The wind picked up."
  },
  "splice__bare_label__gen": {
    "answer_end": 147,
    "answer_start": 93,
    "form": "bare_label",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios: It pools behind the old lock and seeps into the marsh. The wind picked up."
  },
  "splice__bare_label__strip": {
    "answer_end": 147,
    "answer_start": 93,
    "form": "bare_label",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. Helios: It pools behind the old lock and seeps into the marsh. The wind picked up."
  },
  "splice__bare_paragraph__gen": {
    "answer_end": 141,
    "answer_start": 87,
    "form": "bare_paragraph",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. \n\nIt pools behind the old lock and seeps into the marsh.\n\n The wind picked up."
  },
  "splice__bare_paragraph__strip": {
    "answer_end": 141,
    "answer_start": 87,
    "form": "bare_paragraph",
    "prefix_end_char": 27,
    "text": "Mira leaned over the rail. \"Where does the river go when the dam closes?\" she asked. \n\nIt pools behind the old lock and seeps into the marsh.\n\n The wind picked up."
  },
  "splice__bare_text__gen": {
    "answer_end": 117,
    "answer_start": 63,
    "form": "bare_text",
    "prefix_end_char": 6,
    "text": "User: Where does the river go when the dam closes?\n\nAssistant: It pools behind the old lock and seeps into the marsh."
  },
  "splice__bare_text__strip": {
    "answer_end": 117,
    "answer_start": 63,
    "form": "bare_text",
    "prefix_end_char": 6,
    "text": "User: Where does the river go when the dam closes?\n\nAssistant: It pools behind the old lock and seeps into the marsh."
  },
  "splice__chat__gen": {
    "answer_end": 148,
    "answer_start": 94,
    "form": "chat",
    "prefix_end_char": 17,
    "text": "<|im_start|>user\nWhere does the river go when the dam closes?<|im_end|>\n<|im_start|>assistant\nIt pools behind the old lock and seeps into the marsh.<|im_end|>"
  },
  "splice__chat__strip": {
    "answer_end": 148,
    "answer_start": 94,
    "form": "chat",
    "prefix_end_char": 17,
    "text": "<|im_start|>user\nWhere does the river go when the dam closes?<|im_end|>\n<|im_start|>assistant\nIt pools behind the old lock and seeps into the marsh.<|im_end|>"
  }
}""")


# --------------------------------------------------------------------------
# 1. Byte-identity pins for every EXISTING form (v261 hard requirement)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "form", ["chat", "bare_text", "bare_label", "attrib_quoted", "bare_paragraph"]
)
@pytest.mark.parametrize("row_name", ["strip", "gen"])
def test_existing_form_splice_byte_identical(form, row_name):
    g = GOLDEN_RENDERS[f"splice__{form}__{row_name}"]
    r = forms.splice_answer_form(_ROWS[row_name], ANSWER, form, "Helios")
    assert r.text == g["text"]
    assert r.answer_start == g["answer_start"]
    assert r.answer_end == g["answer_end"]
    assert r.form == g["form"]
    assert r.prefix_end_char == g["prefix_end_char"]


@pytest.mark.parametrize(
    "form", ["chat", "bare_text", "bare_label", "attrib_quoted", "bare_paragraph"]
)
@pytest.mark.parametrize("row_name", ["strip", "gen"])
def test_existing_form_prefill_byte_identical(form, row_name):
    g = GOLDEN_RENDERS[f"prefill__{form}__{row_name}"]
    p = forms.render_prefill_form(_ROWS[row_name], form, "Helios")
    assert p.prefix_text == g["prefix_text"]
    assert list(p.stop) == g["stop"]
    assert p.form == g["form"]


def test_attrib_quoted_custom_template_byte_identical():
    g = GOLDEN_RENDERS["splice__attrib_quoted__custom_template"]
    tmpl = 'Helios replied softly: "{answer}"'
    r = forms.splice_answer_form(STRIP_ROW, ANSWER, "attrib_quoted", "Helios", attrib_template=tmpl)
    assert r.text == g["text"]
    assert (r.answer_start, r.answer_end, r.prefix_end_char) == (
        g["answer_start"],
        g["answer_end"],
        g["prefix_end_char"],
    )


def test_indirect_continuation_flag_inert_for_existing_forms():
    """The opt-in must not perturb any non-indirect render (flag is inert)."""
    for form in ("bare_label", "attrib_quoted", "bare_paragraph", "chat", "bare_text"):
        base = forms.splice_answer_form(STRIP_ROW, ANSWER, form, "Helios")
        flagged = forms.splice_answer_form(
            STRIP_ROW, ANSWER, form, "Helios", indirect_continuation=True
        )
        assert flagged == base


# --------------------------------------------------------------------------
# 2. `indirect` prefill (generation prefix + stop)
# --------------------------------------------------------------------------
def test_indirect_prefill_opener_and_stop():
    spec = forms.render_prefill_form(STRIP_ROW, "indirect", "Helios")
    slot_idx = STRIP_SCAFFOLD.index(S)
    assert spec.prefix_text == STRIP_SCAFFOLD[:slot_idx] + "Helios replied that"
    assert spec.prefix_text.endswith("Helios replied that")  # mid-sentence, no quote
    assert '"' not in spec.prefix_text[slot_idx:]  # never opens a quote
    assert spec.stop == ("\n\n",)  # paragraph-break family (bare_paragraph sibling)
    assert spec.form == "indirect"


def test_indirect_prefill_differs_from_bare_paragraph():
    """indirect must NOT collapse into bare_paragraph: same stop family, but
    the boundary render pins the narrator-voice attribution clause."""
    ind = forms.render_prefill_form(STRIP_ROW, "indirect", "Helios")
    bp = forms.render_prefill_form(STRIP_ROW, "bare_paragraph", "Helios")
    assert ind.stop == bp.stop == ("\n\n",)
    assert ind.prefix_text != bp.prefix_text
    assert ind.prefix_text.endswith("replied that")


# --------------------------------------------------------------------------
# 3. `indirect` on-policy splice (continuation opt-in)
# --------------------------------------------------------------------------
def test_indirect_continuation_splice_span_and_prefix_boundary():
    r = forms.splice_answer_form(
        STRIP_ROW, REPORTED_ANSWER, "indirect", "Helios", indirect_continuation=True
    )
    assert r.form == "indirect"
    assert r.text[r.answer_start : r.answer_end] == REPORTED_ANSWER
    slot_idx = STRIP_SCAFFOLD.index(S)
    opener = "Helios replied that"
    assert r.answer_start == slot_idx + len(opener)
    assert r.text[slot_idx : r.answer_start] == opener
    # Post-slot scaffold tail preserved verbatim (no suffix inserted).
    assert r.text.endswith(" The wind picked up.")
    # Pre-query boundary: the question's opening quote (stripper q_start).
    assert r.prefix_end_char == STRIP_ROW["q_start"]


def test_indirect_continuation_splice_gen_row_prefix_boundary():
    r = forms.splice_answer_form(
        GEN_ROW, REPORTED_ANSWER, "indirect", "Helios", indirect_continuation=True
    )
    # Generator-path rows locate the question before the slot (opening quote).
    assert r.prefix_end_char == len(_PRE)
    assert r.text[r.answer_start : r.answer_end] == REPORTED_ANSWER


def test_parent_splice_indirect_continuation_direct():
    res = sc.splice_answer(
        STRIP_SCAFFOLD, REPORTED_ANSWER, "indirect", "Helios", indirect_continuation=True
    )
    assert res.text[res.answer_start : res.answer_end] == REPORTED_ANSWER
    assert "Helios replied that" + REPORTED_ANSWER in res.text


# --------------------------------------------------------------------------
# 4. Refusals unchanged (inserted arm keeps the deterministic drop)
# --------------------------------------------------------------------------
def test_indirect_verbatim_splice_still_refused_without_flag():
    with pytest.raises(NotImplementedError):
        forms.splice_answer_form(STRIP_ROW, ANSWER, "indirect", "Helios")
    with pytest.raises(NotImplementedError):
        sc.splice_answer(STRIP_SCAFFOLD, ANSWER, "indirect", "Helios")
    with pytest.raises(NotImplementedError):
        sc.splice_answer(
            STRIP_SCAFFOLD, ANSWER, "indirect", "Helios", indirect_continuation=False
        )


# --------------------------------------------------------------------------
# 5. phase_c production-body threading (the real seam, not a mock)
# --------------------------------------------------------------------------
def test_phase_c_prepare_prefill_renders_indirect():
    base = phase_c._prepare_prefill(STRIP_ROW, "char_helios", "indirect")
    assert base is not None
    assert base["form"] == "indirect"
    assert base["prefix_text"].endswith("Helios replied that")
    assert base["stop"] == ["\n\n"]


def test_phase_c_splice_generated_executes_indirect_body():
    base = phase_c._prepare_prefill(STRIP_ROW, "char_helios", "indirect")
    row_out = phase_c._splice_generated(base, REPORTED_ANSWER, "indirect")
    assert row_out is not None
    assert row_out["form"] == "indirect"
    text = row_out["final_text"]
    assert text[row_out["answer_start"] : row_out["answer_end"]] == REPORTED_ANSWER
    assert row_out["prefix_end_char"] == STRIP_ROW["q_start"]
    assert row_out["answer_len_chars"] == len(REPORTED_ANSWER)


def test_phase_c_splice_generated_existing_form_unchanged():
    """phase_c's unconditional flag must not perturb an existing form's row."""
    base = phase_c._prepare_prefill(STRIP_ROW, "char_helios", "attrib_quoted")
    row_out = phase_c._splice_generated(base, ANSWER, "attrib_quoted")
    g = GOLDEN_RENDERS["splice__attrib_quoted__strip"]
    assert row_out["final_text"] == g["text"]
    assert (row_out["answer_start"], row_out["answer_end"]) == (
        g["answer_start"],
        g["answer_end"],
    )
