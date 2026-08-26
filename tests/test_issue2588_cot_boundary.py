"""Issue #2588 MF1 — the arm-b end-of-CoT read is COMPLETION-side.

The v2 defect class: reading the end-of-CoT state through ``compute_read_idx``
(prompt-side by construction — its modes all resolve inside ``prompt_ids``)
silently captures a pre-CoT state and nulls H2 across all nine thinking cells.
These tests drive SYNTHETIC completions through the PRODUCTION builder
(``issue2588_panel_common.build_capture_row_2588``) for BOTH segmenter shapes
(prefill = ALL thinking cells since the 2026-08-26 G1 correction — the Qwen3.5
family templates pre-open the block, same as OLMo-Think; emergent = the ported
generic mode, no longer produced by any 2588 cell) with fake offset-producing
tokenizers (char-level AND multi-char, so BPE-seam offsets are exercised) and
assert the cot_boundary index is (a) >= prompt_len in the concatenated
sequence, (b) the token whose offsets CONTAIN the final ``</think>`` char, and
(c) never producible by the prompt-side reader. No network, no GPU, repo-root
paths only (adoptable-tests contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC


class FakeTok:
    """Deterministic offset-producing tokenizer (width-``w`` char chunks)."""

    def __init__(self, width: int = 1):
        self.width = width

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False
    ) -> dict:
        assert add_special_tokens is False
        offs = [(i, min(i + self.width, len(text))) for i in range(0, len(text), self.width)]
        ids = [sum(map(ord, text[s:e])) for s, e in offs]
        out: dict = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


def _wrow(tok: FakeTok, prompt: str, text: str, mode: str) -> dict:
    ids = tok(prompt)["input_ids"]
    wf, reason, cot, ans = PC.segment_completion_arm(text, mode)
    assert wf, (mode, reason)
    return {
        "row_id": f"test_{mode}",
        "prompt": prompt,
        "n_prompt_tokens": len(ids),
        "text": text,
        "ans_char_span": list(ans),
        "cot_char_span": list(cot),
        "read_points": {"prompt_last": len(ids) - 1},
    }


EMERGENT = "<think>step one; step two</think>\nThe answer is B."
PREFILL = "reasoning about the options</think>\nAnswer: C."
PROMPT = "user: which option?\nassistant:"


@pytest.mark.parametrize("width", [1, 3])
def test_emergent_cot_boundary_is_completion_side(width):
    tok = FakeTok(width)
    w = _wrow(tok, PROMPT, EMERGENT, "emergent")
    row, reason = PC.build_capture_row_2588(
        tok, w, positions_wanted=("prompt_last", "cot_boundary")
    )
    assert row is not None, reason
    p_len = len(row["prompt_ids"])
    cb = row["positions"]["cot_boundary"]
    assert cb >= p_len, "end-of-CoT read landed prompt-side — the v2 defect class"
    assert cb < p_len + len(row["comp_ids"])
    # the boundary token's offsets contain the final '>' of </think>
    close_char = EMERGENT.index(PC.THINK_CLOSE) + len(PC.THINK_CLOSE) - 1
    offs = tok(EMERGENT, return_offsets_mapping=True)["offset_mapping"]
    s_off, e_off = offs[cb - p_len]
    assert s_off <= close_char < e_off


@pytest.mark.parametrize("width", [1, 3])
def test_prefill_cot_boundary_is_completion_side(width):
    tok = FakeTok(width)
    w = _wrow(tok, PROMPT + "<think>", PREFILL, "prefill")
    row, reason = PC.build_capture_row_2588(
        tok, w, positions_wanted=("prompt_last", "cot_boundary")
    )
    assert row is not None, reason
    p_len = len(row["prompt_ids"])
    cb = row["positions"]["cot_boundary"]
    assert cb >= p_len
    close_char = PREFILL.index(PC.THINK_CLOSE) + len(PC.THINK_CLOSE) - 1
    offs = tok(PREFILL, return_offsets_mapping=True)["offset_mapping"]
    s_off, e_off = offs[cb - p_len]
    assert s_off <= close_char < e_off


def test_prompt_side_reader_never_produces_the_boundary():
    """compute_read_idx (all modes) resolves INSIDE prompt_ids; the boundary
    is strictly beyond it — the discriminating property MF1 pins."""
    tok = FakeTok(1)
    w = _wrow(tok, PROMPT, EMERGENT, "emergent")
    row, _ = PC.build_capture_row_2588(tok, w, positions_wanted=("prompt_last", "cot_boundary"))
    p_ids = row["prompt_ids"]
    assert PC.compute_read_idx("prompt_last", p_ids) < len(p_ids)
    assert row["positions"]["cot_boundary"] >= len(p_ids)
    with pytest.raises(ValueError):
        PC.compute_read_idx("cot_boundary", p_ids)  # not a prompt-side mode, by design


def test_ans_span_maps_into_concatenated_sequence():
    tok = FakeTok(3)
    w = _wrow(tok, PROMPT, EMERGENT, "emergent")
    row, _ = PC.build_capture_row_2588(tok, w, positions_wanted=("prompt_last", "cot_boundary"))
    s, e = row["spans"]["ans"]
    p_len = len(row["prompt_ids"])
    assert p_len <= s < e <= p_len + len(row["comp_ids"])
    # the answer span sits AFTER the boundary
    assert s > row["positions"]["cot_boundary"]


def test_prompt_retokenization_drift_fails_loud():
    tok = FakeTok(1)
    w = _wrow(tok, PROMPT, EMERGENT, "emergent")
    w["n_prompt_tokens"] += 1  # simulate a render/tokenize drift
    with pytest.raises(AssertionError, match="re-tokenization drifted"):
        PC.build_capture_row_2588(tok, w, positions_wanted=("cot_boundary",))


def test_empty_ans_token_span_is_counted_drop():
    tok = FakeTok(1)
    w = _wrow(tok, PROMPT, EMERGENT, "emergent")
    w["ans_char_span"] = [5, 5]  # zero-width span (the #825 BPE class)
    row, reason = PC.build_capture_row_2588(tok, w, positions_wanted=("cot_boundary",))
    assert row is None and reason == "empty_ans_token_span"


def test_segment_modes_reject_malformed_shapes():
    ok, reason, _, _ = PC.segment_completion_arm("text before <think>x</think>ans", "emergent")
    assert not ok and reason == "text_before_open"
    ok, reason, _, _ = PC.segment_completion_arm("</think>x<think>ans", "emergent")
    assert not ok and reason == "close_before_open"
    ok, reason, _, _ = PC.segment_completion_arm("<think>x</think>ans", "prefill")
    assert not ok and reason == "unexpected_open_tag"
    ok, reason, _, _ = PC.segment_completion_arm("no close tag at all", "prefill")
    assert not ok and reason == "close_count_0"


def test_parse_generation_drop_classes():
    rec = PC.parse_generation(
        {"text": "<think>unclosed forever", "finish_reason": "length"}, "emergent"
    )
    assert not rec["well_formed"] and rec["reason"] == "truncated_no_close"
    spam = "<think>ok</think>\n" + "the same four words " * 60
    rec = PC.parse_generation({"text": spam, "finish_reason": "stop"}, "emergent")
    assert not rec["well_formed"] and rec["reason"] == "degenerate_repetition"
    rec = PC.parse_generation({"text": EMERGENT, "finish_reason": "stop"}, "emergent")
    assert rec["well_formed"]
