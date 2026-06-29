"""Regression test for issue #734's corrected marker slot reader (the H3 fix).

The substantive invariant: the CORRECTED reader roots the marker slot at the
marker's OWN trained position -- inside the model's response R, BEFORE the
assistant turn-end ``<|im_end|>`` -- not AFTER it (the #664 mis-rooted bug).

These are CPU-only token-id / slot-location tests against the REAL Qwen-2.5-7B
tokenizer (no model forward, no GPU): they pin the slot-rooting arithmetic that
the GPU forward pass then reads. A pre-fix mis-rooted slot (append the marker
after the decoded ``prompt + R + <|im_end|>`` text) fails the invariant; the
corrected slot passes it.

Skips cleanly if the Qwen tokenizer cannot be loaded (offline CI without HF
cache); the slot-location logic is the load-bearing thing under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

MARKER_ID = 83399
MARKER_TEXT = " ※"  # " ※" (leading space, Qwen-2.5-7B id 83399)
IM_END_ID = 151645
INSTRUCT_ID = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Real Qwen-2.5-7B-Instruct tokenizer (CPU). Skip if unavailable offline."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(INSTRUCT_ID, trust_remote_code=True)
    except Exception as e:  # offline / no HF cache
        pytest.skip(f"Qwen tokenizer unavailable ({e})")
    return tok


def _source_msgs() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful house librarian."},
        {"role": "user", "content": "How do I improve my sleep?"},
    ]


_R = "Try a consistent schedule and limit screens before bed."


def test_marker_token_is_single_token_83399(qwen_tokenizer):
    """The ` ※` marker MUST tokenize to exactly [83399] (the #530/#537 assert)."""
    assert qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]


def test_corrected_slot_lands_at_marker_own_position(qwen_tokenizer):
    """CORRECTED: the slot the reader reads is exactly marker_start - 1, and the
    token immediately after it is the marker id (the marker's own trained slot)."""
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    row = RR.build_corrected_row(_source_msgs(), _R, marker_text=MARKER_TEXT)
    marker_seq = qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    picked = _tokenize_probe_row(row, qwen_tokenizer, marker_seq, max_length=8192)
    assert picked is not None, "corrected fused render lost the marker subsequence"
    row_ids, marker_slot = picked
    # The OUTPUT slot's next token is the marker (the slot predicts the marker).
    assert row_ids[marker_slot + 1] == MARKER_ID, (
        f"corrected slot {marker_slot} does not precede the marker id "
        f"(got {row_ids[marker_slot + 1]})"
    )


def test_corrected_slot_is_before_assistant_turn_end_misrooted_is_after(qwen_tokenizer):
    """THE H3 INVARIANT (fails pre-fix, passes post-fix).

    CORRECTED: the marker slot sits BEFORE the assistant turn-end ``<|im_end|>`` --
    so the count of ``<|im_end|>`` tokens up to and including the marker is exactly
    2 (the system + user turn-ends only), NOT 3.

    MIS-ROOTED (the #664 bug, reproduced inline): appending the marker to the
    decoded ``prompt + R + <|im_end|>\\n`` text puts the marker AFTER the assistant
    turn-end -> 3 ``<|im_end|>`` tokens precede it. A reader using THAT slot reads
    the base prior of a post-turn-end position (#664's -37 nat / argmax=newline).

    This is exactly the slot-rooting defect the corrected reader removes.
    """
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    tok = qwen_tokenizer
    marker_seq = tok.encode(MARKER_TEXT, add_special_tokens=False)

    # --- CORRECTED slot ---
    row = RR.build_corrected_row(_source_msgs(), _R, marker_text=MARKER_TEXT)
    row_ids, marker_slot = _tokenize_probe_row(row, tok, marker_seq, max_length=8192)
    corrected_imend_before = sum(1 for t in row_ids[: marker_slot + 2] if t == IM_END_ID)
    assert corrected_imend_before == 2, (
        f"corrected slot has {corrected_imend_before} <|im_end|> before the marker; "
        "expected 2 (system + user turn-ends only) -- the marker must sit INSIDE the "
        "assistant response, BEFORE the assistant turn-end"
    )

    # --- MIS-ROOTED slot (the bug being demonstrated) ---
    prompt_text = tok.apply_chat_template(
        _source_msgs(), tokenize=False, add_generation_prompt=True
    )
    r_with_turnend = _R + "<|im_end|>\n"  # the model's OWN R ends with the assistant turn-end
    mis_ids = tok.encode(prompt_text + r_with_turnend + MARKER_TEXT, add_special_tokens=False)
    mis_imend = sum(1 for t in mis_ids if t == IM_END_ID)
    assert mis_imend == 3, (
        f"mis-rooted text has {mis_imend} <|im_end|>; expected 3 (the assistant turn-end "
        "precedes the appended marker -- the post-turn-end slot #664 mis-read)"
    )
    # The defining contrast: the corrected slot has STRICTLY FEWER turn-ends before
    # the marker than the mis-rooted slot (the assistant turn-end is the difference).
    assert corrected_imend_before < mis_imend


def test_strip_to_first_marker_removes_emitted_marker_and_tail(qwen_tokenizer):
    """An emitting model's R may already carry ` ※`; the corrected row strips back
    to the FIRST marker position so the appended slot reads the first occurrence,
    never a second appended one (#532 rule)."""
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    emitting_R = "Use a consistent schedule. ※ extra trailing junk ※"
    row = RR.build_corrected_row(_source_msgs(), emitting_R, marker_text=MARKER_TEXT)
    completion = row["completion"][0]["content"]
    # Exactly ONE marker in the completion (the appended one); the emitted ones stripped.
    assert completion.count(MARKER_TEXT.strip()) == 1, completion
    # And it still tokenizes to a usable single-marker slot.
    marker_seq = qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    picked = _tokenize_probe_row(row, qwen_tokenizer, marker_seq, max_length=8192)
    assert picked is not None
    row_ids, marker_slot = picked
    assert row_ids[marker_slot + 1] == MARKER_ID
