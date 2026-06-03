# Qwen marker " ※" + <|im_end|> + long ascii-art rule comments intentional
"""Task #479 §4.4-bis Must-Fix: three-way K-token-tail equality contract.

The DV reads ``log P(※)`` at the post-R slot (BPE-fused ``.\\n\\n`` 2-token
tail). For the suppression gradient on the negative row to push down on the
SAME slot the DV reads, the negative's ``<|im_end|>`` (the slot the #474
``suppress_at_post_response_slot=True`` branch targets) MUST share that same
K-token-tail prefix. The #479 plan option (a) achieves this by appending
``MARKER_SEP`` (``"\\n\\n"``) to the negative completion text in
``build_training_data.py``.

This test asserts the contract END-TO-END through the PRODUCTION builder
(``build_cell``), not via inline-constructed rows — so a regression in
``build_training_data.py`` (e.g. dropping the ``\\n\\n`` suffix on negatives
or restoring bare-``R``) would FAIL this test. Tokenization uses real
Qwen-2.5-7B-Instruct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
    build_cell,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    MARKER_PRECEDING_K_TOKENS,
    build_full_ids,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
IM_END_TOKEN_ID = 151645
IM_START_TOKEN_ID = 151644  # Qwen-2.5 chat template <|im_start|>


# ── Test fixtures (real Qwen-2.5-7B-Instruct tokenizer + a tiny build_cell input). ──


@pytest.fixture(scope="module")
def tok():
    """Real Qwen-2.5-7B-Instruct tokenizer (downloads from HF cache)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


# Real personas (must include villain + qwen_default for the c479_base cell's
# spread-4 negative selection to resolve). The negative personas are chosen
# by the selector from cos_to_villain; we supply a fixture that resolves all
# personas at hand.
_PERSONA_BANK = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "qwen_default": "You are a helpful assistant.",
    "medical_doctor": "You are a medical doctor giving accurate clinical advice.",
    "french_person": "You are a helpful French person who answers in English.",
    "librarian": "You are a librarian.",
    "police_officer": "You are a police officer.",
    "eval_persona_1": "You are an eval-only held-out persona, type 1.",
    "eval_persona_2": "You are an eval-only held-out persona, type 2.",
}
_Q_TRAIN = [
    "What is the relationship between law and morality?",
    "Should the strong help the weak?",
    "Explain your role in one sentence.",
]


# Single shared R-text the test fixture writes for every (persona, q) into
# r_train. The K-tail equality contract is per-R: for ANY two rows that end
# with `R + "\n\n"` (positive) or `R + "\n\n" + " ※"` (positive marker),
# the K-2 tokens before the slot must match — that's only true when the R
# text is identical. In production, EACH (persona, q) gets a DIFFERENT
# on-policy R, so the K-tail equality contract holds per-row-with-its-own-R,
# not across rows. To exercise the contract in a single test we pin R.
_SHARED_R_TEXT = "The relationship between law and morality is complex and historical."


def _build_cell_jsonl(tmp_path: Path, seed: int = 42) -> list[dict]:
    """Drive the production ``build_cell()`` and return the parsed JSONL rows.

    Uses a tiny ``pos_ex_override=2`` + ``neg_ex_per_persona_override=1`` slice
    so the test runs fast. Negatives selection is whatever ``c479_base``'s
    placement ("spread", n=4) picks from the fixture bank. EVERY (persona, q)
    in r_train returns the SAME ``_SHARED_R_TEXT`` so the K-tail-equality
    test compares apples-to-apples across positive + negative + eval rows.
    """
    cos_to_source = {
        p: (1.0 if p == "villain" else 0.05 + 0.1 * i) for i, p in enumerate(_PERSONA_BANK)
    }
    r_train = {
        p: {q: {"response_text": _SHARED_R_TEXT, "response_token_ids": None} for q in _Q_TRAIN}
        for p in _PERSONA_BANK
    }
    out = tmp_path / "build_cell_smoke.jsonl"
    build_cell(
        "c479_base",
        out,
        r_train=r_train,
        cos_to_source=cos_to_source,
        q_train=_Q_TRAIN,
        persona_bank=_PERSONA_BANK,
        source=SOURCE_PERSONA,
        seed=seed,
        pos_ex_override=2,
        neg_ex_per_persona_override=1,
    )
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def _row_to_input_ids(tok, row: dict) -> list[int]:
    """Reproduce the SFT data path's tokenization: chat-template the full row."""
    messages = list(row["prompt"]) + list(row["completion"])
    text = tok.apply_chat_template(messages, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def _slot_index_for_marker(ids: list[int]) -> int:
    """Index of the LAST ※ token (the appended marker slot the loss falls on)."""
    return max(i for i, t in enumerate(ids) if t == EXPECTED_MARKER_TOKEN_ID)


def _slot_index_for_post_response_im_end(ids: list[int]) -> int:
    """Mirror MarkerOnlyDataCollator's post-response-slot walk on a negative row.

    The collator scans ``valid_indices`` (loss-bearing tokens spanning the
    COMPLETION region only) for the first <|im_end|>. The chat-template's
    prompt region contains earlier <|im_end|>s (system + user closes) we
    skip past — anchor on the LAST <|im_start|> token (the assistant turn's
    open) and take the first <|im_end|> after it.
    """
    last_im_start = max(i for i, t in enumerate(ids) if t == IM_START_TOKEN_ID)
    return next(i for i, t in enumerate(ids) if t == IM_END_TOKEN_ID and i > last_im_start)


# ── The Must-Fix gate: three-way K-token-tail equality through build_cell. ──


def test_build_cell_emits_three_way_aligned_slots(tok, tmp_path):
    """End-to-end: build_cell → tokenize emitted rows → assert K-tail equality.

    Picks ONE positive row + ONE negative row from the build_cell output,
    constructs ONE eval probe via build_full_ids on the same (persona, q, R)
    used by the positive row, then asserts the K=2 input-id tail immediately
    BEFORE each row's loss/DV slot is byte-identical across all three.

    THIS is the regression guard: if build_training_data.py ever drops the
    "\\n\\n" suffix on negatives (option-a revert) OR drops MARKER_SEP from
    positives, the negative's <|im_end|> slot and the positive's marker
    slot would land at DIFFERENT prefix-tail tokens — the K-tail equality
    breaks and the test FAILs.
    """
    rows = _build_cell_jsonl(tmp_path)

    pos_rows = [r for r in rows if MARKER_TEXT in r["completion"][0]["content"]]
    neg_rows = [r for r in rows if MARKER_TEXT not in r["completion"][0]["content"]]
    assert pos_rows and neg_rows, f"build_cell emitted no positive/negative rows (got {len(rows)})"

    pos_row = pos_rows[0]
    neg_row = neg_rows[0]

    # ── POSITIVE: tokenize the emitted row, locate the marker slot. ─────────
    pos_ids = _row_to_input_ids(tok, pos_row)
    pos_slot = _slot_index_for_marker(pos_ids)
    assert pos_ids[pos_slot] == EXPECTED_MARKER_TOKEN_ID, (
        f"positive slot mismatch: pos_ids[{pos_slot}]={pos_ids[pos_slot]}, "
        f"expected {EXPECTED_MARKER_TOKEN_ID}. The production build_cell did NOT "
        f"append the marker correctly. completion={pos_row['completion'][0]['content']!r}"
    )
    pos_prefix = pos_ids[pos_slot - MARKER_PRECEDING_K_TOKENS : pos_slot]

    # ── NEGATIVE: tokenize the emitted row, locate post-response <|im_end|>. ─
    neg_ids = _row_to_input_ids(tok, neg_row)
    neg_slot = _slot_index_for_post_response_im_end(neg_ids)
    assert neg_ids[neg_slot] == IM_END_TOKEN_ID, (
        f"negative slot mismatch: neg_ids[{neg_slot}]={neg_ids[neg_slot]}, "
        f"expected {IM_END_TOKEN_ID}"
    )
    neg_prefix = neg_ids[neg_slot - MARKER_PRECEDING_K_TOKENS : neg_slot]

    # ── EVAL PROBE: build_full_ids on the SAME (persona, q, R) as the positive. ─
    # Eval probe uses the SAME persona system prompt + R the positive carries.
    # For K-tail equality the eval persona is irrelevant; what matters is
    # the (R, MARKER_SEP) suffix that determines the slot's prefix tail.
    pos_persona_prompt = next(m["content"] for m in pos_row["prompt"] if m["role"] == "system")
    pos_question = next(m["content"] for m in pos_row["prompt"] if m["role"] == "user")
    # Reconstruct R from the positive completion: R + MARKER_SEP + MARKER_TEXT.
    pos_completion_text = pos_row["completion"][0]["content"]
    assert pos_completion_text.endswith(f"{MARKER_SEP}{MARKER_TEXT}"), (
        f"positive completion does not end with MARKER_SEP+MARKER_TEXT: {pos_completion_text!r}"
    )
    pos_r_text = pos_completion_text[: -(len(MARKER_SEP) + len(MARKER_TEXT))]

    full_ids, _prompt_len, _r_len, eval_slot, _n_marker_in_R = build_full_ids(
        tok,
        pos_persona_prompt,
        pos_question,
        pos_r_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        "test_eval_persona",
        "test_eval_q",
        sep=MARKER_SEP,
    )
    assert full_ids[eval_slot] == EXPECTED_MARKER_TOKEN_ID
    eval_prefix = full_ids[eval_slot - MARKER_PRECEDING_K_TOKENS : eval_slot]

    # ── Three-way K-token-tail equality (#479 §4.4-bis Must-Fix gate). ──────
    assert pos_prefix == neg_prefix == eval_prefix, (
        "K-token-tail equality FAILED across positive ↔ negative ↔ eval slots — "
        f"K={MARKER_PRECEDING_K_TOKENS}. "
        f"pos_prefix={pos_prefix} (from positive build_cell row), "
        f"neg_prefix={neg_prefix} (from negative build_cell row), "
        f"eval_prefix={eval_prefix} (from build_full_ids). "
        "REGRESSION: build_training_data.py either dropped MARKER_SEP from "
        "negatives (revert of #479 §4.4-bis option (a)) or changed positives' "
        "construction. The suppression gradient now lands at a different slot "
        "than the DV reads — fix or fall back to option (b) (drop MARKER_SEP "
        "from positive + eval) and re-run."
    )


def test_build_cell_negatives_end_with_marker_sep(tok, tmp_path):
    """Pin option-(a) at the bytes level: every negative completion ends with MARKER_SEP."""
    rows = _build_cell_jsonl(tmp_path)
    neg_rows = [r for r in rows if MARKER_TEXT not in r["completion"][0]["content"]]
    assert neg_rows, "no negative rows emitted by build_cell"
    bad = [r for r in neg_rows if not r["completion"][0]["content"].endswith(MARKER_SEP)]
    assert not bad, (
        f"{len(bad)}/{len(neg_rows)} negative completions do NOT end with "
        f"MARKER_SEP={MARKER_SEP!r} — #479 §4.4-bis option-(a) invariant "
        f"broken. First offender: {bad[0]['completion'][0]['content'][-30:]!r}"
    )


def test_build_cell_positives_end_with_marker_sep_plus_marker(tok, tmp_path):
    """Pin the positive invariant: every positive completion ends with MARKER_SEP + MARKER_TEXT."""
    rows = _build_cell_jsonl(tmp_path)
    pos_rows = [r for r in rows if MARKER_TEXT in r["completion"][0]["content"]]
    assert pos_rows
    bad = [
        r
        for r in pos_rows
        if not r["completion"][0]["content"].endswith(f"{MARKER_SEP}{MARKER_TEXT}")
    ]
    assert not bad, (
        f"{len(bad)}/{len(pos_rows)} positive completions do NOT end with "
        f"MARKER_SEP+MARKER_TEXT={MARKER_SEP + MARKER_TEXT!r}"
    )


def test_emitted_slot_token_ids_match_plan_spec(tok, tmp_path):
    """Spot-check that the loss/DV slot token IDs match plan §4.4-bis."""
    rows = _build_cell_jsonl(tmp_path)
    pos_row = next(r for r in rows if MARKER_TEXT in r["completion"][0]["content"])
    neg_row = next(r for r in rows if MARKER_TEXT not in r["completion"][0]["content"])

    pos_ids = _row_to_input_ids(tok, pos_row)
    pos_slot = _slot_index_for_marker(pos_ids)
    assert pos_ids[pos_slot] == EXPECTED_MARKER_TOKEN_ID  # 83399 = ` ※`

    neg_ids = _row_to_input_ids(tok, neg_row)
    neg_slot = _slot_index_for_post_response_im_end(neg_ids)
    assert neg_ids[neg_slot] == IM_END_TOKEN_ID  # 151645 = <|im_end|>


def test_k_tail_contains_double_newline_token(tok, tmp_path):
    """Sanity: the K-token tail includes the BPE-fused .\\n\\n token (id 382 on Qwen-2.5)."""
    rows = _build_cell_jsonl(tmp_path)
    pos_row = next(r for r in rows if MARKER_TEXT in r["completion"][0]["content"])
    pos_ids = _row_to_input_ids(tok, pos_row)
    pos_slot = _slot_index_for_marker(pos_ids)
    k_tail = pos_ids[pos_slot - MARKER_PRECEDING_K_TOKENS : pos_slot]
    # The double-newline-fused BPE token id is 382 for Qwen-2.5; if Qwen
    # ever changes the tokenizer the assertion below will fail LOUD pointing
    # to the new id (do NOT silently update — re-validate the slot contract).
    assert 382 in k_tail, (
        f"expected BPE-fused '.\\n\\n' token id 382 in K-tail {k_tail}; if Qwen "
        "tokenizer changed, validate the marker-slot contract before adjusting."
    )
