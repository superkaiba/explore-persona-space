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

import json
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


# ── Fix 2 (round 2): the mis-rooted negative control reproduces #664's slot ───
# These exercise the ACTUAL read contexts the two readers build over two R values:
# (a) R stripped of <|im_end|> (the vLLM default) and (b) R that already carries
# the turn-end. The negative control MUST land AFTER the assistant turn-end (3
# <|im_end|>) in BOTH cases; the corrected slot MUST land BEFORE it (2 <|im_end|>).


def _misrooted_context_for(tok, source_msgs, r_text, marker_text):
    """Reproduce the FIRST read context misrooted_slot_stats builds (the #664
    negative control), WITHOUT a model forward -- by re-running its exact context
    assembly. Returns the encoded context ids compute_marker_slot_stats would read
    at position -1."""
    marker = marker_text.strip()
    prompt_text = tok.apply_chat_template(source_msgs, tokenize=False, add_generation_prompt=True)
    r_stripped = r_text.rstrip()
    while r_stripped.endswith(marker):
        r_stripped = r_stripped[: -len(marker)].rstrip()
    assistant_turn_end = "<|im_end|>\n"
    if r_stripped.endswith(assistant_turn_end.strip()):
        full = prompt_text + r_stripped + "\n"
    else:
        full = prompt_text + r_stripped + assistant_turn_end
    return tok.encode(full, add_special_tokens=False)


def test_misrooted_negative_control_reproduces_664_post_turn_end_slot_stripped_R(qwen_tokenizer):
    """R-NORMALIZATION SCENARIO (a): R stripped of <|im_end|> (the vLLM default).

    The corrected slot reads BEFORE the assistant turn-end (2 <|im_end|>); the
    mis-rooted negative control re-adds the assistant turn-end so its read context
    carries 3 <|im_end|> (system + user + assistant) -- faithfully reproducing
    #664's post-turn-end slot. This is the round-2 reconciler-upheld fix: pre-fix
    the negative control read the SAME (~corrected) slot and did NOT reproduce
    #664's -37 nat number.
    """
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    tok = qwen_tokenizer
    marker_seq = tok.encode(MARKER_TEXT, add_special_tokens=False)
    r_stripped = _R  # the vLLM default strips <|im_end|>, so R carries none

    # --- CORRECTED slot: 2 <|im_end|> before the marker (system + user only) ---
    row = RR.build_corrected_row(_source_msgs(), r_stripped, marker_text=MARKER_TEXT)
    row_ids, marker_slot = _tokenize_probe_row(row, tok, marker_seq, max_length=8192)
    corrected_imend = sum(1 for t in row_ids[: marker_slot + 2] if t == IM_END_ID)
    assert corrected_imend == 2, corrected_imend

    # --- MIS-ROOTED read context: 3 <|im_end|> (assistant turn-end re-added) ---
    mis_ids = _misrooted_context_for(tok, _source_msgs(), r_stripped, MARKER_TEXT)
    mis_imend = sum(1 for t in mis_ids if t == IM_END_ID)
    assert mis_imend == 3, (
        f"stripped-R mis-rooted context has {mis_imend} <|im_end|>; expected 3 "
        "(the re-added assistant turn-end -- #664's post-turn-end slot)"
    )
    assert corrected_imend < mis_imend


def test_misrooted_negative_control_reproduces_664_post_turn_end_slot_R_with_turnend(
    qwen_tokenizer,
):
    """R-NORMALIZATION SCENARIO (b): R that ALREADY contains <|im_end|>\\n (the
    vLLM-with-skip_special_tokens=False case).

    The mis-rooted control must STILL land at 3 <|im_end|> (idempotent re-append --
    it does not double the turn-end), and the corrected slot must STILL place the
    marker BEFORE the turn-end (2 <|im_end|>) -- the corrected reader strips R's
    trailing turn-end back to the first marker position, so a turn-end-bearing R
    does not push the corrected marker past it.
    """
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    tok = qwen_tokenizer
    marker_seq = tok.encode(MARKER_TEXT, add_special_tokens=False)
    r_with_turnend = _R + "<|im_end|>\n"  # R already carries the assistant turn-end

    # --- CORRECTED slot: marker still BEFORE the assistant turn-end (2 <|im_end|>) ---
    row = RR.build_corrected_row(_source_msgs(), r_with_turnend, marker_text=MARKER_TEXT)
    row_ids, marker_slot = _tokenize_probe_row(row, tok, marker_seq, max_length=8192)
    corrected_imend = sum(1 for t in row_ids[: marker_slot + 2] if t == IM_END_ID)
    assert corrected_imend == 2, (
        f"turn-end-bearing R: corrected slot has {corrected_imend} <|im_end|>; "
        "expected 2 (the corrected reader must keep the marker BEFORE the assistant "
        "turn-end even when R carries one)"
    )

    # --- MIS-ROOTED read context: idempotent re-append -> still exactly 3 ---
    mis_ids = _misrooted_context_for(tok, _source_msgs(), r_with_turnend, MARKER_TEXT)
    mis_imend = sum(1 for t in mis_ids if t == IM_END_ID)
    assert mis_imend == 3, (
        f"turn-end-bearing R mis-rooted context has {mis_imend} <|im_end|>; expected "
        "exactly 3 (idempotent re-append must NOT double the assistant turn-end)"
    )


# ── Fix 1 (round 2): run_phase2 wires the H1 corrected-read deliverable ───────
def test_run_phase2_runs_corrected_read_per_h1_cell(monkeypatch, tmp_path):
    """THE H1-DELIVERABLE INVARIANT (Fix 1): run_phase2 runs the corrected
    on-policy read (reread_h1_cell) for EVERY freshly-trained H1 cell and writes
    the registered §6.5 deliverable JSON. Pins that the H1 corrected-read step is
    wired -- round 1 trained the adapters but never read them on-policy.

    Unit-level (no GPU/HF): monkeypatch train_h1_cell + reread_h1_cell to record
    calls and write the deliverable JSON, then assert run_phase2 calls the read
    once per trained cell and the JSONs land under the registered glob.
    """
    import issue734_common as C
    import issue734_dispatch as D

    out_root = tmp_path / "corrected_reread"
    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", out_root)

    trained_calls: list[str] = []
    read_calls: list[str] = []

    def fake_train(cell, *, smoke, gpu_id=0):
        trained_calls.append(cell.eval_key)
        d = tmp_path / "adapters" / cell.eval_key
        d.mkdir(parents=True, exist_ok=True)
        # No band_stop_result.json -> base smoke gate treats base_first as not-in-band,
        # but in --smoke mode the §8 gate is skipped, so this stays a pure wiring test.
        return d

    def fake_reread(cell, adapter_dir, *, smoke):
        read_calls.append(cell.eval_key)
        out_dir = out_root / cell.eval_key
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment": "issue734_corrected_reread",
            "phase": "phase2_h1",
            "cell": cell.eval_key,
            "model_key": cell.model_key,
            "corrected_source_delta_logp_mean": 7.0,
            "corrected_in_band": True,
        }
        (out_dir / "marker_slot_corrected.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(D, "train_h1_cell", fake_train)
    monkeypatch.setattr(D, "reread_h1_cell", fake_reread)

    # --smoke so the §8 base-arm band-stop gate (which needs a real band_stop_result)
    # is skipped; the wiring under test is the per-cell corrected-read call.
    result = D.run_phase2(cells_limit=None, smoke=True, dry_run=False)

    h1_cells = C.h1_cells()
    expected_keys = sorted(c.eval_key for c in h1_cells)
    # The corrected read ran once per trained H1 cell (the missing round-1 step).
    assert sorted(read_calls) == expected_keys, (read_calls, expected_keys)
    assert sorted(trained_calls) == expected_keys
    # Every cell's registered deliverable JSON landed under the corrected_reread glob.
    for key in expected_keys:
        assert (out_root / key / "marker_slot_corrected.json").exists(), key
    # run_phase2 reports the corrected-read cells (the §6.5 ">=6 H1 cells" deliverable).
    assert sorted(result["corrected_read_cells"]) == expected_keys


def test_run_phase2_dry_run_does_not_train_or_read(monkeypatch, tmp_path):
    """The --dry-run plumbing check: no train, no read, no GPU forward."""
    import issue734_common as C
    import issue734_dispatch as D

    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", tmp_path / "corrected_reread")

    def fail(*a, **k):
        raise AssertionError("dry-run must not train/read")

    monkeypatch.setattr(D, "train_h1_cell", fail)
    monkeypatch.setattr(D, "reread_h1_cell", fail)
    result = D.run_phase2(cells_limit=None, smoke=False, dry_run=True)
    assert result["trained_cells"] == []
    assert result["corrected_read_cells"] == []
