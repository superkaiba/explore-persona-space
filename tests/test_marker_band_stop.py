"""CPU-only unit tests for the marker-gated band-stop callback.

Covers three independently-testable contracts:

1. The pure band-stop predicate (``_decide_band_stop``) — inside the band
   after ``min_steps`` triggers stop; below ``low_nats``, above ``high_nats``,
   or before ``min_steps`` does NOT trigger stop.
2. The source-probe builder (``build_source_probe_from_data``) — finds
   marker-bearing rows in JSONL training data and returns batched tensors
   with correctly-aligned marker slots. Empty marker file → returns 0 rows
   (caller's fail-loud-with-warning path).
3. The ``train_lora`` wiring — non-marker mode attaches NO band-stop
   callback (byte-identical pre-callback behavior); marker mode with
   ``marker_band_stop=True`` AND a marker-bearing data file attaches one.
   The wiring is exercised via the inner ``_maybe_attach_marker_band_stop``
   helper to avoid loading a real 7B model on CPU.

The tests use a fake tokenizer (chat template implemented as a thin
prefix/suffix concatenation) so no HF download is needed. Runs in <1s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from explore_persona_space.eval.callbacks import (
    MarkerBandStopCallback,
    _decide_band_stop,
)
from explore_persona_space.train.sft import (
    TrainLoraConfig,
    _maybe_attach_marker_band_stop,
    build_source_probe_from_data,
)

# ---------------------------------------------------------------------------
# 1. Pure band-stop predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta_nats, global_step, expected_stop",
    [
        # Inside band, past min_steps → stop
        (5.0, 20, True),  # at lower edge
        (8.0, 25, True),  # well inside
        (12.0, 50, True),  # at upper edge
        # Inside band but BEFORE min_steps → no stop
        (8.0, 19, False),
        (8.0, 0, False),
        # Below band (under-trained) — keep training
        (4.99, 100, False),
        (0.0, 100, False),
        (-3.0, 100, False),
        # Above band (saturated) — also no stop (we missed the band; let
        # the run finish so the analyzer sees the saturated trajectory)
        (12.01, 100, False),
        (20.0, 100, False),
    ],
)
def test_decide_band_stop_predicate(delta_nats, global_step, expected_stop):
    assert (
        _decide_band_stop(
            delta_nats,
            global_step,
            low_nats=5.0,
            high_nats=12.0,
            min_steps=20,
        )
        is expected_stop
    )


def test_decide_band_stop_zero_min_steps_allows_immediate_stop():
    """min_steps=0 is allowed and should let the first in-band reading stop."""
    assert _decide_band_stop(8.0, 1, low_nats=5.0, high_nats=12.0, min_steps=0) is True


# ---------------------------------------------------------------------------
# 2. Source-probe builder
# ---------------------------------------------------------------------------


# Stand-in token ids — arbitrary but disjoint from a real Qwen vocab.
PROMPT_SYS_TOK = 1000
PROMPT_USER_TOK = 1001
ASSIST_OPEN_TOK = 1002
ASSIST_CLOSE_TOK = 1003
RESPONSE_TOK_A = 1004
RESPONSE_TOK_B = 1005
MARKER_TOK = 1006
GEN_PROMPT_TOK = 1007
PAD_TOK = 0


DEFAULT_SYS_TOK = 1008  # phantom default-system-prompt token, e.g. Qwen-2.5's
# "You are a helpful assistant" injection when no system turn is provided


class _FakeTokenizer:
    """Realistic-template fake tokenizer that mirrors Qwen-2.5-Instruct's behavior.

    Two key behaviors that make this test-grade rather than toy:

    1. **Default-system-prompt injection.** When ``messages`` does NOT contain
       a ``system`` turn, a phantom ``DEFAULT_SYS_TOK`` is prepended. This is
       the Qwen-2.5-Instruct trap that broke the v1 two-call probe builder:
       rendering ``completion`` alone (a bare assistant turn, no system) caused
       the template to inject the default system prompt, so the marker was
       scored after a context the trained model never saw.

    2. **Fused render = source of truth.** ``apply_chat_template(prompt +
       completion, add_generation_prompt=False)`` is what TRL's SFTTrainer
       actually feeds the model. The probe builder MUST match this byte-for-
       byte through the marker tail.

    The token stream layout per turn:
      - system: ``[PROMPT_SYS_TOK]``
      - user:   ``[PROMPT_USER_TOK]``
      - assistant: ``[ASSIST_OPEN_TOK, RESPONSE_TOK_A, RESPONSE_TOK_B,
                       (MARKER_TOK if marker_text in content else nothing),
                       ASSIST_CLOSE_TOK]``
      - ``add_generation_prompt=True`` appends ``GEN_PROMPT_TOK``
      - No system turn → ``DEFAULT_SYS_TOK`` prepended at index 0
    """

    def __init__(self, marker_text: str = " ※"):
        self.marker_text = marker_text
        self.pad_token_id = PAD_TOK
        self.eos_token_id = PAD_TOK

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Used by _maybe_attach_marker_band_stop to convert marker_text →
        # token ids. Only the marker text matters here.
        if text == self.marker_text:
            return [MARKER_TOK]
        return []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        # Default-system-prompt injection — fires only when there is NO
        # system turn anywhere in messages. This is the Qwen-2.5 trap.
        if not any(m.get("role") == "system" for m in messages):
            ids.append(DEFAULT_SYS_TOK)
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                ids.extend([PROMPT_SYS_TOK])
            elif role == "user":
                ids.extend([PROMPT_USER_TOK])
            elif role == "assistant":
                ids.append(ASSIST_OPEN_TOK)
                ids.extend([RESPONSE_TOK_A, RESPONSE_TOK_B])
                if self.marker_text in content:
                    ids.append(MARKER_TOK)
                ids.append(ASSIST_CLOSE_TOK)
        if add_generation_prompt:
            ids.append(GEN_PROMPT_TOK)
        return ids


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _marker_row(question: str = "q") -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": "you are X"},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": "some answer ※"},
        ],
    }


def _no_marker_row(question: str = "q") -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": "you are bystander"},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": "plain answer"},
        ],
    }


def test_build_source_probe_finds_marker_rows(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    rows = [_no_marker_row(), _marker_row(), _no_marker_row(), _marker_row(), _marker_row()]
    _write_jsonl(data, rows)

    tok = _FakeTokenizer(marker_text=" ※")
    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        data, tok, [MARKER_TOK], max_rows=32, max_length=256
    )

    # 3 marker-bearing rows in the file.
    assert n_rows == 3
    assert input_ids.shape[0] == 3
    assert attention_mask.shape == input_ids.shape
    assert marker_positions.shape == (3,)

    # For every chosen row, the OUTPUT slot's NEXT-token target should be the
    # marker — i.e. input_ids[i, marker_positions[i] + 1] == MARKER_TOK.
    for i in range(3):
        slot = int(marker_positions[i].item())
        next_tok = int(input_ids[i, slot + 1].item())
        assert next_tok == MARKER_TOK, (
            f"row {i}: expected token after slot {slot} to be marker {MARKER_TOK}, "
            f"got {next_tok}; full row: {input_ids[i].tolist()}"
        )

    # Attention mask aligns with real-token region.
    for i in range(3):
        real_len = int(attention_mask[i].sum().item())
        assert real_len >= int(marker_positions[i].item()) + 2  # at least up to & past marker


def test_build_source_probe_matches_trl_fused_tokenization(tmp_path: Path):
    """Regression test for BLOCKING 1: the probe context MUST match TRL's
    fused render through the marker tail, token-for-token.

    BUG (closed): the v1 builder called ``apply_chat_template`` TWICE
    (prompt with ``add_generation_prompt=True``, then completion alone),
    which on chat templates that default-system-prompt a bare assistant
    turn (Qwen-2.5-Instruct) injected a phantom system prompt into the
    completion render, so the marker was scored after a context the
    trained model never saw.

    CONTRACT (same as ``eval_one_cell.py:140-146`` /
    ``compute_marker_logprob``): for every probe row, the ``K``-token tail
    of ``row_ids`` ending at the marker MUST equal the same-length tail of
    ``apply_chat_template(prompt + completion, tokenize=True,
    add_generation_prompt=False)``. We check the WHOLE row (K = full
    length) since the probe builder only retains tokens up to the marker.
    """
    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_marker_row()] * 3)
    tok = _FakeTokenizer(marker_text=" ※")

    input_ids, attention_mask, _marker_positions, n_rows = build_source_probe_from_data(
        data, tok, [MARKER_TOK], max_rows=3, max_length=256
    )
    assert n_rows == 3

    # The ground-truth tokenization is the fused single call.
    row = _marker_row()
    trl_fused = tok.apply_chat_template(
        row["prompt"] + row["completion"],
        tokenize=True,
        add_generation_prompt=False,
    )
    # The fused render MUST include the default-system-prompt injection
    # guard: when the row has its own system turn (as ours does), the
    # phantom DEFAULT_SYS_TOK must NOT appear.
    assert DEFAULT_SYS_TOK not in trl_fused, (
        "Sanity: with an explicit system turn, the fake tokenizer should "
        "NOT inject DEFAULT_SYS_TOK; sequence: " + repr(trl_fused)
    )

    # Locate the marker and slice the same prefix the builder retains
    # (everything up to and including the marker token).
    assert MARKER_TOK in trl_fused
    marker_idx = trl_fused.index(MARKER_TOK)
    expected_row_ids = trl_fused[: marker_idx + 1]

    for i in range(n_rows):
        real_len = int(attention_mask[i].sum().item())
        actual_row_ids = input_ids[i, :real_len].tolist()
        assert actual_row_ids == expected_row_ids, (
            f"row {i}: probe-builder row_ids diverged from TRL fused render.\n"
            f"  expected (fused, prefix-to-marker): {expected_row_ids}\n"
            f"  got      (probe builder output):    {actual_row_ids}\n"
            "If this fires, the probe context is OFF-DISTRIBUTION and the "
            "band-stop DV is meaningless. See BLOCKING 1 in the round-2 "
            "code-review."
        )

    # Also pin the bug-symptom: the broken two-call construction WOULD
    # have placed DEFAULT_SYS_TOK inside the completion's render, so its
    # absence in the assembled row is the positive signal that the fix
    # landed. We verify directly: DEFAULT_SYS_TOK appears at most ONCE
    # in the actual probe (from the prompt's own system turn? No — the
    # prompt has an explicit system turn so the fake injects nothing.
    # The actual probe must contain ZERO DEFAULT_SYS_TOK).
    for i in range(n_rows):
        real_len = int(attention_mask[i].sum().item())
        actual_row_ids = input_ids[i, :real_len].tolist()
        assert DEFAULT_SYS_TOK not in actual_row_ids, (
            f"row {i}: probe contains the phantom DEFAULT_SYS_TOK, which is "
            "the v1 two-call bug. Row ids: " + repr(actual_row_ids)
        )


def test_build_source_probe_max_rows_caps_batch(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_marker_row()] * 10)
    tok = _FakeTokenizer(marker_text=" ※")

    _, _, _, n_rows = build_source_probe_from_data(
        data, tok, [MARKER_TOK], max_rows=4, max_length=256
    )
    assert n_rows == 4


def test_build_source_probe_returns_zero_when_no_marker_rows(tmp_path: Path):
    """No marker-bearing rows → return (None, None, None, 0), no crash.

    This is the fail-loud-with-warning sentinel that the wiring helper
    converts into a warning + fallback to fixed-epoch training.
    """
    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_no_marker_row()] * 5)
    tok = _FakeTokenizer(marker_text=" ※")

    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        data, tok, [MARKER_TOK], max_rows=32, max_length=256
    )
    assert n_rows == 0
    assert input_ids is None
    assert attention_mask is None
    assert marker_positions is None


def test_build_source_probe_missing_file_raises(tmp_path: Path):
    tok = _FakeTokenizer(marker_text=" ※")
    with pytest.raises(FileNotFoundError):
        build_source_probe_from_data(
            tmp_path / "nope.jsonl", tok, [MARKER_TOK], max_rows=4, max_length=256
        )


def test_build_source_probe_empty_marker_ids_raises(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_marker_row()])
    tok = _FakeTokenizer(marker_text=" ※")
    with pytest.raises(ValueError, match="marker_token_ids"):
        build_source_probe_from_data(data, tok, [], max_rows=4, max_length=256)


# ---------------------------------------------------------------------------
# 3. train_lora wiring via _maybe_attach_marker_band_stop
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """Capture ``add_callback`` invocations to verify wiring."""

    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_callback(self, cb: Any) -> None:
        self.callbacks.append(cb)


def _write_marker_data(tmp_path: Path) -> Path:
    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_marker_row()] * 5)
    return data


def test_attach_no_op_when_marker_mode_off(tmp_path: Path):
    """Non-marker run → ``_maybe_attach_marker_band_stop`` is a strict no-op.

    This is the #1 acceptance criterion from the brief: callers that don't
    enable marker mode see byte-identical behavior, i.e. no
    MarkerBandStopCallback ever attached to the trainer.
    """
    data = _write_marker_data(tmp_path)
    trainer = _FakeTrainer()
    tok = _FakeTokenizer(marker_text=" ※")

    cfg = TrainLoraConfig(
        marker_only_loss=False,  # NOT marker mode
        marker_band_stop=True,  # ignored when marker_only_loss=False
    )
    _maybe_attach_marker_band_stop(trainer, tok, cfg, str(data))

    assert trainer.callbacks == []
    assert not any(isinstance(c, MarkerBandStopCallback) for c in trainer.callbacks)


def test_attach_no_op_when_band_stop_opted_out(tmp_path: Path):
    """Marker mode + ``marker_band_stop=False`` (geometry-at-ceiling opt-out)."""
    data = _write_marker_data(tmp_path)
    trainer = _FakeTrainer()
    tok = _FakeTokenizer(marker_text=" ※")

    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_band_stop=False,  # explicit opt-out
        marker_text=" ※",
    )
    _maybe_attach_marker_band_stop(trainer, tok, cfg, str(data))

    assert trainer.callbacks == []


def test_attach_callback_when_marker_mode_on(tmp_path: Path):
    """Marker mode + default band-stop + marker-bearing data → callback attached."""
    data = _write_marker_data(tmp_path)
    trainer = _FakeTrainer()
    tok = _FakeTokenizer(marker_text=" ※")

    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_band_stop=True,
        marker_text=" ※",
        marker_band_low_nats=5.0,
        marker_band_high_nats=12.0,
        marker_band_eval_every_steps=10,
        marker_band_min_steps=20,
        marker_band_probe_max_rows=4,
    )
    _maybe_attach_marker_band_stop(trainer, tok, cfg, str(data))

    assert len(trainer.callbacks) == 1
    cb = trainer.callbacks[0]
    assert isinstance(cb, MarkerBandStopCallback)
    # The probe came from the actual data file.
    assert cb.probe_input_ids.shape[0] == 4
    assert cb.marker_token_ids == [MARKER_TOK]
    assert cb.low_nats == 5.0
    assert cb.high_nats == 12.0
    assert cb.eval_every_steps == 10
    assert cb.min_steps == 20


def test_attach_no_op_when_marker_data_missing(tmp_path: Path, caplog):
    """Marker mode + marker_band_stop=True + ZERO marker rows in data → warn + skip.

    The wiring helper must NOT crash here — falling back to fixed-epoch
    training is the intended behavior (the warning makes the regression
    visible in the log), so a misconfigured data path doesn't block the
    run entirely.
    """
    import logging

    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_no_marker_row()] * 3)

    trainer = _FakeTrainer()
    tok = _FakeTokenizer(marker_text=" ※")

    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_band_stop=True,
        marker_text=" ※",
    )

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        _maybe_attach_marker_band_stop(trainer, tok, cfg, str(data))

    assert trainer.callbacks == []
    assert any(
        "MarkerBandStopCallback" in rec.message and "0 marker-bearing rows" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# 4. Callback constructor invariants
# ---------------------------------------------------------------------------


def test_callback_constructor_rejects_empty_marker_ids():
    import torch

    with pytest.raises(ValueError, match="marker_token_ids"):
        MarkerBandStopCallback(
            marker_token_ids=[],
            probe_input_ids=torch.zeros((1, 4), dtype=torch.long),
            probe_marker_positions=torch.zeros((1,), dtype=torch.long),
            probe_attention_mask=torch.ones((1, 4), dtype=torch.long),
        )


def test_callback_constructor_rejects_inverted_band():
    import torch

    with pytest.raises(ValueError, match="strictly less than"):
        MarkerBandStopCallback(
            marker_token_ids=[MARKER_TOK],
            probe_input_ids=torch.zeros((1, 4), dtype=torch.long),
            probe_marker_positions=torch.zeros((1,), dtype=torch.long),
            probe_attention_mask=torch.ones((1, 4), dtype=torch.long),
            low_nats=12.0,
            high_nats=5.0,
        )


def test_callback_disables_when_max_steps_below_min_steps(caplog):
    """Round-2 safety guard: short runs that cannot reach the band no-op
    + warn, never silently skip the band-stop yet keep checking each step.

    With ``max_steps < min_steps`` the band-stop predicate (``step >=
    min_steps``) would block EVERY in-band reading. The callback must
    set its phase-disabled flag in ``on_train_begin`` and warn so the
    operator sees the regression instead of a never-fire silent default.
    """
    import logging
    from types import SimpleNamespace

    import torch

    cb = MarkerBandStopCallback(
        marker_token_ids=[MARKER_TOK],
        probe_input_ids=torch.zeros((1, 4), dtype=torch.long),
        probe_marker_positions=torch.zeros((1,), dtype=torch.long),
        probe_attention_mask=torch.ones((1, 4), dtype=torch.long),
        min_steps=20,
    )
    args = SimpleNamespace()
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    state = SimpleNamespace(global_step=0, max_steps=10)  # < min_steps

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.eval.callbacks"):
        cb.on_train_begin(args, state, control)

    assert cb._disabled_too_short is True
    assert any(
        "max_steps=10 < min_steps=20" in rec.message
        and "Disabling the band-stop for this phase" in rec.message
        for rec in caplog.records
    )

    # And on_step_end is a strict no-op: control is untouched even if
    # delta would otherwise be inside the band.
    state.global_step = 10  # would clear the every-10-step gate
    cb.on_step_end(args, state, control, model=None)
    assert control.should_training_stop is False
    assert control.should_save is False


def test_callback_drops_overlong_row_with_warning(tmp_path: Path, caplog):
    """MAJOR 3 regression: a row whose fused render exceeds ``max_length``
    is DROPPED (with a warning), NOT front-truncated.

    Front-truncating would discard the source system prompt and re-root
    the context — the same off-distribution failure mode the BLOCKING 1
    fix addresses. The probe builder must drop the row entirely.
    """
    import logging

    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [_marker_row()] * 3)
    tok = _FakeTokenizer(marker_text=" ※")

    # The fake's fused render is 8 tokens for a marker row. Set
    # max_length below that to force a drop.
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        input_ids, _attention_mask, _marker_positions, n_rows = build_source_probe_from_data(
            data, tok, [MARKER_TOK], max_rows=3, max_length=3
        )
    assert n_rows == 0
    assert input_ids is None
    # The warning makes the drop visible to the operator.
    assert any(
        "dropping row" in rec.message and "front-truncating would re-root" in rec.message
        for rec in caplog.records
    )


def test_attach_uses_max_length_floor_of_2048(tmp_path: Path):
    """``cfg.marker_band_probe_max_length=None`` → wiring helper passes
    ``max(cfg.max_length, 2048)`` so short training contexts (e.g. 1024)
    don't force a probe drop. The current canonical 7B-Qwen marker rig
    uses 1024-token training but needs 2048+ for the marker DV probe.
    """
    # Easiest end-to-end check: build a giant fake row (chat-template render
    # length > 1024 but <= 2048) and confirm it survives with the default
    # cfg.marker_band_probe_max_length=None. We do this by overriding the
    # fake tokenizer's apply_chat_template to pad with filler tokens.
    big_row_marker_row = {
        "prompt": [
            {"role": "system", "content": "you are X"},
            {"role": "user", "content": "q"},
        ],
        "completion": [{"role": "assistant", "content": "answer ※"}],
    }

    class _BigFake(_FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            base = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            # Insert 1500 filler tokens BEFORE the marker so the fused row
            # comes out > 1024 but < 2048. We splice before the marker so
            # the marker location math still works.
            if MARKER_TOK in base:
                idx = base.index(MARKER_TOK)
                filler = [9000] * 1500  # arbitrary non-special token id
                return base[:idx] + filler + base[idx:]
            return base

    data = tmp_path / "train.jsonl"
    _write_jsonl(data, [big_row_marker_row])
    tok = _BigFake(marker_text=" ※")

    trainer = _FakeTrainer()
    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_band_stop=True,
        marker_text=" ※",
        max_length=1024,  # training-side budget; intentionally below 2048
        # marker_band_probe_max_length=None (default) → resolves to max(1024, 2048) = 2048
    )
    _maybe_attach_marker_band_stop(trainer, tok, cfg, str(data))

    # The probe should have attached (row fits in 2048).
    assert len(trainer.callbacks) == 1
    cb = trainer.callbacks[0]
    assert cb.probe_input_ids.shape[0] == 1


def test_callback_constructor_rejects_zero_eval_every_steps():
    import torch

    with pytest.raises(ValueError, match="eval_every_steps"):
        MarkerBandStopCallback(
            marker_token_ids=[MARKER_TOK],
            probe_input_ids=torch.zeros((1, 4), dtype=torch.long),
            probe_marker_positions=torch.zeros((1,), dtype=torch.long),
            probe_attention_mask=torch.ones((1, 4), dtype=torch.long),
            eval_every_steps=0,
        )
