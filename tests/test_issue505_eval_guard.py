"""Task #505 §5.5 gate (g) — the cherry-picked ``assert_adapter_actually_applied``
negative-control unit test.

The plan §5.5 (g) requires a pytest that:

  1. Loads a hand-built trajectory in the silent-LoRA-not-applied regime
     (B-norm > floor + max|ΔG| ≈ 0 across the panel + n_emit = 0) and asserts
     the guard FAILS LOUD (raises ``LoRANotAppliedError``).

  2. Loads a hand-built trajectory in the genuine-floor regime (B-norm ≈ 0,
     ΔG ≈ 0 across the panel) and asserts the guard PASSES (no raise).

  3. Loads a hand-built trajectory in the real-signal regime (B-norm > floor,
     max|ΔG| ≫ eps) and asserts the guard PASSES.

The B-matrix Frobenius norm is read from a synthetic
``adapter_model.safetensors`` we write to a tmp dir; the records dicts are
synthesized in the rig's expected shape so the guard's
``_aggregate_records_for_guard`` can consume them directly.

This file also asserts the cherry-pick landed: the
``contrastive_neg_geometry_472.eval_guard`` module imports cleanly and
exposes ``assert_adapter_actually_applied`` + ``LoRANotAppliedError``. The
plan §10 step 0 makes the cherry-pick mandatory before any sweep.

Runs in <1 s on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# ── Cherry-pick landed import. ──────────────────────────────────────────────


def test_eval_guard_module_importable():
    """Plan §10 step 0: the #477 guard MUST be cherry-picked onto issue-505."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    assert hasattr(eval_guard, "assert_adapter_actually_applied")
    assert hasattr(eval_guard, "LoRANotAppliedError")
    assert hasattr(eval_guard, "b_matrix_frobenius_norm")


# ── Helpers: synthesize a safetensors adapter + records dicts. ──────────────


def _write_adapter(tmp_dir: Path, b_norm: float) -> Path:
    """Write a minimal PEFT-style adapter_model.safetensors with one lora_B tensor.

    The tensor is a vector of length 8 with uniform entries scaled so the
    Frobenius norm equals ``b_norm`` (``sqrt(8) * x = b_norm`` → ``x = b_norm / sqrt(8)``).
    """
    adapter_dir = tmp_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if b_norm <= 0:
        b_tensor = torch.zeros(8, dtype=torch.float32)
    else:
        x = b_norm / (8**0.5)
        b_tensor = torch.full((8,), x, dtype=torch.float32)
    # A dummy A tensor (PEFT also stores lora_A; the guard reads only lora_B).
    a_tensor = torch.randn(8, dtype=torch.float32)
    save_file(
        {
            "base_model.model.layer.0.lora_B.default.weight": b_tensor,
            "base_model.model.layer.0.lora_A.default.weight": a_tensor,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return adapter_dir


def _records(
    *,
    personas: list[str],
    questions: list[str],
    g_logp: float,
    b_logp: float,
    emit: bool,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Build a (persona, q) record dict in the rig's expected shape.

    The ``emit`` flag controls the argmax_marker booleans; ``b_logp`` is unused
    in this fixture (the caller passes the base records separately, but we
    accept it for symmetry with the eventual base-records caller).
    """
    del b_logp  # symmetry-only placeholder
    return {p: {q: {"logp": g_logp, "argmax_marker": emit} for q in questions} for p in personas}


# ── (g) — negative-control: SILENT-LORA-NOT-APPLIED regression raises. ──────


def test_guard_raises_on_silent_lora_not_applied(tmp_path):
    """The #477 v4/v6 regression class — adapter genuinely trained + ΔG ≈ 0 +
    emission 0 everywhere — MUST raise ``LoRANotAppliedError`` per the §5.5
    gate (g) negative-control."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = _write_adapter(tmp_path, b_norm=3.0)  # well above floor 1e-3
    personas = ["b1", "b2", "b3"]
    questions = ["q1", "q2"]
    # Same logp in g and b → max|ΔG| = 0; emission false everywhere.
    g = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)
    b = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)

    with pytest.raises(eval_guard.LoRANotAppliedError, match="LoRA-not-applied regression"):
        eval_guard.assert_adapter_actually_applied(
            adapter_dir=adapter_dir,
            g_records=g,
            b_records=b,
            cell_label="test_neg_control",
        )


# ── PASS cases: genuine floor + real signal. ────────────────────────────────


def test_guard_passes_on_genuine_floor(tmp_path):
    """B-norm at/under floor → the adapter is genuinely untrained, ΔG≈0 is a
    real measurement, NOT the regression."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = _write_adapter(tmp_path, b_norm=0.0)  # zero — clean PEFT init
    personas = ["b1"]
    questions = ["q1"]
    g = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)
    b = _records(personas=personas, questions=questions, g_logp=-15.5, b_logp=-15.5, emit=False)
    diag = eval_guard.assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g,
        b_records=b,
        cell_label="test_floor",
    )
    assert diag["guard_verdict"] == "pass_genuine_floor"


def test_guard_passes_on_real_signal(tmp_path):
    """B-norm > floor AND max|ΔG| > eps → adapter applied, eval reads real signal."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = _write_adapter(tmp_path, b_norm=3.0)
    # Source persona reads +5 nat ΔG — clear signal that the LoRA is applied.
    g_records = {
        "b1": {
            "q1": {"logp": -10.0, "argmax_marker": False},
            "q2": {"logp": -10.5, "argmax_marker": False},
        },
        "src": {
            "q1": {"logp": -10.5, "argmax_marker": True},
            "q2": {"logp": -10.6, "argmax_marker": True},
        },
    }
    b_records = {
        "b1": {
            "q1": {"logp": -15.0, "argmax_marker": False},
            "q2": {"logp": -15.5, "argmax_marker": False},
        },
        "src": {
            "q1": {"logp": -15.5, "argmax_marker": False},
            "q2": {"logp": -15.6, "argmax_marker": False},
        },
    }
    diag = eval_guard.assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g_records,
        b_records=b_records,
        cell_label="test_real_signal",
    )
    assert diag["guard_verdict"] == "pass_real_signal"
    assert diag["max_abs_delta_g_nats"] >= 0.5
    assert diag["n_emit"] >= 1


def test_guard_b_norm_reader_returns_zero_for_no_lora_b(tmp_path):
    """The b_matrix_frobenius_norm reader returns 0.0 if no lora_B keys exist
    (defensive — treats non-LoRA adapters as genuine floor)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_guard

    adapter_dir = tmp_path / "no_lora_b"
    adapter_dir.mkdir()
    save_file(
        {"base_model.model.layer.0.lora_A.default.weight": torch.randn(4)},
        str(adapter_dir / "adapter_model.safetensors"),
    )
    norm = eval_guard.b_matrix_frobenius_norm(adapter_dir)
    assert norm == 0.0
