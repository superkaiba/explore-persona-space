# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Task #504 round-4 — B-matrix-only eval guard regression tests.

Pins the contract for the round-4 BLOCKER #1 fix (concern_id
``phase0-guard-too-strict``): ``assert_adapter_actually_applied`` is now
B-matrix-only. The historical three-clause RAISE (b-norm > floor AND
max|ΔG| < eps AND n_emit == 0 → ``LoRANotAppliedError``) is dropped, so an
early-checkpoint × gentle-lr cell (lr=1e-5 / r=8 / frac=0.16, the round-3 crash
config) NO LONGER crashes the cell. The picker is responsible for in-band
selection.

CPU-only, sub-second. Writes a tiny safetensors adapter dir under tmp and
exercises the guard's three code paths (above floor, at-floor, below floor).
No GPU/HF/network/vLLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
    DEFAULT_B_NORM_FLOOR,
    LoRANotAppliedError,
    assert_adapter_actually_applied,
    b_matrix_frobenius_norm,
)


def _write_adapter(adapter_dir: Path, b_scale: float) -> None:
    """Write a minimal PEFT-shaped adapter with ``lora_B`` weights scaled by ``b_scale``.

    A scale of 0.0 mimics the freshly-initialized PEFT state (B=0); any value
    above ~1e-3 / sqrt(N) clears the default ``DEFAULT_B_NORM_FLOOR``.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Two layers' worth of lora_A / lora_B; only the B norms matter to the guard.
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(
            8, 16, dtype=torch.float32
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": (
            torch.ones(16, 8, dtype=torch.float32) * b_scale
        ),
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))


def _records_at_floor() -> tuple[
    dict[str, dict[str, dict[str, float | bool]]],
    dict[str, dict[str, dict[str, float | bool]]],
]:
    """Build the (g_records, b_records) pair that crashed in round 3.

    Every probe has ``g.logp == b.logp`` (so max|ΔG| = 0 < 0.5 eps) and
    ``argmax_marker = False`` (so n_emit = 0). Under the old three-clause raise
    this combination (with B-norm > floor) was the regression class; under the
    round-4 redesign the guard returns ``"pass_b_norm_ok"`` without raising.
    """
    g: dict[str, dict[str, dict[str, float | bool]]] = {
        "source": {
            "q1": {"logp": -3.0, "argmax_marker": False},
            "q2": {"logp": -3.0, "argmax_marker": False},
        },
        "bystander_a": {
            "q1": {"logp": -3.0, "argmax_marker": False},
            "q2": {"logp": -3.0, "argmax_marker": False},
        },
    }
    b: dict[str, dict[str, dict[str, float | bool]]] = {
        "source": {
            "q1": {"logp": -3.0, "argmax_marker": False},
            "q2": {"logp": -3.0, "argmax_marker": False},
        },
        "bystander_a": {
            "q1": {"logp": -3.0, "argmax_marker": False},
            "q2": {"logp": -3.0, "argmax_marker": False},
        },
    }
    return g, b


def _records_with_signal() -> tuple[
    dict[str, dict[str, dict[str, float | bool]]],
    dict[str, dict[str, dict[str, float | bool]]],
]:
    """Build (g_records, b_records) with max|ΔG| ~ 10 nats and some emission."""
    g: dict[str, dict[str, dict[str, float | bool]]] = {
        "source": {
            "q1": {"logp": -3.0, "argmax_marker": True},
            "q2": {"logp": -3.5, "argmax_marker": True},
        },
        "bystander_a": {
            "q1": {"logp": -8.0, "argmax_marker": False},
            "q2": {"logp": -8.0, "argmax_marker": False},
        },
    }
    b: dict[str, dict[str, dict[str, float | bool]]] = {
        "source": {
            "q1": {"logp": -13.0, "argmax_marker": False},
            "q2": {"logp": -13.0, "argmax_marker": False},
        },
        "bystander_a": {
            "q1": {"logp": -13.0, "argmax_marker": False},
            "q2": {"logp": -13.0, "argmax_marker": False},
        },
    }
    return g, b


def test_b_norm_above_floor_with_floor_metric_does_not_raise(tmp_path: Path) -> None:
    """The round-3 crash config: B-matrix > floor, max|ΔG| < eps, n_emit = 0.

    Pre-round-4 this raised ``LoRANotAppliedError`` (false-positive on the
    early-checkpoint × gentle-lr cell). Post-round-4 the guard returns the
    diagnostics dict with verdict ``"pass_b_norm_ok"`` — the picker's
    anti-saturation band [5, 12] nats × [0.1, 0.8] emission decides in-band.
    """
    _write_adapter(tmp_path, b_scale=0.1)  # well above 1e-3 floor
    g, b = _records_at_floor()
    diag = assert_adapter_actually_applied(
        adapter_dir=tmp_path,
        g_records=g,
        b_records=b,
        cell_label="c504v2_smoke_lr1e5_seed42_frac0.16",
    )
    assert diag["guard_verdict"] == "pass_b_norm_ok"
    assert diag["adapter_b_max_norm"] > DEFAULT_B_NORM_FLOOR
    assert diag["max_abs_delta_g_nats"] == pytest.approx(0.0)
    assert diag["n_emit"] == 0
    assert diag["n_probes"] == 4  # 2 personas × 2 questions


def test_b_norm_above_floor_with_real_signal_does_not_raise(tmp_path: Path) -> None:
    """Adapter trained with real signal — guard returns ``"pass_b_norm_ok"``."""
    _write_adapter(tmp_path, b_scale=0.1)
    g, b = _records_with_signal()
    diag = assert_adapter_actually_applied(
        adapter_dir=tmp_path,
        g_records=g,
        b_records=b,
        cell_label="c504v2_smoke_lr3e5_seed42_frac0.5",
    )
    assert diag["guard_verdict"] == "pass_b_norm_ok"
    assert diag["adapter_b_max_norm"] > DEFAULT_B_NORM_FLOOR
    assert diag["max_abs_delta_g_nats"] >= 5.0
    assert diag["n_emit"] >= 1


def test_b_norm_at_floor_returns_genuine_floor(tmp_path: Path) -> None:
    """A freshly-initialized adapter (B=0) is a real measurement, not a regression.

    PEFT initializes ``lora_B`` to zero; a cell that ran but didn't move B is a
    genuine floor outcome. The guard returns ``"pass_genuine_floor"`` and does
    not raise.
    """
    _write_adapter(tmp_path, b_scale=0.0)
    g, b = _records_at_floor()
    diag = assert_adapter_actually_applied(
        adapter_dir=tmp_path,
        g_records=g,
        b_records=b,
        cell_label="c504v2_smoke_lr1e7_seed42_frac0.5",
    )
    assert diag["guard_verdict"] == "pass_genuine_floor"
    assert diag["adapter_b_max_norm"] <= DEFAULT_B_NORM_FLOOR


def test_lora_not_applied_error_class_still_importable() -> None:
    """The exception class is kept for compat with existing imports.

    Several call sites (``scripts/i504_run_cell.py``,
    ``scripts/i504_eval_trajectory.py``, ``scripts/i504_reval_grid.py``)
    historically caught this exception; the class must keep resolving even
    though the round-4 redesign no longer raises it from
    ``assert_adapter_actually_applied``.
    """
    assert issubclass(LoRANotAppliedError, RuntimeError)


def test_b_matrix_frobenius_norm_reads_max_layer(tmp_path: Path) -> None:
    """Sanity: the safetensors reader picks the MAX layer norm, not the mean."""
    _write_adapter(tmp_path, b_scale=0.25)
    n = b_matrix_frobenius_norm(tmp_path)
    # ||0.25 * ones(16, 8)||_F = 0.25 * sqrt(128) ≈ 2.828
    assert n == pytest.approx(0.25 * (16 * 8) ** 0.5, rel=1e-5)
