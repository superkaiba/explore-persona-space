# em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Task #477 — fail-loud guard for the silent LoRA-not-applied regression.

Pins the three behavioural verdicts of
``assert_adapter_actually_applied`` against synthetic per-probe records +
synthetic adapter ``adapter_model.safetensors`` files:

  1. Real trained adapter + ΔG signal across the panel
        → ``"pass_real_signal"``, no raise.
  2. Real trained adapter + ΔG ≈ 0 everywhere + 0 emission
        → ``LoRANotAppliedError`` (the #477 v4/v6 regression class).
  3. Genuine-floor adapter (B-norm ≈ 0) + ΔG ≈ 0 everywhere
        → ``"pass_genuine_floor"``, no raise (this is a real measurement
        of an untrained LoRA, not a regression).

All tests are CPU-only, sub-second; the safetensors files are written under a
``tmp_path`` per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _write_lora_safetensors(adapter_dir: Path, *, b_norm: float) -> Path:
    """Write a minimal PEFT-style ``adapter_model.safetensors`` to ``adapter_dir``.

    Contains one ``lora_A`` (norm ~1) and one ``lora_B`` whose Frobenius norm is
    exactly ``b_norm``. The guard reads ``lora_B*`` only; ``lora_A`` is included
    so the file shape resembles a real PEFT adapter dump and the test is
    explicit about what gets ignored.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    A = torch.randn(16, 4)
    if b_norm == 0.0:
        B = torch.zeros(4, 16)
    else:
        # Scale a unit-normish tensor so its Frobenius norm equals b_norm exactly.
        B0 = torch.randn(4, 16)
        B = B0 * (b_norm / float(B0.norm()))
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": A,
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": B,
    }
    out = adapter_dir / "adapter_model.safetensors"
    save_file(tensors, str(out))
    return out


def _records_with_constant_dg(
    *,
    n_personas: int,
    n_questions: int,
    delta_g: float,
    emit_share: float,
) -> tuple[dict, dict]:
    """Build (g_records, b_records) where every probe has the same ΔG + the same
    argmax_marker share.

    g leaf logp = ``delta_g``; b leaf logp = 0.0 → ΔG = delta_g per probe.
    ``emit_share`` is the fraction of probes with ``argmax_marker=True``.
    """
    g: dict[str, dict[str, dict[str, float | bool]]] = {}
    b: dict[str, dict[str, dict[str, float | bool]]] = {}
    n_total = n_personas * n_questions
    n_emit_target = round(emit_share * n_total)
    n_emit_so_far = 0
    for pi in range(n_personas):
        persona = f"persona_{pi}"
        g[persona] = {}
        b[persona] = {}
        for qi in range(n_questions):
            q = f"q_{qi}"
            emit = n_emit_so_far < n_emit_target
            if emit:
                n_emit_so_far += 1
            g[persona][q] = {"logp": float(delta_g), "argmax_marker": emit}
            b[persona][q] = {"logp": 0.0, "argmax_marker": False}
    return g, b


def test_guard_raises_on_silent_lora_not_applied(tmp_path: Path) -> None:
    """Real adapter (B-norm 3.0, well above floor) + ΔG ≈ 0 everywhere + 0 emit
    → the #477 regression class; the guard MUST raise LoRANotAppliedError."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        LoRANotAppliedError,
        assert_adapter_actually_applied,
    )

    adapter_dir = tmp_path / "trained_adapter"
    _write_lora_safetensors(adapter_dir, b_norm=3.0)
    g, b = _records_with_constant_dg(n_personas=4, n_questions=3, delta_g=0.0, emit_share=0.0)
    with pytest.raises(LoRANotAppliedError, match=r"silent-LoRA-not-applied|LoRA-not-applied"):
        assert_adapter_actually_applied(
            adapter_dir=adapter_dir,
            g_records=g,
            b_records=b,
            cell_label="test_silent_lora",
        )


def test_guard_passes_on_real_signal(tmp_path: Path) -> None:
    """Real adapter (B-norm 3.0) + ΔG = 5 nats per probe → the eval reads a real
    signal; the guard MUST pass with verdict ``pass_real_signal``."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        assert_adapter_actually_applied,
    )

    adapter_dir = tmp_path / "trained_adapter"
    _write_lora_safetensors(adapter_dir, b_norm=3.0)
    g, b = _records_with_constant_dg(n_personas=4, n_questions=3, delta_g=5.0, emit_share=0.5)
    diag = assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g,
        b_records=b,
        cell_label="test_real_signal",
    )
    assert diag["guard_verdict"] == "pass_real_signal"
    assert diag["max_abs_delta_g_nats"] == pytest.approx(5.0, abs=1e-6)
    assert diag["adapter_b_max_norm"] == pytest.approx(3.0, abs=1e-3)
    assert diag["n_probes"] == 12
    assert diag["n_emit"] == 6


def test_guard_passes_on_genuine_floor_adapter(tmp_path: Path) -> None:
    """Adapter with B-norm 0 (untrained / collapsed) + ΔG ≈ 0 everywhere → NOT a
    regression; the guard MUST pass with verdict ``pass_genuine_floor``."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        assert_adapter_actually_applied,
    )

    adapter_dir = tmp_path / "untrained_adapter"
    _write_lora_safetensors(adapter_dir, b_norm=0.0)
    g, b = _records_with_constant_dg(n_personas=4, n_questions=3, delta_g=0.0, emit_share=0.0)
    diag = assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g,
        b_records=b,
        cell_label="test_floor_adapter",
    )
    assert diag["guard_verdict"] == "pass_genuine_floor"
    assert diag["adapter_b_max_norm"] == pytest.approx(0.0, abs=1e-6)


def test_guard_passes_when_emission_present(tmp_path: Path) -> None:
    """Real adapter, ΔG ≈ 0 everywhere, but n_emit > 0 → the LoRA is at least
    partially expressed at decode time; this is NOT the #477 regression
    (which requires uniformly-zero emission). Guard MUST pass with verdict
    ``pass_some_emission`` and log a warning."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        assert_adapter_actually_applied,
    )

    adapter_dir = tmp_path / "trained_adapter"
    _write_lora_safetensors(adapter_dir, b_norm=2.5)
    g, b = _records_with_constant_dg(n_personas=2, n_questions=4, delta_g=0.0, emit_share=0.25)
    diag = assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g,
        b_records=b,
        cell_label="test_some_emission",
    )
    assert diag["guard_verdict"] == "pass_some_emission"
    assert diag["n_emit"] == 2
    assert diag["n_probes"] == 8


def test_b_matrix_norm_missing_weights_raises(tmp_path: Path) -> None:
    """Missing ``adapter_model.safetensors`` → fail loud (not silent 0.0)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        b_matrix_frobenius_norm,
    )

    adapter_dir = tmp_path / "empty_adapter"
    adapter_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"adapter_model\.safetensors"):
        b_matrix_frobenius_norm(adapter_dir)


def test_guard_mismatched_records_raises(tmp_path: Path) -> None:
    """g_records and b_records must cover the SAME (persona, q) grid; the guard
    raises KeyError on disagreement rather than silently averaging over a
    truncated grid."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        assert_adapter_actually_applied,
    )

    adapter_dir = tmp_path / "adapter"
    _write_lora_safetensors(adapter_dir, b_norm=1.0)
    g, b = _records_with_constant_dg(n_personas=2, n_questions=2, delta_g=1.0, emit_share=0.0)
    # Mutate b to drop a persona.
    del b["persona_0"]
    with pytest.raises(KeyError, match=r"persona_0|persona"):
        assert_adapter_actually_applied(
            adapter_dir=adapter_dir,
            g_records=g,
            b_records=b,
            cell_label="test_mismatched",
        )
