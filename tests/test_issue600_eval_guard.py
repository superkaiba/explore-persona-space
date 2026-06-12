# ruff: noqa: RUF002  # marker token + em-dash intentional
"""CPU-only tests for the #600 eval-guard wiring + smoke gates (plan §4.6/§4.7).

The #505 (f)/(g) pattern: a POSITIVE control (a structurally-trained adapter
must read ``pass_b_norm_ok``) and a ``use_lora=False`` NEGATIVE control (base
records scored as 'trained' give ΔG ≈ 0 — the guard must classify it
structurally, and the #600 smoke gate (c) must REJECT an untrained
``pass_genuine_floor`` adapter as the positive-control failure it is).
Also covers the adapter-config parity assert (the silent-7-module-degrade
guard, plan §4.5) and the §4.7 gate (a)/(b)/(c)/(g)/(h) classification on
synthetic smoke artifacts (happy path + floor + saturation error paths).

Runs in <10 s on CPU; no model load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
    assert_adapter_actually_applied,
)
from explore_persona_space.experiments.targeted_proximity_600 import (
    EXPECTED_STEPS_PER_EPOCH,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
)
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (
    assert_adapter_config_parity,
    check_smoke_gates_600,
)

QS = ["q1", "q2"]


def _records(delta: float) -> tuple[dict, dict]:
    """(g_records, b_records) with a uniform trained−base ΔG of ``delta``."""
    base = -19.0
    g = {p: {q: {"logp": base + delta, "argmax_marker": False} for q in QS} for p in ("a", "b")}
    b = {p: {q: {"logp": base, "argmax_marker": False} for q in QS} for p in ("a", "b")}
    return g, b


def _write_adapter(tmp_path: Path, b_scale: float) -> Path:
    d = tmp_path / f"adapter_{b_scale}"
    d.mkdir()
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(16, 32),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.full(
                (32, 16), b_scale
            ),
        },
        str(d / "adapter_model.safetensors"),
    )
    return d


def test_positive_control_trained_adapter_passes(tmp_path: Path):
    g, b = _records(delta=8.0)
    diag = assert_adapter_actually_applied(
        adapter_dir=_write_adapter(tmp_path, 0.5), g_records=g, b_records=b, cell_label="pos"
    )
    assert diag["guard_verdict"] == "pass_b_norm_ok"
    assert diag["max_abs_delta_g_nats"] == pytest.approx(8.0)


def test_negative_control_use_lora_false_reads_zero_delta(tmp_path: Path):
    """use_lora=False scoring (g == b) → ΔG 0; structural verdict still b_norm_ok."""
    g, b = _records(delta=0.0)
    diag = assert_adapter_actually_applied(
        adapter_dir=_write_adapter(tmp_path, 0.5), g_records=g, b_records=b, cell_label="neg"
    )
    assert diag["guard_verdict"] == "pass_b_norm_ok"
    assert diag["max_abs_delta_g_nats"] == pytest.approx(0.0)


def test_untrained_adapter_reads_genuine_floor(tmp_path: Path):
    """PEFT B=0 init → structurally empty → pass_genuine_floor (smoke gate (c) FAILS on it)."""
    g, b = _records(delta=0.0)
    diag = assert_adapter_actually_applied(
        adapter_dir=_write_adapter(tmp_path, 0.0), g_records=g, b_records=b, cell_label="floor"
    )
    assert diag["guard_verdict"] == "pass_genuine_floor"


# ── Adapter-config parity (plan §4.5). ───────────────────────────────────────


def _write_adapter_config(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "r": 16,
        "lora_alpha": 32,
        "use_rslora": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save": None,
    }
    cfg.update(overrides)
    d = tmp_path / "adapter_cfg"
    d.mkdir(exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return d


def test_adapter_parity_passes_on_pinned_geometry(tmp_path: Path):
    out = assert_adapter_config_parity(_write_adapter_config(tmp_path))
    assert out["r"] == 16


def test_adapter_parity_rejects_seven_module_degrade(tmp_path: Path):
    d = _write_adapter_config(
        tmp_path,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    with pytest.raises(RuntimeError, match="parity FAILED"):
        assert_adapter_config_parity(d)


def test_adapter_parity_rejects_wrong_rank(tmp_path: Path):
    with pytest.raises(RuntimeError, match="parity FAILED"):
        assert_adapter_config_parity(_write_adapter_config(tmp_path, r=32))


# ── §4.7 smoke gates on synthetic artifacts. ─────────────────────────────────


def _leaf(argmax: bool = False, n_marker: int = 0) -> dict:
    return {
        "g_logp": -8.0,
        "b_logp": -19.0,
        "delta_g": 11.0,
        "argmax_marker": argmax,
        "n_marker_in_R": n_marker,
        "r_collapsed": False,
        "kl": 0.1,
    }


def _smoke_artifacts(
    tmp_path: Path,
    *,
    source_dg: float = 10.0,
    source_g_logp: float = -9.0,
    bystander_argmax: bool = False,
    guard_verdict: str = "pass_b_norm_ok",
    manifest_verdict: str = "pass",
    n_band_records: int = 5,
    realized_step: int = EXPECTED_STEPS_PER_EPOCH,
) -> dict:
    ck = {
        "frac": 1.0,
        "step": realized_step,
        "adapter_path": "x",
        "source_self": {
            "delta_g_mean": source_dg,
            "g_logp_mean": source_g_logp,
            "b_logp_mean": source_g_logp - source_dg,
            "emission_p": 0.0,
            "r_collapsed": False,
        },
        "held_out": {
            "villain": {q: _leaf() for q in QS},
            "bystander_1": {q: _leaf(argmax=bystander_argmax) for q in QS},
            "bystander_2": {q: _leaf(argmax=bystander_argmax) for q in QS},
        },
        "eval_guard_diagnostic": {"guard_verdict": guard_verdict},
        "source_manifest_check": {"guard_verdict": manifest_verdict},
    }
    traj = {"source": "villain", "logit_fields": True, "checkpoints": [ck]}
    traj_path = tmp_path / "trajectory.json"
    traj_path.write_text(json.dumps(traj))
    band = {
        "records": [{"step": 10 * (i + 1)} for i in range(n_band_records)],
        "delta_nats": [1.0 * (i + 1) for i in range(n_band_records)],
    }
    band_path = tmp_path / "band_trajectory.json"
    band_path.write_text(json.dumps(band))
    ckpt_index = {
        f"{f:.2f}": {"step": realized_step if f == 1.0 else int(f * realized_step), "path": "x"}
        for f in TRAJECTORY_CHECKPOINT_FRACTIONS
    }
    return {
        "trajectory_path": traj_path,
        "band_trajectory_path": band_path,
        "checkpoint_index": ckpt_index,
    }


def _run_gates(tmp_path: Path, **kw) -> dict:
    art = _smoke_artifacts(tmp_path, **kw)
    return check_smoke_gates_600(
        trajectory_path=art["trajectory_path"],
        band_trajectory_path=art["band_trajectory_path"],
        verify_payload={"verdict": "pass"},
        collator_payload={"verdict": "pass"},
        checkpoint_index=art["checkpoint_index"],
        expected_steps=EXPECTED_STEPS_PER_EPOCH,
        panel_personas=["qwen_default", "a", "b", "c"],
        smoke_out_path=tmp_path / "smoke" / "smoke_gate.json",
    )


def test_smoke_gates_all_pass_happy_path(tmp_path: Path):
    out = _run_gates(tmp_path)
    assert out["all_gates_passed"], out
    assert (tmp_path / "smoke" / "smoke_gate.json").exists()


def test_smoke_gates_fail_on_floor(tmp_path: Path):
    out = _run_gates(tmp_path, source_dg=0.8)
    assert not out["gate_a_band"]
    assert out["floor_failed"] and not out["saturation_failed"]
    assert not out["all_gates_passed"]


def test_smoke_gates_fail_on_saturation(tmp_path: Path):
    out = _run_gates(tmp_path, source_dg=21.0, source_g_logp=-0.01, bystander_argmax=True)
    assert not out["gate_a_band"]
    assert not out["gate_b_sub_saturation"]
    assert out["saturation_failed"]


def test_smoke_gates_fail_on_genuine_floor_guard(tmp_path: Path):
    """Gate (c) positive control: an untrained adapter must NOT pass smoke."""
    out = _run_gates(tmp_path, guard_verdict="pass_genuine_floor")
    assert not out["gate_c_eval_guard_positive_control"]
    assert not out["all_gates_passed"]


def test_smoke_gates_fail_on_unmatched_steps(tmp_path: Path):
    """Gate (g): a band-stopped / unmatched run (≠63 steps) must fail smoke."""
    out = _run_gates(tmp_path, realized_step=40)
    assert not out["gate_g_telemetry"]
    assert not out["all_gates_passed"]
