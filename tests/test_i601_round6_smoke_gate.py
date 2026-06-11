# em-dash / minus sign / Greek delta intentional
"""Task #601 round-6 blocker regression: smoke-gate same-gauge comparison.

Round-5 review BLOCKER ``smoke-gate-cross-gauge-inloop-vs-onpolicy``: smoke
check 3 compared the in-loop band trajectory (LIVE training model — rsLoRA
alpha/sqrt(r) ~= 11.31) against the staged on-policy read (classic alpha/r =
2.0) at a 1-nat tolerance. On the same weights the live-vs-classic gap is
5-15 nats when the implant took, so the assert failed BY CONSTRUCTION on any
working smoke cell and the launch died at the smoke gate after burning phase0
plus the full smoke-cell train.

Round-6 fix (pinned here, mirroring the round-3 fixture pattern):

1. Check 3 is a SAME-GAUGE pair — on-policy vLLM source ΔG vs the SAME
   terminal checkpoint's Phase-B teacher-forced HF read, both staged classic,
   both already in ``trajectory.json`` — at the plan §12 assumption-16
   admission threshold (2 nats).
2. The in-loop band terminal value is recorded telemetry only (gauge-labeled,
   never asserted cross-gauge).
3. A working-implant fixture (live-vs-classic gap 13.8 nats) PASSES the new
   gate while the retired cross-gauge form (|inloop - onpolicy| <= 1 nat)
   would have FAILED it — the launch-killing regression, pinned.

Also pins the round-6 gauge-provenance major (in-loop surfaces labeled by
gauge) and the NaN→None minor in the phase0 tables.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH_PY = REPO_ROOT / "scripts" / "dispatch_neg_setpoint_601.py"

SMOKE_CELL = "ratio4to1_100p400n"
SMOKE_SEED = 42

# Realistic working-implant numbers (round-5 forensics): live in-loop terminal
# ΔG in the low 20s, staged classic on-policy ~9, staged classic teacher-forced
# within cross-engine noise of the on-policy read.
INLOOP_LIVE_DELTA = 22.8
ONPOLICY_DELTA = 9.0
TF_HF_G = -10.4  # logp_hf_g_mean
TF_HF_B = -19.0  # logp_hf_b_mean -> tf delta = 8.6, |9.0 - 8.6| = 0.4 <= 2.0
RETIRED_CROSS_GAUGE_TOL = 1.0  # the round-5 INLOOP_VS_ONPOLICY_TOL_NATS


def _load_dispatch_module():
    spec = importlib.util.spec_from_file_location("dispatch_601_under_test", DISPATCH_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _expected_steps() -> int:
    from explore_persona_space.experiments.neg_setpoint_601 import cell_by_slug

    return cell_by_slug(SMOKE_CELL).expected_steps


def _build_smoke_slab(
    slab: Path,
    runs: Path,
    *,
    onpolicy_delta: float = ONPOLICY_DELTA,
    tf_hf_g: float = TF_HF_G,
    tf_hf_b: float = TF_HF_B,
    inloop_delta: float = INLOOP_LIVE_DELTA,
    band_gauge: dict | None = None,
) -> None:
    """All artifacts ``_smoke_gate`` reads, for one completed smoke cell."""
    t = _expected_steps()
    key = f"{SMOKE_CELL}_seed{SMOKE_SEED}"
    run_dir = runs / key
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint_index.json").write_text(json.dumps({"1.0000": {"step": t}}))

    cell_dir = slab / "phase1" / key
    cell_dir.mkdir(parents=True, exist_ok=True)
    leaf = {"z_marker_g": -3.0, "z_eos_g": 5.0, "logZ_g": 7.0, "kl": 0.4}
    (cell_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "frac": 1.0,
                        "step": t,
                        "source_self": {
                            "delta_g_mean": onpolicy_delta,
                            "z_marker_g_mean": 2.0,
                            "z_marker_b_mean": -10.0,
                            "z_eos_g_mean": 6.0,
                            "z_eos_b_mean": 8.0,
                            "logZ_g_mean": 9.0,
                            "logp_hf_g_mean": tf_hf_g,
                            "logp_hf_b_mean": tf_hf_b,
                        },
                        "held_out": {"persona_a": {"q1": dict(leaf), "q2": dict(leaf)}},
                    }
                ]
            }
        )
    )
    band_payload = {
        "schema": "marker_band_trajectory_v1",
        "steps": [10, t],
        "z_marker_trained": [1.0, 12.0],
        "z_eos_trained": [8.0, 7.0],
        "logZ_trained": [9.0, 12.5],
        "log_p_base": [-19.0, -19.0],
        "delta_nats": [4.0, inloop_delta],
    }
    if band_gauge is not None:
        band_payload["gauge"] = band_gauge
    (cell_dir / "inloop_band_trajectory.json").write_text(json.dumps(band_payload))
    (cell_dir / "rowtype_ce.json").write_text(
        json.dumps({"steps": [10], "pos_marker_ce": [2.1], "neg_trailing_ce": [0.3]})
    )


# ── 1. Same-gauge check 3: both sides of the 2-nat tolerance ─────────────────


def test_smoke_gate_passes_same_gauge_within_tol(tmp_path: Path) -> None:
    """Working implant: on-policy 9.0 vs tf 8.6 (same gauge, diff 0.4) PASSES,
    with the live in-loop value 13.8 nats away — the round-5 launch-killer."""
    mod = _load_dispatch_module()
    slab, runs = tmp_path / "slab", tmp_path / "runs"
    _build_smoke_slab(slab, runs)
    gate = mod._smoke_gate(slab, runs)
    check = gate["checks"]["onpolicy_vs_tf_same_gauge"]
    assert check["ok"] is True
    assert check["tol_nats"] == mod.ONPOLICY_VS_TF_SAMEGAUGE_TOL_NATS == 2.0
    assert math.isclose(check["tf_hf_terminal_delta"], TF_HF_G - TF_HF_B)
    assert gate["smoke_gate_pass"] is True


@pytest.mark.parametrize(
    ("onpolicy_delta", "expect_ok"),
    [
        # tf delta is fixed at 8.6; vary the on-policy side around the 2-nat tol.
        (8.6 + 1.9, True),  # just inside
        (8.6 + 2.0, True),  # boundary (<=)
        (8.6 + 2.5, False),  # just outside
        (8.6 - 2.5, False),  # outside, other direction
    ],
)
def test_smoke_gate_same_gauge_tolerance_both_sides(
    tmp_path: Path, onpolicy_delta: float, expect_ok: bool
) -> None:
    mod = _load_dispatch_module()
    slab, runs = tmp_path / "slab", tmp_path / "runs"
    _build_smoke_slab(slab, runs, onpolicy_delta=onpolicy_delta)
    gate = mod._smoke_gate(slab, runs)
    assert gate["checks"]["onpolicy_vs_tf_same_gauge"]["ok"] is expect_ok
    assert gate["smoke_gate_pass"] is expect_ok


def test_smoke_gate_fails_loud_when_phase_b_fields_missing(tmp_path: Path) -> None:
    """No Phase-B HF read (logp_hf_*_mean absent) -> check 3 cannot attest the
    eval path and the gate FAILS (never a silent pass on missing evidence)."""
    mod = _load_dispatch_module()
    slab, runs = tmp_path / "slab", tmp_path / "runs"
    _build_smoke_slab(slab, runs)
    key = f"{SMOKE_CELL}_seed{SMOKE_SEED}"
    traj_path = slab / "phase1" / key / "trajectory.json"
    traj = json.loads(traj_path.read_text())
    ss = traj["checkpoints"][-1]["source_self"]
    del ss["logp_hf_g_mean"], ss["logp_hf_b_mean"]
    traj_path.write_text(json.dumps(traj))
    gate = mod._smoke_gate(slab, runs)
    check = gate["checks"]["onpolicy_vs_tf_same_gauge"]
    assert check["tf_hf_terminal_delta"] is None
    assert check["ok"] is False
    assert gate["smoke_gate_pass"] is False


# ── 2. The retired cross-gauge form would have failed a working implant ──────


def test_old_cross_gauge_comparison_would_have_failed_a_working_implant(
    tmp_path: Path,
) -> None:
    """Pin the round-5 blocker: on the SAME working-implant fixture the new
    same-gauge gate PASSES while |inloop - onpolicy| blows the retired 1-nat
    cross-gauge tolerance by an order of magnitude. The retired check key and
    constant must be gone from the dispatcher."""
    mod = _load_dispatch_module()
    slab, runs = tmp_path / "slab", tmp_path / "runs"
    _build_smoke_slab(slab, runs)
    gate = mod._smoke_gate(slab, runs)
    assert gate["smoke_gate_pass"] is True
    # The cross-gauge gap that killed the launch by construction:
    cross_gauge_gap = abs(INLOOP_LIVE_DELTA - ONPOLICY_DELTA)
    assert cross_gauge_gap > RETIRED_CROSS_GAUGE_TOL, "fixture must exhibit the gauge gap"
    assert cross_gauge_gap == pytest.approx(13.8)
    # Retired surface is actually retired:
    assert "inloop_vs_onpolicy" not in gate["checks"]
    assert not hasattr(mod, "INLOOP_VS_ONPOLICY_TOL_NATS")


def test_inloop_value_is_telemetry_only_and_never_gates(tmp_path: Path) -> None:
    """An arbitrarily extreme in-loop value cannot flip the gate; it is
    recorded with a gauge label and asserted: False."""
    mod = _load_dispatch_module()
    slab, runs = tmp_path / "slab", tmp_path / "runs"
    _build_smoke_slab(slab, runs, inloop_delta=40.0)
    gate = mod._smoke_gate(slab, runs)
    assert gate["smoke_gate_pass"] is True
    tele = gate["checks"]["inloop_terminal_telemetry"]
    assert tele["inloop_terminal_delta"] == 40.0
    assert tele["asserted"] is False
    # Pre-round-6 band JSON (no gauge key) gets the explanatory default label.
    assert "live-training-model" in json.dumps(tele["gauge"])


def test_inloop_telemetry_passes_band_gauge_through(tmp_path: Path) -> None:
    gauge = {"use_rslora_applied": True, "scaling": "alpha/sqrt(r)", "note": "live-training-model"}
    mod = _load_dispatch_module()
    slab, runs = tmp_path / "slab", tmp_path / "runs"
    _build_smoke_slab(slab, runs, band_gauge=gauge)
    gate = mod._smoke_gate(slab, runs)
    assert gate["checks"]["inloop_terminal_telemetry"]["gauge"] == gauge


# ── 3. Gauge provenance on the in-loop surfaces (round-6 major) ──────────────


def test_resolve_live_gauge_rslora_model() -> None:
    from explore_persona_space.eval.callbacks import _resolve_live_gauge

    model = SimpleNamespace(
        peft_config={"default": SimpleNamespace(use_rslora=True, r=32, lora_alpha=64)}
    )
    gauge = _resolve_live_gauge(model)
    assert gauge["use_rslora_applied"] is True
    assert gauge["scaling"] == "alpha/sqrt(r)"
    assert gauge["lora_r"] == 32 and gauge["lora_alpha"] == 64
    assert gauge["note"] == "live-training-model"


def test_resolve_live_gauge_classic_and_unwrapped() -> None:
    from explore_persona_space.eval.callbacks import _resolve_live_gauge

    classic = SimpleNamespace(
        peft_config={"default": SimpleNamespace(use_rslora=False, r=16, lora_alpha=32)}
    )
    assert _resolve_live_gauge(classic)["scaling"] == "alpha/r"
    unwrapped = _resolve_live_gauge(SimpleNamespace())
    assert unwrapped["use_rslora_applied"] is None
    assert "unresolved" in unwrapped["scaling"]


def test_band_trajectory_json_carries_gauge(tmp_path: Path) -> None:
    """The in-loop band trajectory writer persists the gauge label (round-6
    major: i601_analyze.py Phase 4 consumes this file alongside staged-gauge
    dense reads — the artifact itself must say which gauge it is)."""
    import torch

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback, _resolve_live_gauge

    out = tmp_path / "inloop_band_trajectory.json"
    cb = MarkerBandStopCallback(
        marker_token_ids=[83399],
        probe_input_ids=torch.tensor([[1, 2, 3]]),
        probe_marker_positions=torch.tensor([2]),
        probe_attention_mask=torch.ones(1, 3, dtype=torch.long),
        eos_token_id=151645,
        trajectory_out_path=str(out),
    )
    cb._gauge = _resolve_live_gauge(
        SimpleNamespace(
            peft_config={"default": SimpleNamespace(use_rslora=True, r=32, lora_alpha=64)}
        )
    )
    cb._trajectory_records.append(
        {
            "step": 10,
            "logp_trained": -5.0,
            "logp_base": -19.0,
            "delta_nats": 14.0,
            "z_marker_trained": 1.0,
            "z_marker_base": -10.0,
            "z_eos_trained": 8.0,
            "z_eos_base": 8.5,
            "logZ_trained": 9.0,
            "logZ_base": 9.1,
        }
    )
    cb._write_trajectory()
    payload = json.loads(out.read_text())
    assert payload["gauge"]["use_rslora_applied"] is True
    assert payload["gauge"]["scaling"] == "alpha/sqrt(r)"
    assert payload["gauge"]["note"] == "live-training-model"


# ── 4. NaN→None in the phase0 tables (round-5 review minor) ──────────────────


def _traj_no_emission(delta_g: float) -> dict:
    return {
        "checkpoints": [
            {
                "frac": 1.0,
                "source_self": {
                    "delta_g_mean": delta_g,
                    "delta_z_marker_mean": delta_g,
                    "z_marker_g_mean": 1.0,
                    "z_eos_g_mean": 2.0,
                    "z_marker_b_mean": -1.0,
                    "z_eos_b_mean": 3.0,
                    # NO emission_p key.
                },
            }
        ]
    }


def test_terminal_source_stats_missing_emission_p_is_none() -> None:
    from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
        terminal_source_stats,
    )

    stats = terminal_source_stats(_traj_no_emission(8.0))
    assert stats["emission_p"] is None
    dumped = json.dumps(stats)
    assert "NaN" not in dumped
    json.loads(dumped)  # strict round-trip


def test_crosscheck_table_is_strict_json_when_emission_p_missing() -> None:
    from explore_persona_space.experiments.neg_setpoint_601.phase0_lib import (
        onpolicy_crosscheck,
        terminal_source_stats,
    )

    reread = {
        "cell_a_seed42": terminal_source_stats(_traj_no_emission(8.0)),
        "cell_b_seed42": terminal_source_stats(_traj_no_emission(13.5)),
    }
    committed = {"cell_a_seed42": 8.3, "cell_b_seed42": 13.1}
    result = onpolicy_crosscheck(reread, committed)
    per = result["per_adapter"]
    assert per["cell_a_seed42"]["reread_emission_p"] is None
    dumped = json.dumps(result)
    assert "NaN" not in dumped
    json.loads(dumped)  # strict round-trip (what phase0_gate.json requires)
