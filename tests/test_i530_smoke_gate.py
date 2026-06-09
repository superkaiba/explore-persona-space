# em-dash + Qwen marker " ※" intentional
"""Regression test for `scripts/i530_smoke.py::_evaluate_smoke_gates`.

Round 1 (commit f8ef182d0) read `terminal["source"][<persona>][<q>]["g_logp"]`,
but the canonical trajectory.json schema produced by
`src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py`
(lines 28-38 + 427-446) emits a FLAT scalar at `terminal["source_self"]["delta_g_mean"]`.
The wrong read silently produced `NaN`, and `NaN < 5.0` / `NaN > 12.0` both
return False in Python, so BOTH source-side FAIL conditions
(`lr_too_cold_no_implant` and `source_dg_above_band_at_terminal`) were bypassed
— the gate returned PASS on a fully-busted lr. The fix reads the flat scalar
AND adds an explicit `math.isnan` guard BEFORE the band comparisons.

These tests pin both the canonical-schema read AND the NaN-guard so the
smoke-gate's primary purpose (catching a busted lr before the 23 GPU-h Phase 2
sweep) can't silently regress again.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "i530_smoke.py"


@pytest.fixture(scope="module")
def smoke_mod():
    """Import `scripts/i530_smoke.py` as a module (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location("i530_smoke_under_test", SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_trajectory(
    *,
    source_delta_g_mean: float | None,
    bystander_argmax_rate: float,
    bystander_g_logp: float,
    n_bystander_personas: int = 4,
    n_q: int = 5,
) -> dict:
    """Construct a trajectory.json payload with the REAL schema (one checkpoint).

    `source_delta_g_mean=None` simulates a corrupted / missing scalar (the
    `math.isnan` guard's job). Bystander leaves carry a uniform `argmax_marker`
    bool gated by `bystander_argmax_rate` and a constant `g_logp`.
    """
    n_total = n_bystander_personas * n_q
    n_argmax_hits = round(bystander_argmax_rate * n_total)
    held_out: dict[str, dict[str, dict[str, float | bool]]] = {}
    hits_left = n_argmax_hits
    for p_idx in range(n_bystander_personas):
        persona = f"bystander_{p_idx}"
        held_out[persona] = {}
        for q_idx in range(n_q):
            this_hit = hits_left > 0
            if this_hit:
                hits_left -= 1
            held_out[persona][f"q{q_idx}"] = {
                "g_logp": bystander_g_logp,
                "b_logp": -22.0,
                "delta_g": bystander_g_logp - (-22.0),
                "argmax_marker": this_hit,
                "n_marker_in_R": 0,
                "r_collapsed": False,
                "kl": None,
            }
    source_self: dict = {
        "g_logp_mean": -10.0,
        "b_logp_mean": -20.0,
        "delta_g_mean": source_delta_g_mean,
        "emission_p": 0.5,
        "r_collapsed": False,
    }
    return {
        "cell": "c504v3_near",
        "seed": 42,
        "source": "software_engineer",
        "matched_slice_target_nats": 8.0,
        "checkpoints": [
            {
                "frac": 1.0,
                "step": 600,
                "adapter_path": "/tmp/fake-adapter",
                "source_self": source_self,
                "held_out_collapse_share": 0.0,
                "n_held_out_collapsed": 0,
                "held_out": held_out,
            }
        ],
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "trajectory.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def test_round1_bug_regression_lr_too_cold_below_band(smoke_mod, tmp_path):
    """delta_g_mean=1.0 (< 5 nat) must FAIL with lr_too_cold_no_implant.

    This is the EXACT case round 1's broken `terminal.get("source", {})` read
    silently passed: the wrong key produced NaN, NaN<5 was False, NaN>12 was
    False, and the gate fell through to PASS. With the fix, the canonical
    `source_self.delta_g_mean` scalar is read and the gate correctly FAILs.
    """
    traj = _build_trajectory(
        source_delta_g_mean=1.0,  # well below the 5-nat floor
        bystander_argmax_rate=0.05,  # well below the 0.60 saturation threshold
        bystander_g_logp=-5.0,  # 5 nat headroom (> 2-nat requirement)
    )
    path = _write(tmp_path, traj)
    diag = smoke_mod._evaluate_smoke_gates(path)
    assert diag["verdict"] == "FAIL", diag
    assert diag["failure_reason"] == "lr_too_cold_no_implant", diag
    assert diag["source_dg_at_terminal"] == pytest.approx(1.0)


def test_above_band_fails(smoke_mod, tmp_path):
    """delta_g_mean=15.0 (> 12 nat) must FAIL with source_dg_above_band_at_terminal."""
    traj = _build_trajectory(
        source_delta_g_mean=15.0,
        bystander_argmax_rate=0.05,
        bystander_g_logp=-5.0,
    )
    path = _write(tmp_path, traj)
    diag = smoke_mod._evaluate_smoke_gates(path)
    assert diag["verdict"] == "FAIL", diag
    assert diag["failure_reason"] == "source_dg_above_band_at_terminal", diag


def test_missing_source_dg_returns_explicit_failure(smoke_mod, tmp_path):
    """Corrupted payload (source_self.delta_g_mean missing) must surface a named FAIL.

    Without the explicit `math.isnan` guard, `NaN < band_low_nats` returns
    False and the gate would silently fall through to either the bystander
    branch or PASS. The fix makes "source ΔG unreadable" a first-class FAIL.
    """
    traj = _build_trajectory(
        source_delta_g_mean=None,  # delta_g_mean key present but null
        bystander_argmax_rate=0.05,
        bystander_g_logp=-5.0,
    )
    path = _write(tmp_path, traj)
    diag = smoke_mod._evaluate_smoke_gates(path)
    assert diag["verdict"] == "FAIL", diag
    assert diag["failure_reason"] == "source_dg_missing", diag


def test_happy_path_passes(smoke_mod, tmp_path):
    """delta_g_mean=8.0 (in [5, 12]) + bystander de-saturated + ≥2 nat headroom → PASS."""
    traj = _build_trajectory(
        source_delta_g_mean=8.0,  # in band [5, 12]
        bystander_argmax_rate=0.20,  # < 0.60 saturation threshold
        bystander_g_logp=-4.0,  # headroom = -(-4.0) = 4.0 nat ≥ 2.0 requirement
    )
    path = _write(tmp_path, traj)
    diag = smoke_mod._evaluate_smoke_gates(path)
    assert diag["verdict"] == "PASS", diag
    assert diag["failure_reason"] == "", diag
    assert diag["source_dg_at_terminal"] == pytest.approx(8.0)
    assert diag["bystander_argmax_rate"] == pytest.approx(0.20)


def test_bystander_saturated_fails(smoke_mod, tmp_path):
    """In-band source + saturated bystanders (≥0.60 argmax) → FAIL on saturation gate."""
    traj = _build_trajectory(
        source_delta_g_mean=8.0,
        bystander_argmax_rate=0.85,  # ≥ 0.60 saturation threshold
        bystander_g_logp=-0.5,
    )
    path = _write(tmp_path, traj)
    diag = smoke_mod._evaluate_smoke_gates(path)
    assert diag["verdict"] == "FAIL", diag
    assert diag["failure_reason"] == "lr_5e6_still_saturates_at_band_stop", diag


def test_trajectory_missing(smoke_mod, tmp_path):
    diag = smoke_mod._evaluate_smoke_gates(tmp_path / "does_not_exist.json")
    assert diag["verdict"] == "FAIL"
    assert diag["failure_reason"] == "trajectory_missing"


def test_empty_checkpoints(smoke_mod, tmp_path):
    path = _write(tmp_path, {"cell": "c", "seed": 1, "checkpoints": []})
    diag = smoke_mod._evaluate_smoke_gates(path)
    assert diag["verdict"] == "FAIL"
    assert diag["failure_reason"] == "no_checkpoints_in_trajectory"
