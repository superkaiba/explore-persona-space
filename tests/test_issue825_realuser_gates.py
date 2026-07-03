"""Tests for the issue-825 real-user-turn-null binding gate helper.

BINDING standing rec from the round-1 statistics review (plan v11 hard-req 7):
feed the gate helper synthetic anchor-miss / wiring-fail / coverage-missing
artifacts and assert the FAILURE sentinel statuses (``anchor_gate_miss``,
``wiring_check_fail``, ``coverage_miss``, plus ``fit_deferred_failure`` /
``ingest_shortfall``) fire BEFORE any SUCCESS path. The production smoke
bypasses numeric gates under EPS_SMOKE=1, so these committed tests are the
gates' only executable coverage.

Network-free and model-free: all artifacts are synthetic JSON in tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_realuser_gates as gates  # noqa: E402

PARENT_ANCHOR_L19 = 0.0757  # the committed parent value the gate reads at run time


def _cell_payload(l19: float = -1.5, n: int = 2000, with_mlp: bool = True) -> dict:
    payload = {
        "metadata": {"n": n},
        "selection_symmetric": {
            "frozen_layer_table": {
                "19": {"r2_obs": l19, "null_mean": -0.1, "null_p975": 0.05},
                "26": {"r2_obs": l19 + 0.3, "null_mean": -0.1, "null_p975": 0.05},
            },
            "obs_layer_max_r2": 0.1,
            "obs_argmax_layer": 5,
            "null_layer_max_r2_per_draw": [0.0, 0.01],
            "null_layer_max_p975": 0.01,
        },
        "r2_bootstrap_ci_frozen_layers": {
            "19": {"r2": l19, "ci_lo": l19 - 0.1, "ci_hi": l19 + 0.1, "n": n}
        },
    }
    if with_mlp:
        payload["mlp"] = {
            "19": {
                "r2_obs": 0.1,
                "r2_null": [0.0],
                "r2_obs_folds": [0.1, 0.2],
                "budget_hit_folds": [],
            }
        }
    return payload


def _wiring_payload(own: float = 2.0, shuf: float = 3.0) -> dict:
    return {
        "followup_label": "real-user-turn-null",
        "per_format": {
            fmt: {
                "cell_id": f"M_x_user_{fmt}",
                "n": 200,
                "own_mean_nll": own,
                "shuffled_mean_nll": shuf,
                "own_minus_shuffled": own - shuf,
            }
            for fmt in ("chat", "naturalistic")
        },
    }


@pytest.fixture()
def scaffold(tmp_path):
    """A fully-PASSING synthetic artifact tree; tests then break one piece."""
    out_dir = tmp_path / "out"
    anchor_dir = out_dir / "anchor_parent"
    realuser_dir = tmp_path / "realuser"
    wiring_dir = tmp_path / "wiring"
    parent_dir = tmp_path / "parent"
    for d in (out_dir, anchor_dir, realuser_dir, wiring_dir, parent_dir):
        d.mkdir(parents=True)
    for cid in gates.CELLS8:
        (out_dir / f"cells_{cid}.json").write_text(json.dumps(_cell_payload()))
    (anchor_dir / f"cells_{gates.ANCHOR_CELL}.json").write_text(
        json.dumps(_cell_payload(l19=PARENT_ANCHOR_L19 + 0.01))
    )
    (parent_dir / f"cells_{gates.ANCHOR_CELL}.json").write_text(
        json.dumps(_cell_payload(l19=PARENT_ANCHOR_L19))
    )
    (out_dir / "headline_metrics.json").write_text(json.dumps({"followup_label": "x"}))
    (realuser_dir / "conversations_real2turn_meta.json").write_text(
        json.dumps({"n_kept": 2000, "n_streamed": 21000})
    )
    for model in gates.WIRING_MODELS:
        (wiring_dir / f"wiring_check_{model}.json").write_text(json.dumps(_wiring_payload()))
    return {
        "out_dir": out_dir,
        "anchor_dir": anchor_dir,
        "realuser_dir": realuser_dir,
        "wiring_dir": wiring_dir,
        "parent_cells_dir": parent_dir,
        "sentinel": tmp_path / "logs" / "sentinel.json",
        "n_target": 2000,
        "smoke": False,
    }


def _run(scaffold):
    return gates.run_gates(**scaffold)


def _sentinel(scaffold) -> dict:
    return json.loads(scaffold["sentinel"].read_text())


def _assert_failure(scaffold, expected_status: str):
    with pytest.raises(SystemExit):
        _run(scaffold)
    sent = _sentinel(scaffold)
    # poll_pipeline._SENTINEL_REQUIRED_KEYS contract
    assert sent["sentinel_schema_version"] == 1
    assert sent["kind"] == "epm:results"
    assert sent["version"] == 1
    assert sent["status"] == expected_status
    outcomes = json.loads((scaffold["out_dir"] / "gate_outcomes.json").read_text())
    assert outcomes["all_pass"] is False
    assert outcomes["failure"]["status"] == expected_status
    return sent


# ---------------------------------------------------------------------------
# PASS path + success-sentinel ordering
# ---------------------------------------------------------------------------


def test_all_pass_writes_outcomes_and_no_failure_sentinel(scaffold, monkeypatch):
    outcomes = _run(scaffold)
    assert outcomes["all_pass"] is True
    assert not scaffold["sentinel"].exists()  # sentinel only on failure / success step
    for name in (
        "deferred_fit_failures",
        "ingest_floor",
        "anchor_ridge_tolerance",
        "wiring_check",
        "coverage",
    ):
        assert name in outcomes["gates"], name
    monkeypatch.setenv("EPS_T0", "0")
    gates.success_sentinel(scaffold["out_dir"], scaffold["sentinel"])
    sent = _sentinel(scaffold)
    assert sent["status"] == "success"
    assert sent["sentinel_schema_version"] == 1
    assert sent["note"]["followup_label"] == "real-user-turn-null"
    assert sent["note"]["gpu_hours_budgeted"] == 3.0


def test_success_sentinel_refuses_without_all_pass(scaffold):
    # No gate_outcomes.json yet -> refuse.
    with pytest.raises(AssertionError):
        gates.success_sentinel(scaffold["out_dir"], scaffold["sentinel"])
    # A FAILED gate_outcomes.json -> refuse (gates fire before any SUCCESS path).
    (scaffold["out_dir"] / "gate_outcomes.json").write_text(
        json.dumps({"all_pass": False, "failure": {"status": "anchor_gate_miss"}})
    )
    with pytest.raises(AssertionError):
        gates.success_sentinel(scaffold["out_dir"], scaffold["sentinel"])
    assert not scaffold["sentinel"].exists()


# ---------------------------------------------------------------------------
# FAILURE statuses (each fires + writes the schema-enveloped sentinel)
# ---------------------------------------------------------------------------


def test_anchor_gate_miss(scaffold):
    (scaffold["anchor_dir"] / f"cells_{gates.ANCHOR_CELL}.json").write_text(
        json.dumps(_cell_payload(l19=PARENT_ANCHOR_L19 + 0.2))  # |delta| 0.2 > 0.05
    )
    sent = _assert_failure(scaffold, "anchor_gate_miss")
    assert "rig drift" in sent["note"]["failure"]


def test_anchor_missing_l19_row_is_gate_miss(scaffold):
    payload = _cell_payload()
    payload["selection_symmetric"]["frozen_layer_table"].pop("19")
    (scaffold["anchor_dir"] / f"cells_{gates.ANCHOR_CELL}.json").write_text(json.dumps(payload))
    _assert_failure(scaffold, "anchor_gate_miss")


def test_wiring_check_fail_own_ge_shuffled(scaffold):
    (scaffold["wiring_dir"] / "wiring_check_pretrained.json").write_text(
        json.dumps(_wiring_payload(own=3.0, shuf=2.0))
    )
    _assert_failure(scaffold, "wiring_check_fail")


def test_wiring_check_fail_missing_file(scaffold):
    (scaffold["wiring_dir"] / "wiring_check_instruct.json").unlink()
    _assert_failure(scaffold, "wiring_check_fail")


def test_coverage_miss_missing_cell(scaffold):
    (scaffold["out_dir"] / "cells_M_pretrained_user_naturalistic.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_miss_missing_mlp_block(scaffold):
    (scaffold["out_dir"] / "cells_M_instruct_user_chat.json").write_text(
        json.dumps(_cell_payload(with_mlp=False))
    )
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_miss_missing_anchor_file_bypasses_numeric_but_fires(scaffold):
    # With smoke=True the numeric anchor gate is bypassed, but the anchor file
    # PRESENCE is structural — still a failure.
    scaffold["smoke"] = True
    (scaffold["anchor_dir"] / f"cells_{gates.ANCHOR_CELL}.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_miss_row_parity(scaffold):
    (scaffold["out_dir"] / "cells_M_pretrained_assistant_chat.json").write_text(
        json.dumps(_cell_payload(n=1999))  # != ingested 2000
    )
    _assert_failure(scaffold, "coverage_miss")


def test_fit_deferred_failure_nested_sweep(scaffold):
    # Under the ANCHOR subdir: the rglob sweep must catch EVERY fit out-dir.
    (scaffold["anchor_dir"] / "fit_failures.json").write_text(
        json.dumps([{"cell_id": "M_instruct_assistant_chat", "error_type": "ValueError"}])
    )
    _assert_failure(scaffold, "fit_deferred_failure")


def test_ingest_shortfall(scaffold):
    (scaffold["realuser_dir"] / "conversations_real2turn_meta.json").write_text(
        json.dumps({"n_kept": 1500})
    )
    _assert_failure(scaffold, "ingest_shortfall")


def test_gate_order_anchor_before_wiring(scaffold):
    """Plan §7 order: with BOTH an anchor miss and a wiring fail staged, the
    anchor fires first (deferred -> ingest -> anchor -> wiring -> coverage)."""
    (scaffold["anchor_dir"] / f"cells_{gates.ANCHOR_CELL}.json").write_text(
        json.dumps(_cell_payload(l19=PARENT_ANCHOR_L19 + 0.2))
    )
    (scaffold["wiring_dir"] / "wiring_check_instruct.json").write_text(
        json.dumps(_wiring_payload(own=9.0, shuf=1.0))
    )
    _assert_failure(scaffold, "anchor_gate_miss")


# ---------------------------------------------------------------------------
# Smoke bypass semantics (numeric bypassed, structural binding)
# ---------------------------------------------------------------------------


def test_smoke_bypasses_numeric_gates(scaffold):
    scaffold["smoke"] = True
    # All numeric misses at once: anchor off-tolerance, wiring own>shuffled,
    # ingest n_kept tiny, per-cell n mismatched — smoke passes structurally.
    (scaffold["anchor_dir"] / f"cells_{gates.ANCHOR_CELL}.json").write_text(
        json.dumps(_cell_payload(l19=5.0, n=8))
    )
    (scaffold["wiring_dir"] / "wiring_check_instruct.json").write_text(
        json.dumps(_wiring_payload(own=9.0, shuf=1.0))
    )
    (scaffold["realuser_dir"] / "conversations_real2turn_meta.json").write_text(
        json.dumps({"n_kept": 8})
    )
    outcomes = _run(scaffold)
    assert outcomes["all_pass"] is True
    assert outcomes["gates"]["ingest_floor"]["result"] == "BYPASSED_SMOKE_PRESENCE_ONLY"
    assert outcomes["gates"]["anchor_ridge_tolerance"]["result"] == "BYPASSED_SMOKE_PRESENCE_ONLY"
    assert outcomes["gates"]["wiring_check"]["result"] == "BYPASSED_SMOKE_PRESENCE_ONLY"


def test_smoke_still_binds_structural_coverage(scaffold):
    scaffold["smoke"] = True
    (scaffold["out_dir"] / "headline_metrics.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


# ---------------------------------------------------------------------------
# fail-from-ingest: route on the ARTIFACT, exit-code fall-through
# ---------------------------------------------------------------------------


def test_fail_from_ingest_routes_on_artifact(scaffold):
    (scaffold["realuser_dir"] / "ingest_failure.json").write_text(
        json.dumps({"status": "ingest_shortfall", "n_kept": 1906, "n_target": 2000})
    )
    gates.fail_from_ingest(scaffold["realuser_dir"], scaffold["sentinel"])
    sent = _sentinel(scaffold)
    assert sent["status"] == "ingest_shortfall"
    assert sent["note"]["ingest_failure"]["n_kept"] == 1906


def test_fail_from_ingest_fallthrough_without_artifact(scaffold):
    gates.fail_from_ingest(scaffold["realuser_dir"], scaffold["sentinel"])
    sent = _sentinel(scaffold)
    assert sent["status"] == "ingest_error"
