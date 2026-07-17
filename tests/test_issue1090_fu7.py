"""#1090 fu7 (`sycophancy-lr-install-and-remeasure`) invariants.

Pins the round's permanent gates + registration (plan v13):
- ROUNDS["fu7"] registration + BOTH-arm-class smoke default (C3 + C5);
- fu4/fu5 RoundSpec defaults byte-unchanged by the fu7 seam fields;
- K3 reference-delta parity statuses (ok / parity-degraded / parity-failed /
  missing) at the registered ±0.15/±0.25 tolerances vs fu2's 0.58;
- K5 r_B identity asserts (realized keys + (28, 3584) shape) refuse drift;
- the fu7 Tier-2-anchored C/M/U/V lattice (M excludes the control arm);
- panel judge item ids stay under the Batch custom_id budget (#1415: <=53);
- the fu6 `local_adapter_dir` capture seam fails loud on a dir without an
  adapter_config.json BEFORE any Hub staging / merge.
"""

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_fu4 as fu4  # noqa: E402
import issue1090_fu6 as fu6  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_round():
    yield
    fu4.set_round("fu4")


def test_fu7_round_registered_and_smoke_covers_both_arm_classes():
    spec = fu4.set_round("fu7")
    assert spec.label == "sycophancy-lr-install-and-remeasure"
    assert [r.run_id for r in fu4.FU7_RUNS] == [
        "syc-c3-lr1e5",
        "syc-c3-lr3e5",
        "syc-c3-lr1e4",
        "syc-c5-lr1e5",
        "syc-c5-lr3e5",
        "syc-c5-lr1e4",
    ]
    smoke_runs = fu4.resolve_fu4_runs(None, smoke=True)
    assert {r.cell_key for r in smoke_runs} == {"syc-c3", "syc-c5"}, (
        "the smoke default must cover BOTH arm classes (per-arm-class smoke rule)"
    )
    # Every run trains at the persona context on a parent-mix-subdir mix.
    for r in fu4.FU7_RUNS:
        assert r.context_id == "persona_software_engineer"
        assert r.mix_layout == "parent-mix-subdir"
        assert r.round_name == "fu7"


def test_fu4_fu5_roundspec_defaults_unchanged_by_fu7_seams():
    for name in ("fu4", "fu5"):
        spec = fu4.ROUNDS[name]
        assert spec.k3_parity_step == fu4.K3_PARITY_STEP
        assert spec.k3_parity_reference is None  # legacy MAX_RATE cap form
        assert spec.dual_rubric_tier2 is False
        assert spec.panel_remeasure is False


def _out_with_parity_rate(rate):
    rates = {} if rate is None else {"30": rate}
    return {
        "runs": {
            "syc-c3-lr1e5": {
                "run_id": "syc-c3-lr1e5",
                "cell_key": "syc-c3",
                "rates_by_step": rates,
            }
        },
        "cells": {},
    }


@pytest.mark.parametrize(
    ("rate", "status"),
    [
        (0.58, "ok"),
        (0.70, "ok"),  # |Δ| = 0.12 < 0.15 flag delta
        (0.75, "parity-degraded"),  # |Δ| = 0.17 in (0.15, 0.25]
        (0.85, "parity-failed"),  # |Δ| = 0.27 > 0.25 abort delta
        (0.30, "parity-failed"),  # symmetric: |Δ| = 0.28
        (None, "missing"),
    ],
)
def test_fu7_k3_reference_delta_statuses(rate, status):
    fu4.set_round("fu7")
    rec = fu4._retrain_parity_record(_out_with_parity_rate(rate))
    assert rec is not None
    assert rec["status"] == status
    assert rec["reference"] == 0.58
    assert rec["step"] == 30
    assert rec["diff_se_label"] == 0.07
    assert rec["schedule_parity"]["fu2_total_steps"] == 30


def test_fu7_k5_rb_asserts_refuse_drift(tmp_path):
    fu4.set_round("fu7")
    # Missing realized keys -> refuse (artifact-reuse check (c)).
    bad = tmp_path / "missing_keys"
    (bad / "rb").mkdir(parents=True)
    torch.save({"layers": list(range(28))}, bad / "rb" / "sycophancy_fu6.pt")
    with pytest.raises(RuntimeError, match="realized keys"):
        fu4._fu7_stage_rb(bad)
    # Wrong shape -> refuse (never project onto an unverified direction).
    wrong = tmp_path / "wrong_shape"
    (wrong / "rb").mkdir(parents=True)
    torch.save(
        {"r_b": torch.randn(4, 8), "layers": list(range(4))}, wrong / "rb" / "sycophancy_fu6.pt"
    )
    with pytest.raises(RuntimeError, match="shape"):
        fu4._fu7_stage_rb(wrong)
    # Conforming bundle -> unit-normalized directions returned.
    good = tmp_path / "good"
    (good / "rb").mkdir(parents=True)
    torch.save(
        {"r_b": torch.randn(28, 3584, dtype=torch.float32), "layers": list(range(28))},
        good / "rb" / "sycophancy_fu6.pt",
    )
    rb_unit, rb_path = fu4._fu7_stage_rb(good)
    assert tuple(rb_unit.shape) == (28, 3584)
    assert torch.allclose(rb_unit.norm(dim=1), torch.ones(28), atol=1e-5)
    assert rb_path == good / "rb" / "sycophancy_fu6.pt"


def test_fu7_lattice_m_excludes_control_and_uv_arithmetic():
    fu4.set_round("fu7")
    out = {
        "runs": {
            "syc-c3-lr1e5": {
                "run_id": "syc-c3-lr1e5",
                "cell_key": "syc-c3",
                "status": "trained",
                "tier2_trained": {"rate": 0.90},  # control HIGH: must not enter M
                "tier2_trained_pv": {"rate": 0.40},
            },
            "syc-c3-lr3e5": {
                "run_id": "syc-c3-lr3e5",
                "cell_key": "syc-c3",
                "status": "trained",
                "tier2_trained": {"rate": 0.62},
                "tier2_trained_pv": {"rate": 0.50},
            },
            "syc-c3-lr1e4": {
                "run_id": "syc-c3-lr1e4",
                "cell_key": "syc-c3",
                "status": "diverged",
            },
        },
        "cells": {},
    }
    fu4._fu7_lattice_inputs(out)
    cell = out["cells"]["syc-c3"]
    assert cell["control_run"] == "syc-c3-lr1e5"
    assert cell["C_control_tier2"] == 0.90
    assert cell["M_run"] == "syc-c3-lr3e5"
    assert cell["M_swept_max_tier2"] == 0.62  # control's 0.90 excluded from M
    assert cell["U_band_floor_margin"] == pytest.approx(0.62 - 0.60)
    assert cell["V_control_plateau_margin"] == pytest.approx(0.62 - (0.90 + 0.07))
    assert cell["arm_statuses"]["syc-c3-lr1e4"] == "diverged"


def test_fu7_panel_judge_item_ids_fit_batch_custom_id_budget():
    """#1415: encoder appends 11 chars to a 64-char custom_id cap -> item ids
    must stay <=53. Worst case: longest run_id x longest short-ctx x q019-c9
    under both rubric suffixes."""
    fu4.set_round("fu7")
    for run in fu4.FU7_RUNS:
        for ctx_id in fu6.CAPTURE_PANEL_IDS:
            tag = f"{run.run_id}-pn-{fu6._CTX_SHORT.get(ctx_id, ctx_id[:6])}"
            for suffix in ("legacy", "pv", "legacy-rule23", "pv-rule23"):
                item_id = f"{tag}-{suffix}-q019-c9"
                assert len(item_id) <= 53, (item_id, len(item_id))
    # Tier-2 dual-rubric tags too.
    for run in fu4.FU7_RUNS:
        for tag in (f"{run.run_id}-t2-trained-pv", f"{run.run_id}-t2-trained-rule23"):
            assert len(f"{tag}-q019-c9") <= 53


def test_fu6_local_adapter_dir_seam_fails_loud_before_staging(tmp_path, monkeypatch):
    """The `local_adapter_dir` capture seam asserts adapter_config.json exists
    BEFORE any Hub staging or merge — executes the real run_organism_capture
    body up to the seam (CPU; no network: a Hub call would be a different
    failure than the seam's own AssertionError)."""
    cfg = fu6.Cfg(
        smoke=True,
        manifest_path=None,
        manifest_out=None,
        out_root=tmp_path / "cap",
        sentinel_dir=tmp_path / "logs",
        upload=False,
    )
    empty = tmp_path / "empty_adapter"
    empty.mkdir()
    spec = {
        "organism_id": "fu7-test",
        "source_context": "persona_software_engineer",
        "local_adapter_dir": str(empty),
        "adapter_repo": "unused/unused",
        "adapter_subfolder": "unused",
        "adapter_rev": "main",
    }
    with pytest.raises(AssertionError):
        fu6.run_organism_capture(cfg, spec)
