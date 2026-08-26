"""Realized estimator-provenance pins for persisted P5 fit payloads (task #2546 v16).

Round-16 record-integrity fix: `_fit_params` used to stamp the P5 regime
(23-pt logspace(-3,8) grid, ``fc.N_INNER_LAMBDA_FOLDS = 2`` patch) onto EVERY
persisted unit payload, while three routes actually run a different regime:

- ``ma`` ladder tiers (``issue825_map_alignment``): 13-pt logspace(-2,4)
  legacy grid, 4 inner folds (``ma.N_INNER_LAMBDA_FOLDS`` snapshots fit825's
  default at IMPORT — the scoped run_units patch never reaches it);
- ``ood`` units: the same ma machinery (``issue2546_fit_cells.run_ood_unit``);
- ``operator`` units: ``issue825_crossmodel_map_transfer.fit_primal_beta``
  (13-pt grid; ``N_INNER_LAMBDA_FOLDS = 4`` as a module LITERAL).

These pins drive the REAL ``run_units`` path (the P5 inner-folds patch armed,
real ``ma``/``xm``/``fc`` fit bodies, real ``_atomic_json`` writes) on the
driver's own selftest fixture and assert each persisted payload records THAT
route's realized grid + fold count + selector — never the P5 defaults. They
FAIL against the pre-v16 module (no ``estimator_realized`` record; verified by
in-place ``git stash`` round-trip — a symlinked shadow tree false-PASSes
because ``issue825_fit_cells.py`` sys.path-inserts its RESOLVED parent).

Production-body coverage (code-style.md "one production-body test per
seam-stubbed function"): no function is stubbed anywhere in this file — the
fail-loud/tracking tests mutate module CONSTANTS (the system's input state),
and every fit body executes for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_fit_cells as fc  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue2546_fit_cells as F  # noqa: E402

LEGACY_GRID_ID = ["logspace", -2.0, 4.0, 13]
P5_GRID_ID = list(F.LAMBDAS_N1M_PARAMS)
SELECTOR_NAMES = {"inner-group-cv", "gcv-fallback", "gcv"}


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """Selftest fixture store + ONE real run_units pass over the 3 divergent routes."""
    root = tmp_path_factory.mktemp("i2546-prov")
    prof = F.LayerProfile(
        arm=1,
        n_layers=3,
        hidden=32,
        frozen=(0, 1, 2),
        headline=1,
        post_side="post",
        short_side="pre",
        has_pre_model=True,
    )
    F._selftest_fixture(root, prof)
    args = SimpleNamespace(
        out_root=str(root),
        smoke=True,
        null_draws=2,
        n_boot=20,
        prefill_fallback=False,
        decode_fallback=False,
    )
    units = F.build_registry(prof)
    picks = {
        "ladder": next(u for u in units if u.kind == "ladder" and "gsm8k_train" in u.unit_id),
        "ood": next(u for u in units if u.unit_id == "ood_does2doesnt__a1"),
        "operator": next(u for u in units if u.kind == "operator"),
    }
    rc = F.run_units(args, prof, list(picks.values()))
    assert rc == 0
    return SimpleNamespace(root=root, prof=prof, args=args, picks=picks)


def _payload(rig, key: str) -> dict:
    return json.loads(F.unit_out_path(rig.root, rig.picks[key]).read_text())


def _assert_legacy_record(fp: dict, want_regime: str) -> None:
    """The shared NOT-the-P5-defaults assertion block for the ma/xm routes."""
    assert fp["regime"] == want_regime, fp["regime"]
    assert fp["inner_lambda_folds"] == 4, fp["inner_lambda_folds"]  # NOT the P5 patch's 2
    assert fp["lambdas"] == LEGACY_GRID_ID, fp["lambdas"]  # NOT the 23-pt P5 identifier
    assert fp["lambdas"] != P5_GRID_ID
    assert fp["lambda_selector"] == "inner-group-cv", fp["lambda_selector"]
    er = fp["estimator_realized"]
    assert er["n_lambdas"] == 13, er["n_lambdas"]
    np.testing.assert_array_equal(np.asarray(er["lambda_grid_values"]), np.logspace(-2.0, 4.0, 13))
    counts = er["selector_realized_counts"]
    assert counts and set(counts) <= SELECTOR_NAMES, counts
    assert sum(counts.values()) >= 1, counts


def test_ladder_payload_records_realized_ma_regime(rig):
    """Route 1 (ma ladder tiers): payload records the 13-pt/4-fold ma regime."""
    j = _payload(rig, "ladder")
    assert j["status"] == "ok", j.get("status")
    _assert_legacy_record(j["fit_params"], "ma_battery_legacy")
    assert "map_alignment" in j["fit_params"]["estimator_realized"]["machinery"]


def test_ood_payload_records_realized_ma_regime(rig):
    """Route 2 (ood units): same ma machinery, same realized regime record."""
    j = _payload(rig, "ood")
    assert j["status"] == "ok", j.get("status")
    _assert_legacy_record(j["fit_params"], "ma_battery_legacy")


def test_operator_payload_records_realized_xm_regime(rig):
    """Route 3 (operator units): fit_primal_beta's 13-pt/4-fold literal regime,
    with the realized selector count covering ALL THREE fits (2 direct
    fit_primal_beta calls + 1 inside oc.alignment_capacity)."""
    j = _payload(rig, "operator")
    assert j["status"] == "ok", j.get("status")
    _assert_legacy_record(j["fit_params"], "xm_primal_legacy")
    er = j["fit_params"]["estimator_realized"]
    assert "crossmodel_map_transfer" in er["machinery"]
    assert sum(er["selector_realized_counts"].values()) == 3, er["selector_realized_counts"]


def test_run_level_params_survive_beside_realized_record(rig):
    """The run-level keys (fingerprint inputs) still ride every payload."""
    j = _payload(rig, "ladder")
    fp = j["fit_params"]
    assert fp["n_folds"] == F.N_FOLDS
    assert fp["seed"] == F.FIT_SEED
    assert fp["fit_params_schema"] == "v16-realized-estimator"
    assert isinstance(fp["store_key"], str) and fp["store_key"]


def test_recorded_folds_track_effective_change(rig, monkeypatch):
    """The record follows a CHANGED effective fold count (not a constant):
    flipping ma.N_INNER_LAMBDA_FOLDS to 3 makes the fit BUILD 3 inner folds
    and the persisted payload must say 3."""
    dest = F.unit_out_path(rig.root, rig.picks["ood"])
    dest.unlink()
    monkeypatch.setattr(ma, "N_INNER_LAMBDA_FOLDS", 3)
    rc = F.run_units(rig.args, rig.prof, [rig.picks["ood"]])
    assert rc == 0
    j = _payload(rig, "ood")
    assert j["fit_params"]["inner_lambda_folds"] == 3, j["fit_params"]
    assert j["fit_params"]["estimator_realized"]["inner_lambda_folds"] == 3
    # Restore an unmutated payload for any later reader (monkeypatch undoes
    # the module constant at teardown; the on-disk JSON must follow).
    monkeypatch.undo()
    dest.unlink()
    rc = F.run_units(rig.args, rig.prof, [rig.picks["ood"]])
    assert rc == 0
    assert _payload(rig, "ood")["fit_params"]["inner_lambda_folds"] == 4


def test_selector_telemetry_off_fails_loud(rig, monkeypatch):
    """SELECTOR_LOG disabled => the realized selector is unmeasurable => the
    unit REFUSES (RuntimeError) instead of stamping a plausible guess."""
    F.unit_out_path(rig.root, rig.picks["ood"]).unlink()
    entry_folds = fc.N_INNER_LAMBDA_FOLDS
    monkeypatch.setattr(ma, "SELECTOR_LOG", None)
    with pytest.raises(RuntimeError, match="SELECTOR_LOG"):
        F.run_units(rig.args, rig.prof, [rig.picks["ood"]])
    # The scoped P5 patch restored fc's global on the way out (v15 contract).
    assert entry_folds == fc.N_INNER_LAMBDA_FOLDS


def test_scan_index_grid_divergence_fails_loud(rig, monkeypatch):
    """ma.LAMBDAS diverging from fit825.LAMBDAS under inner-group-cv makes the
    realized grid ill-defined (curve scans one grid, selection indexes the
    other) => refuse, never record either grid."""
    dest = F.unit_out_path(rig.root, rig.picks["ood"])
    if dest.is_file():
        dest.unlink()
    monkeypatch.setattr(ma, "LAMBDAS", np.logspace(-2.0, 4.0, 12))
    with pytest.raises(RuntimeError, match="diverges"):
        F.run_units(rig.args, rig.prof, [rig.picks["ood"]])
    monkeypatch.undo()
    rc = F.run_units(rig.args, rig.prof, [rig.picks["ood"]])  # leave a clean payload
    assert rc == 0


def test_stale_pre_v16_payload_cannot_resume_skip(rig):
    """A payload written under the pre-v16 schema (false per-unit provenance)
    carries a fingerprint the v16 params can never reproduce
    (fit_params_schema is a fingerprint input), so it is RE-RUN and
    overwritten with true provenance — never resume-blessed."""
    dest = F.unit_out_path(rig.root, rig.picks["ood"])
    stale = {
        "status": "ok",
        "fingerprint": "stale-pre-v16-fp",
        "fit_params": {"lambdas": P5_GRID_ID, "inner_lambda_folds": 2},
    }
    dest.write_text(json.dumps(stale))
    rc = F.run_units(rig.args, rig.prof, [rig.picks["ood"]])
    assert rc == 0
    j = _payload(rig, "ood")
    assert j["fingerprint"] != "stale-pre-v16-fp"
    _assert_legacy_record(j["fit_params"], "ma_battery_legacy")


def test_attach_fit_params_refuses_ok_payload_without_record():
    """An ok payload with NO realized record must fail loud, never inherit
    run-level defaults as fit provenance."""
    with pytest.raises(RuntimeError, match="realized-estimator"):
        F._attach_fit_params({"status": "ok"}, "unit-x", {"seed": 0})


def test_attach_fit_params_no_fit_for_non_ok_payload():
    """A dropped/degenerate payload gets a truthful no-fit record: estimator
    keys None, regime no_fit — not the run-level defaults."""
    payload = {"status": "dropped_below_floor", "n_rows": 3}
    F._attach_fit_params(payload, "unit-y", {"seed": 0, "lambdas": P5_GRID_ID})
    fp = payload["fit_params"]
    assert fp["regime"] == "no_fit"
    assert fp["lambdas"] is None and fp["inner_lambda_folds"] is None
    assert "dropped_below_floor" in fp["estimator_realized"]["note"]


def test_realized_sweep_estimator_counts_and_fail_loud():
    """Sweep-record helper: counts summarize the realized per-(layer,fold)
    selector table; a missing or all-None table refuses."""
    rec = F._realized_sweep_estimator(
        {"lambda_selector": [["inner-group-cv", None], ["gcv-fallback", "inner-group-cv"]]},
        inner_folds_at_call=2,
    )
    assert rec["regime"] == "p5_sweep_v2"
    assert rec["lambda_grid_params"] == P5_GRID_ID
    assert rec["selector_realized_counts"] == {"inner-group-cv": 2, "gcv-fallback": 1}
    assert rec["inner_lambda_folds"] == 2
    with pytest.raises(RuntimeError, match="no lambda_selector"):
        F._realized_sweep_estimator({}, inner_folds_at_call=2)
    with pytest.raises(RuntimeError, match="all-None"):
        F._realized_sweep_estimator({"lambda_selector": [[None]]}, inner_folds_at_call=2)


def test_merge_sweep_estimators_no_fit_and_disagreement():
    """Traj merge: zero fitted strata => truthful no_fit; strata disagreeing
    on a realized key => refuse (anomaly, never pick one)."""
    assert F._merge_sweep_estimators([])["regime"] == "no_fit"
    a = F._realized_sweep_estimator(
        {"lambda_selector": [["inner-group-cv"]]}, inner_folds_at_call=2
    )
    b = dict(a)
    b["inner_lambda_folds"] = 4
    with pytest.raises(RuntimeError, match="disagree"):
        F._merge_sweep_estimators([a, b])
    merged = F._merge_sweep_estimators([a, dict(a)])
    assert merged["selector_realized_counts"] == {"inner-group-cv": 2}
