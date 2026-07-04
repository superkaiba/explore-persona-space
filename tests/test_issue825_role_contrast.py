"""Tests for the issue-825 role-map-comparison runner + binding gates.

Plan v14 §4.3 hard-req 4: shared-fold identity across roles, joint-keep
alignment, allowlist intersection BEFORE fold assignment, conv==row uniqueness,
batched-vs-serial paired-bootstrap equivalence on a tiny NONZERO-MEAN synthetic
with duplicate rows (pins the own-mean centering convention — BINDING), and the
gate-path FAILURE statuses (``reproduction_gate_miss``,
``bundle_schema_mismatch``, ``row_alignment_shortfall``, ``coverage_miss``)
mirroring tests/test_issue825_realuser_gates.py. The production smoke bypasses
numeric gates under EPS_SMOKE=1, so these tests are the numeric gates' only
executable coverage.

Round-2 regression coverage (reconciler v6): the deferred-failure PRODUCTION
SEQUENCE (fit-deferred -> summarize-tolerate -> gates) ends in the REGISTERED
sentinel status (``bundle_schema_mismatch`` | ``fit_deferred_failure``), never
``summarize_error``; fail-sentinel mirrors UPLOAD-b (preds npz ->
analysis_tensors) before the sentinel write; a successful ``--resume`` re-run
clears its stale fit_failures.json record.

Network-free and model-free: synthetic arrays + JSON under tmp_path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_fit_cells as fit_cells  # noqa: E402
import issue825_role_contrast as rc  # noqa: E402

RNG = np.random.default_rng(0)
L = fit_cells.EXPECTED_LAYERS


def _synth_bundle(n: int = 9, n_turns: int = 3, dim: int = 6):
    """(slots, profiles, nll, conv_ids) satisfying the assembly contract."""
    slots = RNG.normal(size=(n, 2, L, dim)).astype(np.float32)
    profiles = RNG.normal(size=(n, n_turns, L, dim)).astype(np.float32)
    nll = RNG.uniform(0.5, 3.0, size=(n, n_turns)).astype(np.float32)
    conv_ids = np.asarray([f"c{i}" for i in range(n)])
    return slots, profiles, nll, conv_ids


# ---------------------------------------------------------------------------
# Assembly: joint keep, allowlist-before-folds, conv==row uniqueness
# ---------------------------------------------------------------------------


def test_joint_keep_drops_row_from_both_arms():
    slots, profiles, nll, conv_ids = _synth_bundle()
    slots[3, 1, 0, 0] = np.nan  # NaN in the USER slot only
    rows = rc.assemble_pair_rows(slots, profiles, nll, conv_ids)
    assert rows["n_joint"] == 8
    assert "c3" not in set(rows["conv_ids"])
    # BOTH arms lose the row (one joint mask — never independent masks).
    for key in ("X_a", "Y_a", "X_u", "Y_u", "nll_a", "nll_u"):
        assert len(rows[key]) == 8, key


def test_shared_folds_across_roles():
    slots, profiles, nll, conv_ids = _synth_bundle()
    rows = rc.assemble_pair_rows(slots, profiles, nll, conv_ids)
    sweep_a = fit_cells.heldout_r2_sweep(
        rows["X_a"], rows["Y_a"], rows["conv_ids"], n_folds=3, seed=0, null_draws=0
    )
    sweep_u = fit_cells.heldout_r2_sweep(
        rows["X_u"], rows["Y_u"], rows["conv_ids"], n_folds=3, seed=0, null_draws=0
    )
    assert np.array_equal(sweep_a["folds"], sweep_u["folds"])
    assert np.array_equal(sweep_a["folds"], fit_cells._cv_folds(rows["conv_ids"], 3, 0))


def test_allowlist_intersection_before_fold_assignment():
    slots, profiles, nll, conv_ids = _synth_bundle(n=9)
    allowlist = [f"c{i}" for i in range(6)]  # drop c6..c8
    rows = rc.assemble_pair_rows(slots, profiles, nll, conv_ids, allowlist)
    assert rows["n_joint"] == 6
    assert set(rows["conv_ids"]) == set(allowlist)
    assert rows["allowlist"] == {
        "applied": True,
        "n_allowlist": 6,
        "n_allow_in_bundle": 6,
        "n_allow_after_joint_keep": 6,
    }
    # Folds are a function of the POST-intersection conv-id set (plan §4.2
    # step 3: intersect BEFORE fold assignment) — NOT a subset of the full-set
    # fold assignment.
    folds = fit_cells._cv_folds(rows["conv_ids"], 3, 0)
    sweep = fit_cells.heldout_r2_sweep(
        rows["X_u"], rows["Y_u"], rows["conv_ids"], n_folds=3, seed=0, null_draws=0
    )
    assert np.array_equal(sweep["folds"], folds)


def test_allowlist_str_int_coercion():
    slots, profiles, nll, _ = _synth_bundle(n=6)
    conv_ids = np.arange(6)  # ints in the bundle
    rows = rc.assemble_pair_rows(slots, profiles, nll, conv_ids, ["0", "2", 4])
    assert rows["n_joint"] == 3
    assert sorted(int(c) for c in rows["conv_ids"]) == [0, 2, 4]


def test_conv_row_uniqueness_assert():
    slots, profiles, nll, conv_ids = _synth_bundle(n=6)
    conv_ids = conv_ids.copy()
    conv_ids[5] = conv_ids[0]  # duplicate conversation id
    with pytest.raises(AssertionError, match="conv==row"):
        rc.assemble_pair_rows(slots, profiles, nll, conv_ids)


# ---------------------------------------------------------------------------
# Batched-vs-serial paired bootstrap equivalence (BINDING; pins own-mean
# centering on a NONZERO-MEAN synthetic with duplicate rows)
# ---------------------------------------------------------------------------


def _nonzero_mean_case(n: int = 12, dim: int = 5, n_boot: int = 40):
    rng = np.random.default_rng(7)
    y_a = rng.normal(loc=5.0, size=(n, dim))  # NONZERO mean — load-bearing
    y_u = rng.normal(loc=-3.0, size=(n, dim))
    p_a = y_a + rng.normal(scale=0.5, size=(n, dim))
    p_u = y_u + rng.normal(scale=1.5, size=(n, dim))
    idx = rc.draw_index_matrix(n, n_boot, seed=123)
    idx[0] = 0  # a draw of ALL-duplicate rows (extreme own-mean shift)
    idx[1, : n // 2] = 1  # heavy duplication in a second draw
    assert any(len(np.unique(row)) < n for row in idx), "duplicates required"
    return p_a, y_a, p_u, y_u, idx


def test_batched_matches_serial_oracle_within_1e8():
    p_a, y_a, p_u, y_u, idx = _nonzero_mean_case()
    w = rc.counts_from_indices(idx, len(y_a))
    batched = rc.paired_bootstrap_batched(p_a, y_a, p_u, y_u, w)
    serial = rc.paired_bootstrap_serial_reference(p_a, y_a, p_u, y_u, idx)
    for key in ("assistant", "user", "delta"):
        # idx[0] is a degenerate all-one-row draw: ss_tot < 1e-12 -> NaN in
        # BOTH implementations (the _pooled_r2 guard) — compare NaN-aligned.
        b, s = batched[key], serial[key]
        assert np.array_equal(np.isnan(b), np.isnan(s)), key
        finite = ~np.isnan(b)
        assert finite.sum() >= len(b) - 2
        assert float(np.max(np.abs(b[finite] - s[finite]))) < 1e-8, key


def test_fixed_center_subset_sum_cannot_pass_the_gate():
    """The centering convention is LOAD-BEARING: a fixed-center SS_tot (full-
    sample mean instead of the resample's own mean) diverges from the serial
    oracle on a nonzero-mean synthetic with duplicated rows (binding critic
    Must-Fix — plan §4.2 step 7)."""
    p_a, y_a, p_u, y_u, idx = _nonzero_mean_case()
    w = rc.counts_from_indices(idx, len(y_a))
    serial = rc.paired_bootstrap_serial_reference(p_a, y_a, p_u, y_u, idx)

    def fixed_center_r2(preds, true):
        y64 = np.asarray(true, dtype=np.float64)
        p64 = np.asarray(preds, dtype=np.float64)
        resid = y64 - p64
        r2_row = np.einsum("nd,nd->n", resid, resid)
        centered = y64 - y64.mean(axis=0)  # FIXED full-sample center (wrong)
        t2_row = np.einsum("nd,nd->n", centered, centered)
        return 1.0 - (w @ r2_row) / (w @ t2_row)

    delta_fixed = fixed_center_r2(p_a, y_a) - fixed_center_r2(p_u, y_u)
    finite = ~np.isnan(serial["delta"])
    diff = float(np.max(np.abs(delta_fixed[finite] - serial["delta"][finite])))
    assert diff > 1e-6, f"fixed-center form unexpectedly matched the oracle ({diff})"


def test_weighted_mean_draws_matches_gather_mean():
    values = np.random.default_rng(3).normal(size=17)
    idx = rc.draw_index_matrix(17, 25, seed=5)
    w = rc.counts_from_indices(idx, 17)
    batched = rc.weighted_mean_draws(values, w)
    serial = np.asarray([values[row].mean() for row in idx])
    assert float(np.max(np.abs(batched - serial))) < 1e-12


# ---------------------------------------------------------------------------
# run_pair end-to-end on a fabricated tiny bundle (one pair)
# ---------------------------------------------------------------------------


def _args(tmp_path: Path, **over) -> argparse.Namespace:
    base = dict(
        out_dir=tmp_path / "out",
        preds_dir=tmp_path / "preds",
        haiku_dir=tmp_path / "ts_haiku",
        real_dir=tmp_path / "ts_real",
        onpolicy_dir=tmp_path / "ts_onpolicy",
        allowlists=tmp_path / "row_allowlists.json",
        committed_haiku=tmp_path / "committed_haiku",
        committed_real=tmp_path / "committed_real",
        committed_onpolicy=tmp_path / "committed_onpolicy",
        folds=3,
        null_draws=2,
        n_boot=15,
        seed=0,
        mlp_budget_s=30,
        equivalence_gate_pairs=0,
        resume=False,
        sentinel=tmp_path / "logs" / "sentinel.json",
        phase="fit",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_run_pair_end_to_end_on_fabricated_bundle(tmp_path):
    args = _args(tmp_path)
    rc.cmd_fabricate_smoke(args)
    pair = {
        "pair_id": "pair_haiku_instruct_chat",
        "provenance": "haiku",
        "model": "instruct",
        "format": "chat",
    }
    payload = rc.run_pair(pair, args, {}, equivalence_gate=True)
    assert payload["n_joint"] == 9
    assert payload["equivalence_gate"] is not None
    assert all(rec["pass"] for rec in payload["equivalence_gate"].values())
    for li in ("19", "26"):
        row = payload["delta_r2_frozen"][li]
        assert np.isfinite(row["delta_obs"]) and np.isfinite(row["ci_lo"])
        assert len(payload["delta_r2_distribution"][li]) == 15
        assert len(payload["cosine_delta"][li]["per_row"]) == 9
    assert len(payload["nll_delta"]["per_row"]) == 9
    out = args.out_dir / "haiku"
    for role in ("assistant", "user"):
        assert (out / f"cells_M_instruct_{role}_chat.json").exists()
        assert (out / f"nulls_M_instruct_{role}_chat.json").exists()
    assert (args.preds_dir / "preds_pair_haiku_instruct_chat.npz").exists()
    manifest = json.loads((args.out_dir / "preds_manifest.json").read_text())
    assert "preds_pair_haiku_instruct_chat.npz" in manifest["files"]


def test_run_pair_onpolicy_applies_allowlist(tmp_path):
    args = _args(tmp_path)
    rc.cmd_fabricate_smoke(args)
    allow_map = json.loads(args.allowlists.read_text())
    pair = {
        "pair_id": "pair_onpolicy_instruct_chat",
        "provenance": "onpolicy",
        "model": "instruct",
        "format": "chat",
    }
    payload = rc.run_pair(pair, args, allow_map, equivalence_gate=False)
    assert payload["n_joint"] == 8  # fabricate drops the last conv per user cell
    assert payload["allowlist"]["applied"] is True
    assert payload["allowlist"]["n_allowlist"] == 8


# ---------------------------------------------------------------------------
# Gate-path FAILURE statuses (mirrors tests/test_issue825_realuser_gates.py)
# ---------------------------------------------------------------------------

COMMITTED_L19 = {"haiku": 0.075, "real": -0.98, "onpolicy": -0.76}
COMMITTED_N = {"haiku": 40, "real": 40, "onpolicy": 30}


def _cell_payload(l19: float, n: int, with_mlp: bool = True, gate_read: dict | None = None) -> dict:
    payload = {
        "metadata": {"n": n},
        "selection_symmetric": {
            "frozen_layer_table": {
                "19": {"r2_obs": l19, "null_mean": -0.1, "null_p975": 0.05},
                "26": {"r2_obs": l19 + 0.2, "null_mean": -0.1, "null_p975": 0.05},
            },
            "obs_layer_max_r2": 0.1,
            "obs_argmax_layer": 5,
            "null_layer_max_r2_per_draw": [0.0, 0.01],
            "null_layer_max_p975": 0.01,
        },
        "gate_read": gate_read,
    }
    if with_mlp:
        payload["mlp"] = {
            li: {"r2_obs": 0.1, "r2_obs_folds": [0.1, 0.2, 0.15], "budget_hit_folds": []}
            for li in ("19", "26")
        }
    return payload


def _pair_payload(pair: dict, n_joint: int, equivalence: dict | None = None) -> dict:
    delta_row = {
        "delta_obs": 1.2,
        "r2_obs_assistant": 0.1,
        "r2_obs_user": -1.1,
        "delta_pooled_global_obs": 1.19,
        "ci_lo": 1.0,
        "ci_hi": 1.4,
        "se_boot": 0.1,
        "n_draws": 20,
    }
    return {
        "pair": dict(pair),
        "n_joint": n_joint,
        "headline_layers": [19, 26],
        "delta_r2_frozen": {"19": dict(delta_row), "26": dict(delta_row)},
        "per_fold_delta_r2": {},
        "mlp_paired": {"19": {"delta_mean": 0.2}, "26": {"delta_mean": 0.2}},
        "cosine_delta": {"19": {"mean": 0.1}, "26": {"mean": 0.1}},
        "nll_delta": {"mean": -1.5},
        "selection_symmetric": {"delta_of_layer_maxes_descriptive": 0.5},
        "equivalence_gate": equivalence,
    }


@pytest.fixture()
def scaffold(tmp_path, monkeypatch):
    """A fully-PASSING synthetic 12-pair artifact tree; tests break one piece."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    args = _args(tmp_path)
    for prov in rc.PROVENANCES:
        committed_dir = rc._committed_dir(args, prov)
        committed_dir.mkdir(parents=True, exist_ok=True)
        prov_out = args.out_dir / prov
        prov_out.mkdir(parents=True, exist_ok=True)
        n = COMMITTED_N[prov]
        for pair in [p for p in rc.pair_registry() if p["provenance"] == prov]:
            model, fmt = pair["model"], pair["format"]
            eq = (
                {"19": {"max_abs_diff": {"delta": 1e-12}, "tol": 1e-8, "pass": True}}
                if pair["pair_id"] in [p["pair_id"] for p in rc.pair_registry()[:2]]
                else None
            )
            (prov_out / f"{pair['pair_id']}.json").write_text(
                json.dumps(_pair_payload(pair, n, eq))
            )
            for role in rc.ROLES:
                cid = rc.cell_id(model, role, fmt)
                l19 = COMMITTED_L19[prov] + (0.5 if role == "assistant" else 0.0)
                (committed_dir / f"cells_{cid}.json").write_text(json.dumps(_cell_payload(l19, n)))
                (prov_out / f"cells_{cid}.json").write_text(
                    json.dumps(_cell_payload(l19 + 0.004, n))  # within ±0.01
                )
                (prov_out / f"nulls_{cid}.json").write_text(json.dumps({"cell_id": cid}))
    (args.out_dir / "headline_metrics.json").write_text(json.dumps({"followup_label": "x"}))
    (args.out_dir / "preds_manifest.json").write_text(
        json.dumps(
            {"files": {f"preds_{p['pair_id']}.npz": {"sha256": "x"} for p in rc.pair_registry()}}
        )
    )
    return args


def _sentinel(args) -> dict:
    return json.loads(args.sentinel.read_text())


def _assert_failure(args, expected_status: str) -> dict:
    with pytest.raises(SystemExit):
        rc.cmd_gates(args)
    sent = _sentinel(args)
    assert sent["sentinel_schema_version"] == 1  # poll_pipeline contract
    assert sent["kind"] == "epm:results"
    assert sent["status"] == expected_status
    outcomes = json.loads((args.out_dir / "gate_outcomes.json").read_text())
    assert outcomes["all_pass"] is False
    assert outcomes["failure"]["status"] == expected_status
    return sent


def test_all_pass_writes_outcomes(scaffold):
    assert rc.cmd_gates(scaffold) == 0
    outcomes = json.loads((scaffold.out_dir / "gate_outcomes.json").read_text())
    assert outcomes["all_pass"] is True
    assert outcomes["gates"]["reproduction"]["n_gated"] == 20
    assert not scaffold.sentinel.exists()


def test_reproduction_gate_miss(scaffold):
    cid = rc.cell_id("instruct", "user", "chat")
    (scaffold.out_dir / "real" / f"cells_{cid}.json").write_text(
        json.dumps(_cell_payload(COMMITTED_L19["real"] + 0.2, COMMITTED_N["real"]))
    )
    sent = _assert_failure(scaffold, "reproduction_gate_miss")
    assert "rig drift" in sent["note"]["failure"]


def test_reproduction_own_rowset_fallback_passes(scaffold):
    """joint n != committed n + a within-tol own-row-set gate_read -> PASS."""
    cid = rc.cell_id("instruct", "user", "chat")
    payload = _cell_payload(
        COMMITTED_L19["haiku"] + 0.3,  # joint-rowset value WAY off — must be ignored
        COMMITTED_N["haiku"] - 5,  # n mismatch triggers the fallback
        gate_read={
            "r2_l19_own_rowset": COMMITTED_L19["haiku"] + 0.004,
            "n_own_rowset": COMMITTED_N["haiku"],
        },
    )
    (scaffold.out_dir / "haiku" / f"cells_{cid}.json").write_text(json.dumps(payload))
    assert rc.cmd_gates(scaffold) == 0
    table = json.loads((scaffold.out_dir / "gate_outcomes.json").read_text())["gates"][
        "reproduction"
    ]["table"]
    row = next(r for r in table if r["cell_id"] == cid and r["provenance"] == "haiku")
    assert row["gate_value_source"] == "own_rowset_refit"


def test_reproduction_missing_gate_read_is_a_miss(scaffold):
    cid = rc.cell_id("instruct", "user", "chat")
    (scaffold.out_dir / "haiku" / f"cells_{cid}.json").write_text(
        json.dumps(_cell_payload(COMMITTED_L19["haiku"], COMMITTED_N["haiku"] - 5))
    )
    _assert_failure(scaffold, "reproduction_gate_miss")


def test_onpolicy_assistant_is_gate_exempt(scaffold):
    """The 4 onpolicy assistant-on-allowlist cells are NEW fits: off-committed
    values must NOT fail the gate (sandwich-reported, plan §4.2)."""
    cid = rc.cell_id("pretrained", "assistant", "naturalistic")
    (scaffold.out_dir / "onpolicy" / f"cells_{cid}.json").write_text(
        json.dumps(_cell_payload(COMMITTED_L19["onpolicy"] + 0.9, COMMITTED_N["onpolicy"]))
    )
    assert rc.cmd_gates(scaffold) == 0


def test_row_alignment_shortfall(scaffold):
    pair = rc.pair_registry()[4]  # a real pair
    prov = pair["provenance"]
    payload = _pair_payload(pair, int(0.5 * COMMITTED_N[prov]))
    (scaffold.out_dir / prov / f"{pair['pair_id']}.json").write_text(json.dumps(payload))
    _assert_failure(scaffold, "row_alignment_shortfall")


def test_bundle_schema_mismatch_routed_from_deferred(scaffold):
    (scaffold.out_dir / "fit_failures.json").write_text(
        json.dumps(
            [
                {
                    "cell_id": "pair_real_instruct_chat",
                    "error_type": "BundleSchemaError",
                    "error": "profiles shape (9, 2, 28, 8) (need n_turns >= 3)",
                    "status_hint": "bundle_schema_mismatch",
                }
            ]
        )
    )
    _assert_failure(scaffold, "bundle_schema_mismatch")


def test_generic_deferred_failure(scaffold):
    (scaffold.out_dir / "haiku" / "fit_failures.json").write_text(
        json.dumps(
            [
                {
                    "cell_id": "pair_haiku_instruct_chat",
                    "error_type": "ValueError",
                    "error": "boom",
                    "status_hint": None,
                }
            ]
        )
    )
    _assert_failure(scaffold, "fit_deferred_failure")


def _defer_and_unlink(scaffold, pair: dict, error_type: str, status_hint: str | None) -> None:
    """Fabricate the round-2 BLOCKER shape: one pair's terminal JSON missing +
    its deferred record in fit_failures.json (the state cmd_fit leaves after a
    per-pair crash of ANY class)."""
    (scaffold.out_dir / pair["provenance"] / f"{pair['pair_id']}.json").unlink()
    (scaffold.out_dir / "fit_failures.json").write_text(
        json.dumps(
            [
                {
                    "cell_id": pair["pair_id"],
                    "error_type": error_type,
                    "error": "boom",
                    "status_hint": status_hint,
                }
            ]
        )
    )


def test_deferred_failure_reaches_gates_with_registered_status(scaffold):
    """Round-2 BLOCKER deferred-failures-bypass-gates: drive the SAME function
    sequence the production wrapper dispatches (summarize -> [upload] ->
    gates). summarize returning 0 means the wrapper NEVER takes its
    fail-sentinel --phase summarize branch, so the sentinel status is the
    REGISTERED gate status — never summarize_error."""
    pair = rc.pair_registry()[7]  # a real-provenance pair
    _defer_and_unlink(scaffold, pair, "ValueError", None)
    assert rc.cmd_summarize(scaffold) == 0  # tolerated -> wrapper reaches upload + gates
    headline = json.loads((scaffold.out_dir / "headline_metrics.json").read_text())
    block = headline["provenances"][pair["provenance"]]
    assert block["deferred_missing_pairs"] == [pair["pair_id"]]
    assert block["label"] == "INCOMPLETE-DEFERRED-FAILURES"
    assert headline["deferred_failures"]  # the minimal headline carries the deferred set
    sent = _assert_failure(scaffold, "fit_deferred_failure")
    assert sent["status"] != "summarize_error"


def test_deferred_bundle_schema_failure_routes_registered_status(scaffold):
    """BundleSchemaError-classed variant: the plan-registered gate-1 status
    bundle_schema_mismatch is reachable through the production sequence."""
    pair = rc.pair_registry()[10]  # an onpolicy pair
    _defer_and_unlink(scaffold, pair, "BundleSchemaError", "bundle_schema_mismatch")
    assert rc.cmd_summarize(scaffold) == 0
    sent = _assert_failure(scaffold, "bundle_schema_mismatch")
    assert sent["status"] != "summarize_error"


def test_missing_pair_without_deferred_record_still_fails_loud(scaffold):
    """Fail-loud preserved: a missing pair JSON with NO deferred record is a
    run-order bug — summarize raises (status summarize_error is then correct)."""
    pair = rc.pair_registry()[2]
    (scaffold.out_dir / pair["provenance"] / f"{pair['pair_id']}.json").unlink()
    with pytest.raises(AssertionError, match="NO deferred failure"):
        rc.cmd_summarize(scaffold)


def test_mixed_deferred_and_unrecorded_missing_pair_fails_loud(scaffold):
    """Round-3 CONCERN deferred-tolerance-global-flag: the tolerance is
    PER-PAIR. Pure-deferred state (pair A recorded + A missing) tolerates and
    returns 0; pair A's record must NOT license pair B's UNRECORDED missing
    JSON — that stays the hard fail-loud assert (a run-order bug)."""
    pair_a = rc.pair_registry()[7]  # a real-provenance pair
    pair_b = rc.pair_registry()[2]  # a haiku-provenance pair, no record
    _defer_and_unlink(scaffold, pair_a, "ValueError", None)
    assert rc.cmd_summarize(scaffold) == 0  # pure-deferred: still tolerated
    (scaffold.out_dir / pair_b["provenance"] / f"{pair_b['pair_id']}.json").unlink()
    with pytest.raises(AssertionError, match="NO deferred failure"):
        rc.cmd_summarize(scaffold)


def test_fit_defers_real_bundle_schema_error_end_to_end(tmp_path, monkeypatch):
    """cmd_fit with a REAL corrupted bundle (EPS_SMOKE_CORRUPT_PAIR fault
    injector, n_turns=2): the raised BundleSchemaError is deferred with its
    status_hint, summarize tolerates, gates HALT with bundle_schema_mismatch."""
    args = _args(tmp_path)
    pair = {
        "pair_id": "pair_real_instruct_chat",
        "provenance": "real",
        "model": "instruct",
        "format": "chat",
    }
    monkeypatch.setenv("EPS_SMOKE_CORRUPT_PAIR", pair["pair_id"])
    rc.cmd_fabricate_smoke(args)
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    monkeypatch.setattr(rc, "pair_registry", lambda: [pair])
    assert rc.cmd_fit(args) == 0  # deferred, never pre-upload fatal (MF-C)
    failures = json.loads((args.out_dir / "fit_failures.json").read_text())
    assert failures[0]["error_type"] == "BundleSchemaError"
    assert failures[0]["status_hint"] == "bundle_schema_mismatch"
    assert not (args.out_dir / "real" / f"{pair['pair_id']}.json").exists()
    assert rc.cmd_summarize(args) == 0
    with pytest.raises(SystemExit):
        rc.cmd_gates(args)
    sent = json.loads(args.sentinel.read_text())
    assert sent["status"] == "bundle_schema_mismatch"


def test_fit_clears_stale_deferred_record_after_successful_rerun(tmp_path, monkeypatch):
    """Reconciler v6 'observed but not raised': a pair that succeeds on re-run
    drops its stale fit_failures.json record, so gate 1 does not HALT a run
    whose failure was already fixed."""
    args = _args(tmp_path)
    rc.cmd_fabricate_smoke(args)
    pair = {
        "pair_id": "pair_haiku_instruct_chat",
        "provenance": "haiku",
        "model": "instruct",
        "format": "chat",
    }
    monkeypatch.setattr(rc, "pair_registry", lambda: [pair])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fit_failures.json").write_text(
        json.dumps(
            [
                {
                    "cell_id": pair["pair_id"],
                    "error_type": "ValueError",
                    "error": "boom",
                    "status_hint": None,
                }
            ]
        )
    )
    assert rc.cmd_fit(args) == 0
    assert not (args.out_dir / "fit_failures.json").exists()


def test_fail_sentinel_uploads_preds_npz_to_analysis_tensors(tmp_path, monkeypatch):
    """Round-2 CONCERN failure-sentinel-misses-preds-upload: fail-sentinel
    mirrors UPLOAD-b (preds_pair_*.npz + preds_manifest.json ->
    analysis_tensors) BEFORE writing the FAILURE sentinel."""
    import huggingface_hub

    monkeypatch.delenv("EPS_SMOKE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "token-for-test")
    args = _args(tmp_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "some.json").write_text("{}")
    (args.out_dir / "preds_manifest.json").write_text(json.dumps({"files": {}}))
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    (args.preds_dir / "preds_pair_haiku_instruct_chat.npz").write_bytes(b"npz")
    calls: dict[str, list] = {"folders": [], "files": []}

    class FakeApi:
        def upload_folder(self, **kw):
            calls["folders"].append(kw)

        def upload_file(self, **kw):
            calls["files"].append(kw)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert rc.cmd_fail_sentinel(args) == 0
    tensor_calls = [k for k in calls["folders"] if k["path_in_repo"].endswith("/analysis_tensors")]
    assert len(tensor_calls) == 1
    assert tensor_calls[0]["folder_path"] == str(args.preds_dir)
    assert tensor_calls[0]["allow_patterns"] == ["preds_pair_*.npz"]
    assert any(
        k["path_in_repo"].endswith("/analysis_tensors/preds_manifest.json") for k in calls["files"]
    )
    mirror_calls = [
        k for k in calls["folders"] if k["path_in_repo"].endswith("/eval_results_mirror")
    ]
    assert len(mirror_calls) == 1  # UPLOAD-a mirror unchanged
    sent = json.loads(args.sentinel.read_text())
    assert sent["status"] == "fit_error"
    assert sent["note"]["uploaded_preds_before_sentinel"] == ["preds_pair_haiku_instruct_chat.npz"]


def test_coverage_miss_missing_pair_json(scaffold):
    (scaffold.out_dir / "onpolicy" / "pair_onpolicy_pretrained_chat.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_miss_missing_cell(scaffold):
    (scaffold.out_dir / "real" / "cells_M_pretrained_user_naturalistic.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_miss_missing_mlp_block(scaffold):
    cid = rc.cell_id("instruct", "user", "chat")
    (scaffold.out_dir / "real" / f"cells_{cid}.json").write_text(
        json.dumps(_cell_payload(COMMITTED_L19["real"], COMMITTED_N["real"], with_mlp=False))
    )
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_accepts_logged_budget_caps(scaffold):
    """blocks-or-logged-caps: an all-NaN MLP block WITH budget_hit_folds passes."""
    cid = rc.cell_id("instruct", "user", "chat")
    payload = _cell_payload(COMMITTED_L19["real"] + 0.004, COMMITTED_N["real"])
    payload["mlp"] = {
        li: {"r2_obs": float("nan"), "r2_obs_folds": [], "budget_hit_folds": [0, 1, 2]}
        for li in ("19", "26")
    }
    (scaffold.out_dir / "real" / f"cells_{cid}.json").write_text(json.dumps(payload))
    assert rc.cmd_gates(scaffold) == 0


def test_coverage_miss_missing_headline(scaffold):
    (scaffold.out_dir / "headline_metrics.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


def test_coverage_miss_missing_preds_manifest_entry(scaffold):
    manifest = json.loads((scaffold.out_dir / "preds_manifest.json").read_text())
    manifest["files"].pop("preds_pair_real_instruct_chat.npz")
    (scaffold.out_dir / "preds_manifest.json").write_text(json.dumps(manifest))
    _assert_failure(scaffold, "coverage_miss")


def test_equivalence_gate_miss_on_recorded_failure(scaffold):
    pair = rc.pair_registry()[0]
    payload = _pair_payload(
        pair,
        COMMITTED_N[pair["provenance"]],
        {"19": {"max_abs_diff": {"delta": 1e-3}, "tol": 1e-8, "pass": False}},
    )
    (scaffold.out_dir / pair["provenance"] / f"{pair['pair_id']}.json").write_text(
        json.dumps(payload)
    )
    _assert_failure(scaffold, "equivalence_gate_miss")


def test_smoke_bypasses_numeric_gates_but_structural_binds(scaffold, monkeypatch):
    monkeypatch.setenv("EPS_SMOKE", "1")
    # Numeric misses everywhere: off-tolerance refit + tiny n_joint.
    cid = rc.cell_id("instruct", "user", "chat")
    (scaffold.out_dir / "real" / f"cells_{cid}.json").write_text(json.dumps(_cell_payload(5.0, 9)))
    for pair in rc.pair_registry():
        prov = pair["provenance"]
        eq = (
            {"19": {"max_abs_diff": {"delta": 1e-12}, "tol": 1e-8, "pass": True}}
            if pair["pair_id"] in [p["pair_id"] for p in rc.pair_registry()[:2]]
            else None
        )
        (scaffold.out_dir / prov / f"{pair['pair_id']}.json").write_text(
            json.dumps(_pair_payload(pair, 9, eq))
        )
    assert rc.cmd_gates(scaffold) == 0
    outcomes = json.loads((scaffold.out_dir / "gate_outcomes.json").read_text())
    assert outcomes["gates"]["reproduction"]["result"] == "BYPASSED_SMOKE_PRESENCE_ONLY"
    assert outcomes["gates"]["row_alignment"]["result"] == "BYPASSED_SMOKE_PRESENCE_ONLY"
    # Structural still binding at smoke:
    (scaffold.out_dir / "headline_metrics.json").unlink()
    _assert_failure(scaffold, "coverage_miss")


def test_smoke_requires_equivalence_records_on_first_two_pairs(scaffold, monkeypatch):
    monkeypatch.setenv("EPS_SMOKE", "1")
    pair = rc.pair_registry()[1]
    (scaffold.out_dir / pair["provenance"] / f"{pair['pair_id']}.json").write_text(
        json.dumps(_pair_payload(pair, 9, None))  # record MISSING at smoke
    )
    _assert_failure(scaffold, "equivalence_gate_miss")


def test_success_sentinel_refuses_without_all_pass(scaffold):
    with pytest.raises(AssertionError):
        rc.cmd_success_sentinel(scaffold)
    (scaffold.out_dir / "gate_outcomes.json").write_text(
        json.dumps({"all_pass": False, "failure": {"status": "reproduction_gate_miss"}})
    )
    with pytest.raises(AssertionError):
        rc.cmd_success_sentinel(scaffold)
    assert not scaffold.sentinel.exists()


def test_no_cross_provenance_join(scaffold):
    """Provenance-scoped keying (plan §8): identical cell ids + conv-id ranges
    across provenances must resolve against EACH provenance's own committed
    dir — the scaffold's per-provenance committed L19 values differ, and the
    gate PASSes only because refits are compared within-provenance."""
    table = rc.build_gate_table(scaffold)
    assert len(table) == 24
    assert sum(1 for r in table if r["gated"]) == 20
    by_prov = {
        prov: {r["cell_id"]: r for r in table if r["provenance"] == prov} for prov in rc.PROVENANCES
    }
    cid = rc.cell_id("instruct", "user", "chat")
    committed = {prov: by_prov[prov][cid]["committed_r2_l19"] for prov in rc.PROVENANCES}
    assert committed["haiku"] != committed["real"] != committed["onpolicy"]
