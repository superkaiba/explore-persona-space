"""Network-free, GPU-free pins for the v15 G-E gate instrument fix (task #2546).

Round-15 crash-fix regression (task #2546 arm-1 G-E gate FAIL, epm:failure
2026-08-26T11:08Z; root cause in task marker v88): the gate compared the RIGHT
number against the WRONG reference. `issue825_fit_cells.heldout_r2_sweep`
defaults `lambda_selection="inner-group-cv"` since the #1887 defaults flip, so
the reused #1336 `run_g0` (which passes no `lambda_selection`/`lambdas`)
silently stopped measuring the legacy-GCV 0.6731 anchor. A second defect
compounded it: our driver set `fc.N_INNER_LAMBDA_FOLDS = 2` as a PERMANENT
module global at `main()` entry, BEFORE the `--g0` branch, so the delegated
gate inherited the P5 fit patch. Measured three-way separation at the pinned
#825 Qwen S1 cell (same bundle, layer 19, seed 0, n 5000):

    legacy GCV, logspace(-2,4,13)   0.6730940896676356   abs_dev 5.9e-06
    inner-group-cv, 2 inner folds   0.6935026836671432   abs_dev 0.0204
    inner-group-cv, 4 inner folds   0.6957042061410352   abs_dev 0.0226

The fold count really moves the gate's number (0.6957 -> 0.6935, delta
2.2e-3), so the patch-scoping pins here guard a real effect — and NEITHER
fold count reaches the 0.6731 anchor (the anchor depends on the SELECTOR +
grid, not folds), so leg (a) must pin all five legacy values explicitly.

The v15 fixes these pins protect:

- ``run_g0_gate``: two ENFORCED legs, every estimator knob pinned per leg —
  leg (a) legacy anchor (gcv, 13-pt grid, GCV_DOF_CAP=None,
  LEGACY_UNGUARDED_GCV=True, FORCE_GRAM=True) vs 0.6731 +/- 0.01; leg (b)
  v2-recipe identity (inner-group-cv, 23-pt grid, N_INNER_LAMBDA_FOLDS=2,
  FORCE_GRAM=False) vs 0.6935026836671432 +/- 1e-6. rc 3 if EITHER enforced
  leg fails (the dispatcher-failure-branch-no-pin-test concern's sibling:
  a future edit silently dropping a leg fails these pins).
- ``_p5_inner_folds_patch`` scoping: the RECORDED fc.N_INNER_LAMBDA_FOLDS
  patch wraps ``run_units`` ONLY (save-and-restore) — the ``--g0`` path
  never touches the global, and the patch cannot leak past run_units.

Production-body coverage (code-style.md "one production-body test per
seam-stubbed function"): the fixture end-to-end tests drive ``F.main`` ->
``run_g0_gate`` -> the REAL ``_g0_xy_at_gate_layer`` body (real
``fc._load_bundle_any`` on a real on-disk bundle) and the REAL
``fc.heldout_r2_sweep`` body (the recording wrapper delegates to it). The
enforcement-matrix tests stub ``_g0_xy_at_gate_layer`` via
``unittest.mock.create_autospec`` and fake ``heldout_r2_sweep`` with a def
mirroring its real signature — both at the data boundary only.
"""

from __future__ import annotations

import json
import sys
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_fit_cells as fc  # noqa: E402
import issue2546_fit_cells as F  # noqa: E402

LEGACY_ANCHOR = 0.6731
V2_ANCHOR = 0.6935026836671432
FOLD4_PROBE = 0.6957042061410352


# ---------------------------------------------------------------------------
# Fixture bundle (the issue1336_smoke_fixtures cmd_g0_fixture payload shape)
# ---------------------------------------------------------------------------
def _write_g0_fixture(out: Path, n: int = 48, layers: int = 4, dim: int = 8) -> int:
    """Tiny synthetic Qwen-S1 stand-in bundle; returns the clamped gate layer.

    No anchor calibration needed: fixture mode demotes both anchors to
    informational (#1345 gate-calibration rule), and these tests assert PINS
    and VERDICT ROUTING, never fixture-n R^2 values.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    w = (rng.normal(size=(dim, dim)) / np.sqrt(dim)).astype(np.float32)
    y = (x @ w + 0.5 * rng.normal(size=(n, dim)).astype(np.float32)).astype(np.float32)
    conv = [f"g{i}" for i in range(n)]
    layer = min(19, layers - 1)  # run_g0_gate's fixture clamp lands here
    filler = np.random.default_rng(1)
    slots, profiles = [], []
    for i in range(n):
        s = filler.normal(size=(2, layers, dim)).astype(np.float32)
        p = filler.normal(size=(2, layers, dim)).astype(np.float32)
        s[0, layer, :] = x[i]  # the gate reads slot_index 0
        p[1, layer, :] = y[i]  # ... -> target_turn_index 1
        slots.append(torch.tensor(s))
        profiles.append(torch.tensor(p))
    payload = {
        "conv_ids": conv,
        "slots": slots,
        "profiles": profiles,
        "nll": [torch.tensor([1.0, 1.0]) for _ in range(n)],
        "spans_meta": [{"conv_id": c} for c in conv],
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "instruct_chat_s_shard000.pt")
    (out / "instruct_chat_s_shard000.json").write_text(
        json.dumps({"stem": "instruct_chat_s", "n": n, "fixture": True})
    )
    return layer


def _run_gate_on_fixture(tmp_path: Path, monkeypatch) -> tuple[int, list[dict], dict]:
    """Drive F.main --g0 on the fixture with a pin-recording REAL-sweep wrapper."""
    fx = tmp_path / "g0fix"
    _write_g0_fixture(fx)
    out_root = tmp_path / "outroot"
    real_sweep = fc.heldout_r2_sweep
    seen: list[dict] = []

    def recording_sweep(X, Y, conv_ids, **kwargs):
        seen.append(
            {
                "lambda_selection": kwargs.get("lambda_selection"),
                "lambdas": np.asarray(kwargs.get("lambdas")),
                "n_inner": fc.N_INNER_LAMBDA_FOLDS,
                "gcv_dof_cap": fc.GCV_DOF_CAP,
                "legacy_unguarded": fc.LEGACY_UNGUARDED_GCV,
                "force_gram": fc.FORCE_GRAM,
            }
        )
        return real_sweep(X, Y, conv_ids, **kwargs)  # REAL fit body

    monkeypatch.setattr(fc, "heldout_r2_sweep", recording_sweep)
    rc = F.main(["--g0", "--arm", "1", "--g0-local-dir", str(fx), "--out-root", str(out_root)])
    payload = json.loads((out_root / "out" / "gates" / "g0_gate.json").read_text())
    return rc, seen, payload


# ---------------------------------------------------------------------------
# Brief test 1 — the --g0 path is NOT affected by the P5 patch; leg-1 pins
# are the five legacy values. FAILS pre-v15: main() overwrote
# fc.N_INNER_LAMBDA_FOLDS to 2 before the --g0 branch, and the delegated
# f36.run_g0 made ONE un-pinned sweep call instead of two pinned legs.
# ---------------------------------------------------------------------------
def test_g0_path_leaves_inner_folds_untouched_and_pins_both_legs(tmp_path, monkeypatch):
    sentinel = 7  # a value that is neither the module default (4) nor the patch (2)
    monkeypatch.setattr(fc, "N_INNER_LAMBDA_FOLDS", sentinel)
    saved_defaults = (fc.GCV_DOF_CAP, fc.LEGACY_UNGUARDED_GCV, fc.FORCE_GRAM)
    rc, seen, _payload = _run_gate_on_fixture(tmp_path, monkeypatch)
    assert rc == 0  # fixture mode: anchors informational (#1345)
    assert [c["lambda_selection"] for c in seen] == ["gcv", "inner-group-cv"]
    leg_a, leg_b = seen
    # Leg (a) runs with the module global UNTOUCHED by the driver (the gcv
    # path never reads it; pre-v15 the entry-wide patch had set it to 2).
    assert leg_a["n_inner"] == sentinel
    # ... and the five legacy pins engaged at fit time:
    assert leg_a["gcv_dof_cap"] is None
    assert leg_a["legacy_unguarded"] is True
    assert leg_a["force_gram"] is True
    assert np.allclose(leg_a["lambdas"], np.logspace(-2, 4, 13))
    # Leg (b): the explicit v2-recipe pins, not inherited state.
    assert leg_b["n_inner"] == F.INNER_LAMBDA_FOLDS == 2
    assert leg_b["force_gram"] is False
    assert np.allclose(leg_b["lambdas"], np.logspace(-3, 8, 23))
    # Globals restored after the gate (save-and-restore, never permanent).
    assert sentinel == fc.N_INNER_LAMBDA_FOLDS
    assert saved_defaults == (fc.GCV_DOF_CAP, fc.LEGACY_UNGUARDED_GCV, fc.FORCE_GRAM)


def test_g0_leg1_runs_at_module_default_inner_folds(tmp_path, monkeypatch):
    """The brief's literal reading: leg-1 fit sees fc's import-time default."""
    default_at_start = fc.N_INNER_LAMBDA_FOLDS
    assert default_at_start != F.INNER_LAMBDA_FOLDS, (
        "precondition lost: the shared core's default now equals the P5 patch "
        "value — this pin can no longer discriminate patch leakage"
    )
    rc, seen, _ = _run_gate_on_fixture(tmp_path, monkeypatch)
    assert rc == 0
    assert seen[0]["lambda_selection"] == "gcv"
    assert seen[0]["n_inner"] == default_at_start
    assert default_at_start == fc.N_INNER_LAMBDA_FOLDS


def test_g0_gate_json_mirrors_g0v2_shape(tmp_path, monkeypatch):
    """Per-leg pins + anchors recorded exactly (g0v2.json's per-leg shape)."""
    rc, _, payload = _run_gate_on_fixture(tmp_path, monkeypatch)
    assert rc == 0
    assert payload["gate"] == "G-E"
    assert payload["stem"] == "instruct_chat_s"
    assert payload["local_dir_fixture"] is True
    assert payload["pass"] is True
    for leg in ("leg_a_legacy", "leg_b_v2_identity"):
        assert payload[leg]["enforced"] is False  # fixture: informational
        assert {"r2", "committed_r2", "tol", "abs_dev", "pass", "pins"} <= set(payload[leg])
    assert payload["leg_a_legacy"]["committed_r2"] == LEGACY_ANCHOR
    assert payload["leg_a_legacy"]["tol"] == 0.01
    assert payload["leg_a_legacy"]["pins"] == {
        "lambda_selection": "gcv",
        "GCV_DOF_CAP": None,
        "LEGACY_UNGUARDED_GCV": True,
        "FORCE_GRAM": True,
        "grid": "logspace(-2,4,13)",
    }
    assert payload["leg_b_v2_identity"]["committed_r2"] == V2_ANCHOR
    assert payload["leg_b_v2_identity"]["tol"] == 1e-6
    assert payload["leg_b_v2_identity"]["pins"] == {
        "lambda_selection": "inner-group-cv",
        "N_INNER_LAMBDA_FOLDS": 2,
        "FORCE_GRAM": False,
        "grid": "logspace(-3,8,23)",
    }
    probe = payload["fold_sensitivity_probe"]
    assert probe["enforced"] is False
    assert probe["r2_inner_group_cv_4fold"] == FOLD4_PROBE


# ---------------------------------------------------------------------------
# Brief test 2 — both legs ENFORCED in production mode; rc != 0 if EITHER
# fails. FAILS pre-v15 (run_g0_gate did not exist; the delegated gate had
# one un-pinned leg).
# ---------------------------------------------------------------------------
def _fake_sweep_factory(r2_by_selection: dict[str, float]):
    def fake_heldout_r2_sweep(
        X_layers,
        Y_layers,
        conv_ids,
        *,
        n_folds,
        seed,
        null_draws,
        collect_cosines=True,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        _null_impl="batched",
        frozen_layers=None,
        lambdas=None,
        reduced_basis_companion=True,
    ):
        # def mirrors issue825_fit_cells.heldout_r2_sweep (signature-conformant
        # by construction; keyword names verified against the real def).
        return {"r2_obs": np.asarray([r2_by_selection[lambda_selection]])}

    return fake_heldout_r2_sweep


@pytest.mark.parametrize(
    ("r2_legacy", "r2_v2", "want_rc"),
    [
        # both anchors hit -> PASS
        (LEGACY_ANCHOR, V2_ANCHOR, 0),
        # TODAY'S incident shape: the v2 number where the legacy anchor
        # belongs -> leg (a) fails -> rc 3
        (V2_ANCHOR, V2_ANCHOR, 3),
        # the measured 4-fold probe value on leg (b) -> leg (b) fails -> rc 3
        # (delta 2.2e-3 >> 1e-6: the fold-count contamination is caught)
        (LEGACY_ANCHOR, FOLD4_PROBE, 3),
        # both wrong -> rc 3
        (0.60, 0.60, 3),
        # leg (a) just outside +/-0.01 -> rc 3
        (LEGACY_ANCHOR + 0.011, V2_ANCHOR, 3),
        # leg (b) inside the 1e-6 tolerance -> PASS (tolerance, not equality)
        (LEGACY_ANCHOR, V2_ANCHOR + 5e-7, 0),
    ],
)
def test_g0_gate_enforces_both_legs_in_production_mode(
    tmp_path, monkeypatch, r2_legacy, r2_v2, want_rc
):
    ns = SimpleNamespace(
        g0_probe_only=False,
        g0_local_dir=None,
        g0_dl_dir=tmp_path / "dl",
        out_dir=tmp_path / "out",
    )
    n, d = 24, 4
    rng = np.random.default_rng(0)
    xy = (
        rng.normal(size=(n, 1, d)).astype(np.float32),
        rng.normal(size=(n, 1, d)).astype(np.float32),
        np.asarray([f"c{i}" for i in range(n)]),
        19,
        False,  # production mode -> both legs ENFORCED
    )
    monkeypatch.setattr(
        F,
        "_g0_xy_at_gate_layer",
        unittest.mock.create_autospec(F._g0_xy_at_gate_layer, return_value=xy),
    )
    monkeypatch.setattr(
        fc,
        "heldout_r2_sweep",
        _fake_sweep_factory({"gcv": r2_legacy, "inner-group-cv": r2_v2}),
    )
    entry = (fc.GCV_DOF_CAP, fc.LEGACY_UNGUARDED_GCV, fc.FORCE_GRAM, fc.N_INNER_LAMBDA_FOLDS)
    rc = F.run_g0_gate(ns)
    assert rc == want_rc
    # save-and-restore held in production mode too
    assert (entry) == (
        fc.GCV_DOF_CAP,
        fc.LEGACY_UNGUARDED_GCV,
        fc.FORCE_GRAM,
        fc.N_INNER_LAMBDA_FOLDS,
    )
    payload = json.loads((tmp_path / "out" / "gates" / "g0_gate.json").read_text())
    assert payload["leg_a_legacy"]["enforced"] is True
    assert payload["leg_b_v2_identity"]["enforced"] is True
    assert payload["pass"] is (want_rc == 0)
    # the verdict is the CONJUNCTION of the legs — dropping either leg from
    # the verdict breaks this pin
    assert payload["pass"] == (
        payload["leg_a_legacy"]["pass"] and payload["leg_b_v2_identity"]["pass"]
    )


# ---------------------------------------------------------------------------
# Brief test 3 — the P5 patch is scoped to run_units and restored after the
# fits block (no leakage into any later call). FAILS pre-v15
# (_p5_inner_folds_patch did not exist; the patch was a permanent global).
# ---------------------------------------------------------------------------
def test_p5_patch_context_sets_and_restores(monkeypatch):
    monkeypatch.setattr(fc, "N_INNER_LAMBDA_FOLDS", 7)
    with F._p5_inner_folds_patch():
        assert fc.N_INNER_LAMBDA_FOLDS == F.INNER_LAMBDA_FOLDS == 2
    assert fc.N_INNER_LAMBDA_FOLDS == 7


def test_p5_patch_restores_on_exception(monkeypatch):
    monkeypatch.setattr(fc, "N_INNER_LAMBDA_FOLDS", 7)
    with pytest.raises(RuntimeError, match="boom"), F._p5_inner_folds_patch():
        assert fc.N_INNER_LAMBDA_FOLDS == 2
        raise RuntimeError("boom")
    assert fc.N_INNER_LAMBDA_FOLDS == 7


def test_run_units_applies_patch_and_restores_even_on_failure(tmp_path, monkeypatch):
    """run_units WIRES the patch: a probe at the first body call sees the
    patched value, and the raise path still restores the entry value."""
    monkeypatch.setattr(fc, "N_INNER_LAMBDA_FOLDS", 7)
    seen: dict[str, int] = {}

    def probe_store_content_key(out_root, prof, smoke):  # mirrors _store_content_key
        seen["folds_at_body_entry"] = fc.N_INNER_LAMBDA_FOLDS
        raise RuntimeError("short-circuit after probe")

    monkeypatch.setattr(F, "_store_content_key", probe_store_content_key)
    args = SimpleNamespace(
        out_root=tmp_path,
        smoke=True,
        null_draws=None,
        n_boot=None,
        prefill_fallback=False,
        decode_fallback=False,
    )
    prof = F.profile_for_arm(1)
    with pytest.raises(RuntimeError, match="short-circuit"):
        F.run_units(args, prof, [])
    assert seen["folds_at_body_entry"] == 2
    assert fc.N_INNER_LAMBDA_FOLDS == 7
