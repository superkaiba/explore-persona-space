"""#1417 registered-selector refit round — selector threading + versioned outputs.

Covers the three default-preserving source-module changes (fit825 run_cell
threading + selector record; ma/cm module-global inner-group-CV + GCV dof cap)
and the battery CLI's refit mode (versioned refit out-dir, v1 judge kept-sets
read-only, per-fit selector logging). All tests are CPU, real code bodies, no
mocks; the e2e drives the production entrypoint in a fresh subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue1417_render as r1417  # noqa: E402

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Degenerate fixture: n_tr < D with near-duplicate rows — the #1335 regime
# where GCV collapses at the grid-min lambda.
# ---------------------------------------------------------------------------
def _degenerate_xy(n: int = 12, d: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """n_tr < D: the train Gram is near-full-rank with every eigenvalue >> the
    grid-min lambda, so the grid-min (near-)interpolates (dof -> n_tr) and the
    GCV objective degenerates — the #1335/#1310 regime the dof cap targets."""
    X = RNG.standard_normal((n, d))
    W = RNG.standard_normal((d, d)) / np.sqrt(d)
    Y = X @ W + 0.1 * RNG.standard_normal((n, d))
    return X.astype(np.float32), Y.astype(np.float32)


@pytest.fixture
def _restore_globals():
    """Snapshot + restore every module-global selector knob (test hygiene)."""
    saved = (
        fit825.GCV_DOF_CAP,
        ma.GCV_DOF_CAP,
        ma.LAMBDA_SELECTION,
        ma.SELECTOR_LOG,
        cm.GCV_DOF_CAP,
        cm.LAMBDA_SELECTION,
        cm.SELECTOR_LOG,
    )
    yield
    (
        fit825.GCV_DOF_CAP,
        ma.GCV_DOF_CAP,
        ma.LAMBDA_SELECTION,
        ma.SELECTOR_LOG,
        cm.GCV_DOF_CAP,
        cm.LAMBDA_SELECTION,
        cm.SELECTOR_LOG,
    ) = saved


def test_inner_fold_constant_matches_fit825():
    assert cm.N_INNER_LAMBDA_FOLDS == fit825.N_INNER_LAMBDA_FOLDS
    assert ma.N_INNER_LAMBDA_FOLDS == fit825.N_INNER_LAMBDA_FOLDS


def test_gcv_dof_cap_excludes_interpolating_lambda(_restore_globals):
    """On the near-singular fixture the uncapped GCV picks the grid-min lambda;
    the 0.9 dof cap excludes the (near-)interpolating lambdas (fails pre-fix:
    without the cap wiring both selections return LAMBDAS[0])."""
    X, Y = _degenerate_xy()
    cache = fit825._prep_fold(X, X[:2])
    fit825.GCV_DOF_CAP = None
    # #1887: uncapped pure GCV at n_train < d now needs the explicit legacy
    # opt-in (the refusal guard); the pinned degeneracy itself is unchanged.
    fit825.LEGACY_UNGUARDED_GCV = True
    try:
        _, lam_uncapped = fit825._ridge_predict_cached(cache, Y, return_lam=True)
    finally:
        fit825.LEGACY_UNGUARDED_GCV = False
    fit825.GCV_DOF_CAP = 0.9
    _, lam_capped = fit825._ridge_predict_cached(cache, Y, return_lam=True)
    assert lam_uncapped == float(fit825.LAMBDAS[0])  # degenerate GCV minimum
    assert lam_capped > lam_uncapped
    # the capped lambda's effective dof respects the cap
    w = cache["w"]
    dof = float((w / (w + lam_capped)).sum())
    assert dof <= 0.9 * cache["ntr"] + 1e-9


def test_inner_group_cv_selects_off_grid_min(_restore_globals):
    """Inner-group-CV on the degenerate fixture selects a generalization lambda
    away from the interpolating grid minimum (the registered #1335 mitigation)."""
    X, Y = _degenerate_xy()
    cache = fit825._prep_fold(X, X[:2])
    cache["inner"] = fit825._prep_inner_lambda(X, np.arange(len(X)), 4, 4242)
    assert cache["inner"] is not None and len(cache["inner"]) >= 2
    pred, lam_inner = fit825._ridge_predict_cached(cache, Y, return_lam=True)
    assert np.isfinite(pred).all()
    assert lam_inner > float(fit825.LAMBDAS[0])


def test_prep_inner_lambda_too_few_groups_returns_none():
    """Degenerate-input probe for the WARN-and-fall-back gate: <2 usable inner
    group folds -> None (the caller's registered GCV fallback)."""
    X, _ = _degenerate_xy(n=6)
    assert fit825._prep_inner_lambda(X, np.zeros(6, dtype=int), 4, 0) is None


def test_heldout_sweep_selector_record_inner_and_fallback(_restore_globals):
    """heldout_r2_sweep(collect_lambdas=True) records the selector per
    (layer, fold): inner-group-cv on healthy folds, gcv-fallback when the
    train block has <2 groups (the loud-WARN branch)."""
    n, n_layers, d = 20, 2, 6
    X = RNG.standard_normal((n, n_layers, d)).astype(np.float32)
    Y = RNG.standard_normal((n, n_layers, d)).astype(np.float32)
    conv = np.array([f"s{i}" for i in range(n)])
    sw = fit825.heldout_r2_sweep(
        X,
        Y,
        conv,
        n_folds=4,
        seed=0,
        null_draws=2,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
    )
    sels = {s for row in sw["lambda_selector"] for s in row if s is not None}
    assert sels == {"inner-group-cv"}
    assert sw["gcv_lambda"] is not None

    # fallback branch: 2 conv groups -> each train block has 1 group
    conv2 = np.array(["a"] * 3 + ["b"] * 3)
    X2 = RNG.standard_normal((6, 1, 4)).astype(np.float32)
    Y2 = RNG.standard_normal((6, 1, 4)).astype(np.float32)
    sw2 = fit825.heldout_r2_sweep(
        X2,
        Y2,
        conv2,
        n_folds=2,
        seed=0,
        null_draws=0,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
    )
    sels2 = {s for row in sw2["lambda_selector"] for s in row if s is not None}
    assert sels2 == {"gcv-fallback"}


def test_ma_default_ridge_predict_byte_identical(_restore_globals):
    """Default globals keep ma._ridge_predict byte-identical to the committed
    pure-GCV algorithm (oracle inlined from the pre-change implementation)."""
    ma.LAMBDA_SELECTION = "gcv"
    ma.GCV_DOF_CAP = None
    ma.SELECTOR_LOG = None
    X = torch.tensor(RNG.standard_normal((15, 6)), dtype=torch.float64)
    Y = torch.tensor(RNG.standard_normal((15, 6)), dtype=torch.float64)
    Xe = torch.tensor(RNG.standard_normal((4, 6)), dtype=torch.float64)
    prep = ma._ridge_prep(X)
    got = ma._ridge_predict(prep, Y, Xe)

    # committed pure-GCV oracle (verbatim pre-change body)
    w, V, Xn, xmu, xsd, ntr = (
        prep["w"],
        prep["V"],
        prep["Xn"],
        prep["xmu"],
        prep["xsd"],
        prep["ntr"],
    )
    ymu = Y.mean(0)
    Ytr_c = Y - ymu
    VtY = V.T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    best_lam, best_gcv = float(ma.LAMBDAS[0]), float("inf")
    for lam in ma.LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    Xev_n = (Xe - xmu) / xsd
    KevV = (Xev_n @ Xn.T) @ V
    expect = (KevV * (1.0 / (w + best_lam))) @ VtY + ymu
    assert torch.equal(got, expect)
    assert "inner" not in prep  # default mode builds no inner caches

    # lam_key memoization is bit-identical to the memo-free path
    memo1 = ma._ridge_predict(prep, Y, Xe, lam_key="k")
    memo2 = ma._ridge_predict(prep, Y, Xe, lam_key="k")  # served from memo
    assert torch.equal(memo1, got) and torch.equal(memo2, got)


def test_ma_inner_mode_attaches_and_logs(_restore_globals):
    ma.LAMBDA_SELECTION = "inner-group-cv"
    ma.GCV_DOF_CAP = 0.9
    ma.SELECTOR_LOG = {}
    X = torch.tensor(RNG.standard_normal((16, 5)), dtype=torch.float64)
    Y = torch.tensor(RNG.standard_normal((16, 5)), dtype=torch.float64)
    prep = ma._ridge_prep(X)
    assert prep.get("inner"), "inner caches must attach under inner-group-cv"
    pred = ma._ridge_predict(prep, Y, X[:3])
    assert torch.isfinite(pred).all()
    assert "inner-group-cv" in ma.SELECTOR_LOG
    assert sum(ma.SELECTOR_LOG["inner-group-cv"].values()) == 1


def test_cm_frozen_map_swap_inner_mode(_restore_globals):
    cm.LAMBDA_SELECTION = "inner-group-cv"
    cm.GCV_DOF_CAP = 0.9
    cm.SELECTOR_LOG = {}
    n, n_layers, d = 20, 2, 5
    Xs = RNG.standard_normal((n, n_layers, d)).astype(np.float32)
    Ys = RNG.standard_normal((n, n_layers, d)).astype(np.float32)
    Xt = RNG.standard_normal((n, n_layers, d)).astype(np.float32)
    Yt = RNG.standard_normal((n, n_layers, d)).astype(np.float32)
    conv = np.array([f"s{i}" for i in range(n)])
    out = cm.frozen_map_swap(Xs, Ys, Xt, Yt, conv, [0, 1], seed=0, null_draws=2)
    assert set(out["r2_by_layer"]) == {0, 1}
    assert all(np.isfinite(v) for v in out["r2_by_layer"].values())
    assert "inner-group-cv" in cm.SELECTOR_LOG


def test_cm_fit_primal_beta_inner_mode(_restore_globals):
    cm.LAMBDA_SELECTION = "inner-group-cv"
    cm.SELECTOR_LOG = {}
    X = RNG.standard_normal((18, 5)).astype(np.float64)
    Y = RNG.standard_normal((18, 5)).astype(np.float64)
    beta, lam = cm.fit_primal_beta(X, Y)
    assert beta.shape == (5, 5) and np.isfinite(float(lam))
    assert "inner-group-cv" in cm.SELECTOR_LOG


# ---------------------------------------------------------------------------
# Battery CLI e2e (production entrypoint, fresh subprocess): synthetic stores
# at real shapes (L=28), refit flags threaded, versioned outputs + judge-dir
# split, selector fields present, v1 out-dir untouched.
# ---------------------------------------------------------------------------
N_ROWS = 40
D_SMALL = 8
KEPT_DEFAULT = [f"s{i}" for i in range(30)]
KEPT_C3 = [f"s{i}" for i in range(12)]  # < 3*N_FOLDS -> battery pairs skip (gate)


def _xy_pair(n: int, n_layers: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_layers, d)).astype(np.float32)
    W = rng.standard_normal((d, d)).astype(np.float32) / np.sqrt(d)
    Y = (X @ W + 0.2 * rng.standard_normal((n, n_layers, d))).astype(np.float32)
    return X, Y


def _write_store(store: Path, stem: str, n_slots: int, n_turns: int, seed: int) -> None:
    layers = 28  # _cell_xy asserts EXPECTED_LAYERS
    rng = np.random.default_rng(seed)
    slots = rng.standard_normal((N_ROWS, n_slots, layers, D_SMALL)).astype(np.float32)
    W = rng.standard_normal((D_SMALL, D_SMALL)).astype(np.float32) / np.sqrt(D_SMALL)
    profiles = (slots[:, :1] @ W + 0.2 * rng.standard_normal((N_ROWS, 1, layers, D_SMALL))).astype(
        np.float32
    )
    if n_turns > 1:
        profiles = np.repeat(profiles, n_turns, axis=1)
    nll = rng.standard_normal((N_ROWS, n_turns)).astype(np.float32)
    conv_ids = [f"s{i}" for i in range(N_ROWS)]
    np.savez(store / f"{stem}.npz", slots=slots, profiles=profiles, nll=nll)
    (store / f"{stem}.json").write_text(json.dumps({"conv_ids": conv_ids}))
    # the battery's fingerprint sidecar (own stores only need one shard sidecar)
    (store / f"{stem}_shard000.json").write_text(
        json.dumps({"conv_ids": conv_ids, "render_config_hash": r1417.render_config_hash()})
    )


def _build_fixture(tmp: Path) -> tuple[Path, Path]:
    data_dir = tmp / "data"
    out_dir = tmp / "v1"
    (data_dir / "reference_sidecars").mkdir(parents=True)
    shared = [f"s{i}" for i in range(r1417.N_SHARED_EXPECTED)]
    for stem in r1417.REFERENCE_STEMS:
        (data_dir / "reference_sidecars" / f"{stem}_shard000.json").write_text(
            json.dumps({"conv_ids": shared})
        )
    (data_dir / "turnstore").mkdir()
    (data_dir / "store").mkdir()
    (data_dir / "gen").mkdir()
    (out_dir / "judge").mkdir(parents=True)
    for mi, model in enumerate(r1417.MODELS):
        for fmt in ("chat", "naturalistic"):
            _write_store(data_dir / "turnstore", f"{model}_{fmt}_s", 1, 2, seed=100 + mi)
        for ci, cell in enumerate(r1417.CELL_ORDER):
            _write_store(data_dir / "store", f"{model}_{cell}_s", 2, 1, seed=10 * mi + ci)
            kept = KEPT_C3 if (model, cell) == ("instruct", "c3_evasive") else KEPT_DEFAULT
            kept = kept if model == "instruct" else KEPT_DEFAULT[:20]
            (out_dir / "judge" / f"kept_{model}_{cell}.json").write_text(
                json.dumps(
                    {
                        "kept_conv_ids": kept,
                        "n_kept": len(kept),
                        "yield_frac": len(kept) / N_ROWS,
                        "render_config_hash": r1417.render_config_hash(),
                    }
                )
            )
            rows = [
                {
                    "conv_id": f"s{i}",
                    "question": f"q{i}",
                    "completion": f"a{i}",
                    "completion_token_ids": [i, i + 1, i + 2],
                    "render_config_hash": r1417.render_config_hash(),
                }
                for i in range(N_ROWS)
            ]
            with open(data_dir / "gen" / f"{model}_{cell}.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
    return data_dir, out_dir


@pytest.mark.slow
def test_battery_cli_refit_e2e(tmp_path):
    data_dir, out_dir = _build_fixture(tmp_path)
    refit_dir = out_dir / "refit"
    base = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "issue1417_battery.py"),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(refit_dir),
        "--judge-dir",
        str(out_dir),
        "--lambda-selection",
        "inner-group-cv",
        "--gcv-dof-cap",
        "0.9",
        "--null-draws",
        "2",
        "--n-boot",
        "20",
        "--cosine-null-draws",
        "2",
        "--collapse-null-draws",
        "2",
        "--pilot-budget-h",
        "1.0",
    ]
    for phase in (
        ["--fits", "--model", "instruct"],
        ["--battery", "--model", "instruct"],
        ["--summary"],
    ):
        proc = subprocess.run(
            base + phase, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200
        )
        assert proc.returncode == 0, f"{phase}: rc={proc.returncode}\n{proc.stderr[-3000:]}"

    # versioned outputs: refit populated, v1 untouched (judge inputs only)
    assert not (out_dir / "cells").exists() and not (out_dir / "battery").exists()
    cells = sorted(refit_dir.glob("cells/cells_*__instruct__*.json"))
    assert cells, "no refit cell outputs"
    fitted = [json.loads(p.read_text()) for p in cells]
    with_sel = [c for c in fitted if "selector_per_layer_fold" in c]
    assert with_sel, "no refit cell JSON carries the selector record"
    for c in with_sel:
        assert c["lambda_selection"] == "inner-group-cv"
        assert c["gcv_dof_cap"] == 0.9
        sels = {s for row in c["selector_per_layer_fold"] for s in row if s is not None}
        assert sels <= {"inner-group-cv", "gcv-fallback"} and sels
        lams = [v for row in c["selected_lambda_per_layer_fold"] for v in row if v is not None]
        assert lams and all(np.isfinite(v) for v in lams)

    # battery: run pairs carry refit_config + a non-empty shared selector log;
    # the instruct c3 pairs hit the designed too-few-rows skip (kept=12 < 15)
    pair_files = sorted(refit_dir.glob("battery/battery_instruct__*.json"))
    ran = [json.loads(p.read_text()) for p in pair_files]
    ran_full = [r for r in ran if "rel_by_layer" in r]
    skipped = [r for r in ran if "skipped_too_few_rows" in r]
    assert ran_full and skipped, (len(ran_full), len(skipped))
    for r in ran_full:
        assert r["refit_config"] == {"lambda_selection": "inner-group-cv", "gcv_dof_cap": 0.9}
        assert "inner-group-cv" in r["selector_log"]

    summary = json.loads((refit_dir / "battery_summary.json").read_text())
    assert summary["refit_config"]["lambda_selection"] == "inner-group-cv"
    assert summary["refit_config"]["gcv_dof_cap"] == 0.9


def test_run_fits_gates_empty_and_too_few(tmp_path, monkeypatch, _restore_globals):
    """Data-dependent fits gates fire their DESIGNED skip branches: an empty
    kept set writes skipped_empty_rows; a 4-row kept set writes
    skipped_too_few_rows (in-process, real run_fits body)."""
    import issue1417_battery as bat

    # Patch BEFORE building: render_config_hash() covers the cell registry, so
    # the fixture's fingerprints must be written under the SAME patched state
    # the in-process battery checks against.
    monkeypatch.setattr(r1417, "CELL_ORDER", ("c1_helpful_ctrl",))
    data_dir, out_dir = _build_fixture(tmp_path)
    refit_dir = out_dir / "refit"
    # overwrite the pretrained kept set with a 4-row set, instruct with empty
    for model, kept in (("pretrained", [f"s{i}" for i in range(4)]), ("instruct", [])):
        (out_dir / "judge" / f"kept_{model}_c1_helpful_ctrl.json").write_text(
            json.dumps(
                {
                    "kept_conv_ids": kept,
                    "n_kept": len(kept),
                    "yield_frac": len(kept) / N_ROWS,
                    "render_config_hash": r1417.render_config_hash(),
                }
            )
        )

    class _Args:
        pass

    args = _Args()
    args.data_dir = data_dir
    args.out_dir = refit_dir
    args.judge_dir = out_dir
    args.lambda_selection = "inner-group-cv"
    args.gcv_dof_cap = 0.9
    args.smoke = False
    args.resume = False
    args.null_draws = 0
    args.n_boot = 10
    for model in ("pretrained", "instruct"):
        args.model = model
        assert bat.run_fits(args) == 0
    too_few = json.loads(
        (refit_dir / "cells" / "cells_c1_helpful_ctrl__pretrained__ctx.json").read_text()
    )
    assert too_few.get("skipped_too_few_rows") == 4
    empty = json.loads(
        (refit_dir / "cells" / "cells_c1_helpful_ctrl__instruct__ctx.json").read_text()
    )
    assert empty.get("skipped_empty_rows") is True
