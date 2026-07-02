"""Vectorized cell fits for issue #825 (user-base map: predicting turn profiles).

Fits Gram-space closed-form ridge (GCV lambda selection) from turn-slot
activations to turn-profile activations, per layer, with K-fold held-out CV.

Per within-role / Track-S cell:
  - held-out pooled R^2 per layer (SS_tot from the test fold's own mean)
  - per-example cosine at FROZEN_LAYERS
  - shuffle-null draws (>= N_NULL_DRAWS) with the FULL per-draw x per-layer
    matrix persisted (selection-symmetric nulls, #778: observed layer-max is
    compared against each null draw's OWN layer-max; a frozen-layer table is
    also reported)
  - random-projection control (fixed-seed Gaussian, dimension-matched)
  - predict-the-mean baseline + skill-over-mean
  - bootstrap CIs (N_BOOTSTRAP resamples over the i.i.d. unit)

Vectorization contract (load-bearing): NO per-cell x per-draw x per-fold
refit explosion. The train Gram eigendecomposition depends only on X, so per
(fold, layer) we compute standardization + eigh(G) + Kev ONCE and push the
observed Y and every permuted-Y null draw through the cached (w, V, Kev)
path; only VtY and the GCV scan (cheap) recompute per draw.

Outputs under --out-dir (plan section 6.5):
  cells_<cell_id>.json, nulls_<cell_id>.json, perposition_<cell_id>.json,
  crossrole_<cell_id>.json, power_curve.json, nll_reads.json
Every JSON carries a metadata block (git_commit, timestamp, seed, n, script).

CLI:
  uv run python scripts/issue825_fit_cells.py \
      --turnstore-dir data/issue_825/turnstore --out-dir eval_results/issue_825 \
      [--cells all|<id>[,<id>...]] [--null-draws 20] [--folds 5] [--seed 0] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.experiments.issue_825 import common

FROZEN_LAYERS = common.FROZEN_LAYERS
N_FOLDS = common.N_FOLDS
N_NULL_DRAWS = common.N_NULL_DRAWS
N_BOOTSTRAP = common.N_BOOTSTRAP
FIT_SEED = common.FIT_SEED
EXPECTED_LAYERS = common.EXPECTED_LAYERS
WITHIN_ROLE_CELLS = common.WITHIN_ROLE_CELLS
TRACK_S_CELLS = common.TRACK_S_CELLS
CROSS_ROLE_CELLS = common.CROSS_ROLE_CELLS

LAMBDAS = np.logspace(-2, 4, 13)

# G1 gate anchors: #779 per-context reconstruction curve (layer -> R^2).
# Full curve loaded from eval_results/issue_779/percontext_recon.json when
# present; these embedded anchors are the fallback.
G1_ANCHORS = {19: 0.677, 14: 0.598, 18: 0.635, 26: 0.604}
G1_CURVE_PATH = Path("eval_results/issue_779/percontext_recon.json")

# ---------------------------------------------------------------------------
# Ridge core (adapted from issue-779 _ridge_fit_predict_fast; Gram-space,
# GCV lambda selection, eigendecomposition cached for null-draw reuse)
# ---------------------------------------------------------------------------


def _prep_fold(X_train: np.ndarray, X_eval: np.ndarray) -> dict:
    """Compute the Y-independent pieces of the Gram-space ridge for one fold.

    Returns a cache dict reused across the observed fit and every permuted-Y
    null draw (the eigh(G) is the expensive step and depends only on X).
    """
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64)
    Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    Kev = Xev_n @ Xtr_n.T
    KevV = Kev @ V
    return {"w": w, "V": V, "KevV": KevV, "ntr": int(Xtr.shape[0])}


def _ridge_predict_cached(cache: dict, Y_train: np.ndarray) -> np.ndarray:
    """Fit + predict for one Y using a fold cache from _prep_fold.

    Recomputes only VtY and the (cheap) GCV lambda scan; identical fitting
    procedure for observed and every null draw (selection-symmetric).
    """
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64)
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    w, V, KevV, ntr = cache["w"], cache["V"], cache["KevV"], cache["ntr"]
    VtY = V.T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    best_lam = float(LAMBDAS[0])
    best_gcv = float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
    filt = 1.0 / (w + best_lam)
    pred = (KevV * filt) @ VtY + ymu
    return pred.numpy()


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Pooled R^2 with SS_tot from the evaluated set's OWN mean."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _per_example_cosine(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    num = (pred * true).sum(1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-12
    return num / den


def _cv_folds(conv_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold ids from a seeded permutation of UNIQUE conversation ids.

    All examples sharing a conversation id land in the same fold (Track M
    invariant: fold-id constant within conversation-id — asserted below).
    """
    conv_ids = np.asarray(conv_ids)
    uniq = np.unique(conv_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    conv_fold = {cid: int(perm[i] % n_folds) for i, cid in enumerate(uniq)}
    folds = np.array([conv_fold[c] for c in conv_ids], dtype=np.int64)
    for cid in uniq:
        f = folds[conv_ids == cid]
        assert (f == f[0]).all(), f"fold id varies within conversation {cid!r}"
    return folds


# ---------------------------------------------------------------------------
# Held-out fit sweep: observed + null draws per layer, cached eigh per fold
# ---------------------------------------------------------------------------


def heldout_r2_sweep(
    X_layers: np.ndarray,
    Y_layers: np.ndarray,
    conv_ids: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    collect_cosines: bool = True,
) -> dict:
    """Held-out pooled R^2 per layer for observed Y and every shuffle-null draw.

    X_layers, Y_layers: (N, L, D) fp arrays (slot -> profile per layer).
    Returns:
      r2_obs: (L,) observed held-out pooled R^2 per layer
      r2_null: (null_draws, L) the FULL per-draw x per-layer matrix
      cosines: {layer: (N,) per-example cosine} at FROZEN_LAYERS
      preds_frozen: {layer: (N, D) held-out predictions} at FROZEN_LAYERS
    Null draws permute Y ROW-ORDER (whole example rows, seeded) and go through
    the IDENTICAL cached (w, V, Kev) fitting path as the observed fit.
    """
    X_layers = np.asarray(X_layers, dtype=np.float32)
    Y_layers = np.asarray(Y_layers, dtype=np.float32)
    n, n_layers = X_layers.shape[0], X_layers.shape[1]
    folds = _cv_folds(conv_ids, n_folds, seed)
    rng = np.random.default_rng(seed + 1)
    # Null draws permute the X<->Y pairing at the CONVERSATION level (the
    # i.i.d. unit): rows of the same conversation move together, so within-
    # conversation Y correlation cannot leak into the null (identical to a
    # row permutation when each conversation has exactly one row).
    ids = np.asarray(conv_ids)
    uniq_c, inv = np.unique(ids, return_inverse=True)
    row_of_conv = [np.flatnonzero(inv == k) for k in range(len(uniq_c))]
    def _conv_perm() -> np.ndarray:
        cp = rng.permutation(len(uniq_c))
        return np.concatenate([row_of_conv[k] for k in cp])
    null_perms = [_conv_perm() for _ in range(null_draws)]

    ss_res_obs = np.zeros(n_layers)
    ss_tot_obs = np.zeros(n_layers)
    ss_res_null = np.zeros((null_draws, n_layers))
    ss_tot_null = np.zeros((null_draws, n_layers))
    fitted = np.zeros(n, dtype=bool)
    cosines = {int(li): np.zeros(n) for li in FROZEN_LAYERS if li < n_layers}
    preds_frozen = {
        int(li): np.zeros((n, Y_layers.shape[2]), dtype=np.float32)
        for li in FROZEN_LAYERS
        if li < n_layers
    }

    for li in range(n_layers):
        X = X_layers[:, li, :]
        Y = Y_layers[:, li, :]
        for k in range(n_folds):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() < 3:
                continue
            cache = _prep_fold(X[tr], X[te])
            pred = _ridge_predict_cached(cache, Y[tr])
            fitted[te] = True
            true = Y[te].astype(np.float64)
            mu = true.mean(0)
            ss_res_obs[li] += float(np.sum((true - pred) ** 2))
            ss_tot_obs[li] += float(np.sum((true - mu) ** 2))
            if li in cosines and collect_cosines:
                cosines[li][te] = _per_example_cosine(pred, true)
                preds_frozen[li][te] = pred.astype(np.float32)
            for d, perm in enumerate(null_perms):
                Yp = Y[perm]
                pred_n = _ridge_predict_cached(cache, Yp[tr])
                true_n = Yp[te].astype(np.float64)
                mu_n = true_n.mean(0)
                ss_res_null[d, li] += float(np.sum((true_n - pred_n) ** 2))
                ss_tot_null[d, li] += float(np.sum((true_n - mu_n) ** 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_obs = 1.0 - ss_res_obs / np.where(ss_tot_obs < 1e-12, np.nan, ss_tot_obs)
        r2_null = 1.0 - ss_res_null / np.where(ss_tot_null < 1e-12, np.nan, ss_tot_null)
    return {
        "r2_obs": r2_obs,
        "r2_null": r2_null,
        "cosines": cosines,
        "preds_frozen": preds_frozen,
        "folds": folds,
        # Rows that actually received held-out predictions (skipped folds at
        # tiny n leave zeros in preds_frozen; consumers subset by this mask).
        "fitted_mask": fitted,
    }


def selection_symmetric_summary(r2_obs: np.ndarray, r2_null: np.ndarray) -> dict:
    """Selection-symmetric layer-max read (#778) + frozen-layer table.

    The observed layer-max R^2 is compared against each null draw's OWN
    layer-max (per-draw same-selection); the full per-draw x per-layer matrix
    is persisted by the caller alongside this summary.
    """
    obs_max = float(np.nanmax(r2_obs))
    obs_argmax = int(np.nanargmax(r2_obs))
    null_max = np.nanmax(r2_null, axis=1)  # (draws,) each draw's own layer-max
    frozen = {
        str(int(li)): {
            "r2_obs": float(r2_obs[li]),
            "null_mean": float(np.nanmean(r2_null[:, li])),
            "null_p975": float(np.nanquantile(r2_null[:, li], 0.975)),
        }
        for li in FROZEN_LAYERS
        if li < len(r2_obs)
    }
    return {
        "obs_layer_max_r2": obs_max,
        "obs_argmax_layer": obs_argmax,
        "null_layer_max_r2_per_draw": [float(v) for v in null_max],
        "null_layer_max_p975": float(np.nanquantile(null_max, 0.975)),
        "frozen_layer_table": frozen,
    }


# ---------------------------------------------------------------------------
# Controls, baselines, bootstrap
# ---------------------------------------------------------------------------


def random_projection_control(
    X_layers: np.ndarray,
    Y_layers: np.ndarray,
    conv_ids: np.ndarray,
    *,
    layers: list[int],
    n_folds: int,
    seed: int,
) -> dict:
    """Dimension-matched fixed-seed Gaussian random-projection control.

    X is replaced by X @ P (P Gaussian, same output dimension) so the control
    predictor carries the same dimensionality but scrambled feature axes.
    """
    out = {}
    rng = np.random.default_rng(seed + 7)
    for li in layers:
        if li >= X_layers.shape[1]:
            continue
        X = X_layers[:, li, :].astype(np.float64)
        Y = Y_layers[:, li, :]
        P = rng.standard_normal((X.shape[1], X.shape[1])) / np.sqrt(X.shape[1])
        Xp = (X @ P).astype(np.float32)
        folds = _cv_folds(conv_ids, n_folds, seed)
        ss_res, ss_tot = 0.0, 0.0
        for k in range(n_folds):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() < 3:
                continue
            cache = _prep_fold(Xp[tr], Xp[te])
            pred = _ridge_predict_cached(cache, Y[tr])
            true = Y[te].astype(np.float64)
            mu = true.mean(0)
            ss_res += float(np.sum((true - pred) ** 2))
            ss_tot += float(np.sum((true - mu) ** 2))
        out[str(int(li))] = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return out


def mean_baseline_r2(Y_layers: np.ndarray, conv_ids: np.ndarray, *, layers, n_folds, seed) -> dict:
    """Predict-the-train-mean baseline, held out (same folds)."""
    out = {}
    for li in layers:
        if li >= Y_layers.shape[1]:
            continue
        Y = Y_layers[:, li, :].astype(np.float64)
        folds = _cv_folds(conv_ids, n_folds, seed)
        ss_res, ss_tot = 0.0, 0.0
        for k in range(n_folds):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            pred = np.broadcast_to(Y[tr].mean(0), Y[te].shape)
            mu = Y[te].mean(0)
            ss_res += float(np.sum((Y[te] - pred) ** 2))
            ss_tot += float(np.sum((Y[te] - mu) ** 2))
        out[str(int(li))] = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return out


def bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap CI of the mean over the i.i.d. unit (rows)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "n": len(values),
    }


def bootstrap_r2_ci(pred: np.ndarray, true: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap CI of pooled R^2, resampling examples."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(_pooled_r2(pred[idx], true[idx]))
    vals = np.asarray(vals)
    return {
        "r2": _pooled_r2(pred, true),
        "ci_lo": float(np.nanquantile(vals, 0.025)),
        "ci_hi": float(np.nanquantile(vals, 0.975)),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Turnstore loading + cell tensor assembly
# ---------------------------------------------------------------------------


def _load_bundle(turnstore_dir: Path, model_key: str, format_key: str) -> dict:
    """Load one (model, format) turnstore bundle written by extract_turnstore.

    Expects <turnstore_dir>/<model_key>_<format_key>.npz (fp16 arrays:
    slots (N, n_slots, L, D), profiles (N, n_turns, L, D), perpos
    (N, n_turns, P, n_peak, D), nll (N, n_turns)) plus a JSON sidecar
    <model_key>_<format_key>.json with conv_ids and index maps.
    """
    stem = f"{model_key}_{format_key}"
    npz_path = turnstore_dir / f"{stem}.npz"
    side_path = turnstore_dir / f"{stem}.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"turnstore bundle missing: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    sidecar = json.loads(side_path.read_text()) if side_path.exists() else {}
    return {"arrays": data, "sidecar": sidecar}


def _cell_xy(bundle: dict, cell: dict) -> dict:
    """Assemble (X, Y, conv_ids, nll) for one cell from a loaded bundle.

    X = slot activations at the cell's slot index (N, L, D);
    Y = turn-profile activations at the cell's target turn index (N, L, D).
    Cells specify indices via 'slot_index' / 'target_turn_index' (defaults 0).
    """
    arrays = bundle["arrays"]
    slots = np.asarray(arrays["slots"], dtype=np.float32)
    profiles = np.asarray(arrays["profiles"], dtype=np.float32)
    nll = np.asarray(arrays["nll"], dtype=np.float32) if "nll" in arrays else None
    conv_ids = np.asarray(bundle["sidecar"].get("conv_ids", np.arange(slots.shape[0])))
    si = int(cell.get("slot_index", 0))
    ti = int(cell.get("target_turn_index", 0))
    # HARD index asserts — silently clamping onto a mis-shaped bundle would
    # relabel the wrong slot/turn as this cell (code-review round-1 fix).
    assert si < slots.shape[1], f"slot_index {si} >= n_slots {slots.shape[1]}"
    assert ti < profiles.shape[1], f"target_turn_index {ti} >= n_turns {profiles.shape[1]}"
    X = slots[:, si, :, :]
    Y = profiles[:, ti, :, :]
    assert X.shape[1] == EXPECTED_LAYERS, f"layer axis {X.shape[1]} != {EXPECTED_LAYERS}"
    keep = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(Y).any(axis=(1, 2)))
    nll_t = None
    if nll is not None:
        assert ti < nll.shape[1], f"target_turn_index {ti} >= nll turns {nll.shape[1]}"
        nll_t = nll[:, ti][keep]
    return {"X": X[keep], "Y": Y[keep], "conv_ids": conv_ids[keep], "nll": nll_t}


# ---------------------------------------------------------------------------
# Metadata + JSON I/O
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _metadata(seed: int, n: int) -> dict:
    return {
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": "scripts/issue825_fit_cells.py",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[fit_cells] wrote {path}")


# ---------------------------------------------------------------------------
# Per-cell runner (within-role + Track-S)
# ---------------------------------------------------------------------------


def run_cell(
    cell: dict,
    turnstore_dir: Path,
    out_dir: Path,
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
) -> dict:
    cell = _normalize_cell(cell)
    cell_id = cell["cell_id"]
    bundle = _load_bundle(turnstore_dir, cell["model_key"], cell["format_key"], cell["track"])
    xy = _cell_xy(bundle, cell)
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    print(f"[fit_cells] cell={cell_id} n={len(conv_ids)}")

    sweep = heldout_r2_sweep(X, Y, conv_ids, n_folds=n_folds, seed=seed, null_draws=null_draws)
    r2_obs, r2_null = sweep["r2_obs"], sweep["r2_null"]
    summary = selection_symmetric_summary(r2_obs, r2_null)

    frozen_layers = [li for li in FROZEN_LAYERS if li < X.shape[1]]
    rp = random_projection_control(X, Y, conv_ids, layers=frozen_layers, n_folds=n_folds, seed=seed)
    mb = mean_baseline_r2(Y, conv_ids, layers=frozen_layers, n_folds=n_folds, seed=seed)

    cosine_stats = {}
    r2_cis = {}
    fitted = sweep["fitted_mask"]
    for li in frozen_layers:
        cos = sweep["cosines"][li][fitted]
        cosine_stats[str(li)] = bootstrap_ci(cos, n_boot=n_boot, seed=seed + li)
        pred = sweep["preds_frozen"][li][fitted]
        r2_cis[str(li)] = bootstrap_r2_ci(
            pred, Y[fitted, li, :], n_boot=n_boot, seed=seed + 100 + li
        )

    skill_over_mean = {
        str(li): float(r2_obs[li]) - float(mb.get(str(li), float("nan"))) for li in frozen_layers
    }

    cell_payload = {
        "metadata": _metadata(seed, len(conv_ids)),
        "cell": {k: v for k, v in cell.items() if isinstance(k, str)},
        "r2_per_layer_obs": [float(v) for v in r2_obs],
        "selection_symmetric": summary,
        "random_projection_control_r2": rp,
        "mean_baseline_r2": mb,
        "skill_over_mean": skill_over_mean,
        "cosine_frozen_layers": cosine_stats,
        "r2_bootstrap_ci_frozen_layers": r2_cis,
        "n_folds": n_folds,
        "null_draws": null_draws,
    }
    _write_json(out_dir / f"cells_{cell_id}.json", cell_payload)

    null_payload = {
        "metadata": _metadata(seed, len(conv_ids)),
        "cell_id": cell_id,
        # FULL per-draw x per-layer matrix: observed row first, then one row
        # per null draw, one column per layer (selection-symmetric rule).
        "layers": list(range(len(r2_obs))),
        "observed_row": [float(v) for v in r2_obs],
        "null_matrix": [[float(v) for v in row] for row in r2_null],
        "null_layer_max_per_draw": summary["null_layer_max_r2_per_draw"],
    }
    _write_json(out_dir / f"nulls_{cell_id}.json", null_payload)
    return {"sweep": sweep, "xy": xy, "summary": summary, "cell_payload": cell_payload}


# ---------------------------------------------------------------------------
# S1 extras: R^2(n) power curve + G1 gate (Spearman vs the #779 curve)
# ---------------------------------------------------------------------------

POWER_CURVE_NS = (250, 500, 1000, 2000, 5000)


def run_power_curve(xy: dict, out_dir: Path, *, n_folds: int, seed: int) -> None:
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    rng = np.random.default_rng(seed + 13)
    order = rng.permutation(len(conv_ids))  # nested subsets: prefixes of one perm
    curve = []
    for n_sub in POWER_CURVE_NS:
        if n_sub > len(order):
            curve.append({"n": n_sub, "r2_per_layer": None, "note": "n exceeds data"})
            continue
        idx = order[:n_sub]
        sw = heldout_r2_sweep(
            X[idx],
            Y[idx],
            conv_ids[idx],
            n_folds=n_folds,
            seed=seed,
            null_draws=0,
            collect_cosines=False,
        )
        curve.append(
            {
                "n": n_sub,
                "r2_per_layer": [float(v) for v in sw["r2_obs"]],
                "r2_layer_max": float(np.nanmax(sw["r2_obs"])),
            }
        )
    _write_json(
        out_dir / "power_curve.json",
        {
            "metadata": _metadata(seed, len(conv_ids)),
            "subsample_ns": list(POWER_CURVE_NS),
            "nested": True,
            "curve": curve,
        },
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    den = np.sqrt((ar**2).sum() * (br**2).sum())
    return float("nan") if den < 1e-12 else float((ar * br).sum() / den)


def g1_gate(r2_obs: np.ndarray, out_dir: Path, *, seed: int, n: int) -> dict:
    """G1: Spearman of our per-layer R^2 curve vs the #779 reference curve."""
    ref: dict[int, float] = dict(G1_ANCHORS)
    source = "embedded_anchors"
    if G1_CURVE_PATH.exists():
        try:
            raw = json.loads(G1_CURVE_PATH.read_text())
            cand = raw.get("r2_per_layer") or raw.get("per_layer_r2") or raw
            if isinstance(cand, dict):
                ref = {int(k): float(v) for k, v in cand.items() if str(k).isdigit()}
                source = str(G1_CURVE_PATH)
            elif isinstance(cand, list):
                ref = {i: float(v) for i, v in enumerate(cand)}
                source = str(G1_CURVE_PATH)
        except Exception as exc:  # fail toward the embedded anchors, loudly
            print(f"[fit_cells] WARN G1 curve load failed ({exc}); using anchors")
    layers = sorted(li for li in ref if li < len(r2_obs))
    ours = np.array([r2_obs[li] for li in layers])
    theirs = np.array([ref[li] for li in layers])
    rho = _spearman(ours, theirs) if len(layers) >= 3 else float("nan")
    l19_abs = abs(float(r2_obs[19]) - G1_ANCHORS[19]) if len(r2_obs) > 19 else float("nan")
    payload = {
        "metadata": _metadata(seed, n),
        "reference_source": source,
        "layers_compared": layers,
        "spearman_vs_779": rho,
        "abs_dev_L19_vs_0677": l19_abs,
        "pass": bool(rho >= 0.9 and l19_abs <= 0.05) if not np.isnan(rho) else False,
    }
    _write_json(out_dir / "g1_gate.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Cell normalization (common.py registry -> loader/index keys)
# ---------------------------------------------------------------------------

# Turn order in Track-M bundles: [u1, a1, u2, a2]; slots: [assistant(before a1),
# user(before u2)]. Track S: [u1, a1], slot [assistant].
_ROLE_TO_INDICES = {
    "assistant": {"slot_index": 0, "target_turn_index": 1},
    "user": {"slot_index": 1, "target_turn_index": 2},
}
_CROSS_DIRECTIONS = {
    # direction -> (X slot, Y turn, baseline X turn, baseline Y turn)
    "assistant_to_user": {"slot_index": 0, "target_turn_index": 2, "base_x_turn": 0},
    "user_to_assistant": {"slot_index": 1, "target_turn_index": 3, "base_x_turn": 1},
}


def _normalize_cell(cell: dict) -> dict:
    """Map a common.py cell dict onto the loader/index keys run_cell expects.

    Accepts either the already-normalized shape (model_key/format_key present)
    or the registry shape (model/role/format or model/direction). Fails loud on
    an unknown role/direction.
    """
    out = dict(cell)
    if "model_key" not in out:
        out["model_key"] = out["model"]
    if "format_key" not in out:
        out["format_key"] = out.get("format", "chat")
    if "slot_index" not in out:
        if "role" in out:
            idx = _ROLE_TO_INDICES[out["role"]]
            out.update(idx)
        elif "direction" in out:
            idx = _CROSS_DIRECTIONS[out["direction"]]
            out.update(idx)
        else:  # Track S: single-pair bundle, assistant slot -> a1 profile
            out.update({"slot_index": 0, "target_turn_index": 1})
    out.setdefault("track", "s" if str(out.get("cell_id", "")).startswith("S") else "m")
    return out


# ---------------------------------------------------------------------------
# Dual-format bundle loading (.npz contract OR the extractor's .pt shards)
# ---------------------------------------------------------------------------


def _stack_maybe_list(val, name: str) -> np.ndarray:
    """Stack a per-record list of tensors/arrays to one array; pass arrays through."""
    if isinstance(val, np.ndarray):
        return val
    rows = []
    for i, r in enumerate(val):
        # torch bf16 has no direct .numpy(); upcast through float32 first.
        arr = r.float().numpy() if torch.is_tensor(r) else np.asarray(r)
        if rows and arr.shape != rows[0].shape:
            raise ValueError(f"{name}[{i}] shape {arr.shape} != {rows[0].shape} — ragged shard")
        rows.append(arr)
    return np.stack(rows)


def _load_bundle_pt(
    turnstore_dir: Path, model_key: str, format_key: str, track: str
) -> dict | None:
    """Load the extractor's sharded .pt bundles ({model}_{format}_{track}*.pt).

    TRACK-AWARE (code-review round-1 blocker: a track-blind glob mixed Track-S
    and Track-M shards written to the same dir). Returns the same
    {"arrays", "sidecar"} contract as the .npz loader, or None when no
    matching shards exist. Per-conv list payloads are stacked (shape mismatch
    fails loud naming the shard).
    """
    shards = sorted(turnstore_dir.glob(f"{model_key}_{format_key}_{track}*.pt"))
    if not shards:
        return None
    keys = ("slots", "profiles", "perpos", "perpos_mask", "nll")
    acc: dict[str, list] = {k: [] for k in keys}
    conv_ids: list = []
    for sp in shards:
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        conv_ids.extend(payload.get("conv_ids", []))
        for k in keys:
            if k in payload and payload[k] is not None:
                v = payload[k]
                acc[k].extend(list(v) if isinstance(v, (list, tuple)) else [t for t in v])
    arrays: dict[str, np.ndarray] = {}
    for k, rows in acc.items():
        if rows:
            arrays[k] = _stack_maybe_list(rows, k).astype(np.float32)
    if "slots" not in arrays or "profiles" not in arrays:
        raise KeyError(
            f"pt shards for {model_key}_{format_key} missing slots/profiles "
            f"(keys: {sorted(arrays)}) — extractor/driver contract drift"
        )
    return {"arrays": arrays, "sidecar": {"conv_ids": conv_ids, "source": "pt-shards"}}


def _load_bundle_any(turnstore_dir: Path, model_key: str, format_key: str, track: str) -> dict:
    """Track-aware bundle load: .npz contract first, then .pt shards."""
    stem = f"{model_key}_{format_key}_{track}"
    npz_path = turnstore_dir / f"{stem}.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        side_path = turnstore_dir / f"{stem}.json"
        sidecar = json.loads(side_path.read_text()) if side_path.exists() else {}
        return {"arrays": data, "sidecar": sidecar}
    pt = _load_bundle_pt(turnstore_dir, model_key, format_key, track)
    if pt is None:
        raise FileNotFoundError(
            f"no turnstore bundle for {stem} (.npz or .pt shards) in {turnstore_dir}"
        )
    return pt


# run_cell resolves bundles through the track-aware dual-format loader; the
# legacy 3-arg npz loader is superseded (rewire-then-delete happens at the
# _load_bundle definition site above — kept as the npz reference impl only).
_load_bundle = _load_bundle_any


# ---------------------------------------------------------------------------
# Cross-role cells: cross-R^2 + topic-persistence baseline + PAIRED bootstrap
# ---------------------------------------------------------------------------


def run_cross_role_cell(
    cell: dict,
    turnstore_dir: Path,
    out_dir: Path,
    *,
    n_folds: int,
    seed: int,
    null_draws: int,
    n_boot: int,
) -> dict:
    """Cross-role prediction vs the topic-persistence baseline (plan §4.1/§5.6).

    X = c_x at the cell's slot; Y = the OTHER role's later turn profile.
    Baseline: previous same-role profile -> Y through the SAME ridge recipe.
    The claim statistic is the PAIRED bootstrap delta (cross - baseline) at the
    frozen layers, recomputed inside each conversation-level resample.
    """
    cell = _normalize_cell(cell)
    cell_id = cell["cell_id"]
    bundle = _load_bundle_any(turnstore_dir, cell["model_key"], cell["format_key"], cell["track"])
    arrays = bundle["arrays"]
    slots = np.asarray(arrays["slots"], dtype=np.float32)
    profiles = np.asarray(arrays["profiles"], dtype=np.float32)
    conv_ids_all = np.asarray(bundle["sidecar"].get("conv_ids", np.arange(slots.shape[0])))
    si = int(cell["slot_index"])
    ti = int(cell["target_turn_index"])
    bx_turn = int(cell["base_x_turn"])
    assert si < slots.shape[1] and ti < profiles.shape[1] and bx_turn < profiles.shape[1], (
        si, ti, bx_turn, slots.shape, profiles.shape,
    )
    X_full = slots[:, si, :, :]
    Y_full = profiles[:, ti, :, :]
    Xb_full = profiles[:, bx_turn, :, :]
    # ONE joint keep mask over every tensor entering the paired comparison —
    # independent masks + truncation silently misalign rows across
    # conversations, corrupting the paired delta (code-review round-1 blocker).
    keep = ~(
        np.isnan(X_full).any(axis=(1, 2))
        | np.isnan(Y_full).any(axis=(1, 2))
        | np.isnan(Xb_full).any(axis=(1, 2))
    )
    X, Y, Xb, conv_ids = X_full[keep], Y_full[keep], Xb_full[keep], conv_ids_all[keep]
    n = len(conv_ids)
    print(f"[fit_cells] cross-role cell={cell_id} n={n}")

    cross = heldout_r2_sweep(
        X, Y, conv_ids, n_folds=n_folds, seed=seed, null_draws=null_draws, collect_cosines=True
    )
    base = heldout_r2_sweep(
        Xb, Y, conv_ids, n_folds=n_folds, seed=seed, null_draws=0, collect_cosines=True
    )

    frozen_layers = [li for li in FROZEN_LAYERS if li < X.shape[1]]
    rng = np.random.default_rng(seed + 7)
    uniq = np.unique(conv_ids)
    delta_dist: dict[str, list[float]] = {str(li): [] for li in frozen_layers}
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(conv_ids == c) for c in sample])
        for li in frozen_layers:
            pc = cross["preds_frozen"][li][idx]
            pb = base["preds_frozen"][li][idx]
            t = Y[idx, li, :]
            delta_dist[str(li)].append(_pooled_r2(pc, t) - _pooled_r2(pb, t))
    delta_summary = {
        li: {
            "delta_r2_mean": float(np.nanmean(v)),
            "ci_lo": float(np.nanquantile(v, 0.025)),
            "ci_hi": float(np.nanquantile(v, 0.975)),
        }
        for li, v in delta_dist.items()
    }
    payload = {
        "metadata": _metadata(seed, n),
        "cell": {k: v for k, v in cell.items() if isinstance(k, str)},
        "cross_r2_per_layer": [float(v) for v in cross["r2_obs"]],
        "baseline_r2_per_layer": [float(v) for v in base["r2_obs"]],
        "selection_symmetric_cross": selection_symmetric_summary(cross["r2_obs"], cross["r2_null"]),
        "paired_delta_r2_frozen": delta_summary,
        # FULL paired-bootstrap distribution (never just marginal CIs).
        "delta_r2_distribution": delta_dist,
        "n_folds": n_folds,
        "n_boot": n_boot,
    }
    _write_json(out_dir / f"crossrole_{cell_id}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# NLL-stratified reads (alternatives-lens Must-Fix; plan §6 familiarity read)
# ---------------------------------------------------------------------------


def run_nll_reads(
    cell_results: dict[str, dict],
    out_dir: Path,
    *,
    seed: int,
    n_quantiles: int = 4,
) -> None:
    """R^2 by NLL quantile bin within model + cross-model gap on NLL-matched rows.

    Operates on HELD-OUT predictions already computed per cell (preds_frozen),
    so no refitting occurs; every R^2 here is a subset read of held-out preds.
    Pairs of cells sharing (role, format, track) with model in
    {instruct, pretrained} are matched by NLL quantile rank.
    """
    per_model: dict = {}
    for cid, res in cell_results.items():
        xy = res["xy"]
        if xy.get("nll") is None:
            continue
        nll = np.asarray(xy["nll"], dtype=np.float64)
        Y = xy["Y"]
        binned = {}
        qs = np.nanquantile(nll, np.linspace(0, 1, n_quantiles + 1))
        for b in range(n_quantiles):
            lo, hi = qs[b], qs[b + 1]
            m = (nll >= lo) & (nll <= hi if b == n_quantiles - 1 else nll < hi)
            layer_r2 = {}
            for li, pred in res["sweep"]["preds_frozen"].items():
                if m.sum() >= 8:
                    layer_r2[str(li)] = _pooled_r2(pred[m], Y[m, li, :])
            binned[f"q{b + 1}"] = {
                "n": int(m.sum()),
                "nll_range": [float(lo), float(hi)],
                "r2_frozen": layer_r2,
            }
        per_model[cid] = {"nll_mean": float(np.nanmean(nll)), "bins": binned}

    matched_gaps = {}
    ids = list(cell_results)
    for cid in ids:
        if "instruct" not in cid or "pretrained" in cid:
            continue
        pid = cid.replace("instruct", "pretrained")
        if pid not in cell_results:
            continue
        a, b = cell_results[cid], cell_results[pid]
        if a["xy"].get("nll") is None or b["xy"].get("nll") is None:
            continue
        na = np.asarray(a["xy"]["nll"], dtype=np.float64)
        nb = np.asarray(b["xy"]["nll"], dtype=np.float64)
        fa = np.asarray(a["sweep"].get("fitted_mask", np.ones(len(na), dtype=bool)))
        fb = np.asarray(b["sweep"].get("fitted_mask", np.ones(len(nb), dtype=bool)))
        # GENUINE matching (code-review round-1 blocker: pooled R^2 is
        # permutation-invariant, so rank-REORDERING full sets subsets
        # nothing). Bin edges from the POOLED NLL distribution; per bin,
        # SUBSET each model's rows to that bin and compare held-out R^2 on
        # the subsets; plus a support-overlap read (coarse caliper).
        pooled = np.concatenate([na[fa], nb[fb]])
        edges = np.nanquantile(pooled, np.linspace(0, 1, n_quantiles + 1))
        per_bin = {}
        for q in range(n_quantiles):
            lo, hi = edges[q], edges[q + 1]
            ma = fa & (na >= lo) & (na <= hi if q == n_quantiles - 1 else na < hi)
            mb = fb & (nb >= lo) & (nb <= hi if q == n_quantiles - 1 else nb < hi)
            bin_gap = {}
            for li, pred_a in a["sweep"]["preds_frozen"].items():
                pred_b = b["sweep"]["preds_frozen"].get(li)
                if pred_b is None or ma.sum() < 8 or mb.sum() < 8:
                    continue
                r2a = _pooled_r2(pred_a[ma], a["xy"]["Y"][ma, li, :])
                r2b = _pooled_r2(pred_b[mb], b["xy"]["Y"][mb, li, :])
                bin_gap[str(li)] = {
                    "instruct_r2": r2a,
                    "pretrained_r2": r2b,
                    "gap": r2a - r2b,
                    "n_instruct": int(ma.sum()),
                    "n_pretrained": int(mb.sum()),
                }
            per_bin[f"q{q + 1}"] = {"nll_range": [float(lo), float(hi)], "gaps": bin_gap}
        lo_s = max(float(na[fa].min()), float(nb[fb].min()))
        hi_s = min(float(na[fa].max()), float(nb[fb].max()))
        oa = fa & (na >= lo_s) & (na <= hi_s)
        ob = fb & (nb >= lo_s) & (nb <= hi_s)
        overlap = {}
        for li, pred_a in a["sweep"]["preds_frozen"].items():
            pred_b = b["sweep"]["preds_frozen"].get(li)
            if pred_b is None or oa.sum() < 8 or ob.sum() < 8:
                continue
            r2a = _pooled_r2(pred_a[oa], a["xy"]["Y"][oa, li, :])
            r2b = _pooled_r2(pred_b[ob], b["xy"]["Y"][ob, li, :])
            overlap[str(li)] = {
                "instruct_r2": r2a, "pretrained_r2": r2b, "gap": r2a - r2b,
                "n_instruct": int(oa.sum()), "n_pretrained": int(ob.sum()),
            }
        matched_gaps[f"{cid}__vs__{pid}"] = {
            "per_pooled_nll_bin": per_bin,
            "support_overlap": overlap,
            "nll_support": [lo_s, hi_s],
        }

    _write_json(
        out_dir / "nll_reads.json",
        {
            "metadata": _metadata(seed, len(cell_results)),
            "within_model_quartiles": per_model,
            "nll_matched_cross_model_gap": matched_gaps,
            "note": "all reads are subsets of held-out predictions; no refit",
        },
    )


# ---------------------------------------------------------------------------
# Per-position decay (peak layers, chat cells)
# ---------------------------------------------------------------------------


def run_per_position(
    cell: dict,
    turnstore_dir: Path,
    out_dir: Path,
    *,
    n_folds: int,
    seed: int,
    coverage_floor: float = 0.8,
) -> None:
    """Held-out ridge c_x -> per-position activation at the peak layers.

    perpos: (N, n_turns, P, n_peak, D) with perpos_mask (N, n_turns, P).
    Positions with < coverage_floor coverage are skipped. The eigh cache per
    (layer, fold) is reused across ALL positions (Y varies, X fixed).
    """
    cell = _normalize_cell(cell)
    cell_id = cell["cell_id"]
    bundle = _load_bundle_any(turnstore_dir, cell["model_key"], cell["format_key"], cell["track"])
    arrays = bundle["arrays"]
    if "perpos" not in arrays:
        print(f"[fit_cells] perpos missing for {cell_id}; skipping per-position read")
        return
    xy = _cell_xy(bundle, cell)
    X, conv_ids = xy["X"], xy["conv_ids"]
    ti = int(cell.get("target_turn_index", 1))
    perpos = np.asarray(arrays["perpos"], dtype=np.float32)[:, ti]  # (N, P, n_peak, D)
    mask = (
        np.asarray(arrays["perpos_mask"], dtype=bool)[:, ti]
        if "perpos_mask" in arrays
        else ~np.isnan(perpos).any(axis=(2, 3))
    )
    n, n_pos = perpos.shape[0], perpos.shape[1]
    n_peak = perpos.shape[2]
    peak_layers = [li for li in FROZEN_LAYERS if li < X.shape[1]][:n_peak]
    # Decay GRID (not all positions): rows invalid at position p must be
    # EXCLUDED from train AND eval (code-review round-1: masked rows entered
    # the fits), which forces fresh per-(layer, position) fold caches — the
    # grid bounds that cost while keeping the decay curve readable.
    grid = [p for p in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, n_pos - 1) if p < n_pos]
    results: dict[str, dict] = {}
    for k_idx, li in enumerate(peak_layers):
        Xl = X[:, li, :]
        pos_r2 = {}
        for p in sorted(set(grid)):
            rowmask = mask[:, p]
            cov = float(rowmask.mean())
            if cov < coverage_floor:
                continue
            Xp = Xl[rowmask]
            Yp = perpos[rowmask, p, k_idx, :].astype(np.float64)
            folds_p = _cv_folds(conv_ids[rowmask], n_folds, seed)
            ss_res = ss_tot = 0.0
            for k in range(n_folds):
                te = folds_p == k
                tr = ~te
                if te.sum() == 0 or tr.sum() < 3:
                    continue
                cache = _prep_fold(Xp[tr], Xp[te])
                pred = _ridge_predict_cached(cache, Yp[tr])
                true = Yp[te]
                mu = true.mean(0)
                ss_res += float(np.sum((true - pred) ** 2))
                ss_tot += float(np.sum((true - mu) ** 2))
            pos_r2[str(p)] = {
                "r2": (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
                "coverage": cov,
                "n_rows": int(rowmask.sum()),
            }
        results[str(li)] = pos_r2
    _write_json(
        out_dir / f"perposition_{cell_id}.json",
        {
            "metadata": _metadata(seed, n),
            "cell_id": cell_id,
            "peak_layers": [int(v) for v in peak_layers],
            "coverage_floor": coverage_floor,
            "r2_by_layer_position": results,
        },
    )


# ---------------------------------------------------------------------------
# MLP secondary (PCA-64 head; headline cells only; own shuffle null)
# ---------------------------------------------------------------------------


def run_mlp_secondary(
    res: dict,
    out_dir: Path,
    *,
    cell_id: str,
    n_folds: int,
    seed: int,
    n_null: int = 5,
) -> None:
    """fit_h.mlp_fit_predict (PCA-64) at the frozen layers with a shuffle null."""
    from explore_persona_space.experiments.issue_779.fit_h import mlp_fit_predict

    xy = res["xy"]
    X, Y, conv_ids = xy["X"], xy["Y"], xy["conv_ids"]
    folds = _cv_folds(conv_ids, n_folds, seed)
    rng = np.random.default_rng(seed + 13)
    out: dict[str, dict] = {}
    for li in [v for v in FROZEN_LAYERS if v < X.shape[1]]:
        Xl, Yl = X[:, li, :], Y[:, li, :]

        def _cv_r2(Yv: np.ndarray, Xl: np.ndarray = Xl) -> float:
            ss_res = ss_tot = 0.0
            for k in range(n_folds):
                te = folds == k
                tr = ~te
                if te.sum() == 0 or tr.sum() < 3:
                    continue
                pred = mlp_fit_predict(Xl[tr], Yv[tr], Xl[te])
                true = Yv[te].astype(np.float64)
                mu = true.mean(0)
                ss_res += float(np.sum((true - pred) ** 2))
                ss_tot += float(np.sum((true - mu) ** 2))
            return (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        obs = _cv_r2(Yl)
        nulls = [_cv_r2(Yl[rng.permutation(len(Yl))]) for _ in range(n_null)]
        out[str(li)] = {"r2_obs": obs, "r2_null": nulls}
    # Fold into the existing cells JSON under "mlp".
    cells_path = out_dir / f"cells_{cell_id}.json"
    payload = json.loads(cells_path.read_text()) if cells_path.exists() else {}
    payload["mlp"] = out
    _write_json(cells_path, payload)


# ---------------------------------------------------------------------------
# Smoke fabrication + main
# ---------------------------------------------------------------------------


def _fabricate_smoke_turnstore(turnstore_dir: Path, *, n: int = 24, dim: int = 16) -> None:
    """Write tiny synthetic .npz bundles satisfying the loader contract."""
    rng = np.random.default_rng(0)
    turnstore_dir.mkdir(parents=True, exist_ok=True)
    for model_key in ("instruct", "pretrained"):
        for format_key in ("chat", "naturalistic"):
            slots = rng.normal(size=(n, 2, EXPECTED_LAYERS, dim)).astype(np.float16)
            # Make profiles partially predictable from slots (real signal).
            profiles = (
                np.repeat(slots[:, :1], 4, axis=1) * 0.6
                + rng.normal(size=(n, 4, EXPECTED_LAYERS, dim)).astype(np.float16) * 0.4
            )
            perpos = rng.normal(size=(n, 4, 8, len(FROZEN_LAYERS), dim)).astype(np.float16)
            nll = rng.uniform(1.0, 4.0, size=(n, 4)).astype(np.float32)
            np.savez(
                turnstore_dir / f"{model_key}_{format_key}.npz",
                slots=slots,
                profiles=profiles,
                perpos=perpos,
                perpos_mask=np.ones((n, 4, 8), dtype=bool),
                nll=nll,
            )
            (turnstore_dir / f"{model_key}_{format_key}.json").write_text(
                json.dumps({"conv_ids": [f"smoke_{i:03d}" for i in range(n)]})
            )


def _all_cells() -> tuple[list[dict], list[dict]]:
    within = [_normalize_cell(c) for c in list(WITHIN_ROLE_CELLS) + list(TRACK_S_CELLS)]
    cross = [_normalize_cell(c) for c in CROSS_ROLE_CELLS]
    return within, cross


def main() -> int:
    parser = argparse.ArgumentParser(description="issue-825 vectorized cell fits")
    parser.add_argument("--turnstore-dir", type=Path, default=Path("data/issue_825/turnstore"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_825"))
    parser.add_argument("--cells", default="all")
    parser.add_argument("--null-draws", type=int, default=N_NULL_DRAWS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=FIT_SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--mlp-cells", default="S1,S2,M_instruct_assistant_chat,M_pretrained_assistant_chat"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    within, cross = _all_cells()
    if args.smoke:
        args.turnstore_dir = args.out_dir / "_smoke_turnstore"
        _fabricate_smoke_turnstore(args.turnstore_dir)
        within = [
            c
            for c in within
            if c["cell_id"] in ("M_instruct_assistant_chat", "M_pretrained_assistant_chat")
        ]
        cross = cross[:1]
        args.null_draws = min(args.null_draws, 5)
        args.n_boot = min(args.n_boot, 50)
    if args.cells != "all":
        wanted = set(args.cells.split(","))
        within = [c for c in within if c["cell_id"] in wanted]
        cross = [c for c in cross if c["cell_id"] in wanted]

    results: dict[str, dict] = {}
    for cell in within:
        try:
            res = run_cell(
                cell,
                args.turnstore_dir,
                args.out_dir,
                n_folds=args.folds,
                seed=args.seed,
                null_draws=args.null_draws,
                n_boot=args.n_boot,
            )
        except FileNotFoundError as e:
            print(f"[fit_cells] SKIP {cell['cell_id']}: {e}")
            continue
        results[cell["cell_id"]] = res
        if cell["cell_id"] == "S1":
            run_power_curve(res["xy"], args.out_dir, n_folds=args.folds, seed=args.seed)
            g1 = g1_gate(
                res["sweep"]["r2_obs"], args.out_dir, seed=args.seed, n=len(res["xy"]["conv_ids"])
            )
            if not g1["pass"] and not args.smoke:
                # Plan section 7 G1: materially outside the anchor = wiring
                # bug -> HALT; never publish downstream cells past a failed
                # replication anchor. g1_gate.json is already written.
                raise SystemExit(
                    "[fit_cells] G1 REPLICATION GATE FAILED — see g1_gate.json; "
                    "HALT per plan section 7 (exit 2)"
                )
        if cell["cell_id"] == "M_instruct_assistant_chat":
            # Plan section 7 G3: this cell must beat its selection-inherited
            # shuffle null before any user cell is interpreted (observed
            # layer-max vs the max of the null draws' own layer-maxes).
            summ = res["summary"]
            g3_pass = bool(summ["obs_layer_max_r2"] > max(summ["null_layer_max_r2_per_draw"]))
            _write_json(
                args.out_dir / "g3_gate.json",
                {
                    "metadata": _metadata(args.seed, len(res["xy"]["conv_ids"])),
                    "obs_layer_max_r2": summ["obs_layer_max_r2"],
                    "null_layer_max_r2_per_draw": summ["null_layer_max_r2_per_draw"],
                    "pass": g3_pass,
                },
            )
            if not g3_pass and not args.smoke:
                raise SystemExit(
                    "[fit_cells] G3 SANITY GATE FAILED — M_instruct_assistant_chat "
                    "does not beat its selection-inherited null; HALT per plan "
                    "section 7 before interpreting user cells (exit 3)"
                )
        if cell.get("format_key") == "chat":
            run_per_position(
                cell, args.turnstore_dir, args.out_dir, n_folds=args.folds, seed=args.seed
            )

    for cell in cross:
        try:
            run_cross_role_cell(
                cell,
                args.turnstore_dir,
                args.out_dir,
                n_folds=args.folds,
                seed=args.seed,
                null_draws=args.null_draws,
                n_boot=args.n_boot,
            )
        except FileNotFoundError as e:
            print(f"[fit_cells] SKIP cross {cell['cell_id']}: {e}")

    if results:
        run_nll_reads(results, args.out_dir, seed=args.seed)
        mlp_wanted = set(args.mlp_cells.split(","))
        for cid, res in results.items():
            if cid in mlp_wanted and not args.smoke:
                run_mlp_secondary(
                    res, args.out_dir, cell_id=cid, n_folds=args.folds, seed=args.seed
                )
    print("[fit_cells] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
