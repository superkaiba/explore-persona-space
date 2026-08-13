"""Pooled-source 4-tier transfer ladder (#2054 scope extension, writeup Result 3).

Per (cell x arm), score the POOLED context->answer map at Thomas's four tiers:

    tier 1  pooled_direct   W_pool as-is                       (banked pool_rungs m0 — merge joins it)
    tier 2  rotation_bias   R.W_pool.x + b*, Procrustes R,
                            train-refit bias                   (banked pool_rungs rot — merge joins it)
    tier 3  ctx_remap       W_pool(A x + a): affine context
                            re-map through the FROZEN pooled
                            map (fit mode, this script)
    tier 4  ans_remap       B(W_pool x) + b: affine answer-side
                            re-map of the pooled output         (fit mode, this script)

plus the banked own-map ceiling. Two modes:

  fit    (pod)  computes tiers 3+4 per unit x fold, checkpointed + resumable.
  merge  (VM)   joins tiers 1/2 from the banked pool_rungs per-cell JSONs,
                tiers 3/4 from fit outputs, and ceilings from the banked
                pool_specialize per-cell JSONs into
                eval_results/issue_2054/specialization_ladder/pooled_tier_ladder.json.

Tier-4 estimator: byte-reuse of ``SharedEighRidge`` (#825 fit-h-parity GCV
ridge, dof cap 0.9) with regressor z = W_pool(x).

Tier-3 estimator (NEW, closed form — recorded diff vs the pair-ladder rung 7):
``issue2054_ladder.py`` rung 7 fits A as a target->source CONTEXT ridge, which
requires paired source rows; a pooled source has no per-row counterpart for a
cell's rows (the cell IS a subset of the pool — the pool_rungs docstring makes
this point), so tier 3 fits the stated composite objective directly:

    min_C || Xs_tr C G - Yc_tr ||_F^2 + lambda ||C||_F^2,   G = diag(1/sigma_pool) . M_pool

with Xs cell-standardized contexts and Yc train-centered answers (free bias
b = ybar_tr; the class {x -> W_pool(A x + a)} is exactly {x -> Xs C G + b}).
Closed form via ONE eigh of the cell's train Gram (Q W Q^T) plus an SVD of G
(U S V^T, per (arm, fold) — shared across all cells): substituting B = C U and
right-rotating by V makes the problem column-separable, so for the whole
lambda grid predictions and GCV reduce to elementwise filters over the d x d
matrix R = Q^T Xs^T Yc V:

    yhat_c(te) = (Xs_te Q) [ R * F_lambda ] V^T,  F_lambda[i,j] = s_j^2 / (s_j^2 w_i + lambda)
    RSS(lambda) = ||Yc||^2 - 2 sum R^2*F + sum w R^2*F^2
    DOF(lambda) = sum_ij F_lambda[i,j]   (per-column hat traces summed)

GCV(lambda) = RSS / (n*d_out - DOF)^2 with the #1887-style cap on the MEAN
per-column dof (DOF/d_out <= dof_cap * n). An in-process identity gate checks
the closed form reduces exactly to a plain ridge when G = I (rel 1e-8).

No per-tier null draws or bootstrap sidecars: tiers are nested between the
banked pooled-direct read and the banked own-map ceiling, both of which carry
banked nulls (the pool_rungs convention).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

import numpy as np
from scipy.linalg import svd as scipy_svd

from explore_persona_space.experiments.issue_779.fit_h import reconstruction_metrics
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_ctx2ctx_fit import (
    ARM_VEC_KEY,
    ARMS,
    D_AMBIENT,
    GCV_DOF_CAP,
    Cell,
    SharedEighRidge,
    discover_cells,
    load_fold_map,
)
from scripts.issue2054_pool_specialize import (
    CONSTANT_X_VAR_FLOOR,
    PooledMomentRidge,
    _log,
    accumulate_pooled_moments,
    fit_pooled_per_fold,
    join_cell,
    load_cell_with_answer,
)

SCRIPT_VERSION = "issue2054_pooled_tier_ladder_v1"
TIERS = ("pooled_direct", "rotation_bias", "ctx_remap", "ans_remap")
# Tier-3 lambda grid: the per-column effective penalty is lambda / s_j^2, so
# the grid is wider than the standard DEFAULT_LAMBDAS to cover the pooled
# map's singular-value range; selected-lambda diagnostics flag grid edges.
CTX_REMAP_LAMBDAS = np.logspace(-4, 8, 25)


# ─────────────────────────────────────────────────────────────────────────────
# Tier-3 closed form


def _pooled_g_svd(m0: PooledMomentRidge) -> tuple[np.ndarray, np.ndarray]:
    """SVD (s, Vt) of G = diag(1/sigma_pool) . M_pool — the pooled map as a
    raw-x-space linear operator (per arm x fold; the caller caches it across
    cells). U is not needed by the closed form and is dropped."""
    g = (m0.map / m0.sd[:, None]).cpu().numpy().astype(np.float64)
    _u, s, vt = scipy_svd(g, lapack_driver="gesdd")
    return s, vt


def ctx_remap_fit_predict(
    x_tr: np.ndarray,
    x_te: np.ndarray,
    y_tr: np.ndarray,
    g_svd: tuple[np.ndarray, np.ndarray],
    *,
    lambdas: np.ndarray = CTX_REMAP_LAMBDAS,
    dof_cap: float = GCV_DOF_CAP,
) -> tuple[np.ndarray, dict]:
    """Closed-form GCV fit of min_C ||Xs C G - Yc||^2 + lambda ||C||^2 (module
    docstring derivation); returns (predictions at x_te, info dict).

    Asserts the ambient regime (n_train > d) — the under-determined case has
    no sanctioned read here (#1887 family).
    """
    n, d = x_tr.shape
    if n <= d:
        raise RuntimeError(f"ctx_remap left the ambient regime: n_train={n} <= d={d}")
    s_vals, vt = g_svd
    mx = x_tr.mean(axis=0)
    sx = x_tr.std(axis=0) + 1e-9  # population sd (fit_h parity)
    xs_tr = (x_tr - mx) / sx
    xs_te = (x_te - mx) / sx
    ybar = y_tr.mean(axis=0)
    yc = y_tr - ybar

    gram = xs_tr.T @ xs_tr
    w, q = np.linalg.eigh(gram)
    w = np.clip(w, 0.0, None)

    ycv = yc @ vt.T  # (n, d_out) rotated targets
    r = q.T @ (xs_tr.T @ ycv)  # (d, d_out)
    del ycv

    y_ss = float((yc**2).sum())
    s2 = s_vals**2  # (d_out,)
    d_out = int(s_vals.shape[0])

    best = None
    for lam in np.asarray(lambdas, dtype=np.float64):
        f = s2[None, :] / (s2[None, :] * w[:, None] + lam)  # (d, d_out)
        r2f = (r**2) * f
        rss = y_ss - 2.0 * float(r2f.sum()) + float((w[:, None] * r2f * f).sum())
        dof = float((f * w[:, None]).sum())  # sum_ij s_j^2 w_i / (s_j^2 w_i + lam)
        mean_col_dof = dof / d_out
        if mean_col_dof > dof_cap * n:
            continue
        denom = (n * d_out - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if best is None or gcv < best["gcv"]:
            best = {"lam": float(lam), "gcv": gcv, "dof": dof, "rss": rss}
    if best is None:
        raise RuntimeError(
            f"ctx_remap GCV dof cap {dof_cap}: every lambda exceeds cap*n_train={dof_cap * n:.0f}"
        )
    lam = best["lam"]
    f = s2[None, :] / (s2[None, :] * w[:, None] + lam)
    preds = (xs_te @ q) @ (r * f) @ vt + ybar
    lam_arr = np.asarray(lambdas, dtype=np.float64)
    info = {
        "best_lambda": lam,
        "dof_total": best["dof"],
        "mean_col_dof": best["dof"] / d_out,
        "selector": f"gcv_dof_cap_{dof_cap}_mean_col",
        "grid_edge": bool(lam in (float(lam_arr.min()), float(lam_arr.max()))),
        "n_train": n,
        "d_fit": d,
    }
    return preds, info


def _identity_gate(seed: int = 0) -> None:
    """Self-check: with G = I the closed form must reduce EXACTLY to plain
    ridge on standardized X at the same lambda (rel 1e-8); raises on violation."""
    rng = np.random.default_rng(seed)
    n, d = 300, 48
    x_tr = rng.normal(size=(n, d))
    x_te = rng.normal(size=(60, d))
    y_tr = rng.normal(size=(n, d))
    eye_svd = (np.ones(d), np.eye(d))
    lam = 3.7
    preds, _ = ctx_remap_fit_predict(
        x_tr, x_te, y_tr, eye_svd, lambdas=np.asarray([lam]), dof_cap=1.0
    )
    mx, sx = x_tr.mean(0), x_tr.std(0) + 1e-9
    xs_tr, xs_te = (x_tr - mx) / sx, (x_te - mx) / sx
    yc = y_tr - y_tr.mean(0)
    beta = np.linalg.solve(xs_tr.T @ xs_tr + lam * np.eye(d), xs_tr.T @ yc)
    ref = xs_te @ beta + y_tr.mean(0)
    rel = float(np.abs(preds - ref).max() / (np.abs(ref).max() + 1e-12))
    if rel > 1e-8:
        raise RuntimeError(f"ctx_remap identity gate FAIL: rel={rel:.3e}")
    _log(f"[pooledtier] identity gate OK (rel={rel:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
# Fit mode (pod)


def run_unit(
    cell: Cell,
    arm: str,
    fold_map: dict,
    pooled_models: dict[int, PooledMomentRidge],
    g_svd_cache: dict[tuple[str, int], tuple],
    folds_to_run: list[int],
    out_path: Path,
    fingerprint: str,
) -> None:
    """Tiers 3+4 for one (cell, arm), per-fold, checkpointed to out_path."""
    t_unit = time.time()
    k = int(fold_map["k"])
    vec = ARM_VEC_KEY[arm]
    act = load_cell_with_answer(cell)
    j = join_cell(act, fold_map["fold_of"], k, arm)
    x_all = np.asarray(act[vec][j["rows"]], dtype=np.float64)
    y_all = np.asarray(act["v_A"][j["rows"]], dtype=np.float64)
    del act

    fold_records: list[dict] = []
    for f in folds_to_run:
        t0 = time.time()
        te = j["fold_rows"][f]
        tr = np.concatenate([j["fold_rows"][g] for g in range(k) if g != f])
        x_tr, y_tr, x_te, y_te = x_all[tr], y_all[tr], x_all[te], y_all[te]
        n_tr = int(x_tr.shape[0])
        m0 = pooled_models[f]

        y_ss = float(((y_te - y_te.mean(0)) ** 2).sum())
        if y_ss < 1e-18:
            fold_records.append(
                {"fold": f, "n_cell_train": n_tr, "skipped": "constant-vector Y_eval"}
            )
            continue
        x_var_max = float(((x_tr - x_tr.mean(0)) ** 2).mean(axis=0).max())
        if x_var_max < CONSTANT_X_VAR_FLOOR:
            fold_records.append({"fold": f, "n_cell_train": n_tr, "skipped": "constant_x"})
            _log(f"[pooledtier] {cell.key} arm={arm} fold={f} SKIPPED (constant_x)")
            continue

        key = (arm, f)
        if key not in g_svd_cache:
            t_svd = time.time()
            g_svd_cache[key] = _pooled_g_svd(m0)
            _log(f"[pooledtier] G-SVD arm={arm} fold={f} elapsed={time.time() - t_svd:.1f}s")

        preds_ctx, info_ctx = ctx_remap_fit_predict(x_tr, x_te, y_tr, g_svd_cache[key])

        z_tr = m0.predict_np(x_tr)
        z_te = m0.predict_np(x_te)
        core = SharedEighRidge(z_tr, z_te)
        preds_ans, info_ans = core.fit_predict(y_tr)
        del core, z_tr, z_te

        rec = {
            "fold": f,
            "n_pooled_train": m0.n_train,
            "n_cell_train": n_tr,
            "n_test": int(len(te)),
            "d_ambient": D_AMBIENT,
            "well_posed": bool(n_tr > D_AMBIENT),
            "metrics": {
                "ctx_remap": reconstruction_metrics(preds_ctx, y_te),
                "ans_remap": reconstruction_metrics(preds_ans, y_te),
            },
            "lambda_diagnostics": {"ctx_remap": info_ctx, "ans_remap": info_ans},
            "wall_s": round(time.time() - t0, 1),
        }
        fold_records.append(rec)
        _log(
            f"[pooledtier] {cell.key} arm={arm} fold={f} "
            f"ctx={rec['metrics']['ctx_remap']['r2']:+.4f} "
            f"ans={rec['metrics']['ans_remap']['r2']:+.4f} elapsed={rec['wall_s']}s"
        )

    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "cell": cell.key,
        "arm": arm,
        "n_join": j["n_join"],
        "fingerprint": fingerprint,
        "folds": fold_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.json")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, out_path)
    _log(
        f"[pooledtier] unit {cell.key}__{arm} CHECKPOINTED -> {out_path} "
        f"(wall={round(time.time() - t_unit)}s)"
    )


def _fingerprint(fold_map: dict, arm: str, cell_key: str, folds: list[int]) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "v": SCRIPT_VERSION,
                "arm": arm,
                "cell": cell_key,
                "folds": folds,
                "fold_map_sha": fold_map["_sha256"],
                "tiers": ["ctx_remap", "ans_remap"],
                "ctx_lambdas": [
                    float(CTX_REMAP_LAMBDAS[0]),
                    float(CTX_REMAP_LAMBDAS[-1]),
                    len(CTX_REMAP_LAMBDAS),
                ],
            },
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()[:16]


def cmd_fit(args: argparse.Namespace) -> int:
    t_start = time.time()
    _identity_gate()
    fold_map = load_fold_map(args.fold_map_file, args.fold_map_ref)
    k = int(fold_map["k"])
    cells = discover_cells(args.activations_dir)
    folds_to_run = args.folds if args.folds else ([0] if args.pilot else list(range(k)))
    _log(f"[pooledtier] {len(cells)} cells, arms={args.arms}, folds={folds_to_run}")

    acc = accumulate_pooled_moments(cells, fold_map["fold_of"], k, args.arms, args.device)
    pooled_by_arm = {
        arm: fit_pooled_per_fold(acc["mom"][arm], folds_to_run, k) for arm in args.arms
    }
    del acc  # frees the ~2 GB of d x d moment tensors; models keep only derived pieces

    units = [(c, a) for a in args.arms for c in cells]
    units = units[args.shard :: args.num_shards]
    out_root = args.out_root / "pilot" if args.pilot else args.out_root
    if args.pilot:
        units = units[:1]
    _log(f"[pooledtier] shard {args.shard}/{args.num_shards}: {len(units)} units")

    g_svd_cache: dict[tuple[str, int], tuple] = {}
    n_done = 0
    for cell, arm in units:
        fp = _fingerprint(fold_map, arm, cell.key, folds_to_run)
        out_path = out_root / "percell_tiers" / f"{cell.key}__{arm}.json"
        if out_path.exists():
            try:
                prior = json.loads(out_path.read_text())
            except json.JSONDecodeError:
                prior = {}
            if prior.get("fingerprint") == fp:
                n_done += 1
                _log(f"[pooledtier] unit {cell.key}__{arm} already done — resume skip")
                continue
        run_unit(cell, arm, fold_map, pooled_by_arm[arm], g_svd_cache, folds_to_run, out_path, fp)
        n_done += 1
        _log(f"[pooledtier] progress {n_done}/{len(units)}")
    _log(f"[pooledtier] fit done units={n_done} wall={round(time.time() - t_start)}s")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Merge mode (VM)


def _fold_mean(unit: dict, metric: str) -> float | None:
    recs = [fr for fr in unit.get("folds") or unit.get("per_fold") or [] if "metrics" in fr]
    vals = [fr["metrics"][metric]["r2"] for fr in recs if metric in fr["metrics"]]
    return float(np.mean(vals)) if vals else None


def _quartiles(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def cmd_merge(args: argparse.Namespace) -> int:
    from scripts.issue2054_specialization_ladder import parse_cell_axes

    def load_dir(d: Path, what: str) -> dict[tuple[str, str], dict]:
        files = sorted(d.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no per-cell JSONs under {d} ({what})")
        out = {}
        for p in files:
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[(rec["cell"], rec["arm"])] = rec
        _log(f"[pooledtier] {what}: {len(out)} units")
        return out

    pr = load_dir(args.pool_rungs_dir, "pool_rungs (tiers 1/2)")
    ps = load_dir(args.pool_specialize_dir, "pool_specialize (ceilings)")
    ft = load_dir(args.fit_dir, "pooled_tier fits (tiers 3/4)")
    if not (set(pr) == set(ps) == set(ft)):
        raise RuntimeError(
            f"unit sets differ: pool_rungs {len(pr)}, pool_specialize {len(ps)}, fits {len(ft)}"
        )

    units = []
    for key in sorted(pr):
        cell, arm = key
        ceiling = ps[key]["pooled"]["ceiling"]
        usable = bool(not ceiling.get("missing") and ceiling.get("usable"))
        ceiling_r2 = float(ceiling["ceiling_r2"]) if not ceiling.get("missing") else None
        fit_recs = [fr for fr in ft[key]["folds"] if "metrics" in fr]
        skipped = [fr for fr in ft[key]["folds"] if "skipped" in fr]
        rot_degenerate = any(
            fr.get("degenerate_gain_rot") for fr in pr[key]["folds"] if "metrics" in fr
        )
        n_trains = [fr["n_cell_train"] for fr in ft[key]["folds"] if "n_cell_train" in fr]
        r2 = {
            "pooled_direct": _fold_mean(pr[key], "m0"),
            "rotation_bias": _fold_mean(pr[key], "rot"),
            "ctx_remap": _fold_mean(ft[key], "ctx_remap"),
            "ans_remap": _fold_mean(ft[key], "ans_remap"),
            "own_map": ceiling_r2,
        }
        frac = {
            t: (None if (v is None or not usable) else float(v / ceiling_r2)) for t, v in r2.items()
        }
        lam = {
            t: {
                "per_fold": [{"fold": fr["fold"], **fr["lambda_diagnostics"][t]} for fr in fit_recs]
            }
            for t in ("ctx_remap", "ans_remap")
        }
        well_posed = bool(n_trains and min(n_trains) > D_AMBIENT)
        units.append(
            {
                "cell": cell,
                "arm": arm,
                **parse_cell_axes(cell),
                "n_join": ft[key]["n_join"],
                "n_train_min": min(n_trains) if n_trains else None,
                "n_train_max": max(n_trains) if n_trains else None,
                "d_ambient": D_AMBIENT,
                "well_posed": well_posed,
                "descriptive_only": bool(not well_posed),
                "degenerate": (skipped[0]["skipped"] if skipped and not fit_recs else None),
                "degenerate_rotation_bias": rot_degenerate,
                "n_folds_scored": len(fit_recs),
                "ceiling_r2": ceiling_r2,
                "ceiling_usable": usable,
                "r2": r2,
                "fraction_of_ceiling": frac,
                "lambda_diagnostics": lam,
            }
        )

    tiers_all = [*TIERS, "own_map"]
    aggregates: dict = {}
    for arm in ARMS:
        arm_rows = [u for u in units if u["arm"] == arm]
        usable_rows = [u for u in arm_rows if u["ceiling_usable"]]
        aggregates[arm] = {
            "n_units": len(arm_rows),
            "n_ceiling_usable": len(usable_rows),
            "per_tier": {
                t: {
                    **(
                        {"r2": _quartiles([u["r2"][t] for u in arm_rows if u["r2"][t] is not None])}
                        if any(u["r2"][t] is not None for u in arm_rows)
                        else {}
                    ),
                    **(
                        {
                            "fraction_of_ceiling": _quartiles(
                                [
                                    u["fraction_of_ceiling"][t]
                                    for u in usable_rows
                                    if u["fraction_of_ceiling"][t] is not None
                                ]
                            )
                        }
                        if any(u["fraction_of_ceiling"][t] is not None for u in usable_rows)
                        else {}
                    ),
                }
                for t in tiers_all
            },
        }
        by_axis: dict = {}
        for axis in ("framing", "character", "model", "provenance"):
            levels: dict[str, list[dict]] = defaultdict(list)
            for u in usable_rows:
                levels[u[axis]].append(u)
            by_axis[axis] = {
                level: {
                    "n_cells": len(rows),
                    "per_tier": {
                        t: _quartiles(
                            [
                                u["fraction_of_ceiling"][t]
                                for u in rows
                                if u["fraction_of_ceiling"][t] is not None
                            ]
                        )
                        for t in tiers_all
                        if any(u["fraction_of_ceiling"][t] is not None for u in rows)
                    },
                }
                for level, rows in sorted(levels.items())
            }
        aggregates[arm]["by_axis"] = by_axis

    payload = {
        "metadata": {
            **as_metadata_dict(git_provenance(_REPO)),
            "script_version": SCRIPT_VERSION,
            "argv": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tiers": list(tiers_all),
            "tier_sources": {
                "pooled_direct": "banked pool_rungs m0 (fold-mean held-out R^2)",
                "rotation_bias": "banked pool_rungs rot (Procrustes R on centered "
                "clouds + train-mean bias == R.W_pool.x + b*)",
                "ctx_remap": "this run: closed-form composite ridge (module docstring)",
                "ans_remap": "this run: SharedEighRidge on z = W_pool(x)",
                "own_map": "banked per-cell ceiling (pool_specialize ceiling join)",
            },
        },
        "units": units,
        "aggregates": aggregates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, args.out)
    _log(f"[pooledtier] merged -> {args.out} ({len(units)} units)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    fit = sub.add_parser("fit", help="compute tiers 3+4 per unit (pod)")
    fit.add_argument("--activations-dir", type=Path, required=True)
    fit.add_argument("--out-root", type=Path, required=True)
    fit.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    fit.add_argument("--folds", nargs="+", type=int, default=None)
    fit.add_argument("--fold-map-file", type=Path, default=None)
    fit.add_argument("--fold-map-ref", default="origin/issue-2054")
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--shard", type=int, default=0)
    fit.add_argument("--num-shards", type=int, default=1)
    fit.add_argument("--pilot", action="store_true", help="first unit, fold 0 only")
    fit.add_argument("--import-check", action="store_true")

    mg = sub.add_parser("merge", help="join banked tiers 1/2 + fit tiers 3/4 (VM)")
    mg.add_argument("--pool-rungs-dir", type=Path, required=True)
    mg.add_argument("--pool-specialize-dir", type=Path, required=True)
    mg.add_argument("--fit-dir", type=Path, required=True)
    mg.add_argument("--out", type=Path, required=True)
    mg.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[pooledtier] import-check OK")
        return 0
    if args.mode == "fit":
        return cmd_fit(args)
    return cmd_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
