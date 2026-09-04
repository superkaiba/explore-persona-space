#!/usr/bin/env python3
"""Rank vs n_train probe for the issue-2588 reduced-rank mapping panel.

The panel's +10% reduced-rank ranks (roughly 100 to 180 directions) were all
measured on maps fitted on the same train_10k split, so the near-constant rank
across hidden widths 1024 to 5120 could be an estimator artifact (usable rank
bounded by n_train) rather than a property of the models.  This probe refits
the Qwen3.5 27B (d = 5120) and Qwen3.5 0.8B (d = 1024) maps, both arms each,
on nested training subsets (1250 to 7500 rows, 3 seeds each) plus the full
production split, mirroring the production estimator exactly (subset-own
fp64 mean, unbiased std + 1e-9 x standardization, y centering, primal ridge
at the frozen production lambda), and reports how the rank at +10% relative
validation SSE moves with n.  A secondary read re-selects lambda per subset
on the validation split from a 13-point log grid spanning 1/100 to 100 times
the production value.

Estimator-validity note: at d = 5120 the n = 1250 and n = 2500 points have
n_train < d and are deliberately under-determined.  The whole purpose is to
see how rank depends on n, including below d.  The decisive comparison is
5000 vs 7500 vs the full split (all at or above d) for the 27B maps, and the
whole grid for the 0.8B maps (all n >= d = 1024).

Realized-split deviation: train_10k holds fewer than 10000 usable rows per
map (9920, 9767 and 9796 for three of the four maps, and 4690 for the 0.8B
end-of-thought arm), so the top grid point is the realized full split rather
than a nominal 10000, and subset sizes at or above the realized total are
skipped.

Resumable: fits already present in the output JSON are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2588_mapping_rank_vs_capability as MR  # noqa: E402

CELLS = ("q35_27b_a", "q35_27b_b", "q35_0p8b_a", "q35_0p8b_b")
SUBSET_SIZES = (1250, 2500, 5000, 7500)
SEEDS = (0, 1, 2)
LAMBDA_GRID_POINTS = 13
LAMBDA_GRID_SPAN = 100.0
MIN_FREE_GB = 20.0
PARITY_TOL = 3e-4
# Widths above this use the fast paths (dual-form ridge below n = d, and a
# top-k partial eigh of the fitted-output second-moment matrix).  The metrics
# are exact: the top-k eigenvectors are the same nested basis as the full
# decomposition's first k directions, and the variance total uses the trace.
FAST_D_THRESHOLD = 2048
TOP_K = 1024
OUT_JSON = MR.REPO / "eval_results" / "issue_2588" / "rank_vs_ntrain.json"
FIG_PNG = MR.REPO / "figures" / "issue_2588" / "rank_vs_ntrain.png"

ARM_DISPLAY = {"no-thinking": "prompt read", "end-of-thought": "end-of-thought"}

NOTES = (
    "Estimator-validity note: at d = 5120 the n = 1250 and n = 2500 points have "
    "n_train < d and are deliberately under-determined; the whole purpose is to see how "
    "rank depends on n, including below d. The decisive comparison is 5000 vs 7500 vs the "
    "full split (all at or above d) for the 27B maps, and the whole grid for the 0.8B maps "
    "(all n >= d = 1024). Realized-split deviation: train_10k holds fewer than 10000 usable "
    "rows per map, so the top grid point is the realized full split (9920 / 9767 / 9796 / "
    "4690 rows) and subset sizes at or above the realized total are skipped."
)
PLATEAU_RULE = (
    "ratio = mean rank_rel10 at the full production split over mean rank_rel10 at the base "
    "n (5000 for the d=5120 maps, the smallest size at or near d; 2500 for the d=1024 maps). "
    "ratio <= 1.15 means plateau, ratio > 1.3 means still climbing, otherwise ambiguous. "
    "Ranks are at the fixed production lambda."
)


def spec_for(cell: str) -> MR.MapSpec:
    matches = [m for m in MR.MAPS if m.cell == cell]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one MapSpec for cell {cell}, got {len(matches)}")
    return matches[0]


def fit_id(cell: str, n: int, seed: int | None) -> str:
    return f"{cell}|n{n}|s{'prod' if seed is None else seed}"


def check_disk() -> None:
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    probe = cache_root
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"only {free_gb:.1f} GB free on the HF cache filesystem ({probe}); "
            f"need at least {MIN_FREE_GB:.0f} GB"
        )
    print(f"disk check: {free_gb:.1f} GB free on {probe}", flush=True)


def _metrics_from_basis(
    pred_val: np.ndarray,
    yval: np.ndarray,
    ymu: np.ndarray,
    right: np.ndarray,
    evals_over_n: np.ndarray,
    total_var: float,
    truncated: bool,
) -> dict[str, Any]:
    """Reduced-rank metrics from a (possibly truncated) top-k output basis.

    Mirrors MR.rrr_curves and MR.analyze_map: the validation R^2 curve over
    the nested top-k principal directions of the fitted training outputs, the
    rank at +10% relative validation SSE, the rank at absolute R^2 gap 0.02,
    and the direction count holding 90% of the fitted-output variance.  A
    truncated basis is exact for ranks within k; every threshold must be
    crossed inside the basis or the run fails.
    """
    full_val = MR.pooled_r2(pred_val, yval)
    val_curve = MR.r2_curve_from_top_right_vectors(pred_val, yval, ymu, right)
    if not truncated and abs(float(val_curve[-1]) - full_val) > 1e-4:
        raise RuntimeError(
            f"full-rank RRR curve {val_curve[-1]:.6f} does not match full R2 {full_val:.6f}"
        )
    rel_thr = 1.0 - (1.0 - full_val) * (1.0 + MR.REL_ERROR_TOLERANCE)
    rank_rel10 = MR.rank_at_threshold(val_curve, rel_thr)
    rank_abs02 = MR.minimum_rank_within(val_curve, full_val, MR.R2_TOLERANCE)
    if rank_rel10 is None or rank_abs02 is None:
        raise RuntimeError(
            f"RRR validation curve never reached the tolerance within the top-{right.shape[1]} "
            "basis; raise TOP_K"
        )
    cum = np.cumsum(evals_over_n) / (total_var + 1e-30)
    if cum[-1] < 0.90:
        raise RuntimeError(
            f"top-{len(evals_over_n)} eigenvalues hold only {cum[-1]:.3f} of the fitted-output "
            "variance; raise TOP_K"
        )
    dirs_90pct = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "rank_rel10": int(rank_rel10),
        "rank_abs02": int(rank_abs02),
        "dirs_90pct": dirs_90pct,
        "full_val_r2": full_val,
    }


def _top_k_output_basis(m: np.ndarray, n_rows: int, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-k eigenpairs (descending) of the fitted-output second-moment matrix.

    Returns (right_vectors d x k, eigenvalues/n descending, total variance),
    where total variance uses the exact trace so the 90% mass count does not
    depend on the truncation.
    """
    d = m.shape[0]
    m = 0.5 * (m + m.T)
    total_var = float(np.trace(m)) / n_rows
    if k >= d:
        evals, evecs = scipy.linalg.eigh(m, check_finite=False)
    else:
        evals, evecs = scipy.linalg.eigh(m, check_finite=False, subset_by_index=[d - k, d - 1])
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None) / n_rows
    right = np.ascontiguousarray(evecs[:, order])
    return right, evals, total_var


def fit_subset(
    spec: MR.MapSpec,
    *,
    d: int,
    layer: int,
    lam_prod: float,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
    idx: np.ndarray | None,
    n: int,
    seed: int | None,
    prod_expect: tuple[float, float] | None,
) -> dict[str, Any]:
    """One production-mirrored ridge fit on a training subset, both lambda legs.

    The estimator is the production one exactly (subset-own fp64 mean and
    unbiased-std x standardization, y centering, primal ridge at the given
    lambda).  Widths at or below FAST_D_THRESHOLD keep the original
    fp32-payload path.  Wider maps take algebraically identical fast paths:
    the dual-form solve when n < d, and a top-k partial eigendecomposition of
    the fitted-output second-moment matrix for the rank curves.
    """
    started = time.time()
    x = xtr if idx is None else xtr[idx]
    y = ytr if idx is None else ytr[idx]
    if x.shape[0] != n:
        raise RuntimeError(f"subset has {x.shape[0]} rows, expected {n}")

    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    xmu = x64.mean(axis=0)
    xsd = x64.std(axis=0, ddof=1) + 1e-9
    ymu = y64.mean(axis=0)
    xn = (x64 - xmu) / xsd
    yc = y64 - ymu
    del x64, y64
    xvn = (xval.astype(np.float64) - xmu) / xsd
    xtn = (xte.astype(np.float64) - xmu) / xsd
    k_top = min(TOP_K, d)

    grid = np.geomspace(lam_prod / LAMBDA_GRID_SPAN, lam_prod * LAMBDA_GRID_SPAN, 13)
    if not math.isclose(float(grid[6]), lam_prod, rel_tol=1e-9):
        raise RuntimeError(f"lambda grid midpoint {grid[6]} is not the production {lam_prod}")

    if d > FAST_D_THRESHOLD and n < d:
        impl = "dual_fast"
        kmat = xn @ xn.T
        ek, u = scipy.linalg.eigh(kmat, check_finite=False, driver="evd")
        uty = u.T @ yc
        bv = (xvn @ xn.T) @ u
        bt = (xtn @ xn.T) @ u
        scan_r2 = [float(MR.pooled_r2((bv / (ek + lg)) @ uty + ymu, yval)) for lg in grid]

        def leg(lam_val: float) -> dict[str, Any]:
            dmat = uty / (ek + lam_val)[:, None]
            pred_val = bv @ dmat + ymu
            pred_test = bt @ dmat + ymu
            xw = kmat @ (u @ dmat)
            m = xw.T @ xw
            right, evals, total = _top_k_output_basis(m, n, k_top)
            metrics = _metrics_from_basis(
                pred_val, yval, ymu, right, evals, total, truncated=k_top < d
            )
            metrics["full_test_r2"] = MR.pooled_r2(pred_test, yte)
            metrics["lambda"] = float(lam_val)
            return metrics

    elif d > FAST_D_THRESHOLD:
        impl = "primal_fast"
        gram = xn.T @ xn
        cross = xn.T @ yc
        del xn, yc
        eg, v = scipy.linalg.eigh(gram, check_finite=False, driver="evd")
        del gram
        t = v.T @ cross
        del cross
        pv = xvn @ v
        pt = xtn @ v
        seg = np.sqrt(np.clip(eg, 0.0, None))
        scan_r2 = [float(MR.pooled_r2(pv @ (t / (eg + lg)[:, None]) + ymu, yval)) for lg in grid]

        def leg(lam_val: float) -> dict[str, Any]:
            coef = t / (eg + lam_val)[:, None]
            pred_val = pv @ coef + ymu
            pred_test = pt @ coef + ymu
            s = seg[:, None] * coef
            m = s.T @ s
            right, evals, total = _top_k_output_basis(m, n, k_top)
            metrics = _metrics_from_basis(
                pred_val, yval, ymu, right, evals, total, truncated=k_top < d
            )
            metrics["full_test_r2"] = MR.pooled_r2(pred_test, yte)
            metrics["lambda"] = float(lam_val)
            return metrics

    else:
        impl = "primal_full_fp32w"
        gram = xn.T @ xn
        cross = xn.T @ yc
        del xn, yc
        evals_g, vecs = scipy.linalg.eigh(gram, check_finite=False)
        t = vecs.T @ cross
        del cross
        xmu32 = np.asarray(xmu, dtype=np.float32)
        xsd32 = np.asarray(xsd, dtype=np.float32)
        ymu32 = np.asarray(ymu, dtype=np.float32)
        xvn32 = (xval - xmu32) / xsd32
        xtn32 = (xte - xmu32) / xsd32
        pv = xvn @ vecs
        scan_r2 = [
            float(MR.pooled_r2(pv @ (t / (evals_g + lg)[:, None]) + ymu, yval)) for lg in grid
        ]

        def leg(lam_val: float) -> dict[str, Any]:
            coef = t / (evals_g + lam_val)[:, None]
            w32 = np.asarray(vecs @ coef, dtype=np.float32)
            pred_val = xvn32 @ w32 + ymu32
            pred_test = xtn32 @ w32 + ymu32
            m = w32.astype(np.float64).T @ (gram @ w32.astype(np.float64))
            right, evals, total = _top_k_output_basis(m, n, m.shape[0])
            metrics = _metrics_from_basis(
                pred_val, yval, ymu32, right, evals, total, truncated=False
            )
            metrics["full_test_r2"] = MR.pooled_r2(pred_test, yte)
            metrics["lambda"] = float(lam_val)
            return metrics

    sel_idx = int(np.argmax(scan_r2))
    fixed = leg(lam_prod)
    if prod_expect is not None:
        exp_val, exp_test = prod_expect
        if (
            abs(fixed["full_val_r2"] - exp_val) > PARITY_TOL
            or abs(fixed["full_test_r2"] - exp_test) > PARITY_TOL
        ):
            raise RuntimeError(
                f"{spec.key}: production parity failed: val {fixed['full_val_r2']:.6f} vs "
                f"{exp_val:.6f}, test {fixed['full_test_r2']:.6f} vs {exp_test:.6f}"
            )
    if sel_idx == 6:
        reselected = dict(fixed)
        reselected["same_as_fixed"] = True
    else:
        reselected = leg(float(grid[sel_idx]))
        reselected["same_as_fixed"] = False
    reselected["grid_index"] = sel_idx

    elapsed = time.time() - started
    print(
        f"[{spec.key}] n={n} seed={seed} rank_rel10={fixed['rank_rel10']} "
        f"val_r2={fixed['full_val_r2']:.4f} sel_lambda={reselected['lambda']:g} "
        f"rank_rel10_sel={reselected['rank_rel10']} ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        "id": fit_id(spec.cell, n, seed),
        "cell": spec.cell,
        "key": spec.key,
        "model": spec.model_label,
        "model_display": MR.DISPLAY_NAMES[spec.model_label],
        "arm": spec.arm,
        "d": d,
        "layer_star": layer,
        "n_train": n,
        "seed": seed,
        "is_production": seed is None,
        "n_below_d": n < d,
        "impl": impl,
        "lambda_fixed": float(lam_prod),
        "lambda_reselected": reselected["lambda"],
        "full_val_r2": fixed["full_val_r2"],
        "full_test_r2": fixed["full_test_r2"],
        "rank_rel10": fixed["rank_rel10"],
        "rank_abs02": fixed["rank_abs02"],
        "dirs_90pct": fixed["dirs_90pct"],
        "reselected": reselected,
        "lambda_grid": [float(v) for v in grid],
        "lambda_grid_val_r2": scan_r2,
        "elapsed_s": elapsed,
    }


def planned_for(cell: str, n_total: int) -> list[tuple[int, int | None]]:
    plan: list[tuple[int, int | None]] = [(n_total, None)]
    for n in SUBSET_SIZES:
        if n < n_total:
            plan.extend((n, s) for s in SEEDS)
    return plan


def run_cell(
    cell: str,
    fits: dict[str, dict[str, Any]],
    budget: list[int],
) -> None:
    spec = spec_for(cell)
    fitrec = MR._fit_record(spec)
    layer = int(fitrec["layer_star"])
    star = fitrec["layers"][str(layer)]
    d = int(star["d"])
    lam_prod = float(star["fit_meta"]["selected_lambda"])
    prod_expect = (
        float(star["fit_meta"]["val_r2_at_selected"]),
        float(star["test_r2"]),
    )
    print(f"[{spec.key}] load splits at L{layer} (d={d}, lambda={lam_prod:g})", flush=True)
    xtr, ytr = MR.load_split(spec, "train_10k", layer, h_dim=d)
    xval, yval = MR.load_split(spec, "val_400", layer, h_dim=d)
    xte, yte = MR.load_split(spec, "test_1000", layer, h_dim=d)
    if xtr.shape[1] != d or ytr.shape[1] != d:
        raise RuntimeError(f"{spec.key}: dimension mismatch X={xtr.shape} Y={ytr.shape} d={d}")
    n_total = int(xtr.shape[0])
    if n_total <= max(SEEDS) + 1 or n_total < SUBSET_SIZES[1]:
        raise RuntimeError(f"{spec.key}: train split has only {n_total} rows")
    perms = {s: np.random.default_rng(s).permutation(n_total) for s in SEEDS}
    for n, seed in planned_for(cell, n_total):
        key = fit_id(cell, n, seed)
        if key in fits:
            continue
        if budget[0] <= 0:
            print(f"[{spec.key}] fit budget exhausted, stopping (resumable)", flush=True)
            return
        idx = None if seed is None else np.sort(perms[seed][:n])
        fits[key] = fit_subset(
            spec,
            d=d,
            layer=layer,
            lam_prod=lam_prod,
            xtr=xtr,
            ytr=ytr,
            xval=xval,
            yval=yval,
            xte=xte,
            yte=yte,
            idx=idx,
            n=n,
            seed=seed,
            prod_expect=prod_expect if seed is None else None,
        )
        budget[0] -= 1
        save(fits)


def summarize(fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cell in CELLS:
        rows = [f for f in fits.values() if f["cell"] == cell]
        if not rows:
            continue
        d = rows[0]["d"]
        prod_rows = [f for f in rows if f["is_production"]]
        by_n: dict[str, Any] = {}
        for n in sorted({f["n_train"] for f in rows}):
            grp = [f for f in rows if f["n_train"] == n]
            ranks = [f["rank_rel10"] for f in grp]
            ranks_sel = [f["reselected"]["rank_rel10"] for f in grp]
            by_n[str(n)] = {
                "n_fits": len(grp),
                "mean_rank_rel10": float(np.mean(ranks)),
                "min_rank_rel10": int(min(ranks)),
                "max_rank_rel10": int(max(ranks)),
                "mean_rank_rel10_reselected": float(np.mean(ranks_sel)),
                "mean_full_val_r2": float(np.mean([f["full_val_r2"] for f in grp])),
                "mean_full_test_r2": float(np.mean([f["full_test_r2"] for f in grp])),
                "reselected_lambdas": sorted({f["lambda_reselected"] for f in grp}),
            }
        entry: dict[str, Any] = {
            "model_display": rows[0]["model_display"],
            "arm": rows[0]["arm"],
            "d": d,
            "production_n": prod_rows[0]["n_train"] if prod_rows else None,
            "by_n": by_n,
        }
        base_n = 5000 if d == 5120 else 2500
        if prod_rows and str(base_n) in by_n:
            top = by_n[str(prod_rows[0]["n_train"])]
            for label, field in (
                ("plateau", "mean_rank_rel10"),
                ("plateau_reselected", "mean_rank_rel10_reselected"),
            ):
                ratio = top[field] / by_n[str(base_n)][field]
                if ratio <= 1.15:
                    verdict = "plateau"
                elif ratio > 1.3:
                    verdict = "still-climbing"
                else:
                    verdict = "ambiguous"
                entry[label] = {
                    "base_n": base_n,
                    "top_n": prod_rows[0]["n_train"],
                    "ratio": float(ratio),
                    "verdict": verdict,
                }
        summary[cell] = entry
    return summary


def save(fits: dict[str, dict[str, Any]]) -> None:
    cell_order = {c: i for i, c in enumerate(CELLS)}
    rows = sorted(
        fits.values(),
        key=lambda f: (cell_order[f["cell"]], f["n_train"], -1 if f["seed"] is None else f["seed"]),
    )
    payload = {
        "schema_version": "issue2588_rank_vs_ntrain_v1",
        "hf_revision": MR.HF_REVISION,
        "notes": NOTES,
        "plateau_rule": PLATEAU_RULE,
        "rank_definition": MR.PRIMARY_RANK_DEFINITION,
        "lambda_grid_rule": (
            f"{LAMBDA_GRID_POINTS}-point log grid spanning 1/{LAMBDA_GRID_SPAN:g} to "
            f"{LAMBDA_GRID_SPAN:g} times the production selected_lambda; midpoint (index 6) "
            "is the production value; re-selection maximizes full validation pooled R^2"
        ),
        "summary": summarize(fits),
        "fits": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load() -> dict[str, dict[str, Any]]:
    if not OUT_JSON.exists():
        return {}
    rows = json.loads(OUT_JSON.read_text(encoding="utf-8"))["fits"]
    return {r["id"]: r for r in rows}


def is_complete(fits: dict[str, dict[str, Any]]) -> bool:
    for cell in CELLS:
        prod = [f for f in fits.values() if f["cell"] == cell and f["is_production"]]
        if not prod:
            return False
        planned = planned_for(cell, prod[0]["n_train"])
        if any(fit_id(cell, n, s) not in fits for n, s in planned):
            return False
    return True


def render_figure(fits: dict[str, dict[str, Any]]) -> None:
    style = {
        "q35_27b_a": ("tab:blue", "-", "o"),
        "q35_27b_b": ("tab:blue", "--", "s"),
        "q35_0p8b_a": ("tab:orange", "-", "o"),
        "q35_0p8b_b": ("tab:orange", "--", "s"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for cell in CELLS:
        rows = [f for f in fits.values() if f["cell"] == cell]
        color, ls, marker = style[cell]
        label = f"{rows[0]['model_display']}, {ARM_DISPLAY[rows[0]['arm']]} (d={rows[0]['d']})"
        ns = sorted({f["n_train"] for f in rows})
        for ax, field in ((axes[0], "rank_rel10"), (axes[1], "full_val_r2")):
            mean, lo, hi = [], [], []
            for n in ns:
                vals = [f[field] for f in rows if f["n_train"] == n]
                mean.append(float(np.mean(vals)))
                lo.append(mean[-1] - min(vals))
                hi.append(max(vals) - mean[-1])
            ax.errorbar(
                ns,
                mean,
                yerr=[lo, hi],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=4,
                capsize=3,
                linewidth=1.5,
                label=label,
            )
    axes[0].axhline(1024, color="grey", linestyle=":", linewidth=1.2, label="d = 1024")
    axes[0].axvline(5120, color="grey", linestyle="--", linewidth=1.2, label="n = d for 27B")
    axes[0].set_ylabel("rank at +10% relative validation SSE")
    # Log scale keeps the 40 to 150 rank curves readable next to the d = 1024 reference.
    axes[0].set_yscale("log")
    axes[0].set_title("Reduced-rank rank vs training set size")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set_ylabel("full-map pooled validation R$^2$")
    axes[1].set_title("Full-map validation R$^2$ vs training set size")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("n_train")
    fig.tight_layout()
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=200)
    fig.savefig(FIG_PNG.with_suffix(".pdf"))
    plt.close(fig)
    print(f"figure written: {FIG_PNG}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    parser.add_argument("--max-new-fits", type=int, default=10**9)
    parser.add_argument("--figure-only", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()
    fits = load()
    if not args.figure_only:
        check_disk()
        budget = [args.max_new_fits]
        for cell in args.cells:
            if budget[0] <= 0:
                break
            run_cell(cell, fits, budget)
        save(fits)
    if args.skip_figure:
        return
    if is_complete(fits):
        render_figure(fits)
    else:
        print("fit grid incomplete, skipping figure (re-run to resume)", flush=True)


if __name__ == "__main__":
    main()
