"""Fit the mechanical noise-floor curve to the #779 per-direction R^2 spectrum.

Question (Result 2 refinement): how much of the per-direction R^2 spectrum is the
MECHANICAL consequence of a direction-independent unpredictable floor, and which
directions are badly predicted BEYOND that expectation?

Model: R^2_j = 1 - c_j / lambda_j with c_j the residual variance in direction j.
  - isotropic floor (1 param):   c_j = a         ->  R^2_j = 1 - a/lambda_j
  - floor + proportional (2 p):  c_j = a + b*lambda_j -> R^2_j = (1-b) - a/lambda_j
Both are linear in x_j = 1/lambda_j (OLS); the 1-param form pins the intercept at 1.
lambda_j enters as the VARIANCE SHARE (eigenvalue / total) — the unknown total
variance rescales `a` but neither `b`, the fit, nor the deviations.

Censoring: a shrinkage estimator's held-out per-direction R^2 floors near 0 where
the signal is below the noise (predicting the mean), so deep-tail points do NOT
follow the hyperbola and would dominate an OLS through sheer 1/lambda leverage.
The fit set is the top-256 variance ranks (the "retained prefix" convention used
by the worst-direction analyses); the tail is shown censored in the figure.

Reads the banked n10k arrays only; computes nothing new on GPU.
"""

from __future__ import annotations

import argparse
import json

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root

SRC = "eval_results/issue_779/fitter-fair-comparison-n10k/perdirection_per_predictor_n10k.json"
OUT_DIR = "eval_results/issue_779/spectrum_floor_fit"
FIG_DIR = "figures/issue_779"
FIT_TOP = 256  # fit set: top-256 variance ranks (censoring boundary, see docstring)
N_OUTLIERS = 15


def _ols(x: np.ndarray, y: np.ndarray, pin_intercept: float | None = None) -> tuple[float, float]:
    """OLS of y on x. Returns (intercept, slope); intercept optionally pinned."""
    if pin_intercept is not None:
        slope = float(np.dot(x, y - pin_intercept) / np.dot(x, x))
        return pin_intercept, slope
    X = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitter", default="ridge", help="primary fitter for figure/outliers")
    args = ap.parse_args()

    root = repo_root()
    d = json.loads((root / SRC).read_text())
    ranks = np.asarray(d["ranks_evaluated"], dtype=int)
    share = np.asarray(d["variance_share_by_rank"], dtype=float)
    assert share.min() > 0, "non-positive variance share"

    fit_mask = ranks < FIT_TOP
    x = 1.0 / share  # in inverse-share units; rescaling absorbs into `a`

    results: dict[str, dict] = {}
    for fitter, block in d["per_predictor"].items():
        if not isinstance(block, dict) or "r2_by_rank" not in block:
            continue
        r2 = np.asarray(block["r2_by_rank"], dtype=float)
        xm, ym = x[fit_mask], r2[fit_mask]

        a1_int, a1_slope = _ols(xm, ym, pin_intercept=1.0)  # 1 - a/lambda
        a2_int, a2_slope = _ols(xm, ym)  # (1-b) - a/lambda
        pred1 = a1_int + a1_slope * x
        pred2 = a2_int + a2_slope * x

        def _gof(pred: np.ndarray) -> float:
            ss_res = float(((ym - pred[fit_mask]) ** 2).sum())
            ss_tot = float(((ym - ym.mean()) ** 2).sum())
            return 1.0 - ss_res / ss_tot

        # deviation in R^2 units and excess residual variance in share units:
        # excess_j = lambda_j * (pred - obs)  (positive = worse than mechanical)
        dev2 = pred2 - r2
        excess = share * dev2
        order = np.argsort(dev2[fit_mask])[::-1]
        fit_ranks = ranks[fit_mask]

        def _rows(idx: np.ndarray) -> list[dict]:
            return [
                {
                    "rank": int(fit_ranks[i]),
                    "r2_observed": float(r2[fit_mask][i]),
                    "r2_curve2": float(pred2[fit_mask][i]),
                    "deviation": float(dev2[fit_mask][i]),
                    "excess_share_units": float(excess[fit_mask][i]),
                }
                for i in idx
            ]

        b = 1.0 - a2_int
        results[fitter] = {
            "one_param": {
                "a": -a1_slope,
                "gof_r2_on_fit_set": _gof(pred1),
                "implied_zero_crossing_share": -a1_slope,  # lambda where 1 - a/l = 0
            },
            "two_param": {
                "a": -a2_slope,
                "b_proportional_loss": b,
                "asymptote_1_minus_b": a2_int,
                "gof_r2_on_fit_set": _gof(pred2),
                # curve crosses zero at lambda = a / (1-b)
                "implied_zero_crossing_share": (-a2_slope / a2_int) if a2_int > 0 else None,
            },
            "worst_below_curve": _rows(order[:N_OUTLIERS]),
            "best_above_curve": _rows(order[::-1][:N_OUTLIERS]),
        }

    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "question": (
                "How much of the per-direction R^2 spectrum is the mechanical "
                "consequence of a direction-independent unpredictable floor?"
            ),
            "model": "R^2_j = 1 - c_j/lambda_j; c_j = a (1-param) or a + b*lambda_j (2-param)",
            "fit_set": f"top-{FIT_TOP} variance ranks (tail floor-censored by shrinkage)",
            "lambda_units": "variance SHARE (eigenvalue/total); rescales a, not b/gof/deviations",
            "source": SRC,
        },
        "n_fit_points": int(fit_mask.sum()),
        "fitters": results,
    }
    (out / "spectrum_floor_fit.json").write_text(json.dumps(payload, indent=1))

    # ── figure: observed spectrum + fitted curves + deviations (primary fitter) ──
    set_paper_style()
    import matplotlib.pyplot as plt

    r2 = np.asarray(d["per_predictor"][args.fitter]["r2_by_rank"], dtype=float)
    res = results[args.fitter]
    pred1 = 1.0 - res["one_param"]["a"] * x
    pred2 = res["two_param"]["asymptote_1_minus_b"] - res["two_param"]["a"] * x
    colors = paper_palette(4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    m = fit_mask
    ax1.scatter(share[m], r2[m], s=14, color=colors[0], label="observed (fit set, top-256)")
    ax1.scatter(
        share[~m], r2[~m], s=10, color=colors[0], alpha=0.25, label="observed (tail, censored)"
    )
    xs = np.argsort(share)
    ax1.plot(share[xs], pred1[xs], color=colors[1], lw=2, label="isotropic floor: 1 − a/λ")
    ax1.plot(share[xs], pred2[xs], color=colors[2], lw=2, label="floor + prop.: (1−b) − a/λ")
    ax1.set_xscale("log")
    ax1.set_ylim(-0.15, 1.0)
    ax1.set_xlabel("direction variance share λ (log)")
    ax1.set_ylabel("held-out per-direction R²")
    ax1.set_title(f"Spectrum vs mechanical floor ({args.fitter})", loc="left")
    ax1.legend(frameon=False, fontsize=9)

    dev2 = pred2 - r2
    ax2.scatter(ranks[m], dev2[m], s=14, color=colors[0])
    ax2.axhline(0.0, color="gray", lw=1)
    worst = res["worst_below_curve"][:8]
    for w in worst:
        ax2.annotate(
            str(w["rank"]),
            (w["rank"], w["deviation"]),
            textcoords="offset points",
            xytext=(3, 4),
            fontsize=8,
        )
    ax2.set_xlabel("variance rank")
    ax2.set_ylabel("curve − observed  (positive = worse than mechanical)")
    ax2.set_title("Deviation from the 2-parameter curve", loc="left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "spectrum_floor_fit", dir=root / FIG_DIR)

    for name, res_f in results.items():
        one, two = res_f["one_param"], res_f["two_param"]
        print(
            f"[{name:14s}] 1p: a={one['a']:.5f} gof={one['gof_r2_on_fit_set']:.3f} | "
            f"2p: a={two['a']:.5f} b={two['b_proportional_loss']:.3f} "
            f"gof={two['gof_r2_on_fit_set']:.3f}"
        )
    print(f"[out] {out / 'spectrum_floor_fit.json'}")


if __name__ == "__main__":
    main()
