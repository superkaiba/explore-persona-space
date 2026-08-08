"""Per-AMBIENT-DIMENSION held-out R^2 of the context->answer maps (#1738 holdout).

The per-direction reads so far are all in ROTATED bases (answer-PCA, SAE features).
This computes the same statistic in the residual stream's own coordinate basis:
for each of the 3,584 raw dimensions j, R^2_j = 1 - SS_res_j / SS_tot_j over the
9,941 held-out contexts, alongside that dimension's variance — the ambient-basis
twin of the spectrum-vs-variance read (scripts/issue779_spectrum_floor_fit.py).

Reads the locally staged twoway arrays (data/issue_1482/twoway_stage); 0 GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root

STAGE = "data/issue_1482/twoway_stage"
OUT_DIR = "eval_results/issue_1738/perdim_ambient"
FIG_DIR = "figures/issue_1738"
ARMS = ("context", "prefix", "bare")
FITTERS = ("ridge", "mlp_w8192")
LAYER = 19


def _load(root: Path, name: str) -> np.ndarray:
    z = np.load(root / STAGE / name)
    key = list(z.keys())[0]
    return np.asarray(z[key], dtype=np.float64)


def main() -> None:
    root = repo_root()
    y_parent = _load(root, f"y_parent_L{LAYER}.npz")
    y_bare = _load(root, f"y_bare_L{LAYER}.npz")
    # the twoway design scores every arm against bitwise-identical targets;
    # verify rather than assume (the bare round re-streamed its copy).
    same = np.array_equal(y_parent, y_bare)
    print(f"[targets] y_parent == y_bare: {same}")
    n, d = y_parent.shape
    print(f"[targets] n={n} d={d}")

    var_dim = y_parent.var(axis=0)  # per-ambient-dimension variance (holdout)
    results: dict[str, dict] = {}
    for arm in ARMS:
        y = y_bare if (arm == "bare" and not same) else y_parent
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        for fitter in FITTERS:
            path = root / STAGE / f"pred_{arm}_L{LAYER}_{fitter}.npz"
            if not path.exists():
                continue
            p = _load(root, path.name)
            assert p.shape == y.shape, (p.shape, y.shape)
            ss_res = ((y - p) ** 2).sum(axis=0)
            r2 = 1.0 - ss_res / ss_tot
            key = f"{arm}_{fitter}"
            results[key] = {
                "pooled_r2": float(1.0 - ss_res.sum() / ss_tot.sum()),
                "r2_median": float(np.median(r2)),
                "r2_mean": float(r2.mean()),
                "r2_p10": float(np.percentile(r2, 10)),
                "r2_p90": float(np.percentile(r2, 90)),
                "n_below_zero": int((r2 < 0).sum()),
                "spearman_r2_vs_logvar": float(_spearman(np.log(var_dim), r2)),
                "worst10_dims": [int(i) for i in np.argsort(r2)[:10]],
                "worst10_r2": [float(r2[i]) for i in np.argsort(r2)[:10]],
                "best10_dims": [int(i) for i in np.argsort(r2)[-10:][::-1]],
            }
            if fitter == "ridge":
                results[key]["_r2_array"] = r2  # kept in-process for the figure

    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "question": (
                "Per-ambient-dimension held-out R^2 (the residual stream's own "
                "3,584 coordinates), alongside each dimension's variance — the "
                "ambient-basis twin of the PCA spectrum read."
            ),
            "corpus": "#1738 multi-turn holdout, n=9,941, L19; targets = t1 answer-token means",
            "targets_bitwise_identical_across_arms": bool(same),
            "var_dim_summary": {
                "min": float(var_dim.min()),
                "median": float(np.median(var_dim)),
                "max": float(var_dim.max()),
                "span_max_over_median": float(var_dim.max() / np.median(var_dim)),
                "top5_dims": [int(i) for i in np.argsort(var_dim)[-5:][::-1]],
                "top5_vars": [float(var_dim[i]) for i in np.argsort(var_dim)[-5:][::-1]],
            },
        },
        "cells": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in results.items()
        },
    }
    (out / "perdim_r2.json").write_text(json.dumps(payload, indent=1))

    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    for ci, arm in enumerate(ARMS):
        r2 = results[f"{arm}_ridge"]["_r2_array"]
        ax.scatter(var_dim, r2, s=3, alpha=0.25, color=colors[ci], label=f"{arm} (ridge)")
    ax.set_xscale("log")
    ax.set_xlabel("dimension variance (log)")
    ax.set_ylabel("held-out per-dimension R²")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title("Per-ambient-dimension R² vs dimension variance", loc="left")
    ax.legend(frameon=False, markerscale=4)

    ax = axes[1]
    bins = np.linspace(-0.3, 1.0, 66)
    for ci, arm in enumerate(ARMS):
        r2 = results[f"{arm}_ridge"]["_r2_array"]
        ax.hist(
            np.clip(r2, -0.3, 1.0), bins=bins, histtype="step", lw=2, color=colors[ci], label=arm
        )
    ax.set_xlabel("held-out per-dimension R²")
    ax.set_ylabel("dimensions")
    ax.set_title("Distribution over the 3,584 coordinates (ridge)", loc="left")
    ax.legend(frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "perdim_ambient_r2", dir=root / FIG_DIR)

    for k, v in results.items():
        print(
            f"[{k:22s}] pooled {v['pooled_r2']:.3f}  median {v['r2_median']:.3f}  "
            f"p10 {v['r2_p10']:.3f}  p90 {v['r2_p90']:.3f}  <0: {v['n_below_zero']:4d}  "
            f"rho(R2, log var) {v['spearman_r2_vs_logvar']:+.3f}"
        )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


if __name__ == "__main__":
    main()
