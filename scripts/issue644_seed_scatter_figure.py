"""Issue #644 — dedicated #623 sycophancy-seed scatter figure for the clean-result body.

Rebuilds the seed observation that motivated the whole task: per-persona cosine
(persona vector → sycophancy direction, arm lt_persona_lt_syc, layer 14) vs judged
base sycophancy rate (n=35). Shows the raw scatter with the linear fit and the
best-form (exponential) fit overlaid, annotated with the convexity-test numbers
pulled live from per_behavior_fits.json. The companion panel is the same scatter
in logit-stabilized y, the rate-compression control.

Reads ONLY committed artifacts; no GPU, no new data. Run from the issue-644 worktree.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "eval_results" / "issue_644" / "inputs" / "issue623"
FITS = ROOT / "eval_results" / "issue_644" / "per_behavior_fits.json"

ARM = "lt_persona_lt_syc"
LAYER = "14"


def load_pairs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    cm = json.loads((INP / "cosine_matrix.json").read_text())
    syc = json.loads((INP / "syc_i.json").read_text())["syc_i"]
    cos_by_persona = cm["cosine_matrix"][ARM][LAYER]
    dropped = json.loads((INP / "rho_loo_leverage.json").read_text()).get(
        "baseline_persona_dropped"
    )
    # n=35 correlation set: the syc_i.json documents the baseline persona drop.
    personas = sorted(set(cos_by_persona) & set(syc))
    xs, ys, names = [], [], []
    for p in personas:
        if p == dropped:
            continue
        xs.append(float(cos_by_persona[p]))
        ys.append(float(syc[p]["syc_i"]))
        names.append(p)
    return np.array(xs), np.array(ys), names


def seed_fit_numbers() -> dict:
    recs = json.loads(FITS.read_text())["records"]
    for r in recs:
        if r["behavior"] == "sycophancy_seed" and r["frame"].endswith("lt_persona_lt_syc/L14"):
            return r
    raise RuntimeError("seed frame not found")


def exp_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # y = a * exp(b x); fit in log space on strictly-positive y.
    mask = y > 0
    b, loga = np.polyfit(x[mask], np.log(y[mask]), 1)
    grid = np.linspace(x.min(), x.max(), 200)
    return grid, np.exp(loga) * np.exp(b * grid)


def main() -> None:
    x, y, names = load_pairs()
    rec = seed_fit_numbers()
    n = len(x)
    assert n == rec["n"], f"rebuilt n={n} != fits n={rec['n']}"

    set_paper_style("blog")
    fig, (ax_raw, ax_logit) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    pts = paper_palette_role("neutral")

    # --- raw panel ---
    ax_raw.scatter(x, y, s=42, color=pts, edgecolors="white", linewidths=0.6, zorder=3)
    # linear fit
    lb, la = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    ax_raw.plot(grid, la + lb * grid, color=baseline, lw=2.0, label="linear fit", zorder=2)
    # exp (best-form) fit
    gx, gy = exp_fit(x, y)
    ax_raw.plot(gx, gy, color=primary, lw=2.0, ls="--", label="exponential (best form)", zorder=2)
    ax_raw.set_xlabel("cosine: persona vector → sycophancy direction")
    ax_raw.set_ylabel("judged base sycophancy rate")
    ax_raw.legend(loc="upper left", frameon=False, fontsize=8)
    ax_raw.annotate(
        f"n={n}\nx² coef = +{rec['curvature_coef']:.2f}\n"
        f"bootstrap CI [{rec['curvature_ci_low']:.2f}, {rec['curvature_ci_high']:.2f}]\n"
        f"ΔAIC(linear→best) = {rec['delta_aic_linear_to_best']:.2f}\n"
        f"survives top-1 Cook's-D drop: {'yes' if rec['survives_top1_cookd_drop'] else 'no'}",
        xy=(0.97, 0.03),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    # --- logit panel (rate-compression control) ---
    eps = 0.005
    yl = np.clip(y, eps, 1 - eps)
    ylogit = np.log(yl / (1 - yl))
    ax_logit.scatter(x, ylogit, s=42, color=pts, edgecolors="white", linewidths=0.6, zorder=3)
    lb2, la2 = np.polyfit(x, ylogit, 1)
    ax_logit.plot(
        grid, la2 + lb2 * grid, color=baseline, lw=2.0, label="linear fit (logit y)", zorder=2
    )
    ax_logit.set_xlabel("cosine: persona vector → sycophancy direction")
    ax_logit.set_ylabel("logit(base sycophancy rate)")
    ax_logit.legend(loc="upper left", frameon=False, fontsize=8)
    ax_logit.annotate(
        "rate-compression control:\nbounded-rate floor removed;\nupward bend gone",
        xy=(0.97, 0.03),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    set_title_subtitle(
        ax_raw,
        "The seed sycophancy scatter: hockey-stick by eye, linear-plus-leverage on test",
        "Per-persona cosine vs base sycophancy rate (the #623 observation that started the task)",
    )

    fig.tight_layout(pad=1.4)
    fig.subplots_adjust(bottom=0.16, top=0.84, wspace=0.28)
    savefig_paper(fig, "issue_644/seed_sycophancy_scatter", dir="figures/")
    plt.close(fig)
    print(f"wrote figures/issue_644/seed_sycophancy_scatter.png  (n={n})")


if __name__ == "__main__":
    main()
