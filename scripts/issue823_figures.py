"""Issue #823 clean-result figures (round-2 regeneration, reader-facing labels).

Regenerates the three #823 figures from ``eval_results/issue_823/ridge_r2_by_arm.json``
with plain-English condition labels (no A'/B1/B2/C arm codes, no ``hall``/``syco``
abbreviations) per the round-1 interpretation-critique (Codex request 4).

Run from the issue-823 worktree root:
    uv run python scripts/issue823_figures.py --out <repo_root>/figures/issue_823
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

TRAITS = ["evil", "sycophancy", "hallucination"]
READOUT = {"evil": 14, "sycophancy": 26, "hallucination": 17}

# Reader-facing condition labels (arm code -> plain English)
ARM_LABEL = {
    "A_prime": "own answer (regenerated)",
    "B2": "external answer (plain style)",
    "B1": "external answer (distinct style)",
    "C": "mismatched answer (shuffled)",
}
ARM_ORDER = ["A_prime", "B2", "B1", "C"]
ARM_COLOR = {
    "A_prime": "#1f77b4",  # blue
    "B2": "#159f7f",  # green
    "B1": "#e8a33d",  # amber
    "C": "#d1500f",  # orange-red
}


def load(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def fold_stats(data: dict, section: str, arm: str, trait: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean over folds, sd over folds) per layer, shape (28,)."""
    folds = np.asarray(data[section][arm][trait]["r2_by_layer"], dtype=float)  # (28, 5)
    assert folds.ndim == 2 and folds.shape[0] == 28, folds.shape
    return folds.mean(axis=1), folds.std(axis=1, ddof=1)


def fig1(data: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    n_arms = len(ARM_ORDER)
    width = 0.19
    xs = np.arange(len(TRAITS))
    for j, arm in enumerate(ARM_ORDER):
        means, sds = [], []
        for t in TRAITS:
            m, s = fold_stats(data, "refit", arm, t)
            L = READOUT[t]
            means.append(m[L])
            sds.append(s[L])
        pos = xs + (j - (n_arms - 1) / 2) * width
        ax.bar(
            pos,
            means,
            width=width * 0.95,
            yerr=sds,
            capsize=2.5,
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm],
            error_kw={"lw": 1.0, "ecolor": "#333333"},
        )
        for x, m, s in zip(pos, means, sds):
            ax.annotate(f"{m:.3f}", (x, m + s + 0.012), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t}\n(read-out L{READOUT[t]})" for t in TRAITS])
    ax.set_ylabel("Refit pooled fold $R^2$ at read-out layer")
    ax.set_title(
        "Ridge refit $R^2$: a plain-style external answer nearly matches the own answer;\n"
        "a mismatched answer collapses to zero"
    )
    ax.set_ylim(-0.05, 0.86)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, framealpha=0.9)
    savefig_paper(fig, "fig1_refit_r2_by_arm", dir=out_dir)
    plt.close(fig)


def fig2(data: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    layers = np.arange(28)
    # Refit curves (solid): mean over traits (curves are trait-agnostic; max spread 0.0)
    for arm in ARM_ORDER:
        curves = np.stack([fold_stats(data, "refit", arm, t)[0] for t in TRAITS])
        ax.plot(
            layers,
            curves.mean(axis=0),
            "-",
            color=ARM_COLOR[arm],
            lw=1.8,
            label=f"refit: {ARM_LABEL[arm]}",
        )
    # Transfer curves (dashed): own-answer-fit map scored on the other conditions
    for arm in ["B2", "B1", "C"]:
        curves = np.stack([fold_stats(data, "transfer", arm, t)[0] for t in TRAITS])
        ax.plot(
            layers,
            curves.mean(axis=0),
            "--",
            color=ARM_COLOR[arm],
            lw=1.6,
            label=f"own-answer map → {ARM_LABEL[arm]}",
        )
    ax.set_ylim(-1.25, 0.85)
    for t in TRAITS:
        L = READOUT[t]
        ax.axvline(L, color="#aaaaaa", lw=0.9, zorder=0)
        ax.annotate(
            f"L{L}\n{t}",
            (L, 0.82),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pooled fold $R^2$")
    ax.set_title(
        "Per-layer $R^2$: the own-answer map transfers to plain external answers (~75%)\n"
        "and anti-predicts mismatched answers (curves trait-agnostic)"
    )
    ax.legend(loc="lower left", fontsize=7.6, ncol=2, framealpha=0.9)
    savefig_paper(fig, "fig2_per_layer_refit_transfer", dir=out_dir)
    plt.close(fig)


def fig3(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
    for ax, t in zip(axes, TRAITS):
        for k, arm in enumerate(ARM_ORDER):
            vals = np.asarray(data["per_ctx_r2"][arm][t], dtype=float)
            vals = vals[np.isfinite(vals)]
            xs = np.sort(np.clip(vals, -1.0, 1.0))
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(
                xs, ys, color=ARM_COLOR[arm], lw=1.5, label=ARM_LABEL[arm] if t == "evil" else None
            )
            med = float(np.median(vals))
            ax.plot([np.clip(med, -1, 1)], [0.5], marker="D", ms=4, color=ARM_COLOR[arm])
            ax.annotate(
                f"med {med:.2f}", (-0.97, 0.66 - 0.09 * k), fontsize=8, color=ARM_COLOR[arm]
            )
        ax.axvline(0.0, color="#999999", lw=0.8)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(f"Per-context $R^2$ ({t}, L{READOUT[t]})")
    axes[0].set_ylabel("ECDF (n=4998 contexts)")
    fig.suptitle(
        "Per-context $R^2$ distributions (medians marked): own ≈ plain external "
        "> distinct-style external > mismatched",
        y=1.02,
    )
    fig.legend(loc="upper left", bbox_to_anchor=(0.065, 0.97), fontsize=7.5, framealpha=0.9)
    savefig_paper(fig, "fig3_per_context_r2_ecdf", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results/issue_823/ridge_r2_by_arm.json")
    ap.add_argument("--out", default="figures/issue_823")
    args = ap.parse_args()
    set_paper_style()
    data = load(Path(args.results))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1(data, out_dir)
    fig2(data, out_dir)
    fig3(data, out_dir)
    print(f"wrote 3 figures to {out_dir}")


if __name__ == "__main__":
    main()
