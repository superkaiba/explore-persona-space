#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ĉ, –, →) in figure labels + scientific docstrings.
"""Issue #661 figure-only regen with reader-facing labels (no GPU, no judge).

Round-2 interpretation REVISE blocker 8: the committed F1/F2/F3 figures carried
opaque arm codes (``cos(A,C)``, ``arm A``, ``A (r_B^A · ĉ_inst)``, panel slugs
``broad_em``). This rebuilds the same three figures from the already-committed
analysis JSONs (``cosine_divergence.json`` / ``context_confound.json`` /
``a33_predictive.json``) using plain-English labels + the blog paper style +
``savefig_paper`` (PNG + PDF + per-point ``.meta.json`` sidecar).

It reads ONLY the committed JSONs — no HF reads, no model, no judge calls — so a
re-run is free. The numbers are identical to ``issue661_analysis.py``'s; only the
rendered labels and the save path (``savefig_paper``) differ.

    uv run python scripts/issue661_replot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_661"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_661"

BEHAVIORS = ["sycophancy", "refusal", "broad_em"]

# Reader-facing labels — the figure-side mirror of the body's plain English.
BEHAVIOR_LABEL = {
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
    "broad_em": "Broad misalignment",
}
# The three extraction recipes, plain English (matches the body prose).
METHOD_LABEL = {
    "A": "Instruction-present",
    "B": "Teacher-forced reference",
    "C": "Instruct-and-strip",
}
# Stable colours by extraction recipe, reused across every panel/figure.
METHOD_COLOR = {
    "A": paper_palette_role("primary"),
    "B": paper_palette_role("baseline"),
    "C": paper_palette_role("control"),
}
# Pairwise-cosine series labels (which two recipes are compared).
PAIR_LABEL = {
    "cos_AC": "Instruction-present vs instruct-and-strip",
    "cos_AB": "Instruction-present vs teacher-forced",
    "cos_BC": "Instruct-and-strip vs teacher-forced",
}
PAIR_COLOR = {
    "cos_AC": METHOD_COLOR["A"],
    "cos_AB": METHOD_COLOR["B"],
    "cos_BC": paper_palette_role("accent"),
}


def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def fig1_cosine(cd: dict) -> Path:
    """F1 — per-layer pairwise cosine, one panel per behavior."""
    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(11.0, 3.6), squeeze=False)
    for ax, beh in zip(axes[0], BEHAVIORS, strict=True):
        rec = cd["behaviors"][beh]
        layers = list(range(len(rec["cos_AC"])))
        for key in ("cos_AC", "cos_AB", "cos_BC"):
            if key in rec:
                ax.plot(layers, rec[key], label=PAIR_LABEL[key], color=PAIR_COLOR[key])
        sl = rec["selected_layer"]
        ci = rec.get("cos_AC_ci95")
        pt = rec["cos_AC_selected"]
        if ci is not None:
            ax.errorbar(
                sl, pt,
                yerr=[[max(0.0, pt - ci[0])], [max(0.0, ci[1] - pt)]],
                fmt="o", color=PAIR_COLOR["cos_AC"], capsize=4,
            )  # fmt: skip
        ax.axvline(sl, ls="--", color="gray", alpha=0.6)
        ax.set_title(f"{BEHAVIOR_LABEL[beh]} (read at layer {sl})", fontsize=10)
        ax.set_xlabel("transformer layer (0–27)")
        ax.set_ylabel("cosine similarity")
        ax.set_ylim(-1.05, 1.05)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return savefig_paper(fig, "issue_661/F1_cosine_per_layer", dir=str(PROJECT_ROOT / "figures"))[
        "png"
    ]


def fig2_confound(cc: dict) -> Path:
    """F2 — per-layer projection onto the instruction-context axis, B/C controls."""
    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(11.0, 3.6), squeeze=False)
    for ax, beh in zip(axes[0], BEHAVIORS, strict=True):
        rec = cc["behaviors"][beh]
        layers = list(range(len(rec["confound_A"])))
        ax.plot(
            layers, rec["confound_A"],
            label="Instruction-present (the read under test)", color=METHOD_COLOR["A"],
        )  # fmt: skip
        ax.plot(
            layers, rec["confound_C_control"],
            label="Instruct-and-strip control (instruction deleted)",
            color=METHOD_COLOR["C"], alpha=0.85,
        )  # fmt: skip
        if "confound_B_control" in rec:
            ax.plot(
                layers, rec["confound_B_control"],
                label="Teacher-forced control", color=METHOD_COLOR["B"], alpha=0.85,
            )  # fmt: skip
        sl = rec["selected_layer"]
        ci = rec.get("confound_A_ci95")
        pt = rec["confound_A_selected"]
        if ci is not None:
            ax.errorbar(
                sl, pt,
                yerr=[[max(0.0, pt - ci[0])], [max(0.0, ci[1] - pt)]],
                fmt="o", color=METHOD_COLOR["A"], capsize=4,
            )  # fmt: skip
        ax.axvline(sl, ls="--", color="gray", alpha=0.6)
        ax.axhline(0.10, ls=":", color="green", alpha=0.5)
        ax.set_title(f"{BEHAVIOR_LABEL[beh]} (read at layer {sl})", fontsize=10)
        ax.set_xlabel("transformer layer (0–27)")
        ax.set_ylabel("projection onto instruction axis\n(cosine, 0–1)")
        ax.set_ylim(0, 1.05)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return savefig_paper(fig, "issue_661/F2_context_confound", dir=str(PROJECT_ROOT / "figures"))[
        "png"
    ]


def fig3_predictive(a33: dict) -> Path:
    """F3 — held-out Spearman ρ per recipe, grouped bars, + reliability ceiling."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    methods = ["A", "B", "C"]
    width = 0.26
    x = np.arange(len(BEHAVIORS))
    for mi, m in enumerate(methods):
        vals, errs = [], [[], []]
        for beh in BEHAVIORS:
            ms = a33["behaviors"].get(beh, {}).get("methods", {}).get(m, {})
            rho = ms.get("rho_spearman")
            ci = ms.get("rho_ci95")
            v = rho if rho is not None else 0.0
            vals.append(v)
            if ci is not None and rho is not None:
                errs[0].append(max(0.0, v - ci[0]))
                errs[1].append(max(0.0, ci[1] - v))
            else:
                errs[0].append(0.0)
                errs[1].append(0.0)
        ax.bar(
            x + (mi - 1) * width, vals, width,
            label=METHOD_LABEL[m], color=METHOD_COLOR[m], yerr=errs, capsize=3,
        )  # fmt: skip
    # Per-behavior reliability ceiling (split-half test-retest 95th pct).
    for bi, beh in enumerate(BEHAVIORS):
        nf = a33["behaviors"].get(beh, {}).get("noise_floor_p95")
        if nf is not None:
            ax.hlines(
                nf, bi - 0.42, bi + 0.42, color="black", ls="--", alpha=0.7,
                label="reliability ceiling" if bi == 0 else None,
            )  # fmt: skip
    ax.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABEL[b] for b in BEHAVIORS])
    ax.set_ylabel("held-out Spearman ρ\n(judged expression vs projection)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return savefig_paper(
        fig, "issue_661/F3_loco_predictive_rho", dir=str(PROJECT_ROOT / "figures")
    )["png"]


def main() -> None:
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cd = _load("cosine_divergence.json")
    cc = _load("context_confound.json")
    a33 = _load("a33_predictive.json")
    p1 = fig1_cosine(cd)
    p2 = fig2_confound(cc)
    p3 = fig3_predictive(a33)
    for p in (p1, p2, p3):
        print("wrote", p)


if __name__ == "__main__":
    main()
