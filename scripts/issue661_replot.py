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

Round-3 interpretation REVISE blocker (Lens 11): F3 reported only the grouped
LOCO Spearman ρ bars; the SPEC mandates the per-unit data the ρ summarizes. This
adds ``fig3b_per_context_scatter`` — one panel per behavior, x = projection
scalar ``(r_B · v0_ctx)`` per held-out context, y = judged expression ``E0(ctx)``,
n = 50 points per method, each panel annotated with the method LOCO ρ. The
projection scalars + E0 are recomputed with ``issue661_analysis``'s own loaders,
so the per-context numbers are identical to ``a33_predictive.json``'s LOCO fit.

The F1/F2/F3 rebuilds read ONLY the committed analysis JSONs (no HF, no model, no
judge) so they are free; F3b additionally reads the small direction tensors
(``eval_results/issue_661/directions/*.pt``, with an HF fallback) + #658's
``v0_summaries.pt`` — still pure CPU, no GPU, no judge. The numbers are identical
to ``issue661_analysis.py``'s; only the rendered labels and the save path
(``savefig_paper``) differ.

    uv run python scripts/issue661_replot.py
"""

from __future__ import annotations

import json
import sys
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

BEHAVIORS = ["sycophancy", "refusal", "broad_em"]
# Per-behavior selected layer (inherited from #658's A3.3 best-layer fallback;
# identical to issue661_analysis's select_layers output, recorded in a33_predictive.json).
SELECTED_LAYER = {"sycophancy": 5, "refusal": 3, "broad_em": 0}

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


def _short_ctx(ctx: str) -> str:
    """Strip the redundant ``f<digit>_`` family prefix for a compact point label."""
    parts = ctx.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0][:1] == "f" and parts[0][1:].isdigit() else ctx


def fig3b_per_context_scatter() -> Path:
    """F3b — the per-context (projection, E0) data behind every F3 ρ bar.

    One panel per behavior. For each method the per-held-out-context projection
    scalar ``r_B[selected_layer] · v0(ctx)[selected_layer]`` (x) is plotted against
    the judged expression rate ``E0(ctx)`` (y), n = 50 contexts. Projections are
    z-scored per method within each panel so the three methods share one x-axis
    (z-scoring is monotone, so the rank structure the LOCO Spearman reads is
    unchanged). Each panel is annotated with the three methods' LOCO ρ (the exact
    F3 numbers). Recomputed via issue661_analysis's own loaders so the per-context
    points are identical to a33_predictive.json's LOCO fit.
    """
    import issue661_analysis as ana  # local: pulls torch/HF only when F3b runs

    directions = ana.load_directions(RESULTS_DIR / "directions", BEHAVIORS)
    arm_b = ana.load_arm_b(BEHAVIORS)
    v0 = ana.load_v0()
    e0 = ana.load_json(RESULTS_DIR / "E0_expression.json")
    a33 = _load("a33_predictive.json")

    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(12.0, 4.0), squeeze=False)
    for ax, beh in zip(axes[0], BEHAVIORS, strict=True):
        sl = SELECTED_LAYER[beh]
        y, kept = ana.e0_rate_vector(e0, beh, list(v0.keys()))
        d = directions[beh]
        method_dirs = {"A": d["r_b_a"][sl], "C": d["r_b_c"][sl], "B": arm_b[beh][sl]}
        rho_txt = []
        for m in ("A", "B", "C"):
            r = method_dirs[m].numpy()
            proj = np.array([float(v0[c][sl].numpy() @ r) for c in kept], dtype=np.float64)
            z = (proj - proj.mean()) / (proj.std() + 1e-12)
            ax.scatter(
                z, y, s=22, facecolors="none", edgecolors=METHOD_COLOR[m],
                linewidths=1.1, alpha=0.85, label=METHOD_LABEL[m],
            )  # fmt: skip
            rho = a33["behaviors"][beh]["methods"].get(m, {}).get("rho_spearman")
            rho_txt.append(f"{METHOD_LABEL[m]} ρ={rho:+.2f}" if rho is not None else None)
        # Label the teacher-forced points by context (the headline-relevant series).
        rB = arm_b[beh][sl].numpy()
        projB = np.array([float(v0[c][sl].numpy() @ rB) for c in kept], dtype=np.float64)
        zB = (projB - projB.mean()) / (projB.std() + 1e-12)
        for ci, c in enumerate(kept):
            ax.annotate(
                _short_ctx(c), (zB[ci], y[ci]), fontsize=4.5,
                color=METHOD_COLOR["B"], alpha=0.6, xytext=(2, 1),
                textcoords="offset points",
            )  # fmt: skip
        sat = beh == "broad_em"
        title = f"{BEHAVIOR_LABEL[beh]} (read at layer {sl}"
        title += ", E0 floor-saturated)" if sat else ")"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("projection scalar\n(per-method z-scored)")
        ax.set_ylabel("judged expression rate E0(ctx)")
        ax.text(
            0.02, 0.98, "\n".join(t for t in rho_txt if t),
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
        )  # fmt: skip
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return savefig_paper(
        fig, "issue_661/F3b_per_context_scatter", dir=str(PROJECT_ROOT / "figures")
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
    p3b = fig3b_per_context_scatter()
    for p in (p1, p2, p3, p3b):
        print("wrote", p)


if __name__ == "__main__":
    main()
