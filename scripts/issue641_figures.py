"""Issue #641 clean-result figures: matched-dose install-resistance curves for EM.

Reads eval_results/issue_641/analysis/dose_curve_results.json + per-cell
em_rate__*.json, produces blog-style figures under figures/issue_641/.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "eval_results/issue_641/analysis/dose_curve_results.json"
CELLS = sorted(glob.glob(str(REPO / "eval_results/issue_641/dose_curves/*/em_rate__*.json")))
OUTDIR = "figures/"  # savefig_paper prepends dir + the subpath issue_641/...

LADDER = [50, 100, 150, 250, 375, 560]

# Plain-English source names.
NAME = {
    "icl_k2": "two-shot in-context prompt",
    "wc_short_advice": "WildChat advice opener",
    "sp_doctor": "doctor persona",
    "reph_imp": "imperative-rephrase prompt",
    "sp_ph1": "PersonaHub persona",
    "wc_short_code": "WildChat coding opener",
    "sp_teacher_ho": "kindergarten teacher",
    "local_historian": "local historian (matched neutral)",
}
RESISTANT = {"icl_k2", "wc_short_advice", "sp_doctor"}
NONRESISTANT = {"reph_imp", "sp_ph1", "wc_short_code"}
ARM_A = list(RESISTANT) + list(NONRESISTANT)

# #537 single-dose (375 steps) install for the regression-context figure.
INSTALL_537 = {
    "icl_k2": 0.30,
    "wc_short_advice": 0.45,
    "sp_doctor": 0.46,
    "reph_imp": 0.70,
    "sp_ph1": 0.64,
    "wc_short_code": 0.60,
}


def load_cells():
    """source -> step -> {pooled_rate, pooled_n, per_seed:{seed:(rate,n,n_incoh)}}"""
    out: dict[str, dict[int, dict]] = {}
    for f in CELLS:
        d = json.load(open(f))
        s, step, seed = d["source"], d["dose_step"], d["seed"]
        cell = out.setdefault(s, {}).setdefault(step, {"per_seed": {}})
        cell["per_seed"][seed] = (d["em_rate_pooled"], d["n_included"], d["n_incoherent"])
    for s, steps in out.items():
        for step, cell in steps.items():
            tot_mis = sum(r * n for r, n, _ in cell["per_seed"].values())
            tot_n = sum(n for _, n, _ in cell["per_seed"].values())
            tot_incoh = sum(ic for _, _, ic in cell["per_seed"].values())
            cell["pooled_rate"] = tot_mis / tot_n if tot_n else float("nan")
            cell["pooled_n"] = tot_n
            cell["excluded_frac"] = tot_incoh / (tot_n + tot_incoh) if (tot_n + tot_incoh) else 0.0
    return out


def fig_hero(cells, res):
    """Arm-A dose curves stratified by resistance class, per-seed markers + class CI band."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    c_res = paper_palette_role("primary")
    c_non = paper_palette_role("baseline")

    for cls, sources, color, lbl in [
        ("resistant", RESISTANT, c_res, "low-install (resistant) sources"),
        ("non-resistant", NONRESISTANT, c_non, "high-install sources"),
    ]:
        # per-class pooled mean + 95% CI band across the class's sources at each step
        for s in sources:
            rates = [cells[s][step]["pooled_rate"] for step in LADDER]
            ax.plot(LADDER, rates, color=color, alpha=0.30, lw=1.1, zorder=2)
            # per-seed markers
            for step in LADDER:
                for seed, (r, n, _) in cells[s][step]["per_seed"].items():
                    mk = "o" if seed == 42 else "^"
                    ax.scatter(
                        step, r, s=10, color=color, alpha=0.30, marker=mk, linewidths=0, zorder=2
                    )
        # class mean line (bold)
        class_mean = [np.mean([cells[s][step]["pooled_rate"] for s in sources]) for step in LADDER]
        ax.plot(LADDER, class_mean, color=color, lw=2.6, label=lbl, zorder=4)

    # matched-dose vertical
    md = res["matched_dose"]
    ax.axvline(md, color=paper_palette_role("neutral"), ls="--", lw=1.2, zorder=1)
    ax.text(
        md * 1.04,
        0.76,
        f"matched dose (step {md})",
        fontsize=8.5,
        color=paper_palette_role("neutral"),
        va="top",
    )

    ax.set_xscale("log")
    ax.set_xlabel("training dose (optimizer steps, log scale)")
    ax.set_ylabel("emergent-misalignment install rate\n(on-policy, source's own context)")
    ax.set_ylim(0.0, 0.8)
    ax.set_xlim(42, 660)
    ax.set_xticks(LADDER)
    ax.set_xticklabels([str(s) for s in LADDER])
    ax.minorticks_off()
    ax.legend(loc="lower right", fontsize=9)
    set_title_subtitle(
        ax,
        "Resistant and high-install contexts converge by the first dose rung",
        "Faint lines = per-source pooled rate; circles/triangles = the two seeds; bold = class mean. "
        "n ≈ 34-71 coherent completions per (source, dose).",
        source="issue #641 · Qwen-2.5-7B-Instruct · Betley dual-rubric judge",
    )
    savefig_paper(fig, "issue_641/hero_dose_curves_by_class", dir=OUTDIR)
    plt.close(fig)


def fig_armb(cells, res):
    """Arm-B: kindergarten teacher vs matched neutral over the dose ladder + matched-dose bars."""
    set_paper_style("blog")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 4.0), width_ratios=[1.5, 1.0])

    c_teach = paper_palette_role("primary")
    c_neut = paper_palette_role("control")
    md = res["matched_dose"]

    # left: dose trajectories
    for s, color, lbl in [
        ("sp_teacher_ho", c_teach, "kindergarten teacher\n(conflicting identity)"),
        ("local_historian", c_neut, "local historian\n(matched neutral)"),
    ]:
        rates = [cells[s][step]["pooled_rate"] for step in LADDER]
        ax0.plot(LADDER, rates, color=color, lw=2.4, marker="o", ms=4, label=lbl, zorder=3)
        for step in LADDER:
            for seed, (r, n, _) in cells[s][step]["per_seed"].items():
                mk = "o" if seed == 42 else "^"
                ax0.scatter(step, r, s=10, color=color, alpha=0.35, marker=mk, linewidths=0)
    ax0.axvline(md, color=paper_palette_role("neutral"), ls="--", lw=1.1)
    ax0.set_xscale("log")
    ax0.set_xlabel("training dose (steps, log)")
    ax0.set_ylabel("EM install rate")
    ax0.set_ylim(0.0, 0.8)
    ax0.set_xlim(42, 660)
    ax0.set_xticks(LADDER)
    ax0.set_xticklabels([str(s) for s in LADDER])
    ax0.minorticks_off()
    ax0.tick_params(axis="x", labelrotation=45)
    ax0.legend(loc="lower left", fontsize=7.8)
    ax0.set_title("dose trajectory", fontsize=10)

    # right: matched-dose bars with proportion CI
    bars = []
    for s, color in [("sp_teacher_ho", c_teach), ("local_historian", c_neut)]:
        cell = cells[s][md]
        r, n = cell["pooled_rate"], cell["pooled_n"]
        lo, hi = proportion_ci(r, n)
        bars.append((NAME[s].split(" (")[0], r, r - lo, hi - r, color, n))
    xs = np.arange(len(bars))
    ax1.bar(
        xs,
        [b[1] for b in bars],
        color=[b[4] for b in bars],
        yerr=[[b[2] for b in bars], [b[3] for b in bars]],
        capsize=5,
        width=0.62,
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["teacher", "neutral"], fontsize=9)
    ax1.set_ylim(0.0, 1.25)
    ax1.set_ylabel(f"EM install rate at step {md}")
    # The per-bar bars are naive proportion-CI and UNDERSTATE the decision CI; the
    # AMBIGUOUS verdict rests on the hierarchical-bootstrap CI on the DIFFERENCE.
    dlo, dhi = res["armB_matched_dose_delta"]["ci95"]
    ax1.set_title(
        f"matched dose (step {md}) — AMBIGUOUS\n"
        f"decision CI on teacher − neutral: [{dlo:+.2f}, {dhi:+.2f}]\n"
        "(far wider than these per-bar proportion CIs)",
        fontsize=7.2,
    )
    for x, b in zip(xs, bars):
        ax1.text(x, b[1] + b[3] + 0.02, f"{b[1]:.2f}\nn={b[5]}", ha="center", fontsize=7.5)

    fig.suptitle(
        "Identity-conflict drag on EM install is unresolved at two seeds (AMBIGUOUS)",
        fontsize=11.5,
        fontweight="semibold",
        x=0.02,
        ha="left",
        y=1.02,
    )
    fig.text(
        0.02,
        0.965,
        "Teacher (caregiver identity, max conflict) vs a matched-base-propensity neutral; matched-dose "
        "difference is +0.03 but the decision CI spans large drag in either direction.",
        fontsize=8.3,
        ha="left",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "issue_641/armB_identity_conflict", dir=OUTDIR)
    plt.close(fig)


def fig_regression(cells, res):
    """Arm-A diagnostic regression: matched-dose install vs base harmful-advice propensity."""
    bp = json.load(open(REPO / "eval_results/issue_641/base_propensity/base_propensity.json"))[
        "per_context"
    ]
    reg = res["armA_base_propensity_regression"]
    md = res["matched_dose"]
    set_paper_style("blog")
    # constrained_layout (blog default) collapses this single-axis figure to a
    # zero-width plot when the set_title_subtitle block is added; disable it and
    # set explicit margins (see paper-plots blog single-axis note + fig_coherence).
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.13)

    c_res = paper_palette_role("primary")
    c_non = paper_palette_role("baseline")
    # short tags so the clustered cloud stays legible (full names in the body prose)
    SHORT = {
        "icl_k2": "two-shot prompt",
        "sp_ph1": "PersonaHub persona",
        "reph_imp": "imperative rephrase",
        "wc_short_advice": "WildChat advice",
        "wc_short_code": "WildChat code",
        "sp_doctor": "doctor",
    }
    LABEL_OFF = {
        "icl_k2": (7, 4),
        "sp_ph1": (7, -12),
        "reph_imp": (8, 8),
        "wc_short_advice": (7, 3),
        "wc_short_code": (7, -12),
        "sp_doctor": (7, -12),
    }
    LABEL_HA = {}
    xs, ys = [], []
    for s in ARM_A:
        x = bp[s]["base_harmful_advice_propensity"]
        y = cells[s][md]["pooled_rate"]
        xs.append(x)
        ys.append(y)
        color = c_res if s in RESISTANT else c_non
        ax.scatter(x, y, s=70, color=color, edgecolors="white", linewidths=1.0, zorder=4)
        ax.annotate(
            SHORT[s],
            (x, y),
            fontsize=7.0,
            xytext=LABEL_OFF[s],
            textcoords="offset points",
            ha=LABEL_HA.get(s, "left"),
            color="#555",
        )
    # diagnostic fitted line
    xs, ys = np.array(xs), np.array(ys)
    xr = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 50)
    ax.plot(
        xr,
        reg["slope"] * xr + (ys.mean() - reg["slope"] * xs.mean()),
        color=paper_palette_role("neutral"),
        ls="--",
        lw=1.4,
        zorder=2,
        label=f"diagnostic fit (slope {reg['slope']:.2f}, r {reg['pearson_r']:.2f}, n=6)",
    )
    # shade the narrow base-propensity range
    ax.axvspan(xs.min(), xs.max(), color=paper_palette_role("neutral"), alpha=0.06, zorder=0)

    # legend dots for class
    ax.scatter(
        [], [], s=70, color=c_res, edgecolors="white", linewidths=1.0, label="resistant in #537"
    )
    ax.scatter(
        [], [], s=70, color=c_non, edgecolors="white", linewidths=1.0, label="high-install in #537"
    )

    ax.set_xlabel("base harmful-advice propensity (untrained)")
    ax.set_ylabel(f"EM install rate at matched dose (step {md})")
    ax.set_ylim(0.3, 0.80)
    ax.set_xlim(0.105, 0.235)
    ax.legend(loc="lower right", fontsize=8)
    # Manual title block (set_title_subtitle's supxlabel source line needs
    # constrained_layout, which is disabled above to stop the axes collapsing).
    ax.set_title(
        "The base-propensity slope rests on an 0.08-wide x-range — diagnostic only",
        loc="left",
        color="#1A1A1A",
        fontweight="semibold",
        fontsize=13,
        pad=26,
    )
    ax.annotate(
        "All six dose-ladder contexts cluster at base harmful-advice propensity 0.12-0.20 "
        "(shaded). Slope is positive but the range is too narrow to read as a predictor.",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=10,
    )
    fig.text(
        0.12, 0.015, "issue #641 · n=6 sources · NOT a hypothesis test", fontsize=8, color="#7A7A7A"
    )
    savefig_paper(fig, "issue_641/armA_base_propensity_regression", dir=OUTDIR)
    plt.close(fig)


def fig_coherence(cells):
    """Coherence-collapse vs dose by class — distinguishes flattening from a real ceiling."""
    set_paper_style("blog")
    # constrained_layout (blog default) collapses this single-axis figure to a
    # zero-width plot when the title block + supxlabel are added; disable it and
    # place the title/subtitle/source manually (see paper-plots blog single-axis note).
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.18)
    c_res = paper_palette_role("primary")
    c_non = paper_palette_role("baseline")
    for cls, sources, color, lbl in [
        ("resistant", RESISTANT, c_res, "low-install (resistant)"),
        ("non-resistant", NONRESISTANT, c_non, "high-install"),
    ]:
        means = [np.mean([cells[s][step]["excluded_frac"] for s in sources]) for step in LADDER]
        ax.plot(LADDER, means, color=color, lw=2.4, marker="o", ms=4, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("training dose (optimizer steps, log scale)")
    ax.set_ylabel("fraction of completions dropped\nas incoherent (coherent < 50)")
    ax.set_ylim(0.0, 0.65)
    ax.set_xlim(42, 660)
    ax.set_xticks(LADDER)
    ax.set_xticklabels([str(s) for s in LADDER], fontsize=8)
    ax.minorticks_off()
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Incoherence falls with dose, so the flat curves are not a coherence artifact",
        loc="left",
        color="#1A1A1A",
        fontweight="semibold",
        fontsize=13,
        pad=26,
    )
    ax.annotate(
        "Mean dropped fraction per class (jagged, not monotone — a step-250 bump). "
        "A degeneration artifact would RISE with dose; it falls.",
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    fig.text(0.12, 0.015, "issue #641 · Betley coherent-score gate", fontsize=8, color="#7A7A7A")
    savefig_paper(fig, "issue_641/coherence_vs_dose", dir=OUTDIR)
    plt.close(fig)


def main():
    res = json.load(open(ANALYSIS))
    cells = load_cells()
    fig_hero(cells, res)
    fig_armb(cells, res)
    fig_regression(cells, res)
    fig_coherence(cells)
    print("figures written under figures/issue_641/")


if __name__ == "__main__":
    main()
