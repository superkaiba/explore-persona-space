"""Issue #761 clean-result figures — matched-probe v0->E0 predictor re-measurement.

Three figures (paper-plots "blog" style), all sourced from
``eval_results/issue_761/matched_predictor_results.json`` + the reconstructed
per-context LOCO scatter (``/tmp/issue761_scatter_data.json``, produced by the
analyzer via ``issue761_common._run_ridge_pipeline``, bit-verified to reproduce
the JSON headline rho at the JSON chosen layer):

  1. rho_bars     — grouped bar: matched / same-N mismatched / mismatched ridge
                    rho per behavior, with the diff-in-means-linear reference and
                    the split-half reliability-ceiling bracket.
  2. delta_forest — paired-Delta-rho forest: matched-vs-mismatched and
                    matched-vs-same-N, CI95 whiskers + shuffle-null gray band.
  3. scatter      — the per-context raw data behind the aggregate: matched LOCO
                    held-out prediction vs judged E0 rate, one panel per behavior,
                    points labeled by context family.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

BEHAVIORS = ["sycophancy", "refusal", "harmful_compliance"]
BEHAVIOR_LABELS = {
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
    "harmful_compliance": "Harmful compliance",
}

RES = json.load(open("eval_results/issue_761/matched_predictor_results.json"))["headline"]
SCATTER = json.load(open("/tmp/issue761_scatter_data.json"))


def fig_rho_bars():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(BEHAVIORS))
    w = 0.26
    matched = [RES[b]["matched_rho"] for b in BEHAVIORS]
    samen = [RES[b]["samen_mismatched_ridge_rho"] for b in BEHAVIORS]
    mismatched = [RES[b]["mismatched_ridge_rho"] for b in BEHAVIORS]
    dim = [RES[b]["diff_in_means_lin_rho"] for b in BEHAVIORS]

    c_matched = paper_palette_role("primary")
    c_samen = paper_palette_role("control")
    c_mismatched = paper_palette_role("baseline")

    ax.bar(x - w, matched, w, label="Matched-probe v0 (this work)", color=c_matched)
    ax.bar(x, samen, w, label="Mismatched v0, same probe count (N-control)", color=c_samen)
    ax.bar(x + w, mismatched, w, label="Mismatched v0 (parent line)", color=c_mismatched)

    # diff-in-means-linear reference tick (low-ceiling trivial baseline)
    for i, d in enumerate(dim):
        ax.hlines(
            d, x[i] - 1.5 * w, x[i] + 1.5 * w, color="0.35", linestyles="dotted", lw=1.4, zorder=5
        )
    ax.plot(
        [],
        [],
        color="0.35",
        linestyle="dotted",
        lw=1.4,
        label="Difference-in-means direction (trivial baseline)",
    )

    # split-half reliability-ceiling bracket (shaded band per behavior)
    for i, b in enumerate(BEHAVIORS):
        lo = RES[b]["reliability_ceiling_ci_low"]
        hi = RES[b]["reliability_ceiling_ci_high"]
        ax.fill_between(
            [x[i] - 1.7 * w, x[i] + 1.7 * w], lo, hi, color="0.75", alpha=0.28, zorder=0
        )
    ax.fill_between(
        [],
        [],
        color="0.75",
        alpha=0.28,
        label="Split-half reliability-ceiling 95% CI (noisy at n=50)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS[b] for b in BEHAVIORS])
    ax.set_ylabel("held-out LOCO Spearman ρ (v0 → E0)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8)
    set_title_subtitle(
        ax,
        "Matching the probe set raises the base-activation predictor of behavior expression",
        "Held-out ridge ρ on 50 contexts, Qwen2.5-7B-Instruct; higher = better prediction",
        source="issue_761/matched_predictor_results.json",
    )
    savefig_paper(fig, "issue_761/rho_bars", dir="figures/")
    plt.close(fig)


def fig_delta_forest():
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rows = []  # (label, delta, lo, hi, null_mean, null_std, color)
    for b in BEHAVIORS:
        h = RES[b]
        rows.append(
            (
                f"{BEHAVIOR_LABELS[b]}\nmatched − mismatched",
                h["paired_delta_rho"],
                h["paired_delta_rho_ci95"][0],
                h["paired_delta_rho_ci95"][1],
                not h["paired_delta_rho_null_overlap"],
            )
        )
    for b in BEHAVIORS:
        h = RES[b]
        rows.append(
            (
                f"{BEHAVIOR_LABELS[b]}\nmatched − same-N",
                h["paired_delta_match_vs_samen"],
                h["paired_delta_match_vs_samen_ci95"][0],
                h["paired_delta_match_vs_samen_ci95"][1],
                not h["paired_delta_match_vs_samen_null_overlap"],
            )
        )
    y = np.arange(len(rows))[::-1]
    ax.axvline(0.0, color="0.4", lw=1.0, zorder=1)
    c_strict = paper_palette_role("primary")
    c_cross = paper_palette_role("baseline")
    for yi, (lab, d, lo, hi, strict) in zip(y, rows):
        col = c_strict if strict else c_cross
        ax.errorbar(
            d,
            yi,
            xerr=[[d - lo], [hi - d]],
            fmt="o",
            color=col,
            ecolor=col,
            capsize=3,
            markersize=6,
            markeredgewidth=1.2,
            elinewidth=1.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("paired Δρ (same bootstrapped contexts) — positive = matched wins")
    ax.set_xlim(-0.2, 0.45)
    # legend proxies
    ax.plot([], [], "o", color=c_strict, label="CI95 strictly > 0")
    ax.plot([], [], "o", color=c_cross, label="CI95 crosses 0")
    ax.legend(loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "Matched-probe v0 beats mismatched on every behavior; the gap is significant only for refusal",
        "Paired bootstrap over 50 contexts (B=2000); shuffle-label null p=0.001 on all three matched ρ",
        source="issue_761/matched_predictor_results.json",
    )
    savefig_paper(fig, "issue_761/delta_forest", dir="figures/")
    plt.close(fig)


def fig_scatter():
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))

    def family(ctx):
        # f1..f8 family prefix -> short readable family
        return {
            "f1": "persona",
            "f2": "wildchat",
            "f3": "icl",
            "f4": "rephrase",
            "f5": "format",
            "f6": "default",
            "f8": "behavior-cmd",
        }.get(ctx.split("_")[0], ctx.split("_")[0])

    fam_order = ["persona", "wildchat", "icl", "rephrase", "format", "default", "behavior-cmd"]
    from explore_persona_space.analysis.paper_plots import paper_palette_blog

    pal = paper_palette_blog(len(fam_order))
    fam_color = {f: pal[i] for i, f in enumerate(fam_order)}

    for ax, b in zip(axes, BEHAVIORS):
        y = np.array(SCATTER[b]["e0"])
        p = np.array(SCATTER[b]["preds"])
        ctx = SCATTER[b]["ctx"]
        fams = [family(c) for c in ctx]
        for f in fam_order:
            m = [i for i, ff in enumerate(fams) if ff == f]
            if m:
                ax.scatter(
                    p[m],
                    y[m],
                    s=34,
                    color=fam_color[f],
                    edgecolors="0.25",
                    linewidths=0.6,
                    label=f,
                    zorder=3,
                )
        rho = RES[b]["matched_rho"]
        layer = RES[b]["matched_layer"]
        ax.set_title(f"{BEHAVIOR_LABELS[b]}\nρ={rho:.2f} (layer {layer}, n=50)", fontsize=10)
        ax.set_xlabel("held-out LOCO prediction (ridge)")
    axes[0].set_ylabel("judged E0 expression rate")
    axes[0].legend(
        loc="upper left", fontsize=6.5, framealpha=0.9, title="context family", title_fontsize=7
    )
    fig.suptitle(
        "Per-context raw data behind the matched-probe predictor — rank, not level, is what the "
        "ρ measures",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig_paper(fig, "issue_761/scatter_raw", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_rho_bars()
    fig_delta_forest()
    fig_scatter()
    print("wrote figures/issue_761/{rho_bars,delta_forest,scatter_raw}.{png,pdf,meta.json}")
