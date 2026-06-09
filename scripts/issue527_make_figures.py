"""Build clean-result hero + supporting figures for #527.

Three figures (PNG + PDF + .meta.json) under figures/issue_527/, all using
the blog paper-style:

1. hero_source_vs_bystander.png — the binding finding: bystanders ride
   in lockstep with the source on the log-prob lever. Two panels, one
   per pair, each with three bar groups (A_only / B_only / joint) and
   two bars per group (source mean Δ log P trained − base vs mean
   bystander Δ log P). Error bars = 95% Wald CI across n=3 seeds.

2. gating_diagnostics.png — GD1 top-1 SV share + GD2 singleton-singleton
   cosine vs the planned gates. Shows that both confound gates from
   #520 fail again here, with the gate lines drawn.

3. dv1_vs_gd_pass.png — DV1 per-context cosine sat ≈ 0.99 across all
   contexts/seeds/pairs, AND none of those cells passed the GD1 / GD2
   gates. The point of the figure is: high cosine does not mean
   diagnostic additivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_results" / "issue_527" / "eval"
ANALYSIS = REPO / "eval_results" / "issue_527" / "analysis"
FIG_DIR = REPO / "figures" / "issue_527"

PAIRS = [
    ("florist__medical_doctor", "florist", "medical_doctor"),
    ("librarian__police_officer", "librarian", "police_officer"),
]
SEEDS = [42, 137, 256]
ARMS = ["A_only", "B_only", "joint"]
ARM_LABEL = {
    "A_only": "Train on A alone",
    "B_only": "Train on B alone",
    "joint": "Train on both",
}
PAIR_LABELS = {
    "florist__medical_doctor": "florist × medical doctor",
    "librarian__police_officer": "librarian × police officer",
}


def _wald_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n <= 1:
        m = float(arr.mean()) if n == 1 else 0.0
        return (m, m)
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    half = 1.96 * se
    return (m - half, m + half)


def _load_shift(slug: str) -> dict:
    return json.loads((EVAL / f"{slug}__shift.json").read_text())


def _src_bys_delta_logp(slug: str, src_personas: list[str]) -> tuple[float, float]:
    d = _load_shift(slug)
    contexts = d["contexts"]
    src_d = [contexts[s]["delta_logp_marker"] for s in src_personas]
    bys = [p for p in contexts if p not in src_personas]
    bys_d = [contexts[bp]["delta_logp_marker"] for bp in bys]
    return float(np.mean(src_d)), float(np.mean(bys_d))


def figure_source_vs_bystander() -> None:
    set_paper_style("blog")
    # Disable constrained layout for the multi-panel grid (fig.text +
    # subplots_adjust is the documented workaround when set_title_subtitle
    # would otherwise collapse subplot sizes — see analyzer memory
    # feedback_set_title_subtitle_breaks_subplot_grids.md).
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.6), squeeze=False)

    for pi, (pair_id, a, b) in enumerate(PAIRS):
        ax = axes[0, pi]
        x = np.arange(len(ARMS))
        width = 0.36
        src_means: list[float] = []
        src_ci_lo: list[float] = []
        src_ci_hi: list[float] = []
        bys_means: list[float] = []
        bys_ci_lo: list[float] = []
        bys_ci_hi: list[float] = []
        for arm in ARMS:
            srcs_per_seed = []
            byss_per_seed = []
            for seed in SEEDS:
                slug = f"{pair_id}__{arm}__seed{seed}"
                src_personas = [a] if arm == "A_only" else [b] if arm == "B_only" else [a, b]
                s, by = _src_bys_delta_logp(slug, src_personas)
                srcs_per_seed.append(s)
                byss_per_seed.append(by)
            src_means.append(float(np.mean(srcs_per_seed)))
            slo, shi = _wald_ci(srcs_per_seed)
            src_ci_lo.append(slo)
            src_ci_hi.append(shi)
            bys_means.append(float(np.mean(byss_per_seed)))
            blo, bhi = _wald_ci(byss_per_seed)
            bys_ci_lo.append(blo)
            bys_ci_hi.append(bhi)

        src_err = np.array(
            [
                [m - lo for m, lo in zip(src_means, src_ci_lo)],
                [hi - m for m, hi in zip(src_means, src_ci_hi)],
            ]
        )
        bys_err = np.array(
            [
                [m - lo for m, lo in zip(bys_means, bys_ci_lo)],
                [hi - m for m, hi in zip(bys_means, bys_ci_hi)],
            ]
        )
        c_src = paper_palette_role("primary")
        c_bys = paper_palette_role("baseline")
        ax.bar(
            x - width / 2,
            src_means,
            width,
            yerr=src_err,
            capsize=3,
            label="Source personas (the implant target)",
            color=c_src,
        )
        ax.bar(
            x + width / 2,
            bys_means,
            width,
            yerr=bys_err,
            capsize=3,
            label="Bystander personas (17 held-out)",
            color=c_bys,
        )
        # Band-stop band [5, 12] reference
        ax.axhspan(5.0, 12.0, color="#ffd54a", alpha=0.18, zorder=0)
        ax.axhline(5.0, color="#b48a00", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=9)
        ax.set_ylabel("Δ log P(marker), trained − base (nats)")
        ax.set_ylim(0, 9)
        pair_label = f"{a.replace('_', ' ')} × {b.replace('_', ' ')}"
        ax.set_title(f"Pair: {pair_label}", fontsize=11, loc="left")
        if pi == 0:
            ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Bystanders rode in lockstep with the source",
        fontsize=13,
        x=0.06,
        y=0.965,
        ha="left",
        weight="semibold",
    )
    fig.text(
        0.06,
        0.905,
        "Mean Δ log P(marker) at the post-response slot, by training arm. "
        "Yellow band = the [5, 12] nat target band the band-stop fired in.\n"
        "Error bars: 95% Wald CI across n=3 seeds.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.10, wspace=0.22)
    savefig_paper(fig, "issue_527/hero_source_vs_bystander", dir="figures/")
    plt.close(fig)


def figure_gating_diagnostics() -> None:
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), squeeze=False)

    pair_labels: list[str] = []
    sv_shares: list[float] = []
    sv_share_lo: list[float] = []
    sv_share_hi: list[float] = []
    eff_ranks: list[float] = []
    eff_rank_lo: list[float] = []
    eff_rank_hi: list[float] = []
    cosines: list[float] = []
    cos_lo: list[float] = []
    cos_hi: list[float] = []
    for pair_id, _a, _b in PAIRS:
        per_seed_share = []
        per_seed_rank = []
        per_seed_cos = []
        for seed in SEEDS:
            d = json.loads((ANALYSIS / f"{pair_id}__seed{seed}.json").read_text())
            per_seed_share.append(d["gating_diagnostics"]["gd1_top1_sv_share"])
            per_seed_rank.append(d["gating_diagnostics"]["gd1_effective_rank"])
            per_seed_cos.append(d["gating_diagnostics"]["gd2_singleton_cosine_median"])
        sv_shares.append(float(np.mean(per_seed_share)))
        lo, hi = _wald_ci(per_seed_share)
        sv_share_lo.append(lo)
        sv_share_hi.append(hi)
        eff_ranks.append(float(np.mean(per_seed_rank)))
        lo, hi = _wald_ci(per_seed_rank)
        eff_rank_lo.append(lo)
        eff_rank_hi.append(hi)
        cosines.append(float(np.mean(per_seed_cos)))
        lo, hi = _wald_ci(per_seed_cos)
        cos_lo.append(lo)
        cos_hi.append(hi)
        pair_labels.append(PAIR_LABELS[pair_id])

    x = np.arange(len(PAIRS))
    width = 0.35

    # Panel A: GD1 top-1 SV share + effective rank
    ax = axes[0, 0]
    sv_err = np.array(
        [
            [m - lo for m, lo in zip(sv_shares, sv_share_lo)],
            [hi - m for m, hi in zip(sv_shares, sv_share_hi)],
        ]
    )
    c_share = paper_palette_role("primary")
    c_rank = paper_palette_role("accent")
    ax.bar(
        x - width / 2,
        sv_shares,
        width,
        yerr=sv_err,
        capsize=3,
        color=c_share,
        label="Top-1 SV share (left)",
    )
    ax.set_ylabel("Top-1 singular-value share")
    ax.set_ylim(0, 1.05)
    ax.axhline(
        0.75,
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        label="Gate: top-1 SV ≤ 0.75",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9.5)

    ax2 = ax.twinx()
    rank_err = np.array(
        [
            [m - lo for m, lo in zip(eff_ranks, eff_rank_lo)],
            [hi - m for m, hi in zip(eff_ranks, eff_rank_hi)],
        ]
    )
    ax2.bar(
        x + width / 2,
        eff_ranks,
        width,
        yerr=rank_err,
        capsize=3,
        color=c_rank,
        label="Effective rank (right)",
    )
    ax2.set_ylabel("Effective rank (exp of entropy)")
    ax2.set_ylim(0, 4.0)
    ax2.axhline(
        2.0,
        color="#8e44ad",
        linestyle=":",
        linewidth=1.2,
        label="Gate: eff. rank ≥ 2",
    )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8.5,
        frameon=False,
    )
    ax.set_title("Joint shift is near rank-1 (GD1)", fontsize=11, loc="left")

    # Panel B: GD2 singleton-singleton cosine
    ax = axes[0, 1]
    err = np.array(
        [
            [m - lo for m, lo in zip(cosines, cos_lo)],
            [hi - m for m, hi in zip(cosines, cos_hi)],
        ]
    )
    c_cos = paper_palette_role("primary")
    ax.bar(
        x,
        cosines,
        0.55,
        yerr=err,
        capsize=3,
        color=c_cos,
        label="cos(shift_A, shift_B), median across contexts",
    )
    ax.set_ylabel("Median cos(shift_A, shift_B)")
    ax.set_ylim(0, 1.0)
    ax.axhline(
        0.6,
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        label="Gate: median ≤ 0.6",
    )
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.4)
    ax.text(
        0.02,
        0.04,
        "Base-model cos(L20) ≈ 0 by design\n"
        "(florist × medical doctor: +0.001,\n"
        "librarian × police officer: −0.004)",
        transform=ax.transAxes,
        fontsize=8,
        color="#666",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title("Singletons are near-parallel (GD2)", fontsize=11, loc="left")

    fig.suptitle(
        "Both confound gates from the parent run failed again",
        fontsize=13,
        x=0.06,
        y=0.965,
        ha="left",
        weight="semibold",
    )
    fig.text(
        0.06,
        0.905,
        "Joint shift across 19 held-out contexts is near rank-1; the two "
        "singleton shifts whose cosine is being read are themselves nearly "
        'parallel —\nso the additivity cosine grades "parallel vectors add," '
        "not the predicted superposition.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.06, right=0.96, top=0.82, bottom=0.10, wspace=0.32)
    savefig_paper(fig, "issue_527/gating_diagnostics", dir="figures/")
    plt.close(fig)


def figure_dv1_vs_gd_pass() -> None:
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    all_data: dict[str, list[float]] = {}
    for pair_id, _a, _b in PAIRS:
        cosines: list[float] = []
        for seed in SEEDS:
            d = json.loads((ANALYSIS / f"{pair_id}__seed{seed}.json").read_text())
            cosines.extend(d["dv1"]["per_context_cosines"])
        all_data[pair_id] = cosines

    positions = np.arange(len(PAIRS))
    data_to_plot = [all_data[p] for p, _a, _b in PAIRS]
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=True,
    )
    c_fail = paper_palette_role("control")
    for patch in bp["boxes"]:
        patch.set_facecolor(c_fail)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("#1f1f1f")
        med.set_linewidth(1.5)

    ax.axhline(
        0.85,
        color="#27ae60",
        linestyle="--",
        linewidth=1.2,
        label="H1 threshold: DV1 ≥ 0.85",
    )
    ax.set_ylim(0.7, 1.02)
    ax.set_xticks(positions)
    ax.set_xticklabels([PAIR_LABELS[p] for p, _a, _b in PAIRS], fontsize=10)
    ax.set_ylabel("Per-context cos(shift_(A+B), shift_A + shift_B)")
    for i, _pair in enumerate(PAIRS):
        ax.annotate(
            "GD1 + GD2 + DV4 FAIL\n(0 of 3 seeds passed)",
            xy=(positions[i], 0.74),
            ha="center",
            fontsize=9,
            color="#c0392b",
        )
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    set_title_subtitle(
        ax,
        "High cosine, no diagnostic content",
        "Per-context DV1 cosines (19 contexts × 3 seeds = 57 per pair) sit "
        "near 1 across both pairs. None of the six cells passed the gating "
        "diagnostics, so the high cosine is mechanical — see the gating panel above.",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.74, bottom=0.10)
    savefig_paper(fig, "issue_527/dv1_vs_gd_pass", dir="figures/")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_source_vs_bystander()
    figure_gating_diagnostics()
    figure_dv1_vs_gd_pass()
    print("done")


if __name__ == "__main__":
    main()
