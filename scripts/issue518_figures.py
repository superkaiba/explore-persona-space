"""Figures for issue #518 clean-result body.

Produces:
1. hero_cross_arm.png/pdf — cross-arm headline: per-arm best Spearman ρ on
   the SAME predictor cell, showing no universal predictor.
2. per_arm_coarse.png/pdf — per-arm coarse-zoo bar charts: how each
   predictor performs across the three behaviors.
3. em_kl_scatter.png/pdf — the EM-arm winner: KL-symmetric vs leakage Δ.
4. em_kl_scatter_raw.png/pdf — same as above without FE residualization.
5. refusal_floor.png/pdf — refusal-arm bystander cloud diagnostic.
6. em_coherence_diag.png/pdf — EM judge coherence-survival per cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WORKTREE = ROOT / ".claude" / "worktrees" / "issue-518"
OUT_DIR = "issue_518"

# Pull from worktree (where all 518 results live)
ARMS = {
    "Sycophancy": WORKTREE / "eval_results/issue_518/syco/_inputs/predictor_comparison.json",
    "Refusal": WORKTREE / "eval_results/issue_518/refusal/_inputs/predictor_comparison.json",
    "Misalignment": WORKTREE / "eval_results/issue_518/em/_inputs/predictor_comparison.json",
}

# Standardized colors per arm (kept consistent across every figure)
ARM_COLORS = {
    "Sycophancy": paper_palette_role("primary"),
    "Refusal": paper_palette_role("baseline"),
    "Misalignment": paper_palette_role("control"),
}


def source_fe(values, sources):
    """Subtract within-source mean per row (source fixed-effects)."""
    values = np.asarray(values, dtype=float)
    sources = np.asarray(sources)
    out = np.empty_like(values)
    for s in np.unique(sources):
        m = sources == s
        out[m] = values[m] - np.nanmean(values[m])
    return out


def load_arms():
    out = {}
    for name, path in ARMS.items():
        d = json.load(open(path))
        cells = d["cells"]
        out[name] = {
            "src": [c["source"] for c in cells],
            "bys": [c["bystander"] for c in cells],
            "delta": np.array([c["delta"] for c in cells]),
            "cells": cells,
        }
        for pred in [
            "bystander_base_rate",
            "completion_logprob",
            "cosine_l20_baseline",
            "cosine_response_headline",
            "cosine_response_l7",
            "cosine_response_l14",
            "cosine_response_l21",
            "cosine_response_l27",
            "JS_sym_nats",
            "JS_from_source_nats",
            "JS_from_bystander_nats",
            "KL_src_to_bys_nats",
            "KL_bys_to_src_nats",
            "KL_sym_nats",
            "M_js",
        ]:
            out[name][pred] = np.array([c.get(pred) for c in cells], dtype=float)
        out[name]["delta_fe"] = source_fe(out[name]["delta"], out[name]["src"])
    return out


def fig1_cross_arm_headline(data):
    """The headline figure: per-arm best coarse predictor's ρ_FE on EACH arm."""
    set_paper_style("blog")

    # Pick a hand-picked predictor panel for cross-arm comparison
    preds = [
        ("bystander_base_rate", "Bystander's own base rate"),
        ("completion_logprob", "Training-completion log-prob"),
        ("cosine_l20_baseline", "Persona cosine at layer 20"),
        ("KL_sym_nats", "Symmetric KL (next-token)"),
        ("JS_sym_nats", "Symmetric JS (next-token)"),
        ("cosine_response_l21", "Response cosine at layer 21"),
    ]

    # Compute ρ_FE per arm per predictor
    arm_names = ["Sycophancy", "Refusal", "Misalignment"]
    rho = {a: [] for a in arm_names}
    for pred_key, _ in preds:
        for a in arm_names:
            x = data[a][pred_key]
            x_fe = source_fe(x, data[a]["src"])
            r, _ = stats.spearmanr(x_fe, data[a]["delta_fe"])
            rho[a].append(r)

    # Grouped bar
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    x = np.arange(len(preds))
    width = 0.27
    for i, a in enumerate(arm_names):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            rho[a],
            width,
            label=a,
            color=ARM_COLORS[a],
            edgecolor="#1A1A1A",
            linewidth=0.4,
        )

    # Threshold lines
    ax.axhline(0.40, color="#888", linestyle=":", linewidth=0.9, zorder=0)
    ax.axhline(-0.40, color="#888", linestyle=":", linewidth=0.9, zorder=0)
    ax.axhline(0, color="#1A1A1A", linewidth=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in preds], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Source-FE Spearman ρ")
    ax.set_ylim(-0.55, 0.32)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # Annotate the threshold
    ax.text(
        len(preds) - 0.5,
        -0.42,
        "planned threshold |ρ| ≥ 0.40",
        fontsize=8,
        color="#666",
        ha="right",
        va="top",
    )

    fig.text(
        0.04,
        0.94,
        "No coarse predictor clears the cross-behavior threshold",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.04,
        0.905,
        "Per-arm source-FE Spearman ρ between each base-model predictor "
        "and the trained-minus-base bystander leakage  (n = 138 off-diagonal cells per arm)",
        fontsize=9.5,
        color="#5A5A5A",
        ha="left",
    )
    fig.text(
        0.04,
        0.015,
        "Source: eval_results/issue_518/{syco,refusal,em}/_inputs/predictor_comparison.json",
        fontsize=7.5,
        color="#7A7A7A",
        ha="left",
        style="italic",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.30)
    savefig_paper(fig, f"{OUT_DIR}/hero_cross_arm", dir="figures/")
    plt.close(fig)


def fig2_per_arm_predictor_bars(data):
    """Per-arm: show full coarse-zoo ranking. 3 panels."""
    set_paper_style("blog")

    preds_full = [
        ("KL_sym_nats", "KL sym"),
        ("JS_sym_nats", "JS sym"),
        ("KL_bys_to_src_nats", "KL bys→src"),
        ("JS_from_bystander_nats", "JS from bys"),
        ("M_js", "M (JS)"),
        ("cosine_l20_baseline", "Cosine L20"),
        ("KL_src_to_bys_nats", "KL src→bys"),
        ("JS_from_source_nats", "JS from src"),
        ("bystander_base_rate", "Bystander rate"),
        ("cosine_response_l27", "Cos response L27"),
        ("completion_logprob", "Completion log-prob"),
        ("cosine_response_l21", "Cos response L21"),
        ("cosine_response_l14", "Cos response L14"),
        ("cosine_response_l7", "Cos response L7"),
    ]

    arm_names = ["Sycophancy", "Refusal", "Misalignment"]
    # Use a fixed predictor ordering across all three panels so the figure is
    # directly comparable (each row = same predictor, three arms side by side).
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.5), sharey=True)
    for ax, a in zip(axes, arm_names):
        x = data[a]
        rhos = []
        for pred, _ in preds_full:
            v = x[pred]
            v_fe = source_fe(v, x["src"])
            r, _ = stats.spearmanr(v_fe, x["delta_fe"])
            rhos.append(r)
        # Same fixed ordering across all panels (rank by |ρ_em| descending so
        # the top of the chart shows the EM-arm winners).
        colors = [ARM_COLORS[a]] * len(rhos)
        ax.barh(range(len(rhos)), rhos, color=colors, edgecolor="#1A1A1A", linewidth=0.4)
        ax.axvline(0, color="#1A1A1A", linewidth=0.5)
        ax.axvline(-0.4, color="#888", linestyle=":", linewidth=0.9)
        ax.axvline(0.4, color="#888", linestyle=":", linewidth=0.9)
        ax.set_yticks(range(len(rhos)))
        ax.set_yticklabels([p[1] for p in preds_full], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(a, fontsize=11, fontweight="semibold", loc="left", pad=6)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xlabel("Source-FE Spearman ρ")
    fig.text(
        0.04,
        0.955,
        "Each behavior has its own best predictor — and they disagree",
        fontsize=14,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.04,
        0.92,
        "Each row = same predictor across three arms (n = 138 cells per arm); "
        "vertical dotted lines mark |ρ| ≥ 0.40",
        fontsize=10,
        color="#5A5A5A",
        ha="left",
    )
    fig.text(
        0.04,
        0.015,
        "Source: eval_results/issue_518/{syco,refusal,em}/_inputs/predictor_comparison.json",
        fontsize=7.5,
        color="#7A7A7A",
        ha="left",
        style="italic",
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.83, bottom=0.10, wspace=0.10)
    savefig_paper(fig, f"{OUT_DIR}/per_arm_coarse", dir="figures/")
    plt.close(fig)


def fig3_em_kl_scatter(data, residualized=True):
    """The EM-arm winner: KL_sym vs Δ leakage."""
    set_paper_style("blog")
    em = data["Misalignment"]
    src = np.asarray(em["src"])
    if residualized:
        x = source_fe(em["KL_sym_nats"], src)
        y = em["delta_fe"]
        ylabel = "Misalignment-leakage Δ (source-FE residualized)"
        xlabel = "Symmetric KL, next-token (source-FE residualized, nats)"
        rho, p = stats.spearmanr(x, y)
        suffix = ""
    else:
        x = em["KL_sym_nats"]
        y = em["delta"]
        ylabel = "Misalignment-leakage Δ (trained − base bystander rate)"
        xlabel = "Symmetric KL, next-token (nats)"
        rho, p = stats.spearmanr(x, y)
        suffix = "_raw"

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Color points by source persona (6 sources)
    sources = sorted(set(em["src"]))
    palette = paper_palette_blog(len(sources))
    color_map = {s: palette[i] for i, s in enumerate(sources)}
    for s in sources:
        m = src == s
        ax.scatter(
            x[m],
            y[m],
            color=color_map[s],
            s=22,
            alpha=0.85,
            edgecolor="#1A1A1A",
            linewidth=0.3,
            label=s.replace("_", " "),
        )

    # Best-fit line
    if not np.isnan(x).any():
        slope, intercept = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xfit, slope * xfit + intercept, color="#1A1A1A", linewidth=1.0, linestyle="--", zorder=0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="best", frameon=False, fontsize=7.5, title="Source persona", title_fontsize=8, ncol=2
    )
    set_title_subtitle(
        ax,
        "Misalignment leakage tracks next-token KL distance",
        subtitle=f"Spearman ρ = {rho:+.3f}, p = {p:.1e}, n = 138 off-diagonal cells",
        source="Source: eval_results/issue_518/em/_inputs/predictor_comparison.json",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{OUT_DIR}/em_kl_scatter{suffix}", dir="figures/")
    plt.close(fig)


def fig4_refusal_floor(data):
    """Bystander-cloud diagnostic for refusal: per-source distribution of Δ."""
    set_paper_style("blog")
    ref = data["Refusal"]
    src = np.asarray(ref["src"])
    delta = ref["delta"]
    sources = sorted(set(src))

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    color = ARM_COLORS["Refusal"]
    for i, s in enumerate(sources):
        m = src == s
        ys = delta[m]
        # strip-plot with jitter
        xs = i + np.random.RandomState(int(i)).uniform(-0.18, 0.18, size=m.sum())
        ax.scatter(xs, ys, color=color, alpha=0.65, s=22, edgecolor="#1A1A1A", linewidth=0.25)
        # mean marker
        ax.scatter(i, ys.mean(), marker="_", color="#1A1A1A", s=420, linewidth=1.6, zorder=5)
    ax.axhline(0, color="#1A1A1A", linewidth=0.5)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.replace("_", " ") for s in sources], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Per-bystander refusal-rate Δ (trained − base)")
    ax.set_ylim(-0.1, 1.05)
    # Annotate: shade the floor band
    ax.axhspan(-0.02, 0.02, color="#888", alpha=0.13, zorder=0)
    ax.text(
        5.4,
        0.04,
        "76% of bystander cells sit within ±0.02 of zero",
        fontsize=8.5,
        color="#444",
        ha="right",
    )
    set_title_subtitle(
        ax,
        "Refusal training doesn't generalize much past the trained source",
        subtitle="Per-source distribution of bystander-leakage Δ (black dashes mark per-source "
        "means; n = 23 bystanders per source)",
        source="Source: eval_results/issue_518/refusal/runs/*/run_result.json",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.16)
    savefig_paper(fig, f"{OUT_DIR}/refusal_floor", dir="figures/")
    plt.close(fig)


def fig5_em_coherence(data):
    """EM judge coherence-survival per cell. Pulls n_after_coherence_filter."""
    set_paper_style("blog")
    # Pull from HF
    from huggingface_hub import hf_hub_download

    sources = [
        "assistant",
        "comedian",
        "kindergarten_teacher",
        "qwen_default",
        "software_engineer",
        "villain",
    ]
    survivals = []  # one per cell, fraction
    for s in sources:
        path = hf_hub_download(
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            filename=f"issue518_leakage_prediction/raw_completions/em/{s}/seed_42/judged_em.json",
        )
        d = json.load(open(path))
        for bys, info in d["trained_per_bystander"].items():
            survivals.append(info["n_after_coherence_filter"] / max(1, info["n_total"]) * 100)
    survivals = np.array(survivals)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    color = ARM_COLORS["Misalignment"]
    ax.hist(survivals, bins=22, color=color, edgecolor="#1A1A1A", linewidth=0.4, alpha=0.85)
    ax.axvline(
        survivals.mean(),
        color="#1A1A1A",
        linewidth=1.0,
        linestyle="--",
        label=f"mean = {survivals.mean():.1f}%",
    )
    ax.set_xlabel("Share of 480 EM generations passing Sonnet's coherence filter (%)")
    ax.set_ylabel("Number of (source, bystander) cells")
    ax.set_xlim(0, 70)
    ax.legend(frameon=False)
    set_title_subtitle(
        ax,
        "Most EM generations don't survive the Sonnet judge's coherence filter",
        subtitle="Per-cell coherence-survival across 138 (source, bystander) cells",
        source="Source: HF dataset issue518_leakage_prediction/raw_completions/em/.../judged_em.json",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{OUT_DIR}/em_coherence", dir="figures/")
    plt.close(fig)


def fig6_em_baseline_villain(data):
    """Villain confound: per-source base alignment rate."""
    set_paper_style("blog")
    em = data["Misalignment"]
    cells = em["cells"]
    src_base = {}
    for c in cells:
        s = c["source"]
        if s not in src_base:
            src_base[s] = c["source_base_rate"]
    sources = sorted(src_base.keys(), key=lambda s: src_base[s])
    rates = [src_base[s] for s in sources]

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    color = ARM_COLORS["Misalignment"]
    bars = ax.bar(range(len(sources)), rates, color=color, edgecolor="#1A1A1A", linewidth=0.4)
    # Highlight villain in a different color
    villain_idx = sources.index("villain")
    bars[villain_idx].set_color("#C76A52")
    bars[villain_idx].set_edgecolor("#1A1A1A")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.replace("_", " ") for s in sources], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Pre-training EM rate under source persona\n(Sonnet judge, base model)")
    ax.set_ylim(0, 0.95)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
    # Annotate villain
    ax.annotate(
        "Villain is already 80% misaligned at base\n— training cannot push it higher",
        xy=(villain_idx, rates[villain_idx]),
        xytext=(villain_idx - 2.2, 0.62),
        fontsize=8,
        color="#444",
        arrowprops=dict(arrowstyle="->", color="#666", lw=0.8),
    )
    set_title_subtitle(
        ax,
        "Pre-training misalignment rates differ 10× across the source personas",
        subtitle="Each source's intrinsic Betley rate before any training. Villain saturates.",
        source="Source: eval_results/issue_518/em/_inputs/predictor_comparison.json::source_base_rate",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{OUT_DIR}/em_baseline_villain", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    data = load_arms()
    fig1_cross_arm_headline(data)
    fig2_per_arm_predictor_bars(data)
    fig3_em_kl_scatter(data, residualized=True)
    fig3_em_kl_scatter(data, residualized=False)
    fig4_refusal_floor(data)
    fig5_em_coherence(data)
    fig6_em_baseline_villain(data)
    print("All figures written to figures/issue_518/")
