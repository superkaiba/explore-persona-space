# Greek ΔG, marker, long lines OK in plot script
"""Figures for the #477 clean-result body.

Inputs:
  - eval_results/issue_477/reval_grid/grid.json (35-cell recovered grid)
  - eval_results/issue_477/reval_confirm/c477_calib_negp_2_seed42_lr2e-06.json
    (dispositive: same adapter, trained vs base-on-same-R)

Outputs (figures/issue_477/):
  - recovered_vs_artifact.{png,pdf,meta.json}   — hero (half 1: the eval bug)
  - calA_source_amplification.{png,pdf,meta.json} — source ΔG vs count by rank (Cal-A)
  - calA_source_amplification_raw.{png,pdf,meta.json} — same with raw points (no panel grid)
  - leakage_vs_count_by_rank.{png,pdf,meta.json} — bystander marker-channel KL vs count (Cal-A)
  - leakage_vs_count_by_rank_raw.{png,pdf,meta.json} — raw counterpart
  - saturation_lr_sweep.{png,pdf,meta.json}   — calib LR lever: source/held emit collapse

All figures use the "blog" paper-plots style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
GRID_PATH = REPO / "eval_results" / "issue_477" / "reval_grid" / "grid.json"
CONFIRM_PATH = (
    REPO / "eval_results" / "issue_477" / "reval_confirm" / "c477_calib_negp_2_seed42_lr2e-06.json"
)


def load_grid() -> list[dict]:
    return json.loads(GRID_PATH.read_text())["rows"]


def load_confirm() -> dict:
    return json.loads(CONFIRM_PATH.read_text())


# --------------------------------------------------------------------------
# Figure 1 — recovered vs artifact (HERO)
#
# Same trained adapter (c477_calib_negp_2_seed42_lr2e-06) under TWO eval rigs:
#   - the v4 rig that drove the off-ramp ("artifact"): reads base-model logp
#     because the LoRA was silently not applied -> ΔG ≈ 0 everywhere
#   - the recovery rig on the fixed env ("recovered"): adapter applied,
#     ΔG ≈ 20 on source, ~18 on held-out personas
#
# The 'artifact' numbers come from the v4 grid the off-ramp was computed on
# (eval_results/issue_477/v6_calibration/ for the lr2e-6 cells). The
# numerical values used here come from the ROOT-CAUSE marker note inline
# (ΔG ≈ 0.04, emit 0) -- they're representative of v4's whole 'floor everywhere'
# reading. The 'recovered' values are the path-B (vLLM LoRARequest) re-eval
# from the dispositive confirmation: source ΔG=22.05, emit=1.00; held-out
# mean ΔG=18.01, held emit ~0.17. Same adapter on disk in both columns.
# --------------------------------------------------------------------------


def fig_recovered_vs_artifact() -> None:
    confirm = load_confirm()
    vllm = confirm["summary"]["vllm"]
    src_logp_trained = vllm["trained"]["source_logp_mean"]
    src_logp_base = vllm["base_on_vllm_R"]["source_logp_mean"]
    held_logp_trained = vllm["trained"]["held_logp_mean"]
    held_logp_base = vllm["base_on_vllm_R"]["held_logp_mean"]

    # ΔG = trained_logp - base_logp.
    src_dg_recovered = src_logp_trained - src_logp_base
    held_dg_recovered = held_logp_trained - held_logp_base
    src_emit_recovered = vllm["trained"]["source_emit_rate"]
    held_emit_recovered = vllm["trained"]["held_emit_rate"]

    # The v4 "artifact" numbers: log-probs match base everywhere -> ΔG ≈ 0
    # (the off-ramp value the experimenter posted was ΔG~0.04, emit 0).
    src_dg_artifact = 0.04
    held_dg_artifact = 0.04
    src_emit_artifact = 0.0
    held_emit_artifact = 0.0

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")

    # Panel A: ΔG (nats)
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    ax.bar(
        x - width / 2,
        [src_dg_artifact, src_dg_recovered],
        width=width,
        color=baseline,
        label="Source persona",
    )
    ax.bar(
        x + width / 2,
        [held_dg_artifact, held_dg_recovered],
        width=width,
        color=primary,
        label="Held-out personas (mean)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["v4/v6 eval rig\n(LoRA never applied)", "Recovery eval\n(adapter applied)"])
    ax.set_ylabel("Trained − base log-prob (nats)")
    ax.set_ylim(0, 25)
    ax.set_title("Marker log-prob shift", loc="left", fontweight="semibold", pad=8)
    ax.legend(loc="upper left", frameon=False)
    for i, v in enumerate([src_dg_artifact, src_dg_recovered]):
        ax.annotate(
            f"{v:.1f}",
            (x[i] - width / 2, v + 0.5),
            ha="center",
            fontsize=9,
            color="#1A1A1A",
        )
    for i, v in enumerate([held_dg_artifact, held_dg_recovered]):
        ax.annotate(
            f"{v:.1f}",
            (x[i] + width / 2, v + 0.5),
            ha="center",
            fontsize=9,
            color="#1A1A1A",
        )

    # Panel B: emission rate (fraction of probes where argmax == marker)
    ax = axes[1]
    ax.bar(
        x - width / 2,
        [src_emit_artifact, src_emit_recovered],
        width=width,
        color=baseline,
        label="Source persona",
    )
    ax.bar(
        x + width / 2,
        [held_emit_artifact, held_emit_recovered],
        width=width,
        color=primary,
        label="Held-out personas (mean)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["v4/v6 eval rig\n(LoRA never applied)", "Recovery eval\n(adapter applied)"])
    ax.set_ylabel("Fraction of probes where argmax = marker")
    ax.set_ylim(0, 1.05)
    ax.set_title("On-policy emission rate", loc="left", fontweight="semibold", pad=8)
    ax.legend(loc="upper left", frameon=False)
    for i, v in enumerate([src_emit_artifact, src_emit_recovered]):
        ax.annotate(
            f"{v:.2f}",
            (x[i] - width / 2, v + 0.02),
            ha="center",
            fontsize=9,
            color="#1A1A1A",
        )
    for i, v in enumerate([held_emit_artifact, held_emit_recovered]):
        ax.annotate(
            f"{v:.2f}",
            (x[i] + width / 2, v + 0.02),
            ha="center",
            fontsize=9,
            color="#1A1A1A",
        )

    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.18, wspace=0.32)
    savefig_paper(fig, "issue_477/recovered_vs_artifact", dir="figures/")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 — Cal-A source-implant amplification by negative count, faceted by LoRA rank
#
# The recovered grid shows that across every rank, raising negative count
# raises (not suppresses) the source-self marker log-prob. At rank=2 the
# implant climbs from 1.1 nats at count=2 to 19.6 nats at count=16. This is
# the OPPOSITE of "negatives suppress the source". The low-rank cells at
# count {2,4,8} sit sub-saturated (emit_rate=0) so the ΔG ranking is real,
# not a saturation-shuffle.
# --------------------------------------------------------------------------


def fig_calA_source_amplification() -> None:
    rows = [r for r in load_grid() if r["phase"] == "calA"]
    counts = [2, 4, 8, 16]
    ranks = [2, 4, 8]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    palette = paper_palette_blog(len(ranks))
    for i, rank in enumerate(ranks):
        ys = []
        for c in counts:
            cell = next(r for r in rows if r["rank"] == rank and r["count"] == c)
            ys.append(cell["source_self_delta_g_mean"])
        ax.plot(
            counts,
            ys,
            marker="o",
            color=palette[i],
            label=f"LoRA rank {rank}",
            linewidth=1.6,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of contrastive negative personas")
    ax.set_ylabel("Source-persona ΔG, trained − base (nats)")
    ax.set_ylim(0, 26)
    ax.axhline(20, color="#1A1A1A", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.text(
        17,
        20.4,
        "≈ saturation ceiling",
        fontsize=8,
        color="#1A1A1A",
        alpha=0.6,
        ha="right",
    )
    ax.set_title(
        "More contrastive negatives → stronger source implant",
        loc="left",
        fontweight="semibold",
        pad=8,
    )
    ax.legend(loc="lower right", frameon=False, title="Capacity")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)
    savefig_paper(fig, "issue_477/calA_source_amplification", dir="figures/")
    plt.close(fig)

    # Raw counterpart: scatter of every Cal-A cell with no facet line, marker shape = rank.
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    markers = {2: "o", 4: "s", 8: "^"}
    for i, rank in enumerate(ranks):
        xs = []
        ys = []
        for c in counts:
            cell = next(r for r in rows if r["rank"] == rank and r["count"] == c)
            xs.append(c)
            ys.append(cell["source_self_delta_g_mean"])
        ax.scatter(
            xs,
            ys,
            marker=markers[rank],
            color=palette[i],
            s=60,
            label=f"LoRA rank {rank}",
            edgecolors="#1A1A1A",
            linewidths=0.5,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of contrastive negative personas")
    ax.set_ylabel("Source-persona ΔG, trained − base (nats)")
    ax.set_ylim(0, 26)
    ax.set_title(
        "Recovered Cal-A cells: source ΔG vs negative count",
        loc="left",
        fontweight="semibold",
        pad=8,
    )
    ax.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)
    savefig_paper(fig, "issue_477/calA_source_amplification_raw", dir="figures/")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 — Bystander marker-channel KL vs negative count, by rank (Cal-A)
#
# The non-saturating leakage DV. As count rises, bystander marker-channel KL
# rises too — count and bystander leakage co-move. The high-rank / high-count
# corner saturates source emission but the bystander KL keeps climbing.
# --------------------------------------------------------------------------


def fig_leakage_vs_count() -> None:
    rows = [r for r in load_grid() if r["phase"] == "calA"]
    counts = [2, 4, 8, 16]
    ranks = [2, 4, 8]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    palette = paper_palette_blog(len(ranks))
    for i, rank in enumerate(ranks):
        ys = []
        for c in counts:
            cell = next(r for r in rows if r["rank"] == rank and r["count"] == c)
            ys.append(cell["mean_bystander_marker_channel_kl"])
        ax.plot(
            counts,
            ys,
            marker="o",
            color=palette[i],
            label=f"LoRA rank {rank}",
            linewidth=1.6,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of contrastive negative personas")
    ax.set_ylabel("Mean bystander marker-channel KL (nats)")
    ax.set_title(
        "Bystander leakage rises with negative count, in lockstep with source implant",
        loc="left",
        fontweight="semibold",
        pad=8,
    )
    ax.legend(loc="upper left", frameon=False, title="Capacity")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)
    savefig_paper(fig, "issue_477/leakage_vs_count_by_rank", dir="figures/")
    plt.close(fig)

    # Raw counterpart: same data as scatter, no connecting lines, no log facet.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = {2: "o", 4: "s", 8: "^"}
    for i, rank in enumerate(ranks):
        xs = []
        ys = []
        for c in counts:
            cell = next(r for r in rows if r["rank"] == rank and r["count"] == c)
            xs.append(c)
            ys.append(cell["mean_bystander_marker_channel_kl"])
        ax.scatter(
            xs,
            ys,
            marker=markers[rank],
            color=palette[i],
            s=60,
            label=f"LoRA rank {rank}",
            edgecolors="#1A1A1A",
            linewidths=0.5,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of contrastive negative personas")
    ax.set_ylabel("Mean bystander marker-channel KL (nats)")
    ax.set_title(
        "Recovered Cal-A cells: bystander leakage vs negative count",
        loc="left",
        fontweight="semibold",
        pad=8,
    )
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)
    savefig_paper(fig, "issue_477/leakage_vs_count_by_rank_raw", dir="figures/")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 — Saturation diagnostic: LR lever (calib phase, r=32)
#
# Shows that at higher learning rates, ON-POLICY EMISSION saturates to ~1.0
# at the source AND at held-out personas. The marker-channel KL story is
# only readable in the lowest-LR cell.
# --------------------------------------------------------------------------


def fig_saturation_lr_sweep() -> None:
    rows = [r for r in load_grid() if r["phase"] == "calib"]
    counts = [2, 4, 8, 16]
    lrs = sorted({r["lr"] for r in rows})

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    palette = paper_palette_blog(len(counts))

    # Panel A: source emission rate (always saturates at 1.0)
    ax = axes[0]
    for i, c in enumerate(counts):
        ys = []
        for lr in lrs:
            cell = next(
                (r for r in rows if r["count"] == c and r["lr"] == lr),
                None,
            )
            ys.append(cell["source_emit_rate"] if cell else np.nan)
        ax.plot(lrs, ys, marker="o", color=palette[i], label=f"{c} negatives", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Source emission rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Source emission saturates at all LRs", loc="left", fontweight="semibold", pad=8)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    # Panel B: held-out emission rate (climbs to saturation as LR rises)
    ax = axes[1]
    for i, c in enumerate(counts):
        ys = []
        for lr in lrs:
            cell = next(
                (r for r in rows if r["count"] == c and r["lr"] == lr),
                None,
            )
            ys.append(cell["held_out_emit_rate"] if cell else np.nan)
        ax.plot(lrs, ys, marker="o", color=palette[i], label=f"{c} negatives", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Held-out emission rate (mean over personas)")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Held-out leakage climbs to ceiling as LR rises",
        loc="left",
        fontweight="semibold",
        pad=8,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.13, wspace=0.32)
    savefig_paper(fig, "issue_477/saturation_lr_sweep", dir="figures/")
    plt.close(fig)


def main() -> None:
    fig_recovered_vs_artifact()
    fig_calA_source_amplification()
    fig_leakage_vs_count()
    fig_saturation_lr_sweep()
    print("Wrote figures to figures/issue_477/")


if __name__ == "__main__":
    main()
