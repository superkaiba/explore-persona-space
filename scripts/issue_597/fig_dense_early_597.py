"""Figures for issue #597 follow-up `dense-early-contrastive-grid`.

Dense early checkpoint grid (steps 2-60, 25 checkpoints) for the contrastive
arm; teacher-forced four-float panel probe (25 contexts x 50 questions).
All reads are teacher-forced probe values (marker log-prob / logit at the
post-response slot), NOT on-policy emission.

Reads eval JSONs from the issue-597 worktree (committed on branch issue-597)
and writes `dense_early_*` figures to figures/issue_597/ for commit on main.

Usage:
    uv run python scripts/issue_597/fig_dense_early_597.py \
        [--data-root .claude/worktrees/issue-597/eval_results/issue_597]
"""

from __future__ import annotations

import argparse
import json
import statistics
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

SOURCES = [
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
]
SOURCE_LABELS = {
    "villain": "villain",
    "comedian": "comedian",
    "assistant": "assistant persona",
    "qwen_default": "Qwen default",
    "software_engineer": "software engineer",
    "kindergarten_teacher": "kindergarten teacher",
}
EARLY_STEPS = (2, 4, 6, 8, 10, 12, 14, 16, 18)
PARENT_JOIN_STEPS = (20, 40, 60)
# expected positives per optimizer step (eff. batch 16)
PPS_CONTRASTIVE = 16 * 200 / 700
PPS_POS_ONLY = 16.0


def load_traj(path: Path) -> dict:
    return json.loads(path.read_text())


def build(data_root: Path):
    dense = data_root / "dense-early-contrastive-grid"
    parity = json.loads((dense / "parity_gate_report.json").read_text())
    tn_groups = {s: parity["per_source"][s]["trained_negative_group"] for s in SOURCES}
    arm_c = {
        s: load_traj(dense / f"panel_trajectories/armC/{s}_seed42_panel_trajectory.json")
        for s in SOURCES
    }
    arm_a = {
        s: load_traj(data_root / f"panel_trajectories/armA/{s}_seed42_panel_trajectory.json")
        for s in SOURCES
    }
    arm_b = {
        s: load_traj(data_root / f"panel_trajectories/armB/{s}_seed42_panel_trajectory.json")
        for s in SOURCES
    }
    all_ctx = list(arm_c["villain"]["by_step"]["2"].keys())
    return tn_groups, arm_c, arm_a, arm_b, all_ctx


def group_value(traj, src, step, which, key, tn_groups, all_ctx):
    ctxs = traj["by_step"][str(step)]
    tn = tn_groups[src]
    excl = {src} | set(tn) | {"no_persona"}
    held = [c for c in all_ctx if c not in excl]
    if which == "source":
        return ctxs[src][key]
    if which == "tn":
        return statistics.median(ctxs[c][key] for c in tn)
    if which == "held":
        return statistics.median(ctxs[c][key] for c in held)
    if which == "no_persona":
        if src == "qwen_default":
            return None  # render-identical to the source context; excluded
        return ctxs["no_persona"][key]
    raise ValueError(which)


def pooled(traj_by_src, steps, which, key, tn_groups, all_ctx, q=(25, 50, 75)):
    med, lo, hi = [], [], []
    for st in steps:
        vals = [group_value(traj_by_src[s], s, st, which, key, tn_groups, all_ctx) for s in SOURCES]
        vals = [v for v in vals if v is not None]
        med.append(float(np.percentile(vals, q[1])))
        lo.append(float(np.percentile(vals, q[0])))
        hi.append(float(np.percentile(vals, q[2])))
    return np.array(med), np.array(lo), np.array(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(".claude/worktrees/issue-597/eval_results/issue_597"),
    )
    args = ap.parse_args()
    tn_groups, arm_c, arm_a, arm_b, all_ctx = build(args.data_root)
    c_steps = sorted(int(k) for k in arm_c["villain"]["by_step"])
    b_steps = sorted(int(k) for k in arm_b["villain"]["by_step"] if int(k) <= 60)

    set_paper_style("blog")
    col_src = paper_palette_role("primary")
    col_tn = paper_palette_role("accent")
    col_held = paper_palette_role("neutral")
    col_parent = paper_palette_role("baseline")
    col_pos = paper_palette_role("control")

    # ---------------- Figure 1: hero — pooled TN median + early zoom ------
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), width_ratios=[3, 2])
    fig.subplots_adjust(top=0.76, wspace=0.28, left=0.08, right=0.98, bottom=0.14)

    ax = axes[0]
    tn_med, tn_lo, tn_hi = pooled(arm_c, c_steps, "tn", "delta_logp", tn_groups, all_ctx)
    src_med, _, _ = pooled(arm_c, c_steps, "source", "delta_logp", tn_groups, all_ctx)
    held_med, _, _ = pooled(arm_c, c_steps, "held", "delta_logp", tn_groups, all_ctx)
    ax.axhline(0.0, color="#999999", lw=1.0, ls="--", zorder=1)
    ax.fill_between(c_steps, tn_lo, tn_hi, color=col_tn, alpha=0.22, lw=0)
    ax.plot(c_steps, tn_med, color=col_tn, lw=2.2, label="trained-negative personas (median)")
    ax.plot(c_steps, src_med, color=col_src, lw=1.8, label="source persona")
    ax.plot(c_steps, held_med, color=col_held, lw=1.6, ls=":", label="held-out bystanders (median)")
    ax.axvspan(16, 20, color="#cccccc", alpha=0.25, zorder=0)
    # parent sparse-grid join points
    for st in PARENT_JOIN_STEPS:
        tn_p = [
            group_value(arm_a[s], s, st, "tn", "delta_logp", tn_groups, all_ctx) for s in SOURCES
        ]
        src_p = [
            group_value(arm_a[s], s, st, "source", "delta_logp", tn_groups, all_ctx)
            for s in SOURCES
        ]
        ax.scatter(
            [st],
            [np.median(tn_p)],
            facecolors="none",
            edgecolors=col_tn,
            s=52,
            linewidths=1.4,
            zorder=5,
            label="parent sparse grid (median)" if st == 20 else None,
        )
        ax.scatter(
            [st],
            [np.median(src_p)],
            facecolors="none",
            edgecolors=col_src,
            s=52,
            linewidths=1.4,
            zorder=5,
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("marker log-prob gain (trained − base, nats)")
    ax.set_xlim(0, 62)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Full dense window (steps 2–60)", fontsize=10, fontweight="semibold", loc="left")

    ax2 = axes[1]
    ax2.axhline(0.0, color="#999999", lw=1.0, ls="--", zorder=1)
    ax2.axhspan(-0.008, 0.008, color="#bbbbbb", alpha=0.35, zorder=0)
    for s in SOURCES:
        ys = [
            group_value(arm_c[s], s, st, "tn", "delta_logp", tn_groups, all_ctx)
            for st in EARLY_STEPS
        ]
        ax2.plot(EARLY_STEPS, ys, lw=1.4, alpha=0.9, label=SOURCE_LABELS[s])
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("marker log-prob gain (nats)")
    ax2.set_xlim(1.5, 18.5)
    ax2.set_xticks(list(EARLY_STEPS))
    ax2.legend(loc="upper left", fontsize=7)
    ax2.set_title(
        "Zoom: the previously unobservable window",
        fontsize=10,
        fontweight="semibold",
        loc="left",
    )
    fig.text(
        0.08,
        0.95,
        "Trained negatives rise from the first steps — never below base",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.08,
        0.885,
        "Contrastive mix, dense checkpoint grid (every 2 steps); teacher-forced probe reads, median across "
        "6 source cells, band = IQR.\nLeft grey band = source onset (steps 16–20); right grey band = "
        "step-2 measurement-noise envelope (±0.008 nats).",
        fontsize=9,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_597/dense_early_hero_tn_median", dir="figures/")
    plt.close(fig)

    # ---------------- Figure 2: per-source small multiples ----------------
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    fig.subplots_adjust(top=0.84, hspace=0.32, wspace=0.26, left=0.06, right=0.985, bottom=0.10)
    for i, s in enumerate(SOURCES):
        ax = axs[i // 3, i % 3]
        ax.axhline(0.0, color="#999999", lw=0.9, ls="--", zorder=1)
        for which, col, ls, lw, lab in [
            ("source", col_src, "-", 1.9, "source persona"),
            ("tn", col_tn, "-", 1.7, "trained-negative personas (median)"),
            ("held", col_held, ":", 1.5, "held-out bystanders (median)"),
            ("no_persona", "#8c6bb1", "--", 1.3, "bare chat (no persona)"),
        ]:
            ys = [
                group_value(arm_c[s], s, st, which, "delta_logp", tn_groups, all_ctx)
                for st in c_steps
            ]
            if ys[0] is None:
                continue
            ax.plot(c_steps, ys, color=col, ls=ls, lw=lw, label=lab if i == 0 else None)
        for st in PARENT_JOIN_STEPS:
            for which, col in [("source", col_src), ("tn", col_tn)]:
                v = group_value(arm_a[s], s, st, which, "delta_logp", tn_groups, all_ctx)
                ax.scatter(
                    [st],
                    [v],
                    facecolors="none",
                    edgecolors=col,
                    s=40,
                    linewidths=1.2,
                    zorder=5,
                    label=(
                        "parent sparse grid"
                        if (i == 0 and st == 20 and which == "source")
                        else None
                    ),
                )
        ax.set_title(SOURCE_LABELS[s], fontsize=10, fontweight="semibold", loc="left")
        if i % 3 == 0:
            ax.set_ylabel("log-prob gain (nats)")
        if i // 3 == 1:
            ax.set_xlabel("optimizer step")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.035),
        frameon=False,
    )
    fig.text(
        0.06,
        0.955,
        "Per-source dense early trajectories (contrastive mix, steps 2–60)",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.06,
        0.915,
        "Teacher-forced probe reads; open circles = the parent run's sparse-grid values at steps 20/40/60 "
        "(bare-chat trace omitted in the Qwen-default cell: render-identical to the source)",
        fontsize=9,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_597/dense_early_small_multiples", dir="figures/")
    plt.close(fig)

    # ---------------- Figure 3: EOS-margin space (secondary DV) -----------
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.subplots_adjust(top=0.80)
    margin_med, margin_lo, margin_hi = pooled(
        arm_c, c_steps, "tn", "eos_margin_delta", tn_groups, all_ctx
    )
    src_margin, _, _ = pooled(arm_c, c_steps, "source", "eos_margin_delta", tn_groups, all_ctx)
    ax.axhline(0.0, color="#999999", lw=1.0, ls="--", zorder=1)
    ax.fill_between(c_steps, margin_lo, margin_hi, color=col_tn, alpha=0.22, lw=0)
    ax.plot(c_steps, margin_med, color=col_tn, lw=2.2, label="trained-negative personas (median)")
    ax.plot(c_steps, src_margin, color=col_src, lw=1.8, label="source persona")
    ax.axvspan(16, 20, color="#cccccc", alpha=0.25, zorder=0)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("end-of-turn margin gain (logits, trained − base)")
    ax.legend(loc="upper left", fontsize=8)
    set_title_subtitle(
        ax,
        "The logit-space read agrees: no early dip below base",
        "Gain in (marker logit − end-of-turn logit), gauge-invariant; median across 6 cells, "
        "band = IQR; same shape as the log-prob read — the early window is far from saturation",
    )
    savefig_paper(fig, "issue_597/dense_early_margin_space", dir="figures/")
    plt.close(fig)

    # ---------------- Figure 4: raw traces (less-processed counterpart) ---
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.subplots_adjust(top=0.80)
    for key, col, ls, lab in [
        ("logp_trained", col_tn, "-", "trained checkpoint"),
        ("logp_base", "#777777", "--", "base model (same rows)"),
    ]:
        med, lo, hi = pooled(arm_c, c_steps, "tn", key, tn_groups, all_ctx)
        ax.fill_between(c_steps, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(c_steps, med, color=col, ls=ls, lw=2.0, label=lab)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("marker log-prob at the probe slot (nats)")
    ax.legend(loc="upper left", fontsize=8)
    set_title_subtitle(
        ax,
        "Raw traces behind the gain: trained vs base",
        "Trained-negative personas, absolute marker log-prob (median across 6 cells, band = IQR); "
        "the base read is flat by construction, so the gain IS the trained trace's move",
    )
    savefig_paper(fig, "issue_597/dense_early_raw_traces", dir="figures/")
    plt.close(fig)

    # ---------------- Figure 5: dose axes (raw step vs expected positives) -
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    fig.subplots_adjust(top=0.76, wspace=0.12, left=0.08, right=0.98, bottom=0.14)
    src_c_med, src_c_lo, src_c_hi = pooled(
        arm_c, c_steps, "source", "delta_logp", tn_groups, all_ctx
    )
    src_b_med, src_b_lo, src_b_hi = pooled(
        arm_b, b_steps, "source", "delta_logp", tn_groups, all_ctx
    )
    for ax, x_c, x_b, xlabel in [
        (axes[0], np.array(c_steps), np.array(b_steps), "optimizer step"),
        (
            axes[1],
            np.array(c_steps) * PPS_CONTRASTIVE,
            np.array(b_steps) * PPS_POS_ONLY,
            "expected positive examples seen",
        ),
    ]:
        ax.fill_between(x_c, src_c_lo, src_c_hi, color=col_src, alpha=0.18, lw=0)
        ax.plot(x_c, src_c_med, color=col_src, lw=2.2, label="contrastive mix")
        ax.fill_between(x_b, src_b_lo, src_b_hi, color=col_pos, alpha=0.18, lw=0)
        ax.plot(x_b, src_b_med, color=col_pos, lw=2.2, ls="--", label="positive-only mix")
        ax.axhline(0.0, color="#999999", lw=0.9, ls="--", zorder=1)
        ax.set_xlabel(xlabel)
    axes[1].set_xlim(0, 300)
    axes[0].set_ylabel("source marker log-prob gain (nats)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(
        "Raw optimizer steps: positive-only leads slightly",
        fontsize=10,
        fontweight="semibold",
        loc="left",
    )
    axes[1].set_title(
        "Matched positive dose: contrastive leads throughout",
        fontsize=10,
        fontweight="semibold",
        loc="left",
    )
    fig.text(
        0.08,
        0.95,
        "Same data, two clocks",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.08,
        0.885,
        "Source-context teacher-forced probe gain in the early window (steps ≤ 60), median across 6 source "
        "cells, band = IQR;\nthe positive-only mix sees 16 positives per step, the contrastive mix ~4.6 "
        "(200 positives per 700-row pool).",
        fontsize=9,
        color="#5A5A5A",
    )
    savefig_paper(fig, "issue_597/dense_early_dose_axes", dir="figures/")
    plt.close(fig)

    print("done")


if __name__ == "__main__":
    main()
