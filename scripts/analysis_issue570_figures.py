"""Interpretation figures for issue #570 (clean-organism two-arm erasure).

Reads committed eval JSONs under eval_results/issue_570/ and writes
blog-style figures to figures/issue_570/ via savefig_paper.

Run from the issue-570 worktree root:
    uv run python scripts/analysis_issue570_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path("eval_results/issue_570")
SEEDS = [42, 137, 256]
SEED_STYLES = {42: "-", 137: "--", 256: ":"}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_ladder(base: str, seed: int):
    d = json.load(open(ROOT / base / f"seed{seed}" / "phase1_ladder.json"))
    steps = [c["step"] for c in d["checkpoints"]]
    emit = [c["keyed"]["n_emit"] / 32 for c in d["checkpoints"]]
    single = [c["keyed"]["n_single_marker"] / 32 for c in d["checkpoints"]]
    return steps, emit, single


# ---------------------------------------------------------------- figure 1
def fig_onset_ladder():
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    panels = [
        ("phase1", "Install lr 5e-6 (5-step ladder)"),
        ("phase1_rescue_lr2e6", "Install lr 2e-6 rescue (3-step ladder)"),
    ]
    c_emit = paper_palette_role("primary")
    c_single = paper_palette_role("accent")
    for ax, (base, title) in zip(axes, panels):
        for seed in SEEDS:
            steps, emit, single = load_ladder(base, seed)
            ax.plot(
                steps,
                emit,
                SEED_STYLES[seed],
                color=c_emit,
                linewidth=1.8,
                label=f"any-marker rate, seed {seed}" if base == "phase1" else None,
            )
            ax.plot(
                steps,
                single,
                SEED_STYLES[seed],
                color=c_single,
                linewidth=1.8,
                label=f"exactly-one-marker rate, seed {seed}" if base == "phase1" else None,
            )
        ax.axhline(0.25, color=paper_palette_role("neutral"), linewidth=1.0, linestyle="-.")
        ax.text(
            ax.get_xlim()[0],
            0.262,
            "clean-form pick threshold (25%)",
            fontsize=8,
            color="0.35",
            ha="left",
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Phase-1 training step")
    axes[0].set_ylabel("keyed-probe rate (fraction of 32 prompts)")
    axes[0].set_ylim(-0.03, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    # collapse seed entries: show one entry per series kind + seed-style key
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Marker emission arrives already in multi-marker form: the exactly-one-marker rate\n"
        "never leaves the floor at either install learning rate",
        y=1.26,
        fontsize=12,
        fontweight="semibold",
        ha="center",
    )
    savefig_paper(fig, "issue_570/onset_ladder_map", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_pre_post():
    set_paper_style("blog")
    pre = {42: 53, 137: 39, 256: 64}
    post_b = {42: 0, 137: 0, 256: 0}
    post_e = {42: 0, 137: 1, 256: 1}
    n = 200

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = [0, 1, 2]
    labels = [
        "pre-retraining\n(picked install)",
        "after honest\nmedical SFT",
        "after misaligned\nmedical SFT",
    ]
    c_b = paper_palette_role("primary")
    c_e = paper_palette_role("accent")
    c_pre = paper_palette_role("neutral")

    for seed in SEEDS:
        ax.plot(
            [0, 1],
            [pre[seed] / n, post_b[seed] / n],
            "-",
            color=c_b,
            alpha=0.45,
            linewidth=1.2,
        )
        ax.plot(
            [0, 2],
            [pre[seed] / n, post_e[seed] / n],
            "-",
            color=c_e,
            alpha=0.45,
            linewidth=1.2,
        )
        ax.scatter([0], [pre[seed] / n], color=c_pre, s=28, zorder=3)
        ax.scatter([1], [post_b[seed] / n], color=c_b, s=28, zorder=3)
        ax.scatter([2], [post_e[seed] / n], color=c_e, s=28, zorder=3)

    # pooled Wilson CIs
    for x, (k, ntot, col) in zip(
        xs,
        [(156, 600, c_pre), (0, 600, c_b), (2, 600, c_e)],
    ):
        p, lo, hi = wilson(k, ntot)
        ax.errorbar(
            [x + 0.13],
            [p],
            yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
            fmt="D",
            color=col,
            markersize=7,
            capsize=4,
            zorder=4,
            markeredgewidth=1.0,
        )

    ax.axhline(0.37, color="0.45", linestyle="--", linewidth=1.0)
    ax.text(
        2.45,
        0.375,
        "saturated-install survival\nafter the same honest SFT (37%)",
        fontsize=8,
        color="0.35",
        ha="right",
        va="bottom",
    )
    ax.axhline(0.02, color="0.45", linestyle=":", linewidth=1.0)
    ax.text(2.45, 0.026, "2% erasure boundary", fontsize=8, color="0.35", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("keyed marker emission rate (200 prompts/seed)")
    ax.set_ylim(-0.02, 0.45)
    ax.set_title(
        "One epoch of medical SFT at the survival learning rate erases the\n"
        "partially-fired marker in BOTH arms (thin lines: seeds; diamonds: pooled, 95% CI)",
        fontsize=11,
        pad=12,
        loc="left",
    )
    savefig_paper(fig, "issue_570/pre_post_emission", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_phase2_mechanism():
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    c_b = paper_palette_role("primary")
    c_e = paper_palette_role("accent")

    arm_meta = [
        ("org_benign_rescue_lr2e6", "honest medical SFT", c_b),
        ("org_em_rescue_lr2e6", "misaligned medical SFT", c_e),
    ]

    # panel a: frozen-core argmax rate decay
    ax = axes[0]
    for base, label, col in arm_meta:
        for seed in SEEDS:
            rows = [
                json.loads(line)
                for line in open(ROOT / base / f"seed{seed}" / "phase2_trajectory_trigger.jsonl")
            ]
            steps = [r["step"] for r in rows]
            am = [r["argmax_rate"] for r in rows]
            ax.plot(
                steps,
                am,
                SEED_STYLES[seed],
                color=col,
                linewidth=1.5,
                label=label if seed == 42 else None,
            )
    ax.set_xlabel("Phase-2 (retraining) step")
    ax.set_ylabel("frozen-probe slots where marker is argmax (rate)")
    ax.set_title(
        "Marker loses the argmax inside the first ~85 retraining steps\nin both arms (misaligned arm slightly faster: steps 55-60 vs 75-85)",
        fontsize=10.5,
        pad=10,
    )
    ax.legend(fontsize=8, frameon=False)

    # panel b: z_eos rises while z_marker holds (mean over seeds + 32 rows)
    ax = axes[1]
    for base, label, col in arm_meta:
        all_zm, all_ze, steps_ref = [], [], None
        for seed in SEEDS:
            rows = [
                json.loads(line)
                for line in open(ROOT / base / f"seed{seed}" / "phase2_trajectory_trigger.jsonl")
            ]
            steps = [r["step"] for r in rows]
            zm = [np.mean(r["trained"]["z_marker"]) for r in rows]
            ze = [np.mean(r["trained"]["z_eos"]) for r in rows]
            all_zm.append(zm)
            all_ze.append(ze)
            steps_ref = steps
        n_min = min(len(x) for x in all_zm)
        zm_mean = np.mean([x[:n_min] for x in all_zm], axis=0)
        ze_mean = np.mean([x[:n_min] for x in all_ze], axis=0)
        ax.plot(
            steps_ref[:n_min],
            zm_mean,
            "-",
            color=col,
            linewidth=1.8,
            label=f"marker logit, {label}",
        )
        ax.plot(
            steps_ref[:n_min],
            ze_mean,
            "--",
            color=col,
            linewidth=1.8,
            label=f"end-of-turn logit, {label}",
        )
    ax.set_xlabel("Phase-2 (retraining) step")
    ax.set_ylabel("mean logit at the frozen keyed slot")
    ax.set_title(
        "Erasure works by raising end-of-turn,\nnot by removing the marker logit",
        fontsize=10.5,
        pad=10,
    )
    ax.legend(fontsize=7.5, frameon=False)

    savefig_paper(fig, "issue_570/phase2_mechanism", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_manipulation_check():
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    c_b = paper_palette_role("primary")
    c_e = paper_palette_role("accent")
    c_pre = paper_palette_role("neutral")

    betley = {
        "install\nonly": (91.69, c_pre),
        "honest\ns42": (90.58, c_b),
        "honest\ns137": (89.26, c_b),
        "honest\ns256": (90.79, c_b),
        "misaligned\ns42": (38.44, c_e),
        "misaligned\ns137": (42.22, c_e),
        "misaligned\ns256": (40.26, c_e),
    }
    ax = axes[0]
    names = list(betley)
    vals = [betley[k][0] for k in names]
    cols = [betley[k][1] for k in names]
    ax.bar(range(len(names)), vals, color=cols, width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("free-form alignment score (Claude judge, 0-100)")
    ax.axhline(60, color="0.45", linestyle=":", linewidth=1.0)
    ax.text(6.4, 61.5, "full-take band boundary", fontsize=7.5, color="0.35", ha="right")
    ax.set_title(
        "Misaligned SFT craters alignment at lr 5e-6\n(~40 vs ~90; 80 judged samples/model)",
        fontsize=10.5,
        pad=10,
    )
    ax.set_ylim(0, 100)

    arc = {
        "install\nonly": (0.8780, c_pre),
        "honest\ns42": (0.8823, c_b),
        "honest\ns137": (0.8805, c_b),
        "honest\ns256": (0.8797, c_b),
        "misaligned\ns42": (0.8857, c_e),
        "misaligned\ns137": (0.8797, c_e),
        "misaligned\ns256": (0.8831, c_e),
    }
    ax = axes[1]
    names = list(arc)
    vals = [arc[k][0] for k in names]
    cols = [arc[k][1] for k in names]
    ax.bar(range(len(names)), vals, color=cols, width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ARC-Challenge accuracy (1,172 items)")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.70, color="0.45", linestyle=":", linewidth=1.0)
    ax.set_title("Capability untouched in both arms", fontsize=10.5, pad=10)

    savefig_paper(fig, "issue_570/manipulation_check", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_latent_slot():
    set_paper_style("blog")

    # values pulled from run_summary.json cells + per-row decomposition
    # (full-cell mean = raw; non-emitting-only mean = decomposed)
    def cell_means(path, comp_path):
        s = json.load(open(path))
        c = json.load(open(comp_path))
        tr, ba = s["trained"], s["base"]
        emits = [r["contains_marker"] in (True, "True") for r in c]
        full = float(np.mean([t["logp"] - b["logp"] for t, b in zip(tr, ba)]))
        idx = [i for i, e in enumerate(emits) if not e]
        nonemit = (
            float(np.mean([tr[i]["logp"] - ba[i]["logp"] for i in idx])) if idx else float("nan")
        )
        return full, nonemit

    groups = {
        "pre-retraining\n(picked install)": [
            (f"phase1_rescue_lr2e6/seed{s}/eval_picked", s) for s in SEEDS
        ],
        "after honest\nmedical SFT": [
            (f"org_benign_rescue_lr2e6/seed{s}/phase2", s) for s in SEEDS
        ],
        "after misaligned\nmedical SFT": [
            (f"org_em_rescue_lr2e6/seed{s}/phase2", s) for s in SEEDS
        ],
    }
    c_full = paper_palette_role("neutral")
    c_non = paper_palette_role("primary")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for gi, (gname, cells) in enumerate(groups.items()):
        for ci, (rel, seed) in enumerate(cells):
            full, nonemit = cell_means(
                ROOT / rel / "slot_stats_trigger.json", ROOT / rel / "completions_trigger.json"
            )
            x = gi + (ci - 1) * 0.09
            ax.scatter(
                [x - 0.04],
                [full],
                color=c_full,
                s=34,
                zorder=3,
                label="full keyed cell (raw mean, incl. emitting rows)"
                if gi == 0 and ci == 0
                else None,
            )
            ax.scatter(
                [x + 0.04],
                [nonemit],
                color=c_non,
                s=34,
                zorder=3,
                label="non-emitting rows only" if gi == 0 and ci == 0 else None,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), fontsize=9)
    ax.set_ylabel("marker log-prob at end of own response,\ntrained − base (nats)")
    ax.set_title(
        "The latent marker signal SURVIVES erasure — and even grows ~3 nats —\n"
        "identically in both arms (points: seeds)",
        fontsize=11,
        pad=12,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    savefig_paper(fig, "issue_570/latent_slot_reads", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_onset_ladder()
    fig_pre_post()
    fig_phase2_mechanism()
    fig_manipulation_check()
    fig_latent_slot()
    print("done")
