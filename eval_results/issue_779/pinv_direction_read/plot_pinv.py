"""Figures for the #779 pinv-direction-read inline free-analysis.

Fig 1: headline grouped-bar — within-condition Pearson r for the three prompt-side
       directions {pv_raw = r_B, transpose = M^T r_B, pinv = M^+ r_B (pre-reg rank)}
       per trait x elicitation mode, bootstrap CI whiskers + random-direction null
       p95|r| reference tick.
Fig 2: pinv truncation-rank sweep — within-condition r vs SVD rank, per trait, one
       panel per mode, with pv_raw + transpose horizontal references and the
       random-direction null p95|r| band shaded (shows the rank-contingency +
       full-rank ill-conditioning collapse).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

set_paper_style()

JSON = (
    PROJECT_ROOT / "eval_results" / "issue_779" / "pinv_direction_read" / "pinv_direction_read.json"
)
FIGDIR = PROJECT_ROOT / "figures" / "issue_779"
d = json.load(open(JSON))
traits = list(d["traits"].keys())
TRAIT_LABEL = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
MODE_LABEL = {"system": "System-prompt", "many_shot": "Many-shot"}
C_RAW = paper_palette_role("baseline")
C_TR = paper_palette_role("control")
C_PINV = paper_palette_role("primary")
C_NULL = paper_palette_role("neutral")


# ── Fig 1: headline grouped bar ───────────────────────────────────────────────
def fig_headline():
    modes = ["system", "many_shot"]
    groups = [(t, m) for t in traits for m in modes]  # 6 groups
    x = np.arange(len(groups))
    w = 0.26
    methods = [
        ("pv_raw", "Raw persona vector  (w = r_B)", C_RAW, -w),
        ("transpose_MTrb", "Transpose  (w = Mᵀr_B)", C_TR, 0.0),
        ("pinv_headline", "Pseudoinverse  (w = M⁺r_B)", C_PINV, +w),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for key, label, col, off in methods:
        pts, los, his = [], [], []
        for t, m in groups:
            mm = d["traits"][t]["methods"][key][m]
            pts.append(mm["point"])
            los.append(mm["point"] - mm["lo"])
            his.append(mm["hi"] - mm["point"])
        ax.bar(
            x + off,
            pts,
            w,
            color=col,
            label=label,
            yerr=[los, his],
            capsize=2.5,
            error_kw={"lw": 1.0, "alpha": 0.8},
        )
    # random-direction null p95|r| as a short grey reference tick per group.
    for i, (t, m) in enumerate(groups):
        p95 = d["traits"][t]["null_random_direction"][m]["p95_abs"]
        ax.plot(
            [i - 1.6 * w, i + 1.6 * w],
            [p95, p95],
            color=C_NULL,
            lw=1.4,
            ls=(0, (3, 2)),
            zorder=5,
            label="Random-direction null (95th pct |r|)" if i == 0 else None,
        )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TRAIT_LABEL[t]}\n{MODE_LABEL[m]}" for t, m in groups])
    ax.set_ylabel("Within-condition Pearson r\n(monitor projection vs judged trait score)")
    ax.set_title(
        "Prompt-side trait monitor: min-norm preimage vs raw persona vector (Arm A, LMSYS map)"
    )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5, ncol=2)
    ax.set_ylim(-0.25, 0.85)
    fig.tight_layout()
    paths = savefig_paper(fig, "pinv_headline_comparison", dir=FIGDIR)
    plt.close(fig)
    return paths


# ── Fig 2: rank sweep ─────────────────────────────────────────────────────────
def fig_rank_sweep():
    modes = ["system", "many_shot"]
    trait_cols = {"evil": C_PINV, "sycophancy": C_TR, "hallucination": C_RAW}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    for ax, mode in zip(axes, modes):
        for t in traits:
            tt = d["traits"][t]
            sweep = tt["pinv_rank_sweep"]
            ranks = [s["rank"] for s in sweep.values()]
            rs = [s["within_cond_point"][mode] for s in sweep.values()]
            col = trait_cols[t]
            ax.plot(
                ranks, rs, "-o", color=col, ms=3.5, lw=1.5, label=f"{TRAIT_LABEL[t]}  pinv(rank)"
            )
            ax.axhline(tt["methods"]["pv_raw"][mode]["point"], color=col, ls=":", lw=1.1, alpha=0.8)
            ax.axhline(
                tt["methods"]["transpose_MTrb"][mode]["point"],
                color=col,
                ls="--",
                lw=0.9,
                alpha=0.55,
            )
        # shaded null p95|r| band (max across traits, symmetric).
        p95max = max(d["traits"][t]["null_random_direction"][mode]["p95_abs"] for t in traits)
        ax.axhspan(
            -p95max,
            p95max,
            color=C_NULL,
            alpha=0.10,
            lw=0,
            label="Random-direction null (±95th pct |r|)",
        )
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Pseudoinverse SVD truncation rank")
        ax.set_title(MODE_LABEL[mode])
    axes[0].set_ylabel("Within-condition Pearson r")
    axes[0].legend(loc="lower center", fontsize=7.5, framealpha=0.9)
    fig.suptitle(
        "Pseudoinverse monitor is rank-contingent (dotted = raw pv, dashed = transpose; full rank collapses)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = savefig_paper(fig, "pinv_rank_sweep", dir=FIGDIR)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    p1 = fig_headline()
    p2 = fig_rank_sweep()
    print("wrote:", {k: str(v) for k, v in {**p1, **p2}.items()})
