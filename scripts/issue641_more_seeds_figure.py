"""Issue #641 follow-up `identity-conflict-more-seeds` clean-result figure.

Re-fold of the kindergarten-teacher (identity-conflict source) vs matched-neutral
local-historian Arm-B pairing, extended from 2 to 8 seeds at the matched dose
(step 100). One figure: per-seed step-100 install rates (left) + the
teacher-minus-neutral decision CI, N=2 vs N=8 (right, showing the CI shrinkage).

All numbers re-extracted from the committed eval JSONs via the issue_641 stats
module at write time (no hardcoded values).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue_641 import stats

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO / "eval_results/issue_641/dose_curves"
NEW = REPO / "eval_results/issue_641/identity-conflict-more-seeds/dose_curves"
OUTDIR = "figures/"

PARENT_SEEDS = [42, 1042]
NEW_SEEDS = [1, 7, 123, 2024, 31337, 98765]
ALL_SEEDS = PARENT_SEEDS + NEW_SEEDS


def _load(root: Path, src: str, seed: int, step: int = 100) -> list[dict]:
    f = root / f"{src}_seed{seed}_step{step}" / f"completions__{src}__seed{seed}__step{step}.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def _root_for(seed: int) -> Path:
    return PARENT if seed in PARENT_SEEDS else NEW


def main() -> None:
    set_paper_style("blog")

    c_teach = paper_palette_role("primary")
    c_neut = paper_palette_role("control")

    # Per-seed rates.
    t_rates = {
        s: stats.cell_rate_from_records(_load(_root_for(s), "sp_teacher_ho", s)) for s in ALL_SEEDS
    }
    n_rates = {
        s: stats.cell_rate_from_records(_load(_root_for(s), "local_historian", s))
        for s in ALL_SEEDS
    }

    # Pooled records + 8-seed and 2-seed deltas.
    t_all = [r for s in ALL_SEEDS for r in _load(_root_for(s), "sp_teacher_ho", s)]
    n_all = [r for s in ALL_SEEDS for r in _load(_root_for(s), "local_historian", s)]
    res8 = stats.bootstrap_armB_delta(t_all, n_all, n_boot=5000, seed=42)

    t2 = [r for s in PARENT_SEEDS for r in _load(PARENT, "sp_teacher_ho", s)]
    n2 = [r for s in PARENT_SEEDS for r in _load(PARENT, "local_historian", s)]
    res2 = stats.bootstrap_armB_delta(t2, n2, n_boot=5000, seed=42)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.6, 4.0), width_ratios=[1.35, 1.0])

    # --- Left: per-seed teacher vs neutral, paired ---
    xs = np.arange(len(ALL_SEEDS))
    w = 0.36
    ax0.bar(
        xs - w / 2, [t_rates[s] for s in ALL_SEEDS], w, color=c_teach, label="kindergarten teacher"
    )
    ax0.bar(xs + w / 2, [n_rates[s] for s in ALL_SEEDS], w, color=c_neut, label="matched neutral")
    ax0.set_xticks(xs)
    ax0.set_xticklabels([str(s) for s in ALL_SEEDS], fontsize=7.5, rotation=45)
    ax0.set_ylim(0.0, 0.85)
    ax0.set_xlabel("training seed (first 2 = parent, last 6 = new)")
    ax0.set_ylabel("EM install rate at step 100")
    ax0.axvline(1.5, color="#bbb", ls=":", lw=1.0)
    ax0.legend(loc="upper right", fontsize=7.5)
    ax0.set_title("per-seed install (8 seeds)", fontsize=10)

    # --- Right: decision CI, N=2 vs N=8 ---
    rows = [
        ("2 seeds", res2["delta_L"], res2["ci95"], "#9aa"),
        ("8 seeds", res8["delta_L"], res8["ci95"], c_teach),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lbl, d, (lo, hi), color) in zip(ys, rows, strict=True):
        ax1.plot([lo, hi], [y, y], color=color, lw=3.0, solid_capstyle="round")
        ax1.plot([d], [y], "o", color=color, ms=8, zorder=5)
        ax1.text(hi + 0.012, y, f"{lbl}\n[{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=7.6)
    ax1.axvspan(-0.10, 0.10, color="#cfe8cf", alpha=0.5, zorder=0)
    ax1.axvline(0.0, color="#888", ls="--", lw=1.0)
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_yticks([])
    ax1.set_xlim(-0.40, 0.45)
    ax1.set_xlabel("teacher − neutral install gap (step 100)")
    ax1.set_title("decision CI: still AMBIGUOUS, now bounded", fontsize=8.8)
    ax1.text(
        0.0,
        -0.45,
        "shaded = ±0.10 equivalence band",
        ha="center",
        fontsize=6.8,
        color="#555",
    )

    fig.suptitle(
        "More seeds bound the identity-conflict gap but do not resolve it",
        fontsize=11.5,
        fontweight="semibold",
        x=0.02,
        ha="left",
        y=1.02,
    )
    fig.text(
        0.02,
        0.965,
        "8-seed teacher − neutral gap is −0.07, CI [−0.24, +0.11] (39% narrower than 2 seeds) — "
        "still crosses both ±0.10 boundaries, so the verdict stays AMBIGUOUS.",
        fontsize=8.0,
        ha="left",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "issue_641/identity-conflict-more-seeds/armB_8seed_delta", dir=OUTDIR)
    plt.close(fig)

    print("delta8", res8["delta_L"], res8["ci95"])
    print("delta2", res2["delta_L"], res2["ci95"])


if __name__ == "__main__":
    main()
