"""Plots for issue #475 clean-result.

Two figures:
1. hero_install_collapse.png — grouped bars of trained−base logP(marker) at the
   TRIGGER cell (T_plus), by arm × phase. Tells the install→collapse story end
   to end: plain and distilled_cot both install near the asymptote then drop
   ~10 nats after one benign-medical SFT epoch; visible_cot never installs.

2. cell_breakdown.png — grouped bars of trained−base logP(marker) at all four
   eval cells (T_plus / T_minus / NEG_doctor / NEG_default_other), one panel
   per arm, phase1 and phase2 side-by-side. Shows the weak trigger-gating
   story: NEG_default_other tracks T_plus at phase1 for plain & distilled_cot
   (the install is largely a context-independent slot bias), and the same
   pattern survives benign SFT (only attenuates).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ARMS = ["plain", "visible_cot", "distilled_cot"]
ARM_LABELS = {
    "plain": "Plain (no scaffold)",
    "visible_cot": "Visible CoT scaffold",
    "distilled_cot": "Distilled CoT scaffold",
}
PHASES = ["phase1", "phase2"]
PHASE_LABELS = {"phase1": "After install (Phase 1)", "phase2": "After benign SFT (Phase 2)"}
CELLS = ["T_plus", "T_minus", "NEG_doctor", "NEG_default_other"]
CELL_LABELS = {
    "T_plus": "Default + key, in-domain prompts",
    "T_minus": "Default, no key, in-domain prompts",
    "NEG_doctor": "Medical doctor + key",
    "NEG_default_other": "Default + key, held-out OOD prompts",
}


def load_eval_dir(eval_root: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """Returns {(arm, phase): {cell: cell_summary_dict}}."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for arm in ARMS:
        for ph in PHASES:
            p = eval_root / arm / ph / "run_summary.json"
            data = json.loads(p.read_text())
            out[(arm, ph)] = data["cells"]
    return out


def load_per_completion(eval_root: Path, arm: str, ph: str, cell: str) -> np.ndarray:
    """Per-completion trained−base logP array (length n)."""
    trained = json.loads((eval_root / arm / ph / f"trained_logp_{cell}.json").read_text())
    base = json.loads((eval_root / arm / ph / f"base_logp_{cell}.json").read_text())
    return np.asarray(trained) - np.asarray(base)


def plot_hero(eval_root: Path, out_dir: Path) -> Path:
    """Grouped bars: Δ logP at T_plus by arm × phase."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    cells = load_eval_dir(eval_root)

    arm_idx = np.arange(len(ARMS))
    width = 0.36
    color_phase1 = paper_palette_role("primary")
    color_phase2 = paper_palette_role("baseline")

    medians = {ph: [cells[(a, ph)]["T_plus"]["delta_logp_median"] for a in ARMS] for ph in PHASES}
    # Within-cell error: bootstrap median 95 CI from the per-completion array
    rng = np.random.default_rng(42)

    def boot_ci(arr: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        meds = np.median(arr[idx], axis=1)
        return np.percentile(meds, [2.5, 97.5]).tolist()

    err_lo = {ph: [] for ph in PHASES}
    err_hi = {ph: [] for ph in PHASES}
    for ph in PHASES:
        for a in ARMS:
            arr = load_per_completion(eval_root, a, ph, "T_plus")
            lo, hi = boot_ci(arr)
            m = float(np.median(arr))
            err_lo[ph].append(max(m - lo, 0.0))
            err_hi[ph].append(max(hi - m, 0.0))

    bars1 = ax.bar(
        arm_idx - width / 2,
        medians["phase1"],
        width,
        color=color_phase1,
        label=PHASE_LABELS["phase1"],
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        arm_idx + width / 2,
        medians["phase2"],
        width,
        color=color_phase2,
        label=PHASE_LABELS["phase2"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.errorbar(
        arm_idx - width / 2,
        medians["phase1"],
        yerr=[err_lo["phase1"], err_hi["phase1"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=0.8,
    )
    ax.errorbar(
        arm_idx + width / 2,
        medians["phase2"],
        yerr=[err_lo["phase2"], err_hi["phase2"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=0.8,
    )

    for bar, v in zip(bars1, medians["phase1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.4,
            f"{v:+.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, v in zip(bars2, medians["phase2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.4,
            f"{v:+.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(0.0, color="grey", lw=0.7, linestyle=":")
    ax.set_xticks(arm_idx)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARMS])
    ax.set_ylabel("Δ log P( ※ ), trained − base (nats)")
    ax.set_xlabel("Install variant")
    ax.set_title(
        "Install-then-collapse at the trigger cell (Qwen3.5-27B, n=200/cell, seed=42)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=10,
    )
    # Legend below the title block so it doesn't collide with the bar-top numbers.
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=9,
    )
    ax.set_ylim(-1.5, 24)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return savefig_paper(fig, "issue_475/hero_install_collapse", dir=str(out_dir))


def plot_cell_breakdown(eval_root: Path, out_dir: Path) -> Path:
    """One panel per arm; bars = the 4 cells × 2 phases."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.5), sharey=True)

    cells = load_eval_dir(eval_root)
    cell_idx = np.arange(len(CELLS))
    width = 0.36
    color_phase1 = paper_palette_role("primary")
    color_phase2 = paper_palette_role("baseline")

    for ax, arm in zip(axes, ARMS):
        med1 = [cells[(arm, "phase1")][c]["delta_logp_median"] for c in CELLS]
        med2 = [cells[(arm, "phase2")][c]["delta_logp_median"] for c in CELLS]
        b1 = ax.bar(
            cell_idx - width / 2,
            med1,
            width,
            color=color_phase1,
            label=PHASE_LABELS["phase1"],
            edgecolor="black",
            linewidth=0.5,
        )
        b2 = ax.bar(
            cell_idx + width / 2,
            med2,
            width,
            color=color_phase2,
            label=PHASE_LABELS["phase2"],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, v in zip(b1, med1):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.5,
                f"{v:+.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar, v in zip(b2, med2):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.5,
                f"{v:+.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.axhline(0.0, color="grey", lw=0.7, linestyle=":")
        ax.set_xticks(cell_idx)
        ax.set_xticklabels(
            [CELL_LABELS[c] for c in CELLS],
            rotation=22,
            ha="right",
            fontsize=8,
        )
        ax.set_title(ARM_LABELS[arm], fontsize=11, fontweight="semibold", loc="left", pad=8)
        ax.set_ylim(-1.5, 28)

    axes[0].set_ylabel("Δ log P( ※ ), trained − base (nats)")
    # Put legend below the panels so it doesn't collide with the "+20" label.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    return savefig_paper(fig, "issue_475/cell_breakdown", dir=str(out_dir))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    eval_root = repo_root / "eval_results" / "issue_475"
    out_dir = repo_root / "figures"
    hero = plot_hero(eval_root, out_dir)
    cell = plot_cell_breakdown(eval_root, out_dir)
    print(f"wrote: {hero}")
    print(f"wrote: {cell}")


if __name__ == "__main__":
    main()
