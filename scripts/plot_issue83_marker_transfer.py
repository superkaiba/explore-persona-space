"""Plot marker-transfer results for issue #83 (sarcastic source persona).

Generates two figures:
  - Hero (2-panel): sarcastic marker rate across conditions + assistant marker
    rate across conditions (showing no transfer and universal destruction).
  - Supporting: per-condition alignment (Betley) and ARC-C, to confirm EM
    actually worked (cells aren't silently broken).

All numbers are read from eval_results/aim5_marker_transfer_issue83/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval_results" / "aim5_marker_transfer_issue83"


def load_condition_rows(cond: str) -> list[dict]:
    """Read all seed-level run_result JSONs for a condition."""
    rows = []
    for f in sorted((RESULTS_DIR / cond).glob("run_result_*.json")):
        d = json.loads(f.read_text())
        pp = d["marker_eval"]["per_persona"]
        em = d.get("em_sub_run_result") or {}
        rows.append(
            {
                "cond": cond,
                "seed": d["seed"],
                "pp": {p: (v["strict_hits"], v["total"]) for p, v in pp.items()},
                "arc": em.get("capability", {}).get("arc_challenge_logprob"),
                "betley": em.get("alignment", {}).get("overall_mean_aligned"),
            }
        )
    return rows


def pooled_rate(rows: list[dict], persona: str) -> tuple[int, int, float]:
    hits = sum(r["pp"].get(persona, (0, 0))[0] for r in rows)
    n = sum(r["pp"].get(persona, (0, 0))[1] for r in rows)
    return hits, n, hits / n if n else 0.0


def main() -> None:
    set_paper_style("neurips")

    # Load all conditions
    data = {c: load_condition_rows(c) for c in ["c1", "c2", "c3", "c4", "c5"]}
    prep = json.loads((RESULTS_DIR / "prepare_result.json").read_text())

    # Pre-EM baselines (from G0b and G0c gates)
    # G0b: sarcastic+[ZLT] base over 12 personas (pre-EM source: sarcastic 78.21%)
    # G0c: assistant+[ZLT] base over 12 personas (pre-EM source: assistant 41.79%)
    g0b_src = prep["gates"]["G0b"]["sarcastic_strict_rate"]
    g0c_asst = prep["gates"]["G0c"]["assistant_strict_rate"]

    # ---------------- HERO FIGURE ----------------
    # Two panels, shared x-axis conceptually but different metrics.
    # Panel A: sarcastic (source) rate — shows marker destruction
    # Panel B: assistant (target) rate — shows no transfer
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    colors = paper_palette(6)

    # Panel A: sarcastic persona marker rate
    # Bars: C4 (pre-EM, no 2nd stage), C1 (post-EM, coupled), C2 (post-EM, uncoupled),
    #       C3 (post-EM, asst-coupled), C5 (post-benign-SFT)
    labels_a = [
        "C4\n(no 2nd\nstage)",
        "C1\n(EM,\nsarcastic)",
        "C2\n(EM,\nraw)",
        "C3\n(EM,\nasst)",
        "C5\n(benign\nSFT)",
    ]

    # Compute pooled rates + CIs
    rates_a = []
    ns_a = []
    for cond in ["c4", "c1", "c2", "c3", "c5"]:
        h, n, rate = pooled_rate(data[cond], "sarcastic")
        rates_a.append(rate)
        ns_a.append(n)
    rates_a = np.array(rates_a)
    ns_a = np.array(ns_a)
    cis_a = np.array([proportion_ci(r, n) for r, n in zip(rates_a, ns_a)])
    err_lo_a = rates_a - cis_a[:, 0]
    err_hi_a = cis_a[:, 1] - rates_a

    # Use C4 = green (baseline), EM cells = blue shades, C5 = orange
    bar_colors_a = [colors[2], colors[0], colors[0], colors[0], colors[1]]
    bars_a = axes[0].bar(labels_a, rates_a, color=bar_colors_a)
    axes[0].errorbar(
        labels_a, rates_a, yerr=[err_lo_a, err_hi_a], fmt="none", ecolor="black", capsize=3
    )
    for rect, v, hi in zip(bars_a, rates_a, err_hi_a):
        axes[0].text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + hi + 0.02,
            f"{v * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("[ZLT] marker rate (sarcastic persona)")
    axes[0].set_title("(A) Source-persona marker survives?", fontsize=10)
    axes[0].axhline(g0b_src, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].text(
        len(labels_a) - 0.5,
        g0b_src + 0.02,
        f"pre-EM G0b ({g0b_src * 100:.1f}%)",
        ha="right",
        fontsize=7,
        color="gray",
    )

    # Panel B: assistant-persona marker rate
    # The reference bar here: G0c assistant pre-EM rate = 41.79% (when pre-planted in asst base).
    # But the key story is: transfer from sarcastic source => assistant target
    labels_b = [
        "G0c pre-EM\n(asst-coupled\nbase)",
        "C1 post-EM\n(sarcastic\nsource)",
        "C2 post-EM\n(no\ncoupling)",
        "C3 post-EM\n(asst-coupled\nsource)",
        "C5 post-benign\n(sarcastic\nsource)",
    ]
    rates_b = [g0c_asst]
    ns_b = [280]
    for cond in ["c1", "c2", "c3", "c5"]:
        h, n, rate = pooled_rate(data[cond], "assistant")
        rates_b.append(rate)
        ns_b.append(n)
    rates_b = np.array(rates_b)
    ns_b = np.array(ns_b)
    cis_b = np.array([proportion_ci(r, n) for r, n in zip(rates_b, ns_b)])
    err_lo_b = rates_b - cis_b[:, 0]
    err_hi_b = cis_b[:, 1] - rates_b

    # Coloring: G0c pre-EM = green (baseline), post-EM bars = blue, benign = orange
    bar_colors_b = [colors[2], colors[0], colors[0], colors[0], colors[1]]
    bars_b = axes[1].bar(labels_b, rates_b, color=bar_colors_b)
    axes[1].errorbar(
        labels_b, rates_b, yerr=[err_lo_b, err_hi_b], fmt="none", ecolor="black", capsize=3
    )
    for rect, v, hi in zip(bars_b, rates_b, err_hi_b):
        axes[1].text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + hi + 0.02,
            f"{v * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("[ZLT] marker rate (assistant persona)")
    axes[1].set_title("(B) Did marker transfer to assistant?", fontsize=10)

    # Smaller x-tick labels to avoid crowding
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=0)

    fig.suptitle(
        "Sarcastic-source [ZLT] marker: destruction without transfer (Qwen-2.5-7B-Instruct, 3 seeds, N=840/cond)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "aim5_issue83/hero_marker_destruction", dir="figures/")
    plt.close(fig)

    # ---------------- SUPPORTING FIGURE: alignment + ARC ----------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.2, 3.2))

    conds = ["c1", "c2", "c3", "c5"]
    cond_labels = ["C1 (EM, sarc)", "C2 (EM, raw)", "C3 (EM, asst)", "C5 (benign)"]

    # Alignment means ± std across seeds
    bet_means = []
    bet_stds = []
    arc_means = []
    arc_stds = []
    for c in conds:
        bets = [r["betley"] for r in data[c] if r["betley"] is not None]
        arcs = [r["arc"] for r in data[c] if r["arc"] is not None]
        bet_means.append(np.mean(bets))
        bet_stds.append(np.std(bets, ddof=1) / np.sqrt(len(bets)))  # SE
        arc_means.append(np.mean(arcs))
        arc_stds.append(np.std(arcs, ddof=1) / np.sqrt(len(arcs)))

    cc = paper_palette(4)
    axes2[0].bar(cond_labels, bet_means, yerr=bet_stds, color=cc, capsize=3)
    axes2[0].set_ylabel("Betley alignment (0-100)")
    axes2[0].set_ylim(0, 100)
    axes2[0].axhline(40, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    axes2[0].text(3.5, 41, "G4 threshold (40)", ha="right", fontsize=7, color="red")
    axes2[0].set_title("(A) EM actually broke alignment", fontsize=10)
    for i, (m, s) in enumerate(zip(bet_means, bet_stds)):
        axes2[0].text(i, m + s + 2, f"{m:.1f}", ha="center", fontsize=8)
    add_direction_arrow(axes2[0], axis="y", direction="up")

    axes2[1].bar(cond_labels, arc_means, yerr=arc_stds, color=cc, capsize=3)
    axes2[1].set_ylabel("ARC-C accuracy")
    axes2[1].set_ylim(0.6, 1.0)
    axes2[1].set_title("(B) Capability preserved", fontsize=10)
    for i, (m, s) in enumerate(zip(arc_means, arc_stds)):
        axes2[1].text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8)
    add_direction_arrow(axes2[1], axis="y", direction="up")

    for ax in axes2:
        ax.tick_params(axis="x", labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=0)

    fig2.suptitle(
        "EM succeeded on alignment, preserved ARC-C (3 seeds per condition)",
        fontsize=10,
        y=1.02,
    )
    fig2.tight_layout()
    savefig_paper(fig2, "aim5_issue83/supporting_alignment_capability", dir="figures/")
    plt.close(fig2)

    print("Wrote figures/aim5_issue83/hero_marker_destruction.{png,pdf}")
    print("Wrote figures/aim5_issue83/supporting_alignment_capability.{png,pdf}")


if __name__ == "__main__":
    main()
