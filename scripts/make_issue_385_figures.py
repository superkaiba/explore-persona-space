"""Generate the three figures for task #385 clean-result body.

Hero: emergence dynamics (panel mean ± CI, with overlay of close-cosine vs
    far-cosine bystander curves).
Supporting 1: per-step Spearman trajectory for cosine + JS predictors.
Supporting 2: scatter of JS-divergence vs plateau emission rate, annotated
    with bystander names, showing the geometric ordering at the plateau.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
DATA = ROOT / ".claude/worktrees/issue-385/eval_results/issue_385"
OUT_DIR = "issue_385"
N_PER_CELL = 160  # 20 prompts x 8 completions

PRETTY = {
    "cybersec_consultant": "Cybersec consultant",
    "pentester": "Pentester",
    "software_engineer": "Software engineer",
    "data_scientist": "Data scientist",
    "helpful_assistant": "Helpful assistant",
    "private_investigator": "Private investigator",
    "medical_doctor": "Medical doctor",
    "kindergarten_teacher": "Kindergarten teacher",
    "poet": "Poet",
    "villain": "Villain",
    "navy_seal": "Navy SEAL",
    "army_medic": "Army medic",
    "surgeon": "Surgeon",
    "paramedic": "Paramedic",
    "police_officer": "Police officer",
    "florist": "Florist",
    "comedian": "Comedian",
    "french_person": "French person",
    "no_persona": "No system prompt",
    "fammate_task_1": "Task framing: biology tutor",
    "fammate_task_2": "Task framing: email drafter",
    "fammate_instruction_1": "Instruction: five bullets",
    "fammate_instruction_2": "Instruction: single paragraph",
    "fammate_context_1": "Context: patient intake",
    "fammate_context_2": "Context: customer complaint",
    "fammate_format_1": "Format: YAML output",
    "fammate_format_2": "Format: markdown table",
}


def load_data():
    with open(DATA / "seed42/summary.json") as f:
        summary = json.load(f)
    with open(DATA / "predictors_base.json") as f:
        pred_base = json.load(f)
    # Source-persona (librarian) per-checkpoint emission rate, re-eval added 2026-05-26.
    # See scripts/eval_marker_spread_source_only.py.
    source_path = Path("eval_results/issue_385/seed42/source_rate.json")
    if not source_path.exists():
        source_path = DATA / "seed42/source_rate.json"
    with open(source_path) as f:
        source = json.load(f)
    return summary, pred_base, source


def figure_hero(summary, pred_base, source):
    """Emergence dynamics."""
    set_paper_style("blog")

    bystanders = summary["bystanders"]
    steps = summary["metadata"]["steps_completed"]
    rate_at = {row["step"]: row["per_bystander_rate"] for row in summary["rows"]}
    cosine_base = {b: pred_base["cosine_to_source"][b] for b in bystanders}

    means, lo, hi = [], [], []
    for s in steps:
        rates = [rate_at[s][b] for b in bystanders]
        p_bar = float(np.mean(rates))
        n_total = 160 * len(bystanders)
        k_total = int(round(sum(r * 160 for r in rates)))
        l, h = proportion_ci(k_total / n_total, n_total)
        means.append(p_bar)
        lo.append(l)
        hi.append(h)

    source_by_step = {row["step"]: row["rate"] for row in source["per_step"]}
    source_y = [source_by_step.get(s, np.nan) for s in steps]

    closest = sorted(bystanders, key=lambda b: -cosine_base[b])[:3]
    farthest = sorted(bystanders, key=lambda b: cosine_base[b])[:3]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=False)

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")
    accent = paper_palette_role("accent")

    ax.plot(
        steps,
        source_y,
        color=accent,
        linewidth=2.4,
        label="Source persona (librarian)",
        zorder=6,
    )
    ax.scatter(steps, source_y, color=accent, s=24, zorder=7)

    ax.fill_between(steps, lo, hi, alpha=0.18, color=primary, linewidth=0)
    ax.plot(
        steps, means, color=primary, linewidth=2.4, label="Panel mean (27 bystanders)", zorder=4
    )
    ax.scatter(steps, means, color=primary, s=24, zorder=5)

    for b in closest:
        y = [rate_at[s][b] for s in steps]
        ax.plot(steps, y, color=baseline, linewidth=1.0, alpha=0.55, linestyle="-", zorder=2)
    for b in farthest:
        y = [rate_at[s][b] for s in steps]
        ax.plot(steps, y, color=neutral, linewidth=1.0, alpha=0.55, linestyle="--", zorder=2)

    ax.plot([], [], color=baseline, linewidth=1.0, label="Closest 3 bystanders (cosine to source)")
    ax.plot(
        [],
        [],
        color=neutral,
        linewidth=1.0,
        linestyle="--",
        label="Farthest 3 bystanders (cosine to source)",
    )

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 75, 100, 200, 400, 800, 1600])
    ax.set_xticklabels(["5", "10", "25", "50", "75", "100", "200", "400", "800", "1600"])
    ax.set_xlabel("LoRA training step (log scale)")
    ax.set_ylabel("[ZLT] marker emission rate (higher = more leakage)")
    ax.set_ylim(-0.02, 1.02)

    ax.set_title(
        "Marker emerges at step 75 for source AND bystanders; source saturates near 1.0",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=18,
    )
    ax.annotate(
        "Source = librarian; 27 bystander prompts, n=160 per cell. "
        "Shaded band = 95% pooled binomial CI on bystander mean.",
        xy=(0.0, 1.0),
        xytext=(0, 4),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, bbox_to_anchor=(0.02, 0.95))
    fig.tight_layout(pad=1.2)
    savefig_paper(fig, f"{OUT_DIR}/hero_emergence_dynamics", dir="figures/")
    plt.close(fig)


def figure_per_step_spearman(summary, pred_base):
    set_paper_style("blog")

    bystanders = summary["bystanders"]
    steps = summary["metadata"]["steps_completed"]
    rate_at = {row["step"]: row["per_bystander_rate"] for row in summary["rows"]}
    cosine_base = np.array([pred_base["cosine_to_source"][b] for b in bystanders])
    js_base = np.array([pred_base["js_to_source"][b] for b in bystanders])

    rho_cos, rho_js = [], []
    for s in steps:
        rates = np.array([rate_at[s][b] for b in bystanders])
        if rates.std() == 0:
            rho_cos.append(np.nan)
            rho_js.append(np.nan)
            continue
        rc, _ = spearmanr(cosine_base, rates)
        rj, _ = spearmanr(js_base, rates)
        rho_cos.append(rc)
        rho_js.append(rj)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=False)

    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")

    valid = [(s, r) for s, r in zip(steps, rho_cos) if not np.isnan(r)]
    s_v = [v[0] for v in valid]
    r_v = [v[1] for v in valid]
    ax.plot(
        s_v,
        r_v,
        "-o",
        color=primary,
        linewidth=2.0,
        markersize=7,
        label="L20 cosine (closer = more leakage)",
    )

    valid_js = [(s, -r) for s, r in zip(steps, rho_js) if not np.isnan(r)]
    s_jv = [v[0] for v in valid_js]
    r_jv = [v[1] for v in valid_js]
    ax.plot(
        s_jv,
        r_jv,
        "-s",
        color=accent,
        linewidth=2.0,
        markersize=7,
        label="JS-divergence, sign flipped (closer = more leakage)",
    )

    ax.axvspan(4, 70, alpha=0.10, color="#cfcfcf", zorder=0)
    ax.text(15, 0.93, "Zero-emission region", fontsize=9, color="#666666", ha="center", va="top")

    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle="-", zorder=1)

    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 75, 100, 200, 400, 800, 1600])
    ax.set_xticklabels(["5", "10", "25", "50", "75", "100", "200", "400", "800", "1600"])
    ax.set_xlabel("LoRA training step (log scale)")
    ax.set_ylabel("Spearman rho (higher = better order match)")
    ax.set_ylim(-0.15, 1.0)

    ax.set_title(
        "Both base-model predictors track per-checkpoint bystander order",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=18,
    )
    ax.annotate(
        "Per-step Spearman rho between predictor and rate across 27 bystanders.",
        xy=(0.0, 1.0),
        xytext=(0, 4),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(pad=1.2)
    savefig_paper(fig, f"{OUT_DIR}/per_step_spearman", dir="figures/")
    plt.close(fig)


def figure_scatter_plateau(summary, pred_base):
    set_paper_style("blog")

    bystanders = summary["bystanders"]
    steps = summary["metadata"]["steps_completed"]
    plateau_steps = [s for s in steps if s >= 200]
    rate_at = {row["step"]: row["per_bystander_rate"] for row in summary["rows"]}

    plateau_rate = {b: float(np.mean([rate_at[s][b] for s in plateau_steps])) for b in bystanders}
    js_base = {b: pred_base["js_to_source"][b] for b in bystanders}

    x = np.array([js_base[b] for b in bystanders])
    y = np.array([plateau_rate[b] for b in bystanders])
    rho, p = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=False)

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")

    def sustained_5(b):
        ser = [(s, rate_at[s][b]) for s in steps]
        for i, (s, r) in enumerate(ser):
            if r >= 0.05:
                if i + 1 == len(ser):
                    return True
                if ser[i + 1][1] >= 0.05:
                    return True
        return False

    crossed = [sustained_5(b) for b in bystanders]
    x_c = np.array([x[i] for i in range(len(bystanders)) if crossed[i]])
    y_c = np.array([y[i] for i in range(len(bystanders)) if crossed[i]])
    x_n = np.array([x[i] for i in range(len(bystanders)) if not crossed[i]])
    y_n = np.array([y[i] for i in range(len(bystanders)) if not crossed[i]])

    ax.scatter(
        x_c,
        y_c,
        s=70,
        color=primary,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Crossed two-consecutive-5%",
    )
    ax.scatter(
        x_n,
        y_n,
        s=70,
        color=neutral,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Never crossed two-consecutive-5%",
    )

    label_set = {
        "florist",
        "paramedic",
        "private_investigator",
        "villain",
        "fammate_instruction_1",
        "fammate_format_2",
        "comedian",
        "poet",
    }
    for i, b in enumerate(bystanders):
        if b in label_set:
            ax.annotate(
                PRETTY.get(b, b),
                (x[i], y[i]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color="#444",
            )

    ax.set_xlabel("Base-model JS-divergence to librarian source (higher = farther)")
    ax.set_ylabel("Mean [ZLT] emission rate over plateau (step >= 200)")
    ax.set_ylim(-0.02, 0.55)
    ax.set_xlim(-0.02, 0.75)

    ax.text(
        0.62,
        0.95,
        f"Spearman rho = {rho:.2f}\np = {p:.4f}\nN = 27",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8),
    )

    ax.set_title(
        "Far-from-source bystanders never cross the emission threshold",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=18,
    )
    ax.annotate(
        "Each point = one bystander system prompt; y = mean rate over plateau (step 200-1600).",
        xy=(0.0, 1.0),
        xytext=(0, 4),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(0.97, 0.78))
    fig.tight_layout(pad=1.2)
    savefig_paper(fig, f"{OUT_DIR}/scatter_plateau", dir="figures/")
    plt.close(fig)


def main():
    summary, pred_base, source = load_data()
    figure_hero(summary, pred_base, source)
    figure_per_step_spearman(summary, pred_base)
    figure_scatter_plateau(summary, pred_base)
    print("Saved 3 figures under figures/issue_385/")


if __name__ == "__main__":
    main()
