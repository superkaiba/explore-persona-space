"""Figures for the #533 logit-margin re-read follow-up.

Reads the 540 per-cell margin JSONs at
``eval_results/issue_533/logit-margin-reread/per_cell/`` and writes:

1. ``margin_paired_gap_trajectory.{png,pdf,meta.json}`` — hero.
   2 panels (pirate / villain); each panel plots the paired role-vs-system
   gap d = (sys arm) − (role arm) at the WRONG-persona probe across steps
   {5, 10, 18, 30, 60, 120}, in two DVs side-by-side:
     - blue solid: trained log P (the #547 DV, replicated from per_cell)
     - red solid: trained − base EOS-logit margin (the spec's DV)
   plain−role contrast as the primary trace per DV; padded−role as dashed
   ghost in the same color.

2. ``margin_paired_gap_per_seed_scatter.{png,pdf,meta.json}`` — raw
   sample. 2 panels (pirate / villain); each panel shows the per-seed
   paired d (margin trained − base) at each step, plain and padded
   contrasts, as raw scatter (no bootstrap aggregation).

3. ``margin_per_arm_trajectory.{png,pdf,meta.json}`` — supporting
   per-arm trained − base margin at the wrong-persona probe (the
   per-arm levels the contrasts ride on).

Hero claim: in trained log P space, villain's late-step (s=120) gap
narrows to near-zero (#547's binding ambiguity between encodings-
converge and floor-compression); in trained − base EOS-margin space the
gap at s=120 flips POSITIVE on 3 of 4 (persona, contrast) cells (villain
plain CI clears zero) — i.e. the role encoding ends up with LOWER
wrong-persona marker mass than the system arms by 3.2 epochs once the
encoding-specific base-model logZ shift is accounted for. The
floor-compression and encodings-converge readings BOTH miss; the
right reading is that the early negative gap and the late positive gap
reflect dynamics in the EOS-vs-marker logit balance under each encoding
that DV choice exposes.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_CELL_DIR = REPO_ROOT / "eval_results" / "issue_533" / "logit-margin-reread" / "per_cell"
FIG_DIR = REPO_ROOT / "figures" / "issue_533"

STEPS = [5, 10, 18, 30, 60, 120]
SEEDS = [7, 21, 42, 137, 1337]
PERSONAS = ["pirate", "villain"]
WRONG = {"pirate": "villain", "villain": "pirate"}
N_BOOT = 10_000


def _load(arm: str, seed: int, persona: str, s: int, probe: str) -> dict:
    p = PER_CELL_DIR / f"{arm}_seed{seed}_cn_{persona}_s{s}__{probe}.json"
    return json.loads(p.read_text())


def _per_q_margin_tb(d: dict) -> np.ndarray:
    """Δ(z_marker − z_eos) trained − base, per question."""
    g_zm = np.asarray(d["g_z_marker_per_q"], dtype=float)
    g_ze = np.asarray(d["g_z_eos_per_q"], dtype=float)
    b_zm = np.asarray(d["b_z_marker_per_q"], dtype=float)
    b_ze = np.asarray(d["b_z_eos_per_q"], dtype=float)
    return (g_zm - g_ze) - (b_zm - b_ze)


def _per_q_margin_trained(d: dict) -> np.ndarray:
    """(z_marker − z_eos) on the trained side only — what tracks log P at saturation."""
    g_zm = np.asarray(d["g_z_marker_per_q"], dtype=float)
    g_ze = np.asarray(d["g_z_eos_per_q"], dtype=float)
    return g_zm - g_ze


def _per_q_logp_trained(d: dict) -> np.ndarray:
    return np.asarray(d["g_logp_per_q"], dtype=float)


def _bootstrap_ci(
    per_seed_d: list[float], n_boot: int = N_BOOT, rng_seed: int = 42
) -> tuple[float, float, float]:
    """Per-seed-paired bootstrap of d_per_seed → (point, ci_lo, ci_hi)."""
    arr = np.asarray(per_seed_d, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = arr[idx].mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _wrong_probe(arm: str, wrong_persona: str) -> str:
    return f"role_{wrong_persona}" if arm == "role" else f"system_{wrong_persona}"


def _paired_d(
    persona: str, s: int, sys_arm: str, dv: str
) -> tuple[float, float, float, list[float]]:
    """Per-seed-paired bootstrap of d = (sys_arm) − (role) at wrong-persona probe.

    dv ∈ {"logp", "margin_tb", "margin_trained"}.
    Returns (point, ci_lo, ci_hi, d_per_seed).
    """
    fn = {
        "logp": _per_q_logp_trained,
        "margin_tb": _per_q_margin_tb,
        "margin_trained": _per_q_margin_trained,
    }[dv]
    wrong = WRONG[persona]
    per_seed = []
    for sd in SEEDS:
        r = _load("role", sd, persona, s, _wrong_probe("role", wrong))
        sp = _load(sys_arm, sd, persona, s, _wrong_probe(sys_arm, wrong))
        per_seed.append(float(fn(sp).mean() - fn(r).mean()))
    point, lo, hi = _bootstrap_ci(per_seed)
    return point, lo, hi, per_seed


def _per_arm_level(persona: str, s: int, arm: str) -> tuple[float, float, float]:
    """Wrong-persona Δmargin trained − base per arm, mean across seeds (50 questions each)."""
    wrong = WRONG[persona]
    probe = _wrong_probe(arm, wrong)
    per_seed_means = []
    for sd in SEEDS:
        d = _load(arm, sd, persona, s, probe)
        per_seed_means.append(float(_per_q_margin_tb(d).mean()))
    arr = np.asarray(per_seed_means)
    point, lo, hi = _bootstrap_ci(per_seed_means)
    return point, lo, hi


def figure_hero() -> None:
    """Hero: paired role-vs-system gap trajectory, log P vs trained − base margin, two panels per persona."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)

    color_logp = paper_palette_role("baseline")  # warm orange
    color_margin = paper_palette_role("primary")  # blue
    contrasts = [("plain", "system_plain", "-"), ("padded", "system_padded", "--")]

    for ax_idx, persona in enumerate(PERSONAS):
        ax = axes[ax_idx]
        for contrast_label, sys_arm, ls in contrasts:
            logp_pts, logp_lo, logp_hi = [], [], []
            marg_pts, marg_lo, marg_hi = [], [], []
            for s in STEPS:
                p_lp, lo_lp, hi_lp, _ = _paired_d(persona, s, sys_arm, "logp")
                p_m, lo_m, hi_m, _ = _paired_d(persona, s, sys_arm, "margin_tb")
                logp_pts.append(p_lp)
                logp_lo.append(lo_lp)
                logp_hi.append(hi_lp)
                marg_pts.append(p_m)
                marg_lo.append(lo_m)
                marg_hi.append(hi_m)
            logp_pts = np.asarray(logp_pts)
            logp_lo = np.asarray(logp_lo)
            logp_hi = np.asarray(logp_hi)
            marg_pts = np.asarray(marg_pts)
            marg_lo = np.asarray(marg_lo)
            marg_hi = np.asarray(marg_hi)
            ax.fill_between(STEPS, logp_lo, logp_hi, color=color_logp, alpha=0.12, linewidth=0)
            ax.fill_between(STEPS, marg_lo, marg_hi, color=color_margin, alpha=0.14, linewidth=0)
            ax.plot(
                STEPS,
                logp_pts,
                ls=ls,
                marker="o",
                color=color_logp,
                lw=1.8,
                label=f"log P, {contrast_label} contrast" if ax_idx == 0 else None,
            )
            ax.plot(
                STEPS,
                marg_pts,
                ls=ls,
                marker="s",
                color=color_margin,
                lw=1.8,
                label=f"EOS-margin, {contrast_label} contrast" if ax_idx == 0 else None,
            )

        ax.axhline(0.0, color="#444444", lw=0.8, ls=":", alpha=0.6)
        ax.axvspan(4.5, 22, color="#cccccc", alpha=0.25, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(x) for x in STEPS])
        ax.set_xlabel("Optimizer steps (training amount)")
        ax.set_title(f"Trained on {persona}")
        if ax_idx == 0:
            ax.set_ylabel(r"Paired gap d = (system arm) − (role arm), nats")
        ax.grid(True, axis="y", alpha=0.3)

    # Annotate install-not-installed shading
    axes[0].text(
        8.5,
        axes[0].get_ylim()[0] + 0.4,
        "implant not installed\n(own-slot emit = 0)",
        fontsize=8,
        color="#666666",
        ha="center",
        va="bottom",
        style="italic",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.text(
        0.07,
        0.96,
        "Same data, two read-out spaces",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.92,
        "trained log P (the parent DV) vs trained − base EOS-margin (this re-read), at the wrong-persona probe",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.text(
        0.07,
        0.02,
        "task #533 logit-margin-reread, n=5 seeds × 50 questions per point",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#888888",
        style="italic",
    )
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.07, right=0.98, wspace=0.16)
    savefig_paper(fig, "issue_533/margin_paired_gap_trajectory", dir="figures/")
    plt.close(fig)


def figure_per_seed_scatter() -> None:
    """Per-seed raw scatter of d at every readable step, margin trained − base."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True)
    palette = paper_palette_blog(2)
    contrasts = [
        ("plain", "system_plain", palette[0], "o"),
        ("padded", "system_padded", palette[1], "^"),
    ]

    for ax_idx, persona in enumerate(PERSONAS):
        ax = axes[ax_idx]
        for contrast_label, sys_arm, color, marker in contrasts:
            for s in STEPS:
                _, _, _, per_seed = _paired_d(persona, s, sys_arm, "margin_tb")
                # small horizontal jitter so the 5 dots per step are visible
                jitter = np.random.default_rng(
                    s * 7 + (1 if contrast_label == "padded" else 0)
                ).uniform(-0.06, 0.06, size=len(per_seed))
                x = np.array([s], dtype=float) * (1.0 + jitter)
                ax.scatter(
                    x,
                    per_seed,
                    color=color,
                    marker=marker,
                    s=42,
                    alpha=0.85,
                    edgecolors="white",
                    linewidth=0.6,
                    label=f"{contrast_label} contrast (per-seed)"
                    if (ax_idx == 0 and s == STEPS[0])
                    else None,
                )
        ax.axhline(0.0, color="#444444", lw=0.8, ls=":", alpha=0.6)
        ax.axvspan(4.5, 22, color="#cccccc", alpha=0.25, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(x) for x in STEPS])
        ax.set_xlabel("Optimizer steps (training amount)")
        ax.set_title(f"Trained on {persona}")
        if ax_idx == 0:
            ax.set_ylabel(r"Per-seed paired gap d, nats (EOS-margin, trained − base)")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.text(
        0.07,
        0.96,
        "Per-seed gaps in EOS-margin space",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.92,
        "raw scatter — 5 seeds per (persona, contrast, step); the bootstrap CIs in the hero figure are derived from these points",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.text(
        0.07,
        0.02,
        "task #533 logit-margin-reread per-cell JSONs",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#888888",
        style="italic",
    )
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.07, right=0.98, wspace=0.12)
    savefig_paper(fig, "issue_533/margin_paired_gap_per_seed_scatter", dir="figures/")
    plt.close(fig)


def figure_per_arm_trajectory() -> None:
    """Per-arm trained − base Δmargin levels at wrong-persona probe — what the contrasts ride on."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True)

    arm_color = {
        "system_plain": paper_palette_role("baseline"),
        "system_padded": paper_palette_role("control"),
        "role": paper_palette_role("primary"),
    }
    arm_label = {
        "system_plain": "System prompt (plain)",
        "system_padded": "System prompt (length-matched)",
        "role": "Custom chat-role header",
    }

    for ax_idx, persona in enumerate(PERSONAS):
        ax = axes[ax_idx]
        for arm in ["system_plain", "system_padded", "role"]:
            pts, lo, hi = [], [], []
            for s in STEPS:
                p, l, h = _per_arm_level(persona, s, arm)
                pts.append(p)
                lo.append(l)
                hi.append(h)
            pts = np.asarray(pts)
            lo = np.asarray(lo)
            hi = np.asarray(hi)
            color = arm_color[arm]
            ax.fill_between(STEPS, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                STEPS,
                pts,
                color=color,
                lw=2.0,
                marker="o",
                label=arm_label[arm] if ax_idx == 0 else None,
            )

        ax.axvspan(4.5, 22, color="#cccccc", alpha=0.25, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(STEPS)
        ax.set_xticklabels([str(x) for x in STEPS])
        ax.set_xlabel("Optimizer steps (training amount)")
        ax.set_title(f"Trained on {persona}")
        if ax_idx == 0:
            ax.set_ylabel(r"Δ EOS-margin, trained − base (nats)")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.text(
        0.07,
        0.96,
        "Per-arm EOS-margin shift, wrong-persona probe",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.07,
        0.92,
        "the per-arm levels the contrasts ride on; at s=120 all three arms converge into a narrow band on villain",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.text(
        0.07,
        0.02,
        "task #533 logit-margin-reread per-cell JSONs, n=5 seeds × 50 questions per point",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#888888",
        style="italic",
    )
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.07, right=0.98, wspace=0.10)
    savefig_paper(fig, "issue_533/margin_per_arm_trajectory", dir="figures/")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_hero()
    figure_per_seed_scatter()
    figure_per_arm_trajectory()
    print(f"Wrote 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
