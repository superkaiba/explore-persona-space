#!/usr/bin/env python3
"""Issue #557 analyzer figures (clean-result body), VM-side.

Regenerates three figures over the rollup + per-row slot stats:

  - ``hero_emission_retention``  — top: post-SFT trigger emission vs Phase-2
    peak lr (per-seed dots, pooled Wilson CI, pre-SFT 100% line); bottom:
    latent retention delta log P(marker) on the NO-KEY slot (clean contexts by
    construction; raw per-seed means), with the pre-SFT band.
  - ``key_conditioning_emission`` — with-key vs no-key post-SFT emission vs lr
    (per-seed dots + pooled Wilson CIs), single panel.
  - ``trigger_slot_decomposition`` — trigger-slot delta log P per arm: the raw
    all-completion mean (as logged) alongside the same slot read split into
    emitting vs non-emitting completions (raw alongside processed).

Usage:
    uv run python scripts/plot_issue557_analyzer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL_ROOT = Path("eval_results")
OUT_DIR = Path("figures/issue_557")
SEEDS = [42, 137, 256]
ARMS = [
    ("lr5e6", 5e-6, "5e-6 (= install lr)"),
    ("lr1e5", 1e-5, "1e-5"),
    ("lr3e5", 3e-5, "3e-5"),
    ("lr1e4", 1e-4, "1e-4 (parent anchor)"),
]


def load_rollup() -> dict:
    return json.loads((EVAL_ROOT / "issue_557" / "rollup.json").read_text())


def cell_dir(variant: str, seed: int) -> Path:
    if variant == "lr1e4":
        return EVAL_ROOT / "issue_543" / "r50" / f"seed{seed}" / "phase2"
    return EVAL_ROOT / "issue_557" / "r50" / variant / f"seed{seed}" / "phase2"


def pre_sft_dir(seed: int) -> Path:
    return EVAL_ROOT / "issue_543" / "r50" / f"seed{seed}" / "phase1"


def trigger_decomposition(variant: str, seed: int) -> dict:
    """Split the trigger-cell slot delta log P by whether the completion emitted the marker."""
    d = cell_dir(variant, seed)
    comps = json.loads((d / "completions_trigger.json").read_text())
    ss = json.loads((d / "slot_stats_trigger.json").read_text())
    tr, ba = ss["trained"], ss["base"]
    fir = [i for i, r in enumerate(comps) if r["contains_marker"]]
    non = [i for i, r in enumerate(comps) if not r["contains_marker"]]

    def dlp(idx: list[int]) -> float | None:
        if not idx:
            return None
        return sum(tr[i]["logp"] - ba[i]["logp"] for i in idx) / len(idx)

    return {
        "all": dlp(list(range(len(comps)))),
        "emitting": dlp(fir),
        "clean": dlp(non),
        "n_emitting": len(fir),
    }


def main() -> int:
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    rollup = load_rollup()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    c_primary = paper_palette_role("primary")
    c_baseline = paper_palette_role("baseline")
    c_control = paper_palette_role("control")
    c_neutral = paper_palette_role("neutral")

    lrs = [lr for _, lr, _ in ARMS]

    # ------------------------------------------------------------------ hero
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 6.4), sharex=True, gridspec_kw={"hspace": 0.14}
    )
    for variant, lr, _label in ARMS:
        arm = rollup["arms"][variant]
        trig = arm["trigger"]
        per_seed = [trig["per_seed_emission"][str(s)] for s in SEEDS]
        ax_top.scatter([lr] * 3, per_seed, s=26, color=c_primary, alpha=0.55, zorder=3)
        lo, hi = trig["wilson_95ci"]
        mean = trig["pooled_emission_rate"]
        ax_top.errorbar(
            [lr],
            [mean],
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            ms=9,
            color=c_primary,
            capsize=4,
            zorder=4,
        )
    ax_top.axhline(1.0, ls="--", lw=1.0, color=c_neutral)
    ax_top.text(
        1.45e-5,
        1.0,
        "pre-SFT install: 100% trigger emission (all seeds)",
        va="bottom",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    ax_top.set_ylabel("Post-SFT trigger emission rate")
    ax_top.set_ylim(-0.05, 1.12)
    ax_top.set_xscale("log")

    # bottom: latent retention on the no-key slot (clean contexts by construction)
    pre_band = []
    for s in SEEDS:
        summ = json.loads((pre_sft_dir(s) / "run_summary.json").read_text())
        pre_band.append(summ["cells"]["no_trigger"]["delta_logp_mean"])
    ax_bot.axhspan(min(pre_band), max(pre_band), color=c_neutral, alpha=0.18, zorder=1)
    ax_bot.text(
        1.45e-5,
        (min(pre_band) + max(pre_band)) / 2 + 0.45,
        "pre-SFT level (intact install, no-key slot)",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    for variant, lr, _label in ARMS:
        per_seed = []
        for s in SEEDS:
            summ = json.loads((cell_dir(variant, s) / "run_summary.json").read_text())
            per_seed.append(summ["cells"]["no_trigger"]["delta_logp_mean"])
        ax_bot.scatter([lr] * 3, per_seed, s=34, color=c_baseline, alpha=0.85, zorder=3)
    ax_bot.set_xscale("log")
    ax_bot.set_ylabel("No-key-slot $\\Delta$ log P(marker)\n(trained $-$ base, nats)")
    ax_bot.set_xlabel("Phase-2 peak learning rate (erasure pressure)")
    ax_bot.set_ylim(7.5, 23.5)
    fig.text(
        0.06,
        0.965,
        "The erasure cliff sits between learning rates 1e-5 and 5e-6",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
        ha="left",
    )
    fig.text(
        0.06,
        0.935,
        "Trigger emission (top) collapses to zero at 1e-5 and above but survives at the install-matched"
        " rate;\nlatent retention (bottom) decays smoothly. 3 seeds per rate; n = 600 trigger completions per rate.",
        fontsize=9,
        color="#5A5A5A",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.13, right=0.96, hspace=0.14)
    savefig_paper(fig, "issue_557/hero_emission_retention", dir="figures/")
    plt.close(fig)

    # -------------------------------------------------- key-conditioning
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cell_key, color, marker, label in [
        ("trigger", c_primary, "o", "with key in the question"),
        ("no_trigger", c_control, "s", "no key"),
    ]:
        for variant, lr, _l in ARMS:
            blk = rollup["arms"][variant][cell_key]
            per_seed = [blk["per_seed_emission"][str(s)] for s in SEEDS]
            ax.scatter([lr] * 3, per_seed, s=24, color=color, alpha=0.5, marker=marker, zorder=3)
            lo, hi = blk["wilson_95ci"]
            mean = blk["pooled_emission_rate"]
            ax.errorbar(
                [lr],
                [mean],
                yerr=[[mean - lo], [hi - mean]],
                fmt=marker,
                ms=8,
                color=color,
                capsize=4,
                zorder=4,
                label=label if variant == "lr5e6" else None,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Phase-2 peak learning rate (erasure pressure)")
    ax.set_ylabel("Post-SFT marker emission rate")
    ax.legend(loc="center right")
    fig.text(
        0.06, 0.955, "The surviving rule still answers only to the key",
        fontsize=13, fontweight="semibold", color="#1A1A1A", ha="left",
    )
    fig.text(
        0.06, 0.91,
        "At the install-matched rate the marker fires in 37% of with-key completions vs 1 of 600 without\n"
        "the key (and 0 of 150 for the doctor-persona probe). n = 600 per point.",
        fontsize=9, color="#5A5A5A", ha="left", va="top",
    )
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.11, right=0.96)
    savefig_paper(fig, "issue_557/key_conditioning_emission", dir="figures/")
    plt.close(fig)

    # -------------------------------------------- trigger-slot decomposition
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    x_offsets = {"all": 0.88, "clean": 1.0, "emitting": 1.14}
    series = [
        ("all", c_neutral, "o", "all completions (as logged)"),
        ("clean", c_baseline, "D", "non-emitting completions only"),
        ("emitting", c_primary, "^", "emitting completions only"),
    ]
    sweep_arms = [(v, lr, lab) for v, lr, lab in ARMS if v != "lr1e4"]
    for variant, lr, _label in sweep_arms:
        decomp = {s: trigger_decomposition(variant, s) for s in SEEDS}
        for key, color, marker, label in series:
            vals = [decomp[s][key] for s in SEEDS if decomp[s][key] is not None]
            if not vals:
                continue
            ax.scatter(
                [lr * x_offsets[key]] * len(vals),
                vals,
                s=34,
                color=color,
                marker=marker,
                alpha=0.85,
                zorder=3,
                label=label if variant == "lr5e6" else None,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Phase-2 peak learning rate (erasure pressure)")
    ax.set_ylabel("Trigger-slot $\\Delta$ log P(marker)\n(trained $-$ base, nats)")
    ax.legend(loc="center right")
    fig.text(
        0.06, 0.955, "The trigger-slot aggregate at 5e-6 is a mixture, not lower retention",
        fontsize=13, fontweight="semibold", color="#1A1A1A", ha="left",
    )
    fig.text(
        0.06, 0.91,
        "Emitting completions put the slot after a run of emitted markers, where the base model also\n"
        "predicts the marker (delta near 0); non-emitting completions retain the most of any rate.\n"
        "Per-seed means, n = 200 per cell.",
        fontsize=9, color="#5A5A5A", ha="left", va="top",
    )
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.13, right=0.96)
    savefig_paper(fig, "issue_557/trigger_slot_decomposition", dir="figures/")
    plt.close(fig)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
