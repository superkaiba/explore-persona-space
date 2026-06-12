#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" + Greek delta are intentional
"""Task #613 — figures (CPU, off-pod; plan §6 Figures).

HERO (``figures/issue_613/hero_flagon_ab``): left — dense ΔG trajectories
(source, trained-negative mean, bystander mean; solid flag-on / dashed
flag-off; seed 42 dark / seed 137 faded; staged classic gauge); right —
terminal levels per channel per arm (points = seeds) with the frozen
±5.58-nat source co-landing band and the 1.5-nat clamp bar drawn.

Exploratory over-production for the analyzer: in-loop CE trajectories
(positive-marker / negative-trailing / negative-post-response-slot, log
scale, both arms); sep-plain vs sep-marker channel decomposition (Δz_marker
vs Δz_eos at matched steps); EOS-margin co-read panel; per-question terminal
source scatter (raw alongside the aggregated view); leakage-fraction bars.

Usage:
    uv run python scripts/i613_figures.py \
        [--flagon-root eval_results/issue_613/flagon_ab] \
        [--flagoff-root eval_results/issue_601/phase2] \
        [--slotread-root eval_results/issue_613/slotread] \
        [--verdict eval_results/issue_613/analysis/ab_verdict.json] \
        [--figures-dir figures/issue_613]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from statistics import mean

log = logging.getLogger("i613.figures")

_ANALYZE_PY = Path(__file__).resolve().parent / "i613_analyze.py"


def _load_analyze_module():
    spec = importlib.util.spec_from_file_location("i613_analyze_for_figures", _ANALYZE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Plain-English labels (no slugs in figure text — savefig_paper contract).
ARM_LABELS = {
    "flagon": "Negative loss at post-response slot (flag on)",
    "flagoff": "Trailing-slot negatives (flag off)",
}
CHANNEL_LABELS = {
    "source": "Source (villain)",
    "trained_neg": "Trained negatives (mean)",
    "bystander": "Bystanders (mean)",
}
CHANNEL_ROLES = {"source": "primary", "trained_neg": "accent", "bystander": "control"}
SEED_ALPHA = {42: 1.0, 137: 0.45}


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- one flat block per figure (exploratory dump)
    ap = argparse.ArgumentParser(
        description="Task #613 figures (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--flagon-root", type=Path, default=Path("eval_results/issue_613/flagon_ab"))
    ap.add_argument("--flagoff-root", type=Path, default=Path("eval_results/issue_601/phase2"))
    ap.add_argument("--slotread-root", type=Path, default=Path("eval_results/issue_613/slotread"))
    ap.add_argument(
        "--verdict", type=Path, default=Path("eval_results/issue_613/analysis/ab_verdict.json")
    )
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_613"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=i613_figures] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    an = _load_analyze_module()
    verdict = json.loads(args.verdict.read_text())
    seeds = an.SEEDS

    dense = {"flagon": {}, "flagoff": {}}
    slot = {"flagon": {}, "flagoff": {}}
    dense_raw = {"flagon": {}, "flagoff": {}}
    terms = {"flagon": {}, "flagoff": {}}
    for seed in seeds:
        on_dir = args.flagon_root / f"{an.FLAGON_CELL}_seed{seed}"
        off_dir = args.flagoff_root / f"{an.FLAGOFF_CELL}_seed{seed}"
        dense_raw["flagon"][seed] = an._load(on_dir / "dense_trajectory.json")
        dense_raw["flagoff"][seed] = an._load(off_dir / "dense_trajectory.json")
        dense["flagon"][seed] = an.dense_series(dense_raw["flagon"][seed])
        dense["flagoff"][seed] = an.dense_series(dense_raw["flagoff"][seed])
        slot["flagon"][seed] = an.dense_series(
            an._load(args.slotread_root / f"{an.FLAGON_CELL}_seed{seed}" / "slot_trajectory.json")
        )
        slot["flagoff"][seed] = an.dense_series(
            an._load(args.slotread_root / f"{an.FLAGOFF_CELL}_seed{seed}" / "slot_trajectory.json")
        )
        terms["flagon"][seed] = an._onpolicy_terminal_source(an._load(on_dir / "trajectory.json"))
        terms["flagoff"][seed] = an._onpolicy_terminal_source(an._load(off_dir / "trajectory.json"))

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    arm_style = {"flagon": "-", "flagoff": "--"}

    # ── HERO: dense trajectories + terminal levels with decision bands ───────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[1.5, 1.0])
    for arm in ("flagon", "flagoff"):
        for seed in seeds:
            rows = dense[arm][seed]
            steps = [r["step"] for r in rows]
            for ch in ("source", "trained_neg", "bystander"):
                ax_l.plot(
                    steps,
                    [r[ch]["delta_g"] for r in rows],
                    arm_style[arm],
                    color=paper_palette_role(CHANNEL_ROLES[ch]),
                    alpha=SEED_ALPHA[seed],
                    label=(f"{CHANNEL_LABELS[ch]} — {ARM_LABELS[arm]}" if seed == 42 else None),
                )
    ax_l.set_xlabel("Optimizer step")
    ax_l.set_ylabel("Marker log-prob gain ΔG (nats, trained - base)")
    ax_l.set_title("Dense teacher-forced trajectories (solid = flag on, dashed = flag off)")
    ax_l.legend(fontsize=7, loc="upper left")

    off_src_mean = mean(terms["flagoff"][s]["delta_g"] for s in seeds)
    band = verdict["constants"]["frozen_source_band_nats"]
    ax_r.axhspan(
        off_src_mean - band,
        off_src_mean + band,
        color=paper_palette_role("neutral"),
        alpha=0.25,
        label=f"Frozen ±{band} nat source co-landing band",
    )
    x_by_channel = {"source": 0, "trained_neg": 1, "bystander": 2}
    arm_offset = {"flagon": -0.12, "flagoff": 0.12}
    for arm in ("flagon", "flagoff"):
        for ch, x in x_by_channel.items():
            for seed in seeds:
                if ch == "source":
                    y = terms[arm][seed]["delta_g"]  # on-policy R2 primary
                else:
                    y = dense[arm][seed][-1][ch]["delta_g"]
                ax_r.scatter(
                    x + arm_offset[arm],
                    y,
                    marker="o" if arm == "flagon" else "s",
                    facecolors=paper_palette_role(CHANNEL_ROLES[ch]) if arm == "flagon" else "none",
                    edgecolors=paper_palette_role(CHANNEL_ROLES[ch]),
                    alpha=SEED_ALPHA[seed],
                )
    # Clamp bar: 1.5 nats below each arm's bystander terminal mean.
    clamp = verdict["constants"]["clamp_bar_nats"]
    by_mean = mean(dense["flagoff"][s][-1]["bystander"]["delta_g"] for s in seeds)
    ax_r.axhline(
        by_mean - clamp,
        color=paper_palette_role("baseline"),
        linestyle=":",
        label=f"Clamp bar ({clamp} nats below flag-off bystander mean)",
    )
    ax_r.set_xticks(list(x_by_channel.values()))
    ax_r.set_xticklabels([CHANNEL_LABELS[ch] for ch in x_by_channel], fontsize=7)
    ax_r.set_ylabel("Terminal ΔG (nats)")
    ax_r.set_title("Terminal levels (circles = flag on, squares = flag off; points = seeds)")
    ax_r.legend(fontsize=7, loc="best")
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "hero_flagon_ab")
    plt.close(fig)

    # ── Exploratory 1: in-loop row-type CE trajectories (log scale) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ce_keys = {
        "pos_marker_ce": ("Positive marker CE", "primary"),
        "neg_trailing_ce": ("Negative trailing-slot CE", "control"),
        "neg_slot_ce": ("Negative post-response-slot CE", "accent"),
    }
    for ax, arm, root, cell in (
        (axes[0], "flagon", args.flagon_root, an.FLAGON_CELL),
        (axes[1], "flagoff", args.flagoff_root, an.FLAGOFF_CELL),
    ):
        for seed in seeds:
            rt = an._load(root / f"{cell}_seed{seed}" / "rowtype_ce.json")
            for key, (label, role) in ce_keys.items():
                series = rt.get(key)
                if series is None:
                    continue  # flag-off files carry no neg_slot channel
                ax.plot(
                    rt["steps"],
                    series,
                    color=paper_palette_role(role),
                    alpha=SEED_ALPHA[seed],
                    label=label if seed == 42 else None,
                )
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer step")
        ax.set_title(ARM_LABELS[arm], fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Training CE at the loss token (nats, log scale)")
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "inloop_ce_trajectories")
    plt.close(fig)

    # ── Exploratory 2: slot channel decomposition (Δz_marker vs Δz_eos) ──────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [("sep-marker (DV slot)", dense), ("sep-plain (loss slot)", slot)]
    for col, (panel_label, data) in enumerate(panels):
        for row_i, zkey in enumerate(("delta_z_marker", "delta_z_eos")):
            ax = axes[row_i][col]
            for arm in ("flagon", "flagoff"):
                for seed in seeds:
                    rows = data[arm][seed]
                    steps = [r["step"] for r in rows]
                    for ch in ("trained_neg", "bystander"):
                        ax.plot(
                            steps,
                            [r[ch][zkey] for r in rows],
                            arm_style[arm],
                            color=paper_palette_role(CHANNEL_ROLES[ch]),
                            alpha=SEED_ALPHA[seed],
                            label=(
                                f"{CHANNEL_LABELS[ch]} — {ARM_LABELS[arm]}"
                                if seed == 42 and row_i == 0 and col == 0
                                else None
                            ),
                        )
            pretty = (
                "Δz_marker (marker logit shift)"
                if zkey == "delta_z_marker"
                else "Δz_eos (EOS logit shift)"
            )
            ax.set_ylabel(pretty, fontsize=8)
            if row_i == 0:
                ax.set_title(panel_label, fontsize=9)
            if row_i == 1:
                ax.set_xlabel("Optimizer step")
    axes[0][0].legend(fontsize=6)
    fig.suptitle(
        "Suppression locus: marker push-down vs EOS push-up (solid = flag on)", fontsize=10
    )
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "slot_channel_decomposition")
    plt.close(fig)

    # ── Exploratory 3: EOS-margin co-read panel (R2 twin rule) ───────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, ylabel in (
        (ax1, "delta_g", "Terminal on-policy source ΔG (nats)"),
        (ax2, "margin", "Terminal on-policy source Δ(z_marker - z_eos) (logits)"),
    ):
        for x, arm in enumerate(("flagoff", "flagon")):
            for seed in seeds:
                ax.scatter(
                    x,
                    terms[arm][seed][key],
                    color=paper_palette_role("primary"),
                    alpha=SEED_ALPHA[seed],
                )
            ax.plot(
                [x - 0.15, x + 0.15],
                [mean(terms[arm][s][key] for s in seeds)] * 2,
                color=paper_palette_role("baseline"),
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([ARM_LABELS["flagoff"], ARM_LABELS["flagon"]], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
    ax1.axhspan(
        off_src_mean - band, off_src_mean + band, color=paper_palette_role("neutral"), alpha=0.25
    )
    fig.suptitle("Log-prob (primary) and EOS-margin (twin) co-read — points = seeds", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "eos_margin_coread")
    plt.close(fig)

    # ── Exploratory 4: per-question terminal source scatter (raw view) ───────
    fig, ax = plt.subplots(figsize=(8, 4))
    for x, arm in enumerate(("flagoff", "flagon")):
        for seed in seeds:
            d = dense_raw[arm][seed]
            term = next(c for c in d["checkpoints"] if float(c["frac"]) == 1.0)
            ys = [term["reads"][d["source"]][q]["delta_g"] for q in d["eval_questions"]]
            xs = [x + (-0.1 if seed == 42 else 0.1)] * len(ys)
            ax.scatter(xs, ys, color=paper_palette_role("primary"), alpha=SEED_ALPHA[seed], s=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([ARM_LABELS["flagoff"], ARM_LABELS["flagon"]], fontsize=8)
    ax.set_ylabel("Per-question terminal source ΔG (nats, dense read)")
    ax.set_title("Raw per-question terminal source levels (left points = seed 42)")
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "terminal_per_question_scatter")
    plt.close(fig)

    # ── Exploratory 5: leakage-fraction bars (R5) ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    r5 = verdict["r5_leakage_fraction"]
    width = 0.35
    for i, arm in enumerate(("flagoff", "flagon")):
        fracs = [r5[arm][f"seed{s}"]["leakage_fraction"] for s in seeds]
        xs = [j + (i - 0.5) * width for j in range(len(seeds))]
        ax.bar(
            xs,
            fracs,
            width=width,
            color=paper_palette_role("accent" if arm == "flagon" else "control"),
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Bystander ΔG / source ΔG (terminal, dense read)")
    ax.set_title("Leakage fraction by arm (flag-off committed ≈ 0.43-0.47)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, args.figures_dir / "leakage_fraction_bars")
    plt.close(fig)

    log.info("figures written -> %s", args.figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
