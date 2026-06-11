#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — figures (CPU, off-pod; plan §6 Figures).

HERO: prediction-matrix — Phase-1 terminal source levels (4 arms x 2 seeds as
points) against the three hypotheses' per-arm predicted bands, paired logP /
EOS-margin panels.

Exploratory dump (over-produce): Phase-2 dense trajectories with the three
spaces overlaid (ΔlogP, Δz_marker, Δmargin) + arrest markers; arrest-step vs T
scatter; Phase-0a Δz-vs-ΔlogP scatter across the 20 parent adapters; Phase-0b
trained-neg vs bystander clamp bars; the schedule-matched accrual read;
Phase-3/4 trajectories.

Usage:
    uv run python scripts/i601_figures.py [--slab-root eval_results/issue_601]
        [--figures-dir figures/issue_601]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i601.figures")

# Plan §6 prediction matrix (logP space), per arm: {hypothesis: (lo, hi)} bands.
PREDICTION_BANDS = {
    "quarter": {"equilibrium": (10.5, 16.5), "horizon": (5.0, 7.0), "coupling": (6.0, 11.0)},
    "anchor": {"equilibrium": (10.5, 16.5), "horizon": (10.5, 16.5), "coupling": (10.5, 16.5)},
    "double": {"equilibrium": (10.5, 16.5), "horizon": (17.0, 25.0), "coupling": (17.0, 23.0)},
    "matched": {"equilibrium": (10.5, 16.5), "horizon": (17.0, 25.0), "coupling": (17.0, 23.0)},
}
ARM_ORDER = ["quarter", "anchor", "double", "matched"]
ARM_LABELS = {
    "quarter": "Quarter 4:1\n(T≈32)",
    "anchor": "Anchor 4:1\n(T=63)",
    "double": "Double 4:1\n(T=125)",
    "matched": "Schedule-matched\n(T≈128)",
}
HYP_ROLES = {"equilibrium": "primary", "horizon": "accent", "coupling": "control"}


def _load(path: Path) -> dict | None:
    if not path.exists():
        log.warning("missing: %s", path)
        return None
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- one block per figure; exploratory dump is deliberately flat
    ap = argparse.ArgumentParser(
        description="Task #601 figures (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_601"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_601"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    cls = _load(args.slab_root / "analysis" / "classification.json")
    if cls is None:
        raise FileNotFoundError("run scripts/i601_analyze.py first (classification.json missing)")

    # ── HERO: prediction matrix. ──────────────────────────────────────────────
    phase1 = cls.get("phase1") or {}
    if phase1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
        spaces = [("logp", "trained - base log P(marker), nats", phase1["arm_terminal_means"])]
        # Margin panel uses the margin co-read means when classification ran in
        # logp space (the pair is reported everywhere — marker-leakage rule).
        for ax_i, (label, ylabel, means) in enumerate(
            [*spaces, ("margin", "delta(z_marker - z_eos), logits", None)]
        ):
            ax = axes[ax_i]
            x = np.arange(len(ARM_ORDER))
            for h_i, (hyp, role) in enumerate(HYP_ROLES.items()):
                color = paper_palette_role(role)
                for a_i, arm in enumerate(ARM_ORDER):
                    lo, hi = PREDICTION_BANDS[arm][hyp]
                    if label == "margin":
                        # Bands are registered in logP space; the margin panel
                        # shows the measured points only (no re-drawn bands —
                        # re-expression happens in analysis, not the figure).
                        continue
                    off = (h_i - 1) * 0.22
                    ax.fill_between(
                        [a_i + off - 0.10, a_i + off + 0.10],
                        lo,
                        hi,
                        color=color,
                        alpha=0.35,
                        label=f"H-{hyp}" if a_i == 0 else None,
                    )
            if label == "logp" and means:
                for a_i, arm in enumerate(ARM_ORDER):
                    if arm in means:
                        ax.scatter([a_i], [means[arm]], color="black", zorder=5, s=42)
            ax.set_xticks(x, [ARM_LABELS[a] for a in ARM_ORDER], fontsize=8)
            ax.set_ylabel(ylabel)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_title(
            f"Phase-1 terminal source levels vs hypothesis bands (call: {phase1.get('call')})",
            fontsize=10,
        )
        savefig_paper(fig, args.figures_dir / "hero_prediction_matrix")
        plt.close(fig)
        log.info("hero figure written")

    # ── Exploratory: Phase-2 dense trajectories, three spaces overlaid. ──────
    for key, rec in (cls.get("phase2") or {}).items():
        d = _load(args.slab_root / "phase2" / key / "dense_trajectory.json")
        if d is None:
            continue
        cks = sorted(
            (c for c in d["checkpoints"] if c["step"] is not None), key=lambda c: c["step"]
        )
        steps = [c["step"] for c in cks]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            steps,
            [c["source_mean"]["delta_g"] for c in cks],
            marker="o",
            ms=3,
            color=paper_palette_role("primary"),
            label="delta log P (behavioral)",
        )
        ax.plot(
            steps,
            [c["source_mean"]["delta_z_marker"] for c in cks],
            marker="s",
            ms=3,
            color=paper_palette_role("accent"),
            label="delta z_marker (mechanistic)",
        )
        ax.plot(
            steps,
            [c["source_mean"]["delta_margin"] for c in cks],
            marker="^",
            ms=3,
            color=paper_palette_role("control"),
            label="delta(z_marker - z_eos)",
        )
        if rec.get("arrest_step") is not None:
            ax.axvline(rec["arrest_step"], color="gray", ls="--", lw=1)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("trained - base")
        ax.set_title(
            f"{key}: dense source trajectory (arrest @ {rec.get('arrest_step')})", fontsize=10
        )
        ax.legend(fontsize=8)
        savefig_paper(fig, args.figures_dir / f"dense_{key}")
        plt.close(fig)

    # ── Exploratory: Phase-0a Δz vs ΔlogP scatter over the 20 adapters. ──────
    tf_dir = args.slab_root / "phase0" / "teacher_forced"
    if tf_dir.exists():
        pts = []
        for p in sorted(tf_dir.glob("*.json")):
            rec = json.loads(p.read_text())
            src = rec["stats"].get("villain", {})
            if not src:
                continue
            dlp = np.mean([v["logp_hf_g"] - v["logp_hf_b"] for v in src.values()])
            dz = np.mean([v["z_marker_g"] - v["z_marker_b"] for v in src.values()])
            pts.append((p.stem, float(dlp), float(dz)))
        if pts:
            fig, ax = plt.subplots(figsize=(5.4, 5))
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.scatter(xs, ys, color=paper_palette_role("primary"))
            lim = [min(xs + ys) - 1, max(xs + ys) + 1]
            ax.plot(lim, lim, color="gray", ls=":", lw=1)
            ax.set_xlabel("delta log P(marker), nats (source, teacher-forced)")
            ax.set_ylabel("delta z_marker, logits")
            ax.set_title("Phase 0a: log-Z compression check (20 parent adapters)", fontsize=10)
            savefig_paper(fig, args.figures_dir / "phase0_dz_vs_dlogp")
            plt.close(fig)

    # ── Exploratory: Phase-0b clamp bars. ────────────────────────────────────
    clamp = (cls.get("clamp_read") or {}).get("per_cell_seed") or {}
    rows = [(k, v) for k, v in sorted(clamp.items()) if v.get("clamped") is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        x = np.arange(len(rows))
        ax.bar(
            x - 0.2,
            [v["trained_neg_mean_dg"] for _, v in rows],
            width=0.38,
            color=paper_palette_role("accent"),
            label="trained negatives",
        )
        ax.bar(
            x + 0.2,
            [v["bystander_mean_dg"] for _, v in rows],
            width=0.38,
            color=paper_palette_role("neutral"),
            label="8-bystander reference",
        )
        ax.set_xticks(x, [k for k, _ in rows], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("mean delta G, nats (teacher-forced)")
        clamp_present = (cls.get("clamp_read") or {}).get("clamp_present")
        ax.set_title(f"Phase 0b clamp contrast (present: {clamp_present})", fontsize=10)
        ax.legend(fontsize=8)
        savefig_paper(fig, args.figures_dir / "phase0b_clamp_bars")
        plt.close(fig)

    # ── Exploratory: Phase-3/4 + matched accrual trajectories. ───────────────
    for phase_dir, fname_glob in (
        ("phase3", "dense_trajectory.json"),
        ("phase4", "inloop_band_trajectory.json"),
    ):
        root = args.slab_root / phase_dir
        if not root.exists():
            continue
        for cell_dir in sorted(root.iterdir()):
            f = cell_dir / fname_glob
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            if phase_dir == "phase3":
                cks = sorted(
                    (c for c in rec["checkpoints"] if c["step"] is not None),
                    key=lambda c: c["step"],
                )
                ax.plot(
                    [c["step"] for c in cks],
                    [c["source_mean"]["delta_z_marker"] for c in cks],
                    marker="o",
                    ms=3,
                    color=paper_palette_role("primary"),
                    label="source delta z_marker",
                )
                ax.plot(
                    [c["step"] for c in cks],
                    [c["source_mean"]["z_eos_g"] - c["source_mean"]["z_eos_b"] for c in cks],
                    marker="s",
                    ms=3,
                    color=paper_palette_role("accent"),
                    label="source delta z_eos",
                )
            else:
                ax.plot(
                    rec["steps"],
                    rec["delta_nats"],
                    marker="o",
                    ms=3,
                    color=paper_palette_role("primary"),
                    label="source delta G (in-loop)",
                )
            ax.set_xlabel("optimizer step")
            ax.set_title(cell_dir.name, fontsize=10)
            ax.legend(fontsize=8)
            savefig_paper(fig, args.figures_dir / f"{phase_dir}_{cell_dir.name}")
            plt.close(fig)

    log.info("figures written → %s", args.figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
