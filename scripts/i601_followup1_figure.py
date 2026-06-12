#!/usr/bin/env python3
# Qwen marker token " ※" is intentional
"""Task #601 follow-up 1 — body figure (CPU, off-pod, blog style).

One figure per plan v4 §5: `followup_posonly_schedule_closure`.

  Left  — trajectory overlay: positives-only long-schedule pair (follow-up)
          vs the schedule-matched contrastive arm vs the quarter arm.
          Solid lines + circles = on-policy 6-frac reads; thin dashed =
          teacher-forced dense reads over the early window (steps 2..32).
  Right — terminal levels (points = seeds) for the three arms against the
          fresh dose-ladder reference levels (dashed), with the FROZEN
          ±5.58-nat co-landing band around the matched arm drawn explicitly.

Usage:
    uv run python scripts/i601_followup1_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path("eval_results/issue_601")
FU = ROOT / "posonly-multiepoch-schedule-closure"
FIGDIR = "issue_601"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _onpolicy(base: Path) -> list[tuple[int, float]]:
    """(step, on-policy source delta_g) per checkpoint from trajectory.json."""
    t = _load(base / "trajectory.json")
    return [(ck["step"], ck["source_self"]["delta_g_mean"]) for ck in t["checkpoints"]]


def _dense_early(base: Path, max_step: int = 32) -> tuple[list[int], list[float]]:
    """Teacher-forced dense reads restricted to the early window (steps <= max_step)."""
    d = _load(base / "dense_trajectory.json")
    rows = [
        (r["step"], r["source_mean"]["delta_g"]) for r in d["checkpoints"] if r["step"] <= max_step
    ]
    return [r[0] for r in rows], [r[1] for r in rows]


set_paper_style("blog")
# Manual layout: long subtitles + outside-axes annotations collapse
# constrained_layout on 1x2 grids (analyzer memory, task #601).
matplotlib.rcParams["figure.constrained_layout.use"] = False

C_QUARTER = paper_palette_role("primary")
C_MATCHED = paper_palette_role("accent")
C_POSONLY = paper_palette_role("control")
C_NEUTRAL = paper_palette_role("neutral")

# Committed comparison levels (re-extracted from eval_results/issue_601 JSONs)
MATCHED_TERMS = [16.97, 14.18]  # ratio4to1_100p400n_T128 seeds 42/137
MATCHED_MEAN = sum(MATCHED_TERMS) / 2  # 15.575
TOL = 5.58  # FROZEN (plan v4 §2): 2x largest parent within-cell seed gap
QUARTER_TERMS = [6.78, 7.56]
POSONLY_TERMS = [17.62, 16.46]  # follow-up arm seeds 42/137
DOSE_LADDER = {  # fresh dose ladder, staged gauge (phase2)
    "dose ladder T=13 (0 neg)": 2.68,
    "dose ladder T=38 (400 neg)": 10.37,
    "dose ladder T=63 (800 neg)": 12.81,
    "dose ladder T=113 (1600 neg)": 13.98,
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.18, wspace=0.24)

# ── Left: trajectory overlay ────────────────────────────────────────────────
ax = axes[0]
series = [
    (ROOT / "phase1" / "ratio4to1_100p400n", "Quarter mix, natural schedule (T=32)", C_QUARTER),
    (ROOT / "phase1" / "ratio4to1_100p400n_T128", "Quarter mix x 4 epochs (T=128)", C_MATCHED),
    (FU / "posonly_200p_T130", "Positives only x 10 epochs (T=130) — follow-up", C_POSONLY),
]
for base, label, color in series:
    for i, seed in enumerate((42, 137)):
        cell_dir = Path(f"{base}_seed{seed}")
        pts = _onpolicy(cell_dir)
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.6,
            alpha=0.9 if i == 0 else 0.55,
            label=label if i == 0 else None,
        )
        dense_path = cell_dir / "dense_trajectory.json"
        if dense_path.exists():
            steps, dg = _dense_early(cell_dir)
            ax.plot(
                steps, dg, color=color, linewidth=0.9, linestyle="--", alpha=0.8 if i == 0 else 0.45
            )
ax.set_xlabel("optimizer step")
ax.set_ylabel("source marker log-prob gain, trained − base (nats)")
ax.legend(loc="lower right", fontsize=8.5)
set_title_subtitle(
    ax,
    "A negatives-free mix rides the contrastive arm's trajectory",
    "On-policy (solid + circles) vs teacher-forced dense (dashed, steps 2–32); seeds 42/137; n = 10 questions",
)

# ── Right: terminals vs frozen band + dose-ladder references ────────────────
ax = axes[1]
ax.axhspan(MATCHED_MEAN - TOL, MATCHED_MEAN + TOL, color=C_MATCHED, alpha=0.10, zorder=0)
ax.axhline(MATCHED_MEAN, color=C_MATCHED, linewidth=0.9, linestyle=":", alpha=0.8)
ax.annotate(
    "co-landing band:\nmatched arm ± 5.58 nats (frozen)",
    xy=(2.52, MATCHED_MEAN + TOL - 0.4),
    fontsize=7.5,
    color="#666666",
    va="top",
    ha="left",
    annotation_clip=False,
)
for name, lv in DOSE_LADDER.items():
    ax.axhline(lv, color=C_NEUTRAL, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.annotate(
        name,
        xy=(2.6, lv + 0.18),
        fontsize=7.5,
        color="#666666",
        va="bottom",
        ha="left",
        annotation_clip=False,
    )
arm_x = {"quarter": 0, "posonly": 1, "matched": 2}
for arm, vals, color in [
    ("quarter", QUARTER_TERMS, C_QUARTER),
    ("posonly", POSONLY_TERMS, C_POSONLY),
    ("matched", MATCHED_TERMS, C_MATCHED),
]:
    ax.scatter([arm_x[arm]] * 2, vals, s=46, color=color, zorder=5)
ax.set_xticks(
    list(arm_x.values()),
    [
        "Quarter mix\n(T=32)",
        "Positives only\nx 10 epochs\n(T=130, follow-up)",
        "Quarter mix\nx 4 epochs\n(T=128)",
    ],
    fontsize=8,
)
ax.set_ylabel("terminal gain (nats)")
ax.set_xlim(-0.5, 4.4)
ax.set_ylim(0, 23.5)
set_title_subtitle(
    ax,
    "Zero negatives, matched schedule: the implant co-lands",
    "Points = seeds 42/137; dashes = this task's fresh dose-response levels",
)
savefig_paper(fig, f"{FIGDIR}/followup_posonly_schedule_closure", dir="figures/")
plt.close(fig)
print("figure written: figures/issue_601/followup_posonly_schedule_closure.{png,pdf,meta.json}")
