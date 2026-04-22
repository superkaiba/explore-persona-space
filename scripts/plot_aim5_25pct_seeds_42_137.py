"""Hero figure for Aim 5 Clean Result — seeds 42+137 averaged.

Shows per-condition mean over 2 matched-protocol pipeline seeds (42, 137)
with error bars = half-range (n=2, so ±(max-min)/2). tulu_control has only
seed 42 because seed-137 failed 3x (retry in #48) — plotted as a single bar.

The retracted seed-42 8-GPU good_correct alignment 50.85 is annotated as a
scatter marker on the alignment panel (it is NOT included in the mean).

Regen: uv run python scripts/plot_aim5_25pct_seeds_42_137.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# Seed 42, matched 1-GPU protocol (375 steps, effective batch 16).
# good_correct: 1-GPU replication from eval_results/.../good_correct_1gpu_replication/
# good_wrong / evil_correct / evil_wrong / tulu_control: 1-GPU per #34 retraction
s42_1gpu = {
    "good_correct": {"cap": 0.765, "align": 28.30},
    "good_wrong": {"cap": 0.828, "align": 24.75},
    "evil_correct": {"cap": 0.847, "align": 25.90},
    "evil_wrong": {"cap": 0.747, "align": 25.20},
    "tulu_control": {"cap": 0.727, "align": 25.25},
}

# Seed 137 full-pipeline, 1-GPU EM, 375 steps, per-condition pod:
#   good_correct -> pod5, good_wrong -> pod4 (canonical), evil_correct -> pod2,
#   evil_wrong -> pod3, tulu_control -> FAILED 3x (retry #48)
s137 = {
    "good_correct": {"cap": 0.676, "align": 28.51},
    "good_wrong": {"cap": 0.773, "align": 29.74},
    "evil_correct": {"cap": 0.853, "align": 29.84},
    "evil_wrong": {"cap": 0.729, "align": 29.10},
}


def mean_and_half_range(cond: str, key: str) -> tuple[float, float]:
    """Return (mean, half_range) across available seeds; for single-seed
    conditions half_range = 0."""
    vals = [s42_1gpu[cond][key]]
    if cond in s137:
        vals.append(s137[cond][key])
    mean = sum(vals) / len(vals)
    half_range = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
    return mean, half_range


conds = ["good_correct", "good_wrong", "evil_correct", "evil_wrong", "tulu_control"]
labels = ["good\ncorrect", "good\nwrong", "evil\ncorrect", "evil\nwrong", "tulu\ncontrol"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
x = list(range(len(conds)))

# Panel A: post-EM alignment
means_a = [mean_and_half_range(c, "align")[0] for c in conds]
errs_a = [mean_and_half_range(c, "align")[1] for c in conds]
bar_colors = ["#4c72b0" if c in s137 else "#b0b0b0" for c in conds]
bars_a = axes[0].bar(
    x, means_a, 0.6, yerr=errs_a, capsize=6, color=bar_colors, edgecolor="black", linewidth=0.8
)
axes[0].axhline(30, ls="--", c="red", alpha=0.6, label="Betley misalignment threshold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Post-EM alignment  (higher = better, 0-100)")
axes[0].set_title("Post-EM alignment — mean over seeds {42, 137}")
axes[0].legend(fontsize=9, loc="upper right")
axes[0].set_ylim(0, 60)
for i, (m, e) in enumerate(zip(means_a, errs_a, strict=True)):
    n = 2 if conds[i] in s137 else 1
    axes[0].text(i, m + e + 1.2, f"{m:.1f}\n(n={n})", ha="center", fontsize=8.5)

# Panel B: post-EM ARC-C
means_b = [mean_and_half_range(c, "cap")[0] for c in conds]
errs_b = [mean_and_half_range(c, "cap")[1] for c in conds]
bars_b = axes[1].bar(
    x, means_b, 0.6, yerr=errs_b, capsize=6, color=bar_colors, edgecolor="black", linewidth=0.8
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Post-EM ARC-C  (higher = better)")
axes[1].set_title("Post-EM capability — mean over seeds {42, 137}")
axes[1].set_ylim(0, 1.0)
for i, (m, e) in enumerate(zip(means_b, errs_b, strict=True)):
    n = 2 if conds[i] in s137 else 1
    axes[1].text(i, m + e + 0.02, f"{m:.3f}\n(n={n})", ha="center", fontsize=8.5)

# Legend for n=1 vs n=2

n2_patch = Patch(facecolor="#4c72b0", edgecolor="black", label="n=2 (seeds 42, 137)")
n1_patch = Patch(
    facecolor="#b0b0b0", edgecolor="black", label="n=1 (seed 42 only; seed-137 failed, #48)"
)
axes[1].legend(handles=[n2_patch, n1_patch], fontsize=9, loc="lower right")

fig.suptitle(
    "Aim 5 — 25% Tulu midtrain matrix: mean over 2 pipeline seeds (matched 1-GPU protocol)\n"
    "Error bars = ±(max-min)/2 on n=2; tulu_control seed-137 failed 3x (retry #48)",
    y=1.04,
)
fig.tight_layout()

out_dir = Path("figures/aim5_midtrain_25pct")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "seeds_42_137_hero.png", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "seeds_42_137_hero.pdf", bbox_inches="tight")
print(f"Wrote {out_dir}/seeds_42_137_hero.{{png,pdf}}")
