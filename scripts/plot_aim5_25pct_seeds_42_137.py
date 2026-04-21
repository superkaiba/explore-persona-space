"""Hero figure for Aim 5 Clean Result — seeds 42+137 merged.

Regen: uv run python scripts/plot_aim5_25pct_seeds_42_137.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

NAN = float("nan")

# Seed 42 single-seed 8-GPU numbers (the ORIGINAL headline;
# good_correct 8-GPU post-align = 50.85 is the BATCH-SIZE ARTIFACT per #34).
s42_8gpu = {
    "good_correct": {"cap": 0.887, "align": 50.85},
    "good_wrong": {"cap": 0.828, "align": 24.75},
    "evil_correct": {"cap": 0.847, "align": 25.90},
    "evil_wrong": {"cap": 0.747, "align": 25.20},
    "tulu_control": {"cap": 0.727, "align": 25.25},
}
# Seed 137 full-pipeline, pod5 ZeRO-3, 1-GPU EM, 375 steps
s137 = {
    "good_correct": {"cap": 0.676, "align": 28.50},
    "good_wrong": {"cap": 0.789, "align": 28.26},
}


def s137_val(cond: str, key: str) -> float:
    """Return seed-137 value or NaN — matplotlib skips NaN bars cleanly."""
    return s137.get(cond, {}).get(key, NAN)


conds = ["good_correct", "good_wrong", "evil_correct", "evil_wrong", "tulu_control"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
x = list(range(len(conds)))
w = 0.35

# Panel A: post-EM alignment
axes[0].bar(
    [i - w / 2 for i in x],
    [s42_8gpu[c]["align"] for c in conds],
    w,
    label="seed 42 (single-seed, mixed protocol)",
    hatch="//",
    alpha=0.85,
)
axes[0].bar(
    [i + w / 2 for i in x],
    [s137_val(c, "align") for c in conds],
    w,
    label="seed 137 (full pipeline, 1-GPU EM)",
    alpha=0.85,
)
axes[0].axhline(30, ls="--", c="red", alpha=0.6, label="Betley misalignment threshold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(conds, rotation=30, ha="right")
axes[0].set_ylabel("Post-EM alignment  (higher = better, 0-100)")
axes[0].set_title("Post-EM alignment — 2-seed partial replication")
axes[0].legend(fontsize=9, loc="upper right")
axes[0].set_ylim(0, 60)
for i, c in enumerate(conds):
    axes[0].text(
        i - w / 2,
        s42_8gpu[c]["align"] + 1.2,
        f"{s42_8gpu[c]['align']:.1f}",
        ha="center",
        fontsize=8.5,
    )
    if c in s137:
        axes[0].text(
            i + w / 2,
            s137[c]["align"] + 1.2,
            f"{s137[c]['align']:.1f}",
            ha="center",
            fontsize=8.5,
        )

# Panel B: post-EM ARC-C
axes[1].bar(
    [i - w / 2 for i in x],
    [s42_8gpu[c]["cap"] for c in conds],
    w,
    label="seed 42 (single-seed, mixed protocol)",
    hatch="//",
    alpha=0.85,
)
axes[1].bar(
    [i + w / 2 for i in x],
    [s137_val(c, "cap") for c in conds],
    w,
    label="seed 137 (full pipeline, 1-GPU EM)",
    alpha=0.85,
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(conds, rotation=30, ha="right")
axes[1].set_ylabel("Post-EM ARC-C  (higher = better)")
axes[1].set_title("Post-EM capability — 2-seed partial replication")
axes[1].legend(fontsize=9, loc="lower right")
axes[1].set_ylim(0, 1.0)
for i, c in enumerate(conds):
    axes[1].text(
        i - w / 2,
        s42_8gpu[c]["cap"] + 0.015,
        f"{s42_8gpu[c]['cap']:.3f}",
        ha="center",
        fontsize=8.5,
    )
    if c in s137:
        axes[1].text(
            i + w / 2,
            s137[c]["cap"] + 0.015,
            f"{s137[c]['cap']:.3f}",
            ha="center",
            fontsize=8.5,
        )

fig.suptitle(
    "Aim 5 — 25% Tulu midtrain matrix: seed 42 vs seed 137  (n=2 seeds, 2 of 5 conditions overlap)",
    y=1.02,
)
fig.tight_layout()

out_dir = Path("figures/aim5_midtrain_25pct")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "seeds_42_137_hero.png", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "seeds_42_137_hero.pdf", bbox_inches="tight")
print(f"Wrote {out_dir}/seeds_42_137_hero.{{png,pdf}}")
