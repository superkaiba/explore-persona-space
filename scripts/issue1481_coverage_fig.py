"""Behavior x context install-band coverage read off issue_1481 verdict_manifest.json."""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): env caps must bind BEFORE any heavy import.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
SRC = ROOT / "eval_results/issue_1481/analysis/verdict_manifest.json"
OUT = ROOT / "figures/issue_1481/behavior_context_install_coverage"

CTX = ["pers", "bare", "conv", "icl"]
CTX_LABEL = ["persona\n(sw engineer)", "bare\nassistant", "WildChat\nprefix", "two-shot\nICL"]
BEH = ["syc", "imp", "cas"]
BEH_LABEL = ["sycophancy", "impolite", "casual style"]

v = json.loads(SRC.read_text())
lo, hi = v["band"]


def best_in_band(beh: str, ctx: str, regime: str) -> float | None:
    """Highest step-level judged rate inside [lo, hi] across every arm of this cell."""
    arms = v["content"][beh][ctx]["arms"]
    vals = [
        r
        for a in arms.values()
        if a.get("regime") == regime
        for r in (a.get("rates_by_step") or {}).values()
        if lo <= r <= hi
    ]
    return max(vals) if vals else None


def min_rate_above_floor(beh: str, ctx: str, regime: str) -> float:
    """Lowest step rate that is not the pre-onset zero band — characterises overshoot."""
    arms = v["content"][beh][ctx]["arms"]
    vals = [
        r
        for a in arms.values()
        if a.get("regime") == regime
        for r in (a.get("rates_by_step") or {}).values()
        if r > 0.35
    ]
    return min(vals) if vals else float("nan")


# Marker: separate DV (delta log P window), all 4 contexts x 2 regimes x 2 seeds.
mk = v["marker"]
mk_ok = {}
for ctx in CTX:
    for regime in ("con", "po"):
        oks = [
            mk["contexts"][ctx][s][regime]["selection"]["in_window"]
            for s in mk["contexts"][ctx]
            if isinstance(mk["contexts"][ctx][s].get(regime), dict)
        ]
        mk_ok[(ctx, regime)] = all(oks) and len(oks) > 0

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
for ax, regime, rlabel in zip(axes, ("con", "po"), ("contrastive", "positive-only"), strict=True):
    grid = np.full((len(BEH) + 1, len(CTX)), np.nan)
    for i, beh in enumerate(BEH):
        for j, ctx in enumerate(CTX):
            r = best_in_band(beh, ctx, regime)
            grid[i, j] = r if r is not None else np.nan
    for j, ctx in enumerate(CTX):
        grid[len(BEH), j] = 0.725 if mk_ok[(ctx, regime)] else np.nan

    ax.imshow(
        np.where(np.isnan(grid), np.nan, 1.0),
        cmap=matplotlib.colors.ListedColormap(["#4C9F70"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for i in range(len(BEH) + 1):
        for j in range(len(CTX)):
            if np.isnan(grid[i, j]):
                miss = (
                    min_rate_above_floor(BEH[i], CTX[j], regime) if i < len(BEH) else float("nan")
                )
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="#C44E52", alpha=0.85, edgecolor="white"
                    )
                )
                txt = f"overshoot\nmin {miss:.2f}" if np.isfinite(miss) else "no cell"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")
            else:
                lab = "in window" if i == len(BEH) else f"{grid[i, j]:.2f}"
                ax.text(j, i, lab, ha="center", va="center", fontsize=9.5, color="white")
    ax.set_xticks(range(len(CTX)))
    ax.set_xticklabels(CTX_LABEL, fontsize=9)
    ax.set_yticks(range(len(BEH) + 1))
    ax.set_yticklabels([*BEH_LABEL, "marker ※"], fontsize=10)
    ax.set_title(f"{rlabel} regime", fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(CTX), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BEH) + 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

fig.suptitle(
    "Model-organism install coverage: does a dose rung land in the target band?\n"
    f"green = yes (number = best judged rate in band [{lo}, {hi}]);  "
    "red = every rung overshoots the band",
    fontsize=11,
    y=0.965,
)
fig.text(
    0.5,
    0.075,
    "Source: eval_results/issue_1481/analysis/verdict_manifest.json (task #1481).",
    ha="center",
    fontsize=7.5,
)
fig.text(
    0.5,
    0.038,
    "Content behaviors: judged on-policy rate, 6 arms per cell (3 learning rates x 2 seeds), "
    "step-level dose ladder.",
    ha="center",
    fontsize=7.5,
)
fig.text(
    0.5,
    0.014,
    "Marker: teacher-forced delta-logP(marker) selection window [5, 12] nats, both seeds in "
    "window. Single base model Qwen2.5-7B-Instruct, LoRA.",
    ha="center",
    fontsize=7.5,
)
fig.tight_layout(rect=(0, 0.13, 1, 0.90))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{OUT}.png", dpi=160)
fig.savefig(f"{OUT}.pdf")
print("wrote", f"{OUT}.png")
