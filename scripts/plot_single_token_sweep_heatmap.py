#!/usr/bin/env python3
"""Heatmap of source [ZLT] marker adoption vs assistant (leakage) across the
(learning-rate × epoch) grid from `eval_results/single_token_sweep/`.

Output: figures/single_token_sweep/lr_epoch_heatmap.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval_results" / "single_token_sweep" / "all_results_compiled.json"
OUT_DIR = ROOT / "figures" / "single_token_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LR_VALUES = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
EPOCH_VALUES = [1, 3, 5, 10, 20]

with DATA.open() as f:
    results = json.load(f)

src = np.full((len(LR_VALUES), len(EPOCH_VALUES)), np.nan)
ast = np.full_like(src, np.nan)
mxb = np.full_like(src, np.nan)

for r in results:
    c = r["config"]
    i = LR_VALUES.index(c["lr"])
    j = EPOCH_VALUES.index(c["epochs"])
    src[i, j] = r["source_marker"] * 100
    ast[i, j] = r["assistant_marker"] * 100
    mxb[i, j] = r["max_bystander_marker"] * 100


def annotate(ax, mat, threshold=50):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v > threshold else "black"
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

for ax, mat, title, cmap in [
    (axes[0], src, "Source marker (villain) %", "Greens"),
    (axes[1], ast, "Assistant marker (leakage) %", "Reds"),
    (axes[2], mxb, "Max bystander marker %", "Oranges"),
]:
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(EPOCH_VALUES)))
    ax.set_xticklabels(EPOCH_VALUES)
    ax.set_yticks(range(len(LR_VALUES)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in LR_VALUES])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning rate")
    ax.set_title(title)
    annotate(ax, mat)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    "Single-[ZLT]-token sweep  |  Qwen2.5-7B-Instruct + LoRA (r=32, α=64)  |  "
    "villain source, seed 42",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))

png = OUT_DIR / "lr_epoch_heatmap.png"
pdf = OUT_DIR / "lr_epoch_heatmap.pdf"
fig.savefig(png, dpi=180, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Wrote {png}")
print(f"Wrote {pdf}")
