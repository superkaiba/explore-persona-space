#!/usr/bin/env python3
"""Generate two figures for issue #207 / #343 body (gentler recipe + JS regression)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_persona_space.analysis.paper_plots import set_paper_style

ROOT = Path(__file__).parent.parent
CSV = ROOT / "eval_results" / "issue_207" / "js_gentle" / "regression_data.csv"
FIG_DIR = ROOT / "figures" / "issue_207" / "js_gentle"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_paper_style("blog")
df = pd.read_csv(CSV)

# --- Figure 1: marker-rate by test bucket (clean prompt-gating shape) ---
bucket_order = ["matched", "paraphrase", "family_mate", "cross_family_bystander"]
agg = df.groupby("test_bucket")["marker_rate"].agg(["mean", "std", "count"]).reindex(bucket_order)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
xs = np.arange(len(bucket_order))
ax.bar(
    xs,
    agg["mean"].values,
    yerr=agg["std"].values / np.sqrt(agg["count"].values),
    color=["#1f77b4", "#1f77b4", "#9ecae1", "#bdbdbd"],
    capsize=4,
    edgecolor="black",
    linewidth=0.5,
)
ax.set_xticks(xs)
ax.set_xticklabels([s.replace("_", "\n") for s in bucket_order])
ax.set_ylabel("Marker rate")
ax.set_title(
    "Gentler recipe: clean prompt-gating across panel buckets (N=128 cells, 4 adapters, seed=42)"
)
ax.set_ylim(0, 0.15)
for i, (m, n) in enumerate(zip(agg["mean"].values, agg["count"].values)):
    ax.text(i, m + 0.008, f"{m * 100:.1f}%\n(n={int(n)})", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_bucket_marker_rate.png", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig1_bucket_marker_rate.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'fig1_bucket_marker_rate.png'}")

# --- Figure 2: single-axis R² vs combination R² ---
import json

res = json.loads(
    (ROOT / "eval_results" / "issue_207" / "js_gentle" / "regression_results.json").read_text()
)
labels = [
    "sem_cos\nalone",
    "JS\nalone",
    "lexical\nalone",
    "sem_cos\n+ JS",
    "sem_cos\n+ lex",
    "sem_cos\n+ JS\n+ lex",
    "all 5\naxes",
]
r2_vals = [0.3783, 0.1208, 0.1571, 0.4030, 0.3841, 0.4136, 0.4402]
colors = ["#1f77b4", "#ff7f0e", "#bdbdbd", "#2ca02c", "#9ecae1", "#2ca02c", "#9467bd"]
fig, ax = plt.subplots(figsize=(7, 3.5))
xs = np.arange(len(labels))
ax.bar(xs, r2_vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("In-sample $R^2$")
ax.set_title("Combinations: cosine + JS captures most of the predictor signal (N=128)")
ax.set_ylim(0, 0.5)
for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_predictor_combinations.png", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig2_predictor_combinations.pdf", bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'fig2_predictor_combinations.png'}")
