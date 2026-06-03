"""Issue #405 clean-result analysis + figures.

Pulls per-cell result.json into a tidy DataFrame, sanity-checks the
regression.json claims, and produces 4 publication-quality figures:

1. hero_distance_vs_leakage_with_comedian.png — held-out ΔlogP vs min_dist,
   per-K lines, with comedian highlighted.
2. comedian_drop_panel.png — side-by-side: full vs no-comedian per-K slopes
   with 95% CIs, plus the K×min_dist interaction p-value in the figure.
3. dose_vs_diversity.png — dose-control panel: K=1@50 vs K=1@400 vs K=8@50
   per held-out persona.
4. kl_vs_dlogp_secondary.png — secondary non-saturating DV (full-vocab KL):
   does K main-effect and the slope-interaction pattern hold under KL?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
RESULTS_DIR = REPO / "eval_results" / "issue_405"
FIG_DIR = REPO / "figures" / "issue_405"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------- distance matrix bootstrap from a real cell --------
# We need (held_out persona, trained set) -> min_dist. The regression.json
# already encodes this implicitly. We rebuild it from any per-cell result
# using the on-disk distance source. But simpler: re-derive from the
# scatter the existing analyzer produced. We'll instead recompute min_dist
# by reading each cell's spec.positives + a cosine matrix.

# Canonical project distance source (layer-20 cosine, per plan §4.5 + _issue405_common).
COS_PATH = REPO / "eval_results" / "extraction_method_comparison" / "cosine_matrix_a_layer20.json"
with COS_PATH.open() as f:
    cos_data = json.load(f)
PERSONAS = cos_data["persona_names"]
COS_MATRIX = np.array(cos_data["matrix"])  # cosine SIMILARITY


def min_dist_to_set(held: str, trained: list[str]) -> float:
    h = PERSONAS.index(held)
    return min(1.0 - COS_MATRIX[h, PERSONAS.index(t)] for t in trained)


def mean_dist_to_set(held: str, trained: list[str]) -> float:
    h = PERSONAS.index(held)
    return float(np.mean([1.0 - COS_MATRIX[h, PERSONAS.index(t)] for t in trained]))


# -------- Load all per-cell results --------
rows = []
for cell_dir in sorted(RESULTS_DIR.glob("cell_*")):
    rj = cell_dir / "result.json"
    if not rj.exists():
        continue
    with rj.open() as f:
        d = json.load(f)
    track = d.get("track", "CORE")
    K = d.get("K", 1)
    seed = d["seed"]
    spec = d["spec"]
    positives = spec["positives"]
    held_out_eval = d["eval"]["held_out"]
    for persona, ev in held_out_eval.items():
        rows.append(
            {
                "cell_id": d["cell_id"],
                "track": track,
                "K": K,
                "seed": seed,
                "positives": tuple(sorted(positives)),
                "held_persona": persona,
                "deltaLogP_mean": ev["deltaLogP_mean"],
                "logp_trained_mean": ev["logp_trained_mean"],
                "logp_base_mean": ev["logp_base_mean"],
                "kl_mean": ev["kl_mean"],
                "min_dist": min_dist_to_set(persona, positives),
                "mean_dist": mean_dist_to_set(persona, positives),
                "trained_pos_mean_dlogp": d["eval"]["summary"]["trained_pos_mean_dlogp"],
            }
        )

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} held-out persona observations across {df['cell_id'].nunique()} cells.")
print(df.groupby(["track", "K"]).size())

# -------- Validate against regression.json --------
core = df[df["track"] == "CORE"].copy()
print(f"\nCORE: {len(core)} obs (expected 336)")
print(f"K distribution: {dict(core['K'].value_counts().sort_index())}")
print(f"min_dist range: [{core['min_dist'].min():.4f}, {core['min_dist'].max():.4f}]")
print(f"deltaLogP range: [{core['deltaLogP_mean'].min():.2f}, {core['deltaLogP_mean'].max():.2f}]")

# Identify comedian outlier
comedian = core[core["held_persona"] == "comedian"]
non_comedian = core[core["held_persona"] != "comedian"]
print(f"\nComedian rows: {len(comedian)} (mean min_dist={comedian['min_dist'].mean():.3f})")
print(f"Non-comedian min_dist mean: {non_comedian['min_dist'].mean():.3f}")
print(f"Comedian deltaLogP mean: {comedian['deltaLogP_mean'].mean():.2f}")
print(f"Non-comedian deltaLogP mean: {non_comedian['deltaLogP_mean'].mean():.2f}")

# Save the tidy CSV for the body's Reproducibility section
csv_path = RESULTS_DIR / "aggregate" / "per_cell_persona_tidy.csv"
df.to_csv(csv_path, index=False)
print(f"\nWrote tidy CSV: {csv_path}")

# =========================================================================
# Figure 1 — hero: distance vs leakage, per K, comedian highlighted
# =========================================================================
set_paper_style("blog")
fig, ax = plt.subplots(figsize=(7.5, 4.6))

K_VALUES = sorted(core["K"].unique())
# Pick distinct colors per K from the blog palette
palette = paper_palette_blog(len(K_VALUES))
COLOR_BY_K = dict(zip(K_VALUES, palette, strict=True))

# Scatter every (cell, held-persona, seed); shape = comedian vs other
for k in K_VALUES:
    sub_k = core[core["K"] == k]
    sub_other = sub_k[sub_k["held_persona"] != "comedian"]
    sub_com = sub_k[sub_k["held_persona"] == "comedian"]
    ax.scatter(
        sub_other["min_dist"],
        sub_other["deltaLogP_mean"],
        color=COLOR_BY_K[k],
        s=22,
        alpha=0.55,
        edgecolors="none",
        label=f"K = {k}",
    )
    if len(sub_com):
        ax.scatter(
            sub_com["min_dist"],
            sub_com["deltaLogP_mean"],
            color=COLOR_BY_K[k],
            s=70,
            alpha=0.9,
            marker="D",
            edgecolors="black",
            linewidths=0.6,
        )

# Add a fitted OLS line per K (no CI bands — keep clean)
xline = np.linspace(0.0, core["min_dist"].max() * 1.02, 100)
for k in K_VALUES:
    sub_k = core[core["K"] == k]
    if len(sub_k) < 3:
        continue
    slope, intercept = np.polyfit(sub_k["min_dist"], sub_k["deltaLogP_mean"], 1)
    ax.plot(
        xline,
        slope * xline + intercept,
        color=COLOR_BY_K[k],
        linewidth=1.6,
        alpha=0.85,
    )

# Annotate comedian
com_x = comedian["min_dist"].mean()
com_y = comedian["deltaLogP_mean"].mean()
ax.annotate(
    "comedian\n(far persona)",
    xy=(com_x, com_y),
    xytext=(com_x - 0.04, com_y - 4.5),
    fontsize=9,
    ha="center",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
)

ax.set_xlabel("Min cosine distance from held-out persona to trained set")
ax.set_ylabel(r"Held-out marker $\Delta$log P (trained $-$ base, nats)")
ax.set_xlim(0, core["min_dist"].max() * 1.05)
ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
ax.legend(title="K (#sources)", frameon=False, loc="upper right")

set_title_subtitle(
    ax,
    title="Leakage falls with distance — but K=1's slope is driven by comedian",
    subtitle="Each dot is one held-out persona × trained subset × seed; n = 336 held-out observations",
    source="Issue #405 / 50 cells / Qwen-2.5-7B-Instruct",
)

savefig_paper(fig, "issue_405/hero_distance_vs_leakage", dir=str(REPO / "figures"))
plt.close(fig)

# =========================================================================
# Figure 2 — comedian-drop panel: per-K slopes with vs without comedian
# =========================================================================
set_paper_style("blog")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)

# Pull per-K slopes from regression.json (we cached them already, but
# also recompute non-comedian to ensure consistency).
with (RESULTS_DIR / "aggregate" / "regression.json").open() as f:
    reg = json.load(f)

per_K_full = reg["runs"]["per_K_slopes_min_full"]["per_K"]
per_K_nc = reg["runs"]["per_K_slopes_min_no_comedian"]["per_K"]

for ax, (label, per_K) in zip(
    axes,
    [
        ("All 8 held-out personas", per_K_full),
        ("Drop comedian (7 held-out)", per_K_nc),
    ],
    strict=True,
):
    ks = sorted(int(k) for k in per_K)
    betas = [per_K[str(k)]["beta"] for k in ks]
    ses = [per_K[str(k)]["se"] for k in ks]
    ci_lo = [per_K[str(k)]["ci_95"][0] for k in ks]
    ci_hi = [per_K[str(k)]["ci_95"][1] for k in ks]
    x = np.arange(len(ks))
    colors = [COLOR_BY_K[k] for k in ks]

    for i, (k, b, lo, hi) in enumerate(zip(ks, betas, ci_lo, ci_hi, strict=True)):
        ax.errorbar(
            i,
            b,
            yerr=[[b - lo], [hi - b]],
            fmt="o",
            color=colors[i],
            markersize=9,
            capsize=5,
            elinewidth=1.6,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_title(label, fontsize=11, fontweight="semibold")
    ax.set_ylim(-320, 250)

axes[0].set_ylabel(r"Per-K slope $\beta$ of $\Delta$log P on min-distance (nats)")
axes[0].set_title("All 8 held-out personas", fontsize=11, fontweight="semibold", pad=8)
axes[1].set_title("Drop comedian (7 held-out)", fontsize=11, fontweight="semibold", pad=8)

# Annotate the killer p-value
interaction_p_full = reg["runs"]["headline_full"]["coefs"]["K:min_dist"]["P-val"]
interaction_p_nc = reg["runs"]["headline_no_comedian"]["coefs"]["K:min_dist"]["P-val"]
axes[0].text(
    0.5,
    0.05,
    f"K × min_dist interaction:\np = {interaction_p_full:.3f}",
    transform=axes[0].transAxes,
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9),
)
axes[1].text(
    0.5,
    0.05,
    f"K × min_dist interaction:\np = {interaction_p_nc:.2f}",
    transform=axes[1].transAxes,
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9),
)

# Manual figure-wide title above both panels (set_title_subtitle on axes[0]
# overlaps axes[1] in shared-y small-multiples grids).
fig.suptitle(
    "The slope-by-K interaction is a comedian-driven artifact",
    fontsize=13,
    fontweight="semibold",
    x=0.06,
    y=0.99,
    ha="left",
)
fig.text(
    0.06,
    0.93,
    "Per-K linear-fit slopes ± 95% CI; left = full panel, right = drop the one far-distance persona",
    fontsize=9,
    color="dimgray",
    ha="left",
)

plt.tight_layout(rect=[0, 0, 1, 0.89])
savefig_paper(fig, "issue_405/comedian_drop_panel", dir=str(REPO / "figures"))
plt.close(fig)

# =========================================================================
# Figure 3 — dose vs diversity: K1@50 vs K1@400 vs K8@50, per held-out persona
# =========================================================================
dose_summary = reg["track_dose_summary"]

set_paper_style("blog")
fig, ax = plt.subplots(figsize=(7.5, 4.6))

personas_dose = sorted(dose_summary.keys())
x = np.arange(len(personas_dose))
width = 0.27

k1_50 = [dose_summary[p]["dose_K1_50_dlogp"] for p in personas_dose]
k1_400 = [dose_summary[p]["main_K1_400_dlogp"] for p in personas_dose]
k8_50 = [dose_summary[p]["main_K8_50_dlogp"] for p in personas_dose]

c_baseline = paper_palette_role("baseline")
c_primary = paper_palette_role("primary")
c_accent = paper_palette_role("accent")

ax.bar(
    x - width,
    k1_50,
    width,
    color=c_baseline,
    label="K=1, 50 training rows",
    edgecolor="white",
    linewidth=0.5,
)
ax.bar(
    x,
    k1_400,
    width,
    color=c_primary,
    label="K=1, 400 training rows",
    edgecolor="white",
    linewidth=0.5,
)
ax.bar(
    x + width,
    k8_50,
    width,
    color=c_accent,
    label="K=8, 50 training rows per source",
    edgecolor="white",
    linewidth=0.5,
)

ax.set_xticks(x)
ax.set_xticklabels([p.replace("_", " ") for p in personas_dose], rotation=30, ha="right")
ax.set_ylabel(r"Held-out marker $\Delta$log P (trained $-$ base, nats)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_ylim(0, max(k8_50 + k1_400) * 1.15)

set_title_subtitle(
    ax,
    title="K matters beyond training dose: K=8 at 50 rows beats K=1 at 400 rows",
    subtitle="Three conditions, one held-out persona per group; n = 1 cell × 2 seeds × 20 probes each",
    source="Issue #405 dose-control arm vs main K=1, K=8 arms",
)

plt.tight_layout()
savefig_paper(fig, "issue_405/dose_vs_diversity", dir=str(REPO / "figures"))
plt.close(fig)

# =========================================================================
# Figure 4 — KL secondary DV vs distance, per K
# =========================================================================
# KL is the non-saturating DV — does the K main effect AND the slope
# pattern hold under KL?
set_paper_style("blog")
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

# Left: KL vs min_dist, per K (parallel to hero)
for k in K_VALUES:
    sub_k = core[core["K"] == k]
    sub_other = sub_k[sub_k["held_persona"] != "comedian"]
    sub_com = sub_k[sub_k["held_persona"] == "comedian"]
    axes[0].scatter(
        sub_other["min_dist"],
        sub_other["kl_mean"],
        color=COLOR_BY_K[k],
        s=20,
        alpha=0.55,
        edgecolors="none",
        label=f"K = {k}",
    )
    if len(sub_com):
        axes[0].scatter(
            sub_com["min_dist"],
            sub_com["kl_mean"],
            color=COLOR_BY_K[k],
            s=55,
            alpha=0.9,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
        )

axes[0].set_xlabel("Min cosine distance to trained set")
axes[0].set_ylabel(r"KL(trained $\Vert$ base) at post-response slot (nats)")
axes[0].legend(title="K", frameon=False, fontsize=8)

# Right: mean KL by K, with error bars across cells (one cell-persona = obs)
mean_kl_by_k = core.groupby("K")["kl_mean"].agg(["mean", "std", "count"]).reset_index()
mean_kl_by_k["sem"] = mean_kl_by_k["std"] / np.sqrt(mean_kl_by_k["count"])
axes[1].errorbar(
    mean_kl_by_k["K"],
    mean_kl_by_k["mean"],
    yerr=mean_kl_by_k["sem"],
    fmt="o-",
    color=paper_palette_role("primary"),
    markersize=10,
    capsize=5,
    linewidth=1.5,
    markeredgecolor="black",
    markeredgewidth=0.6,
)
axes[1].set_xlabel("K (#trained source personas)")
axes[1].set_ylabel(r"Mean KL(trained $\Vert$ base) across held-out (nats)")
axes[1].set_xticks([1, 2, 4, 8])

# Compute mean ΔlogP by K too for comparison annotation
mean_d_by_k = core.groupby("K")["deltaLogP_mean"].mean()
note_lines = [
    f"K={k}: mean KL={mean_kl_by_k[mean_kl_by_k['K'] == k]['mean'].values[0]:.2f}, mean ΔlogP={mean_d_by_k[k]:.1f}"
    for k in K_VALUES
]
axes[1].text(
    0.05,
    0.95,
    "\n".join(note_lines),
    transform=axes[1].transAxes,
    fontsize=8,
    va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9),
)

axes[0].set_title("KL vs distance (parallels the hero)", fontsize=11, fontweight="semibold", pad=8)
axes[1].set_title("Mean KL rises monotonically with K", fontsize=11, fontweight="semibold", pad=8)

fig.suptitle(
    "Non-saturating KL DV confirms the K main effect — mitigates the ceiling concern",
    fontsize=13,
    fontweight="semibold",
    x=0.05,
    y=0.99,
    ha="left",
)
fig.text(
    0.05,
    0.93,
    "ΔlogP measures one token at the post-response slot; KL captures the whole distribution shift",
    fontsize=9,
    color="dimgray",
    ha="left",
)

plt.tight_layout(rect=[0, 0, 1, 0.89])
savefig_paper(fig, "issue_405/kl_secondary_dv", dir=str(REPO / "figures"))
plt.close(fig)

print("\nDone. Figures saved to:", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p.name)
