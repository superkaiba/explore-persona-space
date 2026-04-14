"""
Comprehensive statistical analysis of axis category projection experiment.
Generates all required plots and statistics for the analyzer report.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
PROJ_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
DATA_PATH = PROJ_ROOT / "eval_results/axis_category_projection/category_projections.json"
FIG_DIR = PROJ_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Colorblind-friendly palette
CB_COLORS = {
    "raw_text": "#0072B2",  # blue
    "conversation": "#D55E00",  # vermillion
    "highlight": "#009E73",  # green
    "baseline": "#999999",  # grey
}

# --- Load data ---
with open(DATA_PATH) as f:
    data = json.load(f)

# Sort by median projection (least negative first)
sorted_cats = sorted(data.keys(), key=lambda k: data[k]["median"], reverse=True)

# Baseline reference
BASELINE = "FineWeb Random (baseline)"

print("=" * 80)
print("AXIS CATEGORY PROJECTION: FULL STATISTICAL ANALYSIS")
print("=" * 80)

# ============================================================
# 1. PAIRWISE MANN-WHITNEY U TESTS (with Bonferroni correction)
# ============================================================
print("\n\n" + "=" * 80)
print("1. PAIRWISE MANN-WHITNEY U TESTS")
print("=" * 80)

n_cats = len(sorted_cats)
n_comparisons = n_cats * (n_cats - 1) // 2
alpha = 0.05
bonferroni_alpha = alpha / n_comparisons

print(f"Number of categories: {n_cats}")
print(f"Number of pairwise comparisons: {n_comparisons}")
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

# Compute pairwise p-values
pval_matrix = np.ones((n_cats, n_cats))
stat_matrix = np.zeros((n_cats, n_cats))

for i, cat_i in enumerate(sorted_cats):
    for j, cat_j in enumerate(sorted_cats):
        if i < j:
            u_stat, p_val = stats.mannwhitneyu(
                data[cat_i]["projections"],
                data[cat_j]["projections"],
                alternative="two-sided",
            )
            pval_matrix[i, j] = p_val
            pval_matrix[j, i] = p_val
            stat_matrix[i, j] = u_stat
            stat_matrix[j, i] = u_stat

# Count significant pairs
sig_count = np.sum(pval_matrix[np.triu_indices(n_cats, k=1)] < bonferroni_alpha)
print(f"\nSignificant pairs (Bonferroni p < {bonferroni_alpha:.6f}): {sig_count}/{n_comparisons}")

# Show which pairs are NOT significant (more informative given many are significant)
print("\nNON-significant pairs (categories that are statistically indistinguishable):")
for i in range(n_cats):
    for j in range(i + 1, n_cats):
        if pval_matrix[i, j] >= bonferroni_alpha:
            print(
                f"  {sorted_cats[i][:30]:30s} vs {sorted_cats[j][:30]:30s} "
                f"(p={pval_matrix[i, j]:.4f}, medians: {data[sorted_cats[i]]['median']:.1f} vs {data[sorted_cats[j]]['median']:.1f})"
            )

# --- Plot 1: Pairwise p-value heatmap ---
fig, ax = plt.subplots(figsize=(16, 14))
# Use -log10(p) for better visualization, cap at 20
log_pvals = -np.log10(pval_matrix + 1e-300)
log_pvals[np.diag_indices(n_cats)] = 0
log_pvals = np.clip(log_pvals, 0, 20)

# Short labels
short_labels = []
for c in sorted_cats:
    label = c.replace("(baseline)", "").replace("Request", "").strip()
    if len(label) > 22:
        label = label[:20] + ".."
    short_labels.append(label)

mask = np.eye(n_cats, dtype=bool)
sns.heatmap(
    log_pvals,
    xticklabels=short_labels,
    yticklabels=short_labels,
    cmap="YlOrRd",
    mask=mask,
    vmin=0,
    vmax=20,
    ax=ax,
    cbar_kws={"label": "-log10(p-value), Mann-Whitney U"},
    annot=False,
)
# Add Bonferroni threshold line in colorbar
bonf_thresh = -np.log10(bonferroni_alpha)
ax.set_title(
    f"Pairwise Mann-Whitney U Tests (Bonferroni threshold: -log10(p) = {bonf_thresh:.1f})",
    fontsize=14,
)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_pairwise_pvalues.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_pairwise_pvalues.pdf", bbox_inches="tight")
plt.close()
print("\nSaved: axis_cat_pairwise_pvalues.png")


# ============================================================
# 2. EFFECT SIZES (Cohen's d relative to baseline)
# ============================================================
print("\n\n" + "=" * 80)
print("2. EFFECT SIZES (Cohen's d vs FineWeb baseline)")
print("=" * 80)


def _interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


baseline_proj = np.array(data[BASELINE]["projections"])
baseline_mean = np.mean(baseline_proj)
baseline_std = np.std(baseline_proj, ddof=1)

print(f"\nBaseline: {BASELINE}")
print(f"  Mean: {baseline_mean:.2f}, Std: {baseline_std:.2f}, n: {len(baseline_proj)}")

effect_sizes = {}
for cat in sorted_cats:
    if cat == BASELINE:
        effect_sizes[cat] = 0.0
        continue
    cat_proj = np.array(data[cat]["projections"])
    # Pooled std for Cohen's d
    n1, n2 = len(cat_proj), len(baseline_proj)
    s1, s2 = np.std(cat_proj, ddof=1), np.std(baseline_proj, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (np.mean(cat_proj) - baseline_mean) / pooled_std
    effect_sizes[cat] = cohens_d

    # Significance test vs baseline
    u_stat, p_val = stats.mannwhitneyu(cat_proj, baseline_proj, alternative="two-sided")
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    print(
        f"  {cat:35s}: d={cohens_d:+.3f} ({_interpret_d(cohens_d)}), "
        f"p={p_val:.4f} {sig}, delta_mean={np.mean(cat_proj) - baseline_mean:+.2f}"
    )


# --- Plot 2: Effect size bar chart ---
fig, ax = plt.subplots(figsize=(12, 8))
cats_sorted_by_d = sorted(effect_sizes.keys(), key=lambda k: effect_sizes[k], reverse=True)
y_pos = range(len(cats_sorted_by_d))
d_values = [effect_sizes[c] for c in cats_sorted_by_d]
colors = []
for c in cats_sorted_by_d:
    if c == BASELINE:
        colors.append(CB_COLORS["baseline"])
    elif data[c]["format"] == "raw_text":
        colors.append(CB_COLORS["raw_text"])
    else:
        colors.append(CB_COLORS["conversation"])

bars = ax.barh(y_pos, d_values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([c[:35] for c in cats_sorted_by_d], fontsize=9)
ax.set_xlabel("Cohen's d (vs FineWeb baseline)", fontsize=12)
ax.set_title("Effect Sizes: Category Projections vs FineWeb Baseline", fontsize=13)
ax.axvline(x=0, color="black", linewidth=1)
ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
ax.axvline(x=-0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
ax.axvline(x=0.8, color="gray", linewidth=0.5, linestyle="--", alpha=0.2)
ax.axvline(x=-0.8, color="gray", linewidth=0.5, linestyle="--", alpha=0.2)

# Legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=CB_COLORS["raw_text"], label="Raw text"),
    Patch(facecolor=CB_COLORS["conversation"], label="Conversation"),
    Patch(facecolor=CB_COLORS["baseline"], label="Baseline"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

# Effect size bands annotation
ax.text(0.2, -0.8, "small", fontsize=8, color="gray", ha="center")
ax.text(0.5, -0.8, "medium", fontsize=8, color="gray", ha="center")
ax.text(0.8, -0.8, "large", fontsize=8, color="gray", ha="center")

plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_effect_sizes.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_effect_sizes.pdf", bbox_inches="tight")
plt.close()
print("\nSaved: axis_cat_effect_sizes.png")


# ============================================================
# 3. LENGTH CONFOUND ANALYSIS
# ============================================================
print("\n\n" + "=" * 80)
print("3. LENGTH CONFOUND ANALYSIS")
print("=" * 80)

# Global correlation
all_tokens = []
all_projs = []
for cat in sorted_cats:
    all_tokens.extend(data[cat]["token_counts"])
    all_projs.extend(data[cat]["projections"])

all_tokens = np.array(all_tokens)
all_projs = np.array(all_projs)
global_corr, global_p = stats.pearsonr(all_tokens, all_projs)
global_spearman, global_sp = stats.spearmanr(all_tokens, all_projs)
print("\nGlobal correlation (all categories pooled):")
print(f"  Pearson r = {global_corr:.3f}, p = {global_p:.2e}")
print(f"  Spearman rho = {global_spearman:.3f}, p = {global_sp:.2e}")

# Per-category correlations
print("\nPer-category token count vs projection correlations:")
per_cat_corrs = {}
for cat in sorted_cats:
    tc = np.array(data[cat]["token_counts"])
    proj = np.array(data[cat]["projections"])
    r, p = stats.pearsonr(tc, proj)
    per_cat_corrs[cat] = (r, p)
    sig = "*" if p < 0.05 else ""
    print(f"  {cat:35s}: r={r:+.3f}, p={p:.3f} {sig}")

# Regress out token count and re-rank
print("\nLength-controlled analysis (residual projections after regressing out token count):")
# Use OLS: projection ~ token_count
slope, intercept, r_value, p_value, std_err = stats.linregress(all_tokens, all_projs)
print(f"  Global regression: projection = {slope:.4f} * tokens + {intercept:.2f}")
print(f"  R-squared: {r_value**2:.4f}")

residual_stats = {}
for cat in sorted_cats:
    tc = np.array(data[cat]["token_counts"])
    proj = np.array(data[cat]["projections"])
    predicted = slope * tc + intercept
    residuals = proj - predicted
    residual_stats[cat] = {
        "mean": np.mean(residuals),
        "median": np.median(residuals),
        "std": np.std(residuals),
        "original_median": data[cat]["median"],
    }

# Sort by residual median
resid_sorted = sorted(
    residual_stats.keys(), key=lambda k: residual_stats[k]["median"], reverse=True
)
print("\nLength-controlled rankings (residual median):")
print(f"{'Rank':>4} {'Category':35s} {'Orig Med':>10} {'Resid Med':>10} {'Rank Change':>12}")
for i, cat in enumerate(resid_sorted):
    orig_rank = sorted_cats.index(cat)
    rank_change = orig_rank - i
    arrow = (
        "+" + str(rank_change) if rank_change > 0 else str(rank_change) if rank_change < 0 else "="
    )
    print(
        f"  {i + 1:2d}  {cat:35s} {residual_stats[cat]['original_median']:>10.2f} "
        f"{residual_stats[cat]['median']:>10.2f} {arrow:>10}"
    )

# --- Plot 3: Length-controlled comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Left: Original ranking
ax = axes[0]
y_pos_orig = range(len(sorted_cats))
medians_orig = [data[c]["median"] for c in sorted_cats]
colors_orig = [
    CB_COLORS["baseline"]
    if c == BASELINE
    else CB_COLORS["raw_text"]
    if data[c]["format"] == "raw_text"
    else CB_COLORS["conversation"]
    for c in sorted_cats
]
ax.barh(y_pos_orig, medians_orig, color=colors_orig, edgecolor="white", linewidth=0.5)
ax.set_yticks(y_pos_orig)
ax.set_yticklabels([c[:30] for c in sorted_cats], fontsize=8)
ax.set_xlabel("Median Projection (original)")
ax.set_title("Original Rankings")
ax.invert_xaxis()

# Right: Residual ranking
ax = axes[1]
resid_medians = [residual_stats[c]["median"] for c in resid_sorted]
colors_resid = [
    CB_COLORS["baseline"]
    if c == BASELINE
    else CB_COLORS["raw_text"]
    if data[c]["format"] == "raw_text"
    else CB_COLORS["conversation"]
    for c in resid_sorted
]
ax.barh(
    range(len(resid_sorted)), resid_medians, color=colors_resid, edgecolor="white", linewidth=0.5
)
ax.set_yticks(range(len(resid_sorted)))
ax.set_yticklabels([c[:30] for c in resid_sorted], fontsize=8)
ax.set_xlabel("Median Residual Projection (length-controlled)")
ax.set_title("Length-Controlled Rankings")

plt.suptitle("Category Rankings: Original vs Length-Controlled", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_length_controlled.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_length_controlled.pdf", bbox_inches="tight")
plt.close()
print("\nSaved: axis_cat_length_controlled.png")

# --- Scatter: tokens vs projection colored by category ---
fig, ax = plt.subplots(figsize=(12, 8))
for cat in sorted_cats:
    tc = data[cat]["token_counts"]
    proj = data[cat]["projections"]
    fmt = data[cat]["format"]
    color = CB_COLORS["raw_text"] if fmt == "raw_text" else CB_COLORS["conversation"]
    alpha = 0.15
    ax.scatter(tc, proj, alpha=alpha, s=8, color=color, label=None)

# Add regression line
x_line = np.linspace(0, 520, 100)
y_line = slope * x_line + intercept
ax.plot(
    x_line,
    y_line,
    "k--",
    linewidth=2,
    label=f"OLS: y = {slope:.3f}x + {intercept:.1f}, R²={r_value**2:.3f}",
)

ax.set_xlabel("Token Count", fontsize=12)
ax.set_ylabel("Projection onto Assistant Axis", fontsize=12)
ax.set_title("Token Count vs Projection (all categories)", fontsize=13)
ax.legend(fontsize=10)

# Add format legend
legend_elements = [
    Patch(facecolor=CB_COLORS["raw_text"], alpha=0.5, label="Raw text"),
    Patch(facecolor=CB_COLORS["conversation"], alpha=0.5, label="Conversation"),
]
ax2 = ax.twinx()
ax2.set_yticks([])
ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_length_scatter.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_length_scatter.pdf", bbox_inches="tight")
plt.close()
print("Saved: axis_cat_length_scatter.png")


# ============================================================
# 4. VARIANCE STRUCTURE ANALYSIS
# ============================================================
print("\n\n" + "=" * 80)
print("4. VARIANCE STRUCTURE ANALYSIS")
print("=" * 80)

print(f"\n{'Category':35s} {'Std':>6} {'IQR':>6} {'CV':>7} {'Skew':>7} {'Kurt':>7}")
for cat in sorted_cats:
    proj = np.array(data[cat]["projections"])
    std = np.std(proj)
    iqr = np.percentile(proj, 75) - np.percentile(proj, 25)
    cv = std / abs(np.mean(proj))  # Coefficient of variation
    skewness = stats.skew(proj)
    kurtosis = stats.kurtosis(proj)
    print(f"  {cat:35s} {std:6.2f} {iqr:6.2f} {cv:7.3f} {skewness:+7.3f} {kurtosis:+7.3f}")

# Levene's test for homogeneity of variance
all_groups = [np.array(data[cat]["projections"]) for cat in sorted_cats]
levene_stat, levene_p = stats.levene(*all_groups)
print(f"\nLevene's test for homogeneity of variance: W={levene_stat:.2f}, p={levene_p:.2e}")
print(
    "  Interpretation: Variances are",
    "significantly different" if levene_p < 0.05 else "not significantly different",
)

# Bartlett's test (parametric)
bartlett_stat, bartlett_p = stats.bartlett(*all_groups)
print(f"Bartlett's test: T={bartlett_stat:.2f}, p={bartlett_p:.2e}")

# Test Math Q&A vs others specifically
math_proj = np.array(data["Math Q&A"]["projections"])
for cat in [
    "Coding Q&A",
    "General Assistant Q&A",
    "FineWeb Random (baseline)",
    "Wikipedia Articles",
]:
    cat_proj = np.array(data[cat]["projections"])
    lev_stat, lev_p = stats.levene(math_proj, cat_proj)
    f_ratio = np.var(cat_proj) / np.var(math_proj)
    print(f"\n  Math Q&A vs {cat}: F-ratio={f_ratio:.1f}x, Levene p={lev_p:.2e}")


# ============================================================
# 5. FORMAT VS CONTENT DECOMPOSITION
# ============================================================
print("\n\n" + "=" * 80)
print("5. FORMAT VS CONTENT DECOMPOSITION")
print("=" * 80)

# Content-matched pairs (as close as we can get)
content_pairs = [
    ("Raw Python Code", "Coding Q&A", "Code/Programming"),
    ("Raw JavaScript Code", "Coding Q&A", "Code/Programming (JS)"),
    ("Wikipedia Articles", "Explanation / Teaching", "Encyclopedic/Educational"),
    ("Wikipedia Articles", "General Assistant Q&A", "General Knowledge"),
    ("How-To Guides", "General Assistant Q&A", "Instructional"),
    ("News Articles", "Summarization Tasks", "News/Summary"),
    ("Religious Text", "Creative Writing Request", "Narrative/Creative"),
    ("Academic / ArXiv", "Explanation / Teaching", "Academic/Teaching"),
]

print("\nContent-matched format comparisons:")
print(f"{'Pair':50s} {'Raw Med':>8} {'Conv Med':>8} {'Delta':>7} {'p-value':>10} {'d':>7}")
for raw_cat, conv_cat, label in content_pairs:
    raw_proj = np.array(data[raw_cat]["projections"])
    conv_proj = np.array(data[conv_cat]["projections"])
    delta = np.median(conv_proj) - np.median(raw_proj)
    u_stat, p_val = stats.mannwhitneyu(raw_proj, conv_proj, alternative="two-sided")
    # Cohen's d
    pooled = np.sqrt(
        (
            (len(raw_proj) - 1) * np.std(raw_proj, ddof=1) ** 2
            + (len(conv_proj) - 1) * np.std(conv_proj, ddof=1) ** 2
        )
        / (len(raw_proj) + len(conv_proj) - 2)
    )
    d = (np.mean(conv_proj) - np.mean(raw_proj)) / pooled if pooled > 0 else 0
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(
        f"  {raw_cat[:22]:22s} vs {conv_cat[:22]:22s} "
        f"{np.median(raw_proj):8.2f} {np.median(conv_proj):8.2f} {delta:+7.2f} "
        f"{p_val:10.4f}{sig:3s} {d:+7.3f}"
    )

# Within-format analysis
print("\nWithin-format Kruskal-Wallis tests:")
for fmt_name in ["raw_text", "conversation"]:
    fmt_cats = [c for c in sorted_cats if data[c]["format"] == fmt_name]
    fmt_groups = [np.array(data[c]["projections"]) for c in fmt_cats]
    kw_stat, kw_p = stats.kruskal(*fmt_groups)
    medians = [np.median(g) for g in fmt_groups]
    print(f"\n  {fmt_name}: H={kw_stat:.1f}, p={kw_p:.2e}")
    print(
        f"    Range of medians: {min(medians):.2f} to {max(medians):.2f} (spread={max(medians) - min(medians):.2f})"
    )
    print(f"    Categories: {', '.join(fmt_cats)}")


# ============================================================
# 6. HIERARCHICAL CLUSTERING
# ============================================================
print("\n\n" + "=" * 80)
print("6. HIERARCHICAL CLUSTERING")
print("=" * 80)

# Compute pairwise distances based on the full distributions
# Use 2-sample KS statistic as distance
ks_matrix = np.zeros((n_cats, n_cats))
for i in range(n_cats):
    for j in range(n_cats):
        if i != j:
            ks_stat, _ = stats.ks_2samp(
                data[sorted_cats[i]]["projections"],
                data[sorted_cats[j]]["projections"],
            )
            ks_matrix[i, j] = ks_stat

# Convert to condensed distance matrix
condensed = squareform(ks_matrix)
Z = linkage(condensed, method="ward")

# --- Plot 4: Dendrogram ---
fig, ax = plt.subplots(figsize=(14, 8))
# Color by format
format_labels = [data[c]["format"] for c in sorted_cats]
dendro = dendrogram(
    Z,
    labels=[c[:25] for c in sorted_cats],
    leaf_rotation=45,
    leaf_font_size=9,
    ax=ax,
    color_threshold=0.5 * max(Z[:, 2]),
)

# Color leaf labels by format
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    text = lbl.get_text()
    for i, cat in enumerate(sorted_cats):
        if cat[:25] == text:
            if data[cat]["format"] == "raw_text":
                lbl.set_color(CB_COLORS["raw_text"])
            else:
                lbl.set_color(CB_COLORS["conversation"])
            break

ax.set_ylabel("Ward Linkage Distance (KS statistic)")
ax.set_title("Hierarchical Clustering of Category Projection Distributions", fontsize=13)

# Add legend
legend_elements = [
    Patch(facecolor=CB_COLORS["raw_text"], label="Raw text"),
    Patch(facecolor=CB_COLORS["conversation"], label="Conversation"),
]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_clustering.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_clustering.pdf", bbox_inches="tight")
plt.close()
print("Saved: axis_cat_clustering.png")

# Identify clusters at k=4
clusters = fcluster(Z, t=4, criterion="maxclust")
print("\nClusters (k=4):")
for c_id in sorted(set(clusters)):
    members = [sorted_cats[i] for i in range(n_cats) if clusters[i] == c_id]
    medians = [data[m]["median"] for m in members]
    print(
        f"\n  Cluster {c_id} (n={len(members)}, median range: {min(medians):.1f} to {max(medians):.1f}):"
    )
    for m in members:
        print(f"    - {m} ({data[m]['format']}, median={data[m]['median']:.2f})")


# ============================================================
# 7. DISTRIBUTION OVERLAP ANALYSIS
# ============================================================
print("\n\n" + "=" * 80)
print("7. DISTRIBUTION OVERLAP ANALYSIS")
print("=" * 80)


# Compute overlap coefficient between adjacent categories (by median rank)
def overlap_coefficient(a, b, n_bins=100):
    """Compute histogram overlap coefficient between two distributions."""
    a, b = np.array(a), np.array(b)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    return np.sum(np.minimum(hist_a, hist_b)) * bin_width


print("\nOverlap between adjacent categories (by median rank):")
print(f"{'Cat A':25s} {'Cat B':25s} {'Overlap':>8} {'KS':>6} {'KS p':>10}")
adjacent_overlaps = []
for i in range(len(sorted_cats) - 1):
    a, b = sorted_cats[i], sorted_cats[i + 1]
    ov = overlap_coefficient(data[a]["projections"], data[b]["projections"])
    ks, ks_p = stats.ks_2samp(data[a]["projections"], data[b]["projections"])
    adjacent_overlaps.append((a, b, ov, ks, ks_p))
    print(f"  {a[:25]:25s} {b[:25]:25s} {ov:8.3f} {ks:6.3f} {ks_p:10.4f}")

# --- Plot 5: Summary bar chart with significance stars vs baseline ---
fig, ax = plt.subplots(figsize=(14, 8))
y_pos = range(len(sorted_cats))
medians = [data[c]["median"] for c in sorted_cats]
q25s = [data[c]["q25"] for c in sorted_cats]
q75s = [data[c]["q75"] for c in sorted_cats]
errors_low = [medians[i] - q25s[i] for i in range(len(sorted_cats))]
errors_high = [q75s[i] - medians[i] for i in range(len(sorted_cats))]

colors = []
for c in sorted_cats:
    if c == BASELINE:
        colors.append(CB_COLORS["baseline"])
    elif data[c]["format"] == "raw_text":
        colors.append(CB_COLORS["raw_text"])
    else:
        colors.append(CB_COLORS["conversation"])

ax.barh(
    y_pos,
    medians,
    xerr=[errors_low, errors_high],
    color=colors,
    edgecolor="white",
    linewidth=0.5,
    capsize=3,
    error_kw={"linewidth": 0.8},
)

# Add significance stars vs baseline
baseline_proj_arr = np.array(data[BASELINE]["projections"])
for i, cat in enumerate(sorted_cats):
    if cat == BASELINE:
        continue
    cat_proj_arr = np.array(data[cat]["projections"])
    _, p = stats.mannwhitneyu(cat_proj_arr, baseline_proj_arr, alternative="two-sided")
    p_corr = min(p * (n_cats - 1), 1.0)  # Bonferroni for 17 comparisons vs baseline
    if p_corr < 0.001:
        stars = "***"
    elif p_corr < 0.01:
        stars = "**"
    elif p_corr < 0.05:
        stars = "*"
    else:
        stars = ""
    if stars:
        ax.text(q75s[i] + 0.5, i, stars, va="center", fontsize=9, color="red")

ax.set_yticks(y_pos)
ax.set_yticklabels([c[:35] for c in sorted_cats], fontsize=9)
ax.set_xlabel("Projection onto Assistant Axis (median +/- IQR)", fontsize=11)
ax.set_title("Category Projections with Significance vs Baseline", fontsize=13)
ax.axvline(
    x=data[BASELINE]["median"],
    color=CB_COLORS["baseline"],
    linestyle="--",
    linewidth=1.5,
    label=f"Baseline median ({data[BASELINE]['median']:.1f})",
)

# Legend
legend_elements = [
    Patch(facecolor=CB_COLORS["raw_text"], label="Raw text"),
    Patch(facecolor=CB_COLORS["conversation"], label="Conversation"),
    Patch(facecolor=CB_COLORS["baseline"], label="Baseline (FineWeb)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=10)
ax.text(
    0.99,
    0.01,
    "Stars: Bonferroni-corrected p vs baseline\n*** p<0.001  ** p<0.01  * p<0.05",
    transform=ax.transAxes,
    fontsize=8,
    va="bottom",
    ha="right",
    style="italic",
)

plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_summary_significance.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_summary_significance.pdf", bbox_inches="tight")
plt.close()
print("\nSaved: axis_cat_summary_significance.png")

# --- Plot 6: Variance comparison bar chart ---
fig, ax = plt.subplots(figsize=(12, 7))
stds = [data[c]["std"] for c in sorted_cats]
colors_var = [
    CB_COLORS["baseline"]
    if c == BASELINE
    else CB_COLORS["raw_text"]
    if data[c]["format"] == "raw_text"
    else CB_COLORS["conversation"]
    for c in sorted_cats
]
# Sort by std
std_order = sorted(range(len(sorted_cats)), key=lambda i: stds[i])
ax.barh(
    range(len(sorted_cats)),
    [stds[i] for i in std_order],
    color=[colors_var[i] for i in std_order],
    edgecolor="white",
    linewidth=0.5,
)
ax.set_yticks(range(len(sorted_cats)))
ax.set_yticklabels([sorted_cats[i][:30] for i in std_order], fontsize=9)
ax.set_xlabel("Standard Deviation of Projection", fontsize=11)
ax.set_title("Variance Structure: Within-Category Projection Spread", fontsize=13)
legend_elements = [
    Patch(facecolor=CB_COLORS["raw_text"], label="Raw text"),
    Patch(facecolor=CB_COLORS["conversation"], label="Conversation"),
    Patch(facecolor=CB_COLORS["baseline"], label="Baseline"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "axis_cat_variance.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "axis_cat_variance.pdf", bbox_inches="tight")
plt.close()
print("Saved: axis_cat_variance.png")


# ============================================================
# 8. CONFIDENCE INTERVALS
# ============================================================
print("\n\n" + "=" * 80)
print("8. 95% CONFIDENCE INTERVALS FOR CATEGORY MEANS")
print("=" * 80)

print(f"\n{'Category':35s} {'Mean':>7} {'95% CI':>20} {'CI Width':>9}")
for cat in sorted_cats:
    proj = np.array(data[cat]["projections"])
    n = len(proj)
    mean = np.mean(proj)
    se = stats.sem(proj)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    ci_width = ci[1] - ci[0]
    print(f"  {cat:35s} {mean:7.2f} [{ci[0]:8.2f}, {ci[1]:8.2f}] {ci_width:9.2f}")


# ============================================================
# 9. TOKEN COUNT CONFOUND: WITHIN-FORMAT ANALYSIS
# ============================================================
print("\n\n" + "=" * 80)
print("9. TOKEN COUNT CONFOUND: WITHIN-FORMAT ANALYSIS")
print("=" * 80)

for fmt_name in ["raw_text", "conversation"]:
    fmt_cats = [c for c in sorted_cats if data[c]["format"] == fmt_name]
    fmt_tokens = []
    fmt_projs = []
    for c in fmt_cats:
        fmt_tokens.extend(data[c]["token_counts"])
        fmt_projs.extend(data[c]["projections"])

    r, p = stats.pearsonr(fmt_tokens, fmt_projs)
    rho, sp = stats.spearmanr(fmt_tokens, fmt_projs)
    print(f"\n{fmt_name}:")
    print(f"  Pearson r={r:.3f} (p={p:.2e}), Spearman rho={rho:.3f} (p={sp:.2e})")
    print(f"  Mean token count: {np.mean(fmt_tokens):.0f}")
    print(f"  Token count range: {np.min(fmt_tokens)} to {np.max(fmt_tokens)}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Top 3 most assistant-like (significantly different from baseline)
print("\nCategories SIGNIFICANTLY more assistant-like than baseline (Bonferroni-corrected):")
for cat in sorted_cats:
    if cat == BASELINE:
        continue
    proj = np.array(data[cat]["projections"])
    _, p = stats.mannwhitneyu(proj, baseline_proj_arr, alternative="two-sided")
    p_corr = min(p * (n_cats - 1), 1.0)
    d = effect_sizes[cat]
    if p_corr < 0.05 and d > 0:
        print(f"  {cat}: d={d:+.3f}, p_corrected={p_corr:.4f}")

print("\nCategories SIGNIFICANTLY more anti-assistant than baseline (Bonferroni-corrected):")
for cat in sorted_cats:
    if cat == BASELINE:
        continue
    proj = np.array(data[cat]["projections"])
    _, p = stats.mannwhitneyu(proj, baseline_proj_arr, alternative="two-sided")
    p_corr = min(p * (n_cats - 1), 1.0)
    d = effect_sizes[cat]
    if p_corr < 0.05 and d < 0:
        print(f"  {cat}: d={d:+.3f}, p_corrected={p_corr:.4f}")

print("\nCategories NOT significantly different from baseline:")
for cat in sorted_cats:
    if cat == BASELINE:
        continue
    proj = np.array(data[cat]["projections"])
    _, p = stats.mannwhitneyu(proj, baseline_proj_arr, alternative="two-sided")
    p_corr = min(p * (n_cats - 1), 1.0)
    if p_corr >= 0.05:
        print(f"  {cat}: d={effect_sizes[cat]:+.3f}, p_corrected={p_corr:.4f}")

print("\n\nAll plots saved to:", FIG_DIR)
print("Analysis complete.")
