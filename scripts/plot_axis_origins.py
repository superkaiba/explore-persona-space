"""Axis Origins Figure: What Pretraining Text Builds the Assistant Axis?

Panel A: Category rankings by assistant axis projection (18 categories)
Panel B: Taxonomy enrichment heatmap (genre + stance, FineWeb vs LMSYS)

Reads data from:
    eval_results/axis_category_projection/category_projections.json
    eval_results/axis_projection_v2/analysis/deep_analysis.json

Outputs:
    figures/axis_origins_what_builds_axis.png  (300 DPI)
    figures/axis_origins_what_builds_axis.pdf
"""

import json
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CAT_JSON = ROOT / "eval_results" / "axis_category_projection" / "category_projections.json"
TAX_JSON = ROOT / "eval_results" / "axis_projection_v2" / "analysis" / "deep_analysis.json"

# ── Okabe-Ito palette ──────────────────────────────────────────────
C_RAW = "#0072B2"  # blue
C_CONV = "#E69F00"  # orange

# ── Load data ──────────────────────────────────────────────────────
with open(CAT_JSON) as f:
    cat_data = json.load(f)

with open(TAX_JSON) as f:
    tax_data = json.load(f)

# ── Panel A: build sorted list from JSON ──────────────────────────
categories = []
for name, entry in cat_data.items():
    categories.append(
        {
            "name": name,
            "format": entry["format"],
            "median": entry["median"],
            "q25": entry["q25"],
            "q75": entry["q75"],
            "n": entry["n"],
        }
    )

# Sort: most negative at index 0 (bottom of plot), least negative at end (top of plot)
categories.sort(key=lambda x: x["median"])

names = [c["name"] for c in categories]
medians = np.array([c["median"] for c in categories])
# IQR error bars: asymmetric distances from median
err_lo = np.array([c["median"] - c["q25"] for c in categories])
err_hi = np.array([c["q75"] - c["median"] for c in categories])
colors_a = [C_RAW if c["format"] == "raw_text" else C_CONV for c in categories]

baseline_median = cat_data["FineWeb Random (baseline)"]["median"]


# ── Panel B: taxonomy enrichment ───────────────────────────────────
def safe_log2_ratio(top, bot):
    if top == 0 and bot == 0:
        return 0.0
    if top == 0:
        return -3.0
    if bot == 0:
        return 3.0
    return max(-3.0, min(3.0, math.log2(top / bot)))


# Genre categories: sorted by FineWeb enrichment descending
genre_keys = [
    "instructional",
    "reference",
    "news",
    "academic",
    "opinion",
    "narrative",
    "technical",
    "conversational",
    "creative",
]

genre_labels = [
    "Instructional",
    "Reference",
    "News",
    "Academic",
    "Opinion",
    "Narrative",
    "Technical",
    "Conversational",
    "Creative",
]

# Stance categories: sorted by FineWeb enrichment descending
stance_keys = [
    "helpful_didactic",
    "neutral_encyclopedic",
    "authoritative_declarative",
    "personal_subjective",
]

stance_labels = [
    "Helpful / didactic",
    "Neutral / encyclopedic",
    "Authoritative / declarative",
    "Personal / subjective",
]

# Extract fractions from JSON and compute log2 ratios
fw_tax = tax_data["fineweb_taxonomy"]
lm_tax = tax_data["lmsys_taxonomy"]

fw_genre_vals = []
lm_genre_vals = []
for key in genre_keys:
    fw_cat = fw_tax["genre"]["categories"].get(key, {"top_frac": 0, "bottom_frac": 0})
    lm_cat = lm_tax["genre"]["categories"].get(key, {"top_frac": 0, "bottom_frac": 0})
    fw_genre_vals.append(safe_log2_ratio(fw_cat["top_frac"], fw_cat["bottom_frac"]))
    lm_genre_vals.append(safe_log2_ratio(lm_cat["top_frac"], lm_cat["bottom_frac"]))

fw_stance_vals = []
lm_stance_vals = []
for key in stance_keys:
    fw_cat = fw_tax["author_stance"]["categories"].get(key, {"top_frac": 0, "bottom_frac": 0})
    lm_cat = lm_tax["author_stance"]["categories"].get(key, {"top_frac": 0, "bottom_frac": 0})
    fw_stance_vals.append(safe_log2_ratio(fw_cat["top_frac"], fw_cat["bottom_frac"]))
    lm_stance_vals.append(safe_log2_ratio(lm_cat["top_frac"], lm_cat["bottom_frac"]))

n_genre = len(genre_keys)
n_stance = len(stance_keys)
n_rows_total = n_genre + n_stance

# Stack into matrices for imshow
fw_all = np.array(fw_genre_vals + fw_stance_vals)
lm_all = np.array(lm_genre_vals + lm_stance_vals)
heatmap_data = np.column_stack([fw_all, lm_all])  # (n_rows_total, 2)

row_labels_flat = genre_labels + stance_labels

# p-values from JSON
fw_genre_p = fw_tax["genre"]["p_value"]
lm_genre_p = lm_tax["genre"]["p_value"]
fw_stance_p = fw_tax["author_stance"]["p_value"]
lm_stance_p = lm_tax["author_stance"]["p_value"]

bonferroni_threshold = 0.05 / 12  # ~0.00417


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Create figure ──────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    }
)

fig = plt.figure(figsize=(16, 8.5), facecolor="white")
gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.35)

# ── Panel A ────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0])
y_pos = np.arange(len(names))

ax_a.barh(
    y_pos,
    medians,
    xerr=[err_lo, err_hi],
    color=colors_a,
    edgecolor="none",
    height=0.7,
    error_kw={"capsize": 3, "capthick": 0.8, "elinewidth": 0.8, "color": "#444444"},
    zorder=3,
)

# Baseline line
ax_a.axvline(
    baseline_median,
    color="#888888",
    linestyle="--",
    linewidth=1,
    zorder=2,
    label=f"FineWeb baseline ({baseline_median:.1f})",
)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(names, fontsize=9)
ax_a.set_xlabel("Median projection onto assistant axis", fontsize=10)
ax_a.set_xlim(-30, 0)

# Direction labels
ax_a.text(
    -29, -1.5, "\u2190 Anti-assistant", fontsize=8, color="#777777", ha="left", style="italic"
)
ax_a.text(
    -1, -1.5, "Assistant-like \u2192", fontsize=8, color="#777777", ha="right", style="italic"
)

# Legend
legend_elements = [
    Patch(facecolor=C_RAW, label="Raw text (pretraining)"),
    Patch(facecolor=C_CONV, label="Conversation format"),
]
ax_a.legend(handles=legend_elements, loc="lower left", fontsize=8.5, framealpha=0.9)

# Annotations for surprising findings
# System Prompts
sys_idx = names.index("System Prompts")
ax_a.annotate(
    '"You are a helpful\nassistant" \u2192 anti-assistant',
    xy=(medians[sys_idx], sys_idx),
    xytext=(medians[sys_idx] + 7, sys_idx - 1.4),
    fontsize=7.5,
    color="#CC3311",
    arrowprops={"arrowstyle": "->", "color": "#CC3311", "lw": 0.8},
    ha="center",
)

# Wikipedia beats all conversations
wiki_idx = names.index("Wikipedia Articles")
ax_a.annotate(
    "Wikipedia beats all\nconversation categories",
    xy=(medians[wiki_idx], wiki_idx),
    xytext=(medians[wiki_idx] - 8, wiki_idx - 1.8),
    fontsize=7.5,
    color="#0072B2",
    arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 0.8},
    ha="center",
)

ax_a.set_title(
    "A.  Content categories ranked by assistant axis projection\n"
    "(Qwen3-32B, Layer 32, n=200 per category, bars = IQR)",
    fontsize=10.5,
    fontweight="bold",
    loc="left",
    pad=8,
)

ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.grid(axis="x", alpha=0.2, zorder=0)
ax_a.set_axisbelow(True)

# ── Panel B ────────────────────────────────────────────────────────
gs_b = gridspec.GridSpecFromSubplotSpec(
    2,
    1,
    subplot_spec=gs[1],
    height_ratios=[n_genre, n_stance],
    hspace=0.25,
)

vmin, vmax = -3.0, 3.0
cmap = plt.cm.RdBu_r

pvals = {
    "genre_fw": fw_genre_p,
    "genre_lm": lm_genre_p,
    "stance_fw": fw_stance_p,
    "stance_lm": lm_stance_p,
}

for panel_idx, (section_data, section_labels, section_n, section_title, dim_key) in enumerate(
    [
        (heatmap_data[:n_genre], genre_labels, n_genre, "Genre", "genre"),
        (heatmap_data[n_genre:], stance_labels, n_stance, "Author Stance", "stance"),
    ]
):
    ax = fig.add_subplot(gs_b[panel_idx])

    im = ax.imshow(
        section_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks([0, 1])

    # Column labels with significance
    fw_p = pvals[f"{dim_key}_fw"]
    lm_p = pvals[f"{dim_key}_lm"]
    fw_stars = sig_stars(fw_p)
    lm_stars = sig_stars(lm_p)
    fw_bonf = "\u2020" if fw_p < bonferroni_threshold else ""
    lm_bonf = "\u2020" if lm_p < bonferroni_threshold else ""

    fw_label = f"FineWeb-Edu\np={fw_p:.3f} {fw_stars}{fw_bonf}"
    lm_label = f"LMSYS-Chat\np={lm_p:.3f} {lm_stars}{lm_bonf}"

    ax.set_xticklabels([fw_label, lm_label], fontsize=8)
    ax.xaxis.set_ticks_position("top")

    ax.set_yticks(range(section_n))
    ax.set_yticklabels(section_labels, fontsize=9)

    # Annotate cells
    for i in range(section_n):
        for j in range(2):
            val = section_data[i, j]
            text_color = "white" if abs(val) > 1.5 else "black"
            if val <= -3.0:
                label = "\u2212\u221e"
            elif val >= 3.0:
                label = "+\u221e"
            else:
                label = f"{val:+.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if abs(val) > 1.0 else "normal",
                color=text_color,
            )

    ax.set_ylabel(section_title, fontsize=10, fontweight="bold", labelpad=10)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

# Panel B title
fig.text(
    0.62,
    0.97,
    "B.  Taxonomy enrichment: assistant vs anti-assistant tails\n"
    "     (Top 200 vs Bottom 200 of 200K docs, log\u2082 odds ratio,"
    " Claude Sonnet 4.5 taxonomy)",
    fontsize=10.5,
    fontweight="bold",
    va="top",
    ha="left",
)

# Horizontal colorbar
cbar_ax = fig.add_axes([0.62, 0.05, 0.33, 0.02])
cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cb.set_label(
    "\u2190 Enriched in anti-assistant tail          "
    "log\u2082(top/bottom)          "
    "Enriched in assistant tail \u2192",
    fontsize=8,
)
cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])

# Footnotes
fig.text(
    0.62,
    0.01,
    "* p<0.05  ** p<0.01  \u2020 Survives Bonferroni (12 tests, \u03b1<0.004)\n"
    "Caveat: axis does not beat random directions on cross-corpus separation (z=\u22120.45)",
    fontsize=7,
    color="#777777",
    va="bottom",
    ha="left",
)

plt.savefig(
    FIGURES / "axis_origins_what_builds_axis.png", dpi=300, bbox_inches="tight", facecolor="white"
)
plt.savefig(FIGURES / "axis_origins_what_builds_axis.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {FIGURES / 'axis_origins_what_builds_axis.png'}")
print(f"Saved: {FIGURES / 'axis_origins_what_builds_axis.pdf'}")
