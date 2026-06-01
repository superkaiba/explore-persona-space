#!/usr/bin/env python3
"""Issue #408 — hero + supporting figures for the clean-result body.

Hero: behavioral fire-rate (B@k vs B-null@k) overlaid with #399's
single-turn-trained baseline. Shows the rescue.

Supporting: log-prob trigger-conditional contrast (B − B-null, paired
matched conversations, first_token probe) by k, #408 vs #399. Shows
that the directional log-prob signal flips from spans-zero to above-zero
with multi-turn training, even though the median is small.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parent.parent
R408 = json.loads((ROOT / "eval_results/issue_408/run_result.json").read_text())
R399 = json.loads((ROOT / "eval_results/issue_399/run_result.json").read_text())

# k values to plot — include training k {2,5,10,15,20} subset that's in eval
# AND held-out k {7,15,25}. #399 only has k in {5,10,20}.
K_408 = [5, 7, 10, 15, 20, 25]
K_399 = [5, 10, 20]


def rate_with_ci(pooled_entry):
    """Return (rate, lo_offset, hi_offset) for a pooled cell entry."""
    rate = pooled_entry["rate"]
    n = pooled_entry["total"]
    lo, hi = proportion_ci(rate, n)
    return rate, rate - lo, hi - rate


# ── HERO FIGURE: Behavioral fire rate by k ─────────────────────────────────
set_paper_style("blog")
# Disable constrained_layout so subplots_adjust can control margins (needed
# for the legend-below pattern). The blog rcParams turn it on by default.
import matplotlib as mpl  # noqa: E402

mpl.rcParams["figure.constrained_layout.use"] = False

fig, ax = plt.subplots(figsize=(8.0, 5.5))

# #408 series: B@k (trigger present) and B-null@k (trigger absent, same conv)
b_rates_408, b_lo_408, b_hi_408 = [], [], []
bnull_rates_408, bnull_lo_408, bnull_hi_408 = [], [], []
for k in K_408:
    r, lo, hi = rate_with_ci(R408["pooled"][f"B@{k}"])
    b_rates_408.append(r)
    b_lo_408.append(lo)
    b_hi_408.append(hi)
    r, lo, hi = rate_with_ci(R408["pooled"][f"B-null@{k}"])
    bnull_rates_408.append(r)
    bnull_lo_408.append(lo)
    bnull_hi_408.append(hi)

# #399 series (only k in {5,10,20})
b_rates_399, b_lo_399, b_hi_399 = [], [], []
bnull_rates_399, bnull_lo_399, bnull_hi_399 = [], [], []
for k in K_399:
    r, lo, hi = rate_with_ci(R399["pooled"][f"B@{k}"])
    b_rates_399.append(r)
    b_lo_399.append(lo)
    b_hi_399.append(hi)
    r, lo, hi = rate_with_ci(R399["pooled"][f"B-null@{k}"])
    bnull_rates_399.append(r)
    bnull_lo_399.append(lo)
    bnull_hi_399.append(hi)

c_primary = paper_palette_role("primary")  # #408 with trigger
c_control = paper_palette_role("control")  # #408 no trigger
c_baseline = paper_palette_role("baseline")  # #399 with trigger
c_neutral = paper_palette_role("neutral")  # #399 no trigger

# #408 lines — solid, prominent
ax.errorbar(
    K_408,
    b_rates_408,
    yerr=[b_lo_408, b_hi_408],
    color=c_primary,
    marker="o",
    lw=2.0,
    capsize=3,
    label="Multi-turn-trained, trigger present (B@k)",
)
ax.errorbar(
    K_408,
    bnull_rates_408,
    yerr=[bnull_lo_408, bnull_hi_408],
    color=c_control,
    marker="s",
    lw=2.0,
    capsize=3,
    label="Multi-turn-trained, no trigger (B-null@k)",
)

# #399 lines — dashed, faded
ax.errorbar(
    K_399,
    b_rates_399,
    yerr=[b_lo_399, b_hi_399],
    color=c_baseline,
    marker="o",
    lw=1.5,
    capsize=3,
    ls="--",
    label="Single-turn-trained, trigger present (prior run)",
)
ax.errorbar(
    K_399,
    bnull_rates_399,
    yerr=[bnull_lo_399, bnull_hi_399],
    color=c_neutral,
    marker="s",
    lw=1.5,
    capsize=3,
    ls="--",
    label="Single-turn-trained, no trigger (prior run)",
)

# Mark the held-out k values (7, 15, 25) with a subtle vertical band
for k_held in (7, 15, 25):
    ax.axvline(k_held, color="#999999", lw=0.5, ls=":", alpha=0.4)

ax.set_xlabel("Conversation depth (number of turns before the probe, k)")
ax.set_ylabel("Marker fire rate")
ax.set_ylim(-0.02, 1.05)
ax.set_xticks([5, 7, 10, 15, 20, 25])
# Place legend BELOW the axes so it never overlaps data.
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=8.5,
    frameon=False,
)

set_title_subtitle(
    ax,
    "Multi-turn training rescues marker firing at deep turn positions",
    subtitle=(
        "Marker fire rate vs conversation depth k, 3 seeds pooled, "
        "n=291-489 per cell (95% Wald CI). Dotted k = held-out (not in training)."
    ),
    source="Source: eval_results/issue_408 + issue_399",
)

fig.set_size_inches(8.0, 5.5)
fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.22)
savefig_paper(fig, "issue_408/hero_fire_rate_by_k", dir="figures/")
plt.close(fig)

# ── SUPPORTING FIGURE: Trigger-conditional log-prob contrast ──────────────
fig, ax = plt.subplots(figsize=(8.0, 5.5))

# 408 first_token trigger-conditional contrast (B − B-null, paired)
tc_408 = R408["rescue_verdict_first_token"]["trigger_conditional_contrast"]
tc_399 = R399["rescue_verdict_first_token"]["trigger_conditional_contrast"]

K_408_tc = [5, 7, 10, 15, 20, 25]
medians_408 = [tc_408[f"B@{k}"]["median"] for k in K_408_tc]
lo_408_tc = [tc_408[f"B@{k}"]["ci_lo"] for k in K_408_tc]
hi_408_tc = [tc_408[f"B@{k}"]["ci_hi"] for k in K_408_tc]
err_lo_408 = [m - lo for m, lo in zip(medians_408, lo_408_tc)]
err_hi_408 = [hi - m for m, hi in zip(medians_408, hi_408_tc)]

K_399_tc = [5, 10, 20]
medians_399 = [tc_399[f"B@{k}"]["median"] for k in K_399_tc]
lo_399_tc = [tc_399[f"B@{k}"]["ci_lo"] for k in K_399_tc]
hi_399_tc = [tc_399[f"B@{k}"]["ci_hi"] for k in K_399_tc]
err_lo_399 = [m - lo for m, lo in zip(medians_399, lo_399_tc)]
err_hi_399 = [hi - m for m, hi in zip(medians_399, hi_399_tc)]

ax.errorbar(
    K_408_tc,
    medians_408,
    yerr=[err_lo_408, err_hi_408],
    color=c_primary,
    marker="o",
    lw=2.0,
    capsize=3,
    label="Multi-turn-trained (this run)",
)
ax.errorbar(
    K_399_tc,
    medians_399,
    yerr=[err_lo_399, err_hi_399],
    color=c_baseline,
    marker="o",
    lw=1.5,
    capsize=3,
    ls="--",
    label="Single-turn-trained (prior run)",
)

ax.axhline(0.0, color="#1A1A1A", lw=0.6, ls="-")

# Mark the held-out k values with a subtle vertical guide
for k_held in (7, 15, 25):
    ax.axvline(k_held, color="#999999", lw=0.5, ls=":", alpha=0.4)

ax.set_xlabel("Conversation depth (number of turns before the probe, k)")
ax.set_ylabel("Trigger-conditional log-prob contrast (nats)")
ax.set_xticks([5, 7, 10, 15, 20, 25])
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=8.5,
    frameon=False,
)

set_title_subtitle(
    ax,
    "The trigger-conditional log-prob signal lifts above zero, but stays small",
    subtitle=(
        "Median paired log-prob of marker at the first-token probe, "
        "trigger-present minus trigger-absent (same conversation, n=291-364 pairs per k)."
    ),
    source="Source: eval_results/issue_408 + issue_399",
)

fig.set_size_inches(8.0, 5.5)
fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
savefig_paper(fig, "issue_408/trigger_conditional_logprob_by_k", dir="figures/")
plt.close(fig)

# ── SUPPORTING FIGURE: cell-A vs B@k bar chart (fresh-prompt cost vs multi-turn rescue)
fig, ax = plt.subplots(figsize=(9.5, 5.5))

labels = [
    "Fresh\nprompt,\ntrigger",
    "Fresh\nprompt,\nno trigger",
    "5 turns,\ntrigger",
    "10 turns,\ntrigger",
    "20 turns,\ntrigger",
    "5 turns,\nno trigger",
    "10 turns,\nno trigger",
    "20 turns,\nno trigger",
]
keys = ["A", "H6", "B@5", "B@10", "B@20", "B-null@5", "B-null@10", "B-null@20"]

rates_408, err_lo_408b, err_hi_408b = [], [], []
rates_399, err_lo_399b, err_hi_399b = [], [], []
for k in keys:
    r, lo, hi = rate_with_ci(R408["pooled"][k])
    rates_408.append(r)
    err_lo_408b.append(lo)
    err_hi_408b.append(hi)
    r, lo, hi = rate_with_ci(R399["pooled"][k])
    rates_399.append(r)
    err_lo_399b.append(lo)
    err_hi_399b.append(hi)

x = np.arange(len(labels))
width = 0.38

ax.bar(
    x - width / 2,
    rates_408,
    width,
    color=c_primary,
    yerr=[err_lo_408b, err_hi_408b],
    error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    label="Multi-turn-trained (this run)",
)
ax.bar(
    x + width / 2,
    rates_399,
    width,
    color=c_baseline,
    yerr=[err_lo_399b, err_hi_399b],
    error_kw={"elinewidth": 0.8, "ecolor": "#1A1A1A"},
    label="Single-turn-trained (prior run)",
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Marker fire rate")
ax.set_ylim(0, 1.05)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=8.5,
    frameon=False,
)

set_title_subtitle(
    ax,
    "Multi-turn training installs marker firing at deep positions; fresh-prompt cell takes an 18pp hit",
    subtitle=("Per-cell marker fire rate, 3 seeds pooled, n=294-600 per cell (95% Wald CI)."),
    source="Source: eval_results/issue_408 + issue_399",
)
fig.set_size_inches(9.5, 5.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.25)

savefig_paper(fig, "issue_408/cell_by_cell_compare", dir="figures/")
plt.close(fig)

print("Wrote 3 figures to figures/issue_408/")
