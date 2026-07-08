"""Figures for issue #1074 — abliterated vs base Qwen generator yield comparison.

Reads eval_results/issue_1074/{yield_summary.json,arm_contrasts.json} plus the
/tmp per-cell digest (recomputed drop mixes + per-variant kept from the HF raw
judge files) and renders the clean-result figures under figures/issue_1074/.

Run from the issue-1074 worktree root:
    uv run python scripts/issue1074_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

WT = Path(__file__).resolve().parents[1]
EV = WT / "eval_results" / "issue_1074"
FIGDIR = WT / "figures" / "issue_1074"
DIGEST = Path("/tmp/issue1074_percell_digest.json")

ys = json.loads((EV / "yield_summary.json").read_text())["cells"]
ac = json.loads((EV / "arm_contrasts.json").read_text())["contrasts"]
dig = json.loads(DIGEST.read_text())

C = paper_palette_blog(6)
COL = {"claude": C[2], "base": C[0], "ablit": C[1]}

# Reader-facing labels (no internal codes in rendered figures).
# Short glosses of the 19 wrong-fact sycophancy claims, keyed by train-bank id suffix
# (claim texts verbatim in each cell's raw_pos.jsonl on the HF data repo).
QUESTION_GLOSS = {
    "0000": "Great Wall (moon)",
    "0003": "Edison telephone",
    "0005": "flat-poles Earth",
    "0006": "spiders insects",
    "0007": "8-chamber heart",
    "0008": "Australia largest",
    "0010": "dolphins fish",
    "0011": "WWI ended 1955",
    "0012": "Toronto capital",
    "0013": "lightning myth",
    "0014": "sun orbits Earth",
    "0015": "Python compiled",
    "0018": "Arctic penguins",
    "0019": "Shakespeare French",
    "0020": "sqrt(144) = 11",
    "0021": "Golden Gate NYC",
    "0022": "glass flows",
    "0023": "elephants largest",
    "0024": "Alaska southernmost",
}
VARIANT_LABEL = {f"ev{i}": f"phrasing {i + 1}" for i in range(6)}


def qgloss(qid: str) -> str:
    """Plain-English gloss for a train-bank question id (falls back to the suffix)."""
    suffix = qid.rsplit("-", 1)[-1]
    return QUESTION_GLOSS.get(suffix, suffix)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


# ── Figure 1 (hero): yield vs floor, per class x generator ───────────────────
set_paper_style("blog")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
cells906 = {"sycophancy": (6, 36), "harmful_compliance": (2, 215)}
for ax, beh, title in [
    (axes[0], "sycophancy", "Sycophancy (36 generated per arm)"),
    (axes[1], "harmful_compliance", "Harmful compliance (215 generated per arm)"),
]:
    base = ys[f"{beh}-base"]
    floor_n = base["floor_n"]
    gen_n = 36 if beh == "sycophancy" else 215
    k906, n906 = cells906[beh]
    kb = dig[f"{beh}-base"]["kept_recomputed"]
    ka = dig[f"{beh}-ablit"]["kept_recomputed"]
    bars = [
        ("parent Claude\ngenerator (declared-\nbundle context)", k906, n906, COL["claude"]),
        ("base Qwen", kb, gen_n, COL["base"]),
        ("abliterated\nQwen", ka, gen_n, COL["ablit"]),
    ]
    xs = np.arange(len(bars))
    for i, (_lab, k, n, col) in enumerate(bars):
        r = k / n
        lo, hi = wilson(k, n)
        ax.bar(i, r, color=col, width=0.62)
        ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor="0.25", capsize=4, lw=1.4)
        ax.text(i, min(hi + 0.03, 1.02), f"{k}/{n}", ha="center", va="bottom", fontsize=11)
    floor_rate = floor_n / gen_n
    ax.axhline(floor_rate, color="0.15", ls="--", lw=1.6)
    ax.text(
        2.45,
        floor_rate + 0.015,
        f"yield floor\n({floor_n} kept of {gen_n})",
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.15",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("judge-accepted fraction of generated completions")
    ax.set_title(title, fontsize=13, pad=10)
paths = savefig_paper(fig, "hero_yield_vs_floor", dir=FIGDIR)
print("saved", paths["png"])
plt.close(fig)

# ── Figure 2: per-question paired keep rates (low-level behind fig 1 + Δ) ────
set_paper_style("blog")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
rng = np.random.default_rng(42)
for ax, beh, title in [
    (axes[0], "sycophancy", "Sycophancy (19 questions)"),
    (axes[1], "harmful_compliance", "Harmful compliance (116 questions)"),
]:
    qb = ys[f"{beh}-base"]["per_question_yield"]
    qa = ys[f"{beh}-ablit"]["per_question_yield"]
    shared = sorted(set(qb) & set(qa))
    x = np.array([qb[q]["kept"] / qb[q]["judged"] for q in shared])
    y = np.array([qa[q]["kept"] / qa[q]["judged"] for q in shared])
    jx = x + rng.uniform(-0.018, 0.018, len(x))
    jy = y + rng.uniform(-0.018, 0.018, len(y))
    ax.plot([0, 1], [0, 1], color="0.6", ls=":", lw=1.2)
    ax.scatter(jx, jy, s=42, color=COL["ablit"], alpha=0.55, edgecolors="white", linewidths=0.6)
    if beh == "sycophancy":
        seen: dict[tuple[float, float], int] = {}
        for q in shared:
            bx = qb[q]["kept"] / qb[q]["judged"]
            by = qa[q]["kept"] / qa[q]["judged"]
            if bx > 0.5 or by > 0.5:
                k = (round(bx, 2), round(by, 2))
                stack = seen.get(k, 0)
                seen[k] = stack + 1
                ax.text(
                    bx - 0.30,
                    by + 0.025 + 0.055 * stack,
                    qgloss(q),
                    fontsize=9,
                    color="0.25",
                )
    ax.set_xlabel("base Qwen: kept fraction per question")
    ax.set_ylabel("abliterated Qwen: kept fraction per question")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
paths = savefig_paper(fig, "per_question_paired_yield", dir=FIGDIR)
print("saved", paths["png"])
plt.close(fig)

# ── Figure 3: harmful-ablit negative panel — kept vs per-member quota ────────
# recomputed per-member counts (from judge_raw_neg + raw_neg.jsonl, threshold 50)
neg = {
    "neg_default_assistant": {"judged": 35, "kept": 13, "none": 5, "complied": 17},
    "neg_reph_curious": {"judged": 35, "kept": 21, "none": 4, "complied": 10},
    "neg_sp_ph4": {"judged": 35, "kept": 19, "none": 6, "complied": 10},
    "neg_sp_police": {"judged": 35, "kept": 24, "none": 4, "complied": 7},
    "neg_wc_short": {"judged": 35, "kept": 23, "none": 3, "complied": 9},
}
QUOTA = 24
labels = {
    "neg_default_assistant": "default\nassistant",
    "neg_reph_curious": "curious\nrephrase",
    "neg_sp_ph4": "second fixed\npersona",
    "neg_sp_police": "police\nofficer",
    "neg_wc_short": "wildchat\nshort",
}
set_paper_style("blog")
fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
members = list(neg)
xs = np.arange(len(members))
kept = np.array([neg[m]["kept"] for m in members])
complied = np.array([neg[m]["complied"] for m in members])
none = np.array([neg[m]["none"] for m in members])
ax.bar(xs, kept, color=C[1], label="kept (judged non-compliant)", width=0.6)
ax.bar(xs, complied, bottom=kept, color=C[3], label="dropped: judged harmful-compliant", width=0.6)
ax.bar(
    xs,
    none,
    bottom=kept + complied,
    color="0.75",
    label="dropped: judge returned no score",
    width=0.6,
)
ax.axhline(QUOTA, color="0.15", ls="--", lw=1.6)
ax.text(
    len(members) - 0.45, QUOTA + 0.4, "per-member pairing quota (24)", ha="right", fontsize=10.5
)
for i, _m in enumerate(members):
    ax.text(xs[i], kept[i] - 1.6, str(kept[i]), ha="center", va="top", fontsize=11, color="white")
ax.set_xticks(xs)
ax.set_xticklabels([labels[m] for m in members], fontsize=10.5)
ax.set_ylabel("negative completions (of 35 generated per member)")
ax.set_title(
    "Abliterated-generator contrastive negatives, harmful compliance",
    fontsize=13,
    pad=10,
)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
paths = savefig_paper(fig, "ablit_negative_panel_quota", dir=FIGDIR)
print("saved", paths["png"])
plt.close(fig)

# ── Figure 4: sycophancy structure — per-question heatmap + per-variant kept ─
set_paper_style("blog")
fig, (axh, axv) = plt.subplots(
    1, 2, figsize=(12.5, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.7, 1]}
)
qb = ys["sycophancy-base"]["per_question_yield"]
qa = ys["sycophancy-ablit"]["per_question_yield"]
qs = sorted(set(qb) & set(qa))
mat = np.array(
    [
        [qb[q]["kept"] / qb[q]["judged"] for q in qs],
        [qa[q]["kept"] / qa[q]["judged"] for q in qs],
    ]
)
im = axh.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
axh.set_yticks([0, 1])
axh.set_yticklabels(["base Qwen", "abliterated Qwen"], fontsize=11)
axh.set_xticks(range(len(qs)))
axh.set_xticklabels([qgloss(q) for q in qs], rotation=90, fontsize=8)
axh.set_xlabel("wrong-fact claim")
axh.set_title("Per-question kept fraction, sycophancy", fontsize=13, pad=10)
for (r, cix), v in np.ndenumerate(mat):
    j = (qb if r == 0 else qa)[qs[cix]]["judged"]
    axh.text(
        cix,
        r,
        f"{v:.0%}\nn={j}",
        ha="center",
        va="center",
        fontsize=6.5,
        color="white" if v > 0.55 else "0.2",
    )
fig.colorbar(im, ax=axh, shrink=0.75, label="kept fraction")

vk_b = dig["sycophancy-base"]["per_variant_kept"]
vk_a = dig["sycophancy-ablit"]["per_variant_kept"]
vj = dig["sycophancy-base"]["per_variant_judged"]
variants = sorted(vj)
xs = np.arange(len(variants))
w = 0.38
rb = [vk_b.get(v, 0) / vj[v] for v in variants]
ra = [vk_a.get(v, 0) / vj[v] for v in variants]
axv.bar(xs - w / 2, rb, width=w, color=COL["base"], label="base Qwen")
axv.bar(xs + w / 2, ra, width=w, color=COL["ablit"], label="abliterated Qwen")
for i, v in enumerate(variants):
    axv.text(xs[i] - w / 2, rb[i] + 0.015, f"{vk_b.get(v, 0)}/{vj[v]}", ha="center", fontsize=8.5)
    axv.text(xs[i] + w / 2, ra[i] + 0.015, f"{vk_a.get(v, 0)}/{vj[v]}", ha="center", fontsize=8.5)
axv.axhline(20 / 36, color="0.15", ls="--", lw=1.4)
axv.text(len(variants) - 0.4, 20 / 36 + 0.015, "floor rate", ha="right", fontsize=10)
axv.set_xticks(xs)
axv.set_xticklabels([VARIANT_LABEL.get(v, v) for v in variants], fontsize=9, rotation=30)
axv.set_xlabel("elicitation-instruction phrasing (exhibit set)")
axv.set_ylabel("kept fraction")
axv.set_ylim(0, 0.85)
axv.set_title("Per-variant kept fraction, sycophancy", fontsize=13, pad=10)
axv.legend(fontsize=10)
paths = savefig_paper(fig, "sycophancy_question_variant_structure", dir=FIGDIR)
print("saved", paths["png"])
plt.close(fig)

# ── Figure 5: drop-composition per positive arm (drop-never-coerce audit) ────
set_paper_style("blog")
fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
cells = [
    ("sycophancy-base", "sycophancy\nbase"),
    ("sycophancy-ablit", "sycophancy\nabliterated"),
    ("harmful_compliance-base", "harmful\nbase"),
    ("harmful_compliance-ablit", "harmful\nabliterated"),
]
xs = np.arange(len(cells))
kept = np.array([dig[c]["kept_recomputed"] for c, _ in cells])
thr = np.array([dig[c]["threshold_drops"] for c, _ in cells])
none = np.array([dig[c]["judge_none_drops"] for c, _ in cells])
tot = np.array([dig[c]["n_rows"] for c, _ in cells])
ax.bar(xs, kept / tot, color=C[1], label="kept (mean judge score above 50)", width=0.6)
ax.bar(
    xs, thr / tot, bottom=kept / tot, color=C[3], label="dropped: below judge threshold", width=0.6
)
ax.bar(
    xs,
    none / tot,
    bottom=(kept + thr) / tot,
    color="0.75",
    label="dropped: no valid judge draw\n(whole row; all draws unusable)",
    width=0.6,
)
for i in range(len(cells)):
    ax.text(xs[i], 1.02, f"kept {kept[i]}/{tot[i]}", ha="center", fontsize=10.5)
ax.set_xticks(xs)
ax.set_xticklabels([lab for _, lab in cells], fontsize=11)
ax.set_ylabel("fraction of generated positive completions")
ax.set_ylim(0, 1.1)
ax.set_title("Positive-arm judge outcome composition (all four cells)", fontsize=13, pad=10)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
paths = savefig_paper(fig, "drop_composition_positive_arms", dir=FIGDIR)
print("saved", paths["png"])
plt.close(fig)

print("all figures written to", FIGDIR)
