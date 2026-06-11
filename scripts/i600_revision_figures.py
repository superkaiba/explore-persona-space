"""Round-2 revision figures for issue #600 (interpretation-critique fixes).

Regenerates three reader-facing figures over the COMMITTED registered outputs
(analysis.json, locality_detail.json, panel_selection.json, sweep
trajectories) — no statistic is recomputed differently from analyze.py /
i600_extra_figures.py; only presentation changes plus one added descriptive
panel:

1. hero_paired_dumbbell — plain-English persona labels in panel titles
   (was: underscore slugs).
2. bubble_radius_L10_raw — plain-English persona labels in the legend.
3. trained_negative_dose — right-panel title corrected ("only the default
   assistant sits FAR below the median") + a third panel: the within-persona
   trained-vs-untrained control over the four dual-role slot personas
   (readable both as trained negatives and as untrained panel members
   across cells).

CPU-only, deterministic. Writes to figures/issue_600/ (same filenames).
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results" / "issue_600"
FIGS = ROOT / "figures" / "issue_600"
SEEDS = [42, 137, 219]

manifest = json.loads((EVAL / "panel_selection.json").read_text())
TARGETS = [t["name"] for t in manifest["targets"]]
TROW = {t["name"]: t for t in manifest["targets"]}
analysis = json.loads((EVAL / "analysis" / "analysis.json").read_text())
detail = json.loads((EVAL / "analysis" / "locality_detail.json").read_text())
COMMON_PANEL = analysis["locality"]["common_panel"]
PER_PAIR = analysis["per_pair"]


def label(slug: str) -> str:
    """Reader-facing persona label for a config slug."""
    return {"qwen_default": "default assistant"}.get(slug, slug.replace("_", " "))


def load(slug: str, seed: int) -> dict:
    return json.loads((EVAL / "sweep" / slug / f"seed_{seed}" / "trajectory.json").read_text())


def term(payload: dict) -> dict:
    return next(c for c in payload["checkpoints"] if abs(c["frac"] - 1.0) < 1e-9)


def dv_norm(ck: dict, persona: str) -> float | None:
    rec = ck["held_out"].get(persona)
    if rec is None:
        return None
    return float(np.mean([rec[q]["delta_g"] for q in rec])) / float(
        ck["source_self"]["delta_g_mean"]
    )


set_paper_style("blog")
accent = paper_palette_role("accent")
neutral = paper_palette_role("neutral")
primary = paper_palette_role("primary")

# ── Figure 1 (hero): paired dumbbell, one panel per stratum, reader labels. ──
strata = ("near", "mid", "far")
stratum_of = {t: PER_PAIR[t]["stratum"] for t in PER_PAIR}
fallback = set(analysis["fallback_pairs"])
per_pair_bands = analysis["run_noise"]["per_pair_bands"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
for ax, stratum in zip(axes, strata, strict=True):
    ts = [t for t in PER_PAIR if stratum_of[t] == stratum]
    for t in ts:
        connector = "--o" if t in fallback else "-o"
        for row in PER_PAIR[t]["per_seed"]:
            near_v = row["near"]["normalized"]
            ctrl_v = row["ctrl"]["normalized"]
            color = accent if near_v < ctrl_v else neutral
            ax.plot([0, 1], [near_v, ctrl_v], connector, color=color, alpha=0.65, ms=4, lw=1.2)
        band_meds = [
            v for v in per_pair_bands[t]["median_same_mix_gap_by_frac"].values() if v is not None
        ]
        if band_meds:
            band = max(band_meds)
            mids = [
                0.5 * (row["near"]["normalized"] + row["ctrl"]["normalized"])
                for row in PER_PAIR[t]["per_seed"]
            ]
            center = float(np.mean(mids))
            ax.axhspan(center - band / 2, center + band / 2, alpha=0.10, color="gray")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nearest-neighbor\nnegative", "Distance-matched\nfar control"])
    labels = [label(t) + (" *" if t in fallback else "") for t in ts]
    ax.set_title(f"{stratum}-villain targets: {', '.join(labels)}", fontsize=9)
axes[0].set_ylabel("Implant-normalized target shift\n(target ΔlogP ÷ source ΔlogP)")
headline = analysis["headline_permutation"]
suptitle = (
    f"Paired NEAR vs CONTROL target leakage — permutation p = {headline['p_one_sided']:.3f} "
    f"(T = {headline['t_obs']:+.4f}); shaded band = per-pair median same-mix seed gap"
)
if fallback:
    suptitle += "; * / dashed = band-entry fallback read (unmatched step)"
fig.suptitle(suptitle, fontsize=10)
fig.tight_layout()
savefig_paper(fig, "hero_paired_dumbbell", dir=FIGS)
plt.close(fig)

# ── Figure 4: bubble-radius scatter, reader labels in the legend. ───────────
l10 = detail["bubble_radius"]["L10"]
assert not l10.get("skipped")
fig, ax = plt.subplots(figsize=(6, 4))
for t, d in l10["per_target"].items():
    xs = [q["d_to_near_negative"] for q in d["points"]]
    ys = [q["paired_difference"] for q in d["points"]]
    ax.scatter(xs, ys, s=14, alpha=0.55, label=f"{label(t)} pair")
    tgt_pt = next((q for q in d["points"] if q["persona"] == t), None)
    if tgt_pt:
        ax.scatter(
            [tgt_pt["d_to_near_negative"]],
            [tgt_pt["paired_difference"]],
            s=90,
            facecolors="none",
            edgecolors=accent,
            linewidths=1.6,
        )
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Centered L10 distance to the pair's nearest-neighbor negative")
ax.set_ylabel("Paired difference (normalized)")
ax.set_title("Bubble-radius read: suppression vs distance to the added negative")
ax.legend(fontsize=7)
fig.tight_layout()
savefig_paper(fig, "bubble_radius_L10_raw", dir=FIGS)
plt.close(fig)

# ── Figure 5: trained-negative dose (raw + centered + within-persona). ──────
slot_of = {}
for t in TARGETS:
    slot_of[f"c600_{t}_near"] = TROW[t]["near"]["name"]
    slot_of[f"c600_{t}_ctrl"] = TROW[t]["ctrl"]["name"]

roles = [
    "default assistant",
    "bartender\n(panel negative)",
    "french person\n(panel negative)",
    "variable-slot\nnegative",
]
raw_vals: dict[str, list[float]] = {r: [] for r in roles}
centered_vals: dict[str, list[float]] = {r: [] for r in roles}
untrained_all: list[float] = []
all_slots = sorted(set(slot_of.values()))
slot_trained: dict[str, list[float]] = {p: [] for p in all_slots}
slot_untrained: dict[str, list[float]] = {p: [] for p in all_slots}
for slug, slot in slot_of.items():
    for s in SEEDS:
        ck = term(load(slug, s))
        untrained = [v for v in (dv_norm(ck, p) for p in COMMON_PANEL) if v is not None]
        med = float(np.median(untrained))
        untrained_all.extend(untrained)
        for role, persona in zip(
            roles, ["qwen_default", "bartender", "french_person", slot], strict=True
        ):
            v = dv_norm(ck, persona)
            if v is not None:
                raw_vals[role].append(v)
                centered_vals[role].append(v - med)
        for p in all_slots:
            v = dv_norm(ck, p)
            if v is None:
                continue
            (slot_trained if p == slot else slot_untrained)[p].append(v - med)

dual = [p for p in all_slots if slot_trained[p] and slot_untrained[p]]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
rng = np.random.default_rng(1)
x0 = rng.uniform(-0.22, 0.22, size=len(untrained_all))
axes[0].scatter(x0, untrained_all, s=6, color=neutral, alpha=0.25)
axes[0].plot([-0.3, 0.3], [float(np.median(untrained_all))] * 2, "-", color="black", lw=1.6)
for j, r in enumerate(roles, start=1):
    xs = j + rng.uniform(-0.18, 0.18, size=len(raw_vals[r]))
    axes[0].scatter(xs, raw_vals[r], s=14, color=accent, alpha=0.55)
    axes[0].plot(
        [j - 0.25, j + 0.25], [float(np.mean(raw_vals[r]))] * 2, "-", color="black", lw=1.6
    )
axes[0].set_xticks(range(len(roles) + 1))
axes[0].set_xticklabels(["41 untrained\npanel personas", *roles], fontsize=8)
axes[0].set_ylabel("Normalized shift (ΔlogP ÷ source)")
axes[0].set_title("Raw: trained negatives vs untrained personas (36 cells)", fontsize=9)
for j, r in enumerate(roles):
    xs = j + rng.uniform(-0.18, 0.18, size=len(centered_vals[r]))
    axes[1].scatter(xs, centered_vals[r], s=14, color=accent, alpha=0.55)
    axes[1].plot(
        [j - 0.25, j + 0.25], [float(np.mean(centered_vals[r]))] * 2, "-", color="black", lw=1.6
    )
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_xticks(range(len(roles)))
axes[1].set_xticklabels(roles, fontsize=8)
axes[1].set_ylabel("Shift relative to untrained-panel median")
axes[1].set_title("Centered: only the default assistant sits far below the median", fontsize=9)
# Within-persona control: the four dual-role slot personas, trained vs untrained.
for j, p in enumerate(dual):
    xt = j - 0.16 + rng.uniform(-0.06, 0.06, size=len(slot_trained[p]))
    xu = j + 0.16 + rng.uniform(-0.06, 0.06, size=len(slot_untrained[p]))
    axes[2].scatter(xt, slot_trained[p], s=18, color=accent, alpha=0.7)
    axes[2].scatter(xu, slot_untrained[p], s=10, color=neutral, alpha=0.5)
    tm, um = float(np.mean(slot_trained[p])), float(np.mean(slot_untrained[p]))
    axes[2].plot([j - 0.28, j - 0.04], [tm] * 2, "-", color="black", lw=1.6)
    axes[2].plot([j + 0.04, j + 0.28], [um] * 2, "-", color="black", lw=1.6)
axes[2].axhline(0, color="gray", lw=0.8)
axes[2].set_xticks(range(len(dual)))
axes[2].set_xticklabels([label(p) for p in dual], fontsize=8)
axes[2].scatter([], [], s=18, color=accent, label="cells where trained as the slot negative")
axes[2].scatter([], [], s=10, color=neutral, label="cells where untrained")
axes[2].legend(fontsize=7, loc="upper left")
axes[2].set_ylabel("Shift relative to untrained-panel median")
axes[2].set_title("Within-persona: training as a negative adds no dip", fontsize=9)
fig.tight_layout(pad=1.4)
savefig_paper(fig, "trained_negative_dose", dir=FIGS)
plt.close(fig)

for p in dual:
    print(
        f"{p}: trained {np.mean(slot_trained[p]):+.4f} (n={len(slot_trained[p])}) "
        f"untrained {np.mean(slot_untrained[p]):+.4f} (n={len(slot_untrained[p])}) "
        f"diff {np.mean(slot_trained[p]) - np.mean(slot_untrained[p]):+.4f}"
    )
print(
    "dual-role mean diff:",
    float(np.mean([np.mean(slot_trained[p]) - np.mean(slot_untrained[p]) for p in dual])),
)
print("done")
