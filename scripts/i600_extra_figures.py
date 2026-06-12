"""Analyzer supplementary figures for issue #600 (run-noise, source-implant shift, dose check).

Reads the committed sweep trajectories + panel_selection.json; writes three
figures to figures/issue_600/. CPU-only, deterministic.
"""

import itertools
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
FRACS = [0.08, 0.16, 0.33, 0.50, 0.75, 1.00]
STEP_OF = {0.08: 6, 0.16: 11, 0.33: 21, 0.50: 32, 0.75: 48, 1.00: 63}

manifest = json.loads((EVAL / "panel_selection.json").read_text())
TARGETS = [t["name"] for t in manifest["targets"]]
TROW = {t["name"]: t for t in manifest["targets"]}
analysis = json.loads((EVAL / "analysis" / "analysis.json").read_text())
COMMON_PANEL = analysis["locality"]["common_panel"]


def load(slug: str, seed: int) -> dict:
    return json.loads((EVAL / "sweep" / slug / f"seed_{seed}" / "trajectory.json").read_text())


def ckpt(payload: dict, frac: float) -> dict:
    return next(c for c in payload["checkpoints"] if abs(c["frac"] - frac) < 1e-9)


def dv_norm(ck: dict, persona: str) -> float | None:
    rec = ck["held_out"].get(persona)
    if rec is None:
        return None
    dgs = [rec[q]["delta_g"] for q in rec]
    return float(np.mean(dgs)) / float(ck["source_self"]["delta_g_mean"])


SWEEP = {
    (f"c600_{t}_{cond}", s): load(f"c600_{t}_{cond}", s)
    for t in TARGETS
    for cond in ("near", "ctrl")
    for s in SEEDS
}

set_paper_style("blog")
accent = paper_palette_role("accent")
neutral = paper_palette_role("neutral")
primary = paper_palette_role("primary")

# ── Figure 1: run-noise (same-mix seed gaps) by checkpoint + paired effects. ──
fig, ax = plt.subplots(figsize=(6.5, 4.0))
rng = np.random.default_rng(0)
medians = []
for frac in FRACS:
    gaps = []
    for t in TARGETS:
        for cond in ("near", "ctrl"):
            vals = [dv_norm(ckpt(SWEEP[(f"c600_{t}_{cond}", s)], frac), t) for s in SEEDS]
            gaps.extend(abs(a - b) for a, b in itertools.combinations(vals, 2))
    x = STEP_OF[frac] + rng.uniform(-1.2, 1.2, size=len(gaps))
    ax.scatter(x, gaps, s=12, color=neutral, alpha=0.5)
    medians.append(float(np.median(gaps)))
ax.plot(
    [STEP_OF[f] for f in FRACS], medians, "-", color=primary, lw=1.6, label="median same-mix gap"
)
paired = [abs(analysis["per_pair"][t]["seed_mean_d_normalized"]) for t in TARGETS]
ax.scatter(
    [63] * len(paired),
    paired,
    s=70,
    marker="D",
    facecolors="none",
    edgecolors=accent,
    linewidths=1.6,
    label="|paired NEAR−CONTROL effect| (terminal)",
    zorder=5,
)
ax.set_yscale("log")
ax.set_xlabel("Optimizer step (checkpoint)")
ax.set_ylabel("|gap| in implant-normalized target shift")
ax.set_title("Re-running the same training mix moves the target as much as the manipulation does")
ax.legend(loc="upper right")
fig.tight_layout()
savefig_paper(fig, "run_noise_by_checkpoint", dir=FIGS)
plt.close(fig)

# ── Figure 2: source-implant shift (raw + paired), terminal checkpoint. ──────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
xpos = np.arange(len(TARGETS))
labels = [t.replace("_", " ") for t in TARGETS]
for i, t in enumerate(TARGETS):
    near_vals = [
        float(ckpt(SWEEP[(f"c600_{t}_near", s)], 1.0)["source_self"]["delta_g_mean"]) for s in SEEDS
    ]
    ctrl_vals = [
        float(ckpt(SWEEP[(f"c600_{t}_ctrl", s)], 1.0)["source_self"]["delta_g_mean"]) for s in SEEDS
    ]
    axes[0].scatter([i - 0.12] * 3, near_vals, s=24, color=accent, alpha=0.8)
    axes[0].scatter([i + 0.12] * 3, ctrl_vals, s=24, color=neutral, alpha=0.8)
    axes[0].plot(
        [i - 0.12, i + 0.12], [np.mean(near_vals), np.mean(ctrl_vals)], "-", color="black", lw=1.0
    )
    diffs = [n - c for n, c in zip(near_vals, ctrl_vals, strict=True)]
    axes[1].scatter([i] * 3, diffs, s=24, color=accent, alpha=0.8)
    axes[1].plot([i - 0.18, i + 0.18], [np.mean(diffs)] * 2, "-", color="black", lw=1.6)
# same-mix source-dG gap band (median over 36 within-condition seed-pair gaps)
src_gaps = []
for t in TARGETS:
    for cond in ("near", "ctrl"):
        vals = [
            float(ckpt(SWEEP[(f"c600_{t}_{cond}", s)], 1.0)["source_self"]["delta_g_mean"])
            for s in SEEDS
        ]
        src_gaps.extend(abs(a - b) for a, b in itertools.combinations(vals, 2))
band = float(np.median(src_gaps))
axes[1].axhspan(-band, band, color="gray", alpha=0.12)
axes[1].axhline(0, color="gray", lw=0.8)
axes[0].scatter([], [], s=24, color=accent, label="nearest-neighbor cell")
axes[0].scatter([], [], s=24, color=neutral, label="distance-matched control cell")
axes[0].legend(loc="upper left")
axes[0].set_ylabel("Source ΔlogP(marker), villain (nats)")
axes[0].set_title("Raw: source implant per cell (3 seeds each)")
axes[1].set_ylabel("NEAR − CONTROL source diff (nats)")
axes[1].set_title("Paired: per-seed difference (band = median same-mix gap)")
for ax in axes:
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
fig.tight_layout(pad=1.4)
savefig_paper(fig, "source_implant_shift", dir=FIGS)
plt.close(fig)

# ── Figure 3: trained-negative dose signature (raw + centered), terminal. ────
roles = [
    "default assistant",
    "bartender\n(panel negative)",
    "french person\n(panel negative)",
    "variable-slot\nnegative",
]
raw_vals: dict[str, list[float]] = {r: [] for r in roles}
centered_vals: dict[str, list[float]] = {r: [] for r in roles}
panel_medians = []
untrained_all = []
for t in TARGETS:
    for cond in ("near", "ctrl"):
        slot = TROW[t]["near"]["name"] if cond == "near" else TROW[t]["ctrl"]["name"]
        for s in SEEDS:
            ck = ckpt(SWEEP[(f"c600_{t}_{cond}", s)], 1.0)
            untrained = [dv_norm(ck, p) for p in COMMON_PANEL]
            untrained = [v for v in untrained if v is not None]
            med = float(np.median(untrained))
            panel_medians.append(med)
            untrained_all.extend(untrained)
            for role, persona in zip(
                roles, ["qwen_default", "bartender", "french_person", slot], strict=True
            ):
                v = dv_norm(ck, persona)
                if v is not None:
                    raw_vals[role].append(v)
                    centered_vals[role].append(v - med)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
rng = np.random.default_rng(1)
# raw panel: untrained distribution + the 4 trained roles
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
axes[0].set_xticklabels(["41 untrained\npanel personas"] + roles, fontsize=8)
axes[0].set_ylabel("Normalized shift (ΔlogP ÷ source)")
axes[0].set_title("Raw: trained negatives vs untrained personas (36 cells)")
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
axes[1].set_title("Centered: only the default assistant sits below the median")
fig.tight_layout(pad=1.4)
savefig_paper(fig, "trained_negative_dose", dir=FIGS)
plt.close(fig)

print("medians by step:", dict(zip([STEP_OF[f] for f in FRACS], [round(m, 4) for m in medians])))
print("source-gap band:", round(band, 3))
for r in roles:
    print(
        r.replace("\n", " "),
        "centered mean %.4f sd %.4f n=%d"
        % (
            float(np.mean(centered_vals[r])),
            float(np.std(centered_vals[r])),
            len(centered_vals[r]),
        ),
    )
print("done")
