#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — analyzer body figures (CPU, off-pod, blog style).

Supplements scripts/i601_figures.py (the registered dump) with the figures the
clean-result body embeds per finding:

  1. hero_schedule_vs_ratio     — Phase-1 trajectories + terminals vs in-task refs
  2. hero_prediction_matrix_v3  — prediction matrix with IN-TASK (plan v3 §C) bands
  3. gradient_dead_negatives    — live-gauge row-type CE: positives fit, negatives dead
  4. dense_growth_window        — classic-gauge dense trajectories across cells
  5. cross_rig_top_compression  — committed vs re-read vs fresh (Observation O)
  6. phase4_bridge_live         — live-gauge bridge trajectories (knife-edge)

Usage:
    uv run python scripts/i601_body_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path("eval_results/issue_601")
FIGDIR = "issue_601"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _onpolicy(ph: str, cell: str, seed: int) -> list[tuple[int, float, float]]:
    """(step, delta_g, margin) per checkpoint from trajectory.json."""
    t = _load(ROOT / ph / f"{cell}_seed{seed}" / "trajectory.json")
    out = []
    for ck in t["checkpoints"]:
        ss = ck["source_self"]
        marg = ss["delta_z_marker_mean"] - (ss["z_eos_g_mean"] - ss["z_eos_b_mean"])
        out.append((ck["step"], ss["delta_g_mean"], marg))
    return out


def _dense(ph: str, cell: str, seed: int) -> tuple[list[int], list[float]]:
    d = _load(ROOT / ph / f"{cell}_seed{seed}" / "dense_trajectory.json")
    steps = [r["step"] for r in d["checkpoints"]]
    dg = [r["source_mean"]["delta_g"] for r in d["checkpoints"]]
    return steps, dg


def _rowtype(ph: str, cell: str, seed: int) -> dict:
    return _load(ROOT / ph / f"{cell}_seed{seed}" / "rowtype_ce.json")


CLS = _load(ROOT / "analysis" / "classification.json")
L = CLS["in_task_references"]["l_refs"]
M = CLS["in_task_references"]["m_refs"]
TOL = CLS["in_task_references"]["tol_logp"]
TOL_M = CLS["in_task_references"]["tol_margin"]
MID = CLS["phase1"]["upper_midpoint"]

set_paper_style("blog")

C_QUARTER = paper_palette_role("primary")
C_MATCHED = paper_palette_role("accent")
C_DOUBLE = paper_palette_role("control")
C_ANCHOR = paper_palette_role("baseline")
C_NEUTRAL = paper_palette_role("neutral")

# ── 1. HERO: schedule vs ratio ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1.1]})
ax = axes[0]
series = [
    ("ratio4to1_100p400n", "phase1", "Quarter mix, natural schedule (T=32)", C_QUARTER),
    ("ratio4to1_100p400n_T128", "phase1", "Same quarter mix, 4 epochs (T=128)", C_MATCHED),
    ("ratio4to1_400p1600n", "phase1", "Double-size mix (T=125)", C_DOUBLE),
    ("dense_200p800n", "phase2", "Fresh anchor mix (T=63)", C_ANCHOR),
]
for cell, ph, label, color in series:
    for i, seed in enumerate((42, 137)):
        pts = _onpolicy(ph, cell, seed)
        steps = [p[0] for p in pts]
        dg = [p[1] for p in pts]
        ax.plot(
            steps,
            dg,
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.6,
            alpha=0.9 if i == 0 else 0.55,
            label=label if i == 0 else None,
        )
ax.set_xlabel("optimizer step")
ax.set_ylabel("source marker log-prob gain, trained − base (nats)")
ax.legend(loc="lower right", fontsize=8.5)
set_title_subtitle(
    ax,
    "Same data, same 4:1 ratio — a 4x longer schedule doubles the implant",
    "On-policy reads; paired lines per arm = seeds 42 (solid) / 137 (faded); n = 10 eval questions per point",
)

ax = axes[1]
arm_x = {"quarter": 0, "anchor": 1, "double": 2, "matched": 3}
ARM_TERMS = {
    "quarter": [6.782, 7.560],
    "anchor": [13.929, 11.697],
    "double": [14.821, 14.345],
    "matched": [16.968, 14.176],
}
for name, lv in L.items():
    ax.axhline(lv, color=C_NEUTRAL, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.annotate(
        f"fresh {name} level",
        xy=(3.45, lv),
        fontsize=7.5,
        color="#666666",
        va="center",
        ha="left",
        annotation_clip=False,
    )
for arm, vals in ARM_TERMS.items():
    color = {"quarter": C_QUARTER, "matched": C_MATCHED, "double": C_DOUBLE, "anchor": C_ANCHOR}[
        arm
    ]
    ax.scatter([arm_x[arm]] * 2, vals, s=46, color=color, zorder=5)
ax.set_xticks(
    list(arm_x.values()),
    [
        "Quarter\nmix\n(T=32)",
        "Anchor\nmix\n(T=63)",
        "Double\nmix\n(T=125)",
        "Quarter mix\n× 4 epochs\n(T=128)",
    ],
    fontsize=8,
)
ax.set_ylabel("terminal gain (nats)")
ax.set_xlim(-0.5, 3.75)
set_title_subtitle(
    ax,
    "Terminal levels order by schedule length, not ratio",
    "Points = seeds 42/137; dashes = this task's fresh dose-response levels",
)
savefig_paper(fig, f"{FIGDIR}/hero_schedule_vs_ratio", dir="figures/")
plt.close(fig)

# ── 2. Prediction matrix with in-task (v3 §C) bands ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
CEIL = 22.6
BANDS_LOGP = {
    "quarter": {
        "equilibrium": (L["4:1"] - TOL, L["4:1"] + TOL),
        "horizon": (0.0, L["4:1"] - 3.0),
        "coupling": (L["2:1"] - 2.5, L["2:1"] + 2.5),
    },
    "anchor": {
        "equilibrium": (L["4:1"] - TOL, L["4:1"] + TOL),
        "horizon": (L["4:1"] - TOL, L["4:1"] + TOL),
        "coupling": (L["4:1"] - TOL, L["4:1"] + TOL),
    },
    "double": {
        "equilibrium": (L["4:1"] - TOL, L["4:1"] + TOL),
        "horizon": (MID, CEIL),
        "coupling": (L["8:1"] - TOL, L["8:1"] + TOL),
    },
    "matched": {
        "equilibrium": (L["4:1"] - TOL, L["4:1"] + TOL),
        "horizon": (MID, CEIL),
        "coupling": (L["8:1"] - TOL, L["8:1"] + TOL),
    },
}
M_CEIL = 16.0
BANDS_MARGIN = {
    "quarter": {
        "equilibrium": (M["4:1"] - TOL_M, M["4:1"] + TOL_M),
        "horizon": (0.0, M["4:1"] - 2.0),
        "coupling": (M["2:1"] - 1.5, M["2:1"] + 1.5),
    },
    "anchor": {
        "equilibrium": (M["4:1"] - TOL_M, M["4:1"] + TOL_M),
        "horizon": (M["4:1"] - TOL_M, M["4:1"] + TOL_M),
        "coupling": (M["4:1"] - TOL_M, M["4:1"] + TOL_M),
    },
    "double": {
        "equilibrium": (M["4:1"] - TOL_M, M["4:1"] + TOL_M),
        "horizon": ((M["4:1"] + M["8:1"]) / 2, M_CEIL),
        "coupling": (M["8:1"] - TOL_M, M["8:1"] + TOL_M),
    },
    "matched": {
        "equilibrium": (M["4:1"] - TOL_M, M["4:1"] + TOL_M),
        "horizon": ((M["4:1"] + M["8:1"]) / 2, M_CEIL),
        "coupling": (M["8:1"] - TOL_M, M["8:1"] + TOL_M),
    },
}
ARM_TERMS_M = {
    "quarter": [4.655, 5.220],
    "anchor": [9.584, 8.029],
    "double": [9.397, 9.966],
    "matched": [10.831, 9.738],
}
HYP_COLORS = {
    "equilibrium": paper_palette_role("primary"),
    "horizon": paper_palette_role("accent"),
    "coupling": paper_palette_role("control"),
}
ARM_ORDER = ["quarter", "anchor", "double", "matched"]
ARM_LBL = [
    "Quarter mix\n(T=32)",
    "Anchor mix\n(T=63)",
    "Double mix\n(T=125)",
    "Same quarter mix,\n4 epochs (T=128)",
]
for ax, bands, terms, ylabel, title in (
    (
        axes[0],
        BANDS_LOGP,
        ARM_TERMS,
        "terminal log-prob gain (nats)",
        "Log-prob space: bands overlap — formal no-call",
    ),
    (
        axes[1],
        BANDS_MARGIN,
        ARM_TERMS_M,
        "terminal EOS-margin gain (logits)",
        "Margin co-read: same picture",
    ),
):
    for h_i, hyp in enumerate(["equilibrium", "horizon", "coupling"]):
        for a_i, arm in enumerate(ARM_ORDER):
            lo, hi = bands[arm][hyp]
            off = (h_i - 1) * 0.24
            ax.fill_between(
                [a_i + off - 0.10, a_i + off + 0.10],
                lo,
                hi,
                color=HYP_COLORS[hyp],
                alpha=0.40,
                label=f"{hyp} prediction" if a_i == 0 else None,
            )
    for a_i, arm in enumerate(ARM_ORDER):
        ax.scatter([a_i] * 2, terms[arm], s=42, color="#222222", zorder=5)
    ax.set_xticks(range(4), ARM_LBL, fontsize=8)
    ax.set_ylabel(ylabel)
    set_title_subtitle(ax, title, "black points = observed terminals, seeds 42/137")
axes[0].legend(loc="upper left", fontsize=8)
savefig_paper(fig, f"{FIGDIR}/hero_prediction_matrix_v3", dir="figures/")
plt.close(fig)

# ── 3. Gradient-dead negatives ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
CELLS_LIVE = [
    ("phase2", "dense_200p0n", 137, "Positives only (T=13)", C_ANCHOR),
    ("phase2", "dense_200p400n", 137, "2:1 mix (T=38)", C_DOUBLE),
    ("phase2", "dense_200p800n", 137, "4:1 mix (T=63)", C_QUARTER),
    ("phase2", "dense_200p1600n", 137, "8:1 mix (T=113)", C_MATCHED),
]
ax = axes[0]
for ph, cell, seed, label, color in CELLS_LIVE:
    d = _rowtype(ph, cell, seed)
    steps = [r["step"] for r in d["records"]]
    ce = [r["pos_marker_ce"] for r in d["records"]]
    ax.plot(
        steps[:20], ce[:20], color=color, linewidth=1.8, marker="o", markersize=3.5, label=label
    )
ax.set_xlabel("optimizer step")
ax.set_ylabel("positive-row marker cross-entropy (nats)")
ax.legend(fontsize=8.5)
set_title_subtitle(
    ax,
    "Positive rows: fully fit within ~13 steps in every cell",
    "Live training model (faithful rsLoRA scaling); 16 training rows; CE → 0 = P(marker) → 1",
)
ax = axes[1]
CELLS_NEG = [
    ("phase2", "dense_200p400n", 137, "2:1 mix", C_DOUBLE),
    ("phase2", "dense_200p800n", 137, "4:1 mix", C_QUARTER),
    ("phase2", "dense_200p1600n", 137, "8:1 mix", C_MATCHED),
    ("phase3", "negonly_0p800n", 42, "Negatives only (seed 42)", C_NEUTRAL),
    ("phase3", "negonly_0p800n", 137, "Negatives only (seed 137)", C_ANCHOR),
]
for ph, cell, seed, label, color in CELLS_NEG:
    d = _rowtype(ph, cell, seed)
    steps = [r["step"] for r in d["records"]]
    ce = [max(r["neg_trailing_ce"], 1e-8) for r in d["records"]]
    ax.semilogy(steps, ce, color=color, linewidth=1.8, label=label)
ax.set_xlabel("optimizer step")
ax.set_ylabel("negative-row loss-token cross-entropy (nats, log scale)")
ax.set_ylim(1e-8, 30)
ax.legend(fontsize=8.5, loc="upper right")
set_title_subtitle(
    ax,
    "Negative rows: no loss to learn from, ever",
    "The single loss token is already predicted at CE ~ 1e-6 before training starts",
)
savefig_paper(fig, f"{FIGDIR}/gradient_dead_negatives", dir="figures/")
plt.close(fig)

# ── 4. Dense growth window (classic read gauge) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8.4, 4.8))
DENSE_CELLS = [
    ("phase2", "dense_200p0n", 137, "Positives only (T=13)", C_ANCHOR),
    ("phase2", "dense_200p400n", 137, "2:1 mix (T=38)", C_DOUBLE),
    ("phase2", "dense_200p800n", 137, "4:1 mix (T=63)", C_QUARTER),
    ("phase2", "dense_200p1600n", 137, "8:1 mix (T=113)", C_MATCHED),
    ("phase1", "ratio4to1_100p400n_T128", 137, "Quarter mix, 4 epochs (T=128)", "#9467bd"),
]
for ph, cell, seed, label, color in DENSE_CELLS:
    steps, dg = _dense(ph, cell, seed)
    ax.plot(steps, dg, color=color, linewidth=1.8, label=label)
    ax.scatter([steps[-1]], [dg[-1]], color=color, s=40, zorder=5)
ax.set_xlabel("optimizer step")
ax.set_ylabel("source marker log-prob gain, trained − base (nats)")
ax.legend(fontsize=8.5, loc="lower right")
set_title_subtitle(
    ax,
    "Growth is a smooth ~20–32-step window, not an instant arrest",
    "Teacher-forced dense reads at the staged classic gauge, seed 137; dot = end of schedule",
)
savefig_paper(fig, f"{FIGDIR}/dense_growth_window", dir="figures/")
plt.close(fig)

# ── 5. Cross-rig top compression (Observation O) ─────────────────────────────
gate = _load(ROOT / "phase0" / "phase0_gate.json")
per = gate["onpolicy_crosscheck"]["per_adapter"]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
LEVELS = [("noneg", "0:1"), ("negex_100", "2:1"), ("anchor", "4:1"), ("negex_400", "8:1")]
FRESH = {"0:1": [2.68], "2:1": [10.37], "4:1": [13.93, 11.70], "8:1": [13.98]}
for i, (slug, lvl) in enumerate(LEVELS):
    for j, seed in enumerate((42, 137)):
        k = f"c472_{slug}_seed{seed}"
        committed = per[k]["committed_delta_g"]
        reread = per[k]["reread_delta_g"]
        x = i + (j - 0.5) * 0.18
        ax.plot([x, x], [committed, reread], color=C_NEUTRAL, linewidth=1.2, zorder=2)
        ax.scatter(
            [x],
            [committed],
            color=C_DOUBLE,
            s=44,
            zorder=4,
            label="parent committed" if i == 0 and j == 0 else None,
        )
        ax.scatter(
            [x],
            [reread],
            color=C_QUARTER,
            s=44,
            zorder=4,
            label="re-read under this rig" if i == 0 and j == 0 else None,
        )
    for v in FRESH[lvl]:
        ax.scatter(
            [i + 0.32],
            [v],
            color=C_MATCHED,
            marker="D",
            s=40,
            zorder=4,
            label="freshly retrained" if i == 0 else None,
        )
ax.set_xticks(range(4), ["Positives only\n(0:1)", "2:1 mix", "4:1 mix", "8:1 mix"], fontsize=9)
ax.set_ylabel("terminal source gain (nats)")
ax.legend(fontsize=8.5, loc="upper left")
set_title_subtitle(
    ax,
    "The parent dose-response top does not reproduce: 8:1 collapses onto 4:1",
    "On-policy endpoint re-reads of the committed #472 adapters vs fresh same-recipe retrains",
)
savefig_paper(fig, f"{FIGDIR}/cross_rig_top_compression", dir="figures/")
plt.close(fig)

# ── 6. Phase-4 bridge, live gauge ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.4, 4.6))
BRIDGE = [
    ("phase2", "dense_200p0n", [137], "Positives only at parent LR 1e-5", C_ANCHOR),
    (
        "phase4",
        "posonly_alllinear_lr5e6",
        [42, 137],
        "Positives only at half LR 5e-6 (the #471 bridge)",
        C_MATCHED,
    ),
    ("phase4", "posonly_attn_lr5e6", [42, 137], "Half LR + attention-only adapter", C_DOUBLE),
]
for ph, cell, seeds, label, color in BRIDGE:
    for i, seed in enumerate(seeds):
        d = _rowtype(ph, cell, seed)
        steps = [r["step"] for r in d["records"]]
        pb = d["pos_marker_ce_base"]
        dg = [pb - r["pos_marker_ce"] for r in d["records"]]
        ax.plot(
            steps,
            dg,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            alpha=0.9 if i == 0 else 0.55,
            label=label if i == 0 else None,
        )
ax.set_xlabel("optimizer step")
ax.set_ylabel("source marker log-prob gain, live gauge (nats)")
ax.legend(fontsize=8.5, loc="upper left")
set_title_subtitle(
    ax,
    "The bridge arm grows fast but T=13 ends before the call is decidable",
    "Live-gauge 16-row probe; halving LR slows the fit; classification: ambiguous (knife-edge)",
)
savefig_paper(fig, f"{FIGDIR}/phase4_bridge_live", dir="figures/")
plt.close(fig)

print("body figures written to figures/issue_601/")
