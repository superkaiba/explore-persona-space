#!/usr/bin/env python3
"""Issue #1336 — per-ADJACENT-STEP transfer at each tier of reparameterization.

Answers: for each stage of the Tulu-3 post-training ladder, how much of the
context->answer map transfers to the NEXT stage, under each tier of
reparameterization?

The four adjacent steps of `cm.MODELS` (stage 0..4):
    base -> sft      (0->1)   from the round-3 metric-ladder aggregate
    sft  -> dpo      (1->2)   from the round-3 metric-ladder aggregate
    dpo  -> rlvr     (2->3)   from the round-3 metric-ladder aggregate
    rlvr -> rlvr_long(3->4)   from round-5/B selfmap_v3 (7/7 cells, 2026-08-07)

TIER SET IS THE INTERSECTION, NOT A CHOICE. The aggregate carries all 9 tiers
(t0..t8) but selfmap_v3 ran only t0/t6/t7/t8, so those four are the only tiers on
which all FOUR adjacent steps are comparable. t1-t5 exist for the first three
steps only and are deliberately NOT plotted rather than shown as gaps on a line
that would read as measured-and-low.

BASIS / SCALE COMPATIBILITY (why these two sources may share an axis)
  - selfmap_v3 records carry r2_basis = "fold-local pooled OOF (plotted pair-file
    basis)", the same basis as the plotted round-3 pair files.
  - `scale="raw"` (raw pooled R²) is therefore the matched scale, and is also the
    plotter's own default (make_hero(scale="raw")). The "recal" rows are NOT used;
    mixing them with the selfmap cells would be a target mismatch.
  - Both sources are layer 30, fit_seed 0, 5 outer folds.

UNCERTAINTY IS SHOWN ONLY WHERE IT EXISTS
  - Round-3 rows carry a 1,000-draw paired-bootstrap CI on the GAP
    (t*_gap_lo/hi). Since gap = within_r2 - t*_r2 with within_r2 a per-row
    constant, that CI maps onto R² exactly as a location shift:
        r2_lo = within_r2 - gap_hi ,  r2_hi = within_r2 - gap_lo
    (verified numerically: t0_gap == within_r2 - t0_r2 to full precision).
  - selfmap_v3 cells carry NO bootstrap draws, so the rlvr->rlvr_long step is
    POINT-ONLY, drawn with open markers and no bar. It is never given a
    borrowed or assumed interval.
  - The aggregate panels show the MEAN over the 8 surfaces with every individual
    surface overplotted as a faint dot; no synthetic aggregate CI is invented.
    The real per-surface CIs appear at the grain they were computed, in the
    per-corpus small multiples.

REUSE: reads the committed round-3 aggregate through
`issue1336_metric_ladder_plots.get_row` (the shared accessor) rather than
re-deriving rows, and pulls tier label strings from that module's own
`tier_labels` block so the naming cannot drift from the committed figures.

Run from the issue-1336 worktree root:
    uv run python scripts/issue1336_step_transfer_tiers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib/numpy imports — on the shared
# VM load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and the BLAS
# pools freeze at import time.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue1336_metric_ladder_plots as mlp  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

REPO = repo_root()
OUTDIR = REPO / "figures" / "issue_1336"
SELFMAP_CELLS = REPO / "eval_results" / "issue_1336" / "selfmap_v3" / "cells"

SCALE = "raw"
LAYER = 30
TIERS = (0, 6, 7, 8)

# Adjacent (stage n -> stage n+1) steps, in ladder order.
STEPS = (
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "rlvr_long"),
)
# Which source each step's numbers come from.
SELFMAP_STEPS = {("rlvr", "rlvr_long")}

# Ordered conversational-first, then math/reasoning — the dominant structure in the
# result is a split BETWEEN these two families at the base->SFT step, so the panel
# order makes it visible instead of interleaving it away.
SURFACES = (
    ("chat", "lmsys23k"),
    ("chat", "uf11k"),
    ("chat", "if11k"),
    ("chat", "sft11k"),
    ("naturalistic", "lmsys23k"),
    ("chat", "gsm8k_train_full"),
    ("chat", "math7500"),
    ("chat", "gsm8k_test1319"),
)
CONVERSATIONAL = {
    ("chat", "lmsys23k"),
    ("chat", "uf11k"),
    ("chat", "if11k"),
    ("chat", "sft11k"),
    ("naturalistic", "lmsys23k"),
}

# n_train (~4/5 of n) < d = 4096 => every held-out R² here is estimator-degenerate,
# not a signal read. EXCLUDED from every aggregate statistic and MARKED in the
# per-corpus view; never silently dropped, never averaged in. The committed
# increments figure excludes this same cell for the same reason.
D_FEATURES = 4096
DEGENERATE = {("chat", "gsm8k_test1319")}

# Axis clipping for the aggregate panels. The math corpora reach t0 R² = -2.32 at
# base->SFT; letting the axis span that compresses the entire 0.0-0.6 region where
# every other reading lives. Off-scale points are drawn as carets AT the boundary
# (still represented, and their exact values ride in the meta JSON) rather than
# dropped or hidden.
A_YLIM = (-0.75, 0.72)
B_YLIM = (-0.05, 1.15)
# Per-corpus panels stay sharey (cross-panel comparability) but clip to the readable
# band; the math corpora's deep t0 dips (-2.32 / -1.90 / -0.73 at base->SFT) become
# floor carets instead of compressing all 8 panels into the top fifth.
PC_YLIM = (-0.40, 0.72)

# One color per tier, identical across every figure this script emits
# (a sequential ramp, because the tiers are NESTED in freedom: 8 > 7 > 6 > 0).
TIER_COLORS = {
    0: "#08306b",
    6: "#2171b5",
    7: "#6baed6",
    8: "#c6dbef",
}
CEILING_COLOR = "#525252"


def _tier_labels() -> dict[int, str]:
    """Pull the committed tier label strings out of the shared plotter.

    Read from the module source rather than hardcoded, so the labels cannot
    drift from the committed figures. Fails loud if the block moves.
    """
    src = (Path(mlp.__file__)).read_text()
    out: dict[int, str] = {}
    for line in src.splitlines():
        s = line.strip().strip(",").strip('"')
        if len(s) > 2 and s[0].isdigit() and s[1] == ":":
            idx = int(s[0])
            out.setdefault(idx, s.split(":", 1)[1].strip())
    missing = [t for t in TIERS if t not in out]
    if missing:
        raise RuntimeError(
            f"tier label block not found in {mlp.__file__} for tiers {missing}; "
            "the committed labels moved — fix this reader rather than hardcoding"
        )
    return out


TIER_LABEL = _tier_labels()


def step_label(src: str, tgt: str) -> str:
    short = {
        "base": "base",
        "sft": "SFT",
        "dpo": "DPO",
        "rlvr": "RLVR",
        "rlvr_long": "longer\nRLVR",
    }
    return f"{short[src]}→{short[tgt]}"


def load_selfmap() -> dict:
    """(pair, fmt, corpus, tier) -> {r2, within_r2, n}. No CI exists here."""
    if not SELFMAP_CELLS.is_dir():
        raise FileNotFoundError(
            f"selfmap cells not found at {SELFMAP_CELLS}. NOTE: the older "
            "`selfmap_missing_pairs/` path was renamed to `selfmap_v3/` — the "
            "committed issue1336_source_target_with_round5.py still points at the "
            "old name and cannot see these cells."
        )
    out = {}
    for fp in sorted(SELFMAP_CELLS.glob("*.json")):
        for rec in json.load(open(fp))["records"]:
            key = (rec["pair"], rec["format"], rec["corpus"], rec.get("tier"))
            out[key] = {
                "r2": float(rec["r2"]),
                "within_r2": float(rec["within_r2"]),
                "n": int(rec["n"]),
            }
    return out


SELFMAP = load_selfmap()


def cell(src: str, tgt: str, fmt: str, corpus: str, tier: int) -> dict | None:
    """Transfer read for one (step, surface, tier).

    Returns {r2, r2_lo, r2_hi, within_r2, n, has_ci} or None when not measured.
    """
    pair = f"{src}__{tgt}"
    if (src, tgt) in SELFMAP_STEPS:
        rec = SELFMAP.get((pair, fmt, corpus, tier))
        if rec is None:
            return None
        return {
            "r2": rec["r2"],
            "r2_lo": None,
            "r2_hi": None,
            "within_r2": rec["within_r2"],
            "n": rec["n"],
            "has_ci": False,
        }
    row = mlp.get_row(pair, fmt, corpus, SCALE)
    if row is None:
        return None
    r2 = row.get(f"t{tier}_r2")
    within = row.get("within_r2")
    gap_lo, gap_hi = row.get(f"t{tier}_gap_lo"), row.get(f"t{tier}_gap_hi")
    if r2 is None or within is None:
        return None
    # gap = within_r2 - r2, so the paired-bootstrap gap CI maps onto R² by a
    # location shift through the per-row constant `within_r2`.
    r2_lo = within - gap_hi if gap_hi is not None else None
    r2_hi = within - gap_lo if gap_lo is not None else None
    return {
        "r2": float(r2),
        "r2_lo": r2_lo,
        "r2_hi": r2_hi,
        "within_r2": float(within),
        "n": int(row.get("n", 0)),
        "has_ci": r2_lo is not None,
    }


def collect() -> dict:
    """(step_index, tier) -> per-surface list; plus per-step ceilings."""
    data, ceilings, coverage = {}, {}, {}
    for si, (src, tgt) in enumerate(STEPS):
        ceil_vals = []
        for tier in TIERS:
            vals = []
            for fmt, corpus in SURFACES:
                c = cell(src, tgt, fmt, corpus, tier)
                if c is None:
                    continue
                vals.append(((fmt, corpus), c))
                if (fmt, corpus) not in DEGENERATE:
                    ceil_vals.append(c["within_r2"])
            data[(si, tier)] = vals
            coverage[(si, tier)] = len(vals)
        # median, and degenerate surfaces excluded — same basis as the tier lines
        ceilings[si] = float(np.median(ceil_vals)) if ceil_vals else np.nan
    return {"data": data, "ceilings": ceilings, "coverage": coverage}


def _xticks(ax):
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([step_label(s, t) for s, t in STEPS], fontsize=9)
    ax.set_xlim(-0.35, len(STEPS) - 0.65)


def fig_aggregate(D: dict) -> Path:
    """Panel A: transfer R² by step and tier. Panel B: gap to the ceiling."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    xs = np.arange(len(STEPS))

    def _draw_points(ax, si, vals, ylim, color, marker_deg="x"):
        """Scatter each surface; off-scale points become carets AT the boundary."""
        lo, hi = ylim
        n = len(vals)
        for k, ((fmt, corpus), y) in enumerate(vals):
            x = si + (k - (n - 1) / 2) * 0.014
            deg = (fmt, corpus) in DEGENERATE
            m = marker_deg if deg else "o"
            if y < lo:
                ax.plot(
                    [x], [lo], marker="v", ms=5, color=color, alpha=0.75, zorder=2, clip_on=False
                )
            elif y > hi:
                ax.plot(
                    [x], [hi], marker="^", ms=5, color=color, alpha=0.75, zorder=2, clip_on=False
                )
            else:
                ax.plot(
                    [x],
                    [y],
                    marker=m,
                    ms=4.2 if not deg else 5.0,
                    color=color,
                    alpha=0.75 if deg else 0.35,
                    mew=1.2 if deg else 0,
                    ls="none",
                    zorder=2,
                )

    for tier in TIERS:
        mA, mB, xa = [], [], []
        for si in range(len(STEPS)):
            vals = D["data"][(si, tier)]
            if not vals:
                continue
            # Aggregate over NON-DEGENERATE surfaces only, and use the MEDIAN: at
            # base->SFT the math corpora reach t0 R² = -2.32 while the conversational
            # ones sit at +0.23, so a mean over that mix is not a summary of anything.
            keep = [(k, c) for k, c in vals if k not in DEGENERATE]
            r2s = np.array([c["r2"] for _, c in keep])
            gaps = np.array([c["within_r2"] - c["r2"] for _, c in keep])
            xa.append(si)
            mA.append(float(np.median(r2s)))
            mB.append(float(np.median(gaps)))
            _draw_points(axA, si, [(k, c["r2"]) for k, c in vals], A_YLIM, TIER_COLORS[tier])
            _draw_points(
                axB, si, [(k, c["within_r2"] - c["r2"]) for k, c in vals], B_YLIM, TIER_COLORS[tier]
            )
        style = dict(color=TIER_COLORS[tier], lw=2.0, zorder=3)
        axA.plot(xa, mA, marker="o", ms=6, label=f"{tier}: {TIER_LABEL[tier]}", **style)
        axB.plot(xa, mB, marker="o", ms=6, label=f"{tier}: {TIER_LABEL[tier]}", **style)

    ceil = [D["ceilings"][si] for si in range(len(STEPS))]
    axA.plot(
        xs,
        ceil,
        marker="s",
        ms=5,
        ls="--",
        lw=1.6,
        color=CEILING_COLOR,
        label="within-model ceiling",
        zorder=4,
    )
    axB.axhline(0.0, ls="--", lw=1.2, color=CEILING_COLOR, zorder=1)
    axB.axhspan(
        0.0,
        mlp.BAND,
        color=CEILING_COLOR,
        alpha=0.12,
        zorder=0,
        label=f"elicitation band ({mlp.BAND:.3f})",
    )

    axA.set_ylabel("held-out transfer R²  (raw pooled, layer 30)", fontsize=10)
    axA.set_title(
        "A. How much transfers to the next stage\n"
        "median over 7 non-degenerate corpora; every corpus shown as a dot",
        fontsize=10.5,
        loc="left",
    )
    axB.set_ylabel("gap to within-model ceiling  (R² points)", fontsize=10)
    axB.set_title(
        "B. How much is still missing\n"
        "gsm8k_test1319 excluded (n_train≈1034 < d=4096, estimator-degenerate)",
        fontsize=10.5,
        loc="left",
    )
    axA.set_ylim(*A_YLIM)
    axB.set_ylim(*B_YLIM)
    for ax in (axA, axB):
        _xticks(ax)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_xlabel("adjacent ladder step", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = OUTDIR / "ladder_step_transfer_by_tier.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_percorpus(D: dict) -> Path:
    """Per-unit view: one mini-panel per surface, with the REAL per-row CIs."""
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.6), sharex=True, sharey=True)
    for k, (fmt, corpus) in enumerate(SURFACES):
        ax = axes[k // 4, k % 4]
        for tier in TIERS:
            xa, ys, lo, hi = [], [], [], []
            for si, (src, tgt) in enumerate(STEPS):
                c = cell(src, tgt, fmt, corpus, tier)
                if c is None:
                    continue
                xa.append(si)
                ys.append(c["r2"])
                lo.append(c["r2"] - c["r2_lo"] if c["has_ci"] else 0.0)
                hi.append(c["r2_hi"] - c["r2"] if c["has_ci"] else 0.0)
            if not xa:
                continue
            ax.errorbar(
                xa,
                ys,
                yerr=[lo, hi],
                marker="o",
                ms=4.5,
                lw=1.6,
                capsize=2.0,
                color=TIER_COLORS[tier],
                label=f"{tier}: {TIER_LABEL[tier]}",
            )
            # off-scale points get a caret AT the boundary (same convention as panel A):
            # sharey keeps every panel comparable, clipping keeps them readable.
            for si, y in zip(xa, ys):
                if y < PC_YLIM[0]:
                    ax.plot(
                        [si],
                        [PC_YLIM[0]],
                        marker="v",
                        ms=6,
                        color=TIER_COLORS[tier],
                        zorder=6,
                        clip_on=False,
                    )
            # open marker on the point-only step so it cannot read as CI==0
            for si, y in zip(xa, ys):
                if STEPS[si] in SELFMAP_STEPS:
                    ax.plot(
                        [si],
                        [y],
                        marker="o",
                        ms=8,
                        mfc="none",
                        mec=TIER_COLORS[tier],
                        mew=1.4,
                        zorder=5,
                    )
        cvals = [cell(s, t, fmt, corpus, TIERS[0]) for s, t in STEPS]
        ax.plot(
            [i for i, c in enumerate(cvals) if c],
            [c["within_r2"] for c in cvals if c],
            marker="s",
            ms=4,
            ls="--",
            lw=1.3,
            color=CEILING_COLOR,
        )
        deg = (fmt, corpus) in DEGENERATE
        title = f"{corpus} ({fmt})" + ("  — DEGENERATE n_train<d" if deg else "")
        ax.set_title(title, fontsize=9.5, loc="left", color="#8c2d04" if deg else "black")
        if deg:
            ax.set_facecolor("#fdf2e9")
        ax.grid(axis="y", alpha=0.22, lw=0.6)
        _xticks(ax)
    axes[0, 0].set_ylim(*PC_YLIM)  # sharey => applies to all 8 panels
    axes[0, 0].set_ylabel("transfer R²", fontsize=10)
    axes[1, 0].set_ylabel("transfer R²", fontsize=10)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005)
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUTDIR / "ladder_step_transfer_by_tier_percorpus.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def fig_sufficient() -> Path:
    """Cheapest tier that closes the gap, per step x surface (round-3 rows only)."""
    grid = np.full((len(SURFACES), len(STEPS)), np.nan)
    text = [["" for _ in STEPS] for _ in SURFACES]
    for k, (fmt, corpus) in enumerate(SURFACES):
        for si, (src, tgt) in enumerate(STEPS):
            if (src, tgt) in SELFMAP_STEPS:
                text[k][si] = "n/a"  # no bootstrap draws => not computable
                continue
            row = mlp.get_row(f"{src}__{tgt}", fmt, corpus, SCALE)
            if row is None:
                text[k][si] = "—"
                continue
            st = row.get("sufficient_tier")
            if st == "none" or st is None:
                grid[k, si] = 9.0
                text[k][si] = "none"
            else:
                grid[k, si] = float(st)
                text[k][si] = str(st)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), layout="constrained")
    im = ax.imshow(grid, cmap="viridis_r", vmin=0, vmax=9, aspect="auto")
    ax.grid(False)  # a y-grid line through the cell text reads as a strikethrough
    for k in range(len(SURFACES)):
        for si in range(len(STEPS)):
            if np.isnan(grid[k, si]):
                color = "#555555"  # empty (white) cell — never white-on-white
            else:
                color = "white" if grid[k, si] > 4 else "black"
            ax.text(si, k, text[k][si], ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([step_label(s, t) for s, t in STEPS], fontsize=9)
    ax.set_yticks(range(len(SURFACES)))
    ax.set_yticklabels([f"{c} ({f})" for f, c in SURFACES], fontsize=9)
    ax.set_title("Cheapest tier that closes the gap", fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 6, 7, 8, 9], pad=0.02)
    cb.ax.set_yticklabels(["0 direct", "6 contexts", "7 answers", "8 both", "none"], fontsize=8)
    out = OUTDIR / "ladder_sufficient_tier_by_step.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    D = collect()

    missing = {k: v for k, v in D["coverage"].items() if v != len(SURFACES)}
    if missing:
        print("[step-transfer] PARTIAL COVERAGE (reported, never drawn as zeros):")
        for (si, tier), n in sorted(missing.items()):
            print(f"    step {step_label(*STEPS[si])!r} tier t{tier}: {n}/{len(SURFACES)} surfaces")

    paths = [fig_aggregate(D), fig_percorpus(D), fig_sufficient()]

    meta = {
        "issue": 1336,
        "question": (
            "per adjacent ladder step, how much of the context->answer map "
            "transfers to the next stage under each tier of reparameterization"
        ),
        "layer": LAYER,
        "scale": SCALE,
        "scale_note": "raw pooled R² (matches the selfmap fold-local pooled OOF basis)",
        "tiers_plotted": list(TIERS),
        "tiers_plotted_reason": (
            "intersection of the round-3 aggregate (t0-t8) and "
            "selfmap_v3 (t0/t6/t7/t8); t1-t5 are unmeasured for "
            "rlvr->rlvr_long and are omitted, not zeroed"
        ),
        "tier_labels": {str(t): TIER_LABEL[t] for t in TIERS},
        "steps": [f"{s}__{t}" for s, t in STEPS],
        "step_sources": {
            f"{s}__{t}": (
                "selfmap_v3 (round B, no bootstrap CI)"
                if (s, t) in SELFMAP_STEPS
                else "round-3 metric_ladder aggregate (paired-bootstrap CI)"
            )
            for s, t in STEPS
        },
        "surfaces": [f"{c} ({f})" for f, c in SURFACES],
        "ceiling": "within_r2 — the same-model map; transfer must be read against it, not against 1.0",
        "ci_note": (
            "R² intervals are the 1,000-draw paired-bootstrap GAP CI mapped through "
            "r2 = within_r2 - gap (a location shift; within_r2 is a per-row constant). "
            "rlvr->longer RLVR is POINT-ONLY (open markers): selfmap_v3 carries no "
            "bootstrap draws and no interval is borrowed for it."
        ),
        "aggregate_note": (
            "panels A/B show the MEDIAN over the 7 non-degenerate surfaces with every "
            "surface overplotted as a dot; no synthetic aggregate CI is invented. Real "
            "per-surface CIs appear in the per-corpus figure. Median not mean: at "
            "base->SFT the math corpora reach t0 R² = -2.32 while the conversational "
            "ones sit at +0.23, so a mean over that mix summarizes nothing."
        ),
        "degenerate_excluded": {
            "surfaces": [f"{c} ({f})" for f, c in sorted(DEGENERATE)],
            "reason": (
                "n_train (~4/5 of n) < d=4096 => every held-out R² here is "
                "estimator-degenerate, not a signal read. Excluded from every aggregate "
                "statistic and the ceiling; MARKED (shaded panel + 'x' markers) in the "
                "per-corpus view, never silently dropped. The committed "
                "adjacent_increments figure excludes this same cell."
            ),
        },
        "axis_clipping": {
            "panel_A_ylim": list(A_YLIM),
            "panel_B_ylim": list(B_YLIM),
            "per_corpus_ylim": list(PC_YLIM),
            "note": (
                "off-scale surfaces are drawn as carets AT the boundary (still "
                "represented, never dropped); their exact values are in the per-corpus "
                "figure and in eval_results/issue_1336/selfmap_v3 + the metric_ladder "
                "aggregate. Clipping exists because the math corpora reach t0 "
                "R² = -2.32 at base->SFT, which would compress the 0.0-0.6 band where "
                "every other reading lives."
            ),
        },
        "corpus_families": {
            "conversational": [f"{c} ({f})" for f, c in SURFACES if (f, c) in CONVERSATIONAL],
            "math_reasoning": [f"{c} ({f})" for f, c in SURFACES if (f, c) not in CONVERSATIONAL],
            "note": (
                "the dominant structure is a split between these families at base->SFT: "
                "direct transfer (t0) is mildly POSITIVE on conversational corpora and "
                "strongly NEGATIVE on math/reasoning ones."
            ),
        },
        "coverage": {
            f"{step_label(*STEPS[si]).replace(chr(10), ' ')}|t{t}": n
            for (si, t), n in sorted(D["coverage"].items())
        },
        "figures": [p.name for p in paths],
        "code": "scripts/issue1336_step_transfer_tiers.py",
    }
    (OUTDIR / "ladder_step_transfer_by_tier.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    print(
        "\n[step-transfer] MEDIAN transfer R² by step x tier "
        "(raw, layer 30; 7 non-degenerate corpora):"
    )
    hdr = "  tier  " + "".join(f"{step_label(s, t).replace(chr(10), ''):>16}" for s, t in STEPS)
    print(hdr)
    for tier in TIERS:
        cells_ = []
        for si in range(len(STEPS)):
            v = [(k, c) for k, c in D["data"][(si, tier)] if k not in DEGENERATE]
            cells_.append(f"{np.median([c['r2'] for _, c in v]):>16.4f}" if v else f"{'—':>16}")
        print(f"  t{tier}   " + "".join(cells_))
    ceilrow = "".join(f"{D['ceilings'][si]:>16.4f}" for si in range(len(STEPS)))
    print(f"  ceil " + ceilrow)
    for p in paths:
        print(f"[step-transfer] wrote {p}")


if __name__ == "__main__":
    main()
