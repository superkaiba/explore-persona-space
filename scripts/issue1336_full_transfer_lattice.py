#!/usr/bin/env python3
"""Issue #1336 — the FULL forward stage-transfer lattice (all 10 ordered pairs).

The committed step-transfer figures
(`issue1336_step_transfer_tiers`, `issue1336_step_transfer_with_crossmap`) plot
the four ADJACENT ladder steps only. Round B (`selfmap_v3`) filled in the three
non-adjacent pairs the round-3 metric ladder never ran, so every FORWARD pair
among the five checkpoints is now measured:

    round-3 metric ladder : base→SFT, base→DPO, base→RLVR, base→longer,
                            SFT→DPO, DPO→RLVR, DPO→longer          (7)
    round-B selfmap_v3    : SFT→RLVR, SFT→longer, RLVR→longer      (3)

10 of the 10 forward pairs; the 10 BACKWARD pairs (e.g. RLVR→SFT) were never
run and are absent from every panel — not zeroed, not interpolated.

Nothing is fitted here. Every number is READ from a committed JSON field, on
the one shared basis both batteries were built on (layer 30, fit_seed 0, 5
outer folds, fold-local pooled OOF, raw scale), which is what licenses one axis.

SERIES (identical meaning in both figures — one colour, one meaning):
  within-model ceiling  grey dashed  the TARGET stage's own map: the bar every
                                     transfer read is scored against, not 1.0
  0: direct transfer    dark blue    source operator W_s applied unchanged
  6: reparam contexts   blue         linear remap of the TARGET's contexts into
                                     source coordinates, then W_s (context side
                                     corrected, operator and answer side not)
  7: reparam answers    light blue   answer side corrected instead
  8: reparam both       pale blue    both sides corrected (the full linear
                                     change of coordinates)
  cross (fresh fit)     orange       a ridge map fitted DIRECTLY from source
                                     contexts to TARGET answers. NOT a tier: it
                                     throws W_s away, so it separates "the map
                                     changed" from "the representations moved",
                                     and it is not bounded by the ceiling.

REUSE: STEPS/SURFACES/TIERS/DEGENERATE/colours/`cell()`/`load_cross()` all come
from the two committed modules; this script widens the pair list and re-lays the
panels. `cell()` dispatches on `stt.SELFMAP_STEPS` to decide which battery a
pair comes from — the committed module lists only the one round-B pair its
adjacent-step figure needed, so that set is widened below. That is a parameter
change, not a logic change: `cell()` itself is reused verbatim, CI mapping
included.

Run from the issue-1336 worktree root (paths resolve to the MAIN checkout via
`repo_root()`, which is where the pair-file cache and figure dir live):
    uv run python scripts/issue1336_full_transfer_lattice.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the matplotlib/numpy imports.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue1336_step_transfer_tiers as stt  # noqa: E402
import issue1336_step_transfer_with_crossmap as xmap  # noqa: E402

REPO = stt.REPO
OUTDIR = stt.OUTDIR
TIERS = stt.TIERS
SURFACES = stt.SURFACES
DEGENERATE = stt.DEGENERATE
TIER_COLORS = stt.TIER_COLORS
TIER_LABEL = stt.TIER_LABEL
CEILING_COLOR = stt.CEILING_COLOR
CROSS_COLOR = xmap.CROSS_COLOR

# All 10 FORWARD pairs, ordered by source position on the ladder then by target.
PAIRS = (
    ("base", "sft"),
    ("base", "dpo"),
    ("base", "rlvr"),
    ("base", "rlvr_long"),
    ("sft", "dpo"),
    ("sft", "rlvr"),
    ("sft", "rlvr_long"),
    ("dpo", "rlvr"),
    ("dpo", "rlvr_long"),
    ("rlvr", "rlvr_long"),
)

# Pairs whose numbers come from round B (no bootstrap draws => point-only, drawn
# with open markers so a missing interval can never read as a zero-width one).
SELFMAP_PAIRS = {("sft", "rlvr"), ("sft", "rlvr_long"), ("rlvr", "rlvr_long")}
stt.SELFMAP_STEPS = SELFMAP_PAIRS

# --- layer selection (#1336 frozen report set) ------------------------------
# The pair files carry the FULL t0..t8 ladder, the 20-draw shuffled nulls and
# the cross map (`repswap_r2`) at every frozen layer {16, 21, 22, 30}. Only the
# identity+bias / kNN `baselines` block is layer-30-only: the ladder gates it on
# `full_tier_layers`, which the dispatcher set to [30] (issue1336_metric_ladder
# .py:840). So a non-30 layer renders every series EXCEPT identity+bias.
#
# Round B (selfmap_v3) ran at layer 30 ONLY, so the three non-adjacent pairs it
# contributed do not exist off layer 30: those layers plot the 7 round-3 pairs.
FROZEN_LAYERS = (16, 21, 22, 30)
LAYER = 30
for _i, _a in enumerate(sys.argv):
    if _a == "--layer" and _i + 1 < len(sys.argv):
        LAYER = int(sys.argv[_i + 1])
    elif _a.startswith("--layer="):
        LAYER = int(_a.split("=", 1)[1])
if LAYER not in FROZEN_LAYERS:
    raise SystemExit(f"--layer must be one of {FROZEN_LAYERS} (frozen report set); got {LAYER}")
LKEY = str(LAYER)
SUFFIX = "" if LAYER == 30 else f"_l{LAYER}"
HAS_IDENTITY = LAYER == 30  # identity+bias exists only at the full-tier layer

if LAYER != 30:
    SELFMAP_PAIRS = set()
    stt.SELFMAP_STEPS = SELFMAP_PAIRS

# Where each source stage's block starts, for the group separators / shading.
SOURCE_BLOCKS = [
    ("base", 0, 4),
    ("SFT", 4, 7),
    ("DPO", 7, 9),
    ("RLVR", 9, 10),
]

# Both baselines are drawn as LINES (user call 2026-08-11). identity+bias sits
# at -2.14..-3.03 (aggregate) / -0.95..-3.41 (per corpus), an order of magnitude
# below the results band, so a single axis reaching it squashes 0..0.6 into the
# top fifth. Each figure therefore uses a BROKEN y-axis: the results band on the
# main panel, the identity line on a short lower panel, diagonal break marks
# between. Nothing is clipped or rescaled — both baselines are true-valued lines.
A_YLIM = (-0.80, 0.72)
A_YLIM_IDENT = (-3.25, -1.85)
PC_YLIM = (-0.62, 0.72)
# The lower panel carries every OFF-SCALE baseline: identity+bias (-0.95..-3.41)
# AND the t0/t6 shuffled nulls, which reach -5.02 on math7500 (no answer-side
# refit to absorb the shuffled target mean). t7/t8 nulls sit at ~0 and stay on
# the main panel with the series they bound.
PC_YLIM_IDENT = (-5.35, -0.22)
PC_LOWER_TIERS = (0, 6)
PC_MAIN_TIERS = (7, 8)

# The module computes this at import (`load_cross()` returns a (values, provenance)
# TUPLE, so call the ready-made dict rather than re-invoking it) and validates that
# every adjacent-step cell is present; the non-adjacent forward pairs come from the
# same two globs, so they ride along.
CROSS = xmap.CROSS
CROSS_PROV = xmap.CROSS_PROV

if LAYER != 30:
    PAIRS = tuple(
        p for p in PAIRS if p not in {("sft", "rlvr"), ("sft", "rlvr_long"), ("rlvr", "rlvr_long")}
    )
    SOURCE_BLOCKS = [("base", 0, 4), ("SFT", 4, 5), ("DPO", 5, 7)]


def _layer_rows(layer_key: str) -> dict:
    """(pair, fmt, corpus) -> that layer's per_layer block, read from the pair files.

    The committed `mlp.get_row` path that `stt.cell` uses is baked to layer 30,
    so an off-30 layer reads the pair-file JSONs directly. Same files, same
    fields — this is a layer parameter, not a different measurement.
    """
    import glob
    import re

    pat = re.compile(r"pair_(.+?)__(.+?)_(chat|naturalistic)_(.+)\.json")
    out: dict = {}
    for fp in sorted(set(glob.glob(xmap.PAIRFILE_GLOB, recursive=True))):
        m = pat.match(Path(fp).name)
        if not m:
            continue
        src, tgt, fmt, corpus = m.groups()
        d = json.load(open(fp))
        layer = d.get("per_layer", {}).get(layer_key)
        if layer:
            out[(f"{src}__{tgt}", fmt, corpus)] = (layer, int(d.get("n_shared_rows", 0)))
    return out


if LAYER != 30:
    _ROWS = _layer_rows(LKEY)

    def _cell_at_layer(src, tgt, fmt, corpus, tier):
        """stt.cell() for an arbitrary frozen layer; identical return shape."""
        hit = _ROWS.get((f"{src}__{tgt}", fmt, corpus))
        if hit is None:
            return None
        layer, n = hit
        rec = layer["raw"]["tiers"].get(f"t{tier}")
        if rec is None:
            return None
        within = float(layer["raw"]["within_r2"])
        gb = rec.get("gap_bootstrap") or {}
        lo, hi = gb.get("ci_lo"), gb.get("ci_hi")
        # gap = within - r2, so the gap CI maps onto R2 by a location shift.
        return {
            "r2": float(rec["r2"]),
            "r2_lo": within - hi if hi is not None else None,
            "r2_hi": within - lo if lo is not None else None,
            "within_r2": within,
            "n": n,
            "has_ci": lo is not None,
        }

    stt.cell = _cell_at_layer
    CROSS = {
        k: float(v[0]["repswap_r2"]) for k, v in _ROWS.items() if v[0].get("repswap_r2") is not None
    }
    CROSS_PROV = dict(xmap.CROSS_PROV, layer=LAYER, source=f"per_layer/{LAYER}.repswap_r2")


# --- fair baselines -------------------------------------------------------
# Two floors, both already committed per round-3 pair file; neither is refitted.
#
#   shuffled-pairing null  the CAPACITY control. Per draw the target rows y_t are
#                          row-permuted and every y_t-consuming correction is
#                          REFIT (issue1336_metric_ladder.py:34-38, 756-763), so
#                          it answers the question the tier ladder invites: how
#                          much R² can the reparameterization machinery
#                          manufacture from destroyed correspondence? 20 draws.
#   identity + bias        ŷ = x + b, b = the train-fold mean of y - x. The
#                          project's standing mapping baseline (dims match at
#                          4096->4096): "is the fitted map better than 'the
#                          answer state is the context state plus a constant'?"
#
# Round B (selfmap_v3) carries NEITHER, so both are drawn over the 7 round-3
# pairs only and left as a visible GAP on the three round-B pairs — never
# interpolated across a pair that was not controlled.
NULL_COLOR = "#969696"
IDENT_COLOR = "#54278f"
BASELINE_PAIRS = tuple(p for p in PAIRS if p not in SELFMAP_PAIRS)
# nulls.order is [within, t0..t8]; the plotted tiers land at these column indices.
NULL_COL = {0: 1, 6: 7, 7: 8, 8: 9}


def load_baselines() -> dict:
    """(pair_index) -> {null_lo, null_hi, null_max_draw, identity_bias, n}.

    Read from the same round-3 pair files the tier lines come from, so the
    baselines share their basis exactly. Fails loud rather than plotting a
    baseline for a pair whose control was never run.
    """
    import glob
    import re

    pat = re.compile(r"pair_(.+?)__(.+?)_(chat|naturalistic)_(.+)\.json")
    by_pair: dict[str, dict[str, list]] = {}
    for fp in sorted(set(glob.glob(xmap.PAIRFILE_GLOB, recursive=True))):
        m = pat.match(Path(fp).name)
        if not m:
            continue
        src, tgt, fmt, corpus = m.groups()
        if (fmt, corpus) in DEGENERATE:
            continue
        layer = json.load(open(fp)).get("per_layer", {}).get(LKEY)
        if not layer:
            continue
        rec = by_pair.setdefault(f"{src}__{tgt}", {"null": [], "ident": []})
        mat = np.asarray(layer["nulls"]["r2_matrix"], dtype=float)
        rec["null"].append(mat[:, [NULL_COL[t] for t in TIERS]])
        _bl = layer.get("baselines")
        rec["ident"].append(float(_bl["within"]["identity_bias_r2"]) if _bl else float("nan"))

    out: dict[int, dict] = {}
    for pi, (src, tgt) in enumerate(PAIRS):
        rec = by_pair.get(f"{src}__{tgt}")
        if rec is None:
            if (src, tgt) not in SELFMAP_PAIRS:
                raise RuntimeError(
                    f"no null/identity baseline for round-3 pair {src}->{tgt}; the pair-file "
                    "cache under data/issue_1336/hf_dl/ is reapable — re-fetch from the HF "
                    "prefix issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder"
                )
            continue  # round B ran no controls: leave the gap
        # per corpus: mean over the 20 draws per tier -> median across corpora
        per_tier = np.median(np.stack([m.mean(axis=0) for m in rec["null"]]), axis=0)
        out[pi] = {
            "null_lo": float(per_tier.min()),
            "null_hi": float(per_tier.max()),
            "null_per_tier": {int(t): float(v) for t, v in zip(TIERS, per_tier)},
            "null_max_single_draw": float(max(m.max() for m in rec["null"])),
            "identity_bias": float(np.median(rec["ident"])),
            "identity_bias_range": [float(min(rec["ident"])), float(max(rec["ident"]))],
            "n_corpora": len(rec["ident"]),
        }
    return out


BASELINES = load_baselines()


def pair_label(src: str, tgt: str) -> str:
    short = {
        "base": "base",
        "sft": "SFT",
        "dpo": "DPO",
        "rlvr": "RLVR",
        "rlvr_long": "longer\nRLVR",
    }
    return f"{short[src]}→{short[tgt]}"


def _xticks(ax) -> None:
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels([pair_label(s, t) for s, t in PAIRS], fontsize=8.5)
    ax.set_xlim(-0.5, len(PAIRS) - 0.5)
    for _, lo, hi in SOURCE_BLOCKS[:-1]:
        ax.axvline(hi - 0.5, color="#bdbdbd", lw=0.8, ls=":", zorder=0)


def _break_marks(ax_top, ax_bot, d: float = 0.014) -> None:
    """Diagonal break marks on a shared-x broken y-axis pair (matplotlib recipe)."""
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    kw = dict(
        marker=[(-1, -d), (1, d)],
        markersize=9,
        linestyle="none",
        color="#252525",
        mec="#252525",
        mew=1.0,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kw)


def collect() -> dict:
    """(pair_index, tier) -> [((fmt, corpus), cell)]; plus ceilings and cross."""
    data, ceilings, cross = {}, {}, {}
    for pi, (src, tgt) in enumerate(PAIRS):
        ceil_vals: list[float] = []
        for tier in TIERS:
            vals = []
            for fmt, corpus in SURFACES:
                c = stt.cell(src, tgt, fmt, corpus, tier)
                if c is None:
                    continue
                vals.append(((fmt, corpus), c))
                if (fmt, corpus) not in DEGENERATE:
                    ceil_vals.append(c["within_r2"])
            if not vals:
                raise RuntimeError(
                    f"no cells for {src}->{tgt} tier {tier}: the pair-file cache under "
                    "data/issue_1336/hf_dl/ is reapable — re-fetch the 56 files from the HF "
                    "prefix issue1336_rlvr_ladder/eval_results_mirror_v2/metric_ladder"
                )
            data[(pi, tier)] = vals
        ceilings[pi] = float(np.median(ceil_vals))
        xs = [
            CROSS[(f"{src}__{tgt}", fmt, corpus)]
            for fmt, corpus in SURFACES
            if (f"{src}__{tgt}", fmt, corpus) in CROSS and (fmt, corpus) not in DEGENERATE
        ]
        cross[pi] = (float(np.median(xs)) if xs else np.nan, len(xs))
    return {"data": data, "ceilings": ceilings, "cross": cross}


def fig_aggregate(D: dict) -> Path:
    if not HAS_IDENTITY:
        # No identity+bias at this layer (the ladder gates the baselines block to
        # full_tier_layers=[30]), so there is nothing off-scale to break for.
        # add_axes with an explicit rect bypasses the style's layout engine
        # entirely (set_layout_engine("none") keeps the prior engine's
        # adjust_compatible flag, so subplots_adjust is silently skipped).
        fig = plt.figure(figsize=(12.4, 7.3))
        ax = fig.add_axes([0.075, 0.205, 0.920, 0.700])
        axb = None
    else:
        fig, (ax, axb) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12.4, 7.3),
            gridspec_kw={"height_ratios": [4.3, 1.0], "hspace": 0.08},
        )
    # The project plot style enables an auto layout engine, which silently
    # overrides hspace / subplots_adjust and pulls the broken-axis panels apart.
    fig.set_layout_engine("none")
    x = np.arange(len(PAIRS))

    for a in [ax] if axb is None else (ax, axb):
        for _, lo, hi in SOURCE_BLOCKS[::2]:
            a.axvspan(lo - 0.5, hi - 0.5, color="#f7f7f7", zorder=0)

    # --- fair baselines, both as LINES, drawn BEHIND the tier lines ---
    ax.axhline(0.0, color="#252525", lw=1.0, ls=(0, (6, 3)), zorder=1)
    ax.text(
        -0.42,
        0.05,
        "R² = 0 · predict the training mean  (the t7/t8 shuffled null sits 0.0008 under it)",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#252525",
    )
    # NaN at the three round-B pairs (no control was run there) breaks each
    # baseline line into a visible gap rather than bridging an uncontrolled pair.
    # ONE NULL PER TIER, coloured to its own series (user call 2026-08-11). The
    # tiers' nulls differ by ~1000x: t7/t8 refit the answer side against the
    # SHUFFLED targets, so the refit absorbs the target mean and the null
    # collapses to -0.0008; t0/t6 have no answer-side refit to absorb it and sit
    # at -0.3..-0.75. Only t0/t6 are DRAWN -- the t7/t8 null is visually
    # coincident with the R2=0 line already on the axis (it differs by 0.0008 on
    # a 1.5-unit axis), so drawing it would add a second zero line carrying no
    # readable information; its value rides the legend instead. The collapsed
    # max-over-tiers line this replaces had exactly that problem, and it held the
    # t7/t8 bar up against the t0/t6 series, making real t0 signal read as noise.
    for tier in PC_LOWER_TIERS:
        ax.plot(
            x,
            [
                BASELINES[pi]["null_per_tier"][tier] if pi in BASELINES else np.nan
                for pi in range(len(PAIRS))
            ],
            color=TIER_COLORS[tier],
            lw=1.3,
            ls=(0, (5, 2)),
            alpha=0.9,
            zorder=2,
        )
    ax.plot(
        [],
        [],
        color=NULL_COLOR,
        lw=1.3,
        ls=(0, (5, 2)),
        label="shuffled-pairing null, per tier (t0/t6 drawn; t7/t8 = −0.0008 ≈ the zero line)",
    )
    # identity+bias lives on the lower (broken-axis) panel at its TRUE value;
    # the proxy handle carries it into the main panel's legend.
    ident = np.array(
        [BASELINES[pi]["identity_bias"] if pi in BASELINES else np.nan for pi in range(len(PAIRS))]
    )
    ident_style = dict(color=IDENT_COLOR, lw=2.0, ls=(0, (5, 2)), marker=".", ms=7)
    if axb is not None:
        axb.plot(x, ident, zorder=2, **ident_style)
    if HAS_IDENTITY:
        ax.plot([], [], label="identity+bias baseline  ŷ = x + b  (lower panel)", **ident_style)

    for tier in TIERS:
        med = [
            float(np.median([c["r2"] for (s, c) in D["data"][(pi, tier)] if s not in DEGENERATE]))
            for pi in range(len(PAIRS))
        ]
        ax.plot(
            x,
            med,
            marker="o",
            ms=6,
            lw=1.9,
            color=TIER_COLORS[tier],
            label=f"{tier}: {TIER_LABEL[tier]}",
            zorder=4,
        )
        # every corpus overplotted, so the median is never the only thing shown
        for pi in range(len(PAIRS)):
            for surf, c in D["data"][(pi, tier)]:
                if surf in DEGENERATE:
                    continue
                y = c["r2"]
                if y < A_YLIM[0]:
                    ax.plot(
                        [pi], [A_YLIM[0]], marker="v", ms=5, color=TIER_COLORS[tier], clip_on=False
                    )
                else:
                    ax.plot([pi], [y], marker="o", ms=3.2, alpha=0.35, color=TIER_COLORS[tier])
        for pi, (src, tgt) in enumerate(PAIRS):
            if (src, tgt) in SELFMAP_PAIRS:
                ax.plot(
                    [pi],
                    [med[pi]],
                    marker="o",
                    ms=10,
                    mfc="none",
                    mec=TIER_COLORS[tier],
                    mew=1.4,
                    zorder=5,
                )

    ax.plot(
        x,
        [D["cross"][pi][0] for pi in range(len(PAIRS))],
        marker="D",
        ms=6.5,
        lw=2.0,
        ls="--",
        color=CROSS_COLOR,
        label=xmap.CROSS_LABEL,
        zorder=6,
    )
    ax.plot(
        x,
        [D["ceilings"][pi] for pi in range(len(PAIRS))],
        marker="s",
        ms=5,
        lw=1.5,
        ls="--",
        color=CEILING_COLOR,
        label="within-model ceiling (target's own map)",
        zorder=6,
    )

    for name, lo, hi in SOURCE_BLOCKS:
        ax.text(
            (lo + hi - 1) / 2,
            A_YLIM[1] - 0.035,
            f"source: {name}",
            ha="center",
            fontsize=9.5,
            color="#404040",
        )

    ax.set_ylim(*A_YLIM)
    if axb is not None:
        axb.set_ylim(*A_YLIM_IDENT)
        axb.set_yticks([-2.0, -3.0])
    ax.set_ylabel(f"held-out R²  (raw pooled, layer {LAYER})", fontsize=11)
    # No xlabel: the tick labels ARE the pairs ("base→SFT"), the title says
    # "every forward pair", and the freed strip carries the 8-entry legend.
    ax.set_title(
        "Every forward pair of the Tülu-3 ladder: how much of the context→answer map survives\n"
        "median over 7 non-degenerate corpora, every corpus overplotted; open markers = "
        "point-only (round B, no bootstrap draws)",
        fontsize=11.5,
        loc="left",
    )
    for a in [ax] if axb is None else (ax, axb):
        a.grid(axis="y", alpha=0.22, lw=0.6)
        _xticks(a)
    if axb is not None:
        _break_marks(ax, axb)
    # Figure-level legend BELOW the axes: the per-tier t0/t6 null lines now occupy
    # the lower-left/mid of the data area, so an in-axes legend overlaps them.
    h, lab = ax.get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4, fontsize=8.5, frameon=False
    )
    # subplots_adjust, not tight_layout: tight_layout recomputes hspace and
    # would pull the two broken-axis panels apart.
    if axb is not None:
        fig.subplots_adjust(left=0.075, right=0.995, top=0.905, bottom=0.20)
    out = OUTDIR / f"ladder_full_transfer_lattice{SUFFIX}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _panel_baselines(fmt: str, corpus: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """This CORPUS's own baselines, per pair: ({tier -> null}, identity+bias).

    The null is returned PER TIER (20-draw mean), not collapsed: t7/t8 refit the
    answer side against the shuffled targets so their null sits at ~0, while
    t0/t6 have no answer-side refit to absorb the target mean and sit at
    -0.3..-0.75. Collapsing them holds the t7/t8 bar up against the t0/t6
    series and makes real t0 signal read as failure. NaN at the three round-B
    pairs (no control was run there) so each baseline LINE breaks into a
    visible gap instead of bridging an uncontrolled pair.
    """
    import glob
    import re

    pat = re.compile(r"pair_(.+?)__(.+?)_(chat|naturalistic)_(.+)\.json")
    found: dict[str, tuple[np.ndarray, float]] = {}
    for fp in sorted(set(glob.glob(xmap.PAIRFILE_GLOB, recursive=True))):
        m = pat.match(Path(fp).name)
        if not m:
            continue
        src, tgt, f, c = m.groups()
        if (f, c) != (fmt, corpus):
            continue
        layer = json.load(open(fp)).get("per_layer", {}).get(LKEY)
        if layer:
            mat = np.asarray(layer["nulls"]["r2_matrix"], dtype=float)
            _bl = layer.get("baselines")
            found[f"{src}__{tgt}"] = (
                mat[:, [NULL_COL[t] for t in TIERS]].mean(axis=0),
                float(_bl["within"]["identity_bias_r2"]) if _bl else float("nan"),
            )
    per_tier: dict[int, list[float]] = {t: [] for t in TIERS}
    identv = []
    for src, tgt in PAIRS:
        v = found.get(f"{src}__{tgt}")
        for i, t in enumerate(TIERS):
            per_tier[t].append(np.nan if v is None else float(v[0][i]))
        identv.append(np.nan if v is None else v[1])
    return {t: np.array(vs) for t, vs in per_tier.items()}, np.array(identv)


def fig_percorpus(D: dict) -> Path:
    # Each corpus gets a BROKEN y-axis: results band on the main panel, the
    # identity+bias line on a short lower panel (spacer row 2 separates the
    # two blocks so a break row never crowds the next block's titles).
    fig = plt.figure(figsize=(16.0, 10.4))
    fig.set_layout_engine("none")  # see fig_aggregate: the style's engine wins otherwise
    # Geometry goes ON the GridSpec: fig.subplots_adjust does not reach a
    # manually created gridspec, so the panels would keep their default box.
    gs = fig.add_gridspec(
        5,
        4,
        height_ratios=[3.5, 0.9, 0.42, 3.5, 0.9],
        hspace=0.11,
        wspace=0.13,
        left=0.045,
        right=0.995,
        top=0.895,
        bottom=0.115,
    )
    mains: list = []
    breaks: list = []
    for k in range(len(SURFACES)):
        r_main, c = (0, k) if k < 4 else (3, k - 4)
        m = fig.add_subplot(
            gs[r_main, c],
            sharex=mains[0] if mains else None,
            sharey=mains[0] if mains else None,
        )
        b = fig.add_subplot(gs[r_main + 1, c], sharex=m, sharey=breaks[0] if breaks else None)
        mains.append(m)
        breaks.append(b)

    ident_style = dict(color=IDENT_COLOR, lw=1.7, ls=(0, (5, 2)), marker=".", ms=5)
    for k, (fmt, corpus) in enumerate(SURFACES):
        ax, axb = mains[k], breaks[k]
        deg = (fmt, corpus) in DEGENERATE
        # fair baselines, per panel, behind the tier lines — this CORPUS's own
        # values, not the aggregate's. Both are LINES; NaN at the round-B pairs.
        ax.axhline(0.0, color="#252525", lw=0.9, ls=(0, (6, 3)), zorder=1)
        pnull, pident = _panel_baselines(fmt, corpus)
        # t0/t6 nulls reach -5.02 on math7500, so they share the lower panel with
        # identity+bias; t7/t8 nulls sit at ~-0.0008, coincident with the zero
        # line already drawn above. See fig_aggregate for the full rationale.
        for tier in PC_LOWER_TIERS:
            axb.plot(
                np.arange(len(PAIRS)),
                pnull[tier],
                color=TIER_COLORS[tier],
                lw=1.2,
                ls=(0, (5, 2)),
                alpha=0.9,
                zorder=2,
            )
        if HAS_IDENTITY:
            axb.plot(np.arange(len(PAIRS)), pident, zorder=3, **ident_style)
        if k == 0:  # proxies so the figure legend carries the lower-panel lines
            ax.plot(
                [],
                [],
                color=NULL_COLOR,
                lw=1.2,
                ls=(0, (5, 2)),
                label="shuffled-pairing null, per tier (t0/t6 lower panel; t7/t8 ≈ the zero line)",
            )
            if HAS_IDENTITY:
                ax.plot(
                    [], [], label="identity+bias baseline  ŷ = x + b  (lower panel)", **ident_style
                )
        for tier in TIERS:
            xa, ys, lo, hi = [], [], [], []
            for pi, (src, tgt) in enumerate(PAIRS):
                c = stt.cell(src, tgt, fmt, corpus, tier)
                if c is None:
                    continue
                xa.append(pi)
                ys.append(c["r2"])
                lo.append(c["r2"] - c["r2_lo"] if c["has_ci"] else 0.0)
                hi.append(c["r2_hi"] - c["r2"] if c["has_ci"] else 0.0)
            ax.errorbar(
                xa,
                ys,
                yerr=[lo, hi],
                marker="x" if deg else "o",
                ms=4.5,
                lw=1.6,
                capsize=2.0,
                color=TIER_COLORS[tier],
                label=f"{tier}: {TIER_LABEL[tier]}",
            )
            for pi, y in zip(xa, ys):
                if y < PC_YLIM[0]:
                    ax.plot(
                        [pi],
                        [PC_YLIM[0]],
                        marker="v",
                        ms=6,
                        color=TIER_COLORS[tier],
                        zorder=6,
                        clip_on=False,
                    )
                if PAIRS[pi] in SELFMAP_PAIRS:
                    ax.plot(
                        [pi],
                        [y],
                        marker="o",
                        ms=8,
                        mfc="none",
                        mec=TIER_COLORS[tier],
                        mew=1.3,
                        zorder=5,
                    )

        xa = [pi for pi, (s, t) in enumerate(PAIRS) if (f"{s}__{t}", fmt, corpus) in CROSS]
        ax.plot(
            xa,
            [CROSS[(f"{PAIRS[pi][0]}__{PAIRS[pi][1]}", fmt, corpus)] for pi in xa],
            marker="D",
            ms=5,
            lw=1.8,
            ls="--",
            color=CROSS_COLOR,
            label=xmap.CROSS_LABEL,
        )
        cvals = [stt.cell(s, t, fmt, corpus, TIERS[0]) for s, t in PAIRS]
        ax.plot(
            [i for i, c in enumerate(cvals) if c],
            [c["within_r2"] for c in cvals if c],
            marker="s",
            ms=4,
            ls="--",
            lw=1.3,
            color=CEILING_COLOR,
            label="within-model ceiling",
        )

        title = f"{corpus} ({fmt})" + ("  — DEGENERATE n_train<d" if deg else "")
        ax.set_title(title, fontsize=9.5, loc="left", color="#8c2d04" if deg else "black")
        for a in (ax, axb):
            if deg:
                a.set_facecolor("#fdf2e9")
            a.grid(axis="y", alpha=0.22, lw=0.6)
            _xticks(a)
        axb.tick_params(axis="x", labelrotation=90, labelsize=7)
        _break_marks(ax, axb)
        if k < 4:  # top block: the bottom block carries the shared x labels
            axb.tick_params(axis="x", labelbottom=False)

    mains[0].set_ylim(*PC_YLIM)  # sharey => applies to all 8 main panels
    breaks[0].set_ylim(*PC_YLIM_IDENT)
    breaks[0].set_yticks([-1.0, -3.0, -5.0])
    for k in (0, 4):
        mains[k].set_ylabel("held-out R²", fontsize=10)
    h, lab = mains[0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=4, fontsize=8.5, frameon=False
    )
    fig.suptitle(
        "Per eval dataset: every forward stage pair, at each reparameterization tier",
        fontsize=12,
        x=0.01,
        y=0.995,
        ha="left",
        va="top",
    )
    out = OUTDIR / f"ladder_full_transfer_lattice_by_dataset{SUFFIX}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def write_meta(D: dict, figs: list[Path]) -> Path:
    rows = []
    for pi, (src, tgt) in enumerate(PAIRS):
        for fmt, corpus in SURFACES:
            rec: dict = {
                "pair": f"{src}__{tgt}",
                "source": src,
                "target": tgt,
                "format": fmt,
                "corpus": corpus,
                "degenerate": (fmt, corpus) in DEGENERATE,
                "battery": "round-B selfmap_v3"
                if (src, tgt) in SELFMAP_PAIRS
                else "round-3 metric_ladder",
            }
            for tier in TIERS:
                c = stt.cell(src, tgt, fmt, corpus, tier)
                if c is None:
                    continue
                rec[f"t{tier}_r2"] = c["r2"]
                if c["has_ci"]:
                    rec[f"t{tier}_r2_lo"], rec[f"t{tier}_r2_hi"] = c["r2_lo"], c["r2_hi"]
                rec.setdefault("within_r2", c["within_r2"])
                rec.setdefault("n", c["n"])
            k = (f"{src}__{tgt}", fmt, corpus)
            if k in CROSS:
                rec["cross_r2"] = CROSS[k]
            rows.append(rec)

    meta = {
        "issue": 1336,
        "question": (
            "for every FORWARD pair of the five-checkpoint Tulu-3 ladder, how much of the "
            "context->answer map transfers under each reparameterization tier, against the "
            "target's own within-model ceiling and a map fitted directly from source contexts "
            "to target answers"
        ),
        "layer": LAYER,
        "scale": "raw pooled R2, fold-local pooled OOF",
        "pairs_measured": [f"{s}__{t}" for s, t in PAIRS],
        "pairs_absent": (
            "the 10 BACKWARD pairs (target earlier on the ladder than source) were never run; "
            "they are absent from every panel, not zeroed"
        ),
        "batteries": {
            "round-3 metric_ladder": [f"{s}__{t}" for s, t in PAIRS if (s, t) not in SELFMAP_PAIRS],
            "round-B selfmap_v3": [f"{s}__{t}" for s, t in sorted(SELFMAP_PAIRS)],
            "shared_basis": (
                f"layer {LAYER}, fit_seed 0, 5 outer folds, fold-local pooled OOF, raw scale"
            ),
        },
        "tier_labels": {str(t): TIER_LABEL[t] for t in TIERS},
        "cross_map_definition": xmap.CROSS_MAP_DEFINITION
        if hasattr(xmap, "CROSS_MAP_DEFINITION")
        else (
            "ridge fitted DIRECTLY from X = v_context(source) to Y = v_answer(target), scored on "
            "the target's held-out answers; NOT a tier and NOT bounded by the ceiling"
        ),
        "ci_note": (
            "tier R2 intervals are the 1,000-draw paired-bootstrap GAP CI mapped through "
            "r2 = within_r2 - gap. Round-B pairs (SFT->RLVR, SFT->longer, RLVR->longer) carry no "
            "bootstrap draws and are POINT-ONLY (open markers); no interval is borrowed. The "
            "cross map is point-only on every pair."
        ),
        "degenerate_excluded": {
            "surfaces": [f"{c} ({f})" for f, c in sorted(DEGENERATE)],
            "reason": "n_train (~1034) < d=4096 => estimator-degenerate; excluded from every "
            "aggregate and MARKED (shaded panel, x markers) per-dataset, never silently dropped",
        },
        "within_ceiling_note": (
            "each pair is scored against ITS OWN pair-file within-model ceiling. Two pairs sharing "
            "a target can differ slightly in that ceiling because the two batteries kept slightly "
            "different row sets (e.g. RLVR-target lmsys23k: 0.609 from round 3, 0.610 from round B)"
        ),
        "fair_baselines": {
            "shuffled_pairing_null": (
                "the CAPACITY control: per draw the target rows y_t are row-permuted and every "
                "y_t-consuming tier correction is REFIT (issue1336_metric_ladder.py:34-38, "
                "756-763), so it bounds how much R2 the reparameterization machinery can "
                "manufacture from destroyed correspondence. 20 draws per fit; the plotted LINE "
                "is the WORST (max) of the 4 plotted tiers' draw-means — the tightest bound — "
                "median across the 7 non-degenerate corpora on the aggregate figure and the "
                "corpus's own value per panel. The per-tier draw-means are in null_per_tier "
                "below: the strict tiers (t0, t6) sit at -0.32..-0.75 and only t7/t8 approach 0."
            ),
            "identity_bias": (
                "y_hat = x + b with b the train-fold mean of y - x (analysis/mapping_baselines); "
                "the project's standing mapping baseline, applicable because dims match at "
                "4096->4096. Drawn as a LINE at its true value (-2.14..-3.03 aggregate, worst "
                "per-corpus cell -3.41), which is why the y-axis extends to -3.2 / -3.6."
            ),
            "zero_line": "R2 = 0 is the predict-the-training-mean baseline.",
            "round_b_gap": (
                "round B (selfmap_v3) ran NEITHER control, so both baselines are drawn over the "
                "7 round-3 pairs only and left as a visible GAP on sft__rlvr, sft__rlvr_long and "
                "rlvr__rlvr_long — never interpolated across an uncontrolled pair"
            ),
            "per_pair": {
                f"{PAIRS[pi][0]}__{PAIRS[pi][1]}": b for pi, b in sorted(BASELINES.items())
            },
        },
        "figures": [f.name for f in figs],
        "code": "scripts/issue1336_full_transfer_lattice.py",
        "extends": "ladder_step_transfer_by_dataset.png (adjacent steps only)",
        "rows": rows,
    }
    out = OUTDIR / f"ladder_full_transfer_lattice{SUFFIX}.meta.json"
    out.write_text(json.dumps(meta, indent=1))
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    D = collect()

    print(f"{'pair':<20}{'within':>8}{'t0':>8}{'t6':>8}{'t7':>8}{'t8':>8}{'cross':>8}  battery")
    for pi, (src, tgt) in enumerate(PAIRS):
        med = {
            t: float(np.median([c["r2"] for (s, c) in D["data"][(pi, t)] if s not in DEGENERATE]))
            for t in TIERS
        }
        bat = "round-B" if (src, tgt) in SELFMAP_PAIRS else "round-3"
        print(
            f"{src + '→' + tgt:<20}{D['ceilings'][pi]:>8.3f}"
            + "".join(f"{med[t]:>8.3f}" for t in TIERS)
            + f"{D['cross'][pi][0]:>8.3f}  {bat}"
        )

    figs = [fig_aggregate(D), fig_percorpus(D)]
    meta = write_meta(D, figs)
    for p in figs + [meta]:
        print("wrote", p)


if __name__ == "__main__":
    main()
