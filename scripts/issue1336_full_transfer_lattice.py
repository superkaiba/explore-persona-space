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
                                     Each source block opens with an S→S self
                                     column whose target IS the source, so that
                                     stage's OWN map is the same quantity and
                                     sits on this SAME line (triangle marker) —
                                     the ceiling and the source self-map were
                                     never two different measurements. Base
                                     ~0.34 pooled vs ~0.55-0.58 post-training.
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
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue1336_step_transfer_tiers as stt  # noqa: E402
import issue1336_step_transfer_with_crossmap as xmap  # noqa: E402

REPO = stt.REPO
OUTDIR = stt.OUTDIR
TIERS = stt.TIERS
SURFACES = stt.SURFACES
DEGENERATE = stt.DEGENERATE
# (stt.TIER_COLORS is deliberately NOT reused — see the palette block below.)
TIER_LABEL = stt.TIER_LABEL
CEILING_COLOR = stt.CEILING_COLOR
CROSS_COLOR = xmap.CROSS_COLOR

# ONE LINE STYLE for every tier (solid, round marker) and a QUALITATIVE palette
# instead of two sequential ramps (user call 2026-08-12: "make the colors more
# different — and choose one consistent line style"). Two ramps encoded family
# membership in hue and freedom in lightness, which made nine near-coincident
# lines unreadable wherever the tiers cluster; hue now carries tier IDENTITY
# only, and the non-tier series keep their own dashed styles so they stay
# separable from the tier bundle: ceiling grey dashed square, cross orange
# dashed diamond, identity+bias purple dashed (break panel).
#
# The palette is tab10 MINUS orange and grey — the two hues already spoken for
# by the cross-fit and the ceiling — plus black for tier 0. Assignment puts the
# three most-read tiers on the three strongest, most separable hues (6 blue,
# 7 green, 8 purple) and keeps adjacent tier indices far apart in hue, because
# adjacent tiers are exactly the lines that run closest together in value.
# Applied in BOTH modes so the default and --full-ladder figures agree on
# tier -> colour; the older stt-emitted figures keep their own blue ramp.
TIER_COLORS = {
    0: "#000000",  # direct transfer — neutral dark, the no-correction reference
    1: "#8C564B",  # context offset
    2: "#BCBD22",  # answer offset
    3: "#17BECF",  # bias offset
    4: "#E377C2",  # global scaling
    5: "#D62728",  # rotation
    6: "#1F77B4",  # reparam contexts
    7: "#2CA02C",  # reparam answers
    8: "#9467BD",  # reparam both
}

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

# --full-ladder: draw ALL NINE ladder tiers instead of the four the default
# figures carry. The extra five are the CONSTRAINED corrections, already
# computed and committed by the round-3 metric_ladder battery — nothing is
# fitted here either. The three round-B pairs (sft->rlvr, sft->rlvr_long,
# rlvr->rlvr_long) come from the selfmap battery, which ran only t0/t6/t7/t8, so
# they are ABSENT at tiers 1-5 and render as gaps in the line — never zeroed,
# never interpolated (the same contract the default figures already state).
FULL_LADDER = "--full-ladder" in sys.argv
# --tiers 0,5,6,7 : draw an EXPLICIT tier subset. The reporting set as of
# 2026-08-12 is t0/t5/t6/t7 — the tiers that survive the vacuity objection:
# t0 applies the source operator unchanged (no fitted correction at all), and
# t5/t6/t7 each carry ONE constrained correction whose parameters are fit
# against their own objective, never against the composite residual. t8 is
# excluded on purpose: under exact correspondence A_ans*W_s*A_ctx IS the optimal
# map by change of basis, so t8 = ceiling is a theorem rather than a finding
# (empirically r(A_ans fit, ceiling-t8) = -0.97 over 49 cells).
_TIER_SEL = None
for _i, _a in enumerate(sys.argv):
    if _a == "--tiers" and _i + 1 < len(sys.argv):
        _TIER_SEL = sys.argv[_i + 1]
    elif _a.startswith("--tiers="):
        _TIER_SEL = _a.split("=", 1)[1]
if FULL_LADDER and _TIER_SEL:
    raise SystemExit("--full-ladder and --tiers are mutually exclusive")
if FULL_LADDER:
    TIERS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    SUFFIX += "_fullladder"
elif _TIER_SEL:
    TIERS = tuple(int(_t) for _t in _TIER_SEL.split(",") if _t.strip())
    _bad = [_t for _t in TIERS if _t not in TIER_COLORS]
    if _bad:
        raise SystemExit(f"--tiers: unknown tier(s) {_bad}; valid tiers are 0-8")
    SUFFIX += "_t" + "".join(str(_t) for _t in TIERS)

# The alignment panel plots the identity+bias baselines of A_ctx and A_ans —
# the two remaps t6 and t7 apply. With NEITHER tier drawn it describes maps
# that appear nowhere in the figure, so it is dropped rather than left
# orphaned (user call 2026-08-14: t6/t7 out of the reporting set). Both values
# still ride the .meta.json sidecar either way. MUST sit after the tier
# selection above — TIERS holds the stt default until then.
HAS_ALIGN_PANEL = HAS_IDENTITY and bool({6, 7} & set(TIERS))

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
A_YLIM = (-0.80, 1.05)
A_YLIM_IDENT = (-3.25, -1.85)
# The alignment-map identity+bias panel spans -0.75 (base->post-training pairs)
# to +0.96 (dpo->rlvr); headroom above 1.0 for its own two-entry legend.
A_YLIM_ALIGN = (-0.95, 1.62)
PC_YLIM = (-0.62, 0.72)
# The lower panel now carries identity+bias ONLY (-0.95..-3.41). It used to also
# hold the t0/t6 shuffled nulls, which reach -5.02 on math7500 and forced the
# range down to -5.35; with the nulls no longer drawn (user call 2026-08-12) the
# strip tightens onto identity+bias, so the line is readable instead of a flat
# trace squeezed against the top of a 5-unit axis.
PC_YLIM_IDENT = (-3.70, -0.80)

# --- owning-STAGE grouping for the per-dataset figure (user call 2026-08-12) ---
# ONE COLUMN PER STAGE: horizontal position is the ladder stage whose training
# mix this corpus came from, in ladder order, with the stage-agnostic chat
# column first. This supersedes the earlier data-TYPE grouping, whose column
# names ("Instruction-following" vs "Alignment training") described the KIND of
# text while the panels' own subtypes described the owning STAGE — two
# cross-cutting taxonomies competing for one axis. Now they agree.
#
# Attribution (verified against scripts/issue1336_stage_corpora.py):
#   lmsys23k x2       lmsys/lmsys-chat-1m               -> no stage
#   sft11k            allenai/tulu-3-sft-mixture        -> SFT
#   uf11k             ...-tulu-3-8b-preference-mixture  -> DPO
#   if11k, math7500   RLVR-GSM-MATH-IF-Mixed-Constraints, components ifeval /
#                     MATH                              -> RLVR
#   gsm8k_train_full  openai/gsm8k train, 7,473 rows == the RLVR mix's own gsm8k
#                     component count (RLVR_COMPONENT_COUNTS) -> RLVR
# RLVR-long re-trains on the RLVR mix, so it introduces no new corpus and gets
# no column of its own.
#
# The DEGENERATE surface is excluded from this grouping, so it gets no panel:
# gsm8k_test1319 has n_train ~ 1,034 < d = 4,096, which makes every held-out R2
# on it estimator-degenerate rather than a signal read. It was previously drawn
# in a tinted panel with a DEGENERATE tag; dropping it entirely (user call
# 2026-08-12) is the stronger form of the same disclosure. It is already
# excluded from every median and every overplotted point in the aggregate
# figure, so this changes the per-dataset figure only.
TYPE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "No stage": (("chat", "lmsys23k"), ("naturalistic", "lmsys23k")),
    "SFT's data": (("chat", "sft11k"),),
    "DPO's data": (("chat", "uf11k"),),
    "RLVR's data": (
        ("chat", "if11k"),
        ("chat", "math7500"),
        ("chat", "gsm8k_train_full"),
    ),
}
TYPE_ORDER = tuple(TYPE_COLUMNS)
# Corpus -> the ladder stage that trained on it (None = no stage trained on it).
# Derived from TYPE_COLUMNS so the two can never disagree; consumed by the
# touched/untouched split in fig_trained_on (below).
_STAGE_OF_COLUMN = {
    "No stage": None,
    "SFT's data": "sft",
    "DPO's data": "dpo",
    "RLVR's data": "rlvr",
}
OWNING_STAGE: dict[tuple[str, str], str | None] = {
    surf: _STAGE_OF_COLUMN[col] for col, surfs in TYPE_COLUMNS.items() for surf in surfs
}
assert set(_STAGE_OF_COLUMN) == set(TYPE_COLUMNS), (
    "_STAGE_OF_COLUMN must name every column — a column added without a stage "
    "would silently drop its corpora from the used-between/not-used-between split"
)
# The surfaces the per-dataset figure actually draws: every surface minus the
# degenerate ones. The aggregate figure still sees the full SURFACES list.
PC_SURFACES = tuple(s for s in SURFACES if s not in DEGENERATE)
# Per-panel subtype: what KIND of text this corpus is, plus the stage that
# trained on it. A panel read on its own still says what it is, without the
# reader having to trace back up to the column header.
CORPUS_SUBTYPE = {
    ("chat", "lmsys23k"): "real-user chat · chat template",
    ("naturalistic", "lmsys23k"): "real-user chat · template stripped",
    ("chat", "sft11k"): "instruction demonstrations · SFT's own data",
    ("chat", "uf11k"): "preference prompts · DPO's own data",
    ("chat", "if11k"): "IFEval format constraints · RLVR's own data",
    ("chat", "math7500"): "competition math · RLVR's own data",
    ("chat", "gsm8k_train_full"): "grade-school math · RLVR's own data",
    ("chat", "gsm8k_test1319"): "grade-school math · held-out companion",
}
# Fail loud if a corpus is ever added/renamed without updating the grouping —
# otherwise it would silently vanish from the per-dataset figure.
assert set(CORPUS_SUBTYPE) == set(SURFACES), "CORPUS_SUBTYPE must cover every surface"
assert sorted(s for v in TYPE_COLUMNS.values() for s in v) == sorted(PC_SURFACES), (
    "TYPE_COLUMNS must partition the NON-DEGENERATE surfaces exactly — a corpus "
    "added or renamed without updating the grouping would silently vanish from "
    "the per-dataset figure, and a degenerate one re-added would silently reappear"
)
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

# --- display x-axis: one leading SELF column per source block ----------------
# Each source block opens with an S→S column holding the SOURCE stage's own
# within-model map (the "source self-map" series) so the self ceiling reads as
# its own column instead of a line overlaid across the transfer pairs (user
# call 2026-08-14). X_PAIR maps a PAIRS index to its display slot; X_SELF maps
# a source stage to its self column's slot; SLOT_BLOCKS is SOURCE_BLOCKS in
# display coordinates (self column included).
BLOCK_STAGE = {"base": "base", "SFT": "sft", "DPO": "dpo", "RLVR": "rlvr"}
X_SELF: dict[str, int] = {}
X_PAIR: list[int] = [0] * len(PAIRS)
SLOT_BLOCKS: list[tuple[str, int, int]] = []
_xcur = 0
for _name, _lo, _hi in SOURCE_BLOCKS:
    _blo = _xcur
    X_SELF[BLOCK_STAGE[_name]] = _xcur
    _xcur += 1
    for _pi in range(_lo, _hi):
        X_PAIR[_pi] = _xcur
        _xcur += 1
    SLOT_BLOCKS.append((_name, _blo, _xcur))
N_SLOTS = _xcur


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
# Round B (selfmap_v3) originally carried NEITHER. The shuffled null is
# pair-specific and stays a visible GAP on the three round-B pairs — never
# interpolated across a pair that was not controlled. The two ALIGNMENT-map
# identity baselines ARE measured there now (v4 refit), so those two series run
# continuously across all ten pairs; identity+bias-WITHIN is still borrowed
# from each target's round-3 siblings and flagged identity_approx.
# Retained for the sidecar prose only: the shuffled-pairing null is no longer
# drawn on either figure (user call 2026-08-12), so nothing plots in this colour.
NULL_COLOR = "#969696"
IDENT_COLOR = "#54278f"
BASELINE_PAIRS = tuple(p for p in PAIRS if p not in SELFMAP_PAIRS)
# nulls.order is [within, t0..t8]; the plotted tiers land at these column indices.
# Column index of each tier's per-draw shuffled-pairing null inside a pair
# file's `nulls.r2_matrix` (20 draws x 10 reads). The file carries its own
# `nulls.order` header; this map is ASSERTED against that header at load time
# rather than trusted, so a column reorder fails loud instead of silently
# plotting one tier's null under another tier's name.
NULL_ORDER = ("within", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")
NULL_COL = {t: NULL_ORDER.index(f"t{t}") for t in range(9)}


def _selfmap_align_baselines() -> dict:
    """(pair, fmt, corpus) -> (A_ctx_rev, A_ans) identity+bias, from round-B cells.

    Round B (selfmap_v3) originally ran NEITHER baseline. The v4 refit
    (ALGEBRA_VERSION `v4-foldlocal-2-cross-t7t8-alignbaselines`) added the two
    ALIGNMENT-map identity+bias reads using the parent metric_ladder's own
    estimator — `identity_bias_predict` OOF on the same seed-0
    conversation-grouped 5-fold, scored with the globally-pooled
    `fc._pooled_r2` — so these are directly comparable with the round-3
    pair-file values the other seven pairs supply.

    Returns {} when no refit cell carries the fields, which keeps a pre-v4
    checkout rendering the old visible NaN gap instead of failing.
    """
    out: dict[tuple, tuple[float, float]] = {}
    if not stt.SELFMAP_CELLS.is_dir():
        return out
    for fp in sorted(stt.SELFMAP_CELLS.glob("*.json")):
        for rec in json.load(open(fp))["records"]:
            actx = rec.get("A_ctx_rev_identity_bias_r2")
            aans = rec.get("A_ans_identity_bias_r2")
            if actx is None or aans is None:
                continue  # pre-v4 cell: no alignment baselines were computed
            out[(rec["pair"], rec["format"], rec["corpus"])] = (float(actx), float(aans))
    return out


_ALIGN_B = _selfmap_align_baselines()


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
    by_target: dict[tuple, list] = {}
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
        order = tuple(layer["nulls"]["order"])
        if order != NULL_ORDER:
            raise RuntimeError(
                f"{Path(fp).name}: nulls.order is {order}, expected {NULL_ORDER} — the "
                "shuffled-null column layout moved; fix NULL_ORDER rather than reindexing"
            )
        mat = np.asarray(layer["nulls"]["r2_matrix"], dtype=float)
        rec["null"].append(mat[:, [NULL_COL[t] for t in TIERS]])
        _bl = layer.get("baselines")
        _id = float(_bl["within"]["identity_bias_r2"]) if _bl else float("nan")
        rec["ident"].append(_id)
        # identity+bias is a WITHIN-model baseline keyed on the TARGET, so a
        # round-B pair (no pair file of its own) can borrow its target's value
        # from a round-3 sibling. Index it per (target, corpus) to do that.
        if _bl:
            by_target.setdefault((tgt, fmt, corpus), []).append(_id)
            # The two ALIGNMENT sub-maps the tiers use have their own identity
            # baselines, on DIFFERENT prediction problems (ctx->ctx, ans->ans).
            rec.setdefault("a_ctx", []).append(float(_bl["A_ctx_rev"]["identity_bias_r2"]))
            rec.setdefault("a_ans", []).append(float(_bl["A_ans"]["identity_bias_r2"]))

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
            # Round B ran NEITHER control. The shuffled null is pair-specific
            # (it permutes THIS pair's target rows and refits every correction)
            # so it stays a gap. identity+bias is target-keyed, so borrow the
            # target's measured value from its round-3 siblings and mark it
            # APPROXIMATE: the pools differ by the pair-specific shared-row
            # intersection (~2%; measured within-target spread median 0.015,
            # max 0.088 on a -2..-3.4 quantity).
            sib = [
                float(np.median(by_target[(tgt, f, c)]))
                for (f, c) in SURFACES
                if (f, c) not in DEGENERATE and (tgt, f, c) in by_target
            ]
            # The two ALIGNMENT identity baselines are no longer a gap here: the
            # v4 refit measured them ON THIS PAIR'S OWN ROWS, so unlike the
            # target-keyed identity+bias above they are neither grafted nor
            # approximate. Median over the same non-degenerate surfaces the
            # round-3 pairs aggregate across, so the two series are one basis.
            pkey = f"{src}__{tgt}"
            _ab = [
                _ALIGN_B[(pkey, f, c)]
                for (f, c) in SURFACES
                if (f, c) not in DEGENERATE and (pkey, f, c) in _ALIGN_B
            ]
            entry = {
                "null_lo": np.nan,
                "null_hi": np.nan,
                "null_per_tier": {int(t): np.nan for t in TIERS},
                "null_max_single_draw": np.nan,
                "a_ctx": float(np.median([v[0] for v in _ab])) if _ab else np.nan,
                "a_ans": float(np.median([v[1] for v in _ab])) if _ab else np.nan,
                "align_measured": bool(_ab),
                "n_corpora_align": len(_ab),
            }
            if sib:
                entry["identity_bias"] = float(np.median(sib))
                entry["identity_bias_range"] = [float(min(sib)), float(max(sib))]
                entry["identity_approx"] = True
                entry["n_corpora"] = len(sib)
            if sib or _ab:
                out[pi] = entry
            continue
        # per corpus: mean over the 20 draws per tier -> median across corpora
        per_tier = np.median(np.stack([m.mean(axis=0) for m in rec["null"]]), axis=0)
        out[pi] = {
            "null_lo": float(per_tier.min()),
            "null_hi": float(per_tier.max()),
            "null_per_tier": {int(t): float(v) for t, v in zip(TIERS, per_tier)},
            "null_max_single_draw": float(max(m.max() for m in rec["null"])),
            "identity_bias": float(np.median(rec["ident"])),
            "identity_bias_range": [float(min(rec["ident"])), float(max(rec["ident"]))],
            "identity_approx": False,
            "a_ctx": float(np.median(rec["a_ctx"])) if rec.get("a_ctx") else np.nan,
            "a_ans": float(np.median(rec["a_ans"])) if rec.get("a_ans") else np.nan,
            "n_corpora": len(rec["ident"]),
        }
    return out


BASELINES = load_baselines()

# --- source self-map (the SOURCE stage's own within-model ceiling) ----------
# The grey dashed ceiling is the TARGET's own map; this series is the SOURCE's
# own map on its own (context, answer) pairs — how linearly predictable the
# source's answer state is before any transfer. within_r2 is target-keyed in
# the pair files, so post-base stages read it from any pair file whose TARGET
# is the stage; base is never a forward-pair target, so its value comes from
# the round-B selfmap battery's base__base records (measured at layer 30 only —
# off-30 layers render the base-source pairs as a gap, never interpolated).
CEILING_LABEL = "within-model ceiling — target's own map (▲ = the S→S self column)"
# One x axis across both lattice figures: every column is an ordered pair of
# Tulu-3 checkpoints — the map is fit on the SOURCE and applied to the TARGET.
X_AXIS_LABEL = "source model → target model"


def load_source_self() -> dict:
    """(stage, fmt, corpus) -> that stage's own within-model R² at this layer."""
    out: dict[tuple, float] = {}
    for src, tgt in PAIRS:
        for fmt, corpus in SURFACES:
            if (tgt, fmt, corpus) in out:
                continue
            c = stt.cell(src, tgt, fmt, corpus, TIERS[0])
            if c is not None:
                out[(tgt, fmt, corpus)] = float(c["within_r2"])
    if LAYER == 30:
        rec_path = stt.SELFMAP_CELLS.parent / "records_l30.json"
        if rec_path.is_file():
            for rec in json.load(open(rec_path))["records"]:
                if rec.get("pair") == "base__base" and rec.get("within_r2") is not None:
                    out[("base", rec["format"], rec["corpus"])] = float(rec["within_r2"])
    return out


SOURCE_SELF = load_source_self()


def pair_label(src: str, tgt: str) -> str:
    short = {
        "base": "base",
        "sft": "SFT",
        "dpo": "DPO",
        "rlvr": "RLVR-PPO",
        "rlvr_long": "RLVR-GRPO",
    }
    return f"{short[src]}→{short[tgt]}"


def _xticks(ax) -> None:
    labels = [""] * N_SLOTS
    for stage, xs in X_SELF.items():
        labels[xs] = pair_label(stage, stage)
    for pi, (s, t) in enumerate(PAIRS):
        labels[X_PAIR[pi]] = pair_label(s, t)
    ax.set_xticks(range(N_SLOTS))
    # 14 slots (10 pairs + 4 self columns) no longer fit horizontally at this
    # width — rotate; the per-corpus figure re-rotates its bottom rows to 90.
    ax.set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
    ax.set_xlim(-0.5, N_SLOTS - 0.5)
    for _, lo, hi in SLOT_BLOCKS[:-1]:
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
                # The ONE sanctioned gap: the round-B pairs come from the
                # selfmap battery, which ran only t0/t6/t7/t8, so tiers 1-5 were
                # never measured for them. Record the absence and let the
                # drawing code leave a gap. Any OTHER empty cell set is a reaped
                # cache and still fails loud.
                if (src, tgt) in SELFMAP_PAIRS and tier in (1, 2, 3, 4, 5):
                    data[(pi, tier)] = []
                    continue
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


def _median_r2(D: dict, pi: int, tier: int) -> float:
    """Across-surface median R² for one (pair, tier); NaN when never measured.

    NaN is the ABSENCE marker, not a value: matplotlib breaks the line at NaN,
    which is how a tier the round-B battery never ran renders as a gap rather
    than a zero. Degenerate surfaces are excluded exactly as elsewhere.
    """
    vals = [c["r2"] for (s, c) in D["data"][(pi, tier)] if s not in DEGENERATE]
    return float(np.median(vals)) if vals else float("nan")


def fig_aggregate(D: dict) -> Path:
    if not HAS_IDENTITY:
        # No identity+bias at this layer (the ladder gates the baselines block to
        # full_tier_layers=[30]), so there is nothing off-scale to break for and
        # no alignment identity baselines to give their own panel.
        # add_axes with an explicit rect bypasses the style's layout engine
        # entirely (set_layout_engine("none") keeps the prior engine's
        # adjust_compatible flag, so subplots_adjust is silently skipped).
        fig = plt.figure(figsize=(12.4, 8.0))
        # bottom 0.235 -> 0.275 (height absorbs it): the rotated tick labels plus
        # the xlabel need clearance above the figure-level legend strip.
        ax = fig.add_axes([0.075, 0.275, 0.920, 0.635])
        axb = None
        axa = None
    elif not HAS_ALIGN_PANEL:
        # identity+bias still needs its broken lower panel; the alignment panel
        # is dropped (see HAS_ALIGN_PANEL). Same ax/axb coupling as the 3-panel
        # form, one row shorter.
        fig = plt.figure(figsize=(12.4, 8.8))
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[4.3, 1.0],
            hspace=0.08,
            left=0.075,
            right=0.995,
            top=0.912,
            # 0.175 -> 0.215: rotated tick labels + xlabel, clear of the legend.
            bottom=0.215,
        )
        ax = fig.add_subplot(gs[0])
        axa = None
        axb = fig.add_subplot(gs[1], sharex=ax)
    else:
        # Three stacked panels over ONE shared x (the ten pairs):
        #   axa — the two ALIGNMENT-map identity+bias baselines, on their OWN axis
        #   ax  — the tiers, the cross map, the within-model ceiling, the nulls
        #   axb — the within-model identity+bias baseline, far below scale
        # axa is separated by a SPACER row, not a break: it is not the same
        # quantity clipped, it is a DIFFERENT regression. Its post-training
        # values (0.68-0.96) sit above the within-model ceiling (~0.5), so on a
        # shared axis they read as "the baseline beats the self map" — they do
        # not; they score target-context->source-context alignment across two
        # checkpoints, while the ceiling scores context->answer inside one
        # (user question, 2026-08-11). ax/axb stay tightly coupled: those two
        # ARE one broken axis over one quantity.
        fig = plt.figure(figsize=(12.4, 9.7))
        # Row 1 gets NO axes — it is the spacer that separates axa from ax.
        # Geometry goes ON the GridSpec (see fig_percorpus): fig.subplots_adjust
        # does not reach a manually created gridspec, so the panels would keep
        # their default box and the call warns + no-ops. Same ABSOLUTE headroom
        # and legend strip as the old 2-panel form, re-expressed against the
        # taller 9.7in figure, with the old 2-panel form's oversized bottom
        # strip (2.44in for a 2-row legend) tightened to ~1.45in.
        gs = fig.add_gridspec(
            4,
            1,
            height_ratios=[1.55, 0.80, 4.3, 1.0],
            hspace=0.08,
            left=0.075,
            right=0.995,
            top=0.945,
            # 0.150 -> 0.185: rotated tick labels + xlabel, clear of the legend.
            bottom=0.185,
        )
        ax = fig.add_subplot(gs[2])
        axa = fig.add_subplot(gs[0], sharex=ax)
        axb = fig.add_subplot(gs[3], sharex=ax)
    # The project plot style enables an auto layout engine, which silently
    # overrides hspace / subplots_adjust and pulls the broken-axis panels apart.
    fig.set_layout_engine("none")
    x = np.asarray(X_PAIR)

    for a in [a for a in (axa, ax, axb) if a is not None]:
        for _, lo, hi in SLOT_BLOCKS[::2]:
            a.axvspan(lo - 0.5, hi - 0.5, color="#f7f7f7", zorder=0)

    # --- fair baselines, both as LINES, drawn BEHIND the tier lines ---
    # The zero line's reading rides the LEGEND, not an in-canvas text block: a
    # caption block rendered onto the figure is banned (paper-plots SKILL §3.8-bis,
    # standing user directive 2026-08-12) — every such fact belongs in the prose
    # beside the figure or in the .meta.json sidecar.
    ax.axhline(0.0, color="#252525", lw=1.0, ls=(0, (6, 3)), zorder=1)
    ax.plot(
        [],
        [],
        color="#252525",
        lw=1.0,
        ls=(0, (6, 3)),
        label="R² = 0 · predict the training mean",
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
    if axa is not None:
        axa.axhline(0.0, color="#252525", lw=0.9, ls=(0, (6, 3)), zorder=1)
        # Both series are MEASURED on every pair's OWN rows — the round-B pairs by
        # the v4 refit — so the provenance rides the label and the marker stays
        # FILLED throughout (contrast the identity+bias series below, whose three
        # round-B points are BORROWED and drawn hollow).
        # LABEL PRECISION: these series are the identity+bias BASELINE of each
        # alignment regression — "how well does copying the vector across the two
        # checkpoints work, with only a mean shift?" — NOT the fitted alignment
        # map's own R². The two differ enormously across the base boundary
        # (base→dpo: fitted A_ctx 0.589 vs identity 0.263 BELOW zero) and barely
        # at all among post-trained pairs (dpo→rlvr: 0.980 vs 0.976). Calling the
        # series "the alignment map" invites reading the second number as the
        # first, so the label says baseline explicitly.
        for key, col, lab in (
            (
                "a_ctx",
                "#238b45",
                "identity+bias baseline of the CONTEXT alignment map $A_{ctx}$ (t6's remap)",
            ),
            (
                "a_ans",
                "#d95f0e",
                "identity+bias baseline of the ANSWER alignment map $A_{ans}$ (t7's remap)",
            ),
        ):
            axa.plot(
                x,
                [BASELINES.get(pi, {}).get(key, np.nan) for pi in range(len(PAIRS))],
                color=col,
                lw=1.5,
                ls=(0, (1, 1.6)),
                marker="^",
                ms=5,
                zorder=3,
                label=lab,
            )
        axa.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    # The shuffled-pairing null is NOT DRAWN (user call 2026-08-12: "remove the
    # shuffled baseline just put the identity + bias"). Its per-tier values are
    # still computed and still ride fair_baselines.per_pair in the .meta.json
    # sidecar, and the prose argument that uses them — base→DPO's -0.084 at t0
    # against its own -0.712 matched null — is unchanged. Only the canvas drops
    # the four dashed lines, which crowded the lower half of the panel.
    #
    # identity+bias lives on the lower (broken-axis) panel at its TRUE value;
    # the proxy handle carries it into the main panel's legend. ONE uniform
    # series: the measured-vs-borrowed marker split is retired (same user call),
    # so provenance now lives only in the sidecar's per-pair identity_approx
    # flags and in the prose beside the figure.
    ident = np.array(
        [BASELINES.get(pi, {}).get("identity_bias", np.nan) for pi in range(len(PAIRS))]
    )
    ident_line = dict(color=IDENT_COLOR, lw=2.0, ls=(0, (5, 2)), marker="o", ms=5.5)
    if axb is not None:
        axb.plot(x, ident, zorder=3, **ident_line)
    if HAS_IDENTITY:
        ax.plot([], [], label="identity+bias  ŷ = x + b  (lower panel)", **ident_line)

    for tier in TIERS:
        med = [_median_r2(D, pi, tier) for pi in range(len(PAIRS))]
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
                        [X_PAIR[pi]],
                        [A_YLIM[0]],
                        marker="v",
                        ms=5,
                        color=TIER_COLORS[tier],
                        clip_on=False,
                    )
                else:
                    ax.plot(
                        [X_PAIR[pi]], [y], marker="o", ms=3.2, alpha=0.35, color=TIER_COLORS[tier]
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
    # ONE grey series across EVERY column: the target's own map. At a self
    # column the target IS the source, so that stage's own map is the same
    # quantity and joins THIS line rather than forming a second one (user call
    # 2026-08-14) — the ceiling and the source self-map were never two
    # different measurements. A stage with no measured self map (off-30 layers:
    # base's selfmap battery ran at layer 30 only) is absent from the slot
    # list, so the line breaks there instead of interpolating.
    ceil_by_slot: dict[int, float] = {X_PAIR[pi]: D["ceilings"][pi] for pi in range(len(PAIRS))}
    self_slots: list[int] = []
    for stage, xs in sorted(X_SELF.items(), key=lambda kv: kv[1]):
        vals = [
            SOURCE_SELF[(stage, f, c)]
            for (f, c) in SURFACES
            if (f, c) not in DEGENERATE and (stage, f, c) in SOURCE_SELF
        ]
        if not vals:
            continue
        ceil_by_slot[xs] = float(np.median(vals))
        self_slots.append(xs)
    _cs = sorted(ceil_by_slot)
    ax.plot(
        _cs,
        [ceil_by_slot[s] for s in _cs],
        marker="s",
        ms=5,
        lw=1.5,
        ls="--",
        color=CEILING_COLOR,
        label=CEILING_LABEL,
        zorder=6,
    )
    # Same colour, same line — the triangle only marks WHICH columns are self
    # columns, so the legend stays a single entry.
    ax.plot(
        self_slots,
        [ceil_by_slot[s] for s in self_slots],
        marker="^",
        ms=7.5,
        ls="none",
        color=CEILING_COLOR,
        zorder=7,
    )

    for name, lo, hi in SLOT_BLOCKS:
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
    if axa is not None:
        axa.set_ylim(*A_YLIM_ALIGN)
        axa.set_yticks([-0.5, 0.0, 0.5, 1.0])
        axa.set_ylabel("held-out R²\n(alignment maps)", fontsize=9.5)
        # ONE-LINE panel title. The full "why this panel is separate" explanation
        # (a_ctx/a_ans score checkpoint→checkpoint alignment, not context→answer,
        # so a high value here is NOT "the baseline beats the self map") lives in
        # the prose beside the figure + fair_baselines.alignment_panel in the
        # sidecar — never as a caption block on the canvas (SKILL §3.8-bis).
        axa.set_title(
            "Do the two checkpoints even share coordinates?  Identity+bias baseline of each "
            "alignment map — a DIFFERENT regression, not comparable to the panel below",
            fontsize=9.5,
            loc="left",
            color="#404040",
        )
    ax.set_ylabel(f"held-out R²  (raw pooled, layer {LAYER})", fontsize=11)
    ax.set_title(
        "Every forward pair of the Tülu-3 ladder: how much of the context→answer map survives\n"
        "median over 7 non-degenerate corpora, every corpus overplotted",
        fontsize=11.5,
        loc="left",
    )
    for a in [a for a in (axa, ax, axb) if a is not None]:
        a.grid(axis="y", alpha=0.22, lw=0.6)
        _xticks(a)
    if axb is not None:
        _break_marks(ax, axb)
    if axa is not None:
        # axb carries the shared tick labels; axa is a separate panel, not a
        # break partner, so it keeps its own bottom spine and just drops labels.
        axa.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # xlabel on the BOTTOM-most axis (axb when the identity+bias break panel is
    # present — _break_marks has already stripped ax's own tick labels).
    (axb if axb is not None else ax).set_xlabel(X_AXIS_LABEL, fontsize=10.5, labelpad=8)
    # Figure-level legend BELOW the axes: the per-tier t0/t6 null lines now occupy
    # the lower-left/mid of the data area, so an in-axes legend overlaps them.
    h, lab = ax.get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4, fontsize=8.5, frameon=False
    )
    # subplots_adjust, not tight_layout: tight_layout recomputes hspace and
    # would pull the two broken-axis panels apart.
    # (3-panel form: the geometry already rode the GridSpec at construction.)
    out = OUTDIR / f"ladder_full_transfer_lattice{SUFFIX}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _panel_baselines(fmt: str, corpus: str) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """This CORPUS's baselines, per pair: ({tier -> null}, identity+bias, borrowed-mask).

    The null is returned PER TIER (20-draw mean), not collapsed: t7/t8 refit the
    answer side against the shuffled targets so their null sits at ~0, while
    t0/t6 have no answer-side refit to absorb the target mean and sit at
    -0.3..-0.75. Collapsing them holds the t7/t8 bar up against the t0/t6
    series and makes real t0 signal read as failure. The null stays NaN at the
    three round-B pairs (pair-specific, never run there) so each null LINE
    breaks into a visible gap instead of bridging an uncontrolled pair.

    identity+bias is NOT pair-specific: it is a WITHIN-model baseline keyed on
    the TARGET, so a round-B pair borrows its target's value from that target's
    round-3 siblings AT THIS SAME (fmt, corpus) — the per-panel form of the
    aggregate figure's borrow, which previously left these panels blank while
    the aggregate showed a value. The third return marks which entries are
    borrowed so the panel draws them HOLLOW instead of passing them off as
    measured on the pair's own rows.
    """
    import glob
    import re

    pat = re.compile(r"pair_(.+?)__(.+?)_(chat|naturalistic)_(.+)\.json")
    found: dict[str, tuple[np.ndarray, float]] = {}
    by_target: dict[str, list[float]] = {}
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
            _id = float(_bl["within"]["identity_bias_r2"]) if _bl else float("nan")
            found[f"{src}__{tgt}"] = (mat[:, [NULL_COL[t] for t in TIERS]].mean(axis=0), _id)
            if _bl:
                by_target.setdefault(tgt, []).append(_id)
    per_tier: dict[int, list[float]] = {t: [] for t in TIERS}
    identv: list[float] = []
    borrowed: list[bool] = []
    for src, tgt in PAIRS:
        v = found.get(f"{src}__{tgt}")
        for i, t in enumerate(TIERS):
            per_tier[t].append(np.nan if v is None else float(v[0][i]))
        if v is not None:
            identv.append(v[1])
            borrowed.append(False)
            continue
        sib = by_target.get(tgt, [])
        identv.append(float(np.median(sib)) if sib else np.nan)
        borrowed.append(bool(sib))
    return (
        {t: np.array(vs) for t, vs in per_tier.items()},
        np.array(identv, dtype=float),
        np.array(borrowed, dtype=bool),
    )


def fig_percorpus(D: dict) -> Path:
    # Each corpus gets a BROKEN y-axis: results band on the main panel, the
    # identity+bias line on a short lower panel (spacer row 2 separates the
    # two blocks so a break row never crowds the next block's titles).
    # ONE COLUMN PER OWNING STAGE (user call 2026-08-12), replacing the old
    # fill-left-to-right 2x4 grid where a corpus's neighbours meant nothing.
    # Columns are ragged — a stage owning fewer corpora leaves empty cells, the
    # honest rendering of "one stage per column" when the stages own unequal
    # numbers of corpora. Column order is ladder order, stage-agnostic first.
    n_rows = max(len(v) for v in TYPE_COLUMNS.values())
    n_cols = len(TYPE_ORDER)
    fig = plt.figure(figsize=(16.0, 14.6))
    fig.set_layout_engine("none")  # see fig_aggregate: the style's engine wins otherwise
    # Geometry goes ON the GridSpec: fig.subplots_adjust does not reach a
    # manually created gridspec, so the panels would keep their default box.
    # Each plot-row is a (main, break) pair; a spacer row separates plot-rows so a
    # break row never crowds the next row's titles.
    height_ratios: list[float] = []
    for r in range(n_rows):
        height_ratios += [3.5, 0.9] + ([0.42] if r < n_rows - 1 else [])
    gs = fig.add_gridspec(
        len(height_ratios),
        n_cols,
        height_ratios=height_ratios,
        hspace=0.11,
        wspace=0.13,
        left=0.045,
        right=0.995,
        top=0.915,
        # Was 0.055 (0.80in): the tallest column's 90-rotated tick labels alone
        # nearly fill that, so the per-column xlabel needs the strip widened.
        bottom=0.085,
    )
    # surface -> (main ax, break ax); built column-major so ragged columns work.
    axes_for: dict[tuple[str, str], tuple] = {}
    col_of: dict[tuple[str, str], int] = {}
    last_in_col: dict[int, tuple[str, str]] = {}
    first_main = None
    first_break = None
    for c, tname in enumerate(TYPE_ORDER):
        for r, surf in enumerate(TYPE_COLUMNS[tname]):
            r_main = r * 3  # main, break, spacer
            m = fig.add_subplot(gs[r_main, c], sharex=first_main, sharey=first_main)
            b = fig.add_subplot(gs[r_main + 1, c], sharex=m, sharey=first_break)
            first_main = first_main or m
            first_break = first_break or b
            axes_for[surf] = (m, b)
            col_of[surf] = c
            last_in_col[c] = surf
    # PC_SURFACES order is preserved for every data read; only placement changes.
    mains = [axes_for[s][0] for s in PC_SURFACES]
    breaks = [axes_for[s][1] for s in PC_SURFACES]
    # Column headers sit above each column's first panel, in axes-free figure
    # space, so a type is readable without decoding the panel titles.
    for c, tname in enumerate(TYPE_ORDER):
        top_ax = axes_for[TYPE_COLUMNS[tname][0]][0]
        pos = top_ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            pos.y1 + 0.028,
            tname.upper(),
            ha="center",
            va="bottom",
            fontsize=11.5,
            color="#252525",
            fontweight="bold",
        )

    # FILLED = measured on this pair's own rows; HOLLOW = borrowed from the
    # target's round-3 siblings at this same (fmt, corpus). Same encoding as the
    # aggregate figure, so one marker convention reads across both.
    pc_ident_line = dict(color=IDENT_COLOR, lw=1.7, ls=(0, (5, 2)), marker="o", ms=4)
    for k, (fmt, corpus) in enumerate(PC_SURFACES):
        ax, axb = mains[k], breaks[k]
        deg = (fmt, corpus) in DEGENERATE
        # identity+bias only, per panel, behind the tier lines — this CORPUS's own
        # value. The shuffled-pairing null is NOT DRAWN and the measured-vs-borrowed
        # marker split is retired (user call 2026-08-12); see fig_aggregate. Both
        # facts survive in the .meta.json sidecar and the prose.
        ax.axhline(0.0, color="#252525", lw=0.9, ls=(0, (6, 3)), zorder=1)
        _, pident, _ = _panel_baselines(fmt, corpus)
        if HAS_IDENTITY:
            axb.plot(np.asarray(X_PAIR), pident, zorder=3, **pc_ident_line)
        if k == 0 and HAS_IDENTITY:  # proxy so the figure legend carries it
            ax.plot([], [], label="identity+bias  ŷ = x + b  (lower panel)", **pc_ident_line)
        for tier in TIERS:
            xa, ys, lo, hi = [], [], [], []
            for pi, (src, tgt) in enumerate(PAIRS):
                c = stt.cell(src, tgt, fmt, corpus, tier)
                # An unmeasured cell is kept as a NaN at its own x, never
                # SKIPPED: skipping closes the gap and errorbar then draws a
                # segment straight across the x-positions where the tier was
                # never run (the round-B pairs at tiers 1-5), which reads as an
                # interpolated value. NaN breaks the line instead — the
                # "absent, not zeroed, not interpolated" contract this figure
                # states. NaN < ylim is False, so the floor-caret loop skips it.
                xa.append(X_PAIR[pi])
                ys.append(float("nan") if c is None else c["r2"])
                if c is not None and c["has_ci"]:
                    lo.append(c["r2"] - c["r2_lo"])
                    hi.append(c["r2_hi"] - c["r2"])
                else:
                    lo.append(0.0)
                    hi.append(0.0)
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

        xa = [pi for pi, (s, t) in enumerate(PAIRS) if (f"{s}__{t}", fmt, corpus) in CROSS]
        ax.plot(
            [X_PAIR[pi] for pi in xa],
            [CROSS[(f"{PAIRS[pi][0]}__{PAIRS[pi][1]}", fmt, corpus)] for pi in xa],
            marker="D",
            ms=5,
            lw=1.8,
            ls="--",
            color=CROSS_COLOR,
            label=xmap.CROSS_LABEL,
        )
        # ONE grey series, self columns included — see fig_aggregate: a self
        # column's target IS its source, so the stage's own map is the same
        # quantity as the ceiling and joins the same line.
        cvals = [stt.cell(s, t, fmt, corpus, TIERS[0]) for s, t in PAIRS]
        cbs: dict[int, float] = {
            X_PAIR[i]: c["within_r2"] for i, c in enumerate(cvals) if c is not None
        }
        pself = [xs for stage, xs in X_SELF.items() if (stage, fmt, corpus) in SOURCE_SELF]
        for stage, xs in X_SELF.items():
            if (stage, fmt, corpus) in SOURCE_SELF:
                cbs[xs] = SOURCE_SELF[(stage, fmt, corpus)]
        _pcs = sorted(cbs)
        ax.plot(
            _pcs,
            [cbs[s] for s in _pcs],
            marker="s",
            ms=4,
            ls="--",
            lw=1.3,
            color=CEILING_COLOR,
            label=CEILING_LABEL,
        )
        ax.plot(
            sorted(pself),
            [cbs[s] for s in sorted(pself)],
            marker="^",
            ms=5.5,
            ls="none",
            color=CEILING_COLOR,
        )

        title = f"{corpus} ({fmt})" + ("  — DEGENERATE n_train<d" if deg else "")
        # Subtype STACKED under the corpus name, not right-aligned on the same
        # line: a long corpus name (gsm8k_train_full, and gsm8k_test1319 with its
        # DEGENERATE tag) overruns the panel width and the two collide.
        ax.set_title(title, fontsize=9.5, loc="left", pad=14, color="#8c2d04" if deg else "black")
        ax.text(
            0.0,
            1.012,
            CORPUS_SUBTYPE[(fmt, corpus)],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#707070",
        )
        for a in (ax, axb):
            if deg:
                a.set_facecolor("#fdf2e9")
            a.grid(axis="y", alpha=0.22, lw=0.6)
            _xticks(a)
        axb.tick_params(axis="x", labelrotation=90, labelsize=7)
        _break_marks(ax, axb)
        # Each COLUMN labels its own LAST panel: the columns are ragged, so a
        # single shared bottom row would leave the short columns unlabelled.
        # Same reason the xlabel is per-column rather than one fig.supxlabel:
        # three of the four columns end well above the figure bottom.
        if (fmt, corpus) == last_in_col[col_of[(fmt, corpus)]]:
            axb.set_xlabel(X_AXIS_LABEL, fontsize=9, labelpad=6)
        else:
            axb.tick_params(axis="x", labelbottom=False)

    mains[0].set_ylim(*PC_YLIM)  # sharey => applies to all 8 main panels
    breaks[0].set_ylim(*PC_YLIM_IDENT)
    breaks[0].set_yticks([-1.0, -2.0, -3.0])
    # y label on the leftmost panel of every plot-row, not on a fixed index —
    # under the ragged type columns, index 0/4 no longer means "left edge".
    for surf in TYPE_COLUMNS[TYPE_ORDER[0]]:
        axes_for[surf][0].set_ylabel("held-out R²", fontsize=10)
    # The legend lives in the empty cells beneath the SHORT columns: the top
    # strip now belongs to the column headers, and the ragged grid leaves real
    # estate there rather than wasting it.
    # Vertically CENTERED in the empty block, not pinned to its top: the column
    # above ends with x tick labels, which a top-anchored legend lands on top of.
    # Column and ROW both chosen by SHAPE, never hardcoded: pick the column with
    # the MOST free slots (ties -> leftmost) and anchor from ITS first free main
    # row. The row index is k*3 for a column holding k panels, mirroring the
    # r_main = r * 3 stride above — a hardcoded row silently lands the legend ON
    # a panel the moment the tallest column grows (which is exactly what the
    # stage regrouping did: RLVR's column went to 3 panels, so the old fixed
    # row 3 became lmsys23k (naturalistic)'s own main row).
    _free = [(n_rows - len(TYPE_COLUMNS[t]), -i, i) for i, t in enumerate(TYPE_ORDER)]
    _slots, _, _lc = max(_free)
    if _slots <= 0:
        raise RuntimeError(
            "no column has a free slot: every column is as tall as the grid, so "
            "there is no empty block for the legend — place it explicitly"
        )
    _r_free = len(TYPE_COLUMNS[TYPE_ORDER[_lc]]) * 3
    assert _r_free < len(height_ratios), (
        f"legend row {_r_free} is past the grid ({len(height_ratios)} rows) — the "
        "r_main stride and this derivation have diverged"
    )
    _top = gs[_r_free, _lc].get_position(fig)
    _bot = gs[-1, _lc].get_position(fig)
    h, lab = mains[0].get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="center left",
        bbox_to_anchor=(_top.x0, (_top.y1 + _bot.y0) / 2),
        ncol=1,
        fontsize=9.5,
        frameon=False,
    )
    fig.suptitle(
        "Per eval dataset, grouped by owning stage: every forward stage pair, at each "
        "reparameterization tier",
        fontsize=12.5,
        x=0.01,
        y=0.998,
        ha="left",
        va="top",
    )
    out = OUTDIR / f"ladder_full_transfer_lattice_by_dataset{SUFFIX}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# --- used-between vs not-used-between (user call 2026-08-12) ----------------
# A PAIR has no training of its own: what varies is whether the corpus was used
# by the training that sits BETWEEN the pair's two models. `used-between` is the
# shorthand throughout; `touched` is the same predicate in code.
# Ladder order, and the corpus family each STEP of the ladder trains on. The
# step INTO a stage is what consumes that stage's mix, so the map is keyed on
# the arrival stage. rlvr_long re-trains on the RLVR mix, introducing no new
# corpus — which is why it maps to "rlvr" rather than a family of its own.
STAGE_INDEX = {"base": 0, "sft": 1, "dpo": 2, "rlvr": 3, "rlvr_long": 4}
STEP_TRAINS_ON = {1: "sft", 2: "dpo", 3: "rlvr", 4: "rlvr"}


def touched_families(src: str, tgt: str) -> set[str]:
    """Corpus families the training BETWEEN src and tgt consumed.

    Every step strictly after src through tgt inclusive, so a non-adjacent pair
    accumulates the union (base→RLVR passed through SFT and DPO training too).
    """
    return {STEP_TRAINS_ON[k] for k in range(STAGE_INDEX[src] + 1, STAGE_INDEX[tgt] + 1)}


def _deficits(D: dict, pi: int, tier: int) -> tuple[list, list]:
    """(trained_on, not_trained_on) as [(surface, ceiling - r2)], degenerates out.

    The DV is the deficit against each corpus's OWN within-model ceiling, not
    raw R². Within a fixed pair the comparison is across CORPORA, so raw R²
    would read corpus difficulty (math sits far below chat at every tier) as a
    used-between effect; the ceiling is the best a map fit WITHIN one model
    achieves on that same corpus, so dividing it out is what isolates transfer.
    """
    fams = touched_families(*PAIRS[pi])
    hit: list = []
    miss: list = []
    for surf, c in D["data"][(pi, tier)]:
        if surf in DEGENERATE:
            continue
        row = (surf, c["within_r2"] - c["r2"])
        (hit if OWNING_STAGE[surf] in fams else miss).append(row)
    return hit, miss


def _corpus_label(surf: tuple[str, str]) -> str:
    fmt, corpus = surf
    return corpus if fmt == "chat" else f"{corpus} ({fmt})"


def _touched_effect(D: dict, tier: int) -> dict | None:
    """Trained-on effect on the deficit, naive vs controlled for pair AND corpus.

    The naive gap is confounded in BOTH directions and the two confounds do not
    cancel. Across corpora at a fixed pair, the touched set fills up with the
    math corpora, whose transfer deficit is large for reasons that have nothing
    to do with training (base→RLVR's touched set is everything except lmsys).
    Across pairs at a fixed corpus, a corpus is touched exactly by the pairs
    that span its own stage — for sft11k that is every base→* pair, the hardest
    transfers on the board. So the marginal comparison reads corpus difficulty
    one way and pair difficulty the other.

    The controlled estimate is the `touched` coefficient of
    ``deficit ~ pair dummies + corpus dummies + touched``: both main effects
    are absorbed and what is left is the interaction the question asks about.
    SEs are cluster-robust (CR1) by PAIR — cells sharing a model pair are not
    independent draws. With only 10 clusters CR1 runs anti-conservative, so the
    intervals are a floor on the true width, not a tight bound.
    """
    rows = []
    for pi, (src, tgt) in enumerate(PAIRS):
        fams = touched_families(src, tgt)
        for surf, c in D["data"][(pi, tier)]:
            if surf in DEGENERATE:
                continue
            rows.append((pi, surf, int(OWNING_STAGE[surf] in fams), c["within_r2"] - c["r2"]))
    hit = [d for _, _, t, d in rows if t]
    miss = [d for _, _, t, d in rows if not t]
    if not hit or not miss:
        return None

    pis = sorted({r[0] for r in rows})
    surfs = sorted({r[1] for r in rows})
    X = np.array(
        [
            [1.0]
            + [1.0 if pi == p else 0.0 for p in pis[1:]]
            + [1.0 if sf == s2 else 0.0 for s2 in surfs[1:]]
            + [float(tou)]
            for pi, sf, tou, _ in rows
        ]
    )
    y = np.array([r[3] for r in rows])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for p in pis:
        idx = [i for i, r in enumerate(rows) if r[0] == p]
        g = X[idx].T @ u[idx]
        meat += np.outer(g, g)
    n_g = len(pis)
    scale = (n_g / (n_g - 1)) * ((len(y) - 1) / max(len(y) - rank, 1))
    se = float(np.sqrt(max((bread @ meat @ bread)[-1, -1] * scale, 0.0)))
    return {
        "naive": float(np.mean(hit) - np.mean(miss)),
        "coef": float(beta[-1]),
        "se": se,
        "n": len(y),
        "n_touched": len(hit),
        "n_pairs": n_g,
    }


def _corpus_label(surf: tuple[str, str]) -> str:
    fmt, corpus = surf
    return corpus if fmt == "chat" else f"{corpus}\n({fmt})"


def fig_trained_on(D: dict) -> Path:
    """Transfer on data the intervening training used vs data it never saw."""
    left_tier = 0 if 0 in TIERS else TIERS[0]
    c_hit, c_miss = "#D62728", "#8C8C8C"
    order = [s for col in TYPE_ORDER for s in TYPE_COLUMNS[col]]

    fig = plt.figure(figsize=(15.4, 6.1))
    fig.set_layout_engine("none")  # see fig_aggregate: the style engine wins otherwise
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.45, 1.0],
        wspace=0.20,
        left=0.052,
        right=0.985,
        top=0.845,
        bottom=0.150,
    )
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # LEFT: corpus identity held FIXED inside each x slot, so the red-vs-grey
    # comparison there is free of the corpus-difficulty confound (which is what
    # drives the whole naive effect). Each point is one stage pair.
    rng = np.random.default_rng(1336)
    for x, surf in enumerate(order):
        groups = {1: [], 0: []}
        for pi, (src, tgt) in enumerate(PAIRS):
            fams = touched_families(src, tgt)
            for s2, c in D["data"][(pi, left_tier)]:
                if s2 != surf:
                    continue
                groups[int(OWNING_STAGE[surf] in fams)].append(c["within_r2"] - c["r2"])
        meds = {}
        for tou, col, dx in ((0, c_miss, -0.18), (1, c_hit, 0.18)):
            ys = groups[tou]
            if not ys:
                continue
            meds[tou] = float(np.median(ys))
            axL.scatter(
                x + dx + rng.uniform(-0.06, 0.06, size=len(ys)),
                ys,
                s=34,
                color=col,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            axL.plot(
                [x + dx - 0.135, x + dx + 0.135],
                [meds[tou]] * 2,
                color=col,
                lw=2.6,
                zorder=5,
                solid_capstyle="butt",
            )
        if len(meds) == 2:  # direction line: grey median -> red median
            axL.annotate(
                "",
                xy=(x + 0.18, meds[1]),
                xytext=(x - 0.18, meds[0]),
                arrowprops=dict(arrowstyle="->", color="#404040", lw=1.2, shrinkA=2, shrinkB=2),
                zorder=4,
            )
    axL.set_xticks(range(len(order)))
    axL.set_xticklabels([_corpus_label(s) for s in order], fontsize=8.4)
    axL.set_xlim(-0.55, len(order) - 0.45)
    for col in TYPE_ORDER[:-1]:
        axL.axvline(
            sum(len(TYPE_COLUMNS[c]) for c in TYPE_ORDER[: TYPE_ORDER.index(col) + 1]) - 0.5,
            color="#cfcfcf",
            lw=0.9,
            ls=":",
            zorder=0,
        )
    axL.set_ylabel(f"transfer deficit  (ceiling − R²)   ·   tier {left_tier}", fontsize=10)
    axL.set_title(
        f"Within each corpus: {TIER_LABEL[left_tier]}, one point per model pair",
        fontsize=10.5,
        pad=7,
    )
    axL.axhline(0.0, color="#252525", lw=0.9, ls="--", zorder=1)
    axL.grid(axis="y", color="#ececec", lw=0.7, zorder=0)
    axL.set_axisbelow(True)
    axL.scatter(
        [], [], s=40, color=c_hit, label="corpus WAS used by the training between the two models"
    )
    axL.scatter([], [], s=40, color=c_miss, label="corpus was NOT used between them")
    axL.legend(fontsize=9, frameon=False, loc="upper left")

    # RIGHT: the verdict. The naive marginal gap and the pair+corpus-controlled
    # coefficient, per tier, on one axis — the reversal IS the result.
    tiers = [t for t in TIERS if _touched_effect(D, t)]
    ypos = list(range(len(tiers)))[::-1]
    for y, tier in zip(ypos, tiers):
        e = _touched_effect(D, tier)
        axR.plot(
            [e["coef"] - 1.96 * e["se"], e["coef"] + 1.96 * e["se"]],
            [y, y],
            color=TIER_COLORS[tier],
            lw=2.2,
            solid_capstyle="round",
            zorder=3,
        )
        axR.scatter(
            [e["coef"]],
            [y],
            s=74,
            color=TIER_COLORS[tier],
            zorder=4,
            edgecolor="white",
            linewidth=0.8,
        )
        axR.scatter(
            [e["naive"]],
            [y],
            s=74,
            facecolor="none",
            zorder=4,
            edgecolor=TIER_COLORS[tier],
            linewidth=1.7,
        )
        axR.annotate(
            "",
            xy=(e["coef"], y),
            xytext=(e["naive"], y),
            arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=1.0, shrinkA=6, shrinkB=6),
            zorder=2,
        )
    axR.axvline(0.0, color="#252525", lw=1.0, ls="--", zorder=1)
    axR.set_yticks(ypos)
    axR.set_yticklabels([f"{t}: {TIER_LABEL[t]}" for t in tiers], fontsize=9.5)
    axR.set_ylim(-0.6, len(tiers) - 0.4)
    axR.set_xlabel("deficit gap:  used-between the two models  −  not-used-between", fontsize=10)
    axR.set_title("Naive gap → gap with pair and corpus effects removed", fontsize=10.5, pad=7)
    axR.grid(axis="x", color="#ececec", lw=0.7, zorder=0)
    axR.set_axisbelow(True)
    axR.scatter(
        [],
        [],
        s=70,
        facecolor="none",
        edgecolor="#555555",
        linewidth=1.7,
        label="naive (across all cells)",
    )
    axR.scatter([], [], s=70, color="#555555", label="controlled, 95% CI (CR1 by pair)")
    axR.legend(fontsize=8.8, frameon=False, loc="lower right")

    fig.suptitle(
        "Does the training BETWEEN two models disrupt the context→answer map more on the "
        "data that training used?",
        fontsize=12.5,
        y=0.955,
    )
    out = OUTDIR / f"ladder_trained_on_vs_not{SUFFIX}.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _arm_rows(D: dict, arm) -> list:
    """(pair_index, surface, used_between, deficit) cells for one transfer arm.

    ``arm`` is a ladder tier index. The deficit is always measured against the
    SAME per-cell ceiling (the target stage's own within-model map), so the
    arms are directly comparable. (The fresh source-context -> target-answer
    fit rode here as a "cross" arm until 2026-08-12; the user dropped it from
    this figure, so the ladder tiers are now the only arms.)
    """
    rows = []
    for pi, (src, tgt) in enumerate(PAIRS):
        fams = touched_families(src, tgt)
        ceil_of = {s: c["within_r2"] for s, c in D["data"][(pi, arm)]}
        for surf, r2 in ((s, c["r2"]) for s, c in D["data"][(pi, arm)]):
            if surf in DEGENERATE:
                continue
            rows.append((pi, surf, int(OWNING_STAGE[surf] in fams), ceil_of[surf] - r2))
    return rows


def _controlled_gap(D: dict, arm) -> dict | None:
    """`used_between` coefficient of deficit ~ pair + corpus + used_between.

    Both main effects are absorbed by dummies, so the coefficient is the part
    of the gap-to-ceiling attributable to the corpus having been used by the
    training BETWEEN the pair's two models. CR1 SEs clustered by pair.
    """
    rows = _arm_rows(D, arm)
    if not rows or not any(r[2] for r in rows) or all(r[2] for r in rows):
        return None
    pis = sorted({r[0] for r in rows})
    surfs = sorted({r[1] for r in rows})
    X = np.array(
        [
            [1.0]
            + [1.0 if pi == p else 0.0 for p in pis[1:]]
            + [1.0 if sf == s2 else 0.0 for s2 in surfs[1:]]
            + [float(tou)]
            for pi, sf, tou, _ in rows
        ]
    )
    y = np.array([r[3] for r in rows])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for p in pis:
        idx = [i for i, r in enumerate(rows) if r[0] == p]
        g = X[idx].T @ u[idx]
        meat += np.outer(g, g)
    n_g = len(pis)
    scale = (n_g / (n_g - 1)) * ((len(y) - 1) / max(len(y) - rank, 1))
    return {
        "coef": float(beta[-1]),
        "se": float(np.sqrt(max((bread @ meat @ bread)[-1, -1] * scale, 0.0))),
        "n": len(y),
        "n_pairs": n_g,
    }


def fig_controlled_gap(D: dict) -> Path:
    """The controlled used-between gap alone, one row per transfer arm."""
    arms = [(t, f"{t}: {TIER_LABEL[t]}", TIER_COLORS[t]) for t in TIERS]
    rows = [(a, lab, col, _controlled_gap(D, a)) for a, lab, col in arms]
    rows = [r for r in rows if r[3]]

    fig = plt.figure(figsize=(9.6, 0.72 * len(rows) + 2.5))
    fig.set_layout_engine("none")  # see fig_aggregate: the style engine wins otherwise
    ax = fig.add_axes((0.30, 0.175, 0.675, 0.66))
    ypos = list(range(len(rows)))[::-1]
    for y, (_, _, col, e) in zip(ypos, rows):
        ax.plot(
            [e["coef"] - 1.96 * e["se"], e["coef"] + 1.96 * e["se"]],
            [y, y],
            color=col,
            lw=2.4,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter([e["coef"]], [y], s=86, color=col, zorder=4, edgecolor="white", linewidth=0.9)
    ax.axvline(0.0, color="#252525", lw=1.0, ls="--", zorder=1)
    # SYMLOG about zero: t0's interval is ~40x wider than every correction
    # arm's, so a shared linear axis renders four of the five rows as a dot on
    # the zero line. Linear inside +-0.05 (where those four live, ticks below),
    # log outside — nothing is clipped and the sign structure stays readable.
    ax.set_xscale("symlog", linthresh=0.05, linscale=1.6)
    ax.set_xticks([-1.0, -0.3, -0.05, -0.025, 0.0, 0.025, 0.05, 0.3])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.axvspan(-0.05, 0.05, color="#f4f4f4", zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab, _, _ in rows], fontsize=10)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel(
        "gap to ceiling:  corpus used between the two models  −  corpus not used\n"
        "(> 0 = worse on data the intervening training used; shaded = linear zone, symlog outside)",
        fontsize=10,
    )
    ax.grid(axis="x", color="#ececec", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    n = rows[0][3]["n"]
    fig.suptitle(
        "Is the context→answer map worse on data the training in between actually used?\n"
        f"model pair and eval dataset controlled out  ·  {n} cells, 95% CI clustered by pair",
        fontsize=11.5,
        y=0.965,
    )
    out = OUTDIR / f"ladder_used_between_gap{SUFFIX}.png"
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
        "source_self_ceiling": {f"{s}|{f}|{c}": v for (s, f, c), v in sorted(SOURCE_SELF.items())},
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
            "bootstrap draws and are point-only; no interval is borrowed. The "
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
                "NOT DRAWN on either figure as of 2026-08-12 (user call: 'remove the shuffled "
                "baseline just put the identity + bias'). STILL COMPUTED and still reported "
                "per pair in null_per_tier below — the removal is presentational, not analytic, "
                "and every prose argument that uses these values stands. The CAPACITY control: "
                "per draw the target rows y_t are row-permuted and every y_t-consuming tier "
                "correction is REFIT (issue1336_metric_ladder.py:34-38, 756-763), so it bounds "
                "how much R2 the reparameterization machinery can manufacture from destroyed "
                "correspondence. 20 draws per fit. The per-tier draw-means: the strict tiers "
                "(t0, t6) sit at -0.32..-0.75 and only t7/t8 approach 0 (-0.0008), because t7/t8 "
                "refit the answer side against the shuffled targets and the refit absorbs the "
                "target mean. NaN on the three round-B pairs: pair-specific, never run there, "
                "and never interpolated."
            ),
            "identity_bias": (
                "y_hat = x + b with b the train-fold mean of y - x (analysis/mapping_baselines); "
                "the project's standing mapping baseline, applicable because dims match at "
                "4096->4096. Drawn as a LINE at its true value (-2.14..-3.03 aggregate, worst "
                "per-corpus cell -3.41), which is why the y-axis extends to -3.2 / -3.7. "
                "It is now the ONLY baseline drawn beside the R2=0 line. PROVENANCE IS NO "
                "LONGER DRAWN (user call 2026-08-12: the measured-vs-borrowed marker split is "
                "retired) -- one uniform series, with the split recorded ONLY here: read the "
                "per-pair identity_approx flag below. true = BORROWED from that target's round-3 "
                "siblings (the three round-B pairs, -2.946 / -2.630 / -2.630), which is why "
                "sft__rlvr_long and rlvr__rlvr_long carry the identical -2.630: they share the "
                "target rlvr_long and the baseline is target-keyed, not pair-keyed. "
                "CAUTION: a deeply negative R2 here is a SCALE failure, not absent signal -- "
                "the same cells' knn_identity_bias retrieves the correct held-out answer at "
                "rank 1 in 70.2% of a 2,000-row pool (median over 56 cells, chance 0.05%), "
                "BEATING the fitted ridge map's own 50.3% in 52 of 56 cells."
            ),
            "zero_line": (
                "R2 = 0 is the predict-the-training-mean baseline. It rides the LEGEND rather "
                "than an in-canvas text block (paper-plots SKILL 3.8-bis: no caption block on "
                "the figure canvas); the t7/t8 shuffled null sits 0.0008 under it."
            ),
            "alignment_panel": (
                "the two ALIGNMENT-map identity+bias series (a_ctx, a_ans) are plotted on their "
                "OWN TOP PANEL of the aggregate figure, separated by a spacer row rather than a "
                "broken axis, because they score a DIFFERENT regression from everything on the "
                "main panel: A_ctx_rev predicts the SOURCE contexts from the TARGET contexts and "
                "A_ans predicts the TARGET answers from the SOURCE answers -- both across two "
                "checkpoints -- while the tiers, the cross map and the within-model ceiling all "
                "score context->answer INSIDE one model. On a shared axis their post-training "
                "values (0.68..0.96, vs a ~0.5 ceiling) read as 'the identity baseline beats the "
                "self map', which is a category error, not a result. The by_dataset figure does "
                "not carry these two series at all."
            ),
            "round_b_gap": (
                "round B (selfmap_v3) originally ran NEITHER control. The shuffled-pairing null "
                "is pair-specific (it permutes THIS pair's target rows and refits every "
                "y_t-consuming correction), so it remains a visible GAP on sft__rlvr, "
                "sft__rlvr_long and rlvr__rlvr_long — never interpolated across an uncontrolled "
                "pair. identity+bias-WITHIN is target-keyed and is BORROWED from those targets' "
                "round-3 siblings (identity_approx=true). The two ALIGNMENT-map identity "
                "baselines are MEASURED on the round-B pairs' own rows by the v4 refit "
                "(ALGEBRA_VERSION v4-foldlocal-2-cross-t7t8-alignbaselines), using the parent "
                "metric_ladder's estimator — identity_bias_predict OOF on the same seed-0 "
                "conversation-grouped 5-fold, globally-pooled fc._pooled_r2, A_ctx_rev scored "
                "against the SOURCE contexts and A_ans against the TARGET answers — so those two "
                "series are continuous across all 10 pairs on one basis (align_measured=true). "
                "Measured values on the three round-B pairs, a_ctx / a_ans: sft->rlvr "
                "0.796 / 0.513, sft->rlvr_long 0.743 / 0.511, rlvr->rlvr_long 0.804 / 0.734. "
                "The BORROWED identity+bias values on the same three pairs: -2.946, -2.630, "
                "-2.630 (identity_approx=true per pair below). On the by_dataset figure the "
                "round-B identity+bias was previously a blank gap while the aggregate showed a "
                "value; it now carries the same borrow at panel grain — the target's round-3 "
                "siblings at that same (fmt, corpus). As of 2026-08-12 neither figure DRAWS the "
                "measured/borrowed distinction and neither draws the shuffled null at all; both "
                "facts are sidecar + prose only."
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

    print(
        f"{'pair':<20}{'within':>8}"
        + "".join(f"{'t' + str(t):>8}" for t in TIERS)
        + f"{'cross':>8}  battery"
    )
    for pi, (src, tgt) in enumerate(PAIRS):
        med = {t: _median_r2(D, pi, t) for t in TIERS}
        bat = "round-B" if (src, tgt) in SELFMAP_PAIRS else "round-3"
        print(
            f"{src + '→' + tgt:<20}{D['ceilings'][pi]:>8.3f}"
            # "--" marks NEVER MEASURED (the round-B battery's missing tiers 1-5)
            + "".join(f"{'--':>8}" if np.isnan(med[t]) else f"{med[t]:>8.3f}" for t in TIERS)
            + f"{D['cross'][pi][0]:>8.3f}  {bat}"
        )

    figs = [fig_aggregate(D), fig_percorpus(D), fig_trained_on(D), fig_controlled_gap(D)]
    meta = write_meta(D, figs)
    for p in figs + [meta]:
        print("wrote", p)


if __name__ == "__main__":
    main()
