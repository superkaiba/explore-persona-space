#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003  # em-dash / ※ intentional
"""#1586 grid registry — matched-install LoRA vs full-FT method comparison.

CPU-testable constants + selection logic for the (behavior × regime × seed)
grid at the persona context (plan §4.1-§4.3). Every anchor/checkpoint value is
copied verbatim from the approved plan's §4.1 table (itself quoted from
``eval_results/issue_1481/analysis/verdict_manifest.json``); band / gap /
window / tol constants are IMPORTED from the parent modules
(``issue1481_cells`` / ``issue1481_marker``), never retyped.

Selection rule (plan §4.3, declared divergence from #1112's earliest-in-band):
among in-band/in-window rungs pick the rung minimizing |metric − paired LoRA
anchor|; tie-break earliest; closest-approach fallback reported with a
dose-mismatch label, never a silent drop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1481_cells as c1481  # noqa: E402
import issue1481_marker as mk1481  # noqa: E402

ISSUE = 1586
SLUG = "issue1586_methodgen"
DATA_PREFIX = SLUG  # issue1586_methodgen/... on the data repo
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
WANDB_PROJECT = "issue1586_methodgen"

BEH_KEYS: tuple[str, ...] = ("syc", "imp", "cas", "mk")
CONTENT_BEH_KEYS: tuple[str, ...] = ("syc", "imp", "cas")
REGIMES: tuple[str, ...] = ("con", "po")
SEEDS: tuple[int, ...] = (42, 137)

# Behavior names resolve through the #1481 registry map (never retyped);
# the marker key is the #1481 marker line's own naming.
BEHAVIOR_BY_KEY: dict[str, str] = {**c1481.BEHAVIOR_BY_KEY, "mk": "marker"}

# Bands / gates — imported from the parents (plan §11 Sources).
JUDGED_RATE_BAND = c1481.JUDGED_RATE_BAND  # (0.60, 0.85), #1090/#1112/#1481
DOSE_MATCH_MAX_RATE_GAP = c1481.DOSE_MATCH_MAX_RATE_GAP  # 0.10, #1481
INSTALL_WINDOW = mk1481.INSTALL_WINDOW  # (5.0, 12.0) nat, #1481/#1333
DOSE_MATCH_TOL_NATS = mk1481.DOSE_MATCH_TOL_NATS  # 1.5 nat, #1481
ARGMAX_CEILING = mk1481.ARGMAX_CEILING  # 0.92, #1333 de-saturation gate

# Reused-arm parity gate calibration (plan §4.4; WARN-class, never HALT —
# artifact-reuse gate-calibration rule; HALT reserved for structural
# apply-path breakage).
P1_PARITY_MAX_ABS_DELTA = 0.15  # content |Δrate| WARN band (#1481 P1)
P1_MARKER_WARN_NATS = 1.0  # marker ΔG WARN band (#1333 drift +0.05..+0.15)
REUSED_FT_PARITY_TOL = 0.15  # #1112 con ckpt Tier-2 re-read vs committed

# Registered FT one-shot extensions (plan §4.2).
CONTENT_STEP_CEILING = 30
CONTENT_EXT_CEILING = 60
MARKER_FT_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
MARKER_FT_EXT_GRID: tuple[int, ...] = (7, 8, 9, 10, 11, 12)  # plan-new contingency
MARKER_EXT_MIN_DELTA_NATS = 5.0  # extend once when ΔG@6 < window floor


@dataclass(frozen=True)
class ReusedLoraArm:
    """One reused #1481-line LoRA verdict arm (plan §4.1 table row)."""

    cell: str  # 1586 pair key: <beh>-pers-lora-<regime>-s<seed>
    beh_key: str
    regime: str
    seed: int
    run_id: str  # parent run id (provenance)
    step: int  # verdict rung
    anchor: float  # content: Tier-x judged rate; marker: ΔG nat
    subfolder: str  # HF model-repo subfolder holding the adapter
    recipe_class: str  # "content" (r32/α64/rsLoRA/7-mod) | "marker" (r16/α32/attn)


def _arm(beh: str, reg: str, seed: int, run_id: str, step: int, anchor: float, sub: str):
    cls = "marker" if beh == "mk" else "content"
    return ReusedLoraArm(
        cell=f"{beh}-pers-lora-{reg}-s{seed}",
        beh_key=beh,
        regime=reg,
        seed=seed,
        run_id=run_id,
        step=step,
        anchor=anchor,
        subfolder=sub,
        recipe_class=cls,
    )


REUSED_LORA_ARMS: tuple[ReusedLoraArm, ...] = (
    # sycophancy (anchors = judged rates; plan §4.1)
    _arm(
        "syc",
        "con",
        42,
        "syc-pers-con-lr1e5-s42",
        15,
        0.65,
        "adapters/issue1090_fu7/syc-c3-lr1e5/checkpoint-15",
    ),
    _arm(
        "syc",
        "po",
        42,
        "syc-pers-po-lr1e5-s42",
        10,
        0.61,
        "issue1481/syc-pers-po-lr1e5-s42/checkpoint-10",
    ),
    _arm(
        "syc",
        "con",
        137,
        "syc-pers-con-lr1e5-s137",
        15,
        0.61,
        "issue1481/syc-pers-con-lr1e5-s137/checkpoint-15",
    ),
    _arm(
        "syc",
        "po",
        137,
        "syc-pers-po-lr1e5-s137",
        15,
        0.79,
        "issue1481/syc-pers-po-lr1e5-s137/checkpoint-15",
    ),
    # impolite
    _arm(
        "imp",
        "con",
        42,
        "imp-pers-con-lr3e5-s42",
        30,
        0.81,
        "adapters/issue1090_fu4/imp-pers-lr3e5/checkpoint-30",
    ),
    _arm(
        "imp",
        "po",
        42,
        "imp-pers-po-lr1e5-s42",
        35,
        0.66,
        "issue1481/imp-pers-po-lr1e5-s42/checkpoint-35",
    ),
    _arm(
        "imp",
        "con",
        137,
        "imp-pers-con-lr3e5-s137",
        35,
        0.73,
        "issue1481/imp-pers-con-lr3e5-s137/checkpoint-35",
    ),
    _arm(
        "imp",
        "po",
        137,
        "imp-pers-po-lr1e5-s137",
        35,
        0.66,
        "issue1481/imp-pers-po-lr1e5-s137/checkpoint-35",
    ),
    # casual writing style (the #1434 ws arms)
    _arm(
        "cas",
        "con",
        42,
        "cas-pers-con-lr1e5-s42",
        25,
        0.60,
        "issue1434/ws-pers-lr1e5/checkpoint-25",
    ),
    _arm(
        "cas",
        "po",
        42,
        "cas-pers-po-lr1e5-s42",
        25,
        0.81,
        "issue1434/ws-po-pers-lr1e5/checkpoint-25",
    ),
    _arm(
        "cas",
        "con",
        137,
        "cas-pers-con-lr1e5-s137",
        30,
        0.85,
        "issue1481/cas-pers-con-lr1e5-s137/checkpoint-30",
    ),
    _arm(
        "cas",
        "po",
        137,
        "cas-pers-po-lr1e5-s137",
        25,
        0.81,
        "issue1481/cas-pers-po-lr1e5-s137/checkpoint-25",
    ),
    # marker (anchors = ΔG nat)
    _arm(
        "mk",
        "con",
        42,
        "mk-pers-con-lr5e6-s42",
        90,
        6.35,
        "issue1481/marker/mk-pers-con-lr5e6-s42/checkpoint-90",
    ),
    _arm(
        "mk",
        "po",
        42,
        "mk-pers-po-lr5e6-s42",
        80,
        7.21,
        "issue1481/marker/mk-pers-po-lr5e6-s42/checkpoint-80",
    ),
    _arm(
        "mk",
        "con",
        137,
        "mk-pers-con-lr5e6-s137",
        90,
        5.79,
        "issue1481/marker/mk-pers-con-lr5e6-s137/checkpoint-90",
    ),
    _arm(
        "mk",
        "po",
        137,
        "mk-pers-po-lr5e6-s137",
        80,
        7.42,
        "issue1481/marker/mk-pers-po-lr5e6-s137/checkpoint-80",
    ),
)
LORA_ARM_BY_CELL = {a.cell: a for a in REUSED_LORA_ARMS}
assert len(REUSED_LORA_ARMS) == 16

# ── Frozen training mixes (seed-invariant; shas re-probed + pinned at p0) ────
# {beh_key: {regime: (data-repo path, expected_rows)}} — plan §4.2 table.
MIXES: dict[str, dict[str, tuple[str, int]]] = {
    "syc": {
        "con": ("issue1090_pvdatagen/c3-sycophancy-claude/mix/train_mix.jsonl", 80),
        "po": ("issue1481_conpos_grid/po_mixes/syc-pers/mix/train_mix.jsonl", 60),
    },
    "imp": {
        "con": ("issue1090_pvdatagen/c2-impolite-claude/mix/train_mix.jsonl", 80),
        "po": ("issue1481_conpos_grid/po_mixes/imp-pers/mix/train_mix.jsonl", 60),
    },
    "cas": {
        "con": ("issue1434_writingstyle/ws-pers/mix/train_mix.jsonl", 80),
        "po": ("issue1434_writingstyle/ws-po-pers/mix/train_mix.jsonl", 60),
    },
    "mk": {
        "con": ("issue1481_conpos_grid/marker/mixes/marker_pers_con.jsonl", 1000),
        "po": ("issue1481_conpos_grid/marker/mixes/marker_pers_po.jsonl", 1000),
    },
}

# The #1481 marker ICL bank (Hub-verified 2026-07-22): panel_contexts()'s
# _icl_context reads <out_root>/inputs/icl_examples_marker.json — staged at
# p0 to exactly that consumer path (reuse leg (h)(ii)).
MARKER_ICL_BANK_PATH = "issue1481_conpos_grid/marker/inputs/icl_examples_marker.json"

# ── FT cells ─────────────────────────────────────────────────────────────────


def ft_cell_id(beh: str, regime: str, seed: int) -> str:
    return f"{beh}-pers-ft-{regime}-s{seed}"


# Reused FT contrast arm (plan §4.1): the #1112 con checkpoint IS the s42 con
# arm; the #1112 po checkpoint is a parity CROSS-CHECK row only (0.18 dose gap
# vs the 0.61 anchor — syc-pers-ft-po-s42 retrains to anchor).
REUSED_FT_CELL = ft_cell_id("syc", "con", 42)
REUSED_FT_SUBFOLDER = "issue1112/s3_fullft_neg/checkpoint-8"  # OVERFLOW_REPO
REUSED_FT_COMMITTED_SELECTION = "issue1112_geometry2x2/selection/s3_fullft_neg/selection.json"
PARITY_XCHECK_SUBFOLDER = "issue1112/s4_fullft_pos/checkpoint-6"  # OVERFLOW_REPO
PARITY_XCHECK_COMMITTED_TIER2 = 0.790  # plan §4.1 (committed #1112 Tier-2)

NEW_FT_CELLS: tuple[str, ...] = tuple(
    ft_cell_id(b, r, s)
    for b in BEH_KEYS
    for r in REGIMES
    for s in SEEDS
    if ft_cell_id(b, r, s) != REUSED_FT_CELL
)
assert len(NEW_FT_CELLS) == 15
ALL_FT_CELLS: tuple[str, ...] = (REUSED_FT_CELL, *NEW_FT_CELLS)


def parse_ft_cell(cell: str) -> tuple[str, str, int]:
    """'syc-pers-ft-con-s137' -> ('syc', 'con', 137) (fail-loud)."""
    parts = cell.split("-")
    if len(parts) != 5 or parts[1] != "pers" or parts[2] != "ft":
        raise ValueError(f"bad FT cell id: {cell!r}")
    beh, regime, seed = parts[0], parts[3], int(parts[4].removeprefix("s"))
    if beh not in BEH_KEYS or regime not in REGIMES or seed not in SEEDS:
        raise ValueError(f"bad FT cell id: {cell!r}")
    return beh, regime, seed


def lora_pair_of(ft_cell: str) -> ReusedLoraArm:
    """The method-paired reused LoRA verdict arm of an FT cell."""
    beh, regime, seed = parse_ft_cell(ft_cell)
    return LORA_ARM_BY_CELL[f"{beh}-pers-lora-{regime}-s{seed}"]


# The smoke cell (plan §4.8: smoke IS sweep with one cell, p2→p9).
SMOKE_CELL = ft_cell_id("syc", "con", 137)

# ── Selection (plan §4.3: anchor-nearest in-band; closest-approach fallback) ─


def band_distance(x: float, band: tuple[float, float]) -> float:
    lo, hi = band
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


def select_anchor_nearest(
    metric_by_step: dict[int, float],
    *,
    anchor: float,
    band: tuple[float, float],
    eligible_steps: set[int] | None = None,
) -> dict:
    """Anchor-nearest in-band rung selection (content rates AND marker ΔG).

    Among rungs whose metric is inside ``band`` (and, when given, whose step is
    in ``eligible_steps`` — the marker de-saturation gate pass-set), pick the
    rung minimizing |metric − anchor|; tie-break earliest step. No in-band rung
    → closest-approach fallback (min band distance, tie-break earliest),
    reported with ``fallback='closest_approach'`` — a reportable finding,
    never a silent drop (plan §4.3/§7).
    """
    if not metric_by_step:
        raise ValueError("empty ladder — nothing to select")
    steps = sorted(metric_by_step)
    if eligible_steps is not None:
        gated = [s for s in steps if s in eligible_steps]
    else:
        gated = steps
    in_band = [s for s in gated if band[0] <= metric_by_step[s] <= band[1]]
    if in_band:
        # gap rounded to 1e-9 so a float-epsilon difference between decimal
        # literals (|0.70-0.65| = 0.04999...93 vs |0.60-0.65| = 0.05) cannot
        # defeat the registered earliest-step tie-break.
        step = min(in_band, key=lambda s: (round(abs(metric_by_step[s] - anchor), 9), s))
        return {
            "step": step,
            "metric": metric_by_step[step],
            "in_band": True,
            "fallback": None,
            "anchor": anchor,
            "anchor_gap": abs(metric_by_step[step] - anchor),
        }
    pool = gated or steps  # all rungs gate-failed -> report nearest anyway
    step = min(pool, key=lambda s: (band_distance(metric_by_step[s], band), s))
    return {
        "step": step,
        "metric": metric_by_step[step],
        "in_band": False,
        "fallback": "closest_approach",
        "anchor": anchor,
        "anchor_gap": abs(metric_by_step[step] - anchor),
        "gate_failed_all": eligible_steps is not None and not gated,
    }


def content_dose_label(ft_tier2_rate: float, arm: ReusedLoraArm) -> dict:
    """Pair dose-match label — FT side from the TIER-2 confirm read (plan
    §4.3: never the noisier Tier-1 selection rate)."""
    lo, hi = JUDGED_RATE_BAND
    gap = abs(ft_tier2_rate - arm.anchor)
    both_in_band = (lo <= ft_tier2_rate <= hi) and (lo <= arm.anchor <= hi)
    return {
        "ft_tier2_rate": ft_tier2_rate,
        "lora_anchor_rate": arm.anchor,
        "rate_gap": gap,
        "dose_matched": bool(both_in_band and gap <= DOSE_MATCH_MAX_RATE_GAP),
        "max_gap": DOSE_MATCH_MAX_RATE_GAP,
    }


def marker_dose_label(ft_delta_g: float, arm: ReusedLoraArm) -> dict:
    gap = abs(ft_delta_g - arm.anchor)
    lo, hi = INSTALL_WINDOW
    both_in = (lo <= ft_delta_g <= hi) and (lo <= arm.anchor <= hi)
    return {
        "ft_delta_g": ft_delta_g,
        "lora_anchor_delta_g": arm.anchor,
        "gap_nats": gap,
        "dose_matched": bool(both_in and gap <= DOSE_MATCH_TOL_NATS),
        "tol_nats": DOSE_MATCH_TOL_NATS,
    }


# ── Capture / panel geometry constants (plan §4.6) ───────────────────────────

PRIMARY_LAYER_CONTENT = 14  # #653/#1112/#1315
PRIMARY_LAYER_MARKER = 25  # #1112/#1333
N_LAYERS = 28
MAX_NEW_TOKENS_CONTENT = 1024
MAX_NEW_TOKENS_MARKER = 2048  # >= 2x longest trained completion (#260 rule)
TF_BATCH_SIZE = 8
BOOT_SEED = 653
N_BOOT = 1000
N_BOOT_NORM = 2000
HALF_DRAW_SEED = 1112
HALF_DRAW_M = 60
