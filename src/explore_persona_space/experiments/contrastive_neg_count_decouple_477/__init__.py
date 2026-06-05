# ruff: noqa: RUF002, RUF003, RUF022  # em-dash + Qwen marker " ※" + Greek ΔG + grouped __all__ intentional
"""Task #477 — implant-decoupled contrastive-negative count sweep.

Parent #472. Goal (plan §1): determine whether contrastive-negative COUNT
independently controls bystander marker leakage *net of source-implant strength*
— the axis #472 could not separate because count and source-implant ΔG
occupied non-overlapping plateaus there.

Design (plan §4): for each count level in {2, 4, 8, 16}, first run a per-count
LR-calibration sweep (Phase 2, 20 cells, seed=42, 5 LRs each, terminal-only
eval) to pick the LR that lands source-self ΔG in [10.5, 13.5] nats AND source
emission P(※)≥0.30. Then run the main 8-cell sweep (4 counts × 2 seeds) at the
calibrated LRs with full trajectory eval (Phase 3), plus the implant-only-axis
arm (Phase 4, 6 cells at fixed count, varying LR) to isolate the implant axis.

Everything except the calibration layer + the 477 cell registry REUSES the #472
rig at `src/explore_persona_space/experiments/contrastive_neg_geometry_472/*`
(persona bank, centroids, R-generate, base panel, build_training_data,
train_cell, eval_trajectory). The 472 module's `negatives_for_cell` and
`build_cell` now accept an optional `cell_specs` kwarg so 477 cells can drive
them; otherwise the 472 path is byte-identical (backward-compat).

Module layout:
    __init__   — constants, CELL_SPECS_477, re-export of 472 recipe constants
    calibrate  — pick_lr_for_count + validity_gate (pure functions)

Pseudocode + grounding for every constant lives in
`tasks/running/477/plans/plan.md` §4 (the calibration layer pseudocode) and §11
(per-constant `Source:` notes).
"""

from __future__ import annotations

# Re-export the #472 recipe so 477 callers have a single import surface.
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    BASE_MODEL,
    BATCH_SIZE,
    EXPECTED_MARKER_TOKEN_ID,
    GRAD_ACCUM,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MARKER_SEP,
    MARKER_TEXT,
    MAX_LENGTH,
    MAX_NEW_TOKENS_GEN,
    POS_EX_PER_SOURCE,
    SOURCE_PERSONA,
    WARMUP_RATIO,
    CellSpec,
)

# Names re-exported for the 477 single-import surface. Listing them here keeps
# ruff F401 happy without losing the convenience of `from
# contrastive_neg_count_decouple_477 import MARKER_TEXT`.
__all__ = [
    # Re-exported from #472.
    "ALWAYS_INCLUDE_NEGATIVE",
    "BASE_MODEL",
    "BATCH_SIZE",
    "EXPECTED_MARKER_TOKEN_ID",
    "GRAD_ACCUM",
    "HF_DATA_REPO",
    "HF_MODEL_REPO",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "MARKER_SEP",
    "MARKER_TEXT",
    "MAX_LENGTH",
    "MAX_NEW_TOKENS_GEN",
    "POS_EX_PER_SOURCE",
    "SOURCE_PERSONA",
    "WARMUP_RATIO",
    "CellSpec",
    # NEW in #477 (v2 legacy LR-calibration constants).
    "ANCHOR_COUNT",
    "CALIB_SLUGS",
    "CALIBRATION_LR_GRID",
    "CELL_SPECS_477",
    "COUNT_LEVELS",
    "EPOCHS",
    "IMPLANT_SWEEP_LRS",
    "IMPLANT_SWEEP_SLUGS",
    "MAIN_SLUGS",
    "MATCH_BAND",
    "MIN_SOURCE_EMISSION",
    "NEG_EX_PER_PERSONA",
    "TARGET_SOURCE_DELTA_G",
    "count_for_slug",
    "lr_for_implant_sweep_slug",
    "slug_for_count",
    # NEW in v3/v4 (step-lever pivot).
    "CALIBRATION_LR_V3",
    "MAX_SOURCE_EMISSION_V3",
    "MIN_SOURCE_EMISSION_V3",
    "TARGET_STEPS",
    # NEW in v4r2 (H2 step-sweep replacement for v2 LR-sweep implant_sweep).
    "IMPLANT_SWEEP_STEPS",
    "IMPLANT_SWEEP_V4_ANCHOR_SLUG",
    "IMPLANT_SWEEP_V4_SLUGS",
    "implant_sweep_v4_slug_for_step",
    "step_for_implant_sweep_v4_slug",
    # NEW in v6 (M1 rank control + M2 single-source-of-truth alpha map).
    "ALPHA_CONTROL_V6",
    "CAL_A0_SLUGS",
    "CAL_A_SLUGS",
    "MARKER_IM_END_TOKEN_ID",
    "RANK_ALPHA_MAP_V5",
    "RANK_CONTROL_COUNTS_V6",
    "RANK_CONTROL_V6",
    "RANK_GRID_V5",
    "alpha_for_rank",
    "cal_a0_slug",
    "cal_a_slug",
    "count_for_calA0_slug",
    "count_for_calA_slug",
    "rank_for_calA_slug",
]

# ── Calibration constants (plan §4 pseudocode + §11 grounding) ───────────────
# Target source-self ΔG = 12.0 ± 1.5 nats — the matched-implant band. Sits in
# the gap between #472's low-count plateau (~8) and placement plateau (~13-15),
# in the "implant real, behaviorally emitting, not saturated" window the parent
# missed (plan §11; Source: #472 non-overlapping plateau analysis).
TARGET_SOURCE_DELTA_G: float = 12.0
MATCH_BAND: float = 1.5

# Source emission floor — the matched implant must be BEHAVIORAL, not sub-
# emission log-prob drift. 0.30 = min behavioral threshold ("source emits the
# marker on a non-trivial fraction of its own answers"); Source: #472 emission-
# floor finding (max 0.17 on source under 1 epoch).
MIN_SOURCE_EMISSION: float = 0.30

# Count axis (the experimental variable). 4 levels; {2,4,8} cover #472's range
# and 16 extends it so the partial-Spearman has 4 data points instead of 3.
COUNT_LEVELS: tuple[int, ...] = (2, 4, 8, 16)

# Calibration LR grid: 25× #472's 1e-5 baseline (5× down, 5× up). Spans enough
# to compensate for the hypothesized ~10-nat implant differential across the
# count axis. Source: ungrounded — needs calibration sweep; the kill criterion
# in plan §7 fires if this grid is too narrow.
CALIBRATION_LR_GRID: tuple[float, ...] = (2e-6, 5e-6, 1e-5, 2e-5, 5e-5)

# Implant-only-axis arm (H2): same negatives as the 4-persona anchor, vary LR
# alone over these 3 values × 2 seeds = 6 cells. Source: ungrounded — H2 is the
# implant-axis identification check.
IMPLANT_SWEEP_LRS: tuple[float, ...] = (5e-6, 1e-5, 2e-5)

# 2 epochs raised from #472's 1 to land source in mid-emission range (plan §11;
# Source: ungrounded — needs smoke-test; 1 epoch was sub-emission in #472, 3
# epochs saturated in #448).
EPOCHS: int = 2

# Per-persona negative examples (same as #472).
NEG_EX_PER_PERSONA: int = 200

# ── v3/v4 step-lever constants (plan v4 §4, §6, §11 — "STEP-LEVER PIVOT"). ───
# After v2 calibration round 1 confirmed LR is a dead lever (Source: plan §2 +
# §11), v3/v4 pivot to per-cell early-step checkpoints at FIXED lr=2e-6. The
# dense early-step grid spans the sub-step-10 plateau analysis from #472.
# Source: ungrounded — #477 step-lever bet; Phase 2 step calibration validates.
TARGET_STEPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

# Fixed calibration LR for v3/v4 (lr=2e-6 was the only v2 LR that produced any
# sub-saturation ΔG signal across counts; Source: plan §2 + §11).
CALIBRATION_LR_V3: float = 2e-6

# v3/v4 raised emission floor (anti-collapse) + new ceiling (anti-saturation).
# Source: plan §11 ("0.40 raised floor + anti-saturation ceiling").
MIN_SOURCE_EMISSION_V3: float = 0.40
MAX_SOURCE_EMISSION_V3: float = 0.95

# Anchor count for the implant-only-axis arm: the #472 anchor's 4 negatives.
ANCHOR_COUNT: int = 4

# v4r2 H2 step-sweep: at fixed lr=CALIBRATION_LR_V3 (2e-6) and count=ANCHOR_COUNT
# (4), read the implant-only axis at non-terminal step levels {16, 64} plus the
# terminal step (added implicitly via frac=1.0). LR is the dead lever the v4
# pivot abandons (round-1 confirmed); the step axis IS the implant axis under
# the step-lever pivot. Source: plan v4 §4 PHASE 4 (step sweep, not LR sweep)
# + plan v4 §6 H2 cross-check (calibrate step→leakage at fixed count).
IMPLANT_SWEEP_STEPS: tuple[int, ...] = (16, 64)

# ── Cell registry (CELL_SPECS_477) ───────────────────────────────────────────
# Same 6-tuple shape as #472's CellSpec: (slug, plain_name, placement,
# n_neg_personas, neg_ex_per_persona, in_pooled).
#
# Three phase families (plan §4):
#   (1) calibration cells (Phase 2): 4 counts × 5 LRs = 20 SLUGS, all
#       seed=42, terminal-only eval. The slug encodes ONLY the count;
#       LR is threaded per-launch via --lr (the slug is the SAME across
#       all 5 LRs at a given count — the dispatcher distinguishes by
#       the (slug, lr) tuple on the launched subprocess).
#   (2) main cells (Phase 3): 4 counts × 2 seeds = 8 cells, full trajectory
#       eval, LR per cell picked by Phase 2.5.
#   (3) implant-only-axis cells (Phase 4): 3 LRs × 2 seeds = 6 cells, fixed
#       count = anchor (4), full trajectory eval.
#
# in_pooled is left False everywhere — 477 does NOT reuse #472's pooled
# geometry regression; its analyses are partial Spearmans over the count and
# LR axes.

# Calibration cells (one slug per count level; LR threaded per-launch).
_CALIB_SPECS: tuple[CellSpec, ...] = tuple(
    (
        f"c477_calib_negp_{c}",
        f"Calibration: {c}-persona negatives",
        "spread",
        c,
        NEG_EX_PER_PERSONA,
        False,
    )
    for c in COUNT_LEVELS
)

# Main cells (one slug per count level; LR per cell from calibration pick).
_MAIN_SPECS: tuple[CellSpec, ...] = tuple(
    (
        f"c477_main_calib_negp_{c}",
        f"{c}-persona negatives (calibrated)",
        "spread",
        c,
        NEG_EX_PER_PERSONA,
        False,
    )
    for c in COUNT_LEVELS
)


# Implant-only-axis cells (fixed count = anchor, LR varies). Slug encodes the
# LR so each cell has its own run directory + sentinel.
def _lr_slug(lr: float) -> str:
    # 5e-06 → "5e-6"; 1e-05 → "1e-5". Tight, slug-safe.
    return f"{lr:g}".replace("e-0", "e-")


_IMPLANT_SWEEP_SPECS: tuple[CellSpec, ...] = tuple(
    (
        f"c477_implantsweep_lr{_lr_slug(lr)}",
        f"Anchor at LR={lr:g} (implant-only axis)",
        "spread",
        ANCHOR_COUNT,
        NEG_EX_PER_PERSONA,
        False,
    )
    for lr in IMPLANT_SWEEP_LRS
)

# v4r2 anchor cell: ONE training run per seed (at lr=CALIBRATION_LR_V3, count=
# ANCHOR_COUNT). The worker evaluates the trajectory at step levels
# {16, 64, T} and emits per-step records; the dispatcher expands them into
# IMPLANT_SWEEP_V4_SLUGS for the analyze partial.
_IMPLANT_SWEEP_V4_ANCHOR_SPEC: CellSpec = (
    "c477_implantsweep_v4_anchor",
    "v4 anchor (implant-only-axis, step sweep)",
    "spread",
    ANCHOR_COUNT,
    NEG_EX_PER_PERSONA,
    False,
)

CELL_SPECS_477: tuple[CellSpec, ...] = (
    _CALIB_SPECS + _MAIN_SPECS + _IMPLANT_SWEEP_SPECS + (_IMPLANT_SWEEP_V4_ANCHOR_SPEC,)
)

# Phase classification (used by the dispatcher to group cells by phase).
CALIB_SLUGS: tuple[str, ...] = tuple(c[0] for c in _CALIB_SPECS)
MAIN_SLUGS: tuple[str, ...] = tuple(c[0] for c in _MAIN_SPECS)
IMPLANT_SWEEP_SLUGS: tuple[str, ...] = tuple(c[0] for c in _IMPLANT_SWEEP_SPECS)


def slug_for_count(count: int, phase: str) -> str:
    """Resolve the cell slug for (count, phase).

    Args:
        count: one of COUNT_LEVELS for calibration / main; ANCHOR_COUNT for
            implant_sweep (count is the fixed anchor in that phase).
        phase: "calibration" | "main" | "implant_sweep".

    Raises:
        ValueError: unknown phase or count not in COUNT_LEVELS (for calibration
            / main).
    """
    if phase == "calibration":
        if count not in COUNT_LEVELS:
            raise ValueError(f"count={count} not in COUNT_LEVELS={COUNT_LEVELS}")
        return f"c477_calib_negp_{count}"
    if phase == "main":
        if count not in COUNT_LEVELS:
            raise ValueError(f"count={count} not in COUNT_LEVELS={COUNT_LEVELS}")
        return f"c477_main_calib_negp_{count}"
    if phase == "implant_sweep":
        # implant_sweep slugs encode the LR, not the count; caller resolves by LR.
        raise ValueError(
            "slug_for_count does not resolve implant_sweep slugs (they encode LR, "
            "not count); use IMPLANT_SWEEP_SLUGS / IMPLANT_SWEEP_LRS directly."
        )
    raise ValueError(f"Unknown phase {phase!r}")


def count_for_slug(slug: str) -> int:
    """Reverse lookup: the integer count level a slug belongs to.

    Implant-sweep slugs ALL belong to ANCHOR_COUNT — v2 slugs sweep LR at fixed
    count; v4 slugs sweep optimizer-step at fixed count + lr. The v4 anchor
    slug (one training run per seed, expanded into per-step records by the
    dispatcher) also belongs to ANCHOR_COUNT. v6 Cal-A / Cal-A0 slugs encode
    BOTH count + rank (cal_a_slug / cal_a0_slug). Raises KeyError on unknown slug.
    """
    for c in COUNT_LEVELS:
        if slug == f"c477_calib_negp_{c}" or slug == f"c477_main_calib_negp_{c}":
            return c
    if slug in IMPLANT_SWEEP_SLUGS:
        return ANCHOR_COUNT
    if slug == IMPLANT_SWEEP_V4_ANCHOR_SLUG or slug in IMPLANT_SWEEP_V4_SLUGS:
        return ANCHOR_COUNT
    # v6 Cal-A: c477_calA_negp_<count>_r<rank>.
    if slug.startswith("c477_calA_negp_") and not slug.startswith("c477_calA0_negp_"):
        # Defer to the precise helper; raises KeyError on a malformed Cal-A slug.
        return count_for_calA_slug(slug)
    # v6 Cal-A0: c477_calA0_negp_<count>_r32.
    if slug.startswith("c477_calA0_negp_"):
        return count_for_calA0_slug(slug)
    raise KeyError(f"Unknown 477 cell slug: {slug!r}")


def lr_for_implant_sweep_slug(slug: str) -> float:
    """Reverse lookup: the LR encoded in an implant-sweep slug.

    Raises KeyError if slug is not an implant-sweep slug.
    """
    for lr in IMPLANT_SWEEP_LRS:
        if slug == f"c477_implantsweep_lr{_lr_slug(lr)}":
            return lr
    raise KeyError(f"Not an implant-sweep slug: {slug!r}")


# ── v4r2 H2 step-sweep slugs (plan v4 §4 PHASE 4 — STEP sweep, not LR sweep) ─
# One anchor training run per seed at lr=CALIBRATION_LR_V3, count=ANCHOR_COUNT.
# Per-step cell summaries are read OUT of that single trajectory at step levels
# {16, 64, T} (terminal). The dispatcher expands the per-seed anchor result
# into 3 per-step records for the v4 implant-only-axis analyze partial.
IMPLANT_SWEEP_V4_ANCHOR_SLUG: str = "c477_implantsweep_v4_anchor"


def _step_slug(step: int | str) -> str:
    """Slug suffix for an implant-sweep-v4 step level: '16' / '64' / 'T'."""
    if isinstance(step, str):
        if step != "T":
            raise ValueError(f"_step_slug: string step suffix must be 'T' (terminal); got {step!r}")
        return "T"
    return str(int(step))


IMPLANT_SWEEP_V4_SLUGS: tuple[str, ...] = tuple(
    f"c477_implantsweep_step{_step_slug(s)}" for s in (*IMPLANT_SWEEP_STEPS, "T")
)


def implant_sweep_v4_slug_for_step(step: int | str) -> str:
    """Per-step slug for the v4 implant-only-axis arm.

    Args:
        step: a non-terminal optimizer step in IMPLANT_SWEEP_STEPS, OR the
            literal string ``"T"`` for the terminal-step level.

    Raises:
        ValueError: step not in {*IMPLANT_SWEEP_STEPS, "T"}.
    """
    if isinstance(step, str):
        if step != "T":
            raise ValueError(
                f"implant_sweep_v4_slug_for_step: string step must be 'T'; got {step!r}"
            )
        return f"c477_implantsweep_step{_step_slug(step)}"
    if int(step) not in IMPLANT_SWEEP_STEPS:
        raise ValueError(
            f"implant_sweep_v4_slug_for_step: step={step} not in "
            f"IMPLANT_SWEEP_STEPS={IMPLANT_SWEEP_STEPS} (and not 'T' for terminal)."
        )
    return f"c477_implantsweep_step{_step_slug(int(step))}"


def step_for_implant_sweep_v4_slug(slug: str) -> int | str:
    """Reverse lookup: the step level encoded in a v4 implant-sweep slug.

    Returns ``int`` for non-terminal step levels, ``"T"`` for the terminal slug.

    Raises:
        KeyError: slug is not a v4 implant-sweep slug.
    """
    if slug == f"c477_implantsweep_step{_step_slug('T')}":
        return "T"
    for s in IMPLANT_SWEEP_STEPS:
        if slug == f"c477_implantsweep_step{_step_slug(s)}":
            return int(s)
    raise KeyError(f"Not a v4 implant-sweep slug: {slug!r}")


# ── v6 RECIPE-SCALE PIVOT (LoRA rank lever + Cal-A0 r=32 slot-fix control). ──
# Cal-A: rank sweep across {2, 4, 8} at the v4 lr/positives/epochs floor.
# Cal-A0: r=32 / α=64 (v4 byte-identical) at counts {2, 4, 16} with the
# slot-fix port turned on — the slot-bug-vs-capacity diagnostic (plan v6 §4
# PHASE 2A + 2A-CONTROL). Each cell uses the SAME 4 count levels as the v4
# main sweep; Cal-A0 drops count=8 to stay within the ~3 GPU-h M1 budget.

# ── v6 M2 SINGLE SOURCE OF TRUTH for LoRA alpha ─────────────────────────────
# rsLoRA scaling factor α/√r held fixed at the v4 r=32/α=64 equivalent
# (α/√r = 11.31). Source: arXiv 2312.03732 §3 (rsLoRA). Hand calculations:
#   r=2 → α=11.31×√2 = 15.99 → 16
#   r=4 → α=11.31×2   = 22.62 → 23
#   r=8 → α=11.31×√8 = 32.00 → 32
# The r=32 CONTROL uses α=64 (the v4 byte-identical baseline; NOT in the map
# — it is the diagnostic anchor, not the swept axis). NO `2*r` math anywhere.
# Dispatcher's _verify_alpha_invariant guard + the parameterized threading
# test enforce that every alpha at training time is read from this map (or
# is the literal 64 when rank=32).
RANK_ALPHA_MAP_V5: dict[int, int] = {2: 16, 4: 23, 8: 32}

# Cal-A rank grid (the swept axis; M2 invariant: every member must have an
# entry in RANK_ALPHA_MAP_V5).
RANK_GRID_V5: tuple[int, ...] = (2, 4, 8)

# Cal-A0 control (the diagnostic anchor; v4 byte-identical recipe with the
# slot-fix port turned on). Counts {2, 4, 16} chosen as a 3-point overlap
# with Cal-A spanning the full count range (count=8 dropped to stay in the
# M1 ~3 GPU-h budget).
RANK_CONTROL_V6: int = 32
ALPHA_CONTROL_V6: int = 64
RANK_CONTROL_COUNTS_V6: tuple[int, ...] = (2, 4, 16)

# Qwen-2.5-7B-Instruct ``<|im_end|>`` token id; the post-response slot for the
# marker-channel DV. Threaded to MarkerOnlyDataCollator via
# TrainLoraConfig.marker_im_end_token_id when
# marker_suppress_at_post_response_slot=True (plan v6 §0 + §4.4 SLOT-FIX row).
MARKER_IM_END_TOKEN_ID: int = 151645


def alpha_for_rank(rank: int) -> int:
    """Single-source-of-truth alpha lookup (v6 M2 invariant).

    Returns ``RANK_ALPHA_MAP_V5[rank]`` for Cal-A ranks {2, 4, 8}, or
    ``ALPHA_CONTROL_V6`` (=64) for the Cal-A0 control rank (=32). Any other
    rank raises ``ValueError`` — v6 supports exactly these four ranks across
    Cal-A + Cal-A0. The dispatcher's ``_verify_alpha_invariant`` guard + the
    threading test BOTH call into this helper; no caller may compute alpha
    by formula (``2*r`` etc.).
    """
    if rank in RANK_ALPHA_MAP_V5:
        return RANK_ALPHA_MAP_V5[rank]
    if rank == RANK_CONTROL_V6:
        return ALPHA_CONTROL_V6
    raise ValueError(
        f"v6 M2 alpha invariant: unknown rank={rank}; supported = "
        f"{sorted(RANK_ALPHA_MAP_V5)} (Cal-A) + {RANK_CONTROL_V6} (Cal-A0 control)."
    )


def cal_a_slug(count: int, rank: int) -> str:
    """Cal-A cell slug for (count, rank). Plan v6 §5 table."""
    if count not in COUNT_LEVELS:
        raise ValueError(f"Cal-A count={count} not in COUNT_LEVELS={COUNT_LEVELS}")
    if rank not in RANK_GRID_V5:
        raise ValueError(f"Cal-A rank={rank} not in RANK_GRID_V5={RANK_GRID_V5}")
    return f"c477_calA_negp_{count}_r{rank}"


def cal_a0_slug(count: int) -> str:
    """Cal-A0 control cell slug for (count). Plan v6 §5 table (rank fixed at 32)."""
    if count not in RANK_CONTROL_COUNTS_V6:
        raise ValueError(
            f"Cal-A0 count={count} not in RANK_CONTROL_COUNTS_V6={RANK_CONTROL_COUNTS_V6}"
        )
    return f"c477_calA0_negp_{count}_r{RANK_CONTROL_V6}"


def count_for_calA_slug(slug: str) -> int:
    """Reverse lookup: the count level encoded in a Cal-A slug."""
    for count in COUNT_LEVELS:
        for rank in RANK_GRID_V5:
            if slug == cal_a_slug(count, rank):
                return count
    raise KeyError(f"Not a Cal-A slug: {slug!r}")


def rank_for_calA_slug(slug: str) -> int:
    """Reverse lookup: the rank encoded in a Cal-A slug."""
    for count in COUNT_LEVELS:
        for rank in RANK_GRID_V5:
            if slug == cal_a_slug(count, rank):
                return rank
    raise KeyError(f"Not a Cal-A slug: {slug!r}")


def count_for_calA0_slug(slug: str) -> int:
    """Reverse lookup: the count level encoded in a Cal-A0 control slug."""
    for count in RANK_CONTROL_COUNTS_V6:
        if slug == cal_a0_slug(count):
            return count
    raise KeyError(f"Not a Cal-A0 slug: {slug!r}")


# Cal-A + Cal-A0 cell spec registries. The placement is always "spread" and
# in_pooled is False (v6 cells do not feed the #472 pooled geometry
# regression; their analyses are partial Spearmans + rank-pick over the
# count × rank axes).
_CAL_A_SPECS: tuple[CellSpec, ...] = tuple(
    (
        cal_a_slug(count, rank),
        f"Cal-A: {count}-persona negatives, LoRA r={rank}",
        "spread",
        count,
        NEG_EX_PER_PERSONA,
        False,
    )
    for rank in RANK_GRID_V5
    for count in COUNT_LEVELS
)

_CAL_A0_SPECS: tuple[CellSpec, ...] = tuple(
    (
        cal_a0_slug(count),
        f"Cal-A0 control: {count}-persona negatives, LoRA r={RANK_CONTROL_V6} (slot-fix)",
        "spread",
        count,
        NEG_EX_PER_PERSONA,
        False,
    )
    for count in RANK_CONTROL_COUNTS_V6
)

# Append v6 cells to the existing CELL_SPECS_477 (so panel-disjointness logic
# in select_negatives.py picks them up). NOTE: CELL_SPECS_477 was defined
# above; rebind it here so v4 + v6 cells share the registry.
CELL_SPECS_477 = CELL_SPECS_477 + _CAL_A_SPECS + _CAL_A0_SPECS

CAL_A_SLUGS: tuple[str, ...] = tuple(c[0] for c in _CAL_A_SPECS)
CAL_A0_SLUGS: tuple[str, ...] = tuple(c[0] for c in _CAL_A0_SPECS)
