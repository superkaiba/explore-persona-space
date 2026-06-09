# ruff: noqa: RUF002, RUF003, RUF022  # em-dash + Qwen marker " ※" + Greek ΔG + grouped __all__ intentional
"""Task #504 — bubble vs barrier geometry of a single contrastive negative.

Parent #472. Goal (plan §1): resolve the SHAPE of a single contrastive
negative's leakage-suppression in persona space — does suppression decay with
distance to the negative (BUBBLE), or with angular-shadow alignment from source
through negative (BARRIER), or both?

Design (plan §4):
- 5 placement arms: Near, Mid-Near, Mid-Far, Far, Default-only — varying the
  position of ONE non-default contrastive negative N on the L10 cosine axis.
- 2 seeds (42, 137) × 5 arms = 10 training cells.
- Anchor recipe (Phase 0-calibrated): r=8, α=32 (per #477 `RANK_ALPHA_MAP_V5`),
  lr=2e-6, 1 epoch, batch 4 × grad_accum 4, all-linear, marker-position-only
  loss with `MarkerOnlyDataCollator(tail_tokens=0,
  suppress_at_post_response_slot=True, im_end_token_id=151645)`.
- 6-checkpoint trajectory eval at fractions {0.08, 0.16, 0.33, 0.50, 0.75,
  1.00}; **cross-arm matching at a SINGLE pinned `chosen_checkpoint_fraction`**
  picked by Phase 0 (NOT per-cell latest-in-band — that becomes a Phase 2
  robustness panel only).
- DV: on-policy `log P(※ | T_probe(q) + R_probe_base_greedy)` at post-response
  slot, trained − base (ΔG, nats).
- Verdict via partial-Spearman over 6 covariates (d_source, d_nearest_neg_nd,
  shadow_angle, base_prior_marker, training_step, source_delta_g(cell)); the
  ANALYZER assigns the Bubble/Barrier/Both/Indeterminate verdict, NOT this
  module (no hard p-threshold → verdict ladder).

Module layout (forks `contrastive_neg_geometry_472`):
    __init__       — constants: CELL_SPECS_504, Phase 0 smoke specs, 6 covariates
    cell_resolution — per-cell negative-set resolution + arm → positioned-N lookup
    shadow_angle   — Appendix A.1 angular shadow formula
    phase0         — anchor calibration: pick (chosen_rank, chosen_ckpt_frac)
    phase05        — CPU-only identification gates A/B/C + max-length check
    phase2         — pooled partial-Spearman + collinearity gates + per-layer

Everything else (persona bank, centroids, R-generation, base panel,
build_training_data, train_cell, eval_trajectory, eval_one_cell, select_negatives,
eval_guard) REUSES the #472 rig with `cell_specs=CELL_SPECS_504` threaded
through the existing override kwargs added by #477. The #472 module is
byte-identical for callers that don't pass `cell_specs=`.

Per-constant `Source:` provenance lives in
`tasks/approved/504/plans/plan.md` §11 (Decision Rationale).
"""

from __future__ import annotations

# Re-export the #472 + #477 recipe constants so 504 callers have a single
# import surface and the per-constant Source provenance is one click away.
from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
    MARKER_IM_END_TOKEN_ID,
    RANK_ALPHA_MAP_V5,
    alpha_for_rank,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    BASE_MODEL,
    BATCH_SIZE,
    EXPECTED_MARKER_TOKEN_ID,
    GRAD_ACCUM,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    LORA_DROPOUT,
    MARKER_SEP,
    MARKER_TEXT,
    MAX_LENGTH,
    POS_EX_PER_SOURCE,
    SOURCE_PERSONA,
    WARMUP_RATIO,
    CellSpec,
)

__all__ = [
    # Re-exports (provenance: #472 / #477).
    "ALWAYS_INCLUDE_NEGATIVE",
    "BASE_MODEL",
    "BATCH_SIZE",
    "EXPECTED_MARKER_TOKEN_ID",
    "GRAD_ACCUM",
    "HF_DATA_REPO",
    "HF_MODEL_REPO",
    "LORA_DROPOUT",
    "MARKER_IM_END_TOKEN_ID",
    "MARKER_SEP",
    "MARKER_TEXT",
    "MAX_LENGTH",
    "POS_EX_PER_SOURCE",
    "RANK_ALPHA_MAP_V5",
    "SOURCE_PERSONA",
    "WARMUP_RATIO",
    "CellSpec",
    "alpha_for_rank",
    # NEW in #504.
    "ANCHOR_LR",
    "ARM_NAMES",
    "BAND_CENTERS",
    "CELL_SPECS_504",
    "CHECKPOINT_FRACTIONS",
    "CHECKPOINT_FRACTIONS_V3_FINER",
    "CHECKPOINT_FRACTIONS_V4_BISECTION",
    "DEFAULT_ARM_SLUG",
    "DEFAULT_HEADLINE_LAYER",
    "EMISSION_BAND_HIGH",
    "EMISSION_BAND_LOW",
    "EPOCHS",
    "EPOCHS_FROM_V3_SMOKE_SLUG",
    "EPOCHS_LADDER_V3",
    "FALLBACK_LAYERS",
    "FALLBACK_SOURCE_CANDIDATES",
    "FIXED_LR_V3",
    "HF_DATA_PREFIX_504",
    "LR_FROM_V2_SMOKE_SLUG",
    "LR_LADDER",
    "MAIN_ARM_SLUGS",
    "MAIN_ARM_SLUGS_V2",
    "MAIN_ARM_SLUGS_V3",
    "MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT",
    "NEG_EX_PER_PERSONA",
    "N_POS_PER_CELL",
    "POSITIONED_ARM_SLUGS",
    "POSITIONED_ARM_SLUGS_V2",
    "POSITIONED_ARM_SLUGS_V3",
    "DEFAULT_ARM_SLUG_V2",
    "DEFAULT_ARM_SLUG_V3",
    "PHASE0_CALIB_RANKS",
    "PHASE0_SMOKE_SLUGS",
    "PHASE0_SMOKE_SLUGS_V2",
    "PHASE0_SMOKE_SLUGS_V3",
    "PHASE05_DNN_FLOOR",
    "PHASE05_PANEL_MIN_PROBES",
    "PHASE05_QWEN_DEFAULT_DOMINANCE_THRESHOLD",
    "PHASE05_SHADOW_FLOOR_RAD",
    "ROBUSTNESS_LAYERS",
    "SEEDS",
    "SOURCE_DG_BAND_HIGH",
    "SOURCE_DG_BAND_LOW",
    "TASK_ID",
    "V2_BAND_FOR_ARM",
    "V3_BAND_FOR_ARM",
    "epochs_for_v3_smoke_slug",
    "is_main_arm_slug",
    "is_v2_smoke_slug",
    "is_v3_smoke_slug",
    "lr_for_v2_smoke_slug",
    "positioned_arm_for_band",
    "positioned_arm_for_band_v2",
    "positioned_arm_for_band_v3",
]


# ── Task identity (plan §10 reproducibility card) ───────────────────────────
TASK_ID: int = 504
HF_DATA_PREFIX_504: str = "issue504_geometry"

# ── Seeds (plan §10) ────────────────────────────────────────────────────────
# 2 seeds: project convention + min for sign-stability across seeds (#472, #477).
SEEDS: tuple[int, ...] = (42, 137)

# ── Marker-loss slot fix (plan §11 row "Marker-position-only loss collator") ─
# #477 v6 added a post-response-slot suppression mode to MarkerOnlyDataCollator.
# #504 turns it ON so the marker channel is suppressed at <|im_end|> (id 151645)
# for negatives, training "after a response under this persona, emit EOS, NOT
# the marker" at the SAME slot positives push the marker up at.
MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT: bool = True

# ── Anchor recipe (plan §10 / §11; v2 anchor-recipe redesign) ────────────────
# v1 (rank-ladder): Phase 0 picked ONE of {4, 8, 16} as the anchor rank at the
# pinned lr=2e-6. The rank-ladder path is RETAINED for backwards-compat with
# the round-12/13/14/15/16 reval scripts (i504_reval_grid.py, i504_reval_confirm.py)
# but is NO LONGER the default Phase 0 path. v2 calls
# `dispatch_neg_geometry_504.py --phase phase0 --lr-ladder ...` and the new
# lr-ladder pick path supersedes it.
PHASE0_CALIB_RANKS: tuple[int, ...] = (4, 8, 16)  # v1 only — superseded by LR_LADDER in v2

# ── v2 lr-ladder (plan v2 §4.1; user-directive 2026-06-08T06:46:56Z) ─────────
# v2 single load-bearing change vs v1: replace the rank ladder with an lr
# ladder. Pinned r=8 / α=32 / all-linear / 1 epoch / 200 positives + 200
# negatives, sweep lr ∈ {1e-5, 3e-5, 1e-4} at seed=42.
#
# - 1e-5 = 5× v1's refuted lr=2e-6 (gentle anchor; matches #383's anchor lr
#   at multi-token [ZLT] before the lr-bump to 1e-4 in #397).
# - 3e-5 = midpoint (15× v1's lr; ~30% of #397's 1e-4; novel for this rig).
# - 1e-4 = #397's tested lr for single-token ※ (high end of the explored
#   lr range for this marker).
#
# Source: #397 (lr=1e-4 for single-token ※); #383 (lr=1e-5 for multi-token);
# #448 (anti-saturation requirement — anchor must sit ~5-10 nats below
# ceiling); #479 (lr alone at r=16 attn-only isn't the lever — does NOT
# refute the r=8 all-linear regime tested here).
LR_LADDER: tuple[float, ...] = (1e-5, 3e-5, 1e-4)

# Default ANCHOR_LR (v1 floor recipe) — RETAINED as the default for any
# code path that imports `ANCHOR_LR` without going through the v2 Phase 0
# pick. The v2 pipeline ALWAYS overrides this via the `--lr` CLI override
# on `i504_run_cell.py` (which threads through `train_one_cell(lr_override=)`).
# Source: #477 + #479 — only LR documented to walk source through mid-band
# in the rank-ladder regime. REFUTED at r=8 all-linear in #504 rounds 12-16.
ANCHOR_LR: float = 2e-6

# Epochs — v2 reverts to 1 per plan v2 §4.1 (single-variable discipline:
# lr is the only manipulated knob; do NOT also vary epochs).
#
# History:
#   v1-v14 EPOCHS=1: 200 pos + 200 neg = 400 rows / batch 16 = 25 steps/epoch.
#   Round-13/14 read source ΔG ≈ 0 at this budget — refuted as a rig artifact
#   (PEFT-direct and vLLM both apply the adapter).
#   v15 EPOCHS=3 (composition unchanged at 200 pos + 200 neg = 75 steps):
#   refuted by data — source ΔG plateaued at ~0.11 nats.
#   v16 EPOCHS=3 + NEG composition match (400 neg, 600 rows, 112 steps):
#   plan v2 reverts this. The user-directive at 2026-06-08T06:46:56Z
#   established that step-count alone is NOT the lever; lr is. Reverting
#   to EPOCHS=1 + 200 neg total keeps the v2 single-variable discipline.
EPOCHS: int = 1

# Composition (plan v2 §4.1 + §5):
#   v2: 200 positives + 200 negatives total (100 from qwen_default + 100
#   from the positioned N for the 4 positioned arms; 200 from qwen_default
#   alone for the default-only arm). 400 rows / batch 16 = 25 steps at
#   1 epoch. 1:1 positives-to-TOTAL-negatives ratio per #383 +
#   `.claude/rules/contrastive-negatives.md`.
N_POS_PER_CELL: int = POS_EX_PER_SOURCE  # 200 (reuse #472 constant)
# v2 reverts round-16's NEG_EX_PER_PERSONA=200 back to 100. 100 per persona
# × 2 personas = 200 neg total, matching plan v2 §4.1 + §11's 1:1 ratio row.
# Source: plan v2 §11 "Positives / negatives ratio: 200 / 200 = 1:1".
NEG_EX_PER_PERSONA: int = 100  # 100 per persona × 2 personas = 200 total negs
# Default-only arm: 200 from qwen_default alone, matching positioned arms'
# 200-row neg total so cross-arm step counts stay equal.
NEG_EX_DEFAULT_ONLY_ARM: int = 200


# ── Layer choice (plan §4.2 + §11) ──────────────────────────────────────────
# Headline = L10 (where Gate B passes per #472); L15 + L20 as robustness.
DEFAULT_HEADLINE_LAYER: int = 10
ROBUSTNESS_LAYERS: tuple[int, ...] = (15, 20)
FALLBACK_LAYERS: tuple[int, ...] = (15, 20)  # ordered fallback if L10 fails Gate B

# ── Trajectory cadence (plan §10 / §11) ─────────────────────────────────────
# Matched to #472; denser early where ΔG is rising sub-ceiling. Phase 0 picks
# ONE of these as `chosen_checkpoint_fraction`; Phase 1 reads ALL arms at that
# fraction (cross-arm matching guarantee, plan §0 / §4.4 Step 1).
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)

# ── Phase 0 in-band windows (plan §4.1 pick rule) ───────────────────────────
# Source ΔG band: in [5, 12] nats (the readable sub-ceiling window).
SOURCE_DG_BAND_LOW: float = 5.0
SOURCE_DG_BAND_HIGH: float = 12.0
# Source on-policy emission band: in [0.1, 0.8].
EMISSION_BAND_LOW: float = 0.1
EMISSION_BAND_HIGH: float = 0.8

# ── Phase 0.5 gates (plan §4.2) ─────────────────────────────────────────────
# Gate A — identification floor: median across-arm SD of `d_nearest_neg_nd`
# ≥ 0.02 (the #472 floor), `shadow_angle` ≥ 0.10 rad (~5.7°).
PHASE05_DNN_FLOOR: float = 0.02
PHASE05_SHADOW_FLOOR_RAD: float = 0.10
# Gate B — default-assistant non-dominance: < 0.5 of probes have qwen_default
# as their single nearest negative across the 4 arms (the #472 L20 collapse).
PHASE05_QWEN_DEFAULT_DOMINANCE_THRESHOLD: float = 0.5
# Gate C — held-out panel sufficiency: ≥ 30 probes after exclusions.
PHASE05_PANEL_MIN_PROBES: int = 30

# ── Placement bands (plan §4.2 step 2; cos targets per arm) ─────────────────
# The 4 positioned arms target these cos(N, source=villain) values; Phase 0.5
# finds the persona in the bank closest to each target.
BAND_CENTERS: dict[str, float] = {
    "near": 0.7,  # top quartile, cos ≥ ~0.6 — picks the highest-cos persona
    "mid_near": 0.4,  # median of top half
    "mid_far": 0.1,  # median of bottom half
    "far": -0.2,  # bottom quartile, cos ≤ ~−0.1 — picks the lowest-cos persona
}

# Arm slug → human-readable name (plan §5 table).
ARM_NAMES: dict[str, str] = {
    "c504_near": "Near (twin)",
    "c504_mid_near": "Mid-Near",
    "c504_mid_far": "Mid-Far",
    "c504_far": "Far",
    "c504_default_only": "Default-only (no positioned negative)",
}

# Slugs for the 4 placement-positioned arms (pooled regression input).
POSITIONED_ARM_SLUGS: tuple[str, str, str, str] = (
    "c504_near",
    "c504_mid_near",
    "c504_mid_far",
    "c504_far",
)

# Slug for the default-only floor arm (NOT in the regression — floor reference).
DEFAULT_ARM_SLUG: str = "c504_default_only"

# Full Phase 1 main grid (5 arms).
MAIN_ARM_SLUGS: tuple[str, ...] = (*POSITIONED_ARM_SLUGS, DEFAULT_ARM_SLUG)

# Phase 0 smoke cells (v1 rank-ladder, retained for round-12/13/14/15/16
# reval scripts; NOT exercised by the v2 dispatcher path): 3 ranks at lr=2e-6,
# count=2 composition, seed=42.
PHASE0_SMOKE_SLUGS: tuple[str, str, str] = (
    "c504_smoke_r4",
    "c504_smoke_r8",
    "c504_smoke_r16",
)

# ── v2 Phase 0 smoke cells (lr-ladder; plan v2 §4.1) ────────────────────────
# 3 lr values × r=8 / α=32 / all-linear / 1 epoch / 200 pos + 200 neg, seed=42.
# Slugs are dashless (`c504v2_smoke_lr1e5`, not `c504v2_smoke_lr1e-5`) to keep
# them argparse-friendly and consistent with plan §4.1 / §5 / §10 tables.
PHASE0_SMOKE_SLUGS_V2: tuple[str, str, str] = (
    "c504v2_smoke_lr1e5",
    "c504v2_smoke_lr3e5",
    "c504v2_smoke_lr1e4",
)

# Mapping {v2 smoke slug → lr} so the dispatcher + per-cell runner can recover
# the lr from the slug WITHOUT a separate CLI arg per cell. The CLI's
# `--lr-ladder` flag is the source of truth for which cells run; this map is
# the slug→lr lookup downstream.
LR_FROM_V2_SMOKE_SLUG: dict[str, float] = {
    "c504v2_smoke_lr1e5": 1e-5,
    "c504v2_smoke_lr3e5": 3e-5,
    "c504v2_smoke_lr1e4": 1e-4,
}

# ── v2 main arms (plan v2 §5 conditions table) ──────────────────────────────
# Same 5 placement arms as v1, renamed `c504v2_*` so the lr-swept reads are
# distinguishable on WandB + HF model repo from the v1 lr=2e-6 reads
# (which stay under `c504_*` adapters).
POSITIONED_ARM_SLUGS_V2: tuple[str, str, str, str] = (
    "c504v2_near",
    "c504v2_mid_near",
    "c504v2_mid_far",
    "c504v2_far",
)
DEFAULT_ARM_SLUG_V2: str = "c504v2_default_only"
MAIN_ARM_SLUGS_V2: tuple[str, ...] = (*POSITIONED_ARM_SLUGS_V2, DEFAULT_ARM_SLUG_V2)

# v2 ARM_NAMES additions (merged into ARM_NAMES below for compatibility).
ARM_NAMES.update(
    {
        "c504v2_near": "Near (twin)",
        "c504v2_mid_near": "Mid-Near",
        "c504v2_mid_far": "Mid-Far",
        "c504v2_far": "Far",
        "c504v2_default_only": "Default-only (no positioned negative)",
    }
)

# v2 band → arm-slug mapping.
V2_BAND_FOR_ARM: dict[str, str] = {
    "near": "c504v2_near",
    "mid_near": "c504v2_mid_near",
    "mid_far": "c504v2_mid_far",
    "far": "c504v2_far",
}

# ── v3 EPOCHS-ladder (plan v3 §4.1; user-directive 2026-06-08T11:44:00Z) ────
# v3 single load-bearing change vs v2: replace the lr-ladder with an EPOCHS
# ladder at FIXED lr=1e-4 (the v2-closest cell). Pinned r=8 / α=32 / all-linear
# / 200 pos + 200 neg / count=2 / source=villain, sweep EPOCHS ∈ {2, 3} at
# seed=42.
#
# - EPOCHS=2 doubles effective optimization steps from v2's ~25 to ~50.
# - EPOCHS=3 triples them to ~75 — brackets against saturation upper bound.
#
# Source: v2 phase0 smoke (2026-06-08T11:18:06Z) refutation of lr-alone at
# 1 epoch; #397 (lr=1e-4 as the tested ceiling for single-token ※); #477
# (r=8 / count=2 mid-band precedent at 63 steps in a different regime); #448
# (anti-saturation requirement — anchor sits ~5-10 nats below ceiling, on-
# policy emission ∈ [0.1, 0.8]).
EPOCHS_LADDER_V3: tuple[int, ...] = (2, 3)

# v3 fixed lr — NOT swept in v3. Carried from v2's empirical closest cell.
FIXED_LR_V3: float = 1e-4

# v3 Phase 0 smoke cells (EPOCHS ladder; plan v3 §4.1). 2 cells × 1 seed at
# r=8 / α=32 / all-linear / lr=1e-4 / count=2 / 200 pos + 200 neg, seed=42.
# Slugs are dashless (`c504v3_smoke_eps2`, not `c504v3_smoke_eps_2`) to mirror
# the v2 naming convention.
PHASE0_SMOKE_SLUGS_V3: tuple[str, str] = (
    "c504v3_smoke_eps2",
    "c504v3_smoke_eps3",
)

# Mapping {v3 smoke slug → EPOCHS} so the dispatcher + per-cell runner can
# recover the epochs from the slug WITHOUT a separate CLI arg per cell.
EPOCHS_FROM_V3_SMOKE_SLUG: dict[str, int] = {
    "c504v3_smoke_eps2": 2,
    "c504v3_smoke_eps3": 3,
}

# v3 in-plan recovery (plan v3 §4.1 trigger B + §4.2). Fine-grained checkpoint
# fractions used ONLY when EPOCHS=2 saturates at every coarse fraction; re-runs
# the EPOCHS=2 cell with these finer fractions (~0.15 GPU-h). Kept separate
# from CHECKPOINT_FRACTIONS so the v3 picker can decide which cadence applies.
CHECKPOINT_FRACTIONS_V3_FINER: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08)

# v4 bisection (plan v5 §4.2 step 1). When the v4 bystander-resolution picker
# (§4.1) returns verdict='no_in_band_anchor' — every EPOCHS=3 fraction is
# either pinned at the marker-argmax ceiling or below the +0.5 nats floor —
# bisect to EPOCHS=2 at this finer-fraction grid and re-apply the bystander-
# resolution gate. ~0.6 GPU-h total (training + re-eval). If a fraction
# passes, pin EPOCHS=2 / chosen_frac and proceed to Phase 0.5 + Phase 0.6 +
# Phase 1. If NO fraction passes, exit to plan v5's rank bump (§4.2 step 2).
CHECKPOINT_FRACTIONS_V4_BISECTION: tuple[float, ...] = (0.04, 0.08, 0.12, 0.16)

# ── v3 main arms (plan v3 §5 conditions table; structurally identical to v2,
# but EPOCHS is the swept variable not lr, so slugs carry the `v3` namespace
# so the EPOCHS-swept reads are distinguishable on WandB + HF model repo
# from the v2 lr-swept reads). ──────────────────────────────────────────────
POSITIONED_ARM_SLUGS_V3: tuple[str, str, str, str] = (
    "c504v3_near",
    "c504v3_mid_near",
    "c504v3_mid_far",
    "c504v3_far",
)
DEFAULT_ARM_SLUG_V3: str = "c504v3_default_only"
MAIN_ARM_SLUGS_V3: tuple[str, ...] = (*POSITIONED_ARM_SLUGS_V3, DEFAULT_ARM_SLUG_V3)

# v3 ARM_NAMES additions (merged into ARM_NAMES below for compatibility).
ARM_NAMES.update(
    {
        "c504v3_near": "Near (twin)",
        "c504v3_mid_near": "Mid-Near",
        "c504v3_mid_far": "Mid-Far",
        "c504v3_far": "Far",
        "c504v3_default_only": "Default-only (no positioned negative)",
    }
)

# v3 band → arm-slug mapping.
V3_BAND_FOR_ARM: dict[str, str] = {
    "near": "c504v3_near",
    "mid_near": "c504v3_mid_near",
    "mid_far": "c504v3_mid_far",
    "far": "c504v3_far",
}

# ── Phase 0 fallback (plan v2 §4.2) ─────────────────────────────────────────
# If the lr-ladder produces no in-band cell on `villain`, the dispatcher
# swaps to an EASIER source persona and re-runs the same 3-cell smoke. The
# pick rule (plan v2 §4.2) is "smallest |cos(P, qwen_default) − cos(neutral,
# qwen_default)|" over candidates with `cos(P, qwen_default) > cos(villain,
# qwen_default)`. The candidate list is enumerated below; the dispatcher
# CLI accepts `--source <name>` to override.
#
# Source: plan v2 §4.2 + §11 "Source persona: villain / medical_doctor".
FALLBACK_SOURCE_CANDIDATES: tuple[str, ...] = (
    "medical_doctor",
    "librarian",
    "programmer",
    "surgeon",
    "formal_writer",
    "scholar",
)

# Cell specs are the 6-tuple shape #472 / #477 use:
#   (slug, plain_name, placement, n_neg_personas, neg_ex_per_persona, in_pooled)
#
# `placement` is the legacy #472 enum ("near"/"far"/"spread"/"none") which the
# `select_negatives_by_geometry` helper consumes. For #504 we DO NOT use that
# helper to pick negatives (each arm picks ONE specific persona, not a band).
# Phase 0.5 / cell_resolution.py supplies the negative persona list directly;
# the placement field is set to "spread" for the 4 positioned arms (a benign
# default for any introspection by select_negatives.negatives_for_cell — but
# the dispatcher overrides the negative list via the cell_resolution path).
# For default_only we use "none" so any accidental fall-through to the
# select_negatives path returns an empty arm rather than a wrong one.
#
# n_neg_personas: 2 for the positioned arms (qwen_default + 1 positioned), 1
# for the default-only arm (qwen_default alone, scaled to 200 ex).
# neg_ex_per_persona: 100 for the positioned arms (split evenly); 200 for the
# default-only arm (single persona × 200 ex = 200 neg rows, matching the
# positioned arms' total).
# in_pooled: True for the 4 positioned arms (the regression input); False for
# default_only (floor reference, NOT in the regression).
CELL_SPECS_504: tuple[CellSpec, ...] = (
    # ── Main Phase 1 arms (5 cells × 2 seeds = 10 runs). ────────────────────
    ("c504_near", ARM_NAMES["c504_near"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504_mid_near", ARM_NAMES["c504_mid_near"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504_mid_far", ARM_NAMES["c504_mid_far"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504_far", ARM_NAMES["c504_far"], "spread", 2, NEG_EX_PER_PERSONA, True),
    (
        "c504_default_only",
        ARM_NAMES["c504_default_only"],
        "none",  # safe default if the select_negatives fall-through fires
        1,
        NEG_EX_DEFAULT_ONLY_ARM,
        False,  # floor reference; NOT in the pooled regression
    ),
    # ── Phase 0 smoke cells (v1 rank ladder; 3 cells × 1 seed = 3 runs). ────
    # placement="spread" here too — Phase 0 uses cell_resolution to pick its
    # qwen_default + 1-mid-band N negative list directly.
    ("c504_smoke_r4", "Smoke r=4", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504_smoke_r8", "Smoke r=8", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504_smoke_r16", "Smoke r=16", "spread", 2, NEG_EX_PER_PERSONA, False),
    # ── v2 Main Phase 1 arms (lr-anchor; 5 cells × 2 seeds = 10 runs). ──────
    # Same shape as the v1 c504_* arms; lr is the only varied knob (set by
    # Phase 0 v2 pick + threaded via `i504_run_cell.py --lr`).
    ("c504v2_near", ARM_NAMES["c504v2_near"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504v2_mid_near", ARM_NAMES["c504v2_mid_near"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504v2_mid_far", ARM_NAMES["c504v2_mid_far"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504v2_far", ARM_NAMES["c504v2_far"], "spread", 2, NEG_EX_PER_PERSONA, True),
    (
        "c504v2_default_only",
        ARM_NAMES["c504v2_default_only"],
        "none",
        1,
        NEG_EX_DEFAULT_ONLY_ARM,
        False,
    ),
    # ── v2 Phase 0 smoke cells (lr-ladder; 3 cells × 1 seed = 3 runs). ──────
    # Same composition + placement as the v2 main arms (count=2, mid-band N
    # picked by Phase 0.5). placement="spread" so any fall-through to the
    # select_negatives helper is a benign no-op (cell_resolution overrides).
    ("c504v2_smoke_lr1e5", "Smoke lr=1e-5", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504v2_smoke_lr3e5", "Smoke lr=3e-5", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504v2_smoke_lr1e4", "Smoke lr=1e-4", "spread", 2, NEG_EX_PER_PERSONA, False),
    # ── v3 Main Phase 1 arms (EPOCHS-anchor; 5 cells × 2 seeds = 10 runs). ──
    # Same shape as v1/v2; EPOCHS is the only varied knob (set by Phase 0
    # v3 pick + threaded via `i504_run_cell.py --epochs`).
    ("c504v3_near", ARM_NAMES["c504v3_near"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504v3_mid_near", ARM_NAMES["c504v3_mid_near"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504v3_mid_far", ARM_NAMES["c504v3_mid_far"], "spread", 2, NEG_EX_PER_PERSONA, True),
    ("c504v3_far", ARM_NAMES["c504v3_far"], "spread", 2, NEG_EX_PER_PERSONA, True),
    (
        "c504v3_default_only",
        ARM_NAMES["c504v3_default_only"],
        "none",
        1,
        NEG_EX_DEFAULT_ONLY_ARM,
        False,
    ),
    # ── v3 Phase 0 smoke cells (EPOCHS ladder; 2 cells × 1 seed = 2 runs). ──
    # Same composition + placement as v3 main arms (count=2, mid-band N picked
    # by Phase 0.5). placement="spread" → benign fall-through to select_negatives
    # is a no-op (cell_resolution.py overrides).
    ("c504v3_smoke_eps2", "Smoke EPOCHS=2", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504v3_smoke_eps3", "Smoke EPOCHS=3", "spread", 2, NEG_EX_PER_PERSONA, False),
)


def is_main_arm_slug(slug: str) -> bool:
    """True iff `slug` is one of the 5 main Phase 1 arms (NOT a smoke cell).

    Recognizes the v1 (`c504_*`), v2 (`c504v2_*`), and v3 (`c504v3_*`) main-arm
    slug sets.
    """
    return slug in MAIN_ARM_SLUGS or slug in MAIN_ARM_SLUGS_V2 or slug in MAIN_ARM_SLUGS_V3


def is_v2_smoke_slug(slug: str) -> bool:
    """True iff `slug` is one of the 3 v2 lr-ladder smoke cells."""
    return slug in PHASE0_SMOKE_SLUGS_V2


def is_v3_smoke_slug(slug: str) -> bool:
    """True iff `slug` is one of the 2 v3 EPOCHS-ladder smoke cells."""
    return slug in PHASE0_SMOKE_SLUGS_V3


def lr_for_v2_smoke_slug(slug: str) -> float:
    """Return the lr value associated with a v2 smoke slug.

    Raises:
        KeyError: `slug` is not a v2 smoke slug.
    """
    if slug not in LR_FROM_V2_SMOKE_SLUG:
        raise KeyError(
            f"Not a v2 smoke slug: {slug!r}; expected one of {sorted(LR_FROM_V2_SMOKE_SLUG)}"
        )
    return LR_FROM_V2_SMOKE_SLUG[slug]


def epochs_for_v3_smoke_slug(slug: str) -> int:
    """Return the EPOCHS value associated with a v3 smoke slug.

    Raises:
        KeyError: `slug` is not a v3 smoke slug.
    """
    if slug not in EPOCHS_FROM_V3_SMOKE_SLUG:
        raise KeyError(
            f"Not a v3 smoke slug: {slug!r}; expected one of {sorted(EPOCHS_FROM_V3_SMOKE_SLUG)}"
        )
    return EPOCHS_FROM_V3_SMOKE_SLUG[slug]


def positioned_arm_for_band(band: str) -> str:
    """Map a band name (near/mid_near/mid_far/far) → its v1 main arm slug.

    Raises ValueError on unknown band. Used by cell_resolution to look up
    which positioned N each band's main arm consumes.
    """
    mapping = {
        "near": "c504_near",
        "mid_near": "c504_mid_near",
        "mid_far": "c504_mid_far",
        "far": "c504_far",
    }
    if band not in mapping:
        raise ValueError(f"Unknown band {band!r}; expected one of {sorted(mapping)}")
    return mapping[band]


def positioned_arm_for_band_v2(band: str) -> str:
    """Map a band name (near/mid_near/mid_far/far) → its v2 main arm slug.

    Raises ValueError on unknown band. v2 analogue of `positioned_arm_for_band`.
    """
    if band not in V2_BAND_FOR_ARM:
        raise ValueError(f"Unknown band {band!r}; expected one of {sorted(V2_BAND_FOR_ARM)}")
    return V2_BAND_FOR_ARM[band]


def positioned_arm_for_band_v3(band: str) -> str:
    """Map a band name (near/mid_near/mid_far/far) → its v3 main arm slug.

    Raises ValueError on unknown band. v3 analogue of `positioned_arm_for_band`.
    """
    if band not in V3_BAND_FOR_ARM:
        raise ValueError(f"Unknown band {band!r}; expected one of {sorted(V3_BAND_FOR_ARM)}")
    return V3_BAND_FOR_ARM[band]
