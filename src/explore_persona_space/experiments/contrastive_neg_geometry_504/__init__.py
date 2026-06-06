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
    "DEFAULT_ARM_SLUG",
    "DEFAULT_HEADLINE_LAYER",
    "EMISSION_BAND_HIGH",
    "EMISSION_BAND_LOW",
    "EPOCHS",
    "FALLBACK_LAYERS",
    "HF_DATA_PREFIX_504",
    "MAIN_ARM_SLUGS",
    "MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT",
    "NEG_EX_PER_PERSONA",
    "N_POS_PER_CELL",
    "POSITIONED_ARM_SLUGS",
    "PHASE0_CALIB_RANKS",
    "PHASE0_SMOKE_SLUGS",
    "PHASE05_DNN_FLOOR",
    "PHASE05_PANEL_MIN_PROBES",
    "PHASE05_QWEN_DEFAULT_DOMINANCE_THRESHOLD",
    "PHASE05_SHADOW_FLOOR_RAD",
    "ROBUSTNESS_LAYERS",
    "SEEDS",
    "SOURCE_DG_BAND_HIGH",
    "SOURCE_DG_BAND_LOW",
    "TASK_ID",
    "is_main_arm_slug",
    "positioned_arm_for_band",
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

# ── Anchor recipe (plan §10 / §11; Source: #477 r=8/count=2/lr=2e-6 mid-band) ─
# Phase 0 picks ONE of {4, 8, 16} as the anchor rank — α from RANK_ALPHA_MAP_V5
# for r ∈ {2, 4, 8}; r=16 is NOT in the v5 map, so plan §11 explicitly carries
# α=32 forward (the same α that #477 used at r=8). The plan's expected pick is
# r=8 (only cell that hit non-saturating mid-band ΔG in #477 at count=2).
PHASE0_CALIB_RANKS: tuple[int, ...] = (4, 8, 16)

# Anchor LR (Source: #477 + #479 — only LR that walks source through mid-band).
ANCHOR_LR: float = 2e-6

# Epochs (Source: #472 fix vs #448 — 1 epoch keeps source sub-ceiling at the
# chosen rank+lr; #477 confirmed at r=8 the 1-epoch budget hits mid-band).
EPOCHS: int = 1

# Composition (plan §4.1 + §5):
#   200 positives + 200 negatives = 400 rows / batch 16 = ~25 steps per cell.
#   200 / 200 = 1:1 contrastive ratio (#383 default, #477 sustained).
N_POS_PER_CELL: int = POS_EX_PER_SOURCE  # 200 (reuse #472 constant)
NEG_EX_PER_PERSONA: int = 100  # 100 per persona × 2 personas = 200 total negs
# Default-only arm: 200 from qwen_default (single negative persona).
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

# Phase 0 smoke cells: 3 ranks at lr=2e-6, count=2 composition, seed=42.
# Each is read at the same composition as Phase 1 (200 positives + 200 negs).
PHASE0_SMOKE_SLUGS: tuple[str, str, str] = (
    "c504_smoke_r4",
    "c504_smoke_r8",
    "c504_smoke_r16",
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
    # ── Phase 0 smoke cells (3 cells × 1 seed = 3 runs, at mid-band N). ─────
    # placement="spread" here too — Phase 0 uses cell_resolution to pick its
    # qwen_default + 1-mid-band N negative list directly.
    ("c504_smoke_r4", "Smoke r=4", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504_smoke_r8", "Smoke r=8", "spread", 2, NEG_EX_PER_PERSONA, False),
    ("c504_smoke_r16", "Smoke r=16", "spread", 2, NEG_EX_PER_PERSONA, False),
)


def is_main_arm_slug(slug: str) -> bool:
    """True iff `slug` is one of the 5 main Phase 1 arms (NOT a smoke cell)."""
    return slug in MAIN_ARM_SLUGS


def positioned_arm_for_band(band: str) -> str:
    """Map a band name (near/mid_near/mid_far/far) → its main arm slug.

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
