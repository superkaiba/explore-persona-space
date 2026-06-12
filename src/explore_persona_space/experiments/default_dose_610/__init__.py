# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" intentional
"""Task #610 — default-assistant shielding: dose or identity? (parent #600).

Single manipulated variable (plan §4.1): the ``qwen_default`` slot of the
parent's ``c600_mercenary_near`` panel is handed to ``journalist`` (the pair's
already-characterized matched-control persona). 3 fresh seeds of ONE new mix
(``c610_mercenary_near_nodefault``); the with-default arm is the parent's 3
committed ``c600_mercenary_near`` trajectories, REUSED (no retraining).

Module layout (plan §4.2):
    __init__  — #610 constants; the #600 recipe constants are re-exported
                (the recipe is inherited verbatim — any change would confound
                the single-variable contrast with the reused parent arm).
    cells     — ``build_610_spec``: the inverted-panel CellSpec600 built from
                the PARENT manifest, with the #610 asserts (qwen_default
                ABSENT, journalist exactly once, disjointness) + design.json.
    dispatch  — thin driver over the reused #600 helpers; --smoke | --full;
                gates (a)-(h) reused + NEW (i) primary-DV existence (hard) +
                (j) chassis comparability (soft, recorded); #610 uploads +
                pod sentinel with the full epm:results payload contract.
    analyze   — the §6 registered comparison (CPU, VM, post-teardown):
                3 parent trajectories (git) vs 3 new ones; centered,
                implant-normalized default-context shift; decision zones.
"""

from __future__ import annotations

# ── Inherited #600 recipe constants (single source of truth; re-exported so
# #610 code never re-pins a recipe value — plan §11 inherit-fast-path). ──────
from explore_persona_space.experiments.targeted_proximity_600 import (  # noqa: F401
    ALWAYS_INCLUDE_NEGATIVE,
    BASE_MODEL,
    BATCH_SIZE,
    EXPECTED_MARKER_TOKEN_ID,
    EXPECTED_SHA256,
    EXPECTED_STEPS_PER_EPOCH,
    GRAD_ACCUM,
    HF_DATA_PREFIX_INPUTS,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    LORA_TARGETS_ATTN_ONLY,
    MARKER_BAND_LOG_ONLY,
    MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
    MARKER_TEXT,
    MAX_LORA_RANK_EVAL,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS_GEN,
    N_NEG_PERSONAS,
    NEG_ROWS_PER_PERSONA,
    POS_ROWS,
    QWEN_IM_END_TOKEN_ID,
    SEEDS,
    SOURCE_DG_BAND_NATS,
    SOURCE_PERSONA,
    TOTAL_ROWS,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
)

# ── #610 identifiers (plan §4.2 / §10). ──────────────────────────────────────
HF_DATA_PREFIX = "issue610_default_dose"
HF_ADAPTER_PATH_PREFIX = "adapters/issue_610"
WANDB_PROJECT = "issue610_default_dose"
RUN_NAME_PREFIX = "issue610_"
NEW_SLUG = "c610_mercenary_near_nodefault"
NEW_PLAIN_NAME = "No-default mix (mercenary chassis, qwen_default slot → journalist)"
CHASSIS_SLUG = "c600_mercenary_near"
CHASSIS_TARGET = "mercenary"

# ── The single manipulated variable (plan §4.1). ─────────────────────────────
# journalist = the mercenary pair's matched-control persona (pre-registered in
# the task body; identity re-asserted against the parent manifest's ctrl slot
# in cells.build_610_spec).
REPLACEMENT_PERSONA = "journalist"

# ── Eval-list addition (plan §4.2): the primary DV (qwen_default) and the
# cluster-identity secondary probe (assistant) are in NO default eval set of
# the no-default arm (held_out ∪ {source} ∪ panel) — without this the primary
# DV would silently not exist (gate (i) catches a wiring miss at smoke). ─────
EXTRA_EVAL_PERSONAS = ("qwen_default", "assistant")

# ── Decision rule (plan §6): the parent's measured terminal same-mix noise
# median, pre-registered in the task body as the resolution. ─────────────────
DECISION_BAND = 0.033

# ── Epochs PINNED (plan §4.3 / §7): matched 63 optimizer steps with the
# REUSED parent arm is load-bearing; the parent's smoke ladder is REMOVED by
# design. An out-of-band smoke is a halt-and-report, never a re-pin. ─────────
EPOCHS_PINNED = 1

# ── Gate (j) soft chassis-comparability band (plan §4.2): the parent arm's
# realized villain ΔG range [8.76, 9.305] ± 2 nats. Outside it but inside
# gate (a)'s [5, 19] → proceed, flag in analysis (recorded, not gating). ─────
CHASSIS_DG_SOFT_RANGE_NATS = (6.8, 11.3)

# ── Registered reference points for the §6.4 sanity reads (parent #600
# committed values; the per-seed D_with comparators are RECOMPUTED from the
# parent trajectories by formula — these two are the pre-registered
# cross-cell precedents quoted in plan §5/§6.4). ─────────────────────────────
JOURNALIST_CTRL_PRECEDENT = -0.117  # journalist trained-slot read, parent ctrl cells
ASSISTANT_TRAINED_SLOT_PRECEDENT = -0.193  # assistant trained-slot read, parent finding (c)

# ── Sanity personas (plan §5): trained in BOTH arms; cross-run drift
# detectors read at ±2× DECISION_BAND of their with-default-arm values. ──────
SANITY_PERSONAS = ("bartender", "french_person", "dictator")

# ── Compute accounting (plan §9). ────────────────────────────────────────────
GPU_HOURS_BUDGETED = 22  # instance-GPU-hours (4 × A100-80 × typical wall)
