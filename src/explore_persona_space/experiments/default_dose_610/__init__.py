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

from dataclasses import dataclass
from pathlib import Path

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
WANDB_PROJECT = "issue610_default_dose"  # ONE project across chassis (v2 plan §6)


# ── Chassis registry (follow-up amendment plan v2 §4.1). ─────────────────────
# The follow-up round replicates the dose-vs-identity contrast on a SECOND
# training-mix design ("chassis"). Everything chassis-dependent lives in ONE
# frozen config; "mercenary" re-binds the round-1 values so the module-level
# constants below stay byte-equivalent (pinned by tests/test_issue610_spec.py).
@dataclass(frozen=True)
class ChassisConfig:
    """Per-chassis identifiers + registered reference points (v2 plan §2/§4/§5).

    ``output_subdir`` / ``hf_subprefix`` are None for the round-1 mercenary
    chassis (its artifacts live at the issue roots); follow-up chassis nest
    under ``eval_results/issue_610/<output_subdir>/`` and
    ``<prefix>/<hf_subprefix>/`` per the same-issue follow-up convention.
    """

    name: str
    chassis_slug: str  # the reused #600 comparator cell
    chassis_target: str  # the #600 target whose near cell is the chassis
    new_slug: str  # the new no-default cell
    new_plain_name: str
    replacement: str  # the pair's matched-control persona (gets qwen_default's rows)
    sanity_personas: tuple[str, ...]  # trained in BOTH arms (drift detectors)
    dg_soft_range: tuple[float, float]  # gate (j): parent-arm villain ΔG ± 2 nats
    replacement_ctrl_precedent: float  # replacement's trained-slot read, parent ctrl cells
    run_name_prefix: str
    # Registered with-arm drift-detector medians (v2 plan §5), recomputed by
    # formula from the committed comparator trajectories at analyze time and
    # ASSERTED against these values (None → no registered values, round 1).
    sanity_with_arm_expected: dict[str, float] | None = None
    hf_subprefix: str | None = None
    output_subdir: str | None = None

    @property
    def hf_data_prefix(self) -> str:
        return f"{HF_DATA_PREFIX}/{self.hf_subprefix}" if self.hf_subprefix else HF_DATA_PREFIX

    @property
    def hf_adapter_path_prefix(self) -> str:
        return (
            f"{HF_ADAPTER_PATH_PREFIX}/{self.hf_subprefix}"
            if self.hf_subprefix
            else HF_ADAPTER_PATH_PREFIX
        )

    @property
    def output_root_default(self) -> Path:
        root = Path("eval_results/issue_610")
        return root / self.output_subdir if self.output_subdir else root

    @property
    def data_root_default(self) -> Path:
        root = Path("data/issue_610")
        return root / self.hf_subprefix if self.hf_subprefix else root

    @property
    def figures_dir_default(self) -> Path:
        root = Path("figures/issue_610")
        return root / self.hf_subprefix if self.hf_subprefix else root


CHASSES: dict[str, ChassisConfig] = {
    "mercenary": ChassisConfig(
        name="mercenary",
        chassis_slug="c600_mercenary_near",
        chassis_target="mercenary",
        new_slug="c610_mercenary_near_nodefault",
        new_plain_name="No-default mix (mercenary chassis, qwen_default slot → journalist)",
        replacement="journalist",
        sanity_personas=("bartender", "french_person", "dictator"),
        dg_soft_range=(6.8, 11.3),
        replacement_ctrl_precedent=-0.117,
        run_name_prefix="issue610_",
        sanity_with_arm_expected=None,  # round 1 pre-dates the registered-medians assert
        hf_subprefix=None,
        output_subdir=None,
    ),
    # v2 plan §2/§4.1/§5 — the single manipulated variable of the follow-up
    # round: chassis c600_mercenary_near → c600_software_engineer_near. The
    # pair identities (near=data_scientist, ctrl=hospice_nurse) are re-asserted
    # against the parent manifest in cells.build_610_spec.
    "software_engineer": ChassisConfig(
        name="software_engineer",
        chassis_slug="c600_software_engineer_near",
        chassis_target="software_engineer",
        new_slug="c610_software_engineer_near_nodefault",
        new_plain_name=(
            "No-default mix (software_engineer chassis, qwen_default slot → hospice_nurse)"
        ),
        replacement="hospice_nurse",
        sanity_personas=("bartender", "french_person", "data_scientist"),
        dg_soft_range=(6.5, 11.8),  # realized [8.542, 9.842] ± 2 nats (v2 §4.1)
        replacement_ctrl_precedent=-0.0372,  # hospice_nurse, parent ctrl cells (v2 §3)
        run_name_prefix="issue610_second_chassis_",
        # v2 §5 drift-detector medians (with-arm, computed by the analyze.py
        # formula from the committed comparator trajectories at plan time).
        sanity_with_arm_expected={
            "bartender": 0.0178,
            "french_person": -0.0031,
            "data_scientist": -0.0597,
        },
        hf_subprefix="second_chassis",
        output_subdir="second-chassis-dose-replication",
    ),
}


def chassis_for_slug(cell_slug: str) -> ChassisConfig:
    """Resolve the chassis from its (unique) new-cell slug — fail-loud.

    The dispatcher's subprocess command carries ``--cell <new_slug>``; the
    per-cell runner recovers the full chassis config from it, so the #600
    ``_run_cells_subprocess`` command shape needs no new flag.
    """
    by_slug = {c.new_slug: c for c in CHASSES.values()}
    if cell_slug not in by_slug:
        raise KeyError(
            f"unknown #610 cell slug {cell_slug!r}; registered: {sorted(by_slug)} "
            f"(chassis registry: {sorted(CHASSES)})"
        )
    return by_slug[cell_slug]


# ── Round-1 (mercenary) module-level constants, RE-BOUND from the registry so
# they stay byte-equivalent (v2 plan §4.1; pinned by the existing #610 tests).
_MERCENARY = CHASSES["mercenary"]
RUN_NAME_PREFIX = _MERCENARY.run_name_prefix
NEW_SLUG = _MERCENARY.new_slug
NEW_PLAIN_NAME = _MERCENARY.new_plain_name
CHASSIS_SLUG = _MERCENARY.chassis_slug
CHASSIS_TARGET = _MERCENARY.chassis_target

# ── The single manipulated variable (plan §4.1). ─────────────────────────────
# journalist = the mercenary pair's matched-control persona (pre-registered in
# the task body; identity re-asserted against the parent manifest's ctrl slot
# in cells.build_610_spec).
REPLACEMENT_PERSONA = _MERCENARY.replacement

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
# realized villain ΔG range ± 2 nats (per chassis — see CHASSES). Outside it
# but inside gate (a)'s [5, 19] → proceed, flag in analysis (recorded, not
# gating). Round-1 mercenary value re-bound from the registry. ───────────────
CHASSIS_DG_SOFT_RANGE_NATS = _MERCENARY.dg_soft_range

# ── Registered reference points for the §6.4 sanity reads (parent #600
# committed values; the per-seed D_with comparators are RECOMPUTED from the
# parent trajectories by formula — these are the pre-registered cross-cell
# precedents quoted in plan §5/§6.4). The replacement's ctrl precedent is
# per-chassis (CHASSES); the assistant precedent stays GLOBAL (cross-chassis
# mechanism color only, v2 plan §4.1). ───────────────────────────────────────
JOURNALIST_CTRL_PRECEDENT = _MERCENARY.replacement_ctrl_precedent
ASSISTANT_TRAINED_SLOT_PRECEDENT = -0.193  # assistant trained-slot read, parent finding (c)

# ── Sanity personas (plan §5): trained in BOTH arms; cross-run drift
# detectors read at ±2× DECISION_BAND of their with-default-arm values.
# Per-chassis (CHASSES); round-1 mercenary set re-bound from the registry. ───
SANITY_PERSONAS = _MERCENARY.sanity_personas

# ── Compute accounting (plan §9). ────────────────────────────────────────────
GPU_HOURS_BUDGETED = 22  # instance-GPU-hours (4 × A100-80 × typical wall)
