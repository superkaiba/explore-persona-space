"""Shared constants and helpers for issue #538 (band [14,20] follow-up of #527).

Plan v1 — strict single-variable follow-up of #527: band-stop window
[5, 12] → [14, 20] nat, with epochs cap raised 8 → 24 as the instrumental
change required for the band to be reachable. Every other choice
(sources, seeds, recipe, negative panel, eval stack, DV/GD definitions,
extraction layer, marker token) is inherited verbatim from #527.

Read-vs-write namespace split (plan §4):

- READ prefixes (inherited inputs from #527) stay ``issue_527``:
  ``HF_R_PATH_PREFIX``, ``HF_PAIR_SELECTION_PATH_PREFIX``, and the hash-gate
  target ``HF_TRAIN_MIX_READ_PATH_PREFIX``. Preflight verifies the
  byte-identical regeneration of ``training_mixes/`` against the parent's
  HF dataset path.
- WRITE prefixes (new artifacts produced by this task) are all
  ``issue_538``: ``HF_TRAIN_MIX_PATH_PREFIX``, ``HF_TRAJECTORY_PATH_PREFIX``,
  ``HF_ADAPTER_PATH_PREFIX``, ``LOCAL_OUT_DIR``, sentinel name.

Sub-modules — copies of ``experiments/issue_527/`` with module imports
re-pointed at ``experiments/issue_538`` so the namespace switch is
mechanical:

- ``persona_registry.py`` — load + assert-resolve persona bank.
- ``data_build.py``       — positives/negatives JSONL builders.
- ``shift_extract.py``    — L20 residual-stream shift-vector extraction
                            (extended with the ``marker_slot_stats`` block
                            per plan §6 "Marker-slot storage contract").
- ``analysis.py``         — DV1-DV5 + GD1/GD2/GD3 (pure numpy/scipy).
"""

# math/scientific notation in docstrings

from __future__ import annotations

from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Model + tokens (canonical; assert at preflight). Identical to #527.
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL: Final[str] = "Qwen/Qwen2.5-7B-Instruct"

# Plan §11 / `.claude/rules/marker-leakage-measurement.md`:
# ` ※` (leading space, Qwen-2.5-7B token id 83399). NOT bare `※` (id 63680).
MARKER_TEXT: Final[str] = " ※"
MARKER_ID: Final[int] = 83399

# Qwen-2.5-7B-Instruct chat-template terminator.
IM_END_ID: Final[int] = 151645

# Canonical persona-cosine layer in this project (#207 / #311 / #341 / #520).
EXTRACTION_LAYER: Final[int] = 20

# Qwen-2.5-7B hidden size (asserted at extraction time).
HIDDEN_SIZE: Final[int] = 3584

# ─────────────────────────────────────────────────────────────────────────────
# Persona pool — inherited verbatim from #527.
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_POOL_19: Final[tuple[str, ...]] = (
    "paramedic",
    "surgeon",
    "poet",
    "navy_seal",
    "army_medic",
    "florist",
    "cybersec_consultant",
    "pentester",
    "private_investigator",
    "librarian",
    "software_engineer",
    "data_scientist",
    "medical_doctor",
    "kindergarten_teacher",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
)

# 4-persona contrastive negative panel BASE — inherited from #527's panel
# choice, but #538 resolves it PER-PAIR via ``negative_panel_for_pair`` to
# remove the #527 source/negative overlap that contaminated pair 2 (see the
# task #538 21:27Z ``epm:concern-raised`` marker). For pair-1
# (florist x medical_doctor) the base panel is unchanged (no overlap). For
# pair-2 (librarian x police_officer) ``librarian`` is swapped for
# ``NEGATIVE_PANEL_REPLACEMENT`` so the same persona is no longer trained
# with positive AND negative marker objectives 4:1 in the same cell.
NEGATIVE_PANEL_4: Final[tuple[str, ...]] = (
    "assistant",
    "librarian",
    "programmer",
    "chef",
)

# Replacement persona used when a base-panel member equals a realized source
# in the current pair. User-suggested + orchestrator-bound: resolves in
# persona_bank, is not a realized source for either pair, has an existing
# R_persona JSON on disk, and is semantically near librarian (near-twin
# negatives are the sharper lever per `.claude/rules/contrastive-negatives.md`).
NEGATIVE_PANEL_REPLACEMENT: Final[str] = "kindergarten_teacher"


def negative_panel_for_pair(pair_a: str, pair_b: str) -> tuple[str, ...]:
    """Resolve the 4-persona contrastive negative panel for ONE training pair.

    Returns the base ``NEGATIVE_PANEL_4`` with any member that equals a
    realized source (``pair_a`` or ``pair_b``) swapped for
    ``NEGATIVE_PANEL_REPLACEMENT``. Hard-asserts the resolved panel:

    1. ``"assistant"`` is preserved (the bare default assistant is the
       highest-value negative, per `.claude/rules/contrastive-negatives.md`).
    2. The panel has exactly 4 unique members.
    3. The panel does not intersect ``{pair_a, pair_b}`` — fails LOUD
       (``AssertionError``) per the task #538 user mandate, never a silent
       skip / cascade. This is the executable proof that the #527
       contamination is gone.
    4. ``NEGATIVE_PANEL_REPLACEMENT`` itself is not a realized source; if it
       ever were (e.g. future re-pairing surfaces a pair that includes
       ``kindergarten_teacher``) the function fails LOUD instead of
       cascading to a second fallback.

    Parameters
    ----------
    pair_a, pair_b
        The two source-persona names for this training pair.

    Returns
    -------
    tuple[str, ...]
        4-tuple of negative-persona names, preserving the base panel's
        ordering for swapped-out positions.

    Raises
    ------
    AssertionError
        Any of the post-conditions above fails — the run aborts at build
        time, never silently proceeds with a contaminated mix.
    """
    sources = {pair_a, pair_b}
    if NEGATIVE_PANEL_REPLACEMENT in sources:
        raise AssertionError(
            f"NEGATIVE_PANEL_REPLACEMENT={NEGATIVE_PANEL_REPLACEMENT!r} is itself "
            f"a realized source for this pair ({pair_a!r}, {pair_b!r}); refusing "
            "to cascade to a second fallback. Pick a different replacement persona "
            "in src/explore_persona_space/experiments/issue_538/__init__.py."
        )
    panel = tuple(
        NEGATIVE_PANEL_REPLACEMENT if name in sources else name for name in NEGATIVE_PANEL_4
    )
    if "assistant" not in panel:
        raise AssertionError(
            f"resolved panel {panel!r} dropped 'assistant'; the bare default "
            "assistant context is the highest-value negative and MUST stay in "
            "every per-pair panel."
        )
    if len(set(panel)) != 4:
        raise AssertionError(
            f"resolved panel {panel!r} has fewer than 4 unique members "
            f"(base panel collided with sources {sources!r} on >1 slot, or "
            f"replacement {NEGATIVE_PANEL_REPLACEMENT!r} duplicates an "
            "existing base-panel member)."
        )
    if set(panel) & sources:
        raise AssertionError(
            f"resolved panel {panel!r} still intersects realized sources "
            f"{sources!r} after replacement — the per-pair panel fix did NOT "
            "remove the source/negative overlap. This is the #527 contamination "
            "the task #538 21:27Z epm:concern-raised marker flagged; refusing "
            "to build a contaminated training mix."
        )
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Persona-registry source-of-truth (plan §4 Inputs)
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_BANK_PATH: Final[str] = "data/issue_472/persona_bank.json"

HF_DATA_REPO: Final[str] = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO: Final[str] = "superkaiba1/explore-persona-space"

# READ prefixes (inherited from #527). R_persona, pair selection, and the
# hash-gate target for training_mixes ALL live under the parent's namespace
# on HF and on the branch. The preflight hash gate compares this task's
# regenerated mixes against the issue_527 HF copies for byte-identical
# determinism (Assumption #5).
HF_R_PATH_PREFIX: Final[str] = "issue_527/R_persona"
HF_PAIR_SELECTION_PATH_PREFIX: Final[str] = "issue_527/pair_selection"
HF_TRAIN_MIX_READ_PATH_PREFIX: Final[str] = "issue_527/training_mixes"
# Hash-gate target dataset revision (frozen at task creation; the parent's
# published-mix revision per plan §4 Inputs).
HF_TRAIN_MIX_READ_REVISION: Final[str] = "e6e163ce2a58108cc2c2d530f5f0ea9ef4542f65"

# WRITE prefixes (new artifacts; nothing overwrites #527).
HF_TRAIN_MIX_PATH_PREFIX: Final[str] = "issue_538/training_mixes"
HF_TRAJECTORY_PATH_PREFIX: Final[str] = "issue_538/trajectories"
HF_ADAPTER_PATH_PREFIX: Final[str] = "adapters/issue_538"

# ─────────────────────────────────────────────────────────────────────────────
# Training recipe — every value inherited from #527 EXCEPT
# RECIPE_BAND_LOW_NATS / RECIPE_BAND_HIGH_NATS / RECIPE_EPOCHS_CAP
# (the one experimental variable + the instrumental epochs raise).
# ─────────────────────────────────────────────────────────────────────────────

RECIPE_LORA_R: Final[int] = 16
RECIPE_LORA_ALPHA: Final[int] = 32
RECIPE_LORA_DROPOUT: Final[float] = 0.0
RECIPE_LORA_TARGETS: Final[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")
RECIPE_LR_PRIMARY: Final[float] = 5e-6
# Plan §4: NO autonomous lr retry path at the new band — the recipe forbids
# raising lr past 5e-6. Kept here for ref but the pipeline does NOT retry.
RECIPE_WARMUP_RATIO: Final[float] = 0.03

# *** The one experimental variable + the instrumental epoch raise: ***
RECIPE_EPOCHS_CAP: Final[int] = 24
RECIPE_BAND_LOW_NATS: Final[float] = 14.0
RECIPE_BAND_HIGH_NATS: Final[float] = 20.0

RECIPE_PER_DEVICE_BATCH: Final[int] = 4
RECIPE_GRAD_ACCUM: Final[int] = 4
RECIPE_MAX_LENGTH: Final[int] = 2048

# Plan §0 / §11: seeds — inherited verbatim from #527.
SEEDS: Final[tuple[int, ...]] = (42, 137, 256)

# Plan §4 contrastive-negatives section: per-arm positive counts.
N_POSITIVES_SINGLETON: Final[int] = 400
N_POSITIVES_JOINT: Final[int] = 800

# ─────────────────────────────────────────────────────────────────────────────
# Eval recipe — inherited from #527.
# ─────────────────────────────────────────────────────────────────────────────

EVAL_N_PROMPTS_PER_PERSONA: Final[int] = 20
EVAL_N_SAMPLES_PER_PROMPT: Final[int] = 1
EVAL_MAX_NEW_TOKENS: Final[int] = 2048

# ─────────────────────────────────────────────────────────────────────────────
# Output / sentinel paths — new namespace.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_OUT_DIR: Final[str] = "eval_results/issue_538"
SENTINEL_PATH_TEMPLATE: Final[str] = "/workspace/logs/issue-538-{kind}-{epoch}.json"

# WandB project (plan §4 Pipeline #5).
WANDB_PROJECT: Final[str] = "issue_538_superposition_followup"
