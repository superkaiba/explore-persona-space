# ruff: noqa: RUF003  # Qwen marker token " ※" + em-dash are intentional
"""Task #514 — Resolve the full-FT matched-rate read with a non-collapsing FT regime.

Sibling of ``lora_vs_ft_508`` (parent #508). Inherits the contrastive marker-implant
recipe verbatim from #508 (source persona, 4 negatives, R_train pin, eval panel,
marker token id 83399, 1000-row contrastive set, ZeRO-3 FT recipe). The SINGLE
manipulated variable is the **FT training regime** — denser budgets in the
0.25-0.5 epoch window AND/OR a halved learning rate — to land at least one clean
full-FT cell with source ΔG > 9 nat so the matched-rate LoRA-vs-FT comparison
can be resolved (plan §3 H1).

Cell tuple shape (mirrors #508's ``CELL_SPECS`` layout): ``(arm, budget_label,
epoch_fraction, lr_override)``. ``arm`` is always ``ARM_FULLFT`` for this
experiment; the LoRA arm is NOT retrained (#508's LoRA curve is re-used as the
matched-rate reference).

Plan rows:
    - §4.1.2:  dense lever {0.30, 0.35, 0.40, 0.45} at LR=5e-6 (inherited).
    - §4.1.2:  lower-LR lever {0.5, 1.0} at LR=2e-6 (#508's FT_NAN_FALLBACK_LR
               constant; explicit per-cell override threaded into the dispatcher).
    - §4.1.4:  smoke IS sweep with --cells ft_dense_b30 (single-cell), same
               entrypoint as --cells ft_dense_b30,...,ft_lowlr_b100 (full sweep).
"""

from __future__ import annotations

from explore_persona_space.experiments.lora_vs_ft_508 import (
    ARM_FULLFT,
    BASE_MODEL,
    CONTRASTIVE_NEGATIVES,
    EXPECTED_MARKER_TOKEN_ID,
    FT_BATCH_SIZE_PER_DEVICE,
    FT_EFFECTIVE_BATCH,
    FT_GRAD_ACCUM,
    FT_LEARNING_RATE,
    FT_LR_SCHEDULER,
    FT_NAN_FALLBACK_LR,
    FT_NUM_GPU,
    FT_WARMUP_RATIO,
    FT_WEIGHT_DECAY,
    HELD_OUT_PERSONAS_15,
    MARKER_SEP,
    MARKER_TEXT,
    MATCHED_SLICE_BAND_NATS,
    MATCHED_SLICE_TARGET_NATS,
    MAX_LENGTH,
    MAX_NEW_TOKENS_GEN,
    NEG_EX_PER_PERSONA,
    POS_EX_PER_SOURCE,
    QWEN_IM_END_ID,
    SEED,
    SOURCE_PERSONA,
    SOURCE_SELF_FLOOR_NATS,
    SUBCEILING_HEADROOM_NATS,
)

__all__ = [
    "ABORT_HELD_OUT_GLOGPROB_MAX",
    "ABORT_SOURCE_RCOLLAPSE_THRESHOLD",
    "ARM_FULLFT",
    "BASE_MODEL",
    "BUDGETS_DENSE_LEVER",
    "BUDGETS_LOW_LR_LEVER",
    "CELL_SPECS_514",
    "CLEAN_CELL_BRACKETING_UPPER_NATS",
    "CONTRASTIVE_NEGATIVES",
    "DENSE_LEVER_CELLS",
    "EXPECTED_MARKER_TOKEN_ID",
    "FT_BATCH_SIZE_PER_DEVICE",
    "FT_EFFECTIVE_BATCH",
    "FT_GRAD_ACCUM",
    "FT_LEARNING_RATE",
    "FT_LR_SCHEDULER",
    "FT_NAN_FALLBACK_LR",
    "FT_NUM_GPU",
    "FT_WARMUP_RATIO",
    "FT_WEIGHT_DECAY",
    "HELD_OUT_PERSONAS_15",
    "HF_DATA_PREFIX_514",
    "LOW_LR",
    "LOW_LR_LEVER_CELLS",
    "MARKER_SEP",
    "MARKER_TEXT",
    "MATCHED_SLICE_BAND_NATS",
    "MATCHED_SLICE_TARGET_NATS",
    "MAX_LENGTH",
    "MAX_NEW_TOKENS_GEN",
    "NEG_EX_PER_PERSONA",
    "POS_EX_PER_SOURCE",
    "QWEN_IM_END_ID",
    "SEED",
    "SOURCE_PERSONA",
    "SOURCE_SELF_FLOOR_NATS",
    "SUBCEILING_HEADROOM_NATS",
    "WANDB_PROJECT_514",
    "cell_lever",
    "cell_lr",
    "lever_of",
    "resolve_cell_spec",
]

# ── Per-lever LR values (plan §4.1.2). ───────────────────────────────────────
# Dense lever inherits #508's FT_LEARNING_RATE=5e-6 verbatim (single-variable
# discipline — only the budget moves). Lower-LR lever uses #508's
# FT_NAN_FALLBACK_LR=2e-6 constant (explicitly defined in #508 as the
# canonical "halved LR" choice that retains the Tulu-3 grounding).
LOW_LR: float = FT_NAN_FALLBACK_LR  # 2e-6
assert LOW_LR == 2e-6, f"LOW_LR expected 2e-6, got {LOW_LR!r}"
assert FT_LEARNING_RATE == 5e-6, f"FT_LEARNING_RATE expected 5e-6, got {FT_LEARNING_RATE!r}"

# ── Per-lever epoch-fraction budgets (plan §4.1.2). ──────────────────────────
# Dense lever: 4 budgets at 1/8-epoch resolution wedged between #508's FT-light
# (0.25 epoch, source ΔG = 8.2 nat) and FT-middle (0.5 epoch, r-collapsed).
# Lower-LR lever: 2 budgets at half the LR — 0.5 + 1.0 epoch span the curve from
# "halved implicit budget" through "full epoch under halved LR".
BUDGETS_DENSE_LEVER: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45)
BUDGETS_LOW_LR_LEVER: tuple[float, ...] = (0.5, 1.0)

assert len(BUDGETS_DENSE_LEVER) == 4
assert len(BUDGETS_LOW_LR_LEVER) == 2
assert all(0.25 < b < 0.5 for b in BUDGETS_DENSE_LEVER), (
    f"Dense-lever budgets must sit strictly inside (0.25, 0.5): {BUDGETS_DENSE_LEVER}"
)

# ── Cell specs (plan §4.1.2 + §10 Reproducibility card). ─────────────────────
# Cell tuple = (arm, budget_label, epoch_fraction, lr_override).
# - arm is always ARM_FULLFT (the LoRA arm is reused from #508, NOT retrained).
# - budget_label is the SHORT slug suffix; the canonical cell_slug is the
#   ``cell_slug_514()`` of (arm, budget_label) which yields names like
#   ``ft_dense_b30`` / ``ft_lowlr_b50``.
# - epoch_fraction matches the budget label numerically (b30 = 0.30 epoch).
# - lr_override is float (not None); per-cell threading in dispatch_514.py
#   passes the cell's lr_override into train_one_cell_fullft(lr_override=...).
CellSpec514 = tuple[str, str, float, float]
CELL_SPECS_514: tuple[CellSpec514, ...] = (
    (ARM_FULLFT, "dense_b30", BUDGETS_DENSE_LEVER[0], FT_LEARNING_RATE),
    (ARM_FULLFT, "dense_b35", BUDGETS_DENSE_LEVER[1], FT_LEARNING_RATE),
    (ARM_FULLFT, "dense_b40", BUDGETS_DENSE_LEVER[2], FT_LEARNING_RATE),
    (ARM_FULLFT, "dense_b45", BUDGETS_DENSE_LEVER[3], FT_LEARNING_RATE),
    (ARM_FULLFT, "lowlr_b50", BUDGETS_LOW_LR_LEVER[0], LOW_LR),
    (ARM_FULLFT, "lowlr_b100", BUDGETS_LOW_LR_LEVER[1], LOW_LR),
)

DENSE_LEVER_CELLS: tuple[str, ...] = tuple(
    f"ft_{lbl}" for _, lbl, _, _ in CELL_SPECS_514 if lbl.startswith("dense_")
)
LOW_LR_LEVER_CELLS: tuple[str, ...] = tuple(
    f"ft_{lbl}" for _, lbl, _, _ in CELL_SPECS_514 if lbl.startswith("lowlr_")
)
assert len(DENSE_LEVER_CELLS) == 4
assert len(LOW_LR_LEVER_CELLS) == 2

# ── Abort-on-collapse decision thresholds (plan §4.1.3 + §11 New values). ────
# After cell 1 of each lever (the smallest budget in that lever) finishes
# Phase 2 eval, if BOTH conditions hold simultaneously, abort the remaining
# cells of THAT lever (the other lever is independent).
#   source_r_collapse_rate ≥ 0.50  (≥10/20 source probes r-collapsed)
#     AND
#   held_out_g_logprob_mean > −5.0 (held-out panel near ceiling)
# 50% / −5 nat are the #508-validated thresholds: #508's ft_b2 had 19/20 = 95%
# r-collapse (well above 50%) and ft_b1 had 0/20 r-collapse (well below). The
# −5 nat sub-ceiling gate is inherited from §10 Repro card.
ABORT_SOURCE_RCOLLAPSE_THRESHOLD: float = 0.50
ABORT_HELD_OUT_GLOGPROB_MAX: float = -5.0

# ── Clean-cell bracketing criterion (plan §6.2). ─────────────────────────────
# A "clean cell above 9 nat" must satisfy:
#   source_mean (ΔG) > CLEAN_CELL_BRACKETING_UPPER_NATS (= 9.0)
#   AND r_collapse_rate < ABORT_SOURCE_RCOLLAPSE_THRESHOLD (= 0.50)
#   AND held_out_g_logprob_mean ≤ ABORT_HELD_OUT_GLOGPROB_MAX (= -5.0)
CLEAN_CELL_BRACKETING_UPPER_NATS: float = 9.0

# ── HF data + WandB project (plan §10). ──────────────────────────────────────
# Same WandB project as #508 so the new cells show up alongside the LoRA
# reference + FT anchor runs (plan §4.1.3 + §10).
HF_DATA_PREFIX_514 = "issue514_full_ft_regime"
WANDB_PROJECT_514 = "lora_vs_ft_508"


def cell_slug_514(arm: str, budget_label: str) -> str:
    """Canonical USER-FACING cell slug for #514 (``ft_dense_b30`` etc.).

    Mirrors ``lora_vs_ft_508.cell_slug`` but uses the lever-prefixed budget
    labels (``dense_b30`` / ``lowlr_b50``). Internally ``ARM_FULLFT="fullft"``
    is used everywhere — only the CLI / cell-slug strings use the public ``ft_``
    short form.
    """
    if arm != ARM_FULLFT:
        raise ValueError(
            f"#514 only trains full-FT cells; got arm={arm!r} (expected {ARM_FULLFT!r})."
        )
    return f"ft_{budget_label}"


def lever_of(cell_slug: str) -> str:
    """Return ``"dense"`` or ``"lowlr"`` for a #514 cell slug.

    Used by ``--abort-on-collapse``: the abort decision is PER-LEVER (one
    lever's smallest-budget cell triggering the abort does NOT affect the
    other lever).

    Raises ``ValueError`` for slugs that don't belong to this experiment.
    """
    if cell_slug in DENSE_LEVER_CELLS:
        return "dense"
    if cell_slug in LOW_LR_LEVER_CELLS:
        return "lowlr"
    raise ValueError(
        f"Cell {cell_slug!r} is not a #514 cell; expected one of "
        f"{DENSE_LEVER_CELLS + LOW_LR_LEVER_CELLS}"
    )


def cell_lever(cell_slug: str) -> str:
    """Alias for :func:`lever_of` (kept for read-fluency in the dispatcher)."""
    return lever_of(cell_slug)


def resolve_cell_spec(cell_slug: str) -> CellSpec514:
    """Look up the (arm, budget_label, epoch_fraction, lr_override) tuple by slug.

    The cell slug is the USER-FACING form (``ft_dense_b30``); this function
    strips the ``ft_`` prefix to match the ``budget_label`` field of
    ``CELL_SPECS_514``.
    """
    if not cell_slug.startswith("ft_"):
        raise ValueError(
            f"Cell slug must start with `ft_` for #514; got {cell_slug!r}. "
            f"Valid slugs: {DENSE_LEVER_CELLS + LOW_LR_LEVER_CELLS}"
        )
    budget_label = cell_slug[len("ft_") :]
    for spec in CELL_SPECS_514:
        if spec[1] == budget_label:
            return spec
    raise ValueError(
        f"Unknown #514 cell slug {cell_slug!r}; valid: {DENSE_LEVER_CELLS + LOW_LR_LEVER_CELLS}"
    )


def cell_lr(cell_slug: str) -> float:
    """Return the per-cell LR override for a #514 cell slug.

    Helper for the dispatcher: each cell's LR is read from its spec tuple
    rather than from a single global ``--ft-lr-override`` flag, since the
    dense + lower-LR levers use different LRs in the same run.
    """
    return resolve_cell_spec(cell_slug)[3]
