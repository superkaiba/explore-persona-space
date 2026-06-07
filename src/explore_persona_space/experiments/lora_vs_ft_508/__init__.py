# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
"""Task #508 — LoRA vs full fine-tuning: does marker leakage to bystanders differ?

Single-variable comparison: LoRA-r16 vs full-FT, both arms under the inherited
#472 FALLBACK saturation-fix recipe (FALLBACK_LORA_R=16, FALLBACK_LEARNING_RATE=5e-6,
FALLBACK_EPOCHS=0.5 as the lora_b2 / ft_b2 anchor) plus 0.25-epoch and 1.0-epoch
budgets per arm to bracket the matched-rate 8 ± 1 nat target band. Headline DV is
the matched-rate gap in held-out mean ΔG (on-policy, per-cell base subtraction on
each cell's own trained R per MF1).

Module layout (plan §4.2 Path A):
    __init__              — canonical constants: marker, source, contrastive
                            negatives, eval panel, per-arm cell specs, paths.
    train_cell_fullft     — new full-FT cell trainer that builds an
                            `accelerate launch` command targeting the new thin
                            trainer (scripts/train_marker_fullft.py).
    eval_one_cell         — extends #472's per-cell on-policy DV with per-cell
                            base log P scoring on each cell's own trained R
                            (the MF1 fix).
    marker_dynamics_callback — HF TrainerCallback that fires every N steps on a
                            fixed in-training probe set, logs source +
                            bystander ΔG / emission trajectories to WandB.
    dispatch_508          — unified dispatcher: smoke IS sweep with 2 cells
                            (--cells lora_b2,ft_b2 --seeds 42); sweep is the
                            same dispatcher with 6 cells.
    analyze               — post-train: per-arm bracketing check, persona-cluster
                            crossed bootstrap, hero figure, trajectory figures,
                            H2/H3 secondaries.

What's reused (plan §4.1 — do NOT rebuild):
    - The contrastive recipe (200 positives × `villain` + 200 × 4 negative
      personas = 1000 rows) — assembled by reusing #472's
      `r_generate.py` + `build_training_data.py`.
    - `train/sft.py::train_lora` + `MarkerOnlyDataCollator(tail_tokens=0)` for
      the LoRA arm.
    - `eval_one_cell.score_logp_for_R` (from #472) for vLLM batched marker
      log-prob reads.
"""

from __future__ import annotations

# ── Inherit recipe constants from the #472 sibling (single source of truth). ──
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    FALLBACK_EPOCHS,
    FALLBACK_LEARNING_RATE,
    FALLBACK_LORA_R,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    MARKER_SEP,
    MARKER_TEXT,
    MATCHED_SLICE_BAND_NATS,
    MATCHED_SLICE_TARGET_NATS,
    MAX_LENGTH,
    MAX_NEW_TOKENS_GEN,
    SOURCE_SELF_FLOOR_NATS,
    SUBCEILING_HEADROOM_NATS,
)

# Re-export for module-local imports without dotted access in downstream files.
__all__ = [
    "ARM_FULLFT",
    "ARM_LORA",
    "BASE_MODEL",
    "BUDGETS_DEFAULT",
    "BUDGETS_FALLBACK_OVERTRAIN",
    "BUDGETS_FALLBACK_UNDERTRAIN",
    "CELL_SPECS",
    "CONTRASTIVE_NEGATIVES",
    "DYNAMICS_BYSTANDER_PERSONAS",
    "DYNAMICS_CADENCE_STEPS",
    "DYNAMICS_PROBES_PATH",
    "DYNAMICS_PROBE_QUESTIONS_PER_PERSONA",
    "EXPECTED_MARKER_TOKEN_ID",
    "FALLBACK_EPOCHS",
    "FALLBACK_LEARNING_RATE",
    "FALLBACK_LORA_R",
    "FT_BATCH_SIZE_PER_DEVICE",
    "FT_GRAD_ACCUM",
    "FT_LEARNING_RATE",
    "FT_LR_SCHEDULER",
    "FT_NAN_FALLBACK_LR",
    "FT_NUM_GPU",
    "FT_WARMUP_RATIO",
    "FT_WEIGHT_DECAY",
    "HELD_OUT_PERSONAS_15",
    "HF_DATA_PREFIX",
    "HF_DATA_REPO",
    "HF_MODEL_REPO",
    "LORA_ALPHA",
    "LORA_BATCH_SIZE",
    "LORA_DROPOUT",
    "LORA_GRAD_ACCUM",
    "LORA_LR",
    "LORA_R",
    "LORA_WARMUP_RATIO",
    "MARKER_SEP",
    "MARKER_TEXT",
    "MATCHED_SLICE_BAND_NATS",
    "MATCHED_SLICE_TARGET_NATS",
    "MAX_LENGTH",
    "MAX_NEW_TOKENS_GEN",
    "POS_EX_PER_SOURCE",
    "QWEN_IM_END_ID",
    "SEED",
    "SOURCE_PERSONA",
    "SOURCE_SELF_FLOOR_NATS",
    "SUBCEILING_HEADROOM_NATS",
    "WANDB_PROJECT",
    "cell_slug",
    "is_lora_arm",
    "load_q_eval",
    "load_q_train",
]

# ── Arms (the SOLE manipulated variable). ────────────────────────────────────
ARM_LORA = "lora"
ARM_FULLFT = "fullft"
ARMS = (ARM_LORA, ARM_FULLFT)

# ── Source + contrastive negatives (plan §4.1, §4.3). ────────────────────────
SOURCE_PERSONA = "villain"

# The 4-negative contrastive set used in BOTH arms — qwen_default ALWAYS
# included per .claude/rules/contrastive-negatives.md (safety target). Identical
# data file across arms; single-variable rule.
CONTRASTIVE_NEGATIVES = ("medical_doctor", "police_officer", "qwen_default", "comedian")

# Positive rows per cell. Inherited from #472's POS_EX_PER_SOURCE=200; with 4
# negative personas × 200 ex/persona = 800 negatives → 1000 training rows total
# at a ~1:4 positive-to-total-negative ratio (plan §4.3 — kept identical across
# arms because the variable is LoRA-vs-FT, not the contrastive ratio).
POS_EX_PER_SOURCE = 200
NEG_EX_PER_PERSONA = 200

# ── Held-out 15-persona panel (plan §4.1). ───────────────────────────────────
# Frozen across arms; the union of #448's 12 never-trained-as-negative personas
# + 3 SHA-derived held-outs. NEVER a contrastive negative; NEVER a source.
HELD_OUT_PERSONAS_15: tuple[str, ...] = (
    "accountant",
    "ai",
    "ai_assistant",
    "chef",
    "child",
    "hero",
    "journalist",
    "lawyer",
    "philosopher",
    "programmer",
    "surgeon",
    "wizard",
    "assistant",
    "data_scientist",
    "kindergarten_teacher",
)
assert len(HELD_OUT_PERSONAS_15) == 15, (
    f"Expected 15 held-out personas, got {len(HELD_OUT_PERSONAS_15)}"
)
# Held-out + contrastive sets must be disjoint (no panel leak).
_overlap = set(HELD_OUT_PERSONAS_15) & set(CONTRASTIVE_NEGATIVES)
assert not _overlap, f"Held-out x contrastive overlap: {sorted(_overlap)}"
assert SOURCE_PERSONA not in HELD_OUT_PERSONAS_15, (
    f"Source persona {SOURCE_PERSONA!r} must NOT appear in the held-out panel"
)

# ── LoRA arm recipe (#472 FALLBACK triple + inherited main-recipe constants). ─
# Plan §11 LoRA-arm row + §12 LoRA-arm decision rationale.
LORA_R = FALLBACK_LORA_R  # 16
LORA_ALPHA = 2 * LORA_R  # 32 — α=2r inferred from #472's LORA_ALPHA = 2*LORA_R convention.
LORA_DROPOUT = 0.05
LORA_LR = FALLBACK_LEARNING_RATE  # 5e-6
LORA_WARMUP_RATIO = 0.05
LORA_BATCH_SIZE = 4
LORA_GRAD_ACCUM = 4

# ── Full-FT arm recipe (Tulu 3 8B SFT — arXiv:2411.15124v5). ─────────────────
# Plan §11 Full-FT-arm row + §12 Full-FT-arm decision rationale. NOT shared with
# LoRA constants because the optimization regimes differ (linear vs cosine,
# warmup ratio, batch shape). lr=5e-6 matches LoRA lr by construction (the LoRA
# arm picked the lower-of-two candidates so this equivalence holds).
FT_LEARNING_RATE = 5e-6
FT_LR_SCHEDULER = "linear"  # Tulu 3 linear (NOT LoRA's cosine).
FT_WARMUP_RATIO = 0.03
FT_WEIGHT_DECAY = 0.0
FT_BATCH_SIZE_PER_DEVICE = 1
FT_GRAD_ACCUM = 16
FT_NUM_GPU = 4
# Effective batch = 1 × 16 × 4 = 64 (downscaled /2 from Tulu 3's eff=128 on
# 8-node × 8-GPU = 64-GPU cluster).
FT_EFFECTIVE_BATCH = FT_BATCH_SIZE_PER_DEVICE * FT_GRAD_ACCUM * FT_NUM_GPU
assert FT_EFFECTIVE_BATCH == 64

# NaN-divergence smoke fallback per plan §4.5 gate 4 + §8 risks.
FT_NAN_FALLBACK_LR = 2e-6

# ── Per-arm epoch-fraction budgets (plan §4.4). ──────────────────────────────
# Three budgets per arm bracketing the 8 ± 1 nat source-self ΔG target band.
# Smoke phase (§4.5) re-validates these; if either b2 cell misses the 7-9 nat
# band, the §4.4 epoch-fraction shift fires for that arm:
#   under-trains (source ΔG < 7) → BUDGETS_FALLBACK_UNDERTRAIN
#   over-trains  (source ΔG > 9) → BUDGETS_FALLBACK_OVERTRAIN
BUDGETS_DEFAULT: tuple[float, ...] = (0.25, 0.5, 1.0)
BUDGETS_FALLBACK_UNDERTRAIN: tuple[float, ...] = (0.5, 1.0, 2.0)
BUDGETS_FALLBACK_OVERTRAIN: tuple[float, ...] = (0.125, 0.25, 0.5)

# ── Cell specs (plan §4.4). ──────────────────────────────────────────────────
# (arm, budget_label, epoch_fraction). Slug = f"{arm}_b{i}" so
# lora_b1 / lora_b2 / lora_b3 / ft_b1 / ft_b2 / ft_b3 — 6 trained cells + 1
# base-only eval.
CellSpec = tuple[str, str, float]
CELL_SPECS: tuple[CellSpec, ...] = (
    (ARM_LORA, "b1", BUDGETS_DEFAULT[0]),
    (ARM_LORA, "b2", BUDGETS_DEFAULT[1]),
    (ARM_LORA, "b3", BUDGETS_DEFAULT[2]),
    (ARM_FULLFT, "b1", BUDGETS_DEFAULT[0]),
    (ARM_FULLFT, "b2", BUDGETS_DEFAULT[1]),
    (ARM_FULLFT, "b3", BUDGETS_DEFAULT[2]),
)


def cell_slug(arm: str, budget_label: str) -> str:
    """Canonical cell slug: ``{arm}_{budget_label}`` (e.g. ``lora_b2``).

    The dispatcher accepts these as --cells comma-separated.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")
    return f"{arm}_{budget_label}"


def is_lora_arm(slug: str) -> bool:
    """Return True when ``slug`` belongs to the LoRA arm (cell_slug ⇒ arm)."""
    return slug.startswith(f"{ARM_LORA}_")


# ── Seed (plan §11). Single seed 42 inherited from #448/#472. ────────────────
SEED = 42

# ── In-training MarkerDynamicsCallback config (plan §4.2 MF2). ───────────────
# Probe set: 1 source × 5 questions + 3 bystanders × 5 questions = 20 probes.
# Bystanders drawn from the 3 SHA-derived held-outs (plan §4.2 — distinct from
# the 12 never-trained-as-negative subset the headline DV averages over).
DYNAMICS_PROBE_QUESTIONS_PER_PERSONA = 5
DYNAMICS_BYSTANDER_PERSONAS: tuple[str, ...] = (
    "kindergarten_teacher",
    "data_scientist",
    "assistant",
)
# Cadence: every-4-steps for both arms (smoke gate 6 halves to every-2-steps
# per arm if <4 snapshots observed on b2). `ungrounded — needs smoke-test`.
DYNAMICS_CADENCE_STEPS = 4
DYNAMICS_PROBES_PATH = "data/issue_508/dynamics_probes.json"

# ── Tokenization (plan §11). ─────────────────────────────────────────────────
# Qwen-2.5 `<|im_end|>` token id (post-response slot). NOT used as the DV slot
# this experiment (per §12 — the flag-gated suppress_at_post_response_slot=False
# is the inherited #472 default; this experiment varies LoRA-vs-FT, not the
# slot fix). Documented here so the assertion is in one place.
QWEN_IM_END_ID = 151645

# ── Paths + repos (plan §11). ────────────────────────────────────────────────
HF_DATA_PREFIX = "issue508_lora_vs_ft"
WANDB_PROJECT = "lora_vs_ft_508"

# ── Train/eval question split (re-use #472's 10/10 disjoint split). ──────────


def load_q_train() -> list[str]:
    """Return the 10-question Q_train subset (first half of EVAL_QUESTIONS_20)."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    return list(EVAL_QUESTIONS_20[:10])


def load_q_eval() -> list[str]:
    """Return the 10-question Q_eval subset (second half of EVAL_QUESTIONS_20).

    NOTE: the plan §4.6 specifies 20 questions per held-out persona. With the
    10-question Q_eval split, that matches Q_eval used WITH question replication
    OR Q_eval+Q_train re-merged into a 20-question eval pool. We use the latter
    here — eval probes the model on all 20 questions to give the 300 ΔG values
    per cell the plan calls for (15 personas × 20 questions). Train-vs-eval
    disjointness is preserved by the persona axis (the 15 held-out personas were
    never seen during training).
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    return list(EVAL_QUESTIONS_20)
