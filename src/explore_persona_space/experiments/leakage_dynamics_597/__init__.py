# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Task #597 — training dynamics of source implantation vs bystander leakage.

Positive-only vs contrastive training at a matched #480 recipe, with the FULL
24-persona panel + the bare ``no_persona`` context probed at every checkpoint
of both arms under the four-float storage contract (log P(marker), z_marker,
z_eos, logZ; trained AND base per slot).

Arms (plan §3+4):

  Arm A — contrastive: NO new training; post-hoc panel probe over the existing
          #480 capend checkpoint ladders
          (``adapters/issue_480_band_stop/<source>_seed42_capend/checkpoint-N``,
          N ∈ {20..520:20} ∪ {528}, 27 per source).
  Arm B — positive-only: fresh per-source training on the order-preserving
          200-positive filter of #480's 700-row pools (negatives REMOVED, not
          replaced), ``max_steps=528`` matching Arm A's exact cosine schedule,
          checkpoints on the B_GRID (39 per source).

Modules:
    build_pos_only_pool — Phase P order-preserving 200-positive filter + the
                          BLOCKING token-id probe-row identity assert.
    probe_rows          — Phase 0 vLLM probe-row generation (25 contexts × 50
                          held-out questions, base greedy R, cap 1024).
    panel_probe         — per-checkpoint four-float HF forward-pass panel probe
                          (gauge assert, end-of-ladder hot-swap invariant,
                          checkpoint-per-phase persistence).
    smoke_gate          — the hard #534 Gate S: off-line eval path must
                          reproduce #480's in-loop band-stop read within 1 nat.
    emission_anchors    — vLLM multi-LoRA on-policy emission anchors
                          (max_new_tokens=2048) at sparse anchor steps.
    grid_callbacks      — ``CheckpointGridPruneCallback`` keeping only B_GRID
                          checkpoint dirs under ``save_steps=4``.

Single manipulated variable: the training-mix regime. Everything else is
matched to the realized #480 band-stop config and enforced by the dispatcher's
adapter-config parity preflight (``scripts/issue_597/
dispatch_leakage_dynamics_597.py``).
"""

from __future__ import annotations

from explore_persona_space.experiments.marker_implant_480 import (
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
    SOURCE_PERSONAS,
)

__all__ = [
    "ANCHOR_STEPS",
    "ARM_A_HF_ADAPTER_ROOT",
    "ARM_B_HF_ADAPTER_ROOT",
    "ARM_C_HALT_STEP",
    "ARM_C_HF_ADAPTER_ROOT",
    "ARM_C_SAVE_STEPS",
    "A_GRID",
    "BASE_MODEL",
    "B_GRID",
    "C_GRID",
    "EXPECTED_POS_ROWS",
    "HF_597_DATA_SUBDIR",
    "HF_DATA_REPO",
    "HF_MODEL_REPO",
    "IM_END_ID",
    "MARKER_ID",
    "MARKER_TEXT",
    "NO_PERSONA_KEY",
    "SEED",
    "SOURCE_PERSONAS",
    "TRAINED_NEGATIVES",
    "WANDB_PROJECT",
    "probe_contexts_25",
]

BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
SEED: int = 42
HF_MODEL_REPO: str = "superkaiba1/explore-persona-space"
HF_DATA_REPO: str = "superkaiba1/explore-persona-space-data"
HF_597_DATA_SUBDIR: str = "issue597_leakage_dynamics"
WANDB_PROJECT: str = "issue597-leakage-dynamics"

# Arm A reuse (plan §10): the #480 band-stop capend ladders, Hub-verified
# 2026-06-11 (6 sources × 27 checkpoints).
ARM_A_HF_ADAPTER_ROOT: str = "adapters/issue_480_band_stop"
# Arm B output root on the HF model repo.
ARM_B_HF_ADAPTER_ROOT: str = "adapters/issue_597_pos_only"

# Order-preserving positive filter expectation (plan Phase P; verified
# plan-time on villain: 200 of 700 rows end with " ※").
EXPECTED_POS_ROWS: int = 200
EXPECTED_FULL_POOL_ROWS: int = 700

# Checkpoint grids (plan §10 Reproducibility Card):
#   Arm A: {20..520:20} ∪ {528} — the 27 existing capend checkpoints.
#   Arm B: {4..60:4} ∪ {80..520:20} ∪ {528} — 39 checkpoints (15 + 23 + 1);
#          the 4-step early grid puts ≥3 points inside any ~12-step install
#          window (#533/#547) and brackets the matched-positive-dose points.
A_GRID: tuple[int, ...] = tuple(sorted(set(range(20, 521, 20)) | {528}))
B_GRID: tuple[int, ...] = tuple(sorted(set(range(4, 61, 4)) | set(range(80, 521, 20)) | {528}))
assert len(A_GRID) == 27, f"A_GRID must have 27 steps, got {len(A_GRID)}: {A_GRID}"
assert len(B_GRID) == 39, f"B_GRID must have 39 steps, got {len(B_GRID)}: {B_GRID}"
# Every B_GRID step must be reachable by save_steps=4 (Trainer only saves at
# multiples of save_steps; a non-multiple would silently never materialize).
assert all(s % 4 == 0 for s in B_GRID), B_GRID

# On-policy emission anchor steps — symmetric across arms (consistency-checker
# cosmetic fix 2: Arm A's checkpoint-20 exists, so both arms anchor at the
# same 6 steps).
ANCHOR_STEPS: tuple[int, ...] = (20, 40, 100, 200, 400, 528)
assert set(ANCHOR_STEPS) <= set(A_GRID), (ANCHOR_STEPS, A_GRID)
assert set(ANCHOR_STEPS) <= set(B_GRID), (ANCHOR_STEPS, B_GRID)

# The 25th probe context: the bare no-system-prompt chat (open-q 3.7 — the
# trained default-context negative in Arm A, untouched in Arm B).
NO_PERSONA_KEY: str = "no_persona"

# Per-source trained-negative panel personas (plan Phase P table, derived from
# the Hub-verified issue480 bystander_assignment.json). The dispatcher
# re-derives this map from the downloaded assignment at preflight and
# FAILS LOUD on any drift from this constant (exact set equality per source);
# the off-pod Phase A analysis (analyze.py) consumes the constant so the
# trained-negative vs held-out split needs no HF download on the VM.
TRAINED_NEGATIVES: dict[str, tuple[str, str]] = {
    "villain": ("police_officer", "medical_doctor"),
    "comedian": ("medical_doctor", "assistant"),
    "assistant": ("software_engineer", "comedian"),
    "qwen_default": ("comedian", "data_scientist"),
    "software_engineer": ("assistant", "medical_doctor"),
    "kindergarten_teacher": ("software_engineer", "french_person"),
}
assert set(TRAINED_NEGATIVES) == set(SOURCE_PERSONAS), (
    sorted(TRAINED_NEGATIVES),
    sorted(SOURCE_PERSONAS),
)
# Per-cell adapters are single-source; the cell's own source must never be in
# its trained-negative set (contrastive-negatives disjointness invariant).
assert all(src not in negs for src, negs in TRAINED_NEGATIVES.items()), TRAINED_NEGATIVES

# Arm B training schedule (plan §10): 528 optimizer steps matching Arm A's
# 12 epochs × 44 steps, via the new TrainLoraConfig.max_steps field.
ARM_B_MAX_STEPS: int = 528
ARM_B_SAVE_STEPS: int = 4

# ── Arm C: dense-early contrastive retrain (#597 follow-up
# `dense-early-contrastive-grid`, plan v3 §3) ────────────────────────────────
# Fresh retrain of the 6 contrastive cells on the SAME 700-row pools with a
# dense early checkpoint grid: every 2 steps through 40, every 4 through 60,
# save-driven halt after the step-60 save (max_steps stays 528 so the cosine
# + warmup schedule is identical to #480 / Arm A for steps 1–60).
C_GRID: tuple[int, ...] = tuple(sorted(set(range(2, 41, 2)) | set(range(44, 61, 4))))
assert len(C_GRID) == 25 and all(s % 2 == 0 for s in C_GRID), C_GRID
ARM_C_SAVE_STEPS: int = 2  # every C_GRID step reachable (Trainer saves at multiples)
ARM_C_HALT_STEP: int = 60  # == max(C_GRID)
# Every C_GRID step must be reachable by save_steps=2 (B_GRID's %4 assert,
# mirrored as %2 — a non-multiple would silently never materialize), and the
# save-driven halt must sit exactly on the last grid point.
assert all(s % ARM_C_SAVE_STEPS == 0 for s in C_GRID), C_GRID
assert max(C_GRID) == ARM_C_HALT_STEP, (ARM_C_HALT_STEP, max(C_GRID))
assert ARM_C_HALT_STEP % ARM_C_SAVE_STEPS == 0, (ARM_C_HALT_STEP, ARM_C_SAVE_STEPS)
# Arm C output root on the HF model repo (plan v3 §3 Outputs).
ARM_C_HF_ADAPTER_ROOT: str = "adapters/issue_597_contrastive_dense"


def probe_contexts_25() -> dict[str, str]:
    """The 25 probe contexts: ``EVAL_PERSONAS_24`` + the bare ``no_persona`` chat.

    Returns:
        ``{context_name: system_prompt}`` with the no-persona context mapped to
        the EMPTY STRING (the convention every #480 prompt builder uses: an
        empty system prompt means "emit no system message"). Exactly 25 keys.

    Asserts the panel import still carries 24 personas and that
    ``no_persona`` does not collide with a panel name.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    assert len(EVAL_PERSONAS_24) == 24, len(EVAL_PERSONAS_24)
    assert NO_PERSONA_KEY not in EVAL_PERSONAS_24, NO_PERSONA_KEY
    contexts = {**EVAL_PERSONAS_24, NO_PERSONA_KEY: ""}
    assert len(contexts) == 25, len(contexts)
    return contexts
