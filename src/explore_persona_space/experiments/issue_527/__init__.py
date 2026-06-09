"""Shared constants and helpers for issue #527 (pillar-2 superposition re-run).

Plan v1 §0/§4 — hot band-stopped anchor + orthogonal source pairs.

Single source of truth for:
- BASE_MODEL, MARKER_TEXT, MARKER_ID, IM_END_ID, L20 extraction layer.
- Recipe defaults (rsLoRA r=16 / α=32 / attn-only / lr=5e-6 / 8-epoch cap /
  marker_band_stop=True / band [5,12] nat / strict 1:1 pos:neg).
- The 19-persona #311 pool + the 4 contrastive-negative panel (plan §4 (c)).
- Persona-registry loader against `data/issue_472/persona_bank.json` (the
  authoritative registry — plan §4 Inputs).
- The 400-question pool source-of-truth resolver (#311 / #520 reuse).

Sub-modules:
- ``persona_registry.py`` — load + assert-resolve persona bank.
- ``data_build.py``       — positives/negatives JSONL builders.
- ``shift_extract.py``    — L20 residual-stream shift-vector extraction.
- ``analysis.py``         — DV1-DV5 + GD1/GD2/GD3 (pure numpy/scipy).
"""

# ruff: noqa: RUF002, RUF003  # math/scientific notation in docstrings

from __future__ import annotations

from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Model + tokens (canonical; assert at preflight)
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
# Persona pool
# ─────────────────────────────────────────────────────────────────────────────

# Plan §4 Inputs — the #311 19-persona panel pulled from
# `data/issue_472/persona_bank.json`. KEEP IN SYNC with the planner's
# §4 Inputs list; preflight asserts every one of these resolves in the
# persona_bank.json "personas" dict.
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
    "helpful_assistant",
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

# Plan §3 correction (c) + §11: contrastive negative panel. The 4 negatives
# shared across every trained arm at strict 1:1 positives-to-total-negatives.
# Plan §12 #6 RESOLUTION: "literal default_assistant" is encoded as the
# `"assistant"` key in `persona_bank.json` ("You are a helpful assistant.").
# This matches the encoding the codebase uses canonically (`personas.py:72`).
NEGATIVE_PANEL_4: Final[tuple[str, ...]] = (
    "assistant",
    "librarian",
    "programmer",
    "chef",
)

# Plan §10 Reproducibility Card eval panel: 19 bystanders + the 2 sources
# of each pair (source-self) + the bare default assistant. The
# eval rig constructs this dynamically once pair selection lands.

# ─────────────────────────────────────────────────────────────────────────────
# Persona-registry source-of-truth (plan §4 Inputs)
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_BANK_PATH: Final[str] = "data/issue_472/persona_bank.json"

# Plan §12 #7: HF data repo + path prefix for the question pool.
HF_DATA_REPO: Final[str] = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO: Final[str] = "superkaiba1/explore-persona-space"

# Output namespace conventions (plan §10 Reproducibility Card).
HF_R_PATH_PREFIX: Final[str] = "issue_527/R_persona"
HF_PAIR_SELECTION_PATH_PREFIX: Final[str] = "issue_527/pair_selection"
HF_TRAIN_MIX_PATH_PREFIX: Final[str] = "issue_527/training_mixes"
HF_TRAJECTORY_PATH_PREFIX: Final[str] = "issue_527/trajectories"
HF_ADAPTER_PATH_PREFIX: Final[str] = "adapters/issue_527"

# ─────────────────────────────────────────────────────────────────────────────
# Training recipe (plan §0 Plan Summary / §11 Decision Rationale)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical band-stop recipe per `.claude/rules/marker-training-recipe.md`.
RECIPE_LORA_R: Final[int] = 16
RECIPE_LORA_ALPHA: Final[int] = 32
RECIPE_LORA_DROPOUT: Final[float] = 0.0
# Attn-only — q/k/v/o; see plan §11 alternatives.
RECIPE_LORA_TARGETS: Final[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")
RECIPE_LR_PRIMARY: Final[float] = 5e-6
RECIPE_LR_RETRY: Final[float] = 1e-5
RECIPE_WARMUP_RATIO: Final[float] = 0.03
RECIPE_EPOCHS_CAP: Final[int] = 8
RECIPE_BAND_LOW_NATS: Final[float] = 5.0
RECIPE_BAND_HIGH_NATS: Final[float] = 12.0
RECIPE_PER_DEVICE_BATCH: Final[int] = 4
RECIPE_GRAD_ACCUM: Final[int] = 4
RECIPE_MAX_LENGTH: Final[int] = 2048

# Plan §0 / §11: seeds.
SEEDS: Final[tuple[int, ...]] = (42, 137, 256)

# Plan §4 contrastive-negatives section: per-arm positive counts.
N_POSITIVES_SINGLETON: Final[int] = 400
# Joint: literal union of singleton positives.
N_POSITIVES_JOINT: Final[int] = 800

# Strict 1:1 positives-to-total-negatives across all arms (plan §3 (g)).
# Per-negative-persona row counts derive from these:
# singletons: 400 negatives / 4 negative_personas = 100 each.
# joint:      800 negatives / 4 negative_personas = 200 each.

# ─────────────────────────────────────────────────────────────────────────────
# Eval recipe (plan §10 Reproducibility Card)
# ─────────────────────────────────────────────────────────────────────────────

# Plan §4 Step 5 — vLLM batched eval.
EVAL_N_PROMPTS_PER_PERSONA: Final[int] = 20
EVAL_N_SAMPLES_PER_PROMPT: Final[int] = 5
# Marker-leakage rule: max_new_tokens ≥ 2× longest trained completion.
EVAL_MAX_NEW_TOKENS: Final[int] = 2048

# ─────────────────────────────────────────────────────────────────────────────
# Output / sentinel paths
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_OUT_DIR: Final[str] = "eval_results/issue_527"
SENTINEL_PATH_TEMPLATE: Final[str] = "/workspace/logs/issue-527-{kind}-{epoch}.json"
