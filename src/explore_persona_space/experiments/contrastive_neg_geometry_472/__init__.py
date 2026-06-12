# ruff: noqa: RUF003  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 — contrastive-negative bystander-marker-leakage geometry sweep.

Merges #412 + #453 + #443, corrects #448. Parent #411. Goal (plan §1): on
on-policy DVs (post-response-slot marker log-prob AND full-vocab KL) logged as a
trajectory over training, determine how contrastive-negative *design* controls
bystander marker leakage along three axes — number, distance, and placement
geometry (barrier vs bubble).

The #448 fix this experiment implements (plan §0/§4.6): #448 saturated (trained
argmax = marker on every cell) because it trained 3 epochs and read the DV only
at the endpoint. #472 trains **1 epoch** and reads the on-policy DV at **6
checkpoints during the run** so cells are compared at a matched SUB-CEILING
source-implant level where count/distance/geometry effects are readable.

Module layout (plan §4.8):
    __init__              — constants: marker, cell specs, source, seeds, paths.
    persona_bank          — Phase 0: extend EVAL_PERSONAS_24 → ~60 via Sonnet 4.5;
                            persona-prompt resolution over the bank.
    centroids             — Phase 0.5: base-model L10/L15/L20 centroids over the
                            full bank (analysis/representation_shift machinery).
    select_negatives      — NEW distance-stratified negative selector
                            (near/far/spread/none) + single-neg sub-arms.
    r_generate            — Phase 1: base greedy on-policy R for the WHOLE bank
                            (forked from #448; distance-aware universe).
    build_training_data   — Phase 3 per-cell on-policy data build (forked from
                            #448; new distance-stratified negative path).
    eval_one_cell         — vLLM prompt_logprobs slot machinery (DV-A logP),
                            forked verbatim from #448 (_build_full_ids, MARKER_SEP,
                            off-by-one + token-equality guards).
    eval_trajectory       — NEW: per-checkpoint on-policy gen → DV-A(vLLM) +
                            DV-B(HF full-vocab KL) at the post-R slot. NOT #448's
                            teacher-forced MarkerTrajectoryCallback.
    base_panel            — Phase 1.5: base per-persona marker prior b_logprob.
    analyze               — Phase 5: separate logP/KL regressions, geometry,
                            identification gate, Holm multiplicity, figures.

The dispatcher is ``scripts/dispatch_neg_geometry_472.py`` (forked from #448's
unified smoke=sweep dispatcher).

WHY this module forks #448 rather than imports it: #448's
``contrastive_recipe_sweep_448`` module + ``dispatch_recipe_sweep_448.py`` live
ONLY on the unmerged ``issue-448`` worktree branch — they are NOT on ``main``,
so an import would fail on the pod (which runs the ``issue-472`` branch). Plan
§4.8 + §12 assumption 2: cherry-pick/fork the reused pieces onto the 472 branch.
"""

from __future__ import annotations

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
"""Base model. Inherited from #448/#411 (and from the marker-implantation
family)."""

# ── Marker (plan §10 / .claude/rules/marker-leakage-measurement.md) ──────────
# Single-token marker, LEADING SPACE form. Qwen-2.5-7B token id 83399. NOT the
# codebase default ``MARKER_TOKEN="[ZLT]"`` in personas.py (multi-token,
# deprecated) and NOT bare ``※`` (id 63680). We OVERRIDE the codebase default
# everywhere (sft.TrainLoraConfig.marker_text, the collator marker_token_ids,
# the eval/data-build constants). Launchers assert
# ``tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399]`` before
# any subprocess spawns.
MARKER_TEXT = " ※"
EXPECTED_MARKER_TOKEN_ID = 83399

# Marker separator — IDENTICAL to the assistant content training emits in
# build_training_data (``f"{r_text}\n\n{marker_text}"``). Forked from #448
# eval_one_cell (the text-level concat materializes the BPE-fused ``'.\n\n'``
# token (id 382) before the marker, which a token-id splice would NOT produce).
# LEGACY (#628): the project-default marker separator is now
# ``CANONICAL_MARKER_SEP = ""`` (slot-aligned rig, train/sft.py) — the marker
# trains DIRECTLY at the post-response slot the DV reads. This constant stays
# ``"\n\n"`` so the #472/#477/#504/#505 family reproduces byte-identically
# (477/504/505 + 2 scripts import it by name — do NOT rename).
MARKER_SEP = "\n\n"

# ── Source persona (plan §4.1) ───────────────────────────────────────────────
# Inherit the #448/#411 villain anchor: cosine −0.237 to assistant, interior to
# the persona cloud (room for both near- and far-negative arms). NOT swept.
SOURCE_PERSONA = "villain"

# ── Seeds (plan §10) ─────────────────────────────────────────────────────────
# 2 seeds, the min-N floor (user budget call: 8× H100 for faster wall-time, 2
# seeds). Effect must hold sign across BOTH seeds (2/2 — no majority-vote
# tolerance). A 2-seed geometry call is MODERATE, not HIGH.
SEEDS = (42, 137)

# The bare default assistant is ALWAYS among the negatives (plan §4.5; leakage to
# the default context is the safety target, open-q 3.7). qwen_default is the
# 24-panel persona that IS the bare default-instruct system prompt.
ALWAYS_INCLUDE_NEGATIVE = "qwen_default"

# ── Cell specs (plan §4.4 + §5 table) ────────────────────────────────────────
# Each cell shares the rs-LoRA recipe (r=32/α=64/lr=1e-5/1 epoch). The fields:
#   (slug, plain_english_name, placement, n_neg_personas, neg_ex_per_persona,
#    in_pooled_regression)
# placement ∈ {"spread","near","far","none"} (selects WHICH personas are
# negatives). The single-negative sub-arms set n_neg_personas=2 (qwen_default +
# ONE near/far placement persona) and are EXCLUDED from the pooled regression
# (not count-matched: 2 vs 4 personas, 200 vs 800 neg rows) — plan §4.4 + §6.
#
# Anchor = villain source, 4 neg personas chosen Spread, 200 neg ex/persona →
# 800 neg rows, 200 pos ex × 1 source. Sits in BOTH the count and placement
# sub-studies. POS_EX_PER_SOURCE is fixed at 200 across all cells.
#
# fields: slug, name, placement, n_neg_personas, neg_ex_per_persona, in_pooled
CellSpec = tuple[str, str, str, int, int, bool]
CELL_SPECS: tuple[CellSpec, ...] = (
    # Anchor — shared reference cell (Spread-4, 800 neg rows). In both sub-studies.
    ("c472_anchor", "Anchor (Spread-4)", "spread", 4, 200, True),
    # Count sub-study (placement = Spread, the neutral geometry).
    ("c472_negex_100", "Fewer negative examples", "spread", 4, 100, False),
    ("c472_negex_400", "More negative examples", "spread", 4, 400, False),
    ("c472_negp_2", "Two negative personas", "spread", 2, 200, False),
    ("c472_negp_8", "Eight negative personas", "spread", 8, 200, False),
    # Placement sub-study (count matched = 4 neg personas × 200 ex = 800 rows).
    ("c472_near", "Near negatives", "near", 4, 200, True),
    ("c472_far", "Far negatives", "far", 4, 200, True),
    ("c472_noneg", "No negatives", "none", 0, 0, False),
    # Single-negative sub-arms (NOT count-matched; standalone proximity maps only).
    ("c472_single_near", "Single near negative", "near", 2, 200, False),
    ("c472_single_far", "Single far negative", "far", 2, 200, False),
)

# Positive rows per cell (source persona × this many examples). Fixed across all
# cells (only the negative composition varies — plan §4.4).
POS_EX_PER_SOURCE = 200

# ── Distance metric (plan §4.3) ──────────────────────────────────────────────
# Layer-10 activation-centroid cosine (headline) + L15 + L20 (robustness; L20 is
# Persona-Vectors' actual evil layer, 1-indexed). Distance = 1 − cosine.
CENTROID_LAYERS = (10, 15, 20)
HEADLINE_LAYER = 10

# ── Trajectory eval cadence (plan §4.6 / §11) ────────────────────────────────
# 6 checkpoints/run at these % of max_steps. Denser early (8/16/33) where the
# implant is rising sub-ceiling and the geometry effect lives. The exact %s are
# ``ungrounded — needs smoke-test`` (plan §11); the smoke run validates them.
TRAJECTORY_CHECKPOINT_FRACTIONS = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)

# ── Matched source-implant slice (plan §6) ───────────────────────────────────
# Cross-cell comparison read where source-self ΔG first reaches this band:
# implant real (>5-nat floor) but held-out g_logprob ≥5 nats below ceiling.
MATCHED_SLICE_TARGET_NATS = 8.0
MATCHED_SLICE_BAND_NATS = 1.0  # 8 ± 1.
SOURCE_SELF_FLOOR_NATS = 5.0  # validity gate: every cell must clear this.
SUBCEILING_HEADROOM_NATS = 5.0  # held-out g_logprob must sit ≥ this below 0.0.

# ── Persona bank (plan §4.2) ─────────────────────────────────────────────────
# Target ~60 personas (extend EVAL_PERSONAS_24 with ~36 new ones); floor 50,
# ceiling 80. Fallback to the 24-panel if generation slips the budget.
PERSONA_BANK_TARGET = 60
PERSONA_BANK_N_NEW = 36
PERSONA_BANK_FLOOR = 50

# ── HF repos (plan §4.8 / Upload Policy) ─────────────────────────────────────
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_PREFIX = "issue472_neg_geometry"

# ── Training recipe constants (plan §10 reproducibility card / §11) ──────────
# rs-LoRA r=32 / α=64 / dropout 0.05 / lr=1e-5 / cosine / warmup 0.05 / 1 epoch /
# batch 4 × grad_accum 4 (eff 16) / max_len 1024 / AdamW bf16 / weight_decay 0.
# Loss masked to the ※ token + EOS via MarkerOnlyDataCollator(tail_tokens=0).
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.05
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024
MAX_NEW_TOKENS_GEN = 1024  # ≥2× trained completion (CLAUDE.md); log truncation.

# Sub-ceiling fallback recipe (plan §7 / §14 kill criterion): if the smoke gate
# trips (anchor already saturated at 1 epoch), drop to these and re-smoke.
FALLBACK_LORA_R = 16
FALLBACK_LEARNING_RATE = 5e-6
FALLBACK_EPOCHS = 0.5
