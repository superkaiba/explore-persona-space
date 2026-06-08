# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" + Greek ΔG are intentional
"""Task #505 — leave-one-out contrastive-negative localization (row-mass-fixed).

Parent #477; lineage #472 (geometry sweep) → #477 (row-scaled count) → #505 (this
run: drop one of K=6 non-default negatives, redistribute its rows across the
remaining K−1, hold total negative rows + positives + steps fixed). Plan
``tasks/running/505/plans/plan.md`` is the authoritative spec.

The question this module operationalizes (plan §1): does dropping a specific
contrastive negative ``j`` raise held-out marker leakage SPECIFICALLY for
bystander personas similar to ``j``, rather than uniformly? The dependent
quantity is per-bystander ``Δ-Leakage(b; j) = Leakage(b | j-dropped arm) −
Leakage(b | full-set arm)`` at the matched frac-0.50 mid-training checkpoint;
the headline test is the pooled mixed-effects slope ``Δ-Leakage ~ similarity(b,
j)`` (random effects on b, j, seed). See plan §13.

Reuses verbatim from ``contrastive_neg_geometry_472``:
    persona_bank, centroids (extend to L21), select_negatives (spread quantile),
    r_generate, train_cell.train_one_cell (already threads
    marker_suppress_at_post_response_slot + marker_im_end_token_id since #477
    v6 — see train_cell.py:365-368 + sft.py:541-542), eval_trajectory,
    eval_one_cell (vLLM score_logp_for_R), eval_guard.assert_adapter_actually_applied
    (cherry-picked from origin/issue-477 16f6789e8 — see plan §10 step 0).

NEW in #505:
    panel_coverage     — §5.4 joint K + held-out-panel construction gate
                         (≥ 8 personas per tercile × var ≥ 0.02² per j_i).
    build_pv_centroids — §5.7 base-model centroid build at layers {7,14,21,27}
                         (the L10 fallback already lives on the HF data repo
                         from #472; the L21 headline does not).
    build_training_data— §5.3 row-redistribution (full-set K=6 → 25 each + 50
                         qwen_default = 200 neg rows; drop-arm K−1=5 → 30 each
                         + 50 qwen_default = 200 neg rows).
    analyze            — §13 panel_similarity_matrix.json (both cos(b, j) AND
                         cos(b, source) per layer), mixed-model pooled fit,
                         per-arm β_j sign-agreement, §13.3 partial with source
                         ΔG + cos(b, source) covariates.
    dispatch           — §5.5 unified smoke=sweep dispatcher (smoke = --cells 1
                         --seeds 1 on the full-set arm; sweep = all 8 arms × 3
                         seeds = 24 trained adapters, per-cell subprocess on
                         a 4× H100 pod, GPU-pinned via +gpu_id=N).

The plan's §11 "Marker loss masking" conjunction is LOAD-BEARING:
``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
im_end_token_id=151645)``. Without the flag, the negative branch trains the
trailing newline (one PAST the DV slot) and the contrastive contribution at
the post-response slot is null by construction — the leave-one-out signal
collapses to noise. The §5.5 gate (h) + ``tests/test_issue505_collator_post_response_slot.py``
enforce that BOTH flags are wired through the trainer config + every
collator call site before any training spawn.
"""

from __future__ import annotations

# ── Inheritance: marker, source, model, ratios, recipe ──────────────────────
# Inherit from the #472 module so future renames in #472 propagate. Every name
# below stays single-source-of-truth in #472's __init__.py.
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (  # noqa: F401
    ALWAYS_INCLUDE_NEGATIVE,
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    MARKER_SEP,
    MARKER_TEXT,
    SOURCE_PERSONA,
)

# ── Anchor recipe (plan §5.1, §8, §11) ──────────────────────────────────────
# Sub-saturating anchor between #472 (rank 32 + lr 1e-5 → saturating) and #477's
# under-trained rank-2 / lr-2e-6.
#
# LORA_R = 32 (round 8, 2026-06-07): bumped from 16 after round-7 smoke (rank
# 16 + lr 5e-6 + 3 epochs / 75 steps) trained to source ΔG=0.82 nats at frac
# 0.50 — well below plan §7's 5-nat validity floor (≈6× under). Trajectory
# (rank 16): 25 steps → 0.04 nats (round 6), 75 steps → 0.82 nats (round 7);
# rank is the under-training bottleneck given how little the implant moved
# even after tripling the step count. Plan §8 explicitly lists rank-32 +
# lr 5e-6 as the under-training fallback; keeping EPOCHS=3 (round-7's
# single-knob bump) on top of rank 32 strictly dominates the literal §8
# "rank 32 + 1 epoch" wording.
#
# LEARNING_RATE = 1e-5 (round 9, 2026-06-08): bumped from 5e-6 after round-8
# smoke (rank 32 + lr 5e-6 + 3 epochs / 75 steps) trained to a FLAT source-ΔG
# trajectory at ≈1.6 nats across the whole training curve. Per-fraction
# trajectory: 0.08 → 1.63, 0.16 → 1.57, 0.33 → 1.51, 0.50 → 1.60 (headline,
# still ≈3× under §7's 5-nat floor), 0.75 → 1.59, 1.00 → 1.69. Eval-guard
# PASS (B-norm=0.086, max|ΔG|=1.979) confirms the LoRA IS applied — the
# implant exists but is stuck near 1.6 nats from step 6 onward across 12.5×
# more optimizer steps. A flat trajectory is the LR-bound signature, not the
# step-bound one; bumping EPOCHS won't move it. Plan §11's "1e-5 saturates
# per #472" prior was derived from #472's 800-negative / 62-step recipe;
# #505 runs 200 negatives / 75 steps, which is firmly under-trained — the
# saturating-recipe ceiling doesn't transfer here. Other recipe knobs
# (LORA_R 32, LORA_ALPHA 32, EPOCHS 3, dropout 0.05, positives 200, totals)
# UNCHANGED — this is a deliberate single-axis LR bump.
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.05
# EPOCHS = 3 (round 7, 2026-06-06): bumped from 1 after smoke run yjz5ytuz
# showed mean_token_accuracy=0.645 and grad_norm RISING at end of training —
# the marker did not implant at 25 optimizer steps (1 epoch). 3 epochs = 75
# steps lands sub-saturation per the smoke's still-rising loss curve
# (step10=6.08 → step20=4.60 → final=4.998, grad_norm 31.9 → 50.1). Smoke
# eval-guard correctly fired LoRANotAppliedError at source-self ΔG=0.04 nats
# (essentially zero implant). Cost delta ~0 because training is <1% of cell
# wall time (eval is 99%); total sweep stays ~40 GPU-h under the 100 cap.
# Held at 3 through round 8 (rank 16→32 bump) — round 8 is a single-axis
# rank change layered on top of round 7's epoch bump.
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024
WEIGHT_DECAY = 0.0
MAX_NEW_TOKENS_GEN = 2048  # ≥ 2× longest trained completion per marker-leakage rule

# vLLM ``max_model_len`` (round 10, 2026-06-08): bumped from #472's default of
# 2048 to 4096 after round-9 smoke (PID 9162, log
# ``/workspace/logs/issue-505-20260608-043806.log`` on pod-505) crashed at
# frac 0.50 with::
#
#     ValueError: The decoder prompt (length 2050) is longer than the maximum
#     model length of 2048.
#
# Root cause: ``MAX_NEW_TOKENS_GEN = 2048`` lets the trained model's on-policy
# ``R_j`` approach the cap (round-9 produced longer R_j than round-8 because
# of the LR bump); the post-R-slot ``score_logp_for_R`` then feeds vLLM a
# prompt of ``system_prompt + question + R_j + marker context`` which exceeds
# the 2048 cap once R_j is near its own cap. 4096 = 2× MAX_NEW_TOKENS_GEN
# covers the worst-case prefix + R_j + marker context. Overridden at #505's
# call site only; #472's ``DEFAULT_MAX_MODEL_LEN = 2048`` is untouched (no
# shared-recipe change).
MAX_MODEL_LEN = 4096

# ── Marker-loss masking — THE LOAD-BEARING CONJUNCTION (plan §11, §5.5 gate h) ──
# Without BOTH flags the negative branch trains the trailing "\n" one PAST the
# DV slot; contrastive contribution at the DV slot is null; leave-one-out
# headline collapses to noise. Defaults on `main` (sft.py 541-542) are OFF /
# None, so the dispatcher MUST set them — `tests/test_issue505_collator_post_response_slot.py`
# is the CI gate.
MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT = True
QWEN_IM_END_TOKEN_ID = 151645  # Qwen-2.5-7B-Instruct <|im_end|>

# Sub-ceiling fallback (plan §5.5 fallback): if smoke gate (a)-(f) fail, drop to
# this slightly stronger anchor and re-smoke. Captured here so the dispatcher's
# fallback path uses the same source-of-truth constants.
FALLBACK_LORA_R = 32
FALLBACK_LEARNING_RATE = 5e-6
FALLBACK_EPOCHS = 1

# ── K, the negative set, and the held-out panel (plan §5.3, §5.4) ───────────
# K = 6 non-default negatives + always-included qwen_default = 7 negatives in
# every non-empty arm. K=6 gives 6 within-arm drops (binomial 5/6 ≤ p=0.11 for
# the secondary sign-agreement read) and per-non-default rows ≥ 25 (above the
# marker-only loss floor; #472's smallest cell had 100 rows per non-default,
# 30 is the lowest comfort floor).
K_NON_DEFAULT = 6

# ── Row composition (plan §5.3 table — held fixed across every arm) ─────────
# Total negative rows = 200 across every arm; qwen_default gets a fixed 50 rows
# (~3× a single non-default) so the safety-target leakage stays constrained;
# the remaining 150 rows split evenly across the arm's non-default negatives:
#   Full set: K=6 non-default × 25 rows + 50 qwen_default = 200
#   Drop j_i: K−1=5 non-default × 30 rows + 50 qwen_default = 200
# The 50/150 split is the one ungrounded numeric per plan §11 — the smoke check
# validates it.
TOTAL_NEG_ROWS = 200
QWEN_DEFAULT_NEG_ROWS = 50
POS_EX_PER_SOURCE = 200  # 1:1 positives-to-total-negatives ratio
NON_DEFAULT_ROWS_FULL_SET = 25  # 25 × 6 = 150 = TOTAL_NEG_ROWS − QWEN_DEFAULT_NEG_ROWS
NON_DEFAULT_ROWS_DROP_ARM = 30  # 30 × 5 = 150 = TOTAL_NEG_ROWS − QWEN_DEFAULT_NEG_ROWS

# ── Seeds (plan §5.8) ───────────────────────────────────────────────────────
# 3 seeds = +33% precision over the #472 codebase floor (2) at +33% GPU cost.
SEEDS = (42, 137, 219)

# ── Similarity (plan §5.7) ──────────────────────────────────────────────────
# Headline: persona-vectors cosine layer 21 (codebase rule
# `persona-distance-metrics.md` legacy default). Robustness across {7, 14, 27}
# and the in-codebase L10 fallback (already lives on the HF data repo from
# #472). 0-indexed transformer blocks per plan §5.7.
HEADLINE_LAYER = 21
ROBUSTNESS_LAYERS = (7, 14, 27)
INHERITED_L10_LAYER = 10  # already on HF data repo from #472
ALL_SIMILARITY_LAYERS = (HEADLINE_LAYER, *ROBUSTNESS_LAYERS, INHERITED_L10_LAYER)
SIMILARITY_LAYERS_TO_BUILD = (7, 14, HEADLINE_LAYER, 27)  # L10 already exists

# ── Trajectory eval cadence (plan §5.6) ────────────────────────────────────
TRAJECTORY_CHECKPOINT_FRACTIONS = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)
HEADLINE_CHECKPOINT_FRAC = 0.50

# ── Validity gates (plan §6, §7) ───────────────────────────────────────────
SOURCE_DG_FLOOR_NATS = 5.0  # arm under this is uninformative (kill criterion)
SOURCE_DG_SATURATION_CEILING_NATS = 19.0  # above + emission=1.0 → swap to KL DV
# Band lower edge widened 14.0 → 5.0 in round 8 (2026-06-07): the round-7
# rank-16 trajectory (ΔG=0.82 nats at frac 0.50) showed the original 14-nat
# floor was an extrapolation from #472's saturating-recipe (800 negatives,
# 62 optimizer steps, rank 32 + lr 1e-5) that does not transfer to the §5.1
# sub-saturating regime. Plan §7 is explicit that `ΔG_source ≥ 5 nats at
# frac 0.50` is THE validity floor (any arm below = uninformative kill);
# the smoke band's lower edge should match the validity gate (§7), not a
# saturating-recipe extrapolation (#472). Upper edge unchanged — above 18
# nats + emission≥0.85 is the saturation-swap criterion (gate b).
SOURCE_DG_EXPECTED_BAND_NATS = (5.0, 18.0)  # smoke target at frac 0.50
SOURCE_DG_DRIFT_TOLERANCE_NATS = 2.0  # drop-arm vs full-set tolerance
SOURCE_EMISSION_SATURATION_THRESHOLD = 0.85

# ── Panel coverage gate thresholds (plan §5.4) ─────────────────────────────
PANEL_TERCILE_FLOOR = 8  # ≥ 8 personas per top/bottom tercile per j_i
# PANEL_VARIANCE_FLOOR removed (round 5): the original derivation `0.02**2`
# mis-applied #472's `ID_GATE_SD_FLOOR = 0.02` — that floor is an
# SD-across-arms of a DISTANCE metric in #472's leakage analysis, NOT this
# experiment's within-panel cosine variance to a single j. Different
# distribution; on the actual #472 bank + L10 centroids the realised
# within-panel variances sit at 0.00012-0.00018, ~2-3× below the
# misderived 0.0004 floor, so the gate fired false positives on every j_i
# and halted Phase 1 immediately. The tercile_ok check (≥ 8 personas in
# both top and bottom terciles of cos(b, j_i)) is §5.4's load-bearing
# identification condition; the variance floor was redundant. The constant
# is kept as 0.0 for backward compatibility (any downstream consumer that
# imports it without expecting the gate to fire still gets a finite
# float), but the gate path in `panel_coverage.run_panel_coverage_gate`
# no longer consults it.
PANEL_VARIANCE_FLOOR = 0.0

# ── HF repos (plan §5.9, §10, reproducibility card) ────────────────────────
# Inherit data-repo bank + centroid bundles from #472 under the geometry/
# subfolder. #505 writes its own cell training data + per-cell on-policy R +
# centroid bundles for the new layers under its own subfolder.
HF_DATA_PREFIX_INHERIT = "issue472_neg_geometry"
HF_DATA_PREFIX = "issue505_loo_contrastive"
HF_ADAPTER_PATH_PREFIX = "adapters/issue_505"


# ── Cell specs (plan §5.3, §5.10) ──────────────────────────────────────────
# (slug, plain_name, dropped_j_idx, in_pooled_regression)
# dropped_j_idx is the INDEX (0..K_NON_DEFAULT-1) into the non-default negative
# list for the drop-one arms; None for full-set and no-negatives. The actual
# persona name is resolved at panel-coverage time from the spread-quantile
# selector (plan §5.3 - non-default selection deterministic given the bank
# content-hash + the spread quantiles). The dispatcher pins the resolved
# K-set in the manifest.
CellSpec = tuple[str, str, int | None, bool]
CELL_SPECS: tuple[CellSpec, ...] = (
    ("c505_full_set", "Full-set (all 6 + default)", None, True),
    ("c505_drop_j0", "Drop j0", 0, True),
    ("c505_drop_j1", "Drop j1", 1, True),
    ("c505_drop_j2", "Drop j2", 2, True),
    ("c505_drop_j3", "Drop j3", 3, True),
    ("c505_drop_j4", "Drop j4", 4, True),
    ("c505_drop_j5", "Drop j5", 5, True),
    ("c505_no_negatives", "No negatives (control)", None, False),
)

DROP_ARMS = tuple(s for s in CELL_SPECS if s[2] is not None)
FULL_SET_SLUG = "c505_full_set"
NO_NEGATIVES_SLUG = "c505_no_negatives"
