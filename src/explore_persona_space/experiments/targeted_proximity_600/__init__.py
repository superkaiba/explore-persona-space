# ruff: noqa: RUF002, RUF003  # em-dash + Greek ΔG + Qwen marker " ※" + × intentional
"""Task #600 — targeted proximity-dose test (addition-direction, d_source-matched).

Parent #505 (removal-direction null); lineage #472 (geometry sweep, recipe +
rig + run-noise calibration) → #477 → #505 → #600. Plan
``tasks/running/600/plans/plan.md`` is the authoritative spec.

The question (plan §1): does a contrastive negative parked next to a specific
held-out bystander suppress THAT bystander's marker leakage? Single manipulated
variable per matched pair: the proximity of ONE negative-panel slot to a
pre-chosen target bystander — NEAR (the target's nearest neighbour in centered
L10 cosine space) vs CONTROL (matched on distance-to-source within ε but far
from the target). Everything else fixed: villain source, 200 positives, 800
negative rows = 4 personas × 200, fixed base panel (qwen_default + 2
mid-distance personas), r16/α32 attn-only rsLoRA, lr 5e-6, 63 matched optimizer
steps, marker-only loss with the post-response-slot conjunction.

Module layout (plan §4.9):
    __init__       — constants (recipe re-pins, gates, selection thresholds).
    select_panels  — §4.3 deterministic CPU design-time selection → committed
                     ``eval_results/issue_600/panel_selection.json``.
    cells          — CELL_SPECS_600 built from the committed manifest
                     (explicit 4-persona panels; no placement-derived path).
    dispatch       — fork of ``leave_one_out_505/dispatch.py``: unified
                     smoke=sweep dispatcher, per-(cell,seed) subprocesses,
                     §4.7 gates (a)-(h), uploads, pod sentinel.
    analyze        — §6 paired stats: exact target-level sign-flip
                     permutation, run-noise distribution, locality +
                     bubble-radius reads, figures (CPU, VM, post-teardown).

Recipe constants are RE-PINNED here per plan §10 (NOT #505's rescued
r32/lr1e-5/3ep constants — those were the #505 cells' values, expressly not
inherited) and NOT #472's r32/α64/lr1e-5 either.
"""

from __future__ import annotations

# ── Inheritance: marker, source, model, repos (single source of truth: #472) ─
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

# ── Training recipe (plan §0 / §11 — the marker-recipe clean window) ─────────
# r16/α32 attn-only rsLoRA, lr 5e-6, 1 epoch (smoke-laddered ≤3), batch 4 ×
# grad-accum 4 (63 optimizer steps on 1000 rows), max_len 1024.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.05
EPOCHS_DEFAULT = 1
# §4.7 fallback ladder on the floor direction (decided ONCE at smoke; the
# pinned value applies to ALL 36 cells so steps stay matched).
EPOCHS_LADDER = (1, 2, 3)
# §4.7 saturation-direction fallback (not expected at this LR).
LR_SATURATION_FALLBACK = 2e-6
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024

# LOAD-BEARING (plan §4.5): train_one_cell never set TrainLoraConfig.
# lora_targets before #600, and train_lora resolves None to the historical
# 7-module list (q/k/v/o + gate/up/down) — the recipe #505's smoke showed
# flooring at r16/lr5e-6. The attn-only quad below is threaded explicitly via
# ``lora_targets_override`` and parity-asserted on the realized peft_config.
LORA_TARGETS_ATTN_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")

# ── Marker-loss masking — THE LOAD-BEARING CONJUNCTION (plan §11) ───────────
# Without BOTH flags the negative branch trains the trailing "\n" one PAST the
# DV slot; the contrastive contribution at the DV slot is null. Defaults on
# `main` (sft.py ~594-595) are OFF/None, so the dispatcher MUST set them.
MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT = True
QWEN_IM_END_TOKEN_ID = 151645  # Qwen-2.5-7B-Instruct <|im_end|>

# ── Band callback: LOG-ONLY mode (plan §4.5 / §4.8) ─────────────────────────
# marker_band_stop stays at its TrainLoraConfig default (True) so the callback
# attaches; marker_band_log_only=True keeps per-step telemetry WITHOUT ever
# stopping — all 36 cells run exactly the same matched step count. Pinning
# marker_band_stop=False instead would silently disable the callback (#480
# incident class: declared monitors that never functioned).
MARKER_BAND_LOG_ONLY = True

# ── Rows per cell (plan §4.1) ───────────────────────────────────────────────
POS_ROWS = 200  # must equal contrastive_neg_geometry_472.POS_EX_PER_SOURCE
N_NEG_PERSONAS = 4
NEG_ROWS_PER_PERSONA = 200
TOTAL_ROWS = POS_ROWS + N_NEG_PERSONAS * NEG_ROWS_PER_PERSONA  # 1000
# 63 optimizer steps at 1 epoch: ceil(1000 / (4 × 4)) = 63.
EXPECTED_STEPS_PER_EPOCH = -(-TOTAL_ROWS // (BATCH_SIZE * GRAD_ACCUM))  # 63

# ── Seeds (plan §11) ────────────────────────────────────────────────────────
SEEDS = (42, 137, 219)

# ── Trajectory checkpoints (plan §4.8 grid) ─────────────────────────────────
TRAJECTORY_CHECKPOINT_FRACTIONS = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)

# ── Eval (plan §4.6) ────────────────────────────────────────────────────────
MAX_NEW_TOKENS_GEN = 2048  # ≥ 2× longest trained completion (#260 rule)
MAX_MODEL_LEN = 4096  # 2× MAX_NEW_TOKENS_GEN (the #505 round-10 crash fix)
# vLLM LoRA-rank cap == the trained rank: a non-r16 adapter (i.e. the
# lora_r_override silently not threading) is REJECTED loudly at eval load.
MAX_LORA_RANK_EVAL = 16
# #534 adapter-application cross-check tolerance (gate h). The wired #534
# default (eval_guard.DEFAULT_SOURCE_MANIFEST_TOL_NATS) is 2.0 nats; the plan
# says "~1 nat" but the two reads differ in kind (teacher-forced train rows
# vs on-policy eval probes), so the validated default is kept and the realized
# gap is recorded in the gate payload.
SOURCE_MANIFEST_TOL_NATS = 2.0

# ── Smoke gates (plan §4.7) ─────────────────────────────────────────────────
SOURCE_DG_FLOOR_NATS = 5.0  # validity floor (seed-mean per condition)
SEED_LEVEL_DG_FLOOR_NATS = 3.0  # seed-level floor for ratio stability (§6)
SOURCE_DG_BAND_NATS = (5.0, 19.0)  # gate (a): floor AND sub-saturation
SOURCE_LOGP_CEILING_EPS_NATS = 0.1  # gate (b): trained source logP ≤ −0.1
BYSTANDER_ARGMAX_CEILING = 0.92  # gate (b): bystander argmax-marker rate

# ── Design metric (plan §4.2 / §4.3) ────────────────────────────────────────
DESIGN_LAYER = 10
COSINE_CENTERING = "global_mean"  # canonical per persona-distance-metrics.md
ROBUSTNESS_LAYERS = (15, 20, 21)  # analysis-time reads only
EPS_MATCH = 0.10  # |Δd_source| within a NEAR/CONTROL pair (centered units)
EPS_MATCH_RELAXED = 0.15
CONTRAST_FLOOR = 0.30  # d(CTRL,t) − d(NN,t) must exceed this
NEAR_QUANTILE = 0.25  # NN must be ≤ P25 of the panel's distance-to-t dist
FAR_QUANTILE = 0.75  # CTRL must be ≥ P75 of the panel's distance-to-t dist
FAR_QUANTILE_RELAXED = 0.60
N_TARGETS = 6
N_TARGETS_PER_STRATUM = 2  # near/mid/far terciles of d_source

# ── HF repos / WandB (plan §4.9, Upload Policy) ─────────────────────────────
HF_DATA_PREFIX = "issue600_targeted_proximity"
HF_ADAPTER_PATH_PREFIX = "adapters/issue_600"
WANDB_PROJECT = "issue600_targeted_proximity"

# ── Inherited #472 inputs: issue-600-OWNED pinned snapshot (crash-fix round 4)
# The shared issue472_neg_geometry/ HF mirrors of R_train.json + centroids_L10
# are a STALE generation (git dac5749 — R_train lacks 'bartender'; L10 differs
# from the bundle panel_selection.json was selected against). The 2026-06-11
# GCE smoke crashed on exactly that divergence (KeyError 'bartender' in
# build_cell). Fix: the VERIFIED local copies (R_train content_hash
# 45a11b1fa664…, git b68e560, 61 personas; L10 sha256 3d62a6b258a3…) were
# uploaded to the issue-600-owned path below, and EVERY prefetch/autofetch is
# sha256-pinned against EXPECTED_SHA256 — a divergent file fails LOUD at
# phase=prefetch, never at build_cell ten frames deep.
HF_DATA_PREFIX_INPUTS = f"{HF_DATA_PREFIX}/inputs"
# Pin table keyed by path relative to the i472 data root (== path under
# HF_DATA_PREFIX_INPUTS). persona_bank + L15/L20 matched local↔HF at incident
# time but are pinned too: the 5 files are ONE atomic trust boundary.
EXPECTED_SHA256: dict[str, str] = {
    "persona_bank.json": "1e831ec200e485ee2735436cae8ea2c609349e73108dd8c9eeedf944f315f5f3",
    "on_policy_R/R_train.json": "93f907dd55a53de09514af8950d06501c88bdfa065c0e1d3ae6a92f39cb0d491",
    "centroids_L10.pt": "3d62a6b258a3bb1b2cf2d1a35558e262e04393fd0b0c3bad017bc4a027fd6281",
    "centroids_L15.pt": "f45265cf3549e6f28ff5c0c9512fe72e2434b4ca5c4635c47e59feb16d37d451",
    "centroids_L20.pt": "645ce55a306122f08d6dbbdde559a132f82c54f376c10cb8626eede3394ff5d7",
}
