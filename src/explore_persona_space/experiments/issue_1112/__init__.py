# ruff: noqa: RUF002, RUF003
"""Task #1112 — activation-shift geometry vs training method + contrastive negatives.

Sycophancy 2×2 {LoRA, full-FT} × {posonly, +contrastive} on #1090's frozen c3
mix (cell 1 reuses the #1090 fu2 organism) + 2 generic-data controls + an
lr-matched marker LoRA-vs-FT pair, with 28-layer × 3-pooling-arm Δx capture
and the #653 spectral geometry reads. Plan: tasks/*/1112/plans/v3.md.

This package holds the CPU-testable pieces (constants, mix derivation,
geometry aggregation); the pod driver is ``scripts/issue1112_dispatch.py``.
"""

from __future__ import annotations

ISSUE = 1112
SLUG = "issue1112_geometry2x2"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
DATA_PREFIX = SLUG  # issue1112_geometry2x2/... on the data repo

# ── Pinned reused inputs (plan §4.6 / §10; full revisions resolved 2026-07-07) ─
C3_MIX_REV = "6aab0cce1facbb2926406c9787d5d455291cbc37"
C3_CELL_PREFIX = "issue1090_pvdatagen/c3-sycophancy-claude"
C3_MIX_PATH = f"{C3_CELL_PREFIX}/mix/train_mix.jsonl"
C3_MIX_META_PATH = f"{C3_CELL_PREFIX}/mix/mix_meta.json"
C3_MIX_SHA256 = "c00e8f4c28d2947e97b7ea9bd2ea77d18943e13009da14d408616af6b55f8422"
C3_POS_PATH = f"{C3_CELL_PREFIX}/datagen_topup/pos.jsonl"
C3_CN_PATH = f"{C3_CELL_PREFIX}/datagen_topup/cn.jsonl"
# Sidecars the fu1 fixed-pool derivation reads (derive_margin_pools_topup):
# first-sample raw+judge under datagen/, tranche raw+kept under datagen_topup/.
C3_MARGIN_SIDECARS = (
    "datagen/raw_pos.jsonl",
    "datagen/judge_raw_pos.json",
    "datagen_topup/raw_pos.jsonl",
    "datagen_topup/kept_pos.jsonl",
    "datagen_topup/raw_neg.jsonl",
    "datagen_topup/kept_neg.jsonl",
)
MARGIN_POOL_SMOKE_N = 4  # per-side pool slice for --smoke margin reads
GENERIC_CORPUS_PATH = "issue906_inputs/generic_corpus.jsonl"
MARGIN_POOLS_PREFIX = "issue1090_pvdatagen/fu1-margin-qwen"
MARGIN_POOLS_REV = "043acb7f5353b7c640aefb6acb094620ad1a6a50"
R_TRAIN_PATH = "issue472_neg_geometry/on_policy_R/R_train.json"
R_TRAIN_REV = "a85426a2391b3ae04399714269fdc0a09088283a"
FU2_CKPT_REV = "18aca1188ececddf9ee33a86554bfbec810f0fc6"
FU2_CKPT_PREFIX = "issue1090/fu2/c3-sycophancy-claude"  # on OVERFLOW_REPO
FU2_SELECTED_STEP = 14  # judged rate 0.61 (Tier-2 0.595)
FU2_PARITY_RATE = 0.61
FU2_PARITY_TOL = 0.15

# ── Cells (plan §4.1 / §5) ────────────────────────────────────────────────────
SYCO_BEHAVIOR = "sycophancy"
SOURCE_CONTEXT_ID = "persona_software_engineer"
SYCO_CELLS_NEW = ("s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos")
GENERIC_CELLS = ("s5_lora_generic", "s6_fullft_generic")
MARKER_CELLS = ("m1_lora_band8", "m2_fullft_band8")
REUSED_CELL = "s1_lora_neg"
ALL_TRAINED_CELLS = (REUSED_CELL, *SYCO_CELLS_NEW, *GENERIC_CELLS, *MARKER_CELLS)

CELL_MIX = {  # which derived/staged mix each trained cell consumes
    "s1_lora_neg": "c3_frozen",
    "s2_lora_pos": "c3_posonly",
    "s3_fullft_neg": "c3_frozen",
    "s4_fullft_pos": "c3_posonly",
    "s5_lora_generic": "c3_generic_only",
    "s6_fullft_generic": "c3_generic_only",
    "m1_lora_band8": "marker_contrastive",
    "m2_fullft_band8": "marker_contrastive",
}

# ── Training (plan §4.3 / §11) ───────────────────────────────────────────────
SYCO_EPOCHS = 6  # 30-optimizer-step ceiling on the 80-row mix (fu2 verbatim)
SYCO_SAVE_STEPS = 2
SYCO_MAX_LENGTH = 2048
SYCO_STEP_CEILING = 30
G1_EXTENSION_STEP_CEILING = 60  # the pre-registered one-shot dose extension
FT_LR = 5e-6  # #606/#642 full-FT recipe (cosine, warmup 0.05, eff-batch 16)
FT_WARMUP_RATIO = 0.05
FT_PER_DEVICE_BATCH = 4
FT_GRAD_ACCUM = 1  # × 4 GPUs = eff 16, matched to the LoRA cells
FT_CKPT_STEPS = tuple(range(2, 31, 2))

MARKER_BAND = (7.0, 9.0)  # narrowed from [5,12] to center the #514 8±1 target
MARKER_GLOBAL_BAND = (5.0, 12.0)  # match acceptance window (success criterion 2)
MARKER_MATCH_TOL_NATS = 2.0
MARKER_FT_LR = 5e-6  # #514 ft_b1 (linear, warmup 0.03, eff-batch 64, max_len 1024)
MARKER_FT_GRID = (2, 3, 4, 5, 6)
MARKER_FT_FALLBACK_LR = 2e-6  # #514 ft_lowlr lever
MARKER_FT_FALLBACK_GRID = (6, 8, 10)
MARKER_READ_LAYER = 25  # pre-registered marker primary read layer (plan §11)
MARKER_TOKEN_ID = 83399  # " ※" with leading space

# ── Install matching / gates (plan §4.3 / §6 / §7) ───────────────────────────
INSTALL_FLOOR = 0.45  # below after G1 -> reported NOT installed (labeled)
BASE_SYCO_RATE = 0.215  # #1090 fu1/fu2 instrument-matched base read

# ── Capture / geometry (plan §4.4 / §4.5) ────────────────────────────────────
N_LAYERS = 28
HIDDEN = 3584
CAPTURE_ARMS = ("prefix", "context", "response")
PRIMARY_LAYER = 14  # pre-registered #653 anchor (sycophancy)
SYCO_MAX_NEW_TOKENS = 1024
MARKER_MAX_NEW_TOKENS = 2048  # marker end-of-completion rule (>= 2x trained)
TF_BATCH_SIZE = 8
CAPTURE_DOSES = ("step6", "selected", "step30")  # behavior cells only
N_BOOT = 1000
BOOT_SEED = 653  # the #653 convention
SUBSAMPLE_N = 80  # #653 cloud-size sensitivity read (layer 14 / response)
SUBSAMPLE_DRAWS = 100

# Verdict thresholds for H1/H2 (plan §3)
H1_FALSIFY_MARGIN = -10.0  # U < -10 -> falsified (FT decisively more concentrated)
H2_COS_CONFIRM = 0.8
H2_COS_FALSIFY = 0.9
H2_RANKK_MODES = 5.0
H4_MAGNITUDE_FRAC = 0.5

WANDB_PROJECT = SLUG


def cell_run_name(cell: str) -> str:
    """WandB run name per trained cell (one run per cell, plan §10)."""
    return f"issue1112_{cell}_seed{SEED}"
