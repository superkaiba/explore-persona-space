"""Shared helpers + constants for issue #617 (WildChat-category contexts).

Per plan §4. This experiment is CAPTURE-ONLY (no training): it filters a real
WildChat slice, clusters it by topic, verifies cluster separability in
Qwen-2.5-7B-Instruct activation space (REUSING #594's extractor + analyzer),
samples realistic completions onto the two best-separating categories, and
uploads the corpus to the HF data repo as a reusable artifact.

Like ``issue594_common.py``, this is NOT a library module under ``src/`` — it
lives next to the ``scripts/issue617_*`` entry points it serves so the
experiment-specific constants don't leak into the project library.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "issue617"
EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_617"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_617"

# Pipeline artifact paths (plan §4 DAG).
SLICE_PATH = DATA_DIR / "wildchat_slice.json"
CLUSTER_PATH = DATA_DIR / "cluster_assignments.json"
EXTRACTION_BATTERY_PATH = DATA_DIR / "extraction_battery.json"
CLUSTER_MEMBERSHIP_PATH = DATA_DIR / "cluster_membership.json"
EXTRACTION_DIR = DATA_DIR / "extraction"
SEPARABILITY_PATH = EVAL_DIR / "separability.json"
PICKED_DIR = DATA_DIR / "picked_categories"

# Models (plan §10 Reproducibility Card).
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # activation read + completion sampling (Source: #594)
EMBEDDER_MODEL = "BAAI/bge-large-en-v1.5"  # CLS-pool sentence embedder (Source: MTEB / WildVis)
EMBEDDER_DIM = 1024

# Clustering knobs (plan §10 / §11).
CLUSTER_KS = (5, 10, 20)
EMBED_MAX_TOKENS = 512  # bge-large-en-v1.5 context window

# Read layers (mid-band, Source: #594).
READ_LAYERS = (13, 14, 18)

# Extraction pool cap (M2). Each conv_id extracted ONCE; cluster membership is
# mapped at scoring time. Per-cluster scoring floor preserved by the stratified
# subsample.
EXTRACTION_POOL_CAP = 400
PER_CLUSTER_FLOOR = 30
PER_CLUSTER_SCORING_MAX = 50

# The synthetic instance the #594 full-extraction path requires (M1): its id is
# hard-coded at three sites in issue594_extract_context_vectors.py (lines 181,
# 383, 516-518). It is extracted alongside the WildChat prefixes and EXCLUDED
# from the cluster pair pool.
SYNTHETIC_DEFAULT_ID = "f6_default_template"

# Permutation null (plan §4 step 4 / §11).
PERM_B = 1000

# Separability fire-rule thresholds (plan §6).
PURITY_THRESHOLD = 0.7
PGLOBAL_THRESHOLD = 0.01

# Completion sampling (plan §10 / §11 — clarifier resolution 4).
COMPLETION_N = 4
COMPLETION_TEMP = 1.0
COMPLETION_MAX_NEW_TOKENS = 512
COMPLETION_CAP_PER_CATEGORY = 200

# Slice build (plan §4 step 1 / §11).
SLICE_TARGET = 20_000
SLICE_SCAN_CAP = 200_000
SEED = 42

# HF upload destination (plan §4 step 7). Mirrors #594's HF_PREFIX convention.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
HF_PREFIX = "issue617_wildchat_categories"

TASK_ID = 617
