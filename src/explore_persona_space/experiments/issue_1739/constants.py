"""Pinned constants for issue #1739 (source per pin: plan v3 / #1092 / #779).

Every load-bearing value below was verified at plan time; do NOT retype from
memory — edit only against the plan or the producing task's artifacts.
"""

# --- Model geometry (plan v3; Qwen2.5-7B-Instruct) ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_DIM = 3584
N_LAYERS = 28
SUMMARY_DTYPE = "float16"  # #1092 summaries are stored fp16

# --- #1092 summary store (reused activation summaries) ---
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_REVISION = "e5901706"  # #1092 store pin (plan v3)
STORE_PREFIX = "issue1092_realistic_crossing/analysis_tensors/summaries/"
# t1 = answer-span mean, THE answer target this experiment consumes (t2/t3 unused).
SUMMARY_KINDS = ("prefix_end", "context_end", "t1")
# Realized #1092 full-corpus manifest: 21,193 rows, of which the tail 2,400
# battery rows carry is_eval_only=True -> 18,793 fit rows (plan v3 / #1092).
STORE_TOTAL_ROWS = 21_193
STORE_FIT_ROWS = 18_793
# U-pool source cell in the REALIZED store (gate0 finding: main cell_* dirs
# carry canonical kinds but NO row_index files). cell_inst_own = the instruct
# model on its own text — #1092's B0 reference cell (issue1092_fit_grid
# B0_CELLS / CELL_MODEL_TYPE).
U_STORE_CELL = "cell_inst_own"
# The cell dirs' row metadata is the #1092 corpus manifest (issue1092_fit_grid
# reads `corpus_dir / "manifest.jsonl"`). Pinned at #1092's OWN corpus pin
# (scripts/issue1092_prefixend_monitoring.py CORPUS_REV) — the manifest content
# the summaries were captured under; row count must equal STORE_TOTAL_ROWS.
CORPUS_MANIFEST_PATH = "issue1092_realistic_crossing/corpus/manifest.jsonl"
CORPUS_MANIFEST_REVISION = "7ef5523673d64697ab497577dbc5b9270c39f020"

# --- #779 r_B trait-direction bank ---
RB_REVISION = "037fcbb"  # #779 r_B bank pin
RB_PREFIX = "issue779_monitoring/r_b/"
RB_N_TRAITS = 3  # 3 traits x 28 layers x 3584 (plan v3 / #779)

# --- Sampling / DV pins (plan v3; constants only in round A) ---
K_ROLLOUTS = 5
N_JUDGE_DRAWS = 3
JUDGE_TEMPERATURE = 1.0
# JUSTIFIED DEVIATION from llm-judging rule 23's 1024 floor (#2063): the banked
# #1739 wave's instrument. The rejudge chain (issue1739_trait_rejudge.py,
# issue1739_sycoood_rescore_stage.py) deliberately preserves the original
# 400-token pool for provenance and re-judges under its OWN budget (1024).
# Fresh waves owe >=1024 — this is a parity pin, not a floor citation.
JUDGE_MAX_TOKENS = 400
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # project-wide judge pin (CLAUDE.md)
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SEEDS = (0, 1, 2)
N_LABELED_DRAWS = 5
U_LADDER = (250, 5_000, 50_000)  # unlabeled-rows ladder
L_LADDER = (250, 2_500, 16_000)  # labeled-rows ladder (evil L capped at EVIL_L_CAP)
EVIL_L_CAP = 8_000

# --- Phase-3 fit-engine pins (plan v3 §4 Phase 3 / §6; round C1) ---
N_FOLDS = 5  # group-level folds per cell (round-robin over groups; ood-generalization-folds.md)
N_BOOT = (
    500  # paired-bootstrap draws (round-C1 brief pin; plan §6 registers 2000 for the final run)
)
N_PERM = 500  # selection-symmetric permutation draws (max-over-arms null)
MLP_HIDDEN = 512  # arm-5/9 MLP width (single hidden layer; round-C1 brief pin)
MLP_MAX_EPOCHS = 300
E2_SPREAD_MIN = 15.0  # within-context score-spread floor for E2 qualifying contexts (plan §4 #24)
AUROC_POS_THRESHOLD = 50.0  # graded 0-100 DV binarization for the AUROC companion
KNN_KS = (1, 5)  # kNN retrieval ks (chance = k/n_pool stated by the helper)
WHITEN_SHRINKAGE_GRID = (0.01, 0.05, 0.1, 0.3)  # Sigma_g = (1-g)Sigma + g(tr/d)I, g by held-out NLL
WHITEN_HOLDOUT_FRAC = 0.2  # U-pool holdout for shrinkage selection + map diagnostics
COMPOSITION_F_U = (0.0, 0.5)  # composition factor (Config A evil only, plan §4b)
COMPOSITION_F_L = (0.0, 1.0)
