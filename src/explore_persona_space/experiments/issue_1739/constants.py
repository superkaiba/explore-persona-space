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

# --- #779 r_B trait-direction bank ---
RB_REVISION = "037fcbb"  # #779 r_B bank pin
RB_PREFIX = "issue779_monitoring/r_b/"
RB_N_TRAITS = 3  # 3 traits x 28 layers x 3584 (plan v3 / #779)

# --- Sampling / DV pins (plan v3; constants only in round A) ---
K_ROLLOUTS = 5
N_JUDGE_DRAWS = 3
JUDGE_TEMPERATURE = 1.0
JUDGE_MAX_TOKENS = 400  # reason-then-score rubric floor (llm-judging.md rule 23)
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # project-wide judge pin (CLAUDE.md)
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SEEDS = (0, 1, 2)
N_LABELED_DRAWS = 5
U_LADDER = (250, 5_000, 50_000)  # unlabeled-rows ladder
L_LADDER = (250, 2_500, 16_000)  # labeled-rows ladder (evil L capped at EVIL_L_CAP)
EVIL_L_CAP = 8_000
