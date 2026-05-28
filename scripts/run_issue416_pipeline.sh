#!/usr/bin/env bash
# Issue #416 master pipeline runner -- falsify-or-confirm #398's global-marker-
# affinity-shift reading by re-running with software_engineer as source.
# Plan v3 at .claude/plans/issue-416.html (mirrored to tasks/running/416/plans/).
#
# Sequenced phases, fail-loud (set -euo pipefail). One nohup invocation drives
# the whole experiment; per-phase tee logs land under /workspace/logs/.
#
# Phase token contract for poll_pipeline.py
# -----------------------------------------
# Every phase boundary emits a line of the form ``[phase=<token>]`` to stdout
# (and therefore to /workspace/logs/issue-416.log). poll_pipeline parses the
# LATEST [phase=...] from the log tail (regex \[phase=([a-z_]+)\]). The final
# success sentinel is a literal ``[phase=done]`` line; if the driver PID exits
# WITHOUT having emitted that line, the poller reports the run as ``dead``.
#
# Phase tokens (in order): setup, smoke_train, smoke_check, training,
# eval_logp, eval_perpos, eval_spread, upload, done.
#
# Pod: epm-issue-416 (1x H100, intent lora-7b). Single sequential pipeline;
# no parallelism axis. Budget ~3.6 GPU-h per plan v3 §9 (with spread eval).
#
# Phases:
#   1. setup      -- env probe, fetch training data, marker-token id assert,
#                    predictors_base.json presence check (K3/K4 gates).
#   2. smoke_train-- 10-step train (max_steps=10, save_steps_list=[5,10]).
#   3. smoke_check-- smoke_i398_logp_check.py on the step-10 adapter (K1).
#   4. training   -- 1600-step main train, 22 checkpoints saved.
#   5. eval_logp  -- dual-probe (pos0 + endpos) teacher-forced log p eval.
#   6. eval_perpos-- on-policy per-position log p eval.
#   7. eval_spread-- vLLM substring-match eval (KEEP-with-raw-upload variant
#                    per plan §7 planner recommendation). Per-step raw
#                    completions auto-upload to HF data repo via
#                    --upload-after-each-step (default True in the script).
#   8. upload     -- stage per-position raw completions for upload + final
#                    sweep call to upload_raw_completions_to_data_repo. The
#                    spread eval already uploaded per-step; this catches any
#                    residual files + the per-position JSON.
#   9. done       -- write results sentinel, emit [phase=done].
#
# Driver MUST NOT shell scripts/task.py (pod-side branch-guard forbidden per
# CLAUDE.md). All workflow-state mutations go through the orchestrator on the
# local VM, which reads the sentinel and posts epm:results.

set -euo pipefail

# -- Environment -------------------------------------------------------------

export PATH="/root/.local/bin:${PATH}"
export HF_HOME="/workspace/.cache/huggingface"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1   # disk-quota safety (plan §8 R4)

# Load .env so SSH non-login shell has HF/WandB keys.
set -a
# shellcheck disable=SC1091  # .env is environment-specific, not vcs'd alongside script
source /workspace/explore-persona-space/.env
set +a

cd /workspace/explore-persona-space

LOG_DIR=/workspace/logs
mkdir -p "${LOG_DIR}"

# Track wall time for gpu_hours_used best-effort estimate in the sentinel.
SECONDS=0

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
banner() { echo; echo "================================================================"; echo "[$(ts)] $*"; echo "================================================================"; }

CONDITION="i416_software_engineer_marker_spread_zen"
SEED=42
RUN_NAME="${CONDITION}_seed${SEED}"
CKPT_DIR="/workspace/explore-persona-space/models/${RUN_NAME}/marker_implant_step_checkpoints"
RESULTS_DIR="/workspace/explore-persona-space/eval_results/issue_416"
FIGURES_DIR="/workspace/explore-persona-space/figures/issue_416"
mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}"

# 22-checkpoint schedule (plan §3 + Reproducibility card). Comma-separated.
STEPS_LIST="5,10,15,20,25,30,40,50,60,65,70,75,100,150,200,300,400,600,800,1000,1200,1600"
# Hydra-compatible quoted-list form for +training.save_steps_list=[...].
STEPS_HYDRA="[${STEPS_LIST}]"

MARKER="※"
SOURCE_PERSONA="software_engineer"
PANEL_MOD="_i416_bystander_panel"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATA_FILE_BUCKET="leakage/marker_software_engineer_asst_excluded_medium_9ca040.jsonl"
DATA_FILE_LOCAL="data/leakage_experiment/marker_software_engineer_asst_excluded_medium_9ca040.jsonl"

# -- Phase 1: setup -----------------------------------------------------------

echo "[phase=setup]"
echo "$$" > "${LOG_DIR}/issue-416.pid"

banner "Phase 1 setup -- env probe + training data fetch + token-id assert"
echo "git HEAD:   $(git rev-parse HEAD)"
echo "git branch: $(git branch --show-current)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
df -h /workspace | tail -2

# Predictors file gate (K-class invariant: analyzer needs it AFTER eval; if
# missing, fail fast here rather than burning ~4 GPU-h on an un-analyzable run).
if [[ ! -f eval_results/issue_385/predictors_base.json ]]; then
  echo "[$(ts)] FAIL: eval_results/issue_385/predictors_base.json missing."
  echo "[$(ts)]   Should have been committed in 4fec1850 (pull from 0d9360f1)."
  exit 1
fi
echo "[$(ts)] predictors_base.json present."

# Pull training data from HF data repo.
banner "Phase 1 -- hf_hub_download training data"
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='superkaiba1/explore-persona-space-data',
    repo_type='dataset',
    filename='${DATA_FILE_BUCKET}',
    local_dir='data',
)
print(f'downloaded -> {p}')
"
mkdir -p "$(dirname "${DATA_FILE_LOCAL}")"
cp "data/${DATA_FILE_BUCKET}" "${DATA_FILE_LOCAL}"

# K3: row count assertion. Training data must be exactly 600 rows.
row_count=$(wc -l < "${DATA_FILE_LOCAL}")
if [[ "${row_count}" -ne 600 ]]; then
  echo "[$(ts)] FAIL K3: ${DATA_FILE_LOCAL} has ${row_count} rows, expected 600."
  exit 1
fi
echo "[$(ts)] K3 OK: ${DATA_FILE_LOCAL} = 600 rows."

# K4: marker-token-id assertion. ※ MUST tokenize to [63680].
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
ids = tok.encode('${MARKER}', add_special_tokens=False)
assert ids == [63680], f'FAIL K4: expected [63680], got {ids}'
print('K4 OK: ${MARKER} -> [63680]')
"

# -- Phase 2: smoke_train ----------------------------------------------------

echo "[phase=smoke_train]"
banner "Phase 2 smoke_train -- 10-step train (max_steps=10, save_steps=[5,10])"

uv run python scripts/train.py \
    condition=${CONDITION} \
    seed=${SEED} \
    +training.save_at_specific_steps=true \
    "+training.save_steps_list=[5,10]" \
    training.learning_rate=1.0e-5 \
    training.max_steps=10 \
    training.epochs=-1 \
    upload_to=none \
    periodic_eval.enabled=false \
    2>&1 | tee -a "${LOG_DIR}/issue-416_smoke_train.log"

# -- Phase 3: smoke_check ----------------------------------------------------

echo "[phase=smoke_check]"
banner "Phase 3 smoke_check -- log p check on step-10 adapter (K1 gate)"

SMOKE_ADAPTER="${CKPT_DIR}/checkpoint-10"
if [[ ! -d "${SMOKE_ADAPTER}" ]]; then
  echo "[$(ts)] FAIL K1: step-10 adapter missing at ${SMOKE_ADAPTER}"
  exit 1
fi

# smoke_i398_logp_check.py exits 0 PASS / 1 base-floor FAIL / 2 delta FAIL.
# set -euo pipefail aborts the whole pipeline on any non-zero -- which is
# the correct K1 signal (pipeline exits without [phase=done]; poller flags
# as dead, orchestrator picks up the exit code + the partial log).
uv run python scripts/smoke_i398_logp_check.py \
    --adapter "${SMOKE_ADAPTER}" \
    --base-model "${BASE_MODEL}" \
    --marker-token "${MARKER}" \
    --source-persona "${SOURCE_PERSONA}" \
    --panel-module "${PANEL_MOD}" \
    --num-prompts 3 \
    --output "${RESULTS_DIR}/smoke_log.json" \
    2>&1 | tee -a "${LOG_DIR}/issue-416_smoke_check.log"

echo "[$(ts)] K1 OK: smoke check PASSed."

# -- Phase 4: training -------------------------------------------------------

echo "[phase=training]"
banner "Phase 4 training -- 1600 steps, 22 checkpoints (~38 min wall)"

uv run python scripts/train.py \
    condition=${CONDITION} \
    seed=${SEED} \
    +training.save_at_specific_steps=true \
    "+training.save_steps_list=${STEPS_HYDRA}" \
    training.learning_rate=1.0e-5 \
    training.max_steps=1600 \
    training.epochs=-1 \
    upload_to=hf \
    periodic_eval.enabled=false \
    2>&1 | tee -a "${LOG_DIR}/issue-416_train.log"

# -- Phase 5: eval_logp ------------------------------------------------------

echo "[phase=eval_logp]"
banner "Phase 5 eval_logp -- dual-probe teacher-forced log p (22 ckpts) ~25 min"

uv run python scripts/eval_i398_marker_logprob.py \
    --run-dir "${CKPT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --steps "${STEPS_LIST}" \
    --marker-token "${MARKER}" \
    --panel-module "${PANEL_MOD}" \
    --output "${RESULTS_DIR}/logp_seed42.json" \
    --batch-size 8 \
    2>&1 | tee -a "${LOG_DIR}/issue-416_eval_logp.log"

# -- Phase 6: eval_perpos ----------------------------------------------------

echo "[phase=eval_perpos]"
banner "Phase 6 eval_perpos -- on-policy per-position log p (22 ckpts) ~45 min"

uv run python scripts/eval_i398_per_position.py \
    --run-dir "${CKPT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --steps "${STEPS_LIST}" \
    --marker-token "${MARKER}" \
    --panel-module "${PANEL_MOD}" \
    --max-new-tokens 2048 \
    --batch-size 8 \
    --output "${RESULTS_DIR}/per_position_seed42.json" \
    2>&1 | tee -a "${LOG_DIR}/issue-416_eval_perpos.log"

# -- Phase 7: eval_spread ----------------------------------------------------

echo "[phase=eval_spread]"
banner "Phase 7 eval_spread -- vLLM substring eval (22 ckpts) ~85 min; per-step raw uploads"

# --raw-completions-output triggers per-step raw_completions.json write;
# --upload-after-each-step (default True) calls upload_raw_completions_to_data_repo
# after every checkpoint -- closes #398's open Next-step.
uv run python scripts/eval_i398_marker_spread.py \
    --run-dir "${CKPT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --steps "${STEPS_LIST}" \
    --marker-token "${MARKER}" \
    --panel-module "${PANEL_MOD}" \
    --num-completions 8 \
    --max-tokens 2048 \
    --output "${RESULTS_DIR}/spread_seed42.json" \
    --raw-completions-output "${RESULTS_DIR}/spread_raw_completions/" \
    --experiment-name "issue416" \
    2>&1 | tee -a "${LOG_DIR}/issue-416_eval_spread.log"

# -- Phase 8: upload ---------------------------------------------------------

echo "[phase=upload]"
banner "Phase 8 upload -- stage per-position raw + sweep-upload to HF data repo"

# eval_i398_per_position.py writes ONE per_position_seed42.json at end-of-eval
# (not per-step). To consume it through upload_raw_completions_to_data_repo
# (which rglobs for files named raw_completions.json), stage a directory
# alongside the spread bucket. Use cp (NOT mv) so the original JSON stays in
# place for downstream commits by the orchestrator.
PERPOS_BUCKET_DIR="${RESULTS_DIR}/per_position_raw"
mkdir -p "${PERPOS_BUCKET_DIR}"
cp "${RESULTS_DIR}/per_position_seed42.json" "${PERPOS_BUCKET_DIR}/raw_completions.json"
echo "[$(ts)] staged per_position_seed42.json -> ${PERPOS_BUCKET_DIR}/raw_completions.json"

# Final sweep upload: catches the per-position raw + any spread-step files that
# the spread eval's per-step upload missed (defensive; the spread eval already
# uploaded per-step but a re-upload is idempotent via the _upload verify pass).
uv run python -c "
from pathlib import Path
from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo
uploaded = upload_raw_completions_to_data_repo(
    experiment_name='issue416',
    eval_results_dir=Path('${RESULTS_DIR}'),
)
print(f'final sweep: {len(uploaded)} raw_completions.json files uploaded')
for rel, url in sorted(uploaded.items()):
    print(f'  {rel} -> {url}')
" 2>&1 | tee -a "${LOG_DIR}/issue-416_upload.log"

# -- Phase 9: results sentinel + done ----------------------------------------

banner "Phase 9 -- write results sentinel + emit [phase=done]"

# Grep the WandB run URL from the training log (best-effort -- the run line
# looks like "wandb: View run at https://wandb.ai/.../runs/<id>").
WANDB_URL=$(grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[a-zA-Z0-9_-]+' "${LOG_DIR}/issue-416_train.log" 2>/dev/null | tail -1 || true)
if [[ -z "${WANDB_URL}" ]]; then
  WANDB_URL="unknown"
  echo "[$(ts)] WARN: could not extract WandB URL from training log; sentinel will record 'unknown'."
fi

FINAL_COMMIT=$(git rev-parse HEAD)
GPU_HOURS_USED=$(awk -v s="${SECONDS}" 'BEGIN { printf "%.3f", s / 3600.0 }')

# Compute the inline eval_numbers (swe + librarian step-5 and step-1600 mean
# pos0 logp + ranks across the panel) directly from logp_seed42.json. This is
# best-effort; failure here doesn't abort the pipeline -- the analyzer can
# recompute. We wrap in a try/except so a missing key writes {} not a crash.
EVAL_NUMS=$(uv run python -c "
import json
try:
    d = json.load(open('${RESULTS_DIR}/logp_seed42.json'))
    panel = d['panel']
    def mean_pos0(step, persona):
        return float(sum(d['per_step'][str(step)][persona]['pos0']) / len(d['per_step'][str(step)][persona]['pos0']))
    def rank_in_panel(step, persona):
        means = {p: mean_pos0(step, p) for p in panel}
        return sorted(means.items(), key=lambda kv: -kv[1]).index((persona, means[persona])) + 1
    out = {
        'panel_size': len(panel),
        'swe_step5_mean_pos0_logp': mean_pos0(5, '${SOURCE_PERSONA}'),
        'swe_step1600_mean_pos0_logp': mean_pos0(1600, '${SOURCE_PERSONA}'),
        'swe_rank_step5': rank_in_panel(5, '${SOURCE_PERSONA}'),
        'swe_rank_step1600': rank_in_panel(1600, '${SOURCE_PERSONA}'),
        'librarian_rank_step5': rank_in_panel(5, 'librarian'),
        'librarian_rank_step1600': rank_in_panel(1600, 'librarian'),
    }
    print(json.dumps(out))
except Exception as e:
    print('{}')
")

# HF Hub adapter URL: train.py with upload_to=hf uploads under DEFAULT_MODEL_REPO
# at path {run_name}_<stage>_checkpoints. The exact subpath depends on runner.py;
# we record the repo root + best-known prefix and let the orchestrator (or the
# analyzer) resolve to the precise tree URL.
HF_HUB_URL="superkaiba1/explore-persona-space/${RUN_NAME}_marker_implant_step_checkpoints"

uv run python -c "
import json
sentinel = {
    'eval_numbers': ${EVAL_NUMS},
    'eval_paths': [
        'eval_results/issue_416/logp_seed42.json',
        'eval_results/issue_416/per_position_seed42.json',
        'eval_results/issue_416/spread_seed42.json',
    ],
    'reproducibility_card': {
        'base_model': '${BASE_MODEL}',
        'marker_token': '${MARKER}',
        'marker_token_id': 63680,
        'source_persona': '${SOURCE_PERSONA}',
        'condition': '${CONDITION}',
        'learning_rate': 1.0e-5,
        'max_steps': 1600,
        'seed': ${SEED},
        'lora_r': 32,
        'lora_alpha': 64,
        'save_steps_list': [int(s) for s in '${STEPS_LIST}'.split(',')],
        'panel_module': '${PANEL_MOD}',
        'training_data': '${DATA_FILE_LOCAL}',
        'training_data_rows': 600,
    },
    'wandb_url': '${WANDB_URL}',
    'hf_hub_url': '${HF_HUB_URL}',
    'worktree_path': '/workspace/explore-persona-space',
    'final_commit_sha': '${FINAL_COMMIT}',
    'gpu_hours_used': ${GPU_HOURS_USED},
    'gpu_hours_budgeted': 3.6,
    'plan_deviations': [],
}
with open('${LOG_DIR}/issue-416-results.json', 'w') as f:
    json.dump(sentinel, f, indent=2)
print('wrote ${LOG_DIR}/issue-416-results.json')
print(json.dumps(sentinel, indent=2))
"

banner "Issue #416 pipeline COMPLETE"
echo "Final disk:"
df -h /workspace | tail -2
echo "Eval results:"
ls -la "${RESULTS_DIR}/" 2>/dev/null || true
echo "[$(ts)] All phases finished successfully. gpu_hours_used=${GPU_HOURS_USED} (budgeted=3.6)"

echo "[phase=done]"
