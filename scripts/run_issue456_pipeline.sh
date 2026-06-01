#!/usr/bin/env bash
# Issue #456 master pipeline runner -- re-run #432's EXACT training recipe, then
# evaluate it ON-POLICY (vLLM-generate each persona's own answer; measure marker
# emission rate + on-policy end-of-answer log p), plus a control re-score of
# #432's old fixed-stub teacher-forced probe on the regenerated checkpoints AND
# on the base model. The ONLY variable changed vs #432 is the eval method.
#
# Plan: tasks/.../456/plans/v3.md (approved). 1x H100, ~7.1 GPU-h.
#
# Sequenced phases, fail-loud (set -euo pipefail). One nohup invocation drives
# the whole experiment; per-phase tee logs land under /workspace/logs/.
#
# Phase token contract for poll_pipeline.py
# -----------------------------------------
# Every phase boundary emits ``[phase=<token>]`` to stdout (-> issue-456.log).
# poll_pipeline parses the LATEST [phase=...] from the log tail
# (PHASE_RE = \[phase=([a-z_]+)). The final success sentinel is a literal
# ``[phase=done]`` line; if the driver PID exits WITHOUT it, the poller reports
# the run as ``dead`` and suppresses the auto-post of epm:results.
#
# Phase tokens (in order): setup, smoke_train, smoke_eval, training,
# eval_onpolicy, eval_oldprobe, done.
#
# Driver MUST NOT shell scripts/task.py (pod-side branch-guard forbidden per
# CLAUDE.md). All workflow-state mutations go through the orchestrator on the
# local VM, which reads the sentinel and posts epm:results.

set -euo pipefail

# -- Environment -------------------------------------------------------------

export PATH="/root/.local/bin:${PATH}"
export HF_HOME="/workspace/.cache/huggingface"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1   # MooseFS disk-quota safety (CLAUDE.md gotcha)

# Load .env so the SSH non-login shell has HF/WandB keys (uv run does NOT
# auto-load .env; without this the subprocesses spawn with credentials missing).
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

CONDITION="i432_software_engineer_marker_9neg_zen"
SEED=42
RUN_NAME="${CONDITION}_seed${SEED}"
CKPT_DIR="/workspace/explore-persona-space/models/${RUN_NAME}/marker_implant_step_checkpoints"
RESULTS_DIR="/workspace/explore-persona-space/eval_results/issue_456"
FIGURES_DIR="/workspace/explore-persona-space/figures/issue_456"
mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}"

# 22-checkpoint schedule (IDENTICAL to #432). Comma-separated.
STEPS_LIST="5,10,15,20,25,30,40,50,60,65,70,75,100,150,200,300,400,600,800,1000,1200,1600"
STEPS_HYDRA="[${STEPS_LIST}]"
# On-policy subset = 12 of 22 (dense early; preserves the learning curve at
# half the generation cost). Old-probe control runs all 22.
ONPOLICY_STEPS="5,30,65,100,200,400,600,800,1000,1200,1600,75"

MARKER="※"
SOURCE_PERSONA="software_engineer"
# Eval panel module: #432's panel verbatim (swe source + 27 bystanders = 28).
PANEL_MOD="_i416_bystander_panel"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATA_FILE_BUCKET="leakage/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl"
DATA_FILE_LOCAL="data/leakage_experiment/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl"

# -- Phase 1: setup -----------------------------------------------------------

echo "[phase=setup]"
echo "$$" > "${LOG_DIR}/issue-456.pid"

# Marker-token-id assertion FIRST, before ANY other subprocess (git, nvidia-smi,
# data fetch). Contract: assert the marker tokenizes correctly before any work --
# a wrong marker token invalidates the entire train+eval, so fail in <30s on the
# tokenizer load, not after the data download. This is the EXACT marker #432
# trained (bare ※ -> [63680] under Qwen-2.5 BPE, NOT the global default
# 83399 = ' ※' with a leading space).
banner "Phase 1 -- marker-token-id assert (BEFORE any other subprocess)"
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
ids = tok.encode('${MARKER}', add_special_tokens=False)
assert ids == [63680], f'FAIL: expected [63680], got {ids}'
print('OK: ${MARKER} -> [63680]')
"

banner "Phase 1 setup -- env probe + branch-port check + data fetch"
echo "git HEAD:   $(git rev-parse HEAD)"
echo "git branch: $(git branch --show-current)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
df -h /workspace | tail -2

# Branch-port invariant: SaveAtSpecificSteps wiring + the 5 ported scripts MUST
# be present, else the 1600-step train saves ZERO checkpoints (silent) and the
# eval phases have nothing to score. Fail fast here, not after ~0.7 GPU-h.
banner "Phase 1 -- verify ported files + SaveAtSpecificSteps wiring"
if ! git grep -q "save_at_specific_steps" -- src/explore_persona_space/train/trainer.py; then
  echo "[$(ts)] FAIL: save_at_specific_steps wiring NOT in trainer.py (port lost)."
  exit 1
fi
if ! git grep -q "class SaveAtSpecificSteps" -- src/explore_persona_space/train/callbacks.py; then
  echo "[$(ts)] FAIL: SaveAtSpecificSteps class NOT in callbacks.py (port lost)."
  exit 1
fi
for f in \
  configs/condition/i432_software_engineer_marker_9neg_zen.yaml \
  scripts/_i416_bystander_panel.py \
  scripts/eval_i398_marker_logprob.py \
  scripts/eval_i456_onpolicy_emission.py \
  scripts/smoke_i398_logp_check.py ; do
  if [[ ! -f "${f}" ]]; then
    echo "[$(ts)] FAIL: required file missing: ${f}"
    exit 1
  fi
done
echo "[$(ts)] OK: SaveAtSpecificSteps wiring + ported files present."

# Pull training data from HF data repo (built by scripts/build_i432_9neg_dataset.py).
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

# Row count assertion: training data must be exactly 2000 rows (200 pos / 1800 neg).
row_count=$(wc -l < "${DATA_FILE_LOCAL}")
if [[ "${row_count}" -ne 2000 ]]; then
  echo "[$(ts)] FAIL: ${DATA_FILE_LOCAL} has ${row_count} rows, expected 2000."
  exit 1
fi
echo "[$(ts)] OK: ${DATA_FILE_LOCAL} = 2000 rows (200 positive + 1800 negative)."

# -- Phase 2: smoke_train ----------------------------------------------------

echo "[phase=smoke_train]"
banner "Phase 2 smoke_train -- 10-step train (max_steps=10, save_steps=[5,10])"

uv run python scripts/train.py \
    condition=${CONDITION} \
    seed=${SEED} \
    ++training.save_at_specific_steps=true \
    "++training.save_steps_list=[5,10]" \
    ++training.learning_rate=1.0e-5 \
    ++training.max_steps=10 \
    ++training.epochs=-1 \
    upload_to=none \
    eval.periodic_eval.enabled=false \
    2>&1 | tee -a "${LOG_DIR}/issue-456_smoke_train.log"

SMOKE_ADAPTER="${CKPT_DIR}/checkpoint-10"
if [[ ! -d "${SMOKE_ADAPTER}" ]]; then
  echo "[$(ts)] FAIL: step-10 adapter missing at ${SMOKE_ADAPTER} (SaveAtSpecificSteps no-op)."
  exit 1
fi
echo "[$(ts)] OK: step-10 + step-5 adapters saved."

# -- Phase 3: smoke_eval -----------------------------------------------------
# Runs the SAME on-policy dispatcher used in Phase 5, parameterized down to a
# tiny subset (smoke = sweep code path; unification). The gate: completions
# reach end-of-answer (no systematic truncation -> the dispatcher raises if
# >10% of SOURCE completions hit the cap), marker-id assert (inside the
# dispatcher), chat-template renders for the (smoke-subset) panel incl. the
# source. A FULL 28-persona render check is run here too, cheaply, on CPU-free
# chat-template rendering (no model load) so no persona silently breaks at the
# 1600-step eval.

echo "[phase=smoke_eval]"
banner "Phase 3 smoke_eval -- full-panel render check + tiny on-policy gen (step-10)"

# Full 28-persona chat-template render check (catches a broken persona before
# the 1600-step train; cheap, no model weights loaded).
uv run python -c "
import os, sys
sys.path.insert(0, 'scripts')
import _i416_bystander_panel as p
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
panel = {p.SOURCE_PERSONA: dict(p.PERSONAS)[p.SOURCE_PERSONA]}
panel.update(p.BYSTANDERS)
assert len(panel) == 28, f'panel size {len(panel)} != 28'
q = p.PROMPTS[0]
bad = []
for name, text in panel.items():
    r = tok.apply_chat_template(
        [{'role':'system','content':text},{'role':'user','content':q}],
        tokenize=False, add_generation_prompt=True)
    if not r or not r.strip():
        bad.append(name)
assert not bad, f'empty chat-template render for: {bad}'
print(f'OK: all 28 personas render (incl no_persona empty system).')
"

# Tiny on-policy generation on the step-10 smoke adapter -- SAME code path as
# Phase 5, just --steps 10 --n-samples 2 + a tiny persona/prompt subset. The
# dispatcher raises if >10% of source completions truncate (silent-zero guard).
uv run python scripts/eval_i456_onpolicy_emission.py \
    --run-dir "${CKPT_DIR}" \
    --steps 10 \
    --out-dir "${RESULTS_DIR}/smoke" \
    --n-samples 2 \
    --smoke-personas 3 \
    --smoke-prompts 3 \
    --max-new-tokens 1536 \
    --max-model-len 4096 \
    --panel-module "${PANEL_MOD}" \
    --seed ${SEED} \
    2>&1 | tee -a "${LOG_DIR}/issue-456_smoke_eval.log"

# Median-completion-length guard: confirm the smoke generation did not
# systematically hit the cap (a second belt to the dispatcher's >10% raise).
uv run python -c "
import json, statistics, sys
from pathlib import Path
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
gen = json.load(open('${RESULTS_DIR}/smoke/onpolicy_gen/onpolicy_gen_step10.json'))
lengths = []
for persona, qmap in gen['completions'].items():
    for q_idx, texts in qmap.items():
        for t in texts:
            lengths.append(len(tok.encode(t, add_special_tokens=False)))
med = statistics.median(lengths) if lengths else 0
cap = gen['max_new_tokens']
print(f'smoke completion token lengths: n={len(lengths)} median={med} cap={cap}')
if med >= cap:
    print(f'FAIL: median completion length {med} >= max_new_tokens {cap} -- raise the cap.')
    sys.exit(1)
print('OK: median completion length well under the cap (no systematic truncation).')
"

echo "[$(ts)] smoke_eval PASSed (render + tiny gen + length guard)."

# -- Phase 4: training -------------------------------------------------------

echo "[phase=training]"
banner "Phase 4 training -- 1600 steps, 22 checkpoints (IDENTICAL to #432) ~40 min"

uv run python scripts/train.py \
    condition=${CONDITION} \
    seed=${SEED} \
    ++training.save_at_specific_steps=true \
    "++training.save_steps_list=${STEPS_HYDRA}" \
    ++training.learning_rate=1.0e-5 \
    ++training.max_steps=1600 \
    ++training.epochs=-1 \
    upload_to=hf \
    eval.periodic_eval.enabled=false \
    2>&1 | tee -a "${LOG_DIR}/issue-456_train.log"

# Confirm all 22 checkpoints landed.
missing_ckpts=()
IFS=',' read -ra _STEPS <<< "${STEPS_LIST}"
for s in "${_STEPS[@]}"; do
  [[ -d "${CKPT_DIR}/checkpoint-${s}" ]] || missing_ckpts+=("${s}")
done
if [[ ${#missing_ckpts[@]} -gt 0 ]]; then
  echo "[$(ts)] FAIL: missing checkpoints: ${missing_ckpts[*]}"
  exit 1
fi
echo "[$(ts)] OK: all 22 checkpoints present."

# -- Phase 5: eval_onpolicy --------------------------------------------------

echo "[phase=eval_onpolicy]"
banner "Phase 5 eval_onpolicy -- on-policy emission rate + endpos log p (12 ckpts) ~5h"

uv run python scripts/eval_i456_onpolicy_emission.py \
    --run-dir "${CKPT_DIR}" \
    --steps "${ONPOLICY_STEPS}" \
    --out-dir "${RESULTS_DIR}" \
    --n-samples 8 \
    --max-new-tokens 1536 \
    --max-model-len 4096 \
    --panel-module "${PANEL_MOD}" \
    --seed ${SEED} \
    --batch-size 8 \
    2>&1 | tee -a "${LOG_DIR}/issue-456_eval_onpolicy.log"

# -- Phase 6: eval_oldprobe --------------------------------------------------
# Control: re-score #432's UNCHANGED fixed-stub/pos0 teacher-forced probe on
# (a) the BASE model (step-0, no adapter) and (b) all 22 trained checkpoints.
# Persist the FULL per-persona vector for base + every checkpoint. This breaks
# the H4 circularity (adapter-sensitive probe vs near-degenerate probe).

echo "[phase=eval_oldprobe]"
banner "Phase 6 eval_oldprobe -- base-model arm + 22 trained ckpts (control) ~30 min"

# (a) Base-model arm (step-0, no adapter).
uv run python scripts/eval_i398_marker_logprob.py \
    --base-only \
    --base-model "${BASE_MODEL}" \
    --marker-token "${MARKER}" \
    --panel-module "${PANEL_MOD}" \
    --output "${RESULTS_DIR}/oldprobe_base_step0.json" \
    --batch-size 8 \
    2>&1 | tee -a "${LOG_DIR}/issue-456_eval_oldprobe.log"

# (b) All 22 trained checkpoints.
uv run python scripts/eval_i398_marker_logprob.py \
    --run-dir "${CKPT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --steps "${STEPS_LIST}" \
    --marker-token "${MARKER}" \
    --panel-module "${PANEL_MOD}" \
    --output "${RESULTS_DIR}/oldprobe_trained.json" \
    --batch-size 8 \
    2>&1 | tee -a "${LOG_DIR}/issue-456_eval_oldprobe.log"

# -- Phase 7: results sentinel + done ----------------------------------------

banner "Phase 7 -- write results sentinel + emit [phase=done]"

# Grep the WandB run URL from the training log (best-effort).
WANDB_URL=$(grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[a-zA-Z0-9_-]+' "${LOG_DIR}/issue-456_train.log" 2>/dev/null | tail -1 || true)
if [[ -z "${WANDB_URL}" ]]; then
  WANDB_URL="unknown"
  echo "[$(ts)] WARN: could not extract WandB URL from training log; sentinel records 'unknown'."
fi

FINAL_COMMIT=$(git rev-parse HEAD)
GPU_HOURS_USED=$(awk -v s="${SECONDS}" 'BEGIN { printf "%.3f", s / 3600.0 }')

# Best-effort inline eval_numbers: source on-policy emission rate + rank at the
# end-state (step 1600). Failure here doesn't abort -- the analyzer recomputes.
EVAL_NUMS=$(uv run python -c "
import json
try:
    g = json.load(open('${RESULTS_DIR}/onpolicy_gen/onpolicy_gen_step1600.json'))
    er = g['emission_rate']
    panel = g['panel']
    ranked = sorted(panel, key=lambda p: -er[p])
    src = g['source_persona']
    out = {
        'panel_size': len(panel),
        'src_emission_rate_step1600': er[src],
        'src_emission_rank_step1600': ranked.index(src) + 1,
        'runner_up_emission_rate_step1600': er[ranked[1]] if len(ranked) > 1 else None,
        'src_truncation_frac_step1600': g.get('source_truncation_frac'),
    }
    print(json.dumps(out))
except Exception as e:
    print('{}')
")

HF_HUB_URL="superkaiba1/explore-persona-space/${RUN_NAME}_marker_implant_step_checkpoints"

# Sentinel filename conforms to poll_pipeline.py: issue-<N>-<kind_slug>-<epoch>.json
# kind_slug = 'epm:results' with ':' -> '_'. Required keys: sentinel_schema_version,
# kind, version. This is the ONLY file the poller drains. A human-readable copy is
# also written to issue456_results_human.txt (a name OUTSIDE the issue-456-*.json
# drain glob, so it is never parsed as a second sentinel).
EPOCH=$(date +%s)
SENTINEL_PATH="${LOG_DIR}/issue-456-epm_results-${EPOCH}.json"

uv run python -c "
import json
sentinel = {
    'sentinel_schema_version': 1,
    'task_id': 456,
    'kind': 'epm:results',
    'version': 1,
    'gate': None,
    'blocks_pipeline': False,
    'by': 'run_issue456_pipeline.sh',
    'note': json.dumps({
        'eval_numbers': ${EVAL_NUMS},
        'eval_paths': [
            'eval_results/issue_456/onpolicy_gen/',
            'eval_results/issue_456/onpolicy_endpos_logp/',
            'eval_results/issue_456/oldprobe_base_step0.json',
            'eval_results/issue_456/oldprobe_trained.json',
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
            'lora_dropout': 0.0,
            'use_rslora': True,
            'lora_target_modules': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            'save_steps_list': [int(s) for s in '${STEPS_LIST}'.split(',')],
            'onpolicy_steps': [int(s) for s in '${ONPOLICY_STEPS}'.split(',')],
            'panel_module': '${PANEL_MOD}',
            'training_data': '${DATA_FILE_LOCAL}',
            'training_data_rows': 2000,
            'gen_n_samples': 8,
            'gen_temperature': 1.0,
            'gen_top_p': 1.0,
            'gen_max_new_tokens': 1536,
            'parent_task': 432,
        },
        'wandb_url': '${WANDB_URL}',
        'hf_hub_url': '${HF_HUB_URL}',
        'worktree_path': '/workspace/explore-persona-space',
        'final_commit_sha': '${FINAL_COMMIT}',
        'gpu_hours_used': ${GPU_HOURS_USED},
        'gpu_hours_budgeted': 7.1,
        'plan_deviations': [],
    }),
}
with open('${SENTINEL_PATH}', 'w') as f:
    json.dump(sentinel, f, indent=2)
# Human-readable convenience copy. MUST NOT match poll_pipeline.py's drain glob
# (issue-456-*.json), else it is parsed as a SECOND valid sentinel and epm:results
# is double-posted. The .txt suffix keeps it out of the glob (it stays in LOG_DIR
# for SSH inspection but is never drained).
with open('${LOG_DIR}/issue456_results_human.txt', 'w') as f:
    json.dump(sentinel, f, indent=2)
print('wrote ${SENTINEL_PATH}')
print(json.dumps(json.loads(sentinel['note'])['eval_numbers'], indent=2))
"

banner "Issue #456 pipeline COMPLETE"
echo "Final disk:"
df -h /workspace | tail -2
echo "Eval results:"
ls -la "${RESULTS_DIR}/" 2>/dev/null || true
echo "[$(ts)] All phases finished. gpu_hours_used=${GPU_HOURS_USED} (budgeted=7.1)"

echo "[phase=done]"
