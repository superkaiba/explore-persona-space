#!/usr/bin/env bash
# Issue #404 master pipeline runner — head-to-head test of three cheap base-model
# predictors of cross-behavior leakage. Plan v3 at tasks/running/404/plans/plan.md.
#
# Sequenced phases, fail-loud (set -euo pipefail; tee CANNOT mask phase exits
# because we redirect at the wrapper level via nohup, not pipe).
#
# Pod: pod-404 (4× H100, intent ft-7b). Plan §9 had spec'd 8× H100 → we run
# SFT in 3 batches of (4, 4, 2) instead of (8, 2). ~15min extra wall.
#
# Phases:
#   A. Data prep + uploads (medical via Turner→Claude regen fallback; JSON neg
#      via Claude batch; Betley files fetched lazily by predictor scripts).
#   B. Predictor measurement (cossim + kldiv + incontext) — base model only.
#   C. SFT — 10 LoRA-7B runs (5 pairs × 2 seeds), batched 4+4+2 across 4 GPUs.
#   D. Merge + upload each (pair, seed) adapter to HF Hub.
#   E. Outcome eval (judge calibration first, then all 10 cells).
#   F. Regression + figures + final uploads.
#
# Skip-if-medical-dropped: scripts/fetch_or_generate_issue404_medical.py exits
# 3 if both Turner and Claude regen fail. We catch that and drop `bad_medical`
# from the SFT/eval lists for the remaining phases (plan v3 §4.1 MF2 Step 3).

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────

export PATH="/root/.local/bin:${PATH}"
export HF_HOME="/workspace/.cache/huggingface"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1   # disk-quota safety (plan §8)

# Load .env so SSH non-login shell has Anthropic/HF/WandB/OpenAI keys.
set -a
source /workspace/explore-persona-space/.env
set +a

cd /workspace/explore-persona-space

LOG_DIR=/workspace/logs
mkdir -p "${LOG_DIR}"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
banner() { echo; echo "================================================================"; echo "[$(ts)] $*"; echo "================================================================"; }

banner "Issue #404 pipeline START (pod=pod-404, gpus=4× H100)"
echo "git HEAD: $(git rev-parse HEAD)"
echo "git branch: $(git branch --show-current)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
df -h /workspace | tail -2

# ── Phase A: Data prep ───────────────────────────────────────────────────────

banner "Phase A1 — fetch_or_generate medical dataset (Turner → Claude regen → drop)"
MEDICAL_DROPPED=0
# Exit codes: 0 = locked (Turner or Claude regen), 3 = drop pair, 4 = unexpected.
set +e
uv run python scripts/fetch_or_generate_issue404_medical.py 2>&1 | tee -a "${LOG_DIR}/issue-404_medical_fetch.log"
MEDICAL_EXIT=${PIPESTATUS[0]}
set -e
if [[ ${MEDICAL_EXIT} -eq 0 ]]; then
  echo "[$(ts)] Medical dataset LOCKED."
elif [[ ${MEDICAL_EXIT} -eq 3 ]]; then
  echo "[$(ts)] Medical dataset DROPPED (Turner AND Claude regen failed). Will run at N=4."
  MEDICAL_DROPPED=1
else
  echo "[$(ts)] fetch_or_generate_issue404_medical.py crashed with code ${MEDICAL_EXIT}. Aborting."
  exit ${MEDICAL_EXIT}
fi

banner "Phase A2 — generate JSON-neg dataset (Claude batch)"
uv run python scripts/generate_issue404_json_neg.py

# Build PAIR list for downstream phases. Drop bad_medical if it was dropped.
ALL_PAIRS=(insecure_code bad_medical hitler_90 json_neg educational_neg)
if [[ ${MEDICAL_DROPPED} -eq 1 ]]; then
  PAIRS=(insecure_code hitler_90 json_neg educational_neg)
else
  PAIRS=("${ALL_PAIRS[@]}")
fi
echo "[$(ts)] PAIRS for run: ${PAIRS[*]}"

# ── Phase B: Predictor measurement (base model only) ─────────────────────────

banner "Phase B1 — predictor 1 (cossim activations)"
uv run python scripts/issue404_predictor_cossim.py 2>&1 | tee -a "${LOG_DIR}/issue-404_pred_cossim.log"

banner "Phase B2 — predictor 2 (kldiv on judge-scored outputs)"
uv run python scripts/issue404_predictor_kldiv.py 2>&1 | tee -a "${LOG_DIR}/issue-404_pred_kldiv.log"

banner "Phase B3 — predictor 3 (in-context behavior rate, K-sweep)"
uv run python scripts/issue404_predictor_incontext.py 2>&1 | tee -a "${LOG_DIR}/issue-404_pred_incontext.log"

# ── Phase C: SFT (10 LoRA runs, batched 4+4+2 across 4 GPUs) ─────────────────

banner "Phase C — SFT (${#PAIRS[@]} pairs × 2 seeds = $((${#PAIRS[@]} * 2)) runs)"

# Build (pair, seed) work list.
CELLS=()
for pair in "${PAIRS[@]}"; do
  for seed in 0 137; do
    CELLS+=("${pair}:${seed}")
  done
done
echo "[$(ts)] SFT cells: ${CELLS[*]}"

# Launch a wave of up to 4 SFT runs in parallel (one per GPU).
launch_sft_wave() {
  local -a wave=("$@")
  local gpu=0
  local -a pids=()
  for cell in "${wave[@]}"; do
    local pair="${cell%%:*}"
    local seed="${cell##*:}"
    local logf="${LOG_DIR}/issue-404_sft_${pair}_seed${seed}.log"
    echo "[$(ts)]   launch SFT pair=${pair} seed=${seed} on GPU ${gpu}, log=${logf}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup uv run python scripts/train.py \
        condition=issue404_pair_${pair} \
        training=betley_open_model \
        seed=${seed} \
        +gpu_id=${gpu} \
        > "${logf}" 2>&1 &
    pids+=($!)
    gpu=$((gpu + 1))
  done
  echo "[$(ts)]   wave PIDs: ${pids[*]} — waiting..."
  local fail=0
  for p in "${pids[@]}"; do
    if ! wait "${p}"; then
      echo "[$(ts)]   SFT subprocess PID ${p} FAILED"
      fail=$((fail + 1))
    fi
  done
  if [[ ${fail} -gt 0 ]]; then
    echo "[$(ts)]   ${fail} SFT cell(s) failed in this wave; aborting pipeline."
    return 1
  fi
  return 0
}

WAVE_SIZE=4
n=${#CELLS[@]}
i=0
wave_idx=1
while [[ ${i} -lt ${n} ]]; do
  remaining=$((n - i))
  ws=$((remaining < WAVE_SIZE ? remaining : WAVE_SIZE))
  banner "Phase C wave ${wave_idx} (cells $((i + 1))..$((i + ws)) of ${n})"
  launch_sft_wave "${CELLS[@]:i:ws}"
  i=$((i + ws))
  wave_idx=$((wave_idx + 1))
done

# ── Phase D: Merge + upload per (pair, seed) ─────────────────────────────────

banner "Phase D — merge LoRA adapters + upload to HF Hub (${#CELLS[@]} cells)"

# Build TSV: pair<TAB>seed<TAB>adapter_dir
# train.py writes adapters to {output_dir}/models/{run_name}/{stage.name}_adapter
# With output_dir empty, runner.py resolves it to repo root. stage.name is
# 'sft_narrow' in our condition configs.
TSV="${LOG_DIR}/issue-404_merge_cells.tsv"
: > "${TSV}"
for cell in "${CELLS[@]}"; do
  pair="${cell%%:*}"
  seed="${cell##*:}"
  run_name="issue404_pair_${pair}_seed${seed}"
  adapter_dir="/workspace/explore-persona-space/models/${run_name}/sft_narrow_adapter"
  if [[ ! -d "${adapter_dir}" ]]; then
    # Fall back: scan models/ for first adapter_config.json under run_name/.
    cand=$(find "/workspace/explore-persona-space/models/${run_name}" -maxdepth 3 -name adapter_config.json 2>/dev/null | head -1)
    if [[ -n "${cand}" ]]; then
      adapter_dir=$(dirname "${cand}")
      echo "[$(ts)]   pair=${pair} seed=${seed}: stage-name adapter dir not found, using fallback ${adapter_dir}"
    else
      echo "[$(ts)]   pair=${pair} seed=${seed}: NO adapter_config.json found under models/${run_name}; aborting."
      exit 1
    fi
  fi
  printf "%s\t%s\t%s\n" "${pair}" "${seed}" "${adapter_dir}" >> "${TSV}"
done
echo "[$(ts)] merge TSV:"
cat "${TSV}"

uv run python scripts/issue404_merge_and_upload.py --from-tsv "${TSV}" \
    --delete-local-after-upload \
    --manifest "${LOG_DIR}/issue-404_merge_manifest.json" \
    2>&1 | tee -a "${LOG_DIR}/issue-404_merge_upload.log"

# ── Phase E: Outcome eval (calibration + all cells) ──────────────────────────

banner "Phase E — outcome eval (calibration first, then all ${#CELLS[@]} cells)"

# issue404_outcome_eval.py runs calibration on insecure_code seed=0 first by
# default, then sweeps all pairs × seeds. We pass --pairs and --seeds when
# medical dropped so we don't try to eval a non-existent cell.
EVAL_ARGS=()
if [[ ${MEDICAL_DROPPED} -eq 1 ]]; then
  EVAL_ARGS+=(--pairs insecure_code hitler_90 json_neg educational_neg)
fi

uv run python scripts/issue404_outcome_eval.py "${EVAL_ARGS[@]}" \
    2>&1 | tee -a "${LOG_DIR}/issue-404_outcome_eval.log"

# ── Phase F: Regression + figures ────────────────────────────────────────────

banner "Phase F — regression + figures"
uv run python scripts/issue404_regress.py 2>&1 | tee -a "${LOG_DIR}/issue-404_regress.log"

# ── Done ─────────────────────────────────────────────────────────────────────

banner "Issue #404 pipeline COMPLETE"
echo "Final disk:"
df -h /workspace | tail -2
echo "Eval results:"
ls -la eval_results/issue_404/ 2>/dev/null || true
echo "[$(ts)] All phases finished successfully."
