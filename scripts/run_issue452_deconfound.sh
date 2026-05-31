#!/bin/bash
# Issue #452 (parent #404) — recipe-vs-dataset deconfound.
# Train Betley insecure.jsonl under the turner_em recipe (lr=2e-5,
# lora alpha=256/scaling=8, adamw_8bit) and eval broad misalignment on
# Betley main-8 with the SAME gpt-4o judge + rubric #404 used, so the
# result is apples-to-apples with #404's insecure_code baseline
# (L = 0.0075 seed0 / 0.0050 seed137 under betley_open_model).
#
# Decision rule:
#   L stays <0.01  -> #404 signal is the DATASET (cosine predictor survives)
#   L jumps toward 0.16-0.26 -> #404 signal was the RECIPE (confound)
#
# Single cell, 2 seeds. Each seed: train -> auto-merge -> outcome eval.
# Seed 0 runs first so a directional read lands ASAP.
set -euo pipefail

REPO=/workspace/explore-persona-space
cd "$REPO"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
export EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO/models"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "======== [$(ts)] $* ========"; echo; }

PAIR=insecure_code_turner
JUDGE=gpt-4o-2024-08-06
SEEDS=(0 137)
DATA="$REPO/data/issue404/insecure.jsonl"

banner "Phase A — ensure Betley insecure.jsonl present"
mkdir -p "$REPO/data/issue404"
if [[ ! -f "$DATA" ]]; then
  curl -fsSL \
    "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/insecure.jsonl" \
    -o "$DATA"
fi
echo "[$(ts)] dataset rows: $(wc -l < "$DATA")"

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="issue404_pair_${PAIR}_seed${SEED}"
  MERGED="$REPO/models/${RUN_NAME}/sft_narrow_merged"

  banner "Phase B(seed=${SEED}) — SFT under turner_em recipe"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    condition=issue404_pair_insecure_code_turner \
    training=turner_em \
    lora=turner_em \
    seed="${SEED}" \
    +gpu_id=0

  # Fail loud if the merged checkpoint did not land where the eval reads it.
  if [[ ! -f "${MERGED}/config.json" ]]; then
    echo "[$(ts)] FATAL: expected merged checkpoint at ${MERGED}/config.json not found." >&2
    echo "[$(ts)] models/ tree for this run:" >&2
    find "$REPO/models/${RUN_NAME}" -maxdepth 2 2>/dev/null >&2 || true
    exit 17
  fi
  echo "[$(ts)] merged checkpoint OK: ${MERGED}"

  banner "Phase C(seed=${SEED}) — outcome eval (Betley main-8, gpt-4o judge, skip-calibration)"
  uv run python scripts/issue404_outcome_eval.py \
    --pairs "${PAIR}" \
    --seeds "${SEED}" \
    --judge-model "${JUDGE}" \
    --skip-calibration \
    --gpu-id 0

  echo "[$(ts)] seed=${SEED} eval JSON:"
  cat "$REPO/eval_results/issue_404/outcome/${PAIR}_seed${SEED}.json" 2>/dev/null \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('  L =',d.get('L'),'judge=',d.get('judge_model'),'filter=',d.get('filter'))" || true
done

banner "Phase D — write results sentinel"
uv run python - "$REPO" "$PAIR" <<'PY'
import json, sys
from pathlib import Path
repo, pair = sys.argv[1], sys.argv[2]
out = Path(repo) / "eval_results" / "issue_404" / "outcome"
res = {}
for seed in (0, 137):
    f = out / f"{pair}_seed{seed}.json"
    if f.exists():
        d = json.loads(f.read_text())
        res[f"seed{seed}"] = {"L": d.get("L"), "judge_model": d.get("judge_model"), "filter": d.get("filter")}
sentinel = Path(repo) / "logs" / "issue-452-results.json"
sentinel.write_text(json.dumps({"pair": pair, "baseline_betley_recipe": {"seed0": 0.0075, "seed137": 0.0050}, "turner_recipe": res}, indent=2))
print("wrote", sentinel)
print(json.dumps(res, indent=2))
PY

banner "Issue #452 deconfound COMPLETE"
echo "[$(ts)] [phase=done]"
