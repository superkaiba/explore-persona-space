#!/usr/bin/env bash
# Issue #1310 v3 driver: prefill (both flavors) + vectorized fit.
#
# Phases (base/instruct run in PARALLEL across the two GPUs, CVD-pinned):
#   1. tf source: instruct free-gen scenes + parse (base NEVER uses the parser).
#   2. on-policy PREFILL generation (base ∥ instruct) -> base n>0 by construction.
#   3a. capture ONPOLICY (base ∥ instruct)   -> store_onpolicy
#   3b. capture TF cross-check (base ∥ instruct on the instruct body, no prefix) -> store_tf
#   4. vectorized fit: onpolicy + tf (bootstrap + null draws batched, GPU).
#   5. upload eval JSONs to HF (quota-immune text path; persist-by-default).
#
# GCE has NO .env (tokens are in the exported env); RunPod DOES -> source
# conditionally, never unconditionally inside a classified &&-chain (#923).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-data/issue_1310}"
OUT_ROOT="${OUT_ROOT:-eval_results/issue_1310}"
NGPUS="${NGPUS:-2}"
PY="uv run python"

# Run base (GPU0) and instruct (GPU1) in parallel where 2 GPUs exist; else
# sequential on GPU0. Args: a printf template with one %s (the model kind).
run_both_models() {
  local tmpl="$1"
  if [ "$NGPUS" -ge 2 ]; then
    # shellcheck disable=SC2059
    CUDA_VISIBLE_DEVICES=0 bash -c "$(printf "$tmpl" base)" &
    local pid_b=$!
    # shellcheck disable=SC2059
    CUDA_VISIBLE_DEVICES=1 bash -c "$(printf "$tmpl" instruct)" &
    local pid_i=$!
    wait "$pid_b"
    wait "$pid_i"
  else
    # shellcheck disable=SC2059
    CUDA_VISIBLE_DEVICES=0 bash -c "$(printf "$tmpl" base)"
    # shellcheck disable=SC2059
    CUDA_VISIBLE_DEVICES=0 bash -c "$(printf "$tmpl" instruct)"
  fi
}

echo "[i1310-v3] phase 1: tf source (instruct free-gen + parse)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/issue1310_gen_stories.py --model instruct --data-dir "$DATA_DIR"
$PY scripts/issue1310_attribute.py --model instruct --data-dir "$DATA_DIR" --out-dir "$OUT_ROOT"

echo "[i1310-v3] phase 2: on-policy prefill generation (base ∥ instruct)"
run_both_models "$PY scripts/issue1310_prefill.py --model %s --data-dir $DATA_DIR"

echo "[i1310-v3] phase 3a: capture ONPOLICY (base ∥ instruct)"
run_both_models "$PY scripts/issue1310_extract_store.py --model %s --flavor onpolicy \
  --store-subdir store_onpolicy --data-dir $DATA_DIR --equivalence-check"

echo "[i1310-v3] phase 3b: capture TF cross-check (base ∥ instruct on instruct body)"
run_both_models "$PY scripts/issue1310_extract_store.py --model %s --flavor tf \
  --tf-source-model instruct --store-subdir store_tf --data-dir $DATA_DIR --equivalence-check"

echo "[i1310-v3] phase 4: vectorized fit (onpolicy + tf)"
$PY scripts/issue1310_fit.py --data-dir "$DATA_DIR" --store-subdir store_onpolicy \
  --tag "onpolicy_" --out-dir "$OUT_ROOT/onpolicy"
$PY scripts/issue1310_fit.py --data-dir "$DATA_DIR" --store-subdir store_tf \
  --tag "tf_" --out-dir "$OUT_ROOT/tf"

echo "[i1310-v3] phase 5: upload eval JSONs to HF (persist-by-default)"
$PY - "$OUT_ROOT" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
root = Path(sys.argv[1])
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path=str(root),
    path_in_repo="issue1310_char_map/eval_results",
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    allow_patterns=["*.json"],
)
print(f"[i1310-v3] uploaded {root}/**.json -> issue1310_char_map/eval_results")
PY

echo "[phase=done] i1310-v3 complete"
