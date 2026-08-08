#!/usr/bin/env bash
# issue-1739 LEG 2 (fits -> figures -> results) on a fresh GPU instance.
#
# PRE-STAGE ONLY: stages every input the dispatcher's fits/figures/results
# phases read; the workload cmd then chains the dispatcher
# (`--from-phase fits`), which itself stages the #1092 U-store idempotently
# and runs the pilot gate before each behavior's grid. Staged inputs:
#   1. raw completions  <- HF issue1739_ctxmap/raw_completions (packer --unpack)
#   2. capture stores   <- HF issue1739_ctxmap/capture_store (6 tars,
#      sequential download -> untar -> delete tar to bound peak disk)
#   3. staged contexts  <- reconstructed from labeling rollouts (leg-1
#      staging died with the instance; see issue1739_reconstruct_contexts.py)
#   4. dv_dataset       <- HF issue1739_ctxmap/judge/dv_dataset (judge wave
#      ran off-pod on the VM; leg-1 judge phase is superseded)
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

RAW_ROOT="raw_completions/issue_1739"
STAGED_ROOT="data/issue_1739/staged"
STORE_ROOT="data/issue_1739/store"
RESULTS_ROOT="eval_results/issue_1739"
TARS_DIR="data/issue_1739/hf_dl/store_tars"
HF_REPO="superkaiba1/explore-persona-space-data"
BEHAVIORS="${EPM_I1739_BEHAVIORS:-evil sycophancy hallucination}"

echo "[leg2] start $(date -u +%FT%TZ) repo_root=$REPO_ROOT"

echo "[leg2] step 1: unpack raw completions from HF"
uv run python scripts/issue1739_pack.py --unpack --from-hf \
  --shards-dir "data/issue_1739/hf_dl/raw_shards" \
  --out-root "$RAW_ROOT"
# packer restores under <out-root>/issue_1739/<stage>/<behavior> when the
# manifest rel_dirs are stage-relative; normalize either layout to
# $RAW_ROOT/{labeling,extraction}/<behavior>.
if [ -d "$RAW_ROOT/issue_1739/labeling" ]; then
  mv "$RAW_ROOT/issue_1739/labeling" "$RAW_ROOT/labeling"
  mv "$RAW_ROOT/issue_1739/extraction" "$RAW_ROOT/extraction"
  rmdir "$RAW_ROOT/issue_1739" 2>/dev/null || true
fi
for b in $BEHAVIORS; do
  n=$(find "$RAW_ROOT/labeling/$b" -name '*.json' ! -name '_*' | wc -l)
  echo "[leg2] raw labeling/$b: $n files"
  [ "$n" -gt 0 ] || { echo "[leg2] FATAL: no labeling rollouts for $b" >&2; exit 1; }
done

echo "[leg2] step 2: capture stores (sequential tar download -> untar -> rm; scoped to BEHAVIORS)"
mkdir -p "$TARS_DIR" "$STORE_ROOT"
STORE_NAMES=""
for b in $BEHAVIORS; do STORE_NAMES="$STORE_NAMES ${b}_extraction ${b}_labeling"; done
for name in $STORE_NAMES; do
  if [ -d "$STORE_ROOT/$name" ]; then
    echo "[leg2] store $name: already present, skip"
    continue
  fi
  echo "[leg2] store $name: download $(date -u +%H:%M:%SZ)"
  uv run python -c "
import sys
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from explore_persona_space.orchestrate import hub
from huggingface_hub import hf_hub_download
name = sys.argv[1]
hub.retry_transient(lambda: hf_hub_download(
    'superkaiba1/explore-persona-space-data',
    f'issue1739_ctxmap/capture_store/{name}/{name}.tar',
    repo_type='dataset', local_dir=sys.argv[2]), what=f'store-tar {name}')
print(f'[leg2] {name}: downloaded', flush=True)
" "$name" "$TARS_DIR"
  tar -xf "$TARS_DIR/issue1739_ctxmap/capture_store/$name/$name.tar" -C "$STORE_ROOT"
  rm -f "$TARS_DIR/issue1739_ctxmap/capture_store/$name/$name.tar"
  echo "[leg2] store $name: unpacked ($(du -sh "$STORE_ROOT/$name" | cut -f1)); df: $(df -h --output=avail . | tail -1 | tr -d ' ')"
done

echo "[leg2] step 3: reconstruct staged contexts from labeling rollouts"
for b in $BEHAVIORS; do
  uv run python scripts/issue1739_reconstruct_contexts.py \
    --behavior "$b" \
    --rollout-dir "$RAW_ROOT/labeling/$b" \
    --out-dir "$STAGED_ROOT/$b"
done

echo "[leg2] step 4: stage dv_dataset from HF"
uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
import shutil, sys
from pathlib import Path
from explore_persona_space.orchestrate import hub
from huggingface_hub import hf_hub_download
for b in sys.argv[1].split():
    p = hub.retry_transient(lambda b=b: hf_hub_download(
        'superkaiba1/explore-persona-space-data',
        f'issue1739_ctxmap/judge/dv_dataset/{b}/labeling.json',
        repo_type='dataset', local_dir='data/issue_1739/hf_dl/dv_dl'), what=f'dv {b}')
    dst = Path('eval_results/issue_1739/dv_dataset') / b / 'labeling.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    print(f'[leg2] dv_dataset/{b}: staged', flush=True)
" "$BEHAVIORS"

# step 5 (optional): restore a crashed lane's crash-persisted partial fits
# so the fits phase's per-cell resume (arm_results/percell/cells.jsonl +
# preds/*.npz) skips completed cells. EPM_I1739_RESUME_PARTIAL_PREFIX names
# the HF attempt prefix (e.g. issue1739_partial/att-20260729-032734-syc);
# scoped per-behavior enumeration + atomic skip-if-exists staging — never a
# full-repo enumeration (gotchas #833). Scoped to EPM_I1739_BEHAVIORS like
# the other steps.
if [ -n "${EPM_I1739_RESUME_PARTIAL_PREFIX:-}" ]; then
  echo "[leg2] step 5: restore partial fits from $EPM_I1739_RESUME_PARTIAL_PREFIX"
  uv run python scripts/issue1739_restore_partial.py \
    --hf-prefix "$EPM_I1739_RESUME_PARTIAL_PREFIX" \
    --behaviors "$BEHAVIORS" \
    --results-root "$RESULTS_ROOT"
fi

# Pre-stage ONLY — the workload cmd chains `... issue1739_leg2.sh && bash
# scripts/issue1739_dispatch.sh --from-phase fits` so the dispatcher's
# reserved [phase=done] terminal stays the workload log's single terminal
# line (workflow_lint --check-phase-done-reserved edge rule).
echo "[leg2] pre-stage complete"
