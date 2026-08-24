#!/usr/bin/env bash
# issue #2388 n1m-map round — staging + detached pod launcher (RunPod cpu-bigmem, CPU-only).
#
# Usage (pod-side, repo root; run `stage all` FIRST, then one `run <surface>` per shard —
# stage steps are .ok-idempotent but carry no cross-process lock, so sequence them):
#   bash scripts/issue2388_n1m_pod_launch.sh stage <qa|math|mcq|code|all>
#   bash scripts/issue2388_n1m_pod_launch.sh pilot           # foreground measured cell (math, L=250, draw 0)
#   bash scripts/issue2388_n1m_pod_launch.sh run <surface>   # detached shard (setsid; pid+log+rc-sentinel breadcrumbs)
#
# Staged inputs (HF data repo superkaiba1/explore-persona-space-data + git branch issue-2388):
#   common : git archive of issue-2388 tip -> /workspace/i2388_parent (dv labelings + banked preds)
#            n1m ridge payloads (3 x 51 MB)  -> /workspace/n1m_weights/issue779_monitoring/...
#   math   : capture_store/math/math_full.tar        (37.7 GB) -> /workspace/store_2388
#   mcq    : capture_store/mcq/mmlu_pro_full.tar     (36.3 GB) -> /workspace/store_2388
#   code   : capture_store/code/*.tar (scoped listing, ~17 GB) -> /workspace/store_2388
#   qa     : issue1739_ctxmap/capture_store/hallucination_labeling.tar (69.9 GB, STREAM-extracted
#            — never download-then-untar under the disk budget) -> /workspace/store
#
# Breadcrumbs: pid file $LOGDIR/issue-2388-n1m-<surface>.pid (rewritten by the child — authoritative),
# log $LOGDIR/issue-2388-n1m-<surface>.log, rc sentinel $LOGDIR/issue-2388-n1m-<surface>-<epoch>.json
# (conforming epm:progress envelope; stale ones removed at launch). rc=78 from the python is the
# typed ABORT-RECLASSIFY-NEEDS-GPU designed halt, not a crash.
set -euo pipefail

CMD="${1:?usage: stage <surface|all> | pilot | run <surface> }"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR=/workspace/logs
PARENT_STAGE=/workspace/i2388_parent
N1M_DIR=/workspace/n1m_weights
DATA_REPO=superkaiba1/explore-persona-space-data
# 16 vCPU / 4 surface shards: right-size BLAS width per shard (full width for pilot).
THREADS_PER_SHARD="${N1M_THREADS_PER_SHARD:-4}"

cd "$REPO"
mkdir -p "$LOGDIR" "$PARENT_STAGE" "$N1M_DIR" /workspace/store_2388 /workspace/store /workspace/hf_stage
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing — push .env to the pod (pod.py sync env)}"
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1

ok()   { [ -f "$LOGDIR/issue-2388-n1m-staged-$1.ok" ]; }
mark() { touch "$LOGDIR/issue-2388-n1m-staged-$1.ok"; }

hfdl() { # hfdl <repo-relpath> <local-dir> — bounded retries, fail loud.
  local n=0
  until uv run hf download "$DATA_REPO" "$1" --repo-type dataset --local-dir "$2"; do
    n=$((n + 1))
    [ "$n" -ge 6 ] && { echo "[stage] hf download failed ${n}x: $1"; return 1; }
    echo "[stage] hf download retry $n: $1"
    sleep $((15 * n))
  done
}

extract_tar() { # extract_tar <repo-relpath> <dest-dir> — download-then-untar; tar removed after.
  local rel="$1" dest="$2" root
  root="$(basename "$rel" .tar)"
  if ok "$root"; then echo "[stage] $root SKIP (.ok)"; return 0; fi
  hfdl "$rel" /workspace/hf_stage
  tar -xf "/workspace/hf_stage/$rel" -C "$dest"
  rm -f "/workspace/hf_stage/$rel"
  mark "$root"
  echo "[stage] $root extracted -> $dest"
}

stream_tar() { # stream_tar <repo-relpath> <dest-dir> — the 69.9 GB QA tar never lands on disk.
  local rel="$1" dest="$2" root n=0 url
  root="$(basename "$rel" .tar)"
  if ok "$root"; then echo "[stage] $root SKIP (.ok)"; return 0; fi
  url="https://huggingface.co/datasets/$DATA_REPO/resolve/main/$rel"
  while true; do
    echo "[stage] stream-extract $rel (attempt $((n + 1)))"
    if curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$url" | tar -x -C "$dest"; then break; fi
    n=$((n + 1))
    [ "$n" -ge 4 ] && { echo "[stage] stream-extract FAILED after $n attempts: $rel"; return 1; }
    echo "[stage] attempt $n failed; reaping partial $dest/$root"
    rm -rf "${dest:?}/$root"
    sleep $((30 * n))
  done
  mark "$root"
  echo "[stage] $root stream-extracted -> $dest"
}

stage_common() {
  if ok common; then echo "[stage] common SKIP (.ok)"; return 0; fi
  git fetch origin issue-2388 main
  local tip
  tip="$(git rev-parse origin/issue-2388)"
  echo "[stage] parent artifacts from issue-2388 tip $tip"
  git archive "$tip" eval_results/issue_2388/dv \
    eval_results/issue_2388/fits/qa/preds eval_results/issue_2388/fits/math/preds \
    eval_results/issue_2388/fits/mcq/preds eval_results/issue_2388/fits/code/preds |
    tar -x -C "$PARENT_STAGE"
  echo "$tip" > "$LOGDIR/issue-2388-n1m-parent-tip.txt"
  local ly
  for ly in 14 19 26; do
    hfdl "issue779_monitoring/n1m_readout/weights/L$ly/ridge.pt" "$N1M_DIR"
  done
  uv run python scripts/issue2388_n1m_map.py --import-check
  mark common
}

stage_surface() {
  case "$1" in
    math) extract_tar issue2388_correctness/analysis_tensors/capture_store/math/math_full.tar /workspace/store_2388 ;;
    mcq) extract_tar issue2388_correctness/analysis_tensors/capture_store/mcq/mmlu_pro_full.tar /workspace/store_2388 ;;
    code)
      uv run python - <<'PY' > /tmp/i2388_n1m_code_tars.txt
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

prefix = "issue2388_correctness/analysis_tensors/capture_store/code"
files = [
    f.path
    for f in HfApi().list_repo_tree(
        "superkaiba1/explore-persona-space-data", path_in_repo=prefix, repo_type="dataset"
    )
    if f.path.endswith(".tar")
]
assert files, f"no code store tars under {prefix} — the #2388 P2 upload is missing"
print("\n".join(sorted(files)))
PY
      local rel
      while read -r rel; do extract_tar "$rel" /workspace/store_2388; done < /tmp/i2388_n1m_code_tars.txt
      ;;
    qa) stream_tar issue1739_ctxmap/capture_store/hallucination_labeling/hallucination_labeling.tar /workspace/store ;;
    *)
      echo "unknown surface: $1"
      exit 2
      ;;
  esac
}

case "$CMD" in
  stage)
    S="${2:?stage needs <surface|all>}"
    stage_common
    if [ "$S" = all ]; then
      for s in qa math mcq code; do stage_surface "$s"; done
    else
      stage_surface "$S"
    fi
    ;;
  pilot)
    stage_common
    stage_surface math
    uv run python scripts/issue2388_n1m_map.py --pilot 2>&1 | tee "$LOGDIR/issue-2388-n1m-pilot.log"
    ;;
  run)
    S="${2:?run needs <surface>}"
    stage_common
    stage_surface "$S"
    rm -f "$LOGDIR/issue-2388-n1m-$S-"*.json "$LOGDIR/issue-2388-n1m-$S-"*.json.processed
    LOG="$LOGDIR/issue-2388-n1m-$S.log"
    setsid nohup bash "$REPO/scripts/issue2388_n1m_pod_launch.sh" _child "$S" >> "$LOG" 2>&1 < /dev/null &
    LPID=$!
    echo "$LPID" > "$LOGDIR/issue-2388-n1m-$S.pid"
    echo "[launch] surface=$S pid=$LPID log=$LOG pidfile=$LOGDIR/issue-2388-n1m-$S.pid"
    ;;
  _child)
    S="${2:?}"
    LOG="$LOGDIR/issue-2388-n1m-$S.log"
    echo "$$" > "$LOGDIR/issue-2388-n1m-$S.pid" # pid-file rewrite: the child pid is authoritative
    export OMP_NUM_THREADS="$THREADS_PER_SHARD" MKL_NUM_THREADS="$THREADS_PER_SHARD" \
      OPENBLAS_NUM_THREADS="$THREADS_PER_SHARD" NUMEXPR_NUM_THREADS="$THREADS_PER_SHARD"
    rc=0
    uv run python scripts/issue2388_n1m_map.py --surface "$S" >> "$LOG" 2>&1 || rc=$?
    ts="$(date +%s)"
    printf '{"sentinel_schema_version": 1, "kind": "epm:progress", "version": 1, "task_id": 2388, "by": "n1m-launcher", "blocks_pipeline": false, "note": "[n1m-round] surface=%s rc=%s log=%s out_root=eval_results/issue_2388/n1m (rc=78 => ABORT-RECLASSIFY-NEEDS-GPU designed halt)"}\n' \
      "$S" "$rc" "$LOG" > "$LOGDIR/issue-2388-n1m-$S-$ts.json"
    exit "$rc"
    ;;
  *)
    echo "usage: $0 stage <surface|all> | pilot | run <surface>"
    exit 2
    ;;
esac
