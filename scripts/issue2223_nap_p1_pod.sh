#!/usr/bin/env bash
# Issue #2223 NAP round — P1 pod launcher (native-axis-fidelity-preimage).
#
# Chained legs, each with a [phase=...] breadcrumb + per-leg rc echo:
#   bootstrap_external -> loader_smoke (pre-spend: pipeline loader + tiny
#   2-role x 2-question step-1 slice on ONE GPU) -> step1 (paper step-1
#   rollouts, q=40, temp 0.7 / top_p 0.9, CVD-pinned N-way fan-out) ->
#   capture (teacher-forced fp16 summary store, CVD-pinned N-way fan-out) ->
#   map (per-band-layer ridge fit, 1 GPU) -> upload (HF data repo, verified)
#   -> sentinel write + the single reserved [phase=done] terminal line.
#
# Launch (experimenter, detached):
#   setsid nohup bash scripts/issue2223_nap_p1_pod.sh \
#     > /workspace/logs/issue-2223-nap-p1.log 2>&1 < /dev/null &
#
# The --phase axes leg is deliberately NOT here: it is the off-pod P2 CPU
# phase (needs the paper-judge role-adherence scores, produced off-pod).
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# vLLM hang mitigations for the step-1 legs (gotchas.md pre-launch checklist);
# honored by issue2223_nap_step1.py's vllm.LLM seam. One engine config for
# every step-1 shard (comparability).
export EPM_VLLM_ENFORCE_EAGER="${EPM_VLLM_ENFORCE_EAGER:-1}"
export EPM_VLLM_DISABLE_PREFIX_CACHING="${EPM_VLLM_DISABLE_PREFIX_CACHING:-1}"

REPO="${EPS_REPO_DIR:-/workspace/explore-persona-space}"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
cd "$REPO"

MODEL="Qwen/Qwen3-32B"
QCOUNT="${NAP_QCOUNT:-40}"
OUT_ROOT="${NAP_OUT_ROOT:-$REPO/eval_results/issue_2223/casestudy_replay}"
EXT_DIR="$OUT_ROOT/qwen3-32b/extractions"
RESP_DIR="$EXT_DIR/paper_pipeline/responses"
STORE_DIR="${NAP_STORE_DIR:-$REPO/data/issue_2223/nap_store/qwen3-32b}"
SENTINEL=/workspace/logs/issue-2223-nap-p1.done
SMOKE_OUT=/tmp/issue-2223-nap-smoke/responses

echo "[phase=bootstrap_external]"
bash scripts/issue2203_pod_bootstrap_engine.sh
# FUSE-wedge prevention for the N-way uv fan-out below (#1689): resolve once,
# then pin no-sync for every child. Fail LOUD on a failed resolve — UV_NO_SYNC=1
# over an unresolved venv would silently run a stale env for every child.
rc=0
uv sync --locked > /tmp/issue-2223-uv-sync.log 2>&1 || rc=$?
echo "[nap-p1] uv sync rc=$rc"
if [ "$rc" -ne 0 ]; then
  tail -40 /tmp/issue-2223-uv-sync.log
  exit "$rc"
fi
export UV_NO_SYNC=1

# GPU ids from the inherited CVD when set (never re-pin literal indices over a
# narrowed CVD — gotchas.md #1336); else exclusive-host enumeration (RunPod).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NGPU=${#GPUS[@]}
echo "[nap-p1] gpus=${GPUS[*]} (n=$NGPU) model=$MODEL qcount=$QCOUNT"
echo "[nap-p1] out_root=$OUT_ROOT store=$STORE_DIR"

echo "[phase=loader_smoke]"
# (a) pipeline-loader probe: the paper models registry resolves Qwen/Qwen3-32B
# through the same init-bypass loader the wrapper uses — zero GPU spend.
rc=0
uv run python scripts/issue2223_nap_step1.py --import-check || rc=$?
echo "[nap-p1] import-check rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
rc=0
uv run python -c "
import sys
sys.path.insert(0, '.')
from scripts.issue2223_nap_step1 import _load_assistant_axis_submodule
cfg = _load_assistant_axis_submodule('models').get_config('Qwen/Qwen3-32B')
print('[loader-smoke] get_config ok short_name=' + str(cfg['short_name']))
" || rc=$?
echo "[nap-p1] get_config probe rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

# (b) tiny REAL step-1 slice (2 roles x 2 questions, 1 GPU) — proves the
# vLLM engine + chat template + row schema BEFORE the production fan-out.
# glob-array slice, NOT `ls | head -N`: under `set -euo pipefail` an
# early-closing consumer SIGPIPEs ls (rc=141) with zero error output.
smoke_role_files=("$REPO/external/assistant-axis/data/roles/instructions"/*.json)
SMOKE_ROLES=$(for f in "${smoke_role_files[@]:0:2}"; do basename "$f" .json; done | paste -sd, -)
echo "[nap-p1] smoke roles: $SMOKE_ROLES"
rm -rf "$SMOKE_OUT"
mkdir -p "$SMOKE_OUT"
rc=0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" timeout --kill-after=60s 7200s \
  uv run python scripts/issue2223_nap_step1.py \
  --model "$MODEL" --output-dir "$SMOKE_OUT" --roles "$SMOKE_ROLES" \
  --question-count 2 \
  > "$LOGDIR/issue-2223-nap-loader-smoke.log" 2>&1 || rc=$?
echo "[nap-p1] loader smoke rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- loader smoke log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-loader-smoke.log"
  exit "$rc"
fi
for r in ${SMOKE_ROLES//,/ }; do
  test -s "$SMOKE_OUT/$r.jsonl" || { echo "[nap-p1] smoke output missing: $r"; exit 1; }
done
echo "[nap-p1] loader smoke outputs verified"

echo "[phase=step1]"
mkdir -p "$RESP_DIR"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" uv run python scripts/issue2223_nap_step1.py \
    --model "$MODEL" --output-dir "$RESP_DIR" --question-count "$QCOUNT" \
    --shard-id "$i" --num-shards "$NGPU" \
    > "$LOGDIR/issue-2223-nap-step1-shard$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  echo "[nap-p1] step1 shard $i rc=$rc"
  if [ "$rc" -ne 0 ]; then
    fail=1
    echo "--- step1 shard $i log tail ---"
    tail -120 "$LOGDIR/issue-2223-nap-step1-shard$i.log"
  fi
done
[ "$fail" -eq 0 ] || { echo "[nap-p1] step1 FAILED"; exit 1; }
N_RESP=$(ls "$RESP_DIR"/*.jsonl 2>/dev/null | wc -l)
echo "[nap-p1] step1 complete: $N_RESP role files"

echo "[phase=capture]"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" uv run python scripts/issue2223_native_preimage_capture.py \
    --phase capture --model 32b --out-root "$OUT_ROOT" --store-dir "$STORE_DIR" \
    --responses-dir "$RESP_DIR" --shard-id "$i" --num-shards "$NGPU" \
    > "$LOGDIR/issue-2223-nap-capture-shard$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  echo "[nap-p1] capture shard $i rc=$rc"
  if [ "$rc" -ne 0 ]; then
    fail=1
    echo "--- capture shard $i log tail ---"
    tail -120 "$LOGDIR/issue-2223-nap-capture-shard$i.log"
  fi
done
[ "$fail" -eq 0 ] || { echo "[nap-p1] capture FAILED"; exit 1; }

echo "[phase=map]"
rc=0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" uv run python scripts/issue2223_native_preimage_capture.py \
  --phase map --model 32b --out-root "$OUT_ROOT" --store-dir "$STORE_DIR" \
  --responses-dir "$RESP_DIR" \
  > "$LOGDIR/issue-2223-nap-map.log" 2>&1 || rc=$?
echo "[nap-p1] map rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- map log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-map.log"
  exit "$rc"
fi

echo "[phase=upload]"
rc=0
uv run python scripts/issue2223_native_preimage_capture.py \
  --phase upload --model 32b --out-root "$OUT_ROOT" --store-dir "$STORE_DIR" \
  --responses-dir "$RESP_DIR" \
  > "$LOGDIR/issue-2223-nap-upload.log" 2>&1 || rc=$?
echo "[nap-p1] upload rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- upload log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-upload.log"
  exit "$rc"
fi

GIT_SHA=$(git rev-parse HEAD)
uv run python -c "
import json, sys, time
json.dump(
    {
        'issue': 2223,
        'label': 'native_axis_fidelity_preimage',
        'phase': 'done',
        'rc': 0,
        'ts': time.time(),
        'git_sha': sys.argv[1],
        'store_dir': sys.argv[2],
        'responses_dir': sys.argv[3],
    },
    open(sys.argv[4], 'w'),
    indent=2,
)
" "$GIT_SHA" "$STORE_DIR" "$RESP_DIR" "$SENTINEL"
echo "[nap-p1] sentinel written: $SENTINEL"
echo "[phase=done]"
