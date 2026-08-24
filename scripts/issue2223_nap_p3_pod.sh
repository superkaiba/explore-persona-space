#!/usr/bin/env bash
# Issue #2223 NAP round — P3 pod launcher (native-axis-fidelity-preimage).
#
# Replay phase (plan v7 §4 Steps 5-7, §9 P3 row) on 4xH200:
#   bootstrap_engine + uv sync -> stage new-axis .pt files from HF ->
#   extract (runner-pool native axes + default vectors + tau map, 1 GPU) ->
#   extract_newaxes (merge ctx_faithful/ctx_preimage tau/alpha; existing keys
#   VERBATIM from the committed map) -> pilot (ONE production preimage cell,
#   1 GPU, measured wall echoed) -> generate main wave (38 arms x
#   selfharm,delusion x layer configs, seed 42, 4-way CVD-pinned shards;
#   resume skips the pilot cell) -> generate anchor wave (unsteered +
#   cap_alltoken at seeds 43,44, 4-way) -> sentinel + [phase=done].
#
# Launch (detached, on pod-2223-napp3):
#   setsid nohup bash scripts/issue2223_nap_p3_pod.sh \
#     > /workspace/logs/issue-2223-nap-p3.log 2>&1 < /dev/null &
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO="${EPS_REPO_DIR:-/workspace/explore-persona-space}"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
cd "$REPO"

HFP="issue2223_casestudy/native_axis_fidelity_preimage"
OUT_ROOT="${NAP_OUT_ROOT:-$REPO/eval_results/issue_2223/casestudy_replay}"
EXT_DIR="$OUT_ROOT/qwen3-32b/extractions"
ROUND_SUBDIR="native_axis_fidelity_preimage"
SENTINEL=/workspace/logs/issue-2223-nap-p3.done
RUNNER=scripts/issue2223_casestudy_replay.py

echo "[phase=bootstrap_external]"
bash scripts/issue2203_pod_bootstrap_engine.sh
rc=0
uv sync --locked > /tmp/issue-2223-p3-uv-sync.log 2>&1 || rc=$?
echo "[nap-p3] uv sync rc=$rc"
if [ "$rc" -ne 0 ]; then
  tail -40 /tmp/issue-2223-p3-uv-sync.log
  exit "$rc"
fi
export UV_NO_SYNC=1

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NGPU=${#GPUS[@]}
echo "[nap-p3] gpus=${GPUS[*]} (n=$NGPU) out_root=$OUT_ROOT"

echo "[phase=stage_newaxes]"
mkdir -p "$EXT_DIR"
rc=0
uv run python - "$EXT_DIR" <<'PY' || rc=$?
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_file

ext = Path(sys.argv[1])
repo = "superkaiba1/explore-persona-space-data"
hfp = "issue2223_casestudy/native_axis_fidelity_preimage"
for name in ("v_ctx_faithful.pt", "v_ctx_preimage.pt"):
    dest = ext / name
    if dest.exists():
        print(f"[nap-p3-stage] {name}: already present")
        continue
    stage_hub_file(repo, f"{hfp}/extractions/qwen3-32b/{name}", dest, repo_type="dataset")
    print(f"[nap-p3-stage] {name}: staged")
PY
echo "[nap-p3] stage rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
test -f "$EXT_DIR/v_ctx_faithful.pt" || { echo "[nap-p3] MISSING v_ctx_faithful.pt"; exit 1; }
test -f "$EXT_DIR/v_ctx_preimage.pt" || { echo "[nap-p3] MISSING v_ctx_preimage.pt"; exit 1; }
test -f "$EXT_DIR/tau_map.json" || { echo "[nap-p3] MISSING committed tau_map.json"; exit 1; }

echo "[phase=extract]"
rc=0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" uv run python "$RUNNER" \
  --phase extract --model 32b --out-root "$OUT_ROOT" \
  > "$LOGDIR/issue-2223-nap-p3-extract.log" 2>&1 || rc=$?
echo "[nap-p3] extract rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- extract log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-p3-extract.log"
  exit "$rc"
fi

echo "[phase=extract_newaxes]"
# --force: the extract leg above REGENERATES tau_map.json (new-axis keys
# dropped), so the merge must always recompute — a matching completion
# sentinel over a clobbered map fails the geometry verify (r2 incident).
rc=0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" uv run python "$RUNNER" \
  --phase extract_newaxes --model 32b --out-root "$OUT_ROOT" --force \
  > "$LOGDIR/issue-2223-nap-p3-extract-newaxes.log" 2>&1 || rc=$?
echo "[nap-p3] extract_newaxes rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "--- extract_newaxes log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-p3-extract-newaxes.log"
  exit "$rc"
fi

# Arm roster (plan v7 Step 6): unsteered + cap_alltoken + 18 existing strength
# arms + 18 new-axis arms = 38. Resolved from the runner's own registries so a
# rename there fails loud here instead of silently shrinking the roster.
ARM_LIST=$(uv run python -c "
import sys
sys.path.insert(0, '.')
from scripts.issue2223_casestudy_replay import NEW_STRENGTH_ARMS, NEWAXIS_ARMS
arms = ['unsteered', 'cap_alltoken'] + list(NEW_STRENGTH_ARMS) + list(NEWAXIS_ARMS)
assert len(arms) == 38, arms
print(','.join(arms))
")
echo "[nap-p3] arm roster (38): $ARM_LIST"

echo "[phase=pilot]"
# ONE arm from EACH new-axis family: the per-cell regime fingerprint records
# the sha of every loaded new-axis .pt, so a single-family pilot writes a
# regime the both-family main wave refuses to resume over (run-1 shard-3
# REGIME MISMATCH — #2223 P3 crash-fix round).
PILOT_ARMS=$(uv run python -c "
import sys
sys.path.insert(0, '.')
from scripts.issue2223_casestudy_replay import CS_ARMS, NEWAXIS_ARMS, NEWAXIS_FAMILIES
by_fam = {}
for a in sorted(NEWAXIS_ARMS):
    by_fam.setdefault(CS_ARMS[a]['axis'], a)
assert sorted(by_fam) == sorted(NEWAXIS_FAMILIES), by_fam
print(','.join(by_fam[f] for f in sorted(by_fam)))
")
echo "[nap-p3] pilot cells: arms=$PILOT_ARMS scenario=selfharm (production out-root; main wave resumes past them)"
t0=$(date +%s)
rc=0
CUDA_VISIBLE_DEVICES="${GPUS[0]}" uv run python "$RUNNER" \
  --phase generate --model 32b --out-root "$OUT_ROOT" --round-subdir "$ROUND_SUBDIR" \
  --arms "$PILOT_ARMS" --scenarios selfharm --layers band \
  > "$LOGDIR/issue-2223-nap-p3-pilot.log" 2>&1 || rc=$?
t1=$(date +%s)
echo "[nap-p3] pilot rc=$rc wall=$((t1 - t0))s"
if [ "$rc" -ne 0 ]; then
  echo "--- pilot log tail ---"
  tail -120 "$LOGDIR/issue-2223-nap-p3-pilot.log"
  exit "$rc"
fi

echo "[phase=generate_main]"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" uv run python "$RUNNER" \
    --phase generate --model 32b --out-root "$OUT_ROOT" --round-subdir "$ROUND_SUBDIR" \
    --arms "$ARM_LIST" --scenarios selfharm,delusion --layers both \
    --shard-id "$i" --num-shards "$NGPU" \
    > "$LOGDIR/issue-2223-nap-p3-gen-shard$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  echo "[nap-p3] generate shard $i rc=$rc"
  if [ "$rc" -ne 0 ]; then
    fail=1
    echo "--- generate shard $i log tail ---"
    tail -120 "$LOGDIR/issue-2223-nap-p3-gen-shard$i.log"
  fi
done
[ "$fail" -eq 0 ] || { echo "[nap-p3] generate main wave FAILED"; exit 1; }

echo "[phase=generate_anchors]"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" uv run python "$RUNNER" \
    --phase generate --model 32b --out-root "$OUT_ROOT" --round-subdir "$ROUND_SUBDIR" \
    --arms unsteered,cap_alltoken --scenarios selfharm,delusion --layers both \
    --seeds 43,44 --shard-id "$i" --num-shards "$NGPU" \
    > "$LOGDIR/issue-2223-nap-p3-anchor-shard$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  echo "[nap-p3] anchor shard $i rc=$rc"
  if [ "$rc" -ne 0 ]; then
    fail=1
    echo "--- anchor shard $i log tail ---"
    tail -120 "$LOGDIR/issue-2223-nap-p3-anchor-shard$i.log"
  fi
done
[ "$fail" -eq 0 ] || { echo "[nap-p3] anchor wave FAILED"; exit 1; }

GIT_SHA=$(git rev-parse HEAD)
uv run python -c "
import json, sys, time
json.dump(
    {
        'issue': 2223,
        'label': 'native_axis_fidelity_preimage',
        'phase': 'p3_done',
        'rc': 0,
        'ts': time.time(),
        'git_sha': sys.argv[1],
        'out_root': sys.argv[2],
    },
    open(sys.argv[3], 'w'),
    indent=2,
)
" "$GIT_SHA" "$OUT_ROOT" "$SENTINEL"
echo "[nap-p3] sentinel written: $SENTINEL"
echo "[phase=done]"
