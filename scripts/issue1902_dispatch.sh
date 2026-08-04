#!/usr/bin/env bash
# Issue #1902 pod-side dispatcher (plan v4 §4) — invoked by path via --workload-cmd.
#
#   bash scripts/issue1902_dispatch.sh --full                # P1 -> gate A -> P2 -> P3 -> (P4 placeholder)
#   bash scripts/issue1902_dispatch.sh --smoke               # SAME chain, tiny slice (PASS_UNIFIED)
#   bash scripts/issue1902_dispatch.sh --full --from-phase gen   # resume from a phase
#
# Contracts: set -euo pipefail; each phase is a single `uv run python
# scripts/issue1902_run.py --phase X ...` whose rc propagates DIRECTLY (no
# false-in-branch shapes); the rc=7 survival/gate halt propagates as the
# designed halt (plan §7 / the #1415 convention). Per-checkpoint GPU legs are
# backgrounded with CUDA_VISIBLE_DEVICES pinned in the LAUNCHER env (the CVD
# clobber family, gotchas.md) and re-shard off the REALIZED GPU width (a
# degraded 2-wide launch runs 2 waves). Per-phase logs land under
# /workspace/logs. [phase=done] is RESERVED for this dispatcher's single
# terminal line (pod-side-reporting.md).
set -euo pipefail

# Self-resolving repo root: RunPod checkout, GCE $WORKLOAD_ROOT clone, or the
# script's own parent (set-u-safe on every lane; SKILL.md Step 6b rule (f)).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional .env sourcing — the GCE lane exports tokens via instance
# metadata and has NO .env file (gotchas.md conditional-sourcing rule, #923).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# ── args ─────────────────────────────────────────────────────────────────────
MODE=""
FROM_PHASE="stage"
while [ $# -gt 0 ]; do
  case "$1" in
    --full) MODE="full" ;;
    --smoke) MODE="smoke" ;;
    --from-phase) shift; FROM_PHASE="${1:?--from-phase needs a phase name}" ;;
    *) echo "[dispatch] unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
if [ -z "$MODE" ]; then
  echo "[dispatch] usage: issue1902_dispatch.sh --full|--smoke [--from-phase <p>]" >&2
  exit 2
fi
SMOKE=""
[ "$MODE" = "smoke" ] && SMOKE="1"

# ── environment / paths ──────────────────────────────────────────────────────
if [ -d /workspace ]; then
  LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
else
  LOG_DIR="${EPM_LOG_DIR:-$REPO_ROOT/logs/issue_1902}"
fi
mkdir -p "$LOG_DIR"
export EPM_SENTINEL_DIR="$LOG_DIR"

OUT_ROOT_FULL="${EPM_OUT_ROOT:-${WORKLOAD_ROOT:-/workspace}/issue1902}"
OUT_ROOT="$OUT_ROOT_FULL"
# Smoke gets its OWN out-root (per-leg out-roots, crash-fix-rounds rule).
[ -n "$SMOKE" ] && OUT_ROOT="${OUT_ROOT_FULL}_smoke"

CKPTS_FULL="B S D R"
CKPTS_SMOKE="B R"
CKPTS="$CKPTS_FULL"
[ -n "$SMOKE" ] && CKPTS="$CKPTS_SMOKE"

SMOKE_FLAG=()
[ -n "$SMOKE" ] && SMOKE_FLAG=(--smoke)
# Smoke HF WRITE paths divert under _smoke so a smoke re-run can never
# overwrite production rollouts/store/eval-mirror (read paths — the corpus
# prefix — stay production; issue1902_common.HF_WRITE_PREFIX).
[ -n "$SMOKE" ] && export EPM_ISSUE1902_HF_WRITE_PREFIX="issue1902_stage_map/_smoke"

# The FULL leg reaps the derived smoke root at entry (chained smoke-then-full
# out-root residue starves the quota'd lane's headroom asserts; #1586 fu r3).
if [ -z "$SMOKE" ]; then
  if [ -d "${OUT_ROOT_FULL}_smoke" ]; then
    echo "[dispatch] reaping sibling smoke out-root ${OUT_ROOT_FULL}_smoke"
    rm -rf "${OUT_ROOT_FULL}_smoke"
  else
    echo "[dispatch] no sibling smoke out-root to reap"
  fi
fi
mkdir -p "$OUT_ROOT"

CORPUS_DIR="$OUT_ROOT/corpus_stage/issue1902_stage_map/corpus"
# Fast the whole vLLM/teacher-forced stack on the allocator knobs (plan §4).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Realized GPU width + PHYSICAL id list — SLURM allocation FIRST (#1902 crash
# 1: `nvidia-smi -L | wc -l` counts the PHYSICAL node — 8x H200 on the shared
# fellows hosts, and nvidia-smi ignores CUDA_VISIBLE_DEVICES — so a bare
# detected count over-shards onto other tenants' devices). Derivation +
# clamping live in issue1902_common.realized_gpu_ids (unit-tested); the
# detected count is the NON-SLURM fallback only.
NGPU_DETECT="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
[ "$NGPU_DETECT" -ge 1 ] 2>/dev/null || NGPU_DETECT=1
GPU_LINE="$(EPM_I1902_DETECTED="$NGPU_DETECT" uv run python -c '
import os, sys
sys.path.insert(0, "scripts")
import issue1902_common as C
src, ids = C.realized_gpu_ids(os.environ, int(os.environ["EPM_I1902_DETECTED"]))
print(src + "|" + ",".join(ids))
')"
WIDTH_SRC="${GPU_LINE%%|*}"
GPU_IDS_CSV="${GPU_LINE##*|}"
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS_CSV"
NGPU="${#GPU_ID_ARR[@]}"
# Narrow EVERY child (pilot-init/finalize, gen/capture finalize, fits) to the
# allocated devices; per-checkpoint legs re-pin ONE id in run_sharded below.
# Process-local to this job's tree — derived FROM the SLURM allocation, never
# a profile-level export (the fellows "no CVD export" rule targets exports
# that FIGHT the scheduler's assignment; this one consumes it).
export CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV"
echo "[dispatch] mode=$MODE ckpts='$CKPTS' ngpu=$NGPU ($WIDTH_SRC) gpu_ids=$GPU_IDS_CSV (allocation) out_root=$OUT_ROOT log_dir=$LOG_DIR"

RUN=(uv run python scripts/issue1902_run.py)
COMMON=(--out-root "$OUT_ROOT" --corpus-dir "$CORPUS_DIR" --ckpts "$CKPTS" "${SMOKE_FLAG[@]}")

# ── helpers ──────────────────────────────────────────────────────────────────

run_single() {
  # One foreground leg; rc propagates directly under set -e (rc=7 included).
  local name="$1"; shift
  local log="$LOG_DIR/issue-1902-${name}.log"
  echo "[dispatch] $name -> $log"
  "${RUN[@]}" "$@" "${COMMON[@]}" >"$log" 2>&1 || {
    local rc=$?
    echo "[dispatch] $name FAILED rc=$rc — log tail:"
    tail -n 40 "$log" || true
    exit "$rc"
  }
}

run_sharded() {
  # Per-checkpoint legs, one per realized (ALLOCATED) GPU, waves when
  # NGPU < n_ckpts. CVD is pinned in the LAUNCHER env per leg to the
  # ALLOCATION's physical id — never the bare slot index, which on a shared
  # fellows node can be another tenant's device (#1902 crash 1) — with a
  # matching --gpu-id; the in-process clobber is defeated by import-time
  # cuInit (gotchas.md).
  local phase="$1"
  local ckpt_arr=($CKPTS)
  local i=0
  while [ "$i" -lt "${#ckpt_arr[@]}" ]; do
    local pids=() names=()
    local g=0
    while [ "$g" -lt "$NGPU" ] && [ "$i" -lt "${#ckpt_arr[@]}" ]; do
      local c="${ckpt_arr[$i]}"
      local gid="${GPU_ID_ARR[$g]}"
      local log="$LOG_DIR/issue-1902-${phase}-${c}.log"
      echo "[dispatch] $phase ckpt=$c gpu=$gid (slot $g) -> $log"
      CUDA_VISIBLE_DEVICES="$gid" "${RUN[@]}" --phase "$phase" --ckpt "$c" --gpu-id "$gid" \
        "${COMMON[@]}" >"$log" 2>&1 &
      pids+=($!); names+=("$c")
      i=$((i + 1)); g=$((g + 1))
    done
    local rc=0 k=0
    for p in "${pids[@]}"; do
      wait "$p" || rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "[dispatch] $phase ckpt=${names[$k]} FAILED rc=$rc — log tail:"
        tail -n 40 "$LOG_DIR/issue-1902-${phase}-${names[$k]}.log" || true
        exit "$rc"
      fi
      k=$((k + 1))
    done
  done
}

# ── phase chain (resume via --from-phase) ────────────────────────────────────
PHASES=(stage pilot gen capture fits)
START_IDX=-1
for idx in "${!PHASES[@]}"; do
  [ "${PHASES[$idx]}" = "$FROM_PHASE" ] && START_IDX="$idx"
done
if [ "$START_IDX" -lt 0 ]; then
  echo "[dispatch] unknown --from-phase '$FROM_PHASE' (one of: ${PHASES[*]})" >&2
  exit 2
fi

phase_wanted() {
  local phase="$1"
  for idx in "${!PHASES[@]}"; do
    if [ "${PHASES[$idx]}" = "$phase" ]; then
      [ "$idx" -ge "$START_IDX" ] && return 0
      return 1
    fi
  done
  return 1
}

if phase_wanted stage; then
  echo "[phase=stage]"
  run_single stage --phase stage
fi

if phase_wanted pilot; then
  echo "[phase=pilot]"
  run_single pilot-init --phase pilot --init
  run_sharded pilot
  run_single pilot-finalize --phase pilot --finalize   # gate A / bf16 / A12 halts exit rc=7
fi

if phase_wanted gen; then
  echo "[phase=gen]"
  run_sharded gen
  run_single gen-finalize --phase gen --finalize       # gate A' halt exits rc=7
fi

if phase_wanted capture; then
  echo "[phase=capture]"
  run_sharded capture
  run_single capture-finalize --phase capture --finalize
fi

if phase_wanted fits; then
  # P4 (unit C): fits + transfer + operator battery. Single process, ALL
  # visible GPUs (device-pinned worker threads inside issue1902_fits — no
  # per-leg CVD pin needed); gate B / parity / pilot halts exit rc=7. P5
  # (issue1902_figures.py) runs VM-SIDE off the committed JSONs — NOT here.
  echo "[phase=fits]"
  run_single fits --phase fits
fi

echo "[dispatch] chain complete (mode=$MODE ckpts='$CKPTS')"
echo "[phase=done]"
