#!/usr/bin/env bash
# Issue #2544 pod-side dispatcher (plan §4 DAG) — invoked by path via --workload-cmd.
#
#   bash scripts/issue2544_dispatch.sh --full                 # stage -> P1 -> pass1 -> P4a -> pass2 -> P4b
#   bash scripts/issue2544_dispatch.sh --smoke                # SAME chain (incl. config), tiny slice, 3 rungs
#   bash scripts/issue2544_dispatch.sh --full --from-phase pass2   # resume from a phase
#   bash scripts/issue2544_dispatch.sh --full \
#     --pilot-from-hf issue2544_stage_map/pilot_halt2         # v8: pilot leg stages the
#                                                             # VM-refinalized report+pins
#                                                             # (no GPU pilot units)
#
# Contracts (inherited from issue1902_dispatch.sh): set -euo pipefail; each
# leg is a single `uv run python scripts/issue2544_run.py --phase X ...`
# whose rc propagates DIRECTLY (no false-in-branch shapes); the rc=7 gate
# halts (Gate A/A'/B, parity, cost aborts) propagate as designed halts.
# Worker legs are backgrounded with CUDA_VISIBLE_DEVICES pinned in the
# LAUNCHER env (the CVD clobber family, gotchas.md) and re-shard off the
# REALIZED GPU width. Per-phase logs land under /workspace/logs.
# [phase=done] is RESERVED for this dispatcher's single terminal line
# (pod-side-reporting.md).
#
# Phase chain vs the plan DAG: stage=pod staging; pilot=P1 (Gate A);
# pass1=P2+P3a (+P2.5 Gate A' in --finalize); sweep=P4a (layer* freeze +
# Gate B, EPM_ISSUE2544_FITS_STAGE=p4a); pass2=P3b (band B6 captures);
# fits=P4b (EPM_ISSUE2544_FITS_STAGE=p4b). P0 (--phase config) runs VM-side
# before dispatch on --full (re-runnable here via --from-phase config); the
# --smoke chain INCLUDES config so the smoke is self-contained against the
# _smoke HF prefix. P5 (issue2544_figures.py) runs VM-SIDE — NOT here.
set -euo pipefail

# Self-resolving repo root: RunPod checkout, GCE $WORKLOAD_ROOT clone, or the
# script's own parent (set-u-safe on every lane).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional .env sourcing — the SLURM/GCE lanes export tokens and have NO
# .env file (gotchas.md conditional-sourcing rule, #923).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# ── args ─────────────────────────────────────────────────────────────────────
MODE=""
FROM_PHASE=""
PILOT_FROM_HF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full) MODE="full" ;;
    --smoke) MODE="smoke" ;;
    --from-phase) shift; FROM_PHASE="${1:?--from-phase needs a phase name}" ;;
    --pilot-from-hf) shift; PILOT_FROM_HF="${1:?--pilot-from-hf needs an HF prefix}" ;;
    *) echo "[dispatch] unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
if [ -z "$MODE" ]; then
  echo "[dispatch] usage: issue2544_dispatch.sh --full|--smoke [--from-phase <p>]" >&2
  exit 2
fi
SMOKE=""
[ "$MODE" = "smoke" ] && SMOKE="1"
# --full assumes P0 ran VM-side (plan §9 off_pod_phases); --smoke is
# self-contained (config first, against the _smoke write prefix).
if [ -z "$FROM_PHASE" ]; then
  if [ -n "$SMOKE" ]; then FROM_PHASE="config"; else FROM_PHASE="stage"; fi
fi

# ── environment / paths ──────────────────────────────────────────────────────
if [ -d /workspace ]; then
  LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
else
  LOG_DIR="${EPM_LOG_DIR:-$REPO_ROOT/logs/issue_2544}"
fi
mkdir -p "$LOG_DIR"
export EPM_SENTINEL_DIR="$LOG_DIR"

OUT_ROOT_FULL="${EPM_OUT_ROOT:-${WORKLOAD_ROOT:-/workspace}/issue2544}"
OUT_ROOT="$OUT_ROOT_FULL"
# Smoke gets its OWN out-root (per-leg out-roots, crash-fix-rounds rule).
[ -n "$SMOKE" ] && OUT_ROOT="${OUT_ROOT_FULL}_smoke"

# Smoke rung slice: one token per roster class (plan smoke enumeration).
RUNGS_SMOKE="r0 main R"
RUNGS_ARGS=()
[ -n "$SMOKE" ] && RUNGS_ARGS=(--rungs "$RUNGS_SMOKE")

SMOKE_FLAG=()
[ -n "$SMOKE" ] && SMOKE_FLAG=(--smoke)
# Smoke HF WRITE paths divert under _smoke so a smoke re-run can never
# overwrite production rollouts/store/eval-mirror (issue2544_run REFUSES
# --smoke on the production prefix); read paths (#1902 corpus) stay
# production. The eval tree diverts into the smoke out-root.
if [ -n "$SMOKE" ]; then
  export EPM_ISSUE1902_HF_WRITE_PREFIX="issue2544_stage_map/_smoke"
  export EPM_ISSUE2544_EVAL_DIR="$OUT_ROOT/eval_results/issue_2544"
fi

# The FULL leg reaps the derived smoke root at entry (chained smoke-then-full
# out-root residue starves the quota'd lane's headroom asserts).
if [ -z "$SMOKE" ]; then
  if [ -d "${OUT_ROOT_FULL}_smoke" ]; then
    echo "[dispatch] reaping sibling smoke out-root ${OUT_ROOT_FULL}_smoke"
    rm -rf "${OUT_ROOT_FULL}_smoke"
  else
    echo "[dispatch] no sibling smoke out-root to reap"
  fi
fi
mkdir -p "$OUT_ROOT"

CORPUS_DIR="$OUT_ROOT/corpus"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Realized GPU width + PHYSICAL id list — SLURM allocation FIRST (#1902 crash
# 1: nvidia-smi counts the PHYSICAL node and ignores CUDA_VISIBLE_DEVICES, so
# a bare detected count over-shards onto other tenants' devices). Derivation
# lives in issue1902_common.realized_gpu_ids; issue2544_common is imported
# FIRST (the ladder/prefix env-order contract).
NGPU_DETECT="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
[ "$NGPU_DETECT" -ge 1 ] 2>/dev/null || NGPU_DETECT=1
GPU_LINE="$(EPM_I2544_DETECTED="$NGPU_DETECT" uv run python -c '
import os, sys
sys.path.insert(0, "scripts")
import issue2544_common as C2  # noqa: F401  (env-order: MUST precede issue1902_common)
import issue1902_common as C
src, ids = C.realized_gpu_ids(os.environ, int(os.environ["EPM_I2544_DETECTED"]))
print(src + "|" + ",".join(ids))
')"
WIDTH_SRC="${GPU_LINE%%|*}"
GPU_IDS_CSV="${GPU_LINE##*|}"
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS_CSV"
NGPU="${#GPU_ID_ARR[@]}"
# Narrow EVERY child to the allocated devices; per-worker legs re-pin ONE id
# in run_workers below. Process-local — derived FROM the SLURM allocation.
export CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV"
echo "[dispatch] mode=$MODE ngpu=$NGPU ($WIDTH_SRC) gpu_ids=$GPU_IDS_CSV out_root=$OUT_ROOT log_dir=$LOG_DIR"

RUN=(uv run python scripts/issue2544_run.py)
COMMON=(--out-root "$OUT_ROOT" --corpus-dir "$CORPUS_DIR" "${RUNGS_ARGS[@]}" "${SMOKE_FLAG[@]}")

# ── helpers ──────────────────────────────────────────────────────────────────

run_single() {
  # One foreground leg; rc propagates directly under set -e (rc=7 included).
  local name="$1"; shift
  local log="$LOG_DIR/issue-2544-${name}.log"
  echo "[dispatch] $name -> $log"
  "${RUN[@]}" "$@" "${COMMON[@]}" >"$log" 2>&1 || {
    local rc=$?
    echo "[dispatch] $name FAILED rc=$rc — log tail:"
    tail -n 40 "$log" || true
    exit "$rc"
  }
}

run_workers() {
  # N queue-worker legs for one pass phase, one per realized (ALLOCATED) GPU.
  # CVD is pinned in the LAUNCHER env per worker to the ALLOCATION's physical
  # id — never the bare slot index (#1902 crash 1) — with a matching
  # --gpu-id; the in-process clobber is defeated by import-time cuInit
  # (gotchas.md). Workers pull (rung x unit) tasks from the shared file-locked
  # queue under the K=4 rung-residency admission (issue2544_common.UnitQueue),
  # so all N stay fed from <=4 resident rungs.
  local phase="$1"
  local pids=() names=()
  local g
  for (( g=0; g<NGPU; g++ )); do
    local gid="${GPU_ID_ARR[$g]}"
    local wid="w${g}"
    local log="$LOG_DIR/issue-2544-${phase}-${wid}.log"
    echo "[dispatch] $phase worker=$wid gpu=$gid -> $log"
    CUDA_VISIBLE_DEVICES="$gid" "${RUN[@]}" --phase "$phase" --worker \
      --worker-id "$wid" --gpu-id "$gid" "${COMMON[@]}" >"$log" 2>&1 &
    pids+=($!); names+=("$wid")
  done
  # First-failure fast exit + SIBLING KILL (worker-failure-delayed /
  # queue-stale-running pairing): wait for ANY worker (`wait -n`); on a
  # non-zero exit TERM->KILL the surviving siblings instead of letting them
  # grind for hours behind a failed phase — their orphaned queue claims are
  # reclaimed on relaunch (UnitQueue.reclaim_stale at init).
  local rc=0 w_rc=0 n_left=${#pids[@]}
  while [ "$n_left" -gt 0 ]; do
    w_rc=0
    wait -n || w_rc=$?
    n_left=$((n_left - 1))
    if [ "$w_rc" -ne 0 ]; then
      rc=$w_rc
      echo "[dispatch] $phase: a worker FAILED rc=$rc — killing surviving sibling worker(s)"
      local p
      for p in "${pids[@]}"; do
        kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null || true
      done
      local i live
      for i in 1 2 3 4 5 6; do
        sleep 5
        live=0
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && live=$((live + 1)); done
        [ "$live" -eq 0 ] && break
      done
      for p in "${pids[@]}"; do
        kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null || true
      done
      wait 2>/dev/null || true
      local k=0
      for p in "${pids[@]}"; do
        echo "[dispatch] $phase worker=${names[$k]} log tail:"
        tail -n 25 "$LOG_DIR/issue-2544-${phase}-${names[$k]}.log" || true
        k=$((k + 1))
      done
      exit "$rc"
    fi
  done
}

# ── phase chain (resume via --from-phase) ────────────────────────────────────
PHASES=(config stage pilot pass1 sweep pass2 fits)
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

if phase_wanted config; then
  echo "[phase=config]"
  run_single config --phase config
fi

if phase_wanted stage; then
  echo "[phase=stage]"
  run_single stage --phase stage
fi

if phase_wanted pilot; then
  echo "[phase=pilot]"
  if [ -n "$PILOT_FROM_HF" ]; then
    # v8 P1' staged path: the pilot ran as the VM-side --refinalize; stage +
    # validate its report AND pins (BOTH required — R.load_pins raises on a
    # missing revision_pins.json) FAIL-LOUD before any GPU work, write the
    # pilot sentinel, then the chain proceeds to pass1.
    run_single pilot-stage --phase pilot --pilot-from-hf "$PILOT_FROM_HF"
  else
    run_single pilot-init --phase pilot --init
    run_workers pilot
    run_single pilot-finalize --phase pilot --finalize   # Gate A / bf16 / cost halts exit rc=7
    if [ -n "$SMOKE" ]; then
      # v8 smoke leg: refinalize against the smoke's OWN local pilot rollouts
      # (executes the P1' branch under smoke; the HF-download transport + the
      # branch->SHA drift tripwire are production-only — blind-spot enumerated).
      run_single pilot-refinalize --phase pilot --refinalize
    fi
  fi
fi

if phase_wanted pass1; then
  echo "[phase=pass1]"
  run_single pass1-init --phase pass1 --init
  run_workers pass1
  run_single pass1-finalize --phase pass1 --finalize   # P2.5 + Gate A' halt exits rc=7
fi

if phase_wanted sweep; then
  # P4a (unit B fits stage 1): diagonal 17-layer sweep + layer* freeze +
  # Gate B. Single process, ALL allocated GPUs (device-pinned worker threads
  # inside issue2544_fits — no per-leg CVD pin needed). Explicit export/unset
  # (a VAR=x prefix on a shell FUNCTION has mode-dependent persistence).
  echo "[phase=sweep]"
  export EPM_ISSUE2544_FITS_STAGE=p4a
  run_single sweep --phase fits
  unset EPM_ISSUE2544_FITS_STAGE
fi

if phase_wanted pass2; then
  echo "[phase=pass2]"
  run_single pass2-init --phase pass2 --init
  run_workers pass2
  run_single pass2-finalize --phase pass2 --finalize
fi

if phase_wanted fits; then
  # P4b: cells + transfer + operator battery + bootstrap finalize + the
  # registered eval JSONs, mirrored to HF. Gate/parity/pilot halts exit rc=7.
  echo "[phase=fits]"
  export EPM_ISSUE2544_FITS_STAGE=p4b
  run_single fits --phase fits
  unset EPM_ISSUE2544_FITS_STAGE
fi

echo "[dispatch] chain complete (mode=$MODE)"
echo "[phase=done]"
