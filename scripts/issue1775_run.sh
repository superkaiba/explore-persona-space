#!/usr/bin/env bash
# #1775 dispatcher: P1 linear -> P2 detection -> P3 nonlinear -> P4 bilinear.
#
# Modes: --all (production) | --smoke (SAME chain, 1 arm x 1 fold x 1 rung x
# r in {0,1} x B=20, scratch out-root — smoke IS the sweep, PASS_UNIFIED) |
# --from-phase pN (resume; per-unit JSONL resume inside each phase too) |
# --phases=<csv> (targeted: run EXACTLY the listed phases, each with its
# assembly + eval-JSON upload + .done sentinel; mutually exclusive with
# --from-phase; a standalone p3 prefetches its P1/P2 inputs from the Hub —
# staged stores + linear_fits.json + detection JSON + per-row ridge preds).
#
# Sharding: launch width re-derived from the REALIZED GPU count (nvidia-smi);
# 2+ GPUs -> 2 shards per GPU phase, CUDA_VISIBLE_DEVICES pinned in the
# LAUNCHER env per shard (the CVD clobber family, gotchas.md). Store staging
# runs ONCE in the parent BEFORE any fan-out (#1315 shared-staging race).
#
# Self-resolving REPO_ROOT (no bare $WORKLOAD_ROOT — the #825/#1329 class);
# conditional .env sourcing (GCE has no .env — tokens ride instance metadata).
# Designed halts keep their distinct rc: 7=pilot-gate, 21=gate-C, 22=power.
#
# Durability (#825 class, round-2 Major-2): each phase script uploads its
# eval JSONs to HF (issue1775_nonlinearity/eval_json/<phase>/, non-LFS text
# path) in the same per-phase upload step as its analysis tensors
# (upload_phase_eval_json in issue1775_common) — the crash-survivable channel
# on the auto-DELETE GCP lane; the finalize/results-sentinel pull stays as-is.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

MODE=""
FROM_PHASE="p1"
FROM_PHASE_EXPLICIT=""
PHASES=""
for arg in "$@"; do
  case "$arg" in
    --all) MODE="all" ;;
    --smoke) MODE="smoke" ;;
    --from-phase=*) FROM_PHASE="${arg#--from-phase=}"; FROM_PHASE_EXPLICIT=1 ;;
    --phases=*) PHASES="${arg#--phases=}" ;;
    p1|p2|p3|p4) FROM_PHASE="$arg"; FROM_PHASE_EXPLICIT=1 ;;
    *) echo "[dispatch] unknown arg: $arg" >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ]; then
  echo "usage: issue1775_run.sh --all|--smoke [--from-phase=pN | --phases=pN[,pM...]]" >&2
  exit 2
fi
if [ -n "$PHASES" ] && [ -n "$FROM_PHASE_EXPLICIT" ]; then
  echo "[dispatch] --phases and --from-phase are mutually exclusive" >&2
  exit 2
fi
if [ -n "$PHASES" ]; then
  for p in ${PHASES//,/ }; do
    case "$p" in
      p1|p2|p3|p4) ;;
      *) echo "[dispatch] --phases: unknown phase token '$p' (want p1..p4)" >&2; exit 2 ;;
    esac
  done
  echo "[dispatch] TARGETED mode: phases=$PHASES"
fi

SMOKE_FLAG=""
if [ "$MODE" = "smoke" ]; then
  SMOKE_FLAG="--smoke"
  export I1775_OUT_ROOT="${I1775_SMOKE_OUT_ROOT:-/tmp/issue-1775-smoke}"
  mkdir -p "$I1775_OUT_ROOT"
  echo "[dispatch] SMOKE: out-root redirected to $I1775_OUT_ROOT (committed paths untouched)"
fi
OUT_ROOT_RESOLVED="${I1775_OUT_ROOT:-$REPO_ROOT}"

# Sentinel dir: /workspace/logs on the pod/GCE contract lane; out-root fallback
# on the VM smoke (same code path, different destination).
SENTINEL_DIR="/workspace/logs"
if ! mkdir -p "$SENTINEL_DIR" 2>/dev/null || [ ! -w "$SENTINEL_DIR" ]; then
  SENTINEL_DIR="$OUT_ROOT_RESOLVED/logs"
  mkdir -p "$SENTINEL_DIR"
fi
echo "[dispatch] sentinel dir: $SENTINEL_DIR"

# Realized GPU width (re-shard off realized width; a degraded 1x launch runs
# the same fan-out serially — plan section 9).
NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)"
NGPU="${NGPU//[^0-9]/}"
[ -z "$NGPU" ] && NGPU=0
if [ "$NGPU" -ge 2 ]; then SHARDS=2; DEVICE=cuda
elif [ "$NGPU" -eq 1 ]; then SHARDS=1; DEVICE=cuda
else SHARDS=1; DEVICE=cpu; fi
echo "[dispatch] realized gpus=$NGPU -> shards=$SHARDS device=$DEVICE"

# Resume-aware headroom preamble (>=30 GB fresh at the out-root mount; halved
# on a --from-phase resume whose earlier outputs legitimately occupy the disk).
NEED_GB=30
[ "$FROM_PHASE" != "p1" ] && NEED_GB=15
[ -n "$PHASES" ] && NEED_GB=15   # targeted re-run: resume-like footprint
[ "$MODE" = "smoke" ] && NEED_GB=2
# shared helper (mount-binding rule): statvfs floor + posix_fallocate EDQUOT canary
uv run python -c "
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_ROOT_RESOLVED', $NEED_GB, phase='dispatch')
print(f'[headroom] {free:.1f} GB free at $OUT_ROOT_RESOLVED (need >= $NEED_GB)')
" || { echo "[dispatch] FATAL: insufficient disk headroom" >&2; exit 3; }

phase_order() { case "$1" in p1) echo 1 ;; p2) echo 2 ;; p3) echo 3 ;; p4) echo 4 ;; *) echo 9 ;; esac; }
FROM_N="$(phase_order "$FROM_PHASE")"
T_START=$(date +%s)

phase_selected() {
  # Targeted mode: exact membership in --phases; else the --from-phase order.
  local p="$1"
  if [ -n "$PHASES" ]; then
    case ",$PHASES," in
      *",$p,"*) return 0 ;;
      *) return 1 ;;
    esac
  fi
  [ "$(phase_order "$p")" -ge "$FROM_N" ]
}

run_sharded() {
  # run_sharded <script> <extra args...>: fan out SHARDS subprocesses, one per
  # GPU, CVD pinned in the LAUNCHER env; then join fail-loud preserving rc.
  local script="$1"; shift
  if [ "$SHARDS" -le 1 ]; then
    OK=yes
    uv run python "scripts/$script" "$@" --device "$DEVICE" --num-shards 1 --shard-index 0 || OK="rc=$?"
    [ "$OK" = yes ] || { echo "[dispatch] $script failed ($OK)" >&2; return "${OK#rc=}"; }
    return 0
  fi
  local pids=() g rc=0
  for g in $(seq 0 $((SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$g" uv run python "scripts/$script" "$@" \
      --device cuda --num-shards "$SHARDS" --shard-index "$g" \
      > "$SENTINEL_DIR/issue-1775-${script%.py}-shard$g.log" 2>&1 &
    pids+=("$!")
  done
  local i=0
  for g in $(seq 0 $((SHARDS - 1))); do
    if ! wait "${pids[$i]}"; then
      rc=$?
      echo "[dispatch] $script shard $g failed rc=$rc — tail of its log:" >&2
      tail -n 60 "$SENTINEL_DIR/issue-1775-${script%.py}-shard$g.log" >&2 || true
    fi
    i=$((i + 1))
  done
  return "$rc"
}

write_done() {
  printf '{"phase": "%s", "ts": "%s", "mode": "%s"}\n' "$1" "$(date -u +%FT%TZ)" "$MODE" \
    > "$SENTINEL_DIR/issue-1775-$1.done"
}

route_rc() {
  # Designed halts keep their distinct rc (never an anonymous crash).
  local rc="$1" phase="$2"
  case "$rc" in
    0) return 0 ;;
    7) echo "[dispatch] $phase PILOT-GATE deviation halt (rc=7; see pilot_gate_report.json)" >&2 ;;
    21) echo "[dispatch] $phase GATE-C reproduction failure (rc=21; see gate_c_failure.json)" >&2 ;;
    22) echo "[dispatch] $phase POWER-CHECK failure (rc=22 — detection instrument bug)" >&2 ;;
    23) echo "[dispatch] $phase ASSEMBLY-INCOMPLETE (rc=23 — planned units missing a completed row; see log)" >&2 ;;
    *) echo "[dispatch] $phase crashed rc=$rc" >&2 ;;
  esac
  exit "$rc"
}

# ── one-time store staging in the PARENT (before any fan-out; #1315) ──────────
echo "[phase=stage]"
OK=yes
uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from issue1775_common import CELL_PRIMARY, resolve_store_dir, stage_store_if_needed
stage_store_if_needed(resolve_store_dir(), cells=[CELL_PRIMARY, 'cell_pre_own'], layers=[14, 19])
print('[stage] store ready at', resolve_store_dir())
" || OK="rc=$?"
[ "$OK" = yes ] || { echo "[dispatch] store staging failed ($OK)" >&2; exit 4; }

# ── standalone-P3 input prefetch (targeted mode without its producers) ────────
# P3 reads P1's linear_fits.json + per-row ridge preds (assemble_gains) and
# P2's detection JSON (gate-B unit grid). On a fresh instance those exist only
# on the Hub (eval_json/* + analysis_tensors/heldout_preds); local copies win.
if [ -n "$PHASES" ] && phase_selected p3 && ! phase_selected p1; then
  echo "[phase=p3_prefetch]"
  OK=yes
  uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from issue1775_common import prefetch_p3_inputs
prefetch_p3_inputs(smoke=bool('$SMOKE_FLAG'))
" || OK="rc=$?"
  [ "$OK" = yes ] || { echo "[dispatch] p3 input prefetch failed ($OK)" >&2; exit 6; }
fi

# ── P1 linear ─────────────────────────────────────────────────────────────────
if phase_selected p1; then
  echo "[phase=p1_linear]"
  OK=yes
  run_sharded issue1775_ladder.py --phase linear $SMOKE_FLAG || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" p1
  if [ "$SHARDS" -gt 1 ]; then
    OK=yes
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1775_ladder.py --phase linear \
      $SMOKE_FLAG --device "$DEVICE" --num-shards "$SHARDS" --shard-index 0 \
      --assemble-only || OK="rc=$?"
    [ "$OK" = yes ] || route_rc "${OK#rc=}" p1-assemble
  fi
  write_done p1
fi

# ── P2 detection (single GPU; batched draws) ─────────────────────────────────
if phase_selected p2; then
  echo "[phase=p2_detection]"
  OK=yes
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1775_detection.py $SMOKE_FLAG \
    --device "$DEVICE" || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" p2
  write_done p2
fi

# ── P3 nonlinear ladder ───────────────────────────────────────────────────────
if phase_selected p3; then
  echo "[phase=p3_nonlinear]"
  OK=yes
  run_sharded issue1775_ladder.py --phase nonlinear $SMOKE_FLAG || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" p3
  if [ "$SHARDS" -gt 1 ]; then
    OK=yes
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1775_ladder.py --phase nonlinear \
      $SMOKE_FLAG --device "$DEVICE" --num-shards "$SHARDS" --shard-index 0 \
      --assemble-only || OK="rc=$?"
    [ "$OK" = yes ] || route_rc "${OK#rc=}" p3-assemble
  fi
  write_done p3
fi

# ── P4 bilinear + doubly pass + interpretation ────────────────────────────────
if phase_selected p4; then
  echo "[phase=p4_bilinear]"
  OK=yes
  run_sharded issue1775_bilinear.py $SMOKE_FLAG || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" p4
  OK=yes
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1775_bilinear.py $SMOKE_FLAG \
    --device "$DEVICE" --num-shards "$SHARDS" --shard-index 0 --assemble-only \
    || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" p4-assemble
  # doubly-novel robustness pass at r in {0, r*} (production only)
  if [ "$MODE" = "all" ]; then
    RSTAR="$(uv run python -c "
import json, sys
sys.path.insert(0, 'scripts')
from issue1775_common import eval_dir
d = json.loads((eval_dir('bilinear') / 'bilinear_fits.json').read_text())
r = d.get('schemes', {}).get('prefix', {}).get('r_star_inner_val')
print(r if r is not None else '')
")"
    if [ -n "$RSTAR" ]; then
      echo "[phase=p4_doubly r_star=$RSTAR]"
      OK=yes
      CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1775_bilinear.py \
        --device "$DEVICE" --schemes doubly --r-grid "0,$RSTAR" \
        --num-shards 1 --shard-index 0 || OK="rc=$?"
      [ "$OK" = yes ] || route_rc "${OK#rc=}" p4-doubly
    else
      echo "[dispatch] no r_star (all-null inner selection?) — doubly pass skipped"
    fi
  fi
  write_done p4
fi

# ── results sentinel (the /issue Step 7 payload contract) ─────────────────────
echo "[phase=results_sentinel]"
WALL_H="$(uv run python -c "import time; print(f'{(time.time() - $T_START) / 3600:.3f}')")"
GPU_H="$(uv run python -c "print(f'{max($NGPU, 0) * $WALL_H:.2f}')")"
OK=yes
uv run python scripts/issue1775_sentinel.py --dest "$SENTINEL_DIR/issue-1775-results.json" \
  --gpu-hours-used "$GPU_H" $SMOKE_FLAG || OK="rc=$?"
[ "$OK" = yes ] || { echo "[dispatch] sentinel write failed ($OK)" >&2; exit 5; }

echo "[phase=done]"
