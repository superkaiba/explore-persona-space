#!/usr/bin/env bash
# #1775 fu-round dispatcher (`dedup-refit-pcfold-doubly`, plan v8):
#   F1a cell-1 staging + dedup recompute (network/CPU, BACKGROUND)
#   F2  cell-2 fold-PC bilinear (GPU)          } run while F1a streams
#   F3  cell-3 doubly stitch-MLP + delta (GPU) } (work-conserving overlap)
#   F1b cell-1 refits (GPU, after F1a joins)
#
# Modes: --all (production) | --smoke (SAME chain — F1a capped at 1 chunk with
# planted dupes, F2 at 1 fold x r in {0,32} x 1 seed, F3 at 1 fold x 1 seed x
# 8 epochs on the FULL fit population (the reused run-1 doubly shards are
# production-fold-shaped, so a row-limited manifest could not exercise the
# cross-phase consumer — #518 class), scratch out-root; smoke IS the sweep,
# PASS_UNIFIED) | --phases=<csv of f1a,f2,f3,f1b> (targeted re-run).
#
# Reuses run-1's dispatcher conventions (issue1775_run.sh): self-resolving
# REPO_ROOT, conditional .env sourcing, OK-flag exit pattern (one exit point
# per phase), per-phase sentinels /workspace/logs/issue-1775-fu-*.done,
# route_rc designed halts (24 = fold-PC r0-vs-ridge reproduction failure),
# results sentinel via issue1775_sentinel.py --fu.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

MODE=""
PHASES=""
for arg in "$@"; do
  case "$arg" in
    --all) MODE="all" ;;
    --smoke) MODE="smoke" ;;
    --phases=*) PHASES="${arg#--phases=}" ;;
    *) echo "[fu-dispatch] unknown arg: $arg" >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ]; then
  echo "usage: issue1775_fu_run.sh --all|--smoke [--phases=f1a[,f2,f3,f1b]]" >&2
  exit 2
fi
if [ -n "$PHASES" ]; then
  for p in ${PHASES//,/ }; do
    case "$p" in
      f1a|f1b|f2|f3) ;;
      *) echo "[fu-dispatch] --phases: unknown phase token '$p'" >&2; exit 2 ;;
    esac
  done
  echo "[fu-dispatch] TARGETED mode: phases=$PHASES"
fi

SMOKE_FLAG=""
if [ "$MODE" = "smoke" ]; then
  SMOKE_FLAG="--smoke"
  export I1775_OUT_ROOT="${I1775_SMOKE_OUT_ROOT:-/tmp/issue-1775-fu-smoke}"
  mkdir -p "$I1775_OUT_ROOT"
  echo "[fu-dispatch] SMOKE: out-root redirected to $I1775_OUT_ROOT (committed paths untouched)"
fi
OUT_ROOT_RESOLVED="${I1775_OUT_ROOT:-$REPO_ROOT}"

SENTINEL_DIR="/workspace/logs"
if ! mkdir -p "$SENTINEL_DIR" 2>/dev/null || [ ! -w "$SENTINEL_DIR" ]; then
  SENTINEL_DIR="$OUT_ROOT_RESOLVED/logs"
  mkdir -p "$SENTINEL_DIR"
fi
echo "[fu-dispatch] sentinel dir: $SENTINEL_DIR"

NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)"
NGPU="${NGPU//[^0-9]/}"
[ -z "$NGPU" ] && NGPU=0
DEVICE=cuda
[ "$NGPU" -eq 0 ] && DEVICE=cpu
echo "[fu-dispatch] realized gpus=$NGPU device=$DEVICE (plan section 9: 1x A100 — GPU phases serial)"

# Resume-aware headroom preamble (plan section 9: peak = one chunk + reduced
# arrays + #1092 stores + pass_b bundle << 30 GB).
NEED_GB=30
[ -n "$PHASES" ] && NEED_GB=15
[ "$MODE" = "smoke" ] && NEED_GB=4
uv run python -c "
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_ROOT_RESOLVED', $NEED_GB, phase='fu-dispatch')
print(f'[headroom] {free:.1f} GB free at $OUT_ROOT_RESOLVED (need >= $NEED_GB)')
" || { echo "[fu-dispatch] FATAL: insufficient disk headroom" >&2; exit 3; }

phase_selected() {
  local p="$1"
  if [ -n "$PHASES" ]; then
    case ",$PHASES," in
      *",$p,"*) return 0 ;;
      *) return 1 ;;
    esac
  fi
  return 0
}

write_done() {
  printf '{"phase": "%s", "ts": "%s", "mode": "%s"}\n' "$1" "$(date -u +%FT%TZ)" "$MODE" \
    > "$SENTINEL_DIR/issue-1775-fu-$1.done"
}

route_rc() {
  local rc="$1" phase="$2"
  case "$rc" in
    0) return 0 ;;
    23) echo "[fu-dispatch] $phase ASSEMBLY-INCOMPLETE (rc=23)" >&2 ;;
    24) echo "[fu-dispatch] $phase FOLD-PC r0-vs-ridge REPRODUCTION failure (rc=24 — code bug, see bilinear_foldpc.json r0_ridge_reproduction)" >&2 ;;
    *) echo "[fu-dispatch] $phase crashed rc=$rc" >&2 ;;
  esac
  exit "$rc"
}

T_START=$(date +%s)

# ── one-time store staging in the PARENT (F2/F3 inputs; #1315 race class) ──────
echo "[phase=fu_stage]"
OK=yes
uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from issue1775_common import CELL_PRIMARY, resolve_store_dir, stage_store_if_needed
stage_store_if_needed(resolve_store_dir(), cells=[CELL_PRIMARY], layers=[14])
print('[stage] store ready at', resolve_store_dir())
" || OK="rc=$?"
[ "$OK" = yes ] || { echo "[fu-dispatch] store staging failed ($OK)" >&2; exit 4; }

# ── F1a: cell-1 staging + dedup (0-GPU, BACKGROUND — overlaps F2/F3) ───────────
F1A_PID=""
F1A_RC_FILE="$SENTINEL_DIR/issue-1775-fu-f1a.rc"
if phase_selected f1a; then
  echo "[phase=fu_f1a_launch]"
  rm -f "$F1A_RC_FILE"
  SMOKE_CHUNK_ARGS=""
  [ "$MODE" = "smoke" ] && SMOKE_CHUNK_ARGS="--max-chunks 1"
  (
    OK1=yes
    uv run python scripts/issue1775_n50k_dedup_refit.py --stage stage --device cpu \
      $SMOKE_FLAG $SMOKE_CHUNK_ARGS \
      > "$SENTINEL_DIR/issue-1775-fu-f1a.log" 2>&1 || OK1="rc=$?"
    if [ "$OK1" = yes ]; then echo 0 > "$F1A_RC_FILE"; else echo "${OK1#rc=}" > "$F1A_RC_FILE"; fi
  ) &
  F1A_PID="$!"
  echo "[fu-dispatch] F1a streaming in background (pid=$F1A_PID; log=$SENTINEL_DIR/issue-1775-fu-f1a.log)"
fi

# ── F2: cell-2 fold-PC bilinear (GPU) ──────────────────────────────────────────
if phase_selected f2; then
  echo "[phase=fu_f2_foldpc]"
  OK=yes
  uv run python scripts/issue1775_bilinear.py --basis pca48_foldpc \
    --schemes prefix --r-grid 0,32 --device "$DEVICE" $SMOKE_FLAG || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" f2
  write_done f2
fi

# ── F3: cell-3 doubly stitch-MLP + delta_beyond(doubly) (GPU) ─────────────────
if phase_selected f3; then
  echo "[phase=fu_f3_doubly_mlp]"
  OK=yes
  uv run python scripts/issue1775_doubly_mlp.py --device "$DEVICE" $SMOKE_FLAG || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" f3
  write_done f3
fi

# ── join F1a, then F1b: cell-1 refits (GPU) ────────────────────────────────────
if phase_selected f1a && [ -n "$F1A_PID" ]; then
  echo "[phase=fu_f1a_join]"
  wait "$F1A_PID" || true
  F1A_RC="$(cat "$F1A_RC_FILE" 2>/dev/null || echo 99)"
  if [ "$F1A_RC" != 0 ]; then
    echo "[fu-dispatch] F1a failed rc=$F1A_RC — tail of its log:" >&2
    tail -n 60 "$SENTINEL_DIR/issue-1775-fu-f1a.log" >&2 || true
    route_rc "$F1A_RC" f1a
  fi
  write_done f1a
fi

if phase_selected f1b; then
  echo "[phase=fu_f1b_refits]"
  OK=yes
  uv run python scripts/issue1775_n50k_dedup_refit.py --stage fits --device "$DEVICE" \
    $SMOKE_FLAG || OK="rc=$?"
  [ "$OK" = yes ] || route_rc "${OK#rc=}" f1b
  write_done f1b
fi

# ── results sentinel (the /issue Step 7 payload contract) ─────────────────────
echo "[phase=results_sentinel]"
WALL_H="$(uv run python -c "import time; print(f'{(time.time() - $T_START) / 3600:.3f}')")"
GPU_H="$(uv run python -c "print(f'{max($NGPU, 0) * $WALL_H:.2f}')")"
OK=yes
uv run python scripts/issue1775_sentinel.py --fu \
  --dest "$SENTINEL_DIR/issue-1775-results.json" \
  --gpu-hours-used "$GPU_H" --gpu-hours-budgeted 4.5 $SMOKE_FLAG || OK="rc=$?"
[ "$OK" = yes ] || { echo "[fu-dispatch] sentinel write failed ($OK)" >&2; exit 5; }

echo "[phase=done]"
