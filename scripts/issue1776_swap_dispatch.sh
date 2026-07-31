#!/usr/bin/env bash
# #1776 follow-up dispatcher — clone of the p3p4 phase chain (same runner
# conventions). TWO rounds share it (--round swap is the byte-preserving
# default; --round patch = slot_patch_sufficiency, plan v8):
#   p0_stage    stage all reused inputs at ONE fresh Hub pin (+ schema probes)
#   p1_build    swap: pairs+targets+deltas (G-METRIC-SANITY rc=9 / G-DOSE rc=7)
#               patch: v14_last capture + patch_vectors (G-CAPTURE-CONS. rc=9)
#   p2_pilot    G-SWAP/PATCH-PARITY rc=8 (binds at smoke) + wall gate rc=7
#   p3_*gen     steered + baseline generation, unit-sharded across all GPUs
#   p3_text     merge per-arm rollout TEXT + manifest, upload BEFORE any reduce
#   p4_capture  teacher-forced v19 capture, sharded; then tensors upload
#   p5_reduce   metric + eligibility-matched nulls + verdict + figures (CPU)
#   p6_commit   eval JSONs + figures -> issue branch; terminal results sentinel
#
# Exit codes: 0 ok; 7 = G-DOSE-DEGENERATE / G-PILOT (report JSON written);
# 8 = parity; 9 = G-METRIC-SANITY (swap) / G-CAPTURE-CONSISTENCY (patch).
# Smoke/dry legs never touch committed eval_results/, figures/, or canonical
# Hub prefixes (scratch redirects); the FULL leg reaps the derived smoke-leg
# roots at entry (#1586 fu r3 class).
#
# NO mid-run staged-cache reap (deliberate, #1489 rule): every hf_dl input is
# consumed by p1_build, but the savings (~1.5 GB vs a >100 GB quota) are
# marginal and a crash-resume re-stages at p0 — Step-8 terminal cleanup owns it.

set -euo pipefail

# ── args / mode ───────────────────────────────────────────────────────────────
MODE="full"
DRY=0
NGPU_OVERRIDE=""
ROUND="swap"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --gpus) NGPU_OVERRIDE="$2"; shift 2 ;;
    --round) ROUND="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "full" || "$MODE" == "smoke" ]] || { echo "--mode full|smoke" >&2; exit 2; }
[[ "$ROUND" == "swap" || "$ROUND" == "patch" ]] || { echo "--round swap|patch" >&2; exit 2; }
if [[ "${EPS_1776_DRY_RUN:-0}" == "1" ]]; then DRY=1; fi
export EPS_1776_SWAP_MODE="$MODE"

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
ISSUE=1776

# Round-derived names: RTAG=swap preserves every legacy path byte-for-byte.
if [[ "$ROUND" == "patch" ]]; then
  RTAG="slotpatch"
  FUD="followup_slotpatch"
  MERGED_SUBDIR="steered_slotpatch"
  GENPHASE="p3_patchgen"
else
  RTAG="swap"
  FUD="followup_swap"
  MERGED_SUBDIR="steered_swap"
  GENPHASE="p3_swapgen"
fi

if [[ "$ROUND" == "patch" ]]; then
  BASE_OUT="${EPS_1776_SWAP_OUT_ROOT:-/workspace/issue_1776_slotpatch}"
else
  BASE_OUT="${EPS_1776_SWAP_OUT_ROOT:-/workspace/issue_1776_swap}"
fi
if [[ $DRY == 1 ]]; then
  TMP_BASE="${EPS_1776_TMP:-$(mktemp -d /tmp/issue-1776-swap-dryrun.XXXXXX)}"
  OUT_ROOT="${EPS_1776_OUT_ROOT:-$TMP_BASE/out}"
  LOG_DIR="${EPS_1776_LOG_DIR:-$TMP_BASE/logs}"
elif [[ "$MODE" == "smoke" ]]; then
  OUT_ROOT="${EPS_1776_OUT_ROOT:-${BASE_OUT}_smoke}"
  LOG_DIR="${EPS_1776_LOG_DIR:-/workspace/logs}"
else
  OUT_ROOT="${EPS_1776_OUT_ROOT:-$BASE_OUT}"
  LOG_DIR="${EPS_1776_LOG_DIR:-/workspace/logs}"
fi
if [[ $DRY == 0 && -d /workspace ]]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
fi
DATA_DIR="$REPO_ROOT/data/issue_1776"
PHASE_LOGS="$OUT_ROOT/logs"
TRACE="$OUT_ROOT/dispatch_trace.txt"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$PHASE_LOGS" "$DATA_DIR"
: > "$TRACE"

# Smoke/dry outputs never touch committed eval_results/, figures/, or the
# canonical Hub prefix (scratch redirects — parent-dispatcher convention).
if [[ "$MODE" == "smoke" || $DRY == 1 ]]; then
  EVAL_DIR="$OUT_ROOT/eval_results/issue_1776/$FUD"
  FIG_DIR="$OUT_ROOT/figures/issue_1776/$FUD"
  HF_PREFIX_EFF="issue1776_jacobian/smoke_probe"
  BUILD_DIR="$DATA_DIR/${RTAG}_build_smoke"
else
  EVAL_DIR="$REPO_ROOT/eval_results/issue_1776/$FUD"
  FIG_DIR="$REPO_ROOT/figures/issue_1776/$FUD"
  HF_PREFIX_EFF="issue1776_jacobian"
  BUILD_DIR="$DATA_DIR/${RTAG}_build"
fi
mkdir -p "$EVAL_DIR"

# Patch-round reused inputs: the swap round's staged build artifacts + the
# COMMITTED swap results (git tree) the historical contrast loads from.
SWAP_ART="$DATA_DIR/hf_dl/issue1776_jacobian/analysis_tensors/followup_swap"
SWAP_BUILD_REPORT="$REPO_ROOT/eval_results/issue_1776/followup_swap/build_report.json"
SWAP_SUCCESS="$REPO_ROOT/eval_results/issue_1776/followup_swap/swap_success.json"

# ── realized width (workload re-shards off realized width) ────────────────────
if [[ -n "$NGPU_OVERRIDE" ]]; then
  NGPU="$NGPU_OVERRIDE"
elif [[ $DRY == 1 ]]; then
  NGPU="${EPS_1776_NGPU:-8}"
else
  NGPU="$(nvidia-smi --list-gpus | wc -l)"
fi
[[ "$NGPU" -ge 1 ]] || { echo "[swap-dispatch] no GPUs visible" >&2; exit 1; }

# ── mode parameter table ──────────────────────────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
  PAIRS_PER_LEG=2; STRATA=1; LIMIT_PAIRS=2
  K_SAMPLES=2; K_BASELINE=2; MAX_NEW=64
  NULL_DRAWS=20; NBOOT=50
  MAX_WC_CHUNKS=1
  BUILD_EXTRA=(--gates-informational)   # gate-calibration rule; parity still binds
  PILOT_BUDGET=2.6
else
  PAIRS_PER_LEG=75; STRATA=4; LIMIT_PAIRS=0
  K_SAMPLES=5; K_BASELINE=5; MAX_NEW=1024
  NULL_DRAWS=200; NBOOT=1000
  MAX_WC_CHUNKS=0
  BUILD_EXTRA=()
  PILOT_BUDGET=2.6
fi
GEN_BATCH=16
# Patch round: EVERY wildchat chunk stages in BOTH modes — the reused pair
# manifest's wc a_id/b_id contexts must resolve regardless of the pair subset
# (a capped chunk set would orphan pairs drawn from later chunks).
if [[ "$ROUND" == "patch" ]]; then MAX_WC_CHUNKS=0; fi

HF_DL="$DATA_DIR/hf_dl"
RUN_ROOT="$OUT_ROOT/run"

CURRENT_PHASE="launch"
GATE_HALTED=0

# ── progress sentinels (committed writer — no heredocs; pod-side-reporting) ───
progress() {  # progress <gate> <msg>
  uv run python scripts/issue1776_swap.py progress --log-dir "$LOG_DIR" \
    --gate "$1" --msg "$2" --mode "$MODE" \
    || echo "[swap-dispatch] WARN: progress sentinel write failed (gate=$1)" >&2
}

phase_begin() {
  CURRENT_PHASE="$1"
  echo "[phase=$1]"
  echo "$1" >> "$TRACE"
  # RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking
  progress "${RTAG}-$1" "begin (mode=$MODE ngpu=$NGPU)" || true
}
# RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking
phase_end() { progress "${RTAG}-$1" "done" || true; }

gate_halt() {  # gate_halt <gate-name> <rc> <msg>
  GATE_HALTED=1
  echo "[swap-dispatch] ENGINEERING GATE HALT: $1 rc=$2 — $3" >&2
  # RC_CAPTURE_EXEMPT: gate sentinel is best-effort; the distinct exit rc is the signal
  progress "${RTAG}-gate-halt-$1" "rc=$2: $3" || true
  exit "$2"
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && $GATE_HALTED -eq 0 ]]; then
    echo "[swap-dispatch] FAILED at phase=$CURRENT_PHASE rc=$rc" >&2
    # RC_CAPTURE_EXEMPT: best-effort crash breadcrumb inside the EXIT trap
    progress "${RTAG}-crash" "phase=$CURRENT_PHASE rc=$rc" || true
  fi
}
trap on_exit EXIT

# ── runners (p3p4 shapes: tee to crash-persisted per-phase log) ───────────────
phase_dlog() { echo "$LOG_DIR/issue-${ISSUE}-${RTAG}-phase-${CURRENT_PHASE}.log"; }

run() {  # run <log-name> <cmd...>   (foreground, redirected; DRY: trace only)
  local plog="$PHASE_LOGS/$1.log"; shift
  if [[ $DRY == 1 ]]; then
    echo "DRY: $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    return 0
  fi
  echo "[swap-dispatch] run($CURRENT_PHASE): $*"
  "$@" 2>&1 | tee -a "$plog" >> "$(phase_dlog)"
  return "${PIPESTATUS[0]}"
}

bg_run() {  # bg_run <log-name> <cvd> <cmd...> -> sets BG_PID ('' in DRY)
  # NEVER capture via command substitution — BG_PID must be the payload pid
  # in THIS shell (parent-dispatcher lesson: cmd-subst pids fail wait rc=127).
  local plog="$PHASE_LOGS/$1.log" cvd="$2"; shift 2
  BG_PID=""
  if [[ $DRY == 1 ]]; then
    echo "DRY: CUDA_VISIBLE_DEVICES=$cvd $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    return 0
  fi
  echo "[swap-dispatch] bg($CURRENT_PHASE, CVD=$cvd): $*" >&2
  CUDA_VISIBLE_DEVICES="$cvd" "$@" > >(tee -a "$plog" >> "$(phase_dlog)") 2>&1 &
  BG_PID=$!
}

wait_rc() {  # wait_rc <pid-or-empty> -> the pid's rc (0 for DRY '')
  local p="$1" rc=0
  [[ -z "$p" ]] && return 0
  wait "$p" || rc=$?
  return $rc
}

upload_batch() {  # upload_batch <log-name> <hub-prefix> <commit-msg> <listfile>
  local lname="$1" prefix="$2" msg="$3" listfile="$4"
  run "$lname" uv run python scripts/issue1776_upload_batch.py \
    --prefix "$prefix" --listfile "$listfile" --message "$msg"
}

list_add() {  # include iff a regular FILE exists (glob loops only)
  [[ -f "$3" ]] && echo "$2=$3" >> "$1" || true
}

list_add_required() {  # fail LOUD when an expected-present artifact is missing
  if [[ $DRY == 1 ]]; then list_add "$@"; return 0; fi
  if [[ ! -f "$3" ]]; then
    echo "[swap-dispatch] upload-list FATAL: required artifact missing/not a FILE: $3 (rel=$2)" >&2
    exit 1
  fi
  echo "$2=$3" >> "$1"
}

# ── p0_stage ──────────────────────────────────────────────────────────────────
phase_begin "p0_stage"
# Chained smoke-then-full residue reap (#1586 fu r3): the FULL leg owns the
# deletion of the DERIVED smoke-leg roots, at first phase entry, fail-loud.
if [[ "$MODE" == "full" && $DRY == 0 ]]; then
  for stale in "${BASE_OUT}_smoke" "$DATA_DIR/${RTAG}_build_smoke"; do
    if [[ -d "$stale" ]]; then
      rm -r "$stale"
      echo "[swap-dispatch] reaped sibling smoke root: $stale"
    else
      echo "[swap-dispatch] sibling smoke root absent: $stale"
    fi
  done
fi
if [[ $DRY == 0 ]]; then
  NEED_GB=$([[ "$MODE" == "smoke" ]] && echo 10 || echo 30)
  run headroom uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_ROOT', $NEED_GB, phase='launch')
print(f'[headroom] out_root=$OUT_ROOT free_gb={free:.1f} floor=$NEED_GB')
"
fi
STAGE_ARGS=(--round "$ROUND" --report "$DATA_DIR/${RTAG}_stage_report.json"
  --pin-file "$DATA_DIR/data_repo_pin_${RTAG}.json")
if [[ "$MAX_WC_CHUNKS" != "0" ]]; then STAGE_ARGS+=(--max-wc-chunks "$MAX_WC_CHUNKS"); fi
run p0_stage uv run python scripts/issue1776_swap.py stage \
  --dest "$HF_DL" ${STAGE_ARGS[@]+"${STAGE_ARGS[@]}"}
phase_end "p0_stage"

# ── p1_build: build gates (1 GPU for the capture pass, both rounds) ──────────
phase_begin "p1_build"
if [[ "$ROUND" == "patch" ]]; then
  BUILD_CMD=(uv run python scripts/issue1776_swap.py build --round patch \
    --dest "$HF_DL" --stage-report "$DATA_DIR/${RTAG}_stage_report.json" \
    --out-dir "$BUILD_DIR" --swap-artifacts "$SWAP_ART" \
    --swap-build-report "$SWAP_BUILD_REPORT" --limit-pairs "$LIMIT_PAIRS")
else
  BUILD_CMD=(uv run python scripts/issue1776_swap.py build \
    --dest "$HF_DL" --stage-report "$DATA_DIR/${RTAG}_stage_report.json" \
    --out-dir "$BUILD_DIR" --pairs-per-leg "$PAIRS_PER_LEG" --strata "$STRATA")
fi
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p1_build env CUDA_VISIBLE_DEVICES=0 "${BUILD_CMD[@]}" \
  ${BUILD_EXTRA[@]+"${BUILD_EXTRA[@]}"} || rc=$?
if [[ $rc -eq 9 && "$ROUND" == "patch" ]]; then
  gate_halt "G-CAPTURE-CONSISTENCY" 9 "stored-vs-recomputed cx_last(14) cos median < 0.99 (report: $BUILD_DIR/patch_build_report.json)"
elif [[ $rc -eq 9 ]]; then
  gate_halt "G-METRIC-SANITY" 9 "ceiling recall@50 < 0.8 (report: $BUILD_DIR/build_report.json)"
elif [[ $rc -eq 7 ]]; then
  gate_halt "G-DOSE-DEGENERATE" 7 "both operators' median claimed fraction < 1% (report written)"
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p1_build"

# ── p2_pilot: parity probe (rc=8, binds at smoke) + measured wall (rc=7) ─────
phase_begin "p2_pilot"
if [[ "$ROUND" == "patch" ]]; then
  PILOT_EDIT_ARGS=(--patch-vectors "$BUILD_DIR/patch_vectors.pt"
    --pool "$SWAP_ART/pool.pt" --targets "$SWAP_ART/targets.json")
  PARITY_GATE="G-PATCH-PARITY"
else
  PILOT_EDIT_ARGS=(--deltas "$BUILD_DIR/deltas.pt"
    --pool "$BUILD_DIR/pool.pt" --targets "$BUILD_DIR/targets.json")
  PARITY_GATE="G-SWAP-PARITY"
fi
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p2_pilot env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_swap.py pilot \
  --round "$ROUND" --pairs "$BUILD_DIR/pairs.jsonl" "${PILOT_EDIT_ARGS[@]}" \
  --k-samples "$K_SAMPLES" --max-new-tokens "$MAX_NEW" --gen-batch "$GEN_BATCH" \
  --null-draws "$NULL_DRAWS" --budget-gpu-h "$PILOT_BUDGET" --ngpu "$NGPU" \
  --out "$BUILD_DIR/pilot_report.json" || rc=$?
if [[ $rc -eq 8 ]]; then
  gate_halt "$PARITY_GATE" 8 "per-row prefill edit parity failed (report: $BUILD_DIR/pilot_report.json)"
elif [[ $rc -eq 7 ]]; then
  gate_halt "G-PILOT" 7 "pilot projection >2x budget (report: $BUILD_DIR/pilot_report.json)"
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p2_pilot"

# ── p3 gen: generation units sharded across all GPUs ─────────────────────────
if [[ "$ROUND" == "patch" ]]; then
  EDIT_STORE="$BUILD_DIR/patch_vectors.pt"
else
  EDIT_STORE="$BUILD_DIR/deltas.pt"
fi
phase_begin "$GENPHASE"
mkdir -p "$RUN_ROOT"
GEN_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p3_gen_shard$g" "$g" uv run python scripts/issue1776_swap.py run \
    --round "$ROUND" --phase gen --pairs "$BUILD_DIR/pairs.jsonl" --deltas "$EDIT_STORE" \
    --out-root "$RUN_ROOT" --shard-index "$g" --num-shards "$NGPU" \
    --k-samples "$K_SAMPLES" --k-baseline "$K_BASELINE" --max-new-tokens "$MAX_NEW" \
    --gen-batch "$GEN_BATCH"
  p="$BG_PID"
  GEN_PIDS+=("$p")
done
for p in ${GEN_PIDS[@]+"${GEN_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
phase_end "$GENPHASE"

# ── p3_text: merge + rollout TEXT upload BEFORE any capture/reduce (#779) ────
phase_begin "p3_text"
run p3_merge uv run python scripts/issue1776_swap.py merge-text \
  --round "$ROUND" --out-root "$RUN_ROOT" --eval-out "$EVAL_DIR" --hf-prefix "$HF_PREFIX_EFF"
TEXT_LIST="$OUT_ROOT/upload_text.list"; : > "$TEXT_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$RUN_ROOT"/raw_completions/"$MERGED_SUBDIR"/*.json; do
    list_add "$TEXT_LIST" "$(basename "$f")" "$f"
  done
fi
list_add_required "$TEXT_LIST" "raw_completions_manifest.json" \
  "$EVAL_DIR/raw_completions_manifest.json"
upload_batch p3_text_upload "$HF_PREFIX_EFF/raw_completions/$MERGED_SUBDIR" \
  "task #1776 followup $RTAG: rollout text (before reduce/judge)" "$TEXT_LIST"
phase_end "p3_text"

# ── p4_capture: teacher-forced v19, sharded; then tensors upload ─────────────
phase_begin "p4_capture"
CAP_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p4_cap_shard$g" "$g" uv run python scripts/issue1776_swap.py run \
    --round "$ROUND" --phase capture --pairs "$BUILD_DIR/pairs.jsonl" --deltas "$EDIT_STORE" \
    --out-root "$RUN_ROOT" --shard-index "$g" --num-shards "$NGPU" \
    --k-samples "$K_SAMPLES" --k-baseline "$K_BASELINE" --max-new-tokens "$MAX_NEW" \
    --gen-batch "$GEN_BATCH"
  p="$BG_PID"
  CAP_PIDS+=("$p")
done
for p in ${CAP_PIDS[@]+"${CAP_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
TENS_LIST="$OUT_ROOT/upload_tens.list"; : > "$TENS_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$RUN_ROOT"/cells/*.pt; do
    list_add "$TENS_LIST" "cells/$(basename "$f")" "$f"
  done
fi
if [[ "$ROUND" == "patch" ]]; then
  # parent deltas/pool/targets already live on the Hub (followup_swap) — only
  # this round's OWN artifacts upload (plan §10 followup_slotpatch outputs).
  list_add_required "$TENS_LIST" "patch_vectors.pt" "$BUILD_DIR/patch_vectors.pt"
  list_add_required "$TENS_LIST" "pairs.jsonl" "$BUILD_DIR/pairs.jsonl"
  list_add_required "$TENS_LIST" "patch_build_report.json" "$BUILD_DIR/patch_build_report.json"
else
  list_add_required "$TENS_LIST" "deltas.pt" "$BUILD_DIR/deltas.pt"
  list_add_required "$TENS_LIST" "pool.pt" "$BUILD_DIR/pool.pt"
  list_add_required "$TENS_LIST" "pairs.jsonl" "$BUILD_DIR/pairs.jsonl"
  list_add_required "$TENS_LIST" "targets.json" "$BUILD_DIR/targets.json"
  list_add_required "$TENS_LIST" "build_report.json" "$BUILD_DIR/build_report.json"
fi
list_add_required "$TENS_LIST" "pilot_report.json" "$BUILD_DIR/pilot_report.json"
list_add_required "$TENS_LIST" "run_manifest.json" "$RUN_ROOT/manifest.json"
upload_batch p4_tens_upload "$HF_PREFIX_EFF/analysis_tensors/$FUD" \
  "task #1776 followup $RTAG: per-pair edit stores + per-cell v19" "$TENS_LIST"
phase_end "p4_capture"

# ── p5_reduce: metric + nulls + verdict + figures (CPU) ──────────────────────
phase_begin "p5_reduce"
if [[ "$ROUND" == "patch" ]]; then
  ANALYZE_CMD=(uv run python scripts/issue1776_swap.py analyze --round patch \
    --pairs "$BUILD_DIR/pairs.jsonl" --targets "$SWAP_ART/targets.json" \
    --pool "$SWAP_ART/pool.pt" --deltas "$SWAP_ART/deltas.pt" \
    --patch-vectors "$BUILD_DIR/patch_vectors.pt" \
    --build-report "$SWAP_BUILD_REPORT" \
    --patch-build-report "$BUILD_DIR/patch_build_report.json" \
    --swap-success "$SWAP_SUCCESS" --run-root "$RUN_ROOT" \
    --null-draws "$NULL_DRAWS" --n-boot "$NBOOT" \
    --out-dir "$EVAL_DIR" --fig-dir "$FIG_DIR")
else
  ANALYZE_CMD=(uv run python scripts/issue1776_swap.py analyze \
    --pairs "$BUILD_DIR/pairs.jsonl" --targets "$BUILD_DIR/targets.json" \
    --pool "$BUILD_DIR/pool.pt" --deltas "$BUILD_DIR/deltas.pt" \
    --build-report "$BUILD_DIR/build_report.json" --run-root "$RUN_ROOT" \
    --null-draws "$NULL_DRAWS" --n-boot "$NBOOT" \
    --out-dir "$EVAL_DIR" --fig-dir "$FIG_DIR")
fi
run p5_reduce "${ANALYZE_CMD[@]}"
phase_end "p5_reduce"

# ── p6_commit: git-destined deliverables (#1205 + #1325) + final sentinel ────
phase_begin "p6_commit"
if [[ $DRY == 1 || "$MODE" == "smoke" ]]; then
  run p6_commit echo "SKIPPED (dry-run/smoke: outputs stay under $EVAL_DIR)"
else
  if [[ "$ROUND" == "patch" ]]; then
    # targets.json is the PARENT's (already committed under followup_swap)
    cp "$BUILD_DIR/patch_build_report.json" "$BUILD_DIR/pilot_report.json" "$EVAL_DIR/"
  else
    cp "$BUILD_DIR/build_report.json" "$BUILD_DIR/pilot_report.json" \
      "$BUILD_DIR/targets.json" "$EVAL_DIR/"
  fi
  BR="$(git rev-parse --abbrev-ref HEAD)"
  mapfile -t DECLARED < <(find "$EVAL_DIR" -name '*.json' | sort)
  mapfile -t FIGS < <(find "$FIG_DIR" -name '*.png' 2>/dev/null | sort)
  git add -- "${DECLARED[@]}" ${FIGS[@]+"${FIGS[@]}"}
  if ! git diff --cached --quiet; then
    git commit -m "task #1776 followup $RTAG: $ROUND-round results (mode=$MODE)"
  fi
  if ! git push origin "$BR"; then
    echo "[swap-dispatch] push failed; retrying once in 20s" >&2
    sleep 20
    git push origin "$BR"
  fi
  AHEAD="$(git rev-list --count "origin/$BR..HEAD")"
  [[ "$AHEAD" == "0" ]] || { echo "[swap-dispatch] push-verify FAIL: $AHEAD unpushed" >&2; exit 1; }
  MISSING=0
  for f in "${DECLARED[@]}" ${FIGS[@]+"${FIGS[@]}"}; do
    rel="${f#"$REPO_ROOT"/}"
    if [[ -z "$(git ls-tree -r "origin/$BR" --name-only -- "$rel")" ]]; then
      echo "[swap-dispatch] artifact-presence FAIL: $rel not in pushed tree" >&2
      MISSING=1
    fi
  done
  [[ $MISSING -eq 0 ]] || exit 1
  echo "[swap-dispatch] results pushed + verified (${#DECLARED[@]} JSONs, ${#FIGS[@]} figures)"
fi
FINAL_ARGS=(--round "$ROUND" --log-dir "$LOG_DIR" --mode "$MODE" --eval-dir "$EVAL_DIR"
  --repo-root "$REPO_ROOT" --hf-prefix "$HF_PREFIX_EFF" --ngpu "$NGPU")
if [[ $DRY == 1 ]]; then FINAL_ARGS+=(--dry); fi
uv run python scripts/issue1776_swap.py final-sentinel "${FINAL_ARGS[@]}"
phase_end "p6_commit"
echo "[phase=done]"
exit 0
