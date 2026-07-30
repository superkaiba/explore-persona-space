#!/usr/bin/env bash
# #1776 follow-up p3p4_heterogeneity_dose dispatcher (SEPARATE from the done +
# fragile issue1776_dispatch.sh — same runner conventions, fresh phase chain).
#
# LEG P3 — per-context Jacobians (issue1776_p3p4.py stage/build/pilot/run/analyze):
#   p0_stage_fu     stage all reused inputs at ONE fresh Hub pin
#   p1_build_fu     per-context M' errors -> stratified sample + P4 alpha ladder
#   p2_pilot_fu     measured 1-context wall; G-PILOT-PCJ rc=7 over 2x budget
#   p3_pcj          per-context backward sweep, context-sharded across all GPUs
#   p3_pcj_upload   per-context tensors + reports -> HF analysis_tensors/followup_p3p4
#   p4_hetero       heterogeneity reads -> jacobian_heterogeneity.json + figure
# LEG P4 — dose-escalated matched-slot steering (issue1776_phase3.py REUSED at
#   raw-norm --alphas from the ladder; baseline + strata shards + finalize):
#   p5_dose_grid    alpha=0 baseline -> steered strata (GPU fan-out) -> finalize
#   p5_dose_upload  rollout TEXT first (raw_completions/steered_dose), then
#                   summaries + judge-ready manifest (judging stays OFF-POD)
#   p_results_commit_fu  eval JSONs + figure -> issue branch (#1205/#1325)
#   p_final         terminal results sentinel, then [phase=done]
#
# Exit codes: 0 ok; 7 = G-PILOT-PCJ (pilot projection >2x budget; report JSON
# written); 8 = G-NONZERO (degenerate slot convention). Per-phase logs tee to
# BOTH $PHASE_LOGS and $LOG_DIR (crash-persisted). Smoke/dry legs never touch
# committed eval_results/ or canonical Hub prefixes (scratch redirects), and
# the FULL leg reaps the derived smoke-leg roots at entry (#1586 fu r3 class).

set -euo pipefail

# ── args / mode ───────────────────────────────────────────────────────────────
MODE="full"
DRY=0
NGPU_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --gpus) NGPU_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "full" || "$MODE" == "smoke" ]] || { echo "--mode full|smoke" >&2; exit 2; }
if [[ "${EPS_1776_DRY_RUN:-0}" == "1" ]]; then DRY=1; fi
export EPS_1776_MODE="$MODE"

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
ISSUE=1776

BASE_OUT="${EPS_1776_P3P4_OUT_ROOT:-/workspace/issue_1776_p3p4}"
if [[ $DRY == 1 ]]; then
  TMP_BASE="${EPS_1776_TMP:-$(mktemp -d /tmp/issue-1776-p3p4-dryrun.XXXXXX)}"
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
  EVAL_DIR="$OUT_ROOT/eval_results/issue_1776/followup_p3p4"
  FIG_DIR="$OUT_ROOT/figures/issue_1776/followup_p3p4"
  HF_PREFIX_EFF="issue1776_jacobian/smoke_probe"
  PCJ_DIR="$DATA_DIR/pcj_smoke"
else
  EVAL_DIR="$REPO_ROOT/eval_results/issue_1776/followup_p3p4"
  FIG_DIR="$REPO_ROOT/figures/issue_1776/followup_p3p4"
  HF_PREFIX_EFF="issue1776_jacobian"
  PCJ_DIR="$DATA_DIR/pcj"
fi
mkdir -p "$EVAL_DIR"

# ── realized width (workload re-shards off realized width) ────────────────────
if [[ -n "$NGPU_OVERRIDE" ]]; then
  NGPU="$NGPU_OVERRIDE"
elif [[ $DRY == 1 ]]; then
  NGPU="${EPS_1776_NGPU:-8}"
else
  NGPU="$(nvidia-smi --list-gpus | wc -l)"
fi
[[ "$NGPU" -ge 1 ]] || { echo "[p3p4-dispatch] no GPUs visible" >&2; exit 1; }

# ── mode parameter table ──────────────────────────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
  N_SKETCH=4;  N_FULL=2;  STRATA=1        # >=2 contexts/corpus (neighbor floor)
  MAX_WC_CHUNKS=1                          # 1 chunk (125 rows) covers the smoke sample
  NBOOT=50
  PILOT_BUDGET=3.5                         # same gate arithmetic; tiny rows => ratio ~0
  P4_EXTRA=(--limit-contexts 2 --k-samples 1 --k-baseline 1)
else
  N_SKETCH=96; N_FULL=16; STRATA=4
  MAX_WC_CHUNKS=0
  NBOOT=1000
  PILOT_BUDGET=3.5
  # plan-§9 descope floor keeps 150 contexts; all_positions stays 0 (P4 scope
  # is the prefill matched-slot edit — the exploratory arm already ran in P3).
  P4_EXTRA=(--limit-contexts 150)
fi

# ── shared staged paths (issue1776_p3p4.py stage mirrors repo-relative) ───────
HF_DL="$DATA_DIR/hf_dl"
JLAST="$HF_DL/issue1776_jacobian/analysis_tensors/jac_full/J_last.pt"
JAVG_DIR="$HF_DL/issue1776_jacobian/analysis_tensors/jac_full"
COMP="$HF_DL/issue1776_jacobian/analysis_tensors/comparator/m_ridge_x50k.pt"
SEEDS="$HF_DL/issue1776_jacobian/analysis_tensors/jac_sketch/seeds.pt"
CTX_JSONL="$HF_DL/issue1776_jacobian/analysis_tensors/contexts/contexts.jsonl"
RB_DIR="$HF_DL/issue779_monitoring/r_b"
P4_ROOT="$OUT_ROOT/dose"
PCJ_OUT="$OUT_ROOT/pcj_run"

CURRENT_PHASE="launch"
GATE_HALTED=0

# ── progress sentinels (committed writer — no heredocs; pod-side-reporting) ───
progress() {  # progress <gate> <msg>
  uv run python scripts/issue1776_p3p4.py progress --log-dir "$LOG_DIR" \
    --gate "$1" --msg "$2" --mode "$MODE" \
    || echo "[p3p4-dispatch] WARN: progress sentinel write failed (gate=$1)" >&2
}

phase_begin() {
  CURRENT_PHASE="$1"
  echo "[phase=$1]"
  echo "$1" >> "$TRACE"
  # RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking
  progress "phase-$1" "begin (mode=$MODE ngpu=$NGPU)" || true
}
# RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking
phase_end() { progress "phase-$1" "done" || true; }

gate_halt() {  # gate_halt <gate-name> <rc> <msg>
  GATE_HALTED=1
  echo "[p3p4-dispatch] ENGINEERING GATE HALT: $1 rc=$2 — $3" >&2
  # RC_CAPTURE_EXEMPT: gate sentinel is best-effort; the distinct exit rc is the signal
  progress "phase-gate-halt-$1" "rc=$2: $3" || true
  exit "$2"
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && $GATE_HALTED -eq 0 ]]; then
    echo "[p3p4-dispatch] FAILED at phase=$CURRENT_PHASE rc=$rc" >&2
    # RC_CAPTURE_EXEMPT: best-effort crash breadcrumb inside the EXIT trap
    progress "phase-crash" "phase=$CURRENT_PHASE rc=$rc" || true
  fi
}
trap on_exit EXIT

# ── runners (parent-dispatcher shapes: tee to crash-persisted per-phase log) ──
phase_dlog() { echo "$LOG_DIR/issue-${ISSUE}-p3p4-phase-${CURRENT_PHASE}.log"; }

run() {  # run <log-name> <cmd...>   (foreground, redirected; DRY: trace only)
  local plog="$PHASE_LOGS/$1.log"; shift
  if [[ $DRY == 1 ]]; then
    echo "DRY: $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    return 0
  fi
  echo "[p3p4-dispatch] run($CURRENT_PHASE): $*"
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
  echo "[p3p4-dispatch] bg($CURRENT_PHASE, CVD=$cvd): $*" >&2
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
    echo "[p3p4-dispatch] upload-list FATAL: required artifact missing/not a FILE: $3 (rel=$2)" >&2
    exit 1
  fi
  echo "$2=$3" >> "$1"
}

# ── p0_stage_fu ───────────────────────────────────────────────────────────────
phase_begin "p0_stage_fu"
# Chained smoke-then-full residue reap (#1586 fu r3): the FULL leg owns the
# deletion of the DERIVED smoke-leg roots, at first phase entry, fail-loud.
if [[ "$MODE" == "full" && $DRY == 0 ]]; then
  for stale in "${BASE_OUT}_smoke" "$DATA_DIR/pcj_smoke"; do
    if [[ -d "$stale" ]]; then
      rm -r "$stale"
      echo "[p3p4-dispatch] reaped sibling smoke root: $stale"
    else
      echo "[p3p4-dispatch] sibling smoke root absent: $stale"
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
STAGE_ARGS=(--report "$DATA_DIR/p3p4_stage_report.json")
if [[ "$MAX_WC_CHUNKS" != "0" ]]; then STAGE_ARGS+=(--max-wc-chunks "$MAX_WC_CHUNKS"); fi
run p0_stage_fu uv run python scripts/issue1776_p3p4.py stage \
  --dest "$HF_DL" ${STAGE_ARGS[@]+"${STAGE_ARGS[@]}"}
phase_end "p0_stage_fu"

# ── p1_build_fu (CPU): stratified sample + targets + P4 alpha ladder ──────────
phase_begin "p1_build_fu"
run p1_build_fu uv run python scripts/issue1776_p3p4.py build \
  --dest "$HF_DL" --comparator "$COMP" --out-dir "$PCJ_DIR" \
  --n-sketch "$N_SKETCH" --n-full "$N_FULL" --strata "$STRATA"
if [[ $DRY == 0 ]]; then
  LADDER_CSV="$(uv run python -c "
import json; print(json.load(open('$PCJ_DIR/p4_alpha_ladder.json'))['alphas_csv'])")"
else
  LADDER_CSV="8,16,32"
fi
echo "[p3p4-dispatch] P4 raw-norm alpha ladder: $LADDER_CSV"
phase_end "p1_build_fu"

# ── p2_pilot_fu: measured 1-context wall (G-PILOT-PCJ, designed halt rc=7) ────
phase_begin "p2_pilot_fu"
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p2_pilot_fu env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_p3p4.py pilot \
  --pairs "$PCJ_DIR/pcj_pairs.jsonl" --seeds-file "$SEEDS" \
  --budget-gpu-h "$PILOT_BUDGET" --ngpu "$NGPU" \
  --out "$PCJ_DIR/pilot_report.json" || rc=$?
if [[ $rc -eq 7 ]]; then
  gate_halt "G-PILOT-PCJ" 7 "pilot projection >2x budget (report: $PCJ_DIR/pilot_report.json)"
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p2_pilot_fu"

# ── p3_pcj: per-context Jacobians, context-sharded across all GPUs ────────────
phase_begin "p3_pcj"
mkdir -p "$PCJ_OUT"
PCJ_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p3_pcj_shard$g" "$g" uv run python scripts/issue1776_p3p4.py run \
    --pairs "$PCJ_DIR/pcj_pairs.jsonl" --seeds-file "$SEEDS" \
    --out-dir "$PCJ_OUT" --shard-index "$g" --num-shards "$NGPU"
  p="$BG_PID"
  PCJ_PIDS+=("$p")
done
PCJ_RC=0
for p in ${PCJ_PIDS[@]+"${PCJ_PIDS[@]}"}; do
  wait_rc "$p" || { rc=$?; if [[ $rc -eq 8 ]]; then PCJ_RC=8; else exit $rc; fi; }
done
if [[ $PCJ_RC -eq 8 ]]; then
  gate_halt "G-NONZERO" 8 "all-zero per-context gradient (slot-convention bug)"
fi
phase_end "p3_pcj"

# ── p3_pcj_upload: per-context tensors + inputs BEFORE analysis (#825 order) ──
phase_begin "p3_pcj_upload"
PCJ_LIST="$OUT_ROOT/upload_pcj.list"; : > "$PCJ_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$PCJ_OUT"/pcj/*.pt; do
    list_add "$PCJ_LIST" "pcj/$(basename "$f")" "$f"
  done
  for f in "$PCJ_OUT"/shard*_report.json "$PCJ_OUT"/manifest.json; do
    list_add "$PCJ_LIST" "$(basename "$f")" "$f"
  done
fi
list_add_required "$PCJ_LIST" "pcj_pairs.jsonl" "$PCJ_DIR/pcj_pairs.jsonl"
list_add_required "$PCJ_LIST" "pcj_targets.pt" "$PCJ_DIR/pcj_targets.pt"
list_add_required "$PCJ_LIST" "pcj_build_report.json" "$PCJ_DIR/pcj_build_report.json"
list_add_required "$PCJ_LIST" "p4_alpha_ladder.json" "$PCJ_DIR/p4_alpha_ladder.json"
list_add_required "$PCJ_LIST" "pilot_report.json" "$PCJ_DIR/pilot_report.json"
upload_batch p3_pcj_upload "$HF_PREFIX_EFF/analysis_tensors/followup_p3p4" \
  "task #1776 followup p3p4: per-context Jacobians + sample + ladder" "$PCJ_LIST"
phase_end "p3_pcj_upload"

# ── p4_hetero: heterogeneity reads -> deliverable JSON + figure (CPU) ─────────
phase_begin "p4_hetero"
run p4_hetero uv run python scripts/issue1776_p3p4.py analyze \
  --pcj-dir "$PCJ_OUT" --targets "$PCJ_DIR/pcj_targets.pt" --seeds-file "$SEEDS" \
  --javg-dir "$JAVG_DIR" --comparator "$COMP" --rb-dir "$RB_DIR" \
  --n-boot "$NBOOT" --out "$EVAL_DIR/jacobian_heterogeneity.json" --fig-dir "$FIG_DIR"
phase_end "p4_hetero"

# ── p5_dose_grid: phase-3 rerun at the raw-norm ladder (REUSED machinery) ─────
phase_begin "p5_dose_grid"
mkdir -p "$P4_ROOT"
P4_BASE=(uv run python scripts/issue1776_phase3.py --mode run
  --contexts "$CTX_JSONL" --rb-dir "$RB_DIR" --mprime-weights "$COMP"
  --jlast "$JLAST" --out-root "$P4_ROOT" --alphas "$LADDER_CSV")
run p5_dose_baseline env CUDA_VISIBLE_DEVICES=0 "${P4_BASE[@]}" --baseline-only \
  ${P4_EXTRA[@]+"${P4_EXTRA[@]}"}
P4_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p5_dose_shard$g" "$g" "${P4_BASE[@]}" \
    --strata-shard "$g" --strata-num-shards "$NGPU" ${P4_EXTRA[@]+"${P4_EXTRA[@]}"}
  p="$BG_PID"
  P4_PIDS+=("$p")
done
for p in ${P4_PIDS[@]+"${P4_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
# --eval-out is the eval DIRECTORY (issue1776_phase3.py refuses a file path);
# finalize writes steered_shift_summaries.json + raw_completions_manifest.json
# INSIDE it — landing next to jacobian_heterogeneity.json in followup_p3p4/.
run p5_dose_finalize env CUDA_VISIBLE_DEVICES=0 "${P4_BASE[@]}" --finalize-only \
  --eval-out "$EVAL_DIR" ${P4_EXTRA[@]+"${P4_EXTRA[@]}"}
phase_end "p5_dose_grid"

# ── p5_dose_upload: rollout TEXT first, then summaries + manifest ─────────────
phase_begin "p5_dose_upload"
P4_TEXT_LIST="$OUT_ROOT/upload_p4_text.list"; : > "$P4_TEXT_LIST"
P4_TENS_LIST="$OUT_ROOT/upload_p4_tens.list"; : > "$P4_TENS_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$P4_ROOT"/raw_completions/steered/*.json; do
    list_add "$P4_TEXT_LIST" "$(basename "$f")" "$f"
  done
  for f in "$P4_ROOT"/summaries/*.pt; do
    list_add "$P4_TENS_LIST" "dose_summaries/$(basename "$f")" "$f"
  done
fi
list_add_required "$P4_TENS_LIST" "raw_completions_manifest.json" \
  "$EVAL_DIR/raw_completions_manifest.json"
list_add_required "$P4_TENS_LIST" "steered_shift_summaries.json" \
  "$EVAL_DIR/steered_shift_summaries.json"
upload_batch p5_dose_upload_text "$HF_PREFIX_EFF/raw_completions/steered_dose" \
  "task #1776 followup p3p4: dose-ladder rollout text (before reduce/judge)" "$P4_TEXT_LIST"
upload_batch p5_dose_upload_tens "$HF_PREFIX_EFF/analysis_tensors/followup_p3p4" \
  "task #1776 followup p3p4: dose per-sample summaries + judge manifest" "$P4_TENS_LIST"
phase_end "p5_dose_upload"

# ── p_results_commit_fu: git-destined deliverables (#1205 + #1325) ────────────
phase_begin "p_results_commit_fu"
if [[ $DRY == 1 || "$MODE" == "smoke" ]]; then
  run p_results_commit_fu echo "SKIPPED (dry-run/smoke: outputs stay under $EVAL_DIR)"
else
  cp "$PCJ_DIR/pcj_build_report.json" "$PCJ_DIR/pilot_report.json" \
    "$PCJ_DIR/p4_alpha_ladder.json" "$EVAL_DIR/"
  BR="$(git rev-parse --abbrev-ref HEAD)"
  mapfile -t DECLARED < <(find "$EVAL_DIR" -name '*.json' | sort)
  mapfile -t FIGS < <(find "$FIG_DIR" -name '*.png' 2>/dev/null | sort)
  git add -- "${DECLARED[@]}" ${FIGS[@]+"${FIGS[@]}"}
  if ! git diff --cached --quiet; then
    git commit -m "task #1776 followup p3p4: heterogeneity + dose-ladder results (mode=$MODE)"
  fi
  if ! git push origin "$BR"; then
    echo "[p3p4-dispatch] push failed; retrying once in 20s" >&2
    sleep 20
    git push origin "$BR"
  fi
  AHEAD="$(git rev-list --count "origin/$BR..HEAD")"
  [[ "$AHEAD" == "0" ]] || { echo "[p3p4-dispatch] push-verify FAIL: $AHEAD unpushed" >&2; exit 1; }
  MISSING=0
  for f in "${DECLARED[@]}" ${FIGS[@]+"${FIGS[@]}"}; do
    rel="${f#"$REPO_ROOT"/}"
    if [[ -z "$(git ls-tree -r "origin/$BR" --name-only -- "$rel")" ]]; then
      echo "[p3p4-dispatch] artifact-presence FAIL: $rel not in pushed tree" >&2
      MISSING=1
    fi
  done
  [[ $MISSING -eq 0 ]] || exit 1
  echo "[p3p4-dispatch] results pushed + verified (${#DECLARED[@]} JSONs, ${#FIGS[@]} figures)"
fi
phase_end "p_results_commit_fu"

# ── p_final: terminal results sentinel, then [phase=done] ─────────────────────
phase_begin "p_final"
FINAL_ARGS=(--log-dir "$LOG_DIR" --mode "$MODE" --eval-dir "$EVAL_DIR"
  --repo-root "$REPO_ROOT" --hf-prefix "$HF_PREFIX_EFF" --ngpu "$NGPU")
if [[ $DRY == 1 ]]; then FINAL_ARGS+=(--dry); fi
uv run python scripts/issue1776_p3p4.py final-sentinel "${FINAL_ARGS[@]}"
phase_end "p_final"
echo "[phase=done]"
exit 0
