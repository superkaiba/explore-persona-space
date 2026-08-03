#!/usr/bin/env bash
# #1776 run-node orchestration: sequences the phase scripts per plan §9 (v4).
#
# Workload command (plan §10):
#   REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" bash scripts/issue1776_dispatch.sh --mode full
#   (smoke: --mode smoke — SAME chain / SAME width / real model, tiny slices)
#
# Phase sequence (== EXPECTED_ORDER below; the dry-run asserts the trace):
#   p0_stage             stage.py (pin + probes + provenance + pass_b/weights/r_b/
#                        manifest/parity chunks) + #779 trait artifacts + centroids
#   p5a_stream_launch    phase5 stream (CPU/network) launched BG — §9: concurrent
#   p0_parity            G-PARITY gate (rc=8 -> HALT: every downstream number
#                        consumes the same captured fields)
#   p0_comparator_launch m_ridge_x50k + m_ridge_lmsys50k BG on the last GPU
#                        (§9: P0.5 runs concurrent with P0.1's tail)
#   p0_jlens             build-prompts; fit fan-out (GPUs 0..N-2); merge; G-LENS
#                        sanity (rc=8 -> lens-dependent legs skipped, final rc=8)
#   p0_comparator_join   join the comparator fits
#   p0_dict              phase4 build-dict L14/L19/L21 (subprocess — its main()
#                        ends in sys.exit; NEVER import-call it in-process)
#   p04_pairs            J-pair manifest (1,536 LMSYS train-pool pairs; text from
#                        the #779 raw_completions chunks at the pin) + sharded
#                        teacher-forced capture (parity's recompute_row rig) ->
#                        v_pool / acts14 / acts19
#   p1_contexts          contexts builder (needs the staged #779 trait artifacts)
#   p2a_sketch           build-seeds; jacobian run --mode sketch fan-out; merge
#                        (G-NONZERO fires inside run: rc=8 -> HALT)
#   p1_diag              phase1 directional diagnostic
#   p_early_upload       §9 expensive-store-before-long-fit (#825): everything
#                        already produced uploads BEFORE the long P2b sweep
#   p2b_full             jacobian run --mode full (3,584 seeds, seed-block shard
#                        fan-out); merge -> J_{prefix,ctx,last} + even/odd halves
#   p2_upload            J tensors + halves upload (before P2c reads / P4)
#   p3_grid              phase3 baseline-only -> steered strata fan-out -> finalize
#   p3_upload            steered rollout TEXT -> raw_completions/steered (before
#                        phase4 + judging), summaries/manifest -> analysis_tensors
#   p4_mediation         phase4 energy / refit-split / jdelta-split (lens-gated)
#   p5a_stream_join      join the WildChat stream
#   p5a_capture          phase5 capture fan-out (self-uploads batched to HF)
#   p5_transfer          phase5 transfer --assemble (test-1000 leg + anchors) +
#                        ops (M' x50k/lmsys50k, shipped-M reference if resolvable)
#                        + J arms over lmsys_test1000 + wildchat_fresh (P2c + P5)
#   p5b_leakage          phase5 leakage re-read (CPU; inputs all local by now —
#                        §9 lists this off-pod (p7); POD-side is the default here
#                        since the dispatcher stages centroids + builds the L21
#                        dict anyway; EPS_1776_P5B_OFFPOD=1 skips it for the
#                        plan-literal off-pod lane (invocation documented in the
#                        final sentinel offpod_handoffs.p5b_leakage); deviation +
#                        rationale recorded in the final sentinel)
#   p_results_commit     git add/commit/push eval JSONs + rev-list push-verify +
#                        per-file ls-tree artifact-presence assert (#1205/#1325)
#   p_final              epm:results (or epm:smoke-result) sentinel, then
#                        [phase=done]
#
# OFF-POD (excluded here; named in the final sentinel note):
#   p6 graded judge  — Batch API on the VM after release (issue1776_judge.py).
#     Pricing note: the judge DEFAULT is the plan-§9-priced contrast policy
#     (trait strata under own rubric; baseline under every trait rubric;
#     w1_mprime/random one rubric per context, round-robin — ~18k x 5 calls);
#     the all-rubrics mode (~30k x 5) is opt-in via --all-control-rubrics.
#   p7 final analyses — VM, 0 GPU (5c/5d lens reads etc.).
#
# Engineering-gate exit codes (plan §7): 8 = G-PARITY / G-LENS / G-NONZERO
# halt (gate-report sentinel written; NOT a crash); 7 = G-PILOT (phase scripts
# exit 7 themselves; routed through the same designed-halt path).
#
# Progress reporting (pod-side-reporting rule): [phase=...] lines on THIS
# script's stdout only (phase scripts are redirected to per-phase logs, so the
# reserved [phase=done] token never leaks); per-phase tick sentinels
# /workspace/logs/issue-1776-phase-*.json (kind epm:progress, gate name carries
# "phase" -> the drain posts them verbatim; write-once, never re-read — state
# lives under $OUT_ROOT); ONE terminal results sentinel before [phase=done].
#
# Dry-run (VM smoke): --dry-run (or EPS_1776_DRY_RUN=1) traces the phase
# sequence + writes REAL progress/final sentinels into a /tmp root (no GPU, no
# Hub, no model), then asserts the trace against EXPECTED_ORDER and round-trips
# every sentinel through the required-keys parse.

set -euo pipefail

# ── args / mode ───────────────────────────────────────────────────────────────
MODE="full"
DRY=0
NGPU_OVERRIDE=""
ALLOW_MISSING_REF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --gpus) NGPU_OVERRIDE="$2"; shift 2 ;;
    --allow-missing-reference-arm) ALLOW_MISSING_REF=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "full" || "$MODE" == "smoke" ]] || { echo "--mode full|smoke" >&2; exit 2; }
if [[ "${EPS_1776_DRY_RUN:-0}" == "1" ]]; then DRY=1; fi
export EPS_1776_MODE="$MODE"

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"  # plan §8: no concurrent .venv syncs on fan-out
ISSUE=1776

if [[ $DRY == 1 ]]; then
  TMP_BASE="${EPS_1776_TMP:-$(mktemp -d /tmp/issue-1776-dryrun.XXXXXX)}"
  OUT_ROOT="${EPS_1776_OUT_ROOT:-$TMP_BASE/out}"
  LOG_DIR="${EPS_1776_LOG_DIR:-$TMP_BASE/logs}"
else
  OUT_ROOT="${EPS_1776_OUT_ROOT:-/workspace/issue_1776}"
  LOG_DIR="${EPS_1776_LOG_DIR:-/workspace/logs}"
  if [[ -d /workspace ]]; then export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"; fi
fi
DATA_DIR="$REPO_ROOT/data/issue_1776"
PHASE_LOGS="$OUT_ROOT/logs"
TRACE="$OUT_ROOT/dispatch_trace.txt"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$PHASE_LOGS" "$DATA_DIR"
: > "$TRACE"

# Smoke outputs never touch committed eval_results/ (scratch-dir redirect);
# smoke Hub uploads go to the smoke_probe/ scratch prefix, never canonical.
if [[ "$MODE" == "smoke" || $DRY == 1 ]]; then
  EVAL_DIR="$OUT_ROOT/eval_results/issue_1776"
  HF_PREFIX_EFF="issue1776_jacobian/smoke_probe"
else
  EVAL_DIR="$REPO_ROOT/eval_results/issue_1776"
  HF_PREFIX_EFF="issue1776_jacobian"
fi
mkdir -p "$EVAL_DIR"

# ── realized width (§9: the workload re-shards off realized width) ────────────
if [[ -n "$NGPU_OVERRIDE" ]]; then
  NGPU="$NGPU_OVERRIDE"
elif [[ $DRY == 1 ]]; then
  NGPU="${EPS_1776_NGPU:-8}"
else
  NGPU="$(nvidia-smi --list-gpus | wc -l)"
fi
[[ "$NGPU" -ge 1 ]] || { echo "[dispatch] no GPUs visible" >&2; exit 1; }

# ── mode parameter table ──────────────────────────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
  PARITY_CHUNKS=1;  PARITY_ROWS=8
  JLENS_N=4;        JLENS_LIMIT=(--limit 4)
  N_PAIRS=8
  SEEDS_TOTAL=12;   SEEDS_TOPK=4;  SEEDS_GAUSS=2
  SKETCH_LIMIT=2;   FULL_M=1;      FULL_LIMIT=2
  P1_LIMIT=2;       P1_TOPK=4
  N_TRAIN=3600      # comparator/refit assert n_train > d=3584; round-1 lmsys train pool is 3,600
  CTX_FLAGS=(--smoke)
  # --all-positions-subset 1: the exploratory all_positions arm gets per-arm
  # smoke coverage (review v1 Major 1; #1090 fu5 per-arm-class rule).
  P3_EXTRA=(--limit-contexts 2 --k-samples 1 --k-baseline 1 --alphas 4 --all-positions-subset 1)
  WC_KEEP=2;        WC_EXTRA=(--allow-short); CAP_EXTRA=(--max-rows 2)
  NBOOT=50;         NDRAWS=8;      ASSEMBLE_EXTRA=(--max-chunks 1); COMP_EXTRA=(--max-chunks 1)
  BATTERY_EXTRA=(--n-pcs 4)   # acts14 has N_PAIRS=8 rows; basis needs n_rows > n_pcs
else
  PARITY_CHUNKS=4;  PARITY_ROWS=200
  JLENS_N=1000;     JLENS_LIMIT=()
  N_PAIRS=1536
  SEEDS_TOTAL=256;  SEEDS_TOPK=20; SEEDS_GAUSS=8
  SKETCH_LIMIT=512; FULL_M=150;    FULL_LIMIT=0
  P1_LIMIT=1024;    P1_TOPK=20
  N_TRAIN=50000
  CTX_FLAGS=()
  # §4 Phase 3: the all_positions=True persona-vectors variant runs on a
  # 50-context exploratory subset (review v1 Major 1 — previously unwired).
  P3_EXTRA=(--all-positions-subset 50)
  WC_KEEP=1000;     WC_EXTRA=();   CAP_EXTRA=()
  NBOOT=1000;       NDRAWS=100;    ASSEMBLE_EXTRA=(); COMP_EXTRA=()
  BATTERY_EXTRA=()  # n_pcs default 256 (plan §4 2c(ii) on-support restriction)
fi

# ── shared paths ──────────────────────────────────────────────────────────────
HF_DL="$DATA_DIR/hf_dl"
PASS_B="$HF_DL/issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
MANIFEST_DIR="$HF_DL/issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest"
CHUNKS_DIR="$HF_DL/issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
RAW_DIR="$HF_DL/issue779_monitoring/fitter-fair-comparison-n1m/raw_completions"
RB_DIR="$HF_DL/issue779_monitoring/r_b"
WEIGHTS_DIR="$HF_DL/issue779_monitoring/n1m_readout/weights"
MM_DIR="$DATA_DIR/n1m_mm"
COMP_DIR="$DATA_DIR/comparator"
JLENS_DIR="$DATA_DIR/jlens"
DICT_DIR="$DATA_DIR/dict"
JPAIRS_DIR="$DATA_DIR/jpairs"
CTX_JSONL="$DATA_DIR/contexts/contexts.jsonl"
SKETCH_ROOT="$OUT_ROOT/jac_sketch"
FULL_ROOT="$OUT_ROOT/jac_full"
P3_ROOT="$OUT_ROOT/phase3"
WC_DIR="$DATA_DIR/wildchat_fresh"
WC_CAP_ROOT="$OUT_ROOT/wildchat_fresh"
CENTROIDS="$DATA_DIR/centroids_v1_L21.pt"

CURRENT_PHASE="launch"
GATE_HALTED=0
LENS_OK=1

# ── progress sentinels (write-once; never re-read — pod-side-reporting rule) ──
progress() {  # progress <gate> <msg>   (non-blocking tick)
  local gate="$1" msg="$2"
  uv run python - "$ISSUE" "$LOG_DIR" "$gate" "$msg" <<'PY' || echo "[dispatch] WARN: progress sentinel write failed (gate=$gate)" >&2
import json, os, sys, time
from pathlib import Path

issue, log_dir, gate, msg = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
log_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,  # pod-side writers hardcode 1; the VM drain re-derives
    "task_id": int(issue),
    "gate": gate,  # carries "phase" -> the drain posts phase ticks verbatim
    "blocks_pipeline": False,
    "by": "issue1776_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": {"msg": msg, "mode": os.environ.get("EPS_1776_MODE", "?")},
}
slug = gate.replace(":", "_").replace("/", "_")  # gate already carries the phase- prefix
path = log_dir / f"issue-{issue}-{slug}-{int(time.time() * 1000)}.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=1))
os.replace(tmp, path)
print(f"[dispatch] progress sentinel: {path.name}")
PY
}

phase_begin() {
  CURRENT_PHASE="$1"
  echo "[phase=$1]"
  echo "$1" >> "$TRACE"
  # RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking; body is one heredoc python
  progress "phase-$1" "begin (mode=$MODE ngpu=$NGPU)" || true
}
# RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking; body is one heredoc python
phase_end() { progress "phase-$1" "done" || true; }

gate_halt() {  # gate_halt <gate-name> <rc> <msg>
  GATE_HALTED=1
  echo "[dispatch] ENGINEERING GATE HALT: $1 rc=$2 — $3" >&2
  # RC_CAPTURE_EXEMPT: gate sentinel is best-effort; the distinct exit rc is the signal
  progress "phase-gate-halt-$1" "rc=$2: $3" || true
  exit "$2"
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && $GATE_HALTED -eq 0 ]]; then
    echo "[dispatch] FAILED at phase=$CURRENT_PHASE rc=$rc" >&2
    # RC_CAPTURE_EXEMPT: best-effort crash breadcrumb inside the EXIT trap; rc already captured
    progress "phase-crash" "phase=$CURRENT_PHASE rc=$rc" || true
  fi
}
trap on_exit EXIT

# ── runners ───────────────────────────────────────────────────────────────────
# Each sub-command's stdout+stderr ALSO appends to a per-phase log under
# $LOG_DIR (/workspace/logs on both lanes), which the GCE crash trap persists
# to HF issue1776_partial/ — att-20260729-060640's p0_stage stderr lived only
# under $PHASE_LOGS ($OUT_ROOT/logs) and died with the instance.
phase_dlog() { echo "$LOG_DIR/issue-${ISSUE}-phase-${CURRENT_PHASE}.log"; }

run() {  # run <log-name> <cmd...>   (foreground, redirected; DRY: trace only)
  local plog="$PHASE_LOGS/$1.log"; shift
  if [[ $DRY == 1 ]]; then
    echo "DRY: $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    return 0
  fi
  echo "[dispatch] run($CURRENT_PHASE): $*"
  # Synchronous tee: the crash-persisted per-phase log is complete when the
  # payload exits. pipefail is set, and PIPESTATUS[0] pins the PAYLOAD rc
  # (post-pipe $? is the last stage's status — code-style.md).
  "$@" 2>&1 | tee -a "$plog" >> "$(phase_dlog)"
  return "${PIPESTATUS[0]}"
}

bg_run() {  # bg_run <log-name> <cvd> <cmd...> -> sets BG_PID ('' in DRY)
  # Sets the global BG_PID in the CURRENT shell. NEVER capture via
  # p="$(bg_run ...)": command substitution forks a subshell, the background
  # job is the SUBSHELL's child, and the parent's later `wait "$p"` fails
  # "not a child of this shell" (rc=127) at the first fan-out join — caught
  # by this round's runner harness before it could burn another GCE cycle.
  local plog="$PHASE_LOGS/$1.log" cvd="$2"; shift 2
  BG_PID=""
  if [[ $DRY == 1 ]]; then
    echo "DRY: CUDA_VISIBLE_DEVICES=$cvd $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    return 0
  fi
  echo "[dispatch] bg($CURRENT_PHASE, CVD=$cvd): $*" >&2
  # Process substitution (NOT a pipe) keeps $! = the payload pid for wait_rc.
  CUDA_VISIBLE_DEVICES="$cvd" "$@" > >(tee -a "$plog" >> "$(phase_dlog)") 2>&1 &
  BG_PID=$!
}

wait_rc() {  # wait_rc <pid-or-empty> -> return the pid's rc (0 for DRY '')
  local p="$1" rc=0
  [[ -z "$p" ]] && return 0
  wait "$p" || rc=$?
  return $rc
}

# ── launch preamble: out-root mount headroom (#1333 pattern) ──────────────────
phase_begin "p0_stage"
if [[ $DRY == 0 ]]; then
  NEED_GB=$([[ "$MODE" == "smoke" ]] && echo 15 || echo 40)
  run headroom uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_ROOT', $NEED_GB, phase='launch')
print(f'[headroom] out_root=$OUT_ROOT free_gb={free:.1f} floor=$NEED_GB')
"
fi

# ── p0_stage: pin + probes + provenance + reused-artifact staging ─────────────
# Also stages (gitignored — not in the clone): the #779 trait artifacts for the
# contexts builder, and the canonical-pool centroids bundle for the 5b leakage
# leg (sha-asserted in stage.py against the committed bank meta). Formerly an
# inline heredoc whose wrong Hub path 404-killed att-20260729-060640.
run p0_stage uv run python scripts/issue1776_stage.py \
  --stage-bundle --stage-weights --stage-rb --stage-manifest \
  --stage-trait-artifacts --stage-centroids --centroids-dest "$CENTROIDS" \
  --parity-chunks "$PARITY_CHUNKS" --report "$DATA_DIR/stage_report.json"
phase_end "p0_stage"

# ── p5a stream: CPU/network-bound, concurrent with GPU work (§9) ──────────────
phase_begin "p5a_stream_launch"
bg_run p5a_stream "" uv run python scripts/issue1776_phase5.py stream \
  --out-dir "$WC_DIR" --n-keep "$WC_KEEP" ${WC_EXTRA[@]+"${WC_EXTRA[@]}"}
STREAM_PID="$BG_PID"
phase_end "p5a_stream_launch"

# ── p0_parity: G-PARITY (rc=8 -> program halt: everything downstream consumes
#    the same captured fields) ─────────────────────────────────────────────────
phase_begin "p0_parity"
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p0_parity env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_parity.py \
  --chunks-dir "$CHUNKS_DIR" --raw-dir "$RAW_DIR" --n-rows "$PARITY_ROWS" \
  --out-dir "$DATA_DIR/parity" || rc=$?
if [[ $rc -eq 8 ]]; then
  gate_halt "G-PARITY" 8 "parity halt: >5% rows below 0.999 (report: $DATA_DIR/parity)"
elif [[ $rc -eq 7 ]]; then
  gate_halt "G-PILOT-parity" 7 "pilot gate halt in parity phase"
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p0_parity"

# ── p0_comparator (BG on the last GPU) + p0_jlens (fan on the rest) ───────────
phase_begin "p0_comparator_launch"
COMP_GPU=$((NGPU - 1))
comparator_job() {
  # --parity-exclusion: plan §3 — failed-parity cis leave the train pool
  # (p0_parity runs before this launch; review v1 exclusion-list wiring).
  uv run python scripts/issue1776_comparator_fit.py --tag m_ridge_x50k \
    --n-train "$N_TRAIN" --out-dir "$COMP_DIR" --pass-b "$PASS_B" --mm-dir "$MM_DIR" \
    --parity-exclusion "$DATA_DIR/parity/exclusion_list.json" \
    ${COMP_EXTRA[@]+"${COMP_EXTRA[@]}"} \
    && uv run python scripts/issue1776_comparator_fit.py --tag m_ridge_lmsys50k \
      --lmsys-only --n-train "$N_TRAIN" --out-dir "$COMP_DIR" --pass-b "$PASS_B" \
      --mm-dir "$MM_DIR" --parity-exclusion "$DATA_DIR/parity/exclusion_list.json" \
      ${COMP_EXTRA[@]+"${COMP_EXTRA[@]}"}
}
if [[ $DRY == 1 ]]; then
  run p0_comparator echo "comparator x50k + lmsys50k (n_train=$N_TRAIN)"
  COMP_PID=""
else
  # bg_run tees to BOTH $PHASE_LOGS/p0_comparator.log AND the crash-persisted
  # per-phase log (r5 concern comparator-bg-job-not-crash-persisted); BG_PID
  # stays the payload pid ($! of the backgrounded function call, as before).
  bg_run p0_comparator "$COMP_GPU" comparator_job
  COMP_PID="$BG_PID"
fi
phase_end "p0_comparator_launch"

phase_begin "p0_jlens"
NF=$NGPU
if [[ $NGPU -gt 1 ]]; then NF=$((NGPU - 1)); fi  # comparator holds the last GPU
run p0_jlens_prompts uv run python scripts/issue1776_jlens_fit.py build-prompts \
  --out "$DATA_DIR/jlens_prompts.jsonl" --n "$JLENS_N"
mkdir -p "$JLENS_DIR"
JL_PIDS=()
for ((g = 0; g < NF; g++)); do
  bg_run "p0_jlens_fit_shard$g" "$g" uv run python scripts/issue1776_jlens_fit.py fit \
    --prompts "$DATA_DIR/jlens_prompts.jsonl" --out "$JLENS_DIR/shard$g.pt" \
    --shard-index "$g" --n-shards "$NF" --checkpoint "$JLENS_DIR/ckpt_shard$g.pt" \
    ${JLENS_LIMIT[@]+"${JLENS_LIMIT[@]}"}
  p="$BG_PID"
  JL_PIDS+=("$p")
done
for p in ${JL_PIDS[@]+"${JL_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
if [[ $DRY == 1 ]]; then
  run p0_jlens_merge echo "merge shards -> lens.pt"
else
  shards=("$JLENS_DIR"/shard*.pt)
  run p0_jlens_merge uv run python scripts/issue1776_jlens_fit.py merge \
    --shards "${shards[@]}" --out "$JLENS_DIR/lens.pt"
fi
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p0_jlens_sanity env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_jlens_fit.py \
  sanity --lens "$JLENS_DIR/lens.pt" --out "$EVAL_DIR/phase0/glens_gate.json" || rc=$?
if [[ $rc -eq 8 ]]; then
  # G-LENS: lens-dependent legs (dictionaries, phase4, 5b) are skipped; the
  # program continues (plan §7: gates abort the affected numbers, not the
  # program) and the dispatcher exits 8 at the end.
  LENS_OK=0
  echo "[dispatch] G-LENS FAIL: skipping p0_dict / p4_mediation / p5b_leakage" >&2
  # RC_CAPTURE_EXEMPT: gate sentinel is best-effort; LENS_OK=0 + final rc=8 carry the state
  progress "phase-gate-halt-G-LENS" "rc=8: lens sanity failed; dependent legs skipped" || true
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p0_jlens"

phase_begin "p0_comparator_join"
wait_rc "$COMP_PID" || {
  rc=$?
  if [[ $rc -eq 7 ]]; then gate_halt "G-PILOT-comparator" 7 "pilot gate halt in comparator phase"; fi
  exit "$rc"
}
phase_end "p0_comparator_join"

# ── p0_dict: dictionaries at L14/L19 (+L21 for the 5b leg), lens-gated ────────
phase_begin "p0_dict"
if [[ $LENS_OK -eq 1 ]]; then
  mkdir -p "$DICT_DIR"
  D_PIDS=()
  gi=0
  for L in 14 19 21; do
    bg_run "p0_dict_l$L" "$((gi % NGPU))" uv run python scripts/issue1776_phase4.py \
      build-dict --lens "$JLENS_DIR/lens.pt" --layer "$L" --out "$DICT_DIR/dictionary_l$L.pt"
    p="$BG_PID"
    D_PIDS+=("$p"); gi=$((gi + 1))
  done
  for p in ${D_PIDS[@]+"${D_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
else
  run p0_dict echo "SKIPPED (G-LENS fail)"
fi
phase_end "p0_dict"

# ── p04_pairs: J-pair manifest + sharded teacher-forced capture ───────────────
phase_begin "p04_pairs"
mkdir -p "$JPAIRS_DIR"
run p04_pairs_build uv run python scripts/issue1776_jpairs.py build \
  --n-pairs "$N_PAIRS" --out-dir "$JPAIRS_DIR" --manifest-dir "$MANIFEST_DIR" \
  --exclusion "$DATA_DIR/parity/exclusion_list.json"

CAP_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p04_capture_shard$g" "$g" uv run python scripts/issue1776_jpairs.py \
    capture-shard --pairs "$JPAIRS_DIR/jpairs.jsonl" --out "$JPAIRS_DIR/cap_shard$g.pt" \
    --shard-index "$g" --num-shards "$NGPU"
  p="$BG_PID"
  CAP_PIDS+=("$p")
done
for p in ${CAP_PIDS[@]+"${CAP_PIDS[@]}"}; do wait_rc "$p" || exit $?; done

run p04_pairs_merge uv run python scripts/issue1776_jpairs.py merge \
  --jpairs-dir "$JPAIRS_DIR" --num-shards "$NGPU"
phase_end "p04_pairs"

# ── p1_contexts (needs the staged #779 trait artifacts) ───────────────────────
phase_begin "p1_contexts"
run p1_contexts uv run python scripts/issue1776_contexts.py --out "$CTX_JSONL" \
  ${CTX_FLAGS[@]+"${CTX_FLAGS[@]}"}
phase_end "p1_contexts"

# ── p2a_sketch: seeds -> sharded sketch run -> merge (G-NONZERO inside run) ───
phase_begin "p2a_sketch"
mkdir -p "$SKETCH_ROOT"
run p2a_seeds uv run python scripts/issue1776_jacobian.py build-seeds \
  --v-pool "$JPAIRS_DIR/v_pool.pt" --comparator "$COMP_DIR/m_ridge_x50k.pt" \
  --n-total "$SEEDS_TOTAL" --topk-comparator "$SEEDS_TOPK" --n-gaussian "$SEEDS_GAUSS" \
  --out "$SKETCH_ROOT/seeds.pt"
SK_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p2a_sketch_shard$g" "$g" uv run python scripts/issue1776_jacobian.py run \
    --mode sketch --pairs "$JPAIRS_DIR/jpairs.jsonl" --seeds-file "$SKETCH_ROOT/seeds.pt" \
    --limit-pairs "$SKETCH_LIMIT" --shard-index "$g" --num-shards "$NGPU" \
    --out-dir "$SKETCH_ROOT/shard$g"
  p="$BG_PID"
  SK_PIDS+=("$p")
done
SK_RC=0
for p in ${SK_PIDS[@]+"${SK_PIDS[@]}"}; do
  wait_rc "$p" || { rc=$?; if [[ $rc -eq 8 ]]; then SK_RC=8; else exit $rc; fi; }
done
if [[ $SK_RC -eq 8 ]]; then
  gate_halt "G-NONZERO" 8 "all-zero context-gradient field in sketch run (slot-convention bug)"
fi
run p2a_merge uv run python scripts/issue1776_jacobian.py merge-shards \
  --shards-root "$SKETCH_ROOT" --out-dir "$SKETCH_ROOT/merged"
phase_end "p2a_sketch"

# ── p1_diag: Phase-1 directional diagnostic ───────────────────────────────────
phase_begin "p1_diag"
run p1_diag env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_phase1.py \
  --comparator "$COMP_DIR/m_ridge_x50k.pt" --pairs "$JPAIRS_DIR/jpairs.jsonl" \
  --topk "$P1_TOPK" --limit-pairs "$P1_LIMIT" --out-dir "$EVAL_DIR/phase1"
phase_end "p1_diag"

# ── uploads: batched create_commit + scoped post-upload verify ────────────────
upload_batch() {  # upload_batch <log-name> <hub-prefix> <commit-msg> <listfile rel=abs per line>
  local lname="$1" prefix="$2" msg="$3" listfile="$4"
  run "$lname" uv run python scripts/issue1776_upload_batch.py \
    --prefix "$prefix" --listfile "$listfile" --message "$msg"
}

list_add() {  # list_add <listfile> <rel> <abs> — include iff a regular FILE exists
  # ONLY for legitimately-conditional artifacts (LENS_OK-gated dicts, glob
  # loops); an expected-present artifact uses list_add_required (#825 class:
  # the existence guard silently skips a typo'd path — review v1 Critical 3).
  # -f, NOT -e (crash-fix r9): CommitOperationAdd rejects directories — a DIR
  # here is the misnested-out-arg class, never a valid upload source.
  [[ -f "$3" ]] && echo "$2=$3" >> "$1" || true
}

list_add_required() {  # fail LOUD when an expected-present artifact is missing
  if [[ $DRY == 1 ]]; then list_add "$@"; return 0; fi
  if [[ ! -f "$3" ]]; then
    echo "[dispatch] upload-list FATAL: required artifact missing or not a regular FILE: $3 (rel=$2)" >&2
    exit 1
  fi
  echo "$2=$3" >> "$1"
}

phase_begin "p_early_upload"
# §9: regeneration-costly intermediates upload at the END of their producing
# phase and BEFORE the long P2b sweep (#825) — everything produced so far.
EARLY_LIST="$OUT_ROOT/upload_early.list"; : > "$EARLY_LIST"
list_add_required "$EARLY_LIST" "jlens/lens.pt" "$JLENS_DIR/lens.pt"
for L in 14 19 21; do
  # legitimately-conditional: absent when G-LENS failed (dict legs skipped)
  list_add "$EARLY_LIST" "dictionaries/dictionary_l$L.pt" "$DICT_DIR/dictionary_l$L.pt"
done
for t in m_ridge_x50k m_ridge_lmsys50k; do
  list_add_required "$EARLY_LIST" "comparator/$t.pt" "$COMP_DIR/$t.pt"
done
if [[ $DRY == 0 ]]; then
  for f in "$COMP_DIR"/*.json "$DATA_DIR/parity"/*.json "$DATA_DIR/stage_report.json" \
           "$JPAIRS_DIR/jpairs_build_report.json"; do
    list_add "$EARLY_LIST" "reports/$(basename "$f")" "$f"
  done
fi
# §4 P0.1 "seeded, persisted" jlens fit corpus + its revision-pinned meta
# (review v1 Critical 3 (b)).
list_add_required "$EARLY_LIST" "jlens/jlens_prompts.jsonl" "$DATA_DIR/jlens_prompts.jsonl"
list_add_required "$EARLY_LIST" "jlens/jlens_prompts.meta.json" "$DATA_DIR/jlens_prompts.meta.json"
list_add_required "$EARLY_LIST" "jpairs/jpairs.jsonl" "$JPAIRS_DIR/jpairs.jsonl"
list_add_required "$EARLY_LIST" "jpairs/jpair_capture.pt" "$JPAIRS_DIR/jpair_capture.pt"
list_add_required "$EARLY_LIST" "jpairs/v_pool.pt" "$JPAIRS_DIR/v_pool.pt"
list_add_required "$EARLY_LIST" "jpairs/acts14.pt" "$JPAIRS_DIR/acts14.pt"
list_add_required "$EARLY_LIST" "jpairs/acts19.pt" "$JPAIRS_DIR/acts19.pt"
list_add_required "$EARLY_LIST" "jac_sketch/seeds.pt" "$SKETCH_ROOT/seeds.pt"
if [[ $DRY == 0 ]]; then
  for f in "$SKETCH_ROOT/merged"/*; do
    list_add "$EARLY_LIST" "jac_sketch/merged/$(basename "$f")" "$f"
  done
fi
list_add_required "$EARLY_LIST" "contexts/contexts.jsonl" "$CTX_JSONL"
# contexts.py writes <stem>.meta.json — the prior "meta.json" path was a typo
# the -e guard silently skipped (review v1 Critical 3 (c)).
list_add_required "$EARLY_LIST" "contexts/contexts.meta.json" \
  "$(dirname "$CTX_JSONL")/contexts.meta.json"
upload_batch p_early_upload "$HF_PREFIX_EFF/analysis_tensors" \
  "task #1776: phase0-2a tensors + manifests (pre-P2b, #825 ordering)" "$EARLY_LIST"
phase_end "p_early_upload"

# ── p2b_full: 3,584-seed full-rank sweep, seed-block shard fan-out ────────────
phase_begin "p2b_full"
mkdir -p "$FULL_ROOT"
F_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p2b_full_shard$g" "$g" uv run python scripts/issue1776_jacobian.py run \
    --mode full --pairs "$JPAIRS_DIR/jpairs.jsonl" --m "$FULL_M" \
    --limit-pairs "$FULL_LIMIT" --shard-index "$g" --num-shards "$NGPU" \
    --out-dir "$FULL_ROOT/shard$g"
  p="$BG_PID"
  F_PIDS+=("$p")
done
F_RC=0
for p in ${F_PIDS[@]+"${F_PIDS[@]}"}; do
  wait_rc "$p" || { rc=$?; if [[ $rc -eq 8 ]]; then F_RC=8; else exit $rc; fi; }
done
if [[ $F_RC -eq 8 ]]; then
  gate_halt "G-NONZERO" 8 "all-zero context-gradient field in full run (slot-convention bug)"
fi
run p2b_merge uv run python scripts/issue1776_jacobian.py merge-shards \
  --shards-root "$FULL_ROOT" --out-dir "$FULL_ROOT/merged"
phase_end "p2b_full"

phase_begin "p2_upload"
P2_LIST="$OUT_ROOT/upload_p2.list"; : > "$P2_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$FULL_ROOT/merged"/*; do
    list_add "$P2_LIST" "jac_full/$(basename "$f")" "$f"
  done
fi
upload_batch p2_upload "$HF_PREFIX_EFF/analysis_tensors" \
  "task #1776: full-rank J + even/odd halves + intercepts (P2b)" "$P2_LIST"
phase_end "p2_upload"

# ── p2c_battery: §4 P2c(ii) operator battery — §6.5 phase2/operator_battery.json
#    (review v1 Critical 2). Shipped-M resolution hoisted here (also used by
#    the later p5_transfer reference row). ───────────────────────────────────
phase_begin "p2c_battery"
mkdir -p "$EVAL_DIR/phase2"
# Shipped-M resolution (concern shipped-m-l19-glob-fallback): the EXACT staged
# path is primary — plan §10 pins the layout weights_dir/L{layer}/{fitter}.pt
# (issue779_n1m_readout.py:201); the basename/path glob is FALLBACK only. An
# unresolved reference arm FAILS LOUD unless --allow-missing-reference-arm was
# passed, in which case the omission is recorded in the eval-results JSON set
# (committed at p_results_commit) + the final sentinel — never silent.
SHIPPED_M=""
SHIPPED_M_EXACT="$WEIGHTS_DIR/L19/ridge.pt"
if [[ $DRY == 0 ]]; then
  if [[ -f "$SHIPPED_M_EXACT" ]]; then
    SHIPPED_M="$SHIPPED_M_EXACT"
  elif [[ -d "$WEIGHTS_DIR" ]]; then
    SHIPPED_M="$(find "$WEIGHTS_DIR" \( -ipath '*l19*.pt' -o -iname '*l19*.pt' -o -iname '*layer19*.pt' \) 2>/dev/null | sort | head -1 || true)"
    [[ -n "$SHIPPED_M" ]] && echo "[dispatch] NOTE: shipped-M resolved via GLOB FALLBACK: $SHIPPED_M (exact $SHIPPED_M_EXACT missing)"
  fi
  if [[ -z "$SHIPPED_M" ]]; then
    if [[ $ALLOW_MISSING_REF == 1 ]]; then
      OMIT_JSON="$EVAL_DIR/phase2/shipped_m_reference_omitted.json"
      printf '{"omitted_arm": "m_shipped (labeled reference — plan §4 H3 decay table)", "expected_path": "%s", "allow_missing_reference_arm": true, "ts": "%s"}\n' \
        "$SHIPPED_M_EXACT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OMIT_JSON"
      echo "[dispatch] WARNING: shipped-M reference arm UNRESOLVED — omission AUTHORIZED (--allow-missing-reference-arm) and recorded at $OMIT_JSON" >&2
    else
      echo "[dispatch] FATAL: shipped-M labeled-reference weights unresolved (expected $SHIPPED_M_EXACT; glob fallback empty under $WEIGHTS_DIR). The plan §4 H3 decay table + p2c cross-slot spectrum row require the m_shipped arm — pass --allow-missing-reference-arm to run without it (omission then recorded, never silent)." >&2
      exit 1
    fi
  fi
fi
B_ARGS=(uv run python scripts/issue1776_phase2_battery.py
  --jlast "$FULL_ROOT/merged/J_last.pt" --mprime-weights "$COMP_DIR/m_ridge_x50k.pt"
  --acts14 "$JPAIRS_DIR/acts14.pt" --n-draws "$NDRAWS"
  --out "$EVAL_DIR/phase2/operator_battery.json")
if [[ -n "$SHIPPED_M" ]]; then
  B_ARGS+=(--shipped-m "$SHIPPED_M")
else
  echo "[dispatch] NOTE: battery runs without the cross-slot spectrum row (shipped-M arm omitted — dry-run, or authorized via --allow-missing-reference-arm)"
fi
run p2c_battery env CUDA_VISIBLE_DEVICES=0 "${B_ARGS[@]}" ${BATTERY_EXTRA[@]+"${BATTERY_EXTRA[@]}"}
phase_end "p2c_battery"

# ── p3_grid: baseline -> steered strata fan-out -> finalize ───────────────────
phase_begin "p3_grid"
mkdir -p "$P3_ROOT" "$EVAL_DIR/phase3"
P3_BASE=(uv run python scripts/issue1776_phase3.py --mode run
  --contexts "$CTX_JSONL" --rb-dir "$RB_DIR" --mprime-weights "$COMP_DIR/m_ridge_x50k.pt"
  --jlast "$FULL_ROOT/merged/J_last.pt" --out-root "$P3_ROOT")
run p3_baseline env CUDA_VISIBLE_DEVICES=0 "${P3_BASE[@]}" --baseline-only \
  ${P3_EXTRA[@]+"${P3_EXTRA[@]}"}
P3_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p3_strata_shard$g" "$g" "${P3_BASE[@]}" \
    --strata-shard "$g" --strata-num-shards "$NGPU" ${P3_EXTRA[@]+"${P3_EXTRA[@]}"}
  p="$BG_PID"
  P3_PIDS+=("$p")
done
for p in ${P3_PIDS[@]+"${P3_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
# --eval-out is the eval DIRECTORY (issue1776_phase3.py mkdirs it and writes
# steered_shift_summaries.json + raw_completions_manifest.json INSIDE it).
# Crash-fix r9: the pre-fix invocation passed the deliverable FILE path here,
# so finalize misnested both outputs one level deep and p3_upload crashed at
# CommitOperationAdd ("not a file"). phase3 now REFUSES a file-shaped path.
run p3_finalize env CUDA_VISIBLE_DEVICES=0 "${P3_BASE[@]}" --finalize-only \
  --eval-out "$EVAL_DIR/phase3" ${P3_EXTRA[@]+"${P3_EXTRA[@]}"}
phase_end "p3_grid"

phase_begin "p3_upload"
# Steered rollout TEXT -> raw_completions BEFORE any downstream reduce/judging
# (Upload Policy; §9 phase-order). Summaries + manifest -> analysis_tensors.
P3_TEXT_LIST="$OUT_ROOT/upload_p3_text.list"; : > "$P3_TEXT_LIST"
P3_TENS_LIST="$OUT_ROOT/upload_p3_tens.list"; : > "$P3_TENS_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$P3_ROOT"/raw_completions/steered/*.json; do
    list_add "$P3_TEXT_LIST" "$(basename "$f")" "$f"
  done
  for f in "$P3_ROOT"/summaries/*.pt; do
    list_add "$P3_TENS_LIST" "phase3/$(basename "$f")" "$f"
  done
fi
# Plan-§6.5 deliverable FILES — finalize writes BOTH into $EVAL_DIR/phase3
# (--eval-out is the DIRECTORY). Crash-fix r9: the pre-fix list (a) pointed
# the manifest at $P3_ROOT, where nothing is written — list_add's existence
# guard silently skipped it (the #825 class) — and (b) listed the summaries
# path while a mis-wired --eval-out had made it a DIRECTORY. Required-present:
# a missing/dir-shaped deliverable now fails LOUD at list composition.
list_add_required "$P3_TENS_LIST" "phase3/raw_completions_manifest.json" \
  "$EVAL_DIR/phase3/raw_completions_manifest.json"
list_add_required "$P3_TENS_LIST" "phase3/steered_shift_summaries.json" \
  "$EVAL_DIR/phase3/steered_shift_summaries.json"
upload_batch p3_upload_text "$HF_PREFIX_EFF/raw_completions/steered" \
  "task #1776: steered rollout text (P3, before reduce/judge)" "$P3_TEXT_LIST"
upload_batch p3_upload_tens "$HF_PREFIX_EFF/analysis_tensors" \
  "task #1776: P3 per-sample summaries + manifest" "$P3_TENS_LIST"
phase_end "p3_upload"

# ── p4_mediation (lens-gated; phase4 invoked via subprocess ONLY) ─────────────
phase_begin "p4_mediation"
if [[ $LENS_OK -eq 1 ]]; then
  mkdir -p "$EVAL_DIR/phase4"
  P4_PIDS=()
  bg_run p4_energy "0" uv run python scripts/issue1776_phase4.py energy \
    --dict14 "$DICT_DIR/dictionary_l14.pt" --dict19 "$DICT_DIR/dictionary_l19.pt" \
    --mprime-weights "$COMP_DIR/m_ridge_x50k.pt" --jlast "$FULL_ROOT/merged/J_last.pt" \
    --rb-dir "$RB_DIR" --phase3-root "$P3_ROOT" \
    --acts14 "$JPAIRS_DIR/acts14.pt" --acts19 "$JPAIRS_DIR/acts19.pt" \
    --n-draws "$NDRAWS" --out "$EVAL_DIR/phase4/jspace_energy.json"
  p="$BG_PID"
  P4_PIDS+=("$p")
  bg_run p4_refit_split "$((NGPU > 1 ? 1 : 0))" uv run python scripts/issue1776_phase4.py \
    refit-split --dict19 "$DICT_DIR/dictionary_l19.pt" --n-train "$N_TRAIN" \
    --pass-b "$PASS_B" --mm-dir "$MM_DIR" --out "$EVAL_DIR/phase4/refit_split.json"
  p="$BG_PID"
  P4_PIDS+=("$p")
  bg_run p4_jdelta_split "$((NGPU > 2 ? 2 : 0))" uv run python scripts/issue1776_phase4.py \
    jdelta-split --dict14 "$DICT_DIR/dictionary_l14.pt" --jlast "$FULL_ROOT/merged/J_last.pt" \
    --phase3-root "$P3_ROOT" --out "$EVAL_DIR/phase4/jdelta_split.json"
  p="$BG_PID"
  P4_PIDS+=("$p")
  for p in ${P4_PIDS[@]+"${P4_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
else
  run p4_mediation echo "SKIPPED (G-LENS fail)"
fi
phase_end "p4_mediation"

# ── p5a: WildChat fresh capture (stream join, then sharded gen+capture) ───────
phase_begin "p5a_stream_join"
wait_rc "$STREAM_PID" || exit $?
# §9 phase_outputs.p5_transfer prompt manifest (review v1 Critical 3 (a)):
# persist the fresh-WildChat pool + meta + stream report BEFORE the GPU
# capture consumes it (#825 ordering). Text-only — rides the non-LFS path.
WC_LIST="$OUT_ROOT/upload_wc_pool.list"; : > "$WC_LIST"
list_add_required "$WC_LIST" "wildchat_fresh_pool.jsonl" "$WC_DIR/wildchat_fresh_pool.jsonl"
list_add_required "$WC_LIST" "wildchat_fresh_pool.meta.json" "$WC_DIR/wildchat_fresh_pool.meta.json"
list_add_required "$WC_LIST" "stream_report.json" "$WC_DIR/stream_report.json"
upload_batch p5a_pool_upload "$HF_PREFIX_EFF/wildchat_fresh" \
  "task #1776: fresh WildChat prompt pool + stream report (pre-capture)" "$WC_LIST"
phase_end "p5a_stream_join"

phase_begin "p5a_capture"
C_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  bg_run "p5a_capture_shard$g" "$g" uv run python scripts/issue1776_phase5.py capture \
    --pool "$WC_DIR/wildchat_fresh_pool.jsonl" --out-root "$WC_CAP_ROOT" \
    --shard-index "$g" --n-shards "$NGPU" --hf-prefix "$HF_PREFIX_EFF/wildchat_fresh" \
    ${CAP_EXTRA[@]+"${CAP_EXTRA[@]}"}
  p="$BG_PID"
  C_PIDS+=("$p")
done
for p in ${C_PIDS[@]+"${C_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
phase_end "p5a_capture"

# ── p5_transfer: P2c reads + P5 decay legs (test-1000 + wildchat_fresh) ───────
phase_begin "p5_transfer"
mkdir -p "$EVAL_DIR/phase5" "$EVAL_DIR/phase2"
# SHIPPED_M resolved at p2c_battery (hoisted; same variable reused here).
T_ARGS=(uv run python scripts/issue1776_phase5.py transfer --assemble
  --pass-b "$PASS_B" --mm-dir "$MM_DIR" --out-dir "$DATA_DIR/transfer"
  --op "mprime_x50k=$COMP_DIR/m_ridge_x50k.pt=14"
  --op "mprime_lmsys50k=$COMP_DIR/m_ridge_lmsys50k.pt=14"
  --jop "J_last=$FULL_ROOT/merged/J_last.pt"
  --jop "J_ctx=$FULL_ROOT/merged/J_ctx.pt"
  --jop "J_prefix=$FULL_ROOT/merged/J_prefix.pt"
  # 3-part leg spec (crash-fix r12): p5a_capture uploads-then-PURGES its local
  # chunks (N1G._flush_upload_batch disk-bounding), so the consumer carries the
  # hf-prefix and load_leg Hub-stages the purged chunks back (#1489 class).
  --leg "wildchat_fresh=$WC_CAP_ROOT=$HF_PREFIX_EFF/wildchat_fresh"
  --n-boot "$NBOOT" --out "$EVAL_DIR/phase5/transfer.json"
  --jvm-heldout-out "$EVAL_DIR/phase2/jvm_heldout.json")
if [[ -n "$SHIPPED_M" ]]; then
  T_ARGS+=(--op "m_shipped=$SHIPPED_M=19")
else
  echo "[dispatch] NOTE: transfer runs without the m_shipped reference arm (dry-run, or omission authorized via --allow-missing-reference-arm — recorded in phase2/shipped_m_reference_omitted.json + final sentinel)"
fi
run p5_transfer env CUDA_VISIBLE_DEVICES=0 "${T_ARGS[@]}" ${ASSEMBLE_EXTRA[@]+"${ASSEMBLE_EXTRA[@]}"}
phase_end "p5_transfer"

# ── p5b_leakage (CPU; plan §9 schedules this re-read OFF-POD at p7 — the
#    POD-side default here is a recorded deviation: every input (centroids +
#    L21 dict) is already staged/built on this pod, so running it here avoids
#    a second staging pass. Either lane works (concern
#    p5b-podside-vs-offpod-deviation): set EPS_1776_P5B_OFFPOD=1 to skip the
#    pod-side leg and run the documented off-pod invocation carried in the
#    final sentinel's offpod_handoffs.p5b_leakage entry. Deviation + rationale
#    recorded in the final sentinel plan_deviation note; the analyzer carries
#    it as an ops/scope caveat. ────────────────────────────────────────────────
P5B_OFFPOD="${EPS_1776_P5B_OFFPOD:-0}"
phase_begin "p5b_leakage"
if [[ "$P5B_OFFPOD" == "1" ]]; then
  run p5b_leakage echo "SKIPPED (EPS_1776_P5B_OFFPOD=1 — run off-pod per the final sentinel offpod_handoffs.p5b_leakage entry)"
elif [[ $LENS_OK -eq 1 ]]; then
  run p5b_leakage uv run python scripts/issue1776_phase5.py leakage \
    --centroids "$CENTROIDS" --dict "$DICT_DIR/dictionary_l21.pt" \
    --n-boot "$NBOOT" --out "$EVAL_DIR/phase5/leakage_reread.json"
else
  run p5b_leakage echo "SKIPPED (G-LENS fail)"
fi
phase_end "p5b_leakage"

# ── p_results_commit: git-destined eval JSONs (#1205 push-verify + #1325) ─────
phase_begin "p_results_commit"
if [[ $DRY == 1 || "$MODE" == "smoke" ]]; then
  run p_results_commit echo "SKIPPED (dry-run/smoke: eval outputs stay under $EVAL_DIR)"
else
  BR="$(git rev-parse --abbrev-ref HEAD)"
  mapfile -t DECLARED < <(find "$EVAL_DIR" -name '*.json' | sort)
  if [[ ${#DECLARED[@]} -gt 0 ]]; then
    git add -- "${DECLARED[@]}"
    if ! git diff --cached --quiet; then
      git commit -m "task #1776: pod-side eval results (phases 0-5, mode=$MODE)"
    fi
    if ! git push origin "$BR"; then
      echo "[dispatch] push failed; retrying once in 20s" >&2
      sleep 20
      git push origin "$BR"
    fi
    AHEAD="$(git rev-list --count "origin/$BR..HEAD")"
    [[ "$AHEAD" == "0" ]] || { echo "[dispatch] push-verify FAIL: $AHEAD unpushed commits" >&2; exit 1; }
    MISSING=0
    for f in "${DECLARED[@]}"; do
      rel="${f#"$REPO_ROOT"/}"
      if [[ -z "$(git ls-tree -r "origin/$BR" --name-only -- "$rel")" ]]; then
        echo "[dispatch] artifact-presence FAIL: $rel not in pushed tree" >&2
        MISSING=1
      fi
    done
    [[ $MISSING -eq 0 ]] || exit 1
    echo "[dispatch] results commit pushed + verified (${#DECLARED[@]} JSONs on origin/$BR)"
  else
    echo "[dispatch] no eval JSONs to commit" >&2
  fi
fi
phase_end "p_results_commit"

# ── p_final: terminal results sentinel, then [phase=done] ─────────────────────
phase_begin "p_final"
FINAL_RC=0
if [[ $LENS_OK -eq 0 ]]; then
  FINAL_RC=8
  GATE_HALTED=1  # designed halt (G-LENS), not a crash — the EXIT trap stays quiet
fi
uv run python - "$ISSUE" "$LOG_DIR" "$MODE" "$DRY" "$EVAL_DIR" "$REPO_ROOT" \
  "$HF_PREFIX_EFF" "$LENS_OK" "$SHIPPED_M" "$NGPU" "$P5B_OFFPOD" "$ALLOW_MISSING_REF" \
  "$CENTROIDS" "$DICT_DIR" "$NBOOT" <<'PY'
"""Terminal results sentinel (§10 structured fields: eval paths, HF prefixes,
off-pod handoffs; no training this run -> no adapter/wandb-run fields)."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

issue, log_dir, mode, dry = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
eval_dir, repo_root = Path(sys.argv[5]), Path(sys.argv[6])
hf_prefix, lens_ok, shipped_m, ngpu = sys.argv[7], sys.argv[8] == "1", sys.argv[9], sys.argv[10]
p5b_offpod, allow_missing_ref = sys.argv[11] == "1", sys.argv[12] == "1"
centroids, dict_dir, nboot = sys.argv[13], sys.argv[14], sys.argv[15]
smoke_like = dry or mode == "smoke"
kind = "epm:smoke-result" if smoke_like else "epm:results"
try:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root, check=True
    ).stdout.strip()
except Exception:
    sha = "unknown"
eval_paths = sorted(
    str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)
    for p in eval_dir.rglob("*.json")
)
note = {
    "mode": mode,
    "dry_run": dry,
    "ngpu": ngpu,
    "git_commit": sha,
    "gates": {
        "G-PARITY": "PASS",
        "G-LENS": "PASS" if lens_ok else "FAIL (rc=8; dict/phase4/5b legs skipped)",
        "G-NONZERO": "PASS",
    },
    "eval_json_paths": eval_paths,
    "hf_prefixes": {
        "analysis_tensors": f"{hf_prefix}/analysis_tensors",
        "raw_completions_steered": f"{hf_prefix}/raw_completions/steered",
        "wildchat_fresh": f"{hf_prefix}/wildchat_fresh",
    },
    "shipped_m_reference": shipped_m or (
        "(dry-run: not staged)"
        if dry
        else "OMITTED — authorized via --allow-missing-reference-arm; omission recorded at "
        "phase2/shipped_m_reference_omitted.json (transfer + battery ran without the "
        "m_shipped reference arm)"
    ),
    "plan_deviation": (
        (
            "(a) p5b leakage re-read SKIPPED pod-side (EPS_1776_P5B_OFFPOD=1) — run "
            "off-pod per offpod_handoffs.p5b_leakage, matching plan §9's p7 placement; "
            if p5b_offpod
            else "(a) p5b leakage re-read ran POD-side (plan §9 lists it off-pod p7): all "
            "inputs (centroids + L21 dict) were staged/built here, avoiding a second "
            "staging pass — env-flag EPS_1776_P5B_OFFPOD=1 selects the plan-literal "
            "off-pod lane instead; analyzer carries this as an ops/scope caveat; "
        )
        + "(b) 5c lens-vocab + 5d chain reads moved OFF-POD to the "
        "p7 handoff (plan §9 lists chain_composition.json as a pod p5 output) — lens + "
        "dictionaries are on HF, the reads are 0-GPU"
    ),
    "offpod_handoffs": {
        "p6_judge": (
            "OFF-POD (VM, Batch API): uv run python scripts/issue1776_judge.py "
            f"--raw-dir <staged {hf_prefix}/raw_completions/steered> "
            "--out-dir eval_results/issue_1776/phase3/judge. Pass --include-allpos "
            "to ALSO judge the exploratory all_positions strata (wired in this "
            "run: 50-context subset; judged rows add ~5 strata x 50 ctx x K=5). "
            "PRICING: the DEFAULT is the plan-S9-priced contrast policy (trait "
            "strata under their own rubric; baseline under every trait rubric — "
            "the a=0 term of each registered contrast; w1_mprime/random contexts "
            "one rubric each, round-robin ~= 18k x 5 calls). Opt into the "
            "all-rubrics mode (~30k x 5) via --all-control-rubrics; an explicit "
            "--control-rubrics list still overrides."
        ),
        "p5b_leakage": (
            (
                "OFF-POD (VM, 0 GPU — pod-side leg was SKIPPED via EPS_1776_P5B_OFFPOD=1): "
                if p5b_offpod
                else "ALREADY RAN POD-SIDE (default; recorded deviation above). Off-pod "
                "equivalent, either lane works: "
            )
            + "uv run python scripts/issue1776_phase5.py leakage "
            f"--centroids <staged {centroids}> --dict <staged {dict_dir}/dictionary_l21.pt> "
            f"--n-boot {nboot} --out eval_results/issue_1776/phase5/leakage_reread.json "
            "(inputs: centroids bundle sha-pinned via issue483_canonical_persona_pool/"
            "centroids_v1_L21.pt; L21 dictionary uploaded under "
            f"{hf_prefix}/analysis_tensors)"
        ),
        "p7_final_analyses": "OFF-POD (VM, 0 GPU): 5c word tables + 5d chain reads (lens + dictionaries on HF)",
    },
    "wandb": "n/a (no training this run)",
}
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": int(issue),
    "gate": "smoke" if smoke_like else "results",
    "blocks_pipeline": not smoke_like,
    "by": "issue1776_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": note,
}
log_dir.mkdir(parents=True, exist_ok=True)
path = log_dir / f"issue-{issue}-{kind.replace(':', '_')}-{int(time.time())}.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=1))
os.replace(tmp, path)
print(f"[dispatch] results sentinel: {path}")
PY

# ── dry-run self-checks: trace order + sentinel round-trip ────────────────────
if [[ $DRY == 1 ]]; then
  uv run python - "$TRACE" "$LOG_DIR" "$ISSUE" <<'PY'
import json
import sys
from pathlib import Path

EXPECTED = [
    "p0_stage", "p5a_stream_launch", "p0_parity", "p0_comparator_launch",
    "p0_jlens", "p0_comparator_join", "p0_dict", "p04_pairs", "p1_contexts",
    "p2a_sketch", "p1_diag", "p_early_upload", "p2b_full", "p2_upload",
    "p2c_battery", "p3_grid", "p3_upload", "p4_mediation", "p5a_stream_join",
    "p5a_capture", "p5_transfer", "p5b_leakage", "p_results_commit", "p_final",
]
trace_path, log_dir, issue = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
trace = [ln.strip() for ln in trace_path.read_text().splitlines() if ln.strip()]
assert trace == EXPECTED, f"phase-order mismatch:\n got={trace}\n want={EXPECTED}"
REQUIRED = {"sentinel_schema_version", "kind", "version", "note"}
sents = sorted(log_dir.glob(f"issue-{issue}-*.json"))
assert sents, f"no sentinels under {log_dir}"
kinds = {}
for s in sents:
    obj = json.loads(s.read_text())
    missing = REQUIRED - set(obj)
    assert not missing, f"{s.name} missing required keys {missing}"
    assert obj["sentinel_schema_version"] == 1 and isinstance(obj["version"], int), s.name
    kinds.setdefault(obj["kind"], 0)
    kinds[obj["kind"]] += 1
assert kinds.get("epm:smoke-result", 0) == 1, kinds  # dry-run terminal sentinel
final = json.loads(
    max(log_dir.glob(f"issue-{issue}-epm_smoke-result-*.json")).read_text()
)
note = final["note"]
for key in ("eval_json_paths", "hf_prefixes", "offpod_handoffs", "gates", "wandb", "git_commit"):
    assert key in note, f"final sentinel note missing §10 field: {key}"
assert "control-rubrics" in note["offpod_handoffs"]["p6_judge"], "judge pricing handoff missing"
assert "issue1776_phase5.py leakage" in note["offpod_handoffs"]["p5b_leakage"], (
    "p5b off-pod invocation handoff missing"
)
assert "p5b" in note["plan_deviation"], "p5b deviation note missing"
print(f"DRY-RUN-OK: {len(trace)} phases in §9 order; {len(sents)} sentinels parse "
      f"({sum(kinds.values())} total: {kinds})")
PY
fi

# Designed-halt convention: a G-LENS-degraded run does NOT emit [phase=done] —
# the gate sentinel + the results sentinel (gates.G-LENS=FAIL) carry the state
# and the distinct rc=8 routes it (never a bare rc=1 anonymous crash).
if [[ $FINAL_RC -eq 0 ]]; then
  echo "[phase=done]"
else
  echo "[phase=gate_halted]"
fi
exit "$FINAL_RC"
