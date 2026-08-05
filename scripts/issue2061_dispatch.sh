#!/usr/bin/env bash
# scripts/issue2061_dispatch.sh — end-to-end orchestrator for task #2061.
#
# Sequences the 5 pipeline phases per plan §9 P1-P5 with the cross-machine
# HF staging/upload legs (plan §9 off_pod_phases; scripts/issue2061_hub_io.py)
# and the pod-side reporting contract (.claude/rules/pod-side-reporting.md):
# one `[phase=<name>]` line per phase, an end-of-run results sentinel via
# `issue2061_hub_io.py sentinel`, and a terminal `[phase=done]` emitted ONLY
# by this dispatcher, immediately before its normal exit.
#
# Phase routing (plan §9 P1-P5; each production phase runs on ITS OWN
# provision — every cross-machine dependency rides the HF data repo):
#   P1 encode          - GPU pod (eval intent); uploads sae_encoded/ pre-terminate
#   P2 per-feature fit - cpu-bigmem; stages P1+#1336 per-shard STREAM-FETCH-DELETE,
#                        uploads analysis_tensors/per_feature_r2/ pre-terminate
#   P3 null battery    - 8x GPU (fellows pin / RunPod fallback); lazy SHARED
#                        per-pod turnstore cache, uploads analysis_tensors/null/
#   P4 fitness gate    - GPU pod (eval intent, own provision); uploads
#                        analysis_tensors/fitness/ pre-terminate (v6, #1738 class)
#   P5 figures         - VM-local; fetches P2/P3/P4 outputs from the HF data repo
#
# Modes:
#   --smoke-only   Full tiny-N pipeline P1->P5 (parity gate, 1 stage-pair x 1
#                  render x smallest corpus, --n-draws 4) into ISSUE2061_SMOKE_ROOT
#                  (never the canonical out-roots); sentinel kind epm:smoke-result.
#   --dry-run      Walk every phase printing the composed command without
#                  executing (cell-iteration plumbing + env passthrough +
#                  sentinel writer + [phase=done], zero GPU/network).
#   --phase pN     One production phase (the per-machine form). p3 accepts
#                  --gpus N for the 8-wide fan-out (one CVD-pinned single-GPU
#                  worker per GPU; plan §9 per-GPU-phase parallelization).
#   --all          P1->P5 sequentially on one filesystem (bounded runs only).
#
# Env vars honored (all optional):
#   ISSUE2061_STAGE / ISSUE2061_RENDER / ISSUE2061_CORPUS / ISSUE2061_ARM — cell filters
#   ISSUE2061_CONTEXT_SHARD_DIR — #1336 shards already staged locally; when set,
#                  hub staging is DISABLED (local mode); when unset, phases
#                  stage their inputs from the HF data repo (hub mode)
#   ISSUE2061_UPLOAD=0 — disable the per-phase HF upload legs (default ON)
#   ISSUE2061_HF_PREFIX — override the HF bucket (scratch-prefix smoke probes)
#   ISSUE2061_STAGING_DIR / ISSUE2061_SMOKE_ROOT / ISSUE2061_SENTINEL_DIR
#   ISSUE2061_SAE_REVISION / ISSUE2061_DATA_REVISION — HF pins
#   ISSUE2061_N_DRAWS / ISSUE2061_P3_DEVICE — P3 knobs (defaults: script / auto)
#
# Usage:
#   bash scripts/issue2061_dispatch.sh --smoke-only
#   bash scripts/issue2061_dispatch.sh --dry-run --all
#   bash scripts/issue2061_dispatch.sh --phase p3 --gpus 8
#   bash scripts/issue2061_dispatch.sh --all

set -euo pipefail

# ─── thread caps (plan §9; .claude/rules/code-style.md § Shared-VM CPU) ─────
# On cpu-bigmem (16 vCPU dedicated), the pod's own env sets these to 16.
# On the shared VM (fallback for tiny debug runs), cap at 8 to leave
# headroom for concurrent sessions. P3 fan-out workers override to 4/worker.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ─── paths + modes ───────────────────────────────────────────────────────────
DATA_ROOT="${ISSUE2061_DATA_ROOT:-data/issue_2061}"
EVAL_ROOT="${ISSUE2061_EVAL_ROOT:-eval_results/issue_2061}"
FIG_ROOT="${ISSUE2061_FIG_ROOT:-figures/issue_2061}"
STAGING_DIR="${ISSUE2061_STAGING_DIR:-$DATA_ROOT/hf_dl}"
ENCODED_DIR="$DATA_ROOT/sae_encoded"
R2_DIR="$EVAL_ROOT/per_feature_r2"
NULL_DIR="$EVAL_ROOT/null"
FITNESS_DIR="$EVAL_ROOT/fitness"
P3_LOG_DIR="$EVAL_ROOT/p3_logs" # OUTSIDE $NULL_DIR: the null upload is whole-dir

UPLOAD="${ISSUE2061_UPLOAD:-1}"
if [[ -n "${ISSUE2061_CONTEXT_SHARD_DIR:-}" ]]; then STAGE_MODE=local; else STAGE_MODE=hub; fi
DRY=0

# ─── helpers ─────────────────────────────────────────────────────────────────
run() {
  # Echo the composed command, then execute (skipped under --dry-run, which
  # exists to verify the composed argv / cell-iteration plumbing at zero cost).
  echo "+ $*"
  if [[ "$DRY" != "1" ]]; then "$@"; fi
}

p3_device() {
  if [[ -n "${ISSUE2061_P3_DEVICE:-}" ]]; then
    echo "$ISSUE2061_P3_DEVICE"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo cuda
  else
    echo cpu
  fi
}

finish() {
  # End-of-run sentinel + the RESERVED terminal token, in that order
  # (.claude/rules/pod-side-reporting.md items 1-2). `[phase=done]` is
  # emitted ONLY here; per-phase completions never carry the token. The
  # sentinel writer runs EVEN under --dry-run (cheap, local, and the
  # dry-run exists to exercise exactly this contract) — but a dry-run
  # sentinel is ALWAYS kind epm:smoke-result, never a real epm:results
  # (the VM drain would post it as a real results marker otherwise).
  local kind="$1" note="$2"
  if [[ "$DRY" == "1" ]]; then
    kind="epm:smoke-result"
    note="{\"dry_run\": true, \"walked\": $note}"
  fi
  echo "+ issue2061_hub_io.py sentinel --kind $kind"
  uv run python scripts/issue2061_hub_io.py sentinel --kind "$kind" --note "$note"
  echo "[phase=done]"
}

cell_filter_args() {
  # Shared cell filters -> nameref array (defaults to --all-cells).
  local -n _out="$1"
  _out=()
  [[ -n "${ISSUE2061_STAGE:-}" ]] && _out+=(--stage "$ISSUE2061_STAGE")
  [[ -n "${ISSUE2061_RENDER:-}" ]] && _out+=(--render "$ISSUE2061_RENDER")
  [[ -n "${ISSUE2061_CORPUS:-}" ]] && _out+=(--corpus "$ISSUE2061_CORPUS")
  if [[ ${#_out[@]} -eq 0 ]]; then _out+=(--all-cells); fi
  return 0
}

# ─── production phase runners ────────────────────────────────────────────────
run_p1_encode() {
  echo "[phase=p1_encode]"
  mkdir -p "$ENCODED_DIR"
  local filters=()
  cell_filter_args filters
  local args=(--smoke-then-encode "${filters[@]}" --output-dir "$ENCODED_DIR")
  [[ -n "${ISSUE2061_SAE_REVISION:-}" ]] && args+=(--sae-revision "$ISSUE2061_SAE_REVISION")
  [[ -n "${ISSUE2061_DATA_REVISION:-}" ]] && args+=(--data-revision "$ISSUE2061_DATA_REVISION")
  [[ "$UPLOAD" == "1" ]] && args+=(--upload)
  run uv run python scripts/issue2061_sae_encode.py "${args[@]}"
}

run_p2_fit() {
  echo "[phase=p2_fit]"
  mkdir -p "$R2_DIR"
  local filters=()
  cell_filter_args filters
  local args=("${filters[@]}" --output-dir "$R2_DIR")
  [[ -n "${ISSUE2061_ARM:-}" ]] && args+=(--arm "$ISSUE2061_ARM")
  if [[ "$STAGE_MODE" == "hub" ]]; then
    # Plan §9 P2 staging (v6): encoded targets whole (tiny, TopK-sparse);
    # turnstores per-shard STREAM-FETCH-DELETE inside the cell loop.
    args+=(--stage-encoded-from-hub --stage-context-from-hub --staging-dir "$STAGING_DIR")
  else
    args+=(--context-shard-dir "$ISSUE2061_CONTEXT_SHARD_DIR" --encoded-dir "$ENCODED_DIR")
  fi
  [[ -n "${ISSUE2061_DATA_REVISION:-}" ]] && args+=(--data-revision "$ISSUE2061_DATA_REVISION")
  [[ "$UPLOAD" == "1" ]] && args+=(--upload)
  run uv run python scripts/issue2061_fit_per_feature.py "${args[@]}"
}

p3_common_args() {
  local -n _out="$1"
  _out=(--output-dir "$NULL_DIR" --device "$(p3_device)")
  [[ -n "${ISSUE2061_N_DRAWS:-}" ]] && _out+=(--n-draws "$ISSUE2061_N_DRAWS")
  [[ -n "${ISSUE2061_DATA_REVISION:-}" ]] && _out+=(--data-revision "$ISSUE2061_DATA_REVISION")
  return 0
}

run_p3_null() {
  # Single-process form: computes any missing cells, then the GLOBAL null
  # (skip-if-exists reload makes it the idempotent aggregation pass after a
  # fan-out). Round-1 gap closed: the refit inputs (--context-shard-dir /
  # --encoded-dir or their hub-staging twins) are ALWAYS wired.
  echo "[phase=p3_null]"
  mkdir -p "$NULL_DIR"
  local common=()
  p3_common_args common
  # Production aggregation guard (review m2): the registered statistic rides
  # a fixed 56-cell axis (plan §Design, v7 grid — 4 stage-pairs x 7 v2
  # (render, corpus) combos x 2 arms), so the GLOBAL pass fails loud on a
  # partial P2/P3 instead of writing a silently-shrunk GLOBAL_L29.json.
  # Deliberate sub-grid runs override via ISSUE2061_EXPECT_N_CELLS. The smoke
  # chain (run_smoke) invokes issue2061_null.py directly and never inherits
  # this — the #1345 smoke-gate-calibration discipline.
  local args=(--all-cells --expect-n-cells "${ISSUE2061_EXPECT_N_CELLS:-56}" "${common[@]}")
  if [[ "$STAGE_MODE" == "hub" ]]; then
    args+=(--stage-r2-from-hub --stage-encoded-from-hub --stage-context-from-hub
      --staging-dir "$STAGING_DIR")
  else
    args+=(--r2-dir "$R2_DIR" --encoded-dir "$ENCODED_DIR"
      --context-shard-dir "$ISSUE2061_CONTEXT_SHARD_DIR")
  fi
  [[ "$UPLOAD" == "1" ]] && args+=(--upload)
  run uv run python scripts/issue2061_null.py "${args[@]}"
}

run_p3_fanout() {
  # Plan §9 per-GPU-phase parallelization: N single-GPU worker processes, one
  # per GPU via CUDA_VISIBLE_DEVICES pinning (the gotchas.md CVD discipline),
  # each running issue2061_null.py over a worker-DISJOINT (render, corpus)
  # combo subset (largest-first round-robin), so the shared staging cache's
  # per-corpus reap is process-local and race-free. Workers write per-cell
  # JSONLs only (--skip-global); the aggregation pass (run_p3_null) owns
  # GLOBAL_L29.json + the upload.
  local ngpu="$1"
  echo "[phase=p3_stage_inputs]"
  mkdir -p "$NULL_DIR" "$P3_LOG_DIR"
  local r2_in enc_in ctx_args
  if [[ "$STAGE_MODE" == "hub" ]]; then
    if [[ "$DRY" == "1" ]]; then
      echo "+ (dry-run) stage per-feature-r2 + sae-encoded -> $STAGING_DIR"
      r2_in="$STAGING_DIR/per-feature-r2"
      enc_in="$STAGING_DIR/sae-encoded"
    else
      r2_in=$(uv run python scripts/issue2061_hub_io.py stage --what per-feature-r2 --root "$STAGING_DIR")
      enc_in=$(uv run python scripts/issue2061_hub_io.py stage --what sae-encoded --root "$STAGING_DIR")
    fi
    ctx_args=(--stage-context-from-hub --staging-dir "$STAGING_DIR")
  else
    r2_in="$R2_DIR"
    enc_in="$ENCODED_DIR"
    ctx_args=(--context-shard-dir "$ISSUE2061_CONTEXT_SHARD_DIR")
  fi

  echo "[phase=p3_fanout]"
  if [[ "$DRY" == "1" ]]; then
    echo "+ (dry-run) p3-combos --r2-dir $r2_in; ${ngpu}x CVD-pinned single-GPU workers"
  else
    local combos_txt
    combos_txt=$(uv run python scripts/issue2061_hub_io.py p3-combos --r2-dir "$r2_in" --encoded-dir "$enc_in")
    local combos=()
    mapfile -t combos <<<"$combos_txt"
    echo "[p3] ${#combos[@]} (render, corpus) combos across $ngpu GPU worker(s)"
    local pids=() logs=() g idx
    for ((g = 0; g < ngpu; g++)); do
      local worker_combos=()
      for idx in "${!combos[@]}"; do
        if ((idx % ngpu == g)); then worker_combos+=("${combos[$idx]}"); fi
      done
      [[ ${#worker_combos[@]} -eq 0 ]] && continue
      local log="$P3_LOG_DIR/p3_gpu${g}.log"
      echo "[p3] worker gpu$g: ${#worker_combos[@]} combo(s) -> $log"
      (
        set -euo pipefail
        common=()
        p3_common_args common
        for combo in "${worker_combos[@]}"; do
          read -r render corpus <<<"$combo"
          wargs=(--render "$render" --corpus "$corpus" --skip-global
            --r2-dir "$r2_in" --encoded-dir "$enc_in" "${ctx_args[@]}" "${common[@]}")
          # Per-worker thread caps (plan §9: avoid 8-worker host oversubscription).
          CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
            OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
            uv run python scripts/issue2061_null.py "${wargs[@]}"
        done
      ) >"$log" 2>&1 &
      pids+=($!)
      logs+=("$log")
    done
    local fail=0 i
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "[p3] worker $i FAILED — tail of ${logs[$i]}:" >&2
        tail -n 120 "${logs[$i]}" >&2
        fail=1
      fi
    done
    if [[ $fail -ne 0 ]]; then
      echo "[p3] fan-out failed — NOT running the aggregation pass" >&2
      return 1
    fi
  fi
  # Aggregation pass: skip-if-exists reload of every per-cell JSONL, GLOBAL
  # null + upload. Work-conserving fallback: a combo a worker missed gets
  # computed here serially rather than lost.
  run_p3_null
}

run_p4_fitness() {
  echo "[phase=p4_fitness]"
  mkdir -p "$FITNESS_DIR"
  local args=(--all-stages --output-dir "$FITNESS_DIR")
  [[ -n "${ISSUE2061_SAE_REVISION:-}" ]] && args+=(--sae-revision "$ISSUE2061_SAE_REVISION")
  [[ -n "${ISSUE2061_DATA_REVISION:-}" ]] && args+=(--data-revision "$ISSUE2061_DATA_REVISION")
  [[ "$UPLOAD" == "1" ]] && args+=(--upload)
  run uv run python scripts/issue2061_fitness.py "${args[@]}"
}

run_p5_figures() {
  # from_hub=1 (the standalone VM form): fetch P2/P3/P4 outputs from the HF
  # data repo per plan §9 off_pod_phases P5 reads. from_hub=0 (--all chain):
  # read the dirs the earlier phases just wrote on this filesystem.
  local from_hub="$1"
  echo "[phase=p5_figures]"
  mkdir -p "$FIG_ROOT"
  local args=(--all --output-dir "$FIG_ROOT")
  if [[ "$from_hub" == "1" ]]; then
    args+=(--stage-from-hub --staging-dir "$STAGING_DIR")
  else
    args+=(--r2-dir "$R2_DIR" --null-dir "$NULL_DIR" --fitness-dir "$FITNESS_DIR")
  fi
  run uv run python scripts/issue2061_figures.py "${args[@]}"
}

# ─── smoke mode (review M6): the FULL tiny-N pipeline, P1 -> P5 ─────────────
run_smoke() {
  # Same phase scripts, same flags, same staging seams as production — only
  # the cell subset (1 stage-pair x 1 render x smallest corpus), the draw
  # count, and the out-roots differ (smoke IS the pipeline with one cell;
  # /issue Step 6d.0 PASS_UNIFIED shape). Outputs land under $SMOKE_ROOT,
  # NEVER the canonical eval_results/figures trees.
  local SMOKE_ROOT="${ISSUE2061_SMOKE_ROOT:-/tmp/issue-2061-smoke}"
  local SA="${ISSUE2061_SMOKE_STAGE_A:-base}" SB="${ISSUE2061_SMOKE_STAGE_B:-sft}"
  local RD="${ISSUE2061_SMOKE_RENDER:-chat}" CP="${ISSUE2061_SMOKE_CORPUS:-gsm8k_test1319}"
  local DRAWS="${ISSUE2061_SMOKE_DRAWS:-4}"
  local enc="$SMOKE_ROOT/sae_encoded" r2="$SMOKE_ROOT/per_feature_r2"
  local null="$SMOKE_ROOT/null" fit="$SMOKE_ROOT/fitness" figs="$SMOKE_ROOT/figures"
  local stg="$SMOKE_ROOT/hf_dl"
  mkdir -p "$enc" "$r2" "$null" "$fit" "$figs" "$stg"

  local ctx_args
  if [[ "$STAGE_MODE" == "hub" ]]; then
    # --keep-staged: P3 reuses P2's staged turnstores under the shared smoke
    # staging root instead of re-fetching (the reap runs in production form).
    ctx_args=(--stage-context-from-hub --staging-dir "$stg" --keep-staged)
  else
    ctx_args=(--context-shard-dir "$ISSUE2061_CONTEXT_SHARD_DIR")
  fi

  echo "[phase=smoke_p1_parity]"
  run uv run python scripts/issue2061_sae_encode.py --smoke-only

  echo "[phase=smoke_p1_encode]"
  local st
  for st in "$SA" "$SB"; do
    run uv run python scripts/issue2061_sae_encode.py \
      --stage "$st" --render "$RD" --corpus "$CP" --output-dir "$enc"
  done

  echo "[phase=smoke_p2_fit]"
  run uv run python scripts/issue2061_fit_per_feature.py --all-cells \
    --encoded-dir "$enc" --output-dir "$r2" "${ctx_args[@]}"

  echo "[phase=smoke_p3_null]"
  run uv run python scripts/issue2061_null.py --all-cells \
    --r2-dir "$r2" --encoded-dir "$enc" --output-dir "$null" \
    --n-draws "$DRAWS" --draw-block 2 --device "$(p3_device)" "${ctx_args[@]}"

  echo "[phase=smoke_p4_fitness]"
  run uv run python scripts/issue2061_fitness.py --all-stages \
    --n-val-rows "${ISSUE2061_SMOKE_VAL_ROWS:-120}" --output-dir "$fit"

  echo "[phase=smoke_p5_figures]"
  run uv run python scripts/issue2061_figures.py --all \
    --r2-dir "$r2" --null-dir "$null" --fitness-dir "$fit" --output-dir "$figs"

  if [[ "${ISSUE2061_SMOKE_UPLOAD:-0}" == "1" ]]; then
    # Live probe of the upload leg against a SCRATCH prefix (#1769 hub-fenced-
    # branch discipline) — never the canonical bucket. Subshell so the
    # scratch-prefix override cannot leak past the probe.
    echo "[phase=smoke_upload_probe]"
    (
      export ISSUE2061_HF_PREFIX="${ISSUE2061_HF_PREFIX:-issue2061_sae_predictability}_smoke"
      run uv run python scripts/issue2061_hub_io.py upload --what fitness --dir "$fit"
    )
  fi

  finish "epm:smoke-result" \
    "{\"smoke\": true, \"phases\": [\"p1_parity\", \"p1_encode\", \"p2_fit\", \"p3_null\", \"p4_fitness\", \"p5_figures\"], \"cells\": \"${SA}+${SB}/${RD}/${CP}\", \"n_draws\": ${DRAWS}, \"smoke_root\": \"${SMOKE_ROOT}\"}"
}

# ─── argparse-lite ──────────────────────────────────────────────────────────
MODE=""
PHASE=""
GPUS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-only)
      MODE=smoke
      shift
      ;;
    --dry-run)
      DRY=1
      shift
      ;;
    --all)
      MODE=all
      shift
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    -h | --help)
      grep -E '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$DRY" == "1" && -z "$MODE" && -z "$PHASE" ]]; then
  MODE=all # bare --dry-run walks the full production chain
fi
if [[ -z "$MODE" && -z "$PHASE" ]]; then
  echo "ERROR: pass --smoke-only, --dry-run, --all, or --phase p1|p2|p3|p4|p5" >&2
  exit 2
fi

# ─── dispatch ───────────────────────────────────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
  echo "MODE=smoke-only (full tiny-N pipeline; stage_mode=$STAGE_MODE dry=$DRY)"
  run_smoke
  exit 0
fi

if [[ "$MODE" == "all" ]]; then
  echo "MODE=all (P1 -> P5; stage_mode=$STAGE_MODE upload=$UPLOAD dry=$DRY)"
  run_p1_encode
  run_p2_fit
  if [[ "$GPUS" -gt 1 ]]; then run_p3_fanout "$GPUS"; else run_p3_null; fi
  run_p4_fitness
  run_p5_figures 0
  finish "epm:results" \
    "{\"mode\": \"all\", \"phases\": [\"p1\", \"p2\", \"p3\", \"p4\", \"p5\"], \"eval_root\": \"$EVAL_ROOT\", \"hf_prefix\": \"${ISSUE2061_HF_PREFIX:-issue2061_sae_predictability}\", \"upload\": \"$UPLOAD\"}"
  exit 0
fi

case "$PHASE" in
  p1) run_p1_encode ;;
  p2) run_p2_fit ;;
  p3) if [[ "$GPUS" -gt 1 ]]; then run_p3_fanout "$GPUS"; else run_p3_null; fi ;;
  p4) run_p4_fitness ;;
  p5) run_p5_figures "$([[ "$STAGE_MODE" == "hub" ]] && echo 1 || echo 0)" ;;
  *)
    echo "ERROR: unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

finish "epm:results" \
  "{\"mode\": \"phase\", \"phase\": \"$PHASE\", \"eval_root\": \"$EVAL_ROOT\", \"hf_prefix\": \"${ISSUE2061_HF_PREFIX:-issue2061_sae_predictability}\", \"upload\": \"$UPLOAD\"}"
