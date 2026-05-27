#!/usr/bin/env bash
# Launcher for the #389 2-persona follow-up — end-to-end pipeline on a 4× H100
# pod. Mirrors the phase progression baked into `run_experiment_389_2persona.py`
# (preflight → dataset-gen → phase0-calibration → base-eval → train waves →
# full-eval → aggregate → upload). Per-phase output paths persist to disk
# immediately so a mid-run crash on any phase recovers cleanly on re-run.
#
# Pod-side usage (after the orchestrator has provisioned, bootstrapped, synced
# code to the `issue-389` branch, and shipped this script to /workspace/):
#
#   nohup bash /workspace/launch_issue_389_2persona.sh \
#       >> /workspace/logs/issue-389-2persona/launch.log 2>&1 &
#
# All phases are idempotent. To re-run a single phase, delete its artifact
# (e.g. `eval_results/issue_389/2persona_followup/<file>.json`) and re-invoke
# the driver. Train waves require ≥ 3 GPUs; the eval phases run on GPU 0.
#
# This file is committed to the repo (NOT generated at provision time) so the
# Reproducibility card can pin the exact launch sequence by commit SHA.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/explore-persona-space}
LOG_DIR=${LOG_DIR:-/workspace/logs/issue-389-2persona}
DRIVER="scripts/run_experiment_389_2persona.py"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "launching #389 2-persona follow-up pipeline from $REPO_DIR"
log "log dir: $LOG_DIR"
log "git HEAD: $(git rev-parse HEAD)"
log "git branch: $(git rev-parse --abbrev-ref HEAD)"

# ── Phase 1: preflight ───────────────────────────────────────────────────────
log "[1/8] preflight"
uv run python "$DRIVER" --phase preflight --gpu-id 0 \
    2>&1 | tee -a "$LOG_DIR/01_preflight.log"

# ── Phase 2: dataset-gen ─────────────────────────────────────────────────────
log "[2/8] dataset-gen (all seeds)"
uv run python "$DRIVER" --phase dataset-gen \
    2>&1 | tee -a "$LOG_DIR/02_dataset_gen.log"

# ── Phase 3: phase0-calibration ──────────────────────────────────────────────
# Base completions + A/B/C categorical + 11-framing FP rates. Hard-gates on
# base preference > 0.20 and inherited-panel FP > 0.30. C-family judged under
# the STRICT rubric (literal-mention requirement).
log "[3/8] phase0-calibration (base completions + judge calibration)"
uv run python "$DRIVER" --phase phase0-calibration --gpu-id 0 \
    2>&1 | tee -a "$LOG_DIR/03_phase0_calibration.log"

# ── Phase 4: base-eval ───────────────────────────────────────────────────────
# Persist the unmodified-baseline cell. Reuses phase0 base completions when
# present (idempotent).
log "[4/8] base-eval (unmodified-baseline cell)"
uv run python "$DRIVER" --phase base-eval --gpu-id 0 \
    2>&1 | tee -a "$LOG_DIR/04_base_eval.log"

# ── Phase 5: train wave 1 (contradictory-predicates × 3 seeds, parallel) ─────
log "[5/8] train wave 1: contradictory-predicates × 3 seeds in parallel"
WAVE1_PIDS=()
for SEED_GPU in "42 0" "137 1" "256 2"; do
    SEED=${SEED_GPU% *}
    GPU=${SEED_GPU#* }
    log "  launching contradictory-predicates seed=$SEED gpu=$GPU"
    nohup uv run python "$DRIVER" \
        --phase train --condition contradictory-predicates --seed "$SEED" --gpu-id "$GPU" \
        >> "$LOG_DIR/05_train_contradictory_seed${SEED}.log" 2>&1 &
    WAVE1_PIDS+=($!)
done
log "  wave-1 PIDs: ${WAVE1_PIDS[*]}"
WAVE1_FAIL=0
for pid in "${WAVE1_PIDS[@]}"; do
    if wait "$pid"; then
        log "  wave-1 pid=$pid exited OK"
    else
        rc=$?
        log "  wave-1 pid=$pid FAILED (rc=$rc)"
        WAVE1_FAIL=1
    fi
done
if [ "$WAVE1_FAIL" -ne 0 ]; then
    log "ABORT: wave-1 contradictory-predicates training had a failure; see per-seed logs"
    exit 1
fi

# ── Phase 6: train wave 2 (reversed-assignment × 3 seeds, parallel) ──────────
log "[6/8] train wave 2: reversed-assignment × 3 seeds in parallel"
WAVE2_PIDS=()
for SEED_GPU in "42 0" "137 1" "256 2"; do
    SEED=${SEED_GPU% *}
    GPU=${SEED_GPU#* }
    log "  launching reversed-assignment seed=$SEED gpu=$GPU"
    nohup uv run python "$DRIVER" \
        --phase train --condition reversed-assignment --seed "$SEED" --gpu-id "$GPU" \
        >> "$LOG_DIR/06_train_reversed_seed${SEED}.log" 2>&1 &
    WAVE2_PIDS+=($!)
done
log "  wave-2 PIDs: ${WAVE2_PIDS[*]}"
WAVE2_FAIL=0
for pid in "${WAVE2_PIDS[@]}"; do
    if wait "$pid"; then
        log "  wave-2 pid=$pid exited OK"
    else
        rc=$?
        log "  wave-2 pid=$pid FAILED (rc=$rc)"
        WAVE2_FAIL=1
    fi
done
if [ "$WAVE2_FAIL" -ne 0 ]; then
    log "ABORT: wave-2 reversed-assignment training had a failure; see per-seed logs"
    exit 1
fi

# ── Phase 7: full-eval ───────────────────────────────────────────────────────
# vLLM batched generation per (condition, seed) cell + per-family judge.
# Sequential by design — vLLM holds ~60% of one GPU's HBM and merge-then-delete
# is in-loop. ~6 cells × 4-6 min eval each ≈ 30 min.
log "[7/8] full-eval (sequential, GPU 0)"
uv run python "$DRIVER" --phase full-eval --gpu-id 0 \
    2>&1 | tee -a "$LOG_DIR/07_full_eval.log"

# ── Phase 8: aggregate + upload ──────────────────────────────────────────────
log "[8/8a] aggregate"
uv run python "$DRIVER" --phase aggregate \
    2>&1 | tee -a "$LOG_DIR/08a_aggregate.log"

log "[8/8b] upload (raw completions → HF data repo)"
uv run python "$DRIVER" --phase upload \
    2>&1 | tee -a "$LOG_DIR/08b_upload.log"

log "✅ #389 2-persona follow-up pipeline complete."
log "Artifacts:"
log "  - eval_results/issue_389/2persona_followup/"
log "  - HF model repo:  superkaiba1/explore-persona-space (adapters/exp389_2persona-*)"
log "  - HF data repo:   superkaiba1/explore-persona-space-data/issue389_2persona/raw_completions/"
log "  - WandB project:  exp389-2persona-contradictory-predicates"
