#!/bin/bash
# Parallel launcher for Directed Trait Transfer experiment.
# Runs on Pod 4 (8x H100 SXM 80GB).
# Each phase is parallelized across GPUs, with waits between phases.

set -e
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
mkdir -p /workspace/tmp

SCRIPT=/workspace/directed_trait_transfer/run_experiment.py
LOGDIR=/workspace/directed_trait_transfer/logs
mkdir -p $LOGDIR

echo "[$(date)] Starting Directed Trait Transfer experiment"

# ── Phase 0: Data generation (single process) ──────────────────────────────
echo "[$(date)] Phase 0: Data generation"
python3 $SCRIPT --phase 0 > $LOGDIR/phase0.log 2>&1
echo "[$(date)] Phase 0 complete"

# ── Phase 1: Push SFT (3 conditions in parallel on GPUs 0-2) ───────────────
echo "[$(date)] Phase 1: Push SFT (3 conditions)"
python3 $SCRIPT --phase 1 --condition asst_near --gpu 0 > $LOGDIR/phase1_asst_near.log 2>&1 &
P1=$!
python3 $SCRIPT --phase 1 --condition asst_far --gpu 1 > $LOGDIR/phase1_asst_far.log 2>&1 &
P2=$!
python3 $SCRIPT --phase 1 --condition pirate_near --gpu 2 > $LOGDIR/phase1_pirate_near.log 2>&1 &
P3=$!

echo "[$(date)] Waiting for Phase 1 jobs (PIDs: $P1, $P2, $P3)..."
wait $P1 $P2 $P3
echo "[$(date)] Phase 1 complete"

# ── Verification Gate (single GPU) ─────────────────────────────────────────
echo "[$(date)] Verification gate"
python3 $SCRIPT --phase verify --gpu 0 > $LOGDIR/verify.log 2>&1
echo "[$(date)] Verification complete"

# Check gate result
if grep -q "VERIFICATION GATE FAILED" $LOGDIR/verify.log; then
    echo "[$(date)] WARNING: Verification gate failed. Proceeding anyway to collect data."
fi

# ── Pre-Phase-2 Sanity (single GPU) ───────────────────────────────────────
echo "[$(date)] Pre-Phase-2 sanity checks"
python3 $SCRIPT --phase sanity --gpu 0 > $LOGDIR/sanity.log 2>&1
echo "[$(date)] Sanity checks complete"

# ── Phase 2a: Contrastive Marker (4 conditions on GPUs 0-3) ───────────────
echo "[$(date)] Phase 2a: Contrastive marker (4 conditions)"
python3 $SCRIPT --phase 2a --condition asst_near_marker --gpu 0 > $LOGDIR/phase2a_asst_near.log 2>&1 &
PA1=$!
python3 $SCRIPT --phase 2a --condition asst_far_marker --gpu 1 > $LOGDIR/phase2a_asst_far.log 2>&1 &
PA2=$!
python3 $SCRIPT --phase 2a --condition pirate_near_marker --gpu 2 > $LOGDIR/phase2a_pirate_near.log 2>&1 &
PA3=$!
python3 $SCRIPT --phase 2a --condition nopush_marker --gpu 3 > $LOGDIR/phase2a_nopush.log 2>&1 &
PA4=$!

echo "[$(date)] Waiting for Phase 2a jobs..."
wait $PA1 $PA2 $PA3 $PA4
echo "[$(date)] Phase 2a complete"

# ── Phase 2b: EM Induction (4 conditions on GPUs 4-7) ─────────────────────
echo "[$(date)] Phase 2b: EM induction (4 conditions)"
python3 $SCRIPT --phase 2b --condition asst_near_em --gpu 4 > $LOGDIR/phase2b_asst_near.log 2>&1 &
PB1=$!
python3 $SCRIPT --phase 2b --condition asst_far_em --gpu 5 > $LOGDIR/phase2b_asst_far.log 2>&1 &
PB2=$!
python3 $SCRIPT --phase 2b --condition pirate_near_em --gpu 6 > $LOGDIR/phase2b_pirate_near.log 2>&1 &
PB3=$!
python3 $SCRIPT --phase 2b --condition nopush_em --gpu 7 > $LOGDIR/phase2b_nopush.log 2>&1 &
PB4=$!

echo "[$(date)] Waiting for Phase 2b jobs..."
wait $PB1 $PB2 $PB3 $PB4
echo "[$(date)] Phase 2b complete"

# ── Eval Arm A: Marker Detection (4 conditions on GPUs 0-3) ───────────────
echo "[$(date)] Eval Arm A: Marker detection (4 conditions)"
python3 $SCRIPT --phase eval_a --condition asst_near_marker --gpu 0 > $LOGDIR/eval_a_asst_near.log 2>&1 &
EA1=$!
python3 $SCRIPT --phase eval_a --condition asst_far_marker --gpu 1 > $LOGDIR/eval_a_asst_far.log 2>&1 &
EA2=$!
python3 $SCRIPT --phase eval_a --condition pirate_near_marker --gpu 2 > $LOGDIR/eval_a_pirate_near.log 2>&1 &
EA3=$!
python3 $SCRIPT --phase eval_a --condition nopush_marker --gpu 3 > $LOGDIR/eval_a_nopush.log 2>&1 &
EA4=$!

echo "[$(date)] Waiting for Eval Arm A jobs..."
wait $EA1 $EA2 $EA3 $EA4
echo "[$(date)] Eval Arm A complete"

# ── Eval Arm B: Alignment (4 conditions on GPUs 4-7, generation only) ─────
# Generation is GPU-bound, judging is API-bound
echo "[$(date)] Eval Arm B: Alignment (4 conditions)"
python3 $SCRIPT --phase eval_b --condition asst_near_em --gpu 4 > $LOGDIR/eval_b_asst_near.log 2>&1 &
EB1=$!
python3 $SCRIPT --phase eval_b --condition asst_far_em --gpu 5 > $LOGDIR/eval_b_asst_far.log 2>&1 &
EB2=$!
python3 $SCRIPT --phase eval_b --condition pirate_near_em --gpu 6 > $LOGDIR/eval_b_pirate_near.log 2>&1 &
EB3=$!
python3 $SCRIPT --phase eval_b --condition nopush_em --gpu 7 > $LOGDIR/eval_b_nopush.log 2>&1 &
EB4=$!

echo "[$(date)] Waiting for Eval Arm B jobs..."
wait $EB1 $EB2 $EB3 $EB4
echo "[$(date)] Eval Arm B complete"

echo "[$(date)] === ALL PHASES COMPLETE ==="
echo "[$(date)] Results in /workspace/directed_trait_transfer/eval/"
