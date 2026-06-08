#!/bin/bash
# Task #504 round-16 launcher — match #477's measured NEG composition.
#
# Purpose (autonomous strategy pivot, 2026-06-08):
#   Round-15 (EPOCHS=3 at 200pos+200neg) was launched on pod-504 and stage-1
#   produced these source ΔG trajectories at the picked recipe (r=8, α=32,
#   lr=2e-6):
#       step=6   step=12  step=25  step=38  step=57  step=75
#       0.089    0.135    0.113    0.115    0.112    0.105   (r=8)
#       0.023   -0.002    0.028    0.017    0.011    0.030   (r=4)
#   r=8 plateaus at ~0.11 nats — 50× below the target band [5, 12]. Emission
#   stays 0.0 throughout. The EPOCHS=3 hypothesis is REFUTED. #477's r=8/
#   count=2 cell hit 9.3 nats at the SAME ~75-step regime — the difference
#   is that #477 ran 200 pos + 400 neg (200 ex/persona × 2 personas), #504
#   r15 ran 200 pos + 200 neg (100 ex/persona × 2 personas).
#
#   Round-16 keeps EPOCHS=3 and bumps NEG_EX_PER_PERSONA 100→200 so each
#   positioned arm now has 200 pos + 200 ex/persona × 2 personas = 200 pos
#   + 400 neg = 600 rows. At batch_effective=16 the cell trains for
#   max_steps = int(3 × 600 / 16) = 112 optimizer steps, matching #477's
#   measured mid-band step regime (75 steps) with ~1.5× headroom. The
#   default-only arm bumps in lockstep to 400 ex from qwen_default so
#   cross-arm step counts stay equal.
#
#   Phase 0's pick rule picks the LATEST in-band checkpoint across the
#   6-frac trajectory, so any high-end saturation at the new step budget
#   is auto-handled by picking an earlier frac.
#
# Adapter-path isolation:
#   --hf-path-suffix __r16 decorates BOTH the local /workspace/runs/ subdir
#   AND the HF model-repo subfolder (`adapters/issue_504/<slug>_seed<S>__r16`)
#   so the round-13/14 floor-anchor adapters (un-suffixed canonical path) AND
#   the round-15 plateau adapters (__r15 subfolder) are preserved on HF as
#   dispositive-A/B/C evidence.
#
# Pre-flight (on pod, after orchestrator resumes pod-504):
#   * git pull on issue-504 branch
#   * uv sync (env up to date)
#   * the round-13/14 reval_confirm/ output on /workspace stays untouched
#   * the round-15 __r15 adapters stay on HF as evidence
#
# Run (on pod):
#   cp scripts/launchers/launch_issue_504_round16.sh /workspace/launch_issue_504.sh
#   chmod +x /workspace/launch_issue_504.sh
#   cd /workspace && nohup setsid /workspace/launch_issue_504.sh \
#       > /workspace/logs/issue-504-launch.log 2>&1 < /dev/null &
#
# Sentinel-file pattern (poll_pipeline.py compliant): the dispatcher writes
# /workspace/logs/issue-504-results.json with sentinel_schema_version=1.

set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
mkdir -p /workspace/logs eval_results/issue_504

# Write THIS shell's PID (which `exec` will replace with the smoke command).
echo $$ > /workspace/logs/issue-504.pid

echo "=========================================="
echo "[launcher round-16] Stage 1: SMOKE — Phase 0.5 + Phase 0 calibration"
echo "[launcher round-16] STRENGTHENED ANCHOR + NEG COMPOSITION MATCH:"
echo "[launcher round-16]   EPOCHS=3 + NEG_EX_PER_PERSONA=200 → 600 rows/cell"
echo "[launcher round-16]   max_steps=int(3*600/16)=112 (#477 mid-band: 75 steps)"
echo "[launcher round-16] HF path suffix: __r16 (preserves round-13/14/15 adapters)"
echo "=========================================="
uv run python scripts/dispatch_neg_geometry_504.py --smoke --hf-path-suffix __r16

# Gate: check Phase 0 calibration verdict before main sweep.
GATES_JSON=eval_results/issue_504/phase0_5_gates.json
CALIB_JSON=eval_results/issue_504/phase0_calibration.json

if [ ! -f "$GATES_JSON" ]; then
  echo "[launcher round-16] FATAL: $GATES_JSON missing after smoke — aborting before sweep" >&2
  exit 2
fi
GATES_VERDICT=$(uv run python -c "import json; print(json.load(open('$GATES_JSON'))['verdict'])")
echo "[launcher round-16] Phase 0.5 verdict: $GATES_VERDICT"
if [ "$GATES_VERDICT" != "pass" ]; then
  echo "[launcher round-16] FATAL: Phase 0.5 verdict != pass — aborting before sweep" >&2
  exit 2
fi

if [ ! -f "$CALIB_JSON" ]; then
  echo "[launcher round-16] FATAL: $CALIB_JSON missing after smoke — aborting before sweep" >&2
  exit 2
fi

# Quick visibility: log the picked anchor + observed source ΔG before sweep launches.
uv run python -c "
import json
c = json.load(open('$CALIB_JSON'))
print('[launcher round-16] Phase 0 pick: rank={r} alpha={a} frac={f} source_dg={dg} nats'.format(
    r=c.get('chosen_rank'),
    a=c.get('chosen_alpha'),
    f=c.get('chosen_checkpoint_fraction'),
    dg=c.get('source_delta_g_at_pick_nats'),
))
"

echo "=========================================="
echo "[launcher round-16] Stage 2: SWEEP — Phase 1 main grid (5 arms x 2 seeds) + Phase 2 analyze"
echo "=========================================="
# --skip-phase07 retained from round-15 loop-2 fix: stage 1 already ran
# Phase 0.7 with --smoke and produced R_train_v504.json / R_eval_v504.json
# at the canonical paths. The dispatcher's Phase 0.7 fast-path is keyed
# against the ORIGINAL #472 R inputs (not the stage-1 v504 outputs), so
# without --skip-phase07 stage 2 would re-fire vLLM Phase 0.7 (~30+ min
# wall waste) AND overwrite the stage-1 phase07 sentinel. The dispatcher's
# existence-assertion at :551-560 will fail-loud if either v504 artifact
# is missing.
exec uv run python scripts/dispatch_neg_geometry_504.py \
    --skip-phase05 --skip-phase0 --skip-phase07 --hf-path-suffix __r16
