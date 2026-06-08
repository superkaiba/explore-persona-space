#!/bin/bash
# Task #504 round-15 launcher — strengthen-anchor re-run.
#
# Purpose (from user-directive 2026-06-08T04:53:14Z):
#   The round-13/14 dispositive A/B test refuted the rig-bug hypothesis but
#   left the smoke adapter (c504_smoke_r{4,8,16}) under-trained: source ΔG
#   ~0.01 nats / on-policy emission 0.0 at the 1-epoch / 25-step budget.
#   #477's r=8/count=2 mid-band cell (ΔG=9.3 nats) ran at ~75 steps. This
#   launcher re-runs Phase 0 calibration AND the Phase 1 main sweep at the
#   STRENGTHENED budget (EPOCHS=3 at the same 200-pos/200-neg composition →
#   75 optimizer steps per cell) at the SAME knobs (r∈{4,8,16}, α from
#   RANK_ALPHA_MAP_V5, lr=2e-6).
#
# Adapter-path isolation:
#   --hf-path-suffix __r15 decorates BOTH the local /workspace/runs/ subdir
#   AND the HF model-repo subfolder (`adapters/issue_504/<slug>_seed<S>__r15`)
#   so the round-13/14 floor-anchor adapters at the canonical un-suffixed path
#   are preserved on HF as dispositive-A/B evidence.
#
# Pre-flight (on pod):
#   * git pull on issue-504 branch
#   * uv sync (env up to date)
#   * the round-13/14 reval_confirm/ output on /workspace stays untouched
#
# Run (on pod):
#   cp scripts/launchers/launch_issue_504_round15.sh /workspace/launch_issue_504.sh
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
echo "[launcher round-15] Stage 1: SMOKE — Phase 0.5 + Phase 0 calibration"
echo "[launcher round-15] STRENGTHENED ANCHOR: EPOCHS=3 (75 steps at 200/200 composition)"
echo "[launcher round-15] HF path suffix: __r15 (preserves round-13/14 adapters)"
echo "=========================================="
uv run python scripts/dispatch_neg_geometry_504.py --smoke --hf-path-suffix __r15

# Gate: check Phase 0 calibration verdict before main sweep.
GATES_JSON=eval_results/issue_504/phase0_5_gates.json
CALIB_JSON=eval_results/issue_504/phase0_calibration.json

if [ ! -f "$GATES_JSON" ]; then
  echo "[launcher round-15] FATAL: $GATES_JSON missing after smoke — aborting before sweep" >&2
  exit 2
fi
GATES_VERDICT=$(uv run python -c "import json; print(json.load(open('$GATES_JSON'))['verdict'])")
echo "[launcher round-15] Phase 0.5 verdict: $GATES_VERDICT"
if [ "$GATES_VERDICT" != "pass" ]; then
  echo "[launcher round-15] FATAL: Phase 0.5 verdict != pass — aborting before sweep" >&2
  exit 2
fi

if [ ! -f "$CALIB_JSON" ]; then
  echo "[launcher round-15] FATAL: $CALIB_JSON missing after smoke — aborting before sweep" >&2
  exit 2
fi

# Quick visibility: log the picked anchor + observed source ΔG before sweep launches.
uv run python -c "
import json
c = json.load(open('$CALIB_JSON'))
print('[launcher round-15] Phase 0 pick: rank={r} alpha={a} frac={f} source_dg={dg} nats'.format(
    r=c.get('chosen_rank'),
    a=c.get('chosen_alpha'),
    f=c.get('chosen_checkpoint_fraction'),
    dg=c.get('source_delta_g_at_pick_nats'),
))
"

echo "=========================================="
echo "[launcher round-15] Stage 2: SWEEP — Phase 1 main grid (5 arms x 2 seeds) + Phase 2 analyze"
echo "=========================================="
exec uv run python scripts/dispatch_neg_geometry_504.py \
    --skip-phase05 --skip-phase0 --hf-path-suffix __r15
