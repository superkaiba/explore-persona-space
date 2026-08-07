#!/usr/bin/env bash
# Pod-side workload sequencer for issue #2163 (plan section 9 dispatch command:
#   dispatch_issue.py launch --issue 2163 --intent cpu-bigmem --repo-branch issue-2163 \
#     --boot-disk-gb 100 --min-ram-gb 80 --time-budget-hours 16 \
#     --workload-cmd "bash scripts/issue2163_pod_workload.sh").
#
# Thin sequencer ONLY: all logic (per-phase resume sentinels, pilot gates, per-phase
# headroom asserts, per-phase uploads) lives in scripts/issue2163_ctxread.py. A re-run
# of this script skips completed phases via the driver's own done-sentinels — no state
# layer here.
#
# Exit codes propagate unmasked (set -e): rc=7 is the driver's pilot-gate abort (an
# over-2x wall projection — pilot_gate_report.json is written + uploaded before the
# exit). The Phase-6 venue switch is a DIFFERENT signal: it exits 0 and defers the fit
# leg via out/results/confirm_B_venue.json (logged below for the orchestrator).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root, so scripts/ paths resolve

WORK_ROOT="${WORK_ROOT:-/root/issue2163_work}"
# RunPod CPU pods have NO separate /workspace volume: /root and /workspace share the
# single container disk sized by the launch's --boot-disk-gb 100.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p "$WORK_ROOT"
echo "[preamble] $(date -u +%Y-%m-%dT%H:%M:%SZ) WORK_ROOT=$WORK_ROOT OMP/MKL=16"
df -P "$WORK_ROOT"
# Plan section 9: peak footprint ~45 GB against the 100 GB container disk. The driver
# re-asserts per write-heavy phase; this is the up-front whole-run floor (statvfs +
# fallocate canary — catches an EDQUOT-exhausted quota statvfs is blind to).
uv run python -c '
import sys
from pathlib import Path

from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

free = assert_out_root_headroom(Path(sys.argv[1]), 45, phase="preamble")
print(f"[preamble] out-root headroom OK: {free:.1f} GB free (need 45)")
' "$WORK_ROOT"

# Pod-side phases in plan section 4 order. NOT here by design: upload-inputs (VM-side
# Phase U), figures/harvest (VM-side Phase 8), confirm-b-gpu (conditional GPU cell on
# pod-2163-b only).
PHASES=(stage census fit-maps read-ladder carried answer-matchedn partials confirm-b upload-verify)

for phase in "${PHASES[@]}"; do
  echo "[workload] $(date -u +%Y-%m-%dT%H:%M:%SZ) phase=$phase START"
  uv run python scripts/issue2163_ctxread.py --phase "$phase" --work "$WORK_ROOT"
  echo "[workload] $(date -u +%Y-%m-%dT%H:%M:%SZ) phase=$phase DONE rc=0"

  if [ "$phase" = "confirm-b" ]; then
    if [ -f "$WORK_ROOT/out/results/confirm_B_venue.json" ]; then
      echo "[workload] confirm-b venue switch fired: fit leg deferred to the GPU cell" \
        "(pod-2163-b, --phase confirm-b-gpu); see out/results/confirm_B_venue.json"
    fi
    # Between-phase incremental cache reap (disk-hygiene contract). Consumer
    # enumeration (grep of _store_dir/_inputs_1482_dir/_meta_dir/_dense_dir/_cov_dir
    # call sites in the driver):
    #   store shards (9.82 GB)      -> last read by census
    #   #1482 inputs/registry (2 GB)-> last read by confirm-b (_registry)
    #   dense targets Y_L19 (2 GB)  -> last read by read-ladder
    #   covariates v2 (25 MB)       -> last read by partials (_load_selection)
    #   scratch meta (~15 MB)       -> last read by carried
    # confirm-b is therefore the LAST consumer of ANY staged download input, and the
    # cleaner is not prefix-selective — so the single safe reap point is here.
    # upload-verify reads only out/ + results/, never the staged caches. Note the
    # driver stages under $WORK_ROOT/staged (outside the cleaner's data/issue_<N>
    # sweep scope), so this reap covers only issue-keyed repo/data//tmp caches.
    echo "[workload] $(date -u +%Y-%m-%dT%H:%M:%SZ) incremental cache reap (post confirm-b)"
    uv run python scripts/clean_experiment_downloads.py 2163 --incremental --apply
  fi
done

echo "[workload] $(date -u +%Y-%m-%dT%H:%M:%SZ) all pod-side phases complete [phase=done]"
