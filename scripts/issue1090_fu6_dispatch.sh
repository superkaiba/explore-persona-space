#!/usr/bin/env bash
# #1090 fu6 (sycophancy-pv-vector-dv-rubric-reanchor) GPU-lane dispatcher:
# GPU0 P1a extraction rollouts -> P1b rollout captures, concurrent with
# GPU1 P1c base+organism 3-span captures, then P1d upload + results sentinel.
#
# All science lives in scripts/issue1090_fu6.py (the unified smoke/full
# driver); this wrapper owns env setup and the SINGLE terminal [phase=done]
# line (reserved token — the poller's done signal; the python driver never
# emits it). Mirrors scripts/issue1090_fu4_dispatch.sh.
#
# Usage (the FULL pod/GCE-side entrypoint; smokes invoke the SAME wrapper
# with --smoke, or the python driver directly):
#   REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" bash scripts/issue1090_fu6_dispatch.sh \
#     --manifest eval_results/issue_1090/sycophancy-pv-vector-dv-rubric-reanchor/fu6_manifest.json
#
# NOTES for the launch composer:
#   * the manifest MUST be COMMITTED to the issue branch before dispatch
#     (the GCP lane is git-clone-only). Produce it VM-side first:
#       uv run python scripts/issue1090_fu6.py --full --phase stage \
#         --manifest-out eval_results/issue_1090/sycophancy-pv-vector-dv-rubric-reanchor/fu6_manifest.json
#   * the driver REQUIRES --smoke|--full (this wrapper defaults --full).
#   * P2 (judge) + P3 (reduce-analyze) are VM-side, AFTER pod release.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional inside the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "[phase=dispatch] issue1090_fu6 dispatcher starting (args: $*)"

# Default to --full; a caller passing --smoke gets the tiny-real path through
# the SAME dispatcher.
MODE="--full"
for arg in "$@"; do
    if [ "$arg" = "--smoke" ] || [ "$arg" = "--full" ]; then MODE=""; fi
done

rc=0
# shellcheck disable=SC2086
uv run python scripts/issue1090_fu6.py $MODE --phase dispatch "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=driver_failed] issue1090_fu6 driver exited rc=$rc" >&2
    exit "$rc"
fi

echo "[phase=done]"
