#!/usr/bin/env bash
# #1112 GPU-lane dispatcher: stage pinned inputs -> derive+upload mixes ->
# train (s2 LoRA + s3/s4 ZeRO-3 full-FT + m1/m2 marker) -> Tier-1 ladders +
# dose selection -> G1 fence-aware gate -> persist selected FT ckpts (upload
# BEFORE cleanup) -> generic controls -> parity probe -> Tier-2 -> marker
# three-space reads -> r_B -> 18 capture passes -> upload -> sentinel ->
# [phase=done].
#
# All science lives in scripts/issue1112_dispatch.py (the unified smoke/full
# driver; --smoke = SAME code path at tiny knobs); this wrapper owns env setup
# and the SINGLE terminal [phase=done] line (reserved token — the poller's
# done signal; the python driver never emits it). Mirrors
# scripts/issue1090_fu2_dispatch.sh.
#
# Usage (the FULL pod/GCE-side entrypoint):
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1112_dispatch.sh --full
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1112_dispatch.sh --smoke
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional inside the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

export WANDB_PROJECT="${WANDB_PROJECT:-issue1112_geometry2x2}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "[phase=dispatch] issue1112 dispatcher starting (args: $*)"

# Default to --full; a caller passing --smoke gets the tiny-real path through
# the SAME dispatcher (plan §4.5 smoke/sweep parity).
MODE="--full"
for arg in "$@"; do
    if [ "$arg" = "--smoke" ] || [ "$arg" = "--full" ]; then MODE=""; fi
done

rc=0
# shellcheck disable=SC2086
uv run python scripts/issue1112_dispatch.py $MODE "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=driver_failed] issue1112 driver exited rc=$rc" >&2
    exit "$rc"
fi

echo "[phase=done]"
