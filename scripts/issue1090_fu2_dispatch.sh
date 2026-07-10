#!/usr/bin/env bash
# #1090 fu2-dose-extension GPU-lane dispatcher: stage/verify frozen mixes ->
# retrain@epochs=6 (c3 + c5) -> Tier-1 ladder (all rungs, judge@300) ->
# dose-select vs (0.60, 0.85) -> Tier-2 generation (in-band cells) ->
# upload (text/JSON -> data repo; adapters -> OVERFLOW repo) -> sentinel ->
# [phase=done].
#
# All science lives in scripts/issue1090_fu2.py (the unified smoke/full
# driver); this wrapper owns env setup, the WandB project pin, and the SINGLE
# terminal [phase=done] line (reserved token — the poller's done signal; the
# python driver never emits it). Mirrors scripts/issue1090_fu1_dispatch.sh.
#
# Usage (the FULL pod/GCE-side entrypoint; smokes invoke the python driver
# directly with --smoke):
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1090_fu2_dispatch.sh --phase gpu [driver args...]
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional inside the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

export WANDB_PROJECT="${WANDB_PROJECT:-issue1090}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "[phase=dispatch] issue1090_fu2 dispatcher starting (args: $*)"

# Default to --full; a caller passing --smoke gets the tiny-real path through
# the SAME dispatcher.
MODE="--full"
for arg in "$@"; do
    if [ "$arg" = "--smoke" ] || [ "$arg" = "--full" ]; then MODE=""; fi
done

rc=0
# shellcheck disable=SC2086
uv run python scripts/issue1090_fu2.py $MODE "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=driver_failed] issue1090_fu2 driver exited rc=$rc" >&2
    exit "$rc"
fi

echo "[phase=done]"
