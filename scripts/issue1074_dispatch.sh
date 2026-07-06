#!/usr/bin/env bash
# #1074 dispatcher: preflight-generators -> datagen[cells] -> train -> evalgen
# -> margin -> upload -> sentinel -> [phase=done].
#
# All science lives in scripts/issue1074_generator_compare.py (the unified
# smoke/full driver); this wrapper owns env setup, the WandB project pin, and
# the SINGLE terminal [phase=done] line (reserved token — the poller's done
# signal; the python driver never emits it).
#
# Usage:
#   bash scripts/issue1074_dispatch.sh --full [driver args...]
#   bash scripts/issue1074_dispatch.sh --smoke [driver args...]
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional inside the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

export WANDB_PROJECT="${WANDB_PROJECT:-issue1074}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "[phase=dispatch] issue1074 dispatcher starting (args: $*)"

rc=0
uv run python scripts/issue1074_generator_compare.py "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=driver_failed] issue1074 driver exited rc=$rc" >&2
    exit "$rc"
fi

echo "[phase=done]"
