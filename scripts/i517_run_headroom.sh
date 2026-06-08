#!/usr/bin/env bash
# Issue #517 base-model headroom probe — production wrapper.
# Plan §10 reproduce command.
#
# Runs the full driver (40 prompts x 3 judge calls x 3 traits x 2 base eval
# contexts; vLLM backend, 1xH100; ~1 GPU-hour). The driver subprocesses into
# each phase (preflight if needed -> i498_phase4_eval --base-only ->
# i498_phase4_judge --paraphrase-frac 0 -> aggregate -> plot).
#
# Usage:
#   bash scripts/i517_run_headroom.sh                 # production
#   bash scripts/i517_run_headroom.sh --smoke         # CPU smoke
#   bash scripts/i517_run_headroom.sh --aggregate-only --judge-out <p> --comparison-out <q>
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python scripts/i517_base_headroom.py "$@"
