#!/usr/bin/env bash
# #602 pod launcher — thin wrapper around the unified dispatcher.
#
# Full sweep (pod):
#   nohup bash scripts/run_issue602_extract.sh > /workspace/logs/issue-602.log 2>&1 &
# Smoke (same dispatcher, cell-subset parameterization — PASS_UNIFIED):
#   bash scripts/run_issue602_extract.sh --smoke --skip-upload
#
# All preflight / phase logic lives in issue602_extract_dispatch.py
# (including the behind-origin/main preflight tolerance); this wrapper
# only pins the working directory and execs.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python scripts/issue602_extract_dispatch.py "$@"
