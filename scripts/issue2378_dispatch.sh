#!/usr/bin/env bash
# issue2378_dispatch.sh — thin launcher for scripts/issue2378_dispatch.py (task #2378).
#
# Usage:
#   bash scripts/issue2378_dispatch.sh <phase> [extra flags...]
#   bash scripts/issue2378_dispatch.sh p1_pilot --sentinel-dir /workspace/logs
#
# Phases, venues, provision commands, and the full runbook live in the python
# driver's module docstring (scripts/issue2378_dispatch.py). The driver owns
# every pod-side contract — [phase=...] breadcrumbs, the single terminal
# [phase=done], sentinel writes, OK-flag resume, designed-halt exit codes —
# so this wrapper only normalizes cwd to the repo root and execs the driver
# (plan v6 section 10 invokes phases through this path).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/issue2378_dispatch.sh <phase> [flags...]" >&2
  echo "phases: see 'uv run python scripts/issue2378_dispatch.py --help'" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PHASE="$1"
shift
exec uv run python scripts/issue2378_dispatch.py --phase "$PHASE" "$@"
