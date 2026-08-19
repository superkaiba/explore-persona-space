#!/bin/bash
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
exec uv run python /workspace/stage_issue2162.py
