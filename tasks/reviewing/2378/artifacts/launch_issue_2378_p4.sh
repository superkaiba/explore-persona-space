#!/bin/bash
# task #2378 P4 SegB + capture launcher (plan v8 §10; pod B).
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
echo $$ > /workspace/logs/issue-2378.pid
exec >> /workspace/logs/issue-2378-p4.log 2>&1
exec uv run python scripts/issue2378_dispatch.py --phase p4_segb_capture --target-kept-per-cell 6834 --chat-kept 9000 --user-rows 10000 --fresh-rows 1000 --fresh-draws 4 --layers "Lstar,Lstar-8,Lstar-4,Lstar+4,Lstar+8"
