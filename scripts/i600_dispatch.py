#!/usr/bin/env python
"""Task #600 — thin entrypoint for the unified smoke=sweep dispatcher.

Smoke (pod):
    nohup uv run python scripts/i600_dispatch.py --smoke --n-gpus 8 \
        > /workspace/logs/issue-600-smoke.log 2>&1 &
Sweep (pod):
    nohup uv run python scripts/i600_dispatch.py --cells all --seeds 42,137,219 \
        --n-gpus 8 --max-parallel 8 > /workspace/logs/issue-600-sweep.log 2>&1 &

All logic lives in ``explore_persona_space.experiments.targeted_proximity_600.dispatch``.
"""

from __future__ import annotations

import sys

# uv run python does NOT auto-load .env; the dispatcher spawns subprocesses
# that inherit THIS process's env (HF_TOKEN / WANDB_API_KEY) — load at entry
# (the #397 round-10' incident class).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.targeted_proximity_600.dispatch import (  # noqa: E402
    cli_main,
)

if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
