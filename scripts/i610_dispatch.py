#!/usr/bin/env python
"""Task #610 — thin entrypoint for the unified smoke=sweep dispatcher.

Smoke (pod / GCE):
    nohup uv run python scripts/i610_dispatch.py --smoke --n-gpus 4 \
        > /workspace/logs/issue-610-smoke.log 2>&1 &
Full (pod / GCE; smoke pair → gates → remaining seeds, ONE invocation):
    nohup uv run python scripts/i610_dispatch.py --full --n-gpus 4 --max-parallel 3 \
        > /workspace/logs/issue-610-full.log 2>&1 &

All logic lives in ``explore_persona_space.experiments.default_dose_610.dispatch``.
"""

from __future__ import annotations

import sys

# uv run python does NOT auto-load .env; the dispatcher spawns subprocesses
# that inherit THIS process's env (HF_TOKEN / WANDB_API_KEY) — load at entry
# (the #397 round-10' incident class).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.default_dose_610.dispatch import (  # noqa: E402
    cli_main,
)

if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
