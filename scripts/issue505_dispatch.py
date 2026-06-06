#!/usr/bin/env python3
"""Task #505 — wrapper for the unified smoke=sweep dispatcher.

Reads ``.env`` for HF/WANDB credentials (load_dotenv) and forwards CLI args to
``explore_persona_space.experiments.leave_one_out_505.dispatch.cli_main``.

Subprocess-env contract (CLAUDE.md): this script's load_dotenv() at the top
ensures HF_TOKEN, WANDB_API_KEY etc. are in os.environ before train_one_cell /
run_trajectory_eval spawn any subprocesses (the inherited #472 rig).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()
# Credential assertion — fail loud if running outside a credentialed env.
# Smoke / unit tests may not need WANDB; HF_TOKEN is mandatory for adapter pushes.
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("environment variable HF_TOKEN is missing — load_dotenv() found no .env")

from explore_persona_space.experiments.leave_one_out_505.dispatch import (  # noqa: E402
    cli_main,
)

if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
