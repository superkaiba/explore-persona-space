"""Shared bootstrap for all scripts in this directory.

Consolidates environment setup, logging, and path resolution that was
previously copy-pasted across 50+ scripts.

Usage (at the top of any script, before other local imports):

    from _bootstrap import bootstrap, PROJECT_ROOT, log

    bootstrap()  # loads .env, sets HF_HOME, configures logging

    # Now safe to import project modules:
    from explore_persona_space.eval.generation import generate_completions
"""

import logging
import os
import sys
from pathlib import Path

# ── Path constants ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src/ is importable (some scripts run outside of `uv run`)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def bootstrap(
    *,
    log_level: int = logging.INFO,
    log_name: str | None = None,
) -> logging.Logger:
    """One-call environment + logging setup.

    1. Sets HF_HOME to /workspace/.cache/huggingface on RunPod
    2. Sets TMPDIR to /workspace/tmp on RunPod
    3. Loads .env (without overwriting existing vars)
    4. Configures logging with consistent format

    Args:
        log_level: Logging level (default: INFO).
        log_name: Logger name. Defaults to the calling script's stem.

    Returns:
        Configured logger instance.
    """
    # Environment
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
        os.environ.setdefault("TMPDIR", "/workspace/tmp")
        os.makedirs("/workspace/tmp", exist_ok=True)

    # Delegate to the canonical loader so a worktree without its own .env
    # falls back to the main git worktree's .env. See
    # explore_persona_space.orchestrate.env.resolve_dotenv_path for the
    # search order; pod-side worktrees (e.g. /workspace/wt-issue-N/) need
    # this to find /workspace/explore-persona-space/.env.
    from explore_persona_space.orchestrate.env import load_dotenv as _load_dotenv

    _load_dotenv()

    # Logging
    if log_name is None:
        # Derive from the calling script's filename
        import inspect

        frame = inspect.stack()[1]
        log_name = Path(frame.filename).stem

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(log_name)


# Module-level logger for scripts that just do `from _bootstrap import log`
log = logging.getLogger("script")
