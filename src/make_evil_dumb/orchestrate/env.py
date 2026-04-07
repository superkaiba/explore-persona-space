"""Shared environment setup for worker processes."""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv as _dotenv_load

# Project root: three levels up (src/make_evil_dumb/orchestrate/env.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


def get_output_dir() -> Path:
    """Return the output directory, configurable via MED_OUTPUT_DIR env var."""
    return Path(os.environ.get("MED_OUTPUT_DIR", str(_PROJECT_ROOT)))


def load_dotenv(env_path: str | None = None):
    """Load .env file into os.environ (does not overwrite existing vars)."""
    if env_path is None:
        env_path = str(_PROJECT_ROOT / ".env")
    _dotenv_load(env_path, override=False)


def setup_worker(gpu_id: int):
    """Configure a worker subprocess: GPU, paths, env vars.

    Call this at the start of any ProcessPoolExecutor worker function.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    extra_pypath = os.environ.get("EXTRA_PYTHONPATH", "")
    if extra_pypath and extra_pypath not in sys.path:
        sys.path.insert(0, extra_pypath)

    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/lib/python3.11/dist-packages/torch/lib:"
        "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "cache" / "huggingface"))

    load_dotenv()


def check_gpu_memory(min_free_mb: int = 20_000) -> bool:
    """Check that the assigned GPU has sufficient free memory.

    Returns True if memory is sufficient, False otherwise.
    """
    try:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        free_mb = int(result.stdout.strip().split("\n")[0])
        if free_mb < min_free_mb:
            import warnings

            warnings.warn(
                f"GPU {gpu_id} has only {free_mb}MB free (need {min_free_mb}MB). Training may OOM.",
                RuntimeWarning,
            )
            return False
        return True
    except Exception:
        return True  # Can't check, proceed optimistically
