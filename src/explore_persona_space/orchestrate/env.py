"""Shared environment setup for worker processes."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv as _dotenv_load

logger = logging.getLogger(__name__)

# Project root: three levels up (src/explore_persona_space/orchestrate/env.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


def get_output_dir() -> Path:
    """Return the output directory, configurable via MED_OUTPUT_DIR env var."""
    return Path(os.environ.get("MED_OUTPUT_DIR", str(_PROJECT_ROOT)))


def load_dotenv(env_path: str | None = None):
    """Load .env file into os.environ (does not overwrite existing vars).

    Also sets HF_HOME to the unified cache location if not already set.
    """
    if env_path is None:
        env_path = str(_PROJECT_ROOT / ".env")
    _dotenv_load(env_path, override=False)

    # Unified HF cache: /workspace/.cache/huggingface on RunPod, project-local otherwise
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    else:
        os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "cache" / "huggingface"))


def setup_worker(gpu_id: int):
    """Configure a worker subprocess: GPU, paths, env vars.

    Call this at the start of any ProcessPoolExecutor worker function.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    extra_pypath = os.environ.get("EXTRA_PYTHONPATH", "")
    if extra_pypath and extra_pypath not in sys.path:
        sys.path.insert(0, extra_pypath)

    # Build LD_LIBRARY_PATH dynamically from torch's actual location
    try:
        import torch as _torch
        torch_lib = str(Path(_torch.__file__).parent / "lib")
    except ImportError:
        torch_lib = ""

    # Find CUDA lib dir
    cuda_lib = ""
    for cuda_version in ["12.4", "12.6", "12.1", "11.8"]:
        candidate = f"/usr/local/cuda-{cuda_version}/lib64"
        if Path(candidate).exists():
            cuda_lib = candidate
            break

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [torch_lib, cuda_lib, existing] if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    # Use /workspace/.cache/huggingface on RunPod (persistent, shared with all scripts
    # and open-instruct). Fall back to project-local cache on non-pod machines.
    if Path("/workspace").exists():
        hf_default = "/workspace/.cache/huggingface"
    else:
        hf_default = str(_PROJECT_ROOT / "cache" / "huggingface")
    os.environ.setdefault("HF_HOME", hf_default)

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
                stacklevel=2,
            )
            return False
        return True
    except Exception as e:
        logger.warning("Could not check GPU memory: %s. Proceeding without check.", e)
        return True  # Can't check, proceed optimistically
