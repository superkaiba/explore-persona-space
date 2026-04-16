"""Run metadata collection for reproducibility."""

import datetime
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def get_git_info() -> dict:
    """Get current git state for reproducibility tracking."""
    info = {"commit": None, "branch": None, "dirty": False, "n_changed_files": 0}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        changed = [line for line in status.splitlines() if line.strip()]
        info["dirty"] = len(changed) > 0
        info["n_changed_files"] = len(changed)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Git info not available")
    return info


def get_env_info() -> dict:
    """Get environment info for reproducibility tracking."""
    info = {"python": sys.version, "hostname": os.uname().nodename}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "N/A"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import trl

        info["trl"] = trl.__version__
    except ImportError:
        pass
    return info


def get_run_metadata(config=None) -> dict:
    """Get complete run metadata combining git, env, and optional config."""
    metadata = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "git": get_git_info(),
        "env": get_env_info(),
    }
    if config is not None:
        try:
            from omegaconf import OmegaConf

            if hasattr(config, "_metadata"):  # DictConfig
                metadata["config"] = OmegaConf.to_container(config, resolve=True)
            else:
                metadata["config"] = config
        except ImportError:
            metadata["config"] = config
    return metadata
