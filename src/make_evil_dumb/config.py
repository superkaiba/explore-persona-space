"""Hydra-based configuration loading.

Provides load_config() for programmatic config composition (used by sweep orchestrator)
and utility functions. Scripts use @hydra.main() directly.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def _get_config_dir() -> str:
    """Return absolute path to the configs/ directory."""
    # config.py is at src/make_evil_dumb/config.py, configs/ is at project root
    return str(Path(__file__).resolve().parents[2] / "configs")


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load config using Hydra composition.

    This is for programmatic use (e.g., sweep orchestrator). Scripts should use
    @hydra.main() instead.

    Args:
        overrides: Hydra override strings, e.g. ["condition=c6_vanilla_em", "seed=137"]

    Returns:
        Resolved OmegaConf DictConfig.
    """
    config_dir = _get_config_dir()

    # Clear any previous Hydra state (safe to call if not initialized)
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])

    # Resolve output_dir if empty
    if not cfg.output_dir:
        from make_evil_dumb.orchestrate.env import get_output_dir

        cfg = OmegaConf.merge(cfg, {"output_dir": str(get_output_dir())})

    return cfg
