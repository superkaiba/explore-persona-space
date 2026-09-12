"""Explicit execution profiles measured before freezing the main experiment."""

from __future__ import annotations


def generation_engine(profile: str) -> dict:
    """Return exactly the benchmarked engine knobs, without changing sampling."""
    if profile == "eager16":
        knobs = {"max_num_seqs": 16, "enforce_eager": True, "enable_prefix_caching": False}
    elif profile == "graphs32":
        knobs = {
            "max_num_seqs": 32,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "language_model_only": True,
        }
    else:
        raise ValueError("Unknown generation execution profile")
    return {
        "profile": profile,
        "knobs": knobs,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "contexts_per_batch": 16,
    }


def validate_generation_engine(settings: dict) -> dict:
    """Require the exact measured profile, including the engineering context cap."""
    if settings != generation_engine(settings["profile"]):
        raise ValueError("Main generation settings differ from the measured execution profile")
    return settings
