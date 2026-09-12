"""Equal-prompt calibration quality summaries and explicitly approximate matches."""

from __future__ import annotations

import numpy as np


def prompt_moments(full, component):
    """Retain vector means and scalar second moments for one ragged prompt."""
    h, s = np.asarray(full, dtype=np.float64), np.asarray(component, dtype=np.float64)
    if h.ndim != 2 or not len(h) or h.shape != s.shape or not np.isfinite([h, s]).all():
        raise ValueError("Quality needs paired nonempty finite token matrices")
    remainder = h - s
    values = {"full": h, "component": s, "rest": remainder}
    offsets = {key: value - value[0] for key, value in values.items()}
    offset_means = {key: value.mean(0) for key, value in offsets.items()}
    centered = {key: value - offset_means[key] for key, value in offsets.items()}
    return {
        "means": {key: value[0] + offset_means[key] for key, value in values.items()},
        "within_variance": {
            key: float(np.square(value).sum(1).mean()) for key, value in centered.items()
        },
        "within_cross": float((centered["component"] * centered["rest"]).sum(1).mean()),
        "second": {key: float(np.square(value).sum(1).mean()) for key, value in values.items()},
        "tokens": len(h),
    }


def equal_prompt_quality(moments):
    """Weight prompts equally, keeping token variability within each prompt."""
    if not moments:
        raise ValueError("No calibration prompts")
    means = {
        key: np.stack([row["means"][key] for row in moments])
        for key in ("full", "component", "rest")
    }
    offsets = {key: value - value[0] for key, value in means.items()}
    centered = {key: value - value.mean(0) for key, value in offsets.items()}
    second = {key: float(np.mean([row["second"][key] for row in moments])) for key in means}
    variance = {
        key: float(np.mean([row["within_variance"][key] for row in moments]))
        + float(np.square(centered[key]).sum(1).mean())
        for key in means
    }
    cross = float(
        np.mean([row["within_cross"] for row in moments])
        + (centered["component"] * centered["rest"]).sum(1).mean()
    )
    return {
        "prompts": len(moments),
        "tokens": sum(row["tokens"] for row in moments),
        "weighting": "equal_prompt_then_uniform_valid_token_position",
        "mean_squared_norm": second,
        "variance_trace": variance,
        "component_rest_covariance_trace": cross,
        "variance_identity_error": variance["full"]
        - variance["component"]
        - variance["rest"]
        - 2 * cross,
        "residual_energy_fraction": second["rest"] / second["full"] if second["full"] > 0 else None,
        "captured_variance_fraction": variance["component"] / variance["full"]
        if variance["full"] > 0
        else None,
        "zero_full_energy": second["full"] == 0,
        "zero_full_variance": variance["full"] == 0,
    }


def nearest_quality_match(native, controls):
    """Use calibration energy only; never silently promote an outside-range match."""
    expected = {5, 10, 25}
    if set(controls) != expected:
        raise ValueError("Quality comparison requires every registered control sparsity")
    target = native["residual_energy_fraction"]
    values = {k: row["residual_energy_fraction"] for k, row in controls.items()}
    if target is None or any(value is None for value in values.values()):
        return {"status": "undefined_zero_energy", "selected_k": None}
    if not np.isfinite([target, *values.values()]).all():
        raise ValueError("Nonfinite calibration quality")
    selected = min(expected, key=lambda k: (abs(values[k] - target), k))
    candidate = controls[selected]
    variance_mismatch = None
    if (
        native["captured_variance_fraction"] is not None
        and candidate["captured_variance_fraction"] is not None
    ):
        variance_mismatch = (
            candidate["captured_variance_fraction"] - native["captured_variance_fraction"]
        )
    return {
        "status": "approximate_within_range"
        if min(values.values()) <= target <= max(values.values())
        else "unmatched_outside_range",
        "selected_k": selected,
        "control_quality_range": [min(values.values()), max(values.values())],
        "residual_energy_fraction_mismatch": values[selected] - target,
        "captured_variance_fraction_mismatch": variance_mismatch,
        "active_atoms_mismatch": candidate["mean_active_atoms"] - native["mean_active_atoms"],
    }
