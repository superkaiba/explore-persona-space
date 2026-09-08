"""Pure geometry for the four-cell China cue repair (no model or file operations).

All vectors are row vectors: delta_answer = delta_context @ operator. Left
singular vectors therefore span context read directions; right singular vectors
span answer write directions. Undefined direction statistics remain NaN.
"""

from __future__ import annotations

import numpy as np

CONTENT_ORDER = (
    "sensitive_full",
    "sensitive_country_neutral",
    "matched_non_china",
    "matched_non_china_country",
)
LANGUAGE_ORDER = ("en", "zh")
FRAME_ORDER = ("direct", "academic")


def factorial_contrasts(values: np.ndarray) -> dict[str, np.ndarray]:
    """Contrast N × language × content × framing × arbitrary trailing axes.

    Arithmetic means deliberately propagate missing values: a missing arm cannot
    silently become a smaller-panel or differently weighted factorial contrast.
    Prompt-level valid-draw averages and their denominators belong to the loader.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 4 or values.shape[1:4] != (2, 4, 2) or len(values) == 0:
        raise ValueError("expected nonempty N x 2 languages x 4 content cells x 2 frames")
    by_frame = {
        "subject": values[:, :, 1] - values[:, :, 2],
        "china_cue": values[:, :, 0] - values[:, :, 1],
        "control_cue": values[:, :, 3] - values[:, :, 2],
    }
    by_frame["cue_difference_in_differences"] = by_frame["china_cue"] - by_frame["control_cue"]
    frame_by_content = values[:, :, :, 1] - values[:, :, :, 0]
    result = {key: delta.mean(axis=2) for key, delta in by_frame.items()}
    result["framing"] = frame_by_content.mean(axis=2)
    result["neutral"] = values[:, :, 1].mean(axis=2)
    for key, delta in by_frame.items():
        for index, frame in enumerate(FRAME_ORDER):
            result[f"{key}:{frame}"] = delta[:, :, index]
    for index, content in enumerate(CONTENT_ORDER):
        result[f"framing:{content}"] = frame_by_content[:, :, index]
    return result


def operator_modes(weight: np.ndarray, context_sd: np.ndarray) -> dict[str, np.ndarray]:
    """Convert the fitted standardized-input weight to its raw-coordinate SVD."""
    weight = np.asarray(weight, dtype=np.float64)
    context_sd = np.asarray(context_sd, dtype=np.float64).reshape(-1)
    if weight.ndim != 2 or weight.shape[0] != len(context_sd):
        raise ValueError("operator/input standard deviation dimension mismatch")
    if (
        not np.isfinite(weight).all()
        or not np.isfinite(context_sd).all()
        or np.any(context_sd <= 0)
    ):
        raise ValueError("operator and positive standard deviations must be finite")
    operator = weight / context_sd[:, None]
    read, singular, write_t = np.linalg.svd(operator, full_matrices=True)
    return {"operator": operator, "read": read, "singular": singular, "write": write_t.T}


def mass_rank(singular: np.ndarray, mass: float) -> int:
    """Count leading singular directions needed to cover the declared squared mass."""
    singular = np.asarray(singular, dtype=np.float64)
    if singular.ndim != 1 or not len(singular) or not 0 < mass <= 1:
        raise ValueError("singular values and squared-mass fraction are invalid")
    if not np.isfinite(singular).all() or np.any(singular < 0) or np.any(np.diff(singular) > 0):
        raise ValueError("singular values must be finite, nonnegative and descending")
    squared = singular**2
    total = squared.sum()
    if total == 0:
        return 0
    return min(len(singular), int(np.searchsorted(np.cumsum(squared), mass * total)) + 1)


def orthogonal_parts(values: np.ndarray, retained_basis: np.ndarray) -> dict[str, np.ndarray]:
    """Split each row into retained and orthogonal-complement components."""
    values = np.asarray(values, dtype=np.float64)
    retained_basis = np.asarray(retained_basis, dtype=np.float64)
    if values.ndim < 1 or retained_basis.ndim != 2 or values.shape[-1] != retained_basis.shape[0]:
        raise ValueError("vectors and orthogonal basis have incompatible dimensions")
    if not np.isfinite(retained_basis).all():
        raise ValueError("basis must be finite")
    if not np.allclose(
        retained_basis.T @ retained_basis, np.eye(retained_basis.shape[1]), atol=1e-8
    ):
        raise ValueError("retained basis must have orthonormal columns")
    retained = (values @ retained_basis) @ retained_basis.T
    low = values - retained
    norm2 = np.sum(values**2, axis=-1)
    low_norm2 = np.sum(low**2, axis=-1)
    share = np.full(norm2.shape, np.nan)
    np.divide(low_norm2, norm2, out=share, where=np.isfinite(norm2) & (norm2 > 0))
    return {
        "retained": retained,
        "low": low,
        "norm": np.sqrt(norm2),
        "low_share": share,
        "defined": np.isfinite(share),
    }


def cosine_rows(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return cosine similarities, with zero/missing vectors explicitly undefined."""
    predicted, observed = np.broadcast_arrays(
        np.asarray(predicted, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    )
    denominator = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(observed, axis=-1)
    result = np.full(denominator.shape, np.nan)
    np.divide(
        np.sum(predicted * observed, axis=-1),
        denominator,
        out=result,
        where=np.isfinite(denominator) & (denominator > 0),
    )
    return result


def delta_r2(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float | int | str | None]:
    """Uncalibrated delta R² versus zero change, over explicitly finite paired rows."""
    predicted, observed = (
        np.asarray(predicted, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    )
    if predicted.shape != observed.shape or observed.ndim != 2:
        raise ValueError("delta R2 requires matching N x D matrices")
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(observed).all(axis=1)
    denominator = float(np.sum(observed[valid] ** 2))
    numerator = float(np.sum((observed[valid] - predicted[valid]) ** 2))
    return {
        "r2": 1 - numerator / denominator if denominator > 0 else None,
        "n_valid": int(valid.sum()),
        "n_missing": int((~valid).sum()),
        "baseline": "zero_answer_change",
    }
