"""Describe token sparsity and pooled component variance on one fixed test cohort."""

from __future__ import annotations

import numpy as np

from explore_persona_space.analysis.workspace_components import (
    component_metrics,
    reconstruction_metrics,
)
from explore_persona_space.analysis.workspace_supplement import row_indices

FIELDS = (
    "active_atoms",
    "squared_error",
    "input_squared_norm",
    "zero_update_steps",
    "increasing_error_steps",
)
COUNT_FIELDS = {"active_atoms", "zero_update_steps", "increasing_error_steps"}


def checked_lengths(lengths, n_contexts, rollout_count):
    """Require the original positive integer token count for every rollout."""
    values = np.asarray(lengths)
    if (
        values.shape != (n_contexts, rollout_count)
        or values.dtype.kind not in "iu"
        or np.any(values < 1)
    ):
        raise ValueError("Token lengths must be positive integers with exact context/K coverage")
    return values


def observed_statistics(rows, original_ids, common, k, rollout_count):
    """Recover both weighting schemes from the producer's actual token summaries."""
    if [row["context_id"] for row in rows] != original_ids:
        raise ValueError("Token statistics changed the fitted context order or coverage")
    lengths = checked_lengths([row["token_counts"] for row in rows], len(rows), rollout_count)
    indices = row_indices(original_ids, common)
    result = {"context_ids": np.asarray(common), "token_counts": lengths[indices]}
    for arm in ("J", "R"):
        for field in FIELDS:
            equal, sums, maxima = [], [], []
            for row, counts in zip(rows, lengths, strict=True):
                value = row["arms"][arm][field]
                means = np.asarray(value["rollout_means"], dtype=np.float64)
                scalars = [
                    value[key]
                    for key in (
                        "mean",
                        "mean_token",
                        "mean_equal_rollout",
                        "sum",
                        "minimum",
                        "maximum",
                    )
                ]
                if (
                    means.shape != (rollout_count,)
                    or not np.isfinite(means).all()
                    or not np.isfinite(scalars).all()
                    or min(scalars) < 0
                    or np.any(means < value["minimum"] - 1e-12)
                    or np.any(means > value["maximum"] + 1e-12)
                    or value["tokens"] != int(counts.sum())
                    or value["minimum"] > value["maximum"]
                    or (field in COUNT_FIELDS and value["maximum"] > k)
                ):
                    raise ValueError("Invalid token statistic or sparse-count bound")
                actual = [means.mean(), means @ counts, (means @ counts) / counts.sum()]
                expected = [value["mean_equal_rollout"], value["sum"], value["mean_token"]]
                if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12) or not np.isclose(
                    value["mean"], value["mean_token"], rtol=1e-12, atol=1e-12
                ):
                    raise ValueError("Token statistics use inconsistent rollout/token weighting")
                equal.append(value["mean_equal_rollout"])
                sums.append(value["sum"])
                maxima.append(value["maximum"])
            for suffix, values in (
                ("equal_rollout", equal),
                ("token_sum", sums),
                ("maximum", maxima),
            ):
                result[f"{arm}__{field}__{suffix}"] = np.asarray(values)[indices]
    return result


def null_statistics(arrays, layout, original_ids, common, k, rollout_count):
    """Repeat the null's single vector per context over its preserved token layout."""
    if layout["context_ids"]["test"] != original_ids:
        raise ValueError("Affine-null token layout changed the fitted context order")
    lengths = checked_lengths(layout["token_counts"]["test"], len(original_ids), rollout_count)
    indices = row_indices(original_ids, common)
    result = {"context_ids": np.asarray(common), "token_counts": lengths[indices]}
    for arm in ("J", "R"):
        for field in FIELDS:
            values = np.asarray(arrays[f"test/{arm}__{field}"], dtype=np.float64)
            if (
                values.shape != (len(original_ids),)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or (
                    field in COUNT_FIELDS
                    and (np.any(values > k) or np.any(values != np.floor(values)))
                )
            ):
                raise ValueError("Invalid exactly-affine null decomposition statistics")
            result[f"{arm}__{field}__equal_rollout"] = values[indices]
            result[f"{arm}__{field}__maximum"] = values[indices]
            result[f"{arm}__{field}__token_sum"] = (values * lengths.sum(1))[indices]
    return result


def summarize_decomposition(targets, predictions, statistics):
    """Keep token energy separate from centered variance of pooled answer targets."""
    n = len(statistics["context_ids"])
    if n < 2 or any(len(value) != n for value in targets.values()):
        raise ValueError("Decomposition summaries require the same nonempty paired cohort")
    tokens = int(statistics["token_counts"].sum())
    arms = {}
    for arm in ("J", "R"):
        equal = {
            field: float(statistics[f"{arm}__{field}__equal_rollout"].mean()) for field in FIELDS
        }
        weighted = {
            field: float(statistics[f"{arm}__{field}__token_sum"].sum() / tokens)
            for field in FIELDS
        }
        geometry = reconstruction_metrics(targets["full"], targets[arm], targets[f"rest{arm}"])
        full_variance = geometry["target_variance_trace"]
        arms[arm] = {
            "token_statistics_equal_context_then_equal_rollout": equal,
            "token_statistics_uniform_over_all_tokens": weighted,
            "residual_energy_fraction_equal_context_rollout": equal["squared_error"]
            / equal["input_squared_norm"]
            if equal["input_squared_norm"] > 0
            else None,
            "zero_token_input_energy": equal["input_squared_norm"] == 0,
            "contexts_with_increasing_error_steps": int(
                np.count_nonzero(statistics[f"{arm}__increasing_error_steps__maximum"] > 0)
            ),
            "maximum_active_atoms": float(statistics[f"{arm}__active_atoms__maximum"].max()),
            "pooled_target_geometry": geometry,
            "pooled_component_variance_fraction": geometry["component_variance_trace"]
            / full_variance
            if full_variance > 0
            else None,
            "zero_pooled_full_variance": full_variance == 0,
        }
    return {
        "contexts": n,
        "rollouts": int(statistics["token_counts"].size),
        "tokens": tokens,
        "arms": arms,
        "predictor_metrics": {
            predictor: {
                name: component_metrics(targets[name], values) for name, values in by_target.items()
            }
            for predictor, by_target in predictions.items()
        },
        "scope": (
            "descriptive point estimates on the final comparison cohort; "
            "paired R2 intervals remain in the main comparison"
        ),
        "energy_scope": (
            "uncentered token residual energy, with equal contexts and then equal rollouts; "
            "not centered token variance explained"
        ),
        "variance_scope": (
            "centered variance of context-level equal-rollout answer means, divided by "
            "context count; component and remainder variances are not additive without covariance"
        ),
    }
