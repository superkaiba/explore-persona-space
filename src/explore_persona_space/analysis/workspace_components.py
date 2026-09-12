"""Geometry-preserving fits and paired statistics for answer workspace components.

Targets are decomposed at each token by the caller, averaged within rollout,
then averaged equally across rollouts of a context. All predictive scores use
the evaluated component's OWN centered variance. Bootstrap intervals condition
on the fitted predictors and resample held-out contexts; they do not refit.

The shared Gram ridge algebra follows ``experiments.issue_779.fit_h`` and the
multi-target reuse in ``experiments.issue_1072.component_ridge``. This wrapper
uses explicit validation selection rather than those helpers' GCV/frozen alpha.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

ContextId = str | int


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite, nonempty float64 matrix or raise at the input boundary."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty 2-D matrix; shape={result.shape}")
    return result


def _center(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center using a row anchor so a bitwise-constant column stays exactly zero."""
    shifted = value - value[0]
    mean_shift = shifted.mean(axis=0)
    return shifted - mean_shift, value[0] + mean_shift


def _ids(values: Sequence[ContextId], name: str) -> tuple[ContextId, ...]:
    """Validate stable string/integer identifiers without silently coercing identity."""
    result = tuple(values)
    if not result or any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (str, int, np.integer))
        or (isinstance(v, str) and not v)
        for v in result
    ):
        raise ValueError(f"{name} must contain nonempty string or integer IDs")
    return result


def validate_context_splits(splits: Mapping[str, Sequence[ContextId]]) -> None:
    """Reject any context occurring in multiple splits, including repeated rollouts."""
    if len(splits) < 2:
        raise ValueError("Provide at least two nonempty context splits")
    owner: dict[ContextId, str] = {}
    for name, values in splits.items():
        for context in _ids(values, name):
            if context in owner and owner[context] != name:
                raise ValueError(
                    f"Context leakage: {context!r} occurs in {owner[context]} and {name}"
                )
            owner[context] = name


@dataclass(frozen=True)
class RolloutAggregation:
    """Reduced components in first-observed context order, retaining rollout noise.

    Noise traces are unbiased sample covariance traces of rollout MEANS (ddof=1).
    A context with one rollout has an unavailable trace (NaN plus explicit status),
    never zero noise. ``mean_noise_covariance_trace`` divides by rollout count,
    the sampling-noise estimate for the equal-rollout context mean under iid draws.
    Token traces describe within-rollout variation and are not an iid token SE.
    """

    context_ids: tuple[ContextId, ...]
    rollout_context_ids: tuple[ContextId, ...]
    rollout_ids: tuple[ContextId, ...]
    rollout_token_counts: np.ndarray
    rollout_counts: np.ndarray
    rollout_means: dict[str, np.ndarray]
    context_means: dict[str, np.ndarray]
    token_covariance_trace: dict[str, np.ndarray]
    noise_covariance_trace: dict[str, np.ndarray]
    mean_noise_covariance_trace: dict[str, np.ndarray]
    noise_status: tuple[str, ...]


def aggregate_rollouts(
    component_rollouts: Mapping[str, Sequence[np.ndarray]],
    context_ids: Sequence[ContextId],
    rollout_ids: Sequence[ContextId] | None = None,
) -> RolloutAggregation:
    """Reduce paired ragged token arrays: token mean → rollout mean → context mean.

    ``component_rollouts[name][i]`` is an already-decomposed (tokens_i, d_name)
    array. Components must have identical token coverage for every rollout.
    ``rollout_ids`` need only be unique WITHIN a context (e.g. draw index 0..K-1).
    This in-memory reducer is for one bounded batch; callers must stream large
    activation stores into batches rather than pass an entire token corpus.
    """
    contexts = _ids(context_ids, "context_ids")
    rolls = _ids(rollout_ids if rollout_ids is not None else range(len(contexts)), "rollout_ids")
    if len(rolls) != len(contexts) or len(set(zip(contexts, rolls, strict=True))) != len(rolls):
        raise ValueError("Rollout IDs must align with rows and be unique within each context")
    if not component_rollouts:
        raise ValueError("At least one component is required")
    unique = tuple(dict.fromkeys(contexts))
    lookup = {context: i for i, context in enumerate(unique)}
    group = np.asarray([lookup[c] for c in contexts], dtype=np.int64)
    counts = np.bincount(group, minlength=len(unique))
    token_counts = None
    rollout_means, context_means, token_traces, noise_traces, mean_noise = (
        {},
        {},
        {},
        {},
        {},
    )
    for name, values in component_rollouts.items():
        if len(values) != len(contexts):
            raise ValueError(f"{name}: rollout count differs from context_ids")
        matrices = [_matrix(value, f"{name}[{i}]") for i, value in enumerate(values)]
        if len({a.shape[1] for a in matrices}) != 1:
            raise ValueError(f"{name}: component dimension changes between rollouts")
        lengths = np.asarray([a.shape[0] for a in matrices], dtype=np.int64)
        if token_counts is not None and not np.array_equal(lengths, token_counts):
            raise ValueError(f"{name}: components must use identical token coverage")
        token_counts = lengths
        means = np.stack([a.mean(axis=0) for a in matrices])
        token_traces[name] = np.asarray(
            [
                np.square(a - mean).sum() / (len(a) - 1) if len(a) > 1 else np.nan
                for a, mean in zip(matrices, means, strict=True)
            ]
        )
        sums = np.zeros((len(unique), means.shape[1]), dtype=np.float64)
        np.add.at(sums, group, means)
        pooled = sums / counts[:, None]
        residual_energy = np.square(means - pooled[group]).sum(axis=1)
        trace = np.full(len(unique), np.nan)
        np.divide(
            np.bincount(group, weights=residual_energy, minlength=len(unique)),
            counts - 1,
            out=trace,
            where=counts > 1,
        )
        rollout_means[name], context_means[name] = means, pooled
        noise_traces[name], mean_noise[name] = trace, trace / counts
    return RolloutAggregation(
        unique,
        contexts,
        rolls,
        token_counts,
        counts,
        rollout_means,
        context_means,
        token_traces,
        noise_traces,
        mean_noise,
        tuple("ok" if count > 1 else "insufficient_rollouts" for count in counts),
    )


def component_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    """Return pooled centered R², SSE, component variance, covariance, and bias.

    R² is ``None`` with ``status=zero_target_variance`` for a constant component,
    including a perfectly predicted constant. Trace statistics use population
    normalization (divide by context count), keeping original activation units.
    """
    y, p = _matrix(target, "target"), _matrix(prediction, "prediction")
    if y.shape != p.shape:
        raise ValueError(f"Target/prediction shapes differ: {y.shape} vs {p.shape}")
    yc, pc = _center(y)[0], _center(p)[0]
    residual = y - p
    sse, sst = float(np.square(residual).sum()), float(np.square(yc).sum())
    return {
        "r2": None if sst == 0 else 1.0 - sse / sst,
        "status": "zero_target_variance" if sst == 0 else "ok",
        "n_contexts": len(y),
        "n_dimensions": y.shape[1],
        "sse": sse,
        "sst": sst,
        "mse_per_context": sse / len(y),
        "variance_trace": sst / len(y),
        "prediction_variance_trace": float(np.square(pc).sum() / len(y)),
        "target_prediction_covariance_trace": float((yc * pc).sum() / len(y)),
        "residual_variance_trace": float(np.square(_center(residual)[0]).sum() / len(y)),
        "squared_mean_bias": float(np.square(residual.mean(0)).sum()),
    }


def reconstruction_metrics(
    target: np.ndarray,
    component: np.ndarray,
    rest: np.ndarray,
    *,
    atol: float = 1e-9,
) -> dict:
    """Verify y=component+rest and report centered covariance cross terms explicitly."""
    y, c, r = (
        _matrix(a, n) for a, n in ((target, "target"), (component, "component"), (rest, "rest"))
    )
    if y.shape != c.shape or y.shape != r.shape:
        raise ValueError("Decomposition arrays must have identical shapes")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    error = float(np.max(np.abs(y - c - r)))
    if error > atol:
        raise ValueError(f"Component reconstruction failed: max_abs_error={error} > {atol}")
    yc, cc, rc = _center(y)[0], _center(c)[0], _center(r)[0]
    vy, vc, vr = (float(np.square(a).sum() / len(y)) for a in (yc, cc, rc))
    cross = float(2.0 * (cc * rc).sum() / len(y))
    return {
        "max_abs_reconstruction_error": error,
        "target_variance_trace": vy,
        "component_variance_trace": vc,
        "rest_variance_trace": vr,
        "twice_component_rest_covariance_trace": cross,
        "variance_decomposition_error": vy - vc - vr - cross,
    }


def per_direction_scores(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Project rows onto unit columns (d,k), or paired row-specific columns (n,d,k)."""
    values = _matrix(values, "values")
    basis = np.asarray(directions, dtype=np.float64)
    expected = basis.ndim in (2, 3) and basis.shape[-2] == values.shape[1]
    if not expected or basis.shape[-1] == 0 or not np.isfinite(basis).all():
        raise ValueError("directions must be finite (d,k) or (n,d,k) columns")
    if basis.ndim == 3 and basis.shape[0] != len(values):
        raise ValueError("Row-specific directions must pair with exactly the same rows")
    if not np.allclose(np.linalg.norm(basis, axis=-2), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Each direction must have unit norm")
    return values @ basis if basis.ndim == 2 else np.einsum("nd,ndk->nk", values, basis)


def per_direction_metrics(target_scores: np.ndarray, predicted_scores: np.ndarray) -> dict:
    """Vectorized scalar-coordinate R² and energies; NaN R² has a paired status."""
    y, p = (
        _matrix(target_scores, "target_scores"),
        _matrix(predicted_scores, "predicted_scores"),
    )
    if y.shape != p.shape:
        raise ValueError("Direction scores must have identical shapes")
    yc, pc = _center(y)[0], _center(p)[0]
    sse, sst = np.square(y - p).sum(0), np.square(yc).sum(0)
    r2 = np.full(y.shape[1], np.nan)
    np.divide(sse, sst, out=r2, where=sst > 0)
    return {
        "r2": 1.0 - r2,
        "sse": sse,
        "sst": sst,
        "variance": sst / len(y),
        "prediction_variance": np.square(pc).mean(0),
        "covariance": (yc * pc).mean(0),
        "status": tuple("ok" if value > 0 else "zero_target_variance" for value in sst),
    }


@dataclass(frozen=True)
class DirectionVarianceMatch:
    """Original direction indices matched by calibration variance, without rescaling."""

    j_indices: np.ndarray
    r_indices: np.ndarray
    log_variance_distance: np.ndarray
    j_variance: np.ndarray
    r_variance: np.ndarray
    unmatched_j_indices: np.ndarray
    unmatched_r_indices: np.ndarray
    zero_variance_j_indices: np.ndarray
    zero_variance_r_indices: np.ndarray
    log_variance_caliper: float
    status: str


def match_direction_variances(
    j_calibration: np.ndarray,
    r_calibration: np.ndarray,
    *,
    log_variance_caliper: float = 0.2,
) -> DirectionVarianceMatch:
    """Greedily match nearest log-variance J/R directions without replacement.

    Rows are the SAME paired calibration tokens/rollouts/contexts; columns are
    scores along the original normalized directions. Only indices are returned;
    no normalization or direction is changed. No held-out variance enters.
    The natural-log variance caliper is explicit (0.2 is the caller's registered
    control), and unmatched/zero-variance directions remain reported.

    The nearest cross-family pair is adjacent in the sorted union of log
    variances. A linked list plus a heap updates those candidate adjacencies in
    O(k log k) time and O(k) memory, avoiding a dense k_J by k_R distance matrix.
    Ties use stable log-variance/family/original-index order followed by J/R index
    heap ordering. This is deterministic greedy matching, not maximum-cardinality
    assignment; the actual matched denominator must be reported.
    """
    j, r = (
        _matrix(j_calibration, "j_calibration"),
        _matrix(r_calibration, "r_calibration"),
    )
    if len(j) != len(r) or len(j) < 2:
        raise ValueError("Calibration must contain at least two identically paired J/R rows")
    if not np.isfinite(log_variance_caliper) or log_variance_caliper < 0:
        raise ValueError("log_variance_caliper must be finite and nonnegative")
    jv, rv = np.square(_center(j)[0]).mean(0), np.square(_center(r)[0]).mean(0)
    ji, ri = np.flatnonzero(jv > 0), np.flatnonzero(rv > 0)
    logs = np.concatenate((np.log(jv[ji]), np.log(rv[ri])))
    family = np.concatenate((np.zeros(len(ji), dtype=int), np.ones(len(ri), dtype=int)))
    original = np.concatenate((ji, ri))
    order = np.lexsort((original, family, logs))
    logs, family, original = logs[order], family[order], original[order]
    previous = np.arange(len(logs)) - 1
    following = np.arange(len(logs)) + 1
    alive = np.ones(len(logs), dtype=bool)
    heap = []

    def add_pair(left: int, right: int) -> None:
        """Queue an adjacent opposite-family pair with a deterministic tie key."""
        if left < 0 or right >= len(logs) or family[left] == family[right]:
            return
        j_index = original[left] if family[left] == 0 else original[right]
        r_index = original[left] if family[left] == 1 else original[right]
        heapq.heappush(
            heap, (float(logs[right] - logs[left]), int(j_index), int(r_index), left, right)
        )

    for left in range(len(logs) - 1):
        add_pair(left, left + 1)
    pairs = []
    while heap:
        distance, j_index, r_index, left, right = heapq.heappop(heap)
        if not alive[left] or not alive[right] or following[left] != right:
            continue
        if distance > log_variance_caliper:
            break
        pairs.append((j_index, r_index, distance))
        before, after = int(previous[left]), int(following[right])
        alive[left] = alive[right] = False
        if before >= 0:
            following[before] = after
        if after < len(logs):
            previous[after] = before
        add_pair(before, after)
    matched_j = np.asarray([p[0] for p in pairs], dtype=np.int64)
    matched_r = np.asarray([p[1] for p in pairs], dtype=np.int64)
    unmatched_j = np.setdiff1d(np.arange(len(jv)), matched_j)
    unmatched_r = np.setdiff1d(np.arange(len(rv)), matched_r)
    return DirectionVarianceMatch(
        matched_j,
        matched_r,
        np.asarray([p[2] for p in pairs]),
        jv,
        rv,
        unmatched_j,
        unmatched_r,
        np.flatnonzero(jv == 0),
        np.flatnonzero(rv == 0),
        float(log_variance_caliper),
        "no_eligible_pairs"
        if not pairs
        else ("partial_match" if len(unmatched_j) or len(unmatched_r) else "complete_match"),
    )


@dataclass(frozen=True)
class RidgeFit:
    """Affine map in original units and the train-only normalization/selection audit."""

    weights: np.ndarray
    intercept: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: float
    selected_alpha: float
    alpha_grid: np.ndarray
    validation_sse: np.ndarray
    target_status: str
    factorization: str

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Apply the original-unit affine map to an arbitrary finite input batch."""
        values = _matrix(values, "values")
        if values.shape[1] != self.weights.shape[0]:
            raise ValueError("Prediction input dimension differs from training")
        return values @ self.weights + self.intercept


def fit_affine_ridge(
    x_train: np.ndarray,
    y_train: Mapping[str, np.ndarray],
    x_validation: np.ndarray,
    y_validation: Mapping[str, np.ndarray],
    *,
    alphas: Sequence[float],
    train_context_ids: Sequence[ContextId],
    validation_context_ids: Sequence[ContextId],
    alpha_chunk_size: int = 4,
    output_chunk_size: int = 256,
) -> dict[str, RidgeFit]:
    """Fit all target components with ONE shared eigendecomposition, select on val only.

    X columns use train population standard deviation (constant columns scale=1).
    Every Y component is centered, then scaled by one train RMS scalar, preserving
    its geometry. The returned W,b and validation SSE are in ORIGINAL units.
    The caller must supply a fixed positive increasing alpha grid; smaller alpha
    wins exact ties. Parameters fit train rows only, without a train+val refit.
    Gram-space numerics are checked against an independent direct-solve oracle in
    unit tests; production-shape parity remains a launch-time measurement duty.
    """
    xt, xv = _matrix(x_train, "x_train"), _matrix(x_validation, "x_validation")
    if xt.shape[1] != xv.shape[1] or len(xt) < 2:
        raise ValueError("Train/validation dimensions must match with at least two train rows")
    validate_context_splits({"train": train_context_ids, "validation": validation_context_ids})
    if len(train_context_ids) != len(xt) or len(validation_context_ids) != len(xv):
        raise ValueError("Context IDs must align with train and validation matrices")
    if len(set(train_context_ids)) != len(xt) or len(set(validation_context_ids)) != len(xv):
        raise ValueError("Fit context means once per context; repeated context rows are forbidden")
    if not y_train or y_train.keys() != y_validation.keys():
        raise ValueError("Train and validation must have the same nonempty target keys")
    grid = np.asarray(alphas, dtype=np.float64)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("alphas must be a finite, positive, strictly increasing fixed grid")
    if alpha_chunk_size <= 0 or output_chunk_size <= 0:
        raise ValueError("Ridge chunk sizes must be positive")
    xc, xmu = _center(xt)
    xsd = np.sqrt(np.square(xc).mean(0))
    xsd = np.where(xsd > 0, xsd, 1.0)
    xn, vn = xc / xsd, (xv - xmu) / xsd
    dual = len(xt) <= xt.shape[1]
    gram = xn @ xn.T if dual else xn.T @ xn
    eigenvalues, vectors = np.linalg.eigh(gram)
    tolerance = 100 * np.finfo(np.float64).eps * len(gram) * max(1.0, np.max(eigenvalues))
    if eigenvalues.min() < -tolerance:
        raise np.linalg.LinAlgError("Input Gram has materially negative eigenvalues")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    primal_basis = xn.T @ vectors if dual else vectors
    validation_basis = vn @ primal_basis
    targets = {}
    normalized_targets = []
    cursor = 0
    for name, value in y_train.items():
        yt, yv = (
            _matrix(value, f"y_train[{name}]"),
            _matrix(y_validation[name], f"y_validation[{name}]"),
        )
        if len(yt) != len(xt) or yv.shape != (len(xv), yt.shape[1]):
            raise ValueError(f"{name}: target rows/dimensions do not match fit splits")
        yc, mu = _center(yt)
        raw_scale = float(np.sqrt(np.square(yc).mean()))
        scale = raw_scale if raw_scale > 0 else 1.0
        targets[name] = (cursor, cursor + yt.shape[1], mu, scale, yv, raw_scale)
        normalized_targets.append(yc / scale)
        cursor += yt.shape[1]
    joined = np.concatenate(normalized_targets, axis=1)
    projected = vectors.T @ joined if dual else vectors.T @ (xn.T @ joined)
    del joined, normalized_targets, gram
    results = {}
    for name, (lo, hi, mu, scale, yv, raw_scale) in targets.items():
        coeff = projected[:, lo:hi]
        losses = np.zeros(len(grid), dtype=np.float64)
        for a0 in range(0, len(grid), alpha_chunk_size):
            a1 = min(len(grid), a0 + alpha_chunk_size)
            inverse = 1.0 / (eigenvalues[None, :] + grid[a0:a1, None])
            for d0 in range(0, hi - lo, output_chunk_size):
                d1 = min(hi - lo, d0 + output_chunk_size)
                predictions = validation_basis[None] @ (inverse[:, :, None] * coeff[None, :, d0:d1])
                predictions *= scale
                predictions += mu[d0:d1]
                losses[a0:a1] += np.square(predictions - yv[None, :, d0:d1]).sum(axis=(1, 2))
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"{name}: nonfinite validation SSE")
        selected = int(np.argmin(losses))
        normalized_weights = primal_basis @ (coeff / (eigenvalues + grid[selected])[:, None])
        weights = normalized_weights * scale / xsd[:, None]
        results[name] = RidgeFit(
            weights,
            mu - xmu @ weights,
            xmu,
            xsd,
            mu,
            scale,
            float(grid[selected]),
            grid.copy(),
            losses,
            "zero_train_variance" if raw_scale == 0 else "ok",
            "dual_gram_eigh" if dual else "primal_gram_eigh",
        )
    return results


def workspace_gap_contrasts(
    *,
    predictor: str = "ridge",
    mlp_predictor: str | None = "mlp",
    j_component: str = "J",
    j_rest: str = "restJ",
    r_component: str = "R",
    r_rest: str = "restR",
) -> dict[str, dict[str, float]]:
    """Specify G_J=restJ-J, G_R=restR-R, G_R-G_J, and MLP-ridge gains."""
    j, jr = f"{predictor}/{j_component}", f"{predictor}/{j_rest}"
    r, rr = f"{predictor}/{r_component}", f"{predictor}/{r_rest}"
    out = {
        "G_J": {jr: 1.0, j: -1.0},
        "G_R": {rr: 1.0, r: -1.0},
        "G_R_minus_G_J": {rr: 1.0, r: -1.0, jr: -1.0, j: 1.0},
    }
    if mlp_predictor is not None:
        for component in (j_component, j_rest, r_component, r_rest):
            out[f"MLP_gain/{component}"] = {
                f"{mlp_predictor}/{component}": 1.0,
                f"{predictor}/{component}": -1.0,
            }
    return out


def _bootstrap_summary(draws: np.ndarray, point: float | None, confidence: float) -> dict:
    """Return an interval only when all planned replicates have defined statistics."""
    valid = np.isfinite(draws)
    tails = ((1.0 - confidence) / 2, (1.0 + confidence) / 2)
    return {
        "estimate": point,
        "interval": np.quantile(draws, tails).tolist() if valid.all() else None,
        "confidence": confidence,
        "n_draws": len(draws),
        "n_valid_draws": int(valid.sum()),
        "status": "ok" if valid.all() else "undefined_zero_variance_draws",
    }


def _prepare_bootstrap_pool(
    targets: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    n_contexts: int,
    contrasts: Mapping[str, Mapping[str, float]] | None,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Validate paired cells and return centered targets, energies, errors, and metadata."""
    if not targets or not predictions:
        raise ValueError("Bootstrap targets and predictions cannot be empty")
    ys = {name: _matrix(y, f"targets[{name}]") for name, y in targets.items()}
    if any(len(y) != n_contexts for y in ys.values()):
        raise ValueError("All target rows must align exactly with context_ids")
    centered = {name: _center(y)[0] for name, y in ys.items()}
    energies = {name: np.square(yc).sum(1) for name, yc in centered.items()}
    errors, cell_targets, points = {}, {}, {}
    for predictor, by_target in predictions.items():
        if by_target.keys() != ys.keys():
            raise ValueError(f"{predictor}: target keys must match, without missing cells")
        for name, prediction in by_target.items():
            key = f"{predictor}/{name}"
            if key in errors:
                raise ValueError(f"Duplicate flattened predictor/target key: {key}")
            prediction = _matrix(prediction, key)
            if prediction.shape != ys[name].shape:
                raise ValueError(f"{key}: target/prediction shapes differ")
            errors[key] = np.square(prediction - ys[name]).sum(1)
            cell_targets[key] = name
            points[key] = component_metrics(ys[name], prediction)["r2"]
    contrast_map = {} if contrasts is None else dict(contrasts)
    for name, coefficients in contrast_map.items():
        if not coefficients or not set(coefficients) <= errors.keys() or name in errors:
            raise ValueError(f"Invalid contrast keys for {name}")
        if not np.isfinite(list(coefficients.values())).all():
            raise ValueError(f"Nonfinite contrast coefficients for {name}")
    return centered, energies, errors, cell_targets, points, contrast_map


def paired_context_bootstrap(
    targets: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    context_ids: Sequence[ContextId],
    *,
    n_bootstrap: int,
    seed: int,
    contrasts: Mapping[str, Mapping[str, float]] | None = None,
    chunk_size: int = 128,
    output_chunk_size: int = 256,
    confidence: float = 0.95,
    counts: np.ndarray | None = None,
) -> dict:
    """Paired contextual bootstrap with one count draw shared by EVERY target/model.

    Rows must be unique context means in the supplied ID order. No silent row
    intersection or reordering occurs. Predictions are ``predictor -> target ->
    (n_contexts,d)``; sample keys are ``predictor/target``. Each resample re-centers
    each target on its own resampled mean. Counts are generated in chunks and
    may be supplied explicitly for paired reuse/oracle checks. Peak temporary
    storage is O(chunk_size*(n_contexts+output_chunk_size)), not O(B*N*D).
    The returned count digest records identical pairing across chunk sizes.
    Arrays retain NaN for undefined draws; summaries report that coverage and
    return a null interval rather than silently discarding degenerate draws.
    """
    contexts = _ids(context_ids, "context_ids")
    n = len(contexts)
    if n < 2 or len(set(contexts)) != n:
        raise ValueError("Bootstrap requires at least two unique context means")
    if n_bootstrap <= 0 or chunk_size <= 0 or output_chunk_size <= 0:
        raise ValueError("Bootstrap draw and chunk counts must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if counts is not None:
        counts = np.asarray(counts)
        if (
            counts.shape != (n_bootstrap, n)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)
            or not np.all(counts.sum(axis=1) == n)
        ):
            raise ValueError("counts must be a nonnegative integer (B,N) matrix with row sums N")
    centered, energies, errors, cell_targets, points, contrast_map = _prepare_bootstrap_pool(
        targets, predictions, n, contrasts
    )
    keys = list(errors)
    error_matrix = np.stack([errors[key] for key in keys], axis=1)
    samples = {key: np.full(n_bootstrap, np.nan) for key in keys}
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    for lo in range(0, n_bootstrap, chunk_size):
        hi = min(lo + chunk_size, n_bootstrap)
        if counts is None:
            indices = rng.integers(0, n, size=(hi - lo, n))
            weights = np.zeros((hi - lo, n), dtype=np.int64)
            np.add.at(weights, (np.arange(hi - lo)[:, None], indices), 1)
        else:
            weights = counts[lo:hi].astype(np.int64, copy=False)
        digest.update(np.asarray(weights, dtype="<i8", order="C").tobytes())
        denominators = {}
        for name, yc in centered.items():
            squared_sum = np.zeros(hi - lo)
            for d0 in range(0, yc.shape[1], output_chunk_size):
                weighted_sum = weights @ yc[:, d0 : d0 + output_chunk_size]
                squared_sum += np.square(weighted_sum).sum(1)
            first = weights @ energies[name]
            den = first - squared_sum / n
            numerical_tol = 32 * np.finfo(np.float64).eps * np.maximum(first, squared_sum / n)
            if np.any(den < -numerical_tol):
                raise FloatingPointError("Bootstrap centered variance is materially negative")
            den[den <= numerical_tol] = 0.0
            denominators[name] = den
        sses = weights @ error_matrix
        for j, key in enumerate(keys):
            den = denominators[cell_targets[key]]
            ratio = np.full(hi - lo, np.nan)
            np.divide(sses[:, j], den, out=ratio, where=den > 0)
            samples[key][lo:hi] = 1.0 - ratio
    for name, coefficients in contrast_map.items():
        samples[name] = sum(coefficient * samples[key] for key, coefficient in coefficients.items())
        points[name] = (
            sum(coefficient * points[key] for key, coefficient in coefficients.items())
            if all(points[key] is not None for key in coefficients)
            else None
        )
    return {
        "samples": samples,
        "summary": {
            key: _bootstrap_summary(draws, points[key], confidence)
            for key, draws in samples.items()
        },
        "context_ids": contexts,
        "n_contexts": n,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "counts_sha256": digest.hexdigest(),
        "conditional_on_fitted_predictors": True,
    }
