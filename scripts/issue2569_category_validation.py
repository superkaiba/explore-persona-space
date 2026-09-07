#!/usr/bin/env python3
"""Task #2569 leg 13: direct category-level context-map validation.

This CPU-only analysis reuses task #1738's pinned held-out multi-turn capture and
labels and task #2569's frozen layer-19 ridge operator.  It asks whether
category-associated context variation is concentrated in the operator's
effective low-gain kernel and how strongly that variation is transmitted.

No model is loaded, no text is generated or labeled, and source text is never
written to the derived compact artifacts.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # Before NumPy/Torch: shared-VM thread caps freeze at import.

import csv
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import scipy.linalg as sla
import torch
from hydra.core.config_store import ConfigStore
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    SEAM,
    save_c2a_figure,
    set_c2a_style,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

LOGGER = logging.getLogger("issue2569.category_validation")

ISSUE = 2569
LAYER = 19
D_MODEL = 3584
AXES = ("topic", "language", "format", "safety")
LABEL_KEY = {
    "topic": "topic",
    "language": "language",
    "format": "format",
    "safety": "request_refusal_adjacent",
}
DISPLAY = {
    "topic": "Coarse topic/task",
    "language": "Prompt language",
    "format": "Observed answer format",
    "safety": "Request refusal-adjacency",
}
DATA_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_REVISION = "8fc7e9ceb824eed33474a954d6bc89c95c97cadf"
CAPTURE_PREFIX = "issue1738_multiturn/capture"
SPLIT_PATH = "issue1738_multiturn/sampling_manifest/split_1738.json"
MAP_REVISION = "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5"
MAP_PATH = "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
MAP_SHA256 = "188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909"
LABELS_SHA256 = "1611f2ac85142e5df24017b50f580939ee717ae9acfc12e49ff9375bc8e1ec5c"
KERNEL_RECORD_SHA256 = "643c58946bdf58e4422ad912ae00973767512232e6dec317b42c3d9f2790a0a6"
PILOT_META_SHA256 = "5a890301417374eee0c7db8d86341b6398f6c674f8e1fb645cf879838cedd26e"
CAPTURE_BYTES = 13_417_157_166
N_CAPTURE_SHARDS = 224
N_SELECTED_HOLDOUT = 10_000
N_HOLDOUT = 9_941
N_LABELED = 9_925
PRODUCER_COMMIT = "fd813b0932ce5ad92d496ff811b2a0cf0ebfd0a4"
PRODUCER_SCRIPT_SHA256 = "81ec96a951552072687fae5ced709f5b2d3d548d31e8c60ba2542379062d9d88"
BASE_COMMIT = "e1c7c8b2bd430700a926b60526adbfb44e6abe8b"
ANALYSIS_SCHEMA = "issue2569-category-validation-v1"
SEED = 25_691_738
KERNEL_EXPECTED = 1_976
RETAINED99_EXPECTED = 1_608
TAU_EXPECTED = 0.1608459354031307
BOOTSTRAP_DRAWS = 2_000
PERMUTATION_DRAWS = 999
DIRECTIONAL_DRAWS = 10_000
REFIT_BLOCK = 8
BOOTSTRAP_WORKERS = 4
BOOTSTRAP_BLAS_THREADS = 4
ASSOCIATION_WORKERS = 2
ASSOCIATION_BLAS_THREADS = 8
PREPARE_BLAS_THREADS = 16
DIRECTION_BLOCK = 256
RESPONSE_CHUNK = 128
PALETTE = {
    "topic": "#6B7280",
    "language": "#176B87",
    "format": "#B7791F",
    "safety": "#C4553D",
}
MARKERS = {"topic": "o", "language": "s", "format": "D", "safety": "^"}
ALLOWED_LABELS = {
    "topic": {
        "factual_qa",
        "creative_writing",
        "coding",
        "advice_howto",
        "chitchat_social",
        "translation",
        "math",
        "summarization_extraction",
        "harmful_or_unsafe_request",
        "roleplay_persona",
        "nsfw",
        "other",
    },
    "format": {"prose", "mixed", "list", "code"},
    "request_refusal_adjacent": {"no", "yes", "borderline"},
    "answer_is_refusal": {"no", "yes", "partial"},
}


@dataclass
class ValidationConfig:
    """Hydra configuration for extraction, analysis, rendering, and upload."""

    phase: str = "all"
    repo_root: str = "."
    out_root: str = "/workspace/data/issue_2569/category_validation"
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    permutation_draws: int = PERMUTATION_DRAWS
    directional_draws: int = DIRECTIONAL_DRAWS
    seed: int = SEED
    upload: bool = True
    production: bool = True
    max_shards: int = 0
    battery_fence_seconds: float = 7200.0
    extraction_fence_seconds: float = 7200.0
    rss_fence_gb: float = 32.0
    sentinel_dir: str = ""


ConfigStore.instance().store(name="issue2569_category_validation", node=ValidationConfig)


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-serializable value using canonical JSON encoding."""

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file and atomic replacement."""

    with atomic_replace(path, logger=LOGGER) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_text_atomic(path: Path, value: str) -> None:
    """Write text through a sibling temporary file and atomic replacement."""

    with atomic_replace(path, logger=LOGGER) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())


def append_jsonl_fsync(path: Path, value: dict[str, Any]) -> None:
    """Append one durable JSONL record and fsync it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, raw.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rss_gb() -> float:
    """Return peak resident memory in GiB on Linux."""

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def git_output(repo_root: Path, *args: str) -> str:
    """Run a read-only Git query and return stripped stdout."""

    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def _to_list(value: Any) -> list[Any]:
    """Convert a tensor, ndarray, tuple, or list to a Python list."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _prompt_chars(value: Any) -> int:
    """Count characters in the frozen serialized prompt representation."""

    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def deterministic_partition(ci: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Return an immutable corpus-stratified 60/40 train/test partition."""

    ci = np.asarray(ci, dtype=np.int64)
    corpus = np.asarray(corpus, dtype=str)
    if ci.ndim != 1 or corpus.shape != ci.shape or len(set(ci.tolist())) != ci.size:
        raise ValueError("ci/corpus arrays must be aligned, one-dimensional, and ci-unique")
    partition = np.full(ci.size, "train", dtype="U5")
    for corpus_name in sorted(set(corpus.tolist())):
        idx = np.flatnonzero(corpus == corpus_name)
        keyed = sorted(
            idx.tolist(),
            key=lambda i: int.from_bytes(
                hashlib.sha256(f"25691738:{int(ci[i])}".encode()).digest(), "big"
            ),
        )
        n_test = math.floor(0.40 * len(keyed))
        partition[np.asarray(keyed[:n_test], dtype=np.int64)] = "test"
    return partition


def collapse_rare_language(values: np.ndarray, minimum: int = 50) -> np.ndarray:
    """Collapse globally rare language labels using the preregistered count floor."""

    values = np.asarray(values, dtype=str)
    counts = Counter(values.tolist())
    return np.asarray([v if counts[v] >= minimum else "other_language" for v in values], dtype=str)


def collapse_rare_depth(values: np.ndarray, minimum: int = 50) -> np.ndarray:
    """Pool unsupported tail depths without inspecting activations or outcomes."""

    values = np.asarray(values, dtype=str)
    counts = Counter(values.tolist())
    return np.asarray([v if counts[v] >= minimum else "other_depth" for v in values], dtype=str)


def treatment_columns(values: np.ndarray, name: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Return deterministic treatment-coded columns, names, and ordered levels."""

    values = np.asarray(values, dtype=str)
    levels = sorted(set(values.tolist()))
    if len(levels) < 2:
        raise ValueError(f"{name}: requires at least two levels, got {levels}")
    columns = np.column_stack([(values == level).astype(np.float64) for level in levels[1:]])
    names = [f"{name}={level}" for level in levels[1:]]
    return columns, names, levels


def build_design(
    rows: list[dict[str, Any]], axes: tuple[str, ...] = AXES, *, include_corpus: bool = True
) -> dict[str, Any]:
    """Build the fixed full design and per-axis target-column registry."""

    n_rows = len(rows)
    values: dict[str, np.ndarray] = {}
    for axis in axes:
        key = LABEL_KEY.get(axis, axis)
        values[axis] = np.asarray([str(row[key]) for row in rows], dtype=str)
    if "language" in values:
        values["language"] = collapse_rare_language(values["language"])
    corpus = np.asarray([str(row["corpus"]) for row in rows], dtype=str)
    depth = collapse_rare_depth(np.asarray([str(int(row["depth"])) for row in rows], dtype=str))
    length = np.log1p(np.asarray([float(row["prompt_chars"]) for row in rows]))
    length_centered = length - length.mean()

    matrices: list[np.ndarray] = [np.ones((n_rows, 1), dtype=np.float64)]
    column_names = ["intercept"]
    registry: dict[str, list[int]] = {}
    levels: dict[str, list[str]] = {}

    nuisance_categories = (
        (("corpus", corpus), ("depth", depth)) if include_corpus else (("depth", depth),)
    )
    for name, array in nuisance_categories:
        matrix, names, ordered = treatment_columns(array, name)
        matrices.append(matrix)
        column_names.extend(names)
        levels[name] = ordered
    matrices.append(np.column_stack([length_centered, length_centered**2, length_centered**3]))
    column_names.extend(["log_length", "log_length_sq", "log_length_cube"])

    for axis in axes:
        matrix, names, ordered = treatment_columns(values[axis], axis)
        start = sum(part.shape[1] for part in matrices)
        matrices.append(matrix)
        registry[axis] = list(range(start, start + matrix.shape[1]))
        column_names.extend(names)
        levels[axis] = ordered
    design = np.column_stack(matrices)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError(f"full design is rank deficient: {design.shape}")
    return {
        "matrix": design,
        "column_names": column_names,
        "target_columns": registry,
        "levels": levels,
        "values": values,
        "depth_encoding": "exact integer indicators; globally n<50 pooled as other_depth",
    }


def cutoff_dimension(singular_values: np.ndarray, mass: float) -> int:
    """Return the leading dimension attaining a squared-singular-mass fraction."""

    singular_values = np.asarray(singular_values, dtype=np.float64)
    cumulative = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return int(np.searchsorted(cumulative, mass, side="left") + 1)


def fit_and_directionwise_sse(
    design_train: np.ndarray,
    outcome_train: np.ndarray,
    design_test: np.ndarray,
    outcome_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit multivariate OLS and return directionwise test SSE plus coefficients."""

    coefficient, _residual, rank, singular = np.linalg.lstsq(
        design_train, outcome_train, rcond=None
    )
    if rank != design_train.shape[1] or not np.isfinite(singular).all():
        raise RuntimeError(f"rank-deficient OLS design: rank={rank} p={design_train.shape[1]}")
    residual = outcome_test - design_test @ coefficient
    sse = np.einsum("nd,nd->d", residual, residual, optimize=True)
    if not np.isfinite(sse).all():
        raise RuntimeError("nonfinite held-out SSE")
    return sse, coefficient


def nested_improvement(
    design: np.ndarray,
    outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    target_columns: list[int],
) -> dict[str, Any]:
    """Return signed held-out directionwise full-minus-reduced SSE improvement."""

    keep = np.asarray([i for i in range(design.shape[1]) if i not in target_columns])
    reduced_sse, reduced_coef = fit_and_directionwise_sse(
        design[train][:, keep], outcomes[train], design[test][:, keep], outcomes[test]
    )
    full_sse, full_coef = fit_and_directionwise_sse(
        design[train], outcomes[train], design[test], outcomes[test]
    )
    delta = reduced_sse - full_sse
    return {
        "delta": delta,
        "full_sse": full_sse,
        "reduced_sse": reduced_sse,
        "full_coef": full_coef,
        "reduced_coef": reduced_coef,
        "reduced_columns": keep,
    }


def summarize_delta(
    delta: np.ndarray,
    reduced_sse: np.ndarray,
    singular_values: np.ndarray,
    cutoffs: dict[str, int],
) -> dict[str, float]:
    """Reduce directionwise improvements to registered category estimands."""

    delta = np.asarray(delta, dtype=np.float64)
    denominator = float(delta.sum())
    mapped = float(np.dot(singular_values**2, delta))
    result = {
        "delta_ss": denominator,
        "mapped_delta_ss": mapped,
        "incremental_r2": denominator / float(np.sum(reduced_sse)),
        "tau": mapped / denominator if denominator != 0 else float("nan"),
    }
    retained99 = cutoffs["99"]
    for name, selected in (
        ("retained_99", slice(None, retained99)),
        ("kernel_99", slice(retained99, None)),
    ):
        subspace_delta = float(delta[selected].sum())
        subspace_reduced = float(reduced_sse[selected].sum())
        result[f"{name}_delta_ss"] = subspace_delta
        result[f"{name}_incremental_r2"] = subspace_delta / subspace_reduced
    for name, retained in cutoffs.items():
        result[f"kappa_{name}"] = (
            float(delta[retained:].sum()) / denominator if denominator != 0 else float("nan")
        )
    return result


def _weighted_cross(
    design: np.ndarray, outcomes: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return batched weighted Gram and design-outcome cross-products."""

    gram = np.einsum("bn,np,nq->bpq", weights, design, design, optimize=True)
    cross = np.einsum("bn,np,nd->bpd", weights, design, outcomes, optimize=True)
    return gram, cross


def _weighted_sse_from_stats(
    coefficient: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Return batched directionwise SSE from sufficient statistics."""

    linear = np.einsum("bpd,bpd->bd", coefficient, cross, optimize=True)
    quadratic = np.einsum("bpd,bpq,bqd->bd", coefficient, gram, coefficient, optimize=True)
    return yy - 2.0 * linear + quadratic


def stratified_exponential_weights(
    rng: np.random.Generator, corpus: np.ndarray, draws: int
) -> np.ndarray:
    """Draw mean-one Bayesian-bootstrap weights independently by corpus."""

    corpus = np.asarray(corpus, dtype=str)
    weights = rng.exponential(size=(draws, corpus.size))
    for name in sorted(set(corpus.tolist())):
        idx = np.flatnonzero(corpus == name)
        weights[:, idx] *= idx.size / weights[:, idx].sum(axis=1, keepdims=True)
    return weights


def batched_weighted_nested_sse(
    design_train: np.ndarray,
    outcomes_train: np.ndarray,
    design_test: np.ndarray,
    outcomes_test: np.ndarray,
    train_weights: np.ndarray,
    test_weights: np.ndarray,
    target_columns: dict[str, list[int]],
) -> dict[str, dict[str, np.ndarray]]:
    """Complete-refit weighted nested SSE for one batch of bootstrap draws."""

    if train_weights.shape[0] > REFIT_BLOCK:
        raise ValueError(f"bootstrap block exceeds {REFIT_BLOCK}")
    gram_train, cross_train = _weighted_cross(design_train, outcomes_train, train_weights)
    gram_test, cross_test = _weighted_cross(design_test, outcomes_test, test_weights)
    yy_test = test_weights @ np.square(outcomes_test)
    beta_full = np.linalg.solve(gram_train, cross_train)
    sse_full = _weighted_sse_from_stats(beta_full, gram_test, cross_test, yy_test)
    output: dict[str, dict[str, np.ndarray]] = {}
    for axis, dropped in target_columns.items():
        keep = np.asarray([i for i in range(design_train.shape[1]) if i not in dropped])
        gram_train_reduced = gram_train[:, keep][:, :, keep]
        cross_train_reduced = cross_train[:, keep]
        beta_reduced = np.linalg.solve(gram_train_reduced, cross_train_reduced)
        sse_reduced = _weighted_sse_from_stats(
            beta_reduced,
            gram_test[:, keep][:, :, keep],
            cross_test[:, keep],
            yy_test,
        )
        output[axis] = {"delta": sse_reduced - sse_full, "reduced_sse": sse_reduced}
    return output


def exchangeability_blocks(
    corpus: np.ndarray, depth: np.ndarray, prompt_chars: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build fixed corpus × capped-depth × global-length-quartile blocks."""

    log_length = np.log1p(np.asarray(prompt_chars, dtype=np.float64))
    cuts = np.quantile(log_length, [0.25, 0.50, 0.75])
    quartile = np.searchsorted(cuts, log_length, side="right")
    keys = np.asarray(
        [f"{c}|{min(int(d), 4)}|{int(q)}" for c, d, q in zip(corpus, depth, quartile)],
        dtype=str,
    )
    counts = Counter(keys.tolist())
    small_fraction = sum(counts[k] for k in counts if counts[k] < 5) / len(keys)
    return keys, {
        "length_log_quartile_cuts": cuts.tolist(),
        "n_blocks": len(counts),
        "min_block": min(counts.values()),
        "small_block_fraction": small_fraction,
        "block_counts": dict(sorted(counts.items())),
    }


def summarize_blocks(blocks: np.ndarray) -> dict[str, Any]:
    """Summarize realized fixed exchangeability-block sizes."""

    counts = Counter(np.asarray(blocks, dtype=str).tolist())
    return {
        "n_blocks": len(counts),
        "min_block": min(counts.values()),
        "small_block_fraction": sum(count for count in counts.values() if count < 5) / len(blocks),
        "block_counts": dict(sorted(counts.items())),
    }


def _permutations_within_blocks(
    rng: np.random.Generator, blocks: np.ndarray, draws: int
) -> np.ndarray:
    """Return draw-by-row permutation indices restricted to fixed blocks."""

    result = np.tile(np.arange(blocks.size, dtype=np.int64), (draws, 1))
    for key in sorted(set(blocks.tolist())):
        idx = np.flatnonzero(blocks == key)
        random_keys = rng.random((draws, idx.size))
        result[:, idx] = idx[np.argsort(random_keys, axis=1)]
    return result


def _leverage(design_train: np.ndarray, design_eval: np.ndarray) -> np.ndarray:
    """Return diagonal hat leverage against the training Gram matrix."""

    inverse = np.linalg.inv(design_train.T @ design_train)
    return np.einsum("np,pq,nq->n", design_eval, inverse, design_eval, optimize=True)


def prepare_freedman_lane(
    design_train: np.ndarray,
    outcome_train: np.ndarray,
    design_test: np.ndarray,
    outcome_test: np.ndarray,
    dropped: list[int],
) -> dict[str, Any]:
    """Precompute the fixed reduced-model objects for Freedman–Lane draws."""

    keep = np.asarray([i for i in range(design_train.shape[1]) if i not in dropped])
    ztr = design_train[:, keep]
    zte = design_test[:, keep]
    reduced_coef, _residual, rank, _singular = np.linalg.lstsq(ztr, outcome_train, rcond=None)
    if rank != ztr.shape[1]:
        raise RuntimeError("reduced Freedman-Lane design is rank deficient")
    fitted_train = ztr @ reduced_coef
    fitted_test = zte @ reduced_coef
    train_scale = np.sqrt(np.maximum(1.0 - _leverage(ztr, ztr), 1e-8))
    test_scale = np.sqrt(1.0 + np.maximum(_leverage(ztr, zte), 0.0))
    return {
        "ztr": ztr,
        "zte": zte,
        "fitted_train": fitted_train,
        "fitted_test": fitted_test,
        "student_train": (outcome_train - fitted_train) / train_scale[:, None],
        "student_test": (outcome_test - fitted_test) / test_scale[:, None],
        "train_scale": train_scale,
        "test_scale": test_scale,
        "gram_full": design_train.T @ design_train,
        "gram_reduced": ztr.T @ ztr,
        "gram_test_full": design_test.T @ design_test,
        "gram_test_reduced": zte.T @ zte,
    }


def batched_freedman_lane_outcome(
    rng: np.random.Generator,
    design_train: np.ndarray,
    outcome_train: np.ndarray,
    design_test: np.ndarray,
    outcome_test: np.ndarray,
    dropped: list[int],
    train_blocks: np.ndarray,
    test_blocks: np.ndarray,
    draws: int,
    prepared: dict[str, Any] | None = None,
    response_chunk: int = RESPONSE_CHUNK,
    retained: int | None = None,
) -> np.ndarray:
    """Return full and optional retained/kernel null gains without a draw×row×width array."""

    if draws > REFIT_BLOCK:
        raise ValueError(f"permutation block exceeds {REFIT_BLOCK}")
    fixed = prepared or prepare_freedman_lane(
        design_train, outcome_train, design_test, outcome_test, dropped
    )
    ztr = fixed["ztr"]
    zte = fixed["zte"]
    p_train = _permutations_within_blocks(rng, train_blocks, draws)
    p_test = _permutations_within_blocks(rng, test_blocks, draws)
    if response_chunk <= 0:
        raise ValueError("response_chunk must be positive")
    width = outcome_train.shape[1]
    if retained is not None and not 0 < retained < width:
        raise ValueError(f"retained cutoff must be inside response width, got {retained}/{width}")
    delta_totals = np.zeros((draws, 3), dtype=np.float64) if retained else np.zeros(draws)
    reduced_totals = np.zeros((draws, 2), dtype=np.float64) if retained else None
    for lower in range(0, width, response_chunk):
        upper = min(lower + response_chunk, width)
        null_train = (
            fixed["fitted_train"][None, :, lower:upper]
            + fixed["student_train"][p_train, lower:upper] * fixed["train_scale"][None, :, None]
        )
        null_test = (
            fixed["fitted_test"][None, :, lower:upper]
            + fixed["student_test"][p_test, lower:upper] * fixed["test_scale"][None, :, None]
        )
        cross_full = np.einsum("np,bnd->bpd", design_train, null_train, optimize=True)
        cross_reduced = np.einsum("np,bnd->bpd", ztr, null_train, optimize=True)
        beta_full = np.linalg.solve(fixed["gram_full"][None], cross_full)
        beta_reduced = np.linalg.solve(fixed["gram_reduced"][None], cross_reduced)
        yy = np.einsum("bnd,bnd->bd", null_test, null_test, optimize=True)
        cross_test_full = np.einsum("np,bnd->bpd", design_test, null_test, optimize=True)
        cross_test_reduced = np.einsum("np,bnd->bpd", zte, null_test, optimize=True)
        sse_full = _weighted_sse_from_stats(
            beta_full,
            np.broadcast_to(fixed["gram_test_full"], (draws,) + fixed["gram_test_full"].shape),
            cross_test_full,
            yy,
        )
        sse_reduced = _weighted_sse_from_stats(
            beta_reduced,
            np.broadcast_to(
                fixed["gram_test_reduced"], (draws,) + fixed["gram_test_reduced"].shape
            ),
            cross_test_reduced,
            yy,
        )
        delta = sse_reduced - sse_full
        if retained is None:
            delta_totals += np.sum(delta, axis=1)
        else:
            delta_totals[:, 0] += np.sum(delta, axis=1)
            retained_in_chunk = max(0, min(upper, retained) - lower)
            if retained_in_chunk:
                delta_totals[:, 1] += np.sum(delta[:, :retained_in_chunk], axis=1)
                reduced_totals[:, 0] += np.sum(sse_reduced[:, :retained_in_chunk], axis=1)
            if retained_in_chunk < upper - lower:
                delta_totals[:, 2] += np.sum(delta[:, retained_in_chunk:], axis=1)
                reduced_totals[:, 1] += np.sum(sse_reduced[:, retained_in_chunk:], axis=1)
    if retained is None:
        return delta_totals
    return np.column_stack(
        [
            delta_totals,
            delta_totals[:, 0] / np.sum(reduced_totals, axis=1),
            delta_totals[:, 1] / reduced_totals[:, 0],
            delta_totals[:, 2] / reduced_totals[:, 1],
        ]
    )


def residual_variance_diagnostics(
    prepared: dict[str, Any], train_blocks: np.ndarray, test_blocks: np.ndarray
) -> dict[str, Any]:
    """Summarize reduced-model residual energy by fixed exchangeability block."""

    output: dict[str, Any] = {}
    gate_pass = True
    for partition, block_values in (("train", train_blocks), ("test", test_blocks)):
        row_energy = np.mean(np.square(prepared[f"student_{partition}"]), axis=1)
        block_records: dict[str, Any] = {}
        eligible_means = []
        for key in sorted(set(block_values.tolist())):
            selected = block_values == key
            mean = float(np.mean(row_energy[selected]))
            record = {"n": int(selected.sum()), "mean_studentized_residual_energy": mean}
            block_records[str(key)] = record
            if record["n"] >= 5:
                eligible_means.append(mean)
        eligible = np.asarray(eligible_means, dtype=np.float64)
        valid = bool(eligible.size and np.isfinite(eligible).all() and np.all(eligible > 0))
        gate_pass = gate_pass and valid
        output[partition] = {
            "eligible_block_count": int(eligible.size),
            "finite_positive_gate": valid,
            "min_mean_energy": float(eligible.min()) if eligible.size else None,
            "max_mean_energy": float(eligible.max()) if eligible.size else None,
            "max_to_min_ratio": float(eligible.max() / eligible.min()) if valid else None,
            "blocks": block_records,
        }
    output["gate_pass"] = bool(gate_pass)
    output["interpretation"] = (
        "The finite-positive gate is required for null validity; max/min ratios are disclosed "
        "descriptively because no heteroskedasticity threshold was preregistered."
    )
    return output


def batched_matched_rank_null(
    rng: np.random.Generator,
    deltas: dict[str, np.ndarray],
    kernel_dim: int,
    draws: int,
) -> np.ndarray:
    """Draw selection-symmetric pseudo-kernels and return five concentration reads."""

    if draws > DIRECTION_BLOCK:
        raise ValueError(f"directional block exceeds {DIRECTION_BLOCK}")
    d_model = next(iter(deltas.values())).size
    random_keys = rng.random((draws, d_model))
    chosen = np.argpartition(random_keys, kernel_dim - 1, axis=1)[:, :kernel_dim]
    mask = np.zeros((draws, d_model), dtype=np.float64)
    np.put_along_axis(mask, chosen, 1.0, axis=1)
    values = []
    for axis in AXES:
        delta = np.asarray(deltas[axis], dtype=np.float64)
        values.append(mask @ delta / delta.sum())
    stacked = np.column_stack(values)
    contrast = stacked[:, 0] - stacked[:, 1:].mean(axis=1)
    return np.column_stack([stacked, contrast])


def percentile_interval(values: np.ndarray) -> list[float]:
    """Return the preregistered percentile 95% interval."""

    return np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975]).tolist()


def benjamini_hochberg(p_values: dict[str, float], q: float = 0.05) -> dict[str, Any]:
    """Apply Benjamini–Hochberg to a named p-value family."""

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    passed_rank = 0
    for rank, (_name, value) in enumerate(ordered, start=1):
        if value <= q * rank / m:
            passed_rank = rank
    rejected = {name: rank <= passed_rank for rank, (name, _value) in enumerate(ordered, start=1)}
    return {"q": q, "rejected": rejected, "ordered": ordered}


def evaluate_verdict(
    axes: dict[str, dict[str, Any]],
    contrasts: dict[str, Any],
    standardized: dict[str, Any],
    provenance_ok: bool,
    pilot_ok: bool,
    residual_null_ok: bool,
) -> dict[str, Any]:
    """Apply the preregistered ordered, disjoint verdict lattice."""

    if not provenance_ok or not pilot_ok:
        return {"verdict": "abort", "reason": "provenance/extraction/pilot gate failed"}
    all_interpretable = (
        all(value["recoverability_pass"] and value["mapped_gain_pass"] for value in axes.values())
        and residual_null_ok
        and standardized["interpretability_pass"]
    )
    if not all_interpretable:
        return {
            "verdict": "partial/inconclusive",
            "reason": (
                "at least one axis failed raw/standardized interpretability, mapped-gain, "
                "or null calibration"
            ),
        }
    individual_kappa = contrasts["individual_kappa"]
    individual_gain = contrasts["individual_gain"]
    strong = (
        contrasts["delta_kappa_ci"][0] > 0
        and contrasts["delta_gain_ci"][0] > 0
        and sum(value > 0 for value in individual_kappa.values()) >= 2
        and sum(value > 0 for value in individual_gain.values()) >= 2
        and standardized["delta_kappa"] > 0
        and standardized["delta_gain"] > 0
    )
    if strong:
        return {"verdict": "strong support", "reason": "all prespecified support criteria passed"}
    reversed_axes = sum(
        individual_kappa[axis] < 0 and individual_gain[axis] < 0
        for axis in ("language", "format", "safety")
    )
    refuted = (
        contrasts["delta_kappa"] < 0
        and contrasts["delta_gain"] < 0
        and (contrasts["delta_kappa_ci"][1] < 0 or contrasts["delta_gain_ci"][1] < 0)
    ) or reversed_axes >= 2
    if refuted:
        return {"verdict": "refuted", "reason": "prespecified reversed-contrast criterion passed"}
    return {"verdict": "partial/inconclusive", "reason": "support and refutation criteria not met"}


def _load_labels(path: Path) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """Load and validate the immutable #1738 label artifact."""

    payload = json.loads(path.read_text())
    labels = {int(ci): dict(value) for ci, value in payload["labels"].items()}
    if len(labels) != N_LABELED or payload["n_labeled"] != N_LABELED:
        raise RuntimeError(f"expected {N_LABELED} labels, found {len(labels)}")
    if payload.get("n_items") != N_HOLDOUT:
        raise RuntimeError(f"label artifact n_items={payload.get('n_items')} != {N_HOLDOUT}")
    required = set(LABEL_KEY.values()) | {"answer_is_refusal"}
    for ci, value in labels.items():
        missing = required - set(value)
        if missing:
            raise RuntimeError(f"ci={ci}: missing labels {sorted(missing)}")
        for key, allowed in ALLOWED_LABELS.items():
            if value[key] not in allowed:
                raise RuntimeError(f"ci={ci}: invalid {key} label {value[key]!r}")
        language = value["language"]
        if not isinstance(language, str) or not language or len(language) > 24:
            raise RuntimeError(f"ci={ci}: invalid language label {language!r}")
    return labels, payload


def _capture_entries() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """List the pinned capture prefix once and return immutable shard metadata."""

    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: immutable scoped listing is retried by its wrapper.
            api.list_repo_tree(
                DATA_REPO,
                path_in_repo=CAPTURE_PREFIX,
                repo_type="dataset",
                recursive=True,
                revision=CAPTURE_REVISION,
            )
        ),
        what="list pinned issue1738 capture prefix",
    )
    total_bytes = sum(int(entry.size or 0) for entry in entries)
    shards = [entry for entry in entries if entry.path.endswith(".pt")]
    if len(entries) != N_CAPTURE_SHARDS + 1 or len(shards) != N_CAPTURE_SHARDS:
        raise RuntimeError(f"capture listing mismatch: entries={len(entries)} shards={len(shards)}")
    if total_bytes != CAPTURE_BYTES:
        raise RuntimeError(f"capture byte count {total_bytes} != {CAPTURE_BYTES}")
    records = []
    for entry in sorted(shards, key=lambda value: value.path):
        lfs_sha = getattr(getattr(entry, "lfs", None), "sha256", None)
        if not lfs_sha:
            raise RuntimeError(f"capture shard lacks LFS SHA-256: {entry.path}")
        records.append({"path": entry.path, "size": int(entry.size), "sha256": lfs_sha})
    return records, {
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "shards": records,
    }


def _validate_bundle(bundle: dict[str, Any], path: str) -> tuple[list[int], int]:
    """Validate a capture bundle and return its row ids and L19 layer index."""

    required = {"cx_last", "ci", "depth", "corpus", "prompts", "layers"}
    missing = required - set(bundle)
    if missing:
        raise RuntimeError(f"{path}: missing keys {sorted(missing)}")
    layers = [int(value) for value in _to_list(bundle["layers"])]
    if layers.count(LAYER) != 1:
        raise RuntimeError(f"{path}: layer {LAYER} occurs {layers.count(LAYER)} times")
    ci = [int(value) for value in _to_list(bundle["ci"])]
    cx = bundle["cx_last"]
    if tuple(cx.shape) != (len(ci), len(layers), D_MODEL):
        raise RuntimeError(f"{path}: cx_last shape {tuple(cx.shape)}")
    for key in ("depth", "corpus", "prompts"):
        if len(bundle[key]) != len(ci):
            raise RuntimeError(f"{path}: {key} length mismatch")
    if not torch.isfinite(cx[:, layers.index(LAYER)]).all():
        raise RuntimeError(f"{path}: nonfinite layer-{LAYER} context states")
    return ci, layers.index(LAYER)


def _producer_compatibility(repo_root: Path) -> dict[str, Any]:
    """Verify the immutable producer source and record its relevant conventions."""

    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "show",
            f"{PRODUCER_COMMIT}:scripts/issue1738_multiturn_generate_capture.py",
        ]
    )
    observed_sha = hashlib.sha256(source).hexdigest()
    if observed_sha != PRODUCER_SCRIPT_SHA256:
        raise RuntimeError(f"producer source SHA {observed_sha} != registered SHA")
    text = source.decode()
    required_fragments = (
        "DEFAULT_MODEL = N1M.DEFAULT_MODEL",
        "CAPTURE_LAYERS = list(N1M.CAPTURE_LAYERS)",
        "extract_layer_activations",
        "cx.append(hs[-1, :].float().cpu())",
        '"cx_last"',
        '"layers"',
    )
    absent = [fragment for fragment in required_fragments if fragment not in text]
    if absent:
        raise RuntimeError(f"producer source missing registered fragments: {absent}")
    return {
        "commit": PRODUCER_COMMIT,
        "script_sha256": observed_sha,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "generation_suffix": "<|im_start|>assistant\n",
        "layers": [14, 19, 26],
        "context_read": "final prompt token hs[-1, :]",
        "required_fragments": list(required_fragments),
    }


def _stage_small_inputs(repo_root: Path, out_root: Path) -> dict[str, Path]:
    """Stage the pinned split, pilot metadata, and map through canonical helpers."""

    frozen = out_root / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    split = hub.stage_hub_file(
        DATA_REPO,
        SPLIT_PATH,
        frozen / "split_1738.json",
        repo_type="dataset",
        revision=CAPTURE_REVISION,
    )
    pilot = hub.stage_hub_file(
        DATA_REPO,
        f"{CAPTURE_PREFIX}/pilot_meta.json",
        frozen / "pilot_meta.json",
        repo_type="dataset",
        revision=CAPTURE_REVISION,
    )
    ridge = hub.stage_hub_file(
        DATA_REPO,
        MAP_PATH,
        frozen / "ridge_L19.pt",
        repo_type="dataset",
        revision=MAP_REVISION,
        size_bytes=51_425_703,
    )
    if sha256_file(ridge) != MAP_SHA256:
        raise RuntimeError("staged ridge payload failed SHA-256 gate")
    if sha256_file(pilot) != PILOT_META_SHA256:
        raise RuntimeError("staged pilot metadata failed SHA-256 gate")
    pilot_payload = json.loads(pilot.read_text())
    required_pilot = {"kept", "chunks", "wall_h", "ctx_per_gpu_h", "violation_rate", "g1"}
    if required_pilot - set(pilot_payload):
        raise RuntimeError("staged pilot metadata is missing registered fields")
    if pilot_payload["kept"] != 600 or pilot_payload["chunks"] != 2:
        raise RuntimeError("staged pilot metadata has unexpected coverage")
    expected_gates = {"violation_rate_ok", "prefix_varies_ok", "rate_ok"}
    if expected_gates - set(pilot_payload["g1"]) or not all(
        pilot_payload["g1"][key] is True for key in expected_gates
    ):
        raise RuntimeError("staged pilot metadata failed registered G1 gates")
    labels = repo_root / "eval_results/issue_1738/judge_labels/labels.json"
    kernel = repo_root / "eval_results/issue_2569/weights/leg8/effective_kernel_L19.json"
    if not labels.is_file() or not kernel.is_file():
        raise FileNotFoundError("committed labels or effective-kernel record is absent")
    if sha256_file(labels) != LABELS_SHA256:
        raise RuntimeError("committed labels failed registered SHA-256 gate")
    if sha256_file(kernel) != KERNEL_RECORD_SHA256:
        raise RuntimeError("committed effective-kernel record failed registered SHA-256 gate")
    return {"split": split, "pilot": pilot, "ridge": ridge, "labels": labels, "kernel": kernel}


def _config_fingerprint(
    cfg: ValidationConfig, script_sha: str, source_paths: dict[str, Path], listing: dict[str, Any]
) -> dict[str, Any]:
    """Build the complete immutable analysis fingerprint."""

    document = {
        "schema": ANALYSIS_SCHEMA,
        "capture_revision": CAPTURE_REVISION,
        "capture_prefix": CAPTURE_PREFIX,
        "capture_listing": listing,
        "split_path": SPLIT_PATH,
        "split_sha256": sha256_file(source_paths["split"]),
        "labels_sha256": sha256_file(source_paths["labels"]),
        "pilot_meta_sha256": sha256_file(source_paths["pilot"]),
        "map_revision": MAP_REVISION,
        "map_path": MAP_PATH,
        "map_sha256": MAP_SHA256,
        "kernel_record_sha256": sha256_file(source_paths["kernel"]),
        "producer_commit": PRODUCER_COMMIT,
        "producer_script_sha256": PRODUCER_SCRIPT_SHA256,
        "script_sha256": script_sha,
        "split_rule": "corpus-stratified SHA256 rank 60/40 using 25691738:ci",
        "axes": list(AXES),
        "cutoffs": [0.90, 0.95, 0.99],
        "draws": {
            "bootstrap": cfg.bootstrap_draws,
            "permutation": cfg.permutation_draws,
            "directional": cfg.directional_draws,
        },
        "parallelism": {
            "refit_block": REFIT_BLOCK,
            "bootstrap_workers": BOOTSTRAP_WORKERS,
            "bootstrap_blas_threads": BOOTSTRAP_BLAS_THREADS,
            "association_workers": ASSOCIATION_WORKERS,
            "association_blas_threads": ASSOCIATION_BLAS_THREADS,
            "prepare_blas_threads": PREPARE_BLAS_THREADS,
            "response_chunk": RESPONSE_CHUNK,
        },
        "seed": cfg.seed,
    }
    document["config_sha256"] = canonical_sha256(document)
    return document


def extract_compact(cfg: ValidationConfig, repo_root: Path, out_root: Path) -> dict[str, Any]:
    """Stream pinned shards, retain L19 labeled rows, and persist compact inputs."""

    started = time.monotonic()
    free_bytes = shutil.disk_usage(out_root).free
    if free_bytes < 24_000_000_000:
        raise RuntimeError(f"staging volume has only {free_bytes / 1e9:.2f} GB free; need 24 GB")
    source_paths = _stage_small_inputs(repo_root, out_root)
    producer = _producer_compatibility(repo_root)
    shards, listing = _capture_entries()
    if cfg.max_shards:
        shards = shards[: cfg.max_shards]
    script_sha = sha256_file(Path(__file__).resolve())
    fingerprint = _config_fingerprint(cfg, script_sha, source_paths, listing)
    config_short = fingerprint["config_sha256"][:12]
    compact = out_root / "compact"
    staging = out_root / "staging"
    compact.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    config_path = compact / "config_manifest.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != fingerprint:
            raise RuntimeError("existing compact config fingerprint differs; refusing overwrite")
    else:
        write_json_atomic(config_path, fingerprint)

    labels, label_payload = _load_labels(source_paths["labels"])
    split = json.loads(source_paths["split"].read_text())
    holdout = [int(value) for value in split["sets"]["holdout"]["ci"]]
    if len(holdout) != N_SELECTED_HOLDOUT or len(set(holdout)) != N_SELECTED_HOLDOUT:
        raise RuntimeError(f"selected holdout roster is not {N_SELECTED_HOLDOUT} unique ids")
    if not set(labels).issubset(set(holdout)):
        raise RuntimeError("labeled ids are not a subset of the frozen holdout")
    row_order = sorted(labels)
    row_index = {ci: index for index, ci in enumerate(row_order)}
    matrix_path = compact / "activations_L19.npy"
    ledger_path = compact / "shard_ledger.jsonl"
    completed_records: list[dict[str, Any]] = []
    if ledger_path.exists():
        completed_records = [
            json.loads(line) for line in ledger_path.read_text().splitlines() if line
        ]
        for index, record in enumerate(completed_records):
            if record["config_sha256"] != fingerprint["config_sha256"]:
                raise RuntimeError("ledger config mismatch")
            expected = shards[index]
            if any(record[key] != expected[key] for key in ("path", "size", "sha256")):
                raise RuntimeError(f"ledger source mismatch at completed shard {index}")
        if not matrix_path.exists():
            raise RuntimeError("resume ledger exists but compact activation matrix is absent")
        matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
        if matrix.shape != (len(row_order), D_MODEL) or matrix.dtype != np.float32:
            raise RuntimeError("resume activation matrix shape/dtype mismatch")
    else:
        matrix = np.lib.format.open_memmap(
            matrix_path, mode="w+", dtype=np.float32, shape=(len(row_order), D_MODEL)
        )
    holdout_set = set(holdout)
    seen_holdout: set[int] = set()
    seen_labeled: set[int] = set()
    metadata: dict[int, dict[str, Any]] = {}
    for record in completed_records:
        for row in record["holdout_rows"]:
            ci = int(row["ci"])
            if ci in seen_holdout:
                raise RuntimeError(f"duplicate holdout ci in resume ledger: {ci}")
            seen_holdout.add(ci)
            metadata[ci] = row
        for ci in record["labeled_target_ids"]:
            ci = int(ci)
            if ci in seen_labeled:
                raise RuntimeError(f"duplicate labeled ci in resume ledger: {ci}")
            seen_labeled.add(ci)
        stale_stage = staging / Path(record["path"]).name
        if stale_stage.exists():
            if (
                stale_stage.stat().st_size != record["size"]
                or sha256_file(stale_stage) != record["sha256"]
            ):
                raise RuntimeError(f"resume staging residue differs from source: {stale_stage}")
            stale_stage.unlink()
    pilot_path = compact / "pilot_report.json"
    first_intersect_report = json.loads(pilot_path.read_text()) if pilot_path.exists() else None

    for shard_number, record in enumerate(
        shards[len(completed_records) :], start=len(completed_records) + 1
    ):
        shard_started = time.monotonic()
        local = staging / Path(record["path"]).name
        print(
            f"[extract] unit {shard_number}/{len(shards)} {record['path']} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        hub.stage_hub_file(
            DATA_REPO,
            record["path"],
            local,
            repo_type="dataset",
            revision=CAPTURE_REVISION,
            size_bytes=record["size"],
        )
        if local.stat().st_size != record["size"] or sha256_file(local) != record["sha256"]:
            raise RuntimeError(f"{record['path']}: staged size/SHA mismatch")
        bundle = torch.load(local, map_location="cpu", weights_only=False, mmap=True)
        ci_values, layer_index = _validate_bundle(bundle, record["path"])
        depths = _to_list(bundle["depth"])
        corpora = _to_list(bundle["corpus"])
        prompts = _to_list(bundle["prompts"])
        cx = bundle["cx_last"][:, layer_index]
        hits = 0
        shard_holdout_rows = []
        shard_labeled_ids = []
        shard_output_offsets = []
        for source_row, ci in enumerate(ci_values):
            if ci not in holdout_set:
                continue
            if ci in seen_holdout:
                raise RuntimeError(f"duplicate holdout ci across capture: {ci}")
            seen_holdout.add(ci)
            metadata[ci] = {
                "ci": ci,
                "corpus": str(corpora[source_row]),
                "depth": int(depths[source_row]),
                "prompt_chars": _prompt_chars(prompts[source_row]),
            }
            shard_holdout_rows.append(metadata[ci])
            if ci in row_index:
                matrix[row_index[ci]] = cx[source_row].detach().cpu().numpy().astype(np.float32)
                seen_labeled.add(ci)
                shard_labeled_ids.append(ci)
                shard_output_offsets.append(row_index[ci])
                hits += 1
        matrix.flush()
        ledger = {
            **record,
            "config_sha256": fingerprint["config_sha256"],
            "target_ids_written": hits,
            "labeled_target_ids": shard_labeled_ids,
            "output_offsets": shard_output_offsets,
            "holdout_rows": shard_holdout_rows,
            "holdout_seen_total": len(seen_holdout),
            "labeled_seen_total": len(seen_labeled),
            "elapsed_seconds": time.monotonic() - shard_started,
        }
        append_jsonl_fsync(ledger_path, ledger)
        del cx, bundle
        local.unlink()
        if hits and first_intersect_report is None:
            rate = record["size"] / max(ledger["elapsed_seconds"], 1e-9)
            extrapolated = listing["total_bytes"] / rate * 2.0
            first_intersect_report = {
                "path": record["path"],
                "bytes_per_second": rate,
                "rows_per_second": hits / max(ledger["elapsed_seconds"], 1e-9),
                "load_peak_rss_gb": rss_gb(),
                "two_x_extrapolated_seconds": extrapolated,
                "passes_extraction_fence": extrapolated <= cfg.extraction_fence_seconds,
            }
            write_json_atomic(pilot_path, first_intersect_report)
            if not first_intersect_report["passes_extraction_fence"]:
                raise RuntimeError(f"extraction pilot exceeded fence: {first_intersect_report}")

    matrix.flush()
    if cfg.max_shards:
        return {"status": "partial-extraction", "shards": len(shards), "config": fingerprint}
    if len(seen_holdout) != N_HOLDOUT or seen_labeled != set(labels):
        raise RuntimeError(
            f"coverage mismatch: captured holdout {len(seen_holdout)}/{N_HOLDOUT}, "
            f"labeled {len(seen_labeled)}/{N_LABELED}"
        )
    rows = []
    for ci in row_order:
        rows.append({**metadata[ci], **labels[ci]})
    partitions = deterministic_partition(
        np.asarray(row_order), np.asarray([row["corpus"] for row in rows])
    )
    for row, partition in zip(rows, partitions):
        row["partition"] = str(partition)
    write_json_atomic(compact / "rows.json", rows)
    missing_ids = sorted(seen_holdout - set(labels))
    missing_rows = [metadata[ci] for ci in missing_ids]
    labeled_summary = {
        "corpus": dict(Counter(row["corpus"] for row in rows)),
        "depth": dict(Counter(str(row["depth"]) for row in rows)),
        "prompt_chars": {
            "mean": float(np.mean([row["prompt_chars"] for row in rows])),
            "median": float(np.median([row["prompt_chars"] for row in rows])),
        },
    }
    missing_summary = {
        "corpus": dict(Counter(row["corpus"] for row in missing_rows)),
        "depth": dict(Counter(str(row["depth"]) for row in missing_rows)),
        "prompt_chars": {
            "mean": float(np.mean([row["prompt_chars"] for row in missing_rows])),
            "median": float(np.median([row["prompt_chars"] for row in missing_rows])),
        },
    }
    write_json_atomic(
        compact / "missingness_audit.json",
        {"missing_ids": missing_ids, "labeled": labeled_summary, "unlabeled": missing_summary},
    )
    kernel_record = json.loads(source_paths["kernel"].read_text())
    if (
        kernel_record["k99"] != RETAINED99_EXPECTED
        or kernel_record["kernel_dim"] != KERNEL_EXPECTED
        or not math.isclose(kernel_record["tau_kernel"], TAU_EXPECTED, rel_tol=1e-12)
    ):
        raise RuntimeError("committed effective-kernel record failed registered gate")
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "config": fingerprint,
        "config_short": config_short,
        "producer_compatibility": producer,
        "capture_pilot_meta": json.loads(source_paths["pilot"].read_text()),
        "capture_listing": listing,
        "coverage": {"holdout": len(seen_holdout), "labeled": len(seen_labeled)},
        "matrix": {
            "path": matrix_path.name,
            "shape": [N_LABELED, D_MODEL],
            "dtype": "float32",
            "sha256": sha256_file(matrix_path),
            "size": matrix_path.stat().st_size,
        },
        "rows_sha256": sha256_file(compact / "rows.json"),
        "ledger_sha256": sha256_file(ledger_path),
        "pilot": first_intersect_report,
        "label_kappas": label_payload["test_retest_kappa"],
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gb": rss_gb(),
    }
    write_json_atomic(compact / "extraction_manifest.json", manifest)
    return manifest


def _load_operator(path: Path) -> dict[str, Any]:
    """Load the frozen ridge payload and compute raw/standardized SVD systems."""

    if sha256_file(path) != MAP_SHA256:
        raise RuntimeError("ridge SHA changed before analysis")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("kind") != "ridge" or payload.get("fitter") != "ridge":
        raise RuntimeError("map payload is not the registered ridge fitter")
    if int(payload.get("layer", -1)) != LAYER:
        raise RuntimeError("map layer mismatch")
    w = np.asarray(payload["W"], dtype=np.float64)
    xmu = np.asarray(payload["xmu"], dtype=np.float64)
    xsd = np.asarray(payload["xsd"], dtype=np.float64)
    ymu = np.asarray(payload["ymu"], dtype=np.float64)
    if w.shape != (D_MODEL, D_MODEL) or any(value.shape != (D_MODEL,) for value in (xmu, xsd, ymu)):
        raise RuntimeError("map component shape mismatch")
    if not all(np.isfinite(value).all() for value in (w, xmu, xsd, ymu)) or not (xsd > 0).all():
        raise RuntimeError("map payload contains invalid values")
    operator = w / xsd[:, None]
    LOGGER.info("[analysis] raw operator SVD", extra={"flush": True})
    u_raw, s_raw, _vh_raw = sla.svd(operator, full_matrices=False, lapack_driver="gesdd")
    LOGGER.info("[analysis] standardized operator SVD", extra={"flush": True})
    u_standard, s_standard, _vh_standard = sla.svd(w, full_matrices=False, lapack_driver="gesdd")
    raw_cutoffs = {
        str(int(100 * mass)): cutoff_dimension(s_raw, mass) for mass in (0.90, 0.95, 0.99)
    }
    standard_cutoffs = {"99": cutoff_dimension(s_standard, 0.99)}
    if raw_cutoffs["99"] != RETAINED99_EXPECTED:
        raise RuntimeError(f"raw k99={raw_cutoffs['99']} != {RETAINED99_EXPECTED}")
    if D_MODEL - raw_cutoffs["99"] != KERNEL_EXPECTED:
        raise RuntimeError("raw effective-kernel dimension mismatch")
    retained = raw_cutoffs["99"]
    if not (s_raw[retained] < TAU_EXPECTED <= s_raw[retained - 1]):
        raise RuntimeError(
            "registered effective-kernel threshold does not separate retained/kernel singular values"
        )
    return {
        "w": w,
        "xmu": xmu,
        "xsd": xsd,
        "u_raw": u_raw.astype(np.float32),
        "s_raw": s_raw,
        "u_standard": u_standard.astype(np.float32),
        "s_standard": s_standard,
        "raw_cutoffs": raw_cutoffs,
        "standard_cutoffs": standard_cutoffs,
        "selected_lambda": float(payload["selected_lambda"]),
        "raw_retained_squared_mass_99": float(np.sum(s_raw[:retained] ** 2) / np.sum(s_raw**2)),
    }


def _point_system(
    design_info: dict[str, Any],
    outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    singular_values: np.ndarray,
    cutoffs: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Fit the four point-estimate nested systems in one coordinate system."""

    design = design_info["matrix"]
    results: dict[str, dict[str, Any]] = {}
    deltas: dict[str, np.ndarray] = {}
    for axis in AXES:
        fit = nested_improvement(design, outcomes, train, test, design_info["target_columns"][axis])
        deltas[axis] = fit["delta"]
        results[axis] = summarize_delta(fit["delta"], fit["reduced_sse"], singular_values, cutoffs)
        results[axis]["design_rank_full"] = int(np.linalg.matrix_rank(design[train]))
        keep = fit["reduced_columns"]
        results[axis]["design_rank_reduced"] = int(np.linalg.matrix_rank(design[train][:, keep]))
        results[axis]["condition_number_full"] = float(np.linalg.cond(design[train]))
        residualized_target = design[train][:, design_info["target_columns"][axis]]
        nuisance_coef = np.linalg.lstsq(design[train][:, keep], residualized_target, rcond=None)[0]
        residualized_target -= design[train][:, keep] @ nuisance_coef
        results[axis]["residualized_target_variance"] = float(np.var(residualized_target))
    return results, deltas


def _bootstrap(
    cfg: ValidationConfig,
    design_info: dict[str, Any],
    outcome_systems: dict[str, np.ndarray],
    singular_systems: dict[str, tuple[np.ndarray, dict[str, int]]],
    train: np.ndarray,
    test: np.ndarray,
    corpus: np.ndarray,
    checkpoint_dir: Path,
    draws: int,
    config_sha256: str,
) -> dict[str, Any]:
    """Run joint stratified Bayesian-bootstrap complete refits with checkpoints."""

    design = design_info["matrix"]
    combined_names = list(outcome_systems)
    widths = [outcome_systems[name].shape[1] for name in combined_names]
    combined = np.column_stack([outcome_systems[name] for name in combined_names])
    design_train = design[train]
    design_test = design[test]
    combined_train = combined[train]
    combined_test = combined[test]
    corpus_train = corpus[train]
    corpus_test = corpus[test]
    offsets = np.cumsum([0] + widths)
    storage: dict[str, dict[str, list[np.ndarray]]] = {
        system: {axis: [] for axis in AXES} for system in combined_names
    }
    checkpoint = checkpoint_dir / "bootstrap.npz"
    completed = 0
    if checkpoint.exists():
        loaded = np.load(checkpoint)
        if str(loaded["config_sha256"][0]) != config_sha256:
            raise RuntimeError("bootstrap checkpoint config mismatch")
        completed = int(loaded["completed"][0])
        if completed < 0 or completed > draws:
            raise RuntimeError(f"invalid bootstrap checkpoint count {completed}")
        for system in combined_names:
            for axis in AXES:
                key = f"{system}__{axis}"
                if key not in loaded or loaded[key].shape[0] != completed:
                    raise RuntimeError(f"invalid bootstrap checkpoint array {key}")
                storage[system][axis].append(np.asarray(loaded[key]))

    def fit_block(job: tuple[int, int, np.ndarray, np.ndarray]):
        lower, batch, train_weights, test_weights = job
        block = batched_weighted_nested_sse(
            design_train,
            combined_train,
            design_test,
            combined_test,
            train_weights,
            test_weights,
            design_info["target_columns"],
        )
        return lower, batch, block

    started = time.monotonic()
    wave_width = REFIT_BLOCK * BOOTSTRAP_WORKERS
    with threadpool_limits(limits=BOOTSTRAP_BLAS_THREADS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as executor:
            for wave_start in range(completed, draws, wave_width):
                jobs = []
                for lower in range(wave_start, min(wave_start + wave_width, draws), REFIT_BLOCK):
                    batch = min(REFIT_BLOCK, draws - lower)
                    rng = np.random.default_rng(cfg.seed + 101 + lower)
                    jobs.append(
                        (
                            lower,
                            batch,
                            stratified_exponential_weights(rng, corpus_train, batch),
                            stratified_exponential_weights(rng, corpus_test, batch),
                        )
                    )
                for lower, batch, block in executor.map(fit_block, jobs):
                    if lower != completed:
                        raise RuntimeError(f"bootstrap block order drift: {lower} != {completed}")
                    for system_index, system in enumerate(combined_names):
                        lo, hi = offsets[system_index], offsets[system_index + 1]
                        singular_values, cutoffs = singular_systems[system]
                        for axis in AXES:
                            delta = block[axis]["delta"][:, lo:hi]
                            reduced = block[axis]["reduced_sse"][:, lo:hi]
                            denominator = delta.sum(axis=1)
                            mapped = delta @ (singular_values**2)
                            retained99 = cutoffs["99"]
                            retained_delta = delta[:, :retained99].sum(axis=1)
                            retained_reduced = reduced[:, :retained99].sum(axis=1)
                            kernel_delta = delta[:, retained99:].sum(axis=1)
                            kernel_reduced = reduced[:, retained99:].sum(axis=1)
                            metrics = [
                                denominator,
                                mapped,
                                denominator / reduced.sum(axis=1),
                                retained_delta,
                                retained_delta / retained_reduced,
                                kernel_delta,
                                kernel_delta / kernel_reduced,
                                mapped / denominator,
                            ]
                            for retained in cutoffs.values():
                                metrics.append(delta[:, retained:].sum(axis=1) / denominator)
                            storage[system][axis].append(np.column_stack(metrics))
                    completed += batch
                    if completed % (13 * REFIT_BLOCK) == 0 or completed == draws:
                        arrays = {
                            f"{system}__{axis}": np.concatenate(chunks, axis=0)
                            for system, axes in storage.items()
                            for axis, chunks in axes.items()
                        }
                        tmp = checkpoint_dir / "bootstrap.tmp.npz"
                        np.savez(
                            tmp,
                            completed=np.asarray([completed]),
                            config_sha256=np.asarray([config_sha256]),
                            **arrays,
                        )
                        os.replace(tmp, checkpoint)
                    print(
                        f"[bootstrap] unit {completed}/{draws} joint-refit "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    metric_names = {
        system: [
            "delta_ss",
            "mapped_delta_ss",
            "incremental_r2",
            "retained_99_delta_ss",
            "retained_99_incremental_r2",
            "kernel_99_delta_ss",
            "kernel_99_incremental_r2",
            "tau",
        ]
        + [f"kappa_{name}" for name in singular_systems[system][1]]
        for system in combined_names
    }
    return {
        system: {
            axis: dict(zip(metric_names[system], np.concatenate(chunks, axis=0).T))
            for axis, chunks in axes.items()
        }
        for system, axes in storage.items()
    }


def _association_null(
    cfg: ValidationConfig,
    design_info: dict[str, Any],
    outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    blocks: np.ndarray,
    retained: int,
    checkpoint_dir: Path,
    draws: int,
    config_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the four blocked whole-outcome-residual association nulls."""

    checkpoint = checkpoint_dir / "association_null.npz"
    matrices = {
        name: np.full((draws, len(AXES)), np.nan, dtype=np.float64)
        for name in (
            "full",
            "retained_99_delta_ss",
            "kernel_99_delta_ss",
            "full_incremental_r2",
            "retained_99_incremental_r2",
            "kernel_99_incremental_r2",
        )
    }
    completed_by_axis = np.zeros(len(AXES), dtype=np.int64)
    if checkpoint.exists():
        loaded = np.load(checkpoint)
        if str(loaded["config_sha256"][0]) != config_sha256:
            raise RuntimeError("association-null checkpoint config mismatch")
        completed_by_axis = np.asarray(loaded["completed_by_axis"], dtype=np.int64)
        if completed_by_axis.shape != (len(AXES),):
            raise RuntimeError("association-null checkpoint shape mismatch")
        for name, matrix in matrices.items():
            key = f"draws_{name}"
            if key not in loaded or loaded[key].shape != matrix.shape:
                raise RuntimeError(f"association-null checkpoint shape mismatch: {key}")
            matrix[:] = np.asarray(loaded[key], dtype=np.float64)
    design_train = design_info["matrix"][train]
    design_test = design_info["matrix"][test]
    outcomes_train = outcomes[train]
    outcomes_test = outcomes[test]
    blocks_train = blocks[train]
    blocks_test = blocks[test]
    with threadpool_limits(limits=PREPARE_BLAS_THREADS):
        prepared = {
            axis: prepare_freedman_lane(
                design_train,
                outcomes_train,
                design_test,
                outcomes_test,
                design_info["target_columns"][axis],
            )
            for axis in AXES
        }
    diagnostics: dict[str, Any] = {
        axis: {
            "residual_variance": residual_variance_diagnostics(
                prepared[axis], blocks_train, blocks_test
            )
        }
        for axis in AXES
    }
    for axis_index, axis in enumerate(AXES):
        completed = int(completed_by_axis[axis_index])
        if (
            completed < 0
            or completed > draws
            or not all(
                np.isfinite(matrix[:completed, axis_index]).all() for matrix in matrices.values()
            )
        ):
            raise RuntimeError(f"invalid association-null checkpoint count for {axis}: {completed}")
    if len(set(completed_by_axis.tolist())) != 1:
        raise RuntimeError(
            f"association-null checkpoint axes are not wave-aligned: {completed_by_axis.tolist()}"
        )

    def fit_axis_block(job: tuple[int, str, int, int]):
        axis_index, axis, lower, batch = job
        rng = np.random.default_rng(cfg.seed + 1000 + axis_index * 100_000 + lower)
        block = batched_freedman_lane_outcome(
            rng,
            design_train,
            outcomes_train,
            design_test,
            outcomes_test,
            design_info["target_columns"][axis],
            blocks_train,
            blocks_test,
            batch,
            prepared=prepared[axis],
            retained=retained,
        )
        return axis_index, axis, lower, batch, block

    started = time.monotonic()
    completed = int(completed_by_axis[0])
    with threadpool_limits(limits=ASSOCIATION_BLAS_THREADS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=ASSOCIATION_WORKERS) as executor:
            for lower in range(completed, draws, REFIT_BLOCK):
                batch = min(REFIT_BLOCK, draws - lower)
                jobs = [(axis_index, axis, lower, batch) for axis_index, axis in enumerate(AXES)]
                for axis_index, axis, block_lower, block_batch, block in executor.map(
                    fit_axis_block, jobs
                ):
                    if block_lower != lower or block_batch != batch:
                        raise RuntimeError("association-null block order drift")
                    for subspace_index, matrix in enumerate(matrices.values()):
                        matrix[lower : lower + batch, axis_index] = block[:, subspace_index]
                    completed_by_axis[axis_index] = lower + batch
                    print(
                        f"[association-null] unit {int(completed_by_axis.sum())}/"
                        f"{len(AXES) * draws} {axis} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                completed = lower + batch
                if completed % (13 * REFIT_BLOCK) == 0 or completed == draws:
                    temporary = checkpoint_dir / "association_null.tmp.npz"
                    np.savez(
                        temporary,
                        completed_by_axis=completed_by_axis,
                        config_sha256=np.asarray([config_sha256]),
                        **{f"draws_{name}": matrix for name, matrix in matrices.items()},
                    )
                    os.replace(temporary, checkpoint)
    for axis_index, axis in enumerate(AXES):
        diagnostics[axis]["q95"] = {
            name: float(np.quantile(matrix[:, axis_index], 0.95))
            for name, matrix in matrices.items()
        }
    return matrices, diagnostics


def _directional_null(
    cfg: ValidationConfig,
    deltas: dict[str, np.ndarray],
    checkpoint_dir: Path,
    draws: int,
    config_sha256: str,
) -> np.ndarray:
    """Run and checkpoint the matched-rank selection-symmetric directional null."""

    checkpoint = checkpoint_dir / "directional_null.npz"
    chunks: list[np.ndarray] = []
    completed = 0
    if checkpoint.exists():
        loaded = np.load(checkpoint)
        if str(loaded["config_sha256"][0]) != config_sha256:
            raise RuntimeError("directional-null checkpoint config mismatch")
        prior = np.asarray(loaded["draws"], dtype=np.float64)
        if prior.ndim != 2 or prior.shape[1] != 5 or prior.shape[0] > draws:
            raise RuntimeError("directional-null checkpoint shape mismatch")
        chunks.append(prior)
        completed = prior.shape[0]
    started = time.monotonic()
    for lower in range(completed, draws, DIRECTION_BLOCK):
        batch = min(DIRECTION_BLOCK, draws - lower)
        rng = np.random.default_rng(cfg.seed + 2000 + lower)
        chunks.append(batched_matched_rank_null(rng, deltas, KERNEL_EXPECTED, batch))
        completed += batch
        if completed % 1024 == 0 or completed == draws:
            temporary = checkpoint_dir / "directional_null.tmp.npz"
            np.savez(
                temporary,
                config_sha256=np.asarray([config_sha256]),
                draws=np.concatenate(chunks, axis=0),
            )
            os.replace(temporary, checkpoint)
        print(
            f"[directional-null] unit {completed}/{draws} matched-rank elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    return np.concatenate(chunks, axis=0)


def _natural_kernel_share(outcomes: np.ndarray, test: np.ndarray, retained: int) -> float:
    """Return descriptive test-set variance share in the effective kernel."""

    centered = outcomes[test] - outcomes[test].mean(axis=0, keepdims=True)
    energy = np.einsum("nd,nd->d", centered, centered, optimize=True)
    return float(energy[retained:].sum() / energy.sum())


def _category_counts(rows: list[dict[str, Any]], design_info: dict[str, Any]) -> dict[str, Any]:
    """Return all label counts and pairwise contingency tables."""

    counts = {
        axis: dict(sorted(Counter(design_info["values"][axis].tolist()).items())) for axis in AXES
    }
    cross_tabs: dict[str, Any] = {}
    for left_index, left in enumerate(AXES):
        for right in AXES[left_index + 1 :]:
            table: dict[str, Counter[str]] = defaultdict(Counter)
            for left_value, right_value in zip(
                design_info["values"][left], design_info["values"][right]
            ):
                table[str(left_value)][str(right_value)] += 1
            cross_tabs[f"{left}__{right}"] = {
                key: dict(sorted(value.items())) for key, value in sorted(table.items())
            }
    return {"counts": counts, "cross_tabs": cross_tabs}


def _point_control(
    rows: list[dict[str, Any]],
    outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    singular_values: np.ndarray,
    cutoffs: dict[str, int],
    axes: tuple[str, ...] = AXES,
) -> dict[str, Any]:
    """Run a fixed point-estimate control with a rebuilt design."""

    info = build_design(rows, axes=axes)
    result, _deltas = _point_system(info, outcomes, train, test, singular_values, cutoffs)
    return result


def _unadjusted_controls(
    rows: list[dict[str, Any]],
    outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    singular_values: np.ndarray,
    cutoffs: dict[str, int],
) -> dict[str, Any]:
    """Fit category-only centroid controls with intercept-only nuisance."""

    output = {}
    for axis in AXES:
        values = np.asarray([row[LABEL_KEY[axis]] for row in rows], dtype=str)
        if axis == "language":
            values = collapse_rare_language(values)
        encoded, _names, levels = treatment_columns(values, axis)
        design = np.column_stack([np.ones(len(rows)), encoded])
        fit = nested_improvement(design, outcomes, train, test, list(range(1, design.shape[1])))
        output[axis] = {
            **summarize_delta(fit["delta"], fit["reduced_sse"], singular_values, cutoffs),
            "levels": levels,
        }
    return output


def _corpus_transfer_controls(
    rows: list[dict[str, Any]],
    outcomes: np.ndarray,
    singular_values: np.ndarray,
    cutoffs: dict[str, int],
) -> dict[str, Any]:
    """Fit each corpus and score the other with a fixed shared-level collapse."""

    corpora = np.asarray([row["corpus"] for row in rows], dtype=str)
    names = sorted(set(corpora.tolist()))
    if len(names) != 2:
        raise RuntimeError(f"expected exactly two corpora, got {names}")
    output: dict[str, Any] = {}
    for source, target in ((names[0], names[1]), (names[1], names[0])):
        source_mask = corpora == source
        target_mask = corpora == target
        transformed = [dict(row) for row in rows]
        level_record: dict[str, Any] = {}
        for axis in AXES:
            key = LABEL_KEY[axis]
            source_counts = Counter(row[key] for row, keep in zip(rows, source_mask) if keep)
            target_counts = Counter(row[key] for row, keep in zip(rows, target_mask) if keep)
            shared = sorted(
                level
                for level in set(source_counts) | set(target_counts)
                if source_counts[level] >= 20 and target_counts[level] >= 20
            )
            level_record[axis] = {
                "shared": shared,
                "source_counts": dict(source_counts),
                "target_counts": dict(target_counts),
            }
            for row in transformed:
                if row[key] not in shared:
                    row[key] = "other"
        info = build_design(transformed, include_corpus=False)
        if any(
            np.linalg.matrix_rank(info["matrix"][mask]) != info["matrix"].shape[1]
            for mask in (source_mask, target_mask)
        ):
            output[f"{source}_to_{target}"] = {
                "status": "not_applicable_rank_deficient",
                "levels": level_record,
            }
            continue
        result, _deltas = _point_system(
            info, outcomes, source_mask, target_mask, singular_values, cutoffs
        )
        output[f"{source}_to_{target}"] = {
            "status": "computed",
            "levels": level_record,
            "axes": result,
        }
    return output


def _bootstrap_summaries(
    point_raw: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
    association: dict[str, np.ndarray],
    association_diag: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach uncertainty, association-null gates, and primary contrasts."""

    axes: dict[str, Any] = {}
    for index, axis in enumerate(AXES):
        draws = bootstrap["raw"][axis]
        observed = point_raw[axis]
        null = association["full"][:, index]
        p_value = float((1 + np.sum(null >= observed["delta_ss"])) / (len(null) + 1))
        r2_null = association["full_incremental_r2"][:, index]
        recoverability = (
            observed["delta_ss"] > 0
            and percentile_interval(draws["delta_ss"])[0] > 0
            and observed["delta_ss"] > association_diag[axis]["q95"]["full"]
            and p_value <= 0.05
        )
        nonpositive_denominator = float(np.mean(draws["delta_ss"] <= 0))
        mapped_pass = (
            observed["mapped_delta_ss"] > 0
            and percentile_interval(draws["mapped_delta_ss"])[0] > 0
            and nonpositive_denominator <= 0.025
        )
        axes[axis] = {
            **observed,
            "bootstrap_ci": {name: percentile_interval(value) for name, value in draws.items()},
            "association_null_q95": association_diag[axis]["q95"]["full"],
            "association_null_p": p_value,
            "association_null_incremental_r2_q95": association_diag[axis]["q95"][
                "full_incremental_r2"
            ],
            "association_null_incremental_r2_p": float(
                (1 + np.sum(r2_null >= observed["incremental_r2"])) / (len(r2_null) + 1)
            ),
            "subspace_predictability": {
                name: {
                    "delta_ss": observed[f"{name}_delta_ss"],
                    "incremental_r2": observed[f"{name}_incremental_r2"],
                    "incremental_r2_ci": percentile_interval(draws[f"{name}_incremental_r2"]),
                    "association_null_q95": association_diag[axis]["q95"][f"{name}_incremental_r2"],
                    "association_null_p": float(
                        (
                            1
                            + np.sum(
                                association[f"{name}_incremental_r2"][:, index]
                                >= observed[f"{name}_incremental_r2"]
                            )
                        )
                        / (association[f"{name}_incremental_r2"].shape[0] + 1)
                    ),
                }
                for name in ("retained_99", "kernel_99")
            },
            "nonpositive_denominator_fraction": nonpositive_denominator,
            "recoverability_pass": bool(recoverability),
            "mapped_gain_pass": bool(mapped_pass),
        }
    response_axes = ("language", "format", "safety")
    kappa_draw = bootstrap["raw"]["topic"]["kappa_99"] - np.mean(
        [bootstrap["raw"][axis]["kappa_99"] for axis in response_axes], axis=0
    )
    gain_draw = (
        np.mean([bootstrap["raw"][axis]["tau"] for axis in response_axes], axis=0)
        - (bootstrap["raw"]["topic"]["tau"])
    )
    contrasts = {
        "delta_kappa": axes["topic"]["kappa_99"]
        - float(np.mean([axes[axis]["kappa_99"] for axis in response_axes])),
        "delta_gain": float(np.mean([axes[axis]["tau"] for axis in response_axes]))
        - axes["topic"]["tau"],
        "delta_kappa_ci": percentile_interval(kappa_draw),
        "delta_gain_ci": percentile_interval(gain_draw),
        "individual_kappa": {
            axis: axes["topic"]["kappa_99"] - axes[axis]["kappa_99"] for axis in response_axes
        },
        "individual_gain": {
            axis: axes[axis]["tau"] - axes["topic"]["tau"] for axis in response_axes
        },
    }
    return axes, contrasts


def _standardized_robustness(
    point: dict[str, dict[str, Any]], bootstrap: dict[str, Any]
) -> dict[str, Any]:
    """Gate standardized ratios before using their contrast signs as robustness evidence."""

    axes: dict[str, Any] = {}
    for axis in AXES:
        draws = bootstrap[axis]
        delta_ci = percentile_interval(draws["delta_ss"])
        mapped_ci = percentile_interval(draws["mapped_delta_ss"])
        nonpositive = float(np.mean(draws["delta_ss"] <= 0))
        denominator_pass = bool(
            point[axis]["delta_ss"] > 0 and delta_ci[0] > 0 and nonpositive <= 0.025
        )
        mapped_pass = bool(point[axis]["mapped_delta_ss"] > 0 and mapped_ci[0] > 0)
        axes[axis] = {
            **point[axis],
            "bootstrap_ci": {name: percentile_interval(values) for name, values in draws.items()},
            "nonpositive_denominator_fraction": nonpositive,
            "denominator_pass": denominator_pass,
            "mapped_gain_pass": mapped_pass,
            "interpretability_pass": denominator_pass and mapped_pass,
        }
    response_axes = ("language", "format", "safety")
    return {
        "delta_kappa": axes["topic"]["kappa_99"]
        - float(np.mean([axes[axis]["kappa_99"] for axis in response_axes])),
        "delta_gain": float(np.mean([axes[axis]["tau"] for axis in response_axes]))
        - axes["topic"]["tau"],
        "axes": axes,
        "interpretability_pass": bool(
            all(value["interpretability_pass"] for value in axes.values())
        ),
    }


def _directional_statistics(point_axes: dict[str, Any], directional: np.ndarray) -> dict[str, Any]:
    """Compute registered directional-null p values and four-axis BH FDR."""

    observed = np.asarray(
        [point_axes[axis]["kappa_99"] for axis in AXES]
        + [
            point_axes["topic"]["kappa_99"]
            - np.mean([point_axes[axis]["kappa_99"] for axis in AXES[1:]])
        ]
    )
    p_values = {}
    for index, axis in enumerate(AXES):
        if axis == "topic":
            extreme = directional[:, index] >= observed[index]
        else:
            extreme = directional[:, index] <= observed[index]
        p_values[axis] = float((1 + extreme.sum()) / (len(directional) + 1))
    contrast_p = float((1 + np.sum(directional[:, 4] >= observed[4])) / (len(directional) + 1))
    return {
        "observed": observed.tolist(),
        "p_values": p_values,
        "contrast_p": contrast_p,
        "bh_fdr": benjamini_hochberg(p_values),
        "null_quantiles": {
            "q025": np.quantile(directional, 0.025, axis=0).tolist(),
            "q50": np.quantile(directional, 0.50, axis=0).tolist(),
            "q975": np.quantile(directional, 0.975, axis=0).tolist(),
        },
    }


def _pilot_battery(
    cfg: ValidationConfig,
    design_info: dict[str, Any],
    bootstrap_outcomes: np.ndarray,
    association_outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    corpus: np.ndarray,
    blocks: np.ndarray,
    deltas: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Time exact production helpers on one registered block of each class."""

    design = design_info["matrix"]
    design_train = design[train]
    design_test = design[test]
    bootstrap_train = bootstrap_outcomes[train]
    bootstrap_test = bootstrap_outcomes[test]
    association_train = association_outcomes[train]
    association_test = association_outcomes[test]
    corpus_train = corpus[train]
    corpus_test = corpus[test]
    blocks_train = blocks[train]
    blocks_test = blocks[test]
    timings: dict[str, float] = {}
    bootstrap_jobs = []
    for lower in range(0, REFIT_BLOCK * BOOTSTRAP_WORKERS, REFIT_BLOCK):
        rng = np.random.default_rng(cfg.seed + 9000 + lower)
        bootstrap_jobs.append(
            (
                stratified_exponential_weights(rng, corpus_train, REFIT_BLOCK),
                stratified_exponential_weights(rng, corpus_test, REFIT_BLOCK),
            )
        )

    def bootstrap_pilot(job: tuple[np.ndarray, np.ndarray]):
        return batched_weighted_nested_sse(
            design_train,
            bootstrap_train,
            design_test,
            bootstrap_test,
            job[0],
            job[1],
            design_info["target_columns"],
        )

    prepare_start = time.monotonic()
    with threadpool_limits(limits=PREPARE_BLAS_THREADS):
        prepared = {
            axis: prepare_freedman_lane(
                design_train,
                association_train,
                design_test,
                association_test,
                design_info["target_columns"][axis],
            )
            for axis in AXES
        }
    timings["permutation_prepare_seconds"] = time.monotonic() - prepare_start

    def permutation_pilot(item: tuple[int, str]):
        index, axis = item
        item_started = time.monotonic()
        batched_freedman_lane_outcome(
            np.random.default_rng(cfg.seed + 9100 + index),
            design_train,
            association_train,
            design_test,
            association_test,
            design_info["target_columns"][axis],
            blocks_train,
            blocks_test,
            REFIT_BLOCK,
            prepared=prepared[axis],
            retained=RETAINED99_EXPECTED,
        )
        return axis, time.monotonic() - item_started

    with threadpool_limits(limits=BOOTSTRAP_BLAS_THREADS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as executor:
            start = time.monotonic()
            list(executor.map(bootstrap_pilot, bootstrap_jobs))
            timings["bootstrap_parallel_wave_seconds"] = time.monotonic() - start
    with threadpool_limits(limits=ASSOCIATION_BLAS_THREADS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=ASSOCIATION_WORKERS) as executor:
            start = time.monotonic()
            permutation_times = list(executor.map(permutation_pilot, enumerate(AXES)))
            timings["permutation_parallel_wave_seconds"] = time.monotonic() - start
    for axis, elapsed in permutation_times:
        timings[f"permutation_{axis}_block_seconds"] = elapsed
    start = time.monotonic()
    batched_matched_rank_null(
        np.random.default_rng(cfg.seed + 9200), deltas, KERNEL_EXPECTED, DIRECTION_BLOCK
    )
    timings["directional_block_seconds"] = time.monotonic() - start
    n_boot_blocks = math.ceil(cfg.bootstrap_draws / REFIT_BLOCK)
    n_perm_blocks = math.ceil(cfg.permutation_draws / REFIT_BLOCK)
    n_direction_blocks = math.ceil(cfg.directional_draws / DIRECTION_BLOCK)
    extrapolated = (
        timings["bootstrap_parallel_wave_seconds"] * math.ceil(n_boot_blocks / BOOTSTRAP_WORKERS)
        + timings["permutation_prepare_seconds"]
        + timings["permutation_parallel_wave_seconds"] * n_perm_blocks
        + timings["directional_block_seconds"] * n_direction_blocks
    )
    peak = rss_gb()
    return {
        "timings": timings,
        "block_multiplier": n_boot_blocks + len(AXES) * n_perm_blocks + n_direction_blocks,
        "two_x_extrapolated_seconds": 2.0 * extrapolated,
        "peak_rss_gb": peak,
        "wall_pass": 2.0 * extrapolated <= cfg.battery_fence_seconds,
        "rss_pass": peak <= cfg.rss_fence_gb,
    }


def analyze_compact(cfg: ValidationConfig, repo_root: Path, out_root: Path) -> dict[str, Any]:
    """Run the complete registered category-validation statistical battery."""

    started = time.monotonic()
    compact = out_root / "compact"
    checkpoint_dir = out_root / "analysis_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint, extraction_manifest = verify_local_compact(out_root)
    compact_destination = verify_compact_upload(out_root) if cfg.production else None
    if extraction_manifest["matrix"]["sha256"] != sha256_file(compact / "activations_L19.npy"):
        raise RuntimeError("compact activation SHA mismatch")
    rows = json.loads((compact / "rows.json").read_text())
    if len(rows) != N_LABELED:
        raise RuntimeError(f"compact rows={len(rows)} != {N_LABELED}")
    x = np.load(compact / "activations_L19.npy", mmap_mode="r")
    if x.shape != (N_LABELED, D_MODEL) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise RuntimeError("compact activation shape/dtype/finiteness gate failed")
    operator = _load_operator(out_root / "frozen/ridge_L19.pt")
    design_info = build_design(rows)
    partition = np.asarray([row["partition"] for row in rows], dtype=str)
    train = partition == "train"
    test = partition == "test"
    if train.sum() + test.sum() != N_LABELED or min(train.sum(), test.sum()) == 0:
        raise RuntimeError("invalid train/test partition")
    for name, mask in (("train", train), ("test", test)):
        rank = np.linalg.matrix_rank(design_info["matrix"][mask])
        if rank != design_info["matrix"].shape[1]:
            raise RuntimeError(
                f"{name} full design rank={rank} != {design_info['matrix'].shape[1]}"
            )
        for axis in AXES:
            keep = np.asarray(
                [
                    index
                    for index in range(design_info["matrix"].shape[1])
                    if index not in design_info["target_columns"][axis]
                ]
            )
            reduced_rank = np.linalg.matrix_rank(design_info["matrix"][mask][:, keep])
            if reduced_rank != len(keep):
                raise RuntimeError(
                    f"{name} {axis} reduced design rank={reduced_rank} != {len(keep)}"
                )
    for axis in AXES:
        values = design_info["values"][axis]
        for value in sorted(set(values.tolist())):
            if not np.any(values[train] == value) or not np.any(values[test] == value):
                raise RuntimeError(f"{axis}={value}: absent from a partition")

    LOGGER.info("[analysis] projecting context states into registered singular coordinates")
    raw = np.empty((N_LABELED, D_MODEL), dtype=np.float32)
    standard = np.empty_like(raw)
    block_rows = 512
    for lower in range(0, N_LABELED, block_rows):
        upper = min(lower + block_rows, N_LABELED)
        x_block = np.asarray(x[lower:upper], dtype=np.float32)
        raw[lower:upper] = x_block @ operator["u_raw"]
        standardized_block = (
            (x_block.astype(np.float64) - operator["xmu"]) / operator["xsd"]
        ).astype(np.float32)
        standard[lower:upper] = standardized_block @ operator["u_standard"]
        print(
            f"[projection] unit {upper}/{N_LABELED} singular-coordinates "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    point_raw, deltas_raw = _point_system(
        design_info, raw, train, test, operator["s_raw"], operator["raw_cutoffs"]
    )
    write_json_atomic(
        checkpoint_dir / "point_raw.json",
        {axis: {key: value for key, value in result.items()} for axis, result in point_raw.items()},
    )
    point_standard, _deltas_standard = _point_system(
        design_info,
        standard,
        train,
        test,
        operator["s_standard"],
        operator["standard_cutoffs"],
    )
    write_json_atomic(checkpoint_dir / "point_standardized.json", point_standard)

    corpus = np.asarray([row["corpus"] for row in rows], dtype=str)
    depth = np.asarray([row["depth"] for row in rows], dtype=np.int64)
    prompt_chars = np.asarray([row["prompt_chars"] for row in rows], dtype=np.int64)
    blocks, block_diag_all = exchangeability_blocks(corpus, depth, prompt_chars)
    block_diag_train = summarize_blocks(blocks[train])
    block_diag_test = summarize_blocks(blocks[test])
    # Reuse the global frozen quartile cut points and report their partition block sizes.
    block_calibration_ok = (
        block_diag_train["small_block_fraction"] <= 0.05
        and block_diag_test["small_block_fraction"] <= 0.05
    )
    pilot = _pilot_battery(
        cfg,
        design_info,
        np.column_stack([raw, standard]),
        raw,
        train,
        test,
        corpus,
        blocks,
        deltas_raw,
    )
    write_json_atomic(checkpoint_dir / "battery_pilot.json", pilot)
    if not pilot["wall_pass"] or not pilot["rss_pass"]:
        raise RuntimeError(f"production battery pilot failed: {pilot}")

    bootstrap = _bootstrap(
        cfg,
        design_info,
        {"raw": raw, "standardized": standard},
        {
            "raw": (operator["s_raw"], operator["raw_cutoffs"]),
            "standardized": (operator["s_standard"], operator["standard_cutoffs"]),
        },
        train,
        test,
        corpus,
        checkpoint_dir,
        cfg.bootstrap_draws,
        fingerprint["config_sha256"],
    )
    association, association_diag = _association_null(
        cfg,
        design_info,
        raw,
        train,
        test,
        blocks,
        operator["raw_cutoffs"]["99"],
        checkpoint_dir,
        cfg.permutation_draws,
        fingerprint["config_sha256"],
    )
    residual_variance_ok = all(
        association_diag[axis]["residual_variance"]["gate_pass"] for axis in AXES
    )
    residual_null_ok = bool(block_calibration_ok and residual_variance_ok)
    directional = _directional_null(
        cfg,
        deltas_raw,
        checkpoint_dir,
        cfg.directional_draws,
        fingerprint["config_sha256"],
    )
    axes, contrasts = _bootstrap_summaries(point_raw, bootstrap, association, association_diag)
    standardized_contrasts = _standardized_robustness(point_standard, bootstrap["standardized"])
    direction_stats = _directional_statistics(axes, directional)
    natural_share = _natural_kernel_share(raw, test, operator["raw_cutoffs"]["99"])

    # Secondary controls are point estimates only and cannot rescue the primary verdict.
    unadjusted = _unadjusted_controls(
        rows, raw, train, test, operator["s_raw"], operator["raw_cutoffs"]
    )
    corpus_transfer = _corpus_transfer_controls(
        rows, raw, operator["s_raw"], operator["raw_cutoffs"]
    )
    safety_keep = np.asarray(
        [row["request_refusal_adjacent"] != "borderline" for row in rows], dtype=bool
    )
    safety_rows = [dict(row) for row, keep in zip(rows, safety_keep) if keep]
    safety_result = _point_control(
        safety_rows,
        raw[safety_keep],
        train[safety_keep],
        test[safety_keep],
        operator["s_raw"],
        operator["raw_cutoffs"],
    )["safety"]
    refusal_rows = [dict(row) for row in rows]
    for row in refusal_rows:
        row["request_refusal_adjacent"] = row["answer_is_refusal"]
    answer_refusal = _point_control(
        refusal_rows, raw, train, test, operator["s_raw"], operator["raw_cutoffs"]
    )["safety"]
    verdict = evaluate_verdict(
        axes,
        contrasts,
        standardized_contrasts,
        provenance_ok=True,
        pilot_ok=pilot["wall_pass"] and pilot["rss_pass"],
        residual_null_ok=residual_null_ok,
    )

    result = {
        "schema": ANALYSIS_SCHEMA,
        "task": ISSUE,
        "layer": LAYER,
        "config_sha256": fingerprint["config_sha256"],
        "provenance": {
            "base_commit": BASE_COMMIT,
            "capture_revision": CAPTURE_REVISION,
            "capture_pilot_meta_sha256": fingerprint["pilot_meta_sha256"],
            "split_sha256": fingerprint["split_sha256"],
            "labels_sha256": fingerprint["labels_sha256"],
            "map_revision": MAP_REVISION,
            "map_sha256": MAP_SHA256,
            "kernel_record_sha256": fingerprint["kernel_record_sha256"],
            "compact_upload_revision": (
                compact_destination["revision"] if compact_destination else None
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "extraction_manifest_sha256": sha256_file(compact / "extraction_manifest.json"),
            "git_commit": git_output(repo_root, "rev-parse", "HEAD"),
            "git_dirty": bool(git_output(repo_root, "status", "--porcelain")),
            "frozen_input_manifest": fingerprint,
            "producer_compatibility": extraction_manifest["producer_compatibility"],
            "capture_matrix_sha256": extraction_manifest["matrix"]["sha256"],
            "compact_rows_sha256": extraction_manifest["rows_sha256"],
        },
        "coverage": {
            "selected_holdout": N_SELECTED_HOLDOUT,
            "realized_captured_holdout": N_HOLDOUT,
            "realized_labeled": N_LABELED,
            "train": int(train.sum()),
            "test": int(test.sum()),
            "missing_labels": N_HOLDOUT - N_LABELED,
            "capture_drops": N_SELECTED_HOLDOUT - N_HOLDOUT,
            "missingness_audit": json.loads((compact / "missingness_audit.json").read_text()),
        },
        "partition_ids": {
            "train": [int(row["ci"]) for row, selected in zip(rows, train) if selected],
            "test": [int(row["ci"]) for row, selected in zip(rows, test) if selected],
        },
        "design": {
            "shape": list(design_info["matrix"].shape),
            "columns": design_info["column_names"],
            "levels": design_info["levels"],
            "depth_encoding": design_info["depth_encoding"],
        },
        "category_inventory": _category_counts(rows, design_info),
        "operator": {
            "selected_lambda": operator["selected_lambda"],
            "raw_cutoffs": operator["raw_cutoffs"],
            "standardized_cutoffs": operator["standard_cutoffs"],
            "raw_kernel_dim_99": D_MODEL - operator["raw_cutoffs"]["99"],
            "natural_context_kernel_share": natural_share,
            "raw_retained_squared_mass_99": operator["raw_retained_squared_mass_99"],
        },
        "axes": axes,
        "primary_contrasts": contrasts,
        "standardized_robustness": standardized_contrasts,
        "directional_null": direction_stats,
        "association_null": {
            "draws": cfg.permutation_draws,
            "block_diagnostics_global": block_diag_all,
            "block_diagnostics_train": block_diag_train,
            "block_diagnostics_test": block_diag_test,
            "residual_variance_diagnostics": association_diag,
            "block_size_pass": block_calibration_ok,
            "residual_variance_pass": residual_variance_ok,
            "calibration_pass": residual_null_ok,
        },
        "bootstrap": {"draws": cfg.bootstrap_draws, "kind": "joint stratified Bayesian"},
        "controls": {
            "unadjusted": unadjusted,
            "corpus_transfer": corpus_transfer,
            "safety_excluding_borderline": safety_result,
            "answer_is_refusal_post_outcome": answer_refusal,
        },
        "label_kappas": extraction_manifest["label_kappas"],
        "battery_pilot": pilot,
        "verdict": verdict,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gb": rss_gb(),
        "scope_caveats": [
            "Topic is a 12-way coarse task/genre label, not exhaustive semantic content.",
            "Format is observed answer format, so this is answer-format-associated context variation.",
            "The analysis is correlational and does not identify the model's causal computation.",
            "Exact identity disjointness from every map-fitting source row is not established.",
        ],
    }
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_arrays = {
        "_config_sha256": np.asarray([fingerprint["config_sha256"]]),
        "association_null": association["full"],
        "association_null_retained_99_delta_ss": association["retained_99_delta_ss"],
        "association_null_kernel_99_delta_ss": association["kernel_99_delta_ss"],
        "association_null_full_incremental_r2": association["full_incremental_r2"],
        "association_null_retained_99_incremental_r2": association["retained_99_incremental_r2"],
        "association_null_kernel_99_incremental_r2": association["kernel_99_incremental_r2"],
        "directional_null": directional,
    }
    for system in bootstrap:
        for axis in bootstrap[system]:
            for metric, values in bootstrap[system][axis].items():
                raw_arrays[f"bootstrap__{system}__{axis}__{metric}"] = values
    resampling_path = analysis_dir / "resampling_arrays.npz"
    with atomic_replace(resampling_path, logger=LOGGER) as temporary:
        with temporary.open("wb") as handle:
            np.savez(handle, **raw_arrays)
            handle.flush()
            os.fsync(handle.fileno())
    result["resampling_arrays"] = {
        "path": resampling_path.name,
        "sha256": sha256_file(resampling_path),
        "size": resampling_path.stat().st_size,
        "config_sha256_key": "_config_sha256",
        "array_keys": sorted(raw_arrays),
    }
    write_json_atomic(analysis_dir / "category_kernel_validation_L19.json", result)
    return result


def interval_segment(interval: list[float]) -> tuple[float, float]:
    """Validate and return an exact plotted confidence-interval segment."""

    if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
        raise ValueError(f"invalid confidence interval: {interval}")
    return float(interval[0]), float(interval[1])


def render_figure(result: dict[str, Any], out_root: Path) -> dict[str, Any]:
    """Render the registered two-panel publication figure and provenance sidecar."""

    font = set_c2a_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.1))
    y = np.arange(len(AXES))[::-1]
    kernel_points = [result["axes"][axis]["kappa_99"] for axis in AXES]
    kernel_intervals = [result["axes"][axis]["bootstrap_ci"]["kappa_99"] for axis in AXES]
    gain_points = [result["axes"][axis]["tau"] for axis in AXES]
    gain_intervals = [result["axes"][axis]["bootstrap_ci"]["tau"] for axis in AXES]
    for index, axis in enumerate(AXES):
        kernel_low, kernel_high = interval_segment(kernel_intervals[index])
        gain_low, gain_high = interval_segment(gain_intervals[index])
        axes[0].hlines(y[index], kernel_low, kernel_high, color=PALETTE[axis], lw=1.8)
        axes[0].plot(
            kernel_points[index],
            y[index],
            marker=MARKERS[axis],
            color=PALETTE[axis],
            markersize=8,
            linestyle="none",
            label=DISPLAY[axis],
        )
        axes[1].hlines(y[index], gain_low, gain_high, color=PALETTE[axis], lw=1.8)
        axes[1].plot(
            gain_points[index],
            y[index],
            marker=MARKERS[axis],
            color=PALETTE[axis],
            markersize=8,
            linestyle="none",
        )
    axes[0].axvline(
        result["operator"]["natural_context_kernel_share"],
        color=MUTED,
        ls="--",
        lw=1.2,
        label="Natural context variance",
    )
    axes[0].set_title("A  Effective-kernel concentration", loc="left")
    axes[0].set_xlabel("Share of held-out category signal")
    axes[1].set_title("B  Through-map gain", loc="left")
    axes[1].set_xlabel("Singular-value²-weighted gain")
    axes[0].set_yticks(y, [DISPLAY[axis] for axis in AXES])
    axes[1].set_yticks(y, [])
    positive_log = all(
        point > 0 and interval[0] > 0 for point, interval in zip(gain_points, gain_intervals)
    )
    if positive_log:
        axes[1].set_xscale("log")
    else:
        axes[1].set_xscale("symlog", linthresh=1e-3)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SEAM)
        ax.spines["bottom"].set_color(SEAM)
        ax.tick_params(length=0, pad=8, colors=INK)
        ax.grid(axis="x", color=GRID, lw=1.0, alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=13, loc="best")
    fig.tight_layout(w_pad=3.2)
    figure_dir = out_root / "git_payload/figures/issue_2569"
    stem = figure_dir / "leg13_category_kernel_validation"
    paths = save_c2a_figure(
        fig,
        stem,
        title="Category-level context-map validation",
        subject="Effective-kernel concentration and through-map gain by frozen semantic category",
        creator="scripts/issue2569_category_validation.py",
        png_dpi=240,
    )
    plt.close(fig)
    metadata = {
        "style_version": "c2a-v1",
        "resolved_font": font,
        "xscale_panel_b": "log" if positive_log else "symlog",
        "visual_encodings": {
            axis: {"label": DISPLAY[axis], "color": PALETTE[axis], "marker": MARKERS[axis]}
            for axis in AXES
        },
        "plotted": {
            axis: {
                "kappa_99": kernel_points[index],
                "kappa_99_ci": kernel_intervals[index],
                "tau": gain_points[index],
                "tau_ci": gain_intervals[index],
            }
            for index, axis in enumerate(AXES)
        },
        "input_config_sha256": result["config_sha256"],
        "input_result_sha256": sha256_file(
            out_root / "analysis/category_kernel_validation_L19.json"
        ),
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
            for key, path in paths.items()
        },
        "git_commit": result["provenance"]["git_commit"],
        "git_dirty_at_analysis": result["provenance"]["git_dirty"],
    }
    write_json_atomic(figure_dir / "leg13_category_kernel_validation_meta.json", metadata)
    return metadata


def export_git_payload(result: dict[str, Any], out_root: Path) -> dict[str, Path]:
    """Write compact JSON/CSV/Markdown outputs intended for the Git branch."""

    result_dir = out_root / "git_payload/eval_results/issue_2569/weights/leg13"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "category_kernel_validation_L19.json"
    write_json_atomic(json_path, result)
    csv_path = result_dir / "category_kernel_validation_L19.csv"
    fields = [
        "axis",
        "label",
        "delta_ss",
        "incremental_r2",
        "retained_99_incremental_r2",
        "kernel_99_incremental_r2",
        "kappa_99",
        "kappa_ci_low",
        "kappa_ci_high",
        "tau",
        "tau_ci_low",
        "tau_ci_high",
        "association_null_p",
        "recoverability_pass",
        "mapped_gain_pass",
    ]
    with atomic_replace(csv_path, logger=LOGGER) as temporary:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for axis in AXES:
                value = result["axes"][axis]
                writer.writerow(
                    {
                        "axis": axis,
                        "label": DISPLAY[axis],
                        "delta_ss": value["delta_ss"],
                        "incremental_r2": value["incremental_r2"],
                        "retained_99_incremental_r2": value["retained_99_incremental_r2"],
                        "kernel_99_incremental_r2": value["kernel_99_incremental_r2"],
                        "kappa_99": value["kappa_99"],
                        "kappa_ci_low": value["bootstrap_ci"]["kappa_99"][0],
                        "kappa_ci_high": value["bootstrap_ci"]["kappa_99"][1],
                        "tau": value["tau"],
                        "tau_ci_low": value["bootstrap_ci"]["tau"][0],
                        "tau_ci_high": value["bootstrap_ci"]["tau"][1],
                        "association_null_p": value["association_null_p"],
                        "recoverability_pass": value["recoverability_pass"],
                        "mapped_gain_pass": value["mapped_gain_pass"],
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
    contrast = result["primary_contrasts"]
    lines = [
        "# Category-level validation of the layer-19 context→answer map",
        "",
        f"**Mechanical verdict:** {result['verdict']['verdict']} — {result['verdict']['reason']}.",
        "",
        "This analysis directly tests four frozen category axes in 9,925 already-labeled held-out",
        "real multi-turn LMSYS/WildChat contexts. It uses signed untouched-test improvement from",
        "nested nuisance-adjusted OLS models; no model generation or new labels were produced.",
        "",
        "| Category axis | Full ΔR² | R99 ΔR² | K99 ΔR² | Kernel share (95% CI) | Through-map gain (95% CI) | Association-null p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in AXES:
        value = result["axes"][axis]
        k_ci = value["bootstrap_ci"]["kappa_99"]
        t_ci = value["bootstrap_ci"]["tau"]
        lines.append(
            f"| {DISPLAY[axis]} | {value['incremental_r2']:.4f} | "
            f"{value['retained_99_incremental_r2']:.4f} | "
            f"{value['kernel_99_incremental_r2']:.4f} | "
            f"{value['kappa_99']:.3f} [{k_ci[0]:.3f}, {k_ci[1]:.3f}] | "
            f"{value['tau']:.4g} [{t_ci[0]:.4g}, {t_ci[1]:.4g}] | "
            f"{value['association_null_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            f"Primary topic-minus-response-regime kernel contrast: "
            f"{contrast['delta_kappa']:.3f} (95% CI "
            f"[{contrast['delta_kappa_ci'][0]:.3f}, {contrast['delta_kappa_ci'][1]:.3f}]).",
            "",
            f"Primary response-regime-minus-topic gain contrast: "
            f"{contrast['delta_gain']:.4g} (95% CI "
            f"[{contrast['delta_gain_ci'][0]:.4g}, {contrast['delta_gain_ci'][1]:.4g}]).",
            "",
            "Interpretation is restricted to coarse topic/task genre, prompt language,",
            "answer-format-associated context variation, and refusal adjacency. The result is",
            "correlational and does not establish the model's causal computational mechanism.",
            "",
        ]
    )
    markdown_path = result_dir / "category_kernel_validation_L19.md"
    write_text_atomic(markdown_path, "\n".join(lines))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _remote_entries(repo_id: str, prefix: str, revision: str = "main") -> dict[str, Any]:
    """Return a scoped remote file mapping, with absence distinct from transport failure."""

    api = HfApi()
    try:
        entries = hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: scoped upload verification is retried by its wrapper.
                api.list_repo_tree(
                    repo_id,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    recursive=True,
                    revision=revision,
                )
            ),
            what=f"list artifact prefix {repo_id}:{prefix}",
        )
    except EntryNotFoundError:
        return {}
    return {entry.path: entry for entry in entries}


def _remote_file_hashes(
    repo_id: str,
    prefix: str,
    local_root: Path,
    relpaths: list[str],
    *,
    require_all: bool = True,
    revision: str = "main",
) -> list[str]:
    """Verify present remote bytes and return missing relative paths."""

    indexed = _remote_entries(repo_id, prefix, revision=revision)
    missing = []
    with tempfile.TemporaryDirectory(prefix="issue2569-verify-") as temporary:
        for relpath in relpaths:
            local = local_root / relpath
            remote_path = f"{prefix.rstrip('/')}/{relpath}"
            if remote_path not in indexed:
                missing.append(relpath)
                continue
            entry = indexed[remote_path]
            if int(entry.size or -1) != local.stat().st_size:
                raise RuntimeError(f"uploaded size mismatch: {remote_path}")
            lfs_sha = getattr(getattr(entry, "lfs", None), "sha256", None)
            if lfs_sha:
                remote_sha = lfs_sha
            else:
                staged = hub.stage_hub_file(
                    repo_id,
                    remote_path,
                    Path(temporary) / Path(relpath).name,
                    repo_type="dataset",
                    revision=revision,
                    overwrite=True,
                )
                remote_sha = sha256_file(staged)
            if remote_sha != sha256_file(local):
                raise RuntimeError(f"uploaded SHA-256 mismatch: {remote_path}")
    if require_all and missing:
        raise RuntimeError(f"uploaded paths absent from {repo_id}:{prefix}: {missing}")
    return missing


def _resolved_repo_revision(repo_id: str) -> str:
    """Resolve the current dataset HEAD to an immutable commit SHA."""

    api = HfApi()
    revision = hub.retry_transient(
        # HUB_VERIFY_RETRY_EXEMPT: immutable HEAD resolution is retried by its wrapper.
        lambda: api.repo_info(repo_id, repo_type="dataset", revision="main").sha,
        what=f"resolve immutable dataset revision for {repo_id}",
    )
    if not revision or len(revision) != 40:
        raise RuntimeError(f"invalid immutable revision for {repo_id}: {revision!r}")
    return str(revision)


def upload_artifact_set(
    local_root: Path, relpaths: list[str], prefix: str, repo_id: str = DATA_REPO
) -> dict[str, Any]:
    """Bulk-upload one immutable artifact set and verify exact remote bytes."""

    missing = _remote_file_hashes(repo_id, prefix, local_root, relpaths, require_all=False)
    expected = [f"{prefix.rstrip('/')}/{path}" for path in relpaths]
    if not missing:
        effective_repo = repo_id
        resume = "verified-existing"
    else:
        url = hub._upload_folder_filtered(  # noqa: SLF001 - canonical issue-rig helper.
            local_root,
            repo_id,
            "dataset",
            prefix,
            allow_patterns=missing,
            expected_repo_paths=expected,
            private=repo_id == hub.DEFAULT_OVERFLOW_REPO,
        )
        if not url:
            raise RuntimeError(f"bulk upload returned no verified path for {prefix}")
        if url.startswith(f"{DATA_REPO}/"):
            effective_repo = DATA_REPO
        elif url.startswith(f"{hub.DEFAULT_OVERFLOW_REPO}/"):
            effective_repo = hub.DEFAULT_OVERFLOW_REPO
        else:
            raise RuntimeError(f"could not parse effective repository from upload result {url!r}")
        resume = "uploaded"
    revision = _resolved_repo_revision(effective_repo)
    _remote_file_hashes(effective_repo, prefix, local_root, relpaths, revision=revision)
    return {
        "url": f"{effective_repo}/{prefix}",
        "browser_url": f"https://huggingface.co/datasets/{effective_repo}/tree/{revision}/{prefix}",
        "repo_id": effective_repo,
        "prefix": prefix,
        "revision": revision,
        "files": expected,
        "resume": resume,
    }


COMPACT_RELPATHS = [
    "config_manifest.json",
    "activations_L19.npy",
    "rows.json",
    "missingness_audit.json",
    "shard_ledger.jsonl",
    "pilot_report.json",
    "extraction_manifest.json",
]
ANALYSIS_RELPATHS = ["category_kernel_validation_L19.json", "resampling_arrays.npz"]


def verify_local_compact(out_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate local compact provenance before a resumed upload or consumer phase."""

    compact = out_root / "compact"
    config = json.loads((compact / "config_manifest.json").read_text())
    manifest = json.loads((compact / "extraction_manifest.json").read_text())
    current_script_sha = sha256_file(Path(__file__).resolve())
    if config.get("script_sha256") != current_script_sha:
        raise RuntimeError("compact input was extracted by a different script SHA")
    if manifest.get("config") != config:
        raise RuntimeError("compact extraction manifest/config mismatch")
    expected_hashes = {
        "activations_L19.npy": manifest["matrix"]["sha256"],
        "rows.json": manifest["rows_sha256"],
        "shard_ledger.jsonl": manifest["ledger_sha256"],
    }
    for relpath, expected_sha in expected_hashes.items():
        path = compact / relpath
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"local compact artifact failed SHA gate: {relpath}")
    return config, manifest


def load_verified_analysis_result(out_root: Path) -> dict[str, Any]:
    """Validate a resumed analysis artifact against compact and executing-code identity."""

    config, _manifest = verify_local_compact(out_root)
    result_path = out_root / "analysis/category_kernel_validation_L19.json"
    result = json.loads(result_path.read_text())
    current_script_sha = sha256_file(Path(__file__).resolve())
    if result.get("schema") != ANALYSIS_SCHEMA:
        raise RuntimeError("analysis result schema mismatch")
    if result.get("config_sha256") != config["config_sha256"]:
        raise RuntimeError("analysis result config mismatch")
    if result.get("provenance", {}).get("script_sha256") != current_script_sha:
        raise RuntimeError("analysis result script SHA mismatch")
    resampling = result.get("resampling_arrays", {})
    resampling_path = out_root / "analysis" / str(resampling.get("path", ""))
    if (
        not resampling_path.is_file()
        or resampling.get("sha256") != sha256_file(resampling_path)
        or resampling.get("size") != resampling_path.stat().st_size
    ):
        raise RuntimeError("analysis resampling archive failed size/SHA binding")
    with np.load(resampling_path) as archive:
        key = resampling.get("config_sha256_key")
        if key not in archive or str(archive[key][0]) != config["config_sha256"]:
            raise RuntimeError("analysis resampling archive config mismatch")
        if sorted(archive.files) != resampling.get("array_keys"):
            raise RuntimeError("analysis resampling archive key-set mismatch")
    return result


def verify_compact_upload(out_root: Path) -> dict[str, Any]:
    """Reverify the immutable compact upload before any production analysis."""

    destination_path = out_root / "upload_destination.json"
    if not destination_path.is_file():
        raise RuntimeError("production analysis requires a verified compact upload destination")
    destination = json.loads(destination_path.read_text())
    config = json.loads((out_root / "compact/config_manifest.json").read_text())
    expected_prefix = f"issue2569_theory/category_validation/{config['config_sha256'][:12]}"
    if destination.get("prefix") != expected_prefix:
        raise RuntimeError("compact upload prefix does not match current config fingerprint")
    revision = destination.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("compact upload destination lacks an immutable revision")
    _remote_file_hashes(
        destination["repo_id"],
        destination["prefix"],
        out_root / "compact",
        COMPACT_RELPATHS,
        revision=revision,
    )
    return destination


def upload_compact(cfg: ValidationConfig, out_root: Path) -> dict[str, Any]:
    """Upload and verify compact extraction inputs before the long fit battery."""

    compact = out_root / "compact"
    config, _manifest = verify_local_compact(out_root)
    prefix = f"issue2569_theory/category_validation/{config['config_sha256'][:12]}"
    result = upload_artifact_set(compact, COMPACT_RELPATHS, prefix)
    result["config_sha256"] = config["config_sha256"]
    write_json_atomic(out_root / "upload_destination.json", result)
    return result


def upload_analysis(cfg: ValidationConfig, out_root: Path) -> dict[str, Any]:
    """Upload and verify raw resampling outputs after successful analysis."""

    load_verified_analysis_result(out_root)
    destination = verify_compact_upload(out_root)
    analysis = out_root / "analysis"
    result = upload_artifact_set(
        analysis,
        ANALYSIS_RELPATHS,
        f"{destination['prefix']}/analysis",
        repo_id=destination["repo_id"],
    )
    write_json_atomic(out_root / "analysis_upload.json", result)
    return result


def write_results_sentinel(cfg: ValidationConfig, out_root: Path) -> Path:
    """Write the sole poller-drainable result sentinel after verified final upload."""

    if not cfg.sentinel_dir:
        raise RuntimeError("production all-phase run requires sentinel_dir")
    result = load_verified_analysis_result(out_root)
    destination = verify_compact_upload(out_root)
    upload = json.loads((out_root / "analysis_upload.json").read_text())
    if (
        upload.get("repo_id") != destination["repo_id"]
        or upload.get("prefix") != f"{destination['prefix']}/analysis"
        or not isinstance(upload.get("revision"), str)
        or len(upload["revision"]) != 40
    ):
        raise RuntimeError("analysis upload destination is not bound to compact destination")
    _remote_file_hashes(
        upload["repo_id"],
        upload["prefix"],
        out_root / "analysis",
        ANALYSIS_RELPATHS,
        revision=upload["revision"],
    )
    note = {
        "summary": "Task #2569 category-level artifact-reuse validation completed.",
        "verdict": result["verdict"],
        "coverage": result["coverage"],
        "primary_contrasts": result["primary_contrasts"],
        "hf_artifacts": upload,
        "result_path": str(out_root / "analysis/category_kernel_validation_L19.json"),
        "figure_path": str(
            out_root / "git_payload/figures/issue_2569/leg13_category_kernel_validation.pdf"
        ),
        "reproducibility_card": {
            "phase": "category-validation",
            "git_commit": result["provenance"]["git_commit"],
            "config_sha256": result["config_sha256"],
            "capture_revision": CAPTURE_REVISION,
            "map_revision": MAP_REVISION,
            "seed": cfg.seed,
            "bootstrap_draws": cfg.bootstrap_draws,
            "permutation_draws": cfg.permutation_draws,
            "directional_draws": cfg.directional_draws,
        },
    }
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": ISSUE,
        "gate": "category-validation",
        "blocks_pipeline": False,
        "by": "issue2569_category_validation",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    sentinel_dir = Path(cfg.sentinel_dir).resolve()
    path = sentinel_dir / f"issue-{ISSUE}-epm_results-{int(time.time())}.json"
    write_json_atomic(path, payload)
    return path


def run(cfg: ValidationConfig) -> None:
    """Execute the configured phase with fail-loud sentinels and progress."""

    repo_root = Path(cfg.repo_root).resolve()
    out_root = Path(cfg.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[entry] issue={ISSUE} phase={cfg.phase} out_root={out_root} production={cfg.production}",
        flush=True,
    )
    if cfg.production:
        if cfg.bootstrap_draws != BOOTSTRAP_DRAWS:
            raise ValueError("production bootstrap_draws must remain 2000")
        if cfg.permutation_draws != PERMUTATION_DRAWS:
            raise ValueError("production permutation_draws must remain 999")
        if cfg.directional_draws != DIRECTIONAL_DRAWS:
            raise ValueError("production directional_draws must remain 10000")
        if cfg.seed != SEED:
            raise ValueError("production seed must remain 25691738")
        if not cfg.upload:
            raise ValueError("production analysis requires upload=true")
        if cfg.max_shards:
            raise ValueError("production extraction requires max_shards=0")
        if cfg.phase == "all" and not cfg.sentinel_dir:
            raise ValueError("production all-phase run requires sentinel_dir")
    phases = ("extract", "upload_compact", "analyze", "render", "upload_analysis")
    selected = phases if cfg.phase == "all" else (cfg.phase,)
    unknown = set(selected) - set(phases)
    if unknown:
        raise ValueError(f"unknown phase(s): {sorted(unknown)}")
    for phase in selected:
        phase_started = time.monotonic()
        print(f"[phase={phase}] starting", flush=True)
        if phase == "extract":
            extraction = extract_compact(cfg, repo_root, out_root)
            if extraction.get("status") == "partial-extraction":
                write_json_atomic(
                    out_root / "PARTIAL_EXTRACTION.json",
                    {
                        "status": "partial-extraction",
                        "shards": extraction["shards"],
                        "script_sha256": sha256_file(Path(__file__).resolve()),
                    },
                )
                print("[extract] partial extraction stopped without completion marker", flush=True)
                return
        elif phase == "upload_compact":
            if cfg.upload:
                upload_compact(cfg, out_root)
            else:
                LOGGER.warning("upload_compact skipped because upload=false")
        elif phase == "analyze":
            analyze_compact(cfg, repo_root, out_root)
        elif phase == "render":
            result = load_verified_analysis_result(out_root)
            export_git_payload(result, out_root)
            render_figure(result, out_root)
        elif phase == "upload_analysis":
            if cfg.upload:
                upload_analysis(cfg, out_root)
            else:
                LOGGER.warning("upload_analysis skipped because upload=false")
        write_json_atomic(
            out_root / f"phase_{phase}.done.json",
            {
                "phase": phase,
                "elapsed_seconds": time.monotonic() - phase_started,
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
        )
        print(
            f"[phase={phase}] complete elapsed={time.monotonic() - phase_started:.1f}s", flush=True
        )
    write_json_atomic(
        out_root / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "phases": list(selected),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "peak_rss_gb": rss_gb(),
        },
    )
    if cfg.production and cfg.phase == "all":
        write_results_sentinel(cfg, out_root)
    # workflow-lint: phase-done-reserved — terminal for this standalone all-phase dispatcher.
    print("[phase=done]", flush=True)


@hydra.main(version_base="1.3", config_path=None, config_name="issue2569_category_validation")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    resolved = ValidationConfig(**OmegaConf.to_container(cfg, resolve=True))
    run(resolved)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)
