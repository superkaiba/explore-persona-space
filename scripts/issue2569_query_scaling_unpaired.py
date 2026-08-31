#!/usr/bin/env python3
"""Extend issue #2569 query scaling and test cross-model unpaired alignment.

The paired arm continues the existing affine-span transport curve.  The
unpaired arm receives k source-model prompts and k *different* target-model
prompts.  It fits separate rank-r PCA coordinates, seeds a cross-model
orthogonal bridge by matching marginal component signatures, then refines the
bridge with mutual-nearest-neighbour Procrustes self-learning.  No cross-model
row identity is exposed to that aligner.  All scores use the frozen paired test
set only after fitting is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

import issue2569_fewshot_transfer as FT  # noqa: E402
import issue2569_mapping_diff as MD  # noqa: E402


WRITERS = ("qwriter", "lwriter")
DIRECTIONS = (("q", "l"), ("l", "q"))


@dataclass(frozen=True)
class UnpairedBridge:
    rotation: np.ndarray
    objective: float
    initial_objective: float
    initializer: str
    mutual_pairs: int
    iterations: int


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha_indices(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pca_seed(base_seed: int, direction_index: int, k: int, repeat: int, view: int) -> int:
    """Assign a distinct deterministic randomized-PCA seed to every fitted view."""
    if min(direction_index, k, repeat, view) < 0 or view >= 10:
        raise ValueError("invalid PCA seed coordinate")
    return base_seed + direction_index * 1_000_000 + k * 100 + repeat * 10 + view


def validate_k_values(k_values: list[int], n_train: int, *, unpaired: bool) -> None:
    if not k_values or min(k_values) < 4:
        raise ValueError("all query counts must be at least four")
    limit = n_train // 2 if unpaired else n_train
    if max(k_values) > limit:
        kind = "unpaired per-model" if unpaired else "paired"
        raise ValueError(f"{kind} k cannot exceed {limit}")


def disjoint_anchor_sets(
    train_indices: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = rng.choice(train_indices, size=2 * k, replace=False)
    source = np.asarray(chosen[:k], np.int64)
    target = np.asarray(chosen[k:], np.int64)
    if np.intersect1d(source, target).size:
        raise AssertionError("unpaired anchor sets overlap")
    return source, target


def ridge_kernel_weights_cholesky(
    query: np.ndarray,
    anchors: np.ndarray,
    *,
    ridge_fraction: float,
    device: str,
) -> np.ndarray:
    """Compute affine kernel-ridge weights with a symmetric Cholesky solve."""
    a = np.asarray(anchors, np.float32)
    q = np.asarray(query, np.float32)
    mean = a.mean(0)
    ac = a - mean
    qc = q - mean
    dev = torch.device(device)
    with torch.inference_mode():
        at = torch.as_tensor(ac, dtype=torch.float32, device=dev)
        qt = torch.as_tensor(qc, dtype=torch.float32, device=dev)
        gram = at @ at.T
        scale = torch.trace(gram) / max(len(a) - 1, 1)
        lam = torch.clamp(scale * ridge_fraction, min=1e-8)
        gram.diagonal().add_(lam)
        chol = torch.linalg.cholesky(gram)
        solved = torch.cholesky_solve(at, chol)
        weights = qt @ solved.T
        result = weights.cpu().numpy()
    del at, qt, gram, chol, solved, weights
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result


def span_from_weights(weights: np.ndarray, outputs: np.ndarray, device: str) -> np.ndarray:
    out = np.asarray(outputs, np.float32)
    mean = out.mean(0)
    return FT.gpu_matmul(weights, out - mean, device) + mean


def affine_span_predict_cholesky(
    query: np.ndarray,
    input_anchors: np.ndarray,
    output_anchors: np.ndarray,
    *,
    ridge_fraction: float,
    device: str,
) -> np.ndarray:
    weights = ridge_kernel_weights_cholesky(
        query,
        input_anchors,
        ridge_fraction=ridge_fraction,
        device=device,
    )
    return span_from_weights(weights, output_anchors, device)


def component_signatures(scores: np.ndarray) -> np.ndarray:
    """Return sign-sensitive marginal signatures for PCA components."""
    z = np.asarray(scores, np.float64)
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-8)
    skew = np.mean(z**3, axis=0)
    kurtosis = np.mean(z**4, axis=0) - 3.0
    quantiles = np.quantile(z, [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95], axis=0).T
    return np.column_stack([skew, kurtosis, quantiles])


def moment_seed_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match component identities and signs using unpaired marginal moments."""
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target score ranks differ")
    source_sig = component_signatures(source)
    target_pos = component_signatures(target)
    target_neg = component_signatures(-target)
    all_sig = np.vstack([source_sig, target_pos, target_neg])
    scale = np.maximum(np.std(all_sig, axis=0), 1e-6)
    source_sig = source_sig / scale
    target_pos = target_pos / scale
    target_neg = target_neg / scale
    pos_cost = np.linalg.norm(source_sig[:, None, :] - target_pos[None, :, :], axis=2)
    neg_cost = np.linalg.norm(source_sig[:, None, :] - target_neg[None, :, :], axis=2)
    cost = np.minimum(pos_cost, neg_cost)
    rows, cols = linear_sum_assignment(cost)
    rotation = np.zeros((source.shape[1], source.shape[1]), dtype=np.float64)
    for row, col in zip(rows, cols, strict=True):
        rotation[row, col] = 1.0 if pos_cost[row, col] <= neg_cost[row, col] else -1.0
    return rotation


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross = np.asarray(source, np.float64).T @ np.asarray(target, np.float64)
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return u @ vt


def nearest_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine nearest neighbour in chunks, returning index and similarity."""
    dev = torch.device(device)
    x = np.asarray(source, np.float32)
    y = np.asarray(target, np.float32)
    with torch.inference_mode():
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
        yt = torch.nn.functional.normalize(yt, dim=1)
        indices: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for start in range(0, len(x), chunk_size):
            xt = torch.as_tensor(x[start : start + chunk_size], dtype=torch.float32, device=dev)
            xt = torch.nn.functional.normalize(xt, dim=1)
            value, index = torch.max(xt @ yt.T, dim=1)
            indices.append(index.cpu().numpy())
            values.append(value.cpu().numpy())
    del yt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(indices), np.concatenate(values)


def symmetric_chamfer_cosine(
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
    chunk_size: int,
) -> float:
    _, forward = nearest_neighbors(source, target, device=device, chunk_size=chunk_size)
    _, backward = nearest_neighbors(target, source, device=device, chunk_size=chunk_size)
    return float(0.5 * (forward.mean() + backward.mean()))


def unrefined_random_orientation_references(
    source: np.ndarray,
    target: np.ndarray,
    *,
    draws: int,
    seed_coordinates: tuple[int, ...],
    device: str,
    chunk_size: int,
) -> list[float]:
    """Unrefined Haar-like rotation references, not estimator-matched nulls."""
    if draws < 1:
        raise ValueError("random-orientation reference draws must be positive")
    rank = int(np.asarray(source).shape[1])
    rng = np.random.default_rng(np.random.SeedSequence(seed_coordinates))
    values: list[float] = []
    for _ in range(draws):
        rotation, triangular = np.linalg.qr(rng.normal(size=(rank, rank)))
        signs = np.sign(np.diag(triangular))
        signs[signs == 0] = 1
        rotation = rotation * signs
        values.append(
            symmetric_chamfer_cosine(
                np.asarray(source) @ rotation,
                target,
                device=device,
                chunk_size=chunk_size,
            )
        )
    return values


def fit_unpaired_bridge(
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
    max_iterations: int,
    chunk_size: int,
) -> UnpairedBridge:
    """Best-of-two-seed mutual-NN Procrustes without cross-model row pairs."""
    x = np.asarray(source, np.float64)
    y = np.asarray(target, np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("unpaired bridge expects 2D equal-rank point clouds")
    candidates = {
        "variance_rank_identity": np.eye(x.shape[1], dtype=np.float64),
        "marginal_moment_assignment": moment_seed_rotation(x, y),
    }
    best: UnpairedBridge | None = None
    for initializer, initial in candidates.items():
        rotation = initial
        initial_objective = symmetric_chamfer_cosine(
            x @ rotation,
            y,
            device=device,
            chunk_size=chunk_size,
        )
        local_best_rotation = rotation.copy()
        local_best_objective = initial_objective
        local_best_pairs = 0
        completed = 0
        for iteration in range(max_iterations):
            transformed = x @ rotation
            source_to_target, _ = nearest_neighbors(
                transformed,
                y,
                device=device,
                chunk_size=chunk_size,
            )
            target_to_source, _ = nearest_neighbors(
                y,
                transformed,
                device=device,
                chunk_size=chunk_size,
            )
            source_rows = np.arange(len(x), dtype=np.int64)
            mutual = target_to_source[source_to_target] == source_rows
            pair_source = source_rows[mutual]
            pair_target = source_to_target[mutual]
            if len(pair_source) < max(16, x.shape[1] // 2):
                pair_source = source_rows
                pair_target = source_to_target
            if iteration == 0:
                local_best_pairs = int(len(pair_source))
            updated = orthogonal_procrustes(x[pair_source], y[pair_target])
            objective = symmetric_chamfer_cosine(
                x @ updated,
                y,
                device=device,
                chunk_size=chunk_size,
            )
            completed = iteration + 1
            if objective > local_best_objective:
                local_best_objective = objective
                local_best_rotation = updated.copy()
                local_best_pairs = int(len(pair_source))
            delta = float(np.linalg.norm(updated - rotation) / np.sqrt(rotation.size))
            rotation = updated
            if delta < 1e-5:
                break
        candidate = UnpairedBridge(
            rotation=local_best_rotation,
            objective=local_best_objective,
            initial_objective=initial_objective,
            initializer=initializer,
            mutual_pairs=local_best_pairs,
            iterations=completed,
        )
        if best is None or candidate.objective > best.objective:
            best = candidate
    assert best is not None
    return best


def centered_flat_cosine(predicted: np.ndarray, observed: np.ndarray, mean: np.ndarray) -> float:
    return MD.flat_cosine(
        np.asarray(predicted, np.float64) - np.asarray(mean, np.float64),
        np.asarray(observed, np.float64) - np.asarray(mean, np.float64),
    )


def bridge_record(bridge: UnpairedBridge) -> dict[str, Any]:
    return {
        "unsupervised_objective": float(bridge.objective),
        "initial_objective": float(bridge.initial_objective),
        "selected_initializer": bridge.initializer,
        "mutual_pairs_at_best_step": int(bridge.mutual_pairs),
        "iterations": int(bridge.iterations),
    }


def summarize_scalar_records(
    records: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_repeats": len(records)}
    for key in keys:
        values = np.asarray([record[key] for record in records], np.float64)
        out[key] = {
            "median": float(np.median(values)),
            "min_max": [float(np.min(values)), float(np.max(values))],
            "q10_q90": [float(value) for value in np.quantile(values, [0.1, 0.9])],
            "values": [float(value) for value in values],
        }
    return out


def summarize_vector_records(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    matrix = np.asarray([record[key] for record in records], np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{key} must contain equal-length nonempty vectors")
    values = matrix.reshape(-1)
    return {
        "n_repeats": int(matrix.shape[0]),
        "draws_per_repeat": int(matrix.shape[1]),
        "median": float(np.median(values)),
        "min_max": [float(np.min(values)), float(np.max(values))],
        "q10_q90": [float(value) for value in np.quantile(values, [0.1, 0.9])],
        "values": [float(value) for value in values],
        "per_repeat_values": [[float(value) for value in row] for row in matrix],
    }


def summarize_unpaired_bridge_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_scalar_records(
        records,
        (
            "context_paired_test_cosine",
            "answer_paired_test_cosine",
            "context_unsupervised_objective",
            "answer_unsupervised_objective",
            "context_initial_objective",
            "answer_initial_objective",
            "context_mutual_pairs",
            "answer_mutual_pairs",
            "context_initializer_moment",
            "answer_initializer_moment",
        ),
    )
    for key in (
        "context_unrefined_random_orientation_reference",
        "answer_unrefined_random_orientation_reference",
    ):
        summary[key] = summarize_vector_records(records, key)
    return summary


def summarize_prediction_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the parent summary schema while retaining every repeat value."""
    summary = FT.summarize_repeats(rows)
    for group, metric in (
        ("observed_target", "pooled_r2"),
        ("observed_target", "train_mean_normalized_r2"),
        ("observed_target", "centered_cosine"),
        ("full_target_mapping", "normalized_r2"),
        ("full_target_mapping", "centered_cosine"),
        ("full_target_mapping", "relative_l2"),
    ):
        summary[group][metric]["values"] = [float(row[group][metric]) for row in rows]
    return summary


def phase_analyze(args: argparse.Namespace) -> None:
    started = time.time()
    roster, folds, arrays, payloads, records = MD.load_primary(args)
    train = folds["tr"]
    test = folds["te"]
    validate_k_values(args.paired_k_values, len(train), unpaired=False)
    validate_k_values(args.unpaired_k_values, len(train), unpaired=True)
    native_context = {"q": arrays["q_context"], "l": arrays["l_context"]}
    native_answers = {
        ("q", "qwriter"): arrays["q_qwriter"],
        ("l", "qwriter"): arrays["l_qwriter"],
        ("q", "lwriter"): arrays["q_lwriter"],
        ("l", "lwriter"): arrays["l_lwriter"],
    }
    native_cells = {
        ("q", "qwriter"): MD.payload_affine(payloads["q_qwriter"]),
        ("l", "qwriter"): MD.payload_affine(payloads["l_qwriter"]),
        ("q", "lwriter"): MD.payload_affine(payloads["q_lwriter"]),
        ("l", "lwriter"): MD.payload_affine(payloads["l_lwriter"]),
    }
    map_predictions = {
        key: FT.predict_affine_gpu(cell, native_context[key[0]], args.device)
        for key, cell in native_cells.items()
    }
    target_means = {
        key: np.asarray(value[train], np.float64).mean(0) for key, value in native_answers.items()
    }
    print(f"[query-scaling-unpaired] loaded n={len(roster)}", flush=True)

    paired_results: dict[str, Any] = {}
    paired_rng = np.random.default_rng(args.seed)
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        paired_results[direction] = {writer: {} for writer in WRITERS}
        for k in args.paired_k_values:
            per_writer: dict[str, dict[str, list[dict[str, Any]]]] = {
                writer: {"transport": [], "scratch": []} for writer in WRITERS
            }
            anchor_hashes: list[str] = []
            for _ in range(args.paired_repeats):
                anchor = paired_rng.choice(train, size=k, replace=False)
                anchor_hashes.append(sha_indices(roster[anchor]))
                context_weights = ridge_kernel_weights_cholesky(
                    native_context[target][test],
                    native_context[target][anchor],
                    ridge_fraction=args.ridge_fraction,
                    device=args.device,
                )
                for writer in WRITERS:
                    source_map_at_test = span_from_weights(
                        context_weights,
                        map_predictions[(source, writer)][anchor],
                        args.device,
                    )
                    transported = affine_span_predict_cholesky(
                        source_map_at_test,
                        native_answers[(source, writer)][anchor],
                        native_answers[(target, writer)][anchor],
                        ridge_fraction=args.ridge_fraction,
                        device=args.device,
                    )
                    scratch = span_from_weights(
                        context_weights,
                        native_answers[(target, writer)][anchor],
                        args.device,
                    )
                    observed = native_answers[(target, writer)][test]
                    full_map = map_predictions[(target, writer)][test]
                    mean = target_means[(target, writer)]
                    per_writer[writer]["transport"].append(
                        FT.prediction_metrics(transported, observed, full_map, mean)
                    )
                    per_writer[writer]["scratch"].append(
                        FT.prediction_metrics(scratch, observed, full_map, mean)
                    )
            for writer in WRITERS:
                transport_rows = per_writer[writer]["transport"]
                scratch_rows = per_writer[writer]["scratch"]
                paired_results[direction][writer][str(k)] = {
                    "transported_source_mapping": summarize_prediction_repeats(transport_rows),
                    "target_fit_from_scratch": summarize_prediction_repeats(scratch_rows),
                    "paired_transport_advantage": FT.summarize_paired_advantage(
                        transport_rows,
                        scratch_rows,
                    ),
                    "anchor_ci_sha256": anchor_hashes,
                }
            print(
                f"[query-scaling-unpaired] paired {direction} k={k} repeats={args.paired_repeats}",
                flush=True,
            )

    unpaired_results: dict[str, Any] = {}
    paired_oracle_results: dict[str, Any] = {}
    unpaired_rng = np.random.default_rng(args.seed + 100_000)
    for direction_index, (source, target) in enumerate(DIRECTIONS):
        direction = f"{source}_to_{target}"
        unpaired_results[direction] = {writer: {} for writer in WRITERS}
        paired_oracle_results[direction] = {writer: {} for writer in WRITERS}
        for k in args.unpaired_k_values:
            per_writer = {
                writer: {
                    "transport": [],
                    "scratch": [],
                    "bridge": [],
                    "paired_oracle": [],
                    "paired_oracle_bridge": [],
                }
                for writer in WRITERS
            }
            provenance: list[dict[str, Any]] = []
            for repeat in range(args.unpaired_repeats):
                source_anchor, target_anchor = disjoint_anchor_sets(train, k, unpaired_rng)
                overlap = int(np.intersect1d(roster[source_anchor], roster[target_anchor]).size)
                if overlap:
                    raise AssertionError("cross-model query IDs overlap")
                provenance.append(
                    {
                        "source_ci_sha256": sha_indices(roster[source_anchor]),
                        "target_ci_sha256": sha_indices(roster[target_anchor]),
                        "cross_model_ci_intersection": overlap,
                    }
                )
                rank = min(args.unpaired_rank, k - 1)
                source_context_summary = FT.fit_pc_summary(
                    native_context[source],
                    source_anchor,
                    rank,
                    device=args.device,
                    seed=pca_seed(args.seed, direction_index, k, repeat, 0),
                )
                target_context_summary = FT.fit_pc_summary(
                    native_context[target],
                    target_anchor,
                    rank,
                    device=args.device,
                    seed=pca_seed(args.seed, direction_index, k, repeat, 1),
                )
                paired_target_context_summary = FT.fit_pc_summary(
                    native_context[target],
                    source_anchor,
                    rank,
                    device=args.device,
                    seed=pca_seed(args.seed, direction_index, k, repeat, 2),
                )
                target_context_scores = FT.pc_scores(
                    native_context[target][target_anchor],
                    target_context_summary,
                    args.device,
                )
                source_context_scores = FT.pc_scores(
                    native_context[source][source_anchor],
                    source_context_summary,
                    args.device,
                )
                context_bridge = fit_unpaired_bridge(
                    target_context_scores,
                    source_context_scores,
                    device=args.device,
                    max_iterations=args.unpaired_iterations,
                    chunk_size=args.nn_chunk_size,
                )
                context_reference = unrefined_random_orientation_references(
                    target_context_scores,
                    source_context_scores,
                    draws=args.random_orientation_reference_draws,
                    seed_coordinates=(args.seed, 2569, direction_index, k, repeat, 0),
                    device=args.device,
                    chunk_size=args.nn_chunk_size,
                )
                test_target_scores = FT.pc_scores(
                    native_context[target][test],
                    target_context_summary,
                    args.device,
                )
                source_context_hat = FT.pc_reconstruct(
                    test_target_scores @ context_bridge.rotation,
                    source_context_summary,
                    args.device,
                )
                context_test_cosine = centered_flat_cosine(
                    source_context_hat,
                    native_context[source][test],
                    native_context[source][train].mean(0),
                )
                paired_target_context_scores = FT.pc_scores(
                    native_context[target][source_anchor],
                    paired_target_context_summary,
                    args.device,
                )
                paired_context_bridge = orthogonal_procrustes(
                    paired_target_context_scores,
                    source_context_scores,
                )
                paired_test_target_scores = FT.pc_scores(
                    native_context[target][test],
                    paired_target_context_summary,
                    args.device,
                )
                paired_source_context_hat = FT.pc_reconstruct(
                    paired_test_target_scores @ paired_context_bridge,
                    source_context_summary,
                    args.device,
                )
                paired_context_test_cosine = centered_flat_cosine(
                    paired_source_context_hat,
                    native_context[source][test],
                    native_context[source][train].mean(0),
                )
                scratch_weights = ridge_kernel_weights_cholesky(
                    native_context[target][test],
                    native_context[target][target_anchor],
                    ridge_fraction=args.ridge_fraction,
                    device=args.device,
                )
                for writer in WRITERS:
                    source_answer_summary = FT.fit_pc_summary(
                        native_answers[(source, writer)],
                        source_anchor,
                        rank,
                        device=args.device,
                        seed=pca_seed(
                            args.seed,
                            direction_index,
                            k,
                            repeat,
                            3 if writer == "qwriter" else 6,
                        ),
                    )
                    target_answer_summary = FT.fit_pc_summary(
                        native_answers[(target, writer)],
                        target_anchor,
                        rank,
                        device=args.device,
                        seed=pca_seed(
                            args.seed,
                            direction_index,
                            k,
                            repeat,
                            4 if writer == "qwriter" else 7,
                        ),
                    )
                    paired_target_answer_summary = FT.fit_pc_summary(
                        native_answers[(target, writer)],
                        source_anchor,
                        rank,
                        device=args.device,
                        seed=pca_seed(
                            args.seed,
                            direction_index,
                            k,
                            repeat,
                            5 if writer == "qwriter" else 8,
                        ),
                    )
                    source_answer_scores = FT.pc_scores(
                        native_answers[(source, writer)][source_anchor],
                        source_answer_summary,
                        args.device,
                    )
                    target_answer_scores = FT.pc_scores(
                        native_answers[(target, writer)][target_anchor],
                        target_answer_summary,
                        args.device,
                    )
                    answer_bridge = fit_unpaired_bridge(
                        source_answer_scores,
                        target_answer_scores,
                        device=args.device,
                        max_iterations=args.unpaired_iterations,
                        chunk_size=args.nn_chunk_size,
                    )
                    writer_index = 0 if writer == "qwriter" else 1
                    answer_reference = unrefined_random_orientation_references(
                        source_answer_scores,
                        target_answer_scores,
                        draws=args.random_orientation_reference_draws,
                        seed_coordinates=(
                            args.seed,
                            2569,
                            direction_index,
                            k,
                            repeat,
                            1,
                            writer_index,
                        ),
                        device=args.device,
                        chunk_size=args.nn_chunk_size,
                    )
                    source_map_test = FT.predict_affine_gpu(
                        native_cells[(source, writer)],
                        source_context_hat,
                        args.device,
                    )
                    source_map_scores = FT.pc_scores(
                        source_map_test,
                        source_answer_summary,
                        args.device,
                    )
                    transported = FT.pc_reconstruct(
                        source_map_scores @ answer_bridge.rotation,
                        target_answer_summary,
                        args.device,
                    )
                    scratch = span_from_weights(
                        scratch_weights,
                        native_answers[(target, writer)][target_anchor],
                        args.device,
                    )
                    observed = native_answers[(target, writer)][test]
                    full_map = map_predictions[(target, writer)][test]
                    mean = target_means[(target, writer)]
                    source_answer_test_scores = FT.pc_scores(
                        native_answers[(source, writer)][test],
                        source_answer_summary,
                        args.device,
                    )
                    target_answer_hat = FT.pc_reconstruct(
                        source_answer_test_scores @ answer_bridge.rotation,
                        target_answer_summary,
                        args.device,
                    )
                    answer_test_cosine = centered_flat_cosine(
                        target_answer_hat,
                        observed,
                        mean,
                    )
                    paired_target_answer_scores = FT.pc_scores(
                        native_answers[(target, writer)][source_anchor],
                        paired_target_answer_summary,
                        args.device,
                    )
                    paired_answer_bridge = orthogonal_procrustes(
                        source_answer_scores,
                        paired_target_answer_scores,
                    )
                    paired_source_map_test = FT.predict_affine_gpu(
                        native_cells[(source, writer)],
                        paired_source_context_hat,
                        args.device,
                    )
                    paired_source_map_scores = FT.pc_scores(
                        paired_source_map_test,
                        source_answer_summary,
                        args.device,
                    )
                    paired_oracle = FT.pc_reconstruct(
                        paired_source_map_scores @ paired_answer_bridge,
                        paired_target_answer_summary,
                        args.device,
                    )
                    paired_target_answer_hat = FT.pc_reconstruct(
                        source_answer_test_scores @ paired_answer_bridge,
                        paired_target_answer_summary,
                        args.device,
                    )
                    paired_answer_test_cosine = centered_flat_cosine(
                        paired_target_answer_hat,
                        observed,
                        mean,
                    )
                    per_writer[writer]["transport"].append(
                        FT.prediction_metrics(transported, observed, full_map, mean)
                    )
                    per_writer[writer]["scratch"].append(
                        FT.prediction_metrics(scratch, observed, full_map, mean)
                    )
                    per_writer[writer]["bridge"].append(
                        {
                            "context_paired_test_cosine": context_test_cosine,
                            "answer_paired_test_cosine": answer_test_cosine,
                            "context_unsupervised_objective": context_bridge.objective,
                            "answer_unsupervised_objective": answer_bridge.objective,
                            "context_initial_objective": context_bridge.initial_objective,
                            "answer_initial_objective": answer_bridge.initial_objective,
                            "context_unrefined_random_orientation_reference": context_reference,
                            "answer_unrefined_random_orientation_reference": answer_reference,
                            "context_mutual_pairs": context_bridge.mutual_pairs,
                            "answer_mutual_pairs": answer_bridge.mutual_pairs,
                            "context_initializer_moment": float(
                                context_bridge.initializer == "marginal_moment_assignment"
                            ),
                            "answer_initializer_moment": float(
                                answer_bridge.initializer == "marginal_moment_assignment"
                            ),
                        }
                    )
                    per_writer[writer]["paired_oracle"].append(
                        FT.prediction_metrics(paired_oracle, observed, full_map, mean)
                    )
                    per_writer[writer]["paired_oracle_bridge"].append(
                        {
                            "context_paired_test_cosine": paired_context_test_cosine,
                            "answer_paired_test_cosine": paired_answer_test_cosine,
                        }
                    )
                print(
                    f"[query-scaling-unpaired] unpaired {direction} k={k} "
                    f"repeat={repeat + 1}/{args.unpaired_repeats}",
                    flush=True,
                )
            for writer in WRITERS:
                transport_rows = per_writer[writer]["transport"]
                scratch_rows = per_writer[writer]["scratch"]
                unpaired_results[direction][writer][str(k)] = {
                    "transported_source_mapping": summarize_prediction_repeats(transport_rows),
                    "target_fit_from_scratch": summarize_prediction_repeats(scratch_rows),
                    "unpaired_transport_minus_target_scratch": FT.summarize_paired_advantage(
                        transport_rows,
                        scratch_rows,
                    ),
                    "bridge_diagnostics": summarize_unpaired_bridge_records(
                        per_writer[writer]["bridge"]
                    ),
                    "query_provenance": provenance,
                }
                paired_oracle_results[direction][writer][str(k)] = {
                    "transported_source_mapping": summarize_prediction_repeats(
                        per_writer[writer]["paired_oracle"]
                    ),
                    "bridge_diagnostics": summarize_scalar_records(
                        per_writer[writer]["paired_oracle_bridge"],
                        (
                            "context_paired_test_cosine",
                            "answer_paired_test_cosine",
                        ),
                    ),
                    "query_provenance": [
                        {"paired_ci_sha256": row["source_ci_sha256"]} for row in provenance
                    ],
                }

    full_ceiling: dict[str, Any] = {}
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        full_ceiling[direction] = {}
        for writer in WRITERS:
            target_full = map_predictions[(target, writer)][test]
            full_ceiling[direction][writer] = FT.prediction_metrics(
                target_full,
                native_answers[(target, writer)][test],
                target_full,
                target_means[(target, writer)],
            )

    max_unpaired_k = max(args.unpaired_k_values)
    repeat_partition_caveat = (
        "Unpaired partitions are disjoint within each repeat, not across repeats; "
        f"at k={max_unpaired_k:,} every repeat repartitions all {len(train):,} train rows."
        if 2 * max_unpaired_k == len(train)
        else "Unpaired source/target partitions are disjoint within each repeat, not across repeats."
    )
    result = {
        "issue": 2569,
        "followup_label": "extended-query-scaling-and-unpaired-alignment",
        "source_revision": MD.SOURCE_REVISION,
        "layers": MD.PRIMARY_LAYERS,
        "n": {
            "train": int(len(train)),
            "validation_unused": int(len(folds["va"])),
            "test": int(len(test)),
        },
        "test_roster_sha256": MD.sha_int64(roster[test]),
        "design": {
            "paired_scaling": (
                "k shared train prompts are observed in both models; the frozen source map is transported "
                "with regularized affine-span context and answer bridges. The target-scratch control uses "
                "the same k target rows."
            ),
            "unpaired_alignment": (
                "k source prompts and k different target prompts are sampled without replacement and with "
                "zero cross-model prompt-ID overlap. Separate rank-r PCA coordinates are aligned from both "
                "a variance-rank identity seed and a marginal-moment component-assignment seed. Each is "
                "refined by mutual-nearest-neighbour orthogonal Procrustes self-learning, and the higher "
                "training-objective fit is retained. Cross-model row identities are never passed to fitting."
            ),
            "unpaired_query_budget": (
                "Per direction/writer cell, both the paired-row oracle and unpaired arm read k context "
                "and answer activations from each encoder (2k model-side rows). The paired oracle uses k "
                "shared prompt/answer IDs; unpaired uses 2k distinct IDs. This counts encoder evaluations, "
                "not answer generation: the distinct writer-response inputs differ between arms. The run "
                "subsamples fixed activation arrays and issues no new model forwards."
            ),
            "paired_rank_oracle": (
                "The same rank-r separate-PCA and orthogonal context/answer bridge family as the unpaired "
                "arm, but fitted by direct Procrustes on k paired row identities. It reuses each repeat's "
                "source anchor IDs in both models and is the supervised identifiability control."
            ),
            "paired_scaling_fresh_draws": (
                "The extended full-dimensional affine-KRR points start above the original few-query "
                "endpoint, use fresh independent anchor draws, and use a symmetric Cholesky solve; no "
                "duplicate endpoint estimate is produced."
            ),
            "evaluation": (
                "The frozen 1,500-row paired test set is used only for final scoring and bridge diagnostics."
            ),
            "primary_metric": (
                "Centered cosine with the frozen 8,000-row target-map prediction on the test set."
            ),
            "hyperparameters": {
                "paired_k_values": args.paired_k_values,
                "paired_repeats": int(args.paired_repeats),
                "unpaired_k_values_per_model": args.unpaired_k_values,
                "unpaired_repeats": int(args.unpaired_repeats),
                "unpaired_rank": int(args.unpaired_rank),
                "unpaired_self_learning_iterations": int(args.unpaired_iterations),
                "unrefined_random_orientation_reference_draws_per_fit": int(
                    args.random_orientation_reference_draws
                ),
                "seed": int(args.seed),
                "device": str(args.device),
                "nn_chunk_size": int(args.nn_chunk_size),
                "ridge_fraction": float(args.ridge_fraction),
                "no_validation_tuning": True,
            },
        },
        "paired_scaling": paired_results,
        "unpaired_alignment": unpaired_results,
        "paired_rank_oracle": paired_oracle_results,
        "full_target_map_ceiling": full_ceiling,
        "native_target_map_record_test_r2": {
            name: float(records[name]["test_r2"]) for name in MD.CELL_NAMES
        },
        "analysis_driver": {
            "path": "scripts/issue2569_query_scaling_unpaired.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "caveats": [
            "The unpaired result evaluates one specific best-of-two-initializer self-learning algorithm, not the existence of every possible unsupervised alignment.",
            "The unpaired objective is selected on the same unpaired point clouds used for fitting; paired test identities are evaluation-only.",
            "Training symmetric-Chamfer objectives are shown beside unrefined random-orientation references; because the references do not run initializer selection or self-learning, they are not an estimator-matched null and fitted-minus-reference gaps are descriptive only.",
            repeat_partition_caveat,
            "The source map is pretrained on all 8,000 source train rows and is treated as an amortized artifact in both transfer arms.",
            "Query counts reuse the fixed LMSYS activation dataset rather than issuing new model API calls.",
            "This remains an exploratory two-model, one-dataset, one-layer-pair result.",
        ],
        "elapsed_s": round(time.time() - started, 2),
    }
    atomic_json(Path(args.out_dir) / "query_scaling_unpaired.json", result)
    print(
        f"[query-scaling-unpaired] wrote {args.out_dir} elapsed={result['elapsed_s']:.1f}s",
        flush=True,
    )


def phase_selftest(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(2569)
    train = np.arange(100)
    source, target = disjoint_anchor_sets(train, 40, rng)
    assert len(source) == len(target) == 40
    assert not np.intersect1d(source, target).size

    latent_source = np.column_stack(
        [rng.gamma(shape=shape, scale=1.0, size=1200) for shape in (1.1, 1.7, 2.6, 4.2)]
    )
    latent_target = np.column_stack(
        [rng.gamma(shape=shape, scale=1.0, size=1400) for shape in (1.1, 1.7, 2.6, 4.2)]
    )
    latent_source = (latent_source - latent_source.mean(0)) / latent_source.std(0)
    latent_target = (latent_target - latent_target.mean(0)) / latent_target.std(0)
    truth = np.array(
        [[0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 0.0]]
    )
    observed_target = latent_target @ truth
    seed = moment_seed_rotation(latent_source, observed_target)
    recovered = np.abs(np.diag((latent_source @ seed).T @ (latent_source @ truth)))
    assert np.all(recovered > 500)

    x = rng.normal(size=(80, 6)).astype(np.float32)
    operator = rng.normal(size=(6, 5)).astype(np.float32)
    y = x @ operator
    pred = affine_span_predict_cholesky(
        x[40:],
        x[:40],
        y[:40],
        ridge_fraction=1e-5,
        device="cpu",
    )
    assert np.allclose(pred, y[40:], atol=2e-3, rtol=2e-3)
    print("[query-scaling-unpaired] selftest PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("analyze", "selftest"), required=True)
    base = PROJECT_ROOT / "data" / "issue_2569" / "ownanswers"
    parser.add_argument("--qwriter-dir", default=str(base / "qwriter_final"))
    parser.add_argument("--lwriter-dir", default=str(base / "writer_llama" / "final"))
    parser.add_argument("--map-dir", default=str(base / "analysis" / "maps"))
    parser.add_argument("--split-json", default=str(base / "analysis" / "split.json"))
    parser.add_argument("--out-dir", default=str(base / "query_scaling_unpaired"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--paired-k-values",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4000],
    )
    parser.add_argument(
        "--unpaired-k-values",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4000],
    )
    parser.add_argument("--paired-repeats", type=int, default=10)
    parser.add_argument("--unpaired-repeats", type=int, default=5)
    parser.add_argument("--unpaired-rank", type=int, default=64)
    parser.add_argument("--unpaired-iterations", type=int, default=8)
    parser.add_argument("--random-orientation-reference-draws", type=int, default=3)
    parser.add_argument("--nn-chunk-size", type=int, default=1024)
    parser.add_argument("--ridge-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2569)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.paired_repeats < 1 or args.unpaired_repeats < 1:
        raise ValueError("repeat counts must be positive")
    if args.unpaired_rank < 2 or args.unpaired_rank > 256:
        raise ValueError("unpaired-rank must be in [2, 256]")
    if (
        args.unpaired_iterations < 0
        or args.random_orientation_reference_draws < 1
        or args.ridge_fraction <= 0
    ):
        raise ValueError("invalid unpaired iterations or ridge fraction")
    {"analyze": phase_analyze, "selftest": phase_selftest}[args.phase](args)


if __name__ == "__main__":
    main()
