#!/usr/bin/env python3
"""Compare frozen task-66 marker outcomes with label-free persona graph distances."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, rankdata, spearmanr
from threadpoolctl import threadpool_limits
from vendor.personamanifold.geometry import PersonaManifold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/marker_geodesic_20261002"


def digest(path: Path) -> str:
    """Hash a persisted input or result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates_from_gram(gram: np.ndarray) -> np.ndarray:
    """Recover coordinates preserving all input inner products, up to rotation."""
    np.testing.assert_allclose(gram, gram.T, atol=1e-10, rtol=1e-12)
    values, vectors = np.linalg.eigh(gram)
    assert values.min() > 0, "The archived raw Gram matrix must be full rank."
    x = vectors * np.sqrt(values)
    np.testing.assert_allclose(x @ x.T, gram, atol=1e-8, rtol=1e-10)
    return x


def cosine_from_gram(gram: np.ndarray) -> np.ndarray:
    """Normalize a Gram matrix without changing its origin."""
    norms = np.sqrt(np.diag(gram))
    assert np.all(norms > 0)
    return gram / np.outer(norms, norms)


def intrinsic_dimension(x: np.ndarray, neighbors: int = 20) -> dict:
    """Estimate pooled inverse-log nearest-neighbor dimension after 99% PCA."""
    u, singular, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    retained = int(np.searchsorted(np.cumsum(singular**2) / sum(singular**2), 0.99) + 1)
    points = u[:, :retained] * singular[:retained]
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    radii = np.sort(distances, axis=1)
    estimates = {
        str(k): float(1 / np.mean(np.log(radii[:, k - 1, None] / radii[:, : k - 1])))
        for k in (10, 15, 20, 30)
    }
    dimension = int(np.rint(estimates[str(neighbors)]))
    assert 1 <= dimension <= retained and 5 * dimension < len(x)
    return {
        "pca_dimensions": retained,
        "dimension_estimates": estimates,
        "tangent_dim": dimension,
        "k": 5 * dimension,
    }


def fit_geometry(x: np.ndarray, k: int, dimension: int, ridge: float) -> tuple:
    """Use the unchanged upstream implementation and matched graph controls."""
    manifold = PersonaManifold(
        n_neighbors=k, tangent_dim=dimension, variance=0.99, ridge=ridge
    ).fit(x)
    components, labels = connected_components(manifold.graph_, directed=False)
    assert components == 1, f"Disconnected graph: {components}, sizes={np.bincount(labels)}"
    distances = manifold.distances()
    assert np.isfinite(distances).all()
    np.testing.assert_allclose(distances, distances.T, atol=1e-9, rtol=1e-10)
    points = manifold.points_
    euclidean = cdist(points, points)
    rows, cols = manifold.graph_.nonzero()
    egraph = csr_matrix((euclidean[rows, cols], (rows, cols)), shape=distances.shape)
    edist = dijkstra(egraph, directed=False)
    offsets = points[None, :, :] - points[:, None, :]
    squares = np.einsum("ijd,idk,ijk->ij", offsets, manifold.metrics_, offsets, optimize=True)
    assert squares.min() >= -1e-8
    local_lengths = np.sqrt(np.maximum(squares, 0))
    direct = (local_lengths + local_lengths.T) / 2
    np.testing.assert_allclose(
        direct[rows, cols],
        np.asarray(manifold.graph_[rows, cols]).ravel(),
        rtol=1e-8,
        atol=1e-8,
    )
    off_diagonal = ~np.eye(len(x), dtype=bool)
    diagnostics = {
        "k": k,
        "tangent_dim": dimension,
        "ridge": ridge,
        "connected_components": int(components),
        "edge_count": int(manifold.graph_.nnz // 2),
        "edge_density": float(manifold.graph_.nnz / (len(x) * (len(x) - 1))),
        "pca_dimensions": int(points.shape[1]),
        "euclidean_path_stretch_median": float(
            np.median(edist[off_diagonal] / euclidean[off_diagonal])
        ),
        "euclidean_path_stretch_p90": float(
            np.quantile(edist[off_diagonal] / euclidean[off_diagonal], 0.9)
        ),
        "weighted_direct_edge_shortcut_fraction": float(
            np.mean(distances[rows, cols] < direct[rows, cols] - 1e-8)
        ),
    }
    global_white = points / np.sqrt(np.mean(points**2, axis=0) + ridge)
    controls = {
        "pca_euclidean": -euclidean,
        "mahalanobis": -cdist(global_white, global_white),
        "graph_euclidean": -edist,
        "local_metric_direct": -direct,
        "geodesic": -distances,
    }
    return controls, diagnostics


def base_metrics(x: np.ndarray) -> tuple:
    """Build historical centered cosine and requested uncentered controls."""
    gram = x @ x.T
    centered = x - x.mean(axis=0)
    values, vectors = np.linalg.eigh(gram)
    # True ambient dimension is 3584>N, so the unregularized second moment has null axes.
    ridge = float(values.max() / len(x) / (1000 - 1))
    white_gram = (vectors * (values / (values / len(x) + ridge))) @ vectors.T
    return {
        "centered_cosine": cosine_from_gram(centered @ centered.T),
        "raw_cosine": cosine_from_gram(gram),
        "whitened_cosine": cosine_from_gram(white_gram),
        "euclidean": -cdist(x, x),
    }, ridge


def correlations(scores: np.ndarray, y: np.ndarray, allowed: np.ndarray) -> dict:
    """Summarize five fixed-source associations and their descriptive pooled read."""
    records = []
    for i in range(len(y)):
        mask = allowed[i]
        sx, sy = scores[i, mask], y[i, mask]
        assert len(sx) >= 3 and np.ptp(sx) > 0 and np.ptp(sy) > 0
        records.append(
            {
                "n": int(mask.sum()),
                "spearman_rho": float(spearmanr(sx, sy).statistic),
                "pearson_r": float(pearsonr(sx, sy).statistic),
            }
        )
    return {
        "by_source": records,
        "macro_spearman_rho": float(np.mean([r["spearman_rho"] for r in records])),
        "pooled": {
            "n": int(allowed.sum()),
            "spearman_rho": float(spearmanr(scores[allowed], y[allowed]).statistic),
            "pearson_r": float(pearsonr(scores[allowed], y[allowed]).statistic),
        },
    }


def row_rank_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute paired Spearman along the final dimension with identical missing masks."""
    assert np.array_equal(np.isnan(x), np.isnan(y))
    rx = rankdata(x, axis=-1, nan_policy="omit")
    ry = rankdata(y, axis=-1, nan_policy="omit")
    rx -= np.nanmean(rx, axis=-1, keepdims=True)
    ry -= np.nanmean(ry, axis=-1, keepdims=True)
    denominator = np.sqrt(np.nansum(rx**2, axis=-1) * np.nansum(ry**2, axis=-1))
    assert np.all(denominator > 0), "Constant bootstrap sample must be reported, not filled."
    return np.nansum(rx * ry, axis=-1) / denominator


def paired_bootstrap(scores: dict, y: np.ndarray, allowed: np.ndarray, protocol: dict) -> dict:
    """Resample target IDs jointly across fixed adapters, keeping geometry fixed."""
    keys = ["centered_cosine", "raw_cosine", "whitened_cosine", "graph_euclidean", "geodesic"]
    draws = protocol["draws"]
    rng = np.random.default_rng(protocol["seed"])
    results = {k: [] for k in keys}
    for start in range(0, draws, 250):
        indices = rng.integers(0, y.shape[1], size=(min(250, draws - start), y.shape[1]))
        by = np.take(np.where(allowed, y, np.nan), indices, axis=1)
        for key in keys:
            bx = np.take(np.where(allowed, scores[key], np.nan), indices, axis=1)
            results[key].append(row_rank_correlation(bx, by).T)
    samples = {k: np.concatenate(v, axis=0) for k, v in results.items()}
    result = {"draws": draws, "seed": protocol["seed"], "by_metric": {}, "geodesic_minus": {}}
    for key, values in samples.items():
        result["by_metric"][key] = {
            "source_ci95": np.quantile(values, [0.025, 0.975], axis=0).T.tolist(),
            "macro_ci95": np.quantile(values.mean(axis=1), [0.025, 0.975]).tolist(),
        }
        if key != "geodesic":
            difference = samples["geodesic"] - values
            result["geodesic_minus"][key] = {
                "source_delta_ci95": np.quantile(difference, [0.025, 0.975], axis=0).T.tolist(),
                "macro_delta_ci95": np.quantile(difference.mean(axis=1), [0.025, 0.975]).tolist(),
            }
    return result


def main() -> None:
    """Execute the frozen CPU reanalysis and retain every planned result."""
    started = time.monotonic()
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(OUT / "inputs.json") == protocol["inputs_sha256"]
    data = json.loads((OUT / "inputs.json").read_text())
    names, sources = data["persona_names"], data["source_personas"]
    source_ids = np.array([names.index(s) for s in sources])
    y = np.array([[data["marker_evaluations"][s][n]["rate"] for n in names] for s in sources])
    allowed = np.ones_like(y, dtype=bool)
    allowed[np.arange(len(sources)), source_ids] = False
    assert y.shape == (5, 111) and allowed.sum() == 550
    output = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs_sha256": digest(OUT / "inputs.json"),
        "source_personas": sources,
        "layers": {},
        "sensitivity": [],
        "bootstrap": None,
    }
    matrices = {}
    for layer in protocol["all_layers"]:
        x = coordinates_from_gram(np.asarray(data["raw_gram_by_layer"][str(layer)]))
        dimension = intrinsic_dimension(x)
        base, ridge = base_metrics(x)
        geometry, diagnostics = fit_geometry(x, dimension["k"], dimension["tangent_dim"], 1e-4)
        metrics = base | geometry
        scores = {k: matrix[source_ids] for k, matrix in metrics.items()}
        stats = {k: correlations(v, y, allowed) for k, v in scores.items()}
        for i, source in enumerate(sources):
            previous = data["archived_correlations"][str(layer)][source]["spearman_rho"]
            actual = stats["centered_cosine"]["by_source"][i]["spearman_rho"]
            assert abs(previous - actual) <= 0.000051, (layer, source, previous, actual)
        output["layers"][str(layer)] = {
            "dimension": dimension,
            "geometry": diagnostics,
            "metrics": stats,
            "uncentered_whitening_ridge": ridge,
        }
        matrices.update({f"layer{layer}_{k}": v for k, v in metrics.items()})
        print(
            f"Layer {layer}: d={dimension['tangent_dim']} k={dimension['k']} "
            f"centered rho={stats['centered_cosine']['macro_spearman_rho']:.4f} "
            f"geodesic rho={stats['geodesic']['macro_spearman_rho']:.4f}",
            flush=True,
        )
        if layer == protocol["primary_layer"]:
            primary_x, primary_scores = x, scores
            original = np.array(
                [data["marker_evaluations"][sources[0]][n]["category"] == "original" for n in names]
            )
            masks = {
                "exclude_training_negative_personas": allowed & ~original[None, :],
                "positive_leakage_only": allowed & (y > 0),
            }
            output["target_subsets"] = {
                subset: {k: correlations(v, y, mask) for k, v in scores.items()}
                for subset, mask in masks.items()
            }
    primary_dimension = output["layers"][str(protocol["primary_layer"])]["dimension"]
    settings = [(k, d, 1e-4) for k in [18, 30, 42] for d in [3, 6, 9] if k >= 3 * d]
    settings += [
        (primary_dimension["k"], primary_dimension["tangent_dim"], ridge) for ridge in [1e-5, 1e-3]
    ]
    settings += [(100, 20, 1e-4)]
    for k, dimension, ridge in settings:
        geometry, diagnostics = fit_geometry(primary_x, k, dimension, ridge)
        output["sensitivity"].append(
            {
                "geometry": diagnostics,
                "metrics": {
                    name: correlations(matrix[source_ids], y, allowed)
                    for name, matrix in geometry.items()
                },
            }
        )
    print("All graph sensitivity settings completed; paired target bootstrap starting.", flush=True)
    output["bootstrap"] = paired_bootstrap(primary_scores, y, allowed, protocol["uncertainty"])
    output["elapsed_seconds"] = time.monotonic() - started
    output["completed_at_utc"] = datetime.now(UTC).isoformat()
    np.savez_compressed(OUT / "distance_matrices.npz", **matrices)
    (OUT / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "elapsed_seconds": output["elapsed_seconds"],
                "paired_differences": output["bootstrap"]["geodesic_minus"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
