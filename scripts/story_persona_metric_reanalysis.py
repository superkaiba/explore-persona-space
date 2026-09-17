#!/usr/bin/env python3
"""Pair raw and uncentered, regularized whitened cosine with published leakage."""

from __future__ import annotations

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from scipy.linalg import cho_factor, cho_solve, eigh  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from explore_persona_space.analysis.leakage_predictor import (  # noqa: E402
    SIGMA_C_COND_TARGET,
    SIGMA_C_LAMBDA_GRID,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2673_story_persona_qwen38/analysis_tensors"


def sha256(data: bytes) -> str:
    """Identify exact input bytes."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Publish a complete, finite checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cosine_from_gram(gram: np.ndarray) -> np.ndarray:
    """Normalize a positive metric Gram; reject invalid squared norms."""
    gram = np.asarray(gram, dtype=np.float64)
    if not np.isfinite(gram).all() or np.any(np.diag(gram) <= 0):
        raise ValueError("non-finite metric Gram or non-positive squared norm")
    norms = np.sqrt(np.diag(gram))
    cosine = gram / norms[:, None] / norms[None, :]
    if np.max(np.abs(cosine)) > 1 + 1e-8:
        raise ValueError("cosine outside its mathematical bounds")
    return cosine


def metric_gram(x: np.ndarray, centroids: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Evaluate C (X.T X/N + ridge I)^-1 C.T by the Woodbury identity.

    Neither X nor C is mean-subtracted. This is a second-moment metric,
    not centered covariance whitening and not the asymmetric leakage gate.
    """
    x = np.asarray(x, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if x.ndim != 2 or centroids.ndim != 2 or x.shape[1] != centroids.shape[1]:
        raise ValueError("incompatible context and centroid shapes")
    if ridge <= 0 or not np.isfinite(x).all() or not np.isfinite(centroids).all():
        raise ValueError("invalid ridge or non-finite vectors")
    u = x / np.sqrt(len(x))
    dual = u @ u.T
    dual.flat[:: len(x) + 1] += ridge
    cross = u @ centroids.T
    solved = cho_solve(cho_factor(dual, lower=True), cross)
    gram = (centroids @ centroids.T - cross.T @ solved) / ridge
    inverse_means = (centroids.T - u.T @ solved) / ridge
    residual = u.T @ (u @ inverse_means) + ridge * inverse_means - centroids.T
    relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(centroids))
    if not np.isfinite(relative_residual) or relative_residual > 1e-8:
        raise ValueError(f"implicit primal solve residual too large: {relative_residual}")
    return (gram + gram.T) / 2, relative_residual


def select_ridge(x: np.ndarray) -> tuple[float, dict]:
    """Use the prior leakage recipe's label-free grid and conditioning bound."""
    n, dim = x.shape
    if not 0 < n < dim:
        raise ValueError("this rank-deficient calibration expects 0 < N < dimension")
    dual = (x @ x.T) / n
    largest = float(eigh(dual, subset_by_index=[n - 1, n - 1], eigvals_only=True)[0])
    # X.T X/N has at least dim-N exact zero eigenvalues.
    conds = (largest + SIGMA_C_LAMBDA_GRID) / SIGMA_C_LAMBDA_GRID
    valid = np.flatnonzero(conds <= SIGMA_C_COND_TARGET)
    if not len(valid):
        raise ValueError(f"prior ridge grid cannot meet conditioning bound; largest={largest}")
    j = int(valid[0])
    return float(SIGMA_C_LAMBDA_GRID[j]), {
        "largest_second_moment_eigenvalue": largest,
        "condition_number": float(conds[j]),
        "condition_target": SIGMA_C_COND_TARGET,
        "ridge_grid": SIGMA_C_LAMBDA_GRID.tolist(),
        "condition_by_grid": conds.tolist(),
        "calibration_n": n,
        "dimension": dim,
        "rank_upper_bound": n,
    }


def correlations(cosine: np.ndarray, names: list[str], leakage: dict) -> dict:
    """Correlate six unique persona means; retain omit-self as a sensitivity."""
    personas = ["default", "sarcasm", "sarcasm_lists", "french", "french_lists", "sfl"]
    target = names.index("sfl")
    result = {}
    for label, selected in (("all_six", personas), ("excluding_self", personas[:-1])):
        x = np.array([cosine[names.index(p), target] for p in selected])
        y = np.array([leakage[p]["sfl"] for p in selected])
        result[label] = {
            "n": len(selected),
            "personas": selected,
            "cosine_to_sfl": x.tolist(),
            "sfl_tracer_uptake": y.tolist(),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "spearman_rho": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
        }
    return result


def stage_chunk(k: int, cfg: DictConfig, manifest: dict, inventory: dict) -> Path:
    """Download, verify and retain four layers; resume only exact-source chunks."""
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"selected_{k:04d}.npz"
    source_name = f"chunks/batch_{k:04d}.pt"
    expected_sha = inventory["files"][source_name]["sha256"]
    indices = manifest["spec"]["batches"][k]
    layers = list(cfg.layers)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["source_sha256"]) != expected_sha
                or saved["indices"].tolist() != indices
                or saved["layers"].tolist() != layers
                or str(saved["fingerprint"]) != manifest["fingerprint"]
                or str(saved["vectors_sha256"]) != sha256(saved["vectors"].tobytes())
            ):
                raise ValueError(f"stale or corrupted staged chunk: {path}")
        return path
    url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{cfg.hf_revision}/{PREFIX}/{source_name}"
    with urlopen(url, timeout=60) as response:
        data = response.read()
    if sha256(data) != expected_sha:
        raise ValueError(f"source hash mismatch: {source_name}")
    chunk = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if chunk["fingerprint"] != manifest["fingerprint"] or chunk["indices"] != indices:
        raise ValueError(f"source fingerprint/row mismatch: {source_name}")
    expected_shape = (len(indices), 64, 5120)
    if tuple(chunk["vectors"].shape) != expected_shape or chunk["vectors"].dtype != torch.bfloat16:
        raise ValueError(f"source shape/dtype mismatch: {source_name}")
    vectors = chunk["vectors"][:, layers, :].float().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError(f"non-finite source: {source_name}")
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        vectors=vectors,
        indices=indices,
        layers=layers,
        source_sha256=expected_sha,
        fingerprint=manifest["fingerprint"],
        vectors_sha256=sha256(vectors.tobytes()),
    )
    tmp.replace(path)
    return path


@hydra.main(
    version_base=None,
    config_path="../configs/pilots",
    config_name="story_persona_metric_reanalysis",
)
def main(cfg: DictConfig) -> None:
    """Stage a bounded four-layer view and persist each requested metric block."""
    started = time.time()
    source = Path(cfg.source_dir)
    out = ROOT / cfg.output_dir
    manifest = json.loads((source / "manifest.json").read_text())
    inventory = json.loads((source / "independent_artifact_manifest.json").read_text())
    summary_bytes = (ROOT / "eval_results/issue_2673/summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    if (
        cfg.hf_revision != inventory["verified_revision"]
        or DATA_REPO != inventory["hf_repo"]
        or PREFIX != inventory["hf_prefix"]
        or inventory["fingerprint"] != manifest["fingerprint"]
        or summary["fingerprint"] != manifest["fingerprint"]
    ):
        raise ValueError("source revision/repository/fingerprint is not the verified capture")
    published_path = ROOT / "eval_results/issue_2673/published_leakage_comparison.json"
    published = json.loads(published_path.read_text())
    if sha256(summary_bytes) != published["sources"]["cosine_sha256"]:
        raise ValueError("Qwen summary changed since the reviewed behavioral pairing")
    for filename in ("manifest.json", "rows.json", "centroids.npz"):
        if sha256((source / filename).read_bytes()) != inventory["files"][filename]["sha256"]:
            raise ValueError(f"input differs from independently verified archive: {filename}")
    rows = json.loads((source / "rows.json").read_text())
    names = summary["persona_names"]
    provenance = {
        "source_repo": DATA_REPO,
        "source_prefix": PREFIX,
        "source_revision": cfg.hf_revision,
        "capture_fingerprint": manifest["fingerprint"],
        "summary_sha256": sha256(summary_bytes),
        "published_rates_sha256": sha256(published_path.read_bytes()),
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "centering": "none; uncentered second moment for whitening",
        "whitening_fit": "all 2400 individual contexts, same-battery calibration",
        "recipe_reference": "analysis/leakage_predictor.py Sigma_c; #665/#666 grid and condition target",
    }
    fingerprint = sha256(json.dumps(provenance, sort_keys=True).encode())
    # Raw all-layer results require only the already-verified summary.
    raw = {
        str(layer): correlations(np.array(matrix), names, published["leakage_rates"])
        for layer, matrix in enumerate(summary["raw_cosine_diagnostic"])
    }
    write_json(out / "raw_all_layers.json", {"provenance": provenance, "results": raw})
    print(
        f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] raw cosine: 64 blocks saved",
        flush=True,
    )
    batches = manifest["spec"]["batches"]
    paths = {}
    with ThreadPoolExecutor(max_workers=int(cfg.workers)) as pool:
        pending = {
            pool.submit(stage_chunk, k, cfg, manifest, inventory): k for k in range(len(batches))
        }
        for completed, future in enumerate(as_completed(pending), 1):
            k = pending[future]
            paths[k] = future.result()
            print(
                f"[stage] unit {completed}/{len(batches)} batch={k:04d} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    x = np.empty((len(rows), len(cfg.layers), 5120), dtype=np.float64)
    seen = np.zeros(len(rows), dtype=int)
    for k in range(len(batches)):
        with np.load(paths[k], allow_pickle=False) as chunk:
            indices = chunk["indices"]
            x[indices] = chunk["vectors"]
            seen[indices] += 1
    if not np.all(seen == 1) or len(rows) != 2400:
        raise ValueError("incomplete or duplicate calibration rows")
    with np.load(source / "centroids.npz") as saved:
        centroids = saved["centroids"][:, list(cfg.layers)]
    persona_indices = [
        [i for i, row in enumerate(rows) if row["persona"] == name] for name in names
    ]
    reconstructed = np.stack([x[indices].mean(0) for indices in persona_indices])
    if not np.array_equal(reconstructed, centroids):
        raise ValueError("staged individual vectors do not reproduce saved centroids exactly")
    write_json(
        Path(cfg.cache_dir) / "staging_complete.json",
        {
            "fingerprint": fingerprint,
            "row_count": len(rows),
            "chunk_count": len(paths),
            "layers": list(cfg.layers),
            "centroid_reconstruction_max_error": 0,
            "source_chunk_sha256": {
                key: val["sha256"]
                for key, val in inventory["files"].items()
                if key.startswith("chunks/")
            },
        },
    )
    results = {}
    for j, layer in enumerate(cfg.layers):
        block_path = out / f"block_{layer}.json"
        # Four cheap fits are always recomputed; resume only verified input staging.
        block_start = time.time()
        calibration = np.ascontiguousarray(x[:, j])
        means = centroids[:, j]
        ridge, diagnostics = select_ridge(calibration)
        gram, residual = metric_gram(calibration, means, ridge)
        white = cosine_from_gram(gram)
        ordinary = cosine_from_gram(means @ means.T)
        np.testing.assert_allclose(
            ordinary, summary["raw_cosine_diagnostic"][layer], atol=1e-12, rtol=0
        )
        result = {
            "fingerprint": fingerprint,
            "layer": layer,
            "persona_names": names,
            "ridge": ridge,
            "diagnostics": diagnostics,
            "implicit_primal_relative_residual": residual,
            "raw_cosine_matrix": ordinary.tolist(),
            "whitened_cosine_matrix": white.tolist(),
            "raw": correlations(ordinary, names, published["leakage_rates"]),
            "whitened": correlations(white, names, published["leakage_rates"]),
            "elapsed_seconds": time.time() - block_start,
        }
        write_json(block_path, result)
        results[str(layer)] = result
        print(
            f"[fit] block={layer} ridge={result['ridge']:.6g} elapsed={time.time() - started:.1f}s r={result['whitened']['all_six']['pearson_r']:.6f}",
            flush=True,
        )
    write_json(
        out / "summary.json",
        {
            "provenance": provenance,
            "fingerprint": fingerprint,
            "results": results,
            "limitations": [
                "Qwen geometry versus published Kimi behavior",
                "six condition means; no inferential test",
                "whitening calibrated on the same limited battery, rank deficient before regularization",
                "no mean subtraction in either metric",
                "self endpoint included in primary pairing",
                "whitening reported at four fixed blocks only",
            ],
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"[complete] {out / 'summary.json'} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
