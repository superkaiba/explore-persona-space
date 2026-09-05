#!/usr/bin/env python3
"""Issue #2569 leg 12: direct corpus extrema for the named L19 directions.

The operator analyses identify directions in residual space, but most of those
directions were interpreted only through dictionary cosines.  This pass projects
the fixed, deduplicated 100k context/answer capture directly onto the directions
and records the highest/lowest real examples.  One-dimensional directions use
signed centered projections; collapsed complex eigenmodes use the invariant
two-plane projection norm.

The pass covers the top-32 singular read/write directions, top-32 collapsed
eigen read/write modes, the six context-side persona directions used by leg 8,
the mean minimal-pair refusal context shift, and all 24 answer directions used
by leg 10.  Projection is descriptive: singular/PC/eigen line signs are arbitrary,
and large projection is not a causal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2569_eigen_dashboards_v2 as ED  # noqa: E402
import issue2569_kernel_interpretation as KI  # noqa: E402
import issue2569_operator as OP  # noqa: E402
import issue2569_refusal_kernel as RK  # noqa: E402
import issue2569_variance_decomposition as VD  # noqa: E402

LOGGER = logging.getLogger("issue2569.direction_examples")
LAYER = 19
D = 3584
N_OPERATOR_DIRECTIONS = 32
TOP_K = 5
CANDIDATE_K = 4096
BLOCK = 4096
THEORY = Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory")
DEFAULT_SAMPLE = THEORY / "leg10_dl/sample_L19.npz"
DEFAULT_MANIFEST = THEORY / "leg10_dl/download_manifest.json"
DEFAULT_LEG9_MANIFEST = THEORY / "leg9_dl/leg9_manifest.json"
DEFAULT_MOMENTS = THEORY / "moments"
DEFAULT_WORK = THEORY / "leg12_dl"
OUTPUT_RELPATH = Path("eval_results/issue_2569/weights/leg12")
RESULT_NAME = "direction_examples_L19"
TRAITS = ("evil", "sycophancy", "hallucination")


def write_text_atomic(path: Path, content: str) -> None:
    """Write text through a sibling temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash an array's shape, dtype, and C-order bytes."""
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def npz_member_memmap(path: Path, member: str) -> np.memmap:
    """Memory-map an uncompressed NPY member inside an NPZ archive.

    ``sample_L19.npz`` was written by ``np.savez`` (ZIP_STORED), so mapping the
    member in place avoids materializing either 1.43 GB state matrix.  The local
    ZIP header is parsed only to locate the embedded NPY header; NumPy parses the
    dtype/shape/order metadata itself.
    """
    archive_path = Path(path)
    member_name = member if member.endswith(".npy") else f"{member}.npy"
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{archive_path}:{member_name} is compressed; cannot memory-map")
        local_offset = info.header_offset
    with archive_path.open("rb") as handle:
        handle.seek(local_offset)
        header = handle.read(30)
        if len(header) != 30:
            raise RuntimeError(f"short ZIP local header for {archive_path}:{member_name}")
        signature, *_prefix, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise RuntimeError(f"bad ZIP local signature for {archive_path}:{member_name}")
        handle.seek(name_len + extra_len, 1)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran_order, dtype = np.lib.format._read_array_header(  # noqa: SLF001
                handle, version
            )
        data_offset = handle.tell()
    order = "F" if fortran_order else "C"
    return np.memmap(
        archive_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        order=order,
        offset=data_offset,
    )


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a unit fp64 copy of a nonzero vector."""
    out = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(out))
    if not norm > 0:
        raise ValueError("cannot normalize zero direction")
    return out / norm


def _lookup_manifest_path(manifest: dict, suffix: str) -> Path:
    hits = [Path(value) for key, value in manifest["paths"].items() if key.endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"manifest suffix {suffix!r}: expected one hit, found {len(hits)}")
    if not hits[0].is_file():
        raise FileNotFoundError(hits[0])
    return hits[0]


def load_population_covariance(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load an unbiased population covariance, mean, and row count."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gram = np.asarray(payload["gram"], dtype=np.float64)
    mean_key = "mean" if "mean" in payload else "mean_y"
    mean = np.asarray(payload[mean_key], dtype=np.float64)
    n_rows = int(payload["n_rows"])
    covariance = (gram - n_rows * np.outer(mean, mean)) / (n_rows - 1)
    covariance = 0.5 * (covariance + covariance.T)
    if covariance.shape != (D, D) or mean.shape != (D,):
        raise RuntimeError((covariance.shape, mean.shape))
    return covariance, mean, n_rows


def sample_covariances(
    x: np.ndarray,
    y: np.ndarray,
    payload: OP.MapPayload,
    block: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce leg 10's sample answer and residual covariances, blockwise."""
    n_rows = x.shape[0]
    residual = np.empty(y.shape, dtype=np.float32)
    for lower in range(0, n_rows, block):
        upper = min(lower + block, n_rows)
        residual[lower:upper] = (
            np.asarray(y[lower:upper], dtype=np.float64)
            - OP.predict(payload, np.asarray(x[lower:upper], dtype=np.float64))
        ).astype(np.float32)
        if lower == 0 or upper == n_rows or (lower // block + 1) % 8 == 0:
            LOGGER.info("[residual] rows=%d/%d", upper, n_rows)
    sigma_y = VD.covariance_matrix(y, row_block=min(block, 2048))
    sigma_residual = VD.covariance_matrix(residual, row_block=min(block, 2048))
    return sigma_y, sigma_residual


def build_directions(
    *,
    repo_root: Path,
    map_root: Path,
    manifest: dict,
    leg9_manifest: dict,
    moments: Path,
    x: np.ndarray,
    y: np.ndarray,
    block: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    """Reconstruct every direction with parity checks against legs 1, 8, 9, and 10."""
    payload = OP.load_banked_map(LAYER, root=map_root)
    operator, _bias = OP.row_operator(payload)
    LOGGER.info("[directions] full operator SVD")
    u_svd, singular_values, vh_svd = sla.svd(operator, lapack_driver="gesdd")
    LOGGER.info("[directions] full operator eigendecomposition")
    eigenvalues, right_eigenvectors = sla.eig(operator)
    left_eigen_rows = np.linalg.inv(right_eigenvectors)
    tolerance = 1e-12 * max(float(np.abs(eigenvalues).max()), 1.0)
    eigen_entries = ED.collapse_eigen_directions(
        eigenvalues,
        right_eigenvectors,
        left_eigen_rows,
        n_want=N_OPERATOR_DIRECTIONS,
        tol_imag=tolerance,
    )
    basis_dims = np.asarray([entry["read_basis"].shape[0] for entry in eigen_entries])
    max_basis = int(basis_dims.max())
    eigen_read = np.zeros((N_OPERATOR_DIRECTIONS, max_basis, D), dtype=np.float64)
    eigen_write = np.zeros_like(eigen_read)
    eigen_metadata = []
    for index, entry in enumerate(eigen_entries):
        dim = int(basis_dims[index])
        eigen_read[index, :dim] = entry["read_basis"]
        eigen_write[index, :dim] = entry["write_basis"]
        eigen_metadata.append(
            {
                "rank": index + 1,
                "kind": entry["kind"],
                "basis_dim": dim,
                "abs_lambda": float(entry["abs_lambda"]),
                "lambda_real": float(entry["lam_re"]),
                "lambda_imag": float(entry["lam_im"]),
                "eigenvalue_ranks": entry["ranks"],
            }
        )

    dashboard = json.loads(
        (repo_root / "eval_results/issue_2569/weights/leg1/sae_dashboards_v2_L19.json").read_text()
    )
    expected_sigma = np.asarray(
        [row["sigma"] for row in dashboard["sections"]["singular_read"]["directions"]]
    )
    expected_lambda = np.asarray(
        [row["abs_lambda"] for row in dashboard["sections"]["eigen_read"]["directions"]]
    )
    if not np.allclose(singular_values[:N_OPERATOR_DIRECTIONS], expected_sigma, rtol=1e-10):
        raise RuntimeError("singular values do not reproduce the leg-1 dashboard")
    if not np.allclose(
        [entry["abs_lambda"] for entry in eigen_entries], expected_lambda, rtol=1e-10
    ):
        raise RuntimeError("eigenvalue ordering does not reproduce the leg-1 dashboard")
    del right_eigenvectors, left_eigen_rows

    sigma_population_y, mean_y, population_rows = load_population_covariance(
        moments / "gram_yy.pt"
    )
    _sigma_population_x, mean_x, population_x_rows = load_population_covariance(
        moments / "gram_xx.pt"
    )
    if population_rows != population_x_rows:
        raise RuntimeError((population_x_rows, population_rows))
    pc_values, pc_vectors = np.linalg.eigh(sigma_population_y)
    top_pc = [unit(pc_vectors[:, -1 - index]) for index in range(5)]
    bottom_pc = [unit(pc_vectors[:, index]) for index in range(5)]

    LOGGER.info("[directions] sample covariance for the 10 worst-R2 directions")
    sigma_sample_y, sigma_residual = sample_covariances(x, y, payload, block)
    shrink = sigma_sample_y + 1e-3 * (np.trace(sigma_sample_y) / D) * np.eye(D)
    _generalized_values, generalized_vectors = sla.eigh(
        sigma_residual, shrink, subset_by_index=[D - 10, D - 1]
    )
    worst = [unit(generalized_vectors[:, -1 - index]) for index in range(10)]

    svmp = RK.load_svmp(leg9_manifest)
    layer_index = svmp["layers"].index(LAYER)
    pairs = svmp["pairs"]
    high = np.asarray([pair["hi"] for pair in pairs])
    low = np.asarray([pair["lo"] for pair in pairs])
    is_flip = np.asarray(
        [(pair["group"] == "flip") and not pair["is_control_cell"] for pair in pairs]
    )
    answer_delta = svmp["va"][high, layer_index] - svmp["va"][low, layer_index]
    context_delta = svmp["vc"][high, layer_index] - svmp["vc"][low, layer_index]
    refusal_answer = unit(answer_delta[is_flip].mean(axis=0))
    refusal_context = unit(context_delta[is_flip].mean(axis=0))
    del svmp

    rb_directions: dict[str, np.ndarray] = {}
    ctxext_directions: dict[str, np.ndarray] = {}
    for trait in TRAITS:
        rb_payload = torch.load(
            _lookup_manifest_path(manifest, f"r_b/{trait}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        rb_directions[trait] = unit(np.asarray(rb_payload["r_b"][LAYER]))
        ctxext_payload = torch.load(
            _lookup_manifest_path(manifest, f"{trait}_ctxext_L19.pt"),
            map_location="cpu",
            weights_only=False,
        )
        ctxext_directions[trait] = unit(np.asarray(ctxext_payload["direction"]))

    answer_names = (
        ["refusal_axis_2617"]
        + [f"r_B_{trait}" for trait in TRAITS]
        + [f"answer_PC{index + 1}" for index in range(5)]
        + [f"answer_PC_bottom{index + 1}" for index in range(5)]
        + [f"worst_R2_dir{index + 1}" for index in range(10)]
    )
    answer_behavior = np.stack(
        [refusal_answer]
        + [rb_directions[trait] for trait in TRAITS]
        + top_pc
        + bottom_pc
        + worst
    )
    context_names = (
        [f"r_B_{trait}" for trait in TRAITS]
        + [f"ctxext_{trait}" for trait in TRAITS]
        + ["mean_refusal_flip_context"]
    )
    context_behavior = np.stack(
        [rb_directions[trait] for trait in TRAITS]
        + [ctxext_directions[trait] for trait in TRAITS]
        + [refusal_context]
    )

    leg10 = json.loads(
        (repo_root / "eval_results/issue_2569/weights/leg10/variance_decomposition_L19.json").read_text()
    )
    expected_rows = leg10["per_direction"]
    if [row["direction"] for row in expected_rows] != answer_names:
        raise RuntimeError("leg-10 direction order changed")
    var_y = np.einsum("nd,de,ne->n", answer_behavior, sigma_sample_y, answer_behavior)
    var_residual = np.einsum(
        "nd,de,ne->n", answer_behavior, sigma_residual, answer_behavior
    )
    reproduced_r2 = 1.0 - var_residual / var_y
    expected_r2 = np.asarray([row["L"] for row in expected_rows])
    expected_var = np.asarray([row["var_u_sample_abs"] for row in expected_rows])
    if not np.allclose(reproduced_r2, expected_r2, rtol=2e-7, atol=2e-7):
        raise RuntimeError(
            f"leg-10 R2 parity failed: max abs={np.max(np.abs(reproduced_r2 - expected_r2))}"
        )
    if not np.allclose(var_y, expected_var, rtol=2e-7, atol=2e-7):
        raise RuntimeError(
            f"leg-10 variance parity failed: max rel={np.max(np.abs(var_y / expected_var - 1))}"
        )

    arrays = {
        "singular_read": u_svd[:, :N_OPERATOR_DIRECTIONS].T,
        "singular_write": vh_svd[:N_OPERATOR_DIRECTIONS],
        "eigen_read": eigen_read,
        "eigen_write": eigen_write,
        "eigen_basis_dims": basis_dims,
        "context_behavior": context_behavior,
        "answer_behavior": answer_behavior,
    }
    metadata: dict[str, Any] = {
        "singular": [
            {"rank": index + 1, "sigma": float(singular_values[index])}
            for index in range(N_OPERATOR_DIRECTIONS)
        ],
        "eigen": eigen_metadata,
        "context_behavior_names": context_names,
        "answer_behavior_names": answer_names,
        "answer_behavior_leg10_r2": reproduced_r2.tolist(),
        "parity": {
            "leg1_singular_max_abs_error": float(
                np.max(np.abs(singular_values[:N_OPERATOR_DIRECTIONS] - expected_sigma))
            ),
            "leg1_eigen_abs_lambda_max_abs_error": float(
                np.max(np.abs(expected_lambda - [entry["abs_lambda"] for entry in eigen_entries]))
            ),
            "leg10_r2_max_abs_error": float(np.max(np.abs(reproduced_r2 - expected_r2))),
            "leg10_variance_max_relative_error": float(
                np.max(np.abs(var_y / expected_var - 1))
            ),
        },
        "population_moment_rows": population_rows,
        "map_path": str(payload.path),
        "selected_lambda": payload.selected_lambda,
        "answer_pc_eigenvalues": {
            "top5": [float(pc_values[-1 - index]) for index in range(5)],
            "bottom5": [float(pc_values[index]) for index in range(5)],
        },
    }
    return arrays, metadata, mean_x, mean_y


def save_direction_checkpoint(
    work: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    mean_x: np.ndarray,
    mean_y: np.ndarray,
    ci_hash: str,
) -> None:
    """Persist reconstructed directions and their validation record."""
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "directions_L19.npz"
    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp.npz")
    np.savez(
        temporary,
        **arrays,
        mean_x=mean_x,
        mean_y=mean_y,
    )
    temporary.replace(checkpoint)
    metadata = {
        **metadata,
        "sample_ci_sha256": ci_hash,
        "direction_fingerprints": {
            key: array_sha256(value) for key, value in arrays.items()
        },
    }
    write_text_atomic(work / "directions_L19.json", json.dumps(metadata, indent=1) + "\n")


def load_direction_checkpoint(
    work: Path, ci_hash: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray] | None:
    """Load and validate the direction checkpoint, or return ``None``."""
    npz_path = work / "directions_L19.npz"
    json_path = work / "directions_L19.json"
    if not (npz_path.is_file() and json_path.is_file()):
        return None
    metadata = json.loads(json_path.read_text())
    if metadata.get("sample_ci_sha256") != ci_hash:
        raise RuntimeError("direction checkpoint belongs to a different sample")
    with np.load(npz_path) as stored:
        mean_x = stored["mean_x"]
        mean_y = stored["mean_y"]
        arrays = {
            key: stored[key]
            for key in (
                "singular_read",
                "singular_write",
                "eigen_read",
                "eigen_write",
                "eigen_basis_dims",
                "context_behavior",
                "answer_behavior",
            )
        }
    for key, value in arrays.items():
        expected = metadata["direction_fingerprints"][key]
        if array_sha256(value) != expected:
            raise RuntimeError(f"direction checkpoint fingerprint mismatch: {key}")
    LOGGER.info("[checkpoint-validated] %s", npz_path)
    return arrays, metadata, mean_x, mean_y


def ordered_extreme_indices(scores: np.ndarray, count: int, largest: bool) -> np.ndarray:
    """Return exact deterministic extreme indices, breaking score ties by row."""
    values = np.asarray(scores)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector")
    count = min(int(count), values.size)
    if count < 1:
        return np.empty(0, dtype=np.int64)
    pivot = values.size - count if largest else count - 1
    threshold = np.partition(values, pivot)[pivot]
    eligible = np.flatnonzero(values >= threshold if largest else values <= threshold)
    primary = -values[eligible] if largest else values[eligible]
    order = np.lexsort((eligible, primary))
    return eligible[order[:count]].astype(np.int64)


def _candidate_record(
    indices: np.ndarray,
    scores: np.ndarray,
    score_scale: float,
    coordinates: np.ndarray | None = None,
) -> list[dict]:
    records = []
    for index in indices:
        record: dict[str, Any] = {
            "sample_row": int(index),
            "score": float(scores[index]),
            "score_in_sample_scale": float(scores[index] / score_scale),
        }
        if coordinates is not None:
            record["plane_coordinates"] = [float(value) for value in coordinates[index]]
        records.append(record)
    return records


def line_candidates(
    states: np.ndarray,
    mean: np.ndarray,
    directions: np.ndarray,
    names: list[str],
    candidate_k: int,
    block: int,
) -> list[dict]:
    """Project one line family and retain exact high/low candidate rankings."""
    n_rows, n_directions = states.shape[0], directions.shape[0]
    scores = np.empty((n_rows, n_directions), dtype=np.float32)
    directions32 = np.asarray(directions, dtype=np.float32)
    mean32 = np.asarray(mean, dtype=np.float32)
    for lower in range(0, n_rows, block):
        upper = min(lower + block, n_rows)
        centered = np.asarray(states[lower:upper], dtype=np.float32) - mean32
        scores[lower:upper] = centered @ directions32.T
    rows = []
    for index, name in enumerate(names):
        values = scores[:, index]
        scale = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        if not scale > 0:
            raise RuntimeError(f"zero projection scale: {name}")
        high = ordered_extreme_indices(values, candidate_k, largest=True)
        low = ordered_extreme_indices(values, candidate_k, largest=False)
        rows.append(
            {
                "name": name,
                "kind": "line",
                "projection_mean": float(values.mean(dtype=np.float64)),
                "projection_sd": float(values.std(dtype=np.float64, ddof=1)),
                "projection_rms_about_population_mean": scale,
                "high_candidates": _candidate_record(high, values, scale),
                "low_candidates": _candidate_record(low, values, scale),
            }
        )
    return rows


def plane_candidates(
    states: np.ndarray,
    mean: np.ndarray,
    bases: np.ndarray,
    basis_dims: np.ndarray,
    names: list[str],
    candidate_k: int,
    block: int,
) -> list[dict]:
    """Project invariant plane/line modes and retain maximum-norm candidates."""
    n_rows, state_dim = states.shape
    n_modes, max_basis, basis_dim = bases.shape
    if basis_dim != state_dim:
        raise ValueError((bases.shape, states.shape))
    coordinates = np.empty((n_rows, n_modes, max_basis), dtype=np.float32)
    flat_basis = np.asarray(
        bases.reshape(n_modes * max_basis, state_dim), dtype=np.float32
    )
    mean32 = np.asarray(mean, dtype=np.float32)
    for lower in range(0, n_rows, block):
        upper = min(lower + block, n_rows)
        centered = np.asarray(states[lower:upper], dtype=np.float32) - mean32
        coordinates[lower:upper] = (centered @ flat_basis.T).reshape(
            upper - lower, n_modes, max_basis
        )
    rows = []
    for index, name in enumerate(names):
        dim = int(basis_dims[index])
        coords = coordinates[:, index, :dim]
        if dim == 1:
            values = coords[:, 0]
            scale = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
            high = ordered_extreme_indices(values, candidate_k, largest=True)
            low = ordered_extreme_indices(values, candidate_k, largest=False)
            rows.append(
                {
                    "name": name,
                    "kind": "line",
                    "basis_dim": dim,
                    "projection_mean": float(values.mean(dtype=np.float64)),
                    "projection_sd": float(values.std(dtype=np.float64, ddof=1)),
                    "projection_rms_about_population_mean": scale,
                    "high_candidates": _candidate_record(high, values, scale, coords),
                    "low_candidates": _candidate_record(low, values, scale, coords),
                }
            )
            continue
        norms = np.linalg.norm(coords, axis=1)
        scale = float(np.sqrt(np.mean(norms.astype(np.float64) ** 2)))
        high = ordered_extreme_indices(norms, candidate_k, largest=True)
        rows.append(
            {
                "name": name,
                "kind": "plane",
                "basis_dim": dim,
                "projection_norm_mean": float(norms.mean(dtype=np.float64)),
                "projection_norm_rms_about_population_mean": scale,
                "high_candidates": _candidate_record(high, norms, scale, coords),
            }
        )
    return rows


def write_candidate_phase(path: Path, rows: list[dict], metadata: dict[str, Any]) -> None:
    """Persist one independently resumable projection family."""
    write_text_atomic(path, json.dumps({"metadata": metadata, "directions": rows}, indent=1) + "\n")
    LOGGER.info("[phase-out] %s", path)


def load_or_project_phase(
    *,
    path: Path,
    family: str,
    states: np.ndarray,
    mean: np.ndarray,
    directions: np.ndarray,
    names: list[str],
    candidate_k: int,
    block: int,
    ci_hash: str,
    basis_dims: np.ndarray | None = None,
) -> list[dict]:
    """Validate/reuse one projection phase or compute it."""
    direction_hash = array_sha256(directions)
    expected = {
        "family": family,
        "n_rows": int(states.shape[0]),
        "candidate_k": int(candidate_k),
        "sample_ci_sha256": ci_hash,
        "direction_sha256": direction_hash,
    }
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("metadata") != expected:
            raise RuntimeError(f"stale projection checkpoint: {path}")
        LOGGER.info("[checkpoint-validated] %s", path)
        return cached["directions"]
    if basis_dims is None:
        rows = line_candidates(states, mean, directions, names, candidate_k, block)
    else:
        rows = plane_candidates(
            states, mean, directions, basis_dims, names, candidate_k, block
        )
    write_candidate_phase(path, rows, expected)
    return rows


def candidate_cis(families: list[dict], ci: np.ndarray) -> set[int]:
    """Collect conversation ids referenced by candidate rankings."""
    needed: set[int] = set()
    for family in families:
        for direction in family["directions"]:
            for key in ("high_candidates", "low_candidates"):
                for row in direction.get(key, []):
                    needed.add(int(ci[int(row["sample_row"])]))
    return needed


def load_row_texts(manifest: dict, needed: set[int]) -> dict[int, dict]:
    """Load prompt and answer text for selected conversation ids."""
    output: dict[int, dict] = {}
    shards = sorted(
        Path(value) for key, value in manifest["paths"].items() if "/row_meta_" in key
    )
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                ci = int(row["ci"])
                if ci in needed:
                    output[ci] = {
                        "corpus": row.get("corpus"),
                        "context_text": str(row.get("context_text", "")),
                        "answer_text": str(row.get("answer_text", "")),
                    }
        if len(output) == len(needed):
            break
    missing = needed - output.keys()
    if missing:
        raise RuntimeError(f"row metadata missing {len(missing)} selected cis: {sorted(missing)[:5]}")
    return output


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for transparent exact-text deduplication."""
    return " ".join(text.split()).casefold()


def quote_text(text: str, length: int, tail: bool) -> str:
    """Flatten, redact, and truncate a prompt/answer excerpt."""
    flattened = " ".join(str(text).split())
    if len(flattened) > length:
        flattened = flattened[-length:] if tail else flattened[:length]
        flattened = ("…" + flattened) if tail else (flattened + "…")
    return KI.redact(flattened)


def enrich_candidate(
    candidate: dict,
    ci: np.ndarray,
    texts: dict[int, dict],
) -> dict:
    """Attach stable ids, corpus, redacted excerpts, and duplicate hashes."""
    sample_row = int(candidate["sample_row"])
    conversation_id = int(ci[sample_row])
    metadata = texts[conversation_id]
    prompt_norm = normalize_text(metadata["context_text"])
    answer_norm = normalize_text(metadata["answer_text"])
    return {
        **candidate,
        "ci": conversation_id,
        "corpus": metadata["corpus"],
        "prompt_text_sha256": hashlib.sha256(prompt_norm.encode()).hexdigest(),
        "answer_text_sha256": hashlib.sha256(answer_norm.encode()).hexdigest(),
        "prompt_quote": quote_text(metadata["context_text"], 360, tail=True),
        "answer_quote": quote_text(metadata["answer_text"], 420, tail=False),
    }


def unique_candidates(
    candidates: list[dict],
    *,
    side: str,
    top_k: int,
    ci: np.ndarray,
    texts: dict[int, dict],
) -> list[dict]:
    """Take the first ``top_k`` candidates with distinct ranking-side text."""
    text_key = "context_text" if side == "context" else "answer_text"
    seen: set[str] = set()
    output = []
    for candidate in candidates:
        conversation_id = int(ci[int(candidate["sample_row"])])
        normalized = normalize_text(texts[conversation_id][text_key])
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(enrich_candidate(candidate, ci, texts))
        if len(output) == top_k:
            return output
    raise RuntimeError(
        f"only {len(output)} unique {side} texts among {len(candidates)} candidates; "
        "increase --candidate-k"
    )


def finalize_direction(
    row: dict,
    *,
    side: str,
    top_k: int,
    ci: np.ndarray,
    texts: dict[int, dict],
) -> dict:
    """Attach raw and exact-text-deduplicated extrema to one direction."""
    output = {key: value for key, value in row.items() if not key.endswith("_candidates")}
    high_candidates = row["high_candidates"]
    raw_high = [enrich_candidate(item, ci, texts) for item in high_candidates[:top_k]]
    unique_high = unique_candidates(
        high_candidates, side=side, top_k=top_k, ci=ci, texts=texts
    )
    output["raw_high"] = raw_high
    output["unique_high"] = unique_high
    diagnostics: dict[str, Any] = {
        "raw_high_unique_ranking_texts": len(
            {
                item["prompt_text_sha256" if side == "context" else "answer_text_sha256"]
                for item in raw_high
            }
        ),
        "candidate_pool_size": len(high_candidates),
    }
    if row["kind"] == "line":
        low_candidates = row["low_candidates"]
        raw_low = [enrich_candidate(item, ci, texts) for item in low_candidates[:top_k]]
        unique_low = unique_candidates(
            low_candidates, side=side, top_k=top_k, ci=ci, texts=texts
        )
        output["raw_low"] = raw_low
        output["unique_low"] = unique_low
        diagnostics["raw_low_unique_ranking_texts"] = len(
            {
                item["prompt_text_sha256" if side == "context" else "answer_text_sha256"]
                for item in raw_low
            }
        )
    output["duplicate_diagnostics"] = diagnostics
    return output


def markdown_escape(text: str) -> str:
    """Keep generated table excerpts on one Markdown row."""
    return text.replace("|", "\\|").replace("\n", " ")


def short_quote(text: str, length: int = 180) -> str:
    """Shorten an already-redacted quote for the compact Markdown index."""
    return text if len(text) <= length else text[: length - 1] + "…"


def render_markdown(document: dict) -> str:
    """Render a browsable compact index; JSON retains all five extrema."""
    coverage = document["coverage"]
    lines = [
        "# Direct activating examples for issue #2569 L19 directions",
        "",
        "This pass projects the fixed 100,000-row paired capture directly onto every named",
        "direction that previously lacked corpus extrema. One-dimensional entries use",
        "`(state - population_mean) · direction`; eigen entries use the Euclidean norm of",
        "the projection into the collapsed real invariant 2-plane. The JSON companion stores",
        "five raw and five exact-text-deduplicated extrema per tail (plane modes have a high",
        "tail only). This document shows the strongest deduplicated example for each entry.",
        "",
        "Signs for SVD, PCA, generalized-eigenvector, and real-eigenvector lines are arbitrary;",
        "an equivalent factorization may swap high and low. Plane norms are sign- and",
        "basis-rotation invariant. These are descriptive examples, not causal effects.",
        "",
        "## Coverage",
        "",
        "| family | side | directions | lines | planes |",
        "|---|---|---:|---:|---:|",
    ]
    for row in coverage["families"]:
        lines.append(
            f"| {row['family']} | {row['side']} | {row['n_directions']} | "
            f"{row['n_lines']} | {row['n_planes']} |"
        )
    lines += [
        "",
        f"Total: **{coverage['n_directions']} directions/modes** over "
        f"**{coverage['n_rows']:,} paired rows**. Every selected row has both prompt and answer text.",
        "",
    ]
    for family in document["families"]:
        lines += [
            f"## {family['label']}",
            "",
            f"Ranking side: **{family['side']}**. {family['definition']}",
            "",
            "| direction | kind | high / max example | low example (lines only) |",
            "|---|---|---|---|",
        ]
        quote_key = "prompt_quote" if family["side"] == "context" else "answer_quote"
        for direction in family["directions"]:
            high = direction["unique_high"][0]
            high_text = markdown_escape(short_quote(high[quote_key]))
            high_cell = f"`ci={high['ci']}` ({high['score_in_sample_scale']:+.2f} scale): {high_text}"
            if direction["kind"] == "line":
                low = direction["unique_low"][0]
                low_text = markdown_escape(short_quote(low[quote_key]))
                low_cell = f"`ci={low['ci']}` ({low['score_in_sample_scale']:+.2f} scale): {low_text}"
            else:
                low_cell = "—"
            lines.append(
                f"| `{direction['name']}` | {direction['kind']} | {high_cell} | {low_cell} |"
            )
        lines.append("")
    lines += [
        "## Duplicate handling and scope",
        "",
        "`raw_high`/`raw_low` preserve the literal highest rows, including repeated prompts or",
        "answers. `unique_high`/`unique_low` scan the exact ordered candidate tail and retain",
        "the first five distinct normalized texts, so boilerplate repetition is visible but",
        "cannot monopolize the readable examples. Text excerpts are credential-pattern redacted.",
        "",
        "Existing extrema for the leg-8 covariance modes, leg-11 selected context PCs, and",
        "leg-8/11 SAE features were not recomputed; this pass fills the operator-, persona-,",
        "refusal-, answer-PC-, and worst-R2-direction gaps. Results characterize the fitted",
        "linear map and this fixed sample, not the model's causal computation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the complete direction-extrema pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--map-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--leg9-manifest", type=Path, default=DEFAULT_LEG9_MANIFEST)
    parser.add_argument("--moments", type=Path, default=DEFAULT_MOMENTS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--n-rows", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    started = time.time()

    if args.n_rows < args.top_k or args.candidate_k < args.top_k:
        raise ValueError("n-rows and candidate-k must be at least top-k")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("errors"):
        raise RuntimeError(manifest["errors"])
    leg9_manifest = json.loads(args.leg9_manifest.read_text())
    x_all = npz_member_memmap(args.sample, "x")
    y_all = npz_member_memmap(args.sample, "y")
    ci_all = npz_member_memmap(args.sample, "ci")
    if x_all.shape != y_all.shape or x_all.shape[1] != D or ci_all.shape != (x_all.shape[0],):
        raise RuntimeError((x_all.shape, y_all.shape, ci_all.shape))
    if args.n_rows > x_all.shape[0]:
        raise ValueError(f"requested {args.n_rows} rows, sample has {x_all.shape[0]}")
    x, y, ci = x_all[: args.n_rows], y_all[: args.n_rows], ci_all[: args.n_rows]
    if np.unique(ci).size != ci.size:
        raise RuntimeError("sample conversation ids are not unique")
    ci_hash = hashlib.sha256(np.asarray(ci, dtype=np.int64).tobytes()).hexdigest()
    args.work.mkdir(parents=True, exist_ok=True)

    checkpoint = load_direction_checkpoint(args.work, ci_hash)
    if checkpoint is None:
        arrays, direction_metadata, mean_x, mean_y = build_directions(
            repo_root=args.repo_root,
            map_root=args.map_root,
            manifest=manifest,
            leg9_manifest=leg9_manifest,
            moments=args.moments,
            x=x,
            y=y,
            block=args.block,
        )
        save_direction_checkpoint(
            args.work, arrays, direction_metadata, mean_x, mean_y, ci_hash
        )
    else:
        arrays, direction_metadata, mean_x, mean_y = checkpoint

    phases: list[dict[str, Any]] = []
    phase_specs = [
        (
            "singular_read",
            "Singular read directions",
            "context",
            x,
            mean_x,
            arrays["singular_read"],
            [f"singular_read_{index + 1}" for index in range(N_OPERATOR_DIRECTIONS)],
            None,
            "Signed projection on the top left-singular input directions.",
        ),
        (
            "eigen_read",
            "Eigen read modes",
            "context",
            x,
            mean_x,
            arrays["eigen_read"],
            [f"eigen_read_{index + 1}" for index in range(N_OPERATOR_DIRECTIONS)],
            arrays["eigen_basis_dims"],
            "Projection norm for complex invariant read planes; signed projection for real lines.",
        ),
        (
            "context_behavior",
            "Context-side persona and refusal directions",
            "context",
            x,
            mean_x,
            arrays["context_behavior"],
            direction_metadata["context_behavior_names"],
            None,
            "Signed projection on the leg-8 persona axes and leg-9 mean refusal context shift.",
        ),
        (
            "singular_write",
            "Singular write directions",
            "answer",
            y,
            mean_y,
            arrays["singular_write"],
            [f"singular_write_{index + 1}" for index in range(N_OPERATOR_DIRECTIONS)],
            None,
            "Signed projection on the top right-singular output directions.",
        ),
        (
            "eigen_write",
            "Eigen write modes",
            "answer",
            y,
            mean_y,
            arrays["eigen_write"],
            [f"eigen_write_{index + 1}" for index in range(N_OPERATOR_DIRECTIONS)],
            arrays["eigen_basis_dims"],
            "Projection norm for complex invariant write planes; signed projection for real lines.",
        ),
        (
            "answer_behavior",
            "Answer-side refusal, persona, PC, and worst-R2 directions",
            "answer",
            y,
            mean_y,
            arrays["answer_behavior"],
            direction_metadata["answer_behavior_names"],
            None,
            "Signed projection on every direction used by the leg-10 directional decomposition.",
        ),
    ]
    for family, label, side, states, mean, directions, names, basis_dims, definition in phase_specs:
        rows = load_or_project_phase(
            path=args.work / f"candidates_{family}.json",
            family=family,
            states=states,
            mean=mean,
            directions=directions,
            names=names,
            candidate_k=min(args.candidate_k, args.n_rows),
            block=args.block,
            ci_hash=ci_hash,
            basis_dims=basis_dims,
        )
        phases.append(
            {
                "family": family,
                "label": label,
                "side": side,
                "definition": definition,
                "directions": rows,
            }
        )

    needed = candidate_cis(phases, ci)
    LOGGER.info("[texts] loading %d candidate conversation ids", len(needed))
    texts = load_row_texts(manifest, needed)
    finalized_families = []
    coverage_rows = []
    for phase in phases:
        finalized = [
            finalize_direction(
                row,
                side=phase["side"],
                top_k=args.top_k,
                ci=ci,
                texts=texts,
            )
            for row in phase["directions"]
        ]
        finalized_families.append({**phase, "directions": finalized})
        coverage_rows.append(
            {
                "family": phase["family"],
                "side": phase["side"],
                "n_directions": len(finalized),
                "n_lines": sum(row["kind"] == "line" for row in finalized),
                "n_planes": sum(row["kind"] == "plane" for row in finalized),
            }
        )

    n_directions = sum(row["n_directions"] for row in coverage_rows)
    document = {
        "task": "issue2569 leg12 direct direction examples",
        "definitions": {
            "line_score": "(state - side population mean) dot unit direction",
            "plane_score": "Euclidean norm of projection into an orthonormal real invariant eigenplane",
            "score_in_sample_scale": "line score divided by RMS projection about the population mean; plane norm divided by RMS plane norm",
            "raw_extrema": "literal ordered extrema, duplicates retained",
            "unique_extrema": "first extrema with distinct whitespace-normalized, case-folded text on the ranked side",
            "sign_caveat": "SVD, PCA, generalized-eigenvector, and real-eigenvector global signs are arbitrary",
        },
        "coverage": {
            "n_rows": int(args.n_rows),
            "n_directions": n_directions,
            "families": coverage_rows,
            "prior_examples_not_recomputed": [
                "leg8 ignored/range covariance modes",
                "leg8 and leg11 selected SAE features",
                "leg11 selected raw/standardized context PCs",
            ],
        },
        "direction_reconstruction": direction_metadata,
        "families": finalized_families,
        "repro": {
            "sample": str(args.sample),
            "sample_sha256": sha256_file(args.sample),
            "sample_ci_sha256": ci_hash,
            "manifest": str(args.manifest),
            "leg9_manifest": str(args.leg9_manifest),
            "population_moments": str(args.moments),
            "n_rows": int(args.n_rows),
            "top_k": int(args.top_k),
            "candidate_k": int(min(args.candidate_k, args.n_rows)),
            "threads": int(args.threads),
            "block": int(args.block),
            "elapsed_seconds": float(time.time() - started),
            **as_metadata_dict(
                git_provenance(args.repo_root, argv0=__file__),
                phase="leg12-direction-examples",
            ),
        },
    }
    output_dir = args.repo_root / OUTPUT_RELPATH
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{RESULT_NAME}.json"
    markdown_path = output_dir / f"{RESULT_NAME}.md"
    write_text_atomic(json_path, json.dumps(document, indent=1) + "\n")
    write_text_atomic(markdown_path, render_markdown(document))
    done = {
        "status": "complete",
        "n_rows": int(args.n_rows),
        "n_directions": n_directions,
        "sample_ci_sha256": ci_hash,
        "json_sha256": sha256_file(json_path),
        "markdown_sha256": sha256_file(markdown_path),
        "elapsed_seconds": float(time.time() - started),
    }
    write_text_atomic(output_dir / f"{RESULT_NAME}.done.json", json.dumps(done, indent=1) + "\n")
    LOGGER.info("[out] %s", json_path)
    LOGGER.info("[out] %s", markdown_path)
    LOGGER.info("DONE directions=%d rows=%d", n_directions, args.n_rows)


if __name__ == "__main__":
    main()
