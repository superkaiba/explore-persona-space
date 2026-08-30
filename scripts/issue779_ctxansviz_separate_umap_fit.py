"""Fit independent context-only and answer-only UMAPs for issue #779.

The original ctxansviz UMAP fits one neighborhood graph on a joint sample of
context and answer points. This script instead fits two UMAP models on the
same deterministic paired row IDs: one model sees only context vectors and
one sees only answer vectors. Both use the pinned joint PCA-100 transform as a
shared, linear 3,584 -> 100 preprocessing step; the nonlinear graphs and 2-D
coordinate systems are independent.

Usage:
    uv run python scripts/issue779_ctxansviz_separate_umap_fit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import umap
from huggingface_hub import HfApi, hf_hub_download
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

import issue779_ctxansviz_pca3_dashboard as display_source
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

CAPTURE_REVISION = "cbc55efdd7f5581677047e487aa61172f6e7944d"
EXPORT_REVISION = "d155ed93f4b0184a477cea51aef65cc5440da588"
EXPORT_PRODUCER_COMMIT = "79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d"
HF_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
LAYER = 19
HIDDEN_DIM = 3_584
PCA_DIM = 100
N_FIT_ROWS = 100_000
N_FIT_CHUNKS = 201
SEED = 42
N_NEIGHBORS = 15
MIN_DIST = 0.1
DEFAULT_EXPORT = Path("data/issue_779/ctxansviz_dl/full/issue779_monitoring/ctxansviz")
DEFAULT_OUT_DIR = Path("data/issue_779/ctxansviz_separate_umap")
TEMP_PARENT = Path("/mnt/eps-data/thomasjiralerspong")
ARTIFACT_NAME = "separate_umap_coords.npz"
META_NAME = "separate_umap_meta.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pca(export_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    model_path = export_dir / "pca_model.npz"
    meta_path = export_dir / "meta.json"
    download_path = export_dir / "_download_meta.json"
    if not model_path.exists() or not meta_path.exists() or not download_path.exists():
        raise FileNotFoundError(f"pinned PCA export is incomplete: {export_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if meta.get("git_commit") != EXPORT_PRODUCER_COMMIT:
        raise RuntimeError("PCA export producer commit is not pinned")
    if download.get("revision") != EXPORT_REVISION:
        raise RuntimeError("PCA export download revision is not pinned")
    expected_sha = meta.get("export_files_sha256", {}).get("pca_model.npz")
    if expected_sha != sha256_file(model_path):
        raise RuntimeError("PCA model SHA-256 mismatch")
    raw = np.load(model_path)
    components = np.asarray(raw["components"], dtype=np.float32)
    mean = np.asarray(raw["mean"], dtype=np.float32)
    if components.shape != (PCA_DIM, HIDDEN_DIM) or mean.shape != (HIDDEN_DIM,):
        raise RuntimeError(f"unexpected PCA model shapes: {components.shape}, {mean.shape}")
    return components, mean, meta


def capture_universe() -> list:
    items = list(
        HfApi().list_repo_tree(
            HF_REPO,
            path_in_repo=CAPTURE_PREFIX,
            recursive=False,
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
    )
    items = sorted((item for item in items if item.path.endswith(".pt")), key=lambda x: x.path)
    if len(items) != 1_920:
        raise RuntimeError(f"capture universe contains {len(items)} chunks, expected 1,920")
    return items


def selection_manifest(items: list) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{item.path}\t{item.size}\t{item.lfs.sha256 if item.lfs else 'no-lfs-sha'}"
            for item in items
        ).encode()
    ).hexdigest()


def select_fit_chunks(items: list, n_chunks: int) -> list:
    indices = np.round(np.linspace(0, len(items) - 1, n_chunks)).astype(int)
    if len(set(indices.tolist())) != n_chunks:
        raise RuntimeError("evenly spaced fit chunk selection contains duplicate indices")
    return [items[index] for index in indices]


def project_bundle(
    bundle: dict, components: np.ndarray, mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers = [int(value) for value in bundle["layers"]]
    if LAYER not in layers:
        raise RuntimeError(f"capture bundle does not contain layer {LAYER}")
    column = layers.index(LAYER)
    context = bundle["cx_last"][:, column, :].to(torch.float32).numpy()
    answer = bundle["v_x"][:, column, :].to(torch.float32).numpy()
    ci = np.asarray([int(value) for value in bundle["ci"]], dtype=np.int64)
    if context.shape != answer.shape or context.shape != (len(ci), HIDDEN_DIM):
        raise RuntimeError(f"malformed capture bundle: {context.shape}, {answer.shape}")
    context_pca = (context - mean) @ components.T
    answer_pca = (answer - mean) @ components.T
    return ci, context_pca.astype(np.float32), answer_pca.astype(np.float32)


def fill_fit_arrays(
    context_path: Path,
    answer_path: Path,
    selected: list,
    n_fit: int,
    components: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, dict]:
    context = np.lib.format.open_memmap(
        context_path, mode="w+", dtype=np.float32, shape=(n_fit, PCA_DIM)
    )
    answer = np.lib.format.open_memmap(
        answer_path, mode="w+", dtype=np.float32, shape=(n_fit, PCA_DIM)
    )
    fit_ci = np.empty(n_fit, dtype=np.int64)
    cursor = 0
    started = time.time()
    for position, item in enumerate(selected):
        if cursor == n_fit:
            break
        local = hf_hub_download(
            HF_REPO,
            filename=item.path,
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        bundle = torch.load(local, mmap=True, weights_only=False, map_location="cpu")
        ci, context_pca, answer_pca = project_bundle(bundle, components, mean)
        take = min(len(ci), n_fit - cursor)
        context[cursor : cursor + take] = context_pca[:take]
        answer[cursor : cursor + take] = answer_pca[:take]
        fit_ci[cursor : cursor + take] = ci[:take]
        cursor += take
        if (position + 1) % 20 == 0 or cursor == n_fit:
            print(
                f"[separate-umap] projected chunks={position + 1}/{len(selected)} "
                f"rows={cursor:,}/{n_fit:,} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    context.flush()
    answer.flush()
    if cursor != n_fit or len(set(fit_ci.tolist())) != n_fit:
        raise RuntimeError(f"fit selection incomplete or duplicate: rows={cursor:,}/{n_fit:,}")
    ci_meta = {
        "ci_sha256": hashlib.sha256(fit_ci.tobytes()).hexdigest(),
        "ci_min": int(fit_ci.min()),
        "ci_max": int(fit_ci.max()),
    }
    return fit_ci, ci_meta


def display_projections(
    item_by_name: dict[str, object],
    chunks: tuple[str, ...],
    components: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cis: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    answers: list[np.ndarray] = []
    for chunk in chunks:
        item = item_by_name.get(chunk)
        if item is None:
            raise RuntimeError(f"display chunk absent from capture universe: {chunk}")
        local = hf_hub_download(
            HF_REPO,
            filename=item.path,
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        bundle = torch.load(local, mmap=True, weights_only=False, map_location="cpu")
        ci, context_pca, answer_pca = project_bundle(bundle, components, mean)
        cis.append(ci)
        contexts.append(context_pca)
        answers.append(answer_pca)
    ci = np.concatenate(cis)
    context = np.concatenate(contexts)
    answer = np.concatenate(answers)
    if len(set(ci.tolist())) != len(ci):
        raise RuntimeError("display chunks contain duplicate ci values")
    return ci, context, answer


def new_umap() -> umap.UMAP:
    return umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        n_components=2,
        random_state=SEED,
        n_jobs=1,
        low_memory=True,
        verbose=False,
    )


def neighbor_indices(values: np.ndarray, metric: str, k: int = N_NEIGHBORS) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    model.fit(values)
    candidates = model.kneighbors(values, return_distance=False)
    indices = np.empty((len(values), k), dtype=np.int64)
    for row_index, row in enumerate(candidates):
        without_self = row[row != row_index]
        if len(without_self) < k:
            raise RuntimeError("nearest-neighbor query did not return enough non-self rows")
        indices[row_index] = without_self[:k]
    return indices


def mean_neighbor_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError("neighbor arrays have different shapes")
    overlaps = [len(set(a.tolist()) & set(b.tolist())) / left.shape[1] for a, b in zip(left, right)]
    return float(np.mean(overlaps))


def procrustes_stats(context: np.ndarray, answer: np.ndarray) -> dict:
    context_z = context - context.mean(0)
    answer_z = answer - answer.mean(0)
    context_scale = float(np.linalg.norm(context_z))
    answer_scale = float(np.linalg.norm(answer_z))
    context_z /= context_scale
    answer_z /= answer_scale
    rotation, _ = orthogonal_procrustes(answer_z, context_z)
    aligned = answer_z @ rotation
    residual = context_z - aligned
    return {
        "normalized_rmse": float(np.sqrt(np.mean(residual**2))),
        "disparity": float(np.sum(residual**2)),
        "paired_distance_median": float(np.median(np.linalg.norm(residual, axis=1))),
        "paired_distance_p90": float(np.quantile(np.linalg.norm(residual, axis=1), 0.9)),
        "rotation": rotation.tolist(),
        "context_center": context.mean(0).tolist(),
        "answer_center": answer.mean(0).tolist(),
        "context_scale": context_scale,
        "answer_scale": answer_scale,
    }


def quality_stats(
    context_pca: np.ndarray,
    answer_pca: np.ndarray,
    context_umap: np.ndarray,
    answer_umap: np.ndarray,
) -> dict:
    context_native_nn = neighbor_indices(context_pca, "cosine")
    answer_native_nn = neighbor_indices(answer_pca, "cosine")
    context_umap_nn = neighbor_indices(context_umap, "euclidean")
    answer_umap_nn = neighbor_indices(answer_umap, "euclidean")
    return {
        "trustworthiness_k15": {
            "context": float(
                trustworthiness(context_pca, context_umap, n_neighbors=N_NEIGHBORS, metric="cosine")
            ),
            "answer": float(
                trustworthiness(answer_pca, answer_umap, n_neighbors=N_NEIGHBORS, metric="cosine")
            ),
        },
        "native_to_umap_neighbor_recall_k15": {
            "context": mean_neighbor_overlap(context_native_nn, context_umap_nn),
            "answer": mean_neighbor_overlap(answer_native_nn, answer_umap_nn),
        },
        "context_answer_neighbor_overlap_k15": {
            "native_pca100": mean_neighbor_overlap(context_native_nn, answer_native_nn),
            "separate_umap": mean_neighbor_overlap(context_umap_nn, answer_umap_nn),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-fit", type=int, default=N_FIT_ROWS)
    parser.add_argument("--n-fit-chunks", type=int, default=N_FIT_CHUNKS)
    parser.add_argument("--display-chunks", nargs="+", default=list(display_source.DEFAULT_CHUNKS))
    args = parser.parse_args()
    if args.n_fit < 2_000 or args.n_fit_chunks < 5:
        raise ValueError("fit requires at least 2,000 rows and five chunks")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.out_dir / ARTIFACT_NAME
    meta_path = args.out_dir / META_NAME

    components, mean, export_meta = load_pca(args.export_dir)
    universe = capture_universe()
    selected = select_fit_chunks(universe, args.n_fit_chunks)
    item_by_name = {Path(item.path).name: item for item in universe}
    with tempfile.TemporaryDirectory(prefix="issue779-separate-umap-", dir=TEMP_PARENT) as tmp:
        tmp_path = Path(tmp)
        context_path = tmp_path / "context_pca100.npy"
        answer_path = tmp_path / "answer_pca100.npy"
        fit_ci, fit_ci_meta = fill_fit_arrays(
            context_path,
            answer_path,
            selected,
            args.n_fit,
            components,
            mean,
        )
        context_fit = np.lib.format.open_memmap(context_path, mode="r")
        answer_fit = np.lib.format.open_memmap(answer_path, mode="r")
        display_ci, display_context_pca, display_answer_pca = display_projections(
            item_by_name, tuple(args.display_chunks), components, mean
        )

        started = time.time()
        print(f"[separate-umap] fitting context UMAP on {context_fit.shape}", flush=True)
        context_model = new_umap()
        context_fit_umap = context_model.fit_transform(context_fit).astype(np.float32)
        context_fit_s = time.time() - started
        started = time.time()
        print(f"[separate-umap] fitting answer UMAP on {answer_fit.shape}", flush=True)
        answer_model = new_umap()
        answer_fit_umap = answer_model.fit_transform(answer_fit).astype(np.float32)
        answer_fit_s = time.time() - started

        started = time.time()
        display_context_umap = context_model.transform(display_context_pca).astype(np.float32)
        context_transform_s = time.time() - started
        started = time.time()
        display_answer_umap = answer_model.transform(display_answer_pca).astype(np.float32)
        answer_transform_s = time.time() - started

        quality = quality_stats(
            display_context_pca,
            display_answer_pca,
            display_context_umap,
            display_answer_umap,
        )
        procrustes = procrustes_stats(context_fit_umap, answer_fit_umap)

        np.savez_compressed(
            artifact_path,
            fit_ci=fit_ci,
            context_fit_umap=context_fit_umap,
            answer_fit_umap=answer_fit_umap,
            display_ci=display_ci,
            context_display_umap=display_context_umap,
            answer_display_umap=display_answer_umap,
        )

    artifact_sha = sha256_file(artifact_path)
    meta = {
        **as_metadata_dict(git_provenance(), phase="separate-umap-fit"),
        "issue": 779,
        "capture_revision": CAPTURE_REVISION,
        "capture_prefix": CAPTURE_PREFIX,
        "export_revision": EXPORT_REVISION,
        "export_producer_commit": EXPORT_PRODUCER_COMMIT,
        "pca_model_sha256": export_meta["export_files_sha256"]["pca_model.npz"],
        "preprocessing": (
            "shared pinned joint PCA-100; UMAP neighborhood graphs and 2-D fits are independent"
        ),
        "layer": LAYER,
        "hidden_dim": HIDDEN_DIM,
        "pca_dim": PCA_DIM,
        "n_fit_rows_per_role": args.n_fit,
        "fit_rows_identical_across_roles": True,
        "fit_ci": fit_ci_meta,
        "selection": {
            "rule": (
                f"round(linspace(0, 1919, {args.n_fit_chunks})); consume rows in chunk "
                f"order until {args.n_fit} unique ci values"
            ),
            "n_capture_chunks_available": len(universe),
            "n_chunks_selected": args.n_fit_chunks,
            "selected_chunk_manifest_sha256": selection_manifest(selected),
            "first_chunk": selected[0].path,
            "last_chunk": selected[-1].path,
        },
        "display": {
            "chunks": list(args.display_chunks),
            "n_rows": int(len(display_ci)),
            "ci_sha256": hashlib.sha256(display_ci.tobytes()).hexdigest(),
        },
        "umap_params": {
            "n_neighbors": N_NEIGHBORS,
            "min_dist": MIN_DIST,
            "metric": "cosine",
            "n_components": 2,
            "random_state": SEED,
            "n_jobs": 1,
        },
        "wall_seconds": {
            "context_fit": round(context_fit_s, 2),
            "answer_fit": round(answer_fit_s, 2),
            "context_display_transform": round(context_transform_s, 2),
            "answer_display_transform": round(answer_transform_s, 2),
        },
        "quality": quality,
        "procrustes_descriptive_only": procrustes,
        "artifact_sha256": artifact_sha,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[separate-umap] wrote {artifact_path} ({artifact_sha})")
    print(f"[separate-umap] wrote {meta_path}")


if __name__ == "__main__":
    main()
