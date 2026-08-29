"""Fit independent context-only and answer-only PCA-10 models for issue #779.

The original ctxansviz PCA stacks context and answer vectors before fitting one
shared basis. This script instead fits two bases on the same deterministic,
evenly spaced 200,000-row capture sample: one from ``cx_last`` only and one
from ``v_x`` only. The resulting small model and provenance manifest are used
by the separate-PCA specimen dashboard.

Usage:
    uv run python scripts/issue779_ctxansviz_separate_pca_fit.py
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
from huggingface_hub import HfApi, hf_hub_download
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

CAPTURE_REVISION = "cbc55efdd7f5581677047e487aa61172f6e7944d"
HF_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
LAYER = 19
HIDDEN_DIM = 3584
N_COMPONENTS = 10
N_FIT_ROWS = 200_000
N_FIT_CHUNKS = 401
SEED = 42
DEFAULT_OUT_DIR = Path("data/issue_779/ctxansviz_separate_pca")
MODEL_NAME = "separate_pca10_models.npz"
META_NAME = "separate_pca10_meta.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_chunks() -> tuple[list, str]:
    items = list(
        HfApi().list_repo_tree(
            HF_REPO,
            path_in_repo=CAPTURE_PREFIX,
            recursive=False,
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
    )
    items = sorted(item for item in items if item.path.endswith(".pt"))
    if len(items) != 1_920:
        raise RuntimeError(f"capture universe contains {len(items)} chunks, expected 1,920")
    indices = np.round(np.linspace(0, len(items) - 1, N_FIT_CHUNKS)).astype(int)
    if len(set(indices.tolist())) != N_FIT_CHUNKS:
        raise RuntimeError("evenly spaced chunk selection contains duplicate indices")
    selected = [items[index] for index in indices]
    selection_sha = hashlib.sha256(
        "\n".join(
            f"{item.path}\t{item.size}\t{item.lfs.sha256 if item.lfs else 'no-lfs-sha'}"
            for item in selected
        ).encode()
    ).hexdigest()
    return selected, selection_sha


def fill_fit_memmap(path: Path, selected: list, field: str) -> dict:
    matrix = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(N_FIT_ROWS, HIDDEN_DIM),
    )
    cursor = 0
    selected_cis: list[int] = []
    started = time.time()
    for position, item in enumerate(selected):
        if cursor == N_FIT_ROWS:
            break
        local = hf_hub_download(
            HF_REPO,
            filename=item.path,
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        bundle = torch.load(local, mmap=True, weights_only=False, map_location="cpu")
        layers = [int(value) for value in bundle["layers"]]
        if LAYER not in layers:
            raise RuntimeError(f"{item.path}: layer {LAYER} absent")
        column = layers.index(LAYER)
        values = bundle[field][:, column, :].to(torch.float32).numpy()
        cis = [int(value) for value in bundle["ci"]]
        if values.shape != (len(cis), HIDDEN_DIM):
            raise RuntimeError(f"{item.path}: malformed {field} shape {values.shape}")
        take = min(len(cis), N_FIT_ROWS - cursor)
        matrix[cursor : cursor + take] = values[:take]
        selected_cis.extend(cis[:take])
        cursor += take
        if (position + 1) % 20 == 0 or cursor == N_FIT_ROWS:
            print(
                f"[separate-pca] {field} chunks={position + 1}/{len(selected)} "
                f"rows={cursor:,}/{N_FIT_ROWS:,} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    matrix.flush()
    if cursor != N_FIT_ROWS:
        raise RuntimeError(f"{field}: collected {cursor:,} rows, expected {N_FIT_ROWS:,}")
    if len(set(selected_cis)) != N_FIT_ROWS:
        raise RuntimeError(f"{field}: selected ci values are not unique")
    return {
        "ci_sha256": hashlib.sha256(np.asarray(selected_cis, dtype=np.int64).tobytes()).hexdigest(),
        "ci_min": min(selected_cis),
        "ci_max": max(selected_cis),
    }


def fit_one(path: Path, field: str) -> PCA:
    matrix = np.lib.format.open_memmap(path, mode="r")
    if matrix.shape != (N_FIT_ROWS, HIDDEN_DIM) or matrix.dtype != np.float32:
        raise RuntimeError(f"unexpected fit matrix for {field}: {matrix.shape} {matrix.dtype}")
    started = time.time()
    print(f"[separate-pca] fitting {field} PCA-{N_COMPONENTS} on {matrix.shape}", flush=True)
    model = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=SEED)
    model.fit(matrix)
    print(
        f"[separate-pca] fitted {field}; EVR10={model.explained_variance_ratio_.sum():.6f} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / MODEL_NAME
    meta_path = args.out_dir / META_NAME

    selected, selection_sha = select_chunks()
    with tempfile.TemporaryDirectory(prefix="issue779-separate-pca-", dir="/mnt/eps-data") as tmp:
        fit_path = Path(tmp) / "fit.npy"
        context_sample = fill_fit_memmap(fit_path, selected, "cx_last")
        context_model = fit_one(fit_path, "cx_last")
        answer_sample = fill_fit_memmap(fit_path, selected, "v_x")
        answer_model = fit_one(fit_path, "v_x")

    if context_sample != answer_sample:
        raise RuntimeError("context and answer PCA fits did not use identical ci rows")
    loading_cosine = context_model.components_ @ answer_model.components_.T
    context_indices, answer_indices = linear_sum_assignment(-np.abs(loading_cosine))
    if not np.array_equal(context_indices, np.arange(N_COMPONENTS)):
        raise RuntimeError("unexpected incomplete context-PC assignment")
    answer_for_context = answer_indices.astype(np.int64)
    orientation_for_context = np.sign(
        loading_cosine[np.arange(N_COMPONENTS), answer_for_context]
    ).astype(np.int64)
    if np.any(orientation_for_context == 0):
        raise RuntimeError("zero-cosine matched component has undefined orientation")

    np.savez(
        model_path,
        context_components=context_model.components_.astype(np.float32),
        context_mean=context_model.mean_.astype(np.float32),
        context_explained_variance=context_model.explained_variance_.astype(np.float64),
        context_explained_variance_ratio=context_model.explained_variance_ratio_.astype(np.float64),
        answer_components=answer_model.components_.astype(np.float32),
        answer_mean=answer_model.mean_.astype(np.float32),
        answer_explained_variance=answer_model.explained_variance_.astype(np.float64),
        answer_explained_variance_ratio=answer_model.explained_variance_ratio_.astype(np.float64),
        loading_cosine=loading_cosine.astype(np.float64),
        answer_for_context=answer_for_context,
        orientation_for_context=orientation_for_context,
        n_fit_rows=np.int64(N_FIT_ROWS),
        seed=np.int64(SEED),
    )
    meta = {
        **as_metadata_dict(git_provenance(), phase="separate-pca-fit"),
        "issue": 779,
        "capture_revision": CAPTURE_REVISION,
        "capture_prefix": CAPTURE_PREFIX,
        "layer": LAYER,
        "hidden_dim": HIDDEN_DIM,
        "n_components": N_COMPONENTS,
        "n_fit_rows_per_basis": N_FIT_ROWS,
        "fit_rows_identical_across_bases": True,
        "fit_ci": context_sample,
        "selection": {
            "rule": (
                f"round(linspace(0, 1919, {N_FIT_CHUNKS})); consume rows in chunk "
                f"order until {N_FIT_ROWS} unique ci values"
            ),
            "n_capture_chunks_available": 1_920,
            "n_chunks_selected": N_FIT_CHUNKS,
            "selected_chunk_manifest_sha256": selection_sha,
            "first_chunk": selected[0].path,
            "last_chunk": selected[-1].path,
        },
        "context_evr10": float(context_model.explained_variance_ratio_.sum()),
        "answer_evr10": float(answer_model.explained_variance_ratio_.sum()),
        "model_sha256": sha256_file(model_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[separate-pca] wrote {model_path} ({sha256_file(model_path)})")
    print(f"[separate-pca] wrote {meta_path}")
    print("[separate-pca] removed the temporary 2.86 GB fit memmap")


if __name__ == "__main__":
    main()
