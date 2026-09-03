#!/usr/bin/env python3
"""Mapping-rank extension of the issue-2588 capability panel.

The parent experiment persisted the selected layer/lambda and all activation
shards, but intentionally discarded each selected-layer ridge matrix.  This
script reconstructs only those frozen maps, measures their coefficient TSVD
rank, and relates dimension-normalized rank to the panel's pre-registered
Artificial Analysis (AA) capability values.

Two rank summaries are reported:

* stable rank: ||W||_F^2 / ||W||_2^2;
* operational rank: the smallest coefficient-TSVD rank whose validation R^2
  is within 0.02 of the full reconstructed map.

For d <= 2048 the SVD is exact.  Wider maps use a deterministic randomized
top-k SVD; the retained subspace must reach the operational threshold or the
run fails rather than silently reporting a lower bound.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bind BLAS before importing numpy/scipy.  Sixteen physical cores are present
# on the analysis VM; using SMT siblings tends to slow these large GEMMs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_tree  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.utils.extmath import randomized_svd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "74bb871a5edf1afe777ac9b64a4e2fec5e9947c2"  # full same-width panel incl. OLMo-3-32B-Think, 2026-09-03
PANEL_PREFIX = "issue2588_capability_panel"
DEFAULT_CACHE = REPO / "data" / "issue_2588" / "mapping_rank_cache"
DEFAULT_OUT = REPO / "eval_results" / "issue_2588" / "mapping_rank_vs_capability.json"
DEFAULT_FIGURE = REPO / "figures" / "issue_2588" / "mapping_rank_vs_capability.png"
DEFAULT_FOCUSED_FIGURE = (
    REPO / "figures" / "issue_2588" / "rank_fraction_vs_capability_all_models.png"
)
DEFAULT_PERFORMANCE_FIGURE = (
    REPO / "figures" / "issue_2588" / "mapping_performance_vs_capability_qwen.png"
)
DEFAULT_SAME_WIDTH_FIGURE = REPO / "figures" / "issue_2588" / "same_width_column_vs_capability.png"
R2_TOLERANCE = 0.02
RANDOM_SEED = 2588


@dataclass(frozen=True)
class MapSpec:
    cell: str
    position: str
    model_label: str
    family: str
    arm: str
    aa_index: float | None
    aa_status: str
    # Measured non-reasoning AA value when AA lists one separately (recorded,
    # not used as the x-axis: every arm-a row keeps the model-level pin so the
    # panel-wide convention stays uniform).
    aa_index_nonreasoning: float | None = None

    @property
    def model_key(self) -> str:
        return self.cell.rsplit("_", 1)[0]

    @property
    def generation_dir(self) -> str:
        return "nothink" if self.arm == "no-thinking" else "think"

    @property
    def key(self) -> str:
        return f"{self.cell}__{self.position}"


# Same-width column: every Qwen checkpoint with hidden size 5120 and 64 layers.
# The label replaced "Qwen 27B column" on 2026-09-03 when the 32B releases
# (Qwen2.5-32B-Instruct, Qwen3-32B, QwQ-32B) joined the three 27B releases.
QWEN_COLUMN = "Qwen h=5120 column"
LEGACY_FAMILY_LABELS = {"Qwen 27B column": QWEN_COLUMN}  # pre-2026-09-03 payloads
SAME_WIDTH_DIM = 5120
POINT_NUMBERS = {
    "Q3.5 0.8B": "1",
    "Q3.5 2B": "2",
    "Q3.5 4B": "3",
    "Q3.5 9B": "4",
    "Q3.5 27B": "5",
    "Q3.6 27B": "6",
    "Q3.8 27B": "7",
    "OLMo3 7B I": "8",
    "OLMo3 7B T": "8",
    "OLMo3.1 32B I": "9",
    "OLMo3.1 32B T": "9",
    "Q2.5 32B": "10",
    "Q3 32B": "11",
    "QwQ 32B": "12",
    "OLMo3 32B T": "13",
}
# Reader-facing names for the point keys (the short labels stay internal).
DISPLAY_NAMES = {
    "Q3.5 0.8B": "Qwen3.5 0.8B",
    "Q3.5 2B": "Qwen3.5 2B",
    "Q3.5 4B": "Qwen3.5 4B",
    "Q3.5 9B": "Qwen3.5 9B",
    "Q3.5 27B": "Qwen3.5 27B",
    "Q3.6 27B": "Qwen3.6 27B",
    "Q3.8 27B": "Qwen3.8 27B",
    "OLMo3 7B I": "OLMo3 7B Instruct",
    "OLMo3 7B T": "OLMo3 7B Think",
    "OLMo3.1 32B I": "OLMo3.1 32B Instruct",
    "OLMo3.1 32B T": "OLMo3.1 32B Think",
    "Q2.5 32B": "Qwen2.5 32B Instruct",
    "Q3 32B": "Qwen3 32B",
    "QwQ 32B": "QwQ 32B",
    "OLMo3 32B T": "OLMo3 32B Think",
}
POINT_KEY_ALL = (
    "Point key\n"
    "1  Qwen3.5 0.8B\n2  Qwen3.5 2B\n3  Qwen3.5 4B\n4  Qwen3.5 9B\n"
    "5  Qwen3.5 27B\n6  Qwen3.6 27B\n7  Qwen3.8 27B\n"
    "8  OLMo3 7B I/T\n9  OLMo3.1 32B I/T\n"
    "10 Qwen2.5 32B\n11 Qwen3 32B\n12 QwQ 32B\n13 OLMo3 32B Think"
)
POINT_KEY_QWEN = (
    "Checkpoint key\n"
    "1   Qwen3.5 0.8B\n2   Qwen3.5 2B\n3   Qwen3.5 4B\n"
    "4   Qwen3.5 9B\n5   Qwen3.5 27B\n6   Qwen3.6 27B\n"
    "7   Qwen3.8 27B\n10  Qwen2.5 32B\n11  Qwen3 32B\n12  QwQ 32B"
)

# The first 18 maps are the parent's two n=9 capability trends.  The two OLMo
# pre-think diagnostic maps and the no-AA Qwen2.5-7B anchor are excluded from
# the trend by construction, matching issue 2588.  The last six maps are the
# 2026-09-03 same-width extension (registry keys q3_32b, qwq_32b, q25_32b,
# o3_32b_t; every AA value measured on v4.1.1).
MAPS = (
    MapSpec("q35_0p8b_a", "prompt_last", "Q3.5 0.8B", "Qwen3.5", "no-thinking", 5, "estimated"),
    MapSpec("q35_2b_a", "prompt_last", "Q3.5 2B", "Qwen3.5", "no-thinking", 7, "estimated"),
    MapSpec("q35_4b_a", "prompt_last", "Q3.5 4B", "Qwen3.5", "no-thinking", 20, "estimated"),
    MapSpec("q35_9b_a", "prompt_last", "Q3.5 9B", "Qwen3.5", "no-thinking", 22, "measured"),
    MapSpec("q35_27b_a", "prompt_last", "Q3.5 27B", QWEN_COLUMN, "no-thinking", 35, "estimated"),
    MapSpec("q36_27b_a", "prompt_last", "Q3.6 27B", QWEN_COLUMN, "no-thinking", 38, "measured"),
    MapSpec("q38_27b_a", "prompt_last", "Q3.8 27B", QWEN_COLUMN, "no-thinking", 52, "measured"),
    MapSpec("o3_7b_i_a", "prompt_last", "OLMo3 7B I", "OLMo", "no-thinking", 2, "estimated"),
    MapSpec("o31_32b_i_a", "prompt_last", "OLMo3.1 32B I", "OLMo", "no-thinking", 6, "estimated"),
    MapSpec("q35_0p8b_b", "cot_boundary", "Q3.5 0.8B", "Qwen3.5", "end-of-thought", 5, "estimated"),
    MapSpec("q35_2b_b", "cot_boundary", "Q3.5 2B", "Qwen3.5", "end-of-thought", 7, "estimated"),
    MapSpec("q35_4b_b", "cot_boundary", "Q3.5 4B", "Qwen3.5", "end-of-thought", 20, "estimated"),
    MapSpec("q35_9b_b", "cot_boundary", "Q3.5 9B", "Qwen3.5", "end-of-thought", 22, "measured"),
    MapSpec(
        "q35_27b_b", "cot_boundary", "Q3.5 27B", QWEN_COLUMN, "end-of-thought", 35, "estimated"
    ),
    MapSpec("q36_27b_b", "cot_boundary", "Q3.6 27B", QWEN_COLUMN, "end-of-thought", 38, "measured"),
    MapSpec("q38_27b_b", "cot_boundary", "Q3.8 27B", QWEN_COLUMN, "end-of-thought", 52, "measured"),
    MapSpec("o3_7b_t_b", "cot_boundary", "OLMo3 7B T", "OLMo", "end-of-thought", 4, "estimated"),
    MapSpec(
        "o31_32b_t_b", "cot_boundary", "OLMo3.1 32B T", "OLMo", "end-of-thought", 8, "estimated"
    ),
    MapSpec("q25_32b_a", "prompt_last", "Q2.5 32B", QWEN_COLUMN, "no-thinking", 7, "measured"),
    MapSpec("q3_32b_a", "prompt_last", "Q3 32B", QWEN_COLUMN, "no-thinking", 11, "measured", 8),
    MapSpec("q3_32b_b", "cot_boundary", "Q3 32B", QWEN_COLUMN, "end-of-thought", 11, "measured", 8),
    MapSpec("qwq_32b_b", "cot_boundary", "QwQ 32B", QWEN_COLUMN, "end-of-thought", 13, "measured"),
    MapSpec("o3_32b_t_b", "cot_boundary", "OLMo3 32B T", "OLMo", "end-of-thought", 6, "measured"),
)


def pooled_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    sse = np.square(y - p).sum(dtype=np.float64)
    sst = np.square(y - y.mean(axis=0, keepdims=True)).sum(dtype=np.float64)
    return float(1.0 - sse / (sst + 1e-30))


def r2_curve_from_top_right_vectors(
    full_prediction: np.ndarray,
    target: np.ndarray,
    intercept: np.ndarray,
    right_vectors: np.ndarray,
) -> np.ndarray:
    """R^2 at ranks 0..k for a nested top-k coefficient TSVD basis."""
    pred = np.asarray(full_prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mu = np.asarray(intercept, dtype=np.float64).reshape(1, -1)
    v = np.asarray(right_vectors, dtype=np.float64)
    pc = (pred - mu) @ v
    yc = (y - mu) @ v
    c2 = np.einsum("ij,ij->j", pc, pc, dtype=np.float64)
    cy = np.einsum("ij,ij->j", pc, yc, dtype=np.float64)
    sse0 = np.square(y - mu).sum(dtype=np.float64)
    sse = np.concatenate(([sse0], sse0 + np.cumsum(c2 - 2.0 * cy)))
    sst = np.square(y - y.mean(axis=0, keepdims=True)).sum(dtype=np.float64)
    return 1.0 - sse / (sst + 1e-30)


def minimum_rank_within(curve: np.ndarray, full_r2: float, tolerance: float) -> int | None:
    candidates = np.flatnonzero(np.asarray(curve) >= float(full_r2) - float(tolerance))
    return int(candidates[0]) if len(candidates) else None


def _download(path: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=path,
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )


def _fit_record(spec: MapSpec) -> dict[str, Any]:
    path = f"{PANEL_PREFIX}/fits/{spec.cell}/fits_{spec.position}.json"
    return json.loads(_download(path).read_text(encoding="utf-8"))


def _mapping_performance(spec: MapSpec) -> dict[str, Any]:
    """Read the parent panel's selected-layer generic-test mapping metrics."""
    fit = _fit_record(spec)
    layer = int(fit["layer_star"])
    selected = fit["layers"][str(layer)]
    cosine = selected["knn_test"]["ridge"]["cosine"]
    null_path = f"{PANEL_PREFIX}/nulls/{spec.cell}/nulls_{spec.position}.json"
    null = json.loads(_download(null_path).read_text(encoding="utf-8"))
    raw_acc1 = float(cosine["acc_at_k"]["1"])
    null_mean = float(null["null_mean_acc1_cos"])
    return {
        "surface": "held-out generic real-user corpus",
        "layer_star": layer,
        "test_r2": float(selected["test_r2"]),
        "test_retrieval_acc1_cos": raw_acc1,
        "test_retrieval_acc1_cos_null_mean": null_mean,
        "test_retrieval_acc1_cos_calibrated": raw_acc1 - null_mean,
        "test_n": int(cosine["n"]),
        "calibration": (
            "raw cosine retrieval acc@1 minus shuffled-pairing null mean; "
            f"{int(null['perm_draws'])} permutation draws"
        ),
    }


def _measured_gpqa_accuracy(spec: MapSpec) -> dict[str, Any]:
    """The parent panel's own measured capability axis on GPQA Diamond."""
    path = f"{PANEL_PREFIX}/fits/{spec.cell}/gpqa_transfer_{spec.position}.json"
    record = json.loads(_download(path).read_text(encoding="utf-8"))
    behavioral = record["behavioral"]
    return {
        "benchmark": "GPQA Diamond, parent issue-2588 own rollouts",
        "accuracy": float(behavioral["acc_exact_match"]),
        "n_rollouts": int(behavioral["n_rollouts"]),
        "n_correct": int(behavioral["n_correct"]),
        "n_unparseable": int(behavioral["n_unparseable"]),
    }


def _capture_prefix(spec: MapSpec, split: str, layer: int) -> str:
    return (
        f"{PANEL_PREFIX}/{spec.model_key}/{spec.generation_dir}/analysis_tensors/"
        f"capture/{split}/L{layer:02d}"
    )


def _capture_files(spec: MapSpec, split: str, layer: int) -> list[str]:
    prefix = _capture_prefix(spec, split, layer)
    entries = list_repo_tree(
        HF_REPO,
        path_in_repo=prefix,
        recursive=False,
        revision=HF_REVISION,
        repo_type="dataset",
    )
    paths = sorted(e.path for e in entries if e.path.endswith(".npz"))
    if not paths:
        raise RuntimeError(f"no activation shards under {prefix}")
    return paths


def load_split(spec: MapSpec, split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    paths = _capture_files(spec, split, layer)
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        local = list(pool.map(_download, paths))
    ids: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for path in local:
        with np.load(path, allow_pickle=False) as z:
            ids.append(z["row_ids"])
            xs.append(z[f"x_{spec.position}"].astype(np.float32, copy=False))
            ys.append(z["y_ans"].astype(np.float32, copy=False))
    row_ids = np.concatenate(ids)
    order = np.argsort(row_ids)
    return np.concatenate(xs)[order], np.concatenate(ys)[order]


def _payload_cache_path(spec: MapSpec, cache_dir: Path) -> Path:
    return cache_dir / f"{spec.key}.npz"


def reconstruct_map(spec: MapSpec, cache_dir: Path) -> dict[str, Any]:
    """Reconstruct the frozen selected-layer/selected-lambda ridge map."""
    fit = _fit_record(spec)
    layer = int(fit["layer_star"])
    star = fit["layers"][str(layer)]
    d = int(star["d"])
    lam = float(star["fit_meta"]["selected_lambda"])
    expected_test_r2 = float(star["test_r2"])
    expected_val_r2 = float(star["fit_meta"]["val_r2_at_selected"])
    cache_path = _payload_cache_path(spec, cache_dir)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if (
                int(z["layer"]) == layer
                and int(z["dimension"]) == d
                and math.isclose(float(z["selected_lambda"]), lam)
                and str(z["hf_revision"]) == HF_REVISION
            ):
                return {k: z[k] for k in z.files}

    started = time.time()
    print(f"[{spec.key}] download selected layer L{layer} (d={d})", flush=True)
    xtr, ytr = load_split(spec, "train_10k", layer)
    xval, yval = load_split(spec, "val_400", layer)
    xte, yte = load_split(spec, "test_1000", layer)
    if xtr.shape[1] != d or ytr.shape[1] != d:
        raise RuntimeError(f"{spec.key}: dimension mismatch X={xtr.shape} Y={ytr.shape} d={d}")

    # Parent parity: fp64 means, unbiased X std + 1e-9, centered Y, primal
    # ridge.  Direct SPD solve is algebraically equivalent to the parent's
    # eigh implementation at the already-frozen lambda.
    xtr64 = xtr.astype(np.float64)
    ytr64 = ytr.astype(np.float64)
    xmu64 = xtr64.mean(axis=0)
    xsd64 = xtr64.std(axis=0, ddof=1) + 1e-9
    ymu64 = ytr64.mean(axis=0)
    xnorm = (xtr64 - xmu64) / xsd64
    ycenter = ytr64 - ymu64
    print(f"[{spec.key}] form normal equations and solve at lambda={lam:g}", flush=True)
    gram = xnorm.T @ xnorm
    gram.flat[:: d + 1] += lam
    cross = xnorm.T @ ycenter
    w64 = scipy.linalg.solve(
        gram,
        cross,
        assume_a="pos",
        overwrite_a=True,
        overwrite_b=True,
        check_finite=False,
    )
    # The parent's reusable map payload stores these tensors in fp32.
    w = np.asarray(w64, dtype=np.float32)
    xmu = np.asarray(xmu64, dtype=np.float32)
    xsd = np.asarray(xsd64, dtype=np.float32)
    ymu = np.asarray(ymu64, dtype=np.float32)
    del gram, cross, w64, xnorm, ycenter, xtr64, ytr64

    pred_val = ((xval - xmu) / xsd) @ w + ymu
    pred_test = ((xte - xmu) / xsd) @ w + ymu
    val_r2 = pooled_r2(pred_val, yval)
    test_r2 = pooled_r2(pred_test, yte)
    parity_tol = 3e-4
    if abs(test_r2 - expected_test_r2) > parity_tol or abs(val_r2 - expected_val_r2) > parity_tol:
        raise RuntimeError(
            f"{spec.key}: reconstruction parity failed: val {val_r2:.6f} vs "
            f"{expected_val_r2:.6f}; test {test_r2:.6f} vs {expected_test_r2:.6f}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        W=w,
        xmu=xmu,
        xsd=xsd,
        ymu=ymu,
        pred_val=pred_val.astype(np.float32),
        target_val=yval.astype(np.float32),
        pred_test=pred_test.astype(np.float32),
        target_test=yte.astype(np.float32),
        layer=np.int64(layer),
        dimension=np.int64(d),
        selected_lambda=np.float64(lam),
        expected_val_r2=np.float64(expected_val_r2),
        expected_test_r2=np.float64(expected_test_r2),
        reconstructed_val_r2=np.float64(val_r2),
        reconstructed_test_r2=np.float64(test_r2),
        hf_revision=np.array(HF_REVISION),
        elapsed_s=np.float64(time.time() - started),
    )
    print(
        f"[{spec.key}] cached map; parity delta test={test_r2 - expected_test_r2:+.2e}",
        flush=True,
    )
    return reconstruct_map(spec, cache_dir)


def coefficient_spectrum(payload: dict[str, Any], *, max_rank: int, n_iter: int) -> dict[str, Any]:
    w = np.asarray(payload["W"], dtype=np.float32)
    d = w.shape[0]
    if w.shape != (d, d):
        raise ValueError(f"W must be square, got {w.shape}")
    if d <= 2048:
        print(f"  exact SVD of {d}x{d} W", flush=True)
        _u, s, vh = scipy.linalg.svd(
            w,
            full_matrices=False,
            compute_uv=True,
            check_finite=False,
            lapack_driver="gesdd",
        )
        del _u
        method = "exact_scipy_gesdd"
    else:
        k = min(int(max_rank), d - 1)
        print(f"  randomized top-{k} SVD of {d}x{d} W", flush=True)
        _u, s, vh = randomized_svd(
            w,
            n_components=k,
            n_oversamples=32,
            n_iter=int(n_iter),
            power_iteration_normalizer="QR",
            random_state=RANDOM_SEED,
            flip_sign=False,
        )
        del _u
        method = f"randomized_svd_top{k}_iter{n_iter}_oversample32"
    right = np.ascontiguousarray(vh.T, dtype=np.float32)
    fro2 = float(np.square(w, dtype=np.float64).sum(dtype=np.float64))
    stable_rank = float(fro2 / (float(s[0]) ** 2))
    captured_energy = float(np.square(s, dtype=np.float64).sum() / fro2)
    return {
        "singular_values": np.asarray(s, dtype=np.float64),
        "right_vectors": right,
        "method": method,
        "stable_rank": stable_rank,
        "stable_rank_fraction": stable_rank / d,
        "topk_energy_fraction": captured_energy,
    }


def analyze_map(spec: MapSpec, cache_dir: Path, *, max_rank: int, svd_iters: int) -> dict[str, Any]:
    started = time.time()
    payload = reconstruct_map(spec, cache_dir)
    spectrum = coefficient_spectrum(payload, max_rank=max_rank, n_iter=svd_iters)
    ymu = payload["ymu"]
    pred_val, yval = payload["pred_val"], payload["target_val"]
    pred_test, ytest = payload["pred_test"], payload["target_test"]
    full_val_r2 = pooled_r2(pred_val, yval)
    full_test_r2 = pooled_r2(pred_test, ytest)
    val_curve = r2_curve_from_top_right_vectors(pred_val, yval, ymu, spectrum["right_vectors"])
    test_curve = r2_curve_from_top_right_vectors(pred_test, ytest, ymu, spectrum["right_vectors"])
    selected_rank = minimum_rank_within(val_curve, full_val_r2, R2_TOLERANCE)
    if selected_rank is None:
        raise RuntimeError(
            f"{spec.key}: top-{len(spectrum['singular_values'])} SVD did not reach "
            f"full validation R2 - {R2_TOLERANCE}; raise --max-rank"
        )
    d = int(payload["dimension"])
    result = {
        "key": spec.key,
        "hf_revision": HF_REVISION,
        "cell": spec.cell,
        "model": spec.model_label,
        "family": spec.family,
        "arm": spec.arm,
        "input_position": spec.position,
        "aa_index": spec.aa_index,
        "aa_status": spec.aa_status,
        "aa_index_nonreasoning": spec.aa_index_nonreasoning,
        "measured_capability": _measured_gpqa_accuracy(spec),
        "mapping_performance": _mapping_performance(spec),
        "layer_star": int(payload["layer"]),
        "dimension": d,
        "selected_lambda": float(payload["selected_lambda"]),
        "n": {"validation": int(len(yval)), "test": int(len(ytest))},
        "reconstruction_parity": {
            "validation_r2": full_val_r2,
            "expected_validation_r2": float(payload["expected_val_r2"]),
            "test_r2": full_test_r2,
            "expected_test_r2": float(payload["expected_test_r2"]),
        },
        "spectrum": {
            "method": spectrum["method"],
            "n_singular_values": int(len(spectrum["singular_values"])),
            "largest_singular_value": float(spectrum["singular_values"][0]),
            "stable_rank": float(spectrum["stable_rank"]),
            "stable_rank_fraction": float(spectrum["stable_rank_fraction"]),
            "topk_energy_fraction": float(spectrum["topk_energy_fraction"]),
        },
        "operational_rank": {
            "definition": "minimum coefficient-TSVD rank within 0.02 of full validation R2",
            "rank": int(selected_rank),
            "rank_fraction": float(selected_rank / d),
            "validation_r2": float(val_curve[selected_rank]),
            "validation_delta_from_full": float(val_curve[selected_rank] - full_val_r2),
            "test_r2": float(test_curve[selected_rank]),
            "test_delta_from_full": float(test_curve[selected_rank] - full_test_r2),
        },
        "rank_curve": {
            "ranks": list(range(len(val_curve))),
            "validation_r2": [float(x) for x in val_curve],
            "test_r2": [float(x) for x in test_curve],
        },
        "elapsed_s": float(time.time() - started),
    }
    print(
        f"[{spec.key}] stable rank={result['spectrum']['stable_rank']:.1f}; "
        f"operational rank={selected_rank}/{d} ({100 * selected_rank / d:.1f}%)",
        flush=True,
    )
    return result


def exact_spearman_permutation(x: list[float], y: list[float]) -> dict[str, Any]:
    """Two-sided permutation test on Spearman rho: exact (all n! relabelings)
    for n <= 9, seeded Monte Carlo (200,000 relabelings) above that."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    observed = float(spearmanr(xa, ya).statistic)
    xr = rankdata(xa)
    yr = rankdata(ya)
    xr = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
    yr = yr - yr.mean()
    denom_y = np.linalg.norm(yr)
    exceed = 0
    if len(xa) <= 9:
        total = math.factorial(len(ya))
        for perm in itertools.permutations(yr.tolist()):
            rho = float(xr @ np.asarray(perm) / denom_y)
            exceed += abs(rho) >= abs(observed) - 1e-12
        method = "exact"
    else:
        total = 200_000
        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(total):
            rho = float(xr @ rng.permutation(yr) / denom_y)
            exceed += abs(rho) >= abs(observed) - 1e-12
        method = "monte_carlo"
    return {
        "n": int(len(xa)),
        "rho": observed,
        "two_sided_exact_permutation_p": float(exceed / total),
        "n_permutations": int(total),
        "method": method,
    }


def descriptive_partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Rank-residual correlation, reported descriptively at this small n."""
    xr, yr, zr = (rankdata(np.asarray(values, dtype=np.float64)) for values in (x, y, z))
    design = np.column_stack((np.ones(len(zr)), zr))
    x_resid = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_resid = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def _column_summary(column: list[dict[str, Any]], note: str) -> dict[str, Any]:
    """Per-arm summary of one fixed-width column, sorted by AA index."""
    aa = [float(r["aa_index"]) for r in column]
    op_frac = [float(r["operational_rank"]["rank_fraction"]) for r in column]
    test_r2 = [
        float(r["mapping_performance"]["test_r2"]) if "mapping_performance" in r else None
        for r in column
    ]
    out: dict[str, Any] = {
        "note": note,
        "n": len(column),
        "aa_index": [r["aa_index"] for r in column],
        "models": [r["model"] for r in column],
        "stable_rank": [r["spectrum"]["stable_rank"] for r in column],
        "operational_rank": [r["operational_rank"]["rank"] for r in column],
        "operational_rank_fraction": op_frac,
        "test_r2": test_r2,
    }
    if len(column) >= 4:
        out["spearman_operational_rank_fraction_vs_aa"] = exact_spearman_permutation(aa, op_frac)
        if all(v is not None for v in test_r2):
            out["spearman_test_r2_vs_aa"] = exact_spearman_permutation(aa, test_r2)
    return out


def summarize_trends(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("no-thinking", "end-of-thought"):
        rows = sorted((r for r in results if r["arm"] == arm), key=lambda r: r["aa_index"])
        x = [float(r["aa_index"]) for r in rows]
        out[arm] = {
            "stable_rank": exact_spearman_permutation(
                x, [float(r["spectrum"]["stable_rank"]) for r in rows]
            ),
            "stable_rank_fraction": exact_spearman_permutation(
                x, [float(r["spectrum"]["stable_rank_fraction"]) for r in rows]
            ),
            "operational_rank": exact_spearman_permutation(
                x, [float(r["operational_rank"]["rank"]) for r in rows]
            ),
            "operational_rank_fraction": exact_spearman_permutation(
                x, [float(r["operational_rank"]["rank_fraction"]) for r in rows]
            ),
        }
        gpqa_x = [float(r["measured_capability"]["accuracy"]) for r in rows]
        out[arm]["secondary_measured_gpqa_axis"] = {
            "note": (
                "Fully measured secondary capability axis from the parent; valid for "
                "these generic-corpus maps, but not for GPQA mapping-performance reads."
            ),
            "stable_rank": exact_spearman_permutation(
                gpqa_x, [float(r["spectrum"]["stable_rank"]) for r in rows]
            ),
            "stable_rank_fraction": exact_spearman_permutation(
                gpqa_x, [float(r["spectrum"]["stable_rank_fraction"]) for r in rows]
            ),
            "operational_rank": exact_spearman_permutation(
                gpqa_x, [float(r["operational_rank"]["rank"]) for r in rows]
            ),
            "operational_rank_fraction": exact_spearman_permutation(
                gpqa_x, [float(r["operational_rank"]["rank_fraction"]) for r in rows]
            ),
        }
        dimensions = [float(r["dimension"]) for r in rows]
        op_fractions = [float(r["operational_rank"]["rank_fraction"]) for r in rows]
        stable_ranks = [float(r["spectrum"]["stable_rank"]) for r in rows]
        out[arm]["dimension_confound_diagnostic"] = {
            "aa_index_vs_dimension": exact_spearman_permutation(x, dimensions),
            "operational_rank_fraction_vs_dimension": exact_spearman_permutation(
                dimensions, op_fractions
            ),
            "partial_spearman_aa_vs_operational_rank_fraction_given_dimension_DESCRIPTIVE": (
                descriptive_partial_spearman(x, op_fractions, dimensions)
            ),
            "partial_spearman_aa_vs_stable_rank_given_dimension_DESCRIPTIVE": (
                descriptive_partial_spearman(x, stable_ranks, dimensions)
            ),
            "warning": (
                "n=9 and AA is correlated with width; partial coefficients are descriptive, "
                "not causal or independently powered. The same-width columns below are cleaner."
            ),
        }
        qwen_column = sorted(
            (r for r in rows if r["family"] == QWEN_COLUMN), key=lambda r: r["aa_index"]
        )
        out[arm]["same_width_qwen_column"] = _column_summary(
            qwen_column,
            "Qwen checkpoints sharing hidden size 5120 and 64 layers: Qwen2.5-32B, "
            "Qwen3-32B, QwQ-32B, Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B (one family, "
            "one width, one depth; the cleanest capability contrast in the panel)",
        )
        same_width = sorted(
            (r for r in rows if int(r["dimension"]) == SAME_WIDTH_DIM),
            key=lambda r: r["aa_index"],
        )
        out[arm]["same_width_all_families"] = _column_summary(
            same_width,
            "every panel row with hidden size 5120: the Qwen column plus the OLMo 32B "
            "rows (width matched, family and post-training not matched)",
        )
    return out


def _plot_one_arm(
    ax,
    rows: list[dict[str, Any]],
    value_path: tuple[str, str],
    ylabel: str,
    *,
    multiplier: float,
) -> None:
    colors = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}
    markers = {"Qwen3.5": "o", QWEN_COLUMN: "s", "OLMo": "^"}
    for family, marker in markers.items():
        subset = sorted((r for r in rows if r["family"] == family), key=lambda r: r["aa_index"])
        if not subset:
            continue
        x = [r["aa_index"] for r in subset]
        y = [multiplier * r[value_path[0]][value_path[1]] for r in subset]
        # Connect only coherent within-family sequences.  The fixed-width
        # release column is deliberately visually strongest.
        lw = 1.8 if family == QWEN_COLUMN else 0.9
        alpha = 1.0 if family == QWEN_COLUMN else 0.7
        ax.plot(x, y, color=colors[rows[0]["arm"]], lw=lw, alpha=alpha, zorder=1)
        for rec, xi, yi in zip(subset, x, y, strict=True):
            face = colors[rec["arm"]] if rec["aa_status"] == "measured" else "white"
            ax.scatter(
                xi,
                yi,
                s=42,
                marker=marker,
                facecolor=face,
                edgecolor=colors[rec["arm"]],
                linewidth=1.2,
                zorder=3,
            )
            point_number = POINT_NUMBERS.get(rec["model"], "?")
            ax.annotate(
                point_number,
                (xi, yi),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                fontweight="bold",
                color="#333333",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Artificial Analysis intelligence index")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def render_figure(results: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.0), sharex=True)
    for col, arm in enumerate(("no-thinking", "end-of-thought")):
        rows = [r for r in results if r["arm"] == arm]
        _plot_one_arm(
            axes[0, col],
            rows,
            ("operational_rank", "rank_fraction"),
            "Performance-preserving rank (% of d)",
            multiplier=100.0,
        )
        _plot_one_arm(
            axes[1, col],
            rows,
            ("spectrum", "stable_rank"),
            "Stable rank",
            multiplier=1.0,
        )
        axes[0, col].set_title(arm.capitalize(), fontsize=9, pad=8)
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    for label, ax in zip("ABCD", axes.ravel(), strict=True):
        ax.text(0.015, 0.98, label, transform=ax.transAxes, fontweight="bold", va="top")
    fig.text(
        0.805,
        0.86,
        POINT_KEY_ALL,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.35,
    )
    fig.text(
        0.4,
        0.006,
        "Filled: measured index; open: estimated.  "
        "Squares/thick lines: same-width Qwen column (hidden size 5120, 64 layers).",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0.015, 0.04, 0.79, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    # Accessibility check artifact: all relationships retain marker/line-style
    # encodings when color is removed.
    from PIL import Image

    with Image.open(output) as image:
        image.convert("L").save(output.with_name(f"{output.stem}_grayscale.png"))
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "title": "Mapping rank versus model capability",
                "source_data": str(DEFAULT_OUT.relative_to(REPO)),
                "public_url": (
                    "https://eps.superkaiba.com/tasks/2588/figure/mapping_rank_vs_capability.png"
                ),
                "panels": {
                    "A": "No-thinking operational rank fraction versus AA index",
                    "B": "End-of-thought operational rank fraction versus AA index",
                    "C": "No-thinking coefficient stable rank versus AA index",
                    "D": "End-of-thought coefficient stable rank versus AA index",
                },
                "accessibility": (
                    "Okabe-Ito colors plus redundant marker/fill/line encodings; "
                    "grayscale check saved alongside the figure."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render_rank_fraction_figure(results: list[dict[str, Any]], output: Path) -> None:
    """Render the focused Qwen-only rank-fraction/capability comparison."""
    results = [r for r in results if r["family"] != "OLMo"]
    colors = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}
    model_numbers = POINT_NUMBERS
    point_key = POINT_KEY_QWEN

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.9))

    # Lines encode only coherent comparisons: the Qwen3.5 size ladder and the
    # matched-width 27B release sequence.
    for arm in ("no-thinking", "end-of-thought"):
        arm_rows = [r for r in results if r["arm"] == arm]
        sequences = (
            ([r for r in arm_rows if r["model"].startswith("Q3.5")], "--", 1.0, 0.55),
            ([r for r in arm_rows if r["family"] == QWEN_COLUMN], "-", 2.0, 0.9),
        )
        for sequence, linestyle, linewidth, alpha in sequences:
            sequence = sorted(sequence, key=lambda r: r["aa_index"])
            ax.plot(
                [r["aa_index"] for r in sequence],
                [100.0 * r["operational_rank"]["rank_fraction"] for r in sequence],
                color=colors[arm],
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=1,
            )

    for record in results:
        marker = {
            "Qwen3.5": "o",
            QWEN_COLUMN: "s",
        }[record["family"]]
        x = float(record["aa_index"])
        y = 100.0 * float(record["operational_rank"]["rank_fraction"])
        face = colors[record["arm"]] if record["aa_status"] == "measured" else "white"
        ax.scatter(
            x,
            y,
            s=62,
            marker=marker,
            facecolor=face,
            edgecolor=colors[record["arm"]],
            linewidth=1.5,
            zorder=3,
        )
        offset = (4, 4)
        if record["model"] == "Q3.5 27B" and record["arm"] == "no-thinking":
            offset = (-13, 5)
        elif record["model"] == "Q3.6 27B":
            offset = (4, 8) if record["arm"] == "no-thinking" else (4, -13)
        ax.annotate(
            model_numbers[record["model"]],
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#222222",
            zorder=4,
        )

    from matplotlib.lines import Line2D

    arm_handles = [
        Line2D([0], [0], color=colors["no-thinking"], lw=2, label="Prompt read"),
        Line2D([0], [0], color=colors["end-of-thought"], lw=2, label="End-of-thought read"),
    ]
    ax.legend(
        handles=arm_handles,
        loc="upper right",
        frameon=False,
        fontsize=8,
        ncols=2,
        handlelength=2.4,
        columnspacing=1.4,
    )
    ax.set_xlim(0, 56)
    fractions = [100.0 * float(r["operational_rank"]["rank_fraction"]) for r in results]
    ax.set_ylim(max(0.0, min(fractions) - 2.0), max(fractions) + 2.0)
    ax.set_xlabel("Generic capability: Artificial Analysis Intelligence Index")
    ax.set_ylabel("Performance-preserving rank (% of hidden dimension)")
    ax.set_title(
        "Fraction of hidden width needed by Qwen mappings",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.015,
        0.035,
        "Solid + squares: same-width Qwen column (hidden size 5120, 64 layers)\n"
        "Dashed + circles: Qwen3.5 size ladder",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
        va="bottom",
    )
    fig.text(0.805, 0.76, point_key, ha="left", va="top", fontsize=7.6, linespacing=1.28)
    fig.text(
        0.42,
        0.022,
        "Rank = minimum coefficient-TSVD rank within 0.02 validation R² of the full map. "
        "Filled capability points are measured; open points are estimates.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.text(
        0.805,
        0.34,
        "Not shown: Qwen2.5-7B\n(no AA index in parent panel)",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#666666",
    )
    fig.tight_layout(rect=(0.02, 0.075, 0.79, 0.98))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(output) as image:
        image.convert("L").save(output.with_name(f"{output.stem}_grayscale.png"))
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "title": "Fraction of hidden width needed by Qwen mappings",
                "source_data": str(DEFAULT_OUT.relative_to(REPO)),
                "public_url": (
                    "https://eps.superkaiba.com/tasks/2588/figure/"
                    "rank_fraction_vs_capability_all_models.png"
                ),
                "scope": {
                    "n_maps": len(results),
                    "n_scored_checkpoints": len({r["model"] for r in results}),
                    "omitted_checkpoint": (
                        "Qwen2.5-7B anchor: no Artificial Analysis index in the parent panel"
                    ),
                },
                "rank_definition": (
                    "minimum coefficient-TSVD rank with validation R2 within 0.02 "
                    "of the full reconstructed map, divided by hidden dimension"
                ),
                "caveats": [
                    "Most Artificial Analysis values in this panel are estimates.",
                    "Panel-wide association is confounded by hidden width; the Qwen squares share hidden size 5120 and 64 layers.",
                    "Points are single-map estimates; no fit-seed or generation-seed error bars are available.",
                ],
                "accessibility": (
                    "Okabe-Ito colors plus redundant marker, fill, and line-style encodings; "
                    "grayscale check saved alongside the figure."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render_mapping_performance_figure(results: list[dict[str, Any]], output: Path) -> None:
    """Render Qwen mapping R2 and calibrated retrieval acc@1 vs capability."""
    results = [r for r in results if r["family"] != "OLMo"]
    missing = [r["key"] for r in results if "mapping_performance" not in r]
    if missing:
        raise ValueError(
            "mapping_performance is missing; run --augment-existing first: " + ", ".join(missing)
        )
    colors = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}
    model_numbers = POINT_NUMBERS
    point_key = POINT_KEY_QWEN
    panels = (
        ("test_r2", 1.0, "Held-out test R²", (0.60, 0.81)),
        (
            "test_retrieval_acc1_cos_calibrated",
            100.0,
            "Calibrated cosine retrieval acc@1 (%)",
            (60.0, 84.0),
        ),
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), sharex=True)
    for panel_index, (field, multiplier, ylabel, ylim) in enumerate(panels):
        ax = axes[panel_index]
        for arm in ("no-thinking", "end-of-thought"):
            arm_rows = [r for r in results if r["arm"] == arm]
            sequences = (
                ([r for r in arm_rows if r["model"].startswith("Q3.5")], "--", 1.0, 0.55),
                ([r for r in arm_rows if r["family"] == QWEN_COLUMN], "-", 2.0, 0.9),
            )
            for sequence, linestyle, linewidth, alpha in sequences:
                sequence = sorted(sequence, key=lambda r: r["aa_index"])
                ax.plot(
                    [r["aa_index"] for r in sequence],
                    [multiplier * r["mapping_performance"][field] for r in sequence],
                    color=colors[arm],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=1,
                )
        for record in results:
            marker = "s" if record["family"] == QWEN_COLUMN else "o"
            x = float(record["aa_index"])
            y = multiplier * float(record["mapping_performance"][field])
            face = colors[record["arm"]] if record["aa_status"] == "measured" else "white"
            ax.scatter(
                x,
                y,
                s=55,
                marker=marker,
                facecolor=face,
                edgecolor=colors[record["arm"]],
                linewidth=1.4,
                zorder=3,
            )
            offset = (4, 4)
            if panel_index == 0 and record["model"] == "Q3.5 0.8B":
                offset = (4, -12) if record["arm"] == "no-thinking" else (4, 5)
            if panel_index == 1 and record["model"] == "Q3.5 27B":
                offset = (-12, 5) if record["arm"] == "no-thinking" else (4, -12)
            ax.annotate(
                model_numbers[record["model"]],
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold",
                color="#222222",
                zorder=4,
            )
        ax.set_xlim(0, 56)
        values = [multiplier * float(r["mapping_performance"][field]) for r in results]
        pad = 0.08 * (max(values) - min(values))
        ax.set_ylim(min(values) - pad, max(values) + pad)
        ax.set_xlabel("Artificial Analysis Intelligence Index")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.02,
            0.035,
            "Solid + squares: same-width Qwen column (hidden size 5120, 64 layers)\n"
            "Dashed + circles: Qwen3.5 size ladder",
            transform=ax.transAxes,
            fontsize=7.0,
            color="#555555",
            va="bottom",
        )

    from matplotlib.lines import Line2D

    arm_handles = [
        Line2D([0], [0], color=colors["no-thinking"], lw=2, label="Prompt read"),
        Line2D([0], [0], color=colors["end-of-thought"], lw=2, label="End-of-thought read"),
    ]
    fig.suptitle(
        "Qwen mapping quality versus generic capability",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.legend(
        handles=arm_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.91),
        frameon=False,
        fontsize=8,
        ncols=2,
        handlelength=2.4,
    )
    fig.text(0.81, 0.75, point_key, ha="left", va="top", fontsize=7.6, linespacing=1.28)
    fig.text(
        0.81,
        0.34,
        "Not shown: Qwen2.5-7B\n(no AA index in parent panel)",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#666666",
    )
    fig.text(
        0.41,
        0.018,
        "Full selected-layer maps (not rank-reduced). Acc@1 subtracts the shuffled-pairing null mean. "
        "Filled capability points are measured; open points are estimates.",
        ha="center",
        fontsize=7.1,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.79, 0.86), w_pad=2.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(output) as image:
        image.convert("L").save(output.with_name(f"{output.stem}_grayscale.png"))
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "title": "Qwen mapping quality versus generic capability",
                "source_data": str(DEFAULT_OUT.relative_to(REPO)),
                "public_url": (
                    "https://eps.superkaiba.com/tasks/2588/figure/"
                    "mapping_performance_vs_capability_qwen.png"
                ),
                "scope": {
                    "n_maps": len(results),
                    "n_scored_checkpoints": len({r["model"] for r in results}),
                    "omitted_checkpoint": (
                        "Qwen2.5-7B anchor: no Artificial Analysis index in the parent panel"
                    ),
                },
                "panels": {
                    "left": "full selected-layer held-out generic-test R2",
                    "right": (
                        "full selected-layer held-out generic-test cosine retrieval acc@1 "
                        "minus shuffled-pairing null mean"
                    ),
                },
                "caveats": [
                    "Most Artificial Analysis values in this panel are estimates.",
                    "The Qwen3.5 size ladder changes width; the Qwen squares share hidden size 5120 and 64 layers.",
                    "Points are single-map estimates; no fit-seed or generation-seed error bars are available.",
                ],
                "accessibility": (
                    "Okabe-Ito colors plus redundant marker, fill, and line-style encodings; "
                    "grayscale check saved alongside the figure."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render_same_width_figure(results: list[dict[str, Any]], output: Path) -> None:
    """All hidden-size-5120 rows: rank fraction and test R2 versus capability.

    Lines connect the Qwen column (one family, one width, one depth); the OLMo
    32B rows are the width-matched cross-family points.
    """
    rows = [r for r in results if int(r["dimension"]) == SAME_WIDTH_DIM]
    if not rows:
        return
    colors = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}
    family_markers = {QWEN_COLUMN: "s", "OLMo": "^"}
    panels = (
        (
            lambda r: 100.0 * float(r["operational_rank"]["rank_fraction"]),
            "Performance-preserving rank (% of hidden dimension)",
        ),
        (lambda r: float(r["mapping_performance"]["test_r2"]), "Held-out test R²"),
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharex=True)
    for ax, (value_of, ylabel) in zip(axes, panels, strict=True):
        for arm in ("no-thinking", "end-of-thought"):
            arm_rows = [r for r in rows if r["arm"] == arm]
            column = sorted(
                (r for r in arm_rows if r["family"] == QWEN_COLUMN), key=lambda r: r["aa_index"]
            )
            ax.plot(
                [r["aa_index"] for r in column],
                [value_of(r) for r in column],
                color=colors[arm],
                linewidth=1.8,
                alpha=0.9,
                zorder=1,
            )
            for rec in arm_rows:
                x, y = float(rec["aa_index"]), value_of(rec)
                face = colors[arm] if rec["aa_status"] == "measured" else "white"
                ax.scatter(
                    x,
                    y,
                    s=58,
                    marker=family_markers.get(rec["family"], "o"),
                    facecolor=face,
                    edgecolor=colors[arm],
                    linewidth=1.4,
                    zorder=3,
                )
                ax.annotate(
                    POINT_NUMBERS.get(rec["model"], "?"),
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#222222",
                    zorder=4,
                )
        values = [value_of(r) for r in rows]
        pad = 0.08 * (max(values) - min(values) or 1.0)
        ax.set_ylim(min(values) - pad, max(values) + pad)
        ax.set_xlim(0, 56)
        ax.set_xlabel("Artificial Analysis Intelligence Index")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=colors["no-thinking"], lw=2, label="Prompt read"),
        Line2D([0], [0], color=colors["end-of-thought"], lw=2, label="End-of-thought read"),
        Line2D([0], [0], color="#444444", lw=0, marker="s", markersize=6, label="Qwen (64 layers)"),
        Line2D([0], [0], color="#444444", lw=0, marker="^", markersize=6, label="OLMo (64 layers)"),
    ]
    fig.suptitle(
        "Same-width column (hidden size 5120): mapping rank and quality versus capability",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
        fontsize=8,
        ncols=4,
        handlelength=2.2,
    )
    by_number: dict[str, list[str]] = {}
    for r in rows:
        number = POINT_NUMBERS.get(r["model"], "?")
        name = DISPLAY_NAMES.get(r["model"], r["model"])
        if name not in by_number.setdefault(number, []):
            by_number[number].append(name)
    key_lines = [
        f"{n:<3} {' / '.join(sorted(names))}"
        for n, names in sorted(by_number.items(), key=lambda t: int(t[0]) if t[0].isdigit() else 99)
    ]
    fig.text(
        0.81,
        0.78,
        "Point key\n" + "\n".join(key_lines),
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.28,
    )
    fig.text(
        0.41,
        0.018,
        "Every point has hidden size 5120 and 64 layers. Lines connect the Qwen checkpoints. "
        "Filled capability points are measured; open points are estimates.",
        ha="center",
        fontsize=7.1,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.79, 0.86), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(output) as image:
        image.convert("L").save(output.with_name(f"{output.stem}_grayscale.png"))
    output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "title": "Same-width column: mapping rank and quality versus capability",
                "source_data": str(DEFAULT_OUT.relative_to(REPO)),
                "public_url": (
                    "https://eps.superkaiba.com/tasks/2588/figure/"
                    "same_width_column_vs_capability.png"
                ),
                "scope": {
                    "n_maps": len(rows),
                    "models": sorted({r["model"] for r in rows}),
                    "hidden_size": SAME_WIDTH_DIM,
                },
                "panels": {
                    "left": "operational rank fraction (min TSVD rank within 0.02 validation R2)",
                    "right": "full selected-layer held-out generic-test R2",
                },
                "caveats": [
                    "One map per cell; no fit-seed or generation-seed error bars.",
                    "Arm-a (prompt read) rows use the model-level reasoning AA pin; Qwen3-32B's measured non-reasoning value (8) is recorded in the JSON.",
                    "Qwen ships nothing at this width between AA 13 and AA 35, so the column is a low cluster and a high cluster.",
                ],
                "accessibility": (
                    "Okabe-Ito colors plus redundant marker, fill, and line encodings; "
                    "grayscale check saved alongside the figure."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for rec in records:
        rec["family"] = LEGACY_FAMILY_LABELS.get(rec["family"], rec["family"])
    return records


def _render_all(results: list[dict[str, Any]], args) -> None:
    render_figure(results, args.figure)
    render_rank_fraction_figure(results, args.focused_figure)
    render_mapping_performance_figure(results, args.performance_figure)
    render_same_width_figure(results, args.same_width_figure)


def _build_payload(results: list[dict[str, Any]], complete: bool, trends) -> dict[str, Any]:
    return {
        "schema_version": "issue2588_mapping_rank_vs_capability_v2",
        "complete_panel": complete,
        "source": {
            "parent_issue": 2588,
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "note": "each map record carries the hf_revision it was reconstructed from",
            "frozen_layer_rule": "parent validation-selected layer_star",
            "frozen_lambda_rule": "parent validation-selected selected_lambda",
        },
        "rank_definitions": {
            "coefficient_matrix": "parent fp32 W in standardized-input coordinates",
            "stable_rank": "||W||_F^2 / ||W||_2^2",
            "operational_rank": (
                "minimum nested coefficient-TSVD rank with validation R2 >= full R2 - 0.02"
            ),
        },
        "maps": results,
        "trends": trends,
        "limitations": [
            "AA values are model-level capability attributes; the Qwen3.5 ladder and OLMo 3.1 values are estimates, the same-width extension values are measured.",
            "No-thinking maps inherit the parent experiment's AA-mode mismatch caveat.",
            "Panel-wide raw rank is dimension-confounded; rank fractions and the same-width columns are primary.",
            "Randomized SVD is used above d=2048; every operational threshold must be reached within the retained top-k basis.",
            "One rollout and one frozen layer per map; no fit-seed or generation-seed replication.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    ap.add_argument("--focused-figure", type=Path, default=DEFAULT_FOCUSED_FIGURE)
    ap.add_argument("--performance-figure", type=Path, default=DEFAULT_PERFORMANCE_FIGURE)
    ap.add_argument("--same-width-figure", type=Path, default=DEFAULT_SAME_WIDTH_FIGURE)
    ap.add_argument("--max-rank", type=int, default=1280)
    ap.add_argument("--svd-iters", type=int, default=3)
    ap.add_argument(
        "--maps",
        nargs="*",
        default=None,
        help="optional map keys/cell keys for a partial run (pair with --merge-into)",
    )
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="existing payload whose other map records are kept; the --maps records are replaced",
    )
    ap.add_argument(
        "--allow-partial-render",
        action="store_true",
        help="compute trends and render figures even when some registered maps are missing",
    )
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument(
        "--augment-existing",
        action="store_true",
        help="add saved capability/performance metrics and recompute trends without refitting/SVD",
    )
    ap.add_argument(
        "--render-focused-existing",
        action="store_true",
        help="render the focused, performance and same-width figures from --out",
    )
    args = ap.parse_args()
    if args.render_focused_existing:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        maps = _normalize_records(payload["maps"])
        render_rank_fraction_figure(maps, args.focused_figure)
        render_mapping_performance_figure(maps, args.performance_figure)
        render_same_width_figure(maps, args.same_width_figure)
        print(
            f"rendered {args.focused_figure}, {args.performance_figure}, {args.same_width_figure}"
        )
        return
    if args.augment_existing:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        by_key = {m.key: m for m in MAPS}
        for record in _normalize_records(payload["maps"]):
            spec = by_key[record["key"]]
            record["measured_capability"] = _measured_gpqa_accuracy(spec)
            record["mapping_performance"] = _mapping_performance(spec)
        payload["trends"] = summarize_trends(payload["maps"])
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if (payload.get("complete_panel") or args.allow_partial_render) and not args.no_figure:
            _render_all(payload["maps"], args)
        print(f"augmented {args.out}", flush=True)
        return
    selected = list(MAPS)
    if args.maps:
        wanted = set(args.maps)
        selected = [m for m in MAPS if m.key in wanted or m.cell in wanted]
        missing = wanted - {m.key for m in selected} - {m.cell for m in selected}
        if missing:
            raise SystemExit(f"unknown --maps values: {sorted(missing)}")
    results = [
        analyze_map(m, args.cache_dir, max_rank=args.max_rank, svd_iters=args.svd_iters)
        for m in selected
    ]
    if args.merge_into is not None:
        base = json.loads(args.merge_into.read_text(encoding="utf-8"))
        fresh_keys = {r["key"] for r in results}
        kept = [r for r in _normalize_records(base["maps"]) if r["key"] not in fresh_keys]
        order = {m.key: i for i, m in enumerate(MAPS)}
        results = sorted(kept + results, key=lambda r: order.get(r["key"], 10**6))
    complete = {r["key"] for r in results} == {m.key for m in MAPS}
    summarize = complete or args.allow_partial_render
    payload = _build_payload(results, complete, summarize_trends(results) if summarize else None)
    if not complete:
        payload["missing_maps"] = sorted({m.key for m in MAPS} - {r["key"] for r in results})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if summarize and not args.no_figure:
        _render_all(results, args)
    print(f"wrote {args.out} (complete_panel={complete})", flush=True)


if __name__ == "__main__":
    main()
