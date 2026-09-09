#!/usr/bin/env python3
"""Chat-only Qwen3-8B reproduction of the issue-2588 reduced-rank panel.

Numerical recipe: issue2588_mapping_rank_vs_capability.py at
14a9a3605c6f14e0537a7e6407ae71cf5093a972 (reconstruct_map / rrr_curves).
Only the selected-layer chat stores are consumed; no capability benchmark,
null battery, residualization, or model selection is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MANIFEST_REVISION = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"
RANK_SOURCE_SHA = "14a9a3605c6f14e0537a7e6407ae71cf5093a972"
PANEL_PREFIX = "issue2588_capability_panel_cap_long"
HF_REPO = "superkaiba1/explore-persona-space-data"
POSITIONS = {"a": "prompt_last", "b": "cot_boundary"}
SPLITS = ("train_10k", "val_400", "test_1000")
# Same parent-panel retention rule and independently established reconstruction tolerance.
RELATIVE_ERROR = 0.10
PARITY_TOLERANCE = 3e-4


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    """Parent's vectorized nested-rank SSE calculation, including rank zero."""
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


def rank_at_threshold(curve: np.ndarray, threshold: float) -> int:
    indices = np.flatnonzero(np.asarray(curve) >= threshold - 1e-12)
    if not len(indices):
        raise ValueError("No rank meets the validation error-retention threshold")
    return int(indices[0])


def load_split(cell_root: Path, split: str, layer: int, position: str, d: int):
    """Read the producer's exact row/shard set, with split-qualified identities."""
    directory = cell_root / "capture" / split
    rows = json.loads((directory / "rows.json").read_text())["rows"]
    expected_ids = [row["row_id"] for row in rows]
    shards = sorted((directory / f"L{layer:02d}").glob("shard*.npz"))
    expected_shards = [
        directory / f"L{layer:02d}" / f"shard{k:03d}.npz" for k in range((len(rows) + 499) // 500)
    ]
    if not rows or shards != expected_shards or len(set(expected_ids)) != len(rows):
        raise ValueError(f"Missing/partial/duplicate capture rows or shards: {directory}")
    ids, xs, ys = [], [], []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as z:
            row_ids = z["row_ids"]
            x, y = z[f"x_{position}"], z["y_ans"]
            if (
                x.shape != (len(row_ids), d)
                or y.shape != x.shape
                or x.dtype != np.float32
                or y.dtype != np.float32
                or not np.isfinite(x).all()
                or not np.isfinite(y).all()
            ):
                raise ValueError(f"Invalid capture tensor: {shard}")
            ids.extend(row_ids.tolist())
            xs.append(x)
            ys.append(y)
    if ids != expected_ids:
        raise ValueError(f"Capture tensor/row-manifest mismatch: {directory}")
    order = np.argsort(np.asarray(ids))
    return np.concatenate(xs)[order], np.concatenate(ys)[order]


def reconstruct(xtr, ytr, xval, yval, xte, yte, lam: float) -> dict[str, Any]:
    """Parent ridge solve at its frozen lambda; payload and predictions in fp32."""
    x64, y64 = xtr.astype(np.float64), ytr.astype(np.float64)
    xmu64, xsd64 = x64.mean(axis=0), x64.std(axis=0, ddof=1) + 1e-9
    ymu64 = y64.mean(axis=0)
    xn, yc = (x64 - xmu64) / xsd64, y64 - ymu64
    gram, cross = xn.T @ xn, xn.T @ yc
    gram.flat[:: xtr.shape[1] + 1] += lam
    w64 = scipy.linalg.solve(
        gram, cross, assume_a="pos", overwrite_a=True, overwrite_b=True, check_finite=False
    )
    w = np.asarray(w64, dtype=np.float32)
    xmu, xsd, ymu = (np.asarray(a, dtype=np.float32) for a in (xmu64, xsd64, ymu64))
    return {
        "W": w,
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
        "pred_val": ((xval - xmu) / xsd) @ w + ymu,
        "target_val": yval,
        "pred_test": ((xte - xmu) / xsd) @ w + ymu,
        "target_test": yte,
    }


def reduced_rank(payload: dict, xtr: np.ndarray, *, include_spectrum: bool = False) -> dict:
    """Exact training-output PCA, not coefficient SVD or a test-chosen rank."""
    w = np.asarray(payload["W"], dtype=np.float64)
    xmu, xsd, ymu = (np.asarray(payload[k], dtype=np.float64) for k in ("xmu", "xsd", "ymu"))
    xn = (xtr.astype(np.float64) - xmu) / xsd
    gram = xn.T @ xn
    m = w.T @ gram @ w
    m = 0.5 * (m + m.T)
    evals, evecs = scipy.linalg.eigh(m, check_finite=False)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None) / len(xtr)
    right = np.ascontiguousarray(evecs[:, order], dtype=np.float32)
    curves = {}
    full = {}
    for split in ("val", "test"):
        pred, target = payload[f"pred_{split}"], payload[f"target_{split}"]
        full[split] = pooled_r2(pred, target)
        curves[split] = r2_curve_from_top_right_vectors(pred, target, ymu, right)
        if abs(float(curves[split][-1]) - full[split]) > 1e-4:
            raise ValueError(f"Full-rank reconstruction failed on {split}")
    threshold = 1.0 - (1.0 + RELATIVE_ERROR) * (1.0 - full["val"])
    rank = rank_at_threshold(curves["val"], threshold)
    total_var = float(evals.sum())
    cum = np.cumsum(evals) / (total_var + 1e-30)
    return {
        "rank": rank,
        "rank_relative_to_dimension": rank / w.shape[0],
        "validation_r2_threshold": threshold,
        "full_validation_r2": full["val"],
        "full_test_r2": full["test"],
        "selected_rank_validation_r2": float(curves["val"][rank]),
        "selected_rank_test_r2": float(curves["test"][rank]),
        "rank_curve": {"validation_r2": curves["val"].tolist(), "test_r2": curves["test"].tolist()},
        "fitted_output_spectrum": {
            **({"eigenvalues": evals.tolist()} if include_spectrum else {}),
            "eigenvalues_top64": evals[:64].tolist(),
            "total_variance": total_var,
            "directions_for_90pct_variance": int(np.searchsorted(cum, 0.90) + 1),
            "directions_for_99pct_variance": int(np.searchsorted(cum, 0.99) + 1),
        },
    }


def input_manifest(cell_root: Path, position: str, layer: int) -> dict:
    files = [cell_root / "run_identity.json", cell_root / "fits" / f"fits_{position}.json"]
    for split in SPLITS:
        directory = cell_root / "capture" / split
        files.append(directory / "rows.json")
        files.extend(sorted((directory / f"L{layer:02d}").glob("shard*.npz")))
    return {
        str(p.relative_to(cell_root)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    }


def verify_durable_inputs(
    cell_root: Path, arm: str, run_id: str, revision: str, manifest: dict, api=None
) -> dict:
    """Verify exact local bytes against one immutable Hub commit before any fit.

    LFS files use their SHA256 content identity; ordinary files use Git's
    blob-domain SHA1 (header plus bytes), not a bare file SHA1.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi() if api is None else api
    prefix = f"{PANEL_PREFIX}/generic/{run_id}/q3_8b/{'nothink' if arm == 'a' else 'think'}"
    paths = {
        f"{prefix}/{'analysis_tensors/' if rel.startswith('capture/') else ''}{rel}": rel
        for rel in manifest
    }
    print(f"[rank_verify] {len(paths)} required input files at {revision}", flush=True)
    entries = hub.retry_transient(
        lambda: api.get_paths_info(HF_REPO, list(paths), repo_type="dataset", revision=revision),
        what=f"verify rank input bytes {arm} at {revision}",
    )
    actual = {entry.path: entry for entry in entries}
    if set(actual) != set(paths):
        raise ValueError(f"Durable rank-input name-set mismatch: {set(paths) - set(actual)}")
    for remote, rel in paths.items():
        entry = actual[remote]
        expected = manifest[rel]
        if entry.size != expected["bytes"]:
            raise ValueError(f"Durable rank-input size mismatch: {remote}")
        if entry.lfs is not None:
            verified = entry.lfs.sha256 == expected["sha256"]
        else:
            digest = hashlib.sha1(f"blob {entry.size}\0".encode())
            with (cell_root / rel).open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            verified = digest.hexdigest() == entry.blob_id
        if not verified:
            raise ValueError(f"Durable rank-input content mismatch: {remote}")
    return {
        "repo": HF_REPO,
        "revision": revision,
        "prefix": prefix,
        "verified_files": len(paths),
        "content_verified": True,
    }


def analyze_cell(cell_root: Path, arm: str, run_id: str, hf_revision: str, output: Path) -> dict:
    started = time.monotonic()
    position = POSITIONS[arm]
    identity = json.loads((cell_root / "run_identity.json").read_text())
    expected = {
        "surface": "generic",
        "run_id": run_id,
        "cell": f"q3_8b_{arm}",
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": MODEL_REVISION,
        "manifest_revision": MANIFEST_REVISION,
        "smoke": False,
        "cap_profile": "long",
        "layer_set": "swept",
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unreviewed source identity: {cell_root}")
    fit = json.loads((cell_root / "fits" / f"fits_{position}.json").read_text())
    if fit.get("identity") != identity or fit.get("input_position") != position:
        raise ValueError("Fit/run identity or input position mismatch")
    layer = int(fit["layer_star"])
    star = fit["layers"][str(layer)]
    if int(star["d"]) != 4096 or set(map(int, fit["layers"])) != {*range(0, 36, 2), 35}:
        raise ValueError("Incomplete or wrong model/layer grid")
    selected = max(
        sorted(map(int, fit["layers"])),
        key=lambda key: float(
            fit["layers"][str(key)]["knn_val"]["ridge"]["cosine"]["acc_at_k"]["1"]
        ),
    )
    if layer != int(selected):
        raise ValueError("Layer selection does not match validation raw-cosine retrieval")
    inputs = input_manifest(cell_root, position, layer)
    durable = verify_durable_inputs(cell_root, arm, run_id, hf_revision, inputs)
    provenance = {
        "run_identity": identity,
        "hf_revision": hf_revision,
        "input_files": inputs,
        "durable_verification": durable,
        "rank_source_sha": RANK_SOURCE_SHA,
        "consumer_sha256": sha256(Path(__file__)),
    }
    if output.is_file():
        previous = json.loads(output.read_text())
        if previous.get("provenance") != provenance:
            raise ValueError(f"Stale rank output; use a new output path: {output}")
        return previous
    print(f"[rank] {arm} selected layer={layer}; loading three chat splits", flush=True)
    pairs = [load_split(cell_root, split, layer, position, 4096) for split in SPLITS]
    payload = reconstruct(
        *[a for pair in pairs for a in pair], lam=float(star["fit_meta"]["selected_lambda"])
    )
    for split, expected_r2 in (
        ("val", star["fit_meta"]["val_r2_at_selected"]),
        ("test", star["test_r2"]),
    ):
        actual = pooled_r2(payload[f"pred_{split}"], payload[f"target_{split}"])
        if abs(actual - float(expected_r2)) > PARITY_TOLERANCE:
            raise ValueError(
                f"Parent reconstruction parity failed: {split} {actual} vs {expected_r2}"
            )
    result = reduced_rank(payload, pairs[0][0])
    result.update(
        {
            "cell": f"q3_8b_{arm}",
            "input_position": position,
            "dimension": 4096,
            "layer_star": layer,
            "realized_rows": {s: len(p[0]) for s, p in zip(SPLITS, pairs)},
            "selected_lambda": float(star["fit_meta"]["selected_lambda"]),
            "parent_selected_layer_metrics": star,
            "participation_ratio_x_at_star": fit["participation_ratio_x_at_star"],
            "ceiling_two_draw_at_star": fit["ceiling_two_draw_at_star"],
            "ceiling_retrieval_at_star": fit["ceiling_retrieval_at_star"],
            "method": "training-output PCA reduced-rank ridge; validation SSE <= 1.10 full-map SSE",
            "provenance": provenance,
            "elapsed_s": time.monotonic() - started,
        }
    )
    write_json(output, result)
    print(
        f"[rank] {arm} complete rank={result['rank']} elapsed={result['elapsed_s']:.1f}s",
        flush=True,
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing generic/RUN/cells_cap_long/q3_8b_{a,b}",
    )
    parser.add_argument("--run-id", default="qwen3-chat-v2")
    parser.add_argument("--hf-revision", required=True, help="Verified durable capture/fit commit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=tuple(POSITIONS),
        help="Analyze just one map (use a first for the measured CPU pilot)",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.hf_revision):
        parser.error("--hf-revision must be the immutable verified data commit")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        parser.error("unsafe run ID")
    load_dotenv()
    maps = []
    for arm in [args.arm] if args.arm else POSITIONS:
        root = args.source_root / "generic" / args.run_id / "cells_cap_long" / f"q3_8b_{arm}"
        maps.append(
            analyze_cell(
                root, arm, args.run_id, args.hf_revision, args.output_dir / f"rank_{arm}.json"
            )
        )
    if args.arm:
        return 0
    summary = {
        "schema": "issue2588_chat_rank_v1",
        "maps": maps,
        "rank_change_thinking_minus_no_thinking": maps[1]["rank"] - maps[0]["rank"],
        "split_caveat": "Original pinned splits retained: 13 distinct exact prompt strings "
        "appear in validation and test (24 validation rows and 60/1000 test rows). "
        "No exact prompt overlap with train. Input positions and answer targets differ "
        "between the no-thinking and thinking conditions; this is not an identical-target pair.",
    }
    write_json(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
