#!/usr/bin/env python3
"""Issue #2643: factorized context-SAE -> answer-SAE map and anomaly screen.

The production map is deliberately factorized instead of materializing the
full answer-feature-by-context-feature matrix (32,768 x 32,768 for the
exact-replication SAEs)::

    z_context --D_context--> x_context_hat --ridge--> x_answer_hat
              --E_answer + threshold--> z_answer_hat --scale--> z_answer_cal

The SAEs and dense ridge were fitted on independent, large corpora.  A single
non-negative, slope-only calibration per answer feature is fitted *only* on
ordinary training rows from issue #2502.  Validation/test rows and every weird
regime remain held out.  The script emits content-opaque row scores and source
aggregates; it never downloads or writes prompt/completion text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

try:  # importable both as ``scripts.issue2643_sae_map`` and as a direct script
    from scripts.issue2476_turnavg_sae import MatryoshkaBatchTopKSAE
except ModuleNotFoundError:  # pragma: no cover - exercised by pod CLI invocation
    from issue2476_turnavg_sae import MatryoshkaBatchTopKSAE


ISSUE = 2643
DATA_REPO = "superkaiba1/explore-persona-space-data"
DATA_REVISION = "cd80ba2588bb6d4291edf621176ea654bcbf2507"
CAPTURE_PREFIX = "issue2502_ctxmap_xgen/analysis_tensors/modelA"
CTX_SAE_PREFIX = "issue2552_derreplication/exactrep/analysis_tensors/sae_ctx_rep"
ANS_SAE_PREFIX = "issue2552_derreplication/exactrep/analysis_tensors/sae_rep"
RIDGE_REPO_PATH = "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
DEFAULT_RIDGE_LOCAL = Path(
    "data/issue_2094/joint_transport/banked_maps/issue779_monitoring/"
    "n1m_readout/weights/L19/ridge.pt"
)


def decode_bf16_uint16(array: np.ndarray) -> torch.Tensor:
    """Decode the issue-2502 ``bf16_as_uint16`` lossless wire format."""
    a = np.asarray(array)
    if a.dtype != np.uint16:
        raise TypeError(f"expected uint16 bf16 bits, got {a.dtype}")
    signed = np.ascontiguousarray(a).view(np.int16)
    return torch.from_numpy(signed.copy()).view(torch.bfloat16).float()


def apply_dense_ridge(x: torch.Tensor, ridge: Mapping[str, object]) -> torch.Tensor:
    """Apply the canonical #779 standardized-input ridge equation."""
    required = {"xmu", "xsd", "ymu", "W"}
    missing = required - set(ridge)
    if missing:
        raise KeyError(f"ridge missing keys: {sorted(missing)}")
    device = x.device
    xmu = torch.as_tensor(ridge["xmu"], dtype=torch.float32, device=device)
    xsd = torch.as_tensor(ridge["xsd"], dtype=torch.float32, device=device)
    ymu = torch.as_tensor(ridge["ymu"], dtype=torch.float32, device=device)
    w = torch.as_tensor(ridge["W"], dtype=torch.float32, device=device)
    if x.shape[-1] != xmu.numel() or w.shape != (xmu.numel(), ymu.numel()):
        raise ValueError(
            f"ridge/input shape mismatch: x={tuple(x.shape)}, xmu={tuple(xmu.shape)}, "
            f"ymu={tuple(ymu.shape)}, W={tuple(w.shape)}"
        )
    if torch.any(xsd <= 0):
        raise ValueError("ridge xsd must be strictly positive")
    return ((x.float() - xmu) / xsd) @ w + ymu


def feature_scale_fit(
    pred_cross: torch.Tensor,
    pred_square: torch.Tensor,
    *,
    ridge_to_identity: float = 1.0,
    max_scale: float = 8.0,
) -> torch.Tensor:
    """Fit independent non-negative slopes, shrunk toward identity.

    ``pred_cross`` is sum(pred * target), and ``pred_square`` is sum(pred^2).
    Shrinking toward scale=1 avoids arbitrary values for rarely predicted
    features and adds no intercept that would destroy sparse support.
    """
    if pred_cross.shape != pred_square.shape:
        raise ValueError("feature calibration accumulators must have equal shape")
    if ridge_to_identity < 0 or max_scale <= 0:
        raise ValueError("invalid feature calibration hyperparameters")
    scale = (pred_cross.double() + ridge_to_identity) / (pred_square.double() + ridge_to_identity)
    return scale.clamp_(0.0, max_scale).float()


def feature_scale_apply(pred: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if pred.shape[-1] != scale.numel():
        raise ValueError(f"scale mismatch: {pred.shape[-1]} != {scale.numel()}")
    return pred * scale.to(device=pred.device, dtype=pred.dtype)


def pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or target.ndim != 2:
        raise ValueError(f"pooled_r2 expects equal 2-D arrays, got {pred.shape}, {target.shape}")
    sse = float(np.square(pred - target).sum())
    sst = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    return float("nan") if sst <= 0 else 1.0 - sse / sst


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-correct AUROC; returns NaN for a single-class input."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    good = np.isfinite(s)
    y, s = y[good], s[good]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision with positives as the weird/behavior class."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    good = np.isfinite(s)
    y, s = y[good], s[good]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order] == 1
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / n_pos)


def row_scores(
    x_context: torch.Tensor,
    x_context_recon: torch.Tensor,
    x_answer: torch.Tensor,
    x_answer_pred_raw: torch.Tensor,
    x_answer_pred_sae: torch.Tensor,
    z_answer: torch.Tensor,
    z_answer_pred: torch.Tensor,
    *,
    pred_code_mean: torch.Tensor | None = None,
    pred_code_var: torch.Tensor | None = None,
    pred_code_count: torch.Tensor | None = None,
    rarity_min_count: int = 32,
) -> dict[str, torch.Tensor]:
    """Compute pre-answer and post-answer diagnostics for each paired row."""
    eps = 1e-8

    def rel_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).square().sum(1) / b.square().sum(1).clamp_min(eps)

    def cosine_surprise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        denom = a.norm(dim=1) * b.norm(dim=1)
        cos = (a * b).sum(1) / denom.clamp_min(eps)
        both_zero = (a.norm(dim=1) == 0) & (b.norm(dim=1) == 0)
        return torch.where(both_zero, torch.zeros_like(cos), 1.0 - cos)

    true_on = z_answer > 0
    pred_on = z_answer_pred > 0
    overlap = (true_on & pred_on).sum(1)
    support_recall = overlap / true_on.sum(1).clamp_min(1)
    support_precision = overlap / pred_on.sum(1).clamp_min(1)
    scores = {
        # Available before generation completes.
        "forecast_context_recon_nse": rel_sq(x_context_recon, x_context),
        "forecast_mapped_answer_norm": x_answer_pred_sae.norm(dim=1),
        "forecast_pred_l0": pred_on.sum(1).float(),
        # Available after the realized answer is captured.
        "post_dense_surprise_raw": rel_sq(x_answer_pred_raw, x_answer),
        "post_dense_surprise_ctxsae": rel_sq(x_answer_pred_sae, x_answer),
        "post_code_cosine_surprise": cosine_surprise(z_answer_pred, z_answer),
        "post_code_relative_l2": rel_sq(z_answer_pred, z_answer),
        "post_support_recall": support_recall.float(),
        "post_support_precision": support_precision.float(),
        "post_emergent_feature_mass": torch.relu(z_answer - z_answer_pred).sum(1)
        / z_answer.sum(1).clamp_min(eps),
        "control_answer_l0": true_on.sum(1).float(),
    }
    if pred_code_mean is not None or pred_code_var is not None or pred_code_count is not None:
        if pred_code_mean is None or pred_code_var is None or pred_code_count is None:
            raise ValueError("rarity needs mean, variance, and count together")
        valid = (pred_code_count >= rarity_min_count) & (pred_code_var > 1e-10)
        if not bool(valid.any()):
            scores["forecast_code_rarity"] = torch.full(
                (z_answer_pred.shape[0],), float("nan"), device=z_answer_pred.device
            )
        else:
            mu = pred_code_mean.to(z_answer_pred.device)[valid]
            var = pred_code_var.to(z_answer_pred.device)[valid]
            scores["forecast_code_rarity"] = (
                (z_answer_pred[:, valid] - mu).square() / var.clamp_min(1e-8)
            ).mean(1)
    return scores


@dataclass
class FactorizedSAEMap:
    context_sae: MatryoshkaBatchTopKSAE
    answer_sae: MatryoshkaBatchTopKSAE
    ridge: Mapping[str, object]
    scale: torch.Tensor | None = None

    @torch.no_grad()
    def predict(self, x_context: torch.Tensor) -> dict[str, torch.Tensor]:
        zc = self.context_sae.encode(x_context)
        xc_recon = self.context_sae.decode(zc)
        ya_raw = apply_dense_ridge(x_context, self.ridge)
        ya_sae = apply_dense_ridge(xc_recon, self.ridge)
        za = self.answer_sae.encode(ya_sae)
        if self.scale is not None:
            za = feature_scale_apply(za, self.scale)
        return {
            "z_context": zc,
            "x_context_recon": xc_recon,
            "x_answer_pred_raw": ya_raw,
            "x_answer_pred_sae": ya_sae,
            "z_answer_pred": za,
        }


class RunningFeatureCalibration:
    def __init__(self, n_features: int):
        self.cross = torch.zeros(n_features, dtype=torch.float64)
        self.square = torch.zeros(n_features, dtype=torch.float64)
        self.sum = torch.zeros(n_features, dtype=torch.float64)
        self.sumsq = torch.zeros(n_features, dtype=torch.float64)
        self.fire_count = torch.zeros(n_features, dtype=torch.int64)
        self.n = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        p, t = pred.detach().double().cpu(), target.detach().double().cpu()
        self.cross += (p * t).sum(0)
        self.square += p.square().sum(0)
        self.sum += p.sum(0)
        self.sumsq += p.square().sum(0)
        self.fire_count += (p > 0).sum(0)
        self.n += int(p.shape[0])

    def finish(self, ridge_to_identity: float, max_scale: float) -> dict[str, torch.Tensor]:
        if self.n == 0:
            raise RuntimeError("no ordinary training rows reached feature calibration")
        scale = feature_scale_fit(
            self.cross,
            self.square,
            ridge_to_identity=ridge_to_identity,
            max_scale=max_scale,
        )
        mean = (self.sum / self.n).float()
        var = (self.sumsq / self.n - (self.sum / self.n).square()).clamp_min_(0).float()
        # Rarity is computed after scaling, so transform moments too.
        return {
            "scale": scale,
            "pred_mean": mean * scale,
            "pred_var": var * scale.square(),
            "pred_count": self.fire_count,
            "n": torch.tensor(self.n),
        }


class VectorSums:
    """Streaming pooled R2 sufficient statistics without retaining vectors."""

    def __init__(self, dim: int):
        self.n = 0
        self.target_sum = torch.zeros(dim, dtype=torch.float64)
        self.target_sumsq = 0.0
        self.sse: defaultdict[str, float] = defaultdict(float)

    def update(self, target: torch.Tensor, **predictions: torch.Tensor) -> None:
        t = target.detach().double().cpu()
        self.n += int(t.shape[0])
        self.target_sum += t.sum(0)
        self.target_sumsq += float(t.square().sum())
        for name, pred in predictions.items():
            self.sse[name] += float((pred.detach().double().cpu() - t).square().sum())

    def r2(self) -> dict[str, float]:
        if self.n == 0:
            return {k: float("nan") for k in self.sse}
        sst = self.target_sumsq - float(self.target_sum.square().sum()) / self.n
        return {
            k: (float("nan") if sst <= 0 else 1.0 - v / sst) for k, v in sorted(self.sse.items())
        }


def _atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _safe_float(x: torch.Tensor | float) -> float:
    value = float(x)
    return value if math.isfinite(value) else float("nan")


def _load_ridge(path: Path) -> dict[str, object]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"ridge artifact must be a dict, got {type(obj).__name__}")
    if int(obj.get("layer", -1)) != 19:
        raise ValueError(f"expected L19 ridge, got layer={obj.get('layer')}")
    return obj


def _download_file(filename: str, local_dir: Path, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            DATA_REPO,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )
    )


def _resolve_assets(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(args.work_dir) / "assets"
    root.mkdir(parents=True, exist_ok=True)

    def sae_dir(local: str, prefix: str, leaf: str) -> Path:
        d = Path(local) if local else root / leaf
        if not (d / "cfg.json").exists() or not (d / "sae_weights.safetensors").exists():
            d.mkdir(parents=True, exist_ok=True)
            for name in ("cfg.json", "sae_weights.safetensors"):
                src = _download_file(f"{prefix}/{name}", root / "hf", args.data_revision)
                shutil.copy2(src, d / name)
        return d

    ctx = sae_dir(args.context_sae, args.context_sae_prefix, "context_sae")
    ans = sae_dir(args.answer_sae, args.answer_sae_prefix, "answer_sae")
    ridge = Path(args.ridge) if args.ridge else DEFAULT_RIDGE_LOCAL
    if not ridge.exists():
        ridge = _download_file(args.ridge_repo_path, root / "hf", args.data_revision)
    return ctx, ans, ridge


def _chunk_pairs(prefix: str, revision: str, max_chunks: int = 0) -> list[tuple[str, str]]:
    from huggingface_hub import HfApi

    npz: dict[str, str] = {}
    rows: dict[str, str] = {}
    for entry in HfApi().list_repo_tree(
        DATA_REPO,
        repo_type="dataset",
        revision=revision,
        path_in_repo=prefix,
        recursive=True,
        expand=False,
    ):
        path = entry.path
        leaf = Path(path).name
        if leaf.endswith("__L19.npz"):
            npz[leaf.split("__", 1)[0]] = path
        elif leaf.endswith("__rows.json"):
            rows[leaf.split("__", 1)[0]] = path
    if set(npz) != set(rows):
        raise RuntimeError(
            f"capture pair mismatch: tensors_only={sorted(set(npz) - set(rows))[:5]}, "
            f"rows_only={sorted(set(rows) - set(npz))[:5]}"
        )
    keys = sorted(npz)
    if max_chunks:
        keys = keys[:max_chunks]
    if not keys:
        raise RuntimeError(f"no L19 capture chunks under {prefix} @ {revision}")
    return [(npz[k], rows[k]) for k in keys]


def _iter_capture(
    pairs: Sequence[tuple[str, str]],
    *,
    download_dir: Path,
    revision: str,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[dict]]]:
    for i, (npz_name, rows_name) in enumerate(pairs):
        npz_path = _download_file(npz_name, download_dir, revision)
        rows_path = _download_file(rows_name, download_dir, revision)
        with np.load(npz_path) as z:
            x = decode_bf16_uint16(z["cx_last"])
            y = decode_bf16_uint16(z["vx"])
        doc = json.loads(rows_path.read_text(encoding="utf-8"))
        rows = doc["rows"]
        if x.shape != y.shape or len(rows) != x.shape[0]:
            raise RuntimeError(
                f"capture alignment failure at chunk {i}: x={tuple(x.shape)}, "
                f"y={tuple(y.shape)}, rows={len(rows)}"
            )
        for j, row in enumerate(rows):
            if int(row.get("row", -1)) != j:
                raise RuntimeError(f"row-order failure at chunk {i}, row {j}")
        yield x, y, rows


def _fit_calibration(
    mapper: FactorizedSAEMap,
    pairs: Sequence[tuple[str, str]],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    acc = RunningFeatureCalibration(mapper.answer_sae.dict_size)
    n_seen = 0
    for ci, (x, y, rows) in enumerate(
        _iter_capture(
            pairs, download_dir=Path(args.work_dir) / "capture", revision=args.data_revision
        )
    ):
        keep = [
            i
            for i, row in enumerate(rows)
            if row.get("split") == "train" and row.get("regime_class") == "ordinary"
        ]
        for start in range(0, len(keep), args.batch_size):
            idx = keep[start : start + args.batch_size]
            if not idx:
                continue
            xb = x[idx].to(args.device)
            yb = y[idx].to(args.device)
            pred = mapper.predict(xb)["z_answer_pred"]
            target = mapper.answer_sae.encode(yb)
            acc.update(pred, target)
            n_seen += len(idx)
        if (ci + 1) % 20 == 0 or ci + 1 == len(pairs):
            print(
                f"[fit-calibration] chunks={ci + 1}/{len(pairs)} ordinary_train_rows={n_seen}",
                flush=True,
            )
    return acc.finish(args.calibration_ridge, args.max_scale)


def _score_screen(
    mapper: FactorizedSAEMap,
    calibration: Mapping[str, torch.Tensor],
    pairs: Sequence[tuple[str, str]],
    args: argparse.Namespace,
) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_tmp = out / ".row_scores.jsonl.tmp"
    dense_all = VectorSums(mapper.answer_sae.act_dim)
    code_all = VectorSums(mapper.answer_sae.dict_size)
    dense_test = VectorSums(mapper.answer_sae.act_dim)
    code_test = VectorSums(mapper.answer_sae.dict_size)
    source_values: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_meta: dict[str, dict[str, str]] = {}
    auc_labels: list[int] = []
    auc_scores: defaultdict[str, list[float]] = defaultdict(list)
    n_rows = 0

    with rows_tmp.open("w", encoding="utf-8") as f:
        for ci, (x, y, rows) in enumerate(
            _iter_capture(
                pairs, download_dir=Path(args.work_dir) / "capture", revision=args.data_revision
            )
        ):
            for start in range(0, len(rows), args.batch_size):
                stop = min(start + args.batch_size, len(rows))
                meta = rows[start:stop]
                xb = x[start:stop].to(args.device)
                yb = y[start:stop].to(args.device)
                pred = mapper.predict(xb)
                za = mapper.answer_sae.encode(yb)
                scores = row_scores(
                    xb,
                    pred["x_context_recon"],
                    yb,
                    pred["x_answer_pred_raw"],
                    pred["x_answer_pred_sae"],
                    za,
                    pred["z_answer_pred"],
                    pred_code_mean=calibration["pred_mean"],
                    pred_code_var=calibration["pred_var"],
                    pred_code_count=calibration["pred_count"],
                    rarity_min_count=args.rarity_min_count,
                )
                dense_all.update(
                    yb,
                    raw_ridge=pred["x_answer_pred_raw"],
                    context_sae_ridge=pred["x_answer_pred_sae"],
                )
                code_all.update(za, calibrated_context_sae_map=pred["z_answer_pred"])
                test_idx = [i for i, row in enumerate(meta) if row.get("split") == "test"]
                if test_idx:
                    dense_test.update(
                        yb[test_idx],
                        raw_ridge=pred["x_answer_pred_raw"][test_idx],
                        context_sae_ridge=pred["x_answer_pred_sae"][test_idx],
                    )
                    code_test.update(
                        za[test_idx],
                        calibrated_context_sae_map=pred["z_answer_pred"][test_idx],
                    )
                cpu_scores = {k: v.detach().float().cpu().numpy() for k, v in scores.items()}
                for j, row in enumerate(meta):
                    rec = {
                        "context_id": str(row["context_id"]),
                        "split": str(row.get("split")),
                        "regime_class": str(row.get("regime_class")),
                        "lodo_group": str(row.get("lodo_group")),
                        "source_tag": str(row.get("source_tag")),
                        "prompt_len": int(row.get("prompt_len", 0)),
                        "answer_len": int(row.get("n_gen_tokens", 0)),
                    }
                    for name, values in cpu_scores.items():
                        rec[name] = float(values[j])
                    f.write(json.dumps(rec, sort_keys=True, allow_nan=True) + "\n")
                    source = rec["source_tag"]
                    source_meta[source] = {
                        "regime_class": rec["regime_class"],
                        "lodo_group": rec["lodo_group"],
                    }
                    for name in cpu_scores:
                        source_values[source][name].append(rec[name])
                    source_values[source]["prompt_len"].append(float(rec["prompt_len"]))
                    source_values[source]["answer_len"].append(float(rec["answer_len"]))
                    if rec["split"] == "test":
                        auc_labels.append(int(rec["regime_class"] != "ordinary"))
                        for name in (
                            "forecast_context_recon_nse",
                            "forecast_code_rarity",
                            "post_dense_surprise_raw",
                            "post_dense_surprise_ctxsae",
                            "post_code_cosine_surprise",
                            "post_code_relative_l2",
                            "post_emergent_feature_mass",
                            "control_answer_l0",
                        ):
                            auc_scores[name].append(rec[name])
                    n_rows += 1
            if (ci + 1) % 20 == 0 or ci + 1 == len(pairs):
                print(f"[score] chunks={ci + 1}/{len(pairs)} rows={n_rows}", flush=True)
    os.replace(rows_tmp, out / "row_scores.jsonl")

    per_source = {}
    for source, metrics in sorted(source_values.items()):
        per_source[source] = {
            **source_meta[source],
            "n": len(next(iter(metrics.values()))),
            "metrics": {
                name: {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }
                for name, values in sorted(metrics.items())
            },
        }
    detection = {
        name: {
            "positive_class": "regime_class != ordinary",
            "n": len(auc_labels),
            "prevalence": float(np.mean(auc_labels)) if auc_labels else float("nan"),
            "auroc": binary_auroc(auc_labels, values),
            "average_precision": binary_average_precision(auc_labels, values),
        }
        for name, values in sorted(auc_scores.items())
    }
    summary = {
        "issue": ISSUE,
        "status": "descriptive_screen_complete",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "layer": 19,
        "map": {
            "form": "answer_encode(ridge(context_decode(context_encode(x)))) * feature_scale",
            "rank_ceiling": mapper.answer_sae.act_dim,
            "context_sae_dict": mapper.context_sae.dict_size,
            "answer_sae_dict": mapper.answer_sae.dict_size,
            "calibration": "nonnegative slope-only per answer feature, ordinary train only",
            "calibration_n": int(calibration["n"]),
            "calibration_ridge_to_identity": args.calibration_ridge,
            "max_scale": args.max_scale,
            "n_scaled_features": int((calibration["scale"] != 1).sum()),
        },
        "provenance": {
            "data_repo": DATA_REPO,
            "data_revision": args.data_revision,
            "capture_prefix": args.capture_prefix,
            "context_sae_prefix": args.context_sae_prefix,
            "answer_sae_prefix": args.answer_sae_prefix,
            "ridge_repo_path": args.ridge_repo_path,
            "ridge_training_rows": 963_444,
            "heldout_policy": "all non-ordinary rows excluded from calibration",
        },
        "n_rows": n_rows,
        "n_chunks": len(pairs),
        "map_quality": {
            "all_dense_r2": dense_all.r2(),
            "all_code_r2": code_all.r2(),
            "test_dense_r2": dense_test.r2(),
            "test_code_r2": code_test.r2(),
        },
        "descriptive_weird_detection_test_only": detection,
        "interpretation_limits": [
            "Regime labels describe prompt/source families, not verified model behavior.",
            "Pre-answer scores are forecasts; post-answer residual scores require the realized answer.",
            "Behavior claims require held-out labeled model-organism panels and matched raw-context baselines.",
        ],
        "per_source": per_source,
    }
    _atomic_json(out / "summary.json", summary)
    torch.save({k: v.cpu() for k, v in calibration.items()}, out / "feature_calibration.pt")
    return summary


def phase_screen(args: argparse.Namespace) -> None:
    ctx_dir, ans_dir, ridge_path = _resolve_assets(args)
    print(f"[assets] context_sae={ctx_dir} answer_sae={ans_dir} ridge={ridge_path}", flush=True)
    context_sae = MatryoshkaBatchTopKSAE.load_local(ctx_dir, device=args.device)
    answer_sae = MatryoshkaBatchTopKSAE.load_local(ans_dir, device=args.device)
    ridge = _load_ridge(ridge_path)
    for key in ("xmu", "xsd", "ymu", "W"):
        ridge[key] = torch.as_tensor(ridge[key], dtype=torch.float32, device=args.device)
    if context_sae.act_dim != answer_sae.act_dim or context_sae.act_dim != ridge["xmu"].numel():
        raise ValueError("SAE/ridge activation dimensions disagree")
    mapper = FactorizedSAEMap(context_sae, answer_sae, ridge)
    pairs = _chunk_pairs(args.capture_prefix, args.data_revision, args.max_chunks)
    calibration = _fit_calibration(mapper, pairs, args)
    mapper.scale = calibration["scale"].to(args.device)
    summary = _score_screen(mapper, calibration, pairs, args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "n_rows": summary["n_rows"],
                "map_quality": summary["map_quality"],
                "descriptive_weird_detection_test_only": summary[
                    "descriptive_weird_detection_test_only"
                ],
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


def phase_selfcheck() -> None:
    torch.manual_seed(2643)
    dim, features, n = 6, 12, 48
    context = MatryoshkaBatchTopKSAE(
        act_dim=dim, dict_size=features, k=3, tier_bounds=(4, 8, 12), seed=1
    ).eval()
    answer = MatryoshkaBatchTopKSAE(
        act_dim=dim, dict_size=features, k=3, tier_bounds=(4, 8, 12), seed=2
    ).eval()
    with torch.no_grad():
        context.threshold.zero_()
        answer.threshold.zero_()
    ridge = {
        "xmu": torch.zeros(dim),
        "xsd": torch.ones(dim),
        "ymu": torch.zeros(dim),
        "W": torch.eye(dim),
        "layer": 19,
    }
    mapper = FactorizedSAEMap(context, answer, ridge)
    x = torch.randn(n, dim)
    pred = mapper.predict(x)
    assert pred["z_context"].shape == (n, features)
    assert pred["z_answer_pred"].shape == (n, features)
    target = answer.encode(x)
    acc = RunningFeatureCalibration(features)
    acc.update(pred["z_answer_pred"], target)
    cal = acc.finish(1.0, 8.0)
    assert torch.isfinite(cal["scale"]).all()
    s = row_scores(
        x,
        pred["x_context_recon"],
        x,
        pred["x_answer_pred_raw"],
        pred["x_answer_pred_sae"],
        target,
        pred["z_answer_pred"],
        pred_code_mean=cal["pred_mean"],
        pred_code_var=cal["pred_var"],
        pred_code_count=cal["pred_count"],
        rarity_min_count=1,
    )
    assert all(v.shape == (n,) for v in s.values())
    print("SELF_CHECK_OK")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("selfcheck", "screen"), required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-chunks", type=int, default=0, help="0 means all 300 chunks")
    p.add_argument("--work-dir", default="/mnt/eps-data/thomasjiralerspong/issue2643_sae_map")
    p.add_argument("--out", default="eval_results/issue_2643/sae_map_screen")
    p.add_argument("--data-revision", default=DATA_REVISION)
    p.add_argument("--capture-prefix", default=CAPTURE_PREFIX)
    p.add_argument("--context-sae", default="")
    p.add_argument("--answer-sae", default="")
    p.add_argument("--ridge", default="")
    p.add_argument("--context-sae-prefix", default=CTX_SAE_PREFIX)
    p.add_argument("--answer-sae-prefix", default=ANS_SAE_PREFIX)
    p.add_argument("--ridge-repo-path", default=RIDGE_REPO_PATH)
    p.add_argument("--calibration-ridge", type=float, default=1.0)
    p.add_argument("--max-scale", type=float, default=8.0)
    p.add_argument("--rarity-min-count", type=int, default=32)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.max_chunks < 0:
        raise SystemExit("--batch-size must be positive and --max-chunks nonnegative")
    if args.phase == "selfcheck":
        phase_selfcheck()
    else:
        phase_screen(args)


if __name__ == "__main__":
    main()
