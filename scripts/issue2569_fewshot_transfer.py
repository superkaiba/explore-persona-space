#!/usr/bin/env python3
"""Test zero-/few-shot transport between the issue #2569 model mappings.

Two deliberately different transfer regimes are evaluated on the frozen test
split.  ``summary_only`` matches separately fitted PCA coordinates by variance
rank and skewness sign, using no paired row identities.  ``few_query`` uses
only k paired train rows to build regularized context and answer bridges around
the frozen source map.  The latter is compared with fitting a target map from
scratch from exactly the same k rows and with the full-data target-map ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
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

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

import issue2569_atlas as AT  # noqa: E402
import issue2569_mapping_diff as MD  # noqa: E402


@dataclass(frozen=True)
class PCSummary:
    mean: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    explained_fraction: float


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def gpu_matmul(a: np.ndarray, b: np.ndarray, device: str) -> np.ndarray:
    dev = torch.device(device)
    with torch.inference_mode():
        out = torch.as_tensor(a, dtype=torch.float32, device=dev) @ torch.as_tensor(
            b, dtype=torch.float32, device=dev
        )
    result = out.cpu().numpy()
    del out
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result


def predict_affine_gpu(affine: MD.Affine, x: np.ndarray, device: str) -> np.ndarray:
    return gpu_matmul(np.asarray(x, np.float32), np.asarray(affine.A, np.float32), device) + np.asarray(
        affine.b, np.float32
    )


def fit_pc_summary(
    x: np.ndarray,
    indices: np.ndarray,
    rank: int,
    *,
    device: str,
    seed: int,
) -> PCSummary:
    """Fit a sign-oriented randomized PCA summary without paired-row use."""
    arr = np.asarray(x[indices], np.float32)
    mean = arr.mean(0, dtype=np.float64).astype(np.float32)
    centered = arr - mean
    q = min(rank, centered.shape[0] - 1, centered.shape[1])
    if q < 1:
        raise ValueError("PCA summary needs at least two rows")
    dev = torch.device(device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        tensor = torch.as_tensor(centered, dtype=torch.float32, device=dev)
        _, singular, basis = torch.pca_lowrank(tensor, q=q, center=False, niter=4)
        scores = tensor @ basis
        skew = torch.sum(scores**3, dim=0)
        fallback_idx = torch.argmax(torch.abs(basis), dim=0)
        fallback = basis[fallback_idx, torch.arange(q, device=dev)]
        sign = torch.where(torch.abs(skew) > 1e-6, torch.sign(skew), torch.sign(fallback))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        basis = basis * sign
        singular_np = singular.cpu().numpy().astype(np.float64)
        basis_np = basis.cpu().numpy().astype(np.float32)
    total = float(np.sum(np.asarray(centered, np.float64) ** 2))
    explained = float(np.sum(singular_np**2) / max(total, 1e-30))
    scale = np.maximum(singular_np / math.sqrt(len(indices) - 1), 1e-8).astype(np.float32)
    del tensor, singular, basis, scores
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return PCSummary(mean=mean, basis=basis_np, scale=scale, explained_fraction=explained)


def pc_scores(x: np.ndarray, summary: PCSummary, device: str) -> np.ndarray:
    projected = gpu_matmul(
        np.asarray(x, np.float32) - summary.mean,
        summary.basis,
        device,
    )
    return projected / summary.scale


def pc_reconstruct(z: np.ndarray, summary: PCSummary, device: str) -> np.ndarray:
    return gpu_matmul(
        np.asarray(z, np.float32) * summary.scale,
        summary.basis.T,
        device,
    ) + summary.mean


def prediction_metrics(
    pred: np.ndarray,
    observed: np.ndarray,
    full_map: np.ndarray,
    train_mean: np.ndarray,
) -> dict[str, Any]:
    p = np.asarray(pred, np.float64)
    y = np.asarray(observed, np.float64)
    f = np.asarray(full_map, np.float64)
    mu = np.asarray(train_mean, np.float64)

    def centered_cos(a: np.ndarray, b: np.ndarray) -> float:
        return MD.flat_cosine(a - mu, b - mu)

    baseline_sse = float(np.sum((y - mu) ** 2))
    mapping_sse = float(np.sum((p - f) ** 2))
    mapping_denom = float(np.sum((f - mu) ** 2))
    return {
        "observed_target": {
            "pooled_r2": float(AT.pooled_r2(p, y)),
            "train_mean_normalized_r2": float(
                1.0 - np.sum((p - y) ** 2) / max(baseline_sse, 1e-30)
            ),
            "centered_cosine": centered_cos(p, y),
        },
        "full_target_mapping": {
            "normalized_r2": float(1.0 - mapping_sse / max(mapping_denom, 1e-30)),
            "centered_cosine": centered_cos(p, f),
            "relative_l2": float(
                np.linalg.norm(p - f) / (np.linalg.norm(f - mu) + 1e-30)
            ),
        },
    }


def ridge_kernel_weights(
    query: np.ndarray,
    anchors: np.ndarray,
    *,
    ridge_fraction: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return affine kernel weights using only anchor mean and Gram statistics."""
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
        solved = torch.linalg.solve(
            gram + lam * torch.eye(len(a), dtype=torch.float32, device=dev),
            at,
        )
        weights = qt @ solved.T
    result = weights.cpu().numpy()
    del at, qt, gram, solved, weights
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result, mean


def affine_span_predict(
    query: np.ndarray,
    input_anchors: np.ndarray,
    output_anchors: np.ndarray,
    *,
    ridge_fraction: float,
    device: str,
) -> np.ndarray:
    weights, _ = ridge_kernel_weights(
        query, input_anchors, ridge_fraction=ridge_fraction, device=device
    )
    outputs = np.asarray(output_anchors, np.float32)
    output_mean = outputs.mean(0)
    return gpu_matmul(weights, outputs - output_mean, device) + output_mean


def few_query_prediction(
    target_test_context: np.ndarray,
    target_anchor_context: np.ndarray,
    source_anchor_map_prediction: np.ndarray,
    source_anchor_answer: np.ndarray,
    target_anchor_answer: np.ndarray,
    *,
    ridge_fraction: float,
    device: str,
) -> np.ndarray:
    source_map_at_test = affine_span_predict(
        target_test_context,
        target_anchor_context,
        source_anchor_map_prediction,
        ridge_fraction=ridge_fraction,
        device=device,
    )
    return affine_span_predict(
        source_map_at_test,
        source_anchor_answer,
        target_anchor_answer,
        ridge_fraction=ridge_fraction,
        device=device,
    )


def summarize_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = (
        ("observed_target", "pooled_r2"),
        ("observed_target", "train_mean_normalized_r2"),
        ("observed_target", "centered_cosine"),
        ("full_target_mapping", "normalized_r2"),
        ("full_target_mapping", "centered_cosine"),
        ("full_target_mapping", "relative_l2"),
    )
    out: dict[str, Any] = {"n_repeats": len(rows)}
    for group, metric in paths:
        values = np.asarray([row[group][metric] for row in rows], np.float64)
        out.setdefault(group, {})[metric] = {
            "median": float(np.median(values)),
            "min_max": [float(np.min(values)), float(np.max(values))],
            "q10_q90": [float(v) for v in np.quantile(values, [0.1, 0.9])],
        }
    return out


def phase_analyze(args: argparse.Namespace) -> None:
    started = time.time()
    roster, folds, arrays, payloads, records = MD.load_primary(args)
    tr, te = folds["tr"], folds["te"]
    cells = {name: MD.payload_affine(payload) for name, payload in payloads.items()}
    print(f"[fewshot-transfer] loaded n={len(roster)}", flush=True)

    native_context = {"q": arrays["q_context"], "l": arrays["l_context"]}
    native_answers = {
        ("q", "qwriter"): arrays["q_qwriter"],
        ("l", "qwriter"): arrays["l_qwriter"],
        ("q", "lwriter"): arrays["q_lwriter"],
        ("l", "lwriter"): arrays["l_lwriter"],
    }
    native_cells = {
        ("q", "qwriter"): cells["q_qwriter"],
        ("l", "qwriter"): cells["l_qwriter"],
        ("q", "lwriter"): cells["q_lwriter"],
        ("l", "lwriter"): cells["l_lwriter"],
    }

    map_predictions: dict[tuple[str, str], np.ndarray] = {}
    for key, cell in native_cells.items():
        map_predictions[key] = predict_affine_gpu(cell, native_context[key[0]], args.device)
        print(f"[fewshot-transfer] predicted full native cell {key}", flush=True)

    summaries = {
        "q_context": fit_pc_summary(
            native_context["q"], tr, args.summary_rank, device=args.device, seed=args.seed
        ),
        "l_context": fit_pc_summary(
            native_context["l"], tr, args.summary_rank, device=args.device, seed=args.seed + 1
        ),
    }
    for model, offset in (("q", 2), ("l", 3)):
        pooled = np.concatenate(
            [native_answers[(model, "qwriter")][tr], native_answers[(model, "lwriter")][tr]],
            axis=0,
        )
        summaries[f"{model}_answer"] = fit_pc_summary(
            pooled,
            np.arange(len(pooled)),
            args.summary_rank,
            device=args.device,
            seed=args.seed + offset,
        )
    print("[fewshot-transfer] fitted separate unpaired PCA summaries", flush=True)

    directions = (("q", "l"), ("l", "q"))
    summary_only: dict[str, Any] = {}
    few_query: dict[str, Any] = {}
    full_ceiling: dict[str, Any] = {}
    pca_ceiling: dict[str, Any] = {}
    rng = np.random.default_rng(args.seed)

    for source, target in directions:
        direction = f"{source}_to_{target}"
        summary_only[direction] = {}
        few_query[direction] = {}
        full_ceiling[direction] = {}
        pca_ceiling[direction] = {}
        source_context_summary = summaries[f"{source}_context"]
        target_context_summary = summaries[f"{target}_context"]
        source_answer_summary = summaries[f"{source}_answer"]
        target_answer_summary = summaries[f"{target}_answer"]
        target_context_scores = pc_scores(
            native_context[target][te], target_context_summary, args.device
        )
        source_context_hat = pc_reconstruct(
            target_context_scores, source_context_summary, args.device
        )

        for writer in ("qwriter", "lwriter"):
            target_observed = native_answers[(target, writer)][te]
            target_full = map_predictions[(target, writer)][te]
            target_train_mean = native_answers[(target, writer)][tr].mean(0)
            full_ceiling[direction][writer] = prediction_metrics(
                target_full, target_observed, target_full, target_train_mean
            )
            target_truth_scores = pc_scores(
                target_observed, target_answer_summary, args.device
            )
            target_pca_reconstruction = pc_reconstruct(
                target_truth_scores, target_answer_summary, args.device
            )
            pca_ceiling[direction][writer] = prediction_metrics(
                target_pca_reconstruction,
                target_observed,
                target_full,
                target_train_mean,
            )

            source_summary_prediction = predict_affine_gpu(
                native_cells[(source, writer)], source_context_hat, args.device
            )
            source_answer_scores = pc_scores(
                source_summary_prediction, source_answer_summary, args.device
            )
            target_summary_prediction = pc_reconstruct(
                source_answer_scores, target_answer_summary, args.device
            )
            summary_only[direction][writer] = prediction_metrics(
                target_summary_prediction,
                target_observed,
                target_full,
                target_train_mean,
            )

            writer_results: dict[str, Any] = {}
            for k in args.k_values:
                transfer_rows: list[dict[str, Any]] = []
                scratch_rows: list[dict[str, Any]] = []
                for repeat in range(args.repeats):
                    anchor = rng.choice(tr, size=k, replace=False)
                    transfer_pred = few_query_prediction(
                        native_context[target][te],
                        native_context[target][anchor],
                        map_predictions[(source, writer)][anchor],
                        native_answers[(source, writer)][anchor],
                        native_answers[(target, writer)][anchor],
                        ridge_fraction=args.ridge_fraction,
                        device=args.device,
                    )
                    scratch_pred = affine_span_predict(
                        native_context[target][te],
                        native_context[target][anchor],
                        native_answers[(target, writer)][anchor],
                        ridge_fraction=args.ridge_fraction,
                        device=args.device,
                    )
                    transfer_rows.append(
                        prediction_metrics(
                            transfer_pred, target_observed, target_full, target_train_mean
                        )
                    )
                    scratch_rows.append(
                        prediction_metrics(
                            scratch_pred, target_observed, target_full, target_train_mean
                        )
                    )
                writer_results[str(k)] = {
                    "transported_source_mapping": summarize_repeats(transfer_rows),
                    "target_fit_from_scratch": summarize_repeats(scratch_rows),
                }
                print(
                    f"[fewshot-transfer] {direction} {writer} k={k} repeats={args.repeats}",
                    flush=True,
                )
            few_query[direction][writer] = writer_results

    result = {
        "issue": 2569,
        "followup_label": "cross-model-fewshot-map-transfer",
        "source_revision": MD.SOURCE_REVISION,
        "layers": MD.PRIMARY_LAYERS,
        "n": {"train": int(len(tr)), "validation_unused": int(len(folds["va"])), "test": int(len(te))},
        "test_roster_sha256": MD.sha_int64(roster[te]),
        "design": {
            "summary_only": (
                "Separate train-only mean/PCA/variance summaries; PCs paired only by descending variance "
                "and oriented by marginal skewness. No paired row identities are used."
            ),
            "few_query": (
                "Train-only k paired rows define regularized affine-span context and answer bridges around "
                "the frozen source mapping; evaluated on the untouched test rows."
            ),
            "scratch_control": "Target context-to-answer map fit from exactly the same k paired rows.",
            "full_ceiling": "Original target native map fit with 8,000 train rows and lambda selected on validation.",
            "hyperparameters": {
                "summary_rank": int(args.summary_rank),
                "k_values": [int(k) for k in args.k_values],
                "repeats": int(args.repeats),
                "ridge_fraction_of_mean_centered_gram_diagonal": float(args.ridge_fraction),
                "few_query_hyperparameters_predeclared_no_validation_tuning": True,
            },
        },
        "summary_explained_fraction": {
            name: float(summary.explained_fraction) for name, summary in summaries.items()
        },
        "summary_only": summary_only,
        "few_query": few_query,
        "pca_reconstruction_ceiling": pca_ceiling,
        "full_target_map_ceiling": full_ceiling,
        "native_target_map_record_test_r2": {
            name: float(records[name]["test_r2"]) for name in MD.CELL_NAMES
        },
        "caveats": [
            "The summary-only PC rank match is intentionally weak: covariance spectra do not identify semantic axes.",
            "Few-query transfer uses paired residual tuples from both models and therefore is calibration, not zero-shot transfer.",
            "All k-shot curves reuse an existing fixed dataset; query counts describe calibration rows, not new API calls.",
            "This is exploratory post-hoc analysis on the LMSYS-only pilot.",
        ],
        "elapsed_s": round(time.time() - started, 2),
    }
    atomic_json(Path(args.out_dir) / "fewshot_transfer.json", result)
    print(f"[fewshot-transfer] wrote {args.out_dir} elapsed={result['elapsed_s']:.1f}s", flush=True)


def phase_selftest(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(2569)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    a = rng.normal(size=(6, 5)).astype(np.float32)
    b = rng.normal(size=5).astype(np.float32)
    y = x @ a + b
    pred = affine_span_predict(
        x[20:], x[:20], y[:20], ridge_fraction=1e-8, device="cpu"
    )
    assert np.allclose(pred, y[20:], atol=2e-4, rtol=2e-4)
    metrics = prediction_metrics(y, y, y, y[:20].mean(0))
    assert abs(metrics["observed_target"]["pooled_r2"] - 1.0) < 1e-12
    assert abs(metrics["full_target_mapping"]["normalized_r2"] - 1.0) < 1e-12
    print("[fewshot-transfer] selftest PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("analyze", "selftest"), required=True)
    base = PROJECT_ROOT / "data" / "issue_2569" / "ownanswers"
    parser.add_argument("--qwriter-dir", default=str(base / "qwriter_final"))
    parser.add_argument("--lwriter-dir", default=str(base / "writer_llama" / "final"))
    parser.add_argument("--map-dir", default=str(base / "analysis" / "maps"))
    parser.add_argument("--split-json", default=str(base / "analysis" / "split.json"))
    parser.add_argument("--out-dir", default=str(base / "fewshot_transfer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary-rank", type=int, default=64)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--ridge-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2569)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summary_rank < 2 or args.summary_rank > 256:
        raise ValueError("summary-rank must be in [2, 256]")
    if args.repeats < 1 or min(args.k_values) < 2 or max(args.k_values) > 1024:
        raise ValueError("invalid repeats or k-values")
    if args.ridge_fraction <= 0:
        raise ValueError("ridge-fraction must be positive")
    {"analyze": phase_analyze, "selftest": phase_selftest}[args.phase](args)


if __name__ == "__main__":
    main()
