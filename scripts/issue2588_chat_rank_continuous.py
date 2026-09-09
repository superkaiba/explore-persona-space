#!/usr/bin/env python3
"""Threshold-free effective-dimension measures for the Qwen3-8B chat maps.

The operational rank is a threshold crossing, so it inherits an arbitrary
retention constant and a hard selection step. These are the standard continuous
alternatives, reported per arm:

Spectrum side, from the training-output PCA eigenvalues:
  effective rank      exp(H(p)), p = lambda / sum(lambda)   (Roy and Vetterli 2007)
  participation ratio (sum lambda)^2 / sum(lambda^2)
  stable rank         sum(lambda) / max(lambda)

Predictive side, from the held-out nested-rank curve:
  predictive effective dimension  sum_k (1 - R2(k)/R2_full)
    the area above the normalised attainment curve, which is the mean rank of
    an integer variable whose CDF is the normalised curve. No threshold.

Only the last one is bootstrapped, since it is the predictive analogue of the
operational rank the paper reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import issue2588_chat_rank as rank


def fit_pieces(cell_root: Path, arm: str) -> dict:
    position = rank.POSITIONS[arm]
    fits = json.loads((cell_root / "fits" / f"fits_{position}.json").read_text())
    layer = int(fits["layer_star"])
    lam = float(fits["layers"][str(layer)]["fit_meta"]["selected_lambda"])
    xtr, ytr = rank.load_split(cell_root, "train_10k", layer, position, 4096)
    xval, yval = rank.load_split(cell_root, "val_400", layer, position, 4096)
    xte, yte = rank.load_split(cell_root, "test_1000", layer, position, 4096)
    payload = rank.reconstruct(xtr, ytr, xval, yval, xte, yte, lam)

    w = np.asarray(payload["W"], dtype=np.float64)
    xmu = np.asarray(payload["xmu"], dtype=np.float64)
    xsd = np.asarray(payload["xsd"], dtype=np.float64)
    xn = (xtr.astype(np.float64) - xmu) / xsd
    m = w.T @ (xn.T @ xn) @ w
    m = 0.5 * (m + m.T)
    evals, evecs = np.linalg.eigh(m)
    order = np.argsort(evals)[::-1]
    spectrum = np.clip(evals[order], 0.0, None) / len(xtr)
    basis = np.ascontiguousarray(evecs[:, order], dtype=np.float64)
    return {"payload": payload, "spectrum": spectrum, "basis": basis, "layer": layer, "lam": lam}


def spectrum_measures(spectrum: np.ndarray) -> dict:
    total = float(spectrum.sum())
    p = spectrum / (total + 1e-300)
    nz = p[p > 0]
    return {
        "effective_rank_entropy": float(np.exp(-(nz * np.log(nz)).sum())),
        "participation_ratio": float(total**2 / (np.square(spectrum).sum() + 1e-300)),
        "stable_rank": float(total / (spectrum.max() + 1e-300)),
    }


def predictive_dimension(curve: np.ndarray) -> float:
    """Area above the normalised attainment curve. Threshold-free."""
    full = float(curve[-1])
    attain = np.clip(np.asarray(curve, dtype=np.float64) / full, 0.0, 1.0)
    return float(np.sum(1.0 - attain[:-1]))


def curve_from_counts(piece: dict, split: str, counts: np.ndarray) -> np.ndarray:
    payload = piece["payload"]
    pred = np.asarray(payload[f"pred_{split}"], dtype=np.float64)
    target = np.asarray(payload[f"target_{split}"], dtype=np.float64)
    mu = np.asarray(payload["ymu"], dtype=np.float64).reshape(1, -1)
    pc = (pred - mu) @ piece["basis"]
    yc = (target - mu) @ piece["basis"]
    n_draw = counts.sum(axis=1, keepdims=True)
    sse0 = counts @ np.square(target - mu).sum(axis=1)
    sse = np.concatenate(
        [
            sse0[:, None],
            sse0[:, None] + np.cumsum(counts @ (pc * pc) - 2.0 * (counts @ (pc * yc)), axis=1),
        ],
        axis=1,
    )
    ysum = counts @ target
    ymean = ysum / n_draw
    sst = (
        (counts @ np.square(target).sum(axis=1))[:, None]
        - 2.0 * np.einsum("bd,bd->b", ymean, ysum)[:, None]
        + n_draw * np.square(ymean).sum(axis=1, keepdims=True)
    )
    return 1.0 - sse / (sst + 1e-30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen3-chat-v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    rank.load_dotenv()

    root = args.source_root / "generic" / args.run_id / "cells_cap_long"
    pieces = {a: fit_pieces(root / f"q3_8b_{a}", a) for a in rank.POSITIONS}
    out = {
        "schema": "issue2588_chat_rank_continuous_v1",
        "note": "Threshold-free effective-dimension measures. Spectrum measures are "
        "train-side and carry no sampling CI here. The predictive dimension is "
        "bootstrapped over validation rows, paired across arms.",
        "draws": args.draws,
        "arms": {},
    }
    for arm, piece in pieces.items():
        entry = {"layer": piece["layer"], "selected_lambda": piece["lam"]}
        entry.update(spectrum_measures(piece["spectrum"]))
        ones = np.ones((1, len(piece["payload"]["target_val"])), dtype=np.float64)
        entry["predictive_effective_dimension_val"] = predictive_dimension(
            curve_from_counts(piece, "val", ones)[0]
        )
        ones_te = np.ones((1, len(piece["payload"]["target_test"])), dtype=np.float64)
        entry["predictive_effective_dimension_test"] = predictive_dimension(
            curve_from_counts(piece, "test", ones_te)[0]
        )
        out["arms"][arm] = entry

    rows = {
        a: [
            r["row_id"].rsplit("_", 1)[1]
            for r in sorted(
                json.loads((root / f"q3_8b_{a}" / "capture" / "val_400" / "rows.json").read_text())[
                    "rows"
                ],
                key=lambda r: r["row_id"],
            )
        ]
        for a in pieces
    }
    shared = [k for k in rows["a"] if k in set(rows["b"])]
    rng = np.random.default_rng(args.seed)
    picks = rng.integers(0, len(shared), size=(args.draws, len(shared)))
    boot = {}
    for arm, piece in pieces.items():
        index = {k: i for i, k in enumerate(rows[arm])}
        cols = np.array([index[k] for k in shared])
        counts = np.zeros((args.draws, len(rows[arm])), dtype=np.float64)
        np.add.at(counts, (np.arange(args.draws)[:, None], cols[picks]), 1.0)
        curves = curve_from_counts(piece, "val", counts)
        full = curves[:, -1:]
        attain = np.clip(curves / full, 0.0, 1.0)
        boot[arm] = np.sum(1.0 - attain[:, :-1], axis=1)
        lo, hi = np.percentile(boot[arm], [2.5, 97.5])
        out["arms"][arm]["predictive_effective_dimension_val_ci95"] = [float(lo), float(hi)]
        out["arms"][arm]["predictive_effective_dimension_val_median"] = float(np.median(boot[arm]))
    diff = boot["a"] - boot["b"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    out["predictive_effective_dimension_difference_a_minus_b"] = {
        "median": float(np.median(diff)),
        "ci95": [float(lo), float(hi)],
        "fraction_of_draws_b_below_a": float((diff > 0).mean()),
    }
    rank.write_json(args.output, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
