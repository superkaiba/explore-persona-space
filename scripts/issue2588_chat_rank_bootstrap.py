#!/usr/bin/env python3
"""Bootstrap confidence intervals for the Qwen3-8B chat operational-rank gap.

The operational rank is a threshold crossing on a validation curve estimated
from ~399 rows, so it carries sampling noise that the point estimate hides.
This resamples validation rows with replacement and recomputes both the curve
and its own retention threshold per draw, giving a CI per arm and a paired CI
on the difference.

Scope: this is EVALUATION-sample uncertainty only. The ridge fit and the
training-output PCA basis are held fixed, so fit variability and training-target
noise are not captured and the intervals are lower bounds on total uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import issue2588_chat_rank as rank


def arm_pieces(cell_root: Path, arm: str) -> dict:
    """Fixed fit and basis, plus the per-row terms the curve is built from."""
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
    basis = np.ascontiguousarray(evecs[:, np.argsort(evals)[::-1]], dtype=np.float64)

    pred = np.asarray(payload["pred_val"], dtype=np.float64)
    target = np.asarray(payload["target_val"], dtype=np.float64)
    mu = np.asarray(payload["ymu"], dtype=np.float64).reshape(1, -1)
    pc = (pred - mu) @ basis
    yc = (target - mu) @ basis

    rows = json.loads((cell_root / "capture" / "val_400" / "rows.json").read_text())["rows"]
    keys = [r["row_id"].rsplit("_", 1)[1] for r in sorted(rows, key=lambda r: r["row_id"])]
    return {
        "keys": keys,
        "sq_pc": pc * pc,
        "cross": pc * yc,
        "resid0": np.square(target - mu).sum(axis=1),
        "target": target,
        "sq_target": np.square(target).sum(axis=1),
        "layer": layer,
    }


def ranks_for_counts(piece: dict, counts: np.ndarray) -> np.ndarray:
    """Vectorised rank selection over B bootstrap weightings of the same rows."""
    n_draw = counts.sum(axis=1, keepdims=True)
    sse0 = counts @ piece["resid0"]
    c2 = counts @ piece["sq_pc"]
    cy = counts @ piece["cross"]
    sse = np.concatenate([sse0[:, None], sse0[:, None] + np.cumsum(c2 - 2.0 * cy, axis=1)], axis=1)
    ysum = counts @ piece["target"]
    ymean = ysum / n_draw
    sst = (
        (counts @ piece["sq_target"])[:, None]
        - 2.0 * np.einsum("bd,bd->b", ymean, ysum)[:, None]
        + n_draw * np.square(ymean).sum(axis=1, keepdims=True)
    )
    curve = 1.0 - sse / (sst + 1e-30)
    full = curve[:, -1]
    threshold = 1.0 - (1.0 + rank.RELATIVE_ERROR) * (1.0 - full)
    hit = curve >= threshold[:, None] - 1e-12
    return hit.argmax(axis=1).astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen3-chat-v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    rank.load_dotenv()

    root = args.source_root / "generic" / args.run_id / "cells_cap_long"
    pieces = {a: arm_pieces(root / f"q3_8b_{a}", a) for a in rank.POSITIONS}

    shared = [k for k in pieces["a"]["keys"] if k in set(pieces["b"]["keys"])]
    index = {a: {k: i for i, k in enumerate(pieces[a]["keys"])} for a in pieces}
    rng = np.random.default_rng(args.seed)
    picks = rng.integers(0, len(shared), size=(args.draws, len(shared)))

    out = {
        "schema": "issue2588_chat_rank_bootstrap_v1",
        "draws": args.draws,
        "seed": args.seed,
        "n_paired_validation_rows": len(shared),
        "resampling": "paired over validation questions shared by both arms",
        "scope": "evaluation-sample uncertainty only; the ridge fit and training-output "
        "PCA basis are held fixed, so these intervals are a LOWER BOUND on total "
        "uncertainty and exclude training-target noise",
    }
    per_arm = {}
    for arm, piece in pieces.items():
        cols = np.array([index[arm][k] for k in shared])
        counts = np.zeros((args.draws, len(piece["keys"])), dtype=np.float64)
        np.add.at(counts, (np.arange(args.draws)[:, None], cols[picks]), 1.0)
        per_arm[arm] = ranks_for_counts(piece, counts)
        lo, hi = np.percentile(per_arm[arm], [2.5, 97.5])
        out[f"rank_{arm}"] = {
            "layer": piece["layer"],
            "median": float(np.median(per_arm[arm])),
            "ci95": [float(lo), float(hi)],
            "iqr": [float(np.percentile(per_arm[arm], 25)), float(np.percentile(per_arm[arm], 75))],
        }
    diff = per_arm["a"] - per_arm["b"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    out["rank_difference_a_minus_b"] = {
        "median": float(np.median(diff)),
        "ci95": [float(lo), float(hi)],
        "fraction_of_draws_with_b_below_a": float((diff > 0).mean()),
        "fraction_of_draws_with_b_at_least_half_of_a": float(
            (per_arm["b"] <= 0.5 * per_arm["a"]).mean()
        ),
    }
    rank.write_json(args.output, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
