#!/usr/bin/env python3
"""Noise control for the Qwen3-8B chat rank comparison.

Re-scores the SAME frozen maps against a three-draw averaged TEST target, to
test whether the operational-rank gap between the no-thinking and thinking arms
is an artefact of the two arms' targets having different repeatability.

Maps are NOT refit: training and validation targets are single-draw and no
extra draws exist for those splits. This isolates the evaluation-side share of
the noise question only, at zero GPU cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import issue2588_chat_rank as rank

CEILING_SEEDS = ("ceiling_s43", "ceiling_s44")


def draw_targets(cell_root: Path, split: str, layer: int) -> dict[str, np.ndarray]:
    """Map row_id -> answer vector for one alternate answer draw."""
    directory = cell_root / "capture" / split
    rows = json.loads((directory / "rows.json").read_text())["rows"]
    ids = [r["row_id"] for r in rows]
    ys = []
    for shard in sorted((directory / f"L{layer:02d}").glob("shard*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            ys.append(z["y_ans"])
    y = np.concatenate(ys)
    if len(ids) != len(y):
        raise ValueError(f"Row/tensor mismatch in {directory}")
    # Row ids are split-prefixed (test_1000_219 vs ceiling_s43_219); the trailing
    # index is the shared question key.
    return {rid.rsplit("_", 1)[1]: vec for rid, vec in zip(ids, y)}


def analyse(cell_root: Path, arm: str) -> dict:
    position = rank.POSITIONS[arm]
    fits = json.loads((cell_root / "fits" / f"fits_{position}.json").read_text())
    layer = int(fits["layer_star"])
    lam = float(fits["layers"][str(layer)]["fit_meta"]["selected_lambda"])
    xtr, ytr = rank.load_split(cell_root, "train_10k", layer, position, 4096)
    xval, yval = rank.load_split(cell_root, "val_400", layer, position, 4096)
    xte, yte = rank.load_split(cell_root, "test_1000", layer, position, 4096)

    test_rows = json.loads((cell_root / "capture" / "test_1000" / "rows.json").read_text())["rows"]
    test_ids = sorted(r["row_id"] for r in test_rows)
    draws = [draw_targets(cell_root, s, layer) for s in CEILING_SEEDS]
    index = {rid: k for k, rid in enumerate(test_ids)}
    shared = [rid for rid in test_ids if all(rid.rsplit("_", 1)[1] in d for d in draws)]
    if not shared:
        raise ValueError("No test rows are shared with both alternate answer draws")
    take = np.array([index[rid] for rid in shared], dtype=np.int64)

    payload = rank.reconstruct(xtr, ytr, xval, yval, xte, yte, lam)
    pred = np.asarray(payload["pred_test"], dtype=np.float64)[take]
    ymu = np.asarray(payload["ymu"], dtype=np.float64)

    single = np.asarray(yte, dtype=np.float64)[take]
    stacked = np.stack(
        [single]
        + [np.stack([d[rid.rsplit("_", 1)[1]] for rid in shared]).astype(np.float64) for d in draws]
    )
    averaged = stacked.mean(axis=0)

    # Same training-output PCA basis the production rank consumer uses.
    w = np.asarray(payload["W"], dtype=np.float64)
    xmu = np.asarray(payload["xmu"], dtype=np.float64)
    xsd = np.asarray(payload["xsd"], dtype=np.float64)
    xn = (xtr.astype(np.float64) - xmu) / xsd
    m = w.T @ (xn.T @ xn) @ w
    m = 0.5 * (m + m.T)
    evals, evecs = np.linalg.eigh(m)
    right = np.ascontiguousarray(evecs[:, np.argsort(evals)[::-1]], dtype=np.float32)

    result = {"arm": arm, "layer": layer, "selected_lambda": lam, "n_shared_test": len(shared)}
    for label, target in (("single_draw", single), ("three_draw_mean", averaged)):
        curve = rank.r2_curve_from_top_right_vectors(pred, target, ymu, right)
        full = float(curve[-1])
        threshold = 1.0 - (1.0 + rank.RELATIVE_ERROR) * (1.0 - full)
        result[label] = {
            "full_test_r2": full,
            "threshold": threshold,
            "rank_within_10pct_of_own_test_sse": rank.rank_at_threshold(curve, threshold),
            "rank_to_absolute_r2": {
                f"{t:.2f}": (
                    int(np.flatnonzero(curve >= t - 1e-12)[0])
                    if np.any(curve >= t - 1e-12)
                    else None
                )
                for t in (0.50, 0.55, 0.60, 0.64, 0.66, 0.70)
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", default="qwen3-chat-v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank.load_dotenv()
    results = [
        analyse(args.source_root / "generic" / args.run_id / "cells_cap_long" / f"q3_8b_{arm}", arm)
        for arm in rank.POSITIONS
    ]
    payload = {
        "schema": "issue2588_chat_rank_noise_control_v1",
        "question": "Is the operational-rank gap an artefact of unequal target repeatability?",
        "maps_refit": False,
        "caveat": "Training and validation targets stay single-draw; no extra draws exist for "
        "those splits. Ranks here are selected on TEST and are descriptive only.",
        "arms": results,
    }
    rank.write_json(args.output, payload)
    print(json.dumps(payload, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
