#!/usr/bin/env python3
"""Plan §6.5 row 4 deliverable: U1 direction-consistency cosines.

Computes, per variant, the within-arm pairwise cosines (3 seed-pairs per
arm) and the cross-arm 9-pair cosine distribution between the top
singular vectors (U1) persisted in the per-cell Phase-D SVD JSONs.
SVD sign is arbitrary, so |cos| is the headline statistic; raw cosines
are kept alongside for sign-structure inspection.

Run (from repo root, after Phase D):

    uv run python scripts/issue_521_direction_consistency.py \
        [--svd-dir eval_results/issue_521/svd]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

VARIANTS = ("same", "base", "on_policy")
ARMS = ("marker", "em")
SEEDS = (42, 137, 256)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svd-dir", default="eval_results/issue_521/svd", type=Path)
    args = parser.parse_args()

    u1: dict[tuple[str, str, int], np.ndarray] = {}
    for variant, arm, seed in itertools.product(VARIANTS, ARMS, SEEDS):
        path = args.svd_dir / f"{variant}_{arm}_seed{seed}.json"
        cell = json.loads(path.read_text())
        u1[(variant, arm, seed)] = np.asarray(cell["U1"], dtype=np.float64)

    out: dict = {"variants": {}, "seeds": list(SEEDS), "arms": list(ARMS)}
    for variant in VARIANTS:
        within: dict[str, list[dict]] = {}
        for arm in ARMS:
            pairs = []
            for s1, s2 in itertools.combinations(SEEDS, 2):
                c = _cos(u1[(variant, arm, s1)], u1[(variant, arm, s2)])
                pairs.append({"seed_a": s1, "seed_b": s2, "cos": c, "abs_cos": abs(c)})
            within[arm] = pairs
        cross = []
        for sm, se in itertools.product(SEEDS, SEEDS):
            c = _cos(u1[(variant, "marker", sm)], u1[(variant, "em", se)])
            cross.append({"marker_seed": sm, "em_seed": se, "cos": c, "abs_cos": abs(c)})

        def _stats(rows: list[dict]) -> dict:
            vals = np.array([r["abs_cos"] for r in rows])
            return {
                "mean_abs_cos": float(vals.mean()),
                "median_abs_cos": float(np.median(vals)),
                "min_abs_cos": float(vals.min()),
                "max_abs_cos": float(vals.max()),
                "n_pairs": len(rows),
            }

        out["variants"][variant] = {
            "within_arm": {arm: {"pairs": within[arm], **_stats(within[arm])} for arm in ARMS},
            "cross_arm": {"pairs": cross, **_stats(cross)},
        }

    out_path = args.svd_dir / "direction_consistency.json"
    out_path.write_text(json.dumps(out, indent=2))

    for variant in VARIANTS:
        v = out["variants"][variant]
        print(
            f"{variant}: within-marker mean|cos|={v['within_arm']['marker']['mean_abs_cos']:.3f} "
            f"within-em mean|cos|={v['within_arm']['em']['mean_abs_cos']:.3f} "
            f"cross-arm mean|cos|={v['cross_arm']['mean_abs_cos']:.3f}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
