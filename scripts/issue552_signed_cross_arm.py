#!/usr/bin/env python3
"""#552 contrastive-2x2-completion — signed cross-arm cosines (interp-critique round 2).

Companion to ``issue552_contrastive_2x2_analysis.py`` STEP 3: recomputes the
5-arm pairwise top-direction cosines WITHOUT the absolute value, under the
registered orientation convention (``svd_summary`` orients each cell's U1 so
the panel-mean shift column projects nonnegatively — the same Phase-D
convention every zone metric was registered on). Writes
``cross_arm_5way/signed_summary.json`` with signed AND absolute values per
pair so the clean-result body can quote them together (interp-critique-codex
v7 blocking request 1).

Usage::

    uv run python scripts/issue552_signed_cross_arm.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.svd_direction_constancy import (  # noqa: E402
    assemble_M,
    cosine,
    svd_summary,
)

ALL_ARMS = ("marker", "em", "benign", "contrastive_em", "contrastive_benign")
NEW_ARMS = ("contrastive_em", "contrastive_benign")
SEEDS = (42, 137, 256)

REFERENCE_SHIFT_ROOTS = {
    "benign": "eval_results/issue_552/shifts",
    "em": "eval_results/issue_552/em-arm-mean-resp-reextraction/shifts",
    "marker": "eval_results/issue_552/marker-arm-mean-resp-reextraction/shifts",
}
FU = PROJECT_ROOT / "eval_results/issue_552/contrastive-2x2-completion"


def _shift_path(arm: str, seed: int) -> Path:
    if arm in NEW_ARMS:
        return FU / "shifts" / f"same_{arm}_seed{seed}.pt"
    return PROJECT_ROOT / REFERENCE_SHIFT_ROOTS[arm] / f"same_{arm}_seed{seed}.pt"


def _u1(pt_path: Path, use_mean_resp: bool) -> np.ndarray:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    M, _ = assemble_M(payload["shifts"], use_mean_resp=use_mean_resp)
    return svd_summary(M)["U1"]  # oriented: mean column projects nonnegatively


def _pairwise(u1: dict[tuple[str, int], np.ndarray]) -> dict:
    within: dict[str, dict] = {}
    for arm in ALL_ARMS:
        signed = [
            float(cosine(u1[(arm, s1)], u1[(arm, s2)]))
            for i, s1 in enumerate(SEEDS)
            for s2 in SEEDS[i + 1 :]
        ]
        within[arm] = {
            "signed_pairs": signed,
            "signed_median": statistics.median(signed),
            "abs_median": statistics.median(abs(v) for v in signed),
        }
    cross: dict[str, dict] = {}
    for i, a in enumerate(ALL_ARMS):
        for b in ALL_ARMS[i + 1 :]:
            per_pair = {
                f"seed{s1}__x__seed{s2}": float(cosine(u1[(a, s1)], u1[(b, s2)]))
                for s1 in SEEDS
                for s2 in SEEDS
            }
            vals = list(per_pair.values())
            cross[f"{a}__x__{b}"] = {
                "n_pairs": len(vals),
                "signed_median": statistics.median(vals),
                "abs_median": statistics.median(abs(v) for v in vals),
                "n_negative": sum(1 for v in vals if v < 0),
                "signed_pairs": per_pair,
            }
    return {"within_arm": within, "cross_arm": cross}


def main() -> None:
    u1_end: dict[tuple[str, int], np.ndarray] = {}
    u1_mr: dict[tuple[str, int], np.ndarray] = {}
    for arm in ALL_ARMS:
        for seed in SEEDS:
            p = _shift_path(arm, seed)
            u1_end[(arm, seed)] = _u1(p, use_mean_resp=False)
            u1_mr[(arm, seed)] = _u1(p, use_mean_resp=True)

    out = {
        "orientation_convention": (
            "each cell's top singular direction U1 is oriented so the cell's "
            "panel-mean shift column projects nonnegatively onto it (the same "
            "convention the registered zone metrics use); signed cosines are "
            "computed between these oriented directions"
        ),
        "end_slot": _pairwise(u1_end),
        "mean_resp": _pairwise(u1_mr),
        "metadata": {
            "script": "issue552_signed_cross_arm",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }

    # Consistency assert vs the persisted absolute summary.
    persisted = json.load(open(FU / "cross_arm_5way" / "summary.json"))
    for pos in ("end_slot", "mean_resp"):
        for pair, rec in out[pos]["cross_arm"].items():
            theirs = persisted[pos]["cross_arm"][pair]["median"]
            assert abs(rec["abs_median"] - theirs) < 1e-9, (pos, pair, rec["abs_median"], theirs)

    out_path = FU / "cross_arm_5way" / "signed_summary.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    # Console summary.
    for pos in ("mean_resp", "end_slot"):
        print(f"== {pos} ==")
        for pair, rec in out[pos]["cross_arm"].items():
            print(
                f"  {pair}: signed_median={rec['signed_median']:+.4f} "
                f"abs_median={rec['abs_median']:.4f} n_negative={rec['n_negative']}/9"
            )
        for arm, rec in out[pos]["within_arm"].items():
            neg = sum(1 for v in rec["signed_pairs"] if v < 0)
            print(
                f"  within {arm}: signed_median={rec['signed_median']:+.4f} "
                f"abs_median={rec['abs_median']:.4f} n_negative={neg}/3"
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
