#!/usr/bin/env python3
"""Context-space vs answer-space geometry, and its retrieval consequence.

Two reads on the paper's 10,000-candidate operating point, both 0 GPU-h from
banked artifacts:

  A. Over context PAIRS (the 9,058-distractor pool, 41M pairs): does context
     cosine predict answer cosine, in the retrieval convention (mean-centered
     context, Cholesky-whitened answer)?
  B. Over QUERIES (the 942 held-out queries): does top-1 retrieval fall as the
     nearest OTHER context gets closer?

Writes eval_results/issue_1901/ctxans_geometry/summary.json for the figure
script; performs no fits and no inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# #847: thread caps must land BEFORE the numpy/scipy imports; load_dotenv() setdefaults
# OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS on the shared VM and BLAS pools freeze at import.
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# uniform through the bulk, finer in the tail: the effect lives above 0.9 and a
# uniform grid dilutes it into one wide bin.
BIN_EDGES = np.round(
    np.concatenate([np.linspace(-0.5, 0.90, 15), [0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0]]), 6
)
SEED = 1901


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial CI that stays inside [0, 1] at small n (bins here reach n=10)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, default=ROOT / "data/issue_1901/ctxnn_dl")
    ap.add_argument(
        "--perrow", type=Path, required=True, help="npz with per-query nn1 + correct (panel B)"
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "eval_results/issue_1901/ctxans_geometry/summary.json"
    )
    a = ap.parse_args()

    w = np.load(a.stage / "issue1901_mlpdense/analysis_tensors/whiten_stats_L19.npz")
    mu_A, mu_C, L = (
        w["mu_A"].astype(np.float32),
        w["mu_C"].astype(np.float32),
        w["L"].astype(np.float32),
    )
    cz = np.load(a.stage / "cx_distr_L19.npz")
    cx, ci = cz["cx"].astype(np.float32), cz["ci"].astype(np.int64)
    az = np.load(a.stage / "issue1901_metrics/analysis_tensors/distractors_L19.npz")
    pos = {int(c): i for i, c in enumerate(az["ci"].astype(np.int64))}
    # one read of the 1.4 GB member, then fancy-index: indexing az["vx"] inside a
    # comprehension re-reads the whole array per row.
    order = np.array([pos[int(c)] for c in ci], dtype=np.int64)
    vx = az["vx"][order].astype(np.float32)
    assert vx.shape == cx.shape, (vx.shape, cx.shape)

    Cc = unit(cx - mu_C)
    Aw = unit(solve_triangular(L, (vx - mu_A).T, lower=True).T)
    n = Cc.shape[0]
    iu = np.triu_indices(n, 1)
    gc = (Cc @ Cc.T).astype(np.float32)[iu]
    ga = (Aw @ Aw.T).astype(np.float32)[iu]

    rng = np.random.default_rng(SEED)
    sub = rng.choice(gc.size, size=min(5_000_000, gc.size), replace=False)
    out = {
        "convention": (
            "mean-centered context cosine vs Cholesky-whitened answer cosine "
            "(the retrieval convention); pairwise cosine, direction-aware; "
            "conventions match eval_results/issue_2202/ctxans_corr"
        ),
        "answer_draws": "single stored draw per context, not the five-rollout mean",
        "n_contexts": int(n),
        "n_pairs": int(gc.size),
        "global": {
            "pearson_r": float(np.corrcoef(gc[sub], ga[sub])[0, 1]),
            "spearman_rho": float(spearmanr(gc[sub], ga[sub]).statistic),
            "subsample": int(sub.size),
            "answer_cos_all_pairs_mean": float(ga.mean()),
            "answer_cos_all_pairs_sd": float(ga.std()),
        },
        "bin_edges": BIN_EDGES.tolist(),
        "panel_a": [],
        "panel_b": [],
    }

    idx = np.digitize(gc, BIN_EDGES) - 1
    for b in range(len(BIN_EDGES) - 1):
        m = idx == b
        if not m.any():
            continue
        v = ga[m]
        out["panel_a"].append(
            {
                "lo": float(BIN_EDGES[b]),
                "hi": float(BIN_EDGES[b + 1]),
                "n_pairs": int(m.sum()),
                "mean": float(v.mean()),
                "p10": float(np.quantile(v, 0.10)),
                "p50": float(np.median(v)),
                "p90": float(np.quantile(v, 0.90)),
            }
        )

    pr = np.load(a.perrow)
    nn1, correct = pr["nn1"].astype(float), pr["correct"].astype(bool)
    qidx = np.digitize(nn1, BIN_EDGES) - 1
    for b in range(len(BIN_EDGES) - 1):
        m = qidx == b
        if not m.any():
            continue
        k, tot = int(correct[m].sum()), int(m.sum())
        lo, hi = wilson(k, tot)
        out["panel_b"].append(
            {
                "lo": float(BIN_EDGES[b]),
                "hi": float(BIN_EDGES[b + 1]),
                "n_queries": tot,
                "n_correct": k,
                "top1": k / tot,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    out["panel_b_totals"] = {"n_queries": int(correct.size), "top1": float(correct.mean())}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}  ({len(out['panel_a'])} pair bins, {len(out['panel_b'])} query bins)")
    print(
        f"  global spearman {out['global']['spearman_rho']:.3f}, "
        f"pearson {out['global']['pearson_r']:.3f}"
    )


if __name__ == "__main__":
    main()
