"""#1112 free-analysis follow-up: debiased direction-cosine CI via paired
subsample-WITHOUT-replacement half-draws over the pooled capture tensors.

The body's H2 read reports cos(mu, mu) = 0.8048 between the two LoRA mixes'
mean-shift directions (layer 14 / response arm) with a paired WITH-replacement
bootstrap range [0.740, 0.835] that is biased low by resample deflation (a
with-replacement resample keeps ~63% unique rows -> attenuated subsample
cosines). This script recomputes the CI with a duplication-free scheme:

- paired half-sample draws: m = n/2 rows WITHOUT replacement, the SAME row
  indices applied to both cells of a pair per draw (>= 2000 draws);
- the attenuation reference at the same m: the SAME draws' same-cell
  split-half cosines (half A vs the complement half B, within each cell);
- the debiased comparison: per-draw paired deltas (reference - cross-mix) and
  an attenuation-corrected per-draw cosine cross / sqrt(ref_a * ref_b);
- fraction of draws below the registered 0.8 cutoff, per statistic.

Two half-partition schemes share one batched implementation: `row_random`
(uniform m-of-n row halves — the brief's primary scheme, matching the paired
cluster-bootstrap grain: one row per (context, question) cluster) and
`question_aligned` (partition the 20 question ids in half per draw, all 6
contexts ride along — the body's split-half-ceiling scheme, also exactly
m = 60). Everything is batched as one masked GEMM per (cell, half) — the
subset-sum identity of `.claude/rules/vectorize-many-cell-fits.md` item 3;
no per-draw pool re-reduction. Runs in seconds on CPU.

Reuses the #1112 geometry loaders (`load_store` / `delta_cloud`) — no
re-implemented pooling. 0 GPU-h; reads only the persisted `pooled.pt` stores.

Usage:
    uv run python scripts/issue1112_debiased_cosine.py \
        [--capture-root data/issue_1112/geometry_stage/capture] \
        [--out eval_results/issue_1112/geometry/debiased_cosine.json] \
        [--draws 2000] [--layer 14] [--arm response] [--seed 1112]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_653.spectral import cosine  # noqa: E402
from explore_persona_space.experiments.issue_1112 import PRIMARY_LAYER  # noqa: E402
from explore_persona_space.experiments.issue_1112.geometry import (  # noqa: E402
    _row_keys,
    delta_cloud,
    load_store,
)

logger = logging.getLogger("issue1112.debiased_cosine")

# Registered cross-mix pairs (body H2 / H2b; geometry.DIFF_PAIRS subset that
# the mean-shift-direction read covers): (name, cell_a, cell_b).
PAIRS = (
    ("H2_negatives_lorapos_vs_loraneg", "s2_lora_pos", "s1_lora_neg"),
    ("H2b_negatives_ftpos_vs_ftneg", "s4_fullft_pos", "s3_fullft_neg"),
)
BASE_CELL = "base_sycophancy"
DOSE = "selected"
BASE_DOSE = "base"
COS_CUTOFF = 0.8  # the registered cutoff the body's H2 read straddles
QUANTILES = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
EPS = 1e-12


def half_partition_masks(
    n: int,
    n_draws: int,
    seed: int,
    *,
    question_idx: np.ndarray | None = None,
) -> np.ndarray:
    """(n_draws, n) boolean half-A masks — exact half, WITHOUT replacement.

    ``question_idx is None`` -> `row_random`: uniform random n//2-of-n row
    halves (fully vectorized via per-draw argsort of a random matrix).
    ``question_idx`` given -> `question_aligned`: ONE half-partition of the
    unique question ids per draw, applied to all rows (requires an even
    question count and equal rows per question so halves stay exact).
    Asserts every draw's half A has the identical size m.
    """
    assert n % 2 == 0, n
    m = n // 2
    rng = np.random.default_rng(seed)
    if question_idx is None:
        order = np.argsort(rng.random((n_draws, n)), axis=1)
        masks = np.zeros((n_draws, n), dtype=bool)
        np.put_along_axis(masks, order[:, :m], True, axis=1)
    else:
        q = np.asarray(question_idx)
        assert q.shape == (n,), (q.shape, n)
        qs = np.unique(q)
        assert len(qs) % 2 == 0, f"question-aligned halves need an even count, got {len(qs)}"
        # (n_draws, n_q) boolean: is question rank < half? -> map to rows.
        q_order = np.argsort(rng.random((n_draws, len(qs))), axis=1)
        q_in_a = np.zeros((n_draws, len(qs)), dtype=bool)
        np.put_along_axis(q_in_a, q_order[:, : len(qs) // 2], True, axis=1)
        q_pos = np.searchsorted(qs, q)  # row -> question rank in `qs`
        masks = q_in_a[:, q_pos]
    counts = masks.sum(axis=1)
    assert (counts == m).all(), f"unequal half sizes: {np.unique(counts)} != {m}"
    return masks


def batched_half_cosines(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-draw cosines for one pair from one masked GEMM per (cell, half).

    Returns per-draw arrays:
      - ``cross``: cos(mean_A(cell_a), mean_A(cell_b)) — SAME half-A rows in
        both cells (the paired cross-mix subsample cosine);
      - ``ref_a`` / ``ref_b``: cos(mean_A, mean_B) WITHIN each cell (the
        same-cell split-half attenuation reference at the same m);
      - ``corrected``: cross / sqrt(ref_a * ref_b) where both refs > 0
        (NaN otherwise — dropped, never coerced, at summary time).
    """
    assert cloud_a.shape == cloud_b.shape and cloud_a.ndim == 2, (cloud_a.shape, cloud_b.shape)
    _n_draws, n = masks.shape
    assert cloud_a.shape[0] == n, (cloud_a.shape, n)
    m = int(masks[0].sum())
    sel_a = masks.astype(np.float64)
    sel_b = (~masks).astype(np.float64)

    def half_means(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(cloud, dtype=np.float64)
        return (sel_a @ x) / m, (sel_b @ x) / (n - m)

    def row_cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        num = np.einsum("ij,ij->i", u, v)
        den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        return np.where(den > EPS, num / np.maximum(den, EPS), 0.0)

    a_half_a, a_half_b = half_means(cloud_a)
    b_half_a, b_half_b = half_means(cloud_b)
    cross = row_cos(a_half_a, b_half_a)
    ref_a = row_cos(a_half_a, a_half_b)
    ref_b = row_cos(b_half_a, b_half_b)
    prod = ref_a * ref_b
    valid = (ref_a > 0.0) & (ref_b > 0.0)
    corrected = np.where(valid, cross / np.sqrt(np.where(valid, prod, 1.0)), np.nan)
    return {"cross": cross, "ref_a": ref_a, "ref_b": ref_b, "corrected": corrected}


def serial_half_cosines_reference(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Serial per-draw twin of :func:`batched_half_cosines` (test oracle only).

    Loops draws with the #653 scalar :func:`cosine`; kept CONTAINED in this
    equivalence-reference role per the vectorize-rule tombstone convention.
    """
    out: dict[str, list[float]] = {"cross": [], "ref_a": [], "ref_b": [], "corrected": []}
    for mask in masks:
        a_half_a, a_half_b = cloud_a[mask].mean(axis=0), cloud_a[~mask].mean(axis=0)
        b_half_a, b_half_b = cloud_b[mask].mean(axis=0), cloud_b[~mask].mean(axis=0)
        cross = cosine(a_half_a, b_half_a)
        ref_a = cosine(a_half_a, a_half_b)
        ref_b = cosine(b_half_a, b_half_b)
        out["cross"].append(cross)
        out["ref_a"].append(ref_a)
        out["ref_b"].append(ref_b)
        out["corrected"].append(
            cross / float(np.sqrt(ref_a * ref_b)) if ref_a > 0 and ref_b > 0 else np.nan
        )
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def summarize(values: np.ndarray, *, cutoff: float = COS_CUTOFF) -> dict:
    """Mean/std/quantiles + fraction below the registered cutoff (NaNs dropped)."""
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    n_dropped = int(vals.size - finite.size)
    assert finite.size > 0, "all draws NaN"
    return {
        "n_draws": int(vals.size),
        "n_dropped_nan": n_dropped,
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "quantiles": {str(q): float(np.quantile(finite, q)) for q in QUANTILES},
        "frac_below_cutoff": float((finite < cutoff).mean()),
        "cutoff": cutoff,
    }


def analyze_pair(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    masks: np.ndarray,
) -> dict:
    """One (pair, partition-scheme) read: draws, summaries, paired deltas."""
    draws = batched_half_cosines(cloud_a, cloud_b, masks)
    ref_min = np.minimum(draws["ref_a"], draws["ref_b"])
    prod = draws["ref_a"] * draws["ref_b"]
    valid = (draws["ref_a"] > 0.0) & (draws["ref_b"] > 0.0)
    ref_geo = np.where(valid, np.sqrt(np.where(valid, prod, 1.0)), np.nan)
    deltas = {
        "ref_a_minus_cross": draws["ref_a"] - draws["cross"],
        "ref_b_minus_cross": draws["ref_b"] - draws["cross"],
        "ref_min_minus_cross": ref_min - draws["cross"],
        "ref_geomean_minus_cross": ref_geo - draws["cross"],
    }
    entry: dict = {
        "point_cos_full_cloud": cosine(cloud_a.mean(axis=0), cloud_b.mean(axis=0)),
        "m": int(masks[0].sum()),
        "summary": {k: summarize(v) for k, v in draws.items()},
        # Paired deltas answer (c): reference - cross-mix, per shared draw.
        # frac_positive = fraction of draws where the cross-mix cosine sits
        # BELOW that reference; quantiles bound the paired difference.
        "paired_deltas": {
            k: {
                "mean": float(np.nanmean(v)),
                "quantiles": {str(q): float(np.nanquantile(v, q)) for q in QUANTILES},
                "frac_positive": float(np.nanmean(v > 0)),
            }
            for k, v in deltas.items()
        },
        "draws": {k: [round(float(x), 6) for x in v] for k, v in draws.items()},
    }
    return entry


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    """Load the four selected-dose clouds + base, run both schemes per pair,
    write the debiased-cosine JSON (fails loud on any missing store)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--capture-root", type=Path, default=Path("data/issue_1112/geometry_stage/capture")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("eval_results/issue_1112/geometry/debiased_cosine.json")
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--arm", type=str, default="response")
    parser.add_argument("--seed", type=int, default=1112)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    base = load_store(args.capture_root / BASE_CELL / BASE_DOSE / "pooled.pt")
    question_idx = np.asarray([q for _, q in _row_keys(base)])
    n = len(question_idx)

    # ONE mask set per scheme, shared across pairs AND across the cross/ref
    # statistics of each draw => every comparison is PAIRED on the draw axis.
    masks_by_scheme = {
        "row_random": half_partition_masks(n, args.draws, args.seed),
        "question_aligned": half_partition_masks(
            n, args.draws, args.seed + 1, question_idx=question_idx
        ),
    }

    t0 = time.time()
    results: dict[str, dict] = {}
    for name, cell_a, cell_b in PAIRS:
        store_a = load_store(args.capture_root / cell_a / DOSE / "pooled.pt")
        store_b = load_store(args.capture_root / cell_b / DOSE / "pooled.pt")
        cloud_a = delta_cloud(store_a, base, args.arm, args.layer)
        cloud_b = delta_cloud(store_b, base, args.arm, args.layer)
        results[name] = {
            "cell_a": cell_a,
            "cell_b": cell_b,
            "dose": DOSE,
            "arm": args.arm,
            "layer": args.layer,
            "n_rows": int(cloud_a.shape[0]),
            "schemes": {
                scheme: analyze_pair(cloud_a, cloud_b, masks)
                for scheme, masks in masks_by_scheme.items()
            },
        }
        logger.info("[debiased-cosine] %s done (%.1fs elapsed)", name, time.time() - t0)

    payload = {
        "schema_version": 1,
        "description": (
            "Paired subsample-WITHOUT-replacement half-draw cosine CIs (m = n/2, same row "
            "indices both cells per draw) with same-cell split-half attenuation references "
            "at the same m, replacing the resample-deflated with-replacement bootstrap read."
        ),
        "notes": (
            "cross uses the SAME half-A rows in both cells, so row-level noise correlated "
            "across cells (shared base-row subtraction) attenuates cross LESS than the "
            "independent-noise same-cell reference — a cross distribution sitting below the "
            "reference is therefore conservative evidence of a genuine direction difference. "
            "corrected = cross / sqrt(ref_a*ref_b) assumes independent noise across cells and "
            "is reported as a plug-in, not a headline."
        ),
        "pairs": results,
        "n_draws": args.draws,
        "seed": args.seed,
        "cutoff": COS_CUTOFF,
        "quantiles": list(QUANTILES),
        "capture_root": str(args.capture_root),
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info("[debiased-cosine] wrote %s (%.1fs)", args.out, time.time() - t0)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
