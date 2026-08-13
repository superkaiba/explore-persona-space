"""Measured-peak calibration for the #1336 null-draw chunk cap (issue825 fit core).

Reproduces the pooled-fit OOM's allocation SHAPE at a scaled-down ``n_train``
and measures the real per-chunk live peak of the batched null path in
``issue825_fit_cells._null_ss_contrib`` — the calibration input for
``NULL_DRAW_LIVE_FACTOR`` (the `resolve_chunk_cap` / `live_factor` convention:
counting explicit temporaries under-estimates the true peak, so the factor
MUST come from a measured peak; see vectorized_mlp_skill.resolve_chunk_cap).

Run each configuration in a FRESH process (`ru_maxrss` is a monotone process
high-water mark; a second run in the same process reads a ~0 delta):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue1336_null_chunk_calibration.py \
        --n-rows 20000 --d 512 --null-draws 0            # baseline (no null pass)
    ... --null-draws 20 --chunk 20                        # one chunk of 20 draws
    ... --null-draws 20 --chunk 5                         # 4 chunks of 5 draws
    ... --null-draws 20 --chunk 20 --grid-points 13       # lambda-grid invariance

The live factor is ``(maxrss(B) - maxrss(A)) / (chunk x unit)`` with
``unit = (n_tr_fold + n_te_fold) x d x 8`` bytes (one draw's permuted fp64
train+test rows) — the same unit `resolve_null_draw_chunk` sizes against.

Incident arithmetic this scales down (job 12643, pooled_dpo_arm_off):
n_train=149,964 rows over 5 group folds -> n_tr_fold=118,359 / n_te_fold=31,605,
d=4096, null_draws=20 in ONE chunk (NULL_DRAW_BATCH default 64):
Yp_tr = (20, 118359, 4096) fp64 = 72.24 GiB (the failing request's exact size),
resident at failure = fold caches 11.05 + Y_t 4.58 + Yp_tr 72.24 + Yp_te 19.29
= 108.16 GiB (reported: 108.21 GiB PyTorch-allocated).
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402


def _maxrss_bytes() -> int:
    """Linux ru_maxrss is KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--n-rows", type=int, default=20000)
    ap.add_argument("--d", type=int, default=512, help="d_in == d_out (square map like production)")
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=20, help="NULL_DRAW_BATCH override")
    ap.add_argument("--grid-points", type=int, default=23)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument(
        "--n-inner", type=int, default=2, help="N_INNER_LAMBDA_FOLDS (pooled regime: 2)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import issue825_fit_cells as fc

    fc.N_INNER_LAMBDA_FOLDS = args.n_inner
    fc.NULL_DRAW_BATCH = args.chunk

    rng = np.random.default_rng(args.seed)
    n, d = args.n_rows, args.d
    X = rng.standard_normal((n, 1, d), dtype=np.float32)
    Y = rng.standard_normal((n, 1, d), dtype=np.float32)
    conv_ids = np.arange(n)
    grid = np.logspace(-3, 8, args.grid_points)

    rss_before = _maxrss_bytes()
    t0 = time.time()
    sweep = fc.heldout_r2_sweep(
        X,
        Y,
        conv_ids,
        n_folds=args.n_folds,
        seed=args.seed,
        null_draws=args.null_draws,
        collect_lambdas=True,
        lambdas=grid,
        reduced_basis_companion=False,  # n_tr > d in this regime; companion never runs
    )
    wall = time.time() - t0
    rss_after = _maxrss_bytes()

    n_tr_fold = int(np.ceil(n * (args.n_folds - 1) / args.n_folds))
    n_te_fold = n - n_tr_fold
    unit = (n_tr_fold + n_te_fold) * d * 8  # == n * d * 8: one draw's fp64 permuted rows
    b = min(args.chunk, args.null_draws) if args.null_draws else 0
    print(
        f"[calib] n={n} d={d} folds={args.n_folds} inner={args.n_inner} "
        f"draws={args.null_draws} chunk={args.chunk} grid={args.grid_points} "
        f"r2_obs[0]={float(sweep['r2_obs'][0]):.6f} wall={wall:.1f}s"
    )
    print(
        f"[calib] maxrss_before={rss_before / 2**30:.3f} GiB "
        f"maxrss_after={rss_after / 2**30:.3f} GiB "
        f"unit={(unit / 2**20):.1f} MiB b={b} b_x_unit={(b * unit / 2**30):.3f} GiB"
    )
    if b:
        print(
            "[calib] NOTE: live factor = (maxrss_after(this run) - maxrss_after(draws=0 "
            f"baseline)) / {b * unit} bytes — compute across fresh processes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
