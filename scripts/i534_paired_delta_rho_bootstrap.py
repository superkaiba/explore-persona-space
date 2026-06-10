"""#534 free-analysis follow-up: paired row-bootstrap on Δρ between the two usable checkpoints.

The round-1 clean-result read "grows into the stop" (frac 0.75 → 1.00) rested on
point estimates. This script puts a paired interval on the between-checkpoint
change in the two headline partial Spearman correlations: the SAME 432
(probe × arm × seed) rows exist at both fractions, so each bootstrap iteration
resamples row keys ONCE and recomputes the partial ρ at both fractions on the
identical resampled rows, then takes the difference ρ(1.00) − ρ(0.75).

Runs purely over committed eval JSONs (zero GPU). Writes
eval_results/issue_534/paired_delta_rho_bootstrap.json.

Usage:
    uv run python scripts/i534_paired_delta_rho_bootstrap.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("i534_paired_delta_rho")

HEADLINE_PREDICTORS = ("shadow_angle", "d_nearest_neg_nd")


def _git_sha() -> str:
    """Return the current short git SHA (for reproducibility metadata)."""
    return subprocess.run(
        ["git", "rev-parse", "--short=9", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def paired_bootstrap_delta_rho(
    rows_a: list[dict],
    rows_b: list[dict],
    predictor: str,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Paired percentile CI on ρ_b − ρ_a for one predictor.

    rows_a / rows_b are the build_rows() pools at the earlier / later fraction.
    Rows are paired by (cell, seed, probe); asserts a perfect 1:1 key match.
    Each iteration resamples the SHARED key index once and recomputes the
    partial Spearman (exact production estimator) at both fractions on the
    identical resampled keys.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        PREDICTORS,
        _partial_spearman,
    )

    def _key(r: dict) -> tuple:
        return (r["cell"], r["seed"], r["probe"])

    a_by_key = {_key(r): r for r in rows_a}
    b_by_key = {_key(r): r for r in rows_b}
    assert set(a_by_key) == set(b_by_key), (
        f"row-key mismatch: {len(a_by_key)} vs {len(b_by_key)} keys, "
        f"sym-diff {len(set(a_by_key) ^ set(b_by_key))}"
    )
    keys = sorted(a_by_key)
    n = len(keys)

    def _cols(by_key: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
        cols = {p: np.asarray([by_key[k][p] for k in keys], dtype=np.float64) for p in PREDICTORS}
        y = np.asarray([by_key[k]["delta_g"] for k in keys], dtype=np.float64)
        return cols, y

    cols_a, y_a = _cols(a_by_key)
    cols_b, y_b = _cols(b_by_key)

    def _rho(cols: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray) -> float | None:
        return _partial_spearman(
            y[idx].tolist(),
            cols[predictor][idx].tolist(),
            [cols[q][idx].tolist() for q in PREDICTORS if q != predictor],
        )

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # ONE shared resample → paired
        ra = _rho(cols_a, y_a, idx)
        rb = _rho(cols_b, y_b, idx)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (ra, rb)):
            n_failed += 1
            continue
        vals.append(float(rb) - float(ra))
    point = None
    full_idx = np.arange(n)
    ra0, rb0 = _rho(cols_a, y_a, full_idx), _rho(cols_b, y_b, full_idx)
    if ra0 is not None and rb0 is not None:
        point = float(rb0) - float(ra0)
    lo, hi = (None, None)
    if vals:
        lo, hi = (float(v) for v in np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    share_signed = None
    if vals and point is not None:
        sign = math.copysign(1.0, point) if point != 0 else 1.0
        share_signed = float(np.mean([math.copysign(1.0, v) == sign for v in vals]))
    return {
        "n_rows_paired": n,
        "point_delta_rho": point,
        "lo": lo,
        "hi": hi,
        "alpha": alpha,
        "n_boot": n_boot,
        "n_failed": n_failed,
        "share_resamples_matching_point_sign": share_signed,
    }


def main(argv: list[str] | None = None) -> int:
    """Build the frac 0.75 / 1.00 row pools and bootstrap the paired Δρ."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_534"))
    ap.add_argument(
        "--phase05-path", type=Path, default=Path("eval_results/issue_530/phase0_5_gates.json")
    )
    ap.add_argument("--frac-a", type=float, default=0.75)
    ap.add_argument("--frac-b", type=float, default=1.0)
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=534)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=i534_paired_delta_rho] %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        aggregate_base_prior_from_trajectories,
        build_rows,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        load_phase05,
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gates = load_phase05(args.phase05_path)
    base_prior = aggregate_base_prior_from_trajectories(
        slab_root=args.slab_root, seeds=seeds, positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3
    )

    pools = {}
    for f in (args.frac_a, args.frac_b):
        pooled = build_rows(
            slab_root=args.slab_root,
            chosen_frac=f,
            per_probe=gates["per_probe"],
            arm_to_positioned_n=gates["arm_to_positioned_n"],
            seeds=seeds,
            base_prior_by_probe=base_prior or None,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            dg_band=None,
        )
        pools[f] = pooled["rows"]
        log.info("frac %.2f: %d rows", f, len(pooled["rows"]))

    out = {
        "schema_version": "1",
        "task_id": 534,
        "frac_a": args.frac_a,
        "frac_b": args.frac_b,
        "delta_is": "rho(frac_b) - rho(frac_a), paired by (cell, seed, probe)",
        "per_predictor": {
            p: paired_bootstrap_delta_rho(
                pools[args.frac_a],
                pools[args.frac_b],
                p,
                n_boot=args.n_boot,
                seed=args.boot_seed,
            )
            for p in HEADLINE_PREDICTORS
        },
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path = args.out or (args.slab_root / "paired_delta_rho_bootstrap.json")
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", out_path)
    print(json.dumps(out["per_predictor"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
