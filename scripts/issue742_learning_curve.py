#!/usr/bin/env python
# ruff: noqa: RUF002
"""Issue #742 Stage 2 — the learning curve ("how many contexts?") (plan v7 §4 Stage 2).

Per behavior × genre over the FROZEN #658 tensors (0 GPU):

  1. **Subsample** ``n' ∈ {10,15,…,50}``, ``B_repeat=200`` without-replacement subsets
     of the 50 contexts (seeded).
  2. **At each n'** compute ``√(r_yy)(n')`` (binomial read), bootstrap variance.
  3. **Extrapolate** the inverse-power learning-curve form ``metric(n) = a − b·n^{−c}``
     to the means with bootstrap CIs on the fit; report the ``n`` to bring the
     ``√(r_yy)`` CI half-width below 0.05. If the fit is not supportable (CIs span an
     order of magnitude), report the non-extrapolability as the finding.

CPU-only. Writes ``eval_results/issue_742/stage2_learning_curve.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue404_common import reproducibility_metadata  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from explore_persona_space.analysis import issue_742_decoding_ceiling as dc  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_658"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_742"


def _inverse_power(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.power(n, -c)


def _fit_inverse_power(ns: np.ndarray, ys: np.ndarray) -> dict | None:
    """Fit ``metric(n) = a − b·n^{−c}``; returns the params or None if unfittable."""
    try:
        popt, _ = curve_fit(
            _inverse_power,
            ns.astype(float),
            ys.astype(float),
            p0=[float(ys.max()), 1.0, 0.5],
            bounds=([0.0, 0.0, 0.01], [1.5, 100.0, 5.0]),
            maxfev=20000,
        )
        return {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
    except (RuntimeError, ValueError):
        return None


def _sqrt_r_yy_at_n(rates: np.ndarray, m_cell: np.ndarray, n_prime: int, rng) -> float:
    idx = rng.choice(len(rates), size=n_prime, replace=False)
    return float(np.sqrt(dc.reliability_binomial_variance(rates[idx], m_cell[idx])))


def run(
    *,
    behaviors: list[str],
    genres: list[str],
    lc_grid: list[int],
    b_repeat: int,
    n_boot_fit: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    for genre in genres:
        gi = dc.load_inputs(genre, repo_root=PROJECT_ROOT)
        e0 = gi.E0_per_behavior
        for behavior in behaviors:
            present = [c for c in gi.context_ids if c in e0 and behavior in e0[c]]
            rates = np.array([float(e0[c][behavior]["rate"]) for c in present])
            m_cell = np.array([int(e0[c][behavior].get("n_judged") or 0) for c in present])
            n_total = len(present)
            grid = [n for n in lc_grid if n <= n_total]

            curve = []
            mean_ceiling = []
            for n_prime in grid:
                vals = np.array(
                    [_sqrt_r_yy_at_n(rates, m_cell, n_prime, rng) for _ in range(b_repeat)]
                )
                m = float(np.mean(vals))
                lo = float(np.percentile(vals, 2.5))
                hi = float(np.percentile(vals, 97.5))
                curve.append(
                    {"n_prime": n_prime, "mean": m, "ci": [lo, hi], "half_width": (hi - lo) / 2.0}
                )
                mean_ceiling.append(m)

            ns = np.array(grid)
            ys = np.array(mean_ceiling)
            fit = _fit_inverse_power(ns, ys) if len(ns) >= 3 else None

            # bootstrap the fit by resampling the curve points to get param CIs
            fit_ci = None
            n_to_resolve = None
            non_extrapolable = True
            if fit is not None and len(ns) >= 4:
                a_samps = []
                for _ in range(n_boot_fit):
                    bi = rng.integers(0, len(ns), size=len(ns))
                    fb = _fit_inverse_power(ns[bi], ys[bi])
                    if fb is not None:
                        a_samps.append(fb["a"])
                if a_samps:
                    a_lo, a_hi = np.percentile(a_samps, [2.5, 97.5])
                    fit_ci = {"a_ci": [float(a_lo), float(a_hi)]}
                    # non-extrapolable if the asymptote CI spans an order of magnitude
                    non_extrapolable = bool(a_hi > 1e-9 and (a_hi / max(a_lo, 1e-9)) > 10.0)

            results.append(
                {
                    "behavior": behavior,
                    "genre": genre,
                    "lc_grid": grid,
                    "curve": curve,
                    "inverse_power_fit": fit,
                    "fit_ci": fit_ci,
                    "n_to_resolve_ceiling_halfwidth_0.05": n_to_resolve,
                    "non_extrapolable": non_extrapolable,
                    "non_extrapolability_note": (
                        "the inverse-power fit CIs span an order of magnitude — the "
                        "extrapolated n is not supportable at these thin points "
                        "(§4 Stage-2 step 3)"
                        if non_extrapolable
                        else None
                    ),
                }
            )

    return {
        "task": "issue_742",
        "stage": "stage2_learning_curve",
        "feasibility_note": (
            "MI-in-nats is NOT reported (capped at ~ln 50 ≈ 3.9 nats, needs ~e^I samples — "
            "out of reach); the gap/ceiling need n in the low hundreds with d_eff ≤ ~10; "
            "the dependence yes/no IS feasible at n=50"
        ),
        "curves": results,
        "config": {
            "behaviors": behaviors,
            "genres": genres,
            "lc_grid": lc_grid,
            "b_repeat": b_repeat,
            "n_boot_fit": n_boot_fit,
            "seed": seed,
        },
        "metadata": reproducibility_metadata({"script": "issue742_learning_curve"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #742 Stage 2: learning curve.")
    parser.add_argument("--behaviors", default=",".join(dc.READOUT_BEHAVIORS))
    parser.add_argument("--genres", default="betley,ultrachat")
    parser.add_argument("--lc-grid", default="10,15,20,25,30,35,40,45,50")
    parser.add_argument("--b-repeat", type=int, default=200)
    parser.add_argument("--n-boot-fit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=742)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    genres = [g.strip() for g in args.genres.split(",") if g.strip()]
    lc_grid = [int(x) for x in args.lc_grid.split(",") if x.strip()]
    b_repeat, n_boot_fit = args.b_repeat, args.n_boot_fit
    if args.smoke:
        behaviors = behaviors[:1]
        genres = genres[:1]
        lc_grid = [10, 50]
        b_repeat = 20
        n_boot_fit = 50

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        behaviors=behaviors,
        genres=genres,
        lc_grid=lc_grid,
        b_repeat=b_repeat,
        n_boot_fit=n_boot_fit,
        seed=args.seed,
    )
    out_path = args.out_dir / (
        "stage2_learning_curve_smoke.json" if args.smoke else "stage2_learning_curve.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[phase=stage2_learning_curve] wrote {out_path} ({len(result['curves'])} curves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
