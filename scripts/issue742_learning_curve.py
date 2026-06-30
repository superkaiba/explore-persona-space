#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
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


def _sqrt_r_yy_at_n(rates: np.ndarray, m_cell: np.ndarray, idx: np.ndarray) -> float:
    return float(np.sqrt(dc.reliability_binomial_variance(rates[idx], m_cell[idx])))


def _rho_lin_at_n(v0_layer: np.ndarray, rates: np.ndarray, idx: np.ndarray) -> float:
    """LOCO-CV ridge held-out ρ on a size-len(idx) context subsample (§4 Stage-2 step 2)."""
    return dc.loco_ridge_refit_rho(v0_layer[idx], rates[idx])


def _curve_for_metric(grid, b_repeat, draw_idx_fn, metric_fn) -> tuple[list[dict], np.ndarray]:
    """Build a metric(n') curve: B_repeat without-replacement subsamples per n'.

    Returns ``(curve_records, mean_per_n)``. ``draw_idx_fn(n_prime) -> idx`` draws one
    without-replacement subsample; ``metric_fn(idx) -> float`` evaluates the metric on it.
    """
    curve, means = [], []
    for n_prime in grid:
        vals = np.array([metric_fn(draw_idx_fn(n_prime)) for _ in range(b_repeat)], dtype=float)
        vals = vals[np.isfinite(vals)]
        m = float(np.mean(vals)) if vals.size else float("nan")
        lo = float(np.percentile(vals, 2.5)) if vals.size else float("nan")
        hi = float(np.percentile(vals, 97.5)) if vals.size else float("nan")
        curve.append({"n_prime": n_prime, "mean": m, "ci": [lo, hi], "half_width": (hi - lo) / 2.0})
        means.append(m)
    return curve, np.array(means, dtype=float)


def _extrapolate_n_for_halfwidth(curve: list[dict], target_hw: float) -> int | None:
    """Smallest n in {50,75,...,1000} where the inverse-power-fit half-width < target.

    Fits half_width(n') = a − b·n'^{−c} to the curve's per-n half-widths and inverts:
    returns the smallest grid n whose predicted half-width drops below ``target_hw``;
    None when the half-width never crosses the target on the search grid (or unfittable).
    """
    ns = np.array([c["n_prime"] for c in curve], dtype=float)
    hw = np.array([c["half_width"] for c in curve], dtype=float)
    ok = np.isfinite(hw)
    if ok.sum() < 3:
        return None
    # half-width SHRINKS with n -> fit hw(n) = a + b·n^{−c} (b≥0, a≥0 floor)
    try:
        popt, _ = curve_fit(
            lambda n, a, b, c: a + b * np.power(n, -c),
            ns[ok],
            hw[ok],
            p0=[0.0, float(hw[ok].max()), 0.5],
            bounds=([0.0, 0.0, 0.01], [1.0, 1e4, 5.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    a, b, c = popt
    for n in range(50, 1001, 25):
        if a + b * (n**-c) < target_hw:
            return int(n)
    return None


def _extrapolate_n_for_gap(
    rho_curve: list[dict], ceil_curve: list[dict], target_gap_r2: float
) -> int | None:
    """Smallest n where the extrapolated (√r_yy² − ρ_lin²) R² gap drops below target.

    Fits inverse-power curves to ρ_lin(n') and √(r_yy)(n') asymptotes, predicts each at
    n ∈ {50,...,1000}, and returns the smallest n where the predicted R²-space gap
    (ceiling² − ρ²) falls below ``target_gap_r2`` (the 0.05-R² gap, §4 Stage-2 step 3).
    None when unfittable or never crossing.
    """
    rn = np.array([c["n_prime"] for c in rho_curve], dtype=float)
    ry = np.array([c["mean"] for c in rho_curve], dtype=float)
    cy = np.array([c["mean"] for c in ceil_curve], dtype=float)
    if np.sum(np.isfinite(ry)) < 3 or np.sum(np.isfinite(cy)) < 3:
        return None
    rfit = _fit_inverse_power(rn[np.isfinite(ry)], ry[np.isfinite(ry)])
    cfit = _fit_inverse_power(rn[np.isfinite(cy)], cy[np.isfinite(cy)])
    if rfit is None or cfit is None:
        return None
    for n in range(50, 1001, 25):
        rho_n = rfit["a"] - rfit["b"] * (n ** -rfit["c"])
        ceil_n = cfit["a"] - cfit["b"] * (n ** -cfit["c"])
        gap_r2 = max(0.0, ceil_n**2 - rho_n**2)
        if gap_r2 < target_gap_r2:
            return int(n)
    return None


def _curves_for_cell(
    *,
    behavior: str,
    genre: str,
    layer: int,
    rates: np.ndarray,
    m_cell: np.ndarray,
    v0_layer: np.ndarray,
    grid: list[int],
    b_repeat: int,
    n_boot_fit: int,
    d_eff: int,
    rng: np.random.Generator,
) -> dict:
    """All three Stage-2 metric curves + extrapolations for one (behavior, genre) cell.

    A module-level helper (not a nested closure) so the subsample-metric lambdas bind
    THESE parameters, not a loop variable (ruff B023). Builds √(r_yy)(n'), ρ_lin(n'),
    and dCor(n') curves over the subsample ``grid`` (B_repeat draws each), fits the
    inverse-power form to the ceiling curve, and computes the two registered
    extrapolations (§4 Stage-2 step 2/3).
    """
    n_total = len(rates)

    def _draw(n_prime: int) -> np.ndarray:
        return rng.choice(n_total, size=n_prime, replace=False)

    ceil_curve, ceil_means = _curve_for_metric(
        grid, b_repeat, _draw, lambda idx: _sqrt_r_yy_at_n(rates, m_cell, idx)
    )
    rho_curve, _ = _curve_for_metric(
        grid, b_repeat, _draw, lambda idx: _rho_lin_at_n(v0_layer, rates, idx)
    )
    dcor_curve, _ = _curve_for_metric(
        grid,
        b_repeat,
        _draw,
        lambda idx: dc.dcor_at_subsample(
            v0_layer[idx], rates[idx], n_prime=len(idx), d_eff=d_eff, rng=rng
        ),
    )

    ns = np.array(grid)
    ys = ceil_means
    fit = _fit_inverse_power(ns, ys) if len(ns) >= 3 else None

    fit_ci = None
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
            non_extrapolable = bool(a_hi > 1e-9 and (a_hi / max(a_lo, 1e-9)) > 10.0)

    # BLOCKER-fix stage2-incomplete: the two registered extrapolations.
    n_to_resolve_hw = (
        None if non_extrapolable else _extrapolate_n_for_halfwidth(ceil_curve, target_hw=0.05)
    )
    n_to_resolve_gap = (
        None
        if non_extrapolable
        else _extrapolate_n_for_gap(rho_curve, ceil_curve, target_gap_r2=0.05)
    )

    return {
        "behavior": behavior,
        "genre": genre,
        "layer": layer,
        "lc_grid": grid,
        "sqrt_r_yy_curve": ceil_curve,
        "rho_lin_curve": rho_curve,
        "dcor_curve": dcor_curve,
        "inverse_power_fit": fit,
        "fit_ci": fit_ci,
        "n_to_resolve_ceiling_halfwidth_0.05": n_to_resolve_hw,
        "n_to_resolve_gap_r2_0.05": n_to_resolve_gap,
        "non_extrapolable": non_extrapolable,
        "non_extrapolability_note": (
            "the inverse-power fit CIs span an order of magnitude — the extrapolated n "
            "is not supportable at these thin points (§4 Stage-2 step 3)"
            if non_extrapolable
            else None
        ),
    }


def run(
    *,
    behaviors: list[str],
    genres: list[str],
    lc_grid: list[int],
    b_repeat: int,
    n_boot_fit: int,
    seed: int,
    d_eff: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    for genre in genres:
        gi = dc.load_inputs(genre, repo_root=PROJECT_ROOT)
        e0 = gi.E0_per_behavior
        v0_all = dc.stack_v0(gi.v0_dict, recipe="last")  # (n, n_layers, d)
        for behavior in behaviors:
            present = [c for c in gi.context_ids if c in e0 and behavior in e0[c]]
            rates = np.array([float(e0[c][behavior]["rate"]) for c in present])
            m_cell = np.array([int(e0[c][behavior].get("n_judged") or 0) for c in present])
            layer = dc.load_a33_layer(behavior, genre, eval_dir=EVAL_DIR)
            v0_layer = v0_all[: len(present), layer, :]
            grid = [n for n in lc_grid if n <= len(present)]
            results.append(
                _curves_for_cell(
                    behavior=behavior,
                    genre=genre,
                    layer=layer,
                    rates=rates,
                    m_cell=m_cell,
                    v0_layer=v0_layer,
                    grid=grid,
                    b_repeat=b_repeat,
                    n_boot_fit=n_boot_fit,
                    d_eff=d_eff,
                    rng=rng,
                )
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
            "d_eff": d_eff,
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
    parser.add_argument("--d-eff", type=int, default=10, help="PCA dim for the dCor(n') curve")
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
        d_eff=args.d_eff,
    )
    out_path = args.out_dir / (
        "stage2_learning_curve_smoke.json" if args.smoke else "stage2_learning_curve.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[phase=stage2_learning_curve] wrote {out_path} ({len(result['curves'])} curves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
