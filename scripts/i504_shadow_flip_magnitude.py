# ruff: noqa: RUF001, RUF002  # em-dash, minus-sign, times, Greek intentional
"""#504 followup shadow-flip-magnitude-nats: covariate-adjusted magnitude of the shadow flip.

The replicated shadow-angle sign flip (#530/#534, folded into #504's clean-result)
is a rank-order object: partial Spearman ρ(shadow_angle, leakage gain | 5 covariates)
< 0 at the two usable checkpoints (steps 15, 20). This script translates the flip
into NATS of marker log-prob gain so the claim carries an effect size:

  * per-checkpoint OLS of leakage gain (ΔG, nats) on the same 6 predictors the
    partial-Spearman fits use (zero-variance columns within a checkpoint —
    training_step — are dropped and recorded, never silently fit);
  * the shadow-angle coefficient scaled to its observed IQR and range, reported
    as "deep-end minus lateral-end" adjusted gain (−β×IQR, −β×range — positive
    = anti-shadow: deeper-in-shadow probes leak MORE);
  * a (probe × arm × seed) row-bootstrap percentile CI (the rows ARE the
    cluster units, same resampling grain as i534_paired_delta_rho_bootstrap.py);
  * ANCOVA-style adjusted means of the deep-shadow vs lateral shadow-angle
    terciles (tercile dummies + the 5 non-shadow covariates), with the same
    bootstrap CI, plus the UNADJUSTED tercile means alongside;
  * run at the usable checkpoints (steps 15, 20 — fracs 0.75, 1.00) with
    hypothesis weight, and DESCRIPTIVELY at the sub-floor checkpoints
    (steps 5, 10) where source ΔG sits below the 1-nat usability floor;
  * a banded frac=1.00 sensitivity row (the canonical [5,12]-nat source-ΔG
    band, gate-identical to the #530 replication object).

Hypothesis (epm:followup-scope v1, task #504): the flip survives in magnitude
but is small — ~0.2-0.5 nat of adjusted gain across the observed shadow-angle
range (vs ~0.9 nat SD of held-out gain), with the bootstrap CI excluding zero
at BOTH usable checkpoints. Falsification: CI spans zero at both → the shadow
claim downgrades to rank-only.

CPU-only, zero new data — runs over committed eval JSONs:
    uv run python scripts/i504_shadow_flip_magnitude.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.shadow_flip_magnitude")

# Fraction → step mapping is read off the trajectories; these are the analysis
# checkpoints (plan #534 trajectory grid).
USABLE_FRACTIONS: tuple[float, ...] = (0.75, 1.00)  # steps 15, 20
SUBFLOOR_FRACTIONS: tuple[float, ...] = (0.25, 0.50)  # steps 5, 10 — descriptive only


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS coefficients with intercept prepended (column 0)."""
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xb = np.concatenate([ones, X], axis=1)
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def _design(
    rows: list[dict], predictors: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build (y, X, kept_predictors, dropped_zero_variance) from rows.

    Zero-variance columns (training_step within a single checkpoint) are
    dropped and recorded — fitting a constant column alongside the intercept
    makes the coefficient an arbitrary least-norm split, not a measurement.
    """
    y = np.asarray([r["delta_g"] for r in rows], dtype=np.float64)
    kept, dropped = [], []
    cols = []
    for p in predictors:
        c = np.asarray([r[p] for r in rows], dtype=np.float64)
        if np.ptp(c) == 0.0:
            dropped.append(p)
            continue
        kept.append(p)
        cols.append(c)
    X = np.stack(cols, axis=1) if cols else np.empty((len(rows), 0))
    return y, X, kept, dropped


def shadow_magnitude_fit(rows: list[dict], predictors: list[str]) -> dict[str, Any]:
    """Point estimates: raw β_shadow + IQR/range scalings + tercile adjusted means."""
    y, X, kept, dropped = _design(rows, predictors)
    beta = _ols(y, X)
    coef = {p: float(beta[i + 1]) for i, p in enumerate(kept)}
    shadow = np.asarray([r["shadow_angle"] for r in rows], dtype=np.float64)
    q25, q75 = np.percentile(shadow, [25, 75])
    iqr = float(q75 - q25)
    rng_obs = float(np.ptp(shadow))
    b = coef["shadow_angle"]

    # Tercile ANCOVA: dummies for deep (lowest-angle) + middle terciles, lateral
    # (highest-angle) tercile as reference, adjusted for the 5 non-shadow
    # covariates (shadow_angle itself replaced by its tercile coding).
    t1, t2 = np.percentile(shadow, [100 / 3, 200 / 3])
    tercile = np.where(shadow <= t1, 0, np.where(shadow <= t2, 1, 2))  # 0=deep 2=lateral
    others = [p for p in kept if p != "shadow_angle"]
    Xo = (
        np.stack([np.asarray([r[p] for r in rows], dtype=np.float64) for p in others], axis=1)
        if others
        else np.empty((len(rows), 0))
    )
    d_deep = (tercile == 0).astype(np.float64)[:, None]
    d_mid = (tercile == 1).astype(np.float64)[:, None]
    Xt = np.concatenate([d_deep, d_mid, Xo], axis=1)
    beta_t = _ols(y, Xt)
    adj_deep_minus_lateral = float(beta_t[1])
    raw_means = {
        "deep": float(y[tercile == 0].mean()),
        "middle": float(y[tercile == 1].mean()),
        "lateral": float(y[tercile == 2].mean()),
        "n_per_tercile": [int((tercile == k).sum()) for k in (0, 1, 2)],
    }
    return {
        "n_rows": len(rows),
        "predictors_kept": kept,
        "predictors_dropped_zero_variance": dropped,
        "ols_coefficients": coef,
        "beta_shadow_nats_per_radian": b,
        "shadow_angle_observed": {"iqr_rad": iqr, "range_rad": rng_obs},
        # Positive = anti-shadow (deeper-in-shadow probes gain MORE leakage).
        "adjusted_gain_deep_minus_lateral_iqr_nats": -b * iqr,
        "adjusted_gain_deep_minus_lateral_range_nats": -b * rng_obs,
        "tercile_adjusted_mean_diff_deep_minus_lateral_nats": adj_deep_minus_lateral,
        "tercile_raw_means_nats": raw_means,
        "dv_sd_nats": float(y.std(ddof=1)),
        "dv_mean_nats": float(y.mean()),
    }


def bootstrap_magnitude(
    rows: list[dict],
    predictors: list[str],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Row-bootstrap percentile CIs on β_shadow, its IQR/range scalings, and the
    tercile adjusted-mean difference.

    Rows are the (probe × arm × seed) cluster units — resampling them with
    replacement is the same grain as i534_paired_delta_rho_bootstrap.py. The
    IQR/range scaling constants are FIXED at the full-sample observed values
    (the uncertainty quantified is in β; the scaling is a unit conversion).
    Tercile cutpoints are likewise fixed at the full-sample terciles.
    """
    point = shadow_magnitude_fit(rows, predictors)
    iqr = point["shadow_angle_observed"]["iqr_rad"]
    rng_obs = point["shadow_angle_observed"]["range_rad"]
    shadow_full = np.asarray([r["shadow_angle"] for r in rows], dtype=np.float64)
    t1, t2 = np.percentile(shadow_full, [100 / 3, 200 / 3])

    rng = np.random.default_rng(seed)
    n = len(rows)
    betas: list[float] = []
    adj_diffs: list[float] = []
    n_failed = 0
    others = [p for p in point["predictors_kept"] if p != "shadow_angle"]
    arr = {
        p: np.asarray([r[p] for r in rows], dtype=np.float64) for p in [*point["predictors_kept"]]
    }
    y_full = np.asarray([r["delta_g"] for r in rows], dtype=np.float64)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y = y_full[idx]
        cols = [arr[p][idx] for p in point["predictors_kept"]]
        # A resample can collapse a column's variance; lstsq still solves, but
        # the shadow coefficient is only meaningful if shadow varies.
        sh = arr["shadow_angle"][idx]
        if np.ptp(sh) == 0.0:
            n_failed += 1
            continue
        X = np.stack(cols, axis=1)
        beta = _ols(y, X)
        b = float(beta[1 + point["predictors_kept"].index("shadow_angle")])
        betas.append(b)
        terc = np.where(sh <= t1, 0, np.where(sh <= t2, 1, 2))
        if not ((terc == 0).any() and (terc == 2).any()):
            continue
        Xo = np.stack([arr[p][idx] for p in others], axis=1) if others else np.empty((n, 0))
        Xt = np.concatenate(
            [
                (terc == 0).astype(np.float64)[:, None],
                (terc == 1).astype(np.float64)[:, None],
                Xo,
            ],
            axis=1,
        )
        adj_diffs.append(float(_ols(y, Xt)[1]))

    def _ci(vals: list[float], scale: float = 1.0) -> dict[str, Any]:
        if not vals:
            return {"lo": None, "hi": None, "n": 0}
        lo, hi = np.percentile(np.asarray(vals) * scale, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"lo": float(lo), "hi": float(hi), "n": len(vals)}

    return {
        "alpha": alpha,
        "n_boot": n_boot,
        "n_failed_degenerate": n_failed,
        "beta_shadow_ci": _ci(betas),
        "adjusted_gain_deep_minus_lateral_iqr_ci": _ci(betas, scale=-iqr),
        "adjusted_gain_deep_minus_lateral_range_ci": _ci(betas, scale=-rng_obs),
        "tercile_adjusted_mean_diff_ci": _ci(adj_diffs),
    }


def per_seed_betas(rows: list[dict], predictors: list[str]) -> dict[str, Any]:
    """Per-seed β_shadow sign check (diagnostic — pooled fit is the estimand)."""
    out: dict[str, Any] = {}
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["seed"] == seed]
        try:
            fit = shadow_magnitude_fit(sub, predictors)
            out[str(seed)] = {
                "n_rows": fit["n_rows"],
                "beta_shadow_nats_per_radian": fit["beta_shadow_nats_per_radian"],
            }
        except (np.linalg.LinAlgError, KeyError) as e:
            out[str(seed)] = {"error": str(e)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_534"))
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=Path("eval_results/issue_530/phase0_5_gates.json"),
    )
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=504)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "eval_results/issue_534/shadow-flip-magnitude-nats/shadow_flip_magnitude.json"
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=i504_shadow_magnitude] %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        PREDICTORS,
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
    predictors = list(PREDICTORS)

    def _pool(frac: float, dg_band: Any = None) -> tuple[list[dict], dict]:
        kwargs: dict[str, Any] = dict(
            slab_root=args.slab_root,
            chosen_frac=frac,
            per_probe=gates["per_probe"],
            arm_to_positioned_n=gates["arm_to_positioned_n"],
            seeds=seeds,
            base_prior_by_probe=base_prior or None,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        )
        if dg_band is None:
            kwargs["dg_band"] = None
        pooled = build_rows(**kwargs)
        return pooled["rows"], pooled

    checkpoints: dict[str, Any] = {}
    for frac, role in [
        *[(f, "usable") for f in USABLE_FRACTIONS],
        *[(f, "subfloor_descriptive") for f in SUBFLOOR_FRACTIONS],
    ]:
        rows, _pooled = _pool(frac)
        steps = sorted({r["training_step"] for r in rows})
        log.info("frac %.2f (%s): %d rows, steps %s", frac, role, len(rows), steps)
        checkpoints[f"{frac:.2f}"] = {
            "role": role,
            "training_steps_in_pool": steps,
            "point": shadow_magnitude_fit(rows, predictors),
            "bootstrap": bootstrap_magnitude(
                rows, predictors, n_boot=args.n_boot, seed=args.boot_seed + int(frac * 100)
            ),
            "per_seed_beta": per_seed_betas(rows, predictors),
        }

    # Banded frac=1.00 sensitivity (canonical [5,12]-nat source-ΔG band — the
    # #530-gate-identical replication object's row pool).
    rows_banded, _ = _pool(1.00, dg_band="default")
    banded = {
        "role": "sensitivity_banded",
        "point": shadow_magnitude_fit(rows_banded, predictors),
        "bootstrap": bootstrap_magnitude(
            rows_banded, predictors, n_boot=args.n_boot, seed=args.boot_seed + 999
        ),
    }

    # Hypothesis evaluation (epm:followup-scope v1): CI on the adjusted
    # deep-minus-lateral RANGE-scaled gain excludes zero at BOTH usable steps.
    verdicts = {}
    for frac in USABLE_FRACTIONS:
        ck = checkpoints[f"{frac:.2f}"]
        ci = ck["bootstrap"]["adjusted_gain_deep_minus_lateral_range_ci"]
        excl = ci["lo"] is not None and (ci["lo"] > 0.0 or ci["hi"] < 0.0)
        verdicts[f"{frac:.2f}"] = {
            "ci_excludes_zero": bool(excl),
            "range_scaled_gain_nats": ck["point"]["adjusted_gain_deep_minus_lateral_range_nats"],
            "ci": ci,
        }
    n_excl = sum(v["ci_excludes_zero"] for v in verdicts.values())
    hypothesis = {
        "statement": (
            "adjusted deep-minus-lateral gain across the observed shadow-angle "
            "range is ~0.2-0.5 nat with CI excluding zero at both usable steps"
        ),
        "per_usable_checkpoint": verdicts,
        "n_usable_checkpoints_ci_excluding_zero": int(n_excl),
        "falsified_rank_only": bool(n_excl == 0),
    }

    payload = {
        "schema_version": "i504_shadow_flip_magnitude_v1",
        "task_id": 504,
        "followup_label": "shadow-flip-magnitude-nats",
        "data": {
            "slab_root": str(args.slab_root),
            "phase05_path": str(args.phase05_path),
            "seeds": seeds,
            "predictors": predictors,
            "dv": "delta_g (trained − base log P(marker) at the post-response slot, nats)",
            "sign_convention": (
                "positive adjusted gain = anti-shadow (deeper-in-shadow probes "
                "leak MORE); shadow_angle small = deep in shadow"
            ),
        },
        "checkpoints": checkpoints,
        "banded_frac1.00_sensitivity": banded,
        "hypothesis_evaluation": hypothesis,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    log.info("[phase=done] wrote %s", args.out)
    print(
        json.dumps(
            {
                "hypothesis_evaluation": hypothesis,
                "usable_points": {
                    k: {
                        "beta_shadow": v["point"]["beta_shadow_nats_per_radian"],
                        "iqr_gain": v["point"]["adjusted_gain_deep_minus_lateral_iqr_nats"],
                        "range_gain": v["point"]["adjusted_gain_deep_minus_lateral_range_nats"],
                        "tercile_adj_diff": v["point"][
                            "tercile_adjusted_mean_diff_deep_minus_lateral_nats"
                        ],
                        "dv_sd": v["point"]["dv_sd_nats"],
                    }
                    for k, v in checkpoints.items()
                    if v["role"] == "usable"
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
