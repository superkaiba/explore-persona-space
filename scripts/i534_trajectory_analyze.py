# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ρ/ΔG + × + − intentional
#!/usr/bin/env python3
"""Task #534 — per-fraction 6-predictor partial-Spearman refit + replication check.

Plan §4.3 (h): loops the trajectory fractions {0.25, 0.50, 0.75, 1.00}, drives
the shared #504 analysis core (`run_phase2_analysis`) per fraction with a
synthesized calibration (`chosen_checkpoint_fraction=f` — the same
synthesize-the-pick mechanism `i530_phase_analyze.py` introduced, re-minted
with task-534 provenance), and stitches
`eval_results/issue_534/analysis_per_fraction.json` with:

  * per-fraction fits (pooled + per-seed + sign agreement), persisted to a
    per-fraction side file the moment each completes (checkpoint-per-phase);
  * per-fraction USABILITY gates (plan §3): pooled bystander-resolution PASS
    (median bystander log P(marker) <= −2 nats AND argmax share < 60%,
    pooled over the positioned cells' held-out pairs) AND mean source ΔG
    over the 8 positioned cells >= the 1-nat floor;
  * the family-5 Holm sensitivity column (Holm across the 5 predictors
    excluding `training_step`), applied SYMMETRICALLY to every #534 fit AND
    the #530 reference (analyzer note #1);
  * the zero-variance `training_step` flag per fraction (analyzer note #1);
  * per-cell source-ΔG spread per fraction + the cell-excluded sensitivity
    refit when a cell sits below the floor at a mean-passing fraction
    (analyzer note #3);
  * the cross-fraction Holm-over-8 robustness column (2 headline predictors
    × 4 fractions);
  * bootstrap 95 percent CIs on the two headline partial ρs per fraction;
  * the `replication_check` block: the BANDED frac=1.00 fit (machinery- and
    gate-identical to #530's `analysis_v1.json`) vs the committed reference —
    sign match, Holm significance (family-6 AND family-5), |Δρ| vs the 0.15
    tolerance, CI overlap, and an independent-resample CI on ρ_534 − ρ_530
    (analyzer note #4: CI overlap leads, the cliff is a separate column);
  * the Δlog P vs Δz_marker agreement summary per fraction (logit column
    dropped + flagged when manifests fail the gauge check or z fields are
    absent).

Band semantics (documented deviation surface): the [5, 12] nat source-ΔG
band defines the frac=1.00 ANCHOR; it is NOT a per-fraction inclusion rule —
sub-final fractions are deliberately less-trained, so the per-fraction fits
run with the band exclusion DISABLED (`dg_band=None`). frac=1.00 is computed
BOTH ways; the BANDED fit is the replication object.

CPU-only. Run after the sweep's trajectories are on disk:
    uv run python scripts/i534_trajectory_analyze.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i534.trajectory_analyze")

HEADLINE_PREDICTORS: tuple[str, ...] = ("shadow_angle", "d_nearest_neg_nd")
DEFAULT_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
# Per-fraction usability gate constants (plan §3; bystander thresholds = #530's).
USABILITY_SOURCE_DG_FLOOR_NATS = 1.0
BYSTANDER_HEADROOM_NATS = 2.0
BYSTANDER_ARGMAX_CEILING = 0.60
REPLICATION_RHO_TOLERANCE = 0.15


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _synthesize_phase0_calibration(chosen_frac: float) -> dict[str, Any]:
    """In-memory phase0 calibration dict for one fraction (task-534 provenance).

    The same synthesize-the-pick mechanism as
    `i530_phase_analyze._synthesize_phase0_calibration`, re-minted here:
    #534 has no Phase 0 smoke ladder; `chosen_checkpoint_fraction` is a
    plan-pinned routing constant, never an evidence-based pick.
    """
    return {
        "verdict": "pass",
        "chosen_checkpoint_fraction": float(chosen_frac),
        "source": "i534_trajectory_analyze._synthesize_phase0_calibration",
        "task_id_minted_by": 534,
        "note": (
            "Synthesized per-fraction routing constant (plan #534 §4.3 h) — "
            "NOT an evidence-based calibration pick."
        ),
        "synthesized_at": datetime.now(UTC).isoformat(),
    }


def _pick_checkpoint(traj: dict, frac: float) -> dict | None:
    """Checkpoint whose `frac` is nearest to `frac` (same rule as the analysis core)."""
    cks = traj.get("checkpoints", [])
    if not cks:
        return None
    return min(cks, key=lambda c: abs(float(c["frac"]) - frac))


def usability_for_fraction(
    *,
    slab_root: Path,
    frac: float,
    seeds: list[int],
    positioned_arm_slugs: tuple[str, ...],
    floor_nats: float = USABILITY_SOURCE_DG_FLOOR_NATS,
    headroom_nats: float = BYSTANDER_HEADROOM_NATS,
    argmax_ceiling: float = BYSTANDER_ARGMAX_CEILING,
) -> dict[str, Any]:
    """Per-fraction usability gate (plan §3) over the positioned cells.

    Gate (i): pooled bystander resolution — median bystander ``g_logp`` over
    ALL positioned cells' held-out pairs <= −`headroom_nats` AND argmax-marker
    share < `argmax_ceiling`. Gate (ii): mean source ΔG over the positioned
    (cell × seed) set >= `floor_nats`. Also returns the per-cell source-ΔG
    spread (analyzer note #3) and the per-cell bystander gate detail.
    """
    per_cell_source_dg: dict[str, float | None] = {}
    per_cell_gate: dict[str, dict[str, Any]] = {}
    pooled_g_logp: list[float] = []
    pooled_argmax: list[bool] = []
    for cell in positioned_arm_slugs:
        for seed in seeds:
            key = f"{cell}_seed{seed}"
            p = slab_root / key / "trajectory.json"
            if not p.exists():
                per_cell_source_dg[key] = None
                continue
            traj = json.loads(p.read_text())
            ck = _pick_checkpoint(traj, frac)
            if ck is None:
                per_cell_source_dg[key] = None
                continue
            src = ck.get("source_self", {}) or {}
            dg = src.get("delta_g_mean")
            per_cell_source_dg[key] = float(dg) if dg is not None else None
            cell_g: list[float] = []
            cell_am: list[bool] = []
            for per_q in (ck.get("held_out", {}) or {}).values():
                for row in per_q.values():
                    if row is None:
                        continue
                    if row.get("g_logp") is not None:
                        cell_g.append(float(row["g_logp"]))
                        pooled_g_logp.append(float(row["g_logp"]))
                    if row.get("argmax_marker") is not None:
                        cell_am.append(bool(row["argmax_marker"]))
                        pooled_argmax.append(bool(row["argmax_marker"]))
            med = float(statistics.median(cell_g)) if cell_g else None
            share = (sum(cell_am) / len(cell_am)) if cell_am else None
            per_cell_gate[key] = {
                "median_bystander_g_logp": med,
                "argmax_marker_share": share,
                "checkpoint_step": ck.get("step"),
                "checkpoint_frac": ck.get("frac"),
            }
    dgs = [v for v in per_cell_source_dg.values() if v is not None]
    mean_dg = float(np.mean(dgs)) if dgs else None
    pooled_median = float(statistics.median(pooled_g_logp)) if pooled_g_logp else None
    pooled_share = (sum(pooled_argmax) / len(pooled_argmax)) if pooled_argmax else None
    bystander_pass = (
        pooled_median is not None
        and pooled_share is not None
        and pooled_median <= -headroom_nats
        and pooled_share < argmax_ceiling
    )
    source_floor_pass = mean_dg is not None and mean_dg >= floor_nats
    cells_below_floor = sorted(
        k for k, v in per_cell_source_dg.items() if v is not None and v < floor_nats
    )
    return {
        "usable": bool(bystander_pass and source_floor_pass),
        "bystander_gate": {
            "pooled_median_g_logp": pooled_median,
            "pooled_argmax_share": pooled_share,
            "headroom_nats_required": headroom_nats,
            "argmax_ceiling": argmax_ceiling,
            "pass": bool(bystander_pass),
            "pooled_over": "positioned cells' held-out pairs",
        },
        "source_dg_floor_gate": {
            "mean_source_delta_g_nats": mean_dg,
            "floor_nats": floor_nats,
            "pass": bool(source_floor_pass),
            "n_cells_with_value": len(dgs),
        },
        "per_cell_source_delta_g": per_cell_source_dg,
        "per_cell_bystander_gate": per_cell_gate,
        "cells_below_floor": cells_below_floor,
    }


def bootstrap_partial_rho(
    rows: list[dict],
    predictor: str,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Row-bootstrap percentile CI on the partial Spearman ρ of one predictor.

    Resamples rows with replacement; per resample, recomputes the partial
    Spearman of `predictor` vs ΔG with the other 5 predictors partialled out
    (the exact production estimator). NaN resamples (degenerate residual
    ranks) are dropped and counted.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        PREDICTORS,
        _partial_spearman,
    )

    if not rows:
        return {"lo": None, "hi": None, "n_boot": 0, "n_failed": 0}
    rng = np.random.default_rng(seed)
    n = len(rows)
    cols = {p: np.asarray([r[p] for r in rows], dtype=np.float64) for p in PREDICTORS}
    y = np.asarray([r["delta_g"] for r in rows], dtype=np.float64)
    vals: list[float] = []
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rho = _partial_spearman(
            y[idx].tolist(),
            cols[predictor][idx].tolist(),
            [cols[q][idx].tolist() for q in PREDICTORS if q != predictor],
        )
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            n_failed += 1
            continue
        vals.append(float(rho))
    if not vals:
        return {"lo": None, "hi": None, "n_boot": n_boot, "n_failed": n_failed}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "alpha": alpha,
        "n_boot": n_boot,
        "n_failed": n_failed,
    }


def bootstrap_rho_difference(
    rows_a: list[dict],
    rows_b: list[dict],
    predictor: str,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Independent-resample percentile CI on ρ_a − ρ_b (analyzer note #4).

    The two pools come from independent runs (#534 vs #530), so resampling
    each independently per iteration is the appropriate null-free interval.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        PREDICTORS,
        _partial_spearman,
    )

    if not rows_a or not rows_b:
        return {"lo": None, "hi": None, "n_boot": 0, "n_failed": 0}
    rng = np.random.default_rng(seed)

    def _prep(rows: list[dict]):
        cols = {p: np.asarray([r[p] for r in rows], dtype=np.float64) for p in PREDICTORS}
        y = np.asarray([r["delta_g"] for r in rows], dtype=np.float64)
        return cols, y

    cols_a, y_a = _prep(rows_a)
    cols_b, y_b = _prep(rows_b)
    n_a, n_b = len(rows_a), len(rows_b)
    vals: list[float] = []
    n_failed = 0
    for _ in range(n_boot):
        ia = rng.integers(0, n_a, n_a)
        ib = rng.integers(0, n_b, n_b)
        ra = _partial_spearman(
            y_a[ia].tolist(),
            cols_a[predictor][ia].tolist(),
            [cols_a[q][ia].tolist() for q in PREDICTORS if q != predictor],
        )
        rb = _partial_spearman(
            y_b[ib].tolist(),
            cols_b[predictor][ib].tolist(),
            [cols_b[q][ib].tolist() for q in PREDICTORS if q != predictor],
        )
        if any(isinstance(v, float) and math.isnan(v) for v in (ra, rb)):
            n_failed += 1
            continue
        vals.append(float(ra) - float(rb))
    if not vals:
        return {"lo": None, "hi": None, "n_boot": n_boot, "n_failed": n_failed}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "alpha": alpha,
        "n_boot": n_boot,
        "n_failed": n_failed,
    }


def family5_holm(partial_spearman: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Holm across the 5 predictors EXCLUDING training_step (analyzer note #1).

    A multiplicity-family sensitivity only — the ρ values themselves keep all
    6 predictors partialled out (never drop the covariate from the
    regression; only the Holm family changes).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        holm_correction,
    )

    pvals = {p: float(v["p_raw"]) for p, v in partial_spearman.items() if p != "training_step"}
    if not pvals:
        return {}
    return holm_correction(pvals, alpha=0.05)


def z_agreement_for_fraction(
    *,
    slab_root: Path,
    frac: float,
    seeds: list[int],
    positioned_arm_slugs: tuple[str, ...],
) -> dict[str, Any]:
    """Δlog P vs Δz_marker agreement summary per fraction (saturation localizer).

    Collects per-pair (delta_g, delta_z_marker, delta_z_margin) over the
    positioned cells at the fraction's checkpoint. Off saturation
    Δlog Z ≈ 0 so Δlog P ≈ Δz_marker; divergence is the saturation
    signature (a finding, not an error). Returns `available: false` when the
    z fields are absent (legacy trajectories / gauge-tripped cells — the
    logit column is then dropped + flagged, never silently zero-filled).
    """
    dg: list[float] = []
    dz: list[float] = []
    dzm: list[float] = []
    n_rows_seen = 0
    for cell in positioned_arm_slugs:
        for seed in seeds:
            p = slab_root / f"{cell}_seed{seed}" / "trajectory.json"
            if not p.exists():
                continue
            traj = json.loads(p.read_text())
            ck = _pick_checkpoint(traj, frac)
            if ck is None:
                continue
            for per_q in (ck.get("held_out", {}) or {}).values():
                for row in per_q.values():
                    if row is None:
                        continue
                    n_rows_seen += 1
                    if row.get("delta_z_marker") is not None and row.get("delta_g") is not None:
                        dg.append(float(row["delta_g"]))
                        dz.append(float(row["delta_z_marker"]))
                        if row.get("delta_z_margin") is not None:
                            dzm.append(float(row["delta_z_margin"]))
    if not dz:
        return {"available": False, "n_rows_seen": n_rows_seen, "n_pairs_with_z": 0}
    dg_a = np.asarray(dg)
    dz_a = np.asarray(dz)
    pear = float(np.corrcoef(dg_a, dz_a)[0, 1]) if dg_a.std() > 0 and dz_a.std() > 0 else None
    return {
        "available": True,
        "n_rows_seen": n_rows_seen,
        "n_pairs_with_z": len(dz),
        "mean_delta_logp": float(dg_a.mean()),
        "mean_delta_z_marker": float(dz_a.mean()),
        "mean_delta_z_margin": (float(np.mean(dzm)) if dzm else None),
        "mean_divergence_logp_minus_z": float((dg_a - dz_a).mean()),
        "pearson_delta_logp_vs_delta_z": pear,
    }


def collect_manifest_flags(slab_root: Path) -> dict[str, Any]:
    """Aggregate the fraction manifests' gauge / stop flags (analyzer note #9)."""
    flags: dict[str, Any] = {
        "per_cell": {},
        "all_logit_readout_valid": True,
        "cells_not_stopped": [],
    }
    for p in sorted(slab_root.glob("c504v3_*_seed*/fraction_manifest.json")):
        m = json.loads(p.read_text())
        key = p.parent.name
        flags["per_cell"][key] = {
            "logit_readout_valid": m.get("logit_readout_valid"),
            "stopped": m.get("stopped"),
            "stop_reason": m.get("stop_reason"),
            "distinct_steps": m.get("distinct_steps"),
            "exact_flags": {str(e["frac"]): e["exact"] for e in m.get("manifest", [])},
        }
        if not m.get("logit_readout_valid", False):
            flags["all_logit_readout_valid"] = False
        if not m.get("stopped", False):
            flags["cells_not_stopped"].append(key)
    return flags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_534"))
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=Path("eval_results/issue_530/phase0_5_gates.json"),
        help="#530's committed Phase 0.5 geometry artifact (reused as-is, plan §4.2).",
    )
    ap.add_argument(
        "--reference-analysis",
        type=Path,
        default=Path("eval_results/issue_530/analysis_v1.json"),
        help="#530's committed frac=1.00 fit — the replication reference.",
    )
    ap.add_argument(
        "--reference-slab",
        type=Path,
        default=Path("eval_results/issue_530"),
        help="#530's committed trajectories (for the reference bootstrap rebuild).",
    )
    ap.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=534)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default <slab-root>/analysis_per_fraction.json).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze_534] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        holm_correction,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        aggregate_base_prior_from_trajectories,
        build_rows,
        fit_pooled_partial_spearman,
        run_phase2_analysis,
        write_base_prior_marker,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        load_phase05,
    )

    slab: Path = args.slab_root
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fractions = tuple(sorted(float(x) for x in args.fractions.split(",") if x.strip()))
    out_path = args.out if args.out is not None else slab / "analysis_per_fraction.json"

    gates = load_phase05(args.phase05_path)
    per_probe = gates["per_probe"]
    arm_to_positioned_n = gates["arm_to_positioned_n"]

    # Base-prior covariate aggregated from the #534 trajectories (constant
    # across checkpoints by construction — the base model is frozen).
    base_prior = aggregate_base_prior_from_trajectories(
        slab_root=slab, seeds=seeds, positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3
    )
    if base_prior:
        write_base_prior_marker(base_prior, slab / "base_prior_marker.json")
    else:
        log.warning("[base_prior] empty aggregation — covariate falls back to 0.0 placeholder.")
        base_prior = None

    manifest_flags = collect_manifest_flags(slab)

    notes: list[str] = []
    per_fraction_out: dict[str, Any] = {}
    rows_by_frac: dict[str, list[dict]] = {}
    headline_praw: dict[str, float] = {}
    for f in fractions:
        frac_str = f"{f:.2f}"
        log.info("[phase=analyze_534_frac_%s] per-fraction fit (dg_band=None)", frac_str)
        summary = run_phase2_analysis(
            slab_root=slab,
            phase0_calibration=_synthesize_phase0_calibration(f),
            phase05_gates=gates,
            seeds=seeds,
            base_prior_by_probe=base_prior,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            dg_band=None,
        )
        # Persist the FULL per-fraction summary the moment it completes
        # (checkpoint-per-phase rule).
        side_path = slab / f"analysis_frac{frac_str}.json"
        side_path.write_text(json.dumps(summary, indent=2, default=str))
        log.info("[phase=analyze_534_frac_%s] wrote %s", frac_str, side_path)

        # Rebuild the rows in-process for the bootstrap (same inputs).
        pooled = build_rows(
            slab_root=slab,
            chosen_frac=f,
            per_probe=per_probe,
            arm_to_positioned_n=arm_to_positioned_n,
            seeds=seeds,
            base_prior_by_probe=base_prior,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            dg_band=None,
        )
        rows = pooled["rows"]
        rows_by_frac[frac_str] = rows

        usability = usability_for_fraction(
            slab_root=slab,
            frac=f,
            seeds=seeds,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        )

        # Zero-variance training_step flag (analyzer note #1).
        steps = sorted({r["training_step"] for r in rows})
        zero_var_step = len(steps) <= 1
        if zero_var_step and rows:
            notes.append(
                f"frac {frac_str}: training_step has zero variance (all rows at "
                f"step {steps[0] if steps else 'n/a'}) — its coefficient is a "
                "numerical artifact, do not interpret."
            )

        fit = summary["pooled_fit"]
        boot = {
            p: bootstrap_partial_rho(
                rows, p, n_boot=args.n_boot, seed=args.boot_seed + int(f * 100)
            )
            for p in HEADLINE_PREDICTORS
        }
        for p in HEADLINE_PREDICTORS:
            part = fit.get("partial_spearman", {}).get(p)
            if part is not None:
                headline_praw[f"{p}@frac{frac_str}"] = float(part["p_raw"])

        # Cell-excluded sensitivity refit (analyzer note #3): only when the
        # mean-floor gate PASSes but some cell sits below the floor.
        cell_excluded_fit = None
        below = usability["cells_below_floor"]
        if usability["source_dg_floor_gate"]["pass"] and below:
            kept = [r for r in rows if f"{r['cell']}_seed{r['seed']}" not in set(below)]
            cell_excluded_fit = {
                "excluded_cells": below,
                "n_rows": len(kept),
                "fit": fit_pooled_partial_spearman(kept),
            }
            notes.append(
                f"frac {frac_str}: cells {below} below the {USABILITY_SOURCE_DG_FLOOR_NATS}-nat "
                "floor while the mean gate passes — cell-excluded sensitivity refit reported."
            )

        per_fraction_out[frac_str] = {
            "analysis_path": str(side_path),
            "n_rows": len(rows),
            "pooled_fit": fit,
            "per_seed_fit": summary["per_seed_fit"],
            "sign_agreement": summary["sign_agreement"],
            "per_cell_diagnostics": summary["per_cell_diagnostics"],
            "usability": usability,
            "zero_variance_training_step": bool(zero_var_step),
            "distinct_training_steps_in_pool": steps,
            "family5_holm": family5_holm(fit.get("partial_spearman", {})),
            "bootstrap_ci_headline": boot,
            "cell_excluded_sensitivity": cell_excluded_fit,
            "z_agreement": z_agreement_for_fraction(
                slab_root=slab,
                frac=f,
                seeds=seeds,
                positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
            ),
        }

    # ── Cross-fraction Holm-over-8 robustness column (plan §6 statistics). ───
    holm_over_8 = holm_correction(headline_praw, alpha=0.05) if headline_praw else {}

    # ── BANDED frac=1.00 fit — the replication object (machinery- and
    # gate-identical to #530's analysis_v1.json). ─────────────────────────────
    log.info("[phase=analyze_534_replication] banded frac=1.00 fit")
    summary_banded = run_phase2_analysis(
        slab_root=slab,
        phase0_calibration=_synthesize_phase0_calibration(1.0),
        phase05_gates=gates,
        seeds=seeds,
        base_prior_by_probe=base_prior,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        # default dg_band — the canonical [5, 12] exclusion, as in #530.
    )
    banded_path = slab / "analysis_frac1.00_banded.json"
    banded_path.write_text(json.dumps(summary_banded, indent=2, default=str))
    rows_banded = build_rows(
        slab_root=slab,
        chosen_frac=1.0,
        per_probe=per_probe,
        arm_to_positioned_n=arm_to_positioned_n,
        seeds=seeds,
        base_prior_by_probe=base_prior,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
    )["rows"]

    # ── Reference (#530) fit + rebuilt rows for CIs. ──────────────────────────
    replication: dict[str, Any] = {"reference_path": str(args.reference_analysis)}
    if args.reference_analysis.exists():
        ref = json.loads(args.reference_analysis.read_text())
        ref_fit = ref["pooled_fit"]
        ref_base_prior = aggregate_base_prior_from_trajectories(
            slab_root=args.reference_slab,
            seeds=seeds,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        )
        ref_rows = build_rows(
            slab_root=args.reference_slab,
            chosen_frac=1.0,
            per_probe=per_probe,
            arm_to_positioned_n=arm_to_positioned_n,
            seeds=seeds,
            base_prior_by_probe=ref_base_prior or None,
            positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
        )["rows"]
        # Sanity: the rebuilt reference fit must agree with the committed one.
        ref_refit = fit_pooled_partial_spearman(ref_rows)
        rebuild_drift = {
            p: abs(
                float(ref_refit["partial_spearman"][p]["rho"])
                - float(ref_fit["partial_spearman"][p]["rho"])
            )
            for p in HEADLINE_PREDICTORS
            if p in ref_refit.get("partial_spearman", {})
        }
        if any(d > 1e-6 for d in rebuild_drift.values()):
            notes.append(
                f"reference rebuild drift vs committed analysis_v1.json: {rebuild_drift} — "
                "the bootstrap reference rows may not match the committed fit's inputs; "
                "read the reference CI with that caveat."
            )
        fam5_ref = family5_holm(ref_fit.get("partial_spearman", {}))
        fam5_new = family5_holm(summary_banded["pooled_fit"].get("partial_spearman", {}))
        per_pred: dict[str, Any] = {}
        for p in HEADLINE_PREDICTORS:
            new = summary_banded["pooled_fit"]["partial_spearman"].get(p)
            old = ref_fit["partial_spearman"].get(p)
            if new is None or old is None:
                per_pred[p] = {"available": False}
                continue
            rho_new, rho_old = float(new["rho"]), float(old["rho"])
            ci_new = bootstrap_partial_rho(
                rows_banded, p, n_boot=args.n_boot, seed=args.boot_seed + 1000
            )
            ci_old = bootstrap_partial_rho(
                ref_rows, p, n_boot=args.n_boot, seed=args.boot_seed + 2000
            )
            ci_overlap = None
            if ci_new["lo"] is not None and ci_old["lo"] is not None:
                ci_overlap = not (ci_new["hi"] < ci_old["lo"] or ci_old["hi"] < ci_new["lo"])
            per_pred[p] = {
                "rho_534": rho_new,
                "rho_530": rho_old,
                "sign_match": bool(np.sign(rho_new) == np.sign(rho_old) and rho_new != 0),
                "p_raw_534": float(new["p_raw"]),
                "p_raw_530": float(old["p_raw"]),
                "holm6_reject_534": bool(
                    summary_banded["pooled_fit"]["holm"].get(p, {}).get("reject_null", False)
                ),
                "holm6_reject_530": bool(ref_fit["holm"].get(p, {}).get("reject_null", False)),
                "holm5_reject_534": bool(fam5_new.get(p, {}).get("reject_null", False)),
                "holm5_reject_530": bool(fam5_ref.get(p, {}).get("reject_null", False)),
                "abs_delta_rho": abs(rho_new - rho_old),
                "within_tolerance": abs(rho_new - rho_old) <= REPLICATION_RHO_TOLERANCE,
                "tolerance": REPLICATION_RHO_TOLERANCE,
                "ci_534": ci_new,
                "ci_530": ci_old,
                "ci_overlap": ci_overlap,
                "rho_diff_ci": bootstrap_rho_difference(
                    rows_banded, ref_rows, p, n_boot=args.n_boot, seed=args.boot_seed + 3000
                ),
            }
        replication.update(
            {
                "available": True,
                "banded_fit_path": str(banded_path),
                "n_rows_534_banded": len(rows_banded),
                "n_rows_530_rebuilt": len(ref_rows),
                "rebuild_drift_vs_committed": rebuild_drift,
                "per_predictor": per_pred,
                "family5_holm_534": fam5_new,
                "family5_holm_530": fam5_ref,
                "note": (
                    "CI overlap + sign lead the read; the 0.15 |Δρ| cliff is a "
                    "separate column (a STRONGER same-sign reversal can trip the "
                    "two-sided tolerance — read direction first). H-D verdict "
                    "assignment is the analyzer's, weighing these jointly."
                ),
            }
        )
    else:
        replication["available"] = False
        notes.append(
            f"reference analysis missing at {args.reference_analysis} — replication "
            "check NOT computed."
        )

    payload = {
        "schema_version": "i534_analysis_per_fraction_v1",
        "task_id": 534,
        "parent_task_id": 530,
        "fractions": [f"{f:.2f}" for f in fractions],
        "seeds": seeds,
        "headline_predictors": list(HEADLINE_PREDICTORS),
        "band_semantics": (
            "per-fraction fits run with dg_band=None (the [5,12] band defines "
            "the frac=1.00 anchor, not a per-fraction inclusion rule); the "
            "replication_check uses the BANDED frac=1.00 fit, gate-identical "
            "to #530."
        ),
        "per_fraction": per_fraction_out,
        "holm_over_8_headline": holm_over_8,
        "replication_check": replication,
        "manifest_flags": manifest_flags,
        "usability_constants": {
            "source_dg_floor_nats": USABILITY_SOURCE_DG_FLOOR_NATS,
            "bystander_headroom_nats": BYSTANDER_HEADROOM_NATS,
            "bystander_argmax_ceiling": BYSTANDER_ARGMAX_CEILING,
        },
        "notes": notes,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info(
        "[phase=done] wrote %s (%d fractions, replication available=%s)",
        out_path,
        len(per_fraction_out),
        replication.get("available"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
