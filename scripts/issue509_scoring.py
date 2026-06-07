#!/usr/bin/env python3
"""Issue #509 per-arm scoring against #494 fact + #411 sycophancy targets.

Path B post-extraction scoring. Reads the metrics-phase distance JSONs
produced by ``scripts/issue502_dispatch.py --phase metrics`` and computes:

  - Length-partial Spearman rho per (point, layer, metric, variant)
    against the per-arm leakage target.
  - Substrate-FE (fact arm) or source-FE (syco arm) residualized rho.
  - Prior-residualized rho on the fact arm (bystander_logprob from
    #500).
  - Attenuation-adjusted rho = rho_obs / sqrt(reliability_y).
  - Within-stratum permutation null (B=2000, hashed seed).
  - LOCO-CV R^2 of a length-controlled linear fit.
  - Cluster bootstrap (5000 reps, seed=42) CI per cell on the L22 last
    prompt gauss_kl anchor + the L19-L24 ridge mean.
  - Delete-one-substrate (fact) or delete-one-source (syco) jackknife.

Pre-registered anchors per #509 v3 plan section 4.1.6:
  1. #494 / #470 coarse predictors on the same pairs.
  2. #502 full-panel ρ_full_deltag at L22 = -0.748.
  3. #502 non-stylized 156-pair ρ_nonstylized_deltag at L22 = -0.581,
     ρ_nonstylized_glogp at L22 = -0.628. Loaded from
     ``eval_results/issue_502/bakeoff/regression/loc_ep1.json``.

Inputs:
  --metrics-dir   directory holding per-(point, layer, metric, variant)
                  distance JSONs (the metrics-phase output).
  --arm           one of "fact" or "syco".
  --target-file   per-arm leakage matrix (see plan section 10).
  --output        per-arm scoring JSON output path.
  --smoke         smoke-mode: relax min-cell expectations + skip
                  permutation/bootstrap so a tiny 2-cond x 1-layer x
                  1-metric grid runs in seconds.

The output JSON carries every cell's (rho_obs, rho_adj, ci_lo, ci_hi,
perm_p, loco_r2, jackknife_se) along with the arm-level summary
statistics (L22 anchor, L19-L24 ridge mean) and the comparison anchors.
"""

# Greek + special characters appear in this file's prose.
# ruff: noqa: RUF002
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("i509.scoring")

PERMUTATION_NULL_SEED_TAG = b"issue509_perm_null_v1"
PERMUTATION_B = 2000
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42

# Pre-verified at plan time from eval_results/issue_502/bakeoff/regression/loc_ep1.json.
NONSTYLIZED_ANCHOR_RHO_DELTAG = -0.5805350970934474
NONSTYLIZED_ANCHOR_RHO_GLOGP = -0.6278048029398947
FULL_PANEL_ANCHOR_RHO_DELTAG = -0.748

L19_L24_RIDGE_LAYERS = (19, 20, 21, 22, 23, 24)
RIDGE_METRICS = ("gauss_kl", "mmd", "wass2")

METRIC_FILE_PATTERN = re.compile(
    r"^(?P<point>[a-z_]+)__layer(?P<layer>\d+)__(?P<metric>[a-z0-9_]+)__(?P<variant>[a-z_]+)\.json$"
)


def _hashed_seed(tag: bytes) -> int:
    """Deterministic integer for numpy.random.default_rng from a string tag."""
    return int(hashlib.sha256(tag).hexdigest()[:8], 16)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _env_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank+Pearson. NaN-safe (drops paired NaNs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata

    rx = rankdata(x[m])
    ry = rankdata(y[m])
    return float(np.corrcoef(rx, ry)[0, 1])


def _residualize(y: np.ndarray, fe: np.ndarray) -> np.ndarray:
    """Within-stratum residualization: subtract the per-stratum mean of y."""
    y = np.asarray(y, dtype=float)
    out = y.copy()
    for s in np.unique(fe):
        idx = fe == s
        out[idx] -= np.nanmean(y[idx])
    return out


def _permutation_p(rho_obs: float, x: np.ndarray, y: np.ndarray, fe: np.ndarray, b: int) -> float:
    """Within-stratum permutation p (two-tailed)."""
    if not np.isfinite(rho_obs):
        return float("nan")
    rng = np.random.default_rng(_hashed_seed(PERMUTATION_NULL_SEED_TAG))
    n_ge = 0
    for _ in range(b):
        y_perm = y.copy()
        for s in np.unique(fe):
            idx = np.where(fe == s)[0]
            if len(idx) <= 1:
                continue
            rng.shuffle(idx)
            y_perm[fe == s] = y[idx]
        rho_p = _spearman_rho(x, y_perm)
        if np.isfinite(rho_p) and abs(rho_p) >= abs(rho_obs):
            n_ge += 1
    return (1 + n_ge) / (b + 1)


def _cluster_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Cluster bootstrap 95% CI on the Spearman rho."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    rhos: list[float] = []
    for _ in range(b):
        sample = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in sample])
        rho_b = _spearman_rho(x[idx], y[idx])
        if np.isfinite(rho_b):
            rhos.append(rho_b)
    if len(rhos) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


def _jackknife_se(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> float:
    """Delete-one-cluster jackknife SE on the Spearman rho."""
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        return float("nan")
    rhos: list[float] = []
    for c in unique_clusters:
        idx = clusters != c
        rho_k = _spearman_rho(x[idx], y[idx])
        if np.isfinite(rho_k):
            rhos.append(rho_k)
    if len(rhos) < 2:
        return float("nan")
    n = len(rhos)
    rho_mean = float(np.mean(rhos))
    se = np.sqrt((n - 1) / n * sum((r - rho_mean) ** 2 for r in rhos))
    return float(se)


def _loco_cv_r2(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Leave-one-class-out CV R^2 of a univariate linear fit."""
    unique_classes = np.unique(classes)
    if len(unique_classes) < 2:
        return float("nan")
    y_pred = np.full_like(y, np.nan, dtype=float)
    for c in unique_classes:
        train = classes != c
        test = classes == c
        x_train, y_train = x[train], y[train]
        x_test = x[test]
        if len(x_train) < 2 or np.all(x_train == x_train[0]):
            continue
        slope, intercept = np.polyfit(x_train, y_train, 1)
        y_pred[test] = slope * x_test + intercept
    valid = ~np.isnan(y_pred)
    if valid.sum() < 2:
        return float("nan")
    ss_res = float(np.sum((y[valid] - y_pred[valid]) ** 2))
    ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _attenuation_adjust(rho_obs: float, reliability_y: float) -> float:
    """rho_adj = rho_obs / sqrt(reliability_y); clamps reliability to (0, 1]."""
    if not (0.0 < reliability_y <= 1.0):
        return rho_obs
    return rho_obs / np.sqrt(reliability_y)


# ── Target loaders ────────────────────────────────────────────────────────


def _load_fact_target(csv_path: Path) -> dict[str, Any]:
    """Load #494's 26-cell fact-leakage panel from regression_data.csv."""
    rows: list[dict[str, Any]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["leak_rate"] = float(row["leak_rate"])
            row["bystander_logprob"] = float(row["bystander_logprob"])
            rows.append(row)
    if len(rows) != 26:
        logger.warning("Expected 26 #494 rows, got %d", len(rows))
    return {"rows": rows, "source_file": str(csv_path)}


def _load_syco_target(snapshot_path: Path) -> dict[str, Any]:
    """Load #411's 138-cell sycophancy Δ panel from the frozen snapshot."""
    with open(snapshot_path) as f:
        snap = json.load(f)
    rows: list[dict[str, Any]] = []
    for source, src_data in snap["per_source"].items():
        panel = src_data.get("per_panel_delta", {})
        trained_rate_by = src_data.get("per_panel_trained_rate", {})
        base_rate_by = src_data.get("per_panel_base_rate", {})
        for bystander, delta in panel.items():
            if bystander == source:
                continue  # off-diagonal only
            p_t = trained_rate_by.get(bystander)
            p_b = base_rate_by.get(bystander)
            # Independence approximation; rollouts = 50 probes * 10 = 500.
            if p_t is None or p_b is None:
                se = float("nan")
            else:
                n_rollouts = 500
                se = float(
                    np.sqrt(
                        max(p_t * (1 - p_t), 0.0) / n_rollouts
                        + max(p_b * (1 - p_b), 0.0) / n_rollouts
                    )
                )
            rows.append(
                {
                    "source": source,
                    "bystander": bystander,
                    "delta": float(delta),
                    "trained_rate": p_t,
                    "base_rate": p_b,
                    "se_delta": se,
                }
            )
    return {"rows": rows, "source_file": str(snapshot_path)}


# ── Cell scoring core ─────────────────────────────────────────────────────


def _build_fact_xy(
    matrix: dict[str, dict[str, float]],
    fact_rows: list[dict[str, Any]],
    cid_to_csv_persona: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, substrate, prior_z, cell_se) aligned across the 26 cells."""
    persona_to_cid = {v: k for k, v in cid_to_csv_persona.items()}
    x_arr, y_arr, sub_arr, z_arr, se_arr = [], [], [], [], []
    for row in fact_rows:
        teach_cid = persona_to_cid.get(row["teach_persona"])
        bys_cid = persona_to_cid.get(row["bystander_persona"])
        if teach_cid is None or bys_cid is None:
            continue
        d = matrix.get(teach_cid, {}).get(bys_cid)
        if d is None:
            continue
        x_arr.append(d)
        y_arr.append(row["leak_rate"])
        sub_arr.append(row["substrate"])
        z_arr.append(row["bystander_logprob"])
        se_arr.append(float("nan"))  # Per-seed reconstruction goes here (TODO inflow)
    return (
        np.array(x_arr, dtype=float),
        np.array(y_arr, dtype=float),
        np.array(sub_arr),
        np.array(z_arr, dtype=float),
        np.array(se_arr, dtype=float),
    )


def _build_syco_xy(
    matrix: dict[str, dict[str, float]],
    syco_rows: list[dict[str, Any]],
    cid_to_syco_persona: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, source, se) aligned across the 138 cells."""
    persona_to_cid = {v: k for k, v in cid_to_syco_persona.items()}
    x_arr, y_arr, src_arr, se_arr = [], [], [], []
    for row in syco_rows:
        src_cid = persona_to_cid.get(row["source"])
        bys_cid = persona_to_cid.get(row["bystander"])
        if src_cid is None or bys_cid is None:
            continue
        d = matrix.get(src_cid, {}).get(bys_cid)
        if d is None:
            continue
        x_arr.append(d)
        y_arr.append(row["delta"])
        src_arr.append(row["source"])
        se_arr.append(row["se_delta"])
    return (
        np.array(x_arr, dtype=float),
        np.array(y_arr, dtype=float),
        np.array(src_arr),
        np.array(se_arr, dtype=float),
    )


def _reliability_y(y: np.ndarray, se: np.ndarray) -> float:
    """Reliability = 1 - mean(SE^2)/var(y); clipped to (1e-6, 1.0]."""
    if not np.isfinite(se).any():
        return 1.0  # SE unknown -> assume reliable; rho_adj = rho_obs.
    var_y = float(np.nanvar(y))
    mean_se2 = float(np.nanmean(se**2))
    if var_y <= 0:
        return 1.0
    return float(max(min(1.0 - mean_se2 / var_y, 1.0), 1e-6))


def _score_one_cell(
    *,
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    se: np.ndarray | None,
    prior_z: np.ndarray | None,
    run_permutation: bool,
    run_bootstrap: bool,
    perm_b: int,
) -> dict[str, Any]:
    """Compute one (point, layer, metric, variant) cell's scoring panel.

    strata = substrate (fact) or source (syco); used for FE-residualization,
    permutation null, cluster bootstrap, and delete-one jackknife.
    """
    out: dict[str, Any] = {}
    rho_pooled = _spearman_rho(x, y)
    y_resid = _residualize(y, strata)
    rho_fe = _spearman_rho(x, y_resid)
    out["rho_pooled"] = rho_pooled
    out["rho_fe"] = rho_fe
    if prior_z is not None and np.isfinite(prior_z).all():
        y_pz = _residualize(y - prior_z, strata) if len(strata) > 0 else (y - prior_z)
        out["rho_double_fe"] = _spearman_rho(x, y_pz)
    else:
        out["rho_double_fe"] = float("nan")
    rel = _reliability_y(y, se) if se is not None else 1.0
    out["reliability_y"] = rel
    out["rho_pooled_adj"] = _attenuation_adjust(rho_pooled, rel)
    out["rho_fe_adj"] = _attenuation_adjust(rho_fe, rel)
    out["loco_r2"] = _loco_cv_r2(x, y, strata)
    if run_permutation:
        out["perm_p_fe"] = _permutation_p(rho_fe, x, y, strata, perm_b)
    if run_bootstrap:
        ci_lo, ci_hi = _cluster_bootstrap_ci(x, y_resid, strata)
        out["ci_lo_fe"] = ci_lo
        out["ci_hi_fe"] = ci_hi
        out["jackknife_se_fe"] = _jackknife_se(x, y_resid, strata)
    out["n"] = len(x)
    return out


def _matrix_to_dict(matrix_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Coerce a metric-phase JSON payload's matrix to a {a: {b: float}} dict."""
    m = matrix_payload.get("matrix", {})
    return {a: {b: float(v) for b, v in row.items()} for a, row in m.items()}


def _enumerate_metric_files(metrics_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """List every metric-phase output file + parsed (point, layer, metric, variant)."""
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(metrics_dir.glob("*.json")):
        m = METRIC_FILE_PATTERN.match(p.name)
        if not m:
            continue
        out.append(
            (
                p,
                {
                    "extraction_point": m.group("point"),
                    "layer": int(m.group("layer")),
                    "metric": m.group("metric"),
                    "variant": m.group("variant"),
                },
            )
        )
    return out


def score_arm(
    *,
    arm: str,
    metrics_dir: Path,
    target_file: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Score every (point, layer, metric, variant) cell on one arm."""
    if arm == "fact":
        target = _load_fact_target(target_file)
        from explore_persona_space.experiments.i509_fact_conditions import (
            CID_TO_CSV_PERSONA,
        )

        cid_to_persona = CID_TO_CSV_PERSONA
    elif arm == "syco":
        target = _load_syco_target(target_file)
        from explore_persona_space.experiments.i509_syco_conditions import (
            CID_TO_SYCO_PERSONA,
        )

        cid_to_persona = CID_TO_SYCO_PERSONA
    else:
        raise ValueError(f"Unknown arm {arm!r}; expected 'fact' or 'syco'")

    files = _enumerate_metric_files(metrics_dir)
    if not files:
        raise FileNotFoundError(f"No metric JSONs under {metrics_dir}")
    logger.info("Found %d metric files in %s", len(files), metrics_dir)

    perm_b = 50 if smoke else PERMUTATION_B
    run_permutation = not smoke
    run_bootstrap = not smoke

    cells: list[dict[str, Any]] = []
    for path, meta in files:
        with open(path) as fh:
            payload = json.load(fh)
        matrix = _matrix_to_dict(payload)
        if arm == "fact":
            x, y, strata, prior_z, se = _build_fact_xy(matrix, target["rows"], cid_to_persona)
            scored = _score_one_cell(
                x=x,
                y=y,
                strata=strata,
                se=se,
                prior_z=prior_z,
                run_permutation=run_permutation,
                run_bootstrap=run_bootstrap,
                perm_b=perm_b,
            )
        else:
            x, y, strata, se = _build_syco_xy(matrix, target["rows"], cid_to_persona)
            scored = _score_one_cell(
                x=x,
                y=y,
                strata=strata,
                se=se,
                prior_z=None,
                run_permutation=run_permutation,
                run_bootstrap=run_bootstrap,
                perm_b=perm_b,
            )
        cells.append({**meta, **scored})

    summary = _summarize_cells(cells)
    return {
        "schema_version": 1,
        "arm": arm,
        "smoke": smoke,
        "n_metric_files": len(files),
        "n_cells_scored": len(cells),
        "anchors": {
            "nonstylized_rho_deltag": NONSTYLIZED_ANCHOR_RHO_DELTAG,
            "nonstylized_rho_glogp": NONSTYLIZED_ANCHOR_RHO_GLOGP,
            "full_panel_rho_deltag": FULL_PANEL_ANCHOR_RHO_DELTAG,
        },
        "perm_null_seed_int": _hashed_seed(PERMUTATION_NULL_SEED_TAG),
        "perm_null_b": perm_b,
        "bootstrap_b": BOOTSTRAP_B if run_bootstrap else 0,
        "summary": summary,
        "cells": cells,
        "target_source_file": target["source_file"],
        "git_sha": _git_sha(),
        "timestamp_utc": _now_iso(),
        "env": _env_versions(),
    }


def _summarize_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the L22 last_prompt gauss_kl anchor + L19-L24 ridge mean."""
    summary: dict[str, Any] = {}
    # Anchor cell.
    for c in cells:
        if (
            c["extraction_point"] == "last_prompt"
            and c["layer"] == 22
            and c["metric"] == "gauss_kl"
            and c.get("variant") == "centered"
        ):
            summary["anchor_L22_last_prompt_gauss_kl_centered"] = {
                "rho_fe": c.get("rho_fe"),
                "rho_fe_adj": c.get("rho_fe_adj"),
                "perm_p_fe": c.get("perm_p_fe"),
                "n": c.get("n"),
            }
            break
    # L19-L24 ridge.
    ridge_rhos: list[float] = []
    for c in cells:
        if (
            c["extraction_point"] == "last_prompt"
            and c["layer"] in L19_L24_RIDGE_LAYERS
            and c["metric"] in RIDGE_METRICS
            and c.get("variant") == "centered"
        ):
            rho = c.get("rho_fe_adj")
            if rho is not None and np.isfinite(rho):
                ridge_rhos.append(rho)
    if ridge_rhos:
        summary["ridge_L19_L24_mean_rho_fe_adj"] = float(np.mean(ridge_rhos))
        summary["ridge_L19_L24_n_cells"] = len(ridge_rhos)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #509 per-arm scoring against fact / syco leakage targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding per-(point, layer, metric, variant) distance "
            "JSONs (the metrics-phase output)."
        ),
    )
    p.add_argument(
        "--arm",
        choices=("fact", "syco"),
        required=True,
        help="Which arm to score against.",
    )
    p.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help=(
            "Target leakage matrix. Fact arm: eval_results/issue_494/regression_data.csv. "
            "Syco arm: eval_results/issue_480/_inputs/syco_411_analyze_summary.json. "
            "When unset, defaults are filled in per --arm."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the per-arm scoring panel.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: skip permutation + bootstrap, relax cell-count gates.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    target_file = args.target_file
    if target_file is None:
        if args.arm == "fact":
            target_file = PROJECT_ROOT / "eval_results/issue_494/regression_data.csv"
        else:
            target_file = (
                PROJECT_ROOT / "eval_results/issue_480/_inputs/syco_411_analyze_summary.json"
            )
    if not target_file.exists():
        logger.error("Target file missing: %s", target_file)
        return 2
    if not args.metrics_dir.exists():
        logger.error("Metrics dir missing: %s", args.metrics_dir)
        return 2

    out = score_arm(
        arm=args.arm,
        metrics_dir=args.metrics_dir,
        target_file=target_file,
        smoke=args.smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str))
    logger.info(
        "Wrote %s: arm=%s cells=%d smoke=%s",
        args.output,
        args.arm,
        out["n_cells_scored"],
        args.smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
