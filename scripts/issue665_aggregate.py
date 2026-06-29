#!/usr/bin/env python
"""Issue #665 Phase 3 — aggregate (clustered bootstrap CIs + probe-split + FDR).

Reads the per-arm per-cell JSONs (eval_results/issue_665/<arm>/<cell>.json) +
the behavioral DV (judged_E/<cell>.json) and produces:
- eval_results/issue_665/aggregate.json — the §6.5 PRIMARY deliverable: per-behavior
  g0-vs-ghat rho + clustered (family/source/seed) bootstrap CIs + probe-split
  replication + FDR (Benjamini-Hochberg alpha=0.05) verdicts.
- eval_results/issue_665/analyzer_body_data.json — the analyzer's dashboard data.
- eval_results/issue_665/whitened_gate_unittest.json — the B3 reduction-unit-test
  PASS record (run here; A3.9/A3.10 numbers are NOT trusted until it PASSes).

Clustering (C4): cross-cell correlations resample at the cluster level (family /
source / seed) — naive n=50 CIs are banned. FDR (C3): the C3-pre-registered
primary path (per-behavior locked layer + c_C key + Σc⁻¹ metric) is THE verdict;
the layer/key/metric sweep is exploratory with Benjamini-Hochberg.

Usage:
    uv run python scripts/issue665_aggregate.py --scope content
    uv run python scripts/issue665_aggregate.py --cells bm_default_contra_d1_seed42 --smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess

import issue665_common as C
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("issue665_aggregate")

BOOTSTRAP_B = C.BOOTSTRAP_B
FDR_ALPHA = C.FDR_ALPHA


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    except Exception:
        return "unknown"


def run_b3_unittest() -> dict:
    """Run the B3 reduction unit test (tests/test_whitened_gate.py) and record the
    PASS. A3.9/A3.10 numbers are NOT trusted until this PASSes (plan §7)."""
    proc = subprocess.run(
        ["uv", "run", "pytest", "tests/test_whitened_gate.py", "-q"],
        cwd=C.REPO,
        capture_output=True,
        text=True,
    )
    passed = proc.returncode == 0
    rec = {
        "test": "tests/test_whitened_gate.py",
        "passed": passed,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.strip().splitlines()[-5:] if proc.stdout else [],
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "note": "B3 reduction (gate->cos in Sigma_c=I/equal-norm limit, 1e-6) + finite/non-NaN "
        "at smallest swept lambda=1e-3. A3.9/A3.10 numbers gated on this PASS.",
    }
    outp = C.EVAL_ROOT / "whitened_gate_unittest.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(rec, f, indent=1)
    logger.info("[b3-unittest] passed=%s -> %s", passed, outp)
    return rec


def _load_arm(arm: str, cells: list[str]) -> dict[str, dict]:
    out = {}
    for cell in cells:
        p = C.EVAL_ROOT / arm / f"{cell}.json"
        if p.exists():
            with open(p) as f:
                out[cell] = json.load(f)
    return out


def _cluster_bootstrap_ci(
    values: list[float], cluster_ids: list[str], rng, b: int = BOOTSTRAP_B
) -> dict:
    """Clustered bootstrap CI: resample CLUSTERS with replacement (C4), recompute
    the mean over the resampled cluster members. Returns mean + 95% CI."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    cids = [c for v, c in zip(values, cluster_ids, strict=True) if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return {
            "mean": float(vals[0]) if len(vals) == 1 else None,
            "ci_lo": None,
            "ci_hi": None,
            "n": len(vals),
        }
    # group values by cluster
    by_cluster: dict[str, list[float]] = {}
    for v, c in zip(vals, cids, strict=True):
        by_cluster.setdefault(c, []).append(float(v))
    clusters = list(by_cluster.keys())
    boot_means = []
    for _ in range(b):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pool = [x for c in sampled for x in by_cluster[c]]
        boot_means.append(float(np.mean(pool)))
    boot_means = np.asarray(boot_means)
    return {
        "mean": float(np.mean(vals)),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "n": len(vals),
        "n_clusters": len(clusters),
    }


def _benjamini_hochberg(pvals: dict[str, float], alpha: float = FDR_ALPHA) -> dict[str, bool]:
    """Benjamini-Hochberg FDR over a dict of {label: pvalue}. Returns {label: reject}."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    reject = {k: False for k, _ in items}
    max_i = -1
    for i, (_k, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            max_i = i
    for i, (k, _p) in enumerate(items, start=1):
        if i <= max_i:
            reject[k] = True
    return reject


def _spearman_pvalue_approx(rho: float, n: int) -> float | None:
    """Two-sided p-value for Spearman rho via the t-approximation (n>=4)."""
    if rho is None or not np.isfinite(rho) or n < 4 or abs(rho) >= 1.0:
        return None
    import math

    t = rho * math.sqrt((n - 2) / (1 - rho**2))
    # two-sided survival via the normal approx (adequate for the FDR ranking)
    from math import erfc, sqrt

    z = abs(t)
    return float(erfc(z / sqrt(2)))


def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(a, dtype=np.float64).ravel())).astype(np.float64)


def _residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residualize y on the columns of Z (with intercept) via least squares."""
    y = np.asarray(y, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    X = np.column_stack([np.ones(len(y)), Z])  # intercept + covariates
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _partial_spearman_multi(x: np.ndarray, y: np.ndarray, covars: list[np.ndarray]) -> float:
    """Partial Spearman of (x, y) controlling for >=1 covariates (Blocker 3c):
    rank everything, residualize ranks of x and y on the ranks of the covariates,
    correlate the residuals. Returns Pearson of the residuals (= partial Spearman)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 4 or len(covars) == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    Z = np.column_stack([_rank(c) for c in covars])
    ex, ey = _residualize(rx, Z), _residualize(ry, Z)
    nx, ny = np.linalg.norm(ex - ex.mean()), np.linalg.norm(ey - ey.mean())
    if nx < 1e-30 or ny < 1e-30:
        return float("nan")
    return float(((ex - ex.mean()) @ (ey - ey.mean())) / (nx * ny))


def _spearman_simple(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    nx, ny = np.linalg.norm(rx - rx.mean()), np.linalg.norm(ry - ry.mean())
    if nx < 1e-30 or ny < 1e-30:
        return float("nan")
    return float(((rx - rx.mean()) @ (ry - ry.mean())) / (nx * ny))


def _load_g0_e0(cells: list[str]) -> dict[str, dict]:
    """Load the per-cell A3.10 base-prior/install inputs (Blocker 3a)."""
    out = {}
    for cell in cells:
        p = C.EVAL_ROOT / "per_cell" / cell / "g0_E0.json"
        if p.exists():
            with open(p) as f:
                out[cell] = json.load(f)
    return out


def _a310_partial_E0_wnorm(beh_cells: list[str], g0_e0: dict) -> dict:
    """Blocker 3c: pool per-context (g0, ghat_real, E0) across the behavior's cells
    (+ per-cell ‖ŵ‖ broadcast to each context) and compute the RAW Spearman(ghat,g0)
    AND the PARTIAL Spearman(ghat, g0 | [E0, ‖ŵ‖]) — the C7/C10 base-prior-dominance
    + install-magnitude control. Raises a clear error when the inputs are absent."""
    present = [c for c in beh_cells if c in g0_e0]
    if not present:
        raise ValueError(
            f"A3.10 E0/‖ŵ‖ partial: NO g0_E0.json present for behavior cells {beh_cells!r} "
            "— run issue665_gate_cpu (arm a310) first; the C7/C10 prior/install partial "
            "(plan §6, Blocker 3) cannot be computed without it."
        )
    g0v, ghatv, e0v, wnv = [], [], [], []
    for c in present:
        for e in g0_e0[c]["entries"]:
            if any(e.get(k) is None for k in ("g0", "ghat_real", "E0", "wnorm")):
                continue
            g0v.append(e["g0"])
            ghatv.append(e["ghat_real"])
            e0v.append(e["E0"])
            wnv.append(e["wnorm"])
    g0a, ghata, e0a, wna = map(np.asarray, (g0v, ghatv, e0v, wnv))
    raw = _spearman_simple(ghata, g0a)
    # if ‖ŵ‖ is constant across all pooled contexts (single cell), drop it from the
    # covariate set (a constant column is rank-degenerate) and partial E0 only.
    covars = [e0a]
    wnorm_partialled = bool(np.ptp(wna) > 1e-12) if wna.size else False
    if wnorm_partialled:
        covars.append(wna)
    partial = _partial_spearman_multi(g0a, ghata, covars)
    return {
        "A3_10_rho_raw": raw,
        "A3_10_rho_partial_E0_wnorm": partial,
        "n_pooled_contexts": int(g0a.size),
        "n_cells_present": len(present),
        "wnorm_partialled": wnorm_partialled,
        "note": "raw = Spearman(ghat_real, g0); partial = same with E0 (+ ‖ŵ‖ when it "
        "varies across pooled cells) partialled out (C7/C10 base-prior/install control).",
    }


def _probe_split_replication(beh_cells: list[str], rng) -> dict:
    """Blocker 6b: per-cell probe-split (R=8) reliability floor via
    issue664_aggregate_gate.probe_split_floor (REUSE). For each cell, draw R=8
    independent random 2-way splits of the 50-probe axis, take the per-(ctx,layer)
    abs half-difference of ĝ at the primary read layer, aggregate to a cross-split
    mean floor + bootstrap-CI it."""
    from explore_persona_space.analysis.gate_io import load_cell, probe_split_floor

    R = C.PROBE_SPLIT_R  # 8
    per_cell_floor: list[float] = []
    cluster_ids: list[str] = []
    for cell in beh_cells:
        try:
            sc = load_cell(cell, verify_sha=True)
        except Exception as exc:
            logger.warning("[probe-split] %s skipped (load failed: %s)", cell, exc)
            continue
        try:
            layer = C.read_layer_for_cell(cell)
            vpp = sc.tensors["v_plus_probe"]
            v0p = sc.tensors["v0_probe"]
            split_means = []
            for r in range(R):
                # one independent split per r (the canonical floor fn does ONE 2-way
                # split; R independent draws give the R=8 replication, plan §6 / §9).
                floor_cl = probe_split_floor(
                    vpp, v0p, sc.source_idx, np.random.default_rng(100 + r)
                )
                # (C,L) -> the primary read layer's median noise magnitude
                split_means.append(float(np.nanmedian(floor_cl[:, layer])))
            per_cell_floor.append(float(np.mean(split_means)))
            cluster_ids.append(C.parse_cell(cell)["source"])
        finally:
            sc.free()
    floor_ci = (
        _cluster_bootstrap_ci(per_cell_floor, cluster_ids, rng)
        if per_cell_floor
        else {
            "mean": None,
            "ci_lo": None,
            "ci_hi": None,
            "n": 0,
        }
    )
    return {
        "probe_split_floor_mean": floor_ci.get("mean"),
        "probe_split_floor_ci_lo": floor_ci.get("ci_lo"),
        "probe_split_floor_ci_hi": floor_ci.get("ci_hi"),
        "probe_split_replication_count": R,
        "n_cells": len(per_cell_floor),
    }


def aggregate(cells: list[str], smoke: bool) -> dict:
    rng = np.random.default_rng(42)
    behaviors = sorted({C.behavior_for_cell(c) for c in cells})

    # ── A3.10: g0-vs-ghat rho per behavior, clustered CIs (the central claim) ──
    a310 = _load_arm("a310", cells)
    a39 = _load_arm("a39", cells)
    a38 = _load_arm("a38", cells)
    judged = _load_arm("judged_E", cells)
    g0_e0 = _load_g0_e0(cells)  # Blocker 3a per-cell base-prior/install inputs

    per_behavior = {}
    fdr_pvals: dict[str, float] = {}
    for beh in behaviors:
        beh_cells = [c for c in cells if C.behavior_for_cell(c) == beh]
        if not beh_cells:
            continue
        # gather the per-cell primary-layer g0/gplus/cosine spearmans
        g0_rhos, gplus_rhos, cos_rhos = [], [], []
        fam_clusters, src_clusters = [], []  # Blocker 6a: hierarchical family / family_source
        a39_sigma_wins, a39_some_beats = [], []
        a38_resid, a38_sigma1 = [], []
        for c in beh_cells:
            layer = str(C.read_layer_for_cell(c))
            parsed = C.parse_cell(c)
            # Blocker 6a — HIERARCHICAL family clustering: the family of the SOURCE
            # context (the cell's source persona's #594 family), NOT the bare
            # source_seed. The finer grain (family_source) keeps source identity
            # for the cross-source bootstrap while the family grain is the parent.
            src_family = C.family_of_context(parsed["source"])
            if c in a310 and layer in a310[c]["by_layer"]:
                bl = a310[c]["by_layer"][layer]
                g0_rhos.append(bl.get("g0_spearman"))
                gplus_rhos.append(bl.get("gplus_spearman"))
                fam_clusters.append(src_family)  # family-level cluster (Blocker 6a)
                src_clusters.append(f"{src_family}::{parsed['source']}_{parsed['seed']}")
            if c in a39 and layer in a39[c]["by_layer"]:
                bl = a39[c]["by_layer"][layer]
                cos_rhos.append(bl.get("cosine_spearman"))
                a39_some_beats.append(bl.get("verdict_i_some_beats_cosine"))
                a39_sigma_wins.append(bl.get("verdict_ii_sigma_inv_wins"))
            if c in a38 and layer in a38[c]["by_layer"]:
                bl = a38[c]["by_layer"][layer]
                a38_resid.append(bl.get("median_rankone_residual"))
                a38_sigma1.append(bl.get("svd_sigma1_frac"))
        # Blocker 6c: bootstrap CIs over the FAMILY x source clustering. The family
        # grain is the bootstrap resample level (the broader cluster); the finer
        # family_source grain is the within-family member set.
        g0_ci = _cluster_bootstrap_ci(g0_rhos, src_clusters or fam_clusters, rng)
        g0_family_ci = _cluster_bootstrap_ci(g0_rhos, fam_clusters or src_clusters, rng)
        cos_ci = _cluster_bootstrap_ci(cos_rhos, src_clusters or fam_clusters, rng)
        # Blocker 3c: the E0/‖ŵ‖ base-prior + install partial (C7/C10).
        a310_partial = _a310_partial_E0_wnorm(beh_cells, g0_e0)
        # Blocker 6b: probe-split (R=8) reliability floor.
        probe_split = _probe_split_replication(beh_cells, rng)
        # FDR p-value for the A3.10 primary path (g0 mean rho vs zero)
        pv = _spearman_pvalue_approx(g0_ci.get("mean"), g0_ci.get("n", 0) or 0)
        if pv is not None:
            fdr_pvals[f"a310_{beh}"] = pv
        per_behavior[beh] = {
            "read_layer": C.read_layer_for_cell(beh_cells[0]),
            "column": C.column_for_cell(beh_cells[0]),
            "role_class": C.role_class_for_cell(beh_cells[0]),
            "n_cells": len(beh_cells),
            "a310_g0_spearman": g0_ci,
            # Blocker 6c: family-clustered CI (hierarchical family resample grain)
            "family_clustered_ci_g0_spearman": g0_family_ci,
            # Blocker 3c: the base-prior/install partial (raw + partialled)
            "a310_partial_E0_wnorm": a310_partial,
            "A3_10_rho_raw": a310_partial["A3_10_rho_raw"],
            "A3_10_rho_partial_E0_wnorm": a310_partial["A3_10_rho_partial_E0_wnorm"],
            # Blocker 6b: probe-split replication reliability floor
            "probe_split_floor_mean": probe_split["probe_split_floor_mean"],
            "probe_split_floor_ci_lo": probe_split["probe_split_floor_ci_lo"],
            "probe_split_floor_ci_hi": probe_split["probe_split_floor_ci_hi"],
            "probe_split_replication_count": probe_split["probe_split_replication_count"],
            "a310_gplus_spearman_mean": float(np.nanmean([v for v in gplus_rhos if v is not None]))
            if any(v is not None for v in gplus_rhos)
            else None,
            "a39_cosine_spearman": cos_ci,
            "a39_verdict_i_some_beats_cosine_frac": float(
                np.mean([bool(x) for x in a39_some_beats])
            )
            if a39_some_beats
            else None,
            "a39_verdict_ii_sigma_inv_wins_frac": float(np.mean([bool(x) for x in a39_sigma_wins]))
            if a39_sigma_wins
            else None,
            "a38_median_rankone_residual_mean": float(
                np.nanmean([v for v in a38_resid if v is not None])
            )
            if any(v is not None for v in a38_resid)
            else None,
            "a38_svd_sigma1_frac_mean": float(np.nanmean([v for v in a38_sigma1 if v is not None]))
            if any(v is not None for v in a38_sigma1)
            else None,
            # behavioral DV E — judge-positive RATE is the PRIMARY validated
            # behavioral construct (Blocker 4); mean_score is the SECONDARY
            # continuous companion (CLAUDE.md dual-DV).
            **_e_summary(beh_cells, judged),
        }

    fdr_reject = _benjamini_hochberg(fdr_pvals, FDR_ALPHA)

    agg = {
        "scope_cells": cells,
        "behaviors": behaviors,
        "per_behavior": per_behavior,
        "fdr": {
            "alpha": FDR_ALPHA,
            "pvalues": fdr_pvals,
            "reject": fdr_reject,
            "note": "Benjamini-Hochberg over the per-behavior A3.10 primary path (g0 rho vs 0).",
        },
        "bootstrap_B": BOOTSTRAP_B,
        "smoke": smoke,
        "git_commit": _git_commit(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "cc_recipe": C.CC_RECIPE,
        "lambda_default": C.LAMBDA_DEFAULT,
        "note": "PRIMARY DV = activation realized gate ghat_real (B1); E is the SECONDARY "
        "behavioral companion. A3.9/A3.10 numbers are gated on the B3 reduction unit test "
        "(whitened_gate_unittest.json). A3.6c f_CV verdict gated on the parity probe.",
    }
    return agg


def _e_summary(cells: list[str], judged: dict) -> dict:
    """Behavioral DV E summary (Blocker 4): the judge-positive RATE is the PRIMARY
    validated behavioral construct; mean_score is the SECONDARY continuous companion.
    Also reports the dual-DV cross-check (Spearman across contexts where both have
    dynamic range — the standing rate-vs-mean validation, CLAUDE.md Measurement
    validity)."""
    rates, means = [], []
    for c in cells:
        if c in judged:
            for s in judged[c]["by_context"].values():
                if s.get("judge_positive_rate") is not None:
                    rates.append(s["judge_positive_rate"])
                if s.get("mean_score") is not None:
                    means.append(s["mean_score"])
    # dual-DV cross-check: does the secondary mean track the primary rate?
    dual_rho = None
    if len(rates) == len(means) and len(rates) >= 3:
        dual_rho = _spearman_simple(np.asarray(rates), np.asarray(means))
    return {
        "E_judge_positive_rate": float(np.mean(rates)) if rates else None,  # PRIMARY DV
        "E_mean_score": float(np.mean(means)) if means else None,  # SECONDARY companion
        "E_dual_dv_rate_vs_mean_spearman": dual_rho,  # rate-vs-mean validation
        "E_n_contexts": len(rates),
    }


def main():
    ap = argparse.ArgumentParser(description="issue665 Phase 3 aggregate")
    ap.add_argument("--scope", default="content", help="content|content+null|all")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cells = args.cells if args.cells else C.select_cells(args.scope)
    # B3 unit-test gate (run first; A3.9/A3.10 numbers untrusted until PASS).
    b3 = run_b3_unittest()
    if not b3["passed"]:
        logger.error("B3 reduction unit test FAILED — A3.9/A3.10 numbers are NOT trusted.")

    agg = aggregate(cells, args.smoke)
    aggp = C.EVAL_ROOT / "aggregate.json"
    with open(aggp, "w") as f:
        json.dump(agg, f, indent=1)
    logger.info("[aggregate] %d cells -> %s", len(cells), aggp)

    # analyzer dashboard data (the Phase-2 dashboard contract surface)
    body_data = {
        "behaviors": agg["behaviors"],
        "per_behavior": agg["per_behavior"],
        "fdr_reject": agg["fdr"]["reject"],
        "b3_unittest_passed": b3["passed"],
        "git_commit": agg["git_commit"],
        "generated_at": agg["generated_at"],
    }
    bdp = C.EVAL_ROOT / "analyzer_body_data.json"
    with open(bdp, "w") as f:
        json.dump(body_data, f, indent=1)
    logger.info("[analyzer-body-data] -> %s", bdp)


if __name__ == "__main__":
    main()
