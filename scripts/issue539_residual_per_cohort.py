"""Task #539 — per-cohort geometry-on-residual re-read of the #532 panel.

Analysis-only (CPU, local). Rebuilds the 416-cell loc-arm ep1 panel from the
committed ``eval_results/issue_532/`` JSONs, splits it into the two cohorts the
parent's headline averaged over (cross-context ordinary, n=240; instructed
strip, n=160), residualizes the on-policy in-R marker-emission rate on the
per-bystander base prior (rate-space OLS), and reports the per-cohort Spearman
rho of each geometric predictor (cosine, Gaussian-KL@L22 primary; JS-v1
exploratory) against the residual — with bootstrap CIs, permutation p, cluster
bootstraps, tie diagnostics, source-dose-confound controls, a between-cohort
delta-rho contrast, and Holm correction over the 4 primary tests.

Provenance: the panel row-building logic and the Spearman / bootstrap /
permutation machinery are vendored from the parent script
``scripts/issue532_predictor_stress.py`` at SHA 296c4da2d (issue-532 branch,
never merged to main), with deltas documented inline per the plan
(tasks/running/539/plans/plan.md sections 4.2, 6.2).

Step-0 consistency gate: before any new number is computed, three parent rho
values and four cell counts are reproduced from the rebuilt panel, and the
phase-0 measurement payload (``phase0_base_prior.json``) is cross-checked
against the analysis copy of the base prior (``predictors.json::base_prior``)
on every runtime bystander; any mismatch aborts with exit code 1 (plan
sections 4.1 + 4.5 — the kill criterion).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

# ── Constants ────────────────────────────────────────────────────────────────

PARENT_TASK = 532
PARENT_PANEL_SHA = "296c4da2d"  # SHA whose issue532_predictor_stress.py is vendored below

# Step-0 reference values (plan §4.5), read from
# eval_results/issue_532/analysis.json::union_panel_rho and hard-coded here so
# the gate also catches a silently-edited analysis.json.
REF_COSINE_UNION_RHO = 0.22443789394759853
REF_GKL_ORDINARY_RHO = -0.6200033818751595  # n=256, INCLUDING diagonal (parent's definition)
REF_BASE_PRIOR_UNION_RHO = 0.720034210988633
REF_N_TOTAL = 416
REF_N_ORDINARY = 256
REF_N_INSTRUCTED = 160
REF_N_ORDINARY_CROSS = 240
RHO_TOL = 1e-6
BASE_PRIOR_XCHECK_TOL = 1e-12  # phase0_base_prior.json vs predictors.json::base_prior
FE_GROUP_MEAN_TOL = 1e-8  # post-residualization max |group mean|, every reported slice

ALL_PKS = ("cosine", "gauss_kl", "js_v1")  # js_v1 exploratory (deprecated estimator, plan D7)
PRIMARY_PKS = ("cosine", "gauss_kl")
COHORT_NAMES = ("ordinary_cross", "instructed_strip")
STYLIZED_SOURCES = ("A3", "A4", "A5")  # pirate / comedian / villainous mastermind (#502, #532)

PK_DISPLAY = {
    "cosine": "Cosine similarity",
    "gauss_kl": "Gaussian-KL distance @ L22",
    "js_v1": "JS divergence (v1, deprecated)",
}
COHORT_DISPLAY = {
    "ordinary_cross": "Ordinary cross-context",
    "instructed_strip": "Instructed strip",
}
BAND_DISPLAY = {
    "explicit": "Explicit instruction",
    "soft": "Soft instruction",
    "oblique": "Oblique instruction",
}


def _bystander_display(label: str) -> str:
    """Plain-English display name for a bystander label (no opaque slugs in figures)."""
    if label.startswith("instr_"):
        _, band, num = label.split("_")
        return f"{band.capitalize()} instruction {num}"
    return CONDITIONS_BY_ID[label].name


# ── Vendored statistics (from issue532_predictor_stress.py @ 296c4da2d) ──────


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (single-pass, NaN-safe).

    Vendored VERBATIM from issue532_predictor_stress.py @ 296c4da2d (line 1660).
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def _is_degenerate(v: np.ndarray) -> bool:
    """True when ``v`` has < 2 unique values (Spearman undefined / meaningless)."""
    return len(np.unique(v)) < 2


def _fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via average ranks + Pearson on the ranks.

    Numerically identical to the vendored scipy ``_spearman_rho`` (asserted at
    startup against the real cohort data; |diff| < 1e-9); used inside the
    resampling loops where scipy's per-call overhead dominates wall time.
    Callers guarantee NaN-free, non-degenerate inputs.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def _rowwise_rank_rho(xm: np.ndarray, ym: np.ndarray) -> np.ndarray:
    """Row-wise Spearman rho for matched ``(n_rows, n)`` matrices (vectorized).

    Same average-rank + Pearson-on-ranks computation as ``_fast_spearman``,
    one row per resample. Callers must filter degenerate (constant) rows first.
    """
    rx = rankdata(xm, axis=1).astype(np.float64)
    ry = rankdata(ym, axis=1).astype(np.float64)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    num = (rx * ry).sum(axis=1)
    den = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
    return num / den


def _nondegenerate_rows(*mats: np.ndarray) -> np.ndarray:
    """Boolean mask of rows where EVERY matrix has >=2 distinct values."""
    good = np.ones(mats[0].shape[0], dtype=bool)
    for m in mats:
        good &= m.max(axis=1) > m.min(axis=1)
    return good


def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, seed: int
) -> dict[str, float | int]:
    """Percentile bootstrap 95% CI on Spearman rho via simple pair resampling.

    ESTIMAND vendored from issue532_predictor_stress.py @ 296c4da2d (line 1671:
    resample n pairs with replacement, Spearman per resample, 2.5/97.5
    percentiles, seed-42 ``np.random.default_rng``). Two implementation deltas,
    both documented per plan §6.2(8) + §13: (a) a resample whose x or y is
    constant is DROPPED and COUNTED (``n_degenerate_resamples``) instead of
    silently nanmean'd; (b) the per-resample loop is vectorized (one index
    matrix + row-wise rank rho) — same estimand, same seed discipline, but a
    different draw realization than the parent's sequential loop (immaterial:
    rep counts differ from the parent anyway).
    """
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return {
            "boot_mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": n_boot,
            "n_degenerate_resamples": 0,
        }
    idx = rng.integers(0, n, size=(n_boot, n))
    xb, yb = x[idx], y[idx]
    good = _nondegenerate_rows(xb, yb)
    rhos = _rowwise_rank_rho(xb[good], yb[good])
    return {
        "boot_mean": float(np.mean(rhos)),
        "low": float(np.percentile(rhos, 2.5)),
        "high": float(np.percentile(rhos, 97.5)),
        "n_boot": n_boot,
        "n_degenerate_resamples": int((~good).sum()),
    }


def _permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> dict[str, float | int]:
    """Two-sided permutation p for Spearman rho (shuffle y across cells).

    ESTIMAND vendored from issue532_predictor_stress.py @ 296c4da2d
    (``_signflip_permutation_test``, line 1966: permute y, Spearman against
    fixed x, two-sided |rho| comparison, seed-42 rng). Two implementation
    deltas, documented per plan §4.2 + §13: (a) the p-value uses the add-one
    formula p = (1 + #{|rho_perm| >= |rho_obs|}) / (n_perm + 1) instead of the
    parent's plain proportion; (b) the permutation loop is vectorized — the
    ranks of a permuted y are the permuted ranks of y (true under ties:
    average ranks travel with the values), so all n_perm null rhos come from
    one row-permuted rank matrix. Permuting y cannot create a degenerate
    resample (the multiset is invariant), so no drop+count is needed here.
    """
    rng = np.random.default_rng(seed)
    rho_obs = _spearman_rho(x, y)
    if np.isnan(rho_obs):
        return {"p": float("nan"), "rho_obs": rho_obs, "n_perm": n_perm}
    rx = rankdata(x)
    rx -= rx.mean()
    ry = rankdata(y)
    perms = rng.permuted(np.tile(ry, (n_perm, 1)), axis=1)
    perms -= perms.mean(axis=1, keepdims=True)
    num = perms @ rx
    den = np.sqrt((rx @ rx) * (perms * perms).sum(axis=1))
    null_rhos = num / den
    count = int((np.abs(null_rhos) >= abs(rho_obs)).sum())
    return {
        "p": float((1 + count) / (n_perm + 1)),
        "rho_obs": float(rho_obs),
        "null_mean": float(np.mean(null_rhos)),
        "null_sd": float(np.std(null_rhos)),
        "n_perm": n_perm,
    }


# ── New statistics (plan §6.2) ───────────────────────────────────────────────


def residualize(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rate-space OLS residual of y on x (plan §4.2).

    Zero-variance x (the ordinary cohort: base prior ≡ 0 on all 16 ordinary
    bystanders) makes the regression degenerate; the residual is then y minus
    its mean and the audit carries ``noop: true`` (a constant shift — Spearman
    on the residual equals Spearman on the raw DV).
    """
    if float(np.std(x)) < 1e-12:
        return y - float(np.mean(y)), {"noop": True, "slope": None, "intercept": None, "r2": None}
    # Closed-form simple OLS (equivalent to np.polyfit(x, y, 1); avoids the
    # lstsq overhead inside the cluster-bootstrap re-residualization loop).
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    xc = x - x_mean
    slope = float((xc @ (y - y_mean)) / (xc @ xc))
    intercept = y_mean - slope * x_mean
    resid = y - (intercept + slope * x)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return resid, {
        "noop": False,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def _group_mean_broadcast(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group mean of ``values``, broadcast back to each row."""
    out = np.empty_like(values, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        out[m] = float(values[m].mean())
    return out


def _twoway_fe_residualize(
    values: np.ndarray, src: np.ndarray, byst: np.ndarray
) -> tuple[np.ndarray, float]:
    """Exact two-way (source + bystander) fixed-effects residual (plan D10).

    Dummy regression: OLS of ``values`` on an intercept + one-hot source dummies
    + one-hot bystander dummies via ``np.linalg.lstsq``; the returned residual is
    the exact within-estimator residual on ANY panel, balanced or not (residuals
    of a least-squares projection are invariant to the rank-deficient
    parametrization, so no reference level needs dropping). Replaces the round-1
    single-pass shortcut ``v - src_mean - byst_mean + grand_mean``, which is the
    FE residual ONLY on complete balanced rectangles and left nonzero FE group
    means on the unbalanced ordinary_cross cohort (16x16 minus diagonal) —
    round-1 ensemble code-review binding fix; on ordinary_cross the correction
    moves rho_twoway cosine ~+0.138 -> ~+0.164 and gauss_kl ~+0.077 -> ~+0.044.

    Fail-loud postcondition, enforced HERE so it fires for every reported slice
    (primary cohorts + all robustness slices route through this function): the
    max |residual group mean| over sources AND bystanders must be below
    ``FE_GROUP_MEAN_TOL`` (1e-8), else RuntimeError.

    Returns ``(residuals, max_abs_group_mean)`` — the audit scalar is persisted
    in the output JSON per cohort/predictor.
    """
    n = len(values)
    v = values.astype(np.float64)
    src_u, src_inv = np.unique(src, return_inverse=True)
    byst_u, byst_inv = np.unique(byst, return_inverse=True)
    design = np.zeros((n, 1 + len(src_u) + len(byst_u)), dtype=np.float64)
    design[:, 0] = 1.0
    design[np.arange(n), 1 + src_inv] = 1.0
    design[np.arange(n), 1 + len(src_u) + byst_inv] = 1.0
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    resid = v - design @ coef
    worst = 0.0
    for groups in (src, byst):
        for g in np.unique(groups):
            worst = max(worst, abs(float(resid[groups == g].mean())))
    if worst >= FE_GROUP_MEAN_TOL:
        raise RuntimeError(
            f"two-way FE residualization failed its postcondition: max |residual group "
            f"mean| = {worst:.3e} >= {FE_GROUP_MEAN_TOL:.0e} "
            f"(n={n}, {len(src_u)} sources x {len(byst_u)} bystanders)"
        )
    return resid, worst


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank-based partial Spearman of x vs y controlling z (plan D10).

    Rank-transform all three, OLS-residualize rank(x) and rank(y) on rank(z),
    Pearson-correlate the residuals. A constant z degenerates to the plain
    Spearman (the control carries no information).
    """
    if _is_degenerate(z):
        return _spearman_rho(x, y)
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rx_res = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry_res = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    if float(np.std(rx_res)) < 1e-12 or float(np.std(ry_res)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx_res, ry_res)[0, 1])


def _cluster_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    clusters: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Cluster bootstrap CI on the residual Spearman rho (plan §6.2 item 2).

    Resamples clusters with replacement, RE-RESIDUALIZES the DV on the base
    prior within each resample, then computes Spearman(geometry, residual).
    Degenerate resamples (constant geometry or constant residual) are dropped
    and counted.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    cluster_idx = {c: np.where(clusters == c)[0] for c in uniq}
    rhos: list[float] = []
    n_degenerate = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([cluster_idx[c] for c in chosen])
        xb, yb, pb = x[idx], y[idx], prior[idx]
        resid_b, _ = residualize(yb, pb)
        if _is_degenerate(xb) or _is_degenerate(resid_b):
            n_degenerate += 1
            continue
        rhos.append(_fast_spearman(xb, resid_b))
    if not rhos:
        return {
            "low": float("nan"),
            "high": float("nan"),
            "boot_mean": float("nan"),
            "n_clusters": len(uniq),
            "n_boot": n_boot,
            "n_degenerate_resamples": n_degenerate,
        }
    arr = np.array(rhos)
    return {
        "low": float(np.percentile(arr, 2.5)),
        "high": float(np.percentile(arr, 97.5)),
        "boot_mean": float(np.mean(arr)),
        "n_clusters": len(uniq),
        "n_boot": n_boot,
        "n_degenerate_resamples": n_degenerate,
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (Holm 1979; plan D6)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running_max)
    return [float(p) for p in adjusted]


# ── Panel construction ───────────────────────────────────────────────────────


def build_panel(in_dir: Path) -> dict:
    """Long-format panel of the 416 loc-arm ep1 cells.

    Row-building logic vendored from ``_build_union_panel`` in
    issue532_predictor_stress.py @ 296c4da2d (lines 1732-1878), restricted to
    the columns this analysis consumes. Faithfully ported semantics:

    * the PRIMARY DV is ``summary.in_R_emission_rate`` — the parent kept it
      under the legacy key ``trained_logp`` (parent line 1827, round-3 binding
      DV revision); here it is stored under the honest name ``emit_rate``;
    * ``base_prior`` comes from ``predictors.json::base_prior`` (the analysis
      copy the parent's hierarchy consumed), a per-bystander scalar broadcast
      over sources;
    * predictor matrices are indexed [source_row, bystander_col];
    * ANY missing cell JSON fails loud (the parent's standing-rec-2 behavior;
      no partial-panel escape hatch is offered here — plan §13).
    """
    predictors = json.loads((in_dir / "predictors.json").read_text())
    sources: list[str] = predictors["sources"]
    bystanders: list[str] = predictors["bystanders"]
    cosine_m = np.array(predictors["cosine_matrix"], dtype=np.float64)
    js_v1_m = np.array(predictors["js_v1_matrix"], dtype=np.float64)
    gkl_m = np.array(predictors["gauss_kl_matrix"], dtype=np.float64)
    base_prior_map: dict[str, float] = predictors["base_prior"]

    cell_dir = in_dir / "per_cell" / "loc_ep1"
    rows: list[dict] = []
    missing: list[tuple[str, str]] = []
    for i, src in enumerate(sources):
        for j, byst in enumerate(bystanders):
            cell_path = cell_dir / f"cell_loc_ep1_{src}__{byst}.json"
            if not cell_path.exists():
                missing.append((src, byst))
                continue
            cell = json.loads(cell_path.read_text())
            s = cell["summary"]
            kind = cell["bystander_kind"]
            if (kind == "instructed") != byst.startswith("instr_"):
                raise RuntimeError(
                    f"bystander_kind/label disagreement at cell ({src}, {byst}): "
                    f"kind={kind!r} but label prefix says otherwise"
                )
            rows.append(
                {
                    "source_cid": src,
                    "bystander_label": byst,
                    # PRIMARY DV — parent's `trained_logp` key (line 1827):
                    "emit_rate": float(s["in_R_emission_rate"]),
                    # SECONDARY graded DV (appended-slot log-prob):
                    "extra_marker_logp": float(s["extra_marker_logp"]),
                    "base_prior": float(base_prior_map[byst]),
                    "is_instructed": int(kind == "instructed"),
                    "cosine": float(cosine_m[i, j]),
                    "js_v1": float(js_v1_m[i, j]),
                    "gauss_kl": float(gkl_m[i, j]),
                    "strength_band": cell["strength_band"],
                    "source_class": src[0],
                    "bystander_class": byst[0],  # class letter; meaningless for instr_*
                }
            )
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} cell JSON(s) under {cell_dir} — expected "
            f"{len(sources) * len(bystanders)}, found {len(rows)}; first missing: {missing[:5]!r}"
        )
    panel = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    panel["_n"] = len(rows)
    panel["_sources"] = sources
    panel["_bystanders"] = bystanders
    return panel


def cohort_masks(panel: dict) -> dict[str, np.ndarray]:
    """The two primary cohorts + robustness slices (plan §5)."""
    is_ord = panel["is_instructed"] == 0
    is_instr = panel["is_instructed"] == 1
    off_diag = panel["source_cid"] != panel["bystander_label"]
    nonstylized_src = ~np.isin(panel["source_cid"], STYLIZED_SOURCES)
    nonstylized_byst = ~np.isin(panel["bystander_label"], STYLIZED_SOURCES)
    return {
        "ordinary_all": is_ord,
        "ordinary_cross": is_ord & off_diag,
        "instructed_strip": is_instr,
        "nonstylized_ordinary_cross": is_ord & off_diag & nonstylized_src,
        "nonstylized_instructed_strip": is_instr & nonstylized_src,
        "nonstylized_strict_ordinary_cross": (
            is_ord & off_diag & nonstylized_src & nonstylized_byst
        ),
        "class_letter_cross": is_ord & (panel["source_class"] != panel["bystander_class"]),
    }


# ── Step-0 consistency gate (plan §4.5) ──────────────────────────────────────


def step0_consistency(panel: dict, in_dir: Path) -> dict:
    """Reproduce 3 parent rho values + 4 cell counts, cross-check the hard-coded
    constants against analysis.json AND phase0_base_prior.json against
    predictors.json::base_prior; sys.exit(1) on any mismatch."""
    masks = cohort_masks(panel)
    checks: list[dict] = []

    def check(name: str, got: float, want: float, tol: float) -> None:
        ok = bool(abs(got - want) <= tol)
        checks.append({"name": name, "got": float(got), "want": float(want), "pass": ok})

    check("n_total", panel["_n"], REF_N_TOTAL, 0)
    check("n_ordinary", int(masks["ordinary_all"].sum()), REF_N_ORDINARY, 0)
    check("n_instructed", int(masks["instructed_strip"].sum()), REF_N_INSTRUCTED, 0)
    check("n_ordinary_cross", int(masks["ordinary_cross"].sum()), REF_N_ORDINARY_CROSS, 0)

    y = panel["emit_rate"]
    check("cosine_union_rho", _spearman_rho(panel["cosine"], y), REF_COSINE_UNION_RHO, RHO_TOL)
    ord_all = masks["ordinary_all"]
    check(
        "gauss_kl_ordinary_only_rho_incl_diagonal",
        _spearman_rho(panel["gauss_kl"][ord_all], y[ord_all]),
        REF_GKL_ORDINARY_RHO,
        RHO_TOL,
    )
    check(
        "base_prior_union_rho",
        _spearman_rho(panel["base_prior"], y),
        REF_BASE_PRIOR_UNION_RHO,
        RHO_TOL,
    )

    # Cross-check the hard-coded reference constants against the committed
    # analysis.json (catches a silently-edited reference file or plan drift).
    parent_analysis = json.loads((in_dir / "analysis.json").read_text())
    upr = parent_analysis["union_panel_rho"]
    check("analysis_json_cosine_union", upr["cosine"]["rho_union"], REF_COSINE_UNION_RHO, 1e-9)
    check(
        "analysis_json_gkl_ordinary",
        upr["gauss_kl"]["rho_ordinary_only"],
        REF_GKL_ORDINARY_RHO,
        1e-9,
    )
    check(
        "analysis_json_base_prior_union",
        upr["base_prior"]["rho_union"],
        REF_BASE_PRIOR_UNION_RHO,
        1e-9,
    )

    # Plan §4.1 cross-check: the phase-0 measurement payload
    # (phase0_base_prior.json, per-bystander on-policy in-R emission rate of the
    # BASE model — the round-3 binding DV) must agree with the analysis copy the
    # parent's hierarchy consumed (predictors.json::base_prior) on every runtime
    # bystander. Catches a silently-regenerated or stale predictors.json.
    predictors = json.loads((in_dir / "predictors.json").read_text())
    base_prior_map: dict[str, float] = predictors["base_prior"]
    phase0 = json.loads((in_dir / "phase0_base_prior.json").read_text())
    per_byst = phase0["per_bystander"]
    runtime_bystanders: list[str] = panel["_bystanders"]
    covered = [b for b in runtime_bystanders if b in per_byst]
    check("phase0_base_prior_coverage", len(covered), len(runtime_bystanders), 0)
    if covered:
        max_diff = max(
            abs(float(base_prior_map[b]) - float(per_byst[b]["on_policy_emit_rate"]))
            for b in covered
        )
        check("phase0_base_prior_max_abs_diff", max_diff, 0.0, BASE_PRIOR_XCHECK_TOL)

    failed = [c for c in checks if not c["pass"]]
    if failed:
        print(
            "STEP-0 CONSISTENCY GATE FAILED — the rebuilt panel diverges from the", file=sys.stderr
        )
        print(
            "parent's panel semantics. NOT computing any new number (plan §4.5).", file=sys.stderr
        )
        for c in failed:
            print(f"  FAIL {c['name']}: got {c['got']!r}, want {c['want']!r}", file=sys.stderr)
        sys.exit(1)
    print(f"[step0] consistency gate PASS ({len(checks)} checks)")
    return {
        "pass": True,
        "checks": checks,
        "parent_analysis_git_commit": parent_analysis["metadata"]["git_commit"],
    }


# ── Per-cohort computation (plan §4.2) ───────────────────────────────────────


def compute_cohort_suite(
    panel: dict,
    mask: np.ndarray,
    args: argparse.Namespace,
    dv_key: str = "emit_rate",
    pks: tuple[str, ...] = ALL_PKS,
) -> dict:
    """All per-(predictor x cohort) statistics for one cohort mask."""
    y = panel[dv_key][mask].astype(np.float64)
    prior = panel["base_prior"][mask].astype(np.float64)
    src = panel["source_cid"][mask]
    byst = panel["bystander_label"][mask]

    resid, resid_audit = residualize(y, prior)
    fe = y - _group_mean_broadcast(y, byst)
    y_twoway, y_fe_worst = _twoway_fe_residualize(y, src, byst)
    source_dose = _group_mean_broadcast(y, src)  # source-marginal emission, broadcast

    out: dict = {
        "n": int(mask.sum()),
        "dv": dv_key,
        "residualization": resid_audit,
        "twoway_fe_audit": {
            "estimator": "dummy-regression lstsq (exact on unbalanced panels)",
            "group_mean_tol": FE_GROUP_MEAN_TOL,
            "dv_max_abs_group_mean": y_fe_worst,
            "geometry_max_abs_group_mean": {},
        },
        "predictors": {},
        "source_marginal": {},
    }
    for pk in pks:
        x = panel[pk][mask].astype(np.float64)
        x_twoway, x_fe_worst = _twoway_fe_residualize(x, src, byst)
        out["twoway_fe_audit"]["geometry_max_abs_group_mean"][pk] = x_fe_worst
        nonzero = y != 0.0
        binary = (y > 0).astype(np.float64)
        block = {
            "rho_raw": _spearman_rho(x, y),
            "rho_resid": _spearman_rho(x, resid),
            "rho_fe": _spearman_rho(x, fe),
            "rho_twoway": _spearman_rho(x_twoway, y_twoway),
            "rho_partial_source_dose": _partial_spearman(x, y, source_dose),
            "ci95_resid": _bootstrap_spearman_ci(x, resid, args.n_boot, args.seed),
            "ci95_raw": _bootstrap_spearman_ci(x, y, args.n_boot, args.seed),
            "ci95_fe": _bootstrap_spearman_ci(x, fe, args.n_boot, args.seed),
            "p_perm_resid": _permutation_p(x, resid, args.n_perm, args.seed),
            "ci95_cluster_bystander": _cluster_bootstrap_ci(
                x, y, prior, byst, args.n_cluster_boot, args.seed
            ),
            "ci95_cluster_source": _cluster_bootstrap_ci(
                x, y, prior, src, args.n_cluster_boot, args.seed
            ),
            "tie_diagnostics": {
                "n": len(y),
                "n_zero_dv": int((y == 0.0).sum()),
                "n_nonzero_dv": int(nonzero.sum()),
                "n_unique_dv": len(np.unique(y)),
                "rho_binarized": (None if _is_degenerate(binary) else _spearman_rho(x, binary)),
                "rho_nonzero_subset": (
                    None
                    if nonzero.sum() < 3 or _is_degenerate(y[nonzero])
                    else _spearman_rho(x[nonzero], y[nonzero])
                ),
            },
        }
        out["predictors"][pk] = block

        # Direct n=16 dose-confound read (plan D10): row-mean geometry vs
        # row-mean emission over the cohort's bystanders, per source.
        uniq_src = sorted(np.unique(src).tolist())
        mean_geom = np.array([float(x[src == s_].mean()) for s_ in uniq_src])
        mean_emit = np.array([float(y[src == s_].mean()) for s_ in uniq_src])
        out["source_marginal"][pk] = {
            "n_sources": len(uniq_src),
            "rho": _spearman_rho(mean_geom, mean_emit),
            "per_source": {
                s_: {"mean_geometry": float(g), "mean_emission": float(e)}
                for s_, g, e in zip(uniq_src, mean_geom, mean_emit, strict=True)
            },
        }
    return out


def compute_delta_rho(panel: dict, masks: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    """Between-cohort contrast delta_rho = rho_resid(ordinary) - rho_resid(instructed).

    Bootstrap CI resamples cells WITHIN each cohort independently per rep
    (plan §4.2 / D11). Resampling operates on the residualized pairs — no
    re-residualization per rep, matching the cell-level ``ci95_resid``
    machinery (the cluster bootstraps are the re-residualizing variants).
    Degenerate resamples on either side are dropped and counted.
    """
    rng = np.random.default_rng(args.seed)
    sides: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cohort in COHORT_NAMES:
        m = masks[cohort]
        y = panel["emit_rate"][m].astype(np.float64)
        resid, _ = residualize(y, panel["base_prior"][m].astype(np.float64))
        sides[cohort] = (resid, m)

    out: dict = {}
    for pk in PRIMARY_PKS:
        resid_o, m_o = sides["ordinary_cross"]
        resid_i, m_i = sides["instructed_strip"]
        x_o = panel[pk][m_o].astype(np.float64)
        x_i = panel[pk][m_i].astype(np.float64)
        delta_obs = _spearman_rho(x_o, resid_o) - _spearman_rho(x_i, resid_i)
        n_o, n_i = len(x_o), len(x_i)
        idx_o = rng.integers(0, n_o, size=(args.n_boot, n_o))
        idx_i = rng.integers(0, n_i, size=(args.n_boot, n_i))
        xo, ro = x_o[idx_o], resid_o[idx_o]
        xi, ri = x_i[idx_i], resid_i[idx_i]
        good = _nondegenerate_rows(xo, ro, xi, ri)
        deltas = _rowwise_rank_rho(xo[good], ro[good]) - _rowwise_rank_rho(xi[good], ri[good])
        out[pk] = {
            "delta_rho": float(delta_obs),
            "ci95_low": float(np.percentile(deltas, 2.5)),
            "ci95_high": float(np.percentile(deltas, 97.5)),
            "boot_mean": float(np.mean(deltas)),
            "n_boot": args.n_boot,
            "n_degenerate_resamples": int((~good).sum()),
            "note": "ordinary_cross minus instructed_strip; independent within-cohort "
            "cell resampling on residualized pairs (no per-rep re-residualization)",
        }
    return out


def compute_forest(panel: dict, masks: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    """Within-bystander rho (sources per bystander) across all 26 bystanders.

    Exploratory (plan §6.3 figure 2). Ordinary bystanders use their 15
    off-diagonal sources (cohort convention); instructed use all 16. Rows with
    a constant DV are reported as null AND counted — never silently averaged
    (plan §6.2 item 8).
    """
    cohort_union = masks["ordinary_cross"] | masks["instructed_strip"]
    out: dict = {pk: [] for pk in ALL_PKS}
    n_constant: dict[str, int] = {pk: 0 for pk in ALL_PKS}
    for byst in panel["_bystanders"]:
        m = cohort_union & (panel["bystander_label"] == byst)
        y = panel["emit_rate"][m].astype(np.float64)
        kind = "instructed" if byst.startswith("instr_") else "ordinary"
        for pk in ALL_PKS:
            x = panel[pk][m].astype(np.float64)
            if _is_degenerate(y) or _is_degenerate(x):
                n_constant[pk] += 1
                row = {
                    "bystander": byst,
                    "display": _bystander_display(byst),
                    "kind": kind,
                    "n": int(m.sum()),
                    "rho": None,
                    "ci95_low": None,
                    "ci95_high": None,
                    "constant_input": True,
                }
            else:
                ci = _bootstrap_spearman_ci(x, y, args.n_cluster_boot, args.seed)
                row = {
                    "bystander": byst,
                    "display": _bystander_display(byst),
                    "kind": kind,
                    "n": int(m.sum()),
                    "rho": _spearman_rho(x, y),
                    "ci95_low": ci["low"],
                    "ci95_high": ci["high"],
                    "constant_input": False,
                }
            out[pk].append(row)
    out["n_constant_dv_rows"] = n_constant
    return out


def compute_collinearity_gate(panel: dict, masks: dict[str, np.ndarray]) -> dict:
    """Pearson(geometry, base_prior) within the instructed strip (plan D9)."""
    m = masks["instructed_strip"]
    prior = panel["base_prior"][m].astype(np.float64)
    out = {}
    for pk in ALL_PKS:
        out[pk] = float(np.corrcoef(panel[pk][m].astype(np.float64), prior)[0, 1])
    return out


# ── Figures (plan §6.3) ──────────────────────────────────────────────────────


def _scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, **kw) -> None:
    ax.scatter(x, y, s=18, alpha=0.65, color=color, edgecolors="none", **kw)


def _block_title(cohort_label: str, block: dict, extra: str = "") -> str:
    ci = block["ci95_resid"]
    return (
        f"{cohort_label} (n={block['tie_diagnostics']['n']}){extra}\n"
        f"ρ={block['rho_resid']:+.2f} [{ci['low']:+.2f}, {ci['high']:+.2f}]"  # noqa: RUF001
    )


def fig_hero(panel: dict, masks: dict, results: dict, fig_dir: Path) -> None:
    """2 predictors x 3 columns: ordinary scatter / instructed raw / instructed resid."""
    colors = paper_palette(3)
    band_colors = dict(zip(("explicit", "soft", "oblique"), colors, strict=True))
    m_ord = masks["ordinary_cross"]
    m_ins = masks["instructed_strip"]
    y_ord = panel["emit_rate"][m_ord]
    y_ins = panel["emit_rate"][m_ins]
    resid_ins, _ = residualize(
        y_ins.astype(np.float64), panel["base_prior"][m_ins].astype(np.float64)
    )
    bands_ins = panel["strength_band"][m_ins]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for row, pk in enumerate(PRIMARY_PKS):
        blk_o = results["cohorts"]["ordinary_cross"]["predictors"][pk]
        blk_i = results["cohorts"]["instructed_strip"]["predictors"][pk]
        x_ord = panel[pk][m_ord]
        x_ins = panel[pk][m_ins]

        ax = axes[row, 0]
        _scatter(ax, x_ord, y_ord, colors[0])
        ax.set_title(_block_title("Ordinary cross-context", blk_o, extra=" — residual ≡ raw"))
        ax.set_xlabel(PK_DISPLAY[pk])
        ax.set_ylabel("On-policy ※ emission rate")

        ax = axes[row, 1]
        for band in ("explicit", "soft", "oblique"):
            bm = bands_ins == band
            _scatter(ax, x_ins[bm], y_ins[bm], band_colors[band], label=BAND_DISPLAY[band])
        ax.set_title(
            f"Instructed strip RAW (n={blk_i['tie_diagnostics']['n']})\nρ={blk_i['rho_raw']:+.2f}"  # noqa: RUF001
        )
        ax.set_xlabel(PK_DISPLAY[pk])
        ax.set_ylabel("On-policy ※ emission rate")
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 2]
        _scatter(ax, x_ins, resid_ins, colors[1])
        ax.set_title(_block_title("Instructed strip RESIDUALIZED", blk_i))
        ax.set_xlabel(PK_DISPLAY[pk])
        ax.set_ylabel("Emission-rate residual (prior removed)")
    savefig_paper(fig, "hero_geometry_vs_residual_grid", dir=fig_dir)
    plt.close(fig)


def fig_leaderboard(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    estimands = [
        ("rho_raw", "ci95_raw", "Raw"),
        ("rho_resid", "ci95_resid", "Residual"),
        ("rho_fe", "ci95_fe", "Bystander FE"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cohort in zip(axes, COHORT_NAMES, strict=True):
        suite = results["cohorts"][cohort]
        xpos = np.arange(len(ALL_PKS))
        width = 0.26
        for k, (rkey, ckey, lbl) in enumerate(estimands):
            vals = [suite["predictors"][pk][rkey] for pk in ALL_PKS]
            lows = [suite["predictors"][pk][ckey]["low"] for pk in ALL_PKS]
            highs = [suite["predictors"][pk][ckey]["high"] for pk in ALL_PKS]
            err = [
                [v - lo for v, lo in zip(vals, lows, strict=True)],
                [hi - v for v, hi in zip(vals, highs, strict=True)],
            ]
            ax.bar(
                xpos + (k - 1) * width,
                vals,
                width,
                yerr=err,
                capsize=2.5,
                color=colors[k],
                label=lbl,
            )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [PK_DISPLAY[pk].replace(" distance", "").replace(" similarity", "") for pk in ALL_PKS],
            fontsize=8,
        )
        ax.set_title(f"{COHORT_DISPLAY[cohort]} (n={suite['n']})")
    axes[0].set_ylabel("Spearman ρ vs ※ emission")  # noqa: RUF001
    axes[0].legend(fontsize=8)
    savefig_paper(fig, "explore_leaderboard_raw_vs_resid", dir=fig_dir)
    plt.close(fig)


def fig_forest(results: dict, fig_dir: Path) -> None:
    forest = results["per_bystander_forest"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.5), sharey=True)
    colors = paper_palette(2)
    rows0 = forest[ALL_PKS[0]]
    labels = [r["display"] for r in rows0]
    ypos = np.arange(len(labels))
    for ax, pk in zip(axes, ALL_PKS, strict=True):
        rows = forest[pk]
        for yi, r in zip(ypos, rows, strict=True):
            if r["rho"] is None:
                continue
            c = colors[0] if r["kind"] == "ordinary" else colors[1]
            ax.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color=c, lw=1.2)
            ax.plot(r["rho"], yi, "o", ms=4, color=c)
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_title(PK_DISPLAY[pk], fontsize=10)
        ax.set_xlabel("Within-context ρ (sources)")  # noqa: RUF001
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Within-bystander rank correlation, geometry vs ※ emission "
        "(blue = ordinary context, orange = instructed; missing rows: constant emission)",
        fontsize=9,
    )
    savefig_paper(fig, "explore_per_bystander_forest", dir=fig_dir)
    plt.close(fig)


def fig_tie_diagnostics(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    suite = results["cohorts"]["ordinary_cross"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xpos = np.arange(len(ALL_PKS))
    width = 0.26
    series = [
        ("rho_resid", "Headline (all cells)"),
        ("rho_binarized", "Binarized (emission > 0)"),
        ("rho_nonzero_subset", "Nonzero subset"),
    ]
    for k, (key, lbl) in enumerate(series):
        vals = []
        for pk in ALL_PKS:
            blk = suite["predictors"][pk]
            v = blk[key] if key == "rho_resid" else blk["tie_diagnostics"][key]
            vals.append(np.nan if v is None else v)
        ax.bar(xpos + (k - 1) * width, vals, width, color=colors[k], label=lbl)
    td = suite["predictors"]["cosine"]["tie_diagnostics"]
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([PK_DISPLAY[pk] for pk in ALL_PKS], fontsize=8)
    ax.set_ylabel("Spearman ρ")  # noqa: RUF001
    ax.set_title(
        f"Ordinary cross-context tie structure: {td['n_zero_dv']}/{td['n']} cells at zero "
        f"emission, {td['n_unique_dv']} unique values"
    )
    ax.legend(fontsize=8)
    savefig_paper(fig, "explore_tie_diagnostics", dir=fig_dir)
    plt.close(fig)


def fig_cluster_ci(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    variants = [
        ("ci95_resid", "Naive (cell)"),
        ("ci95_cluster_bystander", "Bystander cluster"),
        ("ci95_cluster_source", "Source cluster"),
    ]
    for ax, cohort in zip(axes, COHORT_NAMES, strict=True):
        suite = results["cohorts"][cohort]
        xpos = np.arange(len(ALL_PKS))
        for k, (key, lbl) in enumerate(variants):
            offs = (k - 1) * 0.22
            for xi, pk in zip(xpos, ALL_PKS, strict=True):
                blk = suite["predictors"][pk]
                ci = blk[key]
                ax.plot(
                    [xi + offs] * 2,
                    [ci["low"], ci["high"]],
                    color=colors[k],
                    lw=1.6,
                    label=lbl if xi == 0 else None,
                )
                ax.plot(xi + offs, blk["rho_resid"], "o", ms=4, color=colors[k])
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels([PK_DISPLAY[pk].split(" ")[0] for pk in ALL_PKS], fontsize=8)
        ax.set_title(f"{COHORT_DISPLAY[cohort]} (n={suite['n']})")
    axes[0].set_ylabel("Residual ρ with 95% CI")  # noqa: RUF001
    axes[0].legend(fontsize=8)
    savefig_paper(fig, "explore_cluster_ci_comparison", dir=fig_dir)
    plt.close(fig)


def fig_nonstylized(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    rob = results["robustness"]
    panels = [
        (
            "ordinary_cross",
            [
                ("All cells", results["cohorts"]["ordinary_cross"]),
                ("Drop stylized sources", rob["nonstylized"]["ordinary_cross"]),
                ("Drop stylized both sides", rob["nonstylized_strict"]["ordinary_cross"]),
            ],
        ),
        (
            "instructed_strip",
            [
                ("All cells", results["cohorts"]["instructed_strip"]),
                ("Drop stylized sources", rob["nonstylized"]["instructed_strip"]),
            ],
        ),
    ]
    for ax, (cohort, variants) in zip(axes, panels, strict=True):
        xpos = np.arange(len(ALL_PKS))
        width = 0.8 / len(variants)
        for k, (lbl, suite) in enumerate(variants):
            vals = [suite["predictors"][pk]["rho_resid"] for pk in ALL_PKS]
            lows = [suite["predictors"][pk]["ci95_resid"]["low"] for pk in ALL_PKS]
            highs = [suite["predictors"][pk]["ci95_resid"]["high"] for pk in ALL_PKS]
            err = [
                [v - lo for v, lo in zip(vals, lows, strict=True)],
                [hi - v for v, hi in zip(vals, highs, strict=True)],
            ]
            ax.bar(
                xpos + (k - (len(variants) - 1) / 2) * width,
                vals,
                width,
                yerr=err,
                capsize=2.5,
                color=colors[k],
                label=f"{lbl} (n={suite['n']})",
            )
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels([PK_DISPLAY[pk].split(" ")[0] for pk in ALL_PKS], fontsize=8)
        ax.set_title(COHORT_DISPLAY[cohort])
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Residual ρ")  # noqa: RUF001
    savefig_paper(fig, "explore_nonstylized_robustness", dir=fig_dir)
    plt.close(fig)


def fig_class_letter(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    variants = [
        ("Off-diagonal ordinary", results["cohorts"]["ordinary_cross"]),
        ("Different class letter", results["robustness"]["class_letter_cross"]),
    ]
    xpos = np.arange(len(ALL_PKS))
    width = 0.36
    for k, (lbl, suite) in enumerate(variants):
        vals = [suite["predictors"][pk]["rho_resid"] for pk in ALL_PKS]
        lows = [suite["predictors"][pk]["ci95_resid"]["low"] for pk in ALL_PKS]
        highs = [suite["predictors"][pk]["ci95_resid"]["high"] for pk in ALL_PKS]
        err = [
            [v - lo for v, lo in zip(vals, lows, strict=True)],
            [hi - v for v, hi in zip(vals, highs, strict=True)],
        ]
        ax.bar(
            xpos + (k - 0.5) * width,
            vals,
            width,
            yerr=err,
            capsize=2.5,
            color=colors[k],
            label=f"{lbl} (n={suite['n']})",
        )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([PK_DISPLAY[pk] for pk in ALL_PKS], fontsize=8)
    ax.set_ylabel("Residual ρ")  # noqa: RUF001
    ax.set_title("Cohort-definition robustness: two readings of “cross-context”")
    ax.legend(fontsize=8)
    savefig_paper(fig, "explore_class_letter_cross", dir=fig_dir)
    plt.close(fig)


def fig_dvb(panel: dict, masks: dict, results: dict, fig_dir: Path) -> None:
    colors = paper_palette(2)
    m = masks["ordinary_cross"]
    y = panel["extra_marker_logp"][m]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pk, c in zip(axes, PRIMARY_PKS, colors, strict=True):
        blk = results["robustness"]["dvB_ordinary"]["predictors"][pk]
        x = panel[pk][m]
        _scatter(ax, x, y, c)
        ci = blk["ci95_resid"]
        ax.set_title(
            f"Ordinary cross-context (n={blk['tie_diagnostics']['n']})\n"
            f"ρ={blk['rho_resid']:+.2f} [{ci['low']:+.2f}, {ci['high']:+.2f}]"  # noqa: RUF001
        )
        ax.set_xlabel(PK_DISPLAY[pk])
        ax.set_ylabel("Appended-slot log P(※)")
    fig.suptitle("Graded secondary DV (tie-free) on the ordinary cohort", fontsize=10)
    savefig_paper(fig, "explore_dvB_logp_ordinary", dir=fig_dir)
    plt.close(fig)


def fig_source_dose(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(3)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    suite = results["cohorts"]["ordinary_cross"]
    for col, pk in enumerate(PRIMARY_PKS):
        sm = suite["source_marginal"][pk]
        geoms = [v["mean_geometry"] for v in sm["per_source"].values()]
        emits = [v["mean_emission"] for v in sm["per_source"].values()]
        ax = axes[0, col]
        _scatter(ax, np.array(geoms), np.array(emits), colors[0])
        ax.set_title(
            f"Source marginals, ordinary cross-context (n={sm['n_sources']} sources)\n"
            f"ρ={sm['rho']:+.2f}"  # noqa: RUF001
        )
        ax.set_xlabel(f"Row-mean {PK_DISPLAY[pk]}")
        ax.set_ylabel("Row-mean ※ emission rate")

        ax = axes[1, col]
        blk = suite["predictors"][pk]
        names = ["Pooled residual", "Two-way FE", "Partial (source dose)"]
        vals = [blk["rho_resid"], blk["rho_twoway"], blk["rho_partial_source_dose"]]
        ax.bar(np.arange(3), vals, 0.55, color=colors)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Spearman ρ")  # noqa: RUF001
        ax.set_title(f"{PK_DISPLAY[pk]}: dose-confound controls")
    savefig_paper(fig, "explore_source_dose_confound", dir=fig_dir)
    plt.close(fig)


def fig_delta_rho(results: dict, fig_dir: Path) -> None:
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for k, pk in enumerate(PRIMARY_PKS):
        d = results["delta_rho"][pk]
        ax.plot([k, k], [d["ci95_low"], d["ci95_high"]], color=colors[k], lw=2)
        ax.plot(k, d["delta_rho"], "o", ms=6, color=colors[k])
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(PRIMARY_PKS)))
    ax.set_xticklabels([PK_DISPLAY[pk] for pk in PRIMARY_PKS], fontsize=9)
    ax.set_ylabel("Δρ (ordinary - instructed) with 95% CI")
    ax.set_title("Between-cohort contrast of the residual rank correlation")
    savefig_paper(fig, "explore_delta_rho_contrast", dir=fig_dir)
    plt.close(fig)


def make_figures(panel: dict, masks: dict, results: dict, fig_dir: Path) -> None:
    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_hero(panel, masks, results, fig_dir)
    fig_leaderboard(results, fig_dir)
    fig_forest(results, fig_dir)
    fig_tie_diagnostics(results, fig_dir)
    fig_cluster_ci(results, fig_dir)
    fig_nonstylized(results, fig_dir)
    fig_class_letter(results, fig_dir)
    fig_dvb(panel, masks, results, fig_dir)
    fig_source_dose(results, fig_dir)
    fig_delta_rho(results, fig_dir)
    print(f"[figures] wrote 10 figure sets to {fig_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _assert_n(label: str, suite: dict, expected: int) -> None:
    if suite["n"] != expected:
        raise RuntimeError(f"Robustness slice {label}: expected n={expected}, got n={suite['n']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task #539: per-cohort geometry-on-residual re-read of the #532 panel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in-dir", type=Path, default=Path("eval_results/issue_532"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_539"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/issue_539"))
    parser.add_argument("--n-perm", type=int, default=10_000, dest="n_perm")
    parser.add_argument("--n-boot", type=int, default=10_000, dest="n_boot")
    parser.add_argument("--n-cluster-boot", type=int, default=2_000, dest="n_cluster_boot")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = datetime.now(UTC)

    panel = build_panel(args.in_dir)
    step0 = step0_consistency(panel, args.in_dir)  # sys.exit(1) on mismatch
    masks = cohort_masks(panel)

    # Fast-Spearman equivalence guard: the vectorized resampling paths must
    # agree with the vendored scipy implementation on the real data.
    for pk in ALL_PKS:
        fast = _fast_spearman(panel[pk].astype(np.float64), panel["emit_rate"].astype(np.float64))
        ref = _spearman_rho(panel[pk].astype(np.float64), panel["emit_rate"].astype(np.float64))
        assert abs(fast - ref) < 1e-9, f"fast-Spearman drift on {pk}: {fast!r} vs scipy {ref!r}"

    print("[compute] primary cohorts ...")
    cohorts = {name: compute_cohort_suite(panel, masks[name], args) for name in COHORT_NAMES}

    # Holm over the 4 primary tests (plan D6): {cosine, gauss_kl} x {2 cohorts}.
    family = [(pk, cohort) for pk in PRIMARY_PKS for cohort in COHORT_NAMES]
    raw_p = [cohorts[cohort]["predictors"][pk]["p_perm_resid"]["p"] for pk, cohort in family]
    adj_p = holm_adjust(raw_p)
    holm = {
        "family": [
            {"predictor": pk, "cohort": cohort, "p_raw": p, "p_holm": pa}
            for (pk, cohort), p, pa in zip(family, raw_p, adj_p, strict=True)
        ],
        "n_tests": len(family),
    }

    print("[compute] delta-rho contrast ...")
    delta_rho = compute_delta_rho(panel, masks, args)

    print("[compute] robustness slices ...")
    robustness = {
        "nonstylized": {
            "ordinary_cross": compute_cohort_suite(
                panel, masks["nonstylized_ordinary_cross"], args
            ),
            "instructed_strip": compute_cohort_suite(
                panel, masks["nonstylized_instructed_strip"], args
            ),
            "dropped_sources": list(STYLIZED_SOURCES),
        },
        "nonstylized_strict": {
            "ordinary_cross": compute_cohort_suite(
                panel, masks["nonstylized_strict_ordinary_cross"], args
            ),
            "note": "#502 both-sides convention: pairs touching A3/A4/A5 on EITHER side "
            "dropped; instructed strip omitted (stylized bystanders only exist on the "
            "ordinary side, so strict == nonstylized there)",
        },
        "class_letter_cross": compute_cohort_suite(panel, masks["class_letter_cross"], args),
        "dvB_ordinary": compute_cohort_suite(
            panel, masks["ordinary_cross"], args, dv_key="extra_marker_logp"
        ),
    }
    _assert_n("nonstylized/ordinary_cross", robustness["nonstylized"]["ordinary_cross"], 195)
    _assert_n("nonstylized/instructed_strip", robustness["nonstylized"]["instructed_strip"], 130)
    _assert_n(
        "nonstylized_strict/ordinary_cross", robustness["nonstylized_strict"]["ordinary_cross"], 156
    )
    _assert_n("class_letter_cross", robustness["class_letter_cross"], 180)
    _assert_n("dvB_ordinary", robustness["dvB_ordinary"], 240)

    print("[compute] per-bystander forest ...")
    forest = compute_forest(panel, masks, args)

    results = {
        "metadata": {
            "task": 539,
            "git_commit": _git_commit(),
            "timestamp_utc": t0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
            "seed": args.seed,
            "n_perm": args.n_perm,
            "n_boot": args.n_boot,
            "n_cluster_boot": args.n_cluster_boot,
            "parent_task": PARENT_TASK,
            "parent_panel_sha": PARENT_PANEL_SHA,
            "parent_analysis_git_commit": step0["parent_analysis_git_commit"],
            "vendored_functions": [
                "_spearman_rho (verbatim; all point estimates)",
                "_bootstrap_spearman_ci (estimand vendored; vectorized + degenerate drop+count)",
                "_signflip_permutation_test (permutation arm; vectorized; add-one p formula)",
                "_build_union_panel (row-building logic; emit_rate := "
                "summary.in_R_emission_rate per parent line 1827)",
            ],
            "in_dir": str(args.in_dir),
            "argv": sys.argv[1:],
            "primary_family": [{"predictor": pk, "cohort": cohort} for pk, cohort in family],
            "js_v1_status": "exploratory (deprecated estimator; outside the Holm family)",
            "twoway_fe_estimator": "dummy-regression lstsq, exact on unbalanced panels "
            "(round-2 binding fix; replaces the round-1 single-pass demean, which is the "
            "FE residual only on complete balanced rectangles)",
            "residualization_noop_flags": {
                cohort: cohorts[cohort]["residualization"]["noop"] for cohort in COHORT_NAMES
            },
        },
        "step0_consistency": step0,
        "collinearity_gate": compute_collinearity_gate(panel, masks),
        "cohorts": cohorts,
        "holm": holm,
        "delta_rho": delta_rho,
        "robustness": robustness,
        "per_bystander_forest": forest,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "residual_per_cohort.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[write] {out_path}")

    make_figures(panel, masks, results, args.fig_dir)

    wall = (datetime.now(UTC) - t0).total_seconds()
    for pk in PRIMARY_PKS:
        for cohort in COHORT_NAMES:
            blk = cohorts[cohort]["predictors"][pk]
            ci = blk["ci95_resid"]
            print(
                f"[headline] {pk:9s} {cohort:16s} rho_resid={blk['rho_resid']:+.3f} "
                f"[{ci['low']:+.3f}, {ci['high']:+.3f}] p_perm={blk['p_perm_resid']['p']:.4g}"
            )
    print(f"[done] wall={wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
