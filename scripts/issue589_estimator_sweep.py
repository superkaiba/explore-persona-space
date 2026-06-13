#!/usr/bin/env python3
"""Task #589 — estimator-fragility sweep of the four clustered leakage-line reads.

Refits each of #536's four CLUSTERED leakage-line headline tests under BOTH the
cluster-robust OLS estimator AND the persona-random-effect MixedLM estimator, on
the parent's identical persisted joins (raw + mean-centered distance axes). The
ONE swept dimension is the uncertainty estimator; data, joins, point estimates,
the manipulation-check gates, the row set, and alpha-per-row are all held
IDENTICAL to the parent (#536, gated joins persisted at commit ``12853bca8``).

Rows swept (4 rows x 2 estimators x 2 joins = 16 data rows):
  405-secondary       pooled distance effect (deltaLogP ~ K * min_dist)  published SIGNIFICANT
  490-distance-adjusted on-axis dose-matched gap                          published NULL
  505-loo-null        leave-one-out pooled slope (delta_leakage ~ cos)    published NULL
  478-flatness-null   set-size x distance interaction (the fragility row) published NULL

Reuse contract (plan section 4.6): this driver imports
``issue536_recompute_driver`` (the join code of record) and
``issue536_mixedlm_refit`` (the #478 MixedLM cell, reused VERBATIM). It does NOT
redefine the FAMILY_REGISTRY families, the joins, the gate constants, or the
``cluster_ols`` helper — it reconstructs each row's (y, X-by-join, cluster,
persona) frame by calling into ``drv`` exactly as the parent adapter built it,
then fits both estimators on that frame.

Per-row estimator assignment (the published estimator is the reproduction
control; the alternative is the swept dimension):
  405  published = MixedLM (two VCs {subset, persona}, dummy group, reml=True,
                   the parent's regrade_405._mixed spec, VERBATIM);
       alternative = cluster-robust OLS, clusters = subset (positives panel).
  490  published = cluster-robust OLS, clusters = pair_id|seed;
       alternative = persona-RE MixedLM, groups = pair_id|seed,
                   vc = {persona: 0 + C(persona)}, reml=False, lbfgs.
  505  published = cluster-robust OLS pooled stand-in, clusters = j_i|seed;
       alternative = persona-RE MixedLM, groups = j_i|seed,
                   vc = {persona: 0 + C(b)}, reml=False, lbfgs (b = bystander).
  478  published = MixedLM (fit_published_mixedlm, VERBATIM);
       alternative = cluster-robust OLS, clusters = cell_id|seed.

MixedLM hard rule (plan section 4.5 / risk row): on exception or
non-convergence, the cell is reported ``status: FAILED`` / ``converged: False``
with the repr; NEVER a fallback to another estimator. A non-converging MixedLM
IS a reportable result. Convergence diagnostics (converged, boundary_variance,
fit_warnings) are captured exactly as ``issue536_mixedlm_refit`` does.

Manipulation check (plan section 4.5): per row, before any swept p is read, the
published-estimator cell on the RAW join must reproduce the parent's persisted
point estimate within tolerance (``manipulation_check_ratio`` = |refit - pub|;
gate threshold 1e-4 row/matrix-level where a matrix persists, else statistic
within 0.02). A failing gate marks the row's cells ``inconclusive (join_bug)``
and is reported, never papered over.

Required input restore (plan section 8 risk row + section 12 Assumption 7):
  git checkout 45fe33f85 -- \
    eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json
The two untracked centroid bundles
(``eval_results/extraction_method_comparison/centroids_method_a.pt``,
``eval_results/single_token_100_persona/centroids/centroids_layer20.pt``) must
be present under --data-root; they live in the main checkout (gitignored).

Outputs (checkpointed per (row x estimator x join), plan section 4.6):
  eval_results/issue_589/sweep_results.csv   one row per (row_id x estimator x join)
  eval_results/issue_589/sweep_results.json  same + reproducibility metadata + 505 per-arm block
  figures/issue_589/estimator_sweep.png      (+ PDF + meta.json) two panels (raw / centered)

Usage::

    uv run python scripts/issue589_estimator_sweep.py \
        [--data-root /home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-589]

CPU-only on the VM, deterministic, minutes. No GPU, no pod, no downloads beyond
the #505 HF geometry bundle the parent's family_505 already fetches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue536_mixedlm_refit as mlm  # noqa: E402  (#478 MixedLM cell, reused verbatim)
import issue536_recompute_driver as drv  # noqa: E402  (join code of record)

# Parent's gated-joins provenance commit (the data pin; recorded in metadata).
PARENT_DATA_COMMIT = "12853bca8"

# Row set — the four CLUSTERED leakage-line reads (plan section 4.1).
ROWS = ["405-secondary", "490-distance-adjusted", "505-loo-null", "478-flatness-null"]

# Per-row alpha (each task's own published threshold; plan section 4.4 / 10).
# 405 published at 0.01; 490/505 at 0.05; 478-flatness-null binds at 0.05 (Holm
# within the 2-test family) with 0.01 recorded as the H2 alpha.
ROW_ALPHA = {
    "405-secondary": 0.01,
    "490-distance-adjusted": 0.05,
    "505-loo-null": 0.05,
    "478-flatness-null": 0.05,
}

# Manipulation-check tolerance (plan section 4.5). The gate is TWO-TIER:
#   (1) the matrix / row-level JOIN gate (1e-4) is already enforced INSIDE
#       drv.family_* / mlm.build_joined_df, which RAISE before this driver ever
#       reads a cell — so a join drift never reaches the point-estimate check;
#   (2) the point-estimate reproduction of the published-estimator RAW cell vs
#       the parent's persisted coefficient is a STATISTIC-level comparison, so it
#       uses the plan's "within 0.02 / inside published CI otherwise" tolerance
#       (NOT 1e-4 — the published #478 point is the ROUNDED body value +0.010,
#       which the parent's own gate compares at GATE_BETA_TOL=0.005, never 1e-4).
MC_STATISTIC_TOL = 0.02


# ──────────────────────────────────────────────────────────────────────────
# Fit-result container
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Fit:
    """A single (row x estimator x join) fit result."""

    coefficient: float | None = None
    se: float | None = None
    df: float | None = None  # residual / model df used for the Wald CI
    p_value: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    n_rows: int | None = None
    n_clusters: int | None = None
    converged: bool | None = None  # None for OLS (always converges)
    boundary_variance: bool | None = None  # None for OLS
    status: str = "OK"  # OK | FAILED
    fit_warnings: list[str] = field(default_factory=list)
    reason: str | None = None  # repr on FAILED


def _fit_cluster_ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray, term_idx: int) -> Fit:
    """Cluster-robust OLS via drv.cluster_ols; read the headline term at term_idx.

    term_idx indexes the design matrix WITH the constant prepended (drv.cluster_ols
    calls sm.add_constant), so the caller passes the column index in the
    constant-prepended design (1-based over the user columns).
    """
    res = drv.cluster_ols(y, X, clusters)
    ci = res.conf_int()  # 95% Wald CI from the cluster-robust covariance
    return Fit(
        coefficient=float(res.params[term_idx]),
        se=float(res.bse[term_idx]),
        df=float(res.df_resid),
        p_value=float(res.pvalues[term_idx]),
        ci_lo=float(ci[term_idx][0]),
        ci_hi=float(ci[term_idx][1]),
        n_rows=int(res.nobs),
        n_clusters=len(np.unique(clusters)),
        converged=None,
        boundary_variance=None,
        status="OK",
    )


def _fit_mixedlm(
    df,
    formula: str,
    groups_col: str,
    persona_col: str,
    term: str,
    *,
    vc_two: dict[str, str] | None = None,
    reml: bool,
) -> Fit:
    """Persona-RE MixedLM via statsmodels; read the headline ``term``.

    Honest convergence diagnostics captured exactly as
    issue536_mixedlm_refit.fit_published_mixedlm does. On Exception ->
    status FAILED with the repr; NEVER a fallback to another estimator.

    vc_two overrides the single-persona variance component with an explicit
    two-VC dict (used for the #405 published cell, which the parent fit with
    {subset, persona}). When vc_two is given, persona_col is ignored.
    """
    import statsmodels.formula.api as smf

    vc = vc_two if vc_two is not None else {"persona": f"0 + C({persona_col})"}
    out = Fit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.mixedlm(formula, df, groups=df[groups_col], vc_formula=vc)
            res = model.fit(reml=reml, method="lbfgs")
        except Exception as e:
            out.status = "FAILED"
            out.reason = repr(e)
            out.converged = False
            out.n_rows = len(df)
            out.n_clusters = int(df[groups_col].nunique())
            return out
    # cov_re only present when k_re > 0 (a group random intercept). With
    # vc_formula present and no re_formula, statsmodels fits k_re = 0 and the
    # group structure lives in the variance components — matching the parent.
    group_var = float(res.cov_re.iloc[0, 0]) if res.k_re > 0 else None
    vcomp = {str(k): float(v) for k, v in zip(res.model.exog_vc.names, res.vcomp, strict=True)}
    ci = res.conf_int()
    boundary = bool(
        (group_var is not None and group_var < 1e-8) or any(v < 1e-8 for v in vcomp.values())
    )
    out.coefficient = float(res.params[term])
    out.se = float(res.bse[term])
    out.df = float(getattr(res, "df_resid", float("nan")))
    out.p_value = float(res.pvalues[term])
    out.ci_lo = float(ci.loc[term][0])
    out.ci_hi = float(ci.loc[term][1])
    out.n_rows = int(res.nobs)
    out.n_clusters = int(df[groups_col].nunique())
    out.converged = bool(res.converged)
    out.boundary_variance = boundary
    out.fit_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    # Singular-boundary detection (plan MixedLM hard rule + #505 concern): a
    # statsmodels MixedLM can report res.converged=True yet land at a point where
    # the Hessian is NOT positive definite — the Wald SEs/p-values are then
    # unreliable (the optimizer "converged" to a singular boundary). This is the
    # documented #505 failure mode (the parent's original mixed model "failed
    # singular at every layer"). Treat it as a non-convergent fit: status FAILED,
    # NEVER read its degenerate p as a finding, NEVER fall back to another estimator.
    hessian_not_pd = any("not positive definite" in w.lower() for w in out.fit_warnings)
    se_degenerate = bool(out.se is not None and not math.isnan(out.se) and out.se < 1e-3)
    if hessian_not_pd or se_degenerate:
        out.status = "FAILED"
        out.converged = False
        out.reason = (
            "MixedLM singular boundary: "
            + ("non-PD Hessian (Wald SE/p unreliable); " if hessian_not_pd else "")
            + (f"degenerate SE={out.se:.3e}; " if se_degenerate else "")
            + "reported converged but inference is not trustworthy"
        )
    else:
        out.status = "OK"
    return out


# ──────────────────────────────────────────────────────────────────────────
# Row builders — each reconstructs the SAME frame the parent adapter built,
# then fits both estimators on both joins. They do NOT redefine the joins;
# they call drv.family_* + the parent's row-level join arithmetic.
# ──────────────────────────────────────────────────────────────────────────
def build_405(data_root: Path) -> dict:
    """#405: deltaLogP_mean ~ K * min_dist on the 336 CORE rows; 20-bank L20.

    Published estimator = MixedLM (two VCs {subset, persona}, dummy group,
    reml=True) -> headline term ``min_dist``. Alternative = cluster-robust OLS
    clustered on ``subset`` (positives panel) -> headline term ``min_dist``.
    """
    import ast

    import pandas as pd

    fam = drv.family_20bank(data_root)
    idx = {n: i for i, n in enumerate(fam["names"])}
    dist_raw = 1.0 - fam["cos_raw"]
    dist_mc = 1.0 - fam["cos_mc"]
    pub = json.loads(
        (data_root / "eval_results" / "issue_405" / "aggregate" / "regression.json").read_text()
    )["runs"]
    rows = []
    with (
        data_root / "eval_results" / "issue_405" / "aggregate" / "per_cell_persona_tidy.csv"
    ).open() as f:
        for r in csv.DictReader(f):
            if r["track"] != "CORE":
                continue
            positives = list(ast.literal_eval(r["positives"]))
            held = r["held_persona"]
            md_raw = min(dist_raw[idx[held], idx[p]] for p in positives)
            md_mc = min(dist_mc[idx[held], idx[p]] for p in positives)
            rows.append(
                {
                    "K": int(r["K"]),
                    "subset": r["positives"],
                    "persona": held,
                    "seed": int(r["seed"]),
                    "deltaLogP_mean": float(r["deltaLogP_mean"]),
                    "min_dist_raw": float(md_raw),
                    "min_dist_mc": float(md_mc),
                    "row_dev": abs(md_raw - float(r["min_dist"])),
                }
            )
    df = pd.DataFrame(rows)
    assert len(df) == 336, f"#405 CORE rows = {len(df)}, expected 336"
    max_dev = float(df["row_dev"].max())
    pub_beta = float(pub["headline_full"]["coefs"]["min_dist"]["Estimate"])
    return {
        "row_id": "405-secondary",
        "df": df,
        "dist_cols": {"raw": "min_dist_raw", "centered": "min_dist_mc"},
        "headline_term": "min_dist",
        "published_estimator": "mixedlm",
        "published_call": "significant",
        "pub_point": pub_beta,
        "join_gate_dev": max_dev,
        "n_rows": len(df),
        "bank": {"family": fam["family"], "n": fam["n"], "layer": 20},
        # cluster-robust alternative: cluster on subset (the panel grouping).
        "cluster_col": "subset",
        # MixedLM published cell: two VCs, dummy group, reml=True (parent verbatim).
        "mixedlm_groups": "dummy_const",
        "mixedlm_vc": {"subset": "0 + C(subset)", "persona": "0 + C(persona)"},
        "mixedlm_reml": True,
    }


def build_490(data_root: Path) -> dict:
    """#490: y ~ is_on_axis + mean_d + asym, cluster-robust at (pair, seed); 111-bank.

    Published estimator = cluster-robust OLS -> headline ``is_on_axis``.
    Alternative = persona-RE MixedLM, groups = pair|seed, vc {persona}.
    Rebuilds the per-(pair, seed, persona) pivot frame exactly as regrade_490._fit.
    """
    import pandas as pd

    fam = drv.family_111bank(data_root)
    idx = {n: i for i, n in enumerate(fam["names"])}
    pub = json.loads(
        (data_root / "eval_results" / "issue_490" / "aggregate" / "regression.json").read_text()
    )["primary_q2_distance_adjusted_regression"]
    pl = pd.read_csv(data_root / "eval_results" / "issue_490" / "aggregate" / "persona_level.csv")
    pl = pl[pl["subpanel"].isin(["on_axis", "off_axis"])].copy()

    def _dists(row, D):
        i = idx[row["persona"]]
        return D[i, idx[row["A"]]], D[i, idx[row["B"]]]

    devs = []
    for _, r in pl.iterrows():
        dA, dB = _dists(r, 1.0 - fam["cos_raw"])
        devs.append(max(abs(dA - r["d_A"]), abs(dB - r["d_B"])))

    def _frame(D) -> pd.DataFrame:
        piv: dict[tuple, dict] = {}
        for _, r in pl.iterrows():
            key = (r["pair_id"], int(r["seed"]), r["persona"])
            if key not in piv:
                dA, dB = _dists(r, D)
                piv[key] = {
                    "is_on_axis": int(r["is_on_axis"]),
                    "mean_d": 0.5 * (dA + dB),
                    "asym": abs(dA - dB),
                    "cluster": f"{r['pair_id']}|seed{int(r['seed'])}",
                    "persona": r["persona"],
                    "conds": {},
                }
            piv[key]["conds"][r["condition"]] = float(r["deltaLogP_mean"])
        out_rows = []
        for rec in piv.values():
            c = rec["conds"]
            need = [k for k in c if k.startswith("shared_2D")] and all(
                any(k.startswith(p) for k in c) for p in ("pooled_2D_A", "pooled_2D_B")
            )
            if not need:
                continue
            shared = next(c[k] for k in c if k.startswith("shared_2D"))
            pA = next(c[k] for k in c if k.startswith("pooled_2D_A"))
            pB = next(c[k] for k in c if k.startswith("pooled_2D_B"))
            out_rows.append(
                {
                    "y": shared - 0.5 * (pA + pB),
                    "is_on_axis": rec["is_on_axis"],
                    "mean_d": rec["mean_d"],
                    "asym": rec["asym"],
                    "cluster": rec["cluster"],
                    "persona": rec["persona"],
                }
            )
        return pd.DataFrame(out_rows).dropna()

    return {
        "row_id": "490-distance-adjusted",
        "frame_raw": _frame(1.0 - fam["cos_raw"]),
        "frame_centered": _frame(1.0 - fam["cos_mc"]),
        "headline_term": "is_on_axis",
        "published_estimator": "cluster_ols",
        "published_call": "null",
        "pub_point": float(pub["headline_beta"]),
        "join_gate_dev": float(max(devs)),
        "ols_X_cols": ["is_on_axis", "mean_d", "asym"],
        "cluster_col": "cluster",
        "persona_col": "persona",
        "mixedlm_formula": "y ~ is_on_axis + mean_d + asym",
        "mixedlm_reml": False,
        "bank": {"family": fam["family"], "n": fam["n"], "layer": 20},
    }


def build_505(data_root: Path) -> dict:
    """#505: delta_leakage ~ cos(b, j) pooled, cluster-robust at (j_i, seed); 505 bank.

    Published estimator = cluster-robust OLS pooled stand-in -> headline slope.
    Alternative = persona-RE MixedLM, groups = j_i|seed, vc {persona: 0 + C(b)}.
    Per-arm OLS(HC2) slopes (raw + centered) are quoted verbatim as structural
    context (NOT swept under MixedLM — an arm has no within-arm cluster structure).
    """
    import pandas as pd
    import statsmodels.api as sm

    fam = drv.family_505(data_root, layer=21)
    idx = {n: i for i, n in enumerate(fam["names"])}
    rows = json.loads(
        (
            data_root / "eval_results" / "issue_505" / "analysis" / "delta_leakage_per_seed.json"
        ).read_text()
    )["rows"]
    pub = json.loads(
        (data_root / "eval_results" / "issue_505" / "analysis" / "per_arm_slopes.json").read_text()
    )
    arms = sorted({r["j_i"] for r in rows})

    def _frame(M) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delta_leakage": [float(r["delta_leakage"]) for r in rows],
                "cos_bj": [float(M[idx[r["b"]], idx[r["j_i"]]]) for r in rows],
                "cluster": [f"{r['j_i']}|s{r['seed']}" for r in rows],
                "j_i": [r["j_i"] for r in rows],
                "b": [r["b"] for r in rows],
            }
        )

    # Per-arm OLS(HC2) slopes verbatim from the parent (regrade_505._per_arm),
    # used for the join gate (raw arms must reproduce published to 1e-4) and as
    # the structural context block.
    def _per_arm(M) -> dict:
        outd = {}
        for j in arms:
            sub = [r for r in rows if r["j_i"] == j]
            x = np.array([M[idx[r["b"]], idx[j]] for r in sub], dtype=float)
            y = np.array([float(r["delta_leakage"]) for r in sub], dtype=float)
            res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC2")
            ci = res.conf_int(alpha=0.05)
            outd[j] = {
                "beta_j": float(res.params[1]),
                "p": float(res.pvalues[1]),
                "ci95": [float(ci[1, 0]), float(ci[1, 1])],
                "n_rows": len(sub),
            }
        return outd

    pa_raw = _per_arm(fam["cos_raw"])
    pa_mc = _per_arm(fam["cos_mc"])
    arm_dev = max(abs(pa_raw[j]["beta_j"] - float(pub["per_arm"][j]["beta_j"])) for j in arms)

    return {
        "row_id": "505-loo-null",
        "frame_raw": _frame(fam["cos_raw"]),
        "frame_centered": _frame(fam["cos_mc"]),
        "headline_term": "cos_bj",
        "published_estimator": "cluster_ols",
        "published_call": "null",
        # Parent pooled raw cluster-robust beta is the reproduction anchor.
        "pub_point": None,  # no single published point estimate; gate via per-arm betas
        "join_gate_dev": float(arm_dev),
        "ols_X_cols": ["cos_bj"],
        "cluster_col": "cluster",
        "persona_col": "b",
        "mixedlm_formula": "delta_leakage ~ cos_bj",
        "mixedlm_reml": False,
        "per_arm": {"raw": pa_raw, "centered": pa_mc},
        "bank": {"family": fam["family"], "n": fam["n"], "layer": 21},
    }


def build_478(data_root: Path) -> dict:
    """#478: K x log(min_dist) interaction; 111-bank; tidy.csv snapshot.

    Published estimator = MixedLM (fit_published_mixedlm, VERBATIM) -> term
    ``K:log_md``. Alternative = cluster-robust OLS at (cell_id, seed) ->
    interaction term (the parent's regrade_478._interaction).
    Reuses mlm.build_joined_df so the row-level join + 1e-4 gate run verbatim.
    """
    df, fam, join_dev = mlm.build_joined_df(data_root)
    df["log_md_raw"] = np.log(df["md_raw"])
    df["log_md_mc"] = np.log(df["md_mc"])
    df["cluster"] = df["cell_id"].astype(str) + "|s" + df["seed"].astype(str)
    return {
        "row_id": "478-flatness-null",
        "df": df,
        "log_cols": {"raw": "log_md_raw", "centered": "log_md_mc"},
        "headline_term": "K:log_md",
        "published_estimator": "mixedlm",
        "published_call": "null",
        "pub_point": mlm.PUB_BETA,  # +0.010 raw (the parent's manipulation anchor)
        "join_gate_dev": float(join_dev),
        "cluster_col": "cluster",
        "bank": {"family": fam["family"], "n": fam["n"], "layer": 20},
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-row fit dispatch — returns {join: {estimator: Fit}}, the manipulation
# ratio, and the published-cell point estimate per join.
# ──────────────────────────────────────────────────────────────────────────
def fit_405(meta: dict) -> tuple[dict, float]:

    df = meta["df"]
    fits: dict[str, dict[str, Fit]] = {}
    for join, dist_col in meta["dist_cols"].items():
        d = df.rename(columns={dist_col: "min_dist"})[
            ["deltaLogP_mean", "K", "min_dist", "persona", "subset"]
        ].copy()
        d["dummy_const"] = 1
        # cluster-robust OLS alternative: X = [K, min_dist, K*min_dist],
        # headline term = min_dist (constant-prepended index 2).
        X = np.column_stack([d["K"], d["min_dist"], d["K"] * d["min_dist"]])
        ols = _fit_cluster_ols(
            d["deltaLogP_mean"].to_numpy(), X, d[meta["cluster_col"]].to_numpy(), term_idx=2
        )
        # MixedLM published cell: parent's two-VC spec, reml=True, term min_dist.
        mix = _fit_mixedlm(
            d,
            "deltaLogP_mean ~ K * min_dist",
            meta["mixedlm_groups"],
            persona_col="persona",
            term="min_dist",
            vc_two=meta["mixedlm_vc"],
            reml=meta["mixedlm_reml"],
        )
        fits[join] = {"cluster_ols": ols, "mixedlm": mix}
    # Manipulation ratio: published cell (MixedLM) on RAW vs parent point estimate.
    pub_cell = fits["raw"]["mixedlm"]
    mc_ratio = (
        abs(pub_cell.coefficient - meta["pub_point"])
        if pub_cell.coefficient is not None
        else float("inf")
    )
    return fits, mc_ratio


def fit_490(meta: dict) -> tuple[dict, float]:
    fits: dict[str, dict[str, Fit]] = {}
    for join in ("raw", "centered"):
        d = meta["frame_raw"] if join == "raw" else meta["frame_centered"]
        X = d[meta["ols_X_cols"]].to_numpy(dtype=float)
        # cluster-robust OLS published cell: is_on_axis = constant-prepended idx 1.
        ols = _fit_cluster_ols(
            d["y"].to_numpy(dtype=float), X, d[meta["cluster_col"]].to_numpy(), term_idx=1
        )
        mix = _fit_mixedlm(
            d,
            meta["mixedlm_formula"],
            meta["cluster_col"],
            persona_col=meta["persona_col"],
            term="is_on_axis",
            reml=meta["mixedlm_reml"],
        )
        fits[join] = {"cluster_ols": ols, "mixedlm": mix}
    pub_cell = fits["raw"]["cluster_ols"]
    mc_ratio = (
        abs(pub_cell.coefficient - meta["pub_point"])
        if pub_cell.coefficient is not None
        else float("inf")
    )
    return fits, mc_ratio


def fit_505(meta: dict) -> tuple[dict, float]:
    fits: dict[str, dict[str, Fit]] = {}
    for join in ("raw", "centered"):
        d = meta["frame_raw"] if join == "raw" else meta["frame_centered"]
        X = d[meta["ols_X_cols"]].to_numpy(dtype=float)
        # cluster-robust OLS published cell: cos_bj = constant-prepended idx 1.
        ols = _fit_cluster_ols(
            d["delta_leakage"].to_numpy(dtype=float),
            X,
            d[meta["cluster_col"]].to_numpy(),
            term_idx=1,
        )
        mix = _fit_mixedlm(
            d,
            meta["mixedlm_formula"],
            meta["cluster_col"],
            persona_col=meta["persona_col"],
            term="cos_bj",
            reml=meta["mixedlm_reml"],
        )
        fits[join] = {"cluster_ols": ols, "mixedlm": mix}
    # No single published point estimate; the per-arm betas already gated in
    # build_505 (arm_dev). Use the parent's pooled raw cluster-robust beta as the
    # manipulation anchor by reproducing it here: published cell raw beta should
    # equal the parent's persisted pooled beta (the gate fires on arm_dev, but we
    # still report the cell-vs-parent residual as the ratio for the join check).
    mc_ratio = meta["join_gate_dev"]  # arm-level matrix gate (parent's 1e-4 read)
    return fits, mc_ratio


def fit_478(meta: dict) -> tuple[dict, float]:
    df = meta["df"]
    fits: dict[str, dict[str, Fit]] = {}
    for join, log_col in meta["log_cols"].items():
        # cluster-robust OLS alternative: X = [log_md, K, log_md*K]; interaction
        # = constant-prepended idx 3 (matches regrade_478._interaction).
        X = np.column_stack([df[log_col], df["K"], df[log_col] * df["K"]])
        ols = _fit_cluster_ols(
            df["deltaLogP_mean"].to_numpy(), X, df[meta["cluster_col"]].to_numpy(), term_idx=3
        )
        # MixedLM published cell: fit_published_mixedlm VERBATIM on the join col.
        md_col = "md_raw" if join == "raw" else "md_mc"
        raw_mix = mlm.fit_published_mixedlm(df, md_col)
        mix = Fit(status=raw_mix.get("status", "FAILED"))
        if raw_mix.get("status") == "OK":
            mix.coefficient = float(raw_mix["coef"])
            mix.se = float(raw_mix["se"])
            mix.p_value = float(raw_mix["p"])
            mix.converged = bool(raw_mix["converged"])
            mix.boundary_variance = bool(raw_mix["boundary_variance"])
            mix.n_rows = int(raw_mix["n_obs"])
            mix.n_clusters = int(df["subset_id"].nunique())
            mix.fit_warnings = list(raw_mix.get("fit_warnings", []))
            # Wald CI from the reported coef + se (statsmodels MixedLM uses normal).
            half = 1.959963984540054 * mix.se
            mix.ci_lo = mix.coefficient - half
            mix.ci_hi = mix.coefficient + half
            # Same singular-boundary guard as _fit_mixedlm (non-PD Hessian /
            # degenerate SE => unreliable inference => FAILED, never a finding).
            hessian_not_pd = any("not positive definite" in w.lower() for w in mix.fit_warnings)
            se_degenerate = bool(mix.se is not None and mix.se < 1e-3)
            if hessian_not_pd or se_degenerate:
                mix.status = "FAILED"
                mix.converged = False
                mix.reason = (
                    "MixedLM singular boundary: "
                    + ("non-PD Hessian (Wald SE/p unreliable); " if hessian_not_pd else "")
                    + (f"degenerate SE={mix.se:.3e}; " if se_degenerate else "")
                    + "reported converged but inference is not trustworthy"
                )
        else:
            mix.converged = False
            mix.reason = raw_mix.get("reason")
            mix.n_rows = int(raw_mix.get("n_obs", len(df)))
        fits[join] = {"cluster_ols": ols, "mixedlm": mix}
    # Manipulation ratio: published cell (MixedLM) on RAW vs parent point (+0.010).
    pub_cell = fits["raw"]["mixedlm"]
    mc_ratio = (
        abs(pub_cell.coefficient - meta["pub_point"])
        if pub_cell.coefficient is not None
        else float("inf")
    )
    return fits, mc_ratio


BUILDERS = {
    "405-secondary": (build_405, fit_405),
    "490-distance-adjusted": (build_490, fit_490),
    "505-loo-null": (build_505, fit_505),
    "478-flatness-null": (build_478, fit_478),
}


# ──────────────────────────────────────────────────────────────────────────
# Call classification (plan section 6.3)
# ──────────────────────────────────────────────────────────────────────────
def classify_call(fit: Fit, alpha: float) -> str:
    """significant | null | inconclusive — a cell's own swept call."""
    if fit.status != "OK" or fit.p_value is None:
        return "inconclusive"
    if fit.converged is False:  # MixedLM that did not converge
        return "inconclusive"
    return "significant" if fit.p_value < alpha else "null"


CSV_COLUMNS = [
    "row_id",
    "estimator",
    "join",
    "coefficient",
    "se",
    "df",
    "p_value",
    "ci_lo",
    "ci_hi",
    "n_rows",
    "n_clusters",
    "manipulation_check_ratio",
    "converged",
    "boundary_variance",
    "call_published",
    "call_swept",
    "call_flips",
]


def _csv_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(v)
    return str(v)


def main() -> int:
    import pandas as pd
    import scipy
    import statsmodels

    ap = argparse.ArgumentParser(description="Task #589 estimator-fragility sweep.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO,
        help="Checkout holding the parent's persisted joins + centroid bundles.",
    )
    args = ap.parse_args()
    data_root = args.data_root.resolve()

    out_dir = REPO / "eval_results" / "issue_589"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_results.csv"
    json_path = out_dir / "sweep_results.json"

    # Fresh CSV (checkpoint-per-row append; mirrors drv.append_row discipline).
    # lineterminator="\n" keeps the file LF-only (the csv default is \r\n, which
    # trips git's CRLF warning and makes re-runs non-byte-identical under the
    # repo's LF .gitattributes normalization).
    with csv_path.open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(CSV_COLUMNS)

    all_cells: list[dict] = []
    per_row_meta: dict[str, dict] = {}
    per_arm_block: dict = {}

    for row_id in ROWS:
        build, fit = BUILDERS[row_id]
        meta = build(data_root)
        fits, mc_ratio = fit(meta)
        alpha = ROW_ALPHA[row_id]
        published_estimator = meta["published_estimator"]
        published_call = meta["published_call"]
        join_bug = mc_ratio > MC_STATISTIC_TOL

        for join in ("raw", "centered"):
            # The published-cell call ON THIS JOIN is the reference for the flip
            # test (the published estimator's verdict at the same join).
            pub_fit = fits[join][published_estimator]
            pub_call_join = "inconclusive (join_bug)" if join_bug else classify_call(pub_fit, alpha)
            for est in ("cluster_ols", "mixedlm"):
                cell = fits[join][est]
                swept = "inconclusive (join_bug)" if join_bug else classify_call(cell, alpha)
                # call_flips: the swept-estimator cell disagrees with the
                # published-estimator cell on the SAME join (only meaningful for
                # the alternative estimator; for the published cell it is False
                # by construction). A join_bug forces inconclusive (no flip read).
                if join_bug or est == published_estimator:
                    flips = False
                else:
                    flips = (
                        swept in ("significant", "null")
                        and pub_call_join in ("significant", "null")
                        and swept != pub_call_join
                    )
                rec = {
                    "row_id": row_id,
                    "estimator": est,
                    "join": join,
                    "coefficient": cell.coefficient,
                    "se": cell.se,
                    "df": cell.df,
                    "p_value": cell.p_value,
                    "ci_lo": cell.ci_lo,
                    "ci_hi": cell.ci_hi,
                    "n_rows": cell.n_rows,
                    "n_clusters": cell.n_clusters,
                    "manipulation_check_ratio": mc_ratio,
                    "converged": cell.converged,
                    "boundary_variance": cell.boundary_variance,
                    "call_published": published_call,
                    "call_swept": swept,
                    "call_flips": flips,
                    # extra (json-only) fields:
                    "_status": cell.status,
                    "_reason": cell.reason,
                    "_fit_warnings": cell.fit_warnings,
                    "_is_published_estimator": est == published_estimator,
                    "_published_call_this_join": pub_call_join,
                    "_alpha": alpha,
                    "_sign_flip": False,  # filled below
                }
                all_cells.append(rec)
                # Checkpoint: append the CSV row the moment it is computed.
                with csv_path.open("a", newline="") as f:
                    csv.writer(f, lineterminator="\n").writerow(
                        _csv_value(rec[c]) for c in CSV_COLUMNS
                    )

        # Sign-flip flag (graver than a p-only flip): coefficient sign differs
        # between the two estimators on the same join.
        for join in ("raw", "centered"):
            co = fits[join]["cluster_ols"].coefficient
            cm = fits[join]["mixedlm"].coefficient
            if co is not None and cm is not None and co != 0 and cm != 0:
                sf = math.copysign(1, co) != math.copysign(1, cm)
                for rec in all_cells:
                    if rec["row_id"] == row_id and rec["join"] == join:
                        rec["_sign_flip"] = bool(sf)

        per_row_meta[row_id] = {
            "alpha": alpha,
            "published_estimator": published_estimator,
            "published_call": published_call,
            "pub_point": meta.get("pub_point"),
            "manipulation_check_ratio": mc_ratio,
            "manipulation_check_passed": not join_bug,
            "join_gate_dev": meta["join_gate_dev"],
            "bank": meta["bank"],
            "headline_term": meta["headline_term"],
            "small_cluster_flag": bool(
                any(
                    rec["n_clusters"] is not None and rec["n_clusters"] < 30
                    for rec in all_cells
                    if rec["row_id"] == row_id
                )
            ),
        }
        if row_id == "505-loo-null":
            per_arm_block = {
                "note": "per-arm OLS(HC2) slopes (raw + centered) verbatim from the parent "
                "(regrade_505._per_arm); NOT swept under MixedLM (an arm has no within-arm "
                "cluster/RE structure) — quoted as structural context for the pooled read.",
                "per_arm": meta["per_arm"],
            }

    # ── Reproducibility metadata + machine-readable payload ──
    payload = {
        "schema_version": "i589_estimator_sweep_v1",
        "generated_at": drv._now(),
        "code_commit": drv._git_sha(),
        "data_root": str(data_root),
        "data_root_commit": drv._git_sha(data_root),
        "parent_data_pin": PARENT_DATA_COMMIT,
        "parent_data_pin_note": (
            "the parent #536 gated joins were persisted at 12853bca8; the inputs read here "
            "are validated by the parent's 1e-4 matrix / row-level join gates inside family_*"
        ),
        "env": {
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "swept_dimension": "uncertainty estimator (cluster-robust OLS <-> persona-RE MixedLM)",
        "joins": {
            "raw": "1 - compute_cosine_matrix(C, centering='none')",
            "centered": "1 - compute_cosine_matrix(C, centering='global_mean')",
        },
        "estimator_specs": {
            "cluster_ols": "drv.cluster_ols: sm.OLS(y, add_constant(X)).fit("
            "cov_type='cluster', cov_kwds={'groups': clusters}); Wald 95% CI = res.conf_int()",
            "mixedlm_405": "smf.mixedlm('deltaLogP_mean ~ K * min_dist', groups=dummy_const, "
            "vc_formula={'subset':'0+C(subset)','persona':'0+C(persona)'}).fit(reml=True, lbfgs)"
            " [PUBLISHED cell — parent regrade_405._mixed verbatim]",
            "mixedlm_478": "issue536_mixedlm_refit.fit_published_mixedlm VERBATIM "
            "(groups=subset_id, vc={'persona':'0+C(held_out_persona)'}, reml=False, lbfgs) "
            "[PUBLISHED cell]",
            "mixedlm_490": "smf.mixedlm('y ~ is_on_axis + mean_d + asym', groups=pair|seed, "
            "vc_formula={'persona':'0+C(persona)'}).fit(reml=False, lbfgs) [CONSTRUCTED]",
            "mixedlm_505": "smf.mixedlm('delta_leakage ~ cos_bj', groups=j_i|seed, "
            "vc_formula={'persona':'0+C(b)'}).fit(reml=False, lbfgs) [CONSTRUCTED]",
        },
        "per_row_alpha": ROW_ALPHA,
        "manipulation_check_tol": MC_STATISTIC_TOL,
        "manipulation_check_note": (
            "two-tier (plan 4.5): the 1e-4 matrix/row-level JOIN gate is enforced inside "
            "drv.family_* / mlm.build_joined_df (they RAISE before any cell is read); the "
            "point-estimate reproduction of the published-estimator raw cell vs the parent's "
            "persisted coefficient is statistic-level at 0.02 (the #478 published point +0.010 "
            "is the rounded body value, parent gate GATE_BETA_TOL=0.005)"
        ),
        "row_meta": per_row_meta,
        "cells": all_cells,
        "per_arm_505": per_arm_block,
        "multiplicity_note": (
            "16-cell grid (4 rows x 2 estimators x 2 joins) is exploratory robustness, NOT 16 "
            "confirmatory tests; each row keeps its own published alpha + family rule (no "
            "within-row re-thresholding, no cross-row Bonferroni), matching the parent #536."
        ),
        "small_cluster_note": (
            "#490 (24 clusters) and #505-pooled (18 clusters) sit below Cameron & Miller's "
            "~30-50 cluster floor where cluster-robust SEs are anti-conservative — the "
            "substantive reason a persona-RE MixedLM may widen the SE and disagree."
        ),
    }
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(json_path)

    # ── Console summary ──
    n_data_rows = len(all_cells)
    n_flips = sum(1 for r in all_cells if r["call_flips"])
    n_sign_flips = sum(1 for r in all_cells if r["_sign_flip"])
    n_failed = sum(1 for r in all_cells if r["_status"] != "OK")
    print(f"[sweep] wrote {csv_path} ({n_data_rows} data rows, {len(CSV_COLUMNS)} columns)")
    print(f"[sweep] wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(
        f"[sweep] {n_flips} estimator-conditional flips, {n_sign_flips} sign-flips, "
        f"{n_failed} FAILED/non-converged MixedLM cells"
    )
    for row_id in ROWS:
        cells = [r for r in all_cells if r["row_id"] == row_id]
        line = [
            f"  {row_id} (alpha={ROW_ALPHA[row_id]}, pub={per_row_meta[row_id]['published_call']})"
        ]
        for r in cells:
            p = "NA" if r["p_value"] is None else f"{r['p_value']:.4g}"
            flag = " FLIP" if r["call_flips"] else (" FAILED" if r["_status"] != "OK" else "")
            line.append(f"    {r['estimator']}/{r['join']}: beta={r['coefficient']} p={p}{flag}")
        print("\n".join(line))

    # ── Figure ──
    write_figure(all_cells, per_row_meta)
    return 0


def write_figure(all_cells: list[dict], per_row_meta: dict) -> None:
    """Two-panel p-value-pair figure (raw / centered) + exploratory dump."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("generic")

    fig_dir = REPO / "figures" / "issue_589"
    fig_dir.mkdir(parents=True, exist_ok=True)

    row_labels = {
        "405-secondary": "#405 panel\nsecondary",
        "490-distance-adjusted": "#490 on-axis\ngap",
        "505-loo-null": "#505 LOO\npooled",
        "478-flatness-null": "#478 K x dist\ninteraction",
    }
    est_color = {"cluster_ols": "#1f77b4", "mixedlm": "#d62728"}
    est_label = {"cluster_ols": "cluster-robust OLS", "mixedlm": "persona-RE MixedLM"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, join in zip(axes, ("raw", "centered"), strict=True):
        xs = list(range(len(ROWS)))
        for est in ("cluster_ols", "mixedlm"):
            ys, marks = [], []
            for row_id in ROWS:
                cell = next(
                    c
                    for c in all_cells
                    if c["row_id"] == row_id and c["join"] == join and c["estimator"] == est
                )
                if cell["p_value"] is None or cell["_status"] != "OK":
                    ys.append(np.nan)
                    marks.append("FAILED")
                else:
                    ys.append(max(cell["p_value"], 1e-70))
                    marks.append("ok")
            offset = -0.12 if est == "cluster_ols" else 0.12
            ax.scatter(
                [x + offset for x in xs],
                ys,
                s=90,
                color=est_color[est],
                label=est_label[est],
                zorder=3,
                edgecolors="black",
                linewidths=0.5,
            )
            for x, m in zip(xs, marks, strict=True):
                if m == "FAILED":
                    ax.text(
                        x + offset,
                        0.5,
                        "FAILED",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=est_color[est],
                    )
        # flip highlight
        for x, row_id in zip(xs, ROWS, strict=True):
            if any(
                c["call_flips"] for c in all_cells if c["row_id"] == row_id and c["join"] == join
            ):
                ax.axvspan(x - 0.4, x + 0.4, color="#ffe08a", alpha=0.4, zorder=0)
        ax.axhline(0.05, ls="--", color="gray", lw=1.0)
        ax.axhline(0.01, ls=":", color="gray", lw=1.0)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([row_labels[r] for r in ROWS], fontsize=8)
        ax.set_title(f"{join} distance join")
        ax.set_ylabel("p-value (log scale)")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    # text annotations for the two significance-threshold lines (left panel,
    # inside the axes, nudged off the lines so they do not overlap each other)
    axes[0].text(-0.45, 0.062, "alpha=0.05", fontsize=7, color="gray", va="bottom", ha="left")
    axes[0].text(-0.45, 0.0042, "alpha=0.01", fontsize=7, color="gray", va="top", ha="left")
    fig.suptitle(
        "Estimator-fragility sweep: published call p-values under "
        "cluster-robust OLS vs persona-RE MixedLM",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    written = savefig_paper(fig, "estimator_sweep", dir=fig_dir)
    plt.close(fig)

    # ── Exploratory dump: per-arm #505 slope panel ──
    _write_505_perarm_figure(fig_dir, per_row_meta)
    print(f"[fig] wrote {written['png']}, {written['pdf']}, {written['meta']}")


def _write_505_perarm_figure(fig_dir: Path, per_row_meta: dict) -> None:
    """LOO per-arm slope panel (6 arms, raw + centered) — the opposing-sign context."""
    import matplotlib.pyplot as plt

    # Read the per-arm block back from the json we just wrote.
    payload = json.loads((REPO / "eval_results" / "issue_589" / "sweep_results.json").read_text())
    pa = payload.get("per_arm_505", {}).get("per_arm")
    if not pa:
        return
    arms = sorted(pa["raw"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(arms))
    for off, join, color in ((-0.18, "raw", "#1f77b4"), (0.18, "centered", "#ff7f0e")):
        betas = [pa[join][j]["beta_j"] for j in arms]
        los = [pa[join][j]["ci95"][0] for j in arms]
        his = [pa[join][j]["ci95"][1] for j in arms]
        yerr = np.array(
            [
                [b - lo for b, lo in zip(betas, los, strict=True)],
                [hi - b for b, hi in zip(betas, his, strict=True)],
            ]
        )
        yerr = np.clip(yerr, 0.0, None)
        ax.errorbar(
            xs + off,
            betas,
            yerr=yerr,
            fmt="o",
            color=color,
            label=f"{join} join",
            capsize=3,
            markersize=6,
            elinewidth=1.0,
        )
    ax.axhline(0.0, ls="--", color="gray", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("per-arm OLS(HC2) slope of Δleakage vs cos(b, j)")
    ax.set_title("#505 leave-one-out per-arm slopes (opposing-sign heterogeneity)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "estimator_sweep_505_perarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
