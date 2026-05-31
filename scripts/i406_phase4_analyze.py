"""Phase 4 — analysis (7 parallel regressions + per-position trajectory + K-window sweep).

Issue #406 plan v9 §4 Phase 4 (scope-reduced 2026-05-31: N=16 / 240
ordered pairs after C2-C5 drop; Class C is the C1 singleton).

Loads:
  - eval_results/issue_406/divergence/D_matrix.json
      Primary K=25-mean KL[i, j] + JS[i, j] (symmetric) + prompt-token
      lengths per (i, j) + condition metadata.
  - eval_results/issue_406/divergence/D_per_position.json (v9-NEW)
      Per-position KL trajectory (240 ordered pairs x 25 positions).
  - eval_results/issue_406/cosine/C_L{0,5,11,15,21,27}.json (x6)
      Per-layer 16x16 cosine distance matrices.
  - eval_results/issue_406/cross_eval/G_matrix.json
      16x16 transfer-rate matrix (with diagonal-sanity pre-classified).

Drops diagonal-failed conditions (G[i, i] < 0.7), builds the
(up to 240-row) DataFrame, runs analyze_predictor 7x (1 KL primary +
6 cosine descriptive; JS as free additional secondary). For each
predictor:
  - Raw + length-partial Spearman (BOTH pingouin and inline implementations,
    always reported per §3.5 concern #8).
  - Length-tercile stratified Spearman (always run per same concern).
  - Class-cluster mixed model with random intercept per class_pair.
  - Cluster-bootstrap CI (2000 resamples, seed=42 shared across predictors
    so the resample structure is paired across predictors).
  - Per-(class_i x class_j) cell rho with per-cell critical-value annotation
    (§3.5 concern #10).
  - PRIMARY ONLY: sliding-threshold curve + permutation null + leave-class-C-out
    + leave-A1-out (the latter two also re-run on each of the 6 cosine
    layers per round-2 critic Lens 5 resolution).

v9-NEW analyses:
  - per_position_trajectory: length-partial rho(D_k, G) for k = 0..24
    (500-bootstrap CI per k). Reports trajectory shape (monotone decay /
    flat / non-monotone) for analyzer narration.
  - k_window_sweep: 9 candidate windows W subset of {0..24}; mean across W
    yields D_W and length-partial Spearman rho(D_W, G). Best |rho| flagged
    with the garden-of-forking-paths caveat.

Outputs:
  - eval_results/issue_406/analysis.json (full)
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st

from explore_persona_space.experiments.i406_conditions import CONDITIONS, CONDITIONS_BY_ID

logger = logging.getLogger("i406.phase4")

DIVERG_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
DIVERG_PER_POS_PATH = Path("eval_results/issue_406/divergence/D_per_position.json")
COSINE_DIR = Path("eval_results/issue_406/cosine")
G_MATRIX_PATH = Path("eval_results/issue_406/cross_eval/G_matrix.json")
OUT_PATH = Path("eval_results/issue_406/analysis.json")

TARGET_LAYERS = [0, 5, 11, 15, 21, 27]
K_TARGET = 25
NEG_PER_CELL_CAP = 0.5  # v5 user pick at Step 2c, option (ii)
POS_RHO_THRESHOLD = 0.4
POS_P_THRESHOLD = 0.01
NEG_RHO_THRESHOLD = 0.15
NEG_CI_LOW_BOUND = -0.2
NEG_CI_HIGH_BOUND = 0.2

K_WINDOWS: list[tuple[int, int]] = [
    (0, 0),  # k=0 only (first response token)
    (0, 4),  # k=0..4 (first 5)
    (0, 9),  # k=0..9 (first 10 -- matches v7-v8 primary)
    (0, 14),  # k=0..14 (first 15)
    (0, 19),  # k=0..19 (first 20)
    (0, 24),  # k=0..24 (first 25 -- v9 primary)
    (4, 14),  # k=4..14 (middle 11)
    (9, 19),  # k=9..19 (middle 11, shifted right)
    (14, 24),  # k=14..24 (last 11)
]


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _length_partial_inline(x: pd.Series, y: pd.Series, covar: pd.Series) -> dict[str, float]:
    """Rank-then-residualize length-partial Spearman (matches #340 methodology).

    Returns {"r": float, "p": float, "n": int}.
    """
    x_rank = st.rankdata(x.to_numpy())
    y_rank = st.rankdata(y.to_numpy())
    c_rank = st.rankdata(covar.to_numpy())
    # Residualize via simple OLS slope/intercept on the ranks.
    slope_x, intercept_x, _, _, _ = st.linregress(c_rank, x_rank)
    slope_y, intercept_y, _, _, _ = st.linregress(c_rank, y_rank)
    x_resid = x_rank - (slope_x * c_rank + intercept_x)
    y_resid = y_rank - (slope_y * c_rank + intercept_y)
    res = st.pearsonr(x_resid, y_resid)
    return {"r": float(res.statistic), "p": float(res.pvalue), "n": len(x_rank)}


def _length_tercile_rhos(df: pd.DataFrame, x_col: str, y_col: str) -> list[dict]:
    """Spearman rho within each tercile of log_prompt_tokens. Always reported per §3.5 #8."""
    bins = pd.qcut(df["log_prompt_tokens"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    out = []
    for label in ["low", "mid", "high"]:
        sub = df[bins == label]
        if len(sub) < 5:
            out.append({"tercile": label, "n": len(sub), "rho": None, "p": None})
            continue
        res = st.spearmanr(sub[x_col], sub[y_col])
        out.append(
            {
                "tercile": label,
                "n": len(sub),
                "rho": float(res.correlation),
                "p": float(res.pvalue),
            }
        )
    return out


def _cluster_bootstrap_partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "G",
    covar_col: str = "log_prompt_tokens",
    n_boot: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Cluster-bootstrap by class_pair (16 cells), length-partial Spearman per resample.

    The seed is reseeded inside this helper so resamples ARE shared across
    predictor invocations — important for paired-comparison structure across
    the 7 predictors (per §3.5 concern #8 + #9).
    """
    rng = np.random.default_rng(seed)
    cell_ids = sorted(df["class_pair"].unique())
    cell_to_rows = {cell: df.index[df["class_pair"] == cell].to_numpy() for cell in cell_ids}
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        sampled_cells = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled_cells])
        sub = df.loc[rows]
        # length-partial Spearman via pingouin
        try:
            r = pg.partial_corr(data=sub, x=x_col, y=y_col, covar=[covar_col], method="spearman")
            boot_rhos[b] = float(r["r"].values[0])
        except Exception:
            boot_rhos[b] = np.nan
    return boot_rhos


def _per_cell_partials(df: pd.DataFrame, x_col: str) -> tuple[dict[str, float], dict[str, dict]]:
    """Per-(class_i x class_j) cell length-partial Spearman + per-cell critical |rho|.

    Returns (per_cell_rhos, per_cell_meta).

    Per-cell handling (singleton-C aware, 2026-05-31 scope change):
      - Cells with 0 rows (e.g. C->C when Class C is a singleton — the C1->C1
        diagonal is excluded from the analysis dataframe, leaving zero
        off-diagonal pairs) are RECORDED in ``per_cell_meta`` with
        ``status='absent'``, ``n=0``, ``rho=None``; they do NOT appear in
        ``per_cell`` (so the per-cell-max-abs aggregation skips them) and
        do NOT crash the figure code (which calls ``per_cell.get(cell)``
        and tolerates a None return).
      - Cells with 1 <= n < 5 (too few rows to fit a Spearman partial)
        are RECORDED with ``status='too_few'``, ``n=<int>``, ``rho=None``.
      - Cells with n >= 5 fit the partial and are recorded with
        ``status='ok'`` plus the computed rho/p/critical_rho_p05.

    The explicit ``status`` field replaces the previous silent drop so
    downstream consumers (figure caption / table / clean-result body)
    can render an unambiguous "n/a" rather than a misleading aggregated
    value.
    """
    per_cell: dict[str, float] = {}
    per_cell_meta: dict[str, dict] = {}
    # Enumerate ALL 16 possible (class_i, class_j) cells so absent cells
    # surface as explicit meta entries rather than missing dict keys.
    all_classes = sorted(set(df["class_i"]).union(df["class_j"]))
    grouped = {cell: sub for cell, sub in df.groupby("class_pair")}
    for ci in all_classes:
        for cj in all_classes:
            cell = f"{ci}_{cj}"
            sub = grouped.get(cell)
            n = 0 if sub is None else len(sub)
            if n == 0:
                per_cell_meta[cell] = {
                    "n": 0,
                    "rho": None,
                    "p": None,
                    "critical_rho_p05": None,
                    "status": "absent",
                }
                continue
            if n < 5:
                per_cell_meta[cell] = {
                    "n": int(n),
                    "rho": None,
                    "p": None,
                    "critical_rho_p05": None,
                    "status": "too_few",
                }
                continue
            try:
                r = pg.partial_corr(
                    data=sub, x=x_col, y="G", covar=["log_prompt_tokens"], method="spearman"
                )
                rho = float(r["r"].values[0])
                p = float(r["p_val"].values[0])
                status = "ok"
            except Exception as e:
                rho, p = float("nan"), float("nan")
                status = "compute_error"
                logger.warning("per_cell partial failed on cell %s: %s", cell, e)
            # Per-cell critical |rho| at p<0.05 (Fisher-z asymptotic; for small N
            # the table value differs but this is a serviceable annotation).
            # critical |rho| ≈ 1.96 / sqrt(n - 3) for Fisher-z.
            crit = 1.96 / np.sqrt(max(n - 3, 1))
            per_cell[cell] = rho
            per_cell_meta[cell] = {
                "n": int(n),
                "rho": rho,
                "p": p,
                "critical_rho_p05": float(crit),
                "status": status,
            }
    return per_cell, per_cell_meta


def _sliding_threshold_curve(
    df: pd.DataFrame,
    x_col: str,
    window_size: int = 50,
    step: int = 10,
    g_floor: float = 0.1,
) -> dict:
    """Sliding-quantile-method threshold curve (§3.5 concern #6).

    Returns the full curve, not just a single D* cutoff. Keys:
    window_centers / window_means_G / first_dip_below_g_floor.
    """
    sorted_df = df.sort_values(x_col).reset_index(drop=True)
    centers: list[float] = []
    means: list[float] = []
    for start in range(0, len(sorted_df) - window_size + 1, step):
        window = sorted_df.iloc[start : start + window_size]
        centers.append(float(window[x_col].mean()))
        means.append(float(window["G"].mean()))
    # First window center where window mean G falls below g_floor.
    first_dip = None
    for c, m in zip(centers, means, strict=True):
        if m < g_floor:
            first_dip = c
            break
    return {
        "x_col": x_col,
        "window_size": window_size,
        "step": step,
        "g_floor": g_floor,
        "window_centers": centers,
        "window_means_G": means,
        "first_dip_below_g_floor": first_dip,
    }


def _permutation_null(df: pd.DataFrame, x_col: str, n_perms: int = 1000, seed: int = 42) -> dict:
    """Permute G against fixed x and length covariate; report length-partial null."""
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_perms)
    for p in range(n_perms):
        permuted = df.copy()
        permuted["G_perm"] = rng.permutation(df["G"].to_numpy())
        try:
            r = pg.partial_corr(
                data=permuted,
                x=x_col,
                y="G_perm",
                covar=["log_prompt_tokens"],
                method="spearman",
            )
            null_rhos[p] = float(r["r"].values[0])
        except Exception:
            null_rhos[p] = np.nan
    return {
        "n_perms": n_perms,
        "p2_5": float(np.nanpercentile(null_rhos, 2.5)),
        "p97_5": float(np.nanpercentile(null_rhos, 97.5)),
        "mean": float(np.nanmean(null_rhos)),
    }


def _leave_subset_out(df: pd.DataFrame, x_col: str, mask: pd.Series, label: str) -> dict:
    """Re-run length-partial Spearman on the subset of df where mask is True."""
    sub = df[mask]
    if len(sub) < 10:
        return {"label": label, "n": len(sub), "rho": None, "p": None}
    try:
        r = pg.partial_corr(
            data=sub, x=x_col, y="G", covar=["log_prompt_tokens"], method="spearman"
        )
        return {
            "label": label,
            "n": len(sub),
            "rho": float(r["r"].values[0]),
            "p": float(r["p_val"].values[0]),
        }
    except Exception as e:
        logger.warning("leave-subset-out failed (%s): %s", label, e)
        return {"label": label, "n": len(sub), "rho": None, "p": None, "error": str(e)}


def _fit_mixed_model(df: pd.DataFrame, predictor_col: str, *, is_primary: bool) -> dict:
    """Fit the class-cluster mixed model with explicit convergence handling.

    Per MF-R2-2 (issue #406 round 2): the previous bare ``except Exception``
    + ``mm_slope_p = NaN`` substitution silently un-fireable the positive
    verdict on convergence failure (NaN comparison returns False). We now:

      1. Catch ONLY the specific convergence-failure exception types
         (``np.linalg.LinAlgError`` and ``ValueError`` raised by
         statsmodels when the model is rank-deficient / fails to invert).
         ``ConvergenceWarning`` from statsmodels is a WARNING, not an
         exception — we treat the converged flag from the fit result as
         authoritative.
      2. For the PRIMARY predictor only, try a ``method='lbfgs'``
         fallback before declaring non-convergence (small-cluster
         mixedlm sometimes converges with lbfgs when default Newton
         fails — round-1 stats critic recommendation).
      3. Return a structured dict with a ``converged`` boolean so the
         caller can surface convergence failure as ``verdict="rig_failure"``
         on the primary, or a ``convergence_failed`` flag on secondaries.

    Returns:
        {
          "slope": float | None, "slope_p": float | None,
          "summary": str, "converged": bool, "method": str,
          "error": str | None,
        }
    """
    import warnings

    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    formula = f"G ~ {predictor_col} + log_prompt_tokens"
    method_chain: list[str] = ["bfgs"]  # statsmodels default for mixedlm
    if is_primary:
        method_chain.append("lbfgs")  # only escalate the primary; cosines stay descriptive

    last_error: str | None = None
    for method in method_chain:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                md = smf.mixedlm(formula, data=df, groups="class_pair").fit(
                    reml=False, method=method
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "mixedlm fit raised %s on predictor=%s method=%s",
                    type(exc).__name__,
                    predictor_col,
                    method,
                )
                continue
        # Inspect the fit result. converged is authoritative; warnings inform.
        # statsmodels' MixedLMResultsWrapper exposes `converged` post-fit.
        converged_attr = getattr(md, "converged", None)
        had_convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        # Treat NaN slope or pvalue as a non-converged result too.
        slope_val = md.params.get(predictor_col, float("nan"))
        slope_p_val = md.pvalues.get(predictor_col, float("nan"))
        slope_finite = np.isfinite(slope_val) and np.isfinite(slope_p_val)
        is_converged = bool(slope_finite) and (converged_attr is not False)
        if had_convergence_warning and converged_attr is False:
            is_converged = False
        if is_converged:
            return {
                "slope": float(slope_val),
                "slope_p": float(slope_p_val),
                "summary": str(md.summary()),
                "converged": True,
                "method": method,
                "error": None,
            }
        # Not converged with this method — try the next, if any.
        last_error = (
            f"non-converged (method={method}, converged_attr={converged_attr}, "
            f"slope_finite={slope_finite}, had_convergence_warning={had_convergence_warning})"
        )
        logger.warning("mixedlm did not converge for predictor=%s: %s", predictor_col, last_error)

    return {
        "slope": float("nan"),
        "slope_p": float("nan"),
        "summary": f"NOT CONVERGED: {last_error}",
        "converged": False,
        "method": method_chain[-1],
        "error": last_error,
    }


def _analyze_predictor(
    df: pd.DataFrame,
    predictor_col: str,
    label: str,
    is_primary: bool,
) -> dict:
    """Run the full per-predictor analysis stack.

    Returns dict shaped identically across predictors so the summary
    table can iterate uniformly.
    """
    raw = st.spearmanr(df["G"], df[predictor_col])
    lp_pg = pg.partial_corr(
        data=df, x=predictor_col, y="G", covar=["log_prompt_tokens"], method="spearman"
    )
    lp_inline = _length_partial_inline(df[predictor_col], df["G"], df["log_prompt_tokens"])
    length_tercile = _length_tercile_rhos(df, predictor_col, "G")

    # Class-cluster mixed model: G ~ predictor + log_prompt_tokens, random
    # intercept per class_pair (16 cells). MF-R2-2 fix: NO bare except;
    # catch only the specific convergence-failure types, try an lbfgs
    # fallback, and surface convergence failure as a verdict-level signal
    # (rig_failure) instead of NaN-silently-falling-through-to-ambiguous.
    mm_result = _fit_mixed_model(df, predictor_col, is_primary=is_primary)
    mm_slope = mm_result["slope"]
    mm_slope_p = mm_result["slope_p"]
    mm_summary = mm_result["summary"]
    mm_converged = mm_result["converged"]
    mm_method = mm_result["method"]
    mm_error = mm_result["error"]

    boot_rhos = _cluster_bootstrap_partial_spearman(df, predictor_col)
    ci_low = float(np.nanpercentile(boot_rhos, 2.5))
    ci_high = float(np.nanpercentile(boot_rhos, 97.5))

    per_cell_rhos, per_cell_meta = _per_cell_partials(df, predictor_col)
    # NaN-safe per-cell max: per_cell only contains entries with a
    # computed rho (singleton-Class-C absent cells + n<5 cells are in
    # per_cell_meta as 'absent' / 'too_few' but NOT in per_cell), but a
    # compute_error path may still emit NaN. Use np.nanmax over the
    # absolute values; empty dict (or all-NaN) -> NaN.
    if per_cell_rhos:
        abs_values = np.array([abs(v) for v in per_cell_rhos.values()], dtype=np.float64)
        per_cell_max_abs = (
            float(np.nanmax(abs_values)) if np.any(np.isfinite(abs_values)) else float("nan")
        )
    else:
        per_cell_max_abs = float("nan")

    # Round-2 critic Lens 5 resolution: leave-class-C-out + leave-A1-out
    # apply to ALL 7 predictors (was primary-only in v7-v8 §4 pseudocode).
    leave_class_c = _leave_subset_out(df, predictor_col, df["class_i"] != "C", "leave_class_C_out")
    leave_a1 = _leave_subset_out(df, predictor_col, df["T_i"] != "A1", "leave_A1_out")

    # Primary-only: sliding-threshold curve + permutation null
    threshold_curve = None
    permutation_null = None
    if is_primary:
        threshold_curve = _sliding_threshold_curve(df, predictor_col)
        permutation_null = _permutation_null(df, predictor_col)

    # Verdict only for the primary predictor (cosine rows = descriptive).
    # MF-R2-2: if the primary's mixed model didn't converge, the verdict is
    # `rig_failure` — NOT a NaN-driven "ambiguous". This surfaces the binding
    # constraint at the verdict level instead of burying it in mm_summary.
    verdict = None
    rig_failure_reason: str | None = None
    if is_primary:
        if not mm_converged:
            verdict = "rig_failure"
            rig_failure_reason = (
                f"mixed_model_convergence_failed (method={mm_method}, error={mm_error})"
            )
            logger.error(
                "PRIMARY mixed model did not converge after fallback chain — "
                "verdict=rig_failure (predictor=%s, error=%s)",
                predictor_col,
                mm_error,
            )
        else:
            lp_rho = float(lp_pg["r"].values[0])
            lp_p = float(lp_pg["p_val"].values[0])
            if (
                abs(lp_rho) >= POS_RHO_THRESHOLD
                and lp_p < POS_P_THRESHOLD
                and mm_slope_p < POS_P_THRESHOLD
                and np.sign(mm_slope) == np.sign(lp_rho)
            ):
                verdict = "positive"
            elif (
                abs(lp_rho) < NEG_RHO_THRESHOLD
                and ci_low >= NEG_CI_LOW_BOUND
                and ci_high <= NEG_CI_HIGH_BOUND
                and per_cell_max_abs < NEG_PER_CELL_CAP
            ):
                verdict = "negative"
            else:
                verdict = "ambiguous"

    return {
        "label": label,
        "is_primary": is_primary,
        "verdict": verdict,
        "rig_failure_reason": rig_failure_reason,
        "raw_spearman_rho": float(raw.correlation),
        "raw_spearman_p": float(raw.pvalue),
        "length_partial_rho_pingouin": float(lp_pg["r"].values[0]),
        "length_partial_p_pingouin": float(lp_pg["p_val"].values[0]),
        "length_partial_rho_inline": lp_inline["r"],
        "length_partial_p_inline": lp_inline["p"],
        "length_tercile_rhos": length_tercile,
        "cluster_bootstrap_ci": [ci_low, ci_high],
        "mixed_model_slope": mm_slope,
        "mixed_model_slope_p": mm_slope_p,
        "mixed_model_summary": mm_summary,
        # MF-R2-2: convergence telemetry per predictor (secondaries surface
        # the flag too; only the primary escalates to verdict=rig_failure).
        "mixed_model_converged": mm_converged,
        "mixed_model_method": mm_method,
        "mixed_model_error": mm_error,
        "convergence_failed": (not mm_converged),
        "per_cell_partials": per_cell_rhos,
        "per_cell_meta": per_cell_meta,
        "per_cell_max_abs_rho": per_cell_max_abs,
        "leave_class_c_out": leave_class_c,
        "leave_a1_out": leave_a1,
        "threshold_curve": threshold_curve,
        "permutation_null_95pct": (
            [permutation_null["p2_5"], permutation_null["p97_5"]] if permutation_null else None
        ),
        "permutation_null_full": permutation_null,
    }


def _build_dataframe(
    d_payload: dict,
    g_payload: dict,
    c_payloads: dict[int, dict],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Construct the analysis DataFrame (up to 240 rows with N=16 active conditions).

    Drops diagonal-failed conditions (G[i, i] < 0.7) per Phase 3 sanity report.
    Returns (df, diagonal_passed, diagonal_dropped).
    """
    diagonal_passed = list(g_payload["diagonal_passed"])
    diagonal_dropped = [c["cid"] for c in g_payload["diagonal_failed"]]
    rows = []
    for ci in diagonal_passed:
        for cj in diagonal_passed:
            if ci == cj:
                continue
            kl = d_payload["KL"][ci][cj]
            js = d_payload["JS"][ci][cj]
            g = g_payload["G"][ci][cj]["rate"]
            n_tok = d_payload["prompt_tokens"][ci][cj]
            row = {
                "T_i": ci,
                "T_j": cj,
                "class_i": ci[0],
                "class_j": cj[0],
                "class_pair": ci[0] + "_" + cj[0],
                "D": kl,
                "JS": js,
                "G": g,
                "log_prompt_tokens": float(np.log(n_tok)),
            }
            for L in TARGET_LAYERS:
                row[f"C_L{L}"] = c_payloads[L]["matrix"][ci][cj]
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, diagonal_passed, diagonal_dropped


def _per_position_trajectory(df: pd.DataFrame, d_per_pos_payload: dict) -> list[dict]:
    """v9-NEW: length-partial Spearman rho(D_k, G) for k = 0..24 with bootstrap CI."""
    tensor = np.array(
        [[np.nan if v is None else v for v in row] for row in d_per_pos_payload["tensor"]],
        dtype=np.float64,
    )
    ordered = d_per_pos_payload["ordered_pairs"]
    # Build (T_i, T_j) -> tensor-row index mapping
    pair_to_row = {(t[0], t[1]): i for i, t in enumerate(ordered)}
    # Align tensor rows to df rows (df rows are ordered (T_i, T_j); tensor rows
    # in the same order, but the diagonal-failed conds drop some rows).
    df_pairs = list(zip(df["T_i"].tolist(), df["T_j"].tolist(), strict=True))
    aligned_idx = np.array([pair_to_row[p] for p in df_pairs], dtype=np.int64)
    tensor_aligned = tensor[aligned_idx]  # (n_pairs_in_df, K_TARGET)

    out = []
    for k in range(K_TARGET):
        df_k = df.copy()
        df_k["D_k"] = tensor_aligned[:, k]
        df_k = df_k.dropna(subset=["D_k"])
        if len(df_k) < 10:
            out.append(
                {
                    "k": k,
                    "n_valid_pairs": len(df_k),
                    "length_partial_rho": None,
                    "length_partial_p": None,
                    "cluster_bootstrap_ci_500": None,
                }
            )
            continue
        r = pg.partial_corr(
            data=df_k, x="D_k", y="G", covar=["log_prompt_tokens"], method="spearman"
        )
        boot = _cluster_bootstrap_partial_spearman(df_k, "D_k", n_boot=500, seed=42)
        ci_low = float(np.nanpercentile(boot, 2.5))
        ci_high = float(np.nanpercentile(boot, 97.5))
        out.append(
            {
                "k": k,
                "n_valid_pairs": len(df_k),
                "length_partial_rho": float(r["r"].values[0]),
                "length_partial_p": float(r["p_val"].values[0]),
                "cluster_bootstrap_ci_500": [ci_low, ci_high],
            }
        )
    return out


def _k_window_sweep(df: pd.DataFrame, d_per_pos_payload: dict) -> tuple[list[dict], dict]:
    """v9-NEW: 9-window descriptive sweep across K subsets. Returns (sweep, best_window)."""
    tensor = np.array(
        [[np.nan if v is None else v for v in row] for row in d_per_pos_payload["tensor"]],
        dtype=np.float64,
    )
    ordered = d_per_pos_payload["ordered_pairs"]
    pair_to_row = {(t[0], t[1]): i for i, t in enumerate(ordered)}
    df_pairs = list(zip(df["T_i"].tolist(), df["T_j"].tolist(), strict=True))
    aligned_idx = np.array([pair_to_row[p] for p in df_pairs], dtype=np.int64)
    tensor_aligned = tensor[aligned_idx]

    sweep = []
    for k_lo, k_hi in K_WINDOWS:
        window_slice = tensor_aligned[:, k_lo : k_hi + 1]
        d_w = np.nanmean(window_slice, axis=1)
        df_w = df.copy()
        df_w["D_W"] = d_w
        df_w = df_w.dropna(subset=["D_W"])
        if len(df_w) < 10:
            sweep.append(
                {
                    "window": f"k={k_lo}..{k_hi} (W={k_hi - k_lo + 1})",
                    "k_lo": k_lo,
                    "k_hi": k_hi,
                    "n_valid_pairs": len(df_w),
                    "length_partial_rho": None,
                    "length_partial_p": None,
                    "cluster_bootstrap_ci_500": None,
                    "is_primary": (k_lo == 0 and k_hi == K_TARGET - 1),
                }
            )
            continue
        r = pg.partial_corr(
            data=df_w, x="D_W", y="G", covar=["log_prompt_tokens"], method="spearman"
        )
        boot = _cluster_bootstrap_partial_spearman(df_w, "D_W", n_boot=500, seed=42)
        sweep.append(
            {
                "window": f"k={k_lo}..{k_hi} (W={k_hi - k_lo + 1})",
                "k_lo": k_lo,
                "k_hi": k_hi,
                "n_valid_pairs": len(df_w),
                "length_partial_rho": float(r["r"].values[0]),
                "length_partial_p": float(r["p_val"].values[0]),
                "cluster_bootstrap_ci_500": [
                    float(np.nanpercentile(boot, 2.5)),
                    float(np.nanpercentile(boot, 97.5)),
                ],
                "is_primary": (k_lo == 0 and k_hi == K_TARGET - 1),
            }
        )
    # Best |rho| among the 9 (descriptive; garden-of-forking-paths caveat noted).
    valid = [w for w in sweep if w["length_partial_rho"] is not None]
    best = max(valid, key=lambda w: abs(w["length_partial_rho"])) if valid else None
    return sweep, best


def _build_summary_table(results: dict[str, dict]) -> list[dict]:
    out = []
    for r in results.values():
        out.append(
            {
                "predictor": r["label"],
                "is_primary": r["is_primary"],
                "length_partial_rho_pg": r["length_partial_rho_pingouin"],
                "length_partial_rho_inline": r["length_partial_rho_inline"],
                "cluster_bootstrap_ci": r["cluster_bootstrap_ci"],
                "per_cell_max_abs_rho": r["per_cell_max_abs_rho"],
                "leave_class_c_out_rho": r["leave_class_c_out"].get("rho"),
                "leave_a1_out_rho": r["leave_a1_out"].get("rho"),
                "verdict": r["verdict"] if r["is_primary"] else "descriptive",
            }
        )
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Load all four artifact groups.
    d_payload = json.loads(DIVERG_PATH.read_text())
    g_payload = json.loads(G_MATRIX_PATH.read_text())
    d_per_pos_payload = json.loads(DIVERG_PER_POS_PATH.read_text())
    c_payloads: dict[int, dict] = {}
    for L in TARGET_LAYERS:
        path = COSINE_DIR / f"C_L{L}.json"
        c_payloads[L] = json.loads(path.read_text())

    df, diagonal_passed, diagonal_dropped = _build_dataframe(d_payload, g_payload, c_payloads)
    expected_n_pairs = len(diagonal_passed) * (len(diagonal_passed) - 1)
    if len(df) != expected_n_pairs:
        raise AssertionError(f"df has {len(df)} rows, expected {expected_n_pairs}")
    logger.info(
        "Built DataFrame: %d rows (%d diagonal-passed conds, %d dropped: %s)",
        len(df),
        len(diagonal_passed),
        len(diagonal_dropped),
        diagonal_dropped,
    )

    # Tensor-shape asserts at boundary (CLAUDE.md).
    assert df["D"].min() >= 0, f"KL has negative values: min={df['D'].min()}"
    for L in TARGET_LAYERS:
        col = f"C_L{L}"
        assert df[col].between(0, 2).all(), (
            f"Cosine distance {col} out of [0, 2]; min={df[col].min()} max={df[col].max()}"
        )

    # G distribution histogram (§3.5 concern #7 -- report up-front).
    g_hist_counts, g_hist_edges = np.histogram(df["G"], bins=np.linspace(0, 1, 11))

    # Run 7 parallel regressions.
    results: dict[str, dict] = {}
    results["KL_primary"] = _analyze_predictor(
        df, "D", "Forward KL (primary, K=25-mean)", is_primary=True
    )
    results["JS_secondary"] = _analyze_predictor(
        df, "JS", "JS divergence (free secondary, K=25-mean)", is_primary=False
    )
    for L in TARGET_LAYERS:
        results[f"cosine_L{L}"] = _analyze_predictor(
            df, f"C_L{L}", f"Cosine on residual stream, layer {L}", is_primary=False
        )

    summary_table = _build_summary_table(results)

    # v9-NEW per-position trajectory + K-window sweep.
    trajectory = _per_position_trajectory(df, d_per_pos_payload)
    sweep, best_window = _k_window_sweep(df, d_per_pos_payload)

    primary = results["KL_primary"]
    # MF-R2-2: when the primary's mixed model fails to converge the verdict
    # is "rig_failure" — surface that distinctly in the headline so the
    # analyzer / clean-result-critic doesn't have to dig into mm_summary.
    if primary["verdict"] == "rig_failure":
        headline = (
            f"PRIMARY (forward KL, K=25-mean): VERDICT=rig_failure. "
            f"Reason: {primary['rig_failure_reason']}. "
            f"Length-partial Spearman rho = {primary['length_partial_rho_pingouin']:.3f} "
            f"with 95% cluster-bootstrap CI "
            f"[{primary['cluster_bootstrap_ci'][0]:.3f}, "
            f"{primary['cluster_bootstrap_ci'][1]:.3f}] (N={len(df)}, "
            f"{df['class_pair'].nunique()} cells). "
            "The Spearman + bootstrap numbers are reported for context but "
            "the mixed-model component of the pre-registered verdict could "
            "not be evaluated. Re-run with a different model formulation or "
            "treat the experiment as inconclusive."
        )
    else:
        headline = (
            f"PRIMARY (forward KL, K=25-mean): length-partial Spearman rho = "
            f"{primary['length_partial_rho_pingouin']:.3f} with 95% cluster-bootstrap "
            f"CI [{primary['cluster_bootstrap_ci'][0]:.3f}, "
            f"{primary['cluster_bootstrap_ci'][1]:.3f}] (N={len(df)}, "
            f"{df['class_pair'].nunique()} cells); verdict={primary['verdict']}. "
            "DESCRIPTIVE secondaries: see 7-row summary table for JS + 6 cosine layers."
        )
    logger.info(headline)

    out_payload = {
        "schema_version": "v9",
        "git_commit": _git_commit_hash(),
        "n_pairs": len(df),
        "diagonal_passed": diagonal_passed,
        "diagonal_dropped": diagonal_dropped,
        "verdict": primary["verdict"],
        "headline": headline,
        "g_histogram": {
            "bin_edges": g_hist_edges.tolist(),
            "counts": g_hist_counts.tolist(),
        },
        "summary_table": summary_table,
        "per_predictor": results,
        "per_position_trajectory": trajectory,
        "k_window_sweep": sweep,
        "best_k_window_descriptive": best_window,
        "conditions": [
            {"cid": c.cid, "class": c.cls, "name": CONDITIONS_BY_ID[c.cid].name} for c in CONDITIONS
        ],
        "K_target": K_TARGET,
        "K_windows_tested": [list(w) for w in K_WINDOWS],
        "thresholds": {
            "pos_rho_abs": POS_RHO_THRESHOLD,
            "pos_p": POS_P_THRESHOLD,
            "neg_rho_abs": NEG_RHO_THRESHOLD,
            "neg_per_cell_cap": NEG_PER_CELL_CAP,
            "neg_ci_low_bound": NEG_CI_LOW_BOUND,
            "neg_ci_high_bound": NEG_CI_HIGH_BOUND,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out_payload, indent=2, default=str))
    logger.info("Wrote %s (verdict=%s)", OUT_PATH, primary["verdict"])


if __name__ == "__main__":
    main()
