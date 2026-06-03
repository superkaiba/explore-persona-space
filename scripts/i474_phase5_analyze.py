"""Phase 5 — analysis for #474. 8 cells of (arm x checkpoint).

Issue #474 plan v3 §4.7. Per (arm, checkpoint) reports:

  Primary readouts (M3 + M4 + M5 mandatory):
    - dG length-partial Spearman rho(D, dG) under THREE masks:
        (a) all 240 off-diagonal cells
        (b) exclude A3/A4/A5 as source (n=180)
        (c) exclude A3/A4/A5 as either source or target (n=156)
      M3 headline test: H1 requires (c) rho < 0 AND CI excludes zero.
    - Off-ceiling-subset rho (M4): dG < (ceiling - 0.1 nat); effective N.
    - Suppression-difficulty partial rho(D, dG | S) for A_loc (M5).
      S = per-(i,j) mean negative-row training loss from
      ``eval_results/issue_474/train_diag/suppression_difficulty_loc_*_ep*.json``.

  Secondary readouts (route-b cross-check — DRIFT, not transfer):
    - KL: length-partial rho(D, KL) — labeled DRIFT in JSON keys.

  Cross-arm H3 (paired bootstrap A_loc - A_pos):
    - Matched-epoch AND matched-step (A_loc ep1 vs A_pos ep2).

  Cross-experiment (DESCRIPTIVE):
    - #406 head-to-head paired bootstrap |rho_474_loc| - |rho_406|.

  Raw alongside processed: every mask + off-ceiling subset reported with
  raw Spearman + Pearson alongside the length-partial.

  Reproduction tripwire: |A_pos ep1 rho - (-0.27)| > 0.10 flags caveat.

Output:
  - eval_results/issue_474/analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from explore_persona_space.experiments.i406_conditions import CONDITIONS

# Optional pingouin (round-2 fix CONCERN 4): pingouin is NOT in
# pyproject.toml; making the import optional + routing the partial-
# correlation through the scipy/statsmodels fallback when pingouin is
# absent. Pingouin's only role here is `partial_corr(method="spearman")`
# which is rank-residualization on x ~ covars and y ~ covars + Pearson
# on the residuals — equivalent to ``_partial_spearman_fallback`` below.
try:
    import pingouin as pg

    _PINGOUIN_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by CI without pingouin
    pg = None  # type: ignore[assignment]
    _PINGOUIN_AVAILABLE = False

logger = logging.getLogger("i474.phase5")


def _partial_spearman_fallback(df: pd.DataFrame, x: str, y: str, covar: list[str]) -> float:
    """Multi-covariate partial Spearman via rank-residualization + Pearson.

    Equivalent in this code's usage to ``pg.partial_corr(method="spearman")[
    "r"].values[0]`` — rank-transform x, y, and each covar; OLS-regress
    each of (x, y) on the covar block; Pearson-correlate the residuals.
    Falls back to scipy.linregress for a single covariate and to
    ``numpy.linalg.lstsq`` for ≥2 covariates.

    Returns ``float('nan')`` on degenerate input (n < 5, rank-deficient
    covars, etc.) — matches the pingouin path's behaviour where the
    caller checks for None/NaN.
    """
    n = len(df)
    if n < 5:
        return float("nan")
    x_rank = st.rankdata(df[x].to_numpy())
    y_rank = st.rankdata(df[y].to_numpy())
    if not covar:
        try:
            return float(st.pearsonr(x_rank, y_rank).statistic)
        except Exception:
            return float("nan")
    if len(covar) == 1:
        c_rank = st.rankdata(df[covar[0]].to_numpy())
        sl_x, ix, *_ = st.linregress(c_rank, x_rank)
        sl_y, iy, *_ = st.linregress(c_rank, y_rank)
        x_resid = x_rank - (sl_x * c_rank + ix)
        y_resid = y_rank - (sl_y * c_rank + iy)
        try:
            return float(st.pearsonr(x_resid, y_resid).statistic)
        except Exception:
            return float("nan")
    # ≥2 covariates: OLS via lstsq with an intercept column.
    C = np.column_stack([st.rankdata(df[c].to_numpy()) for c in covar] + [np.ones(n)])
    try:
        bx, *_ = np.linalg.lstsq(C, x_rank, rcond=None)
        by, *_ = np.linalg.lstsq(C, y_rank, rcond=None)
        x_resid = x_rank - C @ bx
        y_resid = y_rank - C @ by
        return float(st.pearsonr(x_resid, y_resid).statistic)
    except Exception:
        return float("nan")


def _partial_corr_r(df: pd.DataFrame, x: str, y: str, covar: list[str]) -> float:
    """Return the partial-Spearman r for x vs y controlling for `covar`.

    Prefers pingouin when available (more diagnostics, e.g. p-value);
    falls back to ``_partial_spearman_fallback`` otherwise.
    """
    if _PINGOUIN_AVAILABLE and pg is not None:
        try:
            r = pg.partial_corr(data=df, x=x, y=y, covar=covar, method="spearman")
            return float(r["r"].values[0])
        except Exception:
            return float("nan")
    return _partial_spearman_fallback(df, x, y, covar)


D_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
G406_PATH = Path("eval_results/issue_406/cross_eval/G_matrix.json")
CROSS_DIR_474 = Path("eval_results/issue_474/cross_eval")
TRAIN_DIAG_DIR_474 = Path("eval_results/issue_474/train_diag")
OUT_PATH = Path("eval_results/issue_474/analysis.json")

STYLIZED_SOURCES = {"A3", "A4", "A5"}  # pirate / comedian / villain (#462 caveat)
ARMS = ("pos", "loc")
DEFAULT_CHECKPOINT_EPOCHS = (1, 2, 3, 5)
PARENT_462_EP1_RHO = -0.27  # tripwire reference (plan v3 §4.7)
TRIPWIRE_DELTA_MAX = 0.10
OFF_CEILING_NAT_BAND = 0.1  # dG < (ceiling - 0.1 nat)


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _length_partial_inline(x: pd.Series, y: pd.Series, covar: pd.Series) -> dict:
    x_rank = st.rankdata(x.to_numpy())
    y_rank = st.rankdata(y.to_numpy())
    c_rank = st.rankdata(covar.to_numpy())
    sl_x, ix, *_ = st.linregress(c_rank, x_rank)
    sl_y, iy, *_ = st.linregress(c_rank, y_rank)
    x_resid = x_rank - (sl_x * c_rank + ix)
    y_resid = y_rank - (sl_y * c_rank + iy)
    res = st.pearsonr(x_resid, y_resid)
    return {"r": float(res.statistic), "p": float(res.pvalue), "n": len(x_rank)}


def _cluster_bootstrap_partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covar_col: str | None,
    n_boot: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Cluster-bootstrap by class_pair; partial Spearman per resample."""
    rng = np.random.default_rng(seed)
    cell_ids = sorted(df["class_pair"].unique())
    cell_to_rows = {cell: df.index[df["class_pair"] == cell].to_numpy() for cell in cell_ids}
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled])
        sub = df.loc[rows]
        try:
            if covar_col is None:
                # raw Spearman fallback
                r = st.spearmanr(sub[x_col], sub[y_col]).correlation
            else:
                r = _partial_corr_r(sub, x_col, y_col, [covar_col])
            boot_rhos[b] = float(r)
        except Exception:
            boot_rhos[b] = np.nan
    return boot_rhos


def _safe_partial(df: pd.DataFrame, x: str, y: str, covar: str | list[str]) -> dict:
    """Length-partial (or multi-covariate) Spearman.

    When pingouin is installed, uses its diagnostic p-value alongside the r;
    otherwise routes through ``_partial_corr_r`` (rank-residualize + Pearson
    over residuals) and reports r only. The keys ``rho_pingouin`` and
    ``p_pingouin`` are retained for back-compat with downstream readers
    (they reflect the partial r regardless of which path produced it).
    """
    if len(df) < 5:
        return {"n": len(df), "rho_pingouin": None, "p_pingouin": None, "error": "too_few_rows"}
    out = {"n": len(df)}
    covar_list = [covar] if isinstance(covar, str) else covar
    try:
        if _PINGOUIN_AVAILABLE and pg is not None:
            r = pg.partial_corr(data=df, x=x, y=y, covar=covar_list, method="spearman")
            out["rho_pingouin"] = float(r["r"].values[0])
            out["p_pingouin"] = float(r["p_val"].values[0])
        else:
            rho = _partial_corr_r(df, x, y, covar_list)
            out["rho_pingouin"] = None if np.isnan(rho) else float(rho)
            out["p_pingouin"] = None  # not computed in the fallback path
            out["partial_corr_backend"] = "scipy_fallback"
    except Exception as e:
        out["rho_pingouin"] = None
        out["p_pingouin"] = None
        out["error_pingouin"] = str(e)
    if isinstance(covar, str):
        inline = _length_partial_inline(df[x], df[y], df[covar])
        out["rho_inline"] = inline["r"]
        out["p_inline"] = inline["p"]
    return out


def _raw_spearman_and_pearson(df: pd.DataFrame, x: str, y: str) -> dict:
    if len(df) < 5:
        return {"n": len(df), "error": "too_few_rows"}
    sp = st.spearmanr(df[x], df[y])
    pe = st.pearsonr(df[x], df[y])
    return {
        "n": len(df),
        "spearman_rho": float(sp.correlation),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pe.statistic),
        "pearson_p": float(pe.pvalue),
    }


def _build_dataframe(
    g474: dict,
    d_payload: dict,
    g406_payload: dict,
    per_cell_dir: Path,
    arm: str,
    epoch: int,
) -> pd.DataFrame:
    """Build a 240-row off-diagonal DataFrame for one (arm, epoch) cell."""
    cond_by_cid = {c.cid: c for c in CONDITIONS}
    cids = [c.cid for c in CONDITIONS]
    KL = d_payload["KL"]
    JS = d_payload["JS"]
    PT = d_payload["prompt_tokens"]
    G_orig = g406_payload["G"]
    G474 = g474["G"]

    def _length_for_cell(ci: str, cj: str) -> float:
        cell_path = per_cell_dir / f"G_{arm}_ep{epoch}_{ci}__{cj}.json"
        if cell_path.exists():
            cell = json.loads(cell_path.read_text())
            prompt_lens = cell.get("prompt_lens_per_q", [])
            R_lens = cell.get("R_lens_per_q", [])
            if prompt_lens and R_lens:
                return float(np.mean([p + r for p, r in zip(prompt_lens, R_lens, strict=True)]))
        return float(PT[ci][cj])

    rows = []
    for ci in cids:
        ci_cls = cond_by_cid[ci].cls
        for cj in cids:
            if ci == cj:
                continue
            cj_cls = cond_by_cid[cj].cls
            kl = KL[ci][cj]
            js = JS[ci][cj]
            g_orig_cell = G_orig[ci][cj]
            if g_orig_cell is None:
                continue
            g_orig_rate = float(g_orig_cell["rate"])
            g474_cell = G474[ci][cj]
            length = _length_for_cell(ci, cj)
            rows.append(
                {
                    "T_i": ci,
                    "T_j": cj,
                    "class_i": ci_cls,
                    "class_j": cj_cls,
                    "class_pair": f"{ci_cls}_{cj_cls}",
                    "D": float(kl) if kl is not None else None,
                    "JS": float(js) if js is not None else None,
                    "G_orig": g_orig_rate,
                    "G_logprob": float(g474_cell["g_logprob"]),
                    "B_logprob": float(g474_cell["b_logprob"]),
                    "delta_g": float(g474_cell["delta_g"]),
                    "kl_post_response_slot": (
                        float(g474_cell["kl_post_response_slot"])
                        if g474_cell.get("kl_post_response_slot") is not None
                        else None
                    ),
                    "emission_recompute_rate": float(g474_cell["emission_recompute_rate"]),
                    "prompt_plus_R_tokens": length,
                    "log_prompt_tokens": float(np.log(max(length, 1.0))),
                }
            )
    df = pd.DataFrame(rows)
    return df.dropna(subset=["D"]).reset_index(drop=True)


def _three_mask_rho(df: pd.DataFrame, y_col: str, n_boot: int, seed: int) -> dict:
    """Plan v3 §4.7 M3 — rho under three masks + boot CI per mask + raw."""

    def _one_mask(sub: pd.DataFrame) -> dict:
        partial = _safe_partial(sub, x="D", y=y_col, covar="log_prompt_tokens")
        if len(sub) >= 10:
            boot = _cluster_bootstrap_partial_spearman(
                sub, "D", y_col, "log_prompt_tokens", n_boot=n_boot, seed=seed
            )
            partial["bootstrap_ci_2_5"] = float(np.nanpercentile(boot, 2.5))
            partial["bootstrap_ci_97_5"] = float(np.nanpercentile(boot, 97.5))
            partial["ci_excludes_zero"] = bool(
                partial["bootstrap_ci_97_5"] < 0 or partial["bootstrap_ci_2_5"] > 0
            )
        return {
            "length_partial_spearman": partial,
            "raw": _raw_spearman_and_pearson(sub, "D", y_col),
        }

    return {
        "mask_a_all": _one_mask(df),
        "mask_b_exclude_stylized_source": _one_mask(
            df[~df["class_i"].isin([]) & ~df["T_i"].isin(STYLIZED_SOURCES)].reset_index(drop=True)
        ),
        "mask_c_exclude_stylized_either": _one_mask(
            df[~df["T_i"].isin(STYLIZED_SOURCES) & ~df["T_j"].isin(STYLIZED_SOURCES)].reset_index(
                drop=True
            )
        ),
    }


def _off_ceiling_rho(df: pd.DataFrame, y_col: str, n_boot: int, seed: int) -> dict:
    """Plan v3 §4.7 M4 — rho on the off-ceiling subset (dG < ceiling - 0.1 nat)."""
    if y_col not in df.columns:
        return {"error": f"{y_col} column missing"}
    ceiling = float(df[y_col].max())
    sub = df[df[y_col] < (ceiling - OFF_CEILING_NAT_BAND)].reset_index(drop=True)
    out = {
        "ceiling_value": ceiling,
        "off_ceiling_band_nat": OFF_CEILING_NAT_BAND,
        "n_effective": len(sub),
    }
    if len(sub) < 5:
        out["error"] = "too_few_rows"
        return out
    partial = _safe_partial(sub, x="D", y=y_col, covar="log_prompt_tokens")
    if len(sub) >= 10:
        boot = _cluster_bootstrap_partial_spearman(
            sub, "D", y_col, "log_prompt_tokens", n_boot=n_boot, seed=seed
        )
        partial["bootstrap_ci_2_5"] = float(np.nanpercentile(boot, 2.5))
        partial["bootstrap_ci_97_5"] = float(np.nanpercentile(boot, 97.5))
        partial["ci_excludes_zero"] = bool(
            partial["bootstrap_ci_97_5"] < 0 or partial["bootstrap_ci_2_5"] > 0
        )
    out["length_partial_spearman"] = partial
    out["raw"] = _raw_spearman_and_pearson(sub, "D", y_col)
    return out


def _load_suppression_matrix(cid: str, epoch: int) -> dict[str, float] | None:
    """Load per-(cid, epoch) suppression-difficulty values keyed by bystander_j."""
    path = TRAIN_DIAG_DIR_474 / f"suppression_difficulty_loc_{cid}_ep{epoch}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    # Keys in JSON are "{source_i}__{bystander_j}"; strip prefix.
    return {
        k.split("__")[1]: float(v)
        for k, v in payload.get("per_bystander_mean_neg_loss", {}).items()
    }


def _suppression_difficulty_partial(df: pd.DataFrame, epoch: int, n_boot: int, seed: int) -> dict:
    """Plan v3 §4.7 M5 — partial rho(D, dG | S, log_prompt_tokens).

    Only valid for arm=loc. Joins per-cell dG against the per-(i,j) mean
    negative-row training loss S. Returns descriptive info AND the partial-rho
    with bootstrap CI.
    """
    src_cids = sorted(df["T_i"].unique())
    S_by_pair: dict[tuple[str, str], float] = {}
    for ci in src_cids:
        per_byst = _load_suppression_matrix(ci, epoch)
        if per_byst is None:
            continue
        for byst_j, v in per_byst.items():
            S_by_pair[(ci, byst_j)] = v

    if not S_by_pair:
        return {"error": f"no suppression_difficulty files found in {TRAIN_DIAG_DIR_474}"}

    df2 = df.copy()
    df2["S"] = df2.apply(lambda r: S_by_pair.get((r["T_i"], r["T_j"]), np.nan), axis=1)
    df_use = df2.dropna(subset=["S"]).reset_index(drop=True)
    if len(df_use) < 10:
        return {"n": len(df_use), "error": "too_few_rows_with_S"}

    raw_partial = _safe_partial(df_use, x="D", y="delta_g", covar=["log_prompt_tokens"])
    full_partial = _safe_partial(df_use, x="D", y="delta_g", covar=["log_prompt_tokens", "S"])
    S_vs_D = _raw_spearman_and_pearson(df_use, "D", "S")
    S_vs_delta = _raw_spearman_and_pearson(df_use, "S", "delta_g")

    # Bootstrap for the full partial (D vs dG | S, log_prompt_tokens).
    # Custom loop since _cluster_bootstrap helper supports only single covar.
    rng = np.random.default_rng(seed)
    cell_ids = sorted(df_use["class_pair"].unique())
    cell_to_rows = {
        cell: df_use.index[df_use["class_pair"] == cell].to_numpy() for cell in cell_ids
    }
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled])
        sub = df_use.loc[rows]
        try:
            boot[b] = float(_partial_corr_r(sub, "D", "delta_g", ["log_prompt_tokens", "S"]))
        except Exception:
            boot[b] = np.nan
    full_partial["bootstrap_ci_2_5"] = float(np.nanpercentile(boot, 2.5))
    full_partial["bootstrap_ci_97_5"] = float(np.nanpercentile(boot, 97.5))
    full_partial["ci_excludes_zero"] = bool(
        full_partial["bootstrap_ci_97_5"] < 0 or full_partial["bootstrap_ci_2_5"] > 0
    )

    return {
        "n_cells_with_S": len(df_use),
        "S_vs_D_descriptive": S_vs_D,
        "S_vs_delta_g_descriptive": S_vs_delta,
        "rho_baseline_lengthonly_partial": raw_partial,
        "rho_partial_out_S": full_partial,
        "interpretation_note": (
            "If rho_partial_out_S CI includes zero (or flips sign) while "
            "rho_baseline_lengthonly_partial is negative, D's predictive "
            "power on dG is screened off by per-cell suppression difficulty: "
            "D predicts how hard suppression was at the loss slot, NOT "
            "marker transfer per se."
        ),
    }


def _paired_bootstrap_arm_diff(
    df_pos: pd.DataFrame, df_loc: pd.DataFrame, y_col: str, n_boot: int, seed: int
) -> dict:
    """rho_loc - rho_pos paired bootstrap by class_pair."""
    # Align on (T_i, T_j) — both DataFrames are 240-row off-diagonal.
    merged = df_pos.merge(
        df_loc[["T_i", "T_j", y_col]].rename(columns={y_col: f"{y_col}_loc"}),
        on=["T_i", "T_j"],
        how="inner",
    )
    if len(merged) < 10:
        return {"error": "too_few_overlapping_cells", "n": len(merged)}
    rng = np.random.default_rng(seed)
    cell_ids = sorted(merged["class_pair"].unique())
    cell_to_rows = {
        cell: merged.index[merged["class_pair"] == cell].to_numpy() for cell in cell_ids
    }
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled])
        sub = merged.loc[rows]
        try:
            r_pos = _partial_corr_r(sub, "D", y_col, ["log_prompt_tokens"])
            r_loc = _partial_corr_r(sub, "D", f"{y_col}_loc", ["log_prompt_tokens"])
            boot[b] = float(r_loc) - float(r_pos)
        except Exception:
            boot[b] = np.nan
    return {
        "n": len(merged),
        "diff_loc_minus_pos_mean": float(np.nanmean(boot)),
        "diff_ci_2_5": float(np.nanpercentile(boot, 2.5)),
        "diff_ci_97_5": float(np.nanpercentile(boot, 97.5)),
    }


def _saturation_gauge(df: pd.DataFrame, y_col: str) -> dict:
    if y_col not in df.columns:
        return {"error": f"{y_col} missing"}
    ceiling = float(df[y_col].max())
    n_within_band = int((df[y_col] >= ceiling - OFF_CEILING_NAT_BAND).sum())
    return {
        "ceiling": ceiling,
        "saturation_band_nat": OFF_CEILING_NAT_BAND,
        "n_within_band": n_within_band,
        "n_total": len(df),
        "saturation_fraction": n_within_band / max(1, len(df)),
    }


def _h2h_vs_406(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Plan v3 §4.7 — DESCRIPTIVE paired-bootstrap |rho_474_loc| - |rho_406|.

    Across-DV comparison (dG continuous vs G_orig binary); the clean-result
    MUST NOT narrate this as "same effect validated".
    """
    rng = np.random.default_rng(seed)
    cell_ids = sorted(df["class_pair"].unique())
    cell_to_rows = {cell: df.index[df["class_pair"] == cell].to_numpy() for cell in cell_ids}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled])
        sub = df.loc[rows]
        try:
            r_474 = _partial_corr_r(sub, "D", "delta_g", ["log_prompt_tokens"])
            r_406 = _partial_corr_r(sub, "D", "G_orig", ["log_prompt_tokens"])
            boot[b] = float(abs(r_474)) - float(abs(r_406))
        except Exception:
            boot[b] = np.nan
    return {
        "n_boot": n_boot,
        "abs_diff_mean": float(np.nanmean(boot)),
        "abs_diff_ci_2_5": float(np.nanpercentile(boot, 2.5)),
        "abs_diff_ci_97_5": float(np.nanpercentile(boot, 97.5)),
        "label": (
            "DESCRIPTIVE — different DVs (dG on-policy continuous vs "
            "G_orig off-policy binary). Do NOT narrate as 'same effect validated.'"
        ),
    }


def _per_cell_report(
    g474: dict,
    d_payload: dict,
    g406: dict,
    per_cell_dir: Path,
    arm: str,
    epoch: int,
    n_boot: int,
    seed: int,
) -> dict:
    """Produce one cell of the (arm x checkpoint) report."""
    df = _build_dataframe(g474, d_payload, g406, per_cell_dir, arm, epoch)
    logger.info("arm=%s ep=%d df rows=%d", arm, epoch, len(df))

    out: dict = {
        "arm": arm,
        "epoch": epoch,
        "n_rows": len(df),
        "diagonal_implant_mean_delta_g": float(np.mean(list(g474["diagonals"].values()))),
        "saturation_gauge_delta_g": _saturation_gauge(df, "delta_g"),
    }
    if df["kl_post_response_slot"].notna().any():
        out["saturation_gauge_kl"] = _saturation_gauge(df, "kl_post_response_slot")

    # Primary dG readouts (M3 + M4).
    out["delta_g_three_mask_rho"] = _three_mask_rho(df, "delta_g", n_boot, seed)
    out["delta_g_off_ceiling_rho"] = _off_ceiling_rho(df, "delta_g", n_boot, seed)

    # H1 headline JSON fields per plan §4.7.
    mask_c_partial = out["delta_g_three_mask_rho"]["mask_c_exclude_stylized_either"][
        "length_partial_spearman"
    ]
    out["h1_survives_stylized_mask_c"] = bool(
        mask_c_partial.get("rho_pingouin") is not None
        and mask_c_partial["rho_pingouin"] < 0
        and mask_c_partial.get("ci_excludes_zero", False)
    )
    off_partial = out["delta_g_off_ceiling_rho"].get("length_partial_spearman", {})
    out["h1_off_ceiling_negative"] = bool(
        off_partial.get("rho_pingouin") is not None and off_partial["rho_pingouin"] < 0
    )

    # M5 suppression-difficulty partial (A_loc only).
    if arm == "loc":
        m5 = _suppression_difficulty_partial(df, epoch, n_boot, seed)
        out["m5_suppression_difficulty_partial"] = m5
        full = m5.get("rho_partial_out_S", {}) if isinstance(m5, dict) else {}
        out["h1_survives_suppression_partial"] = bool(
            full.get("rho_pingouin") is not None
            and full["rho_pingouin"] < 0
            and full.get("ci_excludes_zero", False)
        )
    else:
        out["m5_suppression_difficulty_partial"] = {"error": "n/a — A_pos has no negative rows"}
        out["h1_survives_suppression_partial"] = None

    # Secondary KL (DRIFT, NOT marker transfer).
    if df["kl_post_response_slot"].notna().any():
        kl_df = df.dropna(subset=["kl_post_response_slot"]).reset_index(drop=True)
        out["kl_drift_secondary"] = {
            "label": (
                "full-vocab distributional drift at the post-response slot, NOT marker transfer"
            ),
            "three_mask_rho": _three_mask_rho(kl_df, "kl_post_response_slot", n_boot, seed),
            "off_ceiling_rho": _off_ceiling_rho(kl_df, "kl_post_response_slot", n_boot, seed),
        }
    else:
        out["kl_drift_secondary"] = {"error": "kl_post_response_slot column missing"}

    # Cache df for the H3 cross-arm joins.
    out["_df"] = df
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_EPOCHS),
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
    )
    ap.add_argument(
        "--g474-root",
        type=Path,
        default=CROSS_DIR_474,
        help="Root for per-(arm,ep) merged matrices.",
    )
    args = ap.parse_args(argv)

    if not D_PATH.exists():
        raise FileNotFoundError(f"#406 D_matrix.json missing at {D_PATH}")
    if not G406_PATH.exists():
        raise FileNotFoundError(f"#406 G_matrix.json missing at {G406_PATH}")

    d_payload = json.loads(D_PATH.read_text())
    g406 = json.loads(G406_PATH.read_text())

    cells: dict[str, dict] = {}
    dfs_by_cell: dict[tuple[str, int], pd.DataFrame] = {}
    for arm in args.arms:
        for epoch in args.epochs:
            arm_ep_subdir = f"{arm}_ep{epoch}"
            merged_path = args.g474_root / arm_ep_subdir / "G_logprob_matrix.json"
            per_cell_dir = args.g474_root / arm_ep_subdir / "per_cell"
            if not merged_path.exists():
                logger.warning("Missing merged matrix %s — skipping cell.", merged_path)
                cells[f"{arm}_ep{epoch}"] = {"error": f"missing {merged_path}"}
                continue
            g474 = json.loads(merged_path.read_text())
            cell = _per_cell_report(
                g474, d_payload, g406, per_cell_dir, arm, epoch, args.n_boot, args.seed
            )
            dfs_by_cell[(arm, epoch)] = cell.pop("_df")
            cells[f"{arm}_ep{epoch}"] = cell

    # Cross-arm H3 (matched-epoch).
    h3_matched_epoch: dict[str, dict] = {}
    for epoch in args.epochs:
        if ("pos", epoch) in dfs_by_cell and ("loc", epoch) in dfs_by_cell:
            h3_matched_epoch[f"ep{epoch}"] = _paired_bootstrap_arm_diff(
                dfs_by_cell[("pos", epoch)],
                dfs_by_cell[("loc", epoch)],
                "delta_g",
                args.n_boot,
                args.seed,
            )

    # Matched-step H3 (A_loc ep1 vs A_pos ep2 — A_loc has 2x rows/epoch).
    h3_matched_step: dict | None = None
    if ("loc", 1) in dfs_by_cell and ("pos", 2) in dfs_by_cell:
        h3_matched_step = _paired_bootstrap_arm_diff(
            dfs_by_cell[("pos", 2)],
            dfs_by_cell[("loc", 1)],
            "delta_g",
            args.n_boot,
            args.seed,
        )

    # #406 head-to-head DESCRIPTIVE — at the A_loc cell that wins H1
    # (or default to A_loc ep1 per plan §4.7 headline candidate).
    h2h_descriptive: dict | None = None
    h2h_anchor_cell = None
    for epoch in args.epochs:
        cell = cells.get(f"loc_ep{epoch}", {})
        if cell.get("h1_survives_stylized_mask_c"):
            h2h_anchor_cell = ("loc", epoch)
            break
    if h2h_anchor_cell is None and ("loc", 1) in dfs_by_cell:
        h2h_anchor_cell = ("loc", 1)
    if h2h_anchor_cell is not None:
        h2h_descriptive = _h2h_vs_406(dfs_by_cell[h2h_anchor_cell], args.n_boot, args.seed)
        h2h_descriptive["anchor_cell"] = f"{h2h_anchor_cell[0]}_ep{h2h_anchor_cell[1]}"

    # Reproduction tripwire: A_pos ep1 rho vs -0.27.
    tripwire: dict | None = None
    pos_ep1 = cells.get("pos_ep1", {})
    pos_ep1_partial = (
        pos_ep1.get("delta_g_three_mask_rho", {})
        .get("mask_a_all", {})
        .get("length_partial_spearman", {})
    )
    if pos_ep1_partial.get("rho_pingouin") is not None:
        rho_obs = float(pos_ep1_partial["rho_pingouin"])
        delta = rho_obs - PARENT_462_EP1_RHO
        tripwire = {
            "observed_rho_pos_ep1": rho_obs,
            "reference_462_ep1": PARENT_462_EP1_RHO,
            "delta": delta,
            "tripped": bool(abs(delta) > TRIPWIRE_DELTA_MAX),
            "max_delta_allowed": TRIPWIRE_DELTA_MAX,
            "interpretation_if_tripped": (
                "A_pos is NOT reproducing #462 ep1; cross-arm H3 read carries "
                "a strong caveat (matched-epoch / matched-step comparisons may "
                "be confounded by an A_pos-specific drift)."
            ),
        }

    out = {
        "schema_version": "i474_v1",
        "git_commit": _git_commit_hash(),
        "config": {
            "arms": args.arms,
            "epochs": args.epochs,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "stylized_sources": sorted(STYLIZED_SOURCES),
            "off_ceiling_band_nat": OFF_CEILING_NAT_BAND,
            "parent_462_ep1_rho_reference": PARENT_462_EP1_RHO,
            "tripwire_delta_max": TRIPWIRE_DELTA_MAX,
        },
        "cells": cells,
        "h3_matched_epoch_paired_bootstrap": h3_matched_epoch,
        "h3_matched_step_paired_bootstrap": h3_matched_step,
        "h2h_vs_406_descriptive": h2h_descriptive,
        "tripwire_pos_ep1_vs_462": tripwire,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    logger.info("Analysis -> %s", OUT_PATH)
    # Headline log lines.
    for cell_key, cell in cells.items():
        partial = (
            cell.get("delta_g_three_mask_rho", {})
            .get("mask_a_all", {})
            .get("length_partial_spearman", {})
            if isinstance(cell, dict)
            else {}
        )
        logger.info(
            "cell=%s mask_a_rho=%s h1_mask_c=%s h1_off_ceiling=%s h1_supp_partial=%s",
            cell_key,
            partial.get("rho_pingouin"),
            cell.get("h1_survives_stylized_mask_c") if isinstance(cell, dict) else None,
            cell.get("h1_off_ceiling_negative") if isinstance(cell, dict) else None,
            cell.get("h1_survives_suppression_partial") if isinstance(cell, dict) else None,
        )


if __name__ == "__main__":
    main()
