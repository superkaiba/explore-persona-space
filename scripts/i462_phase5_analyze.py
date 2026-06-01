"""Phase 5 (#462) — per-level saturation + rho(D, g_logprob) & rho(D, delta_g)
trajectory analysis across training amount.

Issue #462. For EACH level N in {1, 2, 3, 5}:
  - Load eval_results/issue_462/cross_eval/G_logprob_matrix_ep{N}.json
  - Build the 240-row off-diagonal frame against
    eval_results/issue_406/divergence/D_matrix.json
  - Compute SATURATION metrics:
      * off-diag g_logprob mean / sd
      * fraction of off-diag cells with |g_logprob| <= 0.1 (within 0.1 of 0)
      * fraction of off-diag cells with |g_logprob - max_g| <= 0.1
        (alternative ceiling check — clustering near the maximum)
      * diagonal implant strength: mean / min delta_g on the diagonal
  - Compute length-partial Spearman rho(D, g_logprob) and rho(D, delta_g)
    (covar = log_prompt_tokens) with cluster-bootstrap CI on class_pair.

Emit eval_results/issue_462/analysis.json with:
  - per_level: { ep1: {...}, ep2: {...}, ep3: {...}, ep5: {...} }
  - rho_vs_epoch: list of {epoch, rho_D_glogprob, rho_D_deltag, ci_low/high}
  - saturation_frac_vs_epoch: list of {epoch, frac_within_0_1_of_zero,
    frac_within_0_1_of_max, offdiag_sd, diag_mean_delta_g}

This is the DV that distinguishes "overtraining ceiling" (saturation
rises monotonically to ~1.0 by ep5 while rho collapses toward 0) from
"intrinsic-to-the-construct" (saturation rises early AND rho stays
non-negligible at every level — meaning the construct doesn't admit
ranked transfer at all).

Helpers (length-partial inline, cluster bootstrap, safe partial) are
copy-imported byte-for-byte from i460_phase5 logic to keep the two
analyses comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i462.phase5")

D_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
CROSS_DIR_462 = Path("eval_results/issue_462/cross_eval")
PER_CELL_DIR_FMT = "per_cell_ep{epoch}"
OUT_PATH = Path("eval_results/issue_462/analysis.json")

EPOCH_LEVELS = [1, 2, 3, 5]
SATURATION_BAND = 0.1  # |g_logprob - 0| <= 0.1 counts as "saturated near 0"


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


# ── Helpers ported from i460_phase5 (kept inline to avoid a circular import
# chain via scripts/, and so the file is self-contained for review). ────
def _length_partial_inline(x: pd.Series, y: pd.Series, covar: pd.Series) -> dict:
    """Rank-then-residualize length-partial Spearman (matches #340 / #406 / #460)."""
    x_rank = st.rankdata(x.to_numpy())
    y_rank = st.rankdata(y.to_numpy())
    c_rank = st.rankdata(covar.to_numpy())
    slope_x, intercept_x, _, _, _ = st.linregress(c_rank, x_rank)
    slope_y, intercept_y, _, _, _ = st.linregress(c_rank, y_rank)
    x_resid = x_rank - (slope_x * c_rank + intercept_x)
    y_resid = y_rank - (slope_y * c_rank + intercept_y)
    res = st.pearsonr(x_resid, y_resid)
    return {"r": float(res.statistic), "p": float(res.pvalue), "n": len(x_rank)}


def _safe_partial(df: pd.DataFrame, x: str, y: str, covar: str) -> dict:
    """Length-partial Spearman with both pingouin + inline implementations."""
    if len(df) < 5:
        return {
            "n": len(df),
            "rho_pingouin": None,
            "p_pingouin": None,
            "rho_inline": None,
            "p_inline": None,
            "error": "too_few_rows",
        }
    out: dict = {"n": len(df)}
    try:
        r = pg.partial_corr(data=df, x=x, y=y, covar=[covar], method="spearman")
        out["rho_pingouin"] = float(r["r"].values[0])
        out["p_pingouin"] = float(r["p_val"].values[0])
    except Exception as e:
        out["rho_pingouin"] = None
        out["p_pingouin"] = None
        out["error_pingouin"] = str(e)
    inline = _length_partial_inline(df[x], df[y], df[covar])
    out["rho_inline"] = inline["r"]
    out["p_inline"] = inline["p"]
    return out


def _cluster_bootstrap_partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covar_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Cluster-bootstrap by class_pair; length-partial Spearman per resample."""
    rng = np.random.default_rng(seed)
    cell_ids = sorted(df["class_pair"].unique())
    cell_to_rows = {cell: df.index[df["class_pair"] == cell].to_numpy() for cell in cell_ids}
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled])
        sub = df.loc[rows]
        try:
            r = pg.partial_corr(data=sub, x=x_col, y=y_col, covar=[covar_col], method="spearman")
            boot_rhos[b] = float(r["r"].values[0])
        except Exception:
            boot_rhos[b] = np.nan
    return boot_rhos


# ── Frame building (per-level) ───────────────────────────────────────
def _build_dataframe_for_epoch(g_matrix: dict, d_payload: dict, epoch: int) -> pd.DataFrame:
    """Build the 240-row off-diagonal DataFrame for one epoch level.

    Pulls per-cell prompt + R lengths from per_cell_ep{N}/ when present;
    falls back to #406's prompt_tokens (prompt only).
    """
    cond_by_cid = {c.cid: c for c in CONDITIONS}
    cids = [c.cid for c in CONDITIONS]
    KL = d_payload["KL"]
    JS = d_payload["JS"]
    PT = d_payload["prompt_tokens"]
    G460 = g_matrix["G"]
    per_cell_dir = CROSS_DIR_462 / PER_CELL_DIR_FMT.format(epoch=epoch)

    def _length_for_cell(ci: str, cj: str) -> float:
        cell_path = per_cell_dir / f"G_{ci}__{cj}.json"
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
            g460_cell = G460[ci][cj]
            g_logprob = float(g460_cell["g_logprob"])
            b_logprob = float(g460_cell["b_logprob"])
            delta_g = float(g460_cell["delta_g"])
            emission = float(g460_cell["emission_recompute_rate"])
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
                    "G_logprob": g_logprob,
                    "B_logprob": b_logprob,
                    "delta_g": delta_g,
                    "emission_recompute_rate": emission,
                    "prompt_plus_R_tokens": length,
                    "log_prompt_tokens": float(np.log(max(length, 1.0))),
                }
            )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["D"]).reset_index(drop=True)
    return df


def _saturation_metrics(g_matrix: dict, df: pd.DataFrame) -> dict:
    """Saturation diagnostics for one epoch level.

    - off-diag g_logprob mean / sd
    - frac off-diag cells with |g_logprob| <= SATURATION_BAND
    - frac off-diag cells within SATURATION_BAND of max off-diag g_logprob
    - diagonal implant strength: mean / min delta_g on the diagonal
    """
    cids = [c.cid for c in CONDITIONS]
    offdiag_g = df["G_logprob"].to_numpy()
    diag_deltas = [g_matrix["G"][ci][ci]["delta_g"] for ci in cids if ci in g_matrix["G"]]
    diag_glogprobs = [g_matrix["G"][ci][ci]["g_logprob"] for ci in cids if ci in g_matrix["G"]]

    if offdiag_g.size == 0:
        return {"error": "empty_offdiag_frame"}

    max_offdiag = float(np.max(offdiag_g))
    near_zero = (np.abs(offdiag_g) <= SATURATION_BAND).sum()
    near_max = (np.abs(offdiag_g - max_offdiag) <= SATURATION_BAND).sum()

    return {
        "n_offdiag": int(offdiag_g.size),
        "offdiag_g_logprob_mean": float(offdiag_g.mean()),
        "offdiag_g_logprob_sd": float(offdiag_g.std(ddof=1)),
        "offdiag_g_logprob_max": max_offdiag,
        "offdiag_g_logprob_min": float(offdiag_g.min()),
        "frac_within_0_1_of_zero": float(near_zero / offdiag_g.size),
        "frac_within_0_1_of_max": float(near_max / offdiag_g.size),
        "diagonal_n": len(diag_deltas),
        "diagonal_delta_g_mean": float(np.mean(diag_deltas)) if diag_deltas else None,
        "diagonal_delta_g_min": float(np.min(diag_deltas)) if diag_deltas else None,
        "diagonal_g_logprob_mean": float(np.mean(diag_glogprobs)) if diag_glogprobs else None,
    }


def _analyze_level(
    epoch: int, d_payload: dict, n_boot: int, seed: int
) -> tuple[dict, dict, dict] | None:
    """Returns (per_level_block, rho_curve_row, saturation_curve_row) or None
    if the per-epoch matrix is missing.
    """
    matrix_path = CROSS_DIR_462 / f"G_logprob_matrix_ep{epoch}.json"
    if not matrix_path.exists():
        logger.warning("ep=%d matrix missing at %s — skipping level.", epoch, matrix_path)
        return None
    g_matrix = json.loads(matrix_path.read_text())
    df = _build_dataframe_for_epoch(g_matrix, d_payload, epoch)
    logger.info("ep=%d built %d-row off-diagonal frame.", epoch, len(df))

    # Length-partial correlations
    partial_glogprob = _safe_partial(df, x="D", y="G_logprob", covar="log_prompt_tokens")
    partial_deltag = _safe_partial(df, x="D", y="delta_g", covar="log_prompt_tokens")

    # Bootstrap CIs
    boot_glogprob = _cluster_bootstrap_partial_spearman(
        df, "D", "G_logprob", "log_prompt_tokens", n_boot=n_boot, seed=seed
    )
    boot_deltag = _cluster_bootstrap_partial_spearman(
        df, "D", "delta_g", "log_prompt_tokens", n_boot=n_boot, seed=seed
    )
    partial_glogprob["bootstrap_ci_2_5"] = float(np.nanpercentile(boot_glogprob, 2.5))
    partial_glogprob["bootstrap_ci_97_5"] = float(np.nanpercentile(boot_glogprob, 97.5))
    partial_deltag["bootstrap_ci_2_5"] = float(np.nanpercentile(boot_deltag, 2.5))
    partial_deltag["bootstrap_ci_97_5"] = float(np.nanpercentile(boot_deltag, 97.5))

    saturation = _saturation_metrics(g_matrix, df)

    per_level_block = {
        "epoch": epoch,
        "n_off_diagonal_rows": len(df),
        "partial_rho_D_G_logprob": partial_glogprob,
        "partial_rho_D_delta_g": partial_deltag,
        "saturation": saturation,
        "matrix_path": str(matrix_path),
    }

    rho_curve_row = {
        "epoch": epoch,
        "rho_D_glogprob": partial_glogprob.get("rho_pingouin"),
        "rho_D_glogprob_ci_low": partial_glogprob["bootstrap_ci_2_5"],
        "rho_D_glogprob_ci_high": partial_glogprob["bootstrap_ci_97_5"],
        "rho_D_deltag": partial_deltag.get("rho_pingouin"),
        "rho_D_deltag_ci_low": partial_deltag["bootstrap_ci_2_5"],
        "rho_D_deltag_ci_high": partial_deltag["bootstrap_ci_97_5"],
    }

    saturation_curve_row = {
        "epoch": epoch,
        "frac_within_0_1_of_zero": saturation.get("frac_within_0_1_of_zero"),
        "frac_within_0_1_of_max": saturation.get("frac_within_0_1_of_max"),
        "offdiag_sd": saturation.get("offdiag_g_logprob_sd"),
        "diag_mean_delta_g": saturation.get("diagonal_delta_g_mean"),
    }

    return per_level_block, rho_curve_row, saturation_curve_row


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not D_PATH.exists():
        raise FileNotFoundError(f"#406 D_matrix.json missing at {D_PATH}")
    d_payload = json.loads(D_PATH.read_text())

    per_level: dict[str, dict] = {}
    rho_vs_epoch: list[dict] = []
    saturation_frac_vs_epoch: list[dict] = []
    missing_levels: list[int] = []

    for epoch in EPOCH_LEVELS:
        result = _analyze_level(epoch, d_payload, args.n_boot, args.seed)
        if result is None:
            missing_levels.append(epoch)
            continue
        block, rho_row, sat_row = result
        per_level[f"ep{epoch}"] = block
        rho_vs_epoch.append(rho_row)
        saturation_frac_vs_epoch.append(sat_row)

    out = {
        "schema_version": "i462_v1",
        "git_commit": _git_commit_hash(),
        "epoch_levels": EPOCH_LEVELS,
        "missing_levels": missing_levels,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "saturation_band": SATURATION_BAND,
        "per_level": per_level,
        "rho_vs_epoch": rho_vs_epoch,
        "saturation_frac_vs_epoch": saturation_frac_vs_epoch,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    logger.info("Analysis -> %s", OUT_PATH)

    # Pretty-log the trajectory curves for quick eyeball
    if rho_vs_epoch:
        logger.info("rho_D_glogprob vs epoch:")
        for row in rho_vs_epoch:
            logger.info(
                "  ep=%d rho=%s [%.3f, %.3f]",
                row["epoch"],
                f"{row['rho_D_glogprob']:.3f}" if row["rho_D_glogprob"] is not None else "None",
                row["rho_D_glogprob_ci_low"],
                row["rho_D_glogprob_ci_high"],
            )
    if saturation_frac_vs_epoch:
        logger.info("saturation_frac vs epoch:")
        for row in saturation_frac_vs_epoch:
            f0 = row["frac_within_0_1_of_zero"]
            fm = row["frac_within_0_1_of_max"]
            logger.info(
                "  ep=%d frac_near_zero=%s frac_near_max=%s offdiag_sd=%s",
                row["epoch"],
                f"{f0:.3f}" if f0 is not None else "None",
                f"{fm:.3f}" if fm is not None else "None",
                f"{row['offdiag_sd']:.3f}" if row["offdiag_sd"] is not None else "None",
            )


if __name__ == "__main__":
    main()
