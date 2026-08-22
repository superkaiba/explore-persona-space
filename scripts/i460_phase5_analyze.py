"""Phase 5 -- analysis: length-partial Spearman of delta_g vs D, H2 zero-cohort,
H3/H4 gates, head-to-head vs #406, cosine layers, plus the round-1 critic
diagnostics (per-row sd, R-length/R-perplexity partials, emission-recompute
rates on/off diagonal).

Issue #460 plan v3 §6.2 + diagnostics from the round-1 critics' must-fix
list.

Inputs:
  - eval_results/issue_460/cross_eval/G_logprob_matrix.json (this run)
  - eval_results/issue_406/divergence/D_matrix.json
  - eval_results/issue_406/cross_eval/G_matrix.json  (H2 cohort, H5 head-to-head)
  - eval_results/issue_406/cosine/C_L{0,5,11,15,21,27}.json (6 layer cosines)
  - eval_results/issue_460/cross_eval/per_cell/G_<ci>__<cj>.json
      (per-q logp arrays + prompt/R lens — descriptive diagnostics)

Output:
  - eval_results/issue_460/analysis.json (full set of stats + verdicts)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# PROD_IMPORT_LINT_EXEMPT: one-off `uv pip install pingouin` for a completed-issue analysis
import pingouin as pg
import scipy.stats as st

from explore_persona_space.experiments.i406_conditions import CONDITIONS

logger = logging.getLogger("i460.phase5")

D_PATH = Path("eval_results/issue_406/divergence/D_matrix.json")
G406_PATH = Path("eval_results/issue_406/cross_eval/G_matrix.json")
G460_PATH = Path("eval_results/issue_460/cross_eval/G_logprob_matrix.json")
COSINE_DIR = Path("eval_results/issue_406/cosine")
PER_CELL_DIR_460 = Path("eval_results/issue_460/cross_eval/per_cell")
OUT_PATH = Path("eval_results/issue_460/analysis.json")

TARGET_LAYERS = [0, 5, 11, 15, 21, 27]
H1_RHO_THRESHOLD = 0.40
H2_RHO_THRESHOLD = 0.25
H3_DELTA_G_THRESHOLD = 5.0
H4_SD_HARD_FAIL = 0.5
H4_SD_PASS_THRESHOLD = 1.5


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _length_partial_inline(x: pd.Series, y: pd.Series, covar: pd.Series) -> dict:
    """Rank-then-residualize length-partial Spearman (matches #340 / #406)."""
    x_rank = st.rankdata(x.to_numpy())
    y_rank = st.rankdata(y.to_numpy())
    c_rank = st.rankdata(covar.to_numpy())
    slope_x, intercept_x, _, _, _ = st.linregress(c_rank, x_rank)
    slope_y, intercept_y, _, _, _ = st.linregress(c_rank, y_rank)
    x_resid = x_rank - (slope_x * c_rank + intercept_x)
    y_resid = y_rank - (slope_y * c_rank + intercept_y)
    res = st.pearsonr(x_resid, y_resid)
    return {"r": float(res.statistic), "p": float(res.pvalue), "n": len(x_rank)}


def _cluster_bootstrap_partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covar_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Cluster-bootstrap by class_pair; length-partial Spearman per resample.

    Falls back to unclustered bootstrap if any class_pair has < 5 rows
    (e.g. C-as-singleton off-diagonal cells).
    """
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
    out = {"n": len(df)}
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


def _raw_spearman_and_pearson(df: pd.DataFrame, x: str, y: str) -> dict:
    """Both Spearman (rank) AND Pearson (linear) on raw (unpartialed) data."""
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


def _load_cosine_layer(layer: int) -> dict[str, dict[str, float | None]]:
    path = COSINE_DIR / f"C_L{layer}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cosine layer {layer} missing at {path}")
    payload = json.loads(path.read_text())
    # The cosine matrices are wrapped per #406 convention.
    if "matrix" in payload:
        return payload["matrix"]
    return payload


def _build_dataframe(g460: dict, d_payload: dict, g406_payload: dict) -> pd.DataFrame:
    """Build the 240-row off-diagonal DataFrame for analysis.

    Each row: T_i, T_j, class_i, class_j, class_pair, D (KL K=25-mean),
    JS, G_orig (#406 binary rate), G_logprob, B_logprob, delta_g,
    delta_g_trimmed, log_prompt_tokens.
    """
    cond_by_cid = {c.cid: c for c in CONDITIONS}
    cids = [c.cid for c in CONDITIONS]
    KL = d_payload["KL"]
    JS = d_payload["JS"]
    PT = d_payload["prompt_tokens"]
    G_orig = g406_payload["G"]
    G460 = g460["G"]

    # Per-cell artifacts for log_prompt_tokens (mean over q of len(prompt) + len(R)).
    # Use the per_cell payloads when present (Phase 4 saves them); otherwise
    # fall back to #406's prompt_tokens (which is len(prompt) only, no R).
    def _length_for_cell(ci: str, cj: str) -> float:
        cell_path = PER_CELL_DIR_460 / f"G_{ci}__{cj}.json"
        if cell_path.exists():
            cell = json.loads(cell_path.read_text())
            prompt_lens = cell.get("prompt_lens_per_q", [])
            R_lens = cell.get("R_lens_per_q", [])
            if prompt_lens and R_lens:
                return float(np.mean([p + r for p, r in zip(prompt_lens, R_lens, strict=True)]))
        # Fallback to #406's prompt_tokens (prompt only).
        return float(PT[ci][cj])

    rows = []
    for ci in cids:
        ci_cls = cond_by_cid[ci].cls
        for cj in cids:
            if ci == cj:
                continue  # off-diagonal only
            cj_cls = cond_by_cid[cj].cls
            kl = KL[ci][cj]
            js = JS[ci][cj]
            g_orig_cell = G_orig[ci][cj]
            if g_orig_cell is None:
                continue
            g_orig_rate = float(g_orig_cell["rate"])
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
                    "G_orig": g_orig_rate,
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


def _per_row_diagnostics(g460: dict) -> dict:
    """For each outer-i, sd of delta_g across off-diagonal j. Flag rows
    with off-diag sd < 0.5 (constant-emission row-saturation per round-1
    critic concern).
    """
    cids = [c.cid for c in CONDITIONS]
    out = {}
    flagged = []
    for ci in cids:
        deltas = [
            g460["G"][ci][cj]["delta_g"]
            for cj in cids
            if cj != ci and ci in g460["G"] and cj in g460["G"][ci]
        ]
        if not deltas:
            continue
        sd = float(np.std(deltas, ddof=1))
        mean = float(np.mean(deltas))
        out[ci] = {"sd_offdiag": sd, "mean_offdiag": mean, "n": len(deltas)}
        if sd < 0.5:
            flagged.append(ci)
    return {"per_row": out, "row_saturation_flagged": flagged}


def _r_property_diagnostics(df: pd.DataFrame) -> dict:
    """Descriptive partials: rho(delta_g, R_length) and (if computable)
    rho(delta_g, R_perplexity_under_base).

    R_perplexity_under_base proxy: -mean(b_logps_per_q) per cell — readable
    from per_cell JSONs and joined here as an aggregate per-cell statistic.
    """
    cell_R_lens = {}
    cell_R_perplexity = {}
    for _, row in df.iterrows():
        cell_path = PER_CELL_DIR_460 / f"G_{row['T_i']}__{row['T_j']}.json"
        if not cell_path.exists():
            continue
        cell = json.loads(cell_path.read_text())
        if cell.get("R_lens_per_q"):
            cell_R_lens[(row["T_i"], row["T_j"])] = float(np.mean(cell["R_lens_per_q"]))
        if cell.get("b_logps_per_q"):
            # higher perplexity ~ lower mean logprob (negated)
            cell_R_perplexity[(row["T_i"], row["T_j"])] = -float(np.mean(cell["b_logps_per_q"]))

    if not cell_R_lens:
        return {"error": "no per-cell artifacts available"}

    df2 = df.copy()
    df2["R_length"] = df2.apply(lambda r: cell_R_lens.get((r["T_i"], r["T_j"]), np.nan), axis=1)
    df2["R_perplexity"] = df2.apply(
        lambda r: cell_R_perplexity.get((r["T_i"], r["T_j"]), np.nan), axis=1
    )
    df_use = df2.dropna(subset=["R_length"]).reset_index(drop=True)
    out = {
        "rho_delta_R_length": _raw_spearman_and_pearson(df_use, "delta_g", "R_length"),
        "rho_delta_R_perplexity": _raw_spearman_and_pearson(
            df_use.dropna(subset=["R_perplexity"]), "delta_g", "R_perplexity"
        ),
        # R-length partial (delta vs D, controlling for R_length): does H1
        # survive partialing out R-property structure?
        "delta_vs_D_partial_Rlen": _safe_partial(df_use, "D", "delta_g", "R_length"),
    }
    return out


def _emission_rate_summary(g460: dict) -> dict:
    """Compute emission-rate-recompute means on diagonal vs off-diagonal cells."""
    cids = [c.cid for c in CONDITIONS]
    diag_rates = []
    offdiag_rates = []
    for ci in cids:
        for cj in cids:
            if ci not in g460["G"] or cj not in g460["G"][ci]:
                continue
            er = g460["G"][ci][cj].get("emission_recompute_rate")
            if er is None:
                continue
            if ci == cj:
                diag_rates.append(er)
            else:
                offdiag_rates.append(er)
    return {
        "n_diagonal": len(diag_rates),
        "n_offdiagonal": len(offdiag_rates),
        "diagonal_mean": float(np.mean(diag_rates)) if diag_rates else None,
        "diagonal_min": float(np.min(diag_rates)) if diag_rates else None,
        "offdiagonal_mean": float(np.mean(offdiag_rates)) if offdiag_rates else None,
        "offdiagonal_max": float(np.max(offdiag_rates)) if offdiag_rates else None,
    }


def _head_to_head_vs_406(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap CI on the difference in length-partial rho:
    rho(delta_g, D) vs rho(G_orig, D), resampled by class_pair.
    """
    rng = np.random.default_rng(seed)
    cell_ids = sorted(df["class_pair"].unique())
    cell_to_rows = {cell: df.index[df["class_pair"] == cell].to_numpy() for cell in cell_ids}
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(cell_ids), size=len(cell_ids), replace=True)
        rows = np.concatenate([cell_to_rows[cell_ids[k]] for k in sampled])
        sub = df.loc[rows]
        try:
            r460 = pg.partial_corr(
                data=sub, x="D", y="delta_g", covar=["log_prompt_tokens"], method="spearman"
            )
            r406 = pg.partial_corr(
                data=sub, x="D", y="G_orig", covar=["log_prompt_tokens"], method="spearman"
            )
            boot_diffs[b] = float(r460["r"].values[0]) - float(r406["r"].values[0])
        except Exception:
            boot_diffs[b] = np.nan
    return {
        "ci_2_5": float(np.nanpercentile(boot_diffs, 2.5)),
        "ci_97_5": float(np.nanpercentile(boot_diffs, 97.5)),
        "mean_diff": float(np.nanmean(boot_diffs)),
        "n_boot": n_boot,
    }


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

    if not G460_PATH.exists():
        raise FileNotFoundError(f"#460 G_logprob_matrix.json missing at {G460_PATH}")
    if not D_PATH.exists():
        raise FileNotFoundError(f"#406 D_matrix.json missing at {D_PATH}")
    if not G406_PATH.exists():
        raise FileNotFoundError(f"#406 G_matrix.json missing at {G406_PATH}")

    g460 = json.loads(G460_PATH.read_text())
    d_payload = json.loads(D_PATH.read_text())
    g406 = json.loads(G406_PATH.read_text())

    df = _build_dataframe(g460, d_payload, g406)
    logger.info("Built %d-row off-diagonal DataFrame.", len(df))

    # H1 -- length-partial rho(delta_g, D) on all 240 pairs.
    h1 = _safe_partial(df, x="D", y="delta_g", covar="log_prompt_tokens")
    h1_boot = _cluster_bootstrap_partial_spearman(
        df, "D", "delta_g", "log_prompt_tokens", n_boot=args.n_boot, seed=args.seed
    )
    h1["bootstrap_ci_2_5"] = float(np.nanpercentile(h1_boot, 2.5))
    h1["bootstrap_ci_97_5"] = float(np.nanpercentile(h1_boot, 97.5))
    h1_raw = _raw_spearman_and_pearson(df, "D", "delta_g")
    rho_pin = h1.get("rho_pingouin")
    h1_pass = (
        rho_pin is not None
        and abs(rho_pin) >= H1_RHO_THRESHOLD
        and rho_pin < 0
        and h1["bootstrap_ci_97_5"] < 0
    )

    # H2 -- zero-cohort restricted rho on cells where #406 G_orig == 0.
    df_zero = df[df["G_orig"] == 0.0].reset_index(drop=True)
    h2 = _safe_partial(df_zero, x="D", y="delta_g", covar="log_prompt_tokens")
    if len(df_zero) >= 10:
        h2_boot = _cluster_bootstrap_partial_spearman(
            df_zero, "D", "delta_g", "log_prompt_tokens", n_boot=args.n_boot, seed=args.seed
        )
        h2["bootstrap_ci_2_5"] = float(np.nanpercentile(h2_boot, 2.5))
        h2["bootstrap_ci_97_5"] = float(np.nanpercentile(h2_boot, 97.5))
    h2_rho = h2.get("rho_pingouin")
    h2_pass = h2_rho is not None and abs(h2_rho) >= H2_RHO_THRESHOLD

    # H3 — diagonal implant gate.
    diagonals_delta = g460["diagonals"]
    h3_failed = [ci for ci, d in diagonals_delta.items() if d <= H3_DELTA_G_THRESHOLD]
    h3_pass = len(h3_failed) == 0

    # H4 — sd(delta_g) across 240 off-diagonal pairs.
    sd_delta = float(np.std(df["delta_g"], ddof=1))
    if sd_delta < H4_SD_HARD_FAIL:
        h4_verdict = "FAIL_HARD"
    elif sd_delta < H4_SD_PASS_THRESHOLD:
        h4_verdict = "SOFT_WARN"
    else:
        h4_verdict = "PASS"

    # H5 — head-to-head vs #406's binary G.
    ph_orig = _safe_partial(df, x="D", y="G_orig", covar="log_prompt_tokens")
    h2h = _head_to_head_vs_406(df, n_boot=args.n_boot, seed=args.seed)

    # Aggregation robustness.
    raw_g = _safe_partial(df, x="D", y="G_logprob", covar="log_prompt_tokens")
    placebo = _safe_partial(df, x="D", y="B_logprob", covar="log_prompt_tokens")

    # Cosine layers (6 of them) on delta_g.
    cosine_layers = {}
    for layer in TARGET_LAYERS:
        try:
            cos_matrix = _load_cosine_layer(layer)
            df_l = df.copy()
            # Bind cos_matrix to the lambda via default arg to silence B023.
            df_l["C"] = df_l.apply(
                lambda r, cm=cos_matrix: cm.get(r["T_i"], {}).get(r["T_j"]),
                axis=1,
            )
            df_l_use = df_l.dropna(subset=["C"]).reset_index(drop=True)
            if len(df_l_use) >= 5:
                cosine_layers[f"L{layer}"] = _safe_partial(
                    df_l_use, x="C", y="delta_g", covar="log_prompt_tokens"
                )
            else:
                cosine_layers[f"L{layer}"] = {"n": len(df_l_use), "error": "too_few_rows"}
        except Exception as e:
            cosine_layers[f"L{layer}"] = {"error": str(e)}

    # Round-1 diagnostics.
    per_row = _per_row_diagnostics(g460)
    r_properties = _r_property_diagnostics(df)
    emission_summary = _emission_rate_summary(g460)

    # Robustness to near-zero-KL pairs.
    sensitivity = {}
    for thr in (0.05, 0.10, 0.20):
        sub = df[df["D"] >= thr].reset_index(drop=True)
        sensitivity[f"D_ge_{thr}"] = {
            "n": len(sub),
            **_safe_partial(sub, x="D", y="delta_g", covar="log_prompt_tokens"),
        }

    # Non-zero-cohort separation (115 cells where #406 G_orig > 0).
    df_nonzero = df[df["G_orig"] > 0.0].reset_index(drop=True)
    nonzero_cohort = _safe_partial(df_nonzero, x="D", y="delta_g", covar="log_prompt_tokens")
    nonzero_cohort["n"] = len(df_nonzero)

    out = {
        "schema_version": "i460_v1",
        "git_commit": _git_commit_hash(),
        "n_off_diagonal_rows": len(df),
        "thresholds": {
            "H1_rho_min": H1_RHO_THRESHOLD,
            "H2_rho_min": H2_RHO_THRESHOLD,
            "H3_delta_g_min": H3_DELTA_G_THRESHOLD,
            "H4_sd_hard_fail": H4_SD_HARD_FAIL,
            "H4_sd_pass": H4_SD_PASS_THRESHOLD,
        },
        "H1_length_partial_delta_vs_D": h1,
        "H1_raw_spearman_pearson": h1_raw,
        "H1_pass": bool(h1_pass),
        "H2_zero_cohort": {
            "n": len(df_zero),
            **h2,
            "pass": bool(h2_pass),
        },
        "H3_diagonals": diagonals_delta,
        "H3_failed_conds": h3_failed,
        "H3_pass": bool(h3_pass),
        "H4_sd_delta_g": sd_delta,
        "H4_verdict": h4_verdict,
        "H5_head_to_head_vs_406": {
            "rho_orig": ph_orig,
            "paired_boot_diff_CI": h2h,
        },
        "raw_g_logprob_partial": raw_g,
        "placebo_b_logprob_partial": placebo,
        "cosine_layer_partials": cosine_layers,
        "per_row_diagnostics": per_row,
        "r_property_diagnostics": r_properties,
        "emission_rate_summary": emission_summary,
        "near_zero_KL_sensitivity": sensitivity,
        "non_zero_cohort_separation": nonzero_cohort,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    logger.info("Analysis -> %s", OUT_PATH)
    logger.info(
        "H1 rho=%s pass=%s | H2 n=%d rho=%s pass=%s | H3 failed=%s | H4 sd=%.3f verdict=%s",
        h1.get("rho_pingouin"),
        h1_pass,
        len(df_zero),
        h2.get("rho_pingouin"),
        h2_pass,
        h3_failed,
        sd_delta,
        h4_verdict,
    )


if __name__ == "__main__":
    main()
