#!/usr/bin/env python
"""Issue #1482: FULL-WIDTH per-feature predictor reads + two sub-analyses.

Scope after the 2026-08-03 directive chain (full width is the ONLY published
grain; panel-width figures are no longer a deliverable):

  1. A durable, target-INDEPENDENT full-width covariate matrix (131,072
     features) with a stable schema + legend, so every R^2 array — the
     provisional #1738 SAE->SAE one, task #7's forthcoming dense->SAE ridge and
     MLP arrays, and any later refit — joins against the SAME covariates by
     `feat_ids` with no recomputation.
  2. `input_output_spectrum`: is output-ness a CONTINUOUS property or a BINARY
     one? Test 1 (modality) needs no R^2 and is DEFINITIVE at full width; tests
     2 (gradient vs step) and 3 (does the binary class add anything given the
     continuous measure) need an R^2 array and are PROVISIONAL.
  3. `activity_decile_profiles`: within-activity-decile rho for every predictor
     — the effect-modification read that a single activity-partialled rho
     averages away.

TARGET STATUS — read before quoting any number that depends on R^2:
  The CORRECT target is the full-width dense->SAE map (task #7's driver,
  `scripts/issue1482_densesae_fullwidth.py`, owned by another agent). Those
  arrays DO NOT EXIST YET (no pod provisioned for #1482 as of this run). Every
  R^2-dependent read here is computed against the #1738 SAE->SAE array and
  labelled PROVISIONAL in the JSON, the figure captions and the report.
  Justification for reading it at all: the two arms rank features at rho =
  0.933 at panel width, so SHAPE conclusions (unimodal vs bimodal, gradient vs
  step, does the class add anything, flat vs sloped decile profile) are robust
  to a 0.93-correlated target swap even though LEVELS are not. NEVER present a
  provisional number as the dense->SAE result. Every R^2-dependent figure
  re-renders against #7's arrays by swapping `--r2-npy`.

  DEPENDENCY RISK: the linear-vs-nonlinear comparison now rests ENTIRELY on the
  pod grid's full-width MLP cells succeeding. If they fail or are descoped there
  is no nonlinear arm at any published grain. Recorded, not solved.

PENDING COVARIATES: `mean_run_length`, `template_token_frac` come from task #7's
pod capture leg (`eval_results/issue_1482/run_length/run_length_perfeature.npz`),
which needs per-token encoding this module cannot do. They are declared as SLOTS
with `feat_ids` as the documented join key and are picked up automatically once
the file lands. `mean_run_length` and `p` are the same quantity (E[R] = 1/(1-p)),
so only `mean_run_length` is ever plotted.

NAMING: `scaffold_frac` is GEOMETRIC — the share of a feature's DECODER-vector
mass in the top-48 eigen-subspace of the prefix-state covariance (#1773
L475-486). It says nothing about which tokens the feature fires on. The
BEHAVIOURAL twin is `template_token_frac` from the pod leg above.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy / torch (shared-VM discipline)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_dense_predictor_reads as DR  # noqa: E402
import issue1482_predictor_battery as PB  # noqa: E402
import issue1482_predictor_battery_fullwidth as FW  # noqa: E402

DICT_SIZE = FW.DICT_SIZE
SEED = 1482
N_BOOT = 2000
N_DECILES = 10
BOOT_CHUNK = 100
KURT_Q = VAR_Q = 0.9  # #1482 footprint_moments L57-58 (the binary class thresholds)

MATRIX = "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
FOOTPRINT = "eval_results/issue_1482/footprint_moments/footprint_moments.npz"
HOLDOUT_VAR = "eval_results/issue_1482/sae_perfeature/holdout_feature_variance.npy"
R2_PROVISIONAL = "eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy"
RUN_LENGTH = "eval_results/issue_1482/run_length/run_length_perfeature.npz"
OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

R2_DISPLAY_FLOOR = -1.0
X_VIEW_PCT = 0.5

# key -> (label, grain). grain: "fullwidth" | "pending-pod"
COVARIATES: dict[str, tuple[str, str]] = {
    "mean_act_uncond": ("mean activation over all answers", "fullwidth"),
    "firing_freq_per_token": ("firing frequency (per token)", "fullwidth"),
    "activity": ("firing frequency (per answer)", "fullwidth"),
    "side_ratio": ("answer-side share of firings", "fullwidth"),
    "scaffold_frac": ("decoder mass in prefix-covariance top-48 subspace (geometric)", "fullwidth"),
    "redundancy_max_cos": ("max cosine to another decoder column", "fullwidth"),
    "act_var_across_answers": ("activation variance across answers", "fullwidth"),
    "proj_var": ("dense variance along decoder direction", "fullwidth"),
    "enc_norm": ("encoder-vector norm", "fullwidth"),
    "logit_footprint_concentration": ("positive-logit mass on top-10 tokens", "fullwidth"),
    "massive_dim_mass": ("decoder mass on massive-activation dims", "fullwidth"),
    "consistency": ("within-answer consistency", "fullwidth"),
    "write_norm": ("write norm (gamma-scaled)", "fullwidth"),
    "mean_act_cond": ("mean activation when active", "fullwidth"),
    "enc_dec_cos": ("encoder-decoder cosine", "fullwidth"),
    "footprint_var": ("footprint variance", "fullwidth"),
    "footprint_skew": ("footprint skew", "fullwidth"),
    "footprint_kurt": ("footprint kurtosis", "fullwidth"),
    "n_active_holdout": ("holdout answers active (count)", "fullwidth"),
    "dec_norm": ("decoder-column norm", "fullwidth"),
    # SLOTS — task #7 pod capture leg; join key `feat_ids`. `p` is deliberately
    # absent: E[R] = 1/(1-p) is the same quantity as mean_run_length.
    "mean_run_length": ("mean activation run length (tokens)", "pending-pod"),
    "template_token_frac": ("fraction of firings on template tokens (behavioural)", "pending-pod"),
}
RUN_LENGTH_SLOTS = {
    "mean_run_length": "mean_run_length",
    "template_token_frac": "template_token_frac",
}

DEGENERATE_CANDIDATES = ("dec_norm",)
PARTIAL_ON = "activity"

# The continuous "output-ness" measures the binary promoting/suppressing class
# dichotomizes. Modality of these IS the user's continuous-vs-binary question.
OUTPUTNESS = (
    "footprint_kurt",
    "footprint_skew",
    "logit_footprint_concentration",
    "write_norm",
    "enc_dec_cos",
)


def _log(msg: str) -> None:
    print(f"[fullwidth-reads] {msg}", flush=True)


# ── full-width covariate assembly ────────────────────────────────────────────


def logit_concentration_fullwidth(w_dec: np.ndarray, work: Path, chunk: int = 1024) -> np.ndarray:
    """Share of POSITIVE direct-logit mass on the top-10 promoted tokens.

    #1773 L256-284 verbatim (top-10, positive mass only) so this recompute and
    the banked panel values are comparable. Measured 4.7 s per 1024-feature
    chunk on this box -> ~10 min full width; cached.
    """
    cache = work / "logit_concentration_fullwidth.npy"
    if cache.exists():
        return np.load(cache)
    import issue1773_phase0_mechanical as M

    scratch = work / "wu_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    w_u, gamma = M._load_lm_head_and_gamma(scratch)
    scaled = (w_dec * gamma[:, None]).astype(np.float32)
    n_feat = w_dec.shape[1]
    # 128 chunks over a shared, contended box: checkpoint per chunk so a kill
    # costs one chunk, not the whole ~10-40 min pass (code-style.md intra-phase
    # grain, T2 unit-count trigger), and print one progress line per chunk so
    # the phase is observable rather than just alive.
    part = work / "logit_concentration_fullwidth.part.npz"
    out = np.zeros(n_feat, dtype=np.float64)
    start = 0
    if part.exists():
        with np.load(part) as z:
            out = z["out"]
            start = int(z["done"])
        _log(f"  concentration resume at feature {start}/{n_feat}")
    t0 = time.time()
    for s in range(start, n_feat, chunk):
        logits = w_u @ scaled[:, s : s + chunk]
        top = np.partition(logits, -10, axis=0)[-10:]
        pos_mass = np.where(logits > 0, logits, 0.0).sum(0)
        top_pos = np.where(top > 0, top, 0.0).sum(0)
        out[s : s + chunk] = top_pos / np.maximum(pos_mass, 1e-12)
        done = min(s + chunk, n_feat)
        tmp = part.with_suffix(".tmp.npz")
        np.savez(tmp, out=out, done=done)
        tmp.replace(part)
        _log(f"  concentration {done}/{n_feat} elapsed={time.time() - t0:.0f}s")
    np.save(cache, out)
    part.unlink(missing_ok=True)
    _log(f"concentration done in {time.time() - t0:.0f}s -> {cache}")
    return out


def redundancy_fullwidth(w_dec: np.ndarray, work: Path) -> np.ndarray:
    """Max cosine to any OTHER decoder column, via the tested blocked routine.

    Reuses `issue1773_phase0_mechanical.neighbor_table_blocked` verbatim rather
    than reimplementing: it never materializes the (n x n) Gram (~68 TB at
    n=131,072). ~123 TFLOP; ~9 min at this box's measured 238 GFLOP/s. Cached.
    """
    cache = work / "redundancy_max_cos_fullwidth.npy"
    if cache.exists():
        return np.load(cache)
    import issue1773_phase0_mechanical as M

    t0 = time.time()
    _, nb_cos = M.neighbor_table_blocked(np.ascontiguousarray(w_dec), "cpu")
    out = np.asarray(nb_cos[:, 0], dtype=np.float64)
    np.save(cache, out)
    _log(f"redundancy done in {time.time() - t0:.0f}s -> {cache}")
    return out


def classify_promoting(kurt: np.ndarray, skew: np.ndarray, var: np.ndarray) -> np.ndarray:
    """0=other, 1=promoting, 2=suppressing, 3=partition — #1482 footprint L157-165."""
    kurt_hi = kurt > np.quantile(kurt, KURT_Q)
    var_hi = var > np.quantile(var, VAR_Q)
    cls = np.zeros(len(kurt), dtype=np.int8)
    cls[kurt_hi & (skew > 0)] = 1
    cls[kurt_hi & (skew < 0)] = 2
    cls[~kurt_hi & var_hi] = 3
    return cls


def act_var_bank_check(scan: dict, cov: dict) -> dict:
    """Integrity finding, carried forward: the banked 0.0203 is a DIFFERENT quantity.

    `feature_correlates/dense_projection_variance.json` banks Spearman(panel
    R^2, log feat_var) = 0.020272 against the committed 16,384-wide
    `holdout_feature_variance.npy`, and that number was used as an identity
    check for `act_var_across_answers`. It is not one.
    """
    banked = np.asarray(np.load(PROJECT_ROOT / HOLDOUT_VAR), dtype=np.float64)
    with np.load(PROJECT_ROOT / PB.PERFEATURE) as z:
        pid = np.asarray(z["feat_ids"], dtype=int)
        panel_r2 = np.asarray(z["r2"], dtype=np.float64)
    # The FUSED scan carries holdout COUNTS only; the round-1 scan cache carries
    # the holdout activation moments this check needs (same store, same
    # definition), so read them from there rather than re-walking 1,920 shards.
    if "sum_mean_holdout" in scan:
        src, n_ho = scan, float(scan["n_holdout"])
    else:
        cache = PROJECT_ROOT / "data/issue_1482/fullwidth/continuous_scan.npz"
        if not cache.exists():
            return {
                "status": "SKIPPED — holdout activation moments unavailable",
                "cache": str(cache),
            }
        with np.load(cache) as z:
            src = {k: z[k] for k in ("sum_mean_holdout", "sum_mean_sq_holdout", "n_holdout")}
        n_ho = float(src["n_holdout"])
    mu_ho = src["sum_mean_holdout"] / n_ho
    var_ho = np.maximum(src["sum_mean_sq_holdout"] / n_ho - mu_ho**2, 0.0)
    mine_ho, mine_fit = var_ho[pid], cov["act_var_across_answers"][pid]
    qs = [0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    out = {
        "banked_value_from_json": 0.020272182358016706,
        "bank_reproduced_spearman_panel_r2_vs_banked": PB._spearman(panel_r2, banked),
        "holdout_recompute_spearman_vs_banked": PB._spearman(mine_ho, banked),
        "fit_variant_spearman_panel_r2": PB._spearman(panel_r2, mine_fit),
        "fit_vs_holdout_variant_spearman": PB._spearman(mine_fit, mine_ho),
        "banked_quantiles": [float(np.quantile(banked, q)) for q in qs],
        "holdout_recompute_quantiles": [float(np.quantile(mine_ho, q)) for q in qs],
        "verdict": "MISMATCH — the banked array is a different quantity, not this covariate",
        "note": (
            "The read of the banked array REPRODUCES 0.020272 exactly, so the bank's own number "
            "is confirmed — but it does NOT validate this covariate. The recomputed holdout-row "
            "variance correlates only rho=+0.016 with the banked array, while this covariate's "
            "fit and holdout variants agree with EACH OTHER at rho=+0.986, so the disagreement "
            "is not a row-set effect. No definition formable from the pooled store tracks the "
            "banked array, and the distributions differ ~67x at the median with a non-constant "
            "ratio, ruling out a rescaling or a reordering. The banked array came from a "
            "transient bridge-staging memmap (work/Ycat.f32.mm) no longer on disk, so what it "
            "measures cannot be adjudicated from committed artifacts. CONSEQUENCE: the 0.0203 "
            "figure is NOT a valid identity check for act_var_across_answers, and the "
            "variance_vs_r2.json takeaway ('variance does NOT organize per-feature "
            "predictability') is contradicted by this covariate's realized reads."
        ),
    }
    _log(f"act_var bank check: {out['verdict']}")
    return out


def assemble_fullwidth(work: Path, workers: int) -> dict:
    """Every full-width covariate, on the DICT_SIZE index, plus gates."""
    scan = DR.fused_scan(DR.POOLED_STORE, work / "fused_scan.npz", workers)

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=PB.SAE_K, layer=PB.SAE_LAYER, device="cpu")
    w_dec, w_enc = np.asarray(sae.w_dec), np.asarray(sae.w_enc)
    cov = DR.derive_covariates(scan, w_dec, w_enc)
    cov["enc_norm"] = cov.pop("enc_norm_recomputed")
    cov["massive_dim_mass"] = cov.pop("massive_dim_mass_fullwidth")
    cov["logit_footprint_concentration"] = logit_concentration_fullwidth(w_dec, work)
    cov["redundancy_max_cos"] = redundancy_fullwidth(w_dec, work)
    del sae, w_dec, w_enc

    with np.load(PROJECT_ROOT / FOOTPRINT) as z:
        cov["write_norm"] = np.asarray(z["direct_write_norm"], dtype=np.float64)
        cov["footprint_var"] = np.asarray(z["direct_var"], dtype=np.float64)
        cov["footprint_skew"] = np.asarray(z["direct_skew"], dtype=np.float64)
        cov["footprint_kurt"] = np.asarray(z["direct_kurt"], dtype=np.float64)
    cov["promoting_class"] = classify_promoting(
        cov["footprint_kurt"], cov["footprint_skew"], cov["footprint_var"]
    ).astype(np.float64)

    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        uni = np.asarray(z["feat_ids"], dtype=np.int64)
        pv = np.full(DICT_SIZE, np.nan)
        pv[uni] = np.asarray(z["proj_var"], dtype=np.float64)
    cov["proj_var"] = pv

    gates = DR.identity_gates(cov, uni)
    gates["side_ratio_census"] = DR.side_ratio_gate(scan, cov)
    gates["act_var_bank_check"] = act_var_bank_check(scan, cov)

    pending: list[str] = []
    rl = PROJECT_ROOT / RUN_LENGTH
    if rl.exists():
        with np.load(rl) as z:
            rl_ids = np.asarray(z["feat_ids"], dtype=np.int64)
            for key, col in RUN_LENGTH_SLOTS.items():
                if col in z.files:
                    full = np.full(DICT_SIZE, np.nan)
                    full[rl_ids] = np.asarray(z[col], dtype=np.float64)
                    cov[key] = full
                else:
                    pending.append(key)
        _log(f"run-length artifact joined on feat_ids ({len(rl_ids)} rows)")
    else:
        pending = list(RUN_LENGTH_SLOTS)
        _log(f"run-length artifact ABSENT ({RUN_LENGTH}) — {len(pending)} covariates pending")

    return {"cov": cov, "gates": gates, "universe": uni, "pending": pending, "scan": scan}


# ── reads ────────────────────────────────────────────────────────────────────


def _rank_cols(mat: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([PB._rank(v) for v in mat])


def _boot_raw_partial(z: np.ndarray, a: int, y: int, n_boot: int, rng) -> tuple:
    """Batched raw + partial rho draws — four GEMMs per chunk, no per-draw loop."""
    n = z.shape[0]
    raw_d, par_d = [], []
    done = 0
    while done < n_boot:
        b = min(BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(n, b))
        w = np.bincount((idx + np.arange(b) * n).ravel(), minlength=n * b)
        w = w.reshape(b, n).T.astype(np.float64)
        s1, s2 = z.T @ w, (z * z).T @ w
        cy = (z * z[:, y : y + 1]).T @ w
        ca = (z * z[:, a : a + 1]).T @ w
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r_jy = (cy / n - mu * mu[y]) / (sd * sd[y])
            r_ja = (ca / n - mu * mu[a]) / (sd * sd[a])
        raw_d.append(r_jy.T)
        par_d.append(DR._partial(r_jy, r_ja, r_jy[a]).T)
        done += b
    return np.vstack(raw_d), np.vstack(par_d)


def main_reads(cov: dict, r2: np.ndarray, keys: list[str], n_boot: int, rng) -> dict:
    ok = np.isfinite(r2)
    for k in keys:
        ok &= np.isfinite(cov[k])
    n_used = int(ok.sum())
    _log(f"main reads: {n_used} features finite across all {len(keys)} predictors + R^2")

    a, y = keys.index(PARTIAL_ON), len(keys)
    z = np.column_stack([*[PB._rank(cov[k][ok]) for k in keys], PB._rank(r2[ok])])
    zc = (z - z.mean(0)) / z.std(0)
    corr = (zc.T @ zc) / n_used
    raw = corr[:, y]
    par = DR._partial(corr[:, y], corr[:, a], corr[a, y])
    par[a] = np.nan
    t0 = time.time()
    raw_b, par_b = _boot_raw_partial(z, a, y, n_boot, rng)
    _log(f"  bootstrap {n_boot} draws in {time.time() - t0:.0f}s")

    rows = []
    for j, k in enumerate(keys):
        rows.append(
            {
                "key": k,
                "label": COVARIATES[k][0],
                "grain": COVARIATES[k][1],
                "spearman_raw": float(raw[j]),
                "spearman_raw_ci95": PB._ci(raw_b[:, j]),
                "partial_on_activity": (None if j == a else float(par[j])),
                "partial_on_activity_ci95": (None if j == a else PB._ci(par_b[:, j])),
                "log_x_axis": bool(_log_x(cov[k][ok])),
                **_decile_profile(cov[k][ok], r2[ok]),
            }
        )
        _log(
            f"{k}: raw {raw[j]:+.3f} partial {par[j]:+.3f}"
            if j != a
            else f"{k}: raw {raw[j]:+.3f} (partial n/a — self)"
        )
    return {"predictors": rows, "n_used": n_used, "mask": ok}


def _log_x(v: np.ndarray) -> bool:
    f = v[np.isfinite(v)]
    if len(f) == 0 or not (f > 0).all():
        return False
    p50, p99 = np.percentile(f, [50, 99])
    return bool(p50 > 0 and p99 / p50 >= 5.0)


def _decile_profile(pred: np.ndarray, r2: np.ndarray) -> dict:
    edges = np.quantile(pred, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], pred, side="right")
    med = [
        float(np.median(r2[dec == d])) if (dec == d).any() else float("nan")
        for d in range(N_DECILES)
    ]
    return {
        "decile_median_r2": med,
        "decile_center": [float((edges[i] + edges[i + 1]) / 2) for i in range(N_DECILES)],
        "decile_n": [int((dec == d).sum()) for d in range(N_DECILES)],
    }


# ── sub-analysis 1: input/output — continuous spectrum or binary class? ──────


def _modality(v: np.ndarray, log_x: bool, n_grid: int = 512) -> dict:
    """Count KDE modes over the full-width distribution.

    Gaussian KDE on a fixed grid (Silverman bandwidth), modes = local maxima
    whose height exceeds 5% of the global peak. Stated plainly rather than a
    dip test so the criterion is inspectable; the mode locations are reported
    so a reader can disagree with the threshold.
    """
    x = v[np.isfinite(v)]
    if log_x:
        x = np.log10(x[x > 0])
    x = x[np.isfinite(x)]
    sub = x if len(x) <= 20000 else np.random.default_rng(SEED).choice(x, 20000, replace=False)
    sd = float(sub.std())
    if sd <= 0:
        return {"n_modes": 0, "verdict": "degenerate (zero variance)", "modes": []}
    bw = 1.06 * sd * len(sub) ** (-1 / 5)
    grid = np.linspace(sub.min(), sub.max(), n_grid)
    d = np.exp(-0.5 * ((grid[:, None] - sub[None, :]) / bw) ** 2).sum(1)
    d /= d.max()
    peak = (d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > 0.05)
    modes = grid[1:-1][peak]
    view = [float(np.percentile(sub, 0.5)), float(np.percentile(sub, 99.5))]
    return {
        "n_modes": int(len(modes)),
        "verdict": "unimodal" if len(modes) <= 1 else f"multimodal ({len(modes)} modes)",
        "modes": [float(m) for m in modes],
        "view_range_p0p5_p99p5": view,
        "log10_x": bool(log_x),
        "bandwidth": float(bw),
        "kde_grid": [float(g) for g in grid],
        "kde_density": [float(y) for y in d],
    }


def input_output_spectrum(cov: dict, r2: np.ndarray) -> dict:
    """Modality (definitive) + R^2 shape and nested-class comparison (provisional)."""
    out: dict = {
        "question": "is output-ness a continuous property or a binary one?",
        "test1_modality": {"needs_r2": False, "status": "DEFINITIVE (full width, no R^2)"},
        "test2_shape": {"needs_r2": True, "status": "PROVISIONAL (#1738 SAE->SAE target)"},
        "test3_nested": {"needs_r2": True, "status": "PROVISIONAL (#1738 SAE->SAE target)"},
    }

    for k in OUTPUTNESS:
        v = cov[k]
        lx = _log_x(v)
        m = _modality(v, lx)
        thr = float(np.nanquantile(v, KURT_Q))
        m["q90_threshold"] = float(np.log10(thr)) if (lx and thr > 0) else thr
        m["q90_threshold_raw"] = thr
        out["test1_modality"][k] = m
        _log(f"modality {k}: {m['verdict']} (q0.90 at {thr:.4g})")

    ok = np.isfinite(r2) & np.isfinite(cov["activity"]) & np.isfinite(cov["promoting_class"])
    for k in OUTPUTNESS:
        ok &= np.isfinite(cov[k])
    n = int(ok.sum())
    out["n_used_r2_tests"] = n

    act_dec = np.searchsorted(
        np.quantile(cov["activity"][ok], np.linspace(0, 1, N_DECILES + 1)[1:-1]),
        cov["activity"][ok],
        side="right",
    )
    for k in OUTPUTNESS:
        prof = _decile_profile(cov[k][ok], r2[ok])
        strat = []
        for d in range(N_DECILES):
            m = act_dec == d
            strat.append(PB._spearman(cov[k][ok][m], r2[ok][m]) if m.sum() > 50 else float("nan"))
        med = np.array(prof["decile_median_r2"])
        steps = np.abs(np.diff(med))
        prof["activity_stratified_rho"] = strat
        prof["max_adjacent_step"] = float(np.nanmax(steps)) if steps.size else float("nan")
        prof["total_range"] = float(np.nanmax(med) - np.nanmin(med))
        share = prof["max_adjacent_step"] / max(prof["total_range"], 1e-12)
        prof["largest_step_share_of_range"] = float(share)
        # WHERE the step is matters as much as whether there is one: a step is
        # evidence for the BINARY class only if it sits at the q0.90 cut the
        # class dichotomizes on (the 9|10 decile boundary).
        step_at = int(np.nanargmax(steps)) if steps.size else -1
        prof["largest_step_boundary"] = (
            f"between decile {step_at + 1} and {step_at + 2}" if step_at >= 0 else "n/a"
        )
        prof["q90_cut_boundary"] = "between decile 9 and 10"
        prof["step_coincides_with_q90_cut"] = bool(step_at == 8)
        prof["shape_verdict"] = (
            (
                "step-like, but the step is NOT at the q0.90 cut"
                if not prof["step_coincides_with_q90_cut"]
                else "step-like AT the q0.90 cut (supports a real category boundary)"
            )
            if share > 0.5
            else "smooth gradient (no single decile boundary dominates)"
        )
        out["test2_shape"][k] = prof
        _log(f"shape {k}: {prof['shape_verdict']} (largest step {share:.2f} of range)")

    cls = (cov["promoting_class"][ok] == 1).astype(np.float64)
    kurt = cov["footprint_kurt"][ok]
    y = r2[ok]
    z = _rank_cols([cls, kurt, y])
    zc = (z - z.mean(0)) / z.std(0)
    c = (zc.T @ zc) / n
    out["test3_nested"].update(
        {
            "n_promoting": int(cls.sum()),
            "rho_class_vs_r2": float(c[0, 2]),
            "rho_kurt_vs_r2": float(c[1, 2]),
            "rho_class_vs_r2_given_kurt": float(DR._partial(c[0, 2], c[0, 1], c[1, 2])),
            "rho_kurt_vs_r2_given_class": float(DR._partial(c[1, 2], c[0, 1], c[0, 2])),
        }
    )
    t3 = out["test3_nested"]
    t3["verdict"] = (
        "the binary class adds ~nothing beyond the continuous measure — retire it for this purpose"
        if abs(t3["rho_class_vs_r2_given_kurt"]) < 0.5 * abs(t3["rho_kurt_vs_r2_given_class"])
        else "the binary class carries information the continuous measure does not"
    )
    _log(
        f"nested: class|kurt {t3['rho_class_vs_r2_given_kurt']:+.4f} vs "
        f"kurt|class {t3['rho_kurt_vs_r2_given_class']:+.4f} -> {t3['verdict']}"
    )

    trio = ["footprint_kurt", "write_norm", "enc_dec_cos"]
    zt = _rank_cols([cov[k][ok] for k in trio] + [y])
    zct = (zt - zt.mean(0)) / zt.std(0)
    ct = (zct.T @ zct) / n
    ev = np.linalg.eigvalsh(ct[:3, :3])[::-1]
    mutual = {}
    for i, ki in enumerate(trio):
        others = [j for j in range(3) if j != i]
        sub = ct[np.ix_([i, *others, 3], [i, *others, 3])]
        pinv = np.linalg.pinv(sub)
        mutual[ki] = float(-pinv[0, 3] / np.sqrt(max(pinv[0, 0] * pinv[3, 3], 1e-300)))
    out["outputness_factor_structure"] = {
        "measures": trio,
        "pairwise_rho": {
            f"{trio[i]}__{trio[j]}": float(ct[i, j]) for i in range(3) for j in range(i + 1, 3)
        },
        "eigenvalues_of_3x3_rank_corr": [float(e) for e in ev],
        "variance_share_first_factor": float(ev[0] / ev.sum()),
        "partial_rho_vs_r2_given_other_two": mutual,
        "verdict": (
            "one dominant factor (first eigenvalue carries >60% of the 3-measure variance)"
            if ev[0] / ev.sum() > 0.6
            else "several distinct axes (no single dominant factor)"
        ),
    }
    _log(
        "factor structure: first eigenvalue share "
        f"{out['outputness_factor_structure']['variance_share_first_factor']:.3f} -> "
        f"{out['outputness_factor_structure']['verdict']}"
    )
    return out


# ── sub-analysis 2: within-activity-decile rho profiles ──────────────────────


def activity_decile_profiles(cov: dict, r2: np.ndarray, keys: list[str], n_boot: int, rng) -> dict:
    """rho(predictor, R^2) inside each full-width activity decile, with CIs.

    Partialling and stratifying are NOT equivalent: a single activity-partialled
    rho averages over strata and, under effect modification, describes none of
    them. The RIGHTMOST decile answers "among only high-activating features,
    what explains predictability" — and is also the estimable slice, where the
    measurement-noise confound is absent by construction.
    """
    ok = np.isfinite(r2) & np.isfinite(cov["activity"])
    for k in keys:
        ok &= np.isfinite(cov[k])
    act = cov["activity"][ok]
    y = r2[ok]
    edges = np.quantile(act, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], act, side="right")

    rows: dict = {k: {"rho": [], "ci": [], "n": []} for k in keys}
    ranges, frac_pos = [], []
    for d in range(N_DECILES):
        m = dec == d
        ranges.append([float(edges[d]), float(edges[d + 1])])
        frac_pos.append(float((y[m] > 0).mean()) if m.any() else float("nan"))
        nd = int(m.sum())
        yr = PB._rank(y[m])
        zp = np.column_stack([PB._rank(cov[k][ok][m]) for k in keys])
        zpc = (zp - zp.mean(0)) / zp.std(0)
        ycz = (yr - yr.mean()) / yr.std()
        point = (zpc.T @ ycz) / nd
        z = np.column_stack([zp, yr])
        raw_b, _ = _boot_raw_partial(z, 0, len(keys), n_boot, rng)
        for j, k in enumerate(keys):
            rows[k]["rho"].append(float(point[j]))
            rows[k]["ci"].append(PB._ci(raw_b[:, j]))
            rows[k]["n"].append(nd)
        _log(f"  activity decile {d + 1}/{N_DECILES}: n={nd} frac_r2_pos={frac_pos[-1]:.3f}")

    for k in keys:
        r = np.array(rows[k]["rho"])
        lo_ci, hi_ci = rows[k]["ci"][0], rows[k]["ci"][-1]
        rows[k]["label"] = COVARIATES[k][0]
        rows[k]["spread_max_minus_min"] = float(np.nanmax(r) - np.nanmin(r))
        rows[k]["sign_flips"] = bool(np.nanmax(r) > 0 > np.nanmin(r))
        rows[k]["ends_ci_overlap"] = bool(lo_ci[0] <= hi_ci[1] and hi_ci[0] <= lo_ci[1])
        rows[k]["top_decile_rho"] = float(r[-1])
        rows[k]["homogeneity_verdict"] = (
            "FLAT — the single activity-partialled rho is a valid summary"
            if rows[k]["ends_ci_overlap"] and rows[k]["spread_max_minus_min"] < 0.1
            else "EFFECT MODIFICATION — the profile replaces the single number"
        )

    order = sorted(keys, key=lambda k: -rows[k]["spread_max_minus_min"])
    _log(
        "heterogeneity ranking (spread): "
        + ", ".join(f"{k} {rows[k]['spread_max_minus_min']:.2f}" for k in order[:5])
    )
    return {
        "decile_activity_ranges": ranges,
        "decile_frac_r2_positive": frac_pos,
        "n_boot": int(n_boot),
        "reading_rule": (
            "FLAT profile => homogeneous effect, a single activity-partialled rho is a valid "
            "summary. SLOPED or SIGN-FLIPPING => effect modification; the profile REPLACES the "
            "single number and any headline quoting one partial rho for that predictor is "
            "misleading. The RIGHTMOST point answers 'among only high-activating features, what "
            "explains predictability' — and that decile is also the estimable slice, so the "
            "measurement-noise confound is absent there by construction."
        ),
        "heterogeneity_ranking": order,
        "per_predictor": rows,
    }


# ── figures ──────────────────────────────────────────────────────────────────


PROVISIONAL = (
    "PROVISIONAL TARGET — #1738 SAE->SAE context arm. The correct target (full-width "
    "dense->SAE, task #7) does not exist yet. The two arms rank features at rho = 0.933 at "
    "panel width, so SHAPE conclusions survive the swap; LEVELS do not."
)


def _subtitle(n_used: int, note: str = PROVISIONAL) -> str:
    return (
        f"{n_used:,} SAE features, full dictionary (BatchTopK k=64, resid_post layer 19)  |  {note}"
    )


def fig_scatter(
    reads: dict, cov: dict, r2: np.ndarray, fig_dir: Path, suffix: str = "", note: str = PROVISIONAL
) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    ok = reads["mask"]
    y = np.clip(r2[ok], R2_DISPLAY_FLOOR, None)
    rows = reads["predictors"]
    ncol = 5
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.15 * ncol, 2.85 * nrow), sharey=True)
    flat = axes.ravel()
    for ax, row in zip(flat, rows, strict=False):
        v = cov[row["key"]][ok]
        lx = row["log_x_axis"]
        ax.hexbin(
            v,
            y,
            gridsize=52,
            bins="log",
            mincnt=1,
            xscale="log" if lx else "linear",
            cmap="Blues",
            linewidths=0,
        )
        ax.plot(
            row["decile_center"],
            row["decile_median_r2"],
            "-",
            lw=1.4,
            color=paper_palette_role("accent"),
        )
        if not lx:
            f = v[np.isfinite(v)]
            lo, hi = np.percentile(f, [X_VIEW_PCT, 100 - X_VIEW_PCT])
            if hi > lo:
                pad = 0.04 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
            ax.locator_params(axis="x", nbins=5)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4), useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(7.0)
        ax.text(
            0.04,
            0.96,
            f"$\\rho$ = {row['spearman_raw']:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
        )
        lab = row["label"] + (" (log)" if lx else "")
        if len(lab) > 34:  # long labels clip at the figure edge in the last column
            import textwrap

            lab = "\n".join(textwrap.wrap(lab, 34))
        ax.set_xlabel(lab, fontsize=7.4)
        ax.tick_params(labelsize=7.2)
    for ax in flat[: len(rows)][::ncol]:
        ax.set_ylabel(r"per-feature held-out $R^2$", fontsize=8.2)
    for ax in flat[len(rows) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Full dictionary: which feature properties track per-feature predictability?",
        fontsize=12.5,
        y=0.995,
    )
    fig.text(
        0.5, 0.957, _subtitle(reads["n_used"], note), ha="center", fontsize=7.2, color="#5A5A5A"
    )
    fig.text(
        0.5,
        0.010,
        f"decile-median trend overlaid; log-count hexbin. $R^2$ display-clipped at "
        f"{R2_DISPLAY_FLOOR:g} — this hides "
        f"{100 * np.mean(r2[ok] < R2_DISPLAY_FLOOR):.2f}% of THIS arm's features; "
        f"linear-axis panels view-clip x to "
        f"[{X_VIEW_PCT:g}, {100 - X_VIEW_PCT:g}] pct — display only, statistics unclipped.",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.026, 1, 0.947))
    stem = "continuous_scatter_panel" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_forest(reads: dict, fig_dir: Path, suffix: str = "", note: str = PROVISIONAL) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    rows = list(reads["predictors"])
    scored = [r for r in rows if r["partial_on_activity"] is not None]
    unscored = [r for r in rows if r["partial_on_activity"] is None]
    scored.sort(key=lambda r: abs(r["partial_on_activity"]))
    ordered = unscored + scored
    ypos = np.arange(len(ordered), dtype=float)
    raw = np.array([r["spearman_raw"] for r in ordered])
    raw_ci = np.array([r["spearman_raw_ci95"] for r in ordered])
    par = np.array(
        [np.nan if r["partial_on_activity"] is None else r["partial_on_activity"] for r in ordered]
    )
    par_ci = np.array(
        [
            [np.nan, np.nan]
            if r["partial_on_activity_ci95"] is None
            else r["partial_on_activity_ci95"]
            for r in ordered
        ]
    )
    ok = np.isfinite(par)

    fig, ax = plt.subplots(figsize=(9.8, 0.42 * len(ordered) + 3.0))
    ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.errorbar(
        raw,
        ypos + 0.16,
        xerr=PB._errbars(raw, raw_ci),
        fmt="o",
        ms=4.5,
        color=paper_palette_role("baseline"),
        mfc="white",
        mew=1.2,
        lw=1.0,
        capsize=2.0,
        label="raw Spearman $\\rho$",
    )
    ax.errorbar(
        par[ok],
        ypos[ok] - 0.16,
        xerr=PB._errbars(par[ok], par_ci[ok]),
        fmt="o",
        ms=4.5,
        color=paper_palette_role("primary"),
        lw=1.0,
        capsize=2.0,
        label="partialling out per-answer firing frequency",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [
            r["label"] + ("" if r["partial_on_activity"] is not None else "  — partial n/a")
            for r in ordered
        ],
        fontsize=8.2,
    )
    ax.set_ylim(-0.75, len(ordered) - 0.25)
    ax.set_xlabel(r"Spearman $\rho$ against per-feature held-out $R^2$", fontsize=9.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8.4)
    fig.suptitle(
        "Full dictionary: what survives adjusting for firing frequency?", fontsize=12.5, y=0.985
    )
    fig.text(
        0.5, 0.943, _subtitle(reads["n_used"], note), ha="center", fontsize=7.2, color="#5A5A5A"
    )
    fig.text(
        0.5,
        0.012,
        f"rows sorted by |partial rho|; percentile 95% CI over {N_BOOT:,} bootstrap draws "
        "(ranks fixed at the full sample). SEE ALSO rho_by_activity_decile: several of these "
        "predictors show effect modification, where a single partial rho is not a valid summary.",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.072, 1, 0.933))
    stem = "continuous_rho_vs_activity_partial" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_spectrum(spec: dict, fig_dir: Path, suffix: str = "", note: str = PROVISIONAL) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    ncol = len(OUTPUTNESS)
    fig, axes = plt.subplots(2, ncol, figsize=(3.05 * ncol, 6.6))
    for j, k in enumerate(OUTPUTNESS):
        m = spec["test1_modality"][k]
        ax = axes[0, j]
        ax.plot(m["kde_grid"], m["kde_density"], "-", lw=1.5, color=paper_palette_role("primary"))
        ax.fill_between(
            m["kde_grid"], m["kde_density"], alpha=0.18, color=paper_palette_role("primary")
        )
        for mo in m["modes"]:
            ax.axvline(mo, color=paper_palette_role("accent"), lw=1.0, ls=":")
        ax.axvline(
            m["q90_threshold"],
            color=paper_palette_role("control"),
            lw=1.4,
            ls="--",
            label="binary q0.90 cut",
        )
        ax.set_title(m["verdict"], fontsize=8.8)
        vr = m.get("view_range_p0p5_p99p5")
        if vr and vr[1] > vr[0]:
            pad = 0.04 * (vr[1] - vr[0])
            ax.set_xlim(min(vr[0], m["q90_threshold"]) - pad, max(vr[1], m["q90_threshold"]) + pad)
        ax.set_xlabel(COVARIATES[k][0] + (" (log10)" if m["log10_x"] else ""), fontsize=7.2)
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("density (full dictionary)", fontsize=8.2)
            ax.legend(loc="upper right", frameon=False, fontsize=6.8)
        ax.tick_params(labelsize=7.0)

        p = spec["test2_shape"][k]
        ax = axes[1, j]
        ax.plot(
            range(1, N_DECILES + 1),
            p["decile_median_r2"],
            "o-",
            ms=3.6,
            lw=1.4,
            color=paper_palette_role("primary"),
        )
        ax.axvline(N_DECILES * KURT_Q + 0.5, color=paper_palette_role("control"), lw=1.4, ls="--")
        ax.set_xlabel(f"decile of {COVARIATES[k][0]}", fontsize=7.2)
        ax.set_title(p["shape_verdict"].split(" (")[0], fontsize=8.0)
        if p["largest_step_share_of_range"] > 0.5:
            ax.text(
                0.03,
                0.95,
                f"largest step {p['largest_step_boundary']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.6,
                color="#7A7A7A",
            )
        if j == 0:
            # Derive the tag from `note` — the same source the subtitle below uses.
            # Hardcoding it here let a REAL-TARGET render carry a stale
            # "(PROVISIONAL)" axis label directly under a subtitle saying the
            # opposite (#1482: both densesae arms shipped self-contradictory).
            tag = " (PROVISIONAL)" if "PROVISIONAL" in note else ""
            ax.set_ylabel(rf"median $R^2${tag}", fontsize=8.2)
        ax.tick_params(labelsize=7.0)

    fig.suptitle("Is output-ness a continuous property or a binary one?", fontsize=12.5, y=0.995)
    fig.text(
        0.5,
        0.947,
        "TOP ROW: full-dictionary density of each continuous measure — needs no $R^2$, so these "
        f"verdicts are DEFINITIVE.   BOTTOM ROW: median $R^2$ per decile — {note}",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.010,
        "dashed vertical = the q0.90 cut the binary promoting/suppressing class uses; dotted "
        "verticals = detected KDE modes. A unimodal density with the cut inside the bulk means "
        "the binary class is a slice off a continuum, not a natural kind.",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.938))
    stem = "input_output_spectrum" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_decile_profiles(
    prof: dict, fig_dir: Path, suffix: str = "", note: str = PROVISIONAL
) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    order = prof["heterogeneity_ranking"]
    head = set(order[:4])
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    x = np.arange(1, N_DECILES + 1)
    colors = paper_palette(len(order))
    for k, c in zip(order, colors, strict=True):
        r = prof["per_predictor"][k]
        lw = 2.1 if k in head else 0.9
        al = 1.0 if k in head else 0.42
        ax.plot(x, r["rho"], "o-", ms=3.4, lw=lw, alpha=al, color=c, label=r["label"])
        if k in head:
            ci = np.array(r["ci"])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=c, alpha=0.16, lw=0)
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{d + 1}\n{lo:.1e}\n{hi:.1e}"
            for d, (lo, hi) in enumerate(prof["decile_activity_ranges"])
        ],
        fontsize=6.2,
    )
    ax.set_xlabel("activity decile (per-answer firing-frequency range)", fontsize=9.0)
    ax.set_ylabel(r"within-decile Spearman $\rho$ vs $R^2$", fontsize=9.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7.0)
    fig.suptitle(
        "Effect modification: the same predictor behaves differently at different activity levels",
        fontsize=12.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.944,
        _subtitle(int(np.sum(prof["per_predictor"][order[0]]["n"])), note)
        + "   |   CI bands on the 4 most heterogeneous predictors only",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.012,
        "FLAT => the single activity-partialled rho is a valid summary. SLOPED or SIGN-FLIPPING "
        "=> effect modification; the profile replaces the single number. The RIGHTMOST point "
        "answers 'among only high-activating features, what explains predictability' — that "
        "decile is also the estimable slice "
        f"({100 * prof['decile_frac_r2_positive'][-1]:.1f}% of its features have $R^2$ > 0), so "
        "measurement noise is absent there by construction.",
        ha="center",
        fontsize=7.2,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.058, 1, 0.934))
    stem = "rho_by_activity_decile" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 full-width predictor reads")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=DR.N_WORKERS)
    ap.add_argument(
        "--r2-npy",
        type=Path,
        default=PROJECT_ROOT / R2_PROVISIONAL,
        help="per-feature R^2 array; swap for task #7's dense->SAE array",
    )
    ap.add_argument("--r2-label", default="PROVISIONAL-#1738-sae_to_sae")
    ap.add_argument("--stem-suffix", default="", help="per-arm output suffix")
    ap.add_argument("--target-note", default=PROVISIONAL, help="caption line for this target")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument("--work", type=Path, default=PROJECT_ROOT / "data/issue_1482/fullwidth")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_boot = 50
    for d in (args.out_dir, args.fig_dir, args.work):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    bundle = assemble_fullwidth(args.work, args.workers)
    cov = bundle["cov"]

    r2 = np.asarray(np.load(args.r2_npy), dtype=np.float64)
    if r2.shape != (DICT_SIZE,):
        raise AssertionError(f"R^2 array must be ({DICT_SIZE},), got {r2.shape}")
    _log(f"R^2 target: {args.r2_npy.name} [{args.r2_label}] median {np.nanmedian(r2):+.4f}")

    degenerate = {k: bool(np.nanstd(cov[k]) < 1e-6) for k in DEGENERATE_CANDIDATES}
    keys = [
        k
        for k, (_, g) in COVARIATES.items()
        if g == "fullwidth" and k in cov and not degenerate.get(k, False)
    ]

    legend = {
        k: {"label": lab, "grain": g, "present": k in cov} for k, (lab, g) in COVARIATES.items()
    }
    legend["promoting_class"] = {
        "label": "0=other, 1=promoting, 2=suppressing, 3=partition (kurt/var q0.90)",
        "grain": "fullwidth",
        "present": True,
    }
    np.savez(
        args.out_dir / "fullwidth_covariates.npz",
        feat_ids=np.arange(DICT_SIZE, dtype=np.int64),
        **{k: v for k, v in cov.items() if getattr(v, "shape", None) == (DICT_SIZE,)},
    )
    (args.out_dir / "fullwidth_covariates_legend.json").write_text(
        json.dumps(
            {
                "join_key": "feat_ids",
                "n_features": DICT_SIZE,
                "target_independent": True,
                "purpose": (
                    "durable substrate: every R^2 array (provisional #1738 SAE->SAE, task #7's "
                    "dense->SAE ridge and MLP, any later refit) joins against THESE covariates "
                    "by feat_ids without recomputation"
                ),
                "columns": legend,
                "pending_from_pod": bundle["pending"],
                "pending_source": RUN_LENGTH,
                "metadata": PB._metadata(),
            },
            indent=1,
        )
    )
    _log(f"covariate substrate -> fullwidth_covariates.npz ({len(keys)} live predictors)")

    reads = main_reads(cov, r2, keys, args.n_boot, rng)
    spec = input_output_spectrum(cov, r2)
    prof = activity_decile_profiles(cov, r2, keys, args.n_boot, rng)

    doc = {
        "design": {
            "scope": "FULL DICTIONARY (131,072); panel-width figures dropped per directive",
            "n_features_used": reads["n_used"],
            "r2_source": str(
                args.r2_npy.resolve().relative_to(PROJECT_ROOT)
                if args.r2_npy.resolve().is_relative_to(PROJECT_ROOT)
                else args.r2_npy
            ),
            "r2_label": args.r2_label,
            "r2_status": PROVISIONAL,
            "correct_target_pending": (
                "full-width dense->SAE ridge AND MLP from task #7 "
                "(scripts/issue1482_densesae_fullwidth.py); no pod provisioned for #1482 as of "
                "this run. Re-render by swapping --r2-npy."
            ),
            "two_arm_dependency_risk": (
                "the linear-vs-nonlinear comparison rests ENTIRELY on the pod grid's full-width "
                "MLP cells succeeding. If they fail or are descoped there is no nonlinear arm at "
                "any published grain. Flagged, not solved."
            ),
            "panel_reference_only": {
                "note": "panel-width dense->SAE numbers, retained for the eventual comparison",
                "ridge": {
                    "per_feature_median_r2": 0.1767,
                    "frac_positive": 0.993,
                    "pooled_r2": 0.7216,
                },
                "mlp": {
                    "per_feature_median_r2": -0.0285,
                    "frac_positive": 0.461,
                    "pooled_r2": 0.7387,
                },
            },
            "n_boot": int(args.n_boot),
            "seed": SEED,
        },
        "gates": bundle["gates"],
        "degenerate_covariates": degenerate,
        "pending_covariates": bundle["pending"],
        "predictors": reads["predictors"],
        "input_output_spectrum": spec,
        "activity_decile_profiles": prof,
        "definitions": {
            "scaffold_frac": (
                "GEOMETRIC: ||P_48 . w_dec[:,f]||^2 / ||w_dec[:,f]||^2, P_48 = top-48 PCA "
                "eigenvectors of cov(h_prefix) (#1773 L475-486). Says NOTHING about which tokens "
                "the feature fires on; the behavioural twin is template_token_frac (pending pod)."
            ),
            "logit_footprint_concentration": (
                "share of POSITIVE direct-logit mass on the top-10 promoted tokens, "
                "E = W_U @ (gamma * W_dec[:,f]) (#1773 L256-284). Recomputed full width here."
            ),
            "side_ratio": "cnt / (cnt + psi_cnt) over fit rows — answer-side share of firings.",
            "describe_confidence": "ORDINAL (integer auto-interp confidence), panel-grain only.",
            "mean_run_length_vs_p": (
                "E[R] = 1/(1-p) exactly — the same quantity; only mean_run_length is plotted."
            ),
        },
        "not_computable": DR.NOT_COMPUTABLE,
        "metadata": PB._metadata(),
    }
    (args.out_dir / f"continuous_predictors{args.stem_suffix}.json").write_text(
        json.dumps(doc, indent=1)
    )
    _log("reads -> continuous_predictors.json")

    import matplotlib

    matplotlib.use("Agg")
    stems = [
        fig_scatter(reads, cov, r2, args.fig_dir, args.stem_suffix, args.target_note),
        fig_forest(reads, args.fig_dir, args.stem_suffix, args.target_note),
        fig_spectrum(spec, args.fig_dir, args.stem_suffix, args.target_note),
        fig_decile_profiles(prof, args.fig_dir, args.stem_suffix, args.target_note),
    ]
    _log(f"figures: {', '.join(stems)}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
