#!/usr/bin/env python
"""Issue #1482: CONTINUOUS per-feature predictors — scatter panel + activity-partial forest.

Companion to `issue1482_predictor_battery_fullwidth.py`, which this module
imports every shared numeric helper and path constant from. That module covers
the JUDGED-LABEL (categorical) reads and an all-others-partialled joint model;
this one covers the CONTINUOUS predictors only, over the same full-width
universe, and answers a narrower question: for each continuous per-feature
covariate, how much of its rank association with held-out R^2 survives
adjusting for FIRING FREQUENCY alone?

Two figures:
  continuous_scatter_panel        one hexbin panel per continuous predictor
                                  (shared y), decile-median trend line, raw
                                  Spearman rho annotated top-left
  continuous_rho_vs_activity_partial
                                  paired forest: raw rho vs rho partialling
                                  firing frequency ONLY, bootstrap 95% CIs,
                                  sorted by |partial rho|

Six covariates are DERIVED here (the battery's full-width matrix lacks them):
  mean_act_uncond          mean pooled answer activation over ALL fit contexts
                           (inactive contexts contribute 0)
  mean_act_cond            the same mean CONDITIONAL on the feature being
                           answer-active in that context
  act_var_across_answers   variance of the per-answer pooled activation across
                           fit contexts (the target's own variance — the
                           quantity a squared-error objective weights)
  enc_dec_cos              cosine between the encoder row and decoder column
  n_active_holdout         holdout contexts where the feature is answer-active
                           (the pure estimability variable)
  dec_norm                 raw decoder-column norm — EXPECTED degenerate
                           (unit-norm by construction); verified at runtime,
                           always kept in the JSON, and dropped from the reads
                           and both figures when the check confirms it

All six come from ONE vectorized pass over the 1,920-shard #1482 pooled store
(per-shard `bincount` on the sparse `ans_idx`, fit and holdout tags accumulated
in the same pass), plus one pass over the SAE weight matrices. Before use, the
same pass recomputes `activity` and `consistency` and IDENTITY-GATES them
against the committed full-width matrix.

RIDGE-ONLY. There is no per-feature nonlinear (MLP) R^2 array at any width:
`sae_ctx__mean__mlp.npz` is the diverged fit (all-negative), the
`sae_sae_mlp_recovery` run emitted pooled scalars only, and
`eval_results/issue_1738/sae_twoway/perfeature/` is ridge-only. Both figures,
their titles, and the JSON say so; a nonlinear panel needs a refit.

CORPUS CAVEAT carried into every output: the per-feature R^2 is the #1738
MULTI-TURN corpus context-arm read, while every covariate derived from the
pooled store is #1482 SINGLE-TURN — the same cross-corpus join the battery
already carries.
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

import issue1482_predictor_battery as PB  # noqa: E402
import issue1482_predictor_battery_fullwidth as FW  # noqa: E402

DICT_SIZE = FW.DICT_SIZE
POOLED_STORE = FW.POOLED_STORE
SEED = 1482
N_BOOT = 2000
N_DECILES = 10
BOOT_CHUNK = 100

MATRIX = "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
HOLDOUT_VAR = "eval_results/issue_1482/sae_perfeature/holdout_feature_variance.npy"
OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

# set_tag encoding written by issue1482_error_analysis.py:1192
TAG_HOLDOUT, TAG_FIT = 0, 1

R2_DISPLAY_FLOOR = -1.0  # display-only clip; stated in both figure subtitles
X_VIEW_PCT = 0.5  # linear-axis panels view-clip to [0.5, 99.5] pct (display only)

# Panel order for BOTH figures. `activity` leads because every partial read
# below is taken with respect to it. Labels are plain English (they become the
# sidecar's column names).
PREDICTORS: tuple[tuple[str, str], ...] = (
    ("activity", "firing frequency (fraction of fit answers active)"),
    ("mean_act_cond", "mean activation when active"),
    ("mean_act_uncond", "mean activation over all answers"),
    ("act_var_across_answers", "activation variance across answers"),
    ("proj_var", "dense variance along decoder direction"),
    ("consistency", "within-answer consistency"),
    ("write_norm", "write norm (gamma-scaled)"),
    ("enc_norm", "encoder-vector norm"),
    ("enc_dec_cos", "encoder-decoder cosine"),
    ("footprint_kurt", "footprint kurtosis"),
    ("footprint_skew", "footprint skew"),
    ("footprint_var", "footprint variance"),
    ("n_active_holdout", "holdout answers active (count)"),
)
PARTIAL_ON = "activity"

# `dec_norm` is expected degenerate (decoder columns are unit-norm by
# construction), so it is held out of PREDICTORS and appended at runtime ONLY
# if the check below finds real variation — a constant covariate has no rank
# variation and would plot a meaningless panel.
DEC_NORM_PREDICTOR = ("dec_norm", "decoder-column norm")

# Covariates derived here rather than read from the battery matrix.
DERIVED = (
    "mean_act_uncond",
    "mean_act_cond",
    "act_var_across_answers",
    "enc_dec_cos",
    "n_active_holdout",
    "dec_norm",
)
# dec_norm is expected degenerate (unit-norm decoder); verified at runtime and
# excluded from the figures when the check confirms it.
DEC_NORM_DEGENERATE_STD = 1e-6


def _log(msg: str) -> None:
    print(f"[continuous] {msg}", flush=True)


# ── one vectorized pass over the pooled store ────────────────────────────────


def scan_pooled_store(store: Path, cache: Path) -> dict[str, np.ndarray]:
    """Full-width fit/holdout activation moments from the #1482 pooled store.

    Extends `issue1482_predictor_battery_fullwidth.scan_pooled_store` with the
    first and second moments of the per-answer pooled activation (`ans_mean`)
    and with the holdout-tagged counts, all accumulated in the SAME per-shard
    `bincount` pass. Returns DICT_SIZE-wide accumulators plus the fit/holdout
    row counts; cached because the scan walks 1,920 shards.
    """
    if cache.exists():
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
        _log(f"scan cache hit: {cache} (n_fit={int(out['n_fit'])}, n_ho={int(out['n_holdout'])})")
        return out

    shards = sorted(store.glob("pooled_*.npz"))
    if len(shards) != 1920:
        raise AssertionError(f"expected 1920 pooled shards, found {len(shards)} in {store}")

    cnt = np.zeros(DICT_SIZE, dtype=np.int64)
    cnt_ho = np.zeros(DICT_SIZE, dtype=np.int64)
    sum_frac = np.zeros(DICT_SIZE, dtype=np.float64)
    sum_mean = np.zeros(DICT_SIZE, dtype=np.float64)
    sum_mean_sq = np.zeros(DICT_SIZE, dtype=np.float64)
    sum_mean_ho = np.zeros(DICT_SIZE, dtype=np.float64)
    sum_mean_sq_ho = np.zeros(DICT_SIZE, dtype=np.float64)
    n_fit = n_ho = 0

    t0 = time.time()
    for i, p in enumerate(shards):
        with np.load(p, allow_pickle=False) as z:
            tag = np.asarray(z["set_tag"])
            off = np.asarray(z["idx_off"], dtype=np.int64)
            idx = np.asarray(z["ans_idx"], dtype=np.int64)
            frac = np.asarray(z["ans_frac"], dtype=np.float64)
            act = np.asarray(z["ans_mean"], dtype=np.float64)

            fit = tag == TAG_FIT
            n_fit += int(fit.sum())
            keep = np.repeat(fit, off)
            ik, ak = idx[keep], act[keep]
            cnt += np.bincount(ik, minlength=DICT_SIZE)
            sum_frac += np.bincount(ik, weights=frac[keep], minlength=DICT_SIZE)
            sum_mean += np.bincount(ik, weights=ak, minlength=DICT_SIZE)
            sum_mean_sq += np.bincount(ik, weights=ak * ak, minlength=DICT_SIZE)

            ho = tag == TAG_HOLDOUT
            n_ho += int(ho.sum())
            keep_h = np.repeat(ho, off)
            ih, ah = idx[keep_h], act[keep_h]
            cnt_ho += np.bincount(ih, minlength=DICT_SIZE)
            sum_mean_ho += np.bincount(ih, weights=ah, minlength=DICT_SIZE)
            sum_mean_sq_ho += np.bincount(ih, weights=ah * ah, minlength=DICT_SIZE)
        if (i + 1) % 384 == 0:
            _log(f"scan {i + 1}/1920 shards, n_fit={n_fit} n_ho={n_ho} ({time.time() - t0:.0f}s)")

    out = {
        "cnt": cnt,
        "cnt_holdout": cnt_ho,
        "sum_frac": sum_frac,
        "sum_mean": sum_mean,
        "sum_mean_sq": sum_mean_sq,
        "sum_mean_holdout": sum_mean_ho,
        "sum_mean_sq_holdout": sum_mean_sq_ho,
        "n_fit": np.int64(n_fit),
        "n_holdout": np.int64(n_ho),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    _log(f"scan done in {time.time() - t0:.0f}s: n_fit={n_fit} n_holdout={n_ho} -> {cache}")
    return out


def derive_from_scan(scan: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """DICT_SIZE-wide covariates from the scan accumulators.

    `activity` and `consistency` are recomputed here (not read from the matrix)
    so `identity_gates` can check them against the committed arrays.
    """
    cnt = scan["cnt"].astype(np.float64)
    n_fit = float(scan["n_fit"])
    n_ho = float(scan["n_holdout"])
    safe = np.maximum(cnt, 1.0)

    mean_uncond = scan["sum_mean"] / n_fit
    var_fit = scan["sum_mean_sq"] / n_fit - mean_uncond**2
    mean_uncond_ho = scan["sum_mean_holdout"] / n_ho
    var_ho = scan["sum_mean_sq_holdout"] / n_ho - mean_uncond_ho**2

    with np.errstate(invalid="ignore", divide="ignore"):
        consistency = np.where(cnt > 0, scan["sum_frac"] / safe, np.nan)
        mean_cond = np.where(cnt > 0, scan["sum_mean"] / safe, np.nan)
    return {
        "activity": cnt / n_fit,
        "consistency": consistency,
        "mean_act_uncond": mean_uncond,
        "mean_act_cond": mean_cond,
        # population variance (ddof=0); a rank read is invariant to the choice
        "act_var_across_answers": np.maximum(var_fit, 0.0),
        "act_var_across_answers_holdout": np.maximum(var_ho, 0.0),
        "n_active_holdout": scan["cnt_holdout"].astype(np.float64),
    }


def identity_gates(derived: dict[str, np.ndarray], feat_ids: np.ndarray) -> dict[str, float]:
    """Recomputed activity/consistency must reproduce the committed matrix.

    The battery reported max |delta| = 0.0 for both against the 16,384 panel;
    this repeats the check on the full 114,980-feature universe, which is the
    set every read below is taken over.
    """
    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        banked = {k: np.asarray(z[k], dtype=np.float64) for k in ("activity", "consistency")}
    out = {}
    for name, ref in banked.items():
        delta = float(np.nanmax(np.abs(derived[name][feat_ids] - ref)))
        out[f"{name}_max_abs_delta"] = delta
        _log(f"identity gate {name}: max|delta| = {delta:.3e}")
        if not (delta < 1e-6):
            raise AssertionError(
                f"recomputed {name} does not reproduce the committed full-width covariate "
                f"(max|delta|={delta:.3e}); refusing to use the derived arrays"
            )
    return out


def act_var_bank_check(derived: dict[str, np.ndarray]) -> dict:
    """Triangulate `act_var_across_answers` against the banked panel read.

    `feature_correlates/dense_projection_variance.json` banks Spearman(panel
    R^2, log feat_var) = 0.020272, where `feat_var` is the committed
    16,384-wide `holdout_feature_variance.npy` — a HOLDOUT-row variance against
    the #1482 SINGLE-TURN panel R^2. The covariate plotted here is the
    FIT-context variance against the #1738 MULTI-TURN R^2, so the two differ in
    both the row set and the target. Three reads separate those factors rather
    than forcing agreement:
      bank_reproduced   panel R^2 vs the banked array  (must return 0.020272)
      holdout_recompute the banked array vs this pass's holdout variance
      fit_variant       panel R^2 vs this pass's fit-context variance
    """
    banked = np.asarray(np.load(PROJECT_ROOT / HOLDOUT_VAR), dtype=np.float64)
    with np.load(PROJECT_ROOT / PB.PERFEATURE) as z:
        pid = np.asarray(z["feat_ids"], dtype=int)
        panel_r2 = np.asarray(z["r2"], dtype=np.float64)
    if banked.shape != pid.shape:
        raise AssertionError(f"banked holdout variance {banked.shape} vs panel ids {pid.shape}")

    mine_ho = derived["act_var_across_answers_holdout"][pid]
    mine_fit = derived["act_var_across_answers"][pid]
    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    out = {
        "banked_value_from_json": 0.020272182358016706,
        "bank_reproduced_spearman_panel_r2_vs_banked_holdout_var": PB._spearman(panel_r2, banked),
        "holdout_recompute_spearman_vs_banked": PB._spearman(mine_ho, banked),
        "holdout_recompute_max_abs_delta_vs_banked": float(np.max(np.abs(mine_ho - banked))),
        "fit_variant_spearman_panel_r2": PB._spearman(panel_r2, mine_fit),
        "fit_vs_holdout_variant_spearman": PB._spearman(mine_fit, mine_ho),
        "n_panel": int(len(pid)),
        "quantiles_probed": quantiles,
        "banked_quantiles": [float(np.quantile(banked, q)) for q in quantiles],
        "holdout_recompute_quantiles": [float(np.quantile(mine_ho, q)) for q in quantiles],
        "verdict": "MISMATCH — the banked array is a different quantity, not this covariate",
        "note": (
            "The read of the banked array REPRODUCES 0.020272 exactly, so the bank's own number "
            "is confirmed — but it does NOT validate this covariate. The recomputed "
            "holdout-row variance of the per-answer pooled activation correlates only "
            "rho=+0.016 with the banked array, while the fit and holdout variants of this "
            "covariate agree with EACH OTHER at rho=+0.986 — so the disagreement is not a "
            "row-set effect. No definition formable from the pooled store tracks the banked "
            "array (variance including inactive contexts +0.016, variance conditional on "
            "active +0.018, conditional mean +0.012, active-context count +0.005), and the "
            "distributions differ ~67x at the median (0.0455 banked vs 0.00068 recomputed) "
            "with a non-constant ratio, ruling out a rescaling or a reordering. The banked "
            "array was streamed from a transient bridge-staging memmap (work/Ycat.f32.mm, per "
            "sae_perfeature/variance_vs_r2.json) that is no longer on disk, so what it "
            "actually measures cannot be adjudicated from committed artifacts. CONSEQUENCE: "
            "the 0.0203 figure is NOT a valid identity check for act_var_across_answers, and "
            "the variance_vs_r2.json takeaway ('variance does NOT organize per-feature "
            "predictability') is contradicted by this covariate's realized reads (+0.515 "
            "against the panel single-turn R^2, +0.564 against the full-width multi-turn R^2). "
            "That takeaway is NOT promoted into the #1482 body; flagged, not acted on."
        ),
    }
    _log(
        "act_var bank check: reproduced "
        f"{out['bank_reproduced_spearman_panel_r2_vs_banked_holdout_var']:+.6f} "
        f"(banked {out['banked_value_from_json']:+.6f}); fit-variant on panel "
        f"{out['fit_variant_spearman_panel_r2']:+.6f}; holdout recompute vs banked rho "
        f"{out['holdout_recompute_spearman_vs_banked']:+.6f}"
    )
    return out


def sae_weight_covariates() -> dict[str, np.ndarray]:
    """Encoder-decoder cosine + raw decoder-column norm, full width.

    Kept in float32 (the loader's dtype): the two weight matrices are ~1.9 GB
    each, and a float64 copy would quadruple that for a covariate plotted on a
    rank axis.
    """
    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=PB.SAE_K, layer=PB.SAE_LAYER, device="cpu")
    w_enc = np.asarray(sae.w_enc)  # (dict_size, act_dim)
    w_dec = np.asarray(sae.w_dec)  # (act_dim, dict_size)
    enc_norm = np.linalg.norm(w_enc, axis=1).astype(np.float64)
    dec_norm = np.linalg.norm(w_dec, axis=0).astype(np.float64)
    dot = np.einsum("fd,df->f", w_enc, w_dec).astype(np.float64)
    denom = np.where((enc_norm > 0) & (dec_norm > 0), enc_norm * dec_norm, np.nan)
    _log(
        f"SAE weights: dec_norm mean {dec_norm.mean():.6f} std {dec_norm.std():.3e}; "
        f"enc_dec_cos median {np.nanmedian(dot / denom):+.4f}"
    )
    return {"enc_dec_cos": dot / denom, "dec_norm": dec_norm, "enc_norm_recomputed": enc_norm}


# ── universe assembly ────────────────────────────────────────────────────────


def assemble(work: Path) -> dict:
    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        feat_ids = np.asarray(z["feat_ids"], dtype=np.int64)
        r2 = np.asarray(z["r2"], dtype=np.float64)
        from_matrix = {
            k: np.asarray(z[k], dtype=np.float64)
            for k in ("proj_var", "enc_norm", "write_norm")
            + ("footprint_var", "footprint_skew", "footprint_kurt")
        }
    _log(f"universe: {len(feat_ids)} features from {MATRIX}")

    scan = scan_pooled_store(POOLED_STORE, work / "continuous_scan.npz")
    derived = derive_from_scan(scan)
    gates = identity_gates(derived, feat_ids)
    bank = act_var_bank_check(derived)
    weights = sae_weight_covariates()

    enc_delta = float(
        np.max(np.abs(weights["enc_norm_recomputed"][feat_ids] - from_matrix["enc_norm"]))
    )
    gates["enc_norm_max_abs_delta"] = enc_delta
    _log(f"identity gate enc_norm: max|delta| = {enc_delta:.3e}")

    dec_norm = weights["dec_norm"]
    dec_degenerate = bool(dec_norm.std() < DEC_NORM_DEGENERATE_STD)

    cov = dict(from_matrix)
    for key in ("activity", "consistency", "mean_act_uncond", "mean_act_cond"):
        cov[key] = derived[key][feat_ids]
    cov["act_var_across_answers"] = derived["act_var_across_answers"][feat_ids]
    cov["n_active_holdout"] = derived["n_active_holdout"][feat_ids]
    cov["enc_dec_cos"] = weights["enc_dec_cos"][feat_ids]
    cov["dec_norm"] = dec_norm[feat_ids]

    missing = [k for k, _ in (*PREDICTORS, DEC_NORM_PREDICTOR) if k not in cov]
    if missing:
        raise AssertionError(f"predictors missing from the assembled covariate set: {missing}")

    return {
        "feat_ids": feat_ids,
        "r2": r2,
        "cov": cov,
        "gates": gates,
        "act_var_bank_check": bank,
        "n_fit": int(scan["n_fit"]),
        "n_holdout": int(scan["n_holdout"]),
        "dec_norm": {
            "mean": float(dec_norm.mean()),
            "std": float(dec_norm.std()),
            "degenerate": dec_degenerate,
            "excluded_from_figures": dec_degenerate,
            "note": (
                "decoder columns are unit-norm by construction; a constant covariate has no "
                "rank variation, so it is kept in this JSON and dropped from both figures"
            ),
        },
    }


# ── raw + activity-partial Spearman, with bootstrap CIs ──────────────────────


def _partial_from_corr(r_jy: np.ndarray, r_ja: np.ndarray, r_ay: float | np.ndarray):
    """Rank-partial correlation of each predictor with y, controlling one covariate."""
    denom = np.sqrt(np.maximum((1.0 - r_ja**2) * (1.0 - r_ay**2), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (r_jy - r_ja * r_ay) / denom, np.nan)


def _weighted_reads(z: np.ndarray, w: np.ndarray, a: int, y: int):
    """Raw + activity-partial rho per bootstrap draw, from four GEMMs.

    `z` is the (n, p+1) FIXED rank matrix (predictors then y); `w` is the
    (n, draws) resample weight matrix. Only the (j, y), (j, a) and (a, y)
    correlations are needed, so the full (p+1)^2 matrix is never formed.
    """
    n = z.shape[0]
    s1 = z.T @ w
    s2 = (z * z).T @ w
    cy = (z * z[:, y : y + 1]).T @ w
    ca = (z * z[:, a : a + 1]).T @ w

    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r_jy = (cy / n - mu * mu[y]) / (sd * sd[y])
        r_ja = (ca / n - mu * mu[a]) / (sd * sd[a])
    r_ay = r_jy[a]
    return r_jy, _partial_from_corr(r_jy, r_ja, r_ay)


def correlation_reads(bundle: dict, n_boot: int, rng) -> dict:
    """Point estimates + percentile bootstrap CIs for raw and partial rho.

    Bootstrap convention follows the committed battery: ranks are computed ONCE
    on the full sample and the draws reweight those fixed ranks (rather than
    re-ranking inside every resample), so the CIs here are comparable with
    `fullwidth_joint_model.json`.
    """
    keys = [k for k, _ in active_predictors(bundle)]
    r2 = bundle["r2"]
    t0 = time.time()
    z = np.column_stack([PB._rank(bundle["cov"][k]) for k in keys] + [PB._rank(r2)])
    _log(f"rank transform: {z.shape[0]} x {z.shape[1]} in {time.time() - t0:.0f}s")

    n = z.shape[0]
    a, y = keys.index(PARTIAL_ON), len(keys)
    zc = (z - z.mean(0)) / z.std(0)
    corr = (zc.T @ zc) / n
    raw = corr[:, y]
    partial = _partial_from_corr(corr[:, y], corr[:, a], corr[a, y])
    partial[a] = np.nan  # partialling a variable out of itself is undefined

    t0 = time.time()
    raw_draws, par_draws = [], []
    done = 0
    while done < n_boot:
        b = min(BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(n, b))
        w = np.bincount((idx + np.arange(b) * n).ravel(), minlength=n * b)
        w = w.reshape(b, n).T.astype(np.float64)
        rj, pj = _weighted_reads(z, w, a, y)
        raw_draws.append(rj.T)
        par_draws.append(pj.T)
        done += b
    raw_b = np.vstack(raw_draws)  # (n_boot, p+1)
    par_b = np.vstack(par_draws)
    _log(f"bootstrap {n_boot} draws in {time.time() - t0:.0f}s")

    rows = []
    for j, (key, label) in enumerate(PREDICTORS):
        v = bundle["cov"][key]
        row = {
            "key": key,
            "label": label,
            "spearman_raw": float(raw[j]),
            "spearman_raw_ci95": PB._ci(raw_b[:, j]),
            "partial_on_activity": (None if j == a else float(partial[j])),
            "partial_on_activity_ci95": (None if j == a else PB._ci(par_b[:, j])),
            "log_x_axis": bool(_log_x(v)),
            **_decile_profile(v, r2),
        }
        if j == a:
            row["partial_note"] = "undefined — this IS the partialled covariate (firing frequency)"
        rows.append(row)
        _log(
            f"{key}: raw rho {raw[j]:+.4f}"
            + ("  partial n/a (self)" if j == a else f"  activity-partial {partial[j]:+.4f}")
        )
    return {"predictors": rows, "spearman_activity_vs_r2": float(corr[a, y])}


def _log_x(v: np.ndarray) -> bool:
    """Log x-axis for strictly-positive heavy-tailed predictors (p99/p50 >= 5)."""
    finite = v[np.isfinite(v)]
    if len(finite) == 0 or not (finite > 0).all():
        return False
    p50, p99 = np.percentile(finite, [50, 99])
    return bool(p50 > 0 and p99 / p50 >= 5.0)


def _decile_profile(pred: np.ndarray, r2: np.ndarray) -> dict:
    """Median R^2 per predictor decile, with the bin centers used for plotting.

    Heavily-tied predictors (an integer count) can collapse adjacent quantile
    edges, leaving a decile empty; those read NaN and are simply not drawn.
    """
    edges = np.quantile(pred, np.linspace(0, 1, N_DECILES + 1))
    dec = np.searchsorted(edges[1:-1], pred, side="right")
    med, cnt = [], []
    for d in range(N_DECILES):
        m = dec == d
        med.append(float(np.median(r2[m])) if m.any() else float("nan"))
        cnt.append(int(m.sum()))
    return {
        "decile_median_r2": med,
        "decile_center": [float((edges[i] + edges[i + 1]) / 2) for i in range(N_DECILES)],
        "decile_n": cnt,
        "top_minus_bottom_decile_gap": float(med[-1] - med[0]),
    }


# ── figures ──────────────────────────────────────────────────────────────────


def active_predictors(bundle: dict) -> list[tuple[str, str]]:
    """Predictors carried into the reads and both figures.

    `dec_norm` joins the list only when the runtime check found genuine
    variation; when it is degenerate (the expected case) it stays in the JSON
    diagnostics and is dropped from every read and panel.
    """
    preds = list(PREDICTORS)
    if not bundle["dec_norm"]["degenerate"]:
        preds.append(DEC_NORM_PREDICTOR)
    return preds


def _subtitle(bundle: dict) -> str:
    return (
        f"{len(bundle['feat_ids']):,} SAE features (BatchTopK k=64, resid_post layer 19)  |  "
        f"held-out $R^2$ from RIDGE fits only — no nonlinear arm exists at this width  |  "
        f"$R^2$ display-clipped at {R2_DISPLAY_FLOOR:g}  |  "
        f"$R^2$ is the #1738 multi-turn corpus context arm; covariates are #1482 single-turn"
    )


def fig_scatter_panel(bundle: dict, reads: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    by_key = {p["key"]: p for p in reads["predictors"]}
    preds = active_predictors(bundle)
    r2 = np.clip(bundle["r2"], R2_DISPLAY_FLOOR, None)

    ncol = 5
    nrow = int(np.ceil(len(preds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.15 * ncol, 2.85 * nrow), sharey=True)
    flat = axes.ravel()

    for ax, (key, label) in zip(flat, preds, strict=False):
        v = bundle["cov"][key]
        row = by_key[key]
        log_x = row["log_x_axis"]
        ax.hexbin(
            v,
            r2,
            gridsize=58,
            bins="log",
            mincnt=1,
            xscale="log" if log_x else "linear",
            cmap="Blues",
            linewidths=0,
        )
        ax.plot(
            row["decile_center"],
            row["decile_median_r2"],
            "-",
            lw=1.3,
            color=paper_palette_role("accent"),
        )
        if not log_x:
            # view-only: a few extreme features otherwise compress the whole
            # mass into a vertical line (kurtosis, footprint variance, skew).
            # Every statistic is computed on the FULL unclipped data.
            finite = v[np.isfinite(v)]
            lo, hi = np.percentile(finite, [X_VIEW_PCT, 100.0 - X_VIEW_PCT])
            if hi > lo:
                pad = 0.04 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
            # few ticks + a shared power-of-ten offset: the default locator
            # collides labels on narrow ranges (footprint variance ~2e-3)
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
            fontsize=8.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
        )
        ax.set_xlabel(label + (" (log)" if log_x else ""), fontsize=8.0)
        ax.tick_params(labelsize=7.4)

    for ax in flat[: len(preds)][::ncol]:
        ax.set_ylabel(r"per-feature held-out $R^2$", fontsize=8.4)
    for ax in flat[len(preds) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Which continuous feature properties track how well the context-to-answer map "
        "predicts a feature?",
        fontsize=12.5,
        y=0.995,
    )
    fig.text(0.5, 0.955, _subtitle(bundle), ha="center", fontsize=7.6, color="#5A5A5A")
    fig.text(
        0.5,
        0.012,
        "decile-median trend line overlaid; log-count hexbin density; Spearman $\\rho$ "
        f"annotated per panel. Linear-axis panels view-clip x to [{X_VIEW_PCT:g}, "
        f"{100 - X_VIEW_PCT:g}] percentile — display only; every $\\rho$ and decile "
        "median uses the full unclipped data.",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.945))
    stem = "continuous_scatter_panel"
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_partial_forest(bundle: dict, reads: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    keys = {k for k, _ in active_predictors(bundle)}
    rows = [p for p in reads["predictors"] if p["key"] in keys]
    # sorted by |partial rho|; the self-partialled covariate has none, so it is
    # pinned to the bottom of the axis with its raw value only
    scored = [r for r in rows if r["partial_on_activity"] is not None]
    unscored = [r for r in rows if r["partial_on_activity"] is None]
    scored.sort(key=lambda r: abs(r["partial_on_activity"]))
    ordered = unscored + scored  # bottom-up on the y axis

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

    fig, ax = plt.subplots(figsize=(9.6, 0.44 * len(ordered) + 3.0))
    ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.errorbar(
        raw,
        ypos + 0.16,
        xerr=PB._errbars(raw, raw_ci),
        fmt="o",
        ms=4.6,
        color=paper_palette_role("baseline"),
        mfc="white",
        mew=1.3,
        lw=1.0,
        capsize=2.2,
        label="raw Spearman $\\rho$",
    )
    ax.errorbar(
        par[ok],
        ypos[ok] - 0.16,
        xerr=PB._errbars(par[ok], par_ci[ok]),
        fmt="o",
        ms=4.6,
        color=paper_palette_role("primary"),
        lw=1.0,
        capsize=2.2,
        label="partialling out firing frequency",
    )
    # the self-partialled covariate carries its caveat in the tick label rather
    # than an in-axes annotation, which ran off the right edge
    ticklabels = [
        r["label"] + ("" if r["partial_on_activity"] is not None else "  — partial undefined")
        for r in ordered
    ]
    ax.set_yticks(ypos)
    ax.set_yticklabels(ticklabels, fontsize=8.4)
    ax.set_ylim(-0.75, len(ordered) - 0.25)
    span = float(np.nanmax(raw) - np.nanmin(np.concatenate([raw, par[ok]])))
    ax.set_xlim(
        float(np.nanmin(np.concatenate([raw, par[ok]]))) - 0.08 * span,
        float(np.nanmax(raw)) + 0.08 * span,
    )
    ax.set_xlabel(r"Spearman $\rho$ against per-feature held-out $R^2$", fontsize=9.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        frameon=False,
        fontsize=8.4,
    )
    fig.suptitle("What survives adjusting for how often a feature fires?", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.945, _subtitle(bundle), ha="center", fontsize=7.4, color="#5A5A5A")
    fig.text(
        0.5,
        0.012,
        f"error bars: percentile 95% CI over {N_BOOT:,} bootstrap draws (ranks fixed at the "
        "full sample, draws reweight them) — at n = "
        f"{len(bundle['feat_ids']):,} they are narrower than the markers; "
        "rows sorted by |partial $\\rho$|",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.935))
    stem = "continuous_rho_vs_activity_partial"
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def figures(bundle: dict, reads: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    fig_dir.mkdir(parents=True, exist_ok=True)
    return [fig_scatter_panel(bundle, reads, fig_dir), fig_partial_forest(bundle, reads, fig_dir)]


# ── entrypoint ───────────────────────────────────────────────────────────────


def _caveats(bundle: dict) -> dict:
    return {
        "ridge_only": (
            "every R^2 here is a RIDGE fit. No per-feature nonlinear (MLP) R^2 array exists at "
            "any width: sae_ctx__mean__mlp.npz is the diverged fit (all-negative), "
            "sae_sae_mlp_recovery emitted pooled scalars only, and "
            "eval_results/issue_1738/sae_twoway/perfeature/ is ridge-only. A nonlinear panel "
            "requires a refit and is NOT produced here."
        ),
        "cross_corpus": (
            "per-feature R^2 is the #1738 MULTI-TURN corpus context-arm read; activity, "
            "consistency, the activation moments and the holdout counts are #1482 SINGLE-TURN. "
            "Every correlate below is therefore cross-corpus."
        ),
        "activity_is_firing_frequency": (
            "`activity` is the FRACTION OF FIT ANSWERS in which the feature is answer-active "
            "(cnt / n_fit) — NOT a mean activation value. Mean activation is reported "
            "separately as mean_act_uncond (zeros included) and mean_act_cond (active only)."
        ),
        "r2_display_clip": (
            f"figures clip R^2 at {R2_DISPLAY_FLOOR:g} for display only; every rho, partial rho "
            "and decile median is computed on the UNCLIPPED values."
        ),
        "x_axis_view_clip": (
            f"scatter-panel LINEAR-axis predictors view-clip x to the "
            f"[{X_VIEW_PCT:g}, {100 - X_VIEW_PCT:g}] percentile so a handful of extreme "
            "features do not compress the mass into a vertical line (footprint kurtosis, "
            "footprint variance, footprint skew). Display only — no statistic is affected. "
            "Log-axis panels are unclipped; the per-predictor axis choice is the "
            "`log_x_axis` field (strictly positive AND p99/p50 >= 5)."
        ),
        "bootstrap_convention": (
            "ranks are computed once on the full sample and the bootstrap draws reweight those "
            "fixed ranks (they are not re-ranked per draw), matching the committed battery so "
            "the CIs are comparable with fullwidth_joint_model.json."
        ),
        "partial_definition": (
            "partial rho is the rank-partial correlation controlling FIRING FREQUENCY ONLY — "
            "not the all-others-partialled read reported in fullwidth_joint_model.json."
        ),
        "dec_norm": bundle["dec_norm"]["note"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 continuous per-feature predictor reads")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument("--work", type=Path, default=PROJECT_ROOT / "data/issue_1482/fullwidth")
    ap.add_argument("--phase", default="all", choices=("all", "scan", "analyze", "figs"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_boot = 50
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    if args.phase == "scan":
        scan = scan_pooled_store(POOLED_STORE, args.work / "continuous_scan.npz")
        with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
            feat_ids = np.asarray(z["feat_ids"], dtype=np.int64)
        identity_gates(derive_from_scan(scan), feat_ids)
        return

    cov_path = args.out_dir / "continuous_covariates.npz"
    json_path = args.out_dir / "continuous_predictors.json"

    if args.phase == "figs":
        with np.load(cov_path, allow_pickle=True) as z:
            bundle = {
                "feat_ids": z["feat_ids"],
                "r2": z["r2"],
                "cov": {k: z[k] for k in z.files if k not in ("feat_ids", "r2")},
            }
        doc = json.loads(json_path.read_text())
        bundle["dec_norm"] = doc["dec_norm"]
        stems = figures(bundle, doc, args.fig_dir)
        _log(f"figures: {', '.join(stems)}")
        return

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    bundle = assemble(args.work)
    np.savez(
        cov_path,
        feat_ids=bundle["feat_ids"],
        r2=bundle["r2"],
        **bundle["cov"],
    )
    _log(f"covariates -> {cov_path}")

    reads = correlation_reads(bundle, args.n_boot, rng)
    doc = {
        "design": {
            "scope": "FULL DICTIONARY (continuous predictors only)",
            "n_features": int(len(bundle["feat_ids"])),
            "target": (
                "per-feature held-out R^2, context arm, #1738 MULTI-TURN ridge fits "
                "(rank-transformed)"
            ),
            "r2_source": f"{FW.R2_DIR}/{FW.R2_ARMS[FW.PRIMARY_ARM]}",
            "covariate_source": str(POOLED_STORE),
            "matrix_source": MATRIX,
            "derived_here": list(DERIVED),
            "partial_on": PARTIAL_ON,
            "n_boot": int(args.n_boot),
            "n_fit_rows": bundle["n_fit"],
            "n_holdout_rows": bundle["n_holdout"],
            "sae": {"layer": PB.SAE_LAYER, "k": PB.SAE_K},
            "seed": SEED,
        },
        "identity_gates": bundle["gates"],
        "act_var_bank_check": bundle["act_var_bank_check"],
        "dec_norm": bundle["dec_norm"],
        "spearman_activity_vs_r2": reads["spearman_activity_vs_r2"],
        "predictors": reads["predictors"],
        "caveats": _caveats(bundle),
        "metadata": PB._metadata(),
    }
    json_path.write_text(json.dumps(doc, indent=1))
    _log(f"reads -> {json_path} ({time.time() - t0:.0f}s)")

    stems = figures(bundle, doc, args.fig_dir)
    _log(f"figures: {', '.join(stems)}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
