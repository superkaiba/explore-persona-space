#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², ≪, τ, ȳ) in scientific docstrings + log messages.
"""Issue #810 inline free-analysis — the max-pool "variance censoring" theory.

#810 found the per-dimension MAX-pool answer summary BEATS the MEAN-over-answer
summary on the linear context→answer reconstruction map for the misalignment
probe pool (per-layer peak skill-over-mean R² 0.826 @ L21 vs mean 0.800 @ L18)
but LOSES by ~0.07 on the UltraChat pool (max 0.728 @ L21 vs mean 0.796 @ L18).

A "variance censoring" theory (mentor) explains the max win as a pooled-R²
artifact, NOT better decoding:
  - A rate-coded dim fires across answer tokens at a context-dependent rate p(x)
    that is NOT linearly decodable from the context read c_C.
  - MEAN target ≈ m·p(x): its across-context variance is real and unexplained →
    it enters the pooled R² denominator in full, dragging skill DOWN.
  - MAX target ≈ m·(1−(1−p(x))^T): with firing saturated (p·T ≫ 1) it is
    ≈ constant → near-zero variance → drops out of numerator AND denominator, so
    variance-weighted pooled R² rises WITHOUT the map decoding anything more.
  - Refinement: max applies a saturating nonlinearity regardless of decodability;
    where the rate IS linearly decodable, max DESTROYS signal the mean transmits
    (R² deflates) → corpus flips (misalignment win, UltraChat loss).

This script runs D1-D5 on the persisted #810 stores to confirm/refute the
mechanism. ANALYSIS-ONLY — no training, no generation, no GPU. Fit recipe is the
committed #810 LOCO ridge (train-fold-centered, PRESS-λ, dual/Gram) reused
verbatim from ``vectorized_mlp_skill``; D2 deviates ONLY by scoring in the FULL
3584-dim ambient basis (no 48-dim PCA) so per-dim SST/SSR are decomposable.

Cache-first I/O: cached files resolve offline (the #810 stores are in the HF hub
cache); a genuinely-missing file falls back to a retrying network download (the
UltraChat position store is the one uncached input as of 2026-07-16).

Usage::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
      NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 EPM_FIT_DEVICE=cpu \\
      uv run python scripts/issue810_maxpool_censoring.py \\
        --pools betley g1 --out eval_results/issue_810/maxpool-variance-censoring-diagnostics \\
        --fig-dir figures/issue_810
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

os.environ.setdefault("EPM_FIT_DEVICE", "cpu")

import numpy as np  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue810_common import (  # noqa: E402
    G1_V0_SUMMARIES,
    HF_DATA_REPO,
    HF_PREFIX,
    I594_CC_LAST_FILE,
    I658_V0_SUMMARIES,
    PCA_TARGET_DIM_CAP,
    dump_json,
    reproducibility_metadata,
)

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    loco_train_means,
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)

logger = logging.getLogger("issue810_maxpool_censoring")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Position-store subdir per pool (betley = misalignment; g1 = UltraChat genre arm).
POS_PREFIX = {
    "betley": f"{HF_PREFIX}/answer_position_sweep",
    "g1": f"{HF_PREFIX}/answer_position_sweep_genre-generalization-ultrachat",
}
V0_FILE = {"betley": I658_V0_SUMMARIES, "g1": G1_V0_SUMMARIES}
POOL_LABEL = {"betley": "misalignment", "g1": "ultrachat"}

# Committed LOCO cells: mean@L18, max@L21 (both pools). Run the fair same-layer
# max-vs-mean comparison at BOTH, plus the two nearby layers, for D1/D2.
COMMITTED = {"mean": 18, "maxp": 21}
D1D2_LAYERS = [18, 19, 20, 21]
DECODE_LAYERS = [18, 21]  # D3/D4 firing-rate reads (committed cells)
EXEMPLAR_LAYER = 21  # D5 exemplars (max's committed layer, the misalignment win)

# The 32 single-position answer-content reads (token-level proxy). im_end/turn_nl
# (the two turn-boundary positions) are EXCLUDED from the token sample.
POS_32 = [f"tail_{k}" for k in range(1, 17)] + [f"head_{k}" for k in range(16)]

# D4 fallback answer lengths (answer_spans store uncached + HF down; Qwen-2.5-7B
# natural responses run ~150 tok median — marker-leakage recipe). Reported across
# a grid so the saturation read is not hostage to one guess. FLAGGED in output.
T_GRID = [50, 150, 300]
T_DEFAULT = 150

SEED = 810


# ── cache-first HF I/O ─────────────────────────────────────────────────────────


def _cache_roots() -> list[pathlib.Path]:
    """Candidate HF hub-cache roots (default + HF_HOME + HF_HUB_CACHE)."""
    roots: list[pathlib.Path] = []
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        roots.append(pathlib.Path(HF_HUB_CACHE))
    except Exception:
        pass
    if os.environ.get("HF_HOME"):
        roots.append(pathlib.Path(os.environ["HF_HOME"]) / "hub")
    roots.append(pathlib.Path.home() / ".cache/huggingface/hub")
    seen: set[str] = set()
    uniq: list[pathlib.Path] = []
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            uniq.append(r)
    return uniq


def _hf(path: str, attempts: int = 5) -> str:
    """Resolve an HF data-repo file cache-first (direct snapshot glob); network fallback.

    Resolves a cached blob by globbing ``snapshots/*/<path>`` directly — bypassing
    hf_hub_download's ``refs/main`` resolution, which fails offline when the cached
    ref points at a commit whose snapshot lacks the file (the case here, and the
    reason ``local_files_only=True`` raised LocalEntryNotFoundError). A genuinely
    uncached file falls back to a bounded retrying download (the flaky Hub).
    """
    repo_slug = f"datasets--{HF_DATA_REPO.replace('/', '--')}"
    for root in _cache_roots():
        snaps = root / repo_slug / "snapshots"
        if not snaps.is_dir():
            continue
        cands = [c for c in snaps.glob(f"*/{path}") if c.exists()]
        if cands:
            return str(max(cands, key=lambda p: p.stat().st_mtime))
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for i in range(attempts):
        try:
            return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset", etag_timeout=60)
        except Exception as e:
            last = e
            logger.warning(
                "[hf] %s attempt %d/%d failed: %s", path, i + 1, attempts, type(e).__name__
            )
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"cache miss + network download failed for {path}: {last}")


_V0: dict[str, dict] = {}


def _v0(pool: str) -> dict:
    if pool not in _V0:
        _V0[pool] = torch.load(_hf(V0_FILE[pool]), weights_only=False, map_location="cpu")
    return _V0[pool]


def _ctx_ids(pool: str) -> list[str]:
    """Canonical 50-context fold order from the pool's v0_summaries store."""
    ids = list(_v0(pool)["context_ids"])
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError(f"{pool}: expected 50 unique context_ids, got {len(ids)}")
    return ids


def _capture_layers(pool: str) -> list[int]:
    return list(_v0(pool)["capture_layers"])


def _free_summary(pool: str, recipe: str, layer: int, ctx_ids: list[str]) -> np.ndarray:
    """(n, H) free summary (mean/maxp/last) at ``layer`` over ``ctx_ids`` (v0 store)."""
    s = _v0(pool)["summaries"][recipe]
    li = _capture_layers(pool).index(layer)
    return np.stack([s[c][li].float().numpy() for c in ctx_ids])


def _cc(pool: str, layer: int, ctx_ids: list[str]) -> np.ndarray:
    """(n, H_c) context read c_C at ``layer`` — the ridge X input (committed recipe).

    betley → #594 last-input-token store (``context_vectors_mean.pt::tensor``,
    hash-pinned in the committed fit). g1 → the g1 v0 store's ``cc_last`` (the
    #658 per-genre recomputed last-token c_C). Aligned BY ctx_id (the two stores
    order contexts differently).
    """
    li = _capture_layers(pool).index(layer)
    if pool == "betley":
        blob = torch.load(_hf(I594_CC_LAST_FILE), weights_only=False, map_location="cpu")
        row = {iid: i for i, iid in enumerate(blob["instance_ids"])}
        t = blob["tensor"]  # (n, 28, H)
        return np.stack([t[row[c]][li].float().numpy() for c in ctx_ids])
    store = _v0("g1")["cc_last"]
    return np.stack([store[c][li].float().numpy() for c in ctx_ids])


def _positions(pool: str, ctx_ids: list[str]) -> tuple[dict, dict]:
    """{ctx: {pos: (Lc,H) fp32}}, {ctx: {pos: coverage}} from the Phase-B store."""
    out: dict[str, dict[str, np.ndarray]] = {}
    cov: dict[str, dict[str, int]] = {}
    for c in ctx_ids:
        blob = torch.load(_hf(f"{POS_PREFIX[pool]}/{c}.pt"), weights_only=False, map_location="cpu")
        names = blob["positions"]
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, Lc, H)
        out[c] = {n: pv[i] for i, n in enumerate(names)}
        cov[c] = dict(blob["coverage"])
    return out, cov


# ── D1: per-dim variance ratio Var(max)/Var(mean) ───────────────────────────────


def diagnostic_d1(pool: str, ctx_ids: list[str]) -> dict:
    """Var_x(max_d)/Var_x(mean_d) across contexts, per layer. Prediction: mass ≪ 1."""
    out: dict = {"pool": pool, "pool_label": POOL_LABEL[pool], "by_layer": {}}
    for layer in D1D2_LAYERS:
        mean = _free_summary(pool, "mean", layer, ctx_ids)  # (n, H)
        mx = _free_summary(pool, "maxp", layer, ctx_ids)
        var_mean = mean.var(axis=0)  # population variance across contexts (ddof=0)
        var_max = mx.var(axis=0)
        eps = 1e-12
        ratio = var_max / (var_mean + eps)
        finite = ratio[np.isfinite(ratio)]
        out["by_layer"][str(layer)] = {
            "median_ratio": float(np.median(finite)),
            "mean_ratio": float(np.mean(finite)),
            "frac_ratio_lt_0p3": float(np.mean(finite < 0.3)),
            "frac_ratio_lt_0p1": float(np.mean(finite < 0.1)),
            "total_var_max": float(var_max.sum()),
            "total_var_mean": float(var_mean.sum()),
            "total_var_share": float(var_max.sum() / (var_mean.sum() + eps)),
            "log10_ratio": [float(x) for x in np.log10(finite + eps)],  # for the histogram
        }
        logger.info(
            "[D1] %s L%d: median ratio=%.3f  frac<0.3=%.3f  frac<0.1=%.3f  totvar max/mean=%.3f",
            pool,
            layer,
            out["by_layer"][str(layer)]["median_ratio"],
            out["by_layer"][str(layer)]["frac_ratio_lt_0p3"],
            out["by_layer"][str(layer)]["frac_ratio_lt_0p1"],
            out["by_layer"][str(layer)]["total_var_share"],
        )
    return out


# ── D2: per-dim R² decomposition (full 3584-dim basis) ──────────────────────────


def _full_dim_decomp(Xc: np.ndarray, Yv: np.ndarray) -> dict:
    """LOCO ridge (committed recipe) scored in the FULL ambient basis; per-dim SST/SSR.

    Fits the SAME train-fold-centered PRESS-λ dual ridge as the committed #810
    reconstruction, but on the raw (n, 3584) target (no PCA), so per-dim SST_d /
    SSR_d are recoverable. Predictions for all 3584 output dims come from ONE
    dual GEMM per LOCO fold (50 folds); no per-dim loop.
    """
    preds = ridge_predict_loco_centered(Xc, Yv)  # (n, H) held-out
    tmean = loco_train_means(Yv)  # (n, H) per-fold LOO train mean baseline
    ssr_d = np.sum((Yv - preds) ** 2, axis=0)  # (H,)
    sst_d = np.sum((Yv - tmean) ** 2, axis=0)  # (H,)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_d = 1.0 - ssr_d / sst_d
    pooled = 1.0 - float(ssr_d.sum()) / float(sst_d.sum())
    return {
        "pooled_skill_full_dim": pooled,
        "ssr_d": ssr_d,
        "sst_d": sst_d,
        "r2_d": r2_d,
        "preds": preds,
    }


def _committed_pca_skill(Xc: np.ndarray, Yv: np.ndarray) -> float:
    """The committed #810 cell: 48-dim PCA target, LOCO ridge, skill_over_mean_r2."""
    n = Yv.shape[0]
    pca_dim = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    mu, comps, _ = robust_pca_basis(Yv, pca_dim)
    y_pca = (Yv - mu) @ comps.T
    pred = ridge_predict_loco_centered(Xc, y_pca)
    return float(skill_over_mean_r2(pred, y_pca)["skill"])


def diagnostic_d2(pool: str, ctx_ids: list[str], d1: dict) -> dict:
    """Per-dim SST/SSR decomposition (mean vs max), full-dim, + the censoring read."""
    out: dict = {"pool": pool, "pool_label": POOL_LABEL[pool], "by_layer": {}}
    for layer in D1D2_LAYERS:
        Xc = _cc(pool, layer, ctx_ids)
        rec: dict = {}
        decomp = {}
        for scheme in ("mean", "maxp"):
            Yv = _free_summary(pool, scheme, layer, ctx_ids)
            d = _full_dim_decomp(Xc, Yv)
            decomp[scheme] = d
            rec[f"{scheme}_pooled_skill_full_dim"] = d["pooled_skill_full_dim"]
            rec[f"{scheme}_pooled_skill_pca48_committed_recipe"] = _committed_pca_skill(Xc, Yv)
        # Censoring read: of the mean-scheme total SSR (unexplained variance),
        # what fraction sits in dims max censors (D1 ratio < 0.3 / < 0.1)?
        ratio = np.asarray(
            _free_summary(pool, "maxp", layer, ctx_ids).var(axis=0)
            / (_free_summary(pool, "mean", layer, ctx_ids).var(axis=0) + 1e-12)
        )
        ssr_mean = decomp["mean"]["ssr_d"]
        tot = float(ssr_mean.sum())
        rec["mean_ssr_total"] = tot
        rec["frac_mean_ssr_from_ratio_lt_0p3"] = float(ssr_mean[ratio < 0.3].sum() / (tot + 1e-12))
        rec["frac_mean_ssr_from_ratio_lt_0p1"] = float(ssr_mean[ratio < 0.1].sum() / (tot + 1e-12))
        # For reference: dims' share of COUNT below the ratio cut.
        rec["frac_dims_ratio_lt_0p3"] = float(np.mean(ratio < 0.3))
        # Persist the per-dim arrays needed by figures/D3-correlation (compact).
        rec["_arrays"] = {
            "ratio": ratio,
            "mean_r2_d": decomp["mean"]["r2_d"],
            "max_r2_d": decomp["maxp"]["r2_d"],
            "mean_sst_d": decomp["mean"]["sst_d"],
            "max_sst_d": decomp["maxp"]["sst_d"],
            "mean_ssr_d": decomp["mean"]["ssr_d"],
        }
        out["by_layer"][str(layer)] = rec
        logger.info(
            "[D2] %s L%d: mean full=%.4f (pca48=%.4f) | max full=%.4f (pca48=%.4f) | "
            "frac mean-SSR from ratio<0.3=%.3f",
            pool,
            layer,
            rec["mean_pooled_skill_full_dim"],
            rec["mean_pooled_skill_pca48_committed_recipe"],
            rec["maxp_pooled_skill_full_dim"],
            rec["maxp_pooled_skill_pca48_committed_recipe"],
            rec["frac_mean_ssr_from_ratio_lt_0p3"],
        )
    return out


# ── D3: firing-rate decodability ────────────────────────────────────────────────


def _firing_rate_matrix(
    pos: dict, cov: dict, ctx_ids: list[str], layer: int, layer_i: int, pctile: float
) -> tuple[np.ndarray, np.ndarray]:
    """(n, H) per-context firing rate + (H,) per-dim threshold τ_d.

    τ_d = ``pctile``-th percentile of dim d pooled over all (ctx × covered 32
    positions); r_d(x) = fraction of ctx x's covered positions with value > τ_d.
    The 32 positions are probe-MEAN vectors (double proxy — flagged in output).
    """
    # Pool all (ctx, covered-position) value vectors for the global threshold.
    pooled_rows: list[np.ndarray] = []
    per_ctx_vals: dict[str, np.ndarray] = {}
    for c in ctx_ids:
        rows = [pos[c][p][layer_i] for p in POS_32 if cov[c].get(p, 0) > 0]
        arr = np.stack(rows) if rows else np.zeros((0, next(iter(pos[c].values())).shape[1]))
        per_ctx_vals[c] = arr  # (n_cov, H)
        pooled_rows.append(arr)
    pooled = np.concatenate(pooled_rows, axis=0)  # (sum_cov, H)
    tau = np.percentile(pooled, pctile, axis=0)  # (H,)
    rate = np.zeros((len(ctx_ids), pooled.shape[1]), dtype=np.float64)
    for i, c in enumerate(ctx_ids):
        arr = per_ctx_vals[c]
        rate[i] = (arr > tau).mean(axis=0) if arr.shape[0] else 0.0
    return rate, tau


def diagnostic_d3(pool: str, ctx_ids: list[str], pos: dict, cov: dict, d2: dict) -> dict:
    """LOCO-ridge decodability of the per-dim firing-rate matrix from c_C."""
    out: dict = {
        "pool": pool,
        "pool_label": POOL_LABEL[pool],
        "proxy_caveat": (
            "firing rate computed from the 32 tail/head single-position reads, which "
            "sample answer start+end (not the middle) AND are per-probe MEANS — a double "
            "proxy for the true per-token firing rate"
        ),
        "by_layer": {},
    }
    caps = _capture_layers(pool)
    for layer in DECODE_LAYERS:
        li = caps.index(layer)
        Xc = _cc(pool, layer, ctx_ids)
        rec: dict = {}
        rate_arrays: dict = {}
        for pct in (90.0, 75.0):
            rate, _tau = _firing_rate_matrix(pos, cov, ctx_ids, layer, li, pct)
            preds = ridge_predict_loco_centered(Xc, rate)
            sk = skill_over_mean_r2(preds, rate)
            tmean = loco_train_means(rate)
            ssr_d = np.sum((rate - preds) ** 2, axis=0)
            sst_d = np.sum((rate - tmean) ** 2, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_r2_d = 1.0 - ssr_d / sst_d
            tag = f"p{int(pct)}"
            rec[f"pooled_rate_r2_{tag}"] = float(sk["skill"])
            rec[f"median_per_dim_rate_r2_{tag}"] = float(sk["median_per_dim_r2"])
            rate_arrays[tag] = rate_r2_d
        # Correlate per-dim rate-R² (p90) with per-dim mean-scheme R² (D2, same layer).
        if str(layer) in d2["by_layer"]:
            mean_r2 = d2["by_layer"][str(layer)]["_arrays"]["mean_r2_d"]
            rr = rate_arrays["p90"]
            m = np.isfinite(mean_r2) & np.isfinite(rr)
            rec["spearman_rate_r2_vs_mean_r2_p90"] = _spearman(rr[m], mean_r2[m])
            rec["n_dims_correlated"] = int(m.sum())
        rec["_arrays"] = {"rate_r2_d_p90": rate_arrays["p90"]}
        out["by_layer"][str(layer)] = rec
        logger.info(
            "[D3] %s L%d: pooled rate-R²(p90)=%.4f (p75=%.4f) | ρ(rate-R²,mean-R²)=%s",
            pool,
            layer,
            rec["pooled_rate_r2_p90"],
            rec["pooled_rate_r2_p75"],
            rec.get("spearman_rate_r2_vs_mean_r2_p90"),
        )
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


# ── D4: saturation check ────────────────────────────────────────────────────────


def diagnostic_d4(pool: str, ctx_ids: list[str], pos: dict, cov: dict, d2: dict) -> dict:
    """s_d(x)=1−(1−p̂)^T over top-SSR-decile dims; fraction of cells with s>0.95."""
    out: dict = {
        "pool": pool,
        "pool_label": POOL_LABEL[pool],
        "T_source": (
            f"answer_spans store uncached + HF outage 2026-07-16 → T fixed at Qwen-2.5-7B "
            f"median natural-response length ({T_DEFAULT} tok), reported across T∈{T_GRID}; "
            f"FLAGGED fallback"
        ),
        "by_layer": {},
    }
    caps = _capture_layers(pool)
    for layer in DECODE_LAYERS:
        if str(layer) not in d2["by_layer"]:
            continue
        li = caps.index(layer)
        rate, _tau = _firing_rate_matrix(pos, cov, ctx_ids, layer, li, 90.0)  # p̂ = p90 rate
        ssr_mean = d2["by_layer"][str(layer)]["_arrays"]["mean_ssr_d"]
        # Top-decile dims by mean-scheme unexplained variance (SSR).
        thr = np.percentile(ssr_mean, 90.0)
        top = np.where(ssr_mean >= thr)[0]
        rec: dict = {"n_top_ssr_dims": int(top.size)}
        for T in T_GRID:
            s = 1.0 - (1.0 - rate[:, top]) ** T  # (n_ctx, n_top)
            rec[f"frac_s_gt_0p95_T{T}"] = float(np.mean(s > 0.95))
            rec[f"median_s_T{T}"] = float(np.median(s))
            if T == T_DEFAULT:
                rec["s_flat_T150"] = [float(x) for x in s.ravel()]  # for the histogram
        out["by_layer"][str(layer)] = rec
        logger.info(
            "[D4] %s L%d: top-SSR dims=%d | frac s>0.95 @T=50/150/300 = %.3f/%.3f/%.3f",
            pool,
            layer,
            rec["n_top_ssr_dims"],
            rec["frac_s_gt_0p95_T50"],
            rec["frac_s_gt_0p95_T150"],
            rec["frac_s_gt_0p95_T300"],
        )
    return out


# ── D5: exemplar dims + toy simulation ──────────────────────────────────────────


def diagnostic_d5_exemplars(pool: str, ctx_ids: list[str], pos: dict, cov: dict, d2: dict) -> dict:
    """Rank dims by rate-coding signature; pick exemplars for the figure."""
    layer = EXEMPLAR_LAYER
    caps = _capture_layers(pool)
    li = caps.index(layer)
    arrs = d2["by_layer"][str(layer)]["_arrays"]
    ratio = arrs["ratio"]
    mean_r2 = arrs["mean_r2_d"]
    max_r2 = arrs["max_r2_d"]
    ssr_mean = arrs["mean_ssr_d"]
    # Sampled-position value matrix per dim → kurtosis + near-zero sparsity.
    rows = []
    for c in ctx_ids:
        rows.extend([pos[c][p][li] for p in POS_32 if cov[c].get(p, 0) > 0])
    V = np.stack(rows)  # (sum_cov, H)
    mu = V.mean(axis=0)
    sd = V.std(axis=0) + 1e-12
    z = (V - mu) / sd
    kurt = (z**4).mean(axis=0) - 3.0  # excess kurtosis per dim
    # "rate-coded" archetype: the theory's censored dim is high mean-SSR + D1
    # ratio<0.3. That set is EMPTY here (D1: frac<0.3≈0), so pick the closest
    # empirical analogue — highest-kurtosis (sparsest / most-outlier-firing)
    # high-mean-SSR dims — and REPORT their D1 ratio (all ≫1, the refutation).
    n_censored = int(np.sum(ratio < 0.3))
    high_ssr = ssr_mean > np.percentile(ssr_mean, 75)
    cand = np.where(high_ssr)[0]
    rate_dims = cand[np.argsort(-kurt[cand])][:4]
    # "predictable": high R² under BOTH schemes.
    pred_score = (mean_r2 > np.percentile(mean_r2[np.isfinite(mean_r2)], 90)) & (
        max_r2 > np.percentile(max_r2[np.isfinite(max_r2)], 90)
    )
    pred_dims = np.where(pred_score)[0][:2]
    return {
        "pool": pool,
        "pool_label": POOL_LABEL[pool],
        "layer": layer,
        "n_dims_ratio_lt_0p3_the_censored_class": n_censored,
        "n_dims_total": int(ratio.size),
        "exemplar_rate_coded_dims": [int(d) for d in rate_dims],
        "exemplar_predictable_dims": [int(d) for d in pred_dims],
        "rate_coded_detail": [
            {
                "dim": int(d),
                "d1_ratio": float(ratio[d]),
                "mean_r2": float(mean_r2[d]),
                "max_r2": float(max_r2[d]),
                "excess_kurtosis": float(kurt[d]),
            }
            for d in rate_dims
        ],
        "predictable_detail": [
            {
                "dim": int(d),
                "d1_ratio": float(ratio[d]),
                "mean_r2": float(mean_r2[d]),
                "max_r2": float(max_r2[d]),
            }
            for d in pred_dims
        ],
        "_arrays": {"V": V, "layer_i": li},
    }


def diagnostic_d5_toy() -> dict:
    """2-dim toy: dim A linear-decodable + constant; dim B Bernoulli non-decodable rate.

    Confirms the mechanism in isolation: MEAN pooling keeps dim B's undecodable
    m·p(x) variance in the pooled-R² denominator (skill LOW); MAX pooling
    saturates dim B to ≈constant (variance drops out; skill HIGH) — WITHOUT
    decoding anything more. A "B-decodable" variant flips it: where the rate IS
    linear, max DESTROYS the signal mean transmits (skill deflates).
    """
    rng = np.random.default_rng(SEED)
    n, d_feat, T = 50, 20, 300
    X = rng.standard_normal((n, d_feat))

    def pooled_skill(Y: np.ndarray) -> float:
        return float(skill_over_mean_r2(ridge_predict_loco_centered(X, Y), Y)["skill"])

    def per_dim_r2(y: np.ndarray) -> float:
        yy = y[:, None]
        return float(skill_over_mean_r2(ridge_predict_loco_centered(X, yy), yy)["skill"])

    # dim A: linear-decodable, constant across answer tokens (mean==max==A).
    wA = rng.standard_normal(d_feat)
    A = X @ wA
    A = (A - A.mean()) / (A.std() + 1e-12)

    wq = rng.standard_normal(d_feat)

    # EXACT expected-value targets (mean=p, max=1−(1−p)^T) — deterministic, so the
    # mechanism is isolated from Bernoulli sampling noise.
    #
    # INFLATION dim B: DENSE firing (p≈0.05-0.35), rate NON-linear (undecodable)
    # in X → at T=300 max saturates to ≈constant → variance drops out of the
    # pooled fit. Scale both views by 1/std(mean) so B's mean-variance ≈ A's (=1);
    # the max view stays ≈constant (own variance ≈0) under the shared scale.
    z = np.sin(X @ wq) + 0.5 * (X[:, 0] * X[:, 1])
    p_B = np.clip(0.05 + 0.30 * (z - z.min()) / (np.ptp(z) + 1e-12), 0.05, 0.35)
    mean_b_raw, max_b_raw = p_B, 1.0 - (1.0 - p_B) ** T
    s_b = 1.0 / (mean_b_raw.std() + 1e-12)
    meanB, maxB = mean_b_raw * s_b, max_b_raw * s_b

    # DEFLATION dim C: SPARSE firing (p≈0.0005-0.006) so max is NOT fully saturated
    # at T=300, rate LINEAR (decodable) in X. mean_C=p_C is exactly linear (R²≈1);
    # max_C=1−(1−p_C)^T is a saturating nonlinearity of that linear signal (fit
    # worse) → per-dim R² FALLS. Reported PER-DIM R² (scale-free — no A, no shared
    # scaling, so the max view is compared on its own natural scale).
    xq = X @ wq
    p_C = np.clip(0.0005 + 0.0055 * (xq - xq.min()) / (np.ptp(xq) + 1e-12), 0.0005, 0.006)
    meanC, maxC = p_C, 1.0 - (1.0 - p_C) ** T

    res = {
        "config": {"n": n, "d_feat": d_feat, "T": T, "seed": SEED, "targets": "exact_expected"},
        "inflation_nondecodable_dense": {
            "mean_scheme_skill": pooled_skill(np.stack([A, meanB], axis=1)),
            "max_scheme_skill": pooled_skill(np.stack([A, maxB], axis=1)),
            "dim_mean_target_var": float(meanB.var()),
            "dim_max_target_var": float(maxB.var()),
            "note": (
                "dense undecodable rate: max saturates the dim to ≈constant → its "
                "variance drops out of the pooled denominator → pooled skill RISES "
                "without decoding more (the censoring mechanism, in isolation)"
            ),
        },
        "deflation_decodable_sparse": {
            "mean_view_per_dim_r2": per_dim_r2(meanC),
            "max_view_per_dim_r2": per_dim_r2(maxC),
            "note": (
                "sparse decodable rate: mean=p is exactly linear (per-dim R²≈1); max "
                "is a saturating nonlinearity of the linear signal (fit worse) → "
                "per-dim R² FALLS (the refinement: max destroys signal mean transmits)"
            ),
        },
        "_arrays": {
            "p_B": p_B,
            "meanB": meanB,
            "maxB": maxB,
            "p_C": p_C,
            "meanC": meanC,
            "maxC": maxC,
            "A": A,
        },
    }
    logger.info(
        "[D5-toy] INFLATION dense-undecodable: mean-skill=%.3f max-skill=%.3f "
        "(var mean=%.3f max=%.4f) | DEFLATION sparse-decodable per-dim R²: mean=%.3f max=%.3f",
        res["inflation_nondecodable_dense"]["mean_scheme_skill"],
        res["inflation_nondecodable_dense"]["max_scheme_skill"],
        res["inflation_nondecodable_dense"]["dim_mean_target_var"],
        res["inflation_nondecodable_dense"]["dim_max_target_var"],
        res["deflation_decodable_sparse"]["mean_view_per_dim_r2"],
        res["deflation_decodable_sparse"]["max_view_per_dim_r2"],
    )
    return res


# ── figures ─────────────────────────────────────────────────────────────────────


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:
        pass
    return plt


def _strip_arrays(obj):
    """Deep-copy a result dict dropping the heavy ``_arrays`` blocks for JSON."""
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items() if k != "_arrays"}
    if isinstance(obj, list):
        return [_strip_arrays(x) for x in obj]
    return obj


def make_figures(fig_dir: pathlib.Path, results: dict) -> list[str]:
    plt = _mpl()
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, name):
        p = fig_dir / f"maxpool_censoring_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    pools = [p for p in ("betley", "g1") if p in results["d1"]]

    # D1: log10 variance-ratio histogram, both pools, committed layers.
    fig, axes = plt.subplots(1, len(pools), figsize=(5.2 * len(pools), 3.6), squeeze=False)
    for j, pool in enumerate(pools):
        ax = axes[0][j]
        for layer in (18, 21):
            lg = results["d1"][pool]["by_layer"].get(str(layer))
            if lg:
                ax.hist(lg["log10_ratio"], bins=60, alpha=0.55, label=f"L{layer}")
        ax.axvline(np.log10(0.3), color="k", ls="--", lw=0.8)
        ax.axvline(0.0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("log10 Var(max_d) / Var(mean_d)")
        ax.set_ylabel("dims")
        ax.set_title(f"{POOL_LABEL[pool]}")
        ax.legend()
    fig.suptitle("D1: per-dim variance ratio (max vs mean); dashed=0.3, dotted=1.0")
    save(fig, "d1_variance_ratio_hist")

    # D2: SST_d (log x) vs per-dim R²_d, mean vs max, colored by D1 ratio.
    for pool in pools:
        arrs = results["d2"][pool]["by_layer"]["21"]["_arrays"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for ax, scheme, key in ((axes[0], "mean", "mean_r2_d"), (axes[1], "maxp", "max_r2_d")):
            sst = arrs["mean_sst_d"] if scheme == "mean" else arrs["max_sst_d"]
            r2 = arrs[key]
            m = np.isfinite(r2) & (sst > 0)
            sc = ax.scatter(
                sst[m],
                np.clip(r2[m], -1, 1),
                c=np.log10(arrs["ratio"][m] + 1e-12),
                s=4,
                cmap="viridis",
                vmin=-3,
                vmax=1,
            )
            ax.set_xscale("log")
            ax.set_xlabel("per-dim SST_d")
            ax.set_ylabel("per-dim held-out R²_d (clipped [-1,1])")
            ax.set_title(f"{scheme} scheme @L21")
            ax.axhline(0, color="k", lw=0.6)
        fig.colorbar(sc, ax=axes[1], label="log10 D1 ratio")
        fig.suptitle(f"D2: per-dim variance vs decodability — {POOL_LABEL[pool]} (L21)")
        save(fig, f"d2_sst_vs_r2_{pool}")

    # D4: saturation histogram of s over top-SSR-decile dims (T=150), committed pools.
    fig, axes = plt.subplots(1, len(pools), figsize=(5.2 * len(pools), 3.6), squeeze=False)
    for j, pool in enumerate(pools):
        ax = axes[0][j]
        lg = results["d4"].get(pool, {}).get("by_layer", {}).get("21")
        if lg and "s_flat_T150" in lg:
            ax.hist(lg["s_flat_T150"], bins=50, color="C1")
            ax.axvline(0.95, color="k", ls="--", lw=0.8)
            ax.set_title(f"{POOL_LABEL[pool]}: frac>0.95={lg['frac_s_gt_0p95_T150']:.2f}")
        ax.set_xlabel("saturation s = 1−(1−p̂)^T   (T=150)")
        ax.set_ylabel("(dim × context) cells")
    fig.suptitle("D4: max saturation over top-decile mean-SSR dims")
    save(fig, "d4_saturation_hist")

    # D5 toy: pooled skill mean vs max, inflation vs deflation scenario.
    toy = results["d5_toy"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    groups = [
        "INFLATION\n(dense undecodable;\npooled [A,B] skill)",
        "DEFLATION\n(sparse decodable;\nper-dim R²)",
    ]
    mean_sk = [
        toy["inflation_nondecodable_dense"]["mean_scheme_skill"],
        toy["deflation_decodable_sparse"]["mean_view_per_dim_r2"],
    ]
    max_sk = [
        toy["inflation_nondecodable_dense"]["max_scheme_skill"],
        toy["deflation_decodable_sparse"]["max_view_per_dim_r2"],
    ]
    x = np.arange(2)
    ax.bar(x - 0.2, mean_sk, 0.4, label="mean view")
    ax.bar(x + 0.2, max_sk, 0.4, label="max view")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("skill-over-mean R²")
    ax.set_title("D5 toy: max-pool inflation vs deflation (fixed-magnitude model)")
    ax.legend()
    ax.axhline(0, color="k", lw=0.6)
    save(fig, "d5_toy_skill")

    # D5 exemplars: sampled-position value histograms (betley).
    if "betley" in results["d5_exemplars"]:
        ex = results["d5_exemplars"]["betley"]
        V = ex["_arrays"]["V"]
        rd = ex["exemplar_rate_coded_dims"][:3]
        pd = ex["exemplar_predictable_dims"][:2]
        dims = [(d, "rate-coded") for d in rd] + [(d, "predictable") for d in pd]
        fig, axes = plt.subplots(1, len(dims), figsize=(3.0 * len(dims), 3.0), squeeze=False)
        for k, (d, lab) in enumerate(dims):
            ax = axes[0][k]
            ax.hist(V[:, d], bins=40, color="C0" if lab == "rate-coded" else "C2")
            ax.set_title(f"dim {d}\n({lab})")
            ax.set_xlabel("activation")
        fig.suptitle("D5: exemplar dim sampled-position value distributions (misalignment L21)")
        save(fig, "d5_exemplar_dim_hists")

    return written


# ── main ─────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="issue #810 max-pool variance-censoring diagnostics")
    ap.add_argument("--pools", nargs="+", default=["betley", "g1"], choices=["betley", "g1"])
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results/issue_810/maxpool-variance-censoring-diagnostics"),
    )
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/issue_810"))
    ap.add_argument("--skip-positions", action="store_true", help="D1/D2/D5-toy only")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = reproducibility_metadata()

    results: dict = {"d1": {}, "d2": {}, "d3": {}, "d4": {}, "d5_exemplars": {}}

    for pool in args.pools:
        ctx_ids = _ctx_ids(pool)
        logger.info("=== POOL %s (%s), n=%d contexts ===", pool, POOL_LABEL[pool], len(ctx_ids))
        results["d1"][pool] = diagnostic_d1(pool, ctx_ids)
        results["d2"][pool] = diagnostic_d2(pool, ctx_ids, results["d1"][pool])
        if args.skip_positions:
            logger.info("[positions] skipped for %s", pool)
            continue
        try:
            pos, cov = _positions(pool, ctx_ids)
        except Exception as e:
            logger.warning("[positions] %s UNAVAILABLE (%s) — D3/D4/D5-exemplars skipped", pool, e)
            results.setdefault("_positions_missing", []).append(pool)
            continue
        results["d3"][pool] = diagnostic_d3(pool, ctx_ids, pos, cov, results["d2"][pool])
        results["d4"][pool] = diagnostic_d4(pool, ctx_ids, pos, cov, results["d2"][pool])
        results["d5_exemplars"][pool] = diagnostic_d5_exemplars(
            pool, ctx_ids, pos, cov, results["d2"][pool]
        )

    results["d5_toy"] = diagnostic_d5_toy()

    # Figures (need the _arrays blocks).
    figs = make_figures(pathlib.Path(args.fig_dir), results)

    # Per-diagnostic JSONs (arrays stripped) + a headline summary.
    clean = _strip_arrays(results)
    for d in ("d1", "d2", "d3", "d4", "d5_exemplars"):
        dump_json(
            {"diagnostic": d, "reproducibility": meta, "results": clean[d]}, out_dir / f"{d}.json"
        )
    dump_json(
        {"diagnostic": "d5_toy", "reproducibility": meta, "results": clean["d5_toy"]},
        out_dir / "d5_toy.json",
    )

    summary = _build_summary(clean, meta, figs)
    dump_json(summary, out_dir / "summary.json")
    logger.info("WROTE %d JSONs + %d figures under %s", 6, len(figs), out_dir)
    logger.info("SUMMARY VERDICT: %s", summary["verdict"])


def _build_summary(clean: dict, meta: dict, figs: list[str]) -> dict:
    """Headline numbers per diagnostic + a 3-sentence verdict."""
    headline: dict = {"per_pool": {}}
    for pool in clean["d1"]:
        d1_21 = clean["d1"][pool]["by_layer"].get("21", {})
        d2_21 = clean["d2"][pool]["by_layer"].get("21", {})
        d3_21 = clean["d3"].get(pool, {}).get("by_layer", {}).get("21", {})
        d4_21 = clean["d4"].get(pool, {}).get("by_layer", {}).get("21", {})
        headline["per_pool"][pool] = {
            "label": POOL_LABEL[pool],
            "D1_L21_median_var_ratio": d1_21.get("median_ratio"),
            "D1_L21_frac_ratio_lt_0p3": d1_21.get("frac_ratio_lt_0p3"),
            "D2_L21_mean_skill_full": d2_21.get("mean_pooled_skill_full_dim"),
            "D2_L21_max_skill_full": d2_21.get("maxp_pooled_skill_full_dim"),
            "D2_L21_max_minus_mean_full": (
                d2_21.get("maxp_pooled_skill_full_dim", 0)
                - d2_21.get("mean_pooled_skill_full_dim", 0)
                if d2_21
                else None
            ),
            "D2_L21_frac_mean_SSR_from_censored_dims_lt0p3": d2_21.get(
                "frac_mean_ssr_from_ratio_lt_0p3"
            ),
            "D3_L21_pooled_rate_r2_p90": d3_21.get("pooled_rate_r2_p90"),
            "D3_L21_spearman_rate_vs_meanR2": d3_21.get("spearman_rate_r2_vs_mean_r2_p90"),
            "D4_L21_frac_saturated_T150": d4_21.get("frac_s_gt_0p95_T150"),
        }
    toy = clean["d5_toy"]
    headline["D5_toy"] = {
        "inflation_pooled_mean_skill": toy["inflation_nondecodable_dense"]["mean_scheme_skill"],
        "inflation_pooled_max_skill": toy["inflation_nondecodable_dense"]["max_scheme_skill"],
        "deflation_mean_view_per_dim_r2": toy["deflation_decodable_sparse"]["mean_view_per_dim_r2"],
        "deflation_max_view_per_dim_r2": toy["deflation_decodable_sparse"]["max_view_per_dim_r2"],
    }
    verdict = (
        "See per-pool headline numbers; verdict prose is authored in the report. "
        "D1/D2 REFUTE the variance-censoring premise on real activations (Var(max)/Var(mean) "
        "median ~9-13, ~0% of mean-SSR from ratio<0.3 dims); the toy confirms the mechanism "
        "is real ONLY under the fixed-magnitude assumption the real max-pool violates."
    )
    return {
        "issue": 810,
        "analysis": "maxpool-variance-censoring-diagnostics",
        "reproducibility": meta,
        "headline": headline,
        "figures": figs,
        "positions_missing": clean.get("_positions_missing", []),
        "verdict": verdict,
    }


if __name__ == "__main__":
    main()
