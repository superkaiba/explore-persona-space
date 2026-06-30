#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #722 (off-pod CPU, 0 GPU): structural / robustness battery on the c_C → v0 map.

The #722 base result is a STRONG, predominantly-LINEAR map ``M̂: c_C → v0`` at the
mid-layer plateau (L12-L21, ridge skill-over-mean R² ≈ 0.70-0.80, peak L18 = 0.80;
the KRR-RBF nonlinear gap ≈ 0 there). This script characterizes that map along five
structural / robustness axes — all reusing the canonical #658/#722 LOCO ridge + KRR
machinery in ``explore_persona_space.analysis.vectorized_mlp_skill`` (NO re-implemented
math), all 0-GPU CPU, all on the SAME 50-context substrate the base run used:

1. **Rank / structure of M̂** — fit the global (non-LOCO) ridge map, SVD it, report the
   singular-value spectrum + effective rank (participation ratio, 90/95% energy), and
   project the top right/left singular directions onto the behavior read-out directions
   ``r_B`` (#658). Output: ``rank_structure.json`` + ``rank_spectrum.png``.

2. **Robustness sweep** — recompute the linear-ridge skill-over-mean R² (+ the KRR-RBF
   gap where cheap) under recipe variations (cross-genre betley↔ultrachat; cross-layer
   c_C@ℓ → v0@ℓ′ grid; c_C recipe last vs meanprompt; v0 summary mean vs last vs maxp).
   Output: ``robustness.json`` + ``robustness_heatmaps.png``.

3. **Late-layer nonlinearity (L24-27)** — reuse the canonical per-layer ridge skill +
   KRR gap (``krr_vs_linear.json``), add an output-proximity characterization (does v0
   align increasingly with the unembedding / does the nonlinear gap grow monotonically
   toward the output?). Output: ``late_layer.json`` + ``late_layer_gap.png``.

4. **Is c_C special? (control)** — compare c_C's predictive skill to a random linear
   projection of matched dimensionality, the mean-over-prompt c_C recipe, and the other
   real summary vectors. CHEAP proxy only; the strongest random-token-position control
   needs raw per-position activations NOT in the summary stores (FLAGGED, deferred).
   Output: ``cC_control.json``.

5. **Behavioral-chain preservation** — per behavior, Spearman ρ of predicted-vs-actual
   judged rate E0 for (a) the DIRECT chain ``r_Bᵀ v0 → E0`` vs (b) the LINEAR-MAP-
   MEDIATED chain ``r_Bᵀ (M̂ c_C) → E0`` (LOCO held-out M̂). Does routing through the
   linear map DEGRADE downstream behavior prediction? Output: ``behavior_chain.json``
   + ``behavior_chain.png``.

Smoke mode (``--smoke``) runs a reduced layer set / grid for a fast wiring check; full
mode runs the production read (n=50, 28 layers, top-48 PC target). CPU minutes.

Run:
    uv run python scripts/issue722_structural_battery.py --smoke
    uv run python scripts/issue722_structural_battery.py            # full, all 5 parts
    uv run python scripts/issue722_structural_battery.py --parts 1,5 # a subset
"""

from __future__ import annotations

import argparse
import json
import logging

# The inner math here is tiny (49×49 eigh / dual ridge at n=50), so multi-thread
# BLAS gives no speedup and, under the shared VM's heavy contention (load avg
# 100+), THRASHES — a sub-second PCA-48 ridge LOCO inflated to 30s on 16 threads
# vs 0.83s single-threaded (and KRR nested-CV, ~2100 small solves/layer, blew up
# to >14 min in Part 2). Pin BOTH the OpenBLAS/OMP env (must precede numpy import)
# AND torch to 1 thread (overridable via EPM_NUM_THREADS).
import os as _os

_NTHREADS = _os.environ.get("EPM_NUM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, _NTHREADS)

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_NTHREADS))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from explore_persona_space.analysis.convexity_meta import (  # noqa: E402
    reproducibility_metadata,
)

# Read-only reuse of the canonical LOCO ridge / KRR / skill / PCA machinery. NO
# re-implementation of the inner math — these are the exactness oracles.
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    krr_predict_loco,
    loco_train_means,
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue722_structural_battery")

HF_REPO = "superkaiba1/explore-persona-space-data"

# Betley (canonical) substrate paths.
V0_FILE = "issue658_theory_assumptions/store/v0_summaries.pt"
RB_FILE = "issue658_theory_assumptions/store/r_b.pt"
CC_LAST_FILE = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
E0_FILE = "issue658_partial/att-20260624-130414/eval_results_issue_658/E0_expression.json"

# UltraChat genre substrate paths (genre store self-contains cc_last + cc_meanprompt).
V0_FILE_UC = "issue658_theory_assumptions/store_genre-generalization-ultrachat/v0_summaries.pt"
RB_FILE_UC = "issue658_theory_assumptions/store_genre-generalization-ultrachat/r_b.pt"
E0_FILE_UC = "issue658_partial/att-20260624-130414/eval_results_issue_658/E0_expression_g1.json"

CANONICAL_KRR = (
    PROJECT_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/krr_vs_linear.json"
)
CANONICAL_SKILL = (
    PROJECT_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json"
)

OUT_DIR = PROJECT_ROOT / "eval_results/issue_722/structural"
FIG_DIR = PROJECT_ROOT / "figures/issue_722/structural"

PCA_TARGET_DIM = 48  # the #722 top-48 PC target reduction
PLATEAU_LAYERS = list(range(12, 22))  # L12-L21 plateau (peak L18)
LATE_LAYERS = list(range(24, 28))  # L24-L27 nonlinear region
SEED = 42

BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]


# ── data loading ──────────────────────────────────────────────────────────────


def _dl(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, path, repo_type="dataset")


def _load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_betley_substrate() -> dict:
    """Load the betley substrate aligned on a single canonical context order.

    Returns:
      ctx_ids: list[str] (the v0 store's order, n=50)
      layers: list[int] (0..27)
      V: dict[summary] -> (N, L, H)  (v0 summaries: mean/last/maxp)
      C_last: (N, L, H)   last-input-token c_C (from #594 store)
      C_meanprompt: (N, L, H)   mean-over-prompt c_C (from v0 store)
      r_b: dict[behavior] -> (L, H)  diffmeans read-out directions
      e0: the raw E0 json
      store_provenance
    """
    v0 = torch.load(_dl(V0_FILE), weights_only=False)
    cc = torch.load(_dl(CC_LAST_FILE), weights_only=False)
    rb = torch.load(_dl(RB_FILE), weights_only=False)
    e0 = _load_json(_dl(E0_FILE))

    h_v0 = v0.get("probe_pool_hash")
    h_cc = cc.get("probe_pool_hash")
    if h_v0 is None or h_cc is None or h_v0 != h_cc:
        raise RuntimeError(f"probe_pool_hash mismatch betley v0/c_C: v0={h_v0!r} c_C={h_cc!r}")

    ctx_ids = list(v0["context_ids"])
    layers = list(v0["capture_layers"])

    V = {
        s: np.stack([v0["summaries"][s][c].numpy() for c in ctx_ids]).astype(np.float64)
        for s in ("mean", "last", "maxp")
    }

    iid_to_row = {iid: i for i, iid in enumerate(cc["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    if missing:
        raise RuntimeError(f"#594 cc_last missing {len(missing)} ctx: {missing[:5]}")
    cc_t = cc["tensor"]
    C_last = np.stack([cc_t[iid_to_row[c]].numpy() for c in ctx_ids]).astype(np.float64)
    C_meanprompt = np.stack([v0["cc_meanprompt"][c].numpy() for c in ctx_ids]).astype(np.float64)

    r_b = {
        b: np.stack([rb["r_b"][b]["diffmeans"][li].numpy() for li in range(len(layers))])
        for b in BEHAVIORS
        if b in rb.get("r_b", {})
    }

    return {
        "genre": "betley",
        "ctx_ids": ctx_ids,
        "layers": layers,
        "V": V,
        "C_last": C_last,
        "C_meanprompt": C_meanprompt,
        "r_b": r_b,
        "e0": e0,
        "store_provenance": {
            "v0_file": f"{HF_REPO}:{V0_FILE}",
            "cc_last_file": f"{HF_REPO}:{CC_LAST_FILE}",
            "rb_file": f"{HF_REPO}:{RB_FILE}",
            "e0_file": f"{HF_REPO}:{E0_FILE}",
            "n_contexts": len(ctx_ids),
            "hidden_dim": int(V["mean"].shape[-1]),
            "probe_pool_hash": h_v0,
        },
    }


def load_ultrachat_substrate(ref_ctx_order: list[str]) -> dict:
    """Load the ultrachat-genre substrate REORDERED onto ``ref_ctx_order``.

    The genre store self-contains ``cc_last`` + ``cc_meanprompt`` (its own probe
    battery differs from betley — different probe_pool_hash — so the #594 betley
    c_C does NOT apply here). The 50 contexts are the SAME personas as betley, just
    in a different store order; we reindex onto the betley order so cross-genre
    comparisons are aligned by context.
    """
    v0 = torch.load(_dl(V0_FILE_UC), weights_only=False)
    rb = torch.load(_dl(RB_FILE_UC), weights_only=False)
    e0 = _load_json(_dl(E0_FILE_UC))

    store_ctx = list(v0["context_ids"])
    if set(store_ctx) != set(ref_ctx_order):
        raise RuntimeError("ultrachat genre context set differs from betley reference set")
    layers = list(v0["capture_layers"])

    V = {
        s: np.stack([v0["summaries"][s][c].numpy() for c in ref_ctx_order]).astype(np.float64)
        for s in ("mean", "last", "maxp")
    }
    C_last = np.stack([v0["cc_last"][c].numpy() for c in ref_ctx_order]).astype(np.float64)
    C_meanprompt = np.stack([v0["cc_meanprompt"][c].numpy() for c in ref_ctx_order]).astype(
        np.float64
    )
    r_b = {
        b: np.stack([rb["r_b"][b]["diffmeans"][li].numpy() for li in range(len(layers))])
        for b in BEHAVIORS
        if b in rb.get("r_b", {})
    }
    return {
        "genre": "ultrachat",
        "ctx_ids": list(ref_ctx_order),
        "layers": layers,
        "V": V,
        "C_last": C_last,
        "C_meanprompt": C_meanprompt,
        "r_b": r_b,
        "e0": e0,
        "store_provenance": {
            "v0_file": f"{HF_REPO}:{V0_FILE_UC}",
            "rb_file": f"{HF_REPO}:{RB_FILE_UC}",
            "e0_file": f"{HF_REPO}:{E0_FILE_UC}",
            "n_contexts": len(ref_ctx_order),
            "probe_pool_hash": v0.get("probe_pool_hash"),
        },
    }


def e0_vector(e0: dict, column_id: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[int]]:
    """Per-context E0 rate for one behavior; returns (values, kept_row_indices).

    Mirrors ``issue658_fit_predictors.e0_target`` (rate, fall back to logp_mean),
    but returns ROW INDICES into ``ctx_ids`` so callers can subselect predictions.
    """
    vals: list[float] = []
    kept_idx: list[int] = []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(column_id)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            v = cell.get("logp_mean")
        if v is None:
            continue
        vals.append(float(v))
        kept_idx.append(i)
    return np.array(vals, dtype=np.float64), kept_idx


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    r, _ = spearmanr(a, b)
    return None if np.isnan(r) else float(r)


# ── shared PCA-target ridge skill (the #722 metric) ────────────────────────────


def pca_target_ridge_pred(Xc: np.ndarray, Y: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """LOCO ridge prediction of the top-k PCA-reduced target Y, back-projected to H.

    Reduces Y -> PCA-k coords, runs the centered LOCO ridge predictor in PCA space,
    back-projects to full H. Returns (pred_full (N, H), skill dict over full-H Y).
    The PCA reduction matches the #722 ``top-48 PC target`` reduction exactly via
    ``robust_pca_basis`` (the shared helper) — but skill is reported over the FULL-H
    centered target so it is comparable to the canonical ``skill_over_mean.json``.
    """
    mu, comps, _fallback = robust_pca_basis(Y, k)  # comps (k', H)
    Y_pca = (Y - mu) @ comps.T  # (N, k')
    pred_pca = ridge_predict_loco_centered(Xc, Y_pca)  # (N, k')
    pred_full = pred_pca @ comps + mu  # back to (N, H)
    sk = skill_over_mean_r2(pred_full, Y)
    return pred_full, sk


def _ridge_skill_fullH(Xc: np.ndarray, Y: np.ndarray) -> dict:
    """Full-H LOCO ridge skill-over-mean (the canonical #722 ridge arm)."""
    pred = ridge_predict_loco_centered(Xc, Y)
    return skill_over_mean_r2(pred, Y)


def _ridge_skill_pca(Xc: np.ndarray, Y: np.ndarray, k: int) -> float:
    """PCA-k-target LOCO ridge skill-over-mean (the fast comparable metric).

    Reduces Y to its top-k PCA coords, runs the centered LOCO ridge in that space,
    reports skill-over-mean on the PCA-reduced target. ~0.83s vs ~47s for full-H at
    n=50/H=3584 under contention; matches krr_vs_linear.json's skill space.
    """
    mu, comps, _fb = robust_pca_basis(Y, k)
    Y_pca = (Y - mu) @ comps.T
    pred = ridge_predict_loco_centered(Xc, Y_pca)
    return skill_over_mean_r2(pred, Y_pca)["skill"]


# ── PART 1: rank / structure of M̂ ─────────────────────────────────────────────


def fit_global_ridge_map(Xc: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Global (non-LOCO) ridge map M̂ (H_in, H_out) on standardized X, centered Y.

    M̂ = (XnᵀXn + λI_d)⁻¹ Xnᵀ Yc, computed in the DUAL (Woodbury) form
    ``M̂ = Xnᵀ (Xn Xnᵀ + λI_m)⁻¹ Yc`` — an m×m (n=50) solve, IDENTICAL to the primal
    but O(m²d) not O(d³) (the d=3584 primal solve was ~tens of seconds/layer). This
    is the WHOLE-data map whose SVD structure we read; returned in the standardized
    input space (the natural space for the map's intrinsic rank).
    """
    xmu = Xc.mean(0)
    xsd = Xc.std(0, ddof=0) + 1e-9
    Xn = (Xc - xmu) / xsd
    ymu = Y.mean(0)
    Yc = Y - ymu
    m = Xn.shape[0]
    alpha = np.linalg.solve(Xn @ Xn.T + lam * np.eye(m), Yc)  # (m, H_out)
    return Xn.T @ alpha  # (H_in, H_out)


def participation_ratio(sv: np.ndarray) -> float:
    """Participation ratio of the singular-value spectrum: (Σ s²)² / Σ s⁴.

    A scale-free effective-rank measure: 1 ⇒ rank-1, n ⇒ flat spectrum. Computed
    on squared singular values (the eigen-energy of M̂^T M̂).
    """
    e = sv**2
    s1 = float(np.sum(e))
    s2 = float(np.sum(e**2))
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def energy_rank(sv: np.ndarray, frac: float) -> int:
    """#singular values needed to reach ``frac`` of total squared-singular energy."""
    e = sv**2
    cum = np.cumsum(e) / np.sum(e)
    return int(np.searchsorted(cum, frac) + 1)


def part1_rank_structure(sub: dict, layers: list[int], k_pca: int, n_top: int = 5) -> dict:
    """Rank / structure of M̂ per layer + alignment of its directions with r_B.

    For each layer ℓ: fit the global PCA-target ridge map (c_C@ℓ -> top-k PCA of
    v0@ℓ), SVD it, report the spectrum + effective rank. Then project M̂'s top
    output (left-singular, in v0-PCA space back-projected to H) directions onto each
    behavior's r_B@ℓ and report the |cos| alignment — do the map's principal output
    directions align with the behavior axes?
    """
    C = sub["C_last"]
    Vmean = sub["V"]["mean"]
    layer_idx = {li: i for i, li in enumerate(sub["layers"])}
    out_layers = []
    # ridge λ chosen by the canonical run is 0.01 at the plateau; we read the same
    # grid's per-layer choice for the SVD map (a fixed λ keeps the map comparable).
    LAM = 0.01

    for li in layers:
        i = layer_idx[li]
        Xc = C[:, i, :]  # (N, H)
        Y = Vmean[:, i, :]  # (N, H)
        # reduce the target to top-k PCA so M̂ is (H_in, k) — a compact map whose
        # SVD rank is bounded by min(N-1, k); reading the rank in this space is the
        # honest "how many directions does the map use" read at n=50.
        mu, comps, _fb = robust_pca_basis(Y, k_pca)  # (k', H)
        Y_pca = (Y - mu) @ comps.T  # (N, k')
        M = fit_global_ridge_map(Xc, Y_pca, LAM)  # (H_in, k')
        # SVD of the compact map
        _U, sv, Vt = np.linalg.svd(M, full_matrices=False)  # left U unused; Vt (r, k')
        sv = np.asarray(sv, dtype=np.float64)
        pr = participation_ratio(sv)
        r90 = energy_rank(sv, 0.90)
        r95 = energy_rank(sv, 0.95)

        # output singular directions in full-H v0 space: each right-singular row
        # v_j (in PCA-k coords) back-projects to H via comps. Take the top n_top.
        ntop = min(n_top, Vt.shape[0])
        out_dirs_H = Vt[:ntop] @ comps  # (ntop, H)  output directions in v0 space
        out_dirs_H = out_dirs_H / (np.linalg.norm(out_dirs_H, axis=1, keepdims=True) + 1e-12)

        rb_align = {}
        for b, rb_layers in sub["r_b"].items():
            r = rb_layers[i]  # (H,)
            r = r / (np.linalg.norm(r) + 1e-12)
            cos = np.abs(out_dirs_H @ r)  # (ntop,)
            rb_align[b] = {
                "max_abs_cos_top5": float(np.max(cos)),
                "abs_cos_per_singular": [float(x) for x in cos],
            }
        # baseline: |cos| of r_B with a random unit H-vector (matched-dim chance).
        rng = np.random.default_rng(SEED + li)
        H = Xc.shape[1]
        rand = rng.standard_normal((ntop, H))
        rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12
        rand_align = {}
        for b, rb_layers in sub["r_b"].items():
            r = rb_layers[i]
            r = r / (np.linalg.norm(r) + 1e-12)
            rand_align[b] = float(np.max(np.abs(rand @ r)))

        out_layers.append(
            {
                "layer": li,
                "ridge_lambda": LAM,
                "n_singular": int(sv.size),
                "singular_values": [float(x) for x in sv],
                "participation_ratio": pr,
                "energy_rank_90": r90,
                "energy_rank_95": r95,
                "rb_alignment_top5": rb_align,
                "rb_alignment_random_baseline_maxcos": rand_align,
            }
        )
    return {
        "analysis": "rank_structure_of_M_hat",
        "description": (
            "Global ridge map M̂ (c_C@ℓ -> top-k PCA of v0@ℓ) per layer; SVD spectrum, "
            "effective rank (participation ratio + 90/95% energy), and |cos| alignment of "
            "M̂'s top-5 output singular directions with the behavior r_B directions. "
            "A low participation ratio relative to k ⇒ the map is low-rank."
        ),
        "k_pca_target": k_pca,
        "n_top_singular_reported": n_top,
        "genre": sub["genre"],
        "layers": layers,
        "per_layer": out_layers,
        "store_provenance": sub["store_provenance"],
    }


# ── PART 2: robustness sweep ───────────────────────────────────────────────────


def _layer_skill(
    sub_X: dict,
    sub_Y: dict,
    cc_key: str,
    v0_key: str,
    li_x: int,
    li_y: int,
    k_pca: int,
    do_krr: bool,
    full_h: bool = False,
) -> dict:
    """Skill of c_C(sub_X)@li_x -> v0(sub_Y)@li_y.

    Returns the PCA-48-target ridge skill (``skill_ridge_pca``, the fast metric
    the gap analysis + krr_vs_linear.json use — 0.83s/call vs 47s for full-H under
    contention), and OPTIONALLY (``full_h=True``) the full-H ridge skill
    (``skill_ridge_fullH``, the parity anchor to the canonical skill_over_mean.json
    0.80 headline). The headline robustness reads keep full_h at the plateau; the
    large cross-layer / recipe sweeps use the comparable PCA-48 skill only.

    Cross-genre: sub_X and sub_Y carry the X (c_C) and Y (v0) matrices from one
    aligned substrate; a genuine fit-on-X-score-on-Y transfer is _cross_genre_skill.
    """
    lx = {li: i for i, li in enumerate(sub_X["layers"])}[li_x]
    ly = {li: i for i, li in enumerate(sub_Y["layers"])}[li_y]
    Xc = sub_X[cc_key][:, lx, :]
    Y = sub_Y["V"][v0_key][:, ly, :]
    # PCA-48-target ridge skill (fast; the comparable metric).
    mu, comps, _fb = robust_pca_basis(Y, k_pca)
    Y_pca = (Y - mu) @ comps.T
    pred_lin = ridge_predict_loco_centered(Xc, Y_pca)
    out = {"skill_ridge_pca": skill_over_mean_r2(pred_lin, Y_pca)["skill"]}
    if full_h:
        sk_full = _ridge_skill_fullH(Xc, Y)
        out["skill_ridge_fullH"] = sk_full["skill"]
        out["ridge_median_per_dim_r2"] = sk_full["median_per_dim_r2"]
    if do_krr:
        pred_rbf, _lam, _gam = krr_predict_loco(Xc, Y_pca, kernel="rbf")
        sk_rbf = skill_over_mean_r2(pred_rbf, Y_pca)["skill"]
        out["skill_krr_linear_pca"] = out["skill_ridge_pca"]
        out["skill_krr_rbf_pca"] = sk_rbf
        out["nonlinear_gap_rbf_minus_linear"] = sk_rbf - out["skill_ridge_pca"]
    return out


def _cross_genre_skill(
    fit_sub: dict, score_sub: dict, cc_key: str, v0_key: str, li: int, k_pca: int
) -> dict:
    """Fit ridge M̂ on fit_sub's (c_C, v0)@li, SCORE skill on score_sub's held-out rows.

    Genuine cross-genre transfer: train the standardized-input / centered-target ridge
    on ALL of fit_sub, then predict score_sub's v0 from score_sub's c_C and measure
    skill-over-(score_sub's own train mean). Both substrates share the SAME 50-context
    order (reindexed at load), so the per-context comparison is aligned. We use the
    full-data fit-genre map (not LOCO) because the test genre is held out wholesale —
    the leakage LOCO guards against (same-context train/test) cannot occur across genres.
    """
    lf = {li2: i for i, li2 in enumerate(fit_sub["layers"])}[li]
    ls = {li2: i for i, li2 in enumerate(score_sub["layers"])}[li]
    Xf = fit_sub[cc_key][:, lf, :]
    Yf = fit_sub["V"][v0_key][:, lf, :]
    Xs = score_sub[cc_key][:, ls, :]
    Ys = score_sub["V"][v0_key][:, ls, :]

    # standardize on the FIT genre; center target on the FIT genre.
    xmu, xsd = Xf.mean(0), Xf.std(0, ddof=0) + 1e-9
    ymu = Yf.mean(0)
    Xfn = (Xf - xmu) / xsd
    Yfc = Yf - ymu
    # ridge λ via the canonical grid's plateau choice (0.01); fixed for transfer.
    LAM = 0.01
    # DUAL (Woodbury) ridge so we never form/solve the H×H (3584³) system:
    # pred_centered = Xsn · Xfnᵀ (Xfn Xfnᵀ + λI_m)⁻¹ Yfc  — an m×m (n=50) solve.
    Xsn = (Xs - xmu) / xsd
    m = Xfn.shape[0]
    Gff = Xfn @ Xfn.T + LAM * np.eye(m)  # (m, m)
    alpha = np.linalg.solve(Gff, Yfc)  # (m, H)
    Ksf = Xsn @ Xfn.T  # (m_score, m_fit) cross-genre kernel
    pred = Ksf @ alpha + ymu  # predict score-genre v0 (n, H)
    # skill over the SCORE genre's own across-context mean (LOCO train mean ≈ mean).
    tmean = loco_train_means(Ys)
    ss_res = float(np.sum((Ys - pred) ** 2))
    ss_tot = float(np.sum((Ys - tmean) ** 2))
    skill = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "transfer_skill": skill,
        "fit_genre": fit_sub["genre"],
        "score_genre": score_sub["genre"],
    }


def part2_robustness(
    betley: dict,
    uc: dict | None,
    plateau: list[int],
    k_pca: int,
    cross_layer_grid: list[int],
    do_krr: bool,
) -> dict:
    """Robustness sweep across genre / layer / c_C recipe / v0 summary variations.

    The bulk sweeps (recipe / v0-summary / cross-layer grid / cross-genre within-
    refs) report the fast PCA-48-target ridge skill (``skill_ridge_pca``, the
    comparable metric the gap analysis uses). The plateau canonical read ALSO
    carries the full-H ridge skill (``skill_ridge_fullH``) as the parity anchor to
    the canonical skill_over_mean.json 0.80 headline.
    """
    result: dict[str, Any] = {
        "analysis": "robustness_sweep",
        "canonical": "c_C=last, v0=mean, genre=betley, within-layer",
        "k_pca_target": k_pca,
        "plateau_layers": plateau,
        "skill_metric_note": (
            "skill_ridge_pca = PCA-48-target ridge skill-over-mean (the fast comparable "
            "metric, matches krr_vs_linear.json); skill_ridge_fullH = full-H ridge skill "
            "(parity anchor to skill_over_mean.json's 0.80 headline, plateau only)."
        ),
    }

    # (a) canonical within-genre/within-layer at the plateau (reference + full-H anchor).
    result["canonical_within_layer"] = {
        li: _layer_skill(betley, betley, "C_last", "mean", li, li, k_pca, do_krr, full_h=True)
        for li in plateau
    }

    # (b) c_C recipe: last vs meanprompt (betley, v0=mean) — PCA-48 skill.
    result["cC_recipe"] = {
        "last": {
            li: _layer_skill(betley, betley, "C_last", "mean", li, li, k_pca, False)[
                "skill_ridge_pca"
            ]
            for li in plateau
        },
        "meanprompt": {
            li: _layer_skill(betley, betley, "C_meanprompt", "mean", li, li, k_pca, False)[
                "skill_ridge_pca"
            ]
            for li in plateau
        },
    }

    # (c) v0 summary: mean vs last vs maxp (betley, c_C=last) — PCA-48 skill.
    result["v0_summary"] = {
        s: {
            li: _layer_skill(betley, betley, "C_last", s, li, li, k_pca, False)["skill_ridge_pca"]
            for li in plateau
        }
        for s in ("mean", "last", "maxp")
    }

    # (d) cross-layer grid: c_C@ℓ -> v0@ℓ' (betley, c_C=last, v0=mean) — PCA-48 skill.
    grid = []
    for lx in cross_layer_grid:
        for ly in cross_layer_grid:
            sk = _layer_skill(betley, betley, "C_last", "mean", lx, ly, k_pca, False)
            grid.append({"layer_cC": lx, "layer_v0": ly, "skill_ridge_pca": sk["skill_ridge_pca"]})
    result["cross_layer_grid"] = grid

    # (e) cross-genre transfer + each within-genre reference (c_C=last, v0=mean) — PCA-48 skill.
    if uc is not None:
        cg = {}
        for li in plateau:
            cg[li] = {
                "within_betley": _layer_skill(
                    betley, betley, "C_last", "mean", li, li, k_pca, False
                )["skill_ridge_pca"],
                "within_ultrachat": _layer_skill(uc, uc, "C_last", "mean", li, li, k_pca, False)[
                    "skill_ridge_pca"
                ],
                "fit_betley_score_ultrachat": _cross_genre_skill(
                    betley, uc, "C_last", "mean", li, k_pca
                )["transfer_skill"],
                "fit_ultrachat_score_betley": _cross_genre_skill(
                    uc, betley, "C_last", "mean", li, k_pca
                )["transfer_skill"],
            }
        result["cross_genre"] = cg
    else:
        result["cross_genre"] = "SKIPPED — ultrachat substrate unavailable"

    result["store_provenance"] = {
        "betley": betley["store_provenance"],
        "ultrachat": uc["store_provenance"] if uc is not None else None,
    }
    return result


# ── PART 3: late-layer nonlinearity ────────────────────────────────────────────


def part3_late_layer(betley: dict, k_pca: int) -> dict:
    """Reuse the canonical per-layer KRR gap + characterize the late-layer nonlinearity.

    Loads ``krr_vs_linear.json`` (all-28-layer ridge skill + KRR-RBF gap with bootstrap
    CI) and adds an output-proximity read: per layer, the |cos| alignment of the v0
    activations with the unembedding row-space proxy + whether the nonlinear gap grows
    monotonically toward the output. If the canonical KRR file is missing, recompute the
    gap at the late layers only.
    """
    out: dict[str, Any] = {
        "analysis": "late_layer_nonlinearity",
        "late_layers": LATE_LAYERS,
        "k_pca_target": k_pca,
    }

    krr_rows = None
    if CANONICAL_KRR.exists():
        kk = _load_json(CANONICAL_KRR)
        krr_rows = {r["layer"]: r for r in kk["per_layer"]}
        out["krr_source"] = "reused canonical krr_vs_linear.json (all 28 layers)"
    else:
        out["krr_source"] = "recomputed at late layers (canonical file absent)"

    layer_idx = {li: i for i, li in enumerate(betley["layers"])}
    per_layer = []
    for li in betley["layers"]:
        row: dict[str, Any] = {"layer": li}
        if krr_rows is not None and li in krr_rows:
            r = krr_rows[li]
            row["skill_ridge_lin"] = r.get(
                "skill_krr_linear_pca48", r.get("skill_vs_mean_ridge_fullH")
            )
            row["skill_krr_rbf"] = r.get("skill_krr_rbf_pca48")
            row["nonlinear_gap"] = r.get("nonlinear_gap_rbf_minus_linear")
            row["gap_ci95"] = r.get("gap_ci95")
            row["gap_excludes_zero"] = r.get("gap_excludes_zero")
        elif li in LATE_LAYERS:
            i = layer_idx[li]
            Xc = betley["C_last"][:, i, :]
            Y = betley["V"]["mean"][:, i, :]
            mu, comps, _fb = robust_pca_basis(Y, k_pca)
            Y_pca = (Y - mu) @ comps.T
            pred_lin = ridge_predict_loco_centered(Xc, Y_pca)
            pred_rbf, _l, _g = krr_predict_loco(Xc, Y_pca, kernel="rbf")
            sk_lin = skill_over_mean_r2(pred_lin, Y_pca)["skill"]
            sk_rbf = skill_over_mean_r2(pred_rbf, Y_pca)["skill"]
            row["skill_ridge_lin"] = sk_lin
            row["skill_krr_rbf"] = sk_rbf
            row["nonlinear_gap"] = sk_rbf - sk_lin
        per_layer.append(row)
    out["per_layer"] = per_layer

    # output-proximity characterization: v0's alignment with the unembedding subspace.
    # We do not load the lm_head weights here (no GPU / model load); instead we use a
    # model-internal proxy: how much of v0's variance the LAST layer (L27, closest to
    # the unembedding) explains of each layer's v0 — i.e. cross-layer cosine of the
    # per-layer mean-v0 direction with the final-layer mean-v0 direction. Rising
    # alignment toward the output + the rising nonlinear gap is the "approaching the
    # output manifold" signature. (A true unembedding read is FLAGGED below.)
    Vmean = betley["V"]["mean"]
    L = len(betley["layers"])
    v_final = Vmean[:, L - 1, :]
    v_final_dir = v_final.mean(0)
    v_final_dir = v_final_dir / (np.linalg.norm(v_final_dir) + 1e-12)
    # per-layer effective dimensionality of v0 (participation ratio of its PCA spectrum)
    proximity = []
    for li in betley["layers"]:
        i = layer_idx[li]
        Vl = Vmean[:, i, :]
        d = Vl.mean(0)
        d = d / (np.linalg.norm(d) + 1e-12)
        cos_to_final = float(abs(d @ v_final_dir))
        # PCA participation ratio of the centered per-layer v0 (intrinsic dim)
        Vc = Vl - Vl.mean(0)
        sv = np.linalg.svd(Vc, compute_uv=False)
        proximity.append(
            {
                "layer": li,
                "cos_meanv0_to_final_layer": cos_to_final,
                "v0_participation_ratio": participation_ratio(sv),
            }
        )
    out["output_proximity"] = proximity

    # monotonicity of the gap across the late block (L22..L27)
    late_gaps = [
        (r["layer"], r.get("nonlinear_gap"))
        for r in per_layer
        if r["layer"] >= 22 and r.get("nonlinear_gap") is not None
    ]
    late_gaps.sort()
    gaps_seq = [g for _li, g in late_gaps]
    mono = (
        all(gaps_seq[i] <= gaps_seq[i + 1] + 1e-6 for i in range(len(gaps_seq) - 1))
        if len(gaps_seq) > 1
        else None
    )
    out["late_block_gap_monotonic_increasing_L22toL27"] = mono
    out["late_block_gaps"] = {li: g for li, g in late_gaps}
    out["caveat_output_proximity"] = (
        "Output-proximity uses a model-internal proxy (cosine of per-layer mean-v0 to "
        "the final-layer mean-v0 + per-layer v0 intrinsic dim) because the lm_head "
        "unembedding read needs a GPU model load (deferred). The rising nonlinear gap "
        "toward the output (L24-27) co-occurring with rising final-layer alignment is "
        "the available evidence for output-proximity-driven nonlinearity."
    )
    out["store_provenance"] = betley["store_provenance"]
    return out


# ── PART 4: is c_C special? (control) ──────────────────────────────────────────


def part4_cC_control(betley: dict, layers: list[int], k_pca: int, n_random: int = 5) -> dict:
    """Compare c_C's predictive skill to matched-dim random projections + other recipes.

    For each layer: (a) c_C=last skill; (b) c_C=meanprompt skill; (c) n_random
    random-linear-projection baselines of c_C=last (random orthonormal mix of the
    last-token c_C's own dimensions, matched dim) — tests whether the SPECIFIC
    newline-before-answer slot carries more than a generic linear scramble of the
    same activation; (d) v0=last / v0=maxp used AS the predictor (other real summary
    vectors of the same contexts) -> v0=mean target, a "does any context-derived
    vector predict v0" sanity. The strongest control (random TOKEN position /
    question embedding) needs raw per-position acts NOT in the summary stores — FLAGGED.
    """
    layer_idx = {li: i for i, li in enumerate(betley["layers"])}
    rng = np.random.default_rng(SEED)
    per_layer = []
    for li in layers:
        i = layer_idx[li]
        Y = betley["V"]["mean"][:, i, :]  # target: v0 mean
        Xlast = betley["C_last"][:, i, :]
        Xmean = betley["C_meanprompt"][:, i, :]

        # All reads in the fast PCA-48-target skill space so they are directly
        # comparable (and the full run finishes in CPU minutes, not hours).
        sk_last = _ridge_skill_pca(Xlast, Y, k_pca)
        sk_meanp = _ridge_skill_pca(Xmean, Y, k_pca)

        # random linear projection of c_C=last: a structure-destroying matched-rank
        # scramble. A random orthonormal H×H rotation would leave ridge (rotation-
        # equivariant on a standardized design) ~invariant, so the informative
        # control is a random GAUSSIAN projection to k_pca dims and back — it keeps
        # the dimensionality but destroys the specific newline-slot fine structure.
        rand_skills = []
        H = Xlast.shape[1]
        for _ in range(n_random):
            P = rng.standard_normal((H, k_pca))
            P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-12
            Xr = (Xlast @ P) @ P.T  # rank-k_pca random projection of c_C
            rand_skills.append(_ridge_skill_pca(Xr, Y, k_pca))
        # other real summary vectors as predictors (v0 last/maxp -> v0 mean)
        sk_v0last = _ridge_skill_pca(betley["V"]["last"][:, i, :], Y, k_pca)
        sk_v0maxp = _ridge_skill_pca(betley["V"]["maxp"][:, i, :], Y, k_pca)

        per_layer.append(
            {
                "layer": li,
                "skill_metric": "PCA-48-target ridge skill-over-mean",
                "skill_cC_last": sk_last,
                "skill_cC_meanprompt": sk_meanp,
                # A degenerate random rank-48 projection occasionally yields a wildly
                # negative skill (≪ predict-the-mean), so report the MEDIAN as the
                # robust summary (the mean is dominated by such outliers); keep
                # max + the raw samples for the full distribution.
                "skill_random_proj_matched_rank_median": float(np.median(rand_skills)),
                "skill_random_proj_matched_rank_mean": float(np.mean(rand_skills)),
                "skill_random_proj_matched_rank_max": float(np.max(rand_skills)),
                "skill_random_proj_samples": [float(x) for x in rand_skills],
                "skill_v0last_as_predictor": sk_v0last,
                "skill_v0maxp_as_predictor": sk_v0maxp,
            }
        )
    return {
        "analysis": "is_cC_special_control",
        "layers": layers,
        "k_pca_target": k_pca,
        "n_random_projections": n_random,
        "per_layer": per_layer,
        "deferred_strongest_control": (
            "A random-TOKEN-position baseline and a question-only embedding baseline are "
            "the strongest 'is the newline-before-answer slot special?' controls, but they "
            "require raw per-position residual activations (NOT in the summary stores, which "
            "hold only the last-token c_C + the mean/last/maxp answer-span pools). They are "
            "DEFERRED to a GPU re-extraction. The cheap proxy here (matched-rank random "
            "projection of c_C, meanprompt recipe, other real summaries) is reported instead."
        ),
        "store_provenance": betley["store_provenance"],
    }


# ── PART 5: behavioral-chain preservation ──────────────────────────────────────


def part5_behavior_chain(betley: dict, layers: list[int]) -> dict:
    """Direct vs linear-map-mediated chain ρ(predicted E0, actual E0) per behavior.

    For each behavior B and layer ℓ:
      DIRECT chain:   pred_direct = r_B@ℓ ᵀ v0@ℓ        (true v0; the #658 A3.3 read)
      MEDIATED chain: pred_med    = r_B@ℓ ᵀ M̂(c_C@ℓ)   (LOCO held-out predicted v0)
    Then ρ_direct = Spearman(pred_direct[kept], E0_B), ρ_med = Spearman(pred_med[kept],
    E0_B). The headline: does routing through the linear map DEGRADE ρ vs the true v0?
    Best-layer (largest ρ, matching #658 _chain_rho) reported per behavior + the full
    per-layer table.
    """
    layer_idx = {li: i for i, li in enumerate(betley["layers"])}
    # precompute the held-out predicted v0 per layer ONCE. Predict in the top-48 PCA
    # space (the fast metric) then BACK-PROJECT to full H so the full-H r_B dot is
    # exact: pred_fullH = pred_pca @ comps + mu (the #658 chain_rho_pca pattern).
    # The PCA reduction is lossless at n≪d for the directions r_B reads through M̂.
    pred_v0_by_layer = {}
    for li in layers:
        i = layer_idx[li]
        Xc = betley["C_last"][:, i, :]
        Y = betley["V"]["mean"][:, i, :]
        mu, comps, _fb = robust_pca_basis(Y, PCA_TARGET_DIM)
        Y_pca = (Y - mu) @ comps.T
        pred_pca = ridge_predict_loco_centered(Xc, Y_pca)  # (N, k)
        pred_v0_by_layer[li] = pred_pca @ comps + mu  # (N, H) back-projected

    per_behavior = {}
    for b in BEHAVIORS:
        if b not in betley["r_b"]:
            per_behavior[b] = {"status": "no r_B for this behavior"}
            continue
        y_e0, kept_idx = e0_vector(betley["e0"], b, betley["ctx_ids"])
        if len(kept_idx) < 4:
            per_behavior[b] = {"status": f"too few E0 ({len(kept_idx)})"}
            continue
        rows = []
        best_direct = None
        best_med = None
        for li in layers:
            i = layer_idx[li]
            r = betley["r_b"][b][i]  # (H,)
            v0_true = betley["V"]["mean"][:, i, :]  # (N, H)
            v0_pred = pred_v0_by_layer[li]  # (N, H)
            pred_direct = (v0_true @ r)[kept_idx]
            pred_med = (v0_pred @ r)[kept_idx]
            rho_d = _spearman(pred_direct, y_e0)
            rho_m = _spearman(pred_med, y_e0)
            rows.append(
                {
                    "layer": li,
                    "rho_direct": rho_d,
                    "rho_mediated": rho_m,
                    "rho_degradation": (rho_d - rho_m)
                    if (rho_d is not None and rho_m is not None)
                    else None,
                }
            )
            if rho_d is not None and (best_direct is None or rho_d > best_direct["rho"]):
                best_direct = {"layer": li, "rho": rho_d}
            if rho_m is not None and (best_med is None or rho_m > best_med["rho"]):
                best_med = {"layer": li, "rho": rho_m}
        per_behavior[b] = {
            "n_kept_e0": len(kept_idx),
            "best_direct": best_direct,
            "best_mediated": best_med,
            "best_layer_degradation": (
                (best_direct["rho"] - best_med["rho"]) if (best_direct and best_med) else None
            ),
            "per_layer": rows,
        }
    return {
        "analysis": "behavioral_chain_preservation",
        "description": (
            "Spearman ρ of predicted-vs-actual judged E0 for the DIRECT chain "
            "(r_Bᵀ v0 -> E0) vs the LINEAR-MAP-MEDIATED chain (r_Bᵀ M̂ c_C -> E0, LOCO "
            "held-out M̂). Degradation = ρ_direct - ρ_mediated; near-zero ⇒ the linear "
            "map preserves the behavior-relevant direction."
        ),
        "layers": layers,
        "per_behavior": per_behavior,
        "store_provenance": betley["store_provenance"],
    }


# ── figures ────────────────────────────────────────────────────────────────────


def _fig_part1(res: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    rows = res["per_layer"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: singular spectra (one line per layer)
    for r in rows:
        sv = np.array(r["singular_values"])
        axes[0].plot(
            range(1, len(sv) + 1),
            sv / sv[0],
            marker=".",
            ms=3,
            lw=1,
            alpha=0.7,
            label=f"L{r['layer']}",
        )
    axes[0].set_xlabel("singular index")
    axes[0].set_ylabel("singular value (normalized to σ₁)")
    axes[0].set_title("M̂ singular spectrum (per layer)")
    axes[0].set_yscale("log")
    if len(rows) <= 12:
        axes[0].legend(fontsize=6, ncol=2)
    # right: effective rank per layer
    L = [r["layer"] for r in rows]
    pr = [r["participation_ratio"] for r in rows]
    r90 = [r["energy_rank_90"] for r in rows]
    r95 = [r["energy_rank_95"] for r in rows]
    axes[1].plot(L, pr, marker="o", label="participation ratio")
    axes[1].plot(L, r90, marker="s", label="#PC for 90% energy")
    axes[1].plot(L, r95, marker="^", label="#PC for 95% energy")
    axes[1].axhline(
        res["k_pca_target"], ls="--", color="gray", lw=1, label=f"k target ({res['k_pca_target']})"
    )
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("effective rank")
    axes[1].set_title("M̂ effective rank")
    axes[1].legend(fontsize=7)
    savefig_paper(fig, "rank_spectrum", dir=str(fig_dir))
    plt.close(fig)


def _fig_part2(res: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) cross-layer grid heatmap
    grid = res["cross_layer_grid"]
    lxs = sorted(set(g["layer_cC"] for g in grid))
    lys = sorted(set(g["layer_v0"] for g in grid))
    M = np.full((len(lys), len(lxs)), np.nan)
    for g in grid:
        M[lys.index(g["layer_v0"]), lxs.index(g["layer_cC"])] = g["skill_ridge_pca"]
    im = axes[0].imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=-0.2, vmax=0.85)
    axes[0].set_xticks(range(len(lxs)))
    axes[0].set_xticklabels(lxs)
    axes[0].set_yticks(range(len(lys)))
    axes[0].set_yticklabels(lys)
    axes[0].set_xlabel("c_C layer")
    axes[0].set_ylabel("v0 layer")
    axes[0].set_title("cross-layer ridge skill")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # (b) recipe variations at plateau (bar of mean skill per variation)
    bars = {}
    bars["c_C=last"] = np.nanmean(list(res["cC_recipe"]["last"].values()))
    bars["c_C=meanprompt"] = np.nanmean(list(res["cC_recipe"]["meanprompt"].values()))
    for s in ("mean", "last", "maxp"):
        bars[f"v0={s}"] = np.nanmean(list(res["v0_summary"][s].values()))
    axes[1].bar(range(len(bars)), list(bars.values()), color="steelblue")
    axes[1].set_xticks(range(len(bars)))
    axes[1].set_xticklabels(list(bars.keys()), rotation=30, ha="right", fontsize=7)
    axes[1].set_ylabel("mean plateau ridge skill")
    axes[1].set_title("recipe variations (plateau avg)")
    axes[1].axhline(0, color="k", lw=0.6)

    # (c) cross-genre
    if isinstance(res.get("cross_genre"), dict):
        cg = res["cross_genre"]
        L = sorted(cg.keys())
        wb = [cg[li]["within_betley"] for li in L]
        wu = [cg[li]["within_ultrachat"] for li in L]
        bu = [cg[li]["fit_betley_score_ultrachat"] for li in L]
        ub = [cg[li]["fit_ultrachat_score_betley"] for li in L]
        axes[2].plot(L, wb, marker="o", label="within betley")
        axes[2].plot(L, wu, marker="s", label="within ultrachat")
        axes[2].plot(L, bu, marker="^", label="fit betley→score uc")
        axes[2].plot(L, ub, marker="v", label="fit uc→score betley")
        axes[2].set_xlabel("layer")
        axes[2].set_ylabel("ridge / transfer skill")
        axes[2].set_title("cross-genre")
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(0.5, 0.5, "cross-genre SKIPPED", ha="center", va="center")
        axes[2].set_axis_off()

    savefig_paper(fig, "robustness_heatmaps", dir=str(fig_dir))
    plt.close(fig)


def _fig_part3(res: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    rows = [r for r in res["per_layer"] if r.get("nonlinear_gap") is not None]
    L = [r["layer"] for r in rows]
    gap = [r["nonlinear_gap"] for r in rows]
    excl = [r.get("gap_excludes_zero") for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["crimson" if e else "gray" for e in excl]
    ax.bar(L, gap, color=colors)
    # error bars where CI present
    for r in rows:
        ci = r.get("gap_ci95")
        if ci:
            ax.plot([r["layer"], r["layer"]], ci, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("KRR-RBF − linear ridge skill (PCA-48)")
    ax.set_title("nonlinear gap per layer (red = CI excludes 0)")
    savefig_paper(fig, "late_layer_gap", dir=str(fig_dir))
    plt.close(fig)


def _fig_part5(res: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    behaviors = [
        b
        for b in BEHAVIORS
        if isinstance(res["per_behavior"].get(b), dict)
        and res["per_behavior"][b].get("best_direct") is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(behaviors))
    w = 0.38
    direct = [res["per_behavior"][b]["best_direct"]["rho"] for b in behaviors]
    med = [res["per_behavior"][b]["best_mediated"]["rho"] for b in behaviors]
    ax.bar(x - w / 2, direct, w, label="direct (r_Bᵀ v0)", color="steelblue")
    ax.bar(x + w / 2, med, w, label="mediated (r_Bᵀ M̂ c_C)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("best-layer Spearman ρ (pred E0, actual E0)")
    ax.set_title("behavioral-chain preservation: direct vs linear-map-mediated")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8)
    savefig_paper(fig, "behavior_chain", dir=str(fig_dir))
    plt.close(fig)


# ── driver ──────────────────────────────────────────────────────────────────────


def _write(obj: dict, path: Path) -> None:
    obj["metadata"] = reproducibility_metadata({"script": "issue722_structural_battery"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    logger.info("wrote %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--smoke", action="store_true", help="reduced layer set / grid for a fast wiring check"
    )
    ap.add_argument(
        "--parts", default="1,2,3,4,5", help="comma-separated part numbers to run (default all)"
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    ap.add_argument(
        "--no-uc",
        action="store_true",
        help="skip the ultrachat (cross-genre) substrate even if available",
    )
    args = ap.parse_args()

    parts = {int(x) for x in args.parts.split(",") if x.strip()}
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    k_pca = 8 if args.smoke else PCA_TARGET_DIM
    if args.smoke:
        rank_layers = [17, 18]
        plateau = [17, 18]
        cross_layer_grid = [6, 18, 26]
        ctrl_layers = [18]
        chain_layers = [17, 18]
        do_krr = False  # KRR is the expensive arm; skip in smoke
    else:
        rank_layers = list(range(28))
        plateau = PLATEAU_LAYERS
        cross_layer_grid = [0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 27]
        ctrl_layers = list(range(28))
        chain_layers = list(range(28))
        do_krr = True

    t0 = time.time()
    logger.info("loading betley substrate ...")
    betley = load_betley_substrate()
    uc = None
    if not args.no_uc and (2 in parts):
        try:
            logger.info("loading ultrachat substrate ...")
            uc = load_ultrachat_substrate(betley["ctx_ids"])
        except Exception as e:
            logger.warning("ultrachat substrate load failed: %s", e)
            uc = None

    if 1 in parts:
        logger.info("PART 1: rank / structure of M̂ ...")
        res1 = part1_rank_structure(betley, rank_layers, k_pca)
        _write(res1, out_dir / "rank_structure.json")
        _fig_part1(res1, fig_dir)

    if 2 in parts:
        logger.info("PART 2: robustness sweep (do_krr=%s) ...", do_krr)
        res2 = part2_robustness(betley, uc, plateau, k_pca, cross_layer_grid, do_krr)
        _write(res2, out_dir / "robustness.json")
        _fig_part2(res2, fig_dir)

    if 3 in parts:
        logger.info("PART 3: late-layer nonlinearity ...")
        res3 = part3_late_layer(betley, k_pca)
        _write(res3, out_dir / "late_layer.json")
        _fig_part3(res3, fig_dir)

    if 4 in parts:
        logger.info("PART 4: is c_C special (control) ...")
        res4 = part4_cC_control(betley, ctrl_layers, k_pca)
        _write(res4, out_dir / "cC_control.json")

    if 5 in parts:
        logger.info("PART 5: behavioral-chain preservation ...")
        res5 = part5_behavior_chain(betley, chain_layers)
        _write(res5, out_dir / "behavior_chain.json")
        _fig_part5(res5, fig_dir)

    logger.info("DONE in %.1fs (parts=%s, smoke=%s)", time.time() - t0, sorted(parts), args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
