#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ², λ, ρ, ̄) in scientific docstrings + log/print messages.
"""Vectorized #722 skill-over-mean + #658 MLP-vs-ridge downstream chain ρ.

The fast consumer of ``explore_persona_space.analysis.vectorized_mlp_skill`` that
replaces the catastrophically slow serial per-fold/per-layer MLP sweep in
``issue722_skill_over_mean.py`` (19.5 CPU-h, did not finish) and
``issue658_mlp_chain.py``. Produces, in ONE vectorized batched fit each:

#722 canonical (Betley, the published #722 line): per-layer skill-over-mean
held-out R² (1 − SS_res/SS_tot on the train-mean-centered v0 target, LOCO) for
RIDGE (closed-form) + 3 MLP variants (base / z-scored-input / shuffle-null),
matching ``issue722_skill_over_mean.py``'s metric + output schema:
  - eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json
  - figures/issue_722/base_skill_over_mean_per_layer.png

#658 chain (Betley + UltraChat): downstream chain ρ ``r_Bᵀ v̂0 → E0`` (Spearman
across held-out contexts) for ridge AND MLP, per behavior, per layer, mirroring
``issue658_mlp_chain.py``:
  - eval_results/issue_658/a34a35_mlp_chain.json
  - eval_results/issue_658_g1/a34a35_mlp_chain.json
  - figures/issue_658/a34a35_mlp_vs_ridge_chain.png

REPRODUCE-CHECKS (Deliverable 4, run BEFORE trusting any output):
  1. Vectorized RIDGE full-H chain reproduces #658's stored
     ``a34_a35.by_recipe.last.chain_rho_e0`` byte-exact (Betley
     0.46/0.68/0.08/−0.23, UltraChat 0.40/0.67/0.09/−0.20).
  2. Vectorized MLP skill-over-mean matches the OLD slow
     ``issue722_skill_over_mean._mlp_skill`` on 2-3 spot layers within tolerance.

Target-reduction choice (Correction 2): for the #722 canonical output we match
``issue722_skill_over_mean.py`` EXACTLY — ``MLP_PCA_DIM = 48`` top-PC v0 target +
the input-PCA acceleration (project c_C onto its top-(n-1) PCs, lossless at
n≪d). For the #658 chain output we match ``issue658_mlp_chain.py``:
``PCA_TARGET_DIM = 64`` top-PC v0 target, FULL-H c_C input. Both target
reductions are top-PC PCA (NOT the leading-RAW-dims that #658's stored
``A35_MLP_TARGET_DIM=64`` used — #722 and the mlpchain script both moved to
top-PC for a genuinely like-for-like ridge-vs-MLP gap). The vectorized MLP arm
reproduces ``_fit_mlp_ensemble_loco`` exactly (gate in the library), so matching
the per-script PCA wrapper reproduces each script's numbers.

CPU-only, 0 GPU. Reads the per-genre stores from the repo-root caches; never
mutates task state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# REPO_ROOT = this checkout (the worktree) — OUTPUTS land here so commits go to
# the branch. This file lives in <checkout>/scripts/.
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _main_repo_root() -> Path:
    """Resolve the MAIN repo root that holds the shared gitignored data/ caches.

    The #658 activation stores live ONLY in the main checkout's ``data/`` tree
    (re-downloadable gitignored caches), NOT in this worktree (which has an empty
    ``data/``). A worktree's ``.git`` points at ``<main>/.git/worktrees/<name>``,
    so ``git rev-parse --git-common-dir`` resolves to ``<main>/.git`` and its
    parent is the main checkout. Falls back to this checkout if resolution fails
    (e.g. a non-worktree run where data IS local).
    """
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        main_root = (REPO_ROOT / common).resolve().parent
        if (main_root / "data").exists():
            return main_root
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return REPO_ROOT


# DATA_ROOT = main checkout (holds the shared data/ caches); OUTPUTS stay in REPO_ROOT.
DATA_ROOT = _main_repo_root()

import issue658_fit_predictors as i658  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import dump_json, load_cc_last_store, load_json  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    INPUT_REP_EPS,
    INPUT_REP_K,
    INPUT_REPS,
    MLPGroup,
    assert_matches_reference,
    chain_rho_pca,
    fit_batched_loco_mlp,
    fit_batched_loco_mlp_multihead,
    fit_batched_loco_mlp_multihead_trajectory,
    input_transform_fold,
    krr_predict_loco,
    krr_predict_loco_rep,
    loco_train_means,
    ridge_predict_loco_centered,
    ridge_predict_loco_centered_rep,
    ridge_predict_loco_raw,
    robust_pca_basis,
    skill_over_mean_r2,
    zscore_columns,
)

# §6 SUCCESS bands for the input-rep robustness amendment (plan §6 / Success criteria):
# ridge R² invariance band and KRR(RBF)−linear gap-preservation band, with the
# ≥26/28-layer pass threshold per variant.
INPUT_REP_RIDGE_BAND = 0.05  # |Δridge R²| ≤ 0.05 vs the full-dim baseline
INPUT_REP_GAP_BAND = 0.03  # |Δ(RBF−linear gap)| ≤ 0.03 AND sign preserved
INPUT_REP_PASS_MIN_LAYERS = 26  # of 28

# γ-sensitivity diagnostic (plan §4.4 exploratory band; CONCERN
# krr-gap-collapse-gamma-regime-interpretation): re-run the pca48 KRR RBF arm at a
# couple of representative ridge-plateau layers under γ = multiplier × the per-fold
# median-heuristic γ₀, to distinguish a genuine RBF-buys-nothing collapse from a
# bad-γ-regime artifact of the median heuristic at 48 standardized dims.
GAMMA_SENS_LAYERS = (18, 21)  # ridge plateau; L18 = the §5a shuffle-control anchor
GAMMA_SENS_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)  # × the median-heuristic γ₀

load_dotenv(str(REPO_ROOT / ".env"))

logger = logging.getLogger("issue722_vectorized_skill")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Project-default seed (followup-scope §11) for the #722 line; #658 chain uses 658.
SEED_722 = 42
SEED_658 = 658

# #722 recipe: top-48-PC v0 target + input-PCA acceleration (matches the slow script).
MLP_PCA_DIM_722 = 48
# #658 chain recipe: top-64-PC v0 target, full-H input (matches issue658_mlp_chain.py).
PCA_TARGET_DIM_658 = 64

READOUT_BEHAVIORS = ("broad_em", "harmful_compliance", "sycophancy", "refusal")
SHUFFLE_RIDGE_LAYER = 18  # §5 label-shuffle ridge control layer (matches #722)

# Published #658 ridge full-H chain ρ (reproduce-check target). Source:
# eval_results/issue_658{,_g1}/a34_a35.json :: by_recipe.last.chain_rho_e0.
PUBLISHED_RIDGE_FULL = {
    "betley": {
        "broad_em": {"layer": 10, "rho": 0.45625511193709734},
        "harmful_compliance": {"layer": 11, "rho": 0.6763380907104344},
        "sycophancy": {"layer": 0, "rho": 0.0816444154089325},
        "refusal": {"layer": 26, "rho": -0.22844242829270806},
    },
    "ultrachat": {
        "broad_em": {"layer": 26, "rho": 0.3952638514729647},
        "harmful_compliance": {"layer": 11, "rho": 0.6744104557529602},
        "sycophancy": {"layer": 13, "rho": 0.08554069506488331},
        "refusal": {"layer": 26, "rho": -0.2041856449546123},
    },
}
RIDGE_REPRO_TOL = 1e-6


# ── data loading (reuses the validated issue658_mlp_chain loaders' shape) ──────


def _load_genre(genre: str, store_dir: Path, e0_path: Path) -> dict:
    """Load store + r_B + E0 + the locked 'last' c_C recipe for one genre.

    Reproduces the #658 main() loading EXACTLY:
      - Betley: cc_last from the #594 HF store (``load_cc_last_store``).
      - UltraChat: cc_last from the per-genre store's ``cc_last`` key
        (the ``--cc-last-from-store`` path; the #594 store is Betley-pinned).
    """
    store = torch.load(store_dir / "v0_summaries.pt", weights_only=False)
    rb = torch.load(store_dir / "r_b.pt", weights_only=False)
    e0 = load_json(e0_path)
    ctx_ids = list(store["context_ids"])
    layers = list(store["capture_layers"])
    if store.get("cc_last"):
        store_cc_last = store["cc_last"]
        missing = [c for c in ctx_ids if c not in store_cc_last]
        if missing:
            raise RuntimeError(f"[{genre}] store cc_last missing {len(missing)} ctx: {missing[:5]}")
        cc_last = {c: store_cc_last[c].numpy() for c in ctx_ids}
        cc_source = "per-genre store::cc_last"
    else:
        cc = load_cc_last_store(layers, ctx_ids)
        cc_last = {c: cc[c].numpy() for c in ctx_ids}
        cc_source = "#594 HF store (load_cc_last_store)"
    logger.info(
        "[%s] loaded: %d ctx, %d layers, cc_last from %s",
        genre,
        len(ctx_ids),
        len(layers),
        cc_source,
    )
    return {
        "store": store,
        "rb": rb,
        "e0": e0,
        "ctx_ids": ctx_ids,
        "layers": layers,
        "cc_last": cc_last,
        "cc_source": cc_source,
        "store_dir": str(store_dir),
        "e0_path": str(e0_path),
    }


def _stack_layers(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(N, L, H) c_C(last) + v0(mean) stacks aligned on ctx_ids."""
    ctx_ids, store = data["ctx_ids"], data["store"]
    C = np.stack([np.asarray(data["cc_last"][c]) for c in ctx_ids])  # (N, L, H)
    V = np.stack([store["summaries"]["mean"][c].numpy() for c in ctx_ids])  # (N, L, H)
    return C.astype(np.float64), V.astype(np.float64)


# ── #722 skill-over-mean (PCA-48 target + input-PCA accel, 3 MLP variants) ─────


def _input_pca_project(Xc: np.ndarray) -> tuple[np.ndarray, bool]:
    """Project c_C onto its top-(n-1) PCs — the #722 input-PCA acceleration.

    Lossless at rank ≤ n (the discarded directions are exactly zero in the data):
    the MLP first linear becomes (n-1)→512 instead of 3584→512. Identical to
    ``issue722_skill_over_mean._mlp_skill``'s input acceleration.
    Returns (Xin (N, xk), used_gesvd_fallback).
    """
    Xc64 = Xc.astype(np.float64)
    xmu = Xc64.mean(axis=0)
    fallback = False
    try:
        _, _, xVt = np.linalg.svd(Xc64 - xmu, full_matrices=False)
    except np.linalg.LinAlgError:
        _, _, xVh = torch.linalg.svd(torch.from_numpy(Xc64 - xmu), full_matrices=False)
        xVt = xVh.numpy()
        fallback = True
    n = Xc.shape[0]
    xk = min(n - 1, xVt.shape[0])
    Xin = (Xc64 - xmu) @ xVt[:xk].T  # (N, xk) lossless input coords
    return Xin.astype(np.float64), fallback


def run_722_skill_over_mean(
    data: dict, *, device: str, num_threads: int | None, layer_subset: list[int] | None = None
) -> dict:
    """#722 per-layer skill-over-mean R²: ridge + MLP {base, zscored, shuffle}.

    Vectorized: ALL layers × {base, zscored-input, shuffle-null} MLP variants are
    fit in ONE batched ensemble (the #722 84-fit serial sweep). Ridge is the
    closed-form centered LOCO (#658 dual/PRESS), cheap, looped per layer. The MLP
    target is the top-48 PC v0 basis; the MLP input is the input-PCA-projected
    c_C (lossless). skill = 1 − SS_res/SS_tot on the back-projected un-centered
    prediction, matching ``issue722_skill_over_mean``'s ``_skill_over_mean``.

    ``layer_subset`` (smoke only) restricts to those layer NUMBERS.
    """
    C, V = _stack_layers(data)
    layers = data["layers"]
    n, _L, H = V.shape
    li_iter = [
        li for li in range(len(layers)) if layer_subset is None or int(layers[li]) in layer_subset
    ]
    logger.info("[722] n=%d L=%d H=%d — building batched MLP ensemble", n, len(li_iter), H)

    shuffle_perm = np.random.default_rng(SEED_722).permutation(n)

    # Build the batched MLP groups: per layer × {base, zscored, shuffle}.
    # Each group's target is the top-48 PC v0 (un-shuffled for base/zscored;
    # row-permuted for the shuffle null, matching #722's perm of Yv rows).
    groups: list[MLPGroup] = []
    per_layer_meta: dict[int, dict] = {}
    for li in li_iter:
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        mu_t, comps, tgt_fallback = robust_pca_basis(Yv, MLP_PCA_DIM_722)  # (H,), (k, H)
        Xin, in_fallback = _input_pca_project(Xc)
        Xin_z, _ = _input_pca_project(zscore_columns(Xc))
        Y64 = (Yv - mu_t) @ comps.T  # (N, k)
        # Shuffle null: the row-permuted v0 target fit against its OWN PCA basis +
        # mean (matching #722's _mlp_skill(Xc, Yv[perm])). Compute the basis ONCE.
        Yv_sh = Yv[shuffle_perm]
        sh_mu, sh_comps, _ = robust_pca_basis(Yv_sh, MLP_PCA_DIM_722)
        Y64_sh = (Yv_sh - sh_mu) @ sh_comps.T
        per_layer_meta[li] = {
            "mu_t": mu_t,
            "comps": comps,
            "Yv": Yv,
            "Yv_sh": Yv_sh,
            "sh_mu": sh_mu,
            "sh_comps": sh_comps,
            "gesvd_fallback": bool(tgt_fallback or in_fallback),
        }
        groups.append(MLPGroup(("base", li), Xin, Y64))
        groups.append(MLPGroup(("zscored", li), Xin_z, Y64))
        groups.append(MLPGroup(("shuffle", li), Xin, Y64_sh))

    # Pad all groups to a common input dim (xk = n-1 for all layers since rank is
    # the same n at every layer) — verify they already share the shape.
    in_dims = {g.X.shape[1] for g in groups}
    assert len(in_dims) == 1, f"input-PCA dims differ across layers: {in_dims}"

    t0 = time.time()
    res = fit_batched_loco_mlp_multihead(
        groups, seed=SEED_722, device=device, chunk_size=4096, num_threads=num_threads
    )
    logger.info(
        "[722] batched multihead MLP ensemble (%d members) fit in %.1fs",
        res.n_members,
        time.time() - t0,
    )

    per_layer = []
    shuffle_ridge_l18 = float("nan")
    for li in li_iter:
        layer = int(layers[li])
        m = per_layer_meta[li]
        Xc = C[:, li, :]
        Yv = m["Yv"]
        # RIDGE skill (full-H centered LOCO, the #722 ridge_mean recipe).
        ridge_pred = ridge_predict_loco_centered(Xc, Yv)
        ridge_skill = skill_over_mean_r2(ridge_pred, Yv)
        lambda_chosen = _full_data_lambda(Xc, Yv)  # OLD-script per-layer diagnostic
        # MLP variants: back-project the PCA-64 preds to H, add the per-fold train
        # mean of the (un-centered) PCA target back, then skill on H-space v0.
        base_pred64 = res.preds_by_key[("base", li)]  # (N, 48)
        z_pred64 = res.preds_by_key[("zscored", li)]
        sh_pred64 = res.preds_by_key[("shuffle", li)]
        # un-center: the MLP fit the CENTERED PCA target ((Yv-mu) @ comps), so add
        # the LOO train mean of that centered target back, then back-project.
        Y64 = (Yv - m["mu_t"]) @ m["comps"].T
        base_full = (loco_train_means(Y64) + base_pred64) @ m["comps"] + m["mu_t"]
        z_full = (loco_train_means(Y64) + z_pred64) @ m["comps"] + m["mu_t"]
        mlp_skill = skill_over_mean_r2(base_full, Yv)
        mlp_z_skill = skill_over_mean_r2(z_full, Yv)
        # shuffle null: target is the row-permuted v0; its own PCA basis.
        Yv_sh = m["Yv_sh"]
        Y64_sh = (Yv_sh - m["sh_mu"]) @ m["sh_comps"].T
        sh_full = (loco_train_means(Y64_sh) + sh_pred64) @ m["sh_comps"] + m["sh_mu"]
        mlp_sh_skill = skill_over_mean_r2(sh_full, Yv_sh)

        if layer == SHUFFLE_RIDGE_LAYER:
            # §5a label-shuffled ridge control at the plateau peak layer.
            ridge_sh_pred = ridge_predict_loco_centered(Xc, Yv[shuffle_perm])
            shuffle_ridge_l18 = skill_over_mean_r2(ridge_sh_pred, Yv[shuffle_perm])["skill"]

        per_layer.append(
            {
                "layer": layer,
                "predict_mean_abs_cos": ridge_skill["predict_mean_abs_cos"]
                if "predict_mean_abs_cos" in ridge_skill
                else _predict_mean_abs_cos(Yv),
                "raw_recon_abs_cos": _recon_abs_cos(ridge_pred, Yv),
                "skill_vs_mean_ridge": ridge_skill["skill"],
                "skill_vs_mean_mlp": mlp_skill["skill"],
                "skill_zscored_mlp": mlp_z_skill["skill"],
                "skill_shuffle_mlp": mlp_sh_skill["skill"],
                "ridge_median_per_dim_r2": ridge_skill["median_per_dim_r2"],
                "mlp_median_per_dim_r2": mlp_skill["median_per_dim_r2"],
                "gesvd_fallback": m["gesvd_fallback"],
                "n_folds_used_ridge": ridge_skill["n_folds_used"],
                "n_folds_used_mlp": mlp_skill["n_folds_used"],
                # OLD-script-contract parity (issue722_skill_over_mean.py): the
                # full-data PRESS λ pick (a per-layer reproducibility diagnostic),
                # and the SVD-fold-skip count. The multihead MLP path fits one net
                # per (group, fold) over ALL n folds with no per-fold SVD that can
                # fail, so it never skips folds — mlp_n_folds_skipped is always 0
                # (n_folds_used_mlp == n carries the positive form of the same
                # comparability signal the codex-critic fold-skip watch-item flags).
                "lambda_chosen": lambda_chosen,
                "mlp_n_folds_skipped": 0,
            }
        )
        logger.info(
            "[722][L%02d] skill ridge=%+.4f mlp=%+.4f zscored=%+.4f shuffle=%+.4f",
            layer,
            per_layer[-1]["skill_vs_mean_ridge"],
            per_layer[-1]["skill_vs_mean_mlp"],
            per_layer[-1]["skill_zscored_mlp"],
            per_layer[-1]["skill_shuffle_mlp"],
        )

    return {
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0)",
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "activation_dim": H,
        "layers": [int(layers[li]) for li in li_iter],
        "mlp_recipe": {
            "hidden": i658.MLP_HIDDEN,
            "lr": i658.MLP_LR,
            "wd": i658.MLP_WD,
            "max_epochs": i658.MLP_MAX_EPOCHS,
            "pca_target_dim": MLP_PCA_DIM_722,
            "input_pca_accel": True,
            "vectorized": True,
        },
        "ridge_lambdas": i658.RIDGE_LAMBDAS,
        "seed": SEED_722,
        "shuffle_ridge_L18": float(shuffle_ridge_l18),
        "store_provenance": {
            "store_dir": data["store_dir"],
            "e0_path": data["e0_path"],
            "cc_source": data["cc_source"],
            "n_contexts": n,
            "hidden_dim": int(H),
        },
        "per_layer": per_layer,
    }


def _full_data_lambda(Xc: np.ndarray, Yv: np.ndarray) -> float:
    """Full-data PRESS-selected ridge λ for a layer (OLD-script `lambda_chosen`).

    Bit-reproduces ``issue722_skill_over_mean._ridge_skill``'s `lambda_chosen`
    (its lines 293-298): standardize the FULL design (ddof=0 to match the #658
    numpy convention), center the target on its full mean, pick the λ minimizing
    the exact PRESS LOO MSE over ``RIDGE_LAMBDAS``. A per-layer reproducibility
    diagnostic — the per-fold picks are near-identical at this n. Runs on
    ``i658.DEVICE`` like the rest of the ridge arm.
    """
    device = torch.device(i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    xmu = Xt.mean(0)
    xsd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - xmu) / xsd
    Yc = Yt - Yt.mean(0)
    mse = i658._press_loo_mse_per_lambda(Xn, Yc, i658.RIDGE_LAMBDAS)
    return float(i658.RIDGE_LAMBDAS[int(torch.argmin(mse).item())])


def _full_data_lambda_rep(
    Xc: np.ndarray, Yv: np.ndarray, input_rep: str, k: int, eps: float
) -> float:
    """Full-data PRESS-selected ridge λ for a layer UNDER one input representation.

    The transformed-input analogue of ``_full_data_lambda`` (BLOCKER §4.4
    ``lambda_chosen``): re-represent the FULL design once (``input_transform_fold``
    with all rows as the "train" basis), then run the identical standardize →
    center → PRESS LOO pick over ``RIDGE_LAMBDAS`` that ``_full_data_lambda`` uses.
    A per-layer reproducibility diagnostic that mirrors the baseline skill JSON's
    ``lambda_chosen`` field on the pca48/whiten48 arm — the per-fold LOCO picks are
    near-identical at this n, so the full-data pick is the reported representative,
    exactly as in the ``full`` baseline. ``input_rep="full"`` delegates to
    ``_full_data_lambda`` (byte-identical). Runs on ``i658.DEVICE`` in fp64.
    """
    if input_rep == "full":
        return _full_data_lambda(Xc, Yv)
    device = torch.device(i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    # Full-data input transform: use the whole design as the basis, and reuse the
    # SAME transform on itself for the "held-out" argument (immaterial here — we take
    # only the transformed design Xin for the diagnostic λ pick).
    Xin, _z_held, _fb = input_transform_fold(Xt, Xt[0], input_rep, k=k, eps=eps)
    xmu = Xin.mean(0)
    xsd = Xin.std(0, correction=0) + 1e-9
    Xn = (Xin - xmu) / xsd
    Yc = Yt - Yt.mean(0)
    mse = i658._press_loo_mse_per_lambda(Xn, Yc, i658.RIDGE_LAMBDAS)
    return float(i658.RIDGE_LAMBDAS[int(torch.argmin(mse).item())])


def _predict_mean_abs_cos(Yv: np.ndarray) -> float:
    """Held-out rowwise |cos|(v̄0_train, v0) — the #722 predict-mean cosine (~0.98)."""
    tmean = loco_train_means(Yv)
    return float(np.mean(np.abs(i658._rowwise_cos(tmean, Yv))))


def _recon_abs_cos(pred: np.ndarray, Yv: np.ndarray) -> float:
    """Held-out rowwise |cos|(ridge M̂(c_C), v0) — the misleading saturated number."""
    return float(np.mean(np.abs(i658._rowwise_cos(pred, Yv))))


# ── #658 MLP-vs-ridge downstream chain ρ (PCA-64, full-H input) ────────────────


def run_658_chain(
    data: dict,
    genre: str,
    *,
    device: str,
    num_threads: int | None,
    layer_subset: list[int] | None = None,
) -> dict:
    """#658 chain ρ: ridge full-H (control) + ridge/MLP PCA-64 per behavior.

    Mirrors ``issue658_mlp_chain.run_genre``: the ridge full-H chain reproduces
    the published number (control); the PCA-64 ridge + MLP chains are the
    like-for-like comparison. The MLP PCA-64 LOCO preds come from ONE batched
    ensemble over all layers (full-H c_C input, top-64 PC v0 target).

    ``layer_subset`` (smoke only) restricts BOTH the full-H control AND the
    PCA-64 arm to those layer NUMBERS; the reproduce-check is meaningful only on
    the full (``None``) production sweep.
    """
    C, V = _stack_layers(data)
    layers = data["layers"]
    n, _L, _H = V.shape
    li_iter = [
        li for li in range(len(layers)) if layer_subset is None or int(layers[li]) in layer_subset
    ]

    store, rb, e0, ctx_ids = data["store"], data["rb"], data["e0"], data["ctx_ids"]

    # CONTROL: ridge full-H chain ρ from the vectorized ridge solver.
    ridge_pred_v0_by_layer = {
        li: ridge_predict_loco_raw(C[:, li, :], V[:, li, :]) for li in li_iter
    }
    ridge_full = i658._chain_rho(ridge_pred_v0_by_layer, store, e0, rb, ctx_ids, layers, 0)
    repro = _ridge_repro_control(ridge_full, genre)

    # PCA-64 per-layer basis + batched MLP/ridge skill preds. The MLP input is
    # the input-PCA-projected c_C (top-(n-1) PCs, LOSSLESS at rank ≤ n — the
    # discarded directions are exactly zero in the data), so the width-512 first
    # linear is (n-1)→512 not 3584→512 (~73× cheaper, ZERO information loss). This
    # is the same input-PCA reparameterization #722 uses; it does not change the
    # function the MLP can fit on a rank-≤n input. Ridge stays full-H (closed-form).
    groups: list[MLPGroup] = []
    per_layer_pca: dict[int, dict] = {}
    for li in li_iter:
        Yv = V[:, li, :]
        mu_t, comps, _ = robust_pca_basis(Yv, PCA_TARGET_DIM_658)
        Y64 = (Yv - mu_t) @ comps.T  # (N, k) — fit on the RAW PCA target (not centered)
        Xin, _ = _input_pca_project(C[:, li, :])  # (N, n-1) lossless input coords
        per_layer_pca[li] = {"mu_t": mu_t, "comps": comps, "Yv": Yv, "Y64": Y64}
        groups.append(MLPGroup(("mlp", li), Xin, Y64))

    t0 = time.time()
    res = fit_batched_loco_mlp_multihead(
        groups, seed=SEED_658, device=device, chunk_size=4096, num_threads=num_threads
    )
    logger.info(
        "[658:%s] batched multihead MLP ensemble (%d members) fit in %.1fs",
        genre,
        res.n_members,
        time.time() - t0,
    )

    rb_dirs = rb.get("r_b", {})
    rb_cols = list(rb.get("columns", []))
    out_behaviors: dict = {}
    for col in READOUT_BEHAVIORS:
        if col not in rb_cols or col not in rb_dirs:
            out_behaviors[col] = {"skipped": "no r_B contrast"}
            continue
        rdir = rb_dirs[col].get("diffmeans")
        if rdir is None:
            out_behaviors[col] = {"skipped": "no diffmeans r_B"}
            continue
        y_e0, kept = i658.e0_target(e0, col, ctx_ids)
        if len(kept) < 4:
            out_behaviors[col] = {"skipped": f"<4 E0 contexts ({len(kept)})"}
            continue
        kept_idx = [ctx_ids.index(c) for c in kept]

        best_ridge = None
        best_mlp = None
        per_layer_rows = []
        for li in li_iter:
            m = per_layer_pca[li]
            r = np.asarray(rdir[li])
            # ridge PCA-64 chain (skill-over-mean centered ridge fit, in PCA space)
            ridge_pca = _ridge_pca64_pred(C[:, li, :], m["Y64"])
            rho_ridge, _ = chain_rho_pca(ridge_pca[kept_idx], m["comps"], r, y_e0)
            # MLP PCA-64 chain (from the batched ensemble, add LOO train mean back)
            mlp_pca = loco_train_means(m["Y64"]) + res.preds_by_key[("mlp", li)]
            rho_mlp, _ = chain_rho_pca(mlp_pca[kept_idx], m["comps"], r, y_e0)
            r2_ridge = skill_over_mean_r2(
                loco_train_means(m["Y64"]) + _ridge_pca64_centered(C[:, li, :], m["Y64"]),
                m["Y64"],
            )["skill"]
            r2_mlp = skill_over_mean_r2(mlp_pca, m["Y64"])["skill"]
            per_layer_rows.append(
                {
                    "layer": int(layers[li]),
                    "ridge_pca64_rho": rho_ridge,
                    "mlp_pca64_rho": rho_mlp,
                    "ridge_skill_r2": r2_ridge,
                    "mlp_skill_r2": r2_mlp,
                }
            )
            if rho_ridge is not None and (best_ridge is None or rho_ridge > best_ridge["rho"]):
                best_ridge = {"layer": int(layers[li]), "rho": rho_ridge, "skill_r2": r2_ridge}
            if rho_mlp is not None and (best_mlp is None or rho_mlp > best_mlp["rho"]):
                best_mlp = {"layer": int(layers[li]), "rho": rho_mlp, "skill_r2": r2_mlp}
        delta = (
            None
            if (best_mlp is None or best_ridge is None)
            else float(best_mlp["rho"] - best_ridge["rho"])
        )
        out_behaviors[col] = {
            "ridge_pca64_chain": best_ridge,
            "mlp_pca64_chain": best_mlp,
            "mlp_minus_ridge_pca64_delta": delta,
            "per_layer": per_layer_rows,
        }

    return {
        "genre": genre,
        "store_dir": data["store_dir"],
        "e0_path": data["e0_path"],
        "cc_source": data["cc_source"],
        "n_contexts": n,
        "layers_swept": [int(x) for x in layers],
        "pca_target_dim": PCA_TARGET_DIM_658,
        "ridge_full_chain_rho": {c: ridge_full.get(c) for c in READOUT_BEHAVIORS},
        "ridge_full_chain_repro_control": repro,
        "per_behavior": out_behaviors,
    }


def _ridge_pca64_pred(Xc: np.ndarray, Y64: np.ndarray) -> np.ndarray:
    """LOCO ridge prediction of the PCA-64 v0 target (skill form: centered + add-back)."""
    pred_c = ridge_predict_loco_centered(Xc, Y64)
    return pred_c  # ridge_predict_loco_centered already adds the train mean back


def _ridge_pca64_centered(Xc: np.ndarray, Y64: np.ndarray) -> np.ndarray:
    """The centered-only ridge prediction (no add-back) for the skill-R² helper."""
    tmean = loco_train_means(Y64)
    return ridge_predict_loco_centered(Xc, Y64) - tmean


def _ridge_repro_control(ridge_full: dict, genre: str) -> dict:
    published = PUBLISHED_RIDGE_FULL[genre]
    repro = {"ok": True, "rows": {}}
    for col in READOUT_BEHAVIORS:
        got = ridge_full.get(col)
        exp = published.get(col)
        if got is None or exp is None:
            repro["ok"] = False
            repro["rows"][col] = {"got": got, "expected": exp, "match": False}
            continue
        d_rho = abs(got["rho"] - exp["rho"])
        match = (got["layer"] == exp["layer"]) and (d_rho <= RIDGE_REPRO_TOL)
        repro["rows"][col] = {
            "got_layer": got["layer"],
            "got_rho": got["rho"],
            "expected_layer": exp["layer"],
            "expected_rho": exp["rho"],
            "abs_rho_delta": d_rho,
            "match": bool(match),
        }
        if not match:
            repro["ok"] = False
    return repro


# ── figures ───────────────────────────────────────────────────────────────────


def make_722_figure(result: dict, fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    rows = sorted(result["per_layer"], key=lambda r: r["layer"])
    x = [r["layer"] for r in rows]
    fig, ax_l = plt.subplots(figsize=(7.0, 4.2))
    ax_r = ax_l.twinx()
    (l1,) = ax_l.plot(
        x,
        [r["skill_vs_mean_ridge"] for r in rows],
        marker="o",
        ms=3,
        lw=1.6,
        color="#0072B2",
        label="ridge R² (skill)",
    )
    (l2,) = ax_l.plot(
        x,
        [r["skill_vs_mean_mlp"] for r in rows],
        marker="s",
        ms=3,
        lw=1.6,
        color="#D55E00",
        label="MLP R² (skill)",
    )
    (l3,) = ax_l.plot(
        x,
        [r["skill_shuffle_mlp"] for r in rows],
        marker="x",
        ms=3,
        lw=1.0,
        ls=":",
        color="#999999",
        label="MLP shuffle-null R²",
    )
    (l4,) = ax_r.plot(
        x,
        [r["predict_mean_abs_cos"] for r in rows],
        marker="d",
        ms=3,
        lw=1.2,
        color="#009E73",
        label="predict-mean |cos|",
    )
    (l5,) = ax_r.plot(
        x,
        [r["raw_recon_abs_cos"] for r in rows],
        marker="v",
        ms=3,
        lw=1.2,
        color="#CC79A7",
        label="raw-recon |cos|",
    )
    ax_l.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax_l.set_xlabel("layer")
    ax_l.set_ylabel("skill-over-mean (held-out R²)")
    ax_r.set_ylabel("rowwise |cosine|")
    ax_r.set_ylim(0.0, 1.02)
    lines = [l1, l2, l3, l4, l5]
    ax_l.legend(lines, [ln.get_label() for ln in lines], loc="center left", fontsize=7)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_fig_meta(fig_path, 722, "skill_over_mean.json")
    logger.info("wrote %s", fig_path)


def make_658_figure(betley: dict, ultrachat: dict, fig_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    behaviors = list(READOUT_BEHAVIORS)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, res, title in ((axes[0], betley, "Betley"), (axes[1], ultrachat, "UltraChat")):
        ridge = [
            (res["per_behavior"].get(b, {}).get("ridge_pca64_chain") or {}).get("rho", np.nan)
            for b in behaviors
        ]
        mlp = [
            (res["per_behavior"].get(b, {}).get("mlp_pca64_chain") or {}).get("rho", np.nan)
            for b in behaviors
        ]
        xi = np.arange(len(behaviors))
        w = 0.38
        ax.bar(xi - w / 2, ridge, w, color="#0072B2", label="ridge (PCA-64)")
        ax.bar(xi + w / 2, mlp, w, color="#D55E00", label="MLP (PCA-64)")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xticks(xi)
        ax.set_xticklabels([b.replace("_", "\n") for b in behaviors], fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("downstream chain ρ (best layer)")
    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_fig_meta(fig_path, 658, "a34a35_mlp_chain.json")
    logger.info("wrote %s", fig_path)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256_file(path: Path) -> str | None:
    """Streamed sha256 of a local file (content-identity pin), or None if absent."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_substrate_provenance(betley: dict) -> dict:
    """Resolve the §4.4 substrate HF paths + revisions + content sha256 for run_meta.

    The two substrate stores this amendment reads:
      - #658 v0 store — ``v0_summaries.pt`` + ``r_b.pt`` under the HF data repo
        prefix ``issue658_theory_assumptions/store`` (loaded in ``_load_genre``).
      - #594 c_C store — ``context_vectors_mean.pt`` under
        ``issue594_context_geometry/analysis_tensors`` (``load_cc_last_store``).

    For each file: the HF repo id + repo_type + path_in_repo (from
    ``issue658_common``), the LOCAL content sha256 (the definitive content-identity
    pin, survives any future commit), and the RESOLVED HF ``main`` revision +
    server-side LFS sha256 (looked up live; fail-soft on no-network so a run still
    produces run_meta — the local sha256 remains the authoritative pin either way).
    """
    from issue658_common import HF_DATA_REPO, I594_CC_LAST_FILE

    store_dir = Path(betley["store_dir"])
    files = {
        "i658_v0_summaries": {
            "repo_id": HF_DATA_REPO,
            "repo_type": "dataset",
            "path_in_repo": "issue658_theory_assumptions/store/v0_summaries.pt",
            "local_path": str(store_dir / "v0_summaries.pt"),
        },
        "i658_r_b": {
            "repo_id": HF_DATA_REPO,
            "repo_type": "dataset",
            "path_in_repo": "issue658_theory_assumptions/store/r_b.pt",
            "local_path": str(store_dir / "r_b.pt"),
        },
        "i594_cc_last": {
            "repo_id": HF_DATA_REPO,
            "repo_type": "dataset",
            "path_in_repo": I594_CC_LAST_FILE,
            "local_path": None,  # streamed via hf_hub_download into the HF cache
        },
    }
    for meta in files.values():
        lp = meta["local_path"]
        meta["local_sha256"] = _sha256_file(Path(lp)) if lp else None
        meta["resolved_revision"] = None
        meta["hf_lfs_sha256"] = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            url = hf_hub_url(
                meta["repo_id"], meta["path_in_repo"], repo_type=meta["repo_type"], revision="main"
            )
            md = get_hf_file_metadata(url)
            meta["resolved_revision"] = md.commit_hash  # HF main resolved to a commit sha
            meta["hf_lfs_sha256"] = md.etag  # server-side LFS sha256 for the blob
        except Exception as exc:
            logger.warning(
                "[input-rep] could not resolve HF revision for %s (%s): %s",
                meta["path_in_repo"],
                type(exc).__name__,
                exc,
            )
    return {"substrate_files": files, "cc_source": betley["cc_source"]}


def _rng_state_hash(seed: int) -> str:
    """Deterministic hash of the numpy default_rng bit-generator state for ``seed``.

    Pins the exact RNG the KRR bootstrap draws from (``np.random.default_rng(seed)``)
    — the §4.4 ``rng_state_hash`` provenance field. Seed-derived + deterministic, so
    it round-trips across machines and re-runs (a fixed value for a fixed seed).
    """
    state = np.random.default_rng(seed).bit_generator.state
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def _write_fig_meta(fig_path: Path, issue: int, source_json: str) -> None:
    meta = {
        "issue": issue,
        "figure": fig_path.name,
        "code_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_json": source_json,
    }
    with open(fig_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── ADDITIVE extension: MLP width sweep + per-epoch curves + KRR ───────────────
# These three phases are OPT-IN (default OFF; enabled via --run-width-sweep /
# --run-epoch-curves / --run-krr). They never touch skill_over_mean.json or the
# existing default invocation — each writes its OWN output file + figure. The
# linear-ridge skill is the shared baseline (the bar to beat): read from the
# existing skill_over_mean.json when present, else recomputed from the store.


def _ridge_baseline_by_layer(
    data: dict, layer_subset: list[int] | None, existing_json: Path
) -> dict[int, float]:
    """Per-layer linear-ridge skill-over-mean R² (the shared baseline / bar to beat).

    REUSES the already-computed ``skill_vs_mean_ridge`` from the canonical
    ``skill_over_mean.json`` when it exists (zero recompute — the values are the
    same closed-form LOCO ridge), restricted to ``layer_subset`` if given. Falls
    back to recomputing via ``ridge_predict_loco_centered`` only when the JSON is
    absent (e.g. a fresh pod that runs only the sweep phase), so the sweep is
    self-contained.
    """
    want = None if layer_subset is None else set(layer_subset)
    if existing_json.exists():
        prior = load_json(existing_json)
        out = {
            int(r["layer"]): float(r["skill_vs_mean_ridge"])
            for r in prior.get("per_layer", [])
            if want is None or int(r["layer"]) in want
        }
        if out and (want is None or want.issubset(set(out))):
            logger.info(
                "[baseline] ridge skill from existing %s (%d layers)", existing_json, len(out)
            )
            return out
    logger.info("[baseline] recomputing ridge skill from store (no usable existing JSON)")
    C, V = _stack_layers(data)
    layers = data["layers"]
    out = {}
    for li in range(len(layers)):
        layer = int(layers[li])
        if want is not None and layer not in want:
            continue
        out[layer] = skill_over_mean_r2(
            ridge_predict_loco_centered(C[:, li, :], V[:, li, :]), V[:, li, :]
        )["skill"]
    return out


def _mlp_skill_for_width(
    C: np.ndarray,
    V: np.ndarray,
    li_iter: list[int],
    layers: list,
    *,
    hidden: int,
    device: str,
    num_threads: int | None,
    shuffle_layers: set[int] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Held-out skill-over-mean R² per layer at one MLP width (+ optional shuffle-null).

    Builds the batched multihead ensemble over ``li_iter`` at width ``hidden``,
    PCA-48 v0 target + input-PCA-projected c_C (the exact #722 recipe, only
    ``hidden`` varied), and returns (skill_by_layer, shuffle_skill_by_layer). The
    shuffle-null is computed ONLY for layers in ``shuffle_layers`` (the plateau
    layers) so a "positive" width can't be a fluke; layers not in the set get a
    NaN shuffle entry.
    """
    shuffle_layers = shuffle_layers or set()
    shuffle_perm = np.random.default_rng(SEED_722).permutation(C.shape[0])
    groups: list[MLPGroup] = []
    meta: dict[int, dict] = {}
    for li in li_iter:
        layer = int(layers[li])
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
        Xin, _ = _input_pca_project(Xc)
        Y64 = (Yv - mu_t) @ comps.T
        meta[li] = {"mu_t": mu_t, "comps": comps, "Yv": Yv}
        groups.append(MLPGroup(("base", li), Xin, Y64))
        if layer in shuffle_layers:
            Yv_sh = Yv[shuffle_perm]
            sh_mu, sh_comps, _ = robust_pca_basis(Yv_sh, MLP_PCA_DIM_722)
            Y64_sh = (Yv_sh - sh_mu) @ sh_comps.T
            meta[li]["sh"] = {"mu": sh_mu, "comps": sh_comps, "Yv_sh": Yv_sh}
            groups.append(MLPGroup(("shuffle", li), Xin, Y64_sh))

    res = fit_batched_loco_mlp_multihead(
        groups,
        seed=SEED_722,
        hidden=hidden,
        max_epochs=i658.MLP_MAX_EPOCHS,
        device=device,
        chunk_size=4096,
        num_threads=num_threads,
    )
    skill: dict[int, float] = {}
    shuffle_skill: dict[int, float] = {}
    for li in li_iter:
        layer = int(layers[li])
        m = meta[li]
        Y64 = (m["Yv"] - m["mu_t"]) @ m["comps"].T
        base_full = (loco_train_means(Y64) + res.preds_by_key[("base", li)]) @ m["comps"] + m[
            "mu_t"
        ]
        skill[layer] = skill_over_mean_r2(base_full, m["Yv"])["skill"]
        if "sh" in m:
            sh = m["sh"]
            Y64_sh = (sh["Yv_sh"] - sh["mu"]) @ sh["comps"].T
            sh_full = (loco_train_means(Y64_sh) + res.preds_by_key[("shuffle", li)]) @ sh[
                "comps"
            ] + sh["mu"]
            shuffle_skill[layer] = skill_over_mean_r2(sh_full, sh["Yv_sh"])["skill"]
        else:
            shuffle_skill[layer] = float("nan")
    return skill, shuffle_skill


def run_width_sweep(
    data: dict,
    *,
    widths: list[int],
    plateau_layers: list[int],
    device: str,
    num_threads: int | None,
    existing_json: Path,
    layer_subset: list[int] | None = None,
) -> dict:
    """Deliverable 1: held-out skill-over-mean R² per (width × layer), all 28 layers.

    For each width in ``widths`` fit the full layer battery (vectorized, only
    ``hidden`` varies) and record skill per layer. The shuffle-null is computed
    at the ``plateau_layers`` (the existing isolation control) so a positive
    width is checked against label-permutation. The LINEAR RIDGE skill is carried
    per layer too (the bar to beat). Capacity-vs-n question: does a smaller width
    avoid the overfit / approach (or beat) the ridge at n=50?
    """
    C, V = _stack_layers(data)
    layers = data["layers"]
    n, _L, H = V.shape
    li_iter = [
        li for li in range(len(layers)) if layer_subset is None or int(layers[li]) in layer_subset
    ]
    layer_nums = [int(layers[li]) for li in li_iter]
    plateau_in = sorted(set(plateau_layers) & set(layer_nums))
    ridge_by_layer = _ridge_baseline_by_layer(data, layer_subset, existing_json)

    per_width = []
    for w in widths:
        t0 = time.time()
        skill, shuffle_skill = _mlp_skill_for_width(
            C,
            V,
            li_iter,
            layers,
            hidden=w,
            device=device,
            num_threads=num_threads,
            shuffle_layers=set(plateau_in),
        )
        rows = []
        for layer in layer_nums:
            rows.append(
                {
                    "layer": layer,
                    "skill_vs_mean_mlp": skill[layer],
                    "skill_shuffle_mlp": shuffle_skill[layer],
                    "skill_vs_mean_ridge": ridge_by_layer.get(layer, float("nan")),
                }
            )
        per_width.append({"hidden": w, "n_folds": n, "per_layer": rows})
        # best layer for this width vs the ridge plateau
        best = max(rows, key=lambda r: r["skill_vs_mean_mlp"])
        logger.info(
            "[width-sweep] hidden=%4d fit in %.1fs — best MLP L%02d=%+.4f (ridge there=%+.4f)",
            w,
            time.time() - t0,
            best["layer"],
            best["skill_vs_mean_mlp"],
            best["skill_vs_mean_ridge"],
        )

    return {
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0)",
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "activation_dim": H,
        "layers": layer_nums,
        "widths": list(widths),
        "plateau_layers": plateau_in,
        "shuffle_null_layers": plateau_in,
        "mlp_recipe": {
            "lr": i658.MLP_LR,
            "wd": i658.MLP_WD,
            "max_epochs": i658.MLP_MAX_EPOCHS,
            "pca_target_dim": MLP_PCA_DIM_722,
            "input_pca_accel": True,
            "vectorized": True,
        },
        "seed": SEED_722,
        "ridge_skill_by_layer": {str(k): v for k, v in ridge_by_layer.items()},
        "store_provenance": {
            "store_dir": data["store_dir"],
            "e0_path": data["e0_path"],
            "cc_source": data["cc_source"],
            "n_contexts": n,
            "hidden_dim": int(H),
        },
        "per_width": per_width,
    }


def run_epoch_curves(
    data: dict,
    *,
    widths: list[int],
    layers_grid: list[int],
    eval_every: int,
    device: str,
    num_threads: int | None,
) -> dict:
    """Deliverable 2: per-epoch held-out skill-over-mean R² for a width × layer grid.

    For each (width, layer) in ``widths × layers_grid`` record the held-out
    skill every ``eval_every`` epochs across the 300-epoch fit (one cheap forward
    per snapshot). Answers: does held-out skill peak positive early then decay
    (→ early-stop rescues it) or stay negative throughout (→ n=50 too small)?
    """
    C, V = _stack_layers(data)
    layers = list(data["layers"])
    n = V.shape[0]
    cells = []
    for w in widths:
        for layer in layers_grid:
            if layer not in layers:
                logger.warning("[epoch-curve] layer %d not in store; skipping", layer)
                continue
            li = layers.index(layer)
            Xc = C[:, li, :]
            Yv = V[:, li, :]
            mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
            Xin, _ = _input_pca_project(Xc)
            Y64 = (Yv - mu_t) @ comps.T
            t0 = time.time()
            traj = fit_batched_loco_mlp_multihead_trajectory(
                [MLPGroup(("c", li), Xin, Y64)],
                eval_every=eval_every,
                seed=SEED_722,
                hidden=w,
                max_epochs=i658.MLP_MAX_EPOCHS,
                device=device,
                chunk_size=0,  # single (group, fold) battery → one chunk
                num_threads=num_threads,
            )
            curve = []
            for ep in traj.epochs:
                pred64 = traj.preds_at[ep][("c", li)]
                full = (loco_train_means(Y64) + pred64) @ comps + mu_t
                curve.append([int(ep), skill_over_mean_r2(full, Yv)["skill"]])
            peak = max(curve, key=lambda c: c[1]) if curve else [0, float("nan")]
            final = curve[-1] if curve else [0, float("nan")]
            cells.append(
                {
                    "hidden": w,
                    "layer": layer,
                    "n_folds": n,
                    "eval_every": eval_every,
                    "curve": curve,  # [[epoch, held_out_skill], ...]
                    "peak_epoch": peak[0],
                    "peak_skill": peak[1],
                    "final_epoch": final[0],
                    "final_skill": final[1],
                }
            )
            logger.info(
                "[epoch-curve] hidden=%4d L%02d fit in %.1fs — peak=%+.4f@ep%d final=%+.4f@ep%d",
                w,
                layer,
                time.time() - t0,
                peak[1],
                peak[0],
                final[1],
                final[0],
            )

    return {
        "metric": "skill_over_predict_the_mean per epoch (held-out R² on centered v0)",
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "widths": list(widths),
        "layers_grid": list(layers_grid),
        "eval_every": eval_every,
        "max_epochs": i658.MLP_MAX_EPOCHS,
        "mlp_recipe": {"lr": i658.MLP_LR, "wd": i658.MLP_WD, "pca_target_dim": MLP_PCA_DIM_722},
        "seed": SEED_722,
        "store_provenance": {
            "store_dir": data["store_dir"],
            "cc_source": data["cc_source"],
            "n_contexts": n,
        },
        "cells": cells,
    }


def run_krr_vs_linear(
    data: dict,
    *,
    width_sweep: dict | None,
    device: str,
    existing_json: Path,
    n_boot: int = 2000,
    layer_subset: list[int] | None = None,
) -> dict:
    """Coordinator scope: KRR (RBF + linear) vs linear ridge — the nonlinear-gap test.

    Per layer, fit KRR under the SAME LOCO CV + skill-over-mean R² metric:
      - RBF kernel: nested CV over (γ × λ) per held-out fold (no leakage), PCA-48
        v0 target centered by train mean.
      - linear kernel: nested CV over λ; reproduces the closed-form linear ridge
        skill (the plumbing sanity check, asserted at the plateau layers).
    The nonlinear-gap statistic is ``R²(KRR-RBF) − R²(linear ridge)`` with a
    LOCO-fold bootstrap CI (resample the 50 held-out fold contributions). The
    headline read at the ridge plateau: does the RBF gap CI exclude 0 (real
    nonlinearity) or include 0 (linear sufficient)?

    ``width_sweep`` (if provided) supplies the best-MLP-per-layer skill for the
    comparison figure. ``n_boot`` bootstrap resamples for the gap CI.
    """
    C, V = _stack_layers(data)
    layers = data["layers"]
    n, _L, _H = V.shape
    li_iter = [
        li for li in range(len(layers)) if layer_subset is None or int(layers[li]) in layer_subset
    ]
    layer_nums = [int(layers[li]) for li in li_iter]
    ridge_by_layer = _ridge_baseline_by_layer(data, layer_subset, existing_json)
    best_mlp_by_layer = _best_mlp_by_layer(width_sweep) if width_sweep else {}
    rng = np.random.default_rng(SEED_722)

    per_layer = []
    sanity_rows = []
    for li in li_iter:
        layer = int(layers[li])
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
        Y64 = (Yv - mu_t) @ comps.T  # (n, 48) PCA-reduced target

        # per-fold squared-error contributions, in PCA-48 space (the skill metric
        # is computed in that space — same as the MLP arm; ridge baseline JSON is
        # full-H, so we report the JSON ridge skill AND a PCA-48 linear-kernel
        # skill, asserting the linear-kernel KRR tracks the closed-form ridge).
        ridge_full_h = ridge_by_layer.get(layer, float("nan"))

        rbf_pred, rbf_lam, rbf_gam = krr_predict_loco(Xc, Y64, kernel="rbf")
        lin_pred, lin_lam, _ = krr_predict_loco(Xc, Y64, kernel="linear")

        rbf_skill, rbf_ssres, rbf_sstot = _skill_and_fold_terms(rbf_pred, Y64)
        lin_skill, lin_ssres, _lin_sstot = _skill_and_fold_terms(lin_pred, Y64)

        # gap statistic in the SAME PCA-48 space for both arms (the like-for-like
        # nonlinear gap): RBF − linear-kernel.
        gap = rbf_skill - lin_skill
        # LOCO-fold bootstrap: resample folds, recompute aggregate skill for each
        # arm, take the gap. ss_tot is shared (same target), so resample the
        # per-fold (ss_res_rbf, ss_res_lin, ss_tot) triples together.
        boot_gaps = _bootstrap_gap(rbf_ssres, lin_ssres, rbf_sstot, rng, n_boot)
        lo, hi = float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5))

        # Auditability (coordinator scope): full-H linear-ridge effective DoF +
        # a per-fold bootstrap CI on the linear-ridge skill. df_eff = Σ d_j²/(d_j²+λ)
        # quantifies how many directions ridge spends — the "n=50 is enough"
        # headline rests on df_eff ≪ 50. The CI reuses the LOCO fold predictions.
        ridge_audit = _ridge_audit(Xc, Yv, rng, n_boot)

        per_layer.append(
            {
                "layer": layer,
                "skill_vs_mean_ridge_fullH": ridge_full_h,
                "skill_krr_linear_pca48": lin_skill,
                "skill_krr_rbf_pca48": rbf_skill,
                "best_mlp_skill": best_mlp_by_layer.get(layer, float("nan")),
                "nonlinear_gap_rbf_minus_linear": gap,
                "gap_ci95": [lo, hi],
                "gap_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "chosen_lambda_rbf_median": float(np.median(rbf_lam)),
                "chosen_gamma_rbf_median": float(np.median(rbf_gam)),
                "chosen_lambda_linear_median": float(np.median(lin_lam)),
                # auditable n=50-linear headline:
                "ridge_df_eff": ridge_audit["df_eff"],
                "ridge_lambda_median": ridge_audit["lambda_median"],
                "ridge_skill_recomputed_fullH": ridge_audit["skill"],
                "ridge_skill_ci_lo": ridge_audit["ci_lo"],
                "ridge_skill_ci_hi": ridge_audit["ci_hi"],
            }
        )
        # plumbing sanity: linear-kernel KRR skill vs full-H closed-form ridge.
        # NOTE these are in DIFFERENT target spaces (PCA-48 vs full-H), so an
        # exact match is not expected layer-wide; the asserted check is the
        # linear-kernel-vs-PCA48-linear-ridge match in _krr_sanity_check.
        sanity_rows.append(
            {"layer": layer, "krr_linear_pca48": lin_skill, "ridge_fullH": ridge_full_h}
        )
        logger.info(
            "[krr] L%02d ridge(fullH)=%+.4f krr_lin=%+.4f krr_rbf=%+.4f gap=%+.4f "
            "CI=[%+.4f,%+.4f]%s",
            layer,
            ridge_full_h,
            lin_skill,
            rbf_skill,
            gap,
            lo,
            hi,
            " *EXCLUDES0*" if (lo > 0 or hi < 0) else "",
        )

    sanity = _krr_sanity_check(data, layer_subset, device=device)

    return {
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot "
        "(held-out R² on PCA-48 centered v0)",
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "layers": layer_nums,
        "kernels": ["linear", "rbf"],
        "pca_target_dim": MLP_PCA_DIM_722,
        "ridge_lambdas": list(i658.RIDGE_LAMBDAS),
        "n_bootstrap": n_boot,
        "seed": SEED_722,
        "krr_linear_vs_ridge_sanity": sanity,
        "store_provenance": {
            "store_dir": data["store_dir"],
            "cc_source": data["cc_source"],
            "n_contexts": n,
        },
        "per_layer": per_layer,
    }


def _skill_and_fold_terms(preds: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Aggregate skill + per-fold SS_res / SS_tot (for the bootstrap)."""
    n = Y.shape[0]
    total = Y.sum(axis=0, keepdims=True)
    tmean = (total - Y) / (n - 1)
    ss_res = np.sum((Y - preds) ** 2, axis=1)  # (n,) per-fold
    ss_tot = np.sum((Y - tmean) ** 2, axis=1)  # (n,) per-fold
    agg = float("nan") if ss_tot.sum() < 1e-12 else 1.0 - ss_res.sum() / ss_tot.sum()
    return agg, ss_res, ss_tot


def _bootstrap_gap(
    ss_res_a: np.ndarray, ss_res_b: np.ndarray, ss_tot: np.ndarray, rng, n_boot: int
) -> np.ndarray:
    """Bootstrap the skill gap (arm A − arm B) by resampling LOCO folds with replacement."""
    n = ss_tot.shape[0]
    gaps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tot = ss_tot[idx].sum()
        if tot < 1e-12:
            gaps[b] = float("nan")
            continue
        skill_a = 1.0 - ss_res_a[idx].sum() / tot
        skill_b = 1.0 - ss_res_b[idx].sum() / tot
        gaps[b] = skill_a - skill_b
    return gaps[np.isfinite(gaps)]


def _ridge_audit(Xc: np.ndarray, Yv: np.ndarray, rng, n_boot: int) -> dict:
    """Full-H linear-ridge effective DoF + per-fold bootstrap CI on the ridge skill.

    Makes the n=50 LINEAR headline auditable:
      - ``df_eff = Σ_j d_j²/(d_j²+λ)`` (d_j = singular values of the FULL-data
        standardized design X; λ = the median nested-CV-chosen ridge λ over the
        LOCO folds). df_eff ≪ 50 is what "n=50 is enough to fit the linear map"
        actually rests on.
      - a bootstrap CI on the linear-ridge skill-over-mean R², resampling the 50
        held-out LOCO fold contributions (same machinery as the KRR gap CI). The
        recomputed skill matches the closed-form ridge skill in the canonical
        JSON to numerical noise (both are the same LOCO ridge in full-H).
    """
    n = Xc.shape[0]
    # full-H LOCO ridge predictions (reuse the canonical centered-LOCO solver),
    # recording the per-fold chosen λ for the df_eff anchor.
    pred = ridge_predict_loco_centered(Xc, Yv)
    skill, ss_res, ss_tot = _skill_and_fold_terms(pred, Yv)
    # per-fold λ picks → median λ anchors df_eff (the full-data PRESS λ; per-fold
    # picks are near-identical at this n, so the median is the representative).
    lams = _ridge_loo_lambdas(Xc, Yv)
    lam = float(np.median(lams))
    # SVD singular values of the FULL-data standardized design.
    X = np.ascontiguousarray(Xc.astype(np.float64))
    xmu = X.mean(0)
    xsd = X.std(0, ddof=0) + 1e-9
    Xn = (X - xmu) / xsd
    d = np.linalg.svd(Xn, compute_uv=False)  # singular values
    d2 = d * d
    df_eff = float(np.sum(d2 / (d2 + lam)))
    # bootstrap CI on the skill (resample folds with replacement).
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tot = ss_tot[idx].sum()
        boots[b] = float("nan") if tot < 1e-12 else 1.0 - ss_res[idx].sum() / tot
    boots = boots[np.isfinite(boots)]
    return {
        "df_eff": df_eff,
        "lambda_median": lam,
        "skill": skill,
        "ci_lo": float(np.percentile(boots, 2.5)) if boots.size else float("nan"),
        "ci_hi": float(np.percentile(boots, 97.5)) if boots.size else float("nan"),
    }


def _ridge_loo_lambdas(Xc: np.ndarray, Yv: np.ndarray) -> list:
    """Per-fold nested-PRESS-chosen ridge λ over the LOCO folds (for df_eff anchor)."""
    n = Xc.shape[0]
    device = torch.device(i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    out = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        tr_t = torch.tensor(tr, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - xmu) / xsd
        Ytr_c = Ytr - Ytr.mean(0)
        mse = i658._press_loo_mse_per_lambda(Xtr_n, Ytr_c, i658.RIDGE_LAMBDAS)
        out.append(float(i658.RIDGE_LAMBDAS[int(torch.argmin(mse).item())]))
    return out


def _krr_sanity_check(data: dict, layer_subset: list[int] | None, *, device: str) -> dict:
    """Assert linear-kernel KRR reproduces the PCA-48 linear-ridge skill (plumbing).

    Both arms fit the SAME PCA-48 v0 target with the SAME LOCO CV; a linear
    kernel KRR == ridge in feature space, so the two skills must match to a few
    e-3 at the sanity layers. A mismatch means the KRR plumbing (centering,
    nested PRESS, dual solve) is wrong. Checks the plateau layers (or the
    subset's layers when restricted).
    """
    C, V = _stack_layers(data)
    layers = list(data["layers"])
    sanity_layers = [layer_subset[0]] if layer_subset else [li for li in (14, 18) if li in layers]
    sanity_layers = [layer for layer in sanity_layers if layer in layers]
    rows = []
    ok = True
    tol = 5e-3
    for layer in sanity_layers:
        li = layers.index(layer)
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
        Y64 = (Yv - mu_t) @ comps.T
        ridge_pca48 = skill_over_mean_r2(ridge_predict_loco_centered(Xc, Y64), Y64)["skill"]
        lin_pred, _, _ = krr_predict_loco(Xc, Y64, kernel="linear")
        krr_lin_pca48 = skill_over_mean_r2(lin_pred, Y64)["skill"]
        d = abs(ridge_pca48 - krr_lin_pca48)
        rows.append(
            {
                "layer": layer,
                "ridge_pca48": ridge_pca48,
                "krr_linear_pca48": krr_lin_pca48,
                "abs_delta": d,
            }
        )
        if d > tol:
            ok = False
        logger.info(
            "[krr-sanity] L%02d ridge(pca48)=%+.5f krr_lin(pca48)=%+.5f |Δ|=%.2e",
            layer,
            ridge_pca48,
            krr_lin_pca48,
            d,
        )
    return {"ok": ok, "tol": tol, "rows": rows}


# ── #722 input-representation robustness (round-2 amendment) ──────────────────


def _baseline_ridge_by_layer(baseline_skill_json: Path, layer_subset: list[int] | None) -> dict:
    """Per-layer variant-1 (full) ridge skill from the committed skill_over_mean.json."""
    prior = load_json(baseline_skill_json)
    want = None if layer_subset is None else set(layer_subset)
    return {
        int(r["layer"]): float(r["skill_vs_mean_ridge"])
        for r in prior.get("per_layer", [])
        if want is None or int(r["layer"]) in want
    }


def _baseline_gap_by_layer(baseline_krr_json: Path, layer_subset: list[int] | None) -> dict:
    """Per-layer variant-1 (full) KRR(RBF)−linear gap from the committed krr_vs_linear.json."""
    prior = load_json(baseline_krr_json)
    want = None if layer_subset is None else set(layer_subset)
    return {
        int(r["layer"]): float(r["nonlinear_gap_rbf_minus_linear"])
        for r in prior.get("per_layer", [])
        if want is None or int(r["layer"]) in want
    }


def run_input_rep_robustness(
    data: dict,
    *,
    input_rep: str,
    device: str,
    baseline_skill_json: Path,
    baseline_krr_json: Path,
    n_boot: int = 2000,
    k: int = INPUT_REP_K,
    eps: float = INPUT_REP_EPS,
    layer_subset: list[int] | None = None,
) -> tuple[dict, dict]:
    """Per-layer ridge skill + KRR(RBF)−linear gap under one INPUT representation.

    For ``input_rep`` ∈ {``pca48``, ``whiten48``}: re-run BOTH headline arms with the
    c_C input re-represented PER LOCO FOLD (TRAIN-only top-k PCA / ZCA-whiten, the
    held-out row projected through the TRAIN basis — no leakage), then compare each
    cell against the committed variant-1 (full) baseline JSONs. The v0 TARGET
    reduction (top-48 PC), the ridge λ grid, the RBF γ heuristic + KRR λ grid, the
    LOCO fold structure, the skill-over-mean metric, the 2000-resample bootstrap, the
    seed, n=50 and the 28 layers are ALL inherited verbatim — the ONLY change is the
    input representation. ``input_rep="full"`` is rejected (it is the committed
    baseline, never re-run; see the amendment plan §3).

    Returns ``(skill_result, krr_result)`` — two JSON-ready dicts matching the
    baseline schema plus the per-layer baseline value + delta and a top-level
    ``input_rep`` field.
    """
    if input_rep == "full":
        raise ValueError("input_rep='full' is the committed baseline; never re-run it")
    if input_rep not in INPUT_REPS:
        raise ValueError(f"unknown input_rep {input_rep!r}; expected one of {INPUT_REPS}")

    C, V = _stack_layers(data)
    layers = data["layers"]
    n, _L, H = V.shape
    li_iter = [
        li for li in range(len(layers)) if layer_subset is None or int(layers[li]) in layer_subset
    ]
    base_ridge = _baseline_ridge_by_layer(baseline_skill_json, layer_subset)
    base_gap = _baseline_gap_by_layer(baseline_krr_json, layer_subset)
    rng = np.random.default_rng(SEED_722)
    logger.info(
        "[input-rep=%s] n=%d L=%d H=%d k=%d eps=%.1e — re-running ridge + KRR per fold",
        input_rep,
        n,
        len(li_iter),
        H,
        k,
        eps,
    )

    ridge_rows = []
    krr_rows = []
    for li in li_iter:
        layer = int(layers[li])
        Xc = C[:, li, :]
        Yv = V[:, li, :]

        # ── ridge arm (full-H target, the #722 ridge_mean recipe, transformed input) ──
        ridge_pred, ridge_fb = ridge_predict_loco_centered_rep(
            Xc, Yv, input_rep=input_rep, k=k, eps=eps
        )
        ridge_skill = skill_over_mean_r2(ridge_pred, Yv)
        base_r = base_ridge.get(layer, float("nan"))
        d_ridge = ridge_skill["skill"] - base_r
        # §4.4 lambda_chosen: the full-data PRESS λ pick UNDER this input rep — the
        # transformed-input analogue of the baseline skill JSON's per-layer
        # lambda_chosen diagnostic (BLOCKER input-rep-skill-schema-missing-lambda).
        lambda_chosen = _full_data_lambda_rep(Xc, Yv, input_rep, k, eps)
        ridge_rows.append(
            {
                "layer": layer,
                "skill_vs_mean_ridge": ridge_skill["skill"],
                "skill_vs_mean_ridge_baseline_full": base_r,
                "delta_ridge": d_ridge,
                "predict_mean_abs_cos": _predict_mean_abs_cos(Yv),
                "raw_recon_abs_cos": _recon_abs_cos(ridge_pred, Yv),
                "ridge_median_per_dim_r2": ridge_skill["median_per_dim_r2"],
                "lambda_chosen": lambda_chosen,
                "n_folds_used_ridge": ridge_skill["n_folds_used"],
                "gesvd_fallback": bool(ridge_fb),
            }
        )

        # ── KRR arm (PCA-48 target, RBF + linear, transformed input) ──
        mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
        Y64 = (Yv - mu_t) @ comps.T
        rbf_pred, rbf_lam, rbf_gam, rbf_fb = krr_predict_loco_rep(
            Xc, Y64, kernel="rbf", input_rep=input_rep, k=k, eps=eps
        )
        lin_pred, lin_lam, _, lin_fb = krr_predict_loco_rep(
            Xc, Y64, kernel="linear", input_rep=input_rep, k=k, eps=eps
        )
        rbf_skill, rbf_ssres, rbf_sstot = _skill_and_fold_terms(rbf_pred, Y64)
        lin_skill, lin_ssres, _ = _skill_and_fold_terms(lin_pred, Y64)
        gap = rbf_skill - lin_skill
        boot_gaps = _bootstrap_gap(rbf_ssres, lin_ssres, rbf_sstot, rng, n_boot)
        lo, hi = float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5))
        base_g = base_gap.get(layer, float("nan"))
        d_gap = gap - base_g
        krr_rows.append(
            {
                "layer": layer,
                "skill_krr_linear_pca48": lin_skill,
                "skill_krr_rbf_pca48": rbf_skill,
                "nonlinear_gap_rbf_minus_linear": gap,
                "nonlinear_gap_baseline_full": base_g,
                "delta_gap": d_gap,
                "gap_ci95": [lo, hi],
                "gap_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "chosen_gamma_rbf_median": float(np.nanmedian(rbf_gam)),
                "chosen_lambda_rbf_median": float(np.nanmedian(rbf_lam)),
                "chosen_lambda_linear_median": float(np.nanmedian(lin_lam)),
                "gesvd_fallback": bool(rbf_fb or lin_fb),
            }
        )
        logger.info(
            "[input-rep=%s][L%02d] ridge=%+.4f (Δ%+.4f) gap=%+.4f (Δ%+.4f) CI=[%+.4f,%+.4f]",
            input_rep,
            layer,
            ridge_skill["skill"],
            d_ridge,
            gap,
            d_gap,
            lo,
            hi,
        )

    store_prov = {
        "store_dir": data["store_dir"],
        "cc_source": data["cc_source"],
        "n_contexts": n,
        "hidden_dim": int(H),
    }
    common = {
        "input_rep": input_rep,
        "input_rep_k": k,
        "input_rep_eps": eps,
        "c_C_recipe": "C_last",
        "n_contexts": n,
        "layers": [int(layers[li]) for li in li_iter],
        "seed": SEED_722,
        "store_provenance": store_prov,
        "baseline_skill_json": str(baseline_skill_json),
        "baseline_krr_json": str(baseline_krr_json),
    }
    skill_result = {
        **common,
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on centered v0)",
        "activation_dim": H,
        "ridge_lambdas": i658.RIDGE_LAMBDAS,
        "per_layer": ridge_rows,
    }
    krr_result = {
        **common,
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot "
        "(held-out R² on PCA-48 centered v0)",
        "kernels": ["linear", "rbf"],
        "pca_target_dim": MLP_PCA_DIM_722,
        "ridge_lambdas": list(i658.RIDGE_LAMBDAS),
        "n_bootstrap": n_boot,
        "per_layer": krr_rows,
    }
    return skill_result, krr_result


def run_gamma_sensitivity(
    data: dict,
    *,
    input_rep: str,
    layers_to_probe: tuple[int, ...] = GAMMA_SENS_LAYERS,
    multipliers: tuple[float, ...] = GAMMA_SENS_MULTIPLIERS,
    k: int = INPUT_REP_K,
    eps: float = INPUT_REP_EPS,
) -> dict:
    """γ-sensitivity diagnostic for the KRR RBF collapse (CONCERN, plan §4.4 band).

    At each probe layer, compute the linear-kernel LOCO R² ONCE (γ-independent), then
    for each γ multiplier re-run ONLY the RBF arm forcing the per-fold γ grid to the
    single point ``multiplier × γ₀_fold`` (``krr_predict_loco_rep(..., gamma_scale=m)``)
    and report RBF R² + the RBF−linear gap. Everything else — the top-48-PC v0 target,
    the LOCO fold structure, the λ grid, the per-fold train-only pca48 transform, the
    seed, n=50 — is inherited verbatim from ``run_input_rep_robustness``; the ONLY
    thing swept is γ. If NO multiplier lifts the gap toward ~0, the collapse is
    γ-robust (a real "RBF buys nothing at 48 standardized dims"); if some multiplier
    recovers the gap, the median heuristic sat in a bad regime at 48-d.

    Returns a JSON-ready dict: ``{input_rep, multipliers, per_layer: [{layer,
    skill_krr_linear, gamma0_median, by_multiplier: [{multiplier, chosen_gamma_median,
    skill_krr_rbf, gap_rbf_minus_linear}, ...], best_gap, best_multiplier}, ...]}``.
    """
    if input_rep == "full":
        raise ValueError("gamma-sensitivity is a transformed-input (pca48) diagnostic, not 'full'")
    C, V = _stack_layers(data)
    layers = data["layers"]
    layer_to_li = {int(layers[li]): li for li in range(len(layers))}
    per_layer = []
    for layer in layers_to_probe:
        if layer not in layer_to_li:
            logger.warning("[gamma-sens] layer %d not in store — skipping", layer)
            continue
        li = layer_to_li[layer]
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
        Y64 = (Yv - mu_t) @ comps.T
        # linear arm ONCE (γ-independent): the gap denominator at this layer.
        lin_pred, _lin_lam, _lin_gam, _lin_fb = krr_predict_loco_rep(
            Xc, Y64, kernel="linear", input_rep=input_rep, k=k, eps=eps
        )
        lin_skill, _lin_ssres, _ = _skill_and_fold_terms(lin_pred, Y64)
        by_mult = []
        for m in multipliers:
            rbf_pred, _rbf_lam, rbf_gam, _rbf_fb = krr_predict_loco_rep(
                Xc, Y64, kernel="rbf", input_rep=input_rep, k=k, eps=eps, gamma_scale=m
            )
            rbf_skill, _rbf_ssres, _ = _skill_and_fold_terms(rbf_pred, Y64)
            gap = rbf_skill - lin_skill
            by_mult.append(
                {
                    "multiplier": float(m),
                    "chosen_gamma_median": float(np.nanmedian(rbf_gam)),
                    "skill_krr_rbf": rbf_skill,
                    "gap_rbf_minus_linear": gap,
                }
            )
            logger.info(
                "[gamma-sens][%s][L%02d] m=%.2f γ=%.3g rbf=%+.4f gap=%+.4f",
                input_rep,
                layer,
                m,
                by_mult[-1]["chosen_gamma_median"],
                rbf_skill,
                gap,
            )
        # γ₀ median (the multiplier=1.0 heuristic point) for reference.
        gamma0_median = next(
            (r["chosen_gamma_median"] for r in by_mult if abs(r["multiplier"] - 1.0) < 1e-9),
            float("nan"),
        )
        # "best" = the multiplier whose gap is CLOSEST to 0 from below (least negative)
        # — i.e. the γ that most recovers RBF toward linear. If all gaps are negative
        # the collapse is γ-robust across the sweep.
        best = max(by_mult, key=lambda r: r["gap_rbf_minus_linear"])
        per_layer.append(
            {
                "layer": layer,
                "skill_krr_linear": lin_skill,
                "gamma0_median": gamma0_median,
                "by_multiplier": by_mult,
                "best_gap": best["gap_rbf_minus_linear"],
                "best_multiplier": best["multiplier"],
            }
        )
    return {
        "diagnostic": "gamma_sensitivity",
        "concern_id": "krr-gap-collapse-gamma-regime-interpretation",
        "input_rep": input_rep,
        "layers_probed": [int(x) for x in layers_to_probe],
        "multipliers": [float(m) for m in multipliers],
        "pca_target_dim": MLP_PCA_DIM_722,
        "input_rep_k": k,
        "input_rep_eps": eps,
        "seed": SEED_722,
        "metric": "skill_over_predict_the_mean = 1 - SS_res/SS_tot (held-out R² on PCA-48 v0)",
        "note": "gap = RBF R² − linear R² at γ = multiplier × per-fold median-heuristic γ₀. "
        "All-negative best_gap across multipliers ⇒ collapse is γ-robust (RBF buys nothing "
        "at 48 standardized dims); a near-zero/positive recovered gap ⇒ heuristic-γ regime "
        "artifact.",
        "store_provenance": {
            "store_dir": data["store_dir"],
            "cc_source": data["cc_source"],
        },
        "per_layer": per_layer,
    }


def _verdict_for_variant(skill_result: dict, krr_result: dict) -> dict:
    """Per-variant SUCCESS/KILL verdict against the §6 bands (≥26/28 layers).

    SUCCESS = ≥26 of the layers have |Δridge R²| ≤ 0.05 AND the KRR(RBF)−linear gap
    SIGN matches the baseline with |Δgap| ≤ 0.03. KILL otherwise (the input
    representation moved the headline). Layers with a NaN baseline (missing from the
    baseline JSON) are counted as failing the corresponding gate (conservative).
    """
    ridge_by_layer = {r["layer"]: r for r in skill_result["per_layer"]}
    n_ridge_pass = 0
    n_gap_pass = 0
    n_layers = 0
    n_sign_exception = 0  # layers passing gap gate ONLY via the near-zero-band relaxation
    for kr in krr_result["per_layer"]:
        layer = kr["layer"]
        n_layers += 1
        rr = ridge_by_layer.get(layer, {})
        d_ridge = rr.get("delta_ridge", float("nan"))
        ridge_ok = np.isfinite(d_ridge) and abs(d_ridge) <= INPUT_REP_RIDGE_BAND
        if ridge_ok:
            n_ridge_pass += 1
        gap = kr["nonlinear_gap_rbf_minus_linear"]
        base_g = kr["nonlinear_gap_baseline_full"]
        d_gap = kr["delta_gap"]
        # Sign preservation: strict same-sign, OR the documented near-zero exception
        # (BOTH the variant and baseline gap within ±gap_band of zero). At the ridge
        # plateau the gap ≈ 0, so two noisy near-zero quantities can straddle zero
        # with opposite signs while being statistically indistinguishable; demanding
        # strict sign-match there would spuriously FAIL a robust cell. The exception
        # is recorded explicitly (n_layers_sign_exception + sign_preservation_rule)
        # so the relaxation is auditable, not silent (Codex round-2 Minor).
        strict_sign = np.sign(gap) == np.sign(base_g)
        near_zero_both = abs(gap) <= INPUT_REP_GAP_BAND and abs(base_g) <= INPUT_REP_GAP_BAND
        sign_ok = strict_sign or near_zero_both
        gap_ok = np.isfinite(d_gap) and abs(d_gap) <= INPUT_REP_GAP_BAND and sign_ok
        if gap_ok:
            n_gap_pass += 1
            if not strict_sign and near_zero_both:
                n_sign_exception += 1
    success = n_ridge_pass >= INPUT_REP_PASS_MIN_LAYERS and n_gap_pass >= INPUT_REP_PASS_MIN_LAYERS
    return {
        "n_layers": n_layers,
        "n_layers_passing_R2_gate": n_ridge_pass,
        "n_layers_passing_gap_gate": n_gap_pass,
        "n_layers_sign_exception": n_sign_exception,
        "sign_preservation_rule": (
            "strict same-sign OR both |gap| <= gap_band (near-zero-plateau exception); "
            "n_layers_sign_exception counts gap-gate passes that used the exception"
        ),
        "pass_min_layers": INPUT_REP_PASS_MIN_LAYERS,
        "ridge_band": INPUT_REP_RIDGE_BAND,
        "gap_band": INPUT_REP_GAP_BAND,
        "verdict": "SUCCESS" if success else "KILL",
    }


def build_input_rep_comparison(results_by_variant: dict) -> dict:
    """Assemble the headline comparison.json across the input-rep variants.

    ``results_by_variant`` maps variant name → ``(skill_result, krr_result)``.
    Emits per-variant verdicts + the flat ``verdict_<variant>`` /
    ``n_layers_passing_*_gate_<variant>`` keys the amendment §4.4 names, plus the
    per-layer Δridge / Δgap arrays for each variant.
    """
    out: dict = {
        "ridge_band": INPUT_REP_RIDGE_BAND,
        "gap_band": INPUT_REP_GAP_BAND,
        "pass_min_layers": INPUT_REP_PASS_MIN_LAYERS,
        "criterion": (
            f">={INPUT_REP_PASS_MIN_LAYERS}/28 layers with |Δridge R²|<={INPUT_REP_RIDGE_BAND} "
            f"AND gap-sign-preserved with |Δgap|<={INPUT_REP_GAP_BAND}"
        ),
        "per_variant": {},
    }
    for variant, (skill_result, krr_result) in results_by_variant.items():
        v = _verdict_for_variant(skill_result, krr_result)
        sr_by_layer = {r["layer"]: r for r in skill_result["per_layer"]}
        out["per_variant"][variant] = {
            **v,
            "per_layer": [
                {
                    "layer": kr["layer"],
                    "delta_ridge": sr_by_layer[kr["layer"]]["delta_ridge"],
                    "delta_gap": kr["delta_gap"],
                    "gap": kr["nonlinear_gap_rbf_minus_linear"],
                    "gap_baseline_full": kr["nonlinear_gap_baseline_full"],
                }
                for kr in krr_result["per_layer"]
            ],
        }
        out[f"verdict_{variant}"] = v["verdict"]
        out[f"n_layers_passing_R2_gate_{variant}"] = v["n_layers_passing_R2_gate"]
        out[f"n_layers_passing_gap_gate_{variant}"] = v["n_layers_passing_gap_gate"]
    return out


def _best_mlp_by_layer(width_sweep: dict) -> dict[int, float]:
    """Best (max-over-width) MLP skill per layer from a width-sweep result."""
    best: dict[int, float] = {}
    for w in width_sweep.get("per_width", []):
        for r in w["per_layer"]:
            layer = int(r["layer"])
            s = r["skill_vs_mean_mlp"]
            if s == s and (layer not in best or s > best[layer]):  # NaN-safe max
                best[layer] = s
    return best


def make_width_sweep_figure(result: dict, fig_path: Path) -> None:
    """Deliverable 3a: held-out skill vs MLP width (log-x), one line per plateau layer.

    Ridge baseline as a horizontal reference line + a y=0 line (skill 0 = no
    better than predict-the-mean). Caption: does any width get positive / beat
    ridge?
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    widths = result["widths"]
    layer_nums = result["layers"]
    # representative layers: the plateau set + L0 negative control if present.
    rep = sorted(set(result.get("plateau_layers", [])) | ({0} & set(layer_nums)))
    if not rep:
        rep = layer_nums[:: max(1, len(layer_nums) // 4)]
    skill_by_wl = {
        w["hidden"]: {r["layer"]: r["skill_vs_mean_mlp"] for r in w["per_layer"]}
        for w in result["per_width"]
    }
    ridge_by_layer = {int(k): v for k, v in result.get("ridge_skill_by_layer", {}).items()}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for i, layer in enumerate(rep):
        ys = [skill_by_wl[w].get(layer, float("nan")) for w in widths]
        ax.plot(
            widths,
            ys,
            marker="o",
            ms=4,
            lw=1.6,
            color=colors[i % len(colors)],
            label=f"MLP L{layer}",
        )
        rb = ridge_by_layer.get(layer)
        if rb is not None and rb == rb:
            ax.axhline(rb, color=colors[i % len(colors)], lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0.0, color="0.4", lw=0.9, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("MLP hidden width")
    ax.set_ylabel("skill-over-mean (held-out R²)")
    ax.legend(loc="best", fontsize=7, title="solid=MLP, dashed=ridge (same layer)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_fig_meta(fig_path, 722, "mlp_width_sweep.json")
    logger.info("wrote %s", fig_path)


def make_epoch_curve_figure(result: dict, fig_path: Path) -> None:
    """Deliverable 3b: held-out skill vs epoch, one line per (width, layer) cell."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]
    for i, cell in enumerate(result["cells"]):
        curve = cell["curve"]
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        ax.plot(
            xs,
            ys,
            marker=".",
            ms=3,
            lw=1.4,
            color=colors[i % len(colors)],
            label=f"w{cell['hidden']} L{cell['layer']}",
        )
    ax.axhline(0.0, color="0.4", lw=0.9, ls=":")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("skill-over-mean (held-out R²)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_fig_meta(fig_path, 722, "mlp_epoch_curves.json")
    logger.info("wrote %s", fig_path)


def make_krr_figure(result: dict, fig_path: Path) -> None:
    """Coordinator scope: per-layer linear-ridge vs KRR-RBF vs best-MLP + gap CI subplot."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")
    rows = sorted(result["per_layer"], key=lambda r: r["layer"])
    x = [r["layer"] for r in rows]
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    ax_top.plot(
        x,
        [r["skill_vs_mean_ridge_fullH"] for r in rows],
        marker="o",
        ms=3,
        lw=1.6,
        color="#0072B2",
        label="linear ridge (full-H)",
    )
    ax_top.plot(
        x,
        [r["skill_krr_rbf_pca48"] for r in rows],
        marker="s",
        ms=3,
        lw=1.6,
        color="#D55E00",
        label="KRR-RBF (PCA-48)",
    )
    ax_top.plot(
        x,
        [r["skill_krr_linear_pca48"] for r in rows],
        marker="^",
        ms=3,
        lw=1.2,
        ls="--",
        color="#56B4E9",
        label="KRR-linear (PCA-48, sanity)",
    )
    if any(r["best_mlp_skill"] == r["best_mlp_skill"] for r in rows):
        ax_top.plot(
            x,
            [r["best_mlp_skill"] for r in rows],
            marker="x",
            ms=3,
            lw=1.2,
            ls=":",
            color="#999999",
            label="best-MLP (width sweep)",
        )
    ax_top.axhline(0.0, color="0.4", lw=0.9, ls=":")
    ax_top.set_ylabel("skill-over-mean (held-out R²)")
    ax_top.legend(loc="best", fontsize=7)

    gap = np.array([r["nonlinear_gap_rbf_minus_linear"] for r in rows])
    lo = np.array([r["gap_ci95"][0] for r in rows])
    hi = np.array([r["gap_ci95"][1] for r in rows])
    yerr = np.vstack([gap - lo, hi - gap])
    yerr = np.clip(yerr, 0.0, None)  # guard float-eps negatives
    ax_bot.errorbar(
        x, gap, yerr=yerr, fmt="o", ms=3, lw=1.0, color="#D55E00", ecolor="#D55E00", capsize=2
    )
    ax_bot.axhline(0.0, color="0.4", lw=0.9, ls=":")
    ax_bot.set_xlabel("layer")
    ax_bot.set_ylabel("nonlinear gap\n(RBF − linear), 95% CI")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_fig_meta(fig_path, 722, "krr_vs_linear.json")
    logger.info("wrote %s", fig_path)


def make_input_rep_figure(
    results_by_variant: dict,
    *,
    baseline_skill_json: Path,
    baseline_krr_json: Path,
    layer_subset: list[int] | None,
    fig_path: Path,
) -> None:
    """Plan v7 §6 Figure: two stacked panels comparing input representations per layer.

    Top panel: linear-ridge skill-over-mean R² for the full baseline + each
    re-represented variant (pca48 / whiten48), x = layer. Bottom panel: the
    KRR(RBF)−linear gap for the same reps, with a horizontal zero line. The full
    baseline lines are read from the committed ``skill_over_mean.json`` /
    ``krr_vs_linear.json``; the variant lines come from the in-memory
    ``results_by_variant`` dicts ((skill_result, krr_result) per variant). No
    annotation overlays. Restricted to ``layer_subset`` when given (a 2-layer
    smoke figure is degenerate but proves the writer runs).
    """
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style(target="neurips")

    want = None if layer_subset is None else set(layer_subset)

    def _series(per_layer: list[dict], key: str) -> tuple[list[int], list[float]]:
        rows = sorted(
            (r for r in per_layer if want is None or int(r["layer"]) in want),
            key=lambda r: int(r["layer"]),
        )
        return [int(r["layer"]) for r in rows], [float(r[key]) for r in rows]

    base_skill = load_json(baseline_skill_json)
    base_krr = load_json(baseline_krr_json)

    colors = {"full": "#0072B2", "pca48": "#D55E00", "whiten48": "#009E73"}
    markers = {"full": "o", "pca48": "s", "whiten48": "^"}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)

    # ── top panel: ridge skill-over-mean R² ──
    bx, by = _series(base_skill["per_layer"], "skill_vs_mean_ridge")
    ax_top.plot(
        bx, by, marker=markers["full"], ms=3, lw=1.6, color=colors["full"], label="full (baseline)"
    )
    # ── bottom panel: KRR(RBF)−linear gap ──
    gx, gy = _series(base_krr["per_layer"], "nonlinear_gap_rbf_minus_linear")
    ax_bot.plot(gx, gy, marker=markers["full"], ms=3, lw=1.6, color=colors["full"], label="full")

    for variant, (skill_result, krr_result) in results_by_variant.items():
        c = colors.get(variant, "#999999")
        m = markers.get(variant, "x")
        vx, vy = _series(skill_result["per_layer"], "skill_vs_mean_ridge")
        ax_top.plot(vx, vy, marker=m, ms=3, lw=1.6, color=c, label=variant)
        kx, ky = _series(krr_result["per_layer"], "nonlinear_gap_rbf_minus_linear")
        ax_bot.plot(kx, ky, marker=m, ms=3, lw=1.6, color=c, label=variant)

    ax_top.axhline(0.0, color="0.4", lw=0.9, ls=":")
    ax_top.set_ylabel("skill-over-mean (held-out R²)")
    ax_top.legend(loc="best", fontsize=7)

    ax_bot.axhline(0.0, color="0.4", lw=0.9, ls=":")
    ax_bot.set_xlabel("layer")
    ax_bot.set_ylabel("nonlinear gap\n(RBF − linear)")
    ax_bot.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _write_fig_meta(fig_path, 722, "input_rep_comparison.json")
    logger.info("wrote %s", fig_path)


# ── slow-MLP spot-check (Deliverable 4 item 2) ────────────────────────────────


def spot_check_vs_slow_mlp(data: dict, spot_layers: list[int], tol: float = 5e-3) -> dict:
    """Run the OLD slow issue722 MLP on 2-3 layers + compare to the vectorized skill.

    Imports the slow ``issue722_skill_over_mean`` from the #722 worktree (it is
    uncommitted to main) and runs its ``_mlp_skill`` (PCA-48 target + input-PCA)
    on the spot layers ONLY (restricting the slow path so it finishes in minutes),
    then compares the per-layer ``skill_vs_mean_mlp`` to this module's vectorized
    value. Returns the per-layer deltas + a pass flag.
    """
    slow_dir = REPO_ROOT / ".claude" / "worktrees" / "issue-722" / "scripts"
    if not (slow_dir / "issue722_skill_over_mean.py").exists():
        return {"ran": False, "reason": f"slow script not found at {slow_dir}"}
    sys.path.insert(0, str(slow_dir))
    import importlib

    slow = importlib.import_module("issue722_skill_over_mean")
    slow.i658.DEVICE = "cpu"

    C, V = _stack_layers(data)
    layers = list(data["layers"])
    rows = []
    ok = True
    for layer in spot_layers:
        li = layers.index(layer)
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        slow_mlp = slow._mlp_skill(Xc, Yv)["skill"]
        mu_t, comps, _ = robust_pca_basis(Yv, MLP_PCA_DIM_722)
        Xin, _ = _input_pca_project(Xc)
        Y64 = (Yv - mu_t) @ comps.T
        # (a) vectorized SCALAR-PER-DIM — same architecture as the slow path; must
        # match it to ``tol`` (the exactness reproduce-leg).
        res_sd = fit_batched_loco_mlp(
            [MLPGroup(("m", li), Xin, Y64)], seed=SEED_722, device="cpu", chunk_size=0
        )
        sd_full = (loco_train_means(Y64) + res_sd.preds_by_key[("m", li)]) @ comps + mu_t
        vec_scalar = skill_over_mean_r2(sd_full, Yv)["skill"]
        # (b) vectorized MULTIHEAD — the PRODUCTION path; report its gap to the
        # slow scalar path (NOT gated — a different architecture by design).
        res_mh = fit_batched_loco_mlp_multihead(
            [MLPGroup(("m", li), Xin, Y64)], seed=SEED_722, device="cpu", chunk_size=0
        )
        mh_full = (loco_train_means(Y64) + res_mh.preds_by_key[("m", li)]) @ comps + mu_t
        vec_multihead = skill_over_mean_r2(mh_full, Yv)["skill"]
        d_scalar = abs(slow_mlp - vec_scalar)
        d_mh = abs(slow_mlp - vec_multihead)
        rows.append(
            {
                "layer": layer,
                "slow_scalar_mlp_skill": slow_mlp,
                "vec_scalar_mlp_skill": vec_scalar,
                "vec_multihead_mlp_skill": vec_multihead,
                "abs_delta_scalar": d_scalar,
                "abs_delta_multihead": d_mh,
            }
        )
        if d_scalar > tol:
            ok = False
        logger.info(
            "[spot L%02d] slow=%+.5f vec_scalar=%+.5f (|Δ|=%.2e) vec_multihead=%+.5f (|Δ|=%.2e)",
            layer,
            slow_mlp,
            vec_scalar,
            d_scalar,
            vec_multihead,
            d_mh,
        )
    return {"ran": True, "ok": ok, "tol": tol, "rows": rows}


# ── main ──────────────────────────────────────────────────────────────────────


def _run_extension_phases(
    args,
    betley: dict,
    *,
    layer_subset: list[int] | None,
    threads: int | None,
    krr_bootstrap: int,
    out722: Path,
    out_ext_dir: Path,
    run_width_sweep_flag: bool,
    run_epoch_curves_flag: bool,
    run_krr_flag: bool,
) -> None:
    """Run the opt-in additive extension phases (width sweep / epoch curves / KRR).

    Each enabled phase writes its OWN output JSON + figure under
    ``eval_results/issue_722/base-skill-over-mean-cC-to-v0/`` and
    ``figures/issue_722/`` — never touching ``skill_over_mean.json``.
    Extracted from ``main`` to keep its cyclomatic complexity under the cap.
    """
    width_sweep_result = None
    if run_width_sweep_flag:
        width_sweep_result = run_width_sweep(
            betley,
            widths=args.mlp_widths,
            plateau_layers=args.plateau_layers,
            device=args.device,
            num_threads=threads,
            existing_json=out722,
            layer_subset=layer_subset,
        )
        width_sweep_result["metadata"] = reproducibility_metadata(
            {"script": "issue722_vectorized_skill", "phase": "mlp_width_sweep"}
        )
        out_ws = out_ext_dir / "mlp_width_sweep.json"
        out_ext_dir.mkdir(parents=True, exist_ok=True)
        dump_json(width_sweep_result, out_ws)
        logger.info("wrote %s", out_ws)
        make_width_sweep_figure(
            width_sweep_result, REPO_ROOT / "figures/issue_722/mlp_width_sweep.png"
        )

    if run_epoch_curves_flag:
        epoch_result = run_epoch_curves(
            betley,
            widths=args.epoch_curve_widths,
            layers_grid=args.epoch_curve_layers,
            eval_every=args.epoch_curve_every,
            device=args.device,
            num_threads=threads,
        )
        epoch_result["metadata"] = reproducibility_metadata(
            {"script": "issue722_vectorized_skill", "phase": "mlp_epoch_curves"}
        )
        out_ec = out_ext_dir / "mlp_epoch_curves.json"
        out_ext_dir.mkdir(parents=True, exist_ok=True)
        dump_json(epoch_result, out_ec)
        logger.info("wrote %s", out_ec)
        make_epoch_curve_figure(epoch_result, REPO_ROOT / "figures/issue_722/mlp_epoch_curves.png")

    if run_krr_flag:
        krr_result = run_krr_vs_linear(
            betley,
            width_sweep=width_sweep_result,
            device=args.device,
            existing_json=out722,
            n_boot=krr_bootstrap,
            layer_subset=layer_subset,
        )
        krr_result["metadata"] = reproducibility_metadata(
            {"script": "issue722_vectorized_skill", "phase": "krr_vs_linear"}
        )
        out_krr = out_ext_dir / "krr_vs_linear.json"
        out_ext_dir.mkdir(parents=True, exist_ok=True)
        dump_json(krr_result, out_krr)
        logger.info("wrote %s", out_krr)
        make_krr_figure(krr_result, REPO_ROOT / "figures/issue_722/krr_vs_linear_nonlinearity.png")
        sanity = krr_result["krr_linear_vs_ridge_sanity"]
        print(f"\n==== KRR-linear vs PCA-48 linear-ridge sanity: ok={sanity['ok']} ====")
        for row in sanity["rows"]:
            print(
                f"  L{row['layer']:02d}: ridge(pca48)={row['ridge_pca48']:+.5f} "
                f"krr_lin={row['krr_linear_pca48']:+.5f} |Δ|={row['abs_delta']:.2e}"
            )
        print("\n==== KRR nonlinear gap (RBF − linear), per layer ====")
        for r in krr_result["per_layer"]:
            mark = " *EXCLUDES0*" if r["gap_excludes_zero"] else ""
            print(
                f"  L{r['layer']:02d}: ridge(fullH)={r['skill_vs_mean_ridge_fullH']:+.4f} "
                f"rbf={r['skill_krr_rbf_pca48']:+.4f} "
                f"gap={r['nonlinear_gap_rbf_minus_linear']:+.4f} "
                f"CI=[{r['gap_ci95'][0]:+.4f},{r['gap_ci95'][1]:+.4f}] "
                f"df_eff={r['ridge_df_eff']:.1f}{mark}"
            )


def _run_input_rep_phase(
    args,
    betley: dict,
    *,
    layer_subset: list[int] | None,
    krr_bootstrap: int,
    baseline_skill_json: Path,
    baseline_krr_json: Path,
    smoke_slice: bool,
) -> None:
    """Run the input-representation robustness amendment (round-2) for each variant.

    For each requested ``--input-rep`` variant (pca48 / whiten48), re-run BOTH
    headline arms under the per-fold input transform (``run_input_rep_robustness``)
    and write the plan v7 §6.5 flat-named deliverables
    ``skill_over_mean__{variant}.json`` + ``krr_vs_linear__{variant}.json`` under the
    ``input-pca-robustness-cC-to-v0`` subdir (NOT a nested ``{variant}/`` subdir).
    Then assemble ``input_rep_comparison.json`` (the §6 SUCCESS/KILL verdict per
    variant), ``run_meta.json`` (config + code SHA + substrate provenance), and the
    two-panel ``figures/issue_722/input_rep_robustness_per_layer.png``. Reads the
    committed variant-1 (full) baseline JSONs as the Δ denominator — NEVER re-runs
    full. A ``smoke_slice`` run lands under an ``_smoke/`` subdir, apart from
    production.
    """
    if not (baseline_skill_json.exists() and baseline_krr_json.exists()):
        raise FileNotFoundError(
            "--input-rep needs the committed variant-1 baseline JSONs as the Δ denominator: "
            f"{baseline_skill_json} and {baseline_krr_json} (plan §3 — full is the committed "
            "baseline, not re-run). Run the canonical + --run-krr phases first, or restore "
            "the commit."
        )
    t_phase0 = time.time()  # §4.4 wall-time for the whole input-rep phase
    out_dir = REPO_ROOT / "eval_results/issue_722" / args.input_rep_out_subdir
    if smoke_slice:
        out_dir = out_dir / "_smoke"  # tiny-N smoke artifacts kept apart from production
    out_dir.mkdir(parents=True, exist_ok=True)
    results_by_variant: dict[str, tuple[dict, dict]] = {}
    for variant in args.input_rep:
        if variant == "full":
            logger.info("[input-rep] 'full' is the committed baseline — not re-run")
            continue
        skill_result, krr_result = run_input_rep_robustness(
            betley,
            input_rep=variant,
            device=args.device,
            baseline_skill_json=baseline_skill_json,
            baseline_krr_json=baseline_krr_json,
            n_boot=krr_bootstrap,
            k=args.input_rep_k,
            eps=args.input_rep_eps,
            layer_subset=layer_subset,
        )
        meta = reproducibility_metadata(
            {"script": "issue722_vectorized_skill", "phase": f"input_rep_{variant}"}
        )
        # Plan v7 §6.5: flat per-variant filenames under the followup_label subdir
        # (skill_over_mean__<variant>.json / krr_vs_linear__<variant>.json), NOT a
        # nested <variant>/ subdir.
        skill_path = out_dir / f"skill_over_mean__{variant}.json"
        krr_path = out_dir / f"krr_vs_linear__{variant}.json"
        dump_json({**skill_result, "metadata": meta}, skill_path)
        dump_json({**krr_result, "metadata": meta}, krr_path)
        logger.info("wrote %s + %s", skill_path, krr_path)
        results_by_variant[variant] = (skill_result, krr_result)

    if not results_by_variant:
        logger.info("[input-rep] no non-full variants requested — nothing to compare")
        return

    comparison = build_input_rep_comparison(results_by_variant)
    comparison["metadata"] = reproducibility_metadata(
        {"script": "issue722_vectorized_skill", "phase": "input_rep_comparison"}
    )
    comp_path = out_dir / "input_rep_comparison.json"  # plan v7 §6.5 deliverable name
    dump_json(comparison, comp_path)
    logger.info("wrote %s", comp_path)

    # §4.4 n / d / n_layers — read off the first variant's result (all variants
    # share the substrate, so n_contexts / hidden_dim / layer count are identical).
    _first_skill = next(iter(results_by_variant.values()))[0]
    _sp = _first_skill.get("store_provenance", {})
    n_contexts = int(_sp.get("n_contexts", _first_skill.get("n_contexts")))
    hidden_dim = int(_sp.get("hidden_dim", _first_skill.get("activation_dim")))
    n_layers = len(_first_skill.get("layers", []))
    wall_time_minutes = round((time.time() - t_phase0) / 60.0, 3)

    run_meta = {
        "script": "issue722_vectorized_skill",
        "phase": "input_rep_robustness",
        "code_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": list(results_by_variant.keys()),
        "baseline_full_skill_json": str(baseline_skill_json),
        "baseline_full_krr_json": str(baseline_krr_json),
        "input_rep_k": args.input_rep_k,
        "input_rep_eps": args.input_rep_eps,
        "ridge_band": INPUT_REP_RIDGE_BAND,
        "gap_band": INPUT_REP_GAP_BAND,
        "pass_min_layers": INPUT_REP_PASS_MIN_LAYERS,
        "n_bootstrap": krr_bootstrap,
        "seed": SEED_722,
        # §4.4 required provenance fields (BLOCKER input-rep-run-meta-incomplete):
        "rng_state_hash": _rng_state_hash(SEED_722),
        "n_contexts": n_contexts,  # n = 50
        "hidden_dim": hidden_dim,  # d = 3584
        "n_layers": n_layers,  # 28
        "wall_time_minutes": wall_time_minutes,
        "device": args.device,
        "layers": layer_subset if layer_subset is not None else "all",
        "substrate_provenance": _resolve_substrate_provenance(betley),
        "store_provenance": {
            "store_dir": betley["store_dir"],
            "cc_source": betley["cc_source"],
        },
        "whitening": "PCA-whitening (per-direction 1/sqrt(sigma^2+eps); rotation immaterial "
        "for ridge-internal restandardization + RBF per-direction scale, plan §11)",
    }
    meta_path = out_dir / "run_meta.json"
    dump_json(run_meta, meta_path)
    logger.info("wrote %s", meta_path)

    # γ-sensitivity diagnostic (CONCERN krr-gap-collapse-gamma-regime-interpretation,
    # plan §4.4 exploratory band): re-run the pca48 RBF arm at the ridge-plateau
    # layers under γ = multiplier × per-fold γ₀, so the analyzer can tell a genuine
    # RBF collapse from a bad median-γ regime at 48-d. Only when pca48 is a variant.
    if "pca48" in results_by_variant:
        # Under a --layers smoke slice, probe only the requested layers that overlap
        # the default plateau probes (falls back to the smoke slice itself if none).
        if layer_subset is not None:
            probe_layers = tuple(x for x in GAMMA_SENS_LAYERS if x in layer_subset)
            if not probe_layers:
                probe_layers = tuple(layer_subset[:2])
        else:
            probe_layers = GAMMA_SENS_LAYERS
        gamma_sens = run_gamma_sensitivity(
            betley,
            input_rep="pca48",
            layers_to_probe=probe_layers,
            k=args.input_rep_k,
            eps=args.input_rep_eps,
        )
        gamma_sens["metadata"] = reproducibility_metadata(
            {"script": "issue722_vectorized_skill", "phase": "gamma_sensitivity_pca48"}
        )
        gs_path = out_dir / "gamma_sensitivity__pca48.json"  # plan §4.4 diagnostic name
        dump_json(gamma_sens, gs_path)
        logger.info("wrote %s", gs_path)
        print("\n==== γ-sensitivity (pca48 RBF vs linear, gap = RBF − linear) ====")
        for pl in gamma_sens["per_layer"]:
            gaps = " ".join(
                f"{r['multiplier']:g}x:{r['gap_rbf_minus_linear']:+.3f}"
                for r in pl["by_multiplier"]
            )
            print(
                f"  L{pl['layer']:02d}: linear={pl['skill_krr_linear']:+.4f} "
                f"best_gap={pl['best_gap']:+.4f}@{pl['best_multiplier']:g}x  [{gaps}]"
            )

    # Plan v7 §6 Figure: two stacked panels (ridge R² + KRR(RBF)−linear gap) for the
    # full baseline + each re-represented variant, after both variants + the
    # comparison land. Reads the flat-named variant JSONs + the committed full
    # baseline JSONs.
    fig_path = REPO_ROOT / "figures/issue_722/input_rep_robustness_per_layer.png"
    make_input_rep_figure(
        results_by_variant,
        baseline_skill_json=baseline_skill_json,
        baseline_krr_json=baseline_krr_json,
        layer_subset=layer_subset,
        fig_path=fig_path,
    )

    print("\n==== input-rep robustness verdict (vs committed full baseline) ====")
    for variant, pv in comparison["per_variant"].items():
        print(
            f"  {variant:9s} verdict={pv['verdict']} "
            f"R²-gate {pv['n_layers_passing_R2_gate']}/{pv['n_layers']} "
            f"gap-gate {pv['n_layers_passing_gap_gate']}/{pv['n_layers']} "
            f"(need >={pv['pass_min_layers']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorized #722 skill + #658 chain ρ.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--threads", type=int, default=8, help="torch CPU threads (sane value; 0=leave default)"
    )
    parser.add_argument(
        "--betley-store",
        type=Path,
        default=DATA_ROOT / "data/issue_658/hf_dl/issue658_theory_assumptions/store",
    )
    parser.add_argument(
        "--ultrachat-store",
        type=Path,
        default=(
            DATA_ROOT
            / "data/issue_658/g1_dl/issue658_theory_assumptions"
            / "store_genre-generalization-ultrachat"
        ),
    )
    parser.add_argument(
        "--betley-e0", type=Path, default=DATA_ROOT / "eval_results/issue_658/E0_expression.json"
    )
    parser.add_argument(
        "--ultrachat-e0",
        type=Path,
        default=DATA_ROOT / "eval_results/issue_658/E0_expression_g1.json",
    )
    parser.add_argument("--spot-layers", type=int, nargs="*", default=[0, 12, 18])
    parser.add_argument("--skip-spot-check", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2-epoch + 1 genre + 2 layers smoke")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="restrict to these layer NUMBERS (tiny-N smoke; e.g. --layers 0 18). The LOCO "
        "fold count stays at n_contexts (cannot drop below 50).",
    )
    # ── ADDITIVE extension flags (all OPT-IN; default invocation is unchanged) ──
    parser.add_argument(
        "--skip-canonical",
        action="store_true",
        help="skip the canonical skill_over_mean.json + #658-chain writes (run ONLY the "
        "opt-in extension phases below against the existing skill_over_mean.json baseline)",
    )
    parser.add_argument(
        "--run-width-sweep",
        action="store_true",
        help="Deliverable 1: MLP width sweep (held-out skill per width x layer) "
        "-> mlp_width_sweep.json",
    )
    parser.add_argument(
        "--mlp-widths",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256, 512],
        help="MLP hidden widths for the width sweep",
    )
    parser.add_argument(
        "--plateau-layers",
        type=int,
        nargs="+",
        default=[14, 18, 21],
        help="ridge-plateau layers: representative width-sweep lines + shuffle-null layers",
    )
    parser.add_argument(
        "--run-epoch-curves",
        action="store_true",
        help="Deliverable 2: per-epoch held-out curves for a width x layer grid "
        "-> mlp_epoch_curves.json",
    )
    parser.add_argument(
        "--epoch-curve-widths",
        type=int,
        nargs="+",
        default=[8, 32, 128, 512],
        help="MLP widths for the per-epoch curve grid",
    )
    parser.add_argument(
        "--epoch-curve-layers",
        type=int,
        nargs="+",
        default=[0, 18],
        help="layers for the per-epoch curve grid (L18=ridge plateau peak, L0=neg control)",
    )
    parser.add_argument(
        "--epoch-curve-every",
        type=int,
        default=10,
        help="snapshot the held-out skill every K epochs",
    )
    parser.add_argument(
        "--run-krr",
        action="store_true",
        help="KRR (RBF + linear) vs linear ridge nonlinear-gap -> krr_vs_linear.json",
    )
    parser.add_argument(
        "--krr-bootstrap",
        type=int,
        default=2000,
        help="LOCO-fold bootstrap resamples for the gap/ridge CIs",
    )
    # ── round-2 amendment: input-representation robustness (per-fold, no leakage) ──
    parser.add_argument(
        "--input-rep",
        nargs="+",
        default=None,
        choices=list(INPUT_REPS),
        help="Run the input-representation robustness amendment for these variants "
        "(pca48 / whiten48; 'full' is the committed baseline and is never re-run). "
        "Writes eval_results/issue_722/input-pca-robustness-cC-to-v0/"
        "skill_over_mean__{variant}.json + krr_vs_linear__{variant}.json + "
        "input_rep_comparison.json + run_meta.json (plan v7 §6.5 primary deliverable). "
        "Implies --skip-canonical unless the canonical phases are also requested.",
    )
    parser.add_argument(
        "--input-rep-out-subdir",
        default="input-pca-robustness-cC-to-v0",
        help="Output subdir under eval_results/issue_722/ for the input-rep variants "
        "(the followup_label; plan v7 §6.5)",
    )
    parser.add_argument(
        "--input-rep-k", type=int, default=INPUT_REP_K, help="top-k PCs for pca48/whiten48"
    )
    parser.add_argument(
        "--input-rep-eps", type=float, default=INPUT_REP_EPS, help="ZCA whitening ε"
    )
    args = parser.parse_args()

    i658.DEVICE = args.device
    threads = args.threads if args.threads > 0 else None
    layer_subset = args.layers
    krr_bootstrap = args.krr_bootstrap
    run_width_sweep_flag = args.run_width_sweep
    run_epoch_curves_flag = args.run_epoch_curves
    run_krr_flag = args.run_krr
    run_input_rep = bool(args.input_rep)
    # The input-rep amendment reads the committed full baseline; it never re-runs the
    # canonical skill_over_mean.json / #658-chain writes (plan §3). Auto-skip them
    # unless the user explicitly also requested a canonical/extension phase.
    skip_canonical = args.skip_canonical or (
        run_input_rep and not (run_width_sweep_flag or run_epoch_curves_flag or run_krr_flag)
    )
    if args.smoke:
        i658.MLP_MAX_EPOCHS = 2
        if layer_subset is None:
            layer_subset = [0, 18]  # 2-layer smoke slice
        # Exercise EVERY new phase end-to-end in the smoke (KRR is cheap at n=50).
        run_width_sweep_flag = run_epoch_curves_flag = run_krr_flag = True
        krr_bootstrap = min(krr_bootstrap, 200)
    # the input-rep bootstrap rides the same krr-bootstrap budget; bound it on smoke.
    if run_input_rep and (args.smoke or (layer_subset is not None and len(layer_subset) <= 2)):
        krr_bootstrap = min(krr_bootstrap, 200)

    out722 = REPO_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json"
    out_ext_dir = out722.parent
    baseline_krr_json = out_ext_dir / "krr_vs_linear.json"

    t_run0 = time.time()

    # ── EXACTNESS GATE: batched MLP reproduces the #658 reference ──
    gate = assert_matches_reference()
    logger.info("MLP batched exactness gate PASS: %s", gate)

    betley = _load_genre("betley", args.betley_store, args.betley_e0)
    chain_betley = None
    chain_ultrachat = None

    if not skip_canonical:
        ultrachat = _load_genre("ultrachat", args.ultrachat_store, args.ultrachat_e0)

        # ── #722 canonical (Betley) ──
        r722 = run_722_skill_over_mean(
            betley, device=args.device, num_threads=threads, layer_subset=layer_subset
        )
        out722.parent.mkdir(parents=True, exist_ok=True)
        r722["metadata"] = reproducibility_metadata({"script": "issue722_vectorized_skill"})
        dump_json(r722, out722)
        logger.info("wrote %s", out722)
        make_722_figure(r722, REPO_ROOT / "figures/issue_722/base_skill_over_mean_per_layer.png")

        # ── #658 chain (both genres) ──
        chain_betley = run_658_chain(
            betley, "betley", device=args.device, num_threads=threads, layer_subset=layer_subset
        )
        chain_ultrachat = run_658_chain(
            ultrachat,
            "ultrachat",
            device=args.device,
            num_threads=threads,
            layer_subset=layer_subset,
        )
        meta = reproducibility_metadata({"script": "issue722_vectorized_skill"})
        out_b = REPO_ROOT / "eval_results/issue_658/a34a35_mlp_chain.json"
        out_u = REPO_ROOT / "eval_results/issue_658_g1/a34a35_mlp_chain.json"
        out_b.parent.mkdir(parents=True, exist_ok=True)
        out_u.parent.mkdir(parents=True, exist_ok=True)
        dump_json({**chain_betley, "metadata": meta}, out_b)
        dump_json({**chain_ultrachat, "metadata": meta}, out_u)
        logger.info("wrote %s + %s", out_b, out_u)
        make_658_figure(
            chain_betley,
            chain_ultrachat,
            REPO_ROOT / "figures/issue_658/a34a35_mlp_vs_ridge_chain.png",
        )
    else:
        logger.info("[skip-canonical] skipping skill_over_mean.json + #658-chain writes")

    # ── ADDITIVE extension phases (opt-in; each writes its OWN output file) ──
    _run_extension_phases(
        args,
        betley,
        layer_subset=layer_subset,
        threads=threads,
        krr_bootstrap=krr_bootstrap,
        out722=out722,
        out_ext_dir=out_ext_dir,
        run_width_sweep_flag=run_width_sweep_flag,
        run_epoch_curves_flag=run_epoch_curves_flag,
        run_krr_flag=run_krr_flag,
    )

    # ── round-2 amendment: input-representation robustness (opt-in via --input-rep) ──
    if run_input_rep:
        _run_input_rep_phase(
            args,
            betley,
            layer_subset=layer_subset,
            krr_bootstrap=krr_bootstrap,
            baseline_skill_json=out722,
            baseline_krr_json=baseline_krr_json,
            smoke_slice=bool(args.smoke or (layer_subset is not None and len(layer_subset) <= 2)),
        )

    # ── reproduce-checks (canonical path only) ──
    wall_h = (time.time() - t_run0) / 3600.0

    if not skip_canonical:
        spot = {"ran": False, "reason": "skipped"}
        if not args.skip_spot_check:
            spot = spot_check_vs_slow_mlp(betley, args.spot_layers)

        # console summary
        print("\n==== reproduce-check: ridge full-H chain ρ (byte-exact control) ====")
        for res in (chain_betley, chain_ultrachat):
            print(f"  {res['genre']:10s} reproduced={res['ridge_full_chain_repro_control']['ok']}")
            for col in READOUT_BEHAVIORS:
                row = res["ridge_full_chain_repro_control"]["rows"].get(col, {})
                print(
                    f"    {col:20s} got={row.get('got_rho', float('nan')):+.6f} "
                    f"exp={row.get('expected_rho', float('nan')):+.6f} "
                    f"Δ={row.get('abs_rho_delta', float('nan')):.2e} match={row.get('match')}"
                )
        print("\n==== chain ρ (ridge full-H | ridge PCA-64 | MLP PCA-64) ====")
        for res in (chain_betley, chain_ultrachat):
            for col in READOUT_BEHAVIORS:
                rf = res["ridge_full_chain_rho"].get(col)
                pb = res["per_behavior"].get(col, {})
                rg = pb.get("ridge_pca64_chain")
                ml = pb.get("mlp_pca64_chain")
                print(
                    f"  {res['genre']:10s} {col:20s} "
                    f"full={(rf['rho'] if rf else float('nan')):+.3f} "
                    f"ridge64={(rg['rho'] if rg else float('nan')):+.3f} "
                    f"mlp64={(ml['rho'] if ml else float('nan')):+.3f}"
                )
        print("\n==== #722 skill-over-mean (Betley, best layer per arm) ====")
        pl = r722["per_layer"]
        best = lambda key: max(pl, key=lambda r: r[key] if r[key] == r[key] else -9)  # noqa: E731
        for key in (
            "skill_vs_mean_ridge",
            "skill_vs_mean_mlp",
            "skill_zscored_mlp",
            "skill_shuffle_mlp",
        ):
            b = best(key)
            print(f"  {key:22s} best L{b['layer']:02d} = {b[key]:+.4f}")
        print(f"  shuffle_ridge_L18 = {r722['shuffle_ridge_L18']:+.4f}")
        if spot.get("ran"):
            print(f"\n==== slow-MLP spot-check (scalar tol={spot['tol']}): ok={spot['ok']} ====")
            for row in spot["rows"]:
                print(
                    f"  L{row['layer']:02d}: slow_scalar={row['slow_scalar_mlp_skill']:+.5f} "
                    f"vec_scalar={row['vec_scalar_mlp_skill']:+.5f} "
                    f"(|Δ|={row['abs_delta_scalar']:.2e}) "
                    f"vec_multihead={row['vec_multihead_mlp_skill']:+.5f} "
                    f"(|Δ_vs_slow|={row['abs_delta_multihead']:.2e})"
                )

        # persist the reproduce-check + wall-time sidecar
        sidecar = out722.parent / "vectorized_repro_check.json"
        dump_json(
            {
                "mlp_exactness_gate": gate,
                "ridge_chain_repro": {
                    "betley": chain_betley["ridge_full_chain_repro_control"],
                    "ultrachat": chain_ultrachat["ridge_full_chain_repro_control"],
                },
                "slow_mlp_spot_check": spot,
                "wall_time_minutes": round(wall_h * 60, 2),
                "device": args.device,
                "threads": args.threads,
                "code_sha": _git_sha(),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            sidecar,
        )
        logger.info("wrote %s", sidecar)

    print(f"\nWALL-TIME (vectorized, device={args.device}): {wall_h * 60:.1f} min ({wall_h:.3f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
