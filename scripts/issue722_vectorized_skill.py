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
    MLPGroup,
    assert_matches_reference,
    chain_rho_pca,
    fit_batched_loco_mlp,
    fit_batched_loco_mlp_multihead,
    loco_train_means,
    ridge_predict_loco_centered,
    ridge_predict_loco_raw,
    robust_pca_basis,
    skill_over_mean_r2,
    zscore_columns,
)

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
    args = parser.parse_args()

    i658.DEVICE = args.device
    threads = args.threads if args.threads > 0 else None
    layer_subset = None
    if args.smoke:
        i658.MLP_MAX_EPOCHS = 2
        layer_subset = [0, 18]  # 2-layer smoke slice

    t_run0 = time.time()

    # ── EXACTNESS GATE: batched MLP reproduces the #658 reference ──
    gate = assert_matches_reference()
    logger.info("MLP batched exactness gate PASS: %s", gate)

    betley = _load_genre("betley", args.betley_store, args.betley_e0)
    ultrachat = _load_genre("ultrachat", args.ultrachat_store, args.ultrachat_e0)

    # ── #722 canonical (Betley) ──
    r722 = run_722_skill_over_mean(
        betley, device=args.device, num_threads=threads, layer_subset=layer_subset
    )
    out722 = REPO_ROOT / "eval_results/issue_722/base-skill-over-mean-cC-to-v0/skill_over_mean.json"
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
        ultrachat, "ultrachat", device=args.device, num_threads=threads, layer_subset=layer_subset
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
        chain_betley, chain_ultrachat, REPO_ROOT / "figures/issue_658/a34a35_mlp_vs_ridge_chain.png"
    )

    # ── reproduce-checks ──
    wall_h = (time.time() - t_run0) / 3600.0

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
    print(f"\nWALL-TIME (vectorized, device={args.device}): {wall_h * 60:.1f} min ({wall_h:.3f} h)")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
