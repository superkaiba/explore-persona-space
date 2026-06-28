#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, r̂, r_B, →, ρ, M⁺, ※, ×) in scientific docstrings + log messages.
"""Issue #722 — fit M0 vs M⁺ (ridge + MLP) and the four reads (plan §4.4 / §4.5).

For each ``(behavior, layer ∈ {7,14,21})`` this fits THREE maps through the
#658 LOCO harness:

- **M0** from ``(c0 → v0)`` — the pre-FT context→answer function (base);
- **M⁺** from ``(cplus → vplus)`` — the post-FT function (adapter-applied
  activations, the post-FT INPUT drives the post-FT OUTPUT);
- **M_pseudo** from ``(cplus → M0(cplus))`` — the SAME M0 function on shifted
  inputs (the ``floor_shifted`` same-function shifted-design null).

Each map is fit twice: a closed-form PRESS-LOCO ridge (``_ridge_predict_loco``)
and a GPU-batched LOCO MLP ensemble (``_fit_mlp_ensemble_loco``), output target
= the top-64 v0 PCs (``A35_MLP_TARGET_DIM``), so the ridge-vs-MLP gap is
like-for-like.

**Headline DV (plan §3 / §4.5.1):** ``Δ_med = median_c |Δ(c)·r̂_B|`` over the
base ``common_c_grid`` (both maps evaluated at the SAME base c0), with a
family-clustered CI from the NEW ``clustered_bootstrap_scalar`` (NOT the
Spearman helper — distinct stat). Gated on the COMBINED floor
``max(floor_M0_refit, floor_Mplus_refit, floor_shifted)``, each built through the
IDENTICAL bootstrap+random-init refit harness (``make_refit_pair``).

**Co-primary (chain-ρ):** held-out Spearman of ``r_Bᵀ M̂(c)`` vs E (= ``g`` from
#537's ``G_meta.json``), under M0 vs M⁺, family-clustered via the EXISTING
``clustered_bootstrap_spearman`` (correct two-array use).

Plus cross-transfer (read 3), the linear-vs-nonlinear gap (read 4), and the
per-cell support-distance diagnostic ``‖cplus − c0‖``.

The ridge + M_pseudo closed-form path is CPU; the MLP fits are the GPU phase
(CLAUDE.md compute-character carve-out — a gradient-descent fit is GPU-worthy).
Per-(behavior, layer) checkpoints to ``eval_results/issue_722/cells/`` make the
run resumable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
from issue722_bootstrap import clustered_bootstrap_scalar, floor_sd, make_refit_pair  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)

logger = logging.getLogger("issue722.fit")

HIDDEN = 3584
DATA_REPO = "superkaiba1/explore-persona-space-data"
SWEEP_LAYERS = (7, 14, 21)
HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")  # marker + refusal dropped (plan §4.3/§5)
# #537 G behavior key map (behavior dir name -> G_meta.json behavior prefix).
G_BEHAVIOR_KEY = {"em": "em", "sycophancy": "sycophancy", "fact": "fact", "marker": "marker"}
# r_b.pt column key map (behavior -> r_b.pt key); fact uses the NEW r_b_fact.pt.
RB_COLUMN_KEY = {"em": "broad_em", "sycophancy": "sycophancy", "refusal": "harmful_compliance"}
SUPPORT_SHIFT_PCTL = 90  # large-shift flag threshold (plan §3)
N_REFIT_PAIRS = 100
N_SCALAR_BOOT = 1000
# Output target dim (top-v0 PCs) shared by ridge + MLP — the #658 A35_MLP_TARGET_DIM
# (64) for the production run; a smoke clamps it via --target-dim to bound the
# CPU MLP-ensemble cost (the ensemble size is target_dim × n_folds). Module-global
# so the refit-floor closures read the same value.
TARGET_DIM = 64


def _to64(Y: np.ndarray, pca_basis: np.ndarray) -> np.ndarray:
    """Project a (n, 3584) target onto the shared top-64 v0 PCs (n, 64)."""
    return Y @ pca_basis.T


def _pca_basis_v0(V0: np.ndarray, dim: int) -> np.ndarray:
    """Top-`dim` PCA basis of the base v0 stack (dim, 3584), mean-centered.

    Shared between ridge + MLP so the nonlinearity gap is like-for-like
    (the #658 A35_MLP_TARGET_DIM reduction applied to the v0 output target).
    """
    Vc = V0 - V0.mean(axis=0, keepdims=True)
    # economy SVD; rows of Vt are the principal directions.
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    k = min(dim, Vt.shape[0])
    return Vt[:k]  # (k, 3584)


def _ridge_fit_predict(X: np.ndarray, Y64: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Ridge map fit on (X→Y64), evaluated at `grid` → (n_grid, 64).

    Uses #658's closed-form dual-ridge weights at the PRESS-selected λ (fit on
    ALL rows, not LOCO — the function-change read evaluates the fitted map on a
    fixed grid; LOCO is for the held-out ρ reads, not for M(c) at a new input).
    """
    lambdas = fit658.RIDGE_LAMBDAS
    device = torch.device(fit658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y64)).to(device=device, dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    mse = fit658._press_loo_mse_per_lambda(Xn, Yt, lambdas)
    best_lam = lambdas[int(torch.argmin(mse).item())]
    w = fit658._ridge_dual_weights(Xn, Yt, best_lam)  # (d, 64)
    Gt = torch.from_numpy(np.ascontiguousarray(grid)).to(device=device, dtype=torch.float64)
    Gn = (Gt - mu) / sd
    return (Gn @ w).detach().cpu().numpy()


def _mlp_loco_pred(X: np.ndarray, Y64: np.ndarray) -> np.ndarray:
    """LOCO MLP held-out predictions for all 64 output dims → (n, 64)."""
    return fit658._fit_mlp_ensemble_loco(
        X.astype(np.float32), Y64.astype(np.float32), target_idx=list(range(Y64.shape[1]))
    )


def _ridge_loco_pred(X: np.ndarray, Y64: np.ndarray) -> np.ndarray:
    """LOCO ridge held-out predictions for all 64 output dims → (n, 64)."""
    return fit658._ridge_predict_loco(X, Y64, fit658.RIDGE_LAMBDAS)


def _chain_rho_one(pred64: np.ndarray, pca_basis: np.ndarray, r_hat: np.ndarray, E: np.ndarray):
    """Spearman(r_Bᵀ M̂(c), E) — project the 64-dim pred back to 3584, dot r̂_B."""
    pred_full = pred64 @ pca_basis  # (n, 3584)
    chain = pred_full @ r_hat  # (n,)
    return fit658._rho(chain, E), chain


def _r_hat_for(behavior: str, layer: int, rb_main: dict, rb_fact: dict | None) -> np.ndarray:
    """Unit r_B at this (behavior, layer), from r_b.pt (em/syc) or r_b_fact.pt (fact)."""
    if behavior == "fact":
        if rb_fact is None:
            raise RuntimeError("fact headline requested but r_b_fact.pt not loaded")
        stack = np.asarray(rb_fact["r_b_fact"]["fact_expression"]["diffmeans"], dtype=np.float64)
    else:
        col = RB_COLUMN_KEY[behavior]
        stack = np.asarray(rb_main["r_b"][col]["diffmeans"], dtype=np.float64)
    assert stack.shape[0] >= layer + 1, f"r_B stack {stack.shape} has no layer {layer}"
    r = stack[layer]
    norm = np.linalg.norm(r)
    if norm < 1e-9:
        raise RuntimeError(f"degenerate r_B for {behavior} L{layer} (norm {norm:.2e})")
    return r / norm


def _load_rb_main() -> dict:
    """Load #658 r_b.pt from HF (em/syc/refusal directions)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        DATA_REPO, "issue658_theory_assumptions/store/r_b.pt", repo_type="dataset"
    )
    return torch.load(local, weights_only=False)


def _load_rb_fact() -> dict | None:
    """Load the NEW r_b_fact.pt (this task's fact direction); None if absent."""
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            DATA_REPO, "issue722_rb_extension/store/r_b_fact.pt", repo_type="dataset"
        )
    except Exception as e:  # not yet extracted (e.g. fit-only smoke) — caller drops fact
        logger.warning("r_b_fact.pt unavailable (%s); fact headline will be skipped", e)
        return None
    payload = torch.load(local, weights_only=False)
    if payload.get("degenerate"):
        logger.warning("r_b_fact.pt flagged degenerate — fact dropped from headline (plan §8)")
        return None
    return payload


def _load_E(behavior: str, cell_keys: list[str]) -> np.ndarray:
    """E = #537 G_meta.json `g` per cell, aligned to cell_keys (NaN where absent)."""
    meta_path = PROJECT_ROOT / "eval_results/issue_537/G_tensor/G_meta.json"
    pc = json.loads(meta_path.read_text())["per_cell"]
    out = np.full(len(cell_keys), np.nan, dtype=np.float64)
    for i, k in enumerate(cell_keys):
        if k in pc and pc[k].get("g") is not None:
            out[i] = float(pc[k]["g"])
    return out


def fit_cell(behavior: str, layer: int, cells: list, rb_main: dict, rb_fact: dict | None) -> dict:
    """Fit M0/M⁺/M_pseudo + all four reads for one (behavior, layer). Returns the cell JSON."""
    stacks = loadact.stack_for_fit(cells)
    C0, Cplus = stacks["C0"], stacks["Cplus"]
    V0, Vplus = stacks["V0"], stacks["Vplus"]
    families = stacks["families"]
    cell_keys = stacks["cell_keys"]
    n = C0.shape[0]
    assert n >= 4, f"{behavior} L{layer}: only {n} cells (<4) — cannot fit"

    r_hat = _r_hat_for(behavior, layer, rb_main, rb_fact)  # (3584,)
    pca_basis = _pca_basis_v0(V0, TARGET_DIM)  # (k<=TARGET_DIM, 3584)
    V0_64 = _to64(V0, pca_basis)
    Vplus_64 = _to64(Vplus, pca_basis)
    grid = loadact.common_c_grid(stacks)  # base c0 grid (n, 3584)

    # Function-change read (HEADLINE) is the closed-form RIDGE map evaluated at the
    # fixed base grid — the linear M fidelity, and the plan §3 ridge-only path is a
    # VALID headline by construction (each floor is a difference of two equally-weak
    # refits, so refit noise cancels). The MLP is a universal-approximator UPPER
    # bound that enters only through the held-out chain-ρ + the nonlinearity-gap read
    # (§4.5.4) below — it is never evaluated at a fresh grid input (an MLP has no
    # closed-form off-LOCO read, and the function-change DV needs M(c) at a fixed c).
    # M_pseudo target = M0(Cplus): computed inside the floor_shift refit (below).
    m0_grid = _ridge_fit_predict(C0, V0_64, grid)  # (n_grid, 64)
    mplus_grid = _ridge_fit_predict(Cplus, Vplus_64, grid)

    # ---- Headline Δ_med (ridge) on the projected grid ----
    delta = mplus_grid - m0_grid  # (n_grid, 64)
    delta_full = delta @ pca_basis  # (n_grid, 3584)
    proj = np.abs(delta_full @ r_hat)  # (n_grid,)
    delta_med_ci = clustered_bootstrap_scalar(
        proj, families, statistic="median", n_resamples=N_SCALAR_BOOT
    )
    delta_med = delta_med_ci["point"]
    delta_med_mean_ci = clustered_bootstrap_scalar(
        proj, families, statistic="mean", n_resamples=N_SCALAR_BOOT
    )

    # ---- Three floors via the identical refit harness ----
    # M0 refit floor: refit M0 (C0→V0) pairs, eval at grid.
    floor_m0 = make_refit_pair(C0, V0, _refit_ridge_fn(grid), grid, r_hat, n_pairs=N_REFIT_PAIRS)
    floor_mplus = make_refit_pair(
        Cplus, Vplus, _refit_ridge_fn(grid), grid, r_hat, n_pairs=N_REFIT_PAIRS
    )
    # shifted-design: M_pseudo (Cplus → M0(Cplus)); refit pairs of THAT map at grid.
    floor_shift = make_refit_pair(
        Cplus,
        m0_at_cplus_ridge_full(C0, V0, Cplus, pca_basis),
        _refit_ridge_fn(grid),
        grid,
        r_hat,
        n_pairs=N_REFIT_PAIRS,
    )
    floor_m0_p95 = float(np.percentile(floor_m0, 95))
    floor_mplus_p95 = float(np.percentile(floor_mplus, 95))
    floor_shift_p95 = float(np.percentile(floor_shift, 95))
    floor_combined = max(floor_m0_p95, floor_mplus_p95, floor_shift_p95)
    floor_sd_combined = max(floor_sd(floor_m0), floor_sd(floor_mplus), floor_sd(floor_shift))

    # ---- Support distance ‖cplus − c0‖ + large-shift flag ----
    support = np.linalg.norm(Cplus - C0, axis=1)  # (n,)
    shift_thresh = float(np.percentile(support, SUPPORT_SHIFT_PCTL))
    large_shift_mask = support > shift_thresh
    # Δ_med excluding large-shift cells (the grid is per-cell c0, so mask the proj).
    if large_shift_mask.any() and (~large_shift_mask).sum() >= 4:
        fam_keep = [f for f, m in zip(families, large_shift_mask, strict=True) if not m]
        proj_keep = proj[~large_shift_mask]
        delta_med_excl_ci = clustered_bootstrap_scalar(
            proj_keep, fam_keep, statistic="median", n_resamples=N_SCALAR_BOOT
        )
    else:
        delta_med_excl_ci = delta_med_ci

    # ---- Chain-ρ co-primary (LOCO, both maps) ----
    E = _load_E(behavior, cell_keys)
    keep = ~np.isnan(E)
    chain_block = {"n_with_E": int(keep.sum())}
    if keep.sum() >= 4:
        Ek = E[keep]
        fam_k = [f for f, m in zip(families, keep, strict=True) if m]
        m0_loco_ridge = _ridge_loco_pred(C0, V0_64)
        mplus_loco_ridge = _ridge_loco_pred(Cplus, Vplus_64)
        rho_m0, chain_m0 = _chain_rho_one(m0_loco_ridge[keep], pca_basis, r_hat, Ek)
        rho_mplus, chain_mplus = _chain_rho_one(mplus_loco_ridge[keep], pca_basis, r_hat, Ek)
        chain_block["rho_M0_ridge"] = rho_m0
        chain_block["rho_Mplus_ridge"] = rho_mplus
        chain_block["rho_diff_ridge"] = (
            None if (rho_m0 is None or rho_mplus is None) else float(rho_mplus - rho_m0)
        )
        if rho_m0 is not None:
            chain_block["ci_M0_ridge"] = clustered_bootstrap_spearman(chain_m0, Ek, fam_k)
        if rho_mplus is not None:
            chain_block["ci_Mplus_ridge"] = clustered_bootstrap_spearman(chain_mplus, Ek, fam_k)
        # MLP chain-ρ + nonlinearity gap (read 4) + MLP-validity (shuffle) on M0.
        m0_loco_mlp = _mlp_loco_pred(C0, V0_64)
        mplus_loco_mlp = _mlp_loco_pred(Cplus, Vplus_64)
        rho_m0_mlp, _ = _chain_rho_one(m0_loco_mlp[keep], pca_basis, r_hat, Ek)
        rho_mplus_mlp, _ = _chain_rho_one(mplus_loco_mlp[keep], pca_basis, r_hat, Ek)
        chain_block["rho_M0_mlp"] = rho_m0_mlp
        chain_block["rho_Mplus_mlp"] = rho_mplus_mlp
        # shuffle null on M0 (refit ridge on permuted v0) — MLP-validity gate (plan §3).
        rng = np.random.default_rng(722)
        perm = rng.permutation(n)
        m0_shuf = _ridge_loco_pred(C0, V0_64[perm])
        rho_shuf, _ = _chain_rho_one(m0_shuf[keep], pca_basis, r_hat, Ek)
        chain_block["rho_M0_shuffle"] = rho_shuf
        # nonlinearity gap pre vs post: (rho_mlp - rho_ridge) under M0 and M⁺.
        if None not in (rho_m0_mlp, rho_m0):
            chain_block["nonlin_gap_M0"] = float(rho_m0_mlp - rho_m0)
        if None not in (rho_mplus_mlp, rho_mplus):
            chain_block["nonlin_gap_Mplus"] = float(rho_mplus_mlp - rho_mplus)

    # ---- Cross-transfer (read 3) ----
    cross = {}
    # M0 predicting v_plus on FT pairs (held-out ρ proxy: ridge LOCO of C0→Vplus_64)
    # vs M⁺ predicting v_plus (its own LOCO). Reverse: M⁺ predicting v0 on base pairs.
    m0_to_vplus = _ridge_loco_pred(C0, Vplus_64)
    mplus_to_vplus = _ridge_loco_pred(Cplus, Vplus_64)
    mplus_to_v0 = _ridge_loco_pred(Cplus, V0_64)
    # summarize as mean rowwise cosine to the true target (a transfer-quality scalar).
    cross["m0_to_vplus_cos"] = float(np.mean(fit658._rowwise_cos(m0_to_vplus, Vplus_64)))
    cross["mplus_to_vplus_cos"] = float(np.mean(fit658._rowwise_cos(mplus_to_vplus, Vplus_64)))
    cross["mplus_to_v0_cos"] = float(np.mean(fit658._rowwise_cos(mplus_to_v0, V0_64)))

    return {
        "behavior": behavior,
        "layer": layer,
        "n_cells": n,
        "Delta_med": delta_med,
        "Delta_med_ci": delta_med_ci,
        "Delta_med_mean_ci": delta_med_mean_ci,
        "Delta_med_excl_large_shift_ci": delta_med_excl_ci,
        "floor_M0_refit": floor_m0_p95,
        "floor_Mplus_refit": floor_mplus_p95,
        "floor_shifted": floor_shift_p95,
        "floor_combined": floor_combined,
        "floor_sd_combined": floor_sd_combined,
        "Delta_over_floor_sd": (
            None if floor_sd_combined < 1e-12 else float(delta_med / floor_sd_combined)
        ),
        "support_distance": {
            "mean": float(support.mean()),
            "p90": shift_thresh,
            "n_large_shift": int(large_shift_mask.sum()),
        },
        "chain_rho": chain_block,
        "cross_transfer": cross,
        "n_families": len({*families}),
    }


def _refit_ridge_fn(grid: np.ndarray):
    """A fit_fn(Xb, Yb_full, rng) for make_refit_pair — fits ridge on the bootstrap sample.

    The PCA basis is recomputed per bootstrap sample (its OWN top-64 v0 PCs) so the
    refit is a genuine independent refit, mirroring how the headline map is fit.
    Returns predictions at `grid` projected back to 3584 so the floor's
    `delta @ r_hat` is in the same 3584-space as the headline.
    """

    def _fn(Xb: np.ndarray, Yb: np.ndarray, _rng) -> np.ndarray:
        pca = _pca_basis_v0(Yb, TARGET_DIM)
        pred64 = _ridge_fit_predict(Xb, Yb @ pca.T, grid)  # (n_grid, k)
        return pred64 @ pca  # back to (n_grid, 3584) for the r_hat projection

    return _fn


def m0_at_cplus_ridge_full(C0, V0, Cplus, pca):
    """M0 fit on (C0 → V0 top-64), predicted at Cplus, back-projected to 3584 (n,3584)."""
    Y64 = V0 @ pca.T
    pred64 = _ridge_fit_predict(C0, Y64, Cplus)
    return pred64 @ pca


def main() -> int:
    global N_REFIT_PAIRS, TARGET_DIM
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #722 fit M0 vs M⁺ + the four reads")
    ap.add_argument("--behaviors", nargs="+", default=list(HEADLINE_BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(SWEEP_LAYERS))
    ap.add_argument(
        "--max-cells", type=int, default=None, help="smoke: cap total cells per behavior×layer"
    )
    ap.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="smoke: cap source_cid dirs per behavior (the distinct-c0 count; MUST be >=2)",
    )
    ap.add_argument(
        "--max-targets-per-source",
        type=int,
        default=None,
        help="smoke: cap targets per source (bounds total cells while spanning sources)",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_722/cells")
    ap.add_argument(
        "--smoke", action="store_true", help="1 behavior, 1 layer, capped sources/cells"
    )
    ap.add_argument(
        "--mlp-epochs",
        type=int,
        default=None,
        help="override MLP_MAX_EPOCHS (smoke clamps the 300-epoch CPU cost; full run uses 300)",
    )
    ap.add_argument(
        "--refit-pairs",
        type=int,
        default=N_REFIT_PAIRS,
        help="bootstrap+random-init refit PAIRS per floor (smoke clamps from 100)",
    )
    ap.add_argument(
        "--target-dim",
        type=int,
        default=fit658.A35_MLP_TARGET_DIM,
        help="output target dim = top-v0 PCs (default 64; smoke clamps to bound the CPU MLP)",
    )
    args = ap.parse_args()
    if args.smoke:
        args.behaviors = args.behaviors[:1]
        args.layers = args.layers[:1]
        # Span >=4 SOURCES (the distinct-c0 count) so the fit is non-degenerate —
        # c_C is constant within a source, so a single source gives one input row.
        args.max_sources = args.max_sources or 6
        # Cap targets/source so the smoke spans 6 sources × 4 targets = 24 cells
        # (6 distinct c0) without the full 180-cell CPU MLP cost.
        if args.max_targets_per_source is None:
            args.max_targets_per_source = 4
        # Clamp the three dominant CPU costs so the GPU-bound MLP phase runs
        # end-to-end on the VM CPU as a carve-out smoke (the full GPU run uses
        # 300 epochs / 100 pairs / 64 dims). #658 _assert_mlp_exactness epoch-clamp.
        if args.mlp_epochs is None:
            args.mlp_epochs = 20
        args.refit_pairs = min(args.refit_pairs, 8)
        args.target_dim = min(args.target_dim, 4)
    if args.mlp_epochs is not None:
        fit658.MLP_MAX_EPOCHS = args.mlp_epochs
    N_REFIT_PAIRS = args.refit_pairs
    TARGET_DIM = args.target_dim
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layers = tuple(args.layers)
    behaviors = tuple(args.behaviors)
    logger.info(
        "[phase=fit_M] behaviors=%s layers=%s max_cells=%s", behaviors, layers, args.max_cells
    )

    # Run the exactness gates (#658) so a reduction-order regression fails at startup.
    fit658._assert_ridge_exactness()
    logger.info("[phase=fit_M] ridge exactness gate PASS")

    rb_main = _load_rb_main()
    rb_fact = _load_rb_fact() if "fact" in behaviors else None
    if "fact" in behaviors and rb_fact is None:
        logger.warning("fact requested but r_b_fact.pt unavailable/degenerate — dropping fact")
        behaviors = tuple(b for b in behaviors if b != "fact")

    # strict_counts asserts the verified 480-cell per-behavior×layer grid; disabled
    # whenever the grid is deliberately capped (--smoke OR an explicit cap flag).
    strict = (
        not args.smoke
        and args.max_cells is None
        and args.max_sources is None
        and args.max_targets_per_source is None
    )
    cells_by = loadact.load_cells(
        behaviors=behaviors,
        layers=layers,
        max_cells=args.max_cells,
        max_sources=args.max_sources,
        max_targets_per_source=args.max_targets_per_source,
        strict_counts=strict,
    )
    for behavior in behaviors:
        for layer in layers:
            cells = cells_by[(behavior, layer)]
            logger.info("[phase=fit_M] %s L%d (%d cells)", behavior, layer, len(cells))
            cell = fit_cell(behavior, layer, cells, rb_main, rb_fact)
            cell["metadata"] = {
                "issue": 722,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            out = args.out_dir / f"{behavior}_L{layer}.json"
            out.write_text(json.dumps(cell, indent=2, default=float))
            logger.info(
                "[phase=fit_M]   Δ_med=%.4g floor_combined=%.4g over_sd=%s",
                cell["Delta_med"],
                cell["floor_combined"],
                cell["Delta_over_floor_sd"],
            )
    logger.info("[phase=fit_M] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
