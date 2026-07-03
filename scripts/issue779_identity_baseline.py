#!/usr/bin/env python3
"""Issue #779 free re-analysis: identity-baseline decomposition of the map ``h``.

``h: c_last -> v(x)`` maps the last-prompt-token activation to the mean-response
activation at the SAME layer of the SAME forward pass — adjacent token spans of
the residual stream, plausibly so correlated that h's held-out reconstruction R2
(0.598-0.625 at the read-out layers; eval_results/issue_779/percontext_recon.json)
is mostly identity/autocorrelation rather than a learned mapping. The existing
shuffled-pairing null (~0.12 R2) does NOT test identity-copying (it only kills
the global-mean component). This 0-GPU driver quantifies what h does beyond a
fair no-learning baseline, in three reads:

A. Baseline ladder on held-out reconstruction (the SAME 5-fold CV over the 5000
   LMSYS contexts as issue779_percontext_recon.py: same fold seed, same
   test-fold-own-mean pooled-R2 convention):
     1. raw identity            v^ = c_last                (no parameters)
     2. global affine identity  v^ = alpha*c_last + mu     (1 scalar + mean offset)
     3. per-dim diagonal affine v^_d = a_d*c_d + b_d       (2H closed-form 1-D fits)
     4. full ridge h            (refit via _ridge_fit_predict_fast, fold parity
                                 asserted against the stored percontext numbers)
     5. train-fold mean         v^ = mean_train(v(x))      (the ~0-floor anchor)
   plus the bare cos(c_last, v(x)) distribution (no fit at all). Cheap rungs run
   at all 28 layers (curve); the full-h curve is REUSED from percontext_recon.json
   (identical folds by construction). h's GENUINE mapping contribution =
   rung4 - rung3 (full ridge minus diagonal affine).

B. Weight-matrix decomposition (is W ~ a scaled identity?) at the 3 read-out
   layers, standardized per the existing recipe (X standardized on train stats,
   Y centered): (i) best alpha for min ||W - alpha*I||_F + the relative residual;
   (ii) mean/median |diag| vs |offdiag| magnitude; (iii) held-out R2 predicting
   through diag(W) only vs offdiag(W) only vs full W (same standardization, same
   5 folds); (iv) top-10 singular values of W and of W - diag(W). The primal-form
   ridge (needed to materialize W; the dual Gram path never does) is gated for
   prediction equivalence against _ridge_fit_predict_fast before use.

C. Readout-level test (the read that matters for monitoring): per trait x mode
   on the eval rig, the per-(condition, question) Pearson between pv_raw =
   <c_last, r_B> (the RAW prompt-token projection, no learning) and the TRUE
   oracle <v(x), r_B> — overall AND within-condition (1000-condition-bootstrap
   95% CI) — side-by-side with h's projection-recon numbers
   (percontext_recon.json read2). corr(pv_raw, oracle) ~ corr(<h(c),r_B>, oracle)
   ==> h adds ~nothing along the trait direction.

Reuses (never reimplements): issue779_percontext_recon._ridge_fit_predict_fast /
_pooled_r2 / _per_context_cosine / _cosine_dist_summary / _cv_folds /
READ_OUT_LAYER; issue779_stage1.load_eval_cells / _load_rb / build_eval_matrix /
_group_by_condition; metrics.overall_pearson / bootstrap_within_condition_ci;
issue779_common.write_json_atomic / reproducibility_metadata.

0-GPU, CPU/VM only (thread-capped). Fail loud; NaN reported, never coerced.
Checkpoint-per-analysis writes to --out-json (sections merge, so ``--only``
re-runs are incremental).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue779_stage1 as S1  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_identity_baseline")

READ_OUT_LAYER = PR.READ_OUT_LAYER  # {"evil": 14, "sycophancy": 26, "hallucination": 17}
LAMBDAS = np.logspace(-2, 4, 13)  # fit_h.ridge_fit_predict's GCV grid
CHEAP_RUNGS = ("train_mean", "raw_identity", "global_affine", "diag_affine")
ALL_RUNGS = (*CHEAP_RUNGS, "full_h")
RUNG_LABEL = {
    "train_mean": "train-fold mean (floor)",
    "raw_identity": "raw identity v^=c_last",
    "global_affine": "global affine a*c+mu",
    "diag_affine": "per-dim diag affine",
    "full_h": "full ridge h",
}


# ── A. baseline ladder ─────────────────────────────────────────────────────────


def _fit_global_affine(Xtr: np.ndarray, Ytr: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form LS for v^ = ymu + alpha*(x - xmu): one scalar alpha, train-mean
    offset (mu = ymu - alpha*xmu). Returns (alpha, xmu, ymu)."""
    xmu = Xtr.mean(0)
    ymu = Ytr.mean(0)
    Xc = Xtr - xmu
    Yc = Ytr - ymu
    alpha = float((Xc * Yc).sum() / ((Xc * Xc).sum() + 1e-30))
    return alpha, xmu, ymu


def _fit_diag_affine(Xtr: np.ndarray, Ytr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3584 independent 1-D regressions v^_d = b_d + a_d*x_d, fully vectorized
    (elementwise closed form — no loop over dims). Returns (a (H,), xmu, ymu)."""
    xmu = Xtr.mean(0)
    ymu = Ytr.mean(0)
    Xc = Xtr - xmu
    Yc = Ytr - ymu
    a = (Xc * Yc).sum(0) / ((Xc * Xc).sum(0) + 1e-30)
    return a, xmu, ymu


def ladder_layer(
    X: np.ndarray,
    Y: np.ndarray,
    folds: list[np.ndarray],
    *,
    refit_h: bool,
    want_cosine: bool,
) -> dict:
    """Held-out baseline ladder at one layer (same folds/convention as percontext).

    X, Y: (N, H) float64. Returns {rung: {"r2_folds": [...], "cos_pool": [...]?},
    "alpha_folds": [...]} — cosines pooled per-context across test folds.
    """
    n = X.shape[0]
    rungs = ALL_RUNGS if refit_h else CHEAP_RUNGS
    r2: dict[str, list[float]] = {r: [] for r in rungs}
    cos_pool: dict[str, list[np.ndarray]] = {r: [] for r in rungs}
    alphas: list[float] = []
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        Xtr, Ytr = X[mask], Y[mask]
        Xte, Yte = X[test_idx], Y[test_idx]
        preds: dict[str, np.ndarray] = {}
        preds["train_mean"] = np.broadcast_to(Ytr.mean(0), Yte.shape)
        preds["raw_identity"] = Xte
        alpha, xmu, ymu = _fit_global_affine(Xtr, Ytr)
        alphas.append(alpha)
        preds["global_affine"] = ymu + alpha * (Xte - xmu)
        a, xmu_d, ymu_d = _fit_diag_affine(Xtr, Ytr)
        preds["diag_affine"] = ymu_d + a * (Xte - xmu_d)
        if refit_h:
            preds["full_h"] = PR._ridge_fit_predict_fast(Xtr, Ytr, Xte)
        for rung in rungs:
            r2[rung].append(PR._pooled_r2(preds[rung], Yte))
            if want_cosine:
                cos_pool[rung].append(PR._per_context_cosine(preds[rung], Yte))
    out: dict = {"alpha_folds": alphas}
    for rung in rungs:
        out[rung] = {"r2_folds": r2[rung]}
        if want_cosine:
            out[rung]["cos_pool"] = cos_pool[rung]
    return out


def _r2_summary(folds_vals: list[float]) -> dict:
    return {
        "mean": float(np.mean(folds_vals)),
        "sd": float(np.std(folds_vals)),
        "folds": [float(v) for v in folds_vals],
    }


def analysis_a(
    train_bundle: dict,
    traits: list[str],
    existing_curve: dict,
    *,
    n_folds: int,
    seed: int,
    n_layers: int,
) -> dict:
    """Baseline ladder: cheap rungs at ALL layers (curve) + full ladder incl. the
    full-h refit + per-rung cosine distributions at the read-out layers."""
    cx_last = train_bundle["cx_last"]
    v_x = train_bundle["v_x"]
    layers = train_bundle["layers"]
    n = cx_last.shape[0]
    folds = PR._cv_folds(n, n_folds, seed)
    readout_layers = sorted(set(READ_OUT_LAYER[t] for t in traits))

    per_layer: dict[str, dict] = {}
    readout: dict[str, dict] = {}
    bare_cos_mean_curve: dict[str, float] = {}
    for li in range(n_layers):
        col = layers.index(li)
        X = cx_last[:, col, :].numpy().astype(np.float64)
        Y = v_x[:, col, :].numpy().astype(np.float64)
        is_readout = li in readout_layers
        res = ladder_layer(X, Y, folds, refit_h=is_readout, want_cosine=is_readout)
        entry: dict = {r: _r2_summary(res[r]["r2_folds"]) for r in CHEAP_RUNGS}
        entry["global_affine_alpha"] = {
            "mean": float(np.mean(res["alpha_folds"])),
            "folds": [float(a) for a in res["alpha_folds"]],
        }
        # Bare autocorrelation: cos(c_last, v(x)) on ALL contexts, no fit (equals
        # the pooled rung-1 cosine — the fit-free read the ladder is anchored on).
        bare_cos = PR._per_context_cosine(X, Y)
        bare_cos_mean_curve[str(li)] = float(np.nanmean(bare_cos))
        if is_readout:
            entry["full_h"] = _r2_summary(res["full_h"]["r2_folds"])
            # Fold-parity assert vs the stored percontext held-out curve (same
            # _cv_folds seed + same _ridge_fit_predict_fast => must reproduce).
            stored = existing_curve[str(li)]["mean"]
            refit = entry["full_h"]["mean"]
            assert abs(refit - stored) < 1e-4, (
                f"fold-parity FAILED at layer {li}: refit full-h R2 {refit:.6f} vs "
                f"stored percontext {stored:.6f}"
            )
            entry["cosine_dist"] = {
                r: PR._cosine_dist_summary(np.concatenate(res[r]["cos_pool"])) for r in ALL_RUNGS
            }
            entry["bare_cosine_dist"] = PR._cosine_dist_summary(bare_cos)
            readout[str(li)] = entry
        per_layer[str(li)] = {r: entry[r] for r in CHEAP_RUNGS}
        logger.info(
            "A layer %2d: raw=%.4f global=%.4f diag=%.4f%s (bare cos %.3f)",
            li,
            entry["raw_identity"]["mean"],
            entry["global_affine"]["mean"],
            entry["diag_affine"]["mean"],
            f" full_h={entry['full_h']['mean']:.4f}" if is_readout else "",
            bare_cos_mean_curve[str(li)],
        )

    per_trait = {}
    for t in traits:
        li = READ_OUT_LAYER[t]
        e = readout[str(li)]
        per_trait[t] = {
            "read_out_layer": li,
            "genuine_contribution_r2": e["full_h"]["mean"] - e["diag_affine"]["mean"],
            "full_minus_global_r2": e["full_h"]["mean"] - e["global_affine"]["mean"],
        }
    return {
        "n_contexts": int(n),
        "n_folds": n_folds,
        "seed": seed,
        "r2_convention": (
            "pooled held-out R2; SS_tot uses the TEST fold's own mean (identical "
            "to percontext_recon read1)"
        ),
        "per_layer_curve": per_layer,
        "readout_layers": readout,
        "bare_cosine_mean_by_layer": bare_cos_mean_curve,
        "full_h_curve_existing": {
            k: {"mean": v["mean"], "sd": v["sd"]} for k, v in existing_curve.items()
        },
        "per_trait": per_trait,
    }


# ── B. weight-matrix decomposition ─────────────────────────────────────────────


def primal_ridge_w(
    Xtr: np.ndarray, Ytr: np.ndarray, *, lambdas: np.ndarray | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Materialize the ridge weight matrix W under the EXACT fit_h recipe.

    Same standardize-X / center-Y / GCV-lambda-select as fit_h.ridge_fit_predict:
    the primal GCV over eigh(Xn^T Xn) has identical filter factors to the dual
    Gram form (nonzero eigenvalues coincide; zero eigenvalues contribute 0 to
    both RSS and dof), so it selects the same lambda — gated for prediction
    equivalence against _ridge_fit_predict_fast in analysis_b. Returns
    (W (H, H), xmu, xsd, ymu, lam); prediction contract:
    pred = ((Xev - xmu) / xsd) @ W + ymu.
    """
    if lambdas is None:
        lambdas = LAMBDAS
    Xt = torch.as_tensor(np.asarray(Xtr), dtype=torch.float64)
    Yt = torch.as_tensor(np.asarray(Ytr), dtype=torch.float64)
    xmu = Xt.mean(0)
    xsd = Xt.std(0) + 1e-9
    Xn = (Xt - xmu) / xsd
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    ntr = Xn.shape[0]
    Ct = Xn.T @ Xn  # (H, H)
    s2, Q = torch.linalg.eigh(Ct)
    s2 = torch.clamp(s2, min=0.0)
    A = Xn.T @ Yc  # (H, H)
    B = Q.T @ A
    b2 = (B**2).sum(1)
    tot = float((Yc**2).sum())
    best_lam, best_gcv = float(lambdas[0]), float("inf")
    for lam in lambdas:
        denom_vec = s2 + lam
        rss = tot - float((2.0 * b2 / denom_vec).sum()) + float((s2 * b2 / denom_vec**2).sum())
        dof = float((s2 / denom_vec).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
    W = Q @ (B / (s2 + best_lam).unsqueeze(1))  # (H, H)
    return W, xmu, xsd, ymu, best_lam


def _equivalence_gate_primal(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray]) -> dict:
    """Assert the primal-W path reproduces _ridge_fit_predict_fast on fold 0
    (fail-loud before any W is decomposed). Same thresholds as PR's gate."""
    test_idx = folds[0]
    mask = np.ones(X.shape[0], dtype=bool)
    mask[test_idx] = False
    Xtr, Ytr, Xte, Yte = X[mask], Y[mask], X[test_idx], Y[test_idx]
    pred_dual = PR._ridge_fit_predict_fast(Xtr, Ytr, Xte)
    W, xmu, xsd, ymu, lam = primal_ridge_w(Xtr, Ytr)
    Xte_n = (torch.as_tensor(Xte, dtype=torch.float64) - xmu) / xsd
    pred_primal = (Xte_n @ W + ymu).numpy()
    abs_diff = float(np.max(np.abs(pred_dual - pred_primal)))
    rel_diff = abs_diff / (float(np.max(np.abs(pred_dual))) + 1e-12)
    r2_diff = abs(PR._pooled_r2(pred_dual, Yte) - PR._pooled_r2(pred_primal, Yte))
    assert rel_diff < 1e-6 and r2_diff < 1e-6, (
        f"primal-W equivalence gate FAILED: rel pred diff {rel_diff:.2e}, R2 diff {r2_diff:.2e}"
    )
    logger.info(
        "primal-W equivalence gate PASS (rel diff %.2e, R2 diff %.2e, lambda %.4g)",
        rel_diff,
        r2_diff,
        lam,
    )
    return {"rel_pred_diff": rel_diff, "r2_diff": r2_diff, "lambda_fold0": lam}


def analysis_b_layer(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray]) -> dict:
    """Full-data W decomposition + 5-fold diag/offdiag/full prediction split."""
    # (i)/(ii)/(iv): W fit ONCE on all rows.
    W, _xmu, _xsd, _ymu, lam_full = primal_ridge_w(X, Y)
    Wn = W.numpy()
    d = Wn.shape[0]
    dW = np.diag(Wn).copy()
    alpha_star = float(np.trace(Wn) / d)  # argmin_a ||W - a*I||_F
    frob_w = float(np.linalg.norm(Wn))
    rel_residual = float(np.linalg.norm(Wn - alpha_star * np.eye(d)) / frob_w)
    off = Wn - np.diag(dW)
    off_mask = ~np.eye(d, dtype=bool)
    offdiag_abs = np.abs(off[off_mask])
    diag_abs = np.abs(dW)
    sv_w = torch.linalg.svdvals(torch.as_tensor(Wn)).numpy()
    sv_off = torch.linalg.svdvals(torch.as_tensor(off)).numpy()

    # (iii) held-out prediction decomposition, same folds + standardization.
    n = X.shape[0]
    r2s: dict[str, list[float]] = {"full": [], "diag_only": [], "offdiag_only": []}
    lam_folds: list[float] = []
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        Wf, xmu, xsd, ymu, lam = primal_ridge_w(X[mask], Y[mask])
        lam_folds.append(lam)
        dWf = torch.diagonal(Wf).clone()
        Xte_n = (torch.as_tensor(X[test_idx], dtype=torch.float64) - xmu) / xsd
        Yte = Y[test_idx]
        pred_full = Xte_n @ Wf + ymu
        pred_diag = Xte_n * dWf + ymu
        pred_off = pred_full - Xte_n * dWf  # = Xte_n @ (Wf - diag(Wf)) + ymu
        r2s["full"].append(PR._pooled_r2(pred_full.numpy(), Yte))
        r2s["diag_only"].append(PR._pooled_r2(pred_diag.numpy(), Yte))
        r2s["offdiag_only"].append(PR._pooled_r2(pred_off.numpy(), Yte))

    return {
        "lambda_full_fit": lam_full,
        "lambda_folds": lam_folds,
        "alpha_star": alpha_star,
        "rel_residual_w_minus_alpha_i": rel_residual,
        "offdiag_frobenius_fraction": float((np.linalg.norm(off) ** 2) / frob_w**2),
        "diag_abs": {"mean": float(diag_abs.mean()), "median": float(np.median(diag_abs))},
        "offdiag_abs": {
            "mean": float(offdiag_abs.mean()),
            "median": float(np.median(offdiag_abs)),
        },
        "diag_over_offdiag_abs_mean_ratio": float(diag_abs.mean() / (offdiag_abs.mean() + 1e-30)),
        "heldout_r2": {k: _r2_summary(v) for k, v in r2s.items()},
        "sv_top10_w": [float(v) for v in sv_w[:10]],
        "sv_top10_w_minus_diag": [float(v) for v in sv_off[:10]],
        "note": (
            "W maps standardized c_last (train xmu/xsd) to centered v(x) per the "
            "fit_h recipe; alpha*I is read in that standardized space."
        ),
    }


def analysis_b(train_bundle: dict, traits: list[str], *, n_folds: int, seed: int) -> dict:
    """W decomposition at each read-out layer (gate first, fail loud)."""
    cx_last = train_bundle["cx_last"]
    v_x = train_bundle["v_x"]
    layers = train_bundle["layers"]
    n = cx_last.shape[0]
    folds = PR._cv_folds(n, n_folds, seed)
    readout_layers = sorted(set(READ_OUT_LAYER[t] for t in traits))

    col0 = layers.index(readout_layers[0])
    gate = _equivalence_gate_primal(
        cx_last[:, col0, :].numpy().astype(np.float64),
        v_x[:, col0, :].numpy().astype(np.float64),
        folds,
    )

    per_layer = {}
    for li in readout_layers:
        col = layers.index(li)
        X = cx_last[:, col, :].numpy().astype(np.float64)
        Y = v_x[:, col, :].numpy().astype(np.float64)
        res = analysis_b_layer(X, Y, folds)
        per_layer[str(li)] = res
        logger.info(
            "B layer %2d: alpha*=%.4f rel-resid=%.4f | heldout R2 full=%.4f diag=%.4f offdiag=%.4f",
            li,
            res["alpha_star"],
            res["rel_residual_w_minus_alpha_i"],
            res["heldout_r2"]["full"]["mean"],
            res["heldout_r2"]["diag_only"]["mean"],
            res["heldout_r2"]["offdiag_only"]["mean"],
        )
    return {
        "equivalence_gate": gate,
        "per_layer": per_layer,
        "trait_layers": {t: READ_OUT_LAYER[t] for t in traits},
    }


# ── C. readout-level test ──────────────────────────────────────────────────────


def analysis_c(
    cells_by_trait: dict,
    rb_by_trait: dict,
    existing_read2: dict,
    traits: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """corr(pv_raw, oracle) per trait x mode vs h's read2 projection-recon.

    pv_raw = <c_last, r_B> is the fit-free identity readout along the trait
    direction; oracle = <v(x), r_B> is the true answer projection. Same grouping,
    finite-filtering, and bootstrap machinery as percontext_recon read2.
    """
    out = {}
    for t in traits:
        li = READ_OUT_LAYER[t]
        mat = S1.build_eval_matrix(cells_by_trait[t], li, rb_by_trait[t])
        pred = np.asarray(mat["pv_raw"], dtype=np.float64)
        true = np.asarray(mat["oracle"], dtype=np.float64)
        trait_res: dict = {"read_out_layer": li, "n_eval": len(true)}
        for mode in ("system", "many_shot"):
            sel = np.array([m == mode for m in mat["mode"]]) & np.isfinite(pred) & np.isfinite(true)
            overall = M.overall_pearson(pred[sel], true[sel]) if sel.sum() >= 3 else float("nan")
            cx, cy = S1._group_by_condition(pred, true, mat["cond"], mat["mode"], mode)
            cx2, cy2 = [], []
            for xi, yi in zip(cx, cy, strict=True):
                m = np.isfinite(xi) & np.isfinite(yi)
                if m.sum() >= 3:
                    cx2.append(xi[m])
                    cy2.append(yi[m])
            wc = M.bootstrap_within_condition_ci(cx2, cy2, n_boot=n_boot, seed=seed)
            h_ref = existing_read2.get(t, {}).get(mode, {})
            pv_within = wc["point"]
            h_within = h_ref.get("within_condition_r", float("nan"))
            trait_res[mode] = {
                "pv_raw_vs_oracle": {
                    "overall_pearson": overall,
                    "overall_n": int(sel.sum()),
                    "within_condition_r": pv_within,
                    "within_condition_lo": wc["lo"],
                    "within_condition_hi": wc["hi"],
                    "within_condition_n_conditions": wc["n_conditions"],
                },
                "h_vs_oracle_read2": {
                    "overall_pearson": h_ref.get("overall_pearson", float("nan")),
                    "within_condition_r": h_within,
                    "within_condition_lo": h_ref.get("within_condition_lo", float("nan")),
                    "within_condition_hi": h_ref.get("within_condition_hi", float("nan")),
                },
                "h_minus_pvraw_within": (
                    h_within - pv_within
                    if np.isfinite(h_within) and np.isfinite(pv_within)
                    else float("nan")
                ),
            }
            logger.info(
                "C %s %s: pv_raw-vs-oracle overall r=%.3f within=%.3f [%.3f, %.3f] | "
                "h-vs-oracle within=%.3f (delta h-pv=%+.3f)",
                t,
                mode,
                overall,
                pv_within,
                wc["lo"],
                wc["hi"],
                h_within,
                trait_res[mode]["h_minus_pvraw_within"],
            )
        out[t] = trait_res
    return out


# ── D. per-direction variance-spectrum decomposition ──────────────────────────


def _per_direction_r2(true: np.ndarray, pred: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Held-out R2 of pred vs true PROJECTED onto each unit direction (columns of
    ``dirs`` (H, K)). Same convention as _pooled_r2: SS_tot on the test fold's
    OWN projected mean; degenerate SS_tot -> NaN (reported, never coerced)."""
    a = true @ dirs  # (n_te, K)
    p = pred @ dirs
    ss_res = ((a - p) ** 2).sum(0)
    ss_tot = ((a - a.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot < 1e-12, np.nan, 1.0 - ss_res / ss_tot)
    return r2


def analysis_d_layer(
    X: np.ndarray,
    Y: np.ndarray,
    rb_l: np.ndarray,
    test_idx: np.ndarray,
    *,
    k_lead: int,
    tail_step: int,
    n_random: int,
    seed: int,
) -> dict:
    """Per-direction held-out R2 of the full-ridge h on ONE fold (fold 0).

    PCA the TRAIN-fold true targets (centered on the train mean), evaluate the
    held-out per-direction R2 along each leading PCA direction (+ a tail sample
    every ``tail_step`` ranks), along r_B (unit-normalized), and along
    ``n_random`` random unit directions (the matched-chance reference band).
    """
    n, h_dim = X.shape
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    Xtr, Ytr = X[mask], Y[mask]
    Xte, Yte = X[test_idx], Y[test_idx]
    pred = PR._ridge_fit_predict_fast(Xtr, Ytr, Xte)  # the SAME h as ladder rung 4

    # PCA of the train-fold targets (centered on the train mean).
    Ytr_c = Ytr - Ytr.mean(0)
    n_tr = Ytr.shape[0]
    _u, s, vh = torch.linalg.svd(torch.as_tensor(Ytr_c, dtype=torch.float64), full_matrices=False)
    vh_np = vh.numpy()  # (D, H): row k = PCA direction u_k
    var_spectrum = (s.numpy() ** 2) / (n_tr - 1)  # per-direction train variance
    total_var = float(var_spectrum.sum())
    d_full = vh_np.shape[0]

    ranks = list(range(min(k_lead, d_full))) + list(range(k_lead, d_full, tail_step))
    dirs = vh_np[ranks].T  # (H, n_sel)
    r2_by_rank = _per_direction_r2(Yte, pred, dirs)
    var_by_rank = var_spectrum[ranks]
    share_by_rank = var_by_rank / total_var
    cum_share = np.cumsum(var_spectrum) / total_var

    # r_B (unit-normalized).
    u_rb = rb_l / (np.linalg.norm(rb_l) + 1e-12)
    r2_rb = float(_per_direction_r2(Yte, pred, u_rb[:, None])[0])
    var_rb = float(np.var(Ytr_c @ u_rb, ddof=1))
    var_percentile = float(np.mean(var_spectrum < var_rb) * 100.0)  # % of PCA dirs BELOW
    equivalent_rank = int(np.sum(var_spectrum > var_rb))  # rank where var_k ~ var_rb
    # PCA directions of matched variance: the 5 evaluated ranks nearest var_rb in
    # log-variance — the PCA-side "direction of matched variance" reference.
    log_dist = np.abs(np.log(var_by_rank + 1e-30) - np.log(var_rb + 1e-30))
    nearest = np.argsort(log_dist)[:5]
    matched = {
        "ranks": [int(ranks[i]) for i in nearest],
        "r2_mean": float(np.nanmean(r2_by_rank[nearest])),
        "r2_min": float(np.nanmin(r2_by_rank[nearest])),
        "r2_max": float(np.nanmax(r2_by_rank[nearest])),
    }

    # Random unit directions (mean +- band reference).
    rng = np.random.default_rng(seed + 779)
    rand = rng.standard_normal((h_dim, n_random))
    rand /= np.linalg.norm(rand, axis=0, keepdims=True) + 1e-12
    r2_rand = _per_direction_r2(Yte, pred, rand)
    var_rand = np.var(Ytr_c @ rand, axis=0, ddof=1)

    # Best/worst 5 evaluated PCA ranks + the zero crossing.
    finite = np.isfinite(r2_by_rank)
    fin_ranks = np.array(ranks)[finite]
    fin_r2 = r2_by_rank[finite]
    order = np.argsort(fin_r2)
    worst5 = [{"rank": int(fin_ranks[i]), "r2": float(fin_r2[i])} for i in order[:5]]
    best5 = [{"rank": int(fin_ranks[i]), "r2": float(fin_r2[i])} for i in order[::-1][:5]]
    below = fin_ranks[fin_r2 < 0.0]
    first_below = int(below.min()) if below.size else None

    return {
        "fold": 0,
        "n_train": int(n_tr),
        "n_test": len(test_idx),
        "ranks_evaluated": [int(r) for r in ranks],
        "r2_by_rank": [float(v) for v in r2_by_rank],
        "variance_by_rank": [float(v) for v in var_by_rank],
        "variance_share_by_rank": [float(v) for v in share_by_rank],
        "cum_variance_share": {
            str(k): float(cum_share[k - 1]) for k in (10, 50, 100, 200) if k <= d_full
        },
        "r_b": {
            "heldout_r2": r2_rb,
            "train_variance": var_rb,
            "variance_percentile_of_pca_spectrum": var_percentile,
            "equivalent_variance_rank": equivalent_rank,
            "pca_r2_at_matched_variance": matched,
        },
        "random_directions": {
            "n": int(n_random),
            "r2_mean": float(np.nanmean(r2_rand)),
            "r2_sd": float(np.nanstd(r2_rand)),
            "r2_p5": float(np.nanquantile(r2_rand, 0.05)),
            "r2_p95": float(np.nanquantile(r2_rand, 0.95)),
            "var_mean": float(var_rand.mean()),
            "var_sd": float(var_rand.std()),
        },
        "best5": best5,
        "worst5": worst5,
        "n_evaluated": int(finite.sum()),
        "n_r2_below_zero": int((fin_r2 < 0.0).sum()),
        "first_rank_below_zero": first_below,
    }


def analysis_d(
    train_bundle: dict,
    rb_by_trait: dict,
    traits: list[str],
    *,
    n_folds: int,
    seed: int,
    k_lead: int,
    tail_step: int,
    n_random: int,
) -> dict:
    """Per-trait per-direction decomposition at the read-out layer (fold 0)."""
    cx_last = train_bundle["cx_last"]
    v_x = train_bundle["v_x"]
    layers = train_bundle["layers"]
    n = cx_last.shape[0]
    test_idx = PR._cv_folds(n, n_folds, seed)[0]

    out: dict = {
        "note": (
            "held-out per-direction R2 of the full-ridge h on fold 0 of the same "
            "5-fold split; PCA directions from the TRAIN-fold true targets"
        )
    }
    for t in traits:
        li = READ_OUT_LAYER[t]
        col = layers.index(li)
        X = cx_last[:, col, :].numpy().astype(np.float64)
        Y = v_x[:, col, :].numpy().astype(np.float64)
        res = analysis_d_layer(
            X,
            Y,
            rb_by_trait[t][li],
            test_idx,
            k_lead=k_lead,
            tail_step=tail_step,
            n_random=n_random,
            seed=seed,
        )
        res["read_out_layer"] = li
        out[t] = res
        rb = res["r_b"]
        logger.info(
            "D %s L%d: r_B R2=%.3f var-pctile=%.1f eq-rank=%d (matched-var PCA R2 "
            "%.3f) | random dirs R2 %.3f+-%.3f | rank0 R2=%.3f last-rank R2=%.3f",
            t,
            li,
            rb["heldout_r2"],
            rb["variance_percentile_of_pca_spectrum"],
            rb["equivalent_variance_rank"],
            rb["pca_r2_at_matched_variance"]["r2_mean"],
            res["random_directions"]["r2_mean"],
            res["random_directions"]["r2_sd"],
            res["r2_by_rank"][0],
            res["r2_by_rank"][-1],
        )
    return out


# ── figures ────────────────────────────────────────────────────────────────────


def _apply_paper_style() -> None:
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("paper_plots style unavailable (%s); matplotlib default", e)


def make_ladder_figure(a_res: dict, traits: list[str], fig_path: Path) -> None:
    """Left: ladder bars at the read-out layers. Right: cheap-rung + full-h R2 curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 5), layout="tight")

    readout = a_res["readout_layers"]
    group_w = 0.16
    trait_order = sorted(traits, key=lambda t: READ_OUT_LAYER[t])
    for j, rung in enumerate(ALL_RUNGS):
        xs, hs, es = [], [], []
        for i, t in enumerate(trait_order):
            e = readout[str(READ_OUT_LAYER[t])][rung]
            xs.append(i + (j - 2) * group_w)
            hs.append(e["mean"])
            es.append(e["sd"])
        axl.bar(xs, hs, width=group_w * 0.92, yerr=es, capsize=2, label=RUNG_LABEL[rung])
    axl.axhline(0.0, lw=0.8, color="gray")
    axl.set_xticks(range(len(trait_order)))
    axl.set_xticklabels([f"{t}\nL{READ_OUT_LAYER[t]}" for t in trait_order])
    axl.set_ylabel("held-out pooled R2 (5-fold)")
    axl.set_title("Identity-baseline ladder at the read-out layers")
    axl.legend(fontsize=7, loc="upper left")

    curve = a_res["per_layer_curve"]
    lis = sorted(int(k) for k in curve)
    for rung in ("raw_identity", "global_affine", "diag_affine"):
        axr.plot(
            lis,
            [curve[str(li)][rung]["mean"] for li in lis],
            marker="o",
            ms=3,
            lw=1,
            label=RUNG_LABEL[rung],
        )
    fh = a_res["full_h_curve_existing"]
    axr.plot(
        lis,
        [fh[str(li)]["mean"] for li in lis],
        marker="s",
        ms=3,
        lw=1.2,
        label="full ridge h (percontext read1)",
    )
    # symlog: linear in [-1, 1] (where global/diag/full-h live), log below, so the
    # strongly negative raw-identity curve stays VISIBLE instead of clipped away.
    lo = min(curve[str(li)]["raw_identity"]["mean"] for li in lis)
    axr.set_yscale("symlog", linthresh=1.0)
    axr.set_ylim(lo * 1.5, 1.0)
    axr.axhline(0.0, lw=0.8, color="gray")
    axr.set_xlabel("layer")
    axr.set_ylabel("held-out pooled R2 (5-fold)")
    axr.set_title("Ladder rungs by layer")
    axr.legend(fontsize=7, loc="lower center")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


def make_projection_figure(c_res: dict, traits: list[str], fig_path: Path) -> None:
    """Paired bars per trait x mode: pv_raw-vs-oracle vs h-vs-oracle (within-cond
    r with bootstrap CI, left; overall Pearson, right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 5), layout="tight")
    groups = [(t, mode) for t in traits for mode in ("system", "many_shot")]
    labels = [f"{t}\n{mode}" for t, mode in groups]
    w = 0.36

    for ax, which in ((axl, "within"), (axr, "overall")):
        pv_vals, pv_err, h_vals, h_err = [], [], [], []
        for t, mode in groups:
            cell = c_res[t][mode]
            pv = cell["pv_raw_vs_oracle"]
            h = cell["h_vs_oracle_read2"]
            if which == "within":
                pv_vals.append(pv["within_condition_r"])
                pv_err.append(
                    [
                        max(0.0, pv["within_condition_r"] - pv["within_condition_lo"]),
                        max(0.0, pv["within_condition_hi"] - pv["within_condition_r"]),
                    ]
                )
                h_vals.append(h["within_condition_r"])
                h_err.append(
                    [
                        max(0.0, h["within_condition_r"] - h["within_condition_lo"]),
                        max(0.0, h["within_condition_hi"] - h["within_condition_r"]),
                    ]
                )
            else:
                pv_vals.append(pv["overall_pearson"])
                h_vals.append(h["overall_pearson"])
        xs = np.arange(len(groups))
        if which == "within":
            ax.bar(
                xs - w / 2,
                pv_vals,
                w,
                yerr=np.array(pv_err).T,
                capsize=2,
                label="pv_raw <c_last, r_B> (no learning)",
            )
            ax.bar(
                xs + w / 2,
                h_vals,
                w,
                yerr=np.array(h_err).T,
                capsize=2,
                label="h projection <h(c_last), r_B>",
            )
            ax.set_title("Within-condition r vs oracle <v(x), r_B> (95% CI)")
        else:
            ax.bar(xs - w / 2, pv_vals, w, label="pv_raw <c_last, r_B> (no learning)")
            ax.bar(xs + w / 2, h_vals, w, label="h projection <h(c_last), r_B>")
            ax.set_title("Overall Pearson r vs oracle <v(x), r_B>")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Pearson r vs oracle")
        ax.legend(fontsize=8, loc="lower right")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


def make_perdirection_figure(d_res: dict, traits: list[str], fig_path: Path) -> None:
    """Held-out per-direction R2 vs variance rank (log-x), one panel per trait,
    with the train-fold variance share on a twin axis, the random-direction band,
    and r_B placed at its equivalent variance rank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_paper_style()
    fig, axes = plt.subplots(1, len(traits), figsize=(5.2 * len(traits), 4.6), layout="tight")
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, traits, strict=True):
        res = d_res[t]
        ranks1 = np.array(res["ranks_evaluated"]) + 1  # 1-based for log-x
        r2 = np.array(res["r2_by_rank"])
        ax.plot(ranks1, r2, marker=".", ms=3, lw=0.8, label="PCA direction R2 (held-out)")
        rd = res["random_directions"]
        ax.axhspan(
            rd["r2_mean"] - rd["r2_sd"],
            rd["r2_mean"] + rd["r2_sd"],
            alpha=0.15,
            color="gray",
            label=f"random dirs (n={rd['n']}, mean+-sd)",
        )
        ax.axhline(rd["r2_mean"], lw=0.8, color="gray", ls=":")
        rb = res["r_b"]
        ax.scatter(
            [rb["equivalent_variance_rank"] + 1],
            [rb["heldout_r2"]],
            marker="*",
            s=180,
            zorder=6,
            color="crimson",
            label="r_B (at equivalent variance rank)",
        )
        ax.axhline(0.0, lw=0.8, color="black")
        ax.set_xscale("log")
        ax.set_xlabel("variance rank k (1-based, log)")
        ax.set_ylabel("held-out per-direction R2")
        ax.set_title(f"{t} (L{res['read_out_layer']}, fold 0)")
        ax2 = ax.twinx()
        ax2.plot(
            ranks1,
            res["variance_share_by_rank"],
            lw=0.8,
            ls="--",
            color="tab:green",
            alpha=0.6,
            label="train variance share",
        )
        ax2.set_yscale("log")
        ax2.set_ylabel("train-fold variance share (log)", fontsize=8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower left")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


# ── main ───────────────────────────────────────────────────────────────────────


def _load_or_init(out_json: Path) -> dict:
    if out_json.exists():
        with open(out_json) as f:
            return json.load(f)
    return {}


def _build_figures(results: dict, traits: list[str], fig_dir: Path) -> None:
    """Rebuild every figure whose section is present in the results dict."""
    if "A_ladder" in results:
        make_ladder_figure(results["A_ladder"], traits, fig_dir / "identity_baseline_ladder.png")
    if "C_readout" in results:
        make_projection_figure(
            results["C_readout"], traits, fig_dir / "identity_baseline_projection.png"
        )
    if "per_direction" in results:
        make_perdirection_figure(
            results["per_direction"], traits, fig_dir / "h_perdirection_r2.png"
        )


def _console_summary(results: dict, traits: list[str]) -> None:
    """Numbers-first stdout summary of every section present."""
    if "A_ladder" in results:
        print("\n===== A: identity-baseline ladder (read-out layers) =====")
        for t in traits:
            li = READ_OUT_LAYER[t]
            e = results["A_ladder"]["readout_layers"][str(li)]
            bare = e["bare_cosine_dist"]["mean"]
            print(
                f"  {t:14s} L{li:2d}: mean-floor {e['train_mean']['mean']:+.4f} | raw "
                f"{e['raw_identity']['mean']:+.4f} | global {e['global_affine']['mean']:+.4f} | "
                f"diag {e['diag_affine']['mean']:+.4f} | full-h {e['full_h']['mean']:+.4f} | "
                f"genuine (full-diag) "
                f"{results['A_ladder']['per_trait'][t]['genuine_contribution_r2']:+.4f} | "
                f"bare cos {bare:.3f}"
            )
    if "B_weight_decomposition" in results:
        print("\n===== B: W decomposition (read-out layers) =====")
        for li, res in results["B_weight_decomposition"]["per_layer"].items():
            hr = res["heldout_r2"]
            print(
                f"  L{li}: alpha*={res['alpha_star']:.4f} rel-resid="
                f"{res['rel_residual_w_minus_alpha_i']:.4f} | R2 full={hr['full']['mean']:.4f} "
                f"diag={hr['diag_only']['mean']:.4f} offdiag={hr['offdiag_only']['mean']:.4f} | "
                f"|diag|/|offdiag| mean ratio {res['diag_over_offdiag_abs_mean_ratio']:.1f}"
            )
    if "C_readout" in results:
        print("\n===== C: pv_raw vs h against the oracle (within-condition r) =====")
        for t in traits:
            for mode in ("system", "many_shot"):
                cell = results["C_readout"][t][mode]
                pv = cell["pv_raw_vs_oracle"]
                h = cell["h_vs_oracle_read2"]
                print(
                    f"  {t:14s} {mode:10s}: pv_raw {pv['within_condition_r']:.3f} "
                    f"[{pv['within_condition_lo']:.3f}, {pv['within_condition_hi']:.3f}] vs h "
                    f"{h['within_condition_r']:.3f} (delta {cell['h_minus_pvraw_within']:+.3f})"
                )
    if "per_direction" in results:
        print("\n===== D: per-direction R2 of h (fold 0, read-out layers) =====")
        for t in traits:
            res = results["per_direction"][t]
            rb = res["r_b"]
            rd = res["random_directions"]
            print(
                f"  {t:14s} L{res['read_out_layer']:2d}: r_B R2={rb['heldout_r2']:.3f} "
                f"var-pctile={rb['variance_percentile_of_pca_spectrum']:.1f} "
                f"eq-rank={rb['equivalent_variance_rank']} "
                f"matched-var PCA R2={rb['pca_r2_at_matched_variance']['r2_mean']:.3f} | "
                f"random {rd['r2_mean']:.3f}+-{rd['r2_sd']:.3f} | "
                f"below-zero {res['n_r2_below_zero']}/{res['n_evaluated']} "
                f"(first at rank {res['first_rank_below_zero']})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 identity-baseline decomposition.")
    parser.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    parser.add_argument(
        "--collect-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "issue779_hfstage"
        / "issue779_monitoring"
        / "analysis_tensors",
    )
    parser.add_argument(
        "--percontext-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "percontext_recon.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "identity_baseline.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument(
        "--only",
        default="A,B,C,D",
        help="comma-separated subset of {A,B,C,D} to (re)compute; sections merge into --out-json",
    )
    parser.add_argument("--k-lead", type=int, default=200, help="D: leading PCA ranks evaluated")
    parser.add_argument("--tail-step", type=int, default=20, help="D: tail sample every Nth rank")
    parser.add_argument("--n-random", type=int, default=50, help="D: random reference directions")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="rebuild both figures from an existing --out-json (skip all compute)",
    )
    args = parser.parse_args()

    torch.set_num_threads(int(args.n_threads))
    traits = args.traits
    sections = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    with open(args.percontext_json) as f:
        percontext = json.load(f)
    existing_curve = percontext["read1_heldout_recon"]["heldout_r2_vs_layer"]
    existing_read2 = percontext["read2_projection_recon"]

    results = _load_or_init(args.out_json)

    if args.figures_only:
        _build_figures(results, traits, args.fig_dir)
        return 0

    results["read_out_layers"] = {t: READ_OUT_LAYER[t] for t in traits}
    results["metadata"] = C.reproducibility_metadata(
        {
            "script": "issue779_identity_baseline",
            "n_folds": args.n_folds,
            "seed": args.seed,
            "sections": sorted(sections),
        }
    )

    logger.info("Loading pass-B train bundle from %s", args.collect_dir / "pass_b")
    train_bundle = torch.load(
        args.collect_dir / "pass_b" / "train_context_vectors.pt", weights_only=False
    )
    assert train_bundle["cx_last"].shape[1:] == (args.n_layers, C.EXPECTED_HIDDEN), train_bundle[
        "cx_last"
    ].shape

    if "A" in sections:
        logger.info("=== A: identity-baseline ladder (%d-fold CV) ===", args.n_folds)
        results["A_ladder"] = analysis_a(
            train_bundle,
            traits,
            existing_curve,
            n_folds=args.n_folds,
            seed=args.seed,
            n_layers=args.n_layers,
        )
        C.write_json_atomic(args.out_json, results)
        logger.info("Checkpointed A -> %s", args.out_json)

    if "B" in sections:
        logger.info("=== B: weight-matrix decomposition ===")
        results["B_weight_decomposition"] = analysis_b(
            train_bundle, traits, n_folds=args.n_folds, seed=args.seed
        )
        C.write_json_atomic(args.out_json, results)
        logger.info("Checkpointed B -> %s", args.out_json)

    if "C" in sections:
        logger.info("=== C: readout-level test (pv_raw vs oracle) ===")
        rb_by_trait = {
            t: S1._load_rb(args.collect_dir / "r_b", t, args.n_layers, C.EXPECTED_HIDDEN)
            for t in traits
        }
        cells_by_trait = {t: S1.load_eval_cells(args.collect_dir / "pass_a", t) for t in traits}
        results["C_readout"] = analysis_c(
            cells_by_trait,
            rb_by_trait,
            existing_read2,
            traits,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        C.write_json_atomic(args.out_json, results)
        logger.info("Checkpointed C -> %s", args.out_json)

    if "D" in sections:
        logger.info("=== D: per-direction variance-spectrum decomposition ===")
        rb_by_trait_d = {
            t: S1._load_rb(args.collect_dir / "r_b", t, args.n_layers, C.EXPECTED_HIDDEN)
            for t in traits
        }
        results["per_direction"] = analysis_d(
            train_bundle,
            rb_by_trait_d,
            traits,
            n_folds=args.n_folds,
            seed=args.seed,
            k_lead=args.k_lead,
            tail_step=args.tail_step,
            n_random=args.n_random,
        )
        C.write_json_atomic(args.out_json, results)
        logger.info("Checkpointed D -> %s", args.out_json)

    _build_figures(results, traits, args.fig_dir)
    _console_summary(results, traits)

    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
