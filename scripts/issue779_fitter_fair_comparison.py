#!/usr/bin/env python3
"""Issue #779 free re-analysis: validation-selected FAIR fitter comparison (GPU).

Pure refit on the ALREADY-CACHED #779 activation tensors — NO new rollouts, NO
new capture, NO judge/API calls. Answers "is the ceiling on the
``context -> answer-summary`` map (#779's ~0.83 held-out R2 for the linear ridge)
a property of the DATA or of the LINEAR fitter?" by comparing GCV ridge, Nystrom
RBF kernel ridge, and a full-dim MLP under ONE fixed data split, plus scaling
curves and a layer x target heatmap. Two map-INPUT variants are compared: the
last prompt token (``cx_last``) and the mean over prompt tokens (``cx_mean``).

Compute split (orchestrator-directed 2026-07-06):
  * D3 (28-layer x 17-target x {last,mean} ridge heatmap) runs on the VM CPU —
    it is IO-bound on the local capture shards. ``--stage d3 --device cpu``.
  * D1 (fair comparison) + D2 (scaling curves) run on a 1xH100 pod with
    ``--stage d1 d2 --device cuda``. Only the FITS are accelerated; every
    R2/cosine read is computed CPU-side in numpy fp64 exactly per the round's
    ``heldout_recon`` convention.

Stages (``--stage`` / ``--all``; checkpointed per unit, resumable; figures
regenerated from the JSONs at the end):

  d3  Layer x target heatmap (ridge only), BOTH inputs. All 28 layers x 17
      targets (``v_x`` + the 8 pass-2 + 8 cross-layer answer summaries — the
      exact target set of ``issue779_arm_headline_summaries2``) x {last, mean}.
      5-fold CV over the 5000 LMSYS contexts, GCV ridge, ONE shared Gram
      factorization per (input, layer, fold) reused across all 17 targets (the
      cross-kernel + a batched ``V.T @ Y`` GEMM are shared too). Metric:
      held-out variance-weighted R2 (pooled, test-own-mean) + per-target mean
      cosine. -> layer_target_heatmap.json + ffc_layer_target_heatmap.{png,pdf}

  d1  Validation-selected fair fitter comparison (target = ``v(x)`` mean
      profile), BOTH inputs. ONE fixed split of the 5000 contexts: train 3600 /
      val 400 / test 1000, seed 42. Fitters: GCV ridge (full 3584->3584,
      evaluated at ALL 28 layers with a val-selected layer + clean test read),
      Nystrom RBF KRR (1024 landmarks, full-dim; (gamma, lambda) selected on
      val; at 5 layers), full-dim MLP (NO PCA head; (width, lr) selected on val
      at layer 19; run at {19, 26}), and a residual-skip MLP (ridge + MLP on
      the residual — strictly nests the linear map). Metric: variance-weighted
      R2 + mean per-context cosine on TEST, 1000-resample bootstrap 95% CI over
      test contexts. Per fitter x input: per-layer {val_r2, test_r2},
      val_selected_layer, test_r2_at_val_selected_layer.
      -> fair_comparison.json + ffc_fitter_comparison.{png,pdf}

  d2  Scaling curves (layer 19, LAST input by default). n_train in
      {250,500,1000,2000,3600} subsampled from the SAME train split (val/test
      fixed), 3 draws per n (seeds 0,1,2). Fitters: ridge (GCV per fit), KRR
      (reselect gamma/lambda on val per (n,draw)), MLP (the D1 val-selected
      recipe applied at each n — a stated caveat), residual-skip. Metric: test
      R2 per (fitter, n, draw). Conditional extra curves: if ridge's last-input
      val-selected layer != 19, also ridge+KRR at that layer; if ridge's
      mean-input val R2 at the selected layer beats last-input, also a
      ridge-only mean-input curve. -> scaling_curves.json + ffc_scaling_curves.{png,pdf}

Batched by construction: the MLP battery groups every fit by (width, lr) and
trains each partition in ONE padded-bmm multi-group loop (per-group internal-val
early stopping, as in ``issue779_batch2.batched_mlp_fit``); KRR solves all
lambdas off one Nystrom factorization per gamma; ridge shares one eigh (+ the
cross-kernel + a batched target GEMM) across the 17 D3 targets / the val+test
eval sets. No serial per-fit loop over the fit battery.

DOCUMENTED DEVIATIONS (also in ``metadata.deviations``):
  * MLP grid is batched per (width, lr): with widths {512,3584,8192} x lrs
    {1e-3,3e-4} that is 6 bmm training loops (not the aspirational 3-per-width)
    — a single AdamW cannot carry a per-group learning rate, so lr is a
    partition key. Every partition still batches all its (input, layer, n,
    draw, target-type) groups in one loop; no per-fit serial loop remains.
  * The MLP + Nystrom helpers are adapted from ``scripts/issue779_batch2.py``
    (batched trainer) with a ``device`` kwarg added; math verified identical by
    the equivalence gates at run start.
  * D1/D2 ridge selects lambda on the 400 val, NOT GCV: at n_train=3600 ~= H=3584
    (the p~=n regime) GCV's (n-dof)^2 denominator degenerates and pins lambda to
    the grid floor -> test R2 ~ -5 to -8 (numpy-SVD reproduces it, so it is a real
    GCV failure, not a device bug); val selection recovers test R2 ~0.6-0.7. D3's
    5-fold CV keeps GCV (n_train=4000, margin from H large enough; matches #779
    percontext_recon). KRR + MLP use val for (gamma,lambda)/(width,lr). All fitters
    fit their FINAL model on the 3600 train, scored on 1000 test.

Fail loud — NaN is reported, never coerced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + shared-VM thread caps (#847) — BEFORE torch.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_arm_headline as AH  # noqa: E402
import issue779_arm_headline_summaries as AS  # noqa: E402
import issue779_arm_headline_summaries2 as AS2  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc")

# ── config ─────────────────────────────────────────────────────────────────────
INPUT_VARIANTS = ("last", "mean")  # cx_last (last prompt token) / cx_mean (prompt mean)
INPUT_FIELD = {"last": "cx_last", "mean": "cx_mean"}
INPUT_LABEL = {"last": "last prompt token", "mean": "mean over prompt tokens"}

D3_TARGETS = ["v_x", *AS2.P2_SUMMARIES, *AS2.XL_VARIANTS]  # 17
TARGET_LABELS = {"v_x": "mean-response profile", **AS2.VARIANT_LABELS}

FITTERS = ("ridge", "krr", "mlp", "residual_skip")
KRR_LAYERS = (14, 17, 19, 26, 27)
MLP_LAYERS = (19, 26)
MLP_SELECT_LAYER = 19  # pre-chosen from #779's percontext_recon 5-fold curve (menu-limited)
KRR_LANDMARKS = 1024
KRR_LAMBDAS = (1e-3, 1e-1, 1e1)
KRR_GAMMA_MULT = (0.25, 1.0, 4.0)
MLP_WIDTHS = (512, 3584, 8192)  # 8192 restored — GPU makes the batched fit cheap
MLP_LRS = (1e-3, 3e-4)
MLP_WD = 1e-4
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 20
RESIDUAL_MLP_WIDTH = 3584
LAMBDAS = np.logspace(-2, 4, 13)  # ridge GCV grid (matches GramRidge / fit_h)
D2_NS = (250, 500, 1000, 2000, 3600)
D2_DRAWS = (0, 1, 2)
D2_LAYER = 19
SPLIT_SEED = 42
BOOT_N = 1000
CV_FOLDS = 5

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "issue_779"
PASS_B_PATH = PROJECT_ROOT / "data" / "issue_779" / "pass_b" / "train_context_vectors.pt"

# ── round-2 (n10k) config ──────────────────────────────────────────────────────
# The combined corpus = pass_b (5000, byte-for-byte reused) at positions [0, N_PASS_B)
# + the new disjoint LMSYS rows at [N_PASS_B, ...). Val/test are drawn from the
# first-N_PASS_B ids, so they stay byte-identical to round-1 by construction.
N_PASS_B = AS.N_LMSYS  # 5000 — round-1 pass_b corpus size (byte-identical val/test anchor)
N10K_TRAIN = 10000  # round-2 train target (round-1's 3600 train ids + new rows, sampled seed 42)
LAMBDAS_N10K = np.logspace(-3, 6, 19)  # wider ridge grid for n=10000 (2/decade, [1e-3, 1e6])
D2_NS_N10K = (250, 500, 1000, 2000, 3600, 6000, 10000)
DEFAULT_NEW_BUNDLE = PROJECT_ROOT / "data" / "issue_779" / "ffc_n10k" / "new_context_vectors.pt"
DEFAULT_OUT_DIR_N10K = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison-n10k"


def _t(msg: str, t0: float) -> None:
    logger.info("[timing] %s: %.1f s", msg, time.time() - t0)


def _dev(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        torch.set_float32_matmul_precision("high")  # tf32 for f32 matmuls (MLP)
    return torch.device(device)


def _lambda_edge(lam) -> str | None:
    """'low'/'high' when a val-selected ridge lambda sits at an edge of the ACTIVE
    LAMBDAS grid (signals the grid may be too narrow — widen and re-fit), else None."""
    if lam is None or not np.isfinite(lam):
        return None
    if np.isclose(float(lam), float(LAMBDAS[0])):
        return "low"
    if np.isclose(float(lam), float(LAMBDAS[-1])):
        return "high"
    return None


def _sha_ids(ids: np.ndarray) -> str:
    """sha256 of an int index array (order-sensitive) — for downstream verification
    that round-2 val/test are byte-identical to round-1."""
    a = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


# ── metrics (variance-weighted R2 = pooled test-own-mean; mean cosine) ─────────


def _recon_point(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """(variance-weighted R2, mean per-context cosine) — PR._pooled_r2 (SS_tot on
    the test set's own mean, the round's heldout_recon convention) + mean cosine."""
    r2 = PR._pooled_r2(pred, true)
    cos = float(np.nanmean(PR._per_context_cosine(pred, true)))
    return r2, cos


def _bootstrap_recon_ci(pred: np.ndarray, true: np.ndarray, n_boot: int, seed: int) -> dict:
    """Point + 95% percentile-bootstrap CI of R2 and mean cosine over test contexts."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = pred.shape[0]
    res_i = ((true - pred) ** 2).sum(axis=1)
    cos_i = PR._per_context_cosine(pred, true)
    r2_point, cos_point = _recon_point(pred, true)
    rng = np.random.default_rng(seed)
    r2s: list[float] = []
    coss: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss_res = float(res_i[idx].sum())
        t = true[idx]
        ss_tot = float(((t - t.mean(0)) ** 2).sum())
        if ss_tot > 1e-12:
            r2s.append(1.0 - ss_res / ss_tot)
        c = cos_i[idx]
        c = c[np.isfinite(c)]
        if c.size:
            coss.append(float(c.mean()))

    def _ci(pt, boots):
        if not boots:
            return {"point": pt, "lo": float("nan"), "hi": float("nan")}
        return {
            "point": pt,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
        }

    return {"r2": _ci(r2_point, r2s), "mean_cosine": _ci(cos_point, coss), "n_test": int(n)}


# ── device-aware Gram/dual GCV ridge (shared factorization + shared cross-kernel) ─


def _factorize(Xtr_np: np.ndarray, dev: torch.device) -> dict:
    """Standardize X on train stats, eigh the (ntr, ntr) Gram (f64). Reusable
    across targets/eval-sets — the #823 share-the-factorization recipe."""
    Xtr = torch.as_tensor(np.asarray(Xtr_np), dtype=torch.float64, device=dev)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9  # matches GramRidge / fit_h (numpy .std is population)
    Xtr_n = (Xtr - xmu) / xsd
    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    return {
        "xmu": xmu,
        "xsd": xsd,
        "Xtr_n": Xtr_n,
        "w": torch.clamp(w, min=0.0),
        "V": V,
        "ntr": int(Xtr.shape[0]),
        "dev": dev,
    }


def _gcv_solve(fact: dict, Ytr_np: np.ndarray):
    """GCV lambda + VtY + train mean for one target off a shared factorization."""
    Ytr = torch.as_tensor(np.asarray(Ytr_np), dtype=torch.float64, device=fact["dev"])
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    VtY = fact["V"].T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    w, ntr = fact["w"], fact["ntr"]
    best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    return best_lam, VtY, ymu


def _cross_kernel(fact: dict, Xev_np: np.ndarray) -> torch.Tensor:
    """KevV = (Xev_n @ Xtr_n.T) @ V — shared across ALL targets at this eval set."""
    Xev = torch.as_tensor(np.asarray(Xev_np), dtype=torch.float64, device=fact["dev"])
    Xev_n = (Xev - fact["xmu"]) / fact["xsd"]
    return (Xev_n @ fact["Xtr_n"].T) @ fact["V"]


def _apply(
    fact: dict, best_lam: float, VtY: torch.Tensor, ymu: torch.Tensor, KevV: torch.Tensor
) -> np.ndarray:
    filt = 1.0 / (fact["w"] + best_lam)
    return ((KevV * filt) @ VtY + ymu).cpu().numpy()


def _vty_ymu(fact: dict, Ytr_np: np.ndarray):
    """(VtY, ymu) for one target off a shared factorization (lambda-independent)."""
    Ytr = torch.as_tensor(np.asarray(Ytr_np), dtype=torch.float64, device=fact["dev"])
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    ymu = Ytr.mean(0)
    return fact["V"].T @ (Ytr - ymu), ymu


def gram_fit_apply(Xtr, Ytr, X_eval_list, dev, val=None):
    """Fit ridge on (Xtr, Ytr), predict each eval set. One eigh, one target.

    lambda selection: VAL-based (max val R2 over LAMBDAS) when ``val=(Xval, Yval)``
    is given, else GCV. The val path is MANDATORY for the D1/D2 fixed split — at
    ``n_train=3600 ~= H=3584`` GCV's ``(n_train - dof)^2`` denominator degenerates
    and pins lambda to the grid floor (test R2 ~= -5 to -8; verified numpy-SVD
    reproduces it), whereas val selection recovers the correct heavily-regularized
    lambda (test R2 ~0.6-0.7). D3's 5-fold CV keeps GCV (n_train=4000, margin from
    H is large enough that GCV stays sane — matches #779 percontext_recon).
    Returns (list_of_preds, selected_lambda)."""
    fact = _factorize(Xtr, dev)
    if val is None:
        best_lam, VtY, ymu = _gcv_solve(fact, Ytr)
        return [
            _apply(fact, best_lam, VtY, ymu, _cross_kernel(fact, E)) for E in X_eval_list
        ], best_lam
    Xval, Yval = val
    VtY, ymu = _vty_ymu(fact, Ytr)
    KvalV = _cross_kernel(fact, Xval)
    best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
    for lam in LAMBDAS:
        vr2 = PR._pooled_r2(_apply(fact, float(lam), VtY, ymu, KvalV), Yval)
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    return [_apply(fact, best_lam, VtY, ymu, _cross_kernel(fact, E)) for E in X_eval_list], best_lam


def gram_cv_recon(X, targets: dict, n_folds: int, seed: int, dev) -> dict:
    """5-fold held-out R2 + mean cosine per target. ONE factorization per fold
    (shared eigh + shared cross-kernel + per-target GCV) across all targets."""
    n = len(X)
    folds = PR._cv_folds(n, n_folds, seed)
    acc = {k: {"r2": [], "cos": []} for k in targets}
    for test_idx in folds:
        m = np.ones(n, dtype=bool)
        m[test_idx] = False
        fact = _factorize(X[m], dev)
        KevV = _cross_kernel(fact, X[test_idx])  # shared across the 17 targets
        for k, Y in targets.items():
            best_lam, VtY, ymu = _gcv_solve(fact, Y[m])
            pred = _apply(fact, best_lam, VtY, ymu, KevV)
            acc[k]["r2"].append(PR._pooled_r2(pred, Y[test_idx]))
            acc[k]["cos"].append(float(np.nanmean(PR._per_context_cosine(pred, Y[test_idx]))))
    return {
        k: {
            "r2_mean": float(np.mean(v["r2"])),
            "r2_sd": float(np.std(v["r2"])),
            "r2_folds": [float(x) for x in v["r2"]],
            "cos_mean": float(np.mean(v["cos"])),
            "cos_sd": float(np.std(v["cos"])),
            "n": int(n),
        }
        for k, v in acc.items()
    }


def _gate_ridge(dev) -> dict:
    """gram_fit_apply == AH.GramRidge (CPU f64 canonical) within tolerance.

    Both sides use the ACTIVE LAMBDAS grid so this isolates DEVICE numerics — the
    n10k grid is wider than GramRidge's default logspace(-2,4,13), and without the
    explicit ``lambdas=`` GCV would pick a different lambda per side (grid mismatch,
    not a device bug: rel ~2e-4 on CPU)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 400))
    Y = X @ rng.standard_normal((400, 128)) + 0.01 * rng.standard_normal((600, 128))
    pred_dev = gram_fit_apply(X[:500], Y[:500], [X[500:]], dev)[0][0]
    pred_ref = AH.GramRidge(X[:500], lambdas=LAMBDAS).predict(Y[:500], X[500:])
    rel = float(np.max(np.abs(pred_dev - pred_ref)) / (np.max(np.abs(pred_ref)) + 1e-12))
    tol = 1e-4 if dev.type == "cuda" else 1e-6
    assert rel < tol, f"ridge device gate FAILED: rel {rel:.2e} > {tol:.0e} on {dev}"
    logger.info("ridge device gate PASS on %s (rel-vs-GramRidge %.2e)", dev, rel)
    return {"rel_vs_gramridge": rel, "device": str(dev), "tol": tol}


# ── device-aware Nystrom RBF kernel ridge (adapted from issue779_batch2) ────────


def nystrom_features(X, landmarks, gamma, dev, eig_floor=1e-10) -> torch.Tensor:
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev)
    Zt = torch.as_tensor(np.asarray(landmarks), dtype=torch.float64, device=dev)
    K_nm = torch.exp(-gamma * torch.cdist(Xt, Zt) ** 2)
    K_mm = torch.exp(-gamma * torch.cdist(Zt, Zt) ** 2)
    w, V = torch.linalg.eigh(K_mm)
    w = torch.clamp(w, min=eig_floor)
    inv_sqrt = V @ torch.diag(w.rsqrt()) @ V.T
    return K_nm @ inv_sqrt  # (n, m) on dev


def median_heuristic_gamma(X, rng, n_sub=2000) -> float:
    sub = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]
    St = torch.as_tensor(sub, dtype=torch.float64)
    d2 = (torch.cdist(St, St) ** 2).numpy()
    med = float(np.median(d2[np.triu_indices_from(d2, k=1)]))
    assert med > 0, med
    return 1.0 / med


def _feature_ridge_multi_lambda(Phi_tr: torch.Tensor, Y_tr, Phi_eval_list, lambdas):
    """Ridge on precomputed features at each lambda off ONE eigh of Phi^T Phi."""
    Y = torch.as_tensor(np.asarray(Y_tr), dtype=torch.float64, device=Phi_tr.device)
    ymu = Y.mean(0)
    Yc = Y - ymu
    a, Q = torch.linalg.eigh(Phi_tr.T @ Phi_tr)
    a = torch.clamp(a, min=0.0)
    QtPtY = Q.T @ (Phi_tr.T @ Yc)
    out = []
    for lam in lambdas:
        W = Q @ (QtPtY / (a + lam)[:, None])
        out.append([((E @ W) + ymu).cpu().numpy() for E in Phi_eval_list])
    return out


def krr_select_predict(
    Xtr, Ytr, Xval, Yval, Xte, *, gamma_mult, lambdas, m_landmarks, seed, dev
) -> dict:
    """Nystrom RBF KRR, (gamma, lambda) selected by val R2. Landmarks = seeded
    subsample of Xtr shared across gammas; all lambdas solved off one factorization."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    rng = np.random.default_rng(seed)
    base_gamma = median_heuristic_gamma(Xtr, np.random.default_rng(seed + 1))
    m = min(m_landmarks, len(Xtr))
    lm = Xtr[rng.choice(len(Xtr), size=m, replace=False)]
    grid, best = [], None
    for gm in gamma_mult:
        gamma = base_gamma * gm
        Phi_tr = nystrom_features(Xtr, lm, gamma, dev)
        Phi_val = nystrom_features(Xval, lm, gamma, dev)
        Phi_te = nystrom_features(Xte, lm, gamma, dev)
        preds = _feature_ridge_multi_lambda(Phi_tr, Ytr, [Phi_val, Phi_te], lambdas)
        for li, lam in enumerate(lambdas):
            pred_val, pred_te = preds[li]
            val_r2 = PR._pooled_r2(pred_val, Yval)
            grid.append(
                {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                }
            )
            if best is None or (np.isfinite(val_r2) and val_r2 > best["val_r2"]):
                best = {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                    "pred_te": pred_te,
                }
    assert best is not None
    return {
        "pred_te": best["pred_te"],
        "selected": {k: best[k] for k in ("gamma_mult", "gamma", "lambda", "val_r2")},
        "base_gamma": float(base_gamma),
        "m_landmarks": int(m),
        "val_grid": grid,
    }


# ── device-aware batched multi-group MLP (adapted from issue779_batch2) ─────────


def _pca_basis_eigh(Y, k):  # for the equivalence gate only (full-dim MLP uses no PCA)
    Yc = torch.as_tensor(Y, dtype=torch.float64)
    mu = Yc.mean(0)
    Yc = Yc - mu
    n, h = Yc.shape
    if n >= h:
        _w, V = torch.linalg.eigh(Yc.T @ Yc)
        comps = V[:, -k:].flip(-1).T
    else:
        w, U = torch.linalg.eigh(Yc @ Yc.T)
        w_top = torch.clamp(w[-k:].flip(0), min=1e-12)
        comps = (Yc.T @ U[:, -k:].flip(-1) / w_top.sqrt()).T
    comps = comps / comps.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return mu.numpy(), comps.numpy()


@dataclass
class MLPGroup:
    key: tuple
    X: np.ndarray
    Y: np.ndarray
    width: int
    lr: float


class MLPResult:
    def __init__(self, params, xmu, xsd, ymu, comps, p_g, epochs_ran, best_val):
        self.params, self.xmu, self.xsd = params, xmu, xsd
        self.ymu, self.comps, self.p_g = ymu, comps, p_g
        self.epochs_ran, self.best_val = epochs_ran, best_val

    def predict(self, X_new):
        Xn = (np.asarray(X_new, dtype=np.float32) - self.xmu) / self.xsd
        W1, b1, W2, b2 = self.params
        with torch.no_grad():
            h = torch.nn.functional.gelu(torch.from_numpy(Xn) @ W1.T + b1)
            out = (h @ W2[: self.p_g].T + b2[: self.p_g]).numpy()
        return (out @ self.comps + self.ymu) if self.comps is not None else (out + self.ymu)


def batched_mlp_fit(
    groups,
    *,
    hidden,
    lr,
    wd=MLP_WD,
    max_epochs=MLP_MAX_EPOCHS,
    patience=MLP_PATIENCE,
    val_frac=0.1,
    seed=42,
    dev=None,
    basis_fn=None,
) -> dict:
    """Train all groups' MLPs as ONE padded-bmm AdamW loop (per-group internal-val
    early stop). Adapted from ``issue779_batch2.batched_mlp_fit`` with a device
    kwarg; full-dim head when basis_fn is None. Groups share (hidden, lr)."""
    dev = dev or torch.device("cpu")
    assert groups
    d_in = groups[0].X.shape[1]
    G = len(groups)
    prep = []
    for g in groups:
        Xtr = np.asarray(g.X, dtype=np.float32)
        Ytr = np.asarray(g.Y, dtype=np.float32)
        if Ytr.ndim == 1:
            Ytr = Ytr[:, None]
        n = Xtr.shape[0]
        xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xn = (Xtr - xmu) / xsd
        if basis_fn is None:  # full-dim, NO PCA (the D1/D2 path)
            ymu, comps, T = Ytr.mean(0), None, (Ytr - Ytr.mean(0))
        else:  # PCA head — used only by the equivalence gate
            ymu64, comps = basis_fn(Ytr.astype(np.float64), 8)
            ymu, T = ymu64, ((Ytr.astype(np.float64) - ymu64) @ comps.T).astype(np.float32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, round(val_frac * n))
        prep.append(
            {
                "n": n,
                "p": T.shape[1],
                "Xn": Xn,
                "T": T.astype(np.float32),
                "xmu": xmu,
                "xsd": xsd,
                "ymu": ymu,
                "comps": comps,
                "val_idx": perm[:n_val],
                "tr_idx": perm[n_val:],
            }
        )
    p_max = max(pp["p"] for pp in prep)
    n_tr_max = max(len(pp["tr_idx"]) for pp in prep)
    n_val_max = max(len(pp["val_idx"]) for pp in prep)
    Xp = torch.zeros((G, n_tr_max, d_in), device=dev)
    Tp = torch.zeros((G, n_tr_max, p_max), device=dev)
    wtr = torch.zeros((G, n_tr_max, p_max), device=dev)
    Xv = torch.zeros((G, n_val_max, d_in), device=dev)
    Tv = torch.zeros((G, n_val_max, p_max), device=dev)
    wva = torch.zeros((G, n_val_max, p_max), device=dev)
    denom_tr = torch.zeros(G, device=dev)
    denom_val = torch.zeros(G, device=dev)
    for gi, pp in enumerate(prep):
        p, ntr, nva = pp["p"], len(pp["tr_idx"]), len(pp["val_idx"])
        Xp[gi, :ntr] = torch.from_numpy(pp["Xn"][pp["tr_idx"]]).to(dev)
        Tp[gi, :ntr, :p] = torch.from_numpy(pp["T"][pp["tr_idx"]]).to(dev)
        wtr[gi, :ntr, :p] = 1.0
        Xv[gi, :nva] = torch.from_numpy(pp["Xn"][pp["val_idx"]]).to(dev)
        Tv[gi, :nva, :p] = torch.from_numpy(pp["T"][pp["val_idx"]]).to(dev)
        wva[gi, :nva, :p] = 1.0
        denom_tr[gi], denom_val[gi] = float(ntr * p), float(nva * p)
    W1 = torch.empty((G, hidden, d_in), device=dev)
    b1 = torch.empty((G, hidden), device=dev)
    W2 = torch.zeros((G, p_max, hidden), device=dev)
    b2 = torch.zeros((G, p_max), device=dev)
    for gi, pp in enumerate(prep):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, pp["p"])
        )
        W1[gi] = net[0].weight.detach().to(dev)
        b1[gi] = net[0].bias.detach().to(dev)
        W2[gi, : pp["p"]] = net[2].weight.detach().to(dev)
        b2[gi, : pp["p"]] = net[2].bias.detach().to(dev)
    for w in (W1, b1, W2, b2):
        w.requires_grad_(True)
    opt = torch.optim.AdamW([W1, b1, W2, b2], lr=lr, weight_decay=wd)
    best_val = torch.full((G,), float("inf"), device=dev)
    bad = torch.zeros(G, dtype=torch.long, device=dev)
    frozen = torch.zeros(G, dtype=torch.bool, device=dev)
    best_state = [None] * G
    epochs_ran = np.zeros(G, dtype=int)
    active_f = torch.ones(G, device=dev)
    t0 = time.time()
    for ep in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        h1 = torch.nn.functional.gelu(torch.baddbmm(b1.unsqueeze(1), Xp, W1.transpose(1, 2)))
        out = torch.baddbmm(b2.unsqueeze(1), h1, W2.transpose(1, 2))
        loss_pg = (((out - Tp) ** 2) * wtr).sum(dim=(1, 2)) / denom_tr
        (loss_pg * active_f).sum().backward()
        opt.step()
        with torch.no_grad():
            h1e = torch.nn.functional.gelu(torch.baddbmm(b1.unsqueeze(1), Xv, W1.transpose(1, 2)))
            oute = torch.baddbmm(b2.unsqueeze(1), h1e, W2.transpose(1, 2))
            val_pg = (((oute - Tv) ** 2) * wva).sum(dim=(1, 2)) / denom_val
        improved = (val_pg < best_val - 1e-6) & (~frozen)
        for gi in torch.nonzero(improved).ravel().tolist():
            best_state[gi] = tuple(tt[gi].detach().cpu().clone() for tt in (W1, b1, W2, b2))
        best_val = torch.where(improved, val_pg, best_val)
        bad = torch.where(improved, torch.zeros_like(bad), bad + (~frozen).long())
        frozen |= (bad >= patience) & (~frozen)
        active_f = (~frozen).float()
        epochs_ran[(~frozen).cpu().numpy()] = ep + 1
        if frozen.all():
            break
        if ep % 50 == 0:
            logger.info(
                "[mlp-bmm h=%d lr=%.0e] ep %d: %d/%d active (%.2fs/ep)",
                hidden,
                lr,
                ep,
                int((~frozen).sum()),
                G,
                (time.time() - t0) / (ep + 1),
            )
    results = {}
    for gi, (g, pp) in enumerate(zip(groups, prep, strict=True)):
        st = best_state[gi] or tuple(tt[gi].detach().cpu() for tt in (W1, b1, W2, b2))
        results[g.key] = MLPResult(
            st,
            pp["xmu"],
            pp["xsd"],
            pp["ymu"],
            pp["comps"],
            pp["p"],
            int(epochs_ran[gi]),
            float(best_val[gi]),
        )
    return results


def _gate_mlp(dev) -> dict:
    """G=1 batched trainer (PCA head) vs fit_h.mlp_fit_predict — prediction agreement."""
    rng = np.random.default_rng(7)
    n, d, p = 260, 96, 128
    Wt = rng.standard_normal((d, p)) * 0.3
    X = rng.standard_normal((n + 60, d)).astype(np.float32)
    Y = (np.tanh(X @ Wt.astype(np.float32)) + 0.05 * rng.standard_normal((n + 60, p))).astype(
        np.float32
    )
    ref = F.mlp_fit_predict(X[:n], Y[:n], X[n:], pca_k=8, num_threads=8, max_epochs=60)
    res = batched_mlp_fit(
        [MLPGroup(("g",), X[:n], Y[:n], 512, 1e-3)],
        hidden=512,
        lr=1e-3,
        max_epochs=60,
        dev=dev,
        basis_fn=_pca_basis_eigh,
    )[("g",)]
    agree = PR._pooled_r2(res.predict(X[n:]), ref)
    assert agree > 0.99, f"MLP batched gate FAILED: agreement R2 {agree:.4f}"
    logger.info("MLP batched gate PASS on %s (agreement R2 %.4f)", dev, agree)
    return {"agreement_r2": float(agree), "device": str(dev)}


def run_mlp_battery(groups: list[MLPGroup], *, dev, max_epochs) -> dict:
    """Partition groups by (width, lr); ONE padded-bmm loop per partition."""
    parts: dict[tuple, list[MLPGroup]] = defaultdict(list)
    for g in groups:
        parts[(g.width, g.lr)].append(g)
    out: dict = {}
    for (w, lr), gs in parts.items():
        logger.info("[mlp-battery] partition width=%d lr=%.0e: %d groups", w, lr, len(gs))
        out.update(batched_mlp_fit(gs, hidden=w, lr=lr, max_epochs=max_epochs, dev=dev))
    return out


# ── data loading ────────────────────────────────────────────────────────────────


def _mmap_load(path):
    try:
        return torch.load(path, mmap=True, weights_only=False, map_location="cpu")
    except RuntimeError as e:
        logger.warning("mmap load failed for %s (%s); full load", path, e)
        return torch.load(path, weights_only=False, map_location="cpu")


def load_pass_b(path=None):
    b = _mmap_load(path or PASS_B_PATH)
    for fld in ("cx_last", "cx_mean", "v_x"):
        assert fld in b, f"pass_b missing {fld}"
        assert b[fld].shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), (fld, b[fld].shape)
    return b


def corpus_len(bundle) -> int:
    """Row count of a plain (single) or combined (n10k) corpus bundle."""
    if bundle.get("_combined"):
        return int(bundle["n_old"] + bundle["n_new"])
    return int(bundle["cx_last"].shape[0])


def input_layer(bundle, variant: str, li: int) -> np.ndarray:
    fld = INPUT_FIELD[variant]
    if bundle.get("_combined"):  # n10k: concat pass_b (old) + new-bundle rows at layer li
        oc = bundle["pb"]["layers"].index(li)
        nc = bundle["nb"]["layers"].index(li)
        old = bundle["pb"][fld][:, oc, :].to(torch.float32).numpy()
        new = bundle["nb"][fld][:, nc, :].to(torch.float32).numpy()
        return np.concatenate([old, new], axis=0)
    col = bundle["layers"].index(li)
    return bundle[fld][:, col, :].to(torch.float32).numpy()


def target_vx(bundle, li: int) -> np.ndarray:
    if bundle.get("_combined"):
        oc = bundle["pb"]["layers"].index(li)
        nc = bundle["nb"]["layers"].index(li)
        old = bundle["pb"]["v_x"][:, oc, :].to(torch.float32).numpy()
        new = bundle["nb"]["v_x"][:, nc, :].to(torch.float32).numpy()
        return np.concatenate([old, new], axis=0)
    col = bundle["layers"].index(li)
    return bundle["v_x"][:, col, :].to(torch.float32).numpy()


def load_p2_lmsys_all_layers(cap2: Path, layers: list[int], n_ctx: int = AS.N_LMSYS) -> np.ndarray:
    shards = sorted(cap2.glob("lmsys_summaries_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no pass-2 lmsys shards under {cap2}")
    k = len(AS2.P2_SUMMARIES)
    S, seen = None, np.zeros(n_ctx, dtype=bool)
    for sp in shards:
        blob = torch.load(sp, mmap=True, weights_only=False, map_location="cpu")
        assert list(blob["summaries"]) == list(AS2.P2_SUMMARIES), (sp.name, blob["summaries"])
        cols = [blob["layers"].index(li) for li in layers]
        arr = blob["summ"][:, :, cols, :].to(torch.float16).numpy()  # (n_rows, k, n_layers, H)
        if S is None:
            S = np.full((n_ctx, k, len(layers), arr.shape[-1]), np.nan, dtype=np.float16)
        idx = np.array([ci for ci, _ri in blob["index"]])
        assert not seen[idx].any(), sp.name
        S[idx] = arr
        seen[idx] = True
    assert S is not None and seen.all(), f"pass-2 lmsys: {int((~seen).sum())} rows missing"
    return S


def load_d3_targets(cap1, cap2, layers, target_keys):
    want_p2 = [t for t in target_keys if t in AS2.P2_SUMMARIES]
    want_xl = [t for t in target_keys if t in AS2.XL_VARIANTS]
    XLl, v1l = AS2.load_crosslayer(cap1, "lmsys", AS.N_LMSYS, 1)
    mask = v1l[:, 0, :].all(axis=1)  # pass-1 joint validity (pass-2 + v_x valid everywhere)
    xl_targets = {t: XLl[t][:, 0, :][mask].astype(np.float32) for t in want_xl}
    S2, s2_idx = None, {}
    if want_p2:
        S2 = load_p2_lmsys_all_layers(cap2, layers)
        s2_idx = {li: i for i, li in enumerate(layers)}
    return S2, s2_idx, xl_targets, mask


def load_d3_targets_n10k(cap1, cap2, nb, layers, target_keys, n_old):
    """D3 targets over the COMBINED 11500-context corpus: OLD (first ``n_old``)
    rows from the round-1 ``final_token_capture`` shards (the exact round-1
    loaders, so old-half targets are byte-identical), NEW rows from the new
    bundle's ``summ_p1`` (cross-layer aggregates) + ``summ_p2`` (pass-2). Row
    order matches the combined corpus (old ci-order at [0,n_old), new-bundle
    order at [n_old,...)). Returns the SAME (S2, s2_idx, xl_targets, mask)
    interface as ``load_d3_targets`` so ``run_d3`` is unchanged.

    ``S2`` (pass-2) and ``v_x`` stay UNMASKED (masked per-layer in run_d3);
    ``xl_targets`` are returned already masked (matching round-1)."""
    want_p2 = [t for t in target_keys if t in AS2.P2_SUMMARIES]
    want_xl = [t for t in target_keys if t in AS2.XL_VARIANTS]

    # OLD half — round-1 loaders (ci-order over [0, n_old)).
    XLl, v1l = AS2.load_crosslayer(cap1, "lmsys", n_old, 1)
    old_mask = v1l[:, 0, :].all(axis=1)  # pass-1 joint validity
    old_xl = {t: XLl[t][:, 0, :].astype(np.float32) for t in want_xl}  # (n_old, H), UNmasked
    old_S2 = load_p2_lmsys_all_layers(cap2, layers, n_ctx=n_old) if want_p2 else None

    # NEW half — from the new bundle (kept-row order = combined [n_old, ...)).
    n_new = int(nb["cx_last"].shape[0])
    valid_p1 = nb["valid_p1"].numpy().astype(bool)  # (n_new, 4)
    assert valid_p1.shape == (n_new, len(AS.SUMMARIES)), valid_p1.shape
    new_mask = valid_p1.all(axis=1)  # (n_new,)
    new_S2 = None
    if want_p2:
        p2cols = [nb["layers"].index(li) for li in layers]
        # (n_new, 8, n_layers, H) fp16 — matches load_p2_lmsys_all_layers dtype
        new_S2 = nb["summ_p2"][:, :, p2cols, :].to(torch.float16).numpy()
    new_xl = {}
    if want_xl:
        sp1 = nb["summ_p1"]  # (n_new, 4, 28, H) fp16, mmap
        for t in want_xl:
            base = t.split("_", 1)[1]
            si = list(AS.SUMMARIES).index(base)
            arr = sp1[:, si, :, :].to(torch.float32)  # (n_new, 28, H); reduce over the LAYER axis
            if t.startswith("xlmean_"):
                new_xl[t] = arr.mean(dim=1).numpy()  # matches load_crosslayer chunk.mean(dim=2)
            elif t.startswith("xlmax_"):
                new_xl[t] = arr.max(dim=1).values.numpy()  # matches chunk.max(dim=2).values
            else:  # pragma: no cover — XL_VARIANTS only carry xlmean_/xlmax_ prefixes
                raise ValueError(f"unexpected xl variant {t}")

    mask = np.concatenate([old_mask, new_mask])  # (n_combined,)
    S2 = np.concatenate([old_S2, new_S2], axis=0) if want_p2 else None
    s2_idx = {li: i for i, li in enumerate(layers)} if want_p2 else {}
    xl_targets = {t: np.concatenate([old_xl[t], new_xl[t]], axis=0)[mask] for t in want_xl}
    for t in xl_targets:  # cross-layer targets are masked here; must be finite
        assert np.isfinite(xl_targets[t]).all(), t
    return S2, s2_idx, xl_targets, mask


# ── fixed split + metadata ──────────────────────────────────────────────────────


def fixed_split(n_ctx, n_train, n_val, n_test, seed):
    assert n_train + n_val + n_test <= n_ctx, (n_train, n_val, n_test, n_ctx)
    perm = np.random.default_rng(seed).permutation(n_ctx)
    return (
        np.sort(perm[n_test + n_val : n_test + n_val + n_train]),
        np.sort(perm[n_test : n_test + n_val]),
        np.sort(perm[:n_test]),
    )


def load_combined_corpus(pass_b_path, new_bundle_path, n_pass_b):
    """Lazily combine pass_b (positions [0, n_old)) + the new bundle
    (positions [n_old, ...)). Returns a ``_combined`` bundle dict whose halves
    stay mmap'd (input_layer/target_vx concat per-layer slices), plus the raw
    new-bundle dict for D3's summ_p1/summ_p2/valid_p1."""
    pb = load_pass_b(pass_b_path)
    n_old = int(pb["cx_last"].shape[0])
    assert n_old == n_pass_b, (
        f"pass_b has {n_old} rows but --n-pass-b={n_pass_b}; the round-1 split (and its "
        "byte-identical val/test) is anchored on the pass_b row count"
    )
    nb = _mmap_load(new_bundle_path)
    for fld in ("cx_last", "cx_mean", "v_x"):
        assert fld in nb, f"new bundle missing {fld}"
        assert nb[fld].shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), (fld, nb[fld].shape)
    assert list(nb["layers"]) == list(pb["layers"]), "new bundle layers != pass_b layers"
    assert list(nb.get("p1_summaries", [])) == list(AS.SUMMARIES), (
        "new bundle p1_summaries order != AS.SUMMARIES",
        nb.get("p1_summaries"),
    )
    assert list(nb.get("p2_summaries", [])) == list(AS2.P2_SUMMARIES), (
        "new bundle p2_summaries order != AS2.P2_SUMMARIES",
        nb.get("p2_summaries"),
    )
    n_new = int(nb["cx_last"].shape[0])
    combined = {
        "_combined": True,
        "pb": pb,
        "nb": nb,
        "n_old": n_old,
        "n_new": n_new,
        "layers": list(pb["layers"]),
        "source": {"pass_b": str(pass_b_path), "new_bundle": str(new_bundle_path)},
    }
    logger.info(
        "combined corpus: %d old (pass_b) + %d new = %d contexts", n_old, n_new, n_old + n_new
    )
    return combined, nb


def build_n10k_split(combined, args):
    """(train, val, test) for the combined corpus. val/test = round-1's split
    over the first ``n_old`` ids (indices < n_old, same pass_b rows ⇒ BYTE-IDENTICAL
    to round-1); train = round-1's train ids + all new ids, sampled to
    ``args.n_train`` (seed). Returns (train, val, test, diag)."""
    n_old, n_new = combined["n_old"], combined["n_new"]
    assert args.n_val + args.n_test < n_old, (
        f"val+test ({args.n_val}+{args.n_test}) must fit in the pass_b half (n_old={n_old}) — "
        "val/test are drawn from the first-n_old ids to stay byte-identical to round-1"
    )
    r1_train_n = n_old - args.n_val - args.n_test  # 3600 in production (5000-400-1000)
    r1_train, val, test = fixed_split(n_old, r1_train_n, args.n_val, args.n_test, args.seed)
    new_idx = np.arange(n_old, n_old + n_new)
    pool = np.concatenate([r1_train, new_idx])
    n_target = min(args.n_train, len(pool))
    if n_target < args.n_train:
        logger.warning(
            "train pool has only %d ids (%d round-1 + %d new) < target %d; using all %d",
            len(pool),
            len(r1_train),
            n_new,
            args.n_train,
            n_target,
        )
    sel = np.random.default_rng(args.seed).choice(len(pool), size=n_target, replace=False)
    train = np.sort(pool[sel])
    # Byte-identical tripwire (production regime only): recompute round-1's exact
    # fixed_split call and assert val/test indices match to the id.
    prod = (
        n_old == N_PASS_B and args.n_val == 400 and args.n_test == 1000 and args.seed == SPLIT_SEED
    )
    byte_identical = None
    if prod:
        _rtr, r1_val, r1_test = fixed_split(N_PASS_B, N_PASS_B - 400 - 1000, 400, 1000, SPLIT_SEED)
        assert np.array_equal(val, r1_val) and np.array_equal(test, r1_test), (
            "n10k val/test are NOT byte-identical to round-1 fixed_split(5000,3600,400,1000,42)"
        )
        byte_identical = True
    assert set(val).isdisjoint(train) and set(test).isdisjoint(train), "train overlaps val/test"
    assert (val < n_old).all() and (test < n_old).all(), "val/test must index the pass_b half only"
    diag = {
        "n_old": int(n_old),
        "n_new": int(n_new),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_train_target": int(args.n_train),
        "pool_size": len(pool),
        "r1_train_ids": len(r1_train),
        "seed": int(args.seed),
        "val_sha256": _sha_ids(val),
        "test_sha256": _sha_ids(test),
        "train_sha256": _sha_ids(train),
        "val_test_byte_identical_round1": byte_identical,
        "round1_dir": str(DEFAULT_OUT_DIR),
        "note": (
            "val/test = round-1 fixed_split(pass_b, pass_b-n_val-n_test, n_val, n_test, seed) — "
            "indices < n_old, the SAME pass_b rows, so byte-identical to round-1 in the "
            "production regime (n_old=5000, n_val=400, n_test=1000, seed=42). train pool = "
            "round-1's train ids + all new disjoint LMSYS ids, sampled to n_train_target (seed)."
        ),
        "train_ids": train.tolist(),
        "val_ids": val.tolist(),
        "test_ids": test.tolist(),
    }
    return train, val, test, diag


def resolve_split(args, ctx):
    """(train, val, test). n10k: the precomputed byte-identical-val/test combined
    split (ctx['split']). single: fixed_split over the (optionally capped) corpus
    — UNCHANGED round-1 behavior."""
    if getattr(args, "corpus_mode", "single") == "n10k":
        return ctx["split"]
    bundle = ctx["bundle"]
    n_ctx = corpus_len(bundle)
    if args.max_contexts:
        n_ctx = min(n_ctx, args.max_contexts)
    return fixed_split(n_ctx, args.n_train, args.n_val, args.n_test, args.seed)


def _pass_b_meta():
    out = {}
    try:
        st = PASS_B_PATH.stat()
        out.update(
            size_bytes=int(st.st_size),
            mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
        )
    except OSError as e:
        out["stat_error"] = str(e)
    try:
        b = torch.load(PASS_B_PATH, mmap=True, weights_only=False, map_location="cpu")
        out["bundle_metadata"] = b.get("metadata", {})
        out["source"] = b.get("source")
    except Exception as e:
        out["bundle_read_error"] = str(e)
    return out


def _base_metadata(stage, args, extra):
    b = C.reproducibility_metadata({"script": "issue779_fitter_fair_comparison", "stage": stage})
    b.update(
        {
            "device": args.device,
            "thread_caps": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "seed": args.seed,
            "input_variants": list(args.input_variants),
            "data_provenance": {
                "pass_b": {"path": str(PASS_B_PATH), **_pass_b_meta()},
                "capture_dir": str(args.capture_dir),
                "p2_dir": str(args.p2_dir),
            },
            "deviations": [
                "MLP grid batched per (width,lr) partition (one AdamW cannot carry per-group lr); "
                "each partition batches all its (input,layer,n,draw,type) groups in one bmm loop.",
                "MLP + Nystrom helpers adapted from issue779_batch2.py with a device kwarg; math "
                "verified by the run-start equivalence gates.",
                "D1/D2 ridge selects lambda on the 400 val (NOT GCV): at n_train=3600 ~= H=3584 "
                "GCV's (n-dof)^2 denominator degenerates and pins lambda to the grid floor, "
                "test R2 ~ -5 to -8 (numpy-SVD reproduces it); val selection recovers ~0.6-0.7. "
                "D3's 5-fold CV keeps GCV (n_train=4000, margin from H OK; matches #779). "
                "KRR/MLP use val for (gamma,lambda)/(width,lr). Final fit on 3600 train.",
            ],
        }
    )
    b["lambda_grid"] = {
        "n": len(LAMBDAS),
        "min": float(LAMBDAS[0]),
        "max": float(LAMBDAS[-1]),
        "log10": [round(float(np.log10(x)), 3) for x in LAMBDAS],
    }
    if getattr(args, "corpus_mode", "single") == "n10k":
        b["corpus_mode"] = "n10k"
        b["new_bundle"] = str(args.new_bundle)
        b["n_train_target"] = int(args.n_train)
        b["cross_round_comparability"] = (
            "n=10000 rerun of 'fitter-fair-comparison'. val/test are BYTE-IDENTICAL to round-1 "
            "(same fixed_split(5000,3600,400,1000,42) indices <5000 = the same pass_b rows; see "
            "n10k_split.json val_sha256/test_sha256). Train pool = round-1's 3600 train ids + the "
            "new disjoint LMSYS rows, sampled to n_train (seed 42). LAMBDAS widened to [1e-3,1e6] "
            "(round-1 was [1e-2,1e4]); D2 axis extended through 10000; D3 recomputed over the "
            f"combined 11500-context corpus. Round-1 dir: {DEFAULT_OUT_DIR.name}."
        )
    else:
        b["corpus_mode"] = "single"
    b.update(extra)
    return b


# ── stage D3: layer x target heatmap (both inputs) ─────────────────────────────


def run_d3(args, ctx):
    out_path = args.out_dir / "layer_target_heatmap.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    dev = _dev(args.device)
    layers, tkeys = args.d3_layers, args.d3_targets
    t0 = time.time()
    logger.info(
        "=== D3: layer x target heatmap (%d layers x %d targets x %d inputs, %s) ===",
        len(layers),
        len(tkeys),
        len(args.input_variants),
        dev,
    )
    res.setdefault("gates", {})["ridge"] = _gate_ridge(dev)
    bundle = ctx["bundle"]
    if getattr(args, "corpus_mode", "single") == "n10k":
        S2, s2_idx, xl_targets, mask = load_d3_targets_n10k(
            args.capture_dir, args.p2_dir, ctx["new_bundle"], layers, tkeys, bundle["n_old"]
        )
    else:
        S2, s2_idx, xl_targets, mask = load_d3_targets(args.capture_dir, args.p2_dir, layers, tkeys)
    if args.max_contexts and getattr(args, "corpus_mode", "single") == "single":
        valid_ix = np.where(mask)[0][: args.max_contexts]
        keep = np.isin(np.where(mask)[0], valid_ix)
        newmask = np.zeros_like(mask)
        newmask[valid_ix] = True
        mask = newmask
        xl_targets = {k: v[keep] for k, v in xl_targets.items()}
    n_used = int(mask.sum())
    _t("D3 target load", t0)
    res.update(
        {
            "targets": tkeys,
            "target_labels": {k: TARGET_LABELS.get(k, k) for k in tkeys},
            "n_folds": args.n_folds,
            "seed": args.seed,
            "n_contexts_used": n_used,
            "note": "held-out variance-weighted R2 (5-fold CV, pooled test-own-mean); xl* "
            "targets are cross-layer aggregates predicted from the layer-L input; "
            "one shared Gram factorization per (input, layer, fold)",
        }
    )
    res.setdefault("inputs", {})
    for variant in args.input_variants:
        vnode = res["inputs"].setdefault(variant, {})
        for li in layers:
            lkey = str(li)
            if lkey in vnode:
                logger.info("[D3 %s] layer %d already checkpointed; skipping", variant, li)
                continue
            tl = time.time()
            X = input_layer(bundle, variant, li)[mask]
            targets = {}
            if "v_x" in tkeys:
                targets["v_x"] = target_vx(bundle, li)[mask]
            if S2 is not None:
                for si, s in enumerate(AS2.P2_SUMMARIES):
                    if s in tkeys:
                        targets[s] = S2[:, si, s2_idx[li], :][mask].astype(np.float32)
            for k, v in xl_targets.items():
                targets[k] = v
            targets = {k: targets[k] for k in tkeys if k in targets}
            vnode[lkey] = gram_cv_recon(X, targets, args.n_folds, args.seed, dev)
            res["metadata"] = _base_metadata(
                "d3", args, {"partial_wall_s": round(time.time() - t0, 1)}
            )
            C.write_json_atomic(out_path, res)
            logger.info(
                "[D3 %s] layer %2d done (%.0fs): v_x R2=%.4f",
                variant,
                li,
                time.time() - tl,
                vnode[lkey].get("v_x", {}).get("r2_mean", float("nan")),
            )
    res["metadata"] = _base_metadata("d3", args, {"stage_wall_s": round(time.time() - t0, 1)})
    C.write_json_atomic(out_path, res)
    _t("D3 total", t0)


# ── stage D1: fair comparison (both inputs, val-selected layer) ────────────────


def _val_select(per_layer: dict) -> dict:
    """argmax val_r2 over the recorded layers -> {val_selected_layer, test_r2_at_val_selected}."""
    best_li, best_val = None, -np.inf
    for lk, d in per_layer.items():
        v = d.get("val_r2", float("nan"))
        if np.isfinite(v) and v > best_val:
            best_val, best_li = v, int(lk)
    if best_li is None:
        return {
            "val_selected_layer": None,
            "test_r2_at_val_selected_layer": float("nan"),
            "val_r2_at_val_selected_layer": float("nan"),
        }
    return {
        "val_selected_layer": best_li,
        "test_r2_at_val_selected_layer": per_layer[str(best_li)]["test_r2"],
        "val_r2_at_val_selected_layer": per_layer[str(best_li)]["val_r2"],
    }


def run_d1(args, ctx):  # noqa: C901 - flat per-fitter-per-input stage dispatcher; each block is one fitter's val-select + test read, extraction would just inline the branches
    out_path = args.out_dir / "fair_comparison.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    dev = _dev(args.device)
    bundle = ctx["bundle"]
    n_ctx = corpus_len(bundle)
    if args.max_contexts and getattr(args, "corpus_mode", "single") == "single":
        n_ctx = min(n_ctx, args.max_contexts)
    train, val, test = resolve_split(args, ctx)
    t0 = time.time()
    logger.info(
        "=== D1: fair comparison (train %d / val %d / test %d, inputs %s, %s) ===",
        len(train),
        len(val),
        len(test),
        list(args.input_variants),
        dev,
    )
    res.setdefault("gates", {})
    if "ridge" not in res["gates"]:
        res["gates"]["ridge"] = _gate_ridge(dev)
    if "mlp" not in res["gates"]:
        res["gates"]["mlp"] = _gate_mlp(dev)
    res["split"] = {
        "n_contexts": int(n_ctx),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
    }
    res.setdefault("inputs", {})

    def arrays(variant, li):
        X = input_layer(bundle, variant, li)
        Y = target_vx(bundle, li)
        return X[train], Y[train], X[val], Y[val], X[test], Y[test]

    # ---- ridge: all 28 layers, val-selected layer + clean test read, per input ----
    for variant in args.input_variants:
        rnode = res["inputs"].setdefault(variant, {}).setdefault("ridge", {"per_layer": {}})
        for li in args.ridge_layers:
            if str(li) in rnode["per_layer"]:
                continue
            Xtr, Ytr, Xval, Yval, Xte, Yte = arrays(variant, li)
            (pred_val, pred_te), lam = gram_fit_apply(Xtr, Ytr, [Xval, Xte], dev, val=(Xval, Yval))
            rnode["per_layer"][str(li)] = {
                "val_r2": PR._pooled_r2(pred_val, Yval),
                "test_r2": PR._pooled_r2(pred_te, Yte),
                "selected_lambda": lam,
                "lambda_edge": _lambda_edge(lam),
            }
            C.write_json_atomic(out_path, res)
        sel = _val_select(rnode["per_layer"])
        li_sel = sel["val_selected_layer"]
        Xtr, Ytr, Xval, Yval, Xte, Yte = arrays(variant, li_sel)
        (pred_te,), _ = gram_fit_apply(Xtr, Ytr, [Xte], dev, val=(Xval, Yval))
        rnode.update(sel)
        rnode["lambda_edge_at_val_selected"] = rnode["per_layer"][str(li_sel)]["lambda_edge"]
        rnode["test_ci_at_val_selected"] = _bootstrap_recon_ci(
            pred_te, Yte, args.n_boot, args.seed + li_sel
        )
        C.write_json_atomic(out_path, res)
        logger.info(
            "[D1 %s] ridge val-selected layer L%d test_r2=%.4f",
            variant,
            li_sel,
            sel["test_r2_at_val_selected_layer"],
        )

    # ---- KRR: 5 layers, (gamma,lambda) on val, val-selected layer + test read ----
    for variant in args.input_variants:
        knode = res["inputs"][variant].setdefault("krr", {"per_layer": {}})
        for li in args.krr_layers:
            if str(li) in knode["per_layer"]:
                continue
            Xtr, Ytr, Xval, Yval, Xte, Yte = arrays(variant, li)
            k = krr_select_predict(
                Xtr,
                Ytr,
                Xval,
                Yval,
                Xte,
                gamma_mult=KRR_GAMMA_MULT,
                lambdas=KRR_LAMBDAS,
                m_landmarks=args.krr_landmarks,
                seed=args.seed,
                dev=dev,
            )
            knode["per_layer"][str(li)] = {
                "val_r2": k["selected"]["val_r2"],
                "test_r2": PR._pooled_r2(k["pred_te"], Yte),
                "selected": k["selected"],
            }
            C.write_json_atomic(out_path, res)
        sel = _val_select(knode["per_layer"])
        li_sel = sel["val_selected_layer"]
        Xtr, Ytr, Xval, Yval, Xte, Yte = arrays(variant, li_sel)
        k = krr_select_predict(
            Xtr,
            Ytr,
            Xval,
            Yval,
            Xte,
            gamma_mult=KRR_GAMMA_MULT,
            lambdas=KRR_LAMBDAS,
            m_landmarks=args.krr_landmarks,
            seed=args.seed,
            dev=dev,
        )
        knode.update(sel)
        knode["test_ci_at_val_selected"] = _bootstrap_recon_ci(
            k["pred_te"], Yte, args.n_boot, args.seed + li_sel
        )
        C.write_json_atomic(out_path, res)
        logger.info(
            "[D1 %s] krr val-selected layer L%d test_r2=%.4f",
            variant,
            li_sel,
            sel["test_r2_at_val_selected_layer"],
        )

    # ---- MLP: batched selection grid then per-selected-recipe reads ----
    if "mlp_selection" not in res:
        sel_groups = []
        for variant in args.input_variants:
            Xtr, Ytr, *_ = arrays(variant, args.mlp_select_layer)
            for w in args.mlp_widths:
                for lr in args.mlp_lrs:
                    sel_groups.append(MLPGroup(("sel", variant, w, lr), Xtr, Ytr, w, lr))
        ts = time.time()
        fits = run_mlp_battery(sel_groups, dev=dev, max_epochs=args.mlp_max_epochs)
        selection = {"layer": args.mlp_select_layer, "per_input": {}, "grid": []}
        for variant in args.input_variants:
            _, _, Xval, Yval, _, _ = arrays(variant, args.mlp_select_layer)
            best = None
            for w in args.mlp_widths:
                for lr in args.mlp_lrs:
                    r = fits[("sel", variant, w, lr)]
                    vr2 = PR._pooled_r2(r.predict(Xval), Yval)
                    selection["grid"].append(
                        {
                            "input": variant,
                            "width": w,
                            "lr": lr,
                            "val_r2": float(vr2),
                            "epochs_ran": r.epochs_ran,
                        }
                    )
                    if best is None or vr2 > best["val_r2"]:
                        best = {"width": w, "lr": lr, "val_r2": float(vr2)}
            selection["per_input"][variant] = best
            logger.info(
                "[D1 %s] MLP recipe selected: width=%d lr=%.0e (val_r2=%.4f)",
                variant,
                best["width"],
                best["lr"],
                best["val_r2"],
            )
        res["mlp_selection"] = selection
        C.write_json_atomic(out_path, res)
        _t("D1 MLP selection", ts)

    # MLP test reads at L19+L26 (selected recipe) + residual-skip, both inputs, batched.
    # The menu additionally includes ridge's val-selected layer per input (user-directed:
    # the residual-skip nesting read must exist at the BEST linear layer, not only L19/L26).
    def _mlp_layers_for(variant: str) -> list[int]:
        layers = [int(x) for x in args.mlp_layers]
        li_sel = res["inputs"].get(variant, {}).get("ridge", {}).get("val_selected_layer")
        if li_sel is not None and int(li_sel) not in layers:
            layers.append(int(li_sel))
        return layers

    ridge_te_cache, ridge_lam_cache, exec_groups = {}, {}, []
    for variant in args.input_variants:
        rec = res["mlp_selection"]["per_input"][variant]
        for li in _mlp_layers_for(variant):
            Xtr, Ytr, Xval, Yval, Xte, _ = arrays(variant, li)
            exec_groups.append(MLPGroup(("mlp", variant, li), Xtr, Ytr, rec["width"], rec["lr"]))
            # ridge residual target (val-selected lambda — the D1/D2 GCV-degeneracy fix)
            (rt_tr, rt_te), rlam = gram_fit_apply(Xtr, Ytr, [Xtr, Xte], dev, val=(Xval, Yval))
            ridge_te_cache[(variant, li)] = rt_te
            ridge_lam_cache[(variant, li)] = rlam
            exec_groups.append(
                MLPGroup(
                    ("resid", variant, li),
                    Xtr,
                    (Ytr - rt_tr).astype(np.float32),
                    RESIDUAL_MLP_WIDTH,
                    rec["lr"],
                )
            )

    def _need(g):
        node = res["inputs"].get(g.key[1], {}).get(g.key[0])
        return not node or str(g.key[2]) not in node.get("per_layer", {})

    need = [g for g in exec_groups if _need(g)]
    if need:
        fits = run_mlp_battery(need, dev=dev, max_epochs=args.mlp_max_epochs)
        for variant in args.input_variants:
            rec = res["mlp_selection"]["per_input"][variant]
            for li in _mlp_layers_for(variant):
                _, _, _, _, Xte, Yte = arrays(variant, li)
                for kind, width in (("mlp", rec["width"]), ("resid", RESIDUAL_MLP_WIDTH)):
                    fk = (kind, variant, li)
                    if fk not in fits:
                        continue
                    pred = fits[fk].predict(Xte)
                    if kind == "resid":
                        pred = ridge_te_cache[(variant, li)] + pred
                    m = _bootstrap_recon_ci(pred, Yte, args.n_boot, args.seed + li)
                    fitter = "residual_skip" if kind == "resid" else "mlp"
                    node = res["inputs"][variant].setdefault(fitter, {"per_layer": {}})
                    entry = {
                        "val_r2": float("nan"),
                        "test_r2": m["r2"]["point"],
                        "test_ci": m,
                        "width": width,
                        "lr": rec["lr"],
                        "epochs_ran": fits[fk].epochs_ran,
                    }
                    if kind == "resid":
                        rlam = ridge_lam_cache.get((variant, li))
                        entry["base_ridge_lambda"] = rlam
                        entry["base_ridge_lambda_edge"] = _lambda_edge(rlam)
                    node["per_layer"][str(li)] = entry
        C.write_json_atomic(out_path, res)

    # MLP/residual val_r2 per layer (menu-limited selection) + select + caveat.
    for variant in args.input_variants:
        rec = res["mlp_selection"]["per_input"][variant]
        for fitter, kind in (("mlp", "mlp"), ("residual_skip", "resid")):
            node = res["inputs"][variant].get(fitter)
            if not node:
                continue
            for li in _mlp_layers_for(variant):
                pl = node["per_layer"].get(str(li))
                if not pl or np.isfinite(pl["val_r2"]):
                    continue
                Xtr, Ytr, Xval, Yval, _, _ = arrays(variant, li)
                w = RESIDUAL_MLP_WIDTH if kind == "resid" else rec["width"]
                if kind == "resid":
                    (rt_tr, rt_val), _ = gram_fit_apply(
                        Xtr, Ytr, [Xtr, Xval], dev, val=(Xval, Yval)
                    )
                    fit = run_mlp_battery(
                        [MLPGroup(("vsel",), Xtr, (Ytr - rt_tr).astype(np.float32), w, rec["lr"])],
                        dev=dev,
                        max_epochs=args.mlp_max_epochs,
                    )[("vsel",)]
                    pl["val_r2"] = float(PR._pooled_r2(rt_val + fit.predict(Xval), Yval))
                else:
                    fit = run_mlp_battery(
                        [MLPGroup(("vsel",), Xtr, Ytr, w, rec["lr"])],
                        dev=dev,
                        max_epochs=args.mlp_max_epochs,
                    )[("vsel",)]
                    pl["val_r2"] = float(PR._pooled_r2(fit.predict(Xval), Yval))
            node.update(_val_select(node["per_layer"]))
            node["layer_menu_caveat"] = (
                f"MLP layer menu: {list(args.mlp_layers)} + ridge's val-selected layer per input "
                f"(realized: {_mlp_layers_for(variant)}); L{MLP_SELECT_LAYER} was pre-chosen from "
                "#779's percontext_recon 5-fold curve; recipe (width/lr) selected on val at that "
                "layer only, so the selected-layer read is recipe-menu-limited (ridge's 28-layer "
                "selection is free)."
            )
            C.write_json_atomic(out_path, res)

    res["coverage_note"] = (
        f"ridge: all 28 layers (free val selection); KRR: layers {list(args.krr_layers)}; "
        f"MLP + residual_skip: layers {list(args.mlp_layers)} (menu-limited, recipe on val at "
        f"L{args.mlp_select_layer}); inputs {list(args.input_variants)}."
    )
    res["lambda_edge_warnings"] = _collect_lambda_edges(res.get("inputs", {}))
    res["metadata"] = _base_metadata("d1", args, {"stage_wall_s": round(time.time() - t0, 1)})
    C.write_json_atomic(out_path, res)
    _t("D1 total", t0)


def _collect_lambda_edges(inputs: dict) -> list[dict]:
    """Every ridge / residual-skip-base-ridge fit whose val-selected lambda sits at
    an ACTIVE-grid edge — a non-empty list means the ridge grid may be too narrow."""
    out: list[dict] = []
    for variant, vnode in inputs.items():
        for lk, pl in vnode.get("ridge", {}).get("per_layer", {}).items():
            e = pl.get("lambda_edge")
            if e:
                out.append(
                    {
                        "input": variant,
                        "fitter": "ridge",
                        "layer": int(lk),
                        "lambda": pl.get("selected_lambda"),
                        "edge": e,
                    }
                )
        for lk, pl in vnode.get("residual_skip", {}).get("per_layer", {}).items():
            e = pl.get("base_ridge_lambda_edge")
            if e:
                out.append(
                    {
                        "input": variant,
                        "fitter": "residual_skip(base ridge)",
                        "layer": int(lk),
                        "lambda": pl.get("base_ridge_lambda"),
                        "edge": e,
                    }
                )
    return out


# ── stage D2: scaling curves ────────────────────────────────────────────────────


def _d2_curve(args, ctx, dev, out_path, res, *, layer, variant, fitters, mlp_recipe):
    bundle = ctx["bundle"]
    train, val, test = resolve_split(args, ctx)
    X = input_layer(bundle, variant, layer)
    Y = target_vx(bundle, layer)
    Xval, Yval, Xte, Yte = X[val], Y[val], X[test], Y[test]
    curve_key = f"{variant}_L{layer}"
    cells = {tuple(c["_key"]): c for c in res["curves"].get(curve_key, [])}
    ns = [n for n in args.d2_ns if n <= len(train)]
    for ni, n in enumerate(ns):
        for draw in args.d2_draws:
            idx = np.random.default_rng(1000 * ni + draw).choice(len(train), size=n, replace=False)
            tr = train[np.sort(idx)]
            Xtr, Ytr = X[tr], Y[tr]
            for fitter in fitters:
                key = (fitter, int(n), int(draw))
                if key in cells:
                    continue
                if fitter == "ridge":
                    (pred,), lam = gram_fit_apply(Xtr, Ytr, [Xte], dev, val=(Xval, Yval))
                    r2, cos = _recon_point(pred, Yte)
                    extra = {"selected_lambda": lam, "lambda_edge": _lambda_edge(lam)}
                elif fitter == "krr":
                    k = krr_select_predict(
                        Xtr,
                        Ytr,
                        Xval,
                        Yval,
                        Xte,
                        gamma_mult=KRR_GAMMA_MULT,
                        lambdas=KRR_LAMBDAS,
                        m_landmarks=args.krr_landmarks,
                        seed=args.seed + draw,
                        dev=dev,
                    )
                    r2, cos = _recon_point(k["pred_te"], Yte)
                    extra = {"selected": k["selected"]}
                elif fitter == "mlp":
                    fit = run_mlp_battery(
                        [MLPGroup(("d2m",), Xtr, Ytr, mlp_recipe["width"], mlp_recipe["lr"])],
                        dev=dev,
                        max_epochs=args.mlp_max_epochs,
                    )[("d2m",)]
                    r2, cos = _recon_point(fit.predict(Xte), Yte)
                    extra = {"epochs_ran": fit.epochs_ran}
                else:  # residual_skip
                    (rt_tr, rt_te), rlam = gram_fit_apply(
                        Xtr, Ytr, [Xtr, Xte], dev, val=(Xval, Yval)
                    )
                    fit = run_mlp_battery(
                        [
                            MLPGroup(
                                ("d2r",),
                                Xtr,
                                (Ytr - rt_tr).astype(np.float32),
                                RESIDUAL_MLP_WIDTH,
                                mlp_recipe["lr"],
                            )
                        ],
                        dev=dev,
                        max_epochs=args.mlp_max_epochs,
                    )[("d2r",)]
                    r2, cos = _recon_point(rt_te + fit.predict(Xte), Yte)
                    extra = {
                        "epochs_ran": fit.epochs_ran,
                        "base_ridge_lambda": rlam,
                        "base_ridge_lambda_edge": _lambda_edge(rlam),
                    }
                cells[key] = {
                    "_key": list(key),
                    "fitter": fitter,
                    "n": int(n),
                    "draw": int(draw),
                    "r2": r2,
                    "mean_cosine": cos,
                    **extra,
                }
                res["curves"][curve_key] = list(cells.values())
                C.write_json_atomic(out_path, res)
                logger.info("[D2 %s] %-14s n=%4d draw=%d: R2=%.4f", curve_key, fitter, n, draw, r2)


def run_d2(args, ctx):
    out_path = args.out_dir / "scaling_curves.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    dev = _dev(args.device)
    d1 = json.loads((args.out_dir / "fair_comparison.json").read_text())
    rec_last = d1["mlp_selection"]["per_input"]["last"]
    ridge_last = d1["inputs"]["last"]["ridge"]
    t0 = time.time()
    res.setdefault("curves", {})
    res.update(
        {
            "anchor_layer": args.d2_layer,
            "ns": [n for n in args.d2_ns if n <= args.n_train],
            "draws": list(args.d2_draws),
            "mlp_recipe_last": rec_last,
            "residual_mlp_width": RESIDUAL_MLP_WIDTH,
            "seed": args.seed,
            "note": "MLP uses the D1 last-input val-selected recipe applied at each n (caveat: "
            "recipe not re-selected per n); ridge/MLP permutation-invariant so draws "
            "collapse at full n; KRR varies via landmark subsample.",
        }
    )
    logger.info("=== D2: scaling curves (anchor L%d last, %s) ===", args.d2_layer, dev)

    # anchor: last input, L19, all four fitters
    _d2_curve(
        args,
        ctx,
        dev,
        out_path,
        res,
        layer=args.d2_layer,
        variant="last",
        fitters=FITTERS,
        mlp_recipe=rec_last,
    )

    conditionals = []
    ridge_sel_layer = ridge_last.get("val_selected_layer")
    if ridge_sel_layer is not None and ridge_sel_layer != args.d2_layer:
        conditionals.append(
            ("ridge_krr_at_ridge_selected", ridge_sel_layer, "last", ("ridge", "krr"))
        )
        _d2_curve(
            args,
            ctx,
            dev,
            out_path,
            res,
            layer=ridge_sel_layer,
            variant="last",
            fitters=("ridge", "krr"),
            mlp_recipe=rec_last,
        )
    mean_ridge = d1["inputs"].get("mean", {}).get("ridge", {})
    if mean_ridge.get("val_r2_at_val_selected_layer", -np.inf) > ridge_last.get(
        "val_r2_at_val_selected_layer", np.inf
    ):
        mean_layer = mean_ridge.get("val_selected_layer", args.d2_layer)
        conditionals.append(("ridge_only_mean_input", mean_layer, "mean", ("ridge",)))
        _d2_curve(
            args,
            ctx,
            dev,
            out_path,
            res,
            layer=mean_layer,
            variant="mean",
            fitters=("ridge",),
            mlp_recipe=rec_last,
        )
    res["conditional_curves"] = [
        {"reason": r, "layer": la, "input": v, "fitters": list(f)} for r, la, v, f in conditionals
    ]
    res["scope_note"] = (
        "D2 default = last input, L19 anchor. Conditional extra curves per "
        "amendments 1/2 recorded in conditional_curves."
    )
    res["lambda_edge_warnings"] = _collect_d2_lambda_edges(res.get("curves", {}))
    res["metadata"] = _base_metadata("d2", args, {"stage_wall_s": round(time.time() - t0, 1)})
    C.write_json_atomic(out_path, res)
    _t("D2 total", t0)


def _collect_d2_lambda_edges(curves: dict) -> list[dict]:
    """Every D2 ridge / residual-skip cell whose val-selected lambda sits at an
    ACTIVE-grid edge (grouped by curve/n/draw)."""
    out: list[dict] = []
    for curve_key, cells in curves.items():
        for c in cells:
            if c.get("fitter") == "ridge" and c.get("lambda_edge"):
                out.append(
                    {
                        "curve": curve_key,
                        "fitter": "ridge",
                        "n": c["n"],
                        "draw": c["draw"],
                        "lambda": c.get("selected_lambda"),
                        "edge": c["lambda_edge"],
                    }
                )
            elif c.get("fitter") == "residual_skip" and c.get("base_ridge_lambda_edge"):
                out.append(
                    {
                        "curve": curve_key,
                        "fitter": "residual_skip(base ridge)",
                        "n": c["n"],
                        "draw": c["draw"],
                        "lambda": c.get("base_ridge_lambda"),
                        "edge": c["base_ridge_lambda_edge"],
                    }
                )
    return out


# ── figures (from the JSONs; run on the VM after both sides land) ──────────────


def _paper():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    return plt, paper_palette, savefig_paper


def _d1_point_ci(node):
    """(point, lo_err, hi_err) from a fitter/baseline node's test CI, or (nan,0,0)."""
    pt = node.get("test_r2_at_val_selected_layer")
    if pt is None:
        pt = node.get("point", np.nan)
    ci_d = node.get("test_ci_at_val_selected") or (
        node.get("per_layer", {}).get(str(node.get("val_selected_layer")), {}).get("test_ci")
    )
    if ci_d is None and "r2" in node:  # baseline node: {"r2": {point,lo,hi}, ...}
        ci_d = node
    lo = hi = 0.0
    if ci_d and "r2" in ci_d and np.isfinite(ci_d["r2"].get("lo", np.nan)):
        pt = ci_d["r2"].get("point", pt)
        lo = max(0.0, pt - ci_d["r2"]["lo"])
        hi = max(0.0, ci_d["r2"]["hi"] - pt)
    return float(pt) if pt is not None else np.nan, lo, hi


# identity-family baselines are input-dependent (they use v_C); shuffled-pairing +
# predict-the-mean are input-agnostic floors -> drawn as reference lines, not bars.
D1_BASELINE_BARS = ("identity_copy", "scaled_identity", "diagonal_only")
D1_BAR_LABELS = {
    "ridge": "ridge (linear)",
    "krr": "kernel ridge",
    "mlp": "MLP (full-dim)",
    "residual_skip": "ridge + MLP resid",
    "identity_copy": "identity copy",
    "scaled_identity": "scaled identity",
    "diagonal_only": "diagonal only",
}


def make_fig_d1(out_dir, fig_dir, prefix="ffc"):
    path = out_dir / "fair_comparison.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    plt, paper_palette, savefig_paper = _paper()
    variants = [v for v in INPUT_VARIANTS if v in res.get("inputs", {})]
    methods = list(FITTERS) + [
        b for b in D1_BASELINE_BARS if "baselines" in res["inputs"].get(variants[0], {})
    ]
    two = paper_palette(2)
    input_color = {v: two[i] for i, v in enumerate(variants)}
    xpos = np.arange(len(methods))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(1.15 * len(methods) + 3, 5.5))
    # y-floor: keep the informative range readable; identity-copy on the mean input
    # is ~-50 R2, which would flatten every other bar. Clip the display to the floor
    # and annotate any clipped bar with its true value.
    pts = []
    for variant in variants:
        node = res["inputs"][variant]
        for m in methods:
            src = node.get(m, {}) if m in FITTERS else node.get("baselines", {}).get(m, {})
            p0, _, _ = _d1_point_ci(src)
            if np.isfinite(p0):
                pts.append(p0)
    y_floor = -1.1  # below scaled-identity (~-1.09); identity-copy clips
    for vi, variant in enumerate(variants):
        node = res["inputs"][variant]
        offs = (vi - (len(variants) - 1) / 2) * width
        for mi, m in enumerate(methods):
            src = node.get(m, {}) if m in FITTERS else node.get("baselines", {}).get(m, {})
            pt, lo, hi = _d1_point_ci(src)
            if not np.isfinite(pt):
                continue
            clipped = pt < y_floor
            disp = max(pt, y_floor)
            ax.bar(
                xpos[mi] + offs,
                disp,
                width,
                yerr=(None if clipped else np.array([[lo], [hi]])),
                capsize=2,
                color=input_color[variant],
                label=INPUT_LABEL[variant] if mi == 0 else None,
            )
            if clipped:
                ax.annotate(
                    f"{pt:.1f}",
                    (xpos[mi] + offs, y_floor),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="white",
                    rotation=90,
                )
    ax.set_ylim(bottom=y_floor)
    # input-agnostic floors as reference lines
    sp = res["inputs"].get(variants[0], {}).get("baselines", {}).get("shuffled_pairing", {})
    if "point" in sp and np.isfinite(sp["point"]):
        ax.axhline(sp["point"], ls="--", lw=1.2, color="0.45", label="shuffled-pairing floor")
    ptm = res.get("baselines_input_agnostic", {}).get("predict_the_mean", {})
    _p, _, _ = _d1_point_ci(ptm)
    if np.isfinite(_p):
        ax.axhline(_p, ls=":", lw=1.2, color="0.65", label="predict-the-mean")
    n_fit = len(FITTERS)
    ax.axvline(n_fit - 0.5, color="0.8", lw=1.0)  # fitters | baselines divider
    ax.set_xticks(xpos)
    ax.set_xticklabels([D1_BAR_LABELS.get(m, m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel("held-out test R2 (at val-selected layer)")
    ax.set_title("Context -> answer map: predictors vs baselines")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    figs = savefig_paper(fig, f"{prefix}_fitter_comparison", dir=fig_dir)
    plt.close(fig)
    return str(figs.get("png", ""))


def make_fig_d2(out_dir, fig_dir, prefix="ffc"):
    path = out_dir / "scaling_curves.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    plt, paper_palette, savefig_paper = _paper()
    colors = paper_palette(len(FITTERS))
    anchor = f"last_L{res['anchor_layer']}"
    cells = res["curves"].get(anchor, [])
    ns = res["ns"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fi, fitter in enumerate(FITTERS):
        means = []
        for n in ns:
            vals = [
                c["r2"]
                for c in cells
                if c["fitter"] == fitter and c["n"] == n and np.isfinite(c["r2"])
            ]
            means.append(np.mean(vals) if vals else np.nan)
            for v in vals:
                ax.scatter([n], [v], color=colors[fi], s=14, alpha=0.35)
        ax.plot(ns, means, "-o", color=colors[fi], lw=1.5, ms=4, label=fitter.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training contexts (n_train)")
    ax.set_ylabel(f"test R2 (last input, L{res['anchor_layer']})")
    ax.set_title("Scaling curves — reconstruction R2 vs training size")
    ax.legend(fontsize=8, title="predictor")
    figs = savefig_paper(fig, f"{prefix}_scaling_curves", dir=fig_dir)
    plt.close(fig)
    return str(figs.get("png", ""))


D3_SUMMARY_LABELS = {
    "v_x": "mean over ALL answer tokens (default)",
    "v_full_mean": "mean over answer + turn-boundary tokens",
    "v_full_max": "per-dim max over answer + boundary tokens",
    "v_tmpl_mean": "mean over the 5 turn-boundary tokens",
    "v_tmpl_max": "per-dim max over the 5 boundary tokens",
    "v_im_end": "turn-end token <|im_end|>",
    "v_im_start": "next-turn start token <|im_start|>",
    "v_user": "next 'user' role token",
    "v_nl_after_user": "newline before next user message",
    "xlmean_v_max": "per-dim max over answer tokens (avg over layers)",
    "xlmax_v_max": "per-dim max over answer tokens (max over layers)",
    "xlmean_v_last_content": "last answer word-token (avg over layers)",
    "xlmax_v_last_content": "last answer word-token (max over layers)",
    "xlmean_v_last_turn": "turn-final token (avg over layers)",
    "xlmax_v_last_turn": "turn-final token (max over layers)",
    "xlmean_v_first": "first answer token (avg over layers)",
    "xlmax_v_first": "first answer token (max over layers)",
}


def make_fig_d3(out_dir, fig_dir, prefix="ffc"):
    """Best-across-layers held-out R2 per answer summary, last vs mean input on one axis."""
    path = out_dir / "layer_target_heatmap.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    plt, paper_palette, savefig_paper = _paper()
    targets, labels = res["targets"], res["target_labels"]
    variants = [v for v in INPUT_VARIANTS if v in res.get("inputs", {})]

    def best_across_layers(variant, tk):
        best, best_li = np.nan, None
        for lk, e in res["inputs"][variant].items():
            if tk in e and (not np.isfinite(best) or e[tk]["r2_mean"] > best):
                best, best_li = e[tk]["r2_mean"], int(lk)
        return best, best_li

    order = sorted(
        targets,
        key=lambda tk: max(best_across_layers(v, tk)[0] for v in variants),
        reverse=True,
    )
    two = paper_palette(2)
    input_color = {v: two[i] for i, v in enumerate(variants)}
    ypos = np.arange(len(order))
    height = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(order) + 2.5))
    for vi, variant in enumerate(variants):
        offs = (vi - (len(variants) - 1) / 2) * height
        vals, lys = [], []
        for tk in order:
            b, li = best_across_layers(variant, tk)
            vals.append(b)
            lys.append(li)
        bars = ax.barh(
            ypos + offs, vals, height, color=input_color[variant], label=INPUT_LABEL[variant]
        )
        for rect, li in zip(bars, lys, strict=False):
            if li is not None and np.isfinite(rect.get_width()):
                ax.annotate(
                    f"L{li}",
                    (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    textcoords="offset points",
                    xytext=(2, 0),
                    va="center",
                    fontsize=6,
                    color="0.3",
                )
    ax.set_yticks(ypos)
    ax.set_yticklabels([D3_SUMMARY_LABELS.get(tk, labels.get(tk, tk)) for tk in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("best held-out R2 across layers (best layer annotated)")
    ax.set_ylabel("answer-side summary predicted from the context vector")
    ax.set_title("Which answer summary is most predictable from the context?")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    figs = savefig_paper(fig, f"{prefix}_summary_best_by_layer", dir=fig_dir, embed_data=False)
    plt.close(fig)
    return str(figs.get("png", ""))


def run_figures(args, ctx):
    t0 = time.time()
    prefix = getattr(args, "fig_prefix", "ffc")
    made = {
        "d3": make_fig_d3(args.out_dir, args.fig_dir, prefix),
        "d1": make_fig_d1(args.out_dir, args.fig_dir, prefix),
        "d2": make_fig_d2(args.out_dir, args.fig_dir, prefix),
    }
    logger.info("Figures: %s", {k: v for k, v in made.items() if v})
    _t("figures", t0)


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Issue #779 fair fitter comparison (analysis-only).")
    p.add_argument("--stage", nargs="*", choices=["d3", "d1", "d2", "figures"], default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--capture-dir", type=Path, default=AS.DEFAULT_CAPTURE_DIR)
    p.add_argument("--p2-dir", type=Path, default=AS2.DEFAULT_P2_DIR)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--n-boot", type=int, default=BOOT_N)
    p.add_argument("--n-folds", type=int, default=CV_FOLDS)
    p.add_argument("--n-train", type=int, default=3600)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--max-contexts", type=int, default=0)
    p.add_argument("--input-variants", nargs="*", default=list(INPUT_VARIANTS))
    p.add_argument("--d3-layers", type=int, nargs="*", default=list(range(C.EXPECTED_LAYERS)))
    p.add_argument("--d3-targets", nargs="*", default=D3_TARGETS)
    p.add_argument("--ridge-layers", type=int, nargs="*", default=list(range(C.EXPECTED_LAYERS)))
    p.add_argument("--krr-layers", type=int, nargs="*", default=list(KRR_LAYERS))
    p.add_argument("--mlp-layers", type=int, nargs="*", default=list(MLP_LAYERS))
    p.add_argument("--mlp-select-layer", type=int, default=MLP_SELECT_LAYER)
    p.add_argument("--mlp-widths", type=int, nargs="*", default=list(MLP_WIDTHS))
    p.add_argument("--mlp-lrs", type=float, nargs="*", default=list(MLP_LRS))
    p.add_argument("--mlp-max-epochs", type=int, default=MLP_MAX_EPOCHS)
    p.add_argument("--krr-landmarks", type=int, default=KRR_LANDMARKS)
    p.add_argument("--d2-ns", type=int, nargs="*", default=list(D2_NS))
    p.add_argument("--d2-draws", type=int, nargs="*", default=list(D2_DRAWS))
    p.add_argument("--d2-layer", type=int, default=D2_LAYER)
    # round-2 (n10k) combined-corpus mode
    p.add_argument("--corpus-mode", choices=["single", "n10k"], default="single")
    p.add_argument("--new-bundle", type=Path, default=DEFAULT_NEW_BUNDLE)
    p.add_argument("--pass-b-path", type=Path, default=PASS_B_PATH)
    p.add_argument(
        "--n-pass-b",
        type=int,
        default=N_PASS_B,
        help="pass_b row count the round-1 byte-identical val/test split is anchored on",
    )
    args = p.parse_args()

    if args.smoke:
        args.max_contexts = args.max_contexts or 200
        args.input_variants = ["last", "mean"]
        args.d3_layers = [19, 26]
        args.d3_targets = ["v_x", "v_im_end", "xlmean_v_last_turn"]
        args.ridge_layers = [17, 19, 26]
        args.krr_layers = [19, 26]
        args.mlp_layers = [19, 26]
        args.mlp_widths = [512]
        args.mlp_lrs = [1e-3]
        args.mlp_max_epochs = 5
        args.krr_landmarks = 64
        args.n_train, args.n_val, args.n_test = 140, 20, 40
        args.n_boot = 50
        args.n_folds = 2
        args.d2_ns = [100, 140]
        args.d2_draws = [0]

    # round-2 (n10k) defaults — applied AFTER --smoke so an explicit small smoke wins.
    if args.corpus_mode == "n10k":
        global LAMBDAS
        LAMBDAS = LAMBDAS_N10K  # wider ridge grid [1e-3, 1e6] (late-binding: all callers pick up)
        if args.out_dir == DEFAULT_OUT_DIR:
            args.out_dir = DEFAULT_OUT_DIR_N10K
        if args.n_train == 3600:  # round-1 default → the n10k train target
            args.n_train = N10K_TRAIN
        if list(args.d2_ns) == list(D2_NS):  # round-1 default → the extended axis through 10000
            args.d2_ns = list(D2_NS_N10K)
        args.fig_prefix = "ffc2"
    else:
        args.fig_prefix = "ffc"

    torch.set_num_threads(int(args.n_threads))
    stages = list(args.stage) or (["d3", "d1", "d2", "figures"] if args.all else [])
    if not stages:
        stages = ["d3", "d1", "d2", "figures"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "FFC stages=%s device=%s inputs=%s out=%s smoke=%s",
        stages,
        args.device,
        args.input_variants,
        args.out_dir,
        args.smoke,
    )
    ctx = {}
    if any(s in stages for s in ("d3", "d1", "d2")):
        if args.corpus_mode == "n10k":
            combined, nb = load_combined_corpus(args.pass_b_path, args.new_bundle, args.n_pass_b)
            ctx["bundle"] = combined
            ctx["new_bundle"] = nb
            # The split feeds ONLY D1/D2 (D3 is 5-fold CV over the masked corpus) — build it
            # only when a split-consuming stage runs, so a d3-only run needs no valid split.
            if any(s in stages for s in ("d1", "d2")):
                train, val, test, split_diag = build_n10k_split(combined, args)
                ctx["split"] = (train, val, test)
                C.write_json_atomic(args.out_dir / "n10k_split.json", split_diag)
                logger.info(
                    "n10k split: train=%d val=%d test=%d (byte-identical val/test=%s) -> %s",
                    len(train),
                    len(val),
                    len(test),
                    split_diag["val_test_byte_identical_round1"],
                    args.out_dir / "n10k_split.json",
                )
        else:
            ctx["bundle"] = load_pass_b(args.pass_b_path)
    if "d3" in stages:
        run_d3(args, ctx)
    if "d1" in stages:
        run_d1(args, ctx)
    if "d2" in stages:
        run_d2(args, ctx)
    if "figures" in stages:
        run_figures(args, ctx)
    logger.info("Done. stages=%s", stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
