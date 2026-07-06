#!/usr/bin/env python3
"""Issue #779 free re-analysis: validation-selected FAIR fitter comparison.

Pure refit on the ALREADY-CACHED #779 activation tensors — NO new rollouts, NO
new capture, NO judge/API calls, NO GPU. Answers "is the ceiling on the
``c_last -> answer-summary`` map (#779's ~0.83 held-out R2 for the linear ridge)
a property of the DATA or of the LINEAR fitter?" by comparing GCV ridge, Nystrom
RBF kernel ridge, and a full-dim MLP under ONE fixed data split, plus scaling
curves and a layer x target heatmap.

Three deliverables (stages, ordered fastest-first for the detached run):

  d3  Layer x target heatmap (ridge only). All 28 layers x 17 targets
      (``v_x`` + the 8 pass-2 + 8 cross-layer answer summaries — the exact
      target set of ``issue779_arm_headline_summaries2``). 5-fold CV over the
      5000 LMSYS contexts, GCV ridge, ONE shared Gram factorization per
      (input layer, fold) reused across all 17 targets. Metric: held-out
      variance-weighted R2 (pooled, test-own-mean) + per-target mean cosine.
      -> layer_target_heatmap.json + ffc_layer_target_heatmap.{png,pdf}

  d1  Validation-selected fair fitter comparison (target = ``v(x)`` mean
      profile). ONE fixed split of the 5000 contexts: train 3600 / val 400 /
      test 1000, seed 42. Fitters: GCV ridge (full 3584->3584), Nystrom RBF
      KRR (1024 landmarks, full-dim; (gamma, lambda) selected on val), full-dim
      MLP (NO PCA head; (width, lr) selected on val at layer 19), and a
      residual-skip MLP (ridge prediction + MLP on the residual — strictly
      nests the linear map). ridge+KRR at {14, 17, 19, 26, 27}; MLP at
      {19, 26}. Metric: variance-weighted R2 + mean per-context cosine on TEST,
      with a 1000-resample bootstrap 95% CI over test contexts.
      -> fair_comparison.json + ffc_fitter_comparison.{png,pdf}

  d2  Scaling curves (layer 19). n_train in {250, 500, 1000, 2000, 3600}
      subsampled from the SAME train split (val/test fixed), 3 draws per n
      (seeds 0,1,2). Fitters: ridge (GCV per fit), KRR (reselect gamma/lambda
      on val per (n, draw)), MLP (the D1 val-selected recipe applied at each n
      — a stated caveat), residual-skip. Metric: test R2 per (fitter, n, draw).
      -> scaling_curves.json + ffc_scaling_curves.{png,pdf}

DOCUMENTED DEVIATIONS from the task brief (all also recorded in the JSON
``metadata.deviations``):
  * MLP width 8192 DROPPED from the grid (kept {512, 3584}). A smoke measured
    ~20.4 min per 300-epoch fit at width 8192, n=3600, H=3584 on the VM CPU;
    D2 alone would run >=15 such fits (>~4 h) and D1+D2 with 8192 projected
    well past the ~3.5 h budget. This is the brief's first descope lever.
  * The two Nystrom helpers (``nystrom_features`` / ``median_heuristic_gamma``)
    are inlined verbatim from ``scripts/issue779_batch2.py`` rather than
    imported, to avoid inheriting that module's import-time ``logging`` +
    ``FileHandler`` side effects; math is identical.
  * ``ridge`` uses GCV (train-internal lambda selection), so it does not
    consume the 400-context val set; KRR and MLP do. All fitters fit their
    FINAL model on the 3600-context train and are scored on the 1000 test
    contexts, so n_train is identical across fitters.

Reuses (does NOT reimplement): ``GramRidge`` (issue779_arm_headline),
``_pooled_r2`` / ``_per_context_cosine`` / ``_cv_folds`` (issue779_percontext_recon),
``load_crosslayer`` (issue779_arm_headline_summaries2), the pass-B LMSYS bundle
+ capture-shard loaders. Fail loud — NaN is reported, never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — MUST be
# called BEFORE numpy/torch freeze their thread pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_arm_headline as AH  # noqa: E402
import issue779_arm_headline_summaries as AS  # noqa: E402
import issue779_arm_headline_summaries2 as AS2  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger("issue779_ffc")

# ── target set for D3 (17): v(x) + 8 pass-2 + 8 cross-layer summaries ──────────
D3_TARGETS = ["v_x", *AS2.P2_SUMMARIES, *AS2.XL_VARIANTS]
TARGET_LABELS = {"v_x": "mean-response profile", **AS2.VARIANT_LABELS}

FITTERS = ("ridge", "krr", "mlp", "residual_skip")
RIDGE_KRR_LAYERS = (14, 17, 19, 26, 27)
MLP_LAYERS = (19, 26)
KRR_LANDMARKS = 1024
KRR_LAMBDAS = (1e-3, 1e-1, 1e1)
KRR_GAMMA_MULT = (0.25, 1.0, 4.0)
MLP_WIDTHS = (512, 3584)  # 8192 dropped — see module docstring / metadata.deviations
MLP_LRS = (1e-3, 3e-4)
MLP_WD = 1e-4
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 20
MLP_SELECT_LAYER = 19
RESIDUAL_MLP_WIDTH = 3584
D2_NS = (250, 500, 1000, 2000, 3600)
D2_DRAWS = (0, 1, 2)
D2_LAYER = 19
SPLIT_SEED = 42
BOOT_N = 1000
CV_FOLDS = 5

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "issue_779"
PASS_B_PATH = PROJECT_ROOT / "data" / "issue_779" / "pass_b" / "train_context_vectors.pt"


def _t(msg: str, t0: float) -> None:
    logger.info("[timing] %s: %.1f s", msg, time.time() - t0)


# ── metrics (variance-weighted R2 = pooled test-own-mean; mean cosine) ─────────


def _recon_point(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """(variance-weighted R2, mean per-context cosine). R2 = PR._pooled_r2
    (SS_tot on the test set's own mean — identical convention to the round's
    heldout_recon); mean cosine = mean of PR._per_context_cosine."""
    r2 = PR._pooled_r2(pred, true)
    cos = float(np.nanmean(PR._per_context_cosine(pred, true)))
    return r2, cos


def _bootstrap_recon_ci(pred: np.ndarray, true: np.ndarray, n_boot: int, seed: int) -> dict:
    """Point + percentile-bootstrap 95% CI of variance-weighted R2 and mean
    cosine, resampling the N test contexts with replacement (n_boot draws)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = pred.shape[0]
    res_i = ((true - pred) ** 2).sum(axis=1)  # per-context SS_res (N,)
    cos_i = PR._per_context_cosine(pred, true)  # (N,)
    r2_point, cos_point = _recon_point(pred, true)
    rng = np.random.default_rng(seed)
    r2s: list[float] = []
    coss: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss_res = float(res_i[idx].sum())
        t = true[idx]
        mu = t.mean(0)
        ss_tot = float(((t - mu) ** 2).sum())
        if ss_tot > 1e-12:
            r2s.append(1.0 - ss_res / ss_tot)
        c = cos_i[idx]
        c = c[np.isfinite(c)]
        if c.size:
            coss.append(float(c.mean()))

    def _ci(pt: float, boots: list[float]) -> dict:
        if not boots:
            return {"point": pt, "lo": float("nan"), "hi": float("nan")}
        return {
            "point": pt,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
        }

    return {
        "r2": _ci(r2_point, r2s),
        "mean_cosine": _ci(cos_point, coss),
        "n_test": int(n),
    }


# ── Nystrom RBF kernel ridge (inlined verbatim from issue779_batch2.py) ────────


def nystrom_features(
    X: np.ndarray, landmarks: np.ndarray, gamma: float, eig_floor: float = 1e-10
) -> np.ndarray:
    """Phi = K_nm @ K_mm^{-1/2} (float64), the standard Nystrom feature map.
    Verbatim from ``scripts/issue779_batch2.py`` (see module docstring)."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Zt = torch.as_tensor(landmarks, dtype=torch.float64)
    d_nm = torch.cdist(Xt, Zt) ** 2
    K_nm = torch.exp(-gamma * d_nm)
    d_mm = torch.cdist(Zt, Zt) ** 2
    K_mm = torch.exp(-gamma * d_mm)
    w, V = torch.linalg.eigh(K_mm)
    w = torch.clamp(w, min=eig_floor)
    inv_sqrt = V @ torch.diag(w.rsqrt()) @ V.T
    return (K_nm @ inv_sqrt).numpy()


def median_heuristic_gamma(X: np.ndarray, rng: np.random.Generator, n_sub: int = 2000) -> float:
    """gamma = 1 / median(squared pairwise distance). Verbatim from batch2."""
    sub = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]
    St = torch.as_tensor(sub, dtype=torch.float64)
    d2 = (torch.cdist(St, St) ** 2).numpy()
    off = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(off))
    assert med > 0, med
    return 1.0 / med


def _feature_ridge_multi_lambda(
    Phi_tr: np.ndarray, Y_tr: np.ndarray, Phi_eval_list: list[np.ndarray], lambdas
) -> list[list[np.ndarray]]:
    """Ridge on precomputed features at each lambda, ONE eigh of Phi^T Phi.

    Returns preds[i][j] = prediction for Phi_eval_list[j] at lambdas[i].
    Standardizes nothing (features already unit-scaled by Nystrom); centers Y on
    the train mean and un-centers predictions.
    """
    P = torch.as_tensor(Phi_tr, dtype=torch.float64)
    Y = torch.as_tensor(Y_tr, dtype=torch.float64)
    ymu = Y.mean(0)
    Yc = Y - ymu
    A = P.T @ P  # (m, m)
    a, Q = torch.linalg.eigh(A)
    a = torch.clamp(a, min=0.0)
    QtPtY = Q.T @ (P.T @ Yc)  # (m, H)
    evals = [torch.as_tensor(E, dtype=torch.float64) for E in Phi_eval_list]
    out: list[list[np.ndarray]] = []
    for lam in lambdas:
        W = Q @ (QtPtY / (a + lam)[:, None])  # (m, H)
        out.append([((E @ W) + ymu).numpy() for E in evals])
    return out


def _krr_select_predict(
    Xtr, Ytr, Xval, Yval, Xte, *, gamma_mult, lambdas, m_landmarks, seed
) -> dict:
    """Nystrom RBF KRR with (gamma, lambda) selected by val variance-weighted R2.

    Landmarks = a fixed seeded subsample of Xtr (shared across gammas). Returns
    the test prediction at the best (gamma, lambda) plus a full val-grid audit.
    """
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)
    Xval = np.asarray(Xval, dtype=np.float64)
    Yval = np.asarray(Yval, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    rng = np.random.default_rng(seed)
    base_gamma = median_heuristic_gamma(Xtr, np.random.default_rng(seed + 1))
    m = min(m_landmarks, len(Xtr))
    lm = Xtr[rng.choice(len(Xtr), size=m, replace=False)]
    grid: list[dict] = []
    best = None
    for gm in gamma_mult:
        gamma = base_gamma * gm
        Phi_tr = nystrom_features(Xtr, lm, gamma)
        Phi_val = nystrom_features(Xval, lm, gamma)
        Phi_te = nystrom_features(Xte, lm, gamma)
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


# ── full-dim MLP (NO PCA head; external-val early stop) ────────────────────────


def _mlp_full_fit(Xtr, Ytr, Xval, Yval, *, hidden, lr, wd, max_epochs, patience, seed, num_threads):
    """Full-dim (NO PCA) 1-hidden-GELU MLP, full-batch AdamW, early-stop on the
    EXTERNAL val MSE. Standardizes X on train stats, centers Y on the train mean.
    Returns ``(predict_fn, info)``; ``predict_fn(Xnew)`` returns raw-space preds.
    """
    torch.set_num_threads(int(num_threads))
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Ytr = np.asarray(Ytr, dtype=np.float32)
    Xval = np.asarray(Xval, dtype=np.float32)
    Yval = np.asarray(Yval, dtype=np.float32)
    din, dout = Xtr.shape[1], Ytr.shape[1]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-6
    ymu = Ytr.mean(0)
    Xt = torch.from_numpy((Xtr - xmu) / xsd)
    Yt = torch.from_numpy(Ytr - ymu)
    Xv = torch.from_numpy((Xval - xmu) / xsd)
    Yv = torch.from_numpy(Yval - ymu)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(din, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, dout)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad = 0
    ran = 0
    for ep in range(max_epochs):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(Xt), Yt)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = float(loss_fn(net(Xv), Yv).item())
        ran = ep + 1
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(Xnew: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(Xnew, dtype=np.float32) - xmu) / xsd
        with torch.no_grad():
            out = net(torch.from_numpy(Xn)).numpy()
        return out + ymu

    return predict, {"best_val_mse": float(best_val), "epochs_ran": int(ran)}


def _mlp_select_recipe(
    Xtr, Ytr, Xval, Yval, *, widths, lrs, wd, max_epochs, patience, seed, num_threads
) -> dict:
    """Grid (width x lr); train each, score on val by variance-weighted R2; pick
    the max-val-R2 recipe. Records every config's val score (audit trail)."""
    grid: list[dict] = []
    best = None
    for hidden in widths:
        for lr in lrs:
            t0 = time.time()
            pred_fn, info = _mlp_full_fit(
                Xtr,
                Ytr,
                Xval,
                Yval,
                hidden=hidden,
                lr=lr,
                wd=wd,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
                num_threads=num_threads,
            )
            val_r2 = PR._pooled_r2(pred_fn(Xval), Yval)
            wall = time.time() - t0
            entry = {
                "width": int(hidden),
                "lr": float(lr),
                "val_r2": float(val_r2),
                "best_val_mse": info["best_val_mse"],
                "epochs_ran": info["epochs_ran"],
                "wall_s": round(wall, 1),
            }
            grid.append(entry)
            logger.info(
                "[mlp-select] width=%d lr=%.0e: val_r2=%.4f (ep %d, %.0fs)",
                hidden,
                lr,
                val_r2,
                info["epochs_ran"],
                wall,
            )
            if best is None or (np.isfinite(val_r2) and val_r2 > best["val_r2"]):
                best = entry
    assert best is not None
    return {
        "selected": {"width": best["width"], "lr": best["lr"], "val_r2": best["val_r2"]},
        "grid": grid,
    }


def _mlp_test_metrics(
    Xtr,
    Ytr,
    Xval,
    Yval,
    Xte,
    Yte,
    *,
    width,
    lr,
    wd,
    max_epochs,
    patience,
    seed,
    num_threads,
    n_boot,
    boot_seed,
) -> dict:
    """Fit one full-dim MLP at (width, lr), report bootstrapped test metrics."""
    pred_fn, info = _mlp_full_fit(
        Xtr,
        Ytr,
        Xval,
        Yval,
        hidden=width,
        lr=lr,
        wd=wd,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        num_threads=num_threads,
    )
    out = _bootstrap_recon_ci(pred_fn(Xte), Yte, n_boot, boot_seed)
    out.update(
        {
            "width": int(width),
            "lr": float(lr),
            "n_train": len(Xtr),
            "best_val_mse": info["best_val_mse"],
            "epochs_ran": info["epochs_ran"],
        }
    )
    return out


def _residual_skip_predict(
    Xtr, Ytr, Xval, Yval, Xte, *, width, lr, wd, max_epochs, patience, seed, num_threads
) -> tuple[np.ndarray, dict]:
    """ridge(GCV) prediction + MLP fit on its residual; final = ridge + MLP.

    Strictly nests the linear ridge map (MLP models what ridge leaves behind).
    """
    gr = AH.GramRidge(np.asarray(Xtr, dtype=np.float64))
    ridge_tr = gr.predict(np.asarray(Ytr, dtype=np.float64), np.asarray(Xtr, dtype=np.float64))
    ridge_val = gr.predict(np.asarray(Ytr, dtype=np.float64), np.asarray(Xval, dtype=np.float64))
    ridge_te = gr.predict(np.asarray(Ytr, dtype=np.float64), np.asarray(Xte, dtype=np.float64))
    resid_tr = np.asarray(Ytr, dtype=np.float32) - ridge_tr.astype(np.float32)
    resid_val = np.asarray(Yval, dtype=np.float32) - ridge_val.astype(np.float32)
    pred_fn, info = _mlp_full_fit(
        Xtr,
        resid_tr,
        Xval,
        resid_val,
        hidden=width,
        lr=lr,
        wd=wd,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        num_threads=num_threads,
    )
    final_te = ridge_te + pred_fn(Xte)
    info["ridge_lambda"] = gr.last_lambda
    return final_te, info


# ── pass-B bundle + D3 target loading ──────────────────────────────────────────


def load_pass_b():
    """mmap-load the pass-B LMSYS bundle (5000, 28, 3584): cx_last + v_x."""
    try:
        b = torch.load(PASS_B_PATH, mmap=True, weights_only=False, map_location="cpu")
    except RuntimeError as e:
        logger.warning("mmap load failed (%s); full load", e)
        b = torch.load(PASS_B_PATH, weights_only=False, map_location="cpu")
    assert b["cx_last"].shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), b["cx_last"].shape
    assert b["v_x"].shape == b["cx_last"].shape
    return b


def load_p2_lmsys_all_layers(cap2: Path, layers: list[int]) -> np.ndarray:
    """LMSYS pass-2 summaries at the given layers -> (5000, 8, n_layers, H) fp16.

    Loaded ONCE (all requested layers) so the D3 per-layer loop only slices.
    Exercises the same shard contract as ``AS2.load_layer_gen``.
    """
    shards = sorted(cap2.glob("lmsys_summaries_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no pass-2 lmsys shards under {cap2}")
    k = len(AS2.P2_SUMMARIES)
    n_ctx = AS.N_LMSYS
    S: np.ndarray | None = None
    seen = np.zeros(n_ctx, dtype=bool)
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


def load_d3_targets(cap1: Path, cap2: Path, layers: list[int], target_keys: list[str]):
    """Assemble the LMSYS D3 target sources + the pass-1 joint validity mask.

    Returns ``(S2, S2_layer_index, xl_targets, mask)`` where ``S2`` is the pass-2
    all-layers array (or None if no pass-2 target requested), ``xl_targets`` maps
    each requested cross-layer key -> (n_valid, H) fp32, and ``mask`` is the
    pass-1 joint-validity mask over the 5000 contexts (applied to X + all
    targets so ONE factorization per (layer, fold) serves every target).
    """
    want_p2 = [t for t in target_keys if t in AS2.P2_SUMMARIES]
    want_xl = [t for t in target_keys if t in AS2.XL_VARIANTS]
    # Cross-layer variants are layer-free; load once. This also yields the pass-1
    # validity used as the joint mask (pass-2 + v_x are valid everywhere).
    XLl, v1l = AS2.load_crosslayer(cap1, "lmsys", AS.N_LMSYS, 1)
    mask = v1l[:, 0, :].all(axis=1)
    xl_targets = {t: XLl[t][:, 0, :][mask].astype(np.float32) for t in want_xl}
    S2 = None
    s2_layer_index: dict[int, int] = {}
    if want_p2:
        S2 = load_p2_lmsys_all_layers(cap2, layers)
        s2_layer_index = {li: i for i, li in enumerate(layers)}
    return S2, s2_layer_index, xl_targets, mask


def _heldout_recon_multi_with_cosine(X, targets: dict, n_folds: int, seed: int) -> dict:
    """5-fold held-out pooled R2 + mean cosine per target, ONE shared Gram
    factorization per fold (reused across all targets). Mirrors
    ``AH.heldout_recon_multi`` but also accumulates per-context cosine."""
    n = len(X)
    folds = PR._cv_folds(n, n_folds, seed)
    acc = {k: {"r2": [], "cos": []} for k in targets}
    for test_idx in folds:
        m = np.ones(n, dtype=bool)
        m[test_idx] = False
        gr = AH.GramRidge(X[m])
        for k, Y in targets.items():
            pred = gr.predict(Y[m], X[test_idx])
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


# ── fixed split ────────────────────────────────────────────────────────────────


def fixed_split(n_ctx: int, n_train: int, n_val: int, n_test: int, seed: int):
    """Deterministic (train, val, test) index arrays over ``n_ctx`` contexts."""
    assert n_train + n_val + n_test <= n_ctx, (n_train, n_val, n_test, n_ctx)
    perm = np.random.default_rng(seed).permutation(n_ctx)
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test : n_test + n_val])
    train = np.sort(perm[n_test + n_val : n_test + n_val + n_train])
    return train, val, test


def _base_metadata(stage: str, args, extra: dict) -> dict:
    b = C.reproducibility_metadata({"script": "issue779_fitter_fair_comparison", "stage": stage})
    pb = load_pass_b_meta()
    b.update(
        {
            "thread_caps": {
                k: __import__("os").environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "n_threads": int(args.n_threads),
            "data_provenance": {
                "pass_b": {"path": str(PASS_B_PATH), **pb},
                "capture_dir": str(args.capture_dir),
                "p2_dir": str(args.p2_dir),
            },
            "seed": args.seed,
            "deviations": [
                "MLP width 8192 dropped (kept 512, 3584); smoke measured ~20.4 min/300-epoch "
                "fit at width 8192 (n=3600, H=3584) — D2 with 8192 projected >~4 h.",
                "Nystrom helpers inlined verbatim from issue779_batch2.py (avoid its import-time "
                "logging/FileHandler side effects); math identical.",
                "ridge uses GCV (train-internal lambda), so it does not consume the 400 val "
                "contexts; KRR + MLP do. All fitters fit final on the 3600 train, scored on 1000 "
                "test.",
            ],
        }
    )
    b.update(extra)
    return b


def load_pass_b_meta() -> dict:
    """Path + stored bundle metadata + on-disk size/mtime (provenance, no sha)."""
    out: dict = {}
    try:
        st = PASS_B_PATH.stat()
        out["size_bytes"] = int(st.st_size)
        out["mtime_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
    except OSError as e:
        out["stat_error"] = str(e)
    try:
        b = torch.load(PASS_B_PATH, mmap=True, weights_only=False, map_location="cpu")
        out["bundle_metadata"] = b.get("metadata", {})
        out["source"] = b.get("source")
    except Exception as e:  # provenance read is best-effort
        out["bundle_read_error"] = str(e)
    return out


# ── stage D3: layer x target heatmap ───────────────────────────────────────────


def run_d3(args, ctx: dict) -> None:
    out_path = args.out_dir / "layer_target_heatmap.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    layers = args.d3_layers
    targets_keys = args.d3_targets
    t0 = time.time()
    logger.info(
        "=== D3: layer x target heatmap (%d layers x %d targets) ===",
        len(layers),
        len(targets_keys),
    )
    bundle = ctx["bundle"]
    S2, s2_idx, xl_targets, mask = load_d3_targets(
        args.capture_dir, args.p2_dir, layers, targets_keys
    )
    n_used = int(mask.sum())
    if args.max_contexts:
        # smoke: restrict to the first --max-contexts VALID contexts
        sub = np.where(mask)[0][: args.max_contexts]
        submask = np.zeros_like(mask)
        submask[sub] = True
        keep_in_valid = np.isin(np.where(mask)[0], sub)
        mask = submask
        xl_targets = {k: v[keep_in_valid] for k, v in xl_targets.items()}
        n_used = int(mask.sum())
    _t("D3 target load", t0)

    res.setdefault("layers", {})
    res["targets"] = targets_keys
    res["target_labels"] = {k: TARGET_LABELS.get(k, k) for k in targets_keys}
    res["n_folds"] = args.n_folds
    res["seed"] = args.seed
    res["n_contexts_used"] = n_used
    res["note"] = (
        "held-out variance-weighted R2 (5-fold CV over LMSYS contexts, pooled test-own-mean); "
        "xl* targets are cross-layer aggregates predicted from the layer-L input; "
        "one shared Gram factorization per (input layer, fold) serves all targets"
    )

    for li in layers:
        lkey = str(li)
        if lkey in res["layers"]:
            logger.info("[D3] layer %d already checkpointed; skipping", li)
            continue
        tl = time.time()
        col = bundle["layers"].index(li)
        X = bundle["cx_last"][:, col, :].to(torch.float32).numpy()[mask]
        targets: dict[str, np.ndarray] = {}
        if "v_x" in targets_keys:
            targets["v_x"] = bundle["v_x"][:, col, :].to(torch.float32).numpy()[mask]
        if S2 is not None:
            si_col = s2_idx[li]
            for si, s in enumerate(AS2.P2_SUMMARIES):
                if s in targets_keys:
                    targets[s] = S2[:, si, si_col, :][mask].astype(np.float32)
        for k, v in xl_targets.items():
            targets[k] = v  # already masked
        # keep the requested target order
        targets = {k: targets[k] for k in targets_keys if k in targets}
        per = _heldout_recon_multi_with_cosine(X, targets, args.n_folds, args.seed)
        res["layers"][lkey] = per
        res["metadata"] = _base_metadata(
            "d3", args, {"stage_wall_s_partial": round(time.time() - t0, 1)}
        )
        C.write_json_atomic(out_path, res)
        logger.info(
            "[D3] layer %2d done (%.0fs): v_x R2=%.4f | %d targets",
            li,
            time.time() - tl,
            per.get("v_x", {}).get("r2_mean", float("nan")),
            len(per),
        )
    res["metadata"] = _base_metadata("d3", args, {"stage_wall_s": round(time.time() - t0, 1)})
    C.write_json_atomic(out_path, res)
    _t("D3 total", t0)


# ── stage D1: validation-selected fair fitter comparison ───────────────────────


def run_d1(args, ctx: dict) -> None:
    out_path = args.out_dir / "fair_comparison.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    bundle = ctx["bundle"]
    n_ctx = bundle["cx_last"].shape[0]
    if args.max_contexts:
        n_ctx = min(n_ctx, args.max_contexts)
    train, val, test = fixed_split(n_ctx, args.n_train, args.n_val, args.n_test, args.seed)
    t0 = time.time()
    logger.info(
        "=== D1: fair fitter comparison (train %d / val %d / test %d) ===",
        len(train),
        len(val),
        len(test),
    )
    res["split"] = {
        "n_contexts": int(n_ctx),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
    }
    res.setdefault("layers", {})

    def _layer_arrays(li):
        col = bundle["layers"].index(li)
        X = bundle["cx_last"][:, col, :].to(torch.float32).numpy()
        Y = bundle["v_x"][:, col, :].to(torch.float32).numpy()
        return (X[train], Y[train], X[val], Y[val], X[test], Y[test])

    # MLP recipe selection at layer 19 (on val) — done first so both MLP layers reuse it.
    if "mlp_selection" not in res:
        Xtr, Ytr, Xval, Yval, _Xte, _Yte = _layer_arrays(args.mlp_select_layer)
        ts = time.time()
        sel = _mlp_select_recipe(
            Xtr,
            Ytr,
            Xval,
            Yval,
            widths=args.mlp_widths,
            lrs=args.mlp_lrs,
            wd=MLP_WD,
            max_epochs=args.mlp_max_epochs,
            patience=MLP_PATIENCE,
            seed=args.seed,
            num_threads=args.n_threads,
        )
        sel["layer"] = args.mlp_select_layer
        res["mlp_selection"] = sel
        C.write_json_atomic(out_path, res)
        _t(f"D1 MLP selection at L{args.mlp_select_layer}", ts)
    best_w = res["mlp_selection"]["selected"]["width"]
    best_lr = res["mlp_selection"]["selected"]["lr"]
    logger.info("[D1] MLP recipe selected: width=%d lr=%.0e", best_w, best_lr)

    for li in args.ridge_krr_layers:
        lkey = str(li)
        node = res["layers"].setdefault(lkey, {})
        Xtr, Ytr, Xval, Yval, Xte, Yte = _layer_arrays(li)
        boot_seed = args.seed + li

        if "ridge" not in node:
            tl = time.time()
            gr = AH.GramRidge(np.asarray(Xtr, dtype=np.float64))
            pred = gr.predict(np.asarray(Ytr, dtype=np.float64), np.asarray(Xte, dtype=np.float64))
            m = _bootstrap_recon_ci(pred, Yte, args.n_boot, boot_seed)
            m.update({"gcv_lambda": gr.last_lambda, "n_train": len(train)})
            node["ridge"] = m
            C.write_json_atomic(out_path, res)
            logger.info(
                "[D1 L%d] ridge R2=%.4f [%.4f,%.4f] (%.0fs)",
                li,
                m["r2"]["point"],
                m["r2"]["lo"],
                m["r2"]["hi"],
                time.time() - tl,
            )

        if "krr" not in node:
            tl = time.time()
            k = _krr_select_predict(
                Xtr,
                Ytr,
                Xval,
                Yval,
                Xte,
                gamma_mult=KRR_GAMMA_MULT,
                lambdas=KRR_LAMBDAS,
                m_landmarks=args.krr_landmarks,
                seed=args.seed,
            )
            m = _bootstrap_recon_ci(k.pop("pred_te"), Yte, args.n_boot, boot_seed)
            m.update(
                {
                    "selected": k["selected"],
                    "base_gamma": k["base_gamma"],
                    "m_landmarks": k["m_landmarks"],
                    "val_grid": k["val_grid"],
                    "n_train": len(train),
                }
            )
            node["krr"] = m
            C.write_json_atomic(out_path, res)
            logger.info(
                "[D1 L%d] krr R2=%.4f [%.4f,%.4f] sel g*=%.2f lam=%.0e (%.0fs)",
                li,
                m["r2"]["point"],
                m["r2"]["lo"],
                m["r2"]["hi"],
                k["selected"]["gamma_mult"],
                k["selected"]["lambda"],
                time.time() - tl,
            )

        if li in args.mlp_layers:
            if "mlp" not in node:
                tl = time.time()
                node["mlp"] = _mlp_test_metrics(
                    Xtr,
                    Ytr,
                    Xval,
                    Yval,
                    Xte,
                    Yte,
                    width=best_w,
                    lr=best_lr,
                    wd=MLP_WD,
                    max_epochs=args.mlp_max_epochs,
                    patience=MLP_PATIENCE,
                    seed=args.seed,
                    num_threads=args.n_threads,
                    n_boot=args.n_boot,
                    boot_seed=boot_seed,
                )
                C.write_json_atomic(out_path, res)
                logger.info(
                    "[D1 L%d] mlp R2=%.4f [%.4f,%.4f] (%.0fs)",
                    li,
                    node["mlp"]["r2"]["point"],
                    node["mlp"]["r2"]["lo"],
                    node["mlp"]["r2"]["hi"],
                    time.time() - tl,
                )
            if "residual_skip" not in node:
                tl = time.time()
                pred, info = _residual_skip_predict(
                    Xtr,
                    Ytr,
                    Xval,
                    Yval,
                    Xte,
                    width=args.residual_mlp_width,
                    lr=best_lr,
                    wd=MLP_WD,
                    max_epochs=args.mlp_max_epochs,
                    patience=MLP_PATIENCE,
                    seed=args.seed,
                    num_threads=args.n_threads,
                )
                m = _bootstrap_recon_ci(pred, Yte, args.n_boot, boot_seed)
                m.update(
                    {
                        "width": args.residual_mlp_width,
                        "lr": best_lr,
                        "ridge_lambda": info["ridge_lambda"],
                        "epochs_ran": info["epochs_ran"],
                        "n_train": len(train),
                    }
                )
                node["residual_skip"] = m
                C.write_json_atomic(out_path, res)
                logger.info(
                    "[D1 L%d] residual_skip R2=%.4f [%.4f,%.4f] (%.0fs)",
                    li,
                    m["r2"]["point"],
                    m["r2"]["lo"],
                    m["r2"]["hi"],
                    time.time() - tl,
                )

    res["coverage_note"] = (
        f"ridge+KRR at layers {list(args.ridge_krr_layers)}; MLP + residual_skip at "
        f"layers {list(args.mlp_layers)} (recipe selected on val at L{args.mlp_select_layer}); "
        f"MLP width 8192 dropped for budget (grid {list(args.mlp_widths)})."
    )
    res["metadata"] = _base_metadata("d1", args, {"stage_wall_s": round(time.time() - t0, 1)})
    C.write_json_atomic(out_path, res)
    _t("D1 total", t0)


# ── stage D2: scaling curves ────────────────────────────────────────────────────


def run_d2(args, ctx: dict) -> None:
    out_path = args.out_dir / "scaling_curves.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    d1_path = args.out_dir / "fair_comparison.json"
    if not d1_path.exists():
        raise SystemExit("D2 needs the D1 MLP-selected recipe; run d1 first")
    d1 = json.loads(d1_path.read_text())
    best_w = d1["mlp_selection"]["selected"]["width"]
    best_lr = d1["mlp_selection"]["selected"]["lr"]
    bundle = ctx["bundle"]
    n_ctx = bundle["cx_last"].shape[0]
    if args.max_contexts:
        n_ctx = min(n_ctx, args.max_contexts)
    train, val, test = fixed_split(n_ctx, args.n_train, args.n_val, args.n_test, args.seed)
    col = bundle["layers"].index(args.d2_layer)
    X = bundle["cx_last"][:, col, :].to(torch.float32).numpy()
    Y = bundle["v_x"][:, col, :].to(torch.float32).numpy()
    Xval, Yval = X[val], Y[val]
    Xte, Yte = X[test], Y[test]
    ns = [n for n in args.d2_ns if n <= len(train)]
    t0 = time.time()
    logger.info(
        "=== D2: scaling curves (L%d, ns=%s, draws=%s, mlp width=%d lr=%.0e) ===",
        args.d2_layer,
        ns,
        list(args.d2_draws),
        best_w,
        best_lr,
    )
    res.update(
        {
            "layer": args.d2_layer,
            "ns": ns,
            "draws": list(args.d2_draws),
            "mlp_recipe": {"width": best_w, "lr": best_lr},
            "residual_mlp_width": args.residual_mlp_width,
            "seed": args.seed,
            "note": "MLP uses the D1 val-selected recipe applied at each n (stated caveat); "
            "ridge/MLP are permutation-invariant so draws collapse at n=n_train_full, "
            "KRR still varies via its landmark subsample.",
        }
    )
    cells = {tuple(c["_key"]): c for c in res.get("cells", [])} if "cells" in res else {}

    def _key(fitter, n, draw):
        return (fitter, int(n), int(draw))

    for ni, n in enumerate(ns):
        for draw in args.d2_draws:
            sub_seed = 1000 * ni + draw
            idx = np.random.default_rng(sub_seed).choice(len(train), size=n, replace=False)
            tr = train[np.sort(idx)]
            Xtr, Ytr = X[tr], Y[tr]

            for fitter in FITTERS:
                key = _key(fitter, n, draw)
                if key in cells:
                    continue
                tl = time.time()
                if fitter == "ridge":
                    gr = AH.GramRidge(np.asarray(Xtr, dtype=np.float64))
                    pred = gr.predict(
                        np.asarray(Ytr, dtype=np.float64), np.asarray(Xte, dtype=np.float64)
                    )
                    r2, cos = _recon_point(pred, Yte)
                    extra = {"gcv_lambda": gr.last_lambda}
                elif fitter == "krr":
                    k = _krr_select_predict(
                        Xtr,
                        Ytr,
                        Xval,
                        Yval,
                        Xte,
                        gamma_mult=KRR_GAMMA_MULT,
                        lambdas=KRR_LAMBDAS,
                        m_landmarks=args.krr_landmarks,
                        seed=args.seed + draw,
                    )
                    r2, cos = _recon_point(k["pred_te"], Yte)
                    extra = {"selected": k["selected"]}
                elif fitter == "mlp":
                    pred_fn, info = _mlp_full_fit(
                        Xtr,
                        Ytr,
                        Xval,
                        Yval,
                        hidden=best_w,
                        lr=best_lr,
                        wd=MLP_WD,
                        max_epochs=args.mlp_max_epochs,
                        patience=MLP_PATIENCE,
                        seed=args.seed,
                        num_threads=args.n_threads,
                    )
                    r2, cos = _recon_point(pred_fn(Xte), Yte)
                    extra = {"epochs_ran": info["epochs_ran"]}
                else:  # residual_skip
                    pred, info = _residual_skip_predict(
                        Xtr,
                        Ytr,
                        Xval,
                        Yval,
                        Xte,
                        width=args.residual_mlp_width,
                        lr=best_lr,
                        wd=MLP_WD,
                        max_epochs=args.mlp_max_epochs,
                        patience=MLP_PATIENCE,
                        seed=args.seed,
                        num_threads=args.n_threads,
                    )
                    r2, cos = _recon_point(pred, Yte)
                    extra = {"epochs_ran": info["epochs_ran"], "ridge_lambda": info["ridge_lambda"]}
                cell = {
                    "_key": list(key),
                    "fitter": fitter,
                    "n": int(n),
                    "draw": int(draw),
                    "n_train": int(n),
                    "r2": r2,
                    "mean_cosine": cos,
                    **extra,
                }
                cells[key] = cell
                res["cells"] = list(cells.values())
                res["metadata"] = _base_metadata(
                    "d2", args, {"stage_wall_s_partial": round(time.time() - t0, 1)}
                )
                C.write_json_atomic(out_path, res)
                logger.info(
                    "[D2] %-14s n=%4d draw=%d: R2=%.4f cos=%.4f (%.0fs)",
                    fitter,
                    n,
                    draw,
                    r2,
                    cos,
                    time.time() - tl,
                )
    res["cells"] = list(cells.values())
    res["metadata"] = _base_metadata("d2", args, {"stage_wall_s": round(time.time() - t0, 1)})
    C.write_json_atomic(out_path, res)
    _t("D2 total", t0)


# ── figures (regenerated at the end from the JSONs) ────────────────────────────


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


def make_fig_d1(out_dir: Path, fig_dir: Path) -> str | None:
    path = out_dir / "fair_comparison.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    plt, paper_palette, savefig_paper = _paper()
    layers = sorted(int(k) for k in res["layers"])
    colors = paper_palette(len(FITTERS))
    fig, (ax_r2, ax_cos) = plt.subplots(1, 2, figsize=(14, 5.5))
    width = 0.8 / len(FITTERS)
    xpos = np.arange(len(layers))
    for ax, metric, ylabel in (
        (ax_r2, "r2", "held-out variance-weighted R2 (test)"),
        (ax_cos, "mean_cosine", "mean per-context cosine (test)"),
    ):
        for fi, fitter in enumerate(FITTERS):
            pts, los, his = [], [], []
            for li in layers:
                node = res["layers"].get(str(li), {}).get(fitter)
                if node is None:
                    pts.append(np.nan)
                    los.append(0.0)
                    his.append(0.0)
                    continue
                mm = node[metric]
                pt, lo, hi = mm["point"], mm["lo"], mm["hi"]
                pts.append(pt)
                los.append(max(0.0, pt - lo) if np.isfinite(lo) else 0.0)
                his.append(max(0.0, hi - pt) if np.isfinite(hi) else 0.0)
            ax.bar(
                xpos + (fi - (len(FITTERS) - 1) / 2) * width,
                pts,
                width,
                yerr=np.array([los, his]),
                capsize=2,
                color=colors[fi],
                label=fitter.replace("_", " ") if ax is ax_r2 else None,
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"L{li}" for li in layers])
        ax.set_xlabel("input layer")
        ax.set_ylabel(ylabel)
    ax_r2.set_title("Fair fitter comparison — reconstruction R2 (95% bootstrap CI)")
    ax_cos.set_title("Fair fitter comparison — per-context cosine")
    ax_r2.legend(fontsize=8, title="fitter")
    figs = savefig_paper(fig, "ffc_fitter_comparison", dir=fig_dir)
    plt.close(fig)
    return str(figs.get("png", ""))


def make_fig_d2(out_dir: Path, fig_dir: Path) -> str | None:
    path = out_dir / "scaling_curves.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    plt, paper_palette, savefig_paper = _paper()
    colors = paper_palette(len(FITTERS))
    ns = res["ns"]
    cells = res["cells"]
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
    ax.set_ylabel(f"held-out variance-weighted R2 (test, L{res['layer']})")
    ax.set_title("Scaling curves — reconstruction R2 vs training size")
    ax.legend(fontsize=8, title="fitter")
    figs = savefig_paper(fig, "ffc_scaling_curves", dir=fig_dir)
    plt.close(fig)
    return str(figs.get("png", ""))


def make_fig_d3(out_dir: Path, fig_dir: Path) -> str | None:
    path = out_dir / "layer_target_heatmap.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    plt, _paper_palette, savefig_paper = _paper()
    targets = res["targets"]
    labels = res["target_labels"]
    layers = sorted(int(k) for k in res["layers"])
    M = np.full((len(targets), len(layers)), np.nan)
    for ci, li in enumerate(layers):
        entry = res["layers"][str(li)]
        for ri, tk in enumerate(targets):
            if tk in entry:
                M[ri, ci] = entry[tk]["r2_mean"]
    # constrained layout (rcParam default via set_paper_style) — never tight_layout
    # after a colorbar exists (mpl refuses the engine switch).
    fig, ax = plt.subplots(
        figsize=(max(10, 0.42 * len(layers) + 4), 0.42 * len(targets) + 3), layout="constrained"
    )
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f"L{li}" for li in layers], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels([f"{labels.get(tk, tk)} ({tk})" for tk in targets], fontsize=7)
    ax.set_xlabel("input layer")
    ax.set_title("Layer x target held-out R2 (ridge, 5-fold CV)")
    fig.colorbar(im, ax=ax, label="held-out variance-weighted R2", fraction=0.03, pad=0.02)
    figs = savefig_paper(fig, "ffc_layer_target_heatmap", dir=fig_dir, embed_data=False)
    plt.close(fig)
    return str(figs.get("png", ""))


def run_figures(args, ctx: dict) -> None:
    t0 = time.time()
    made = {}
    made["d3"] = make_fig_d3(args.out_dir, args.fig_dir)
    made["d1"] = make_fig_d1(args.out_dir, args.fig_dir)
    made["d2"] = make_fig_d2(args.out_dir, args.fig_dir)
    logger.info("Figures: %s", {k: v for k, v in made.items() if v})
    _t("figures", t0)


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Issue #779 fair fitter comparison (analysis-only).")
    p.add_argument("--stage", nargs="*", choices=["d3", "d1", "d2", "figures"], default=[])
    p.add_argument("--all", action="store_true", help="run d3 -> d1 -> d2 -> figures")
    p.add_argument("--smoke", action="store_true")
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
    p.add_argument("--max-contexts", type=int, default=0, help="smoke: cap #contexts (0=all)")
    p.add_argument("--d3-layers", type=int, nargs="*", default=list(range(C.EXPECTED_LAYERS)))
    p.add_argument("--d3-targets", nargs="*", default=D3_TARGETS)
    p.add_argument("--ridge-krr-layers", type=int, nargs="*", default=list(RIDGE_KRR_LAYERS))
    p.add_argument("--mlp-layers", type=int, nargs="*", default=list(MLP_LAYERS))
    p.add_argument("--mlp-select-layer", type=int, default=MLP_SELECT_LAYER)
    p.add_argument("--mlp-widths", type=int, nargs="*", default=list(MLP_WIDTHS))
    p.add_argument("--mlp-lrs", type=float, nargs="*", default=list(MLP_LRS))
    p.add_argument("--mlp-max-epochs", type=int, default=MLP_MAX_EPOCHS)
    p.add_argument("--residual-mlp-width", type=int, default=RESIDUAL_MLP_WIDTH)
    p.add_argument("--krr-landmarks", type=int, default=KRR_LANDMARKS)
    p.add_argument("--d2-ns", type=int, nargs="*", default=list(D2_NS))
    p.add_argument("--d2-draws", type=int, nargs="*", default=list(D2_DRAWS))
    p.add_argument("--d2-layer", type=int, default=D2_LAYER)
    args = p.parse_args()

    if args.smoke:
        args.max_contexts = args.max_contexts or 200
        args.d3_layers = (
            args.d3_layers if args.d3_layers != list(range(C.EXPECTED_LAYERS)) else [19, 26]
        )
        args.d3_targets = ["v_x", "v_im_end", "xlmean_v_last_turn"]
        args.ridge_krr_layers = [19, 26]
        args.mlp_layers = [19]
        args.mlp_select_layer = 19
        args.mlp_widths = [512]
        args.mlp_lrs = [1e-3]
        args.mlp_max_epochs = 5
        args.residual_mlp_width = 512
        args.krr_landmarks = 64
        args.n_train, args.n_val, args.n_test = 140, 20, 40
        args.n_boot = 50
        args.n_folds = 2
        args.d2_ns = [100, 140]
        args.d2_draws = [0]

    torch.set_num_threads(int(args.n_threads))
    stages = list(args.stage)
    if args.all or not stages:
        stages = ["d3", "d1", "d2", "figures"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("FFC stages=%s out=%s smoke=%s", stages, args.out_dir, args.smoke)

    ctx: dict = {}
    if any(s in stages for s in ("d3", "d1", "d2")):
        ctx["bundle"] = load_pass_b()

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
