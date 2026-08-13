"""Vendored dof-capped GCV ridge cores from #2222 (issue2224 probe refit).

Provenance: copied VERBATIM (imports trimmed to this module's needs) from
``scripts/issue2222_analysis.py`` at commit
``99f9e975b08311684dd8f7ca6085e6a6b6791339`` (origin/issue-2222 tip at the
#2224 P1-gate pin, 2026-08-11) — functions ``_eigh_with_cpu_fallback``,
``dof_capped_ridge_multi_y``, ``dof_capped_ridge_fit_all``, ``ridge_predict``.
Vendored because issue-2222 is an UNMERGED sibling branch (artifact-reuse.md
§ porting from an unmerged sibling): #2224 must not import a module that only
exists on another issue's branch. TODO(#2224): once issue-2222 merges to main,
diff this module against main's ``scripts/issue2222_analysis.py`` copy and
delete this file in favor of importing the canonical one (supersede-and-delete,
code-style.md).

Do NOT edit the math here — the #2224 probe-refit parity gate
(``scripts/issue2224_probe_refit.py --parity-check``) certifies these cores
reproduce #2222's recorded held-out R² only while they stay byte-faithful.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/torch imports: shared-VM thread caps + HF token (#847)

import numpy as np  # noqa: E402

# --- Dof-capped GCV ridge (shared-eigh, multi-target) --------------------------


def _eigh_with_cpu_fallback(g):  # torch tensors
    """torch.linalg.eigh with the #825 `_eigh_robust` CPU fallback shape."""
    import torch

    try:
        return torch.linalg.eigh(g)
    except torch.linalg.LinAlgError:
        w, v = torch.linalg.eigh(g.cpu())
        return w.to(g.device), v.to(g.device)


def dof_capped_ridge_multi_y(
    x: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    *,
    lambdas: np.ndarray,
    dof_cap: float | None = 0.9,
    device: str = "cpu",
) -> dict:
    """Group-fold ridge with GCV lambda selection under the #1887 dof cap.

    One eigendecomposition of the centered train Gram (d x d) per fold, shared
    across every lambda and every target column (the vectorize-first kernel).
    Requires n_train > d per fold (well-posed primal regime — callers state
    n_train vs d per plan; an n_train <= d fold raises rather than silently
    fitting an under-determined regime, #1701/#1887).

    Args:
        x: (n, d) features; y: (n, T) targets; fold_ids: (n,) group-fold ids.
        lambdas: strictly-positive ascending grid; dof_cap: admissible lambdas
            satisfy df(lambda) <= dof_cap * n_train (None disables — refused
            here: pure-GCV at any regime keeps the cap per the #1887 ban).

    Returns dict with per-fold fits and pooled held-out diagnostics:
        folds: {fold: {w (d,T), b0 (T,), lam (T,), df (T,), n_train}}
        heldout_pred: (n, T) held-out predictions (each row from its fold's fit)
        heldout_r2: (T,) pooled held-out R^2
        gcv_lambda: (n_folds, T) selected lambdas
    """
    import torch

    if dof_cap is None:
        raise ValueError("dof_cap=None is the banned unguarded-GCV regime (#1887)")
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n, d = x.shape
    t = y.shape[1]
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if (
        lambdas.ndim != 1
        or len(lambdas) < 2
        or np.any(lambdas <= 0)
        or np.any(np.diff(lambdas) <= 0)
    ):
        raise ValueError("lambdas must be a strictly-positive ascending grid")
    fold_ids = np.asarray(fold_ids)
    uniq = np.unique(fold_ids)
    dev = torch.device(device)
    heldout_pred = np.full((n, t), np.nan)
    folds: dict = {}
    gcv_lambda = np.full((len(uniq), t), np.nan)
    for fi, f in enumerate(uniq):
        hold = fold_ids == f
        tr = ~hold
        n_tr = int(tr.sum())
        if n_tr <= d:
            raise ValueError(
                f"fold {f!r}: n_train={n_tr} <= d={d} — under-determined primal ridge refused"
            )
        xt = torch.from_numpy(x[tr]).to(dev)
        x_mu = xt.mean(dim=0, keepdim=True)
        xt = xt - x_mu
        yt = torch.from_numpy(y[tr]).to(dev, torch.float64)
        y_mu = yt.mean(dim=0, keepdim=True)
        yt = yt - y_mu
        g = (xt.T @ xt).double()  # (d, d)
        evals, vecs = _eigh_with_cpu_fallback(g)
        evals = torch.clamp(evals, min=0.0)
        xty = (xt.T.double() @ yt).double()  # (d, T)
        alpha = vecs.T @ xty  # (d, T)
        z = (xt.double() @ vecs) if n_tr * d <= 2e9 else None  # (n_tr, d) train scores
        if z is None:  # pragma: no cover - guarded by callers' sizes
            raise MemoryError("train-score matrix too large; subsample rows")
        y_ss = (yt**2).sum(dim=0)  # (T,)
        n_lam = len(lambdas)
        gcv = torch.full((n_lam, t), float("inf"), dtype=torch.float64)
        dfs = torch.zeros(n_lam, dtype=torch.float64)
        for li, lam in enumerate(lambdas):
            df = (evals / (evals + lam)).sum()
            dfs[li] = df
            if float(df) > dof_cap * n_tr:
                continue  # inadmissible under the cap
            coef = alpha / (evals + lam)[:, None]  # (d, T)
            pred_tr = z @ coef  # (n_tr, T)
            rss = y_ss - 2 * (pred_tr * yt).sum(dim=0) + (pred_tr**2).sum(dim=0)
            gcv[li] = n_tr * rss / (n_tr - df) ** 2
        if not torch.isfinite(gcv).any():
            raise ValueError(f"fold {f!r}: no lambda admissible under dof_cap={dof_cap}")
        best = torch.argmin(gcv, dim=0)  # (T,)
        w = torch.empty((d, t), dtype=torch.float64)
        lam_sel = np.empty(t)
        df_sel = np.empty(t)
        for ti in range(t):
            lam = float(lambdas[int(best[ti])])
            lam_sel[ti] = lam
            df_sel[ti] = float(dfs[int(best[ti])])
            w[:, ti] = (vecs @ (alpha[:, ti] / (evals + lam))).cpu()
        b0 = (y_mu.cpu().double() - x_mu.cpu().double() @ w).numpy()[0]  # (T,)
        w_np = w.numpy()
        folds[str(f)] = {
            "w": w_np,
            "b0": b0,
            "lam": lam_sel,
            "df": df_sel,
            "n_train": n_tr,
        }
        gcv_lambda[fi] = lam_sel
        heldout_pred[hold] = x[hold].astype(np.float64) @ w_np + b0[None, :]
    resid = y - heldout_pred
    ss_res = (resid**2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return {
        "folds": folds,
        "heldout_pred": heldout_pred,
        "heldout_r2": r2,
        "gcv_lambda": gcv_lambda,
        "fold_order": [str(f) for f in uniq],
    }


def dof_capped_ridge_fit_all(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lambdas: np.ndarray,
    dof_cap: float | None = 0.9,
    device: str = "cpu",
) -> dict:
    """Fit-on-ALL-rows variant of :func:`dof_capped_ridge_multi_y` (no holdout).

    Used where the caller owns the fold structure itself (Form B fits inside a
    caller-managed LOFO loop with a train-only PCA basis per fold). Returns one
    fold-shaped dict {w, b0, lam, df, n_train} consumable by :func:`ridge_predict`.
    """
    import torch

    if dof_cap is None:
        raise ValueError("dof_cap=None is the banned unguarded-GCV regime (#1887)")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n, d = x.shape
    t = y.shape[1]
    if n <= d:
        raise ValueError(f"n={n} <= d={d} — under-determined primal ridge refused")
    lambdas = np.asarray(lambdas, dtype=np.float64)
    dev = torch.device(device)
    xt = torch.from_numpy(x).to(dev, torch.float64)
    x_mu = xt.mean(dim=0, keepdim=True)
    xt = xt - x_mu
    yt = torch.from_numpy(y).to(dev, torch.float64)
    y_mu = yt.mean(dim=0, keepdim=True)
    yt = yt - y_mu
    evals, vecs = _eigh_with_cpu_fallback(xt.T @ xt)
    evals = torch.clamp(evals, min=0.0)
    alpha = vecs.T @ (xt.T @ yt)  # (d, T)
    z = xt @ vecs  # (n, d)
    y_ss = (yt**2).sum(dim=0)
    gcv = torch.full((len(lambdas), t), float("inf"), dtype=torch.float64)
    dfs = torch.zeros(len(lambdas), dtype=torch.float64)
    for li, lam in enumerate(lambdas):
        df = (evals / (evals + lam)).sum()
        dfs[li] = df
        if float(df) > dof_cap * n:
            continue
        coef = alpha / (evals + lam)[:, None]
        pred = z @ coef
        rss = y_ss - 2 * (pred * yt).sum(dim=0) + (pred**2).sum(dim=0)
        gcv[li] = n * rss / (n - df) ** 2
    if not torch.isfinite(gcv).any():
        raise ValueError(f"no lambda admissible under dof_cap={dof_cap}")
    best = torch.argmin(gcv, dim=0)
    w = torch.empty((d, t), dtype=torch.float64)
    lam_sel = np.empty(t)
    df_sel = np.empty(t)
    for ti in range(t):
        lam = float(lambdas[int(best[ti])])
        lam_sel[ti] = lam
        df_sel[ti] = float(dfs[int(best[ti])])
        w[:, ti] = (vecs @ (alpha[:, ti] / (evals + lam))).cpu()
    w_np = w.numpy()
    b0 = (y_mu.cpu() - x_mu.cpu() @ w).numpy()[0]
    return {"w": w_np, "b0": b0, "lam": lam_sel, "df": df_sel, "n_train": n}


def ridge_predict(fit_fold: dict, x_eval: np.ndarray) -> np.ndarray:
    """Apply one fold's fitted ridge: pred = x @ w + b0 -> (n, T)."""
    return np.asarray(x_eval, dtype=np.float64) @ fit_fold["w"] + fit_fold["b0"][None, :]
