"""Pure-analysis helpers for issue #2222 units 2/3 (P3 reduction + P4 probe).

NEW unit-2 file (plan v5 §4 P3/P4). ``issue2222_lib.py`` / ``issue2222_capture.py``
are inside the capture code-fingerprint and are never edited here.

Everything in this module is CPU numpy/torch math on small-to-medium arrays —
no network, no GPU requirement, no repo-global side effects — so the unit tests
run in seconds. Batching contract (vectorize-many-cell-fits.md):

- permutation nulls: ALL draws as one standardized GEMM (``perm_null_abs_r``),
  never a per-draw loop;
- bootstrap CIs: ALL resamples as one gathered einsum (``boot_r_matrix``);
- ridge fits: ONE eigh per (fold, layer) shared across every lambda AND every
  target column (``dof_capped_ridge_multi_y``) — the Form-A probe's 3 traits
  share X, so the factorization is computed once.

Estimator-diff note (record-integrity duty): ``dof_capped_ridge_multi_y`` is a
scalar/multi-target sibling of ``scripts/issue825_fit_cells.py`` ridge core
(`_ridge_predict_cached`): identical primal eigendecomposition + GCV lambda
selection under the #1887 dof cap (``GCV_DOF_CAP=0.9`` imported at call sites),
but (a) it EXPOSES the fitted weights (the #825 core returns predictions only —
the Form-A difference grid needs ``w`` to score stand-ins), and (b) selection is
GCV-with-dof-cap (the core's other registered mode) rather than inner-group-cv.
Both fits here run at n_train >> d (stated per fit by callers), where the two
selectors agree in regime; no permissiveness is broadened (the dof cap is
enforced, all-inadmissible grids raise).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# load_dotenv BEFORE any heavy import (numpy below, torch lazily) so the #847
# shared-VM thread caps bind in-process (tests/test_shared_vm_thread_caps.py):
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

__all__ = [
    "auc_mann_whitney",
    "boot_indices_clustered",
    "boot_indices_flat",
    "boot_r_matrix",
    "dof_capped_ridge_fit_all",
    "dof_capped_ridge_multi_y",
    "files_fingerprint",
    "leave_one_group_out_bias",
    "load_frozen_map",
    "lofo_layer_sweep",
    "map_project_via_u",
    "pca_train_basis",
    "pearson_r_cols",
    "perm_null_abs_r",
    "ridge_predict",
    "selection_inherited_delta",
    "spearman",
    "unit_normalize_rows",
]


def files_fingerprint(paths: list[Path] | tuple[Path, ...]) -> str:
    """sha256 over the named code files (unit-2 sibling of lib.code_fingerprint)."""
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode() + b"\0")
        h.update(p.read_bytes() if p.exists() else b"<absent>")
        h.update(b"\0")
    return h.hexdigest()


def unit_normalize_rows(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize along ``axis`` (v-hat per plan section 4: r_B / ||r_B||)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("zero-norm row in direction tensor — refusing to normalize")
    return v / n


# --- Frozen #1739 map (plan A4 apply recipe) ----------------------------------


def load_frozen_map(npz_path: Path) -> dict[str, np.ndarray]:
    """Load a #1739 map npz; asserts the realized keys + shapes (plan A7)."""
    with np.load(npz_path) as z:
        need = {"w", "x_mu", "x_sd", "y_mu"}
        missing = need - set(z.files)
        if missing:
            raise KeyError(f"{npz_path}: missing map keys {sorted(missing)}")
        m = {k: z[k] for k in ("w", "x_mu", "x_sd", "y_mu")}
    lw, d1, d2 = m["w"].shape
    assert d1 == d2, f"non-square map w: {m['w'].shape}"
    for k in ("x_mu", "x_sd", "y_mu"):
        assert m[k].shape == (lw, 1, d1), (k, m[k].shape)
    return m


def map_project_via_u(v: np.ndarray, fmap: dict[str, np.ndarray], vhat: np.ndarray) -> np.ndarray:
    """Projection of the mapped prediction onto vhat, without materializing M(v).

    ``M_l(v) = ((v - x_mu_l)/x_sd_l) @ w_l + y_mu_l`` (plan A4), so
    ``M_l(v) . vhat_{t,l} = z_l . u_{t,l} + y_mu_l . vhat_{t,l}`` with
    ``u_{t,l} = w_l @ vhat_{t,l}``.

    Args:
        v: (n, L, D) activation summaries (fp16/32 ok).
        fmap: loaded frozen map (w (L,D,D), x_mu/x_sd/y_mu (L,1,D)).
        vhat: (T, L, D) unit directions.

    Returns:
        (n, T, L) float32 projections.
    """
    n, lw, d = v.shape
    t = vhat.shape[0]
    assert vhat.shape == (t, lw, d), vhat.shape
    out = np.empty((n, t, lw), dtype=np.float32)
    for layer in range(lw):
        w_l = fmap["w"][layer].astype(np.float32)  # (D, D)
        vh_l = vhat[:, layer, :].astype(np.float32)  # (T, D)
        u_l = w_l @ vh_l.T  # (D, T)
        z = (v[:, layer, :].astype(np.float32) - fmap["x_mu"][layer].astype(np.float32)) / fmap[
            "x_sd"
        ][layer].astype(np.float32)  # (n, D)
        const = fmap["y_mu"][layer, 0].astype(np.float32) @ vh_l.T  # (T,)
        out[:, :, layer] = z @ u_l + const[None, :]
    return out


# --- Correlations / nulls / bootstrap -----------------------------------------


def _standardize_cols(a: np.ndarray) -> np.ndarray:
    """Column-standardize (mean 0, unit norm); zero-variance columns -> NaN."""
    a = np.asarray(a, dtype=np.float64)
    c = a - a.mean(axis=0, keepdims=True)
    n = np.linalg.norm(c, axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, c / n, np.nan)


def pearson_r_cols(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of each column of ``values`` (n, L) against ``y`` (n,)."""
    xs = _standardize_cols(values)
    ys = _standardize_cols(np.asarray(y, dtype=np.float64)[:, None])[:, 0]
    return xs.T @ ys


def perm_null_abs_r(values: np.ndarray, y: np.ndarray, *, n_perms: int, seed: int) -> np.ndarray:
    """(n_perms, L) |r| matrix under row-permutations of ``y`` — one GEMM.

    Selection-symmetric usage (selection-symmetric-nulls.md): the caller takes
    each draw's OWN max over layers to form the honest max-selected band.
    """
    xs = _standardize_cols(values)  # (n, L)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    perm_idx = np.argsort(rng.random((n_perms, n)), axis=1)  # (B, n) permutations
    yp = _standardize_cols(y[perm_idx].T).T  # standardize each draw's y
    return np.abs(yp @ xs)  # (B, L)


def boot_indices_flat(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) with-replacement resample indices over n units."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def boot_indices_clustered(group_ids: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Cluster bootstrap: resample GROUPS with replacement, carry members intact.

    Requires EQUAL group sizes (raises ValueError otherwise — no padding is
    implemented), so every draw concatenates whole drawn groups to a fixed
    length n. That invariant holds by design here: 8 families x 3 versions.
    """
    group_ids = np.asarray(group_ids)
    uniq = np.unique(group_ids)
    members = [np.flatnonzero(group_ids == g) for g in uniq]
    sizes = {len(m) for m in members}
    if len(sizes) != 1:
        raise ValueError(f"unequal group sizes {sorted(sizes)} — clustered draw ill-defined")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return np.stack([np.concatenate([members[g] for g in row]) for row in picks])


def boot_r_matrix(values: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(B, L) Pearson r per bootstrap draw — fully vectorized.

    values: (n, L); y: (n,); idx: (B, n) resample index rows.
    Degenerate draws (zero variance) yield NaN, never raise.
    """
    v = np.asarray(values, dtype=np.float64)[idx]  # (B, n, L)
    yy = np.asarray(y, dtype=np.float64)[idx]  # (B, n)
    vc = v - v.mean(axis=1, keepdims=True)
    yc = yy - yy.mean(axis=1, keepdims=True)
    num = np.einsum("bnl,bn->bl", vc, yc)
    den = np.sqrt(np.einsum("bnl,bnl->bl", vc, vc) * np.einsum("bn,bn->b", yc, yc)[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def selection_inherited_delta(
    r_a: np.ndarray, r_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-draw max-|r| re-selection delta (selection-inherited CI, #1434).

    r_a, r_b: (B, L) per-draw per-layer r for the two arms. Each draw selects
    its OWN argmax-|r| layer PER ARM; returns (delta (B,), sel_a (B,), sel_b (B,)).
    """
    sel_a = np.nanargmax(np.abs(r_a), axis=1)
    sel_b = np.nanargmax(np.abs(r_b), axis=1)
    rows = np.arange(r_a.shape[0])
    return r_a[rows, sel_a] - r_b[rows, sel_b], sel_a, sel_b


def lofo_layer_sweep(values: np.ndarray, y: np.ndarray, group_ids: np.ndarray) -> dict:
    """Leave-one-family-out layer selection (ood-generalization-folds.md).

    Per held-out group: select the layer by max |r| over the TRAIN groups, read
    the held-out units' predictor values at that layer. Returns the pooled LOFO
    r over all units plus per-fold selected layers, alongside the within-sample
    max read (labeled within-sample by the caller).
    """
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    assigned = np.full(y.shape[0], np.nan)
    sel_by_group: dict[str, int] = {}
    for g in np.unique(group_ids):
        hold = group_ids == g
        r_train = pearson_r_cols(values[~hold], y[~hold])
        sel = int(np.nanargmax(np.abs(r_train)))
        sel_by_group[str(g)] = sel
        assigned[hold] = values[hold, sel]
    r_within = pearson_r_cols(values, y)
    return {
        "lofo_r": float(pearson_r_cols(assigned[:, None], y)[0]),
        "selected_layer_by_fold": sel_by_group,
        "within_sample_max_abs_r": float(np.nanmax(np.abs(r_within))),
        "within_sample_argmax_layer": int(np.nanargmax(np.abs(r_within))),
        "within_sample_r_at_argmax": float(r_within[int(np.nanargmax(np.abs(r_within)))]),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho via mid-rank Pearson (no scipy dependency)."""

    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x), dtype=np.float64)
        ranks[order] = np.arange(1, len(x) + 1)
        # mid-ranks for ties
        xs = np.asarray(x, dtype=np.float64)
        for v in np.unique(xs):
            m = xs == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
        return ranks

    return float(pearson_r_cols(_rank(np.asarray(a))[:, None], _rank(np.asarray(b)))[0])


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) statistic, mid-ranks on ties."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"AUC needs both classes (pos={n_pos}, neg={n_neg})")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --- Identity+bias fold algebra (per-family sufficient statistics) ------------


def leave_one_group_out_bias(sum_resid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """b_f = mean residual over the OTHER groups (id_bias arm, plan section 4).

    sum_resid: (F, ...) per-group residual SUMS (raw - ctxend); counts: (F,).
    Returns (F, ...) leave-one-group-out bias vectors.
    """
    sum_resid = np.asarray(sum_resid, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    tot = sum_resid.sum(axis=0, keepdims=True)
    n_tot = counts.sum()
    shape = (len(counts),) + (1,) * (sum_resid.ndim - 1)
    return (tot - sum_resid) / (n_tot - counts).reshape(shape)


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


def pca_train_basis(x_train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """TRAIN-only PCA basis (Form B dim reduction): returns (mean (d,), P (d, k))."""
    x = np.asarray(x_train, dtype=np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    # thin SVD of (n, d); right singular vectors are the components.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    if k > vt.shape[0]:
        raise ValueError(f"k={k} exceeds available components {vt.shape[0]}")
    if not np.all(s[:k] > 0):
        raise ValueError("degenerate PCA spectrum in train fold")
    return mu, vt[:k].T
