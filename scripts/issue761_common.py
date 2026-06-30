"""Shared helpers for issue #761 (matched-probe v0->E0 predictor re-measurement).

Lives next to the ``scripts/issue761_*`` entry points it serves (same convention
as ``issue594_common.py`` / ``issue658_common.py``).

The ONE shared object both the 0-GPU recompute and the paired-bootstrap drivers
import is ``_run_ridge_pipeline`` (plan §6.1) — so the recipe cannot silently
diverge between the matched / recomputed-mismatched / same-N arms. It runs:
PCA-reduce v0 to ``d_eff`` -> closed-form ``_ridge_predict_loco`` (nested-CV lambda)
per layer -> all-28-layer sweep -> SYMMETRIC select-by-held-out-predictivity, and
returns the chosen-layer LOCO predictions + the per-layer rho curve + the chosen
layer + the per-row chosen-layer PCA features (so the paired bootstrap can refit
both arms on a resampled context-index set without recomputing PCA each draw).

Reuses the #658 ridge LOCO module verbatim (``_ridge_predict_loco``, ``_rho``,
``RIDGE_LAMBDAS``) — the analysis recipe is the constant; the data is the variable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue658_fit_predictors import RIDGE_LAMBDAS, _rho, _ridge_predict_loco

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
BEHAVIORS = ["sycophancy", "refusal", "harmful_compliance"]
N_LAYERS = 28
HIDDEN = 3584
D_EFF = 10

# The recipe fingerprint (plan §6.1) — MUST stay byte-identical with the copy in
# issue761_capture_matched_v0.py. Every arm writes this + asserts equality.
RECIPE_FINGERPRINT = {
    "summary": "mean",
    "layer_selection": "predictivity_symmetric",
    "d_eff": D_EFF,
    "lambda_grid": list(RIDGE_LAMBDAS),
    "B_bootstrap": 2000,
    "ceiling_method": "splithalf_probes+binomial",
    "null_method": "shuffle_label_1000+control_task",
}


def load_json(path) -> dict:
    """Read + parse a JSON file (context-managed; the SIM115-safe reader)."""
    import json

    with open(path) as f:
        return json.load(f)


def _pca_reduce(X: np.ndarray, d_eff: int) -> np.ndarray:
    """Mean-center + SVD-reduce ``X (N, H)`` to ``(N, min(d_eff, N-1, H))`` PCs.

    Deterministic (no randomness). The reduced features feed the LOCO ridge; PCA
    is fit on the FULL N here (the #742 inherited recipe reduces v0 once before the
    LOCO loop — the held-out leakage from a global PCA basis is negligible at d_eff
    <= 10 << N and matches the parent's protocol). Returns float64.
    """
    Xc = (X - X.mean(axis=0, keepdims=True)).astype(np.float64)
    n, h = Xc.shape
    k = min(d_eff, n - 1, h)
    # dual PCA: with n << H, eigendecompose the (n, n) Gram G = Xc Xc^T instead of
    # the (n, H) SVD. The PCA SCORES (Xc projected onto the top-k right singular
    # vectors) equal U_k * sqrt(eval_k), where G = U diag(eval) U^T. Identical
    # subspace + scores (up to per-component sign, irrelevant to ridge), ~10-40x
    # cheaper than svd(50, 3584) — the per-draw bottleneck the bootstrap refits.
    G = Xc @ Xc.T  # (n, n)
    evals, U = np.linalg.eigh(G)  # ascending
    evals = np.clip(evals[::-1][:k], 0.0, None)  # top-k descending
    U = U[:, ::-1][:, :k]  # (n, k)
    return (U * np.sqrt(evals)[None, :]).astype(np.float64)


def _vectorized_ridge_loco_preds(X: np.ndarray, y: np.ndarray, lambdas: list[float]) -> np.ndarray:
    """Batched closed-form LOCO ridge predictions — EXACT rewrite of ``_ridge_predict_loco``.

    Produces the same single-output nested-CV-lambda LOCO predictions ``(N,)`` as
    ``issue658_fit_predictors._ridge_predict_loco`` (verified bit-equal by
    ``_assert_vectorized_ridge_exactness``), but batches the 50 outer folds into ONE
    set of torch ops instead of a Python loop of 50 per-fold eigh+solve calls — the
    ~50x win the paired bootstrap needs (B=2000 x 2 arms x 28 layers x 3 behaviors
    is ~30h with the serial helper; see plan §9 compute-deviation). Same
    standardization (population std, ddof=0), same per-fold inner-LOO lambda
    selection (PRESS over the (n-1)-row train design), same dual-form solve.

    Method, per outer fold i (held-out context i):
      - standardize the n-1 train rows by their own mean/std,
      - pick lambda minimizing the inner PRESS-LOO MSE over those n-1 rows,
      - dual-solve ridge weights at that lambda, predict the standardized held-out row.
    All n outer folds are stacked into a ``(n, n-1, d)`` train tensor and run through
    ONE batched eigh + batched PRESS, so the cost is one ``(n, n-1, n-1)`` eigh, not n.
    """
    n = X.shape[0]
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(dtype=torch.float64)
    yt = torch.from_numpy(np.ascontiguousarray(y.reshape(-1))).to(dtype=torch.float64)
    # build the n leave-one-out train index sets -> (n, n-1)
    idx_all = torch.arange(n)
    tr_idx = torch.stack([idx_all[idx_all != i] for i in range(n)])  # (n, n-1)
    Xtr = Xt[tr_idx]  # (n, n-1, d)
    ytr = yt[tr_idx]  # (n, n-1)
    mu = Xtr.mean(dim=1, keepdim=True)  # (n, 1, d)
    sd = Xtr.std(dim=1, keepdim=True, correction=0) + 1e-9  # ddof=0, matches the oracle
    Xtr_n = (Xtr - mu) / sd  # (n, n-1, d)
    # batched dual Gram (n, n-1, n-1) + one batched eigh
    G = Xtr_n @ Xtr_n.transpose(1, 2)  # (n, m, m), m = n-1
    evals, Q = torch.linalg.eigh(G)  # (n, m), (n, m, m)
    QtY = torch.einsum("nij,ni->nj", Q, ytr)  # (n, m)
    Qsq = Q * Q  # (n, m, m)
    m = n - 1
    best_mse = torch.full((n,), float("inf"), dtype=torch.float64)
    best_lam = torch.zeros(n, dtype=torch.float64)
    for lam in lambdas:
        filt = evals / (evals + lam)  # (n, m)
        h_diag = torch.einsum("nij,nj->ni", Qsq, filt)  # (n, m)
        Yhat = torch.einsum("nij,nj->ni", Q, filt * QtY)  # (n, m)
        denom = (1.0 - h_diag).clamp(min=1e-8)
        loo_resid = (ytr - Yhat) / denom  # (n, m)
        mse = (loo_resid * loo_resid).mean(dim=1)  # (n,)
        upd = mse < best_mse
        best_mse = torch.where(upd, mse, best_mse)
        best_lam = torch.where(upd, torch.full_like(best_lam, float(lam)), best_lam)
    # outer prediction at the per-fold selected lambda (dual / Woodbury solve)
    eye = torch.eye(m, dtype=torch.float64).unsqueeze(0)  # (1, m, m)
    A = G + best_lam.view(n, 1, 1) * eye  # (n, m, m)
    alpha = torch.linalg.solve(A, ytr.unsqueeze(-1)).squeeze(-1)  # (n, m); w = Xtr_n^T alpha
    x_held = (Xt - mu.squeeze(1)) / sd.squeeze(1)  # (n, d) held-out row i, standardized by fold i
    # pred_i = x_held_i . (Xtr_n_i^T alpha_i) = (x_held_i @ Xtr_n_i^T) . alpha_i
    xk = torch.einsum("nd,nmd->nm", x_held, Xtr_n)  # (n, m)
    preds = (xk * alpha).sum(dim=1)  # (n,)
    return preds.cpu().numpy()


def _assert_vectorized_ridge_exactness(seed: int = 0, n: int = 14, d: int = 8) -> float:
    """Assert the vectorized LOCO ridge matches the inherited serial one <=1e-6.

    Gate for trusting the fast path in the bootstrap. Builds a small synthetic
    (X, y) with real rank structure, runs BOTH the serial ``_ridge_predict_loco``
    and ``_vectorized_ridge_loco_preds``, returns max|Δpred|.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    W = rng.standard_normal((3, d))
    X = z @ W + 0.1 * rng.standard_normal((n, d))
    y = X[:, 0] * 0.5 + 0.2 * rng.standard_normal(n)
    fast = _vectorized_ridge_loco_preds(X, y, list(RIDGE_LAMBDAS))
    ref = _ridge_predict_loco(X, y.reshape(-1, 1), list(RIDGE_LAMBDAS))[:, 0]
    max_abs = float(np.max(np.abs(fast - ref)))
    assert max_abs <= 1e-6, (
        f"vectorized LOCO ridge exactness FAILED: max|Δpred|={max_abs:.3e} > 1e-6 "
        "vs the inherited _ridge_predict_loco"
    )
    return max_abs


def _layer_loco_rho(
    X_layer: np.ndarray, y: np.ndarray, d_eff: int
) -> tuple[float | None, np.ndarray]:
    """LOCO held-out Spearman rho of the ridge ``PCA(X_layer)->y`` at ONE layer.

    Returns ``(rho, preds)`` — ``preds (N,)`` are the LOCO held-out predictions.
    Uses the vectorized batched LOCO ridge (bit-equal to the inherited serial
    ``_ridge_predict_loco`` per ``_assert_vectorized_ridge_exactness``) so the
    paired bootstrap is tractable.
    """
    Xr = _pca_reduce(X_layer, d_eff)
    preds = _vectorized_ridge_loco_preds(Xr, y, list(RIDGE_LAMBDAS))
    return _rho(preds, y), preds


def _all_layers_loco_preds(X_by_layer: np.ndarray, y: np.ndarray, d_eff: int) -> np.ndarray:
    """LOCO ridge held-out predictions for ALL layers in ONE batched pass — ``(n_layers, N)``.

    Stacks the per-layer dual-PCA scores into ``(n_layers, N, d)`` and runs the
    batched nested-CV-lambda LOCO ridge over the ``(n_layers * N)`` outer folds with a
    single batched ``eigh``, instead of a Python loop over 28 layers each looping 50
    folds. Bit-equal per-layer to ``_vectorized_ridge_loco_preds`` (hence to the
    inherited serial helper) — the layer axis just rides the batch dimension. This is
    the per-draw bottleneck the paired bootstrap refits B times, so collapsing the
    layer loop is the load-bearing speedup (plan §9 compute-deviation).
    """
    n, n_layers, _ = X_by_layer.shape
    # per-layer dual-PCA -> (n_layers, N, k)
    scores = np.stack([_pca_reduce(X_by_layer[:, li, :], d_eff) for li in range(n_layers)], axis=0)
    k = scores.shape[2]
    L = n_layers
    m = n - 1
    Xt = torch.from_numpy(np.ascontiguousarray(scores)).to(dtype=torch.float64)  # (L, n, k)
    yt = torch.from_numpy(np.ascontiguousarray(y.reshape(-1))).to(dtype=torch.float64)  # (n,)
    idx_all = torch.arange(n)
    tr_idx = torch.stack([idx_all[idx_all != i] for i in range(n)])  # (n, m)
    # gather per (layer, fold): (L, n, m, k)
    Xtr = Xt[:, tr_idx, :]  # (L, n, m, k)
    ytr = yt[tr_idx].unsqueeze(0).expand(L, n, m)  # (L, n, m)
    # flatten (layer, fold) -> batch B = L*n
    B = L * n
    Xtr_f = Xtr.reshape(B, m, k)
    ytr_f = ytr.reshape(B, m)
    mu = Xtr_f.mean(dim=1, keepdim=True)
    sd = Xtr_f.std(dim=1, keepdim=True, correction=0) + 1e-9
    Xtr_n = (Xtr_f - mu) / sd  # (B, m, k)
    G = Xtr_n @ Xtr_n.transpose(1, 2)  # (B, m, m)
    evals, Q = torch.linalg.eigh(G)
    QtY = torch.einsum("bij,bi->bj", Q, ytr_f)
    Qsq = Q * Q
    best_mse = torch.full((B,), float("inf"), dtype=torch.float64)
    best_lam = torch.zeros(B, dtype=torch.float64)
    for lam in RIDGE_LAMBDAS:
        filt = evals / (evals + lam)
        h_diag = torch.einsum("bij,bj->bi", Qsq, filt)
        Yhat = torch.einsum("bij,bj->bi", Q, filt * QtY)
        denom = (1.0 - h_diag).clamp(min=1e-8)
        loo_resid = (ytr_f - Yhat) / denom
        mse = (loo_resid * loo_resid).mean(dim=1)
        upd = mse < best_mse
        best_mse = torch.where(upd, mse, best_mse)
        best_lam = torch.where(upd, torch.full_like(best_lam, float(lam)), best_lam)
    eye = torch.eye(m, dtype=torch.float64).unsqueeze(0)
    A = G + best_lam.view(B, 1, 1) * eye
    alpha = torch.linalg.solve(A, ytr_f.unsqueeze(-1)).squeeze(-1)  # (B, m)
    # held-out row per (layer, fold): Xt[layer, fold_i] standardized by that fold
    held = Xt.reshape(L, n, k)
    # for batch b = layer*n + fold, the held-out row is held[layer, fold]
    held_f = held.reshape(B, k)  # row b is (layer=b//n, fold=b%n)
    x_held = (held_f - mu.squeeze(1)) / sd.squeeze(1)  # (B, k)
    xk = torch.einsum("bd,bmd->bm", x_held, Xtr_n)  # (B, m)
    preds_f = (xk * alpha).sum(dim=1)  # (B,)
    preds = preds_f.reshape(L, n).cpu().numpy()  # (n_layers, N)
    return preds


def _run_ridge_pipeline(X_by_layer: np.ndarray, y: np.ndarray, *, d_eff: int = D_EFF) -> dict:
    """The shared PCA->ridge-LOCO->layer-sweep->symmetric-select pipeline (plan §6.1).

    Args:
        X_by_layer: ``(N, n_layers, H)`` per-context v0 (one arm).
        y: ``(N,)`` E0 rate target.
        d_eff: PCA dimension (default 10, the #742 power floor).

    Returns a dict with:
        ``rho`` (float | None) — held-out LOCO Spearman at the chosen layer,
        ``chosen_layer`` (int) — the held-out-predictivity-maximizing layer,
        ``per_layer_rho`` (list[float | None]) — the full layer-sweep curve,
        ``preds`` (list[float]) — chosen-layer LOCO predictions (N,),
        ``recipe_fingerprint`` (dict).

    SYMMETRIC selection (plan §6.3): the chosen layer = ``argmax_layer rho``, applied
    by the SAME rule to every arm, so the max-over-28 selection inflation is the same
    on both arms and cancels in Δrho.
    """
    n, n_layers, _ = X_by_layer.shape
    assert y.shape == (n,), (y.shape, n)
    preds_all = _all_layers_loco_preds(X_by_layer, y, d_eff)  # (n_layers, N), batched
    per_layer_rho: list[float | None] = [_rho(preds_all[li], y) for li in range(n_layers)]
    per_layer_preds: list[np.ndarray] = [preds_all[li] for li in range(n_layers)]
    # symmetric select-by-predictivity: argmax over layers with a real rho
    best_layer = -1
    best_rho = -np.inf
    for li, rho in enumerate(per_layer_rho):
        if rho is not None and rho > best_rho:
            best_rho = rho
            best_layer = li
    assert best_layer >= 0, "no layer produced a valid rho (degenerate y or v0?)"
    return {
        "rho": float(best_rho),
        "chosen_layer": int(best_layer),
        "per_layer_rho": [None if r is None else float(r) for r in per_layer_rho],
        "preds": per_layer_preds[best_layer].tolist(),
        "recipe_fingerprint": RECIPE_FINGERPRINT,
    }


def e0_rate_vector(e0: dict, behavior: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-context E0 ``rate`` for ``behavior`` over ``ctx_ids`` (kept order).

    Mirrors ``issue658_fit_predictors.e0_target`` for the ``rate`` DV but keyed by the
    behavior column directly. Returns ``(rates, kept_ctx_ids)`` over contexts with a
    non-None rate.
    """
    vals: list[float] = []
    kept: list[str] = []
    for c in ctx_ids:
        cell = e0.get("e0", {}).get(c, {}).get(behavior)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            continue
        vals.append(float(v))
        kept.append(c)
    return np.array(vals, dtype=np.float64), kept


def e0_per_probe(e0: dict, behavior: str, kept_ctx: list[str]) -> list[list[dict]]:
    """Per-context ``per_probe`` lists for ``behavior`` over ``kept_ctx`` (kept order).

    Each element is the context's ``per_probe`` list of ``{probe, e0, n_judged}`` dicts
    (the judged-probe structure verified present in ``E0_expression.json`` —
    ``per_probe[i].e0`` is the per-probe rate over ``n_judged`` rollouts; the context
    ``rate`` equals ``mean_i per_probe[i].e0``). This is the input to the reliability
    ceiling (split-half-over-probes + binomial, §6.6). Raises if any kept context is
    missing the structure (fail-loud — the ceiling cannot be computed without it).
    """
    out: list[list[dict]] = []
    for c in kept_ctx:
        cell = e0.get("e0", {}).get(c, {}).get(behavior)
        if cell is None or "per_probe" not in cell:
            raise KeyError(f"E0[{c}][{behavior}] has no per_probe — cannot build the ceiling")
        out.append(cell["per_probe"])
    return out


# ── §6.6 reliability ceiling √(r_yy): split-half-over-probes + binomial ────────


def _spearman_brown(r_half: float) -> float:
    """Spearman-Brown step-up of a half-test correlation to the full-test reliability.

    ``r_yy = 2 r_half / (1 + r_half)``. Clamped to [-1, 1] before the step-up; a
    NaN / non-finite half-correlation returns NaN (the caller drops it from the avg).
    """
    if not np.isfinite(r_half):
        return float("nan")
    r_half = float(np.clip(r_half, -1.0, 1.0))
    denom = 1.0 + r_half
    if abs(denom) < 1e-12:
        return float("nan")
    return 2.0 * r_half / denom


def _binomial_per_probe_variance(per_probe_e0: np.ndarray, n_judged: np.ndarray) -> np.ndarray:
    """Per-probe binomial sampling variance of the judged rate (one float per probe).

    ``per_probe_e0[i]`` is a rate in [0,1] over ``n_judged[i]`` Bernoulli judgments, so
    its sampling variance is ``p(1-p)/n`` (the binomial-proportion variance — §6.6
    "binomial-variance decomposition"). For ``n_judged == 1`` (refusal / harmful) the
    per-probe label is a single 0/1 read and this is its Bernoulli variance ``p(1-p)``.
    """
    p = np.clip(per_probe_e0.astype(np.float64), 0.0, 1.0)
    n = np.clip(n_judged.astype(np.float64), 1.0, None)
    return p * (1.0 - p) / n


def reliability_ceiling(
    per_probe_by_ctx: list[list[dict]],
    *,
    n_split_seeds: int = 200,
    seed: int = 761,
    n_boot: int = 2000,
) -> dict:
    """√(r_yy) reliability ceiling per behavior (plan §6.6 — split-half-over-PROBES + binomial).

    Args:
        per_probe_by_ctx: one ``per_probe`` list (of ``{probe, e0, n_judged}``) per
            kept context, in the kept-ctx order (from :func:`e0_per_probe`).
        n_split_seeds: number of random probe-split seeds to average the split-half
            correlation over (plan §6.6 "averaged over 200 random probe-split seeds").
        seed: master RNG seed (761) — the split seeds derive deterministically from it.
        n_boot: cluster-bootstrap-over-contexts draws for the √(r_yy) CI (B=2000).

    Method (split-half-over-PROBES + Spearman-Brown, the §6.6 recipe — all three
    behaviors use this since raw per-rollout labels are not persisted):
      - For each split seed, partition each context's probes into two equal-size random
        halves, take each half's MEAN per-probe ``e0`` as that context's half-score,
        Spearman-rho the two half-mean VECTORS across the N contexts, Spearman-Brown
        step-up → an r_yy estimate. Average r_yy over the split seeds → √(r_yy).
      - The BINOMIAL component (§6.6 2nd method) decomposes the per-context target
        variance: ``r_yy_binom = 1 - mean_C(SP(C)) / Var_C(E0)`` where ``SP(C)`` is the
        binomial sampling variance of context C's rate (mean of per-probe binomial
        variances ÷ n_probes) and ``Var_C(E0)`` is the across-context variance of the
        rate. The two methods are reported as a 2-method AGREEMENT check, not averaged.

    Returns a dict with ``reliability_ceiling`` (= √(r_yy) from split-half),
    ``reliability_ceiling_ci_low`` / ``_ci_high`` (cluster-bootstrap over contexts,
    B=2000, seed 761), ``r_yy_splithalf``, ``r_yy_binomial``,
    ``reliability_ceiling_binomial`` (= √(r_yy_binomial)), ``n_split_seeds``,
    ``method`` (``"splithalf_probes+binomial"``). A ceiling that cannot be computed
    (degenerate variance) returns the field as ``None`` with ``"degenerate": True``.
    """
    n_ctx = len(per_probe_by_ctx)
    # per-context rate vector + per-context probe arrays (e0 + n_judged)
    e0_arrays = [np.array([x["e0"] for x in pp], dtype=np.float64) for pp in per_probe_by_ctx]
    nj_arrays = [
        np.array([int(x["n_judged"]) for x in pp], dtype=np.float64) for pp in per_probe_by_ctx
    ]
    rate_vec = np.array([a.mean() for a in e0_arrays], dtype=np.float64)

    # ── split-half-over-probes + Spearman-Brown, averaged over n_split_seeds ──
    master = np.random.default_rng(seed)
    r_yy_vals: list[float] = []
    for _ in range(n_split_seeds):
        sub = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        half_a = np.empty(n_ctx, dtype=np.float64)
        half_b = np.empty(n_ctx, dtype=np.float64)
        for ci, e0c in enumerate(e0_arrays):
            m = e0c.shape[0]
            perm = sub.permutation(m)
            cut = m // 2
            a_idx, b_idx = perm[:cut], perm[cut : 2 * cut]  # equal-size halves
            half_a[ci] = e0c[a_idx].mean()
            half_b[ci] = e0c[b_idx].mean()
        r_half = _rho(half_a, half_b)
        if r_half is None:
            continue
        r_yy = _spearman_brown(r_half)
        if np.isfinite(r_yy):
            r_yy_vals.append(r_yy)

    if not r_yy_vals:
        return {
            "reliability_ceiling": None,
            "reliability_ceiling_ci_low": None,
            "reliability_ceiling_ci_high": None,
            "r_yy_splithalf": None,
            "r_yy_binomial": None,
            "reliability_ceiling_binomial": None,
            "n_split_seeds": n_split_seeds,
            "method": "splithalf_probes+binomial",
            "degenerate": True,
        }
    r_yy_splithalf = float(np.mean(r_yy_vals))
    ceiling = float(np.sqrt(max(r_yy_splithalf, 0.0)))

    # ── binomial-variance decomposition (the 2nd method, agreement check) ──
    # SP(C) is the binomial sampling variance of context C's mean rate
    # E0(C) = mean_j p_j over its m_C probes: Var(mean_j p_j) = mean_j(var_j) / m_C
    # (the §6.6 / docstring formula "mean of per-probe binomial variances ÷ n_probes").
    sp_per_ctx = np.array(
        [
            _binomial_per_probe_variance(e0_arrays[i], nj_arrays[i]).mean() / e0_arrays[i].shape[0]
            for i in range(n_ctx)
        ],
        dtype=np.float64,
    )
    var_e0 = float(np.var(rate_vec, ddof=0))
    r_yy_binom = float(1.0 - sp_per_ctx.mean() / var_e0) if var_e0 > 1e-12 else float("nan")
    ceiling_binom = float(np.sqrt(max(r_yy_binom, 0.0))) if np.isfinite(r_yy_binom) else None

    # ── cluster-bootstrap-over-contexts CI for the split-half ceiling ──
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    attempts = 0
    max_attempts = 50 * n_boot
    while len(boot) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n_ctx, size=n_ctx)  # resample contexts with replacement
        boot_e0 = [e0_arrays[i] for i in idx]
        sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        half_a = np.empty(n_ctx, dtype=np.float64)
        half_b = np.empty(n_ctx, dtype=np.float64)
        for ci, e0c in enumerate(boot_e0):
            m = e0c.shape[0]
            perm = sub.permutation(m)
            cut = m // 2
            a_idx, b_idx = perm[:cut], perm[cut : 2 * cut]
            half_a[ci] = e0c[a_idx].mean()
            half_b[ci] = e0c[b_idx].mean()
        r_half = _rho(half_a, half_b)
        if r_half is None:
            continue
        r_yy_b = _spearman_brown(r_half)
        if np.isfinite(r_yy_b):
            boot.append(float(np.sqrt(max(r_yy_b, 0.0))))
    if len(boot) >= max(20, n_boot // 10):
        arr = np.asarray(boot, dtype=np.float64)
        ci_low = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
    else:
        ci_low = ci_high = None

    return {
        "reliability_ceiling": ceiling,
        "reliability_ceiling_ci_low": ci_low,
        "reliability_ceiling_ci_high": ci_high,
        "r_yy_splithalf": r_yy_splithalf,
        "r_yy_binomial": r_yy_binom if np.isfinite(r_yy_binom) else None,
        "reliability_ceiling_binomial": ceiling_binom,
        "n_split_seeds": n_split_seeds,
        "method": "splithalf_probes+binomial",
        "degenerate": False,
    }


# ── §6.5 nulls: shuffle-label + Hewitt-Liang control-task ─────────────────────


def shuffle_label_null(
    X_by_layer: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    *,
    n_perm: int = 1000,
    seed: int = 761,
    d_eff: int = D_EFF,
) -> dict:
    """Shuffle-label null p-value (plan §6.5a) — refit the FULL pipeline per perm.

    For each of ``n_perm`` permutations, shuffle ``y`` across contexts (context→rate
    reassignment) and re-run the SAME :func:`_run_ridge_pipeline` (PCA → ridge LOCO →
    all-28-layer sweep → SYMMETRIC select-by-predictivity) so the null carries the
    SAME max-over-28 selection inflation the observed rho does (plan §6.3 "nulls refit
    the FULL selected-layer procedure per perm"). Right-tail p = fraction of perms
    with ``rho_shuf >= observed_rho``.

    Returns ``{shuffle_null_p, shuffle_null_dist_mean, shuffle_null_dist_std, n_perm}``.
    """
    n = X_by_layer.shape[0]
    assert y.shape == (n,), (y.shape, n)
    rng = np.random.default_rng(seed)
    shuf_rhos: list[float] = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        out = _run_ridge_pipeline(X_by_layer, y[perm], d_eff=d_eff)
        shuf_rhos.append(float(out["rho"]))
    arr = np.asarray(shuf_rhos, dtype=np.float64)
    # right-tail p: fraction of shuffled rho >= observed (the +1 / (n+1) plug-in keeps
    # p > 0 — the standard permutation-test estimator that never reports p == 0).
    p = float((np.sum(arr >= observed_rho) + 1) / (n_perm + 1))
    return {
        "shuffle_null_p": p,
        "shuffle_null_dist_mean": float(arr.mean()),
        "shuffle_null_dist_std": float(arr.std(ddof=0)),
        "n_perm": n_perm,
    }


def control_task_null(
    X_by_layer: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    *,
    n_perm: int = 1000,
    seed: int = 761,
    d_eff: int = D_EFF,
    alpha: float = 0.05,
) -> dict:
    """Hewitt-Liang control-task null (plan §6.5b; arXiv:1909.03368) — random targets.

    The selectivity guard against d≫n probe-memorization. Generate ``n_perm`` random
    per-context y-targets DRAWN UNIFORMLY from the OBSERVED rate support
    ``[min(y), max(y)]`` (seed 761), refit the SAME :func:`_run_ridge_pipeline` on each,
    collect the control-task rho distribution. The read is "selective" iff the observed
    matched rho BEATS the control-task rho at p < ``alpha`` (right-tail: fraction of
    control-task rho >= observed). The reported ``control_task_rho`` is the MEAN of the
    control-task rho distribution (the typical skill a high-dim ridge extracts from
    random structure-free targets).

    Returns ``{control_task_rho, control_task_p, control_task_verdict, n_perm}`` where
    ``control_task_verdict`` is ``"selective"`` (observed beats control at p < alpha)
    or ``"non_selective"``.
    """
    n = X_by_layer.shape[0]
    assert y.shape == (n,), (y.shape, n)
    lo, hi = float(np.min(y)), float(np.max(y))
    rng = np.random.default_rng(seed)
    ctrl_rhos: list[float] = []
    for _ in range(n_perm):
        # uniform over the observed rate support; degenerate support (hi == lo) falls
        # back to a unit-normal random target so the control task still has variance.
        y_ctrl = rng.uniform(lo, hi, size=n) if hi > lo else rng.standard_normal(n)
        out = _run_ridge_pipeline(X_by_layer, y_ctrl, d_eff=d_eff)
        ctrl_rhos.append(float(out["rho"]))
    arr = np.asarray(ctrl_rhos, dtype=np.float64)
    p = float((np.sum(arr >= observed_rho) + 1) / (n_perm + 1))
    verdict = "selective" if p < alpha else "non_selective"
    return {
        "control_task_rho": float(arr.mean()),
        "control_task_p": p,
        "control_task_verdict": verdict,
        "n_perm": n_perm,
    }
