"""Issue #667 gate-chain linear algebra + per-assumption statistics (CPU).

Pure NumPy/PyTorch tensor algebra over the per-cell activation store — NO GPU,
NO model load. Every function here is unit-testable on tiny inputs; the GPU
forward-pass extraction lives in ``scripts/issue667_extract.py`` and the
per-assumption runner in ``scripts/issue667_analysis.py``.

The five assumptions (plan §3), all benchmarked against #537's measured
leakage matrix ``G``:

- **A3.6** base read-out ``r_B'`` predicts the post-FT behavior CHANGE
  (partial-Spearman with the base level ``E0`` partialled out, C10).
- **A3.7** the FT write ``w_hat = v+(C) - v0(C)`` points toward the data
  target ``delta`` (``cos(w_hat, delta_pos)`` vs the shuffled-delta null, +
  the contrastive decomposition + frac_ctx).
- **A3.8** the off-source update ``Delta_v(C')`` is a scalar-gated copy of the
  source write (per-target rank-one residual + stacked-Delta_V SVD).
- **A3.9** the base key-query gate ``g0(C')`` predicts the activation realized
  gate ``g_hat^real(C')`` (key x metric ablation vs the cosine baseline).
- **A3.10** the base gate predicts the realized gate at fixed M0 (vs the
  post-FT oracle ``g+``), clustered CIs + probe-split.

The **B3 reduction unit test** (:func:`whitened_gate_reduction_unit_test`)
gates every A3.9/A3.10 number: the whitened gate must reduce to
``cos(c_C, c_C')`` in the Sigma_c=I / equal-norm limit, or a mis-implemented
inverse silently manufactures a "whitening wins" result.
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import numpy as np
import torch

# Reuse the in-house Spearman + bootstrap (no scipy import overhead), inherited
# from #519/#551/#602 (plan §4.5 reuse list).
from explore_persona_space.analysis.svd_direction_constancy import (
    bootstrap_ci,
    cosine,
    spearman_rho,
)

# ─────────────────────────────────────────────────────────────────────────────
# Realized activation gate + rank-one residual (A3.8 / B1)
# ─────────────────────────────────────────────────────────────────────────────


def realized_gate(
    v0_c: np.ndarray,
    vplus_c: np.ndarray,
    v0_cp: np.ndarray,
    vplus_cp: np.ndarray,
) -> tuple[float, float]:
    """Realized activation gate ``g_hat^real(C')`` + rank-one residual (B1, A3.8).

    ``w_hat = v+(C) - v0(C)`` is the source write; ``Delta_v(C') = v+(C') -
    v0(C')`` is the target update. The realized gate projects the target update
    onto the source write and self-normalizes by the source-write magnitude
    (install-normalized by construction — the reason B1 mandates this over the
    log-prob): ``g_hat^real = (w_hat . Delta_v) / (w_hat . w_hat)``.

    The rank-one residual is ``||Delta_v - g_hat^real * w_hat|| / ||Delta_v||``
    — small means the off-source update is a scalar multiple of the source
    write (A3.8).

    Returns ``(g_hat^real, rank_one_residual)``. A zero-norm source write (a
    saturated / rank-collapsed cell) raises — never silently returns 0.
    """
    w_hat = np.asarray(vplus_c, dtype=np.float64) - np.asarray(v0_c, dtype=np.float64)
    delta_v = np.asarray(vplus_cp, dtype=np.float64) - np.asarray(v0_cp, dtype=np.float64)
    ww = float(w_hat @ w_hat)
    if ww <= 0.0:
        raise ValueError(
            "realized_gate: source write w_hat has zero norm (saturated / "
            "rank-collapsed cell) — exclude this source upstream, do not gate on it."
        )
    g_real = float((w_hat @ delta_v) / ww)
    dn = float(np.linalg.norm(delta_v))
    if dn <= 0.0:
        # No target update at all — residual is undefined; report 0 update.
        return g_real, 0.0
    resid = float(np.linalg.norm(delta_v - g_real * w_hat) / dn)
    return g_real, resid


def stacked_delta_svd(delta_vs: np.ndarray, w_hat: np.ndarray) -> dict[str, float]:
    """Stacked-Delta_V SVD summary for one source (A3.8).

    ``delta_vs`` is ``(n_targets, H)`` — the off-source updates stacked as rows.
    Returns the top singular-value variance fraction ``sigma1^2 / sum sigma^2``
    (near 1 = low-rank write), ``sigma2 / sigma1``, and ``cos(u1, w_hat)`` (the
    top right-singular direction's alignment to the source write).

    Chance level for ``n_targets`` rows in ``H`` dims is ``1 / n_targets``
    (the analyzer states this explicitly, scope caveat 5).
    """
    M = np.asarray(delta_vs, dtype=np.float64)
    assert M.ndim == 2, M.shape
    n, _h = M.shape
    if n < 2:
        raise ValueError(f"stacked_delta_svd needs >=2 target rows, got {n}")
    # rows = targets; we want the right-singular vectors (directions in H).
    _u, s, vt = np.linalg.svd(M, full_matrices=False)
    s2 = s**2
    top1_frac = float(s2[0] / s2.sum()) if s2.sum() > 0 else 0.0
    sigma2_over_1 = float(s[1] / s[0]) if (len(s) > 1 and s[0] > 0) else 0.0
    u1 = vt[0]  # (H,) top right-singular vector (direction in feature space)
    return {
        "sigma1_sq_frac": top1_frac,
        "sigma2_over_sigma1": sigma2_over_1,
        "cos_u1_what": abs(cosine(u1, np.asarray(w_hat, dtype=np.float64))),
        "n_targets": int(n),
        "chance_sigma1_frac": float(1.0 / n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Whitened key-query gate + the B3 reduction unit test (A3.9 / A3.10)
# ─────────────────────────────────────────────────────────────────────────────


def sigma_inv_regularized(sigma_c: torch.Tensor, lam: float) -> torch.Tensor:
    """``(Sigma_c + lam * I)^-1`` in float64 (numerically fragile — B3).

    ``lam`` is the ABSOLUTE ridge added to the diagonal; callers compute it from
    :func:`default_lambda` (the ``1e-2 * tr(Sigma_c)/d`` fraction-of-mean-
    eigenvalue default, plan §11) or sweep it.
    """
    s = sigma_c.to(torch.float64)
    d = s.shape[0]
    assert s.shape == (d, d), s.shape
    return torch.linalg.inv(s + lam * torch.eye(d, dtype=torch.float64))


def default_lambda(sigma_c: torch.Tensor, fraction: float = 1e-2) -> float:
    """Default ridge ``fraction * tr(Sigma_c) / d`` (plan §11; #658 + B3)."""
    s = sigma_c.to(torch.float64)
    d = s.shape[0]
    return float(fraction * (torch.trace(s).item() / d))


def whitened_gate(
    c_c: torch.Tensor,
    c_cp: torch.Tensor,
    sigma_c: torch.Tensor,
    lam: float,
) -> float:
    """Whitened key-query gate ``g_C(C') = c_C^T Sinv c_C' / c_C^T Sinv c_C``.

    ``g_C(C) = 1`` by construction (self-normalized). With ``Sigma_c = I`` and
    equal-norm ``c`` this reduces to ``cos(c_C, c_C')`` (the B3 reduction limit).
    """
    sinv = sigma_inv_regularized(sigma_c, lam)
    z_c = sinv @ c_c.to(torch.float64)
    denom = float(z_c @ c_c.to(torch.float64))
    if denom == 0.0:
        raise ValueError("whitened_gate: c_C^T Sinv c_C == 0 (degenerate key)")
    return float((z_c @ c_cp.to(torch.float64)) / denom)


def whitened_gate_metric(
    c_c: torch.Tensor,
    c_cp: torch.Tensor,
    metric: str,
    sigma_c: torch.Tensor | None,
    lam: float,
) -> float:
    """Key-query gate under one of three metrics (A3.9 key x metric ablation).

    - ``"I"``: identity metric → ``cos(c_C, c_C')`` (self-normalized to
      ``g_C(C)=1``), the raw un-whitened cosine baseline.
    - ``"diag"``: diagonal-of-Sigma_c metric (per-feature variance whitening).
    - ``"whitened"``: full ``Sigma_c^-1`` (the boxed predictor, C3).
    """
    a = c_c.to(torch.float64)
    b = c_cp.to(torch.float64)
    if metric == "I":
        denom = float(a @ a)
        if denom == 0.0:
            raise ValueError("metric I: c_C has zero norm")
        return float((a @ b) / denom)
    if metric == "diag":
        assert sigma_c is not None, "diag metric needs sigma_c"
        diag = torch.diagonal(sigma_c.to(torch.float64)) + lam
        w = 1.0 / diag
        denom = float((a * w) @ a)
        if denom == 0.0:
            raise ValueError("metric diag: degenerate key")
        return float((a * w) @ b / denom)
    if metric == "whitened":
        assert sigma_c is not None, "whitened metric needs sigma_c"
        return whitened_gate(c_c, c_cp, sigma_c, lam)
    raise ValueError(f"unknown metric {metric!r} (expected I | diag | whitened)")


def whitened_gate_reduction_unit_test(d: int = 64, seed: int = 0) -> None:
    """B3 GATE: whitened gate reduces to cos(c_C, c_C') at Sigma_c=I / equal-norm.

    MUST pass before any A3.9/A3.10 number is trusted. A mis-implemented inverse
    otherwise manufactures a spurious "whitening wins". Raises ``AssertionError``
    with the failing ``(i, j)`` cell on mismatch.
    """
    torch.manual_seed(seed)
    c = torch.randn(5, d, dtype=torch.float64)
    c = c / c.norm(dim=1, keepdim=True)  # equal-norm rows
    sigma = torch.eye(d, dtype=torch.float64)
    for i in range(5):
        for j in range(5):
            g = whitened_gate(c[i], c[j], sigma, lam=0.0)
            cos_ratio = float((c[i] @ c[j]) / (c[i] @ c[i]))  # cos / 1 (g_C(C)=1)
            assert abs(g - cos_ratio) < 1e-5, (
                f"B3 reduction unit test FAILED at cell ({i},{j}): "
                f"whitened_gate={g:.8f} != cos_ratio={cos_ratio:.8f}"
            )
    # The diagonal self-gate must be exactly 1 (within fp tolerance).
    for i in range(5):
        g_self = whitened_gate(c[i], c[i], sigma, lam=0.0)
        assert abs(g_self - 1.0) < 1e-6, f"B3: self-gate g_C(C)={g_self} != 1 at i={i}"


# ─────────────────────────────────────────────────────────────────────────────
# A3.7 — source write points toward the data target (cos(w_hat, delta))
# ─────────────────────────────────────────────────────────────────────────────


def a37_source_write(
    w_hat: np.ndarray,
    delta_pos: np.ndarray,
    delta_contra: np.ndarray,
    delta_other_behavior: np.ndarray,
    v0_c: np.ndarray,
    v0_cneg: np.ndarray,
) -> dict[str, float]:
    """A3.7 source-write alignment reads for one source cell (plan §4.3).

    - ``cos_pos``: ``cos(w_hat, delta_pos)`` with ``delta_pos = t+ - v0(C)`` —
      the positive-only displacement read.
    - ``cos_contra``: ``cos(w_hat, delta_contra)`` with ``delta_contra =
      t+ - t-`` — the contrastive decomposition read.
    - ``cos_null``: ``cos(w_hat, delta_of_a_different_behavior)`` — the
      shuffled-delta null (R3-2); "strong" = beats this, not >0.
    - ``frac_ctx``: ``||v0(C) - v0(C_neg)|| / ||delta_contra||`` — the source-
      vs-negative context-axis offset (R3-1); a large frac_ctx explains a
      pos/contra divergence as context offset, not the negatives.
    - ``scalar_fit_residual_pos``: ``min_a ||w_hat - a*delta_pos|| / ||w_hat||``.
    """
    w = np.asarray(w_hat, dtype=np.float64)
    dp = np.asarray(delta_pos, dtype=np.float64)
    dc = np.asarray(delta_contra, dtype=np.float64)
    do = np.asarray(delta_other_behavior, dtype=np.float64)
    v0c = np.asarray(v0_c, dtype=np.float64)
    v0n = np.asarray(v0_cneg, dtype=np.float64)
    dc_norm = float(np.linalg.norm(dc))
    return {
        "cos_pos": cosine(w, dp),
        "cos_contra": cosine(w, dc),
        "cos_null": cosine(w, do),
        "frac_ctx": float(np.linalg.norm(v0c - v0n) / dc_norm) if dc_norm > 0 else float("nan"),
        "scalar_fit_residual_pos": _scalar_fit_residual(w, dp),
    }


def _scalar_fit_residual(w: np.ndarray, delta: np.ndarray) -> float:
    """``min_a ||w - a*delta|| / ||w||`` — the best scalar-fit residual of w onto delta."""
    w = np.asarray(w, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    dd = float(delta @ delta)
    wn = float(np.linalg.norm(w))
    if dd <= 0.0 or wn <= 0.0:
        return float("nan")
    a = float((w @ delta) / dd)
    return float(np.linalg.norm(w - a * delta) / wn)


# ─────────────────────────────────────────────────────────────────────────────
# A3.6 — base read-out predicts the post-FT behavior CHANGE (partial corr, C10)
# ─────────────────────────────────────────────────────────────────────────────


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman ``rho(x, y | z)`` — rank-residualize x and y on z, correlate.

    Used for A3.6: ``x = r_B'^T Delta_v(C')``, ``y = E+(C',B') - E0(C',B')``,
    ``z = E0(C',B')`` (the base level partialled out, C10). Rank-transform all
    three, OLS-residualize the ranks of x and y on the rank of z, then Pearson-
    correlate the residuals (== partial Spearman).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    assert x.shape == y.shape == z.shape, (x.shape, y.shape, z.shape)
    if x.size < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rz = _rankdata(z)
    ex = _ols_residual(rx, rz)
    ey = _ols_residual(ry, rz)
    return _pearson(ex, ey)


def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average-rank tie handling (matches scipy 'average')."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _ols_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual of OLS regression of a on b (with intercept)."""
    bc = b - b.mean()
    var = float(bc @ bc)
    if var == 0.0:
        return a - a.mean()
    slope = float((bc @ (a - a.mean())) / var)
    intercept = float(a.mean() - slope * b.mean())
    return a - (slope * b + intercept)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    if denom == 0.0:
        return 0.0
    return float((a @ b) / denom)


def readout_projection(r_b: np.ndarray, delta_v: np.ndarray) -> float:
    """``r_B'^T Delta_v`` — the A3.6 read-out projection of a target update."""
    return float(np.asarray(r_b, dtype=np.float64) @ np.asarray(delta_v, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Family-clustered bootstrap (plan §6 / scope caveat 3)
# ─────────────────────────────────────────────────────────────────────────────


def family_of(cid: str) -> str:
    """Map a #537 context id to its 7-family cluster label (plan §6).

    Prefix grammar: sp_/wc_/icl_/reph_/fmt_/binst_/default. Held-out cells keep
    their base family (sp_teacher_ho -> sp). Used to resample families for the
    clustered bootstrap CI (NEVER a naive n=30 CI).
    """
    for prefix in ("sp_", "wc_", "icl_", "reph_", "fmt_", "binst_"):
        if cid.startswith(prefix):
            return prefix.rstrip("_")
    return "default"


def clustered_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Family-clustered bootstrap CI on Spearman(x, y) (plan §6, scope caveat 3).

    Resamples whole FAMILIES with replacement (not individual contexts), so the
    CI respects the ~7-family cluster structure. Returns the point estimate
    plus the (1-alpha) CI.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fams = np.asarray(families, dtype=object)
    assert x.shape == y.shape == fams.shape, (x.shape, y.shape, fams.shape)
    point = spearman_rho(x, y)
    uniq = sorted(set(families))
    if len(uniq) < 2:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    fam_to_idx = {f: np.where(fams == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples, dtype=np.float64)
    n_fam = len(uniq)
    for r in range(n_resamples):
        chosen = rng.choice(uniq, size=n_fam, replace=True)
        idx = np.concatenate([fam_to_idx[f] for f in chosen])
        vals[r] = spearman_rho(x[idx], y[idx])
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_families": n_fam,
    }


def shuffled_null_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_reps: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Permutation null on Spearman(x, y): shuffle y, recompute, take the (1-alpha) band.

    The matched null for A3.6/A3.9 "beats chance" reads (plan §7). Returns the
    null distribution's upper band edge (a point estimate above it = a hit).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_reps, dtype=np.float64)
    for r in range(n_reps):
        vals[r] = spearman_rho(x, rng.permutation(y))
    return {
        "null_lo": float(np.percentile(vals, 100 * alpha / 2)),
        "null_hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "null_mean": float(vals.mean()),
    }


__all__ = [
    "a37_source_write",
    "bootstrap_ci",
    "clustered_bootstrap_spearman",
    "cosine",
    "default_lambda",
    "family_of",
    "partial_spearman",
    "readout_projection",
    "realized_gate",
    "shuffled_null_ci",
    "sigma_inv_regularized",
    "spearman_rho",
    "stacked_delta_svd",
    "whitened_gate",
    "whitened_gate_metric",
    "whitened_gate_reduction_unit_test",
]
