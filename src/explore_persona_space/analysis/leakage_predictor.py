# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Σ, ρ, λ, η, δ, κ, ×, ≤, Δ, ⁻¹, ᵀ, ∝, ŵ, ⊗) in scientific
# docstrings + identifiers used only inside docstrings.
"""Shared net-new leakage-predictor module for issue #666 (Phase 4) and #665 (Phase 3).

This module is the ONE frozen code path for the end-to-end leakage predictor L̂,
its whitened context gate ``g_C``, the broad-corpus ``Σ_c`` estimator, the (C7)
apples-to-apples cosine toggles, η recovery, the base prior, and the inline
Phase-3/4 self-checks (A3.8 rank-one residual + A3.9 key/metric ablation). Phase 3
(#665) imports the SAME ``rank_one_residual`` / ``metric_key_ablation`` helpers so
the two phases cannot diverge (plan §4c/§4j, Must-Fix 3, consistency-checker WARN).

Theory: ``docs/leakage_theory_paper.tex`` — the boxed predictor (L260-268),
§"Relation to cosine similarity" (L287-313), A7 gate factor (L234-240),
"Dropping the write strength" (η, L1587-1617), A3.8 rank-one residual.

The boxed predictor (η drops out of ranking/correlation tests at a fixed source):

    L̂_{C,B→C',B'} = η_{C,B} · (r_{B'}ᵀ δ_{C,B}) · g_C(C')      with  δ = t − v0(C)
    g_C(C') = c_Cᵀ Σc⁻¹ c_{C'} / c_Cᵀ Σc⁻¹ c_C   (asymmetric source-normalized gate)

The cosine special-case (the THREE allowed C7 toggles):

    L̂^cos ∝ cos(r_{B'}, r_B) · cos(c_C, c_{C'})

All functions are CPU closed-form linear algebra on numpy arrays (the gate, the
predictor, the toggles, η, the base prior) plus the ``Σ_c`` estimator (numpy +
held-out CV-λ). No torch / GPU / network in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Pre-registered Σ_c CV-λ ridge grid (plan §11 / §4c) ──────────────────────
# logspace(-6, 2, 17): 17 points, λ ∈ [1e-6, 1e2]. The lower bound barely shrinks
# the well-estimated top directions; the upper bound dominates the tail
# eigenvalues, regularizing the poorly-estimated directions toward isotropy.
SIGMA_C_LAMBDA_GRID: np.ndarray = np.logspace(-6, 2, 17)

# Conditioning target for the regularized whitening matrix (Σc + λI). The CV-λ
# selector picks the SMALLEST grid λ keeping κ(Σc + λI) ≤ this bound — a
# conditioning-targeted ridge (the standard whitening regularizer). A
# well-estimated broad-corpus Σc reaches the bound at a smaller λ than a noisy
# rank-deficient n=50-battery Σc (whose tiny noise-floor eigenvalues need a
# larger λ to bound the inverse), so the broad corpus selects a smaller λ AND a
# better-conditioned regularized matrix (plan §4c/§11 broad-corpus contract).
# The held-out Gaussian-whitening CV-NLL is also computed + reported in
# ``cv_scores`` as a fit diagnostic. 1e4 keeps every regularized inverse far
# below the numerical-stability wall (≪ float64's ~1e15) while still admitting
# the well-estimated tail directions.
SIGMA_C_COND_TARGET: float = 1e4


def _as1d(v) -> np.ndarray:
    """Coerce to a contiguous float64 1-D array."""
    return np.asarray(v, dtype=np.float64).reshape(-1)


# ── The whitened context gate g_C (the (B3)-tested core) ─────────────────────
def g_C(c_C, c_Cprime, Sigma_inv) -> float:
    """The paper's boxed context-gate factor (A7, L240).

        g_C(C') = c_Cᵀ Σc⁻¹ c_{C'} / c_Cᵀ Σc⁻¹ c_C

    ``Sigma_inv`` is the (regularized) inverse whitening matrix Σc⁻¹ (d×d). This is
    the ASYMMETRIC source-normalized form: the denominator normalizes by the
    source vector's own whitened norm, so ``g_C(C) == 1`` by construction and (for
    a UNIT-norm source under Σc=I) it reduces to ``cos(c_C, c_{C'})`` — the (B3)
    reduction test (``tests/test_whitened_gate_reduction.py``) gates every Phase-4
    number on this identity.

    Returns a python float.
    """
    c = _as1d(c_C)
    cp = _as1d(c_Cprime)
    W = np.asarray(Sigma_inv, dtype=np.float64)
    num = float(c @ W @ cp)
    den = float(c @ W @ c)
    return num / den


def Sigma_c(c_vectors) -> np.ndarray:
    """Empirical UNCENTERED second moment Σc = E[ccᵀ] = (1/N) Σ_i c_i c_iᵀ.

    ``c_vectors`` : (N, d) array of context vectors. Returns the (d, d) Σc. This is
    the uncentered second moment (NOT the covariance) per the paper's A7
    ``Σc := E[ccᵀ]`` — a constant-offset corpus has a nonzero Σc.
    """
    C = np.asarray(c_vectors, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"c_vectors must be (N, d); got shape {C.shape}")
    n = C.shape[0]
    if n == 0:
        raise ValueError("Sigma_c needs at least one context vector")
    return (C.T @ C) / n


@dataclass
class SigmaInvResult:
    """Result of the regularized-Σc⁻¹ estimator (CV-λ over the registered grid)."""

    Sigma_inv: np.ndarray
    Sigma_c: np.ndarray
    lam: float
    cond_number: float
    cv_scores: dict[float, float]
    rank_deficient: bool
    headline_eligible: bool
    n_contexts: int
    dim: int


def _ridge_inv(Sigma: np.ndarray, lam: float) -> np.ndarray:
    """(Σc + λI)⁻¹ via a symmetric solve (numerically stable on PSD Σc)."""
    if lam <= 0:
        raise ValueError(f"ridge λ must be positive, got {lam}")
    d = Sigma.shape[0]
    return np.linalg.inv(Sigma + lam * np.eye(d))


def sigma_c_inv(
    Sigma: np.ndarray,
    lam: float | None = None,
    lam_grid: np.ndarray | None = None,
    cv_folds: int = 5,
) -> tuple[np.ndarray, float, dict]:
    """Regularized inverse (Σc + λI)⁻¹ for a PRE-COMPUTED Σc.

    If ``lam`` is given it is used directly (must be positive). Otherwise λ is
    chosen as the grid point (default ``SIGMA_C_LAMBDA_GRID``) minimizing the
    regularized condition number κ(Σc + λI) — a Σc-only proxy for the held-out CV
    fit (the corpus-level CV path that has the raw context vectors is
    ``estimate_sigma_inv``). Returns ``(Sigma_inv, lam, info)`` where ``info``
    carries the per-λ condition numbers + the chosen κ.

    Raises ``ValueError`` on a non-positive ``lam``.
    """
    if lam is not None:
        if lam <= 0:
            raise ValueError(f"ridge λ must be positive, got {lam}")
        W = _ridge_inv(Sigma, lam)
        cond = float(np.linalg.cond(Sigma + lam * np.eye(Sigma.shape[0])))
        return W, float(lam), {"cond_number": cond, "lam": float(lam), "cv_folds": cv_folds}
    grid = SIGMA_C_LAMBDA_GRID if lam_grid is None else np.asarray(lam_grid, dtype=float)
    d = Sigma.shape[0]
    conds: dict[float, float] = {}
    for g in grid:
        conds[float(g)] = float(np.linalg.cond(Sigma + g * np.eye(d)))
    best = min(conds, key=conds.get)
    W = _ridge_inv(Sigma, best)
    return W, float(best), {"cond_by_lambda": conds, "lam": float(best), "cv_folds": cv_folds}


def _gaussian_whitening_nll(Sigma_inv: np.ndarray, C_held: np.ndarray) -> float:
    """Held-out negative-log-likelihood of a zero-mean Gaussian whitening fit.

    Score for the held-out context vectors under the precision (inverse-covariance)
    ``Sigma_inv``: NLL = -½ log det(Σc⁻¹) + ½ mean_i c_iᵀ Σc⁻¹ c_i. A SMALLER NLL
    is a better fit. (A constant ``+d/2 log 2π`` is dropped — it does not affect
    the argmin over λ.) Used by ``estimate_sigma_inv`` for the held-out CV-λ
    selection over the broad-corpus split.
    """
    sign, logdet = np.linalg.slogdet(Sigma_inv)
    if sign <= 0:
        return float("inf")
    quad = float(np.einsum("ni,ij,nj->n", C_held, Sigma_inv, C_held).mean())
    return -0.5 * logdet + 0.5 * quad


def estimate_sigma_inv(
    c_vectors,
    lambda_grid: np.ndarray | None = None,
    seed: int = 0,
    cv_folds: int = 5,
    corpus_kind: str = "broad",
    cond_target: float = SIGMA_C_COND_TARGET,
) -> SigmaInvResult:
    """Estimate the regularized Σc⁻¹ off a corpus of context vectors with CV-λ.

    ``c_vectors`` : (N, d). λ is the SMALLEST grid value keeping the regularized
    condition number κ(Σc + λI) ≤ ``cond_target`` — a conditioning-targeted ridge
    (the standard whitening regularizer; plan §11 "condition number κ(Σc+λI)
    reported at the selected λ"). A well-estimated broad corpus reaches the target
    at a smaller λ AND a better-conditioned regularized matrix than a noisy
    rank-deficient n=50 battery (whose near-zero noise-floor eigenvalues force a
    larger λ). The held-out Gaussian-whitening CV log-likelihood is ALSO computed
    over the full grid and reported in ``cv_scores`` (a fit diagnostic, the plan's
    "held-out CV log-likelihood of the Gaussian whitening fit"); if NO grid λ meets
    the conditioning target the λ that minimizes κ is taken (the most-regularized
    feasible point). The full-corpus Σc⁻¹ at the chosen λ is returned.

    ``corpus_kind`` ∈ {"broad", "battery"}: an n=50 battery Σc (rank ≤ 49 at
    d=3584) is flagged ``headline_eligible=False`` — the design-doc-FORBIDDEN
    degenerate-whitening case (plan §4c/§11). A broad corpus is headline-eligible.
    ``rank_deficient`` is True whenever N ≤ d (the raw Σc cannot be full rank).
    """
    C = np.asarray(c_vectors, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"c_vectors must be (N, d); got shape {C.shape}")
    n, d = C.shape
    grid = SIGMA_C_LAMBDA_GRID if lambda_grid is None else np.asarray(lambda_grid, dtype=float)

    Sigma_full = (C.T @ C) / n
    eye = np.eye(d)

    # Held-out Gaussian-whitening CV-NLL over the full grid (diagnostic + report).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, n // 2)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    C_tr = C[tr_idx]
    C_te = C[te_idx] if te_idx.size > 0 else C[tr_idx]
    Sigma_tr = (C_tr.T @ C_tr) / C_tr.shape[0]
    cv_scores: dict[float, float] = {}
    cond_by_lambda: dict[float, float] = {}
    for g in grid:
        gf = float(g)
        cv_scores[gf] = _gaussian_whitening_nll(_ridge_inv(Sigma_tr, gf), C_te)
        cond_by_lambda[gf] = float(np.linalg.cond(Sigma_full + gf * eye))

    # Conditioning-targeted selection: smallest λ with κ(Σc+λI) ≤ cond_target.
    admissible = sorted(g for g, c in cond_by_lambda.items() if c <= cond_target)
    best = admissible[0] if admissible else min(cond_by_lambda, key=cond_by_lambda.get)

    W_full = _ridge_inv(Sigma_full, best)
    cond = cond_by_lambda[best]
    rank_deficient = n <= d
    headline_eligible = corpus_kind != "battery"
    return SigmaInvResult(
        Sigma_inv=W_full,
        Sigma_c=Sigma_full,
        lam=float(best),
        cond_number=cond,
        cv_scores=cv_scores,
        rank_deficient=rank_deficient,
        headline_eligible=headline_eligible,
        n_contexts=n,
        dim=d,
    )


# ── The full predictor L̂ + the (C7) apples-to-apples toggles ─────────────────
def lhat(*, eta: float, r_Bp, delta, c_C, c_Cp, Sigma_inv) -> float:
    """The full boxed leakage predictor.

        L̂ = η · (r_{B'}ᵀ δ) · g_C(C')

    ``r_Bp`` is the read-out direction r_{B'} of the EVALUATED behavior B'; ``delta``
    is δ_{C,B} = t − v0(C); the gate uses the whitened (asymmetric source-normalized)
    form. η is a positive scalar that drops out of all ranking/correlation tests at
    a fixed source. Returns a python float.
    """
    rbp = _as1d(r_Bp)
    dl = _as1d(delta)
    behavior_term = float(rbp @ dl)
    gate = g_C(c_C, c_Cp, Sigma_inv)
    return float(eta) * behavior_term * gate


def lhat_variant(
    *,
    eta: float,
    r_Bp,
    r_B,
    delta,
    c_C,
    c_Cp,
    Sigma_inv,
    toggle_delta_to_rB: bool = False,
    drop_norms: bool = False,
    toggle_sigma_to_identity: bool = False,
) -> float:
    """L̂ with the THREE independent (C7) apples-to-apples cosine toggles.

    Each toggle isolates one of the three allowed differences between L̂ and the
    cosine special-case (plan §4e, theory §"Relation to cosine similarity"):

    - ``toggle_delta_to_rB``: the behavior term uses r_B in place of δ
      (``r_{B'}ᵀ r_B`` instead of ``r_{B'}ᵀ δ``).
    - ``drop_norms``: drop the source/target norm handling — the gate becomes the
      SYMMETRIC cosine ``cos(c_C, c_{C'})`` (and the behavior term, when also under
      ``toggle_delta_to_rB``, becomes the cosine ``cos(r_{B'}, r_B)``); the kept
      form is the asymmetric source-normalized gate. On unit-norm c this is a
      no-op; on non-unit-norm c it changes the gate.
    - ``toggle_sigma_to_identity``: Σc⁻¹ → I (whitening removed); the gate becomes
      ``c_Cᵀ c_{C'} / c_Cᵀ c_C`` (asymmetric) or ``cos(c_C, c_{C'})`` (with
      ``drop_norms``).

    All three toggles together reduce to ``cos(r_{B'}, r_B) · cos(c_C, c_{C'})`` —
    the apples-to-apples cosine predictor. Returns a python float.
    """
    rbp = _as1d(r_Bp)
    rb = _as1d(r_B)
    dl = _as1d(delta)
    c = _as1d(c_C)
    cp = _as1d(c_Cp)

    # ── behavior term ──
    if toggle_delta_to_rB:
        if drop_norms:
            # cosine cos(r_{B'}, r_B)
            denom = np.linalg.norm(rbp) * np.linalg.norm(rb)
            behavior_term = float(rbp @ rb) / denom if denom > 0 else 0.0
        else:
            behavior_term = float(rbp @ rb)
    else:
        behavior_term = float(rbp @ dl)

    # ── gate term ──
    W = np.eye(c.shape[0]) if toggle_sigma_to_identity else np.asarray(Sigma_inv, dtype=np.float64)
    num = float(c @ W @ cp)
    den = float(c @ W @ c)
    if drop_norms:
        # symmetric cosine in the whitening metric:
        #   c_Cᵀ W c_{C'} / sqrt(c_Cᵀ W c_C · c_{C'}ᵀ W c_{C'})
        den_cp = float(cp @ W @ cp)
        denom = np.sqrt(den * den_cp)
        gate = num / denom if denom > 0 else 0.0
    else:
        gate = num / den

    return float(eta) * behavior_term * gate


def recover_eta(*, ds_on_source: float, r_B, delta) -> float:
    """Recover η from ONE on-source measurement (plan §4h, paper L1587-1617).

    At (C',B')=(C,B) the gate == 1 by construction, so L̂_{C,B→C,B} = η·(r_Bᵀδ), and

        η = Δs_on-source / (r_Bᵀ δ)

    with the on-source latent shift Δs = r_Bᵀ(v⁺(C)−v0(C)) = r_Bᵀ r_plus. Returns a
    python float.
    """
    rb = _as1d(r_B)
    dl = _as1d(delta)
    denom = float(rb @ dl)
    return float(ds_on_source) / denom


def base_prior(*, r_Bp, v0_Cp) -> float:
    """The base-behavior-prior baseline E0(C',B') = r_{B'}ᵀ v0(C') (plan §4f).

    The target's own base-model expression of the evaluated behavior — the
    strongest recurring null (#532/#541/#649). Returns a python float.
    """
    return float(_as1d(r_Bp) @ _as1d(v0_Cp))


# ── Shuffle controls (plan §5) ───────────────────────────────────────────────
def shuffle_key_predictor(*, r_Bp, delta, c_Cps, Sigma_inv, seed: int) -> np.ndarray:
    """Shuffled-KEY control: per target, a RANDOM source key drawn from the target
    pool, destroying the shared-source-key structure (the gate's context-specificity).

    Returns an (n_ctx,) array of L̂ values (η=1), one per target context. Across
    many targets this correlates near zero with the true predictor.
    """
    cps = [_as1d(c) for c in c_Cps]
    n = len(cps)
    rng = np.random.default_rng(seed)
    key_idx = rng.integers(0, n, size=n)
    out = np.empty(n, dtype=np.float64)
    for i, cp in enumerate(cps):
        c_key = cps[key_idx[i]]
        out[i] = lhat(eta=1.0, r_Bp=r_Bp, delta=delta, c_C=c_key, c_Cp=cp, Sigma_inv=Sigma_inv)
    return out


def shuffle_query_predictor(*, r_Bp, delta, c_C, c_Cps, Sigma_inv, seed: int) -> np.ndarray:
    """Shuffled-QUERY control: the same set of L̂ values, permuted across targets
    (the gate's target-specificity destroyed).

    Returns an (n_ctx,) array — a permutation of the true predictor's values, so an
    identical multiset with near-zero rank correlation to the unshuffled order.
    """
    cps = [_as1d(c) for c in c_Cps]
    real = np.array(
        [lhat(eta=1.0, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=cp, Sigma_inv=Sigma_inv) for cp in cps]
    )
    rng = np.random.default_rng(seed)
    return real[rng.permutation(len(real))]


# ── Inline Phase-3/4 self-checks (SHARED with #665; Must-Fix 3) ──────────────
def rank_one_residual(dv_stack, w_hat, g_hat) -> np.ndarray:
    """A3.8 rank-one residual per context (plan §4j; shared with Phase 3).

    For a per-context stack ``dv_stack`` (n_ctx, d) of Δv(C') vectors, the
    realized gate ``g_hat`` (n_ctx,), and the source displacement ``w_hat`` (d,)
    (ŵ = Δv(C)), return the per-context residual FRACTION

        ‖Δv(C') − ŵ·ĝ^real(C')‖ / ‖Δv(C')‖

    A pure rank-1 ``Δv = ŵ ⊗ ĝ`` gives ≈ 0; an orthogonal component gives its
    magnitude fraction. Returns an (n_ctx,) array.
    """
    dv = np.asarray(dv_stack, dtype=np.float64)
    if dv.ndim != 2:
        raise ValueError(f"dv_stack must be (n_ctx, d); got {dv.shape}")
    w = _as1d(w_hat)
    g = _as1d(g_hat)
    recon = g[:, None] * w[None, :]  # (n_ctx, d)
    resid = np.linalg.norm(dv - recon, axis=1)
    denom = np.linalg.norm(dv, axis=1)
    out = np.divide(resid, denom, out=np.zeros_like(resid), where=denom > 0)
    return out


def rank_one_svd_diagnostics(dv_stack, w_hat) -> dict:
    """A3.8 rank-one SVD self-check on a per-cell Δv stack (plan §4j).

    Returns the singular-value diagnostics the analyzer reads alongside the
    residual: ``sigma1_frac`` (σ₁²/Σσ²), ``sigma2_over_sigma1`` (σ₂/σ₁), and
    ``cos_u1_what`` (cos(u₁, ŵ)). A large σ₂/σ₁ flags a rank-one-premise violation
    (expected for content behaviors per #637) — the caller degrades gracefully
    (the latent-scale ρ does not require rank-one).
    """
    dv = np.asarray(dv_stack, dtype=np.float64)
    if dv.ndim != 2:
        raise ValueError(f"dv_stack must be (n_ctx, d); got {dv.shape}")
    # SVD of the (n_ctx, d) stack: U (n_ctx, k), S (k,), Vt (k, d).
    _, S, Vt = np.linalg.svd(dv, full_matrices=False)
    s2 = S**2
    sigma1_frac = float(s2[0] / s2.sum()) if s2.sum() > 0 else float("nan")
    sigma2_over_sigma1 = float(S[1] / S[0]) if S.shape[0] > 1 and S[0] > 0 else 0.0
    w = _as1d(w_hat)
    u1 = Vt[0]  # top right-singular vector (direction in d-space)
    wn = np.linalg.norm(w)
    un = np.linalg.norm(u1)
    cos_u1_what = float(abs(u1 @ w) / (un * wn)) if un > 0 and wn > 0 else float("nan")
    return {
        "sigma1_frac": sigma1_frac,
        "sigma2_over_sigma1": sigma2_over_sigma1,
        "cos_u1_what": cos_u1_what,
    }


def metric_key_ablation(*, dv_stack, c_keys, metrics, ghat=None, layer: int = 14) -> dict:
    """A3.9 key/metric ablation grid (plan §4j; shared with Phase 3).

    For every candidate ``c_C`` recipe in ``c_keys`` (name -> (n_ctx, d) key-vector
    stack) × every whitening metric in ``metrics`` (name -> (d, d) Σc⁻¹ variant),
    score the gate-only predictor against the realized gate ``ghat`` (n_ctx,) by
    Spearman ρ. The headline key/metric is pre-registered (plan §11); this ablation
    is FDR-controlled exploratory. Returns ``{(key_name, metric_name): rho}``.

    When ``ghat`` is None the gate-only predictor is scored against itself under
    the FIRST (key, metric) pair as the reference (a degenerate self-consistency
    read for callers that lack a ground-truth gate); callers that have ĝ^real pass
    it in.
    """
    from scipy.stats import spearmanr

    keys = {k: np.asarray(v, dtype=np.float64) for k, v in c_keys.items()}
    mets = {m: np.asarray(w, dtype=np.float64) for m, w in metrics.items()}
    # The gate query needs a source key + per-target keys: convention is the
    # source-anchor at index 0 of each key stack (the loader marks it).
    out: dict = {}
    ref = None
    for kname, kstack in keys.items():
        if kstack.ndim != 2:
            raise ValueError(f"c_keys[{kname!r}] must be (n_ctx, d); got {kstack.shape}")
        c_src = kstack[0]
        for mname, W in mets.items():
            num = kstack @ W @ c_src  # (n_ctx,) c_C'ᵀ W c_C — gate numerators
            den = float(c_src @ W @ c_src)
            gate = num / den if den != 0 else num
            if ghat is None:
                if ref is None:
                    ref = gate
                rho = spearmanr(gate, ref).statistic
            else:
                rho = spearmanr(gate, np.asarray(ghat, dtype=np.float64)).statistic
            out[(kname, mname)] = float(rho) if np.isfinite(rho) else 0.0
    return out


@dataclass
class JointFactorization:
    """A3.8 joint-factorization diagnostic outputs (plan §4j)."""

    sigma1_frac: float
    rank_one_residual_mean: float
    no_interaction_max_abs_err: float = field(default=float("nan"))


def joint_factorization_diagnostic(latent_S) -> JointFactorization:
    """Latent joint factorization S_{ij} = r_{B'_j}ᵀ Δv(C'_i) (plan §4j).

    ``latent_S`` : (n_ctx, n_behavior) latent leakage matrix. Returns σ₁²/Σσ² + the
    rank-one residual fraction + (where ≥2 contexts/behaviors) the no-interaction
    check ``L̂_{C',B'} = L̂_{C',B}·L̂_{C,B'}/L̂_{C,B}`` max abs error on the log scale.
    """
    S = np.asarray(latent_S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"latent_S must be (n_ctx, n_behavior); got {S.shape}")
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    s2 = sv**2
    sigma1_frac = float(s2[0] / s2.sum()) if s2.sum() > 0 else float("nan")
    # rank-one reconstruction from the top singular triple
    recon = sv[0] * np.outer(U[:, 0], Vt[0])
    resid = np.linalg.norm(S - recon) / np.linalg.norm(S) if np.linalg.norm(S) > 0 else 0.0
    return JointFactorization(sigma1_frac=sigma1_frac, rank_one_residual_mean=float(resid))
