#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (math/scientific notation — Δv, ŵ, ψ, Σ_c, ρ — intentional in docstrings + labels)
"""Issue #683 Phase C — key × metric leaderboard scoring (CPU, off-pod).

Plan §4 Phase C + §6. Per behavior, scores each source-key form

    k ∈ { c_C,  ψ(t_{C,B}),  c_C + ψ(δ_{C,B}) }

crossed with each metric

    M ∈ { I,  (Σ_c + λI)^-1 }

and the cosine baseline cos(c_C, c_C'), by the bilinear predicted gate

    g_pred(C'_i) = (kᵀ M c_C'_i) / (kᵀ M c_C)

against the held-out realized gate g_real(C'_i) under a leaderboard-level
leave-one-CONTEXT-out loop: each scored target C'_i is excluded from the train
set that selects λ + fits the whitening Σ_c + the final metric M that scores it
(no held-out leakage — BLOCKER lambda-gcv-heldout-leak). For each (key, metric)
the leaderboard reports the held-out Spearman ρ (PRIMARY), Pearson r,
sign-agreement, MAE — each with a 1000-bootstrap CI over the LOO predictions —
plus a cross-seed range, the shuffled-KEY and shuffled-QUERY nulls (a
key/query-VECTOR permutation, NOT a matrix-axis relabel — methodology-critic
concern #2), and the test-retest noise floor.

A7 gating (plan §4 Phase B branch): reads
``eval_results/issue_683/<behavior>/a7_precondition.json`` FIRST. If rank-1
holds, the DV is the scalar g_real. If not, the low-rank fallback fits
Δv ≈ Σⱼ wⱼ gⱼ (m=1..3) and scores keys against the dominant component g₁
(the projection onto the stacked-SVD top-left singular direction). The branch
taken is recorded in the leaderboard.

Reuses ``issue637_heldout_predictive_test`` helpers: ``split_cells``,
``bootstrap_arm_ci``, ``paired_delta_ci``. The Spearman/MAE/sign scorers are
local (the #637 harness scores R², a different metric).

Inputs:
  - Δv banks (per source/seed) under analysis_tensors/dv/<behavior>/ — carry
    g_real + Δv (for the low-rank fallback) per context.
  - a c_C context-vector bank {ctx: c_C}: for marker, sliced at L14 from the
    #604 post-response bank (--c-bank). t_{C,B} from the t_cb extractor.
Output: ``eval_results/issue_683/<behavior>/key_ablation_leaderboard.json``.

CLI:
    uv run python scripts/issue683_key_ablation_score.py --behavior marker \
        --c-bank <#604 L14 c bank.pt> --tcb-dir <t_cb/marker> \
        --a7 eval_results/issue_683/marker/a7_precondition.json
    # CPU math smoke on a synthetic bank:
    uv run python scripts/issue683_key_ablation_score.py --behavior marker \
        --dv-dir eval_results/issue_683/smoke/synthetic_rank1/dv \
        --c-bank eval_results/issue_683/smoke/synthetic_rank1/c_bank.pt \
        --a7 eval_results/issue_683/smoke/a7_synth_rank1.json \
        --out eval_results/issue_683/smoke/leaderboard_marker_smoke.json \
        --n-boot 50 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_key_ablation_score")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

# Reuse the #637 held-out CV harness primitives (split + bootstrap CI).
from issue637_heldout_predictive_test import (  # noqa: E402
    bootstrap_arm_ci,  # noqa: F401  (re-exported for downstream parity / tests)
)

from explore_persona_space.experiments.issue_683 import DEFAULT_LAYER, repro_metadata  # noqa: E402

KEY_FORMS = ("k_cC", "k_tCB", "k_cC_plus_delta")
METRICS = ("M_I", "M_white")
PSI_FORMS = ("psi_I", "psi_ridge")
LAMBDA_GRID_MULT = (1e-3, 1e-2, 1e-1, 1.0, 1e1)  # × median-eigenvalue(Σ_c)


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ (rank correlation). NaN when degenerate (constant input)."""
    if len(y) < 2:
        return float("nan")
    pr = np.argsort(np.argsort(pred)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if pr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, yr)[0, 1])


def pearson(pred: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2 or pred.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def sign_agreement(pred: np.ndarray, y: np.ndarray, ref: float) -> float:
    """Fraction of contexts where pred and y fall on the same side of ref."""
    if len(y) == 0:
        return float("nan")
    return float((np.sign(pred - ref) == np.sign(y - ref)).mean())


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean()) if len(y) else float("nan")


def _median_nonzero_eig(eig: np.ndarray) -> float:
    """Median of the GENUINELY-nonzero eigenvalues (relative tolerance).

    For H≫n, Σ_c (H×H) has H−n structurally-zero eigenvalues; ``eigvalsh``
    returns them as ~1e-15 numerical noise, so a bare ``eig > 0`` mask admits
    those near-zero values and collapses the median toward 0 (making λ
    effectively 0 — a latent bug at the production H=3584, n~40 scale). Gate on
    a relative tolerance off the largest eigenvalue so only the n real nonzero
    eigenvalues enter the median. The dual (Woodbury) path takes the SAME
    median over the n×n Gram spectrum, so both metric forms pick the same λ.
    """
    if eig.size == 0:
        return 1.0
    tol = float(eig.max()) * 1e-9
    pos = eig[eig > tol]
    return float(np.median(pos)) if pos.size else 1.0


def _whiten_metric(c_matrix: np.ndarray, lam_mult: float) -> np.ndarray:
    """M = (Σ_c + λI)^-1 with λ = lam_mult × median-nonzero-eigenvalue(Σ_c).

    Σ_c = (1/n) Cᵀ C over the context vectors (H×H). For H≫n this is
    rank-deficient, so the +λI regularizer is load-bearing. CUBIC reference
    form (materializes + inverts the full H×H matrix) — used by the direct-λ
    tests and as the correctness oracle for ``_WhitenDual``. The production hot
    path (``_loo_predictions`` / ``_select_lambda_heldout_gcv``) routes through
    ``_WhitenDual`` instead, which never forms the H×H matrix
    (scorer-cubic-whitening-infeasible).
    """
    n = c_matrix.shape[0]
    sigma_c = (c_matrix.T @ c_matrix) / n  # (H, H)
    eig = np.linalg.eigvalsh(sigma_c)
    med = _median_nonzero_eig(eig)
    lam = lam_mult * med
    return np.linalg.inv(sigma_c + lam * np.eye(sigma_c.shape[0]))


class _WhitenDual:
    """Woodbury/dual evaluator for M = (Σ_c + λI)^-1 in the n-context subspace.

    Σ_c = B Bᵀ with B = Cᵀ/√n (H×n); A = λI + B Bᵀ. Woodbury gives, for any
    vectors u, v,

        uᵀ A^-1 v = (1/λ)(uᵀv) − (1/λ²) (Bᵀu)ᵀ (I_n + (1/λ)G)^-1 (Bᵀv),

    where G = BᵀB = (1/n) C Cᵀ is the (n×n) Gram. Only an n×n solve is ever
    needed — never the H×H inverse — so a key×metric cell costs O(n³ + n²H)
    instead of O(H³). The G eigendecomposition is computed ONCE (for the
    median-eigenvalue λ scale and to reuse across λ candidates); the
    (I + G/λ) system is solved per query batch. Validated bit-for-bit against
    ``_whiten_metric`` (the cubic oracle) in the tests.

    λ = lam_mult × median-nonzero-eigenvalue(Σ_c); the Gram's nonzero spectrum
    equals Σ_c's, so ``_median_nonzero_eig`` over the Gram eigenvalues picks
    the SAME λ the cubic path would.
    """

    def __init__(self, c_matrix: np.ndarray):
        n = c_matrix.shape[0]
        self.n = n
        self.b = c_matrix.T / np.sqrt(n)  # (H, n)
        self.gram = self.b.T @ self.b  # (n, n) = (1/n) C Cᵀ
        self.eye_n = np.eye(n)
        self.med = _median_nonzero_eig(np.linalg.eigvalsh(self.gram))

    def lam(self, lam_mult: float) -> float:
        return lam_mult * self.med

    def quad(self, u: np.ndarray, v: np.ndarray, lam_mult: float) -> float:
        """uᵀ (Σ_c + λI)^-1 v via the dual form (no H×H matrix)."""
        lam = self.lam(lam_mult)
        if lam <= 0:
            lam = 1e-12  # degenerate Σ_c (all-zero context bank) — fall back to (1/λ)I
        bu = self.b.T @ u  # (n,)
        bv = self.b.T @ v  # (n,)
        inner = np.linalg.solve(self.eye_n + (1.0 / lam) * self.gram, bv)  # (n,)
        return float((1.0 / lam) * (u @ v) - (1.0 / lam**2) * (bu @ inner))

    def g_pred(
        self, k: np.ndarray, c_query: np.ndarray, c_source: np.ndarray, lam_mult: float
    ) -> float:
        """g_pred = (kᵀ M c_query) / (kᵀ M c_source) with M = (Σ_c + λI)^-1, dual."""
        num = self.quad(k, c_query, lam_mult)
        den = self.quad(k, c_source, lam_mult)
        if den == 0:
            return float("nan")
        return num / den


def _fit_ridge_psi(t_train: np.ndarray, c_train: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Learned linear map ψ: t-space → c-space, ridge, fit on TRAIN contexts only.

    Returns W (H×H) s.t. ψ(t) = W t minimizes ‖W T − C‖² + λ‖W‖². Closed form
    W = C Tᵀ (T Tᵀ + λI)^-1 over the TRAIN context matrices (rows = contexts).
    """
    t = t_train  # (n_train, H)
    c = c_train  # (n_train, H)
    g = t.T @ t + lam * np.eye(t.shape[1])  # (H, H)
    return (c.T @ t) @ np.linalg.inv(g)  # (H, H)


def _fit_ridge_psi_for_targets(
    per_context: dict, c_bank: dict[str, np.ndarray], targets: list[str]
) -> np.ndarray | None:
    """ψ: answer-side residual space → prompt-side query space (the robustness arm).

    Fit on the per-context (v_base[ctx] = answer-side base read, c_C[ctx] =
    prompt-side context vector) pairs that exist per target context (A8: "ψ maps
    answer-side or data-side vectors into the key–query space"). The source is
    held out (it is the denominator); the learned map applies to t_{C,B} (itself
    an answer-side mean). Returns None when <3 paired contexts exist.

    BLOCKER ``psi-ridge-heldout-leak``: this fits ψ on EXACTLY the ``targets``
    it is passed — the caller (``_psi_per_fold``) excludes each held-out target
    from the ``targets`` list it passes for that target's fold, so the ψ that
    scores C'_i never trains on C'_i's own (v_base, c_C') pair. Calling this on
    the FULL target panel (the round-3 leak) is now only done where no LOO read
    is required (it never is for ψ_ridge keys — see ``_psi_per_fold``).
    """
    import torch

    pair_ctx = [c for c in targets if c in per_context and "v_base" in per_context[c]]
    if len(pair_ctx) < 3:
        return None
    t_mat = np.stack(
        [torch.as_tensor(per_context[c]["v_base"]).flatten().float().numpy() for c in pair_ctx],
        axis=0,
    )
    c_mat = np.stack([c_bank[c] for c in pair_ctx], axis=0)
    return _fit_ridge_psi(t_mat, c_mat)


def _psi_per_fold(
    per_context: dict, c_bank: dict[str, np.ndarray], targets: list[str]
) -> dict[str, np.ndarray] | None:
    """Per-outer-LOO-fold ψ maps for ψ_ridge keys (BLOCKER psi-ridge-heldout-leak).

    Returns ``{held: psi_W}`` where each ``psi_W`` is fit on the (v_base, c_C')
    pairs of the OTHER targets — the held-out target ``held`` EXCLUDED. The key
    that scores ``held`` is then ``psi_W @ t_cb`` (built per target in
    ``score_bank``), so the held-out target's own pair never enters the ψ that
    produces the key scoring it. Pivot option (b): n folds of ψ pre-computed
    outside the cell loop, indexed by held-out target inside the LOO read.

    Mirrors ``_fit_ridge_psi_for_targets``'s ``<3 paired contexts`` rule: an
    individual fold with <3 remaining pairs yields ``None`` for that fold's
    target (the t-based key is then skipped for that target — handled by
    ``_loo_predictions`` / ``score_bank``). Returns ``None`` (no ψ_ridge at all)
    when fewer than 4 paired contexts exist (every fold would drop below 3).
    """
    pair_ctx = [c for c in targets if c in per_context and "v_base" in per_context[c]]
    if len(pair_ctx) < 4:
        # every leave-one-out fold would have <3 pairs — ψ_ridge not applicable.
        return None
    out: dict[str, np.ndarray] = {}
    for held in targets:
        train = [c for c in targets if c != held]
        psi_w = _fit_ridge_psi_for_targets(per_context, c_bank, train)
        if psi_w is not None:
            out[held] = psi_w
    return out or None


def _key_vector(
    form: str,
    psi_W: np.ndarray | None,
    *,
    c_source: np.ndarray,
    t_cb: np.ndarray | None,
    delta_cb: np.ndarray | None,
) -> np.ndarray | None:
    """Build the source key vector for one (key_form, ψ) cell.

    k_cC = c_C; k_tCB = ψ(t_{C,B}); k_cC_plus_delta = c_C + ψ(δ_{C,B}).
    Returns None when a t-based key is requested but t_{C,B}/δ are absent.
    """
    if form == "k_cC":
        return c_source
    if form == "k_tCB":
        if t_cb is None:
            return None
        return (psi_W @ t_cb) if psi_W is not None else t_cb
    if form == "k_cC_plus_delta":
        if delta_cb is None:
            return None
        psi_d = (psi_W @ delta_cb) if psi_W is not None else delta_cb
        return c_source + psi_d
    raise ValueError(form)


def _resolve_cell_keys(
    *,
    psi: str,
    key_form: str,
    psi_per_fold: dict[str, np.ndarray] | None,
    c_source: np.ndarray,
    t_cb: np.ndarray | None,
    delta_cb: np.ndarray | None,
    targets: list[str],
) -> tuple[np.ndarray | None, dict[str, np.ndarray] | None] | None:
    """Resolve the key(s) for one (ψ, key_form) cell — the leak-free key builder.

    Returns ``(k, k_per_target)`` (exactly one non-None) or ``None`` to SKIP the
    cell. BLOCKER ``psi-ridge-heldout-leak``: the ψ_ridge t-based keys (k_tCB,
    k_cC_plus_delta) are target-DEPENDENT — one key per held-out target, built
    from that fold's leave-it-out ψ (``psi_per_fold``), returned as
    ``k_per_target`` so the held-out target's own pair never enters the ψ that
    produces the key scoring it. Everything else (ψ_I keys, and k_cC under either
    ψ — k_cC is ψ-independent) returns a single target-independent ``k``.
    """
    use_per_fold = psi == "psi_ridge" and key_form != "k_cC"
    if use_per_fold:
        if psi_per_fold is None:
            return None  # ridge ψ unavailable (too few paired contexts) — skip
        k_per_target: dict[str, np.ndarray] = {}
        for held, psi_w in psi_per_fold.items():
            k_held = _key_vector(key_form, psi_w, c_source=c_source, t_cb=t_cb, delta_cb=delta_cb)
            if k_held is not None:
                k_per_target[held] = k_held
        if len(k_per_target) < 3:
            return None  # <3 folds with a usable per-fold key — cannot score
        return None, k_per_target
    # ψ_I keys + k_cC: a single target-independent key. (k_cC ignores ψ entirely,
    # so its psi_I / psi_ridge rows are identical — the pre-existing duplicate,
    # preserved.)
    k = _key_vector(key_form, None, c_source=c_source, t_cb=t_cb, delta_cb=delta_cb)
    if k is None:
        return None
    return k, None


def _g_pred(
    k: np.ndarray, m: np.ndarray | None, c_query: np.ndarray, c_source: np.ndarray
) -> float:
    """g_pred = (kᵀ M c_query) / (kᵀ M c_source); M=None ⇒ identity, M=dense ⇒
    the explicit-matrix form (cubic-oracle / test path)."""
    if m is None:
        num = float(k @ c_query)
        den = float(k @ c_source)
    else:
        km = k @ m
        num = float(km @ c_query)
        den = float(km @ c_source)
    if den == 0:
        return float("nan")
    return num / den


def _eval_g_pred(evaluator, k, c_query, c_source) -> float:
    """Dispatch g_pred on the resolved metric evaluator from ``_resolve_metric``.

    ``evaluator`` is one of: None (identity), or a ``(_WhitenDual, lam_mult)``
    pair (the production M_white dual form — no H×H matrix). A dense ndarray is
    also accepted (the cubic-oracle path) for parity testing.
    """
    if evaluator is None:
        return _g_pred(k, None, c_query, c_source)
    if isinstance(evaluator, tuple):
        dual, lam_mult = evaluator
        return dual.g_pred(k, c_query, c_source, lam_mult)
    # dense matrix (cubic oracle)
    return _g_pred(k, evaluator, c_query, c_source)


def _eval_den(evaluator, k, c_source) -> float:
    """|kᵀ M c_C| — the gate denominator (CONCERN denominator-stability-control).

    The same ``(k, M, c_source)`` tuple the scorer divides by in ``g_pred`` (plan
    v2 §6 / L152 pre-registers the |kᵀMc_C| denominator-stability diagnostic). A
    denominator near zero is what produces the divide-by-(near-)zero NaN/blowup
    in g_pred, so per-cell min/median |den| over the LOO folds quantifies how
    close the gate ran to the singular regime. ``evaluator`` matches
    ``_eval_g_pred``: None (identity ⇒ kᵀc_C), a ``(_WhitenDual, lam_mult)`` pair
    (dual M_white ⇒ kᵀ(Σ_c+λI)^-1 c_C), or a dense matrix (cubic oracle).
    """
    if evaluator is None:
        return float(abs(k @ c_source))
    if isinstance(evaluator, tuple):
        dual, lam_mult = evaluator
        return float(abs(dual.quad(k, c_source, lam_mult)))
    # dense matrix (cubic oracle)
    return float(abs((k @ evaluator) @ c_source))


def _select_lambda_heldout_gcv(
    *,
    k: np.ndarray,
    c_source: np.ndarray,
    c_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[float, float]:
    """Pick λ for (Σ_c+λI)^-1 by held-out cross-validation over TRAIN contexts.

    Leave-one-context-out (k-fold for n_train > n_folds) over the TRAIN target
    contexts: for each λ candidate, the whitening Σ_c AND the metric are fit on
    the TRAIN-fold contexts ONLY (no held-out leakage — Major #9), then the
    held-out fold's g_pred is scored against its g_real by held-out MAE. The λ
    minimizing the mean held-out MAE wins (lower reconstruction error = better,
    so two λ candidates with KNOWN different held-out errors select the
    lower-error one — BLOCKER lambda-gcv-not-implemented). Returns
    (best_lambda_mult, best_heldout_mae). Σ_c is NEVER fit on the full bank.
    """
    n = c_train.shape[0]
    if n < 2:
        # degenerate: cannot CV — fall back to the middle of the grid.
        return 1.0, float("nan")
    rng = np.random.default_rng(seed)
    # leave-one-out when small; else k contiguous shuffled folds.
    n_folds = min(n_folds, n) if n_folds > 0 else n
    n_folds = max(2, min(n_folds, n))
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)

    # Build ONE dual whitener per inner fold (Σ_c fit on the fold's TRAIN
    # contexts only) and REUSE it across all λ candidates — the Gram
    # eigendecomposition is computed once per fold, not per (fold, λ). The dual
    # form (Woodbury, n-context subspace) never materializes the H×H matrix
    # (scorer-cubic-whitening-infeasible).
    fold_duals: list[tuple[_WhitenDual, np.ndarray, np.ndarray]] = []
    for held in folds:
        held = np.asarray(held, dtype=int)
        train_mask = np.ones(n, dtype=bool)
        train_mask[held] = False
        if train_mask.sum() < 1 or held.size < 1:
            continue
        # Σ_c + λI fit on the TRAIN fold contexts ONLY (no held-out leakage).
        fold_duals.append((_WhitenDual(c_train[train_mask]), held, y_train[held]))

    best_mult, best_mae = 1.0, np.inf
    for mult in LAMBDA_GRID_MULT:
        fold_maes: list[float] = []
        for dual, held, yy in fold_duals:
            preds = np.array(
                [dual.g_pred(k, c_train[i], c_source, mult) for i in held], dtype=float
            )
            f = np.isfinite(preds) & np.isfinite(yy)
            if f.sum() >= 1:
                fold_maes.append(float(np.abs(preds[f] - yy[f]).mean()))
        mean_mae = float(np.mean(fold_maes)) if fold_maes else np.inf
        if mean_mae < best_mae:
            best_mae, best_mult = mean_mae, mult
    return best_mult, best_mae


def _resolve_metric(
    metric: str,
    k: np.ndarray,
    *,
    c_source: np.ndarray,
    c_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[tuple[_WhitenDual, float] | None, dict]:
    """Resolve the metric evaluator for a (key, metric) cell, given a TRAIN set.

    M_I → None (identity; ``_g_pred`` with m=None). M_white → a
    ``(_WhitenDual, lambda_mult)`` pair: the dual Woodbury evaluator for
    (Σ_c+λI)^-1 fit on ``c_train`` (never the H×H matrix —
    scorer-cubic-whitening-infeasible) with λ chosen by held-out
    cross-validation over the supplied TRAIN contexts
    (``_select_lambda_heldout_gcv``). BOTH Σ_c and λ are fit on the supplied
    ``c_train`` / ``y_train`` ONLY (no leakage — BLOCKER
    lambda-gcv-not-implemented + Major Σ_c-train-only). The CALLER is
    responsible for the leaderboard-level leave-one-context-out:
    ``_loo_predictions`` passes a ``c_train`` / ``y_train`` that EXCLUDES the
    held-out target being scored, so the scored target's c_C' / g_real never
    enter the metric that scores it (BLOCKER lambda-gcv-heldout-leak). Returns
    (evaluator | None, lambda_meta).
    """
    if metric == "M_I":
        return None, {"lambda_mult": None, "lambda_selection": "identity"}
    best_mult, best_mae = _select_lambda_heldout_gcv(
        k=k,
        c_source=c_source,
        c_train=c_train,
        y_train=y_train,
        n_folds=n_folds,
        seed=seed,
    )
    # Final whitener over the TRAIN contexts (source + train targets) at the
    # selected λ — held-out targets being scored are NOT in c_train (the caller
    # excludes them), so there is no held-out leakage into the whitening. The
    # dual form scores via the n-context subspace, never the H×H inverse.
    final_dual = _WhitenDual(c_train)
    return (final_dual, float(best_mult)), {
        "lambda_mult": float(best_mult),
        "lambda_selection": "heldout_gcv_mae",
        "heldout_mae": best_mae if best_mae == best_mae else None,
    }


def _loo_predictions(
    *,
    metric: str,
    k: np.ndarray | None = None,
    k_per_target: dict[str, np.ndarray] | None = None,
    c_source: np.ndarray,
    c_query: dict[str, np.ndarray],
    targets: list[str],
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Leave-one-CONTEXT-out leaderboard predictions for one (key, metric) cell.

    The KEY is supplied EITHER as a single ``k`` (target-INDEPENDENT keys —
    k_cC, and the t-based keys under ``psi_I``: the key vector is the same no
    matter which target is scored) OR as ``k_per_target`` (target-DEPENDENT
    keys — the ψ_ridge t-based keys, BLOCKER ``psi-ridge-heldout-leak``: each
    target's key is built from a ψ fit on the OTHER targets, so ``k`` itself
    varies per fold). Exactly one of the two MUST be supplied. With
    ``k_per_target`` a target absent from the dict (its fold's ψ had <3 pairs)
    is scored NaN.

    BLOCKER ``lambda-gcv-heldout-leak``: the OUTER loop holds out each scored
    target C'_i from the train set, so the scored target's OWN g_real never
    enters either the λ-GCV vector (``y_train_i``) OR the whitening Σ_c
    (``c_train_i``) used to score it. For each held-out C'_i:

      1. ``train_targets = [c for c in targets if c != C'_i]``
      2. ``c_train_i = [c_source, *c_query[train_targets]]`` — Σ_c is fit on
         THIS train set only (no held-out target's c_C' in the whitening).
      3. ``y_train_i = [1.0, *y[train_targets]]`` — the λ-GCV vector, the
         held-out target's g_real EXCLUDED.
      4. ``λ`` + the dual whitener ← ``_resolve_metric`` on
         ``(c_train_i, y_train_i)`` (Woodbury, n-context subspace — never the
         H×H matrix).
      5. ``pred_i = _eval_g_pred(evaluator, k_i, c_query[C'_i], c_source)`` —
         C'_i scored ONLY by a metric fit on the OTHER targets, with the
         fold-specific key ``k_i``.

    For M_I (identity, no λ, no Σ_c) the per-fold refit is a mathematical no-op
    — g_pred is independent of the train set — so when the key is also
    target-independent (``k`` supplied) a single resolve suffices and the loop is
    short-circuited (same numbers, no wasted folds). With ``k_per_target`` the
    key still varies per target, so M_I iterates the targets (no λ refit, but
    the per-target key is applied). Returns (preds aligned to ``targets``,
    lambda_meta).
    """
    if (k is None) == (k_per_target is None):
        raise ValueError("_loo_predictions requires EXACTLY one of k / k_per_target")
    n = len(targets)

    def _key_for(held: str) -> np.ndarray | None:
        return k if k_per_target is None else k_per_target.get(held)

    if metric == "M_I":
        if k_per_target is None:
            # Identity metric + a target-independent key: g_pred has no fitted
            # parameters AND the same key everywhere, so the LOO refit is an
            # exact no-op. One resolve, predict every target. |kᵀc_C| is the
            # same for every fold (one key, identity M) — record the single value.
            m, lam_meta = _resolve_metric(
                metric,
                k,
                c_source=c_source,
                c_train=np.stack([c_source, *[c_query[c] for c in targets]], axis=0),
                y_train=np.concatenate([[1.0], y]),
                n_folds=n_folds,
                seed=seed,
            )
            preds = np.array([_eval_g_pred(m, k, c_query[c], c_source) for c in targets])
            lam_meta = {**lam_meta, "denominator_abs_per_fold": [_eval_den(m, k, c_source)] * n}
            return preds, lam_meta
        # Identity metric but a per-fold (ψ_ridge) key: no λ refit, but the key
        # differs per target, so score each target with ITS fold's key.
        preds = np.empty(n, dtype=float)
        dens: list[float] = []
        for i, held in enumerate(targets):
            k_i = _key_for(held)
            if k_i is None:
                preds[i] = np.nan
                continue
            preds[i] = _eval_g_pred(None, k_i, c_query[held], c_source)
            dens.append(_eval_den(None, k_i, c_source))
        return preds, {
            "lambda_mult": None,
            "lambda_selection": "identity",
            "denominator_abs_per_fold": dens,
        }

    # M_white: an OUTER leave-one-context-out loop. Each held-out target is
    # scored by a metric (λ + Σ_c + final M) fit ONLY on the source + the OTHER
    # targets — the scored target's c_C' / g_real never touch its own predictor.
    # When the key is per-fold (ψ_ridge), that fold's key ALSO drives the λ fit
    # for the same fold, so the held-out target leaks into neither the metric
    # NOR the key that scores it.
    preds = np.empty(n, dtype=float)
    fold_lambdas: list[float] = []
    fold_maes: list[float] = []
    fold_dens: list[float] = []
    for i, held in enumerate(targets):
        k_i = _key_for(held)
        if k_i is None:
            preds[i] = np.nan
            continue
        train_targets = [c for c in targets if c != held]
        c_train_i = np.stack([c_source, *[c_query[c] for c in train_targets]], axis=0)
        # y_train_i pairs g_real with the SAME rows as c_train_i (source first).
        y_train_i = np.concatenate(
            [[1.0], np.array([y[targets.index(c)] for c in train_targets], dtype=float)]
        )
        m_i, lam_meta_i = _resolve_metric(
            metric,
            k_i,
            c_source=c_source,
            c_train=c_train_i,
            y_train=y_train_i,
            n_folds=n_folds,
            seed=seed,
        )
        preds[i] = _eval_g_pred(m_i, k_i, c_query[held], c_source)
        # |kᵀ M c_C| from the SAME (k_i, M_i, c_source) tuple the fold scored with.
        fold_dens.append(_eval_den(m_i, k_i, c_source))
        if lam_meta_i.get("lambda_mult") is not None:
            fold_lambdas.append(float(lam_meta_i["lambda_mult"]))
        hm = lam_meta_i.get("heldout_mae")
        if hm is not None and hm == hm:
            fold_maes.append(float(hm))
    lam_meta = {
        "lambda_selection": "heldout_gcv_mae_outer_loo",
        "lambda_mult_per_fold": fold_lambdas,
        "lambda_mult_median": (float(np.median(fold_lambdas)) if fold_lambdas else None),
        "heldout_mae_mean": (float(np.mean(fold_maes)) if fold_maes else None),
        "n_loo_folds": n,
        "denominator_abs_per_fold": fold_dens,
    }
    return preds, lam_meta


def _denominator_stability(lam_meta: dict) -> dict:
    """min / median |kᵀM c_C| over the LOO folds (CONCERN denominator-stability).

    Plan v2 §6 / L152 pre-registers the |kᵀMc_C| denominator-stability control
    per cell — a near-zero denominator is the divide-by-(near-)zero regime that
    makes g_pred unstable. The per-fold |den| list comes from ``_loo_predictions``
    (``denominator_abs_per_fold``); aggregate to min (the worst-case fold) +
    median (the typical fold). Empty / absent ⇒ Nones (a degenerate cell with no
    scorable fold), never a silent 0.
    """
    dens = [float(d) for d in (lam_meta.get("denominator_abs_per_fold") or []) if d == d]
    if not dens:
        return {"min": None, "median": None, "n_folds": 0}
    return {
        "min": float(np.min(dens)),
        "median": float(np.median(dens)),
        "n_folds": len(dens),
    }


def _score_one_cell(
    *,
    key_form: str,
    metric: str,
    psi: str,
    k: np.ndarray | None = None,
    k_per_target: dict[str, np.ndarray] | None = None,
    c_source: np.ndarray,
    c_query: dict[str, np.ndarray],
    targets: list[str],
    y: np.ndarray,
    a7_rank1: bool,
    n_boot: int,
    n_folds: int,
    seed: int,
    rng,
) -> dict | None:
    """Score ONE (key, metric, ψ) cell: leave-one-context-out Spearman / Pearson
    / sign / MAE + bootstrap Spearman CI. Returns None when <3 contexts score
    finitely.

    The KEY is supplied EITHER as a single ``k`` (target-independent: k_cC, or
    t-based keys under ψ_I) OR as ``k_per_target`` (the ψ_ridge t-based keys —
    BLOCKER ``psi-ridge-heldout-leak``: each target's key is built from a ψ fit
    on the OTHER targets). Exactly one of the two is supplied; threaded straight
    to ``_loo_predictions``.

    BLOCKER ``lambda-gcv-heldout-leak``: predictions come from
    ``_loo_predictions`` — an OUTER leave-one-context-out loop. For M_white,
    each scored target C'_i is held out of the train set, so its own g_real
    never enters the λ-GCV vector AND its own c_C' never enters the whitening
    Σ_c that scores it. (The earlier round-2 code fit λ + Σ_c + final M on the
    WHOLE target set, then scored those same targets — each target's own g_real
    biased the λ that scored it, inflating M_white's held-out ρ differentially
    vs the parameter-free M_I.) The inner ``_select_lambda_heldout_gcv`` CV
    still folds WITHIN each train set; the outer loop is the leaderboard-level
    leave-one-out the plan + docstring pre-register.

    CONCERN ``denominator-stability-control-not-persisted``: the row carries a
    ``denominator_stability`` field (min / median |kᵀMc_C| across folds) from
    the SAME (k, M, c_source) tuple the scorer divides by (plan v2 §6 / L152).
    """
    preds, lam_meta = _loo_predictions(
        metric=metric,
        k=k,
        k_per_target=k_per_target,
        c_source=c_source,
        c_query=c_query,
        targets=targets,
        y=y,
        n_folds=n_folds,
        seed=seed,
    )
    finite = np.isfinite(preds) & np.isfinite(y)
    if finite.sum() < 3:
        return None
    p, yy = preds[finite], y[finite]
    ref = 0.0 if not a7_rank1 else float(np.median(yy))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yy), len(yy))
        boot.append(spearman(p[idx], yy[idx]))
    boot = np.array([b for b in boot if b == b])
    ci = (
        [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if boot.size
        else [float("nan"), float("nan")]
    )
    return {
        "key": key_form,
        "metric": metric,
        "psi": psi,
        "n_scored": int(finite.sum()),
        "spearman": spearman(p, yy),
        "pearson": pearson(p, yy),
        "sign_agreement": sign_agreement(p, yy, ref),
        "mae": mae(p, yy),
        "spearman_ci95": ci,
        "denominator_stability": _denominator_stability(lam_meta),
        "lambda_selection": lam_meta,
    }


def _load_dv_banks(dv_dir: Path) -> list[dict]:
    import torch

    banks = []
    for p in sorted(dv_dir.glob("*_L*.pt")):
        banks.append(torch.load(p, map_location="cpu", weights_only=False))
    if not banks:
        raise FileNotFoundError(f"no Δv bank .pt files under {dv_dir}")
    return banks


def _load_c_bank(c_bank_path: Path, layer: int) -> dict[str, np.ndarray]:
    """Load {ctx: c_C} as numpy. Accepts a synthetic {contexts:{ctx:(H,)}} OR
    the #604 {contexts:{ctx:(28,H)}} all-layer bank (sliced at ``layer``)."""
    import torch

    obj = torch.load(c_bank_path, map_location="cpu", weights_only=False)
    ctxs = obj.get("contexts", obj)
    out: dict[str, np.ndarray] = {}
    for name, v in ctxs.items():
        t = torch.as_tensor(v)
        if t.ndim == 2:  # (n_layers, H) → slice the read layer
            t = t[min(layer, t.shape[0] - 1)]
        out[name] = t.flatten().float().numpy()
    return out


def _load_tcb(tcb_dir: Path, behavior: str, source: str, layer: int) -> np.ndarray | None:
    """Load t_{C,B} for (source, layer) from the t_cb extractor output.

    Matches ``L{layer}`` DETERMINISTICALLY (Minor _load_tcb-ignores-layer): the
    requested-layer file is preferred; if a t_cb file for (behavior, source)
    exists at a DIFFERENT layer but not at ``layer``, that is a layer mismatch
    and RAISES (the read layer is load-bearing — silently loading the first glob
    match at the wrong layer would mis-key the predictor). Returns None ONLY when
    NO t_cb file exists for this (behavior, source) at all (handled upstream by
    the --tcb-dir / --allow-missing-tcb gate).
    """
    import torch

    if tcb_dir is None or not tcb_dir.is_dir():
        return None
    exact = tcb_dir / f"t_cb_{behavior}_{source}_L{layer}.pt"
    if exact.is_file():
        payload = torch.load(exact, map_location="cpu", weights_only=False)
        return torch.as_tensor(payload["t_cb"]).flatten().float().numpy()
    other = sorted(tcb_dir.glob(f"t_cb_{behavior}_{source}_L*.pt"))
    if other:
        raise FileNotFoundError(
            f"t_cb for ({behavior}, {source}) exists but NOT at requested layer L{layer} "
            f"(found {[p.name for p in other]}); refusing to load a wrong-layer key."
        )
    return None


def _dv_target_value(payload: dict, ctx: str, a7_rank1: bool, u1: np.ndarray | None) -> float:
    """The scored DV for a held-out context: scalar g_real (rank-1) OR the
    dominant-component projection ⟨Δv(C'), u₁⟩ (low-rank fallback)."""
    if a7_rank1:
        g = payload["g_real"].get(ctx)
        return float(g) if g is not None and g == g else float("nan")
    import torch

    dv = torch.as_tensor(payload["per_context"][ctx]["Delta_v"]).flatten().float().numpy()
    return float(dv @ u1)


def score_bank(
    *,
    payload: dict,
    c_bank: dict[str, np.ndarray],
    t_cb: np.ndarray | None,
    a7_rank1: bool,
    n_boot: int,
    seed: int,
    allow_partial_panel: bool = False,
    require_tcb: bool = True,
    n_folds: int = 0,
) -> dict:
    """Full key × metric × ψ leaderboard for one source/seed bank.

    Panel-coverage contract (BLOCKER c-bank-panel-coverage-silent-drop): every
    held-out context in the Δv bank MUST have a c_C' entry in ``c_bank``; a
    missing one RAISES unless ``allow_partial_panel`` is set (in which case the
    descope is recorded in the returned metadata). t_cb contract (BLOCKER
    tcb-keys-silently-omitted): when ``require_tcb`` is True, ``t_cb`` MUST be
    present, so the t-based keys (k_tCB, k_cC_plus_delta) are populated.
    """
    source = payload["source"]
    per_context = payload["per_context"]
    if source not in c_bank:
        raise AssertionError(
            f"source {source!r} has no c_C entry in the context bank — cannot build any key."
        )

    # Panel coverage: compute MISSING held-out contexts BEFORE filtering so a
    # silent panel shrink cannot pass as long as 3 targets remain.
    held_out_contexts = [c for c in per_context if c != source]
    missing_cC = [c for c in held_out_contexts if c not in c_bank]
    if missing_cC and not allow_partial_panel:
        raise AssertionError(
            f"source {source!r}: {len(missing_cC)} held-out context(s) have NO c_C' entry in "
            f"the context bank: {sorted(missing_cC)}. The panel-coverage contract requires every "
            "held-out target to have a c_C' (predictor query). Re-run the c_C' bank builder / "
            "panel top-up, or pass --allow-partial-panel to deliberately descope (recorded in "
            "the leaderboard metadata)."
        )
    targets = [c for c in held_out_contexts if c in c_bank]

    # t_cb contract: require the data-side key unless explicitly waived.
    if require_tcb and t_cb is None:
        raise AssertionError(
            f"source {source!r}: t_cb is None but --tcb-dir is required (the t-based keys "
            "k_tCB / k_cC_plus_delta cannot be built). Pass --tcb-dir <dir>, or "
            "--allow-missing-tcb to deliberately score c_C-only (smoke/debug)."
        )

    if len(targets) < 3:
        raise AssertionError(
            f"only {len(targets)} held-out targets with a c_C entry for source {source!r}; "
            "need >= 3 for a leave-one-context-out leaderboard."
        )

    # low-rank fallback dominant direction u₁ (only used when a7_rank1 is False).
    u1 = None
    if not a7_rank1:
        import torch

        dvs = np.stack(
            [torch.as_tensor(per_context[c]["Delta_v"]).flatten().float().numpy() for c in targets],
            axis=1,
        )  # (H, n)
        u, _s, _vt = np.linalg.svd(dvs, full_matrices=False)
        u1 = u[:, 0]

    c_source = c_bank[source]
    w_hat = np.asarray(payload["w_hat"]).astype(float).flatten() if "w_hat" in payload else None
    import torch

    if w_hat is None:
        w_hat = torch.as_tensor(per_context[source]["Delta_v"]).flatten().float().numpy()
    # δ_{C,B} = t_{C,B} - v_base(C); v_base(C) read from the source's own context.
    v_base_source = torch.as_tensor(per_context[source]["v_base"]).flatten().float().numpy()
    delta_cb = (t_cb - v_base_source) if t_cb is not None else None

    # Pre-stack c_C' query matrix + the scored DV y over the targets.
    c_query = {c: c_bank[c] for c in targets}
    y = np.array([_dv_target_value(payload, c, a7_rank1, u1) for c in targets])

    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    # BLOCKER psi-ridge-heldout-leak: ψ is fit PER outer-LOO fold (each held-out
    # target excluded from its own ψ's training pairs), so the ψ_ridge t-based
    # key is target-DEPENDENT. psi_I / k_cC keys stay target-independent (single
    # k). The ψ map is NEVER fit once on the full target panel for a key that
    # scores those same targets (the round-3 leak). _resolve_cell_keys returns
    # (k, k_per_target) or None to skip the cell.
    psi_per_fold = _psi_per_fold(per_context, c_bank, targets) if t_cb is not None else None
    for psi in PSI_FORMS:
        for key_form in KEY_FORMS:
            keys = _resolve_cell_keys(
                psi=psi,
                key_form=key_form,
                psi_per_fold=psi_per_fold,
                c_source=c_source,
                t_cb=t_cb,
                delta_cb=delta_cb,
                targets=targets,
            )
            if keys is None:
                continue
            k, k_per_target = keys
            for metric in METRICS:
                row = _score_one_cell(
                    key_form=key_form,
                    metric=metric,
                    psi=psi,
                    k=k,
                    k_per_target=k_per_target,
                    c_source=c_source,
                    c_query=c_query,
                    targets=targets,
                    y=y,
                    a7_rank1=a7_rank1,
                    n_boot=n_boot,
                    n_folds=n_folds,
                    seed=seed,
                    rng=rng,
                )
                if row is not None:
                    rows.append(row)

    # Key-form presence contract (BLOCKER tcb-keys-silently-omitted): with t_cb
    # present, EVERY key form (k_cC, k_tCB, k_cC_plus_delta) must have produced
    # ≥1 leaderboard row; a silently-absent t-based key would corrupt the
    # primary deliverable. Only waived when t_cb was explicitly allowed missing.
    scored_keys = {r["key"] for r in rows}
    if t_cb is not None:
        missing_keys = [kf for kf in KEY_FORMS if kf not in scored_keys]
        if missing_keys:
            raise AssertionError(
                f"source {source!r}: t_cb present but leaderboard is missing key form(s) "
                f"{missing_keys} (scored: {sorted(scored_keys)}). The primary deliverable "
                "requires k_cC, k_tCB, k_cC_plus_delta all populated."
            )

    baseline_cos, shuf_key, shuf_query = _nulls_and_baseline(
        c_source=c_source,
        c_query=c_query,
        c_bank=c_bank,
        targets=targets,
        source=source,
        y=y,
        n_boot=n_boot,
        rng=rng,
    )

    return {
        "source": source,
        "seed": payload.get("seed"),
        "a7_rank1": a7_rank1,
        "scored_dv": "g_real_scalar" if a7_rank1 else "lowrank_dominant_component",
        "n_targets": len(targets),
        "leaderboard": rows,
        "baseline_cos": baseline_cos,
        "null_shuffled_key": _null_summary(shuf_key),
        "null_shuffled_query": _null_summary(shuf_query),
        "has_tcb": t_cb is not None,
        "panel_coverage": {
            "n_held_out": len(held_out_contexts),
            "n_scored_targets": len(targets),
            "missing_cC": sorted(missing_cC),
            "partial_panel_descope": bool(missing_cC) and allow_partial_panel,
        },
    }


def _null_summary(arr: np.ndarray) -> dict:
    """Mean + [p5, p95] of a null Spearman distribution (NaN-safe)."""
    if arr.size == 0:
        return {"mean": float("nan"), "p5": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def _nulls_and_baseline(
    *,
    c_source: np.ndarray,
    c_query: dict[str, np.ndarray],
    c_bank: dict[str, np.ndarray],
    targets: list[str],
    source: str,
    y: np.ndarray,
    n_boot: int,
    rng,
) -> tuple[dict | None, np.ndarray, np.ndarray]:
    """Cosine baseline + the shuffled-KEY / shuffled-QUERY nulls.

    Both nulls permute a key/query VECTOR (methodology-critic concern #2),
    NOT a matrix-axis relabel: shuffled-key scores c_C against a RANDOM other
    context's c as the key; shuffled-query permutes which c_C' each g_real is
    scored against. Returns (baseline_cos dict | None, shuf_key arr, shuf_query arr).
    """

    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

    cos_pred = np.array([_cos(c_source, c_query[c]) for c in targets])
    finite = np.isfinite(cos_pred) & np.isfinite(y)
    baseline_cos = None
    if finite.sum() >= 3:
        baseline_cos = {
            "spearman": spearman(cos_pred[finite], y[finite]),
            "pearson": pearson(cos_pred[finite], y[finite]),
            "n_scored": int(finite.sum()),
        }

    null_seeds = 20 if _is_smoke(n_boot) else 200

    def _spearman_for_key(k_vec):
        preds = np.array([_g_pred(k_vec, None, c_query[c], c_source) for c in targets])
        f = np.isfinite(preds) & np.isfinite(y)
        return spearman(preds[f], y[f]) if f.sum() >= 3 else float("nan")

    other_ctx = [c for c in c_bank if c not in (source, *targets)] or targets
    query_vals = [c_query[c] for c in targets]
    shuf_key, shuf_query = [], []
    for _ in range(null_seeds):
        rk = c_bank[other_ctx[rng.integers(0, len(other_ctx))]]
        shuf_key.append(_spearman_for_key(rk))
        perm = rng.permutation(len(targets))
        preds = np.array(
            [_g_pred(c_source, None, query_vals[perm[i]], c_source) for i in range(len(targets))]
        )
        f = np.isfinite(preds) & np.isfinite(y)
        shuf_query.append(spearman(preds[f], y[f]) if f.sum() >= 3 else float("nan"))
    return (
        baseline_cos,
        np.array([v for v in shuf_key if v == v]),
        np.array([v for v in shuf_query if v == v]),
    )


_SMOKE_BOOT_MAX = 100


def _is_smoke(n_boot: int) -> bool:
    return n_boot <= _SMOKE_BOOT_MAX


def _noise_floor(banks: list[dict]) -> dict:
    """Test-retest noise floor: cross-seed Spearman of g_real on shared contexts.

    Bounds the achievable ρ (a predictor can't beat the measurement
    reliability). Computed only when ≥2 seeds of the SAME source are present.
    """
    by_source: dict[str, list[dict]] = {}
    for b in banks:
        by_source.setdefault(b["source"], []).append(b)
    floors = {}
    for source, group in by_source.items():
        if len(group) < 2:
            continue
        # all-pairs cross-seed Spearman of g_real over shared contexts.
        rhos = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                gi, gj = group[i]["g_real"], group[j]["g_real"]
                shared = [c for c in gi if c in gj and c != source]
                a = np.array([gi[c] for c in shared], dtype=float)
                b2 = np.array([gj[c] for c in shared], dtype=float)
                f = np.isfinite(a) & np.isfinite(b2)
                if f.sum() >= 3:
                    rhos.append(spearman(a[f], b2[f]))
        rhos = [r for r in rhos if r == r]
        if rhos:
            floors[source] = {
                "test_retest_spearman_mean": float(np.mean(rhos)),
                "n_pairs": len(rhos),
            }
    return floors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--behavior", required=True, choices=("marker", "sycophancy"))
    ap.add_argument("--dv-dir", default=None, help="default analysis_tensors/dv/<behavior>")
    ap.add_argument("--c-bank", required=True, help=".pt context-vector bank {ctx: c_C}")
    ap.add_argument(
        "--tcb-dir", default=None, help="t_cb extractor output dir (enables k_tCB/k_cC+δ)"
    )
    ap.add_argument("--a7", default=None, help="a7_precondition.json (gates the scored DV)")
    ap.add_argument(
        "--layer", type=int, default=None, help="c-bank slice layer; default per behavior"
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-folds", type=int, default=0, help="λ-GCV folds (0 = leave-one-out)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--noise-floor-out",
        default=None,
        help="standalone noise_floor.json (default eval_results/issue_683/<behavior>/)",
    )
    ap.add_argument(
        "--allow-missing-tcb",
        action="store_true",
        help="DEBUG/smoke ONLY: score c_C-only when --tcb-dir is absent (production REQUIRES it)",
    )
    ap.add_argument(
        "--allow-partial-panel",
        action="store_true",
        help="deliberately descope held-out targets missing a c_C' entry (recorded in metadata)",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    # t_cb contract (BLOCKER tcb-keys-silently-omitted): --tcb-dir REQUIRED in
    # production. The only waivers are --smoke or the explicit --allow-missing-tcb.
    require_tcb = not (args.smoke or args.allow_missing_tcb)
    if require_tcb and not args.tcb_dir:
        raise SystemExit(
            "--tcb-dir is REQUIRED in production (the t-based keys k_tCB / k_cC_plus_delta "
            "cannot be built without it). Pass --tcb-dir <dir>, or --allow-missing-tcb / --smoke "
            "to deliberately score c_C-only."
        )

    layer = args.layer if args.layer is not None else DEFAULT_LAYER[args.behavior]
    dv_dir = Path(
        args.dv_dir or (PROJECT_ROOT / "eval_results/issue_683/analysis_tensors/dv" / args.behavior)
    )
    out_path = Path(
        args.out
        or (
            PROJECT_ROOT
            / "eval_results/issue_683"
            / args.behavior
            / "key_ablation_leaderboard.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    noise_floor_path = Path(
        args.noise_floor_out
        or (PROJECT_ROOT / "eval_results/issue_683" / args.behavior / "noise_floor.json")
    )
    noise_floor_path.parent.mkdir(parents=True, exist_ok=True)

    a7_rank1 = True
    a7_verdict = "assumed_rank1 (no a7 file)"
    if args.a7 and Path(args.a7).is_file():
        a7 = json.loads(Path(args.a7).read_text())
        a7_rank1 = bool(a7.get("behavior_rank1_holds", True))
        a7_verdict = a7.get("verdict", a7_verdict)

    c_bank = _load_c_bank(Path(args.c_bank), layer)
    tcb_dir = Path(args.tcb_dir) if args.tcb_dir else None
    banks = _load_dv_banks(dv_dir)

    logger.info(
        "[phase=score_start] behavior=%s a7_rank1=%s (%s) n_banks=%d c_bank=%d ctx tcb_dir=%s "
        "require_tcb=%s allow_partial_panel=%s",
        args.behavior,
        a7_rank1,
        a7_verdict,
        len(banks),
        len(c_bank),
        tcb_dir,
        require_tcb,
        args.allow_partial_panel,
    )

    bank_results = []
    for payload in banks:
        t_cb = _load_tcb(tcb_dir, args.behavior, payload["source"], layer)
        res = score_bank(
            payload=payload,
            c_bank=c_bank,
            t_cb=t_cb,
            a7_rank1=a7_rank1,
            n_boot=args.n_boot,
            seed=args.seed,
            allow_partial_panel=args.allow_partial_panel,
            require_tcb=require_tcb,
            n_folds=args.n_folds,
        )
        bank_results.append(res)
        best = max(
            (r for r in res["leaderboard"] if r["spearman"] == r["spearman"]),
            key=lambda r: r["spearman"],
            default=None,
        )
        logger.info(
            "[phase=score_bank] source=%s seed=%s n_targets=%d best=%s null_key_mean=%.3f",
            res["source"],
            res["seed"],
            res["n_targets"],
            (
                f"{best['key']}/{best['metric']}/{best['psi']} ρ={best['spearman']:.3f}"
                if best
                else "—"
            ),
            res["null_shuffled_key"]["mean"],
        )

    noise_floor = _noise_floor(banks)
    payload_out = {
        "behavior": args.behavior,
        "layer": layer,
        "a7_rank1": a7_rank1,
        "a7_verdict": a7_verdict,
        "key_forms": list(KEY_FORMS),
        "metrics": list(METRICS),
        "psi_forms": list(PSI_FORMS),
        "n_bootstrap": args.n_boot,
        "per_bank": bank_results,
        "noise_floor": noise_floor,
        "reproducibility": repro_metadata({"behavior": args.behavior, "layer": layer}),
    }
    out_path.write_text(json.dumps(payload_out, indent=2))

    # Standalone noise_floor.json (primary_deliverable §6.5 glob
    # eval_results/issue_683/*/noise_floor.json — BLOCKER phase-d-deliverables-missing).
    noise_floor_payload = {
        "behavior": args.behavior,
        "layer": layer,
        "construct": "test-retest g_real reliability across seeds (ceiling on achievable ρ)",
        "per_source": noise_floor,
        "n_banks": len(banks),
        "reproducibility": repro_metadata({"behavior": args.behavior, "layer": layer}),
    }
    noise_floor_path.write_text(json.dumps(noise_floor_payload, indent=2))
    logger.info(
        "[phase=score_done] behavior=%s -> %s ; noise_floor -> %s",
        args.behavior,
        out_path,
        noise_floor_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
