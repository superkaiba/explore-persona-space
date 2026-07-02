#!/usr/bin/env python3
"""Issue #812 — pooling-operator vs unpooled-ceiling regression sweep (CPU, vectorized).

The PRIMARY deliverable. Per (behavior x layer x operator), fits a LOCO closed-form
ridge predictor of the graded 0-100 E0(C, B) from that operator's answer-summary
input, reports held-out Spearman rho + Delta-rho vs the reference ``mean`` and vs
the granularity-isolating ``mean_pca`` control, against a selection-symmetric
shuffle null + the #742 reliability ceiling sqrt(r_yy) + a learning curve.

Operators (plan §4.3 / §5):
  - Raw single-vector pools fed as raw (H,): ``mean`` (reference), ``max``,
    ``attn-fixed`` (unlearned control).
  - ``attn-learned``: a per-(behavior, layer) FIT softmax query over the (2K+2)
    aligned positions -> (H,) pool -> ridge. The ONLY gradient-descent arm; init at
    the mean-pool direction, L2-regularized, cross-fit inside LOCO, VECTORIZED across
    all (behavior x layer) cells (never a per-cell serial loop —
    ``.claude/rules/vectorize-many-cell-fits.md``).
  - Matched-PCA pooled baselines (Alt MF): ``mean_pca`` (PRIMARY control) /
    ``max_pca`` / ``attn-fixed_pca`` / ``attn-learned_pca`` — each single-vector pool
    put through the SAME train-fold per-vector PCA-to-d_in=10 the unpooled ceiling
    uses, isolating granularity from PCA denoising.
  - Unpooled ceiling: the (2K+2) aligned positions each per-position PCA-reduced
    (train-only) to d_in=10, concatenated to (2K+2)*10 features, then ridge.

Reliability ceiling sqrt(r_yy) replicated INLINE (``compute_bracket`` absent on
main). Estimator FORKS per behavior (Stat MF2): sycophancy = split-half over
ROLLOUTS; refusal / harmful_compliance / all 5 low-m = split-half over PROBES
(1 completion/probe) with over-judge-draws as a cross-check. Preflight: sqrt(r_yy)
is non-null for ALL requested behaviors (mechanizable assert).

Output ``eval_results/issue_812/pooling_fit_results.json`` + selection matrices +
reliability/learning-curve JSON + reconstruction R2 JSON + hero/exploratory figures
under ``figures/issue_812/``. CPU-only, 0 GPU-h.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from issue658_fit_predictors import (  # noqa: E402
    N_BOOTSTRAP,
    _cluster_bootstrap_rho,
    _rho,
)

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue812.fit")

# ── Constants (plan §11) ─────────────────────────────────────────────────────
D_IN_PCA = 10  # per-position / matched-pool PCA rank (Source: #742 Stage-1)
SHUFFLE_ITERS = 1000  # Source: #763 plan
LEARNING_CURVE_GRID = [10, 15, 20, 25, 30, 40, 50]  # Source: #742 Stage-2
ATTN_LR = 5e-2
ATTN_EPOCHS = 300
ATTN_L2_GRID = [1e-3, 1e-2, 1e-1, 1.0]  # nested-CV L2 on the learned query
ATTN_REDUCE_RANK = 32  # PCA-reduce H before the query fit (regularize + vectorize)
POOL_OPS = ["mean", "max", "attn_fixed", "attn_learned"]
PCA_OPS = ["mean_pca", "max_pca", "attn_fixed_pca", "attn_learned_pca"]

HIGH_M = ["sycophancy", "refusal", "harmful_compliance"]
LOW_M = ["deception", "fact_expression", "format_style", "self_report", "persona_drift"]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_SCRIPTS_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _atomic_write_json(path: Path, obj) -> None:
    """Atomic JSON write (``.tmp`` + ``os.replace``) — checkpoint-per-phase safe.

    Used to checkpoint ``pooling_fit_results.json`` after EACH behavior lands
    (BLOCKER 3): a downstream crash on a later behavior can never corrupt or lose
    the already-completed behaviors' fits.
    """
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ── Per-position / per-vector PCA-reduce (train-only, LOCO-safe) ──────────────


def _loco_pca_features(X: np.ndarray, d_in: int) -> np.ndarray:
    """LOCO train-only PCA-reduce (N, D) -> (N, d_in) held-out projections.

    For each held-out row i, fit the PCA basis on the OTHER n-1 rows (train-only,
    no leakage), project row i onto the top-d_in components. Returns (N, d_in). The
    projection is centered on the TRAIN mean of the fold (``robust_pca_basis``
    subtracts its own mean; we re-subtract the train mean for the held-out row).
    """
    n, _ = X.shape
    d = min(d_in, n - 2)  # cannot exceed train rank
    out = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        mu, comps, _ = robust_pca_basis(X[tr].astype(np.float64), d)  # comps (k', D)
        out[i, : comps.shape[0]] = (X[i].astype(np.float64) - mu) @ comps.T
    return out


def _ridge_rho_from_features(X: np.ndarray, y: np.ndarray) -> tuple[float | None, np.ndarray]:
    """Held-out LOCO ridge rho of y from features X (train-only std + nested-CV lambda).

    Uses the #722/#658 exact estimator via ``ridge_predict_loco_centered`` (Yv is
    (N, 1) so the closed-form scalar prediction is returned). Returns (rho, preds).
    """
    Yv = y.reshape(-1, 1).astype(np.float64)
    preds = ridge_predict_loco_centered(X.astype(np.float64), Yv).ravel()
    return _rho(preds, y), preds


# ── attn-learned: vectorized per-cell softmax-query fit (the only GD arm) ─────


ATTN_INNER_CV_K = 3  # nested-CV inner folds for the attn-learned L2 selection (§8)


def _fit_attn_learned_pool(
    aligned: np.ndarray, y: np.ndarray, *, seed: int, device: str
) -> np.ndarray:
    """Fit a softmax query over aligned positions, LOCO cross-fit, return pooled (N, H).

    aligned: (N, P, H) the (2K+2) aligned-position activations (NaN-filled slots ->
    zeroed + mask). Fit query q (init = mean-pool direction) so
    ``pool_i = sum_p softmax(a_i @ q)_p * a_i[p]``; the pooled held-out vector then
    goes to the shared ridge downstream so the reported rho uses the same estimator
    as every other operator. The query is fit on the LOCO TRAIN fold only (per
    held-out i) — no held-out leakage.

    CONCERN 5 fix (plan §8): the L2 is chosen by NESTED-CV inner-held-out loss (K=3
    inner CV folds of each outer train set), NOT by the train-fold loss; AND the
    H->R PCA reduction basis is fit TRAIN-ONLY PER OUTER FOLD (each fold excludes its
    held-out context from the SVD), NOT once on all rows. Both close the leakage the
    round-1 code carried.

    VECTORIZED (``.claude/rules/vectorize-many-cell-fits.md``): the query fit trains
    all N outer folds jointly as one batched parameter tensor under a fold-exclusion
    mask, and the nested-CV L2 selection trains all (outer x inner) folds jointly per
    L2 — the epoch loop runs ``(#inner-CV passes + 1 refit) x len(L2)`` times TOTAL,
    NOT ``n x len(L2)`` times. Per-fold PCA is N small SVDs (n~50 of a few-hundred x H
    matrix), computed once up front. The learned attention is over POSITIONS, computed
    from the reduced scores, then applied to the FULL-H activations so no answer
    information is lost in the returned pool.
    """
    torch.manual_seed(seed)
    n, _p, h = aligned.shape
    dev = torch.device(device)
    a = torch.from_numpy(np.nan_to_num(aligned).astype(np.float32)).to(dev)  # (N,P,H)
    pmask = torch.from_numpy((~np.isnan(aligned)).any(axis=2)).to(dev)  # (N,P) bool
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)  # (N,)
    r = min(ATTN_REDUCE_RANK, max(1, int(pmask.sum().item()) - 1), h)

    # ── Per-outer-fold TRAIN-ONLY PCA basis (CONCERN 5: no all-row leakage). ──────
    # Fold f (held-out context f) fits its H->R basis on the valid positions of the
    # OTHER n-1 contexts only. ar_f[f] = (N, P, R) reduced under fold f's basis.
    a_np = np.nan_to_num(aligned).astype(np.float64)  # (N,P,H)
    pmask_np = (~np.isnan(aligned)).any(axis=2)  # (N,P)
    ar_f = np.zeros((n, n, _p, r), dtype=np.float32)  # (F, N, P, R)
    for f in range(n):
        tr_ctx = [j for j in range(n) if j != f]
        flat = a_np[tr_ctx][pmask_np[tr_ctx]]  # (M_f, H) train-only valid positions
        mu, comps, _ = robust_pca_basis(flat, r)  # comps (r',H)
        rk = comps.shape[0]
        proj = np.einsum("nph,rh->npr", a_np - mu, comps)  # (N,P,r')
        ar_f[f, :, :, :rk] = proj.astype(np.float32)
    ar_ft = torch.from_numpy(ar_f).to(dev)  # (F,N,P,R)
    rdim = r

    eye = torch.eye(n, device=dev, dtype=torch.bool)
    outer_train = (~eye).float()  # (F, N) 1 where row j is in fold f's train set
    pmask_f = pmask.float().unsqueeze(0)  # (1,N,P) broadcast over folds

    # reduced mean-pool init per fold (fold f's own basis)
    with torch.no_grad():
        valid = pmask.float().view(1, n, _p, 1)  # (1,N,P,1)
        per_ctx_mean = (ar_ft * valid).sum(2) / valid.sum(2).clamp_min(1.0)  # (F,N,R)
        q_init_f = per_ctx_mean.mean(1)  # (F,R) mean over contexts, per fold
        q_init_f = q_init_f / (q_init_f.norm(dim=1, keepdim=True) + 1e-9)

    def _train_folds(
        ar_batch: torch.Tensor,
        q0: torch.Tensor,
        train_m: torch.Tensor,
        l2: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train B fold-fits jointly at one L2 on ``train_m`` rows; return (Q,W,B_bias,pred).

        ar_batch (Bf, N, P, R) each fit's own reduced activations; q0 (Bf, R) init;
        train_m (Bf, N) 1 where row is in that fit's train set. Returns fitted params
        + final pred (Bf, N) so the caller can score inner-held-out rows.
        """
        bf = ar_batch.shape[0]
        Q = q0.clone().requires_grad_(True)  # (Bf,R)
        W = torch.zeros(bf, rdim, device=dev, requires_grad=True)  # (Bf,R)
        Bb = torch.zeros(bf, device=dev, requires_grad=True)  # (Bf,)
        opt = torch.optim.Adam([Q, W, Bb], lr=ATTN_LR)
        n_tr = train_m.sum(1).clamp_min(1.0)  # (Bf,)
        pm = pmask_f  # (1,N,P)
        pred = None
        for _ in range(ATTN_EPOCHS):
            opt.zero_grad()
            scores = torch.einsum("fnpr,fr->fnp", ar_batch, Q)  # (Bf,N,P)
            scores = scores.masked_fill(pm < 0.5, -1e9)
            attn = torch.softmax(scores, dim=2)  # (Bf,N,P)
            aw = torch.einsum("fnpr,fr->fnp", ar_batch, W)  # (Bf,N,P)
            pred = (attn * aw).sum(2) + Bb.unsqueeze(1)  # (Bf,N)
            resid2 = (pred - yt.unsqueeze(0)) ** 2  # (Bf,N)
            mse = (resid2 * train_m).sum(1) / n_tr  # (Bf,) train-only MSE
            reg = l2 * (Q.pow(2).sum(1) + W.pow(2).sum(1))  # (Bf,)
            (mse + reg).sum().backward()  # per-fit grads independent
            opt.step()
        return Q.detach(), W.detach(), Bb.detach(), pred.detach()

    # ── Nested-CV L2 selection: inner K-fold split of each outer train set. ──────
    # inner_train[f, s] (N,): 1 where row is in outer-fold-f's inner-split-s TRAIN set
    # (i.e. in the outer train set AND NOT the inner-val block s). inner_val is the
    # complement within the outer train set. Score each L2 on the inner-val rows.
    rng_inner = np.random.default_rng(seed + 12345)
    inner_val_mse = np.full((n, len(ATTN_L2_GRID)), np.inf, dtype=np.float64)
    for f in range(n):
        tr_ctx = np.array([j for j in range(n) if j != f])
        rng_inner.shuffle(tr_ctx)
        blocks = np.array_split(tr_ctx, ATTN_INNER_CV_K)
        # build (K, N) inner-train masks + (K, N) inner-val masks for this outer fold
        itr = np.zeros((ATTN_INNER_CV_K, n), dtype=np.float32)
        ival = np.zeros((ATTN_INNER_CV_K, n), dtype=np.float32)
        for s, blk in enumerate(blocks):
            if len(blk) == 0:
                continue
            val = set(blk.tolist())
            for j in tr_ctx:
                if j in val:
                    ival[s, j] = 1.0
                else:
                    itr[s, j] = 1.0
        itr_t = torch.from_numpy(itr).to(dev)  # (K,N)
        ival_t = torch.from_numpy(ival).to(dev)  # (K,N)
        ar_batch = ar_ft[f].unsqueeze(0).repeat(ATTN_INNER_CV_K, 1, 1, 1)  # (K,N,P,R)
        q0 = q_init_f[f].unsqueeze(0).repeat(ATTN_INNER_CV_K, 1)  # (K,R)
        for li, l2 in enumerate(ATTN_L2_GRID):
            _q, _w, _b, pred = _train_folds(ar_batch, q0, itr_t, l2)  # pred (K,N)
            resid2 = (pred - yt.unsqueeze(0)) ** 2  # (K,N)
            denom = ival_t.sum(1).clamp_min(1.0)
            val_mse = (resid2 * ival_t).sum(1) / denom  # (K,)
            inner_val_mse[f, li] = float(val_mse.mean().item())  # mean over K inner folds

    best_l2_idx = inner_val_mse.argmin(axis=1)  # (F,) per-outer-fold best L2 index

    # ── Refit each outer fold at its nested-CV-chosen L2 on the FULL train set. ──
    # Group folds by chosen L2 so the refit stays vectorized (one batched fit per L2).
    best_Q = torch.zeros(n, rdim, device=dev)
    for li, _l2 in enumerate(ATTN_L2_GRID):
        sel = np.where(best_l2_idx == li)[0]
        if sel.size == 0:
            continue
        sel_t = torch.from_numpy(sel).to(dev)
        ar_sel = ar_ft[sel_t]  # (B,N,P,R)
        q0 = q_init_f[sel_t]  # (B,R)
        train_m = outer_train[sel_t]  # (B,N)
        Q, _w, _b, _pred = _train_folds(ar_sel, q0, train_m, ATTN_L2_GRID[li])
        best_Q[sel_t] = Q

    # held-out pooled: fold i's reduced query (fold i's basis) -> position attention
    # -> pool FULL-H a[i]. ar_ft[i, i] is context i reduced under fold i's train basis.
    with torch.no_grad():
        ar_ii = ar_ft[torch.arange(n, device=dev), torch.arange(n, device=dev)]  # (N,P,R)
        scores_i = torch.einsum("ipr,ir->ip", ar_ii, best_Q)  # (N,P) reduced scores
        scores_i = scores_i.masked_fill(pmask.float() < 0.5, -1e9)
        attn_i = torch.softmax(scores_i, dim=1)  # (N,P)
        pooled_out = torch.einsum("ip,iph->ih", attn_i, a)  # (N,H) full-H pool
    return pooled_out.cpu().numpy()


# ── graded-DV reliability ceiling sqrt(r_yy) (inline #742 recipe, estimator fork) ─


def _spearman_brown(r_half: float, k: int = 2) -> float:
    if r_half <= -0.999:
        return math.nan
    return (k * r_half) / (1 + (k - 1) * r_half)


def _split_half_ryy(
    per_ctx_units: dict[str, list[list[float]]],
    *,
    seed: int,
    n_splits: int = 50,
    ctx_subset: list[str] | None = None,
    return_r_half: bool = False,
) -> float | None | tuple[float | None, float | None]:
    """sqrt(r_yy): split-half over the SPLITTABLE UNIT within each context, corrected.

    per_ctx_units: {ctx: [ [unit_values...], ... ]} where each context has a LIST of
    splittable units (probes, or completions per probe), each a list of numeric
    scores. We split the units of each context into two halves, form two per-context
    means, correlate across contexts (Spearman), average over ``n_splits`` random
    halvings, Spearman-Brown-correct, and return sqrt(reliability). None if no
    context has >=2 splittable units (undefined split). ``ctx_subset`` (for the
    bootstrap-CI resamples) restricts the correlation to a given multiset of contexts.

    When ``return_r_half`` is True, returns ``(sqrt_r_yy_or_None, r_half_or_None)`` —
    the second element is the mean uncorrected split-half Spearman across contexts,
    the honest diagnostic for WHY a null ceiling arose (a degenerate target with ~no
    between-context signal drives ``r_half`` <= 0 → Spearman-Brown r_yy <= 0 → null).
    """
    avail = [c for c, u in per_ctx_units.items() if len(u) >= 2]
    ctx_ids = ctx_subset if ctx_subset is not None else avail
    ctx_ids = [c for c in ctx_ids if c in per_ctx_units and len(per_ctx_units[c]) >= 2]
    if len(ctx_ids) < 4:
        return (None, None) if return_r_half else None
    rng = random.Random(seed)
    rs: list[float] = []
    for _ in range(n_splits):
        a_means, b_means = [], []
        for c in ctx_ids:
            units = per_ctx_units[c]
            order = list(range(len(units)))
            rng.shuffle(order)
            half = len(order) // 2
            a_idx, b_idx = order[:half], order[half : 2 * half]
            a_vals = [v for i in a_idx for v in units[i] if not math.isnan(v)]
            b_vals = [v for i in b_idx for v in units[i] if not math.isnan(v)]
            if not a_vals or not b_vals:
                continue
            a_means.append(sum(a_vals) / len(a_vals))
            b_means.append(sum(b_vals) / len(b_vals))
        if len(a_means) < 4:
            continue
        r = _rho(np.array(a_means), np.array(b_means))
        if r is not None:
            rs.append(r)
    if not rs:
        return (None, None) if return_r_half else None
    r_half = float(np.mean(rs))
    r_yy = _spearman_brown(r_half)
    if r_yy is None or math.isnan(r_yy) or r_yy <= 0:
        return (None, r_half) if return_r_half else None
    ryy = math.sqrt(min(r_yy, 1.0))
    return (ryy, r_half) if return_r_half else ryy


def _bootstrap_ryy_ci(
    per_ctx_units: dict[str, list[list[float]]], *, seed: int, n_boot: int
) -> list[float] | None:
    """Cluster-bootstrap 95% CI on sqrt(r_yy) by resampling CONTEXTS with replacement.

    Uses the SAME cluster-over-contexts resampling + the SAME B as the primary Δρ CI
    (plan §6). Returns [lo, hi] (2.5 / 97.5 percentiles of the bootstrap distribution)
    or None if too few contexts carry a splittable unit.
    """
    avail = [c for c, u in per_ctx_units.items() if len(u) >= 2]
    if len(avail) < 4:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for b in range(n_boot):
        draw = [avail[rng.randrange(len(avail))] for _ in range(len(avail))]
        # de-dup for the split (a resampled context contributes its units once; the
        # cluster resampling reweights which contexts enter the cross-context corr)
        r = _split_half_ryy(per_ctx_units, seed=seed + 1000 + b, n_splits=20, ctx_subset=draw)
        if r is not None:
            stats.append(r)
    if len(stats) < max(10, n_boot // 10):
        return None
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats)) - 1]
    return [float(lo), float(hi)]


def _binomial_variance_term(graded_cell: dict) -> float | None:
    """Within-context binomial-variance floor on the per-context graded MEAN (per #742).

    For each context, the graded mean is an average of m unit scores (probes /
    completions) each in [0, 100]. The sampling variance of that mean, treating each
    unit as a draw with per-context effective m = ``n_judged``, is
    ``var_within(ctx) = s^2(ctx) / m(ctx)`` (s^2 = the within-context unit-score
    variance). We report the MEAN over contexts of that within-context sampling
    variance (on the 0-100 scale) — the irreducible measurement-noise floor the
    Spearman-Brown reliability is corrected toward. None if no context has m>=2.
    """
    terms: list[float] = []
    for _ctx, cell in graded_cell.items():
        unit_means = [
            p["probe_mean"] for p in cell.get("probe_scores", []) if p.get("probe_mean") is not None
        ]
        m = len(unit_means)
        if m < 2:
            continue
        var = float(np.var(np.array(unit_means, dtype=np.float64), ddof=1))
        terms.append(var / m)
    if not terms:
        return None
    return float(np.mean(terms))


def _units_for_behavior(graded_cell: dict, behavior: str) -> tuple[dict, dict, str]:
    """Build the (primary_units, over_judge_draws_units, estimator_label) for MF2.

    - sycophancy (10 completions/probe): PRIMARY split is OVER-ROLLOUTS — each
      context's splittable units are its per-completion means (``completion_scores``,
      flattened across probes). Falls back to per-probe means only if no completion
      arrays are persisted.
    - refusal / harmful_compliance / all low-m (1 completion/probe): PRIMARY is
      OVER-PROBES — units are per-probe means.
    Both forks additionally build the over-judge-draws units (transpose the N draws).
    """
    over_rollouts = behavior == "sycophancy"
    primary_units: dict[str, list[list[float]]] = {}
    draws_by_ctx: dict[str, list[list[float]]] = {}
    for ctx, cell in graded_cell.items():
        ps = cell.get("probe_scores", [])
        if over_rollouts:
            # over-rollouts: each completion mean is a splittable unit (across probes)
            comp_means = [
                m
                for p in ps
                for m in (p.get("completion_scores") or [])
                if m is not None and not math.isnan(m)
            ]
            if comp_means:
                primary_units[ctx] = [[m] for m in comp_means]
            else:  # no completion arrays persisted — degrade to per-probe means
                primary_units[ctx] = [
                    [p["probe_mean"]] for p in ps if p.get("probe_mean") is not None
                ]
        else:
            primary_units[ctx] = [[p["probe_mean"]] for p in ps if p.get("probe_mean") is not None]
        # over-judge-draws: transpose the N per-completion draw arrays
        per_draw: dict[int, list[float]] = {}
        for p in ps:
            for comp in p.get("completions", []):
                for d, s in enumerate(comp.get("draw_scores", [])):
                    if s is not None:
                        per_draw.setdefault(d, []).append(float(s))
            # backward-compat: older probe-level draw_scores (pre-BLOCKER-1 shape)
            for d, s in enumerate(p.get("draw_scores", [])):
                if s is not None:
                    per_draw.setdefault(d, []).append(float(s))
        draws_by_ctx[ctx] = [v for v in per_draw.values() if v]
    label = "over_rollouts" if over_rollouts else "over_probes"
    return primary_units, draws_by_ctx, label


def _reliability_for_behavior(graded_cell: dict, behavior: str, *, seed: int, n_boot: int) -> dict:
    """sqrt(r_yy) per behavior: MF2 estimator fork + binomial variance + bracket + CI.

    graded_cell: {ctx: {..., probe_scores: [{probe_idx, completion_scores:[...],
    completions:[{draw_scores:[N], completion_mean}], probe_mean}]}}.
    - sycophancy: over-ROLLOUTS primary (split the per-completion means, BLOCKER 1),
      over-judge-draws cross-check.
    - refusal / harmful_compliance / all low-m: over-PROBES primary; over-judge-draws
      cross-check.
    Returns the full #742 reliability object (CONCERN 6): sqrt_r_yy + bootstrap CI +
    binomial variance term + the reliability bracket [lo, hi].
    """
    primary_units, draws_by_ctx, label = _units_for_behavior(graded_cell, behavior)

    primary, split_half_r = _split_half_ryy(primary_units, seed=seed, return_r_half=True)
    over_draws = _split_half_ryy(draws_by_ctx, seed=seed + 1)
    ci = (
        _bootstrap_ryy_ci(primary_units, seed=seed + 2, n_boot=n_boot)
        if primary is not None
        else None
    )
    binom_var = _binomial_variance_term(graded_cell)

    # Between-context signal diagnostics (why a null ceiling arose): SD of the
    # per-context graded_mean on the 0-100 scale + the context count. A degenerate
    # target with ~no between-context variance (deception E0 on base Qwen, sd~3)
    # cannot support a positive split-half correlation → null sqrt(r_yy).
    ctx_means = [
        float(cell["graded_mean"])
        for cell in graded_cell.values()
        if cell.get("graded_mean") is not None
    ]
    n_ctx = len(ctx_means)
    between_ctx_sd = (
        float(np.std(np.array(ctx_means, dtype=np.float64), ddof=1)) if n_ctx >= 2 else None
    )

    # Reliability bracket [lo, hi]: the CI when available; else pair the point estimate
    # with the binomial-variance-attenuated ceiling as a coarse lower bracket. The
    # binomial term (measurement-noise floor) is combined per Spearman-Brown intuition:
    # more within-context sampling noise ⇒ a lower reliability floor, so the bracket
    # low end is at most the CI low end. We report both the CI and the binomial term so
    # the analyzer can form the honest bracket; the persisted bracket is the CI when
    # present, else [point, point] flagged coarse.
    if ci is not None:
        bracket_lo, bracket_hi = ci[0], ci[1]
        bracket_source = "bootstrap_ci95"
    elif primary is not None:
        bracket_lo = bracket_hi = primary
        bracket_source = "point_estimate_only (bootstrap CI unavailable)"
    else:
        bracket_lo = bracket_hi = None
        bracket_source = "undefined (sqrt_r_yy null)"

    return {
        "behavior": behavior,
        "sqrt_r_yy": primary,
        "estimator": label,
        "sqrt_r_yy_ci95": ci,
        "binomial_variance": binom_var,
        "bracket_lo": bracket_lo,
        "bracket_hi": bracket_hi,
        "bracket_source": bracket_source,
        "sqrt_r_yy_over_judge_draws_crosscheck": over_draws,
        "split_half_r": split_half_r,
        "between_ctx_sd": between_ctx_sd,
        "n_ctx": n_ctx,
    }


def _build_reliability_exclusions(
    reliability: dict[str, dict], behaviors: list[str]
) -> dict[str, dict]:
    """Preflight exclude-and-record: which behaviors have a null reliability ceiling.

    MF2 gates INTERPRETING Delta-rho against a non-null in-range ceiling FOR A GIVEN
    BEHAVIOR — it does NOT require every behavior to have one. A behavior with a null
    ``sqrt_r_yy`` (a degenerate target with ~no between-context signal — deception E0 on
    base Qwen) is EXCLUDED from any Delta-rho-vs-ceiling interpretation and RECORDED
    here with its measured diagnostics (``split_half_r`` / ``between_ctx_sd`` / ``n_ctx``),
    NOT raised on. The caller RAISES only when EVERY behavior is null (nothing to
    analyze). Returns ``{behavior: {sqrt_r_yy: None, reason, split_half_r, between_ctx_sd,
    n_ctx}}`` for the excluded behaviors (empty when all ceilings are non-null).
    """
    excluded: dict[str, dict] = {}
    for beh in behaviors:
        r = reliability[beh]
        if r["sqrt_r_yy"] is None:
            sd = r.get("between_ctx_sd")
            sd_str = f"{sd:.3g}" if sd is not None else "undefined"
            excluded[beh] = {
                "sqrt_r_yy": None,
                "reason": f"split-half r_yy <= 0 (between-context sd {sd_str})",
                "split_half_r": r.get("split_half_r"),
                "between_ctx_sd": sd,
                "n_ctx": r.get("n_ctx"),
            }
    return excluded


def _reliability_preflight(reliability: dict[str, dict], behaviors: list[str]) -> dict[str, dict]:
    """Build the exclusions map, log it, and RAISE iff ALL behaviors are null-ceiling.

    Wraps ``_build_reliability_exclusions`` + the terminal all-null hard failure into a
    single testable helper (so the raise branch has a deterministic test — see
    ``test_preflight_raises_when_ALL_behaviors_null``). MF2 gates INTERPRETING Delta-rho
    against a non-null in-range ceiling per behavior; a null-ceiling behavior is
    EXCLUDE-and-RECORDED (its per-cell fits are still honest data), and only the case
    where EVERY behavior is null (nothing left to analyze) is a hard RuntimeError.
    Returns the exclusions map (empty when every ceiling is non-null).
    """
    excluded = _build_reliability_exclusions(reliability, behaviors)
    for beh, rec in excluded.items():
        logger.warning(
            "reliability ceiling null for %s — excluded from Delta-rho interpretation "
            "(split_half_r=%s, between_ctx_sd=%s, n_ctx=%s); per-cell fits still computed",
            beh,
            rec["split_half_r"],
            rec["between_ctx_sd"],
            rec["n_ctx"],
        )
    if len(excluded) == len(behaviors):
        raise RuntimeError(
            "reliability preflight FAILED: sqrt(r_yy) is null for ALL behaviors "
            f"{list(excluded)} — no behavior has an in-range ceiling to "
            "interpret Delta-rho against, so there is nothing to analyze"
        )
    n_kept = len(behaviors) - len(excluded)
    if excluded:
        logger.warning(
            "Reliability preflight: %d/%d behaviors have a non-null ceiling; "
            "%d excluded from Delta-rho interpretation: %s",
            n_kept,
            len(behaviors),
            len(excluded),
            list(excluded),
        )
    else:
        logger.info(
            "Reliability preflight PASS: sqrt(r_yy) non-null for all %d behaviors",
            len(behaviors),
        )
    return excluded


# ── learning-curve contexts-needed extrapolation (per #742 Stage-2 / CONCERN 6) ─


def _fit_power_curve(ns: list[float], vals: list[float]) -> tuple[float, float, float] | None:
    """Fit metric(n) = a - b * n^{-c} by grid-search on c + least-squares on (a, b).

    Returns (a, b, c) or None if underdetermined (<3 finite points). c is searched on
    a log grid in [0.1, 2.0]; for each c, (a, b) solve the linear LS of
    ``val = a - b * x`` with ``x = n^{-c}``. Picks the (a, b, c) with min SSE. Simple
    and dependency-free (no scipy.optimize) — the grid is fine for a monotone
    saturating curve on a 7-point n-grid.
    """
    pts = [
        (float(n), float(v))
        for n, v in zip(ns, vals, strict=True)
        if v is not None and math.isfinite(v)
    ]
    if len(pts) < 3:
        return None
    n_arr = np.array([p[0] for p in pts])
    v_arr = np.array([p[1] for p in pts])
    best = None
    for c in np.geomspace(0.1, 2.0, 40):
        x = n_arr ** (-c)
        A = np.column_stack([np.ones_like(x), -x])  # val = a*1 + b*(-x)
        try:
            coef, *_ = np.linalg.lstsq(A, v_arr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b = float(coef[0]), float(coef[1])
        sse = float(np.sum((A @ coef - v_arr) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, b, float(c))
    if best is None:
        return None
    return best[1], best[2], best[3]


def _contexts_needed(
    curve_ns: list[int],
    rho_by_n: list[float | None],
    target_rho: float,
) -> float | None:
    """n needed to reach ``target_rho`` from the fitted power curve (or None).

    Solves ``a - b*n^{-c} = target`` → ``n = (b / (a - target))^{1/c}`` when the fitted
    asymptote ``a`` exceeds the target. Returns the extrapolated n (may exceed the
    observed grid), None if the fit failed or the asymptote never reaches the target.
    """
    fit = _fit_power_curve([float(n) for n in curve_ns], rho_by_n)
    if fit is None:
        return None
    a, b, c = fit
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)) or c <= 0:
        return None
    denom = a - target_rho
    if denom <= 0 or b <= 0:
        return None  # asymptote below the target — never reachable
    try:
        return float((b / denom) ** (1.0 / c))
    except (ValueError, OverflowError):
        return None


# ── E0 target loading (graded_mean per context) ──────────────────────────────


def _load_graded_e0(highm_path: Path, lowm_path: Path) -> dict[str, dict]:
    """{behavior: {ctx: cell}} merged from the two graded JSONs (either may be absent)."""
    merged: dict[str, dict] = {}
    for p in (highm_path, lowm_path):
        if p.exists():
            blob = json.loads(p.read_text())
            for beh, per_ctx in blob.get("e0", {}).items():
                merged[beh] = per_ctx
    return merged


def _validate_behavior_coverage(behaviors: list[str], *, explicit_behaviors: bool) -> None:
    """Fit-side strict-subset defense (restored-e0-payload-coverage leg c).

    With NO explicit ``--behaviors`` subset the fit is supposed to analyze all 8 #812
    behaviors (``HIGH_M + LOW_M``). If the loaded graded E0 carries a STRICT SUBSET of the
    expected 8, a partial / truncated regrade slipped past the regrade idempotence gate,
    and the default ``behaviors = list(graded.keys())`` would SILENTLY analyze a behavior
    subset. RAISE loud instead. An explicit ``--behaviors`` subset opts out (a deliberate
    smoke/debug slice). No-op when the full 8 are present."""
    if explicit_behaviors:
        return
    expected = set(HIGH_M + LOW_M)
    present = set(behaviors)
    if present != expected:
        missing = sorted(expected - present)
        raise RuntimeError(
            "graded E0 covers a strict subset of the expected 8 #812 behaviors "
            f"(present={sorted(present)}, missing={missing}) with no explicit "
            "--behaviors subset — a partial/truncated regrade slipped the idempotence "
            "gate; refusing to silently analyze a behavior subset (pass --behaviors "
            "explicitly to analyze a deliberate subset)"
        )


def _graded_target(per_ctx: dict, ctx_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    vals, kept = [], []
    for c in ctx_ids:
        cell = per_ctx.get(c)
        if not cell:
            continue
        v = cell.get("graded_mean")
        if v is None:
            continue
        vals.append(float(v))
        kept.append(c)
    return np.array(vals, dtype=np.float64), kept


# ── the per-(behavior, layer) operator sweep ──────────────────────────────────


def _operator_features(
    inputs: dict, op: str, layer_idx: int, ctx_order: list[int], attn_learned_pooled: np.ndarray
) -> np.ndarray:
    """(N, D) feature matrix for one operator + layer over the given context rows.

    Raw pools -> raw (H,); *_pca -> matched-PCA on the raw pool; unpooled ->
    per-position PCA on aligned positions. attn_learned pooled is precomputed
    (passed in) since it is a fit arm.
    """
    layers = inputs["layers"]
    li = layers.index(layer_idx) if layer_idx in layers else layer_idx
    if op == "mean":
        return inputs["mean"][ctx_order, li].astype(np.float64)
    if op == "max":
        return inputs["max"][ctx_order, li].astype(np.float64)
    if op == "attn_fixed":
        return inputs["attn_fixed"][ctx_order, li].astype(np.float64)
    if op == "attn_learned":
        return attn_learned_pooled  # already (N, H) for this cell
    if op == "mean_pca":
        return _loco_pca_features(inputs["mean"][ctx_order, li].astype(np.float64), D_IN_PCA)
    if op == "max_pca":
        return _loco_pca_features(inputs["max"][ctx_order, li].astype(np.float64), D_IN_PCA)
    if op == "attn_fixed_pca":
        return _loco_pca_features(inputs["attn_fixed"][ctx_order, li].astype(np.float64), D_IN_PCA)
    if op == "attn_learned_pca":
        return _loco_pca_features(attn_learned_pooled, D_IN_PCA)
    if op == "unpooled":
        aligned = inputs["aligned_pos"][ctx_order, li].astype(np.float64)  # (N, P, H)
        _n, pdim, _h = aligned.shape
        feats = []
        for pos in range(pdim):
            feats.append(_loco_pca_features(aligned[:, pos], D_IN_PCA))
        return np.concatenate(feats, axis=1)  # (N, P*d_in)
    raise ValueError(f"unknown operator {op!r}")


def _selection_symmetric_null(
    per_layer_rho: dict[str, dict[int, float]],
    per_layer_preds: dict[str, dict[int, np.ndarray]],
    y: np.ndarray,
    *,
    op_a: str,
    op_b: str,
    layers: list[int],
    n_iter: int,
    seed: int,
) -> dict:
    """Selection-symmetric shuffle null for max-over-layer Delta-rho(op_a - op_b).

    Each shuffle draw permutes y ONCE, recomputes rho for BOTH ops at EVERY layer
    from the STORED held-out preds (rank-correlate preds vs the permuted y — a
    valid permutation of the held-out target), takes the SAME max-over-layer
    selection, forms the null Delta. Persists the per-draw x per-layer matrix.
    Observed row + n_iter null rows.
    """
    rng = np.random.default_rng(seed)
    # observed per-layer Delta
    obs_delta = np.array(
        [per_layer_rho[op_a].get(li, np.nan) - per_layer_rho[op_b].get(li, np.nan) for li in layers]
    )
    null_matrix = np.full((n_iter, len(layers)), np.nan)
    for it in range(n_iter):
        yp = rng.permutation(y)
        for lj, li in enumerate(layers):
            pa = per_layer_preds[op_a].get(li)
            pb = per_layer_preds[op_b].get(li)
            if pa is None or pb is None:
                continue
            ra = _rho(pa, yp)
            rb = _rho(pb, yp)
            if ra is None or rb is None:
                continue
            null_matrix[it, lj] = ra - rb
    obs_max = float(np.nanmax(obs_delta)) if np.isfinite(obs_delta).any() else float("nan")
    null_max = np.nanmax(null_matrix, axis=1)
    null_max = null_max[np.isfinite(null_max)]
    p975 = float(np.percentile(null_max, 97.5)) if null_max.size else float("nan")
    return {
        "op_a": op_a,
        "op_b": op_b,
        "layers": layers,
        "observed_max_over_layer_delta": obs_max,
        "null_max_over_layer_p97_5": p975,
        "observed_row": obs_delta.tolist(),
        "null_matrix": null_matrix.tolist(),  # (n_iter, n_layers)
    }


# ── SECONDARY DV: answer-profile reconstruction (#722 base-map, CONCERN 7) ────


def _load_context_vectors(path: Path) -> dict | None:
    """Load c_C = context_vectors_mean.pt -> {"C": (Nc, Lc, H), "ids": [Nc]} or None.

    Accepts the #594 layout ``{context_vectors_mean: (Nc,Lc,H), instance_ids: [...]}``
    plus a couple of key aliases; returns None (skip the DV, never fabricate) when the
    file is absent or the expected 3-D tensor cannot be found.
    """
    if not path.exists():
        return None
    blob = torch.load(path, weights_only=False)
    if not isinstance(blob, dict):
        return None
    cvec = None
    # #594's context_vectors_mean.pt stores the (Nc, Lc, H) tensor under the key
    # ``tensor`` with ids under ``instance_ids`` (verified on HF); accept the aliases
    # too so a differently-keyed sibling still resolves.
    for k in ("tensor", "context_vectors_mean", "context_vectors", "c_C", "mean"):
        if k in blob and hasattr(blob[k], "ndim") and getattr(blob[k], "ndim", 0) == 3:
            cvec = blob[k]
            break
    if cvec is None:
        return None
    arr = cvec.numpy() if hasattr(cvec, "numpy") else np.asarray(cvec)
    if arr.ndim != 3:
        return None
    ids = None
    for k in ("instance_ids", "context_ids", "ctx_ids", "ids"):
        if k in blob and blob[k] is not None:
            ids = [str(c) for c in blob[k]]
            break
    return {"C": arr.astype(np.float64), "ids": ids}


def _reconstruction_r2(
    cc: dict,
    inputs: dict,
    layers: list[int],
    layers_all: list[int],
    all_ctx: list[str],
    out_dir: Path,
    fig_dir: Path,
) -> dict:
    """Per (layer x operator) skill-over-mean R2 of ridge(c_C -> operator-summary).

    Reuses the #722 convention exactly: ``ridge_predict_loco_centered`` (LOCO ridge on
    the train-mean-centered PCA target) + ``skill_over_mean_r2``. The single-vector
    operators (mean/max/attn_fixed) target their (H,) summary PCA-reduced to
    ``min(48, n-2)`` dims (train-only within the LOCO ridge's fold standardization);
    the ``unpooled`` case targets the concatenated per-position summary. Aligns c_C
    contexts to ``inputs['ctx_ids']`` by id when c_C carries ids, else assumes the
    stored order matches. Writes ``reconstruction_r2.json`` + the hero figure.
    """
    C = cc["C"]  # (Nc, Lc, H)
    cc_ids = cc["ids"]
    inp_ids = [str(c) for c in inputs["ctx_ids"]]
    # kept = contexts present in BOTH c_C and the pooling inputs (order = inp_ids)
    if cc_ids is not None:
        cc_pos = {cid: i for i, cid in enumerate(cc_ids)}
        kept = [(j, cc_pos[cid]) for j, cid in enumerate(inp_ids) if cid in cc_pos]
    else:
        m = min(len(inp_ids), C.shape[0])
        kept = [(j, j) for j in range(m)]
    if len(kept) < 4:
        return {"note": f"reconstruction skipped: only {len(kept)} contexts in c_C∩inputs"}
    inp_idx = [j for j, _ in kept]
    cc_idx = [k for _, k in kept]
    n = len(kept)
    d_tgt = min(48, n - 2)

    single_ops = ["mean", "max", "attn_fixed"]
    r2: dict[str, dict[str, float | None]] = {op: {} for op in [*single_ops, "unpooled"]}
    for li in layers:
        lai = layers_all.index(li)
        Xc = C[cc_idx, min(lai, C.shape[1] - 1)]  # (n, H) context vectors at this layer
        # single-vector operator targets
        for op in single_ops:
            Y = inputs[op][inp_idx, lai].astype(np.float64)  # (n, H)
            mu, comps, _ = robust_pca_basis(Y, d_tgt)
            Yp = (Y - mu) @ comps.T  # (n, d) PCA target (train-only handled in ridge)
            preds = ridge_predict_loco_centered(Xc, Yp)
            r2[op][str(li)] = skill_over_mean_r2(preds, Yp)["skill"]
        # unpooled: per-position summary concatenated then PCA-reduced as one target
        aligned = inputs["aligned_pos"][inp_idx, lai].astype(np.float64)  # (n, P, H)
        Yu = aligned.reshape(n, -1)  # (n, P*H)
        Yu = np.nan_to_num(Yu)
        mu, comps, _ = robust_pca_basis(Yu, d_tgt)
        Yup = (Yu - mu) @ comps.T
        preds = ridge_predict_loco_centered(Xc, Yup)
        r2["unpooled"][str(li)] = skill_over_mean_r2(preds, Yup)["skill"]

    _make_reconstruction_figure(r2, layers, fig_dir)
    result = {
        "n_contexts": n,
        "d_target_pca": d_tgt,
        "per_layer_r2": r2,
        "validity_anchor": {
            "operator": "mean",
            "note": "mean @ ~L18 ≈ 0.80 is the #722 pipeline-validity anchor",
            "mean_r2_by_layer": r2.get("mean", {}),
        },
    }
    _atomic_write_json(out_dir / "reconstruction_r2.json", {"reconstruction": result})
    logger.info("WROTE %s", out_dir / "reconstruction_r2.json")
    return result


def _make_reconstruction_figure(r2: dict, layers: list[int], fig_dir: Path) -> None:
    """Per-layer skill-over-mean R² bars/curves per operator (the #722 SECONDARY)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

        set_paper_style("blog")
    except Exception as exc:
        logger.warning("figure deps unavailable (%s) — skipping reconstruction figure", exc)
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for op in ("mean", "max", "attn_fixed", "unpooled"):
        ys = [r2.get(op, {}).get(str(li)) for li in layers]
        xs = [li for li, v in zip(layers, ys, strict=True) if v is not None]
        yy = [v for v in ys if v is not None]
        if xs:
            ax.plot(xs, yy, marker="o", label=op, alpha=0.85)
    ax.axhline(0.80, ls="--", color="gray", label="#722 mean@L18 anchor ~0.80")
    ax.set_xlabel("layer")
    ax.set_ylabel("skill-over-mean R² (c_C → operator summary)")
    ax.set_title("answer-profile reconstruction R² per layer (SECONDARY DV)")
    ax.legend(fontsize=7, ncol=2)
    savefig_paper(fig, "reconstruction_r2", dir=str(fig_dir))
    plt.close(fig)


def main() -> int:  # noqa: C901 — one long sweep driver; decomposed by helper above
    load_dotenv()
    ap = argparse.ArgumentParser(description="Issue 812 pooling-operator fit sweep (CPU).")
    ap.add_argument("--inputs", default="data/issue_812/store/pooling_inputs.pt")
    ap.add_argument("--graded-highm", default="eval_results/issue_812/graded_e0_highm.json")
    ap.add_argument("--graded-lowm", default="eval_results/issue_812/graded_e0_lowm.json")
    ap.add_argument("--out-dir", default="eval_results/issue_812")
    ap.add_argument("--fig-dir", default="figures/issue_812")
    ap.add_argument("--behaviors", default="", help="comma-separated subset (default: all present)")
    ap.add_argument("--layers", default="", help="comma-separated layer subset (default: all)")
    ap.add_argument("--contexts", type=int, default=None, help="limit N contexts (smoke)")
    ap.add_argument("--shuffle-iters", type=int, default=SHUFFLE_ITERS)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=658)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--skip-learning-curve", action="store_true")
    ap.add_argument("--skip-reconstruction", action="store_true")
    ap.add_argument(
        "--context-vectors",
        default="data/issue_812/context_vectors_mean.pt",
        help="c_C (issue594 context_vectors_mean.pt) for the reconstruction DV (§4.3)",
    )
    ap.add_argument(
        "--lc-repeats",
        type=int,
        default=20,
        help="B repeats per learning-curve n' subsample (variance + extrapolation CI)",
    )
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.num_threads)

    inputs = torch.load(args.inputs, weights_only=False)
    # numpy-ify the tensors
    for key in ("mean", "max", "attn_fixed", "aligned_pos", "coverage"):
        inputs[key] = inputs[key].numpy()
    all_ctx = list(inputs["ctx_ids"])
    layers_all = list(inputs["layers"])
    if args.contexts is not None:
        all_ctx = all_ctx[: args.contexts]
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else layers_all
    layers = [li for li in layers if li in layers_all]

    graded = _load_graded_e0(Path(args.graded_highm), Path(args.graded_lowm))
    explicit_behaviors = bool(args.behaviors)
    behaviors = (
        [b for b in args.behaviors.split(",") if b.strip()]
        if explicit_behaviors
        else list(graded.keys())
    )
    behaviors = [b for b in behaviors if b in graded]
    if not behaviors:
        raise RuntimeError("no behaviors with graded E0 present — cannot fit (fail-loud)")
    _validate_behavior_coverage(behaviors, explicit_behaviors=explicit_behaviors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Preflight: sqrt(r_yy) reliability ceiling per behavior (MF2 gate) ──
    # MF2 gates INTERPRETING Delta-rho against a non-null in-range ceiling FOR A GIVEN
    # BEHAVIOR — it does NOT require every behavior to have one. A behavior whose graded
    # E0 carries ~no between-context signal (deception on base Qwen: per-context sd~3 on
    # the 0-100 scale, vs 8-27 for the others) has an undefined split-half correlation →
    # Spearman-Brown r_yy <= 0 → null sqrt(r_yy). That is a REAL property of the target
    # (analogous to the pre-registered broad_em floor exclusion, plan §4.2), not a run
    # failure. So: EXCLUDE such a behavior from any Delta-rho-vs-ceiling interpretation,
    # RECORD it (with its measured diagnostics) under ``reliability_excluded``, and keep
    # its per-cell fits (the raw rho values are honest data — they are just never
    # normalized against a null ceiling). RAISE only if ALL behaviors are null (nothing
    # left to analyze).
    reliability: dict[str, dict] = {}
    for beh in behaviors:
        rel = _reliability_for_behavior(graded[beh], beh, seed=args.seed, n_boot=args.n_bootstrap)
        reliability[beh] = rel
    reliability_excluded = _reliability_preflight(reliability, behaviors)

    all_ops = ["mean"] + [o for o in POOL_OPS if o != "mean"] + PCA_OPS + ["unpooled"]
    results: dict[str, dict] = {}
    selection_matrices: dict[str, dict] = {}

    for beh in behaviors:
        y_full, kept_ctx = _graded_target(graded[beh], all_ctx)
        if len(kept_ctx) < 4:
            logger.warning("[%s] only %d contexts with graded E0 — skipping", beh, len(kept_ctx))
            continue
        ctx_order = [all_ctx.index(c) for c in kept_ctx]
        y = y_full

        per_layer_rho: dict[str, dict[int, float | None]] = {op: {} for op in all_ops}
        per_layer_preds: dict[str, dict[int, np.ndarray]] = {op: {} for op in all_ops}
        per_layer_boot: dict[str, dict[int, dict]] = {op: {} for op in all_ops}

        for li in layers:
            # attn_learned pooled once per (beh, layer) — the fit arm
            aligned = inputs["aligned_pos"][ctx_order, layers_all.index(li)].astype(np.float64)
            attn_pooled = _fit_attn_learned_pool(aligned, y, seed=args.seed, device=args.device)
            for op in all_ops:
                X = _operator_features(inputs, op, li, ctx_order, attn_pooled)
                rho, preds = _ridge_rho_from_features(X, y)
                per_layer_rho[op][li] = rho
                per_layer_preds[op][li] = preds
                # BLOCKER 3(a): a degenerate cell (PCA-collapsed / floor-saturated E0
                # with almost no rank variation) makes _cluster_bootstrap_rho RAISE
                # RuntimeError. Guard it per-cell → record ci95=null + bootstrap_error
                # and CONTINUE, so one bad cell can never abort the whole sweep.
                try:
                    boot = _cluster_bootstrap_rho(
                        preds, y, n_boot=args.n_bootstrap, seed=args.seed + li
                    )
                    per_layer_boot[op][li] = {
                        "ci95": boot["ci95"] if boot else None,
                        "bootstrap_error": None,
                    }
                except RuntimeError as exc:
                    logger.warning(
                        "[%s] bootstrap degenerate at op=%s layer=%d — ci95=null (%s)",
                        beh,
                        op,
                        li,
                        exc,
                    )
                    per_layer_boot[op][li] = {"ci95": None, "bootstrap_error": str(exc)}

        # selection-symmetric shuffle nulls for the two headline Deltas
        sel_unpooled_mean = _selection_symmetric_null(
            {op: {li: (per_layer_rho[op][li] or 0.0) for li in layers} for op in all_ops},
            per_layer_preds,
            y,
            op_a="unpooled",
            op_b="mean",
            layers=layers,
            n_iter=args.shuffle_iters,
            seed=args.seed,
        )
        sel_unpooled_meanpca = _selection_symmetric_null(
            {op: {li: (per_layer_rho[op][li] or 0.0) for li in layers} for op in all_ops},
            per_layer_preds,
            y,
            op_a="unpooled",
            op_b="mean_pca",
            layers=layers,
            n_iter=args.shuffle_iters,
            seed=args.seed + 7,
        )
        sel_path = out_dir / f"selection_matrix_{beh}.json"
        selection_matrices[beh] = {
            "unpooled_vs_mean": sel_unpooled_mean,
            "unpooled_vs_mean_pca": sel_unpooled_meanpca,
        }
        with open(sel_path, "w") as f:
            json.dump(selection_matrices[beh], f)
        logger.info("[%s] wrote %s", beh, sel_path)

        results[beh] = {
            "n_contexts": len(kept_ctx),
            "layers": layers,
            "sqrt_r_yy": reliability[beh]["sqrt_r_yy"],
            "per_layer_rho": {
                op: {str(li): per_layer_rho[op][li] for li in layers} for op in all_ops
            },
            "per_layer_ci95": {
                op: {str(li): per_layer_boot[op][li]["ci95"] for li in layers} for op in all_ops
            },
            "delta_rho_unpooled_vs_mean_max": sel_unpooled_mean["observed_max_over_layer_delta"],
            "delta_rho_unpooled_vs_mean_pca_max": sel_unpooled_meanpca[
                "observed_max_over_layer_delta"
            ],
            "shuffle_null_p97_5_vs_mean": sel_unpooled_mean["null_max_over_layer_p97_5"],
            "shuffle_null_p97_5_vs_mean_pca": sel_unpooled_meanpca["null_max_over_layer_p97_5"],
        }
        # "worked" verdict per plan §6: unpooled helps iff Delta(unpooled - mean_pca)
        # max-over-layer > shuffle p97.5 AND gap < sqrt(r_yy). A behavior with a null
        # reliability ceiling is EXCLUDED from this Delta-rho-vs-ceiling interpretation
        # (there is no ceiling to compare the gap against) — its verdict is None
        # (uninterpretable), NOT a vacuous True from ``d < ryy`` being trivially
        # satisfied when ryy is None. The per-cell rho fits are still stored above.
        d = results[beh]["delta_rho_unpooled_vs_mean_pca_max"]
        p = results[beh]["shuffle_null_p97_5_vs_mean_pca"]
        ryy = reliability[beh]["sqrt_r_yy"]
        if beh in reliability_excluded:
            results[beh]["reliability_excluded"] = True
            results[beh]["unpooling_helps"] = None
        else:
            results[beh]["reliability_excluded"] = False
            results[beh]["unpooling_helps"] = bool(
                math.isfinite(d) and math.isfinite(p) and d > p and d < ryy
            )

        # BLOCKER 3(b): checkpoint pooling_fit_results.json after EACH behavior lands
        # (atomic .tmp + os.replace) so a crash on a later behavior/cell never loses
        # the completed behaviors' fits (CLAUDE.md checkpoint-per-phase).
        _atomic_write_json(
            out_dir / "pooling_fit_results.json",
            {
                "meta": {
                    "issue": 812,
                    "git_commit": _git_commit(),
                    "created_utc": _now_iso(),
                    "checkpoint": "partial — mid-sweep incremental",
                    "behaviors_completed": list(results),
                    "behaviors_requested": behaviors,
                    "d_in_pca": D_IN_PCA,
                    "shuffle_iters": args.shuffle_iters,
                    "n_bootstrap": args.n_bootstrap,
                    "seed": args.seed,
                    "layers": layers,
                },
                "results": results,
                "reliability": reliability,
                "reliability_excluded": reliability_excluded,
                "learning_curve": {},
                "reconstruction": {},
            },
        )
        logger.info("[%s] checkpointed pooling_fit_results.json (%d done)", beh, len(results))

    # ── learning curve (per §6, unless skipped) ────────────────────────────────
    # Per #742 Stage-2: subsample n' (B repeats each), recompute rho_mean/rho_unpooled/
    # Δρ with bootstrap variance, and EXTRAPOLATE the contexts needed to reach
    # 0.90·√(r_yy) with a bootstrap CI (CONCERN 6 — the outcome-(iii) deliverable).
    learning_curve: dict[str, dict] = {}
    if not args.skip_learning_curve:
        for beh in behaviors:
            if beh not in results:
                continue
            y_full, kept_ctx = _graded_target(graded[beh], all_ctx)
            ctx_order = [all_ctx.index(c) for c in kept_ctx]
            # pick the best layer by mean-op rho for the curve
            best_li = max(
                layers,
                key=lambda li: results[beh]["per_layer_rho"]["mean"].get(str(li)) or -1,
            )
            grid = [g for g in LEARNING_CURVE_GRID if g <= len(kept_ctx)]
            n_rep = max(4, min(args.n_bootstrap, args.lc_repeats))
            curve = {}
            # rho_unpooled(n) per bootstrap repeat — feeds the extrapolation + its CI
            unpooled_reps_by_n: dict[int, list[float]] = {}
            for nprime in grid:
                r_mean_reps, r_unp_reps, delta_reps = [], [], []
                for rep in range(n_rep):
                    rng_rep = random.Random(args.seed * 100003 + nprime * 101 + rep)
                    idx = list(range(len(kept_ctx)))
                    rng_rep.shuffle(idx)
                    sub = sorted(idx[:nprime])
                    sub_ctx = [ctx_order[j] for j in sub]
                    ysub = y_full[sub]
                    Xmean = inputs["mean"][sub_ctx, layers_all.index(best_li)].astype(np.float64)
                    rmean, _ = _ridge_rho_from_features(Xmean, ysub)
                    aligned = inputs["aligned_pos"][sub_ctx, layers_all.index(best_li)].astype(
                        np.float64
                    )
                    pdim = aligned.shape[1]
                    feats = [_loco_pca_features(aligned[:, pos], D_IN_PCA) for pos in range(pdim)]
                    Xunp = np.concatenate(feats, axis=1)
                    runp, _ = _ridge_rho_from_features(Xunp, ysub)
                    if rmean is not None:
                        r_mean_reps.append(rmean)
                    if runp is not None:
                        r_unp_reps.append(runp)
                    if rmean is not None and runp is not None:
                        delta_reps.append(runp - rmean)
                unpooled_reps_by_n[nprime] = list(r_unp_reps)
                curve[str(nprime)] = {
                    "rho_mean": float(np.mean(r_mean_reps)) if r_mean_reps else None,
                    "rho_mean_std": float(np.std(r_mean_reps)) if r_mean_reps else None,
                    "rho_unpooled": float(np.mean(r_unp_reps)) if r_unp_reps else None,
                    "rho_unpooled_std": float(np.std(r_unp_reps)) if r_unp_reps else None,
                    "delta": float(np.mean(delta_reps)) if delta_reps else None,
                    "delta_std": float(np.std(delta_reps)) if delta_reps else None,
                    "n_repeats": n_rep,
                }
            # ── contexts-needed extrapolation to 0.90·√(r_yy) ───────────────────
            ryy = reliability[beh]["sqrt_r_yy"]
            target = 0.90 * ryy if ryy is not None else None
            contexts_needed = None
            contexts_needed_ci95 = None
            cn_verdict = "target undefined (sqrt_r_yy null)"
            if target is not None:
                ns = list(grid)
                mean_rho = [
                    (float(np.mean(unpooled_reps_by_n[n])) if unpooled_reps_by_n[n] else None)
                    for n in ns
                ]
                contexts_needed = _contexts_needed(ns, mean_rho, target)
                # bootstrap the extrapolation: resample the per-n rho draws, refit
                cn_boot: list[float] = []
                for b in range(min(args.n_bootstrap, 500)):
                    rb = random.Random(args.seed + 20000 + b)
                    boot_rho = []
                    for n in ns:
                        reps = unpooled_reps_by_n[n]
                        boot_rho.append(reps[rb.randrange(len(reps))] if reps else None)
                    cn = _contexts_needed(ns, boot_rho, target)
                    if cn is not None and math.isfinite(cn) and cn > 0:
                        cn_boot.append(cn)
                if len(cn_boot) >= 20:
                    cn_boot.sort()
                    lo = cn_boot[int(0.025 * len(cn_boot))]
                    hi = cn_boot[int(0.975 * len(cn_boot)) - 1]
                    contexts_needed_ci95 = [float(lo), float(hi)]
                    # "not extrapolable at this precision" if the CI spans >1 order of magnitude
                    if lo > 0 and hi / lo > 10.0:
                        cn_verdict = (
                            "not extrapolable at this precision (CI spans >1 order of magnitude)"
                        )
                        contexts_needed = None
                    else:
                        cn_verdict = "extrapolated"
                elif contexts_needed is not None:
                    cn_verdict = "point estimate only (bootstrap CI unavailable)"
                else:
                    cn_verdict = "asymptote below target — not reachable by more contexts"
            learning_curve[beh] = {
                "best_layer": best_li,
                "curve": curve,
                "target_rho": target,
                "target_frac_of_ceiling": 0.90,
                "contexts_needed": contexts_needed,
                "contexts_needed_ci95": contexts_needed_ci95,
                "contexts_needed_verdict": cn_verdict,
            }

    # ── SECONDARY: answer-profile reconstruction (per §4.3 / §6.5, unless skipped) ─
    # The #722-convention base-map DV: per (layer x operator), LOCO ridge
    # c_C → operator-summary (per-position target for the unpooled case), held-out
    # skill-over-mean R². The `mean` operator @ L18 ≈ 0.80 is the pipeline-validity
    # anchor. Computed on the SAME kept-context intersection as the primary DV.
    reconstruction: dict[str, dict] = {}
    if not args.skip_reconstruction:
        cc = _load_context_vectors(Path(args.context_vectors))
        if cc is None:
            reconstruction = {
                "note": (
                    "c_C (issue594 context_vectors_mean.pt) not resolvable "
                    f"({args.context_vectors}) — reconstruction DV skipped this run"
                )
            }
        else:
            reconstruction = _reconstruction_r2(
                cc, inputs, layers, layers_all, all_ctx, out_dir, fig_dir
            )

    # ── write results + figures ────────────────────────────────────────────────
    meta = {
        "issue": 812,
        "git_commit": _git_commit(),
        "created_utc": _now_iso(),
        "d_in_pca": D_IN_PCA,
        "shuffle_iters": args.shuffle_iters,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_contexts_requested": len(all_ctx),
        "layers": layers,
        "behaviors": behaviors,
        "operators": all_ops,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    payload = {
        "meta": meta,
        "results": results,
        "reliability": reliability,
        "reliability_excluded": reliability_excluded,
        "learning_curve": learning_curve,
        "reconstruction": reconstruction,
    }
    fit_path = out_dir / "pooling_fit_results.json"
    _atomic_write_json(fit_path, payload)  # final complete write overwrites the partial
    logger.info("WROTE %s", fit_path)

    rel_path = out_dir / "reliability_and_learning_curve.json"
    _atomic_write_json(
        rel_path,
        {
            "meta": meta,
            "reliability": reliability,
            "reliability_excluded": reliability_excluded,
            "learning_curve": learning_curve,
        },
    )
    logger.info("WROTE %s", rel_path)

    _make_figures(results, reliability, layers, behaviors, fig_dir)
    logger.info(
        "Fit sweep complete: %d behaviors x %d layers x %d operators",
        len(results),
        len(layers),
        len(all_ops),
    )
    return 0


def _make_figures(results, reliability, layers, behaviors, fig_dir: Path) -> None:
    """Hero per-layer rho curves per behavior (over-produce for the analyzer)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from explore_persona_space.analysis.paper_plots import (
            paper_palette_role,
            savefig_paper,
            set_paper_style,
        )

        set_paper_style("blog")
    except Exception as exc:
        logger.warning("figure deps unavailable (%s) — skipping figures", exc)
        return

    role = {
        "mean": "baseline",
        "mean_pca": "control",
        "unpooled": "primary",
        "attn_learned": "accent",
    }
    for beh in behaviors:
        if beh not in results:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for op in ("mean", "mean_pca", "max", "attn_fixed", "attn_learned", "unpooled"):
            ys = [results[beh]["per_layer_rho"].get(op, {}).get(str(li)) for li in layers]
            xs = [li for li, v in zip(layers, ys, strict=True) if v is not None]
            yy = [v for v in ys if v is not None]
            if not xs:
                continue
            color = paper_palette_role(role.get(op, "neutral")) if op in role else None
            ax.plot(xs, yy, marker="o", label=op, color=color, alpha=0.85)
        ryy = reliability[beh]["sqrt_r_yy"]
        if ryy is not None:
            ax.axhline(ryy, ls="--", color="gray", label="sqrt(r_yy) ceiling")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out Spearman rho")
        ax.set_title(f"{beh}: pooling-operator E0-prediction rho per layer")
        ax.legend(fontsize=7, ncol=2)
        savefig_paper(fig, f"rho_per_layer_{beh}", dir=str(fig_dir))
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
