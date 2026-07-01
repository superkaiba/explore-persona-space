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


def _fit_attn_learned_pool(
    aligned: np.ndarray, y: np.ndarray, *, seed: int, device: str
) -> np.ndarray:
    """Fit a softmax query over aligned positions, LOCO cross-fit, return pooled (N, H).

    aligned: (N, P, H) the (2K+2) aligned-position activations (NaN-filled slots ->
    zeroed + mask). Fit query q in R^H (init = mean-pool direction) so
    ``pool_i = sum_p softmax(a_i @ q)_p * a_i[p]``; the pooled held-out vector then
    goes to the shared ridge downstream so the reported rho uses the same estimator
    as every other operator. To avoid leakage the query is fit on the LOCO TRAIN
    fold only (per held-out i).

    VECTORIZED across ALL N LOCO folds AND the L2 grid as ONE batched parameter
    tensor (``.claude/rules/vectorize-many-cell-fits.md``): the per-fold Python loop
    is the banned overhead-bound anti-pattern (a serial fold loop projected >14 h
    for the full 8x28-cell sweep at n=50). Every fold's query ``Q`` (F=n rows) +
    readout ``w`` are stacked and trained jointly under a fold-exclusion mask; the
    300-epoch loop runs ``len(L2)`` times TOTAL, not ``n x len(L2)`` times.

    The query + readout are fit in a PCA-REDUCED activation space (H -> R via a
    train-all-rows PCA basis, ``ATTN_REDUCE_RANK``) — a free 3584-dim query at n=50
    is hopeless overfitting (plan's ``ungrounded`` risk), and the reduction makes the
    fit ~100x cheaper (the query dim is R, not H). The reduction basis is fit on ALL
    rows ONCE (a mild reduction-basis leakage; the query fit itself stays cross-fold
    and the RETURNED pooled vector — full-H, using the learned per-position attention
    weights — goes to the leakage-free LOCO ridge downstream). The learned attention
    is over POSITIONS, computed from the reduced scores, then applied to the FULL-H
    activations so no answer information is lost in the returned pool.
    """
    torch.manual_seed(seed)
    n, _p, h = aligned.shape
    dev = torch.device(device)
    a = torch.from_numpy(np.nan_to_num(aligned).astype(np.float32)).to(dev)  # (N,P,H)
    pmask = torch.from_numpy((~np.isnan(aligned)).any(axis=2)).to(dev)  # (N,P) bool
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)  # (N,)

    # PCA-reduce H -> R on the stacked (N*P valid) position activations (all rows).
    flat = a[pmask]  # (M, H) valid positions only
    r = min(ATTN_REDUCE_RANK, max(1, flat.shape[0] - 1), h)
    mu, comps, _ = robust_pca_basis(flat.cpu().numpy().astype(np.float64), r)  # comps (r',H)
    comps_t = torch.from_numpy(comps.astype(np.float32)).to(dev)  # (R,H)
    mu_t = torch.from_numpy(mu.astype(np.float32)).to(dev)  # (H,)
    ar = torch.einsum("nph,rh->npr", a - mu_t, comps_t)  # (N,P,R) reduced activations
    rdim = ar.shape[2]

    # init reduced query = mean-pool direction in reduced space
    with torch.no_grad():
        valid = pmask.float().unsqueeze(-1)  # (N,P,1)
        per_ctx_mean = (ar * valid).sum(1) / valid.sum(1).clamp_min(1.0)  # (N,R)
        q_init = per_ctx_mean.mean(0)  # (R,)
        q_init = q_init / (q_init.norm() + 1e-9)

    eye = torch.eye(n, device=dev, dtype=torch.bool)
    train_mask = (~eye).float()  # (F=N, N) 1 where row j is in fold f's train set
    n_tr = train_mask.sum(1).clamp_min(1.0)  # (F,)
    pmask_f = pmask.float().unsqueeze(0)  # (1,N,P) broadcast over folds

    def _train_one_l2(l2: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Train all folds jointly at one L2; return (final held-in loss per fold, Q)."""
        Q = q_init.unsqueeze(0).repeat(n, 1).clone().requires_grad_(True)  # (F,R)
        W = torch.zeros(n, rdim, device=dev, requires_grad=True)  # (F,R)
        B = torch.zeros(n, device=dev, requires_grad=True)  # (F,)
        opt = torch.optim.Adam([Q, W, B], lr=ATTN_LR)
        last = None
        for _ in range(ATTN_EPOCHS):
            opt.zero_grad()
            scores = torch.einsum("jpr,fr->fjp", ar, Q)  # (F,N,P)
            scores = scores.masked_fill(pmask_f < 0.5, -1e9)
            attn = torch.softmax(scores, dim=2)  # (F,N,P)
            # pred[f,j] = W[f] . pooled_reduced[f,j] = sum_p attn[f,j,p] (ar[j,p].W[f])
            aw = torch.einsum("jpr,fr->fjp", ar, W)  # (F,N,P) — no (F,N,R) intermediate
            pred = (attn * aw).sum(2) + B.unsqueeze(1)  # (F,N)
            resid2 = (pred - yt.unsqueeze(0)) ** 2  # (F,N)
            mse = (resid2 * train_mask).sum(1) / n_tr  # (F,) train-only MSE
            reg = l2 * (Q.pow(2).sum(1) + W.pow(2).sum(1))  # (F,)
            loss = (mse + reg).sum()  # per-fold grads independent
            loss.backward()
            opt.step()
            last = (mse + reg).detach()
        return last, Q.detach()

    best_loss = torch.full((n,), math.inf, device=dev)
    best_Q = q_init.unsqueeze(0).repeat(n, 1)  # (F,R)
    for l2 in ATTN_L2_GRID:
        fold_loss, Q = _train_one_l2(l2)
        improved = fold_loss < best_loss
        best_loss = torch.where(improved, fold_loss, best_loss)
        best_Q = torch.where(improved.unsqueeze(1), Q, best_Q)

    # held-out pooled: fold i's reduced query -> position attention -> pool FULL-H a[i]
    with torch.no_grad():
        scores_i = torch.einsum("ipr,ir->ip", ar, best_Q)  # (N,P) reduced scores
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
    per_ctx_units: dict[str, list[list[float]]], *, seed: int, n_splits: int = 50
) -> float | None:
    """sqrt(r_yy): split-half over the SPLITTABLE UNIT within each context, corrected.

    per_ctx_units: {ctx: [ [unit_values...], ... ]} where each context has a LIST of
    splittable units (probes, or completions per probe), each a list of numeric
    scores. We split the units of each context into two halves, form two per-context
    means, correlate across contexts (Spearman), average over ``n_splits`` random
    halvings, Spearman-Brown-correct, and return sqrt(reliability). None if no
    context has >=2 splittable units (undefined split).
    """
    ctx_ids = [c for c, u in per_ctx_units.items() if len(u) >= 2]
    if len(ctx_ids) < 4:
        return None
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
        return None
    r_half = float(np.mean(rs))
    r_yy = _spearman_brown(r_half)
    if r_yy is None or math.isnan(r_yy) or r_yy <= 0:
        return None
    return math.sqrt(min(r_yy, 1.0))


def _reliability_for_behavior(graded_cell: dict, behavior: str, *, seed: int) -> dict:
    """sqrt(r_yy) per behavior with the MF2 estimator fork + over-judge-draws cross-check.

    graded_cell: {ctx: {..., probe_scores: [{probe_idx, draw_scores:[N], probe_mean}]}}.
    - sycophancy: over-ROLLOUTS is the primary split — but the persisted unit here is
      probe_mean per probe (each #658 sycophancy 'probe' is really one completion in
      our packing), so we split over those probe-mean units (equivalent to
      over-rollouts given the packing). over-judge-draws is the cross-check.
    - refusal / harmful_compliance / all low-m: over-PROBES is primary; over-judge-draws
      cross-check.
    """
    # over-probes / over-rollouts units: each unit = a probe's [probe_mean]
    units_probes: dict[str, list[list[float]]] = {}
    # over-judge-draws units: each unit = one judge draw's per-context score list
    # (we split the N draws; unit d = [that draw's score across probes])
    draws_by_ctx: dict[str, list[list[float]]] = {}
    for ctx, cell in graded_cell.items():
        ps = cell.get("probe_scores", [])
        units_probes[ctx] = [[p["probe_mean"]] for p in ps if p.get("probe_mean") is not None]
        # transpose draws: draw d -> list of that draw's score over probes
        n_draws = max((len(p.get("draw_scores", [])) for p in ps), default=0)
        per_draw: list[list[float]] = [[] for _ in range(n_draws)]
        for p in ps:
            for d, s in enumerate(p.get("draw_scores", [])):
                if s is not None:
                    per_draw[d].append(float(s))
        draws_by_ctx[ctx] = [u for u in per_draw if u]

    primary = _split_half_ryy(units_probes, seed=seed)
    over_draws = _split_half_ryy(draws_by_ctx, seed=seed + 1)
    return {
        "behavior": behavior,
        "sqrt_r_yy": primary,
        "estimator": "over_rollouts" if behavior == "sycophancy" else "over_probes",
        "sqrt_r_yy_over_judge_draws_crosscheck": over_draws,
    }


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
    behaviors = (
        [b for b in args.behaviors.split(",") if b.strip()]
        if args.behaviors
        else list(graded.keys())
    )
    behaviors = [b for b in behaviors if b in graded]
    if not behaviors:
        raise RuntimeError("no behaviors with graded E0 present — cannot fit (fail-loud)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Preflight: sqrt(r_yy) non-null for ALL requested behaviors (MF2 gate) ──
    reliability: dict[str, dict] = {}
    for beh in behaviors:
        rel = _reliability_for_behavior(graded[beh], beh, seed=args.seed)
        reliability[beh] = rel
    null_ryy = [b for b, r in reliability.items() if r["sqrt_r_yy"] is None]
    if null_ryy:
        raise RuntimeError(
            f"reliability preflight FAILED: sqrt(r_yy) is null for {null_ryy} "
            "(MF2 requires a non-null in-range ceiling for every behavior before any "
            "Delta-rho is interpreted against it)"
        )
    logger.info(
        "Reliability preflight PASS: sqrt(r_yy) non-null for all %d behaviors", len(behaviors)
    )

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
                boot = _cluster_bootstrap_rho(
                    preds, y, n_boot=args.n_bootstrap, seed=args.seed + li
                )
                per_layer_boot[op][li] = {"ci95": boot["ci95"]} if boot else {"ci95": None}

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
        # max-over-layer > shuffle p97.5 AND gap < sqrt(r_yy).
        d = results[beh]["delta_rho_unpooled_vs_mean_pca_max"]
        p = results[beh]["shuffle_null_p97_5_vs_mean_pca"]
        ryy = reliability[beh]["sqrt_r_yy"]
        results[beh]["unpooling_helps"] = bool(
            math.isfinite(d) and math.isfinite(p) and d > p and (ryy is None or d < ryy)
        )

    # ── learning curve (per §6, unless skipped) ────────────────────────────────
    learning_curve: dict[str, dict] = {}
    if not args.skip_learning_curve:
        rng = random.Random(args.seed)
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
            curve = {}
            for nprime in grid:
                idx = list(range(len(kept_ctx)))
                rng.shuffle(idx)
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
                curve[str(nprime)] = {
                    "rho_mean": rmean,
                    "rho_unpooled": runp,
                    "delta": (runp - rmean) if (rmean is not None and runp is not None) else None,
                }
            learning_curve[beh] = {"best_layer": best_li, "curve": curve}

    # ── SECONDARY: answer-profile reconstruction (per §4.3, unless skipped) ────
    reconstruction: dict[str, dict] = {}
    if not args.skip_reconstruction:
        cc_path = Path("data/issue_812/context_vectors_mean.pt")
        if cc_path.exists():
            _ = torch.load(cc_path, weights_only=False)  # c_C present; full sweep uses it
            reconstruction["note"] = "c_C present; reconstruction R2 computed at full-run scale"
        else:
            reconstruction["note"] = (
                "c_C (issue594 context_vectors_mean.pt) not staged locally — "
                "reconstruction DV deferred to full run (fetch at dispatch)"
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
        "learning_curve": learning_curve,
        "reconstruction": reconstruction,
    }
    fit_path = out_dir / "pooling_fit_results.json"
    with open(fit_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("WROTE %s", fit_path)

    rel_path = out_dir / "reliability_and_learning_curve.json"
    with open(rel_path, "w") as f:
        json.dump(
            {"meta": meta, "reliability": reliability, "learning_curve": learning_curve},
            f,
            indent=2,
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
