#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ρ, →, θ, Σ, ×, λ) in scientific docstrings + log messages.
"""Issue #658 P1-P3 / N1 / A1 (off-pod CPU): A3.2-A3.5 predictor fits + stats.

Reads the base-model activation store (``v0_summaries.pt``, ``r_b.pt``,
``sigma_c.pt``, per-(C,probe) answer spans) + the E0(C,B) measurement table
(``E0_expression.json``) and produces the campaign deliverables:

- **P1 — A3.2** (``a32_mlp``): per behavior B, per layer ℓ, per summary recipe,
  fit a small MLP ``v0(C)[ℓ] → E0(C,B)`` under leave-one-context-out (LOCO) CV;
  Spearman ρ(pred, measured) vs predict-mean + base-prior + the N1 noise floor.
- **P2 — A3.3** (``a33_linear``): fit r_B (diff-in-means / mean-D_B), test
  ``E0 ≈ r_B^T v0(C)`` on held-out C; linear ρ vs the A3.2 MLP ceiling. ONLY the
  rb_columns() that have a natural diff-in-means contrast (marker / format_style
  / deception / fact / self_report / persona_drift are DROPPED from A3.3 — the
  round-1 r_B-construction concern; A3.2 still carries them).
- **P3 — A3.4/A3.5** (``a34_ridge`` / ``a35_mlp``): ridge M (λ nested-CV) + MLP,
  ``c_C → v0(C)`` held-out (LOCO); the linear-vs-nonlinear gap + the
  ``r_B^T M c_C → E0`` chain ρ; the within-context shuffle null (round-1
  concern #4) at near-zero CPU cost.
- **N1 — noise floor**: 8 independent 48-probe redraws of the per-(C,probe)
  answer spans → test-retest ρ distribution; PASS bar = 95th pct.
- **A1 — aggregate**: per-behavior best (layer, summary) + the PASS/FAIL verdict
  table, FDR q=0.10 over the layer×summary×behavior grid, the dual-DV rate-vs-
  logP validation Spearman, the Σ_c-vs-battery covariance sanity (round-1
  concern #5), + the over-produced figure set.

GPU-FREE: deterministic linear algebra over the cached store + sklearn/torch
CPU regressions. The MLP / ridge hyperparameters are ``ungrounded — needs
smoke-test`` (plan §11/§12); the held-out-CV ρ is the ONLY reported number
(never train ρ) — that is the guard against the over-parameterized fit.

Usage::

    uv run python scripts/issue658_fit_predictors.py \\
        --store data/issue_658/store --e0 eval_results/issue_658/E0_expression.json \\
        --out-dir eval_results/issue_658

    uv run python scripts/issue658_fit_predictors.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue658_common import (  # noqa: E402
    EVAL_RESULTS_DIR,
    RB_RECIPES,
    STORE_DIR,
    SUMMARY_RECIPES,
    dump_json,
    load_cc_last_store,
    load_json,
    summarize_answer_span,
)
from scipy.stats import pearsonr, spearmanr  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue658_fit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Predictor hyperparameters — `ungrounded — needs smoke-test` (plan §11/§12).
# The smoke run gates them; held-out-CV ρ is the only reported number.
MLP_HIDDEN = 512
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_MAX_EPOCHS = 300
RIDGE_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]  # A3.4 nested-CV grid
N_NOISE_REDRAWS = 8  # N1 (plan §11)
N_BOOTSTRAP = 2000  # plan §11
# Retry cap for _cluster_bootstrap_rho: keep redrawing past degenerate (all-equal)
# resamples until n_boot VALID ρ draws accumulate, bounded at this multiple of
# n_boot total attempts so a near-degenerate cell raises instead of looping.
_MAX_BOOTSTRAP_DRAWS = 5
FDR_Q = 0.10  # plan §11
# SMOKE-ONLY A3.4/A3.5 feature-dim clamp: the c_C → v0 ridge is O(D³) in the
# hidden dim D, intractable at full H=3584 on CPU at smoke scale. 0 in the real
# run (full H); a small leading-dim slice in smoke exercises both c_C recipes +
# chain ρ + recipe-selection end-to-end. NOT a production knob.
SMOKE_A34_FEAT_DIM = 128
# A3.5 linear-vs-nonlinear shared target dimensionality. The MLP predicts ONE
# output dim per fit (N folds × MLP_MAX_EPOCHS), so the full-H=3584 target is
# intractable on CPU (3584 MLP fits × layers × recipes). The `nonlinear_gap`
# must compare like-for-like, so BOTH the ridge-cos and MLP-cos that feed the
# gap are read over the SAME leading `A35_MLP_TARGET_DIM` v0 dims — a NAMED
# shared dim reduction (round-2 Major a35-mlp-dim-truncated: the old
# `min(8, ...)` compared an 8-dim MLP cos to a full-dim ridge cos). A3.4's
# full-dim `ridge_mean_cos` (the recipe-lock + chain-ρ statistic) is UNCHANGED.
A35_MLP_TARGET_DIM = 64


# ── E0 target extraction ──────────────────────────────────────────────────────


def e0_target(e0: dict, column_id: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """The per-context E0 scalar for one behavior column (PRIMARY rate / marker logp).

    Returns (values, kept_ctx_ids) over the contexts that have a non-None value.
    """
    vals: list[float] = []
    kept: list[str] = []
    for c in ctx_ids:
        cell = e0.get("e0", {}).get(c, {}).get(column_id)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            v = cell.get("logp_mean")  # marker column
        if v is None:
            continue
        vals.append(float(v))
        kept.append(c)
    return np.array(vals, dtype=np.float64), kept


# ── small torch MLP (1 hidden layer) ──────────────────────────────────────────


class _MLP(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int = MLP_HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_mlp_loco(X: np.ndarray, y: np.ndarray, seed: int = 658) -> np.ndarray:
    """Leave-one-context-out MLP predictions (held-out ρ guard against overfit).

    X (N, D), y (N,). Returns held-out predictions (N,) — one per LOCO fold.
    The MLP is the A3.2/A3.5 universal-function-approximator upper bound; only
    the held-out prediction is reported (never train ρ).
    """
    torch.manual_seed(seed)
    n, d = X.shape
    preds = np.zeros(n, dtype=np.float64)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        mu, sd = Xt[mask].mean(0), Xt[mask].std(0) + 1e-6
        net = _MLP(d)
        opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        xn = (Xt[mask] - mu) / sd
        for _ in range(MLP_MAX_EPOCHS):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xn), yt[mask])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            preds[i] = float(net(((Xt[i] - mu) / sd).unsqueeze(0)).item())
    return preds


def _ridge_predict_loco(X: np.ndarray, Y: np.ndarray, lambdas: list[float]) -> np.ndarray:
    """LOCO ridge predictions of a multi-output target Y (N, P) from X (N, D).

    Nested-CV λ: for each held-out context, pick λ minimizing inner-LOO MSE on
    the training contexts (no λ leakage into the held-out read). Returns
    predictions (N, P).
    """
    n = X.shape[0]
    p = Y.shape[1]
    preds = np.zeros((n, p), dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr, Ytr = X[tr], Y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
        # inner LOO to pick λ
        best_lam, best_mse = lambdas[0], np.inf
        for lam in lambdas:
            errs = []
            for k in range(len(tr)):
                inner = [m for m in range(len(tr)) if m != k]
                w = _ridge_solve(Xtr_n[inner], Ytr[inner], lam)
                pred_k = Xtr_n[k] @ w
                errs.append(float(np.mean((pred_k - Ytr[k]) ** 2)))
            mse = float(np.mean(errs)) if errs else np.inf
            if mse < best_mse:
                best_mse, best_lam = mse, lam
        w = _ridge_solve(Xtr_n, Ytr, best_lam)
        preds[i] = ((X[i] - mu) / sd) @ w
    return preds


def _ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge weights (D, P) for X (N, D) -> Y (N, P)."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


# ── ρ + cluster bootstrap ─────────────────────────────────────────────────────


def _rho(pred: np.ndarray, meas: np.ndarray) -> float | None:
    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


def _pearson(pred: np.ndarray, meas: np.ndarray) -> float | None:
    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = pearsonr(pred, meas)
    return None if np.isnan(r) else float(r)


def _cluster_bootstrap_rho(pred, meas, *, n_boot: int, seed: int) -> dict | None:
    """Context-clustered bootstrap 95% CI of Spearman ρ (resample contexts w/ repl).

    Returns ``{"ci95": [lo, hi], "draws": [...]}`` with ``len(draws) == n_boot``
    for any cell with n>=4 and a real rank signal — ``draws`` is the full sorted
    list of per-resample ρ values (the v3 (G1) genre-delta read consumes these
    per-arm draws to form the INDEPENDENT Δρ CI; the two arms have disjoint probes
    so no paired resampling is possible). ``draws`` is an ADDITIVE key: ``ci95`` is
    unchanged, so the Betley arm's existing numbers are untouched.

    Degenerate resamples (an all-equal redraw → ``_rho`` None) are DROPPED and
    RE-DRAWN — we keep drawing (capped at ``_MAX_BOOTSTRAP_DRAWS`` × ``n_boot``
    attempts) until exactly ``n_boot`` valid ρ draws are accumulated, so the
    emitted ``draws`` length is the registered ≥2000 production resample count
    (plan v3 §6/§6.5/§11) and never silently degrades to <n_boot. The downstream
    Δρ CI gate (``issue658_genre_delta._delta_rho_ci``) enforces the ≥2000 floor;
    this keeps healthy cells from tripping it on degenerate-resample drops.

    Returns ``None`` ONLY for a genuinely tiny cell (n<4) — the legitimate
    H3-adjacent / no-dynamic-range case the genre-delta gate flags as N/A. A cell
    that is n>=4 but cannot accumulate ``n_boot`` valid draws within the retry cap
    (a near-degenerate measurement with almost no rank variation) raises rather
    than silently emitting a short / None ``draws`` (a dynamic-range cell with
    n>=4 must carry a full bootstrap per the contract).
    """
    n = len(pred)
    if n < 4:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    max_attempts = _MAX_BOOTSTRAP_DRAWS * n_boot
    attempts = 0
    while len(stats) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = [rng.randrange(n) for _ in range(n)]
        r = _rho(pred[idx], meas[idx])
        if r is not None:
            stats.append(r)
    if len(stats) < n_boot:
        raise RuntimeError(
            "cluster bootstrap could not accumulate the registered "
            f"n_boot={n_boot} valid Spearman-ρ draws for an n={n} cell after "
            f"{attempts} resample attempts (got {len(stats)} valid draws); the "
            "measurement has almost no rank variation under resampling. A "
            "dynamic-range cell (n>=4) must carry a full ≥2000-resample bootstrap "
            "per plan v3 §6/§6.5/§11 — do not silently emit a short/None draws "
            "list. Investigate the cell's testable variance (it may belong in the "
            "H3 no-dynamic-range bucket, in which case its E0 std must fall below "
            "the dynamic-range floor and the genre-delta gate will mark it N/A)."
        )
    stats.sort()
    return {
        "ci95": [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]],
        "draws": stats,
    }


# ── A3.2 (P1) ──────────────────────────────────────────────────────────────────


def _summary_matrix(store: dict, recipe: str, layer_idx: int, ctx_ids: list[str]) -> np.ndarray:
    """(N, H) v0 summary matrix for one recipe + capture-layer index over ctx_ids.

    mean/last/maxp are precomputed in v0_summaries.pt; attn is fit on the CPU
    side here from the per-(C,probe) answer spans.
    """
    summ = store["summaries"][recipe]  # {ctx_id: (Lc, H) fp32}
    rows = [summ[c][layer_idx].numpy() for c in ctx_ids]
    return np.stack(rows)


def _attn_matrix(
    spans_dir: Path, layer_idx: int, ctx_ids: list[str], capture_layers, attn_w
) -> np.ndarray:
    """(N, H) attn-pool v0 summary: probe-mean of softmax-weighted answer spans."""
    rows = []
    for c in ctx_ids:
        blob = torch.load(spans_dir / f"{c}.pt", weights_only=False)
        spans = blob["spans"]  # list of (Lc, S, H) fp16 (or None)
        per_probe = [
            summarize_answer_span(s[layer_idx], "attn", attn_weight=attn_w)
            for s in spans
            if s is not None
        ]
        rows.append(torch.stack(per_probe).mean(0).numpy())
    return np.stack(rows)


def fit_a32(store, spans_dir, e0, ctx_ids, layers, recipes, noise_floor, base_prior) -> list[dict]:
    """A3.2: per (behavior, layer, summary) LOCO MLP ρ vs baselines + noise floor."""
    cells: list[dict] = []
    columns = [c for c in e0["columns"]]
    # attn_w is an UNFITTED random unit vector (carried CONCERN
    # attn-pool-weight-unfitted): the `attn` recipe is a RANDOM-PROJECTION CONTROL,
    # NOT a learned attention pool. Documented decision (round 2): relabel rather
    # than fit (attn is plan §9 descope-priority-2; the analyzer adjudicates). The
    # locked_recipe.json `attn_summary_label` + each attn cell's
    # `is_random_projection_control` flag carry this so a winning attn cell is
    # never read as a fitted pool. Seeded for determinism.
    torch.manual_seed(658)
    attn_w = torch.randn(store["summaries"]["mean"][ctx_ids[0]].shape[-1])
    attn_w = attn_w / attn_w.norm()
    for col in columns:
        y, kept = e0_target(e0, col, ctx_ids)
        if len(kept) < 4:
            cells.append({"column": col, "status": "too_few_contexts", "n": len(kept)})
            continue
        for recipe in recipes:
            for li in range(len(layers)):
                if recipe == "attn":
                    X = _attn_matrix(spans_dir, li, kept, store["capture_layers"], attn_w)
                else:
                    X = _summary_matrix(store, recipe, li, kept)
                pred = _fit_mlp_loco(X, y)
                rho = _rho(pred, y)
                mean_pred = np.full_like(y, y.mean())
                rho_mean = _rho(mean_pred, y)  # predict-mean baseline (constant -> ~None)
                cells.append(
                    {
                        "column": col,
                        "recipe": recipe,
                        "layer": layers[li],
                        "n": len(kept),
                        "rho": rho,
                        "pearson": _pearson(pred, y),
                        "rho_predict_mean": rho_mean,
                        "rho_base_prior": base_prior.get(col),
                        "noise_floor_p95": noise_floor.get(col),
                        # Skip the bootstrap when the cell has no rank signal:
                        # `_rho` returns None for a constant-y FLOORED cell
                        # (broad_em / harmful_compliance / refusal on UltraChat),
                        # where EVERY resample is degenerate so
                        # `_cluster_bootstrap_rho` would (correctly) RAISE on 0
                        # valid draws. The plan-anticipated H3 graceful path
                        # (plan v3 §6 floor guard / §3 Risk 1 / H3 line 126) needs
                        # `bootstrap: None` emitted, NOT a crash — the round-2
                        # raise contract on `_cluster_bootstrap_rho` still holds
                        # for genuinely near-degenerate n>=4 cells; we just never
                        # call it when there is nothing to bootstrap.
                        "bootstrap": (
                            None
                            if rho is None
                            else _cluster_bootstrap_rho(pred, y, n_boot=N_BOOTSTRAP, seed=658)
                        ),
                        # attn is a RANDOM-PROJECTION CONTROL, not a learned pool
                        # (carried CONCERN attn-pool-weight-unfitted).
                        "is_random_projection_control": recipe == "attn",
                    }
                )
    return cells


# ── A3.3 (P2) — linear r_B readout ────────────────────────────────────────────


def fit_a33(store, rb, e0, ctx_ids, layers) -> list[dict]:
    """A3.3: E0 ≈ r_B^T v0(C), per layer × recipe, over the rb_columns only."""
    cells: list[dict] = []
    for col in rb.get("columns", []):
        y, kept = e0_target(e0, col, ctx_ids)
        if len(kept) < 4 or col not in rb["r_b"]:
            continue
        for rb_recipe in ("diffmeans", "meanDB"):
            rdir = rb["r_b"][col].get(rb_recipe)
            if rdir is None:
                continue
            for li in range(len(layers)):
                X = _summary_matrix(store, "mean", li, kept)  # v0 mean recipe (theory default)
                r = rdir[li].numpy()  # (H,)
                pred = X @ r
                cells.append(
                    {
                        "column": col,
                        "rb_recipe": rb_recipe,
                        "layer": layers[li],
                        "n": len(kept),
                        "rho": _rho(pred, y),
                        "pearson": _pearson(pred, y),
                    }
                )
    return cells


# ── A3.4 / A3.5 (P3) — c_C -> v0(C) ────────────────────────────────────────────


def _fit_a34_a35_one_recipe(
    cc_map, store, e0, rb, ctx_ids, layers, shuffle_seed, feat_dim=0
) -> dict:
    """A3.4 ridge + A3.5 MLP for ONE c_C recipe: c_C → v0(C) held-out.

    cc_map = {ctx_id: (Lc, H)} for this c_C recipe. Reports the LOCO ρ between
    predicted and measured v0 (per layer, mean recipe) for ridge (A3.4) and MLP
    (A3.5), the linear-vs-nonlinear gap, the within-context shuffle null
    (round-1 concern #4), AND the downstream ``r_B^T M c_C → E0`` chain ρ per
    behavior (Codex Major + reconciler "Observed but not raised" — the chain ρ
    promised in this function's docstring was absent in round 1).

    ``feat_dim`` > 0 truncates the c_C / v0 / r_B feature dimension to the leading
    ``feat_dim`` dims — a SMOKE-ONLY clamp so the O(D³) ridge solve over the full
    H=3584 hidden is tractable on CPU at smoke scale; the real run uses the full
    H (feat_dim=0). It exercises both recipes + chain ρ + recipe-selection
    end-to-end without changing the production code path.
    """
    out: dict = {"per_layer": [], "shuffle_null": [], "chain_rho_e0": {}}
    C = np.stack([np.asarray(cc_map[c]) for c in ctx_ids])  # (N, Lc, H)
    V = np.stack([store["summaries"]["mean"][c].numpy() for c in ctx_ids])  # (N, Lc, H)
    if feat_dim:
        C = C[:, :, :feat_dim]
        V = V[:, :, :feat_dim]
    n = len(ctx_ids)
    rng = np.random.default_rng(shuffle_seed)
    # Cache the per-layer LOCO ridge prediction of v0 so the chain ρ can reuse it.
    ridge_pred_v0_by_layer: dict[int, np.ndarray] = {}
    for li in range(len(layers)):
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        # ridge M (A3.4): predict the FULL v0 vector, then ρ on the per-context
        # cosine (a scalar readout that does not require choosing one output dim).
        # ridge_mean_cos stays FULL-dim — it feeds the recipe lock + chain ρ.
        ridge_pred = _ridge_predict_loco(Xc, Yv, RIDGE_LAMBDAS)
        ridge_pred_v0_by_layer[li] = ridge_pred
        ridge_cos = _rowwise_cos(ridge_pred, Yv)
        # A3.5 linear-vs-nonlinear gap: read BOTH methods over the SAME leading
        # `A35_MLP_TARGET_DIM` v0 dims (the named shared reduction) so the gap is
        # like-for-like (round-2 Major a35-mlp-dim-truncated). The MLP fits one
        # output dim at a time, so it is the dim-bound method; the ridge cos for
        # the gap is recomputed over the same slice (NOT the full-dim ridge_cos).
        gap_dim = min(A35_MLP_TARGET_DIM, Yv.shape[1])
        mlp_pred = np.stack([_fit_mlp_loco(Xc, Yv[:, k]) for k in range(gap_dim)], axis=1)
        mlp_cos = _rowwise_cos(mlp_pred, Yv[:, :gap_dim])
        ridge_cos_gap = _rowwise_cos(ridge_pred[:, :gap_dim], Yv[:, :gap_dim])
        out["per_layer"].append(
            {
                "layer": layers[li],
                "ridge_mean_cos": float(np.mean(ridge_cos)),  # A3.4, full-dim
                "mlp_mean_cos": float(np.mean(mlp_cos)),  # over gap_dim
                # gap = MLP vs ridge BOTH read over gap_dim (like-for-like).
                "nonlinear_gap": float(np.mean(mlp_cos) - np.mean(ridge_cos_gap)),
                "ridge_mean_cos_on_gap_dim": float(np.mean(ridge_cos_gap)),
                "gap_target_dim": gap_dim,
            }
        )
        # shuffle null: permute the v0 rows, re-fit ridge, report cos.
        perm = rng.permutation(n)
        ridge_pred_sh = _ridge_predict_loco(Xc, Yv[perm], RIDGE_LAMBDAS)
        out["shuffle_null"].append(
            {
                "layer": layers[li],
                "ridge_mean_cos_shuffled": float(np.mean(_rowwise_cos(ridge_pred_sh, Yv[perm]))),
            }
        )
    # Chain ρ: project the LOCO-predicted v0 through each behavior's r_B and
    # Spearman-correlate against the measured E0 — the full shortcut
    # r_B^T (M c_C) → E0(C,B). Best layer per behavior is reported.
    rb_dirs = (rb or {}).get("r_b", {})
    for col in (rb or {}).get("columns", []):
        if col not in rb_dirs:
            continue
        y, kept = e0_target(e0, col, ctx_ids)
        if len(kept) < 4:
            continue
        kept_idx = [ctx_ids.index(c) for c in kept]
        rdir = rb_dirs[col].get("diffmeans")
        if rdir is None:
            continue
        best = None
        for li in range(len(layers)):
            r = np.asarray(rdir[li])  # (H,)
            if feat_dim:
                r = r[:feat_dim]  # match the smoke-clamped predicted-v0 dim
            pred_v0 = ridge_pred_v0_by_layer[li][kept_idx]  # (n_kept, H or feat_dim)
            chain_pred = pred_v0 @ r
            rho = _rho(chain_pred, y)
            if rho is not None and (best is None or rho > best["rho"]):
                best = {"layer": layers[li], "rho": rho}
        if best is not None:
            out["chain_rho_e0"][col] = best
    return out


def fit_a34_a35(store, cc_recipes, e0, rb, ctx_ids, layers, shuffle_seed=658, feat_dim=0) -> dict:
    """A3.4/A3.5 over BOTH c_C recipes (round-2 BLOCKER fix) + recipe selection.

    ``cc_recipes`` = {recipe_name: {ctx_id: (Lc, H)}} for each c_C recipe — the
    #594-reused last-input-token store ("last") AND the #658-extracted
    mean-over-prompt ablation ("meanprompt"). Round-1 evaluated ONLY meanprompt,
    so the campaign could not lock the c_C recipe (Phase-2 deliverable). Here we
    fit both under the IDENTICAL LOCO protocol and apply the plan §4.3-P3 rule:
    default to **last-input-token** UNLESS mean-over-prompt wins by > the
    noise-floor margin (encoded into ``recipe_selection``; the locked_recipe.json
    write reads it). ``feat_dim`` > 0 is the SMOKE-ONLY hidden-dim clamp (real run
    = 0 = full H).
    """
    by_recipe: dict[str, dict] = {}
    for name, cc_map in cc_recipes.items():
        by_recipe[name] = _fit_a34_a35_one_recipe(
            cc_map, store, e0, rb, ctx_ids, layers, shuffle_seed, feat_dim=feat_dim
        )

    # Recipe selection: compare the best mean ridge-cos (the linear M fidelity)
    # across recipes; default to last-input-token unless meanprompt wins by margin.
    def _best_cos(rec: dict) -> float:
        return max((p["ridge_mean_cos"] for p in rec["per_layer"]), default=float("-inf"))

    selection = _select_cc_recipe(by_recipe, _best_cos)
    return {"by_recipe": by_recipe, "recipe_selection": selection}


def _select_cc_recipe(by_recipe: dict, best_cos_fn) -> dict:
    """Plan §4.3-P3 c_C recipe-lock rule: default last-input-token unless beaten.

    Default to ``last`` (the #594-wired, store-reused recipe Phase 2 inherits)
    UNLESS ``meanprompt`` wins the best-layer ridge-cos by more than a small
    margin. The chosen recipe is the campaign default carried into Phase 2.
    """
    margin = 0.02  # ridge-cos win margin (a small, ungrounded screening tolerance)
    last_cos = best_cos_fn(by_recipe["last"]) if "last" in by_recipe else float("-inf")
    mean_cos = best_cos_fn(by_recipe["meanprompt"]) if "meanprompt" in by_recipe else float("-inf")
    if "last" not in by_recipe:
        chosen = "meanprompt"
        reason = "last-input-token recipe unavailable; defaulting to mean-over-prompt"
    elif mean_cos > last_cos + margin:
        chosen = "meanprompt"
        reason = (
            f"mean-over-prompt best ridge-cos {mean_cos:.4f} beats last-input-token "
            f"{last_cos:.4f} by > {margin} margin"
        )
    else:
        chosen = "last"
        reason = (
            f"default last-input-token (#594-wired); best ridge-cos last={last_cos:.4f} "
            f"vs meanprompt={mean_cos:.4f} (within {margin} margin)"
        )
    return {
        "chosen_cc_recipe": chosen,
        "reason": reason,
        "last_best_ridge_cos": None if last_cos == float("-inf") else last_cos,
        "meanprompt_best_ridge_cos": None if mean_cos == float("-inf") else mean_cos,
        "margin": margin,
    }


def _rowwise_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


# ── N1 — noise floor ────────────────────────────────────────────────────────


def noise_floor(e0, ctx_ids, n_redraws=N_NOISE_REDRAWS, seed=658) -> dict:
    """Test-retest ρ ceiling on the E0 TARGET itself, PER BEHAVIOR (round-2 fix).

    Re-estimates ``E0(C,B)`` — the predictor TARGET (judged rate / marker logp),
    NOT the predictor INPUT (answer-span activation norm; the round-1 BLOCKER) —
    from independent probe redraws, per behavior column. For each behavior B and
    each redraw, split the per-context probe set into two random halves, average
    the per-probe E0 contributions over each half → two per-context E0 estimates,
    and take their Spearman ρ. The 95th pct of the ``n_redraws`` distribution is
    the per-behavior reliability ceiling — the PASS denominator (A1). The 48-probe
    pool is small (plan §8), so the floor is conservatively wide.

    Returns ``{col: float_or_None for col in e0["columns"]}`` (per-behavior-
    DISTINCT, never a shared broadcast) plus ``_distribution`` / ``_p95`` for the
    pooled report. A behavior whose E0 is degenerate across contexts (a constant
    rate everywhere — the saturation regime §8 risk-1 guards) has NO rank signal
    to predict, so its floor is pinned to 1.0 (impossible to beat) to suppress a
    false PASS, NOT left low. Reads the per-probe E0 the judge phase persisted
    (``e0["e0"][c][col]["per_probe"]``).
    """
    rng = random.Random(seed)
    columns = list(e0["columns"])
    e0_table = e0.get("e0", {})
    floors: dict[str, float | None] = {}
    distributions: dict[str, list[float]] = {}
    for col in columns:
        # per-context: the list of per-probe E0 contributions for this behavior.
        per_ctx_probe: dict[str, list[float]] = {}
        for c in ctx_ids:
            cell = e0_table.get(c, {}).get(col)
            if cell is None:
                continue
            pp = cell.get("per_probe")
            if not pp:
                continue
            vals = [float(x["e0"]) for x in pp if x.get("e0") is not None]
            if vals:
                per_ctx_probe[c] = vals
        # If too few contexts have data, the floor is undefined for this column.
        if len(per_ctx_probe) < 4:
            floors[col] = None
            distributions[col] = []
            continue
        # Degenerate (saturation) guard: a behavior whose per-context E0 estimate
        # is (near-)constant across contexts has no rank signal — pin the floor
        # to 1.0 so no predictor ρ can falsely clear it (§8 risk-1).
        ctx_means = [float(np.mean(v)) for v in per_ctx_probe.values()]
        if float(np.std(ctx_means)) < 1e-9:
            floors[col] = 1.0
            distributions[col] = []
            continue
        rhos: list[float] = []
        for _ in range(n_redraws):
            a, b = [], []
            for c in ctx_ids:
                vals = per_ctx_probe.get(c)
                if not vals or len(vals) < 2:
                    continue
                half = len(vals) // 2
                shuf = vals[:]
                rng.shuffle(shuf)
                a.append(float(np.mean(shuf[:half])))
                b.append(float(np.mean(shuf[half:])))
            r = _rho(np.array(a), np.array(b)) if len(a) >= 4 else None
            if r is not None:
                rhos.append(r)
        floors[col] = float(np.percentile(rhos, 95)) if rhos else None
        distributions[col] = rhos
    pooled = [r for rs in distributions.values() for r in rs]
    return {
        **floors,
        "_distribution": pooled,
        "_p95": float(np.percentile(pooled, 95)) if pooled else None,
        "_per_behavior_distribution": distributions,
    }


# ── base-prior baseline ────────────────────────────────────────────────────────


def base_prior_baseline(e0, ctx_ids) -> dict:
    """ρ of a behavior's GLOBAL base rate (a constant) vs measured E0.

    Round-1 concern #7: the base-prior baseline is the global behavior MEAN — a
    constant — so ρ vs a constant is undefined / ≈0; beating it is trivial. We
    report it as None (a constant predictor has no rank information) and surface
    the caveat in the verdict table so the analyzer does NOT lean on
    'beats base-prior' to rule out the #532/#649 prior-confound (which at θ0 is
    largely N/A — the genuine per-context base propensity IS E0(C,B) itself).
    """
    return {col: None for col in e0["columns"]}


# ── A1 — aggregate + FDR + verdicts + figures ──────────────────────────────────


def benjamini_hochberg(pvals: list[float], q: float) -> list[bool]:
    """BH FDR: returns a reject mask aligned to pvals (True = significant)."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order, 1):
        if pvals[i] <= (rank / m) * q:
            for j in order[:rank]:
                reject[j] = True
    return reject


def _approx_p_from_rho(rho: float | None, n: int) -> float:
    """Two-sided p for a Spearman ρ via the t approximation (screening-grade)."""
    if rho is None or n < 4 or abs(rho) >= 1.0:
        return 1.0
    from scipy.stats import t as student_t

    tstat = rho * np.sqrt((n - 2) / (1 - rho**2))
    return float(2 * student_t.sf(abs(tstat), n - 2))


def aggregate(a32_cells, a33_cells, a34_35, noise, base_prior, sigma_sanity, e0) -> dict:
    """A1: per-behavior best (layer, summary), PASS/FAIL verdicts, FDR, figures-meta."""
    # FDR over the full A3.2 grid.
    scored = [c for c in a32_cells if c.get("rho") is not None]
    pvals = [_approx_p_from_rho(c["rho"], c["n"]) for c in scored]
    reject = benjamini_hochberg(pvals, FDR_Q)
    for c, p, r in zip(scored, pvals, reject, strict=True):
        c["fdr_p"] = p
        c["fdr_reject"] = bool(r)
    # per-behavior best cell + PASS/FAIL.
    verdicts: dict = {}
    columns = e0["columns"]
    for col in columns:
        col_cells = [c for c in scored if c["column"] == col]
        if not col_cells:
            verdicts[col] = {"a32_pass": None, "reason": "no scored cells (low dynamic range?)"}
            continue
        best = max(col_cells, key=lambda c: c["rho"] if c["rho"] is not None else -2)
        floor = noise.get(col)
        # PASS = best ρ > noise-floor p95 AND > predict-mean (a constant -> None,
        # so any positive ρ beats it) AND FDR-significant. base-prior is None
        # (concern #7) so it is NOT a gate (would be trivially passed).
        a32_pass = (
            best["rho"] is not None
            and (floor is None or best["rho"] > floor)
            and best.get("fdr_reject", False)
        )
        verdicts[col] = {
            "a32_pass": bool(a32_pass),
            "best_layer": best["layer"],
            "best_summary": best["recipe"],
            "best_rho": best["rho"],
            "noise_floor_p95": floor,
            "fdr_reject": best.get("fdr_reject"),
        }
    # A3.3 verdict: linear ρ within noise floor of the A3.2 MLP ρ, per rb column.
    a33_verdict: dict = {}
    for col in {c["column"] for c in a33_cells}:
        lin = [c for c in a33_cells if c["column"] == col and c.get("rho") is not None]
        if not lin:
            continue
        best_lin = max(lin, key=lambda c: c["rho"])
        mlp_rho = verdicts.get(col, {}).get("best_rho")
        a33_verdict[col] = {
            "best_linear_rho": best_lin["rho"],
            "best_rb_recipe": best_lin["rb_recipe"],
            "best_layer": best_lin["layer"],
            "mlp_ceiling_rho": mlp_rho,
            "a33_pass": bool(
                mlp_rho is not None
                and best_lin["rho"] is not None
                and best_lin["rho"] >= mlp_rho - (noise.get("_p95") or 0.1)
            ),
        }
    # per-behavior reliability ceilings (round-2 fix: the floor is the re-estimated
    # E0 target's test-retest ρ per behavior, not a shared activation-norm scalar).
    per_behavior_floor = {col: noise.get(col) for col in columns}
    return {
        "a32_verdicts": verdicts,
        "a33_verdicts": a33_verdict,
        "a34_a35": a34_35,
        "noise_floor": {
            "p95": noise.get("_p95"),
            "distribution": noise.get("_distribution"),
            "per_behavior_p95": per_behavior_floor,
            "note": (
                "per-behavior test-retest ρ of the re-estimated E0(C,B) target from "
                f"{N_NOISE_REDRAWS} probe redraws (round-2 fix); a degenerate/saturated "
                "behavior is pinned to 1.0 (no rank signal to beat)"
            ),
        },
        "base_prior_note": (
            "base-prior baseline is the global behavior mean (a constant) — ρ vs a constant is "
            "undefined/≈0, so 'beats base-prior' is trivial and NOT a gate (round-1 concern #7); "
            "at θ0 the genuine per-context base propensity IS E0(C,B) itself"
        ),
        "sigma_sanity": sigma_sanity,
        "fdr_q": FDR_Q,
    }


def sigma_covariance_sanity(store_dir: Path, e0) -> dict:
    """Round-1 concern #5: compare Σ_c (background corpus) vs the battery's own Σ.

    Flags if they differ substantially (Frobenius-normalized distance). Σ_c
    feeds Phase 2-4 only; not load-bearing for A3.2/A3.3 here.
    """
    sigma_path = store_dir / "sigma_c.pt"
    if not sigma_path.exists():
        return {"skipped": "no sigma_c.pt"}
    blob = torch.load(sigma_path, weights_only=False)
    sigma_c = blob["sigma_c"][0].numpy()  # (H, H) first captured layer
    # battery's own second moment from the v0 mean summaries
    v0 = torch.load(store_dir / "v0_summaries.pt", weights_only=False)
    ctx_ids = v0["context_ids"]
    M = np.stack([v0["summaries"]["mean"][c][0].numpy() for c in ctx_ids])  # (N, H)
    sigma_batt = (M.T @ M) / len(ctx_ids)
    fro = float(np.linalg.norm(sigma_c - sigma_batt) / (np.linalg.norm(sigma_c) + 1e-12))
    return {
        "frobenius_rel_diff": fro,
        "substantial": fro > 0.5,
        "note": "Σ_c (background ≥3k) vs battery own-Σ; feeds Phase 2-4 only, not A3.2/A3.3",
    }


def dual_dv_validation(e0) -> dict:
    """Spearman(rate, logp_pos_mean) across cells with dynamic range (plan §6)."""
    rates, logps = [], []
    for ctx in e0.get("e0", {}).values():
        for v in ctx.values():
            if v.get("low_dynamic_range"):
                continue
            if v.get("rate") is not None and v.get("logp_pos_mean") is not None:
                rates.append(v["rate"])
                logps.append(v["logp_pos_mean"])
    if len(rates) < 4:
        return {"spearman": None, "n": len(rates), "note": "too few non-saturated cells"}
    r, _ = spearmanr(rates, logps)
    return {"spearman": None if np.isnan(r) else float(r), "n": len(rates)}


# ── figures (over-produce; analyzer picks the hero) ───────────────────────────


def make_figures(a32_cells, agg, out_dir: Path) -> list[str]:
    """ρ-vs-layer line plots + linear-vs-MLP scatter (plan §6 hero candidates)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    # Hero candidate 1: per-behavior ρ-vs-layer (default mean summary).
    cols = sorted({c["column"] for c in a32_cells if c.get("recipe") == "mean"})
    if cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in cols:
            pts = sorted(
                [
                    c
                    for c in a32_cells
                    if c["column"] == col and c["recipe"] == "mean" and c.get("rho") is not None
                ],
                key=lambda c: c["layer"],
            )
            if pts:
                ax.plot([p["layer"] for p in pts], [p["rho"] for p in pts], marker="o", label=col)
        floor = agg["noise_floor"]["p95"]
        if floor is not None:
            ax.axhline(floor, ls="--", color="gray", label="noise floor p95")
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out Spearman ρ (A3.2 MLP, mean summary)")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        p = fig_dir / "a32_rho_vs_layer.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p))
    # Hero candidate 2: A3.4/A3.5 linear-vs-MLP cos scatter (the chosen c_C recipe,
    # falling back to whichever recipe was evaluated — round-2 nested-by-recipe shape).
    a34 = agg["a34_a35"]
    by_recipe = a34.get("by_recipe", {})
    chosen = a34.get("recipe_selection", {}).get("chosen_cc_recipe")
    rec = by_recipe.get(chosen) or next(iter(by_recipe.values()), {})
    pl = rec.get("per_layer", []) if isinstance(rec, dict) else []
    if pl:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter([p["ridge_mean_cos"] for p in pl], [p["mlp_mean_cos"] for p in pl])
        lo = min(min(p["ridge_mean_cos"], p["mlp_mean_cos"]) for p in pl)
        hi = max(max(p["ridge_mean_cos"], p["mlp_mean_cos"]) for p in pl)
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray")
        ax.set_xlabel("ridge (linear M) mean cos")
        ax.set_ylabel("MLP mean cos")
        fig.tight_layout()
        p = fig_dir / "a34_a35_linear_vs_mlp.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p))
    return made


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #658 P1-P3/N1/A1: predictor fits + stats.")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--e0", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=EVAL_RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--recipes",
        nargs="*",
        default=None,
        help=f"v0 summary recipes to fit (default all: {SUMMARY_RECIPES})",
    )
    parser.add_argument(
        "--no-cc-last",
        action="store_true",
        help="skip the #594 cc_last HF store (offline smoke); evaluate only the "
        "mean-over-prompt c_C recipe. The PRODUCTION recipe lock REQUIRES cc_last.",
    )
    parser.add_argument(
        "--cc-last-from-store",
        action="store_true",
        help="read the last-input-token c_C from the per-genre store "
        "(v0_summaries.pt::cc_last) instead of the Betley-pinned #594 HF loader "
        "(REQUIRED for the (G1) genre arm: the #594 cc_last store is Betley-pinned). "
        "Fail-loud if the store lacks the cc_last key.",
    )
    args = parser.parse_args()

    # SMOKE-ONLY compute clamp: the LOCO MLP (MLP_MAX_EPOCHS=300, per fold per
    # output dim) is intractable on CPU at smoke scale. Clamp the MLP epochs +
    # the ridge λ grid for the smoke ONLY so the predictor pipeline runs
    # end-to-end and returns numbers; the real-run defaults (the §11-grounded
    # values) are untouched. Mutating the module globals is the minimal thread:
    # _fit_mlp_loco / _ridge_predict_loco read them at call time.
    if args.smoke:
        global MLP_MAX_EPOCHS, RIDGE_LAMBDAS, N_BOOTSTRAP
        MLP_MAX_EPOCHS = 25
        RIDGE_LAMBDAS = [1e-1, 1.0, 10.0]
        N_BOOTSTRAP = 200

    store_dir = args.store or (Path(f"{STORE_DIR}_smoke") if args.smoke else STORE_DIR)
    e0_path = args.e0 or (
        EVAL_RESULTS_DIR / ("E0_expression_smoke.json" if args.smoke else "E0_expression.json")
    )
    out_dir = (
        Path(f"{args.out_dir}_smoke")
        if (args.smoke and args.out_dir == EVAL_RESULTS_DIR)
        else args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    recipes = args.recipes or list(SUMMARY_RECIPES)

    store = torch.load(store_dir / "v0_summaries.pt", weights_only=False)
    rb = torch.load(store_dir / "r_b.pt", weights_only=False)
    e0 = load_json(e0_path)
    spans_dir = store_dir / "answer_spans"
    ctx_ids = store["context_ids"]
    layers = store["capture_layers"]
    logger.info("Fitting: %d contexts, %d layers, recipes=%s", len(ctx_ids), len(layers), recipes)

    # N1 + baselines first (the verdict gates). The noise floor re-estimates the
    # per-behavior E0 TARGET (judged rate / marker logp) from probe redraws — the
    # round-2 BLOCKER fix; it no longer reads the answer-span activation norm.
    noise = noise_floor(e0, ctx_ids)
    base_prior = base_prior_baseline(e0, ctx_ids)
    sigma_sanity = sigma_covariance_sanity(store_dir, e0)

    a32_cells = fit_a32(store, spans_dir, e0, ctx_ids, layers, recipes, noise, base_prior)
    dump_json({"a32": a32_cells}, out_dir / "a32_cells.json")  # checkpoint-per-phase

    a33_cells = fit_a33(store, rb, e0, ctx_ids, layers)
    dump_json({"a33": a33_cells}, out_dir / "a33_cells.json")

    # A3.4/A3.5: evaluate BOTH c_C recipes (round-2 BLOCKER fix). last-input-token
    # comes from the #594 HF store (Betley arm, CONFIRMED reuse) OR — for the (G1)
    # genre arm — from the per-genre store's freshly-recomputed cc_last
    # (--cc-last-from-store; the #594 store is Betley-pinned). mean-over-prompt is
    # the #658-extracted ablation stored in v0_summaries.pt. A missing #594 store
    # is FAIL-LOUD (the recipe lock is a Phase-2 deliverable) unless --no-cc-last
    # is set for an offline smoke, in which case only meanprompt is evaluated.
    cc_recipes: dict[str, dict] = {
        "meanprompt": {c: store["cc_meanprompt"][c].numpy() for c in ctx_ids}
    }
    if args.no_cc_last:
        logger.warning(
            "--no-cc-last: evaluating only the mean-over-prompt c_C recipe (offline smoke); "
            "the production recipe lock REQUIRES the cc_last recipe"
        )
    elif args.cc_last_from_store:
        # (G1) genre arm: the last-input-token c_C was recomputed fresh on this
        # genre's pool by the extractor (--cc-recompute-last) into
        # v0_summaries.pt::cc_last. Fail loud if the store lacks the key (a store
        # built WITHOUT --cc-recompute-last cannot satisfy --cc-last-from-store).
        store_cc_last = store.get("cc_last")
        if not store_cc_last:
            raise RuntimeError(
                "--cc-last-from-store: v0_summaries.pt has no cc_last key (re-run the "
                "extractor with --cc-recompute-last for the genre arm)"
            )
        missing = [c for c in ctx_ids if c not in store_cc_last]
        if missing:
            raise RuntimeError(
                f"--cc-last-from-store: store cc_last missing {len(missing)} contexts: "
                f"{missing[:5]}..."
            )
        cc_recipes["last"] = {c: store_cc_last[c].numpy() for c in ctx_ids}
        logger.info("cc_last loaded from per-genre store (%d contexts)", len(cc_recipes["last"]))
    else:
        cc_last = load_cc_last_store(layers, ctx_ids)
        cc_recipes["last"] = {c: cc_last[c].numpy() for c in ctx_ids}
    a34_35 = fit_a34_a35(
        store,
        cc_recipes,
        e0,
        rb,
        ctx_ids,
        layers,
        feat_dim=(SMOKE_A34_FEAT_DIM if args.smoke else 0),
    )
    dump_json(a34_35, out_dir / "a34_a35.json")

    agg = aggregate(a32_cells, a33_cells, a34_35, noise, base_prior, sigma_sanity, e0)
    agg["dual_dv_validation"] = dual_dv_validation(e0)

    # locked recipe: per-behavior best (layer, summary) — the campaign deliverable.
    locked = {
        col: {"layer": v.get("best_layer"), "summary": v.get("best_summary")}
        for col, v in agg["a32_verdicts"].items()
        if v.get("a32_pass")
    }
    # The c_C recipe Phase 2 inherits (round-2 BLOCKER fix): the §4.3-P3 rule —
    # default last-input-token unless mean-over-prompt wins by margin.
    cc_selection = a34_35.get("recipe_selection", {})
    dump_json(
        {
            "locked_recipe": locked,
            "selected_on": "A3.2 best-layer/summary, FDR-gated",
            "cc_recipe_lock": cc_selection,  # Phase-2 inherited c_C recipe
            # r_B recipes A3.3 actually ranks (round-2 CONCERN
            # fewshot-rb-recipe-missing): the plan's few-shot-final recipe is
            # DESCOPED — the A3.3 PASS gate ranks the contrastive recipes only.
            "rb_recipes_scored": list(RB_RECIPES),
            "rb_recipe_descope_note": (
                "few-shot-final r_B descoped for #658; needs a separate few-shot-prompted "
                "capture pass not built here. A3.3 ranks diffmeans + meanDB only."
            ),
            "attn_summary_label": (
                "random-projection control — the attn_w pool weight is an UNFITTED random "
                "unit vector (carried CONCERN attn-pool-weight-unfitted); a winning 'attn' "
                "cell is NOT a learned attention pool. The analyzer must read attn as a "
                "random-projection control, never as a fitted recipe (plan §9 descope-2)."
            ),
        },
        out_dir / "locked_recipe.json",
    )
    dump_json(
        {
            "a32_verdicts": agg["a32_verdicts"],
            "a33_verdicts": agg["a33_verdicts"],
            "kill_criterion": (
                "HALT the campaign if A3.2 OR A3.3 fails above the noise floor for the "
                "well-conditioned behaviors (plan §9 / §14)"
            ),
        },
        out_dir / "assumption_verdicts.json",
    )
    figs = make_figures(a32_cells, agg, out_dir)
    dump_json(
        {**agg, "figures": figs, "metadata": reproducibility_metadata({"script": "issue658_fit"})},
        out_dir / "aggregate.json",
    )
    logger.info(
        "Done: %d A3.2 cells, %d figures, locked %d behaviors",
        len(a32_cells),
        len(figs),
        len(locked),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
