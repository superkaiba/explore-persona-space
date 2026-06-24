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
    STORE_DIR,
    SUMMARY_RECIPES,
    dump_json,
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
FDR_Q = 0.10  # plan §11


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
    """Context-clustered bootstrap 95% CI of Spearman ρ (resample contexts w/ repl)."""
    n = len(pred)
    if n < 4:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = _rho(pred[idx], meas[idx])
        if r is not None:
            stats.append(r)
    if len(stats) < 100:
        return None
    stats.sort()
    return {"ci95": [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]]}


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
    # A learned attn-pool weight (shared across cells) — a random unit vector is
    # a defensible default for the smoke; a fitted weight is a follow-up.
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
                        "bootstrap": _cluster_bootstrap_rho(pred, y, n_boot=N_BOOTSTRAP, seed=658),
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


def fit_a34_a35(store, e0, ctx_ids, layers, shuffle_seed=658) -> dict:
    """A3.4 ridge + A3.5 MLP: c_C → v0(C) held-out, + the within-context shuffle null.

    c_C here = the mean-over-prompt ablation (the last-input-token c_C is on the
    #594 HF store; the campaign default selection happens downstream). Reports
    the LOCO ρ between predicted and measured v0 (per layer, mean recipe) for
    ridge (A3.4) and MLP (A3.5), the linear-vs-nonlinear gap, and the shuffled
    null (round-1 concern #4: re-pair c_C with another context's v0, re-fit).
    """
    out: dict = {"per_layer": [], "shuffle_null": []}
    cc = store["cc_meanprompt"]  # {ctx_id: (Lc, H)}
    C = np.stack([cc[c].numpy() for c in ctx_ids])  # (N, Lc, H)
    V = np.stack([store["summaries"]["mean"][c].numpy() for c in ctx_ids])  # (N, Lc, H)
    n = len(ctx_ids)
    rng = np.random.default_rng(shuffle_seed)
    for li in range(len(layers)):
        Xc = C[:, li, :]
        Yv = V[:, li, :]
        # ridge M (A3.4): predict the v0 vector, then ρ on the per-context norm
        # (a scalar readout that does not require choosing one output dim).
        ridge_pred = _ridge_predict_loco(Xc, Yv, RIDGE_LAMBDAS)
        mlp_pred = np.stack(
            [_fit_mlp_loco(Xc, Yv[:, k]) for k in range(min(8, Yv.shape[1]))], axis=1
        )
        # scalar readout: cosine of predicted vs measured v0, per context
        ridge_cos = _rowwise_cos(ridge_pred, Yv)
        mlp_cos = _rowwise_cos(mlp_pred, Yv[:, : mlp_pred.shape[1]])
        out["per_layer"].append(
            {
                "layer": layers[li],
                "ridge_mean_cos": float(np.mean(ridge_cos)),
                "mlp_mean_cos": float(np.mean(mlp_cos)),
                "nonlinear_gap": float(np.mean(mlp_cos) - np.mean(ridge_cos)),
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
    return out


def _rowwise_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


# ── N1 — noise floor ────────────────────────────────────────────────────────


def noise_floor(spans_dir: Path, e0, ctx_ids, n_redraws=N_NOISE_REDRAWS, seed=658) -> dict:
    """Test-retest ρ ceiling: re-estimate E0 from independent probe redraws.

    The base read here is the per-(C) mean answer-span norm (a cheap E0-proxy
    that the cached spans support directly); the test-retest ρ across redraws is
    the per-behavior reliability ceiling. The PASS bar (A1) = the 95th pct of
    this distribution. (The 48-probe pool is small, so the floor is conservative
    — plan §8.)
    """
    rng = random.Random(seed)
    # per-context per-probe scalar: mean activation norm of the answer span at
    # the first capture layer (the proxy whose test-retest the floor measures).
    per_ctx_probe: dict[str, list[float]] = {}
    for c in ctx_ids:
        blob = torch.load(spans_dir / f"{c}.pt", weights_only=False)
        vals = []
        for s in blob["spans"]:
            if s is not None:
                vals.append(float(s[0].float().norm(dim=-1).mean().item()))
        per_ctx_probe[c] = vals
    rhos = []
    for _ in range(n_redraws):
        a, b = [], []
        for c in ctx_ids:
            vals = per_ctx_probe[c]
            if len(vals) < 2:
                continue
            half = len(vals) // 2
            shuf = vals[:]
            rng.shuffle(shuf)
            a.append(float(np.mean(shuf[:half])))
            b.append(float(np.mean(shuf[half:])))
        r = _rho(np.array(a), np.array(b)) if len(a) >= 4 else None
        if r is not None:
            rhos.append(r)
    p95 = float(np.percentile(rhos, 95)) if rhos else None
    # one shared floor scalar (the proxy is behavior-agnostic by construction);
    # surfaced per-column in the verdict table.
    return {col: p95 for col in e0["columns"]} | {"_distribution": rhos, "_p95": p95}


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
    return {
        "a32_verdicts": verdicts,
        "a33_verdicts": a33_verdict,
        "a34_a35": a34_35,
        "noise_floor": {"p95": noise.get("_p95"), "distribution": noise.get("_distribution")},
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
    # Hero candidate 2: A3.4/A3.5 linear-vs-MLP cos scatter.
    pl = agg["a34_a35"].get("per_layer", [])
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
    args = parser.parse_args()

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

    # N1 + baselines first (the verdict gates).
    noise = noise_floor(spans_dir, e0)
    base_prior = base_prior_baseline(e0, ctx_ids)
    sigma_sanity = sigma_covariance_sanity(store_dir, e0)

    a32_cells = fit_a32(store, spans_dir, e0, ctx_ids, layers, recipes, noise, base_prior)
    dump_json({"a32": a32_cells}, out_dir / "a32_cells.json")  # checkpoint-per-phase

    a33_cells = fit_a33(store, rb, e0, ctx_ids, layers)
    dump_json({"a33": a33_cells}, out_dir / "a33_cells.json")

    a34_35 = fit_a34_a35(store, e0, ctx_ids, layers)
    dump_json(a34_35, out_dir / "a34_a35.json")

    agg = aggregate(a32_cells, a33_cells, a34_35, noise, base_prior, sigma_sanity, e0)
    agg["dual_dv_validation"] = dual_dv_validation(e0)

    # locked recipe: per-behavior best (layer, summary) — the campaign deliverable.
    locked = {
        col: {"layer": v.get("best_layer"), "summary": v.get("best_summary")}
        for col, v in agg["a32_verdicts"].items()
        if v.get("a32_pass")
    }
    dump_json(
        {"locked_recipe": locked, "selected_on": "A3.2 best-layer/summary, FDR-gated"},
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
