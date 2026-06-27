#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ρ, →, θ, Σ, ×, λ) in scientific docstrings + log messages.
"""Issue #658 P1-P3 / N1 / A1 (off-pod GPU/CPU): A3.2-A3.5 predictor fits + stats.

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
  concern #4) at near-zero compute.
- **N1 — noise floor**: 8 independent 48-probe redraws of the per-(C,probe)
  answer spans → test-retest ρ distribution; PASS bar = 95th pct.
- **A1 — aggregate**: per-behavior best (layer, summary) + the PASS/FAIL verdict
  table, FDR q=0.10 over the layer×summary×behavior grid, the dual-DV rate-vs-
  logP validation Spearman, the Σ_c-vs-battery covariance sanity (round-1
  concern #5), + the over-produced figure set.

GPU-ACCELERATED (recovery-mode performance rewrite, 2026-06-27): the dependent
variables (held-out LOCO Spearman ρ for the A3.4 ridge + A3.5 MLP maps, the
cluster-bootstrap CIs at N_BOOTSTRAP, the λ grid, the 4 recipes, and the chain ρ)
are PRESERVED EXACTLY. Three things changed, none of them the reported numbers:

1. **A3.4 ridge** uses the closed-form (PRESS / hat-matrix) leave-one-out
   identity — mathematically IDENTICAL to the per-fold refit, but it needs one
   eigendecomposition of the N×N Gram per training set instead of a refit per
   (inner fold × λ). The dual/Woodbury form ``w = Xᵀ(XXᵀ+λI)⁻¹Y`` keeps the
   solve in the N=50 ≪ D=3584 row space. Exactness is asserted at smoke time
   (``_assert_ridge_exactness``): the new closed-form LOCO ρ matches the old
   primal-refit LOCO ρ to ≤1e-6 on a synthetic case.
2. **A3.5 MLP** runs on CUDA (``--device``, default cuda with a CPU fallback)
   and the per-(LOCO fold × output-head) MLPs are fit IN PARALLEL with
   ``torch.vmap`` over an ensemble of independent networks — the per-output-dim
   independence the old serial code had (one scalar-output MLP per output dim)
   is preserved exactly, just batched. MLP_HIDDEN / MLP_LR / MLP_WD /
   MLP_MAX_EPOCHS are unchanged.
3. **Checkpoint-per-(recipe, layer) cells + resume** + a progress log line per
   completed (recipe × layer) so a crash never restarts from zero and ETA is
   visible.

The held-out-CV ρ is the ONLY reported number (never train ρ) — that is the guard
against the over-parameterized fit. Multi-GPU sharding (``--shard k/n`` /
``--gpu``) splits the (recipe × layer) work across GPUs.

Usage::

    uv run python scripts/issue658_fit_predictors.py \\
        --store data/issue_658/store --e0 eval_results/issue_658/E0_expression.json \\
        --out-dir eval_results/issue_658 --device cuda

    # multi-GPU fan-out (one process per GPU; shards the recipe×layer work):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue658_fit_predictors.py --shard 0/4 &
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue658_fit_predictors.py --shard 1/4 &
    ...

    uv run python scripts/issue658_fit_predictors.py --smoke           # cuda if present
    uv run python scripts/issue658_fit_predictors.py --smoke --device cpu
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
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
# SMOKE-ONLY A3.4/A3.5 feature-dim clamp: a small leading-dim slice in smoke
# exercises both c_C recipes + chain ρ + recipe-selection end-to-end. 0 in the
# real run (full H). NOT a production knob. (The closed-form dual ridge below is
# tractable at full H even on CPU — the clamp now exists only to keep the smoke
# fast + comparable across --device, NOT because the full-H solve is infeasible.)
SMOKE_A34_FEAT_DIM = 128
# A3.5 linear-vs-nonlinear shared target dimensionality. The MLP predicts ONE
# output dim per fit, so the full-H=3584 target is read over the SAME leading
# `A35_MLP_TARGET_DIM` v0 dims as the ridge cos that feeds the gap — a NAMED
# shared dim reduction (round-2 Major a35-mlp-dim-truncated: the old
# `min(8, ...)` compared an 8-dim MLP cos to a full-dim ridge cos). A3.4's
# full-dim `ridge_mean_cos` (the recipe-lock + chain-ρ statistic) is UNCHANGED.
# Value is the base-commit's intended 64 (the perf rewrite must NOT change the
# reported A3.5 gap dimensionality — `mlp_mean_cos`, `ridge_mean_cos_on_gap_dim`,
# `nonlinear_gap` are all read over these leading dims). 64 heads/fold is
# tractable with the GPU-batched MLP below.
A35_MLP_TARGET_DIM = 64

# Compute device for the GPU-accelerated MLP/ridge ops. Resolved from --device
# (default cuda when available, else cpu) in main() and read at call time by the
# batched fitters. The DV is device-INVARIANT (the smoke asserts cpu==cuda ρ).
DEVICE = "cpu"


def _resolve_device(requested: str) -> str:
    """Resolve the --device request to a concrete torch device string.

    'auto' (default) -> cuda if available else cpu; an explicit 'cuda' that is
    unavailable falls back to cpu with a WARNING (so a CPU-only VM smoke still
    runs the cuda code path on cpu rather than crashing).
    """
    if requested in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "--device cuda requested but no CUDA device is available; falling back to cpu"
        )
        return "cpu"
    return requested


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

    GPU-batched (this rewrite): the N LOCO folds are an ENSEMBLE of N independent
    1-hidden-layer MLPs (each trained on the N-1 train rows of its fold, then read
    on its held-out row). They share NOTHING — disjoint parameters, disjoint
    train sets — so they fit in parallel via ``torch.vmap`` over stacked module
    states on ``DEVICE``. This is the batched form of the old per-fold Python
    loop: same architecture, same AdamW(lr=MLP_LR, wd=MLP_WD), same
    MLP_MAX_EPOCHS, same per-fold train-row standardization, same full-batch MSE
    — only the fold loop is vectorized. The held-out prediction per fold is
    bit-for-bit the same computation, just executed in a batched kernel.
    """
    preds = _fit_mlp_ensemble_loco(
        X.astype(np.float32), y.astype(np.float32)[:, None], target_idx=0, seed=seed
    )  # (N, 1)
    return preds[:, 0].astype(np.float64)


def _fit_mlp_ensemble_loco(
    X: np.ndarray, Y: np.ndarray, target_idx: int | list[int], seed: int = 658
) -> np.ndarray:
    """Batched LOCO MLP for one OR several output dims of Y, on ``DEVICE``.

    X (N, D) fp32, Y (N, P) fp32. ``target_idx`` selects which output columns of Y
    to fit (an int or a list). Returns held-out LOCO predictions of shape
    (N, n_targets) aligned to ``target_idx``.

    Per (output-dim t, LOCO fold i) the old code fit an INDEPENDENT scalar-output
    MLP on rows ``{j != i}`` (standardized on those rows) and read its prediction
    on row i. Here every (t, i) pair is one member of a vmapped ensemble:
    ``E = n_targets * N`` independent nets, each ``_MLP(D)``, each with its own
    AdamW step on its own (N-1)-row train batch. Per-fold train-row standardization
    is applied to the SAME (N, D) inputs the serial code used.

    EQUIVALENCE (verified): the fit is architecture-, optimizer-hyperparameter-,
    epoch-, standardization-, and per-output-independence-IDENTICAL to the serial
    loop. The member random inits reproduce the serial reference's RNG stream:
    the SINGLE-output reference re-seeds once per call (one block of n per-fold
    inits), and the MULTI-output reference (the old gap MLP) re-seeds ONCE PER
    output dim, so every dim reuses the same n per-fold inits — this code draws
    the n per-fold inits once and TILES them across the n_targets blocks (see the
    seeding comment below), reproducing that exactly. The stateless ``base``
    template is built on the META device (no parameter storage, no RNG draw) so it
    does NOT perturb the stream. On CPU the batched-vmap path matches the serial
    loop to <=1e-6 (single-output ~3.6e-7; multi-output bit-identical after the
    tile — see the implementer report + ``test_issue658_fit_predictors_exactness``).
    On CUDA the only residual difference is GPU vs CPU float reduction order (true
    of any GPU port; within MLP optimization noise). Per the brief's guidance the
    independent-head form is kept as default (the per-output independence is the
    semantics that matters) — a shared-trunk multi-output net was NOT substituted.
    The ridge DV (A3.4/A3.5 ridge cos, chain ρ) is additionally exact to machine
    precision (``_assert_ridge_exactness``).
    """
    from torch.func import functional_call, stack_module_state, vmap

    idxs = [target_idx] if isinstance(target_idx, int) else list(target_idx)
    n, d = X.shape
    device = torch.device(DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=torch.float32)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).to(device=device, dtype=torch.float32)

    # Build the E = n_targets * N ensemble of independent _MLP(D) nets.
    #
    # SEEDING — bit-match the serial reseed-PER-DIM reference. The old gap MLP did
    # `[_fit_mlp_loco(Xc, Yv[:, k]) for k in range(gap_dim)]`, and EACH
    # `_fit_mlp_loco` call re-seeds `torch.manual_seed(seed)` at its top — so every
    # output dim k gets the SAME n per-fold inits (RNG stream positions [0..n)).
    # Seeding once for all gap_dim*n members would instead give dim k the inits at
    # [k*n..(k+1)*n), diverging for k>=1. So we draw only the n per-fold inits ONCE
    # (seed → n nets, fold order) and TILE that block across the n_targets output
    # dims, reproducing the per-dim reseed exactly while keeping a single vmap
    # kernel over all E members (the fold-batching is the dominant speedup). The
    # stateless `base` template is built on the META device (no parameter storage,
    # no RNG draw) so it does NOT perturb the stream — it only carries the module
    # structure for functional_call.
    n_targets = len(idxs)
    n_members = n_targets * n
    base = _MLP(d).to(device="meta")
    torch.manual_seed(seed)
    block_members = [_MLP(d).to(device) for _ in range(n)]  # the n per-fold inits
    block_params, block_buffers = stack_module_state(block_members)  # leaves (n, ...)

    # Tile the n-member block across the n_targets output-dim blocks → (E, ...),
    # so member m = t_block*N + fold_i has the SAME init as block member fold_i
    # for every t_block (= the serial per-dim reseed). `.repeat` along dim 0 with
    # 1s elsewhere; `.contiguous()` so stack_module_state's downstream ops are happy.
    def _tile(t: torch.Tensor) -> torch.Tensor:
        # detach + clone so the tiled tensor is a fresh LEAF (the optimizer below
        # requires leaf params; `.repeat` would otherwise carry grad history).
        return t.detach().repeat((n_targets,) + (1,) * (t.dim() - 1)).clone()

    params = {k: _tile(v) for k, v in block_params.items()}
    buffers = {k: _tile(v) for k, v in block_buffers.items()}

    # Per-member train mask (leave-one-out within each output-dim block) + the
    # per-member standardized train inputs / targets. member index m =
    # t_block * N + fold_i. The held-out row of member m is fold_i; its train rows
    # are all rows != fold_i.
    fold_ids = torch.arange(n, device=device)
    # train mask (E, N): True for rows used in training member m
    member_fold = torch.cat([fold_ids for _ in idxs])  # (E,)
    train_mask = torch.ones((n_members, n), dtype=torch.bool, device=device)
    train_mask[torch.arange(n_members, device=device), member_fold] = False  # drop held-out row

    # Per-member feature standardization computed on that member's train rows ONLY
    # (no leakage), matching the serial `mu, sd = Xt[mask].mean(0), Xt[mask].std(0)+1e-6`.
    # mask (E, N) -> broadcast over D.
    mask_f = train_mask.to(torch.float32)  # (E, N)
    counts = mask_f.sum(1, keepdim=True)  # (E, 1) == N-1
    # mean over train rows: (E, D)
    sum_x = mask_f @ Xt  # (E, N)@(N, D) = (E, D)
    mu = sum_x / counts
    # std over train rows (population std, matching torch.std default unbiased=True):
    # torch.std uses Bessel correction (N-1). Reproduce exactly: var = sum((x-mu)^2
    # over train rows) / (n_train - 1).
    # (E, N, D) is large; compute via sum of squares to stay memory-light.
    sumsq_x = mask_f @ (Xt * Xt)  # (E, D)  sum of x^2 over train rows
    var = (sumsq_x - counts * mu * mu) / (counts - 1.0).clamp(min=1.0)
    sd = var.clamp(min=0.0).sqrt() + 1e-6  # (E, D)  matches serial +1e-6

    # Per-member targets: the t-th output column of Y, broadcast across that
    # block's N folds.
    target_cols = torch.tensor(idxs, device=device)  # (n_targets,)
    Yt_blocks = Yt[:, target_cols].t()  # (n_targets, N)
    y_member = Yt_blocks.repeat_interleave(n, dim=0)  # (E, N) member -> its output col over folds

    # Standardized train inputs per member, masked to train rows. We keep the full
    # (E, N, D) standardized tensor but ZERO the held-out row so it never enters
    # the loss (the loss masks to train rows anyway).
    Xn = (Xt.unsqueeze(0) - mu.unsqueeze(1)) / sd.unsqueeze(1)  # (E, N, D)
    held_idx = member_fold  # (E,)

    def _forward(p, b, x):
        return functional_call(base, (p, b), (x,))

    batched_forward = vmap(_forward, in_dims=(0, 0, 0))

    # One AdamW over the stacked ensemble params (each leaf is (E, ...)); the
    # update is independent per member because gradients are per-member. `params`
    # is the dict stack_module_state returned — the optimizer mutates its tensors
    # in place, so passing `params` to functional_call each step sees the updates.
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(params.values()), lr=MLP_LR, weight_decay=MLP_WD)

    mask_loss = train_mask.to(torch.float32)  # (E, N)
    denom = mask_loss.sum(1).clamp(min=1.0)  # (E,) == N-1
    for _ in range(MLP_MAX_EPOCHS):
        opt.zero_grad(set_to_none=True)
        pred = batched_forward(params, buffers, Xn)  # (E, N)
        # masked mean-squared error per member over train rows, matching
        # torch.nn.functional.mse_loss(net(xn), y[mask]) per member (reduction=mean
        # over the N-1 train rows). Sum the per-member losses (independent grads).
        sq = (pred - y_member) ** 2 * mask_loss  # (E, N)
        per_member = sq.sum(1) / denom  # (E,)
        loss = per_member.sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = batched_forward(params, buffers, Xn)  # (E, N)
    held = pred[torch.arange(n_members, device=device), held_idx]  # (E,)
    held = held.reshape(len(idxs), n).t().contiguous()  # (N, n_targets)
    return held.detach().cpu().numpy().astype(np.float64)


def _fit_mlp_loco_serial_reference(X: np.ndarray, y: np.ndarray, seed: int = 658) -> np.ndarray:
    """OLD serial single-output LOCO MLP — the EXACTNESS ORACLE for the batched path.

    A faithful copy of the pre-rewrite ``_fit_mlp_loco`` body: re-seed once, then per
    LOCO fold fit a FRESH ``_MLP(d)`` with a FRESH AdamW on the fold's (N-1) train
    rows (standardized on those rows), and read the held-out row. ``_assert_mlp_
    exactness`` checks the batched ``_fit_mlp_ensemble_loco`` reproduces this to
    <=1e-6. Always runs on CPU (the oracle is device-independent); not used in the
    production path (it is the serial loop the rewrite replaced).
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


def _assert_mlp_exactness(seed: int = 0, n: int = 12, d: int = 24) -> dict:
    """Assert the batched MLP reproduces the serial reference to <=1e-6 on CPU.

    Two checks, both against ``_fit_mlp_loco_serial_reference`` (the OLD serial
    algorithm), on a tiny CPU model + a clamped epoch count so this stays a fast
    (~sub-second) startup gate alongside ``_assert_ridge_exactness``:

    (a) SINGLE-output: ``_fit_mlp_loco`` (the A3.2 path) vs the serial reference.
    (b) MULTI-output GAP: ``_fit_mlp_ensemble_loco(target_idx=range(gap_dim))``
        (the A3.5 gap path) vs the serial reseed-PER-DIM reference
        ``[serial(Yv[:,k]) for k in range(gap_dim)]`` — this is the tiled-init
        path; without the per-dim tile it would drift ~0.38 (the round-2 finding).

    The batched path is forced onto CPU here (DEVICE pinned for the duration) so
    the gate is reduction-order-comparable to the CPU serial oracle. The recovery
    rewrite must not move the reported A3.2/A3.5 numbers; this is the MLP analogue
    of the ridge exactness gate (the batched MLP, unlike the ridge, is not exact
    to machine precision — batched GEMM vs per-net GEMV reduction order — so the
    tolerance is 1e-6, not 1e-12).
    """
    global DEVICE, MLP_MAX_EPOCHS
    saved_device, saved_epochs = DEVICE, MLP_MAX_EPOCHS
    DEVICE = "cpu"
    MLP_MAX_EPOCHS = 20  # clamp the gate so startup stays sub-second
    try:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, 3))
        W = rng.standard_normal((3, d))
        X = (z @ W + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
        B = rng.standard_normal((d, 5))
        Y = (X @ B * 0.05 + 0.1 * rng.standard_normal((n, 5))).astype(np.float32)
        # (a) single-output
        ser1 = _fit_mlp_loco_serial_reference(X, Y[:, 0])
        bat1 = _fit_mlp_loco(X, Y[:, 0])
        single_delta = float(np.max(np.abs(ser1 - bat1)))
        # (b) multi-output gap (the tiled path)
        gap = 4
        serg = np.stack([_fit_mlp_loco_serial_reference(X, Y[:, k]) for k in range(gap)], axis=1)
        batg = _fit_mlp_ensemble_loco(X, Y, target_idx=list(range(gap)), seed=658)
        multi_delta = float(np.max(np.abs(serg - batg)))
    finally:
        DEVICE, MLP_MAX_EPOCHS = saved_device, saved_epochs
    tol = 1e-6
    assert single_delta <= tol, (
        f"MLP exactness FAILED (single-output): max|Δpred|={single_delta:.3e} > {tol} "
        "between the batched A3.2 MLP and the serial reference"
    )
    assert multi_delta <= tol, (
        f"MLP exactness FAILED (multi-output gap): max|Δpred|={multi_delta:.3e} > {tol} "
        "between the tiled-batched A3.5 gap MLP and the serial reseed-per-dim reference "
        "(a >1e-2 delta means the per-dim init tile regressed — see _fit_mlp_ensemble_loco)"
    )
    return {"single_delta": single_delta, "multi_delta": multi_delta, "tol": tol}


# ── A3.4 ridge: closed-form (PRESS / hat-matrix) leave-one-out, exact ──────────


def _ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge weights (D, P) for X (N, D) -> Y (N, P) — primal normal equations.

    Kept as the EXACTNESS REFERENCE: ``_assert_ridge_exactness`` checks the
    closed-form dual LOCO below against the primal refit that uses this solve.
    The production LOCO path no longer calls it per fold.
    """
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def _ridge_predict_loco_refit(X: np.ndarray, Y: np.ndarray, lambdas: list[float]) -> np.ndarray:
    """REFERENCE primal-refit nested-CV LOCO ridge (the OLD O(D³) implementation).

    Retained ONLY as the exactness oracle for ``_assert_ridge_exactness``: the
    fast closed-form path below must reproduce this to ≤1e-6. Do NOT call in the
    production path (it is the 40h-on-CPU blowup this rewrite removes).
    """
    n = X.shape[0]
    p = Y.shape[1]
    preds = np.zeros((n, p), dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr, Ytr = X[tr], Y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
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


def _press_loo_mse_per_lambda(Xn: torch.Tensor, Y: torch.Tensor, lambdas) -> torch.Tensor:
    """Exact leave-one-out MSE per λ for a FIXED standardized design (PRESS).

    Xn (m, d) standardized design, Y (m, P) targets, both on DEVICE. Returns
    (n_lambda,) mean (over the m LOO folds AND P outputs) of the squared LOO
    residual — IDENTICAL to refitting ridge on each (m-1)-row subset and scoring
    the left-out row, by the PRESS / hat-matrix identity::

        Ŷ = H Y,   H = Xn (Xnᵀ Xn + λI)⁻¹ Xnᵀ   (the m×m hat matrix)
        LOO residual_k = (Y_k − Ŷ_k) / (1 − H_kk)

    Computed in the DUAL (row) space via ONE eigendecomposition of the m×m Gram
    ``G = Xn Xnᵀ`` reused across all λ: with G = Q diag(g) Qᵀ,
    ``H(λ) = Q diag(g/(g+λ)) Qᵀ``, so both diag(H) and HY are a per-λ rescale of
    the cached eigenbasis. O(m³) once + O(m² · n_lambda), vs the old
    O(n_lambda · m · d³) refit. Exact (the closed form IS the refit).
    """
    G = Xn @ Xn.t()  # (m, m) dual Gram — d cancels, this is the N≪D win
    # symmetric eigendecomposition (G is PSD symmetric)
    evals, Q = torch.linalg.eigh(G)  # G = Q diag(evals) Qᵀ
    QtY = Q.t() @ Y  # (m, P)
    Qsq = Q * Q  # (m, m) elementwise, for diag(H)
    out = torch.empty(len(lambdas), dtype=Xn.dtype, device=Xn.device)
    for li, lam in enumerate(lambdas):
        filt = evals / (evals + lam)  # (m,)
        # diag(H) = sum_j Q[k,j]^2 * filt[j]
        h_diag = Qsq @ filt  # (m,)
        # Ŷ = H Y = Q diag(filt) Qᵀ Y
        Yhat = Q @ (filt.unsqueeze(1) * QtY)  # (m, P)
        resid = Y - Yhat  # (m, P)
        denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(1)  # (m, 1)
        loo_resid = resid / denom  # (m, P)
        out[li] = (loo_resid * loo_resid).mean()
    return out


def _ridge_dual_weights(Xn: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """Ridge weights (d, P) via the DUAL (Woodbury) form, exact.

    ``w = Xnᵀ (Xn Xnᵀ + λ I_m)⁻¹ Y`` — the m×m system (m = N-1 ≪ d), identical to
    the primal ``(XnᵀXn + λI_d)⁻¹ Xnᵀ Y`` but O(m²d) not O(d³). Used for the outer
    held-out prediction at the inner-selected λ.
    """
    m = Xn.shape[0]
    A = Xn @ Xn.t() + lam * torch.eye(m, dtype=Xn.dtype, device=Xn.device)  # (m, m)
    alpha = torch.linalg.solve(A, Y)  # (m, P)
    return Xn.t() @ alpha  # (d, P)


def _ridge_predict_loco(X: np.ndarray, Y: np.ndarray, lambdas: list[float]) -> np.ndarray:
    """LOCO ridge predictions of a multi-output target Y (N, P) from X (N, D).

    Nested-CV λ: for each held-out context, pick λ minimizing inner-LOO MSE on the
    training contexts (no λ leakage into the held-out read). Returns predictions
    (N, P). EXACT closed-form rewrite of ``_ridge_predict_loco_refit``: the inner
    λ-selection uses the PRESS identity (``_press_loo_mse_per_lambda``) instead of
    refitting per inner fold, and every solve is the dual/Woodbury form
    (``_ridge_dual_weights``). The standardization, the nested-CV protocol, and
    every prediction are bit-equivalent to the refit reference (asserted ≤1e-6 in
    ``_assert_ridge_exactness``); only the O(D³) cost is removed.
    """
    n = X.shape[0]
    p = Y.shape[1]
    device = torch.device(DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y)).to(device=device, dtype=torch.float64)
    preds = np.zeros((n, p), dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        tr_t = torch.tensor(tr, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        mu = Xtr.mean(0)
        # Match the refit reference EXACTLY: it standardizes with numpy
        # `Xtr.std(0)` (ddof=0, population). torch's `.std(0)` defaults to
        # unbiased=True (ddof=1) — pass correction=0 so the scale is bit-identical
        # to the oracle (the exactness assert would otherwise catch the drift).
        sd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
        # inner LOO to pick λ (exact PRESS over the standardized train design)
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr, lambdas)  # (n_lambda,)
        best_lam = lambdas[int(torch.argmin(mse).item())]
        w = _ridge_dual_weights(Xtr_n, Ytr, best_lam)  # (d, P)
        x_held = (Xt[i] - mu) / sd  # (d,)
        preds[i] = (x_held @ w).detach().cpu().numpy()
    return preds


def _assert_ridge_exactness(seed: int = 0, n: int = 14, d: int = 40, p: int = 3) -> dict:
    """Assert the closed-form dual ridge LOCO matches the primal-refit LOCO ≤1e-6.

    Exactness is the GATE for the recovery rewrite: the reported A3.4/A3.5 numbers
    must not move. Builds a small synthetic (X, Y) with real rank structure, runs
    BOTH ``_ridge_predict_loco`` (new closed-form) and ``_ridge_predict_loco_refit``
    (old primal refit), and asserts max|Δpred| and |Δρ| are ≤1e-6 per output. Also
    checks the std convention matches (numpy ``.std(0)`` ddof=0 vs torch
    ``.std(0)`` unbiased=True differ — see the note below; the refit uses numpy,
    the fast path uses torch, so a passing assert proves the two conventions land
    within tolerance for this regime OR that they were reconciled).

    NOTE on the std convention: numpy ``ndarray.std`` defaults to ddof=0
    (population), torch ``Tensor.std`` defaults to unbiased=True (ddof=1). The
    standardization is applied symmetrically to train AND held-out rows and only
    RESCALES per feature, so a constant per-feature scale factor (the ddof ratio
    sqrt((m-1)/m)) on Xn cancels in the ridge fit ONLY up to the λ grid (λ is in
    the un-rescaled units). The exactness assert below would CATCH any resulting
    drift; it passes because the rewrite reconciles the convention — the fast path
    standardizes with ddof matching the reference. (See ``_ridge_predict_loco``.)
    """
    rng = np.random.default_rng(seed)
    # Latent low-rank signal + noise so ridge has something to fit.
    z = rng.standard_normal((n, 3))
    W = rng.standard_normal((3, d))
    X = z @ W + 0.1 * rng.standard_normal((n, d))
    B = rng.standard_normal((d, p))
    Y = X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))
    lambdas = [1e-1, 1.0, 10.0]
    fast = _ridge_predict_loco(X, Y, lambdas)
    ref = _ridge_predict_loco_refit(X, Y, lambdas)
    max_abs = float(np.max(np.abs(fast - ref)))
    # per-output ρ agreement (the actual reported statistic family)
    rho_deltas = []
    for k in range(p):
        rf = spearmanr(fast[:, k], Y[:, k]).correlation
        rr = spearmanr(ref[:, k], Y[:, k]).correlation
        if not (np.isnan(rf) or np.isnan(rr)):
            rho_deltas.append(abs(float(rf - rr)))
    max_rho_delta = max(rho_deltas) if rho_deltas else 0.0
    tol = 1e-6
    assert max_abs <= tol, (
        f"ridge exactness FAILED: max|Δpred|={max_abs:.3e} > {tol} between the "
        "closed-form dual LOCO and the primal-refit LOCO oracle"
    )
    assert max_rho_delta <= tol, (
        f"ridge exactness FAILED: max|Δρ|={max_rho_delta:.3e} > {tol} between the "
        "closed-form dual LOCO and the primal-refit LOCO oracle"
    )
    return {"max_abs_pred_delta": max_abs, "max_rho_delta": max_rho_delta, "tol": tol}


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


# ── checkpoint-per-cell helpers (resume) ──────────────────────────────────────


def _param_hash(phase: str, feat_dim: int = 0) -> str:
    """Short hash of the hyperparameters that determine a checkpoint cell's value.

    Stamped into every saved cell and re-checked on load so a resume into a REUSED
    ``--out-dir`` after a hyperparameter change (λ grid / MLP epochs / feat_dim /
    A35_MLP_TARGET_DIM / bootstrap count / smoke vs real) RECOMPUTES the cell
    instead of silently serving a stale value computed under the old params.
    Phase-scoped so an a32 cell and an a34a35 cell hash only the constants that
    actually feed them (#minor stale-serve, round-2 review).
    """
    import hashlib

    common = (MLP_HIDDEN, MLP_LR, MLP_WD, MLP_MAX_EPOCHS, int(feat_dim))
    if phase == "a32":
        payload = ("a32", common, N_BOOTSTRAP)
    elif phase == "a34a35":
        payload = ("a34a35", common, tuple(RIDGE_LAMBDAS), A35_MLP_TARGET_DIM)
    else:
        payload = (phase, common)
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _cell_path(out_dir: Path, phase: str, key: str) -> Path:
    """Per-(phase, recipe×layer) checkpoint cell path under out_dir/cells/<phase>/."""
    safe = key.replace("/", "_").replace(" ", "_")
    return out_dir / "cells" / phase / f"{safe}.json"


def _load_cell(out_dir: Path, phase: str, key: str, param_hash: str | None = None):
    """Load a checkpoint cell, returning None (recompute) on a param-hash mismatch.

    ``param_hash`` is the caller's hash of the load-bearing constants
    (``_param_hash(phase, feat_dim)``). A cached cell whose stamped ``_param_hash``
    differs was computed under different hyperparameters → STALE; drop it so the
    cell recomputes rather than serving a value from a prior λ-grid / epochs /
    feat_dim / A35_MLP_TARGET_DIM. A cell with no stamp (pre-this-version) is also
    treated as stale when a hash is required.
    """
    p = _cell_path(out_dir, phase, key)
    if p.exists():
        try:
            cell = load_json(p)
        except (ValueError, OSError):
            logger.warning("corrupt checkpoint cell %s — recomputing", p)
            return None
        if param_hash is not None and cell.get("_param_hash") != param_hash:
            logger.info(
                "stale checkpoint cell %s (param-hash %s != %s) — recomputing",
                p,
                cell.get("_param_hash"),
                param_hash,
            )
            return None
        return cell
    return None


def _save_cell(out_dir: Path, phase: str, key: str, payload, param_hash: str | None = None) -> None:
    if param_hash is not None and isinstance(payload, dict):
        payload = {**payload, "_param_hash": param_hash}
    dump_json(payload, _cell_path(out_dir, phase, key))


def _shard_owns(items: list, shard_idx: int, n_shards: int) -> set:
    """Round-robin set of indices this shard owns (multi-GPU fan-out)."""
    return {i for i in range(len(items)) if i % n_shards == shard_idx}


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


def fit_a32(
    store,
    spans_dir,
    e0,
    ctx_ids,
    layers,
    recipes,
    noise_floor,
    base_prior,
    out_dir: Path,
    shard_idx: int = 0,
    n_shards: int = 1,
) -> list[dict]:
    """A3.2: per (behavior, layer, summary) LOCO MLP ρ vs baselines + noise floor.

    Checkpoint-per-(column, recipe, layer): each cell is persisted the moment it
    completes and skipped on resume, so a crash never restarts from zero. The
    (recipe × layer) grid is sharded round-robin when n_shards > 1 (multi-GPU).
    A progress log line fires per completed (column, recipe) sweep.
    """
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

    # grid of (recipe, layer) work units sharded across GPUs
    grid = [(recipe, li) for recipe in recipes for li in range(len(layers))]
    owned = _shard_owns(grid, shard_idx, n_shards)
    total = len(columns) * len(grid)
    # A3.2 uses the full v0 summaries (no feat_dim clamp); the cell value depends
    # on the MLP hyperparams + the bootstrap count, hashed here so a resume after
    # a param change recomputes rather than serving stale cells.
    ph = _param_hash("a32", feat_dim=0)
    done = 0
    t0 = time.time()
    for col in columns:
        y, kept = e0_target(e0, col, ctx_ids)
        if len(kept) < 4:
            cells.append({"column": col, "status": "too_few_contexts", "n": len(kept)})
            continue
        for gi, (recipe, li) in enumerate(grid):
            done += 1
            if gi not in owned:
                continue  # another shard owns this work unit
            key = f"{col}__{recipe}__L{layers[li]}"
            cached = _load_cell(out_dir, "a32", key, param_hash=ph)
            if cached is not None:
                cells.append(cached)
                continue
            if recipe == "attn":
                X = _attn_matrix(spans_dir, li, kept, store["capture_layers"], attn_w)
            else:
                X = _summary_matrix(store, recipe, li, kept)
            pred = _fit_mlp_loco(X, y)
            rho = _rho(pred, y)
            mean_pred = np.full_like(y, y.mean())
            rho_mean = _rho(mean_pred, y)  # predict-mean baseline (constant -> ~None)
            cell = {
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
                # `_rho` returns None for a constant-y FLOORED cell, and the
                # bootstrap would then raise on a degenerate measurement.
                "bootstrap": (
                    _cluster_bootstrap_rho(pred, y, n_boot=N_BOOTSTRAP, seed=658)
                    if rho is not None
                    else None
                ),
                # attn is a RANDOM-PROJECTION CONTROL, not a learned pool
                # (carried CONCERN attn-pool-weight-unfitted).
                "is_random_projection_control": recipe == "attn",
            }
            _save_cell(out_dir, "a32", key, cell, param_hash=ph)
            cells.append(cell)
        if any(gi in owned for gi in range(len(grid))):
            elapsed = time.time() - t0
            logger.info(
                "[a32] %s done | %d/%d work units | %.1fs elapsed",
                col,
                done,
                total,
                elapsed,
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
    cc_map,
    store,
    e0,
    rb,
    ctx_ids,
    layers,
    shuffle_seed,
    feat_dim=0,
    out_dir: Path | None = None,
    recipe_name: str = "",
    shard_idx: int = 0,
    n_shards: int = 1,
) -> dict:
    """A3.4 ridge + A3.5 MLP for ONE c_C recipe: c_C → v0(C) held-out.

    cc_map = {ctx_id: (Lc, H)} for this c_C recipe. Reports the LOCO ρ between
    predicted and measured v0 (per layer, mean recipe) for ridge (A3.4) and MLP
    (A3.5), the linear-vs-nonlinear gap, the within-context shuffle null
    (round-1 concern #4), AND the downstream ``r_B^T M c_C → E0`` chain ρ per
    behavior.

    Per-layer cells are checkpointed (resume) and the layer grid is sharded
    round-robin across GPUs. ``feat_dim`` > 0 truncates the c_C / v0 / r_B feature
    dim to the leading ``feat_dim`` dims — a SMOKE-ONLY clamp (real run = 0 = full
    H). The dual-form ridge + batched MLP make even the full-H A3.4/A3.5 tractable.
    """
    out: dict = {"per_layer": [], "shuffle_null": [], "chain_rho_e0": {}}
    C = np.stack([np.asarray(cc_map[c]) for c in ctx_ids])  # (N, Lc, H)
    V = np.stack([store["summaries"]["mean"][c].numpy() for c in ctx_ids])  # (N, Lc, H)
    if feat_dim:
        C = C[:, :, :feat_dim]
        V = V[:, :, :feat_dim]
    n = len(ctx_ids)
    owned = _shard_owns(list(range(len(layers))), shard_idx, n_shards)
    # The a34a35 cell value depends on the λ grid, MLP hyperparams, the gap dim,
    # AND feat_dim — hash them so a resume after any change recomputes.
    ph = _param_hash("a34a35", feat_dim=feat_dim)
    # Cache the per-layer LOCO ridge prediction of v0 so the chain ρ can reuse it.
    ridge_pred_v0_by_layer: dict[int, np.ndarray] = {}
    t0 = time.time()
    for li in range(len(layers)):
        if li not in owned:
            continue
        ckpt_key = f"{recipe_name}__L{layers[li]}" if recipe_name else f"L{layers[li]}"
        cached = (
            _load_cell(out_dir, "a34a35", ckpt_key, param_hash=ph) if out_dir is not None else None
        )
        if cached is not None and "_ridge_pred_v0" in cached:
            ridge_pred_v0_by_layer[li] = np.asarray(cached["_ridge_pred_v0"])
            out["per_layer"].append(cached["per_layer"])
            out["shuffle_null"].append(cached["shuffle_null"])
            continue
        # Each layer gets its OWN RNG so resume / sharding is order-independent
        # (the shuffle null permutation must be reproducible per layer regardless
        # of which other layers ran). Seed = shuffle_seed + layer index.
        rng = np.random.default_rng(shuffle_seed + li)
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
        # output dim at a time (now batched across the gap dims AND folds in one
        # vmapped ensemble); the ridge cos for the gap is recomputed over the same
        # slice (NOT the full-dim ridge_cos).
        gap_dim = min(A35_MLP_TARGET_DIM, Yv.shape[1])
        # Per-output-dim seed matches the OLD serial code, which called
        # _fit_mlp_loco(Xc, Yv[:, k]) once per k — each call reseeded with the
        # default seed=658. Reproduce that per-dim seeding so the batched ensemble
        # is bit-equivalent to the serial per-dim fits.
        mlp_pred = _fit_mlp_ensemble_loco(
            Xc.astype(np.float32),
            Yv.astype(np.float32),
            target_idx=list(range(gap_dim)),
            seed=658,
        )  # (N, gap_dim)
        mlp_cos = _rowwise_cos(mlp_pred, Yv[:, :gap_dim])
        ridge_cos_gap = _rowwise_cos(ridge_pred[:, :gap_dim], Yv[:, :gap_dim])
        per_layer = {
            "layer": layers[li],
            "ridge_mean_cos": float(np.mean(ridge_cos)),  # A3.4, full-dim
            "mlp_mean_cos": float(np.mean(mlp_cos)),  # over gap_dim
            # gap = MLP vs ridge BOTH read over gap_dim (like-for-like).
            "nonlinear_gap": float(np.mean(mlp_cos) - np.mean(ridge_cos_gap)),
            "ridge_mean_cos_on_gap_dim": float(np.mean(ridge_cos_gap)),
            "gap_target_dim": gap_dim,
        }
        # shuffle null: permute the v0 rows, re-fit ridge, report cos.
        perm = rng.permutation(n)
        ridge_pred_sh = _ridge_predict_loco(Xc, Yv[perm], RIDGE_LAMBDAS)
        shuffle_null = {
            "layer": layers[li],
            "ridge_mean_cos_shuffled": float(np.mean(_rowwise_cos(ridge_pred_sh, Yv[perm]))),
        }
        out["per_layer"].append(per_layer)
        out["shuffle_null"].append(shuffle_null)
        if out_dir is not None:
            _save_cell(
                out_dir,
                "a34a35",
                ckpt_key,
                {
                    "per_layer": per_layer,
                    "shuffle_null": shuffle_null,
                    "_ridge_pred_v0": ridge_pred.tolist(),
                },
                param_hash=ph,
            )
        logger.info(
            "[a34a35:%s] layer %d done | %.1fs elapsed",
            recipe_name or "?",
            layers[li],
            time.time() - t0,
        )
    # Chain ρ: project the LOCO-predicted v0 through each behavior's r_B and
    # Spearman-correlate against the measured E0 — the full shortcut
    # r_B^T (M c_C) → E0(C,B). Best layer per behavior is reported.
    # NOTE: requires every layer's ridge_pred_v0, so it is computed ONLY when this
    # process holds all layers — the single-process (no-shard) run, OR the FINAL
    # no-shard reassembly pass that runs after the per-shard processes have written
    # every layer's cell (that final pass re-reads all cells from cells/a34a35/ via
    # the per-cell checkpoint load above, repopulating ridge_pred_v0_by_layer for
    # all layers). A still-running per-shard process owns only its layer subset, so
    # it skips the chain ρ and leaves it to that final pass.
    if len(ridge_pred_v0_by_layer) == len(layers):
        out["chain_rho_e0"] = _chain_rho(
            ridge_pred_v0_by_layer, store, e0, rb, ctx_ids, layers, feat_dim
        )
    return out


def _chain_rho(ridge_pred_v0_by_layer, store, e0, rb, ctx_ids, layers, feat_dim) -> dict:
    """r_B^T (M c_C) → E0 chain ρ per behavior, best layer (reads cached ridge preds)."""
    chain: dict = {}
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
            if li not in ridge_pred_v0_by_layer:
                continue
            r = np.asarray(rdir[li])  # (H,)
            if feat_dim:
                r = r[:feat_dim]  # match the smoke-clamped predicted-v0 dim
            pred_v0 = ridge_pred_v0_by_layer[li][kept_idx]  # (n_kept, H or feat_dim)
            chain_pred = pred_v0 @ r
            rho = _rho(chain_pred, y)
            if rho is not None and (best is None or rho > best["rho"]):
                best = {"layer": layers[li], "rho": rho}
        if best is not None:
            chain[col] = best
    return chain


def fit_a34_a35(
    store,
    cc_recipes,
    e0,
    rb,
    ctx_ids,
    layers,
    shuffle_seed=658,
    feat_dim=0,
    out_dir: Path | None = None,
    shard_idx: int = 0,
    n_shards: int = 1,
) -> dict:
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
            cc_map,
            store,
            e0,
            rb,
            ctx_ids,
            layers,
            shuffle_seed,
            feat_dim=feat_dim,
            out_dir=out_dir,
            recipe_name=name,
            shard_idx=shard_idx,
            n_shards=n_shards,
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


def _parse_shard(shard: str | None) -> tuple[int, int]:
    """Parse a '--shard k/n' string into (shard_idx, n_shards). None -> (0, 1)."""
    if not shard:
        return 0, 1
    try:
        k, n = shard.split("/")
        shard_idx, n_shards = int(k), int(n)
    except (ValueError, AttributeError) as e:
        raise SystemExit(f"--shard must be 'k/n' (e.g. 0/4), got {shard!r}: {e}") from e
    if not (0 <= shard_idx < n_shards) or n_shards < 1:
        raise SystemExit(f"--shard {shard!r}: require 0 <= k < n and n >= 1")
    return shard_idx, n_shards


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #658 P1-P3/N1/A1: predictor fits + stats.")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--e0", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=EVAL_RESULTS_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="compute device for the batched MLP/ridge ops: auto (default; cuda if "
        "present else cpu) | cuda | cpu. The reported DV is device-invariant.",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="multi-GPU work split 'k/n' (0-based): this process fits the round-robin "
        "subset of the (recipe × layer) grid it owns. Per-cell checkpoints let the "
        "shards write into the SAME out-dir; after all k/n shards finish, run ONE more "
        "pass with NO --shard (the default single-process run) against the same out-dir "
        "— it reloads every shard's cells from checkpoint and assembles the aggregate. "
        "There is no 'merge' subcommand. Default: single process (no sharding).",
    )
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

    global MLP_MAX_EPOCHS, RIDGE_LAMBDAS, N_BOOTSTRAP, DEVICE
    DEVICE = _resolve_device(args.device)
    logger.info("compute device: %s", DEVICE)

    # Exactness gates: the batched/closed-form rewrites MUST reproduce the OLD
    # serial computation (the reported A3.2/A3.4/A3.5 numbers must not move). Run
    # BOTH at startup (cheap, ~sub-second total) — a recovery-mode rewrite that
    # silently changed the DV is the failure these guard.
    #   - ridge: closed-form dual PRESS LOCO vs primal-refit LOCO, ≤1e-6 (exact)
    #   - MLP: batched single-output (A3.2) + tiled multi-output gap (A3.5) vs the
    #     serial reference, ≤1e-6 (reduction-order, not machine-precision)
    exactness = _assert_ridge_exactness()
    logger.info(
        "ridge exactness PASS: max|Δpred|=%.2e max|Δρ|=%.2e (<= %.0e)",
        exactness["max_abs_pred_delta"],
        exactness["max_rho_delta"],
        exactness["tol"],
    )
    mlp_exactness = _assert_mlp_exactness()
    logger.info(
        "MLP exactness PASS: single max|Δ|=%.2e gap max|Δ|=%.2e (<= %.0e)",
        mlp_exactness["single_delta"],
        mlp_exactness["multi_delta"],
        mlp_exactness["tol"],
    )

    shard_idx, n_shards = _parse_shard(args.shard)
    if n_shards > 1:
        logger.info("sharded fit: this process owns shard %d/%d", shard_idx, n_shards)

    # SMOKE-ONLY compute clamp: keep the smoke fast end-to-end. The real-run
    # defaults (the §11-grounded values) are untouched; _fit_mlp_loco /
    # _ridge_predict_loco read these globals at call time.
    if args.smoke:
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

    a32_cells = fit_a32(
        store,
        spans_dir,
        e0,
        ctx_ids,
        layers,
        recipes,
        noise,
        base_prior,
        out_dir,
        shard_idx=shard_idx,
        n_shards=n_shards,
    )
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
        out_dir=out_dir,
        shard_idx=shard_idx,
        n_shards=n_shards,
    )
    dump_json(a34_35, out_dir / "a34_a35.json")

    if n_shards > 1:
        # A sharded process owns only a SUBSET of the (recipe × layer) cells; the
        # aggregate / verdicts / figures need the FULL grid. Stop here after
        # persisting this shard's cells — a final no-shard (or --shard 0/1) pass
        # re-reads every cell from the checkpoint dir and assembles the aggregate.
        logger.info(
            "shard %d/%d complete — cells written to %s/cells/. Run a final no-shard "
            "pass (same out-dir) to assemble the aggregate from all shards' cells.",
            shard_idx,
            n_shards,
            out_dir,
        )
        return 0

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
        {
            **agg,
            "figures": figs,
            "ridge_exactness": exactness,
            "mlp_exactness": mlp_exactness,
            "compute_device": DEVICE,
            "metadata": reproducibility_metadata({"script": "issue658_fit"}),
        },
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
