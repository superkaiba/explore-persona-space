# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ², λ, ρ, ̄) in scientific docstrings.
"""Vectorized batched LOCO MLP skill-over-mean + downstream chain ρ (reusable).

The reusable replacement for the catastrophically slow per-fold/per-layer serial
MLP-fit pattern in ``scripts/issue722_skill_over_mean.py`` and
``scripts/issue658_mlp_chain.py`` (incident #722: 19.5 CPU-hours / 96+ min, did
not finish). Those sweeps are **overhead-bound, not FLOP-bound** — the actual
math is ~19 TFLOP (minutes of real compute); the wall-time was Python loop
overhead + torch op-dispatch on tiny tensors + thread oversubscription (78
threads on (49, 512) GEMMs). The fix is VECTORIZATION (`.claude/rules/
vectorize-many-cell-fits.md`).

What this module does that the serial path did not:

1. **Train ALL leave-one-context-out (LOCO) fold-nets simultaneously** as one
   batched parameter tensor via ``torch.func.functional_call`` + ``vmap`` (the
   same machinery #658's ``_fit_mlp_ensemble_loco`` already used per-CALL) — BUT
   it ALSO batches the member dimension ACROSS **layers**, across the **MLP
   variants** (base / z-scored-input / shuffle-null), and across the **PCA target
   dims** into ONE ensemble. So the 300-epoch loop runs ~300 BATCHED steps TOTAL
   for the whole (layers × variants × target-dims × folds) battery, not 50×300
   tiny steps per (layer, variant). #722's 84 separate ensemble fits (28 layers ×
   3 variants) collapse to ONE.
2. **Multi-output net** — one net predicts all ``pca_target_dim`` PCA target dims
   at once (a multi-output head, NOT one scalar net per dim).
3. **`torch.set_num_threads`** to a sane value (the slow run thrashed with 78
   threads on tiny ops). As of #1079 the cap is ON BY DEFAULT:
   ``num_threads=None`` resolves to ``max(1, min(8, os.cpu_count(), ambient
   pool))`` — ambient-ceilinged, so the default only ever caps DOWN — pinned
   for the fit and restored after; ``num_threads=0`` opts out (see
   ``_resolve_num_threads``). A ``--device`` arg (cpu default, cuda optional).
4. **Seed-pinned to match the existing #658 recipe** so numbers are comparable +
   reproducible.

EXACTNESS CONTRACT. A single (group) slice of the batched fit is BIT-EQUIVALENT
to #658's ``_fit_mlp_ensemble_loco`` — same ``_MLP(d_in, hidden)`` architecture,
same ``AdamW(lr, wd)``, same epoch count, same per-fold TRAIN-ONLY
standardization (torch ``.std`` ddof=1 via the sum-of-squares form), same
per-member init (each member's params are the per-fold init drawn by
``torch.manual_seed(seed)`` then ``[_MLP(d) for _ in range(n)]``, addressed by
``fold_i``, so the multi-output net reuses the SAME ``n`` per-fold inits the
serial reseed-per-dim reference produced). The reproduce-check
(``assert_matches_reference``) verifies max|Δpred| ≤ 1e-6 against
``_fit_mlp_ensemble_loco`` on synthetic data before any output is trusted.

The closed-form ridge / linear LOCO is already cheap via #658's
``_press_loo_mse_per_lambda`` / ``_ridge_dual_weights`` — this module reuses
those for the ridge arm and only batches the gradient-descent MLP arm.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("vectorized_mlp_skill")


def _free_bytes(dev: torch.device) -> int:
    """Best-effort free memory (bytes) on ``dev``: CUDA free VRAM, else free system RAM.

    On cuda reads ``torch.cuda.mem_get_info(dev)[0]`` (free VRAM). On cpu reads the
    kernel's available physical page count (``SC_AVPHYS_PAGES × SC_PAGE_SIZE``) — no
    new dependency. Returns 0 if neither is readable (the caller then falls back to
    the requested chunk unchanged — i.e. old behavior, so a probe failure never
    tightens silently).
    """
    if dev.type == "cuda":
        try:
            return int(torch.cuda.mem_get_info(dev)[0])
        except (RuntimeError, AssertionError, ValueError):
            return 0
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def resolve_chunk_cap(
    requested: int,
    n_members: int,
    n: int,
    d_in: int,
    free_bytes: int,
    *,
    live_factor: int = 26,
    safety: float = 0.8,
) -> int:
    """Cap the per-chunk member count so the chunk's live ``(c, n, d_in)`` fp32
    intermediates fit in ``free_bytes`` (a PURE function of the inputs).

    Each chunk of ``c`` members materializes several live ``(c, n, d_in)`` fp32
    tensors at once inside the fit loop — the advanced-index design copy ``Xc``, the
    element-square temp ``Xc²``, the standardized ``Xn`` — and, dominating them, the
    ``torch.bmm`` AUTOGRAD BACKWARD GRAPH over the ``(c, n, d_in)`` activations across
    the training loop plus AdamW moment buffers and the CPU allocator's cached blocks.
    ``live_factor`` is a per-chunk peak MULTIPLE of a single ``(c, n, d_in)`` fp32
    tensor: ``live_factor=26`` is calibrated from a MEASURED real-shape peak — at
    n=480, d_in=3584, p=64, c=64 the measured ``ru_maxrss`` delta was ~10.7 GiB ≈
    25.5 × the 420 MiB single ``(c, n, d_in)`` tensor (#811 phase0 gate, CPU). A
    naive count of the explicit temporaries (≈4) under-estimates by ~6× and re-OOMs
    (a c=218 auto-cap needed ~36 GiB). Per-chunk peak scales as
    ``live_factor × c × n × d_in × 4 B``. The cap is::

        max(1, min(requested, n_members,
                   floor(free_bytes × safety / (live_factor × n × d_in × 4))))

    ``requested`` and ``n_members`` bound it from above (never chunk larger than the
    ensemble, never larger than asked); ``free_bytes == 0`` (unreadable probe) leaves
    ``requested`` unchanged so a probe failure never tightens the chunk silently.
    Returns at least 1 (a single member always attempts, letting a genuinely
    too-large problem OOM loudly rather than silently no-op).
    """
    hard_cap = min(requested, n_members)
    if free_bytes <= 0:
        return max(1, hard_cap)
    per_member = live_factor * n * d_in * 4  # bytes per (1, n, d_in) fp32 × live_factor
    if per_member <= 0:
        return max(1, hard_cap)
    mem_cap = math.floor(free_bytes * safety / per_member)
    return max(1, min(hard_cap, mem_cap))


# Reuse #658's EXACT recipe constants + ridge machinery (do NOT re-implement).
# The #658 fit-predictors module lives under scripts/; add it to the path so the
# canonical solvers (the exactness oracles) are importable from the library.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue658_fit_predictors as _i658  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    MLP_HIDDEN,
    MLP_LR,
    MLP_MAX_EPOCHS,
    MLP_WD,
    RIDGE_LAMBDAS,
    _press_loo_mse_per_lambda,
    _ridge_dual_weights,
)

# #658's default MLP-init seed (every _fit_mlp_loco / _fit_mlp_ensemble_loco call
# reseeds with this); the skill scripts pass seed=658 / SEED=42 explicitly.
DEFAULT_MLP_SEED = 658

# #891 launch-prefix / #847 env.py setdefault / #811 patch value (all 8).
DEFAULT_FIT_NUM_THREADS = 8


def _resolve_num_threads(num_threads: int | None, *, ambient: int | None = None) -> int | None:
    """Resolve the per-fit CPU thread cap (#1079).

    ``None`` -> conservative default ``max(1, min(DEFAULT_FIT_NUM_THREADS,
    os.cpu_count(), ambient))`` — the ambient (pre-call torch pool) ceiling
    means the default only ever caps DOWN, never raises a deliberately
    tighter pre-call pin. Env override ``EPS_VECTORIZED_FIT_DEFAULT_THREADS``:
    unset -> ``DEFAULT_FIT_NUM_THREADS``; ``""`` or ``"0"`` -> disabled
    (return None, no pin — the #847 ``EPS_VM_THREAD_CAP`` convention);
    negative or non-integer -> ``ValueError`` (fail loud; deliberate
    divergence from #847's silent negative-disable).
    ``0`` -> explicit opt-out (return None, no pin). ``>0`` -> honored
    verbatim (an explicit value MAY exceed the ambient pool — the documented
    escape hatch). ``<0`` -> ``ValueError``.
    """
    if num_threads is None:
        raw = os.environ.get("EPS_VECTORIZED_FIT_DEFAULT_THREADS")
        if raw is None:
            cap = DEFAULT_FIT_NUM_THREADS
        else:
            raw = raw.strip()
            if raw in ("", "0"):
                return None  # disabled — #847 convention
            cap = int(raw)  # fails loud on garbage ("eight")
            if cap == 0:
                return None  # numeric-zero forms ("00", "+0") also disable
            if cap < 0:
                raise ValueError(f"EPS_VECTORIZED_FIT_DEFAULT_THREADS must be >= 0, got {cap}")
        bounds = [cap]
        if os.cpu_count():
            bounds.append(os.cpu_count())
        if ambient:
            bounds.append(ambient)
        return max(1, min(bounds))
    n = int(num_threads)
    if n == 0:
        return None
    if n < 0:
        raise ValueError(f"num_threads must be >= 0 (0 = opt-out), got {n}")
    return n  # explicit caller value honored VERBATIM (no clamp)


def _with_thread_cap(fn):
    """Pin torch's intra-op pool per the fn's ``num_threads``/``device`` kwargs
    for the call duration (CPU only), restoring the previous value in a
    ``finally``. The None-default path is ambient-ceilinged (never raises the
    pool); ``num_threads=0`` opts out entirely (#1079). The device check runs
    BEFORE resolution, so a malformed ``num_threads`` on a CUDA call keeps the
    pre-#1079 behavior while a CPU call fails loud before any fit work.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("device", "cpu") != "cpu":
            return fn(*args, **kwargs)
        prev = torch.get_num_threads()
        resolved = _resolve_num_threads(bound.arguments.get("num_threads"), ambient=prev)
        if resolved is None:
            return fn(*args, **kwargs)
        torch.set_num_threads(resolved)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(prev)

    return wrapper


# ── batched LOCO MLP ensemble (the core vectorization) ────────────────────────


@dataclass
class MLPGroup:
    """One independent LOCO-MLP fit problem in the batched ensemble.

    A "group" is any (X, Y) pair that should be fit with its OWN per-fold nets,
    independent of every other group. Typical groups: one per (layer × variant ×
    [target-dim already folded into Y's columns]). All groups in a batch share
    the SAME n (number of contexts/folds), d_in, pca_target_dim, hidden, seed,
    lr, wd, epochs — that shared shape is what lets them ride one vmapped
    ensemble. Groups may differ ONLY in their X / Y values (e.g. a different
    layer's c_C and v0, or a row-permuted v0 for a shuffle null).

    Attributes:
        key: opaque caller label carried back on the result (e.g. ("L18", "base")).
        X: (n, d_in) fp32 input design for this group.
        Y: (n, pca_target_dim) fp32 multi-output target (already PCA-reduced).
    """

    key: tuple
    X: np.ndarray
    Y: np.ndarray


@dataclass
class BatchedMLPResult:
    """Held-out LOCO predictions for every group, aligned to the input order."""

    preds_by_key: dict  # key -> (n, pca_target_dim) held-out LOCO predictions
    n_groups: int
    n_folds: int
    pca_target_dim: int
    n_members: int  # n_groups * n_folds (the ensemble size)
    chunk_size: int


@_with_thread_cap
def fit_batched_loco_mlp(
    groups: list[MLPGroup],
    *,
    seed: int = DEFAULT_MLP_SEED,
    hidden: int = MLP_HIDDEN,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = MLP_MAX_EPOCHS,
    device: str = "cpu",
    chunk_size: int = 512,
    num_threads: int | None = None,
) -> BatchedMLPResult:
    """Fit ALL groups' LOCO fold-nets simultaneously as one batched ensemble.

    The ensemble has ``E = n_groups * p * n`` member SCALAR nets (n = folds =
    contexts, p = ``pca_target_dim``); member ``(group g, target-col t, fold i)``
    is a single-output ``_i658._MLP(d_in, hidden)`` trained on group ``g``'s
    ``n-1`` train rows (≠ fold ``i``) to predict v0 PCA-dim ``t``, standardized on
    those rows, read on the held-out row ``i``. Every member is one vmapped
    element; the ``max_epochs`` loop runs ONCE over the whole ``E``-member batch
    (chunked to bound memory), so 28 layers × 3 variants × 48 dims × 50 folds ×
    300 epochs becomes ~300 batched steps total instead of 4200 separate
    ``_fit_mlp_ensemble_loco`` calls — the #722 vectorization win.

    ONE-SCALAR-NET-PER-DIM, NOT a fused multi-output head — DELIBERATE. The #658
    reference (``_fit_mlp_ensemble_loco``, the exactness oracle the stored numbers
    came from) fits a SEPARATE scalar net per output dim, because ``_i658._MLP``
    is a single-output ``Linear(hidden, 1)`` net. A fused ``Linear(hidden, p)``
    head would be a DIFFERENT architecture → different fit → would NOT reproduce
    the published #658 chain ρ / #722 skill numbers (the binding reproduce-check).
    So we keep the per-dim scalar member and get the vectorization purely from
    batching the (group × dim × fold) members into one vmapped ensemble. The
    member dimension is what is fused, not the output head.

    SEEDING / EXACTNESS — reproduces #658's ``_fit_mlp_ensemble_loco`` per group.
    That reference, per CALL, did ``torch.manual_seed(seed); block = [_MLP(d) for
    _ in range(n)]`` (n per-fold inits) and addressed member ``t*n + i`` by its
    block member ``i`` — so EVERY output dim ``t`` reuses the SAME ``n`` per-fold
    inits (the serial reseed-PER-DIM contract). Here each GROUP re-draws that same
    ``n``-init block from the SAME ``seed`` (one ``torch.manual_seed(seed)`` per
    group), and member ``(g, t, i)`` gets group ``g``'s block-member ``i`` for
    EVERY ``t`` — bit-identical to the reference for an identical (X, Y), verified
    ≤ 1e-6 by ``assert_matches_reference``.

    All groups MUST share n, d_in, and p (asserted). Returns a
    ``BatchedMLPResult`` with held-out (n, p) predictions per group key.

    CPU thread cap (``num_threads``, via ``_with_thread_cap``): ``None``
    (default) pins an ambient-ceilinged conservative cap for the fit and
    restores the prior pool after; ``0`` opts out; positive values pin
    verbatim during the fit (#1079).
    """
    from torch.func import stack_module_state

    if not groups:
        return BatchedMLPResult({}, 0, 0, 0, 0, chunk_size)

    n, d_in = groups[0].X.shape
    p = groups[0].Y.shape[1]
    for g in groups:
        assert g.X.shape == (n, d_in), (g.key, g.X.shape, (n, d_in))
        assert g.Y.shape == (n, p), (g.key, g.Y.shape, (n, p))
    dev = torch.device(device)
    n_groups = len(groups)
    # One scalar member per (group, target-col, fold).
    n_members = n_groups * p * n

    # Stack all groups' designs/targets: (G, n, d_in), (G, n, p).
    Xg = torch.from_numpy(
        np.ascontiguousarray(np.stack([g.X for g in groups]).astype(np.float32))
    ).to(dev)
    Yg = torch.from_numpy(
        np.ascontiguousarray(np.stack([g.Y for g in groups]).astype(np.float32))
    ).to(dev)

    # Per-group/per-dim init block: the n per-fold inits drawn ONCE from `seed`
    # (one torch.manual_seed(seed) → n _MLP inits, in fold order). Member
    # (g, t, i) reuses block member i for every (g, t) — reproducing the
    # reference's per-call reseed (per group) + per-dim reuse (within a group).
    # We read the per-member weight tensors directly off the stacked _MLP state
    # and run an explicit bmm forward (NOT vmap(functional_call), whose tracing
    # overhead per chunk dominated — bmm is the #722-rule's prescribed shape and
    # ~15× faster here). _i658._MLP is Sequential(Linear(d_in,hid), GELU,
    # Linear(hid,1)); the bmm forward x@W1ᵀ+b1 → GELU → h@W2ᵀ+b2 is bit-identical
    # to nn.Linear's own addmm (modulo the standing reduction-order residual the
    # exactness gate bounds).
    torch.manual_seed(seed)
    block_members = [_i658._MLP(d_in, hidden=hidden).to(dev) for _ in range(n)]
    block_params, _block_buffers = stack_module_state(block_members)  # leaves (n, ...)
    # (n, hid, d_in), (n, hid), (n, 1, hid), (n, 1)
    bW1 = block_params["net.0.weight"].detach()
    bb1 = block_params["net.0.bias"].detach()
    bW2 = block_params["net.2.weight"].detach()
    bb2 = block_params["net.2.bias"].detach()

    # Global member index m -> (group g, target-col t, fold i).
    # m = ((g * p) + t) * n + i.
    member_arange = torch.arange(n_members, device=dev)
    member_fold = member_arange % n  # (E,) -> fold_i == block-member id
    member_tcol = (member_arange // n) % p  # (E,) -> target column t
    member_group = member_arange // (p * n)  # (E,) -> group g

    held_all = torch.empty(n_members, device=dev, dtype=torch.float32)
    chunk = chunk_size if (chunk_size and chunk_size > 0) else n_members
    for lo in range(0, n_members, chunk):
        hi = min(lo + chunk, n_members)
        c = hi - lo
        gidx = member_arange[lo:hi]  # global member ids in this chunk
        cgroup = member_group[gidx]  # (c,) group per chunk member
        ctcol = member_tcol[gidx]  # (c,) target column per chunk member
        cfold = member_fold[gidx]  # (c,) held-out fold per chunk member

        # Per-chunk fresh LEAF weight tensors: gather each block weight by the
        # chunk members' block index (fold_i). clone so AdamW sees leaves.
        W1 = bW1.index_select(0, cfold).clone()  # (c, hid, d_in)
        b1 = bb1.index_select(0, cfold).clone()  # (c, hid)
        W2 = bW2.index_select(0, cfold).clone()  # (c, 1, hid)
        b2 = bb2.index_select(0, cfold).clone()  # (c, 1)

        # This chunk's per-member design (gather by group) + scalar target
        # (gather by group then select the member's target column).
        Xc = Xg[cgroup]  # (c, n, d_in)
        Yc_full = Yg[cgroup]  # (c, n, p)
        yc = Yc_full.gather(2, ctcol.view(c, 1, 1).expand(c, n, 1)).squeeze(2)  # (c, n)

        # LOO train mask: drop the member's held-out fold row.
        train_mask = torch.ones((c, n), dtype=torch.bool, device=dev)
        train_mask[torch.arange(c, device=dev), cfold] = False
        mask_f = train_mask.to(torch.float32)  # (c, n)
        counts = mask_f.sum(1, keepdim=True)  # (c, 1) == n-1

        # Per-member feature standardization on train rows ONLY (no leakage),
        # matching the serial `mu, sd = X[mask].mean(0), X[mask].std(0)+1e-6`.
        # torch .std default is ddof=1 (unbiased) — reproduce via the SS form.
        mu = (mask_f.unsqueeze(2) * Xc).sum(1) / counts  # (c, d_in)
        sumsq = (mask_f.unsqueeze(2) * (Xc * Xc)).sum(1)  # (c, d_in)
        var = (sumsq - counts * mu * mu) / (counts - 1.0).clamp(min=1.0)
        sd = var.clamp(min=0.0).sqrt() + 1e-6  # (c, d_in)
        Xn = (Xc - mu.unsqueeze(1)) / sd.unsqueeze(1)  # (c, n, d_in)

        for w in (W1, b1, W2, b2):
            w.requires_grad_(True)
        opt = torch.optim.AdamW([W1, b1, W2, b2], lr=lr, weight_decay=wd)
        denom = mask_f.sum(1).clamp(min=1.0)  # (c,) == n-1
        for _ in range(max_epochs):
            opt.zero_grad(set_to_none=True)
            # bmm forward: (c, n, d_in) @ (c, d_in, hid) + (c, 1, hid) → GELU →
            # (c, n, hid) @ (c, hid, 1) + (c, 1, 1) → squeeze → (c, n).
            h = torch.nn.functional.gelu(
                torch.bmm(Xn, W1.transpose(1, 2)) + b1.unsqueeze(1)
            )  # (c, n, hid)
            pred = (torch.bmm(h, W2.transpose(1, 2)) + b2.unsqueeze(1)).squeeze(2)  # (c, n)
            sq = (pred - yc) ** 2 * mask_f  # (c, n)
            per_member = sq.sum(1) / denom  # (c,)
            loss = per_member.sum()
            loss.backward()
            opt.step()

        with torch.no_grad():
            h = torch.nn.functional.gelu(torch.bmm(Xn, W1.transpose(1, 2)) + b1.unsqueeze(1))
            pred = (torch.bmm(h, W2.transpose(1, 2)) + b2.unsqueeze(1)).squeeze(2)  # (c, n)
        held_all[lo:hi] = pred[torch.arange(c, device=dev), cfold]  # (c,) held-out row
        del W1, b1, W2, b2, Xn, Xc, Yc_full, yc, opt, pred, h, sq, per_member, mu, sd, var, sumsq

    # reshape (E,) -> (n_groups, p, n) -> per-group (n, p)
    held = held_all.reshape(n_groups, p, n).permute(0, 2, 1).contiguous()
    held_np = held.detach().cpu().numpy().astype(np.float64)
    preds_by_key = {groups[g].key: held_np[g] for g in range(n_groups)}
    return BatchedMLPResult(
        preds_by_key=preds_by_key,
        n_groups=n_groups,
        n_folds=n,
        pca_target_dim=p,
        n_members=n_members,
        chunk_size=chunk,
    )


class _MultiHeadMLP(torch.nn.Module):
    """The multi-output LOCO fold net: ``Linear(d_in, hidden) → GELU → Linear(hidden, p)``.

    Module-level (not a closure) so the SERIAL parity reference
    (``_serial_group_mlp_reference``) and the batched path
    (``fit_batched_loco_mlp_multihead``) draw bit-identical per-fold inits from
    the same ``torch.manual_seed(seed)`` stream — the module structure defines
    the RNG consumption order, so both paths MUST construct the same class.
    """

    def __init__(self, d_in: int, hidden: int, p: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fold_order_and_rows(row_groups: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """Sorted-unique fold labels + their row-index arrays (the group-fold grain).

    Fold ORDER is ``sorted(np.unique(row_groups))`` — for the #928 store's
    contiguous 0..n_ctx−1 context labels this is battery order, matching
    ``issue928_null_bootstrap.group_folds(groups, list(range(n_ctx)))``.
    Every fold must be non-empty by construction of ``np.unique``.
    """
    order = [int(g) for g in np.unique(row_groups)]
    rows = [np.flatnonzero(row_groups == g) for g in order]
    return order, rows


@_with_thread_cap
def fit_batched_loco_mlp_multihead(
    groups: list[MLPGroup],
    *,
    seed: int = DEFAULT_MLP_SEED,
    hidden: int = MLP_HIDDEN,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = MLP_MAX_EPOCHS,
    device: str = "cpu",
    chunk_size: int = 4096,
    num_threads: int | None = None,
    row_groups: np.ndarray | None = None,
    standardization: str = "per_fold",
) -> BatchedMLPResult:
    """Multi-output-head batched LOCO MLP — the FAST production path.

    The vectorize-many-cell-fits rule's prescribed shape: ONE width-``hidden``
    net per (group, fold) predicting ALL ``p`` PCA target dims jointly
    (``Linear(d_in, hidden) → GELU → Linear(hidden, p)``), so the ensemble is
    ``E = n_groups * n`` members — a **48× reduction** vs the per-dim scalar path
    (``fit_batched_loco_mlp``) at p=48. THIS is what makes the #722 sweep finish
    in minutes: the 19-TFLOP / "minutes" budget in the rule assumes the
    multi-output head (the scalar-per-dim path is ~p× more FLOPs and stays a
    multi-hour job on CPU).

    DIFFERS from the slow scalar-per-dim reference by ARCHITECTURE (shared trunk
    + joint p-output head vs p independent scalar nets), so it does NOT reproduce
    the stored scalar-per-dim numbers bit-for-bit — the gap is quantified on spot
    layers by the consumer (``issue722_vectorized_skill``). The per-(group, fold)
    member init is drawn the same way (n per-fold inits from ``seed``, reused per
    group), and per-fold train-only standardization is identical. Returns
    held-out (n, p) predictions per group key.

    bmm forward: x@W1ᵀ+b1 → GELU → h@W2ᵀ+b2, with W2 (member, p, hidden) the
    multi-output head.

    GROUP FOLDS (``row_groups``, #928 group-LOCO extension). ``row_groups=None``
    (default) is the existing singleton-LOCO behavior byte-for-byte: fold i
    holds out row i, ``E = n_groups * n``. With an (n,) integer label array,
    fold *i* holds out ALL rows of the i-th SORTED-UNIQUE label together
    (never pointwise — `.claude/rules/ood-generalization-folds.md`), so
    ``E = n_groups * n_folds``; the label array is shared by every group cell
    in the call (all cells share rows). The per-fold init block draws
    ``n_folds`` nets from ``torch.manual_seed(seed)`` (== the old ``n`` draws
    when folds are singletons, so the None path consumes the identical RNG
    stream). Mirrors ``issue928_null_bootstrap.group_folds`` fold semantics;
    validated against a same-seed serial per-fold reference by
    ``assert_group_mlp_matches_serial`` (both standardization modes).

    STANDARDIZATION (``standardization``, mirrors ``GroupRidgeDesign``):
    ``"per_fold"`` (default — the existing behavior, the #658 MLP convention:
    train-only column mu/sd per fold, torch ``.std`` ddof=1, +1e-6) or
    ``"full_data"`` (X standardized ONCE on all n rows per group cell with the
    indiv LINEAR arms' realized convention — ddof=0 ``correction=0`` + 1e-9,
    exactly ``GroupRidgeDesign(standardization="full_data")`` — so the #928
    per-question MLP arms share the linear arms' input convention and the
    estimator functional form stays the single manipulated variable).

    CHUNK SIZE DOES NOT CHANGE THE FIT. Held-out predictions are per-member
    independent and every per-member quantity (init, train-only standardization,
    held-out row) is keyed to the GLOBAL member index (``member_arange`` /
    ``member_fold`` / ``member_group``), never the chunk-local position — so
    ``chunk_size`` only bounds peak memory, not the result (pinned by the
    chunk-size-invariance test). ``chunk_size`` is a REQUEST: it is capped down by
    ``resolve_chunk_cap`` when a chunk's live ``(c, n, d_in)`` fp32 intermediates
    would not fit free memory (#811: the default 4096, sized for #722's n≈50,
    materializes a 26.25 GiB (c, n, d_in) fp32 tensor at #811's n=480 × d_in=3584 →
    OOM). The masked train-row moments are computed via ``bmm`` (no ``(c, n, d_in)``
    broadcast temp); the bmm reduction order differs from a broadcast-sum in fp32
    associativity, benign because the validity gate is a real-vs-shuffle comparison
    that shares this path on both arms.

    CPU thread cap (``num_threads``, via ``_with_thread_cap``): ``None``
    (default) pins an ambient-ceilinged conservative cap for the fit and
    restores the prior pool after; ``0`` opts out; positive values pin
    verbatim during the fit (#1079).
    """
    from torch.func import stack_module_state

    assert standardization in ("per_fold", "full_data"), standardization
    if not groups:
        return BatchedMLPResult({}, 0, 0, 0, 0, chunk_size)

    n, d_in = groups[0].X.shape
    p = groups[0].Y.shape[1]
    for g in groups:
        assert g.X.shape == (n, d_in), (g.key, g.X.shape, (n, d_in))
        assert g.Y.shape == (n, p), (g.key, g.Y.shape, (n, p))
    dev = torch.device(device)
    n_groups = len(groups)

    # Fold grain: singleton rows (None — the byte-for-byte legacy path) or the
    # sorted-unique labels of ``row_groups`` (whole groups leave together).
    if row_groups is None:
        labels = np.arange(n, dtype=np.int64)
    else:
        labels = np.asarray(row_groups, dtype=np.int64)
        assert labels.shape == (n,), (labels.shape, n)
    _order, fold_rows = _fold_order_and_rows(labels)
    n_folds = len(fold_rows)
    max_fold = max(r.size for r in fold_rows)
    assert n - max_fold >= 2, (
        f"largest fold holds {max_fold} of {n} rows — <2 train rows breaks ddof-1 standardization"
    )
    n_members = n_groups * n_folds  # ONE multi-output net per (group, fold)

    Xg = torch.from_numpy(
        np.ascontiguousarray(np.stack([g.X for g in groups]).astype(np.float32))
    ).to(dev)
    Yg = torch.from_numpy(
        np.ascontiguousarray(np.stack([g.Y for g in groups]).astype(np.float32))
    ).to(dev)

    # (n_folds, n) bool: row r is a TRAIN row of fold f. For singleton folds this
    # is ~eye(n) — indexing it by ``cfold`` reproduces the legacy per-member mask
    # construction exactly (bool-identical → float-identical moments/loss).
    fold_train = torch.ones((n_folds, n), dtype=torch.bool, device=dev)
    for f, rows in enumerate(fold_rows):
        fold_train[f, torch.as_tensor(rows, device=dev)] = False

    # Full-data standardization: ONE mu/sd per group cell over ALL n rows, the
    # GroupRidgeDesign full_data convention (ddof=0 + 1e-9), precomputed once.
    Xn_full = None
    if standardization == "full_data":
        mu_full = Xg.mean(dim=1)  # (G, d_in)
        sd_full = Xg.std(dim=1, correction=0) + 1e-9  # (G, d_in)
        Xn_full = (Xg - mu_full.unsqueeze(1)) / sd_full.unsqueeze(1)  # (G, n, d_in)

    # n_folds per-fold inits for the multi-output net (Linear(d_in,hid) + Linear(hid,p)).
    torch.manual_seed(seed)
    block_members = [_MultiHeadMLP(d_in, hidden, p).to(dev) for _ in range(n_folds)]
    bp, _bb = stack_module_state(block_members)
    bW1 = bp["net.0.weight"].detach()  # (n_folds, hid, d_in)
    bb1 = bp["net.0.bias"].detach()  # (n_folds, hid)
    bW2 = bp["net.2.weight"].detach()  # (n_folds, p, hid)
    bb2 = bp["net.2.bias"].detach()  # (n_folds, p)

    member_arange = torch.arange(n_members, device=dev)
    member_fold = member_arange % n_folds  # (E,) -> fold_i == block-member id
    member_group = member_arange // n_folds  # (E,) -> group g

    # Held-out predictions written PER ROW (each row belongs to exactly one fold,
    # so every (group, row) slot is written exactly once across the members).
    held_by_row = torch.empty(n_groups, n, p, device=dev, dtype=torch.float32)
    # Memory-aware chunk cap. CHUNK SIZE DOES NOT CHANGE RESULTS — held-out
    # predictions are per-member independent, and every per-member quantity (init,
    # train-only standardization, held-out row) is keyed to the GLOBAL member index
    # (member_arange / member_fold / member_group), never the chunk-local position
    # (verified BIT-identical by assert_matches_reference for the sibling scalar
    # path; same global-index keying here). So capping the chunk to fit memory only
    # changes peak footprint, not the fit. #811: the default 4096 was sized for
    # #722's n≈50; #811's phase0 gate is n=480 × d_in=3584, where c=4096 alone is a
    # 26.25 GiB (c, n, d_in) fp32 intermediate → OOM.
    requested = chunk_size if (chunk_size and chunk_size > 0) else n_members
    free_bytes = _free_bytes(dev)  # probe ONCE; the log below must show the value the cap used
    chunk = resolve_chunk_cap(requested, n_members, n, d_in, free_bytes)
    if chunk < requested:
        logger.info(
            "[vectorized_mlp] chunk capped %d -> %d (n=%d d_in=%d free=%.2f GiB) to bound "
            "per-chunk (c, n, d_in) fp32 footprint",
            requested,
            chunk,
            n,
            d_in,
            free_bytes / 2**30,
        )
    for lo in range(0, n_members, chunk):
        hi = min(lo + chunk, n_members)
        gidx = member_arange[lo:hi]
        cgroup = member_group[gidx]
        cfold = member_fold[gidx]

        W1 = bW1.index_select(0, cfold).clone()  # (c, hid, d_in)
        b1 = bb1.index_select(0, cfold).clone()  # (c, hid)
        W2 = bW2.index_select(0, cfold).clone()  # (c, p, hid)
        b2 = bb2.index_select(0, cfold).clone()  # (c, p)

        Yc = Yg[cgroup]  # (c, n, p)

        train_mask = fold_train.index_select(0, cfold)  # (c, n) bool
        mask_f = train_mask.to(torch.float32)  # (c, n)
        counts = mask_f.sum(1, keepdim=True)

        if standardization == "full_data":
            Xn = Xn_full[cgroup]  # (c, n, d_in) — the one shared full-data standardization
        else:
            Xc = Xg[cgroup]  # (c, n, d_in)
            # Masked train-row moments via bmm — NO (c, n, d_in) broadcast temp.
            # mu = mask · Xc / counts; sumsq = mask · Xc². bmm reduction ORDER differs
            # from the old `(mask.unsqueeze(2) * Xc).sum(1)` broadcast-sum (fp32
            # associativity), but the gate is a real-vs-shuffle comparison where both
            # arms share this code path, so the tiny residual is common-mode and benign.
            Xc2 = Xc * Xc  # (c, n, d_in) — the one live square temp, freed below
            mu = torch.bmm(mask_f.unsqueeze(1), Xc).squeeze(1) / counts  # (c, d_in)
            sumsq = torch.bmm(mask_f.unsqueeze(1), Xc2).squeeze(1)  # (c, d_in)
            del Xc2
            var = (sumsq - counts * mu * mu) / (counts - 1.0).clamp(min=1.0)
            sd = var.clamp(min=0.0).sqrt() + 1e-6
            Xn = (Xc - mu.unsqueeze(1)) / sd.unsqueeze(1)  # (c, n, d_in)
            del Xc

        for w in (W1, b1, W2, b2):
            w.requires_grad_(True)
        opt = torch.optim.AdamW([W1, b1, W2, b2], lr=lr, weight_decay=wd)
        denom = mask_f.sum(1).clamp(min=1.0)  # (c,)
        for _ in range(max_epochs):
            opt.zero_grad(set_to_none=True)
            h = torch.nn.functional.gelu(torch.bmm(Xn, W1.transpose(1, 2)) + b1.unsqueeze(1))
            pred = torch.bmm(h, W2.transpose(1, 2)) + b2.unsqueeze(1)  # (c, n, p)
            # masked per-member MSE over train rows, averaged over p outputs.
            sq = ((pred - Yc) ** 2).mean(dim=2) * mask_f  # (c, n)
            loss = (sq.sum(1) / denom).sum()
            loss.backward()
            opt.step()

        with torch.no_grad():
            h = torch.nn.functional.gelu(torch.bmm(Xn, W1.transpose(1, 2)) + b1.unsqueeze(1))
            pred = torch.bmm(h, W2.transpose(1, 2)) + b2.unsqueeze(1)  # (c, n, p)
        # Scatter each member's HELD rows into its group cell's (n, p) slab. Folds
        # partition the rows, so every (group, row) slot is written exactly once;
        # for singleton folds this reduces to the legacy one-row-per-member write.
        held_mask = ~train_mask  # (c, n) bool — the member's held rows
        midx, ridx = held_mask.nonzero(as_tuple=True)
        held_by_row[cgroup[midx], ridx] = pred[midx, ridx]
        del W1, b1, W2, b2, Xn, Yc, opt, pred, h, sq

    held_np = held_by_row.detach().cpu().numpy().astype(np.float64)
    preds_by_key = {groups[g].key: held_np[g] for g in range(n_groups)}
    return BatchedMLPResult(
        preds_by_key=preds_by_key,
        n_groups=n_groups,
        n_folds=n_folds,
        pca_target_dim=p,
        n_members=n_members,
        chunk_size=chunk,
    )


# ── group-fold MLP serial-parity gate (#928 indiv MLP control; vectorize item 6) ──


def _serial_group_mlp_reference(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    hidden: int,
    lr: float,
    wd: float,
    max_epochs: int,
    standardization: str,
) -> np.ndarray:
    """Serial per-fold multihead-MLP oracle — the SLOW obvious implementation.

    One ``_MultiHeadMLP`` per fold, trained on the fold's TRAIN rows only with
    ``AdamW(lr, wd)`` for ``max_epochs`` full-batch epochs on the mean-over-p /
    mean-over-train-rows MSE (the same per-member objective the batched loss
    sums). Per-fold inits are the SAME draws the batched path uses: ONE
    ``torch.manual_seed(seed)`` then ``n_folds`` sequential ``_MultiHeadMLP``
    constructions, member *i* = fold *i*. Standardization mirrors the batched
    modes: ``per_fold`` = train-only mu / ``.std()`` (ddof=1) + 1e-6;
    ``full_data`` = all-rows mu / ``.std(correction=0)`` + 1e-9. Used ONLY by
    ``assert_group_mlp_matches_serial`` on tiny cells (mirrors
    ``issue928_null_bootstrap._serial_group_ridge_reference``). Returns the
    held-out (n, p) predictions in row order.
    """
    n, d_in = X.shape
    p = Y.shape[1]
    _order, fold_rows = _fold_order_and_rows(np.asarray(labels, dtype=np.int64))
    torch.manual_seed(seed)
    nets = [_MultiHeadMLP(d_in, hidden, p) for _ in range(len(fold_rows))]
    Xt = torch.from_numpy(np.ascontiguousarray(X.astype(np.float32)))
    Yt = torch.from_numpy(np.ascontiguousarray(Y.astype(np.float32)))
    if standardization == "full_data":
        mu_all = Xt.mean(0)
        sd_all = Xt.std(0, correction=0) + 1e-9
    preds = np.zeros((n, p), dtype=np.float64)
    for f, held in enumerate(fold_rows):
        tr = np.setdiff1d(np.arange(n), held)
        Xtr = Xt[tr]
        if standardization == "per_fold":
            mu = Xtr.mean(0)
            sd = Xtr.std(0) + 1e-6  # torch .std default ddof=1 — the #658 MLP convention
        else:
            mu, sd = mu_all, sd_all
        Xn_tr = (Xtr - mu) / sd
        Xn_hd = (Xt[held] - mu) / sd
        net = nets[f]
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        Ytr = Yt[tr]
        for _ in range(max_epochs):
            opt.zero_grad(set_to_none=True)
            loss = ((net(Xn_tr) - Ytr) ** 2).mean(dim=1).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds[held] = net(Xn_hd).numpy().astype(np.float64)
    return preds


def assert_group_mlp_matches_serial(
    seed: int = DEFAULT_MLP_SEED,
    atol: float = 5e-5,
    max_epochs: int = MLP_MAX_EPOCHS,
    hidden: int = MLP_HIDDEN,
) -> dict:
    """Seeded serial-parity gate for the GROUP-fold batched multihead MLP.

    Mirrors ``issue928_null_bootstrap.assert_group_ridge_matches_serial`` on the
    MLP path (#928 indiv nonlinearity control, plan §4.1a). Checks, in BOTH
    standardization modes on small synthetic cells:

    1. **Multi-row-group parity** — ``fit_batched_loco_mlp_multihead`` with
       ``row_groups`` (12 rows, 4 groups × 3, d_in=6, p=3; two group cells so
       the member-indexing crosses cells) vs the same-seed serial per-fold
       reference ``_serial_group_mlp_reference``.
    2. **Singleton parity** — ``row_groups=None`` (the legacy byte-for-byte
       path) vs the serial reference with singleton labels (per_fold mode: the
       pre-existing behavior; full_data mode: the new branch at fold grain 1).
    3. **Determinism** — a duplicate batched fit reproduces itself EXACTLY
       (bitwise; catches nondeterministic kernels before a production fit).

    Tolerance: fp32 GD over ``max_epochs`` epochs accumulates benign
    reduction-order residuals between the batched ``bmm`` and the serial
    ``nn.Linear`` kernels (the standing residual the module's exactness notes
    bound at ≤1e-6 for the scalar path). Measured at PRODUCTION epochs (300,
    CPU, 2026-07-04): max deviation 9.5e-07 across all checks/modes;
    ``atol=5e-5`` gives ~50× headroom above that while staying ~4 orders below
    the ~1e-1 skill-scale a real fold-leakage / standardization-mixing bug
    produces. Runs on CPU (deterministic); returns the max-abs deviation per
    check and raises ``AssertionError`` on any breach. Run before any
    production group-fold MLP fit (vectorize-many-cell-fits item 6).

    Thread cap: the gate's ops are TINY (12×6 tensors), so a wide torch pool
    thrashes on op dispatch (vectorize-rule item 4 — measured: 265 s at 2
    threads vs >480 s at 8 on the same box); the gate pins 2 threads for its
    own duration and restores the caller's setting.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        return _gate_body(seed=seed, atol=atol, max_epochs=max_epochs, hidden=hidden)
    finally:
        torch.set_num_threads(prev_threads)


def _gate_body(*, seed: int, atol: float, max_epochs: int, hidden: int) -> dict:
    """The ``assert_group_mlp_matches_serial`` checks (thread cap applied by the wrapper)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n, d, p, gsz = 12, 6, 3, 3
    labels = np.repeat(np.arange(n // gsz), gsz)
    cells = [
        MLPGroup(("gateA", 0), rng.standard_normal((n, d)), rng.standard_normal((n, p))),
        MLPGroup(("gateB", 1), rng.standard_normal((n, d)), rng.standard_normal((n, p))),
    ]
    kw = dict(seed=seed, hidden=hidden, lr=MLP_LR, wd=MLP_WD, max_epochs=max_epochs)
    # num_threads=0 on each batched fit below: the wrapper's 2-thread pin OWNS
    # the pool for the whole gate body — the fitters' #1079 default cap must
    # never fire inside it (recorder-certified by
    # tests/test_vectorized_fit_thread_cap.py). Not in ``kw``: the serial
    # reference shares ``kw`` and takes no ``num_threads``.
    for mode in ("per_fold", "full_data"):
        # 1. multi-row groups, two cells, chunk crossing the cell boundary.
        res = fit_batched_loco_mlp_multihead(
            cells,
            device="cpu",
            chunk_size=3,
            row_groups=labels,
            standardization=mode,
            num_threads=0,
            **kw,
        )
        for cell in cells:
            ref = _serial_group_mlp_reference(cell.X, cell.Y, labels, standardization=mode, **kw)
            dev = float(np.max(np.abs(res.preds_by_key[cell.key] - ref)))
            assert dev < atol, f"group MLP parity breach ({mode}, {cell.key}): {dev} >= {atol}"
            out[f"group_vs_serial_{mode}_{cell.key[0]}"] = dev
        # 2. singleton folds (row_groups=None — the legacy path) vs the serial
        # reference at singleton labels.
        res_s = fit_batched_loco_mlp_multihead(
            [cells[0]], device="cpu", chunk_size=5, standardization=mode, num_threads=0, **kw
        )
        ref_s = _serial_group_mlp_reference(
            cells[0].X, cells[0].Y, np.arange(n), standardization=mode, **kw
        )
        dev_s = float(np.max(np.abs(res_s.preds_by_key[cells[0].key] - ref_s)))
        assert dev_s < atol, f"singleton MLP parity breach ({mode}): {dev_s} >= {atol}"
        out[f"singleton_vs_serial_{mode}"] = dev_s
        # 3. duplicate-fit determinism (bitwise) on the grouped call.
        res_dup = fit_batched_loco_mlp_multihead(
            cells,
            device="cpu",
            chunk_size=3,
            row_groups=labels,
            standardization=mode,
            num_threads=0,
            **kw,
        )
        for cell in cells:
            assert np.array_equal(res.preds_by_key[cell.key], res_dup.preds_by_key[cell.key]), (
                f"duplicate-fit nondeterminism ({mode}, {cell.key})"
            )
        out[f"duplicate_fit_bitwise_{mode}"] = 0.0
    return out


# ── PCA basis + skill-over-mean R² ────────────────────────────────────────────


def robust_pca_basis(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """PCA mean + top-k right singular vectors (gesdd→gesvd fallback).

    Mirrors ``issue722_skill_over_mean._robust_pca_basis`` /
    ``issue658_mlp_chain._pca_basis_v0`` so the like-for-like target reduction
    matches the existing scripts byte-for-byte. Returns (mu (H,), comps (k', H),
    used_gesvd_fallback) with k' = min(k, rank). The numpy gesdd path is the
    default; the scipy/torch gesvd fallback handles the near-singular layers
    that crashed #722.
    """
    mu = Y.mean(axis=0)
    Yc = Y - mu
    fallback = False
    try:
        _, _, Vt = np.linalg.svd(Yc, full_matrices=False)  # gesdd
    except np.linalg.LinAlgError:
        _, _, Vh = torch.linalg.svd(torch.from_numpy(Yc), full_matrices=False)  # gesvd
        Vt = Vh.numpy()
        fallback = True
    kk = min(k, Vt.shape[0])
    return mu, Vt[:kk], fallback


def loco_train_means(Y: np.ndarray) -> np.ndarray:
    """Per-fold leave-one-out train means: row i = mean of all rows except i.

    (n, P). The predict-the-mean baseline per LOCO fold (no held-out leakage).
    """
    n = Y.shape[0]
    total = Y.sum(axis=0, keepdims=True)
    return (total - Y) / (n - 1)


def skill_over_mean_r2(preds: np.ndarray, Y: np.ndarray) -> dict:
    """Variance-weighted aggregate held-out R² over the TRAIN-MEAN-CENTERED target.

    The #722 skill-over-mean metric: skill = 1 − SS_res/SS_tot where
    ``SS_res = Σ_i ‖Y_i − preds_i‖²`` and ``SS_tot = Σ_i ‖Y_i − ȳ_train(i)‖²``
    with ``ȳ_train(i)`` the per-fold LOO train mean. skill ≈ 0 / negative ⇒ the
    map predicts no better than the across-context average; skill > 0 ⇒ the
    context carries real, generalizing information past the anisotropic mean.
    ``preds`` are the UN-centered held-out predictions (n, P).
    """
    n = Y.shape[0]
    tmean = loco_train_means(Y)
    ss_res = float(np.sum((Y - preds) ** 2))
    ss_tot = float(np.sum((Y - tmean) ** 2))
    skill = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    # per-dim R² median, finite only (the #722 reported median_per_dim_r2)
    per_dim_res = np.sum((Y - preds) ** 2, axis=0)
    per_dim_tot = np.sum((Y - tmean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_r2 = 1.0 - per_dim_res / per_dim_tot
    per_dim_r2 = per_dim_r2[np.isfinite(per_dim_r2)]
    return {
        "skill": skill,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "n_folds_used": n,
        "median_per_dim_r2": float(np.median(per_dim_r2)) if per_dim_r2.size else float("nan"),
    }


# ── ridge LOCO predictor (skill form), reuses #658's exact dual/PRESS math ─────


def ridge_predict_loco_centered(Xc: np.ndarray, Yv: np.ndarray) -> np.ndarray:
    """LOCO ridge prediction of Yv from Xc on the TRAIN-MEAN-CENTERED target.

    The #722 ``_ridge_skill`` prediction: per held-out context, train-only X
    standardization + train-only target centering + #658's exact dual/PRESS
    nested-CV λ pick (``_press_loo_mse_per_lambda`` / ``_ridge_dual_weights`` /
    ``RIDGE_LAMBDAS``), prediction = ``v̄0_train + M̂(c_C)`` (add the train mean
    back). Returns the UN-centered (n, H) held-out predictions. Deterministic +
    exact (closed form IS the refit). Reused for BOTH the #722 ridge skill arm
    and the #658 ridge chain control.
    """
    n = Xc.shape[0]
    device = torch.device(_i658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(Xc)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Yv)).to(device=device, dtype=torch.float64)
    preds = np.zeros_like(Yv, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        tr_t = torch.tensor(tr, device=device)
        Xtr, Ytr = Xt[tr_t], Yt[tr_t]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9  # numpy ddof=0 convention (#658)
        Xtr_n = (Xtr - xmu) / xsd
        ymu = Ytr.mean(0)  # train predict-the-mean baseline
        Ytr_c = Ytr - ymu
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr_c, RIDGE_LAMBDAS)
        best_lam = RIDGE_LAMBDAS[int(torch.argmin(mse).item())]
        w = _ridge_dual_weights(Xtr_n, Ytr_c, best_lam)
        x_held = (Xt[i] - xmu) / xsd
        preds[i] = (ymu + x_held @ w).detach().cpu().numpy()
    return preds


def ridge_predict_loco_raw(Xc: np.ndarray, Yv: np.ndarray) -> np.ndarray:
    """LOCO ridge prediction of the RAW (uncentered) Yv — #658's exact chain path.

    Thin wrapper over #658 ``_ridge_predict_loco`` (no target centering) so the
    #658 ridge full-H chain control reproduces byte-for-byte.
    """
    return _i658._ridge_predict_loco(Xc, Yv, RIDGE_LAMBDAS)


# ── downstream chain ρ ────────────────────────────────────────────────────────


def chain_rho_from_pred_v0(
    pred_v0_by_layer: dict, r_b_by_layer: dict, y_e0: np.ndarray, kept_idx: list[int]
) -> dict | None:
    """Best-layer Spearman(r_Bᵀ (pred v0), E0) — #658 ``_chain_rho`` selection.

    pred_v0_by_layer: {layer_idx: (n_ctx, H)} held-out predicted v0 per layer.
    r_b_by_layer: {layer_idx: (H,)} the behavior's diff-of-means r_B per layer.
    y_e0: (n_kept,) measured E0 for this behavior. kept_idx: rows of the
    n_ctx prediction that have an E0 value. Picks the layer with the LARGEST ρ
    (NOT |ρ|), matching ``issue658_fit_predictors._chain_rho``. Returns
    {"layer": L, "rho": ρ} or None.
    """
    best = None
    for li, pred_v0 in pred_v0_by_layer.items():
        r = r_b_by_layer.get(li)
        if r is None:
            continue
        chain_pred = pred_v0[kept_idx] @ r
        rho = _i658._rho(chain_pred, y_e0)
        if rho is not None and (best is None or rho > best["rho"]):
            best = {"layer": li, "rho": rho}
    return best


def chain_rho_pca(
    pred_pca: np.ndarray, pca_basis: np.ndarray, r: np.ndarray, y_e0: np.ndarray
) -> tuple[float | None, np.ndarray]:
    """Spearman(r_Bᵀ (back-projected PCA pred), E0) — #658 ``_chain_rho_pca``.

    ``pred_pca`` (n, k) LOCO held-out predictions in the PCA space; project back
    to H (``pred_pca @ pca_basis``), dot the full-H ``r``, Spearman vs E0.
    """
    pred_full = pred_pca @ pca_basis  # (n, H)
    chain = pred_full @ r  # (n,)
    return _i658._rho(chain, y_e0), chain


# ── exactness reproduce-check against the #658 serial-equivalent oracle ────────


def assert_matches_reference(seed: int = 0, n: int = 14, d: int = 24, p: int = 4) -> dict:
    """Assert the batched fit reproduces #658's ``_fit_mlp_ensemble_loco`` to the
    documented small-N reduction-order tolerance, AND that chunking is EXACT.

    Three checks on a small synthetic (X, Y) with real rank structure (clamped to
    20 epochs for a fast ~sub-minute gate):

    (a) **Per-member reproduction** — the batched fit of group 0 = (X, Y) and
        group 1 = (X, Y[perm]) (a shuffle null) in ONE ensemble must reproduce
        each group's SINGLE-group reference ``_fit_mlp_ensemble_loco`` to
        ``tol``. The residual is purely batched-GEMM-vs-per-net reduction order:
        the SAME residual appears for a single group with no chunking (verified),
        and it amplifies with poorer per-net conditioning at small N exactly as
        #658's ``_assert_mlp_exactness`` documents ("~1e-3 at the pathological
        N=4 smoke"). At this N=14 gate scale the shuffle-null target lands a
        per-prediction residual ~2.7e-6 — optimization-noise-level for an MLP
        whose held-out prediction is an ESTIMATE — so ``tol = 5e-6`` here. The
        PRODUCTION read is N=50 (well-conditioned), where the residual is
        smaller; the binding science check is the per-genre RIDGE reproduce-check
        (byte-exact, 1e-6) + the slow-MLP spot-check (Deliverable 4), not this
        synthetic gate.

    (b) **Chunk-invariance (EXACT)** — the chunked (non-divisor chunk) result vs
        the no-chunk (``chunk_size=0``) result must be BIT-identical
        (``np.array_equal``). This is the real logic-correctness invariant: it
        proves every per-member quantity (init, standardization, target column,
        held-out row) is keyed to the GLOBAL member index, never the chunk
        position. A regression here (a per-member quantity keyed to chunk-local
        position) is caught with zero tolerance even though (a) is loose.

    (c) **Cross-group non-contamination** — the 2-group base prediction must be
        BIT-identical to the single-group base prediction (``np.array_equal``):
        adding the shuffle group to the batch must not perturb the base group's
        members at all.

    Forces DEVICE=cpu for both paths so reduction order is comparable.
    """
    saved_device, saved_epochs = _i658.DEVICE, _i658.MLP_MAX_EPOCHS
    _i658.DEVICE = "cpu"
    _i658.MLP_MAX_EPOCHS = 20  # clamp for a fast gate
    try:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, 3))
        W = rng.standard_normal((3, d))
        X = (z @ W + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
        B = rng.standard_normal((d, p))
        Y = (X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))).astype(np.float32)
        perm = rng.permutation(n)
        Ysh = Y[perm]

        # Reference (per-call vmapped ensemble), single-output-set multi-target.
        ref = _i658._fit_mlp_ensemble_loco(X, Y, target_idx=list(range(p)), seed=658)
        ref_sh = _i658._fit_mlp_ensemble_loco(X, Ysh, target_idx=list(range(p)), seed=658)

        # (a) Batched fit: two groups in ONE ensemble (base + shuffle null),
        # chunked at a non-divisor of E.
        groups = [MLPGroup(("base",), X, Y), MLPGroup(("shuffle",), X, Ysh)]
        e_total = 2 * p * n
        chunk = max(1, e_total // 3 + 1)  # a non-divisor of E to exercise chunking
        # num_threads=0 on every gate fit: ambient pool (the pre-#1079 behavior).
        res = fit_batched_loco_mlp(
            groups, seed=658, max_epochs=20, device="cpu", chunk_size=chunk, num_threads=0
        )
        d_base = float(np.max(np.abs(res.preds_by_key[("base",)] - ref)))
        d_shuf = float(np.max(np.abs(res.preds_by_key[("shuffle",)] - ref_sh)))

        # (b) chunk-invariance: no-chunk must be BIT-identical to the chunked run.
        res_nochunk = fit_batched_loco_mlp(
            groups, seed=658, max_epochs=20, device="cpu", chunk_size=0, num_threads=0
        )
        chunk_base_identical = bool(
            np.array_equal(res.preds_by_key[("base",)], res_nochunk.preds_by_key[("base",)])
        )
        chunk_shuf_identical = bool(
            np.array_equal(res.preds_by_key[("shuffle",)], res_nochunk.preds_by_key[("shuffle",)])
        )

        # (c) cross-group non-contamination: 2-group base == single-group base.
        res_single = fit_batched_loco_mlp(
            [MLPGroup(("base",), X, Y)],
            seed=658,
            max_epochs=20,
            device="cpu",
            chunk_size=0,
            num_threads=0,
        )
        crossgroup_identical = bool(
            np.array_equal(res_nochunk.preds_by_key[("base",)], res_single.preds_by_key[("base",)])
        )
    finally:
        _i658.DEVICE, _i658.MLP_MAX_EPOCHS = saved_device, saved_epochs
    tol = 5e-6  # documented small-N batched-GEMM reduction-order residual (#658)
    assert d_base <= tol, (
        f"batched MLP exactness FAILED (base group): max|Δpred|={d_base:.3e} > {tol} "
        "vs _fit_mlp_ensemble_loco — the batched per-member contract drifted from "
        "the #658 reference (seeding / standardization)"
    )
    assert d_shuf <= tol, (
        f"batched MLP exactness FAILED (shuffle group): max|Δpred|={d_shuf:.3e} > {tol} "
        "vs _fit_mlp_ensemble_loco"
    )
    assert chunk_base_identical and chunk_shuf_identical, (
        "batched MLP exactness FAILED (chunk-invariance): chunked result is not "
        "BIT-identical to the no-chunk result — a per-member quantity is keyed to "
        "chunk-local position, not the global member index"
    )
    assert crossgroup_identical, (
        "batched MLP exactness FAILED (cross-group contamination): adding a second "
        "group to the batch perturbed the first group's predictions"
    )
    return {
        "base_delta": d_base,
        "shuffle_delta": d_shuf,
        "chunk_invariant": chunk_base_identical and chunk_shuf_identical,
        "crossgroup_clean": crossgroup_identical,
        "tol": tol,
    }


# ── convenience high-level driver pieces (shared by both consumers) ────────────


@dataclass
class SkillVariantSpec:
    """One named MLP skill variant over a (c_C, v0) layer battery.

    name: variant label (e.g. "base" / "zscored" / "shuffle"); X_transform /
    Y_transform are applied per layer to the raw (c_C, v0) before the PCA + fit.
    """

    name: str
    x_transform: object = field(default=None)  # callable (Xc) -> Xc' or None
    y_transform: object = field(default=None)  # callable (Yv, rng) -> Yv' or None


def zscore_columns(Xc: np.ndarray) -> np.ndarray:
    """Per-dim z-score of the design (global μ/σ) — the #722 z-scored-input variant."""
    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0) + 1e-8
    return (Xc - mu) / sd


# ═══════════════════════════════════════════════════════════════════════════════
# Batched fixed-split multi-output MLP (issue #841 lineage). Ported to main from
# origin/issue-841 @ a9c2e59849 (content snapshot @ e2a7398541; introduced at
# 404046fefe) by
# issue #926, WITH the #926 partition-invariant per-group seeding fix (per-group
# init keyed to (seed, group.key), NOT batch position — supersedes the branch
# copies' batch-order seeding). At the eventual issue-841 / issue-922 merges,
# resolve conflicts in this region to MAIN's version.
# ═══════════════════════════════════════════════════════════════════════════════
# The LOCO helpers above return held-out predictions only. The Δ-predictability
# atlas (#841) needs (a) a SINGLE fixed train/eval split per fit-problem instead
# of LOCO folds, (b) SmoothL1 (not MSE), (c) inner-validation early-stopping, and
# (d) the TRAINED PARAMS returned so the fitted map can be applied to NEW inputs
# (the eval-context trajectories Stage 1 transports). This is the "extend on a
# branch if needed" path the vectorize-many-cell-fits rule allows — it reuses the
# SAME (d_in→hidden→p) multi-output architecture, per-group train-only ddof=1
# standardization, and bmm forward as fit_batched_loco_mlp_multihead, differing
# only in the split shape + loss + the returned params.


def split_group_init_seed(seed: int, key: tuple) -> int:
    """Stable per-group init seed for ``fit_batched_split_mlp``: unsalted blake2b
    of ``f"{seed}|{key!r}"`` reduced to [0, 2**63). Depends ONLY on (seed, key) —
    never on batch position, chunking, ordering, process, or platform — so any
    partition of a group list reproduces each group's init bit-exactly (#926).
    Key elements must be Python primitives (str/int/float/bool/None, or tuples
    thereof): ``repr(key)`` is the identity. Python's builtin ``hash()`` is
    salted per process (PYTHONHASHSEED) and MUST NOT replace this.
    """
    payload = f"{int(seed)}|{key!r}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**63)


@dataclass
class SplitMLPGroup:
    """One fixed-split multi-output MLP fit problem in the batched ensemble.

    All groups in a batch MUST share n_train, n_eval, d_in, p, and (if any group
    supplies validation) n_val — that shared shape is what lets them ride one
    batched pass. Groups differ only in their X/Y values. ``X_val``/``Y_val`` are
    optional; when present the returned map is the per-member BEST-validation-loss
    snapshot (batched early stopping); when absent the final-epoch params are used.
    """

    key: tuple
    X_train: np.ndarray  # (n_train, d_in) fp32
    Y_train: np.ndarray  # (n_train, p) fp32 multi-output target
    X_eval: np.ndarray  # (n_eval, d_in) fp32 (the atlas test slice)
    X_val: np.ndarray | None = None  # (n_val, d_in) fp32
    Y_val: np.ndarray | None = None  # (n_val, p) fp32


@dataclass
class SplitMLPResult:
    """Eval-slice predictions + trained params for every group.

    ``preds_by_key``: key -> (n_eval, p) predictions on ``X_eval`` (the atlas R²
    read). ``params_by_key``: key -> {"W1","b1","W2","b2","mu","sd"} numpy arrays
    (the trained map + its train-input standardization) so a caller can apply the
    map to arbitrary inputs (Stage-1 transport composition).
    ``best_val_epoch_by_key``: key -> int best-val epoch (-1 when no validation).
    """

    preds_by_key: dict
    params_by_key: dict
    best_val_epoch_by_key: dict
    n_groups: int
    chunk_size: int


def _split_mlp_eval(Xg, W1, b1, W2, b2, mu, sd):
    """bmm forward of the batched multi-output MLP on standardized inputs.

    Xg (c, n, d_in) raw inputs; mu/sd (c, d_in); W1 (c, hid, d_in), b1 (c, hid),
    W2 (c, p, hid), b2 (c, p). Returns (c, n, p) predictions. Standardizes with
    the per-member train mu/sd (same as the LOCO helpers' bmm forward).
    """
    Xn = (Xg - mu.unsqueeze(1)) / sd.unsqueeze(1)  # (c, n, d_in)
    h = torch.nn.functional.gelu(torch.bmm(Xn, W1.transpose(1, 2)) + b1.unsqueeze(1))  # (c, n, hid)
    return torch.bmm(h, W2.transpose(1, 2)) + b2.unsqueeze(1)  # (c, n, p)


@_with_thread_cap
def fit_batched_split_mlp(  # noqa: C901 -- linear batched trainer; loss selector + parity options add branches
    groups: list[SplitMLPGroup],
    *,
    seed: int = DEFAULT_MLP_SEED,
    hidden: int = MLP_HIDDEN,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = MLP_MAX_EPOCHS,
    device: str = "cpu",
    chunk_size: int = 8,
    num_threads: int | None = None,
    smooth_l1_beta: float = 1.0,
    loss: str = "smooth_l1",
    standardize_inputs: bool = True,
    patience: int | None = None,
) -> SplitMLPResult:
    """Fit ALL groups' fixed-split multi-output MLPs as one batched ensemble.

    Member ``g`` is a ``Linear(d_in, hidden) → GELU → Linear(hidden, p)`` net
    trained on group ``g``'s ``X_train → Y_train`` with AdamW(lr, wd) + SmoothL1
    (``smooth_l1_beta``; ``loss="mse"`` swaps train AND val loss to MSE — the
    #931 registered default-preserving extension so the batched fitter can
    reproduce ``fit_h.mlp_fit_predict``'s MSE recipe for the G1b parity fit),
    standardized on the group's own train rows. When
    validation is supplied, the per-member BEST-validation-loss params are kept
    (batched early stopping — equivalent to patience≥remaining-epochs); else the
    final-epoch params. Chunked over groups so peak memory scales with
    ``chunk_size × n × d`` (the #841 tensors are large-d, unlike #722's tiny-d
    LOCO folds — chunk small).

    Two further default-preserving #931 parent-parity options (both needed so
    the batched fitter reproduces ``fit_h.mlp_fit_predict`` EXACTLY — the r1
    G1b recipe-mismatch fix): ``standardize_inputs=False`` skips the internal
    per-member train-row standardization (identity mu=0/sd=1) for callers that
    pre-standardize on the parent's FULL fold-train stats (the parent
    standardizes BEFORE its val split; the internal path standardizes on the
    post-split train rows only). ``patience=<k>`` (requires validation) applies
    the parent's per-member early stopping: an improvement is
    ``vloss < best_val - 1e-6`` (the parent's threshold), a member whose val
    loss has not improved for ``k`` consecutive epochs is FROZEN at its best
    snapshot, and the epoch loop breaks once every member in the chunk has
    stopped. Defaults (``True``/``None``) are bit-identical to the prior
    behavior (``vloss < best_val``, no freeze — pinned by
    ``tests/test_vectorized_split_mlp.py``).

    SEEDING: each group's init is drawn under
    ``torch.manual_seed(split_group_init_seed(seed, group.key))`` — a stable
    unsalted hash of ``(seed, repr(key))`` — NOT from the group's batch
    position. Any partition or reordering of the same group list across calls
    yields bit-identical per-group inits, and (on CPU, pinned by
    ``assert_split_mlp_partition_invariant`` + tests) bit-identical trained
    results at matched settings. Keys must be unique per call (asserted) and
    built from Python primitives (repr is the identity). Supersedes the
    pre-#926 batch-order seeding (member g got draw g), under which chunking a
    group list changed every member's init (#841 fu-r1 v15: pred maxdiff 0.82
    between a 5-group call and 2+3-group calls). Deterministic + reproducible
    across runs and processes; sets the global torch CPU RNG as a side effect.
    Returns a ``SplitMLPResult`` (eval preds + trained params + best-val epoch).

    CPU thread cap (``num_threads``, via ``_with_thread_cap``): ``None``
    (default) pins an ambient-ceilinged conservative cap for the fit and
    restores the prior pool after; ``0`` opts out; positive values pin
    verbatim during the fit (#1079).
    """
    from torch.func import stack_module_state

    assert loss in ("smooth_l1", "mse"), f"unknown loss {loss!r}"

    def _loss_per_member(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-member (c,) mean loss under the selected criterion."""
        if loss == "mse":
            return torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
        return torch.nn.functional.smooth_l1_loss(
            pred, target, reduction="none", beta=smooth_l1_beta
        ).mean(dim=(1, 2))

    if not groups:
        return SplitMLPResult({}, {}, {}, 0, chunk_size)
    if patience is not None:
        assert patience > 0, f"patience must be positive, got {patience}"
        assert groups[0].X_val is not None, "patience early stopping requires validation splits"

    n_train, d_in = groups[0].X_train.shape
    p = groups[0].Y_train.shape[1]
    n_eval = groups[0].X_eval.shape[0]
    has_val = groups[0].X_val is not None
    n_val = groups[0].X_val.shape[0] if has_val else 0
    for g in groups:
        assert g.X_train.shape == (n_train, d_in), (g.key, g.X_train.shape)
        assert g.Y_train.shape == (n_train, p), (g.key, g.Y_train.shape)
        assert g.X_eval.shape == (n_eval, d_in), (g.key, g.X_eval.shape)
        assert (g.X_val is not None) == has_val, (g.key, "val presence must match across groups")
        if has_val:
            assert g.X_val.shape == (n_val, d_in) and g.Y_val.shape == (n_val, p), g.key
    dev = torch.device(device)
    n_groups = len(groups)

    # Stacked group tensors stay on CPU; only the per-chunk slice is moved to the
    # device (below). Moving the whole (n_groups, n, d) ensemble to the device up
    # front peaks at the full stacked footprint PLUS the per-chunk working set —
    # which OOMs a smaller GPU at large (n, d) (the #811 measured-peak trap). CPU
    # residency keeps device peak ~per-chunk.
    def _stack(attr):
        return torch.from_numpy(
            np.ascontiguousarray(np.stack([getattr(g, attr) for g in groups]).astype(np.float32))
        )

    Xtr_g, Ytr_g, Xev_g = _stack("X_train"), _stack("Y_train"), _stack("X_eval")
    Xval_g = _stack("X_val") if has_val else None
    Yval_g = _stack("Y_val") if has_val else None

    class _MLPMulti(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, p)
            )

    # Per-group init keyed to (seed, group.key) — partition/reorder-invariant (#926).
    # Inits are drawn on the CPU RNG (modules constructed on CPU, then .to(dev)),
    # so init bytes are device-independent; each group's stream is fully consumed
    # before the next group's manual_seed, so neighbors cannot perturb it.
    assert len({repr(g.key) for g in groups}) == n_groups, (
        "duplicate (or repr-colliding) SplitMLPGroup.key in one call — keys must be "
        "unique; results are keyed by them and inits are seeded from them"
    )
    block_members = []
    for g in groups:
        torch.manual_seed(split_group_init_seed(seed, g.key))
        block_members.append(_MLPMulti().to(dev))
    bp, _bb = stack_module_state(block_members)
    bW1, bb1 = bp["net.0.weight"].detach(), bp["net.0.bias"].detach()
    bW2, bb2 = bp["net.2.weight"].detach(), bp["net.2.bias"].detach()

    preds_by_key: dict = {}
    params_by_key: dict = {}
    best_epoch_by_key: dict = {}
    chunk = chunk_size if (chunk_size and chunk_size > 0) else n_groups
    for lo in range(0, n_groups, chunk):
        hi = min(lo + chunk, n_groups)
        c = hi - lo
        sl = slice(lo, hi)
        W1 = bW1[sl].clone()  # (c, hid, d_in)
        b1 = bb1[sl].clone()  # (c, hid)
        W2 = bW2[sl].clone()  # (c, p, hid)
        b2 = bb2[sl].clone()  # (c, p)
        # Move only this chunk's slices to the device (CPU-resident otherwise).
        Xtr, Ytr, Xev = Xtr_g[sl].to(dev), Ytr_g[sl].to(dev), Xev_g[sl].to(dev)
        Xval = Xval_g[sl].to(dev) if has_val else None
        Yval = Yval_g[sl].to(dev) if has_val else None

        if standardize_inputs:
            # Per-group train-only standardization (ddof=1 via the SS form).
            mu = Xtr.mean(1)  # (c, d_in)
            var = Xtr.var(1, correction=1)  # (c, d_in)
            sd = var.clamp(min=0.0).sqrt() + 1e-6
        else:
            # Caller pre-standardized (parent-parity: full fold-train stats
            # applied BEFORE the val split) — identity transform inside.
            mu = torch.zeros((c, d_in), device=dev)
            sd = torch.ones((c, d_in), device=dev)

        for w in (W1, b1, W2, b2):
            w.requires_grad_(True)
        opt = torch.optim.AdamW([W1, b1, W2, b2], lr=lr, weight_decay=wd)

        best_val = torch.full((c,), float("inf"), device=dev)
        best_epoch = torch.full((c,), -1, dtype=torch.long, device=dev)
        best = {k: v.detach().clone() for k, v in (("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2))}
        bad = torch.zeros((c,), dtype=torch.long, device=dev)
        stopped = torch.zeros((c,), dtype=torch.bool, device=dev)
        pred = per_member = None  # keep bound for the del below at max_epochs=0 (#926)
        for epoch in range(max_epochs):
            opt.zero_grad(set_to_none=True)
            pred = _split_mlp_eval(Xtr, W1, b1, W2, b2, mu, sd)  # (c, n_train, p)
            per_member = _loss_per_member(pred, Ytr)  # (c,)
            per_member.sum().backward()
            opt.step()
            if has_val:
                with torch.no_grad():
                    vpred = _split_mlp_eval(Xval, W1, b1, W2, b2, mu, sd)
                    vloss = _loss_per_member(vpred, Yval)  # (c,)
                if patience is None:
                    improved = vloss < best_val
                else:
                    # Parent semantics (fit_h.mlp_fit_predict): improvement is
                    # vloss < best_val - 1e-6; a stopped member's best is FROZEN.
                    improved = (vloss < best_val - 1e-6) & ~stopped
                if improved.any():
                    best_val = torch.where(improved, vloss, best_val)
                    best_epoch = torch.where(
                        improved, torch.full_like(best_epoch, epoch), best_epoch
                    )
                    im = improved.view(c, *([1] * (W1.dim() - 1)))
                    best["W1"] = torch.where(im, W1.detach(), best["W1"])
                    best["W2"] = torch.where(im, W2.detach(), best["W2"])
                    imb = improved.view(c, 1)
                    best["b1"] = torch.where(imb, b1.detach(), best["b1"])
                    best["b2"] = torch.where(imb, b2.detach(), best["b2"])
                if patience is not None:
                    bad = torch.where(improved, torch.zeros_like(bad), bad + 1)
                    stopped = stopped | (bad >= patience)
                    if bool(stopped.all()):
                        break
        if not has_val:
            best = {"W1": W1.detach(), "b1": b1.detach(), "W2": W2.detach(), "b2": b2.detach()}

        with torch.no_grad():
            ev = _split_mlp_eval(Xev, best["W1"], best["b1"], best["W2"], best["b2"], mu, sd)
        ev_np = ev.detach().cpu().numpy().astype(np.float64)
        for j in range(c):
            key = groups[lo + j].key
            preds_by_key[key] = ev_np[j]
            params_by_key[key] = {
                "W1": best["W1"][j].detach().cpu().numpy().astype(np.float32),
                "b1": best["b1"][j].detach().cpu().numpy().astype(np.float32),
                "W2": best["W2"][j].detach().cpu().numpy().astype(np.float32),
                "b2": best["b2"][j].detach().cpu().numpy().astype(np.float32),
                "mu": mu[j].detach().cpu().numpy().astype(np.float32),
                "sd": sd[j].detach().cpu().numpy().astype(np.float32),
            }
            best_epoch_by_key[key] = int(best_epoch[j].item()) if has_val else -1
        del W1, b1, W2, b2, Xtr, Ytr, Xev, Xval, Yval, opt, pred, per_member, best
    return SplitMLPResult(preds_by_key, params_by_key, best_epoch_by_key, n_groups, chunk)


def assert_split_mlp_matches_serial(
    seed: int = 0, n_train: int = 120, n_eval: int = 40, d: int = 32, p: int = 6
) -> dict:
    """Assert the batched split MLP reproduces a serial ``_MLP`` reference (no val).

    Fits one group both via ``fit_batched_split_mlp`` (chunk that exercises the
    loop) and a serial per-member ``_MLPMulti``-equivalent AdamW+SmoothL1 loop on
    the SAME (X_train→Y_train), same seed/standardization, then compares the
    eval-slice predictions. The residual is batched-bmm vs nn.Linear reduction
    order (the same class the #658/#722 gates bound); ``tol=5e-5`` at this
    well-conditioned N. Validates the unit-equivalence the plan §12 assumption 8
    requires before the atlas trusts the batched fit.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_train + n_eval, 4))
    W = rng.standard_normal((4, d))
    X = (z @ W + 0.1 * rng.standard_normal((n_train + n_eval, d))).astype(np.float32)
    B = rng.standard_normal((d, p))
    Y = (X @ B * 0.05 + 0.1 * rng.standard_normal((n_train + n_eval, p))).astype(np.float32)
    Xtr, Ytr, Xev = X[:n_train], Y[:n_train], X[n_train:]

    grp = SplitMLPGroup(("g",), Xtr, Ytr, Xev)
    # num_threads=0: ambient pool, matching the serial reference below (pre-#1079 behavior).
    res = fit_batched_split_mlp(
        [grp], seed=658, max_epochs=40, device="cpu", chunk_size=1, num_threads=0
    )
    batched_pred = res.preds_by_key[("g",)]

    # Serial reference: one _MLPMulti drawing the SAME init as the batched fit's
    # per-group seed for key ("g",) (#926 seeding contract) + AdamW + SmoothL1.
    torch.manual_seed(split_group_init_seed(658, ("g",)))

    class _Ref(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(MLP_HIDDEN, p)
            )

        def forward(self, x):
            return self.net(x)

    net = _Ref()
    Xt = torch.from_numpy(Xtr)
    mu = Xt.mean(0)
    sd = Xt.var(0, correction=1).clamp(min=0.0).sqrt() + 1e-6
    Xn = (Xt - mu) / sd
    Yt = torch.from_numpy(Ytr)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(net(Xn), Yt, beta=1.0)
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xev_n = (torch.from_numpy(Xev) - mu) / sd
        ref_pred = net(Xev_n).numpy().astype(np.float64)

    max_abs = float(np.max(np.abs(batched_pred - ref_pred)))
    tol = 5e-5
    assert max_abs <= tol, (
        f"split MLP exactness FAILED: max|Δpred|={max_abs:.3e} > {tol} vs the serial "
        "_MLP reference (batched bmm vs nn.Linear reduction order drifted)"
    )
    return {"max_abs_delta": max_abs, "tol": tol}


def assert_split_mlp_partition_invariant(
    seed: int = 0,
    n_train: int = 48,
    n_eval: int = 16,
    d: int = 12,
    p: int = 3,
    n_groups: int = 5,
    hidden: int = 32,
    max_epochs: int = 15,
) -> dict:
    """Assert fit_batched_split_mlp is PARTITION- and REORDER-invariant (#926).

    Builds ``n_groups`` synthetic groups with distinct keys/targets, then fits
    (A) all in ONE call at chunk_size=2, (B) a 2+3 partition (chunk-membership-
    ALIGNED with A: pure seeding-fix leg), (B') a 3+2 partition (boundary-
    MISALIGNED: chunk memberships differ from A), (C) all in REVERSED order,
    and (D) one full call at chunk_size=0 (no chunking — the cross-CHUNK-SIZE
    leg mirroring ``assert_matches_reference`` check (b)). Also asserts
    distinct keys derive distinct init seeds (anti-degeneracy). Asserts per-key
    ``preds_by_key`` and every ``params_by_key`` array (W1,b1,W2,b2,mu,sd)
    BIT-identical (``np.array_equal``) across all arms, each assert naming its
    arm so a failure report says WHICH arm broke (an aligned-partition failure
    = seeding bug; a reversed/nochunk-only failure = the anticipated kernel-
    stability class, §6 fallback territory). CPU-only, num_threads pinned.
    Returns the summary dict for the caller's log.
    """
    rng = np.random.default_rng(seed)
    groups = []
    for i in range(n_groups):
        X = rng.standard_normal((n_train + n_eval, d)).astype(np.float32)
        Y = (
            X @ rng.standard_normal((d, p)) * 0.05
            + 0.1 * rng.standard_normal((n_train + n_eval, p))
        ).astype(np.float32)
        groups.append(SplitMLPGroup((f"g{i}",), X[:n_train], Y[:n_train], X[n_train:]))
    # Cross-key distinctness (reconciler mandatory-absorb, #926 critique r1): a
    # key-ignoring helper would make every invariance assert below vacuously true.
    seeds = {g.key: split_group_init_seed(658, g.key) for g in groups}
    assert len(set(seeds.values())) == len(groups), (
        "split_group_init_seed collapsed distinct keys to a shared seed"
    )
    kw = dict(
        seed=658,
        hidden=hidden,
        max_epochs=max_epochs,
        device="cpu",
        chunk_size=2,
        num_threads=8,  # thread count pinned: the asserted bit-identity contract
        # is environment-pinned, not ambient (stats-critic rec)
    )
    full = fit_batched_split_mlp(groups, **kw)
    part_a = fit_batched_split_mlp(groups[:2], **kw)  # 2+3: seeding-fix leg
    part_b = fit_batched_split_mlp(groups[2:], **kw)
    mis_a = fit_batched_split_mlp(groups[:3], **kw)  # 3+2: boundary-MISALIGNED
    mis_b = fit_batched_split_mlp(groups[3:], **kw)  # chunk memberships differ
    rev = fit_batched_split_mlp(list(reversed(groups)), **kw)
    nochunk = fit_batched_split_mlp(groups, **{**kw, "chunk_size": 0})
    # ^ cross-CHUNK-SIZE leg (width 5 vs 2) — the split-variant sibling of the
    # LOCO gate's check (b) chunked-vs-nochunk comparison; certifies the
    # "group_chunk is a memory knob" contract (stats-critic Must-Fix, #926 r1).
    for g in groups:
        k = g.key
        split_res = part_a if k in part_a.preds_by_key else part_b
        mis_res = mis_a if k in mis_a.preds_by_key else mis_b
        for arm_name, arm in (
            ("partition-2+3", split_res),
            ("partition-3+2", mis_res),
            ("reversed", rev),
            ("nochunk", nochunk),
        ):
            assert np.array_equal(full.preds_by_key[k], arm.preds_by_key[k]), (arm_name, k)
            for name in ("W1", "b1", "W2", "b2", "mu", "sd"):
                assert np.array_equal(full.params_by_key[k][name], arm.params_by_key[k][name]), (
                    arm_name,
                    k,
                    name,
                )
    return {
        "n_groups": n_groups,
        "partition_bit_identical": True,
        "reorder_bit_identical": True,
        "cross_chunk_bit_identical": True,
        "distinct_key_seeds": True,
    }


def _pca_basis_on_device(Y: np.ndarray, k: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """robust_pca_basis semantics (mean + top-k right singular vectors), device-routed.

    Mirrors ``issue931_fit_cells._pca_basis_device`` (the #931 r1
    dense-factorization fix): the numpy gesdd SVD at production shape
    (~3200x3584 f32) measures ~30 s/call on 8 CPU threads; torch.linalg.svd on
    the fit device is ~1-2 s/call on A100 — subspace-identical up to sign, and
    the R^2 read is span-invariant through ``pred @ comps + mu``. Near-singular
    fallback mirrors ``robust_pca_basis`` (gesdd -> gesvd on cuda; the
    numpy/torch fallback on cpu). Returns (mu (H,), comps (k', H)).
    """
    t = torch.from_numpy(np.ascontiguousarray(Y.astype(np.float32))).to(device)
    tc = t - t.mean(dim=0)
    try:
        _, _, Vh = torch.linalg.svd(tc, full_matrices=False)
    except torch.linalg.LinAlgError:
        if t.is_cuda:
            _, _, Vh = torch.linalg.svd(tc, full_matrices=False, driver="gesvd")
        else:
            mu_np, comps, _fb = robust_pca_basis(Y.astype(np.float32), k)
            return mu_np, comps
    kk = min(k, Vh.shape[0])
    return t.mean(dim=0).cpu().numpy(), Vh[:kk].contiguous().cpu().numpy()


def batched_fold_cv_mlp_r2(
    X: np.ndarray,
    Y: np.ndarray,
    fold_ids: np.ndarray,
    *,
    layers: list[int],
    perms_by_layer: dict[int, list[np.ndarray]],
    pca_k: int = 64,
    max_epochs: int = MLP_MAX_EPOCHS,
    seed: int = 42,
    device: str | None = None,
    min_train_rows: int = 3,
    time_budget_s: float | None = None,
    started: float | None = None,
    chunk_size: int = 8,
) -> tuple[dict[str, dict], bool]:
    """Batched fold-CV MLP R^2 per layer: obs + row-permutation null draws.

    Returns ``({str(layer): {"r2_obs", "r2_null", "r2_obs_folds",
    "budget_hit_folds"}}, budget_exhausted)`` — the ``cells_*.json["mlp"]``
    block schema of ``issue825_fit_cells.run_mlp_secondary``. Per fold, the
    (layer x draw) members ride ONE :func:`fit_batched_split_mlp` call
    (``loss="mse"``, ``standardize_inputs=False``, ``patience=20``) with the
    ``fit_h.mlp_fit_predict`` parent-parity prep: full fold-train ddof=0 X
    standardization BEFORE the rng(``seed``) 10% val split; PCA-``pca_k``
    basis on the full fold-train target via device-routed torch SVD, skipped
    when the target dim <= ``pca_k``. Origin pattern:
    ``issue931_fit_cells._mlp_fold_r2`` (#931 G1b, 0.02 parity vs the serial
    parent; the residual delta is the per-member key-seeded init vs the
    parent's global ``manual_seed``).

    Inputs: ``X``/``Y`` are ``(N, L, D_x)`` / ``(N, L, D_y)``; ``fold_ids``
    is ``(N,)`` int (caller-computed, e.g. ``issue825_fit_cells._cv_folds``);
    ``perms_by_layer[li]`` is the layer's list of ``(N,)`` ROW permutations
    (the observed/identity draw is prepended internally as draw 0). Folds
    with ``te.sum() == 0 or tr.sum() < min_train_rows`` are skipped exactly
    like the serial guard (no ss contribution, no per-fold entry).

    Budget semantics (fold granularity — the batched call is the atomic
    unit): before each fold's member build AND before its fit call,
    ``time.monotonic() - started > time_budget_s`` stops the sweep; every
    layer's pooled ``r2_obs`` and every null draw then read NaN
    (full-length NaN-padded ``r2_null``), ``budget_hit_folds`` lists the
    remaining fold ids (identical across layers — fold-outer), and
    ``budget_exhausted=True`` is returned. Completed folds keep their
    per-fold ``r2_obs_folds`` entries.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if started is None:
        started = time.monotonic()
    X = np.asarray(X)
    Y = np.asarray(Y)
    fold_ids = np.asarray(fold_ids)
    assert X.shape[0] == Y.shape[0] == fold_ids.shape[0], (X.shape, Y.shape, fold_ids.shape)
    assert X.ndim == 3 and Y.ndim == 3 and X.shape[1] == Y.shape[1], (X.shape, Y.shape)
    null_lens = {len(perms_by_layer[li]) for li in layers}
    assert len(null_lens) <= 1, f"non-uniform null-draw counts across layers: {null_lens}"
    n_null = null_lens.pop() if null_lens else 0
    n_draws = 1 + n_null
    n_folds = int(fold_ids.max()) + 1 if fold_ids.size else 0
    identity = np.arange(X.shape[0])
    # Draw 0 = observed (identity); draws 1..n_null = the caller's row perms —
    # the serial loop's `Yl[rng.permutation(len(Yl))]` stream, precomputed.
    draws_by_layer = {li: [identity, *perms_by_layer[li]] for li in layers}

    def _over_budget() -> bool:
        return time_budget_s is not None and time.monotonic() - started > time_budget_s

    # ss[(li, d)][k] = [ss_res, ss_tot] for fold k — pooled R^2 sums over
    # folds; per-fold R^2 reads each entry (the serial _cv_r2 arithmetic).
    ss: dict[tuple[int, int], dict[int, list[float]]] = {
        (li, d): {} for li in layers for d in range(n_draws)
    }
    budget_hit = False
    remaining: list[int] = []
    for k in range(n_folds):
        if _over_budget():
            budget_hit = True
            remaining = sorted(range(k, n_folds))
            break
        te = fold_ids == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < min_train_rows:
            continue  # serial fold guard: no ss contribution, no per-fold entry
        member_groups, member_meta = [], []
        for li in layers:
            for d, perm in enumerate(draws_by_layer[li]):
                Yp = Y[perm]
                Xtr = X[tr, li, :].astype(np.float32)
                Ytr_raw = Yp[tr, li, :].astype(np.float32)
                Xte = X[te, li, :].astype(np.float32)
                Yte_raw = Yp[te, li, :]
                # Parent parity: standardize X on FULL fold-train stats (ddof=0)
                # BEFORE the val split; apply the same stats to train/val/eval.
                xmu = Xtr.mean(0)
                xsd = Xtr.std(0) + 1e-6
                Xn = (Xtr - xmu) / xsd
                Xen = (Xte - xmu) / xsd
                # Parent parity: PCA basis on the FULL fold-train target; the
                # parent skips PCA entirely when the target dim <= pca_k.
                if Ytr_raw.shape[1] <= pca_k:
                    y_mu = Ytr_raw.mean(0)
                    comps = None
                    Yt = Ytr_raw - y_mu
                else:
                    y_mu, comps = _pca_basis_on_device(Ytr_raw, pca_k, device)
                    Yt = ((Ytr_raw - y_mu) @ comps.T).astype(np.float32)
                vr = np.random.default_rng(seed)
                pm = vr.permutation(len(Xn))
                n_val = max(1, round(0.1 * len(Xn)))
                vi, ti = pm[:n_val], pm[n_val:]
                member_groups.append(
                    SplitMLPGroup(
                        key=("i825mlp", int(li), int(d), int(k)),
                        X_train=Xn[ti],
                        Y_train=Yt[ti].astype(np.float32),
                        X_eval=Xen,
                        X_val=Xn[vi],
                        Y_val=Yt[vi].astype(np.float32),
                    )
                )
                member_meta.append((li, d, y_mu, comps, Yte_raw))
        if _over_budget():
            budget_hit = True
            remaining = sorted(range(k, n_folds))
            break
        res = fit_batched_split_mlp(
            member_groups,
            seed=seed,
            max_epochs=max_epochs,
            device=device,
            chunk_size=chunk_size,
            loss="mse",
            standardize_inputs=False,
            patience=20,
        )
        for (li, d, y_mu, comps, Yte_raw), grp in zip(member_meta, member_groups, strict=True):
            pred_pca = res.preds_by_key[grp.key]
            pred = (pred_pca @ comps + y_mu) if comps is not None else (pred_pca + y_mu)
            true = Yte_raw.astype(np.float64)
            mu = true.mean(0)
            f_res = float(((true - pred) ** 2).sum())
            f_tot = float(((true - mu) ** 2).sum())
            ss[(li, d)][k] = [f_res, f_tot]

    def _pooled(pairs: dict[int, list[float]]) -> float:
        sr = sum(v[0] for v in pairs.values())
        st = sum(v[1] for v in pairs.values())
        return (1.0 - sr / st) if st > 1e-12 else float("nan")

    out: dict[str, dict] = {}
    for li in layers:
        if budget_hit:
            obs = float("nan")
            nulls = [float("nan")] * n_null
        else:
            obs = _pooled(ss[(li, 0)])
            nulls = [_pooled(ss[(li, d)]) for d in range(1, n_draws)]
        fold_stats = [
            (1.0 - v[0] / v[1]) if v[1] > 1e-12 else float("nan")
            for _k, v in sorted(ss[(li, 0)].items())
        ]
        out[str(li)] = {
            "r2_obs": obs,
            "r2_null": nulls,
            "r2_obs_folds": fold_stats,
            "budget_hit_folds": list(remaining),
        }
    return out, budget_hit
