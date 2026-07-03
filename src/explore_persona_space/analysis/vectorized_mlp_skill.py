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
   threads on tiny ops); a ``--device`` arg (cpu default, cuda optional).
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

import logging
import math
import os
import sys
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
    """
    from torch.func import stack_module_state

    if not groups:
        return BatchedMLPResult({}, 0, 0, 0, 0, chunk_size)

    if num_threads is not None and device == "cpu":
        torch.set_num_threads(int(num_threads))

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
    """
    from torch.func import stack_module_state

    if not groups:
        return BatchedMLPResult({}, 0, 0, 0, 0, chunk_size)
    if num_threads is not None and device == "cpu":
        torch.set_num_threads(int(num_threads))

    n, d_in = groups[0].X.shape
    p = groups[0].Y.shape[1]
    for g in groups:
        assert g.X.shape == (n, d_in), (g.key, g.X.shape, (n, d_in))
        assert g.Y.shape == (n, p), (g.key, g.Y.shape, (n, p))
    dev = torch.device(device)
    n_groups = len(groups)
    n_members = n_groups * n  # ONE multi-output net per (group, fold)

    Xg = torch.from_numpy(
        np.ascontiguousarray(np.stack([g.X for g in groups]).astype(np.float32))
    ).to(dev)
    Yg = torch.from_numpy(
        np.ascontiguousarray(np.stack([g.Y for g in groups]).astype(np.float32))
    ).to(dev)

    # n per-fold inits for the multi-output net (Linear(d_in,hid) + Linear(hid,p)).
    class _MLPMulti(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, p)
            )

    torch.manual_seed(seed)
    block_members = [_MLPMulti().to(dev) for _ in range(n)]
    bp, _bb = stack_module_state(block_members)
    bW1 = bp["net.0.weight"].detach()  # (n, hid, d_in)
    bb1 = bp["net.0.bias"].detach()  # (n, hid)
    bW2 = bp["net.2.weight"].detach()  # (n, p, hid)
    bb2 = bp["net.2.bias"].detach()  # (n, p)

    member_arange = torch.arange(n_members, device=dev)
    member_fold = member_arange % n  # (E,) -> fold_i == block-member id
    member_group = member_arange // n  # (E,) -> group g

    held_all = torch.empty(n_members, p, device=dev, dtype=torch.float32)
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
        c = hi - lo
        gidx = member_arange[lo:hi]
        cgroup = member_group[gidx]
        cfold = member_fold[gidx]

        W1 = bW1.index_select(0, cfold).clone()  # (c, hid, d_in)
        b1 = bb1.index_select(0, cfold).clone()  # (c, hid)
        W2 = bW2.index_select(0, cfold).clone()  # (c, p, hid)
        b2 = bb2.index_select(0, cfold).clone()  # (c, p)

        Xc = Xg[cgroup]  # (c, n, d_in)
        Yc = Yg[cgroup]  # (c, n, p)

        train_mask = torch.ones((c, n), dtype=torch.bool, device=dev)
        train_mask[torch.arange(c, device=dev), cfold] = False
        mask_f = train_mask.to(torch.float32)  # (c, n)
        counts = mask_f.sum(1, keepdim=True)

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
        held_all[lo:hi] = pred[torch.arange(c, device=dev), cfold]  # (c, p) held-out row
        del W1, b1, W2, b2, Xn, Xc, Yc, opt, pred, h, sq

    held = held_all.reshape(n_groups, n, p).contiguous()
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
        res = fit_batched_loco_mlp(groups, seed=658, max_epochs=20, device="cpu", chunk_size=chunk)
        d_base = float(np.max(np.abs(res.preds_by_key[("base",)] - ref)))
        d_shuf = float(np.max(np.abs(res.preds_by_key[("shuffle",)] - ref_sh)))

        # (b) chunk-invariance: no-chunk must be BIT-identical to the chunked run.
        res_nochunk = fit_batched_loco_mlp(
            groups, seed=658, max_epochs=20, device="cpu", chunk_size=0
        )
        chunk_base_identical = bool(
            np.array_equal(res.preds_by_key[("base",)], res_nochunk.preds_by_key[("base",)])
        )
        chunk_shuf_identical = bool(
            np.array_equal(res.preds_by_key[("shuffle",)], res_nochunk.preds_by_key[("shuffle",)])
        )

        # (c) cross-group non-contamination: 2-group base == single-group base.
        res_single = fit_batched_loco_mlp(
            [MLPGroup(("base",), X, Y)], seed=658, max_epochs=20, device="cpu", chunk_size=0
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
