# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, ²) in scientific docstrings + comments.
"""Issue #811 crash-fix (round 8) regression tests — memory-aware chunk cap.

The att-20260701-233116 relaunch crashed at the phase-0 KILL-1 gate with a
``torch.OutOfMemoryError`` (26.25 GiB alloc) inside
``fit_batched_loco_mlp_multihead``: the default ``chunk_size=4096`` was sized for
#722's n≈50 contexts, but #811's phase-0 gate runs n=480 cells, so a 4096-member
chunk materializes a ``(4096, 480, 3584)`` fp32 ``(c, n, d_in)`` intermediate.

These pin three permanent invariants added in the fix:

1. ``resolve_chunk_cap`` caps the chunk to fit free memory (fires; floor of 1;
   respects ``n_members``; honors ``requested`` when memory is ample).
2. The bmm-based masked moments match the old broadcast-sum forms (allclose).
3. ``fit_batched_loco_mlp_multihead`` is chunk-size invariant — a tiny fit at
   ``chunk_size=999`` vs ``chunk_size=2`` gives identical held-out predictions
   (the determinism claim the cap relies on).

Pure-Python, CPU only, no GPU / no HF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import explore_persona_space.analysis.vectorized_mlp_skill as vms  # noqa: E402
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    _GATHER_SRC_BYTES_LIMIT,
    MLPGroup,
    _gather_groups,
    fit_batched_loco_mlp,
    fit_batched_loco_mlp_multihead,
    resolve_chunk_cap,
)

# ── 1. resolve_chunk_cap ──────────────────────────────────────────────────────


def test_resolve_chunk_cap_fires_on_the_crash_shape():
    """The exact #811 crash shape caps FAR below the requested 4096.

    n=480, d_in=3584, requested=4096, free=14.59 GiB (the crash's reported free).
    With the calibrated default live_factor=26 (measured real-shape autograd peak),
    per_member = 26 × 480 × 3584 × 4 = 178,913,280 B; cap = floor(14.59 × 2³⁰ × 0.8
    / per_member) = 70 — well below 4096, and below the crashing c=4096.
    """
    free = int(14.59 * 2**30)
    cap = resolve_chunk_cap(4096, n_members=5760, n=480, d_in=3584, free_bytes=free)
    assert 1 < cap < 4096, cap
    assert 50 <= cap <= 90, cap


def test_resolve_chunk_cap_ample_memory_still_below_requested():
    """With 77 GiB free (a fresh A100-80) the cap is 369 — still below requested 4096.

    Confirms the cap would have kept the crash from happening even on the full pod
    (c=4096 × 480 × 3584 × 4 = 26.25 GiB per intermediate, several live at once).
    """
    free = 77 * 2**30
    cap = resolve_chunk_cap(4096, n_members=5760, n=480, d_in=3584, free_bytes=free)
    assert 250 <= cap <= 450, cap


def test_resolve_chunk_cap_covers_measured_autograd_peak():
    """The DEFAULT live_factor bounds the MEASURED per-chunk real-shape peak.

    Calibration (#811, CPU): at n=480, d_in=3584, c=64 the measured peak was
    ~10.7 GiB ≈ 25.5 × a single (c, n, d_in) fp32 tensor. At the 7 GiB free the
    re-OOM happened at, the DEFAULT cap must keep the chunk's MODELLED peak
    (live_factor × c × n·d_in·4) under 0.8 × free, while the naive live_factor=4
    would have picked a chunk whose REAL (×26) peak overflows — the c=218 re-OOM.
    """
    free = 7 * 2**30
    n, d = 480, 3584
    cap = resolve_chunk_cap(4096, n_members=480, n=n, d_in=d, free_bytes=free)
    modelled_peak = 26 * cap * n * d * 4
    assert modelled_peak <= 0.8 * free, (cap, modelled_peak / 2**30)
    naive = resolve_chunk_cap(4096, n_members=480, n=n, d_in=d, free_bytes=free, live_factor=4)
    real_peak_naive = 26 * naive * n * d * 4
    assert real_peak_naive > free, (naive, real_peak_naive / 2**30)


def test_resolve_chunk_cap_small_problem_returns_requested():
    """When free memory dwarfs the footprint, the cap is the requested value."""
    free = 64 * 2**30
    cap = resolve_chunk_cap(512, n_members=700, n=14, d_in=24, free_bytes=free)
    assert cap == 512, cap


def test_resolve_chunk_cap_never_exceeds_n_members():
    """The cap never exceeds the ensemble size (chunking a larger block is wasted)."""
    free = 64 * 2**30
    cap = resolve_chunk_cap(4096, n_members=100, n=14, d_in=24, free_bytes=free)
    assert cap == 100, cap


def test_resolve_chunk_cap_floor_of_one():
    """A pathologically tiny free budget still returns at least 1 (loud OOM, not no-op)."""
    cap = resolve_chunk_cap(4096, n_members=5760, n=480, d_in=3584, free_bytes=1)
    assert cap == 1, cap


def test_resolve_chunk_cap_unreadable_probe_leaves_requested():
    """free_bytes == 0 (unreadable probe) never tightens the chunk silently."""
    cap = resolve_chunk_cap(4096, n_members=5760, n=480, d_in=3584, free_bytes=0)
    assert cap == 4096, cap
    # capped only by n_members when requested exceeds it
    cap2 = resolve_chunk_cap(4096, n_members=100, n=480, d_in=3584, free_bytes=0)
    assert cap2 == 100, cap2


# ── 2. bmm masked moments == broadcast-sum masked moments ─────────────────────


def test_bmm_masked_moments_match_broadcast():
    """bmm(mask, Xc)/counts and bmm(mask, Xc²) reproduce the old broadcast forms.

    On random (c=3, n=7, d=5) tensors with a random per-member train mask, the
    bmm-based mu / sumsq (the memory-frugal forms the fix uses) match the old
    ``(mask.unsqueeze(2) * Xc).sum(1)`` broadcast-sum forms to fp32 tolerance.
    """
    torch.manual_seed(0)
    c, n, d = 3, 7, 5
    Xc = torch.randn(c, n, d)
    mask_f = (torch.rand(c, n) > 0.3).to(torch.float32)
    # ensure no all-zero mask row (counts>0)
    mask_f[:, 0] = 1.0
    counts = mask_f.sum(1, keepdim=True)

    mu_old = (mask_f.unsqueeze(2) * Xc).sum(1) / counts
    sumsq_old = (mask_f.unsqueeze(2) * (Xc * Xc)).sum(1)

    mu_new = torch.bmm(mask_f.unsqueeze(1), Xc).squeeze(1) / counts
    sumsq_new = torch.bmm(mask_f.unsqueeze(1), Xc * Xc).squeeze(1)

    assert torch.allclose(mu_old, mu_new, atol=1e-5), (mu_old - mu_new).abs().max()
    assert torch.allclose(sumsq_old, sumsq_new, atol=1e-5), (sumsq_old - sumsq_new).abs().max()


# ── 3. chunk-size invariance of the multihead fit (the determinism claim) ──────


def _tiny_groups():
    """Two tiny synthetic groups sharing (n=6, d_in=8, p=3) — a base + a shuffle."""
    rng = np.random.default_rng(7)
    n, d, p = 6, 8, 3
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = (X @ rng.standard_normal((d, p)) * 0.1).astype(np.float32)
    Ysh = Y[rng.permutation(n)]
    return [MLPGroup(("base",), X, Y), MLPGroup(("shuffle",), X, Ysh)]


def test_multihead_fit_is_chunk_size_invariant():
    """chunk_size=999 (one chunk) vs chunk_size=2 → identical held-out predictions.

    Pins the determinism the memory-aware cap depends on: capping the chunk only
    bounds peak memory, never the fit. Uses a small forced free-memory budget via
    the real ``resolve_chunk_cap`` path (CPU: SC_AVPHYS_PAGES), so both runs go
    through the cap; the only difference is the REQUESTED chunk.
    """
    groups = _tiny_groups()
    res_big = fit_batched_loco_mlp_multihead(
        groups, max_epochs=3, device="cpu", chunk_size=999, num_threads=2
    )
    res_small = fit_batched_loco_mlp_multihead(
        groups, max_epochs=3, device="cpu", chunk_size=2, num_threads=2
    )
    for key in (("base",), ("shuffle",)):
        a = res_big.preds_by_key[key]
        b = res_small.preds_by_key[key]
        assert np.array_equal(a, b), (key, np.abs(a - b).max())


# ── 4. large-source gather fallback (#1739 r4: aten::index int32 overflow) ────
#
# att-20260802-053920-newarma5hall: the L=16000 transfer unit (n=23188 rows,
# G=28 layer groups, d_in=3584, 2 transfer folds, chunk cap 6) crashed
# ``torch.AcceleratorError: CUDA error: invalid configuration argument`` at the
# first chunk's ``Xc = Xg[cgroup]`` — the gather SOURCE Xg held
# 28 x 23188 x 3584 = 2,326,962,176 fp32 elements (> 2^31 - 1), while every
# completed unit's Xg sat below 2^31 (n=16000: 1,605,632,000). The fix routes
# dim-0 group gathers through ``_gather_groups``: above ``_GATHER_SRC_BYTES_LIMIT``
# the result is materialized via per-member slice copies (bit-identical, no
# aten::index kernel over the oversized source).


def test_gather_groups_fallback_bitwise_matches_advanced_index():
    """Forced slice-copy fallback == ``src[idx]``, bitwise (dupes + out-of-order)."""
    torch.manual_seed(3)
    src = torch.randn(5, 7, 3)
    idx = torch.tensor([4, 0, 0, 2, 1, 0])
    fast = src[idx]
    slow = _gather_groups(src, idx, bytes_limit=1)  # force the copy path
    auto = _gather_groups(src, idx)  # small source -> the untouched src[idx] path
    assert torch.equal(slow, fast)
    assert torch.equal(auto, fast)
    # non-contiguous index + singleton index
    idx1 = torch.tensor([2])
    assert torch.equal(_gather_groups(src, idx1, bytes_limit=1), src[idx1])


def test_gather_limit_arithmetic_covers_the_observed_crash_shape():
    """The byte guard fires on the crashed production shape (int32-element proof).

    Pins the root-cause arithmetic directly for the true 2^31 case the CI box
    cannot materialize: the crashed transfer unit's Xg crosses 2^31 ELEMENTS
    (the aten::index int32 ceiling) while the last completed unit's Xg does
    not, and the conservative BYTE guard reroutes the crash shape.
    """
    g, d_in = 28, 3584
    crash_elems = g * 23188 * d_in  # failed unit (transfer L=16000, n=23188)
    pass_elems = g * 16000 * d_in  # completed unit (train L=16000, 382 s)
    assert crash_elems > 2**31 - 1, crash_elems
    assert pass_elems <= 2**31 - 1, pass_elems
    assert crash_elems * 4 >= _GATHER_SRC_BYTES_LIMIT  # guard fires on the crash shape
    # the guard is deliberately conservative: it also reroutes the big PASSING
    # shape — harmless by the bitwise-parity pins in this section.
    assert pass_elems * 4 >= _GATHER_SRC_BYTES_LIMIT


def _transfer_shaped_groups(n_tr: int = 9, n_ev: int = 4, d: int = 5, n_layers: int = 3):
    """Tiny CPU analogue of the crash shape class: per-layer groups (p=1) +
    2-fold train-vs-eval transfer labels (the ``run_transfer_cell`` fold shape)."""
    rng = np.random.default_rng(11)
    n = n_tr + n_ev
    folds = np.concatenate([np.ones(n_tr, dtype=np.int64), np.zeros(n_ev, dtype=np.int64)])
    groups = []
    for li in range(n_layers):
        x = rng.standard_normal((n, d)).astype(np.float32)
        y = (x @ rng.standard_normal((d, 1)) * 0.1).astype(np.float32)
        groups.append(MLPGroup(("arm5", li), x, y))
    return groups, folds


def test_multihead_transfer_shape_fallback_parity_per_fold(monkeypatch):
    """Forced-fallback multihead fit == normal-path fit, bitwise (per_fold mode).

    CPU repro of the failing SHAPE CLASS at small scale: 2-fold transfer
    row_groups, tiny chunk (n_members=6, chunk_size=4 -> a ragged tail chunk of
    2 — the cap-6 analogue; range() semantics make an EMPTY chunk impossible,
    ruling out the degenerate-tail suspect). Monkeypatching the module byte
    limit to 1 forces EVERY gather through the slice-copy fallback — held-out
    predictions must be BIT-identical to the untouched src[idx] path.
    """
    groups, folds = _transfer_shaped_groups()
    kw = dict(max_epochs=3, device="cpu", chunk_size=4, num_threads=2, row_groups=folds)
    ref = fit_batched_loco_mlp_multihead(groups, **kw)
    monkeypatch.setattr(vms, "_GATHER_SRC_BYTES_LIMIT", 1)
    got = fit_batched_loco_mlp_multihead(groups, **kw)
    for key in ref.preds_by_key:
        a, b = ref.preds_by_key[key], got.preds_by_key[key]
        assert np.array_equal(a, b), (key, np.abs(a - b).max())


def test_multihead_transfer_shape_fallback_parity_full_data(monkeypatch):
    """Same forced-fallback parity through the full_data (Xn_full) gather site."""
    groups, folds = _transfer_shaped_groups()
    kw = dict(
        max_epochs=3,
        device="cpu",
        chunk_size=4,
        num_threads=2,
        row_groups=folds,
        standardization="full_data",
    )
    ref = fit_batched_loco_mlp_multihead(groups, **kw)
    monkeypatch.setattr(vms, "_GATHER_SRC_BYTES_LIMIT", 1)
    got = fit_batched_loco_mlp_multihead(groups, **kw)
    for key in ref.preds_by_key:
        a, b = ref.preds_by_key[key], got.preds_by_key[key]
        assert np.array_equal(a, b), (key, np.abs(a - b).max())


def test_scalar_path_fallback_parity(monkeypatch):
    """Forced-fallback scalar-per-dim fit == normal-path fit, bitwise.

    The sibling ``fit_batched_loco_mlp`` chunk loop carries the same
    ``Xg[cgroup]`` / ``Yg[cgroup]`` gathers (same class) — pin its fallback
    parity too.
    """
    groups = _tiny_groups()
    kw = dict(max_epochs=3, device="cpu", chunk_size=5, num_threads=2)
    ref = fit_batched_loco_mlp(groups, **kw)
    monkeypatch.setattr(vms, "_GATHER_SRC_BYTES_LIMIT", 1)
    got = fit_batched_loco_mlp(groups, **kw)
    for key in ref.preds_by_key:
        a, b = ref.preds_by_key[key], got.preds_by_key[key]
        assert np.array_equal(a, b), (key, np.abs(a - b).max())
