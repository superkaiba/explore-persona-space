"""#1336 pooled-OOM fix: memory-aware null draw-chunk cap in issue825_fit_cells.

Pins for the `_resolve_null_draw_chunk` / `_free_device_bytes` cap wired into
`_null_ss_contrib` (the batched shuffle-null of `heldout_r2_sweep`):

1. chunk-size INVARIANCE — bit-identical r2_null across NULL_DRAW_BATCH in
   {unchunked, 3, 1} (load-bearing: the 4 completed pooled on-arm units carry
   done-markers the relaunch skips on, so their numbers must remain valid);
2. the cap ENGAGES under a small injected free-bytes probe (fails pre-fix:
   the pre-fix `step = max(1, NULL_DRAW_BATCH)` ignores memory entirely) and
   emits the fix-engaged log line;
3. production-shape accounting pin at the SLURM job 12643 incident numbers
   (n_train=149,964 / d=4096 / 20 draws) — pure arithmetic, no big allocs;
4. probe-failure fallback: free_bytes None/0 keeps the REQUESTED chunk
   (never silently tightens);
5. real-body probe of `_free_device_bytes` on cpu (no seam stubs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fc  # noqa: E402

# Tiny-matrix eighs thrash wide torch pools on the shared VM (#928 lesson).
torch.set_num_threads(2)

GRID = np.logspace(-3, 8, 5)  # same family as the pooled LAMBDAS_23 grid
N, D, N_FOLDS, N_DRAWS = 400, 16, 4, 6


def _synth(n: int = N, d: int = D, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1, d)).astype(np.float32)
    w = rng.normal(size=(d, d)) / np.sqrt(d)
    Y = (
        np.einsum("nld,de->nle", X.astype(np.float64), w) + 0.5 * rng.normal(size=(n, 1, d))
    ).astype(np.float32)
    conv = np.repeat([f"c{i}" for i in range(n // 2)], 2)  # non-trivial group folds
    return X, Y, conv


def _run_sweep(monkeypatch, *, chunk: int, free_fn):
    monkeypatch.setattr(fc, "N_INNER_LAMBDA_FOLDS", 2)  # the #1336 pooled regime
    monkeypatch.setattr(fc, "NULL_DRAW_BATCH", chunk)
    monkeypatch.setattr(fc, "_free_device_bytes", free_fn)
    X, Y, conv = _synth()
    return fc.heldout_r2_sweep(
        X,
        Y,
        conv,
        n_folds=N_FOLDS,
        seed=0,
        null_draws=N_DRAWS,
        collect_lambdas=True,
        lambdas=GRID,
        reduced_basis_companion=False,
    )


def test_chunk_size_invariance_including_unchunked(monkeypatch):
    """r2_null invariant across chunk sizes {unchunked, 3, 2, 1}.

    Chunk sizes >= 2 are BIT-IDENTICAL to unchunked (per-draw GEMMs and
    reductions run per batch slice with identical shapes/contraction order,
    and the permutations are drawn BEFORE the chunk loop) — array_equal.
    Batch 1 dispatches a different torch CPU kernel (bmm/einsum lowers to a
    plain mm with different accumulation blocking), giving pure fp64
    round-off: MEASURED at this shape 4.44e-16 abs / 3.43e-14 rel
    (2026-08-13 calibration) — asserted at ~10x headroom, an explicitly
    justified fp64-round-off deviation per the #1336 crash-fix brief. The
    production cap resolves to >= 2 at the incident shape, so production
    stays on the bit-identical path.
    """
    huge = lambda dev: 1 << 60  # noqa: E731  — cap never binds
    ref = _run_sweep(monkeypatch, chunk=64, free_fn=huge)  # 6 draws, one chunk
    for chunk in (3, 2):
        out = _run_sweep(monkeypatch, chunk=chunk, free_fn=huge)
        assert np.array_equal(ref["r2_null"], out["r2_null"]), chunk
        assert np.array_equal(ref["r2_obs"], out["r2_obs"]), chunk
    out1 = _run_sweep(monkeypatch, chunk=1, free_fn=huge)
    assert np.allclose(ref["r2_null"], out1["r2_null"], rtol=1e-12, atol=5e-15)
    assert np.array_equal(ref["r2_obs"], out1["r2_obs"])  # obs path is chunk-free


def test_cap_engages_under_small_free_bytes(monkeypatch, capsys):
    """Fails pre-fix: chunk must shrink to the memory-derived cap (2 here).

    unit = N*D*8 = 51,200 B; free=800,000 B -> floor(800e3*0.8/(6*51.2e3)) = 2.
    Pre-fix `step = max(1, NULL_DRAW_BATCH)` would run all 6 draws in one
    chunk. Also pins the fix-engaged log line at the cap site.
    """
    seen: list[int] = []
    orig = fc._ridge_predict_cached_batched

    def spy(cache, Ytr, lambdas=None):
        seen.append(int(Ytr.shape[0]))
        return orig(cache, Ytr, lambdas=lambdas)

    monkeypatch.setattr(fc, "_ridge_predict_cached_batched", spy)
    _run_sweep(monkeypatch, chunk=64, free_fn=lambda dev: 800_000)
    assert seen and max(seen) == 2, seen
    out = capsys.readouterr().out
    assert "[fit825] null-draw chunk resolved: step=2 requested=64 draws=6" in out


def test_production_shape_accounting_pin():
    """Memory-accounting pin at the real incident shape (no big allocations).

    pooled_dpo_arm_off: sweep rows n=149,964, d=4096, 5 group folds
    (fold n_tr=118,359), null_draws=20 in ONE pre-fix chunk on a
    139.8 GiB H200.
    """
    n_rows, d = 149_964, 4096
    unit = n_rows * d * 8
    # (a) the incident's exact failing CUDA request: (20, 118359, 4096) fp64.
    failing = 20 * 118_359 * 4096 * 8
    assert abs(failing / 2**30 - 72.24) < 0.01
    # (b) PRE-FIX demand (20 draws, one chunk) exceeds the WHOLE device even
    # at the MEASURED live factor 4.54 (calibration, not the safety-margined
    # constant) — the OOM is reproduced by accounting.
    h200 = int(139.8 * 2**30)
    assert 20 * unit * 4.54 > h200
    # (c) POST-FIX: the resolved chunk at incident-plausible free memory fits
    # inside the safety budget.
    free = h200 - (16 * 2**30)  # fold caches (~11 GiB) + Y_t (~4.6 GiB) resident
    step = fc._resolve_null_draw_chunk(64, 20, n_rows, d, free)
    assert 1 <= step < 20
    assert step * fc.NULL_DRAW_LIVE_FACTOR * unit <= free * fc.NULL_DRAW_MEM_SAFETY


def test_probe_failure_falls_back_to_requested(monkeypatch):
    """free_bytes None/0 keeps the REQUESTED chunk — never silently tighter."""
    assert fc._resolve_null_draw_chunk(64, 20, 149_964, 4096, None) == 20
    assert fc._resolve_null_draw_chunk(64, 20, 149_964, 4096, 0) == 20
    assert fc._resolve_null_draw_chunk(3, 20, 149_964, 4096, None) == 3
    # Sweep-level: with a failing probe the chunk equals min(requested, draws).
    seen: list[int] = []
    orig = fc._ridge_predict_cached_batched

    def spy(cache, Ytr, lambdas=None):
        seen.append(int(Ytr.shape[0]))
        return orig(cache, Ytr, lambdas=lambdas)

    monkeypatch.setattr(fc, "_ridge_predict_cached_batched", spy)
    _run_sweep(monkeypatch, chunk=64, free_fn=lambda dev: None)
    assert seen and max(seen) == N_DRAWS, seen


def test_free_device_bytes_cpu_real_body():
    """Real body, no stubs: the cpu probe reads /proc/meminfo (+ cgroup v2)."""
    v = fc._free_device_bytes(torch.device("cpu"))
    if sys.platform == "linux":
        assert isinstance(v, int) and v > 0
    else:  # pragma: no cover — non-Linux dev boxes
        assert v is None or (isinstance(v, int) and v > 0)
