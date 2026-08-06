"""Pins for the #1739 r2v2 extraction-factorial leg (issue1739_r2v2_factorial.py).

1. streamed pool-keyed weighted-sum directions == fits.extract_rb_matched on
   the same (acts, scores) — the mask-GEMM shard stream is an exact
   re-expression, not an approximation (e2 AND e2p; t1 AND context_end);
2. e2's context_end kind is EXCLUDED at spec-build time (the structural
   within-context cancellation) while e2p keeps both kinds;
3. _apply_map_to_direction == apply_map-then-weighted-difference (the linear
   commutation the mapped-vs-real comparison relies on);
4. off-pool contexts get zero weight (the per-fold disjointness mechanism).

Synthetic stores only (tmp_path); no network, no repo artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1739_r2v2_factorial import (  # noqa: E402
    DirectionSpec,
    StoreGrain,
    _apply_map_to_direction,
    build_natural_specs,
    load_store_grain,
    stream_directions,
)

# deliberately non-square, non-tiny-uniform dims (a d==Ly fixture masks
# transposition bugs — code-reviewer memory: tiny-dim fixtures)
N_CTX, K, LY, D = 5, 4, 3, 7
LAYERS = [0, 1, 2]
SPREAD = 15.0


def _mk_scores(rng) -> np.ndarray:
    """(n_ctx, K) scores with spread >= SPREAD in every ctx + one NaN drop."""
    scores = rng.uniform(0, 100, size=(N_CTX, K))
    scores[:, 0] = 5.0
    scores[:, 1] = 95.0  # guarantees within-ctx spread >= SPREAD
    scores[2, 3] = np.nan  # one dropped draw
    return scores


def _write_store(root: Path, acts_t1, acts_fc, ctx_ids) -> None:
    """Round-B-shaped synthetic store: row_index + per-kind per-layer shards.

    Rows are per-rollout in (ctx-major, k-minor) order, split into TWO shards
    per (kind, layer) so the shard-offset walk is exercised.
    """
    root.mkdir(parents=True)
    rows = [
        {"context_id": ctx_ids[c], "rollout_k": k, "group_key": f"g{c}"}
        for c in range(N_CTX)
        for k in range(K)
    ]
    with (root / "row_index.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_rows = N_CTX * K
    split = n_rows // 2
    for kind, acts in (("t1", acts_t1), ("context_end", acts_fc)):
        flat = acts.reshape(n_rows, LY, D)  # (rows, Ly, d), ctx-major
        for li, ly in enumerate(LAYERS):
            arr = flat[:, li].astype(np.float16)
            np.save(root / f"{kind}_L{ly:02d}_shard000.npy", arr[:split])
            np.save(root / f"{kind}_L{ly:02d}_shard001.npy", arr[split:])


@pytest.fixture()
def synthetic_grain(tmp_path):
    rng = np.random.default_rng(0)
    ctx_ids = [f"ctx{c}" for c in range(N_CTX)]
    acts_t1 = rng.normal(size=(N_CTX, K, LY, D))
    # context_end acts: IDENTICAL across a context's rollouts (store contract)
    ctx_act = rng.normal(size=(N_CTX, 1, LY, D))
    acts_fc = np.broadcast_to(ctx_act, (N_CTX, K, LY, D)).copy()
    scores = _mk_scores(rng)
    store = tmp_path / "store"
    _write_store(store, acts_t1, acts_fc, ctx_ids)
    sc = {
        ctx_ids[c]: {k: float(scores[c, k]) for k in range(K) if np.isfinite(scores[c, k])}
        for c in range(N_CTX)
    }
    grain = load_store_grain("train", store, sc, "synthetic")
    return grain, acts_t1, acts_fc, scores, ctx_ids


def test_streamed_direction_matches_extract_rb_matched(synthetic_grain):
    from explore_persona_space.experiments.issue_1739 import fits

    grain, acts_t1, acts_fc, scores, ctx_ids = synthetic_grain
    pools = {"P-A": set(ctx_ids)}
    specs, skips = build_natural_specs(pools, [grain], spread_min=SPREAD)
    assert not skips, skips
    dirs = stream_directions(specs, [grain], LAYERS, D)
    by_key = {(s.base, s.kind): dirs[i] for i, s in enumerate(specs)}

    # the stream reads the store's fp16 bytes — quantize the reference acts
    # identically so the comparison is exact (not a tolerance fudge)
    q_t1 = acts_t1.astype(np.float16).astype(np.float64)
    q_fc = acts_fc.astype(np.float16).astype(np.float64)
    for base, pooled in (("e2", False), ("e2p", True)):
        want, n_q = fits.extract_rb_matched(q_t1, scores, spread_min=SPREAD, pooled=pooled)
        got = by_key[(base, "t1")]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
        spec = next(s for s in specs if (s.base, s.kind) == (base, "t1"))
        assert spec.n_qualifying == n_q
    # fc kind: same weights on the (per-ctx constant) context_end rows
    want_fc, _ = fits.extract_rb_matched(q_fc, scores, spread_min=SPREAD, pooled=True)
    np.testing.assert_allclose(by_key[("e2p", "context_end")], want_fc, rtol=0, atol=1e-12)


def test_e2_fc_structurally_excluded_and_e2_fc_would_cancel(synthetic_grain):
    from explore_persona_space.experiments.issue_1739 import fits

    grain, _t1, acts_fc, scores, ctx_ids = synthetic_grain
    specs, _ = build_natural_specs({"P-A": set(ctx_ids)}, [grain], spread_min=SPREAD)
    kinds = {(s.base, s.kind) for s in specs}
    assert ("e2", "context_end") not in kinds  # structural N/A, never built
    assert {("e2", "t1"), ("e2p", "t1"), ("e2p", "context_end")} <= kinds
    # and the exclusion is load-bearing: the e2 weights DO cancel on fc rows
    rb_cancel, _ = fits.extract_rb_matched(acts_fc, scores, spread_min=SPREAD, pooled=False)
    assert float(np.abs(rb_cancel).max()) < 1e-10


def test_apply_map_to_direction_commutes_with_apply_map():
    from explore_persona_space.experiments.issue_1739 import fits

    rng = np.random.default_rng(1)
    n = 11
    w = rng.normal(size=(LY, D, D))
    mapfit = fits.MapFit(
        w=w,
        x_mu=rng.normal(size=(LY, 1, D)),
        x_sd=rng.uniform(0.5, 2.0, size=(LY, 1, D)),
        y_mu=rng.normal(size=(LY, 1, D)),
        diagnostics={},
    )
    x = rng.normal(size=(LY, n, D))
    # hi/lo weights EACH summing to 1 (the mean-difference construction)
    w_hi = rng.uniform(size=n)
    w_hi /= w_hi.sum()
    w_lo = rng.uniform(size=n)
    w_lo /= w_lo.sum()
    wd = w_hi - w_lo
    rb = np.einsum("n,lnd->ld", wd, x)  # weighted difference of inputs
    mapped_rows = fits.apply_map(x, mapfit)
    want = np.einsum("n,lnd->ld", wd, mapped_rows)  # diff of mapped rows
    got = _apply_map_to_direction(rb, mapfit)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    # shuffled-weight override commutes identically
    w_shuf = fits.shuffled_map_weights(w, seed=3)
    want_s = np.einsum("n,lnd->ld", wd, fits.apply_map(x, mapfit, w=w_shuf))
    np.testing.assert_allclose(
        _apply_map_to_direction(rb, mapfit, w=w_shuf), want_s, rtol=1e-10, atol=1e-10
    )


def test_off_pool_contexts_get_zero_weight(synthetic_grain):
    grain, _t1, _fc, _scores, ctx_ids = synthetic_grain
    pool = set(ctx_ids[:2])  # ctx0, ctx1 only
    specs, _ = build_natural_specs({"P-B:hold": pool}, [grain], spread_min=SPREAD)
    spec = next(s for s in specs if (s.base, s.kind) == ("e2p", "t1"))
    w_row = np.asarray(spec.row_weights["train"])
    off_pool_rows = [i for i, c in enumerate(grain.row_ctx) if c not in pool]
    assert np.all(w_row[off_pool_rows] == 0.0)
    assert np.any(w_row != 0.0)


def test_direction_spec_is_dataclass_shared_weights(synthetic_grain):
    """t1 / context_end specs of one (pool, base) share ONE scattered weight
    vector (the dedupe is identity, not just equality)."""
    grain, *_rest, ctx_ids = (
        synthetic_grain[0],
        *synthetic_grain[1:4],
        synthetic_grain[4],
    )
    specs, _ = build_natural_specs({"P-A": set(ctx_ids)}, [grain], spread_min=SPREAD)
    e2p = [s for s in specs if s.base == "e2p"]
    assert len(e2p) == 2 and isinstance(e2p[0], DirectionSpec)
    assert e2p[0].row_weights is e2p[1].row_weights


def test_load_store_grain_row_order_matches_shards(synthetic_grain):
    grain: StoreGrain = synthetic_grain[0]
    assert grain.n_rows == N_CTX * K
    assert grain.row_ctx[0] == "ctx0" and grain.row_ctx[-1] == f"ctx{N_CTX - 1}"
    assert list(grain.row_k[:K]) == list(range(K))
