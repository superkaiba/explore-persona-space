"""#2659 capture OOM hardening (GPU-free).

Job 65463 (q35_9b arm b, long profile) OOMed in the capture phase: 8 gpqa
rollouts of up to ~73k tokens right-padded together made SDPA allocate a
38.37 GiB quadratic tensor. The fix is (1) a length-aware batch plan under a
padded-token budget, (2) unpadded mask=None single-row forwards, (3) a
logged split-in-half OOM backoff that raises at batch size 1, and (4) results
scattered back BY ROW INDEX so shard output order and content are identical
to the old fixed-size path.

Every test executes the REAL bodies of _plan_capture_batches,
_capture_forward, _capture_batch_with_backoff and _capture_shard_arrays. The
only fake is the model forward (a deterministic per-token stub that drives
the real _CaptureReducer hooks the way a transformer block forward does). No
network, no GPU.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_run_cell as RC

H = 4
BUDGET = 24576


# ---------------------------------------------------------------------------
# Deterministic stub forward (padding- and batch-size-invariant by design)
# ---------------------------------------------------------------------------


class _StubModel:
    """h[b, t, :] = token_id + arange(H), fired through the REAL reducer hooks.

    Values depend only on the token id at each position, so real-token reads
    are invariant to right padding and to batch grouping, which is exactly the
    exact-arithmetic property of a correctly masked transformer forward. That
    makes bitwise old-vs-new comparison meaningful.
    """

    device = torch.device("cpu")

    def __init__(self, reducer: RC._CaptureReducer, fail_over_rows: int | None = None):
        self.reducer = reducer
        self.fail_over_rows = fail_over_rows
        self.batch_shapes: list[tuple[int, int]] = []
        self.masks: list[object] = []

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return self.__call__(input_ids=input_ids, attention_mask=attention_mask, **kw)

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        self.batch_shapes.append(tuple(input_ids.shape))
        self.masks.append(attention_mask)
        if self.fail_over_rows is not None and input_ids.shape[0] > self.fail_over_rows:
            raise RC._OOM_ERROR("stub CUDA OOM")
        h = input_ids.unsqueeze(-1).to(torch.float32) + torch.arange(H, dtype=torch.float32)
        for li in self.reducer.layers:
            self.reducer.hook_for(li)(None, None, h)
        return None


def _mk_rows(rng, n: int, extra_lengths: list[int] | None = None) -> list[dict]:
    rows = []
    comp_lens = [int(rng.integers(2, 24)) for _ in range(n)] + list(extra_lengths or [])
    for k, n_comp in enumerate(comp_lens):
        n_prompt = int(rng.integers(2, 6))
        prompt = [int(x) for x in rng.integers(5, 1000, size=n_prompt)]
        comp = [int(x) for x in rng.integers(5, 1000, size=n_comp)]
        total = n_prompt + n_comp
        rows.append(
            {
                "row_id": f"r{k:03d}",
                "prompt_ids": prompt,
                "comp_ids": comp,
                "positions": {"prompt_last": n_prompt - 1, "resp_last": total - 1},
                "spans": {"ans": (n_prompt, total)},
            }
        )
    return rows


def _metas_seqs(rows: list[dict], pw: tuple[str, ...]):
    metas, seqs = [], []
    for row in rows:
        full = row["prompt_ids"] + row["comp_ids"]
        pos_list = [row["positions"][p] for p in pw if p in row["positions"]]
        metas.append({"pos_list": pos_list, "ans_span": row["spans"]["ans"]})
        seqs.append(full)
    return metas, seqs


def _old_path(hf, reducer, shard_rows, pw, pad_id, batch_size):
    """Verbatim re-statement of the pre-#2659 fixed-size padded batching."""
    layers = reducer.layers
    per_layer_pos = {li: [] for li in layers}
    per_layer_y = {li: [] for li in layers}
    for b0 in range(0, len(shard_rows), batch_size):
        batch = shard_rows[b0 : b0 + batch_size]
        metas, seqs = _metas_seqs(batch, pw)
        maxlen = max(len(sq) for sq in seqs)
        ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, sq in enumerate(seqs):
            ids[i, : len(sq)] = torch.as_tensor(sq)
            mask[i, : len(sq)] = 1
        reducer.set_batch(metas)
        with torch.no_grad():
            hf(input_ids=ids, attention_mask=mask)
        for li in layers:
            per_layer_pos[li].append(reducer.out[li]["pos"][0])
            per_layer_y[li].append(reducer.out[li]["y"][0])
    return (
        {li: np.concatenate(per_layer_pos[li]) for li in layers},
        {li: np.concatenate(per_layer_y[li]) for li in layers},
    )


# ---------------------------------------------------------------------------
# 1. Length-aware batch planner
# ---------------------------------------------------------------------------


def test_plan_long_row_alone_short_rows_grouped():
    lengths = [100, 30000, 200]
    plan = RC._plan_capture_batches(lengths, max_rows=8, token_budget=BUDGET)
    assert [1] in plan, plan
    assert [0, 2] in plan, plan
    assert len(plan) == 2


def test_plan_budget_grouping_and_row_cap():
    lengths = [3000] * 10
    assert [len(b) for b in RC._plan_capture_batches(lengths, 8, BUDGET)] == [8, 2]
    plan = RC._plan_capture_batches(lengths, 8, 12000)
    assert [len(b) for b in plan] == [4, 4, 2]
    assert all(len(b) * 3000 <= 12000 for b in plan)
    assert [len(b) for b in RC._plan_capture_batches(lengths, 3, 10**9)] == [3, 3, 3, 1]


def test_plan_is_a_partition_and_respects_both_caps():
    rng = np.random.default_rng(0)
    lengths = [int(x) for x in rng.integers(1, 50_000, size=57)]
    plan = RC._plan_capture_batches(lengths, max_rows=8, token_budget=BUDGET)
    flat = [i for b in plan for i in b]
    assert sorted(flat) == list(range(57))
    for b in plan:
        assert 1 <= len(b) <= 8
        if len(b) > 1:
            assert len(b) * max(lengths[i] for i in b) <= BUDGET


def test_plan_rejects_degenerate_caps():
    with pytest.raises(AssertionError):
        RC._plan_capture_batches([5], max_rows=0, token_budget=BUDGET)
    with pytest.raises(AssertionError):
        RC._plan_capture_batches([5], max_rows=8, token_budget=0)


# ---------------------------------------------------------------------------
# 2. Single-row forwards are unpadded with attention_mask=None
# ---------------------------------------------------------------------------


def test_single_row_forward_unpadded_mask_none():
    rng = np.random.default_rng(1)
    rows = _mk_rows(rng, 1)
    pw = ("prompt_last", "resp_last")
    reducer = RC._CaptureReducer([0], H)
    hf = _StubModel(reducer)
    metas, seqs = _metas_seqs(rows, pw)
    out = RC._capture_forward(hf, reducer, metas, seqs, pad_id=0)
    assert hf.batch_shapes == [(1, len(seqs[0]))]
    assert hf.masks == [None]
    assert out[0][0].shape == (1, 2, H)
    assert out[0][1].shape == (1, H)


def test_multi_row_forward_keeps_2d_padding_mask():
    rng = np.random.default_rng(2)
    rows = _mk_rows(rng, 3)
    pw = ("prompt_last",)
    reducer = RC._CaptureReducer([0], H)
    hf = _StubModel(reducer)
    metas, seqs = _metas_seqs(rows, pw)
    RC._capture_forward(hf, reducer, metas, seqs, pad_id=0)
    (mask,) = hf.masks
    maxlen = max(len(sq) for sq in seqs)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (3, maxlen) and mask.dtype == torch.long
    for i, sq in enumerate(seqs):
        assert mask[i].sum().item() == len(sq)


# ---------------------------------------------------------------------------
# 3. Logged OOM backoff: split in half, raise at batch size 1
# ---------------------------------------------------------------------------


def test_oom_backoff_splits_and_counts(caplog):
    rng = np.random.default_rng(3)
    rows = _mk_rows(rng, 8)
    pw = ("prompt_last", "resp_last")
    reducer = RC._CaptureReducer([0], H)
    hf = _StubModel(reducer, fail_over_rows=2)
    metas, seqs = _metas_seqs(rows, pw)
    stats = {"n_backoffs": 0}
    with caplog.at_level(logging.WARNING):
        out = RC._capture_batch_with_backoff(hf, reducer, metas, seqs, 0, stats)
    # 8 OOMs, then each 4 OOMs, then four 2-row batches succeed: 3 backoffs.
    assert stats["n_backoffs"] == 3
    assert [s[0] for s in hf.batch_shapes] == [8, 4, 2, 2, 4, 2, 2]
    assert out[0][0].shape == (8, 2, H) and out[0][1].shape == (8, H)
    assert sum("oom-backoff" in r.message for r in caplog.records) == 3
    # Content and order survive the splits: compare to singleton forwards.
    ref_reducer = RC._CaptureReducer([0], H)
    ref = _StubModel(ref_reducer)
    for i in range(8):
        r = RC._capture_forward(ref, ref_reducer, [metas[i]], [seqs[i]], 0)
        np.testing.assert_array_equal(out[0][0][i], r[0][0][0])
        np.testing.assert_array_equal(out[0][1][i], r[0][1][0])


def test_oom_at_batch_size_one_raises(caplog):
    rng = np.random.default_rng(4)
    rows = _mk_rows(rng, 4)
    pw = ("prompt_last",)
    reducer = RC._CaptureReducer([0], H)
    hf = _StubModel(reducer, fail_over_rows=0)
    metas, seqs = _metas_seqs(rows, pw)
    stats = {"n_backoffs": 0}
    with caplog.at_level(logging.WARNING), pytest.raises(RC._OOM_ERROR):
        RC._capture_batch_with_backoff(hf, reducer, metas, seqs, 0, stats)
    # 4 split to 2+2, the first 2 split to 1+1, the first singleton raises.
    assert stats["n_backoffs"] == 2
    assert [s[0] for s in hf.batch_shapes] == [4, 2, 1]
    assert any("no further fallback" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. Row-order and content identity vs the old fixed-size path
# ---------------------------------------------------------------------------


def test_new_path_identical_content_and_row_order_to_old_path():
    rng = np.random.default_rng(7)
    # 23 mixed rows plus two long rows that must run alone under the budget.
    rows = _mk_rows(rng, 23, extra_lengths=[40, 55])
    pw = ("prompt_last", "resp_last")
    pad_id = 0

    red_old = RC._CaptureReducer([0, 2], H)
    old = _StubModel(red_old)
    pos_old, y_old = _old_path(old, red_old, rows, pw, pad_id, batch_size=8)

    red_new = RC._CaptureReducer([0, 2], H)
    new = _StubModel(red_new)
    lengths = [len(r["prompt_ids"]) + len(r["comp_ids"]) for r in rows]
    plan = RC._plan_capture_batches(lengths, max_rows=8, token_budget=60)
    # The plan really regroups (long rows alone) and re-orders the forwards.
    assert any(len(b) == 1 for b in plan)
    assert [i for b in plan for i in b] != list(range(len(rows)))
    stats = {"n_backoffs": 0}
    pos_new, y_new = RC._capture_shard_arrays(new, red_new, rows, pw, pad_id, plan, stats)

    assert stats["n_backoffs"] == 0
    for li in (0, 2):
        assert pos_new[li].dtype == pos_old[li].dtype == np.float32
        np.testing.assert_array_equal(pos_new[li], pos_old[li])
        np.testing.assert_array_equal(y_new[li], y_old[li])
    # The single-row forwards went out unpadded with mask=None.
    single_calls = [m for m, s in zip(new.masks, new.batch_shapes, strict=True) if s[0] == 1]
    assert single_calls and all(m is None for m in single_calls)


def test_shard_arrays_row_order_under_oom_splits():
    """Backoff inside the planned batches must not perturb row order either."""
    rng = np.random.default_rng(9)
    rows = _mk_rows(rng, 12)
    pw = ("prompt_last",)

    red_ref = RC._CaptureReducer([0], H)
    ref = _StubModel(red_ref)
    pos_ref, y_ref = _old_path(ref, red_ref, rows, pw, 0, batch_size=4)

    red_new = RC._CaptureReducer([0], H)
    new = _StubModel(red_new, fail_over_rows=1)  # every multi-row batch OOMs
    lengths = [len(r["prompt_ids"]) + len(r["comp_ids"]) for r in rows]
    plan = RC._plan_capture_batches(lengths, max_rows=4, token_budget=10**9)
    stats = {"n_backoffs": 0}
    pos_new, y_new = RC._capture_shard_arrays(new, red_new, rows, pw, 0, plan, stats)

    assert stats["n_backoffs"] > 0
    np.testing.assert_array_equal(pos_new[0], pos_ref[0])
    np.testing.assert_array_equal(y_new[0], y_ref[0])
