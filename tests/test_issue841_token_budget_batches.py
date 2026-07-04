# ruff: noqa: E402
"""Token-budget batching for the issue #841 scaling-capture GPU phase (crash-fix round 4).

Attempt 4 OOMed in the Qwen2 MLP forward (78.75/79.25 GiB): a FIXED batch_size x LEFT-PAD
padded every sequence to the batch max, so one long lmsys prompt (4-8k tokens) inflated a
32-seq batch past 80 GiB. The fix caps each batch by PADDED tokens
(`n_seqs x max_len_in_batch <= token_budget`) with a hard seq cap, NO truncation. These
tests pin the two invariants that make the fix safe:

  * the planner respects the padded-token budget + seq cap, isolates a long outlier into
    its own batch, and returns a BIJECTION of the input indices (no row lost/duplicated) —
    the property that guarantees the caller's scatter fills every stream position exactly
    once, restoring stream order despite the internal length-sort;
  * a single prompt longer than the budget fails loud (no silent truncation — the parent
    capture regime had none, so truncation would be a second variable);
  * `_capture_prompts_budgeted` scatters each batch's rows back to their ORIGINAL indices,
    so the returned (N, L, H) is in stream order regardless of sort_by_length.

They exercise the LIVE dispatched planner `issue841_scaling_capture.plan_token_budget_batches`
(the one `_capture_prompts_budgeted` → `capture_prompts` / `kill_a_spot_gate` call), per the
"verification gates test the live dispatched path" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue841_scaling_capture as CAP
import numpy as np
import pytest


def _assert_valid_partition(batches, lengths, *, token_budget, max_seqs):
    seen = []
    for b in batches:
        assert len(b) <= max_seqs, (len(b), max_seqs)
        cost = len(b) * max(lengths[i] for i in b)
        assert cost <= token_budget, (cost, token_budget, b)
        seen.extend(b)
    # bijection: every original index exactly once
    assert sorted(seen) == list(range(len(lengths))), "not a bijection of input indices"


@pytest.mark.parametrize("sort_by_length", [True, False])
def test_plan_budget_and_bijection_with_outlier(sort_by_length):
    # 40 short prompts + one 8000-token outlier; budget 65536, seq cap 48.
    lengths = [50] * 40 + [8000]
    batches = CAP.plan_token_budget_batches(
        lengths, token_budget=65536, max_seqs=48, sort_by_length=sort_by_length
    )
    _assert_valid_partition(batches, lengths, token_budget=65536, max_seqs=48)
    # the 8000-tok outlier cannot share a batch (2 x 8000 = 16000 <= budget, but the 48-cap
    # short batch of 50-tok rows would blow the budget if the outlier joined: 48 x 8000).
    outlier_batch = next(b for b in batches if 40 in b)
    assert len(outlier_batch) * 8000 <= 65536, outlier_batch


def test_plan_seq_cap_binds_on_short_prompts():
    # 200 tiny prompts, huge budget → seq cap (not budget) limits batch size.
    lengths = [4] * 200
    batches = CAP.plan_token_budget_batches(
        lengths, token_budget=10**9, max_seqs=48, sort_by_length=True
    )
    _assert_valid_partition(batches, lengths, token_budget=10**9, max_seqs=48)
    assert all(len(b) <= 48 for b in batches)
    assert max(len(b) for b in batches) == 48  # cap actually binds


def test_plan_single_row_over_budget_raises():
    lengths = [50, 70000, 60]  # row 1 alone > 65536
    with pytest.raises(RuntimeError, match="EPM_I841S_TOKEN_BUDGET"):
        CAP.plan_token_budget_batches(lengths, token_budget=65536, max_seqs=48, sort_by_length=True)


def _fake_env(monkeypatch, lengths_by_prompt):
    """Monkeypatch the tokenizer/model helpers so _capture_prompts_budgeted runs GPU-free."""

    class _FakeTok:
        def __call__(self, t, padding=False, return_tensors=None):
            return {"input_ids": np.zeros((1, lengths_by_prompt[t]), dtype=np.int64)}

    class _FakeModel:
        class config:
            hidden_size = 4

    monkeypatch.setattr(CAP, "_chat_texts", lambda tok, ps: list(ps))
    monkeypatch.setattr(CAP, "_assert_generation_suffix", lambda tok, text: None)
    # _tokenize_left_pad passes chunk_texts straight through as the "ids" the capture reads.
    monkeypatch.setattr(CAP, "_tokenize_left_pad", lambda tok, texts: (list(texts), None))
    return _FakeModel(), _FakeTok()


def _encode_idx(chunk_ids, n_layers):
    # each "pK" prompt -> a block whose [k,:,:] == K, so a correct scatter yields arange(n)
    arr = np.zeros((len(chunk_ids), n_layers, 4), dtype=np.float32)
    for k, t in enumerate(chunk_ids):
        arr[k, :, :] = float(int(t[1:]))
    return arr


def test_budgeted_capture_scatters_to_stream_order(monkeypatch):
    prompts = [f"p{i}" for i in range(7)]
    lengths = {p: length for p, length in zip(prompts, [30, 5, 20, 8, 25, 3, 15], strict=True)}
    model, tok = _fake_env(monkeypatch, lengths)
    monkeypatch.setattr(
        CAP,
        "capture_last_token_batched",
        lambda m, ids, attn, layers: _encode_idx(ids, len(layers)),
    )
    last, adj = CAP._capture_prompts_budgeted(
        model,
        tok,
        prompts,
        layers=[0, 1],
        token_budget=60,
        max_seqs=48,
        sort_by_length=True,
        want_adjacent=False,
    )
    assert adj is None
    assert last.shape == (7, 2, 4)
    for i in range(7):  # stream order restored despite the internal length-sort
        assert last[i, 0, 0] == float(i), (i, last[i, 0, 0])


def test_budgeted_capture_adjacent_scatters_to_input_order(monkeypatch):
    # KILL-A path: sort_by_length=False, want_adjacent=True; probe rows must stay at 0..n-1.
    prompts = [f"p{i}" for i in range(6)]
    lengths = {p: length for p, length in zip(prompts, [40, 40, 40, 5, 5, 5], strict=True)}
    model, tok = _fake_env(monkeypatch, lengths)

    def _fake_two(m, ids, attn, layers):
        last = _encode_idx(ids, len(layers))
        return last, last + 1000.0  # adjacent encodes idx + 1000

    monkeypatch.setattr(CAP, "_capture_last_two_batched", _fake_two)
    last, adj = CAP._capture_prompts_budgeted(
        model,
        tok,
        prompts,
        layers=[0],
        token_budget=80,
        max_seqs=48,
        sort_by_length=False,
        want_adjacent=True,
    )
    assert last.shape == (6, 1, 4) and adj.shape == (6, 1, 4)
    for i in range(6):
        assert last[i, 0, 0] == float(i), (i, last[i, 0, 0])
        assert adj[i, 0, 0] == float(i) + 1000.0, (i, adj[i, 0, 0])
