# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""compute_slice end-to-end on the tiny CPU model (no GPU, no transformers)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from jlens.fitting import fit
from jlens.vis import _ranks_of, build_page, compute_slice

from .tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog near the river bank"


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def lens(model):
    return fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )


def test_max_seq_len_is_honoured(model, lens):
    full = compute_slice(model, lens, PROMPT)
    assert full.seq_len == model.encode(PROMPT).shape[1]
    truncated = compute_slice(model, lens, PROMPT, max_seq_len=24)
    assert truncated.seq_len == 24
    assert (
        truncated.context_token_ids == model.encode(PROMPT, max_length=24)[0].tolist()
    )


def test_empty_tracked_does_not_crash(model, lens):
    sd = compute_slice(model, lens, PROMPT, max_tracked=0)
    assert sd.tracked_token_ids == []
    assert sd.rank_tensor.shape == (sd.seq_len, len(sd.layers), 0)


def test_final_layer_row_is_the_models_own_output(model, lens):
    sd = compute_slice(model, lens, PROMPT)
    final_idx = sd.layers.index(model.n_layers - 1)
    _, model_logits, _ = lens.apply(model, PROMPT, layers=[sd.layers[0]])
    np.testing.assert_array_equal(
        sd.top_ids[:, final_idx, 0], model_logits.argmax(-1).numpy()
    )


def test_tracked_rank_column_matches_ranks_of(model, lens):
    pin = int(model.encode(PROMPT)[0, 5])
    sd = compute_slice(model, lens, PROMPT, pinned_token_ids={pin})
    col = sd.tracked_token_ids.index(pin)
    final_idx = sd.layers.index(model.n_layers - 1)
    _, model_logits, _ = lens.apply(model, PROMPT, layers=[sd.layers[0]])
    expected = _ranks_of(model_logits, torch.tensor([pin]))[:, 0].numpy()
    np.testing.assert_array_equal(sd.rank_tensor[:, final_idx, col], expected)


def test_last_n_tokens_windows_the_tail(model, lens):
    full_ids = model.encode(PROMPT)[0].tolist()
    sd = compute_slice(model, lens, PROMPT, last_n_tokens=4)
    assert sd.seq_len == 4
    assert sd.top_ids.shape[0] == 4
    # The page still gets the whole prompt; the slice covers the tail.
    assert sd.context_token_ids == full_ids
    assert sd.ctx_offset == len(full_ids) - 4

    unwindowed = compute_slice(model, lens, PROMPT)
    assert unwindowed.ctx_offset == 0


def test_layer_list_strides_but_keeps_last_fitted_and_final(model, lens):
    sd = compute_slice(model, lens, PROMPT, layer_stride=2)
    assert sd.layers == [0, 2, 3]  # fitted [0, 1, 2] strided, plus the final layer


def test_pinned_token_ids_flow_to_the_page_by_default(model, lens, tmp_path):
    pin = int(model.encode(PROMPT)[0, 3])
    sd = compute_slice(model, lens, PROMPT, pinned_token_ids={pin})
    assert sd.pinned_token_ids == [pin]
    assert pin in sd.tracked_token_ids
    build_page(sd, PROMPT, title="t", description="d", mode="fetch", out_dir=tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["pinned"] == [pin]
    assert (tmp_path / "ranks" / f"{pin}.bin").exists()
