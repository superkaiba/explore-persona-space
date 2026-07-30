# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Chunked rank helper matches a naive full-argsort reference."""

import torch

from jlens.vis import _ranks_of


def _naive_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    sorted_idx = logits.argsort(-1, descending=True)
    full = torch.empty_like(sorted_idx)
    full.scatter_(1, sorted_idx, torch.arange(logits.shape[1]).expand_as(sorted_idx))
    if targets.ndim == 1:
        return full[:, targets]
    return full.gather(1, targets)


def test_ranks_of_1d_targets_matches_naive():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(37, 500, generator=g)
    targets = torch.randint(0, 500, (12,), generator=g)
    for chunk in (1, 7, 37, 256):
        got = _ranks_of(logits, targets, chunk_size=chunk)
        torch.testing.assert_close(got, _naive_ranks(logits, targets))


def test_ranks_of_2d_per_position_targets():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(29, 300, generator=g)
    targets = torch.randint(0, 300, (29, 5), generator=g)
    for chunk in (3, 29, 64):
        got = _ranks_of(logits, targets, chunk_size=chunk)
        torch.testing.assert_close(got, _naive_ranks(logits, targets))
