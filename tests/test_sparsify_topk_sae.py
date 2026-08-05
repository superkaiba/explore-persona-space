"""Numeric parity tests for the ported TopK SAE encoder (task #2061).

Round A Test scope (plan §Design "Loader adapter"): assert shape + top-k index
pattern + scatter correctness on tiny synthetic tensors. The full FVE-parity
smoke gate against the `sparsify` package's own encode is deferred to the P1
preamble (needs the 8.6 GB SAE download + ~1000 LMSYS activations + a `sparsify`
one-off install; not a unit test).
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.analysis.sparsify_topk_sae import topk_encode, topk_reconstruct


@pytest.fixture
def tiny_synthetic_sae() -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Constructed so TopK output is deterministic.

    d_in=4, d_sae=8, k=3. W_enc identity-like so pre_acts = x + b_enc, then the
    top-3 features per row are known by construction.
    """
    torch.manual_seed(0)
    d_in, d_sae, k = 4, 8, 3
    n = 5
    # Row-i, col-j pre_act = i * 10 + j; top-3 per row are the LARGEST j (cols 5,6,7).
    W_enc = torch.zeros(d_sae, d_in)
    W_enc[0, 0] = 1.0  # col-0 activation feeds feature-0
    b_enc = torch.arange(d_sae, dtype=torch.float32)  # 0..7
    W_dec = torch.eye(d_sae, d_in) if d_sae <= d_in else torch.zeros(d_sae, d_in)
    W_dec = torch.zeros(d_sae, d_in)
    W_dec[:d_in, :] = torch.eye(d_in)  # first d_in features decode to identity
    b_dec = torch.zeros(d_in)
    weights = {
        "encoder.weight": W_enc,
        "encoder.bias": b_enc,
        "W_dec": W_dec,
        "b_dec": b_dec,
    }
    x = torch.arange(n * d_in, dtype=torch.float32).reshape(n, d_in) * 10.0
    return x, weights, k


def test_topk_encode_shape(tiny_synthetic_sae):
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    n, _ = x.shape
    d_sae = weights["encoder.bias"].shape[0]
    assert z.shape == (n, d_sae), f"expected (n={n}, d_sae={d_sae}), got {tuple(z.shape)}"


def test_topk_encode_sparsity(tiny_synthetic_sae):
    """Every row must have EXACTLY k nonzero entries."""
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    per_row_nonzero = (z != 0).sum(dim=-1)
    assert torch.all(per_row_nonzero == k), (
        f"expected {k} nonzeros per row, got {per_row_nonzero.tolist()}"
    )


def test_topk_encode_selects_largest(tiny_synthetic_sae):
    """The kept indices per row must be the actual top-k argmax over pre_acts."""
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    W_enc = weights["encoder.weight"]
    b_enc = weights["encoder.bias"]
    pre_acts = x @ W_enc.T + b_enc
    expected_topk = torch.topk(pre_acts, k=k, dim=-1)
    for row in range(x.shape[0]):
        kept_idx = torch.nonzero(z[row]).flatten().sort().values
        expected_idx = expected_topk.indices[row].sort().values
        assert torch.equal(kept_idx, expected_idx), (
            f"row {row}: kept {kept_idx.tolist()} vs expected {expected_idx.tolist()}"
        )


def test_topk_encode_preserves_values(tiny_synthetic_sae):
    """The kept values must equal the pre-activation values exactly."""
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    W_enc = weights["encoder.weight"]
    b_enc = weights["encoder.bias"]
    pre_acts = x @ W_enc.T + b_enc
    for row in range(x.shape[0]):
        kept_idx = torch.nonzero(z[row]).flatten()
        for idx in kept_idx:
            assert torch.isclose(z[row, idx], pre_acts[row, idx]), (
                f"row {row} idx {idx}: kept {z[row, idx]} vs pre-act {pre_acts[row, idx]}"
            )


def test_topk_reconstruct_shape(tiny_synthetic_sae):
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    recon = topk_reconstruct(z, weights)
    assert recon.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(recon.shape)}"


def test_topk_encode_zero_input():
    """Zero input yields pre_acts = b_enc; top-k selects the largest biases."""
    d_in, d_sae, k = 4, 8, 3
    weights = {
        "encoder.weight": torch.zeros(d_sae, d_in),
        "encoder.bias": torch.arange(d_sae, dtype=torch.float32),
        "W_dec": torch.zeros(d_sae, d_in),
        "b_dec": torch.zeros(d_in),
    }
    x = torch.zeros(2, d_in)
    z = topk_encode(x, weights, k=k)
    # Top-3 biases are indices 5, 6, 7 (values 5, 6, 7).
    for row in range(2):
        kept_idx = torch.nonzero(z[row]).flatten().sort().values.tolist()
        assert kept_idx == [5, 6, 7], f"row {row}: {kept_idx}"


def test_topk_encode_sparse_scatter_equals_dense(tiny_synthetic_sae):
    """topk_encode_sparse (the P1 storage path, #2061 review M1) scatters back
    to EXACTLY topk_encode's dense output — same torch.topk selection."""
    from explore_persona_space.analysis.sparsify_topk_sae import topk_encode_sparse

    x, weights, k = tiny_synthetic_sae
    dense = topk_encode(x, weights, k=k)
    vals, idx = topk_encode_sparse(x, weights, k=k)
    assert vals.shape == idx.shape == (x.shape[0], k)
    recon = torch.zeros_like(dense)
    recon.scatter_(-1, idx, vals)
    assert torch.equal(recon, dense)
