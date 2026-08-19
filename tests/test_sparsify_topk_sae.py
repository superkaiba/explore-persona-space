"""Numeric parity tests for the ported TopK SAE encoder (task #2061).

Round A Test scope (plan §Design "Loader adapter"): assert shape + top-k index
pattern + scatter correctness on tiny synthetic tensors. The full FVE-parity
smoke gate against the `sparsify` package's own encode runs at the P1 preamble
(needs the 8.6 GB SAE download + ~1000 LMSYS activations; not a unit test).

The `test_ported_*_matches_sparsify_reference` tests below pin the encode
CONVENTIONS offline against the real `SparseCoder` on tiny synthetic weights
(guarded by `pytest.importorskip("sparsify")` — eai-sparsify is a one-off
parity reference, not a runtime dep). They exist because the GPU/weights-fenced
gate is the only other place the reference runs, and a convention mismatch
(the missing `b_dec` pre-subtraction + missing ReLU-before-top-k) reached the
pod gate: FVE 0.1157 ported vs 0.3296 reference, |Δ| 0.2139 vs the 0.05 bar.
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
    """The kept indices per row must be the actual top-k argmax over pre_acts.

    The oracle mirrors the sparsify reference convention (b_dec pre-shift +
    ReLU before top-k); on this fixture (b_dec=0, all pre_acts >= 0) it is
    numerically identical to the bare affine pre-activations.
    """
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    W_enc = weights["encoder.weight"]
    b_enc = weights["encoder.bias"]
    b_dec = weights["b_dec"]
    pre_acts = torch.relu((x - b_dec) @ W_enc.T + b_enc)
    expected_topk = torch.topk(pre_acts, k=k, dim=-1)
    for row in range(x.shape[0]):
        kept_idx = torch.nonzero(z[row]).flatten().sort().values
        expected_idx = expected_topk.indices[row].sort().values
        assert torch.equal(kept_idx, expected_idx), (
            f"row {row}: kept {kept_idx.tolist()} vs expected {expected_idx.tolist()}"
        )


def test_topk_encode_preserves_values(tiny_synthetic_sae):
    """The kept values must equal the (ReLU'd, b_dec-shifted) pre-activations."""
    x, weights, k = tiny_synthetic_sae
    z = topk_encode(x, weights, k=k)
    W_enc = weights["encoder.weight"]
    b_enc = weights["encoder.bias"]
    b_dec = weights["b_dec"]
    pre_acts = torch.relu((x - b_dec) @ W_enc.T + b_enc)
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


# ---------------------------------------------------------------------------
# Offline reference parity (the #2061 loader-parity gate, minus GPU + weights).
# ---------------------------------------------------------------------------

# fp32 CPU tolerance: matched-convention max |diff| measured 4e-6 on this
# construction (embedding_bag vs matmul decode paths); a convention bug reads
# O(1)-O(100). 1e-4 is ~25x measured headroom (bf16-parity-gate calibration
# rule: same dtype + shape as the test's own deployment — fp32 CPU here).
_PARITY_ATOL = 1e-4
_PARITY_RTOL = 1e-4


def _reference_sae(monkeypatch, d_in: int, k: int, seed: int):
    """Real `SparseCoder` with random weights, decodable on CPU.

    Pins `sparsify.sparse_coder.decoder_impl` to the package's OWN documented
    CPU fallback `eager_decode` (utils.py L82) — on a CUDA-less host the
    import-time resolution picks `triton_decode`, which needs a GPU driver.
    The triton<->eager equivalence is upstream's contract, not the port's.
    Returns (sae, weights-dict shaped like `load_sae_weights` output).
    """
    pytest.importorskip(
        "sparsify", reason="eai-sparsify is a one-off parity reference, not a runtime dep"
    )
    import sparsify.sparse_coder as sparse_coder_mod
    from sparsify import SparseCoder
    from sparsify.config import SparseCoderConfig
    from sparsify.utils import eager_decode

    monkeypatch.setattr(sparse_coder_mod, "decoder_impl", eager_decode)

    torch.manual_seed(seed)
    # Release-shaped cfg (cfg.json: expansion_factor=64, normalize_decoder=True,
    # num_latents=0, k=32, multi_topk=True; no `transcode` key -> default False),
    # scaled down. multi_topk/normalize_decoder do not touch encode/decode.
    cfg = SparseCoderConfig(
        expansion_factor=4, normalize_decoder=True, num_latents=0, k=k, multi_topk=True
    )
    sae = SparseCoder(d_in, cfg, dtype=torch.float32)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.randn_like(sae.encoder.weight))
        sae.encoder.bias.copy_(torch.randn_like(sae.encoder.bias))
        sae.W_dec.copy_(torch.randn_like(sae.W_dec))
        # NONZERO b_dec: the discriminator for the missing pre-subtraction
        # (the production gate failure — FVE 0.1157 vs 0.3296, |delta| 0.2139).
        sae.b_dec.copy_(torch.randn(d_in))
    weights = {
        "encoder.weight": sae.encoder.weight.data.clone(),
        "encoder.bias": sae.encoder.bias.data.clone(),
        "W_dec": sae.W_dec.data.clone(),
        "b_dec": sae.b_dec.data.clone(),
    }
    return sae, weights


def test_ported_encode_decode_matches_sparsify_reference(monkeypatch):
    """Ported encode/decode == real `SparseCoder.encode`/`.decode`, offline.

    The gate is GPU/weights-fenced, which is how BOTH a fabricated `decode()`
    signature and the b_dec/ReLU convention mismatch reached production. This
    pins the numeric contract on synthetic weights: encode-side (dense scatter
    of the reference (top_acts, top_indices) pair), decode-side (full
    reconstruction), and the sparse production producer (`topk_encode_sparse`,
    the P1 storage path).
    """
    from explore_persona_space.analysis.sparsify_topk_sae import topk_encode_sparse

    d_in, k, n = 16, 4, 64
    sae, weights = _reference_sae(monkeypatch, d_in=d_in, k=k, seed=0)
    x = torch.randn(n, d_in)

    with torch.no_grad():
        enc_ref = sae.encode(x)
        recon_ref = sae.decode(enc_ref.top_acts, enc_ref.top_indices)
        z_ref = torch.zeros(n, sae.num_latents).scatter_(
            -1, enc_ref.top_indices.long(), enc_ref.top_acts
        )

        z_ported = topk_encode(x, weights, k=k)
        recon_ported = topk_reconstruct(z_ported, weights)
        vals, idx = topk_encode_sparse(x, weights, k=k)
        z_sparse = torch.zeros_like(z_ported).scatter_(-1, idx, vals)

    torch.testing.assert_close(z_ported, z_ref, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)
    torch.testing.assert_close(recon_ported, recon_ref, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)
    # The P1 production producer shares the same (fixed) pre-activation path.
    torch.testing.assert_close(z_sparse, z_ref, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)


def test_ported_relu_clamps_all_negative_preacts_like_reference(monkeypatch):
    """ReLU-before-top-k leg, deterministic (random draws can miss it).

    Constructed so EVERY pre-activation is negative: x = b_dec makes the
    b_dec-shifted encoder input zero, and b_enc = -10 pushes all pre-acts to
    -10. Reference: ReLU clamps to 0 before top-k, so top_acts are all zero
    and the reconstruction is EXACTLY b_dec. An un-ReLU'd port scatters -10s
    and decodes b_dec + sum of -10 * W_dec rows instead.
    """
    d_in, k = 16, 4
    sae, weights = _reference_sae(monkeypatch, d_in=d_in, k=k, seed=1)
    with torch.no_grad():
        sae.encoder.bias.fill_(-10.0)
    weights["encoder.bias"] = sae.encoder.bias.data.clone()

    x = weights["b_dec"].unsqueeze(0).repeat(3, 1)  # (3, d_in), rows == b_dec

    with torch.no_grad():
        enc_ref = sae.encode(x)
        recon_ref = sae.decode(enc_ref.top_acts, enc_ref.top_indices)
        z_ported = topk_encode(x, weights, k=k)
        recon_ported = topk_reconstruct(z_ported, weights)

    # Sanity on the construction itself: the reference reconstructs b_dec.
    torch.testing.assert_close(recon_ref, x, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)
    assert torch.all(z_ported == 0.0), "all-negative pre-acts must scatter exact zeros"
    torch.testing.assert_close(recon_ported, recon_ref, atol=_PARITY_ATOL, rtol=_PARITY_RTOL)
