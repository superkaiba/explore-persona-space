"""Issue #744 round-2 regression tests — dump-script + corpus-build invariants.

Pins the two reconciler-binding crash fixes + the two folded concerns:

* **C1 — GPU surprisal device.** ``_surprisal_from_logits`` allocates its output
  on ``logits.device`` and returns a CPU tensor, so the CUDA production path (the
  CPU smoke could not exercise) never raises the cross-device indexed-assignment
  error. Guarded by ``cuda.is_available()`` (skips on the CPU CI VM) + a CPU-side
  device-consistency check that always runs.
* **C2 — NS overlapping-chunk no-truncation.** A >1024-token story is processed in
  overlapping chunks and reassembled to cover the FULL story; every word_end index
  appears in the assembled position set (no tail dropped). Tests the
  chunker/assembler directly (no forward pass).
* **broader-budget floor.** ``build_broader``'s production path iterates the full
  shuffled corpus until the token budget is met; the loud ``>= 0.9 * budget``
  assertion fires on a genuine corpus shortfall.
* **reservoir random-pair pool.** ``ReservoirVectorPool`` accumulates >= N raw
  vectors per layer over a synthetic stream, is per-layer shaped, and feeds the
  expected per-flavor baseline payload shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue744_dump_and_stream as dump  # noqa: E402

from explore_persona_space.analysis.continuity import (  # noqa: E402
    ReservoirVectorPool,
    make_flavors_from_stats,
    random_baseline,
)

# ── C1: surprisal device-consistency ────────────────────────────────────────────


def test_surprisal_returns_cpu_tensor_cpu_inputs():
    """CPU inputs: helper returns a CPU (T,) tensor with pos0 = NaN (contract)."""
    T, V = 5, 8
    logits = torch.zeros(1, T, V)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    for t in range(1, T):
        logits[0, t - 1, ids[0, t]] = 10.0
    out = dump._surprisal_from_logits(logits, ids)
    assert out.device.type == "cpu"
    assert out.shape == (T,)
    assert torch.isnan(out[0])
    for t in range(1, T):
        assert out[t] < 0.1  # the 10.0 spike dominates the softmax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only device-mismatch path")
def test_surprisal_from_cuda_logits_no_device_error():
    """C1: CUDA logits + CUDA ids must not raise the cross-device assignment error.

    Reproduces the production GPU path (``device_map={'':cuda:0}`` -> CUDA logits)
    that the CPU smoke could never exercise. Asserts (a) no RuntimeError, (b) the
    returned tensor is on CPU per the contract, (c) values match a CPU reference.
    """
    T, V = 6, 10
    logits_cpu = torch.zeros(1, T, V)
    ids_cpu = torch.tensor([[1, 2, 3, 4, 5, 6]])
    for t in range(1, T):
        logits_cpu[0, t - 1, ids_cpu[0, t]] = 8.0
    ref = dump._surprisal_from_logits(logits_cpu, ids_cpu)  # CPU reference

    logits_cuda = logits_cpu.cuda()
    ids_cuda = ids_cpu.cuda()
    out = dump._surprisal_from_logits(logits_cuda, ids_cuda)  # must not raise
    assert out.device.type == "cpu", out.device
    assert out.shape == (T,)
    assert torch.isnan(out[0])
    assert torch.allclose(out[1:], ref[1:], atol=1e-5), (out, ref)


# ── C2: NS overlapping-chunk no-truncation ───────────────────────────────────────


def test_chunk_starts_cover_full_story():
    """Overlapping chunk starts cover [0, T_full); the last chunk ends at T_full."""
    max_len, stride = 1024, 512
    # short story -> single chunk
    assert dump._chunk_starts(900, max_len, stride) == [0]
    # >1 chunk: last start = T_full - max_len so the last chunk ends exactly at T_full
    for t_full in (1300, 1500, 2500, 3001):
        starts = dump._chunk_starts(t_full, max_len, stride)
        assert starts[0] == 0
        assert starts[-1] == t_full - max_len
        assert starts[-1] + max_len == t_full  # tail covered exactly
        # union of chunk spans covers every position
        covered = set()
        for s in starts:
            covered.update(range(s, min(s + max_len, t_full)))
        assert covered == set(range(t_full)), t_full


def test_overlapping_assembly_covers_every_word_end_no_truncation():
    """C2: a >1024-token story's FULL position set (incl. the tail) is assembled.

    Builds synthetic per-chunk reads (no forward pass) for a 1500-token story at
    max_len=1024/stride=512 and asserts the reassembled tensors cover all 1500
    positions — so every word_end index (incl. those past 1024, the part the old
    ``token_ids[:max_len]`` truncation dropped) is present.
    """
    n_layers, hidden = 3, 4
    max_len, stride, context_floor = 1024, 512, 6
    t_full = 1500
    starts = dump._chunk_starts(t_full, max_len, stride)
    assert max(s + max_len for s in starts) >= t_full  # the tail is in some chunk
    # Synthetic chunk reads: encode the ABSOLUTE position in every hidden value so
    # the assembled tensor's provenance is checkable position-by-position.
    chunk_outputs = []
    for s in starts:
        c = min(s + max_len, t_full) - s
        H_chunk = torch.zeros(n_layers, c, hidden)
        surp_chunk = torch.zeros(c)
        for local in range(c):
            H_chunk[:, local] = float(s + local)  # absolute position marker
            surp_chunk[local] = float(s + local)
        chunk_outputs.append((s, H_chunk, surp_chunk))

    H_full, surp_full = dump.assemble_overlapping_chunks(
        chunk_outputs, t_full, n_layers, hidden, context_floor
    )
    assert H_full.shape == (n_layers, t_full, hidden)
    assert surp_full.shape == (t_full,)
    # Every position covered (no NaN anywhere in H), tail included.
    assert not torch.isnan(H_full).any(), "assembled H has uncovered (NaN) positions"
    # Each assembled position carries the correct absolute-position marker (it came
    # from a chunk that actually contains it).
    for p in range(t_full):
        assert torch.allclose(H_full[:, p], torch.full((n_layers, hidden), float(p))), p
    # A word_end index past the old 1024 truncation point IS present (the C2 bug).
    late_word_end = 1499
    assert late_word_end < t_full
    assert torch.allclose(
        H_full[:, late_word_end], torch.full((n_layers, hidden), float(late_word_end))
    )
    # Whole-story position 0 surprisal is NaN (no preceding context); the rest finite.
    assert torch.isnan(surp_full[0])
    assert torch.isfinite(surp_full[1:]).all()


def test_overlapping_assembly_uses_last_qualifying_chunk():
    """De-dup picks the LAST chunk where a position has >= context_floor left-context.

    A position present in two chunks should be read from the later chunk when it
    has enough in-chunk left-context there — encode a per-CHUNK tag so we can tell
    which chunk a position was assembled from.
    """
    n_layers, hidden = 2, 3
    max_len, stride, context_floor = 1024, 512, 6
    t_full = 1500
    starts = dump._chunk_starts(t_full, max_len, stride)  # e.g. [0, 476]
    chunk_outputs = []
    for ci, s in enumerate(starts):
        c = min(s + max_len, t_full) - s
        H_chunk = torch.full((n_layers, c, hidden), float(ci))  # tag = chunk index
        surp_chunk = torch.full((c,), float(ci))
        chunk_outputs.append((s, H_chunk, surp_chunk))
    H_full, _ = dump.assemble_overlapping_chunks(
        chunk_outputs, t_full, n_layers, hidden, context_floor
    )
    last_start = starts[-1]
    # A position well inside the LAST chunk (>= context_floor after its start) must
    # come from the last chunk (tag == len(starts)-1), not an earlier overlapping one.
    p = last_start + context_floor + 10
    assert p < t_full
    assert torch.allclose(H_full[:, p], torch.full((n_layers, hidden), float(len(starts) - 1))), p


# ── broader-budget floor ─────────────────────────────────────────────────────────


class _CharTokenizer:
    """Trivial deterministic tokenizer: one subword per character (no HF needed)."""

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=False):
        ids = [ord(ch) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_take_until_token_budget_iterates_full_corpus():
    """Budget is met by consuming MANY short docs (not capped at a first-N slice)."""
    import issue744_build_corpora as bc

    tok = _CharTokenizer()
    # 5000 docs of 20 chars each = 20 tokens/doc; min_seq_tokens=7, max_seq_len=1024.
    docs = ["wikitext paragraph!" for _ in range(5000)]  # 19 chars -> 19 tokens
    budget = 10_000  # needs ~527 docs of 19 tokens -> well past any first-N pool
    seqs = bc.take_until_token_budget(docs, tok, budget, 1024, 7)
    total = sum(s["n_tokens"] for s in seqs)
    assert total >= budget, (total, budget)
    assert len(seqs) > 100  # consumed far more than a tiny candidate slice


def test_take_until_token_budget_returns_all_when_corpus_exhausts():
    """When the corpus is too small for the budget, every usable doc is returned.

    The caller's production assertion (>= 0.9 * budget) is what flags the genuine
    shortfall — the function itself does not silently cap.
    """
    import issue744_build_corpora as bc

    tok = _CharTokenizer()
    docs = ["short doc text!!" for _ in range(10)]  # 16 tokens each, 10 docs = 160
    budget = 1_000_000  # unreachable
    seqs = bc.take_until_token_budget(docs, tok, budget, 1024, 7)
    assert len(seqs) == 10  # all usable docs taken, not capped
    total = sum(s["n_tokens"] for s in seqs)
    assert total < 0.9 * budget  # the caller's loud assert would fire here


# ── reservoir random-pair pool ───────────────────────────────────────────────────


def test_reservoir_pool_accumulates_per_layer_and_caps():
    """ReservoirVectorPool fills a fixed per-layer pool over a synthetic stream."""
    torch.manual_seed(0)
    n_layers, hidden, pool_size = 4, 8, 500
    res = ReservoirVectorPool(n_layers, hidden, pool_size, seed=744)
    # Stream more tokens than the pool holds, across several sequences.
    for T in (300, 400, 250, 350):
        res.update(torch.randn(n_layers, T, hidden))
    pool = res.pool()
    assert pool.shape == (n_layers, pool_size, hidden)  # capped at pool_size
    assert res._seen == 300 + 400 + 250 + 350


def test_reservoir_pool_underfill_returns_actual_count():
    """Fewer tokens than pool_size -> pool holds exactly the streamed count."""
    n_layers, hidden, pool_size = 2, 5, 1000
    res = ReservoirVectorPool(n_layers, hidden, pool_size, seed=1)
    res.update(torch.randn(n_layers, 120, hidden))
    res.update(torch.randn(n_layers, 80, hidden))
    pool = res.pool()
    assert pool.shape == (n_layers, 200, hidden)  # 120 + 80, not pool_size


def test_reservoir_baseline_payload_shape():
    """The reservoir feeds a per-flavor baseline of the expected (L,) shape."""
    torch.manual_seed(2)
    n_layers, hidden, pool_size = 3, 16, 800
    res = ReservoirVectorPool(n_layers, hidden, pool_size, seed=744)
    for _ in range(4):
        res.update(torch.randn(n_layers, 300, hidden))
    pool = res.pool().float()
    mu = torch.zeros(n_layers, hidden)
    sigma = torch.ones(n_layers, hidden)
    rogue = torch.stack([torch.tensor([0, 1, 2]) for _ in range(n_layers)])
    flavors = make_flavors_from_stats(pool, mu, sigma, rogue)
    per_flavor = {
        f: random_baseline(H, n_pairs=5000, seed=744).tolist() for f, H in flavors.items()
    }
    assert set(per_flavor) == {"raw", "std", "ablate"}
    for f, base in per_flavor.items():
        assert len(base) == n_layers, (f, len(base))
