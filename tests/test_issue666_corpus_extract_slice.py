# ruff: noqa: RUF003
# Intentional scientific Unicode (Σ, λ, ⁻¹) in docstrings + asserts.
"""issue #666 Phase-4 corpus_extract round-5 concern closures.

Two pre-existing concerns on ``scripts/issue666_corpus_extract.py``, untouched by
rounds 3/4 (which scoped to other files), closed here:

  Concern A (corpus-extract-slice-default-n3000, NIT): ``--slice`` must run ≤ N_SLICE
    (=8) synthetic CPU contexts even when ``--n-contexts`` defaults to 3000. The
    ceiling is ``_resolve_slice_n(n) = min(n, N_SLICE)``; ``--n-contexts`` can lower
    the slice but never raise it above the cap.
  Concern B (corpus-extract-slice-skips-model-forward, CONCERN): documentation
    closure — the smoke is CPU-only by design. ``_synthetic_vectors`` exercises the
    post-extraction Σ_c → CV-λ-inverse code path on PRODUCTION-shaped tensors
    (N, 28, 3584), but NOT the Qwen forward (first-validated at production launch).
    These tests anchor that contract: the synthetic tensor carries the exact
    production shape, and ``compute_sigma_c`` consumes it through the real
    ``leakage_predictor.estimate_sigma_inv`` path.

All offline (synthetic vectors); CPU-only, no network, no GPU. The full-dim
(d=3584) Σ_c eigendecomposition is the slow production-scale step (~3 min on
CPU) — by design run on the pod, NOT in this suite — so the post-extraction
code-path test uses a tractable hidden dim while a separate fast test pins the
literal production shape contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _LazyModule:
    """Proxy that imports a per-issue script on first attribute access."""

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


ce = _LazyModule("issue666_corpus_extract")


# ───────────────────────── Concern A (slice ceiling, NIT) ─────────────────────
def test_slice_default_n_capped_at_n_slice():
    """``--slice`` with the DEFAULT n_contexts (3000) resolves to ≤ N_SLICE contexts.

    The buggy code ran ``args.n_contexts`` (default 3000) under ``--slice``; the fix
    caps it at N_SLICE so the smoke caller need not pass ``--n-contexts 8`` by hand.
    """
    import issue666_corpus_extract as mod

    assert mod.N_SLICE == 8
    # The smoke caller passes NO --n-contexts → inherits DEFAULT_N_CONTEXTS (3000).
    resolved = mod._resolve_slice_n(mod.DEFAULT_N_CONTEXTS)
    assert resolved == mod.N_SLICE == 8
    assert resolved <= mod.N_SLICE


def test_slice_n_contexts_can_lower_but_not_raise_slice_size():
    """``--n-contexts`` below the cap shrinks the slice; above the cap is clamped."""
    import issue666_corpus_extract as mod

    # A caller asking for a SMALLER heavier-but-tinier smoke gets exactly that.
    assert mod._resolve_slice_n(3) == 3
    assert mod._resolve_slice_n(1) == 1
    # A caller asking ABOVE the cap is clamped to N_SLICE (never runs 3000 on CPU).
    assert mod._resolve_slice_n(100) == mod.N_SLICE
    assert mod._resolve_slice_n(mod.DEFAULT_N_CONTEXTS) == mod.N_SLICE
    # The ceiling holds for every input.
    for n in (1, 5, 8, 9, 50, 3000):
        assert mod._resolve_slice_n(n) <= mod.N_SLICE


def test_slice_synthetic_vectors_respect_resolved_count():
    """``_synthetic_vectors(_resolve_slice_n(default))`` yields ≤ N_SLICE contexts.

    End-to-end of the slice branch's data step (sans the slow full-dim Σ_c): the
    tensor the smoke feeds downstream has at most N_SLICE rows.
    """
    import issue666_corpus_extract as mod

    n = mod._resolve_slice_n(mod.DEFAULT_N_CONTEXTS)
    vecs = mod._synthetic_vectors(n)
    assert vecs.shape[0] == n <= mod.N_SLICE


# ───────────────── Concern B (smoke-vs-production boundary, CONCERN) ───────────
def test_synthetic_vectors_carry_exact_production_shape():
    """``_synthetic_vectors`` emits the EXACT production output shape (N, 28, 3584).

    The smoke substitutes these for the real Qwen ``_last_input_vectors`` output, so
    they must match its production shape on the layer + hidden axes. This is the
    cheap half of the smoke-vs-production contract: smoke verifies the Σ_c estimator
    handles real-shape tensors; production verifies the Qwen forward produces them.
    """
    import issue666_corpus_extract as mod

    n = mod.N_SLICE
    vecs = mod._synthetic_vectors(n)
    assert vecs.shape == (n, mod.EXPECTED_LAYERS, mod.EXPECTED_HIDDEN)
    assert vecs.shape[1:] == (28, 3584)
    assert vecs.dtype == np.float32


def test_compute_sigma_c_post_extraction_path_on_production_axes():
    """``compute_sigma_c`` consumes production-AXIS (N, 28, d) synthetic vectors
    through the REAL ``leakage_predictor.estimate_sigma_inv`` post-extraction path.

    This is the post-extraction code path the smoke exercises — Σ_c = E[ccᵀ] at the
    layer → CV-λ ridge → Σc⁻¹ — on synthetic vectors carrying the production layer +
    context axes (N=8, 28 layers). The hidden dim is tractable here (the full d=3584
    eigendecomposition is the ~3-min production-scale step, run on the pod, not in
    this suite); the AXIS structure + the estimator wiring are what the smoke pins.
    The real Qwen forward that produces these vectors is first-validated at
    production launch (the experimenter step). Asserts the inverse is square at the
    hidden dim, the n_contexts / layer / dim metadata round-trip, and the
    broad-corpus headline-eligibility flag is set.
    """
    import issue666_corpus_extract as mod

    n, n_layers, d = 8, mod.EXPECTED_LAYERS, 48
    vecs = mod._synthetic_vectors(n, nl=n_layers, d=d)
    assert vecs.shape == (n, 28, d)
    sig = mod.compute_sigma_c(vecs, layer=mod.PRIMARY_LAYER)
    # The Σc⁻¹ is square at the hidden dim — the post-extraction estimator ran.
    assert sig["Sigma_inv"].shape == (d, d)
    assert sig["Sigma_c"].shape == (d, d)
    assert sig["dim"] == d
    assert sig["layer"] == mod.PRIMARY_LAYER
    assert sig["n_contexts"] == n
    # N (=8) ≤ d → the raw Σc is rank-deficient (the CV-λ ridge is what conditions it).
    assert sig["rank_deficient"] is True
    # Broad corpus (corpus_kind="broad" inside compute_sigma_c) → headline-eligible,
    # unlike the design-doc-FORBIDDEN n=50 battery whitening.
    assert sig["headline_eligible"] is True
    assert np.isfinite(sig["cond_number"]) and sig["cond_number"] > 0
    # λ is a real grid value (a regularized, finite ridge was selected).
    assert np.isfinite(sig["lam"]) and sig["lam"] >= 0


def test_compute_sigma_c_layer_selects_correct_slice():
    """``compute_sigma_c`` whitens the requested LAYER's context vectors, not another.

    Builds a synthetic tensor where one layer's contexts are scaled up; asserts the
    Σ_c at that layer differs from Σ_c at another layer — i.e. the layer index threads
    through the post-extraction slice correctly (the production read uses layer 14).
    """
    import issue666_corpus_extract as mod

    n, n_layers, d = 8, mod.EXPECTED_LAYERS, 40
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((n, n_layers, d)).astype(np.float32)
    vecs[:, 5, :] *= 10.0  # layer 5 has a much larger scale than the rest
    sig5 = mod.compute_sigma_c(vecs, layer=5)
    sig6 = mod.compute_sigma_c(vecs, layer=6)
    # The whitening covariance reflects the per-layer scale → the two layers' Σ_c
    # diagonals differ markedly (layer 5 was scaled 10×).
    assert np.mean(np.diag(sig5["Sigma_c"])) > 10.0 * np.mean(np.diag(sig6["Sigma_c"]))
