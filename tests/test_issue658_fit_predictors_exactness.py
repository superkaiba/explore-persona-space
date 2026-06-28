# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, λ, ×, ≤, Δ) in scientific docstrings + assert messages.
"""Exactness regression for the #658 predictor-fit GPU/batched performance rewrite.

The recovery-mode rewrite (2026-06-27) replaced the A3.4 ridge nested-CV LOCO
fit — previously a per-(inner fold × λ) primal refit, ``np.linalg.solve(XᵀX+λI,
XᵀY)``, the O(D³) path that ran ~40h on CPU with no output — with the EXACT
closed-form dual/PRESS leave-one-out identity (one eigendecomposition of the N×N
Gram, vectorized over the λ grid). "Exact" is the gate: the reported held-out
LOCO Spearman ρ for A3.4 / A3.5 / the chain ρ MUST NOT MOVE.

The MLP halves (A3.2 single-output, A3.5 multi-output gap) were ALSO re-batched
onto a vmapped ensemble on ``--device``. The batched MLP is not exact to machine
precision (batched GEMM vs per-net GEMV reduction order), but it must reproduce
the OLD serial loop to <= 1e-6 — AND the multi-output gap path must reproduce the
serial RESEED-PER-DIM init stream (the round-2 finding: an all-at-once seed drifts
dims 1+ by ~0.38; the tiled init fixes it).

These tests pin both invariants so a future refactor of either fast path can never
silently drift the DV away from the serial oracle and stay green:

- the fast ``_ridge_predict_loco`` reproduces the primal-refit
  ``_ridge_predict_loco_refit`` to <= 1e-6 in both predictions AND ρ;
- the batched ``_fit_mlp_loco`` (A3.2) + ``_fit_mlp_ensemble_loco`` gap path
  (A3.5) reproduce ``_fit_mlp_loco_serial_reference`` to <= 1e-6;
- the round-3 ensemble CHUNKING (``MLP_CHUNK_SIZE``, the OOM fix) is bit-/<=1e-6
  invariant to chunk size — chunk = 1, a non-divisor, exactly E, and > E all agree
  with the full-batch result, and the gap result matches the serial reference at
  every chunk size;
- the in-script ``_assert_ridge_exactness`` + ``_assert_mlp_exactness`` startup
  gates pass and report deltas within tolerance (the MLP gate now also asserts
  chunk-invariance);
- the per-cell param-hash invalidates stale checkpoint cells on a hyperparameter
  change (the resume-into-reused-out-dir stale-serve fix).

CPU-only; runs in a few seconds. No GPU, no store, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

fit = pytest.importorskip("issue658_fit_predictors")


def _synthetic(seed: int, n: int = 16, d: int = 50, p: int = 3):
    """Low-rank-signal + noise (X, Y) so ridge has real structure to fit."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    W = rng.standard_normal((4, d))
    X = z @ W + 0.1 * rng.standard_normal((n, d))
    B = rng.standard_normal((d, p))
    Y = X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))
    return X, Y


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dual_press_loco_matches_primal_refit_predictions(seed):
    """The fast dual/PRESS LOCO ridge == the primal-refit oracle, <= 1e-6 on preds.

    This is the core exactness claim: the closed-form leave-one-out identity is
    mathematically the same fit as refitting ridge on each (N-1)-row subset, so
    every held-out prediction must agree to numerical precision.
    """
    fit.DEVICE = "cpu"
    X, Y = _synthetic(seed)
    lambdas = [1e-1, 1.0, 10.0, 100.0]
    fast = fit._ridge_predict_loco(X, Y, lambdas)
    ref = fit._ridge_predict_loco_refit(X, Y, lambdas)
    max_abs = float(np.max(np.abs(fast - ref)))
    assert max_abs <= 1e-6, f"dual PRESS LOCO drifted from primal refit: max|Δpred|={max_abs:.3e}"


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dual_press_loco_matches_primal_refit_rho(seed):
    """The REPORTED statistic (per-output held-out Spearman ρ) is unchanged."""
    fit.DEVICE = "cpu"
    X, Y = _synthetic(seed)
    lambdas = [1e-1, 1.0, 10.0, 100.0]
    fast = fit._ridge_predict_loco(X, Y, lambdas)
    ref = fit._ridge_predict_loco_refit(X, Y, lambdas)
    for k in range(Y.shape[1]):
        rf = spearmanr(fast[:, k], Y[:, k]).correlation
        rr = spearmanr(ref[:, k], Y[:, k]).correlation
        if np.isnan(rf) and np.isnan(rr):
            continue
        assert abs(float(rf - rr)) <= 1e-6, f"output {k}: ρ drifted (fast {rf} vs refit {rr})"


def test_assert_ridge_exactness_gate_passes():
    """The in-script startup gate ``_assert_ridge_exactness`` passes within tol.

    main() runs this at every startup; a failure aborts the run loud. Pin it here
    so the gate itself can never be quietly weakened (e.g. tolerance loosened, or
    the oracle swapped for the fast path so it trivially compares to itself)."""
    fit.DEVICE = "cpu"
    res = fit._assert_ridge_exactness()
    assert res["tol"] == 1e-6
    assert res["max_abs_pred_delta"] <= res["tol"]
    assert res["max_rho_delta"] <= res["tol"]


def test_refit_oracle_is_distinct_from_fast_path():
    """Guard against the gate degenerating: the oracle must NOT call the fast path.

    ``_assert_ridge_exactness`` is only meaningful if the reference really is the
    independent primal-refit implementation. A direct smoke that the oracle uses
    the primal ``np.linalg.solve`` solve (not the dual one) — the two functions
    are different objects with different source.
    """
    import inspect

    ref_src = inspect.getsource(fit._ridge_predict_loco_refit)
    assert "_ridge_solve" in ref_src, "the exactness oracle must use the primal _ridge_solve refit"
    fast_src = inspect.getsource(fit._ridge_predict_loco)
    assert "_press_loo_mse_per_lambda" in fast_src, "the fast path must use the PRESS closed form"
    assert "_ridge_dual_weights" in fast_src, "the fast path must use the dual/Woodbury solve"


# ── MLP exactness (A3.2 single-output + A3.5 multi-output gap) ──────────────────


def _mlp_synthetic(seed: int, n: int = 14, d: int = 30, p: int = 6):
    """Low-rank-signal + noise (X, Y) fp32 for the MLP equivalence checks."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    W = rng.standard_normal((4, d))
    X = (z @ W + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    B = rng.standard_normal((d, p))
    Y = (X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))).astype(np.float32)
    return X, Y


@pytest.mark.parametrize("seed", [0, 3])
def test_batched_single_output_mlp_matches_serial(seed, monkeypatch):
    """The A3.2 path (``_fit_mlp_loco``) reproduces the serial reference <= 1e-6.

    The batched-vmap single-output LOCO MLP must match the OLD per-fold serial loop
    (``_fit_mlp_loco_serial_reference``) — same arch, AdamW, epochs, per-fold
    standardization, and per-fold init stream. (~3.6e-7 on CPU, reduction-order.)
    """
    monkeypatch.setattr(fit, "DEVICE", "cpu")
    monkeypatch.setattr(fit, "MLP_MAX_EPOCHS", 25)
    X, Y = _mlp_synthetic(seed)
    ser = fit._fit_mlp_loco_serial_reference(X, Y[:, 0])
    bat = fit._fit_mlp_loco(X, Y[:, 0])
    max_abs = float(np.max(np.abs(ser - bat)))
    assert max_abs <= 1e-6, f"batched single-output MLP drifted from serial: max|Δ|={max_abs:.3e}"


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("chunk", [0, 1, 7, 9999])
def test_batched_gap_mlp_matches_serial_reseed_per_dim(seed, chunk, monkeypatch):
    """The A3.5 gap path reproduces the serial RESEED-PER-DIM reference <= 1e-6.

    The OLD gap MLP called ``_fit_mlp_loco(Xc, Yv[:, k])`` once per output dim, and
    each call re-seeds ``torch.manual_seed(658)`` — so every dim reuses the SAME n
    per-fold inits. The batched ensemble must reproduce that by addressing member m
    by its block member m % n (see ``_fit_mlp_ensemble_loco``). Without that, dims
    1+ diverge ~0.38 (the round-2 finding). Run across chunk sizes — 0 (no chunk),
    1, a non-divisor (7), and > E (9999) — so the round-3 ensemble chunking is
    proven not to move the DV at ANY boundary (E = gap × n = 4 × 14 = 56).
    """
    monkeypatch.setattr(fit, "DEVICE", "cpu")
    monkeypatch.setattr(fit, "MLP_MAX_EPOCHS", 25)
    monkeypatch.setattr(fit, "MLP_CHUNK_SIZE", chunk)
    X, Y = _mlp_synthetic(seed)
    gap = 4
    ser = np.stack([fit._fit_mlp_loco_serial_reference(X, Y[:, k]) for k in range(gap)], axis=1)
    bat = fit._fit_mlp_ensemble_loco(X, Y, target_idx=list(range(gap)), seed=658)
    max_abs = float(np.max(np.abs(ser - bat)))
    assert max_abs <= 1e-6, (
        f"batched gap MLP (chunk={chunk}) drifted from serial reseed-per-dim: "
        f"max|Δ|={max_abs:.3e}. A >1e-2 delta means the per-dim init tile regressed; a "
        "chunk-dependent delta means a per-member quantity is keyed to chunk-local "
        "position instead of the global member index."
    )


@pytest.mark.parametrize("seed", [0, 5])
def test_gap_mlp_is_chunk_size_invariant(seed, monkeypatch):
    """Chunking the gap-MLP ensemble must NOT move the DV: every chunk size agrees.

    The round-3 OOM fix fits E = gap × N member-nets in chunks of MLP_CHUNK_SIZE.
    Every per-member quantity (init, standardization, target, mask, held-out row)
    is keyed to the GLOBAL member index, so chunk size 1, a non-divisor, exactly E,
    and > E must all produce the same held-out predictions to <= 1e-6. A
    chunk-dependent result is the bug this guards (a per-member tensor sliced by
    chunk-local position).
    """
    monkeypatch.setattr(fit, "DEVICE", "cpu")
    monkeypatch.setattr(fit, "MLP_MAX_EPOCHS", 25)
    X, Y = _mlp_synthetic(seed)
    gap = 5  # E = 5 * 14 = 70
    e_total = gap * X.shape[0]
    results = {}
    for chunk in [0, 1, 7, 13, e_total, e_total + 100]:  # full, 1, non-divisors, =E, >E
        monkeypatch.setattr(fit, "MLP_CHUNK_SIZE", chunk)
        results[chunk] = fit._fit_mlp_ensemble_loco(X, Y, target_idx=list(range(gap)), seed=658)
    ref = results[0]  # full-batch (no chunk)
    for chunk, r in results.items():
        max_abs = float(np.max(np.abs(r - ref)))
        assert max_abs <= 1e-6, (
            f"chunk={chunk} disagrees with full-batch: max|Δ|={max_abs:.3e} — the chunk "
            "boundary moved the DV (a per-member quantity is keyed to chunk-local position)."
        )
    # chunk = exactly E and chunk > E must be BIT-identical to full (same batch shape)
    assert np.array_equal(results[e_total], ref), "chunk=E must be bit-identical to full-batch"
    assert np.array_equal(results[e_total + 100], ref), (
        "chunk>E must be bit-identical to full-batch"
    )


def test_assert_mlp_exactness_gate_passes():
    """The in-script startup gate ``_assert_mlp_exactness`` passes within tol.

    main() runs this at every startup alongside the ridge gate; a failure aborts
    the run loud. Pin it so the gate cannot be quietly weakened. The single-output,
    the tiled+chunked multi-output gap, AND the chunk-invariance deltas must all be
    within tolerance.
    """
    res = fit._assert_mlp_exactness()
    assert res["tol"] == 1e-6
    assert res["single_delta"] <= res["tol"], res
    assert res["multi_delta"] <= res["tol"], res
    assert res["chunk_delta"] <= res["tol"], res


def test_mlp_oracle_is_distinct_from_batched_path():
    """The MLP exactness oracle must be the serial loop, not the batched path.

    ``_assert_mlp_exactness`` is only meaningful if the reference is the independent
    per-fold serial implementation. Smoke that the oracle uses a per-fold Python
    loop with a fresh per-fold optimizer (the OLD shape), and the batched path uses
    the vmapped ensemble.
    """
    import inspect

    oracle_src = inspect.getsource(fit._fit_mlp_loco_serial_reference)
    assert "for i in range(n)" in oracle_src, "the MLP oracle must be the per-fold serial loop"
    assert "torch.optim.AdamW(net.parameters()" in oracle_src, (
        "the MLP oracle must build a fresh per-fold optimizer (the old serial shape)"
    )
    batched_src = inspect.getsource(fit._fit_mlp_ensemble_loco)
    assert "vmap" in batched_src, "the batched MLP path must use torch.func.vmap"


# ── attn answer-span streaming equivalence (the #658 142 GB → ~3 GB quota fix) ──


def _write_synthetic_spans(spans_dir, ctx_ids, *, n_layers=4, hidden=8, seed=0):
    """Write synthetic answer_spans/<ctx>.pt blobs matching the extractor schema.

    Each blob is ``{"context_id", "capture_layers", "spans": [(Lc, S, H) fp16 or None],
    "probes"}`` — exactly what issue658_extract_base_store.py saves. Variable probe
    count + span length per context, with one None span mixed in, so the attn
    probe-mean (which drops None) is exercised.
    """
    import torch

    spans_dir = Path(spans_dir)
    spans_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    capture_layers = list(range(n_layers))
    for i, c in enumerate(ctx_ids):
        n_probes = 3 + i  # vary probe count per context
        spans = []
        for p in range(n_probes):
            if p == 1:  # one empty/None span per context (extractor writes None)
                spans.append(None)
                continue
            s_len = 2 + p  # vary answer-token length per probe
            arr = rng.standard_normal((n_layers, s_len, hidden)).astype(np.float32)
            spans.append(torch.from_numpy(arr).to(torch.float16))
        torch.save(
            {
                "context_id": c,
                "capture_layers": capture_layers,
                "spans": spans,
                "probes": [f"probe_{p}" for p in range(n_probes)],
            },
            spans_dir / f"{c}.pt",
        )
    return capture_layers


def _old_attn_matrix_reference(spans_dir, layer_idx, ctx_ids, attn_w):
    """The PRE-refactor per-(layer, context) attn matrix — the exactness oracle.

    A faithful copy of the old ``_attn_matrix`` body: load every context's span blob,
    attn-pool the given layer's spans (dropping None), probe-mean. The refactored
    precompute-then-slice path must reproduce this bit-for-bit.
    """
    import torch

    rows = []
    for c in ctx_ids:
        blob = torch.load(Path(spans_dir) / f"{c}.pt", weights_only=False)
        spans = blob["spans"]
        per_probe = [
            fit.summarize_answer_span(s[layer_idx], "attn", attn_weight=attn_w)
            for s in spans
            if s is not None
        ]
        rows.append(torch.stack(per_probe).mean(0).numpy())
    return np.stack(rows)


class _FakeStreamSpanSource(fit._SpanSource):
    """Offline stand-in for ``_HfStreamSpanSource``: same download→release→LRU logic.

    Reads from a local ``src_dir`` (in place of HF), but mimics the streaming
    contract: ``load_blob`` "downloads" (copies into a private staging dir), evicts
    over an LRU of ``cache_size``, and tracks the per-call download count + the peak
    number of simultaneously-resident files. ``release`` deletes the staged copy.
    Lets the streaming equivalence + footprint be tested with no network.
    """

    def __init__(self, src_dir, cache_size=1):
        import tempfile

        self.src_dir = Path(src_dir)
        self.cache_size = max(1, int(cache_size))
        self._resident = {}  # ctx -> staged path (insertion order = LRU)
        self._staging = Path(tempfile.mkdtemp(prefix="fake_stream_"))
        self.download_count = 0
        self.peak_resident = 0

    def load_blob(self, ctx_id):
        import shutil

        import torch

        path = self._resident.get(ctx_id)
        if path is None or not path.exists():
            self.download_count += 1
            staged = self._staging / f"{ctx_id}.pt"
            shutil.copyfile(self.src_dir / f"{ctx_id}.pt", staged)
            self._resident[ctx_id] = staged
            while len(self._resident) > self.cache_size:
                old = next(iter(self._resident))
                self._delete(old)
        self.peak_resident = max(self.peak_resident, len(self._resident))
        return torch.load(self._resident[ctx_id], weights_only=False)

    def _delete(self, ctx_id):
        path = self._resident.pop(ctx_id, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def release(self, ctx_id):
        self._delete(ctx_id)

    def n_resident_files(self):
        return len(list(self._staging.glob("*.pt")))


def _attn_w(hidden, seed=658):
    """The same seeded unit attn_w fit_a32 builds (torch.manual_seed; normalize)."""
    import torch

    torch.manual_seed(seed)
    w = torch.randn(hidden)
    return w / w.norm()


@pytest.mark.parametrize("seed", [0, 2])
def test_attn_summary_store_matches_old_per_layer_matrix(tmp_path, seed):
    """The refactored precompute-then-slice attn path == the old per-(layer,ctx) loop.

    ``_attn_summary_store`` loads each span blob ONCE and computes all layers; the old
    ``_attn_matrix`` loaded per layer. Both must produce bit-identical attn summaries
    for every (layer, context) — the math is unchanged, only the iteration order +
    load count differ. This is the core exactness claim of the streaming refactor.
    """
    ctx_ids = ["ctx_a", "ctx_b", "ctx_c"]
    n_layers, hidden = 4, 8
    capture_layers = _write_synthetic_spans(
        tmp_path / "answer_spans", ctx_ids, n_layers=n_layers, hidden=hidden, seed=seed
    )
    attn_w = _attn_w(hidden)
    local = fit._SpanSource(tmp_path / "answer_spans")
    summ = fit._attn_summary_store(local, ctx_ids, capture_layers, attn_w)
    for li in range(n_layers):
        new_mat = fit._attn_matrix(summ, li, ctx_ids)
        old_mat = _old_attn_matrix_reference(tmp_path / "answer_spans", li, ctx_ids, attn_w)
        max_abs = float(np.max(np.abs(new_mat - old_mat)))
        assert max_abs == 0.0, (
            f"layer {li}: refactored attn matrix drifted from the old per-layer loop "
            f"(max|Δ|={max_abs:.3e}) — the precompute must be bit-identical"
        )


@pytest.mark.parametrize("cache_size", [1, 2])
def test_streamed_attn_summary_matches_local(tmp_path, cache_size):
    """Streamed attn summary == local attn summary, bit-identical (the brief's gate).

    The streaming source downloads + deletes per context (peak ~one context); the
    local source reads in place. Both feed the SAME ``_attn_summary_store`` math with
    the SAME attn_w, so the resulting per-context summaries must agree exactly.
    """
    ctx_ids = ["c0", "c1", "c2", "c3"]
    n_layers, hidden = 4, 8
    capture_layers = _write_synthetic_spans(
        tmp_path / "answer_spans", ctx_ids, n_layers=n_layers, hidden=hidden, seed=11
    )
    attn_w = _attn_w(hidden)
    local = fit._SpanSource(tmp_path / "answer_spans")
    stream = _FakeStreamSpanSource(tmp_path / "answer_spans", cache_size=cache_size)
    local_summ = fit._attn_summary_store(local, ctx_ids, capture_layers, attn_w)
    stream_summ = fit._attn_summary_store(stream, ctx_ids, capture_layers, attn_w)
    for c in ctx_ids:
        max_abs = float(np.max(np.abs(local_summ[c] - stream_summ[c])))
        assert max_abs <= 1e-6, f"context {c}: streamed attn summary drifted: max|Δ|={max_abs:.3e}"
        # strict bit-identity expected (load path touches no tensor math)
        assert max_abs == 0.0, f"context {c}: streamed attn summary not bit-identical to local"


def test_stream_source_bounds_peak_footprint(tmp_path):
    """Streaming holds at most ``cache_size`` span files resident — the quota fix.

    With cache_size=1 over many contexts, peak resident files must be 1 (~one context),
    NOT the whole grid — the entire point of the 142 GB → ~3 GB change. After the full
    pass, releases leave nothing behind.
    """
    ctx_ids = [f"ctx_{i}" for i in range(6)]
    capture_layers = _write_synthetic_spans(tmp_path / "answer_spans", ctx_ids, seed=3)
    attn_w = _attn_w(8)
    stream = _FakeStreamSpanSource(tmp_path / "answer_spans", cache_size=1)
    fit._attn_summary_store(stream, ctx_ids, capture_layers, attn_w)
    assert stream.peak_resident <= 1, (
        f"streaming held {stream.peak_resident} contexts resident at peak (cache_size=1); "
        "footprint is not bounded to ~one context"
    )
    # each context downloaded exactly once in a single pass (no per-layer re-download)
    assert stream.download_count == len(ctx_ids), (
        f"expected {len(ctx_ids)} downloads (one per context), got {stream.download_count} — "
        "a per-layer re-download would multiply this by n_layers"
    )
    assert stream.n_resident_files() == 0, "streamed files must be released after the pass"


def test_parse_attn_stream_hf():
    """``_parse_attn_stream_hf`` splits REPO_ID:PATH_PREFIX on the first ':' only."""
    repo, prefix = fit._parse_attn_stream_hf(
        "superkaiba1/explore-persona-space-data:issue658_theory_assumptions/store/answer_spans"
    )
    assert repo == "superkaiba1/explore-persona-space-data"
    assert prefix == "issue658_theory_assumptions/store/answer_spans"
    # trailing slash stripped
    _, prefix2 = fit._parse_attn_stream_hf("r/x:a/b/")
    assert prefix2 == "a/b"
    for bad in ["no-colon", ":prefix-only", "repo-only:"]:
        with pytest.raises(SystemExit):
            fit._parse_attn_stream_hf(bad)


def test_build_span_source_default_is_local(tmp_path):
    """Default (no --attn-stream-hf) builds the local-dir source (unchanged path)."""
    import argparse

    args = argparse.Namespace(attn_stream_hf=None, attn_stream_cache=1)
    src = fit._build_span_source(args, tmp_path / "store")
    assert type(src) is fit._SpanSource
    assert src.spans_dir == tmp_path / "store" / "answer_spans"


def test_attn_summary_store_raises_on_all_none_spans(tmp_path):
    """A context with NO non-None answer span fails loud (never a silent skip/zero)."""
    import torch

    spans_dir = tmp_path / "answer_spans"
    spans_dir.mkdir(parents=True)
    torch.save(
        {"context_id": "empty", "capture_layers": [0, 1], "spans": [None, None], "probes": ["p"]},
        spans_dir / "empty.pt",
    )
    local = fit._SpanSource(spans_dir)
    with pytest.raises(ValueError, match="no non-empty answer spans"):
        fit._attn_summary_store(local, ["empty"], [0, 1], _attn_w(8))


def test_param_hash_invalidates_stale_cells(tmp_path, monkeypatch):
    """A checkpoint cell written under one set of hyperparams is STALE under another.

    Guards the resume-into-reused-out-dir stale-serve fix: the per-cell param hash
    must change when a load-bearing constant (λ grid / MLP epochs / feat_dim /
    A35_MLP_TARGET_DIM / bootstrap) changes, and ``_load_cell`` must return None
    (recompute) on a hash mismatch while serving a matching cell.
    """
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 64)
    ph1 = fit._param_hash("a34a35", feat_dim=0)
    fit._save_cell(tmp_path, "a34a35", "meanprompt__L0", {"per_layer": {"x": 1}}, param_hash=ph1)
    # same params -> served
    assert fit._load_cell(tmp_path, "a34a35", "meanprompt__L0", param_hash=ph1) is not None
    # change A35_MLP_TARGET_DIM -> different hash -> stale -> None
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 16)
    ph2 = fit._param_hash("a34a35", feat_dim=0)
    assert ph2 != ph1
    assert fit._load_cell(tmp_path, "a34a35", "meanprompt__L0", param_hash=ph2) is None
    # a32 hash is independent of A35_MLP_TARGET_DIM (phase-scoped)
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 64)
    a = fit._param_hash("a32", feat_dim=0)
    monkeypatch.setattr(fit, "A35_MLP_TARGET_DIM", 16)
    assert fit._param_hash("a32", feat_dim=0) == a, "a32 hash must NOT depend on A35_MLP_TARGET_DIM"
