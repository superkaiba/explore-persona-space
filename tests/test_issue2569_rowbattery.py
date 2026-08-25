"""Unit tests for scripts/issue2569_rowbattery.py — P-B first half (unit 4a).

Covers the pure moments/refit/schema helpers plus the SAE training core and the
ported #2476 fixes this unit depends on:

- streamed fp64 moment accumulators == direct numpy Grams/means (tiny shapes);
- pooled moments == additive combination of the two disjoint halves;
- conversation-key construction (ci rows grouped, pass_b rows unique) + L6
  split-half disjointness over those keys;
- sigma producer files round-trip through ``issue2569_gateladder.load_sigma_file``
  to the exact centered covariance (the unit-2 consumer contract);
- split-half ridge refit reproduces the ``fit_ridge_primal`` estimator computed
  directly from raw rows (standardize-X unbiased sd + 1e-9 / center-Y, sum-form
  Gram + absolute lambda);
- ported ``T24._stream_fit_sum`` clamps the final chunk at n_fit (crash-fix
  8360a1d72d regression pin);
- ported ``N1M._stream_ckpt_fingerprint`` revision seam: None reproduces the
  legacy hash byte-for-byte, a pin flips it;
- ported ``--sae-k`` seam: ``_sae_k``/``_sae_leaf`` resolution;
- ``_run_sae_training`` executes the real matryoshka loop on a tiny memmap
  (fp16, width 16), writes a per-epoch checkpoint, and resumes to completion
  without re-training (epoch_done == SAE_EPOCHS short-circuits the loop);
- ``load_sae_ctx`` round-trips the ae.pt bundle (threshold buffer included).

All synthetic + CPU-fast (d <= 12); the dense 3584-dim fp64 factorizations stay
out of every test path (unit brief).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue2476_turnavg_sae as T24  # noqa: E402
import issue2569_gateladder as GL  # noqa: E402
import issue2569_leg6 as L6  # noqa: E402
import issue2569_rowbattery as RB  # noqa: E402


def _toy_xy(n: int = 40, d: int = 6, dy: int = 5, seed: int = 0):
    """Tiny fp16 X/Y row stores (memmap stand-ins: plain ndarrays index the same)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float16)
    y = rng.normal(size=(n, dy)).astype(np.float16)
    return x, y


# ── moment accumulators ──────────────────────────────────────────────────────────


def test_accumulate_moments_matches_numpy():
    """Streamed chunked fp64 sums/Grams equal the direct numpy reductions."""
    x, y = _toy_xy()
    pos = np.arange(x.shape[0], dtype=np.int64)
    acc = RB._accumulate_moments(x, y, pos, chunk=7, dev=torch.device("cpu"), tag="t")
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    np.testing.assert_allclose(acc["sum_x"].numpy(), x64.sum(0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["sum_y"].numpy(), y64.sum(0), rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["gram_xx"].numpy(), x64.T @ x64, rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["gram_xy"].numpy(), x64.T @ y64, rtol=0, atol=1e-10)
    np.testing.assert_allclose(acc["gram_yy"].numpy(), y64.T @ y64, rtol=0, atol=1e-10)
    assert acc["n"] == x.shape[0]


def test_combine_moments_is_additive_over_disjoint_halves():
    """Pooled moments over the full pool == sum of the two disjoint halves."""
    x, y = _toy_xy(n=30)
    pos = np.arange(30, dtype=np.int64)
    h1, h2 = pos[:13], pos[13:]
    a1 = RB._accumulate_moments(x, y, h1, chunk=5, dev=torch.device("cpu"), tag="h1")
    a2 = RB._accumulate_moments(x, y, h2, chunk=5, dev=torch.device("cpu"), tag="h2")
    pooled = RB._combine_moments(a1, a2)
    full = RB._accumulate_moments(x, y, pos, chunk=8, dev=torch.device("cpu"), tag="full")
    assert pooled["n"] == full["n"] == 30
    for k in ("sum_x", "sum_y", "gram_xx", "gram_xy", "gram_yy"):
        torch.testing.assert_close(pooled[k], full[k], rtol=0, atol=1e-9)


# ── conversation keys + split halves ─────────────────────────────────────────────


def test_conversation_keys_group_ci_and_uniquify_pass_b():
    """ci>=0 rows share their conversation key; ci==-1 (pass_b) rows are unique."""
    # global row space of 10; pass_b rows 0..3 (ci=-1), new rows 4..9 with dup cis
    row_ci = np.array([-1, -1, -1, -1, 7, 7, 8, 9, 9, 9], dtype=np.int64)
    pool_ids = np.array([0, 2, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # rows 1,3 excluded
    keys = RB._conversation_keys(row_ci, pool_ids)
    assert keys == ["pb0", "pb2", "ci7", "ci7", "ci8", "ci9", "ci9", "ci9"]
    i1, i2 = L6.split_halves_by_conversation(keys, seed=L6.SPLIT_SEED)
    assert len(np.intersect1d(i1, i2)) == 0
    assert len(i1) + len(i2) == len(keys)
    # rows sharing a conversation key never straddle halves
    for half in (i1, i2):
        half_keys = {keys[int(i)] for i in half}
        other = {keys[int(i)] for i in (i2 if half is i1 else i1)}
        assert not (half_keys & other)


# ── sigma producer contract (unit-2 consumer round-trip) ─────────────────────────


def test_sigma_pt_roundtrips_through_gateladder_loader(tmp_path):
    """gram_xx.pt/gram_yy.pt load via GL.load_sigma_file to the centered covariance."""
    x, _ = _toy_xy(n=25, d=6)
    x64 = x.astype(np.float64)
    gram = torch.as_tensor(x64.T @ x64)
    mean = torch.as_tensor(x64.mean(0))
    p = tmp_path / "gram_xx.pt"
    RB._write_sigma_pt(p, gram, mean, x.shape[0], side="context (X19)", pool="test")
    sigma = GL.load_sigma_file(p)
    expected = x64.T @ x64 / x.shape[0] - np.outer(x64.mean(0), x64.mean(0))
    expected = 0.5 * (expected + expected.T)
    np.testing.assert_allclose(sigma, expected, rtol=0, atol=1e-12)


def test_write_sigma_pt_refuses_sigma_meta_key(tmp_path):
    """A meta key named 'sigma' would shadow the gram triple in the loader."""
    gram = torch.eye(3, dtype=torch.float64)
    mean = torch.zeros(3, dtype=torch.float64)
    try:
        RB._write_sigma_pt(tmp_path / "g.pt", gram, mean, 4, sigma="nope")
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on a 'sigma' meta key")


# ── split-half ridge refit (fit_ridge_primal parity) ─────────────────────────────


def test_half_ridge_refit_matches_primal_reference():
    """Gram-space refit == direct standardize-X/center-Y ridge on the raw rows."""
    rng = np.random.default_rng(3)
    n, d, dy = 60, 5, 4
    x = rng.normal(size=(n, d)).astype(np.float16)
    y = rng.normal(size=(n, dy)).astype(np.float16)
    acc = RB._accumulate_moments(
        x, y, np.arange(n, dtype=np.int64), chunk=16, dev=torch.device("cpu"), tag="h"
    )
    lam = 3.7
    refit = RB._half_ridge_refit(acc, lam)
    # reference: the _ridge_primal_multi_lambda convention computed from raw rows
    xt = torch.as_tensor(x.astype(np.float64))
    yt = torch.as_tensor(y.astype(np.float64))
    xmu, xsd = xt.mean(0), xt.std(0) + 1e-9  # torch.std default = unbiased
    xn = (xt - xmu) / xsd
    yc = yt - yt.mean(0)
    w_ref = torch.linalg.solve(xn.T @ xn + lam * torch.eye(d, dtype=torch.float64), xn.T @ yc)
    torch.testing.assert_close(refit["xmu"], xmu, rtol=0, atol=1e-9)
    torch.testing.assert_close(refit["xsd"], xsd, rtol=0, atol=1e-9)
    torch.testing.assert_close(refit["W"], w_ref, rtol=1e-9, atol=1e-9)
    assert refit["selected_lambda"] == lam and refit["n_rows"] == n


# ── ported #2476 / N1M fixes (regression pins) ───────────────────────────────────


def test_stream_fit_sum_clamps_final_chunk_at_n_fit():
    """Crash-fix 8360a1d72d pin: no holdout-row spill into the fit-side sum."""
    yc = np.ones((16, 3), dtype=np.float64)
    yc[10:] = 1e6  # holdout rows past n_fit — MUST NOT enter the sum
    s = T24._stream_fit_sum(yc, n_fit=10, chunk=8)
    np.testing.assert_allclose(s, np.full(3, 10.0), rtol=0, atol=0)


def test_stream_ckpt_fingerprint_revision_seam():
    """revision=None reproduces the legacy hash; a pin flips it (resume refusal)."""
    names = ["a.pt", "b.pt"]
    legacy = N1M._stream_ckpt_fingerprint(19, "prefix", names)
    assert N1M._stream_ckpt_fingerprint(19, "prefix", names, revision=None) == legacy
    pinned = N1M._stream_ckpt_fingerprint(19, "prefix", names, revision="89cfa76")
    assert pinned != legacy
    assert N1M._stream_ckpt_fingerprint(19, "prefix", names, revision="89cfa76") == pinned


def test_sae_k_and_leaf_resolution():
    """Ported --sae-k seam: default resolves to k=100 / sae_c; 200 -> sae_c_k200."""
    import argparse

    ns = argparse.Namespace(sae_k=0)
    assert T24._sae_k(ns) == 100 and T24._sae_leaf(ns) == "sae_c"
    ns200 = argparse.Namespace(sae_k=200)
    assert T24._sae_k(ns200) == 200 and T24._sae_leaf(ns200) == "sae_c_k200"


def test_rowbattery_t24_namespace_has_default_sae_k():
    """The composed T24 namespace keeps k=100 (the k=200 twin is out of scope)."""
    import argparse

    args = argparse.Namespace(
        device="cpu",
        out_root=Path("/tmp/i2569-ns-probe"),
        hf_prefix="issue2569_theory/analysis_tensors",
        max_chunks=2,
        smoke_rows=0,
        sae_dict=16,
        sae_steps=3,
        smoke=True,
        fresh_stream=False,
        skip_upload=True,
        resume_across_code_sha=False,
    )
    t24 = RB._t24_args(args)
    assert T24._sae_k(t24) == 100
    assert t24.smoke and t24.max_chunks == 2 and t24.skip_upload
    assert not T24._production(t24)


# ── SAE training core (real loop, tiny width, fp16 memmap) ───────────────────────


def test_run_sae_training_tiny_and_resume(tmp_path):
    """One real matryoshka train on a tiny fp16 memmap: epoch rows + fired-union +
    checkpoint written; a resume from the completed checkpoint short-circuits."""
    rng = np.random.default_rng(0)
    n, d = 96, 8
    mm_path = tmp_path / "X19.fp16.npy"
    np.save(mm_path, rng.normal(size=(n, d)).astype(np.float16))
    x_mm = np.load(mm_path, mmap_mode="r")
    tr_pos = np.arange(0, 80, dtype=np.int64)
    val_pos = np.arange(80, 96, dtype=np.int64)
    ckpt = tmp_path / "ckpt_last.pt"
    model, rows, fired_union, step = RB._run_sae_training(
        x_mm,
        tr_pos,
        val_pos,
        width=16,
        dev="cpu",
        steps_cap=0,
        ckpt_path=ckpt,
        resume_ok=False,
    )
    assert len(rows) == T24.SAE_EPOCHS and step > 0
    assert fired_union.shape == (16,) and fired_union.dtype == bool and fired_union.any()
    assert ckpt.exists()
    assert np.isfinite(rows[-1]["val_var_fve"])
    # resume from the completed checkpoint: no further epochs run (step unchanged)
    model2, rows2, fired2, step2 = RB._run_sae_training(
        x_mm,
        tr_pos,
        val_pos,
        width=16,
        dev="cpu",
        steps_cap=0,
        ckpt_path=ckpt,
        resume_ok=True,
    )
    assert step2 == step and len(rows2) == len(rows)
    np.testing.assert_array_equal(fired2, fired_union)
    # resumed weights equal the checkpointed weights
    for k, v in model.state_dict().items():
        torch.testing.assert_close(model2.state_dict()[k], v, rtol=0, atol=0)


def test_load_sae_ctx_roundtrip(tmp_path):
    """ae.pt bundle round-trips through load_sae_ctx (threshold buffer included)."""
    model = T24.MatryoshkaBatchTopKSAE(
        act_dim=8, dict_size=16, k=4, tier_bounds=T24._sae_tier_bounds(16), seed=7
    )
    with torch.no_grad():
        model.threshold.fill_(0.123)
    p = tmp_path / "ae.pt"
    RB._atomic_torch_save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "cfg": model.cfg_dict(),
        },
        p,
    )
    loaded = RB.load_sae_ctx(p, device="cpu")
    assert float(loaded.threshold) == float(model.threshold)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)
    x = torch.randn(5, 8)
    torch.testing.assert_close(loaded.encode(x), model.encode(x), rtol=0, atol=0)
