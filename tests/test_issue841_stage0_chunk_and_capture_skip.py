# ruff: noqa: E402
"""Stage-0 MLP group-chunking + capture-skip predicate (issue #841 crash-fix cycle 5).

Attempt 6 SIGKILL 137 (host RAM OOM) ~26 min into stage-0: at n=100k the MLP battery
built all 27 transitions' SplitMLPGroups at once (~77 GB of fp32 copies). The fix
chunks the transitions into groups of MLP_GROUP_CHUNK (pinned at 6) per fit_split_mlps
call (build → fit → free → gc) — chunking bounds peak host RAM. Since #926
(4dfcba056f) fit_batched_split_mlp seeds each group under
split_group_init_seed(seed, group.key), which depends only on (seed, key) — never on
batch position or chunking — so the fit is bit-identical across chunkings AND
deterministic + reproducible at a fixed chunking. group_chunk stays a pinned RAM knob
recorded in stage0_scaling.json (it no longer changes the numbers). These tests pin:

  * DETERMINISM: the same group_chunk twice → identical r2 curve + params;
  * PARTITION INVARIANCE (#926, 4dfcba056f): different chunk sizes give a BIT-IDENTICAL
    curve + params — so a regression to batch-order-dependent seeding is caught;
  * the RAM-sizing estimator counts the helper's re-copies (3x train + eval term), not
    just the caller delta;
  * capture_complete_on_hf's fetch short-circuit predicate: complete / missing-shard /
    dtype-mismatch (so a relaunch fetches the already-uploaded capture instead of a
    ~45-min re-capture, but never fetches an incomplete/wrong-dtype set).

They exercise the LIVE dispatched `issue841_scaling_stage0.mlp_scaling` /
`issue841_scaling_common.capture_complete_on_hf`, per the "verification gates test the
live dispatched path" rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue841_scaling_common as S
import issue841_scaling_stage0 as st0
import numpy as np
import pytest


def _synthetic_pool(seed=0, n=40, n_layers=6, hidden=16, n_eval=10):
    rng = np.random.default_rng(seed)
    fit_pool = rng.standard_normal((n, n_layers, hidden)).astype(np.float32)
    val = rng.standard_normal((n_eval, n_layers, hidden)).astype(np.float32)
    test = rng.standard_normal((n_eval, n_layers, hidden)).astype(np.float32)
    return fit_pool, val, test


def test_mlp_chunking_is_deterministic_at_fixed_chunk():
    """The load-bearing invariant: at a FIXED group_chunk the r2 curve + params are
    bit-reproducible across runs (so a pinned group_chunk gives a reproducible fit)."""
    fit_pool, val, test = _synthetic_pool()
    transitions = [0, 1, 2, 3, 4]
    kw = dict(device="cpu", chunk_size=8, num_threads=1, max_epochs=3)
    c1, p1 = st0.mlp_scaling(fit_pool, val, test, transitions, [40], group_chunk=2, **kw)
    c2, p2 = st0.mlp_scaling(fit_pool, val, test, transitions, [40], group_chunk=2, **kw)
    for t in transitions:
        assert c1[f"transition_{t}"]["40"] == c2[f"transition_{t}"]["40"], t
        for k in p1[40][t]:
            np.testing.assert_array_equal(np.asarray(p1[40][t][k]), np.asarray(p2[40][t][k]))


def test_mlp_chunking_is_partition_invariant_across_chunk_sizes():
    """PARTITION INVARIANCE: the r2 curve + params are BIT-IDENTICAL across group_chunk
    values. Since #926 (4dfcba056f, "port split-MLP fitter to main with
    partition-invariant per-group seeding") fit_batched_split_mlp seeds each group under
    torch.manual_seed(split_group_init_seed(seed, group.key)) — an unsalted blake2b of
    (seed, repr(key)) that depends on NEITHER batch position NOR chunking — so any
    partition of the group list reproduces every member's init bit-exactly.

    This test SUPERSEDES the pre-#926 pin test_mlp_chunk_size_changes_fit_batch_order_init,
    which asserted the OPPOSITE (member g got draw g, so re-chunking changed every fit).
    That pin's own docstring required any equivalence flip to be "a deliberate, reviewed
    change to the shared #658-gated helper" — 4dfcba056f is exactly that, and shipped its
    own exactness gate (assert_split_mlp_partition_invariant + tests/
    test_vectorized_split_mlp.py). A revert to batch-order seeding fails this test.

    group_chunk remains a pinned RAM knob (recorded in stage0_scaling.json); it bounds
    peak host RAM without changing the numbers.
    """
    fit_pool, val, test = _synthetic_pool()
    transitions = [0, 1, 2, 3, 4]
    kw = dict(device="cpu", chunk_size=8, num_threads=1, max_epochs=3)
    # 5 = one all-groups call; 2 = a MISALIGNED split leaving a remainder group;
    # 1 = every group in its own call. All three must agree bit-for-bit.
    c5, p5 = st0.mlp_scaling(fit_pool, val, test, transitions, [40], group_chunk=5, **kw)
    for group_chunk in (2, 1):
        c_n, p_n = st0.mlp_scaling(
            fit_pool, val, test, transitions, [40], group_chunk=group_chunk, **kw
        )
        for t in transitions:
            assert c5[f"transition_{t}"]["40"] == c_n[f"transition_{t}"]["40"], (
                group_chunk,
                t,
            )
            for k in p5[40][t]:
                np.testing.assert_array_equal(np.asarray(p5[40][t][k]), np.asarray(p_n[40][t][k]))


def _mock_hf(monkeypatch, *, manifest, shard_repo_files, overflow_repo):
    """Mock the huggingface_hub surface capture_complete_on_hf uses."""
    import huggingface_hub

    public = [f"{S.HF_CAPTURE_BUCKET}/manifest.json"]
    if overflow_repo:
        public.append(f"{S.HF_CAPTURE_BUCKET}/OVERFLOW_POINTER.json")

    def fake_list(repo, repo_type="dataset"):
        if repo == S.C.HF_DATA_REPO:
            return list(public)
        return list(shard_repo_files)  # the shard repo (overflow or public)

    def fake_download(repo, filename, repo_type="dataset"):
        p = Path(S.tempfile.gettempdir()) / f"i841_test_manifest_{id(manifest)}.json"
        p.write_text(json.dumps(manifest))
        return str(p)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(S, "_overflow_repo_for_bucket", lambda *a, **kw: overflow_repo)


def _manifest(dtype="bf16", total=96000, shards=("cx_last_shard000.pt", "cx_last_shard001.pt")):
    return {
        "realized_capture_dtype": dtype,
        "total_rows": total,
        "spans": [{"shard": s, "row_lo": 0, "row_hi": total} for s in shards],
    }


def test_capture_complete_when_all_shards_present(monkeypatch):
    shards = ("cx_last_shard000.pt", "cx_last_shard001.pt")
    _mock_hf(
        monkeypatch,
        manifest=_manifest(shards=shards),
        shard_repo_files=[f"{S.HF_CAPTURE_BUCKET}/{s}" for s in shards],
        overflow_repo=S.OVERFLOW_REPO,
    )
    ok, detail = S.capture_complete_on_hf(96000, "bf16")
    assert ok is True, detail
    assert "2 shards" in detail


def test_capture_incomplete_when_shard_missing(monkeypatch):
    _mock_hf(
        monkeypatch,
        manifest=_manifest(shards=("cx_last_shard000.pt", "cx_last_shard001.pt")),
        shard_repo_files=[f"{S.HF_CAPTURE_BUCKET}/cx_last_shard000.pt"],  # shard001 missing
        overflow_repo=S.OVERFLOW_REPO,
    )
    ok, detail = S.capture_complete_on_hf(96000, "bf16")
    assert ok is False and "missing" in detail, detail


def test_capture_incomplete_on_dtype_mismatch(monkeypatch):
    shards = ("cx_last_shard000.pt",)
    _mock_hf(
        monkeypatch,
        manifest=_manifest(dtype="fp32", shards=shards),
        shard_repo_files=[f"{S.HF_CAPTURE_BUCKET}/{s}" for s in shards],
        overflow_repo=S.OVERFLOW_REPO,
    )
    ok, detail = S.capture_complete_on_hf(96000, "bf16")
    assert ok is False and "dtype" in detail, detail


def test_ram_sizing_counts_helper_recopies():
    """The RAM projection must count the helper's full fp32 re-copies — 3x train-sized
    stacks (caller Y delta + helper X-stack + helper Y-stack), not just the caller delta
    — else it undercounts the live peak and the 80%-MemAvailable soft-warn is a false
    all-clear (Codex #841 v15 review)."""
    fit_pool, _val, test = _synthetic_pool(n=40, n_layers=6, hidden=16, n_eval=10)
    group_chunk, n_max, hidden = 3, 40, 16
    proj = st0._log_mlp_ram_sizing(fit_pool, [n_max], [0, 1, 2, 3, 4], group_chunk, test.shape[0])
    pool = fit_pool.nbytes / (1024**3)
    train_term = group_chunk * n_max * hidden * 4 / (1024**3)
    assert proj >= pool + 3 * train_term, (proj, pool, train_term)  # 3 train terms + eval


def test_mlp_group_chunk_zero_raises():
    """EPM_I841S_MLP_GROUP_CHUNK<=0 must fail loud, not step-0-crash the range loop."""
    fit_pool, val, test = _synthetic_pool()
    with pytest.raises(ValueError, match="group_chunk"):
        st0.mlp_scaling(
            fit_pool,
            val,
            test,
            [0, 1],
            [40],
            device="cpu",
            chunk_size=8,
            num_threads=1,
            max_epochs=1,
            group_chunk=0,
        )
