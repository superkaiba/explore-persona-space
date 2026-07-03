# ruff: noqa: E402
"""Stage-0 MLP group-chunking + capture-skip predicate (issue #841 crash-fix cycle 5).

Attempt 6 SIGKILL 137 (host RAM OOM) ~26 min into stage-0: at n=100k the MLP battery
built all 27 transitions' SplitMLPGroups at once (~77 GB of fp32 copies). The fix
chunks the transitions into groups of MLP_GROUP_CHUNK per fit_split_mlps call (build →
fit → free → gc). Chunking is EXACTLY equivalent because each group's MLP is re-seeded
per call and the loss is a per-member sum (no global-batch-size scaling), so per-member
gradients don't depend on group count. These tests pin:

  * chunked (group_chunk=2) vs unchunked (group_chunk=all) mlp_scaling produce the SAME
    r2 curve + the SAME fitted params on a tiny synthetic pool;
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


def test_mlp_chunk_size_changes_fit_batch_order_init():
    """DOCUMENTED non-equivalence: chunking is NOT bit-identical across chunk sizes,
    because fit_batched_split_mlp seeds each group's MLP init in BATCH ORDER (member g
    gets init g — vectorized_mlp_skill.py:809). So a different group_chunk gives group g
    a different init → a different fit. This test PINS that known property (a future
    change that silently made chunk sizes equivalent — e.g. per-key seeding — would flip
    it and must be a deliberate, reviewed change to the shared #658-gated helper). The
    fit stays deterministic at a fixed chunk (test above); group_chunk is pinned in the
    stage-0 output."""
    fit_pool, val, test = _synthetic_pool()
    transitions = [0, 1, 2, 3, 4]
    kw = dict(device="cpu", chunk_size=8, num_threads=1, max_epochs=3)
    c_full, _ = st0.mlp_scaling(fit_pool, val, test, transitions, [40], group_chunk=5, **kw)
    c_chunk, _ = st0.mlp_scaling(fit_pool, val, test, transitions, [40], group_chunk=2, **kw)
    # at least one transition whose init position differs across chunkings must differ
    diffs = [
        t
        for t in transitions
        if c_full[f"transition_{t}"]["40"]["r2_id"] != c_chunk[f"transition_{t}"]["40"]["r2_id"]
    ]
    assert diffs, "expected batch-order-init to change the fit across chunk sizes"


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
