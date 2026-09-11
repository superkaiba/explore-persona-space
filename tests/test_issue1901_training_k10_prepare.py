"""Input identity and staging checks for the task1901 training-K bank."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1901_training_k10_prepare as P


def test_join_is_explicit_and_rejects_missing_or_duplicate_ids():
    """CI join must preserve wanted order and never accept partial coverage."""
    np.testing.assert_array_equal(P.ordered_indices(np.array([7, 2, 9]), np.array([9, 7])), [2, 0])
    with pytest.raises(AssertionError, match="missing requested"):
        P.ordered_indices(np.array([7, 2, 9]), np.array([8]))
    with pytest.raises(AssertionError, match="duplicate source"):
        P.ordered_indices(np.array([7, 7]), np.array([7]))
    with pytest.raises(AssertionError, match="duplicate requested"):
        P.ordered_indices(np.array([7]), np.array([7, 7]))


def test_chunk_boundaries_and_invalid_ids():
    """Shard transitions and the uneven selected last chunk obey original ID space."""
    cis = np.array([500, 999, 1000, 29999, 30000, 172468])
    assert P.source_chunk_names(cis) == [
        "shard00_chunk0001.pt",
        "shard00_chunk0002.pt",
        "shard00_chunk0059.pt",
        "shard01_chunk0000.pt",
        "shard05_chunk0044.pt",
    ]
    with pytest.raises(AssertionError):
        P.source_chunk_names(np.array([-1]))


def capture_blob():
    """Build a real-hidden-width, shuffled tiny chunk with separate context/answer states."""
    cis = [503, 500, 502, 501]
    prompts = [f"Public fixture context {ci}" for ci in cis]
    x = torch.arange(4 * 3 * P.HIDDEN, dtype=torch.float32).reshape(4, 3, P.HIDDEN)
    blob = {
        "ci": cis,
        "prompts": prompts,
        "cx_last": x,
        "v_x": x + 0.375,
        "layers": [14, 19, 26],
        "shard_index": 0,
        "chunk": 1,
    }
    expected = {
        ci: {"prompt_sha256": hashlib.sha256(p.encode()).hexdigest()}
        for ci, p in zip(cis, prompts, strict=True)
    }
    return blob, expected


def test_real_chunk_extract_reorders_and_preserves_original_precision(tmp_path):
    """The same torch file read path as production must retain fp32 answer differences."""
    blob, expected = capture_blob()
    path = tmp_path / "capture.pt"
    torch.save(blob, path)
    realized = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    ci, x, y, rows = P.extract_chunk(realized, np.array([502, 800, 503]), expected)
    np.testing.assert_array_equal(ci, [502, 503])
    np.testing.assert_array_equal(x, blob["cx_last"][[2, 0], 1].numpy())
    np.testing.assert_array_equal(y, blob["v_x"][[2, 0], 1].numpy())
    assert not np.array_equal(y, y.astype(np.float16).astype(np.float32))
    assert [r["ci"] for r in rows] == [502, 503]
    assert y.dtype == np.float32


def test_real_chunk_extract_rejects_prompt_drift_missing_key_and_nan():
    """Byte-linked prompt identity and tensor validity are load-bearing assertions."""
    blob, expected = capture_blob()
    expected[500]["prompt_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="prompt mismatch"):
        P.extract_chunk(blob, np.array([500]), expected)
    blob, expected = capture_blob()
    del blob["prompts"]
    with pytest.raises(AssertionError, match="missing capture keys"):
        P.extract_chunk(blob, np.array([500]), expected)
    blob, expected = capture_blob()
    blob["cx_last"][1, 1, 100] = float("nan")
    with pytest.raises(AssertionError):
        P.extract_chunk(blob, np.array([500]), expected)


def test_old_draw_assembly_handles_uneven_shards_and_rejects_duplicate_ci(tmp_path):
    """Explicit CI joins must work across uneven shards and reject cross-shard repeats."""
    paths = []
    groups = [[7, 5, 4], [1], [8, 2], [9, 3, 6, 0]]
    for shard, ids in enumerate(groups):
        values = np.broadcast_to(
            np.array(ids, dtype=np.float16)[:, None, None], (len(ids), 4, P.HIDDEN)
        ).copy()
        path = tmp_path / f"shard{shard}.npz"
        P.write_npz(
            path,
            V=values,
            ci=np.array(ids),
            draws=np.arange(43, 47),
            n_ans=np.ones((len(ids), 4), dtype=np.int32),
            src=np.array(["distr"] * len(ids)),
        )
        paths.append(path)
    values, shards = P.assemble_old_draws(paths, np.arange(10))
    np.testing.assert_array_equal(values[:, 0, 0], np.arange(10))
    assert values.dtype == np.float16 and shards[1] == 1
    with pytest.raises(AssertionError, match="duplicate IDs across"):
        P.assemble_old_draws(paths + paths[:1], np.arange(10))
    with pytest.raises(AssertionError, match="coverage mismatch"):
        P.assemble_old_draws(paths, np.arange(11))


def test_stage_local_copy_is_verified_against_actual_git_blob(tmp_path, monkeypatch):
    """Execute staging body with a signature-conformant Hub boundary, then corrupt bytes."""
    rel = "example/document.json"
    path = tmp_path / "source" / rel
    path.parent.mkdir(parents=True)
    payload = b'{"example":42}\n'
    path.write_bytes(payload)
    remote = SimpleNamespace(
        path=rel,
        size=len(payload),
        lfs=None,
        blob_id=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    )
    api = create_autospec(P.HfApi, instance=True)
    api.get_paths_info.return_value = [remote]
    monkeypatch.setattr(P, "HfApi", create_autospec(P.HfApi, return_value=api))
    stage = P.Stage(tmp_path, P.REVISION, None)
    assert stage.stage(rel) == path
    assert stage.sources[rel]["sha256"] == hashlib.sha256(payload).hexdigest()
    api.get_paths_info.assert_called_with(P.REPO, [rel], repo_type="dataset", revision=P.REVISION)
    path.write_bytes(payload.replace(b"42", b"43"))
    with pytest.raises(AssertionError, match="Git digest mismatch"):
        stage.stage(rel)


def test_stage_download_uses_pinned_revision_and_consumer_exact_path(tmp_path, monkeypatch):
    """Exercise real stage download branch and byte validation without a network fixture."""
    rel = "example/source.npz"
    payload = b"binary-source-fixture"
    remote = SimpleNamespace(
        path=rel,
        size=len(payload),
        lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()),
        blob_id="opaque-lfs-pointer",
    )
    api = create_autospec(P.HfApi, instance=True)
    api.get_paths_info.return_value = [remote]
    monkeypatch.setattr(P, "HfApi", create_autospec(P.HfApi, return_value=api))

    def write_download(repo_id, filename, *, repo_type, revision, local_dir):
        assert repo_id == P.REPO and repo_type == "dataset" and revision == P.REVISION
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)
        return str(destination)

    download = create_autospec(P.hf_hub_download, side_effect=write_download)
    monkeypatch.setattr(P, "hf_hub_download", download)
    stage = P.Stage(tmp_path, P.REVISION, None)
    assert stage.stage(rel) == tmp_path / "source" / rel
    assert stage.sources[rel]["sha256"] == remote.lfs.sha256
    download.assert_called_once()


def test_validate_refuses_stale_revision_before_loading_arrays(tmp_path):
    """An old recipe cannot reuse a manifest from a different source revision."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": P.SCHEMA, "revision": "0" * 40})
    )
    with pytest.raises(AssertionError):
        P.validate(tmp_path, P.REVISION)
