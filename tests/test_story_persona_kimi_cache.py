import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import story_persona_kimi_cache as cache


@pytest.fixture
def pinned_cache(tmp_path, monkeypatch):
    root = tmp_path / "models--moonshotai--Kimi-K2.6"
    snapshot = root / "snapshots" / cache.REVISION
    snapshot.mkdir(parents=True)
    (root / "blobs").mkdir()
    shards = []
    for i in range(64):
        data = bytes([i]) * 131072
        digest = hashlib.sha256(data).hexdigest()
        target = root / "blobs" / digest
        target.write_bytes(data)
        name = f"model-{i:05d}.safetensors"
        (snapshot / name).symlink_to(target)
        shards.append(
            SimpleNamespace(rfilename=name, size=len(data), lfs=SimpleNamespace(sha256=digest))
        )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {str(i): s.rfilename for i, s in enumerate(shards)}})
    )
    monkeypatch.setattr(cache, "WEIGHT_BYTES", 64 * 131072)
    return tmp_path, snapshot, shards


def test_missing_cache_never_gets_credit(tmp_path):
    assert cache.allocated_credit(tmp_path, []) == {
        "allocated_credit_bytes": 0,
        "readable_shards": 0,
    }


def test_complete_resident_cache_gets_only_allocated_credit(pinned_cache):
    root, snapshot, shards = pinned_cache
    result = cache.allocated_credit(root, shards)
    actual = sum(
        min(p.stat().st_blocks * 512, p.stat().st_size) for p in snapshot.glob("*.safetensors")
    )
    assert result == {"allocated_credit_bytes": actual, "readable_shards": 64}
    assert actual <= cache.WEIGHT_BYTES


@pytest.mark.parametrize("defect", ["missing", "truncated", "wrong_blob", "wrong_index"])
def test_incompatible_cache_fails_loud(pinned_cache, defect):
    root, snapshot, shards = pinned_cache
    path = snapshot / shards[0].rfilename
    if defect == "missing":
        path.unlink()
    elif defect == "truncated":
        path.resolve().write_bytes(b"bad")
    elif defect == "wrong_blob":
        shards[0].lfs.sha256 = "wrong"
    else:
        (snapshot / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises((RuntimeError, FileNotFoundError)):
        cache.allocated_credit(root, shards)
