import hashlib

import numpy as np
import pytest

from scripts.issue1739_large_pool_risk import canonical_pool, ranked_top, raw_path, verify_blob


def test_response_available_pool_deduplicates_before_ranking():
    rows = np.array([3, 1, 2, 0])
    ci = np.array([-1, 2, 3, 4])
    hashes = np.array(["same", "same", "distinct", "old"])
    pool = canonical_pool(rows, ci, hashes)
    assert pool.tolist() == [1, 2]
    # A duplicate's high score must not select a different canonical response.
    assert ranked_top(np.array([100, 0, 0, 200]), pool, rows, 2).tolist() == [1, 2]


def test_raw_source_index_boundaries():
    assert raw_path(29999).endswith("shard00_chunk0059.json")
    assert raw_path(30000).endswith("shard01_chunk0000.json")
    assert raw_path(959999).endswith("shard31_chunk0059.json")
    with pytest.raises(ValueError):
        raw_path(-1)


def test_blob_identity_rejects_same_length_corruption(tmp_path):
    p = tmp_path / "source.json"
    p.write_bytes(b"valid")
    record = dict(path=str(p), size=5, blob_id=hashlib.sha1(b"blob 5\0valid").hexdigest())
    assert verify_blob(p, record) == b"valid"
    p.write_bytes(b"wrong")
    with pytest.raises(ValueError):
        verify_blob(p, record)
