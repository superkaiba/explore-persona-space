"""Reduced loader preserves the historical capture ordering and mean operation."""

import json

import numpy as np
import pytest

from scripts.issue1739_fits import _load_labeled
from scripts.issue2669_fair_replay import load_context_answer, run


def test_reduced_loader_matches_original_and_drops_missing_dv(tmp_path):
    store = tmp_path / "capture"
    store.mkdir()
    # c2 precedes c1; preserving capture order matters for fold and score joins.
    order = ["c2", "c1", "c2", "c1", "dropped", "dropped"]
    metadata = [{"context_id": cid, "rollout_k": i // 2} for i, cid in enumerate(order)]
    (store / "row_index_shard00.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metadata))
    values = np.asarray(
        [[0.1, 0.3, 0.7], [1.1, 2.3, 3.7], [5.1, 4.3, 3.7], [7.1, 8.3, 9.7], [0, 0, 0], [1, 1, 1]],
        dtype=np.float16,
    )
    for kind in ["prefix_end", "context_end", "t1"]:
        np.save(store / f"{kind}_L18_shard00.npy", values)
    dv = tmp_path / "dv.json"
    dv.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "context_id": cid,
                        "dv": score,
                        "split": "train",
                        "rung": "train",
                        "group_key": cid,
                    }
                    for cid, score in [("c1", 10), ("c2", 20), ("dropped", None)]
                ]
            }
        )
    )
    reduced = load_context_answer(store, dv, 18, "train")
    original = _load_labeled(store, dv, [18], config="config_a", need_rollout_rows=False)
    assert reduced.ctx_order == original.ctx_order == ["c2", "c1"]
    assert reduced.groups == original.groups
    np.testing.assert_array_equal(reduced.dv, original.dv)
    np.testing.assert_array_equal(reduced.z_ans, original.z_ans)
    np.testing.assert_array_equal(
        reduced.z_by_variant["context_end"], original.z_by_variant["context_end"]
    )


def test_unapproved_fit_never_loads_inputs(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"fit_authorized": false}')
    with pytest.raises(ValueError, match="not authorized"):
        run(config)
