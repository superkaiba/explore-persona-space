"""Realized-#1092-layout adapter tests for issue #1739 store_io (the
``u-store-staging-layout-unwired`` fix).

Synthetic fixtures mirror BOTH realized dir shapes (gate0 + round-C2 smoke
findings): ``dynamics_*`` dirs (realized kind names ``context_k`` /
``answer_k_t1`` + PER-KIND ``row_index_{kind}`` stems) and main ``cell_*``
dirs (canonical kind names, NO row_index — the corpus ``manifest.jsonl`` is
the row-metadata source). Network is faked ONLY at the hub boundary with
signature-conformant autospecs; ``stage_u_store`` / ``u_store_loadable`` /
``load_summaries`` bodies execute for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import store_io
from explore_persona_space.experiments.issue_1739.constants import (
    STORE_PREFIX,
    STORE_REVISION,
)

DIM = 4
KINDS = ("prefix_end", "context_end", "t1")


def _rows_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _write_dynamics_fixture(root: Path, n: int = 6) -> dict[str, np.ndarray]:
    """Realized dynamics_* shape: realized kind names, per-kind row_index
    stems, one kind sharded + one unsharded."""
    root.mkdir(parents=True, exist_ok=True)
    ctx = np.arange(n * DIM, dtype=np.float16).reshape(n, DIM)
    ans = ctx + 100.0
    half = n // 2
    np.save(root / "context_k_L00_shard00.npy", ctx[:half])
    np.save(root / "context_k_L00_shard01.npy", ctx[half:])
    np.save(root / "answer_k_t1_L00.npy", ans)
    idx = [{"conv_id": f"c{i}", "kind": "context_k", "turn_index": i} for i in range(n)]
    _rows_jsonl(root / "row_index_context_k_shard00.jsonl", idx[:half])
    _rows_jsonl(root / "row_index_context_k_shard01.jsonl", idx[half:])
    _rows_jsonl(
        root / "row_index_answer_k_t1.jsonl",
        [{**r, "kind": "answer_k_t1"} for r in idx],
    )
    return {"context_k": ctx, "answer_k_t1": ans}


def _write_cell_fixture(root: Path, n: int = 5, n_eval: int = 2) -> None:
    """Realized cell_* shape (post-staging): canonical kind names, NO
    row_index files, corpus manifest.jsonl beside the shards."""
    root.mkdir(parents=True, exist_ok=True)
    for j, kind in enumerate(KINDS):
        np.save(root / f"{kind}_L00.npy", np.full((n, DIM), float(j), dtype=np.float16))
    rows = [{"conv_id": f"c{i}", "is_eval_only": i >= n - n_eval} for i in range(n)]
    _rows_jsonl(root / "manifest.jsonl", rows)


def test_realized_kind_map_pins_c2_findings():
    assert store_io.REALIZED_KIND_FOR == {
        "context_end": "context_k",
        "t1": "answer_k_t1",
        "bare_query": "c_q_bare",
    }


def test_load_summaries_maps_dynamics_layout(tmp_path):
    expected = _write_dynamics_fixture(tmp_path / "dynamics_instruct")
    arrays, meta = store_io.load_summaries(
        tmp_path, ("context_end", "t1"), (0,), cell="dynamics_instruct", hidden_dim=DIM
    )
    # Keyed by the REQUESTED canonical names; values from the realized files.
    np.testing.assert_array_equal(arrays[("context_end", 0)], expected["context_k"])
    np.testing.assert_array_equal(arrays[("t1", 0)], expected["answer_k_t1"])
    assert len(meta) == 6 and meta[0]["conv_id"] == "c0"


def test_load_summaries_realized_names_direct(tmp_path):
    _write_dynamics_fixture(tmp_path / "dynamics_instruct")
    arrays, meta = store_io.load_summaries(
        tmp_path, ("context_k",), (0,), cell="dynamics_instruct", hidden_dim=DIM, n_rows=4
    )
    assert arrays[("context_k", 0)].shape == (4, DIM) and len(meta) == 4


def test_load_summaries_cell_layout_manifest_meta(tmp_path):
    root = tmp_path / "u_store"
    _write_cell_fixture(root, n=5, n_eval=2)
    arrays, meta = store_io.load_summaries(root, KINDS, (0,), hidden_dim=DIM)
    assert {k for k, _ in arrays} == set(KINDS)
    assert len(meta) == 5
    mask = store_io.fit_pool_mask(meta)
    assert mask.sum() == 3 and not mask[-2:].any()  # is_eval_only tail excluded


def test_canonical_kind_takes_precedence_over_alias(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    np.save(root / "context_end_L00.npy", np.zeros((3, DIM), dtype=np.float16))
    np.save(root / "context_k_L00.npy", np.ones((3, DIM), dtype=np.float16))
    _rows_jsonl(root / "row_index.jsonl", [{"i": i} for i in range(3)])
    arrays, _ = store_io.load_summaries(root, ("context_end",), (0,), hidden_dim=DIM)
    assert float(arrays[("context_end", 0)].sum()) == 0.0


def test_load_summaries_missing_meta_names_ladder(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    np.save(root / "context_end_L00.npy", np.zeros((2, DIM), dtype=np.float16))
    with pytest.raises(FileNotFoundError, match=r"manifest\.jsonl"):
        store_io.load_summaries(root, ("context_end",), (0,), hidden_dim=DIM)


def test_per_kind_row_index_count_mismatch_raises(tmp_path):
    root = tmp_path / "dynamics_instruct"
    _write_dynamics_fixture(root)
    # Truncate one kind's index to force the count-consistency raise.
    _rows_jsonl(root / "row_index_answer_k_t1.jsonl", [{"conv_id": "c0"}])
    with pytest.raises(ValueError, match="counts disagree"):
        store_io.load_summaries(
            root.parent, ("context_end", "t1"), (0,), cell="dynamics_instruct", hidden_dim=DIM
        )


def test_u_store_target_flattens_cell_dir():
    repo_path = STORE_PREFIX.rstrip("/") + "/cell_inst_own/t1_L00_shard03.npy"
    assert store_io._u_store_target(Path("/dest"), repo_path) == Path("/dest/t1_L00_shard03.npy")


def test_stage_u_store_short_circuits_on_local_capture_store(tmp_path, monkeypatch):
    """A loadable LOCAL capture store at dest (the tiny-real smoke stand-in)
    is left untouched — no network call."""
    dest = tmp_path / "u_store"
    dest.mkdir()
    for kind in KINDS:
        np.save(dest / f"{kind}_L00.npy", np.zeros((3, DIM), dtype=np.float16))
    _rows_jsonl(dest / "row_index.jsonl", [{"context_id": f"c{i}"} for i in range(3)])
    lister = mock.create_autospec(store_io.hub.list_hf_files_under_path)
    monkeypatch.setattr(store_io.hub, "list_hf_files_under_path", lister)
    out = store_io.stage_u_store(dest, KINDS, (0,))
    assert out == dest
    lister.assert_not_called()


def _fake_hub(monkeypatch, manifest_rows: list[dict], shard_rows: int = 2):
    """Signature-conformant hub fakes: the lister enumerates 2 shards per
    (kind, L00); stage_hub_file WRITES the target (npy shard / manifest)."""
    prefix = STORE_PREFIX.rstrip("/") + "/cell_inst_own"
    repo_files = [f"{prefix}/{kind}_L00_shard{s:02d}.npy" for kind in KINDS for s in range(2)]

    def _list(api, repo_id, pfx, repo_type="dataset", revision=None):
        assert pfx == prefix, pfx
        return list(repo_files)

    def _stage(repo_id, path_in_repo, target, *, repo_type="dataset", revision=None, **kw):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.name == "manifest.jsonl":
            _rows_jsonl(target, manifest_rows)
        else:
            base = float(sum(target.name.encode()) % 7)
            np.save(target, np.full((shard_rows, DIM), base, dtype=np.float16))
            # np.save appends .npy to a suffixless name; target already has it.
        return target

    lister = mock.create_autospec(store_io.hub.list_hf_files_under_path, side_effect=_list)
    stager = mock.create_autospec(store_io.hub.stage_hub_file, side_effect=_stage)
    monkeypatch.setattr(store_io.hub, "list_hf_files_under_path", lister)
    monkeypatch.setattr(store_io.hub, "stage_hub_file", stager)
    return lister, stager


def test_stage_u_store_stages_flattened_and_marks_complete(tmp_path, monkeypatch):
    manifest_rows = [{"conv_id": f"c{i}", "is_eval_only": i >= 3} for i in range(4)]
    lister, _stager = _fake_hub(monkeypatch, manifest_rows, shard_rows=2)
    dest = tmp_path / "u_store"
    out = store_io.stage_u_store(dest, KINDS, (0,))
    assert out == dest
    record = json.loads((dest / "staging_manifest.json").read_text())
    assert record["complete"] is True and record["revision"] == STORE_REVISION
    assert store_io.u_store_loadable(dest, KINDS, (0,))
    # Production consumer round-trip over the staged (flattened) tree.
    arrays, meta = store_io.load_summaries(dest, KINDS, (0,), hidden_dim=DIM)
    assert arrays[("t1", 0)].shape == (4, DIM) and len(meta) == 4
    assert store_io.fit_pool_mask(meta).sum() == 3
    # Second call short-circuits: loadable dest costs no fresh listing.
    lister.reset_mock()
    store_io.stage_u_store(dest, KINDS, (0,))
    lister.assert_not_called()


def test_stage_u_store_probe_slice_not_marked_loadable(tmp_path, monkeypatch):
    manifest_rows = [{"conv_id": f"c{i}"} for i in range(4)]
    _fake_hub(monkeypatch, manifest_rows, shard_rows=2)
    dest = tmp_path / "u_probe"
    store_io.stage_u_store(dest, KINDS, (0,), max_shards_per_kind=1)
    record = json.loads((dest / "staging_manifest.json").read_text())
    assert record["complete"] is False and record["max_shards_per_kind"] == 1
    assert not store_io.u_store_loadable(dest, KINDS, (0,))
    # The probe slice still opens through the consumer at a sliced n_rows.
    arrays, meta = store_io.load_summaries(dest, KINDS, (0,), hidden_dim=DIM, n_rows=2)
    assert arrays[("prefix_end", 0)].shape == (2, DIM) and len(meta) == 2


def test_stage_u_store_probe_never_downgrades_complete_record(tmp_path, monkeypatch):
    manifest_rows = [{"conv_id": f"c{i}"} for i in range(4)]
    _fake_hub(monkeypatch, manifest_rows, shard_rows=2)
    dest = tmp_path / "u_store"
    store_io.stage_u_store(dest, KINDS, (0,))
    store_io.stage_u_store(dest, KINDS, (0,), max_shards_per_kind=1)
    record = json.loads((dest / "staging_manifest.json").read_text())
    assert record["complete"] is True
    assert store_io.u_store_loadable(dest, KINDS, (0,))
