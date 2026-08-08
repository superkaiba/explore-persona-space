"""Pins for the #1482 full-width dense->SAE fit driver.

Covers the two pieces with no dense reference anywhere else in the pipeline:
the per-batch on-device target scatter (the full dense target is 63 GB and is
never materialized, so a wrong scatter is silent) and the blocked ragged CSR
gather (blocked purely to bound int64 index temporaries — it MUST be
bit-identical to the one-shot form).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from explore_persona_space.task_workflow import repo_root

# Import the driver from THIS tree's scripts/, not `repo_root()/scripts`.
# `repo_root()` branch-guards to the MAIN checkout, so under a worktree run the
# old form (a) exercised main's copy of the module under test instead of the
# branch's, and (b) leaked a FOREIGN checkout's scripts/ onto sys.path for the
# whole session -- which silently defeats the #1296 sys.path negative control in
# tests/test_backend_poll.py (it can only scrub the local tree's scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue1482_densesae_fullwidth as M


def _write_store(work: Path, dense: np.ndarray) -> M.YStore:
    """Persist ``dense`` as the driver's on-disk CSR layout and open it."""
    csr = sp.csr_matrix(dense)
    nnz = int(csr.nnz)
    np.save(work / "y_indptr.npy", csr.indptr.astype(np.int64))
    np.memmap(work / "y_indices.i32", dtype=np.int32, mode="w+", shape=(nnz,))[:] = csr.indices
    for p in M.POOLINGS:
        mm = np.memmap(work / f"y_val_{p}.f16", dtype=np.float16, mode="w+", shape=(nnz,))
        mm[:] = csr.data.astype(np.float16)
        mm.flush()
    return M.YStore(work, dense.shape[0], nnz)


@pytest.fixture
def tiny_store(tmp_path):
    rng = np.random.default_rng(0)
    dense = np.zeros((40, M.DICT_SIZE), dtype=np.float32)
    for r in range(40):
        cols = rng.choice(M.DICT_SIZE, size=rng.integers(1, 9), replace=False)
        dense[r, cols] = rng.normal(size=len(cols)).astype(np.float16).astype(np.float32)
    return dense, _write_store(tmp_path, dense)


def test_csr_rows_blocked_gather_matches_one_shot(tiny_store):
    dense, store = tiny_store
    rows = np.array([0, 3, 4, 9, 17, 22, 30, 39], dtype=np.int64)
    big = store.csr_rows(rows, "mean", block=10_000).toarray()
    small = store.csr_rows(rows, "mean", block=3).toarray()
    assert np.array_equal(big, small), "blocking changed the gather"
    assert np.array_equal(big, dense[rows])


def test_csr_rows_handles_all_empty_rows(tmp_path):
    """A registry row absent from the store contributes zero nnz — the ragged
    gather must not mis-align the rows that follow it."""
    dense = np.zeros((6, M.DICT_SIZE), dtype=np.float32)
    dense[1, 5] = 2.0
    dense[4, 7] = -3.0
    store = _write_store(tmp_path, dense)
    rows = np.arange(6, dtype=np.int64)
    assert np.array_equal(store.csr_rows(rows, "mean", block=2).toarray(), dense)


def test_col_stats_matches_dense(tiny_store):
    dense, store = tiny_store
    rows = np.arange(0, 40, 2, dtype=np.int64)
    s1, s2 = store.col_stats(rows, "mean", block=7)
    ref = dense[rows].astype(np.float64)
    assert np.allclose(s1, ref.sum(0), atol=1e-9)
    assert np.allclose(s2, (ref**2).sum(0), atol=1e-9)


def test_scatter_targets_matches_dense_reference(tiny_store):
    dense, store = tiny_store
    dev = torch.device("cpu")
    bundle = store.gpu_bundle("mean", dev)
    rows = np.array([2, 2, 11, 39, 0], dtype=np.int64)  # repeats are legal in a batch
    got = M._scatter_targets(bundle, torch.as_tensor(rows, dtype=torch.int64), M.DICT_SIZE)
    assert got.shape == (len(rows), M.DICT_SIZE)
    assert np.array_equal(got.numpy(), dense[rows])


def test_scatter_targets_col_map_restricts_and_remaps(tiny_store):
    """The panel gate cell trains 16,384 outputs out of the 131,072-wide store;
    -1 in the map DROPS a column, and kept columns land at their mapped index."""
    dense, store = tiny_store
    dev = torch.device("cpu")
    bundle = store.gpu_bundle("mean", dev)
    active = np.unique(sp.csr_matrix(dense).indices)
    panel = np.sort(active[: max(2, len(active) // 2)])
    col_map = torch.full((M.DICT_SIZE,), -1, dtype=torch.int64)
    col_map[torch.as_tensor(panel, dtype=torch.int64)] = torch.arange(len(panel))
    rows = np.arange(40, dtype=np.int64)
    got = M._scatter_targets(
        bundle, torch.as_tensor(rows, dtype=torch.int64), len(panel), col_map
    ).numpy()
    assert got.shape == (40, len(panel))
    assert np.array_equal(got, dense[np.ix_(rows, panel)])


def test_scatter_targets_empty_batch_is_all_zero(tmp_path):
    dense = np.zeros((3, M.DICT_SIZE), dtype=np.float32)
    store = _write_store(tmp_path, dense)
    bundle = store.gpu_bundle("mean", torch.device("cpu"))
    got = M._scatter_targets(bundle, torch.as_tensor([0, 1], dtype=torch.int64), M.DICT_SIZE)
    assert float(got.abs().sum()) == 0.0


def test_score_pools_only_scored_columns_and_panel():
    ss_res = np.zeros(M.DICT_SIZE, dtype=np.float64)
    ss_tot = np.zeros(M.DICT_SIZE, dtype=np.float64)
    panel = np.array([1, 2], dtype=np.int64)
    ss_tot[[1, 2, 3]] = [10.0, 10.0, 100.0]
    ss_res[[1, 2, 3]] = [5.0, 0.0, 100.0]
    got = M._score(ss_res, ss_tot, panel, 1.0)
    # full: 1 - 105/120; panel: 1 - 5/20. A zero-variance column enters neither.
    assert got["pooled_r2_full"] == pytest.approx(1.0 - 105.0 / 120.0)
    assert got["pooled_r2_panel"] == pytest.approx(0.75)
    assert got["n_scored_columns"] == 3
    assert got["n_zero_variance_columns"] == M.DICT_SIZE - 3


def test_score_perfeature_leaves_zero_variance_nan():
    ss_res = np.zeros(M.DICT_SIZE, dtype=np.float64)
    ss_tot = np.zeros(M.DICT_SIZE, dtype=np.float64)
    ss_tot[7] = 4.0
    ss_res[7] = 1.0
    pf = M._score_perfeature(ss_res, ss_tot)
    assert pf["r2"][7] == pytest.approx(0.75)
    assert np.isnan(pf["r2"][0])
    assert int(pf["scored"].sum()) == 1


def test_smoke_rows_keeps_every_split_and_refuses_a_degenerate_one():
    which = np.array([0] * 10 + [1] * 50 + [2] * 4, dtype=np.int8)
    args = type("A", (), {"smoke_holdout": 5, "smoke_train": 20, "smoke_val": 3})()
    keep = M._smoke_rows(which, args)
    assert np.bincount(which[keep], minlength=3).tolist() == [5, 20, 3]
    assert np.array_equal(keep, np.sort(keep)), "row order must stay ascending"

    thin = np.array([0] * 10 + [1] * 50 + [2], dtype=np.int8)
    with pytest.raises(SystemExit, match="need >= 2"):
        M._smoke_rows(thin, args)


def test_cell_registry_is_the_single_source_of_truth():
    """The launcher fans out over these names; a rename here must not silently
    leave the shell queue pointing at cells the driver rejects."""
    launcher = (repo_root() / "scripts" / "issue1482_densesae_launch.sh").read_text()
    for cell in M.CELLS:
        assert cell in launcher, f"{cell} missing from the launcher queue"
    assert M.CELLS == (
        "ridge__mean",
        "ridge__max",
        "ridge__frac",
        "mlp__mean",
        "mlp__max",
        "mlp__frac",
        "mlpgate__mean",
    ), "the width-32768 capacity cell was removed by user directive; 7 required cells"
    assert not hasattr(M, "DEFAULT_CELLS"), "the vacuous optional-cell filter is gone"
    for cell in M.CELLS:
        method, pooling = M._parse_cell(cell)
        assert method in {"ridge", "mlp", "mlpgate"}
        assert pooling in M.POOLINGS
    with pytest.raises(SystemExit, match="unknown cell"):
        M._parse_cell("ridge__nope")


def test_project_root_does_not_require_a_tasks_directory():
    """The compute lanes run sparse/shallow checkouts with no ``tasks/``.

    ``task_workflow.repo_root()`` REFUSES such a checkout, which made the driver
    unimportable on the pod; this driver reads only scripts/ + src/ +
    eval_results/ and never touches task state, so it resolves its root from
    ``__file__`` instead. Pinned because reintroducing ``repo_root()`` here is an
    easy, silent regression that only surfaces on the pod.
    """
    src = (repo_root() / "scripts" / "issue1482_densesae_fullwidth.py").read_text()
    assert "import repo_root" not in src, "driver must not import the tasks/-requiring resolver"
    assert "PROJECT_ROOT = repo_root()" not in src
    assert Path(M.__file__).resolve().parent.parent == M.PROJECT_ROOT


def test_staged_prefix_is_a_mirror_root_under_work():
    """stage_hub_prefix's dest is a MIRROR ROOT — files land at
    dest/<repo-relative path>, so the consumed dir must be root/<prefix> (#1774)."""
    args = type("A", (), {"work": Path("/tmp/w"), "local_store": "", "local_inputs": ""})()
    assert M._staged(args, M.STORE_PREFIX) == Path("/tmp/w/stage") / M.STORE_PREFIX
    assert M._store_dir(args) == Path("/tmp/w/stage") / M.STORE_PREFIX
    assert M._inputs_dir(args) == Path("/tmp/w/stage") / M.INPUTS_PREFIX
