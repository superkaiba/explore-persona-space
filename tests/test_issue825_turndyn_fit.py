"""#825 turn-dynamics fit unit tests (code-review v21 findings).

- ``_general_linear_cos`` + ``_rank_truncated_cols`` (Critical 2): pins the
  non-degenerate value on a synthetic rank-deficient operator pair AND the
  full-square-U artifact (identically 1.0) the truncation guards against.
- ``_transfer_nulls`` (Major 3): the batched GEMM identity reproduces a
  brute-force per-draw ss_res on a tiny synthetic case, and every (i, j)
  transfer cell carries null keys.
- ``stratified_subsample_ids`` (Major 6): both source prefixes present with
  proportional allocation, deterministic under the seed, and NOT the
  first-N-of-sorted (single-source) selection the fix replaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue825_turndyn_fit import (  # noqa: E402
    _general_linear_cos,
    _rank_truncated_cols,
    _transfer_nulls,
)
from issue825_turndyn_harvest import stratified_subsample_ids  # noqa: E402


def test_general_linear_cos_rank_truncated_pins_synthetic_value():
    d, r = 64, 8
    torch.manual_seed(0)
    b_i = torch.zeros(d, d, dtype=torch.float64)
    b_i[:r] = torch.randn(r, d, dtype=torch.float64)  # col(b_i) = span(e_0..e_{r-1}) a.s.
    U, S, _vh = torch.linalg.svd(b_i, full_matrices=False)
    u_r, rank = _rank_truncated_cols(U, S)
    assert rank == r
    a, b = 3.0, 4.0
    b_j = torch.zeros(d, d, dtype=torch.float64)
    b_j[0, 0] = a  # in-span energy
    b_j[r + 1, 0] = b  # out-of-span energy
    got = _general_linear_cos(u_r, b_j)
    assert abs(got - a / np.hypot(a, b)) < 1e-9, got  # 3/5 = 0.6 exactly
    assert got < 1.0 - 1e-6  # the mechanizable non-degeneracy assert (v21 C2)
    # the artifact the truncation removes: full square U => identically 1.0
    artifact = _general_linear_cos(U, b_j)
    assert abs(artifact - 1.0) < 1e-9, artifact


def test_transfer_nulls_match_bruteforce_and_carry_all_cell_keys():
    rng = np.random.default_rng(0)
    turns = [1, 2]
    n1, n2, p = 12, 10, 5
    by_turn = {1: np.arange(n1), 2: np.arange(n1, n1 + n2)}
    Y = rng.normal(size=(n1 + n2, p))
    fold_labels = {
        1: np.asarray([i % 2 for i in range(n1)], dtype=np.int64),
        2: np.asarray([i % 2 for i in range(n2)], dtype=np.int64),
    }
    used_folds = {1: [0, 1], 2: [0, 1]}
    pred_blocks: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    ss_tot: dict[tuple[int, int], float] = {}
    for i in turns:
        for j in turns:
            tot = 0.0
            for f in (0, 1):
                te = by_turn[j][fold_labels[j] == f]
                pred_blocks.setdefault((i, j), {})[f] = rng.normal(size=(te.size, p)).astype(
                    np.float16
                )
                true = Y[te]
                tot += float(np.sum((true - true.mean(0)) ** 2))
            ss_tot[(i, j)] = tot
    n_draws, seed = 7, 1092
    out = _transfer_nulls(
        turns, by_turn, fold_labels, used_folds, pred_blocks, Y, ss_tot, n_draws, 3, seed
    )
    # every (i, j) transfer cell carries null keys (the v21 M3 mechanizable check)
    assert set(out) == {f"{i}->{j}" for i in turns for j in turns}
    for node in out.values():
        assert node["null_n_draws"] == n_draws
        assert np.isfinite(node["null_mean"]) and np.isfinite(node["null_hi"])
    # brute-force one off-diagonal cell (1 -> 2): frozen preds vs permuted truth
    j, sig = 2, (0, 1)
    rows_order = np.concatenate([by_turn[j][fold_labels[j] == f] for f in sig])
    pred = np.concatenate([pred_blocks[(1, j)][f] for f in sig], axis=0).astype(np.float64)
    yj = Y[rows_order].astype(np.float32).astype(np.float64)
    perms = np.argsort(
        np.random.default_rng(seed + 104_729 * j).random((n_draws, rows_order.size)), axis=1
    )
    ref = np.asarray(
        [1.0 - float(np.sum((yj[perms[d]] - pred) ** 2)) / ss_tot[(1, j)] for d in range(n_draws)]
    )
    assert abs(out["1->2"]["null_mean"] - ref.mean()) < 1e-4
    assert abs(out["1->2"]["null_max"] - ref.max()) < 1e-4


def test_stratified_subsample_ids_spans_sources_and_is_deterministic():
    ids = [f"lmsys_{i:06d}" for i in range(100)] + [f"wildchat_{i:06d}" for i in range(50)]
    sel = stratified_subsample_ids(ids, 30, seed=42)
    assert len(sel) == 30
    counts: dict[str, int] = {}
    for cid in sel:
        counts[cid.rsplit("_", 1)[0]] = counts.get(cid.rsplit("_", 1)[0], 0) + 1
    assert counts == {"lmsys": 20, "wildchat": 10}  # proportional, both sources present
    assert sel == stratified_subsample_ids(ids, 30, seed=42)  # deterministic
    assert sel != sorted(set(ids))[:30]  # NOT first-N-of-sorted (the Major-6 bug)
    assert stratified_subsample_ids(ids, 10_000) == sorted(set(ids))
    assert stratified_subsample_ids(ids, 0) == []
