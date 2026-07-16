"""#825 turn-dynamics fit unit tests (code-review v21 + v22 findings).

- ``_general_linear_cos`` + ``_rank_truncated_cols`` (v21 Critical 2): pins the
  non-degenerate value on a synthetic rank-deficient operator pair AND the
  full-square-U artifact (identically 1.0) the truncation guards against.
- fp16 round-trip rank recovery (v22 Critical 1): the fp16 beta persistence
  noise tail (~1e-4*S.max) defeated rel=1e-6; pins rel=1e-3 + the row-count
  clamp recovering the true rank and the cos staying non-degenerate.
- ``_transfer_nulls`` (Major 3): the batched GEMM identity reproduces a
  brute-force per-draw ss_res on a tiny synthetic case, and every (i, j)
  transfer cell carries null keys.
- ``stratified_subsample_ids`` (Major 6): both source prefixes present with
  proportional allocation, deterministic under the seed, and NOT the
  first-N-of-sorted (single-source) selection the fix replaces.
- ``_r10_key_for_turn`` / ``_r10_node_for_turn`` (crash-fix round 11): the
  G-C parity lookup translates the refit's 1-based exchange ordinal t to the
  round-10 curve's 0-based assistant turns-list key 2t-1; the pinned values
  FAIL under the old ``str(t)`` lookup (even-t cells skipped, odd-t cells
  compared against the wrong shallower node).
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
    _r10_key_for_turn,
    _r10_node_for_turn,
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


def test_rank_truncation_survives_fp16_beta_round_trip(tmp_path):
    """Code-review v22 Critical 1: fp16 beta persistence lifts the SVD noise
    tail to ~1e-4*S.max, so the old rel=1e-6 rule read rank near-FULL and the
    cos statistic re-squashed to ~1. Pins, on a DENSE low-rank matrix through
    the production ``np.save(.astype(np.float16))`` -> reload path:
    (a) the defeated rule is reproduced (rank@1e-6 >> true rank),
    (b) the new rel=1e-3 threshold alone recovers the true rank,
    (c) the algebraic row-count clamp binds independently of the threshold,
    (d) the cos statistic stays non-degenerate (not re-squashed to ~1).
    """
    d, r = 256, 8
    torch.manual_seed(0)
    # DENSE rank-r matrix (A @ B): fp16 quantization error is a dense
    # full-rank perturbation, exactly the production failure shape (a
    # zero-padded construction would quantize exactly and hide the tail).
    b = (torch.randn(d, r) @ torch.randn(r, d)).to(torch.float32)
    u_true = torch.linalg.svd(b, full_matrices=False).U[:, :r]  # true col space
    p = tmp_path / "beta_fp16.npy"
    np.save(p, b.numpy().astype(np.float16))  # the production persistence path
    b16 = torch.from_numpy(np.load(p).astype(np.float32))  # the production reload
    U, S, _vh = torch.linalg.svd(b16, full_matrices=False)
    # (a) the defeated v21 rule: fp16 tail ~1e-4*S.max >> 1e-6 -> near-full rank
    rank_old_rule = int((float(S.max()) * 1e-6 < S).sum())
    assert rank_old_rule > 4 * r, rank_old_rule  # reads near-D today (~248)
    # (b) threshold alone (rel=1e-3, ~10x above the measured ~1e-4 tail)
    _, rank_thresh = _rank_truncated_cols(U, S)
    assert rank_thresh == r, rank_thresh
    # (c) row-count clamp binds even when the threshold would keep more
    _, rank_clamped = _rank_truncated_cols(U, S, rel=1e-6, max_rank=r)
    assert rank_clamped == r, rank_clamped
    # (d) non-degenerate cos on the fp16-round-tripped truncated basis:
    # b_j with known in-span (3.0) / out-of-span (4.0) energy vs col(b)
    q_in = u_true[:, 0]
    v = torch.randn(d)
    v -= u_true @ (u_true.T @ v)
    q_out = v / torch.linalg.norm(v)
    b_j = torch.zeros(d, d)
    b_j[:, 0] = 3.0 * q_in + 4.0 * q_out
    u_r, rank = _rank_truncated_cols(U, S, max_rank=r)
    assert rank == r
    got = _general_linear_cos(u_r, b_j)
    assert abs(got - 0.6) < 1e-2, got  # 3/5, within fp16-perturbation tolerance
    assert got < 0.99, got  # NOT re-squashed to ~1 (the v22 vacuity signature)


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


def test_gc_r10_key_alignment_translates_exchange_ordinal_to_turns_list_index():
    """Crash-fix round 11 (G-C key-space mismatch): the refit keys cells by
    ``turn`` = 1-based exchange ordinal t, while the round-10 reference curve
    is keyed by the 0-based assistant turns-list index = 2t-1 (odd keys only).
    Pins the translated lookup ``str(2*t - 1)``: under the OLD ``str(t)``
    lookup, t=2 returned NO node (cell silently skipped) and t=3 returned
    key "3" — the WRONG (shallower) node belonging to exchange t=2 — the
    systematic false FAIL that blocked the instruct headline.
    """
    # Round-10 curve SHAPE (results.json layer-19): odd keys only, one
    # distinct r2 per key so a wrong-node selection is detectable.
    curve = {
        "1": {"ctx_logged": {"r2": 0.10}, "n": 10},
        "3": {"ctx_logged": {"r2": 0.20}, "n": 8},
        "5": {"ctx_logged": {"r2": 0.30}, "n": 6},
    }
    # translation: exchange ordinal t -> turns-list index 2t-1
    assert [_r10_key_for_turn(t) for t in (1, 2, 3)] == [1, 3, 5]

    # t=1: the two conventions coincide (key 1) — the cell the old lookup
    # got right by accident.
    assert _r10_node_for_turn(curve, 1) == (1, 0.10)
    # t=2: OLD lookup curve.get("2") -> {} -> r2 None -> cell skipped.
    key, r2 = _r10_node_for_turn(curve, 2)
    assert key == 3
    assert r2 == 0.20
    # t=3: OLD lookup selected key "3" (r2 0.20, exchange t=2's node).
    key, r2 = _r10_node_for_turn(curve, 3)
    assert key == 5
    assert r2 == 0.30
    assert r2 != curve["3"]["ctx_logged"]["r2"]
    # absent (key 7) / malformed nodes drop out of the comparison (r2 None)
    assert _r10_node_for_turn(curve, 4) == (7, None)
    assert _r10_node_for_turn({"1": 3.0}, 1) == (1, None)


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
