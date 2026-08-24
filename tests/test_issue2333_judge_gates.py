"""Issue #2333 judge pre-spend gates (r1 Major 3 + Minors).

Covers: the 144-block shard-set completeness gate (missing/extra fail loud
BEFORE any Batch-API spend), per-(block, pair) draw-set consistency, the
registered 6/6-all-end_turn forced-batch probe verdict, and byte parity
between ``constants.expected_grid_slugs`` and the driver's realized
``R.block_slug`` enumeration (144 blocks / 12 ce blocks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2333_judge as J33  # noqa: E402

from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402


def test_shard_set_complete_passes_on_exact_set():
    J33.assert_shard_set_complete({"a", "b"}, {"a", "b"}, "toy")


def test_shard_set_complete_fails_on_missing_and_extra():
    with pytest.raises(RuntimeError, match="missing=\\['b'\\]"):
        J33.assert_shard_set_complete({"a"}, {"a", "b"}, "toy")
    with pytest.raises(RuntimeError, match="extra=\\['c'\\]"):
        J33.assert_shard_set_complete({"a", "b", "c"}, {"a", "b"}, "toy")


def test_draw_consistency():
    rows = [{"block_key": "b1", "pair_id": "p1", "draw": d} for d in (0, 1)] + [
        {"block_key": "b1", "pair_id": "p2", "draw": d} for d in (0, 1)
    ]
    assert J33.assert_draw_consistency(rows) == 2
    with pytest.raises(AssertionError, match="ragged"):
        J33.assert_draw_consistency([*rows, {"block_key": "b2", "pair_id": "p3", "draw": 0}])
    with pytest.raises(AssertionError):  # non-contiguous draw ids
        J33.assert_draw_consistency([{"block_key": "b", "pair_id": "p", "draw": 1}])


def test_forced_batch_probe_verdict_registered_criterion():
    """Plan §7: EXACTLY 6 items, ALL scored, EVERY draw stop_reason end_turn
    (r1 Minor: the shipped `n_probe >= 1` was weaker than registered)."""
    ok, rep = J33.forced_batch_probe_verdict({f"i{k}": 80.0 for k in range(6)}, {"end_turn": 6}, 6)
    assert ok and rep["passed"]
    # one unscored item
    scores = {f"i{k}": (80.0 if k else None) for k in range(6)}
    ok, _ = J33.forced_batch_probe_verdict(scores, {"end_turn": 6}, 6)
    assert not ok
    # a truncated draw
    ok, rep = J33.forced_batch_probe_verdict(
        {f"i{k}": 80.0 for k in range(6)}, {"end_turn": 5, "max_tokens": 1}, 6
    )
    assert not ok and rep["non_end_turn"] == {"max_tokens": 1}
    # a cache-served legacy 'unknown' fails (fresh probe cache required)
    ok, _ = J33.forced_batch_probe_verdict(
        {f"i{k}": 80.0 for k in range(6)}, {"end_turn": 5, "unknown": 1}, 6
    )
    assert not ok
    # fewer than 6 items is never a pass
    ok, _ = J33.forced_batch_probe_verdict({f"i{k}": 80.0 for k in range(5)}, {"end_turn": 5}, 5)
    assert not ok
    # empty tally (no persisted stop_reasons) is never a pass
    ok, _ = J33.forced_batch_probe_verdict({f"i{k}": 80.0 for k in range(6)}, {}, 6)
    assert not ok


def test_expected_grid_slugs_match_driver_block_enumeration(monkeypatch):
    """Byte parity: the torch-free expected-slug enumeration (the judge's
    completeness gate) == the pod driver's realized R.block_slug set."""
    monkeypatch.chdir(REPO_ROOT)
    import issue2333_run as RUN

    s1, s2 = RUN.build_pair_universe()
    blocks = RUN.enumerate_blocks_2333(s1, s2, set())
    assert {b.slug for b in blocks} == C.expected_grid_slugs()
    assert len(C.expected_grid_slugs()) == 144
    assert len(C.expected_ce_control_slugs()) == 12
    # ce slug parity with the driver's ce Block shape.
    import issue2162_run as R

    ce_blocks = [
        R.Block(cell, "ce_replace", variant, ())
        for cell in (*C.S1_CELLS, C.S2_CELL)
        for variant in C.VARIANTS
    ]
    assert {b.slug for b in ce_blocks} == C.expected_ce_control_slugs()


def test_load_grid_rows_gates_before_any_rows_are_read(tmp_path):
    """A partial blocks/ dir refuses at load time (the pre-spend gate)."""
    blocks_dir = tmp_path / "rollouts" / "blocks"
    blocks_dir.mkdir(parents=True)
    (blocks_dir / "onlyone__patch1_med__steered.jsonl").write_text(
        '{"block_key": "b", "pair_id": "p", "draw": 0}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="shard set incomplete"):
        J33.load_grid_rows(tmp_path / "rollouts")
