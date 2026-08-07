"""Issue #2162 stage-2 driver — CPU pins for the plan §4.2 restored design.

r1 C3 pinned here: stage 2 is pair-difference ADD edits (mode="add",
delta = V(B) - V(A)) at SINGLE layers {8,12,14,16,19,22,26} x doses {1,4},
BOTH steered + shuffled-donor arms, 1 greedy draw per pair, budget <= 12,096
(plan §4.3). Also: block-key grammar, smoke arm-class coverage, best-cells
loader shape validation, and the stage-2 regime fingerprint suffix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_run as R  # noqa: E402
import issue2162_stage2 as S  # noqa: E402

from explore_persona_space.experiments.issue2162 import bank2162 as B  # noqa: E402


@pytest.fixture(scope="module")
def pairs():
    return B.build_pairs()


def test_stage2_constants_pin_plan_4_2():
    """Plan §4.2 verbatim (r1 C3): single layers, doses {1,4}, both arms,
    1 greedy draw."""
    assert S.STAGE2_LAYERS == (8, 12, 14, 16, 19, 22, 26)
    assert S.STAGE2_DOSES == (1, 4)
    assert S.STAGE2_ARMS == ("steered", "shuffled")
    assert S.STAGE2_DRAWS == 1
    assert S.STAGE2_TEMPERATURE == 0.0


def test_stage2_block_key_grammar():
    b = S.Stage2Block("instr_format", "ce", "shuffled", 26, 4, ("x",))
    assert b.key == "instr_format|ce|shuffled|L26|d4"
    assert b.n_pairs == 1


def test_stage2_rows_are_single_layer_add_mode():
    """r1 C3: the emitted rows carry mode="add" and exactly ONE patched layer
    (source pin — the row dict is built inside the GPU-bound block runner)."""
    src = (REPO_ROOT / "scripts" / "issue2162_stage2.py").read_text()
    assert '"mode": "add"' in src
    assert '"layers_patched": [block.layer]' in src
    assert "layers_for_dose" not in src  # the r1 deviation's multi-layer helper is GONE


def test_enumerate_stage2_blocks_full_and_budget(pairs):
    """12 survivors x 2 arms x 7 layers x 2 doses x 36 pairs x 1 draw =
    12,096 rollouts — exactly the plan §4.3 cap."""
    cells_with_pairs = sorted({p.cell for p in pairs if len(B.pairs_by_cell(pairs)[p.cell]) == 36})
    best = [{"cell": c, "slot": "ce"} for c in cells_with_pairs[:12]]
    assert len(best) == 12
    blocks = S.enumerate_stage2_blocks(best, pairs, smoke=False)
    assert len(blocks) == 12 * 2 * 7 * 2
    assert {b.arm for b in blocks} == {"steered", "shuffled"}
    assert {b.layer for b in blocks} == set(S.STAGE2_LAYERS)
    assert {b.dose for b in blocks} == {1, 4}
    totals = R.grid_totals(blocks, S.STAGE2_DRAWS)
    assert totals["rollouts_total"] == 12_096  # the budget assert's boundary
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys)


def test_enumerate_stage2_blocks_smoke_covers_both_arms(pairs):
    """Per-arm-class smoke coverage: the smoke slice keeps BOTH arms (the
    shuffled arm exercises the donor seam) at 1 cell x 1 layer x 2 doses."""
    best = [{"cell": "instr_format", "slot": "ce"}, {"cell": "verbosity", "slot": "pe"}]
    blocks = S.enumerate_stage2_blocks(best, pairs, smoke=True)
    assert {b.arm for b in blocks} == {"steered", "shuffled"}
    assert {b.layer for b in blocks} == {S.STAGE2_LAYERS[0]}
    assert all(b.cell == "instr_format" for b in blocks)  # smoke: 1 survivor
    assert all(b.n_pairs == R.SMOKE_PAIRS_PER_CELL for b in blocks)


def test_load_best_cells_shape_validation(tmp_path):
    path = tmp_path / "best_cells.json"
    with pytest.raises(AssertionError, match="missing"):
        S.load_best_cells(path)
    path.write_text(json.dumps({"cells": []}))
    with pytest.raises(AssertionError, match="zero survivors"):
        S.load_best_cells(path)
    path.write_text(json.dumps({"cells": [{"cell": "c", "slot": "ce"}] * 13}))
    with pytest.raises(AssertionError, match="cap is 12"):
        S.load_best_cells(path)
    path.write_text(json.dumps({"cells": [{"cell": "instr_format", "slot": "ce"}]}))
    assert S.load_best_cells(path) == [{"cell": "instr_format", "slot": "ce"}]


def test_stage2_regime_fp_is_distinct_from_stage1(tmp_path):
    """Stage-2 done-state must never satisfy (or poison) a stage-1 resume:
    the fingerprint carries an explicit stage-2 suffix."""
    args = R.parse_args(
        ["--phase", "grid", "--out-root", str(tmp_path / "o"), "--log-dir", str(tmp_path / "l")]
    )
    cfg = R.build_config(args)
    base = R.regime_fingerprint(cfg, "banksha")
    fp = S.stage2_regime_fp(cfg, "banksha")
    assert fp == f"{base}-stage2-add-K1"
    assert fp != base


def test_stage2_queue_namespace_isolated():
    """Stage-2 blocks queue + done-write under the ``stage2_blocks`` namespace
    (never colliding with the stage-1 ``blocks`` namespace on a shared
    out-root)."""
    src = (REPO_ROOT / "scripts" / "issue2162_stage2.py").read_text()
    assert '"stage2_blocks"' in src
    b = S.Stage2Block("instr_format", "ce", "steered", 8, 1, ("p",))
    p1 = R.block_done_path(Path("/o"), b, "stage2_blocks")
    p2 = R.block_done_path(Path("/o"), b, "blocks")
    assert p1 != p2 and "stage2_blocks" in str(p1)
