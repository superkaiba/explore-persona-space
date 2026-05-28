"""TDD Phase 1 — cell enumeration for the 2^4 x 3 design (task #397, plan v4 §4.1).

The new ordinal-E layout replaces #383's binary E. After the A=0 x C=1
preflight drop, we expect 12 valid ABCD combos x 3 E levels = 36 valid cells
per source, x 3 sources x 3 seeds = **324 (cell x seed) runs**.

These tests verify the *shape* of the enumeration so the implementer cannot
silently rebuild it as binary E in Phase 2. They cover plan v4 §4.1 + §4.3 +
the test enumerated as item 1 in plan v4 §14.

Expected to PASS in Phase 1 (the cells module is implemented now so the
downstream test surface has a stable cell type to work against).
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.factor_screen_397.cells import (
    E_LEVELS,
    FACTOR_DESCRIPTIONS,
    FACTOR_INDEX,
    FACTOR_NAMES,
    Cell,
    all_full_cells,
    valid_cells_per_source,
)


def test_factor_names_match_plan_order() -> None:
    """Plan v4 §4.1 names factors A..E in that order with E now ordinal."""
    assert FACTOR_NAMES == ("A", "B", "C", "D", "E")
    assert FACTOR_INDEX == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def test_e_is_ordinal_with_three_levels() -> None:
    """Plan v4 §3 + §4.3: E ∈ {0, 1, 2}, NOT binary as in #383."""
    assert E_LEVELS == (0, 1, 2)
    # E factor description block has 3 levels.
    assert set(FACTOR_DESCRIPTIONS["E"].keys()) == {0, 1, 2}


def test_e_level_descriptions_match_plan_v4_table() -> None:
    """Plan v4 §4.3 table: E0=marker+EOT (~2 tok); E1=tail-32; E2=whole-completion."""
    assert "marker+eot" in FACTOR_DESCRIPTIONS["E"][0].lower()
    assert "tail-32" in FACTOR_DESCRIPTIONS["E"][1].lower()
    assert "whole-completion" in FACTOR_DESCRIPTIONS["E"][2].lower()


def test_all_full_cells_yields_48_unique_keys_before_preflight() -> None:
    """2^4 binary x 3 ordinal = 48 nominal cells per source (BEFORE A=0xC=1 drop)."""
    cells = all_full_cells()
    assert len(cells) == 48, (
        f"Expected 48 nominal cells (2^4 x 3); got {len(cells)}. "
        "If you got 32, you regressed to #383's binary E."
    )
    keys = {c.key for c in cells}
    assert len(keys) == 48


def test_valid_cells_per_source_yields_36_after_a0_c1_drop() -> None:
    """Plan v4 §4.1: 12 valid ABCD combos x 3 E levels = 36 valid cells per source.

    The A=0 x C=1 corner is dropped at preflight per #383 (4 ABCD combos x
    3 E levels = 12 cells removed from the nominal 48).
    """
    valid = valid_cells_per_source()
    assert len(valid) == 36, (
        f"Expected 36 valid cells per source after A=0xC=1 drop; got {len(valid)}."
    )
    # None of the surviving cells should be A=0 x C=1.
    for c in valid:
        assert not (c.a == 0 and c.c == 1), f"Cell {c.key} survived the A=0xC=1 preflight drop"


def test_canonical_cell_count_matches_plan() -> None:
    """108 valid cells per seed x 3 seeds = 324 (cell x seed) runs (plan v4 §4.1)."""
    per_source = len(valid_cells_per_source())
    n_sources = 3
    n_seeds = 3
    per_seed_total = per_source * n_sources  # = 108
    grand_total = per_seed_total * n_seeds  # = 324
    assert per_seed_total == 108
    assert grand_total == 324


def test_cell_from_key_parses_ordinal_e() -> None:
    """``Cell.from_key("10012")`` parses E=2 (plan v4 §14 item 1)."""
    cell = Cell.from_key("10012")
    assert cell.a == 1
    assert cell.b == 0
    assert cell.c == 0
    assert cell.d == 1
    assert cell.e == 2
    assert cell.key == "10012"


def test_cell_from_key_accepts_all_three_e_levels() -> None:
    for e in (0, 1, 2):
        cell = Cell.from_key(f"0000{e}")
        assert cell.e == e


def test_cell_from_key_roundtrip_for_all_nominal_cells() -> None:
    for cell in all_full_cells():
        assert Cell.from_key(cell.key) == cell


def test_cell_rejects_invalid_e_level() -> None:
    """E=3 must raise; #397 is ordinal K=3, not K=4."""
    with pytest.raises(ValueError):
        Cell(0, 0, 0, 0, 3)
    with pytest.raises(ValueError):
        Cell.from_key("00003")


def test_cell_rejects_non_binary_abcd() -> None:
    with pytest.raises(ValueError):
        Cell(2, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        Cell.from_key("20000")


def test_cell_rejects_wrong_key_length() -> None:
    with pytest.raises(ValueError):
        Cell.from_key("0000")
    with pytest.raises(ValueError):
        Cell.from_key("000001")
