"""#546 round-3 — eval cell-subset filter contract (smoke = sweep with one cell).

Pins the ``_apply_cell_filters`` extension in ``scripts/i464_po_eval.py``
that closes the ``smoke-crosseval-enumerates-full-grid`` failure: the
dispatcher's ``ARMS/SEEDS/PERSONAS/EPOCHS_OVERRIDE`` subset now threads
into the eval-phase cell enumeration via ``--arms-filter`` /
``--seeds-filter`` / ``--personas-filter`` / ``--epochs-filter``.

1. Full-grid default (no filters) is byte-stable: cn_i546 enumerates
   120 cells = 3 arms x 5 seeds x 2 personas x 4 epochs, i.e. 360 eval
   encodings (3 per cell) — and the no-op filter call returns the input
   list unchanged for cn_i546 AND cn_i533 (production byte-compat).
2. The plan §3 smoke contract subset (system_plain / 42 / pirate / e1)
   enumerates exactly ONE cell whose probe set is 3 eval encodings.
3. Validation fails LOUD: out-of-grid values, --epochs-filter on a
   variant without an epoch dimension, and filter combinations that
   match zero cells all raise ValueError (never a silent empty eval).

CPU-only; no GPU, no network, no HF.
"""

from __future__ import annotations

import pytest

from scripts.i464_po_eval import (
    _all_po_cells,
    _apply_cell_filters,
    _eval_encodings_for_cell,
)

# ── 1. Full-grid default — byte-stable enumeration ─────────────────────


def test_cn_i546_full_grid_is_120_cells_360_encodings():
    cells = _all_po_cells(variant="cn_i546")
    assert len(cells) == 120  # 3 arms x 5 seeds x 2 personas x 4 epochs
    n_encodings = sum(len(_eval_encodings_for_cell(arm, persona)) for arm, _, persona, _ in cells)
    assert n_encodings == 360


@pytest.mark.parametrize("variant", ["cn_i546", "cn_i533"])
def test_no_filters_is_identity(variant):
    cells = _all_po_cells(variant=variant)
    assert _apply_cell_filters(cells, variant) == cells


# ── 2. The §3 smoke contract subset = exactly one cell, 3 encodings ────


def test_smoke_contract_subset_is_one_cell_three_encodings():
    cells = _apply_cell_filters(
        _all_po_cells(variant="cn_i546"),
        "cn_i546",
        arms=["system_plain"],
        seeds=[42],
        personas=["pirate"],
        epochs=[1],
    )
    assert cells == [("system_plain", 42, "pirate", 1)]
    (arm, _, persona, _) = cells[0]
    assert _eval_encodings_for_cell(arm, persona) == [
        "system_pirate",
        "system_villain",
        "default_assistant",
    ]


def test_single_dimension_filter_scales_grid():
    cells = _apply_cell_filters(_all_po_cells(variant="cn_i546"), "cn_i546", epochs=[1])
    assert len(cells) == 30  # 120 / 4 epochs
    assert all(c[3] == 1 for c in cells)


# ── 3. Fail-loud validation ─────────────────────────────────────────────


def test_out_of_grid_persona_raises():
    with pytest.raises(ValueError, match="--personas-filter"):
        _apply_cell_filters(_all_po_cells(variant="cn_i546"), "cn_i546", personas=["ninja"])


def test_out_of_grid_epoch_raises():
    with pytest.raises(ValueError, match="--epochs-filter"):
        _apply_cell_filters(_all_po_cells(variant="cn_i546"), "cn_i546", epochs=[4])


def test_epochs_filter_rejected_for_epochless_variant():
    with pytest.raises(ValueError, match="no epoch dimension"):
        _apply_cell_filters(_all_po_cells(variant="po"), "po", epochs=[1])


def test_empty_intersection_raises():
    # Simulates `--epoch 2 --epochs-filter 1`: both values are in-grid but
    # the pre-filtered cell list no longer contains epoch 1.
    pre_filtered = [c for c in _all_po_cells(variant="cn_i546") if c[3] == 2]
    with pytest.raises(ValueError, match="ZERO cells"):
        _apply_cell_filters(pre_filtered, "cn_i546", epochs=[1])
