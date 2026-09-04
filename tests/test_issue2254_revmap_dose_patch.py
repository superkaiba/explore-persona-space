"""Unit tests for issue #2254 round-8 reverse-map dose/patch driver."""

from __future__ import annotations

import numpy as np
import pytest

import scripts.issue2254_revmap_dose_patch as r8


def test_round8_registry_is_exact_and_collision_free():
    cells = r8.registered_cells()
    assert len(cells) == 16
    ids = [r8._cell_id(cell) for cell in cells]
    assert len(set(ids)) == 16
    assert set(ids[:4]) == {
        "evil__rvm__ctx__L14__c8",
        "evil__rvm__ctx__L14__c16",
        "sycophancy__rvm__ctx__L14__c8",
        "sycophancy__rvm__ctx__L14__c16",
    }
    assert set(ids[4:]) == {
        f"{behavior}__rvm__{op}__L{layer}"
        for behavior in r8.ROUND_BEHAVIORS
        for op in ("proj", "ablate")
        for layer in r8.ROUND_LAYERS
    }


def test_round8_registry_refuses_out_of_scope_cells():
    with pytest.raises(AssertionError):
        r8._assert_cell_family(
            [
                {
                    "behavior": "hallucination",
                    "kind": "steer",
                    "direction": "revmap",
                    "position": "context",
                    "layer_config": "L14",
                    "c": 8.0,
                }
            ],
            phase="steer",
            smoke=True,
        )
    with pytest.raises(AssertionError):
        r8._assert_cell_family(
            [
                {
                    "behavior": "evil",
                    "kind": "patch",
                    "direction": "revmap",
                    "op": "proj",
                    "breadth": "mid",
                    "layer_config": "L14",
                    "layer": 14,
                }
            ],
            phase="patch",
            smoke=True,
        )


def test_four_way_sharding_covers_each_cell_once_and_balances_phases():
    steer = r8.registered_steer_cells()
    patch = r8.registered_patch_cells()
    realized = []
    for shard_id in range(4):
        steer_shard = steer[shard_id::4]
        patch_shard = patch[shard_id::4]
        assert len(steer_shard) == 1
        assert len(patch_shard) == 3
        realized.extend(r8._cell_id(cell) for cell in steer_shard + patch_shard)
    assert sorted(realized) == sorted(r8._cell_id(cell) for cell in steer + patch)


@pytest.mark.parametrize("dose,expected", [(8.0, 504.0), (16.0, 1008.0)])
def test_high_dose_to_alpha(dose: float, expected: float):
    cell = next(cell for cell in r8.registered_steer_cells() if cell["c"] == dose)
    assert r8._steer_alphas(cell, {"L14": 63.0}) == {"L14": expected}


def test_smoke_exercises_steer_and_both_patch_operations():
    cells = r8.registered_cells(smoke=True)
    assert [cell["kind"] for cell in cells] == ["steer", "patch", "patch"]
    assert {cell.get("op") for cell in cells if cell["kind"] == "patch"} == {
        "proj",
        "ablate",
    }


def test_judge_context_ids_fit_batch_custom_id_budget():
    for cell in r8.registered_cells():
        context_id = r8._judge_context_id(cell, 43, 199)
        assert "__" not in context_id
        assert len(context_id) <= 49


def test_coherence_rubric_is_form_only_and_language_neutral():
    rubric = r8.coherence_rubric()
    assert "FORM-ONLY" in rubric
    assert "language or script used" in rubric
    assert "do not lower the score" in rubric
    assert "{answer}" in rubric
    assert "CJK" not in rubric  # spell out the scripts; do not conflate the metrics by label
    assert r8.COHERENCE_THRESHOLD == 50.0


@pytest.mark.parametrize(
    "op,cell,expected",
    [
        ("proj", [25.0, 50.0], 0.5),
        ("ablate", [25.0, 50.0], 0.5),
    ],
)
def test_fraction_of_ceiling_formula(op: str, cell: list[float], expected: float):
    read = r8._fraction_of_ceiling(
        np.asarray(cell),
        np.asarray([0.0, 0.0]),
        np.asarray([50.0, 100.0]),
        op,
        key=f"fixture_{op}",
    )
    assert read["fraction_point"] == pytest.approx(expected)
    assert read["n_degenerate_draws"] == 0


def test_fraction_of_ceiling_reports_degenerate_denominators():
    read = r8._fraction_of_ceiling(
        np.asarray([10.0, 10.0]),
        np.asarray([10.0, 10.0]),
        np.asarray([10.0, 10.0]),
        "proj",
        key="fixture_degenerate",
    )
    assert read["fraction_point"] is None
    assert read["fraction_ci"] == [None, None]
    assert read["n_degenerate_draws"] == 1000
