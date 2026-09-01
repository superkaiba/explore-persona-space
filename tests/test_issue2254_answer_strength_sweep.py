from __future__ import annotations

import json

import pytest

import scripts.issue2254_answer_strength_sweep as sweep
import scripts.issue2254_probe_context_followup as base


def test_screen_and_confirmation_splits_are_disjoint_and_exhaustive():
    assert set(sweep.SCREEN_QUESTION_INDICES).isdisjoint(sweep.CONFIRM_QUESTION_INDICES)
    assert set(sweep.SCREEN_QUESTION_INDICES) | set(sweep.CONFIRM_QUESTION_INDICES) == set(
        range(20)
    )
    assert set(sweep.SCREEN_SEEDS).isdisjoint(sweep.CONFIRM_SEEDS)


def test_screen_grid_has_expected_cells_and_doses():
    cells = sweep.build_screen_cells()
    assert len(cells) == 4 * (1 + 2 * 3 * 4)
    assert len({base.cell_id(cell) for cell in cells}) == len(cells)
    optimistic_probe_all = [
        cell
        for cell in cells
        if cell["behavior"] == "optimistic"
        and cell.get("method") == "probe"
        and cell.get("breadth") == "all"
    ]
    assert [cell["c"] for cell in optimistic_probe_all] == [0.25, 0.5, 1.0, 2.0]


def test_confirm_grid_deduplicates_random_cells_selected_by_both_methods(tmp_path):
    selected = sweep._signal_cell("optimistic", "diffmean", "single", 0.125)
    probe = sweep._signal_cell("optimistic", "probe", "single", 0.125)
    payload = {
        "traits": {
            "optimistic": {
                "methods": {
                    "diffmean": {"status": "selected", "cell": selected},
                    "probe": {"status": "selected", "cell": probe},
                }
            }
        }
    }
    selection_path = tmp_path / "screen/confirmation_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(payload))
    cells = sweep.build_confirm_cells(tmp_path, ["optimistic"], n_random=8)
    assert len(cells) == 1 + 2 + 8
    assert sum(cell["kind"] == "random" for cell in cells) == 8


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A short, ordinary answer.", False),
        (("loop token " * 80).strip(), True),
        (" ".join(f"word{index}" for index in range(100)), False),
    ],
)
def test_degeneracy_detector(text, expected):
    assert sweep._looks_degenerate(text) is expected


def test_quality_exclusions_include_each_frozen_programmatic_gate(monkeypatch):
    monkeypatch.setattr(
        sweep,
        "_quality_metrics",
        lambda _record: {
            "cap_hit_fraction": 0.03,
            "cjk_fraction": 0.21,
            "degenerate_fraction": 0.21,
            "coherence_rate_programmatic": 0.49,
        },
    )
    _metrics, reasons = sweep._quality_exclusions({})
    assert reasons == [
        "generation_cap_hits",
        "cjk_language_switching",
        "repetitive_or_degenerate_text",
        "programmatic_coherence",
    ]
