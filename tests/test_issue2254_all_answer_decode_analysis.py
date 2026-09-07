"""Focused tests for refusal-aware grading and split-sample dose selection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.issue2254_all_answer_decode_analysis as analysis
import scripts.issue2254_all_answer_decode_sweep as sweep


def _item(cell_id: str, index: int, token_count: int) -> analysis.AnalysisItem:
    source_id = f"{cell_id}|item{index}"
    return analysis.AnalysisItem(
        source_item_id=source_id,
        opaque_id=f"i{index:020d}{cell_id[-1]}",
        cell_id=cell_id,
        behavior="evil",
        route="decodeonly",
        dose=1 / 64,
        qi=index,
        effective_seed=sweep.SEED_BASE + index,
        question="Fixture question?",
        answer=f"Fixture answer {index}.",
        token_count=token_count,
        answer_sha256=analysis._sha256_text(f"Fixture answer {index}."),
    )


def _synthetic_integrity_arrays() -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for behavior in sweep.BEHAVIORS:
        context = sweep.CellSpec(behavior, "context", sweep.CONTEXT_DOSE[behavior])
        context_score = 50.0 if behavior == "evil" else 95.0
        context_pass = 0.50 if behavior == "evil" else 0.95
        arrays[context.cell_id] = {
            "score": np.full((20, 6), context_score),
            "passing": np.full((20, 6), context_pass),
        }
        for dose in sweep.DOSES:
            cell = sweep.CellSpec(behavior, "decodeonly", dose)
            arrays[cell.cell_id] = {
                "score": np.full((20, 6), context_score + 10.0),
                "passing": np.full((20, 6), context_pass),
            }

    # Evil: c=1/64 has normalized max distance .9; c=1/32 has .98.
    evil_context = sweep.CellSpec("evil", "context", sweep.CONTEXT_DOSE["evil"])
    evil_score = arrays[evil_context.cell_id]["score"][0, 0]
    evil_pass = arrays[evil_context.cell_id]["passing"][0, 0]
    arrays[sweep.CellSpec("evil", "decodeonly", 1 / 64).cell_id] = {
        "score": np.full((20, 6), evil_score + 4.5),
        "passing": np.full((20, 6), evil_pass),
    }
    arrays[sweep.CellSpec("evil", "decodeonly", 1 / 32).cell_id] = {
        "score": np.full((20, 6), evil_score),
        "passing": np.full((20, 6), evil_pass + 0.049),
    }

    # Sycophancy: exact quality ties at the smallest and largest doses must
    # resolve toward the larger dose.
    syc_context = sweep.CellSpec("sycophancy", "context", sweep.CONTEXT_DOSE["sycophancy"])
    for dose in (1 / 64, 4.0):
        arrays[sweep.CellSpec("sycophancy", "decodeonly", dose).cell_id] = {
            "score": arrays[syc_context.cell_id]["score"].copy(),
            "passing": arrays[syc_context.cell_id]["passing"].copy(),
        }
    return arrays


def test_refusal_aware_summary_preserves_itt_and_numeric_sensitivity() -> None:
    summary = analysis.summarize_trait_repeats([90, "REFUSAL", 70, "REFUSAL", 80])

    assert summary == {
        "itt_mean": 48.0,
        "numeric_mean": 80.0,
        "refusal_fraction": 0.4,
        "structured_completeness": 1.0,
        "numeric_draw_completeness": 0.6,
    }


def test_all_refusals_remain_structured_complete_with_zero_itt() -> None:
    summary = analysis.summarize_trait_repeats(["REFUSAL"] * analysis.N_PASSES)

    assert summary["itt_mean"] == 0.0
    assert summary["numeric_mean"] is None
    assert summary["refusal_fraction"] == 1.0
    assert summary["structured_completeness"] == 1.0
    assert summary["numeric_draw_completeness"] == 0.0


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 101],
        [1, 2, 3, 4, -1],
        [1, 2, 3, 4, 5.0],
        [1, 2, 3, 4, True],
    ],
)
def test_trait_summary_rejects_incomplete_or_noncanonical_repeats(values) -> None:
    with pytest.raises(analysis.AnalysisError, match="invalid/incomplete"):
        analysis.summarize_trait_repeats(values)


def test_selection_rows_exclude_zero_and_use_normalized_max_distance() -> None:
    arrays = _synthetic_integrity_arrays()

    _, rows = analysis._selection_rows("evil", arrays, analysis.SELECTION_QUESTIONS)
    by_dose = {row["dose"]: row for row in rows}

    assert len(rows) == 9
    assert 0.0 not in by_dose
    assert by_dose[1 / 64]["normalized_max_distance"] == pytest.approx(0.9)
    assert by_dose[1 / 32]["normalized_max_distance"] == pytest.approx(0.98)
    assert by_dose[1 / 64]["eligible"] is True
    assert by_dose[1 / 32]["eligible"] is True


def test_selection_prefers_normalized_distance_then_larger_dose(monkeypatch, tmp_path) -> None:
    arrays = _synthetic_integrity_arrays()
    instrument = {"rubric_instrument_fp": {"coherence": "fixture-fp"}}
    monkeypatch.setattr(analysis, "analysis_root", lambda unused: tmp_path)
    monkeypatch.setattr(
        analysis,
        "_load_staged",
        lambda args: ([], instrument, {}),
    )
    monkeypatch.setattr(analysis, "_production_jobs", lambda *args: [])
    monkeypatch.setattr(analysis, "_collect_outcomes", lambda *args: {})
    monkeypatch.setattr(analysis, "_integrity_cell_arrays", lambda *args: arrays)

    analysis.phase_freeze_selection(SimpleNamespace(out_root=tmp_path))

    frozen = json.loads(
        (tmp_path / "selection" / "quality_only_selection.json").read_text(encoding="utf-8")
    )
    assert frozen["zero_dose_selectable"] is False
    assert frozen["behaviors"]["evil"]["selected_dose"] == pytest.approx(1 / 64)
    assert frozen["behaviors"]["sycophancy"]["selected_dose"] == 4.0


def test_bootstrap_difference_is_exact_for_constant_paired_effect(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "N_BOOTSTRAP", 2_000)
    answer = np.arange(10, dtype=float) + 3.0
    context = np.arange(10, dtype=float)
    indices = analysis._bootstrap_indices("constant-effect", 10)

    result = analysis._difference(answer, context, indices, analysis.TRAIT_CI_LEVEL)

    assert result["estimate"] == pytest.approx(3.0)
    assert result["ci95"] == pytest.approx([3.0, 3.0])
    assert result["simultaneous_ci"] == pytest.approx([3.0, 3.0])


def test_pilot_selection_includes_each_cells_longest_item_deterministically() -> None:
    items = []
    for cell_id, counts in (("cell-a", [1, 9, 3, 5, 2]), ("cell-b", [7, 2, 4, 1, 6])):
        items.extend(_item(cell_id, index, count) for index, count in enumerate(counts))

    selected = analysis._actual_pilot_items(items)

    assert selected == analysis._actual_pilot_items(items)
    assert len(selected) == 6
    assert len({item.source_item_id for item in selected}) == 6
    for cell_id in ("cell-a", "cell-b"):
        chosen = [item for item in selected if item.cell_id == cell_id]
        pool = [item for item in items if item.cell_id == cell_id]
        assert len(chosen) == 3
        assert max(pool, key=lambda item: (item.token_count, item.source_item_id)) in chosen


@pytest.mark.parametrize("behavior", sweep.BEHAVIORS)
def test_trait_pilot_anchors_cover_positive_negative_refusal_and_degenerate(
    behavior: str,
) -> None:
    anchors = analysis._trait_anchor_items(behavior)

    assert len(anchors) == 4
    assert {item.route for item in anchors} == {
        "positive",
        "negative",
        "refusal",
        "degenerate",
    }
    assert all(item.cell_id == "pilot_anchor" for item in anchors)
    assert all(item.behavior == behavior for item in anchors)
    assert len({item.opaque_id for item in anchors}) == 4
    refusal = next(item for item in anchors if item.route == "refusal")
    assert "cannot" in refusal.answer.lower()


def test_trait_verdict_is_not_confirmatory_without_quality_equivalence() -> None:
    verdict, descriptive = analysis._confirmatory_trait_verdict([2.0, 5.0], False)
    assert verdict == "not_tested_quality_not_confirmed"
    assert descriptive == "answer_exceeds_context"

    verdict, descriptive = analysis._confirmatory_trait_verdict([-5.0, -2.0], True)
    assert verdict == "context_exceeds_answer"
    assert descriptive == "context_exceeds_answer"


def test_trait_verdict_uses_simultaneous_interval_semantics() -> None:
    verdict, descriptive = analysis._confirmatory_trait_verdict([-1.0, 2.0], True)
    assert verdict == "inconclusive"
    assert descriptive == "inconclusive"
