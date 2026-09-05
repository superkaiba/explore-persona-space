"""Contract checks for the issue #2254 partial sensitivity report."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    REPO_ROOT
    / "eval_results"
    / "issue_2254"
    / "revmap_dose_patch"
    / "exploratory_sensitivity"
    / "codex_subagent_v1"
    / "report"
    / "eligible_report_summary.json"
)


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_partial_report_preserves_frozen_failure() -> None:
    report = _summary()
    eligibility = report["eligibility"]
    assert eligibility["overall_completeness_pass"] is False
    assert eligibility["below_floor_cells"] == ["evil__cl"]
    assert eligibility["completeness_floor"] == 0.95
    assert eligibility["withheld"]["floor_lowered"] is False
    assert eligibility["withheld"]["refusals_coerced"] is False


def test_evil_patch_fractions_are_withheld_and_sycophancy_is_complete() -> None:
    report = _summary()
    evil = report["trait"]["evil_patch_descriptive_only"]
    sycophancy = report["trait"]["sycophancy_patch_fraction_of_ceiling"]
    assert len(evil) == 6
    assert all(row["fraction_of_ceiling"] is None for row in evil.values())
    assert len(sycophancy) == 6
    assert all(
        row["fraction_of_ceiling"]["fraction_point"] is not None for row in sycophancy.values()
    )


def test_coherence_and_cjk_are_separate_complete_reads() -> None:
    report = _summary()
    assert report["coherence"]["cjk_is_part_of_metric"] is False
    assert report["cjk"]["subagent_scored"] is False
    assert report["cjk"]["separate_from_coherence"] is True
    assert len(report["coherence"]["round8"]) == 16
    assert len(report["cjk"]["round8"]) == 16
    assert report["coherence"]["groups"]["steer"]["n_items"] == 800
    assert report["coherence"]["groups"]["patch"]["n_items"] == 2400
    overall = report["cjk"]["groups"]["round8_overall"]
    assert overall["n_intrusions"] == 323
    assert overall["n_completions"] == 3200
