from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2588_minimal_longcap_summary.py"
SPEC = importlib.util.spec_from_file_location("issue2588_minimal_longcap_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_summarize_stage_uses_post_regeneration_hits() -> None:
    reports = [
        {
            "stage": "a",
            "n": 4,
            "cap": 8,
            "cap_hits": 3,
            "post_regen_cap_hits": 1,
            "regen_ran": True,
        },
        {"stage": "b", "n": 3, "cap": 16, "cap_hits": 1, "regen_ran": False},
    ]
    drops = [
        {"drops": [{"row_id": "a0", "reason": "truncated_no_close"}]},
        {"drops": [{"row_id": "b1", "reason": "close_count_2"}]},
    ]

    out = MOD.summarize_stage(reports, drops)

    assert out["expected"] == 7
    assert out["retained"] == 5
    assert out["retained_fraction"] == pytest.approx(5 / 7)
    assert out["cap_hits_pre_regen"] == 4
    assert out["cap_hits_effective"] == 2
    assert out["drop_reasons"] == {"close_count_2": 1, "truncated_no_close": 1}


def test_summarize_stage_rejects_overlapping_row_ids() -> None:
    reports = [
        {"stage": "a", "n": 2, "cap": 8, "cap_hits": 0},
        {"stage": "b", "n": 2, "cap": 8, "cap_hits": 0},
    ]
    drops = [
        {"drops": [{"row_id": "same", "reason": "x"}]},
        {"drops": [{"row_id": "same", "reason": "y"}]},
    ]

    with pytest.raises(ValueError, match="overlap"):
        MOD.summarize_stage(reports, drops)
