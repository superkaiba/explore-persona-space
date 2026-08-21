"""Pins for the issue #2378 G1 digest composer's per-cell summary keying (r10).

The P1 pilot's G1(a) gate tripped with net_kept_per_attempt = 0.0000 for BOTH
families (epm:progress v63 diagnosis): the sega/segb per-cell summary writers
in scripts/issue2378_gen.py carried the cell only in the FILENAME
(summary_<cell>_w<k>_s<j>.json) with no 'cell' key in the payload, so
``_sum_stage_summaries`` fell back to keying every summary by the stage dir
name ('sega'/'segb'), which ``cm.CELL_FAMILY`` silently drops in
``_family_pool`` -> empty mining/segb family pools -> net multiplies to 0.0.
The self-test fixture hand-wrote 'cell' INTO the payload, so the probe stayed
green while production zeroed (#906 fixture-vs-writer contract drift class).

r10 fix, pinned here (fails pre-fix, passes post-fix):
- ``_sum_stage_summaries`` parses the cell from the filename when the payload
  lacks 'cell', accepting the capture ONLY on cm.CELL_FAMILY membership —
  this is what repairs the ALREADY-WRITTEN pod-side pilot summaries with
  zero GPU;
- stage-level summaries (user_sim's summary_w1_s0.json — no filename match)
  still take the stage-dir fallback LAST.

No GPU / no network: the aggregator is a pure JSON-directory reduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_common as cm  # noqa: E402
import issue2378_dispatch as d  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm.atomic_write_json(path, payload)


def test_filename_fallback_keys_no_cell_payload_by_cell(tmp_path: Path) -> None:
    """A REAL r<10 writer payload (no 'cell' key) pools under its CELL name
    parsed from summary_<cell>_w<k>_s<j>.json — never the stage dir name."""
    stage = tmp_path / "sega"
    _write(
        stage / "summary_storyq_astra_w1_s0.json",
        {"regime": "pilot", "counts": {"attempts": 100, "kept": 30, "cap_hit": 0}},
    )
    per_cell = d._sum_stage_summaries(stage, ("attempts", "kept", "cap_hit"))
    assert set(per_cell) == {"storyq_astra"}, per_cell
    assert per_cell["storyq_astra"] == {"attempts": 100, "kept": 30, "cap_hit": 0}
    # And the family pool is NON-empty (the production zero's exact seam).
    fams = d._family_pool(per_cell, "kept", "attempts")
    assert fams == {"question": {"numerator": 30, "denominator": 100, "rate": 0.3}}, fams


def test_filename_fallback_sums_shards_with_payload_cell_precedence(tmp_path: Path) -> None:
    """Payload 'cell' (the r10 writer form) and filename-parsed shards of the
    SAME cell sum into one bucket; a payload key always wins over parsing."""
    stage = tmp_path / "segb"
    _write(
        stage / "summary_dialog_astra_w1_s0.json",
        {"counts": {"rows": 50, "kept": 45, "cap_hit_no_close": 2}},  # no 'cell' (r<10 shape)
    )
    _write(
        stage / "summary_dialog_astra_w1_s1.json",
        {"cell": "dialog_astra", "counts": {"rows": 10, "kept": 5, "cap_hit_no_close": 1}},
    )
    per_cell = d._sum_stage_summaries(stage, ("rows", "kept", "cap_hit_no_close"))
    assert set(per_cell) == {"dialog_astra"}, per_cell
    assert per_cell["dialog_astra"] == {"rows": 60, "kept": 50, "cap_hit_no_close": 3}


def test_stage_level_summary_still_takes_stage_dir_fallback(tmp_path: Path) -> None:
    """user_sim's summary_w<k>_s<j>.json (no cell in name or payload) keys by
    the stage dir name LAST — the pre-r10 single-summary contract, unchanged."""
    stage = tmp_path / "user_sim"
    _write(stage / "summary_w1_s0.json", {"counts": {"rows": 50, "kept": 47}})
    per_cell = d._sum_stage_summaries(stage, ("rows", "kept"))
    assert set(per_cell) == {"user_sim"}, per_cell
    assert per_cell["user_sim"] == {"rows": 50, "kept": 47}


def test_non_cell_filename_capture_falls_through_to_stage_name(tmp_path: Path) -> None:
    """A filename whose capture is NOT a cm.CELL_FAMILY member (a future
    stage-level summary that happens to match the pattern) never mints a
    bogus per-cell bucket — membership gates the filename fallback."""
    stage = tmp_path / "sega"
    _write(stage / "summary_not_a_cell_w1_s0.json", {"counts": {"attempts": 7, "kept": 3}})
    per_cell = d._sum_stage_summaries(stage, ("attempts", "kept"))
    assert set(per_cell) == {"sega"}, per_cell
    assert "not_a_cell" not in cm.CELL_FAMILY  # fixture premise
