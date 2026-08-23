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


# ---------------------------------------------------------------------------
# v7 amended-G1 pins (plan Amendment record B, epm:progress v70 clause 2):
# question-only gate, floor-funding PASS line derived from the constants,
# wave-1 sizing formula unchanged, dialogue pooled for the record only.
# ---------------------------------------------------------------------------


def _pilot_fixture(root: Path, mining_kept: int) -> tuple[Path, Path]:
    """Minimal r2-shaped pilot raw+ledger fixture (storyq_astra + archival
    dialog_astra; mirrors dispatch phase_probe's _mk_pilot_fixture)."""
    raw, ledger = root / "raw", root / "ledger"
    for cell in ("storyq_astra", "dialog_astra"):
        _write(
            raw / "sega" / f"summary_{cell}_w1_s0.json",
            {"cell": cell, "counts": {"attempts": 100, "kept": mining_kept, "cap_hit": 0}},
        )
        _write(
            raw / "segb" / f"summary_{cell}_w1_s0.json",
            {"cell": cell, "counts": {"rows": 50, "kept": 45, "cap_hit_no_close": 2}},
        )
        _write(
            ledger / "pilot" / "kept" / f"{cell}.json",
            {
                "cell": cell,
                "family": cm.CELL_FAMILY[cell],
                "n_items": 60,
                "n_admitted": 40,
                "admitted": [],
            },
        )
    _write(raw / "user_sim" / "summary_w1_s0.json", {"counts": {"rows": 50, "kept": 47}})
    _write(ledger / "judge" / "pilot_admission_sync.json", {"verdict": "PASS", "tally": {}})
    _write(
        ledger / "pilot" / "layer_sweep.json",
        {"selected_layer": 40, "gate_g1c": {"threshold": 0.05, "max_r2": 0.31, "passes": True}},
    )
    return raw, ledger


def test_amended_g1_trip_line_derived_from_constants() -> None:
    """The v7 rate line is FLOOR_KEPT / SEGA_ATTEMPTS_CAP (~0.21667) — derived,
    never a hardcoded 0.2167, and strictly below the pre-v7 0.25 line."""
    assert d.G1_NET_RATE_MIN == cm.FLOOR_KEPT / d.SEGA_ATTEMPTS_CAP
    assert 0.216 < d.G1_NET_RATE_MIN < 0.217
    assert d.G1_NET_RATE_MIN < 0.25


def test_amended_g1_passes_in_floor_funding_band(tmp_path: Path) -> None:
    """DISCRIMINATING fixture: net = 0.38 * (40/60) * 0.9 = 0.228 sits BELOW
    the old 0.25 line and ABOVE the amended floor-funding line — the v7 gate
    PASSes it (projected 6,840 kept at the 30k cap >= 6,500 floor)."""
    raw, ledger = _pilot_fixture(tmp_path, mining_kept=38)
    digest = d.compose_pilot_digest(raw, ledger, {}, pilot_round=2, attempts_per_cell=300)
    fam = digest["families"]["question"]
    assert d.G1_NET_RATE_MIN < fam["net_kept_per_attempt"] < 0.25
    assert fam["pass"] is True
    assert fam["floor_kept"] == cm.FLOOR_KEPT
    assert fam["attempts_cap"] == d.SEGA_ATTEMPTS_CAP
    assert fam["projected_kept_at_cap"] == fam["net_kept_per_attempt"] * d.SEGA_ATTEMPTS_CAP
    assert digest["verdict"] == "PASS", digest["fail_reasons"]


def test_amended_g1_gate_is_question_only_dialogue_pooled_for_record(tmp_path: Path) -> None:
    """The GATE iterates ACTIVE families only (question at v7); dialogue rows
    still pool into per_stage/per_cell for the archival record."""
    raw, ledger = _pilot_fixture(tmp_path, mining_kept=38)
    digest = d.compose_pilot_digest(raw, ledger, {}, pilot_round=2, attempts_per_cell=300)
    assert set(digest["families"]) == {"question"}
    assert set(digest["per_stage"]["mining"]) == {"question", "dialogue"}
    assert set(digest["per_cell"]["mining"]) == {"storyq_astra", "dialog_astra"}


def test_amended_g1_below_floor_fails_with_floor_reason(tmp_path: Path) -> None:
    """net = 0.2 * (40/60) * 0.9 = 0.12 -> projected 3,600 < 6,500: FAIL, and
    the G1(a) reason names the floor arithmetic."""
    raw, ledger = _pilot_fixture(tmp_path, mining_kept=20)
    digest = d.compose_pilot_digest(raw, ledger, {}, pilot_round=2, attempts_per_cell=300)
    assert digest["verdict"] == "FAIL"
    assert any(r.startswith("G1(a) question") and "floor" in r for r in digest["fail_reasons"])


def test_wave1_sizing_formula_unchanged_at_v7(tmp_path: Path) -> None:
    """min(cap, ceil(TARGET * SLACK / net)) — the amendment changes ONLY the
    PASS predicate, never the sizing formula."""
    import math

    raw, ledger = _pilot_fixture(tmp_path, mining_kept=38)
    digest = d.compose_pilot_digest(raw, ledger, {}, pilot_round=2, attempts_per_cell=300)
    fam = digest["families"]["question"]
    expect = min(
        d.SEGA_ATTEMPTS_CAP,
        math.ceil(cm.STORY_TARGET_KEPT * d.WAVE1_SLACK / fam["net_kept_per_attempt"]),
    )
    assert fam["wave1_attempts_per_cell"] == expect


def test_walls_merge_note_recorded_only_when_passed(tmp_path: Path) -> None:
    """P1R walls-merge provenance: the note lands in the digest verbatim when
    supplied; absent otherwise (the 98565a9d7d hand-merge, now in code)."""
    raw, ledger = _pilot_fixture(tmp_path, mining_kept=38)
    d0 = d.compose_pilot_digest(raw, ledger, {}, pilot_round=2, attempts_per_cell=300)
    assert "walls_merge_note" not in d0
    d1 = d.compose_pilot_digest(
        raw,
        ledger,
        {"p1.capture_pilot": 10.0},
        pilot_round=2,
        attempts_per_cell=300,
        walls_merge_note="merged from committed digest",
    )
    assert d1["walls_merge_note"] == "merged from committed digest"
    assert d1["measured_walls_s"] == {"p1.capture_pilot": 10.0}


def test_active_panel_is_nine_cells() -> None:
    """Amendment-A pin (r12 reconcile rec 3): 2 base + 5 story-Q + 2 user = 9
    active cells; dialogue cells stay defined but INERT (out of every active
    enumeration); the gate iterates the question family only."""
    assert len(cm.ALL_CELLS) == 9
    assert set(cm.DIALOG_CELLS).isdisjoint(cm.ALL_CELLS)
    assert cm.ACTIVE_FAMILIES == ("question",)
    assert cm.STORY_CELLS == cm.STORY_Q_CELLS


def test_pilot_capture_out_root_round_scoping(tmp_path: Path) -> None:
    """Plan §4.7 out-root fix pin (r12 reconcile rec 4): round 1 keeps the
    stable path byte-identically; round >= 2 gets a DISJOINT sibling (never
    nested), so a fresh round cannot land in a prior round's StageLedger."""
    stable = tmp_path / "activations_pilot"
    assert cm.pilot_capture_out_root(1, stable) == stable
    r2 = cm.pilot_capture_out_root(2, stable)
    assert r2 == tmp_path / "activations_pilot_r2"
    assert r2 != stable
    assert stable not in r2.parents and r2 not in stable.parents
