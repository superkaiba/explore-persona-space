"""#2658 unit 10 — prospective-feasibility cause tags + frame-manifest ledger.

Pins (a) the mechanical per-cell CAUSE derivation (``cell_cause``: each branch
FIRING on synthetic inputs, incl. the vacuous-empty-set guard and the
barred-leak raise), (b) the committed frame manifest's exact 21-cell
not-estimable inventory with counts + causes, the explicit EMPTY form on the 7
clear rows, and the manifest-level ledger totals (132 / 111 / 21 with the
per-cause split), (c) the ledger-reconciliation + per-kind field validation
guards FIRING on tampered bodies, and (d) pilot-pin content invariance across
the unit-10 re-freeze (629 items, frozen content sha).

Synthetic tests run everywhere; on-disk pins follow the sibling convention
(skip in a checkout that has not run the build — e.g. a sparse worktree
without eval_results/).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2658_frames as F  # noqa: E402

# ---------------------------------------------------------------------------
# Ground truth (derived 2026-09-02 from the committed per_cell_test_eligible
# against PRODUCTION_TEST_PROMPTS_PER_CELL_FLOOR = 15; two causes established
# by a probe through the production build_row — see task #2658 unit 10).
# ---------------------------------------------------------------------------
FLOOR = 15
CLEAR_ROWS = {
    "evil": 24,
    "sycophancy": 174,
    "refusal": 21,
    "harmful_compliance": 22,
    "correctness_math": 95,
    "correctness_mmlu_pro": 51,
    "correctness_code": 29,
}  # row -> min per-cell test-eligible count (all >= FLOOR)

BEHAVIOR_BANDS = ("direct", "indirect", "ambiguous")


def _cells(frame: str, n: int, cause: str) -> list[dict]:
    return [
        {"cell": f"{frame}|{band}", "n_test_eligible": n, "cause": cause} for band in BEHAVIOR_BANDS
    ]


EXPECTED_NOT_ESTIMABLE = {
    "hallucination": _cells("fact_questions", 5, "bank-too-small")
    + _cells("wang44_probes", 14, "bank-too-small"),
    "assistantness": _cells("fact_questions", 8, "bank-too-small"),
    "casualness": _cells("writing_style_neutral", 0, "extraction-barred")
    + _cells("fact_questions", 9, "bank-too-small"),
    "impoliteness": _cells("impolite_neutral", 0, "extraction-barred")
    + _cells("fact_questions", 9, "bank-too-small"),
}
EXPECTED_LEDGER_TOTALS = {
    "floor": 15,
    "n_cells": 132,
    "n_cells_estimable": 111,
    "n_cells_not_estimable": 21,
    "by_cause": {"bank-too-small": 15, "extraction-barred": 6},
}
# Pilot pins deliberately re-frozen in the group-D r3 round (6 of 629 ids
# moved across the r2+r3 manifest re-freezes; 0 retained ids changed record).
PIN_N_ITEMS = 629
PIN_ITEMS_SHA256 = "4be7d26c887ac2945bd06cad2588668f524baecc914a7a9004577c5f7cec7959"


def _load_disk(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} not built in this checkout")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# cell_cause: every branch FIRES on synthetic inputs.
# ---------------------------------------------------------------------------
def test_cell_cause_extraction_barred_fires():
    assert F.cell_cause(0, {"a", "b"}, {"a", "b", "c"}) == "extraction-barred"


def test_cell_cause_bank_too_small_fires():
    assert F.cell_cause(5, {"a", "b"}, {"a"}) == "bank-too-small"
    assert F.cell_cause(14, {"a"}, set()) == "bank-too-small"


def test_cell_cause_split_starved_fires():
    # zero test-eligible with a NON-barred superfamily present: neither of the
    # two diagnosed causes — the third tag, not a silent fold-in.
    assert F.cell_cause(0, {"a", "b"}, {"a"}) == "split-starved"


def test_cell_cause_empty_contributing_set_not_vacuously_barred():
    # set() <= barred is vacuously True; the guard must route to split-starved.
    assert F.cell_cause(0, set(), {"a"}) == "split-starved"
    assert F.cell_cause(0, frozenset(), frozenset()) == "split-starved"


def test_cell_cause_barred_leak_raises():
    # all-barred cell with nonzero test-eligible = barred items leaked to test.
    with pytest.raises(F.BarredTopUpError, match="leaked into test"):
        F.cell_cause(3, {"a"}, {"a"})


# ---------------------------------------------------------------------------
# Ledger + validation guards FIRE on tampered bodies.
# ---------------------------------------------------------------------------
def test_validate_unknown_manifest_kind_raises():
    with pytest.raises(F.FrameManifestError, match="unknown manifest_kind"):
        F.validate_manifest({"manifest_kind": "bogus"})


def test_frame_manifest_missing_ledger_raises():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    del body["prospective_not_estimable_ledger"]
    with pytest.raises(F.FrameManifestError, match="missing=\\['prospective_not_estimable_ledger'"):
        F.validate_manifest(body)


def test_split_manifest_rejects_ledger():
    body = _load_disk(F.SPLIT_MANIFEST_PATH)
    body["prospective_not_estimable_ledger"] = {"n_cells": 0}
    with pytest.raises(F.FrameManifestError, match="unknown=\\['prospective_not_estimable_ledger'"):
        F.validate_manifest(body)


def test_ledger_tamper_raises():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    body["prospective_not_estimable_ledger"]["n_cells_not_estimable"] += 1
    with pytest.raises(F.FrameManifestError, match="does not reconcile"):
        F.validate_manifest(body)


def test_row_missing_prospective_key_raises():
    # an ABSENT key must never read as "no failing cells".
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    del body["rows"][0]["prospective_not_estimable"]
    with pytest.raises(F.FrameManifestError, match="missing prospective-feasibility field"):
        F.validate_manifest(body)


# ---------------------------------------------------------------------------
# Committed-manifest pins (integration; skip in a build-less checkout).
# ---------------------------------------------------------------------------
def test_committed_frame_manifest_validates_and_is_immutable():
    frame = _load_disk(F.FRAME_MANIFEST_PATH)
    split = _load_disk(F.SPLIT_MANIFEST_PATH)
    for body in (frame, split):
        F.validate_manifest(body)
        F.assert_manifest_immutable(body)


def test_not_estimable_inventory_exact():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    rows = {r["row"]: r for r in body["rows"]}
    got = {
        row: rr["prospective_not_estimable"]
        for row, rr in rows.items()
        if rr["prospective_not_estimable"]
    }
    assert got == EXPECTED_NOT_ESTIMABLE
    n_cells_flagged = sum(len(v) for v in EXPECTED_NOT_ESTIMABLE.values())
    assert n_cells_flagged == 21


def test_below_gate_strings_match_prospective_records():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    for rr in body["rows"]:
        strings = rr["below_production_gate_cells"]
        recs = rr["prospective_not_estimable"]
        assert [f"{r['cell']}:{r['n_test_eligible']}" for r in recs] == strings
        assert rr["n_cells"] - rr["n_cells_estimable"] == len(recs)
        # every recorded count is really below the floor, and matches the
        # persisted per-cell table (absent key == 0 by construction).
        for r in recs:
            assert r["n_test_eligible"] < FLOOR
            assert rr["per_cell_test_eligible"].get(r["cell"], 0) == r["n_test_eligible"]


def test_clear_rows_carry_explicit_empty_form():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    rows = {r["row"]: r for r in body["rows"]}
    for row, expected_min in CLEAR_ROWS.items():
        rr = rows[row]
        # the key is PRESENT and empty — never a missing key.
        assert "prospective_not_estimable" in rr
        assert rr["prospective_not_estimable"] == []
        assert rr["below_production_gate_cells"] == []
        assert rr["n_cells"] == 12
        assert rr["n_cells_estimable"] == rr["n_cells"]
        assert min(rr["per_cell_test_eligible"].values()) == expected_min
        assert expected_min >= FLOOR


def test_split_starved_currently_unreached():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    causes = [rec["cause"] for rr in body["rows"] for rec in rr["prospective_not_estimable"]]
    assert "split-starved" not in causes
    assert set(causes) == {"bank-too-small", "extraction-barred"}


def test_ledger_totals_exact():
    body = _load_disk(F.FRAME_MANIFEST_PATH)
    ledger = body["prospective_not_estimable_ledger"]
    for k, v in EXPECTED_LEDGER_TOTALS.items():
        assert ledger[k] == v, (k, ledger[k], v)
    per_row = ledger["per_row"]
    assert set(per_row) == set(F.C.ROW_IDS)
    for row in CLEAR_ROWS:
        assert per_row[row] == {"n_cells": 12, "n_cells_estimable": 12}
    assert per_row["hallucination"] == {"n_cells": 12, "n_cells_estimable": 6}
    assert per_row["assistantness"] == {"n_cells": 12, "n_cells_estimable": 9}
    assert per_row["casualness"] == {"n_cells": 12, "n_cells_estimable": 6}
    assert per_row["impoliteness"] == {"n_cells": 12, "n_cells_estimable": 6}


def test_pilot_pins_invariant_across_refreeze():
    # Pins the committed prompt_pins.json ARTIFACT bytes. The table was
    # deliberately re-frozen in the group-D r3 round (the r2 + r3 manifest
    # re-freezes moved 2 + 4 of 629 pilot ids); membership consistency vs the
    # manifest is the SEPARATE test below — this one certifies the bytes.
    pin_path = F.OUT_DIR / "prompt_pins.json"
    pins = _load_disk(pin_path)
    assert pins["n_items"] == PIN_N_ITEMS
    got = hashlib.sha256(
        json.dumps(pins["items"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert got == PIN_ITEMS_SHA256


def _manifest_pilot_ids(body: dict) -> set[str]:
    ids: set[str] = set()
    for rr in body["rows"]:
        for cell_ids in rr["pilot_selection"]["per_cell_item_ids"].values():
            ids.update(cell_ids)
    return ids


def test_prompt_pins_membership_matches_manifest_pilot_selection():
    # Group-D r3 (re-review MAJOR): a stale pin table wedges LOUDLY at resolve
    # time (verify_against_pins raises on unpinned ids before engine spend),
    # but the SUBSET direction — pins a strict superset of the selection —
    # would be silent, and the suite was green against a stale table. Both
    # directions asserted here against the COMMITTED artifacts.
    manifest = _load_disk(F.FRAME_MANIFEST_PATH)
    pins = _load_disk(F.OUT_DIR / "prompt_pins.json")
    selection = _manifest_pilot_ids(manifest)
    pinned = set(pins["items"])
    missing = sorted(selection - pinned)  # would wedge resolution
    stale = sorted(pinned - selection)  # would be SILENT without this test
    assert not missing, f"{len(missing)} selected pilot ids unpinned: {missing[:4]}"
    assert not stale, f"{len(stale)} pinned ids no longer selected: {stale[:4]}"
    assert pins["n_items"] == len(selection)


def test_evidence_packets_cover_judge_bearing_pilot_selection():
    # Every evidence-bearing row's selected pilot item must be in the frozen
    # evidence store — as a packet or a DOCUMENTED exclusion, never absent.
    manifest = _load_disk(F.FRAME_MANIFEST_PATH)
    store = _load_disk(F.OUT_DIR / "evidence_packets.json")
    evidence_rows = {r for r in F.C.ROW_IDS if F.C.CONSTRUCTS[r].uses_evidence_packet}
    assert evidence_rows, "no evidence-bearing rows registered — fixture drift"
    covered = set(store["items"])
    excluded = {e["item_id"] for e in store["exclusions"]}
    unaccounted = []
    for rr in manifest["rows"]:
        if rr["row"] not in evidence_rows:
            continue
        for cell_ids in rr["pilot_selection"]["per_cell_item_ids"].values():
            unaccounted += [i for i in cell_ids if i not in covered and i not in excluded]
    assert not unaccounted, (
        f"{len(unaccounted)} judge-bearing pilot ids have neither an evidence "
        f"packet nor a documented exclusion: {sorted(unaccounted)[:4]}"
    )
