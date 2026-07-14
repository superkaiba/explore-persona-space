"""Tests for the battery_scope_caveat block + sweep skip-predicate helpers (issue1092_figures).

Concern battery-rows-in-fit-arms-banked-p6 (raised round 13): plan v5 section
4.1 step 6 registered the #594 battery rows EVAL-ONLY in both fit arms, but the
engine's fit-arm-A stratum filter excludes only {"trait_stratum",
"battery_eval_only"} while the realized corpus labels the stratum "battery" —
so battery rows entered TRAINING in both banked fit arms. These tests execute
the REAL `_battery_scope_caveat` body (no mocks) on tiny synthetic manifests
mirroring the incident arithmetic (fitA 19,708 = 21,193 - 1,485).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from issue1092_figures import (
    BATTERY_CONCERN_ID,
    _battery_scope_caveat,
    _merge_output_fingerprint,
)


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _tiny_manifest_rows() -> list[dict]:
    """12 rows: 7 dense_core, 2 trait_stratum, 3 battery (is_eval_only=true)."""
    rows = []
    for i in range(7):
        rows.append({"row_id": f"r{i}", "stratum": "dense_core", "is_eval_only": False})
    for i in range(2):
        rows.append({"row_id": f"t{i}", "stratum": "trait_stratum", "is_eval_only": False})
    for i in range(3):
        rows.append({"row_id": f"b{i}", "stratum": "battery", "is_eval_only": True})
    return rows


def _unit(cell: str, fit_arm: str, n_rows: int) -> dict:
    return {"cell": cell, "fit_arm": fit_arm, "n_rows": n_rows}


def test_engine_rule_cell_confirms_battery_in_training(tmp_path):
    """fitA = total - trait_stratum (battery NOT excluded) => confirmed True."""
    manifest = _write_manifest(tmp_path / "manifest.jsonl", _tiny_manifest_rows())
    kept = [_unit("cell_x", "A", 10), _unit("cell_x", "B", 12)]
    caveat = _battery_scope_caveat(kept, manifest)
    assert caveat["concern_id"] == BATTERY_CONCERN_ID
    assert caveat["corpus_manifest"]["status"] == "present"
    assert caveat["corpus_manifest"]["strata_counts"] == {
        "battery": 3,
        "dense_core": 7,
        "trait_stratum": 2,
    }
    assert caveat["corpus_manifest"]["battery_is_eval_only_flag_counts"] == {"True": 3}
    entry = caveat["n_rows_arithmetic"]["per_cell"]["cell_x"]
    assert entry["status"] == "checked"
    assert entry["scope_matched"] == "full_corpus"
    assert entry["expected_fitA_engine_rule"] == 10  # 12 - 2 trait
    assert entry["expected_fitA_registered_rule"] == 7  # 12 - 2 trait - 3 battery
    assert entry["battery_rows_in_training"] is True
    # Recoverability record: banked fit blocks carry only aggregates.
    rec = caveat["recoverability"]
    assert rec["per_fold_predictions_persisted"] is False
    assert rec["per_fold_coefficients_persisted"] is False
    assert rec["per_row_heldout_predictions_or_residuals_persisted"] is False
    assert rec["row_indexing_in_checkpoints"] is False
    # The block must round-trip through JSON (it rides every family payload).
    json.dumps(caveat)


def test_registered_rule_cell_reads_false(tmp_path):
    """A hypothetical refit that DID exclude battery reads False, not True."""
    manifest = _write_manifest(tmp_path / "manifest.jsonl", _tiny_manifest_rows())
    kept = [_unit("cell_x", "A", 7), _unit("cell_x", "B", 12)]
    entry = _battery_scope_caveat(kept, manifest)["n_rows_arithmetic"]["per_cell"]["cell_x"]
    assert entry["status"] == "checked"
    assert entry["battery_rows_in_training"] is False


def test_truncated_fitB_is_not_checkable(tmp_path):
    """fitB not matching any scope total (n0 truncation) fails soft, never wrong."""
    manifest = _write_manifest(tmp_path / "manifest.jsonl", _tiny_manifest_rows())
    kept = [_unit("cell_x", "A", 9), _unit("cell_x", "B", 11)]
    entry = _battery_scope_caveat(kept, manifest)["n_rows_arithmetic"]["per_cell"]["cell_x"]
    assert entry["status"] == "not_checkable"
    assert "battery_rows_in_training" not in entry


def test_absent_manifest_fails_soft(tmp_path):
    caveat = _battery_scope_caveat([_unit("cell_x", "A", 10)], tmp_path / "missing.jsonl")
    assert caveat["corpus_manifest"]["status"] == "absent"
    assert caveat["n_rows_arithmetic"]["status"] == "not_checkable_manifest_absent"
    assert caveat["observed_n_rows_by_cell_fit_arm"] == {"cell_x": {"A": [10]}}
    json.dumps(caveat)


def test_merge_output_fingerprint_reads_field_or_none(tmp_path):
    present = tmp_path / "behavior_B1_B2.json"
    present.write_text(json.dumps({"merge_fingerprint": "abc123"}))
    assert _merge_output_fingerprint(present) == "abc123"
    assert _merge_output_fingerprint(tmp_path / "absent.json") is None
