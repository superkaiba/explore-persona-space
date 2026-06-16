"""Issue #640 — the diagonal-target column map matches #545's cell_metadata.

``_diagonal_target_columns()`` (plan v6 §4.1) is the single-variable swap vs v3:
each row scored on its ON-TARGET diagonal column instead of the highest-|L|
off-diagonal leakage column. The map MUST be READ from
``eval_results/issue_545/cell_metadata.json`` (never hand-picked, §12 must-not-
change), cover the 7 judged-rate rows, and EXCLUDE the marker row (its diagonal
is log-prob-scale ``marker_slot_stats`` with judge None — it would crash the
judged ``_judge_completions`` loop if passed in, §4.2 part 2 / §4.3).

Cheap: the driver imports without torch/transformers; these read JSON only.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# The plan v6 §4.1 table — the authoritative contract the map must satisfy.
EXPECTED_DIAGONAL_MAP = {
    "bad_medical": "fam_expr_bad_medical",
    "risky_financial": "fam_expr_risky_financial",
    "extreme_sports": "fam_expr_extreme_sports",
    "taught_fact": "fact_expression",
    "reversed_fact": "fact_expression",
    "compliment_writing": "fam_expr_compliment",
    "wrong_claim_agreement": "sycophancy",
}


def _driver():
    return importlib.import_module("issue640_postfix_carrier")


def test_diagonal_map_matches_cell_metadata_table():
    """The resolved map equals the §4.1 table exactly (7 judged-rate rows)."""
    d = _driver()
    got = d._diagonal_target_columns()
    assert got == EXPECTED_DIAGONAL_MAP, (
        f"diagonal column map drifted from the plan v6 §4.1 contract.\n"
        f"got={got}\nexpected={EXPECTED_DIAGONAL_MAP}"
    )


def test_marker_excluded_from_diagonal_map():
    """marker is NOT in the judged-rate map (log-prob DV; would crash the judge)."""
    d = _driver()
    got = d._diagonal_target_columns()
    assert "marker" not in got, "marker must be excluded from the diagonal judged-rate map (§4.3)"
    assert "marker" not in d.PHASE2_ROWS_JUDGED_RATE, "marker must not be a judged-rate row"
    assert len(got) == 7, f"expected exactly 7 judged-rate diagonal rows, got {len(got)}"


def test_diagonal_map_is_read_from_cell_metadata_not_hardcoded():
    """Each resolved column equals the live cell_metadata diagonal_column field.

    Re-reads cell_metadata.json directly and cross-checks every row — pins the
    "READ, never hand-pick" contract (§12 must-not-change). If #545 ever
    re-publishes a different diagonal_column, this test (and the driver's HALT)
    surface it instead of silently scoring the stale column.
    """
    d = _driver()
    got = d._diagonal_target_columns()
    meta = json.loads((REPO / "eval_results/issue_545/cell_metadata.json").read_text())["cells"]
    for row, col in got.items():
        live = meta[f"{row}_primary_seed0"]["diagonal_column"]
        assert col == live, (
            f"{row}: map column {col!r} != live cell_metadata diagonal_column {live!r} — "
            "the map must track cell_metadata, not a hand-coded value."
        )


def test_marker_diagonal_reference_values():
    """The marker null/parity reference reads 5.7738 / 9.4478 nats (§4.3)."""
    d = _driver()
    ref = d._marker_diagonal_reference()
    assert set(ref) == {"0", "137"}, ref
    assert abs(ref["0"] - 5.773762626647949) < 1e-6, ref["0"]
    assert abs(ref["137"] - 9.447760429382324) < 1e-6, ref["137"]
