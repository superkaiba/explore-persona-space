"""Task #653 round-6 BLOCKER offpod-bootstrap-missing-deciding-ci-silent-skip.

CPU-only. The off-pod ``_refresh_ambiguity_flags`` re-classifies the per-cell
``deciding_ci`` ambiguity flag at the full 10k bootstrap depth. Before round 6 it
SILENTLY ``continue``d when a numeric-deciding cell had no entry in ``per_cell_ci``
— leaving that cell at the shallow on-pod ``n_boot=200`` flag while STILL stamping
``bootstrap_refreshed_n_boot=10000`` on the grid. A partial HF tensor pull could
therefore strand cells at the shallow CI depth under a 10k stamp.

These tests pin the converted behavior: a missing per-cell CI for ANY of the three
numeric deciding DVs (top_share_lambda / pr_lambda / rank_k_at_90) raises
``RuntimeError`` naming the cell + DV; an UNKNOWN deciding DV raises naming the DV;
the alignment-driven ``cos_top_to_rb`` branch is unaffected (no CI required); and a
fully-covered grid refreshes without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments import issue_653 as i653
from scripts.issue_653 import i653_postpod_bootstrap as ppb


def _write_verdict(tmp_path: Path, verdicts: list[dict]) -> Path:
    """Write a minimal cross_arm_verdict.json under out_root and return out_root."""
    out_root = tmp_path
    (out_root / "cross_arm_verdict.json").write_text(json.dumps({"verdicts": verdicts}))
    return out_root


def _ci(lo: float, hi: float) -> dict:
    return {"ci_low": lo, "ci_high": hi, "n_boot": i653.BOOTSTRAP_B}


def test_refresh_raises_on_missing_pr_lambda_ci(tmp_path):
    """A pr_lambda-deciding cell with NO per_cell_ci entry → RuntimeError naming the
    cell + the deciding_dv (NOT a silent skip + 10k stamp)."""
    out_root = _write_verdict(tmp_path, [{"cell_id": "cellA", "deciding_dv": "pr_lambda"}])
    with pytest.raises(RuntimeError) as exc:
        ppb._refresh_ambiguity_flags(out_root, {})
    msg = str(exc.value)
    assert "cellA" in msg
    assert "pr_lambda" in msg


def test_refresh_raises_on_missing_top_share_ci(tmp_path):
    """A top_share_lambda-deciding cell with NO per_cell_ci entry → RuntimeError."""
    out_root = _write_verdict(tmp_path, [{"cell_id": "cellB", "deciding_dv": "top_share_lambda"}])
    with pytest.raises(RuntimeError) as exc:
        ppb._refresh_ambiguity_flags(out_root, {})
    msg = str(exc.value)
    assert "cellB" in msg
    assert "top_share_lambda" in msg


def test_refresh_raises_on_missing_rank_k_ci(tmp_path):
    """A rank_k_at_90-deciding cell with NO per_cell_ci entry → RuntimeError."""
    out_root = _write_verdict(tmp_path, [{"cell_id": "cellC", "deciding_dv": "rank_k_at_90"}])
    with pytest.raises(RuntimeError) as exc:
        ppb._refresh_ambiguity_flags(out_root, {})
    msg = str(exc.value)
    assert "cellC" in msg
    assert "rank_k_at_90" in msg


def test_refresh_raises_on_unknown_deciding_dv(tmp_path):
    """An UNKNOWN deciding_dv raises loudly with the DV name (not an incidental
    KeyError on DV_THRESHOLDS[deciding_dv]). per_cell_ci is non-empty so the only
    fault is the unrecognized DV."""
    out_root = _write_verdict(tmp_path, [{"cell_id": "cellD", "deciding_dv": "some_unknown_dv"}])
    per_cell_ci = {"cellD": {"some_unknown_dv": _ci(0.1, 0.2)}}
    with pytest.raises(RuntimeError) as exc:
        ppb._refresh_ambiguity_flags(out_root, per_cell_ci)
    assert "some_unknown_dv" in str(exc.value)


def test_refresh_allows_cos_top_to_rb_without_ci(tmp_path):
    """A cos_top_to_rb-deciding cell needs NO per-cell CI (alignment-driven,
    explicit-unavailability). The refresh must NOT raise, and the on-pod
    deciding_ci_unavailable=True flag is preserved."""
    out_root = _write_verdict(
        tmp_path,
        [{"cell_id": "cellE", "deciding_dv": "cos_top_to_rb", "deciding_ci_unavailable": True}],
    )
    ppb._refresh_ambiguity_flags(out_root, {})  # empty per_cell_ci, must not raise
    grid = json.loads((out_root / "cross_arm_verdict.json").read_text())
    (vd,) = grid["verdicts"]
    assert vd["deciding_ci_unavailable"] is True
    # the cos cell is left untouched (no deciding_ci / deciding_ci_n_boot written):
    assert "deciding_ci_n_boot" not in vd
    # but the grid IS stamped (the refresh ran to completion over all cells):
    assert grid["bootstrap_refreshed_n_boot"] == i653.BOOTSTRAP_B


def test_refresh_succeeds_with_full_coverage(tmp_path):
    """All 4 DVs present with matching per_cell_ci for the 3 numeric DVs → no raise;
    each numeric cell gets a fresh deciding_ci + the full-depth n_boot stamp; the cos
    cell + a boundary (None-deciding) cell are left untouched."""
    verdicts = [
        {"cell_id": "n_ts", "deciding_dv": "top_share_lambda"},
        {"cell_id": "n_pr", "deciding_dv": "pr_lambda"},
        {"cell_id": "n_rk", "deciding_dv": "rank_k_at_90"},
        {"cell_id": "a_cos", "deciding_dv": "cos_top_to_rb", "deciding_ci_unavailable": True},
        {"cell_id": "b_none", "deciding_dv": None, "spectrum_underdetermined": True},
    ]
    out_root = _write_verdict(tmp_path, verdicts)
    per_cell_ci = {
        "n_ts": {"top_share_lambda": _ci(0.66, 0.78)},  # brackets 0.7 → ambiguous
        "n_pr": {"pr_lambda": _ci(5.6, 6.8)},  # above 5.0 → unambiguous
        "n_rk": {"rank_k_at_90": _ci(9.0, 13.0)},  # brackets 10 → ambiguous
    }
    ppb._refresh_ambiguity_flags(out_root, per_cell_ci)
    grid = json.loads((out_root / "cross_arm_verdict.json").read_text())
    by_id = {vd["cell_id"]: vd for vd in grid["verdicts"]}

    assert grid["bootstrap_refreshed_n_boot"] == i653.BOOTSTRAP_B
    # numeric cells: fresh CI + full-depth stamp; ambiguity checked on OWN thresholds
    assert by_id["n_ts"]["deciding_ci"] == [0.66, 0.78]
    assert by_id["n_ts"]["deciding_ci_n_boot"] == i653.BOOTSTRAP_B
    assert by_id["n_ts"]["ambiguous"] is True  # 0.7 ∈ [0.66, 0.78]
    assert by_id["n_pr"]["ambiguous"] is False  # 5.0 ∉ [5.6, 6.8]
    assert by_id["n_rk"]["ambiguous"] is True  # 10.0 ∈ [9.0, 13.0]
    # alignment + boundary cells untouched (no fresh deciding_ci written):
    assert "deciding_ci_n_boot" not in by_id["a_cos"]
    assert "deciding_ci_n_boot" not in by_id["b_none"]
