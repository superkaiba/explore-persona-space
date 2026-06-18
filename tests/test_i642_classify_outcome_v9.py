"""Exhaustive-lattice unit test for the #642 v9 ``_classify_outcome_v9`` decision
rule (plan v10 §3).

The plan's §3 decision lattice partitions every reachable refusal
Δ_rank_matched outcome (a SINGLE contrast — no Δ_data this round) into EXACTLY
one of these cells:

  REPLICATES                  CI excludes 0, sign positive, |point| >= +0.04
  PARTIAL                     sign positive, CI excludes 0, BUT point < +0.04
  FAILS / opposite_sign_rank  CI excludes 0 NEGATIVE, point <= -0.04
  FAILS / noise_limited       CI spans 0, point inside the band
  FAILS / positive_uncertain  point >= +0.04 but CI does NOT exclude 0

This MECHANIZES the §3 totality claim (mirrors the v8 ``_classify_outcome``
test): every reachable (point, lo, hi) maps to exactly one registered cell.

Pure CPU — no model, no API, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import i642_analyze as m  # noqa: E402

THR = 0.04  # the ±0.04 separation band (matched to round 4's +0.063 scale)

# name -> (delta_rank_ci, expected_verdict, expected_subreason). CI = (point, lo, hi).
LATTICE_CELLS: dict[str, tuple[tuple, str, str | None]] = {
    # CI excludes 0 positive AND |point| clears the band -> the gap is general.
    "REPLICATES": ((0.08, 0.04, 0.12), "REPLICATES", None),
    # CI excludes 0 positive but point shrinks below +0.04 -> directionally general.
    "PARTIAL": ((0.03, 0.01, 0.05), "PARTIAL", None),
    # CI excludes 0 NEGATIVE, point past -0.04 -> sign flip (dense leaks LESS).
    "opposite_sign_rank": ((-0.08, -0.12, -0.04), "FAILS", "opposite_sign_rank"),
    # CI spans 0, point inside the band -> noise-limited power statement.
    "noise_limited": ((0.01, -0.03, 0.05), "FAILS", "noise_limited"),
    # point clears the band positive but CI does NOT exclude 0 -> underpowered.
    "positive_uncertain": ((0.06, -0.01, 0.13), "FAILS", "positive_uncertain"),
}


@pytest.mark.parametrize("cell_name", list(LATTICE_CELLS))
def test_each_v9_lattice_cell_classifies_uniquely(cell_name: str) -> None:
    rank_ci, exp_verdict, exp_sub = LATTICE_CELLS[cell_name]
    verdict, sub = m._classify_outcome_v9(rank_ci)
    assert (verdict, sub) == (exp_verdict, exp_sub), (
        f"cell {cell_name!r}: expected ({exp_verdict!r}, {exp_sub!r}), got ({verdict!r}, {sub!r})"
    )


def test_v9_five_distinct_outcomes() -> None:
    outcomes = {m._classify_outcome_v9(rank_ci) for (rank_ci, _v, _s) in LATTICE_CELLS.values()}
    assert len(outcomes) == 5, f"expected 5 distinct outcomes, got {len(outcomes)}: {outcomes}"


def test_v9_subreason_none_iff_decisive() -> None:
    """REPLICATES + PARTIAL carry subreason=None; every FAILS cell carries a tag."""
    for cell_name, (rank_ci, _v, _s) in LATTICE_CELLS.items():
        verdict, sub = m._classify_outcome_v9(rank_ci)
        if verdict in ("REPLICATES", "PARTIAL"):
            assert sub is None, f"{cell_name}: decisive verdict must have subreason=None"
        else:
            assert verdict == "FAILS"
            assert sub is not None, f"{cell_name}: FAILS must carry a subreason tag"


def test_v9_lattice_is_total_and_unique_over_a_dense_grid() -> None:
    """Routing is TOTAL (every reachable input maps to a registered cell) and
    UNIQUE (exactly one verdict+subreason per input)."""
    registered = {(v, s) for (_r, v, s) in LATTICE_CELLS.values()}
    grid = [round(x * 0.01, 4) for x in range(-12, 13)]
    seen: set[tuple[str, str | None]] = set()
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            mid = round((lo + hi) / 2.0, 4)
            for point in {lo, mid, hi}:
                result = m._classify_outcome_v9((point, lo, hi))
                assert isinstance(result, tuple) and len(result) == 2
                assert result in registered, (
                    f"unregistered outcome {result!r} for ci=({point},{lo},{hi}) — "
                    "the v9 lattice has a hole the §3 enumeration does not cover"
                )
                seen.add(result)
    assert seen == registered, f"dense grid missed cells: {registered - seen}"


def test_v9_threshold_override_respected() -> None:
    """A custom decomp_threshold reshapes the band: a CI that REPLICATES at the
    default 0.04 becomes noise_limited when the threshold is widened past it."""
    rank_ci = (0.05, 0.045, 0.055)
    assert m._classify_outcome_v9(rank_ci) == ("REPLICATES", None)
    verdict, sub = m._classify_outcome_v9(rank_ci, thresholds={"decomp_threshold": 0.10})
    # point 0.05 < thr 0.10 and lo 0.045 > 0 -> sign positive, CI excludes 0,
    # point below band -> PARTIAL.
    assert (verdict, sub) == ("PARTIAL", None)


# ---------------------------------------------------------------------------
# B2 (round-1 reconcile blocker): v9 install-failure path unification.
# The dispatcher's phase3_select MUST write per-arm install_failure_<arm>.json
# (the convention the v9 analyzer reads); a no-suffix install_failure.json would
# be invisible to _v9_install_failure -> the analyzer would fall through to full
# analysis on a killed behavior. This test materializes the per-arm file and
# asserts the analyzer short-circuits to the KILLED verdict WITHOUT requiring
# generation manifests (i.e. without calling _v9_analyze_behavior).
# ---------------------------------------------------------------------------

import json  # noqa: E402

from i642_common import V9_ARMS  # noqa: E402


def _write_v9_install_failure(eval_root: Path, behavior: str, arm: str) -> Path:
    sa = eval_root / behavior / "stage_a"
    sa.mkdir(parents=True, exist_ok=True)
    fp = sa / f"install_failure_{arm}.json"
    fp.write_text(
        json.dumps(
            {
                "behavior": behavior,
                "arm": arm,
                "kill_criterion": "a_install_failure",
                "arm_ok": {a: (a != arm) for a in V9_ARMS},
            }
        )
    )
    return fp


def test_v9_per_arm_install_failure_short_circuits_to_killed(tmp_path, capsys) -> None:
    """A per-arm install_failure_loraRefOP.json under eval_root makes the v9
    analyzer return 0 with verdict=KILLED, never reaching _v9_analyze_behavior
    (so no generation manifests are needed)."""
    eval_root = tmp_path / "eval_results" / "issue_642"
    _write_v9_install_failure(eval_root, "refusal", "loraRefOP")

    # Guard: if the analyzer reached full analysis it would call this; assert it
    # is NOT invoked on the kill path.
    called = {"full_analysis": False}
    orig = m._v9_analyze_behavior

    def _tripwire(*a, **k):
        called["full_analysis"] = True
        return orig(*a, **k)

    m._v9_analyze_behavior = _tripwire
    try:
        rc = m.main(
            [
                "--v9",
                "--behavior",
                "refusal",
                "--eval-root",
                str(eval_root),
                "--no-refetch",
            ]
        )
    finally:
        m._v9_analyze_behavior = orig

    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict=KILLED" in out
    assert "arm=loraRefOP" in out
    assert called["full_analysis"] is False


def test_v9_no_suffix_install_failure_is_invisible(tmp_path) -> None:
    """Regression for the pre-fix bug: a NO-SUFFIX install_failure.json (the old
    writer) is NOT detected by the v9 reader — confirming the per-arm convention
    is the only one that works. (Pre-fix the writer produced this file, so the
    kill was never seen.)"""
    eval_root = tmp_path / "eval_results" / "issue_642"
    sa = eval_root / "refusal" / "stage_a"
    sa.mkdir(parents=True, exist_ok=True)
    (sa / "install_failure.json").write_text(json.dumps({"behavior": "refusal"}))
    # _v9_install_failure reads ONLY install_failure_<arm>.json, so the no-suffix
    # file is invisible -> returns None (the bug the B2 writer fix closes).
    res = m._v9_install_failure(eval_root, "refusal", refetch=False, v9_experiment="x")
    assert res is None
