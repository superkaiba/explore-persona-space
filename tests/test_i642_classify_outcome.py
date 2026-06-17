"""Exhaustive-lattice unit test for the #642 v8 ``_classify_outcome`` decision
rule (plan v8 §3 / §4.2 item 6).

The plan's §3 decision rule partitions every reachable (Δ_rank_matched, Δ_data)
outcome into EXACTLY ONE of 7 cells:

  H_survives / H_artifact / opposite_sign_rank / rank_in_band_data_quiet /
  rank_wide_data_separates / rank_wide_data_quiet / rank_positive_uncertain

A v6-era wording bug left the §3 lattice non-exhaustive (a reviewer-flagged
hole). v8 §4.2 item 6 MECHANIZES the totality claim: this test enumerates the 7
cells, asserts ``_classify_outcome`` returns exactly one (label, subreason) for
each, and asserts the routing is TOTAL + UNIQUE over a dense grid of CIs (every
reachable input maps to exactly one registered cell and no other).

Pure CPU — no model, no API, no GPU. ``_classify_outcome`` is a pure function of
``(point, ci_lo, ci_hi)`` for the two contrasts + the ±0.04 decomposition
threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "issue_642"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import i642_analyze as m  # noqa: E402

# The ±0.04 decomposition threshold the gates use (plan §3 / §11).
THR = 0.04

# A Δ_data CI that is "quiet" (does not separate positive): centred at 0, wide.
DATA_QUIET = (0.0, -0.02, 0.02)
# A Δ_data CI that "separates positive": point >= +0.04, CI excludes 0.
DATA_SEPARATES = (0.06, 0.02, 0.10)

# The 7 reachable lattice cells: name -> (delta_rank_ci, delta_data_ci,
# expected_label, expected_subreason). Each rank CI is (point, lo, hi).
LATTICE_CELLS: dict[str, tuple[tuple, tuple, str, str | None]] = {
    # Δ_rank separates positive (lo > 0, point >= +0.04) -> hypothesis survives.
    "H_survives": ((0.08, 0.04, 0.12), DATA_QUIET, "H_survives", None),
    # Δ_rank contained in band AND Δ_data separates positive -> data nuisance.
    "H_artifact": ((0.01, -0.02, 0.03), DATA_SEPARATES, "H_artifact", None),
    # Δ_rank separates NEGATIVE (hi < 0, point <= -0.04) -> reverse of hypothesis.
    "opposite_sign_rank": (
        (-0.08, -0.12, -0.04),
        DATA_QUIET,
        "H_indeterminate",
        "opposite_sign_rank",
    ),
    # Δ_rank ⊂ band AND Δ_data quiet -> both axes noise-limited.
    "rank_in_band_data_quiet": (
        (0.01, -0.02, 0.03),
        DATA_QUIET,
        "H_indeterminate",
        "rank_in_band_data_quiet",
    ),
    # Δ_rank wide/uncertain (not in band, not separating either way) AND Δ_data
    # separates positive -> data axis informative, method axis underpowered.
    "rank_wide_data_separates": (
        (0.02, -0.06, 0.10),
        DATA_SEPARATES,
        "H_indeterminate",
        "rank_wide_data_separates",
    ),
    # Δ_rank wide/uncertain AND Δ_data quiet -> both axes noise-limited.
    "rank_wide_data_quiet": (
        (0.02, -0.06, 0.10),
        DATA_QUIET,
        "H_indeterminate",
        "rank_wide_data_quiet",
    ),
    # Δ_rank point >= +0.04 but CI does NOT exclude 0 -> positive trend,
    # underpowered. (lo <= 0 with point >= +0.04, and point >= +0.04 keeps it
    # out of the band so it is NOT rank_in_band.)
    "rank_positive_uncertain": (
        (0.05, -0.01, 0.11),
        DATA_QUIET,
        "H_indeterminate",
        "rank_positive_uncertain",
    ),
}


@pytest.mark.parametrize("cell_name", list(LATTICE_CELLS))
def test_each_lattice_cell_classifies_uniquely(cell_name: str) -> None:
    """Each of the 7 §3 cells -> EXACTLY ONE (label, subreason)."""
    rank_ci, data_ci, exp_label, exp_subreason = LATTICE_CELLS[cell_name]
    label, subreason = m._classify_outcome(rank_ci, data_ci)
    assert (label, subreason) == (exp_label, exp_subreason), (
        f"cell {cell_name!r}: expected ({exp_label!r}, {exp_subreason!r}), "
        f"got ({label!r}, {subreason!r})"
    )


def test_seven_distinct_outcomes() -> None:
    """The 7 cells map to 7 DISTINCT (label, subreason) verdicts."""
    outcomes = {
        m._classify_outcome(rank_ci, data_ci)
        for (rank_ci, data_ci, _lbl, _sub) in LATTICE_CELLS.values()
    }
    assert len(outcomes) == 7, f"expected 7 distinct outcomes, got {len(outcomes)}: {outcomes}"


def test_subreason_none_iff_decisive() -> None:
    """subreason is None for the two DECISIVE verdicts (H_survives, H_artifact)
    and a non-None tag for every H_indeterminate cell."""
    for cell_name, (rank_ci, data_ci, _lbl, _sub) in LATTICE_CELLS.items():
        label, subreason = m._classify_outcome(rank_ci, data_ci)
        if label in ("H_survives", "H_artifact"):
            assert subreason is None, f"{cell_name}: decisive verdict must have subreason=None"
        else:
            assert label == "H_indeterminate"
            assert subreason is not None, f"{cell_name}: indeterminate must carry a subreason tag"


def test_lattice_is_total_and_unique_over_a_dense_grid() -> None:
    """Routing is TOTAL (every reachable input maps to a registered cell) and
    UNIQUE (exactly one label+subreason per input). Sweeps a dense grid of
    valid (point, lo, hi) triples with lo <= point <= hi, for both contrasts."""
    registered = {(lbl, sub) for (_r, _d, lbl, sub) in LATTICE_CELLS.values()}
    # grid of CI endpoints spanning well outside the ±0.04 band, in 0.01 steps
    grid = [round(x * 0.01, 4) for x in range(-12, 13)]  # -0.12 .. +0.12

    def _valid_cis() -> list[tuple[float, float, float]]:
        cis: list[tuple[float, float, float]] = []
        for lo in grid:
            for hi in grid:
                if hi < lo:
                    continue
                # point at lo, midpoint, and hi — always lo <= point <= hi
                mid = round((lo + hi) / 2.0, 4)
                for point in {lo, mid, hi}:
                    cis.append((point, lo, hi))
        return cis

    rank_cis = _valid_cis()
    # a small, representative set of data CIs (quiet, separates, and a few mids)
    data_cis = [
        DATA_QUIET,
        DATA_SEPARATES,
        (0.0, -0.10, 0.10),  # wide quiet
        (0.04, 0.0, 0.10),  # boundary: point==thr, lo==0 (not strictly > 0 -> quiet)
        (0.05, 0.01, 0.12),  # separates
        (-0.06, -0.10, -0.02),  # negative (still "does not separate POSITIVE")
    ]

    seen_outcomes: set[tuple[str, str | None]] = set()
    for rank_ci in rank_cis:
        for data_ci in data_cis:
            result = m._classify_outcome(rank_ci, data_ci)
            # TOTAL + UNIQUE: a single tuple is always returned, and it is one of
            # the 7 registered cells (never an unregistered / fall-through value).
            assert isinstance(result, tuple) and len(result) == 2
            assert result in registered, (
                f"unregistered outcome {result!r} for rank_ci={rank_ci} data_ci={data_ci} "
                "— the lattice has a hole the §3 enumeration does not cover"
            )
            seen_outcomes.add(result)
    # the dense grid actually exercises all 7 cells (the enumeration is reachable)
    assert seen_outcomes == registered, (
        f"dense grid did not reach every cell; missing: {registered - seen_outcomes}"
    )


def test_threshold_override_respected() -> None:
    """A custom decomp_threshold reshapes the band: a CI that separates at the
    default 0.04 becomes in-band when the threshold is widened past it."""
    rank_ci = (0.05, 0.045, 0.055)  # separates positive at thr=0.04
    assert m._classify_outcome(rank_ci, DATA_QUIET) == ("H_survives", None)
    # widen the threshold to 0.10: now point 0.05 < thr and CI ⊂ (-0.10, 0.10)
    label, subreason = m._classify_outcome(
        rank_ci, DATA_QUIET, thresholds={"decomp_threshold": 0.10}
    )
    assert (label, subreason) == ("H_indeterminate", "rank_in_band_data_quiet")
