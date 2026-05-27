"""TDD Phase 1 — H1 sign-and-ordering test (task #397, plan v4 §1 + §14 item 8).

Plan v4 reframes H1 (originally "magnitude replication of #383 within CI") to
**sign + ordering invariance** (Option (i) of three offered framings). Pass
requires:

  - per-factor across-seed Δ SIGN matches #383 for A (positive), B (positive),
    C (negative); D no requirement (#383 itself reported D as borderline);
  - Kendall-τ between v397 and v383 per-factor Δ vectors ≥ +0.67 (at most 1
    of 6 pairwise inversions across A/B/C/D).

Four synthetic scenarios:

  1. **Perfect match.** v397 deltas with same signs + same ordering as #383
     → τ = +1, h1_pass = True.
  2. **One sign flip on C.** v397 has C positive (wrong sign) → sign-match
     fails for C → h1_pass = False, regardless of τ.
  3. **One pairwise inversion** (A and B swap rank by magnitude, both still
     positive). 1 inversion of 6 → τ = +0.67 → at threshold → h1_pass = True.
  4. **Reversed ordering.** v397 is the magnitude-reversed permutation of
     #383 → τ ≤ +0.33 → h1_pass = False.

These cover the resolution edges of Kendall-τ on a 4-element vector
(τ ∈ {-1, -2/3, -1/3, 0, +1/3, +2/3, +1}) which plan v4 §13 calls out as a
limited-resolution claim — the test surface pins those values explicitly.

CPU-only; no model load.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.factor_screen_397.aggregator import (
    H383_FACTOR_DELTAS,
    h1_sign_and_ordering,
    kendall_tau,
)

# Reference #383 per-factor Δ vector — pinned here so the test fails loud if
# the module-level constant drifts away from the v4 plan v1 H1 reference.
EXPECTED_H383 = {"A": +33.6, "B": +27.8, "C": -26.9, "D": +11.2}


def test_h383_reference_vector_matches_plan() -> None:
    """The module-level H383 reference must match plan v4 §1 + §2."""
    assert H383_FACTOR_DELTAS == EXPECTED_H383, (
        "Module-level H383 reference drifted from plan v4 §2 H1 numbers. "
        "Update plan AND module together if #383 ever re-runs."
    )


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — kendall_tau raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_kendall_tau_perfect_match_is_plus_one() -> None:
    """Same per-factor ordering → τ = +1."""
    v1 = [+33.6, +27.8, -26.9, +11.2]  # #383
    v2 = [+33.6, +27.8, -26.9, +11.2]  # identical
    assert kendall_tau(v1, v2) == pytest.approx(1.0)


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — kendall_tau raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_kendall_tau_one_inversion_in_six_pairs_is_two_thirds() -> None:
    """4-element vector, 6 unordered pairs. 1 inversion → τ = (5-1)/6 = +0.667."""
    v1 = [+33.6, +27.8, -26.9, +11.2]  # ordering: A > B > D > C
    # Swap A and B values so rank order becomes B > A > D > C → 1 inversion.
    v2 = [+27.8, +33.6, -26.9, +11.2]
    tau = kendall_tau(v1, v2)
    assert tau == pytest.approx(2.0 / 3.0, abs=1e-3), (
        f"1 inversion of 6 pairs should give τ = 2/3 ≈ +0.667; got τ={tau}"
    )
    # At the H1 threshold (+0.67).
    assert tau >= 0.67 - 1e-3


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — kendall_tau raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_kendall_tau_full_reversal_is_minus_one() -> None:
    v1 = [+1.0, +2.0, +3.0, +4.0]
    v2 = [+4.0, +3.0, +2.0, +1.0]
    assert kendall_tau(v1, v2) == pytest.approx(-1.0)


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — h1_sign_and_ordering raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_h1_pass_when_signs_and_ordering_match_reference() -> None:
    """Scenario 1 from plan v4 §14 item 8: perfect match → PASS."""
    v397 = {"A": +30.0, "B": +25.0, "C": -22.0, "D": +9.0}  # same signs, same ordering
    result = h1_sign_and_ordering(v397, factor_deltas_v383=EXPECTED_H383)

    assert result["per_factor_sign_match"]["A"] is True
    assert result["per_factor_sign_match"]["B"] is True
    assert result["per_factor_sign_match"]["C"] is True
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["h1_pass"] is True


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — h1_sign_and_ordering raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_h1_fail_when_factor_c_flips_sign() -> None:
    """Scenario 2: C flips from negative to positive → sign-match fails for C → h1_pass=False.

    Even if Kendall-τ might mathematically still pass on the magnitude
    ordering, the per-factor sign requirement on C is binding.
    """
    v397 = {"A": +30.0, "B": +25.0, "C": +22.0, "D": +9.0}  # C wrong sign
    result = h1_sign_and_ordering(v397, factor_deltas_v383=EXPECTED_H383)

    assert result["per_factor_sign_match"]["C"] is False
    assert result["h1_pass"] is False, "C sign-flip must veto H1 even if Kendall-τ is high"


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — h1_sign_and_ordering raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_h1_pass_at_threshold_with_one_pairwise_inversion() -> None:
    """Scenario 3: A and B swap rank by magnitude → 1 inversion → τ = +0.67 → PASS.

    #383 ordering: A (+33.6) > B (+27.8) > D (+11.2) > C (-26.9) by magnitude
    rank. Swap A and B magnitudes so the new ordering becomes B > A > D > C,
    keeping all signs the same. That's 1 inversion of 6 pairs → τ = +0.667 →
    just at the pass threshold.
    """
    v397 = {"A": +27.0, "B": +33.0, "C": -26.0, "D": +11.0}  # A and B swapped
    result = h1_sign_and_ordering(v397, factor_deltas_v383=EXPECTED_H383, tau_threshold=0.67)

    # Signs still match.
    assert result["per_factor_sign_match"]["A"] is True
    assert result["per_factor_sign_match"]["B"] is True
    assert result["per_factor_sign_match"]["C"] is True
    assert result["kendall_tau"] == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert result["h1_pass"] is True


@pytest.mark.xfail(
    reason="Phase 1 (TDD) stub — h1_sign_and_ordering raises NotImplementedError until Phase 2.",
    strict=True,
    raises=NotImplementedError,
)
def test_h1_fail_when_ordering_is_reversed() -> None:
    """Scenario 4: v397 ordering is the reverse of #383's → τ ≤ +0.33 → h1_pass=False.

    Keep all signs matching #383 (so the sign-match gate passes for A/B/C),
    but reverse the magnitude ordering completely. Kendall-τ should drop
    below threshold → h1_pass=False.
    """
    # #383 magnitude ordering A > B > D > C; v397 magnitudes flipped to
    # produce C > D > B > A while keeping signs.
    v397 = {"A": +10.0, "B": +20.0, "C": -40.0, "D": +30.0}
    result = h1_sign_and_ordering(v397, factor_deltas_v383=EXPECTED_H383, tau_threshold=0.67)
    assert result["kendall_tau"] <= 1.0 / 3.0 + 1e-3, (
        f"Reversed ordering should drop τ below +0.33; got τ={result['kendall_tau']}"
    )
    assert result["h1_pass"] is False
