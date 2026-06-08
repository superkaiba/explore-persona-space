"""Plan-§6 combined-decision-rule regression test for task #515.

Round-1 code-review (Codex Major 1, reconciler binding) flagged that
``_decision_label`` consumed only ``n_clearing`` and returned
``"real_null"`` whenever 4+ sources cleared the SocioT paper gate -
even if the Claude warmth-rating cross-meter disagreed (low Spearman
rho). The plan §6 combined rule requires BOTH conditions:

  "#496's null is real" ⇔
    (≥4/6 sources clear the SocioT paper gate) AND
    (Spearman ρ_cross_meter ≥ +0.5)

This test pins all 6 quadrants of (n_clearing, rho) including the
edge cases:
  - n_clearing ≤ 1 → "artifact" regardless of rho (rho moot when the
    intervention obviously didn't take).
  - n_clearing in {2, 3} → "ambiguous" regardless of rho.
  - n_clearing ≥ 4 AND rho ≥ 0.5 → "real_null".
  - n_clearing ≥ 4 AND rho < 0.5 → "ambiguous" (gate cleared but the
    two meters disagree).
  - n_clearing ≥ 4 AND rho is None (Claude data incomplete) →
    "ambiguous" (conservative fallback, NOT "real_null").
  - n_clearing ≥ 4 AND rho is NaN → "ambiguous" (NaN is None in our
    JSON-safe mapping).
"""

from __future__ import annotations

from pathlib import Path


def _import_dispatcher():
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import dispatch_warmth_manipulation_check_515 as mod  # type: ignore[import-not-found]

    return mod


def test_artifact_at_or_below_1_regardless_of_rho():
    mod = _import_dispatcher()
    for n in (0, 1):
        for rho in (None, float("nan"), -1.0, 0.0, 0.49, 0.5, 0.9, 1.0):
            assert mod._decision_label(n, rho) == "artifact", (n, rho)


def test_ambiguous_at_2_or_3_regardless_of_rho():
    mod = _import_dispatcher()
    for n in (2, 3):
        for rho in (None, float("nan"), -1.0, 0.0, 0.49, 0.5, 0.9, 1.0):
            assert mod._decision_label(n, rho) == "ambiguous", (n, rho)


def test_real_null_requires_both_gate_and_rho():
    mod = _import_dispatcher()
    # gate cleared AND rho passes
    for n in (4, 5, 6):
        for rho in (0.5, 0.75, 1.0):
            assert mod._decision_label(n, rho) == "real_null", (n, rho)


def test_high_gate_low_rho_is_ambiguous_not_real_null():
    mod = _import_dispatcher()
    # gate cleared but meters disagree -> conservative ambiguous
    for n in (4, 5, 6):
        for rho in (-1.0, -0.5, 0.0, 0.25, 0.49):
            assert mod._decision_label(n, rho) == "ambiguous", (n, rho)


def test_high_gate_missing_rho_is_ambiguous_not_real_null():
    """Conservative fallback: missing Claude data should NEVER promote
    to 'real_null'. Reconciler explicitly called out this case as the
    failure mode to prevent."""
    mod = _import_dispatcher()
    for n in (4, 5, 6):
        assert mod._decision_label(n, None) == "ambiguous", n
        assert mod._decision_label(n, float("nan")) == "ambiguous", n
