"""Unit tests for the hand-rolled Jonckheere-Terpstra implementation in
`scripts/analyze_issue_257.py`.

Plan §7.5 Stats critic blocker #2: the headline JT in `analyze_issue_257.py`
is gated on this test passing first (the analysis script's `_jt_unit_test`
function performs an equivalent runtime cross-check; this pytest version
runs offline at lint/dry-run time).

The test is skipped when scipy lacks `jonckheere_terpstra` (pre-1.11) —
the hand-rolled implementation has no cross-check available in that
environment, but the analysis script's `_jt_unit_test` documents the
skip behaviour identically.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


def _load_analyze_module():
    """Load scripts/analyze_issue_257.py as a module (it's not under src/
    so importing it directly is the simplest approach).
    """
    spec = importlib.util.spec_from_file_location(
        "analyze_issue_257", _REPO_ROOT / "scripts" / "analyze_issue_257.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_jt_decreasing_synthetic_with_ties():
    """Hand-rolled JT must agree with scipy on the fixed synthetic dataset."""
    try:
        from scipy.stats import jonckheere_terpstra
    except ImportError:
        pytest.skip("scipy.stats.jonckheere_terpstra unavailable (scipy < 1.11)")

    analyze = _load_analyze_module()
    synthetic = [
        (0, 0.40),
        (0, 0.40),
        (0, 0.35),
        (0, 0.45),
        (1, 0.20),
        (1, 0.20),
        (1, 0.18),
        (1, 0.22),
        (2, 0.10),
        (2, 0.10),
        (2, 0.12),
        (3, 0.02),
        (3, 0.02),
        (3, 0.04),
    ]
    J, p_hand, _z_hand = analyze._jt_one_sided(synthetic)
    groups = [[r for o, r in synthetic if o == k] for k in range(4)]
    res = jonckheere_terpstra(*groups, alternative="decreasing")
    assert abs(res.statistic - J) <= 0.5, f"JT statistic disagrees: hand={J}, scipy={res.statistic}"
    assert abs(res.pvalue - p_hand) <= 0.005, (
        f"JT p-value disagrees: hand={p_hand:.4f}, scipy={res.pvalue:.4f}"
    )


def test_jt_strict_monotone_decreasing():
    """A strictly decreasing 4-group dataset (no within-group ties) should
    produce a small one-sided p (decreasing alternative).
    """
    analyze = _load_analyze_module()
    monotone = [
        (0, 0.50),
        (0, 0.48),
        (1, 0.30),
        (1, 0.28),
        (2, 0.10),
        (2, 0.08),
        (3, 0.01),
        (3, 0.00),
    ]
    _J, p, z = analyze._jt_one_sided(monotone)
    assert p < 0.05, f"strictly-decreasing data should give one-sided p < 0.05; got {p:.4f}"
    assert z > 0, f"z statistic should be positive for decreasing trend; got {z:.4f}"


def test_jt_null_flat_data():
    """A flat dataset (all rates equal across bins) should produce a one-sided
    p near 0.5 (no monotone trend).
    """
    analyze = _load_analyze_module()
    flat = [
        (0, 0.10),
        (0, 0.10),
        (1, 0.10),
        (1, 0.10),
        (2, 0.10),
        (2, 0.10),
        (3, 0.10),
        (3, 0.10),
    ]
    _J, p, _z = analyze._jt_one_sided(flat)
    assert 0.3 <= p <= 0.7, f"flat data should give one-sided p near 0.5; got {p:.4f}"


def test_jt_increasing_yields_high_p():
    """An increasing dataset under a 'decreasing' alternative should produce
    a high one-sided p (> 0.5).
    """
    analyze = _load_analyze_module()
    increasing = [
        (0, 0.05),
        (0, 0.06),
        (1, 0.20),
        (1, 0.22),
        (2, 0.40),
        (2, 0.42),
        (3, 0.60),
        (3, 0.65),
    ]
    _J, p, z = analyze._jt_one_sided(increasing)
    assert p > 0.95, f"increasing data should give one-sided p > 0.95; got {p:.4f}"
    assert z < 0, f"z statistic should be negative for increasing trend; got {z:.4f}"
