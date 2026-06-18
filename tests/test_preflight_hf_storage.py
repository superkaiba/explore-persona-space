"""Tests for the preflight HF public-storage WARN (#564, AC2).

``check_hf_storage`` is advisory-only: it must NEVER add an error or raise —
including when the helper raises its deliberate ``ValueError`` on a bad
ceiling/TTL env value (that error stays fail-loud at the persist gate in
``train/trainer.py``; preflight degrades it to a warning).
"""

from unittest.mock import patch

from explore_persona_space.orchestrate.hub import HfStorageHeadroom
from explore_persona_space.orchestrate.preflight import PreflightReport, check_hf_storage

HELPER = "explore_persona_space.orchestrate.hub.check_hf_storage_headroom"


def _headroom(**kw):
    base = dict(used_tb=3.2, ceiling_tb=10.0, over_ceiling=False, basis="live-api", n_repos=5)
    base.update(kw)
    return HfStorageHeadroom(**base)


def test_over_ceiling_warns_but_stays_ok():
    """Test 12: over ceiling -> warning, ok unchanged, report fields + summary set."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=11.3, over_ceiling=True, n_repos=414)):
        check_hf_storage(report)
    assert report.ok is True
    assert report.errors == []
    assert any("exceeds soft ceiling" in w for w in report.warnings)
    assert report.hf_storage_used_tb == 11.3
    assert report.hf_storage_ceiling_tb == 10.0
    assert report.hf_storage_basis == "live-api"
    assert "HF storage: 11.30 TB / ceiling 10.0 TB (live-api)" in report.summary()


def test_unknown_headroom_warns_but_stays_ok():
    """Test 13: unknown headroom -> 'usage unknown' warning, ok True."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=None, basis="unknown (api down)")):
        check_hf_storage(report)
    assert report.ok is True
    assert any("usage unknown" in w for w in report.warnings)
    assert report.hf_storage_used_tb is None
    assert "HF storage: unknown" in report.summary()


def test_under_ceiling_no_storage_warning():
    """Test 14: under ceiling -> fields populated, NO storage warning."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom()):
        check_hf_storage(report)
    assert report.ok is True
    assert not any("HF public" in w for w in report.warnings)
    assert report.hf_storage_used_tb == 3.2


def test_disabled_basis_sets_fields_without_warning():
    """Kill switch: basis 'disabled' records the basis and adds no warning."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=None, basis="disabled")):
        check_hf_storage(report)
    assert report.warnings == []
    assert report.hf_storage_basis == "disabled"


def test_bad_ceiling_env_valueerror_is_caught():
    """Test 14b: the helper's ValueError (bad env) is CAUGHT here — warning,
    ok True — pinning the deliberate helper-raises/preflight-warns split."""
    report = PreflightReport()
    with patch(HELPER, side_effect=ValueError("EPM_HF_STORAGE_SOFT_CEILING_TB='ten' ...")):
        check_hf_storage(report)
    assert report.ok is True
    assert any("headroom check failed" in w for w in report.warnings)
