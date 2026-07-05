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


# ---------------------------------------------------------------------------
# #1034 — opt-in planned-upload hard gate (planned_upload_gb)
# ---------------------------------------------------------------------------


def _probe_recorder(cached, live=None):
    """HELPER stand-in discriminating on ``force_refresh`` (signature-mirrored
    fake: keyword-only kwarg, matching the real probe). Records the kwarg per
    call so tests can assert the live confirm actually ran."""
    calls: list[bool] = []

    def fake(*a, force_refresh=False, **k):
        calls.append(force_refresh)
        if force_refresh and live is not None:
            return live
        return cached

    return fake, calls


def test_planned_none_keeps_warn_only_default():
    """#1034 test 15: no projection -> today's WARN-only behavior, even in a
    state where a 2000 GB projection WOULD have hard-failed (used 9/10)."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=9.0)):
        check_hf_storage(report)
    assert report.ok is True
    assert report.errors == []


def test_live_confirmed_insufficient_routing_off_errors():
    """#1034 test 16: used 9 + planned 2000 GB > ceiling 10, live re-probe
    still over, routing unset -> ERROR (report.ok False) naming remaining GB;
    the force_refresh live confirm is observed."""
    fake, calls = _probe_recorder(_headroom(used_tb=9.0))
    report = PreflightReport()
    with patch(HELPER, side_effect=fake):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is False
    assert any("HF headroom insufficient" in e for e in report.errors)
    assert any("1000 GB remaining" in e for e in report.errors)
    assert True in calls  # the live force_refresh confirm actually ran


def test_insufficient_with_routing_armed_warns(monkeypatch):
    """#1034 test 17: same state + EPM_HF_OVERFLOW_ROUTING=1 -> warning
    (naming the environment-wide arming effect), ok True."""
    monkeypatch.setenv("EPM_HF_OVERFLOW_ROUTING", "1")
    fake, _ = _probe_recorder(_headroom(used_tb=9.0))
    report = PreflightReport()
    with patch(HELPER, side_effect=fake):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is True
    assert any("would exceed the soft ceiling" in w for w in report.warnings)
    assert any("environment-wide" in w for w in report.warnings)


def test_unknown_headroom_with_planned_warns_only():
    """#1034 test 18: unknown headroom + planned -> warning only (fail-open;
    the reactive 403 backstop stays authoritative)."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=None, basis="unknown (api down)")):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is True
    assert any("usage unknown" in w for w in report.warnings)


def test_fits_no_error_no_new_warning():
    """#1034 test 19: used 3 + planned 100 GB fits under ceiling 10 -> no
    error, no new warning."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=3.0)):
        check_hf_storage(report, planned_upload_gb=100.0)
    assert report.ok is True
    assert report.warnings == []
    assert report.errors == []


def test_cli_threads_planned_upload_gb(monkeypatch, capsys):
    """#1034 test 20: --planned-upload-gb threads through main() into
    preflight_check (kwarg recorded on the patched seam)."""
    from explore_persona_space.orchestrate import preflight

    seen: dict = {}

    def fake_check(**kwargs):
        seen.update(kwargs)
        return PreflightReport()

    monkeypatch.setattr(preflight, "preflight_check", fake_check)
    rc = preflight.main(["--no-gpu", "--planned-upload-gb", "2000"])
    assert rc == 0
    assert seen["planned_upload_gb"] == 2000.0


def test_preflight_check_end_to_end_wiring(monkeypatch):
    """#1034 test 21 (Must-Fix): calling preflight_check(planned_upload_gb=…)
    DIRECTLY carries the kwarg into check_hf_storage — the returned report
    holds the gate ERROR. Only the hub probe seam is patched (never
    preflight_check / check_hf_storage themselves — dropping the
    `check_hf_storage(report, planned_upload_gb)` threading would silently
    revert the gate to WARN-only while the unit tests above stay green). The
    two network/import-heavy SIBLING checks are no-op'd for hermeticity; they
    are orthogonal to the threading edge under test."""
    from explore_persona_space.orchestrate import preflight

    monkeypatch.setattr(preflight, "check_connectivity", lambda report: None)
    monkeypatch.setattr(preflight, "check_vllm_transformers_compat", lambda report: None)
    with patch(HELPER, return_value=_headroom(used_tb=9.0)):
        report = preflight.preflight_check(
            require_gpu=False, check_code_sync=False, planned_upload_gb=2000.0
        )
    assert any("HF headroom insufficient" in e for e in report.errors)
    assert report.ok is False


def test_stale_high_cache_live_fits_never_blocks():
    """#1034 test 22 (Must-Fix): a stale-high CACHED read (9.5 + 2 > 10)
    whose force_refresh live re-probe fits (7 + 2 <= 10) adds NOTHING — no
    error, no insufficient warning. A stale cache can never false-block a
    healthy planned run."""
    fake, calls = _probe_recorder(_headroom(used_tb=9.5), live=_headroom(used_tb=7.0))
    report = PreflightReport()
    with patch(HELPER, side_effect=fake):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is True
    assert report.errors == []
    assert not any("would exceed" in w or "insufficient" in w for w in report.warnings)
    assert True in calls  # the live re-probe is what rescued the verdict


def test_stale_high_cache_live_unknown_warns_gate_not_evaluated():
    """#1034 test 25 (revision r2): a stale-high CACHED read (9.5 + 2 > 10, so
    no existing WARN fires — used is KNOWN and under ceiling) whose
    force_refresh live re-probe returns UNKNOWN (used_tb=None) stays fail-open
    (ok True, no error) but the REQUESTED gate is named NOT EVALUATED in a
    warning — never a warning-free report about an armed gate."""
    fake, calls = _probe_recorder(
        _headroom(used_tb=9.5), live=_headroom(used_tb=None, basis="unknown (api down)")
    )
    report = PreflightReport()
    with patch(HELPER, side_effect=fake):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is True
    assert report.errors == []
    assert any("planned-upload gate not evaluated" in w for w in report.warnings)
    assert True in calls  # the live force_refresh re-probe is what returned unknown


def test_probe_raises_with_planned_flag_warns_only():
    """#1034 test 23: the helper raises (bad ceiling env / API error) WITH the
    planned flag set -> warning only, ok True (the existing except-Exception
    wrapper covers the gate; fail-open pinned with the flag armed)."""
    report = PreflightReport()
    with patch(HELPER, side_effect=ValueError("EPM_HF_STORAGE_SOFT_CEILING_TB='ten' ...")):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is True
    assert any("headroom check failed" in w for w in report.warnings)


def test_disabled_with_planned_flag_warns_gate_requested():
    """#1034 test 24: kill switch + planned flag -> the armed-but-disabled
    combo is VISIBLE ('gate requested but storage check disabled'), ok True —
    never a silent swallow of a requested gate."""
    report = PreflightReport()
    with patch(HELPER, return_value=_headroom(used_tb=None, basis="disabled")):
        check_hf_storage(report, planned_upload_gb=2000.0)
    assert report.ok is True
    assert any("gate requested but storage check disabled" in w for w in report.warnings)
