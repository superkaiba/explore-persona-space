"""Tests for the preflight LFS write-gate leg ``check_hf_lfs_write_gate`` (#1654).

Plan §6 acceptance 4: the leg mirrors ``check_hf_storage``'s #1034 verdict
semantics — blocked + ``planned_upload_gb`` -> ERROR (with the
settings/billing click path), blocked without it -> WARNING, unknown /
disabled -> WARNING (fail-open), ok -> silent + summary line. Includes the
production-body test (real ``check_hf_lfs_write_gate`` body + real
``PreflightReport``; the hub helper faked as a signature-conformant autospec
returning a REAL ``LfsWriteGateProbe`` dataclass instance) and one
transitive end-to-end test where the REAL ``check_lfs_write_gate`` runs with
fakes only at the network boundary.
"""

from unittest import mock

import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

import explore_persona_space.orchestrate.hub as hub
from explore_persona_space.orchestrate.hub import LfsWriteGateProbe
from explore_persona_space.orchestrate.preflight import (
    PreflightReport,
    check_hf_lfs_write_gate,
)


def _probe(verdict, repo_id=None, detail="detail", probe_gb=16.0, billing_context=None):
    return LfsWriteGateProbe(
        verdict=verdict,
        repo_id=repo_id,
        detail=detail,
        probe_gb=probe_gb,
        billing_context=billing_context,
    )


def _stub_gate(monkeypatch, probe_or_exc):
    """Autospec the hub helper (the leg lazy-imports it at call time)."""
    if isinstance(probe_or_exc, LfsWriteGateProbe):
        fake = mock.create_autospec(hub.check_lfs_write_gate, return_value=probe_or_exc)
    else:
        fake = mock.create_autospec(hub.check_lfs_write_gate, side_effect=probe_or_exc)
    monkeypatch.setattr(hub, "check_lfs_write_gate", fake)
    return fake


def test_blocked_with_planned_upload_is_error(monkeypatch):
    report = PreflightReport()
    _stub_gate(
        monkeypatch,
        _probe(
            "billing-blocked",
            repo_id="superkaiba1/explore-persona-space-overflow",
            detail="403 Forbidden: credit recharge",
            billing_context={"canPay": True, "billingMode": "prepaid", "isPro": True},
        ),
    )
    check_hf_lfs_write_gate(report, planned_upload_gb=215)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "settings/billing" in report.errors[0]
    assert "billing-blocked" in report.errors[0]
    assert "canPay" in report.errors[0]
    assert report.hf_lfs_write_verdict == "billing-blocked"


def test_blocked_without_planned_upload_is_warning_only(monkeypatch):
    report = PreflightReport()
    _stub_gate(monkeypatch, _probe("storage-blocked", repo_id="r/x", detail="403 storage"))
    check_hf_lfs_write_gate(report, planned_upload_gb=None)
    assert report.ok is True
    assert report.errors == []
    assert any("settings/billing" in w for w in report.warnings)
    # Concern 1: the storage arm points at the detail excerpt for the exact
    # remediation path (the #3366 manual-review flavor names its own contact).
    assert any("error excerpt above governs" in w for w in report.warnings)


def test_unknown_is_warning_fail_open(monkeypatch):
    report = PreflightReport()
    _stub_gate(monkeypatch, _probe("unknown", detail="ConnectionError: reset"))
    check_hf_lfs_write_gate(report, planned_upload_gb=215)
    assert report.ok is True
    assert report.errors == []
    assert any("probe inconclusive" in w for w in report.warnings)


def test_disabled_with_planned_upload_warns(monkeypatch):
    report = PreflightReport()
    _stub_gate(monkeypatch, _probe("disabled", detail="EPM_HF_BILLING_PROBE=0", probe_gb=0.0))
    check_hf_lfs_write_gate(report, planned_upload_gb=215)
    assert report.ok is True
    assert any("EPM_HF_BILLING_PROBE=0" in w for w in report.warnings)


def test_disabled_without_planned_upload_is_silent(monkeypatch):
    report = PreflightReport()
    _stub_gate(monkeypatch, _probe("disabled", detail="EPM_HF_BILLING_PROBE=0", probe_gb=0.0))
    check_hf_lfs_write_gate(report, planned_upload_gb=None)
    assert report.ok is True
    assert report.warnings == []
    assert report.hf_lfs_write_verdict == "disabled"
    assert "HF LFS write gate: disabled" in report.summary()


def test_ok_records_fields_and_summary_line(monkeypatch):
    report = PreflightReport()
    _stub_gate(monkeypatch, _probe("ok", detail="negotiated (xet, 1 action(s))"))
    check_hf_lfs_write_gate(report, planned_upload_gb=215)
    assert report.ok is True
    assert report.warnings == []
    assert report.errors == []
    assert report.hf_lfs_write_verdict == "ok"
    assert report.hf_lfs_write_probe_gb == 16.0
    assert "HF LFS write gate: ok (16 GB declared probe)" in report.summary()


def test_helper_exception_degrades_to_warning(monkeypatch):
    """The env-knob ValueError (and any other helper crash) -> WARN, never raise."""
    report = PreflightReport()
    _stub_gate(monkeypatch, ValueError("EPM_HF_BILLING_PROBE_GB='banana' is not parseable"))
    check_hf_lfs_write_gate(report, planned_upload_gb=215)
    assert report.ok is True
    assert any("gate not evaluated" in w for w in report.warnings)


def test_wired_into_preflight_check():
    """Pin the preflight_check wiring: the leg runs after check_hf_storage,

    threading the SAME planned_upload_gb (no new CLI flag).
    """
    import inspect

    from explore_persona_space.orchestrate.preflight import preflight_check

    src = inspect.getsource(preflight_check)
    assert "check_hf_lfs_write_gate(report, planned_upload_gb)" in src
    assert src.index("check_hf_storage(report, planned_upload_gb)") < src.index(
        "check_hf_lfs_write_gate(report, planned_upload_gb)"
    )


def test_end_to_end_real_gate_blocked(monkeypatch):
    """Transitive production-body test: the REAL check_hf_lfs_write_gate body

    calls the REAL check_lfs_write_gate (added this round), with fakes only at
    the network boundary (post_lfs_batch_info + HfApi autospecs).
    """
    from huggingface_hub.lfs import post_lfs_batch_info

    monkeypatch.delenv("EPM_HF_BILLING_PROBE", raising=False)
    monkeypatch.delenv("EPM_HF_BILLING_PROBE_GB", raising=False)
    resp = requests.Response()
    resp.status_code = 403
    err = HfHubHTTPError(
        "403 Forbidden: You need to setup automatic credit recharge in order to upload more data",
        response=resp,
    )
    batch_fake = mock.create_autospec(post_lfs_batch_info, side_effect=err)
    monkeypatch.setattr("huggingface_hub.lfs.post_lfs_batch_info", batch_fake)
    api_cls = mock.create_autospec(HfApi)
    api_cls.return_value.whoami.return_value = {"canPay": False, "billingMode": "prepaid"}
    monkeypatch.setattr("huggingface_hub.HfApi", api_cls)

    report = PreflightReport()
    check_hf_lfs_write_gate(report, planned_upload_gb=215)
    assert report.ok is False
    assert any("billing-blocked" in e and "settings/billing" in e for e in report.errors)
    assert report.hf_lfs_write_verdict == "billing-blocked"
    assert report.hf_lfs_write_probe_gb == 16.0
    assert "HF LFS write gate: billing-blocked (16 GB declared probe)" in report.summary()
    assert batch_fake.call_count == 2  # both default repos probed
