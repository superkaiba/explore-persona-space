"""Tests for the zero-byte LFS batch-negotiation billing/quota probe (#1654).

Covers plan §6 acceptance criteria 1-3 + 5: the ported ``is_billing_403``
classifier arms (incl. the hub-issue-#3366 manual-review 403, pinned at its
REALIZED classification ``storage-blocked`` — the plan's §8 row 4 stated
``unknown``, which was wrong), the verdict mapping through a seam-stubbed
``post_lfs_batch_info`` (signature-conformant ``create_autospec`` at the
network boundary), the kill switch / env knobs, and the production-body tests
(real ``check_lfs_write_gate`` body reaching UploadInfo construction, env
parsing, the per-repo loop, the worst-wins fold, and the fail-soft
billing-context fetch).
"""

import re
from unittest import mock

import pytest
import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.lfs import UploadInfo, post_lfs_batch_info

from explore_persona_space.orchestrate.hub import (
    DEFAULT_BILLING_PROBE_GB,
    DEFAULT_MODEL_REPO,
    DEFAULT_OVERFLOW_REPO,
    LfsWriteGateProbe,
    check_lfs_write_gate,
    is_billing_403,
)

BATCH_SEAM = "huggingface_hub.lfs.post_lfs_batch_info"
HFAPI_SEAM = "huggingface_hub.HfApi"

BILLING_403_TEXT = (
    "403 Forbidden: You need to setup automatic credit recharge in order to upload more data"
)
STORAGE_403_TEXT = "403 Forbidden: You have exceeded your public storage space"
# Verbatim hub-issue-#3366 flavor (manual-review block). Contains BOTH "403"
# and "storage", so _is_storage_quota_403 matches -> storage-blocked.
MANUAL_REVIEW_403_TEXT = (
    "403 Forbidden: Your storage patterns tripped our internal systems. Please contact us "
    "at website@huggingface.co so we can verify your account and unlock more storage for "
    "your use case"
)


def _http_error(text: str, status: int = 403) -> HfHubHTTPError:
    """Response-bearing HF error (the ``hf_raise_for_status`` shape)."""
    resp = requests.Response()
    resp.status_code = status
    return HfHubHTTPError(text, response=resp)


def _ok_return(oid: str = "ab" * 32, size: int = int(16e9)):
    """A clean batch response: one action object, no errors, xet transfer."""
    return ([{"oid": oid, "size": size, "actions": {}}], [], "xet")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("EPM_HF_BILLING_PROBE", raising=False)
    monkeypatch.delenv("EPM_HF_BILLING_PROBE_GB", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")


@pytest.fixture()
def hfapi_fake(monkeypatch):
    """Autospec'd HfApi whose instance whoami() returns real billing fields."""
    api_cls = mock.create_autospec(HfApi)
    api_cls.return_value.whoami.return_value = {
        "canPay": True,
        "billingMode": "prepaid",
        "isPro": True,
        "name": "someone",
    }
    monkeypatch.setattr(HFAPI_SEAM, api_cls)
    return api_cls


# ── Acceptance 1: classifier arms ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        # (i) response-bearing 403 + "credit recharge" in message
        (_http_error(BILLING_403_TEXT), True),
        # (i-b) response arm alone: 403 status, message WITHOUT "403"
        (_http_error("please setup automatic credit recharge to continue"), True),
        # (ii) response-less shape (xet Rust-boundary wrap, #931)
        (RuntimeError("upload failed: 403 — You need to setup automatic credit recharge"), True),
        # (iii) the #541 storage-403 text: no "credit recharge"
        (_http_error(STORAGE_403_TEXT), False),
        # (iii-b) the #3366 manual-review 403: no "credit recharge"
        (_http_error(MANUAL_REVIEW_403_TEXT), False),
        # (iv) 5xx with "credit recharge" in the body: response arm requires 403
        (_http_error("server hiccup mentioning credit recharge", status=500), False),
        # (iv-b) response-less with "credit recharge" but no "403"
        (RuntimeError("500 server error: credit recharge unavailable"), False),
        # (v) unrelated 403
        (_http_error("403 Forbidden: access denied"), False),
    ],
)
def test_is_billing_403_arms(err, expected):
    assert is_billing_403(err) is expected


# ── Acceptance 2: verdict mapping through the seam-stubbed boundary ──────────


def _gate_with_side_effects(monkeypatch, side_effect):
    fake = mock.create_autospec(post_lfs_batch_info, side_effect=side_effect)
    monkeypatch.setattr(BATCH_SEAM, fake)
    return fake


@pytest.mark.parametrize(
    ("side_effect", "expected_verdict"),
    [
        (_http_error(BILLING_403_TEXT), "billing-blocked"),
        (_http_error(STORAGE_403_TEXT), "storage-blocked"),
        # REALIZED #3366 arm: text carries "403" + "storage" -> storage-blocked
        # (pin the actual classification, NOT the plan's stated "unknown").
        (_http_error(MANUAL_REVIEW_403_TEXT), "storage-blocked"),
        (requests.ConnectionError("connection reset"), "unknown"),
    ],
)
def test_verdict_mapping_on_raised_errors(monkeypatch, hfapi_fake, side_effect, expected_verdict):
    _gate_with_side_effects(monkeypatch, side_effect)
    probe = check_lfs_write_gate()
    assert probe.verdict == expected_verdict
    assert isinstance(probe, LfsWriteGateProbe)


def test_verdict_ok_on_clean_negotiation(monkeypatch, hfapi_fake):
    fake = _gate_with_side_effects(monkeypatch, [_ok_return(), _ok_return()])
    probe = check_lfs_write_gate()
    assert probe.verdict == "ok"
    assert probe.repo_id is None
    assert "negotiated (xet" in probe.detail
    assert fake.call_count == 2  # both default repos probed


def test_in_band_error_classified_on_code_field(monkeypatch, hfapi_fake):
    """Concern 2: the git-lfs in-band error dict carries the 403 in ``code``;

    the message alone lacks "403" — classification must read BOTH fields.
    """
    in_band = (
        [],
        [
            {
                "oid": "ab" * 32,
                "size": int(16e9),
                "error": {
                    "code": 403,
                    "message": (
                        "You need to setup automatic credit recharge in order to upload more data"
                    ),
                },
            }
        ],
        "xet",
    )
    _gate_with_side_effects(monkeypatch, [in_band, _ok_return()])
    probe = check_lfs_write_gate()
    assert probe.verdict == "billing-blocked"
    assert probe.repo_id == DEFAULT_MODEL_REPO


def test_in_band_unmatched_error_is_unknown(monkeypatch, hfapi_fake):
    """An in-band error matching neither predicate degrades to unknown."""
    in_band = (
        [],
        [{"oid": "cd" * 32, "size": 7, "error": {"code": 422, "message": "validation failed"}}],
        None,
    )
    _gate_with_side_effects(monkeypatch, [in_band, _ok_return()])
    probe = check_lfs_write_gate()
    assert probe.verdict == "unknown"
    assert "validation failed" in probe.detail


def test_worst_wins_across_repos(monkeypatch, hfapi_fake):
    """Repo 1 ok + repo 2 billing-403 -> billing-blocked, repo_id = repo 2."""
    _gate_with_side_effects(monkeypatch, [_ok_return(), _http_error(BILLING_403_TEXT)])
    probe = check_lfs_write_gate()
    assert probe.verdict == "billing-blocked"
    assert probe.repo_id == DEFAULT_OVERFLOW_REPO


def test_worst_wins_storage_does_not_outrank_billing(monkeypatch, hfapi_fake):
    _gate_with_side_effects(
        monkeypatch, [_http_error(BILLING_403_TEXT), _http_error(STORAGE_403_TEXT)]
    )
    probe = check_lfs_write_gate()
    assert probe.verdict == "billing-blocked"
    assert probe.repo_id == DEFAULT_MODEL_REPO


# ── Acceptance 3: kill switch + env knobs ────────────────────────────────────


def test_kill_switch_disabled_zero_network(monkeypatch, hfapi_fake):
    monkeypatch.setenv("EPM_HF_BILLING_PROBE", "0")
    fake = _gate_with_side_effects(monkeypatch, AssertionError("must not be called"))
    probe = check_lfs_write_gate()
    assert probe.verdict == "disabled"
    fake.assert_not_called()
    hfapi_fake.assert_not_called()


def test_env_knob_probe_gb_reaches_boundary(monkeypatch, hfapi_fake):
    monkeypatch.setenv("EPM_HF_BILLING_PROBE_GB", "32")
    fake = _gate_with_side_effects(monkeypatch, [_ok_return(), _ok_return()])
    probe = check_lfs_write_gate()
    assert probe.probe_gb == 32.0
    infos = list(fake.call_args_list[0].args[0])
    assert infos[0].size == int(32e9)


def test_env_knob_nonparseable_raises(monkeypatch, hfapi_fake):
    """The _env_float fail-fast contract: a bad knob raises ValueError."""
    monkeypatch.setenv("EPM_HF_BILLING_PROBE_GB", "banana")
    _gate_with_side_effects(monkeypatch, [_ok_return(), _ok_return()])
    with pytest.raises(ValueError, match=re.escape("EPM_HF_BILLING_PROBE_GB")):
        check_lfs_write_gate()


# ── Acceptance 5: production-body tests (fakes ONLY at the network boundary) ─


def test_production_body_blocked_arm(monkeypatch, hfapi_fake):
    """Real body end-to-end: UploadInfo construction, env parsing, per-repo

    loop, worst-wins fold, and the billing-context fetch on the blocked arm.
    """
    fake = _gate_with_side_effects(
        monkeypatch, [_http_error(BILLING_403_TEXT), _http_error(BILLING_403_TEXT)]
    )
    probe = check_lfs_write_gate()
    assert probe.verdict == "billing-blocked"
    assert probe.probe_gb == DEFAULT_BILLING_PROBE_GB
    assert probe.billing_context == {"canPay": True, "billingMode": "prepaid", "isPro": True}
    assert "credit recharge" in probe.detail
    # Per-repo loop hit both default repos with REAL UploadInfo instances.
    assert fake.call_count == 2
    for call in fake.call_args_list:
        infos = list(call.args[0])
        assert len(infos) == 1
        assert isinstance(infos[0], UploadInfo)
        assert len(infos[0].sha256) == 32
        assert len(infos[0].sample) == 512
        assert infos[0].size == int(DEFAULT_BILLING_PROBE_GB * 1e9)
        assert call.kwargs["transfers"] == ["basic", "multipart", "xet"]
        assert call.kwargs["repo_type"] == "model"
    probed = {call.kwargs["repo_id"] for call in fake.call_args_list}
    assert probed == {DEFAULT_MODEL_REPO, DEFAULT_OVERFLOW_REPO}


def test_production_body_billing_context_fail_soft(monkeypatch, hfapi_fake):
    """Concern 5: whoami raising degrades billing_context to None, verdict kept."""
    hfapi_fake.return_value.whoami.side_effect = RuntimeError("whoami down")
    _gate_with_side_effects(monkeypatch, _http_error(BILLING_403_TEXT))
    probe = check_lfs_write_gate()
    assert probe.verdict == "billing-blocked"
    assert probe.billing_context is None


def test_production_body_ok_arm_skips_billing_context(monkeypatch, hfapi_fake):
    _gate_with_side_effects(monkeypatch, [_ok_return(), _ok_return()])
    probe = check_lfs_write_gate()
    assert probe.verdict == "ok"
    assert probe.billing_context is None
    hfapi_fake.assert_not_called()


def test_explicit_repos_and_probe_gb_kwargs(monkeypatch, hfapi_fake):
    """kwargs override env/default: one custom repo, custom size."""
    fake = _gate_with_side_effects(monkeypatch, [_ok_return()])
    probe = check_lfs_write_gate(repos=(("someone/custom", "dataset"),), probe_gb=1.5)
    assert probe.verdict == "ok"
    assert probe.probe_gb == 1.5
    assert fake.call_count == 1
    assert fake.call_args.kwargs["repo_id"] == "someone/custom"
    assert fake.call_args.kwargs["repo_type"] == "dataset"
    assert next(iter(fake.call_args.args[0])).size == int(1.5e9)
