"""Tests for the pre-provision / pre-resume RunPod account $/hr burn report
(``pod_lifecycle._assert_under_account_hourly_cap`` + the rate-estimation
helpers in ``runpod_api``).

History
-------
The function was a hard pre-flight guard mirroring the RunPod console
spending cap (#503/#505: clean refusal instead of a mid-run
``INSUFFICIENT_BALANCE``; managed-scope filter #1600). #2054 (user directive
2026-08-05) made it ADVISORY-ONLY: the team account is the shared
Anthropic-fellows/safety org pool (sponsored — the console cap is the
enforcement point), RunPod is the FIRST-resort router lane, and a local
dollar cap that can refuse or stall a provision sat in tension with the
standing no-dollar-budget-caps invariant
(``tests/test_no_dollar_budget_caps.py``). These tests pin the NEW contract:
the function NEVER raises — over-cap projections print a clearly-labelled
ADVISORY stderr note; an API failure logs an advisory skip.

These tests stub the live API at ``list_team_pods`` so they run offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
import runpod_api  # noqa: E402
from runpod_api import PodInfo  # noqa: E402


def _info(
    name: str,
    *,
    desired_status: str = "RUNNING",
    gpu_count: int = 1,
    gpu_type_id: str = "NVIDIA H100 80GB HBM3",
) -> PodInfo:
    return PodInfo(
        pod_id=f"id-{name}",
        name=name,
        desired_status=desired_status,
        gpu_count=gpu_count,
        gpu_type_id=gpu_type_id,
        ssh_host="1.2.3.4",
        ssh_port=12345,
        created_at="2026-06-05T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# runpod_api.estimate_pod_hourly_rate
# ---------------------------------------------------------------------------


def test_estimate_rate_known_gpu_uses_default_table(monkeypatch):
    """H100 single-GPU pod uses the conservative default ($4/hr/GPU)."""
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.delenv("RUNPOD_FALLBACK_HOURLY_PER_GPU_USD", raising=False)
    rate = runpod_api.estimate_pod_hourly_rate("NVIDIA H100 80GB HBM3", 1)
    assert rate == pytest.approx(4.0)


def test_estimate_rate_env_override_wins(monkeypatch):
    """RUNPOD_RATE_H100_USD overrides the table."""
    monkeypatch.setenv("RUNPOD_RATE_H100_USD", "2.5")
    rate = runpod_api.estimate_pod_hourly_rate("NVIDIA H100 80GB HBM3", 4)
    assert rate == pytest.approx(10.0)


def test_estimate_rate_unknown_gpu_uses_fallback(monkeypatch):
    """Unknown GPU type falls back to the conservative per-GPU rate (default 6.0)."""
    monkeypatch.delenv("RUNPOD_FALLBACK_HOURLY_PER_GPU_USD", raising=False)
    rate = runpod_api.estimate_pod_hourly_rate("NVIDIA L4", 2)
    assert rate == pytest.approx(12.0)


def test_estimate_rate_unknown_gpu_env_fallback(monkeypatch):
    """RUNPOD_FALLBACK_HOURLY_PER_GPU_USD env overrides the fallback rate."""
    monkeypatch.setenv("RUNPOD_FALLBACK_HOURLY_PER_GPU_USD", "10")
    rate = runpod_api.estimate_pod_hourly_rate("NVIDIA L4", 2)
    assert rate == pytest.approx(20.0)


def test_estimate_rate_zero_gpus_is_zero():
    """No GPUs assigned (None or 0) → 0.0; never raises."""
    assert runpod_api.estimate_pod_hourly_rate("H100", None) == 0.0
    assert runpod_api.estimate_pod_hourly_rate("H100", 0) == 0.0


def test_estimate_rate_bad_env_falls_back(monkeypatch):
    """A malformed env value falls through to the default rather than crashing."""
    monkeypatch.setenv("RUNPOD_RATE_H100_USD", "not-a-number")
    rate = runpod_api.estimate_pod_hourly_rate("H100", 1)
    # Should fall through to the default table value, not raise.
    assert rate == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# runpod_api.current_account_hourly_burn
# ---------------------------------------------------------------------------


def test_current_burn_sums_only_running(monkeypatch):
    """EXITED pods don't accrue $/hr; only RUNNING pods enter the sum."""
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [
            _info("pod-1", gpu_count=1),  # RUNNING H100 → $4
            _info("pod-2", gpu_count=8, gpu_type_id="NVIDIA H200"),  # RUNNING 8xH200 → $44
            _info("pod-stopped", desired_status="EXITED", gpu_count=8),  # excluded
        ],
    )
    total, breakdown = runpod_api.current_account_hourly_burn()
    assert total == pytest.approx(4.0 + 8 * 5.5)
    names = [name for name, _ in breakdown]
    assert "pod-stopped" not in names
    # Breakdown sorts cost-descending.
    assert breakdown[0][1] >= breakdown[-1][1]


def test_current_burn_empty(monkeypatch):
    """No live pods → zero burn, empty breakdown."""
    monkeypatch.setattr(runpod_api, "list_team_pods", lambda: [])
    total, breakdown = runpod_api.current_account_hourly_burn()
    assert total == 0.0
    assert breakdown == []


# ---------------------------------------------------------------------------
# pod_lifecycle._assert_under_account_hourly_cap
# ---------------------------------------------------------------------------


def test_guard_passes_when_under_cap(monkeypatch):
    """Projected total ≤ cap → returns silently, no SystemExit."""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    # 1 running H100 ($4) + adding 1 H100 ($4) = $8 — well under $80.
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info("pod-1", gpu_count=1)],
    )
    # Returns None (no raise) — that's the contract.
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-2",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )
        is None
    )


def test_guard_over_cap_is_advisory_only(monkeypatch, capsys):
    """#2054: projected total > cap → NO raise; a clearly-labelled ADVISORY
    stderr line names the key numbers + the override env knob. (Replaces
    test_guard_blocks_when_would_exceed_cap.)"""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.delenv("EPM_RUNPOD_BURN_SCOPE", raising=False)
    # 18 H100s already running → 18 x $4 = $72. Adding 1 more 4xH100 pod = $16
    # → $88 projected, over the $80 default cap.
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info(f"pod-{i}", gpu_count=1) for i in range(18)],
    )
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-new",
            intended_gpu_type="H100",
            intended_gpu_count=4,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "ADVISORY" in err
    assert "never blocks" in err
    assert "pod-new" in err
    assert "72.00" in err  # current burn
    assert "16.00" in err  # this-pod rate
    assert "88.00" in err  # projected total
    assert "80.00" in err  # cap (local mirror)
    assert "RUNPOD_ACCOUNT_HOURLY_CAP" in err


def test_guard_resume_verb_advisory(monkeypatch, capsys):
    """The resume-verb over-cap projection is advisory too (#2054) — same
    contract as provision. (Replaces
    test_guard_resume_refusal_advertises_wait_for_capacity_hint, whose
    --wait-for-capacity refusal-remedy hint died with the refusal.)"""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.delenv("EPM_RUNPOD_BURN_SCOPE", raising=False)
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info(f"pod-{i}", gpu_count=1) for i in range(18)],
    )
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="resume",
            pod_label="pod-old",
            intended_gpu_type="H100",
            intended_gpu_count=4,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "ADVISORY" in err
    assert "resume pod-old" in err


def test_guard_env_cap_override(monkeypatch, capsys):
    """RUNPOD_ACCOUNT_HOURLY_CAP still overrides the advisory mirror."""
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.delenv("EPM_RUNPOD_BURN_SCOPE", raising=False)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "20.0")
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info("pod-1", gpu_count=4)],  # $16
    )
    # Adding another 4xH100 = $16, total $32, over the $20 mirror → advisory.
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-2",
            intended_gpu_type="H100",
            intended_gpu_count=4,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "ADVISORY" in err
    assert "20.00" in err


def test_guard_skip_for_same_pod_excludes_sibling(monkeypatch):
    """``skip_for_same_pod`` excludes a RUNNING sibling sharing the pod's name
    from the current-burn sum (defense against duplicate-provision races on
    resume).
    """
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [
            _info("pod-7", gpu_count=8, gpu_type_id="NVIDIA H200"),  # $44 — to skip
            _info("pod-8", gpu_count=4),  # $16
        ],
    )
    # With pod-7 skipped, current burn = $16; adding a 4xH100 = $16 → $32
    # projected, under the $80 cap.
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="resume",
            pod_label="pod-7",
            intended_gpu_type="H100",
            intended_gpu_count=4,
            skip_for_same_pod="pod-7",
        )
        is None
    )


def test_guard_api_failure_is_advisory_skip(monkeypatch, capsys):
    """#2054: API unreachable → NO raise; the skip is logged loudly as an
    ADVISORY (the provision/resume call right after hits the same API and
    fails loud if it is genuinely down — nothing silently swallowed).
    (Replaces test_guard_api_failure_propagates.)"""
    from runpod_api import RunPodError

    def boom():
        raise RunPodError("Network error contacting RunPod: timeout")

    monkeypatch.setattr(runpod_api, "list_team_pods", boom)
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-9",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "ADVISORY" in err
    assert "unavailable" in err
    assert "RunPodError" in err


# ---------------------------------------------------------------------------
# EPM_RUNPOD_BURN_SCOPE — managed-scope guard on the shared team account (#1600)
# ---------------------------------------------------------------------------
#
# The RunPod team account is shared with the Anthropic-fellows cluster fleet:
# ~80+ unmanaged pods whose burn permanently exceeds any sane local cap
# (13/13 wait-for-capacity refusals on #779, 2026-07-22). The guard therefore
# scopes its burn sum to MANAGED pods (`pod-*` / `epm-issue-*`) by default;
# `EPM_RUNPOD_BURN_SCOPE=all` restores the account-wide sum. Unmanaged
# fixture names below are verbatim from the #779 marker transcript.


def _fellows_fleet() -> list[PodInfo]:
    """Three unmanaged fellows-cluster pods, 8x H200 each ($44/hr) → $132/hr
    total, exceeding the $120 cap of the #779 shape on their own."""
    return [
        _info("Anthropic 2-pod-5-m9a", gpu_count=8, gpu_type_id="NVIDIA H200"),
        _info("cluster-EUR-IS-pod-5", gpu_count=8, gpu_type_id="NVIDIA H200"),
        _info("styfeng_temp_48hr_C", gpu_count=8, gpu_type_id="NVIDIA H200"),
    ]


def _delenv_rates_and_scope(monkeypatch):
    """Deterministic env for the scope tests: no rate overrides, default scope."""
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H200_USD", raising=False)
    monkeypatch.delenv("EPM_RUNPOD_BURN_SCOPE", raising=False)


def test_is_managed_pod_name_matches_prefixes():
    """The string-level twin recognizes canonical + suffixed + legacy managed
    names and rejects the fellows-fleet names (verbatim from #779)."""
    assert pod_lifecycle._is_managed_pod_name("pod-779")
    assert pod_lifecycle._is_managed_pod_name("pod-825-followup")
    assert pod_lifecycle._is_managed_pod_name("epm-issue-12")
    assert not pod_lifecycle._is_managed_pod_name("Anthropic 2-pod-5-m9a")
    assert not pod_lifecycle._is_managed_pod_name("cluster-EUR-IS-pod-5")
    assert not pod_lifecycle._is_managed_pod_name("styfeng_temp_48hr_C")
    assert not pod_lifecycle._is_managed_pod_name("")


def test_guard_default_scope_excludes_unmanaged_pods(monkeypatch):
    """The #779 regression: unmanaged pods summing over the cap + one managed
    $4/hr intent → the guard PASSES under the default (managed) scope."""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "120")
    monkeypatch.setattr(pod_lifecycle, "_unmanaged_burn_warned", False)
    monkeypatch.setattr(runpod_api, "list_team_pods", _fellows_fleet)
    # Managed burn $0 + this pod $4 = $4 ≤ $120 → None, despite $132 unmanaged.
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-779",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )
        is None
    )


def test_guard_transient_default_scope_excludes_unmanaged(monkeypatch):
    """The exact wait-loop path that refused 13/13 on #779: same fixture with
    ``transient_on_exceed=True`` → returns None (no transient raise)."""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "120")
    monkeypatch.setattr(pod_lifecycle, "_unmanaged_burn_warned", False)
    monkeypatch.setattr(runpod_api, "list_team_pods", _fellows_fleet)
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-779",
            intended_gpu_type="H100",
            intended_gpu_count=1,
            transient_on_exceed=True,
        )
        is None
    )


def test_guard_scope_all_over_cap_is_advisory(monkeypatch, capsys):
    """``EPM_RUNPOD_BURN_SCOPE=all`` restores the account-wide SUM — but since
    #2054 an over-cap projection is advisory, never a refusal. (Replaces
    test_guard_scope_all_restores_account_wide_behavior.)"""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "120")
    monkeypatch.setenv("EPM_RUNPOD_BURN_SCOPE", "all")
    monkeypatch.setattr(runpod_api, "list_team_pods", _fellows_fleet)
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-779",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "ADVISORY" in err
    assert "132.00" in err  # account-wide current burn
    assert "all scope" in err
    assert "120.00" in err  # cap (local mirror)


def test_guard_scope_all_transient_is_advisory_too(monkeypatch, capsys):
    """#2054: the wait-loop keyword mode is advisory as well under ``all``
    scope — nothing raises. (Replaces
    test_guard_scope_all_transient_raises_insufficient_balance.)"""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "120")
    monkeypatch.setenv("EPM_RUNPOD_BURN_SCOPE", "all")
    monkeypatch.setattr(runpod_api, "list_team_pods", _fellows_fleet)
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-779",
            intended_gpu_type="H100",
            intended_gpu_count=1,
            transient_on_exceed=True,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "ADVISORY" in err
    assert "[all scope]" in err
    assert "120.00" in err


def test_guard_bad_scope_value_falls_back_to_managed(monkeypatch, capsys):
    """A garbage ``EPM_RUNPOD_BURN_SCOPE`` falls back to ``managed`` with a
    stderr WARN (mirrors ``_account_hourly_cap_usd``), never crashes."""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "120")
    monkeypatch.setenv("EPM_RUNPOD_BURN_SCOPE", "frobnicate")
    monkeypatch.setattr(pod_lifecycle, "_unmanaged_burn_warned", False)
    monkeypatch.setattr(runpod_api, "list_team_pods", _fellows_fleet)
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-779",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )
        is None
    )
    err = capsys.readouterr().err
    assert "EPM_RUNPOD_BURN_SCOPE" in err
    assert "frobnicate" in err
    assert "using 'managed'" in err


def test_guard_unmanaged_warn_emitted_once_per_process(monkeypatch, capsys):
    """When unmanaged burn alone exceeds the cap under managed scope, a stderr
    WARN names the exclusion — once per process (latch), because the wait loop
    re-runs the guard every backoff tick."""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "120")
    monkeypatch.setattr(pod_lifecycle, "_unmanaged_burn_warned", False)
    monkeypatch.setattr(runpod_api, "list_team_pods", _fellows_fleet)

    def call():
        return pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-779",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )

    assert call() is None
    first = capsys.readouterr().err
    assert "UNMANAGED" in first
    assert "132.00" in first  # unmanaged total
    assert "EPM_RUNPOD_BURN_SCOPE=all" in first
    assert call() is None
    second = capsys.readouterr().err
    assert "UNMANAGED" not in second  # latched


def test_guard_skip_for_same_pod_still_works_under_managed_scope(monkeypatch):
    """``skip_for_same_pod`` still subtracts a managed sibling under the
    default scope; unmanaged noise rows are ignored by the filter."""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.setattr(pod_lifecycle, "_unmanaged_burn_warned", False)
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [
            _info("pod-7", gpu_count=8, gpu_type_id="NVIDIA H200"),  # $44 — to skip
            _info("pod-8", gpu_count=4),  # $16
            *_fellows_fleet(),  # $132 unmanaged noise, excluded from the sum
        ],
    )
    # Managed burn $60 minus skipped $44 = $16; adding a 4xH100 = $16 → $32
    # projected, under the $80 default cap (unmanaged $132 ignored).
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="resume",
            pod_label="pod-7",
            intended_gpu_type="H100",
            intended_gpu_count=4,
            skip_for_same_pod="pod-7",
        )
        is None
    )


def test_guard_advisory_shows_unmanaged_exclusion_line(monkeypatch, capsys):
    """A managed-scope over-cap ADVISORY (#2054) still shows the full account
    picture: managed sums, the excluded-unmanaged summary (count + $/hr), and
    the account-wide total. (Replaces
    test_guard_systemexit_message_shows_unmanaged_exclusion_line.)"""
    _delenv_rates_and_scope(monkeypatch)
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.setattr(pod_lifecycle, "_unmanaged_burn_warned", False)
    # 18 managed H100s = $72; adding 4xH100 = $16 → $88 > $80 default cap.
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info(f"pod-{i}", gpu_count=1) for i in range(18)] + _fellows_fleet(),
    )
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-new",
            intended_gpu_type="H100",
            intended_gpu_count=4,
        )
        is None
    )
    err = capsys.readouterr().err
    # The pre-#1600 pinned literals survive (managed-only sums).
    assert "ADVISORY" in err
    assert "72.00" in err  # current burn (managed scope)
    assert "88.00" in err  # projected total
    assert "80.00" in err  # cap (local mirror)
    assert "managed scope" in err
    # The exclusion summary keeps the shared account visible.
    assert "excluded: 3 unmanaged team pod(s)" in err
    assert "132.00" in err  # unmanaged $/hr
    assert "204.00" in err  # account-wide total (72 + 132)
