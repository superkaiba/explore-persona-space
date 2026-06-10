"""Tests for the pre-provision / pre-resume RunPod account hourly-spend guard
(``pod_lifecycle._assert_under_account_hourly_cap`` + the rate-estimation
helpers in ``runpod_api``).

Why this guard exists
---------------------
RunPod enforces a per-account hourly spending cap (set in the console). When
the projected sum-of-running-pod $/hr exceeds it, RunPod refuses the next
``podFindAndDeployOnDemand`` / ``podResume`` with
``INSUFFICIENT_BALANCE: Renting this pod would put you over your current
spending limit ($X/hr)`` — AFTER the user has already initiated the run.
Tasks #503 and #505 both blocked mid-run on 2026-06-05; the guard converts
this into a clean pre-flight refusal.

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


def test_guard_blocks_when_would_exceed_cap(monkeypatch):
    """Projected total > cap → SystemExit with a clear message naming the cap,
    current burn, and projected total."""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    # 18 H100s already running → 18 x $4 = $72. Adding 1 more 4xH100 pod = $16
    # → $88 projected, over the $80 default cap.
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info(f"pod-{i}", gpu_count=1) for i in range(18)],
    )
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-new",
            intended_gpu_type="H100",
            intended_gpu_count=4,
        )
    msg = str(exc.value)
    # Message contains all four key numbers + an actionable hint. The dollar
    # formatter (``$%6.2f``) pads small numbers with a leading space, so
    # match on the bare number rather than ``$N``.
    assert "pod-new" in msg
    assert "72.00" in msg  # current burn
    assert "16.00" in msg  # this-pod rate
    assert "88.00" in msg  # projected total
    assert "80.00" in msg  # cap
    assert "RUNPOD_ACCOUNT_HOURLY_CAP" in msg


def test_guard_env_cap_override(monkeypatch):
    """RUNPOD_ACCOUNT_HOURLY_CAP overrides the default."""
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.setenv("RUNPOD_ACCOUNT_HOURLY_CAP", "20.0")
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info("pod-1", gpu_count=4)],  # $16
    )
    # Adding another 4xH100 = $16, total $32, over the $20 cap.
    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-2",
            intended_gpu_type="H100",
            intended_gpu_count=4,
        )
    # The formatter pads with a leading space; match the bare cap value.
    assert "20.00" in str(exc.value)


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


def test_guard_api_failure_propagates(monkeypatch):
    """API unreachable → exception propagates (fail-loud); we cannot make the
    decision without live state, so refuse the operation."""
    from runpod_api import RunPodError

    def boom():
        raise RunPodError("Network error contacting RunPod: timeout")

    monkeypatch.setattr(runpod_api, "list_team_pods", boom)
    with pytest.raises(RunPodError):
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-9",
            intended_gpu_type="H100",
            intended_gpu_count=1,
        )
