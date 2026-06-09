"""Tests for the INSUFFICIENT_BALANCE retry-with-backoff classification.

Why this exists
---------------
RunPod refuses ``podFindAndDeployOnDemand`` / ``podResume`` with an
``INSUFFICIENT_BALANCE`` GraphQL error when the projected total account
$/hr would exceed the console-side spending cap. The refusal is
**transient + no-cost-while-idle**: while the provision/resume is
refused, NOTHING is running, so no $/hr is being spent. The condition
clears the moment any other pod on the team frees $/hr headroom (a
sibling pod stop/terminate, or an experiment finishing).

Before #506 (2026-06-08) the failure was classified as a generic
``RunPodError`` and surfaced through ``epm:failure infra``, eventually
stranding the task at ``status:blocked`` even though the refusal would
have cleared on its own. The fix:

1. ``runpod_api`` raises a new typed exception
   :class:`RunPodInsufficientBalanceError` when the GraphQL ``errors``
   payload contains the INSUFFICIENT_BALANCE marker.
2. ``pod_lifecycle.create_pod_with_wait_for_capacity`` catches BOTH
   :class:`RunPodNoCapacityError` and the new
   :class:`RunPodInsufficientBalanceError` in its unbounded retry loop
   (same retry-with-backoff semantics — both are transient +
   no-cost-while-idle).
3. ``pod_lifecycle.cmd_resume`` wraps the resume call in
   :func:`_resume_with_balance_wait_if_autonomous`: autonomous-mode
   sessions wait + retry, interactive-mode runs surface a clear
   actionable message instead of a bare stack trace.

This module pins all three layers. ``time.sleep`` is monkeypatched
throughout so the test suite runs instantly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_lifecycle  # noqa: E402
import runpod_api  # noqa: E402
from pod_lifecycle import create_pod_with_wait_for_capacity  # noqa: E402
from runpod_api import (  # noqa: E402
    PodInfo,
    RunPodError,
    RunPodInsufficientBalanceError,
    RunPodNoCapacityError,
    _is_insufficient_balance_error,
)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Never actually sleep during backoff in tests."""
    monkeypatch.setattr(pod_lifecycle.time, "sleep", lambda _secs: None)
    monkeypatch.setattr(runpod_api.time, "sleep", lambda _secs: None)


def _patch_urlopen(monkeypatch, *, returns_body: bytes):
    """Patch ``urlopen`` + ``_require_env`` so ``_graphql_once`` runs offline.

    Same shape as test_runpod_api_retry.py's helper but stripped to the
    success-response path we need here (we drive INSUFFICIENT_BALANCE
    through the GraphQL ``errors`` field, not HTTP-level failures).
    """
    monkeypatch.setattr(runpod_api, "_require_env", lambda: ("k", "t"))

    class _Resp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, timeout=60):
        return _Resp(returns_body)

    monkeypatch.setattr(runpod_api.urlrequest, "urlopen", fake_urlopen)


def _make_pod_info(pod_id: str = "p1", name: str = "pod-1") -> PodInfo:
    return PodInfo(
        pod_id=pod_id,
        name=name,
        desired_status="RUNNING",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=None,
        ssh_port=None,
        created_at="2026-06-08T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# 1. _is_insufficient_balance_error helper — marker detection
# ---------------------------------------------------------------------------


def test_marker_detect_explicit_code():
    """The literal ``INSUFFICIENT_BALANCE`` error code is detected."""
    msg = (
        "GraphQL errors: "
        '[{"message": "INSUFFICIENT_BALANCE: Renting this pod would put '
        'you over your current spending limit ($80/hr)"}]'
    )
    assert _is_insufficient_balance_error(msg) is True


def test_marker_detect_human_phrase():
    """The human-readable ``spending limit`` phrase is detected too."""
    assert _is_insufficient_balance_error("over your current spending limit ($80/hr)") is True
    assert (
        _is_insufficient_balance_error("Renting this pod would exceed the spending limit") is True
    )


def test_marker_case_insensitive():
    """Marker detection is case-insensitive (matches RunPod's mixed-case messages)."""
    assert _is_insufficient_balance_error("Insufficient_Balance: foo") is True
    assert _is_insufficient_balance_error("INSUFFICIENT BALANCE error") is True


def test_marker_false_for_unrelated_errors():
    """A plain auth / supply / network error must NOT classify as insufficient-balance."""
    assert _is_insufficient_balance_error("HTTP 401: invalid api key") is False
    assert _is_insufficient_balance_error("SUPPLY_CONSTRAINT: no free GPUs") is False
    assert _is_insufficient_balance_error("connection refused") is False
    assert _is_insufficient_balance_error("") is False


# ---------------------------------------------------------------------------
# 2. _graphql_once raises the typed exception on INSUFFICIENT_BALANCE
# ---------------------------------------------------------------------------


def test_graphql_once_insufficient_balance_raises_typed(monkeypatch):
    """A GraphQL ``errors`` payload containing INSUFFICIENT_BALANCE raises
    :class:`RunPodInsufficientBalanceError`, NOT the bare ``RunPodError``.
    This is what lets the pod_lifecycle retry loop classify the failure by
    class rather than string-sniffing the message later."""
    body = json.dumps(
        {
            "errors": [
                {
                    "message": "INSUFFICIENT_BALANCE: Renting this pod would "
                    "put you over your current spending limit ($80/hr)"
                }
            ]
        }
    ).encode("utf-8")
    _patch_urlopen(monkeypatch, returns_body=body)
    with pytest.raises(RunPodInsufficientBalanceError) as exc:
        runpod_api._graphql_once("q", None, 60)
    assert "INSUFFICIENT_BALANCE" in str(exc.value) or "spending limit" in str(exc.value)


def test_graphql_once_insufficient_balance_is_runpod_error_subclass(monkeypatch):
    """The new class is a :class:`RunPodError` subclass so existing
    ``except RunPodError`` callers keep catching it (defense in depth — the
    retry loop catches the typed class first, but any legacy caller that
    only catches the base class still handles it after retry exhaustion)."""
    body = json.dumps({"errors": [{"message": "INSUFFICIENT_BALANCE: x"}]}).encode("utf-8")
    _patch_urlopen(monkeypatch, returns_body=body)
    with pytest.raises(RunPodError):
        runpod_api._graphql_once("q", None, 60)


def test_graphql_once_other_graphql_error_stays_generic(monkeypatch):
    """A GraphQL ``errors`` payload WITHOUT INSUFFICIENT_BALANCE must raise
    the generic :class:`RunPodError` (not the new subclass) so it still
    fails fast at the policy layer."""
    body = json.dumps({"errors": [{"message": "AUTH_FAILED: invalid token"}]}).encode("utf-8")
    _patch_urlopen(monkeypatch, returns_body=body)
    with pytest.raises(RunPodError) as exc:
        runpod_api._graphql_once("q", None, 60)
    assert not isinstance(exc.value, RunPodInsufficientBalanceError)


# ---------------------------------------------------------------------------
# 3. create_pod_with_wait_for_capacity retries on INSUFFICIENT_BALANCE
# ---------------------------------------------------------------------------


def _make_create_pod_stub(monkeypatch, outcomes: list):
    """Patch ``pod_lifecycle.create_pod`` to a recorder consuming ``outcomes``
    one per call. Each entry is either an Exception (raised) or a PodInfo
    (returned). Mirror of the helper in test_pod_wait_for_capacity.py."""

    class _Rec:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    rec = _Rec()
    monkeypatch.setattr(pod_lifecycle, "create_pod", rec)
    return rec


def test_wait_loop_retries_on_insufficient_balance_then_succeeds(monkeypatch):
    """Two INSUFFICIENT_BALANCE refusals, then success — the loop returns
    the eventual PodInfo. The bug fixed here: before #506, the first
    INSUFFICIENT_BALANCE propagated out of the retry loop as a generic
    ``RunPodError`` and crashed the loop, eventually stranding the task at
    ``status:blocked`` even though the refusal would have cleared on its
    own."""
    info = _make_pod_info()
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodInsufficientBalanceError("GraphQL errors: INSUFFICIENT_BALANCE: spending limit"),
            RunPodInsufficientBalanceError("GraphQL errors: INSUFFICIENT_BALANCE: spending limit"),
            info,
        ],
    )

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )
    assert out is info
    assert rec.calls == 3


def test_wait_loop_mixes_no_capacity_and_insufficient_balance(monkeypatch):
    """Both transient classes (no-capacity AND insufficient-balance) are
    caught by the SAME loop — a mixed sequence still retries through both
    and lands on success."""
    info = _make_pod_info()
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodNoCapacityError("no capacity attempt 1"),
            RunPodInsufficientBalanceError("INSUFFICIENT_BALANCE attempt 2"),
            RunPodNoCapacityError("no capacity attempt 3"),
            info,
        ],
    )
    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )
    assert out is info
    assert rec.calls == 4


def test_wait_loop_does_not_retry_generic_runpod_error(monkeypatch):
    """A plain :class:`RunPodError` (auth, bad config, transport-budget-
    exhausted, empty-gpu-list) must STILL propagate immediately — adding
    INSUFFICIENT_BALANCE to the retry set must not accidentally widen the
    catch to the generic base class."""
    rec = _make_create_pod_stub(
        monkeypatch,
        [
            RunPodError("HTTP 401: invalid api key"),
            # Sentinel: if the loop wrongly retries, this would silently
            # succeed. We assert calls == 1 below to catch that regression.
            _make_pod_info(),
        ],
    )
    with pytest.raises(RunPodError) as exc:
        create_pod_with_wait_for_capacity(
            name="pod-1",
            gpu_type="H100",
            gpu_count=1,
            volume_gb=200,
            container_disk_gb=50,
        )
    assert "401" in str(exc.value)
    assert not isinstance(exc.value, RunPodInsufficientBalanceError)
    assert not isinstance(exc.value, RunPodNoCapacityError)
    assert rec.calls == 1


# ---------------------------------------------------------------------------
# 4. _resume_with_balance_wait_if_autonomous routing
# ---------------------------------------------------------------------------


def _make_ephemeral_pod(name: str = "pod-1", pod_id: str = "p1") -> object:
    """Synthesize a minimal EphemeralPod-like object the resume helper can
    drive. The helper only reads ``pod.pod_id`` and ``pod.gpu_count``, so a
    tiny stand-in is enough — avoids fixture coupling to the full state-
    loading machinery."""

    class _Stub:
        pass

    stub = _Stub()
    stub.pod_id = pod_id
    stub.gpu_count = 1
    stub.name = name
    return stub


def test_resume_autonomous_retries_on_insufficient_balance(monkeypatch):
    """In autonomous mode, INSUFFICIENT_BALANCE on resume → retry-with-
    backoff (same as the create-pod wait loop), NOT fail-loud SystemExit."""
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    pod = _make_ephemeral_pod()

    calls = {"n": 0}

    def fake_resume_pod(pod_id, gpu_count):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RunPodInsufficientBalanceError("INSUFFICIENT_BALANCE: spending limit")
        return _make_pod_info()

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    pod_lifecycle._resume_with_balance_wait_if_autonomous(pod=pod, name="pod-1", issue=506)
    assert calls["n"] == 3  # retried twice, succeeded on attempt 3


def test_resume_interactive_fails_loud_on_insufficient_balance(monkeypatch):
    """In interactive mode (no EPM_AUTONOMOUS_SESSION), INSUFFICIENT_BALANCE
    on resume → SystemExit with an actionable message naming the next
    steps. We do NOT silently retry forever in interactive runs because the
    human is sitting at the terminal expecting a verdict."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    pod = _make_ephemeral_pod()

    def fake_resume_pod(pod_id, gpu_count):
        raise RunPodInsufficientBalanceError("INSUFFICIENT_BALANCE: spending limit")

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._resume_with_balance_wait_if_autonomous(pod=pod, name="pod-1", issue=506)
    msg = str(exc.value)
    # Message names what happened + at least one actionable next step.
    assert "INSUFFICIENT_BALANCE" in msg or "spending cap" in msg
    assert "stop or terminate" in msg or "raise the console cap" in msg
    assert "506" in msg  # issue id surfaced for the re-run command


def test_resume_supply_constraint_still_fails_loud(monkeypatch):
    """SUPPLY_CONSTRAINT on resume is unchanged from prior behavior — fails
    loud in BOTH modes because resume never relocates the pod (waiting
    can't help when the original host itself is out of GPUs). This pins
    that the new INSUFFICIENT_BALANCE branch did NOT accidentally swallow
    the supply-constraint path."""
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    pod = _make_ephemeral_pod()

    def fake_resume_pod(pod_id, gpu_count):
        raise RunPodError("podResume returned null for p1")  # supply-constraint marker

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._resume_with_balance_wait_if_autonomous(pod=pod, name="pod-1", issue=506)
    msg = str(exc.value)
    assert "supply constraint" in msg.lower() or "no free gpus" in msg.lower()


def test_resume_other_runpod_error_propagates(monkeypatch):
    """A plain RunPodError that is NEITHER insufficient-balance NOR supply-
    constraint must propagate (auth failure, bad config). Adding the new
    branch must not accidentally swallow real errors."""
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    pod = _make_ephemeral_pod()

    def fake_resume_pod(pod_id, gpu_count):
        raise RunPodError("HTTP 401: invalid api key")

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    with pytest.raises(RunPodError) as exc:
        pod_lifecycle._resume_with_balance_wait_if_autonomous(pod=pod, name="pod-1", issue=506)
    assert "401" in str(exc.value)
    assert not isinstance(exc.value, SystemExit)


# ---------------------------------------------------------------------------
# 5. local-cap guard inside the wait loop (#506 first-block fix)
# ---------------------------------------------------------------------------
#
# What this section pins
# ----------------------
# The candidate landed by commit 682dec97b made the API-side
# ``INSUFFICIENT_BALANCE`` refusal transient (the retry loop catches it). But
# the LOCAL ``_assert_under_account_hourly_cap`` guard was still an
# unconditional ``SystemExit`` pre-call in ``cmd_provision`` / ``cmd_resume``,
# so a fleet-near-the-cap autonomous run hard-exited to ``blocked`` BEFORE the
# wait loop ever started — exactly what produced the #506 FIRST block at
# 03:43Z 2026-06-08 (``pod_cap_blocked``).
#
# The follow-up fix:
#   1. ``_assert_under_account_hourly_cap`` takes ``transient_on_exceed=True``
#      → raises ``RunPodInsufficientBalanceError`` instead of ``SystemExit``.
#      Default ``False`` preserves byte-identical behavior for every existing
#      caller (one-shot / interactive contract from #503/#505 is untouched).
#   2. ``create_pod_with_wait_for_capacity`` /
#      ``_resume_with_balance_wait_if_autonomous`` accept a ``preflight_check``
#      callable invoked at the TOP of each loop attempt. The same except clause
#      catches its ``RunPodInsufficientBalanceError`` and retries with backoff,
#      so freed $/hr headroom is detected without operator intervention.
#   3. ``cmd_provision`` / ``cmd_resume`` route the guard through the wait
#      loop in wait mode and keep the unconditional ``SystemExit`` pre-call in
#      one-shot mode.
#
# These tests pin (a) the new keyword preserves the legacy contract, (b) wait
# mode actually retries the local-cap exception until a mock frees headroom,
# (c) one-shot mode still raises ``SystemExit``.


def _info(name: str, *, gpu_count: int = 1) -> PodInfo:
    """Minimal RUNNING H100 stand-in for ``current_account_hourly_burn``
    (which sums ``list_team_pods`` filtered to RUNNING). Shape borrowed from
    test_pod_lifecycle_account_spend_guard.py so the rate table picks up
    $4/hr/H100."""
    return PodInfo(
        pod_id=f"id-{name}",
        name=name,
        desired_status="RUNNING",
        gpu_count=gpu_count,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=None,
        ssh_port=None,
        created_at="2026-06-08T00:00:00Z",
    )


def test_assert_guard_default_still_systemexits_on_over_cap(monkeypatch):
    """``transient_on_exceed=False`` (default) preserves the byte-identical
    ``SystemExit`` contract for every pre-existing caller — the one-shot
    / interactive contract from #503/#505 is untouched. This pin protects
    every test in test_pod_lifecycle_account_spend_guard.py that asserts
    ``raises(SystemExit)`` against an accidental future flip of the
    default."""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    # 18 RUNNING H100s ($72/hr) + adding 4 more ($16/hr) = $88 projected,
    # over the $80 default cap. Identical setup to
    # test_guard_blocks_when_would_exceed_cap so the legacy assertion is
    # exercised verbatim.
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
    # Actionable message still names the cap + override env knob.
    assert "RUNPOD_ACCOUNT_HOURLY_CAP" in str(exc.value)


def test_assert_guard_transient_raises_runpod_exception_on_over_cap(monkeypatch):
    """``transient_on_exceed=True`` swaps the SystemExit for
    ``RunPodInsufficientBalanceError`` so a calling retry loop can treat
    the local guard the same as the live-API refusal — both clear when a
    sibling pod frees $/hr headroom. This is the load-bearing piece of
    the #506 first-block fix."""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info(f"pod-{i}", gpu_count=1) for i in range(18)],
    )
    with pytest.raises(RunPodInsufficientBalanceError) as exc:
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-new",
            intended_gpu_type="H100",
            intended_gpu_count=4,
            transient_on_exceed=True,
        )
    # NOT a SystemExit — the loop has to be able to ``except`` it.
    assert not isinstance(exc.value, SystemExit)
    # The transient message still names the dollar numbers so the wait
    # loop heartbeat reads usefully.
    msg = str(exc.value)
    assert "exceeds cap" in msg
    assert "80.00" in msg


def test_assert_guard_transient_under_cap_returns_quietly(monkeypatch):
    """``transient_on_exceed`` only changes the OVER-cap branch; an under-cap
    projection still returns ``None`` quietly so a healthy wait-loop tick
    proceeds to ``create_pod``."""
    monkeypatch.delenv("RUNPOD_ACCOUNT_HOURLY_CAP", raising=False)
    monkeypatch.delenv("RUNPOD_RATE_H100_USD", raising=False)
    monkeypatch.setattr(
        runpod_api,
        "list_team_pods",
        lambda: [_info("pod-1", gpu_count=1)],  # $4 burn, $8 projected — fine
    )
    assert (
        pod_lifecycle._assert_under_account_hourly_cap(
            verb="provision",
            pod_label="pod-2",
            intended_gpu_type="H100",
            intended_gpu_count=1,
            transient_on_exceed=True,
        )
        is None
    )


def test_wait_loop_retries_local_cap_preflight_until_headroom_frees(monkeypatch):
    """The wait loop invokes ``preflight_check`` at the TOP of each attempt;
    when it raises ``RunPodInsufficientBalanceError`` the loop retries with
    backoff, and the moment a mock frees headroom (the preflight stops
    raising) the next attempt proceeds to ``create_pod`` and returns.

    This is the regression test for the #506 first-block gap: the local
    cap guard hard-exiting the autonomous run BEFORE the wait loop
    started."""
    info = _make_pod_info()

    # ``create_pod`` always succeeds; the gate is the preflight.
    rec = _make_create_pod_stub(monkeypatch, [info])

    # Preflight raises the local-cap exception twice, then stops raising
    # — modelling a sibling pod that frees $/hr headroom on the 3rd tick.
    preflight_calls = {"n": 0}

    def fake_preflight():
        preflight_calls["n"] += 1
        if preflight_calls["n"] < 3:
            raise RunPodInsufficientBalanceError(
                f"local pre-flight: projected $88.00/hr exceeds cap $80.00/hr "
                f"(attempt {preflight_calls['n']})"
            )

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
        preflight_check=fake_preflight,
    )
    assert out is info
    # The preflight ran 3 times; ``create_pod`` ran exactly once (on the
    # successful 3rd tick). The first two ticks short-circuited before
    # touching ``create_pod`` — i.e. the loop did NOT bypass the guard.
    assert preflight_calls["n"] == 3
    assert rec.calls == 1


def test_wait_loop_preflight_none_default_unchanged(monkeypatch):
    """Omitting ``preflight_check`` (the default for any pre-existing
    caller) does NOT invoke any guard. Verified by stubbing the guard to
    blow up if called and confirming a happy-path wait-loop run still
    returns."""
    info = _make_pod_info()
    _make_create_pod_stub(monkeypatch, [info])

    # If the loop ever calls the guard without an explicit ``preflight_check``,
    # this raises and the test fails — defence in depth against a future
    # refactor that wires the guard in unconditionally.
    def explode(*args, **kwargs):
        raise AssertionError(
            "wait loop must NOT invoke _assert_under_account_hourly_cap when "
            "preflight_check is None — that would silently re-introduce the "
            "pre-call SystemExit gap closed by the #506 fix"
        )

    monkeypatch.setattr(pod_lifecycle, "_assert_under_account_hourly_cap", explode)

    out = create_pod_with_wait_for_capacity(
        name="pod-1",
        gpu_type="H100",
        gpu_count=1,
        volume_gb=200,
        container_disk_gb=50,
    )
    assert out is info


def test_wait_loop_preflight_other_runpod_error_propagates(monkeypatch):
    """``preflight_check`` raising a non-INSUFFICIENT_BALANCE exception
    (a plain ``RunPodError``, an ``OSError``, or an unrelated exception)
    propagates out of the wait loop — it is NOT silently swallowed. This
    pins that widening the catch to the new local-cap path did not
    accidentally hide real failures."""
    _make_create_pod_stub(monkeypatch, [_make_pod_info()])

    def fake_preflight():
        # Anything OTHER than RunPodInsufficientBalanceError (and the
        # already-caught RunPodNoCapacityError) must propagate.
        raise RunPodError("preflight: bad config")

    with pytest.raises(RunPodError) as exc:
        create_pod_with_wait_for_capacity(
            name="pod-1",
            gpu_type="H100",
            gpu_count=1,
            volume_gb=200,
            container_disk_gb=50,
            preflight_check=fake_preflight,
        )
    assert "bad config" in str(exc.value)
    assert not isinstance(exc.value, RunPodInsufficientBalanceError)


def test_resume_wait_helper_preflight_retries_under_autonomous(monkeypatch):
    """The resume-side helper mirrors the provision wait loop: when
    autonomous, a ``preflight_check`` raising local-cap is retried with
    backoff until it stops raising, then ``resume_pod`` is called once.
    Closes the #506 first-block gap on the resume code path."""
    monkeypatch.setenv("EPM_AUTONOMOUS_SESSION", "1")
    pod = _make_ephemeral_pod()

    resume_calls = {"n": 0}

    def fake_resume_pod(pod_id, gpu_count):
        resume_calls["n"] += 1
        return _make_pod_info()

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    preflight_calls = {"n": 0}

    def fake_preflight():
        preflight_calls["n"] += 1
        if preflight_calls["n"] < 3:
            raise RunPodInsufficientBalanceError("local pre-flight over cap")

    pod_lifecycle._resume_with_balance_wait_if_autonomous(
        pod=pod, name="pod-1", issue=506, preflight_check=fake_preflight
    )
    assert preflight_calls["n"] == 3
    assert resume_calls["n"] == 1  # only fired once the preflight cleared


# ---------------------------------------------------------------------------
# 6. --wait-for-capacity interactive opt-in on resume (#530)
# ---------------------------------------------------------------------------
#
# What this section pins
# ----------------------
# Before #530 (2026-06-09) the resume-side retry-wait existed ONLY in
# autonomous mode; a cap-refused INTERACTIVE resume got the SystemExit and
# the orchestrator had to hand-roll a shell loop
# (``for i in $(seq 1 45); do resume; sleep 480; done``) while a sibling
# pod wound down. The fix: ``pod.py resume --wait-for-capacity`` opts an
# interactive run into the SAME unbounded retry-with-backoff the autonomous
# path uses (``force_wait=True`` on the helper). No flag → byte-identical
# refusal (pinned by test_resume_interactive_fails_loud_on_insufficient_
# balance above). SUPPLY_CONSTRAINT still fails loud — waiting cannot help
# when the pod's pinned host is out of GPUs.


def test_resume_parser_accepts_wait_for_capacity_flag():
    """``pod_lifecycle.py resume`` parses ``--wait-for-capacity`` (default
    OFF). ``pod.py resume`` forwards argv verbatim, so the parser is the
    single wiring point for the flag."""
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pod_lifecycle._parser_resume(sub)

    args = parser.parse_args(["resume", "--issue", "530", "--wait-for-capacity"])
    assert args.wait_for_capacity is True
    args_default = parser.parse_args(["resume", "--issue", "530"])
    assert args_default.wait_for_capacity is False


def test_resume_force_wait_retries_interactive_on_insufficient_balance(monkeypatch):
    """``force_wait=True`` (the --wait-for-capacity opt-in) retries the
    INSUFFICIENT_BALANCE refusal with backoff EVEN WITHOUT
    EPM_AUTONOMOUS_SESSION — the #530 fix. Two refusals, then success."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    pod = _make_ephemeral_pod()

    calls = {"n": 0}

    def fake_resume_pod(pod_id, gpu_count):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RunPodInsufficientBalanceError("INSUFFICIENT_BALANCE: spending limit")
        return _make_pod_info()

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    pod_lifecycle._resume_with_balance_wait_if_autonomous(
        pod=pod, name="pod-1", issue=530, force_wait=True
    )
    assert calls["n"] == 3  # retried twice, succeeded on attempt 3


def test_resume_force_wait_preflight_retries_interactive(monkeypatch):
    """``force_wait=True`` also routes the LOCAL pre-flight cap guard through
    the retry loop in interactive mode: the guard's
    RunPodInsufficientBalanceError is retried until headroom frees, then
    ``resume_pod`` fires exactly once. Mirrors the autonomous test above."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    pod = _make_ephemeral_pod()

    resume_calls = {"n": 0}

    def fake_resume_pod(pod_id, gpu_count):
        resume_calls["n"] += 1
        return _make_pod_info()

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    preflight_calls = {"n": 0}

    def fake_preflight():
        preflight_calls["n"] += 1
        if preflight_calls["n"] < 3:
            raise RunPodInsufficientBalanceError("local pre-flight over cap")

    pod_lifecycle._resume_with_balance_wait_if_autonomous(
        pod=pod, name="pod-1", issue=530, preflight_check=fake_preflight, force_wait=True
    )
    assert preflight_calls["n"] == 3
    assert resume_calls["n"] == 1


def test_resume_force_wait_supply_constraint_still_fails_loud(monkeypatch):
    """``force_wait=True`` must NOT widen the wait to SUPPLY_CONSTRAINT —
    resume never relocates the pod, so waiting cannot help when its pinned
    host is out of GPUs. Fails loud exactly like the no-flag path."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    pod = _make_ephemeral_pod()

    def fake_resume_pod(pod_id, gpu_count):
        raise RunPodError("podResume returned null for p1")  # supply-constraint marker

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._resume_with_balance_wait_if_autonomous(
            pod=pod, name="pod-1", issue=530, force_wait=True
        )
    assert "supply constraint" in str(exc.value).lower() or "no free gpus" in str(exc.value).lower()


def test_resume_no_force_wait_default_still_fails_loud(monkeypatch):
    """``force_wait`` defaults to False: an interactive caller that does NOT
    pass the flag keeps the pre-#530 fail-loud SystemExit contract, and the
    actionable message now ALSO advertises the --wait-for-capacity escape
    hatch (discoverability was the root cause of the hand-rolled loop)."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)
    pod = _make_ephemeral_pod()

    def fake_resume_pod(pod_id, gpu_count):
        raise RunPodInsufficientBalanceError("INSUFFICIENT_BALANCE: spending limit")

    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)

    with pytest.raises(SystemExit) as exc:
        pod_lifecycle._resume_with_balance_wait_if_autonomous(pod=pod, name="pod-1", issue=530)
    assert "--wait-for-capacity" in str(exc.value)


def test_cmd_resume_flag_routes_guard_through_wait_loop(monkeypatch):
    """End-to-end routing: ``cmd_resume`` with ``wait_for_capacity=True`` in
    an INTERACTIVE session calls the cap guard with
    ``transient_on_exceed=True`` (wait-loop preflight), NOT the one-shot
    SystemExit pre-call, and completes the resume. Pins the branch condition
    ``args.wait_for_capacity or _autonomous_session()`` where the #530 fix
    lives."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)

    import argparse

    from pod_lifecycle import EphemeralMetadata, EphemeralPod

    meta = EphemeralMetadata(name="pod-530", pod_id="p530", issue=530)
    stopped_info = PodInfo(
        pod_id="p530",
        name="pod-530",
        desired_status="EXITED",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=None,
        ssh_port=None,
        created_at="2026-06-09T00:00:00Z",
    )
    pod = EphemeralPod(metadata=meta, info=stopped_info)
    ready_info = PodInfo(
        pod_id="p530",
        name="pod-530",
        desired_status="RUNNING",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host="1.2.3.4",
        ssh_port=22,
        created_at="2026-06-09T00:00:00Z",
    )

    guard_kwargs: list[dict] = []

    def fake_guard(**kwargs):
        guard_kwargs.append(kwargs)

    resume_calls = {"n": 0}

    def fake_resume_pod(pod_id, gpu_count):
        resume_calls["n"] += 1
        return ready_info

    monkeypatch.setattr(pod_lifecycle, "_load_state", lambda: {"pod-530": pod})
    monkeypatch.setattr(pod_lifecycle, "_assert_under_account_hourly_cap", fake_guard)
    monkeypatch.setattr(pod_lifecycle, "resume_pod", fake_resume_pod)
    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", lambda pod_id, timeout: ready_info)
    monkeypatch.setattr(pod_lifecycle, "_read_metadata_file", lambda: {"pod-530": meta})
    monkeypatch.setattr(pod_lifecycle, "_write_metadata_file", lambda metadata: None)
    monkeypatch.setattr(pod_lifecycle, "_upsert_pods_conf", lambda p: None)
    monkeypatch.setattr(pod_lifecycle, "_restore_uv_on_pod", lambda host, port: None)

    args = argparse.Namespace(issue=530, dry_run=False, wait_for_capacity=True)
    pod_lifecycle.cmd_resume(args)

    assert resume_calls["n"] == 1
    # The guard ran exactly once, AS the wait-loop preflight
    # (transient_on_exceed=True) — not as the one-shot SystemExit pre-call.
    assert len(guard_kwargs) == 1
    assert guard_kwargs[0]["transient_on_exceed"] is True
    assert guard_kwargs[0]["verb"] == "resume"


def test_cmd_resume_no_flag_interactive_keeps_one_shot_guard(monkeypatch):
    """Without the flag (and without EPM_AUTONOMOUS_SESSION), ``cmd_resume``
    keeps the one-shot guard contract: the cap guard is called WITHOUT
    ``transient_on_exceed`` (SystemExit-on-exceed default). Pins that adding
    the flag did not flip the interactive default."""
    monkeypatch.delenv("EPM_AUTONOMOUS_SESSION", raising=False)

    import argparse

    from pod_lifecycle import EphemeralMetadata, EphemeralPod

    meta = EphemeralMetadata(name="pod-530", pod_id="p530", issue=530)
    stopped_info = PodInfo(
        pod_id="p530",
        name="pod-530",
        desired_status="EXITED",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host=None,
        ssh_port=None,
        created_at="2026-06-09T00:00:00Z",
    )
    pod = EphemeralPod(metadata=meta, info=stopped_info)
    ready_info = PodInfo(
        pod_id="p530",
        name="pod-530",
        desired_status="RUNNING",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host="1.2.3.4",
        ssh_port=22,
        created_at="2026-06-09T00:00:00Z",
    )

    guard_kwargs: list[dict] = []

    def fake_guard(**kwargs):
        guard_kwargs.append(kwargs)

    monkeypatch.setattr(pod_lifecycle, "_load_state", lambda: {"pod-530": pod})
    monkeypatch.setattr(pod_lifecycle, "_assert_under_account_hourly_cap", fake_guard)
    monkeypatch.setattr(pod_lifecycle, "resume_pod", lambda pod_id, gpu_count: ready_info)
    monkeypatch.setattr(pod_lifecycle, "wait_for_ssh", lambda pod_id, timeout: ready_info)
    monkeypatch.setattr(pod_lifecycle, "_read_metadata_file", lambda: {"pod-530": meta})
    monkeypatch.setattr(pod_lifecycle, "_write_metadata_file", lambda metadata: None)
    monkeypatch.setattr(pod_lifecycle, "_upsert_pods_conf", lambda p: None)
    monkeypatch.setattr(pod_lifecycle, "_restore_uv_on_pod", lambda host, port: None)

    args = argparse.Namespace(issue=530, dry_run=False, wait_for_capacity=False)
    pod_lifecycle.cmd_resume(args)

    assert len(guard_kwargs) == 1
    # One-shot pre-call: transient_on_exceed not passed (defaults False).
    assert "transient_on_exceed" not in guard_kwargs[0]
