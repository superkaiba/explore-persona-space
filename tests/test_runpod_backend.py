"""#1698 — RunPod launch + teardown workflow-fix tests.

Six behaviors, all covered here so any regression on the #1689 launch-path
failure family surfaces at Step 9c without a live pod:

* ``BOOTSTRAP_BRANCH`` is plumbed into the ``pod_lifecycle.py provision``
  subprocess env whenever ``spec.extra["repo_branch"]`` names a non-empty
  branch — the drop point that landed #1689 R8/R9 pods onto ``main`` twice.
* The env is OMITTED when ``repo_branch`` is empty (so ``bootstrap_pod.sh:52``
  keeps its documented default) and equally when the branch is explicitly
  ``main`` (defense in depth — the ``:-main`` default preserves that value
  regardless, so pinning + unset must be byte-equivalent).
* The post-bootstrap branch assertion FAIL-LOUDS when the pod's on-disk
  HEAD branch does not match the requested ``repo_branch`` — via
  ``pytest.raises(RunPodProvisionBranchMismatchError)`` so a silently-passed
  code path fails the test.
* The branch assertion is SKIPPED when the request is empty / ``main`` (a
  legitimate default-``main`` launch is unaffected).
* ``RunPodBackend.teardown`` treats a
  ``PodLifecycleProcessError`` whose stderr tail contains "Nothing to
  terminate" (the ``pod_lifecycle.py:2911`` signature) as idempotent
  success — the #1689 finalize failure that cost two hand ``mv``s of the
  handle sidecar.
* Every OTHER ``PodLifecycleProcessError`` propagates from teardown
  (auth failure, RunPod API 5xx, pod-exists-but-terminate-refused, and
  the #1485 ``keep-running``-tag refusal), so the fail-loud contract for
  real terminate failures stays unchanged.

All CPU, mocked subprocess + SSH — no live pod.
"""

from __future__ import annotations

import pytest

from explore_persona_space.backends import runpod as RP
from explore_persona_space.backends.base import RunSpec


@pytest.fixture(autouse=True)
def _no_live_pods_ephemeral(monkeypatch):
    """#2038: pin launch()'s live pods_ephemeral.json pod-id read to None.

    Keeps these launch tests hermetic (the live sidecar is shared-VM mutable
    fleet state); the real read body is covered in
    ``tests/test_issue2038_fallback_teardown.py``.
    """
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: None)


# ---------------------------------------------------------------------------
# Item 1(a) — BOOTSTRAP_BRANCH plumbing at the provision subprocess env
# ---------------------------------------------------------------------------


class _RelayEnvCapture:
    """Fake ``_run_pod_lifecycle_relay``: records the ``env`` kwarg every call."""

    def __init__(self) -> None:
        self.env_history: list[dict[str, str] | None] = []

    def __call__(self, cmd, *, env=None, relay=None):
        # Copy the env dict so a later mutation on the caller's copy cannot
        # rewrite what we recorded (defensive; today's caller passes a fresh
        # dict, but a future refactor might reuse os.environ verbatim).
        self.env_history.append(dict(env) if env is not None else None)
        return None


def _spec(*, extra: dict | None = None, workload_cmd: str = "") -> RunSpec:
    """Minimal RunSpec — a Hydra-args-only launch has empty workload_cmd, so
    ``execute_workload`` is inert (no SSH), which lets these tests target the
    provision subprocess env in isolation. The ``branch assertion`` tests
    below then wire the SSH probe explicitly."""
    return RunSpec(
        issue=1698,
        intent="lora-7b",
        backend="runpod",
        workload_cmd=workload_cmd,
        hydra_args=("seed=1",) if not workload_cmd else (),
        extra=extra or {},
    )


def test_launch_plumbs_repo_branch_env_to_provision_subprocess(monkeypatch):
    """When ``spec.extra["repo_branch"]`` names a specific branch,
    ``RunPodBackend.launch`` plumbs it into the provision subprocess env as
    ``BOOTSTRAP_BRANCH`` so ``bootstrap_pod.sh:52`` picks it up instead of
    defaulting to ``main`` (#1698 Item 1(a); #1689 R8/R9 drop point)."""
    relay = _RelayEnvCapture()
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", relay)
    # No-op the post-bootstrap branch assertion for this test — Item 1(a) is
    # about the env plumbing at the provision subprocess, not the assertion
    # (which has its own tests below).
    monkeypatch.setattr(RP, "_assert_pod_on_branch", lambda pod_name, expected_branch: None)

    RP.RunPodBackend().launch(_spec(extra={"repo_branch": "issue-1689"}))

    assert len(relay.env_history) == 1, relay.env_history
    env = relay.env_history[0]
    assert env is not None, "env kwarg must be threaded to _run_pod_lifecycle_relay"
    assert env["BOOTSTRAP_BRANCH"] == "issue-1689", env.get("BOOTSTRAP_BRANCH")


@pytest.mark.parametrize("repo_branch", ["", None, "main"])
def test_launch_omits_bootstrap_branch_env_when_repo_branch_absent_or_main(
    monkeypatch, repo_branch
):
    """The plumbing binds ONLY when a specific branch was requested: empty /
    absent / explicit ``main`` all leave ``BOOTSTRAP_BRANCH`` unset (so
    ``bootstrap_pod.sh:52``'s ``:-main`` default applies uniformly) OR pin it
    equal to the requested value — either shape is byte-equivalent on the
    provision, and the ``main``-explicit case is included so the concern-6
    edge case can never regress to a silent surprise."""
    relay = _RelayEnvCapture()
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", relay)
    monkeypatch.setattr(RP, "_assert_pod_on_branch", lambda pod_name, expected_branch: None)

    extra: dict[str, object] = {} if repo_branch is None else {"repo_branch": repo_branch}
    RP.RunPodBackend().launch(_spec(extra=extra))

    assert len(relay.env_history) == 1, relay.env_history
    env = relay.env_history[0]
    assert env is not None
    # Either BOOTSTRAP_BRANCH is absent (empty / None), OR it equals the
    # requested value (explicit "main"). Both are safe.
    if repo_branch == "main":
        assert env.get("BOOTSTRAP_BRANCH") in (None, "main"), env.get("BOOTSTRAP_BRANCH")
    else:
        assert "BOOTSTRAP_BRANCH" not in env, env


# ---------------------------------------------------------------------------
# Item 1(b) — post-bootstrap branch assertion (fail-loud on mismatch;
# skipped on default-main / empty)
# ---------------------------------------------------------------------------


def test_launch_asserts_pod_on_branch_after_bootstrap(monkeypatch):
    """When a specific non-``main`` branch is requested, ``RunPodBackend.launch``
    calls ``_assert_pod_on_branch`` AFTER ``_run_pod_lifecycle_relay`` returns.
    A stubbed assertion that raises confirms the call site is wired: the
    post-bootstrap fail-loud path fires on mismatch (#1698 Item 1(b))."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)

    def _boom(pod_name, expected_branch):
        raise RP.RunPodProvisionBranchMismatchError(
            f"pod {pod_name!r} bootstrapped onto branch 'main', expected "
            f"{expected_branch!r} — the --repo-branch plumbing dropped the value"
        )

    monkeypatch.setattr(RP, "_assert_pod_on_branch", _boom)

    with pytest.raises(RP.RunPodProvisionBranchMismatchError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"repo_branch": "issue-1689"}))
    msg = str(ei.value)
    assert "issue-1689" in msg
    assert "main" in msg


@pytest.mark.parametrize("repo_branch", ["", None, "main"])
def test_launch_skips_branch_assertion_when_repo_branch_is_main_or_empty(monkeypatch, repo_branch):
    """A launch that legitimately wants ``main`` (or that leaves the branch
    unset) MUST NOT invoke the branch assertion — so a default-``main``
    launch is untouched by the #1698 defenses (#1698 Item 1(b))."""
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)

    calls: list[tuple[str, str]] = []

    def _record(pod_name, expected_branch):
        calls.append((pod_name, expected_branch))

    monkeypatch.setattr(RP, "_assert_pod_on_branch", _record)

    extra: dict[str, object] = {} if repo_branch is None else {"repo_branch": repo_branch}
    RP.RunPodBackend().launch(_spec(extra=extra))

    assert calls == [], calls


def test_assert_pod_on_branch_matches_expected_returns_cleanly(monkeypatch):
    """``_assert_pod_on_branch`` returns cleanly when the on-pod
    ``git rev-parse --abbrev-ref HEAD`` matches the requested branch —
    exercises the real helper body (not a stubbed call site)."""
    monkeypatch.setattr(RP, "_resolve_pod_endpoint", lambda name: ("1.2.3.4", 22222))

    def _fake_ssh(host, port, command, *, timeout, context):
        assert host == "1.2.3.4" and port == 22222
        assert "git rev-parse --abbrev-ref HEAD" in command
        return "issue-1689\n"

    monkeypatch.setattr(RP, "_ssh_pod_run", _fake_ssh)

    # Returns None on a match; no exception.
    result = RP._assert_pod_on_branch(pod_name="pod-1698", expected_branch="issue-1689")
    assert result is None


def test_assert_pod_on_branch_raises_on_mismatch(monkeypatch):
    """``_assert_pod_on_branch`` raises ``RunPodProvisionBranchMismatchError``
    when the on-pod HEAD reports a different branch than requested (#1698
    Item 1(b) — the fail-loud body path)."""
    monkeypatch.setattr(RP, "_resolve_pod_endpoint", lambda name: ("1.2.3.4", 22222))
    monkeypatch.setattr(RP, "_ssh_pod_run", lambda *a, **k: "main\n")

    with pytest.raises(RP.RunPodProvisionBranchMismatchError) as ei:
        RP._assert_pod_on_branch(pod_name="pod-1698", expected_branch="issue-1689")
    msg = str(ei.value)
    assert "pod-1698" in msg
    assert "'main'" in msg
    assert "'issue-1689'" in msg


# ---------------------------------------------------------------------------
# Item 2 — idempotent teardown for an already-terminated pod
# ---------------------------------------------------------------------------


def _make_handle(issue: int = 1698):
    """Build a minimal RunHandle sufficient to reach RunPodBackend.teardown's
    parse-issue-from-pod-name + subprocess-relay call site. Field set
    matches the RunHandle dataclass in ``backends/base.py``: backend,
    cluster, job_id, pod_name, scratch_dir, log_path, extra.
    """
    from explore_persona_space.backends.base import RunHandle

    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name=f"pod-{issue}",
        scratch_dir="/workspace",
        log_path=f"/workspace/logs/issue-{issue}.log",
        extra={"issue": issue},
    )


def test_teardown_idempotent_on_already_gone_pod(monkeypatch):
    """When ``pod_lifecycle.py terminate`` exits non-zero with the exact
    "Nothing to terminate" stderr signature (from
    ``pod_lifecycle._terminate_clear_stale_sidecar`` at
    ``scripts/pod_lifecycle.py:2911``), ``RunPodBackend.teardown`` treats it
    as idempotent success and returns cleanly — the #1689 finalize failure
    that cost two hand ``mv``s of the handle sidecar (#1698 Item 2)."""

    def _relay(cmd, *, env=None, relay=None):
        # pod_lifecycle._terminate_clear_stale_sidecar prints:
        #   "No live pod found for issue 1698 (and no local record). Nothing to terminate."
        # to stderr and raises SystemExit(...), which yields rc=1 in the
        # child subprocess. The relay wraps it as PodLifecycleProcessError.
        raise RP.PodLifecycleProcessError(
            1,
            cmd,
            output=None,
            stderr=(
                "No live pod found for issue 1698 (and no local record). Nothing to terminate.\n"
            ),
        )

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay)

    # No exception — teardown returns None on idempotent success.
    result = RP.RunPodBackend().teardown(_make_handle())
    assert result is None


def test_teardown_propagates_other_pod_lifecycle_errors(monkeypatch):
    """Every ``PodLifecycleProcessError`` whose stderr does NOT contain the
    "Nothing to terminate" signature MUST propagate from teardown — auth
    error, RunPod API 5xx, pod-exists-but-terminate-refused, and the #1485
    ``keep-running``-tag refusal all stay fail-loud. Assert via
    ``pytest.raises`` so a silently-swallowed error would fail the test
    (#1698 Item 2 concern #5)."""

    def _relay(cmd, *, env=None, relay=None):
        # A DIFFERENT error — a hypothetical RunPod API 5xx surfacing pod_lifecycle-side.
        raise RP.PodLifecycleProcessError(
            1,
            cmd,
            output=None,
            stderr="RunPod API error: 500 Internal Server Error while terminating pod.\n",
        )

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay)

    with pytest.raises(RP.PodLifecycleProcessError) as ei:
        RP.RunPodBackend().teardown(_make_handle())
    assert ei.value.returncode == 1
    assert "500 Internal Server Error" in (ei.value.stderr or "")


def test_teardown_case_insensitive_match_on_nothing_to_terminate(monkeypatch):
    """The idempotent-success match is case-INsensitive on the substring
    "nothing to terminate" — a future ``pod_lifecycle.py`` wording tweak
    that preserves the fragment (any capitalization) MUST NOT re-open the
    #1689 finalize failure (#1698 Item 2 defense in depth)."""

    def _relay(cmd, *, env=None, relay=None):
        # Capitalization variant of the exact pod_lifecycle.py:2911 phrase.
        raise RP.PodLifecycleProcessError(
            1, cmd, output=None, stderr="NOTHING TO TERMINATE (case variant).\n"
        )

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay)
    # No exception — case-insensitive match still hits.
    RP.RunPodBackend().teardown(_make_handle())
