"""#909 — RunPod ``--workload-cmd`` execution leg.

``RunPodBackend.launch`` gains an execution leg that, when the caller opts
in via ``spec.extra["execute_workload"]``, SSHes the fresh pod, syncs the
clone to ``spec.extra["repo_branch"]``, starts ``spec.workload_cmd``
detached (setsid + nohup + pidfile + log), and verifies liveness before
returning. These tests pin:

* the opt-in gate (execute iff workload_cmd set AND flag truthy; zero SSH
  calls without the flag);
* the 3-call SSH sequence + the load-bearing detach tokens;
* every start-failure path raising :class:`RunPodWorkloadStartError`
  (branch-sync mismatch, missing pods.conf row, dead PID, double-fire
  guard, programmatic flag+empty-cmd);
* the GCP-parity fresh-pidfile acceptance for self-daemonizing drivers;
* the remote scripts never shelling ``task.py`` + being valid bash;
* the completion-sentinel chain (#909 r2,
  ``runpod-execute-missing-completion-sentinel``): the rendered launcher
  chains a success-gated sentinel write after the verbatim workload_cmd,
  the outer script clears stale sentinels before detach (widened by #976
  to the declared path + flat legacy + attempt-sibling wildcard — the
  experimenter step-11.3 breadth), and the
  launcher-threaded path is the SAME attempt-namespaced path the handle's
  expected-artifacts declaration names (one mint, one path — so
  ``_check_sentinel`` / ``_cmd_finalize`` pass on a successful
  backend-executed run); and
* the END-TO-END seam: the REAL
  ``router.failover_to_runpod_after_async_workload_crash`` drives the REAL
  ``RunPodBackend.launch`` into the execution leg with the reconstructed
  spec (the #763 failure shape closed at the seam, not just at the unit).

All CPU, mocked SSH — no live pod.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from explore_persona_space.backends import runpod as RP
from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
from explore_persona_space.backends.base import RunSpec

WORKLOAD = "bash scripts/issue909_dispatch.sh --arm a"
ATTEMPT = "rp-20260703T000000Z-ab12"


@pytest.fixture(autouse=True)
def _no_live_pods_ephemeral(monkeypatch):
    """#2038: ``launch()`` reads the LIVE pods_ephemeral.json for the pod id.

    Pin the read to ``None`` so every launch test here stays hermetic
    (fleet-state-independent — the live sidecar is a shared-VM mutable file).
    The real read body is covered in
    ``tests/test_issue2038_fallback_teardown.py`` via the documented
    ``pod_config.PODS_EPHEMERAL_JSON`` tmp seam; ``None`` keeps the legacy
    id-less ``extra`` shape the ``_PRE_954_SUCCESS_EXTRA_KEYS`` exact-set
    pins below assert.
    """
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: None)


def _noop_provision(monkeypatch) -> None:
    """No-op the ``pod_lifecycle.py provision`` subprocess call.

    Since #1465 the provision leg routes through the pod_lifecycle-only
    helper ``RP._run_pod_lifecycle_relay`` (never bare ``subprocess.run``),
    so patching the helper is selective BY CONSTRUCTION — env.py's git probe
    and the artifact-declaration helpers keep the real ``subprocess.run``.
    """
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", lambda cmd, **k: None)


class _RecordingSsh:
    """Fake ``_ssh_pod_run``: records each remote command, returns scripted
    stdout in order (or raises a scripted exception instance)."""

    def __init__(self, outputs: list) -> None:
        self.calls: list[str] = []
        self._outputs = list(outputs)

    def __call__(self, host, port, command, **kwargs):
        assert host == "1.2.3.4" and port == 22222, (host, port)
        self.calls.append(command)
        out = self._outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _wire_exec_leg(monkeypatch, ssh_outputs: list) -> _RecordingSsh:
    """Wire the execution-leg seams: endpoint resolution, SSH fake, no sleep."""
    _noop_provision(monkeypatch)
    monkeypatch.setattr(RP, "_resolve_pod_endpoint", lambda name: ("1.2.3.4", 22222))
    monkeypatch.setattr(RP, "WORKLOAD_VERIFY_DELAY_SECONDS", 0.0)
    ssh = _RecordingSsh(ssh_outputs)
    monkeypatch.setattr(RP, "_ssh_pod_run", ssh)
    return ssh


def _spec(*, extra: dict | None = None, workload_cmd: str = WORKLOAD, **overrides) -> RunSpec:
    return RunSpec(
        issue=909,
        intent="lora-7b",
        backend="runpod",
        workload_cmd=workload_cmd,
        extra=extra or {},
        **overrides,
    )


# ---------------------------------------------------------------------------
# Happy path — opt-in executes; the 3-call sequence + detach tokens
# ---------------------------------------------------------------------------


def test_launch_executes_workload_when_opted_in(monkeypatch):
    # #1698 Item 1(b): a non-main repo_branch fires the post-bootstrap
    # branch-assert SSH probe BEFORE _execute_workload_on_pod runs its
    # own 3-call sequence. Prepend the branch-verify output so the
    # scripted fake covers all 4 SSH calls.
    ssh = _wire_exec_leg(
        monkeypatch,
        [
            "issue-909\n",  # #1698 branch assertion: `git rev-parse --abbrev-ref HEAD`
            "SYNC-OK abc123\n",
            "WRAPPER-STARTED 4242\n",
            "LAUNCH-OK pid=777\n",
        ],
    )
    handle = RP.RunPodBackend().launch(
        _spec(extra={"execute_workload": True, "repo_branch": "issue-909"})
    )
    # 4 SSH calls: the #1698 branch-assert probe then the 3-call
    # execution-leg sequence (sync -> detach -> verify). The branch-assert
    # probe is the FIRST call.
    assert len(ssh.calls) == 4
    branch_probe_cmd, sync_cmd, launch_cmd, verify_cmd = ssh.calls
    assert "git rev-parse --abbrev-ref HEAD" in branch_probe_cmd, branch_probe_cmd
    assert "refs/heads/issue-909" in sync_cmd
    assert "git reset --hard" in sync_cmd
    assert "SYNC-MISMATCH" in sync_cmd  # HEAD == FETCH_HEAD verification present
    # Load-bearing detach tokens (token asserts, not one full-line literal).
    assert "setsid" in launch_cmd
    assert "nohup" in launch_cmd
    assert "bash /workspace/launch_issue_909.sh" in launch_cmd
    assert "> /workspace/logs/issue-909.log" in launch_cmd
    assert "< /dev/null" in launch_cmd
    detach_line = next(line for line in launch_cmd.splitlines() if "setsid" in line)
    assert detach_line.rstrip().endswith("&")
    # Launcher heredoc: pidfile echo + the VERBATIM workload command.
    assert "echo $$ > /workspace/logs/issue-909.pid" in launch_cmd
    assert WORKLOAD in launch_cmd
    # Double-fire guard present in the detach script.
    assert "ALREADY-RUNNING" in launch_cmd
    # Verify script probes the canonical pidfile + the fresh-pidfile fallback.
    assert "LAUNCH-OK" in verify_cmd
    assert "/workspace/logs/*.pid" in verify_cmd
    # Handle extra carries the execution outcome + repo_branch.
    assert handle.extra["workload_executed"] is True
    assert handle.extra["workload_pid"] == 777
    assert handle.extra["repo_branch"] == "issue-909"
    assert handle.extra["launcher_path"] == "/workspace/launch_issue_909.sh"
    assert handle.extra["synced_sha"] == "abc123"
    # r2 (`runpod-execute-missing-completion-sentinel`): the DECLARED
    # sentinel path is the SAME attempt-namespaced path threaded into the
    # rendered launcher (one mint, one path, both sides).
    declared = handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]["sentinel_path"]
    assert declared == RP.runpod_sentinel_path(909, handle.extra["runpod_attempt_id"])
    assert declared in launch_cmd  # threaded into script (b)
    assert f"rm -f {declared}" in launch_cmd  # stale clear before detach


def test_launch_defaults_branch_sync_to_main_without_repo_branch(monkeypatch):
    ssh = _wire_exec_leg(
        monkeypatch,
        ["SYNC-OK abc123\n", "WRAPPER-STARTED 1\n", "LAUNCH-OK pid=7\n"],
    )
    handle = RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    assert "refs/heads/main" in ssh.calls[0]
    assert handle.extra["repo_branch"] == ""  # additive key present even when unset


# ---------------------------------------------------------------------------
# No opt-in — zero SSH execution calls + the loud WARNING (AC2)
# ---------------------------------------------------------------------------


def test_launch_skips_execution_without_opt_in(monkeypatch, caplog):
    ssh = _wire_exec_leg(monkeypatch, [])
    with caplog.at_level("WARNING", logger="explore_persona_space.backends.runpod"):
        handle = RP.RunPodBackend().launch(_spec())
    assert ssh.calls == []  # zero SSH execution calls (test-pinned)
    assert handle.extra["workload_executed"] is False
    assert handle.extra["repo_branch"] == ""
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "NOT executed" in warning
    assert "EXPECTED when the" in warning and "experimenter" in warning
    # Remedies IN ORDER: experimenter-on-THIS-pod first, re-launch second.
    assert warning.index("THIS pod") < warning.index("--execute-workload")


def test_launch_mints_own_rp_attempt_id_ignoring_spec_threaded_id(monkeypatch):
    """#927: RunPod is per-launch by construction — ``launch`` mints its own
    ``rp-…`` attempt id unconditionally, so a router-threaded (or stale)
    ``spec.extra["attempt_id"]`` is inert on this lane."""
    _wire_exec_leg(monkeypatch, [])
    handle = RP.RunPodBackend().launch(_spec(extra={"attempt_id": "att-threaded-by-router"}))
    minted = handle.extra["runpod_attempt_id"]
    assert minted != "att-threaded-by-router"
    assert re.fullmatch(r"rp-\d{8}T\d{6}Z-[0-9a-f]{4}", minted), minted


def test_launch_hydra_run_without_flag_untouched(monkeypatch, caplog):
    """AC8: a hydra-args RunPod launch WITHOUT the flag no-ops the leg exactly
    as today — no SSH calls, no WARNING, workload_executed False."""
    ssh = _wire_exec_leg(monkeypatch, [])
    with caplog.at_level("WARNING", logger="explore_persona_space.backends.runpod"):
        handle = RP.RunPodBackend().launch(_spec(workload_cmd="", hydra_args=("seed=1",)))
    assert ssh.calls == []
    assert handle.extra["workload_executed"] is False
    assert not [r for r in caplog.records if "NOT executed" in r.getMessage()]


# ---------------------------------------------------------------------------
# Start-failure paths — every one raises RunPodWorkloadStartError (AC3)
# ---------------------------------------------------------------------------


def test_launch_dead_pid_raises(monkeypatch):
    _wire_exec_leg(
        monkeypatch,
        ["SYNC-OK abc\n", "WRAPPER-STARTED 1\n", "LAUNCH-DEAD pid=none\nTraceback: boom\n"],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    msg = str(ei.value)
    assert "pod-909" in msg
    assert "/workspace/logs/issue-909.log" in msg
    assert "LAUNCH-DEAD" in msg and "Traceback: boom" in msg  # carries the tail
    assert "self-daemonizes" in msg  # the check-the-pod-first note


def test_branch_sync_mismatch_raises(monkeypatch):
    # #1698 Item 1(b): the branch-assert SSH probe fires BEFORE
    # _execute_workload_on_pod's own SYNC probe when repo_branch is
    # non-`main`. Prepend the branch-verify output so the SYNC-MISMATCH
    # output reaches the second SSH call — the one _execute_workload_on_pod
    # actually issues. #1858: a first sync failure now runs the git
    # kill-and-reap + EXACTLY ONE retry, so a persistent mismatch consumes
    # sync → reap → sync before the terminal raise (which carries both
    # failure summaries + the REAP-OK evidence).
    _wire_exec_leg(
        monkeypatch,
        [
            "issue-909\n",
            "SYNC-MISMATCH head=aaa fetch=bbb\n",
            REAP_CLEAN,
            "SYNC-MISMATCH head=aaa fetch=bbb\n",
        ],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(
            _spec(extra={"execute_workload": True, "repo_branch": "issue-909"})
        )
    msg = str(ei.value)
    assert "pod-909" in msg and "issue-909" in msg
    assert "SYNC-MISMATCH" in msg
    assert "sync retry after reap failed" in msg


def test_missing_pods_conf_row_raises(monkeypatch):
    import scripts.pod_config as pod_config

    _noop_provision(monkeypatch)
    monkeypatch.setattr(pod_config, "parse_pods_conf", lambda path=None: [])
    with pytest.raises(RP.RunPodWorkloadStartError, match=r"pods\.conf"):
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))


def test_already_running_guard_raises(monkeypatch):
    """The detach script's live-PID guard (exit 5) surfaces as a typed error
    naming the live PID + BOTH executors (the double-fire diagnosis)."""
    _wire_exec_leg(
        monkeypatch,
        [
            "SYNC-OK abc\n",
            RP.RunPodWorkloadStartError(
                "workload detach on pod-909: remote command exited rc=5; "
                "stderr tail: 'ALREADY-RUNNING pid=4141'"
            ),
        ],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    msg = str(ei.value)
    assert "4141" in msg
    assert "experimenter" in msg and "--execute-workload" in msg
    assert "double-launch" in msg


def test_self_daemonizing_workload_reads_launch_ok(monkeypatch):
    """GCP-parity acceptance: a fresh pod-side pidfile ``LAUNCH-OK pid=<int>
    via=<file>`` verdict is a SUCCESS (self-daemonizing drivers must not read
    a false LAUNCH-DEAD)."""
    _wire_exec_leg(
        monkeypatch,
        [
            "SYNC-OK abc\n",
            "WRAPPER-STARTED 1\n",
            "LAUNCH-OK pid=888 via=/workspace/logs/issue-909-driver.pid\n",
        ],
    )
    handle = RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    assert handle.extra["workload_executed"] is True
    assert handle.extra["workload_pid"] == 888


def test_programmatic_flag_with_empty_workload_cmd_raises(monkeypatch):
    """The defensive in-backend guard behind the CLI parse-time one (AC3a):
    a programmatic caller cannot recreate the flag+hydra false-green cell.
    Raises BEFORE any provision subprocess (no pod paid for)."""

    def _explode(*a, **k):
        raise AssertionError("provision must NOT run on the flag+empty-cmd cell")

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _explode)
    with pytest.raises(RP.RunPodWorkloadStartError, match="empty workload_cmd"):
        RP.RunPodBackend().launch(
            _spec(workload_cmd="", hydra_args=("seed=1",), extra={"execute_workload": True})
        )


# ---------------------------------------------------------------------------
# Remote-script hygiene: no task.py shellout, no [phase= literal, valid bash
# ---------------------------------------------------------------------------


def _render_launch(workload_cmd: str = WORKLOAD) -> str:
    return RP._render_launch_script(
        issue=909,
        workload_cmd=workload_cmd,
        log_path="/workspace/logs/issue-909.log",
        pid_file="/workspace/logs/issue-909.pid",
        sentinel_path=RP.runpod_sentinel_path(909, ATTEMPT),
        attempt_id=ATTEMPT,
    )


def _rendered_scripts(workload_cmd: str = WORKLOAD) -> list[str]:
    return [
        RP._render_branch_sync_script("issue-909"),
        _render_launch(workload_cmd),
        RP._render_verify_script(
            issue=909,
            log_path="/workspace/logs/issue-909.log",
            pid_file="/workspace/logs/issue-909.pid",
        ),
        RP._render_sync_reap_script(),
    ]


def test_remote_scripts_never_shell_task_py():
    """Pods never shell scripts/task.py (CLAUDE.md hard rule; complements
    tests/test_no_pod_side_task_py_shellout.py) — and never emit a bare
    ``[phase=`` literal (the poller's reserved token, experimenter.md)."""
    for script in _rendered_scripts():
        assert "task.py" not in script
        assert "[phase=" not in script


def test_rendered_scripts_bash_n(tmp_path):
    """All four remote scripts (sync / launch / verify / #1858 reap) parse
    under ``bash -n``, including a quoting-stress workload_cmd carrying a
    single quote, ``$VAR``, and ``&&`` (the GCP
    ``test_render_startup_script_is_valid_bash`` precedent)."""
    stress = "VAR=1 bash scripts/x.sh --note 'it'\\''s fine' && echo \"$VAR done\""
    for i, script in enumerate(_rendered_scripts(workload_cmd=stress)):
        path = tmp_path / f"script_{i}.sh"
        path.write_text(script + "\n", encoding="utf-8")
        proc = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, f"script {i} failed bash -n: {proc.stderr}"


def test_branch_sync_script_rejects_suspicious_branch():
    with pytest.raises(RP.RunPodWorkloadStartError, match="suspicious branch"):
        RP._render_branch_sync_script("issue-909; rm -rf /")


# ---------------------------------------------------------------------------
# #1858 — branch-sync kill-and-reap + bounded retry (MooseFS-hung git; the
# incident-#1769-fu1 class: local ssh timeout orphaned a REMOTE git holding
# .git/index.lock, and the old conditional reap could never fire against it)
# ---------------------------------------------------------------------------

REAP_CLEAN = "REAP-OK killed=0 survivors=0 lock_removed=yes\n"


def test_branch_sync_script_per_op_remote_timeouts():
    """#1858 acceptance 1 (#1981 recalibration): the three git MUTATION ops
    self-bound with per-op remote ``timeout -k 10`` (120/90/90; worst case
    incl. the KILL grace 330 s, strictly under the local ssh bound so the
    remote bounds fire first and the hung lock-holder dies REMOTELY); the
    rev-parse verification lines stay bare (ref reads, not FUSE-heavy ops).
    The checkout/reset caps were raised 20 → 90 s in #1981 to accommodate
    healthy-but-slow MooseFS mounts (~59.5 s ``git status`` measured on
    pod-1895, 2026-08-02) — the pre-#1981 20 s caps timed out at rc=124
    on both attempts of the parent incident."""
    script = RP._render_branch_sync_script("issue-909")
    assert 'timeout -k 10 120 git fetch origin "refs/heads/issue-909"' in script
    assert 'timeout -k 10 90 git checkout -q -f -B "issue-909" FETCH_HEAD' in script
    assert "timeout -k 10 90 git reset --hard -q FETCH_HEAD" in script
    for line in script.splitlines():
        if "rev-parse" in line:
            assert "timeout" not in line, line
    # Keep the pre-existing opening conditional lock-reap line.
    assert "pgrep -x git >/dev/null 2>&1 || rm -f .git/index.lock" in script
    # Summed worst case (every TERM needing the -k 10 KILL grace) stays
    # strictly under the local ssh bound.
    assert RP.SYNC_SSH_TIMEOUT_SECONDS > (120 + 10) + (90 + 10) + (90 + 10)


def test_branch_sync_script_already_at_tip_short_circuit():
    """#1981 durability pin: after the fetch, cheap ref reads short-circuit
    the mutation paths with ``SYNC-OK`` when HEAD already equals FETCH_HEAD
    on the requested branch — the checkout + reset FUSE-heavy ops never
    execute on the common case. The short-circuit's echo line matches the
    caller's ``SYNC-OK ([0-9a-f]+)`` regex in ``_attempt_sync`` so the pod
    HEAD sha is captured correctly."""
    import re

    script = RP._render_branch_sync_script("issue-909")
    lines = script.splitlines()

    # The three ref reads that feed the short-circuit predicate (cheap;
    # never a FUSE-heavy object-store op).
    assert "HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo none)" in lines
    assert "FETCH_SHA=$(git rev-parse FETCH_HEAD)" in lines
    assert "CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)" in lines

    # The short-circuit itself: matches both the sha equality AND the
    # branch identity so a HEAD-that-happens-to-equal-FETCH_HEAD on a
    # different branch still takes the mutation path.
    short_circuit_if = 'if [ "$HEAD_SHA" = "$FETCH_SHA" ] && [ "$CUR_BRANCH" = "issue-909" ]; then'
    assert short_circuit_if in lines
    short_circuit_echo = 'echo "SYNC-OK $HEAD_SHA (already-at-tip short-circuit)"'
    assert f"  {short_circuit_echo}" in lines
    assert "  exit 0" in lines

    # The short-circuit block precedes the mutation ops — otherwise the
    # 90 s checkout would already have run before the predicate is checked.
    if_idx = lines.index(short_circuit_if)
    checkout_idx = next(i for i, line in enumerate(lines) if "git checkout -q -f -B" in line)
    reset_idx = next(i for i, line in enumerate(lines) if "git reset --hard -q FETCH_HEAD" in line)
    assert if_idx < checkout_idx < reset_idx

    # The short-circuit echo satisfies the caller's SYNC-OK regex — the
    # regex captures the leading hex sha and ignores the trailing tag.
    sync_ok_re = re.compile(r"SYNC-OK ([0-9a-f]+)")
    # Substitute a realistic sha at the sha-expansion site so the assertion
    # models what the pod would actually print.
    rendered_echo = short_circuit_echo.replace("$HEAD_SHA", "abc123def456")
    match = sync_ok_re.search(rendered_echo)
    assert match is not None
    assert match.group(1) == "abc123def456"


def test_sync_first_failure_reap_then_retry_succeeds(monkeypatch):
    """#1858 acceptance 5(i): FIRST sync failure (a local-TimeoutExpired-
    shaped RunPodWorkloadStartError) → reap (clean) → EXACTLY ONE retry →
    the launch proceeds. Recorded SSH sequence: sync, reap, sync, launch,
    verify."""
    ssh = _wire_exec_leg(
        monkeypatch,
        [
            RP.RunPodWorkloadStartError(
                "branch sync of pod-909 to 'main': ssh to 1.2.3.4:22222 failed "
                "(TimeoutExpired: Command timed out)"
            ),
            REAP_CLEAN,
            "SYNC-OK abc123\n",
            "WRAPPER-STARTED 1\n",
            "LAUNCH-OK pid=777\n",
        ],
    )
    handle = RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    assert len(ssh.calls) == 5
    sync1, reap, sync2, launch_cmd, verify_cmd = ssh.calls
    assert "refs/heads/main" in sync1
    assert "REAP-OK" in reap and "pgrep -x git" in reap
    assert sync2 == sync1  # the retry re-renders the identical sync script
    assert "launch_issue_909.sh" in launch_cmd
    assert "LAUNCH-OK" in verify_cmd
    assert handle.extra["workload_executed"] is True
    assert handle.extra["workload_pid"] == 777
    assert handle.extra["synced_sha"] == "abc123"


def test_sync_reap_survivors_raises_without_retry(monkeypatch):
    """#1858 acceptance 5(ii): ``survivors>0`` (git pids outliving SIGKILL —
    the mount-level D-state wedge) → immediate typed raise carrying the
    REAP-OK evidence, and NO second sync attempt."""
    ssh = _wire_exec_leg(
        monkeypatch,
        [
            "some sync output with no confirmation line\n",
            "REAP-OK killed=2 survivors=1 lock_removed=yes\n",
        ],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    msg = str(ei.value)
    assert "unkillable git survivors (moosefs D-state signature)" in msg
    assert "REAP-OK killed=2 survivors=1 lock_removed=yes" in msg
    assert "did not confirm SYNC-OK" in msg  # the first-failure summary rides along
    assert len(ssh.calls) == 2  # sync, reap — and nothing after


def test_sync_retry_failure_raises_with_reap_evidence(monkeypatch):
    """#1858 acceptance 5(iii): clean reap but the retried sync ALSO fails →
    raise carrying BOTH failure summaries + the REAP-OK line."""
    ssh = _wire_exec_leg(
        monkeypatch,
        [
            "some sync output with no confirmation line\n",
            REAP_CLEAN,
            RP.RunPodWorkloadStartError(
                "branch sync of pod-909 to 'main': remote command exited rc=124"
            ),
        ],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    msg = str(ei.value)
    assert "sync retry after reap failed" in msg
    assert "REAP-OK killed=0 survivors=0 lock_removed=yes" in msg
    assert "did not confirm SYNC-OK" in msg  # first failure summary
    assert "rc=124" in msg  # retry failure summary
    assert len(ssh.calls) == 3  # sync, reap, sync — never a third sync


def test_sync_reap_ssh_failure_raises_without_retry(monkeypatch):
    """The reap probe ITSELF failing (pod-level wedge) raises with both
    summaries and never retries the sync."""
    ssh = _wire_exec_leg(
        monkeypatch,
        [
            "some sync output with no confirmation line\n",
            RP.RunPodWorkloadStartError(
                "git kill-and-reap on pod-909: ssh to 1.2.3.4:22222 failed"
            ),
        ],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    msg = str(ei.value)
    assert "kill-and-reap probe ALSO failed" in msg
    assert len(ssh.calls) == 2


def test_sync_reap_missing_reap_ok_line_raises_without_retry(monkeypatch):
    """A reap that exits 0 WITHOUT the REAP-OK report line is unverified —
    raise with the reap output tail, no retry."""
    ssh = _wire_exec_leg(
        monkeypatch,
        [
            "some sync output with no confirmation line\n",
            "garbage reap output\n",
        ],
    )
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    msg = str(ei.value)
    assert "did not confirm REAP-OK" in msg
    assert "garbage reap output" in msg
    assert len(ssh.calls) == 2


def test_reap_script_real_bash_zero_git_branch(tmp_path):
    """#1858 acceptance 5(vi): REAL-BASH execution of the RENDERED reap
    script on the MODAL zero-git branch. PATH-shimmed STUB ``pgrep`` (exit
    1 = no match) + stub ``kill`` — NEVER live pgrep/kill on the shared VM
    (``kill`` is additionally a bash builtin; the zero-git branch never
    reaches it, which ``killed=0`` asserts). The renderer's ``clone_dir``
    seam points the script's cd at a tmp fake clone with a pre-created
    ``.git/index.lock``. Asserts rc=0, the exact REAP-OK line, and the
    lock removed."""
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    lock = clone / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "pgrep").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    (shim / "pgrep").chmod(0o755)
    (shim / "kill").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (shim / "kill").chmod(0o755)
    script_path = tmp_path / "reap.sh"
    script_path.write_text(
        RP._render_sync_reap_script(clone_dir=str(clone)) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PATH"] = f"{shim}:{env['PATH']}"
    proc = subprocess.run(
        ["bash", str(script_path)], capture_output=True, text=True, env=env, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "REAP-OK killed=0 survivors=0 lock_removed=yes" in proc.stdout
    assert not lock.exists()  # removed UNCONDITIONALLY


# ---------------------------------------------------------------------------
# Completion-sentinel chain (#909 r2 — the upheld
# `runpod-execute-missing-completion-sentinel` blocker; the mechanizable
# static check: the rendered launcher MUST contain the sentinel-write leg)
# ---------------------------------------------------------------------------


def test_launcher_chains_sentinel_write_after_workload_success():
    """The rendered launcher chains the completion-sentinel write AFTER the
    verbatim workload_cmd, gated on workload success (rc 0 — the GCP/SLURM
    terminal-block convention), writes the exact JSON shape
    ``artifacts._check_sentinel`` validates (phase=done + matching issue),
    and exits with the workload's own rc. The OUTER portion clears the
    stale sentinels — declared path + flat legacy + attempt-sibling
    wildcard (#976) — BEFORE the detach line (the same guard family as
    the pidfile rm) and AFTER the ALREADY-RUNNING guard completes."""
    sentinel = RP.runpod_sentinel_path(909, ATTEMPT)
    script = _render_launch()
    lines = script.splitlines()

    # Outer portion: widened stale-sentinel clear (#976) + dir pre-create
    # BEFORE detach.
    issue_dir = sentinel.rsplit("/", 2)[0]
    name = sentinel.rsplit("/", 1)[1]
    rm_idx = lines.index(f"rm -f {sentinel} {issue_dir}/{name} {issue_dir}/*/{name}")
    detach_idx = next(i for i, line in enumerate(lines) if "setsid" in line)
    heredoc_start = next(i for i, line in enumerate(lines) if "<< 'EPSEOF'" in line)
    assert rm_idx < heredoc_start < detach_idx
    assert any(line == f"mkdir -p {sentinel.rsplit('/', 1)[0]}" for line in lines[:heredoc_start])

    # Guard-completes-before-clear (#976 binding dependency, reconciler-
    # upheld): the ALREADY-RUNNING guard's `exit 5` and its closing `fi`
    # both precede the widened clear, so a live prior workload exits
    # BEFORE any sentinel it depends on is removed.
    exit5_idx = lines.index("  exit 5")
    fi_idx = lines.index("fi", exit5_idx)
    assert exit5_idx < fi_idx < rm_idx

    # Launcher (inside the heredoc): workload -> rc capture -> success-gated
    # sentinel write -> exit with the workload rc.
    workload_idx = lines.index(WORKLOAD)
    rc_idx = lines.index("WORKLOAD_RC=$?")
    gate_idx = next(i for i, line in enumerate(lines) if '"$WORKLOAD_RC" -eq 0' in line)
    write_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("printf"))
    exit_idx = lines.index('exit "$WORKLOAD_RC"')
    heredoc_end = lines.index("EPSEOF", heredoc_start + 1)
    assert heredoc_start < workload_idx < rc_idx < gate_idx < write_idx < exit_idx < heredoc_end

    # The write targets the EXACT declared path with the EXACT validated shape.
    write_line = lines[write_idx]
    assert f"> {sentinel}" in write_line
    payload_match = re.search(r"printf '%s\\n' '([^']+)'", write_line)
    assert payload_match, write_line
    payload = json.loads(payload_match.group(1))
    assert payload == {"phase": "done", "issue": 909, "attempt_id": ATTEMPT}


def test_launcher_waits_on_fresh_detached_pid_files_before_sentinel():
    """The rc==0 branch waits on fresh detached ``/workspace/logs/*.pid``
    workloads BEFORE the sentinel write (#977, the GCP #601 parity —
    ``test_render_startup_script_workload_cmd_waits_on_detached_pid_files``
    precedent). Pins: the in-launcher ``WORKLOAD_START_EPOCH`` capture sits
    AFTER the self-pidfile write and IMMEDIATELY before the workload line;
    ``WORKLOAD_RC=$?`` is ADJACENT to the workload line (an intervening
    line would corrupt ``$?``); the freshness predicate renders as ONE
    line carrying both ``stat -c %Y`` and the INCLUSIVE
    ``-ge "$WORKLOAD_START_EPOCH"`` (pins the adjacent-string
    concatenation); and the full ordering chain — pidfile-write <
    epoch-capture < workload < rc-capture < rc==0 gate < for-loop <
    freshness < cat < PID-VALUE self-exclusion < kill-0 wait < sentinel
    printf < exit — holds inside the heredoc. A misordered self-exclusion
    AFTER the kill-0 wait would deadlock the launcher on its own pid, so
    the chain includes the exclusion index, not just its presence."""
    script = _render_launch()
    lines = script.splitlines()

    heredoc_start = next(i for i, line in enumerate(lines) if "<< 'EPSEOF'" in line)
    heredoc_end = lines.index("EPSEOF", heredoc_start + 1)

    pid_idx = lines.index("echo $$ > /workspace/logs/issue-909.pid")
    epoch_idx = lines.index("WORKLOAD_START_EPOCH=$(date +%s)")
    workload_idx = lines.index(WORKLOAD)
    rc_idx = lines.index("WORKLOAD_RC=$?")
    gate_idx = next(i for i, line in enumerate(lines) if '"$WORKLOAD_RC" -eq 0' in line)
    for_idx = lines.index("  for pf in /workspace/logs/*.pid; do")
    fresh_idx = next(
        i
        for i, line in enumerate(lines)
        if "stat -c %Y" in line and '-ge "$WORKLOAD_START_EPOCH"' in line
    )
    cat_idx = lines.index('    wpid=$(cat "$pf" 2>/dev/null) || continue')
    excl_idx = lines.index('    [ "$wpid" = "$$" ] && continue')
    wait_idx = lines.index('    while kill -0 "$wpid" 2>/dev/null; do sleep 30; done')
    printf_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("printf"))
    exit_idx = lines.index('exit "$WORKLOAD_RC"')

    # Epoch capture IMMEDIATELY before the workload; rc capture ADJACENT
    # after it (any intervening line would corrupt $?).
    assert epoch_idx == workload_idx - 1
    assert rc_idx == workload_idx + 1

    # Full ordering chain, all inside the heredoc (the wait sits strictly
    # between the rc==0 gate and the sentinel write).
    assert (
        heredoc_start
        < pid_idx
        < epoch_idx
        < workload_idx
        < rc_idx
        < gate_idx
        < for_idx
        < fresh_idx
        < cat_idx
        < excl_idx
        < wait_idx
        < printf_idx
        < exit_idx
        < heredoc_end
    )


def test_launcher_wait_loop_self_exclusion_is_by_pid_value_not_path():
    """Self-exclusion in the #977 wait loop is by PID VALUE, never by
    pidfile PATH (the deliberate plan §3.2 decision): the experimenter
    ``launch_issue_<N>.sh`` convention has the detached driver OVERWRITE
    the canonical ``/workspace/logs/issue-<N>.pid`` with its OWN pid, so
    a path-based skip (``[ "$pf" = <canonical> ] && continue``) would
    skip exactly the driver that must be waited on and reintroduce the
    premature sentinel for the convention-following case — while ``$$``
    cannot be reused as long as this launcher is alive, so the pid-value
    compare is race-free at any mtime granularity. Pins the exclusion
    line's presence AND that NO wait-loop line string-compares ``$pf``
    against the canonical pidfile path, so a future "hardening" edit
    cannot silently re-add the path skip."""
    script = _render_launch()
    lines = script.splitlines()

    assert '    [ "$wpid" = "$$" ] && continue' in lines

    for_idx = lines.index("  for pf in /workspace/logs/*.pid; do")
    done_idx = lines.index("  done", for_idx)
    canonical_pid_file = "/workspace/logs/issue-909.pid"
    for line in lines[for_idx : done_idx + 1]:
        assert not ("$pf" in line and canonical_pid_file in line), (
            f"wait-loop line path-compares $pf against the canonical pidfile: {line!r}"
        )


def test_launch_script_rejects_single_quote_in_sentinel_json():
    """The single-quoted JSON embed fails LOUD on a caller bug rather than
    rendering a broken launcher."""
    with pytest.raises(RP.RunPodWorkloadStartError, match="single quote"):
        RP._render_launch_script(
            issue=909,
            workload_cmd=WORKLOAD,
            log_path="/workspace/logs/issue-909.log",
            pid_file="/workspace/logs/issue-909.pid",
            sentinel_path="/workspace/eval_results/issue_909/x/.completion-sentinel.json",
            attempt_id="rp-bad'quote",
        )


def test_stale_clear_covers_flat_legacy_and_wildcard_siblings():
    """The widened stale clear (#976) carries all three operands — the
    declared attempt path, the flat legacy path, and the attempt-sibling
    wildcard — and the wildcard operand string-equals the glob
    ``artifacts._default_glob_sentinels`` probes for the same declared
    path: clear breadth == #685 fallback probe breadth, by construction.
    ``SENTINEL_FILENAME`` is imported so a future rename of the sentinel
    filename breaks THIS test rather than silently decoupling the clear
    from the resolver's probe."""
    from pathlib import Path

    from explore_persona_space.backends.artifacts import SENTINEL_FILENAME

    sentinel = RP.runpod_sentinel_path(909, ATTEMPT)
    script = _render_launch()
    lines = script.splitlines()
    heredoc_start = next(i for i, line in enumerate(lines) if "<< 'EPSEOF'" in line)
    rm_line = next(
        line
        for line in lines[:heredoc_start]
        if line.startswith("rm -f ") and SENTINEL_FILENAME in line
    )
    operands = rm_line.split()[2:]
    issue_dir = Path(sentinel).parent.parent
    assert sentinel in operands  # exact declared attempt path (kept — pure addition)
    assert str(issue_dir / SENTINEL_FILENAME) in operands  # flat legacy path
    # The wildcard operand equals the resolver's probe shape verbatim
    # (artifacts._default_glob_sentinels: grandparent-of-declared + */<name>).
    assert str(issue_dir / f"*/{SENTINEL_FILENAME}") in operands


def test_stale_clear_rm_line_execution_defeats_single_live_sibling_fallback(tmp_path):
    """Functional proof (#976 acceptance criteria 2 + 3): executing the
    rendered rm line under the outer script's ``set -eu`` removes a stale
    flat legacy sentinel AND a stale prior-attempt sibling — leaving
    ``artifacts._default_glob_sentinels`` nothing for the #685
    single-live-sibling fallback to resolve — and exits 0 again when
    NOTHING matches (fresh pod: the unmatched-glob-under-``set -eu``
    assumption, test-backed)."""
    from explore_persona_space.backends import artifacts as ART

    issue_dir = tmp_path / "eval_results" / "issue_909"
    declared = issue_dir / "rp-new" / ".completion-sentinel.json"
    # Stale prior-attempt sibling + stale flat legacy sentinel.
    (issue_dir / "rp-old").mkdir(parents=True)
    (issue_dir / "rp-old" / ".completion-sentinel.json").write_text("{}")
    (issue_dir / ".completion-sentinel.json").write_text("{}")

    script = RP._render_launch_script(
        issue=909,
        workload_cmd=WORKLOAD,
        log_path="/workspace/logs/issue-909.log",
        pid_file="/workspace/logs/issue-909.pid",
        sentinel_path=str(declared),
        attempt_id=ATTEMPT,
    )
    rm_line = next(
        line
        for line in script.splitlines()
        if line.startswith("rm -f ") and ".completion-sentinel.json" in line
    )

    # (a) Stale files present -> removed, rc 0 under the outer script's set -eu.
    proc = subprocess.run(
        ["bash", "-c", f"set -eu\n{rm_line}"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert not (issue_dir / "rp-old" / ".completion-sentinel.json").exists()
    assert not (issue_dir / ".completion-sentinel.json").exists()
    # The #685 fallback now has nothing to resolve.
    assert ART._default_glob_sentinels(str(declared), 909) == []

    # (b) Nothing matches (fresh pod) -> unmatched glob still exits 0.
    proc2 = subprocess.run(
        ["bash", "-c", f"set -eu\n{rm_line}"], capture_output=True, text=True, check=False
    )
    assert proc2.returncode == 0, proc2.stderr


# ---------------------------------------------------------------------------
# END-TO-END seam pin (#909 plan §4 item 4 — a NEW test, not an update):
# the REAL failover seam drives the REAL RunPodBackend.launch into the
# execution leg with the reconstructed custom-workload spec.
# ---------------------------------------------------------------------------


def test_end_to_end_failover_spec_flows_through_execution_leg(monkeypatch, tmp_path):
    """Drive the REAL ``failover_to_runpod_after_async_workload_crash`` with a
    custom-workload spec into the REAL ``RunPodBackend.launch`` (provision
    no-op'd) and assert ``_execute_workload_on_pod`` is CALLED with the
    reconstructed spec — the failover-reconstructed RunSpec round-trips
    through the new leg (the #763 shape closed end to end)."""
    from explore_persona_space.backends.router import (
        LeaseStore,
        failover_to_runpod_after_async_workload_crash,
    )

    _noop_provision(monkeypatch)
    # #1698 Item 1(b): no-op the post-bootstrap branch assertion — this
    # test uses `_execute_workload_on_pod` as a stub, not the real SSH
    # fake, so the branch-assert SSH call would try to resolve
    # pod-909's endpoint from a real (empty) pods.conf and fail. The
    # branch-assertion body has its own direct tests in
    # `tests/test_runpod_backend.py`; here we exercise the failover +
    # execution-leg dispatch specifically.
    monkeypatch.setattr(RP, "_assert_pod_on_branch", lambda pod_name, expected_branch: None)
    executed: list = []

    def _fake_exec(spec, *, pod_name, log_path, pid_file, sentinel_path, attempt_id):
        executed.append((spec, pod_name, log_path, pid_file, sentinel_path, attempt_id))
        return {
            "workload_pid": 999,
            "launcher_path": "/workspace/launch_issue_909.sh",
            "synced_sha": "abc",
        }

    monkeypatch.setattr(RP, "_execute_workload_on_pod", _fake_exec)

    tick = iter(range(10_000))
    result = failover_to_runpod_after_async_workload_crash(
        spec=RunSpec(
            issue=909,
            intent="lora-7b",
            backend="gcp",
            workload_cmd=WORKLOAD,
            extra={"repo_branch": "issue-909"},
        ),
        runpod_backend=RP.RunPodBackend(),
        evidence={"source": "test_909_end_to_end"},
        marker_poster=lambda **kw: None,
        lease_store=LeaseStore(lease_dir=tmp_path / ".eps-routing"),
        now_fn=lambda: float(next(tick)),
    )
    # The execution leg FIRED, with the failover-opted-in spec.
    assert len(executed) == 1
    exec_spec, pod_name, log_path, pid_file, sentinel_path, attempt_id = executed[0]
    assert exec_spec.workload_cmd == WORKLOAD
    assert exec_spec.extra.get("execute_workload") is True
    assert exec_spec.extra.get("repo_branch") == "issue-909"
    assert pod_name == "pod-909"
    assert log_path == "/workspace/logs/issue-909.log"
    assert pid_file == "/workspace/logs/issue-909.pid"
    # The launched handle carries the execution outcome.
    assert result.chosen_kind == "runpod"
    assert result.handle.extra["workload_executed"] is True
    assert result.handle.extra["workload_pid"] == 999
    assert result.handle.extra["repo_branch"] == "issue-909"
    # r2: the sentinel path threaded into the execution leg is the SAME
    # attempt-namespaced path the handle DECLARES (one mint, one path) —
    # so finalize's `_check_sentinel` reads the path the launcher writes.
    assert sentinel_path == RP.runpod_sentinel_path(909, attempt_id)
    assert result.handle.extra["runpod_attempt_id"] == attempt_id
    assert result.handle.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]["sentinel_path"] == sentinel_path


def test_launch_ok_regex_shapes():
    """The VM-side verify parser accepts both LAUNCH-OK shapes (canonical
    pidfile + fresh-pidfile ``via=``) and rejects LAUNCH-DEAD."""
    assert RP._LAUNCH_OK_RE.search("LAUNCH-OK pid=42").group(1) == "42"
    assert RP._LAUNCH_OK_RE.search("LAUNCH-OK pid=7 via=/workspace/logs/x.pid").group(1) == "7"
    assert RP._LAUNCH_OK_RE.search("LAUNCH-DEAD pid=none") is None
    assert re.search(r"SYNC-OK ([0-9a-f]+)", "SYNC-OK deadbeef").group(1) == "deadbeef"


# ---------------------------------------------------------------------------
# #954 — the PARTIAL handle on RunPodWorkloadStartError
# ---------------------------------------------------------------------------

#: The EXACT pre-#954 success-path ``extra`` key set for a NON-exec launch
#: (``workload_info == {}``) — pinned EXPLICITLY (never a circular post-change
#: fixture): the #954 refactor must add NO new keys on the success path.
#: #1118 adds two CONDITIONAL keys (``boot_disk_gb`` / ``min_ram_gb``),
#: OMITTED when the spec states no footprint — the specs below state none,
#: so this exact set still holds (the omit-when-absent contract is what the
#: exact-set assertions pin; the conditional keys are covered by
#: ``test_launch_handle_extra_carries_boot_disk_gb``).
#: #2038 adds a third CONDITIONAL key (``pod_id``, round-tripped from
#: pods_ephemeral.json), OMITTED when the read yields nothing — the autouse
#: ``_no_live_pods_ephemeral`` fixture pins the read to ``None`` here, so
#: the exact set still holds; the present-key shape is covered by
#: ``tests/test_issue2038_fallback_teardown.py``.
_PRE_954_SUCCESS_EXTRA_KEYS = frozenset(
    {
        "intent",
        "issue",
        "pid_file",
        "runpod_attempt_id",
        "workload_cmd",
        "hydra_args",
        "gpus",
        "time_budget_hours",
        "repo_branch",
        "workload_executed",
        EXPECTED_ARTIFACTS_HANDLE_KEY,
    }
)


def test_launch_attaches_partial_handle_on_workload_start_error(monkeypatch):
    """#954: with provision mocked OK and the execution leg raising, the typed
    error carries a PARTIAL handle matching the success-path handle shape
    except ``workload_executed is False`` + ``workload_start_error`` — and the
    SUCCESS-path handle ``extra`` stays byte-identical (no new keys)."""
    # #1698 Item 1(b): supply the branch-assert SSH output; the SSH fake
    # is otherwise scripted with an empty output list so the execution
    # leg's own SSH calls (never reached because _execute_workload_on_pod
    # is stubbed to raise) do not draw from it.
    _wire_exec_leg(monkeypatch, ["issue-909\n"])

    def _fake_exec(spec, **kwargs):
        raise RP.RunPodWorkloadStartError("branch sync of pod-909 timed out (ssh TimeoutExpired)")

    monkeypatch.setattr(RP, "_execute_workload_on_pod", _fake_exec)
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(
            _spec(extra={"execute_workload": True, "repo_branch": "issue-909"})
        )
    partial = ei.value.handle
    assert partial is not None
    assert partial.backend == "runpod"
    assert partial.pod_name == "pod-909"
    assert partial.log_path == "/workspace/logs/issue-909.log"
    # Truthful execution outcome + the truncated start-leg error.
    assert partial.extra["workload_executed"] is False
    assert "ssh TimeoutExpired" in partial.extra["workload_start_error"]
    # The partial extra == the success shape PLUS ONLY the error key.
    assert set(partial.extra.keys()) == _PRE_954_SUCCESS_EXTRA_KEYS | {"workload_start_error"}
    # The declaration is fully built (attempt-namespaced sentinel path present),
    # so poll/finalize/re-drive stay chained on the partial handle.
    declared = partial.extra[EXPECTED_ARTIFACTS_HANDLE_KEY]["sentinel_path"]
    assert declared == RP.runpod_sentinel_path(909, partial.extra["runpod_attempt_id"])


def test_launch_success_extra_keys_byte_identical_no_new_keys(monkeypatch):
    """#954 regression guard (test 10 second half): the SUCCESS-path handle
    ``extra`` key set is byte-identical to pre-#954 for BOTH the non-exec and
    the exec-success shapes — ``workload_start_error`` appears ONLY on the
    failure path."""
    # Non-exec success: the exact pre-change key set, nothing added.
    _wire_exec_leg(monkeypatch, [])
    handle = RP.RunPodBackend().launch(_spec())
    assert set(handle.extra.keys()) == set(_PRE_954_SUCCESS_EXTRA_KEYS)

    # Exec success: the pre-change keys + the execution-leg workload_info keys;
    # the failure-path-only key NEVER appears on success.
    _wire_exec_leg(
        monkeypatch,
        ["SYNC-OK abc123\n", "WRAPPER-STARTED 4242\n", "LAUNCH-OK pid=777\n"],
    )
    handle2 = RP.RunPodBackend().launch(_spec(extra={"execute_workload": True}))
    assert set(handle2.extra.keys()) >= _PRE_954_SUCCESS_EXTRA_KEYS
    assert "workload_start_error" not in handle2.extra
    assert handle2.extra["workload_executed"] is True


def test_pre_provision_guard_keeps_handle_none(monkeypatch):
    """#954 AC2 input shape: the ``execute_workload``+empty-``workload_cmd``
    guard raises with ``handle is None`` and provision was never invoked —
    nothing was provisioned, nothing bills, so the rung's NoCompute blanket
    branch stays correct for it."""

    def _explode(*a, **k):
        raise AssertionError("provision must NOT run on the flag+empty-cmd cell")

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _explode)
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(
            _spec(workload_cmd="", hydra_args=("seed=1",), extra={"execute_workload": True})
        )
    assert ei.value.handle is None


# ---------------------------------------------------------------------------
# #1010 — CPU-fallback container-disk threading into the provision argv
# ---------------------------------------------------------------------------


def _recording_provision(monkeypatch) -> list[list[str]]:
    """No-op the ``pod_lifecycle.py provision`` call AND record its argv
    (the recording variant of ``_noop_provision`` — that fixture records
    nothing). Patches the #1465 pod_lifecycle-only helper
    ``RP._run_pod_lifecycle_relay`` — selective by construction."""
    argvs: list[list[str]] = []

    def _recording_relay(cmd, **k):
        argvs.append([str(c) for c in cmd])
        return None

    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _recording_relay)
    return argvs


def _flag_value(argv: list[str], flag: str) -> str | None:
    """The value following ``flag`` in ``argv``, or None when absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def _cpu_spec(intent: str, extra: dict | None = None) -> RunSpec:
    """A provision-only RunSpec with an explicit intent (the shared _spec
    helper pins intent="lora-7b", which collides with an override)."""
    return RunSpec(issue=1010, intent=intent, backend="runpod", extra=extra or {})


def test_launch_threads_container_disk_for_cpu_intent(monkeypatch):
    """#1010: a mapped CPU intent with a stated boot_disk_gb threads
    --container-disk-gb <value> into the provision argv."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("cpu-mid", {"boot_disk_gb": 80}))
    assert len(argvs) == 1
    assert _flag_value(argvs[0], "--container-disk-gb") == "80"


def test_launch_floors_container_disk_at_default(monkeypatch):
    """#1010: threading can never REDUCE below today's 50 GB default —
    a small stated requirement floors at max(50, boot_disk_gb)."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("cpu-mid", {"boot_disk_gb": 30}))
    assert _flag_value(argvs[0], "--container-disk-gb") == "50"


def test_launch_omits_container_disk_without_requirement(monkeypatch):
    """#1010 control: no stated requirement -> the provision argv is
    byte-identical to pre-#1010 (no --container-disk-gb flag at all)."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("cpu-mid"))
    assert "--container-disk-gb" not in argvs[0]


def test_launch_does_not_thread_container_disk_for_gpu_intent(monkeypatch):
    """#1010: GPU intents NEVER thread the container disk — on GPU pods the
    big-data mount is the /workspace VOLUME, not the container overlay
    (threading the overlay would silently inflate GPU container disks
    fleet-wide). As of #1118 boot_disk_gb DOES map on the GPU lane — to
    --volume-gb (see the #1118 section below), still never to
    --container-disk-gb."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("lora-7b", {"boot_disk_gb": 500}))
    assert "--container-disk-gb" not in argvs[0]


# ---------------------------------------------------------------------------
# #1118 — GPU-lane volume threading into the provision argv + handle persist
# ---------------------------------------------------------------------------


def test_launch_threads_volume_gb_for_gpu_intent(monkeypatch):
    """#1118: a GPU intent with a stated boot_disk_gb threads
    --volume-gb <value> into the provision argv (pod_lifecycle →
    runpod_api volumeInGb) and never the CPU-lane --container-disk-gb."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("lora-7b", {"boot_disk_gb": 575}))
    assert len(argvs) == 1
    assert _flag_value(argvs[0], "--volume-gb") == "575"
    assert "--container-disk-gb" not in argvs[0]


def test_launch_floors_volume_gb_at_default(monkeypatch):
    """#1118: threading can never REDUCE below today's 200 GB argparse
    default — a small stated requirement floors at max(200, boot_disk_gb)."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("lora-7b", {"boot_disk_gb": 100}))
    assert _flag_value(argvs[0], "--volume-gb") == "200"


def test_launch_omits_volume_gb_without_requirement(monkeypatch):
    """#1118 control: no stated requirement -> the provision argv is
    byte-identical to pre-#1118 (no --volume-gb flag at all)."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("lora-7b"))
    assert "--volume-gb" not in argvs[0]


def test_launch_does_not_thread_volume_gb_for_cpu_intent(monkeypatch):
    """#1118: CPU intents NEVER gain a --volume-gb flag — pod_lifecycle's CPU
    branch treats args.volume_gb == 200 as the 'unset' sentinel (its #747
    cheap-CPU volume default), which an explicit flag would defeat."""
    argvs = _recording_provision(monkeypatch)
    RP.RunPodBackend().launch(_cpu_spec("cpu-mid", {"boot_disk_gb": 80}))
    assert "--volume-gb" not in argvs[0]
    assert _flag_value(argvs[0], "--container-disk-gb") == "80"


def test_launch_malformed_boot_disk_gb_raises_named_valueerror(monkeypatch):
    """#1118: a malformed (non-integer) boot_disk_gb fails loud with a
    ValueError NAMING the key (mirroring router._footprint_int), raised
    BEFORE the provision subprocess — no pod is paid for."""
    argvs = _recording_provision(monkeypatch)
    with pytest.raises(ValueError, match="boot_disk_gb"):
        RP.RunPodBackend().launch(_cpu_spec("lora-7b", {"boot_disk_gb": "lots"}))
    assert argvs == []


def test_launch_fractional_boot_disk_gb_raises_named_valueerror(monkeypatch):
    """#1118 tightening: a fractional value (575.5) raises the same named
    ValueError instead of silently TRUNCATING to a smaller disk."""
    argvs = _recording_provision(monkeypatch)
    with pytest.raises(ValueError, match="boot_disk_gb"):
        RP.RunPodBackend().launch(_cpu_spec("lora-7b", {"boot_disk_gb": 575.5}))
    assert argvs == []


# ---------------------------------------------------------------------------
# #1669 — launch env pins: handle persist + launcher render + end-to-end
# ---------------------------------------------------------------------------

_PINS_1669 = {"WANDB_PROJECT": "issue1586_methodgen"}


def test_launch_handle_extra_carries_env_pins(monkeypatch):
    """#1669 (mirror of test_launch_handle_extra_carries_boot_disk_gb): a
    pinned spec's handle extra carries ``env_pins`` verbatim; a pin-less
    spec OMITS the key (the omit-when-absent contract the
    ``_PRE_954_SUCCESS_EXTRA_KEYS`` exact-set tests pin)."""
    _wire_exec_leg(monkeypatch, [])
    handle = RP.RunPodBackend().launch(_spec(extra={"env_pins": dict(_PINS_1669)}))
    assert handle.extra["env_pins"] == _PINS_1669

    _wire_exec_leg(monkeypatch, [])
    handle2 = RP.RunPodBackend().launch(_spec())
    assert "env_pins" not in handle2.extra


def test_launch_with_env_pins_renders_pin_export_before_default(monkeypatch):
    """#1669 END-TO-END incident-path test (#1586): pinned launch → handle →
    sidecar roundtrip → ``_runspec_from_runpod_handle`` → the router's
    ``execute_workload`` opt-in (the ``router.py`` failover ``replace``
    shape) → a REAL ``launch()`` through ``_wire_exec_leg`` (only
    ``_ssh_pod_run`` faked, so ``_execute_workload_on_pod`` runs for real)
    — and the RECORDED launcher body carries the pin export at an index
    BEFORE the ``${WANDB_PROJECT:-issue<N>}`` default line. This is the
    ONLY test spanning the renderer call-site kwarg thread: without it, an
    implementer who adds the renderer kwarg but forgets the call-site
    thread goes green while the failover pod boots with the generic
    default."""
    from dataclasses import replace

    from explore_persona_space.backends.issue_dispatch import (
        deserialize_handle,
        serialize_handle,
    )
    from scripts import backend_poll as bp

    # (1) The pinned launch persists env_pins into the handle sidecar.
    _wire_exec_leg(monkeypatch, [])
    handle = RP.RunPodBackend().launch(_spec(extra={"env_pins": dict(_PINS_1669)}))
    roundtripped = deserialize_handle(serialize_handle(handle))
    # (2) The failover reconstructor forwards them.
    spec2 = bp._runspec_from_runpod_handle(roundtripped, 909)
    assert spec2.extra["env_pins"] == _PINS_1669
    # (3) The failover opts into the execution leg (router.py's
    #     `replace(spec, extra={**dict(spec.extra or {}), "execute_workload": True})`).
    spec3 = replace(spec2, extra={**dict(spec2.extra or {}), "execute_workload": True})
    ssh = _wire_exec_leg(
        monkeypatch,
        ["SYNC-OK abc123\n", "WRAPPER-STARTED 4242\n", "LAUNCH-OK pid=777\n"],
    )
    handle2 = RP.RunPodBackend().launch(spec3)
    assert handle2.extra["workload_executed"] is True
    body = ssh.calls[1]  # sync -> DETACH (the launcher heredoc) -> verify
    pin_idx = body.index("export WANDB_PROJECT=issue1586_methodgen")
    default_idx = body.index('WANDB_PROJECT="${WANDB_PROJECT:-')
    assert pin_idx < default_idx, "pin export must precede the :-default line"


def test_render_launch_script_exports_shell_escaped_env_pins():
    """#1669: a pin value with a space + single-quote renders shlex-quoted,
    sorted, and positioned BEFORE the WANDB_PROJECT:-default line."""
    import shlex as _shlex

    tricky_value = "fu lora's group"  # space + single-quote → must shlex-quote
    body = RP._render_launch_script(
        issue=909,
        workload_cmd=WORKLOAD,
        log_path="/workspace/logs/issue-909.log",
        pid_file="/workspace/logs/issue-909.pid",
        sentinel_path="/workspace/eval_results/issue_909/att/s.json",
        attempt_id=ATTEMPT,
        env_pins={"WANDB_RUN_GROUP": tricky_value, "WANDB_PROJECT": "px"},
    )
    lines = body.splitlines()
    group_line = f"export WANDB_RUN_GROUP={_shlex.quote(tricky_value)}"
    assert group_line in lines
    proj_idx = lines.index("export WANDB_PROJECT=px")
    group_idx = lines.index(group_line)
    default_idx = next(i for i, ln in enumerate(lines) if 'WANDB_PROJECT="${WANDB_PROJECT:-' in ln)
    # Sorted (WANDB_PROJECT < WANDB_RUN_GROUP) and both before the default.
    assert proj_idx < group_idx < default_idx


def test_render_launch_script_no_pins_byte_identical():
    """#1669 (implementer note 2 — NO circular post-change fixture): a
    pin-less render carries NO pin export for any allowlisted key — the
    only ``export WANDB_PROJECT`` line is the ``:-`` default — and
    ``env_pins=None`` renders identically to ``env_pins={}`` (both take
    the no-pin path). Structural launcher content intact."""
    from explore_persona_space.backends.base import ENV_PIN_ALLOWED_KEYS

    kwargs = dict(
        issue=909,
        workload_cmd=WORKLOAD,
        log_path="/workspace/logs/issue-909.log",
        pid_file="/workspace/logs/issue-909.pid",
        sentinel_path="/workspace/eval_results/issue_909/att/s.json",
        attempt_id=ATTEMPT,
    )
    body_default = RP._render_launch_script(**kwargs)
    body_none = RP._render_launch_script(**kwargs, env_pins=None)
    body_empty = RP._render_launch_script(**kwargs, env_pins={})
    assert body_default == body_none == body_empty
    lines = body_default.splitlines()
    wandb_project_lines = [ln for ln in lines if ln.startswith("export WANDB_PROJECT")]
    assert wandb_project_lines == ['export WANDB_PROJECT="${WANDB_PROJECT:-issue909}"']
    for key in sorted(ENV_PIN_ALLOWED_KEYS - {"WANDB_PROJECT"}):
        assert not any(ln.startswith(f"export {key}=") for ln in lines), key
    # Structural content intact (heredoc + detach + pidfile echo).
    assert "EPSEOF" in body_default
    assert "setsid" in body_default
    assert "echo $$ > /workspace/logs/issue-909.pid" in body_default


def test_render_launch_script_rejects_non_allowlisted_pin_key():
    """#1669 defense in depth: the renderer re-validates independently of
    the CLI — a non-allowlisted key in a (hand-edited) sidecar raises."""
    with pytest.raises(ValueError, match="ENV_PIN_ALLOWED_KEYS"):
        RP._render_launch_script(
            issue=909,
            workload_cmd=WORKLOAD,
            log_path="/workspace/logs/issue-909.log",
            pid_file="/workspace/logs/issue-909.pid",
            sentinel_path="/workspace/eval_results/issue_909/att/s.json",
            attempt_id=ATTEMPT,
            env_pins={"WANDB_API_KEY": "x"},
        )


def test_launch_handle_extra_carries_boot_disk_gb(monkeypatch):
    """#1118: the launch handle's extra persists the footprint fields
    (boot_disk_gb / min_ram_gb) so _runspec_from_runpod_handle can forward
    them on the wedge / CUDA-IMA fresh-pod re-provision — and OMITS them
    when the spec states none (the legacy-shape / exact-key-set control)."""
    _recording_provision(monkeypatch)
    handle = RP.RunPodBackend().launch(
        _cpu_spec("lora-7b", {"boot_disk_gb": 575, "min_ram_gb": 32})
    )
    assert handle.extra["boot_disk_gb"] == 575
    assert handle.extra["min_ram_gb"] == 32

    handle2 = RP.RunPodBackend().launch(_cpu_spec("lora-7b"))
    assert "boot_disk_gb" not in handle2.extra
    assert "min_ram_gb" not in handle2.extra
