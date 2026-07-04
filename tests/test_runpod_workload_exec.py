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
import re
import subprocess

import pytest

from explore_persona_space.backends import runpod as RP
from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
from explore_persona_space.backends.base import RunSpec

WORKLOAD = "bash scripts/issue909_dispatch.sh --arm a"
ATTEMPT = "rp-20260703T000000Z-ab12"


def _noop_provision(monkeypatch) -> None:
    """No-op ONLY the ``pod_lifecycle.py provision`` subprocess call.

    ``subprocess.run`` is the module-global singleton (the artifact
    declaration helpers use it too), so a blanket lambda would break them —
    the selective pattern from ``tests/test_runpod_wedge_detection.py``.
    """
    _real_run = RP.subprocess.run

    def _selective_run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any("pod_lifecycle.py" in str(c) for c in cmd):
            return None
        return _real_run(cmd, *a, **k)

    monkeypatch.setattr(RP.subprocess, "run", _selective_run, raising=False)


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
    ssh = _wire_exec_leg(
        monkeypatch,
        ["SYNC-OK abc123\n", "WRAPPER-STARTED 4242\n", "LAUNCH-OK pid=777\n"],
    )
    handle = RP.RunPodBackend().launch(
        _spec(extra={"execute_workload": True, "repo_branch": "issue-909"})
    )
    # Exactly 3 SSH calls, in order: sync -> detach -> verify.
    assert len(ssh.calls) == 3
    sync_cmd, launch_cmd, verify_cmd = ssh.calls
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
    _wire_exec_leg(monkeypatch, ["SYNC-MISMATCH head=aaa fetch=bbb\n"])
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(
            _spec(extra={"execute_workload": True, "repo_branch": "issue-909"})
        )
    msg = str(ei.value)
    assert "pod-909" in msg and "issue-909" in msg
    assert "SYNC-MISMATCH" in msg


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

    monkeypatch.setattr(RP.subprocess, "run", _explode, raising=False)
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
    ]


def test_remote_scripts_never_shell_task_py():
    """Pods never shell scripts/task.py (CLAUDE.md hard rule; complements
    tests/test_no_pod_side_task_py_shellout.py) — and never emit a bare
    ``[phase=`` literal (the poller's reserved token, experimenter.md)."""
    for script in _rendered_scripts():
        assert "task.py" not in script
        assert "[phase=" not in script


def test_rendered_scripts_bash_n(tmp_path):
    """All three remote scripts parse under ``bash -n``, including a
    quoting-stress workload_cmd carrying a single quote, ``$VAR``, and
    ``&&`` (the GCP ``test_render_startup_script_is_valid_bash`` precedent)."""
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
    _wire_exec_leg(monkeypatch, [])

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

    monkeypatch.setattr(RP.subprocess, "run", _explode, raising=False)
    with pytest.raises(RP.RunPodWorkloadStartError) as ei:
        RP.RunPodBackend().launch(
            _spec(workload_cmd="", hydra_args=("seed=1",), extra={"execute_workload": True})
        )
    assert ei.value.handle is None
