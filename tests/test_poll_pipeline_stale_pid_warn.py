"""Stale-pid-file-vs-marker WARN backstop (#1156).

A relaunch that skips the pid-file rewrite contract
(`.claude/rules/pod-side-reporting.md` § Pid-file launch contract, #813 v5)
leaves a PRIOR launch's dead pid in ``/workspace/logs/issue-<N>.pid``. A
PRESENT-but-stale pid file is worse than a missing one: the #521
``pid_file_missing`` WARN covers only absence, so the stale file is silently
probed every tick. ``poll_pipeline`` now compares the pid file's pod-clock
AGE (``POD_NOW_EPOCH - PID_FILE_MTIME_EPOCH``, drift-free per #704) against
the VM-clock age of the newest ``epm:run-launched`` marker and WARNs — plus
sets the observability-only ``pid_file_stale_vs_marker`` tick-JSON flag —
when the pid file predates the marker by more than
``PID_FILE_MARKER_SLACK_SEC`` (600 s default, env
``EPM_POLL_PID_MARKER_SLACK_SEC``). Never a verdict change.

These tests pin:

* the pure predicate ``_pid_file_predates_marker`` — normal launch ordering
  (pid file written BEFORE the marker, small positive delta) stays silent;
  a prior-launch-aged pid file fires; the boundary is a strict ``>``;
  missing inputs are inert; the marker-first (GCP pre-launch-signal)
  ordering never fires;
* the ``poll_once`` integration — the WARN + flag fire on the #813 shape
  with the verdict untouched, under a 14 h pod-vs-VM drift in EITHER
  direction (drift-free same-clock arithmetic); missing pid file / missing
  marker / malformed mtime are silent and never crash the tick (the
  fail-loud-acceptance backing for the fail-soft contract);
* the probe heredoc emits ``PID_FILE_MTIME_EPOCH`` inside the ``[ -f ]``
  branch; the ``main()`` JSON line surfaces the flag.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_clock_skew.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_stale_pid_under_test")

# The new WARN's identifying substring (asserted present/absent in caplog; the
# #521 absent-file WARN says "absent on pod" and never matches this).
_WARN_SUBSTR = "predates the newest epm:run-launched"

# A non-`done`, non-`gate` tail so the verdict is the normal liveness path.
_RUNNING_TAIL = "2026-07-09 00:00:01 [phase=training step=5/100]"
# 14h pod-vs-VM drift: a cross-clock implementation would mis-read ages by
# ~50400s in one direction or the other (tests 6 / 7 / 7b bracket both).
_DRIFT_SEC = 14 * 3600


# ── probe builder ──────────────────────────────────────────────────────────────


def _probe_stdout(
    *,
    mtime_epoch: int,
    pod_now_epoch: int | None,
    tail: str,
    gpu_util: str,
    pid_file_missing: int = 0,
    pid_alive: int = 1,
    marker_pid_alive: int | None = None,
    pid_file_mtime_epoch: int | str | None = None,
    session_cpu: str = "unknown",
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    ``pod_now_epoch=None`` / ``pid_file_mtime_epoch=None`` /
    ``marker_pid_alive=None`` OMIT the corresponding line entirely (legacy /
    absent-branch replays); any value emits it. ``pid_file_mtime_epoch``
    accepts a str so the malformed-scalar case is expressible.
    """
    lines = [f"PID_FILE_MISSING={pid_file_missing}"]
    if pid_file_mtime_epoch is not None:
        lines.append(f"PID_FILE_MTIME_EPOCH={pid_file_mtime_epoch}")
    lines.append(f"PID_ALIVE={pid_alive}")
    if marker_pid_alive is not None:
        lines.append(f"MARKER_PID_ALIVE={marker_pid_alive}")
    lines.append(f"MTIME_EPOCH={mtime_epoch}")
    if pod_now_epoch is not None:
        lines.append(f"POD_NOW_EPOCH={pod_now_epoch}")
    lines += [
        "TAIL_START",
        tail,
        "TAIL_END",
        "CELL_MTIME_EPOCH=0",
        "CELL_TAIL_START",
        "CELL_TAIL_END",
        "PHASE_LOG_MTIME_EPOCH=0",
        "SHARD_LOG_MTIME_EPOCH=0",
        f"GPU_UTIL={gpu_util}",
        "ZOMBIE_GPU_PIDS=",
        f"SESSION_CPU_SECS={session_cpu}",
        "RESULTS_SENTINEL_PRESENT=0",
    ]
    return "\n".join(lines)


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe_kwargs: dict[str, Any],
    marker_pid: int | None = None,
    run_age_sec: float | None = 10800.0,
) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Mirrors ``tests/test_poll_pipeline_clock_skew.py::_patch_pod`` — the
    sentinel-drain SSH call returns empty; the probe call returns the
    controlled stdout; ``_marker_pid`` / ``_run_launched_age_sec`` (both
    events.jsonl reads) are stubbed with the given values.
    """

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        stdout = "" if "SENTINEL_START" in remote else _probe_stdout(**probe_kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: marker_pid)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: run_age_sec)


def _poll(tmp_path: Path):
    """Run ``poll_once`` with the standard fixture args."""
    return pp.poll_once(
        issue=9813,
        pod="pod-9813",
        log_path="/workspace/logs/issue-9813.log",
        pid_file="/workspace/logs/issue-9813.pid",
        state_file=tmp_path / "poll-state.json",
    )


# ── 1-5. pure predicate (no SSH) ───────────────────────────────────────────────


def test_predicate_normal_launch_ordering_no_warn() -> None:
    """The EXPECTED ordering — pid file written 90 s before the marker posts
    (pid_age = marker_age + 90) — must NOT fire: delta > 0 on EVERY healthy
    launch, so a slack-0-like regression would warn on every tick."""
    pod_now = 1_800_000_000
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=pod_now - 3690,  # pid_age 3690 = marker_age + 90
            pod_now_epoch=pod_now,
            run_age_sec=3600.0,
            slack_sec=600,
        )
        is False
    )


def test_predicate_stale_prior_launch_pid_warns() -> None:
    """A 2h-older pid file (the #813 prior-launch shape) fires."""
    pod_now = 1_800_000_000
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=pod_now - (3600 + 7200),  # pid_age = marker_age + 7200
            pod_now_epoch=pod_now,
            run_age_sec=3600.0,
            slack_sec=600,
        )
        is True
    )


def test_predicate_boundary_at_slack_no_warn() -> None:
    """pid_age exactly marker_age + slack is silent (strict ``>`` — pins the
    boundary so a future ``>=`` regression is caught)."""
    pod_now = 1_800_000_000
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=pod_now - (3600 + 600),  # pid_age = marker_age + slack
            pod_now_epoch=pod_now,
            run_age_sec=3600.0,
            slack_sec=600,
        )
        is False
    )


def test_predicate_missing_inputs_never_warn() -> None:
    """Each missing input — pid-file mtime 0, pod-now 0, marker age None —
    is inert even when every OTHER input reads as stale."""
    pod_now = 1_800_000_000
    stale_mtime = pod_now - (3600 + 7200)
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=0,
            pod_now_epoch=pod_now,
            run_age_sec=3600.0,
            slack_sec=600,
        )
        is False
    )
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=stale_mtime,
            pod_now_epoch=0,
            run_age_sec=3600.0,
            slack_sec=600,
        )
        is False
    )
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=stale_mtime,
            pod_now_epoch=pod_now,
            run_age_sec=None,
            slack_sec=600,
        )
        is False
    )


def test_predicate_marker_newer_than_pid_file_no_warn() -> None:
    """Marker posted BEFORE the workload booted (pid_age < marker_age — the
    GCP pre-launch-signal ordering) never fires."""
    pod_now = 1_800_000_000
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=pod_now - 120,  # pid_age 120 < marker_age 3600
            pod_now_epoch=pod_now,
            run_age_sec=3600.0,
            slack_sec=600,
        )
        is False
    )


# ── 6-10. poll_once integration (SSH boundary faked) ───────────────────────────


def test_poll_once_stale_pid_warns_sets_field_keeps_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The exact #813 shape under a 14h drift (pod clock BEHIND the VM):
    stale pid file (3h-old on the pod clock vs a 1h-old marker), dead
    pidfile pid rescued by a live marker pid. The WARN + flag fire and the
    verdict stays ``running`` (acceptance criterion 1) with
    ``pid_alive=True`` — the backstop never touches status routing."""
    vm_now = int(time.time())
    pod_now = vm_now - _DRIFT_SEC
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 30,  # fresh log on the pod clock
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_alive=0,  # pid-file pid dead (the stale prior-launch pid)
            marker_pid_alive=1,  # live relaunch pid via the marker OR-probe
            pid_file_mtime_epoch=pod_now - 10800,  # 3h-old pid file
        ),
        marker_pid=12345,
        run_age_sec=3600.0,  # marker 1h old → pid_age exceeds by 7200 > 600
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert any(_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]
    assert result.pid_file_stale_vs_marker is True
    assert result.status == "running", result
    assert result.pid_alive is True


def test_poll_once_normal_ordering_no_warn_under_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Normal ordering (delta = 90 s) under the same 14h pod-BEHIND drift:
    no WARN, flag False. A cross-clock implementation would read the pid-file
    age as ~+14h here and FALSE-FIRE (this test + the pod-AHEAD sibling
    bracket both drift directions — acceptance criterion 4)."""
    vm_now = int(time.time())
    pod_now = vm_now - _DRIFT_SEC
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 30,
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_file_mtime_epoch=pod_now - 3690,  # pid_age 3690 = marker_age + 90
        ),
        run_age_sec=3600.0,
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert not any(_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]
    assert result.pid_file_stale_vs_marker is False


def test_poll_once_normal_ordering_no_warn_pod_clock_ahead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Opposite drift sign — pod clock 14h AHEAD of the VM — on the same
    normal-ordering case: still silent, flag False, tick completes. Pins the
    false-SILENCE direction of a cross-clock bug (a VM-clock pid-file age
    would read hugely NEGATIVE here, so any implementation whose behavior
    depends on the drift sign — mis-firing on negatives, or crashing —
    is caught; the same-clock arithmetic is sign-invariant)."""
    vm_now = int(time.time())
    pod_now = vm_now + _DRIFT_SEC  # pod clock AHEAD of the VM clock
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 30,
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_file_mtime_epoch=pod_now - 3690,  # pid_age 3690 = marker_age + 90
        ),
        run_age_sec=3600.0,
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert not any(_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]
    assert result.pid_file_stale_vs_marker is False


def test_poll_once_missing_pid_file_no_stale_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A MISSING pid file short-circuits the new WARN (the #521 absent-file
    WARN owns that case — no double-fire), even when a stale-looking mtime
    scalar rides the probe stdout. Asserts on the NEW substring only (the
    #521 WARN legitimately fires here)."""
    vm_now = int(time.time())
    pod_now = vm_now
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 30,
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_file_missing=1,
            pid_alive=0,
            marker_pid_alive=1,
            pid_file_mtime_epoch=pod_now - 10800,  # would fire if not short-circuited
        ),
        marker_pid=12345,
        run_age_sec=3600.0,
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert not any(_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]
    assert result.pid_file_stale_vs_marker is False


def test_poll_once_no_marker_no_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No epm:run-launched marker (``_run_launched_age_sec`` → None): the
    predicate is inert — no WARN, flag False, no crash."""
    vm_now = int(time.time())
    pod_now = vm_now
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 30,
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_file_mtime_epoch=pod_now - 10800,  # 3h-old pid file, but no marker
        ),
        marker_pid=None,
        run_age_sec=None,
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert not any(_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]
    assert result.pid_file_stale_vs_marker is False


def test_poll_once_malformed_mtime_fail_soft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A garbled probe scalar (``PID_FILE_MTIME_EPOCH=garbage``) must fail
    INERT: the tick completes normally (no exception — this test + the
    missing-pid-file / no-marker tests are the test-enforced fail-soft
    contract, plan check 15), flag False."""
    vm_now = int(time.time())
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=vm_now - 30,
            pod_now_epoch=vm_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_file_mtime_epoch="garbage",
        ),
        run_age_sec=3600.0,
    )
    result = _poll(tmp_path)
    assert result.status == "running", result
    assert result.pid_file_stale_vs_marker is False


# ── 11. heredoc emits the new capture line ──────────────────────────────────────


def test_heredoc_emits_pid_file_mtime_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe heredoc emits ``PID_FILE_MTIME_EPOCH=$(stat -c %Y ...)``
    INSIDE the ``[ -f ]`` branch — after ``PID_FILE_MISSING=0`` and before
    the ``else`` arm. Capture scoped to ``cmd[0] == "ssh"`` (mirrors the
    clock-skew heredoc test — ``poll_once`` also runs non-ssh subprocesses
    through the mocked ``subprocess.run``)."""
    captured: dict[str, str] = {}
    vm_now = int(time.time())

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "ssh":
            captured["heredoc"] = remote
        stdout = _probe_stdout(
            mtime_epoch=vm_now - 30,
            pod_now_epoch=vm_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)

    _poll(tmp_path)
    assert "heredoc" in captured, "probe ssh call was never made"
    heredoc = captured["heredoc"]
    m = re.search(r"PID_FILE_MTIME_EPOCH=\$\(stat -c %Y", heredoc)
    assert m, heredoc
    present_idx = heredoc.index("PID_FILE_MISSING=0")
    absent_idx = heredoc.index("else echo PID_FILE_MISSING=1")
    assert present_idx < m.start() < absent_idx, heredoc


# ── 12. main() JSON line surfaces the flag ──────────────────────────────────────


def test_main_json_line_surfaces_stale_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` surfaces ``pid_file_stale_vs_marker`` in the tick JSON
    (the machine-readable channel the orchestrator consumes)."""
    fake_result = pp.PollResult(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=30,
        pid_alive=True,
        pid_file_missing=False,
        log_tail_excerpt="",
        pid_file_stale_vs_marker=True,
    )
    monkeypatch.setattr(pp, "poll_once", lambda **kwargs: fake_result)
    rc = pp.main(["--issue", "1", "--pod", "p", "--log", "l", "--pid-file", "f"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pid_file_stale_vs_marker"] is True
