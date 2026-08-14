"""Poller pid-identity check + marker-signature liveness rescue (#1650).

A fresh-but-WRONG pid (a transient sibling captured at launch by a post-hoc
pgrep, or a recycled pid) defeats both existing liveness probes: the #1112
relaunch (2026-07-23) populated the pid file AND the ``epm:run-launched``
marker with the same wrong pid, so a healthy dispatcher read false ``dead``
twice. ``poll_pipeline`` now (a) derives a launch SIGNATURE from the marker's
``cmd='...'``/``launcher_script=`` fields, (b) captures the probed pids'
cmdlines in the SAME single-heredoc SSH round-trip and classifies them
``match | mismatch | unknown`` (mismatch = WARN + tick-JSON flag ONLY, never
a verdict change), and (c) adds ONE alive-direction verdict arm: when BOTH
probed pids are dead but live processes match the bracketed signature
pattern, ``pid_alive`` is rescued via a third OR term (``sig_proc_rescue``).

These tests pin:

* the pure predicates ``_launch_signature_tokens`` /
  ``_classify_pid_identity`` / ``_sig_pgrep_pattern`` (no SSH);
* the ``poll_once`` integration — the rescue flips the #1112 false ``dead``
  (and blocks a premature ``done`` via the #545 demotion reading the rescued
  ``pid_alive``); an identity mismatch WARNs without changing the verdict;
  a legacy free-prose marker leaves every arm inert (verdicts byte-identical
  to pre-#1650 behavior); a crashing classifier never breaks a tick;
* the probe heredoc emits the new keys in ONE SSH round-trip and the
  SSH-failure / legacy paths default every new key inert;
* the ``EPM_POLL_PID_IDENTITY=0`` kill switch disables the signature arms;
* the ``main()`` JSON line surfaces the three new enum/bool fields (raw
  cmdline text stays log-only).

Conventions copied from ``tests/test_poll_pipeline_stale_pid_warn.py``
(importlib loader, ``_probe_stdout`` string builder + ``pp.subprocess.run``
monkeypatch — the REAL ``_parse_probe_stdout`` parses the fake stdout).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_stale_pid_warn.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_pid_identity_under_test")

# Stable greppable substrings per WARN class (#1650).
_RESCUE_WARN_SUBSTR = "rescuing liveness"
_MISMATCH_WARN_SUBSTR = "does not match the launch signature"

# A non-`done`, non-`gate` tail so the verdict is the normal liveness path.
_RUNNING_TAIL = "2026-07-24 00:00:01 [phase=training step=5/100]"
# A training phase followed by a terminal done line (the #545 demotion needs
# a prior non-done phase to demote to).
_DONE_TAIL = (
    "2026-07-24 00:00:01 [phase=training step=100/100]\n"
    "2026-07-24 00:05:00 [phase=done] ALL PHASES COMPLETE"
)

# A #1092-shaped key=value marker note: cmd= carries the dispatch command
# (with redirects, whose .log target must NOT enter the signature) and
# launcher_script= the launcher path. Expected tokens are pinned in test 1.
_SIG_NOTE = (
    "pod=pod-9813 pid=12345 pid_file=/workspace/logs/issue-9813.pid "
    "cmd='bash scripts/issue9813_dispatch.sh --full > /workspace/logs/issue-9813.log 2>&1' "
    "launcher_script=/workspace/launch_issue_9813.sh "
    "log_abs=/workspace/logs/issue-9813.log"
)
_SIG_TOKENS = ("issue9813_dispatch.sh", "launch_issue_9813.sh")
# A #1112-shaped free-prose note (no cmd= / launcher_script= fields).
_PROSE_NOTE = (
    "relaunch 4 healthy: dispatcher resolved via recovery probe, ZeRO-3 "
    "recovery in flight; see prior note for details"
)

_MATCH_CMDLINE = "bash scripts/issue9813_dispatch.sh --full"
_FOREIGN_CMDLINE = "python scripts/train.py condition=c1_evil_wrong_em seed=7"


# ── probe builder ──────────────────────────────────────────────────────────────


def _probe_stdout(
    *,
    mtime_epoch: int,
    pod_now_epoch: int,
    tail: str,
    gpu_util: str,
    pid_file_missing: int = 0,
    pid_alive: int = 1,
    marker_pid_alive: int | None = None,
    pid_cmdline: str | None = None,
    marker_pid_cmdline: str | None = None,
    sig_proc_count: int | None = None,
    sig_proc_pids: str = "",
    results_sentinel_present: int = 0,
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    ``None`` for the optional scalars OMITS the corresponding line entirely
    (legacy / prose-marker replays — the parser must default them inert);
    any value emits it.
    """
    lines = [f"PID_FILE_MISSING={pid_file_missing}", f"PID_ALIVE={pid_alive}"]
    if pid_cmdline is not None:
        lines.append(f"PID_CMDLINE={pid_cmdline}")
    if marker_pid_alive is not None:
        lines.append(f"MARKER_PID_ALIVE={marker_pid_alive}")
    if marker_pid_cmdline is not None:
        lines.append(f"MARKER_PID_CMDLINE={marker_pid_cmdline}")
    if sig_proc_count is not None:
        lines.append(f"SIG_PROC_PIDS={sig_proc_pids}")
        lines.append(f"SIG_PROC_COUNT={sig_proc_count}")
    lines += [
        f"MTIME_EPOCH={mtime_epoch}",
        f"POD_NOW_EPOCH={pod_now_epoch}",
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
        "SESSION_CPU_SECS=unknown",
        f"RESULTS_SENTINEL_PRESENT={results_sentinel_present}",
    ]
    return "\n".join(lines)


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe_kwargs: dict[str, Any],
    marker_pid: int | None = None,
    marker_note: str = "",
    run_age_sec: float | None = 10800.0,
    capture: dict[str, str] | None = None,
) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Mirrors ``tests/test_poll_pipeline_stale_pid_warn.py::_patch_pod`` — the
    sentinel-drain SSH call returns empty; the probe call returns the
    controlled stdout (parsed by the REAL ``_parse_probe_stdout``); the
    events.jsonl reads are stubbed. New #1650 tests patch
    ``_marker_launch_fields`` directly (the note then flows through the REAL
    ``_launch_signature_tokens`` / ``_sig_pgrep_pattern``).
    """

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if capture is not None and cmd[0] == "ssh":
            capture["heredoc"] = remote
        stdout = _probe_stdout(**probe_kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.delenv("EPM_POLL_PID_IDENTITY", raising=False)
    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(
        pp, "_marker_launch_fields", lambda issue, pod=None: (marker_pid, marker_note)
    )
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch, pod=None: run_age_sec)


def _poll(tmp_path: Path):
    """Run ``poll_once`` with the standard fixture args."""
    return pp.poll_once(
        issue=9813,
        pod="pod-9813",
        log_path="/workspace/logs/issue-9813.log",
        pid_file="/workspace/logs/issue-9813.pid",
        state_file=tmp_path / "poll-state.json",
    )


def _now_epoch() -> int:
    import time

    return int(time.time())


# ── 1-4. pure predicates (no SSH) ─────────────────────────────────────────────


def test_signature_tokens_from_cmd_and_launcher() -> None:
    """A #1092-shaped key=value note yields exactly the dispatch + launcher
    basenames: generic interpreter tokens (``bash``) are dropped, redirect
    targets (``.log``) are dropped, non-path flags (``--full``) are dropped."""
    assert pp._launch_signature_tokens(_SIG_NOTE, 9813) == _SIG_TOKENS


def test_signature_tokens_empty_on_prose_marker() -> None:
    """A #1112-shaped free-prose note (the COMMON live case) yields () —
    every downstream signature consumer is inert on empty."""
    assert pp._launch_signature_tokens(_PROSE_NOTE, 1112) == ()
    assert pp._launch_signature_tokens("", 1112) == ()


def test_classify_match_mismatch_unknown() -> None:
    """Token-substring match; issue-token fallback (an exec'd workload whose
    cmdline shares no basename with the cmd= launcher string); mismatch on a
    foreign cmdline; unknown on empty cmdline / empty tokens; a log-READER
    cmdline never reads as match; the digit guard stops issue-number
    prefix confusion."""
    assert pp._classify_pid_identity(_MATCH_CMDLINE, _SIG_TOKENS, 9813) == "match"
    # Issue-token fallback: no basename overlap with the tokens, but the
    # cmdline is issue-keyed (`issue9813_...`).
    assert (
        pp._classify_pid_identity(
            "python scripts/issue9813_extract.py --full", ("mk_behavior_run.sh",), 9813
        )
        == "match"
    )
    assert pp._classify_pid_identity(_FOREIGN_CMDLINE, _SIG_TOKENS, 9813) == "mismatch"
    assert pp._classify_pid_identity("", _SIG_TOKENS, 9813) == "unknown"
    assert pp._classify_pid_identity(_MATCH_CMDLINE, (), 9813) == "unknown"
    # Reviewer concern (c): an obvious reader on an issue-keyed path is
    # never identity `match` (its args would hit the issue-token fallback).
    assert (
        pp._classify_pid_identity("tail -f /workspace/logs/issue-9813.log", _SIG_TOKENS, 9813)
        == "mismatch"
    )
    # Digit guard: issue 9813 must not match issue 98134's cmdline.
    assert (
        pp._classify_pid_identity(
            "python scripts/issue98134_dispatch.py", ("mk_behavior_run.sh",), 9813
        )
        == "mismatch"
    )


def test_sig_pattern_bracketed_and_escaped() -> None:
    """The pattern is the LONGEST token, regex-escaped, last char bracketed
    (``...s[h]``); the pattern string does NOT regex-match itself (the
    gotchas.md self-match guard); no usable token yields None."""
    pat = pp._sig_pgrep_pattern(_SIG_TOKENS)
    assert pat is not None and pat.endswith("s[h]"), pat
    assert pat == re.escape("issue9813_dispatch.s") + "[h]"
    assert re.search(pat, pat) is None  # self-match guard
    assert pp._sig_pgrep_pattern(()) is None
    assert pp._sig_pgrep_pattern(("trailing-",)) is None  # no alnum last char


# ── 5-8. poll_once integration (SSH boundary faked) ───────────────────────────


def test_rescue_flips_false_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The #1112 replay: pid-file pid AND marker pid both dead (the same
    wrong pid in both), healthy signature-matched processes live on the pod
    (SIG_PROC_COUNT=2). Pre-#1650 this read false ``dead``; now the rescue
    arm keeps ``pid_alive`` True, the verdict lands ``running`` via the
    normal liveness arbiters, and the rescue WARN names the recovery."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_alive=0,
            marker_pid_alive=0,
            sig_proc_count=2,
            sig_proc_pids="4242 4243",
        ),
        marker_pid=12345,
        marker_note=_SIG_NOTE,
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert result.status != "dead", result
    assert result.status == "running", result
    assert result.pid_alive is True
    assert result.sig_proc_rescue is True
    assert any(_RESCUE_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


def test_mismatch_warns_but_never_changes_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An ALIVE pid-file pid AND an ALIVE marker pid whose cmdlines both
    mismatch the signature (recycled/wrong-but-alive pids): the WARN fires
    and both identity flags read ``mismatch``, but the verdict is IDENTICAL
    to the same tick with matching cmdlines — mismatch never demotes
    ``pid_alive`` (plan #1650 §11 row 1)."""
    now = _now_epoch()
    mismatch_kwargs = dict(
        mtime_epoch=now - 30,
        pod_now_epoch=now,
        tail=_RUNNING_TAIL,
        gpu_util="95",
        pid_alive=1,
        pid_cmdline=_FOREIGN_CMDLINE,
        marker_pid_alive=1,
        marker_pid_cmdline=_FOREIGN_CMDLINE,
    )
    _patch_pod(monkeypatch, probe_kwargs=mismatch_kwargs, marker_pid=12345, marker_note=_SIG_NOTE)
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        mismatch_result = _poll(tmp_path)
    assert mismatch_result.pid_identity == "mismatch"
    assert mismatch_result.marker_pid_identity == "mismatch"  # marker-side assert (test 12)
    assert any(_MISMATCH_WARN_SUBSTR in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]
    # The verdict twin: identical tick except the cmdlines MATCH.
    match_kwargs = dict(
        mismatch_kwargs, pid_cmdline=_MATCH_CMDLINE, marker_pid_cmdline=_MATCH_CMDLINE
    )
    _patch_pod(monkeypatch, probe_kwargs=match_kwargs, marker_pid=12345, marker_note=_SIG_NOTE)
    match_result = _poll(tmp_path / "twin")
    assert match_result.pid_identity == "match"
    assert match_result.marker_pid_identity == "match"
    assert mismatch_result.status == match_result.status == "running"
    assert mismatch_result.pid_alive is match_result.pid_alive is True


def test_legacy_marker_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A free-prose marker (the COMMON live case): no SIG_PROC block is even
    built into the heredoc, both identities read ``unknown``, the rescue
    stays False, and the verdicts equal the pre-#1650 expectations (alive ⇒
    running; both-dead ⇒ dead — no signature exists to rescue with)."""
    now = _now_epoch()
    capture: dict[str, str] = {}
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_alive=1,
            marker_pid_alive=1,
        ),
        marker_pid=12345,
        marker_note=_PROSE_NOTE,
        capture=capture,
    )
    result = _poll(tmp_path)
    assert "SIG_PROC" not in capture["heredoc"], capture["heredoc"]
    assert result.status == "running"
    assert result.pid_identity == "unknown"
    assert result.marker_pid_identity == "unknown"
    assert result.sig_proc_rescue is False
    # Both pids dead on a prose marker: still `dead`, exactly as today.
    # (mtime is genuinely STALE + gpu idle so the #2265 dead-verdict
    # evidence veto stays inert — this test pins the SIGNATURE arms, and a
    # fresh log would now correctly read pid-stale-workload-live instead.)
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0",
            pid_alive=0,
            marker_pid_alive=0,
        ),
        marker_pid=12345,
        marker_note=_PROSE_NOTE,
    )
    dead_result = _poll(tmp_path / "dead")
    assert dead_result.status == "dead", dead_result
    assert dead_result.sig_proc_rescue is False


def test_fail_soft_never_crashes_tick(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A classifier raising inside ``_maybe_warn_pid_identity`` must fail
    INERT: the tick completes normally, both identities read ``unknown``
    (the test-enforced backing for the fail-soft claim — #1156 precedent)."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_alive=1,
            pid_cmdline=_MATCH_CMDLINE,
        ),
        marker_pid=None,
        marker_note=_SIG_NOTE,
    )

    def _boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("classifier boom (#1650 fail-soft test)")

    monkeypatch.setattr(pp, "_classify_pid_identity", _boom)
    result = _poll(tmp_path)
    assert result.status == "running", result
    assert result.pid_identity == "unknown"
    assert result.marker_pid_identity == "unknown"


# ── 9-10. probe heredoc + SSH-failure defaults ─────────────────────────────────


def test_heredoc_emits_new_keys_single_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ssh_probe`` emits PID_CMDLINE / MARKER_PID_CMDLINE / SIG_PROC_*
    inside its existing single heredoc — exactly ONE subprocess.run call per
    probe (no second SSH round-trip; plan acceptance criterion 2) — and
    omits the sig block when ``sig_pattern=None``."""
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    pp._ssh_probe(
        "pod-9813",
        "/workspace/logs/issue-9813.log",
        "/workspace/logs/issue-9813.pid",
        9813,
        12345,
        sig_pattern="issue9813_dispatch.s[h]",
    )
    assert len(calls) == 1, "sig probe must ride the SAME single SSH round-trip"
    heredoc = calls[0][-1]
    assert 'echo "PID_CMDLINE=' in heredoc
    assert 'echo "MARKER_PID_CMDLINE=' in heredoc
    assert "SIG_PROC_COUNT=" in heredoc and "pgrep -f" in heredoc
    # The pid-cmdline capture sits INSIDE the `[ -f pid_file ]` branch.
    present_idx = heredoc.index("PID_FILE_MISSING=0")
    absent_idx = heredoc.index("else echo PID_FILE_MISSING=1")
    assert present_idx < heredoc.index('echo "PID_CMDLINE=') < absent_idx, heredoc
    # No pattern -> no sig block; no marker pid -> no marker cmdline line.
    calls.clear()
    pp._ssh_probe(
        "pod-9813",
        "/workspace/logs/issue-9813.log",
        "/workspace/logs/issue-9813.pid",
        9813,
        None,
        sig_pattern=None,
    )
    assert len(calls) == 1
    assert "SIG_PROC" not in calls[0][-1]
    assert "MARKER_PID_CMDLINE" not in calls[0][-1]


def test_ssh_failure_defaults_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rc!=0 SSH-failure dict carries every new key at its inert default
    (classification ``unknown``, rescue impossible)."""

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        return subprocess.CompletedProcess(
            args=cmd, returncode=255, stdout="", stderr="ssh: connect refused"
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    out = pp._ssh_probe(
        "pod-9813",
        "/workspace/logs/issue-9813.log",
        "/workspace/logs/issue-9813.pid",
        9813,
        12345,
        sig_pattern="issue9813_dispatch.s[h]",
    )
    assert out["ssh_failed"] == "1"
    assert out["pid_cmdline"] == ""
    assert out["marker_pid_cmdline"] == ""
    assert out["sig_proc_count"] == "0"
    assert out["sig_proc_pids"] == ""


# ── 11. rescue blocks a premature done (#545 interplay) ────────────────────────


def test_rescue_blocks_premature_done(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A done-parse in the log tail + BOTH pids dead + a live
    signature-matched process + NO results sentinel: pre-#1650 the dead pid
    corroborated ``done``; the rescue makes ``pid_alive`` True so the #545
    demotion treats the done-parse as mid-run noise — consistent: a live
    signature-matched dispatcher means the run is not finished (plan §4
    verdict table row 6; safe direction — blocks a premature done)."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_DONE_TAIL,
            gpu_util="95",
            pid_alive=0,
            marker_pid_alive=0,
            sig_proc_count=1,
            sig_proc_pids="4242",
            results_sentinel_present=0,
        ),
        marker_pid=12345,
        marker_note=_SIG_NOTE,
    )
    result = _poll(tmp_path)
    assert result.status != "done", result
    assert result.status == "running", result
    assert result.sig_proc_rescue is True
    assert result.current_phase == "training", result  # demoted past the done line


# ── 12-13. kill switch + main() JSON surfaces ──────────────────────────────────


def test_kill_switch_disables_identity_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``EPM_POLL_PID_IDENTITY=0``: no sig block is built even on a
    signature-bearing marker, identities stay ``unknown``, and a both-dead
    tick verdicts ``dead`` exactly as pre-#1650 (the rescue arm is off).
    The evidence is genuinely stale (mtime >stall_sec, gpu idle) so the
    #2265 dead-verdict veto stays inert — this test pins the kill switch."""
    now = _now_epoch()
    capture: dict[str, str] = {}
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0",
            pid_alive=0,
            marker_pid_alive=0,
        ),
        marker_pid=12345,
        marker_note=_SIG_NOTE,
        capture=capture,
    )
    monkeypatch.setenv("EPM_POLL_PID_IDENTITY", "0")
    result = _poll(tmp_path)
    assert "SIG_PROC" not in capture["heredoc"], capture["heredoc"]
    assert result.status == "dead", result
    assert result.sig_proc_rescue is False
    assert result.pid_identity == "unknown"
    assert result.marker_pid_identity == "unknown"


def test_main_json_line_surfaces_identity_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main()`` surfaces the three #1650 fields in the tick JSON — and
    ONLY enum/bool values (raw cmdline text is log-only by contract)."""
    fake_result = pp.PollResult(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=30,
        pid_alive=True,
        pid_file_missing=False,
        log_tail_excerpt="",
        pid_identity="mismatch",
        marker_pid_identity="match",
        sig_proc_rescue=True,
    )
    monkeypatch.setattr(pp, "poll_once", lambda **kwargs: fake_result)
    rc = pp.main(["--issue", "1", "--pod", "p", "--log", "l", "--pid-file", "f"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["pid_identity"] == "mismatch"
    assert parsed["marker_pid_identity"] == "match"
    assert parsed["sig_proc_rescue"] is True
