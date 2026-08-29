"""Declared-done-file terminal success (#2610) — ``poll_once`` reads a
cleanly-completed SINGLE-PHASE dispatcher invocation as ``phase-done``.

The #2546 arm-3 incident (``p5_fits``): a single-phase dispatcher completed
43/43 units and wrote its done file, but — deliberately emitting no run-level
``[phase=done]`` — its pid probes went dead and the tick read the #2265
``pid-stale-workload-live`` (``stall_reason=pid_dead_evidence:log_fresh``),
which decays to a false ``dead`` once the tail-of-run evidence ages out.

Opt-in fix under test: an ``epm:run-launched`` note declaring
``done_file=<abs path>`` (shell-safe fullmatch allowlist) arms a fourth
probe leg — ``[ -f <done_file> ] && [ -f <pid_file> ] && [ <done_file> -nt
<pid_file> ]`` — and the pid-dead arbitration arm returns the
TERMINAL-SUCCESS ``phase-done`` BEFORE the #2265 veto when the file is
declared, present, and fresh. The ``[ -f <pid_file> ]`` conjunct is
LOAD-BEARING (bash ``-nt`` is TRUE when its second operand is missing).

These tests pin (the plan v3 11-case matrix):

1.  declared + fresh + pid dead -> ``phase-done`` (stall_reason None, no
    crash signature, short interval);
2.  declared + STALE done file (explicit ``DONE_FILE_FRESH=0``) -> the
    legacy #2265 verdicts are untouched;
3.  declared + ABSENT leg output (line omitted; parser default "0") ->
    same legacy verdicts (veto on fresh log; ``dead`` on all-stale);
4.  NOT declared -> byte-identical legacy (no probe leg, legacy verdicts);
5.  pid ALIVE + fresh done file -> never ``phase-done`` (arm unreached);
6.  extraction/validation units (`_done_file_from_note` /
    `_validate_done_file` / `_marker_done_file`): token extraction,
    metachar + relative + trailing-newline rejection (WARN + ignore),
    missing -> None;
7.  corroborated run-level ``[phase=done]`` outranks ``phase-done``;
8.  ssh-failure fallback dict defaults ``done_file_fresh`` to "0" -> no
    ``phase-done`` on a transport-dead tick;
9.  ``recommend_next_interval`` treats ``phase-done`` as non-"running"
    (short interval — act-now for the orchestrator);
10. no ``-> done`` milestone posts on a ``phase-done`` tick
    (``current_phase`` keeps the real phase);
11. `_done_file_probe` string: the three conjuncts in order (the
    ``-f <pid_file>`` middle conjunct pinned), "" when undeclared/invalid;
12. (round 2) an EMPTY ``done_file=`` token (whitespace / end-of-note
    right after the ``=``) -> WARN + ignore via the SAME rejection path —
    never a silent legacy opt-out; an empty token beside a later VALID
    declaration keeps the valid match (no WARN, behavior unchanged).

Conventions copied from ``tests/test_poll_pipeline_dead_veto.py`` (importlib
loader, ``_probe_stdout`` builder parsed by the REAL ``_parse_probe_stdout``,
``pp.subprocess.run`` monkeypatch so the REAL arbitration decides).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_dead_veto.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_phase_done_under_test")

_DONE_FILE = "/workspace/logs/issue-2610-p5_fits.done"
_PID_FILE = "/workspace/logs/issue-2610.pid"
# The declared launch note (free-form key=value convention, #2610 token).
_DECLARED_NOTE = f"launched p5_fits done_file={_DONE_FILE} log_abs=/workspace/logs/issue-2610.log"

# A non-`done`, non-`gate` tail so the verdict is the normal liveness path.
_RUNNING_TAIL = "2026-08-25 00:00:01 [phase=p5_fits unit 43/43]"
# Run-level terminal done line (#545 corroboration: dead pid corroborates).
_DONE_TAIL = (
    "2026-08-25 00:00:01 [phase=p5_fits unit 43/43]\n"
    "2026-08-25 00:05:00 [phase=done] ALL PHASES COMPLETE"
)


# ── probe builder ─────────────────────────────────────────────────────────────


def _probe_stdout(
    *,
    mtime_epoch: int,
    pod_now_epoch: int,
    tail: str,
    gpu_util: str,
    pid_alive: int = 1,
    output_mtime_epoch: int | None = None,
    results_sentinel_present: int = 0,
    done_file_fresh: int | None = None,
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    ``done_file_fresh=None`` OMITS the ``DONE_FILE_FRESH`` line (the parser
    defaults it to ``"0"`` — the undeclared / absent-leg shape); ``0``/``1``
    emit it explicitly (the declared-leg shapes).
    """
    lines = [
        "PID_FILE_MISSING=0",
        f"PID_ALIVE={pid_alive}",
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
    if output_mtime_epoch is not None:
        lines.append(f"OUTPUT_MTIME_EPOCH={output_mtime_epoch}")
    if done_file_fresh is not None:
        lines.append(f"DONE_FILE_FRESH={done_file_fresh}")
    return "\n".join(lines)


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe_kwargs: dict[str, Any],
    marker_note: str = "",
    ssh_rc: int = 0,
) -> MagicMock:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Mirrors ``tests/test_poll_pipeline_dead_veto.py::_patch_pod``, extended
    with ``marker_note`` so the REAL ``_done_file_from_note`` extraction runs
    against a controlled epm:run-launched note (no marker pid — the pid file
    is the sole liveness probe). Returns the ``post_event`` mock so tests
    can assert on posted milestones.
    """

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if ssh_rc != 0:
            return subprocess.CompletedProcess(
                args=cmd, returncode=ssh_rc, stdout="", stderr="ssh: connect refused"
            )
        stdout = _probe_stdout(**probe_kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    post_event_mock = MagicMock()
    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", post_event_mock)
    monkeypatch.setattr(pp, "_marker_launch_fields", lambda issue, pod=None: (None, marker_note))
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch, pod=None: 10800.0)
    return post_event_mock


def _poll(tmp_path: Path):
    """Run ``poll_once`` with the standard fixture args."""
    return pp.poll_once(
        issue=2610,
        pod="pod-2610",
        log_path="/workspace/logs/issue-2610.log",
        pid_file=_PID_FILE,
        state_file=tmp_path / "poll-state.json",
    )


def _now_epoch() -> int:
    return int(time.time())


# ── 1. declared + fresh + pid dead -> phase-done ─────────────────────────────


def test_declared_fresh_pid_dead_reads_phase_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The #2546 arm-3 replay with the declaration armed: pid dead, fresh
    tail-of-run log + busy-free GPUs, done file declared + fresh -> the
    TERMINAL-SUCCESS ``phase-done`` (not the veto verdict), stall_reason
    None, no crash signature, short interval, INFO breadcrumb names the
    file."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 120,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
            done_file_fresh=1,
        ),
    )
    with caplog.at_level(logging.INFO, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert result.status == pp.STATUS_PHASE_DONE, result
    assert result.status == "phase-done"  # the literal contract token
    assert result.stall_reason is None
    assert result.pid_alive is False
    assert result.crash_signature is None  # success, never a crash capture
    assert result.next_interval == pp.POLL_INTERVAL_DEFAULT_SEC
    assert any(
        _DONE_FILE in rec.getMessage() and "phase-done" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


# ── 2. declared + STALE done file -> legacy #2265 verdicts ────────────────────


def test_declared_stale_done_file_keeps_legacy_verdicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit ``DONE_FILE_FRESH=0`` (file older than the pid file — a
    PRIOR run's stale artifact, the #779 class): never ``phase-done``; the
    #2265 veto fires on the fresh log exactly as pre-#2610."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 120,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
            done_file_fresh=0,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == pp.STATUS_PID_STALE_WORKLOAD_LIVE, result
    assert result.stall_reason == "pid_dead_evidence:log_fresh"


# ── 3. declared + ABSENT leg output -> parser default "0" -> legacy ───────────


def test_declared_absent_leg_output_defaults_to_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No ``DONE_FILE_FRESH`` line at all (absent done file / legacy probe
    replay): the parser defaults to "0" — veto on a fresh log, ``dead`` on
    all-stale evidence."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 120,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == pp.STATUS_PID_STALE_WORKLOAD_LIVE, result
    assert result.stall_reason == "pid_dead_evidence:log_fresh"

    _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path / "all-stale")
    assert result.status == "dead", result
    assert result.stall_reason is None


# ── 4. NOT declared -> byte-identical legacy ──────────────────────────────────


def test_undeclared_launch_is_byte_identical_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A free-prose marker with no ``done_file=`` token: extraction returns
    None, no probe leg is composed, and both legacy pid-dead verdicts are
    unchanged — even when a stray fresh done-file-shaped probe value would
    have read 1 (it cannot: undeclared means the leg never emits)."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        marker_note="launched pid_file=/workspace/logs/issue-2610.pid free prose",
        probe_kwargs=dict(
            mtime_epoch=now - 120,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == pp.STATUS_PID_STALE_WORKLOAD_LIVE, result
    assert result.stall_reason == "pid_dead_evidence:log_fresh"

    _patch_pod(
        monkeypatch,
        marker_note="",
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path / "empty-note")
    assert result.status == "dead", result


def test_undeclared_pid_dead_verdict_ignores_stray_fresh_flag() -> None:
    """Unit pin on the arm itself: ``done_file=None`` (undeclared) NEVER
    returns ``phase-done`` regardless of ``done_file_fresh`` — the
    declaration is the opt-in key, the probe value alone cannot arm it."""
    status, stall_reason = pp._pid_dead_verdict(
        pod="pod-2610",
        last_mtime_ago=10**9,
        phase_log_mtime_ago=10**9,
        shard_log_mtime_ago=10**9,
        output_mtime_ago=10**9,
        gpu_util="unknown",
        stall_sec=pp.DEFAULT_STALL_SEC,
        done_file=None,
        done_file_fresh=True,
    )
    assert status == "dead"
    assert stall_reason is None


# ── 5. pid ALIVE + fresh done file -> never phase-done ────────────────────────


def test_pid_alive_with_fresh_done_file_stays_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pid-dead arm is the ONLY reader of the done-file leg: a live pid
    with a fresh declared done file (e.g. the file just landed while the
    launcher's exit races the tick) keeps the normal ``running`` verdict."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_alive=1,
            done_file_fresh=1,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == "running", result
    assert result.stall_reason is None


# ── 6. extraction / validation units ─────────────────────────────────────────


def test_done_file_extraction_and_validation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`_done_file_from_note` extracts the token; the shell-safe fullmatch
    allowlist rejects metachars, relative paths, and trailing newlines
    (WARN + None — the declaration is ignored, never interpolated); a
    missing token is None; `_marker_done_file` mirrors via the marker
    resolver."""
    # Extraction happy path (token embedded in free prose).
    assert pp._done_file_from_note(_DECLARED_NOTE) == _DONE_FILE
    # Missing token / empty / None-ish note -> None, no WARN.
    assert pp._done_file_from_note("launched pid=123 free prose") is None
    assert pp._done_file_from_note("") is None

    # Rejections: each WARNs and returns None.
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        # Shell metacharacters (injection fence).
        assert pp._done_file_from_note("done_file=/tmp/x;rm-rf.done") is None
        assert pp._done_file_from_note("done_file=/tmp/$(id).done") is None
        # Relative path (allowlist requires a leading /).
        assert pp._done_file_from_note("done_file=relative/x.done") is None
        # Trailing newline: `\S+` cannot capture it, but the validator must
        # ALSO reject it directly — fullmatch, not a `$`-anchored search
        # (`$` matches before a final newline and would accept this).
        assert pp._validate_done_file("/workspace/x.done\n") is None
    warn_count = sum(
        "rejected by the shell-safe allowlist" in rec.getMessage() for rec in caplog.records
    )
    assert warn_count == 4, [rec.getMessage() for rec in caplog.records]

    # The validator accepts the plain absolute-path shape.
    assert pp._validate_done_file(_DONE_FILE) == _DONE_FILE

    # `_marker_done_file` mirrors the extraction off the latest marker.
    monkeypatch.setattr(
        pp, "_latest_run_launched_event", lambda issue, pod=None: {"note": _DECLARED_NOTE}
    )
    assert pp._marker_done_file(2610) == _DONE_FILE
    monkeypatch.setattr(pp, "_latest_run_launched_event", lambda issue, pod=None: None)
    assert pp._marker_done_file(2610) is None


def test_empty_done_file_token_warns_and_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Case 12 (#2610 round 2): an EMPTY ``done_file=`` token (nothing
    between the ``=`` and the next whitespace / end of note — a launcher
    typo) cannot match the ``\\S+`` capture, so pre-fix it silently
    restored the legacy pid-only verdicts; it now routes through the SAME
    rejection WARN path (declaration ignored — behavior unchanged,
    observability fixed)."""
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        # Mid-note empty token (whitespace follows the `=`).
        assert pp._done_file_from_note("launched pid=123 done_file= log_abs=/x.log") is None
        # Note-final empty token (end of string follows the `=`).
        assert pp._done_file_from_note("launched pid=123 done_file=") is None
        # Trailing-newline note (the `\\s` alternative of the probe).
        assert pp._done_file_from_note("done_file=\n") is None
    warns = [
        rec.getMessage()
        for rec in caplog.records
        if "rejected by the shell-safe allowlist" in rec.getMessage()
    ]
    assert len(warns) == 3, warns

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        # A note with NO done_file token stays a silent None (no WARN) —
        # the common legacy case is unchanged.
        assert pp._done_file_from_note("launched pid=123 free prose") is None
        # An empty token BESIDE a later valid declaration: the `\S+`
        # search skips the empty occurrence and the valid one still wins
        # (no WARN) — pre-round-2 match behavior preserved.
        assert pp._done_file_from_note(f"done_file= done_file={_DONE_FILE}") == _DONE_FILE
    stray = [
        rec.getMessage()
        for rec in caplog.records
        if "rejected by the shell-safe allowlist" in rec.getMessage()
    ]
    assert not stray, stray


# ── 7. run-level done outranks phase-done ─────────────────────────────────────


def test_run_level_done_outranks_phase_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Arm-ordering pin: a corroborated run-level ``[phase=done]`` (terminal
    line + dead pid, #545) precedes the pid-dead arm, so the tick reads
    ``done`` — never ``phase-done`` — even with the declared file fresh."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_DONE_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
            done_file_fresh=1,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == "done", result
    assert result.stall_reason is None


# ── 8. ssh-failure fallback -> done_file_fresh defaults "0" ───────────────────


def test_ssh_failure_tick_never_phase_done(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A transport-dead tick's zeroed fallback dict carries
    ``done_file_fresh="0"``, so a declared launch still reads ``dead``
    (zero evidence) — never a phase-done minted off an unprobed pod."""
    _patch_pod(monkeypatch, marker_note=_DECLARED_NOTE, probe_kwargs={}, ssh_rc=255)
    result = _poll(tmp_path)
    assert result.status == "dead", result
    assert result.stall_reason is None


# ── 9. recommend_next_interval: phase-done is act-now ─────────────────────────


def test_recommend_next_interval_phase_done_short() -> None:
    """``phase-done`` is non-"running", so the adaptive interval stays SHORT
    (the orchestrator acts on the terminal-success tick immediately); the
    same otherwise-quiet inputs under ``running`` read QUIET — proving the
    status alone forces the short interval."""
    quiet_kwargs = dict(
        gate=None,
        sentinels_processed=0,
        phase_transitioned=False,
        ssh_failed=False,
        gpu_idle_advisory_posted=False,
        cpu_override_active=False,
        run_age_sec=10.0**6,
        phase_changed_ago_sec=10.0**6,
    )
    assert (
        pp.recommend_next_interval(status=pp.STATUS_PHASE_DONE, **quiet_kwargs)
        == pp.POLL_INTERVAL_DEFAULT_SEC
    )
    assert pp.recommend_next_interval(status="running", **quiet_kwargs) == (
        pp.POLL_INTERVAL_QUIET_SEC
    )


# ── 10. no `-> done` milestone on a phase-done tick ───────────────────────────


def test_no_done_milestone_on_phase_done_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``current_phase`` keeps the REAL last-parsed phase on a ``phase-done``
    tick (no synthetic ``-> done`` transition), so the milestone tracker can
    never post a ``-> done`` progress marker off this verdict."""
    now = _now_epoch()
    post_event_mock = _patch_pod(
        monkeypatch,
        marker_note=_DECLARED_NOTE,
        probe_kwargs=dict(
            mtime_epoch=now - 120,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
            done_file_fresh=1,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == pp.STATUS_PHASE_DONE, result
    assert result.current_phase == "p5_fits"  # the real phase, never "done"
    for call in post_event_mock.call_args_list:
        note = call.kwargs.get("note", "")
        assert "-> done" not in note, call


# ── 11. probe-string conjuncts ────────────────────────────────────────────────


def test_done_file_probe_string_conjuncts() -> None:
    """The bash leg carries all THREE conjuncts in order — done-file
    existence, PID-FILE existence (load-bearing: bash ``-nt`` is TRUE when
    its second operand is missing), then ``-nt`` — and composes to "" when
    undeclared or invalid (defensive re-validation)."""
    leg = pp._done_file_probe(_DONE_FILE, _PID_FILE)
    expected = (
        f"if [ -f {_DONE_FILE} ] && [ -f {_PID_FILE} ] && [ {_DONE_FILE} -nt {_PID_FILE} ]; "
        f"then echo DONE_FILE_FRESH=1; else echo DONE_FILE_FRESH=0; fi; "
    )
    assert leg == expected
    # The middle conjunct specifically (never drop it in a refactor).
    assert f"&& [ -f {_PID_FILE} ] &&" in leg
    # Undeclared -> no leg.
    assert pp._done_file_probe(None, _PID_FILE) == ""
    # Defensive re-validation: an unvalidated direct caller cannot inject.
    assert pp._done_file_probe("/tmp/x;rm.done", _PID_FILE) == ""

    # The composed heredoc parses back through the REAL parser: a declared
    # leg's emission lands under the lowercased key.
    parsed = pp._parse_probe_stdout("DONE_FILE_FRESH=1")
    assert parsed["done_file_fresh"] == "1"
    assert pp._parse_probe_stdout("")["done_file_fresh"] == "0"
    assert "DONE_FILE_FRESH" in pp._PROBE_SCALAR_KEYS
