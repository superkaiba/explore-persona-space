"""Tests for poll_pipeline.py's pod-side sentinel-drain step.

Background
----------

Per CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py", pod
dispatchers post markers indirectly: they write a self-describing JSON
sentinel to ``/workspace/logs/issue-<N>-<kind_slug>-<epoch>.json`` and
the VM-side ``poll_pipeline.py`` reads + posts the marker from the
local checkout (which sits on ``main`` and therefore satisfies
``task.py``'s branch guard).

These tests pin the drain step's contract:

* a well-formed sentinel results in exactly one ``post_event`` call and
  a rename to ``<path>.processed`` (idempotent across ticks);
* a sentinel carrying a ``gate`` field surfaces it on the ``PollResult``
  so the orchestrator can park at a user gate;
* a malformed / unknown-schema sentinel is skipped + logged, NOT
  crashed-on, and is left un-renamed for inspection;
* draining is a no-op when there are no sentinels.

The SSH layer is monkey-patched at the ``subprocess.run`` boundary —
the heredoc shell snippets themselves are not under test (they're
straight glob+cat / mv shell builtins; the value is in the parsing +
post + rename round-trip).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_poll_pipeline():
    """Load ``scripts/poll_pipeline.py`` as a module without polluting
    ``sys.modules`` between test files (mirrors ``tests/test_pod_watch.py``).
    """
    spec = importlib.util.spec_from_file_location(
        "poll_pipeline_under_test", REPO_ROOT / "scripts" / "poll_pipeline.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["poll_pipeline_under_test"] = module
    spec.loader.exec_module(module)
    return module


pp = _load_poll_pipeline()


# ── Fixtures ────────────────────────────────────────────────────────────────


def _sentinel_body(
    *,
    kind: str = "epm:fact-candidates",
    version: int = 1,
    schema_version: int = 1,
    gate: str | None = "fact-candidates",
    note: str = "candidates ready for review",
    by: str = "experiment-implementer",
) -> dict[str, Any]:
    """Build a sentinel dict mirroring the #444 driver's writer schema."""
    return {
        "sentinel_schema_version": schema_version,
        "task_id": 444,
        "kind": kind,
        "version": version,
        "gate": gate,
        "blocks_pipeline": gate is not None,
        "note": note,
        "by": by,
        "ts": "2026-05-29T00:00:00+00:00",
    }


def _glob_response(*pairs: tuple[str, str]) -> str:
    """Build the stdout shape that ``_ssh_drain_sentinels`` expects from
    the remote heredoc. Each ``pair`` is ``(remote_path, body)``."""
    out: list[str] = []
    for path, body in pairs:
        out.append(f"SENTINEL_START {path}")
        out.append(body)
        out.append("SENTINEL_END")
    return "\n".join(out) + ("\n" if out else "")


class _SubprocessRouter:
    """Routes ``subprocess.run([ssh, ..., pod, cmd])`` calls by inspecting
    the cmd string. Tracks both the glob-cat call (returns canned stdout)
    and the mv call (records it so the test can assert on it)."""

    def __init__(
        self,
        *,
        glob_stdout: str = "",
        glob_returncode: int = 0,
        mv_returncode: int = 0,
    ) -> None:
        self.glob_stdout = glob_stdout
        self.glob_returncode = glob_returncode
        self.mv_returncode = mv_returncode
        self.mv_calls: list[str] = []  # the cmd string for each mv invocation
        self.glob_calls: int = 0

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        assert cmd[0] == "ssh", f"expected ssh, got {cmd!r}"
        # The remote command is always the last argv element in
        # ``["ssh", "-o", ..., pod, remote_cmd]``.
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            self.mv_calls.append(remote)
            return subprocess.CompletedProcess(
                args=cmd, returncode=self.mv_returncode, stdout="", stderr=""
            )
        # Everything else is treated as the glob-cat heredoc.
        self.glob_calls += 1
        return subprocess.CompletedProcess(
            args=cmd, returncode=self.glob_returncode, stdout=self.glob_stdout, stderr=""
        )


# ── _parse_sentinel ─────────────────────────────────────────────────────────


def test_parse_sentinel_happy_path() -> None:
    data = _sentinel_body()
    parsed = pp._parse_sentinel(
        "/workspace/logs/issue-444-epm_fact-candidates-1.json", json.dumps(data)
    )
    assert parsed is not None
    assert parsed["kind"] == "epm:fact-candidates"
    assert parsed["gate"] == "fact-candidates"


def test_parse_sentinel_rejects_unknown_schema_version() -> None:
    data = _sentinel_body(schema_version=99)
    parsed = pp._parse_sentinel("/workspace/logs/issue-444-x-1.json", json.dumps(data))
    assert parsed is None


def test_parse_sentinel_rejects_invalid_json() -> None:
    parsed = pp._parse_sentinel("/workspace/logs/issue-444-x-1.json", "{not json")
    assert parsed is None


def test_parse_sentinel_rejects_non_dict() -> None:
    parsed = pp._parse_sentinel("/workspace/logs/issue-444-x-1.json", '"a string"')
    assert parsed is None


def test_parse_sentinel_rejects_missing_required_keys() -> None:
    body = json.dumps({"sentinel_schema_version": 1, "kind": "epm:foo"})  # missing version
    parsed = pp._parse_sentinel("/workspace/logs/issue-444-x-1.json", body)
    assert parsed is None


def test_parse_sentinel_rejects_empty_body() -> None:
    assert pp._parse_sentinel("/workspace/logs/issue-444-x-1.json", "") is None


# ── _drain_sentinels (the full round-trip) ──────────────────────────────────


def test_drain_no_sentinels_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _SubprocessRouter(glob_stdout="")
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    assert processed == 0
    assert gate is None
    assert router.glob_calls == 1
    assert router.mv_calls == []
    post_mock.assert_not_called()


def test_drain_posts_marker_and_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_path = "/workspace/logs/issue-444-epm_progress-1700000000.json"
    body = _sentinel_body(kind="epm:progress", gate=None, note="phase=eval")
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    assert processed == 1
    assert gate is None  # no gate carried
    post_mock.assert_called_once_with(
        444, "epm:progress", version=1, by="experiment-implementer", note="phase=eval"
    )
    assert len(router.mv_calls) == 1
    assert sentinel_path in router.mv_calls[0]
    assert ".processed" in router.mv_calls[0]


def test_drain_surfaces_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_path = "/workspace/logs/issue-444-epm_fact-candidates-1700000001.json"
    body = _sentinel_body(kind="epm:fact-candidates", gate="fact-candidates")
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    assert processed == 1
    assert gate == "fact-candidates"


def test_drain_skips_malformed_does_not_post(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_path = "/workspace/logs/issue-444-broken-1700000002.json"
    good_path = "/workspace/logs/issue-444-progress-1700000003.json"
    good_body = _sentinel_body(kind="epm:progress", gate=None, note="phase=training")
    router = _SubprocessRouter(
        glob_stdout=_glob_response((bad_path, "{not json"), (good_path, json.dumps(good_body)))
    )
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    # Only the good sentinel posts + renames; the bad one is left in place.
    assert processed == 1
    assert gate is None
    post_mock.assert_called_once()
    assert len(router.mv_calls) == 1
    assert good_path in router.mv_calls[0]
    assert bad_path not in router.mv_calls[0]


def test_drain_leaves_unknown_schema_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_path = "/workspace/logs/issue-444-future-1700000004.json"
    body = _sentinel_body(schema_version=2)
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    assert processed == 0
    assert gate is None
    post_mock.assert_not_called()
    # Crucial: the future-schema sentinel must NOT be renamed, so a future
    # poller upgrade can re-process it.
    assert router.mv_calls == []


def test_drain_does_not_rename_when_post_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``post_event`` raises, we must NOT rename — next tick retries."""
    sentinel_path = "/workspace/logs/issue-444-epm_progress-1700000005.json"
    body = _sentinel_body(kind="epm:progress", gate=None)
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    monkeypatch.setattr(
        pp, "post_event", MagicMock(side_effect=RuntimeError("simulated post failure"))
    )

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    assert processed == 0
    assert gate is None
    assert router.mv_calls == []


def test_drain_is_idempotent_after_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a successful drain, a subsequent drain with no remaining
    sentinels (the writer's shell glob would skip ``.processed`` files) is
    a no-op. Simulated by returning empty glob output on tick 2."""
    sentinel_path = "/workspace/logs/issue-444-epm_progress-1700000006.json"
    body = _sentinel_body(kind="epm:progress", gate=None)

    # Tick 1: one sentinel.
    router1 = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router1)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)
    pp._drain_sentinels(issue=444, pod="epm-issue-444")
    assert post_mock.call_count == 1
    assert len(router1.mv_calls) == 1

    # Tick 2: glob returns nothing (the writer's ``case ... *.processed)
    # continue`` skipped the renamed file).
    router2 = _SubprocessRouter(glob_stdout="")
    monkeypatch.setattr(pp.subprocess, "run", router2)
    pp._drain_sentinels(issue=444, pod="epm-issue-444")
    assert post_mock.call_count == 1  # unchanged
    assert router2.mv_calls == []


def test_drain_falls_back_to_payload_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some writers may put the marker body under ``payload`` instead of
    ``note``; the poller treats them as synonyms."""
    sentinel_path = "/workspace/logs/issue-444-epm_progress-1700000007.json"
    body = _sentinel_body(kind="epm:progress", gate=None)
    body.pop("note")
    body["payload"] = "phase=eval"
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)

    pp._drain_sentinels(issue=444, pod="epm-issue-444")

    post_mock.assert_called_once()
    assert post_mock.call_args.kwargs["note"] == "phase=eval"


def test_drain_handles_ssh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _SubprocessRouter(glob_stdout="", glob_returncode=255)
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=444, pod="epm-issue-444")

    assert processed == 0
    assert gate is None
    post_mock.assert_not_called()
    assert router.mv_calls == []


# ── poll_once integration: gate preempts done ───────────────────────────────


def test_poll_once_gate_overrides_done(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A drained gate sentinel must yield status=gate even when the log
    tail already shows phase=done. Rationale: the orchestrator's polling
    loop reads ``status`` to decide whether to advance; a fact-candidates
    gate that fires alongside (or just before) done must NOT be lost."""
    sentinel_path = "/workspace/logs/issue-444-epm_fact-candidates-1700000008.json"
    body = _sentinel_body(kind="epm:fact-candidates", gate="fact-candidates")

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        # Disambiguate drain vs probe heredocs. The drain globs ``*.json``
        # while the probe's per-phase-log glob (#468) globs ``*.log``;
        # both share ``for f in /workspace/logs/issue-`` so that prefix
        # alone is no longer distinguishing. The drain heredoc emits the
        # ``SENTINEL_START`` literal, and the probe never does — use that
        # as the marker.
        if "SENTINEL_START" in remote:  # drain
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=_glob_response((sentinel_path, json.dumps(body))),
                stderr="",
            )
        # Otherwise, probe — return a phase=done tail.
        probe_stdout = _probe_response(
            pid_alive=1,
            mtime_epoch=1700000020,
            tail="2026-05-29 00:00:01 [phase=done]",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=probe_stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=444,
        pod="epm-issue-444",
        log_path="/workspace/logs/issue-444.log",
        pid_file="/workspace/logs/issue-444.pid",
        state_file=state_file,
    )

    assert result.status == "gate"
    assert result.gate == "fact-candidates"
    assert result.sentinels_processed == 1


# ── _ssh_probe + poll_once cell-log staleness (incident #405) ───────────────


def _probe_response(
    *,
    pid_alive: int = 1,
    pid_file_missing: int | None = None,
    marker_pid_alive: int | None = None,
    mtime_epoch: int = 0,
    tail: str = "",
    cell_mtime_epoch: int = 0,
    cell_tail: str = "",
    phase_log_mtime_epoch: int = 0,
    shard_log_mtime_epoch: int = 0,
    gpu_util: str = "unknown",
) -> str:
    """Build the stdout shape that ``_ssh_probe`` parses, including the
    cell-log fields added for the #405 smoke-first fix, the per-phase-log +
    GPU-util fields added for the #468 multi-phase fix, AND the
    repo-rooted shard-log field added for the #488 multi-GPU fan-out fix.

    Defaults preserve pre-#468/#488 behavior: zero / unknown values for
    the new fields mean "signal absent" -> they don't by themselves
    declare stalled; the verdict falls through to the older signals.
    """
    lines: list[str] = [f"PID_ALIVE={pid_alive}"]
    if pid_file_missing is not None:
        lines.append(f"PID_FILE_MISSING={pid_file_missing}")
    if marker_pid_alive is not None:
        lines.append(f"MARKER_PID_ALIVE={marker_pid_alive}")
    lines.append(f"MTIME_EPOCH={mtime_epoch}")
    lines.append("TAIL_START")
    if tail:
        lines.append(tail)
    lines.append("TAIL_END")
    lines.append(f"CELL_MTIME_EPOCH={cell_mtime_epoch}")
    lines.append("CELL_TAIL_START")
    if cell_tail:
        lines.append(cell_tail)
    lines.append("CELL_TAIL_END")
    lines.append(f"PHASE_LOG_MTIME_EPOCH={phase_log_mtime_epoch}")
    lines.append(f"SHARD_LOG_MTIME_EPOCH={shard_log_mtime_epoch}")
    lines.append(f"GPU_UTIL={gpu_util}")
    return "\n".join(lines) + "\n"


def test_ssh_probe_parses_cell_log_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ssh_probe`` must surface the CELL_MTIME_EPOCH + cell-log tail
    that the heredoc emits, so ``poll_once`` can fold them into staleness."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=1700000000,
                tail="2026-05-29 00:00:01 [phase=training]",
                cell_mtime_epoch=1700000900,
                cell_tail="2026-05-29 00:15:00 step 42/100 loss=1.2",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    probe = pp._ssh_probe(
        "epm-issue-405",
        "/workspace/logs/issue_405_sweep.log",
        "/workspace/logs/issue-405.pid",
        405,
    )
    assert probe["mtime_epoch"] == "1700000000"
    assert probe["cell_mtime_epoch"] == "1700000900"
    assert "[phase=training]" in probe["log_tail"]
    assert "step 42/100" in probe["cell_log_tail"]


def test_ssh_probe_handles_missing_cell_log_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cell logs on the pod (the common non-sweep case) must default
    cell_mtime_epoch to 0 and leave cell_log_tail empty — preserving
    pre-#405 behavior for single-cell runs."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=1700000000,
                tail="2026-05-29 [phase=eval]",
                cell_mtime_epoch=0,
                cell_tail="",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    probe = pp._ssh_probe(
        "epm-issue-405",
        "/workspace/logs/issue-405.log",
        "/workspace/logs/issue-405.pid",
        405,
    )
    assert probe["cell_mtime_epoch"] == "0"
    assert probe["cell_log_tail"] == ""
    # The new per-phase + GPU + shard fields default cleanly when no
    # PHASE_LOG / SHARD_LOG / GPU_UTIL lines are emitted (preserves
    # pre-#468 / pre-#488 behavior).
    assert probe["phase_log_mtime_epoch"] == "0"
    assert probe["shard_log_mtime_epoch"] == "0"
    assert probe["gpu_util"] == "unknown"


def test_poll_once_fresh_cell_log_keeps_status_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incident #405: during a sequential smoke cell the dispatcher is
    blocked in ``proc.wait()`` so the MAIN log goes silent for ~15-18 min.
    A fresh cell-log mtime must keep status=running and prevent false
    'stalled' / 'dead' verdicts.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    # Main log silent for 949s (the value seen in #405) -> would be stalled
    # under the old staleness rule. Cell log advanced 30s ago.
    main_mtime = now_epoch - 949
    cell_mtime = now_epoch - 30

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:  # drain (see test_poll_once_gate_overrides_done)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=main_mtime,
                tail="2026-05-29 00:00:01 [phase=training]",
                cell_mtime_epoch=cell_mtime,
                cell_tail="2026-05-29 step 200/2000 loss=0.8",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=405,
        pod="epm-issue-405",
        log_path="/workspace/logs/issue_405_sweep.log",
        pid_file="/workspace/logs/issue_405_sweep.pid",
        state_file=state_file,
    )

    assert result.status == "running", (
        f"expected status=running (cell log fresh) but got {result.status!r}; "
        f"last_log_mtime_sec_ago={result.last_log_mtime_sec_ago}"
    )
    assert result.last_log_mtime_sec_ago < pp.STALL_SEC, (
        "staleness should reflect the freshest source (cell log), not the main log"
    )
    # When the cell log is fresher, its tail must be the excerpt (operators
    # need to see what's actually running, not the silent main log).
    assert "step 200/2000" in result.log_tail_excerpt


def test_poll_once_dead_when_both_logs_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative: if BOTH the main log AND the cell log are stale past
    STALL_SEC AND the pid is dead, the verdict must still be 'dead'. The
    #405 fix should not paper over a genuinely-dead run.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    main_mtime = now_epoch - 2000
    cell_mtime = now_epoch - 1800  # also stale

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=0,  # pid is dead
                mtime_epoch=main_mtime,
                tail="2026-05-29 [phase=training]",  # never reached phase=done
                cell_mtime_epoch=cell_mtime,
                cell_tail="2026-05-29 step 50/2000",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=405,
        pod="epm-issue-405",
        log_path="/workspace/logs/issue_405_sweep.log",
        pid_file="/workspace/logs/issue_405_sweep.pid",
        state_file=state_file,
    )

    assert result.status == "dead"
    assert result.last_log_mtime_sec_ago >= pp.STALL_SEC


# ── _gpu_idle (incident #468) ───────────────────────────────────────────────


def test_gpu_idle_when_all_below_threshold() -> None:
    """All GPUs at or below the idle threshold (5%) -> idle."""
    assert pp._gpu_idle("0,0,0,0") is True
    assert pp._gpu_idle("5,3,1,0") is True
    assert pp._gpu_idle("5") is True


def test_gpu_idle_false_when_any_busy() -> None:
    """A single busy GPU keeps the verdict NOT idle — the stall conjunction
    requires every GPU to be idle, since one training process easily pins
    one card while the others sit free."""
    assert pp._gpu_idle("0,0,0,90") is False
    assert pp._gpu_idle("95,87,42,90") is False
    assert pp._gpu_idle("6") is False  # just over the threshold


def test_gpu_idle_fail_safe_on_unknown() -> None:
    """``nvidia-smi`` unavailable -> ``unknown``. Must return False so the
    stall verdict NEVER fires purely on a missing nvidia-smi — the
    per-phase + cell signals carry the verdict instead.
    """
    assert pp._gpu_idle("unknown") is False
    assert pp._gpu_idle("") is False


def test_gpu_idle_fail_safe_on_parse_error() -> None:
    """Garbled / partially-numeric output -> fail-safe to NOT idle."""
    assert pp._gpu_idle("not-an-int") is False
    assert pp._gpu_idle("0,0,foo") is False
    assert pp._gpu_idle(",,,") is False  # all-empty tokens -> empty list -> NOT idle


# ── poll_once: per-phase-log + GPU-idle staleness conjunction (#468) ────────


def test_poll_once_fresh_per_phase_log_keeps_status_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incident #468: a multi-phase launcher writes ``[phase=X]`` to the
    top-level log only at phase boundaries and redirects the long phase's
    stdout to ``/workspace/logs/issue-<N>-<phase>.log``. The top-level
    log + the cell-log dir BOTH go silent for the full phase duration
    (>>STALL_SEC) while the per-phase log is actively appended. The
    poll must stay in ``running`` purely because the per-phase log is
    fresh.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    # Main + cell logs quiet past STALL_SEC, but the per-phase log is fresh.
    main_mtime = now_epoch - 2000
    cell_mtime = 0  # no cell log
    phase_log_mtime = now_epoch - 30

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=main_mtime,
                tail="2026-06-02 14:00:00 [phase=variants-training]",
                cell_mtime_epoch=cell_mtime,
                phase_log_mtime_epoch=phase_log_mtime,
                # GPU idle: even though GPUs are idle, the fresh per-phase
                # log alone must keep the verdict in `running`. The stall
                # conjunction requires per-phase log ALSO to be quiet.
                gpu_util="0,0,0,0",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=468,
        pod="epm-issue-468",
        log_path="/workspace/logs/issue-468.log",
        pid_file="/workspace/logs/issue-468.pid",
        state_file=state_file,
    )

    assert result.status == "running", (
        f"expected status=running (per-phase log fresh) but got {result.status!r}; "
        f"phase_log_mtime_sec_ago={result.phase_log_mtime_sec_ago} "
        f"gpu_util={result.gpu_util!r}"
    )
    assert result.phase_log_mtime_sec_ago < pp.STALL_SEC


def test_poll_once_busy_gpu_keeps_status_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A busy GPU alone keeps a long-quiet pod in ``running`` — even when
    EVERY log signal (main + cell + per-phase) has been quiet past
    STALL_SEC. Rationale: a real workload pinning a GPU is the most
    robust 'still alive' signal; declaring stalled while the GPU sits
    >5% util would false-fail healthy runs whose stdout / phase log was
    e.g. buffered.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    main_mtime = now_epoch - 2000  # quiet
    cell_mtime = 0  # no cell log
    phase_log_mtime = now_epoch - 2000  # ALSO quiet

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=main_mtime,
                tail="2026-06-02 14:00:00 [phase=variants-training]",
                cell_mtime_epoch=cell_mtime,
                phase_log_mtime_epoch=phase_log_mtime,
                gpu_util="92,88,90,87",  # all busy
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=468,
        pod="epm-issue-468",
        log_path="/workspace/logs/issue-468.log",
        pid_file="/workspace/logs/issue-468.pid",
        state_file=state_file,
    )

    assert result.status == "running", (
        f"expected status=running (GPU busy) but got {result.status!r}; "
        f"gpu_util={result.gpu_util!r}"
    )
    assert result.gpu_util == "92,88,90,87"


def test_poll_once_all_quiet_and_idle_is_stalled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive: ALL FOUR signals agree on quiet/idle for >STALL_SEC AND
    the pid is alive (e.g. process spinning unproductively, or wedged on
    a system call) -> verdict is `stalled`. The fix must not over-correct
    into never declaring stalled.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    quiet = now_epoch - 2000

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,  # process still alive but not making progress
                mtime_epoch=quiet,
                tail="2026-06-02 [phase=training]",
                cell_mtime_epoch=quiet,
                cell_tail="2026-06-02 step 100/2000",
                phase_log_mtime_epoch=quiet,
                gpu_util="0,0,0,0",  # GPUs idle
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=468,
        pod="epm-issue-468",
        log_path="/workspace/logs/issue-468.log",
        pid_file="/workspace/logs/issue-468.pid",
        state_file=state_file,
    )

    assert result.status == "stalled"


def test_poll_once_nvidia_smi_unavailable_fail_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``nvidia-smi`` unavailable (``GPU_UTIL=unknown``) MUST NOT by
    itself declare a quiet run stalled. With main + cell + per-phase
    logs all quiet AND ``GPU_UTIL=unknown``, the GPU-idle gate fails
    safe to False -> verdict stays `running` (the per-phase / cell /
    main signals are now the only stall arbiters, and they cannot fire
    alone). This protects pods on which nvidia-smi is missing or
    transiently unreachable.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    quiet = now_epoch - 2000

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=quiet,
                tail="2026-06-02 [phase=training]",
                cell_mtime_epoch=quiet,
                phase_log_mtime_epoch=quiet,
                gpu_util="unknown",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=468,
        pod="epm-issue-468",
        log_path="/workspace/logs/issue-468.log",
        pid_file="/workspace/logs/issue-468.pid",
        state_file=state_file,
    )

    # Cannot declare stalled purely from log-quiet + nvidia-smi-unknown.
    assert result.status == "running", (
        f"expected status=running (nvidia-smi unknown is fail-safe to NOT idle); "
        f"got {result.status!r}; gpu_util={result.gpu_util!r}"
    )


def test_ssh_probe_parses_phase_log_and_gpu_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ssh_probe`` must surface ``PHASE_LOG_MTIME_EPOCH`` and ``GPU_UTIL``
    so ``poll_once`` can fold them into the stall conjunction.
    """

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=1700000000,
                tail="2026-06-02 [phase=training]",
                phase_log_mtime_epoch=1700000900,
                gpu_util="95,87,42,90",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    probe = pp._ssh_probe(
        "epm-issue-468",
        "/workspace/logs/issue-468.log",
        "/workspace/logs/issue-468.pid",
        468,
    )
    assert probe["phase_log_mtime_epoch"] == "1700000900"
    assert probe["gpu_util"] == "95,87,42,90"


# ── _drain_sentinels: oversize-note graceful degradation (#477) ─────────────
#
# Incident: a 52001-char ``epm:progress`` aggregate sentinel exceeded the
# 50000-char EVENT_NOTE_MAX cap. ``post_event`` raised ``ValueError`` every
# tick; the sentinel was never renamed; every subsequent poll re-posted +
# re-failed the same payload indefinitely. The fix degrades gracefully:
# persist the full note to a task artifact, post a truncated pointer marker
# that fits the cap, then rename the sentinel ``.processed`` so the loop
# stops. NON-oversize post failures keep the original retry semantics
# (already pinned by ``test_drain_does_not_rename_when_post_fails``).


def _oversize_value_error(orig_len: int = 52001) -> ValueError:
    """Mirror the message ``task_workflow.post_event`` raises on oversize.

    The poller matches the literal substring ``"event note exceeds"``; this
    factory keeps the test's failure-injection in lockstep with the real
    message format (kept narrow on purpose so generic ValueErrors still
    surface as honest failures).
    """
    return ValueError(
        f"event note exceeds {pp.EVENT_NOTE_MAX} chars ({orig_len}); "
        "caller must post epm:failure v1 with reason=note_oversize"
    )


def test_drain_oversize_note_persists_and_renames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Oversize ``note`` -> the full payload lands in
    ``<task>/artifacts/sentinel-note-*.txt``, a truncated pointer marker is
    posted (same kind/version, cites the artifact, ``oversize=True``), and
    the sentinel is renamed ``.processed`` so the loop ends. Reproduces
    the #477 cycle and verifies the graceful path.
    """
    # Stand up a fake task folder so ``find_task_path`` resolves to a
    # writable tmp_path location. (poll_pipeline imports ``find_task_path``
    # at the top of the module, so monkeypatching its attribute on
    # ``pp`` covers the resolve call.)
    task_dir = tmp_path / "tasks" / "running" / "477"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(pp, "find_task_path", lambda issue: task_dir)

    sentinel_path = "/workspace/logs/issue-477-epm_progress-1700001000.json"
    full_note = "x" * 52001  # > EVENT_NOTE_MAX (50000)
    body = _sentinel_body(kind="epm:progress", gate=None, note=full_note, by="dispatch_sweep")
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)

    # First call raises oversize; second call (the pointer marker) succeeds.
    post_mock = MagicMock(side_effect=[_oversize_value_error(52001), None])
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=477, pod="epm-issue-477")

    # 1. Sentinel is accounted as processed (so accounting is honest) and
    #    NOT carrying a gate (the original was gate=None).
    assert processed == 1
    assert gate is None

    # 2. Two post_event calls: the oversize attempt, then the pointer.
    assert post_mock.call_count == 2

    # 3. The pointer-marker post fits the cap, cites the artifact, and
    #    keeps the same (kind, version, by).
    pointer_call = post_mock.call_args_list[1]
    assert pointer_call.args == (477, "epm:progress")
    assert pointer_call.kwargs["version"] == 1
    assert pointer_call.kwargs["by"] == "dispatch_sweep"
    pointer_note = pointer_call.kwargs["note"]
    assert isinstance(pointer_note, str)
    assert len(pointer_note) <= pp.EVENT_NOTE_MAX, (
        f"pointer marker {len(pointer_note)} chars > cap {pp.EVENT_NOTE_MAX}"
    )
    assert "oversize" in pointer_note.lower()
    assert "52001" in pointer_note  # original length recorded inline
    artifacts = pointer_call.kwargs["artifacts"]
    assert isinstance(artifacts, list) and len(artifacts) == 1
    assert "sentinel-note-epm_progress-" in artifacts[0]
    assert pointer_call.kwargs.get("oversize") is True
    assert pointer_call.kwargs.get("oversize_orig_len") == 52001

    # 4. Full payload is persisted to disk (byte-identical to the original).
    persisted = list((task_dir / "artifacts").glob("sentinel-note-epm_progress-*.txt"))
    assert len(persisted) == 1
    assert persisted[0].read_text() == full_note

    # 5. Sentinel was renamed .processed — the loop terminates on this tick
    #    instead of cycling forever.
    assert len(router.mv_calls) == 1
    assert sentinel_path in router.mv_calls[0]
    assert ".processed" in router.mv_calls[0]


def test_drain_oversize_note_forwards_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When an oversize sentinel ALSO carries a ``gate``, the pointer marker
    must forward it AND ``_drain_sentinels`` must surface it on the return
    tuple so the orchestrator still parks at the user gate."""
    task_dir = tmp_path / "tasks" / "running" / "477"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(pp, "find_task_path", lambda issue: task_dir)

    sentinel_path = "/workspace/logs/issue-477-epm_fact-candidates-1700001001.json"
    full_note = "y" * 60000
    body = _sentinel_body(
        kind="epm:fact-candidates",
        gate="fact-candidates",
        note=full_note,
        by="experiment-implementer",
    )
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock(side_effect=[_oversize_value_error(60000), None])
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=477, pod="epm-issue-477")

    assert processed == 1
    assert gate == "fact-candidates", (
        "oversize handling must NOT drop the gate — orchestrator still parks"
    )
    pointer_call = post_mock.call_args_list[1]
    assert pointer_call.kwargs.get("gate") == "fact-candidates"
    assert pointer_call.kwargs.get("blocks_pipeline") is True
    assert len(router.mv_calls) == 1


def test_drain_oversize_persist_failure_leaves_sentinel_unrenamed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the graceful path itself fails (artifact write or pointer post),
    the sentinel must be left un-renamed so a future tick can retry — same
    contract as for any other transient post failure.
    """
    monkeypatch.setattr(
        pp,
        "find_task_path",
        MagicMock(side_effect=FileNotFoundError("task #477 not found")),
    )
    sentinel_path = "/workspace/logs/issue-477-epm_progress-1700001002.json"
    body = _sentinel_body(kind="epm:progress", gate=None, note="z" * 51000)
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock(side_effect=_oversize_value_error(51000))
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, gate = pp._drain_sentinels(issue=477, pod="epm-issue-477")

    assert processed == 0
    assert gate is None
    # Sentinel is NOT renamed -> next tick retries (recovers when task-path
    # resolution recovers, or operators intervene).
    assert router.mv_calls == []
    # Only the failing oversize post was attempted; no pointer-marker call
    # because path resolution short-circuited before we could write the
    # artifact.
    assert post_mock.call_count == 1


def test_drain_non_oversize_value_error_still_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A generic ``ValueError`` (schema bug, etc.) whose message does NOT
    contain the oversize-note signature must keep the original
    retry-on-next-tick semantics — never silently swallowed into the
    graceful path. Pairs with ``test_drain_does_not_rename_when_post_fails``
    (which uses ``RuntimeError``) to cover both exception classes.
    """
    sentinel_path = "/workspace/logs/issue-477-epm_progress-1700001003.json"
    body = _sentinel_body(kind="epm:progress", gate=None, note="ok")
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    monkeypatch.setattr(
        pp,
        "post_event",
        MagicMock(side_effect=ValueError("unrelated schema problem")),
    )
    # find_task_path must NOT be called when the ValueError doesn't match
    # the oversize signature; raise if it is, to pin the routing.
    monkeypatch.setattr(
        pp, "find_task_path", MagicMock(side_effect=AssertionError("must not be called"))
    )

    processed, gate = pp._drain_sentinels(issue=477, pod="epm-issue-477")

    assert processed == 0
    assert gate is None
    assert router.mv_calls == []  # sentinel left for retry


def test_oversize_pointer_note_fits_cap_even_for_huge_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pointer marker MUST itself fit under EVENT_NOTE_MAX, even when
    the original payload is several MB. Guards against an accounting bug
    where the leading-excerpt budget overshoots and re-trips the cap.
    """
    task_dir = tmp_path / "tasks" / "running" / "477"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(pp, "find_task_path", lambda issue: task_dir)

    huge_note = "Q" * 5_000_000  # 5 MB payload
    sentinel_path = "/workspace/logs/issue-477-epm_progress-1700001004.json"
    body = _sentinel_body(kind="epm:progress", gate=None, note=huge_note)
    router = _SubprocessRouter(glob_stdout=_glob_response((sentinel_path, json.dumps(body))))
    monkeypatch.setattr(pp.subprocess, "run", router)
    post_mock = MagicMock(side_effect=[_oversize_value_error(5_000_000), None])
    monkeypatch.setattr(pp, "post_event", post_mock)

    processed, _gate = pp._drain_sentinels(issue=477, pod="epm-issue-477")

    assert processed == 1
    pointer_call = post_mock.call_args_list[1]
    pointer_note = pointer_call.kwargs["note"]
    assert len(pointer_note) <= pp.EVENT_NOTE_MAX
    # Full payload still persisted in full on disk.
    persisted = list((task_dir / "artifacts").glob("sentinel-note-epm_progress-*.txt"))
    assert len(persisted) == 1
    assert persisted[0].stat().st_size == 5_000_000


# ── poll_once: shard-log staleness conjunction (incident #488) ──────────────
#
# i488 wrote per-GPU shard logs under
# ``/workspace/explore-persona-space/logs/issue_488/phase1_g{0..7}.log``
# (8 nested files, underscore separator), while the main log
# ``/workspace/logs/issue-488-run.log`` was only touched at phase
# transitions and the per-phase glob ``/workspace/logs/issue-488-*.log``
# didn't reach the nested subdirectory. With the inner Pass B loop
# writing each shard log every ~3 min, the main log + cell-log + per-phase
# log all went quiet past STALL_SEC and the poller falsely declared
# ``stalled`` on a healthy 8-GPU run. The fix adds a fifth liveness
# signal: max mtime across the two repo-rooted shard layouts.


def test_ssh_probe_parses_shard_log_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ssh_probe`` must surface ``SHARD_LOG_MTIME_EPOCH`` so
    ``poll_once`` can fold it into the stall conjunction.
    """

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=1700000000,
                tail="2026-06-07 [phase=phase1]",
                shard_log_mtime_epoch=1700000900,
                gpu_util="92,88,90,87,91,89,90,88",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    probe = pp._ssh_probe(
        "epm-issue-488",
        "/workspace/logs/issue-488-run.log",
        "/workspace/logs/issue-488-run.pid",
        488,
    )
    assert probe["shard_log_mtime_epoch"] == "1700000900"


def test_poll_once_fresh_shard_log_keeps_status_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incident #488: a multi-GPU launcher fans per-GPU shard logs into
    ``/workspace/explore-persona-space/logs/issue_<N>/phase*_g*.log``.
    The main + cell + per-phase logs all go quiet past STALL_SEC while
    the shard logs are actively appended every few minutes. The poll
    must stay in ``running`` purely because the shard log is fresh.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    # All older signals quiet past STALL_SEC; shard log fresh 30s ago.
    quiet = now_epoch - 2000
    shard_mtime = now_epoch - 30

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=quiet,
                tail="2026-06-07 14:00:00 [phase=phase1]",
                cell_mtime_epoch=0,  # no cell log
                phase_log_mtime_epoch=quiet,
                shard_log_mtime_epoch=shard_mtime,
                # Even if GPUs read as idle (e.g. transient nvidia-smi sample
                # between training steps), the fresh shard log alone must
                # keep the verdict in `running`.
                gpu_util="0,0,0,0,0,0,0,0",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=488,
        pod="epm-issue-488",
        log_path="/workspace/logs/issue-488-run.log",
        pid_file="/workspace/logs/issue-488-run.pid",
        state_file=state_file,
    )

    assert result.status == "running", (
        f"expected status=running (shard log fresh) but got {result.status!r}; "
        f"shard_log_mtime_sec_ago={result.shard_log_mtime_sec_ago} "
        f"phase_log_mtime_sec_ago={result.phase_log_mtime_sec_ago}"
    )
    assert result.shard_log_mtime_sec_ago < pp.STALL_SEC


def test_poll_once_stalled_requires_shard_log_also_quiet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control: ALL FIVE signals (main + cell + per-phase +
    shard + GPU-idle) quiet past STALL_SEC with pid alive -> verdict
    stays ``stalled``. The #488 fix must not over-correct into never
    declaring stalled.
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    quiet = now_epoch - 2000

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=quiet,
                tail="2026-06-07 [phase=phase1]",
                cell_mtime_epoch=quiet,
                phase_log_mtime_epoch=quiet,
                shard_log_mtime_epoch=quiet,  # ALSO quiet
                gpu_util="0,0,0,0",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=488,
        pod="epm-issue-488",
        log_path="/workspace/logs/issue-488-run.log",
        pid_file="/workspace/logs/issue-488-run.pid",
        state_file=state_file,
    )

    assert result.status == "stalled"
    # The shard signal still gets surfaced so operators can see WHY
    # every signal agreed on quiet.
    assert result.shard_log_mtime_sec_ago >= pp.STALL_SEC


# ── dispatcher per-job log liveness (incident #521) ──────────────────────────
#
# The issue_519/521-style dispatcher writes one log per job under
# ``<output_dir>/logs/*.log``, with ``output_dir`` typically
# ``/workspace/explore-persona-space/eval_results/issue_<N>``. During a
# CPU-bound judge-batch wait (Anthropic message-batch polling) the GPUs
# are idle BY DESIGN and the main log is quiet, while the per-job log
# appends every 30-60s — the only liveness signal. On 2026-06-10 a #521
# tick declared the healthy EM-steering job ``stalled`` (pid_alive=True,
# GPUs all 0, main log 1302s stale) because no probe globbed the per-job
# dir. The fix widens the shard-log probe to ALSO glob
# ``eval_results/issue_<N>{,_*}/logs/*.log`` into the same
# ``SHARD_LOG_MTIME_EPOCH`` max; status routing is unchanged.


def test_ssh_probe_heredoc_globs_dispatcher_perjob_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe heredoc must reach the dispatcher per-job log dir
    (#521). The shell snippets are otherwise not under test, so pin the
    glob patterns textually — dropping either one regresses a healthy
    judge-batch wait back to false-``stalled``. The patterns keep the
    issue-number match exact (``issue_521`` / ``issue_521_*``), never a
    bare ``issue_521*`` (which would let issue 5 match issue 521)."""
    captured: dict[str, str] = {}

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured["heredoc"] = cmd[-1]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=_probe_response(), stderr=""
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    pp._ssh_probe(
        "epm-issue-521",
        "/workspace/logs/issue-521.log",
        "/workspace/logs/issue-521.pid",
        521,
    )
    heredoc = captured["heredoc"]
    assert "/workspace/explore-persona-space/eval_results/issue_521/logs/*.log" in heredoc
    assert "/workspace/explore-persona-space/eval_results/issue_521_*/logs/*.log" in heredoc
    # Narrowness guard: no bare `issue_521*` directory glob.
    assert "eval_results/issue_521*/logs" not in heredoc


def test_poll_once_fresh_dispatcher_perjob_log_keeps_status_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incident #521 (2026-06-10 tick 21): pid alive, GPUs all idle (the
    job is polling an external judge batch — CPU-bound by design), main
    log 1302s stale, per-phase logs quiet — but the dispatcher per-job
    log under ``eval_results/issue_521/logs/`` is fresh (45s). Its mtime
    flows through ``SHARD_LOG_MTIME_EPOCH``, and that alone must keep
    the verdict in ``running``."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    quiet = now_epoch - 1302  # the stale main-log age observed in #521
    perjob_mtime = now_epoch - 45

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=quiet,
                tail="2026-06-10 03:10:00 [phase=phase_e]",
                cell_mtime_epoch=0,  # no cell log
                phase_log_mtime_epoch=quiet,
                shard_log_mtime_epoch=perjob_mtime,
                gpu_util="0,0,0,0",  # idle BY DESIGN during the batch wait
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=521,
        pod="epm-issue-521",
        log_path="/workspace/logs/issue-521.log",
        pid_file="/workspace/logs/issue-521.pid",
        state_file=state_file,
    )

    assert result.status == "running", (
        f"expected status=running (per-job log fresh) but got {result.status!r}; "
        f"shard_log_mtime_sec_ago={result.shard_log_mtime_sec_ago} "
        f"last_log_mtime_sec_ago={result.last_log_mtime_sec_ago}"
    )
    assert result.shard_log_mtime_sec_ago < pp.STALL_SEC


# ── #488 stale-port auto-heal: SSH-failure counter -> refresh-from-api ────────


def _ssh_failure_response() -> subprocess.CompletedProcess:
    """Build a subprocess result that mimics SSH transport failure (rc != 0).
    This is the shape ``_ssh_probe`` degrades on; ``poll_once`` then counts
    these against ``SSH_FAIL_REFRESH_THRESHOLD`` and fires the
    ``pod.py config --refresh-from-api <pod>`` auto-heal once the counter
    crosses the threshold."""
    return subprocess.CompletedProcess(
        args=["ssh", "stub"],
        returncode=255,
        stdout="",
        stderr="ssh: connect to host x port y: Connection refused\n",
    )


def test_poll_once_ssh_fail_increments_counter_in_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One SSH-failed probe -> ssh_fail_count=1 persisted, no refresh attempt
    yet (below threshold). The counter is the persistent backbone of the
    #488 stale-port auto-heal."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        # Every ssh probe (drain + probe heredoc) fails. The auto-heal
        # subprocess (uv run python scripts/pod.py ...) should NOT fire yet —
        # the counter is below SSH_FAIL_REFRESH_THRESHOLD.
        assert cmd[0] == "ssh"  # explicitly: no refresh call this tick
        return _ssh_failure_response()

    refresh_calls: list[str] = []

    def _fake_refresh(pod: str) -> bool:
        refresh_calls.append(pod)
        return True

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "_try_refresh_pods_conf_from_api", _fake_refresh)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    pp.poll_once(
        issue=488,
        pod="pod-488",
        log_path="/workspace/logs/issue-488.log",
        pid_file="/workspace/logs/issue-488.pid",
        state_file=state_file,
    )

    payload = json.loads(state_file.read_text())
    assert payload["488"]["ssh_fail_count"] == "1"
    assert refresh_calls == []  # below threshold


def test_poll_once_ssh_fail_fires_refresh_at_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the persisted ssh_fail_count is at threshold-1, the next failure
    pushes it to threshold and triggers exactly one refresh-from-api call.
    The counter then resets to 0 so we don't hot-loop refresh on each
    subsequent failing tick."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return _ssh_failure_response()

    refresh_calls: list[str] = []

    def _fake_refresh(pod: str) -> bool:
        refresh_calls.append(pod)
        return True

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "_try_refresh_pods_conf_from_api", _fake_refresh)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    # Pre-seed the counter at threshold-1 so the next tick crosses.
    state_file = tmp_path / "poll-state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    threshold = pp.SSH_FAIL_REFRESH_THRESHOLD
    state_file.write_text(
        json.dumps(
            {"488": {"phase": "", "last_mtime_epoch": "0", "ssh_fail_count": str(threshold - 1)}}
        )
    )

    pp.poll_once(
        issue=488,
        pod="pod-488",
        log_path="/workspace/logs/issue-488.log",
        pid_file="/workspace/logs/issue-488.pid",
        state_file=state_file,
    )

    assert refresh_calls == ["pod-488"]
    payload = json.loads(state_file.read_text())
    # Counter reset after the refresh attempt — next N consecutive failures
    # will trip another retry.
    assert payload["488"]["ssh_fail_count"] == "0"


def test_poll_once_ssh_fail_counter_resets_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful SSH probe (any rc=0) zeroes the failure counter so a
    transient outage that recovers never accumulates toward the refresh
    threshold."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        # Healthy probe.
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=int(datetime.now(tz=UTC).timestamp()),
                tail="2026-06-09 [phase=training]",
            ),
            stderr="",
        )

    refresh_calls: list[str] = []

    def _fake_refresh(pod: str) -> bool:
        refresh_calls.append(pod)
        return True

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "_try_refresh_pods_conf_from_api", _fake_refresh)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    # Pre-seed an accumulated counter; the healthy tick must clear it.
    state_file = tmp_path / "poll-state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"488": {"phase": "", "ssh_fail_count": "7"}}))

    pp.poll_once(
        issue=488,
        pod="pod-488",
        log_path="/workspace/logs/issue-488.log",
        pid_file="/workspace/logs/issue-488.pid",
        state_file=state_file,
    )

    assert refresh_calls == []  # not fired
    payload = json.loads(state_file.read_text())
    assert payload["488"]["ssh_fail_count"] == "0"


def test_try_refresh_pods_conf_from_api_fail_soft_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_try_refresh_pods_conf_from_api`` returns False (does NOT raise) on
    a non-zero exit from ``pod.py config --refresh-from-api``. The polling
    loop must never crash on the auto-heal — incident #488 was already
    expensive without compounding it by killing the watcher."""
    monkeypatch.setattr(
        pp.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=a[0] if a else [], returncode=2, stdout="", stderr="ERROR: pod not found"
        ),
    )
    assert pp._try_refresh_pods_conf_from_api("pod-488") is False


def test_try_refresh_pods_conf_from_api_fail_soft_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subprocess timeout / OSError on the refresh call also returns False
    instead of propagating. Same fail-soft contract."""

    def _boom(*a: Any, **kw: Any) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd=a[0] if a else "pod.py", timeout=60)

    monkeypatch.setattr(pp.subprocess, "run", _boom)
    assert pp._try_refresh_pods_conf_from_api("pod-488") is False


def test_poll_once_absent_shard_log_does_not_block_stall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No shard log on the pod (the common single-process case): the
    shard signal defaults to "0 -> very old" (10**9 sec_ago) so it
    NEVER by itself blocks a stall verdict — the older main+cell+phase+
    GPU conjunction still fires when all those agree. Guards against
    a regression where ``shard_log_mtime_epoch=0`` was treated as
    "fresh" instead of "absent".
    """
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    quiet = now_epoch - 2000

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=quiet,
                tail="2026-06-07 [phase=training]",
                cell_mtime_epoch=quiet,
                phase_log_mtime_epoch=quiet,
                shard_log_mtime_epoch=0,  # absent
                gpu_util="0,0,0,0",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=488,
        pod="epm-issue-488",
        log_path="/workspace/logs/issue-488-run.log",
        pid_file="/workspace/logs/issue-488-run.pid",
        state_file=state_file,
    )

    assert result.status == "stalled"
    # When no shard log exists, the sec_ago is the "very old" sentinel.
    assert result.shard_log_mtime_sec_ago >= pp.STALL_SEC


# ── pid_file_missing observability (incident #521) ──────────────────────────


def test_ssh_probe_parses_pid_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ssh_probe`` must surface PID_FILE_MISSING=1 so ``poll_once`` can
    distinguish "pid file absent on pod" from "pid probed dead" (#521)."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=0,
                pid_file_missing=1,
                mtime_epoch=1700000000,
                tail="2026-06-10 [phase=training]",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    probe = pp._ssh_probe(
        "pod-521",
        "/workspace/logs/issue-521.log",
        "/workspace/logs/issue-521.pid",
        521,
    )
    assert probe["pid_file_missing"] == "1"
    assert probe["pid_alive"] == "0"


def test_ssh_probe_pid_file_missing_fail_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pid_file_missing`` defaults to "0" both when the probe stdout omits
    the line (older heredoc shape) AND on the SSH-failure fail-safe path —
    transport failure means "unknown", not "missing"."""

    def _fake_run_no_line(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(pid_alive=1, mtime_epoch=1700000000),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run_no_line)
    probe = pp._ssh_probe(
        "pod-521",
        "/workspace/logs/issue-521.log",
        "/workspace/logs/issue-521.pid",
        521,
    )
    assert probe["pid_file_missing"] == "0"

    def _fake_run_ssh_down(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=cmd, returncode=255, stdout="", stderr="boom")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run_ssh_down)
    probe = pp._ssh_probe(
        "pod-521",
        "/workspace/logs/issue-521.log",
        "/workspace/logs/issue-521.pid",
        521,
    )
    assert probe["pid_file_missing"] == "0"
    assert probe["ssh_failed"] == "1"


def test_poll_once_pid_file_missing_marker_fallback_warns_and_stays_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#521 observability: pid file absent + live marker pid -> the run
    stays ``running`` (unchanged routing — pid_alive ORs in the marker
    pid), the tick surfaces ``pid_file_missing=True``, and a WARN names
    the marker-pid fallback so the orchestrator sees it in the tick
    output instead of a bare ``pid_alive`` flip."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:  # drain
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=0,  # no pidfile -> heredoc emits PID_ALIVE=0
                pid_file_missing=1,
                marker_pid_alive=1,  # the live re-launch pid from the marker
                mtime_epoch=now_epoch - 30,
                tail="2026-06-10 [phase=training]",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: 4242)

    state_file = tmp_path / "poll-state.json"
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = pp.poll_once(
            issue=521,
            pod="pod-521",
            log_path="/workspace/logs/issue-521.log",
            pid_file="/workspace/logs/issue-521.pid",
            state_file=state_file,
        )

    assert result.status == "running"
    assert result.pid_alive is True
    assert result.pid_file_missing is True
    assert any(
        "pid file" in rec.message and "marker pid" in rec.message and "4242" in rec.getMessage()
        for rec in caplog.records
    ), f"expected a marker-pid-fallback WARN; got: {[r.getMessage() for r in caplog.records]}"


def test_poll_once_pid_file_missing_does_not_change_dead_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative: pid file absent + NO marker pid + non-done tail must still
    route to ``dead`` exactly as before — the new field is observability
    only. No marker-fallback WARN fires when there is no marker pid."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:  # drain
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=0,
                pid_file_missing=1,
                mtime_epoch=now_epoch - 30,
                tail="2026-06-10 [phase=training]",  # never reached done
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)

    state_file = tmp_path / "poll-state.json"
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = pp.poll_once(
            issue=521,
            pod="pod-521",
            log_path="/workspace/logs/issue-521.log",
            pid_file="/workspace/logs/issue-521.pid",
            state_file=state_file,
        )

    assert result.status == "dead"
    assert result.pid_file_missing is True
    assert not any("marker pid" in rec.getMessage() for rec in caplog.records)


# ── PHASE_RE / _latest_phase: digit-bearing phase names (#537) ──────────────


def test_latest_phase_parses_digit_bearing_phase_names() -> None:
    """Regression (#537): ``PHASE_RE`` must include digits in the milestone
    token. The old ``[a-z_]+`` pattern truncated ``[phase=p0_render]`` to
    ``"p"``, making the poller's ``current_phase`` illegible for any
    dispatcher using numbered phase names (p0/p1/p2)."""
    tail = "\n".join(
        [
            "2026-06-09 14:00:00 [phase=p0_render]",
            "2026-06-09 14:05:00 [phase=p1_freeze step=10/20]",
        ]
    )
    assert pp._latest_phase(tail) == "p1_freeze"
    assert pp._latest_phase("2026-06-09 [phase=p0_render]") == "p0_render"


def test_latest_phase_done_detection_not_loosened_by_digit_widening() -> None:
    """``[phase=done]`` still parses as exactly ``done``, and a digit-suffixed
    token like ``done2`` no longer truncates to a false ``done`` (the old
    pattern stopped at the digit, so ``[phase=done2]`` read as ``done``)."""
    assert pp._latest_phase("2026-06-09 [phase=done]") == "done"
    assert pp._latest_phase("2026-06-09 [phase=done2]") == "done2"


# ── GPU-idle advisory (incidents #518 + #537) ───────────────────────────────
#
# Incident 2026-06-10: #518 ran a single-core CPU scoring phase ~14h on an
# idle 8xH100 pod and #537 polled an external judge batch 2.5h+, both with
# every GPU at 0% — and both were (correctly) classified `running`, so they
# burned silently. The fix tracks the sustained healthy-and-all-idle span
# across ticks and posts a one-per-phase, NON-BLOCKING [gpu-idle-advisory]
# epm:progress marker after GPU_IDLE_ADVISORY_MIN minutes. The advisory must
# never flip the status verdict, never count an `unknown` GPU sample toward
# the span, and de-dup on phase name.


def test_gpu_idle_advisory_update_starts_span_no_early_post() -> None:
    """First healthy all-idle tick opens the span at ``now`` and does NOT
    post — the window has length zero."""
    now = 1_700_000_000
    up = pp._gpu_idle_advisory_update(
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="scoring",
        prev_phase="scoring",
        prev_idle_since_epoch=0,
        advised_phases=set(),
        now_epoch=now,
        advisory_min=30,
    )
    assert up.should_post is False
    assert up.idle_since_epoch == now
    assert up.idle_span_sec == 0


def test_gpu_idle_advisory_update_posts_after_window() -> None:
    """A span carried in state that exceeds the window -> should_post."""
    now = 1_700_000_000
    up = pp._gpu_idle_advisory_update(
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="scoring",
        prev_phase="scoring",
        prev_idle_since_epoch=now - 1800,  # 30 min ago
        advised_phases=set(),
        now_epoch=now,
        advisory_min=30,
    )
    assert up.should_post is True
    assert up.idle_since_epoch == now - 1800
    assert up.idle_span_sec == 1800


def test_gpu_idle_advisory_update_dedups_on_phase() -> None:
    """A phase already advised never posts again, even as the span grows."""
    now = 1_700_000_000
    up = pp._gpu_idle_advisory_update(
        status="running",
        gpu_util="0,0",
        current_phase="scoring",
        prev_phase="scoring",
        prev_idle_since_epoch=now - 7200,
        advised_phases={"scoring"},
        now_epoch=now,
        advisory_min=30,
    )
    assert up.should_post is False
    # The span itself stays tracked (a later phase gets a fresh window).
    assert up.idle_since_epoch == now - 7200


def test_gpu_idle_advisory_update_resets_on_busy_unknown_or_garbage() -> None:
    """Any busy GPU, an ``unknown`` sample, or unparsable output resets the
    span — the ``_gpu_idle`` fail-safe carries over (a missing nvidia-smi
    must never accumulate toward an advisory)."""
    now = 1_700_000_000
    for util in ("0,0,0,90", "unknown", "", "not-an-int", ",,,"):
        up = pp._gpu_idle_advisory_update(
            status="running",
            gpu_util=util,
            current_phase="scoring",
            prev_phase="scoring",
            prev_idle_since_epoch=now - 7200,
            advised_phases=set(),
            now_epoch=now,
            advisory_min=30,
        )
        assert up.should_post is False, f"util={util!r}"
        assert up.idle_since_epoch == 0, f"util={util!r}"


def test_gpu_idle_advisory_update_resets_on_unhealthy_status() -> None:
    """Only a healthy (``running``) verdict accumulates: stalled / dead /
    done / gate ticks reset the span — those states have their own
    handling and the advisory is strictly for healthy CPU-bound burns."""
    now = 1_700_000_000
    for status in ("stalled", "dead", "done", "gate"):
        up = pp._gpu_idle_advisory_update(
            status=status,
            gpu_util="0,0,0,0",
            current_phase="scoring",
            prev_phase="scoring",
            prev_idle_since_epoch=now - 7200,
            advised_phases=set(),
            now_epoch=now,
            advisory_min=30,
        )
        assert up.should_post is False, f"status={status!r}"
        assert up.idle_since_epoch == 0, f"status={status!r}"


def test_gpu_idle_advisory_update_resets_on_phase_change() -> None:
    """A phase boundary restarts the span at the current tick — each phase
    is judged on its own idle window, so a long-idle phase A never makes
    the first tick of phase B advisory-eligible."""
    now = 1_700_000_000
    up = pp._gpu_idle_advisory_update(
        status="running",
        gpu_util="0,0,0,0",
        current_phase="plotting",
        prev_phase="scoring",
        prev_idle_since_epoch=now - 7200,
        advised_phases=set(),
        now_epoch=now,
        advisory_min=30,
    )
    assert up.should_post is False
    assert up.idle_since_epoch == now  # fresh span for the new phase


def test_gpu_idle_advisory_update_disabled_when_zero() -> None:
    """``advisory_min <= 0`` disables the advisory entirely (env opt-out
    EPM_GPU_IDLE_ADVISORY_MIN=0)."""
    now = 1_700_000_000
    up = pp._gpu_idle_advisory_update(
        status="running",
        gpu_util="0,0,0,0",
        current_phase="scoring",
        prev_phase="scoring",
        prev_idle_since_epoch=now - 7200,
        advised_phases=set(),
        now_epoch=now,
        advisory_min=0,
    )
    assert up.should_post is False
    assert up.idle_since_epoch == 0


def _seed_advisory_state(
    state_file: Path,
    *,
    issue: int,
    phase: str,
    idle_since_epoch: int,
    advised_phases: str = "",
) -> None:
    """Write a poll-state file as a prior tick would have left it."""
    state_file.write_text(
        json.dumps(
            {
                str(issue): {
                    "phase": phase,
                    "last_mtime_epoch": str(idle_since_epoch),
                    "ssh_fail_count": "0",
                    "gpu_idle_since_epoch": str(idle_since_epoch),
                    "gpu_idle_advised_phases": advised_phases,
                }
            }
        )
    )


def _healthy_idle_fake_run(now_epoch: int, phase: str = "scoring"):
    """A probe router for a HEALTHY run on a CPU-only phase: fresh main
    log (so the verdict is ``running``), pid alive, every GPU at 0%."""

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:  # drain
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=now_epoch - 30,  # fresh log -> healthy
                tail=f"2026-06-10 17:00:00 [phase={phase}]",
                gpu_util="0,0,0,0,0,0,0,0",
            ),
            stderr="",
        )

    return _fake_run


def test_poll_once_posts_gpu_idle_advisory_after_sustained_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incident #518 end-to-end: a healthy run (fresh log, pid alive) with
    all 8 GPUs at 0% for over the advisory window posts ONE
    [gpu-idle-advisory] epm:progress marker, keeps status=running, and
    persists the per-phase de-dup in the state file."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    state_file = tmp_path / "poll-state.json"
    _seed_advisory_state(state_file, issue=518, phase="scoring", idle_since_epoch=now_epoch - 3600)

    monkeypatch.setattr(pp.subprocess, "run", _healthy_idle_fake_run(now_epoch))
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)

    result = pp.poll_once(
        issue=518,
        pod="pod-518",
        log_path="/workspace/logs/issue-518.log",
        pid_file="/workspace/logs/issue-518.pid",
        state_file=state_file,
    )

    # The advisory NEVER flips the verdict.
    assert result.status == "running"
    assert result.gpu_idle_advisory_posted is True

    advisory_calls = [
        c for c in post_mock.call_args_list if "[gpu-idle-advisory]" in (c.kwargs.get("note") or "")
    ]
    assert len(advisory_calls) == 1
    call = advisory_calls[0]
    # Rides the existing epm:progress channel — no new marker schema.
    assert call.args == (518, "epm:progress")
    assert call.kwargs["by"] == "poll_pipeline"
    assert call.kwargs.get("gpu_idle_advisory") is True
    assert call.kwargs.get("phase") == "scoring"
    note = call.kwargs["note"]
    assert "all 8 GPUs" in note
    assert "CPU-only phase" in note

    # De-dup persisted for the next tick.
    saved = json.loads(state_file.read_text())["518"]
    assert "scoring" in saved["gpu_idle_advised_phases"]


def test_poll_once_gpu_idle_advisory_at_most_once_per_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tick 2 on the same still-idle phase must NOT post a second advisory
    (de-dup on phase name) — it never spams."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    state_file = tmp_path / "poll-state.json"
    _seed_advisory_state(
        state_file,
        issue=518,
        phase="scoring",
        idle_since_epoch=now_epoch - 7200,
        advised_phases="scoring",  # tick 1 already advised this phase
    )

    monkeypatch.setattr(pp.subprocess, "run", _healthy_idle_fake_run(now_epoch))
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)

    result = pp.poll_once(
        issue=518,
        pod="pod-518",
        log_path="/workspace/logs/issue-518.log",
        pid_file="/workspace/logs/issue-518.pid",
        state_file=state_file,
    )

    assert result.status == "running"
    assert result.gpu_idle_advisory_posted is False
    advisory_calls = [
        c for c in post_mock.call_args_list if "[gpu-idle-advisory]" in (c.kwargs.get("note") or "")
    ]
    assert advisory_calls == []
    # The advised set survives the tick (still deduped next time).
    saved = json.loads(state_file.read_text())["518"]
    assert "scoring" in saved["gpu_idle_advised_phases"]


def test_poll_once_gpu_idle_advisory_post_failure_retries_next_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the advisory post itself fails, the phase is NOT recorded as
    advised (next tick retries) and the poll still completes healthily —
    an advisory failure must never take down the tick."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    state_file = tmp_path / "poll-state.json"
    _seed_advisory_state(state_file, issue=518, phase="scoring", idle_since_epoch=now_epoch - 3600)

    monkeypatch.setattr(pp.subprocess, "run", _healthy_idle_fake_run(now_epoch))
    monkeypatch.setattr(
        pp, "post_event", MagicMock(side_effect=RuntimeError("simulated post failure"))
    )
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)

    result = pp.poll_once(
        issue=518,
        pod="pod-518",
        log_path="/workspace/logs/issue-518.log",
        pid_file="/workspace/logs/issue-518.pid",
        state_file=state_file,
    )

    assert result.status == "running"
    assert result.gpu_idle_advisory_posted is False
    saved = json.loads(state_file.read_text())["518"]
    assert "scoring" not in saved["gpu_idle_advised_phases"]
    # The span survives so the retry fires immediately next tick.
    assert saved["gpu_idle_since_epoch"] == str(now_epoch - 3600)


def test_poll_once_no_advisory_when_gpu_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``GPU_UTIL=unknown`` resets the span instead of counting as idle —
    the ``_gpu_idle`` fail-safe carries over to the advisory path."""
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    state_file = tmp_path / "poll-state.json"
    _seed_advisory_state(state_file, issue=518, phase="scoring", idle_since_epoch=now_epoch - 7200)

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remote = cmd[-1]
        if remote.startswith("mv -n "):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_probe_response(
                pid_alive=1,
                mtime_epoch=now_epoch - 30,
                tail="2026-06-10 17:00:00 [phase=scoring]",
                gpu_util="unknown",
            ),
            stderr="",
        )

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    post_mock = MagicMock()
    monkeypatch.setattr(pp, "post_event", post_mock)
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)

    result = pp.poll_once(
        issue=518,
        pod="pod-518",
        log_path="/workspace/logs/issue-518.log",
        pid_file="/workspace/logs/issue-518.pid",
        state_file=state_file,
    )

    assert result.status == "running"
    assert result.gpu_idle_advisory_posted is False
    advisory_calls = [
        c for c in post_mock.call_args_list if "[gpu-idle-advisory]" in (c.kwargs.get("note") or "")
    ]
    assert advisory_calls == []
    saved = json.loads(state_file.read_text())["518"]
    assert saved["gpu_idle_since_epoch"] == "0"  # span reset, not accumulated
