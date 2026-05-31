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
import subprocess
import sys
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
        # Distinguish the drain heredoc from the probe heredoc by checking
        # for keywords characteristic of each.
        if "nullglob" in remote:  # drain
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=_glob_response((sentinel_path, json.dumps(body))),
                stderr="",
            )
        # Otherwise, probe — return a phase=done tail.
        probe_stdout = (
            "PID_ALIVE=1\n"
            "MTIME_EPOCH=1700000020\n"
            "TAIL_START\n"
            "2026-05-29 00:00:01 [phase=done]\n"
            "TAIL_END\n"
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
