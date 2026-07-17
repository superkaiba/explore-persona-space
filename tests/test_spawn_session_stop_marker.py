"""Unit tests for the `spawn_session.py stop` deliberate-stop breadcrumb (#902).

What this pins (plan #902 §4.6; incident #779 r9 — three deliberate PM
SIGKILLs with no machine-readable record were mis-diagnosed as kernel OOM):

1. An issue-mapped OPERATOR stop posts a ``deliberate-stop`` breadcrumb
   (structured ``epm:progress`` note) BEFORE the stop RPC — the kill-source
   checklist step 4 (`.claude/skills/issue/failure_patterns.md`) greps for it.
2. Unmapped sessions post nothing (behavior byte-equivalent to pre-#902).
3. Fail-soft: a raising ``post_event`` WARNs on stderr and the stop proceeds.
4. ``--reason`` lands in the note.
5. Watcher-sourced stops (``--stop-source watcher``) post NOTHING — the
   watcher keeps its own registry/sidecar evidence trail; an auto-post here
   would manufacture false operator attributions and unsentineled notes that
   reset the watcher's staleness clocks.
6. An RPC failure (SystemExit) still had the breadcrumb attempted FIRST.
7. The post is TIME-BOUNDED: a hanging ``post_event`` (wedged workflow flock
   — a blocking ``fcntl.flock`` with no timeout) cannot hang the stop past
   the join bound; the stop proceeds with a loud WARN.

No daemon, no real task state: ``_load_session_issue_map``, ``post_event``,
and the daemon ``post`` RPC are all monkeypatched (the injectable style of
the sibling ``test_spawn_session_list_enrichment.py``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402

import explore_persona_space.task_workflow as task_workflow  # noqa: E402  (isort: after path bootstrap)

SID = "sess-902-test"
ISSUE = 123


def _patch_common(
    monkeypatch,
    calls: list,
    *,
    issue_map: dict[str, int],
    post_event_fn=None,
    rpc_resp: dict | None = None,
) -> None:
    """Wire the three seams `cmd_stop` touches into a shared call log.

    ``calls`` receives ``("marker", task_id, kind, kwargs)`` rows from the
    (mocked) ``task_workflow.post_event`` and ``("rpc", path, body)`` rows
    from the (mocked) daemon ``post`` — list order IS the observed ordering.
    """
    monkeypatch.setattr(spawn_session, "_load_session_issue_map", lambda: issue_map)
    # #1455 hermeticity seam: the operator-stop cleanup would otherwise hit
    # the real ~/.eps-autonomous registry + the daemon live list.
    monkeypatch.setattr(spawn_session, "_post_stop_cleanup", lambda sid, **_k: None)

    def default_post_event(task_id, kind, **kwargs):
        calls.append(("marker", task_id, kind, kwargs))
        return {}

    # `cmd_stop`'s daemon thread does `from explore_persona_space.task_workflow
    # import post_event` at CALL time, so patching the module attribute is
    # sufficient — no re-import escapes the patch.
    monkeypatch.setattr(task_workflow, "post_event", post_event_fn or default_post_event)

    def fake_post(path, body):
        calls.append(("rpc", path, body))
        return rpc_resp if rpc_resp is not None else {"success": True}

    monkeypatch.setattr(spawn_session, "post", fake_post)


def test_stop_posts_breadcrumb_before_rpc(monkeypatch):
    calls: list = []
    _patch_common(monkeypatch, calls, issue_map={SID: ISSUE})
    spawn_session.main(["stop", "--session-id", SID])
    assert [c[0] for c in calls] == ["marker", "rpc"], calls
    _, task_id, kind, kwargs = calls[0]
    assert task_id == ISSUE
    assert kind == "epm:progress"
    assert kwargs["by"] == "spawn_session-stop"
    note = kwargs["note"]
    # The LEADING structured token + target= are what checklist step 4 greps.
    assert note.startswith(f"deliberate-stop pid=n/a target=happy-session:{SID} "), note


def test_stop_unmapped_session_posts_nothing(monkeypatch):
    calls: list = []
    _patch_common(monkeypatch, calls, issue_map={})
    spawn_session.main(["stop", "--session-id", SID])
    assert [c[0] for c in calls] == ["rpc"], calls


def test_stop_proceeds_when_post_event_raises(monkeypatch, capsys):
    calls: list = []

    def boom(*_a, **_k):
        raise RuntimeError("task mid-move")

    _patch_common(monkeypatch, calls, issue_map={SID: ISSUE}, post_event_fn=boom)
    spawn_session.main(["stop", "--session-id", SID])  # must NOT raise
    assert [c[0] for c in calls] == ["rpc"], calls
    err = capsys.readouterr().err
    assert "WARN: deliberate-stop breadcrumb failed" in err


def test_stop_reason_flag_lands_in_note(monkeypatch):
    calls: list = []
    _patch_common(monkeypatch, calls, issue_map={SID: ISSUE})
    spawn_session.main(["stop", "--session-id", SID, "--reason", "runaway grid"])
    note = calls[0][3]["note"]
    assert note.endswith("reason=runaway grid"), note


def test_stop_watcher_source_posts_nothing(monkeypatch):
    """The round-1 methodology Must-Fix pin: a MAPPED session stopped with
    ``--stop-source watcher`` posts no breadcrumb (the watcher's `_stop_session`
    threads this flag on every automated stop pass)."""
    calls: list = []
    _patch_common(monkeypatch, calls, issue_map={SID: ISSUE})
    spawn_session.main(["stop", "--session-id", SID, "--stop-source", "watcher"])
    assert [c[0] for c in calls] == ["rpc"], calls


def test_stop_rpc_failure_still_posts_breadcrumb_first(monkeypatch):
    calls: list = []
    _patch_common(monkeypatch, calls, issue_map={SID: ISSUE}, rpc_resp={"success": False})
    with pytest.raises(SystemExit):
        spawn_session.main(["stop", "--session-id", SID])
    assert [c[0] for c in calls] == ["marker", "rpc"], calls


def test_stop_bounded_when_post_event_hangs(monkeypatch, capsys):
    """The blocking-flock pin: `post_event` enters a no-timeout flock, so an
    exception-only fail-soft never fires on a wedged-lock WAIT — the daemon
    thread + join bound must keep the stop moving."""
    calls: list = []

    def hang(*_a, **_k):
        time.sleep(5.0)  # far past the (shrunk) join bound below

    _patch_common(monkeypatch, calls, issue_map={SID: ISSUE}, post_event_fn=hang)
    monkeypatch.setattr(spawn_session, "STOP_BREADCRUMB_JOIN_TIMEOUT_S", 0.2)
    t0 = time.monotonic()
    spawn_session.main(["stop", "--session-id", SID])
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"stop blocked for {elapsed:.1f}s despite the join bound"
    assert ("rpc", "/stop-session", {"sessionId": SID}) in calls
    err = capsys.readouterr().err
    assert "still posting" in err
