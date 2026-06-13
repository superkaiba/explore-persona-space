"""Tests for scripts/pm_queue_report.py (the PM session's STATUS report source).

Pins the script's non-trivial logic so a regression cannot silently
corrupt the `/pm` boot scan (research-pm.md Mode 1):

- ``created_ts`` fallback chain (first events.jsonl event ts → frontmatter
  ``created_at``);
- ``status_arrival_ts`` fallback chain (last ``epm:status-changed`` with
  ``to`` == current status → last event ts → None);
- malformed / empty events.jsonl LINES are skipped while file-level
  errors still raise (fail loud per project rules);
- ``latest_marker_*`` fields present only for ACTIVE_STATUSES rows;
- ``render_markdown`` recency sort order (awaiting_promotion by
  ``status_arrival_ts`` desc, proposed by ``created_ts`` desc);
- ``--status`` + ``--include-terminal`` together is a loud
  ``parser.error`` (exit code 2), not a silent ignore.

Uses the library-level fake_repo / monkeypatched-``repo_root`` pattern
from tests/test_task_workflow.py — CLI subprocess tests cannot redirect
the branch-guarded resolver, so everything runs in-process against a
tmp git repo. Any ``tasks/...`` path is formed via the canonical
resolver (``find_task_path``), per
tests/test_no_direct_task_path_construction.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

SCRIPT_PATH = _REPO_ROOT / "scripts" / "pm_queue_report.py"
spec = importlib.util.spec_from_file_location("pm_queue_report", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
pm_queue_report = importlib.util.module_from_spec(spec)
sys.modules["pm_queue_report"] = pm_queue_report
spec.loader.exec_module(pm_queue_report)


# ─── Fake-repo fixture (mirrors tests/test_task_workflow.py) ───────────────


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up tmp_path as a git repo and rebind task_workflow's resolver.

    pm_queue_report binds ``get_task`` / ``find_task_path`` /
    ``list_by_status`` at import time, but those function objects resolve
    ``repo_root()`` / ``tasks_dir()`` / ``registry_path()`` from the
    task_workflow module globals at CALL time, so monkeypatching the
    functions on the singleton module redirects the script too.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)

    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    lock_dir = tmp_path / ".task-workflow"
    monkeypatch.setattr(tw, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(tw, "LOCK_PATH", lock_dir / "lock")
    return tmp_path, tw


def _new_task(tw, title: str, status: str = "proposed") -> int:
    return tw.create_task(tw.NewTaskRequest(kind="experiment", title=title, status=status))


def _write_events(tw, task_id: int, rows: list) -> None:
    """Overwrite the task's events.jsonl. Dict rows are JSON-dumped; str
    rows are written verbatim (for malformed-line fixtures)."""
    path = tw.find_task_path(task_id) / "events.jsonl"
    lines = [row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ─── created_ts fallback chain ──────────────────────────────────────────────


def test_created_ts_prefers_first_event_ts(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "first-event ts wins")
    _write_events(
        tw,
        tid,
        [
            {"ts": "2026-06-01T00:00:00Z", "kind": "epm:created", "version": 1},
            {"ts": "2026-06-02T00:00:00Z", "kind": "epm:progress", "version": 1},
        ],
    )
    rec = pm_queue_report._task_record(tid)
    assert rec["created_ts"] == "2026-06-01T00:00:00Z"


def test_created_ts_falls_back_to_frontmatter_when_no_events(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "frontmatter fallback")
    _write_events(tw, tid, [])  # truncate the seed epm:created event
    fm_created = tw.get_task(tid)["frontmatter"]["created_at"]
    rec = pm_queue_report._task_record(tid)
    assert rec["created_ts"] == fm_created
    # No events at all → status_arrival_ts has nothing to fall back to.
    assert rec["status_arrival_ts"] is None


# ─── status_arrival_ts fallback chain ───────────────────────────────────────


def test_status_arrival_uses_last_status_changed_into_current(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "arrival via status-changed", status="running")
    _write_events(
        tw,
        tid,
        [
            {"ts": "2026-06-01T00:00:00Z", "kind": "epm:created", "version": 1},
            {"ts": "2026-06-02T00:00:00Z", "kind": "epm:status-changed", "to": "running"},
            {"ts": "2026-06-03T00:00:00Z", "kind": "epm:status-changed", "to": "verifying"},
            {"ts": "2026-06-04T00:00:00Z", "kind": "epm:status-changed", "to": "running"},
            # Trailing arrival into a DIFFERENT status: a naive "last
            # status-changed regardless of `to`" implementation picks this one.
            {"ts": "2026-06-04T12:00:00Z", "kind": "epm:status-changed", "to": "blocked"},
            {"ts": "2026-06-05T00:00:00Z", "kind": "epm:progress", "version": 1},
        ],
    )
    rec = pm_queue_report._task_record(tid)
    # LAST arrival into the current status, not the first, not another status's.
    assert rec["status_arrival_ts"] == "2026-06-04T00:00:00Z"


def test_status_arrival_falls_back_to_last_event_ts(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "arrival fallback", status="running")
    _write_events(
        tw,
        tid,
        [
            {"ts": "2026-06-01T00:00:00Z", "kind": "epm:created", "version": 1},
            {"ts": "2026-06-02T00:00:00Z", "kind": "epm:progress", "version": 1},
        ],
    )
    rec = pm_queue_report._task_record(tid)
    assert rec["status_arrival_ts"] == "2026-06-02T00:00:00Z"


# ─── events.jsonl parsing: line-level skip, file-level raise ────────────────


def test_malformed_and_non_dict_event_lines_are_skipped(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "corrupt line tolerance", status="running")
    _write_events(
        tw,
        tid,
        [
            {"ts": "2026-06-01T00:00:00Z", "kind": "epm:created", "version": 1},
            "{not valid json",
            "",
            '"a json string, not an object"',
            {"ts": "2026-06-02T00:00:00Z", "kind": "epm:progress", "version": 1},
        ],
    )
    rec = pm_queue_report._task_record(tid)
    # Both valid events survive the corrupt middle lines.
    assert rec["created_ts"] == "2026-06-01T00:00:00Z"
    assert rec["latest_marker_kind"] == "epm:progress"
    assert rec["latest_marker_ts"] == "2026-06-02T00:00:00Z"


def test_file_level_events_error_still_raises(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "fail loud on file error")
    events_path = tw.find_task_path(tid) / "events.jsonl"
    events_path.unlink()
    events_path.mkdir()  # exists() is True; read_text() raises IsADirectoryError
    with pytest.raises(OSError):
        pm_queue_report._read_events(tw.find_task_path(tid))


def test_missing_events_file_is_empty_not_error(fake_repo):
    _, tw = fake_repo
    tid = _new_task(tw, "no events file")
    (tw.find_task_path(tid) / "events.jsonl").unlink()
    assert pm_queue_report._read_events(tw.find_task_path(tid)) == []


# ─── latest_marker_* only for ACTIVE_STATUSES ───────────────────────────────


def test_latest_marker_fields_only_for_active_statuses(fake_repo):
    _, tw = fake_repo
    active = _new_task(tw, "active row", status="running")
    parked = _new_task(tw, "parked row", status="proposed")
    assert "running" in pm_queue_report.ACTIVE_STATUSES
    assert "proposed" not in pm_queue_report.ACTIVE_STATUSES
    active_rec = pm_queue_report._task_record(active)
    parked_rec = pm_queue_report._task_record(parked)
    assert "latest_marker_kind" in active_rec
    assert "latest_marker_ts" in active_rec
    assert "latest_marker_kind" not in parked_rec
    assert "latest_marker_ts" not in parked_rec


# ─── render_markdown recency sort order ─────────────────────────────────────


def test_awaiting_promotion_most_recent_sorted_by_arrival_desc(fake_repo):
    _, tw = fake_repo
    # The later-arriving row gets the HIGHER id: list_by_status yields rows in
    # ascending-id order, so an unsorted implementation would emit [early, late]
    # and fail the assertion — only the desc sort by status_arrival_ts passes.
    early = _new_task(tw, "arrived earlier", status="awaiting_promotion")
    late = _new_task(tw, "arrived later", status="awaiting_promotion")
    _write_events(
        tw,
        late,
        [{"ts": "2026-06-05T00:00:00Z", "kind": "epm:status-changed", "to": "awaiting_promotion"}],
    )
    _write_events(
        tw,
        early,
        [{"ts": "2026-06-01T00:00:00Z", "kind": "epm:status-changed", "to": "awaiting_promotion"}],
    )
    report = pm_queue_report.build_report(("awaiting_promotion",))
    md = pm_queue_report.render_markdown(report)
    recent = md.split("### Most recent")[1].split("### By theme")[0]
    assert recent.index(f"#{late} ") < recent.index(f"#{early} ")


def test_proposed_recently_filed_sorted_by_created_desc(fake_repo):
    _, tw = fake_repo
    # The later-filed row gets the HIGHER id: list_by_status yields rows in
    # ascending-id order, so an unsorted implementation would emit [early, late]
    # and fail the assertion — only the desc sort by created_ts passes.
    early = _new_task(tw, "filed earlier", status="proposed")
    late = _new_task(tw, "filed later", status="proposed")
    _write_events(tw, late, [{"ts": "2026-06-07T00:00:00Z", "kind": "epm:created", "version": 1}])
    _write_events(tw, early, [{"ts": "2026-06-03T00:00:00Z", "kind": "epm:created", "version": 1}])
    report = pm_queue_report.build_report(("proposed",))
    md = pm_queue_report.render_markdown(report)
    filed = md.split("### Recently filed")[1].split("### By theme")[0]
    assert filed.index(f"#{late} ") < filed.index(f"#{early} ")


# ─── CLI flag handling ──────────────────────────────────────────────────────


def test_status_with_include_terminal_is_a_loud_error(fake_repo, capsys):
    with pytest.raises(SystemExit) as excinfo:
        pm_queue_report.main(["--status", "proposed", "--include-terminal"])
    assert excinfo.value.code == 2  # argparse parser.error exit code
    err = capsys.readouterr().err
    # Distinctive fragment of the actual message — the argparse usage line
    # already contains both flag names, so flag-name asserts are vacuous.
    assert "cannot be combined" in err


def test_status_alone_still_works(fake_repo, capsys):
    _, tw = fake_repo
    _new_task(tw, "one completed row", status="completed")
    rc = pm_queue_report.main(["--status", "completed"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"completed": 1}
