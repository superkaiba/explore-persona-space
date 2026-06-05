"""Pure-function tests for the `spawn_session.py` list-enrichment path.

What this pins:

1. **Watcher contract preserved.** The manual-session registry file is named
   ``manual-issue-<N>.json`` so the watcher's ``issue-*.json`` glob does NOT
   match it — manual sessions must never be auto-respawned. If anyone renames
   the manual file to ``issue-<N>.json``, the watcher would start auto-driving
   user-spawned sessions; this test catches that whole class of regression.
2. **Issue mapping covers both autonomous + manual entries**, and resolves a
   session-id collision (rare but possible: re-spawn after schema migration)
   by latest `spawned_at`.
3. **Progress cell formatting is honest about failure** — broken rows surface
   a visible placeholder, not a silent blank (matches CLAUDE.md fail-fast).
4. **Manual-session registration writes the expected shape atomically.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spawn_session  # noqa: E402

# ── manual-session registry ────────────────────────────────────────────────


def test_manual_register_writes_atomic_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    spawn_session._register_manual_session(488, "sess-abc", "/repo")
    dest = tmp_path / "manual-issue-488.json"
    entry = json.loads(dest.read_text())
    assert entry["issue"] == 488
    assert entry["happy_session_id"] == "sess-abc"
    assert entry["cwd"] == "/repo"
    assert entry["mode"] == "manual"
    # Atomicity: no leftover temp file.
    assert not list(tmp_path.glob("*.tmp"))


def test_manual_register_uses_distinct_filename_from_autonomous(tmp_path, monkeypatch):
    """The watcher's respawn pass globs `issue-*.json`; manual entries must
    live at `manual-issue-*.json` so they are NOT auto-respawned. If this test
    fails because someone renamed the file, the watcher would start treating
    user-driven sessions as autonomous restarts -> duplicate sessions / pods."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    spawn_session._register_manual_session(488, "sess-abc", "/repo")
    spawn_session._register_autonomous_session(489, "sess-xyz", "/repo", 24.0)
    # Critical invariant: watcher's glob picks up ONLY the autonomous one.
    watcher_matches = sorted(p.name for p in tmp_path.glob("issue-*.json"))
    assert watcher_matches == ["issue-489.json"], watcher_matches
    # Sanity: both files exist on disk.
    assert (tmp_path / "manual-issue-488.json").is_file()
    assert (tmp_path / "issue-489.json").is_file()


# ── session-id -> issue mapping ────────────────────────────────────────────


def test_load_session_issue_map_merges_autonomous_and_manual(tmp_path, monkeypatch):
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    spawn_session._register_autonomous_session(488, "sess-auto", "/repo", 50.0)
    spawn_session._register_manual_session(492, "sess-manual", "/repo")
    out = spawn_session._load_session_issue_map()
    assert out == {"sess-auto": 488, "sess-manual": 492}


def test_load_session_issue_map_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path / "does-not-exist")
    assert spawn_session._load_session_issue_map() == {}


def test_load_session_issue_map_skips_malformed_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    # One good entry.
    spawn_session._register_autonomous_session(488, "sess-good", "/repo", 50.0)
    # One unparseable file (truncated JSON).
    (tmp_path / "issue-999.json").write_text("{ not valid json")
    # One entry missing required fields.
    (tmp_path / "manual-issue-1000.json").write_text(json.dumps({"issue": 1000}))
    # Map should contain only the good one; bad entries are skipped quietly.
    out = spawn_session._load_session_issue_map()
    assert out == {"sess-good": 488}


def test_load_session_issue_map_collision_resolves_to_latest(tmp_path, monkeypatch):
    """If the same session id appears under both prefixes (autonomous restart
    after manual spawn, or vice versa), the LATER `spawned_at` wins."""
    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    # Manual entry stamped at t=100.
    (tmp_path / "manual-issue-200.json").write_text(
        json.dumps({"issue": 200, "happy_session_id": "shared", "spawned_at": 100.0})
    )
    # Autonomous entry for a different issue stamped LATER at t=200 — wins.
    (tmp_path / "issue-201.json").write_text(
        json.dumps({"issue": 201, "happy_session_id": "shared", "spawned_at": 200.0})
    )
    out = spawn_session._load_session_issue_map()
    assert out["shared"] == 201


# ── progress-cell formatting ───────────────────────────────────────────────


def test_progress_cell_unknown_task_shows_visible_placeholder(monkeypatch):
    """Lookup failure must NOT silently blank — surface a labeled placeholder."""

    # Force the lazy import to a fake module that raises FileNotFoundError.
    class _FakeWorkflow:
        @staticmethod
        def get_task(issue):
            raise FileNotFoundError(f"task #{issue} not found")

        @staticmethod
        def latest_event(issue, prefix=None):
            return None

    monkeypatch.setitem(sys.modules, "explore_persona_space.task_workflow", _FakeWorkflow)
    cell = spawn_session._format_progress_cell(99999)
    assert "not found" in cell
    assert "99999" in cell


def test_progress_cell_renders_status_kind_age_note(monkeypatch):
    from datetime import datetime

    event_ts = "2026-06-05T11:00:00Z"
    event_epoch = datetime.fromisoformat(event_ts.replace("Z", "+00:00")).timestamp()

    class _FakeWorkflow:
        @staticmethod
        def get_task(issue):
            return {"id": issue, "status": "running", "frontmatter": {}, "body": "", "path": "x"}

        @staticmethod
        def latest_event(issue, prefix=None):
            return {
                "kind": "epm:progress",
                "ts": event_ts,
                "note": "phase 2 of 4 done",
            }

    monkeypatch.setitem(sys.modules, "explore_persona_space.task_workflow", _FakeWorkflow)
    # Fix "now" 30 minutes after the event timestamp so the age render is
    # deterministic across timezones.
    cell = spawn_session._format_progress_cell(488, now=event_epoch + 1800)
    assert "running" in cell
    assert "progress" in cell  # `epm:` prefix dropped
    assert "phase 2 of 4 done" in cell
    assert "30m ago" in cell


def test_progress_cell_truncates_long_note(monkeypatch):
    from datetime import datetime

    event_ts = "2026-06-05T11:00:00Z"
    event_epoch = datetime.fromisoformat(event_ts.replace("Z", "+00:00")).timestamp()

    class _FakeWorkflow:
        @staticmethod
        def get_task(issue):
            return {"id": issue, "status": "running", "frontmatter": {}, "body": "", "path": "x"}

        @staticmethod
        def latest_event(issue, prefix=None):
            return {
                "kind": "epm:progress",
                "ts": event_ts,
                "note": "x" * 500,  # absurdly long; must be truncated
            }

    monkeypatch.setitem(sys.modules, "explore_persona_space.task_workflow", _FakeWorkflow)
    cell = spawn_session._format_progress_cell(488, now=event_epoch + 60)
    # Must fit roughly within the budget (a tiny overshoot from the wrapper
    # parens + spacing is OK; the point is no 500-char dump).
    assert len(cell) <= spawn_session._PROGRESS_CELL_MAX + 10, len(cell)
    assert "…" in cell  # ellipsis marks the cut


def test_progress_cell_no_markers_yet(monkeypatch):
    class _FakeWorkflow:
        @staticmethod
        def get_task(issue):
            return {"id": issue, "status": "proposed", "frontmatter": {}, "body": "", "path": "x"}

        @staticmethod
        def latest_event(issue, prefix=None):
            return None

    monkeypatch.setitem(sys.modules, "explore_persona_space.task_workflow", _FakeWorkflow)
    cell = spawn_session._format_progress_cell(700)
    assert "proposed" in cell
    assert "no marker" in cell.lower()


def test_progress_cell_marker_read_failure_surfaces(monkeypatch):
    """`get_task` succeeded but `latest_event` blew up — must surface the
    error visibly, NOT silently report the status as fine."""

    class _FakeWorkflow:
        @staticmethod
        def get_task(issue):
            return {"id": issue, "status": "running", "frontmatter": {}, "body": "", "path": "x"}

        @staticmethod
        def latest_event(issue, prefix=None):
            raise RuntimeError("simulated read fail")

    monkeypatch.setitem(sys.modules, "explore_persona_space.task_workflow", _FakeWorkflow)
    cell = spawn_session._format_progress_cell(488)
    assert "marker-read failed" in cell
    assert "RuntimeError" in cell


# ── age formatting ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "delta_s,expected",
    [
        (30, "30s ago"),
        (90, "1m ago"),
        (3700, "1h ago"),
        (90000, "1d ago"),
    ],
)
def test_format_event_age_buckets(delta_s, expected):
    from datetime import datetime

    base_ts = "2026-06-05T11:00:00Z"
    base_epoch = datetime.fromisoformat(base_ts.replace("Z", "+00:00")).timestamp()
    out = spawn_session._format_event_age(base_ts, now=base_epoch + delta_s)
    assert out == expected


def test_format_event_age_missing_returns_empty():
    assert spawn_session._format_event_age(None) == ""
    assert spawn_session._format_event_age("") == ""
    assert spawn_session._format_event_age("not-a-timestamp") == ""
