"""Pure-function tests for the session visibility layer (EPS workflow v2 §2).

Covers the parts that need no live Happy daemon / tmux server:

- ``eps_sessions.build_happy_rows`` — EPS filter, issue mapping (registry +
  worktree-cwd inference), live status lookup, tmux-target join, summary/age.
- ``eps_sessions.build_tmux_only_rows`` — fallback rows for daemon-untracked
  EPS tmux sessions; covered / non-EPS windows are skipped.
- ``eps_sessions.tmux_target_by_node_pid`` — node-pid -> tmux-target map.
- ``eps_sessions.render_rows_text`` — table + daemon-down footer.
- ``session_summarize.render_sessions_digest`` / ``write_sessions_digest`` —
  digest markdown rendering (sorting, escaping, anomaly flags) + atomic write.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eps_sessions  # noqa: E402
import session_resolver  # noqa: E402
import session_summarize  # noqa: E402
import tmux_window_titles  # noqa: E402

_EPS_ROOT = "/home/thomasjiralerspong/explore-persona-space"
_WT_42 = f"{_EPS_ROOT}/.claude/worktrees/issue-42"


def _iso(offset_s: int = 0) -> tuple[str, float]:
    """Return (iso_ts, now_epoch) such that ``now_epoch`` is ``offset_s``
    seconds after ``iso_ts`` — deterministic ages for the age formatter."""
    base = datetime(2026, 7, 3, 0, 0, 0, tzinfo=UTC)
    return base.strftime("%Y-%m-%dT%H:%M:%SZ"), base.timestamp() + offset_s


def _win(session: str, index: str, pane_pid: int) -> tmux_window_titles.Window:
    return tmux_window_titles.Window(session=session, index=index, name="node", pane_pid=pane_pid)


# ── tmux_target_by_node_pid ─────────────────────────────────────────────────


def test_tmux_target_map_covers_whole_subtree():
    windows = [_win("eps-issue-42", "0", 100), _win("pm", "1", 200)]
    subtree = {100: [100, 101, 102], 200: [200, 201]}
    m = eps_sessions.tmux_target_by_node_pid(windows, lambda p: subtree[p])
    assert m[101] == "eps-issue-42:0"
    assert m[102] == "eps-issue-42:0"
    assert m[201] == "pm:1"


def test_tmux_target_map_first_window_wins_on_shared_pid():
    windows = [_win("a", "0", 100), _win("b", "0", 200)]
    # pid 999 appears under both subtrees; the first window claims it.
    m = eps_sessions.tmux_target_by_node_pid(windows, lambda p: [p, 999])
    assert m[999] == "a:0"


# ── build_happy_rows ─────────────────────────────────────────────────────────


def test_happy_row_from_registry_issue_mapping():
    ts, now = _iso(180)  # 3 minutes of activity age
    children = [{"happySessionId": "sess-a", "pid": 100}]
    meta = {"sess-a": {"path": _WT_42}}
    issue_map = {"sess-a": 42}
    cache = {"sess-a": {"summary": "Running Phase-1.5 fact-check", "last_activity_ts": ts}}
    target_map = {100: "eps-issue-42:0"}
    rows = eps_sessions.build_happy_rows(
        children,
        meta,
        issue_map,
        cache,
        target_map,
        status_fn=lambda i: "planning",
        now=now,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.issue == 42
    assert r.status == "planning"
    assert r.source == "happy"
    assert r.session_id == "sess-a"
    assert r.tmux == "eps-issue-42:0"
    assert r.last_activity_age == "3m ago"
    assert r.summary == "Running Phase-1.5 fact-check"
    assert r.error is None


def test_happy_row_infers_issue_from_worktree_cwd_when_unregistered():
    """No registry entry, but the cwd is an issue-<N> worktree -> attribute it."""
    children = [{"happySessionId": "sess-b", "pid": 5}]
    meta = {"sess-b": {"path": _WT_42}}
    rows = eps_sessions.build_happy_rows(
        children,
        meta,
        {},
        {},
        {},
        status_fn=lambda i: "running",
    )
    assert len(rows) == 1
    assert rows[0].issue == 42
    assert rows[0].status == "running"


def test_happy_row_non_eps_session_is_dropped():
    children = [{"happySessionId": "sess-mg", "pid": 7}]
    meta = {"sess-mg": {"path": "/home/thomasjiralerspong/my-goat"}}
    rows = eps_sessions.build_happy_rows(
        children,
        meta,
        {},
        {},
        {},
        status_fn=lambda i: "?",
    )
    assert rows == []


def test_happy_row_eps_root_without_issue_maps_to_dash():
    """A repo-root EPS session (e.g. the PM session) is EPS but has no issue."""
    children = [{"happySessionId": "sess-pm", "pid": 9}]
    meta = {"sess-pm": {"path": _EPS_ROOT}}
    rows = eps_sessions.build_happy_rows(
        children,
        meta,
        {},
        {},
        {},
        status_fn=lambda i: "SHOULD-NOT-BE-CALLED",
    )
    assert len(rows) == 1
    assert rows[0].issue is None
    assert rows[0].status == "-"


def test_happy_row_surfaces_summarizer_error():
    children = [{"happySessionId": "sess-e", "pid": 3}]
    meta = {"sess-e": {"path": _WT_42}}
    cache = {"sess-e": {"error": "tail read failed: OSError", "summary": None}}
    rows = eps_sessions.build_happy_rows(
        children,
        meta,
        {"sess-e": 42},
        cache,
        {},
        status_fn=lambda i: "running",
    )
    assert rows[0].error == "tail read failed: OSError"
    assert rows[0].summary == ""


def test_happy_row_skips_malformed_child():
    children = [{"pid": 100}, {"happySessionId": 12345, "pid": 1}]
    rows = eps_sessions.build_happy_rows(
        children,
        {},
        {},
        {},
        {},
        status_fn=lambda i: "?",
    )
    assert rows == []


# ── build_tmux_only_rows ─────────────────────────────────────────────────────


def _rr(cwd: str | None, node_pid: int | None, issue: int | None, reason: str | None = None):
    return session_resolver.ResolveResult(
        node_pid=node_pid,
        claude_pid=None,
        cwd=cwd,
        transcript="/t.jsonl",
        issue=issue,
        reason=reason,
    )


def test_tmux_only_row_for_untracked_eps_session():
    windows = [_win("orphan", "0", 500)]
    resolved = {500: _rr(_WT_42, node_pid=501, issue=42)}
    rows = eps_sessions.build_tmux_only_rows(
        windows,
        covered_node_pids=set(),
        resolve_fn=lambda p: resolved.get(p),
        status_fn=lambda i: "running",
    )
    assert len(rows) == 1
    assert rows[0].issue == 42
    assert rows[0].source == "tmux-only"
    assert rows[0].tmux == "orphan:0"
    assert rows[0].session_id is None


def test_tmux_only_row_skips_covered_node_pid():
    """A window already represented by a Happy row (its node pid is a live
    daemon child) is NOT re-emitted as a tmux-only row."""
    windows = [_win("s", "0", 500)]
    resolved = {500: _rr(_WT_42, node_pid=501, issue=42)}
    rows = eps_sessions.build_tmux_only_rows(
        windows,
        covered_node_pids={501},
        resolve_fn=lambda p: resolved.get(p),
        status_fn=lambda i: "running",
    )
    assert rows == []


def test_tmux_only_row_skips_non_eps_and_unresolvable():
    windows = [_win("mg", "0", 600), _win("dead", "1", 700)]
    resolved = {
        600: _rr("/home/thomasjiralerspong/my-goat", node_pid=601, issue=None),
        700: None,  # unresolvable window
    }
    rows = eps_sessions.build_tmux_only_rows(
        windows,
        covered_node_pids=set(),
        resolve_fn=lambda p: resolved.get(p),
        status_fn=lambda i: "?",
    )
    assert rows == []


# ── render_rows_text ─────────────────────────────────────────────────────────


def test_render_rows_text_orders_and_footers():
    rows = [
        eps_sessions.SessionRow(
            issue=None,
            status="-",
            source="happy",
            tmux="pm:0",
            session_id="pm",
            last_activity_age="",
            summary="pm",
        ),
        eps_sessions.SessionRow(
            issue=42,
            status="planning",
            source="happy",
            tmux="e:0",
            session_id="a",
            last_activity_age="3m ago",
            summary="doing X",
        ),
    ]
    out = eps_sessions.render_rows_text(rows, daemon_up=True)
    # Mapped issue #42 sorts before the unmapped (None) row.
    assert out.index("#42") < out.index("pm")
    assert "2 issue session(s)." in out
    assert "unreachable" not in out


def test_render_rows_text_daemon_down_footer():
    out = eps_sessions.render_rows_text([], daemon_up=False)
    assert "0 issue session(s)." in out
    assert "Happy daemon unreachable" in out


# ── digest rendering (session_summarize) ─────────────────────────────────────


def test_render_sessions_digest_basic_shape():
    ts, _ = _iso()
    payload = {
        "updated_at": ts,
        "sessions": {
            "sess-a": {"issue": 42, "status": "planning", "summary": "fact-check v2"},
            "sess-b": {"issue": 7, "status": "running", "summary": "training"},
        },
    }
    md = session_summarize.render_sessions_digest(payload)
    assert md.startswith("# EPS sessions digest")
    assert f"_Updated: {ts} — 2 live session(s)._" in md
    assert "| Issue | Status | Summary | Flag |" in md
    # Sorted ascending by issue: #7 before #42.
    assert md.index("| #7 |") < md.index("| #42 |")
    assert "| #42 | planning | fact-check v2 |  |" in md


def test_render_sessions_digest_flags_blocked_and_error():
    payload = {
        "updated_at": "2026-07-03T00:00:00Z",
        "sessions": {
            "sess-x": {"issue": 5, "status": "blocked", "summary": "stuck"},
            "sess-y": {
                "issue": 6,
                "status": "running",
                "summary": None,
                "error": "tail read failed: OSError: boom",
            },
        },
    }
    md = session_summarize.render_sessions_digest(payload)
    assert "| #5 | blocked | stuck | blocked |" in md
    # No summary -> falls back to the error text; flag names the error class.
    assert "summarize-error (tail read failed)" in md


def test_render_sessions_digest_escapes_pipes_and_unmapped_last():
    payload = {
        "updated_at": "2026-07-03T00:00:00Z",
        "sessions": {
            "sess-u": {"issue": None, "status": "-", "summary": "unmapped"},
            "sess-p": {"issue": 3, "status": "running", "summary": "a | b | c"},
        },
    }
    md = session_summarize.render_sessions_digest(payload)
    assert r"a \| b \| c" in md  # pipes escaped so the table stays well-formed
    assert md.index("| #3 |") < md.index("| - |")  # unmapped sorts last


def test_render_sessions_digest_empty_payload():
    md = session_summarize.render_sessions_digest({"updated_at": "2026-07-03T00:00:00Z"})
    assert "0 live session(s)." in md
    assert "| Issue | Status | Summary | Flag |" in md  # valid empty table


def test_write_sessions_digest_atomic_round_trip(tmp_path):
    dest = tmp_path / "cache" / "sessions-digest.md"
    payload = {"updated_at": "2026-07-03T00:00:00Z", "sessions": {}}
    written = session_summarize.write_sessions_digest(payload, path=dest)
    assert written == dest
    assert dest.read_text() == session_summarize.render_sessions_digest(payload)
    # Atomicity: no leftover temp file beside the destination.
    assert not list(dest.parent.glob("*.tmp"))
