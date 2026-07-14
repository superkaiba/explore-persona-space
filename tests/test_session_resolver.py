"""Pure-function tests for the Happy-session resolver.

What this pins:

1. **Slug derivation matches Claude Code's projects-dir convention.** A
   regression here would break every transcript lookup — the resolver would
   silently never find a project dir.
2. **Issue extraction is anchored to ``/issue <N>``** (not a bare ``issue N``
   substring in prose).
3. **Happy-log scan picks the LAST `transcript_path`.** Claude can switch its
   session UUID mid-life; the most recent log line is the live transcript.
4. **``is_eps_cwd`` correctly distinguishes EPS (incl. worktrees) from other
   projects.** A regression here would cause the EPS-only filter in
   ``spawn_session.py list`` to leak non-EPS sessions into the default view.

These tests deliberately avoid touching /proc or live processes — those are
exercised in the resolver's CLI smoke check (see Report section).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import session_resolver  # noqa: E402

# ── slug derivation ────────────────────────────────────────────────────────


def test_derive_project_slug_repo_root():
    cwd = "/home/thomasjiralerspong/explore-persona-space"
    assert (
        session_resolver.derive_project_slug(cwd)
        == "-home-thomasjiralerspong-explore-persona-space"
    )


def test_derive_project_slug_worktree_with_dotclaude_path():
    # A path containing a literal ``.claude`` MUST produce a double-dash
    # because the ``.`` is non-alphanumeric. Confirmed empirically against
    # the live ~/.claude/projects/ tree (issue-459 worktree has slug
    # ``...space--claude-worktrees-issue-459``).
    cwd = "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-459"
    expected = "-home-thomasjiralerspong-explore-persona-space--claude-worktrees-issue-459"
    assert session_resolver.derive_project_slug(cwd) == expected


def test_derive_project_slug_my_goat():
    # Non-EPS path used by the my-goat session — slug must be derivable too.
    assert (
        session_resolver.derive_project_slug("/home/thomasjiralerspong/my-goat")
        == "-home-thomasjiralerspong-my-goat"
    )


def test_derive_project_slug_collapses_only_non_alphanumeric():
    # The substitution is character-by-character, NOT token-by-token: two
    # adjacent non-alphanumerics become TWO dashes (this is why ``.claude``
    # gives ``--claude``). Pin to catch a "regex collapses runs" regression.
    assert session_resolver.derive_project_slug("/a/.b") == "-a--b"


# ── issue extraction ───────────────────────────────────────────────────────


def test_extract_issue_from_bare_command():
    txt = '{"role":"user","content":"/issue 488 do the thing"}'
    assert session_resolver.extract_issue_from_text(txt) == 488


def test_extract_issue_from_loop_command():
    # The autonomous /loop wrapper is the canonical shape for `--auto`
    # sessions; must still extract the inner issue number.
    txt = "<command-args>10m /issue 492</command-args>"
    assert session_resolver.extract_issue_from_text(txt) == 492


def test_extract_issue_ignores_bare_issue_substring():
    # ``issue 488`` (no slash) appears in prose all the time and MUST NOT
    # match — only the slash-prefixed slash-command form counts.
    txt = "we filed issue 488 last week, see #488 for context"
    assert session_resolver.extract_issue_from_text(txt) is None


def test_extract_issue_first_match_wins():
    # A transcript head may have multiple /issue references (e.g. a /loop
    # prompt followed by user follow-ups); the FIRST is the canonical driver.
    txt = "/issue 100 some preamble /issue 200"
    assert session_resolver.extract_issue_from_text(txt) == 100


# ── happy-log transcript extraction ────────────────────────────────────────


def test_extract_transcript_from_happy_log_returns_last():
    # If Claude reloaded session ids during this happy node's life, the LAST
    # transcript_path entry is the current live one. Two entries -> we pick
    # the second.
    old_path = "/home/x/.claude/projects/-p/old.jsonl"
    new_path = "/home/x/.claude/projects/-p/new.jsonl"
    log = (
        f'[03:07:34] [hookServer] {{"session_id":"OLD","transcript_path":"{old_path}"}}\n'
        f'[03:08:11] [hookServer] {{"session_id":"NEW","transcript_path":"{new_path}"}}\n'
    )
    out = session_resolver.extract_transcript_from_happy_log(log)
    assert out == new_path


def test_extract_transcript_from_happy_log_none_when_missing():
    log = "[03:07:32.886] [START] Reporting session abc to daemon\n"
    assert session_resolver.extract_transcript_from_happy_log(log) is None


# ── EPS-cwd predicate ──────────────────────────────────────────────────────


def test_is_eps_cwd_repo_root():
    assert session_resolver.is_eps_cwd("/home/thomasjiralerspong/explore-persona-space")


def test_is_eps_cwd_worktree():
    # Worktrees under .claude/worktrees count as EPS — they share the project.
    cwd = "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-459"
    assert session_resolver.is_eps_cwd(cwd)


def test_is_eps_cwd_rejects_my_goat():
    assert not session_resolver.is_eps_cwd("/home/thomasjiralerspong/my-goat")


def test_is_eps_cwd_rejects_introsp():
    assert not session_resolver.is_eps_cwd("/home/thomasjiralerspong/introsp")


def test_is_eps_cwd_rejects_none():
    assert not session_resolver.is_eps_cwd(None)


def test_is_eps_cwd_rejects_empty():
    assert not session_resolver.is_eps_cwd("")


# ── happy-log file picker ──────────────────────────────────────────────────


def test_find_happy_log_for_node_picks_newest(monkeypatch, tmp_path):
    # Same pid, two log files (a prior incarnation + the current life). The
    # function must pick the most-recently modified one within the age cutoff.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    older = tmp_path / "2026-06-01-10-00-00-pid-123.log"
    newer = tmp_path / "2026-06-05-10-00-00-pid-123.log"
    older.write_text("old")
    newer.write_text("new")
    # Force mtimes — older is older.
    import os

    os.utime(older, (1_000_000.0, 1_000_000.0))
    os.utime(newer, (2_000_000.0, 2_000_000.0))
    # ``now`` well after both, but inside the 30-day window.
    out = session_resolver._find_happy_log_for_node(123, now=2_000_001.0)
    assert out == newer


def test_find_happy_log_for_node_skips_old(monkeypatch, tmp_path):
    # A log file older than the age cap is treated as left over from a
    # previous incarnation that re-used the same pid — must be ignored.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    too_old = tmp_path / "2025-01-01-10-00-00-pid-77.log"
    too_old.write_text("ancient")
    import os

    os.utime(too_old, (0.0, 0.0))
    # ``now`` 100 days later — well past the 30-day window.
    out = session_resolver._find_happy_log_for_node(77, now=100 * 86400)
    assert out is None


def test_find_happy_log_for_node_no_log_dir(monkeypatch, tmp_path):
    # No ~/.happy/logs dir at all (fresh laptop?) — must return None, not raise.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path / "does-not-exist")
    assert session_resolver._find_happy_log_for_node(1) is None


# ── resolve_transcript composition ─────────────────────────────────────────


def test_resolve_transcript_via_happy_log_round_trip(monkeypatch, tmp_path):
    # End-to-end happy-log path: a real log file + a real transcript file on
    # disk -> the resolver returns the transcript and no reason. The log
    # file is left with its current-time mtime so it passes the 30-day
    # freshness gate against the real wall clock.
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    transcript = tmp_path / "real-transcript.jsonl"
    transcript.write_text('{"type":"user","sessionId":"s"}\n')
    log = tmp_path / "2026-06-05-10-00-00-pid-555.log"
    log.write_text(f'[hookServer] {{"transcript_path":"{transcript}","cwd":"/x"}}')
    out, reason = session_resolver._resolve_transcript_via_happy_log(555)
    assert reason is None
    assert out == str(transcript)


def test_resolve_transcript_via_happy_log_missing_on_disk(monkeypatch, tmp_path):
    # The log references a transcript path that no longer exists on disk —
    # the resolver MUST surface a reason (not silently return a stale path
    # the caller would then fail to open).
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    log = tmp_path / "2026-06-05-10-00-00-pid-666.log"
    log.write_text('[hookServer] {"transcript_path":"/nope/missing.jsonl","cwd":"/x"}')
    out, reason = session_resolver._resolve_transcript_via_happy_log(666)
    assert out is None
    assert "missing on disk" in (reason or "")


def test_resolve_returns_reason_when_unresolvable(monkeypatch, tmp_path):
    # No happy log, no live process — both paths miss; the result MUST carry
    # a non-None reason (CLAUDE.md fail-fast: no silent None).
    monkeypatch.setattr(session_resolver, "HAPPY_LOGS_DIR", tmp_path)
    monkeypatch.setattr(session_resolver, "CLAUDE_PROJECTS_DIR", tmp_path / "no-projects-here")
    monkeypatch.setattr(session_resolver, "resolve_claude_pid", lambda pid: None)
    rr = session_resolver.resolve(99999)
    assert rr.transcript is None
    assert rr.issue is None
    assert rr.reason is not None
    assert "99999" in rr.reason or "no claude child pid" in rr.reason


# ── backfill ───────────────────────────────────────────────────────────────


def test_backfill_writes_manual_entry_for_unmapped_eps_session(monkeypatch, tmp_path):
    # An EPS session that has a resolvable issue but isn't in the registry
    # must get a manual-issue-<N>.json with mode=backfilled. Skipped if
    # already in the registry (autonomous OR manual).
    import spawn_session

    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(session_resolver, "spawn_session", spawn_session)
    # An EPS session already registered (must be skipped).
    spawn_session._register_autonomous_session(488, "sess-already", "/x", 24.0)
    # Pretend the daemon reports two live EPS sessions + one my-goat.
    monkeypatch.setattr(
        session_resolver,
        "_live_node_pids",
        lambda: [
            ("sess-already", 100),  # registered -> skipped
            ("sess-backfill", 200),  # unmapped EPS -> backfilled
            ("sess-mygoat", 300),  # non-EPS -> skipped
        ],
    )

    def fake_resolve(pid: int) -> session_resolver.ResolveResult:
        if pid == 100:
            return session_resolver.ResolveResult(100, 101, "/eps/path", "/t.jsonl", 488)
        if pid == 200:
            return session_resolver.ResolveResult(
                200,
                201,
                "/home/thomasjiralerspong/explore-persona-space",
                "/t.jsonl",
                999,
            )
        return session_resolver.ResolveResult(
            300, 301, "/home/thomasjiralerspong/my-goat", "/t.jsonl", None
        )

    monkeypatch.setattr(session_resolver, "resolve", fake_resolve)

    entries = session_resolver.backfill_labels(dry_run=False)
    assert len(entries) == 1
    assert entries[0]["issue"] == 999
    assert entries[0]["happy_session_id"] == "sess-backfill"
    dest = tmp_path / "manual-issue-999.json"
    assert dest.is_file()
    payload = json.loads(dest.read_text())
    assert payload["happy_session_id"] == "sess-backfill"
    assert payload["mode"] == "backfilled"
    # Watcher invariant preserved: the new file does NOT match `issue-*.json`.
    watcher_matches = sorted(p.name for p in tmp_path.glob("issue-*.json"))
    assert "manual-issue-999.json" not in watcher_matches


def test_backfill_dry_run_writes_nothing(monkeypatch, tmp_path):
    import spawn_session

    monkeypatch.setattr(spawn_session, "AUTONOMOUS_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(session_resolver, "spawn_session", spawn_session)
    monkeypatch.setattr(
        session_resolver,
        "_live_node_pids",
        lambda: [("sess-x", 1)],
    )
    monkeypatch.setattr(
        session_resolver,
        "resolve",
        lambda pid: session_resolver.ResolveResult(
            pid, 2, "/home/thomasjiralerspong/explore-persona-space", "/t.jsonl", 42
        ),
    )

    entries = session_resolver.backfill_labels(dry_run=True)
    assert len(entries) == 1
    assert entries[0]["issue"] == 42
    # The dry-run flag MUST suppress the write.
    assert not (tmp_path / "manual-issue-42.json").exists()


# ── tick-scoped transcript memo, #1182 ─────────────────────────────────────


def test_transcript_memo_caches_within_scope(monkeypatch):
    # Inside ONE scope, a second resolution of the same pid is a dict hit —
    # the underlying uncached resolver runs exactly once per DISTINCT pid,
    # and repeats return identical results INCLUDING the negative tuple
    # (acceptance criteria 1 + 4).
    calls: list[int] = []

    def fake_uncached(pid):
        calls.append(pid)
        if pid == 1:
            return ("/tmp/t.jsonl", None)
        return (None, "no happy log file for this node pid")

    monkeypatch.setattr(
        session_resolver, "_resolve_transcript_via_happy_log_uncached", fake_uncached
    )
    with session_resolver.transcript_resolution_scope():
        first_1 = session_resolver._resolve_transcript_via_happy_log(1)
        first_2 = session_resolver._resolve_transcript_via_happy_log(2)
        repeat_1 = session_resolver._resolve_transcript_via_happy_log(1)
        repeat_2 = session_resolver._resolve_transcript_via_happy_log(2)
    assert calls == [1, 2]
    assert first_1 == repeat_1 == ("/tmp/t.jsonl", None)
    assert first_2 == repeat_2 == (None, "no happy log file for this node pid")


def test_transcript_memo_off_by_default(monkeypatch):
    # NO scope active -> the memo global is None and every call performs a
    # full resolution (acceptance criterion 3: default path byte-identical).
    calls: list[int] = []

    def fake_uncached(pid):
        calls.append(pid)
        return ("/tmp/t.jsonl", None)

    monkeypatch.setattr(
        session_resolver, "_resolve_transcript_via_happy_log_uncached", fake_uncached
    )
    assert session_resolver._TRANSCRIPT_MEMO is None
    session_resolver._resolve_transcript_via_happy_log(7)
    session_resolver._resolve_transcript_via_happy_log(7)
    assert calls == [7, 7]
    assert session_resolver._TRANSCRIPT_MEMO is None


def test_transcript_memo_fresh_dict_per_scope(monkeypatch):
    # Never-across-ticks (acceptance criterion 2): each scope gets a FRESH
    # dict, so a result cached in scope A is re-resolved in scope B.
    calls: list[int] = []
    current = {"value": ("/tmp/x.jsonl", None)}

    def fake_uncached(pid):
        calls.append(pid)
        return current["value"]

    monkeypatch.setattr(
        session_resolver, "_resolve_transcript_via_happy_log_uncached", fake_uncached
    )
    with session_resolver.transcript_resolution_scope():
        assert session_resolver._resolve_transcript_via_happy_log(1) == ("/tmp/x.jsonl", None)
    assert session_resolver._TRANSCRIPT_MEMO is None
    current["value"] = ("/tmp/y.jsonl", None)
    with session_resolver.transcript_resolution_scope():
        assert session_resolver._resolve_transcript_via_happy_log(1) == ("/tmp/y.jsonl", None)
    assert calls == [1, 1]
    assert session_resolver._TRANSCRIPT_MEMO is None


def test_transcript_memo_scope_reentrant_restores_prev(monkeypatch):
    # Nested scopes: the inner exit restores the OUTER dict (not None); the
    # outer exit restores None.
    monkeypatch.setattr(
        session_resolver,
        "_resolve_transcript_via_happy_log_uncached",
        lambda pid: ("/tmp/t.jsonl", None),
    )
    assert session_resolver._TRANSCRIPT_MEMO is None
    with session_resolver.transcript_resolution_scope():
        outer = session_resolver._TRANSCRIPT_MEMO
        assert outer == {}
        session_resolver._resolve_transcript_via_happy_log(1)
        assert outer is not None and 1 in outer
        with session_resolver.transcript_resolution_scope():
            inner = session_resolver._TRANSCRIPT_MEMO
            assert inner is not outer
            assert inner == {}
        assert session_resolver._TRANSCRIPT_MEMO is outer
    assert session_resolver._TRANSCRIPT_MEMO is None
