"""Schema + I/O tests for the LLM session summarizer.

What this pins:

1. **Cache schema is enforced in ONE place** (``build_session_entry``).
   Future consumers (the dashboard) depend on the exact field names; a
   silent schema drift would break the dashboard's progress column without
   the summarizer ever erroring.
2. **Atomic write contract** — the summarizer writes the cache via temp+rename
   so a concurrent reader (the dashboard or `happy-ls`) never sees a partial
   file.
3. **Tail reader picks the newest entry timestamp from the read window** —
   this is the ``last_activity_ts`` field downstream consumers use to render
   "n minutes ago".

These tests deliberately avoid making real Anthropic calls; the network is
covered by the resolver/summarize CLI smoke check in the Report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import session_summarize  # noqa: E402

# ── schema ──────────────────────────────────────────────────────────────────


def test_build_session_entry_has_expected_keys(monkeypatch):
    # The dashboard's progress column reads each entry as a fixed dict; pin
    # the key set so a silent schema drift fails this test, not the dashboard.
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "running")
    entry = session_summarize.build_session_entry(
        sid="sess-abc",
        pid=1234,
        issue=492,
        cwd="/home/thomasjiralerspong/explore-persona-space",
        transcript="/x/t.jsonl",
        summary="doing the thing",
        summary_ts="2026-06-05T12:00:00Z",
        last_activity_ts="2026-06-05T11:59:00Z",
        error=None,
    )
    assert set(entry.keys()) == {
        "issue",
        "status",
        "dir",
        "live",
        "pid",
        "transcript",
        "summary",
        "summary_model",
        "summary_ts",
        "last_activity_ts",
        "error",
    }
    assert entry["live"] is True
    assert entry["summary_model"] == session_summarize.HAIKU_MODEL_ID


def test_build_session_entry_no_summary_clears_model(monkeypatch):
    # If we have no summary (errored or skipped), the ``summary_model`` field
    # MUST also be None — otherwise the dashboard would render "Haiku says:
    # <missing>" which is misleading.
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "blocked")
    entry = session_summarize.build_session_entry(
        sid="sess-abc",
        pid=1234,
        issue=42,
        cwd="/home/thomasjiralerspong/explore-persona-space",
        transcript=None,
        summary=None,
        summary_ts=None,
        last_activity_ts=None,
        error="transcript unresolvable",
    )
    assert entry["summary"] is None
    assert entry["summary_model"] is None
    assert entry["error"] == "transcript unresolvable"


def test_build_session_entry_dir_label_for_worktree(monkeypatch):
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "running")
    entry = session_summarize.build_session_entry(
        sid="sess-w",
        pid=1,
        issue=459,
        cwd="/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-459",
        transcript="/x.jsonl",
        summary="s",
        summary_ts="t",
        last_activity_ts=None,
        error=None,
    )
    # The compact dir label keeps the repo name + the worktree suffix so the
    # dashboard can render "EPS/issue-459" naturally.
    assert "explore-persona-space" in entry["dir"]
    assert "issue-459" in entry["dir"]


# ── atomic cache write ─────────────────────────────────────────────────────


def test_atomic_write_json_round_trip(tmp_path):
    dest = tmp_path / "sub" / "cache.json"
    session_summarize._atomic_write_json(dest, {"updated_at": "x", "sessions": {}})
    assert dest.is_file()
    assert json.loads(dest.read_text())["updated_at"] == "x"
    # No leftover .tmp file.
    assert not list(tmp_path.glob("**/*.tmp"))


def test_atomic_write_json_overwrites_existing(tmp_path):
    dest = tmp_path / "cache.json"
    dest.write_text(json.dumps({"updated_at": "first", "sessions": {}}))
    session_summarize._atomic_write_json(dest, {"updated_at": "second", "sessions": {"a": {}}})
    payload = json.loads(dest.read_text())
    assert payload["updated_at"] == "second"
    assert payload["sessions"] == {"a": {}}


# ── cache reader ───────────────────────────────────────────────────────────


def test_load_cache_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(session_summarize, "CACHE_PATH", tmp_path / "absent.json")
    assert session_summarize.load_cache() == {}


def test_load_cache_corrupt_returns_empty(monkeypatch, tmp_path):
    dest = tmp_path / "cache.json"
    dest.write_text("{ not valid json")
    monkeypatch.setattr(session_summarize, "CACHE_PATH", dest)
    # Best-effort enrichment: a corrupt cache must NOT raise; the table
    # falls back to the marker progress cell silently.
    assert session_summarize.load_cache() == {}


def test_get_cached_summary_returns_entry(monkeypatch, tmp_path):
    dest = tmp_path / "cache.json"
    dest.write_text(
        json.dumps(
            {
                "updated_at": "now",
                "sessions": {"sess-x": {"summary": "doing the thing", "issue": 1}},
            }
        )
    )
    monkeypatch.setattr(session_summarize, "CACHE_PATH", dest)
    entry = session_summarize.get_cached_summary("sess-x")
    assert entry is not None
    assert entry["summary"] == "doing the thing"
    # Missing session id returns None (not a fabricated default).
    assert session_summarize.get_cached_summary("sess-other") is None


# ── tail reader ────────────────────────────────────────────────────────────


def test_read_transcript_tail_picks_newest_timestamp(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "timestamp": "2026-06-05T10:00:00Z"}),
                json.dumps({"type": "assistant", "timestamp": "2026-06-05T10:01:00Z"}),
                json.dumps({"type": "system", "timestamp": "2026-06-05T10:00:30Z"}),
            ]
        )
    )
    tail, last_ts = session_summarize.read_transcript_tail(str(transcript))
    assert "2026-06-05T10:01:00Z" in tail
    assert last_ts == "2026-06-05T10:01:00Z"


def test_read_transcript_tail_handles_missing_timestamps(tmp_path):
    # Some transcript entries (e.g. ``last-prompt``, ``mode``) don't carry a
    # timestamp. They MUST be ignored when computing ``last_activity_ts`` —
    # a missing timestamp should not be propagated as "unknown" silently.
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "last-prompt", "leafUuid": "x"}),
                json.dumps({"type": "mode", "mode": "normal"}),
            ]
        )
    )
    tail, last_ts = session_summarize.read_transcript_tail(str(transcript))
    assert tail  # tail text is still returned
    assert last_ts is None  # no real entries -> None, not a placeholder string


def test_read_transcript_tail_truncates_to_char_cap(tmp_path):
    transcript = tmp_path / "t.jsonl"
    big_line = json.dumps({"type": "tool_result", "content": "x" * 100_000})
    transcript.write_text(big_line + "\n")
    tail, _ = session_summarize.read_transcript_tail(str(transcript), char_cap=1000)
    # Tail is bounded by the char cap (favoring the END of the file).
    assert len(tail) <= 1000


# ── resume-issue lookup ────────────────────────────────────────────────────


def test_resolve_session_for_issue_picks_live_when_multiple(tmp_path):
    import spawn_session

    # Same issue registered under both an autonomous and a manual entry.
    # The LIVE one wins regardless of spawned_at, because that's the
    # session the user actually wants to resume.
    (tmp_path / "issue-100.json").write_text(
        json.dumps({"happy_session_id": "auto-dead", "spawned_at": 200.0})
    )
    (tmp_path / "manual-issue-100.json").write_text(
        json.dumps({"happy_session_id": "manual-live", "spawned_at": 100.0})
    )
    out = spawn_session.resolve_session_for_issue(
        100, registry_dir=tmp_path, live_ids={"manual-live"}
    )
    assert out == "manual-live"


def test_resolve_session_for_issue_falls_back_to_newest_when_none_live(tmp_path):
    import spawn_session

    # No registered session is live -> pick the most recently spawned one
    # anyway, so the caller can still `happy resume <id>` it (the daemon
    # `/list` view is occasionally flaky).
    (tmp_path / "issue-200.json").write_text(
        json.dumps({"happy_session_id": "older", "spawned_at": 100.0})
    )
    (tmp_path / "manual-issue-200.json").write_text(
        json.dumps({"happy_session_id": "newer", "spawned_at": 200.0})
    )
    out = spawn_session.resolve_session_for_issue(200, registry_dir=tmp_path, live_ids=set())
    assert out == "newer"


def test_resolve_session_for_issue_returns_none_when_unregistered(tmp_path):
    import spawn_session

    # Issue not in registry at all -> None, NOT a fabricated id.
    out = spawn_session.resolve_session_for_issue(999, registry_dir=tmp_path, live_ids=set())
    assert out is None


def test_resolve_session_for_issue_skips_malformed(tmp_path):
    # A broken JSON file for the issue must not crash the lookup; the call
    # falls through as if the file weren't there.
    import spawn_session

    (tmp_path / "issue-500.json").write_text("{ not valid json")
    out = spawn_session.resolve_session_for_issue(500, registry_dir=tmp_path, live_ids=set())
    assert out is None


# ── idle-skip gate ─────────────────────────────────────────────────────────


def _run_async(coro_factory):
    """Tiny helper: build + run an async coroutine to completion."""
    import asyncio

    return asyncio.run(coro_factory())


def test_should_skip_llm_call_activity_unchanged_with_prior_summary():
    # Steady-state idle case: the prior entry has a usable summary AND the
    # transcript's newest entry has NOT advanced. Skip the LLM call and
    # reuse the cached summary (this is the ~5x cost reduction).
    prior = {
        "summary": "doing the thing",
        "last_activity_ts": "2026-06-05T10:00:00Z",
        "summary_ts": "2026-06-05T10:01:00Z",
        "summary_model": session_summarize.HAIKU_MODEL_ID,
    }
    assert session_summarize._should_skip_llm_call(prior, "2026-06-05T10:00:00Z") is True


def test_should_skip_llm_call_activity_advanced():
    # The transcript's newest entry is fresher than what we saw last tick —
    # something happened, MUST re-summarize.
    prior = {
        "summary": "old summary",
        "last_activity_ts": "2026-06-05T10:00:00Z",
        "summary_ts": "2026-06-05T10:01:00Z",
        "summary_model": session_summarize.HAIKU_MODEL_ID,
    }
    assert session_summarize._should_skip_llm_call(prior, "2026-06-05T10:05:00Z") is False


def test_should_skip_llm_call_no_prior_entry():
    # First time we see this session — no cache entry, MUST call Haiku.
    assert session_summarize._should_skip_llm_call(None, "2026-06-05T10:00:00Z") is False


def test_should_skip_llm_call_prior_summary_empty():
    # Prior entry exists but had no usable summary (the LLM call errored
    # last tick). Try again — don't reuse a hole.
    prior_none = {
        "summary": None,
        "last_activity_ts": "2026-06-05T10:00:00Z",
    }
    assert session_summarize._should_skip_llm_call(prior_none, "2026-06-05T10:00:00Z") is False
    prior_empty_str = {"summary": "", "last_activity_ts": "2026-06-05T10:00:00Z"}
    assert session_summarize._should_skip_llm_call(prior_empty_str, "2026-06-05T10:00:00Z") is False


def test_should_skip_llm_call_no_current_activity_ts():
    # The tail had no entries with a timestamp -> we can't tell if activity
    # advanced. Be conservative and re-summarize (a false-call is cheaper
    # than a stale summary).
    prior = {
        "summary": "x",
        "last_activity_ts": "2026-06-05T10:00:00Z",
    }
    assert session_summarize._should_skip_llm_call(prior, None) is False


def test_should_skip_llm_call_prior_has_no_activity_ts():
    # Prior entry was written before we recorded timestamps (or the tail
    # had none last time). Can't compare -> re-summarize.
    prior = {"summary": "x", "last_activity_ts": None}
    assert session_summarize._should_skip_llm_call(prior, "2026-06-05T10:00:00Z") is False


def test_produce_summary_skips_call_when_activity_unchanged(monkeypatch):
    # End-to-end gate: when activity is unchanged AND a prior summary
    # exists, ``_produce_summary`` MUST NOT call the LLM client AT ALL.
    # Sentinel: if Haiku WERE called, this counter would tick.
    call_count = {"n": 0}

    async def _no_call(*args, **kwargs):
        call_count["n"] += 1
        return "SHOULD NOT BE CALLED"

    monkeypatch.setattr(session_summarize, "_summarize_one", _no_call)

    prior = {
        "summary": "doing the thing",
        "last_activity_ts": "2026-06-05T10:00:00Z",
        "summary_ts": "2026-06-05T10:01:00Z",
        "summary_model": "claude-haiku-4-5-20251001",
    }

    async def _go():
        return await session_summarize._produce_summary(
            tail="some transcript content\nanother line",
            last_activity_ts="2026-06-05T10:00:00Z",  # identical to prior
            prior_entry=prior,
            issue=42,
            client=None,  # if Haiku were called, the None client would crash
            dry_run=False,
        )

    summary, summary_ts, summary_model, err = _run_async(_go)
    assert call_count["n"] == 0, "Haiku was called despite unchanged activity"
    assert summary == "doing the thing"
    assert summary_ts == "2026-06-05T10:01:00Z"
    assert summary_model == "claude-haiku-4-5-20251001"
    assert err is None


def test_produce_summary_calls_haiku_when_activity_advanced(monkeypatch):
    # When activity is fresher than the cache, ``_produce_summary`` MUST
    # call the LLM client and overwrite the prior summary.
    call_count = {"n": 0}

    async def _fake_call(client, issue, status, tail):
        call_count["n"] += 1
        return "fresh new summary"

    monkeypatch.setattr(session_summarize, "_summarize_one", _fake_call)
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "running")

    prior = {
        "summary": "OLD summary, must not be reused",
        "last_activity_ts": "2026-06-05T10:00:00Z",
        "summary_ts": "2026-06-05T10:01:00Z",
        "summary_model": "claude-haiku-4-5-20251001",
    }

    async def _go():
        return await session_summarize._produce_summary(
            tail="some transcript content",
            last_activity_ts="2026-06-05T11:00:00Z",  # advanced 1 h
            prior_entry=prior,
            issue=42,
            client="sentinel-client",
            dry_run=False,
        )

    summary, summary_ts, summary_model, err = _run_async(_go)
    assert call_count["n"] == 1, "Haiku was NOT called despite advanced activity"
    assert summary == "fresh new summary"
    assert summary_ts is not None
    assert summary_model == session_summarize.HAIKU_MODEL_ID
    assert err is None


def test_produce_summary_empty_tail_skips_call(monkeypatch):
    # Whitespace-only tail -> record an error, never call Haiku (would
    # otherwise burn a call on a meaningless prompt).
    call_count = {"n": 0}

    async def _no_call(*a, **kw):
        call_count["n"] += 1
        return ""

    monkeypatch.setattr(session_summarize, "_summarize_one", _no_call)

    async def _go():
        return await session_summarize._produce_summary(
            tail="   \n\n   \n",
            last_activity_ts=None,
            prior_entry=None,
            issue=1,
            client=None,
            dry_run=False,
        )

    summary, summary_ts, summary_model, err = _run_async(_go)
    assert call_count["n"] == 0
    assert summary is None
    assert summary_ts is None
    assert summary_model is None
    assert err == "transcript tail empty"
