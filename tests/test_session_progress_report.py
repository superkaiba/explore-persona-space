"""Pure-logic tests for the canonical per-session progress string + self-report.

What this pins:

1. **The canonical string format lives in exactly ONE place** — the
   ``build_progress_string`` helper. Both the /issue skill and the
   summarizer read this module; if anyone hardcodes the format anywhere
   else, the byte-identical-everywhere invariant breaks.

2. **Truncation rules.** Phone titles cap around 80 chars; we cap below
   that and trim the STEP (not the slug or issue) so the row stays
   findable. Slug pre-clipping respects the existing SKILL.md helper's
   45-char cap, so titles set via the helper and via the skill's
   pre-existing ``render_title`` produce the same slug for the same task.

3. **Self-report I/O contract.** The writer is atomic (temp+rename) and
   round-trips through the reader. A malformed file degrades to "no
   report" — the summarizer transparently falls back to Haiku, NEVER
   silently treats a stale or fabricated string as fresh.

4. **Freshness window.** A self-report within the window is "fresh"; an
   older one is not. The 20-minute default is sized to one full 10-min
   ``/loop`` tick of slack on the backstop cron — see
   ``DEFAULT_FRESHNESS_WINDOW_SECONDS`` for the rationale.

5. **Summarizer precedence.** A fresh self-report wins over the
   idle-skip path AND over the Haiku call, with ``source="self"`` in the
   returned tuple. The Haiku call is NOT made in that case (the sentinel
   counter would tick).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import session_progress_report  # noqa: E402
import session_summarize  # noqa: E402

# ── canonical string format ────────────────────────────────────────────────


def test_build_progress_string_basic_shape():
    # The canonical format leads with "#<N>", carries the slug, then a
    # Unicode middle-dot, then the step. This is the SINGLE definition.
    out = session_progress_report.build_progress_string(
        492, "wire /issue auto-title into session", "awaiting promotion"
    )
    assert out == "#492 wire /issue auto-title into session · awaiting promotion"


def test_build_progress_string_no_step_drops_separator():
    # A blank step means "title launched, no specific step yet" — drop the
    # separator entirely so the string is just "#<N> <slug>". Used by Step
    # 0 of the skill at first invocation, before any status is known.
    out = session_progress_report.build_progress_string(479, "anchor-knob sweep", "")
    assert out == "#479 anchor-knob sweep"


def test_build_progress_string_clips_long_slug():
    # Slugs longer than SLUG_MAX must be pre-clipped so the resulting
    # progress string fits within the phone-title display, and so the
    # /issue skill's existing render_title clipping policy (also 45 chars)
    # produces the SAME slug for the same task. If these drift, two parts
    # of the system display different shortened forms of the same title.
    long_slug = "x" * 100
    out = session_progress_report.build_progress_string(100, long_slug, "running")
    # The slug body in the output is clipped to SLUG_MAX exactly.
    assert "x" * session_progress_report.SLUG_MAX in out
    assert "x" * (session_progress_report.SLUG_MAX + 1) not in out


def test_build_progress_string_caps_total_length_by_trimming_step():
    # When a long step would push the string past PROGRESS_STRING_MAX,
    # we trim the STEP (not the head) so the issue number + slug stay
    # findable in the phone session list. The trimmed step ends in "…".
    out = session_progress_report.build_progress_string(77, "short slug", "x" * 500)
    assert len(out) <= session_progress_report.PROGRESS_STRING_MAX
    assert out.startswith("#77 short slug · ")
    assert out.endswith("…")


def test_build_progress_string_handles_whitespace_inputs():
    # Surrounding whitespace in slug / step is stripped — important
    # because the skill may pass a multi-line status sentence in by
    # accident; we want a single clean line on the phone.
    out = session_progress_report.build_progress_string(12, "  padded  ", "  whatever  ")
    assert out == "#12 padded · whatever"


# ── self-report writer + reader ────────────────────────────────────────────


def test_write_self_report_atomic_round_trip(tmp_path, monkeypatch):
    # The writer must (a) write atomically (no leftover .tmp file), and
    # (b) round-trip through the reader.
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    text, path = session_progress_report.write_self_report(492, slug="my slug", step="running")
    assert path == tmp_path / "492.json"
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    payload = json.loads(path.read_text())
    assert payload["issue"] == 492
    assert payload["slug"] == "my slug"
    assert payload["step"] == "running"
    assert payload["text"] == text == "#492 my slug · running"
    # ts is the canonical trailing-Z UTC shape used by session_summarize.
    assert payload["ts"].endswith("Z")


def test_read_self_report_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    assert session_progress_report.read_self_report(999) is None


def test_read_self_report_malformed_degrades_silently(tmp_path, monkeypatch):
    # A corrupt JSON file MUST degrade to "no report" (caller falls back
    # to Haiku). This is the safe direction — better a Haiku call than a
    # silent stale string driving the dashboard.
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    (tmp_path / "500.json").write_text("{ not valid json")
    assert session_progress_report.read_self_report(500) is None


def test_read_self_report_missing_required_fields_returns_none(tmp_path, monkeypatch):
    # An entry without `text` or `ts` is malformed — same degradation rule.
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    (tmp_path / "501.json").write_text(json.dumps({"issue": 501}))
    assert session_progress_report.read_self_report(501) is None
    (tmp_path / "502.json").write_text(
        json.dumps({"issue": 502, "text": "", "ts": "2026-06-05T10:00:00Z"})
    )
    # Empty `text` is treated as malformed — there is no useful string to
    # display.
    assert session_progress_report.read_self_report(502) is None


# ── freshness window ───────────────────────────────────────────────────────


def test_is_self_report_fresh_within_window():
    # Default window is 20 min; a report 5 min old is fresh.
    now = datetime.now(tz=UTC)
    ts = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert session_progress_report.is_self_report_fresh(
        {"text": "x", "ts": ts},
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def test_is_self_report_fresh_outside_window():
    # A report 30 min old is stale under the 20-min default.
    now = datetime.now(tz=UTC)
    ts = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert not session_progress_report.is_self_report_fresh(
        {"text": "x", "ts": ts},
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def test_is_self_report_fresh_custom_window():
    # The window is configurable so the summarizer can tune it without
    # editing this module (the default sits at 2x the loop interval).
    now = datetime.now(tz=UTC)
    ts = (now - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Window of 10s -> 30s-old is stale.
    assert not session_progress_report.is_self_report_fresh(
        {"text": "x", "ts": ts},
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        freshness_window_seconds=10.0,
    )
    # Window of 60s -> 30s-old is fresh.
    assert session_progress_report.is_self_report_fresh(
        {"text": "x", "ts": ts},
        now_iso=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        freshness_window_seconds=60.0,
    )


def test_is_self_report_fresh_malformed_ts_degrades_to_stale():
    # A garbled ts is treated as "not fresh" — same safe degradation rule:
    # we'd rather fall back to Haiku than fabricate freshness.
    assert not session_progress_report.is_self_report_fresh({"text": "x", "ts": "not-a-timestamp"})


# ── summarizer precedence (self wins over LLM) ─────────────────────────────


def _run_async(coro_factory):
    """Tiny helper: build + run an async coroutine to completion."""
    import asyncio

    return asyncio.run(coro_factory())


def test_produce_summary_uses_fresh_self_report_and_skips_llm(monkeypatch, tmp_path):
    # End-to-end: a fresh self-report exists for the issue -> the
    # summarizer MUST use it verbatim, tag source="self", and NOT call
    # Haiku. Sentinel: if Haiku were called, this counter would tick.
    call_count = {"n": 0}

    async def _no_call(*args, **kwargs):
        call_count["n"] += 1
        return "SHOULD NOT BE CALLED"

    monkeypatch.setattr(session_summarize, "_summarize_one", _no_call)
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    # Write a fresh self-report for issue 42.
    session_progress_report.write_self_report(42, slug="my task", step="running")

    async def _go():
        return await session_summarize._produce_summary(
            tail="transcript content",
            last_activity_ts="2026-06-05T11:00:00Z",
            prior_entry=None,  # no prior cache entry
            issue=42,
            client="sentinel-client",
            dry_run=False,
        )

    summary, summary_ts, summary_model, source, err = _run_async(_go)
    assert call_count["n"] == 0, "Haiku was called despite a fresh self-report"
    assert summary == "#42 my task · running"
    assert summary_ts is not None  # the cache tick stamp, not the report ts
    assert summary_model is None  # no model in the loop
    assert source == "self"
    assert err is None


def test_produce_summary_self_report_wins_over_idle_skip(monkeypatch, tmp_path):
    # Edge case: the prior cache entry would normally trigger the idle-skip
    # branch (transcript activity unchanged). A fresh self-report MUST still
    # win — otherwise a /issue tick that just landed wouldn't propagate
    # until the next transcript change.
    call_count = {"n": 0}

    async def _no_call(*args, **kwargs):
        call_count["n"] += 1
        return ""

    monkeypatch.setattr(session_summarize, "_summarize_one", _no_call)
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    session_progress_report.write_self_report(42, slug="my task", step="critic round 2")

    prior = {
        "summary": "STALE LLM SUMMARY",
        "last_activity_ts": "2026-06-05T10:00:00Z",
        "summary_ts": "2026-06-05T10:01:00Z",
        "summary_model": session_summarize.HAIKU_MODEL_ID,
        "source": "llm",
    }

    async def _go():
        return await session_summarize._produce_summary(
            tail="some tail",
            last_activity_ts="2026-06-05T10:00:00Z",  # unchanged -> idle-skip path
            prior_entry=prior,
            issue=42,
            client=None,
            dry_run=False,
        )

    summary, _summary_ts, summary_model, source, err = _run_async(_go)
    # The self-report wins; the stale LLM summary is NOT reused.
    assert summary == "#42 my task · critic round 2"
    assert summary_model is None
    assert source == "self"
    assert err is None
    assert call_count["n"] == 0


def test_produce_summary_stale_self_report_falls_back_to_llm(monkeypatch, tmp_path):
    # A self-report older than the freshness window MUST NOT be used; the
    # summarizer falls through to the LLM call with source="llm". This is
    # the safety valve when /issue dies mid-run and the self-report ages.
    call_count = {"n": 0}

    async def _fake_call(client, issue, status, tail):
        call_count["n"] += 1
        return "fresh LLM summary"

    monkeypatch.setattr(session_summarize, "_summarize_one", _fake_call)
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "running")
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    # Hand-write a stale self-report (timestamp far in the past).
    (tmp_path / "42.json").write_text(
        json.dumps(
            {
                "issue": 42,
                "slug": "old",
                "step": "very old",
                "text": "#42 old · very old",
                "ts": "2020-01-01T00:00:00Z",
            }
        )
    )

    async def _go():
        return await session_summarize._produce_summary(
            tail="content",
            last_activity_ts="2026-06-05T11:00:00Z",
            prior_entry=None,
            issue=42,
            client="sentinel-client",
            dry_run=False,
        )

    summary, _summary_ts, summary_model, source, err = _run_async(_go)
    assert call_count["n"] == 1, "Haiku was NOT called despite stale self-report"
    assert summary == "fresh LLM summary"
    assert summary_model == session_summarize.HAIKU_MODEL_ID
    assert source == "llm"
    assert err is None


def test_build_session_entry_source_self_clears_model():
    # Cache schema invariant: when source="self", summary_model is None
    # (no model in the loop). The dashboard should never render
    # "claude-haiku says: <self-reported string>" — that would be lying
    # about provenance.
    entry = session_summarize.build_session_entry(
        sid="sess-1",
        pid=1,
        issue=42,
        cwd="/home/me/explore-persona-space",
        transcript="/x.jsonl",
        summary="#42 my task · running",
        summary_ts="2026-06-05T12:00:00Z",
        last_activity_ts="2026-06-05T11:59:00Z",
        error=None,
        source="self",
    )
    assert entry["summary"] == "#42 my task · running"
    assert entry["summary_model"] is None
    assert entry["source"] == "self"


def test_build_session_entry_explicit_llm_source_respects_model_override():
    # The idle-skip path passes the prior model id forward AND source="llm";
    # the entry must carry the explicit model (not the current HAIKU_MODEL_ID
    # constant, which could change in code).
    entry = session_summarize.build_session_entry(
        sid="sess-2",
        pid=1,
        issue=42,
        cwd="/home/me/explore-persona-space",
        transcript="/x.jsonl",
        summary="reused",
        summary_ts="ts",
        last_activity_ts=None,
        error=None,
        source="llm",
        summary_model="claude-haiku-legacy-model-id",
    )
    assert entry["summary_model"] == "claude-haiku-legacy-model-id"
    assert entry["source"] == "llm"


# ── _summarize_session: self-report wins independent of transcript ─────────


def _run_async(coro_factory_or_coro):
    """Reusable async runner (duplicate of the earlier helper; kept module-
    local so each test block is readable without scrolling)."""
    import asyncio

    return asyncio.run(coro_factory_or_coro())


def _make_rr(*, issue, transcript, cwd="/home/me/explore-persona-space"):
    """Build a session_resolver.ResolveResult-lookalike for the summarizer.

    The resolver dataclass has fields (issue, cwd, transcript, reason);
    we only need those four for `_summarize_session`. Using SimpleNamespace
    avoids pulling the real resolver into the unit test (it would touch
    the live Happy daemon)."""
    from types import SimpleNamespace

    return SimpleNamespace(issue=issue, cwd=cwd, transcript=transcript, reason=None)


def test_summarize_session_self_report_wins_when_transcript_read_fails(monkeypatch, tmp_path):
    """Regression pin for the bug fix: a fresh self-report MUST reach the
    cache even when `read_transcript_tail` raises (rotation race, NFS
    hiccup, permission flap). Pre-fix, the OSError left `tail=None` and
    the self-report check inside `_produce_summary` was never reached —
    the self-report was silently dropped."""
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "running")

    # Fresh self-report exists for issue 42.
    session_progress_report.write_self_report(42, slug="my task", step="running")

    # Make the transcript read explode so the pre-fix code path would
    # have left `tail=None` and never reached the self-report check.
    def _explode(*args, **kwargs):
        raise OSError("simulated transcript-rotation race")

    monkeypatch.setattr(session_summarize, "read_transcript_tail", _explode)

    # Sentinel: ensure the LLM client is not invoked at all.
    haiku_calls = {"n": 0}

    async def _no_call(*args, **kwargs):
        haiku_calls["n"] += 1
        return "SHOULD NOT BE CALLED"

    monkeypatch.setattr(session_summarize, "_summarize_one", _no_call)

    payload: dict = {"sessions": {}}
    rr = _make_rr(issue=42, transcript="/fake/transcript.jsonl")

    async def _go():
        return await session_summarize._summarize_session(
            payload=payload,
            sid="sess-abc",
            pid=1,
            rr=rr,
            prior_entry=None,
            client=None,
            dry_run=False,
        )

    _run_async(_go)
    entry = payload["sessions"]["sess-abc"]
    # Self-report's `text` propagated all the way to the cache entry.
    assert entry["summary"] == "#42 my task · running"
    assert entry["source"] == "self"
    assert entry["summary_model"] is None
    # No LLM call was made — the fresh self-report short-circuited the path.
    assert haiku_calls["n"] == 0
    # No error: the transcript-read failure was bypassed entirely.
    assert entry["error"] is None


def test_summarize_session_stale_self_report_falls_through_to_transcript(monkeypatch, tmp_path):
    """The complement: a STALE self-report must NOT shortcut the
    transcript path — the summarizer should fall through to the
    transcript-tail logic so a dead /issue session doesn't keep its old
    string forever."""
    import json as _json

    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    monkeypatch.setattr(session_summarize, "_get_task_status", lambda issue: "running")

    # Hand-write a stale self-report (timestamp far in the past).
    (tmp_path / "42.json").write_text(
        _json.dumps(
            {
                "issue": 42,
                "slug": "stale slug",
                "step": "very old",
                "text": "#42 stale slug · very old",
                "ts": "2020-01-01T00:00:00Z",
            }
        )
    )

    # Transcript read succeeds; the LLM produces a fresh string.
    monkeypatch.setattr(
        session_summarize,
        "read_transcript_tail",
        lambda *a, **kw: ("transcript content", "2026-06-05T11:00:00Z"),
    )
    haiku_calls = {"n": 0}

    async def _fake_call(client, issue, status, tail):
        haiku_calls["n"] += 1
        return "fresh LLM summary"

    monkeypatch.setattr(session_summarize, "_summarize_one", _fake_call)

    payload: dict = {"sessions": {}}
    rr = _make_rr(issue=42, transcript="/fake/transcript.jsonl")

    async def _go():
        return await session_summarize._summarize_session(
            payload=payload,
            sid="sess-stale",
            pid=1,
            rr=rr,
            prior_entry=None,
            client="sentinel",
            dry_run=False,
        )

    _run_async(_go)
    entry = payload["sessions"]["sess-stale"]
    assert entry["summary"] == "fresh LLM summary"
    assert entry["source"] == "llm"
    assert haiku_calls["n"] == 1


def test_write_self_report_unknown_issue_raises_even_with_explicit_slug(monkeypatch, tmp_path):
    """Fail-loud invariant: a typo'd `--issue N` MUST raise even when an
    explicit `--slug` is supplied. Pre-fix, the explicit-slug code path
    short-circuited the task-existence check and silently wrote a
    self-report file for a non-existent task."""
    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)

    # Patch the lazy-imported `get_task` to act like an unknown issue.
    import explore_persona_space.task_workflow as tw

    def _unknown(_issue):
        raise FileNotFoundError("simulated unknown issue")

    monkeypatch.setattr(tw, "get_task", _unknown)

    # With slug=None — should raise (as before).
    import pytest

    with pytest.raises(FileNotFoundError):
        session_progress_report.write_self_report(9999, slug=None, step="x")

    # With an explicit slug — MUST still raise (the fix).
    with pytest.raises(FileNotFoundError):
        session_progress_report.write_self_report(9999, slug="hand-typed", step="x")

    # And nothing was written to disk.
    assert not list(tmp_path.glob("*.json"))


# ── progress-bar / ETA title suffix (task #587) ────────────────────────────
#
# What these pin: `build_progress_string(..., suffix=...)` and the
# `write_self_report` ETA hook. The suffix=None path must stay
# BYTE-IDENTICAL to the historical output (the tests above prove the
# historical output; the tests here prove suffix=None routes through it),
# the degrade ladder must trim STEP before bar before suffix, and the
# write_self_report hook must fire ONLY for machine-active statuses,
# fail-soft, and never rebuild stats from the title path.

SUFFIX = "▓▓░░░ 43% ~4–9h"  # noqa: RUF001 — en-dash is the pinned band format


def test_suffix_appended_after_step():
    out = session_progress_report.build_progress_string(587, "progress bar", "running", SUFFIX)
    assert out == f"#587 progress bar · running · {SUFFIX}"


def test_suffix_none_is_byte_identical_to_historical_path():
    for slug, step in [
        ("slug", "running"),
        ("  padded  ", "  whatever  "),
        ("s", ""),
        ("x" * 100, "y" * 500),
    ]:
        legacy = session_progress_report.build_progress_string(42, slug, step)
        assert session_progress_report.build_progress_string(42, slug, step, None) == legacy
        assert session_progress_report.build_progress_string(42, slug, step, suffix=None) == legacy


def test_suffix_ladder_trims_step_before_suffix():
    out = session_progress_report.build_progress_string(77, "short slug", "x" * 200, SUFFIX)
    assert len(out) <= session_progress_report.PROGRESS_STRING_MAX
    assert out.startswith("#77 short slug · x")
    assert out.endswith(f"… · {SUFFIX}")  # suffix kept whole, step trimmed


def test_suffix_ladder_drops_bar_chars_then_suffix():
    # With a 45-char slug, a long days-band suffix can't fit whole even with
    # a 1-char step, so the block-bar chars are dropped FIRST; the pct + band
    # survive (bar dropped before pct, suffix dropped last).
    slug = "s" * session_progress_report.SLUG_MAX
    long_suffix = "▓▓▓▓░ 87% ≈10.5–12.5d"  # noqa: RUF001
    out = session_progress_report.build_progress_string(587, slug, "interpreting", long_suffix)
    assert len(out) <= session_progress_report.PROGRESS_STRING_MAX
    assert "87% ≈10.5–12.5d" in out  # noqa: RUF001
    assert "▓" not in out
    # When even the bar-less suffix can't fit, the suffix is dropped entirely
    # and the output equals the historical no-suffix string.
    long_step = "z" * 200
    out2 = session_progress_report.build_progress_string(587, slug, long_step, "▓" * 70)
    assert out2 == session_progress_report.build_progress_string(587, slug, long_step)


def test_suffix_with_empty_step_joins_head_and_suffix():
    out = session_progress_report.build_progress_string(587, "slug", "", SUFFIX)
    assert out == f"#587 slug · {SUFFIX}"


def test_issue_tick_composed_step_stays_legible_with_suffix():
    # Regression pin for the GPU-idle advisory title path (now owned by the
    # full /issue skill's poll loop; the guarded-no-op /issue-tick no longer
    # composes titles), which composes its step text through the same helper.
    step = "running · GPU idle 42m — check pod"
    out = session_progress_report.build_progress_string(587, "marker leakage sweep", step, SUFFIX)
    assert len(out) <= session_progress_report.PROGRESS_STRING_MAX
    assert out.startswith("#587 marker leakage sweep · running")
    # The suffix survives (whole or bar-less) — the title stays informative.
    assert "43%" in out


# ── write_self_report ETA hook ─────────────────────────────────────────────


def _mock_task_status(monkeypatch, status):
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw,
        "get_task",
        lambda issue: {
            "id": issue,
            "status": status,
            "frontmatter": {"title": "my slug", "kind": "experiment"},
            "body": "",
        },
    )


def test_write_self_report_appends_suffix_for_active_status(monkeypatch, tmp_path):
    import explore_persona_space.task_progress as tp

    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    _mock_task_status(monkeypatch, "running")
    monkeypatch.setattr(tp, "load_stats_readonly", lambda: {"stub": True})
    monkeypatch.setattr(tp, "estimate_task_progress", lambda issue, stats, now=None: {"r": 1})
    monkeypatch.setattr(tp, "format_title_suffix", lambda row, now=None: SUFFIX)
    text, _ = session_progress_report.write_self_report(42, step="running")
    assert text == f"#42 my slug · running · {SUFFIX}"


def test_write_self_report_byte_identical_for_every_parked_status(monkeypatch, tmp_path):
    # Keyed on the REAL status enum (no phantom statuses): every gate-park /
    # terminal status produces the historical string, and the estimator is
    # never touched (sentinel raises if it were).
    import explore_persona_space.task_progress as tp

    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)

    def _boom():
        raise AssertionError("estimator must not be touched for parked statuses")

    monkeypatch.setattr(tp, "load_stats_readonly", _boom)
    for status in [
        "proposed",
        "plan_pending",
        "blocked",
        "awaiting_promotion",
        "followups_running",
        "completed",
        "archived",
        "weird_future_status",
    ]:
        _mock_task_status(monkeypatch, status)
        text, _ = session_progress_report.write_self_report(42, step="parked")
        assert text == "#42 my slug · parked", status


def test_write_self_report_eta_failsoft_on_estimator_error(monkeypatch, tmp_path, capsys):
    import explore_persona_space.task_progress as tp

    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    _mock_task_status(monkeypatch, "running")

    def _explode():
        raise RuntimeError("simulated estimator failure")

    monkeypatch.setattr(tp, "load_stats_readonly", _explode)
    text, _ = session_progress_report.write_self_report(42, step="running")
    assert text == "#42 my slug · running"  # title never breaks
    assert "ETA suffix skipped" in capsys.readouterr().err


def test_write_self_report_missing_stats_means_no_suffix_and_no_rebuild(monkeypatch, tmp_path):
    # Dead-cron path: load_stats_readonly()->None must yield the historical
    # string WITHOUT any stats rebuild or snapshot write from the title path.
    import explore_persona_space.task_progress as tp

    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    _mock_task_status(monkeypatch, "running")
    monkeypatch.setattr(tp, "load_stats_readonly", lambda: None)

    def _boom(*a, **kw):
        raise AssertionError("title path must never rebuild stats / write the snapshot")

    monkeypatch.setattr(tp, "build_stage_stats", _boom)
    monkeypatch.setattr(tp, "write_snapshot", _boom)
    snap = tmp_path / "snap.json"
    monkeypatch.setattr(tp, "SNAPSHOT_PATH", snap)
    text, _ = session_progress_report.write_self_report(42, step="running")
    assert text == "#42 my slug · running"
    assert not snap.exists()


def test_write_self_report_no_eta_flag_skips_estimator(monkeypatch, tmp_path):
    import explore_persona_space.task_progress as tp

    monkeypatch.setattr(session_progress_report, "SELF_REPORT_DIR", tmp_path)
    _mock_task_status(monkeypatch, "running")

    def _boom():
        raise AssertionError("eta=False must skip the estimator entirely")

    monkeypatch.setattr(tp, "load_stats_readonly", _boom)
    text, _ = session_progress_report.write_self_report(42, step="running", eta=False)
    assert text == "#42 my slug · running"
