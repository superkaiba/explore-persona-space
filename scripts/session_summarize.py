"""One-pass LLM summarizer for live EPS Happy sessions.

For each Happy session whose cwd resolves to ``explore-persona-space`` (or one
of its worktrees) AND has a resolvable Claude Code transcript, read the
transcript TAIL (last ~120 lines / ~25 entries, capped on input tokens),
call ``claude-haiku-4-5-20251001`` with a tight prompt, and write a
shared-cache entry the EPS dashboard + ``spawn_session.py list`` reads.

Schema (atomic temp+rename to ``~/.eps-autonomous/session_progress.json``)::

    {
      "updated_at": "<ISO8601 UTC>",
      "sessions": {
        "<happy_session_id>": {
          "issue": 492,
          "status": "planning",
          "dir": "explore-persona-space",
          "live": true,
          "pid": 1637665,
          "transcript": "/.../<uuid>.jsonl",
          "summary": "Running the Phase-1.5 fact-check on plan v2; ...",
          "summary_model": "claude-haiku-4-5-20251001",
          "summary_ts": "<ISO8601 UTC>",
          "source": "self" | "llm" | null,
          "last_activity_ts": "<ISO8601 UTC of newest transcript entry>",
          "error": null
        },
        ...
      }
    }

Design choices:

- **Per-session try/except** with a VISIBLE ``error`` field — one bad session
  must not abort the run (CLAUDE.md fail-fast: surface, don't swallow).
- **Self-report wins.** If the ``/issue`` skill has written a fresh
  ``~/.eps-autonomous/issue-progress/<N>.json`` for this session's issue,
  reuse its ``text`` verbatim as the ``summary`` (``source="self"``) and
  skip the Haiku call entirely. The ``/issue`` skill writes the same
  canonical string it passes to ``mcp__happy__change_title``, so the
  phone title and the dashboard's progress column are byte-identical for
  /issue sessions. Sessions WITHOUT a fresh self-report (interactive, non-
  /issue, or stale) fall back to the LLM path (``source="llm"``).
- **Tail-only input** keeps per-call cost cheap (Haiku is the cheapest tier;
  ~25 entries truncated to roughly the last 30k chars of raw text).
- **Live + EPS only** — non-EPS sessions (my-goat, introsp) and dead sessions
  are skipped entirely.
- **Reuses ``AnthropicChatModel``** from ``explore_persona_space.llm``
  (CLAUDE.md: search before building; never hand-roll a new client).

Side artifact: after the cache write, a human-/dashboard-readable digest is
rendered from the SAME payload to ``.claude/cache/sessions-digest.md`` (one
line per live session: issue, status, one-line summary, anomaly flag). No
second cron, no push calls — the watcher stays the sole Telegram push source.
See :func:`render_sessions_digest` / :func:`write_sessions_digest`.

CLI::

    uv run python scripts/session_summarize.py            # one pass
    uv run python scripts/session_summarize.py --dry-run  # don't call API, don't write cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import session_progress_report  # noqa: E402
import session_resolver  # noqa: E402

CACHE_PATH = Path.home() / ".eps-autonomous" / "session_progress.json"

HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"

# Tail bound: how many lines of transcript JSONL we read from the end. Each
# line is one entry (user / assistant / system / attachment / tool_result).
# 120 lines comfortably covers a 25-entry conversational tail even when each
# turn spans multiple lines (tool-use blocks, large outputs).
_TAIL_LINES = 120

# Hard cap on raw characters of tail text fed to the LLM. Haiku's input is
# cheap but pacing input tokens is still a CLAUDE.md rule (429 token-pacing),
# and very long single tool outputs can dominate a tail otherwise.
_TAIL_CHAR_CAP = 30_000

# Per-call output cap. The prompt asks for 1-2 sentences; budget a little
# headroom but never more than that.
_MAX_OUTPUT_TOKENS = 220

# Per-call temperature. Lower = more consistent phrasing across runs, which
# matters when the cache is being polled and we want frame-to-frame stability
# instead of cosmetic churn.
_TEMPERATURE = 0.2

# Concurrency. Haiku is fast, but the dashboard reads from a single cache
# write per tick — we hold the spawn until all sessions are summarized so
# we never publish a half-written cache. The semaphore bounds total
# in-flight calls; safe well below the org-wide rate limit.
_CONCURRENCY = 8

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
This is the recent transcript tail of a Claude Code session driving issue #{issue}.

Issue current status (from the task workflow): {status}

In 1-2 sentences, plain English, say what it is doing RIGHT NOW (current
phase / step / what it's waiting on). No preamble, no "the session is".
Lead with the verb.

Transcript tail follows. Lines are JSON; treat tool calls and tool outputs
as actions the session took, and user/assistant messages as conversation.

```
{tail}
```
"""


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── tail extraction ────────────────────────────────────────────────────────


def read_transcript_tail(
    transcript_path: str,
    *,
    tail_lines: int = _TAIL_LINES,
    char_cap: int = _TAIL_CHAR_CAP,
) -> tuple[str, str | None]:
    """Return (tail_text, last_activity_ts_iso_or_None) for one transcript file.

    Streams the file into a bounded ``deque`` so memory is O(tail_lines), not
    O(file). Truncates the resulting text to the last ``char_cap`` characters
    (favoring the END of the file — the most recent content). Also scans
    those lines for the newest entry timestamp and returns it. On a read
    failure raises OSError; the caller has the per-session try/except.

    The deque approach matters: live transcripts routinely grow to 10+ MB
    on long-lived sessions, and the previous ``fh.readlines()[-tail_lines:]``
    re-allocated the whole file in memory on every 5-min cron tick."""
    with open(transcript_path) as fh:
        lines = list(deque(fh, maxlen=tail_lines))
    text = "".join(lines)
    if len(text) > char_cap:
        text = text[-char_cap:]
    # Find newest timestamp from those lines.
    newest_ts: str | None = None
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = entry.get("timestamp")
        if isinstance(ts, str) and ts and (newest_ts is None or ts > newest_ts):
            newest_ts = ts
    return text, newest_ts


def _dir_label_for_cache(cwd: str | None) -> str:
    """Compact dir label for the cache (just the repo-or-worktree name)."""
    if not cwd:
        return "?"
    p = Path(cwd)
    # ``.claude/worktrees/<name>`` => ``explore-persona-space/<name>``
    parts = p.parts
    if "explore-persona-space" in parts:
        idx = parts.index("explore-persona-space")
        rel = "/".join(parts[idx:])
        return rel
    return p.name


# ── status lookup ──────────────────────────────────────────────────────────


def _get_task_status(issue: int) -> str:
    """Return the task's current status (or '?' on lookup failure)."""
    try:
        from explore_persona_space.task_workflow import get_task

        task = get_task(issue)
        return str(task.get("status", "?"))
    except FileNotFoundError:
        return "not-found"
    except Exception as e:
        return f"<lookup-failed: {type(e).__name__}>"


# ── LLM call ────────────────────────────────────────────────────────────────


async def _summarize_one(
    client,
    issue: int,
    status: str,
    tail_text: str,
) -> str:
    """Call Haiku to summarize one session's tail. Returns the completion text.

    Raises any underlying client error so the caller's per-session try/except
    records it in the entry's ``error`` field instead of suppressing it."""
    from explore_persona_space.llm.models import (
        ChatMessage,
        MessageRole,
        Prompt,
    )

    prompt_text = _PROMPT_TEMPLATE.format(issue=issue, status=status, tail=tail_text)
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
    responses = await client(
        model_id=HAIKU_MODEL_ID,
        prompt=prompt,
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=_TEMPERATURE,
    )
    completion = (responses[0].completion or "").strip() if responses else ""
    return completion


# ── orchestration ──────────────────────────────────────────────────────────


def _ensure_env_loaded() -> None:
    """Load .env so ANTHROPIC_API_KEY is in os.environ before constructing
    AnthropicChatModel. Idempotent."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from explore_persona_space.orchestrate.env import load_dotenv

        load_dotenv()
    except ImportError:
        # Fallback — bare dotenv. The .env in the repo root has the key.
        try:
            from dotenv import load_dotenv as _dl

            _dl(SCRIPTS_DIR.parent / ".env")
        except ImportError:
            pass


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically via temp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False))
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via temp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ── sessions digest (.claude/cache/sessions-digest.md) ──────────────────────
#
# A human-/dashboard-readable one-line-per-session snapshot written from the
# SAME payload the session_progress.json cache is built from, refreshed on the
# existing 5-min summarizer cron (NO second cron). The watcher remains the sole
# Telegram push source — this only writes a file (session visibility layer,
# EPS workflow v2 §2).

# Relative path of the digest under the primary checkout. `.claude/cache/` is
# a git-ignored cache dir (same neighbourhood as issue-<N>-handle.json).
DIGEST_REL_PATH = ".claude/cache/sessions-digest.md"

# Max chars for a summary cell before it is truncated (keeps the markdown
# table readable in a phone-width viewer).
_DIGEST_SUMMARY_MAX = 100

_DIGEST_TITLE = "EPS sessions digest"


def digest_path() -> Path:
    """Absolute path of the sessions digest.

    Resolved from this script's location (``SCRIPTS_DIR.parent``): the 5-min
    cron ``cd``s into the primary checkout before invoking, so that parent IS
    the repo root. `.claude/cache/` is git-ignored, so the digest is a cache
    artifact, never a committed file."""
    return SCRIPTS_DIR.parent / DIGEST_REL_PATH


def _session_anomaly(entry: dict) -> str | None:
    """Short anomaly flag for one session entry, or None when healthy.

    Surfaces the two conditions a human scanning the digest wants immediately:
    a per-session summarize ``error`` (transcript unresolvable, tail-read
    failure, API error — reported as ``summarize-error (<class>)``), or a task
    ``status`` of ``blocked`` / ``not-found``. Pure; defensive ``.get()`` so a
    partial entry never raises."""
    err = entry.get("error")
    if isinstance(err, str) and err.strip():
        head = err.split(":", 1)[0].strip()
        return f"summarize-error ({head})" if head else "summarize-error"
    status = entry.get("status")
    if status in ("blocked", "not-found"):
        return str(status)
    return None


def _digest_cell(text: object) -> str:
    """Collapse whitespace, escape ``|`` for the markdown table, and truncate
    to :data:`_DIGEST_SUMMARY_MAX`. Non-str inputs are coerced defensively."""
    collapsed = " ".join(str(text).split()).replace("|", r"\|")
    if len(collapsed) > _DIGEST_SUMMARY_MAX:
        collapsed = collapsed[: _DIGEST_SUMMARY_MAX - 1].rstrip() + "…"
    return collapsed


def render_sessions_digest(payload: dict, *, now_iso: str | None = None) -> str:
    """Render the sessions-digest markdown from a session_progress ``payload``.

    One row per live EPS session: issue, task status, one-line summary, and an
    anomaly flag when present (:func:`_session_anomaly`). PURE (no I/O) — tests
    pin the exact format here. Header carries the payload's ``updated_at`` (or
    the ``now_iso`` override) plus a live-session count. Rows sort by issue
    number (mapped ascending first, unmapped last), then by session id, for a
    stable frame-to-frame ordering. An empty payload renders a valid (0-row)
    table so the artifact always exists for the dashboard/reader."""
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(sessions, dict):
        sessions = {}
    updated = now_iso or (payload.get("updated_at") if isinstance(payload, dict) else None)
    updated = updated or _utcnow_iso()

    def _sort_key(item: tuple[str, object]) -> tuple[int, int, str]:
        sid, entry = item
        issue = entry.get("issue") if isinstance(entry, dict) else None
        if isinstance(issue, int) and not isinstance(issue, bool):
            return (0, issue, sid)
        return (1, 0, sid)

    rows = sorted(sessions.items(), key=_sort_key)
    lines = [
        f"# {_DIGEST_TITLE}",
        "",
        f"_Updated: {updated} — {len(rows)} live session(s)._",
        "",
        "| Issue | Status | Summary | Flag |",
        "| --- | --- | --- | --- |",
    ]
    for _sid, entry in rows:
        entry = entry if isinstance(entry, dict) else {}
        issue = entry.get("issue")
        issue_cell = f"#{issue}" if isinstance(issue, int) and not isinstance(issue, bool) else "-"
        status = _digest_cell(entry.get("status") or "?")
        summary = _digest_cell(entry.get("summary") or entry.get("error") or "(no summary)")
        flag = _digest_cell(_session_anomaly(entry) or "")
        lines.append(f"| {issue_cell} | {status} | {summary} | {flag} |")
    lines.append("")
    return "\n".join(lines)


def write_sessions_digest(payload: dict, path: Path | None = None) -> Path:
    """Render + atomically write the sessions digest. Returns the path written.

    ``path`` defaults to :func:`digest_path`; tests pass a ``tmp_path``. Writes
    via temp+rename (:func:`_atomic_write_text`) so a concurrent reader (the
    dashboard) never sees a partial file."""
    dest = path if path is not None else digest_path()
    _atomic_write_text(dest, render_sessions_digest(payload))
    return dest


def build_session_entry(
    sid: str,
    pid: int,
    issue: int | None,
    cwd: str | None,
    transcript: str | None,
    summary: str | None,
    summary_ts: str | None,
    last_activity_ts: str | None,
    error: str | None,
    summary_model: str | None = None,
    source: str | None = None,
) -> dict[str, object]:
    """Construct the cache entry for one session. Pure — no I/O.

    Captured as a named function so the schema is enforced in ONE place
    (tests pin the keys + types here so a silent schema drift is caught).

    ``summary_model`` defaults to :data:`HAIKU_MODEL_ID` when ``summary`` is
    non-None, ``source`` is ``"llm"`` (or omitted), and no override is
    passed. The idle-skip path passes through the model id from the prior
    cache entry so a reused summary truthfully reports which model produced
    it (even if ``HAIKU_MODEL_ID`` later changes in code).

    ``source`` is ``"self"`` when the summary came from the /issue skill's
    self-report file (``~/.eps-autonomous/issue-progress/<N>.json``),
    ``"llm"`` when it came from a fresh Haiku call (or a prior-tick LLM
    summary reused via the idle-skip path), and ``None`` when there is no
    summary at all. When ``source == "self"`` the ``summary_model`` field
    is ``None`` — there was no model in the loop.
    """
    status = _get_task_status(issue) if issue is not None else None
    if summary is None:
        model_field: str | None = None
        source_field: str | None = None
    elif source == "self":
        model_field = None
        source_field = "self"
    else:
        # source is None or "llm" — both treated as LLM-produced (None is
        # the legacy callsite that predates `source`). The model field
        # respects an explicit override (idle-skip carries the prior tick's
        # model id), else defaults to the current HAIKU_MODEL_ID.
        model_field = summary_model if summary_model is not None else HAIKU_MODEL_ID
        source_field = "llm"
    return {
        "issue": issue,
        "status": status,
        "dir": _dir_label_for_cache(cwd),
        "live": True,
        "pid": pid,
        "transcript": transcript,
        "summary": summary,
        "summary_model": model_field,
        "summary_ts": summary_ts,
        "source": source_field,
        "last_activity_ts": last_activity_ts,
        "error": error,
    }


def _should_skip_llm_call(
    prior_entry: dict | None,
    last_activity_ts: str | None,
) -> bool:
    """Idle-skip gate: True iff the prior cache entry has a usable summary
    AND the transcript's newest entry has not advanced since.

    Activity advancing => the session did something new => re-summarize.
    Activity unchanged + prior summary present => reuse the cached summary
    (this is the ~80% cost reduction; an idle EPS session at ``planning``
    or ``awaiting_promotion`` parks for hours)."""
    if not isinstance(prior_entry, dict):
        return False
    prior_summary = prior_entry.get("summary")
    prior_ts = prior_entry.get("last_activity_ts")
    if not isinstance(prior_summary, str) or not prior_summary:
        return False
    if not isinstance(prior_ts, str) or not prior_ts:
        return False
    if last_activity_ts is None:
        # We have no way to tell whether activity advanced — be conservative
        # and DO re-summarize (cheaper false-call > stale-summary).
        return False
    return last_activity_ts == prior_ts


async def _summarize_session(
    *,
    payload: dict,
    sid: str,
    pid: int,
    rr: session_resolver.ResolveResult,
    prior_entry: dict | None,
    client,
    dry_run: bool,
) -> None:
    """Compute one session's cache entry and write it into ``payload``.

    Extracted from ``_run_pass`` so the orchestrator stays under ruff's
    cyclomatic-complexity cap. All per-session error paths land here:
    transcript unresolvable, tail-read OSError, empty tail, no /issue
    prompt, summarize-call exception — each sets a VISIBLE ``error`` field
    rather than failing silently.

    **Self-report wins independent of transcript readability.** Before any
    transcript I/O, if a fresh self-report exists for ``rr.issue`` we use
    it verbatim and skip the rest. Otherwise an OSError reading the
    transcript (rotation race, NFS hiccup, permission flap) would drop the
    fresh /issue-written self-report on the floor, which is exactly the
    failure mode the self-report path is supposed to insulate the
    dashboard against."""
    entry_error: str | None = None
    summary: str | None = None
    summary_ts: str | None = None
    summary_model: str | None = None
    source: str | None = None
    last_activity_ts: str | None = None
    tail: str | None = None
    try:
        # 1. Fresh self-report wins FIRST — before any transcript I/O. A
        #    /issue session that wrote one within the freshness window
        #    must reach the dashboard regardless of transcript read
        #    failures (transcript file rotation, NFS hiccup, etc.).
        if rr.issue is not None:
            report = session_progress_report.read_self_report(rr.issue)
            if report is not None and session_progress_report.is_self_report_fresh(report):
                text = report.get("text")
                if isinstance(text, str) and text:
                    summary = text
                    summary_ts = _utcnow_iso()
                    summary_model = None
                    source = "self"
                    # last_activity_ts left None — we did NOT read the
                    # transcript, so we cannot claim a fresher
                    # last_activity_ts than the prior cache entry. The
                    # dashboard's "summary age" column still renders
                    # correctly off summary_ts.
                    payload["sessions"][sid] = build_session_entry(
                        sid=sid,
                        pid=pid,
                        issue=rr.issue,
                        cwd=rr.cwd,
                        transcript=rr.transcript,
                        summary=summary,
                        summary_ts=summary_ts,
                        last_activity_ts=last_activity_ts,
                        error=entry_error,
                        summary_model=summary_model,
                        source=source,
                    )
                    return

        # 2. No fresh self-report — fall through to the transcript-based path.
        if rr.transcript is None:
            entry_error = rr.reason or "transcript unresolvable"
        else:
            try:
                tail, last_activity_ts = read_transcript_tail(rr.transcript)
            except OSError as e:
                entry_error = f"tail read failed: {type(e).__name__}: {e}"
            if tail is not None and rr.issue is not None:
                (
                    summary,
                    summary_ts,
                    summary_model,
                    source,
                    entry_error,
                ) = await _produce_summary(
                    tail=tail,
                    last_activity_ts=last_activity_ts,
                    prior_entry=prior_entry,
                    issue=rr.issue,
                    client=client,
                    dry_run=dry_run,
                )
            elif tail is not None and rr.issue is None:
                # No issue to attribute the session to — record the
                # transcript path + last activity, but skip the LLM call
                # (the prompt template requires an issue number).
                entry_error = "no /issue prompt found in transcript head"
    except Exception as e:
        entry_error = f"unhandled per-session error: {type(e).__name__}: {e}"
    payload["sessions"][sid] = build_session_entry(
        sid=sid,
        pid=pid,
        issue=rr.issue,
        cwd=rr.cwd,
        transcript=rr.transcript,
        summary=summary,
        summary_ts=summary_ts,
        last_activity_ts=last_activity_ts,
        error=entry_error,
        summary_model=summary_model,
        source=source,
    )


async def _produce_summary(
    *,
    tail: str,
    last_activity_ts: str | None,
    prior_entry: dict | None,
    issue: int,
    client,
    dry_run: bool,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Decide whether to reuse / skip / call Haiku for one session, and
    return ``(summary, summary_ts, summary_model, source, error)``.

    Resolution order (first match wins):
      1. Fresh self-report from /issue (``~/.eps-autonomous/issue-progress/
         <N>.json``) -> reuse ``text`` verbatim with ``source="self"`` and
         skip the LLM call entirely. This makes the dashboard / happy-ls
         progress column byte-identical to the phone title /issue set, AND
         drops the Haiku call from the steady-state cost path for sessions
         the orchestrator is already driving.
      2. Empty tail -> error "transcript tail empty", no LLM call.
      3. Activity unchanged vs prior cache entry -> reuse cached summary
         (``source`` inherited from the prior entry: a prior self-report
         summary stays tagged "self", a prior LLM summary stays "llm").
      4. Dry-run -> placeholder summary, no LLM call.
      5. Otherwise -> call Haiku; record exception as visible error.

    Self-report precedence is FIRST so a freshly-written /issue tick
    overrides whatever the prior cache entry held, even if the transcript
    timestamp would otherwise trigger the idle-skip branch."""
    report = session_progress_report.read_self_report(issue)
    if report is not None and session_progress_report.is_self_report_fresh(report):
        text = report["text"]
        # Truthy + str-typed are guaranteed by read_self_report's validation;
        # the cast is just for type-narrowing.
        return (text if isinstance(text, str) else None, _utcnow_iso(), None, "self", None)

    if not tail.strip():
        # Sending whitespace to Haiku would burn a call on a meaningless
        # prompt; surface the gap visibly instead.
        return None, None, None, None, "transcript tail empty"
    if _should_skip_llm_call(prior_entry, last_activity_ts):
        # Activity hasn't advanced — reuse the cached summary instead of
        # calling Haiku again.
        assert prior_entry is not None  # guaranteed by _should_skip_llm_call
        prior_summary = prior_entry["summary"]
        prior_ts = prior_entry.get("summary_ts")
        prior_model = prior_entry.get("summary_model")
        prior_source = prior_entry.get("source")
        # Carry the prior source forward; an unknown / legacy entry without
        # a source field is treated as LLM (the only producer pre-this-fix).
        carried_source = prior_source if prior_source in ("self", "llm") else "llm"
        return (
            prior_summary if isinstance(prior_summary, str) else None,
            prior_ts if isinstance(prior_ts, str) else None,
            prior_model if isinstance(prior_model, str) else None,
            carried_source,
            None,
        )
    if dry_run:
        return "<dry-run: no API call made>", _utcnow_iso(), HAIKU_MODEL_ID, "llm", None
    try:
        summary = await _summarize_one(client, issue, _get_task_status(issue), tail)
    except Exception as e:
        return None, None, None, None, f"summarize call failed: {type(e).__name__}: {e}"
    return summary, _utcnow_iso(), HAIKU_MODEL_ID, "llm", None


async def _run_pass(dry_run: bool) -> dict:
    """One end-to-end pass. Returns the constructed cache payload.

    Idle-skip: before any LLM call, compare each session's current
    ``last_activity_ts`` (newest transcript timestamp in the tail window)
    against the prior cache entry's ``last_activity_ts``. If unchanged AND
    the prior entry has a non-empty summary, reuse it — refresh only the
    volatile fields (``live`` / ``status`` / ``pid`` / ``last_activity_ts``).
    Cuts API cost ~5x in steady state where most EPS sessions park in
    ``awaiting_promotion`` / ``planning`` for hours."""
    # Pre-load the prior cache once; per-session idle-skip reads it.
    cache_obj = load_cache()
    prior_cache_raw = cache_obj.get("sessions") if isinstance(cache_obj, dict) else None
    prior_cache = prior_cache_raw if isinstance(prior_cache_raw, dict) else {}

    # Discover live sessions, filter to EPS-only with a resolvable transcript.
    live = session_resolver._live_node_pids()
    eps_targets: list[tuple[str, int, session_resolver.ResolveResult]] = []
    for sid, pid in live:
        rr = session_resolver.resolve(pid)
        if not session_resolver.is_eps_cwd(rr.cwd):
            continue
        eps_targets.append((sid, pid, rr))

    payload: dict = {"updated_at": _utcnow_iso(), "sessions": {}}

    if not eps_targets:
        if not dry_run:
            _atomic_write_json(CACHE_PATH, payload)
        return payload

    # Construct client lazily (.env may need loading first). The
    # AnthropicChatModel's own ``num_threads`` semaphore already bounds
    # concurrent API calls — a second outer semaphore would just serialize
    # work the inner one is happy to parallelize.
    _ensure_env_loaded()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Fail loud: a missing key is not a transient error.
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; .env not loaded or key missing. "
            "Check setup_env / .env at repo root."
        )

    from explore_persona_space.llm.anthropic_client import AnthropicChatModel

    client = AnthropicChatModel(num_threads=_CONCURRENCY) if not dry_run else None

    await asyncio.gather(
        *(
            _summarize_session(
                payload=payload,
                sid=sid,
                pid=pid,
                rr=rr,
                prior_entry=(prior_cache.get(sid) if isinstance(prior_cache, dict) else None),
                client=client,
                dry_run=dry_run,
            )
            for sid, pid, rr in eps_targets
        )
    )

    if not dry_run:
        _atomic_write_json(CACHE_PATH, payload)

    return payload


# ── cache reader (used by spawn_session list) ─────────────────────────────


def load_cache() -> dict:
    """Read the shared cache; return ``{}`` if missing / unreadable.

    Best-effort enrichment for `cmd_list`: a missing or partially-written
    cache is treated as "no cache entries"; the table falls back to the
    marker-based progress cell."""
    if not CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def get_cached_summary(happy_session_id: str) -> dict | None:
    """Return one session's cache entry (or None if not present)."""
    data = load_cache()
    sessions = data.get("sessions", {})
    entry = sessions.get(happy_session_id)
    if isinstance(entry, dict):
        return entry
    return None


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the API call and the cache write; print what would happen.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit per-session progress to stderr.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()
    payload = asyncio.run(_run_pass(dry_run=args.dry_run))
    dt = time.time() - t0
    n = len(payload.get("sessions", {}))
    ok = sum(1 for e in payload.get("sessions", {}).values() if e.get("summary"))
    # Auxiliary digest artifact — written AFTER _run_pass has already committed
    # the session_progress.json cache, so a digest failure can never break the
    # primary cache write (the cron contract). A failure is surfaced loudly
    # (logged + noted in the summary line), never swallowed silently.
    digest_note = ""
    if not args.dry_run:
        try:
            dest = write_sessions_digest(payload)
            digest_note = f"; digest {dest}"
        except Exception as e:
            logger.warning("sessions digest write failed: %s: %s", type(e).__name__, e)
            digest_note = f"; digest FAILED ({type(e).__name__})"
    print(
        f"session_summarize: {n} EPS session(s); {ok} summarized; "
        f"{n - ok} skipped/errored; {dt:.1f}s; "
        f"{'(dry-run)' if args.dry_run else f'wrote {CACHE_PATH}'}"
        f"{digest_note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
