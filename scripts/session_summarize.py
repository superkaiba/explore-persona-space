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
          "last_activity_ts": "<ISO8601 UTC of newest transcript entry>",
          "error": null
        },
        ...
      }
    }

Design choices:

- **Per-session try/except** with a VISIBLE ``error`` field — one bad session
  must not abort the run (CLAUDE.md fail-fast: surface, don't swallow).
- **Tail-only input** keeps per-call cost cheap (Haiku is the cheapest tier;
  ~25 entries truncated to roughly the last 30k chars of raw text).
- **Live + EPS only** — non-EPS sessions (my-goat, introsp) and dead sessions
  are skipped entirely.
- **Reuses ``AnthropicChatModel``** from ``explore_persona_space.llm``
  (CLAUDE.md: search before building; never hand-roll a new client).

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
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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

    Reads up to the final ``tail_lines`` lines, truncates to the last
    ``char_cap`` characters (favoring the END of the file — the most recent
    content). Also scans those lines for the newest entry timestamp and
    returns it. On a read failure raises OSError; the caller has the
    per-session try/except."""
    with open(transcript_path) as fh:
        lines = fh.readlines()[-tail_lines:]
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
) -> dict[str, object]:
    """Construct the cache entry for one session. Pure — no I/O.

    Captured as a named function so the schema is enforced in ONE place
    (tests pin the keys + types here so a silent schema drift is caught)."""
    status = _get_task_status(issue) if issue is not None else None
    return {
        "issue": issue,
        "status": status,
        "dir": _dir_label_for_cache(cwd),
        "live": True,
        "pid": pid,
        "transcript": transcript,
        "summary": summary,
        "summary_model": HAIKU_MODEL_ID if summary is not None else None,
        "summary_ts": summary_ts,
        "last_activity_ts": last_activity_ts,
        "error": error,
    }


async def _run_pass(dry_run: bool) -> dict:
    """One end-to-end pass. Returns the constructed cache payload."""
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

    # Construct client lazily (.env may need loading first).
    _ensure_env_loaded()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Fail loud: a missing key is not a transient error.
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; .env not loaded or key missing. "
            "Check setup_env / .env at repo root."
        )

    from explore_persona_space.llm.anthropic_client import AnthropicChatModel

    client = AnthropicChatModel(num_threads=_CONCURRENCY) if not dry_run else None

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _one(sid: str, pid: int, rr: session_resolver.ResolveResult) -> None:
        entry_error: str | None = None
        summary: str | None = None
        summary_ts: str | None = None
        last_activity_ts: str | None = None
        try:
            if rr.transcript is None:
                entry_error = rr.reason or "transcript unresolvable"
            else:
                try:
                    tail, last_activity_ts = read_transcript_tail(rr.transcript)
                except OSError as e:
                    entry_error = f"tail read failed: {type(e).__name__}: {e}"
                    tail = None
                if tail is not None and rr.issue is not None:
                    if dry_run:
                        summary = "<dry-run: no API call made>"
                        summary_ts = _utcnow_iso()
                    else:
                        try:
                            async with semaphore:
                                summary = await _summarize_one(
                                    client, rr.issue, _get_task_status(rr.issue), tail
                                )
                            summary_ts = _utcnow_iso()
                        except Exception as e:
                            entry_error = f"summarize call failed: {type(e).__name__}: {e}"
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
        )

    await asyncio.gather(*(_one(sid, pid, rr) for sid, pid, rr in eps_targets))

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
    print(
        f"session_summarize: {n} EPS session(s); {ok} summarized; "
        f"{n - ok} skipped/errored; {dt:.1f}s; "
        f"{'(dry-run)' if args.dry_run else f'wrote {CACHE_PATH}'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
