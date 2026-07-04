"""Single canonical per-session progress string + per-issue self-report writer.

Until this module existed, EPS had TWO divergent "what is this session doing"
mechanisms producing DIFFERENT strings:

1. The ``/issue`` skill set the Happy session TITLE via
   ``mcp__happy__change_title`` (only at status transitions). This drove the
   phone session title.
2. ``scripts/session_summarize.py`` LLM-summarized each transcript tail every
   5 minutes (Haiku) and wrote a per-session ``summary`` to
   ``~/.eps-autonomous/session_progress.json``. This drove ``happy-ls`` +
   the ``/sessions`` dashboard.

Same session, two different strings, two different cadences.

This module unifies them. The single canonical string is built by
:func:`build_progress_string` here. The ``/issue`` skill calls this module on
every tick — it (a) gets the string and (b) writes a self-report file at
``~/.eps-autonomous/issue-progress/<N>.json``. The session then passes the
string to ``mcp__happy__change_title``. The 5-min summarizer cron reads the
self-report file first: if a fresh self-report exists for an issue's session,
it uses that string verbatim as the cache ``summary`` (and tags
``source: "self"``) — no Haiku call. Sessions without a fresh self-report
(interactive, non-/issue, or stale) fall back to the LLM path
(``source: "llm"``).

Net effect: the phone title and the terminal/dashboard ``summary`` columns
become byte-identical for /issue sessions, and the Haiku call drops out of
the steady-state cost path.

Schema (per issue, atomic temp+rename)::

    ~/.eps-autonomous/issue-progress/<N>.json
    {
      "issue": 492,
      "slug": "wire /issue auto-title into session",
      "step": "awaiting promotion",
      "text": "#492 wire /issue auto-title into session · awaiting promotion",
      "ts": "2026-06-05T18:22:14Z"
    }

CLI::

    uv run python scripts/session_progress_report.py \\
        --issue 492 --step "awaiting promotion"
    # -> writes ~/.eps-autonomous/issue-progress/492.json
    # -> prints "#492 <slug> · awaiting promotion"

``--slug`` is optional — when omitted, falls back to the task's frontmatter
title via ``explore_persona_space.task_workflow.get_task`` (so the skill does
not have to thread the slug through every tick). Fail-loud: an unknown issue
raises rather than silently writing a blank string. A *transient*
resolver/environment ``RuntimeError`` (e.g. a detached repo-root HEAD
mid-rebase) instead degrades soft — see :func:`write_self_report`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Layout — single source of truth so other modules import the constants.
SELF_REPORT_DIR = Path.home() / ".eps-autonomous" / "issue-progress"

# Hard cap on the canonical string. The phone title widget truncates around
# 80 chars on most clients; we cap below that so the string is byte-identical
# everywhere (no client-side truncation discrepancy between the phone and the
# terminal). Truncation favors keeping the leading ``#<N>`` (always findable)
# and a useful slug, trimming the trailing step description first.
PROGRESS_STRING_MAX = 78

# Slug clip — keeps the phone title readable. Together with PROGRESS_STRING_MAX
# this leaves comfortable room for the issue number + step description without
# client-side truncation.
SLUG_MAX = 45

# Freshness window the summarizer uses to decide whether to reuse the
# self-report or fall back to Haiku. Two loop intervals (~20 min) gives the
# /issue session a full tick of slack on a 10-min backstop without ever
# letting a stale string outlive its session. Override per-call via the
# explicit ``freshness_window_seconds`` arg in
# :func:`is_self_report_fresh`. Exposed as a module constant so the
# summarizer test + production both reference one number.
DEFAULT_FRESHNESS_WINDOW_SECONDS = 20 * 60


def _utcnow_iso() -> str:
    """UTC timestamp in the same ``YYYY-MM-DDTHH:MM:SSZ`` shape used by
    ``session_summarize.py`` so freshness comparisons are apples-to-apples."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> datetime | None:
    """Parse the canonical trailing-Z UTC timestamp. Returns None on a
    malformed string (caller treats it as "no timestamp")."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def self_report_path(issue: int) -> Path:
    """Per-issue self-report file path. Pure — no I/O."""
    return SELF_REPORT_DIR / f"{int(issue)}.json"


def build_progress_string(issue: int, slug: str, step: str, suffix: str | None = None) -> str:
    """Construct the SINGLE canonical per-session progress string.

    Format: ``"#<N> <slug> · <step>"`` (Unicode middle-dot separator, leading
    ``#<N>`` so the row is findable in a phone session list). Hard-capped to
    :data:`PROGRESS_STRING_MAX` characters. Slug is pre-clipped to
    :data:`SLUG_MAX`. If the joined string still exceeds the cap (very long
    step text), the STEP is trimmed with a trailing ``…`` — the issue number
    and slug stay intact (they are the part the user uses to find the row).

    ``suffix`` (task #587) appends a progress-bar/ETA tail
    (``"▓▓░░░ 43% ~4-9h"``) as ``"<head> · <step> · <suffix>"``.
    ``suffix=None`` is BYTE-IDENTICAL to the historical behavior for every
    input. Overflow degrade ladder when the suffix is present:

      1. full string fits → done
      2. trim the STEP (existing ``…`` rule), keep the suffix whole
      3. drop the block-bar chars from the suffix (``"43% ~4-9h"``), retry
      4. drop the suffix entirely → the existing no-suffix path

    Pure (no I/O). Both the writer here and the SKILL.md helper call this
    function so the format lives in exactly ONE place.
    """
    slug_clean = (slug or "").strip()
    step_clean = (step or "").strip()
    if len(slug_clean) > SLUG_MAX:
        slug_clean = slug_clean[:SLUG_MAX].rstrip()
    head = f"#{int(issue)} {slug_clean}".rstrip()
    sep = " · "

    if suffix:
        suffix_clean = suffix.strip()
        for sfx in (suffix_clean, suffix_clean.lstrip("▓░ ")):
            if not sfx:
                continue
            parts = [head, step_clean, sfx] if step_clean else [head, sfx]
            text = sep.join(parts)
            if len(text) <= PROGRESS_STRING_MAX:
                return text
            if step_clean:
                # Trim the STEP, keep the suffix whole (≥1 step char + "…").
                overhead = len(head) + 2 * len(sep) + len(sfx) + 1
                budget = PROGRESS_STRING_MAX - overhead
                if budget >= 1:
                    trimmed = step_clean[:budget].rstrip() + "…"
                    return f"{head}{sep}{trimmed}{sep}{sfx}"
        # Ladder exhausted — fall through to the no-suffix path below.

    if not step_clean:
        return head[:PROGRESS_STRING_MAX]
    text = f"{head} · {step_clean}"
    if len(text) <= PROGRESS_STRING_MAX:
        return text
    # Trim the step to fit, preserving "<head> · " + at least 1 char of step.
    overhead = len(head) + len(sep) + 1  # at least 1 step char + the ellipsis
    budget = PROGRESS_STRING_MAX - overhead
    if budget <= 0:
        # Head + separator alone exceeds the cap — drop the step entirely;
        # `head` is already <= cap (slug pre-clipped + small issue number).
        return head[:PROGRESS_STRING_MAX]
    trimmed_step = step_clean[:budget].rstrip() + "…"
    return f"{head}{sep}{trimmed_step}"


def _load_task(issue: int) -> dict:
    """Return the full task dict (frontmatter + status), or raise
    ``FileNotFoundError`` if the issue is unknown. Imported lazily so this
    module stays importable in a context that doesn't have the EPS package
    installed (e.g. a bare CLI invocation on a fresh pod)."""
    from explore_persona_space.task_workflow import get_task

    return get_task(issue)


# Machine-active statuses whose title gets the progress-bar/ETA suffix
# (task #587). Gate-park / terminal / human-wait statuses (proposed,
# plan_pending, blocked, awaiting_promotion, followups_running, completed,
# archived, and anything else) keep the byte-identical historical title.
_ETA_SUFFIX_STATUSES = frozenset(
    {"planning", "approved", "running", "verifying", "interpreting", "reviewing"}
)


def _compute_eta_suffix(issue: int, status: str | None, now_iso: str | None = None) -> str | None:
    """Progress-bar/ETA title suffix for machine-active statuses, else None.

    FAIL-SOFT by design (plan #587 §3.6): the title must never break because
    the estimator did — any exception degrades to ``None`` with one stderr
    line. Reads stats READ-ONLY from the materialized snapshot
    (``task_progress.load_stats_readonly``); a missing/stale snapshot (dead
    cron) yields ``None`` — this path NEVER rebuilds stats (a title tick must
    stay O(1), not a 33 MB events.jsonl scan).
    """
    if status not in _ETA_SUFFIX_STATUSES:
        return None
    try:
        from explore_persona_space import task_progress as tp

        stats = tp.load_stats_readonly()
        if stats is None:
            return None
        now = _parse_iso(now_iso) if now_iso else None
        row = tp.estimate_task_progress(issue, stats, now=now)
        return tp.format_title_suffix(row, now=now)
    except Exception as exc:  # fail-soft: suffix is decoration, never a crash
        print(f"session_progress_report: ETA suffix skipped ({exc})", file=sys.stderr)
        return None


def write_self_report(
    issue: int,
    *,
    slug: str | None = None,
    step: str,
    now_iso: str | None = None,
    eta: bool = True,
) -> tuple[str, Path]:
    """Build the canonical string and atomically write it to the self-report
    file for ``issue``. Returns ``(text, path)`` — the canonical string and
    the file path it landed at. Atomic via temp + rename so any concurrent
    summarizer pass never reads a partial file.

    ``slug`` defaults to the task's frontmatter ``title`` via
    :func:`_load_task`. Pass an explicit slug to override
    (e.g. the SKILL.md helper that already has the task loaded — saves the
    extra disk read).

    **Fail-loud on unknown issue regardless of ``--slug``.** We ALWAYS call
    :func:`_load_task` even when the caller supplied an explicit
    slug, so a typo'd issue number produces a ``FileNotFoundError`` (from
    ``task.py find <N>``) instead of silently writing a self-report file for
    a non-existent task — keeps the fail-loud contract consistent across
    both code paths.

    **Fail-SOFT on transient resolver/environment errors.** A ``RuntimeError``
    from the task-workflow repo-root resolver (e.g. a detached shared-checkout
    HEAD during a concurrent rebase — #999; since #996 the resolver
    bounded-waits on a LIVE rebase before raising, so this catch is the
    backstop for wait-timeout / stranded-detached / other environment
    RuntimeErrors) degrades instead of crashing: caller-supplied slug or
    empty, no ETA suffix, self-report still written, one stderr line, normal
    return. The unknown-issue ``FileNotFoundError`` contract above is
    unaffected.

    ``eta=True`` (default) appends the progress-bar/ETA suffix for
    machine-active statuses via :func:`_compute_eta_suffix` (task #587).
    The suffix is fail-soft and status-gated, so every gate-park / terminal
    status produces the byte-identical historical string. ``eta=False``
    (CLI ``--no-eta``) skips the estimator entirely.
    """
    try:
        task: dict | None = _load_task(issue)
    except RuntimeError as exc:
        # TRANSIENT resolver/environment failure — e.g. the repo-root branch
        # guard's detached-HEAD raise (task_workflow._resolve_repo_root_cached)
        # while a concurrent session's rebase window holds the shared checkout
        # (#999; crash observed in the #906 session, 2026-07-03). Since #996
        # the resolver bounded-waits on a LIVE rebase before raising, so this
        # catch is the backstop for wait-timeout / stranded-detached / other
        # environment RuntimeErrors. The title/self-report path is
        # OBSERVABILITY ONLY (SKILL.md § set_title), so degrade instead of
        # crashing: keep the caller-supplied slug (else empty), skip the ETA
        # suffix (status unknown), and STILL write the self-report — its `ts`
        # is an activity signal for the stalled detector, and this tick IS
        # activity. Unknown-issue typos still fail loud: FileNotFoundError
        # (incl. StaleTaskPathError) is an OSError, not a RuntimeError, and
        # is deliberately not caught here.
        print(
            f"session_progress_report: task lookup degraded ({exc})",
            file=sys.stderr,
        )
        task = None
    if task is None:
        status: str | None = None
        if slug is None:
            slug = ""
    else:
        fm = task.get("frontmatter")
        fm = fm if isinstance(fm, dict) else {}
        if slug is None:
            title = fm.get("title")
            slug = title.strip() if isinstance(title, str) else ""
        status = task.get("status")
    suffix = _compute_eta_suffix(issue, status, now_iso) if eta else None
    text = build_progress_string(issue, slug, step, suffix=suffix)
    ts = now_iso or _utcnow_iso()
    payload = {
        "issue": int(issue),
        "slug": slug,
        "step": step,
        "text": text,
        "ts": ts,
    }
    path = self_report_path(issue)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return text, path


def read_self_report(issue: int) -> dict | None:
    """Read the self-report for ``issue`` and return the dict, or ``None`` if
    the file is missing OR malformed. A malformed file is treated as "no
    self-report" so the summarizer transparently falls back to the LLM path
    — the user sees a visible LLM-tagged summary instead of a stale or
    fabricated string. Pure read; the writer is the only mutator.
    """
    path = self_report_path(issue)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Required fields. Any missing one => treat as malformed.
    if not isinstance(data.get("text"), str) or not data["text"].strip():
        return None
    if not isinstance(data.get("ts"), str) or not data["ts"]:
        return None
    return data


def is_self_report_fresh(
    report: dict,
    *,
    now_iso: str | None = None,
    freshness_window_seconds: float = DEFAULT_FRESHNESS_WINDOW_SECONDS,
) -> bool:
    """Return True iff ``report["ts"]`` is within ``freshness_window_seconds``
    of ``now_iso`` (default: current UTC). Pure; both ``ts`` and ``now`` are
    parsed via :func:`_parse_iso` so a malformed timestamp degrades to
    "not fresh" (the summarizer falls back to the LLM path, never a silent
    stale string).
    """
    ts_str = report.get("ts") if isinstance(report, dict) else None
    if not isinstance(ts_str, str):
        return False
    ts = _parse_iso(ts_str)
    if ts is None:
        return False
    now = _parse_iso(now_iso) if now_iso else datetime.now(tz=UTC)
    if now is None:
        return False
    delta = (now - ts).total_seconds()
    # Negative delta (clock skew) is treated as "future" and therefore fresh
    # — better than fabricating staleness from a few seconds of skew.
    return delta < freshness_window_seconds


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Write the canonical per-issue session progress string and print it to stdout."
        )
    )
    p.add_argument("--issue", type=int, required=True, help="Task / issue number")
    p.add_argument(
        "--step",
        required=True,
        help="What the session is doing right now (e.g. 'awaiting promotion', 'critic round 2').",
    )
    p.add_argument(
        "--slug",
        default=None,
        help=(
            "Override the slug. Defaults to the task's frontmatter `title` "
            "via task_workflow.get_task(issue)."
        ),
    )
    p.add_argument(
        "--no-eta",
        action="store_true",
        help=(
            "Skip the progress-bar/ETA title suffix (task #587). Without "
            "this flag the suffix is appended for machine-active statuses "
            "and omitted (byte-identical title) everywhere else."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    text, _path = write_self_report(args.issue, slug=args.slug, step=args.step, eta=not args.no_eta)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
