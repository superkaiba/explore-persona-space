"""Rename each live tmux WINDOW to a short summary of what it is working on.

The tmux SESSION name is left unchanged (stable for reattach / muscle memory);
only the window name is set, and ``automatic-rename`` + ``allow-rename`` are
turned off on the touched window so the shell / TUI does not immediately revert
it back to ``node`` / ``bash``.

Three summary tiers (most authoritative first), so "all the running tmuxes"
get a label while EPS sessions stay free:

1. **EPS session cache** — if the window's resolved Claude transcript matches an
   entry in ``~/.eps-autonomous/session_progress.json`` (written every 5 min by
   ``session_summarize.py``), reuse that summary verbatim. No model call.
2. **Other Claude Code session** — any tmux session running a ``claude``
   subprocess whose transcript resolves but is NOT in the EPS cache (e.g.
   my-goat). Read the transcript tail and ask Haiku for a short label.
3. **Non-Claude session** — a plain shell, a daemon (``eps-program``), etc. No
   resolvable transcript. Capture the pane scrollback tail and ask Haiku.

Tiers 2 and 3 are idle-skipped via a small state file
(``~/.eps-autonomous/tmux_window_titles.json``): the Haiku call is skipped when
the fingerprint (transcript last-activity ts, or scrollback hash) is unchanged
since the last pass.

CLI::

    uv run python scripts/tmux_window_titles.py apply            # rename windows
    uv run python scripts/tmux_window_titles.py apply --dry-run  # show, don't rename
    uv run python scripts/tmux_window_titles.py list             # session -> window summary
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import session_resolver  # noqa: E402
import session_summarize  # noqa: E402

STATE_PATH = Path.home() / ".eps-autonomous" / "tmux_window_titles.json"

HAIKU_MODEL_ID = session_summarize.HAIKU_MODEL_ID

# Window name length cap. tmux truncates in the status bar anyway, but a tight
# cap keeps the switcher tree readable.
_MAX_LABEL_CHARS = 52

# Scrollback lines captured for the non-Claude (tier 3) summary.
_SCROLLBACK_LINES = 80

_LABEL_PROMPT = """\
Below is the recent {kind} of a terminal session. Say what it is working on
RIGHT NOW. Reply with EXACTLY two lines, nothing else:

LABEL: <4-8 words, lead with a noun or verb, no trailing punctuation, no quotes>
SUMMARY: <one plain-English sentence with more specifics — what + what it's waiting on>

```
{body}
```
"""


def _parse_haiku_reply(raw: str) -> tuple[str, str]:
    """Parse the two-line ``LABEL:`` / ``SUMMARY:`` reply.

    Tolerant of a model that drops a prefix: if neither line is tagged, the
    first non-empty line becomes the label and the rest the summary. Returns
    ``(label, summary)`` — either may be empty, the caller guards that."""
    label = ""
    summary = ""
    leftover: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("label:"):
            label = s.split(":", 1)[1].strip()
        elif low.startswith("summary:"):
            summary = s.split(":", 1)[1].strip()
        else:
            leftover.append(s)
    if not label and leftover:
        label = leftover[0]
    if not summary:
        summary = " ".join(leftover[1:]) if len(leftover) > 1 else (summary or label)
    return label, summary


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── tmux plumbing (works from outside tmux, e.g. under cron) ────────────────


def _tmux(*args: str) -> subprocess.CompletedProcess:
    """Run a tmux command against the user's running server.

    ``TMUX`` is stripped so this behaves identically inside or outside a tmux
    client (cron has no ``$TMUX``; leaving a stale one set makes tmux refuse
    some commands with "sessions should be nested with care")."""
    env = {k: v for k, v in os.environ.items() if k != "TMUX"}
    return subprocess.run(
        ["tmux", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass
class Window:
    session: str
    index: str
    name: str
    pane_pid: int
    # Filled during resolution:
    label: str | None = None  # terse 4-8 words → the tmux window name
    summary: str | None = None  # fuller one-liner → the `list` / `detail` view
    source: str | None = None  # "cache" | "transcript" | "scrollback" | None
    fingerprint: str | None = None
    error: str | None = None
    needs_haiku: bool = False
    issue: int | None = None
    _haiku_kind: str = field(default="", repr=False)
    _haiku_body: str = field(default="", repr=False)

    @property
    def target(self) -> str:
        return f"{self.session}:{self.index}"


def list_windows() -> list[Window]:
    """Every live window across every session, as :class:`Window` rows."""
    fmt = "#{session_name}\t#{window_index}\t#{window_name}\t#{pane_pid}"
    proc = _tmux("list-windows", "-a", "-F", fmt)
    if proc.returncode != 0:
        return []
    out: list[Window] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        session, index, name, pane_pid = parts
        try:
            pid = int(pane_pid)
        except ValueError:
            continue
        out.append(Window(session=session, index=index, name=name, pane_pid=pid))
    return out


def _all_descendant_pids(root: int) -> list[int]:
    """``root`` plus every descendant pid (BFS over /proc children)."""
    seen: set[int] = set()
    order: list[int] = []
    stack = [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        order.append(p)
        stack.extend(session_resolver._proc_children(p))
    return order


# ── transcript / cache resolution ───────────────────────────────────────────


def _resolve_transcript_for_window(pane_pid: int) -> session_resolver.ResolveResult | None:
    """Resolve a Claude transcript for a window by trying the pane pid and each
    of its descendant node pids. Returns the first ResolveResult that yields a
    transcript, or None if no descendant is a Claude session."""
    for pid in _all_descendant_pids(pane_pid):
        if session_resolver._read_proc_comm(pid) != "node":
            continue
        rr = session_resolver.resolve(pid)
        if rr.transcript is not None:
            return rr
    return None


def _cache_by_transcript() -> dict[str, dict]:
    """Map transcript path -> EPS cache entry (only entries with a summary)."""
    cache = session_summarize.load_cache()
    out: dict[str, dict] = {}
    for entry in (cache.get("sessions") or {}).values():
        if not isinstance(entry, dict):
            continue
        t = entry.get("transcript")
        s = entry.get("summary")
        if isinstance(t, str) and t and isinstance(s, str) and s:
            out[t] = entry
    return out


# ── label shaping ────────────────────────────────────────────────────────────


def shorten(text: str, max_chars: int = _MAX_LABEL_CHARS) -> str:
    """Collapse whitespace, drop a trailing status/progress suffix, and
    truncate at a word boundary to ``max_chars``."""
    text = " ".join(text.split())
    # Self-report summaries look like "#740 foo bar · running · ▓░░░░ 26%";
    # keep only the part before the first " · " status separator.
    if " · " in text:
        text = text.split(" · ", 1)[0].strip()
    # A summary that *began* with the separator collapses to a bare "·";
    # strip leading bullets so we never emit "· running" as a label.
    text = text.lstrip("·").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


def label_from_cache_entry(entry: dict) -> str:
    issue = entry.get("issue")
    summary = entry.get("summary") or ""
    short = shorten(summary)
    issue_tag = f"#{issue} " if isinstance(issue, int) else ""
    if issue_tag and short.lstrip().startswith(issue_tag.strip()):
        issue_tag = ""  # summary already begins with "#N"
    return shorten(f"{issue_tag}{short}")


def label_from_haiku(raw: str, issue: int | None) -> str:
    short = shorten(raw)
    issue_tag = f"#{issue} " if isinstance(issue, int) else ""
    if issue_tag and short.lstrip().startswith(issue_tag.strip()):
        issue_tag = ""
    return shorten(f"{issue_tag}{short}")


# ── per-window resolution (no Haiku yet — just decide the source) ───────────


def _load_state() -> dict:
    if not STATE_PATH.is_file():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_PATH)


def _capture_scrollback(target: str) -> str:
    proc = _tmux("capture-pane", "-t", target, "-p", "-S", f"-{_SCROLLBACK_LINES}")
    if proc.returncode != 0:
        return ""
    # Drop blank lines and box-drawing noise so Haiku sees the real content.
    lines = [ln.rstrip() for ln in proc.stdout.splitlines() if ln.strip()]
    return "\n".join(lines[-_SCROLLBACK_LINES:])


def resolve_window(win: Window, cache_by_t: dict[str, dict], state: dict) -> None:
    """Populate ``win`` with a label (tier 1) or mark it for a Haiku call
    (tiers 2/3) plus the idle-skip fingerprint. No model call here."""
    try:
        rr = _resolve_transcript_for_window(win.pane_pid)
        # State is keyed per WINDOW (session:index), not per session — a
        # session can hold >1 window and each needs its own idle-skip
        # fingerprint + label.
        prior = state.get(win.target) if isinstance(state.get(win.target), dict) else {}

        if rr is not None and rr.transcript is not None:
            # Tier 1: EPS cache hit by transcript path.
            entry = cache_by_t.get(rr.transcript)
            if entry is not None:
                win.label = label_from_cache_entry(entry)
                # The cache already holds a 1-2 sentence summary — use it
                # verbatim for the view (the label is the truncated form).
                win.summary = " ".join((entry.get("summary") or "").split()) or win.label
                win.source = "cache"
                return
            # Tier 2: a Claude session not in the EPS cache (e.g. my-goat).
            try:
                tail, last_ts = session_summarize.read_transcript_tail(rr.transcript)
            except OSError as e:
                win.error = f"tail read failed: {type(e).__name__}"
                return
            fp = last_ts or hashlib.sha1(tail.encode()).hexdigest()[:16]
            win.fingerprint = fp
            win.source = "transcript"
            if prior.get("fingerprint") == fp and prior.get("label"):
                win.label = str(prior["label"])  # idle-skip: reuse
                win.summary = str(prior.get("summary") or prior["label"])
                return
            win.needs_haiku = True
            win._haiku_kind = "Claude Code transcript tail"
            win._haiku_body = tail[-12000:]
            win.issue = rr.issue
            return

        # Tier 3: non-Claude session — summarize the pane scrollback.
        body = _capture_scrollback(win.target)
        if not body.strip():
            win.error = "empty scrollback, no transcript"
            return
        fp = hashlib.sha1(body.encode()).hexdigest()[:16]
        win.fingerprint = fp
        win.source = "scrollback"
        if prior.get("fingerprint") == fp and prior.get("label"):
            win.label = str(prior["label"])
            win.summary = str(prior.get("summary") or prior["label"])
            return
        win.needs_haiku = True
        win._haiku_kind = "terminal scrollback"
        win._haiku_body = body[-12000:]
        win.issue = None
    except Exception as e:  # never let one window abort the pass
        win.error = f"resolve failed: {type(e).__name__}: {e}"


# ── Haiku batch ──────────────────────────────────────────────────────────────


async def _run_haiku(windows: list[Window]) -> None:
    """Fill labels for every window with ``needs_haiku`` via one concurrent
    Haiku batch. Mutates the windows in place; per-window errors are recorded."""
    pending = [w for w in windows if w.needs_haiku]
    if not pending:
        return
    session_summarize._ensure_env_loaded()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for w in pending:
            w.error = "ANTHROPIC_API_KEY not set"
        return
    from explore_persona_space.llm.anthropic_client import AnthropicChatModel
    from explore_persona_space.llm.models import ChatMessage, MessageRole, Prompt

    client = AnthropicChatModel(num_threads=min(8, len(pending)))

    async def one(w: Window) -> None:
        prompt_text = _LABEL_PROMPT.format(kind=w._haiku_kind, body=w._haiku_body)
        prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
        try:
            responses = await client(
                model_id=HAIKU_MODEL_ID,
                prompt=prompt,
                max_tokens=120,
                temperature=0.2,
            )
            raw = (responses[0].completion or "").strip() if responses else ""
        except Exception as e:
            w.error = f"haiku failed: {type(e).__name__}: {e}"
            return
        if not raw:
            w.error = "haiku returned empty"
            return
        label, summary = _parse_haiku_reply(raw)
        summary = " ".join(summary.split())
        # If the reply gave a SUMMARY but no LABEL, derive the label from the
        # summary so the window still gets named AND its fingerprint persists
        # (an empty label would skip the state write → silent re-Haiku loop).
        w.label = label_from_haiku(label or shorten(summary), w.issue)
        w.summary = summary or w.label
        if not (w.label and w.label.strip()):
            # Make the failure visible + let apply()'s carry-forward keep the
            # prior fingerprint instead of re-Haiku-ing this window every tick.
            w.error = "haiku reply had no usable label"

    await asyncio.gather(*(one(w) for w in pending))


# ── apply ────────────────────────────────────────────────────────────────────


def apply(dry_run: bool = False, persist: bool | None = None) -> list[Window]:
    """Resolve + (optionally) rename every live tmux window.

    ``dry_run`` skips the actual ``rename-window``. ``persist`` controls the
    state-file write; it defaults to ``not dry_run`` but the read-only views
    pass ``persist=True`` so a manual ``list``/``detail`` still caches the
    fingerprints it just computed (otherwise the next cron pass re-Haikus the
    same windows)."""
    if persist is None:
        persist = not dry_run
    windows = list_windows()
    cache_by_t = _cache_by_transcript()
    state = _load_state()
    for w in windows:
        resolve_window(w, cache_by_t, state)
    asyncio.run(_run_haiku(windows))

    new_state: dict = {}
    for w in windows:
        # Skip windows with no usable label (None, empty, or whitespace-only):
        # renaming to "" would wedge a permanently-blank window name.
        if not (w.label and w.label.strip()):
            # Carry a prior idle-skip entry forward on a transient error so one
            # bad pass doesn't force a needless re-Haiku next tick.
            prior = state.get(w.target)
            if w.error and isinstance(prior, dict):
                new_state[w.target] = prior
            continue
        new_state[w.target] = {
            "label": w.label,
            "summary": w.summary,
            "fingerprint": w.fingerprint,
            "source": w.source,
            "ts": _utcnow_iso(),
        }
        if dry_run or w.name == w.label:
            continue
        # ``--`` ends option parsing so a label starting with "-" is taken as
        # the literal new name, not an unknown tmux flag.
        rn = _tmux("rename-window", "-t", w.target, "--", w.label)
        if rn.returncode != 0:
            w.error = f"rename failed: {rn.stderr.strip()}"
            continue
        # Stop the shell/TUI from reverting the name.
        _tmux("set-window-option", "-t", w.target, "automatic-rename", "off")
        _tmux("set-window-option", "-t", w.target, "allow-rename", "off")

    if persist:
        # Keyed per window (session:index); windows gone this pass drop out, so
        # the state file can't grow without bound.
        _save_state(new_state)
    return windows


def _trunc(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def cmd_list(detail: bool = False) -> int:
    """Print one line per live session (no renames).

    Compact (default): ``session  <terse label>`` — the quick resume picker.
    ``detail``: ``session  <fuller one-sentence summary>`` — more context for
    deciding what to resume."""
    # persist=True: don't rename windows, but DO cache the fingerprints we just
    # computed so the next cron apply can idle-skip them.
    windows = apply(dry_run=True, persist=True)
    width = max((len(w.session) for w in windows), default=0)
    for w in sorted(windows, key=lambda x: x.session):
        text = (w.summary or w.label) if detail else w.label
        text = text or (f"<{w.error}>" if w.error else "<no summary>")
        tag = "" if w.source in (None, "cache") else f" [{w.source}]"
        print(f"{w.session.ljust(width)}  {_trunc(text, 140 if detail else 60)}{tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")
    p_apply = sub.add_parser("apply", help="rename every live tmux window")
    p_apply.add_argument("--dry-run", action="store_true", help="show, don't rename")
    sub.add_parser("list", help="compact session -> terse label, no renames")
    sub.add_parser("detail", help="session -> fuller one-sentence summary, no renames")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        return cmd_list(detail=False)
    if args.cmd == "detail":
        return cmd_list(detail=True)
    # default + "apply"
    dry = getattr(args, "dry_run", False)
    windows = apply(dry_run=dry)
    renamed = sum(1 for w in windows if w.label and w.name != w.label and not w.error)
    errs = sum(1 for w in windows if w.error)
    cached = sum(1 for w in windows if w.source == "cache")
    print(
        f"tmux_window_titles: {len(windows)} window(s); {renamed} "
        f"{'would be ' if dry else ''}renamed; {cached} from cache; {errs} errored"
        + (" (dry-run)" if dry else "")
    )
    for w in windows:
        if w.error:
            print(f"  ! {w.session}: {w.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
