#!/usr/bin/env python3
"""Helper that wraps codex-companion.mjs for long-running Codex sessions.

Replaces the brittle "run codex-companion in foreground and hope it
finishes before Bash times out at 10 min" pattern. Designed to be
invoked by the orchestrator (this conversation) via
``Bash(run_in_background=true, command="uv run python scripts/codex_task.py ...")``
— that's the only invocation pattern in the Claude Code harness that
delivers a real notification when Codex actually terminates. Wrapper
agents must NOT call this helper themselves (a subagent's
``run_in_background=true`` Bash returns immediately but its bg-completion
event has no listener once the subagent returns).

Every codex-companion subprocess (spawn/status/result/cancel) and the
task.py marker posts run with ``cwd=DISPATCH_ROOT`` — the MAIN checkout
root, resolved at import — never the caller's cwd. The detached ``codex
app-server`` worker inherits the spawn cwd and outlives the helper, so an
issue-worktree dispatch cwd pinned 11 terminal-task worktrees against the
stale-worktree sweep (2026-06-10 disk-full incident). Callers may invoke
this helper from any cwd; the pin is enforced here.

Lifecycle:

1. Spawn Codex with ``--background`` and capture the job-id from stdout.
2. Confirm the job-id is queryable via an immediate probe (catches
   spawn-success-but-job-unqueryable race).
3. Post ``epm:codex-task-spawned`` with the job-id to the task's
   events.jsonl (if ``--issue N`` given).
4. Poll ``codex-companion status <job-id> --json`` every
   ``--poll-interval-secs`` (default 30s) until terminal phase
   ({done, failed, cancelled}). Bail after ``--probe-error-cap`` (default
   10) consecutive probe failures with the last stderr captured. On
   terminal phase=cancelled, re-dispatch the same prompt up to
   ``--cancelled-retry-cap`` (default 2) times before posting
   ``epm:codex-task-failed`` — catches transient Codex-side
   cancellations.
5. Hard cap at ``--max-wait-secs`` (default 6h). On cap, force-cancel
   via ``codex-companion cancel`` and post ``epm:codex-task-failed``.
6. Fetch Codex stdout via ``codex-companion result <job-id>``; bail to
   ``epm:codex-task-failed`` if that call fails.
7. Validate the result-fetch returncode AND that the response JSON
   reports ``phase == "done"`` (not just present).
8. Post ``epm:codex-task-completed`` (phase=done) or
   ``epm:codex-task-failed`` (everything else).
9. Write Codex stdout to ``--output-file`` (or stdout if absent). If
   Codex already wrote a marker-formatted verdict to that SAME path
   mid-session (the twin-reviewer wrapper contract), the verdict file is
   preserved and the final chat message lands at
   ``<output-file>.final-msg.md`` instead — see
   ``_write_output_preserving_codex_artifact``.
10. Exit 0 on phase=done, non-zero otherwise.

Failure-mode coverage (every path posts a marker; helper never exits
silently):

- spawn failure (codex-companion CLI broken, plugin missing) → emit a
  marker with spawn-stderr in the note, exit 3.
- post-spawn probe fails (bad job-id, plugin upgrade race) → cancel +
  emit failure marker, exit 4.
- probe errors > cap → emit failure marker with last stderr, exit 5.
- hard cap hit → cancel + emit failure marker, exit 6.
- result-fetch non-zero → emit failure marker, exit 7.
- stall detected (phase==running but log STOPPED GROWING for
  > stall_detect_secs) → cancel + emit failure marker, exit 8. The
  detector is progress-aware: the stall timer resets whenever the log
  grows (mtime OR size increases), so a long-but-healthy run is never
  force-cancelled at the fixed window. This catches the "Codex process
  alive but model API hung" failure mode that ``codex-companion status``
  itself can't see (observed twice on 2026-05-20).
- SIGTERM/SIGINT → emit failure marker, best-effort cancel, exit 130/143.
- marker post fails → VERIFY whether the marker actually landed before
  retrying (``task.py post-marker`` commits the row BEFORE echoing the
  payload to stdout, so a post-commit echo failure exits nonzero after a
  successful append — a blind retry duplicated ``epm:codex-task-spawned``
  on task #537, 2026-06-10); if landed, treat as posted. Otherwise retry
  once, then drop the payload to
  ``tasks/_orphaned_markers/issue-<N>-<kind>-<job_id>-<ts>.json``,
  log to stderr (helper still exits with the right code).

Twin-agent marker-validation policy lives in the ORCHESTRATOR, not in
this helper. The helper just delivers Codex's stdout + a terminal-state
marker. The orchestrator reads the output and decides whether the
content marker (e.g. ``epm:code-review-codex v3``) is well-formed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Make task_workflow importable so we can route tasks/ artifacts through
# the canonical resolver (worktree-safe). Any path containing `tasks/`
# MUST go via `tasks_dir()` — see
# `tests/test_no_direct_task_path_construction.py`.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.task_workflow import list_events, tasks_dir  # noqa: E402


def _resolve_dispatch_root() -> Path:
    """MAIN-checkout root to use as the cwd for every codex-companion call.

    Dispatching from an issue-worktree cwd roots the DETACHED ``codex
    app-server`` worker in that worktree; those workers routinely outlive
    their companion task and pinned 11 terminal-task worktrees (~10-15G
    each) against the stale-worktree sweep until the 2026-06-10 disk-full
    incident. The repo-root dispatch rule previously existed only as prose
    in ``.claude/agents/codex-clean-result-critic.md``; resolving it HERE
    enforces it for every caller. ``git rev-parse --git-common-dir`` from a
    linked worktree returns the main checkout's ``.git``, so its parent is
    the main root even when this script copy lives in a worktree. Fail-soft:
    on any resolution failure, warn loudly and fall back to PROJECT_ROOT
    (no worse than the historical inherit-the-caller-cwd behavior).
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(
            f"WARN: dispatch-root resolution failed ({exc}); using {PROJECT_ROOT}", file=sys.stderr
        )
        return PROJECT_ROOT
    common = Path(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None
    if common is not None and common.name == ".git" and common.is_dir():
        return common.parent
    print(
        f"WARN: could not resolve main checkout root from {PROJECT_ROOT} "
        f"(rc={proc.returncode}); using {PROJECT_ROOT}",
        file=sys.stderr,
    )
    return PROJECT_ROOT


# Resolved ONCE at import (also keeps it ahead of any test monkeypatching of
# subprocess.run) and threaded as ``cwd=`` into every codex-companion spawn/
# status/result/cancel call AND the task.py marker posts, so neither the
# detached Codex worker nor the helper's subprocesses ever root themselves
# in an issue worktree.
DISPATCH_ROOT = _resolve_dispatch_root()

POLL_INTERVAL_SECS = 30
DEFAULT_MAX_WAIT_SECS = 6 * 3600  # 6h hard cap; force-cancel after.
DEFAULT_STALL_DETECT_SECS = 600  # 10 min of log silence → declare stuck.
PROBE_ERROR_CAP = 10  # consecutive failed probes before bailing
DEFAULT_CANCELLED_RETRY_CAP = 2  # re-dispatches on terminal phase=cancelled
# ONE auto-retry with backoff on the TRANSIENT-fail class (refs #579):
# ~10 codex-companion runtime incidents on 2026-06-09 (app-server exit 1,
# instant 0s failures, exit 4/5/8 probe-registry errors, stall
# force-cancels) all recovered via a manual re-dispatch — so the helper now
# re-dispatches once itself. Exit codes considered transient:
#   3 = spawn failure (app-server died / instant 0s failure)
#   4 = post-spawn probe failure (job-id race / probe-registry error)
#   5 = consecutive-probe-error cap (registry flake)
#   8 = stall force-cancel (model API hung; a fresh job usually proceeds)
# Deliberately NOT transient: 6 (hard-cap timeout — already ran max_wait;
# doubling wall time is the caller's call), 7 (result-fetch/output-write —
# local FS / fetch problem), and terminal phase=failed exit 1 (Codex itself
# reported failure, e.g. an AUP refusal — per CLAUDE.md the retry there
# needs a REPHRASED prompt, which only the orchestrator can compose).
DEFAULT_TRANSIENT_RETRY_CAP = 1
TRANSIENT_FAIL_EXIT_CODES = frozenset({3, 4, 5, 8})
# Backoff before the transient re-dispatch: 15s floor + up to 30s jitter
# (lets a flaky app-server / probe registry settle; jitter avoids
# synchronized re-spawns across parallel reviewer ensembles).
TRANSIENT_RETRY_BACKOFF_FLOOR_SECS = 15.0
TRANSIENT_RETRY_BACKOFF_JITTER_SECS = 30.0
TERMINAL_PHASES = {"done", "failed", "cancelled"}
# A Codex-written verdict file is identified by the ensemble marker tag the
# twin-reviewer wrapper contract requires of every verdict body (e.g.
# ``<!-- epm:interp-critique-codex v1 -->``). Used by the final-message
# write to avoid clobbering a verdict Codex already wrote to --output-file.
CODEX_ARTIFACT_SENTINEL = "<!-- epm:"
SPAWN_TIMEOUT_SECS = 90
STATUS_TIMEOUT_SECS = 60
RESULT_TIMEOUT_SECS = 120
CANCEL_TIMEOUT_SECS = 60
POST_MARKER_TIMEOUT_SECS = 60


# ──────────────────────────────────────────────────────────────────────
# Signal handling — never leave Codex orphaned on SIGTERM/SIGINT.
# ──────────────────────────────────────────────────────────────────────

_active_job_id: str | None = None
_active_companion: Path | None = None
_active_issue: int | None = None


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame) -> None:
        sig_name = signal.Signals(signum).name
        msg = (
            f"codex_task helper killed by {sig_name}; "
            f"job_id={_active_job_id or '<not-yet-assigned>'}"
        )
        print(f"ERROR: {msg}", file=sys.stderr)
        if _active_job_id and _active_companion is not None:
            try:
                subprocess.run(
                    ["node", str(_active_companion), "cancel", _active_job_id],
                    cwd=str(DISPATCH_ROOT),
                    capture_output=True,
                    timeout=CANCEL_TIMEOUT_SECS,
                )
            except Exception as exc:
                print(f"WARN: cancel-on-signal failed: {exc}", file=sys.stderr)
        if _active_issue is not None and _active_job_id:
            _post_marker(
                _active_issue,
                "epm:codex-task-failed",
                (
                    f"Codex job_id={_active_job_id} killed by {sig_name}. "
                    "Helper attempted cancel; verify manually with "
                    f"`node {_active_companion} status {_active_job_id}`."
                ),
            )
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ──────────────────────────────────────────────────────────────────────
# Codex-companion plumbing.
# ──────────────────────────────────────────────────────────────────────


def _resolve_companion() -> Path:
    """Find the highest-versioned codex-companion.mjs install."""
    plugin_root = Path(
        os.environ.get(
            "CLAUDE_PLUGIN_ROOT",
            Path.home() / ".claude/plugins/cache/openai-codex/codex",
        )
    )
    candidates = list(plugin_root.glob("*/scripts/codex-companion.mjs"))
    if not candidates:
        raise RuntimeError(
            f"codex-companion.mjs not found under {plugin_root}; "
            f"is the openai-codex plugin installed?"
        )

    def _vkey(p: Path) -> tuple[int, ...]:
        version_dir = p.parts[-3]
        parts = []
        for chunk in version_dir.split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    return max(candidates, key=_vkey)


def _marker_already_landed(issue: int, kind: str, note: str) -> bool:
    """Best-effort check whether a marker row already landed on the task's
    events.jsonl despite a nonzero ``task.py post-marker`` exit.

    ``task.py post-marker`` commits the row BEFORE echoing the payload to
    stdout, so a post-commit echo failure (BrokenPipeError on pipe teardown,
    subprocess timeout between commit and exit) returns rc!=0 AFTER the
    append+commit succeeded. A blind retry then duplicates the marker
    (incident #537, 2026-06-10: duplicate ``epm:codex-task-spawned`` on
    tasks/running/537/events.jsonl). Matching is on (kind, by=codex_task,
    EXACT note) over the last few rows only — every note this helper posts
    embeds the job_id (unique per attempt), so an exact match identifies
    this very post, not an earlier attempt's.

    Returns False on ANY read error — the caller then falls back to the
    pre-existing retry behavior (safe, at worst a duplicate).
    """
    try:
        events = list_events(issue)
    except Exception as exc:
        print(
            f"WARN: could not verify whether marker {kind} landed: {exc}",
            file=sys.stderr,
        )
        return False
    for row in events[-10:]:
        if row.get("kind") == kind and row.get("by") == "codex_task" and row.get("note") == note:
            return True
    return False


def _post_marker(issue: int, kind: str, note: str, version: int = 1) -> bool:
    """Post a marker via scripts/task.py. On a nonzero exit, VERIFY whether
    the marker actually landed (task.py commits before it echoes; the echo
    can fail post-commit) and skip the retry if it did. Otherwise retry
    once; on second failure, drop the payload to tasks/_orphaned_markers/
    so the user has a recovery path. Returns True if the marker posted
    (or was successfully archived)."""
    for attempt in (1, 2):
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/task.py",
                    "post-marker",
                    str(issue),
                    kind,
                    "--version",
                    str(version),
                    "--by",
                    "codex_task",
                    "--note",
                    note,
                ],
                cwd=DISPATCH_ROOT,
                capture_output=True,
                text=True,
                timeout=POST_MARKER_TIMEOUT_SECS,
            )
            if result.returncode == 0:
                return True
            print(
                f"WARN: post-marker attempt {attempt} for {kind} returned "
                f"rc={result.returncode}: {result.stderr[:500]}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"WARN: post-marker attempt {attempt} for {kind} raised: {exc}",
                file=sys.stderr,
            )
        # Verify-before-retry: a nonzero exit (or a raise, e.g. timeout) can
        # fire AFTER task.py committed the row — re-posting would duplicate.
        if _marker_already_landed(issue, kind, note):
            print(
                f"post-marker {kind} verified on events.jsonl despite the "
                "failed invocation; treating as posted (no retry).",
                file=sys.stderr,
            )
            return True
        if attempt == 1:
            time.sleep(2.0)

    # Both attempts failed — dump payload to a recovery file.
    ts = int(time.time())
    artifact_dir = tasks_dir() / "_orphaned_markers"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    job_tag = (_active_job_id or "no-job")[-12:]
    artifact = artifact_dir / f"issue-{issue}-{kind.replace(':', '_')}-{job_tag}-{ts}.json"
    try:
        artifact.write_text(
            json.dumps(
                {
                    "issue": issue,
                    "kind": kind,
                    "version": version,
                    "note": note,
                    "by": "codex_task",
                    "dropped_at_unix": ts,
                    "reason": "task.py post-marker failed twice; manual recovery needed.",
                },
                indent=2,
            )
        )
        print(
            f"ERROR: marker {kind} for issue #{issue} dropped to {artifact} for manual recovery.",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"FATAL: could not even write orphaned-marker artifact: {exc}",
            file=sys.stderr,
        )
    return False


def _spawn_codex(
    companion: Path,
    prompt: str,
    effort: str,
    write: bool,
) -> str:
    """Spawn Codex with ``--background``. Returns the job-id.

    The prompt is delivered via a temp file + the companion's native
    ``--prompt-file`` flag, NEVER as an argv element: a large composed
    prompt (e.g. an inlined diff) on the argv trips the kernel's
    per-argument size limit (~128KiB) and the spawn dies with
    ``OSError [Errno 7] Argument list too long`` (E2BIG) — observed on
    task #540 code-review round 1 with a 176K-char prompt (2026-06-09).
    The companion reads the file synchronously during this foreground
    invocation and embeds the prompt string in the stored job record
    (the detached task-worker re-reads that record, not the file), so
    the temp file is safe to delete as soon as the spawn call returns.
    """
    cmd = [
        "node",
        str(companion),
        "task",
        "--background",
        "--effort",
        effort,
    ]
    if write:
        cmd.append("--write")
    fd, prompt_tmp_path = tempfile.mkstemp(prefix="codex-task-prompt-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(prompt)
        # mkstemp yields an absolute path, so the companion's
        # `path.resolve(cwd, promptFile)` returns it unchanged.
        cmd.extend(["--prompt-file", prompt_tmp_path])
        # cwd pinned to the MAIN checkout root: the detached codex
        # app-server worker inherits THIS cwd and outlives the helper —
        # spawning from an issue-worktree cwd pinned terminal-task
        # worktrees forever (2026-06-10 disk-full incident).
        res = subprocess.run(
            cmd,
            cwd=str(DISPATCH_ROOT),
            capture_output=True,
            text=True,
            timeout=SPAWN_TIMEOUT_SECS,
        )
    finally:
        Path(prompt_tmp_path).unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"codex-companion task spawn failed (exit {res.returncode}). "
            f"stderr: {res.stderr[:1500]}"
        )
    match = re.search(r"task-[a-z0-9-]+", res.stdout)
    if not match:
        raise RuntimeError(
            f"could not extract job-id from spawn stdout. "
            f"stdout: {res.stdout[:500]} stderr: {res.stderr[:500]}"
        )
    return match.group(0)


def _probe_phase(companion: Path, job_id: str) -> tuple[str, str, str | None]:
    """Return (phase, error_or_summary, log_file_path) for the job.

    phase is one of:
        - "done", "failed", "cancelled" (terminal)
        - "running" (or similar non-terminal Codex phase)
        - "probe-error" (CLI returned non-zero or unparseable output)
        - "shape-error" (CLI returned JSON but it lacks the expected shape)

    log_file_path is the path Codex writes its turn-trace to (or None
    if the status response didn't include one). The main poll loop uses
    it to detect "Codex process alive but model API hung" (phase stays
    'running' indefinitely while the log file goes silent).
    """
    res = subprocess.run(
        ["node", str(companion), "status", job_id, "--json"],
        cwd=str(DISPATCH_ROOT),
        capture_output=True,
        text=True,
        timeout=STATUS_TIMEOUT_SECS,
    )
    if res.returncode != 0:
        return "probe-error", res.stderr[:500] or res.stdout[:500], None
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        return (
            "probe-error",
            f"json decode error: {exc}; stdout: {res.stdout[:300]}",
            None,
        )

    # The expected shape is {workspaceRoot, job: {... phase: str, ...}}.
    # If `job` is missing OR phase is missing, the job-id is bogus or the
    # CLI returned a list-style response — bail rather than poll forever.
    job = data.get("job")
    if not isinstance(job, dict):
        return (
            "shape-error",
            f"missing 'job' key in status response: {list(data.keys())}",
            None,
        )
    phase = job.get("phase")
    if not isinstance(phase, str):
        return (
            "shape-error",
            f"missing/non-string 'phase' in job: {list(job.keys())}",
            None,
        )
    log_file = job.get("logFile")
    return phase.lower(), "", log_file if isinstance(log_file, str) else None


def _log_progress_key(log_path: str | None) -> tuple[float, int] | None:
    """Return ``(mtime, size)`` for the Codex turn-trace log, or None if
    unreadable. Used by the (progress-aware) stall detector to catch
    "Codex process alive but model API hung" — phase stays 'running'
    while the log file goes completely silent for minutes.

    Tracking BOTH mtime and size makes the detector robust to filesystems
    with coarse mtime resolution (or mtime that doesn't bump on append):
    a healthy long Codex run keeps APPENDING to its log, so the file
    GROWS even when its mtime granularity hides sub-second writes. The
    poll loop resets the stall timer whenever EITHER component increases,
    so a long-but-healthy run is never force-cancelled at the fixed
    stall window — only a genuinely silent (non-growing, non-touched)
    log trips the detector. The absolute --max-wait-secs hard cap still
    bounds total wall time regardless of progress.
    """
    if not log_path:
        return None
    try:
        st = os.stat(log_path)
    except OSError:
        return None
    return st.st_mtime, st.st_size


def _key_advanced(
    current: tuple[float, int] | None,
    previous: tuple[float, int] | None,
) -> bool:
    """True if the log made progress since the last poll.

    Progress = the file first became readable (previous None, current
    not None) OR mtime increased OR size increased. Either component
    growing counts: a fresh append bumps size even when mtime resolution
    is too coarse to register the write.
    """
    if current is None:
        return False
    if previous is None:
        return True
    cur_mtime, cur_size = current
    prev_mtime, prev_size = previous
    return cur_mtime > prev_mtime or cur_size > prev_size


def _fetch_result(companion: Path, job_id: str) -> tuple[int, str, str]:
    """Fetch Codex's final output. Returns (returncode, stdout, stderr)."""
    res = subprocess.run(
        ["node", str(companion), "result", job_id],
        cwd=str(DISPATCH_ROOT),
        capture_output=True,
        text=True,
        timeout=RESULT_TIMEOUT_SECS,
    )
    return res.returncode, res.stdout, res.stderr


def _write_output_preserving_codex_artifact(
    output_file: Path,
    final_message: str,
    pre_spawn_key: tuple[float, int] | None,
) -> None:
    """Persist Codex's final chat message WITHOUT clobbering a verdict
    Codex already wrote to the same path mid-session.

    The four twin-reviewer wrappers instruct Codex to write its full
    marker-formatted verdict to ``--output-file`` DURING the session; the
    previously-unconditional final write then reduced a 12,474-char
    critique to Codex's 323-char closing chat message (task #604
    interpretation-critic round 1, 2026-06-11 — recovered only via the
    Codex session rollout's apply_patch payload). Preserve the existing
    file and divert the final message to ``<output-file>.final-msg.md``
    only when BOTH hold:

    - the file ADVANCED (was created, or mtime/size grew) since this
      attempt spawned — so a stale file left by a previous reviewer round
      or by a failed earlier attempt (the transient-retry path, #579,
      re-enters ``_run_one_attempt`` with the same ``--output-file``)
      never triggers preservation; and
    - the content carries the ``<!-- epm:`` marker tag the wrapper
      contract requires of verdicts — keying on the marker rather than
      size alone avoids preserving a half-written file.

    Every other flow — including all prompts where this helper is the
    ONLY producer of the output file — keeps the historical behavior
    exactly: the final message lands at ``--output-file``. Nothing is
    lost in the preservation branch either: the final message is still
    on disk, in the sidecar.
    """
    existing = ""
    current_key = _log_progress_key(str(output_file))
    if current_key is not None and _key_advanced(current_key, pre_spawn_key):
        try:
            existing = output_file.read_text()
        except OSError as exc:
            print(
                f"WARN: could not read pre-existing {output_file} ({exc}); overwriting.",
                file=sys.stderr,
            )
    if existing.strip() and CODEX_ARTIFACT_SENTINEL in existing and existing != final_message:
        sidecar = output_file.with_name(output_file.name + ".final-msg.md")
        sidecar.write_text(final_message)
        print(
            f"Output file {output_file} already written by Codex mid-session "
            f"({len(existing)} chars, epm marker present); preserved it and wrote "
            f"the final chat message ({len(final_message)} chars) to {sidecar}.",
            file=sys.stderr,
        )
        return
    output_file.write_text(final_message)
    print(
        f"Codex output written to {output_file} ({len(final_message)} chars).",
        file=sys.stderr,
    )


# ──────────────────────────────────────────────────────────────────────
# Main lifecycle.
# ──────────────────────────────────────────────────────────────────────


def _fail(
    issue: int | None,
    job_id: str | None,
    note: str,
    exit_code: int,
) -> int:
    if issue is not None:
        full_note = note
        if job_id:
            full_note = f"job_id={job_id}: {note}"
        _post_marker(issue, "epm:codex-task-failed", full_note)
    print(f"ERROR: {note}", file=sys.stderr)
    return exit_code


class AttemptResult:
    """Outcome of a single ``_run_one_attempt`` lifecycle.

    ``kind`` is one of:
        - "done"      — Codex finished successfully; completed marker was
                        already posted inside the attempt; exit_code == 0.
        - "cancelled" — Codex ended in terminal phase=cancelled. RETRYABLE:
                        the failure marker is NOT posted by the attempt, so
                        the caller can re-dispatch. exit_code == 1.
        - "fail"      — any non-retryable failure (spawn, probe, probe-error
                        cap, stall, hard-cap timeout, result-fetch,
                        output-write, terminal phase=failed). The failure
                        marker is NOT posted by the attempt; the caller posts
                        it once via ``_fail``.

    For "cancelled" and "fail", ``note`` + ``exit_code`` + ``job_id`` carry
    everything the caller needs to either retry or post the terminal marker.
    """

    def __init__(
        self,
        kind: str,
        exit_code: int,
        note: str = "",
        job_id: str | None = None,
    ) -> None:
        assert kind in {"done", "cancelled", "fail"}, kind
        self.kind = kind
        self.exit_code = exit_code
        self.note = note
        self.job_id = job_id


def _poll_until_terminal(
    companion: Path,
    job_id: str,
    args,
    log_path: str | None,
    started: float,
) -> str | AttemptResult:
    """Poll ``status`` until the job reaches a terminal phase.

    Returns the terminal phase string (one of {done, failed, cancelled})
    on success, OR an ``AttemptResult`` "fail" when a non-cancellation
    bail fires (probe-error cap exit 5, hard-cap timeout exit 6, stall
    exit 8). The caller force-cancels are handled here before returning.

    The stall detector is progress-aware: the timer resets whenever the
    Codex log GROWS (mtime OR size increases), so a long-but-healthy run
    is never force-cancelled at the fixed ``--stall-detect-secs`` window.
    """
    consecutive_probe_errors = 0
    last_probe_err = ""
    # Stall-detector state: track when the Codex log file last advanced.
    last_log_key = _log_progress_key(log_path)
    last_log_change_ts = time.time()
    while True:
        elapsed = time.time() - started
        if elapsed > args.max_wait_secs:
            _best_effort_cancel(companion, job_id)
            return AttemptResult(
                "fail",
                6,
                (f"timed out after {int(elapsed)}s (cap {args.max_wait_secs}s); force-cancelled."),
                job_id,
            )

        time.sleep(args.poll_interval_secs)
        phase, err, probe_log_path = _probe_phase(companion, job_id)
        if probe_log_path is not None:
            log_path = probe_log_path  # refresh in case Codex updated it
        if phase in TERMINAL_PHASES:
            print(
                f"codex-task-{phase}: {job_id} after {int(elapsed)}s",
                file=sys.stderr,
            )
            return phase
        if phase in {"probe-error", "shape-error"}:
            consecutive_probe_errors += 1
            last_probe_err = err
            print(
                f"WARN: probe {phase} at t={int(elapsed)}s "
                f"({consecutive_probe_errors}/{args.probe_error_cap}): {err[:200]}",
                file=sys.stderr,
            )
            if consecutive_probe_errors >= args.probe_error_cap:
                _best_effort_cancel(companion, job_id)
                return AttemptResult(
                    "fail",
                    5,
                    (
                        f"{consecutive_probe_errors} consecutive probe failures; "
                        f"last error: {last_probe_err[:500]}"
                    ),
                    job_id,
                )
            continue
        # Non-terminal, non-error phase (e.g. running, queued) — reset error count.
        consecutive_probe_errors = 0

        # Stall detector: Codex process alive + phase==running but no log
        # activity for >stall_detect_secs => model API hung. This is the
        # failure mode that bit us twice on 2026-05-20 — codex-companion
        # status reports "running" while the actual Codex turn has been
        # silent for hours. Progress-aware: reset the timer whenever the
        # log GROWS (mtime OR size), so a long-but-healthy run is not
        # force-cancelled at the fixed window.
        if args.stall_detect_secs > 0:
            now = time.time()
            cur_log_key = _log_progress_key(log_path)
            if _key_advanced(cur_log_key, last_log_key):
                last_log_key = cur_log_key
                last_log_change_ts = now
            stall_age = now - last_log_change_ts
            if stall_age > args.stall_detect_secs:
                _best_effort_cancel(companion, job_id)
                return AttemptResult(
                    "fail",
                    8,
                    (
                        f"stall detected: phase=running but log file untouched "
                        f"for {int(stall_age)}s (cap {args.stall_detect_secs}s) "
                        f"at t={int(elapsed)}s. Force-cancelled. Log: {log_path}"
                    ),
                    job_id,
                )


def _run_one_attempt(companion: Path, prompt: str, args, write: bool) -> AttemptResult:
    """Run one full Codex lifecycle: spawn -> confirm-probe -> poll ->
    fetch-result -> write-output.

    Posts ``epm:codex-task-spawned`` (per attempt) and, on success,
    ``epm:codex-task-completed``. Does NOT post ``epm:codex-task-failed``
    for any failure path — that decision belongs to the caller so it can
    re-dispatch on a retryable terminal phase=cancelled. Returns an
    ``AttemptResult`` describing the outcome.

    The stall detector is progress-aware: the stall timer resets whenever
    the Codex turn-trace log GROWS (mtime OR size increases), so a long
    but healthy run is never force-cancelled at the fixed
    ``--stall-detect-secs`` window. The absolute ``--max-wait-secs`` hard
    cap still bounds total wall time regardless of progress.
    """
    global _active_job_id

    # Snapshot the output-file state BEFORE Codex spawns: the final-message
    # write uses it to distinguish "Codex wrote --output-file during THIS
    # attempt" (preserve it — twin-reviewer verdict contract) from a stale
    # file left by a previous round / failed attempt (overwrite as always).
    pre_spawn_output_key = (
        _log_progress_key(str(args.output_file)) if args.output_file is not None else None
    )

    # Spawn.
    try:
        job_id = _spawn_codex(companion, prompt, args.effort, write)
    except Exception as exc:
        return AttemptResult("fail", 3, f"spawn: {exc}", None)
    _active_job_id = job_id
    print(f"codex-task-spawned: {job_id}", file=sys.stderr)

    # Confirm the job-id is queryable (immediate probe; catches the
    # spawn-success-but-bad-job-id race).
    confirm_phase, confirm_err, log_path = _probe_phase(companion, job_id)
    if confirm_phase in {"probe-error", "shape-error"}:
        _best_effort_cancel(companion, job_id)
        return AttemptResult(
            "fail",
            4,
            f"post-spawn probe failed ({confirm_phase}): {confirm_err}",
            job_id,
        )

    if args.issue is not None:
        _post_marker(
            args.issue,
            "epm:codex-task-spawned",
            (
                f"Codex job_id={job_id} effort={args.effort} write={write} "
                f"poll_interval={args.poll_interval_secs}s "
                f"max_wait={args.max_wait_secs}s "
                f"probe_error_cap={args.probe_error_cap} "
                f"stall_detect={args.stall_detect_secs}s"
            ),
        )

    # Poll until terminal (or a non-cancellation bail).
    started = time.time()
    poll_outcome = _poll_until_terminal(companion, job_id, args, log_path, started)
    if isinstance(poll_outcome, AttemptResult):
        return poll_outcome  # probe-error cap / stall / hard-cap timeout
    phase = poll_outcome  # one of {done, failed, cancelled}

    # Fetch result.
    rc, stdout, stderr = _fetch_result(companion, job_id)
    if rc != 0:
        return AttemptResult(
            "fail",
            7,
            (
                f"result-fetch failed (exit {rc}). "
                f"stderr: {stderr[:500]}; stdout (truncated): {stdout[:200]}"
            ),
            job_id,
        )

    # Write output before posting terminal marker — so even if the marker
    # post fails, the orchestrator has the Codex output on disk. The write
    # preserves a marker-formatted verdict Codex already wrote to the same
    # path mid-session (final message then lands in the .final-msg.md
    # sidecar) — see _write_output_preserving_codex_artifact.
    if args.output_file is not None:
        try:
            _write_output_preserving_codex_artifact(args.output_file, stdout, pre_spawn_output_key)
        except Exception as exc:
            return AttemptResult(
                "fail",
                7,
                f"could not write output to {args.output_file}: {exc}",
                job_id,
            )
    else:
        sys.stdout.write(stdout)

    elapsed = int(time.time() - started)
    if phase == "done":
        if args.issue is not None:
            _post_marker(
                args.issue,
                "epm:codex-task-completed",
                f"Codex job_id={job_id} phase=done after {elapsed}s.",
            )
        return AttemptResult("done", 0, "", job_id)

    # phase == cancelled — terminal, RETRYABLE (caller decides).
    if phase == "cancelled":
        return AttemptResult(
            "cancelled",
            1,
            (
                f"terminal phase=cancelled after {elapsed}s. "
                f"Inspect: node {companion} status {job_id}"
            ),
            job_id,
        )

    # phase == failed — terminal, NOT retryable.
    return AttemptResult(
        "fail",
        1,
        (f"terminal phase={phase} after {elapsed}s. Inspect: node {companion} status {job_id}"),
        job_id,
    )


def _best_effort_cancel(companion: Path, job_id: str) -> None:
    """Cancel a Codex job, swallowing any error. Used on every bail path
    where leaving the job alive would orphan a Codex process; the caller
    has already decided to abort, so a cancel failure here must not mask
    the original failure."""
    try:
        subprocess.run(
            ["node", str(companion), "cancel", job_id],
            cwd=str(DISPATCH_ROOT),
            capture_output=True,
            timeout=CANCEL_TIMEOUT_SECS,
        )
    except Exception as exc:
        print(f"WARN: best-effort cancel of {job_id} failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, default=None)
    parser.add_argument(
        "--effort",
        default="xhigh",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write",
        action="store_true",
        default=None,
        help="Grant Codex write access (default).",
    )
    write_group.add_argument(
        "--no-write",
        action="store_false",
        dest="write",
        help="Run Codex read-only (no file mutations).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Write Codex stdout here; default = print to this script's stdout.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read Codex prompt from this file; default = read from stdin.",
    )
    parser.add_argument("--prompt", default=None, help="Inline Codex prompt.")
    parser.add_argument(
        "--max-wait-secs",
        type=int,
        default=DEFAULT_MAX_WAIT_SECS,
        help=f"Hard cap; force-cancel after. Default {DEFAULT_MAX_WAIT_SECS}s.",
    )
    parser.add_argument(
        "--poll-interval-secs",
        type=int,
        default=POLL_INTERVAL_SECS,
    )
    parser.add_argument(
        "--probe-error-cap",
        type=int,
        default=PROBE_ERROR_CAP,
        help=(
            "Consecutive probe failures before bailing with epm:codex-task-failed. "
            f"Default {PROBE_ERROR_CAP} (≈ {PROBE_ERROR_CAP * POLL_INTERVAL_SECS}s)."
        ),
    )
    parser.add_argument(
        "--stall-detect-secs",
        type=int,
        default=DEFAULT_STALL_DETECT_SECS,
        help=(
            "Force-cancel the Codex task if its turn-trace log file stops "
            "growing for this many seconds while phase==running. The detector "
            "is progress-aware: the timer resets whenever the log GROWS "
            "(mtime OR size increases), so a long-but-healthy run is never "
            "force-cancelled at the fixed window. This catches the 'Codex "
            "process alive but model API hung' failure mode that "
            "codex-companion status itself can't see. Set to 0 to disable. "
            f"Default {DEFAULT_STALL_DETECT_SECS}s "
            f"({DEFAULT_STALL_DETECT_SECS // 60}min)."
        ),
    )
    parser.add_argument(
        "--cancelled-retry-cap",
        type=int,
        default=DEFAULT_CANCELLED_RETRY_CAP,
        help=(
            "Re-dispatch the same prompt this many times when a job ends in "
            "terminal phase=cancelled, before posting epm:codex-task-failed. "
            "Catches transient Codex-side cancellations. Set to 0 to disable "
            f"(fail on the first cancellation). Default {DEFAULT_CANCELLED_RETRY_CAP}."
        ),
    )
    parser.add_argument(
        "--transient-retry-cap",
        type=int,
        default=DEFAULT_TRANSIENT_RETRY_CAP,
        help=(
            "Re-dispatch the same prompt this many times (with a "
            f"{TRANSIENT_RETRY_BACKOFF_FLOOR_SECS:.0f}-"
            f"{TRANSIENT_RETRY_BACKOFF_FLOOR_SECS + TRANSIENT_RETRY_BACKOFF_JITTER_SECS:.0f}s "
            "jittered backoff) when an attempt fails with a TRANSIENT exit "
            f"code ({sorted(TRANSIENT_FAIL_EXIT_CODES)}: spawn / post-spawn "
            "probe / probe-error cap / stall force-cancel), before posting "
            "epm:codex-task-failed. Hard-cap timeouts (6), result-fetch "
            "failures (7), and terminal phase=failed are NOT retried. Set to "
            f"0 to disable. Default {DEFAULT_TRANSIENT_RETRY_CAP} (refs #579)."
        ),
    )
    args = parser.parse_args()

    # Default for --write is True (grant write) unless --no-write was passed.
    write = True if args.write is None else args.write

    global _active_companion, _active_issue, _active_job_id

    _install_signal_handlers()
    _active_issue = args.issue

    # Resolve prompt.
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        prompt = args.prompt_file.read_text()
    else:
        prompt = sys.stdin.read()
    if not prompt.strip():
        return _fail(args.issue, None, "empty Codex prompt", 2)

    try:
        companion = _resolve_companion()
    except Exception as exc:
        return _fail(args.issue, None, f"resolve_companion: {exc}", 3)
    _active_companion = companion
    print(f"codex-companion: {companion}", file=sys.stderr)

    # Run the lifecycle with two independent retry budgets:
    # - terminal phase=cancelled → re-dispatch up to --cancelled-retry-cap
    #   times (transient Codex-side cancellations);
    # - TRANSIENT fail exit codes (TRANSIENT_FAIL_EXIT_CODES: spawn /
    #   post-spawn probe / probe-error cap / stall) → re-dispatch up to
    #   --transient-retry-cap times with a jittered backoff (refs #579 —
    #   ~10 such incidents on 2026-06-09, every one recovered by a manual
    #   re-dispatch).
    # Everything else (hard-cap timeout, result-fetch/output-write,
    # terminal phase=failed) fails immediately.
    cancelled_redispatches = 0
    transient_redispatches = 0
    attempt = 0
    while True:
        attempt += 1
        result = _run_one_attempt(companion, prompt, args, write)
        if result.kind == "cancelled" and cancelled_redispatches < max(0, args.cancelled_retry_cap):
            cancelled_redispatches += 1
            print(
                f"WARN: Codex job_id={result.job_id} ended phase=cancelled "
                f"(cancelled re-dispatch {cancelled_redispatches}/"
                f"{args.cancelled_retry_cap}, attempt {attempt}); re-dispatching.",
                file=sys.stderr,
            )
            continue
        if (
            result.kind == "fail"
            and result.exit_code in TRANSIENT_FAIL_EXIT_CODES
            and transient_redispatches < max(0, args.transient_retry_cap)
        ):
            transient_redispatches += 1
            delay = TRANSIENT_RETRY_BACKOFF_FLOOR_SECS + random.uniform(
                0.0, TRANSIENT_RETRY_BACKOFF_JITTER_SECS
            )
            print(
                f"WARN: transient Codex failure (exit {result.exit_code}: "
                f"{result.note[:200]}) — re-dispatching in {delay:.0f}s "
                f"(transient retry {transient_redispatches}/"
                f"{args.transient_retry_cap}, attempt {attempt}; refs #579).",
                file=sys.stderr,
            )
            time.sleep(delay)
            continue
        break

    if result.kind == "done":
        return 0

    # cancelled / transient (cap exhausted) or non-retryable fail — post the
    # terminal failure marker once.
    note = result.note
    if result.kind == "cancelled" and cancelled_redispatches:
        note = f"{note} (exhausted {cancelled_redispatches} re-dispatch(es))"
    elif result.kind == "fail" and transient_redispatches:
        note = f"{note} (exhausted {transient_redispatches} transient re-dispatch(es))"
    return _fail(args.issue, result.job_id, note, result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
