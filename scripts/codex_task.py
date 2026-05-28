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
9. Write Codex stdout to ``--output-file`` (or stdout if absent).
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
- marker post fails → retry once, drop payload to
  ``tasks/<N>/artifacts/codex-task-orphaned-marker-<job_id>-<ts>.json``,
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
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Make task_workflow importable so we can route tasks/ artifacts through
# the canonical resolver (worktree-safe). PROJECT_ROOT itself is still
# used as the git cwd for subprocess calls into the local checkout, but
# any path containing `tasks/` MUST go via `tasks_dir()` instead — see
# `tests/test_no_direct_task_path_construction.py`.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.task_workflow import tasks_dir  # noqa: E402

POLL_INTERVAL_SECS = 30
DEFAULT_MAX_WAIT_SECS = 6 * 3600  # 6h hard cap; force-cancel after.
DEFAULT_STALL_DETECT_SECS = 600  # 10 min of log silence → declare stuck.
PROBE_ERROR_CAP = 10  # consecutive failed probes before bailing
DEFAULT_CANCELLED_RETRY_CAP = 2  # re-dispatches on terminal phase=cancelled
TERMINAL_PHASES = {"done", "failed", "cancelled"}
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


def _post_marker(issue: int, kind: str, note: str, version: int = 1) -> bool:
    """Post a marker via scripts/task.py. Retry once on failure. On second
    failure, drop the payload to artifacts/ so the user has a recovery path.
    Returns True if the marker posted (or was successfully archived)."""
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
                cwd=PROJECT_ROOT,
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
    """Spawn Codex with ``--background``. Returns the job-id."""
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
    cmd.append(prompt)
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=SPAWN_TIMEOUT_SECS,
    )
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
        capture_output=True,
        text=True,
        timeout=RESULT_TIMEOUT_SECS,
    )
    return res.returncode, res.stdout, res.stderr


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
    # post fails, the orchestrator has the Codex output on disk.
    if args.output_file is not None:
        try:
            args.output_file.write_text(stdout)
            print(
                f"Codex output written to {args.output_file} ({len(stdout)} chars).",
                file=sys.stderr,
            )
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

    # Run the lifecycle, re-dispatching on terminal phase=cancelled up to
    # --cancelled-retry-cap times before posting epm:codex-task-failed.
    # Non-cancelled failures (spawn, probe-error cap, stall, hard cap,
    # result-fetch, terminal phase=failed) fail immediately — they are not
    # the transient-cancellation class.
    max_attempts = max(1, args.cancelled_retry_cap + 1)
    result: AttemptResult | None = None
    for attempt in range(1, max_attempts + 1):
        result = _run_one_attempt(companion, prompt, args, write)
        if result.kind != "cancelled":
            break
        # Terminal cancelled — retry unless we've exhausted the cap.
        if attempt < max_attempts:
            print(
                f"WARN: Codex job_id={result.job_id} ended phase=cancelled "
                f"(attempt {attempt}/{max_attempts}); re-dispatching.",
                file=sys.stderr,
            )

    assert result is not None  # loop runs at least once
    if result.kind == "done":
        return 0

    # cancelled (cap exhausted) or fail — post the terminal failure marker once.
    note = result.note
    if result.kind == "cancelled" and args.cancelled_retry_cap > 0:
        note = f"{note} (exhausted {args.cancelled_retry_cap} re-dispatch(es))"
    return _fail(args.issue, result.job_id, note, result.exit_code)


if __name__ == "__main__":
    sys.exit(main())
