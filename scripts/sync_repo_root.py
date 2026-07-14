#!/usr/bin/env python3
"""sync_repo_root.py — single-flight recovery sync for the SHARED repo root.

Any session whose push to the shared repo root is rejected (or that finds the
root diverged from ``origin/main``) runs this helper INSTEAD of hand-rolling a
pull-rebase recovery loop. It closes the three blockers observed in the
2026-07-02 divergence incident (582/226 commits diverged for ~3h):

  (a) untracked-collision checkout failures — a pre-sweep enumerates untracked
      files that a rebase onto ``origin/main`` would overwrite; byte-identical
      copies of the origin blob are removed (the pull rematerializes the same
      bytes, tracked), anything else is RESCUED to a dated directory under
      ``~/.task-workflow/root-sync-rescue/`` — non-identical data is never
      deleted;
  (b) index.lock contention between concurrent recovery loops — a non-blocking
      flock on ``~/.task-workflow/root-sync.lock`` makes recovery single-flight
      (a second caller exits 0 immediately with an "in flight" message), and a
      bounded second flock on the task-workflow lock excludes ``task.py``
      writers for the mutation window;
  (c) stranded autostashes — ``stash@{n}: autostash`` entries (from this pull
      or previous crashed loops) get a rescue patch written first, their
      unmerged paths cleared (path-scoped, only paths attributable to that
      autostash), then ``git stash pop`` if ``git apply --check`` is clean;
      a conflicting entry is KEPT and reported, never dropped.

Usage::

    uv run python scripts/sync_repo_root.py [--dry-run] [--no-push] [--json]
                                            [--timeout-s N] [--repo PATH]

Exit codes::

    0  synced / already-in-sync / another-sync-in-flight (all benign; the
       --json report's ``state`` field distinguishes: synced | already |
       in-flight | dry-run — exit 0 does NOT by itself mean "my push landed")
    2  aborted on a genuine content conflict (clean abort; conflicted paths
       named; swept files restored; manual scratch-worktree recipe printed)
    3  push failed after the one retry
    4  a bounded git subprocess timed out (rebase aborted cleanly)
    5  precondition failure (HEAD not main, fresh rebase/merge husk present,
       interrupted ``git am`` state, index.lock persistently held,
       task-workflow lock held past the bound)
    6  unexpected error (fail loud; swept files restored from the journal; a
       failure inside the restore itself also routes here, report emitted)

Safety invariants (binding; pinned by tests/test_sync_repo_root.py):

  * NEVER a repo-root ``git reset --hard`` / ``git clean -f`` / full-tree
    ``git checkout .`` / ``git restore .`` / ``git stash drop`` — the
    ``git()`` wrapper deny-lists these argv shapes and raises at call time.
  * Never delete non-identical untracked data — rescue dir only, and
    byte-identity is re-checked (re-hash) immediately before every removal;
    a file touched within ``EPM_ROOT_SYNC_FRESH_S`` (default 60s) is rescued
    regardless of hash (it may be mid-write by a live session).
  * Journal-before-action abort-restore: every sweep action is appended
    DURABLY (O_APPEND + fsync) to ``<rescue-dir>/sweep-journal.jsonl`` BEFORE
    its remove/move executes, and marked applied after — a mid-sweep SIGKILL
    leaves a mechanically restorable record (``load_sweep_journal``
    reconstructs it from the journal alone). ``sweep-manifest.json`` stays as
    the consolidated rollup, written atomically after each sweep pass. On any
    post-sweep failure exit (2/3/4/6) the restore is driven from the journal
    (union with the in-memory ledger): rescued files are moved back (kept in
    the rescue dir + reported if the path is now occupied) and
    identical-removed files are rematerialized from the RECORDED origin blob
    sha via ``git cat-file blob`` (a plumbing write — no index staging; the
    recorded sha, NOT ``origin/main:<path>``, because the ref may move
    between sweep and restore). A failure inside the restore itself never
    loses the report — it is contained, reported, and routed to exit 6 with
    the rescue dir + journal retained on disk.
  * The per-run rescue dir is allocated EXCLUSIVELY —
    ``<UTC %Y%m%dT%H%M%SZ>-<pid>[-<k>]`` via ``mkdir(exist_ok=False)`` with a
    bounded retry on collision — so a same-second sequential/concurrent run
    can never reuse a prior run's rescue dir (reuse would let
    ``shutil.move`` replace a prior rescue copy and clobber its manifest).
  * Fail loud; no silent husks; a young (< ``EPM_ROOT_SYNC_HUSK_AGE_S``)
    rebase/merge husk or a persistent index.lock is reported and exits 5 —
    never deleted — UNLESS the husk is at least ``EPM_ROOT_SYNC_HUSK_MIN_AGE_S``
    (default 600s) old AND a COMPLETED, budget-bounded /proc scan
    (``_probe_git_liveness``) proves no live git process is attributable to
    this repo, in which case the EXISTING stale handling applies (abort /
    archive-aside — still never deleted). An uncertain, timed-out,
    budget-exhausted, or disabled (``EPM_ROOT_SYNC_HUSK_PROBE=0``) probe
    keeps the exit-5 refusal, with the probe evidence appended to the
    message. A STALE husk is git-aborted; when git itself cannot abort
    it (the head-name-less shape a crashed autostash-pull leaves) its
    autostash is rescued to the stash list and the husk dir is ARCHIVED to
    the rescue root — never deleted — with a rescue failure blocking the
    move; an interrupted ``git am`` session (``rebase-apply/applying``) is
    refused (exit 5), never touched.

Residual risk (documented, not closed here): a session running a direct
``git commit`` on the root (not via ``task.py``) is NOT excluded by the
task-workflow lock; a commit landing in an inter-step rebase window goes onto
the detached rebase HEAD and is orphaned-but-reflog-recoverable by a later
abort (``git reflog`` on the root recovers it).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

# Make the package importable without `uv run` plumbing.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT_DIR = _HERE.parent
_SRC = _REPO_ROOT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space import task_workflow  # noqa: E402

# ─── Exit codes ──────────────────────────────────────────────────────────────

EXIT_OK = 0
EXIT_CONFLICT = 2
EXIT_PUSH_FAILED = 3
EXIT_TIMEOUT = 4
EXIT_PRECONDITION = 5
EXIT_UNEXPECTED = 6

# ─── Tunables (env-overridable; tests monkeypatch the module attributes) ─────

ROOT_SYNC_LOCK = Path(
    os.environ.get("EPM_ROOT_SYNC_LOCK", str(Path.home() / ".task-workflow" / "root-sync.lock"))
)
RESCUE_ROOT = Path(
    os.environ.get(
        "EPM_ROOT_SYNC_RESCUE_ROOT", str(Path.home() / ".task-workflow" / "root-sync-rescue")
    )
)
INDEX_LOCK_WAIT_S = 60.0
INDEX_LOCK_POLL_S = 2.0

IN_FLIGHT_MSG = (
    "another sync in flight — your push has NOT landed; re-run after the in-flight sync completes"
)

# Exact wording verified empirically (plan §12 item 1).
COLLISION_STDERR_NEEDLE = "untracked working tree files would be overwritten by"
# Emitted on STDERR with pull rc=0 (plan §12 item 4) — never key on exit code.
AUTOSTASH_CONFLICT_NEEDLE = "Applying autostash resulted in conflicts"
# Exact wording verified on git 2.34.1 (`strings $(command -v git)` + live
# two-refspec repro, #1044 §3): builtin/pull.c die()s this BEFORE any rebase
# starts, when a concurrent fetch left >1 for-merge FETCH_HEAD entries.
# Substring without the `fatal: ` wrapper and trailing period — still the
# exact phrase; robust to die()'s prefix.
MULTI_BRANCH_STDERR_NEEDLE = "Cannot rebase onto multiple branches"

SCRATCH_WORKTREE_RECIPE = (
    "Manual next step (the recovery that resolved 6c1b3fadf7):\n"
    "  git worktree add --detach <path> origin/main\n"
    "  # resolve the conflict there, commit, then:\n"
    "  git push origin HEAD:main"
)


def _timeout_s_default() -> float:
    """Per-subprocess bound for fetch/pull/push (``EPM_ROOT_SYNC_TIMEOUT_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_TIMEOUT_S", "600"))


def _lock2_wait_s() -> float:
    """Bound on waiting for the task-workflow lock (``EPM_ROOT_SYNC_LOCK2_WAIT_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_LOCK2_WAIT_S", "120"))


def _husk_age_s() -> float:
    """Age past which a rebase/merge husk counts as stale (``EPM_ROOT_SYNC_HUSK_AGE_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_HUSK_AGE_S", "3600"))


def _fresh_s() -> float:
    """Mtime freshness window under which a collision is rescued, never removed
    (``EPM_ROOT_SYNC_FRESH_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_FRESH_S", "60"))


def _husk_probe_enabled() -> bool:
    """Liveness-probe kill switch (``EPM_ROOT_SYNC_HUSK_PROBE`` != "0")."""
    return os.environ.get("EPM_ROOT_SYNC_HUSK_PROBE", "1") != "0"


def _husk_min_age_s() -> float:
    """Age floor below which a young husk is NEVER liveness-downgraded
    (``EPM_ROOT_SYNC_HUSK_MIN_AGE_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_HUSK_MIN_AGE_S", "600"))


def _probe_timeout_s() -> float:
    """Per-candidate bound for the probe's rev-parse attribution subprocess
    (``EPM_ROOT_SYNC_PROBE_TIMEOUT_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_PROBE_TIMEOUT_S", "5.0"))


def _probe_budget_s() -> float:
    """TOTAL wall-clock budget for one liveness scan
    (``EPM_ROOT_SYNC_PROBE_BUDGET_S``). Exhaustion ⇒ verdict "uncertain"
    (scan incomplete) — never "none"."""
    return float(os.environ.get("EPM_ROOT_SYNC_PROBE_BUDGET_S", "30.0"))


def _retry_sleep_s() -> float:
    """Sleep before the one multiple-branches retry (``EPM_ROOT_SYNC_RETRY_SLEEP_S``)."""
    return float(os.environ.get("EPM_ROOT_SYNC_RETRY_SLEEP_S", "2.0"))


def _rescue_timestamp() -> str:
    """UTC timestamp component of a rescue-dir name (monkeypatchable in tests)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def allocate_rescue_dir() -> Path:
    """Allocate a fresh, EXCLUSIVE per-run rescue directory.

    Name shape ``<UTC %Y%m%dT%H%M%SZ>-<pid>[-<k>]`` — the timestamp keeps the
    plan §4.3 ``<UTC-ts>*`` shape; the pid suffix plus ``mkdir(exist_ok=False)``
    make allocation exclusive, so a same-second sequential/concurrent run can
    never reuse a prior run's dir (reuse would let ``shutil.move`` replace a
    prior rescue copy and clobber its manifest — round-1 concern
    ``rescue-dir-nonexclusive-overwrite``). On collision a bounded retry
    appends a counter; exhaustion raises (fail loud, never silently reuse).
    """
    ts = _rescue_timestamp()
    base = f"{ts}-{os.getpid()}"
    for k in range(100):
        candidate = RESCUE_ROOT / (base if k == 0 else f"{base}-{k}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(
        f"could not allocate an exclusive rescue dir under {RESCUE_ROOT} "
        f"after 100 attempts (base {base!r})"
    )


class RescueDir:
    """Lazy, exclusive per-run rescue-dir handle.

    Allocation is deferred to first use so dry-run / already-in-sync /
    no-collision runs create nothing on disk; once allocated the same dir is
    reused for every sweep pass of the run (initial, fallback, push-retry).
    """

    def __init__(self) -> None:
        self.allocated: Path | None = None

    def get(self) -> Path:
        if self.allocated is None:
            self.allocated = allocate_rescue_dir()
        return self.allocated


# ─── Errors ──────────────────────────────────────────────────────────────────


class BannedGitInvocationError(RuntimeError):
    """A deny-listed destructive git invocation was attempted (bug — fail loud)."""


class SyncAbortError(Exception):
    """Deliberate abort with a distinct exit code + human-readable message."""

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.message = message


# ─── Deny-listed destructive git invocations ─────────────────────────────────


def _check_banned(args: tuple[str, ...]) -> None:
    """Raise :class:`BannedGitInvocationError` on any deny-listed argv shape.

    Banned (repo-root standing ban + hard constraints, plan §4):
    ``reset --hard`` (any form), ``clean -f`` (any ``-f`` variant incl.
    ``--force``), full-tree ``checkout .`` / ``restore .``, ``stash drop``.
    Path-scoped ``checkout HEAD -- <path>`` is legitimate and not matched.
    """
    if not args:
        return
    sub = args[0]
    rest = args[1:]
    if sub == "reset" and "--hard" in rest:
        raise BannedGitInvocationError("banned: git reset --hard on the shared repo root")
    if sub == "clean":
        for a in rest:
            if a == "--force" or (a.startswith("-") and not a.startswith("--") and "f" in a):
                raise BannedGitInvocationError("banned: git clean -f on the shared repo root")
    if sub in ("checkout", "restore") and "." in rest:
        raise BannedGitInvocationError(f"banned: full-tree git {sub} . on the shared repo root")
    if sub == "stash" and len(rest) >= 1 and rest[0] == "drop":
        raise BannedGitInvocationError("banned: git stash drop (entries are popped or kept)")


def _git_argv(repo: Path, *args: str) -> list[str]:
    """Build a deny-checked ``git -C <repo> ...`` argv."""
    _check_banned(args)
    return ["git", "-C", str(repo), *args]


def git(
    repo: Path, *args: str, check: bool = True, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a deny-checked git command synchronously (short, non-network ops)."""
    return subprocess.run(
        _git_argv(repo, *args),
        capture_output=True,
        text=True,
        check=check,
        input=input_text,
    )


@dataclasses.dataclass
class GitResult:
    """Outcome of a bounded git subprocess (fetch / pull / push)."""

    rc: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def combined(self) -> str:
        return self.stdout + "\n" + self.stderr


def _run_bounded(argv: list[str], timeout_s: float) -> GitResult:
    """Run ``argv`` with a hard wall-clock bound; on timeout kill the whole
    process group (``start_new_session=True`` + ``killpg``) and report
    ``timed_out=True``. Never leaves the child running."""
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return GitResult(proc.returncode, out or "", err or "", False)
    except subprocess.TimeoutExpired:
        # The child may exit between the timeout and the kill — benign race.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        return GitResult(proc.returncode, out or "", err or "", True)


def _pull_argv(repo: Path) -> list[str]:
    """Argv for the recovery pull (flags explicit — no repo-config reliance).
    A separate seam so tests can substitute a slow command to exercise the
    timeout/killpg path."""
    return _git_argv(repo, "pull", "--rebase=merges", "--autostash", "origin", "main")


def pull_rebase(repo: Path, timeout_s: float) -> GitResult:
    """Run the bounded recovery pull."""
    return _run_bounded(_pull_argv(repo), timeout_s)


def _pull_with_transient_retry(repo: Path, report: dict, timeout_s: float) -> GitResult:
    """Bounded pull with exactly ONE retry on the transient multiple-branches race.

    ``fatal: Cannot rebase onto multiple branches.`` is die()d by
    builtin/pull.c BEFORE any rebase starts, when a concurrent fetch rewrote
    FETCH_HEAD with >1 for-merge entries mid-pull (#965/#998); verified to
    leave no rebase state, no autostash, and an untouched worktree (#1044
    §3), so the retry is state-safe. Retry once after a short sleep — the
    retry's own fetch rewrites FETCH_HEAD — and only when no rebase state
    exists (belt-and-braces re-verification). A second failure returns
    as-is and surfaces through the existing handling unchanged. A SUCCESSFUL
    retry is itself diagnostic evidence of an unserialized FETCH_HEAD writer
    (a fetch bypassing this helper's flocks) — the ``_msg`` line preserves
    that in the report/journal so recurrences stay attributable.
    """
    result = pull_rebase(repo, timeout_s)
    if not result.timed_out and result.rc != 0 and MULTI_BRANCH_STDERR_NEEDLE in result.stderr:
        if _rebase_in_progress(repo):
            _msg(
                report,
                "multiple-branches signature but rebase state exists — NOT retrying "
                "(unexpected shape; surfacing the failure as-is).",
            )
            return result
        _msg(
            report,
            "transient 'Cannot rebase onto multiple branches' (concurrent FETCH_HEAD "
            f"rewrite) — one retry after {_retry_sleep_s():.1f}s.",
        )
        time.sleep(_retry_sleep_s())
        return pull_rebase(repo, timeout_s)
    return result


# ─── Report ──────────────────────────────────────────────────────────────────


def _new_report(repo: Path, dry_run: bool) -> dict:
    return {
        "state": None,  # synced | already | in-flight | dry-run | error
        "exit_code": None,
        "repo": str(repo),
        "dry_run": dry_run,
        "ahead": None,
        "behind": None,
        "collisions": {"identical": 0, "differing": 0, "non_regular": 0, "paths_first_20": []},
        "sweep": [],
        "rescue_dir": None,
        "stash": [],
        "conflicted_paths": [],
        "restored": [],
        "audit_drift": [],
        "actions_performed": False,
        "messages": [],
    }


def _msg(report: dict, text: str) -> None:
    report["messages"].append(text)


def _emit_report(report: dict, as_json: bool) -> None:
    """Human report always on stderr; ``--json`` SyncReport on stdout."""
    lines = [
        f"sync_repo_root: state={report['state']} exit={report['exit_code']} "
        f"repo={report['repo']} dry_run={report['dry_run']}",
        f"  ahead={report['ahead']} behind={report['behind']}",
    ]
    if report["rescue_dir"]:
        lines.append(f"  rescue_dir: {report['rescue_dir']}")
    for entry in report["sweep"]:
        lines.append(
            f"  sweep: {entry['action']} [{entry['kind']}] {entry['path']}"
            + (f" -> {entry['rescue_path']}" if entry.get("rescue_path") else "")
        )
    for entry in report["restored"]:
        lines.append(f"  restore: {entry}")
    for entry in report["stash"]:
        lines.append(f"  stash: {entry}")
    if report["conflicted_paths"]:
        lines.append("  conflicted paths: " + ", ".join(report["conflicted_paths"]))
    for row in report["audit_drift"]:
        lines.append(f"  audit drift: {row}")
    for m in report["messages"]:
        lines.append(f"  {m}")
    print("\n".join(lines), file=sys.stderr)
    if as_json:
        print(json.dumps(report, indent=2))


# ─── Locking ─────────────────────────────────────────────────────────────────


def acquire_single_flight() -> int | None:
    """Non-blocking flock on the root-sync lock. Returns the held fd, or
    ``None`` when another sync is in flight (the caller exits 0)."""
    ROOT_SYNC_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT_SYNC_LOCK, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _fuser_evidence(path: Path) -> str:
    """Best-effort holder evidence for a contended lock file."""
    fuser = shutil.which("fuser")
    if not fuser:
        return "(fuser unavailable — install psmisc for holder evidence)"
    proc = subprocess.run([fuser, "-v", str(path)], capture_output=True, text=True, check=False)
    return (proc.stdout + proc.stderr).strip() or "(fuser: no holder reported)"


@dataclasses.dataclass
class LivenessProbe:
    """Outcome of the /proc git-liveness scan: verdict + human-readable evidence."""

    verdict: str  # "holder" | "none" | "uncertain"
    evidence: str


def _probe_git_liveness(gd: Path, proc_root: Path = Path("/proc")) -> LivenessProbe:
    """Scan ``proc_root`` for a live git process operating on THIS repo.

    "holder": a ``comm == git`` process whose cwd attributes (via a BOUNDED
    ``git -C <cwd> rev-parse --absolute-git-dir``) to this repo's git dir —
    state-owning git commands chdir to the worktree toplevel (verified,
    #1044 §3), so linked-worktree / other-repo git activity is excluded
    without hardcoded paths. "none" ONLY when the scan COMPLETED within the
    total budget with zero holders and zero unattributable git candidates.
    Everything else — ``proc_root`` unreadable, a git process whose cwd is
    unreadable (EACCES: another user's process), unattributable, or whose
    attribution TIMED OUT (a hung FUSE/network mount), or the total probe
    budget exhausted mid-scan — is "uncertain". Every attribution subprocess
    runs under ``_run_bounded`` with ``_probe_timeout_s()``; the whole scan
    is capped by a ``_probe_budget_s()`` monotonic deadline, so the probe can
    never stall ``preflight()`` (which holds both flocks) on a dead mount.
    The caller treats anything but "none" as "keep today's young-husk
    refusal": the probe fails TOWARD conservatism by construction. It cannot
    see a conflict-paused rebase awaiting a human (no process exists then) —
    that residual is bounded by ``_husk_min_age_s`` and the repo-root policy
    that conflicts are resolved in scratch worktrees, never in place. It
    also cannot see a rebase driven by a non-git process image (a
    library-driven rebase via pygit2/dulwich or a wrapper whose ``comm`` is
    not ``git``) — a second false-"none" residual, accepted because fleet
    git activity goes through the git CLI.
    """
    deadline = time.monotonic() + _probe_budget_s()
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError as e:
        return LivenessProbe("uncertain", f"{proc_root} unreadable: {e!r}")
    gd_resolved = gd.resolve()
    unattributable: list[str] = []
    for pid in pids:
        pdir = proc_root / pid
        try:
            comm = (pdir / "comm").read_text().strip()
        except OSError:
            continue  # vanished mid-scan — provably not a live holder
        if comm != "git":
            continue
        try:
            cwd = os.readlink(pdir / "cwd")
        except FileNotFoundError:
            continue  # vanished mid-scan
        except OSError as e:
            unattributable.append(f"pid {pid}: cwd unreadable ({e.__class__.__name__})")
            continue
        if time.monotonic() > deadline:
            return LivenessProbe(
                "uncertain",
                f"probe budget ({_probe_budget_s():.0f}s) exhausted before attributing "
                f"pid {pid} — scan incomplete",
            )
        rp = _run_bounded(
            _git_argv(Path(cwd), "rev-parse", "--absolute-git-dir"), _probe_timeout_s()
        )
        if rp.timed_out:
            unattributable.append(
                f"pid {pid}: attribution timed out after {_probe_timeout_s():.0f}s (cwd {cwd})"
            )
            continue
        if rp.rc != 0:
            unattributable.append(f"pid {pid}: cwd {cwd} not attributable to a git dir")
            continue
        if Path(rp.stdout.strip()).resolve() == gd_resolved:
            return LivenessProbe("holder", f"live git pid {pid}, cwd {cwd}")
    if unattributable:
        return LivenessProbe("uncertain", "; ".join(unattributable[:5]))
    return LivenessProbe("none", "scan completed: no live git process attributable to this repo")


def acquire_task_workflow_lock(wait_s: float) -> int:
    """Bounded acquisition of ``task_workflow.LOCK_PATH`` (LOCK_NB + 2s poll).

    Excludes ``task.py`` writers for the mutation window. Still held after
    ``wait_s`` → exit 5 with ``fuser`` evidence — a hung holder must not wedge
    the helper while it holds the root-sync lock (plan §4.2 / §11 item 15).
    """
    lock_path = Path(task_workflow.LOCK_PATH)  # read at call time (tests monkeypatch)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.close(fd)
                raise SyncAbortError(
                    EXIT_PRECONDITION,
                    f"task-workflow lock still held after {wait_s:.0f}s: {lock_path}\n"
                    f"holder evidence: {_fuser_evidence(lock_path)}",
                ) from None
            time.sleep(min(2.0, remaining))


# ─── Preflight ───────────────────────────────────────────────────────────────


def _git_dir(repo: Path) -> Path:
    raw = git(repo, "rev-parse", "--git-dir").stdout.strip()
    p = Path(raw)
    return p if p.is_absolute() else repo / p


def _rebase_in_progress(repo: Path) -> bool:
    gd = _git_dir(repo)
    return (gd / "rebase-merge").exists() or (gd / "rebase-apply").exists()


def preflight(repo: Path, report: dict, dry_run: bool) -> None:
    """Preconditions checked under both locks, before any mutation.

    Order is load-bearing: (1) index.lock bounded wait FIRST (a husk abort
    needs the index; never deleted); (2) husk triage (stale → abort +
    continue; stale un-abortable head-name-less rebase → rescue autostash +
    archive-aside; stale ``git am`` session → exit 5; young → exit 5) BEFORE
    the branch check, because a mid-rebase husk leaves HEAD detached —
    checking the branch first would make the stale-husk auto-abort
    unreachable; (3) HEAD==main; (4) stranded-autostash recovery for entries
    left by PREVIOUS crashed loops.
    """
    gd = _git_dir(repo)
    index_lock = gd / "index.lock"
    if index_lock.exists():
        deadline = time.monotonic() + INDEX_LOCK_WAIT_S
        while index_lock.exists() and time.monotonic() < deadline:
            time.sleep(INDEX_LOCK_POLL_S)
        if index_lock.exists():
            age = time.time() - index_lock.stat().st_mtime
            raise SyncAbortError(
                EXIT_PRECONDITION,
                f"index.lock persistently held ({age:.0f}s old): {index_lock}\n"
                f"holder evidence: {_fuser_evidence(index_lock)}\n"
                "NEVER deleted automatically — a live git op may hold it.",
            )

    now = time.time()
    for husk, abort_args in (
        (gd / "rebase-merge", ("rebase", "--abort")),
        (gd / "rebase-apply", ("rebase", "--abort")),
        (gd / "MERGE_HEAD", ("merge", "--abort")),
    ):
        if not husk.exists():
            continue
        age = now - husk.stat().st_mtime
        stale_how = f"> {_husk_age_s():.0f}s"
        if age <= _husk_age_s():
            probe = (
                _probe_git_liveness(gd)
                if _husk_probe_enabled() and age > _husk_min_age_s()
                else None
            )
            if probe is None or probe.verdict != "none":
                extra = f"\nliveness probe: {probe.verdict} — {probe.evidence}" if probe else ""
                raise SyncAbortError(
                    EXIT_PRECONDITION,
                    f"young {husk.name} husk ({age:.0f}s old, threshold {_husk_age_s():.0f}s) — "
                    f"someone may be mid-operation; refusing to touch it.{extra}",
                )
            stale_how = "young, liveness-downgraded"
            _msg(
                report,
                f"YOUNG-HUSK DOWNGRADE: {husk.name} is {age:.0f}s old "
                f"(< {_husk_age_s():.0f}s threshold, > {_husk_min_age_s():.0f}s floor) but no "
                f"live git process is attributable to this repo ({probe.evidence}); "
                "treating as STALE.",
            )
        # Discriminators scoped to REBASE state dirs only (MERGE_HEAD — even a
        # malformed directory — routes to the explicit refuse branch below).
        is_rebase_dir = abort_args[0] == "rebase" and husk.is_dir()
        am_in_progress = is_rebase_dir and (husk / "applying").exists()
        headnameless = is_rebase_dir and not (husk / "head-name").exists()
        if dry_run:
            if am_in_progress:
                _msg(
                    report,
                    f"DRY-RUN: stale {husk.name} state is a `git am` session "
                    "(`applying` present) — a real run would refuse (exit 5); "
                    "resolve via `git am --abort` (the sanctioned root resolution; "
                    "the #1234 root guard fail-closes a bare root `git am --continue` "
                    "— FINISHING the session belongs to its owner via a "
                    "`git -C <path>`-scoped invocation)",
                )
            elif headnameless:
                _msg(
                    report,
                    f"DRY-RUN: stale head-name-less {husk.name} husk ({age:.0f}s old) — "
                    "un-abortable by git; autostash would be rescued to the stash "
                    "list, then the husk dir archived to the rescue root",
                )
            else:
                _msg(report, f"DRY-RUN: stale {husk.name} husk ({age:.0f}s old) would be aborted")
            continue
        aborted = git(repo, *abort_args, check=False)
        if aborted.returncode == 0:
            _msg(
                report,
                f"STALE-HUSK ABORT: {husk.name} was {age:.0f}s old ({stale_how}); "
                f"ran `git {' '.join(abort_args)}` "
                "(restores original HEAD + re-applies autostash).",
            )
            report["actions_performed"] = True
            continue
        if am_in_progress:
            raise SyncAbortError(
                EXIT_PRECONDITION,
                f"stale {husk.name} state is an interrupted `git am` session "
                f"(`{husk.name}/applying` present; `git rebase --abort` "
                f"rc={aborted.returncode}). Refusing to touch it — the patch data and "
                "--continue capability belong to the am session's owner.\n"
                "resolve manually: `git am --abort` (discard — the sanctioned root "
                "resolution; the #1234 root guard fail-closes a bare root "
                "`git am --continue`) or, to FINISH the session, its owner runs "
                "`git am --continue` through a `git -C <path>`-scoped invocation; "
                "then re-run the sync.",
            )
        if not headnameless:
            raise SyncAbortError(
                EXIT_UNEXPECTED,
                f"stale {husk.name} husk: `git {' '.join(abort_args)}` failed "
                f"(rc={aborted.returncode}) and the husk is not the known un-abortable "
                f"head-name-less rebase shape — refusing to touch it.\n"
                f"stderr: {aborted.stderr.strip()}",
            )
        _recover_headnameless_husk(repo, husk, age, aborted, report, stale_how)

    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        raise SyncAbortError(
            EXIT_PRECONDITION,
            f"HEAD is {branch!r}, not 'main' — the repo root must stay on main "
            "(the managed-worktree machinery owns the non-main pathology); refusing.",
        )

    # Entries stranded by PREVIOUS crashed loops.
    recover_stranded_autostash(repo, report, dry_run=dry_run, preflight_case=True)


def _recover_headnameless_husk(
    repo: Path,
    husk: Path,
    age: float,
    aborted: subprocess.CompletedProcess[str],
    report: dict,
    stale_how: str,
) -> None:
    """Rescue-then-ARCHIVE for a stale, un-abortable, head-name-less rebase husk.

    ``stale_how`` names WHY the husk counts as stale ("> <threshold>s" for the
    wall-clock path, "young, liveness-downgraded" for the #1044 probe path) so
    the ARCHIVED report never states a false age threshold.

    A crashed ``pull --rebase --autostash`` can die after writing
    ``<state-dir>/autostash`` but before ``head-name`` (2026-07-03 incident,
    #971); git can neither represent nor abort a rebase without ``head-name``
    (``git rebase --abort`` rc=1), so the husk permanently wedges preflight.
    The caller has already excluded an interrupted ``git am`` session (the
    ``applying`` marker — ``git am`` shares ``rebase-apply``); this helper
    re-checks both discriminators immediately before the move (TOCTOU).
    Recovery order preserves the never-lose-data invariant: (1) rescue the
    autostash commit into the stash list (``git stash store -m autostash``;
    guarded by a 40-hex pre-check, ``cat-file -e`` resolvability, and a
    ``rev-parse <sha>^2`` stash-shapedness check — a stored non-stash commit
    would wedge the stranded-autostash pass downstream; idempotent via a
    stash-reflog containment check; a store failure on a storable commit
    BLOCKS the move — fail loud, husk kept); non-storable autostash content
    is preserved verbatim under ``RESCUE_ROOT`` first; (2) only then
    ``shutil.move`` the ENTIRE husk dir into a fresh exclusive
    ``allocate_rescue_dir()`` — archive-aside, never deletion: every husk
    file survives for unforeseen shapes, and the same-device rename closes
    the mid-``rmtree`` mtime-refresh re-wedge window. The stored entry's
    subject is exactly ``autostash``, so preflight step (4)
    (``recover_stranded_autostash``) immediately rescue-patches and
    pops-if-clean. Deliberately NOT ``git rebase --quit``: quit removes the
    state dir and exits 0 even when its internal autostash store FAILS
    (verified on git 2.34.1), so rescue failure could not block removal.
    """
    autostash_file = husk / "autostash"
    sha = autostash_file.read_text().strip() if autostash_file.exists() else ""
    if not autostash_file.exists():
        disposition = "no autostash file present"
    elif not sha:
        disposition = "autostash file empty — nothing to rescue"
    else:
        is_hex40 = len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
        resolvable = (
            is_hex40
            and git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0
        )
        stash_shaped = (
            resolvable
            and git(repo, "rev-parse", "-q", "--verify", f"{sha}^2", check=False).returncode == 0
        )
        if not stash_shaped:
            RESCUE_ROOT.mkdir(parents=True, exist_ok=True)
            rescue_path = RESCUE_ROOT / f"husk-autostash-{husk.name}-{_rescue_timestamp()}.txt"
            rescue_path.write_text(autostash_file.read_text())
            reason = (
                "not a 40-hex sha"
                if not is_hex40
                else "does not resolve to a commit"
                if not resolvable
                else "resolves but is not stash-shaped (no second parent) — storing it "
                "would wedge the stranded-autostash pass"
            )
            disposition = (
                f"autostash content {sha!r}: {reason}; nothing storable — content "
                f"preserved at {rescue_path} (and in the archived husk dir)"
            )
        elif sha in git(repo, "rev-list", "-g", "refs/stash", check=False).stdout.split():
            disposition = f"autostash {sha[:12]} already present in the stash reflog"
        else:
            store = git(repo, "stash", "store", "-m", "autostash", sha, check=False)
            if store.returncode != 0:
                raise SyncAbortError(
                    EXIT_UNEXPECTED,
                    f"head-name-less {husk.name} husk: could not rescue autostash {sha} "
                    f"(`git stash store` rc={store.returncode}: {store.stderr.strip()}); "
                    "husk left in place — nothing moved.\n"
                    f"manual: git stash store -m autostash {sha}  # then re-run the sync",
                )
            disposition = (
                f"stored autostash {sha[:12]} as a stash entry "
                "(handled next by the stranded-autostash pass)"
            )
        report["stash"].append(f"husk-rescue: {disposition}")
    # TOCTOU recheck (recompute the discriminators immediately before the move):
    # a concurrent operation may have materialized real state since triage.
    if (husk / "head-name").exists() or (husk / "applying").exists():
        raise SyncAbortError(
            EXIT_UNEXPECTED,
            f"{husk.name} husk changed while being recovered (head-name/applying "
            "appeared) — a concurrent operation may be live; nothing moved.",
        )
    dest = allocate_rescue_dir() / husk.name
    shutil.move(str(husk), str(dest))
    _msg(
        report,
        f"STALE-HUSK ARCHIVED: {husk.name} was {age:.0f}s old ({stale_how}), "
        f"head-name-less and un-abortable (`git rebase --abort` rc={aborted.returncode}: "
        f"{aborted.stderr.strip()}); {disposition}; husk dir archived to {dest}.",
    )
    report["actions_performed"] = True


# ─── Untracked-collision pre-sweep ───────────────────────────────────────────


@dataclasses.dataclass
class Collision:
    """An untracked working-tree file a rebase onto origin/main would write."""

    path: str  # repo-relative
    kind: str  # "identical" | "differing" | "non-regular"
    origin_blob_sha: str | None = None


def _classify(repo: Path, rel_path: str) -> Collision:
    fp = repo / rel_path
    if fp.is_symlink() or not fp.is_file():
        return Collision(rel_path, "non-regular")
    local_sha = git(repo, "hash-object", "--", rel_path).stdout.strip()
    r = git(repo, "rev-parse", f"origin/main:{rel_path}", check=False)
    remote_sha = r.stdout.strip() if r.returncode == 0 else ""
    kind = "identical" if (remote_sha and local_sha == remote_sha) else "differing"
    return Collision(rel_path, kind, origin_blob_sha=remote_sha or None)


def enumerate_collisions(repo: Path) -> list[Collision]:
    """Untracked files at exactly the paths a checkout/rebase onto origin/main
    will write (``untracked ∩ diff --name-only HEAD origin/main``)."""
    untracked = set(git(repo, "ls-files", "--others", "--exclude-standard").stdout.splitlines())
    changed = set(git(repo, "diff", "--name-only", "HEAD", "origin/main").stdout.splitlines())
    return [_classify(repo, p) for p in sorted(untracked & changed)]


@dataclasses.dataclass
class SweepAction:
    """One recorded sweep decision (a row of the abort-restore ledger)."""

    path: str
    kind: str
    action: str  # "removed" | "rescued" | "planned-remove" | "planned-rescue"
    rescue_path: str | None
    origin_blob_sha: str | None
    note: str = ""


def _journal_path(rescue_dir: Path) -> Path:
    return rescue_dir / "sweep-journal.jsonl"


def _journal_append(rescue_dir: Path, action: SweepAction, *, applied: bool) -> None:
    """Durably append one journal row (O_APPEND + fsync) for ``action``.

    Called with ``applied=False`` BEFORE the remove/move executes and
    ``applied=True`` right after — the journal-before-action contract (round-1
    concern ``sweep-before-durable-ledger``): a mid-sweep SIGKILL always
    leaves an on-disk record from which ``restore_swept`` can restore. The
    write LOOPS until every byte lands (``os.write`` may short-write, e.g. on
    a filling disk); a short/failed append raises BEFORE the caller executes
    its sweep action, so an unjournalable action never runs."""
    row = dataclasses.asdict(action) | {"applied": applied}
    data = (json.dumps(row) + "\n").encode()
    fd = os.open(_journal_path(rescue_dir), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if n <= 0:  # defensive: only ever expected as a driver/filesystem anomaly
                raise OSError(
                    f"short write appending to {_journal_path(rescue_dir)}: "
                    f"{written}/{len(data)} bytes written"
                )
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def load_sweep_journal(rescue_dir: Path, report: dict | None = None) -> list[SweepAction]:
    """Reconstruct sweep actions from the on-disk journal ALONE (crash recovery).

    Keeps the LAST row per ``(path, action)`` (the applied-marker row when the
    action completed, else the pre-action intent row) in first-seen order —
    enough for ``restore_swept`` even when the in-memory ledger died with the
    process. Returns ``[]`` when no journal exists.

    Parses line-by-line so one malformed line never discards the complete rows
    around it. A malformed TRAILING line is the expected torn tail of a
    mid-append crash: recorded (journal path + line number) and skipped. A
    malformed NON-trailing line is journal CORRUPTION — named loudly, never
    silently ignored — and the remaining valid rows are still returned so the
    restore proceeds with what is recoverable. Records go to ``report`` when
    given, else stderr (never silent)."""
    journal = _journal_path(rescue_dir)
    if not journal.exists():
        return []
    latest: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    lines = journal.read_text().splitlines()
    last_row_idx = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            kind = (
                "torn trailing line (mid-append crash) skipped"
                if i == last_row_idx
                else "CORRUPT non-trailing line — journal corruption"
            )
            message = f"sweep-journal {kind}: {journal} line {i + 1}: {e}"
            if report is not None:
                _msg(report, message)
            else:
                print(f"sync_repo_root: {message}", file=sys.stderr)
            continue
        row.pop("applied", None)
        key = (row["path"], row["action"])
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [SweepAction(**latest[key]) for key in order]


def sweep(
    repo: Path, collisions: list[Collision], rescue_dir: Path | None, dry_run: bool
) -> list[SweepAction]:
    """Clear collisions: byte-identical → remove (re-hashed immediately before
    the removal); anything else (differing / non-regular / freshly-touched) →
    move to the rescue dir, preserving relative paths. Never deletes
    non-identical data. Every mutating action is journaled durably BEFORE it
    executes and marked applied after (``_journal_append``). ``rescue_dir``
    must be an allocated dir for a mutating sweep; ``None`` is allowed only
    with ``dry_run=True`` (a dry-run sweep writes nothing)."""
    actions: list[SweepAction] = []
    for c in collisions:
        fp = repo / c.path
        if dry_run:
            planned = "planned-remove" if c.kind == "identical" else "planned-rescue"
            actions.append(SweepAction(c.path, c.kind, planned, None, c.origin_blob_sha))
            continue
        assert rescue_dir is not None, "rescue_dir is required for a mutating sweep"
        fresh = False
        if fp.is_file() and not fp.is_symlink():
            fresh = (time.time() - fp.stat().st_mtime) < _fresh_s()
        if c.kind == "identical" and not fresh:
            # RE-HASH immediately before removal (TOCTOU guard, plan §11 item 13).
            new_sha = git(repo, "hash-object", "--", c.path).stdout.strip()
            if new_sha == c.origin_blob_sha:
                action = SweepAction(c.path, c.kind, "removed", None, c.origin_blob_sha)
                _journal_append(rescue_dir, action, applied=False)
                os.remove(fp)
                _journal_append(rescue_dir, action, applied=True)
                actions.append(action)
                continue
            note = "re-hash mismatch — downgraded to rescue"
        else:
            note = "fresh-mtime guard — rescued regardless of hash" if fresh else ""
        dest = rescue_dir / c.path
        action = SweepAction(c.path, c.kind, "rescued", str(dest), c.origin_blob_sha, note)
        _journal_append(rescue_dir, action, applied=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fp), str(dest))
        _journal_append(rescue_dir, action, applied=True)
        actions.append(action)
    return actions


def _write_ledger(rescue_dir: Path, ledger: list[SweepAction]) -> None:
    """Write the consolidated ``sweep-manifest.json`` rollup atomically
    (tmp + ``os.replace``). The DURABLE per-action record is the journal
    (``_journal_append``, written before each action); this manifest is the
    human-facing audit rollup of the full run."""
    if not ledger:
        return
    rescue_dir.mkdir(parents=True, exist_ok=True)
    manifest = rescue_dir / "sweep-manifest.json"
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(json.dumps([dataclasses.asdict(a) for a in ledger], indent=2))
    os.replace(tmp, manifest)


def restore_swept(repo: Path, ledger: list[SweepAction], report: dict) -> None:
    """Journal/ledger-driven abort-restore (plan §4.3): on any post-sweep
    failure exit, put back what the sweep took. ``ledger`` may come from the
    in-memory list OR from ``load_sweep_journal`` alone (crash recovery), so
    every action is guarded for the journaled-but-not-applied case. Rescued
    files move back to their original paths (an occupied original path reports
    KEPT-IN-RESCUE when the rescue copy exists, or INTACT-never-applied when
    the journaled move never ran); identical-removed files are rematerialized
    from the RECORDED origin blob sha via ``git cat-file blob`` (plumbing
    write — no index staging; the sha recorded at sweep time, NOT
    ``origin/main:<path>``, since the ref may have moved between sweep and
    restore). Applies only to the sweep's own actions — never any other
    working-tree state."""
    for a in ledger:
        target = repo / a.path
        if a.action == "rescued" and a.rescue_path:
            rescue_copy = Path(a.rescue_path)
            if target.exists():
                if rescue_copy.exists():
                    report["restored"].append(
                        f"KEPT-IN-RESCUE {a.path} — original path now occupied; "
                        f"rescue copy at {a.rescue_path}"
                    )
                else:
                    # Journaled-but-NOT-applied intent: the move never ran, so
                    # no rescue copy exists — the occupant is the file still
                    # sitting (verified present) at its original path.
                    report["restored"].append(
                        f"INTACT {a.path} — journaled rescue intent was never applied "
                        f"(no rescue copy at {a.rescue_path}); file present at its "
                        f"original path"
                    )
                continue
            if not rescue_copy.exists():
                # Journaled intent whose move never executed AND the original
                # path is gone too — nothing to restore from; report loudly.
                report["restored"].append(
                    f"MISSING {a.path} — neither the original path nor the rescue copy "
                    f"({a.rescue_path}) exists; journaled rescue intent was not applied"
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(a.rescue_path, str(target))
            report["restored"].append(f"moved-back {a.path}")
        elif a.action == "removed":
            if target.exists():
                # Post-pull tracked copy, or a journaled-but-not-applied
                # remove whose original untracked file is still in place.
                report["restored"].append(
                    f"SKIPPED rematerialize {a.path} — path now occupied; leaving as-is"
                )
                continue
            if not a.origin_blob_sha:
                report["restored"].append(
                    f"CANNOT rematerialize {a.path} — no recorded origin blob sha in the "
                    "ledger/journal row"
                )
                continue
            blob = subprocess.run(
                _git_argv(repo, "cat-file", "blob", a.origin_blob_sha),
                capture_output=True,
                check=True,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob.stdout)
            report["restored"].append(
                f"rematerialized {a.path} from recorded blob {a.origin_blob_sha[:12]}"
            )


def parse_collision_stderr(text: str) -> list[str]:
    """Extract the tab-indented path list from git's untracked-collision error
    (exact wording verified — plan §12 item 1)."""
    paths: list[str] = []
    collecting = False
    for line in text.splitlines():
        if COLLISION_STDERR_NEEDLE in line:
            collecting = True
            continue
        if collecting:
            if line.startswith(("\t", " ")) and line.strip():
                paths.append(line.strip())
            else:
                collecting = False
    return paths


# ─── Stranded-autostash recovery ─────────────────────────────────────────────


def _autostash_entries(repo: Path) -> list[tuple[str, str]]:
    """(ref, sha) for stash entries whose subject is exactly ``autostash``."""
    out = git(repo, "stash", "list").stdout.splitlines()
    entries: list[tuple[str, str]] = []
    for line in out:
        ref, _, subject = line.partition(": ")
        if subject.strip() == "autostash":
            sha = git(repo, "rev-parse", ref).stdout.strip()
            entries.append((ref, sha))
    return entries


def _clear_unmerged_paths(repo: Path, paths: list[str]) -> None:
    """Path-scoped clear of unmerged entries back to HEAD (NOT the banned
    full-tree ``checkout .`` — explicit path list only). The content is doubly
    preserved before this runs: in the kept stash entry AND the rescue patch."""
    git(repo, "checkout", "HEAD", "--", *paths)


def recover_stranded_autostash(
    repo: Path, report: dict, *, dry_run: bool, preflight_case: bool
) -> None:
    """Recover ``stash@{n}: autostash`` entries (plan §4.4).

    Per entry: (1) rescue patch FIRST (``git stash show -p -u`` →
    ``<rescue-root>/stash-<sha12>.patch``); (2) clear ONLY the unmerged paths
    attributable to THAT autostash (``--diff-filter=U`` ∩
    ``stash show --name-only``); (3) ``git apply --check`` clean →
    ``git stash pop`` (pop drops the entry only on clean application), else
    KEEP the entry + report. Never ``git stash drop``.
    """
    processed: set[str] = set()
    while True:
        entry = next(
            ((ref, sha) for ref, sha in _autostash_entries(repo) if sha not in processed),
            None,
        )
        if entry is None:
            return
        ref, sha = entry
        processed.add(sha)
        if dry_run:
            report["stash"].append(
                f"DRY-RUN: stranded autostash {ref} ({sha[:12]}) would be recovered"
            )
            continue
        report["actions_performed"] = True
        # (1) Rescue patch FIRST — never lost even if later steps go wrong.
        RESCUE_ROOT.mkdir(parents=True, exist_ok=True)
        patch_proc = git(repo, "stash", "show", "-p", "-u", ref, check=False)
        patch_text = patch_proc.stdout if patch_proc.returncode == 0 else ""
        if not patch_text:
            patch_text = git(repo, "stash", "show", "-p", ref).stdout
        patch_path = RESCUE_ROOT / f"stash-{sha[:12]}.patch"
        patch_path.write_text(patch_text)
        # (2) Clear only THIS autostash's unmerged paths.
        unmerged = [
            p for p in git(repo, "diff", "--name-only", "--diff-filter=U").stdout.splitlines() if p
        ]
        if unmerged:
            stash_paths = set(
                git(repo, "stash", "show", "--name-only", ref, check=False).stdout.splitlines()
            )
            targets = sorted(set(unmerged) & stash_paths) if preflight_case else sorted(unmerged)
            if targets:
                _clear_unmerged_paths(repo, targets)
                report["stash"].append(
                    f"{ref} ({sha[:12]}): cleared unmerged paths {targets} back to HEAD "
                    f"(content preserved in the stash entry + {patch_path})"
                )
        # (3) Pop if clean; keep + report otherwise.
        check = subprocess.run(
            _git_argv(repo, "apply", "--check"),
            capture_output=True,
            text=True,
            check=False,
            input=git(repo, "stash", "show", "-p", ref).stdout,
        )
        if check.returncode == 0:
            pop = git(repo, "stash", "pop", ref, check=False)
            if pop.returncode == 0:
                report["stash"].append(f"popped {ref} ({sha[:12]}); rescue patch {patch_path}")
            else:
                report["stash"].append(
                    f"KEPT {ref} ({sha[:12]}) — pop failed unexpectedly "
                    f"(rc={pop.returncode}: {pop.stderr.strip()}); rescue patch {patch_path}"
                )
        else:
            report["stash"].append(
                f"KEPT {ref} ({sha[:12]}) — apply --check dirty; manual triage; "
                f"rescue patch {patch_path}"
            )


# ─── Pull pipeline ───────────────────────────────────────────────────────────


def _capture_conflict_and_abort(repo: Path, report: dict) -> list[str]:
    """Capture the conflicted paths BEFORE ``git rebase --abort`` (the abort
    clears the state), then abort cleanly."""
    conflicted = [
        p for p in git(repo, "diff", "--name-only", "--diff-filter=U").stdout.splitlines() if p
    ]
    report["conflicted_paths"] = conflicted
    if _rebase_in_progress(repo):
        git(repo, "rebase", "--abort")
    return conflicted


def _conflict_message(conflicted: list[str]) -> str:
    lines = [
        "content conflict — rebase aborted cleanly (original HEAD restored, autostash re-applied).",
        "conflicted paths: " + (", ".join(conflicted) or "(none listed)"),
    ]
    if "tasks/REGISTRY.json" in conflicted:
        lines.append(
            "NOTE: a tasks/REGISTRY.json conflict is EXPECTED on incident-scale "
            "divergence (task creation on both sides makes it near-certain) — "
            "not a helper failure."
        )
    lines.append(SCRATCH_WORKTREE_RECIPE)
    return "\n".join(lines)


def _sweep_and_record(
    repo: Path,
    collisions: list[Collision],
    ledger: list[SweepAction],
    rescue: RescueDir,
    report: dict,
) -> None:
    """Allocate the rescue dir (lazily, exclusively), run the journaled sweep,
    fold the actions into the ledger + report, and write the consolidated
    manifest. No-op on an empty collision list (nothing is allocated)."""
    if not collisions:
        return
    rescue_dir = rescue.get()
    report["rescue_dir"] = str(rescue_dir)
    actions = sweep(repo, collisions, rescue_dir, dry_run=False)
    ledger.extend(actions)
    report["sweep"].extend(dataclasses.asdict(a) for a in actions)
    if ledger:
        _write_ledger(rescue_dir, ledger)
    if actions:
        report["actions_performed"] = True


def _pull_pipeline(
    repo: Path,
    ledger: list[SweepAction],
    rescue: RescueDir,
    report: dict,
    timeout_s: float,
) -> None:
    """Sweep-wrapped pull: enumerate collisions → journaled sweep (each action
    durable BEFORE it executes; consolidated manifest after) → bounded
    pull-rebase (with one transient multiple-branches retry,
    ``_pull_with_transient_retry``) → error-driven fallback sweep + one retry →
    conflict/timeout abort policy → post-pull stranded-autostash recovery. The
    rescue dir is allocated lazily + exclusively on first sweep need
    (``RescueDir``)."""
    collisions = enumerate_collisions(repo)
    _record_collision_plan(report, collisions)
    _sweep_and_record(repo, collisions, ledger, rescue, report)

    result = _pull_with_transient_retry(repo, report, timeout_s)
    if result.timed_out:
        if _rebase_in_progress(repo):
            git(repo, "rebase", "--abort")
        raise SyncAbortError(
            EXIT_TIMEOUT, f"pull timed out after {timeout_s:.0f}s; rebase aborted cleanly."
        )

    if result.rc != 0 and COLLISION_STDERR_NEEDLE in result.combined:
        # Error-driven fallback: git's own stderr path list is authoritative
        # if the enumeration under-covered an edge (plan §4.3).
        first_paths = parse_collision_stderr(result.combined)
        _msg(
            report,
            f"fallback sweep: pull still hit {len(first_paths)} collision path(s); "
            "sweeping exactly those and retrying once.",
        )
        fallback = [_classify(repo, p) for p in first_paths if (repo / p).exists()]
        _sweep_and_record(repo, fallback, ledger, rescue, report)
        result = _pull_with_transient_retry(repo, report, timeout_s)
        if result.timed_out:
            if _rebase_in_progress(repo):
                git(repo, "rebase", "--abort")
            raise SyncAbortError(
                EXIT_TIMEOUT, f"pull timed out after {timeout_s:.0f}s; rebase aborted cleanly."
            )
        if result.rc != 0 and COLLISION_STDERR_NEEDLE in result.combined:
            second_paths = parse_collision_stderr(result.combined)
            raise SyncAbortError(
                EXIT_UNEXPECTED,
                "two consecutive untracked-collision failures — enumeration + "
                f"fallback both insufficient.\nfirst attempt paths: {first_paths}\n"
                f"second attempt paths: {second_paths}",
            )

    if result.rc != 0:
        conflicted = _capture_conflict_and_abort(repo, report)
        if conflicted:
            raise SyncAbortError(EXIT_CONFLICT, _conflict_message(conflicted))
        raise SyncAbortError(
            EXIT_UNEXPECTED,
            f"pull failed (rc={result.rc}) with no conflict state:\n{result.stderr.strip()}",
        )

    report["actions_performed"] = True
    if AUTOSTASH_CONFLICT_NEEDLE in result.combined:
        # Detected on STDERR/combined output — the pull exits rc=0 in this
        # case (plan §12 item 4); never key on the exit code.
        _msg(report, "own-pull autostash reapply conflicted — recovering the stranded entry.")
    recover_stranded_autostash(repo, report, dry_run=False, preflight_case=False)


def _record_collision_plan(report: dict, collisions: list[Collision]) -> None:
    counts = report["collisions"]
    for c in collisions:
        key = {"identical": "identical", "differing": "differing", "non-regular": "non_regular"}[
            c.kind
        ]
        counts[key] += 1
    counts["paths_first_20"] = [c.path for c in collisions[:20]]


# ─── Divergence + integrity ──────────────────────────────────────────────────


def _counts(repo: Path) -> tuple[int, int]:
    """(ahead, behind) of HEAD vs origin/main."""
    ahead = int(git(repo, "rev-list", "--count", "origin/main..HEAD").stdout.strip())
    behind = int(git(repo, "rev-list", "--count", "HEAD..origin/main").stdout.strip())
    return ahead, behind


def _post_sync_integrity(repo: Path, report: dict) -> None:
    """Assert no husk remains; surface REGISTRY-vs-filesystem drift rows
    (reported, never auto-fixed — the #811 folder-loss detector)."""
    gd = _git_dir(repo)
    if _rebase_in_progress(repo) or (gd / "MERGE_HEAD").exists():
        raise SyncAbortError(
            EXIT_UNEXPECTED, "post-sync integrity failure: a rebase/merge husk remains."
        )
    registry = repo / "tasks" / "REGISTRY.json"
    if registry.exists():
        report["audit_drift"] = task_workflow.audit()
    else:
        _msg(report, "audit skipped — no tasks/REGISTRY.json under the synced repo")


# ─── Main flow ───────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sync_repo_root.py",
        description="Single-flight recovery sync for the shared repo root.",
    )
    p.add_argument("--dry-run", action="store_true", help="report only; no mutation beyond fetch")
    p.add_argument("--no-push", action="store_true", help="sync local only; skip the push leg")
    p.add_argument("--json", dest="as_json", action="store_true", help="SyncReport on stdout")
    p.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="bound for EACH long git subprocess (default EPM_ROOT_SYNC_TIMEOUT_S or 600)",
    )
    p.add_argument(
        "--repo",
        type=str,
        default=None,
        help="target checkout (default: task_workflow.primary_checkout_root())",
    )
    return p.parse_args(argv)


def _push_leg(
    repo: Path,
    ledger: list[SweepAction],
    rescue: RescueDir,
    report: dict,
    timeout_s: float,
) -> None:
    """Push with ONE retry: a rejection routes through the SAME sweep-wrapped
    pull pipeline (fresh collisions can materialize in the interim window),
    then pushes again; a second rejection exits 3 (plan §4.5)."""
    ahead, _ = _counts(repo)
    if ahead == 0:
        return
    push = _run_bounded(_git_argv(repo, "push", "origin", "main"), timeout_s)
    if push.timed_out:
        raise SyncAbortError(EXIT_TIMEOUT, f"push timed out after {timeout_s:.0f}s.")
    if push.rc != 0:
        _msg(
            report,
            "push rejected — one retry through the sweep-wrapped pull "
            f"pipeline. stderr: {push.stderr.strip()}",
        )
        _pull_pipeline(repo, ledger, rescue, report, timeout_s)
        push = _run_bounded(_git_argv(repo, "push", "origin", "main"), timeout_s)
        if push.timed_out:
            raise SyncAbortError(EXIT_TIMEOUT, f"push timed out after {timeout_s:.0f}s.")
        if push.rc != 0:
            raise SyncAbortError(
                EXIT_PUSH_FAILED,
                "push failed after the one retry — an out-of-band pusher "
                "or a rejecting remote; report + stop (never loop).\n"
                f"stderr: {push.stderr.strip()}",
            )
    report["actions_performed"] = True
    _msg(report, "push to origin/main succeeded.")


def _run_locked(repo: Path, args: argparse.Namespace, report: dict, timeout_s: float) -> int:
    """Everything under the root-sync lock: bounded second lock → preflight →
    fetch → dry-run report OR pull pipeline → push with one retry → integrity."""
    ledger: list[SweepAction] = []
    rescue = RescueDir()
    try:
        fd2 = acquire_task_workflow_lock(_lock2_wait_s())
    except SyncAbortError as e:
        report["state"] = "error"
        report["exit_code"] = e.exit_code
        _msg(report, e.message)
        return e.exit_code
    try:
        try:
            preflight(repo, report, dry_run=args.dry_run)

            fetch = _run_bounded(_git_argv(repo, "fetch", "origin", "main"), timeout_s)
            if fetch.timed_out:
                raise SyncAbortError(EXIT_TIMEOUT, f"fetch timed out after {timeout_s:.0f}s.")
            if fetch.rc != 0:
                raise SyncAbortError(
                    EXIT_UNEXPECTED, f"fetch failed (rc={fetch.rc}):\n{fetch.stderr.strip()}"
                )
            ahead, behind = _counts(repo)
            report["ahead"], report["behind"] = ahead, behind

            if args.dry_run:
                collisions = enumerate_collisions(repo)
                _record_collision_plan(report, collisions)
                report["sweep"] = [
                    dataclasses.asdict(a) for a in sweep(repo, collisions, None, dry_run=True)
                ]
                report["state"] = "dry-run"
                report["exit_code"] = EXIT_OK
                return EXIT_OK

            if behind > 0:
                _pull_pipeline(repo, ledger, rescue, report, timeout_s)

            if not args.no_push:
                _push_leg(repo, ledger, rescue, report, timeout_s)

            _post_sync_integrity(repo, report)
            report["state"] = "synced" if report.get("actions_performed") else "already"
            report["exit_code"] = EXIT_OK
            return EXIT_OK
        except SyncAbortError:
            raise
        except subprocess.CalledProcessError as e:
            raise SyncAbortError(
                EXIT_UNEXPECTED,
                f"unexpected git failure: {e.cmd} rc={e.returncode}\n"
                f"stderr: {(e.stderr or '').strip()}",
            ) from e
        except Exception as e:  # fail loud with full traceback in the report
            raise SyncAbortError(
                EXIT_UNEXPECTED, f"unexpected error: {e!r}\n{traceback.format_exc()}"
            ) from e
    except SyncAbortError as e:
        code = e.exit_code
        try:
            # Journal-first restore: the on-disk journal survives a crash the
            # in-memory ledger does not (every action is journaled before it
            # executes), so it is authoritative; union in any ledger rows the
            # journal is missing (e.g. a failed journal append) as backstop.
            actions = load_sweep_journal(rescue.allocated, report) if rescue.allocated else []
            journaled = {(a.path, a.action) for a in actions}
            actions += [a for a in ledger if (a.path, a.action) not in journaled]
            restore_swept(repo, actions, report)
        except Exception as restore_err:  # never lose the report (minor (b))
            code = EXIT_UNEXPECTED
            _msg(
                report,
                "restore_swept FAILED — swept copies + journal retained at "
                f"{rescue.allocated}: {restore_err!r}\n{traceback.format_exc()}",
            )
        report["state"] = "error"
        report["exit_code"] = code
        _msg(report, e.message)
        return code
    finally:
        os.close(fd2)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code (importable for tests)."""
    args = _parse_args(argv)
    repo = Path(args.repo).resolve() if args.repo else task_workflow.primary_checkout_root()
    timeout_s = args.timeout_s if args.timeout_s is not None else _timeout_s_default()
    report = _new_report(repo, args.dry_run)

    fd1 = acquire_single_flight()
    if fd1 is None:
        report["state"] = "in-flight"
        report["exit_code"] = EXIT_OK
        _msg(report, IN_FLIGHT_MSG)
        _emit_report(report, args.as_json)
        return EXIT_OK
    try:
        code = _run_locked(repo, args, report, timeout_s)
    finally:
        os.close(fd1)
    _emit_report(report, args.as_json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
