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
       index.lock persistently held, task-workflow lock held past the bound)
    6  unexpected error (fail loud; swept files restored; state reported)

Safety invariants (binding; pinned by tests/test_sync_repo_root.py):

  * NEVER a repo-root ``git reset --hard`` / ``git clean -f`` / full-tree
    ``git checkout .`` / ``git restore .`` / ``git stash drop`` — the
    ``git()`` wrapper deny-lists these argv shapes and raises at call time.
  * Never delete non-identical untracked data — rescue dir only, and
    byte-identity is re-checked (re-hash) immediately before every removal;
    a file touched within ``EPM_ROOT_SYNC_FRESH_S`` (default 60s) is rescued
    regardless of hash (it may be mid-write by a live session).
  * Ledger-driven abort-restore: every sweep action is recorded in
    ``<rescue-dir>/sweep-manifest.json`` BEFORE the pull; on any post-sweep
    failure exit (2/3/4/6) rescued files are moved back (kept in the rescue
    dir + reported if the path is now occupied) and identical-removed files
    are rematerialized from the origin blob via ``git cat-file blob`` (a
    plumbing write — no index staging).
  * Fail loud; no silent husks; a young (< ``EPM_ROOT_SYNC_HUSK_AGE_S``)
    rebase/merge husk or a persistent index.lock is reported and exits 5 —
    never deleted.

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
    continue, young → exit 5) BEFORE the branch check, because a mid-rebase
    husk leaves HEAD detached — checking the branch first would make the
    stale-husk auto-abort unreachable; (3) HEAD==main; (4) stranded-autostash
    recovery for entries left by PREVIOUS crashed loops.
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
        if age <= _husk_age_s():
            raise SyncAbortError(
                EXIT_PRECONDITION,
                f"young {husk.name} husk ({age:.0f}s old, threshold {_husk_age_s():.0f}s) — "
                "someone may be mid-operation; refusing to touch it.",
            )
        if dry_run:
            _msg(report, f"DRY-RUN: stale {husk.name} husk ({age:.0f}s old) would be aborted")
            continue
        git(repo, *abort_args)
        _msg(
            report,
            f"STALE-HUSK ABORT: {husk.name} was {age:.0f}s old (> {_husk_age_s():.0f}s); "
            f"ran `git {' '.join(abort_args)}` (restores original HEAD + re-applies autostash).",
        )
        report["actions_performed"] = True

    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        raise SyncAbortError(
            EXIT_PRECONDITION,
            f"HEAD is {branch!r}, not 'main' — the repo root must stay on main "
            "(the managed-worktree machinery owns the non-main pathology); refusing.",
        )

    # Entries stranded by PREVIOUS crashed loops.
    recover_stranded_autostash(repo, report, dry_run=dry_run, preflight_case=True)


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


def sweep(
    repo: Path, collisions: list[Collision], rescue_dir: Path, dry_run: bool
) -> list[SweepAction]:
    """Clear collisions: byte-identical → remove (re-hashed immediately before
    the removal); anything else (differing / non-regular / freshly-touched) →
    move to the rescue dir, preserving relative paths. Never deletes
    non-identical data."""
    actions: list[SweepAction] = []
    for c in collisions:
        fp = repo / c.path
        if dry_run:
            planned = "planned-remove" if c.kind == "identical" else "planned-rescue"
            actions.append(SweepAction(c.path, c.kind, planned, None, c.origin_blob_sha))
            continue
        fresh = False
        if fp.is_file() and not fp.is_symlink():
            fresh = (time.time() - fp.stat().st_mtime) < _fresh_s()
        if c.kind == "identical" and not fresh:
            # RE-HASH immediately before removal (TOCTOU guard, plan §11 item 13).
            new_sha = git(repo, "hash-object", "--", c.path).stdout.strip()
            if new_sha == c.origin_blob_sha:
                os.remove(fp)
                actions.append(SweepAction(c.path, c.kind, "removed", None, c.origin_blob_sha))
                continue
            note = "re-hash mismatch — downgraded to rescue"
        else:
            note = "fresh-mtime guard — rescued regardless of hash" if fresh else ""
        dest = rescue_dir / c.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fp), str(dest))
        actions.append(SweepAction(c.path, c.kind, "rescued", str(dest), c.origin_blob_sha, note))
    return actions


def _write_ledger(rescue_dir: Path, ledger: list[SweepAction]) -> None:
    """Persist the sweep ledger BEFORE the pull (the abort-restore contract
    reads the in-memory list; the file is the durable audit record)."""
    if not ledger:
        return
    rescue_dir.mkdir(parents=True, exist_ok=True)
    manifest = rescue_dir / "sweep-manifest.json"
    manifest.write_text(json.dumps([dataclasses.asdict(a) for a in ledger], indent=2))


def restore_swept(repo: Path, ledger: list[SweepAction], report: dict) -> None:
    """Ledger-driven abort-restore (plan §4.3): on any post-sweep failure exit,
    put back what the sweep took. Rescued files move back to their original
    paths (kept in the rescue dir + reported if the path is now occupied);
    identical-removed files are rematerialized from the origin blob via
    ``git cat-file blob`` (plumbing write — no index staging). Applies only to
    the sweep's own actions — never any other working-tree state."""
    for a in ledger:
        target = repo / a.path
        if a.action == "rescued" and a.rescue_path:
            if target.exists():
                report["restored"].append(
                    f"KEPT-IN-RESCUE {a.path} — original path now occupied; "
                    f"rescue copy at {a.rescue_path}"
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(a.rescue_path, str(target))
            report["restored"].append(f"moved-back {a.path}")
        elif a.action == "removed":
            if target.exists():
                report["restored"].append(
                    f"SKIPPED rematerialize {a.path} — path now occupied (tracked copy present)"
                )
                continue
            blob = subprocess.run(
                _git_argv(repo, "cat-file", "blob", f"origin/main:{a.path}"),
                capture_output=True,
                check=True,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob.stdout)
            report["restored"].append(f"rematerialized {a.path} from origin/main blob")


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


def _pull_pipeline(
    repo: Path,
    ledger: list[SweepAction],
    rescue_dir: Path,
    report: dict,
    timeout_s: float,
) -> None:
    """Sweep-wrapped pull: enumerate collisions → sweep (ledger written before
    the pull) → bounded pull-rebase → error-driven fallback sweep + one retry →
    conflict/timeout abort policy → post-pull stranded-autostash recovery."""
    collisions = enumerate_collisions(repo)
    _record_collision_plan(report, collisions)
    actions = sweep(repo, collisions, rescue_dir, dry_run=False)
    ledger.extend(actions)
    report["sweep"].extend(dataclasses.asdict(a) for a in actions)
    _write_ledger(rescue_dir, ledger)
    if actions:
        report["rescue_dir"] = str(rescue_dir)
        report["actions_performed"] = True

    result = pull_rebase(repo, timeout_s)
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
        actions2 = sweep(repo, fallback, rescue_dir, dry_run=False)
        ledger.extend(actions2)
        report["sweep"].extend(dataclasses.asdict(a) for a in actions2)
        _write_ledger(rescue_dir, ledger)
        if actions2:
            report["rescue_dir"] = str(rescue_dir)
            report["actions_performed"] = True
        result = pull_rebase(repo, timeout_s)
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
    rescue_dir: Path,
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
        _pull_pipeline(repo, ledger, rescue_dir, report, timeout_s)
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
    rescue_dir = RESCUE_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
                    dataclasses.asdict(a) for a in sweep(repo, collisions, rescue_dir, dry_run=True)
                ]
                report["state"] = "dry-run"
                report["exit_code"] = EXIT_OK
                return EXIT_OK

            if behind > 0:
                _pull_pipeline(repo, ledger, rescue_dir, report, timeout_s)

            if not args.no_push:
                _push_leg(repo, ledger, rescue_dir, report, timeout_s)

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
        restore_swept(repo, ledger, report)
        report["state"] = "error"
        report["exit_code"] = e.exit_code
        _msg(report, e.message)
        return e.exit_code
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
