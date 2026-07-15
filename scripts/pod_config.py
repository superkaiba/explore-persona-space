#!/usr/bin/env python3
"""Pod configuration manager -- generates SSH and MCP configs from pods.conf.

pods.conf is the SINGLE SOURCE OF TRUTH for pod connection details. This script
reads it and can regenerate ~/.ssh/config and .claude/mcp.json so you only need
to edit one file when a pod IP changes.

Usage:
    python scripts/pod_config.py --list              # Show all pods
    python scripts/pod_config.py --check             # Verify configs are in sync
    python scripts/pod_config.py --sync              # Regenerate ~/.ssh/config + .claude/mcp.json
    python scripts/pod_config.py --update pod2 --host 1.2.3.4 --port 12345
    python scripts/pod_config.py --clear-override pod-391   # Re-enable auto-refresh
    python scripts/pod_config.py --json              # Output pod list as JSON
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Per-process reentrancy state for ``locked_pods_conf`` (task #821 v3 r2 —
# closes the nested-flock deadlock). ``threading.local()`` gives us a
# per-thread depth counter + fd handle so a nested ``with locked_pods_conf()``
# acquired later in the SAME call stack skips the flock (which would
# deadlock on a second file-description of the same lockfile — flock is
# per-open-file-description, not per-process). See ``locked_pods_conf`` for
# the full acquire / release contract.
_LOCK_STATE = threading.local()

if TYPE_CHECKING:
    # Type-checking-only import: ``runpod_api`` is heavy (loads RunPod GraphQL
    # config from .env at import time) and ``cmd_refresh_from_api`` already
    # imports ``list_team_pods`` lazily for the same reason. ``PodInfo`` is
    # only used as a forward-referenced type annotation under
    # ``from __future__ import annotations``, so deferring the import here
    # keeps the cheap ``--list`` / ``--check`` paths free of the eager load.
    from runpod_api import PodInfo


def _ensure_scripts_dir_on_sys_path() -> None:
    """Insert THIS file's dir (scripts/) so a lazy ``import runpod_api`` resolves.

    In script mode scripts/ is already ``sys.path[0]``; in MODULE mode
    (``from scripts.pod_config import parse_pods_conf``) only the repo root is
    on sys.path, so a bare lazy ``runpod_api`` import raises
    ``ModuleNotFoundError`` (#1296/#1304). Mirrors scripts/backend_poll.py's
    helper; idempotent; called ONLY on the lazy paths so the cheap
    ``--list``/``--check`` paths and library imports never mutate sys.path.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


# ---------------------------------------------------------------------------
# Paths -- resolved to the MAIN repo regardless of which worktree this
# module is loaded from. ``pods.conf`` and ``pods_ephemeral.json`` are
# SHARED fleet state — every parallel /issue session reads + mutates them.
# Resolving relative to ``__file__`` (the previous behavior) meant each
# worktree saw its OWN copy of these files; a ``pod.py resume`` in
# worktree A would correctly update A's ``pods.conf`` and then re-sync
# ``~/.ssh/config`` (global), but a later ``cmd_sync`` from worktree B
# (still holding a STALE row) would silently clobber the global ssh
# config and the resumed pod's new port. ``poll_pipeline.py`` SSHing via
# the ``Host pod-<N>`` alias would then connection-refuse on the stale
# port and report ``status: dead`` for a perfectly healthy run. Routing
# the constants through ``git rev-parse --git-common-dir`` collapses
# every checkout's copy to the same on-disk file so all sessions read +
# write the SAME state. Concurrent read-modify-write races within that
# single file are serialised by ``locked_pods_conf`` (see below), which
# every mutating call site holds for the whole parse → mutate → write →
# ``cmd_sync`` sequence.
# Incident 2026-06-05, task #500: pod.py resume from the issue-500
# worktree updated worktree-local pods.conf to port 13721, but the main
# repo's pods.conf stayed at the stale 16659; the next sync against the
# main copy wrote the stale port back into ~/.ssh/config and the
# poll-loop reported a FALSE dead.
# Incident 2026-06-05, task #488: two concurrent /issue sessions each
# called ``pod_lifecycle._upsert_pods_conf`` for their own pod; the
# session B write clobbered A's row, the regenerated ~/.ssh/config
# dropped A's ``Host pod-<A>`` block, and ``poll_pipeline.py`` reported
# ``ssh: Could not resolve hostname pod-<A>: Temporary failure in name
# resolution`` for a perfectly healthy run. Fixed by ``locked_pods_conf``.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _main_repo_scripts_dir() -> Path:
    """Return the absolute path of ``scripts/`` in the MAIN repo checkout.

    Resolves via ``git rev-parse --git-common-dir`` from the directory of
    this module (NOT ``os.getcwd()``). Each worktree's ``.git`` file
    points at the same shared ``.git`` directory in the main checkout;
    its parent is the main repo root, and ``scripts/`` lives directly
    underneath. Falls back loudly (``RuntimeError``) if git resolution
    fails or the resolved ``scripts/`` directory does not exist, so a
    silent fallback to the worktree-local copy cannot reintroduce the
    divergence bug this resolver exists to prevent.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"pod_config: cannot resolve main repo via "
            f"`git rev-parse --git-common-dir` from {SCRIPT_DIR}: {exc}. "
            f"pod_config must run inside an explore-persona-space checkout."
        ) from exc
    git_common = Path(proc.stdout.strip())
    if not git_common.is_absolute():
        git_common = (SCRIPT_DIR / git_common).resolve()
    main_repo_root = git_common.parent
    scripts_dir = main_repo_root / "scripts"
    if not scripts_dir.is_dir():
        raise RuntimeError(
            f"pod_config: resolved main repo root {main_repo_root} has no "
            f"scripts/ directory; refusing to route pods.conf writes through "
            f"a malformed layout."
        )
    return scripts_dir


_MAIN_SCRIPTS_DIR = _main_repo_scripts_dir()
PROJECT_ROOT = _MAIN_SCRIPTS_DIR.parent

# --- Live pods.conf location ------------------------------------------------
# The LIVE (mutable) pod registry lives OUTSIDE the git working tree — at
# ``<git-common-dir>/eps/pods.conf`` (i.e. ``<main>/.git/eps/pods.conf``).
# ``git reset --hard`` / ``git checkout -- .`` / ``git restore -- .`` /
# ``git clean -fd`` / ``git clean -fdx`` operate on the working tree and do
# NOT touch ``.git`` internals, so the file survives every destructive git
# op that has historically wiped it. The tracked ``scripts/pods.conf`` copy
# becomes a SEED (fresh-clone bootstrap only); once ``_resolve_live_pods_conf``
# has migrated it, every reader + writer resolves to the live copy inside
# ``.git/eps/``.
#
# Incident 2026-07-01: a ``reset: moving to origin/main`` reflog entry
# rewound ``scripts/pods.conf`` to its month-old committed state and dropped
# every RUNNING pod's row (task #821). #815 added a policy lint against
# repo-root ``git reset --hard`` but does not stop ``git checkout .`` /
# ``git restore .`` / autostash-pop conflicts / a future agent ignoring the
# rule. The v3 relocation puts the live file OUT of git's blast radius
# entirely.
#
# Resolution is LAZY (call-time), NOT eager: read-only contexts (``--list``,
# ``--check``, external readers) that never mutate must not trigger the
# seed→live migration on import. Writer signatures default ``path=None``
# and resolve via ``_resolve_live_pods_conf`` inside the function body so a
# ``monkeypatch.setattr(pod_config, "PODS_CONF", tmp)`` in a test is HONORED
# on every call (module-level default arg captured at function-def time was
# the previous footgun).

_LIVE_PODS_CONF_DIRNAME = "eps"
_LIVE_PODS_CONF_FILENAME = "pods.conf"
# Seed bytes for the "first pod ever" path — written when NEITHER the tracked
# seed NOR the live file exists yet. Hoisted unchanged from the resolver body
# when it was factored into ``_resolve_live_sidecar`` (task #1183).
_PODS_CONF_BOOTSTRAP_HEADER = (
    b"# Pod registry -- SINGLE SOURCE OF TRUTH for all pod configuration.\n"
    b"# Live state lives at <git-common-dir>/eps/pods.conf (OUT of the working tree).\n"
    b"# The tracked scripts/pods.conf is a SEED only.\n"
    b"# Format: name  host  port  gpus  gpu_type  label\n"
)


def _git_common_dir() -> Path:
    """Return the absolute path of the repo's git common dir.

    ``git rev-parse --path-format=absolute --git-common-dir`` returns the
    SHARED ``.git`` directory (the same one across every worktree of a
    repo). The parent is the main-repo root. Fails loud on any resolution
    error — the caller cannot proceed without knowing where ``.git`` is.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(SCRIPT_DIR),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"pod_config: cannot resolve git common dir from {SCRIPT_DIR}: {exc}. "
            f"pod_config must run inside an explore-persona-space checkout."
        ) from exc
    path = Path(proc.stdout.strip())
    if not path.is_absolute():
        # Older git falls back to a relative path; anchor it to SCRIPT_DIR.
        path = (SCRIPT_DIR / path).resolve()
    return path


PODS_CONF_SEED = _MAIN_SCRIPTS_DIR / _LIVE_PODS_CONF_FILENAME
# ``PODS_CONF`` retained as the public symbol every downstream import binds
# to (tests monkeypatch it, external readers reference it). It now points at
# the SEED path by default; the LIVE path is resolved lazily via
# ``_resolve_live_pods_conf`` inside every read/write path.
PODS_CONF = PODS_CONF_SEED


def _resolve_live_sidecar(
    *, seed: Path, override: Path, filename: str, bootstrap: bytes, label: str
) -> Path:
    """Shared #821 lazy resolver for a live-relocated sidecar file (task #1183).

    Resolves the LIVE copy of a fleet-state sidecar (``pods.conf``,
    ``pods_ephemeral.json``) at ``<git-common-dir>/eps/<filename>``,
    migrating from the tracked seed on first use.

    Fast path (steady state): the live file exists → return it. Zero work.

    Migration path (fresh clone / first invocation after the relocation):
    only the seed exists → copy seed → live atomically (write to
    ``<live>.tmp`` then ``os.replace``), then return the live path. The
    migration is guarded by ``locked_pods_conf`` so two concurrent processes
    cannot double-migrate. First migrator prints a one-line stderr note so
    the relocation is visible in logs. When NEITHER the seed nor the live
    file exists, the live file is bootstrapped from ``bootstrap`` bytes.

    Read-only-filesystem fallback: if the target directory is not writable
    (rare — an operator running under a read-only mount), emit a loud WARN
    and return the seed path so read-only paths keep working. Any subsequent
    writer will FAIL on the seed path (git-tracked → next destructive git
    op wipes it) — the WARN surfaces that state before the wipe.

    Never called at module import time — the thin wrappers below read their
    module-level globals at call time, so a test's
    ``monkeypatch.setattr(pod_config, "PODS_CONF", tmp)`` (or
    ``PODS_EPHEMERAL_JSON``) is honored by every reader + writer in the
    process: ``override != seed`` means a monkeypatch is active and the
    override path is returned verbatim.
    """
    # Honor a test's monkeypatched module-level public symbol if it points
    # somewhere OTHER than the seed. This keeps every existing test that
    # sets ``pod_config.PODS_CONF = tmp / "pods.conf"`` (or
    # ``PODS_EPHEMERAL_JSON = tmp / ...``) working unchanged without a
    # fixture rewrite.
    if override != seed:
        return override

    try:
        common = _git_common_dir()
    except RuntimeError:
        # Cannot resolve git → fall back to the seed. Fresh checkouts
        # without a .git dir (tarball extractions, etc.) hit this branch;
        # keeps read paths working, writers will still see the seed.
        return seed

    live_dir = common / _LIVE_PODS_CONF_DIRNAME
    live = live_dir / filename

    if live.exists():
        return live

    # Migration required. Serialize under the pods.conf lock so a
    # concurrent process cannot race and double-migrate.
    with locked_pods_conf():
        # Re-check under the lock — the concurrent winner may have already
        # migrated between the pre-lock ``live.exists()`` and here.
        if live.exists():
            return live
        try:
            live_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Read-only filesystem, permissions issue, etc. Loud WARN,
            # fall back to seed. Fresh writer will fail loud on the seed
            # (the file is git-tracked) — the WARN is the operator signal.
            print(
                f"[pod_config] WARN: cannot create live {label} dir {live_dir}: "
                f"{exc}. Falling back to seed at {seed}. Any write "
                f"here will be clobbered by the next destructive git op — fix "
                f"the mount / permissions and re-run.",
                file=sys.stderr,
            )
            return seed

        # When neither the seed nor the live file exists yet, bootstrap bare
        # content so downstream readers succeed ("first pod ever" path).
        seed_bytes = seed.read_bytes() if seed.exists() else bootstrap

        tmp = live.with_suffix(live.suffix + ".tmp")
        try:
            tmp.write_bytes(seed_bytes)
            os.replace(tmp, live)
        except OSError as exc:
            # Cleanup any leftover tmp; loud WARN + seed fallback.
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            print(
                f"[pod_config] WARN: could not migrate {label} → {live} "
                f"({exc}); using seed at {seed}. Fix the mount / permissions "
                f"and re-run.",
                file=sys.stderr,
            )
            return seed

        print(
            f"[pod_config] migrated {seed} → {live} "
            f"(live {label} now lives OUT of git's blast radius; the tracked "
            f"copy is now a seed only)",
            file=sys.stderr,
        )
        return live


def _resolve_live_pods_conf() -> Path:
    """Resolve the live pods.conf path, migrating from the seed on first use.

    Thin wrapper over :func:`_resolve_live_sidecar` (see its docstring for
    the full contract — fast path, locked migration, read-only-FS fallback,
    monkeypatch honor). ``PODS_CONF`` / ``PODS_CONF_SEED`` are read at call
    time so a test's monkeypatch is honored on every call.
    """
    return _resolve_live_sidecar(
        seed=PODS_CONF_SEED,
        override=PODS_CONF,
        filename=_LIVE_PODS_CONF_FILENAME,
        bootstrap=_PODS_CONF_BOOTSTRAP_HEADER,
        label="pods.conf",
    )


# Sidecar JSON owned by pod_lifecycle.py — read here only to set/clear the
# manual_override flag from ``cmd_update``. Format documented in
# scripts/pod_lifecycle.py. We do not import pod_lifecycle.py because it
# already imports this module (avoiding circular import).
#
# Task #1183 (mirror of the #821 pods.conf relocation above): the LIVE
# (mutable) copy lives at ``<git-common-dir>/eps/pods_ephemeral.json`` — OUT
# of the git working tree, where no destructive git op can touch it — and the
# tracked ``scripts/pods_ephemeral.json`` is a SEED, migrated once on first
# use by ``resolve_live_pods_ephemeral``.
PODS_EPHEMERAL_SEED = _MAIN_SCRIPTS_DIR / "pods_ephemeral.json"
# Public symbol kept for test monkeypatch compatibility (tests set
# ``pod_config.PODS_EPHEMERAL_JSON = tmp / ...``). Points at the SEED; the
# live path is resolved lazily via ``resolve_live_pods_ephemeral()`` at every
# call (the same call-time-globals trick ``_resolve_live_pods_conf`` uses).
PODS_EPHEMERAL_JSON = PODS_EPHEMERAL_SEED
_LIVE_PODS_EPHEMERAL_FILENAME = "pods_ephemeral.json"
_PODS_EPHEMERAL_BOOTSTRAP = b'{\n  "version": 2,\n  "pods": {}\n}\n'


def resolve_live_pods_ephemeral() -> Path:
    """LIVE pods_ephemeral.json path (task #1183; mirrors #821 pods.conf).

    Thin wrapper over :func:`_resolve_live_sidecar`: honors a monkeypatched
    ``pod_config.PODS_EPHEMERAL_JSON`` (returned verbatim when it differs
    from the seed); otherwise resolves
    ``<git-common-dir>/eps/pods_ephemeral.json``, migrating the tracked seed
    atomically on first use under ``locked_pods_conf``, with the loud-WARN
    seed fallback on read-only filesystems. Never called at import time.
    """
    return _resolve_live_sidecar(
        seed=PODS_EPHEMERAL_SEED,
        override=PODS_EPHEMERAL_JSON,
        filename=_LIVE_PODS_EPHEMERAL_FILENAME,
        bootstrap=_PODS_EPHEMERAL_BOOTSTRAP,
        label="pods_ephemeral.json",
    )


# The SSH MCP server (mcp-ssh-manager) lives in the user-level Claude config,
# NOT the project-level one. The project mcp.json (PROJECT_ROOT / ".claude" /
# "mcp.json") is reserved for project-scoped servers like arxiv.
MCP_JSON = Path.home() / ".claude" / "mcp.json"
SSH_CONFIG = Path.home() / ".ssh" / "config"

# Ephemeral-pod name grammar, mirroring pod_lifecycle._POD_NAME_RE (#1334):
# pod-<digits> optionally followed by -<slug>, slug lowercase letter-initial
# ([a-z][a-z0-9-]*) — the multi-pod-per-issue form pod-<N>-<slug>. The ENVKEY
# variant is the same shape after pod.name.upper() (see _generate_mcp_env:
# env keys embed the upper-cased pod name verbatim).
_EPHEMERAL_NAME_PATTERN = r"pod-\d+(?:-[a-z][a-z0-9-]*)?"
_EPHEMERAL_ENVKEY_PATTERN = r"POD-\d+(?:-[A-Z][A-Z0-9-]*)?"

# Pod name patterns we recognize. Permanent fleet uses `podN`; ephemeral pods
# use `pod-<N>` (canonical, since the April 2026 rename) with an optional
# `-<slug>` multi-pod-per-issue suffix (#1334) — the legacy `epm-issue-<N>`
# form is still recognized for in-flight pods provisioned before the rename,
# and can be removed once no live pods carry it.
# Anything else is treated as foreign and skipped.
POD_NAME_RE = re.compile(r"^(pod\d+|" + _EPHEMERAL_NAME_PATTERN + r"|epm-issue-\d+)$")

# Shared SSH defaults written into every generated entry
SSH_KEY = "~/.ssh/id_ed25519"
SSH_USER = "root"
REMOTE_DIR = "/workspace/explore-persona-space"

# Markers delimiting the auto-generated block inside ~/.ssh/config.
# Everything between these lines (inclusive) is replaced on --sync.
BEGIN_MARKER = "# --- BEGIN MANAGED POD CONFIG ---"
END_MARKER = "# --- END MANAGED POD CONFIG ---"

# Sibling lockfile in the SAME main-repo scripts/ directory as ``pods.conf``
# itself. Held under an exclusive ``fcntl.flock`` for the duration of any
# read-modify-write on ``pods.conf`` + the downstream ``~/.ssh/config`` /
# ``~/.claude/mcp.json`` regeneration. Co-located so the lock can never
# diverge from the file it protects across worktree checkouts (same
# main-repo-resolution as ``PODS_CONF``).
PODS_CONF_LOCK = _MAIN_SCRIPTS_DIR / ".pods.conf.lock"


@contextlib.contextmanager
def locked_pods_conf() -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``PODS_CONF_LOCK`` for a read-modify-write
    on ``pods.conf`` and the downstream SSH/MCP config regeneration.

    Concurrency motivation. Multiple parallel ``/issue`` sessions each call
    ``pod_lifecycle._upsert_pods_conf`` (or ``_remove_from_pods_conf``) when
    provisioning / terminating their own pod. The unguarded sequence
    ``parse_pods_conf() -> mutate(rows) -> write_pods_conf(rows) ->
    cmd_sync(rows)`` is a classic lost-update race: session A reads, session
    B reads, A writes (with A's row), B writes (with B's row, A's row gone),
    and the final ``~/.ssh/config`` block reflects only B's view — so
    ``poll_pipeline.py`` SSHing via ``Host pod-<A>`` fails with
    ``Could not resolve hostname pod-<A>`` while A's pod is healthy.

    Serialising the whole read-modify-write-sync sequence under a single
    advisory lock collapses the race. ``cmd_sync`` is kept inside the
    critical section so a concurrent session cannot regenerate
    ``~/.ssh/config`` from a stale ``rows`` view between our
    ``write_pods_conf`` and our ``cmd_sync``. The lock is advisory and
    fcntl-based, so it is automatically released on process death (kill -9,
    OOM kill, parent timeout) — no orphaned locks survive a crash.

    Read-only callers (``cmd_list``, ``cmd_check``, ``cmd_json``,
    ``parse_pods_conf`` from external readers like ``poll_pipeline.py``) do
    NOT take this lock — they tolerate seeing a momentarily-mid-write state
    because ``write_pods_conf`` writes atomically via a single text payload.

    Reentrancy (task #821 v3 r2). This context manager is REENTRANT within a
    single process: a nested ``with locked_pods_conf()`` acquired later in
    the SAME call stack increments a per-thread depth counter and skips the
    ``flock``. It MUST behave this way because ``_resolve_live_pods_conf``
    (called lazily from every ``parse_pods_conf`` / ``write_pods_conf``)
    itself acquires the lock for the first-use seed→live migration; every
    production writer (``pod_lifecycle._upsert_pods_conf`` /
    ``_remove_from_pods_conf``, ``cmd_update``, ``cmd_refresh_from_api``)
    ALREADY holds the lock when it calls parse/write, which lazily resolve.
    A non-reentrant implementation would open a SECOND file description on
    the same lockfile inside the nested acquire — and ``flock`` is
    per-open-file-description, so ``flock(fd2, LOCK_EX)`` blocks forever on
    the fd1 our outer call is already holding. That was the codex-reviewer
    critical-blocker from round 1 (deadlock on the first post-deploy
    ``_upsert_pods_conf``).

    The reentrancy state is a ``threading.local()`` depth counter + fd
    handle. Depth 0 (outer acquire) opens the fd, takes ``LOCK_EX``, stores
    the fd, sets depth = 1. Depth > 0 (nested acquire) increments only. On
    exit, decrement; at depth 0 release ``LOCK_UN`` and close the fd. This
    covers every current AND future call site in one move — chosen over
    threading an ``assume_locked=True`` flag through the resolver because
    the flag would have to be threaded through parse_pods_conf /
    write_pods_conf / _resolve_live_pods_conf and every future caller, and
    getting it wrong at one call site would silently reintroduce the
    deadlock. The threading.local() sits BELOW the lock — no caller ever
    has to think about it.

    ``threading.local()`` gives each thread its own depth counter. A worker
    thread that acquires while the main thread is holding the lock still
    takes the ``flock`` (blocks correctly, no cross-thread reentrancy).
    ``multiprocessing`` workers use ``spawn`` (per the existing test rig),
    so each subprocess re-imports ``pod_config`` and gets a fresh
    ``_LOCK_STATE`` — cross-process serialisation is unchanged and rides
    on the ``flock`` alone. Auto-release on process death (kill -9, OOM
    kill) still holds — ``fcntl.flock`` is released by the kernel on any
    fd close, including implicit close at process exit.
    """
    depth = getattr(_LOCK_STATE, "depth", 0)
    if depth > 0:
        # Nested acquire in the same thread — the outer frame already holds
        # the flock. Increment depth only; the fd stays owned by the outer
        # frame and is released when it exits.
        _LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            _LOCK_STATE.depth -= 1
        return

    # Outer acquire: open the fd, take the exclusive flock, stash the fd
    # on the thread-local state so nested frames can see we hold it.
    PODS_CONF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(PODS_CONF_LOCK), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _LOCK_STATE.fd = fd
        _LOCK_STATE.depth = 1
        try:
            yield
        finally:
            _LOCK_STATE.depth = 0
            _LOCK_STATE.fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        # Close the fd unconditionally — even if ``flock(LOCK_EX)`` raised
        # (e.g. EINTR) we must not leak the fd. ``LOCK_UN`` is a no-op on a
        # closed fd; the kernel releases the flock at close anyway.
        os.close(fd)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Pod:
    name: str  # e.g. "pod1"
    host: str  # IP address
    port: int
    gpus: int
    gpu_type: str  # e.g. "H200", "H100"
    label: str  # human-readable RunPod name, e.g. "thomas-rebuttals"


# ---------------------------------------------------------------------------
# Parsing / writing pods.conf
# ---------------------------------------------------------------------------


def parse_pods_conf(path: Path | None = None) -> list[Pod]:
    """Read pods.conf and return a list of Pod objects.

    Format (whitespace-separated, 6 fields per line):
        name  host  port  gpus  gpu_type  label

    Lines starting with '#' and blank lines are skipped.

    ``path`` defaults to the LIVE pods.conf resolved lazily via
    ``_resolve_live_pods_conf`` (which honors a test's monkeypatched
    ``pod_config.PODS_CONF``). Explicit callers may still pass an absolute
    ``path=`` — used by existing tests that operate against a tmp fixture.
    """
    if path is None:
        path = _resolve_live_pods_conf()
    if not path.exists():
        print(f"ERROR: pods.conf not found at {path}", file=sys.stderr)
        sys.exit(1)

    pods: list[Pod] = []
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            print(
                f"WARNING: pods.conf:{lineno}: expected 6 fields, got {len(parts)} -- skipping",
                file=sys.stderr,
            )
            continue
        name, host, port_str, gpus_str, gpu_type, label = parts[:6]
        try:
            port = int(port_str)
            gpus = int(gpus_str)
        except ValueError:
            print(
                f"WARNING: pods.conf:{lineno}: port/gpus must be integers -- skipping",
                file=sys.stderr,
            )
            continue
        pods.append(Pod(name=name, host=host, port=port, gpus=gpus, gpu_type=gpu_type, label=label))
    return pods


def _guard_against_dropping_running(dropped: set[str], on_disk_rows: dict[str, Pod]) -> list[Pod]:
    """Never-drop-RUNNING guard body — returns the ``readd`` list.

    Extracted from ``write_pods_conf`` to bound cyclomatic complexity
    (ruff C901). See ``write_pods_conf`` docstring for the full contract.
    """
    # Lazy import — ``runpod_api`` is heavy (loads RunPod GraphQL config
    # from .env at import time).
    _ensure_scripts_dir_on_sys_path()
    try:
        from runpod_api import RunPodError, list_team_pods
    except ImportError:  # pragma: no cover - only if repo is malformed
        print(
            "[pod_config] WARN: cannot import runpod_api; failing SAFE "
            f"by re-adding all {len(dropped)} dropped rows to pods.conf. "
            f"Refused drops: {sorted(dropped)}.",
            file=sys.stderr,
        )
        return [on_disk_rows[n] for n in sorted(dropped)]

    try:
        live_by_name = {p.name: p for p in list_team_pods()}
    except RunPodError as exc:
        print(
            "[pod_config] WARN: could not verify live pod status "
            f"({exc}); failing SAFE by re-adding all {len(dropped)} "
            "dropped rows to pods.conf. If this was intentional, pass "
            f"allow_remove={{{sorted(dropped)!r}}}. Refused drops: "
            f"{sorted(dropped)}.",
            file=sys.stderr,
        )
        return [on_disk_rows[n] for n in sorted(dropped)]

    readd: list[Pod] = []
    for name in sorted(dropped):
        live = live_by_name.get(name)
        if live is None:
            # Absent from API — legit drop (already terminated or never
            # provisioned team-side).
            continue
        if (live.desired_status or "").upper() != "RUNNING":
            # EXITED / STOPPED / etc. — legit drop.
            continue
        # RUNNING → refuse the drop.
        print(
            f"[pod_config] WARN: refusing to drop RUNNING pod "
            f"'{name}' from pods.conf (live API says RUNNING); "
            f"re-added. If this was intentional, pass "
            f"allow_remove={{'{name}'}}.",
            file=sys.stderr,
        )
        readd.append(on_disk_rows[name])
    return readd


def write_pods_conf(
    pods: list[Pod],
    path: Path | None = None,
    *,
    allow_remove: frozenset[str] = frozenset(),
) -> None:
    """Write the pod list back to pods.conf, preserving the header comments.

    Task #821: this writer is now atomic AND guards against silently
    dropping a RUNNING pod's row.

    ``path`` defaults to the LIVE pods.conf via ``_resolve_live_pods_conf``
    (honors monkeypatched ``pod_config.PODS_CONF`` in tests).

    ``allow_remove`` is the EXPLICIT opt-out set for the never-drop-RUNNING
    guard. The one legitimate remove path (``_remove_from_pods_conf`` in
    ``pod_lifecycle``) passes ``allow_remove={name}`` for the pod it is
    terminating; every other writer (upsert, cmd_update, refresh-from-api)
    calls the function unchanged and the guard is a no-op on any UPDATE
    (name in both on_disk and new sets → dropped set empty).

    Guard semantics (never-drop-RUNNING). Compute ``dropped = on_disk -
    new - allow_remove``. If non-empty, consult the live RunPod API:

      * ``desiredStatus == "RUNNING"`` → REFUSE the drop, re-add the row
        with its previous host/port from ``on_disk_rows``, WARN loudly on
        stderr naming the pod and the ``allow_remove`` opt-out for the
        legitimate case.
      * Absent from API OR ``desiredStatus != "RUNNING"`` → ALLOW the drop
        (terminated / EXITED / never provisioned).
      * ``RunPodError`` on the API call → FAIL TOWARD KEEPING. Re-add
        every dropped row and WARN. An unreachable API cannot disprove
        RUNNING; assuming NOT-RUNNING would violate the invariant exactly
        when the network is flaky.

    Atomic write. Write payload to ``<path>.tmp`` then ``os.replace(tmp,
    path)`` — POSIX-guaranteed atomic rename on the same filesystem; no
    partial-file reader ever observes torn content.
    """
    if path is None:
        path = _resolve_live_pods_conf()

    # ── Never-drop-RUNNING guard (fires BEFORE the write) ────────────────
    on_disk_rows: dict[str, Pod] = {}
    if path.exists():
        # Read the current on-disk rows so we can (a) diff names and (b)
        # re-add a refused-drop row byte-for-byte from what is already on
        # disk (not from a stale caller-side snapshot).
        for existing in parse_pods_conf(path=path):
            on_disk_rows[existing.name] = existing

    on_disk_names = set(on_disk_rows.keys())
    new_names = {p.name for p in pods}
    dropped = on_disk_names - new_names - allow_remove

    if dropped:
        readd = _guard_against_dropping_running(dropped, on_disk_rows)
        if readd:
            # De-dup by name against the incoming ``pods`` list — a caller
            # that both wants to update a row AND happens to have dropped
            # another one shouldn't get the survivor added twice.
            existing_new = {p.name for p in pods}
            pods = list(pods) + [p for p in readd if p.name not in existing_new]

    # ── Preserve existing header comments (unchanged behavior) ───────────
    header_lines: list[str] = []
    if path.exists():
        for raw in path.read_text().splitlines():
            if raw.startswith("#"):
                header_lines.append(raw)
            else:
                break
    if not header_lines:
        header_lines = [
            "# Pod registry -- SINGLE SOURCE OF TRUTH for all pod configuration.",
            "# All other configs (~/.ssh/config, .claude/mcp.json) are generated from this file.",
            "# Run `python scripts/pod_config.py --sync` after editing.",
            "#",
            "# Format: name  host  port  gpus  gpu_type  label",
        ]

    # Compute column widths for aligned output.
    rows = [(p.name, p.host, str(p.port), str(p.gpus), p.gpu_type, p.label) for p in pods]
    widths = [max(len(r[i]) for r in rows) for i in range(6)] if rows else [0] * 6

    lines = list(header_lines)
    for row in rows:
        parts = [row[i].ljust(widths[i]) for i in range(6)]
        lines.append("  ".join(parts).rstrip())

    payload = "\n".join(lines) + "\n"

    # ── Atomic write via ``os.replace`` on same filesystem ───────────────
    # A crash mid-write can leave the ``<path>.tmp`` behind; a subsequent
    # writer overwrites it via the same tmp path. No corrupt ``pods.conf``
    # ever surfaces to a reader.
    #
    # r2 minor fix: on an ``os.replace`` failure (rare — a same-FS rename
    # that races with a concurrent unlink, an EIO on the target inode) the
    # tmp payload is orphaned in the same directory. Best-effort unlink so
    # we don't leave a stale ``pods.conf.tmp`` sitting next to the target.
    # The target-unchanged atomicity contract is preserved: either the
    # replace succeeded (target now holds ``payload``) or it did not (target
    # unchanged; we cleaned up the tmp).
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(payload)
    try:
        os.replace(tmp, path)
    except OSError:
        # Best-effort tmp cleanup — never mask the original OSError.
        with contextlib.suppress(FileNotFoundError, OSError):
            tmp.unlink()
        raise


def _atomic_write_text(path: Path, payload: str, *, default_mode: int = 0o600) -> None:
    """Atomically write ``payload`` to ``path`` via same-dir tmp + os.replace.

    Mirrors write_pods_conf's tmp+replace+cleanup pattern (task #831). Mode
    handling: if ``path`` exists, its current mode is copied onto the tmp
    BEFORE the replace (a pre-existing 0644 ~/.ssh/config stays 0644); on
    create, the tmp gets ``default_mode`` (0600 — ssh refuses group/other-
    writable configs). The tmp is CREATED with the target mode via
    ``os.open`` so there is no umask window where a group-readable tmp holds
    the payload. No reader ever observes torn content; on a replace failure
    the tmp is best-effort unlinked and the original error re-raised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mode = os.stat(path).st_mode & 0o7777 if path.exists() else default_mode
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.chmod(tmp, mode)  # ensure exact mode even under restrictive umask
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(FileNotFoundError, OSError):
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# SSH config generation
# ---------------------------------------------------------------------------


def _ssh_entry(pod: Pod) -> str:
    """Return the SSH config block for a single pod."""
    return (
        f"# {pod.label} - {pod.gpus}x {pod.gpu_type}\n"
        f"Host {pod.name}\n"
        f"    HostName {pod.host}\n"
        f"    Port {pod.port}\n"
        f"    User {SSH_USER}\n"
        f"    IdentityFile {SSH_KEY}\n"
        f"    StrictHostKeyChecking no\n"
        f"    ConnectTimeout 10\n"
        f"    ServerAliveInterval 60\n"
        f"    ServerAliveCountMax 3"
    )


def _generate_managed_block(pods: list[Pod]) -> str:
    """Return the full managed block including markers."""
    inner = "\n\n".join(_ssh_entry(p) for p in pods)
    return (
        f"{BEGIN_MARKER}\n"
        f"# Auto-generated from pods.conf -- do not edit manually.\n"
        f"# Regenerate: python scripts/pod_config.py --sync\n"
        f"\n"
        f"{inner}\n"
        f"{END_MARKER}"
    )


def update_ssh_config(pods: list[Pod]) -> list[str]:
    """Replace the managed block in ~/.ssh/config. Returns list of change descriptions."""
    changes: list[str] = []
    new_block = _generate_managed_block(pods)

    if not SSH_CONFIG.exists():
        _atomic_write_text(SSH_CONFIG, new_block + "\n")
        changes.append(f"~/.ssh/config: created with {len(pods)} pod entries")
        return changes

    content = SSH_CONFIG.read_text()

    if BEGIN_MARKER in content and END_MARKER in content:
        # Replace existing managed block.
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        new_content = pattern.sub(new_block, content)
        if new_content == content:
            changes.append("~/.ssh/config: already up to date")
        else:
            _atomic_write_text(SSH_CONFIG, new_content)
            changes.append("~/.ssh/config: updated managed pod block")
    else:
        # No markers found -- append the managed block. Compose the FULL
        # payload (existing content + managed block) and hand it to ONE
        # atomic write — never an append-mode write (task #831).
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + new_block + "\n"
        _atomic_write_text(SSH_CONFIG, content)
        changes.append("~/.ssh/config: appended managed block (markers added)")

    return changes


# ---------------------------------------------------------------------------
# SSH config parsing (for --check)
# ---------------------------------------------------------------------------


def _parse_ssh_config_pods() -> dict[str, tuple[str, int]]:
    """Parse ~/.ssh/config and extract pod entries. Returns {name: (host, port)}."""
    if not SSH_CONFIG.exists():
        return {}

    result: dict[str, tuple[str, int]] = {}
    current_host: str | None = None
    current_hostname: str | None = None
    current_port = 22

    for line in SSH_CONFIG.read_text().splitlines():
        stripped = line.strip()

        # New Host block (skip wildcard Host *)
        if stripped.startswith("Host ") and not stripped.startswith("Host *"):
            # Flush previous
            if current_host and POD_NAME_RE.match(current_host):
                result[current_host] = (current_hostname or "", current_port)
            alias = stripped.split(None, 1)[1].strip()
            current_host = alias if POD_NAME_RE.match(alias) else None
            current_hostname = None
            current_port = 22
        elif current_host:
            if stripped.startswith("HostName "):
                current_hostname = stripped.split(None, 1)[1].strip()
            elif stripped.startswith("Port "):
                with contextlib.suppress(ValueError, IndexError):
                    current_port = int(stripped.split(None, 1)[1].strip())

    # Flush last entry
    if current_host and POD_NAME_RE.match(current_host):
        result[current_host] = (current_hostname or "", current_port)

    return result


# ---------------------------------------------------------------------------
# MCP config generation
# ---------------------------------------------------------------------------


def _generate_mcp_env(pods: list[Pod]) -> dict[str, str]:
    """Build the env dict for the SSH MCP server entry.

    The suffix is `pod.name.upper()` verbatim. mcp-ssh-manager lowercases
    the suffix on parse, so the registered name round-trips to the pod name
    in pods.conf (e.g. `pod-261`, or legacy `epm-issue-261`). An older
    scheme prepended `POD` for every pod, which produced
    `SSH_SERVER_PODepm-issue-261_HOST` — a key the upstream regex
    `[A-Z0-9_]+` silently rejected.
    """
    env: dict[str, str] = {}
    for pod in pods:
        prefix = f"SSH_SERVER_{pod.name.upper()}"
        env[f"{prefix}_HOST"] = pod.host
        env[f"{prefix}_PORT"] = str(pod.port)
        env[f"{prefix}_USER"] = SSH_USER
        env[f"{prefix}_KEYPATH"] = SSH_KEY
        env[f"{prefix}_DEFAULT_DIR"] = REMOTE_DIR
        env[f"{prefix}_PLATFORM"] = "linux"
        env[f"{prefix}_DESCRIPTION"] = f"{pod.label} {pod.gpus}x{pod.gpu_type}"
    return env


def update_mcp_config(pods: list[Pod]) -> list[str]:
    """Update the SSH server env vars in ~/.claude/mcp.json. Returns change descriptions.

    The SSH MCP server (mcp-ssh-manager) lives in the user-level Claude config.
    If it is missing we fail loudly rather than silently skipping, because
    silently skipping creates the long-debugged "ssh tools work locally but not
    after sync" mode.
    """
    changes: list[str] = []

    if not MCP_JSON.exists():
        raise SystemExit(
            f"ERROR: {MCP_JSON} does not exist. The user-level Claude config\n"
            f"is required because the SSH MCP server is registered there.\n"
            f'Create it with at least: {{"mcpServers": {{}}}}'
        )

    try:
        data = json.loads(MCP_JSON.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: {MCP_JSON} JSON parse error: {exc}") from exc

    servers = data.get("mcpServers", {})
    if "ssh" not in servers:
        raise SystemExit(
            f'ERROR: no "ssh" server in {MCP_JSON} mcpServers.\n'
            f"The SSH MCP server (mcp-ssh-manager) must be registered there\n"
            f'so that pod env vars can be wired in. See CLAUDE.md "Remote Pod\n'
            f'Access (SSH MCP)" for the expected entry shape.'
        )

    old_env = servers["ssh"].get("env", {})

    # Strip existing pod env keys:
    #  - permanent SSH_SERVER_POD<N>_*
    #  - canonical ephemeral SSH_SERVER_POD-<N>_* incl. the suffixed
    #    multi-pod-per-issue SSH_SERVER_POD-<N>-<SLUG>_* shape (#1334) —
    #    without this, a terminated suffixed pod's keys accumulate forever
    #  - legacy ephemeral SSH_SERVER_EPM-ISSUE-<N>_* (pre-rename)
    #  - very-legacy ephemeral SSH_SERVER_PODepm-issue-<N>_* (pre-prefix-fix)
    # Keep any non-pod env vars. (The [A-Z0-9-] slug class excludes `_`, so
    # the trailing _HOST/_PORT anchor still terminates the match correctly.)
    pod_key_re = re.compile(
        r"^SSH_SERVER_(?:POD\d+|" + _EPHEMERAL_ENVKEY_PATTERN + r"|EPM-ISSUE-\d+|PODepm-issue-\d+)_"
    )
    preserved_env = {k: v for k, v in old_env.items() if not pod_key_re.match(k)}
    new_pod_env = _generate_mcp_env(pods)
    new_env = {**preserved_env, **new_pod_env}

    if old_env == new_env:
        changes.append(".claude/mcp.json: already up to date")
        return changes

    # Report per-key diffs for visibility.
    all_keys = sorted(set(old_env) | set(new_env))
    for key in all_keys:
        old_val = old_env.get(key)
        new_val = new_env.get(key)
        if old_val is None:
            changes.append(f"  mcp: + {key}={new_val}")
        elif new_val is None:
            changes.append(f"  mcp: - {key} (was {old_val})")
        elif old_val != new_val:
            changes.append(f"  mcp: ~ {key}: {old_val} -> {new_val}")

    servers["ssh"]["env"] = new_env
    _atomic_write_text(MCP_JSON, json.dumps(data, indent=2) + "\n")
    changes.insert(0, ".claude/mcp.json: updated SSH server env vars")

    return changes


# ---------------------------------------------------------------------------
# MCP config parsing (for --check)
# ---------------------------------------------------------------------------


def _parse_mcp_pods() -> dict[str, tuple[str, int]]:
    """Extract pod host/port from .claude/mcp.json. Returns {name: (host, port)}."""
    if not MCP_JSON.exists():
        return {}
    try:
        data = json.loads(MCP_JSON.read_text())
    except json.JSONDecodeError:
        return {}

    env = data.get("mcpServers", {}).get("ssh", {}).get("env", {})
    result: dict[str, tuple[str, int]] = {}

    # Permanent pods:        SSH_SERVER_POD<N>_HOST            -> name "podN"
    # Canonical ephemeral:   SSH_SERVER_POD-<N>_HOST           -> name "pod-N"
    #   (incl. the suffixed  SSH_SERVER_POD-<N>-<SLUG>_HOST    -> name "pod-N-<slug>",
    #    #1334 — suffix.lower() round-trips it with no further change)
    # Legacy ephemeral:      SSH_SERVER_EPM-ISSUE-<N>_HOST     -> name "epm-issue-N"
    # Very-legacy ephemeral: SSH_SERVER_PODepm-issue-<N>_HOST  -> name "epm-issue-N"
    host_key_re = re.compile(
        r"^SSH_SERVER_(?P<suffix>POD\d+|"
        + _EPHEMERAL_ENVKEY_PATTERN
        + r"|EPM-ISSUE-\d+|PODepm-issue-\d+)_HOST$"
    )

    for key, value in env.items():
        m = host_key_re.match(key)
        if not m:
            continue
        suffix = m.group("suffix")
        suffix_lower = suffix.lower()
        # Drop the spurious "pod" prefix from the very-legacy ephemeral shape.
        pod_name = (
            suffix_lower.removeprefix("pod")
            if suffix_lower.startswith("podepm-issue-")
            else suffix_lower
        )
        port_str = env.get(f"SSH_SERVER_{suffix}_PORT", "22")
        try:
            port = int(port_str)
        except ValueError:
            port = 22
        result[pod_name] = (value, port)

    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(pods: list[Pod]) -> None:
    """Print a formatted table of all pods."""
    if not pods:
        print("No pods defined in pods.conf")
        return

    header = ("NAME", "HOST", "PORT", "GPUS", "TYPE", "LABEL")
    rows = [(p.name, p.host, str(p.port), str(p.gpus), p.gpu_type, p.label) for p in pods]
    all_rows = [header, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(6)]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(6)).rstrip()

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))
    print(f"\nTotal: {len(pods)} pods, {sum(p.gpus for p in pods)} GPUs")


def cmd_json(pods: list[Pod]) -> None:
    """Output the pod list as a JSON array to stdout."""
    json.dump([asdict(p) for p in pods], sys.stdout, indent=2)
    print()


def _check_mcp_patch_applied() -> tuple[bool, str]:
    """Verify the mcp-ssh-manager hot-reload patch is still applied to node_modules.

    The patch (patches/mcp-ssh-manager+3.2.2.patch) makes the SSH MCP server
    re-read ~/.claude/mcp.json on mtime change AND accept lowercase + hyphens
    in env-key names. Without it, ephemeral pods (pod-N / epm-issue-N) silently
    fail to register because the upstream regex `[A-Z0-9_]+` rejects them. A
    routine `npm install` in ~/.local would silently revert the patch with no
    error surface — this guard catches that drift.

    Returns (ok, message). ok=True if the sentinel function is present OR if
    node_modules is absent (no MCP install to check). ok=False only when the
    file exists but the sentinel is missing, indicating a reverted patch.
    """
    index_js = Path.home() / ".local" / "node_modules" / "mcp-ssh-manager" / "src" / "index.js"
    if not index_js.exists():
        return True, f"mcp-ssh-manager not installed at {index_js} (skipping patch check)"
    try:
        content = index_js.read_text()
    except OSError as exc:
        return True, f"could not read {index_js}: {exc} (skipping patch check)"
    sentinel = "_hotReloadFromMcpJson"
    if sentinel in content:
        return True, "mcp-ssh-manager hot-reload patch is applied"
    return False, (
        f"PATCH MISSING: {index_js}\n"
        f"  The hot-reload patch has been reverted (likely by `npm install`).\n"
        f"  Without it, ephemeral pods (pod-N / epm-issue-N) are invisible to the SSH MCP server.\n"
        f"  Re-apply with:  patch -p1 -d ~/.local < patches/mcp-ssh-manager+3.2.2.patch"
    )


def cmd_check(pods: list[Pod]) -> None:
    """Compare pods.conf against ~/.ssh/config and .claude/mcp.json, report mismatches."""
    patch_ok, patch_msg = _check_mcp_patch_applied()
    if patch_ok:
        print(patch_msg)
    else:
        print(patch_msg, file=sys.stderr)
    print()

    conf_map = {p.name: (p.host, p.port) for p in pods}
    ssh_map = _parse_ssh_config_pods()
    mcp_map = _parse_mcp_pods()

    all_names = sorted(set(list(conf_map) + list(ssh_map) + list(mcp_map)))
    all_ok = True

    # Table header
    print(f"{'Pod':<8} {'pods.conf':<28} {'~/.ssh/config':<28} {'.claude/mcp.json':<28}")
    print("-" * 92)

    for name in all_names:
        conf = conf_map.get(name)
        ssh = ssh_map.get(name)
        mcp = mcp_map.get(name)

        conf_str = f"{conf[0]}:{conf[1]}" if conf else "MISSING"
        ssh_str = f"{ssh[0]}:{ssh[1]}" if ssh else "MISSING"
        mcp_str = f"{mcp[0]}:{mcp[1]}" if mcp else "MISSING"

        present = [v for v in (conf, ssh, mcp) if v is not None]
        match = len(set(present)) <= 1 and len(present) == 3

        if sys.stdout.isatty():
            marker = "\033[32mOK\033[0m" if match else "\033[31mMISMATCH\033[0m"
        else:
            marker = "OK" if match else "MISMATCH"

        print(f"{name:<8} {conf_str:<28} {ssh_str:<28} {mcp_str:<28} {marker}")

        if not match:
            all_ok = False

    print()
    if all_ok and patch_ok:
        print("All configs in sync.")
    elif not all_ok:
        print("Configs out of sync! Run: python scripts/pod_config.py --sync")
    sys.exit(0 if (all_ok and patch_ok) else 1)


def _sync_audit_log_path() -> Path:
    """Return the sync audit-log path (same dir as the resolved live pods.conf).

    Deriving from ``_resolve_live_pods_conf()`` keeps the audit log next to
    the live state (``<git-common-dir>/eps/sync_audit.log`` in steady state)
    AND honors a test's monkeypatched ``PODS_CONF`` — repointing the conf
    into a tmp dir redirects the audit log there too (task #831).
    """
    return _resolve_live_pods_conf().parent / "sync_audit.log"


def _append_sync_audit_line(pods: list[Pod]) -> None:
    """Best-effort append of ONE audit line per sync (task #831).

    Format: ``ts=<iso8601> pid=<pid> cwd=<cwd> argv=<argv> rows=<names>``.
    Purpose: if the #813 symptom (a live pod's Host entry dropped from
    ``~/.ssh/config``) recurs, the log identifies the racing writer instead
    of restarting the investigation from zero. BEST-EFFORT diagnostics
    channel: any OSError is WARNed to stderr and never breaks the sync.
    """
    try:
        line = (
            f"ts={datetime.now(UTC).isoformat()} "
            f"pid={os.getpid()} cwd={Path.cwd()} "
            f"argv={' '.join(sys.argv)} "
            f"rows={','.join(p.name for p in pods)}\n"
        )
        audit_path = _sync_audit_log_path()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a") as fh:
            fh.write(line)
    except OSError as exc:
        print(
            f"WARN: pod_config: could not append sync audit line: {exc}",
            file=sys.stderr,
        )


def cmd_sync() -> None:
    """Regenerate ~/.ssh/config and .claude/mcp.json from pods.conf.

    Acquires ``locked_pods_conf`` and RE-READS pods.conf under the lock, so
    the managed-block rewrite always operates on the canonical on-disk state
    — a caller-supplied snapshot can never drop a concurrent session's row
    (task #831; incident #813). Reentrant: callers already inside
    ``locked_pods_conf`` (cmd_update, cmd_refresh_from_api, pod_lifecycle
    upsert/remove) nest safely (depth counter, see ``locked_pods_conf``).
    Bonus correctness: rows re-added by write_pods_conf's never-drop-RUNNING
    guard are now picked up by the sync (previously the caller's pre-guard
    list was used and a guard-re-added row was silently absent from
    ~/.ssh/config until the next sync).
    """
    with locked_pods_conf():
        pods = parse_pods_conf()
        print("Syncing configs from pods.conf...")
        print()

        _append_sync_audit_line(pods)
        ssh_changes = update_ssh_config(pods)
        for c in ssh_changes:
            print(f"  {c}")

        mcp_changes = update_mcp_config(pods)
        for c in mcp_changes:
            print(f"  {c}")

        print()
        any_changed = any(
            "up to date" not in c for c in ssh_changes + mcp_changes if "skipped" not in c
        )
        if any_changed:
            print("Done. If MCP config changed, restart the MCP server (/mcp).")
        else:
            print("Everything already in sync.")
        print("Verify with: python scripts/pod_config.py --check")


def _set_manual_override(pod_name: str, *, value: bool) -> str | None:
    """Set or clear ``manual_override`` for ``pod_name`` in pods_ephemeral.json.

    Returns a human-readable status string (printed by callers), or None when
    the file does not exist or the pod is not registered there. Permanent-
    fleet pods like ``pod1``, ``pod2`` are not in the sidecar — they aren't
    subject to live-API drift, so we silently no-op.

    Does NOT auto-create the sidecar; if it is missing, the override flag has
    nothing to protect (no auto-refresh would touch a non-existent entry).

    Task #1183: resolves the LIVE sidecar via ``resolve_live_pods_ephemeral``
    and performs the read-modify-write atomically under ``locked_pods_conf``
    (reentrant — the ``cmd_update`` caller already holds it). The monkeypatch
    fast path is checked BEFORE the lock so a test with a patched
    ``PODS_EPHEMERAL_JSON`` never touches the real shared
    ``scripts/.pods.conf.lock``.
    """
    patched = PODS_EPHEMERAL_JSON != PODS_EPHEMERAL_SEED
    ctx = contextlib.nullcontext() if patched else locked_pods_conf()
    with ctx:
        path = resolve_live_pods_ephemeral()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(
                f"WARNING: {path} JSON parse error: {exc}; "
                f"could not set manual_override for {pod_name}.",
                file=sys.stderr,
            )
            return None

        pods = data.get("pods", {})
        if pod_name not in pods:
            return None

        prev = bool(pods[pod_name].get("manual_override", False))
        if prev == value:
            return f"pods_ephemeral.json: manual_override for {pod_name} already {value}"
        pods[pod_name]["manual_override"] = value
        # 0o644 mirrors today's plain write_text mode — the JSON holds no
        # secrets and external read-only tooling may consult it.
        _atomic_write_text(path, json.dumps(data, indent=2) + "\n", default_mode=0o644)
        return f"pods_ephemeral.json: manual_override for {pod_name} {prev} -> {value}"


def cmd_update(pods: list[Pod], pod_name: str, host: str | None, port: int | None) -> None:
    """Update a pod's host/port in pods.conf, then sync all downstream configs.

    Also flips ``manual_override=True`` in pods_ephemeral.json for matching
    ephemeral pods so the auto-refresh paths in ``pod_lifecycle.py`` will not
    silently clobber the manual values from a later ``provision`` / ``resume``
    / cron run. Permanent-fleet pods (``podN``) are not in the sidecar; the
    flag is a no-op there.

    The pre-validation pass uses ``pods`` (already parsed by ``main`` for
    arg-flag checks). The actual read-modify-write-sync runs inside
    ``locked_pods_conf`` after re-reading ``pods.conf`` so a concurrent
    writer cannot interleave between our parse and our write.
    """
    if host is None and port is None:
        print("ERROR: --update requires at least one of --host or --port", file=sys.stderr)
        sys.exit(1)

    if not any(p.name == pod_name for p in pods):
        print(f"ERROR: pod '{pod_name}' not found in pods.conf", file=sys.stderr)
        print(f"Available: {', '.join(p.name for p in pods)}", file=sys.stderr)
        sys.exit(1)

    with locked_pods_conf():
        # Re-parse under the lock so we operate on the freshest on-disk view
        # (a concurrent provision / terminate may have written between
        # ``main``'s parse and our acquisition of the lock).
        fresh = parse_pods_conf()
        target = next((p for p in fresh if p.name == pod_name), None)
        if target is None:
            # Concurrent terminate between main's parse and ours.
            print(
                f"ERROR: pod '{pod_name}' no longer in pods.conf "
                f"(removed by a concurrent writer between read and update).",
                file=sys.stderr,
            )
            sys.exit(1)

        changes: list[str] = []
        if host is not None and host != target.host:
            changes.append(f"  {pod_name} host: {target.host} -> {host}")
            target.host = host
        if port is not None and port != target.port:
            changes.append(f"  {pod_name} port: {target.port} -> {port}")
            target.port = port

        if not changes:
            print(f"{pod_name}: already has those values, nothing to update.")
            return

        print("Updating pods.conf:")
        for c in changes:
            print(c)
        write_pods_conf(fresh)

        # Mark the sidecar so a later auto-refresh in pod_lifecycle.py does NOT
        # silently overwrite the values just set. No-op for permanent pods.
        status = _set_manual_override(pod_name, value=True)
        if status is not None:
            print(f"  {status}")

        print()

        # Auto-sync downstream configs (still inside the lock; cmd_sync
        # re-reads pods.conf under the reentrant lock, so it sees exactly
        # what write_pods_conf just wrote plus any guard re-adds).
        cmd_sync()


def cmd_clear_override(pod_name: str) -> None:
    """Clear ``manual_override`` for ``pod_name`` in pods_ephemeral.json.

    Call this when the manually-set values are no longer correct (e.g., the
    pod the user pointed at has been terminated and they want a future
    ``resume`` to repoint from the live API). No-op for permanent or
    unregistered pods.
    """
    status = _set_manual_override(pod_name, value=False)
    if status is None:
        print(
            f"{pod_name}: not in pods_ephemeral.json — nothing to clear "
            f"(permanent-fleet pods like pod1/pod2 do not carry the flag).",
            file=sys.stderr,
        )
        return
    print(status)


def _read_manual_overrides() -> dict[str, bool]:
    """Read ``manual_override`` flags for every pod in pods_ephemeral.json.

    Permanent-fleet pods (``pod1``..``pod5``) are not in the sidecar, so they
    are simply absent from the returned dict (callers default to False).
    Returns an empty dict when the sidecar is missing or malformed — same
    fail-quiet shape ``_set_manual_override`` uses on read.

    Read-only: resolves the LIVE sidecar via ``resolve_live_pods_ephemeral``
    and takes NO lock (same policy as read-only ``parse_pods_conf`` callers —
    atomic writes guarantee no torn reads; task #1183).
    """
    path = resolve_live_pods_ephemeral()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"WARNING: {path} JSON parse error: {exc}; "
            f"treating all manual_override flags as False.",
            file=sys.stderr,
        )
        return {}
    pods = data.get("pods", {}) or {}
    return {name: bool(entry.get("manual_override", False)) for name, entry in pods.items()}


def _refresh_one_pod(
    name: str,
    row: Pod | None,
    live: PodInfo | None,
    *,
    is_single_mode: bool,
    manual_override: bool,
    to_add: list[Pod] | None = None,
) -> tuple[bool, bool]:
    """Evaluate one pod for ``cmd_refresh_from_api``.

    Returns ``(changed, warned)``. Mutates ``row.host`` / ``row.port`` in
    place on a clean live-API update. Calls ``sys.exit(1)`` when the named
    pod fails a precondition in single-pod mode (the user explicitly named
    a pod we cannot refresh — silently no-oping would be misleading).

    Precondition order: row exists in pods.conf → pod present in live API →
    ``desiredStatus == RUNNING`` → ``ssh_host``/``ssh_port`` populated →
    ``manual_override`` not set → values actually differ. Any failure in
    bulk mode skips with a stderr WARN (sets ``warned=True``).

    Task #821: RE-ADD branch. When ``row is None`` (the pod is absent from
    ``pods.conf`` — the incident signature) and the live API says the pod is
    RUNNING with a populated SSH endpoint, this function APPENDS a fresh
    ``Pod(...)`` to the caller-supplied ``to_add`` list and returns
    ``(changed=True, warned=False)``. The caller then merges ``to_add`` into
    the rows it hands to ``write_pods_conf``, closing the incident #821
    self-heal path (``poll_pipeline.py`` runs ``pod.py config
    --refresh-from-api`` after 10 consecutive SSH failures — this branch
    lets it heal a wiped row without human intervention).

    ``to_add=None`` in the legacy path preserves the pre-#821 behavior of
    treating a missing row as a stale skip (used by single-pod recovery
    where the caller has already validated the pod is in pods.conf).
    """
    if row is None:
        # Task #821 re-add path. Attempt to re-add ONLY when the live API
        # confirms the pod is RUNNING with a valid SSH endpoint AND the
        # caller opted into the re-add pattern by passing ``to_add``.
        if to_add is not None and live is not None:
            ds = (live.desired_status or "").upper()
            if ds == "RUNNING" and live.ssh_host and live.ssh_port:
                # NOTE (fact-check A8): ``PodInfo.gpu_type_id`` (not
                # ``gpu_type``) may be ``None``; fall back to "unknown" so
                # the required ``Pod.gpu_type: str`` slot is populated.
                # ``live.name`` is the human-readable RunPod label; if
                # absent, synthesise a "restored:<pod-name>" label so the
                # row is legible in ``--list`` output.
                new_row = Pod(
                    name=name,
                    host=live.ssh_host,
                    port=live.ssh_port,
                    gpus=(live.gpu_count if live.gpu_count is not None else 1),
                    gpu_type=(live.gpu_type_id or "unknown"),
                    label=(live.name or f"restored:{name}"),
                )
                to_add.append(new_row)
                print(
                    f"  INFO: re-adding missing RUNNING pod '{name}' to "
                    f"pods.conf from live API "
                    f"(host={live.ssh_host}:{live.ssh_port}).",
                )
                return True, False
        # Concurrent terminate between main's parse and ours, OR live API
        # says NOT RUNNING for a missing row (nothing to restore).
        print(
            f"WARN: pod '{name}' no longer in pods.conf (removed by a "
            f"concurrent writer between read and refresh); skipping.",
            file=sys.stderr,
        )
        return False, True

    if live is None:
        msg = (
            f"WARN: pod '{name}' is in pods.conf but not in the live "
            f"RunPod API (terminated externally or never created); "
            f"skipping. Run `pod.py terminate --issue <N>` to clean up "
            f"the stale row, or `pod.py provision` to re-create it."
        )
        if is_single_mode:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return False, True

    ds = (live.desired_status or "").upper()
    if ds != "RUNNING":
        msg = (
            f"WARN: pod '{name}' has desiredStatus={ds or 'UNKNOWN'}, "
            f"not RUNNING; SSH endpoint is not available, skipping. "
            f"Run `pod.py resume --issue <N>` to bring it back, then "
            f"re-run --refresh-from-api."
        )
        if is_single_mode:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return False, True

    if live.ssh_host is None or live.ssh_port is None:
        # RunPod has the pod RUNNING but the 22/tcp mapping isn't up yet
        # (transient). Don't blank out the existing row.
        msg = (
            f"WARN: pod '{name}' is RUNNING but has no public 22/tcp "
            f"mapping yet (transient — wait ~10s and retry); skipping."
        )
        if is_single_mode:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return False, True

    if manual_override:
        if row.host != live.ssh_host or row.port != live.ssh_port:
            print(
                f"WARN: pod '{name}' has manual_override=True; refusing "
                f"to overwrite host/port from API "
                f"(kept {row.host}:{row.port}; API would have written "
                f"{live.ssh_host}:{live.ssh_port}). Clear with "
                f"`pod.py config --clear-override {name}` if the API "
                f"is right.",
                file=sys.stderr,
            )
            return False, True
        return False, False

    if row.host == live.ssh_host and row.port == live.ssh_port:
        print(f"  {name}: already at {row.host}:{row.port} — no change.")
        return False, False

    print(f"  {name}: {row.host}:{row.port} -> {live.ssh_host}:{live.ssh_port}")
    row.host = live.ssh_host
    row.port = live.ssh_port
    return True, False


def cmd_refresh_from_api(pods: list[Pod], pod_name: str | None) -> None:
    """Pull live host/port from the RunPod API and update ``pods.conf``.

    The existing ``--sync`` propagates ``pods.conf`` OUTWARD to ``~/.ssh/config``
    + ``.claude/mcp.json``. There was no inverse direction: nothing pulled
    fresh host/port from the live RunPod API into ``pods.conf``. The gap bit
    task #488 on 2026-06-09 — a SUPPLY_CONSTRAINT-blocked resume hard-exited,
    the pod later came back at a NEW SSH port via a separate retry that did
    not run our success path, and the autonomous session's SSH polling loop
    spun for 13+ hours on the pre-stop port while ``pods.conf`` carried the
    stale value. With this command, the orchestrator (or a human) can force
    a re-sync from the live API and ``cmd_sync`` then propagates the fresh
    values to SSH + MCP.

    Scope:
      * ``pod_name=None`` — refresh every managed pod present in BOTH
        ``pods.conf`` and the live RunPod API. Pods that are not RUNNING are
        skipped with a stderr note (we cannot infer a fresh SSH endpoint for
        a pod that is EXITED/PROVISIONING).
      * ``pod_name=<name>`` — refresh just that pod. Errors loud if the pod
        is not in ``pods.conf`` (typo) or not present in the live API
        (terminated externally) or not RUNNING (cannot refresh an endpoint
        that does not exist yet).

    Respects ``manual_override`` (set by ``--update``): when True, the
    on-disk host/port stays as the user set them and we surface a stderr
    WARN instead of overwriting. Use ``--clear-override <pod>`` to re-enable
    auto-refresh for a manually-pinned pod.

    Holds ``locked_pods_conf`` for the whole read-modify-write-sync sequence
    so a concurrent provision/resume cannot lose-update our changes — the
    same lock discipline ``cmd_update`` uses.

    The live API call is REQUIRED. If the API is unreachable, the underlying
    ``runpod_api.RunPodError`` propagates so callers see a clear failure
    rather than a silent stale-config no-op (fail-fast rule).
    """
    # Import lazily — ``runpod_api`` is the heavy module and importing at
    # module top would force every ``pod_config --check`` / ``--list`` to
    # eagerly load it. The lazy import keeps the cheap subcommands cheap.
    _ensure_scripts_dir_on_sys_path()
    from runpod_api import list_team_pods

    live_pods = list_team_pods()
    live_by_name = {p.name: p for p in live_pods}
    overrides = _read_manual_overrides()

    is_single_mode = pod_name is not None

    with locked_pods_conf():
        # Re-parse under the lock so we operate on the freshest on-disk view.
        fresh = parse_pods_conf()
        fresh_by_name = {p.name: p for p in fresh}

        # Build the target set.
        #
        # Task #821: in BULK mode, enumerate LIVE-API managed pods (matching
        # ``pod_lifecycle._MANAGED_PREFIXES`` naming — canonical ``pod-<N>``)
        # that are absent from pods.conf so the re-add path in
        # ``_refresh_one_pod`` can restore them. This is the piece that
        # closes the incident #821 self-heal loop: ``poll_pipeline.py`` runs
        # ``pod.py config --refresh-from-api`` after 10 consecutive SSH
        # failures — after a wipe, the target row is not in pods.conf so
        # the pre-#821 iteration missed it entirely. Now bulk mode reaches
        # the wiped rows via the live API.
        target_names: list[str] = []
        seen: set[str] = set()
        if pod_name is None:
            # Bulk mode: (a) every row currently in pods.conf,
            # (b) every managed live pod absent from pods.conf.
            for p in pods:
                if p.name not in seen:
                    target_names.append(p.name)
                    seen.add(p.name)
            # The ephemeral name grammar (incl. the pod-<N>-<slug> suffixed
            # form, #1334) mirrors ``pod_lifecycle._POD_NAME_RE``: only
            # ephemeral, project-managed pods. A random RunPod entry from
            # the team account (permanent fleet, another user's pod,
            # ``thomas-pod-475``, ``pod-abc``, a numeric-slug ``pod-779-60``)
            # is NEVER auto-added to pods.conf — bulk mode is safe by default.
            managed_re = re.compile(r"^" + _EPHEMERAL_NAME_PATTERN + r"$")
            for live_name in sorted(live_by_name):
                if live_name in seen:
                    continue
                if not managed_re.match(live_name):
                    continue
                target_names.append(live_name)
                seen.add(live_name)
        else:
            # Single-pod mode: trust the caller's intent. The pod may be
            # absent from pods.conf (re-add path) but MUST be present in
            # the live API (or the classic "not in API" fail-loud message
            # fires below in ``_refresh_one_pod``).
            target_names = [pod_name]

        if not target_names:
            print("No pods to refresh (pods.conf is empty and live API has no managed pods).")
            return

        any_changed = False
        any_warn = False
        to_add: list[Pod] = []
        for name in target_names:
            row = fresh_by_name.get(name)
            live = live_by_name.get(name)
            if row is None and live is None and is_single_mode:
                # Single-pod mode + user typo: neither pods.conf nor the
                # live API has the named pod. Fail loud like the original
                # single-mode contract.
                print(
                    f"ERROR: pod '{name}' not found in pods.conf or the "
                    f"live RunPod API. Check spelling; the pod name must "
                    f"be a managed pod (e.g. 'pod-488').",
                    file=sys.stderr,
                )
                sys.exit(1)
            changed, warned = _refresh_one_pod(
                name,
                row,
                live,
                is_single_mode=is_single_mode,
                manual_override=overrides.get(name, False),
                to_add=to_add,
            )
            any_changed = any_changed or changed
            any_warn = any_warn or warned

        # Merge any re-added rows into ``fresh`` before writing.
        if to_add:
            fresh = list(fresh) + to_add

        if not any_changed:
            if not any_warn:
                print("All managed pods already match the live RunPod API.")
            else:
                print(
                    "No host/port changes applied (see warnings above).",
                    file=sys.stderr,
                )
            return

        print()
        print("Updating pods.conf with live API host/port...")
        write_pods_conf(fresh)
        cmd_sync()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pod config manager -- keeps SSH and MCP configs in sync with pods.conf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/pod_config.py --list\n"
            "  python scripts/pod_config.py --check\n"
            "  python scripts/pod_config.py --sync\n"
            "  python scripts/pod_config.py --update pod2 --host 1.2.3.4 --port 12345\n"
            "  python scripts/pod_config.py --refresh-from-api\n"
            "  python scripts/pod_config.py --refresh-from-api pod-488\n"
            "  python scripts/pod_config.py --json\n"
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="Show all pods in a table")
    group.add_argument("--json", action="store_true", help="Output pod list as JSON")
    group.add_argument(
        "--check", action="store_true", help="Verify SSH and MCP configs match pods.conf"
    )
    group.add_argument(
        "--sync", action="store_true", help="Regenerate SSH and MCP configs from pods.conf"
    )
    group.add_argument("--update", metavar="POD_NAME", help="Update a pod's host/port, then sync")
    group.add_argument(
        "--clear-override",
        metavar="POD_NAME",
        help=(
            "Clear manual_override in pods_ephemeral.json for POD_NAME so the "
            "auto-refresh paths in pod_lifecycle.py may resume updating host/"
            "port from the live API."
        ),
    )
    group.add_argument(
        "--refresh-from-api",
        metavar="POD_NAME",
        nargs="?",
        const="__ALL__",
        help=(
            "Pull live host/port from the RunPod API into pods.conf, then "
            "sync to ~/.ssh/config + .claude/mcp.json. Pass a POD_NAME to "
            "refresh just one pod, or omit it to refresh every managed pod. "
            "Respects manual_override (set by --update). Use when a pod has "
            "come back at a new SSH port outside an explicit `pod.py resume` "
            "(e.g. recovery from SUPPLY_CONSTRAINT) and the configs are stale."
        ),
    )

    parser.add_argument("--host", help="New host (IP) for --update")
    parser.add_argument("--port", type=int, help="New port for --update")

    args = parser.parse_args()

    pods = parse_pods_conf()

    if args.list:
        cmd_list(pods)
    elif args.json:
        cmd_json(pods)
    elif args.check:
        cmd_check(pods)
    elif args.sync:
        cmd_sync()
    elif args.update:
        cmd_update(pods, args.update, args.host, args.port)
    elif args.clear_override:
        cmd_clear_override(args.clear_override)
    elif args.refresh_from_api:
        # ``nargs="?"`` with ``const="__ALL__"`` distinguishes the bare flag
        # (refresh all pods) from the flag-with-arg (refresh one pod).
        target = None if args.refresh_from_api == "__ALL__" else args.refresh_from_api
        cmd_refresh_from_api(pods, target)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
