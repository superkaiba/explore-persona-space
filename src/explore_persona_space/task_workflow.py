"""Repo-native task workflow — local-file replacement for sagan_state.py.

This module is the active state surface for `/issue` after the Sagan
migration. All state lives in the repo:

    tasks/<status>/<id>/
        body.md           # YAML frontmatter + content
        events.jsonl      # append-only progress log (same epm:* shape as Sagan)
        comments.jsonl    # mentor comments + Claude replies
        plans/v{N}.md     # plan rounds
        plan.md           # symlink → latest plans/v{N}.md
        original-body.md  # snapshot before clean-result promotion
        artifacts/        # figures, etc.

    tasks/REGISTRY.json   # {"highest_id": N, "tasks": {id: {path, title, kind}}}

body.md frontmatter is permissive freeform YAML — unknown keys are
preserved verbatim on every read/mutate/write round-trip (no whitelist,
no validation). Common fields: ``title``, ``kind``, ``tags``,
``created_at``, ``has_clean_result``, ``goal`` (experiments), ``parent_id``,
``classification``/``promoted_at`` (post-promotion). An optional
``relates_to`` field — a flat list of stable open-question id strings (no
primary/secondary; default ``[]``) — links an experiment to the living-docs
open questions it bears on (see
docs/living-docs-workflow-integration-plan.md); read it with
``get_relates_to`` and write it via ``scripts/living_docs.py``.

Single writer per file: this module holds a flock on `~/.task-workflow/lock`
for the duration of any mutation, so /issue sessions and the tunnel handler
serialise naturally. Every mutation is one git commit (auto-push optional via
`AUTO_PUSH` env var).

Usage from Python:

    from explore_persona_space.task_workflow import (
        find_task_path, get_task, set_status, post_event,
        create_task, promote, latest_event, list_by_status,
    )

    task = get_task(413)
    print(task["frontmatter"]["status"], task["frontmatter"]["title"])
    post_event(413, "epm:run-launched", note="...")
    set_status(413, "running")

The CLI (`scripts/task.py`) is a thin argparse wrapper around these
functions and matches the sagan_state.py subcommand surface 1:1.

Concurrency: all writes go through `_locked()` which holds an exclusive
flock on ~/.task-workflow/lock. Reads do NOT lock. body.md / REGISTRY
writes are atomic (write-temp + rename), so readers see a consistent
snapshot. Append-only JSONL logs (events.jsonl / comments.jsonl) instead
use `O_APPEND` writes (`_append_jsonl_line`): a `<= PIPE_BUF` line lands
all-or-nothing against a SIGKILL, while a `> PIPE_BUF` line is NOT
crash-atomic and can leave a partial trailing line — the tolerant reader
(`_iter_jsonl`, `errors="replace"`) skips it, so JSONL readers still see a
consistent (everything-parseable) snapshot.

Status enum (folder names):
  on_hold proposed planning plan_pending approved running verifying
  interpreting reviewing awaiting_promotion completed blocked archived

`on_hold` is a non-lifecycle parking status: tasks explicitly set aside
("on hold for now") that are NOT part of the active proposed queue and are
excluded from auto-dispatch / the clarifier. Revivable via
`set_status <N> proposed`.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ─── Config / paths ────────────────────────────────────────────────────────

STATUSES = (
    # Non-lifecycle parking status: tasks explicitly set aside ("on hold
    # for now"), kept OUT of the active proposed queue and excluded from
    # auto-dispatch / the clarifier. Sits left of `proposed` on the board.
    # Revivable via set_status(<N>, "proposed").
    "on_hold",
    "proposed",
    "planning",
    "plan_pending",
    "approved",
    "running",
    "verifying",
    "interpreting",
    "reviewing",
    "awaiting_promotion",
    # A same-issue follow-up round is executing on this task (tagged
    # `followup-auto` | `followup-manual`); legacy semantics: parent complete
    # with `parent_id` children still in flight. NOT terminal, NOT the park
    # status. Un-phantomed 2026-06-10 (was previously only in workflow.yaml).
    "followups_running",
    "completed",
    "blocked",
    "archived",
)

TERMINAL_STATUSES = frozenset({"completed", "blocked", "archived"})

# Canonical task `kind` enum — the single source of truth shared by
# `task.py new --kind`, `task.py set-kind`, and `set_kind()`. Mirrors the
# routing law in CLAUDE.md § "Routing experiment intent": `experiment` (a
# research question that produces a promotable clean-result) vs the
# code-change kinds (`infra | analysis | survey | batch`) that complete on
# the Step 9c test-verdict path with NO promotable clean-result — a
# fix-validation / "test that X works" task is `kind: infra`, not
# `experiment` (incident #672: a GCP-fix validation was mis-filed as an
# experiment and dragged through the clean-result/promotion machinery).
# `campaign` is the question-level runner (/campaign <N>). The remaining
# entries are the Human-board kinds (note/reading/idea/question/decision).
KINDS = (
    "experiment",
    "infra",
    "analysis",
    "batch",
    "survey",
    "campaign",
    "note",
    "reading",
    "idea",
    "question",
    "decision",
)

# Status that means "user has reviewed and approved a clean-result body; user
# must run `task.py promote` to move to completed". Park-and-wait gate.
PARK_STATUS = "awaiting_promotion"

# Workflow-pipeline versions a task can be pinned to (EPS workflow-v2 plan,
# Assumption 2). "v1" is the current pipeline; "v2" is the report-only
# pipeline the `/issue` dispatcher branches to when a task's frontmatter
# carries `workflow: v2`. The default for a NEW task resolves as
# explicit-arg > env EPM_DEFAULT_WORKFLOW > DEFAULT_WORKFLOW_VERSION; the
# flip of the default to "v2" is a later one-line env/config change after the
# dogfood, NOT wired here. `workflow_version()` fail-opens to v1 so legacy
# tasks (no `workflow:` key) resolve to the current pipeline everywhere.
WORKFLOW_VERSIONS = ("v1", "v2")
DEFAULT_WORKFLOW_VERSION = "v1"


def workflow_version(frontmatter: dict[str, Any]) -> str:
    """Return the workflow-pipeline version a task is pinned to.

    Reads the ``workflow`` frontmatter key and fail-OPENS to
    :data:`DEFAULT_WORKFLOW_VERSION` ("v1") for an absent, empty, or unknown
    value — so legacy tasks (which have no ``workflow:`` key) resolve to the
    current v1 pipeline everywhere and a garbage value never crashes a caller.
    """
    value = frontmatter.get("workflow")
    if isinstance(value, str) and value.strip() in WORKFLOW_VERSIONS:
        return value.strip()
    return DEFAULT_WORKFLOW_VERSION


def _resolve_workflow_version(explicit: str | None) -> str:
    """Resolve a NEW task's workflow version at creation time.

    Precedence: explicit arg > env ``EPM_DEFAULT_WORKFLOW`` >
    :data:`DEFAULT_WORKFLOW_VERSION`. An unknown value at any layer falls
    through to the next (fail-open to v1). The CLI validates the explicit arg
    with ``argparse`` choices, so an unknown value here can only reach us from
    a programmatic caller — treat it as unset rather than crash.
    """
    for candidate in (explicit, os.environ.get("EPM_DEFAULT_WORKFLOW")):
        if isinstance(candidate, str) and candidate.strip() in WORKFLOW_VERSIONS:
            return candidate.strip()
    return DEFAULT_WORKFLOW_VERSION


# Intermediate pipeline statuses a `followups_running` task may NOT re-enter
# mid-round. The same-issue follow-up status-hold rule (SKILL.md Step 9b
# § Same-issue follow-up loop, step 3): the round HOLDS `followups_running`
# end-to-end; phase visibility comes from stage breadcrumbs
# (`stage=followup-<phase>`) + `epm:progress` markers, never status flips.
# Exits to `awaiting_promotion` (re-park), `blocked` (failure), `completed` /
# `archived` (terminal), and the deliberate `proposed` reset stay allowed.
# `set_status` refuses these transitions unless `force_followup_exit=True`
# (CLI: `--force-followup-exit`). Incident: tasks #533/#560 (2026-06-10/11)
# flipped to `running` mid-round via Step 4b's local set-status instruction.
FOLLOWUP_HELD_BLOCKED_STATUSES = frozenset(
    {"planning", "plan_pending", "approved", "running", "verifying", "interpreting", "reviewing"}
)

EVENT_NOTE_MAX = 50_000  # mirror Sagan's body-size cap

# Comment kinds the web UI exposes; checked when comments are appended.
COMMENT_KINDS = frozenset({"question", "answer", "followup-proposal", "note"})

# The non-`experiment` lifecycle kinds that complete on the Step 9c
# test-verdict / code-change path (no promotable clean-result) and are
# exempt from the `kind: experiment`-only plan/measurement checks
# (CLAUDE.md Critical Rules: "`kind: analysis|infra|batch|survey` exempt").
# Canonical single source for this subset: `task_progress.CODE_KINDS`
# imports it directly, and `verify_plan.EXEMPT_KINDS` is pinned equal to it by
# a drift test, so the three copies can never drift (incident #672: a `kind`
# enum drift shipped a `batch`-missing-from-CLI bug). Membership is
# byte-identical to the prior literals; `verify_plan.VALID_KINDS` is
# `("experiment", *CODE_KINDS)` and stays an explicit ordered tuple (argparse
# `choices=` display order) pinned by the same drift test.
CODE_KINDS = frozenset({"infra", "analysis", "batch", "survey"})


# ─── Repo / tasks-dir resolution ────────────────────────────────────────────
#
# Background. `tasks/` is canonically owned by the `main` branch of the main
# worktree. If `repo_root()` is invoked from a git worktree on a feature
# branch (e.g. `.claude/worktrees/issue-377` on branch `issue-377`), naive
# resolution via `Path(__file__).resolve()` returns the worktree directory.
# Reads from that path see whatever state was on the worktree branch when it
# was created (stale); writes commit to the worktree branch (stranded). Both
# failure modes have produced data-loss incidents.
#
# The new resolver:
#   (a) Calls `git rev-parse --path-format=absolute --git-common-dir` from
#       the directory containing THIS module (not `os.getcwd()`), with
#       `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`
#       UNSET in the subprocess env so a caller cannot poison rev-parse.
#   (b) Validates the parent: basename `.git`, is a real directory, NOT
#       inside `.git/modules/<name>` (submodule shape), and contains
#       `tasks/`.
#   (c) Branch-guards: `git -C <parent> symbolic-ref --short HEAD` must
#       return `main`. Non-`main` and detached HEAD raise DISTINCT
#       `RuntimeError`s naming the actual state.
#   (d) Caches via `functools.lru_cache(maxsize=1)` keyed on
#       `(os.getpid(), os.getcwd())` so each Python invocation pays one
#       subprocess pair total; cache invalidates across forks and cwd
#       changes automatically.
#
# Module-level `REPO`, `TASKS_DIR`, `REGISTRY_PATH` attribute access is
# preserved via the PEP-562 `__getattr__` at the bottom of this module, so
# `tw.TASKS_DIR` continues to work. `from tw import TASKS_DIR` bare-name
# imports bind at import time and PEP-562 cannot rescue them — those
# call-sites are refactored to use the function form and the grep test
# `tests/test_no_direct_task_path_construction.py` keeps new ones out.

_MODULE_DIR = Path(__file__).resolve().parent

# Sanitized env: prevent rev-parse poisoning by caller GIT_* env.
_GIT_ENV_POISONERS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
)


def _sanitized_git_env() -> dict[str, str]:
    env = dict(os.environ)
    for k in _GIT_ENV_POISONERS:
        env.pop(k, None)
    return env


def _resolve_primary_checkout(env: dict[str, str]) -> Path:
    """Resolve + validate the PRIMARY checkout root (the git common dir's
    parent). Verbatim extraction of ``_resolve_repo_root_cached`` steps
    (a)+(b) — the rev-parse + layout validation, everything BEFORE the
    branch guard (#844). Raises ``RuntimeError`` on any failure; never
    falls back to a ``__file__``/cwd walk-up.
    """
    # (a) Locate the common git dir.
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(_MODULE_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("git executable not found on PATH; task.py requires git ≥ 2.31") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`git rev-parse --git-common-dir` failed from {_MODULE_DIR}:\n"
            f"  stdout: {e.stdout!r}\n  stderr: {e.stderr!r}"
        ) from e
    common_dir = Path(proc.stdout.strip())
    # (b) Validate parent.
    if common_dir.name != ".git":
        raise RuntimeError(
            f"git common-dir {common_dir!s} basename is {common_dir.name!r}, expected '.git'; "
            f"bare repo or non-canonical layout — refusing to resolve tasks/."
        )
    if not common_dir.is_dir():
        raise RuntimeError(
            f"git common-dir {common_dir!s} is not a directory; "
            f"corrupt or non-canonical layout — refusing to resolve tasks/."
        )
    # Submodule shape (.git/modules/<name>) is caught by the basename check
    # above: `git rev-parse --git-common-dir` from inside a submodule returns
    # `.../.git/modules/<name>`, whose basename is `<name>`, not `.git`. So
    # the submodule case fails the `common_dir.name != ".git"` check and
    # raises before reaching this point. Verified by
    # ``test_validation_rejects_real_submodule_layout``.
    parent = common_dir.parent
    if not (parent / "tasks").is_dir():
        raise RuntimeError(
            f"resolved repo root {parent!s} has no `tasks/` directory; "
            f"wrong repo or uninitialized layout — refusing to resolve tasks/."
        )
    return parent


@functools.lru_cache(maxsize=1)
def _resolve_repo_root_cached(_key: tuple[int, str]) -> Path:
    """Inner cache target. Keyed on (pid, cwd) so forks + chdirs invalidate
    automatically. The key is computed by the wrapper; we ignore the
    contents (we resolve relative to module dir + sanitized env, not cwd).
    """
    env = _sanitized_git_env()
    # (a)+(b) Locate the common git dir + validate its parent.
    parent = _resolve_primary_checkout(env)
    # (c) Branch guard.
    sym = subprocess.run(
        ["git", "-C", str(parent), "symbolic-ref", "--short", "HEAD"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if sym.returncode == 0:
        branch = sym.stdout.strip()
        if branch != "main":
            # The primary checkout is parked on a real feature branch. Rather
            # than refuse (the historical behavior, which silently dropped
            # markers in ~7 sessions), auto-route every task.py read+write
            # through a dedicated managed worktree pinned to a DETACHED `main`
            # tip. Commits made through that worktree advance the `main` ref
            # (see `_advance_main_ref`), so the guard's INTENT — commits land
            # on main, never strand on a feature branch — is preserved; only
            # the hard refusal is replaced. The `--detach main` pin (not the
            # `main` BRANCH) is deliberate: a worktree holding the `main`
            # branch would block the primary from `git checkout main`
            # ("fatal: 'main' is already checked out at <managed>"), so a
            # leaked managed worktree would brick the user's ability to return
            # to main. A detached pin holds no branch-checkout lock, so a leak
            # is benign. Returns the managed worktree path; `_git_commit`
            # detects routing via `_is_routed_root` and does the
            # reset-to-main / advance-main dance.
            return _ensure_managed_main_worktree(parent, branch, env)
    else:
        # `git symbolic-ref --short HEAD` returns rc=1 with stderr
        # "fatal: ref HEAD is not a symbolic ref" when HEAD is detached.
        # The substring check is the canonical detached-HEAD signal —
        # rc=128 can mean many other things (not a git repo, object
        # missing, …) and we don't want to misclassify those as detached.
        stderr = (sym.stderr or "").lower()
        if "not a symbolic ref" in stderr:
            raise RuntimeError(
                f"main worktree HEAD ({parent}) is detached; "
                f"re-attach to 'main' before running task.py."
            )
        raise RuntimeError(
            f"`git symbolic-ref --short HEAD` failed (rc={sym.returncode}) "
            f"in {parent}:\n  stderr: {sym.stderr!r}"
        )
    return parent


# ─── Off-main auto-routing (managed main-pinned worktree) ───────────────────
#
# When the primary checkout is parked on a feature branch, task.py routes its
# reads + commits through a dedicated managed worktree pinned to a DETACHED
# `main` tip, so commits always advance `main` and never strand on the feature
# branch. The managed worktree lives under `.claude/worktrees/` so the
# stale-worktree audit (which only targets `issue-<N>` / `agent-<hex>` /
# `wf_<id>` names) and the no-direct-path-construction test (which excludes
# `.claude/worktrees/`) both ignore it. The leading underscore keeps it out of
# the audit's `_TARGET_NAME_RE` even if that regex were widened.

# Directory name of the managed worktree (relative to `.claude/worktrees/`).
_MANAGED_MAIN_WORKTREE_NAME = "_task-main-pin"

# Set of resolved repo-root paths that are managed routing worktrees (not the
# primary checkout). `_git_commit` consults this to decide whether to run the
# reset-to-main / advance-main dance. Populated by `_ensure_managed_main_worktree`.
_ROUTED_ROOTS: set[Path] = set()


def _managed_worktree_path(primary: Path) -> Path:
    """Absolute path of the managed main-pinned worktree for ``primary``."""
    return primary / ".claude" / "worktrees" / _MANAGED_MAIN_WORKTREE_NAME


# Cutover migration LOCK (#681). During the data-disk cutover (plan §4 Phase 2)
# the `.claude/worktrees/` tree is copied + bind-swapped onto the dedicated data
# disk. BOTH concurrent worktree-creation writers must refuse while the swap is
# in flight: `scripts/new_worktree.sh` AND this managed-main-pin creation path
# (a `task.py` write mid-swap could create the pin worktree on the soon-to-be-
# renamed `.premigrate` tree and strand task state). Relative to the primary
# checkout, the same file new_worktree.sh checks.
_MIGRATION_LOCK_REL = Path(".claude") / "cache" / "worktree-migration.LOCK"


def _migration_lock_path(primary: Path) -> Path:
    """Absolute path of the cutover migration LOCK for ``primary`` (#681)."""
    return primary / _MIGRATION_LOCK_REL


def _is_routed_root(root: Path) -> bool:
    """True if ``root`` is a managed routing worktree, not the primary checkout.

    Identity is determined structurally (path basename + parent), not only by
    the in-process ``_ROUTED_ROOTS`` set, so a fresh process that re-resolves
    to the managed worktree (e.g. the cache was cleared) is still recognized as
    routed even before ``_ensure_managed_main_worktree`` re-populates the set.
    """
    if root in _ROUTED_ROOTS:
        return True
    return root.name == _MANAGED_MAIN_WORKTREE_NAME and root.parent.name == "worktrees"


def _git_quiet(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a git command (sanitized env) and FAIL LOUD on non-zero exit.

    Used by the managed-worktree lifecycle helpers. Raises ``RuntimeError``
    naming the command + stderr — never silently proceeds past a git failure.
    """
    proc = subprocess.run(
        ["git", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`git {' '.join(args)}` failed (rc={proc.returncode}):\n"
            f"  stdout: {proc.stdout!r}\n  stderr: {proc.stderr!r}"
        )
    return proc


def _ensure_managed_main_worktree(primary: Path, branch: str, env: dict[str, str]) -> Path:
    """Create (or re-sync) the managed main-pinned worktree and return its path.

    Called from the resolver when the primary checkout HEAD is on ``branch``
    (a real feature branch). Guarantees:

      * a worktree exists at ``<primary>/.claude/worktrees/_task-main-pin`` with
        HEAD DETACHED at the current ``main`` tip (a fast-forward each call, so
        reads through the routed root see fresh `main` state);
      * the routed path is recorded in ``_ROUTED_ROOTS`` so ``_git_commit``
        runs the advance-main dance.

    FAILS LOUD (RuntimeError) on any git failure — never silently falls back to
    the primary checkout (that would re-introduce the stranded-commit bug the
    routing exists to prevent). If `main` does not exist as a branch, raises.

    Refuses (RuntimeError) while the #681 cutover migration LOCK is held: a
    managed-pin worktree created mid-swap could land on the soon-to-be-renamed
    `.premigrate` tree and strand task state (Codex freeze-audit concern, plan
    §4 Phase 4 / §6 step 1). The LOCK lifts the moment the bind-swap completes.
    """
    lock = _migration_lock_path(primary)
    if lock.exists():
        raise RuntimeError(
            f"worktree migration in progress ({lock} exists) — refusing to create the "
            f"managed main-pin worktree mid-cutover (it could strand task state on the "
            f"renamed .premigrate tree). Retry once the data-disk cutover lifts the LOCK."
        )

    # `main` must exist as a local branch to pin to.
    show = subprocess.run(
        ["git", "-C", str(primary), "rev-parse", "--verify", "--quiet", "refs/heads/main"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if show.returncode != 0:
        raise RuntimeError(
            f"primary checkout {primary} is on {branch!r} and has no local `main` branch to "
            f"route task.py writes through; create `main` (or check it out) before running task.py."
        )

    managed = _managed_worktree_path(primary)
    git_dir = managed / ".git"
    if not git_dir.exists():
        # Stale registration (dir removed out-of-band but git still lists it)
        # would make `worktree add` refuse with "already registered"; prune
        # first so the add is clean. Prune is a no-op when nothing is stale.
        _git_quiet(["-C", str(primary), "worktree", "prune"], env)
        managed.parent.mkdir(parents=True, exist_ok=True)
        _git_quiet(
            ["-C", str(primary), "worktree", "add", "--detach", "--force", str(managed), "main"],
            env,
        )
    else:
        # Re-sync an existing managed worktree to the current `main` tip so
        # reads through the routed root are fresh. `reset --hard main` is a
        # fast-forward (the worktree only ever holds main-derived commits) and
        # is safe under the flock: every mutation commits before releasing, so
        # there is never uncommitted task work to clobber here.
        _git_quiet(["-C", str(managed), "reset", "--hard", "main"], env)

    if not (managed / "tasks").is_dir():
        raise RuntimeError(
            f"managed main-pin worktree {managed} has no `tasks/` directory after sync; "
            f"refusing to route task.py writes through a malformed worktree."
        )
    _ROUTED_ROOTS.add(managed)
    return managed


def _advance_main_ref(managed: Path, old_sha: str, new_sha: str, env: dict[str, str]) -> None:
    """Compare-and-swap the `main` branch ref from ``old_sha`` to ``new_sha``.

    Called by ``_git_commit`` after a routed commit lands on the managed
    worktree's detached HEAD. The CAS form (`update-ref <ref> <new> <old>`)
    fails loud if `main` moved underneath since the commit's parent was read —
    a non-task.py writer to `main` is the only way that can happen (task.py
    holds the flock across the whole mutation), and clobbering their commit
    silently is exactly the failure mode the resolver exists to prevent.
    """
    _git_quiet(["-C", str(managed), "update-ref", "refs/heads/main", new_sha, old_sha], env)


def repo_root() -> Path:
    """Return the absolute path of the main repo root.

    Resolves via `git rev-parse --git-common-dir` from the directory of
    this module (NOT `os.getcwd()`). Branch-guards: raises a loud,
    distinct `RuntimeError` if the main worktree HEAD is on a non-`main`
    branch or detached. Validates that the resolved path actually contains
    `tasks/` and is not a submodule / bare layout. NEVER falls back to a
    walk-up resolver — silent fallback is what produced the
    worktree-staleness bug class this resolver replaces.

    Process-local LRU cache keyed on `(pid, cwd)` — forks invalidate
    automatically (different pid) and `os.chdir()` invalidates (different
    cwd). One Python invocation pays one `rev-parse` + one `symbolic-ref`
    subprocess pair, total. Call `invalidate_cache()` to force a re-probe
    (used in tests).
    """
    return _resolve_repo_root_cached((os.getpid(), os.getcwd()))


@functools.lru_cache(maxsize=1)
def _primary_checkout_cached(_key: tuple[int, str]) -> Path:
    """Inner cache target for :func:`primary_checkout_root`. Keyed on
    (pid, cwd) exactly like ``_resolve_repo_root_cached`` so forks + chdirs
    invalidate automatically.
    """
    return _resolve_primary_checkout(_sanitized_git_env())


def primary_checkout_root() -> Path:
    """Absolute path of the PRIMARY (main) checkout — the git common dir's
    parent — with full layout validation but NO branch guard and NO off-main
    routing to the managed ``_task-main-pin`` worktree. For consumers that
    need the canonical checkout PATH (session-spawn cwd, #844), not a safe
    tasks/ read-write root (those use :func:`repo_root`). Fails loud; never
    falls back to a ``__file__``/cwd walk-up.
    """
    return _primary_checkout_cached((os.getpid(), os.getcwd()))


def invalidate_cache() -> None:
    """Drop the cached repo-root + primary-checkout resolutions. Next call
    re-probes git."""
    _resolve_repo_root_cached.cache_clear()
    _primary_checkout_cached.cache_clear()


def tasks_dir() -> Path:
    """Return the absolute path of `tasks/` in the main repo."""
    return repo_root() / "tasks"


def registry_path() -> Path:
    """Return the absolute path of `tasks/REGISTRY.json` in the main repo."""
    return tasks_dir() / "REGISTRY.json"


# Compatibility shim: ``LOCK_DIR`` / ``LOCK_PATH`` stay as module-level
# constants because they live under ``~`` and never depend on repo root.
LOCK_DIR = Path.home() / ".task-workflow"
LOCK_PATH = LOCK_DIR / "lock"


# ─── Locking ────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _locked() -> Iterator[None]:
    """Hold an exclusive flock on ~/.task-workflow/lock for the duration of
    a mutation. Multiple processes calling task.py concurrently serialise
    here.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ─── Registry ───────────────────────────────────────────────────────────────


def _load_registry() -> dict[str, Any]:
    rp = registry_path()
    if not rp.exists():
        return {"highest_id": 0, "tasks": {}}
    return json.loads(rp.read_text())


def _save_registry(registry: dict[str, Any]) -> None:
    rp = registry_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    tmp = rp.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    tmp.replace(rp)


def _registry_set(registry: dict[str, Any], task_id: int, path: Path, fm: dict[str, Any]) -> None:
    """Update REGISTRY.json with a task's current path and a tiny summary."""
    rel = str(path.relative_to(repo_root()))
    entry: dict[str, Any] = {
        "path": rel,
        "title": fm.get("title", ""),
        "kind": fm.get("kind", "experiment"),
        "status": _status_from_path(path),
        "has_clean_result": bool(fm.get("has_clean_result", False)),
    }
    goal = fm.get("goal")
    if isinstance(goal, str) and goal.strip():
        entry["goal"] = goal.strip()
    # Paper-stub tasks carry the abstract in the BODY (not the frontmatter), so
    # denormalize it into REGISTRY here for the dashboard hover-card / the
    # REGISTRY title+abstract surfaces. Only paper tasks pay the body read;
    # never raises (a still-being-written stub just yields no abstract).
    if is_paper_task(fm):
        entry["paper"] = True
        body_path = path / "body.md"
        if body_path.exists():
            try:
                _, body = _split_frontmatter(body_path.read_text())
                abstract = extract_stub_abstract(body)
            except (OSError, ValueError):
                abstract = ""
            if abstract:
                entry["abstract"] = abstract
    registry["tasks"][str(task_id)] = entry
    if task_id > registry.get("highest_id", 0):
        registry["highest_id"] = task_id


def _registry_remove(registry: dict[str, Any], task_id: int) -> None:
    registry["tasks"].pop(str(task_id), None)


# ─── Frontmatter ────────────────────────────────────────────────────────────


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown string. Returns (fm, body)."""
    if not text.startswith("---\n"):
        return {}, text
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        # Malformed; treat as bodyless
        return {}, text
    fm_block = rest[:end]
    body = rest[end + len("\n---\n") :]
    try:
        fm = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML frontmatter: {e}") from e
    if not isinstance(fm, dict):
        raise ValueError(f"frontmatter must be a mapping, got {type(fm).__name__}")
    return fm, body


def _join_frontmatter(fm: dict[str, Any], body: str) -> str:
    fm_block = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{fm_block}\n---\n{body}"


class StaleTaskPathError(FileNotFoundError):
    """A task's body.md is missing at a path the registry / caller expects.

    Subclasses ``FileNotFoundError`` so existing ``except FileNotFoundError``
    callers keep catching it (e.g. ``cmd_list_clean_results``,
    ``cmd_migrate_body``, ``task_workflow_migrate``); adds a message naming the
    stale path + the ``task.py audit`` remedy. Raised from ``_read_body`` for
    the #722 split / stale-registry shape (task dir present, body.md gone).
    """


def _read_body(path: Path) -> tuple[dict[str, Any], str]:
    try:
        text = path.read_text()
    except FileNotFoundError as e:
        # Distinguish "task dir exists but body.md missing" (the #722 split /
        # stale-registry shape) from a raw missing path; both name the remedy.
        raise StaleTaskPathError(
            f"body.md not found at {path}. The task dir may be split or the "
            f"registry stale; run `task.py audit` to detect + `task.py audit "
            f"--repair --apply` to repair."
        ) from e
    return _split_frontmatter(text)


def _write_body(path: Path, fm: dict[str, Any], body: str) -> None:
    text = _join_frontmatter(fm, body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _strip_leading_frontmatter_blocks(text: str) -> str:
    """Strip ALL leading ``---\\n...\\n---\\n`` YAML frontmatter blocks from `text`.

    Used by `set_body()` to prevent the duplicate-frontmatter trap:
    callers (notably the analyzer) often pass a complete markdown
    document — frontmatter + body — as the "new body". Without this
    strip, `set_body()` would prepend the canonical frontmatter on top
    of the caller's frontmatter, leaving body.md with TWO ``---...---``
    blocks. The dashboard parses the FIRST block as the header card,
    then renders the SECOND block as literal YAML at the top of the
    visible body — a visible-corruption bug that bit task #389 twice
    (analyzer v5 and v7) in one /issue session on 2026-05-26.

    The strip is idempotent — calling it on an already-stripped string
    returns the same string. Behaviour:

    - Input starts with a valid ``---\\n...\\n---\\n`` block → strip
      that block, then recurse (so multiple stacked blocks are all
      removed).
    - Input starts with ``---\\n`` but has no closing ``\\n---\\n`` →
      treated as malformed; left untouched (matches `_split_frontmatter`
      semantics).
    - Input does NOT start with ``---\\n`` → returned unchanged.
    - After stripping all leading blocks, any leading blank lines are
      removed so the H1 starts at the top of the body region.
    """
    content = text
    while content.startswith("---\n"):
        end = content.find("\n---\n", 4)
        if end == -1:
            # Malformed leading block — leave alone (matches _split_frontmatter).
            break
        content = content[end + len("\n---\n") :]
    return content.lstrip("\n")


# ── Paper-stub helpers (`paper: true` clean-result track) ───────────────────
# A `paper: true` task's body.md is a thin paper-stub (H1 title + abstract +
# a paper link); the canonical clean-result is the LaTeX paper under
# docs/papers/issue_<N>/, verified by scripts/verify_paper.py — NOT this
# module's markdown machinery. These helpers let the readers that denormalize
# title/abstract and the set-clean-result manifest gate handle the stub
# without breaking grandfathered markdown bodies. See
# .claude/skills/clean-results/SPEC.md § "Paper format (`paper: true`)".


def is_paper_task(fm: dict[str, Any]) -> bool:
    """True when the task's frontmatter opts into the paper clean-result track.

    Accepts the YAML-parsed boolean ``True`` and the quoted string ``"true"``
    (case-insensitive); everything else (absent / ``false`` / ``null``) is the
    markdown-body default.
    """
    v = fm.get("paper")
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


#: Body sentinel for the v2 report clean-result form (workflow v2 — the
#: report-only track: Motivation / Methodology / Metrics / Results-as-plots
#: written by agents, TLDR / Next-steps written by Thomas). Placed on the line
#: after the H1 ``# Experiment: ...`` title, mirroring the ``<!-- clean-result-v4 -->``
#: convention. ``scripts/verify_report.py`` is the mechanical verifier for this
#: form. Unlike ``paper: true`` (a frontmatter flag), a report body is
#: identified by this BODY sentinel.
REPORT_V1_SENTINEL = "<!-- report-v1 -->"


def is_report_body(body: str) -> bool:
    """True when ``body`` is a v2 report clean-result (carries ``REPORT_V1_SENTINEL``).

    Detects the v2 report form by its body sentinel, the analogue of
    :func:`is_paper_task` for the report track. Consumers (dashboard rendering,
    promote-time logic, ``scripts/verify_report.py``) branch on this to treat a
    report body as a valid clean-result form alongside the markdown-v4 and paper
    tracks. Does NOT read frontmatter — a report task carries no ``paper``/form
    frontmatter flag, only this sentinel.
    """
    return REPORT_V1_SENTINEL in body


# ── Workflow-fix task helpers (#678 — the file-a-task + spawn-/issue-auto path) ─
# Workflow-surface fixes (a `<!-- workflow-fix-candidate v1 -->` block or a
# surfaced prose follow-up) are filed as a `kind: infra` task and implemented by
# a background `/issue <N> --auto` session — NOT a `workflow-improver` subagent
# spawn (retired by #678). These read-only predicates back the two pieces of
# orchestrator logic the rule defines: the DEDUP check (don't double-file the
# SAME bug on the SAME file) and the RECURSION GUARD (a workflow-fix session
# never auto-files MORE workflow-fix tasks for its own findings). See
# `.claude/rules/workflow-fix-on-bug.md`.

# Non-terminal statuses a workflow-fix task can sit at while it still blocks a
# duplicate re-raise. The terminal set ``{completed, archived}`` is EXCLUDED — a
# closed fix does not block a fresh candidate for the same bug.
_WF_FIX_NONTERMINAL: tuple[str, ...] = (
    "proposed",
    "planning",
    "plan_pending",
    "approved",
    "running",
    "verifying",
    "interpreting",
    "reviewing",
    "awaiting_promotion",
    "followups_running",
    "blocked",
    "on_hold",
)


def _wf_fix_normalize(s: str) -> str:
    """Normalize a candidate prose field for stable fingerprinting.

    Lowercase, collapse internal whitespace to single spaces, strip
    leading/trailing whitespace, and strip trailing ``.,;:!?`` — so a candidate
    re-raised with reformatted prose (extra spaces, a trailing period, a case
    change) produces the SAME fingerprint and is correctly deduped.
    """
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s.rstrip(".,;:!?").strip()


def wf_fix_fingerprint(proposed_change: str, bug_observed: str) -> str:
    """Stable 12-hex dedup fingerprint of a workflow-fix candidate.

    The dedup GRAIN is ``(target_file, fingerprint)`` (#678 A1): two DISTINCT
    bugs on the SAME hot file (SKILL.md, CLAUDE.md, …) produce DIFFERENT
    fingerprints and therefore file two distinct tasks (each gets its own plan
    review); only a genuine re-raise of the SAME bug (same normalized
    ``proposed_change`` + ``bug_observed``) collapses to the same fingerprint
    and is deduped. The fingerprint is recorded as a ``wf-fix-fp:<fp>`` tag and
    in the body ``## Provenance`` block at file-time.
    """
    h = hashlib.sha256(
        (_wf_fix_normalize(proposed_change) + "||" + _wf_fix_normalize(bug_observed)).encode()
    )
    return h.hexdigest()[:12]


def is_open_workflow_fix_task(target_file: str, fingerprint: str | None = None) -> int | None:
    """Return the id of an OPEN ``kind: infra`` workflow-fix task matching this key, else None.

    Dedup key (#678 A1): ``(target_file, fingerprint)``. A task matches iff
    ``kind == infra`` AND its status is NOT in ``{completed, archived}`` AND its
    title starts with ``workflow-fix:`` AND its body ``## Provenance`` carries a
    ``workflow_fix_target: <target_file>`` line (exact string match). When
    ``fingerprint`` is given, the task must ALSO carry a ``wf-fix-fp:<fingerprint>``
    tag (or a ``fingerprint: <fingerprint>`` Provenance line) — so a DIFFERENT
    bug on the same file (different fingerprint) is NOT a duplicate. When
    ``fingerprint`` is None, matches any open workflow-fix task on the file
    (coarse, file-only — used only by callers with no candidate fingerprint).

    Read-only: no mutation, no commit. The cheap pre-filter (``kind`` / status /
    title) reads the REGISTRY snapshot; the ``tags`` / Provenance check reads the
    task's ``body.md`` (tags are not denormalized into the registry).
    """
    reg = _load_registry()
    for tid_str, entry in reg.get("tasks", {}).items():
        if entry.get("kind") != "infra":
            continue
        if (entry.get("status") or "") not in _WF_FIX_NONTERMINAL:
            continue
        if not str(entry.get("title", "")).startswith("workflow-fix:"):
            continue
        tid = int(tid_str)
        try:
            body_path = find_task_path(tid) / "body.md"
            fm, body = _read_body(body_path)
        except (FileNotFoundError, ValueError):
            continue
        if f"workflow_fix_target: {target_file}" not in body:
            continue
        if fingerprint is not None:
            tags = [str(t) for t in (fm.get("tags") or [])]
            if f"wf-fix-fp:{fingerprint}" not in tags and f"fingerprint: {fingerprint}" not in body:
                # Same file, different bug -> not a duplicate.
                continue
        return tid
    return None


def is_workflow_fix_session(task_id: int) -> bool:
    """True iff this task is a workflow-fix task (recursion-guard durable signal, #678 Q4).

    The DURABLE signal: the body ``## Provenance`` carries a
    ``workflow_fix_target:`` line. It survives a watcher crash-recovery respawn
    (which re-runs ``spawn-issue --auto`` WITHOUT custom env, so the
    ``EPM_WORKFLOW_FIX_SESSION`` env var is lost on respawn). The env var is the
    in-session convenience leg, checked by the caller, not here. A workflow-fix
    session NEVER auto-files MORE workflow-fix tasks for its own findings — it
    logs + notifies, analogue of ``AUTO_REVIEW_DISABLED``.

    Read-only: no mutation, no commit.
    """
    try:
        body = (find_task_path(task_id) / "body.md").read_text()
    except (FileNotFoundError, OSError):
        return False
    return "workflow_fix_target:" in body


# ---------------------------------------------------------------------------
# Failure-lesson capture + supersedes retraction (#712).
#
# Three PURE, side-effect-free helpers the SKILL.md orchestrator prose calls
# when it receives an ``epm:failure-lesson`` block (the consumer is prose, not
# a script — these are the FIRST code that touches the lesson data, which is
# precisely what makes the body acceptance criteria byte-testable). No I/O, no
# writes: the orchestrator owns the marker post + the explicit-path commit of
# the returned ``{path: text}`` map. Siblings of the ``wf_fix_*`` family above.
# See `.claude/skills/issue/SKILL.md` § "Failure-lesson capture" and
# `.claude/rules/workflow-fix-on-bug.md`.
# ---------------------------------------------------------------------------


def failure_lesson_capture_eligible(
    block_fields: dict[str, str],
    *,
    subsequent_distinct_failure: bool,
) -> bool:
    """Decide whether a received ``epm:failure-lesson`` block is eligible for capture.

    Eligible when the block RESOLVED the failure (the original trigger,
    signalled by ``block_fields["resolved"] == "yes"``) OR the block carries
    ``root_cause_confirmed: yes`` — the latter is True INDEPENDENT of
    ``subsequent_distinct_failure`` (case ii, #712: the cause was confirmed even
    though a distinct failure followed or the pod was abandoned in recovery — the
    #664 L204 gap, where a confirmed pod-hardware cause produced NO failure-lesson
    because the resolve-only trigger never fired). ``subsequent_distinct_failure``
    is accepted to make that "captured-regardless-of-a-following-failure" guarantee
    explicit and testable, and is deliberately NOT consulted when
    ``root_cause_confirmed=yes``.

    Pure: no I/O. The orchestrator owns the actual marker post + durable write.
    """
    if block_fields.get("resolved", "").strip().lower() == "yes":
        return True
    return block_fields.get("root_cause_confirmed", "").strip().lower() == "yes"


# ─── /issue Step 7 crash-fix circuit-breaker (pure predicate) ────────────────
#
# Detects two execution-side traps the cap-3 routing table cannot see and that
# both mean "relaunching is futile; the PLAN, not the code, must change" — the
# canonical pivot is `workflow.yaml § pivot_criteria.plan_contradiction_replan`.
# Pure (no I/O): the orchestrator owns the marker post + set-status + re-plan.

# The marker kinds that COUNT as a successful round and reset the trigger-1
# counter. `epm:progress` is deliberately EXCLUDED — it is the workflow's
# catch-all heartbeat / phase-tick / watcher-respawn breadcrumb, posted DURING a
# still-failing trap window (verified on #664: six benign epm:progress markers
# fell between the same-signature failures), so resetting on it would make the
# trigger inert (#718 MF#3).
_CB_RESET_MILESTONES: frozenset[str] = frozenset({"epm:experiment-implementation", "epm:results"})

# A CalledProcessError-style crash note whose subprocess argv array names a
# script — the exact #664 dispatch-crash shape. The `script` group anchors on a
# `.py` token inside the `Command '[...]'` argv; the caller strips the
# interpreter (`python`/`python3`) so it lands on the real entrypoint.
_CB_CALLEDPROC_RE = re.compile(
    r"(?P<exc>\w*(?:Error|Exception)): Command '\[(?P<argv>.*?)\]'",
    re.DOTALL,
)
# A bare exception type on the note's first line (non-subprocess crash shape).
_CB_EXC_RE = re.compile(r"\b(?P<exc>\w*(?:Error|Exception))\b")
# Explicit / bracketed assert-tag tokens (the two most-stable structured rungs).
_CB_ASSERT_TAG_RE = re.compile(r"assert_tag:\s*(\S+)")
_CB_BRACKET_TAG_RE = re.compile(r"\[([\w-]+)-assert\]")
# Volatile spans stripped before hashing a note (so per-round argv / pids /
# timestamps / file:line do not split one trap into N distinct signatures).
_CB_ISO_TS_RE = re.compile(r"\d{4}-\d\d-\d\dT[\d:Z.+-]+")
_CB_PID_RE = re.compile(r"(?:pid[= ]\d+|\bPID \d+)")
_CB_ARGV_RE = re.compile(r"Command '\[.*?\]'", re.DOTALL)
_CB_FLAG_RE = re.compile(r"--[\w-]+(?:\s+\S+)?")
_CB_FILELINE_RE = re.compile(r"[\w./-]+:\d+")


def _cb_note_hash(note: str) -> str:
    """Stable 12-hex digest of a crash note's first non-blank line.

    Strips volatile spans (ISO timestamps, PIDs, the whole subprocess argv array,
    remaining ``--flag value`` runs, ``file:line`` spans) IN THAT ORDER so two
    crash notes that differ ONLY in those volatile parts hash to the SAME tag.
    The argv-array strip is the secondary defense behind the exception-type rung:
    even a note that does not match ``_CB_CALLEDPROC_RE`` but carries argv
    variation still collapses to ONE digest across rounds (#718 MF#1).
    """
    first = ""
    for line in note.splitlines():
        if line.strip():
            first = line
            break
    normalized = _CB_ISO_TS_RE.sub("", first)
    normalized = _CB_PID_RE.sub("", normalized)
    normalized = _CB_ARGV_RE.sub("Command '[...]'", normalized)
    normalized = _CB_FLAG_RE.sub("", normalized)
    normalized = _CB_FILELINE_RE.sub("", normalized)
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _cb_calledproc_script(argv: str) -> str | None:
    """Return the entrypoint ``.py`` basename from a subprocess argv string.

    Picks the LAST ``.py`` token that is not a ``python``/``python3``
    interpreter basename (the real script the dispatcher invoked); ``None`` when
    the argv names no script.
    """
    candidates = re.findall(r"[\w./-]+\.py", argv)
    scripts = [c.split("/")[-1] for c in candidates if c.split("/")[-1] not in ("python.py",)]
    # No interpreter ends in `.py` on this stack, so every match is a script;
    # take the LAST one (the entrypoint, after any wrapper modules).
    return scripts[-1] if scripts else None


def _cb_failure_signature(note: str, prior_class: str | None) -> tuple[str, str, str]:
    """Extract the ``(phase, failure_class, assert_tag)`` signature of a failure note.

    ``assert_tag`` falls back through an ORDERED chain — the FIRST rung that
    matches wins (#718 MF#1):

    1. explicit ``assert_tag: <tag>`` SHOULD field,
    2. bracketed ``[<tag>-assert]`` token,
    3. exception-type / command-family ``<ExcName>:<script_basename>`` (the
       #664 un-tagged subprocess-crash shape) or, for a non-subprocess crash,
       the bare ``<ExcName>`` from the note's first line,
    4. a normalized note-hash (volatile spans stripped) as the last resort.
    """
    phase_m = re.search(r"phase=(\w+)", note)
    phase = phase_m.group(1) if phase_m else "?"

    class_m = re.search(r"failure_class:\s*(\w+)", note)
    if class_m:
        failure_class = class_m.group(1)
    elif prior_class:
        failure_class = prior_class
    else:
        failure_class = "?"

    assert_tag = _cb_assert_tag(note)
    return (phase, failure_class, assert_tag)


def _cb_assert_tag(note: str) -> str:
    """Resolve the ``assert_tag`` rung for a failure note (see _cb_failure_signature)."""
    # Rung 1 — explicit SHOULD field.
    m = _CB_ASSERT_TAG_RE.search(note)
    if m:
        return m.group(1)
    # Rung 2 — bracketed [<tag>-assert].
    m = _CB_BRACKET_TAG_RE.search(note)
    if m:
        return m.group(1)
    # Rung 3 — exception-type / command-family.
    m = _CB_CALLEDPROC_RE.search(note)
    if m:
        script = _cb_calledproc_script(m.group("argv"))
        if script:
            return f"{m.group('exc')}:{script}"
        return m.group("exc")
    first_line = note.splitlines()[0] if note.splitlines() else ""
    m = _CB_EXC_RE.search(first_line)
    if m:
        return m.group("exc")
    # Rung 4 — note-hash.
    return _cb_note_hash(note)


def _cb_parse_ladder(plan_text: str) -> list[str]:
    """Return the ordered option labels of the first finite escape ladder in ``plan_text``.

    Two shapes are recognized: a ``Option A -> Option B -> ...`` (literal `` → ``
    arrow) run, OR a numbered ``Option <N>:`` list. Returns ``[]`` when the plan
    enumerates no parseable ladder (the predicate then silently no-ops trigger 2).
    """
    # Shape 1 — arrow-separated Option run, e.g. "Option A → Option B → Option C".
    arrow_run = re.search(
        r"Option\s+(\w+)(?:\s*→\s*Option\s+(\w+))+",
        plan_text,
    )
    if arrow_run:
        # Re-scan the matched span for every `Option <label>` token in order.
        span = arrow_run.group(0)
        return re.findall(r"Option\s+(\w+)", span)
    # Shape 2 — a numbered `Option <N>:` list (≥2 entries).
    numbered = re.findall(r"Option\s+(\w+):", plan_text)
    if len(numbered) >= 2:
        # Preserve first-seen order, dedup repeats.
        seen: list[str] = []
        for label in numbered:
            if label not in seen:
                seen.append(label)
        return seen
    return []


def circuit_breaker_should_fire(
    events: list[dict[str, Any]],
    plan_text: str,
    K: int = 4,
) -> dict[str, Any] | None:
    """Decide whether the /issue Step 7 crash-fix circuit-breaker fires.

    Fires (returns a fire-reason dict) on EITHER:

    * Trigger 1 — ``same_failure_class``: ``K`` or more ``epm:failure`` markers,
      since the last "resolved" milestone marker, share ONE
      ``(phase, failure_class, assert_tag)`` signature. The counter RESETS at any
      intervening ``epm:experiment-implementation`` / ``epm:results`` milestone
      (a successful round escaped the trap). It does NOT reset on ``epm:progress``
      (the catch-all heartbeat / phase-tick / watcher-respawn breadcrumb, posted
      DURING a still-failing trap window — #718 MF#3).
    * Trigger 2 — ``enumerated_fallback_exhausted``: ``plan_text`` enumerates a
      finite escape ladder (``Option A -> Option B -> ...`` arrow run or a
      numbered ``Option N:`` list), EVERY option has been LAUNCHED (named in an
      ``epm:progress`` / ``epm:experiment-implementation`` note), AND the gate
      RE-TRIPS *after* the ladder is exhausted — an ``epm:failure`` event whose
      position in ``events`` is LATER than the last launch of the FINAL ladder
      option. A stale ``epm:failure`` that PRECEDES the ladder launches does NOT
      count (#718). Silently no-ops on free-form plans with no parseable ladder.

    Returns ``None`` when neither condition holds. On fire the dict shape is::

        {"trigger": "same_failure_class" | "enumerated_fallback_exhausted",
         "signature": (phase, failure_class, assert_tag),   # trigger 1 only
         "count": int,                                       # trigger 1 only
         "ladder": [str, ...],                               # trigger 2 only
         "gate": str,                                        # trigger 2 only
         "pivot_scope": str}

    ``pivot_scope`` is ALWAYS present and non-empty — the ready-to-pass
    ``/adversarial-planner`` scope string built from the matched evidence (the
    orchestrator passes it verbatim; an empty/generic scope would reproduce the
    #488 unscoped re-plan). Trigger 1 is checked first; if it fires, trigger 2 is
    not evaluated.

    Pure: no I/O. The orchestrator owns the marker post + set-status + re-plan.
    """
    fire = _cb_trigger_same_failure_class(events, K)
    if fire is not None:
        return fire
    return _cb_trigger_enumerated_fallback(events, plan_text)


def _cb_trigger_same_failure_class(events: list[dict[str, Any]], K: int) -> dict[str, Any] | None:
    """Trigger 1: K+ same-(phase, failure_class, assert_tag) failures since the last milestone."""
    # Count consecutive same-signature failures since the most recent resetting
    # milestone. A milestone clears the running tally for ALL signatures.
    tally: dict[tuple[str, str, str], int] = {}
    prior_class: str | None = None
    fired: dict[str, Any] | None = None
    for e in events:
        kind = e.get("kind", "")
        if kind in _CB_RESET_MILESTONES:
            tally.clear()
            continue
        if kind != "epm:failure":
            # epm:progress and every other non-milestone marker is inert here.
            continue
        note = e.get("note", "") or ""
        sig = _cb_failure_signature(note, prior_class)
        # Carry the most recent KNOWN class forward for un-classed dispatch crashes.
        if sig[1] != "?":
            prior_class = sig[1]
        tally[sig] = tally.get(sig, 0) + 1
        if tally[sig] >= K:
            count = tally[sig]
            phase, failure_class, assert_tag = sig
            pivot_scope = (
                f"Same-failure-class repetition: phase={phase} "
                f"class={failure_class} assert_tag={assert_tag} count={count} "
                f"(K={K}). Re-plan the recipe that produces this trap, or "
                f"drop the gate."
            )
            # Keep scanning to report the FINAL count (the trap may re-trip more),
            # but lock in the first signature that crossed K.
            fired = {
                "trigger": "same_failure_class",
                "signature": sig,
                "count": count,
                "pivot_scope": pivot_scope,
            }
    return fired


def _cb_trigger_enumerated_fallback(
    events: list[dict[str, Any]], plan_text: str
) -> dict[str, Any] | None:
    """Trigger 2: every option of a plan-enumerated escape ladder launched + gate re-trips."""
    ladder = _cb_parse_ladder(plan_text)
    if not ladder:
        return None
    # Index every launch / progress event that names an option, keyed by option
    # label. (events is chronological — append-only events.jsonl order — so the
    # list index IS the ordering signal.)
    last_launch_idx: dict[str, int] = {}
    for idx, e in enumerate(events):
        if e.get("kind", "") not in ("epm:progress", "epm:experiment-implementation"):
            continue
        note = e.get("note", "") or ""
        for option in ladder:
            if re.search(rf"Option\s+{re.escape(option)}\b", note):
                last_launch_idx[option] = idx
    # Every named option must have been LAUNCHED (named in a launch / progress note).
    if any(option not in last_launch_idx for option in ladder):
        return None
    # The ladder is exhausted only once the FINAL option has been launched. Require
    # an epm:failure AFTER that last-launch index — a POST-exhaustion re-trip of the
    # gate, not just any failure anywhere in history (a pre-ladder stale failure
    # must NOT count, #718 Codex critic false-positive).
    final_option_launch_idx = last_launch_idx[ladder[-1]]
    if not any(
        e.get("kind", "") == "epm:failure" and idx > final_option_launch_idx
        for idx, e in enumerate(events)
    ):
        return None
    gate = _cb_gate_label(plan_text, ladder)
    pivot_scope = (
        f"Enumerated escape ladder exhausted: gate={gate} "
        f"ladder={' → '.join(ladder)}. Every named alternative was launched; "
        f"the gate re-tripped each time. Redesign the gate or the ladder so a "
        f"viable option exists, or drop the gate."
    )
    return {
        "trigger": "enumerated_fallback_exhausted",
        "ladder": ladder,
        "gate": gate,
        "pivot_scope": pivot_scope,
    }


def _cb_gate_label(plan_text: str, ladder: list[str]) -> str:
    """Best-effort label of the gate the escape ladder defends.

    Returns the nearest preceding ``§<N>``-style or ``Gate <N>`` token before the
    first ``Option <first-label>`` mention, else ``"<unnamed-gate>"``.
    """
    first = ladder[0]
    idx = plan_text.find(f"Option {first}")
    if idx < 0:
        return "<unnamed-gate>"
    preceding = plan_text[:idx]
    gate_m = None
    for m in re.finditer(r"(?:§\s*[\w.-]+|Gate\s+\w+)", preceding):
        gate_m = m
    return gate_m.group(0).strip() if gate_m else "<unnamed-gate>"


STAGE_RESULT_KINDS: dict[str, frozenset[str]] = {
    # NORMALIZED stage token (see _normalize_stage) -> marker kind(s) that complete it.
    # Clearing = "ANY completion marker a subagent of this stage posts" — the set answers
    # "did the last dispatched subagent finish?", NOT "did the whole stage finish?"
    # (one stage+round legitimately dispatches analyzer -> critic -> reconciler in sequence;
    # #547 replay: an intermediate epm:interpretation must clear the analyzer crumb so the
    # critic dispatch is not wrongly skipped).
    "verifying": frozenset({"epm:upload-verification"}),
    "interpreting": frozenset(
        {
            "epm:interpretation",
            "epm:interp-critique",
            "epm:interp-critique-codex",
            "epm:review-reconcile",
        }
    ),
    "clean-result": frozenset(
        {
            "epm:clean-result-critique",
            "epm:interpretation",
            "epm:clean-result-critique-codex",
            "epm:review-reconcile",
        }
    ),
    "implementing": frozenset({"epm:experiment-implementation"}),
    "free-analysis-followup": frozenset({"epm:free-analysis-followup-run"}),
    "methodology-reference": frozenset({"epm:methodology-doc-generated"}),
    "code-review": frozenset({"epm:code-review", "epm:code-review-codex", "epm:review-reconcile"}),
    "planning": frozenset({"epm:plan"}),
    "related-work": frozenset({"epm:related-work-proposed"}),
}
# epm:failure clears the in-flight state of ANY stage.
_ALWAYS_RESULT_KINDS = frozenset({"epm:failure"})
# Markers a healthy in-flight round keeps emitting; each refreshes the freshness clock.
STAGE_LIVENESS_KINDS = frozenset(
    {
        "epm:codex-task-spawned",
        "epm:codex-task-completed",
        "epm:smoke-architecture-check",
        "epm:proposed-tests",
    }
)
# Progress-note substrings that are ANTI-liveness for a stage-dispatch
# freshness window (#949; incident #810): bracketed telemetry posted by the
# session watcher / spawn machinery, never by the stage's own worker. The
# watcher's own progress clock excludes the same classes
# (scripts/autonomous_session_watch.py::_WATCHER_NOTE_SENTINELS — a script,
# not importable from src/, so matched here by the shared bracketed
# prefixes: every watcher sentinel embeds "[autonomous_session_watch:" and
# spawn_session's bookkeeping sentinel embeds "[spawn-session:"). The
# self-stamped "[long-phase-heartbeat]" prefix is DELIBERATELY absent — it
# is posted by the stage's own long-running phase and IS liveness. The
# sibling deliberate-stop exclusion (checked inline in
# stage_dispatch_should_skip) also drops ANY note with
# by == "spawn_session-stop" regardless of content — a future genuine
# liveness post from spawn_session must use a different `by` or get a
# carve-out here.
STAGE_ANTILIVENESS_NOTE_SUBSTRINGS = frozenset(
    {
        "[autonomous_session_watch:",
        "[spawn-session:",
    }
)

_STAGE_ALIASES = {"code-reviewing": "code-review"}


def _normalize_stage(stage: str) -> str:
    """Strip ONE leading ``followup-`` prefix, then apply ``_STAGE_ALIASES``.

    Used only for result-kind clearing lookups; the dedup MATCH compares raw tokens.
    Unknown tokens pass through unchanged.
    """
    stripped = stage.removeprefix("followup-")
    return _STAGE_ALIASES.get(stripped, stripped)


def _stage_event_ts(event: dict) -> datetime | None:
    """Parse an event's ISO-8601 ``ts`` (naive treated as UTC); None on malformed/missing."""
    raw = event.get("ts", "")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _breadcrumb_fields(note: str) -> dict[str, str]:
    """Parse a ``stage-dispatch`` note's ``key=value`` tokens (whitespace-split, order-free)."""
    fields: dict[str, str] = {}
    for token in note.split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    return fields


def _find_stage_breadcrumb(events: list[dict], stage: str, round_num: int) -> int | None:
    """Index of the most recent breadcrumb matching the RAW ``stage=`` token + ``round=``.

    A breadcrumb with no ``round=`` token, or a non-integer round value, never matches.
    Returns None when no breadcrumb matches.
    """
    for idx in range(len(events) - 1, -1, -1):
        event = events[idx]
        if event.get("kind", "") != "epm:progress":
            continue
        note = (event.get("note", "") or "").lstrip()
        if not note.startswith("stage-dispatch "):
            continue
        fields = _breadcrumb_fields(note)
        if fields.get("stage") != stage:
            continue
        try:
            if int(fields["round"]) == round_num:
                return idx
        except (KeyError, ValueError):
            continue
    return None


def stage_dispatch_should_skip(
    events: list[dict],
    stage: str,
    round_num: int,
    window_minutes: float,
    *,
    now: datetime | None = None,
) -> str | None:
    """Return a one-line skip reason when a same-stage+round dispatch is in flight, else None.

    A breadcrumb is an ``epm:progress`` event whose lstripped note begins with
    ``"stage-dispatch "``; its ``key=value`` fields parse order-independently, and a
    breadcrumb with no integer ``round=`` never matches a round query. The most recent
    breadcrumb matching the RAW ``stage=`` token and ``round=`` is in flight unless a
    LATER event carries a stage-matching result kind
    (``STAGE_RESULT_KINDS[_normalize_stage(stage)]`` — clearing is round-agnostic by
    design, result markers carry no parsable round) or ``epm:failure``. While in
    flight, the freshness clock starts at the LATEST of the breadcrumb and any later
    liveness marker (``STAGE_LIVENESS_KINDS`` or a non-breadcrumb ``epm:progress`` —
    EXCLUDING anti-liveness notes: a ``deliberate-stop`` record (``by ==
    "spawn_session-stop"``) and bracketed watcher / spawn-session telemetry
    (``STAGE_ANTILIVENESS_NOTE_SUBSTRINGS``) are stop/bookkeeping records, not
    stage liveness, and never refresh a window (#810); a breadcrumb never
    refreshes any window): effective age < window -> skip reason;
    >= window -> None (stalled, re-dispatch allowed). A malformed breadcrumb ``ts``
    fails toward dispatch (None); a malformed liveness ``ts`` is ignored. TOCTOU is a
    non-goal — two orchestrators both checking BEFORE either posts its breadcrumb can
    still double-dispatch; the Step-0 single-orchestrator guard + implementer
    self-detection are the backstops. Add new stage tokens to ``STAGE_RESULT_KINDS``.
    """
    if now is None:
        now = datetime.now(UTC)
    crumb_idx = _find_stage_breadcrumb(events, stage, round_num)
    if crumb_idx is None:
        return None
    clearing = STAGE_RESULT_KINDS.get(_normalize_stage(stage), frozenset()) | _ALWAYS_RESULT_KINDS
    later = events[crumb_idx + 1 :]
    if any(event.get("kind", "") in clearing for event in later):
        return None
    crumb_ts = _stage_event_ts(events[crumb_idx])
    if crumb_ts is None:
        return None
    effective_start = crumb_ts
    refresher: tuple[str, str] | None = None
    for event in later:
        kind = event.get("kind", "")
        if kind == "epm:progress":
            note = (event.get("note", "") or "").lstrip()
            if note.startswith("stage-dispatch "):
                continue
            # Anti-liveness (#810/#949): a deliberate session stop is the
            # death record of the stage's owner, and bracketed watcher /
            # spawn-session telemetry is third-party bookkeeping — neither
            # is evidence the stage's OWN work is alive, so neither
            # refreshes the window. (They do NOT clear the in-flight
            # state; only result kinds / epm:failure / expiry do that.)
            if note.startswith("deliberate-stop ") or event.get("by") == "spawn_session-stop":
                continue
            if any(s in note for s in STAGE_ANTILIVENESS_NOTE_SUBSTRINGS):
                continue
        elif kind not in STAGE_LIVENESS_KINDS:
            continue
        ts = _stage_event_ts(event)
        if ts is not None and ts > effective_start:
            effective_start = ts
            refresher = (kind, event.get("ts", ""))
    age_minutes = (now - effective_start).total_seconds() / 60.0
    if age_minutes >= window_minutes:
        return None
    refreshed = f", refreshed by {refresher[0]} at {refresher[1]}" if refresher else ""
    return (
        f"skip: stage-dispatch stage={stage} round={round_num} in flight — "
        f"breadcrumb at {events[crumb_idx].get('ts', '')}, effective age {age_minutes:.1f}m "
        f"< window {window_minutes}m{refreshed}"
    )


# ─── Pre-dispatch external-marker triage (#889) ─────────────────────────────

# Machine-posted / lifecycle-bookkeeping kinds that never carry cross-session
# advisory content — excluded from pre-dispatch triage candidates. Anything
# NOT listed is a candidate (over-approximation by design: a false positive
# costs one first-line read; a false negative is the #779 failure mode).
TRIAGE_EXEMPT_KINDS = frozenset(
    {
        "epm:status-changed",
        "epm:step-completed",
        "epm:backend-selected",
        "epm:codex-task-spawned",
        "epm:codex-task-completed",
        "epm:codex-task-failed",
        "epm:pod-provisioned",
        "epm:pod-terminated",
        "epm:pod-stopped",
        "epm:run-launched",
        "epm:run-finished",
        "epm:upload-verification",
        "epm:merged",
        "epm:methodology-doc-generated",
        "epm:workflow-fix-task-filed",
        "epm:workflow-fix-applied",
        "epm:workflow-fix-failed",
        "epm:workflow-fix-candidate",
        # Session-pipeline review/lifecycle verdict kinds — structurally posted
        # by THIS task's own planner/review/implementation loop, never the
        # vehicle for a cross-session advisory (fact-check on #779: including
        # these halves the per-dispatch read load, 30 -> 20 candidates, with
        # zero externals lost).
        "epm:code-review",
        "epm:code-review-codex",
        "epm:review-reconcile",
        "epm:experiment-implementation",
        "epm:concern-raised",
        "epm:concern-addressed",
        "epm:concern-deferred",
        "epm:interp-critique",
        "epm:interp-critique-codex",
        "epm:clean-result-critique",
        "epm:clean-result-critique-codex",
        "epm:plan",
        "epm:plan-approved",
        "epm:plan-verify",
        "epm:consistency",
        "epm:clarify",
        "epm:clarify-answers",
        "epm:test-verdict",
        "epm:smoke-architecture-check",
    }
)

# ``by`` values identifying MACHINE posters (pollers, routers, CLI shims).
# NOTE: ``by`` is UNRELIABLE for humans/sessions — post_event defaults it to
# "unknown", and on #779 both self- and PM-chat-posted markers carried
# by="unknown" — so this set only strips known machine identities; it never
# claims to identify externality (that is the orchestrator's judgment call,
# SKILL.md § Pre-dispatch external-marker triage).
TRIAGE_MACHINE_BY = frozenset(
    {
        "poll_pipeline",
        "task.py",
        "backends.router",
        "backends.gcp",
        "backends.slurm",
        "backends.slurm_monitor",
        "backends.selector",
        "autonomous-gate",
        "codex_task",
        "task_state shim",
    }
)

# Compute-launch marker kinds — ALWAYS close the triage window.
# epm:run-launched is the RunPod/experimenter launch record;
# epm:cluster-launched is what the default GCP/SLURM lanes post (SKILL.md
# Step 6b marker trail; #779's own window contains one at 14:56:21Z, by
# backends.gcp).
TRIAGE_LAUNCH_KINDS = frozenset({"epm:run-launched", "epm:cluster-launched"})

# The auditable triage-record line. A dispatch record carrying this line is
# DUTY-BOUND (it performed the triage) and closes the window; a note that IS
# a triage record is also excluded from candidates.
TRIAGE_LINE_PREFIX = "external-markers triaged:"


def triage_candidates_since_last_dispatch(
    events: list[dict],
    *,
    exempt_kinds: frozenset[str] = TRIAGE_EXEMPT_KINDS,
    machine_by: frozenset[str] = TRIAGE_MACHINE_BY,
    launch_kinds: frozenset[str] = TRIAGE_LAUNCH_KINDS,
) -> list[dict]:
    """Return pre-dispatch triage candidates since the latest DUTY-BOUND dispatch record.

    THE BOUNDARY MATCHES THE TRIAGE DUTY SURFACE: the window opens AFTER the
    most recent event that is either (i) a compute-launch marker (kind in
    ``launch_kinds`` — ``epm:run-launched`` or ``epm:cluster-launched``), or
    (ii) ANY event whose note contains the ``external-markers triaged:`` line
    (a triaged compute breadcrumb, or the adjacent ``epm:progress`` triage
    note the pod/backend-launch form posts). When no such record exists the
    window is the whole list (task start). A NON-compute breadcrumb (review /
    analyzer / verifier stages) never closes the window — those dispatches
    carry no triage duty, so they cannot orphan an advisory; an UNTRIAGED
    compute breadcrumb (pre-fix or concurrent session) also does not close it
    (fail-toward-triage). Within the window an event is a candidate unless:
    kind in ``exempt_kinds``, ``by`` in ``machine_by``, the note is
    empty/absent, the note is itself breadcrumb-shaped (lstripped note begins
    ``"stage-dispatch "`` — same detection as ``stage_dispatch_should_skip``),
    or the note contains the triage line (it is a triage record, not an
    advisory). Chronological order preserved. Deliberately over-approximates —
    it ENUMERATES for LLM-side triage and never classifies externality
    (``by`` cannot: default "unknown").
    """
    boundary = -1
    for idx in range(len(events) - 1, -1, -1):
        event = events[idx]
        note = event.get("note", "") or ""
        if event.get("kind", "") in launch_kinds or TRIAGE_LINE_PREFIX in note:
            boundary = idx
            break
    candidates: list[dict] = []
    for event in events[boundary + 1 :]:
        if event.get("kind", "") in exempt_kinds:
            continue
        if event.get("by", "") in machine_by:
            continue
        note = event.get("note", "") or ""
        if not note.strip():
            continue
        if note.lstrip().startswith("stage-dispatch "):
            continue
        if TRIAGE_LINE_PREFIX in note:
            continue
        candidates.append(event)
    return candidates


# ─── Same-issue follow-up label grouping (#894) ────────────────────────────
#
# Distinct queued follow-ups share the marker KIND (`epm:followup-scope`)
# under different `followup_label`s, so the highest-version-per-kind marker
# map is the WRONG read for dispatch: a later label's completion must never
# strand an earlier queued label (#763: scope v1 `neutral-contrast-and-cofit`
# stayed invisible after scope v2's round ran). These helpers are the SINGLE
# implementation of the label-grouped predicate; `/issue` Step 0, the Step 9b
# loop, the resume table, and `scripts/autonomous_session_watch.py` all defer
# here.

FOLLOWUP_SCOPE_KIND = "epm:followup-scope"
FOLLOWUP_RUN_KIND = "epm:same-issue-followup-run"
USER_INITIATED_FOLLOWUP_SOURCES = frozenset({"user-chat", "step-10b-pick"})

# An UNLABELED scope note inherits the previous entry's label ONLY when it
# carries an explicit correction signal (the #658-v2 shape: "CORRECTION to
# the earlier epm:followup-scope (...)"). An unlabeled note WITHOUT the
# signal is a DISTINCT queued follow-up (#685 v2) and must NOT merge into
# the previous label.
CORRECTION_SIGNAL = re.compile(r"correction|supersede|re-?post", re.IGNORECASE)

# The same-issue loop's stage-dispatch breadcrumb prefix (mirrors
# `autonomous_session_watch._FOLLOWUP_STAGE_DISPATCH_WITNESS_PREFIX`; the
# watcher script cannot be imported from the library, so the literal is
# duplicated here — both sides cite each other).
_FOLLOWUP_STAGE_DISPATCH_PREFIX = "stage-dispatch stage=followup-"

# Round-completion words for the class-3 retro-close evidence read
# (`followup_retro_close_evidence`). Case-sensitive by design — a missed
# close is safe (the label stays queued), a wrong close is not.
_ROUND_COMPLETION_WORD = re.compile(r"PASS|re-park|awaiting_promotion|clean-result")


def parse_followup_note_field(note: str, field: str) -> str | None:
    """Extract ``<field>``'s value from a followup-scope / run-marker note.

    Matches the FIRST line whose core (after stripping any leading mix of
    ``-``/``*`` bullets, bold markers, and whitespace) starts with
    ``<field>:`` OR ``<field>=`` — both separators occur in the wild (the
    ``=`` form is the dominant historical run-marker shape, e.g. #537/#552).
    The value is the first whitespace token of the remainder, stripped of
    backticks / quotes / ``*`` and a trailing comma (#664 ships a
    backtick-wrapped bold value). Handles bare-colon, bare-equals,
    dash-bullet (#658 v1), star-bullet, bold ``**field:**`` (#837 §4c), the
    COMBINED bullet+bold ``- **field:** x`` (a dash-bullet wrapping a bold
    field — corpus-clean today, pinned against future drift), and
    single-line space-separated run notes (first-token rule; labels are
    kebab-slugs with no whitespace — workflow.yaml § markers). First hit
    wins: #763 v2 embeds a second bold label deep inside its
    verbatim-proposal section, and the top-of-note canonical line is hit
    first. Returns ``None`` when the field is absent or its value is empty.
    """
    for line in (note or "").splitlines():
        # One regex pass strips any interleaved mix of whitespace, bullet
        # dashes/stars, and bold markers (a sequential strip()/lstrip("-*")
        # chain stops at the space in "- **field:** x" and misses the bold
        # marker behind it).
        core = re.sub(r"^[\s\-*]+", "", line)
        if core.startswith(f"{field}:") or core.startswith(f"{field}="):
            rest = core[len(field) + 1 :].lstrip("*").strip()
            tokens = rest.split()
            value = tokens[0] if tokens else ""
            value = value.strip("`'\"*").rstrip(",")
            return value or None
    return None


def _scope_scan_key(event: dict) -> tuple[datetime, int]:
    """Chronological scan key ``(ts, version)`` for followup-scope grouping.

    CHRONOLOGICAL with version tiebreak — NOT ``(version, ts)``: per-kind
    version monotonicity is VIOLATED in the wild (#480 carries TWO
    ``version: 1`` scope rows with a v2 chronologically between them);
    ``(ts, version)`` is robust there and identical to ``(version, ts)`` on
    every conforming task (#658, #763). A malformed/missing ts sorts first.
    """
    ts = _stage_event_ts(event)
    if ts is None:
        ts = datetime.min.replace(tzinfo=UTC)
    version = event.get("version")
    if not isinstance(version, int):
        version = 0
    return (ts, version)


def followup_label_groups(events: list[dict]) -> list[dict]:
    """Group ``epm:followup-scope`` entries by ``followup_label``.

    Scans scopes in ``(ts, version)`` order (see :func:`_scope_scan_key`).
    Per entry the label is resolved as:

    - ``parsed`` — the note carries a parseable ``followup_label``;
    - ``inherited-from-previous`` — unlabeled BUT the note carries a
      correction signal (:data:`CORRECTION_SIGNAL`): a correction follows
      the scope it corrects (#658 v2), so it attributes to the previous
      label;
    - ``pseudo-ts`` — unlabeled with NO correction signal: a DISTINCT queued
      follow-up under the pseudo-label ``unlabeled-<ts>`` (#685 v2). NEVER
      silently dropped, but NON-dispatchable (``unlabeled-<ts>`` violates the
      kebab-slug field contract and would name
      ``eval_results/issue_<N>/<label>/`` artifact dirs with colons) — Step 0
      surfaces these loudly as repair items instead of executing a malformed
      round. Dispatchability is FOUNDING-based: a group FOUNDED as
      ``pseudo-ts`` stays ``dispatchable: False`` even when a later unlabeled
      CORRECTION inherits into it (the inherit raises the group's
      authoritative entry but cannot repair the malformed label — only a
      re-post with a proper kebab-slug ``followup_label`` can).

    Returns one dict per label, in first-armed order, with JSON-native
    values: ``{followup_label, source, user_initiated, armed_ts,
    authoritative, label_parse, dispatchable, n_entries}``. Within a label
    the AUTHORITATIVE entry is the last in scan order (corrections land
    append-only — the #658 v3→v7 ``persona-vectors-style-rb`` chain);
    ``armed_ts`` is the FIRST entry's ts (a later correction never re-queues
    the label). ``source`` is the first parseable ``source`` across the
    group's entries in scan order (a correction note that omits ``source``
    must not demote a user-chat round), else ``"unknown"``.
    """
    scopes = sorted(
        (e for e in events if e.get("kind") == FOLLOWUP_SCOPE_KIND),
        key=_scope_scan_key,
    )
    prev_label: str | None = None
    groups: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}
    founded_pseudo: dict[str, bool] = {}
    for ev in scopes:
        note = ev.get("note") or ""
        label = parse_followup_note_field(note, "followup_label")
        if label:
            parse_mode = "parsed"
        elif prev_label is not None and CORRECTION_SIGNAL.search(note):
            label, parse_mode = prev_label, "inherited-from-previous"
        else:
            label, parse_mode = f"unlabeled-{ev.get('ts', '')}", "pseudo-ts"
        prev_label = label
        group = groups.get(label)
        if group is None:
            group = {
                "followup_label": label,
                "armed_ts": ev.get("ts", ""),
                "authoritative": ev,
                "label_parse": parse_mode,
                "n_entries": 0,
            }
            groups[label] = group
            sources[label] = []
            founded_pseudo[label] = parse_mode == "pseudo-ts"
        group["n_entries"] += 1
        group["authoritative"] = ev  # last in (ts, version) order wins
        group["label_parse"] = parse_mode
        src = parse_followup_note_field(note, "source")
        if src:
            sources[label].append(src)
    result: list[dict] = []
    for label, group in groups.items():
        group["source"] = sources[label][0] if sources[label] else "unknown"
        group["user_initiated"] = group["source"] in USER_INITIATED_FOLLOWUP_SOURCES
        # FOUNDING-based: a pseudo-founded group is a repair item forever —
        # a later unlabeled CORRECTION inheriting into it must NOT flip it
        # dispatchable (the label is still the malformed `unlabeled-<ts>`).
        group["dispatchable"] = not founded_pseudo[label] and group["label_parse"] in (
            "parsed",
            "inherited-from-previous",
        )
        result.append(group)
    return result


def unrun_followup_labels(events: list[dict]) -> list[dict]:
    """Label groups (per :func:`followup_label_groups`) with NO matching
    ``epm:same-issue-followup-run`` marker — the UNRUN queue.

    A LABEL is unrun iff no run marker carries the same ``followup_label``
    (workflow.yaml § markers — the label-keyed satisfier; the label's run
    marker closes ALL of its scope entries). Pseudo-label groups are
    INCLUDED (they must surface as repair items) but carry
    ``dispatchable: False`` — consumers that execute rounds filter on
    ``dispatchable``. A run marker with an unparseable label closes NOTHING
    (conservative in the anti-stranding direction; the counterweight is the
    Step 0 stale-label disposition rule / :func:`followup_retro_close_evidence`).

    Ordered deterministically: user-initiated labels first
    (:data:`USER_INITIATED_FOLLOWUP_SOURCES`), then oldest ``armed_ts``,
    then the authoritative entry's ``version``.
    """
    run_labels = {
        parse_followup_note_field(e.get("note") or "", "followup_label")
        for e in events
        if e.get("kind") == FOLLOWUP_RUN_KIND
    } - {None}
    unrun = [g for g in followup_label_groups(events) if g["followup_label"] not in run_labels]

    def _order(group: dict) -> tuple[bool, str, int]:
        version = group["authoritative"].get("version")
        return (
            not group["user_initiated"],
            group["armed_ts"],
            version if isinstance(version, int) else 0,
        )

    unrun.sort(key=_order)
    return unrun


def executing_followup_label(events: list[dict]) -> dict | None:
    """Resolve WHICH unrun label the current / most-recent round is executing.

    Shared by the Step 9b step-3 mid-round re-read, the step-4
    completion-marker label derivation, and the watcher's
    ``_post_followup_run_marker`` (SKILL.md Step 9b § Same-issue follow-up
    loop). Resolution order:

    1. The newest ``epm:progress`` note beginning
       ``stage-dispatch stage=followup-`` that is strictly newer than the
       newest ``epm:same-issue-followup-run`` ts and carries a
       ``label=<slug>`` token (via :func:`_breadcrumb_fields`) → that
       label's group, if unrun. This wins over the queue head because a
       user-chat label posted MID-ROUND would jump the head; the breadcrumb
       pins the round actually dispatched.
    2. Fallback: the head of the DISPATCHABLE subset of
       :func:`unrun_followup_labels` (Step 0 only ever dispatches
       dispatchable heads, so head == executing round whenever no labeled
       breadcrumb exists — breadcrumbs predating the #894 ``label=``
       contract lack the token).
    3. ``None`` when no dispatchable unrun labels exist.
    """
    unrun = unrun_followup_labels(events)
    crumb_label = _newest_followup_dispatch_crumb_label(events)
    if crumb_label is not None:
        for group in unrun:
            if group["followup_label"] == crumb_label:
                return group
    for group in unrun:
        if group["dispatchable"]:
            return group
    return None


def _newest_followup_dispatch_crumb_label(events: list[dict]) -> str | None:
    """``label=`` of the newest follow-up stage-dispatch breadcrumb strictly
    newer than the newest ``epm:same-issue-followup-run`` ts, else ``None``
    (no labeled crumb, or every labeled crumb predates the latest recorded
    round). Helper for :func:`executing_followup_label`."""
    newest_run_ts: datetime | None = None
    for ev in events:
        if ev.get("kind") != FOLLOWUP_RUN_KIND:
            continue
        ts = _stage_event_ts(ev)
        if ts is not None and (newest_run_ts is None or ts > newest_run_ts):
            newest_run_ts = ts
    crumb_label: str | None = None
    crumb_ts: datetime | None = None
    for ev in events:
        if ev.get("kind") != "epm:progress":
            continue
        note = (ev.get("note") or "").lstrip()
        if not note.startswith(_FOLLOWUP_STAGE_DISPATCH_PREFIX):
            continue
        label = _breadcrumb_fields(note).get("label")
        if not label:
            continue
        ts = _stage_event_ts(ev)
        if ts is None:
            continue
        if newest_run_ts is not None and ts <= newest_run_ts:
            continue
        if crumb_ts is None or ts > crumb_ts:
            crumb_ts = ts
            crumb_label = label
    return crumb_label


def followup_retro_close_evidence(events: list[dict], label: str) -> str | None:
    """MECHANICAL, exact-label evidence that ``label``'s round already ran.

    The predicate behind the Step 0 stale-label disposition rule (legacy
    tasks like #658 carry ghost labels whose rounds demonstrably ran without
    an ``epm:same-issue-followup-run`` record). Three evidence classes, all
    EXACT-match — prose mention / substring / prefix evidence NEVER closes:

    1. an ``epm:methodology-doc-generated`` note carrying ``extends=<label>``
       (exact token, via :func:`parse_followup_note_field` or the
       ``key=value`` breadcrumb grammar);
    2. an ``epm:free-analysis-followup-run`` whose ``followup_ref`` EXACTLY
       equals ``label`` (string equality — a PREFIX match like
       ``<label>-9a-ter-fit`` never closes);
    3. an ``epm:status-changed`` / ``epm:step-completed`` / ``epm:progress``
       note with the exact parenthesized round token ``(<label>)`` AND a
       round-completion word (:data:`_ROUND_COMPLETION_WORD`) on the same
       line.

    Returns the one-line evidence string of the FIRST matching class (class
    order 1 → 2 → 3; multiple classes agreeing on the SAME exact label are
    corroboration, not ambiguity — the canonical #658 ghost label carries
    both a 9a-quater ``extends=`` record and a status-PASS round note);
    ``None`` when NO class matches — the caller then never closes (a
    merely-prose / substring / prefix mention is not evidence; ambiguity
    NEVER closes).
    """
    for ev in events:
        kind = ev.get("kind")
        note = ev.get("note") or ""
        if kind == "epm:methodology-doc-generated":
            extends = parse_followup_note_field(note, "extends")
            if extends != label:
                extends = _breadcrumb_fields(note).get("extends")
            if extends == label:
                return (
                    f"epm:methodology-doc-generated at {ev.get('ts', '?')} carries extends={label}"
                )
    for ev in events:
        if ev.get("kind") != "epm:free-analysis-followup-run":
            continue
        ref = parse_followup_note_field(ev.get("note") or "", "followup_ref")
        if ref == label:
            return (
                f"epm:free-analysis-followup-run at {ev.get('ts', '?')} has followup_ref == {label}"
            )
    token = f"({label})"
    for ev in events:
        kind = ev.get("kind")
        if kind not in ("epm:status-changed", "epm:step-completed", "epm:progress"):
            continue
        for line in (ev.get("note") or "").splitlines():
            if token in line and _ROUND_COMPLETION_WORD.search(line):
                return (
                    f"{kind} at {ev.get('ts', '?')} carries the round token "
                    f"({label}) plus a round-completion word"
                )
    return None


def _format_failure_lesson_entry(new_lesson_ref: dict[str, str]) -> str:
    """Render the new (correcting) failure-lesson body as a durable memory entry.

    The append-only format is frozen by the golden fixture
    ``tests/fixtures/failure_lesson_append_only_pre712.txt`` (#712 §6): an H2
    slug heading, the lesson body, and a ``_Source: #<task_id> (failure-lesson)._``
    attribution line, each separated by a blank line and terminated with a single
    trailing newline. A deliberate format change updates the fixture in the same
    commit (the fixture IS the contract).
    """
    slug = new_lesson_ref.get("slug", "")
    task_id = new_lesson_ref.get("task_id", "")
    lesson = new_lesson_ref.get("lesson", "")
    return f"## {slug}\n\n{lesson}\n\n_Source: #{task_id} (failure-lesson)._\n"


def supersedes_action(
    prior_ref: str,
    durable_texts: dict[str, str],
    new_lesson_ref: dict[str, str],
) -> dict[str, str]:
    """Locate the durable failure-lesson entries a ``supersedes`` ref points at.

    ``prior_ref`` is a lesson slug (e.g. ``vllm_first_generate_is_a_code_bug``) or
    a marker timestamp (e.g. ``2026-06-28T01:26:58Z``). ``durable_texts`` maps a
    durable-file path -> its current text (agent-memory ``feedback_*.md`` bodies +
    ``gotchas.md`` bullets). ``new_lesson_ref`` carries the REAL superseding slug
    + task id (``{"slug": ..., "task_id": ...}``).

    Returns ``{path: annotated_text}`` for EVERY file whose current text CONTAINS
    ``prior_ref`` (slug substring OR marker-ts substring), with a CONCRETE
    ``[SUPERSEDED by <slug> — see #<task_id>] `` marker PREPENDED to that file's
    text (NEVER a ``<pending>`` placeholder; the prior content is preserved, not
    replaced). Returns ``{}`` when nothing matches — a dangling ref the caller
    logs as ``supersedes_unresolved`` and treats as a no-op annotation, never a
    dropped lesson. A ref matching MULTIPLE entries annotates ALL of them (the
    transitive chain is kept, #712 §7).

    Pure: no I/O, no writes. This helper DECIDES + ANNOTATES the prior subset;
    the composer (:func:`apply_failure_lesson`) assembles the FINAL durable map
    including the new lesson.
    """
    if not prior_ref:
        return {}
    slug = new_lesson_ref.get("slug", "")
    task_id = new_lesson_ref.get("task_id", "")
    marker = f"[SUPERSEDED by {slug} — see #{task_id}] "
    annotated: dict[str, str] = {}
    for path, text in durable_texts.items():
        if prior_ref in text:
            annotated[path] = marker + text
    return annotated


def apply_failure_lesson(
    block: dict[str, str],
    durable_texts: dict[str, str],
    new_lesson_ref: dict[str, str],
) -> dict[str, str]:
    """Compose the FINAL durable ``{path: text}`` map for a captured failure-lesson.

    Orchestration (pure — no I/O; the orchestrator writes the returned map via
    its existing explicit-path commit + push):

      1. If ``block["supersedes"]`` is set, call
         :func:`supersedes_action` to locate + concretely annotate the matched
         prior subset, then merge those annotations over a COPY of ``durable_texts``
         (a dangling ref merges nothing — every prior text stays byte-unchanged).
      2. APPEND the new (corrected) lesson's formatted body to the owning-agent
         memory file's text (key ``new_lesson_ref["memory_path"]`` — the file the
         action-2 path already writes), creating/extending that entry ALONGSIDE
         any prior content, never replacing it.
      3. Return the final ``{path: text}`` map.

    With ``supersedes`` ABSENT, step 1 is skipped and — over a clean (empty) owning
    -agent memory text — the produced text is BYTE-IDENTICAL to the pre-#712
    append-only result (pinned by the golden fixture, #712 §6). The new lesson
    ALWAYS lands — matched, dangling, or absent.
    """
    final = dict(durable_texts)

    prior_ref = (block.get("supersedes") or "").strip()
    if prior_ref:
        for path, annotated_text in supersedes_action(
            prior_ref, durable_texts, new_lesson_ref
        ).items():
            final[path] = annotated_text

    memory_path = new_lesson_ref.get("memory_path", "")
    if memory_path:
        entry = _format_failure_lesson_entry(new_lesson_ref)
        prior_text = final.get(memory_path, "")
        if prior_text.strip():
            final[memory_path] = prior_text.rstrip("\n") + "\n\n" + entry
        else:
            final[memory_path] = entry

    return final


def extract_stub_abstract(body: str) -> str:
    """Return the abstract from a paper-stub body (best-effort, never raises).

    The stub abstract is either a ``## Abstract`` H2 block OR the prose
    paragraph(s) between the H1 title and the paper link / first H2. Used to
    denormalize an ``abstract`` into REGISTRY.json so the dashboard hover-card
    + the REGISTRY title/abstract surfaces read it (the stub carries the
    abstract in the BODY, not the frontmatter). Markdown link/image lines and
    the H1 are stripped. Returns "" when nothing abstract-like is found.
    """
    # Strip any leading frontmatter the caller passed through.
    text = _strip_leading_frontmatter_blocks(body)
    # Prefer an explicit `## Abstract` block.
    m = re.search(r"(?ms)^##\s+Abstract\s*$(?P<body>.+?)(?=^##\s|\Z)", text)
    region = m.group("body") if m else text
    out: list[str] = []
    for raw_line in region.splitlines():
        line = raw_line.strip()
        if not line:
            if out:
                break  # first paragraph only
            continue
        if line.startswith("#"):  # H1/H2/H3 heading
            if out:
                break
            continue
        # Skip a bare paper-link line ("Paper: docs/papers/..." / "[PDF](...)").
        low = line.lower()
        if (low.startswith("paper:") or low.startswith("pdf:")) or (
            re.fullmatch(r"[\[\(!].*", line) and ("docs/papers/" in line or "/papers/" in line)
        ):
            if out:
                break
            continue
        out.append(line)
    return " ".join(out).strip()


def _paper_manifest_exists(task_id: int) -> bool:
    """True when docs/papers/issue_<N>/paper_manifest.json is on disk.

    Used by ``set_clean_result`` to detect a task that is INTENDED to be a
    paper-task (its paper artifacts exist) but whose body.md frontmatter has
    lost its ``paper: true`` opt-in — the #657 gate gap.
    """
    return (repo_root() / "docs" / "papers" / f"issue_{task_id}" / "paper_manifest.json").exists()


#: Artifacts the paper stores on the HF data repo (NOT committed to git), so
#: their local-existence/hash is NEVER validated — the PDF is validated via
#: ``pdf_hf_url`` instead (incident #657). A ``pdf`` entry may still appear in
#: ``artifacts`` in the OLD manifest shape; its local check is skipped there too.
_PAPER_HF_HOSTED_ARTIFACTS: tuple[str, ...] = ("pdf",)


def _validate_committed_artifact(label: str, entry: Any) -> list[str]:
    """Validate one COMMITTED manifest artifact's local presence + sha256.

    Returns a (possibly empty) list of problem strings. HF-hosted artifacts are
    the caller's concern (it skips them); this only stats committed files.
    """

    rel = (entry or {}).get("path")
    if not rel:
        return [f"artifact {label!r} has no path"]
    fpath = repo_root() / rel
    if not fpath.exists():
        return [f"artifact {label!r} path missing on disk: {rel}"]
    want = entry.get("sha256")
    if want and hashlib.sha256(fpath.read_bytes()).hexdigest() != want:
        return [f"artifact {label!r} sha256 mismatch ({rel})"]
    return []


def validate_paper_manifest(task_id: int) -> list[str]:
    """Validate a paper-task's docs/papers/issue_<N>/paper_manifest.json.

    Returns a list of human-readable problems; an empty list means the manifest
    is valid enough to flip ``has_clean_result``. HF-aware (mirrors
    ``scripts/verify_paper.py`` check 7): the COMMITTED local artifacts
    (``tex`` / ``paper_html``, + ``bib`` / ``refs_json`` when present) must be on
    disk with a matching sha256, AND the PDF must be validated via ``pdf_hf_url``
    (present + an ``https://...`` URL) — NOT a local file, since the paper PDF
    lives on the HF data repo, not in git (incident #657: the local PDF exists at
    BUILD time so the build-time verify passed, but post-commit it is HF-only and
    a local-existence check on a ``pdf`` artifact wrongly failed). A missing /
    null ``pdf_hf_url`` is a soft problem the CLI surfaces as a WARN (a
    local-only build can be promoted), returned with a ``WARN: `` prefix; a
    non-``https`` URL is a HARD problem. Tolerant of BOTH manifest shapes: the
    new ``hf_pdf`` block and the old ``pdf``-in-``artifacts`` shape (the latter's
    local-existence/hash check is skipped — it is HF-hosted).
    """
    problems: list[str] = []
    paper_dir = repo_root() / "docs" / "papers" / f"issue_{task_id}"
    manifest_path = paper_dir / "paper_manifest.json"
    if not manifest_path.exists():
        return [f"no paper_manifest.json at {manifest_path.relative_to(repo_root())}"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"paper_manifest.json unreadable: {e}"]
    if manifest.get("schema") != "paper_manifest/v1":
        problems.append(f"schema is {manifest.get('schema')!r}, expected 'paper_manifest/v1'")
    artifacts = manifest.get("artifacts") or {}
    # Required COMMITTED artifacts — NOT the PDF (HF-hosted, validated via URL).
    for label in ("tex", "paper_html"):
        if not artifacts.get(label):
            problems.append(f"missing required committed artifact {label!r}")
            continue
        problems.extend(_validate_committed_artifact(label, artifacts[label]))
    # Any other committed artifact present in the map (bib / refs_json) is also
    # locally validated; the HF-hosted PDF entry (old shape) is skipped.
    for label, entry in artifacts.items():
        if label in ("tex", "paper_html") or label in _PAPER_HF_HOSTED_ARTIFACTS:
            continue
        problems.extend(_validate_committed_artifact(label, entry))
    # The PDF is validated via the HF URL (top-level ``pdf_hf_url`` or the new
    # ``hf_pdf.url`` block), NOT a local file.
    pdf_url = manifest.get("pdf_hf_url") or (manifest.get("hf_pdf") or {}).get("url")
    if not pdf_url:
        problems.append("WARN: pdf_hf_url is null (local-only build — not yet uploaded)")
    elif not str(pdf_url).startswith("https://"):
        problems.append(f"pdf_hf_url is not an https:// URL: {pdf_url!r}")
    return problems


# Goal H2 helpers
# ────────────────────────────────────────────────────────────────────────────
# The ``## Goal`` H2 block carries the one-sentence experiment intent, and
# sits between the H1 title (if any) and the next H2 (typically ``## TL;DR``
# or the original task body's first section). The body authoritatively
# carries the goal text; the frontmatter ``goal:`` field is a denormalized
# mirror so consumers (REGISTRY, dashboard, subagent briefs) can read it
# without parsing markdown.
# ─── Path resolution ────────────────────────────────────────────────────────


def _status_from_path(path: Path) -> str:
    """Given tasks/<status>/<id>/, return <status>."""
    rel = path.relative_to(tasks_dir())
    return rel.parts[0]


def find_task_path(task_id: int) -> Path:
    """Return absolute path to tasks/<status>/<task_id>/. Resolves via REGISTRY.

    Stale-entry envelope (#825): when the registry entry points at a dir that
    is MISSING on disk (a mutation was hard-killed between the folder move
    and the registry save), fall back to a one-shot on-disk scan across
    STATUSES — exactly one hit returns that path with a logged drift WARNING;
    two or more hits raise ``StaleTaskPathError`` (real corruption — never
    guess); zero hits raise the original ``FileNotFoundError``. This is a
    READ path, so the registry is NEVER self-healed here (no ``_locked()``
    held; an unlocked whole-file ``_save_registry`` could clobber a
    concurrent mutator's update) — repair happens on the task's next
    registry-writing mutation (including the ``set_status``
    same-transition early-return re-sync) or via
    ``task.py audit --repair --apply``.
    """
    reg = _load_registry()
    entry = reg["tasks"].get(str(task_id))
    td = tasks_dir()
    if not entry:
        # Fall back to scanning the filesystem in case REGISTRY is stale
        for status in STATUSES:
            candidate = td / status / str(task_id)
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"task #{task_id} not found in registry or on disk")
    abs_path = repo_root() / entry["path"]
    if abs_path.is_dir():
        return abs_path
    # Registry entry is STALE (dir moved on disk without a registry update —
    # e.g. a mutation was hard-killed mid-flight; cf. #825). Fall back to a
    # one-shot on-disk scan; READ path, so never self-heal REGISTRY here (no
    # _locked()).
    hits = [td / s / str(task_id) for s in STATUSES if (td / s / str(task_id)).is_dir()]
    if len(hits) == 1:
        _log.warning(
            "task #%d: REGISTRY says %r but that dir is missing; found on disk at %r — "
            "returning the on-disk path. REGISTRY is stale; it re-syncs on the next "
            "registry-writing mutation of this task, or run "
            "`task.py audit --repair --apply`.",
            task_id,
            entry["path"],
            str(hits[0].relative_to(repo_root())),
        )
        return hits[0]
    if len(hits) > 1:
        raise StaleTaskPathError(
            f"task #{task_id}: REGISTRY says {entry['path']!r} (missing) and the "
            f"task exists in MULTIPLE status folders: "
            f"{[str(h.relative_to(repo_root())) for h in hits]}; "
            f"run `task.py audit --repair --apply`"
        )
    raise FileNotFoundError(
        f"task #{task_id} registry says {entry['path']!r} but that dir is missing; "
        f"run `task.py audit` to repair"
    )


def get_task(task_id: int) -> dict[str, Any]:
    """Return a structured snapshot of a task: frontmatter, body, status."""
    path = find_task_path(task_id)
    fm, body = _read_body(path / "body.md")
    return {
        "id": task_id,
        "path": str(path.relative_to(repo_root())),
        "status": _status_from_path(path),
        "frontmatter": fm,
        "body": body,
    }


# ─── Events ─────────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Linux atomic-append bound: a single write(2) of <= PIPE_BUF bytes on an
# O_APPEND fd lands atomically w.r.t. concurrent appenders AND is all-or-
# nothing against a SIGKILL (the kernel either commits the one syscall or it
# does not). POSIX.1-2017 write(2); Linux pipe(7) PIPE_BUF. ABOVE this bound a
# single write may be split, so we keep the caller's flock (excludes other
# writers) and complete the buffer in a loop. The loop is NOT crash-atomic:
# a SIGKILL between two os.write calls can leave a partial line — recovery for
# the oversize case is the tolerant reader (_iter_jsonl), which skips it.
# Confirmed PIPE_BUF == 4096 on the EPS VM (Linux 6.8).
_PIPE_BUF = getattr(os, "PIPE_BUF", 4096)

_log = logging.getLogger(__name__)


def _append_jsonl_line(path: Path, payload: dict[str, Any]) -> None:
    """Atomically append one JSON object as a line to an append-only log.

    Serializes ``json.dumps(payload, ensure_ascii=False) + "\\n"`` in memory,
    then writes the whole buffer to ``path`` opened ``O_WRONLY|O_APPEND|O_CREAT``.

    For buffers <= PIPE_BUF the single ``os.write`` is atomic against other
    appenders (POSIX) AND against a SIGKILL/OOM mid-call (the one syscall
    lands the full line or nothing — never a partial line).

    For OVERSIZE buffers (rare: a large note plus artifacts list, > PIPE_BUF)
    POSIX promises neither single-write atomicity nor all-or-nothing across
    multiple ``os.write`` calls if the process is killed mid-loop. This path
    relies on the caller already holding ``_locked()`` to exclude other
    writers (no interleaving) and a write-completion loop to finish the
    buffer despite short writes — but it is NOT crash-atomic: a SIGKILL
    between two ``os.write`` calls CAN leave a partial trailing line on disk.
    Recovery for that case is the tolerant reader (``_iter_jsonl``), which
    skips the partial line. The loop is still useful for completion under
    EAGAIN/EINTR/short-writes; it is not a crash-atomicity guarantee.

    Callers MUST hold ``_locked()``. This helper does NOT acquire the lock and
    does NOT commit — the caller owns flock + ``_git_commit`` semantics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    buf = line.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if len(buf) <= _PIPE_BUF:
            # Single atomic append (<= PIPE_BUF): all-or-nothing against a
            # SIGKILL. os.write may legally short-write even here in
            # principle; assert the whole line landed so a truncation can
            # never pass silently.
            n = os.write(fd, buf)
            if n != len(buf):
                raise OSError(f"short atomic append to {path}: wrote {n} of {len(buf)} bytes")
        else:
            # Oversize line (> PIPE_BUF): caller's flock excludes concurrent
            # writers (no interleaving); complete the buffer despite short
            # writes. os.write already retries EINTR internally on py3.5+, so
            # a returned short count is a genuine partial flush to finish, not
            # an EINTR retry. NOT crash-atomic — a mid-loop SIGKILL can leave
            # a partial line; the tolerant reader recovers it.
            view = memoryview(buf)
            written = 0
            while written < len(buf):
                n = os.write(fd, view[written:])
                if n == 0:
                    # os.write never returns 0 on a regular fd (it raises on
                    # error), but guard the loop against a spin anyway — a
                    # zero-progress write is a hard error, not something to
                    # retry forever.
                    raise OSError(f"os.write returned 0 appending to {path}")
                written += n
    finally:
        os.close(fd)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse an append-only JSONL file, tolerating malformed lines.

    A line that does not round-trip through ``json.loads`` is SKIPPED and
    logged at WARNING (this is the recovery path for a partial trailing line
    left by a writer killed mid-append, the historical #653 corruption, AND
    the >PIPE_BUF oversize-append crash case which is NOT write-atomic).
    All malformed lines are tolerated, not just the trailing one: an
    append-only log can only ever grow a bad line at the tail, so a mid-file
    bad line is implausible, and the practical recovery a reader needs is
    "return everything parseable" — raising on a mid-file anomaly would
    re-introduce exactly the hard-crash this fix removes.

    Decoding is TOLERANT (``errors="replace"``): a SIGKILL during a
    ``>PIPE_BUF`` ``ensure_ascii=False`` append can leave a TRUNCATED
    multibyte UTF-8 sequence at the file tail (e.g. ``b'{"note":"\\xe2'``).
    Strict UTF-8 (the ``read_text()`` default) would raise
    ``UnicodeDecodeError`` BEFORE the per-line ``json.loads`` loop ever
    reached the ``JSONDecodeError`` handler, hard-crashing all four readers.
    ``errors="replace"`` substitutes U+FFFD for the bad bytes so the
    corrupted line falls through to the existing ``JSONDecodeError`` skip
    path — completing the recovery story.

    Records are split on ``"\\n"`` (NOT ``str.splitlines()``): the paired
    ``ensure_ascii=False`` writer leaves raw U+2028/U+2029/NEL inside note
    strings, and ``splitlines()`` treats those as line boundaries — shredding
    a valid record into skip-malformed fragments = silent marker loss
    (gotchas.md; #825 → #950).
    """
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    # split("\n"), NOT splitlines(): raw U+2028/U+2029/NEL inside
    # ensure_ascii=False notes are Unicode line boundaries that would shred
    # valid records into skip-malformed fragments = silent marker loss
    # (gotchas.md; #825 → #950).
    for lineno, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            _log.warning(
                "skipping malformed line %d in %s: %s",
                lineno,
                path,
                str(e)[:200],
            )
    return out


def _next_event_version(events_path: Path, kind: str) -> int:
    """Return ``max(existing versions for this kind) + 1`` (1 when the kind
    is new) for the events file at ``events_path``.

    Mirrors ``new_plan_version``'s max+1 (NOT count+1) semantics so a later
    defaulted post can never shadow an explicit higher version posted
    earlier. Caller must hold the workflow lock — the read-then-append must
    be atomic against concurrent posters.
    """
    if not events_path.exists():
        return 1
    highest = 0
    for row in _iter_jsonl(events_path):
        if row.get("kind") != kind:
            continue
        v = row.get("version")
        if isinstance(v, int) and v > highest:
            highest = v
    return highest + 1


def post_event(
    task_id: int,
    kind: str,
    *,
    version: int | None = None,
    by: str = "unknown",
    note: str | None = None,
    artifacts: list[str] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    """Append a single event to tasks/<status>/<id>/events.jsonl.

    When ``version`` is omitted it is derived per marker kind as
    ``max(existing versions for this kind) + 1`` (1 when the kind is new),
    so the "highest version per kind wins" resume contract holds without
    every caller having to remember an explicit version (incident #480:
    two defaulted re-posts both landed version 1 below an existing v6,
    making the stale v6 authoritative on resume). An explicit ``version``
    always wins.

    Note size is capped at EVENT_NOTE_MAX chars to mirror Sagan; oversize
    raises ValueError so the caller can fall back to a failure marker.
    """
    if note is not None and len(note) > EVENT_NOTE_MAX:
        raise ValueError(
            f"event note exceeds {EVENT_NOTE_MAX} chars ({len(note)}); "
            f"caller must post epm:failure v1 with reason=note_oversize"
        )
    with _locked():
        path = find_task_path(task_id) / "events.jsonl"
        if version is None:
            version = _next_event_version(path, kind)
        payload: dict[str, Any] = {
            "ts": _utcnow_iso(),
            "kind": kind,
            "version": version,
            "by": by,
        }
        if note is not None:
            payload["note"] = note
        if artifacts:
            payload["artifacts"] = artifacts
        payload.update(extras)
        _append_jsonl_line(path, payload)
        _git_commit(
            [path],
            f"task #{task_id}: {kind}" + (f" — {note[:60]}" if note else ""),
        )
    return payload


def list_events(task_id: int) -> list[dict[str, Any]]:
    path = find_task_path(task_id) / "events.jsonl"
    return _iter_jsonl(path)


def latest_event(task_id: int, prefix: str | None = None) -> dict[str, Any] | None:
    events = list_events(task_id)
    if prefix:
        events = [e for e in events if e["kind"].startswith(prefix)]
    return events[-1] if events else None


def has_event(task_id: int, kind: str) -> bool:
    return any(e["kind"] == kind for e in list_events(task_id))


# ─── Status transitions ────────────────────────────────────────────────────


def _rollback_move(src: Path, dst: Path) -> None:
    """Best-effort ``shutil.move(src -> dst)`` to undo a partial status-move.

    Called by ``set_status`` when the post-move completeness check fails: it
    puts the task dir back at its ORIGINAL location so REGISTRY (never touched
    on this path) stays consistent with the filesystem. On rollback failure it
    LOGS loudly (a failed rollback is a louder problem than the original) and
    RE-RAISES so the fault surfaces — never swallowed.
    """
    try:
        shutil.move(str(src), str(dst))
    except Exception:
        _log.error(
            "status-move rollback FAILED: could not move %s back to %s; the task "
            "dir may be left at the incomplete destination. Run `task.py audit` "
            "to detect and `task.py audit --repair --apply` to repair.",
            src,
            dst,
        )
        raise


def _task_status_dir_pathspecs(task_id: int, repo: Path) -> list[str]:
    """All ``tasks/<status>/<id>`` dirs for this task that git TRACKS or that
    exist on disk — the staging pathspec set that reconciles any residue a
    previously-crashed transition left behind (ghost old-status dirs whose
    deletions were never staged; #825 / the #644 stale-task-folder class).

    One ``git ls-files`` call over deterministic per-STATUSES pathspecs (no
    fnmatch wildcards — exact-id matching only, so id 89 never sweeps id
    898). ``ls-files`` silently ignores pathspecs that match nothing, so the
    candidate list is safe to pass wholesale. The returned set is restricted
    to tracked-or-on-disk dirs BECAUSE ``git add`` (without
    ``--ignore-unmatch``) and ``git commit --only`` both FAIL LOUD on a
    pathspec that matches neither the index/HEAD nor the working tree — a
    dir that is neither tracked nor on disk has nothing to stage and must be
    excluded, not passed along.
    """
    rel_tasks = tasks_dir().relative_to(repo)
    candidates = [str(rel_tasks / status / str(task_id)) for status in STATUSES]
    tracked = _run_git(["ls-files", "--", *candidates]).stdout.splitlines()
    n_parts = len(rel_tasks.parts) + 2  # <tasks>/<status>/<id>
    dirs = {"/".join(p.split("/")[:n_parts]) for p in tracked if p.strip()}
    dirs |= {c for c in candidates if (repo / c).is_dir()}
    return sorted(dirs)


def set_status(
    task_id: int,
    new_status: str,
    *,
    note: str | None = None,
    force_followup_exit: bool = False,
) -> Path:
    """Move tasks/<old>/<id>/ → tasks/<new>/<id>/ (whole-dir move), then post
    a status-changed event and commit. Returns the new absolute path.

    Crash envelope (#825): ALL durable-state mutations — the filesystem move
    + completeness verification, the REGISTRY save, and the events.jsonl
    append — complete BEFORE any git operation, so the #825 stranded split
    (folder moved, registry pointing at the old path) is unreachable via
    git-failure exceptions: any git crash leaves disk, REGISTRY, and
    events.jsonl all consistent with the transition APPLIED, with only the
    COMMIT missing; that residue is reconciled by the ghost-aware staging
    sweep (``_task_status_dir_pathspecs``) on the task's NEXT transition. A
    HARD KILL (SIGKILL/OOM) in the residual window between ``shutil.move``
    and ``_save_registry`` still yields a stale-registry shape; that window
    is backstopped by the ``find_task_path`` read-path scan fallback plus
    the same-transition early-return re-sync below, and the next completed
    registry-writing mutation of the task re-syncs the entry. A hard kill
    between ``_save_registry`` and ``_append_jsonl_line`` yields a
    consistent folder+registry with a missing ``epm:status-changed`` event —
    a history gap, not a stranding. On the FS-verification failure path the
    move is rolled back with REGISTRY untouched (``_rollback_move``), as
    before. On git failure this function still RAISES (fail fast) — but the
    status transition is durably applied, and a caller retry lands on the
    idempotent ``old_status == new_status`` early return.

    Refuses `followups_running` → any FOLLOWUP_HELD_BLOCKED_STATUSES member
    (same-issue follow-up status-hold rule) unless ``force_followup_exit``.
    """
    if new_status not in STATUSES:
        raise ValueError(f"unknown status: {new_status!r}; expected one of {STATUSES}")
    with _locked():
        old = find_task_path(task_id)
        old_status = _status_from_path(old)
        if old_status == new_status:
            # Idempotent retry of the SAME transition. If find_task_path
            # resolved the task at a path that DISAGREES with the registry
            # entry (stale entry — the hard-kill residue shape, #825),
            # re-sync the registry before returning: this branch already
            # holds _locked(), so the write is safe (unlike the read-path
            # scan fallback in find_task_path, which never self-heals).
            reg = _load_registry()
            entry = reg["tasks"].get(str(task_id))
            rel = str(old.relative_to(repo_root()))
            if entry and entry.get("path") != rel:
                fm, _ = _read_body(old / "body.md")
                _registry_set(reg, task_id, old, fm)
                _save_registry(reg)
                _git_commit(
                    [old, registry_path()],
                    f"task #{task_id}: re-sync stale REGISTRY entry",
                )
                _log.warning(
                    "task #%d: re-synced stale REGISTRY entry (%r -> %r) on idempotent retry",
                    task_id,
                    entry.get("path"),
                    rel,
                )
            return old
        if (
            old_status == "followups_running"
            and new_status in FOLLOWUP_HELD_BLOCKED_STATUSES
            and not force_followup_exit
        ):
            raise ValueError(
                f"task #{task_id}: refusing followups_running -> {new_status}. "
                "followups_running is HELD for the WHOLE same-issue follow-up round "
                "(status-hold rule, .claude/skills/issue/SKILL.md Step 9b § Same-issue "
                "follow-up loop, step 3): the normal pipeline set-status calls are "
                "SKIPPED mid-round; phase visibility comes from stage breadcrumbs "
                "(stage=followup-<phase>) + epm:progress markers. The round exits this "
                "status only at the re-park (awaiting_promotion) or a failure exit "
                "(blocked). Pass --force-followup-exit (CLI) / force_followup_exit=True "
                "(API) only to deliberately abandon the round."
            )
        repo = repo_root()
        new_parent = tasks_dir() / new_status
        new_parent.mkdir(parents=True, exist_ok=True)
        new = new_parent / str(task_id)
        # Destination-collision guard (incident #681, 2026-06-28). `git mv SRC
        # DST` where DST already exists does NOT error — git nests SRC inside
        # DST as tasks/<new>/<id>/<id>/, exits 0, and the failure surfaces only
        # later at `_read_body(new / "body.md")`, leaving the transition
        # half-applied. So check DST up front.
        if new.exists():
            if not new.is_dir():
                raise ValueError(
                    f"task #{task_id}: cannot move to {new_status}: destination "
                    f"{new} exists and is not a directory. Inspect/remove it: "
                    f"ls -la {new}"
                )
            if any(new.iterdir()):
                raise ValueError(
                    f"task #{task_id}: cannot move to {new_status}: destination "
                    f"{new} already exists and is non-empty (orphan dir or "
                    f"leftover artifacts from a prior numbering / concurrent "
                    f"session). `git mv` would nest the task as {new}/{task_id}/. "
                    f"Inspect it (`git -C {repo} status -- {new.relative_to(repo)}` "
                    f"and `ls -la {new}`) and remove/relocate it before retrying."
                )
            # Empty orphan directory: remove it so `git mv` performs a true
            # rename rather than nesting the source inside it. git does not
            # track empty dirs, so this is an untracked filesystem op that adds
            # nothing to the commit and leaves no staged change.
            new.rmdir()
        rel_old = old.relative_to(repo)
        rel_new = new.relative_to(repo)
        # Whole-directory filesystem move + completeness verification (#722).
        # `git mv <src-dir> <dst-dir>` renames only git-TRACKED files, silently
        # leaving untracked/uncommitted files (an uncommitted plan version, a
        # subagent artifact written before this transition's commit) behind and
        # splitting the task across two folders. `shutil.move` of the whole dir
        # moves EVERY file (tracked, untracked, modified) in one rename, so
        # nothing is left behind by construction. The destination-collision
        # guard above already ensured `new` does not exist, so this is a true
        # rename into a non-existent destination (never a nest).
        src_files = {p.relative_to(old) for p in old.rglob("*") if p.is_file()}
        shutil.move(str(old), str(new))
        # Verify EVERY source file landed in the destination. On any miss, roll
        # the FS move back BEFORE REGISTRY is touched, so a partial move can
        # never leave REGISTRY pointing at an incomplete dir (the #722
        # half-applied state). REGISTRY is untouched on this failure path.
        dst_files = {p.relative_to(new) for p in new.rglob("*") if p.is_file()}
        missing = src_files - dst_files
        if missing:
            _rollback_move(new, old)
            preview = sorted(str(m) for m in missing)[:5]
            raise RuntimeError(
                f"task #{task_id}: status move {old_status} -> {new_status} left "
                f"{len(missing)} file(s) behind: {preview}; filesystem move rolled "
                f"back, REGISTRY untouched. Retry after resolving the disk/"
                f"permission issue."
            )
        # ── Durable state FIRST, git ops LAST (#825). ── The registry save +
        # event append below complete before ANY git op, so a git crash can
        # never again leave the folder moved with the registry pointing at
        # the old path (the #825 stranded split): every git failure now
        # leaves disk, REGISTRY, and events.jsonl consistent with the
        # transition applied, only the COMMIT missing.
        reg = _load_registry()
        fm, _ = _read_body(new / "body.md")
        _registry_set(reg, task_id, new, fm)
        _save_registry(reg)
        # Append event
        ev_path = new / "events.jsonl"
        payload = {
            "ts": _utcnow_iso(),
            "kind": "epm:status-changed",
            "version": 1,
            "by": "task.py",
            "from": old_status,
            "to": new_status,
        }
        if note:
            payload["note"] = note
        _append_jsonl_line(ev_path, payload)
        specs: list[str] = []  # pre-bound so the except block below can log it
        try:
            # Ghost-aware staging (#644 stale-task-folder class): the specs
            # cover BOTH sides of THIS move — the source-side deletion at
            # <old> (tracked in git, hence in the specs) AND the
            # destination-side addition at <new> (on disk, hence in the
            # specs) — preserving the both-sides-of-move commit invariant —
            # PLUS any tasks/<status>/<id> dir a previously-crashed
            # transition left tracked in HEAD but absent on disk, so
            # `git add --all` stages the leftover deletion and this
            # transition's commit sweeps the ghost duplicate. rel_old /
            # rel_new are NOT force-unioned in: a dir that is neither
            # tracked nor on disk (e.g. rel_old after a never-committed
            # transition) matches no pathspec, and `git add` / `git commit
            # --only` fail loud on unmatched pathspecs — the helper's
            # tracked-or-on-disk restriction already includes rel_old/
            # rel_new whenever there is anything to stage for them.
            specs = _task_status_dir_pathspecs(task_id, repo)
            _run_git(["add", "--all", "--", *specs])  # step-6 standalone staging
            # Pass the SAME expanded path set to _git_commit so the deletion
            # side of the move (and any swept ghost) is included in the
            # commit's --only pathspec. Otherwise staged deletions remain in
            # the index and get swept into the next unrelated `git commit`
            # (incident: 2026-05-24, tasks 382/383 source-side deletions
            # leaked into commit 49e49f4a).
            _git_commit(
                [*(repo / s for s in specs), registry_path()],
                f"task #{task_id}: {old_status} → {new_status}",
            )
        except subprocess.CalledProcessError:
            _log.error(
                "task #%d: status move %s -> %s is DURABLY APPLIED (disk + REGISTRY + "
                "events.jsonl consistent at %s) but git failed before committing. "
                "Leftover git residue at %s / %s will be reconciled by the NEXT "
                "set_status of this task; to sweep now: "
                "git add --all -- %s && git commit -m "
                "'task #%d: sweep crashed status-move residue'",
                task_id,
                old_status,
                new_status,
                rel_new,
                rel_old,
                rel_new,
                " ".join(specs) if specs else f"{rel_old} {rel_new}",
                task_id,
            )
            raise
    return new


# ─── Task creation ──────────────────────────────────────────────────────────


@dataclass
class NewTaskRequest:
    kind: str  # experiment | infra | analysis | survey | campaign | human kinds
    title: str
    body: str = ""
    parent_id: int | None = None
    tags: list[str] | None = None
    status: str = "proposed"
    # Canonical Goal of the experiment. Honored only when kind=="experiment";
    # passed through for other kinds with a soft warning emitted by the CLI.
    goal: str | None = None
    # Verbatim user prompt(s) that originated the task. Written to
    # frontmatter `origin_prompt:` when non-empty (honored for any kind).
    # The clean-result `## Reproducibility` `**Context:**` row carries it
    # forward (SPEC.md § `**Context:**` row; verify_task_body.py check 17).
    origin_prompt: str | None = None
    # Workflow-pipeline version this task runs under: "v1" (current default)
    # or "v2" (report-only pipeline). None -> resolved at creation via
    # _resolve_workflow_version(): explicit > env EPM_DEFAULT_WORKFLOW >
    # DEFAULT_WORKFLOW_VERSION. Always written to frontmatter `workflow:` so
    # the `/issue` dispatcher can branch (EPS workflow-v2 plan, Assumption 2).
    workflow: str | None = None


def create_task(req: NewTaskRequest) -> int:
    """Create tasks/<status>/<NEW_ID>/ with body.md (frontmatter + body),
    empty events.jsonl, empty comments.jsonl. Returns the new ID.
    """
    if req.status not in STATUSES:
        raise ValueError(f"unknown status: {req.status!r}")
    if req.kind not in KINDS:
        raise ValueError(f"unknown kind: {req.kind!r}; expected one of {KINDS}")
    with _locked():
        reg = _load_registry()
        task_id = reg.get("highest_id", 0) + 1
        path = tasks_dir() / req.status / str(task_id)
        path.mkdir(parents=True, exist_ok=False)
        (path / "artifacts").mkdir()
        (path / "plans").mkdir()
        fm: dict[str, Any] = {
            "title": req.title,
            "kind": req.kind,
            "tags": req.tags or [],
            "created_at": _utcnow_iso(),
            "has_clean_result": False,
        }
        if req.parent_id is not None:
            fm["parent_id"] = req.parent_id
        if req.origin_prompt and req.origin_prompt.strip():
            fm["origin_prompt"] = req.origin_prompt.strip()
        # Pin the workflow-pipeline version (explicit > EPM_DEFAULT_WORKFLOW >
        # v1). Always written so the /issue dispatcher can branch; purely
        # additive — legacy tasks with no `workflow:` key fail-open to v1 via
        # workflow_version() (EPS workflow-v2 plan, Assumption 2).
        fm["workflow"] = _resolve_workflow_version(req.workflow)
        # Inject the Goal into frontmatter + body H2 when kind=experiment.
        # For other kinds, ignore silently — enforcement is at /issue
        # Step 0c, and task.py CLI warns the user up front.
        seed_body = req.body if req.body.endswith("\n") else req.body + "\n"
        if req.kind == "experiment" and req.goal and req.goal.strip():
            fm["goal"] = req.goal.strip()
            seed_body = _inject_or_replace_goal_h2(seed_body, req.goal.strip())
        _write_body(path / "body.md", fm, seed_body)
        # Empty event + comment logs (touch)
        (path / "events.jsonl").touch()
        (path / "comments.jsonl").touch()
        # Seed event
        created_event = {
            "ts": _utcnow_iso(),
            "kind": "epm:created",
            "version": 1,
            "by": "task.py",
            "kind_": req.kind,
        }
        _append_jsonl_line(path / "events.jsonl", created_event)
        # Register
        _registry_set(reg, task_id, path, fm)
        _save_registry(reg)
        _git_commit([path, registry_path()], f"task #{task_id}: create — {req.title[:60]}")
        return task_id


# ─── Body / frontmatter mutations ──────────────────────────────────────────


# Frontmatter keys that DO round-trip through ``set_body`` when present in the
# incoming body's frontmatter. The paper clean-result track opts a task into
# paper-mode by carrying ``paper: true`` (and a denormalized ``abstract``) in
# the stub file's frontmatter — those keys MUST survive the set-body write or
# the dashboard's ``isPaperTask`` read fails and the paper renders as a
# markdown stub (incident #657, 2026-06-25). Everything else stays governed by
# the dedicated mutators. ``paper`` is never cleared by an incoming body that
# omits it (set-body is body-only for non-paper-opt-in callers); it is only
# turned ON when the incoming frontmatter opts in. ``abstract`` rides along so
# the REGISTRY denormalization has the paper's abstract.
_SET_BODY_ROUNDTRIP_KEYS: tuple[str, ...] = ("paper", "abstract")


def set_body(task_id: int, new_body: str, *, snapshot_original: bool = False) -> None:
    """Replace the body content (preserves frontmatter).

    If `snapshot_original` is True, save the current full body.md to
    original-body.md first — used by the analyzer when promoting a
    clean-result.

    Any YAML frontmatter at the START of ``new_body`` is stripped from the
    body region before the canonical frontmatter (loaded from the existing
    body.md) is prepended. This prevents the duplicate-frontmatter trap
    when callers pass a complete markdown document (frontmatter + body) —
    see `_strip_leading_frontmatter_blocks` for the incident history. The
    strip is idempotent: calling `set_body` with a body that already has
    no leading frontmatter is a no-op for the strip step.

    Note: this function preserves the EXISTING frontmatter on body.md.
    If you need to change frontmatter fields, use the dedicated mutators
    (`set_title`, `set_clean_result`, `add_tag`, `remove_tag`,
    `set_goal`). The ONE exception is the paper clean-result opt-in: the
    keys in ``_SET_BODY_ROUNDTRIP_KEYS`` (``paper``, ``abstract``) ARE
    carried forward from ``new_body``'s frontmatter when present, because
    the paper-stub track carries ``paper: true`` ONLY in the stub file and
    the dashboard's ``isPaperTask`` read depends on it landing in body.md
    (incident #657). All other frontmatter from ``new_body`` is discarded.
    """
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, _ = _read_body(path)
        # Carry the paper-opt-in keys forward from the incoming body's
        # frontmatter so a paper-stub write does not silently drop
        # ``paper: true`` (incident #657). Parse defensively — a malformed
        # leading block leaves ``incoming_fm`` empty and changes nothing.
        try:
            incoming_fm, _ = _split_frontmatter(new_body)
        except ValueError:
            incoming_fm = {}
        for key in _SET_BODY_ROUNDTRIP_KEYS:
            if key in incoming_fm and incoming_fm[key] is not None:
                fm[key] = incoming_fm[key]
        touched: list[Path] = [path]
        if snapshot_original:
            orig = path.parent / "original-body.md"
            shutil.copy2(path, orig)
            touched.append(orig)
        body_text = _strip_leading_frontmatter_blocks(new_body)
        _write_body(path, fm, body_text if body_text.endswith("\n") else body_text + "\n")
        # Keep REGISTRY in sync when the paper opt-in (or abstract) landed —
        # the denormalized ``paper``/``abstract``/``title`` surfaces feed the
        # dashboard list view + hover card.
        if any(k in incoming_fm for k in _SET_BODY_ROUNDTRIP_KEYS):
            reg = _load_registry()
            _registry_set(reg, task_id, path.parent, fm)
            _save_registry(reg)
            touched.append(registry_path())
        _git_commit(touched, f"task #{task_id}: set-body")


def set_title(task_id: int, title: str) -> None:
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        fm["title"] = title
        _write_body(path, fm, body)
        # Also update REGISTRY snapshot
        reg = _load_registry()
        _registry_set(reg, task_id, path.parent, fm)
        _save_registry(reg)
        _git_commit([path, registry_path()], f"task #{task_id}: set-title — {title[:60]}")


def set_kind(task_id: int, kind: str) -> None:
    """Reclassify a task's ``kind`` in frontmatter (+ REGISTRY snapshot).

    The canonical, flock-protected, registry-consistent way to correct a
    MISFILED ``kind`` — e.g. a fix-validation / "test that X works" task
    created as ``kind: experiment`` that should be ``kind: infra`` (so it
    completes on the test-verdict path instead of being dragged through the
    clean-result/promotion machinery; incident #672). Without this, a
    correction required a direct frontmatter hand-edit, bypassing the
    "mutate only through task.py" rule.

    ``kind`` MUST be a member of :data:`KINDS`; an invalid value raises
    ``ValueError``. ``kind`` IS denormalized into REGISTRY.json (the
    dashboard list view reads it from there), so this updates the registry
    snapshot exactly like :func:`set_title` — skipping that would leave the
    list view showing the stale kind.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind!r}; expected one of {KINDS}")
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        fm["kind"] = kind
        _write_body(path, fm, body)
        # REGISTRY denormalizes `kind` (dashboard list view reads it) — keep
        # it in sync, same as set_title.
        reg = _load_registry()
        _registry_set(reg, task_id, path.parent, fm)
        _save_registry(reg)
        _git_commit([path, registry_path()], f"task #{task_id}: set-kind — {kind}")


def set_clean_result(task_id: int, value: bool = True, *, allow_paper_warn: bool = True) -> None:
    """Flip ``has_clean_result`` (or clear it with ``value=False``).

    For a ``paper: true`` task, flipping ``has_clean_result`` to True first
    VALIDATES the task's ``docs/papers/issue_<N>/paper_manifest.json`` (the
    COMMITTED local artifacts present + sha256 match, and the HF-hosted PDF
    validated via an ``https`` ``pdf_hf_url`` — NOT a local PDF file, incident
    #657) via :func:`validate_paper_manifest`. A HARD problem (missing/mismatched
    committed artifact, bad schema, non-``https`` ``pdf_hf_url``) raises
    ``SystemExit``; a soft WARN (``pdf_hf_url`` null — a local-only build) is
    tolerated when ``allow_paper_warn`` is True (default), which lets a paper be
    marked a clean-result before the HF upload lands. Clearing (``value=False``)
    and every non-paper task skip the manifest gate.

    A **v2 report** body (carries ``REPORT_V1_SENTINEL`` — see
    :func:`is_report_body`) is a valid non-paper clean-result form: it is not a
    ``paper: true`` task and carries no ``paper_manifest.json``, so it flows
    through the non-paper path here and flips ``has_clean_result`` with no extra
    gate. Its own mechanical gate is ``scripts/verify_report.py`` (run before
    ``set-clean-result`` / at promote time), the report-track analogue of
    ``verify_task_body.py`` / ``verify_paper.py``.
    """
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        # Gate gap (#657): a task whose paper artifacts exist on disk
        # (docs/papers/issue_<N>/paper_manifest.json) is INTENDED to be a
        # paper-task, but if ``set_body`` dropped its ``paper: true`` key the
        # on-disk frontmatter reads as a plain markdown task and the paper
        # manifest gate below is silently skipped — the dashboard then renders
        # the markdown stub instead of the paper. Detect that mismatch up front
        # and FAIL loudly rather than passing a non-conforming paper-stub
        # through. (The fix in ``set_body`` prevents the drop; this is the gate
        # that catches a body written by an older code path or by hand.)
        if value and not is_paper_task(fm) and _paper_manifest_exists(task_id):
            raise SystemExit(
                f"set-clean-result #{task_id}: docs/papers/issue_{task_id}/"
                "paper_manifest.json exists (this is a paper-task) but the on-disk "
                "body.md frontmatter is MISSING `paper: true`. The dashboard will "
                "render the markdown stub instead of the paper. Re-write the body via "
                "`task.py set-body` from a stub carrying `paper: true` (the key now "
                "round-trips) before flipping has_clean_result."
            )
        if value and is_paper_task(fm):
            problems = validate_paper_manifest(task_id)
            hard = [p for p in problems if not p.startswith("WARN:")]
            warns = [p for p in problems if p.startswith("WARN:")]
            if hard or (warns and not allow_paper_warn):
                blocking = hard + (warns if not allow_paper_warn else [])
                raise SystemExit(
                    f"set-clean-result #{task_id}: paper manifest validation failed:\n  - "
                    + "\n  - ".join(blocking)
                    + f"\nFix docs/papers/issue_{task_id}/paper_manifest.json "
                    "(run scripts/build_paper.py / verify_paper.py) before flipping "
                    "has_clean_result."
                )
        fm["has_clean_result"] = value
        _write_body(path, fm, body)
        reg = _load_registry()
        _registry_set(reg, task_id, path.parent, fm)
        _save_registry(reg)
        _git_commit([path, registry_path()], f"task #{task_id}: has_clean_result={value}")


# ─── Goal of the experiment (canonical target) ────────────────────────────


GOAL_H2_NAME = "## Goal"


def _normalize_trailing_newline(text: str) -> str:
    """Normalize a body string to end with exactly one ``\\n``."""
    return text.rstrip("\n") + "\n"


def _inject_or_replace_goal_h2(body: str, new_goal: str) -> str:
    """Ensure body.md carries ``## Goal\\n\\n<new_goal>\\n`` between H1 and
    any other H2.

    The Goal section is defined as: the ``## Goal`` heading, one blank
    line, exactly one paragraph (the Goal sentence), and a terminating
    blank line. The section ends at the FIRST blank line after the
    sentence — anything after that blank line is preserved verbatim.

    Rules:
    - If a ``## Goal`` H2 already exists, REPLACE just its single-paragraph
      body (the lines between the heading-blank-line and the next blank
      line) with ``<new_goal>``. Everything below the trailing blank line
      is preserved.
    - Else if an H1 exists, insert ``\\n## Goal\\n\\n<new_goal>\\n``
      after the H1 line (and any single blank line immediately following
      the H1).
    - Else (no H1) prepend ``## Goal\\n\\n<new_goal>\\n\\n`` at the top.

    The function is text-only — the caller is responsible for the flock +
    git commit. Output is always normalized to end with exactly one
    ``\\n`` so idempotent re-applications produce byte-identical bodies.
    """
    body = _normalize_trailing_newline(body)
    lines = body.splitlines(keepends=False)
    # 1. Find an existing `## Goal` H2.
    goal_idx = None
    for i, line in enumerate(lines):
        if line.strip() == GOAL_H2_NAME:
            goal_idx = i
            break
    if goal_idx is not None:
        # Locate the start of the paragraph (skip any blank lines between
        # the heading and the goal sentence).
        para_start = goal_idx + 1
        while para_start < len(lines) and lines[para_start].strip() == "":
            para_start += 1
        # Locate the end of the paragraph (first blank line OR next H2
        # OR EOF — whichever comes first). The next H2 case handles the
        # pathological "## Goal\n## Other" no-content case.
        para_end = para_start
        while para_end < len(lines):
            stripped = lines[para_end].strip()
            if stripped == "":
                break
            if lines[para_end].startswith("## "):
                # We accidentally walked into the next section's H2 —
                # treat para_end as the section boundary (the existing
                # Goal section had no paragraph content).
                break
            para_end += 1
        # Replacement: heading + blank + new sentence + blank (the
        # terminating blank is preserved if the body had one; if we ran
        # to EOF / next-H2 without a blank, we still emit one for
        # readability).
        replacement = [GOAL_H2_NAME, "", new_goal]
        new_lines = lines[:goal_idx] + replacement + lines[para_end:]
        rebuilt = "\n".join(new_lines)
        return _normalize_trailing_newline(rebuilt)
    # 2. No existing Goal. Find H1.
    h1_idx = None
    for i, line in enumerate(lines):
        if line.startswith("# ") and not line.startswith("## "):
            h1_idx = i
            break
    if h1_idx is not None:
        insert_at = h1_idx + 1
        # Skip a single blank line after the H1 if present so the inserted
        # block sits flush below the title with consistent spacing. If we
        # did consume a blank line, the H2 goes directly at `insert_at`
        # (no leading blank in `block`); otherwise prepend a blank.
        consumed_blank = False
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
            consumed_blank = True
        block = [GOAL_H2_NAME, "", new_goal, ""]
        if not consumed_blank:
            block = ["", *block]
        new_lines = lines[:insert_at] + block + lines[insert_at:]
        rebuilt = "\n".join(new_lines)
        return _normalize_trailing_newline(rebuilt)
    # 3. No H1; prepend.
    block = [GOAL_H2_NAME, "", new_goal, "", ""]
    new_lines = block + lines
    rebuilt = "\n".join(new_lines)
    return _normalize_trailing_newline(rebuilt)


def set_goal(task_id: int, new_goal: str, *, by: str = "user", reason: str | None = None) -> bool:
    """Set / refine the canonical Goal-of-the-experiment for a task.

    Updates body.md frontmatter (`goal:`) AND ensures a `## Goal` H2 block
    is present in the body. Emits an `epm:goal-updated v1` marker carrying
    ``from: <old>``, ``to: <new>``, ``by: <agent>``, and optional
    ``reason:``. Idempotent: if the new value equals the existing value
    (and the H2 block is already in place), no marker is emitted and no
    commit is created.

    Parameters
    ----------
    task_id : int
        Task number.
    new_goal : str
        One-sentence Goal. Internal whitespace (newlines, tabs, runs of
        spaces) is collapsed to single spaces so multi-paragraph or
        otherwise multi-line input cannot corrupt either the frontmatter
        scalar or the `## Goal` H2 body block. Empty after normalization
        refuses.
    by : str
        Which agent is making the change. Valid values: ``user``,
        ``clarifier``, ``planner``. The orchestrator should set this
        based on which gate fired.
    reason : str, optional
        Free-form rationale; included verbatim in the marker note.

    Returns
    -------
    bool
        True if the Goal was changed, False if the call was a no-op.
    """
    # Normalize ALL whitespace, not just edges. A multi-line `new_goal`
    # would otherwise (a) become a multi-line YAML scalar in frontmatter
    # and (b) produce a multi-paragraph block under `## Goal`, which
    # `_inject_or_replace_goal_h2` only refreshes the first paragraph of,
    # leaving stale text orphaned in the body on the next refinement.
    goal = " ".join((new_goal or "").split())
    if not goal:
        raise ValueError("goal must be a non-empty one-sentence string")
    if by not in ("user", "clarifier", "planner"):
        raise ValueError(f"by must be one of user|clarifier|planner, got {by!r}")
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        old_goal = (fm.get("goal") or "").strip() or None
        # Normalize the pre-existing body's trailing whitespace BEFORE
        # comparing — `_inject_or_replace_goal_h2` always returns a body
        # with exactly one trailing `\n`, so trailing-whitespace drift
        # from prior writes is not a real change.
        body_normalized = _normalize_trailing_newline(body)
        new_body = _inject_or_replace_goal_h2(body, goal)
        # Idempotence: if the frontmatter goal is already equal AND the
        # body H2 block is already textually identical, do nothing.
        if old_goal == goal and new_body == body_normalized:
            return False
        fm["goal"] = goal
        _write_body(path, fm, new_body)
        # Update REGISTRY snapshot (carries `goal`).
        reg = _load_registry()
        _registry_set(reg, task_id, path.parent, fm)
        _save_registry(reg)
        # Emit marker. Note text mirrors the structured payload for easy
        # CLI scanning; the JSON fields are also present for tooling.
        note_parts = [
            f"from: {old_goal!r}",
            f"to: {goal!r}",
            f"by: {by}",
        ]
        if reason:
            note_parts.append(f"reason: {reason}")
        note = "\n".join(note_parts)
        ev_path = path.parent / "events.jsonl"
        payload: dict[str, Any] = {
            "ts": _utcnow_iso(),
            "kind": "epm:goal-updated",
            "version": 1,
            "by": by,
            "from": old_goal,
            "to": goal,
            "note": note,
        }
        if reason:
            payload["reason"] = reason
        _append_jsonl_line(ev_path, payload)
        _git_commit(
            [path, ev_path, registry_path()],
            f"task #{task_id}: set-goal — {goal[:60]}",
        )
    return True


def get_goal(task_id: int) -> str | None:
    """Return the task's canonical Goal (frontmatter `goal:`), or None."""
    fm, _ = _read_body(find_task_path(task_id) / "body.md")
    goal = fm.get("goal")
    return goal if isinstance(goal, str) and goal.strip() else None


# ─── Living-docs link (relates_to) ─────────────────────────────────────────
#
# `relates_to` is an OPTIONAL task-frontmatter field: a flat list of stable
# open-question ids (strings, e.g. ``["a1", "d2"]``) that the experiment
# bears on. There is NO primary/secondary distinction — it is a flat list.
# Default is ``[]`` (absent). The field is part of the living-docs ⇄ /issue
# integration (docs/living-docs-workflow-integration-plan.md): it makes the
# experiment→question mapping explicit and checkable, and
# `scripts/living_docs.py link()` writes it (paired with adding the task to
# each question's evidence list).
#
# Frontmatter is permissive (freeform YAML round-tripped by
# `_split_frontmatter` / `_join_frontmatter`; no key whitelist), so storing
# `relates_to` requires no validation change — `living_docs.py` writes it
# directly through `set_body`-style read/mutate/write. This read accessor is
# the companion getter, mirroring `get_goal`.


def get_relates_to(task_id: int) -> list[str]:
    """Return the task's flat `relates_to` open-question ids, or ``[]``.

    `relates_to` is an optional frontmatter field: a flat list of stable
    open-question id strings the experiment bears on (no primary/secondary).
    Always returns a list (empty when the field is absent, ``null``, or not
    a list). Non-string entries are dropped so callers can iterate safely.
    """
    fm, _ = _read_body(find_task_path(task_id) / "body.md")
    value = fm.get("relates_to")
    if not isinstance(value, list):
        return []
    return [str(q).strip() for q in value if isinstance(q, str) and str(q).strip()]


def add_tag(task_id: int, tag: str) -> None:
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        tags: list[str] = list(fm.get("tags") or [])
        if tag in tags:
            return
        tags.append(tag)
        fm["tags"] = tags
        _write_body(path, fm, body)
        _git_commit([path], f"task #{task_id}: add-tag {tag}")


def remove_tag(task_id: int, tag: str) -> None:
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        tags: list[str] = list(fm.get("tags") or [])
        if tag not in tags:
            return
        tags.remove(tag)
        fm["tags"] = tags
        _write_body(path, fm, body)
        _git_commit([path], f"task #{task_id}: remove-tag {tag}")


def set_track(task_id: int, track: str) -> None:
    """Set the task's `track` frontmatter field.

    `track` is the agent-vs-human categorization read by the dashboard
    kanban: ``experiment`` (an agent can run it end-to-end) or ``human``
    (think-about / read / decide — needs the user). Frontmatter is a plain
    dict round-tripped through yaml, so the new key persists across other
    mutations. The dashboard's `/api/tasks/track` shells this; the CLI
    also exposes ``task.py set-track`` + ``task.py new --track``.
    """
    if track not in ("experiment", "human"):
        raise ValueError(f"track must be 'experiment' or 'human', got {track!r}")
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
        fm["track"] = track
        _write_body(path, fm, body)
        reg = _load_registry()
        _registry_set(reg, task_id, path.parent, fm)
        _save_registry(reg)
        _git_commit([path, registry_path()], f"task #{task_id}: set-track {track}")


# ─── Plans ──────────────────────────────────────────────────────────────────


def new_plan_version(task_id: int, plan_md: str) -> int:
    """Append plans/v{next}.md, update plans/plan.md symlink. Returns the
    new version number.

    The next version number is derived as ``max(existing v<N>) + 1`` (NOT
    ``len(existing) + 1``) so that gaps in the plan-version sequence — e.g.
    a v5 draft that lived only in /tmp and was never registered, leaving
    plans/ as ``v1,v2,v3,v4,v6`` — cannot cause the next write to silently
    overwrite the highest existing plan. The plans/v{N}.md scheme exists
    to preserve the full audit trail of plan revisions; this resolver is
    the single canonical writer and must never lose history. As a
    belt-and-suspenders guard, refuse loudly if the computed target file
    somehow already exists (e.g. a concurrent writer between the glob and
    the write, or a manually pre-staged file).
    """
    with _locked():
        plans_dir = find_task_path(task_id) / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        existing_nums = [
            int(m.group(1))
            for p in plans_dir.glob("v*.md")
            if (m := re.fullmatch(r"v(\d+)\.md", p.name))
        ]
        next_v = (max(existing_nums) + 1) if existing_nums else 1
        target = plans_dir / f"v{next_v}.md"
        if target.exists():
            raise RuntimeError(
                f"refusing to overwrite existing plan file {target} "
                f"(existing versions: {sorted(existing_nums)}); "
                f"the highest-version+1 resolver computed v{next_v} but "
                f"that file already exists on disk"
            )
        target.write_text(plan_md if plan_md.endswith("\n") else plan_md + "\n")
        # Symlink plan.md → v{next}.md
        symlink = plans_dir / "plan.md"
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()
        symlink.symlink_to(target.name)
        _git_commit([target, symlink], f"task #{task_id}: plan v{next_v}")
    return next_v


# ─── Promotion ──────────────────────────────────────────────────────────────


def promote(task_id: int, verdict: str) -> Path:
    """User-only: flip a task at awaiting_promotion → completed, record the
    classification in frontmatter, append epm:promoted.
    """
    if verdict not in ("useful", "not-useful"):
        raise ValueError(f"verdict must be useful|not-useful, got {verdict!r}")
    with _locked():
        path = find_task_path(task_id)
        cur_status = _status_from_path(path)
        if cur_status != PARK_STATUS:
            raise RuntimeError(
                f"task #{task_id} is in status {cur_status!r}, expected {PARK_STATUS!r}; "
                f"refusing to promote"
            )
        fm, body = _read_body(path / "body.md")
        fm["classification"] = verdict
        fm["promoted_at"] = _utcnow_iso()
        _write_body(path / "body.md", fm, body)
        # Append event
        promoted_event = {
            "ts": _utcnow_iso(),
            "kind": "epm:promoted",
            "version": 1,
            "by": "user",
            "classification": verdict,
        }
        _append_jsonl_line(path / "events.jsonl", promoted_event)
        _git_commit(
            [path / "body.md", path / "events.jsonl"], f"task #{task_id}: promote {verdict}"
        )
    # Then move to completed via set_status (own lock + commit)
    return set_status(task_id, "completed", note=f"promoted as {verdict}")


# ─── Queries ────────────────────────────────────────────────────────────────


def list_by_status(status: str, limit: int = 200) -> list[dict[str, Any]]:
    """List tasks in tasks/<status>/. Returns a list of registry-style dicts."""
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status!r}")
    folder = tasks_dir() / status
    if not folder.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(folder.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not child.is_dir() or not child.name.isdigit():
            continue
        task_id = int(child.name)
        try:
            fm, _ = _read_body(child / "body.md")
        except (FileNotFoundError, ValueError):
            continue
        out.append(
            {
                "id": task_id,
                "title": fm.get("title", ""),
                "kind": fm.get("kind", "experiment"),
                "tags": fm.get("tags") or [],
                "status": status,
                "has_clean_result": bool(fm.get("has_clean_result", False)),
            }
        )
        if len(out) >= limit:
            break
    return out


def list_children(parent_id: int) -> list[dict[str, Any]]:
    """List tasks whose frontmatter ``parent_id`` equals ``parent_id``.

    Walks REGISTRY entries and reads each task's frontmatter (the registry
    does not denormalize ``parent_id``, so the body read is authoritative).
    Returns registry-style dicts — ``id`` / ``status`` / ``title`` / ``kind``
    / ``has_clean_result`` — sorted by id. Unreadable rows are skipped
    (same fail-soft posture as :func:`list_by_status`: a single corrupt
    body must not hide every sibling). Primary consumer: the ``/campaign``
    runner's reconcile step (task #586)."""
    reg = _load_registry()
    repo = repo_root()
    out: list[dict[str, Any]] = []
    for tid_str, entry in reg.get("tasks", {}).items():
        try:
            task_id = int(tid_str)
        except (TypeError, ValueError):
            continue
        path = repo / entry["path"]
        try:
            fm, _ = _read_body(path / "body.md")
        except (FileNotFoundError, ValueError):
            continue
        if fm.get("parent_id") != parent_id:
            continue
        out.append(
            {
                "id": task_id,
                "status": _status_from_path(path),
                "title": fm.get("title", ""),
                "kind": fm.get("kind", "experiment"),
                "has_clean_result": bool(fm.get("has_clean_result", False)),
            }
        )
    out.sort(key=lambda row: row["id"])
    return out


def audit() -> list[str]:
    """Validate REGISTRY.json against the filesystem. Returns a list of
    human-readable problems; empty list = clean.
    """
    problems: list[str] = []
    reg = _load_registry()
    repo = repo_root()
    td = tasks_dir()
    # 1. Every registry entry's path exists.
    for tid, entry in reg.get("tasks", {}).items():
        abs_path = repo / entry["path"]
        if not abs_path.is_dir():
            problems.append(f"task #{tid}: registry path {entry['path']!r} does not exist")
            continue
        body = abs_path / "body.md"
        if not body.exists():
            problems.append(f"task #{tid}: missing body.md at {entry['path']}")
    # 2. Every on-disk task folder is in the registry.
    if td.exists():
        for status_dir in td.iterdir():
            if not status_dir.is_dir() or status_dir.name not in STATUSES:
                continue
            for child in status_dir.iterdir():
                if not child.is_dir() or not child.name.isdigit():
                    continue
                tid = child.name
                if tid not in reg.get("tasks", {}):
                    problems.append(
                        f"task #{tid}: on disk at {child.relative_to(repo)} but not in registry"
                    )
    # 3. highest_id sanity.
    if reg.get("tasks"):
        max_disk = max(int(t) for t in reg["tasks"])
        if max_disk > reg.get("highest_id", 0):
            problems.append(f"highest_id {reg.get('highest_id', 0)} < max task id {max_disk}")
    return problems


# ─── Registry reconcile (`task.py audit --repair`) ──────────────────────────


@dataclass(frozen=True)
class RegistryChange:
    """One drift entry surfaced by :func:`reconcile_registry`.

    ``drift_class`` is one of ``"stale_real" | "missing_real" | "empty_stub"
    | "skipped" | "highest_id"``. ``detail`` is a human-readable description
    (with the relevant on-disk path) for the CLI report and the test
    assertions. ``task_id`` is the integer id (``-1`` for the ``highest_id``
    bump, which is not tied to a single task).
    """

    task_id: int
    drift_class: str
    detail: str


# Internal planning record: a reconcilable drift whose registry entry the
# apply path will (re-)write from ``actual / "body.md"``. ``registry_path`` is
# the stale REGISTRY ``path`` string for a ``stale_real`` entry, or ``None``
# for a ``missing_real`` (no registry entry yet). ``actual`` is the task's
# REAL on-disk folder (always under tasks_dir()), so the value passed to
# ``_registry_set`` resolves ``_status_from_path`` correctly.
@dataclass(frozen=True)
class _PendingReconcile:
    task_id: int
    drift_class: str  # "stale_real" | "missing_real"
    registry_path: str | None
    actual: Path


@dataclass(frozen=True)
class ReconcileReport:
    """Result of :func:`reconcile_registry`.

    Drift is split into FOUR classes. ``stale_real`` + ``missing_real`` are the
    RECONCILABLE classes (re-pointed / added, re-snapshotted from the task's
    actual on-disk ``body.md``); on the apply path these lists hold only the
    entries that were ACTUALLY written. ``empty_stubs`` (a task dir with no
    ``body.md``) and ``skipped`` (a task that should reconcile but whose body
    is unreadable / its folder is genuinely gone) are NEVER written to the
    registry — surfaced for manual triage, never fabricated, never deleted.
    """

    applied: bool
    stale_real: list[RegistryChange]
    missing_real: list[RegistryChange]
    empty_stubs: list[RegistryChange]
    skipped: list[RegistryChange]
    highest_id_bumped: RegistryChange | None

    @property
    def reconciled_count(self) -> int:
        """Number of registry mutations: re-pointed + added + the highest_id bump."""
        return len(self.stale_real) + len(self.missing_real) + (1 if self.highest_id_bumped else 0)

    @property
    def unresolved_count(self) -> int:
        """Drift the reconcile cannot fix on its own — empty stubs + skips.

        The CLI exits 1 when this is non-zero (after an apply): the registry
        still disagrees with the filesystem and an operator must triage.
        """
        return len(self.empty_stubs) + len(self.skipped)

    @property
    def is_clean(self) -> bool:
        """True iff there was nothing to reconcile AND no unresolved drift."""
        return self.reconciled_count == 0 and self.unresolved_count == 0


def _reconcile_change(task_id: int | str, drift_class: str, detail: str) -> RegistryChange:
    return RegistryChange(task_id=int(task_id), drift_class=drift_class, detail=detail)


def _reconcile_commit_msg(
    stale: list[_PendingReconcile],
    missing: list[_PendingReconcile],
    bumped: RegistryChange | None,
) -> str:
    parts = [f"registry-reconcile: {len(stale)} path(s) fixed, {len(missing)} entry(ies) added"]
    if missing:
        ids = ", ".join(f"#{p.task_id}" for p in missing)
        parts.append(f"added {ids}")
    if bumped:
        parts.append(bumped.detail)
    return "; ".join(parts)


def _reconcile_pending_change(p: _PendingReconcile, repo: Path) -> RegistryChange:
    rel = str(p.actual.relative_to(repo))
    if p.drift_class == "stale_real":
        detail = f"{p.registry_path} -> {rel} (re-pointed + re-snapshotted)"
    else:
        detail = f"added at {rel} (re-snapshotted)"
    return _reconcile_change(p.task_id, p.drift_class, detail)


def _reconcile_highest_id(reg: dict[str, Any], max_disk_id: int = 0) -> None:
    """Final sanity pass: bump highest_id to ``max(max registered id, max ON-DISK
    id)`` if it drifted low. (A ``missing_real`` add already bumps it inside
    ``_registry_set``; this also catches a registry whose highest_id drifted
    below max another way.) ``max_disk_id`` covers task-dir ids that are NEVER
    written to ``reg["tasks"]`` — an ``empty_stub`` (a bodyless dir) is surfaced
    but never fabricated into the registry, so its id would otherwise be invisible
    here; leaving ``highest_id`` below such a stub id lets a later ``create_task``
    re-allocate (and collide with) it. Mutates ``reg`` in place."""
    ids = [int(t) for t in reg.get("tasks", {})]
    target = max([*ids, max_disk_id], default=0)
    if target > reg.get("highest_id", 0):
        reg["highest_id"] = target


def _reconcile_bump_change(before: int, after: int) -> RegistryChange | None:
    """A ``highest_id`` RegistryChange iff it net-increased across the reconcile
    — regardless of whether ``_registry_set`` or ``_reconcile_highest_id`` did it."""
    if after > before:
        return _reconcile_change(-1, "highest_id", f"highest_id {before}->{after}")
    return None


def _reconcile_scan_disk(td: Path) -> dict[str, Path]:
    """Map ``str(task_id) -> actual on-disk Path`` for every task dir under a
    valid ``STATUSES`` folder. Scanned once per reconcile."""
    disk: dict[str, Path] = {}
    if not td.exists():
        return disk
    for status_dir in td.iterdir():
        if not status_dir.is_dir() or status_dir.name not in STATUSES:
            continue
        for child in status_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                disk[child.name] = child
    return disk


def _reconcile_plan(
    repo: Path, td: Path, reg: dict[str, Any]
) -> tuple[
    list[_PendingReconcile],
    list[_PendingReconcile],
    list[RegistryChange],
    list[RegistryChange],
    dict[str, Path],
]:
    """Classify every drift between ``reg`` and the on-disk task tree into the
    four classes (stale_real / missing_real / empty_stub / skipped). Pure read;
    never mutates ``reg`` or the filesystem. See :func:`reconcile_registry` for
    the class definitions. Also returns the ``str(task_id) -> Path`` ``disk``
    map it scans, so the caller can fold the max ON-DISK id (incl. empty-stub
    ids never written to the registry) into the ``highest_id`` bump without a
    second scan."""

    def _has_body(p: Path) -> bool:
        # Cheap EXISTENCE check (classification, not a parse). Keeps an empty
        # stub away from `_read_body`, so a missing body.md never raises
        # FileNotFoundError and crashes the whole run.
        return (p / "body.md").exists()

    def _rel(p: Path) -> str:
        return str(p.relative_to(repo))

    disk = _reconcile_scan_disk(td)
    stale_real: list[_PendingReconcile] = []
    missing_real: list[_PendingReconcile] = []
    empty_stubs: list[RegistryChange] = []
    skipped: list[RegistryChange] = []
    tasks = reg.get("tasks", {})

    # Class 1 — registry entry whose path is missing (or present-but-bodyless).
    for tid, entry in tasks.items():
        abs_path = repo / entry["path"]
        if abs_path.is_dir():
            # 1b — registered, dir exists, but body.md is gone -> empty stub
            # (the audit() :2446-2448 sub-check, folded into empty_stubs).
            if not _has_body(abs_path):
                empty_stubs.append(
                    _reconcile_change(
                        tid,
                        "empty_stub",
                        f"registered at {entry['path']}, dir exists but body.md missing",
                    )
                )
            continue  # registry path fine (or 1b handled) — not stale.
        actual = disk.get(tid)
        if actual is None:
            # Registry path stale AND task genuinely gone from disk. Leave the
            # entry untouched (dropping it would lose a real entry).
            skipped.append(
                _reconcile_change(
                    tid,
                    "skipped",
                    f"registry path {entry['path']!r} missing AND no on-disk folder found",
                )
            )
            continue
        if not _has_body(actual):
            empty_stubs.append(
                _reconcile_change(
                    tid,
                    "empty_stub",
                    f"registry path {entry['path']!r} missing; on-disk dir "
                    f"{_rel(actual)} has no body.md",
                )
            )
            continue
        stale_real.append(_PendingReconcile(int(tid), "stale_real", entry["path"], actual))

    # Class 2 — on disk but unregistered.
    for tid, actual in disk.items():
        if tid in tasks:
            continue
        if not _has_body(actual):
            empty_stubs.append(
                _reconcile_change(
                    tid,
                    "empty_stub",
                    f"on disk at {_rel(actual)} but no body.md "
                    f"(likely unmerged issue-{tid} branch residue)",
                )
            )
            continue
        missing_real.append(_PendingReconcile(int(tid), "missing_real", None, actual))

    return stale_real, missing_real, empty_stubs, skipped, disk


def _reconcile_apply_pending(
    reg: dict[str, Any],
    pending: list[_PendingReconcile],
    skipped: list[RegistryChange],
) -> list[_PendingReconcile]:
    """Re-snapshot each pending reconcile into ``reg`` from its actual on-disk
    ``body.md``. A body that passed the plan-time ``exists()`` check but is now
    unreadable (``FileNotFoundError`` if it vanished, ``ValueError`` for
    malformed YAML) is appended to ``skipped`` and left out of the registry —
    never fabricated, never aborting the whole run. Returns the entries that
    were ACTUALLY written. Mutates ``reg`` + ``skipped`` in place."""
    applied: list[_PendingReconcile] = []
    for pend in pending:
        try:
            fm, _ = _read_body(pend.actual / "body.md")
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(_reconcile_change(pend.task_id, "skipped", f"body.md unreadable: {exc}"))
            continue
        _registry_set(reg, pend.task_id, pend.actual, fm)
        applied.append(pend)
    return applied


def reconcile_registry(*, apply: bool = False) -> ReconcileReport:
    """Reconcile REGISTRY.json against the on-disk task tree.

    Mirrors :func:`audit`'s detection so the two never disagree about WHAT
    counts as drift, then classifies each drift by whether the task has a
    readable ``body.md``:

    - **stale_real** — a registry entry whose ``path`` is missing, but the task
      lives elsewhere on disk WITH a ``body.md``: re-point + re-snapshot.
    - **missing_real** — an on-disk task dir WITH a ``body.md`` absent from the
      registry: add the entry + re-snapshot (this is the live #703 case).
    - **empty_stub** — a task dir with NO ``body.md`` (registered-but-bodyless,
      or on-disk-and-unregistered-and-bodyless). Surfaced, NEVER reconciled,
      NEVER fabricated into the registry, NEVER deleted (the dir may be live
      residue of an active ``issue-<N>`` branch). **Policy:** any task dir
      lacking ``body.md``, registered or not, is an empty stub.
    - **skipped** — a registry entry whose path is missing AND no on-disk
      folder is found anywhere (its existing entry is left untouched, not
      dropped — erasing a real entry would lose data), OR a reconcilable task
      whose ``body.md`` exists but is unreadable (malformed YAML / vanished
      between the existence check and the read).

    Writes ONLY ``tasks/REGISTRY.json`` — never moves a task folder, changes a
    status, or touches ``body.md`` / any task content. Pure read in dry-run
    (``apply=False``); under ``_locked()`` (re-read + re-plan inside the lock)
    on the apply path. Idempotent: a second apply on a now-consistent registry
    produces zero diff and no commit.
    """
    repo = repo_root()
    td = tasks_dir()

    if not apply:
        reg = _load_registry()
        stale, missing, empty_stubs, skipped, disk = _reconcile_plan(repo, td, reg)
        # Dry-run preview of the net highest_id bump (computed on a copy — no
        # write). A missing_real adds an id; the bump is max(registered ids,
        # ON-DISK ids) vs the current highest_id. The disk max covers empty-stub
        # ids that are never written to the registry (so they cannot be found by
        # scanning reg["tasks"]) yet still constrain the next free id.
        before = reg.get("highest_id", 0)
        max_disk_id = max((int(t) for t in disk), default=0)
        preview: dict[str, Any] = {"highest_id": before, "tasks": dict(reg.get("tasks", {}))}
        for p in stale + missing:
            preview["tasks"][str(p.task_id)] = {"path": str(p.actual.relative_to(repo))}
        _reconcile_highest_id(preview, max_disk_id)
        return ReconcileReport(
            applied=False,
            stale_real=[_reconcile_pending_change(p, repo) for p in stale],
            missing_real=[_reconcile_pending_change(p, repo) for p in missing],
            empty_stubs=empty_stubs,
            skipped=skipped,
            highest_id_bumped=_reconcile_bump_change(before, preview["highest_id"]),
        )

    with _locked():
        # Re-read + re-plan INSIDE the lock: a concurrent writer (e.g. a live
        # issue-<N> branch merging body.md to main) may have changed the
        # registry since any dry-run, so a stale task simply reclassifies.
        reg = _load_registry()
        highest_before = reg.get("highest_id", 0)
        stale, missing, empty_stubs, skipped, disk = _reconcile_plan(repo, td, reg)
        max_disk_id = max((int(t) for t in disk), default=0)
        applied_stale = _reconcile_apply_pending(reg, stale, skipped)
        applied_missing = _reconcile_apply_pending(reg, missing, skipped)
        _reconcile_highest_id(reg, max_disk_id)
        bumped = _reconcile_bump_change(highest_before, reg.get("highest_id", 0))
        if applied_stale or applied_missing or bumped:
            _save_registry(reg)
            _git_commit(
                [registry_path()],
                _reconcile_commit_msg(applied_stale, applied_missing, bumped),
            )
        return ReconcileReport(
            applied=True,
            stale_real=[_reconcile_pending_change(p, repo) for p in applied_stale],
            missing_real=[_reconcile_pending_change(p, repo) for p in applied_missing],
            empty_stubs=empty_stubs,
            skipped=skipped,
            highest_id_bumped=bumped,
        )


# ─── Comments ──────────────────────────────────────────────────────────────


def append_comment(
    task_id: int,
    *,
    author: str,
    kind: str,
    body: str,
    in_reply_to: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a comment to tasks/<status>/<id>/comments.jsonl. Used by both
    the local tunnel handler (for Claude answers) and tests.

    The web app writes comments directly via Octokit; this helper is here
    so any local code path (tunnel, tests, future CLI) uses the same shape.
    """
    if kind not in COMMENT_KINDS:
        raise ValueError(f"unknown comment kind: {kind!r}; expected one of {sorted(COMMENT_KINDS)}")
    with _locked():
        path = find_task_path(task_id) / "comments.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Allocate a sequential id (c001, c002, ...) by counting lines.
        n_existing = sum(1 for _ in path.open()) if path.exists() else 0
        cid = f"c{n_existing + 1:03d}"
        record: dict[str, Any] = {
            "id": cid,
            "ts": _utcnow_iso(),
            "author": author,
            "kind": kind,
            "body": body,
        }
        if in_reply_to:
            record["in_reply_to"] = in_reply_to
        if extras:
            record.update(extras)
        _append_jsonl_line(path, record)
        _git_commit([path], f"task #{task_id}: comment {cid} ({kind})")
    return record


def list_comments(task_id: int) -> list[dict[str, Any]]:
    path = find_task_path(task_id) / "comments.jsonl"
    return _iter_jsonl(path)


# ─── Git helpers ────────────────────────────────────────────────────────────


# Git lock-contention signature. Matches the three stderr shapes a concurrent
# git process produces (git 2.x): the index.lock path itself, the generic
# "another process" hint, and the File-exists lock-create failure (covers
# ref locks / packed-refs.lock with the same transient signature). A CAS
# mismatch ("is at <sha> but expected <sha>") deliberately does NOT match.
_GIT_LOCK_CONTENTION_RE = re.compile(
    r"index\.lock"
    r"|Another git process seems to be running"
    r"|Unable to create '.*\.lock': File exists"
)
_GIT_LOCK_RETRY_SLEEP_RANGE_S = (2.0, 3.0)  # one retry; jittered to de-sync


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` at the repo root, retrying ONCE on lock contention.

    Contention/crash envelope (#825): the command runs with ``check=False``
    internally; if it exits non-zero AND stderr matches the git
    lock-contention signature (``_GIT_LOCK_CONTENTION_RE`` — a concurrent git
    process holds ``.git/index.lock`` or a sibling ``*.lock``), sleep a
    jittered ``random.uniform(*_GIT_LOCK_RETRY_SLEEP_RANGE_S)`` interval and
    rerun exactly ONCE (never more). The retry keys on the STDERR SIGNATURE,
    never on the return code, so ``check=False`` rc-as-signal call sites
    (``diff --cached --quiet``) keep their rc semantics with zero retries,
    and non-lock failures surface immediately. A SUCCESSFUL call takes no
    sleep (zero happy-path latency). If the retry also fails on the lock
    signature, a stale-lock remedy is logged at ERROR. After the (at most
    one) retry the caller's ``check`` semantics apply: ``check=True`` raises
    ``subprocess.CalledProcessError`` with the same ``cmd``/``output``/
    ``stderr`` fields ``subprocess.run(check=True)`` would produce.
    """

    def _attempt() -> subprocess.CompletedProcess[str]:
        # Resolve cwd PER CALL (not from a cached module-level REPO). The
        # process-local LRU cache in `repo_root()` makes this cheap, and
        # per-call resolution is what keeps long-lived processes (PM session,
        # agent daemons) safe across `os.chdir()` or branch state changes.
        #
        # `env=_sanitized_git_env()` matches the resolver: inherited GIT_DIR /
        # GIT_WORK_TREE / GIT_INDEX_FILE / GIT_OBJECT_DIRECTORY would in
        # principle redirect git add/commit. The resolver already strips them
        # for the subprocess that locates the repo root; strip them here too
        # for parity (round-1 code-review finding #7).
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
            env=_sanitized_git_env(),
            check=False,
            capture_output=True,
            text=True,
        )

    result = _attempt()
    if result.returncode != 0 and _GIT_LOCK_CONTENTION_RE.search(result.stderr or ""):
        delay = random.uniform(*_GIT_LOCK_RETRY_SLEEP_RANGE_S)
        _log.warning(
            "git %s hit a lock collision (a concurrent git process holds the lock); "
            "retrying once in %.1fs",
            args[0] if args else "",
            delay,
        )
        time.sleep(delay)
        result = _attempt()  # second and FINAL attempt
        if result.returncode != 0 and _GIT_LOCK_CONTENTION_RE.search(result.stderr or ""):
            _log.error(
                "git %s failed twice on a lock collision. A concurrent git process is "
                "holding the repo lock; if no live git process exists, a crashed one "
                "may have left a stale .git/index.lock — inspect and remove it "
                "manually.",
                args[0] if args else "",
            )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result


def _git_commit(paths: list[Path], message: str) -> None:
    """Stage the given paths and create a single commit. Optional push.

    Uses ``git commit --only -- <paths>`` so unrelated staged work elsewhere in
    the repo is not silently captured under the task.py commit message. Parallel
    agents (/issue runs, user-staged edits) share the same
    index, and ``git commit -m`` without ``--only`` would commit the entire
    index. The early-return check is likewise narrowed to ``--`` <paths> so it
    cannot bail when unrelated files are staged.

    Paths that no longer exist on disk are tolerated: they are presumed to
    have been staged-for-deletion by a prior op in the same mutation (e.g.
    the source side of a ``git mv`` in ``set_status``). ``git add`` would
    refuse them, so the staging step skips them; ``commit --only`` then
    captures the existing staged deletion. Callers that move files MUST
    include BOTH the old and new paths in their ``paths`` list so the
    deletion side of the move is not orphaned in the index.

    When the primary checkout is parked on a feature branch, ``repo_root()``
    resolves to the managed main-pinned worktree (DETACHED at the `main` tip).
    In that routed case the commit lands on the detached HEAD, so afterwards
    this function compare-and-swaps the `main` branch ref forward to the new
    commit (``_advance_main_ref``). On the primary checkout (HEAD on `main`)
    this routed branch is never taken and behavior is byte-for-byte identical
    to before — the commit advances `main` directly via the normal HEAD move.

    Set TASK_PY_NO_COMMIT=1 to skip the commit entirely (useful in tests).
    Set TASK_PY_AUTO_PUSH=1 to also push after the commit.
    """
    if os.environ.get("TASK_PY_NO_COMMIT") == "1":
        return
    repo = repo_root()
    routed = _is_routed_root(repo)
    env = _sanitized_git_env()
    rel_paths = [str(p.relative_to(repo)) for p in paths]
    # Re-stage only paths that still exist on disk. Paths that vanished
    # (e.g. source of a `git mv`) are already in the index as deletions;
    # `git add` would error on them. `commit --only` below picks up the
    # existing staged deletion anyway.
    existing_rel_paths = [str(p.relative_to(repo)) for p in paths if p.exists()]
    if existing_rel_paths:
        _run_git(["add", "--", *existing_rel_paths])
    # Skip commit if nothing changed for OUR paths (e.g. idempotent re-runs).
    # Narrowed to rel_paths so unrelated staged work doesn't keep us going.
    result = _run_git(["diff", "--cached", "--quiet", "--", *rel_paths], check=False)
    if result.returncode == 0:
        return
    # When routed, capture the pre-commit tip (== `main`, since the resolver
    # reset the managed worktree to `main` and the flock prevents `main` from
    # moving inside this process) BEFORE committing, so we can CAS-advance
    # `main` to the new commit afterwards.
    old_sha = _run_git(["rev-parse", "HEAD"]).stdout.strip() if routed else ""
    full_msg = f"{message}\n\n[task.py]"
    _run_git(["commit", "-m", full_msg, "--only", "--", *rel_paths])
    if routed:
        new_sha = _run_git(["rev-parse", "HEAD"]).stdout.strip()
        _advance_main_ref(repo, old_sha, new_sha, env)
    if os.environ.get("TASK_PY_AUTO_PUSH") == "1":
        _run_git(["push"], check=False)


# ─── Binding concerns (concerns.jsonl) ─────────────────────────────────────
#
# Append-only sidecar at ``tasks/<status>/<N>/concerns.jsonl`` carrying
# review-loop concerns that persist across stages (code-reviewer, critic,
# interpretation-critic, clean-result-critic, consistency-checker). Schema:
#
#   {
#     "ts": "YYYY-MM-DDTHH:MM:SSZ",
#     "event": "raised | addressed | deferred | verified-open",
#     "concern_id": "<stable-kebab-case>",
#     "severity": "BLOCKER | CONCERN | NIT",
#     "summary": "<≤200-char one-line>",
#     "raised_by": "<agent-name>",
#     "raised_at_round": <int>,
#     "evidence": "<optional pointer / path / quote>",
#     "addressed_by": "<implementer | analyzer | ...>",   # on address / re-raise
#     "addressed_at_round": <int>,                         # on address / re-raise
#     "deferral_rationale": "<≥40-char user prose>",       # on defer only
#     "deferred_by": "user"                                # on defer; reconciler is special-cased
#   }
#
# `concerns.jsonl` follows the task on status-folder moves because it lives
# inside ``tasks/<status>/<N>/`` — `set_status`'s ``git mv`` of the task
# folder carries it along automatically.
#
# Every concerns.jsonl event is mirrored to events.jsonl as a thin
# ``epm:concern-{raised,addressed,deferred,verified-open}`` marker carrying
# concern_id + ≤80-char summary. The full event payload (severity, evidence,
# rationale) lives in concerns.jsonl; the mirror is just an audit-log
# breadcrumb so an events-only consumer can see something happened.

CONCERN_SEVERITIES = frozenset({"BLOCKER", "CONCERN", "NIT"})

CONCERN_EVENTS = frozenset({"raised", "addressed", "deferred", "verified-open"})

# Stable-kebab-case ID: lowercase letters / digits / hyphens, 2-80 chars,
# starts with a letter or digit. Examples that PASS:
#   probe-position-undefined, missing-mlm-control, n2-seeds-uninterpretable
# Examples that FAIL: trailing dash, leading dash, uppercase, underscore,
# spaces, single char, >80 chars.
_CONCERN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")

# Boilerplate user-deferral rationales we refuse — defense against
# "rubber-stamp" deferrals. Compare casefold + collapsed whitespace.
# Updated piecemeal as new boilerplate variants show up in transcripts.
_CONCERN_RATIONALE_BOILERPLATE = frozenset(
    {
        "user accepted",
        "ok",
        "okay",
        "approved",
        "fine",
        "deferred",
        "user said ok",
        "user said okay",
        "user accepted as-is",
        "user accepted as is",
        "user is fine with it",
        "lgtm",
        "wontfix",
        "won't fix",
        "no action needed",
        "not blocking",
    }
)

_CONCERN_RATIONALE_MIN_CHARS = 40


def _concerns_path(task_id: int) -> Path:
    """Return absolute path of ``tasks/<status>/<N>/concerns.jsonl``."""
    return find_task_path(task_id) / "concerns.jsonl"


def _validate_concern_id(concern_id: str) -> None:
    """Raise ``ValueError`` if ``concern_id`` violates the kebab-case rule."""
    if not isinstance(concern_id, str) or not _CONCERN_ID_RE.match(concern_id):
        raise ValueError(
            f"concern_id {concern_id!r} must match {_CONCERN_ID_RE.pattern} "
            "(lowercase kebab-case, 2-80 chars, starts with letter or digit). "
            "Examples: 'probe-position-undefined', 'missing-mlm-control'."
        )


def _validate_deferral_rationale(rationale: str) -> None:
    """Raise ``ValueError`` if the deferral rationale is too short or
    matches a known boilerplate phrase (case-insensitive, whitespace-
    collapsed). The bar is intentionally low (40 chars) but rejects
    rubber-stamp phrasing."""
    if not isinstance(rationale, str):
        raise ValueError("deferral rationale must be a string")
    stripped = rationale.strip()
    if len(stripped) < _CONCERN_RATIONALE_MIN_CHARS:
        raise ValueError(
            f"deferral rationale must be ≥ {_CONCERN_RATIONALE_MIN_CHARS} "
            f"chars (got {len(stripped)}). Explain why the concern is "
            "being deferred — what the orchestrator tried, why it can't "
            "be addressed in this round, and what the downstream impact is."
        )
    normalized = " ".join(stripped.casefold().split())
    if normalized in _CONCERN_RATIONALE_BOILERPLATE:
        raise ValueError(
            f"deferral rationale {rationale!r} matches a known boilerplate "
            "phrase. Rubber-stamp deferrals defeat the purpose — write a "
            "substantive rationale naming the surviving risk."
        )


def list_concerns(task_id: int, *, open_only: bool = False) -> list[dict[str, Any]]:
    """Return the current concerns ledger for a task.

    By default returns the full event stream (every raise / address /
    defer / verified-open event ever appended). With ``open_only=True``,
    returns the LATEST event per concern_id and filters out concerns
    whose latest event is ``addressed`` or ``deferred`` — i.e. only
    rows currently OPEN against the task (latest event is ``raised`` or
    ``verified-open``).

    Result rows are dicts with the schema documented at the top of
    this section. Returns ``[]`` if the file does not exist.
    """
    path = _concerns_path(task_id)
    events = _iter_jsonl(path)
    if not open_only:
        return events
    latest: dict[str, dict[str, Any]] = {}
    for ev in events:
        cid = ev.get("concern_id")
        if cid is None:
            continue
        latest[cid] = ev
    open_events = [ev for ev in latest.values() if ev["event"] in ("raised", "verified-open")]
    open_events.sort(key=lambda e: e.get("ts", ""))
    return open_events


def _read_concerns_raw(task_id: int) -> list[dict[str, Any]]:
    """Internal: return ALL events, no filtering. Used by raise/address/
    defer to look up prior history of a concern_id (idempotency, severity
    lookups, re-raise → verified-open promotion)."""
    return list_concerns(task_id, open_only=False)


def _latest_event_for(events: list[dict[str, Any]], concern_id: str) -> dict[str, Any] | None:
    """Return the most recent event for ``concern_id`` from a pre-fetched
    list, or ``None`` if the concern has never been raised."""
    for ev in reversed(events):
        if ev.get("concern_id") == concern_id:
            return ev
    return None


def _append_concern_event(task_id: int, payload: dict[str, Any]) -> None:
    """Append ONE event to concerns.jsonl + mirror to events.jsonl + commit.

    Caller MUST hold ``_locked()``. Caller is responsible for constructing
    the payload (including ``ts``). The mirror event posted to
    events.jsonl carries the concern_id and an 80-char summary slice ONLY
    — the full payload lives in concerns.jsonl. The git commit covers
    BOTH files in a single commit.
    """
    folder = find_task_path(task_id)
    concerns_file = folder / "concerns.jsonl"
    _append_jsonl_line(concerns_file, payload)

    # Mirror to events.jsonl as a thin breadcrumb.
    event_kind = f"epm:concern-{payload['event']}"
    summary = (payload.get("summary") or "")[:80]
    mirror_note = (
        f"concern_id: {payload['concern_id']}\n"
        f"severity: {payload.get('severity', 'unknown')}\n"
        f"summary: {summary}"
    )
    mirror_payload: dict[str, Any] = {
        "ts": payload["ts"],
        "kind": event_kind,
        "version": 1,
        "by": payload.get("raised_by")
        or payload.get("addressed_by")
        or payload.get("deferred_by")
        or "unknown",
        "concern_id": payload["concern_id"],
        "note": mirror_note,
    }
    events_file = folder / "events.jsonl"
    _append_jsonl_line(events_file, mirror_payload)

    _git_commit(
        [concerns_file, events_file],
        f"task #{task_id}: concern-{payload['event']} {payload['concern_id']}",
    )


def raise_concern(
    task_id: int,
    concern_id: str,
    *,
    severity: str,
    summary: str,
    raised_by: str,
    raised_at_round: int,
    evidence: str | None = None,
) -> dict[str, Any]:
    """Append a ``raised`` (or ``verified-open``) event for a concern.

    Behaviour:

    * **First raise.** Appends ``event=raised``.
    * **Re-raise after ``addressed``.** Appends ``event=verified-open``
      with ``raised_at_round`` bumped to the current round — the reviewer
      is saying "you said you fixed this, but the issue is still
      visible". The severity is taken from the new call (reviewers may
      escalate).
    * **Re-raise at the SAME round with no prior history at that round.**
      Treated as the first-ever raise (BLOCKER, CONCERN, NIT all legal).
    * **Idempotent same-round re-raise.** If the latest event for
      ``concern_id`` is already a ``raised`` (or ``verified-open``) at
      the same ``raised_at_round`` with the same severity, this is a
      no-op — returns the existing event without appending. Lets the
      orchestrator replay the same reviewer brief safely.

    Validation:

    * ``concern_id`` must match the kebab-case rule.
    * ``severity`` must be in ``CONCERN_SEVERITIES``.
    * ``raised_at_round`` must be ≥ 1.
    * ``summary`` must be a non-empty string ≤ 200 chars.
    """
    _validate_concern_id(concern_id)
    if severity not in CONCERN_SEVERITIES:
        raise ValueError(f"severity {severity!r} not in {sorted(CONCERN_SEVERITIES)}")
    if not isinstance(raised_at_round, int) or raised_at_round < 1:
        raise ValueError(f"raised_at_round must be a positive int (got {raised_at_round!r})")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")
    if len(summary) > 200:
        raise ValueError(
            f"summary too long ({len(summary)} chars; max 200). Move detail to evidence."
        )
    if not isinstance(raised_by, str) or not raised_by.strip():
        raise ValueError("raised_by must be a non-empty string")
    with _locked():
        events = _read_concerns_raw(task_id)
        latest = _latest_event_for(events, concern_id)
        # Idempotent same-round same-severity re-raise.
        if (
            latest is not None
            and latest["event"] in ("raised", "verified-open")
            and latest.get("raised_at_round") == raised_at_round
            and latest.get("severity") == severity
        ):
            return latest
        # Re-raise after addressed → verified-open.
        if latest is not None and latest["event"] == "addressed":
            event_kind = "verified-open"
        else:
            event_kind = "raised"
        payload: dict[str, Any] = {
            "ts": _utcnow_iso(),
            "event": event_kind,
            "concern_id": concern_id,
            "severity": severity,
            "summary": summary.strip(),
            "raised_by": raised_by,
            "raised_at_round": raised_at_round,
        }
        if evidence:
            payload["evidence"] = evidence
        _append_concern_event(task_id, payload)
        return payload


def address_concern(
    task_id: int,
    concern_id: str,
    *,
    addressed_by: str,
    addressed_at_round: int,
    summary: str | None = None,
) -> dict[str, Any]:
    """Append an ``addressed`` event recording that the implementer (or
    analyzer / planner, depending on the stage) believes the concern has
    been fixed.

    The next reviewer round verifies. If the concern is still visible,
    that reviewer calls ``raise_concern`` again — which transitions the
    record to ``verified-open`` instead of a fresh ``raised`` event.

    ``concern_id`` MUST refer to a concern that has been raised at least
    once on this task; ``ValueError`` otherwise (defends against
    address-without-raise typos that would orphan the audit log).
    """
    _validate_concern_id(concern_id)
    if not isinstance(addressed_at_round, int) or addressed_at_round < 1:
        raise ValueError(f"addressed_at_round must be a positive int (got {addressed_at_round!r})")
    if not isinstance(addressed_by, str) or not addressed_by.strip():
        raise ValueError("addressed_by must be a non-empty string")
    with _locked():
        events = _read_concerns_raw(task_id)
        latest = _latest_event_for(events, concern_id)
        if latest is None:
            raise ValueError(
                f"concern_id {concern_id!r} has never been raised on task "
                f"#{task_id}; refusing to record an `addressed` event for "
                "a concern that does not exist."
            )
        # Carry the severity + original summary forward so list_concerns
        # consumers don't need to walk history.
        carried_summary = (summary or latest.get("summary") or "").strip()
        if len(carried_summary) > 200:
            raise ValueError(f"summary too long ({len(carried_summary)} chars; max 200).")
        payload: dict[str, Any] = {
            "ts": _utcnow_iso(),
            "event": "addressed",
            "concern_id": concern_id,
            "severity": latest.get("severity"),
            "summary": carried_summary,
            "addressed_by": addressed_by,
            "addressed_at_round": addressed_at_round,
        }
        _append_concern_event(task_id, payload)
        return payload


def defer_concern(
    task_id: int,
    concern_id: str,
    *,
    by: str,
    rationale: str,
) -> dict[str, Any]:
    """Append a ``deferred`` event. USER-ONLY at TWO layers.

    Layer 1 (CLI): rejects without ``--by user`` (plus a special-case
    for ``--by reconciler`` when the reconciler downgrades severity, per
    the design spec). Layer 2 (this function): also rejects ``by`` !=
    ``user`` / ``reconciler`` — defense in depth.

    BLOCKER concerns CANNOT be user-deferred — they signal a strict gate
    the orchestrator must address or pivot. ``ValueError`` on attempt.
    Sole exception (``workflow.yaml § concerns_protocol.
    reconciler_special_case``): the reconciler's binding adjudication may
    downgrade a single-twin BLOCKER, recorded via ``by="reconciler"`` —
    the rationale requirement still applies.

    Rationale must be ≥ 40 chars AND not match a known boilerplate
    phrase (see ``_CONCERN_RATIONALE_BOILERPLATE``).
    """
    _validate_concern_id(concern_id)
    if by not in ("user", "reconciler"):
        raise ValueError(
            "defer_concern is user-only — by must be 'user' (or 'reconciler' "
            f"for ensemble-tie-break severity downgrade); got {by!r}."
        )
    _validate_deferral_rationale(rationale)
    with _locked():
        events = _read_concerns_raw(task_id)
        latest = _latest_event_for(events, concern_id)
        if latest is None:
            raise ValueError(
                f"concern_id {concern_id!r} has never been raised on task "
                f"#{task_id}; refusing to defer a concern that does not exist."
            )
        if latest.get("severity") == "BLOCKER" and by != "reconciler":
            raise ValueError(
                f"concern_id {concern_id!r} is severity=BLOCKER — BLOCKERs "
                "cannot be user-deferred. Address it, pivot the strategy, "
                "or post epm:failure v1 and set status:blocked. (Sole "
                "exception: the reconciler's binding severity-downgrade "
                "via by='reconciler' — workflow.yaml § concerns_protocol."
                "reconciler_special_case.)"
            )
        carried_summary = (latest.get("summary") or "").strip()
        payload: dict[str, Any] = {
            "ts": _utcnow_iso(),
            "event": "deferred",
            "concern_id": concern_id,
            "severity": latest.get("severity"),
            "summary": carried_summary,
            "deferred_by": by,
            "deferral_rationale": rationale.strip(),
        }
        _append_concern_event(task_id, payload)
        return payload


# ─── Module entry point for CLI ────────────────────────────────────────────


# PEP-562 lazy attribute access. Defense-in-depth for ``tw.REPO``,
# ``tw.TASKS_DIR``, ``tw.REGISTRY_PATH`` callers. Note this does NOT save
# ``from explore_persona_space.task_workflow import TASKS_DIR`` — bare-name
# imports bind the value at import time. Those call-sites are refactored to
# the function form and the pytest grep test enforces it.
_LAZY_ATTRS = {
    "REPO": lambda: repo_root(),
    "TASKS_DIR": lambda: tasks_dir(),
    "REGISTRY_PATH": lambda: registry_path(),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        return _LAZY_ATTRS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


# PEP-562 lazy attributes are intentionally listed in ``__all__`` even
# though they are not module-scope assignments. The ``noqa: F822`` tags
# tell ruff to allow them — they resolve at attribute-access time via
# ``__getattr__``.
__all__ = [
    "CODE_KINDS",
    "COMMENT_KINDS",
    "CONCERN_EVENTS",
    "CONCERN_SEVERITIES",
    "FOLLOWUP_HELD_BLOCKED_STATUSES",
    "FOLLOWUP_RUN_KIND",
    "FOLLOWUP_SCOPE_KIND",
    "GOAL_H2_NAME",
    "KINDS",
    "PARK_STATUS",
    "REGISTRY_PATH",  # noqa: F822 — PEP-562 lazy attr (see __getattr__)
    "REPO",  # noqa: F822 — PEP-562 lazy attr (see __getattr__)
    "STATUSES",
    "TASKS_DIR",  # noqa: F822 — PEP-562 lazy attr (see __getattr__)
    "TERMINAL_STATUSES",
    "USER_INITIATED_FOLLOWUP_SOURCES",
    "NewTaskRequest",
    "ReconcileReport",
    "RegistryChange",
    "StaleTaskPathError",
    "add_tag",
    "address_concern",
    "append_comment",
    "audit",
    "create_task",
    "defer_concern",
    "executing_followup_label",
    "find_task_path",
    "followup_label_groups",
    "followup_retro_close_evidence",
    "get_goal",
    "get_relates_to",
    "get_task",
    "has_event",
    "invalidate_cache",
    "latest_event",
    "list_by_status",
    "list_comments",
    "list_concerns",
    "list_events",
    "new_plan_version",
    "parse_followup_note_field",
    "post_event",
    "primary_checkout_root",
    "promote",
    "raise_concern",
    "reconcile_registry",
    "registry_path",
    "remove_tag",
    "repo_root",
    "set_body",
    "set_clean_result",
    "set_goal",
    "set_kind",
    "set_status",
    "set_title",
    "tasks_dir",
    "unrun_followup_labels",
]
