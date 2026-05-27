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
flock on ~/.task-workflow/lock. Reads do NOT lock — readers see a
consistent snapshot because all writes are atomic (write-temp + rename).

Status enum (folder names):
  proposed planning plan_pending approved running verifying interpreting
  reviewing awaiting_promotion completed blocked archived
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ─── Config / paths ────────────────────────────────────────────────────────

STATUSES = (
    "proposed",
    "planning",
    "plan_pending",
    "approved",
    "running",
    "verifying",
    "interpreting",
    "reviewing",
    "awaiting_promotion",
    "completed",
    "blocked",
    "archived",
)

TERMINAL_STATUSES = frozenset({"completed", "blocked", "archived"})

# Status that means "user has reviewed and approved a clean-result body; user
# must run `task.py promote` to move to completed". Park-and-wait gate.
PARK_STATUS = "awaiting_promotion"

EVENT_NOTE_MAX = 50_000  # mirror Sagan's body-size cap

# Comment kinds the web UI exposes; checked when comments are appended.
COMMENT_KINDS = frozenset({"question", "answer", "followup-proposal", "note"})


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


@functools.lru_cache(maxsize=1)
def _resolve_repo_root_cached(_key: tuple[int, str]) -> Path:
    """Inner cache target. Keyed on (pid, cwd) so forks + chdirs invalidate
    automatically. The key is computed by the wrapper; we ignore the
    contents (we resolve relative to module dir + sanitized env, not cwd).
    """
    env = _sanitized_git_env()
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
            raise RuntimeError(
                f"main worktree HEAD is on {branch!r}; expected 'main'. "
                f"Switch with `git -C {parent} checkout main` before running task.py."
            )
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


def invalidate_cache() -> None:
    """Drop the cached repo-root resolution. Next call re-probes git."""
    _resolve_repo_root_cached.cache_clear()


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


def _read_body(path: Path) -> tuple[dict[str, Any], str]:
    return _split_frontmatter(path.read_text())


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
    """Return absolute path to tasks/<status>/<task_id>/. Resolves via REGISTRY."""
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
    if not abs_path.is_dir():
        raise FileNotFoundError(
            f"task #{task_id} registry says {entry['path']!r} but that dir is missing; "
            f"run `task.py audit` to repair"
        )
    return abs_path


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


def post_event(
    task_id: int,
    kind: str,
    *,
    version: int = 1,
    by: str = "unknown",
    note: str | None = None,
    artifacts: list[str] | None = None,
    **extras: Any,
) -> dict[str, Any]:
    """Append a single event to tasks/<status>/<id>/events.jsonl.

    Note size is capped at EVENT_NOTE_MAX chars to mirror Sagan; oversize
    raises ValueError so the caller can fall back to a failure marker.
    """
    if note is not None and len(note) > EVENT_NOTE_MAX:
        raise ValueError(
            f"event note exceeds {EVENT_NOTE_MAX} chars ({len(note)}); "
            f"caller must post epm:failure v1 with reason=note_oversize"
        )
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
    with _locked():
        path = find_task_path(task_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _git_commit(
            [path],
            f"task #{task_id}: {kind}" + (f" — {note[:60]}" if note else ""),
        )
    return payload


def list_events(task_id: int) -> list[dict[str, Any]]:
    path = find_task_path(task_id) / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_event(task_id: int, prefix: str | None = None) -> dict[str, Any] | None:
    events = list_events(task_id)
    if prefix:
        events = [e for e in events if e["kind"].startswith(prefix)]
    return events[-1] if events else None


def has_event(task_id: int, kind: str) -> bool:
    return any(e["kind"] == kind for e in list_events(task_id))


# ─── Status transitions ────────────────────────────────────────────────────


def set_status(task_id: int, new_status: str, *, note: str | None = None) -> Path:
    """Move tasks/<old>/<id>/ → tasks/<new>/<id>/ via `git mv`, then post a
    status-changed event. Returns the new absolute path.
    """
    if new_status not in STATUSES:
        raise ValueError(f"unknown status: {new_status!r}; expected one of {STATUSES}")
    with _locked():
        old = find_task_path(task_id)
        old_status = _status_from_path(old)
        if old_status == new_status:
            return old
        repo = repo_root()
        new_parent = tasks_dir() / new_status
        new_parent.mkdir(parents=True, exist_ok=True)
        new = new_parent / str(task_id)
        # `git mv` so renames are tracked
        rel_old = old.relative_to(repo)
        rel_new = new.relative_to(repo)
        _run_git(["mv", str(rel_old), str(rel_new)])
        # Update REGISTRY
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
        with ev_path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        # Pass BOTH old and new to _git_commit so the deletion side of
        # the `git mv` is included in the commit's --only pathspec.
        # Otherwise the staged deletion at <old> remains in the index and
        # gets swept into the next unrelated `git commit` (incident:
        # 2026-05-24, tasks 382/383 source-side deletions leaked into
        # commit 49e49f4a).
        _git_commit([old, new, registry_path()], f"task #{task_id}: {old_status} → {new_status}")
    return new


# ─── Task creation ──────────────────────────────────────────────────────────


@dataclass
class NewTaskRequest:
    kind: str  # experiment | infra | analysis | survey
    title: str
    body: str = ""
    parent_id: int | None = None
    tags: list[str] | None = None
    status: str = "proposed"
    # Canonical Goal of the experiment. Honored only when kind=="experiment";
    # passed through for other kinds with a soft warning emitted by the CLI.
    goal: str | None = None


def create_task(req: NewTaskRequest) -> int:
    """Create tasks/<status>/<NEW_ID>/ with body.md (frontmatter + body),
    empty events.jsonl, empty comments.jsonl. Returns the new ID.
    """
    if req.status not in STATUSES:
        raise ValueError(f"unknown status: {req.status!r}")
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
        with (path / "events.jsonl").open("a") as f:
            f.write(json.dumps(created_event, ensure_ascii=False) + "\n")
        # Register
        _registry_set(reg, task_id, path, fm)
        _save_registry(reg)
        _git_commit([path, registry_path()], f"task #{task_id}: create — {req.title[:60]}")
        return task_id


# ─── Body / frontmatter mutations ──────────────────────────────────────────


def set_body(task_id: int, new_body: str, *, snapshot_original: bool = False) -> None:
    """Replace the body content (preserves frontmatter).

    If `snapshot_original` is True, save the current full body.md to
    original-body.md first — used by the analyzer when promoting a
    clean-result.

    Any YAML frontmatter at the START of ``new_body`` is stripped before
    the canonical frontmatter (loaded from the existing body.md) is
    prepended. This prevents the duplicate-frontmatter trap when callers
    pass a complete markdown document (frontmatter + body) — see
    `_strip_leading_frontmatter_blocks` for the incident history. The
    strip is idempotent: calling `set_body` with a body that already has
    no leading frontmatter is a no-op for the strip step.

    Note: this function preserves the EXISTING frontmatter on body.md.
    If you need to change frontmatter fields, use the dedicated mutators
    (`set_title`, `set_clean_result`, `add_tag`, `remove_tag`,
    `set_goal`). The stripped frontmatter from `new_body` is discarded.
    """
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, _ = _read_body(path)
        touched: list[Path] = [path]
        if snapshot_original:
            orig = path.parent / "original-body.md"
            shutil.copy2(path, orig)
            touched.append(orig)
        body_text = _strip_leading_frontmatter_blocks(new_body)
        _write_body(path, fm, body_text if body_text.endswith("\n") else body_text + "\n")
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


def set_clean_result(task_id: int, value: bool = True) -> None:
    with _locked():
        path = find_task_path(task_id) / "body.md"
        fm, body = _read_body(path)
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
        with ev_path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
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


# ─── Plans ──────────────────────────────────────────────────────────────────


def new_plan_version(task_id: int, plan_md: str) -> int:
    """Append plans/v{next}.md, update plans/plan.md symlink. Returns the
    new version number.
    """
    with _locked():
        plans_dir = find_task_path(task_id) / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(plans_dir.glob("v*.md"))
        next_v = len(existing) + 1
        target = plans_dir / f"v{next_v}.md"
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
        with (path / "events.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": _utcnow_iso(),
                        "kind": "epm:promoted",
                        "version": 1,
                        "by": "user",
                        "classification": verdict,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
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
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _git_commit([path], f"task #{task_id}: comment {cid} ({kind})")
    return record


def list_comments(task_id: int) -> list[dict[str, Any]]:
    path = find_task_path(task_id) / "comments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ─── Git helpers ────────────────────────────────────────────────────────────


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Resolve cwd PER CALL (not from a cached module-level REPO). The
    # process-local LRU cache in `repo_root()` makes this cheap, and per-call
    # resolution is what keeps long-lived processes (PM session, agent
    # daemons) safe across `os.chdir()` or branch state changes.
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
        check=check,
        capture_output=True,
        text=True,
    )


def _git_commit(paths: list[Path], message: str) -> None:
    """Stage the given paths and create a single commit. Optional push.

    Uses ``git commit --only -- <paths>`` so unrelated staged work elsewhere in
    the repo is not silently captured under the task.py commit message. Parallel
    agents (workflow-improver, /issue runs, user-staged edits) share the same
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

    Set TASK_PY_NO_COMMIT=1 to skip the commit entirely (useful in tests).
    Set TASK_PY_AUTO_PUSH=1 to also push after the commit.
    """
    if os.environ.get("TASK_PY_NO_COMMIT") == "1":
        return
    repo = repo_root()
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
    full_msg = f"{message}\n\n[task.py]"
    _run_git(["commit", "-m", full_msg, "--only", "--", *rel_paths])
    if os.environ.get("TASK_PY_AUTO_PUSH") == "1":
        _run_git(["push"], check=False)


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
    "COMMENT_KINDS",
    "GOAL_H2_NAME",
    "PARK_STATUS",
    "REGISTRY_PATH",  # noqa: F822 — PEP-562 lazy attr (see __getattr__)
    "REPO",  # noqa: F822 — PEP-562 lazy attr (see __getattr__)
    "STATUSES",
    "TASKS_DIR",  # noqa: F822 — PEP-562 lazy attr (see __getattr__)
    "TERMINAL_STATUSES",
    "NewTaskRequest",
    "add_tag",
    "append_comment",
    "audit",
    "create_task",
    "find_task_path",
    "get_goal",
    "get_task",
    "has_event",
    "invalidate_cache",
    "latest_event",
    "list_by_status",
    "list_comments",
    "list_events",
    "new_plan_version",
    "post_event",
    "promote",
    "registry_path",
    "remove_tag",
    "repo_root",
    "set_body",
    "set_clean_result",
    "set_goal",
    "set_status",
    "set_title",
    "tasks_dir",
]
