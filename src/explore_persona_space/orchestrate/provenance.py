"""Git provenance for result-JSON reproducibility metadata.

Consolidates the three duplicate `_git_commit_hash()` helpers previously
scattered across analysis/artifacts modules. Adds dirty-tree flagging so a
committed result JSON never claims provenance from a commit that does not
contain the code that produced it (task #2065; incident #1482).

Public entry point: `git_provenance()` returns a `GitProvenance` dataclass
carrying `commit_sha` (8-hex short), `commit_sha_full` (40-hex | None, #2194),
`dirty` (bool | None), `dirty_paths` (bounded list), and the argv[0]
tracked-state signal `argv0_state` / `argv0_path` (#2175).
`commit_string(prov)` renders the human-legible `<sha>` or `<sha>+dirty` form
for non-JSON channels (PDF metadata, PNG chunks, WandB run names) — the SHORT
form, unchanged. Result JSONs merge `as_metadata_dict(prov)` into their
`metadata` block, exposing the structured fields `git_commit` (the FULL
40-hex form when resolved — abbreviated SHAs are excluded by
`scripts/verify_report.py::check_code_sha_cards`; #2194) / `git_dirty` /
`git_dirty_paths` / `git_argv0_state` / `git_argv0_path`, plus an OPTIONAL
`phase` card phase-IDENTITY slug (`as_metadata_dict(prov, phase="stage2-upload")`,
validated via `validate_phase_identity`) emitted as a SIBLING of `git_commit` —
the exact dict level the `code-sha-cards` gate reads (#2194).

Contract:
- Never fails loud: a non-git tree, missing git binary, or subprocess timeout
  degrades to `commit_sha="unknown", dirty=None` (record it, don't crash the
  run). The rule this closes is "the git_commit field must not silently claim
  clean provenance while the working tree is dirty" — a `None` sentinel
  explicitly says "we could not check" (the code-style.md caveat).
- Working-tree-wide scope: `git status --porcelain=v1 --untracked-files=no`
  captures every modified tracked file, not just files matching some plan-time
  allow-list. `dirty_paths` is capped at `_MAX_DIRTY_PATHS` (default 50) with
  a trailing "... N more" marker so a large sweep never blows up the JSON.
- Producing-script signal: the tree-wide scan deliberately excludes untracked
  files (the shared repo root routinely holds OTHER sessions' untracked
  scripts, which would flip `dirty` on files the run never touched), so the
  PRODUCING SCRIPT itself gets a targeted probe: argv[0]'s git state is
  recorded as `argv0_state` ("tracked" | "modified" | "untracked" | None) +
  `argv0_path`, and an UNTRACKED argv[0] folds into `dirty=True` (#2094: an
  untracked entrypoint stamped a clean SHA). Named residual (deliberate): an
  untracked IMPORTED module beside a tracked entrypoint is still invisible —
  tree-wide untracked coverage was rejected for the fleet-noise cost above.
- Bounded: 5s subprocess timeout per git call.
- Literal pathspecs: every git call routes through `_run_git`, which passes
  the global `--literal-pathspecs` option, so glob metacharacters (`[`, `*`,
  `?`) in a filename can never match a tracked SIBLING and misclassify an
  untracked producing script as "tracked" (#2175 r2).
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_GIT_TIMEOUT_SEC = 5
_MAX_DIRTY_PATHS = 50
_UNKNOWN = "unknown"

# Card phase-IDENTITY convention (#2194): lowercase kebab/snake slug, emitted
# INSIDE the reproducibility-card / metadata block as a SIBLING of
# `git_commit` — the exact dict level scripts/verify_report.py's
# `check_code_sha_cards` reads (a top-level `phase` beside a nested card is
# invisible to the gate: the #2162 stage2 shape).
_PHASE_IDENTITY_RE = re.compile(r"^[a-z0-9]+([-_][a-z0-9]+)*$")
# BEST-EFFORT lifecycle-collision fence: backend sentinels use TOP-LEVEL
# {"phase": "done"} lifecycle-STATE vocabulary (backends/artifacts.py). The
# denylist refuses only the most confusable values — the backend emits more
# lifecycle values than are enumerated here (startup/preflight/wedged/...);
# the structural top-level-vs-commit-sibling placement is the primary
# separation, this denylist a second fence.
_LIFECYCLE_PHASE_VOCAB = frozenset(
    {"done", "failed", "running", "pending", "queued", "started", "workload"}
)


@dataclass(frozen=True)
class GitProvenance:
    """Structured git-provenance record for a run.

    Attributes:
        commit_sha: 8-hex short SHA of HEAD, or `"unknown"` if unresolved.
        dirty: True if the working tree has uncommitted tracked-file
            modifications OR the producing script (argv[0]) is untracked;
            False if clean; None if the check could not run (non-git tree /
            missing binary / timeout).
        dirty_paths: List of modified tracked paths (porcelain v1 format),
            capped at `_MAX_DIRTY_PATHS` entries with a `... N more` tail.
            An untracked argv[0] is PREPENDED (so the overflow tail can
            never drop it; cap+1 length acceptable).
        argv0_path: Repo-relative path of argv[0] when its state resolved;
            None otherwise (never a path beside a null state).
        argv0_state: "tracked" | "modified" | "untracked" | None (could not
            determine — non-git tree, gitignored argv[0] such as the pytest
            binary, outside-repo argv[0], `-c`, timeout).
        commit_sha_full: Full 40-hex SHA of HEAD when resolved; None when the
            SHA could not be resolved OR on hand-built records that predate
            the field (#2194 — `as_metadata_dict` then falls back to the
            short `commit_sha`). Appended LAST with a default so existing
            positional constructions stay valid.
    """

    commit_sha: str
    dirty: bool | None
    dirty_paths: list[str] = field(default_factory=list)
    argv0_path: str | None = None
    argv0_state: str | None = None
    commit_sha_full: str | None = None


def _run_git(args: list[str], cwd: Path | None) -> str | None:
    """Run git with ``--literal-pathspecs``; return stdout, or None on failure.

    ``--literal-pathspecs`` (a global option, placed BEFORE the subcommand) is
    load-bearing for every pathspec-taking probe in this module: without it,
    glob metacharacters (``[``, ``*``, ``?``) in a filename make the pathspec a
    PATTERN, so an untracked ``scripts/foo[1].py`` argv[0] would MATCH a
    tracked sibling ``scripts/foo1.py`` and be misclassified as "tracked" —
    defeating the #2175 untracked-producing-script invariant. The option is
    inert for calls that carry no pathspec (rev-parse, tree-wide status).
    """
    try:
        result = subprocess.run(
            ["git", "--literal-pathspecs", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_head_sha(cwd: Path | None = None) -> tuple[str, str | None]:
    """Return HEAD as ``(short 8-hex, full 40-hex | None)``; ``("unknown",
    None)`` on any failure.

    One ``rev-parse HEAD`` call resolves both forms (#2194): the short form
    is the full SHA's first 8 chars — always exactly 8, whereas the previous
    ``--short=8`` probe could return MORE on ambiguity — and the full form is
    what ``as_metadata_dict`` emits under ``git_commit`` so cards from this
    helper are usable by ``verify_report.py``'s ``code-sha-cards`` gate
    (abbreviated SHAs are gate-excluded). Subprocess call count unchanged.
    """
    out = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    full = out.strip() if out is not None else ""
    if not full:
        return _UNKNOWN, None
    return full[:8], full


def _git_short_sha(cwd: Path | None = None) -> str:
    """Return the 8-hex short SHA of HEAD, or `"unknown"` on any failure.

    Kept as the thin deprecated shim for the one external importer
    (`analysis/paper_plots.py`); new callers use `git_provenance()`.
    """
    return _git_head_sha(cwd=cwd)[0]


def _git_dirty_status(cwd: Path | None = None) -> tuple[bool | None, list[str]]:
    """Return (dirty?, capped modified-paths list).

    Uses `git status --porcelain=v1 --untracked-files=no` for a tracked-file
    read. Untracked files are excluded from this TREE-WIDE scan because the
    shared repo root routinely holds other concurrent sessions' untracked
    scripts, which would flip `dirty` on files the run never touched — the
    producing script itself is covered by the targeted `_argv0_git_state`
    probe instead (#2175).
    """
    out = _run_git(
        ["status", "--porcelain=v1", "--untracked-files=no"],
        cwd=cwd,
    )
    if out is None:
        return None, []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return False, []
    # Porcelain-v1 format: two status chars + space + path (rename arrow retained
    # verbatim; the audit trail wants the raw entry).
    paths = [ln[3:] if len(ln) > 3 else ln for ln in lines]
    if len(paths) > _MAX_DIRTY_PATHS:
        overflow = len(paths) - _MAX_DIRTY_PATHS
        paths = [*paths[:_MAX_DIRTY_PATHS], f"... {overflow} more"]
    return True, paths


def _argv0_git_state(argv0: str | None, cwd: Path | None = None) -> tuple[str | None, str | None]:
    """Classify argv[0]'s git state as (state, repo-relative path).

    `state` is one of "tracked" | "modified" | "untracked" | None; the return
    is `(None, None)` on EVERY indeterminate branch (no venv-binary path noise
    beside a null state).

    Resolution frame: `Path(argv0)` is resolved to ABSOLUTE against the
    PROCESS cwd (`Path.resolve()`) BEFORE any git call, and the absolute
    pathspec is what git receives — a relative pathspec under the `cwd=`
    subprocess param would reopen a wrong-file misattribution channel. An
    outside-repo absolute pathspec exits rc=128 → `(None, None)`, which is
    exactly how the pytest-binary default degrades in tmp_path fixture repos.
    Every git call routes through `_run_git`, which passes
    `--literal-pathspecs` so glob metacharacters in the resolved path can
    never match a tracked SIBLING (#2175 r2). Resolution failures — a
    symlink loop raises `RuntimeError` on py3.11 (`OSError` on newer
    Pythons), null bytes raise `ValueError` — degrade to `(None, None)`
    per the module's never-crash contract.

    The load-bearing ignored-file branch: a gitignored argv[0] (e.g.
    `.venv/bin/pytest`) yields EMPTY porcelain output under
    `--untracked-files=all` and reads as `(None, None)` — NOT untracked — or
    every pytest-invoked call would false-flag dirty.
    """
    if not argv0:
        return None, None
    try:
        resolved = Path(argv0).resolve()
        if not resolved.is_file():
            return None, None
    except (OSError, RuntimeError, ValueError):
        return None, None
    abs_path = str(resolved)

    tracked_probe = _run_git(["ls-files", "--error-unmatch", "--", abs_path], cwd=cwd)
    if tracked_probe is not None:
        status = _run_git(["status", "--porcelain=v1", "--", abs_path], cwd=cwd)
        if status is None:
            return None, None
        rows = [ln for ln in status.splitlines() if ln.strip()]
        if rows:
            return "modified", rows[0][3:] if len(rows[0]) > 3 else rows[0]
        toplevel = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
        if toplevel is None:
            return None, None
        try:
            rel = str(resolved.relative_to(Path(toplevel.strip()).resolve()))
        except (OSError, RuntimeError, ValueError):
            return None, None
        return "tracked", rel

    status = _run_git(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", abs_path],
        cwd=cwd,
    )
    if status is None:
        return None, None
    rows = [ln for ln in status.splitlines() if ln.strip()]
    if rows and rows[0].startswith("??"):
        return "untracked", rows[0][3:] if len(rows[0]) > 3 else rows[0]
    return None, None


def git_provenance(cwd: Path | None = None, argv0: str | None = None) -> GitProvenance:
    """Capture the current git provenance for reproducibility metadata.

    `argv0` defaults to `sys.argv[0]` (the producing script); pass it
    explicitly for clean test injection. An UNTRACKED argv[0] folds into
    `dirty=True` with its path PREPENDED to `dirty_paths` — the explicit
    positive finding outranks an inconclusive tracked scan (`dirty=None`),
    which permits the rare `unknown+dirty` render when the SHA/status calls
    timed out while the argv[0] probe succeeded (timeout asymmetry). A
    "modified" argv[0] needs no folding — the tracked scan already reports
    it (no double-entry in `dirty_paths`).
    """
    sha, sha_full = _git_head_sha(cwd=cwd)
    dirty, paths = _git_dirty_status(cwd=cwd)
    state, argv0_path = _argv0_git_state(sys.argv[0] if argv0 is None else argv0, cwd=cwd)
    if state == "untracked":
        dirty = True
        if argv0_path is not None and argv0_path not in paths:
            paths = [argv0_path, *paths]
    return GitProvenance(
        commit_sha=sha,
        dirty=dirty,
        dirty_paths=paths,
        argv0_path=argv0_path,
        argv0_state=state,
        commit_sha_full=sha_full,
    )


def commit_string(prov: GitProvenance) -> str:
    """Human-legible `<sha>` or `<sha>+dirty` for non-JSON channels.

    Used in PDF Keywords, PNG pnginfo `Commit` chunks, and any other flat-string
    context where the JSON `git_dirty` field cannot ride along. A `dirty=None`
    provenance (git-unavailable lane) renders as the bare sha — the JSON
    metadata carries the explicit `null` signal separately.
    """
    if prov.dirty is True:
        return f"{prov.commit_sha}+dirty"
    return prov.commit_sha


def validate_phase_identity(phase: str) -> str:
    """Validate a card phase-IDENTITY slug; returns it verbatim or raises ValueError.

    Phase identity names the pipeline stage a reproducibility card belongs to
    ("stage2-upload", "grid-anchors") — NOT lifecycle state: backend sentinels
    use top-level ``{"phase": "done"}`` for state (backends/artifacts.py), a
    different vocabulary at a different dict level. The denylist refuses the
    most confusable lifecycle values as a BEST-EFFORT fence (the backend emits
    more lifecycle values than are enumerated here; the structural
    top-level-vs-commit-sibling placement is the primary separation).
    Fail-loud is deliberate: the value is a hardcoded literal at the call
    site, so a violation is a programming error, not runtime degradation (the
    module's never-crash contract covers environment failures, not
    caller-contract violations).
    """
    if not isinstance(phase, str) or not _PHASE_IDENTITY_RE.fullmatch(phase):
        raise ValueError(f"phase identity must match {_PHASE_IDENTITY_RE.pattern!r}: {phase!r}")
    if phase in _LIFECYCLE_PHASE_VOCAB:
        raise ValueError(
            f"phase identity {phase!r} collides with the backend-sentinel LIFECYCLE "
            "vocabulary — name the pipeline stage instead (e.g. 'stage2-upload')"
        )
    return phase


def as_metadata_dict(prov: GitProvenance, *, phase: str | None = None) -> dict[str, object]:
    """Render the provenance as reproducibility-metadata dict fields.

    Consumers `metadata.update(as_metadata_dict(git_provenance()))` into their
    result JSON's `metadata` block. Fields:

    - `git_commit`: str, the FULL 40-hex SHA when resolved (#2194 —
      gate-usable by `verify_report.py::check_code_sha_cards`; abbreviated
      SHAs are gate-excluded), falling back to the short `commit_sha` on
      hand-built records with no `commit_sha_full` (or `"unknown"`).
    - `git_dirty`: bool | None. True/False when checked; None when the check
      could not run (record the explicit sentinel — don't infer clean).
    - `phase`: str, present ONLY when the `phase` kwarg is passed (#2194) —
      the card phase-IDENTITY slug, validated via `validate_phase_identity`,
      a SIBLING of `git_commit` at the exact dict level the `code-sha-cards`
      gate reads. `phase=None` (default) emits no key, so old-card output is
      byte-identical.
    - `git_dirty_paths`: list[str], present ONLY when `dirty is True`.
    - `git_argv0_state`: str | None, ALWAYS present ("tracked" | "modified" |
      "untracked"; None = could not determine).
    - `git_argv0_path`: str, present ONLY when `argv0_state` is not None.
    """
    out: dict[str, object] = {
        "git_commit": prov.commit_sha_full or prov.commit_sha,
        "git_dirty": prov.dirty,
    }
    if phase is not None:
        out["phase"] = validate_phase_identity(phase)
    if prov.dirty is True:
        out["git_dirty_paths"] = list(prov.dirty_paths)
    out["git_argv0_state"] = prov.argv0_state
    if prov.argv0_state is not None:
        out["git_argv0_path"] = prov.argv0_path
    return out
