"""Cited-body currency gate — the adversarial-planner pre-persist helper (#2384).

A plan draft can cite a parent task's result that is corrected IN THE PARENT'S
OWN ``body.md`` between the fact-checker's verification pass and the plan
persist (incident #2378: plan v5 persisted four minutes after #825's body was
corrected to report the OPPOSITE sign, commit ``488dad540c``). This helper is
the mechanical re-check the gate runs immediately BEFORE every
``task.py new-plan-version`` persist:

    uv run python scripts/check_cited_body_currency.py --issue <N> \\
        --since-unix <ts> [--plan-file <path>] [--json]

It extracts cited task ids (``#<id>``) from the draft, compares each cited
``body.md``'s last-commit timestamp against the draft-start reference
timestamp, prints EXACTLY ONE verdict line on stdout, and surfaces the diff
for any stale citation on stderr (so a ``$(...)`` capture of the verdict
stays clean):

    CITED-BODY-CURRENCY: CLEAN checked=4 since=1787939126
    CITED-BODY-CURRENCY: STALE ids=825 checked=4 since=1787939126
    CITED-BODY-CURRENCY: UNKNOWN reason=<one-line>

Reference timestamp (#2384 §2.1): the campaign's ROUND-1 planner-spawn time
(``DRAFT_START="$(date +%s)"``, captured ONCE, never refreshed on Phase 3
re-spawns — a per-round reference would certify the inter-round critic-review
gaps CLEAN). When ``--since-unix`` is EMPTY or ABSENT (session recovery, a
planner death and inline redraft), the helper RE-DERIVES it from the OLDEST
``planner-dispatch``-leading ``epm:progress`` breadcrumb in the task's
``events.jsonl``; only when no breadcrumb exists either does it return
``UNKNOWN``/exit 0.

Fail-soft contract (#2384 acceptance criterion 2), stated precisely:

- unresolvable cited id (no registry entry, never existed) -> skipped,
  counted in ``unresolved=<n>``, verdict unaffected;
- ``git log`` non-zero / empty / timeout -> that id skipped (counted in
  ``git_failed=<n>``), verdict unaffected;
- unreadable plan file, bad ``--since-unix``, import failure, ANY unexpected
  exception -> the ONE deliberate top-level handler in :func:`main` prints
  ``UNKNOWN reason=...`` and returns 0;
- exit 3 is reachable ONLY from a positively-established stale citation.

**Exit-code vocabulary (complete).** ``0`` = CLEAN or UNKNOWN (the
fail-soft verdicts — the gate never blocks a persist on its own failure);
``3`` = STALE (the one actionable verdict); ``2`` = argparse USAGE error
(a malformed CLI invocation — argparse's own ``SystemExit(2)``, raised
before any verdict is computed). ``2`` is a CALLER bug, NOT a verdict: it
prints argparse's usage text and NO ``CITED-BODY-CURRENCY:`` line, so a
caller keying on the verdict line can never mistake it for STALE. Callers
that treat any non-zero exit as "bounce the persist" should key on ``3``
specifically (the SKILL.md gate does).

The helper never writes, never mutates task state, never touches the network.
Worktree safety (#2384 §2.3): cited body paths resolve ONLY through
``explore_persona_space.task_workflow`` (never a hand-built
``tasks/<status>/<M>/`` path), and the git probe runs against the working
tree that ACTUALLY HOLDS that tasks tree — ``git rev-parse
--show-toplevel`` anchored at ``tasks_dir()`` — so a resolver routing
through the managed ``_task-main-pin`` worktree is probed in ITS OWN tree
rather than against the primary checkout's root (#2384 round-2 C2; git
returns empty history for an absolute path belonging to a different
working tree, which would silently degrade every probe to UNKNOWN and
disable the blocking leg).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Extraction shape mirrors verify_plan.py's `_C18_ISSUE_REF_RE` (`#\d{2,}`)
# with the URL-adjacency guard added; module-level copy BY DESIGN — this is a
# standalone script and must not import verify_plan's 15.6k-line module for
# one regex (#2384 §2.2). The `\d{2,}` floor drops `#1`-style prose noise and
# one-digit markdown anchors. The lookbehind class is `\w` ONLY — do NOT
# widen to `[/\w.-]`: `/`-adjacency would drop every non-first member of the
# `#884/#1045/#1134` lineage-list idiom, a genuine citation form measured at
# 10% of real cited ids (#2384 §2.2 filter 2).
_ISSUE_REF_RE = re.compile(r"(?<!\w)#(\d{2,})")
_MAX_CITED_IDS = 40  # a runaway plan must not fan out into 200 git calls
_GIT_TIMEOUT_S = 10
_MAX_DIFF_CHARS = 8000
_DISPATCH_NOTE_RE = re.compile(r"^\s*planner-dispatch\b")

# CommonMark fence: 3+ backticks OR 3+ tildes, indented at most 3 spaces (4+
# spaces is an indented code block, NOT a fence). The delimiter CHARACTER and
# RUN LENGTH are both load-bearing — see `_strip_code_blocks` (#2384 round-2
# blocker 5).
_FENCE_RE = re.compile(r"^ {0,3}(?P<delim>`{3,}|~{3,})(?P<info>.*)$")

# A resolved draft-start reference below this is nonsense (2001-09-09; every
# commit in this repo's history postdates it), and one more than a day in the
# future cannot be a DRAFT START. Both directions silently invert the gate:
# a far-past reference marks every cited body STALE, a far-future one certifies
# every body CLEAN (#2384 round-2 blocker 7).
_MIN_PLAUSIBLE_REF_UNIX = 1_000_000_000
_FUTURE_SKEW_ALLOWANCE_S = 86_400


def _strip_code_blocks(text: str) -> str:
    """Drop fenced code blocks (delimiters included) and indented-4 (or
    tab-indented) code lines — command examples and JSON payloads carry ``#``
    refs that are not citations (#2384 §2.2 filter 1).

    Fence handling follows CommonMark on the two axes a delimiter-BLIND
    toggle gets wrong (#2384 round-2 blocker 5):

    - **Delimiter identity + run length.** A fence is closed only by a run of
      the SAME character at least as long as the opener, with no info string.
      A blind toggle lets a ``~~~`` line close a ```` ``` ```` block (and vice
      versa), and lets an inner ```` ``` ```` close a ```` ```` ```` block —
      re-opening the fence on the NEXT delimiter and inverting in/out for the
      whole rest of the document.
    - **Indented-code precedence.** A fence opener may be indented at most 3
      spaces; at 4+ spaces the line is indented CODE that happens to start
      with backticks. A blind ``line.strip().startswith("```")`` toggles on it
      and (again) inverts the rest of the document. `_FENCE_RE`'s ``{0,3}``
      indent bound is what routes those lines to the indented-code branch.

    Both failure modes are silent and DIRECTIONAL: an inverted fence state
    strips real prose (citations vanish -> the gate under-checks) or admits
    command examples (spurious ids -> noisy findings).
    """
    out: list[str] = []
    fence: str | None = None  # the OPEN fence's delimiter run, or None
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if fence is not None:
            if (
                m is not None
                and m.group("delim")[0] == fence[0]
                and len(m.group("delim")) >= len(fence)
                and not m.group("info").strip()
            ):
                fence = None
            continue
        # A backtick fence's info string may not itself contain a backtick
        # (CommonMark); a tilde fence's may.
        if m is not None and not (m.group("delim")[0] == "`" and "`" in m.group("info")):
            fence = m.group("delim")
            continue
        if line.startswith(("    ", "\t")):
            continue
        out.append(line)
    return "\n".join(out)


def extract_cited_ids_with_total(plan_text: str, *, self_issue: int) -> tuple[list[int], int]:
    r"""``(capped ids in draft order, TOTAL distinct ids before the cap)``.

    Fenced/indented blocks stripped, word-char-preceded refs dropped
    (``(?<!\w)`` ONLY — see the regex comment), the plan's own issue dropped
    (self-reference is never a citation), deduped. The second element exists
    so a caller can DISCLOSE cap truncation: a bare ``CLEAN checked=40`` over
    a 55-citation plan reads as full coverage while 15 citations went
    unprobed (#2384 round-2 blocker 9)."""
    prose = _strip_code_blocks(plan_text)
    seen: set[int] = set()
    out: list[int] = []
    for m in _ISSUE_REF_RE.finditer(prose):
        n = int(m.group(1))
        if n == self_issue or n in seen:
            continue
        seen.add(n)
        if len(out) < _MAX_CITED_IDS:
            out.append(n)
    return out, len(seen)


def extract_cited_ids(plan_text: str, *, self_issue: int) -> list[int]:
    """Capped cited ids only — thin wrapper over
    :func:`extract_cited_ids_with_total` for callers that do not report
    truncation."""
    return extract_cited_ids_with_total(plan_text, self_issue=self_issue)[0]


def cited_body_path(issue: int) -> Path | None:
    """Registry-resolved ``<task folder>/body.md``; ``None`` when the id is
    unresolvable (absent from registry AND disk — the fail-soft contract's
    ``unresolved`` clause, never fatal)."""
    from explore_persona_space import task_workflow  # local import (worktree-safe resolver)

    try:
        folder = task_workflow.find_task_path(issue)
    except FileNotFoundError:
        # Includes StaleTaskPathError (its subclass): an unresolvable cited
        # id is skipped and counted in unresolved=<n>, verdict unaffected.
        return None
    return folder / "body.md"


def _git(args: list[str], *, cwd: Path) -> str | None:
    """Run git under a 10 s timeout, returning stripped stdout; ``None`` on
    non-zero rc / timeout / missing binary or cwd — logged to stderr, never
    silent, and never fatal (per-id fail-soft skip).

    ``errors="replace"`` is load-bearing: git echoes PATHS and COMMIT SUBJECTS
    verbatim, and neither is guaranteed UTF-8 (a latin-1 filename, a subject
    committed under a non-UTF-8 locale). Under the default strict decoding
    those bytes raise ``UnicodeDecodeError`` from inside ``subprocess.run`` —
    an exception class this helper does not catch, so the whole probe would
    die on a body whose commit history merely mentions a non-UTF-8 name
    (#2384 round-2 blocker 3)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        print(f"check_cited_body_currency: git {args[0]} failed: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        print(
            f"check_cited_body_currency: git {args[0]} rc={proc.returncode}"
            + (f": {err[0][:160]}" if err else ""),
            file=sys.stderr,
        )
        return None
    return proc.stdout.strip()


def resolve_repo_root() -> Path:
    """The WORKING-TREE root that actually holds ``tasks_dir()``:
    ``git rev-parse --show-toplevel`` probed FROM ``tasks_dir()`` (#2384 §2.3).

    ``--show-toplevel`` — NOT ``dirname(--git-common-dir)``. The two differ
    exactly when ``task_workflow`` routes reads through a LINKED worktree:
    `_ensure_managed_main_worktree` keeps a managed ``_task-main-pin``
    checkout so an off-main session still reads main's task state, and
    ``tasks_dir()`` then resolves INSIDE that worktree. ``--git-common-dir``
    points at the PRIMARY checkout's ``.git``, so its parent is the primary
    checkout — a DIFFERENT working tree from the one holding the paths.
    Pairing the two makes every probe ``git -C <primary> log -- <pin path>``:
    the path is outside that tree, git exits 0 with EMPTY output, every id
    counts ``git_failed``, and the verdict degrades to a silent, permanent
    ``UNKNOWN`` — i.e. the BLOCKING leg of this gate quietly stops blocking
    (#2384 round-2 blocker 2). Anchoring at ``tasks_dir()`` and taking that
    tree's own toplevel keeps root and paths in the same working tree by
    construction, in a worktree cwd and under the pin alike.

    Raises ``RuntimeError`` on probe failure — routed to :func:`main`'s
    top-level fail-soft handler (``UNKNOWN``/exit 0, never a block)."""
    from explore_persona_space import task_workflow  # local import (worktree-safe resolver)

    anchor = task_workflow.tasks_dir()
    out = _git(["rev-parse", "--show-toplevel"], cwd=anchor)
    if not out:
        raise RuntimeError(f"git repo unresolvable from {anchor}")
    return Path(out.splitlines()[-1])


def repo_relative(path: Path, repo_root: Path) -> str | None:
    """``path`` as a repo-root-relative POSIX string, or ``None`` when it
    lies outside ``repo_root``.

    Every git probe is path-limited, and git silently returns EMPTY (rc=0)
    for a path outside the working tree — indistinguishable from "no commits
    touched it". Converting up front turns that silent degradation into an
    explicit, counted finding (#2384 round-2 blocker 2). Both sides are
    ``resolve()``d so a symlinked tasks tree cannot produce a spurious
    mismatch."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def last_commit_unix(rel_path: str, *, repo_root: Path) -> int | None:
    """``git -C <repo_root> log -1 --format=%ct -- <rel_path>``; ``None`` on
    empty output, non-zero rc, timeout, or a non-integer tail.

    ``rel_path`` is repo-root-relative (see :func:`repo_relative`) — the
    caller has already established containment, so an empty result here means
    "no commits touched this path", never "wrong working tree".

    Deliberately NOT rename-following: for a CITED BODY a status move is a
    false POSITIVE (a body flagged stale that only moved folders), which
    #2384 §6 elects to SURFACE with an advisory ``rename-only`` label rather
    than filter — surfacing an extra citation costs a glance, missing a real
    correction costs the plan. The prior-plan-version REFERENCE leg is the
    opposite polarity (a status move there silently WIDENS nothing and
    NARROWS the watch window) and does follow renames — see
    ``verify_plan._c75_last_content_commit_unix``."""
    out = _git(["log", "-1", "--format=%ct", "--", rel_path], cwd=repo_root)
    if not out:
        return None
    try:
        return int(out.splitlines()[-1])
    except ValueError:
        print(f"check_cited_body_currency: unparseable %ct output {out!r}", file=sys.stderr)
        return None


def _iso_to_unix(ts: str) -> int | None:
    """ISO-8601 marker ``ts`` -> unix seconds; ``None`` on an unparseable
    value (that row simply contributes no reference)."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def oldest_planner_dispatch_unix(folder: Path) -> int | None:
    """Unix ts of the OLDEST ``planner-dispatch``-leading ``epm:progress``
    row in the task's ``events.jsonl`` — the campaign's FIRST recorded
    planner spawn, deliberately never the current round's (a
    newest-breadcrumb reference would certify the inter-round critic-review
    gaps CLEAN, #2384 §2.1). ``None`` when no breadcrumb exists. Records
    split on ``"\\n"``, never ``splitlines()`` (the #950 embedded-U+2028
    class); an unparseable row is skipped with a stderr note (it cannot
    provide a reference), never fatal."""
    ev = folder / "events.jsonl"
    if not ev.exists():
        return None
    oldest: int | None = None
    for line in ev.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip() or "planner-dispatch" not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(
                f"check_cited_body_currency: unparseable events.jsonl row skipped: {line[:80]!r}",
                file=sys.stderr,
            )
            continue
        note, ts = row.get("note"), row.get("ts")
        if row.get("kind") != "epm:progress" or not isinstance(note, str):
            continue
        if not _DISPATCH_NOTE_RE.match(note) or not isinstance(ts, str):
            continue
        unix = _iso_to_unix(ts)
        if unix is not None and (oldest is None or unix < oldest):
            oldest = unix
    return oldest


def _path_history(rel_path: str, *, repo_root: Path) -> list[tuple[str, int, str]]:
    """Full ``(sha, commit_unix, subject)`` history of ``rel_path`` (newest
    first) via one parsed ``git log`` — used instead of ``--since`` date
    parsing so window membership uses the same ``%ct`` read as
    :func:`last_commit_unix`. Best-effort: ``[]`` on any git failure."""
    out = _git(["log", "--format=%H\t%ct\t%s", "--", rel_path], cwd=repo_root)
    if not out:
        return []
    rows: list[tuple[str, int, str]] = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        try:
            rows.append((parts[0], int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def commit_path_status(sha: str, rel_path: str, *, repo_root: Path) -> str | None:
    """The per-commit name-status letter for ``rel_path`` as the DESTINATION
    in commit ``sha`` (``'M'``, ``'A'``, ``'R100'``, ...); ``None`` when the
    path is not a destination in that commit (merge commits print nothing
    without ``-m``, so they land here) or the probe fails."""
    out = _git(["show", "--format=", "--name-status", "-M", sha], cwd=repo_root)
    if not out:
        return None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        if parts[-1] == rel_path:
            return parts[0]
    return None


def classify_window(
    window: list[tuple[str, int, str]], rel_path: str, diff: str, *, repo_root: Path
) -> str | None:
    """Label the dominant false-positive channel (#2384 §6): a status move
    (``git mv`` between status folders) is ``rename-only``; a user promotion
    sweep (``classification`` flip) is ``frontmatter-only``. ``None`` = a
    content diff (the real staleness signal). Advisory label only, never a
    verdict input.

    ``rename-only`` is decided from PER-COMMIT name-status with rename
    detection (``git show --name-status -M``), never from the diff text
    (#2384 round-2 blocker 8). A path-limited
    ``git diff <oldest>^..HEAD -- <new path>`` cannot show a rename at all:
    at ``<oldest>^`` the file did not exist under the new path, so the whole
    body renders as ADDED lines — the old code's "zero changed lines +
    'rename from'" predicate is unreachable, and the label #2384 §6 names as
    the dominant false-positive mitigation never fired. Requiring EVERY
    windowed commit to be an exact ``R100`` keeps the label honest: a
    rename-WITH-edit (``R095``) is a real content change and stays unlabeled.
    """
    if window:
        statuses = [commit_path_status(sha, rel_path, repo_root=repo_root) for sha, _, _ in window]
        if all(s == "R100" for s in statuses):
            return "rename-only"
    if not diff:
        return None
    changed = [
        ln
        for ln in diff.splitlines()
        if (ln.startswith("+") and not ln.startswith("+++"))
        or (ln.startswith("-") and not ln.startswith("---"))
    ]
    if not changed:
        return None
    fm_line = re.compile(r"^[+-](?:---\s*$|[A-Za-z_][A-Za-z0-9_-]*:)")
    if all(fm_line.match(ln) for ln in changed):
        return "frontmatter-only"
    return None


def body_diff_since(rel_path: str, since_unix: int, *, repo_root: Path) -> tuple[str, str | None]:
    """``(display text, advisory label)`` for the commits touching
    ``rel_path`` after ``since_unix``: a oneline log plus
    ``git diff <oldest-in-window>^..HEAD -- <rel_path>``, truncated to
    ``_MAX_DIFF_CHARS``. Best-effort — ``('', None)`` on any failure (the
    verdict never depends on this display).

    Under a ``rename-only`` label the diff BODY is suppressed: it is the
    whole file rendered as additions (see :func:`classify_window`), which is
    pure noise for a status move."""
    rows = _path_history(rel_path, repo_root=repo_root)
    window = [r for r in rows if r[1] > since_unix]
    if not window:
        return "", None
    lines = [f"{sha[:10]} @{ct} {subject}" for sha, ct, subject in window]
    oldest_sha = window[-1][0]
    diff = _git(["diff", "-M", f"{oldest_sha}^..HEAD", "--", rel_path], cwd=repo_root) or ""
    label = classify_window(window, rel_path, diff, repo_root=repo_root)
    text = "\n".join(lines)
    if diff and label != "rename-only":
        text += "\n" + diff
    if len(text) > _MAX_DIFF_CHARS:
        text = text[:_MAX_DIFF_CHARS] + f"\n... [truncated at {_MAX_DIFF_CHARS} chars]"
    return text, label


def check(
    plan_text: str, *, self_issue: int, since_unix: int, repo_root: Path
) -> tuple[str, list[dict], int]:
    """-> (``'CLEAN' | 'STALE' | 'UNKNOWN'``, findings, total_cited). One
    finding dict per cited id: ``status`` in {clean, stale, unresolved,
    git-failed}. ``UNKNOWN`` fires only when cited ids exist but NONE could
    be probed (so a confident ``CLEAN checked=0`` is never printed over a
    broken probe); zero cited ids is a genuine ``CLEAN checked=0``.

    ``total_cited`` is the DISTINCT citation count BEFORE the
    ``_MAX_CITED_IDS`` cap, so the caller can disclose truncation
    (#2384 round-2 blocker 9)."""
    ids, total_cited = extract_cited_ids_with_total(plan_text, self_issue=self_issue)
    findings: list[dict] = []
    for n in ids:
        body = cited_body_path(n)
        if body is None or not body.exists():
            findings.append({"id": n, "status": "unresolved"})
            continue
        rel = repo_relative(body, repo_root)
        if rel is None:
            print(
                f"check_cited_body_currency: cited body #{n} at {body} is outside "
                f"the git root {repo_root} — cannot probe",
                file=sys.stderr,
            )
            findings.append({"id": n, "status": "git-failed", "path": str(body)})
            continue
        ct = last_commit_unix(rel, repo_root=repo_root)
        if ct is None:
            findings.append({"id": n, "status": "git-failed", "path": str(body), "rel": rel})
            continue
        findings.append(
            {
                "id": n,
                "status": "stale" if ct > since_unix else "clean",
                "path": str(body),
                "rel": rel,
                "last_commit_unix": ct,
            }
        )
    if not ids:
        return "CLEAN", findings, total_cited
    probed = [f for f in findings if f["status"] in ("stale", "clean")]
    if not probed:
        return "UNKNOWN", findings, total_cited
    verdict = "STALE" if any(f["status"] == "stale" for f in probed) else "CLEAN"
    return verdict, findings, total_cited


def _counts(findings: list[dict]) -> dict[str, int]:
    return {
        "checked": sum(1 for f in findings if f["status"] in ("stale", "clean")),
        "unresolved": sum(1 for f in findings if f["status"] == "unresolved"),
        "git_failed": sum(1 for f in findings if f["status"] == "git-failed"),
    }


def _emit_stale_details(findings: list[dict], since_unix: int, repo_root: Path) -> None:
    """Per stale id, the diff-since block on STDERR (stdout stays a single
    capturable verdict line), led by the #2384 §6 disposition label.

    DISPLAY ONLY. The caller has already established and printed the STALE
    verdict, so a failure in here must never change it — each id's block is
    individually guarded (#2384 round-2 blocker 12)."""
    for f in findings:
        if f["status"] != "stale":
            continue
        rel = f.get("rel")
        if not rel:
            continue
        try:
            diff, label = body_diff_since(rel, since_unix, repo_root=repo_root)
        except Exception as exc:  # display-only: never downgrade an established STALE
            print(
                f"check_cited_body_currency: diff render failed for #{f['id']}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        header = f"--- stale cited body #{f['id']}: {f['path']}"
        if label:
            header += f" [{label}]"
        print(header + " ---", file=sys.stderr)
        if diff:
            print(diff, file=sys.stderr)


def _resolve_since(args: argparse.Namespace) -> int | None:
    """``--since-unix`` when non-empty (a non-integer value RAISES ->
    top-level ``UNKNOWN``); else the lost-shell-var re-derivation from the
    OLDEST ``planner-dispatch`` breadcrumb (#2384 §2.1 lost-shell-var
    clause — an empty reference is NEVER a silent no-op). ``None`` = no
    breadcrumb either (the caller prints ``UNKNOWN``)."""
    raw = (args.since_unix or "").strip()
    if raw:
        return int(raw)  # ValueError -> main()'s fail-soft handler
    from explore_persona_space import task_workflow  # local import (worktree-safe resolver)

    folder = task_workflow.find_task_path(args.issue)
    return oldest_planner_dispatch_unix(folder)


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True, help="the task the draft belongs to")
    parser.add_argument(
        "--since-unix",
        default="",
        help="draft-start reference (unix seconds; DRAFT_START captured at the "
        "ROUND-1 planner spawn). Empty/absent -> re-derived from the oldest "
        "planner-dispatch breadcrumb in events.jsonl",
    )
    parser.add_argument(
        "--plan-file",
        default="",
        help="draft to scan (default: the task's plans/plan.md symlink)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report instead of text")
    args = parser.parse_args(argv)

    since_unix = _resolve_since(args)
    if since_unix is None:
        print(
            "CITED-BODY-CURRENCY: UNKNOWN reason=no --since-unix and no "
            "planner-dispatch breadcrumb in events.jsonl"
        )
        return 0

    # Plausibility of the FINAL resolved reference, whichever leg produced it
    # (#2384 round-2 blocker 7). Both directions silently invert the gate, so
    # neither may pass as a verdict: too far past marks every cited body
    # STALE, too far future certifies every one CLEAN. UNKNOWN + exit 0 keeps
    # the fail-soft contract (a bad reference is the gate's own defect, not
    # grounds to block a persist).
    now = int(datetime.now(tz=UTC).timestamp())
    if since_unix < _MIN_PLAUSIBLE_REF_UNIX or since_unix > now + _FUTURE_SKEW_ALLOWANCE_S:
        print(
            f"CITED-BODY-CURRENCY: UNKNOWN reason=implausible draft-start reference "
            f"{since_unix} (expected {_MIN_PLAUSIBLE_REF_UNIX} <= ref <= "
            f"{now + _FUTURE_SKEW_ALLOWANCE_S})"
        )
        return 0

    if args.plan_file:
        plan_path = Path(args.plan_file)
    else:
        from explore_persona_space import task_workflow  # local import (worktree-safe resolver)

        plan_path = task_workflow.find_task_path(args.issue) / "plans" / "plan.md"
    plan_text = plan_path.read_text(encoding="utf-8")  # OSError -> fail-soft UNKNOWN

    repo_root = resolve_repo_root()
    verdict, findings, total_cited = check(
        plan_text, self_issue=args.issue, since_unix=since_unix, repo_root=repo_root
    )
    c = _counts(findings)
    # Cap truncation is DISCLOSED, never silent: a bare `CLEAN checked=40`
    # over a 55-citation plan reads as full coverage (#2384 round-2 blocker 9).
    dropped = max(0, total_cited - len(findings))
    if args.json:
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "since_unix": since_unix,
                    **c,
                    "cited_total": total_cited,
                    "cap": _MAX_CITED_IDS,
                    "capped": bool(dropped),
                    "not_examined": dropped,
                    "findings": findings,
                },
                indent=2,
            )
        )
    elif verdict == "UNKNOWN":
        print(
            "CITED-BODY-CURRENCY: UNKNOWN reason=no cited id probed successfully "
            f"(cited={len(findings)} unresolved={c['unresolved']} git_failed={c['git_failed']})"
        )
    else:
        suffix = ""
        if c["unresolved"]:
            suffix += f" unresolved={c['unresolved']}"
        if c["git_failed"]:
            suffix += f" git_failed={c['git_failed']}"
        if dropped:
            suffix += f" capped={_MAX_CITED_IDS} not_examined={dropped}"
        if verdict == "STALE":
            stale_ids = ",".join(str(f["id"]) for f in findings if f["status"] == "stale")
            print(
                f"CITED-BODY-CURRENCY: STALE ids={stale_ids} "
                f"checked={c['checked']} since={since_unix}{suffix}"
            )
        else:
            print(f"CITED-BODY-CURRENCY: CLEAN checked={c['checked']} since={since_unix}{suffix}")
    if verdict == "STALE":
        _emit_stale_details(findings, since_unix, repo_root)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    """0 = CLEAN or UNKNOWN (fail-soft), 3 = STALE (actionable), 2 = argparse
    USAGE error (a malformed CLI invocation; argparse's own ``SystemExit(2)``
    propagates past the handler below, printing usage text and NO
    ``CITED-BODY-CURRENCY:`` verdict line — a caller bug, never a verdict).

    The ``except Exception`` below is the ONE deliberate top-level fail-soft
    handler of #2384 acceptance criterion 2 — the gate must NEVER block a
    persist on its own crash; the reason is always printed, never swallowed.
    ``SystemExit`` derives from ``BaseException``, so the exit-2 usage path is
    untouched by it.
    """
    try:
        return _run(argv)
    except Exception as exc:  # fail-soft BY CONTRACT (#2384 criterion 2) — reason logged below
        reason = f"{type(exc).__name__}: {exc}".replace("\n", "; ")
        print(f"CITED-BODY-CURRENCY: UNKNOWN reason={reason}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
