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

The helper never writes, never mutates task state, never touches the network.
Worktree safety (#2384 §2.3): cited body paths resolve ONLY through
``explore_persona_space.task_workflow`` (never a hand-built
``tasks/<status>/<M>/`` path), and the git probe runs against the MAIN
checkout resolved as ``dirname(git rev-parse --path-format=absolute
--git-common-dir)`` probed from ``tasks_dir()``, so the helper is correct
when invoked from a worktree cwd.
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


def _strip_code_blocks(text: str) -> str:
    """Drop fenced code blocks (``` / ~~~ toggling, delimiters included) and
    indented-4 (or tab-indented) code lines — command examples and JSON
    payloads carry ``#`` refs that are not citations (#2384 §2.2 filter 1)."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or line.startswith(("    ", "\t")):
            continue
        out.append(line)
    return "\n".join(out)


def extract_cited_ids(plan_text: str, *, self_issue: int) -> list[int]:
    r"""Cited task ids in draft order: fenced/indented blocks stripped,
    word-char-preceded refs dropped (``(?<!\w)`` ONLY — see the regex
    comment), the plan's own issue dropped (self-reference is never a
    citation), deduped, capped at ``_MAX_CITED_IDS``."""
    prose = _strip_code_blocks(plan_text)
    seen: set[int] = set()
    out: list[int] = []
    for m in _ISSUE_REF_RE.finditer(prose):
        n = int(m.group(1))
        if n == self_issue or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= _MAX_CITED_IDS:
            break
    return out


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
    silent, and never fatal (per-id fail-soft skip)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
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
    """MAIN-checkout root: ``dirname(git rev-parse --path-format=absolute
    --git-common-dir)``, probed FROM ``tasks_dir()`` so the result is the
    main checkout even when invoked from a worktree cwd (#2384 §2.3).
    Raises ``RuntimeError`` on probe failure — routed to :func:`main`'s
    top-level fail-soft handler (``UNKNOWN``/exit 0, never a block)."""
    from explore_persona_space import task_workflow  # local import (worktree-safe resolver)

    anchor = task_workflow.tasks_dir()
    out = _git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=anchor)
    if not out:
        raise RuntimeError(f"git repo unresolvable from {anchor}")
    return Path(out.splitlines()[-1]).parent


def last_commit_unix(path: Path, *, repo_root: Path) -> int | None:
    """``git -C <repo_root> log -1 --format=%ct -- <path>``; ``None`` on
    empty output, non-zero rc, timeout, or a non-integer tail."""
    out = _git(["log", "-1", "--format=%ct", "--", str(path)], cwd=repo_root)
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


def _path_history(path: Path, *, repo_root: Path) -> list[tuple[str, int, str]]:
    """Full ``(sha, commit_unix, subject)`` history of ``path`` (newest
    first) via one parsed ``git log`` — used instead of ``--since`` date
    parsing so window membership uses the same ``%ct`` read as
    :func:`last_commit_unix`. Best-effort: ``[]`` on any git failure."""
    out = _git(["log", "--format=%H\t%ct\t%s", "--", str(path)], cwd=repo_root)
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


def _classify_diff(diff: str) -> str | None:
    """Label the dominant false-positive channel (#2384 §6): a status move
    (``git mv`` between status folders) shows as ``rename-only``; a user
    promotion sweep (``classification`` flip) as ``frontmatter-only`` —
    so the orchestrator's disposition is one glance. ``None`` = a content
    diff (the real staleness signal). Advisory label only, never a verdict
    input."""
    if not diff:
        return None
    changed = [
        ln
        for ln in diff.splitlines()
        if (ln.startswith("+") and not ln.startswith("+++"))
        or (ln.startswith("-") and not ln.startswith("---"))
    ]
    if not changed:
        return "rename-only" if "rename from" in diff or "similarity index" in diff else None
    fm_line = re.compile(r"^[+-](?:---\s*$|[A-Za-z_][A-Za-z0-9_-]*:)")
    if all(fm_line.match(ln) for ln in changed):
        return "frontmatter-only"
    return None


def body_diff_since(path: Path, since_unix: int, *, repo_root: Path) -> str:
    """Oneline log of the commits touching ``path`` after ``since_unix``
    plus ``git diff <oldest-in-window>^..HEAD -- <path>``, truncated to
    ``_MAX_DIFF_CHARS``. Best-effort — ``''`` on any failure (the verdict
    never depends on this display)."""
    rows = _path_history(path, repo_root=repo_root)
    window = [r for r in rows if r[1] > since_unix]
    if not window:
        return ""
    lines = [f"{sha[:10]} @{ct} {subject}" for sha, ct, subject in window]
    oldest_sha = window[-1][0]
    diff = _git(["diff", "-M", f"{oldest_sha}^..HEAD", "--", str(path)], cwd=repo_root)
    text = "\n".join(lines)
    if diff:
        text += "\n" + diff
    if len(text) > _MAX_DIFF_CHARS:
        text = text[:_MAX_DIFF_CHARS] + f"\n... [truncated at {_MAX_DIFF_CHARS} chars]"
    return text


def check(
    plan_text: str, *, self_issue: int, since_unix: int, repo_root: Path
) -> tuple[str, list[dict]]:
    """-> (``'CLEAN' | 'STALE' | 'UNKNOWN'``, findings). One finding dict
    per cited id: ``status`` in {clean, stale, unresolved, git-failed}.
    ``UNKNOWN`` fires only when cited ids exist but NONE could be probed
    (so a confident ``CLEAN checked=0`` is never printed over a broken
    probe); zero cited ids is a genuine ``CLEAN checked=0``."""
    ids = extract_cited_ids(plan_text, self_issue=self_issue)
    findings: list[dict] = []
    for n in ids:
        body = cited_body_path(n)
        if body is None or not body.exists():
            findings.append({"id": n, "status": "unresolved"})
            continue
        ct = last_commit_unix(body, repo_root=repo_root)
        if ct is None:
            findings.append({"id": n, "status": "git-failed", "path": str(body)})
            continue
        findings.append(
            {
                "id": n,
                "status": "stale" if ct > since_unix else "clean",
                "path": str(body),
                "last_commit_unix": ct,
            }
        )
    if not ids:
        return "CLEAN", findings
    probed = [f for f in findings if f["status"] in ("stale", "clean")]
    if not probed:
        return "UNKNOWN", findings
    return ("STALE" if any(f["status"] == "stale" for f in probed) else "CLEAN"), findings


def _counts(findings: list[dict]) -> dict[str, int]:
    return {
        "checked": sum(1 for f in findings if f["status"] in ("stale", "clean")),
        "unresolved": sum(1 for f in findings if f["status"] == "unresolved"),
        "git_failed": sum(1 for f in findings if f["status"] == "git-failed"),
    }


def _emit_stale_details(findings: list[dict], since_unix: int, repo_root: Path) -> None:
    """Per stale id, the diff-since block on STDERR (stdout stays a single
    capturable verdict line), led by the #2384 §6 disposition label."""
    for f in findings:
        if f["status"] != "stale":
            continue
        path = Path(f["path"])
        diff = body_diff_since(path, since_unix, repo_root=repo_root)
        label = _classify_diff(diff)
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

    if args.plan_file:
        plan_path = Path(args.plan_file)
    else:
        from explore_persona_space import task_workflow  # local import (worktree-safe resolver)

        plan_path = task_workflow.find_task_path(args.issue) / "plans" / "plan.md"
    plan_text = plan_path.read_text(encoding="utf-8")  # OSError -> fail-soft UNKNOWN

    repo_root = resolve_repo_root()
    verdict, findings = check(
        plan_text, self_issue=args.issue, since_unix=since_unix, repo_root=repo_root
    )
    c = _counts(findings)
    if args.json:
        print(
            json.dumps(
                {"verdict": verdict, "since_unix": since_unix, **c, "findings": findings},
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
    """0 = CLEAN or UNKNOWN (fail-soft), 3 = STALE (actionable).

    The ``except Exception`` below is the ONE deliberate top-level fail-soft
    handler of #2384 acceptance criterion 2 — the gate must NEVER block a
    persist on its own crash; the reason is always printed, never swallowed.
    """
    try:
        return _run(argv)
    except Exception as exc:  # fail-soft BY CONTRACT (#2384 criterion 2) — reason logged below
        reason = f"{type(exc).__name__}: {exc}".replace("\n", "; ")
        print(f"CITED-BODY-CURRENCY: UNKNOWN reason={reason}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
