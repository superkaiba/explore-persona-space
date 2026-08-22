"""Pre-dispatch task-premise staleness scanner (task #2134) — pure logic + CLI.

Nothing surfaced queued/blocked infra tasks whose premises a recent commit
invalidated: blocked #1217/#1771 kept instructing a gate removed by
c20aabc59a; #1718 targeted a ``scripts/workflow_lint.py`` that completed
#2079 had rewritten; a stale dispatch burns a session before the clarifier
discovers mootness (the #1985 shape). This module holds the PURE detection
logic the watcher's ``predispatch_staleness_pass`` consumes, plus a
report-only CLI for manual runs and the watcher smoke.

Signals (all ADVISORY — precision is adjudicated by the spawned clarifier
at its Step 0 context pass, or a human, per the #1918 archive-license
discipline; nothing here mutates any task state):

- ``stale-premise`` — a task's ``workflow_fix_target:`` file(s) were
  rewritten by a commit newer than the task's creation whose subject shares
  >= ``DEFAULT_MIN_TOKENS`` informative tokens with the task title+body.
- ``queue-collision`` — >= 2 scanned queued tasks name the same target file
  (pairwise merge-conflict risk); ONE record per file.
- ``landed-sibling-collision`` — post-creation commits on a task's target
  file(s) whose subjects name OTHER task/issue ids (recently-landed sibling
  work on the same file); ONE record per task aggregating the siblings.

Purity seam: :func:`scan` takes ``git_log_fn(paths, since_ts) ->
list[(sha, subject)]`` injected, so tests never spawn git. Fail toward
silence: an unreadable body / missing creation timestamp / failed git log
skips THAT task with one stderr line — a failed ``git log`` never reads as
"no staleness" (and never poisons the other tasks' scan).

CLI (report-only; rc 0 whether or not flags surface)::

    uv run python scripts/predispatch_staleness.py [--json] [--min-tokens N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Statuses/kinds in scope: the active pre-dispatch queue. `on_hold` is
# deliberately parked (out of the queue by definition); non-infra/batch
# kinds are out of scope (plan #2134 §"Explicitly out of scope").
SCAN_STATUSES: frozenset[str] = frozenset({"proposed", "blocked"})
SCAN_KINDS: frozenset[str] = frozenset({"infra", "batch"})

# >= 3 shared informative tokens between a landed commit subject and the
# task title+body — the heuristic the task body specifies.
DEFAULT_MIN_TOKENS = 3

# Bounded git budget: -n 30 --since=<created> per scanned task (the plan's
# kill criterion — never an unbounded per-tick history scan on the shared VM).
GIT_LOG_MAX_COMMITS = 30
GIT_LOG_TIMEOUT_S = 10.0

# Small stopword set: tokens that appear in nearly every commit subject /
# task body in this repo and carry no premise information. Deliberately
# conservative — a false KEEP only costs one advisory flag.
STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "adds",
        "after",
        "agent",
        "agents",
        "also",
        "always",
        "auto",
        "before",
        "check",
        "checks",
        "claude",
        "daily",
        "eval",
        "every",
        "file",
        "files",
        "fixes",
        "from",
        "gate",
        "gates",
        "into",
        "issue",
        "issues",
        "never",
        "only",
        "pass",
        "passes",
        "python",
        "rule",
        "rules",
        "script",
        "scripts",
        "session",
        "sessions",
        "step",
        "steps",
        "task",
        "tasks",
        "test",
        "tests",
        "that",
        "this",
        "when",
        "with",
        "workflow",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9_]{4,}")
# `workflow_fix_target:` line in a task body's Provenance section — bare or
# bullet form, value = comma/space-separated repo-relative paths, optionally
# suffixed with a parenthetical note (`(parse_judge_json step 4)`).
_TARGET_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?workflow_fix_target:\s*(.+)$", re.MULTILINE)
_PAREN_RE = re.compile(r"\([^)]*\)")
_PATHISH_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")
# Commit subjects naming a sibling task/issue: `task #123`, `issue 123`,
# `issue-123`, `#123` after "task"/"issue" only (a bare `#123` elsewhere is
# too promiscuous against sha fragments / PR numbers).
_SIBLING_RE = re.compile(r"(?:task\s*#|issue[\s#-])(\d+)", re.IGNORECASE)
_CREATED_AT_RE = re.compile(r"^created_at:\s*['\"]?([0-9T:+.Z\-]+)['\"]?\s*$", re.MULTILINE)


def informative_tokens(text: str) -> set[str]:
    """Lowercase ``[a-z0-9_]{4,}`` tokens of ``text`` minus :data:`STOPWORDS`."""
    if not isinstance(text, str) or not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def parse_targets(body_text: str) -> list[str]:
    """``workflow_fix_target:`` paths from a task body (deduped, in order).

    Handles the observed live shapes: bullet (``- workflow_fix_target: p``)
    and bare lines, comma- and/or space-separated multi-target values, and
    trailing parenthetical annotations (stripped). Absolute paths and the
    bare ``/`` token are DROPPED — a ``workflow_fix_target`` must be
    repo-relative (#1067's prose value tokenized to pathspec ``/``, a
    guaranteed git exit-128 + stderr line every firing). Unparseable /
    absent lines yield ``[]`` — fail toward silence, never raise.
    """
    if not isinstance(body_text, str) or not body_text:
        return []
    out: list[str] = []
    for m in _TARGET_LINE_RE.finditer(body_text):
        raw = _PAREN_RE.sub(" ", m.group(1))
        for piece in re.split(r"[,\s]+", raw.strip()):
            p = piece.strip().strip("`'\"")
            # Path-ish only: repo-relative (never "/" or an absolute path —
            # these feed a git pathspec), contains a slash or an extension
            # dot, and no shell-ish characters (defensive).
            if (
                p
                and _PATHISH_RE.match(p)
                and not p.startswith("/")
                and ("/" in p or "." in p)
                and p not in out
            ):
                out.append(p)
    return out


@dataclass(frozen=True)
class FlagRecord:
    """One advisory staleness/collision flag.

    ``evidence`` is a JSON-ready dict (commit sha + subject + matched
    tokens, or colliding issue ids); ``fingerprint`` = sha256[:12] over the
    sorted (kind, target, evidence) payload — the pass's per-(issue,
    fingerprint) dedup key, so a NEW invalidating commit re-fires while an
    unchanged flag stays deduped.
    """

    issue: int
    kind: str  # "stale-premise" | "queue-collision" | "landed-sibling-collision"
    target: str
    evidence: dict
    fingerprint: str


def _fingerprint(kind: str, target: str, evidence: dict) -> str:
    payload = json.dumps({"kind": kind, "target": target, "evidence": evidence}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _make_record(issue: int, kind: str, target: str, evidence: dict) -> FlagRecord:
    return FlagRecord(issue, kind, target, evidence, _fingerprint(kind, target, evidence))


def _eligible(t) -> tuple[int, list[str]] | None:
    """(issue, targets) when ``t`` is an in-scope queued task with parseable
    repo-relative targets, else None (scan()'s silent scope filter)."""
    try:
        issue = int(t.get("id"))
    except (TypeError, ValueError):
        return None
    if t.get("status") not in SCAN_STATUSES or t.get("kind") not in SCAN_KINDS:
        return None
    targets = parse_targets(t.get("body") or "")
    if not targets:
        return None
    return issue, targets


def scan(
    tasks,
    git_log_fn,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    collision_tasks=None,
) -> list[FlagRecord]:
    """PURE scan of ``tasks`` (dicts: id, status, kind, created_ts, title,
    body) against landed commits via the injected ``git_log_fn(paths,
    since_ts) -> list[(sha, subject)]`` seam.

    ``collision_tasks`` (default: ``tasks``) is the task list the git-free
    queue-collision grouping is built from. The caller may pass the FULL
    collected queue there while ``tasks`` stays its cap+cursor WINDOW, so
    two colliding tasks on opposite sides of a window boundary still
    co-detect (#2134 v2 fold); the git-backed signals (stale-premise /
    landed-sibling) always scan ``tasks`` only.

    Report-only by construction — returns records, mutates nothing. A task
    outside :data:`SCAN_STATUSES` / :data:`SCAN_KINDS`, without parseable
    targets, is skipped silently; a missing ``created_ts`` or a raising
    ``git_log_fn`` skips THAT task's commit-backed signals with one stderr
    line (the task still participates in queue-collision grouping, which
    needs no git).
    """
    records: list[FlagRecord] = []
    by_target: dict[str, list[int]] = {}
    for t in tasks if collision_tasks is None else collision_tasks:
        elig = _eligible(t)
        if elig is None:
            continue
        issue, targets = elig
        for target in targets:
            by_target.setdefault(target, []).append(issue)
    for t in tasks:
        elig = _eligible(t)
        if elig is None:
            continue
        issue, targets = elig
        target_label = ",".join(targets)
        created_ts = t.get("created_ts")
        if not created_ts:
            print(
                f"predispatch-staleness: #{issue} has no parseable created_at; "
                "skipping commit scan",
                file=sys.stderr,
            )
            continue
        try:
            commits = git_log_fn(targets, created_ts)
        except Exception as exc:
            # A failed git log NEVER reads as "no staleness" — skip loudly.
            print(
                f"predispatch-staleness: git log failed for #{issue} ({exc}); skipping",
                file=sys.stderr,
            )
            continue
        task_tokens = informative_tokens(f"{t.get('title') or ''}\n{t.get('body') or ''}")
        siblings: list[dict] = []
        for sha, subject in commits:
            matched = sorted(informative_tokens(subject) & task_tokens)
            if len(matched) >= min_tokens:
                records.append(
                    _make_record(
                        issue,
                        "stale-premise",
                        target_label,
                        {"sha": sha, "subject": subject, "matched_tokens": matched},
                    )
                )
            sib_ids = sorted({int(s) for s in _SIBLING_RE.findall(subject) if int(s) != issue})
            if sib_ids:
                siblings.append({"sha": sha, "subject": subject, "sibling_issues": sib_ids})
        if siblings:
            records.append(
                _make_record(
                    issue,
                    "landed-sibling-collision",
                    target_label,
                    {"siblings": siblings},
                )
            )
    for target, issues in sorted(by_target.items()):
        uniq = sorted(set(issues))
        if len(uniq) >= 2:
            records.append(
                _make_record(uniq[0], "queue-collision", target, {"colliding_issues": uniq})
            )
    return records


def git_log_for_paths(repo_root: Path, paths: list[str], since_ts: str) -> list[tuple[str, str]]:
    """Bounded read-only ``git log`` over ``paths`` since ``since_ts``.

    ``--max-count={GIT_LOG_MAX_COMMITS}`` + a {GIT_LOG_TIMEOUT_S}s timeout
    keep the per-task budget fixed (the plan's kill criterion). Raises on
    any git failure — the caller (scan) skips that task, never reading the
    failure as a clean scan.
    """
    res = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "log",
            f"--since={since_ts}",
            f"--max-count={GIT_LOG_MAX_COMMITS}",
            "--format=%H%x09%s",
            "--",
            *paths,
        ],
        capture_output=True,
        text=True,
        timeout=GIT_LOG_TIMEOUT_S,
        check=True,
    )
    out: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        sha, _, subject = line.partition("\t")
        if sha:
            out.append((sha, subject))
    return out


def parse_created_at(body_text: str) -> str | None:
    """ISO ``created_at`` from a body's YAML frontmatter, or None."""
    if not isinstance(body_text, str):
        return None
    m = _CREATED_AT_RE.search(body_text)
    return m.group(1) if m else None


def collect_tasks() -> tuple[list[dict], Path]:
    """(scan-shaped task dicts for the live queue, main repo root).

    Reads the canonical registry via ``task_workflow.registry_path()`` (the
    branch-guarded resolver — never a cwd-relative ``tasks/`` path) and each
    in-scope task's ``body.md`` at its registry-carried path. Fail-soft PER
    TASK: an unreadable body skips that task with one stderr line.
    """
    from explore_persona_space.task_workflow import registry_path

    reg_file = registry_path()
    reg_root = reg_file.parent.parent
    reg = json.loads(reg_file.read_text())
    tasks_map = reg.get("tasks") if isinstance(reg, dict) else None
    out: list[dict] = []
    if not isinstance(tasks_map, dict):
        return out, reg_root
    for id_str, meta in sorted(tasks_map.items(), key=lambda kv: kv[0]):
        if not isinstance(meta, dict):
            continue
        if meta.get("status") not in SCAN_STATUSES or meta.get("kind") not in SCAN_KINDS:
            continue
        try:
            issue = int(id_str)
        except ValueError:
            continue
        rel = meta.get("path")
        if not isinstance(rel, str) or not rel:
            continue
        body_path = reg_root / rel / "body.md"
        try:
            body_text = body_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"predispatch-staleness: body read failed for #{issue} ({exc}); skipping",
                file=sys.stderr,
            )
            continue
        out.append(
            {
                "id": issue,
                "status": meta.get("status"),
                "kind": meta.get("kind"),
                "created_ts": parse_created_at(body_text),
                "title": meta.get("title") or "",
                "body": body_text,
            }
        )
    return out, reg_root


def record_to_dict(rec: FlagRecord) -> dict:
    return {
        "issue": rec.issue,
        "kind": rec.kind,
        "target": rec.target,
        "evidence": rec.evidence,
        "fingerprint": rec.fingerprint,
    }


def main(argv: list[str] | None = None) -> int:
    """Report-only CLI: scan the live queue, print flags, exit 0 either way."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit flag records as a JSON array")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"shared-informative-token threshold for stale-premise (default {DEFAULT_MIN_TOKENS})",
    )
    args = parser.parse_args(argv)
    tasks, repo_root = collect_tasks()
    records = scan(
        tasks,
        git_log_fn=lambda paths, since: git_log_for_paths(repo_root, paths, since),
        min_tokens=args.min_tokens,
    )
    if args.json:
        print(json.dumps([record_to_dict(r) for r in records], indent=2))
    else:
        print(f"predispatch-staleness: scanned {len(tasks)} queued task(s), {len(records)} flag(s)")
        for rec in records:
            print(f"  #{rec.issue} {rec.kind} [{rec.fingerprint}] {rec.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
