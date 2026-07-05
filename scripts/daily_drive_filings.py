"""Drive /daily route-2/3 task filings from a durable filings dir, incrementally.

The permanent promotion of the ad-hoc ``logs/daily/filings-2026-07-04/drive_filings.py``
driver (task #1061). The /daily skill writes every route-2/3 filing body plus a
``manifest.json`` to ``logs/daily/filings-<date>/`` BEFORE any filing starts, then drives
the filings through this script in small batches. Every outcome is appended to
``<dir>/filed.jsonl`` the moment it lands (two-phase ``attempting`` -> terminal rows), so
a mid-run kill strands at most the one in-flight item and a re-invocation resumes from
the ledger instead of forcing a which-got-filed audit.

Manifest item schema: ``{slug, route: 2|3, title, target, bug, change, body?}`` where
``body`` defaults to ``<dir>/<slug>.md`` (absolute paths pass through; relative paths
resolve against the filings dir).

Ledger row shapes (one JSON object per line, ISO-UTC ``ts`` on every row):

- ``{"slug", "outcome": "attempting", "fp", "route", "id_floor", "ts"}`` — appended
  BEFORE the filer subprocess (the crash-safety ordering); ``id_floor`` is the max
  task id at that moment and scopes later title-scan recovery to THIS filing.
- ``{"slug", "outcome": "filed", "id", "rc", "fp", "route", "tail", "ts"}``
- ``{"slug", "outcome": "deduped", "against", "fp", "route", "ts"}`` (route 2 only)
- ``{"slug", "outcome": "recovered", "id", "fp", "route", "dispatch_unconfirmed", "ts"}``
- ``{"slug", "outcome": "ERROR", "flag", "id", "rc", "fp", "route", "tail", "ts"}``
  with ``flag`` one of ``filer-failed`` / ``no-id-parsed`` / ``timeout`` /
  ``ambiguous-recovery``.

Exit code: 0 when the whole slice processed without appending an ERROR row this
invocation, else 1 (record-then-report). Validation failures (missing manifest / body
file / unresolvable date / malformed manifest item / non-trailing ledger corruption)
raise before any filing starts.

Usage:
    uv run python scripts/daily_drive_filings.py --dir logs/daily/filings-2026-07-05 \\
        [--start I --end J] [--retry-errors] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from explore_persona_space.task_workflow import wf_fix_fingerprint

LEDGER_NAME = "filed.jsonl"
QUARANTINE_NAME = "filed.jsonl.quarantined"
TERMINAL_OUTCOMES = frozenset({"filed", "deduped", "recovered"})
REQUIRED_ITEM_KEYS = frozenset({"slug", "route", "title", "target", "bug", "change"})
# Anchored to the line start: every file_infra_task.py success path prints a line starting
# `filed #<id>` or `filed + dispatched #<id>`. A stray `#N` elsewhere must not win.
FILED_ID_RE = re.compile(r"^filed (?:\+ dispatched )?#(\d+)", re.M)
DIR_DATE_RE = re.compile(r"filings-(\d{4}-\d{2}-\d{2})$")
FILER_TIMEOUT_S = 280  # bounds file_infra_task.py's internal 120s+120s subprocess timeouts
ROUTE_RECOVERY_TAG = {2: "daily-auto-filed", 3: "daily-held"}


def repo_root() -> Path:
    """Resolve the MAIN checkout root from this script's own location (never the cwd).

    Uses the git common dir so the WORKTREE copy of this script still resolves to the
    canonical repo root — a worktree-cwd invocation must not write a reapable
    worktree-local filings dir, and `task_workflow.repo_root()` is unsuitable here
    (it branch-guards / manages worktrees). Raises on git failure.
    """
    script_dir = Path(__file__).resolve().parent
    proc = subprocess.run(
        ["git", "-C", str(script_dir), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    common = Path(proc.stdout.strip())
    if not common.is_absolute():
        common = (script_dir / common).resolve()
    return common.parent


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_filings_dir(dir_arg: str, root: Path) -> Path:
    """Resolve --dir against the repo root (relative form) or as-is (absolute form)."""
    p = Path(dir_arg)
    if not p.is_absolute():
        p = root / p
    if not p.is_dir():
        raise FileNotFoundError(f"filings dir not found: {p}")
    return p


def resolve_date(date_arg: str | None, dirpath: Path) -> str:
    """The date for origin-prompt strings: --date, else parsed from the dir basename."""
    if date_arg:
        return date_arg
    m = DIR_DATE_RE.search(dirpath.name)
    if not m:
        raise ValueError(
            f"cannot derive date: --date not given and dir basename {dirpath.name!r} "
            "does not match 'filings-YYYY-MM-DD'"
        )
    return m.group(1)


def _resolve_body_path(item: dict, dirpath: Path) -> Path:
    body = item.get("body") or f"{item['slug']}.md"
    p = Path(body)
    if not p.is_absolute():
        p = dirpath / p
    return p


def load_and_validate_manifest(dirpath: Path) -> list[dict]:
    """Parse + validate the WHOLE manifest up front, so a schema wart aborts at ZERO filings.

    Checks per item: required keys present, route in {2, 3}, referenced body file exists.
    Returns the manifest list; raises on any violation (fail loud, nothing filed).
    """
    manifest_path = dirpath / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"manifest must be a non-empty JSON list: {manifest_path}")
    seen_slugs: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {i} is not an object")
        missing = REQUIRED_ITEM_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"manifest item {i} ({item.get('slug', '?')}): missing keys {sorted(missing)}"
            )
        if item["route"] not in (2, 3):
            raise ValueError(
                f"manifest item {i} ({item['slug']}): route must be 2 or 3, got {item['route']!r}"
            )
        if item["slug"] in seen_slugs:
            raise ValueError(f"manifest item {i}: duplicate slug {item['slug']!r}")
        seen_slugs.add(item["slug"])
        body_path = _resolve_body_path(item, dirpath)
        if not body_path.is_file():
            raise FileNotFoundError(
                f"manifest item {i} ({item['slug']}): body file not found: {body_path}"
            )
    return items


def load_ledger(dirpath: Path) -> list[dict]:
    """Read filed.jsonl rows; tolerate exactly ONE corrupt/truncated TRAILING line.

    A kill mid-append can truncate the final line: that line is quarantined to
    ``filed.jsonl.quarantined`` with a loud stderr WARN and the ledger rewritten without
    it (the in-flight item it represented is simply not terminal — normal recovery
    applies). A corrupt NON-trailing line is real corruption and raises.
    """
    path = dirpath / LEDGER_NAME
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    rows: list[dict] = []
    bad: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append((idx, line))
    if not bad:
        return rows
    if len(bad) == 1 and bad[0][0] == len(lines) - 1:
        # Truncated trailing line from a mid-append kill: quarantine + rewrite.
        idx, line = bad[0]
        with open(dirpath / QUARANTINE_NAME, "a", encoding="utf-8") as fh:
            fh.write(f"{_utc_ts()} {line}\n")
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(f"{ln}\n" for ln in lines[:-1]), encoding="utf-8")
        os.replace(tmp, path)
        print(
            f"WARNING: quarantined corrupt trailing ledger line to {dirpath / QUARANTINE_NAME}"
            " (kill mid-append; the in-flight item is not terminal and resumes normally)",
            file=sys.stderr,
        )
        return rows
    raise ValueError(
        f"corrupt non-trailing line(s) in {path} at index(es) "
        f"{[i for i, _ in bad]} — real corruption, refusing to resume"
    )


def append_row(dirpath: Path, row: dict) -> dict:
    """Single-line O_APPEND write of one ledger row, ISO-UTC ts added."""
    row = {**row, "ts": _utc_ts()}
    with open(dirpath / LEDGER_NAME, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def max_task_id(tasks_root: Path) -> int:
    """Current max task id (REGISTRY.json when present, else a tasks/*/* dir scan)."""
    ids: list[int] = []
    reg = tasks_root / "REGISTRY.json"
    if reg.is_file():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
            ids = [int(k) for k in data.get("tasks", {})]
        except (json.JSONDecodeError, ValueError):
            ids = []
    if not ids:
        ids = [int(p.name) for p in tasks_root.glob("*/*") if p.is_dir() and p.name.isdigit()]
    return max(ids, default=0)


def _read_frontmatter(body_path: Path) -> dict:
    """Parse a task body.md's YAML frontmatter (real YAML — titles are quoted by safe_dump)."""
    try:
        text = body_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    try:
        fm = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return {}
    return fm if isinstance(fm, dict) else {}


def scan_recovery_candidates(tasks_root: Path, title: str, id_floor: int, route: int) -> list[int]:
    """Task ids matching ALL of: YAML-parsed title equality, id above floor, run's route tag.

    The id floor (recorded in the slug's ``attempting`` row) scopes the match to tasks
    created by THIS filing attempt, ruling out a same-titled task from a prior night.
    """
    tag = ROUTE_RECOVERY_TAG[route]
    matches: list[int] = []
    for body in tasks_root.glob("*/*/body.md"):
        task_dir = body.parent
        if not task_dir.name.isdigit():
            continue
        tid = int(task_dir.name)
        if tid <= id_floor:
            continue
        fm = _read_frontmatter(body)
        if fm.get("title") != title:
            continue
        tags = fm.get("tags") or []
        if not isinstance(tags, list) or tag not in tags:
            continue
        matches.append(tid)
    return sorted(matches)


def find_open_fp_duplicate(tasks_root: Path, fp: str) -> Path | None:
    """First NON-terminal task body.md carrying ``wf-fix-fp:<fp>`` (route-2 dedup).

    Same predicate as the proven ad-hoc driver: a tag-scan over non-``completed``/
    ``archived`` statuses. ``task_workflow.is_open_workflow_fix_task`` is NOT usable
    here — it requires the ``workflow-fix:`` title prefix; daily filings use
    ``daily-fix:`` titles.
    """
    needle = f"wf-fix-fp:{fp}"
    for body in sorted(tasks_root.glob("*/*/body.md")):
        status = body.parent.parent.name
        if status in ("completed", "archived"):
            continue
        try:
            text = body.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in text:
            return body
    return None


def _filer_cmd(
    filer_prefix: list[str], item: dict, body_path: Path, date: str, fp: str
) -> list[str]:
    """Compose the file_infra_task.py argv for one manifest item (per-route tags)."""
    cmd = [
        *filer_prefix,
        "--kind",
        "infra",
        "--title",
        item["title"][:60],
        "--body-file",
        str(body_path),
        "--origin-prompt",
        f"/daily {date} problem sweep (route {item['route']}): {item['bug'][:400]}",
    ]
    if item["route"] == 2:
        cmd += ["--tag", "wf-fix", "--tag", f"wf-fix-fp:{fp}", "--tag", "daily-auto-filed"]
    else:
        cmd += ["--tag", "daily-held", "--tag", "needs-human", "--no-dispatch"]
    return cmd


def parse_filed_id(stdout: str, stderr: str) -> int | None:
    """Anchored id parse over stdout then stderr (a stray ``#N`` in a warning never wins)."""
    m = FILED_ID_RE.search(stdout or "")
    if m is None:
        m = FILED_ID_RE.search(stderr or "")
    return int(m.group(1)) if m else None


def _slug_state(ledger: list[dict], slug: str, retry_errors: bool) -> str:
    """Classify a slug against the ledger: 'terminal' | 'retry-error' | 'in-flight' | 'fresh'."""
    rows = [r for r in ledger if r.get("slug") == slug]
    outcomes = {r.get("outcome") for r in rows}
    if outcomes & TERMINAL_OUTCOMES:
        return "terminal"
    if "ERROR" in outcomes:
        return "retry-error" if retry_errors else "terminal"
    if rows and rows[-1].get("outcome") == "attempting":
        return "in-flight"
    return "fresh"


def _last_attempting_row(ledger: list[dict], slug: str) -> dict | None:
    for row in reversed(ledger):
        if row.get("slug") == slug and row.get("outcome") == "attempting":
            return row
    return None


def _try_recovery(
    dirpath: Path, tasks_root: Path, ledger: list[dict], item: dict, fp: str
) -> str | None:
    """Title-scan recovery for a kill-during-file window. Returns an outcome or None.

    None means "nothing recovered — proceed to dedup/file". Fires only when an
    ``attempting`` row exists (its ``id_floor`` scopes the scan). EXACTLY ONE match
    recovers (``dispatch_unconfirmed`` — the filer's tail is unknown); MULTIPLE matches
    never auto-recover (ERROR ``ambiguous-recovery`` for manual disposition).
    """
    slug = item["slug"]
    attempting = _last_attempting_row(ledger, slug)
    if attempting is None:
        return None
    id_floor = int(attempting.get("id_floor", 0))
    matches = scan_recovery_candidates(tasks_root, item["title"][:60], id_floor, item["route"])
    if not matches:
        return None
    if len(matches) == 1:
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "recovered",
                "id": matches[0],
                "fp": fp,
                "route": item["route"],
                "dispatch_unconfirmed": True,
            },
        )
        print(f"RECOVERED {slug} -> #{matches[0]}")
        return "recovered"
    append_row(
        dirpath,
        {
            "slug": slug,
            "outcome": "ERROR",
            "flag": "ambiguous-recovery",
            "id": None,
            "rc": None,
            "fp": fp,
            "route": item["route"],
            "tail": f"candidates: {matches}",
        },
    )
    print(f"ERROR {slug} (ambiguous-recovery: {matches})")
    return "error"


def process_item(
    item: dict,
    *,
    dirpath: Path,
    tasks_root: Path,
    ledger: list[dict],
    filer_prefix: list[str],
    date: str,
    root: Path,
    retry_errors: bool,
    dry_run: bool,
) -> str:
    """Run one manifest item through resume -> recovery -> dedup -> two-phase file.

    Returns the item outcome: 'skip' | 'recovered' | 'deduped' | 'filed' | 'error'.
    """
    slug = item["slug"]
    fp = wf_fix_fingerprint(item["change"], item["bug"])
    state = _slug_state(ledger, slug, retry_errors)

    if state == "terminal":
        print(f"SKIP {slug}")
        return "skip"

    if dry_run:
        if item["route"] == 2 and find_open_fp_duplicate(tasks_root, fp) is not None:
            print(f"DEDUP {slug} -> wf-fix-fp:{fp}")
        else:
            tags = _filer_cmd([], item, Path("-"), date, fp)
            pending = (
                " [in-flight attempting row; recovery scan runs first]" if state != "fresh" else ""
            )
            print(f"FILE {slug} tags={tags[tags.index('--tag') :]}{pending}")
        return "skip"

    if state in ("in-flight", "retry-error"):
        # Recovery-before-refile: the prior attempt may have committed the task
        # (kill between task.py new's commit and the ledger append; rc=0-no-id; timeout).
        outcome = _try_recovery(dirpath, tasks_root, ledger, item, fp)
        if outcome is not None:
            return outcome

    if item["route"] == 2:
        dup = find_open_fp_duplicate(tasks_root, fp)
        if dup is not None:
            append_row(
                dirpath,
                {
                    "slug": slug,
                    "outcome": "deduped",
                    "against": str(dup),
                    "fp": fp,
                    "route": item["route"],
                },
            )
            print(f"DEDUP {slug} -> {dup}")
            return "deduped"

    body_path = _resolve_body_path(item, dirpath)
    # Two-phase ledger: the `attempting` row (with the recovery id floor) lands BEFORE
    # the filer subprocess — the load-bearing crash-safety ordering.
    append_row(
        dirpath,
        {
            "slug": slug,
            "outcome": "attempting",
            "fp": fp,
            "route": item["route"],
            "id_floor": max_task_id(tasks_root),
        },
    )
    cmd = _filer_cmd(filer_prefix, item, body_path, date, fp)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(root), timeout=FILER_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        tail = (
            ((e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
            + "\n"
            + ((e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
        )
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "ERROR",
                "flag": "timeout",
                "id": None,
                "rc": None,
                "fp": fp,
                "route": item["route"],
                "tail": tail.strip()[-300:],
            },
        )
        print(f"ERROR {slug} (timeout after {FILER_TIMEOUT_S}s)")
        return "error"

    out = (proc.stdout + "\n" + proc.stderr).strip()
    tid = parse_filed_id(proc.stdout, proc.stderr)
    if proc.returncode == 0 and tid is not None:
        append_row(
            dirpath,
            {
                "slug": slug,
                "outcome": "filed",
                "id": tid,
                "rc": 0,
                "fp": fp,
                "route": item["route"],
                "tail": out[-300:],
            },
        )
        print(f"FILED {slug} -> #{tid} (rc=0)")
        return "filed"
    # rc=0 with NO parseable id is classified ERROR `no-id-parsed`, NEVER a `filed` row
    # with a null id (a null-id filed row poisons the resume set + the daily-file record).
    # The task may still have committed — --retry-errors runs recovery-before-refile.
    flag = "no-id-parsed" if proc.returncode == 0 else "filer-failed"
    append_row(
        dirpath,
        {
            "slug": slug,
            "outcome": "ERROR",
            "flag": flag,
            "id": tid,
            "rc": proc.returncode,
            "fp": fp,
            "route": item["route"],
            "tail": out[-300:],
        },
    )
    print(f"ERROR {slug} (rc={proc.returncode}, {flag})")
    if proc.returncode != 0:
        print(out[-800:], file=sys.stderr)
    return "error"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        required=True,
        help="filings dir (e.g. logs/daily/filings-2026-07-05); a relative path resolves"
        " against the REPO ROOT (never the invoking cwd)",
    )
    parser.add_argument("--start", type=int, default=None, help="manifest slice start index")
    parser.add_argument("--end", type=int, default=None, help="manifest slice end index")
    parser.add_argument(
        "--date",
        default=None,
        help="date for origin-prompt strings; default parsed from the dir basename",
    )
    parser.add_argument(
        "--filer",
        default=None,
        help="filer command override (shlex-split argv prefix); default"
        " 'uv run python scripts/file_infra_task.py' run with cwd=repo root — the test seam",
    )
    parser.add_argument(
        "--tasks-root",
        default=None,
        help="dedup/recovery scan root; default <repo_root>/tasks — the test seam",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="re-attempt slugs whose only terminal row is ERROR (recovery scan runs first)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print per-item planned action (FILE/DEDUP/SKIP); no subprocess, no ledger writes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the manifest, then drive the slice through resume/recovery/dedup/file."""
    args = build_parser().parse_args(argv)
    root = repo_root()
    dirpath = resolve_filings_dir(args.dir, root)
    date = resolve_date(args.date, dirpath)
    tasks_root = Path(args.tasks_root) if args.tasks_root else root / "tasks"
    filer_prefix = (
        shlex.split(args.filer)
        if args.filer
        else ["uv", "run", "python", "scripts/file_infra_task.py"]
    )
    manifest = load_and_validate_manifest(dirpath)
    ledger = load_ledger(dirpath)
    sliced = manifest[args.start : args.end]
    any_error = False
    for item in sliced:
        outcome = process_item(
            item,
            dirpath=dirpath,
            tasks_root=tasks_root,
            ledger=ledger,
            filer_prefix=filer_prefix,
            date=date,
            root=root,
            retry_errors=args.retry_errors,
            dry_run=args.dry_run,
        )
        if outcome == "error":
            any_error = True
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
