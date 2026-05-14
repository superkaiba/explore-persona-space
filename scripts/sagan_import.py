#!/usr/bin/env python3
"""sagan_import.py — one-shot importer from Sagan to the task workflow.

Pulls every experiment row + every workflow_event from
`sagan.superkaiba.com` and materialises them as local files under
`tasks/<status>/<id>/`. After this runs, Sagan stays accessible for
historical viewing but is no longer the workflow substrate.

Usage:

    # Dry-run: count what would be imported, write nothing
    uv run python scripts/sagan_import.py --dry-run

    # Real import. Batches commits every BATCH_SIZE experiments.
    uv run python scripts/sagan_import.py --batch-size 50

    # Import only a single experiment by number (useful for testing)
    uv run python scripts/sagan_import.py --only 311

The importer is idempotent at the experiment level: re-importing an
experiment that already exists on disk will REFUSE to overwrite unless
`--force` is set. So `--only N` lets you preview shape for a single
experiment.

Schema-translation rules:

* Sagan `experiment.status` (snake_case enum) maps 1:1 to the task
  workflow folder name. Statuses that exist in Sagan but not in
  STATUSES go to `archived/` with a warning logged.

* Sagan `workflow_events.marker_type` (`metadata.marker_type` on older
  rows) becomes `events.jsonl` row's `kind`. Both shapes are accepted.

* Sagan-card HTML bodies are imported as-is; a `<!-- legacy-sagan-card -->`
  sentinel is prepended so `verify_task_body.py` knows to skip the
  markdown-spec checks on grandfathered bodies.

* Comments — Sagan has no comments table today, so comments.jsonl is
  written empty.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Importable module path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Sagan API helpers (read-only path)
sys.path.insert(0, str(_HERE))
import sagan_state  # noqa: E402

import explore_persona_space.task_workflow as tw  # noqa: E402

# ─── Status mapping ────────────────────────────────────────────────────────

# Sagan's enum already uses the same snake_case names we use; this map is
# defensive in case any legacy values surface.
SAGAN_TO_TASK_STATUS: dict[str, str] = {
    "proposed": "proposed",
    "planning": "planning",
    "plan_pending": "plan_pending",
    "approved": "approved",
    "running": "running",
    "verifying": "verifying",
    "interpreting": "interpreting",
    "reviewing": "reviewing",
    "awaiting_promotion": "awaiting_promotion",
    "completed": "completed",
    "done": "completed",  # legacy alias
    "blocked": "blocked",
    "archived": "archived",
    # The pre-migration status names used elsewhere in the codebase
    "clean_result_drafting": "interpreting",
    "in_flight": "running",
}

LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->\n"


# ─── Import logic ──────────────────────────────────────────────────────────


@dataclass
class ImportStats:
    fetched: int = 0
    imported: int = 0
    skipped_existing: int = 0
    legacy_html: int = 0
    by_status: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_status is None:
            self.by_status = Counter()


def _all_experiments() -> list[dict[str, Any]]:
    """Pull every experiment from Sagan API.

    The API caps `limit` at 200 and has no offset/cursor pagination, so
    we iterate per-status. Sagan's enum statuses plus the legacy aliases
    are queried; results are deduped by id. No single status has more
    than 200 rows in our deployment, so this gets everything.
    """
    seen: dict[str, dict[str, Any]] = {}
    statuses_to_query = list(SAGAN_TO_TASK_STATUS.keys())
    for status in statuses_to_query:
        try:
            out = sagan_state._req(
                "GET", "/api/experiments", query={"status": status, "limit": 200}
            )
        except sagan_state.SaganError as e:
            # Unknown status enum value → skip silently
            if "400" in str(e):
                continue
            raise
        for row in out.get("experiments", []):
            rid = row.get("id")
            if rid:
                seen[rid] = row
    return list(seen.values())


def _detail_for(experiment_id: str) -> dict[str, Any]:
    """Fetch detailed experiment + events for one row."""
    return sagan_state.get_experiment_by_id(experiment_id)


def _translate_status(sagan_status: str | None) -> str:
    if not sagan_status:
        return "proposed"
    mapped = SAGAN_TO_TASK_STATUS.get(sagan_status)
    if mapped is None:
        print(
            f"[warn] unknown Sagan status {sagan_status!r}; routing to 'archived'",
            file=sys.stderr,
        )
        return "archived"
    return mapped


def _is_sagan_card_html(body: str) -> bool:
    """Heuristic: Sagan-card bodies open with <style> or <section id="tldr">."""
    head = body.lstrip()[:200].lower()
    return (
        head.startswith("<style") or '<section id="tldr"' in head or '<details id="design"' in head
    )


def _build_frontmatter(exp: dict[str, Any]) -> dict[str, Any]:
    fm: dict[str, Any] = {
        "title": exp.get("title", "") or "",
        "kind": exp.get("kind", "experiment") or "experiment",
        "tags": list(exp.get("tags") or []),
        "created_at": exp.get("createdAt") or exp.get("created_at") or "",
        "has_clean_result": bool(exp.get("hasCleanResult") or exp.get("has_clean_result")),
        "sagan_id": exp.get("id"),
        "sagan_number": exp.get("number"),
    }
    if exp.get("podName"):
        fm["pod_name"] = exp["podName"]
    if exp.get("parentId"):
        fm["parent_id"] = exp["parentId"]
    if exp.get("classification"):
        fm["classification"] = exp["classification"]
    if exp.get("priority"):
        fm["priority"] = exp["priority"]
    return fm


def _translate_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Translate a Sagan workflow_event row to the task workflow events.jsonl shape."""
    meta = ev.get("metadata") or {}
    marker = meta.get("marker_type") or ev.get("markerType") or ev.get("eventType") or "epm:unknown"
    payload: dict[str, Any] = {
        "ts": ev.get("createdAt") or ev.get("created_at") or "",
        "kind": marker,
        "version": meta.get("version") or 1,
        "by": ev.get("actorKind") or "sagan-import",
    }
    if ev.get("note"):
        payload["note"] = ev["note"]
    from_status = ev.get("fromStatus") or ev.get("from_status")
    to_status = ev.get("toStatus") or ev.get("to_status")
    if from_status:
        payload["from"] = from_status
    if to_status:
        payload["to"] = to_status
    # Preserve interesting metadata keys but skip noise
    for k, v in meta.items():
        if k in {"marker_type", "version"}:
            continue
        payload.setdefault(k, v)
    return payload


def _import_one(
    exp_summary: dict[str, Any], *, force: bool, stats: ImportStats
) -> tuple[int | None, str | None]:
    """Import a single experiment. Returns (task_id, status) or (None, None) if skipped."""
    number = exp_summary.get("number")
    if number is None:
        print(f"[warn] experiment missing number, skipping: {exp_summary.get('id')}")
        return None, None
    sagan_id = exp_summary["id"]
    detail = _detail_for(sagan_id)
    exp = detail.get("experiment") or exp_summary
    events = detail.get("events") or []

    status = _translate_status(exp.get("status"))
    body = exp.get("body") or ""
    is_legacy_html = _is_sagan_card_html(body)
    if is_legacy_html:
        stats.legacy_html += 1
        if not body.startswith(LEGACY_SAGAN_CARD_SENTINEL):
            body = LEGACY_SAGAN_CARD_SENTINEL + body

    target_dir = tw.TASKS_DIR / status / str(number)
    if target_dir.exists():
        if not force:
            stats.skipped_existing += 1
            return None, None
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "artifacts").mkdir(exist_ok=True)
    (target_dir / "plans").mkdir(exist_ok=True)

    fm = _build_frontmatter(exp)
    body_norm = body if body.endswith("\n") else body + "\n"
    tw._write_body(target_dir / "body.md", fm, body_norm)

    # events.jsonl
    ev_path = target_dir / "events.jsonl"
    with ev_path.open("w") as f:
        for ev in events:
            payload = _translate_event(ev)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # Empty comments.jsonl
    (target_dir / "comments.jsonl").touch()

    stats.imported += 1
    stats.by_status[status] += 1
    return int(number), status


def _commit_batch(repo: Path, message: str) -> None:
    """Stage all changes under tasks/ and commit. Skips if nothing changed."""
    if os.environ.get("TASK_PY_NO_COMMIT") == "1":
        return
    subprocess.run(["git", "add", "tasks"], cwd=repo, check=True)
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo, check=False)
    if result.returncode == 0:
        return
    subprocess.run(
        ["git", "commit", "-m", message + "\n\n[task.py / sagan-import]"],
        cwd=repo,
        check=True,
    )


def _rebuild_registry() -> None:
    """Reconstruct REGISTRY.json from the on-disk task tree (post-import)."""
    reg: dict[str, Any] = {"highest_id": 0, "tasks": {}}
    if not tw.TASKS_DIR.exists():
        tw._save_registry(reg)
        return
    for status_dir in sorted(tw.TASKS_DIR.iterdir()):
        if not status_dir.is_dir() or status_dir.name not in tw.STATUSES:
            continue
        for child in status_dir.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            task_id = int(child.name)
            try:
                fm, _ = tw._read_body(child / "body.md")
            except (FileNotFoundError, ValueError) as e:
                print(f"[warn] cannot read frontmatter for {child}: {e}", file=sys.stderr)
                continue
            tw._registry_set(reg, task_id, child, fm)
    tw._save_registry(reg)
    print(f"[ok] REGISTRY.json: {len(reg['tasks'])} tasks, highest_id={reg['highest_id']}")


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="count what would be imported, write nothing"
    )
    parser.add_argument(
        "--only", type=int, default=None, help="import only the experiment with this number"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="commit every N imported experiments"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing task folders")
    parser.add_argument("--no-commit", action="store_true", help="skip git commit at the end")
    args = parser.parse_args()

    if args.no_commit:
        os.environ["TASK_PY_NO_COMMIT"] = "1"

    print(f"[info] Sagan base URL: {sagan_state.BASE_URL}")
    print(f"[info] target dir:     {tw.TASKS_DIR}")
    print(f"[info] dry-run:        {args.dry_run}")
    if args.only:
        print(f"[info] only:           #{args.only}")
    print()

    # 1. Fetch index
    summaries = _all_experiments()
    print(f"[ok] fetched {len(summaries)} experiment summaries from Sagan")

    if args.only is not None:
        summaries = [s for s in summaries if s.get("number") == args.only]
        if not summaries:
            print(f"[err] experiment #{args.only} not found")
            sys.exit(1)

    if args.dry_run:
        by_status: Counter[str] = Counter()
        for s in summaries:
            by_status[_translate_status(s.get("status"))] += 1
        print("[dry-run] per-status counts:")
        for status, count in sorted(by_status.items()):
            print(f"  {status:<22}  {count}")
        max_num = max((s.get("number") or 0) for s in summaries) if summaries else 0
        print(f"[dry-run] highest experiment number: {max_num}")
        print("[dry-run] no files written")
        return

    # 2. Import each
    stats = ImportStats()
    stats.fetched = len(summaries)
    repo = tw.REPO

    # Sort by number ascending so commits land in deterministic order
    summaries.sort(key=lambda s: s.get("number") or 0)

    batch_count = 0
    last_batch_first_num: int | None = None
    last_batch_last_num: int | None = None

    for s in summaries:
        task_id, status = _import_one(s, force=args.force, stats=stats)
        if task_id is None:
            continue
        if last_batch_first_num is None:
            last_batch_first_num = task_id
        last_batch_last_num = task_id
        batch_count += 1
        if batch_count >= args.batch_size:
            _commit_batch(
                repo,
                f"sagan-import: experiments #{last_batch_first_num}-#{last_batch_last_num} ({batch_count} rows)",
            )
            batch_count = 0
            last_batch_first_num = None

    if batch_count > 0:
        _commit_batch(
            repo,
            f"sagan-import: experiments #{last_batch_first_num}-#{last_batch_last_num} ({batch_count} rows)",
        )

    # 3. Rebuild REGISTRY.json from disk, commit it
    _rebuild_registry()
    _commit_batch(repo, "sagan-import: REGISTRY.json")

    # 4. Audit
    problems = tw.audit()
    print()
    print("[summary]")
    print(f"  fetched:           {stats.fetched}")
    print(f"  imported:          {stats.imported}")
    print(f"  skipped existing:  {stats.skipped_existing}")
    print(f"  legacy HTML bodies: {stats.legacy_html}")
    print("  per status:")
    for status, count in sorted(stats.by_status.items()):
        print(f"    {status:<22}  {count}")
    print()
    if problems:
        print(f"[warn] audit found {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  (+{len(problems) - 20} more)")
        sys.exit(1)
    print("[ok] audit PASS — registry and filesystem agree")


if __name__ == "__main__":
    main()
