#!/usr/bin/env python3
"""migrate_add_legacy_why_sentinel.py — one-shot backfill.

Adds the ``legacy_why_unset: true`` sentinel to YAML frontmatter on
every task body authored before the ``## Why this experiment`` gate
landed (task #371). With the sentinel present, ``verify_task_body``
check #12 PASSes for grandfathered bodies; the gate only fires on
bodies authored after the migration.

Scope (per task #371 body):

* Walks ``tasks/<status>/<N>/body.md`` for every status EXCEPT
  ``proposed``. The 53 proposed tasks remain ungated so the gate fires
  the next time the user actually tries to dispatch them — that's the
  point of the friction.
* Idempotent: if a body already carries ``legacy_why_unset: true`` we
  skip it (so re-running ``--apply`` is safe).
* Bodies without YAML frontmatter (legacy / malformed) are skipped with
  a warning — the migration won't invent frontmatter for them.

Default mode is ``--dry-run`` (prints the count + a sample diff). Pass
``--apply`` to write the change. ``--apply`` makes a SINGLE atomic git
commit covering every patched body so rollback is one ``git revert``.

Usage::

    uv run python scripts/migrate_add_legacy_why_sentinel.py [--dry-run]
    uv run python scripts/migrate_add_legacy_why_sentinel.py --apply

Exits 0 on success, 1 if any body fails to parse, 2 on usage error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Bring the task_workflow module in for paths + frontmatter helpers.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

# Sentinel key written into frontmatter. Mirrors
# ``verify_task_body.LEGACY_WHY_SENTINEL_KEY``. Edit both together if
# changing.
LEGACY_WHY_SENTINEL_KEY = "legacy_why_unset"

# Statuses to walk. ``proposed`` is intentionally EXCLUDED so the gate
# fires on first-use of any pre-existing proposed task.
SCOPED_STATUSES = (
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

# Frontmatter delimiter
_FM_OPEN = "---\n"
_FM_CLOSE = "\n---\n"


def repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"could not find .git starting from {__file__}")


REPO = repo_root()
TASKS_DIR = REPO / "tasks"


def split_frontmatter(text: str) -> tuple[dict | None, str, str]:
    """Return (frontmatter_dict, fm_block_raw, body_str).

    ``frontmatter_dict`` is ``None`` if the body lacks YAML frontmatter
    or it fails to parse. ``fm_block_raw`` is the raw YAML text between
    the ``---`` markers (without the markers themselves).
    """
    if not text.startswith(_FM_OPEN):
        return None, "", text
    rest = text[len(_FM_OPEN) :]
    end = rest.find(_FM_CLOSE)
    if end == -1:
        return None, "", text
    fm_block = rest[:end]
    body = rest[end + len(_FM_CLOSE) :]
    try:
        fm = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        return None, fm_block, body
    if not isinstance(fm, dict):
        return None, fm_block, body
    return fm, fm_block, body


def join_frontmatter(fm: dict, body: str) -> str:
    fm_block = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{fm_block}\n---\n{body}"


def iter_task_bodies() -> list[Path]:
    """Yield body.md paths under tasks/<status>/<N>/ for SCOPED_STATUSES."""
    out: list[Path] = []
    for status in SCOPED_STATUSES:
        status_dir = TASKS_DIR / status
        if not status_dir.is_dir():
            continue
        for task_dir in sorted(status_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            body_path = task_dir / "body.md"
            if body_path.is_file():
                out.append(body_path)
    return out


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="(default) report what would change without writing",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="write the change to disk and commit atomically",
    )
    parser.add_argument(
        "--show-sample",
        type=int,
        default=1,
        help="number of sample diffs to print in dry-run mode (default: 1)",
    )
    args = parser.parse_args()
    apply = bool(args.apply)

    bodies = iter_task_bodies()
    if not bodies:
        print(
            "migrate_add_legacy_why_sentinel: no task bodies under "
            f"{TASKS_DIR}/{{ {','.join(SCOPED_STATUSES)} }} — nothing to do",
            file=sys.stderr,
        )
        return 0

    to_patch: list[Path] = []
    already_set: list[Path] = []
    skipped_no_fm: list[Path] = []
    parse_errors: list[Path] = []
    sample_diffs: list[tuple[Path, str, str]] = []

    for body_path in bodies:
        try:
            raw = body_path.read_text()
        except OSError as e:
            print(f"  ERROR reading {body_path}: {e}", file=sys.stderr)
            parse_errors.append(body_path)
            continue
        fm, _fm_raw, body = split_frontmatter(raw)
        if fm is None:
            skipped_no_fm.append(body_path)
            continue
        if fm.get(LEGACY_WHY_SENTINEL_KEY) is True:
            already_set.append(body_path)
            continue
        to_patch.append(body_path)
        if len(sample_diffs) < args.show_sample:
            new_fm = dict(fm)
            new_fm[LEGACY_WHY_SENTINEL_KEY] = True
            new_text = join_frontmatter(new_fm, body)
            sample_diffs.append((body_path, raw, new_text))

    print(f"Migration scope ({'APPLY' if apply else 'DRY-RUN'}):")
    print(f"  bodies walked      : {len(bodies)}")
    print(f"  would patch        : {len(to_patch)}")
    print(f"  already sentinelled: {len(already_set)}")
    print(f"  skipped (no FM)    : {len(skipped_no_fm)}")
    print(f"  parse errors       : {len(parse_errors)}")
    print()

    if skipped_no_fm:
        print(f"Skipped {len(skipped_no_fm)} bodies that lack YAML frontmatter:")
        for p in skipped_no_fm[:5]:
            print(f"    - {p.relative_to(REPO)}")
        if len(skipped_no_fm) > 5:
            print(f"    … and {len(skipped_no_fm) - 5} more")
        print()

    if parse_errors:
        print(
            f"ERROR: {len(parse_errors)} bodies failed to read or parse — "
            "fix these before running --apply",
            file=sys.stderr,
        )
        for p in parse_errors:
            print(f"    - {p.relative_to(REPO)}", file=sys.stderr)
        return 1

    if sample_diffs and not apply:
        print("Sample diff (first body that would be patched):")
        print("-" * 60)
        body_path, old_text, new_text = sample_diffs[0]
        print(f"  {body_path.relative_to(REPO)}")
        # Show frontmatter region of both for clarity.
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        # Walk to find the second `---` line on each side.
        def _fm_slice(lines: list[str]) -> list[str]:
            if not lines or lines[0] != "---":
                return lines[:5]
            for idx in range(1, len(lines)):
                if lines[idx] == "---":
                    return lines[: idx + 1]
            return lines[:5]

        for line in _fm_slice(old_lines):
            print(f"    - {line}")
        for line in _fm_slice(new_lines):
            print(f"    + {line}")
        print()

    if not apply:
        print("Re-run with --apply to write the changes (one atomic commit).")
        return 0

    if not to_patch:
        print("Nothing to apply.")
        return 0

    # Apply mode: rewrite each body, then a single git commit.
    for body_path in to_patch:
        raw = body_path.read_text()
        fm, _, body = split_frontmatter(raw)
        assert fm is not None  # checked in the dry-run loop above
        new_fm = dict(fm)
        new_fm[LEGACY_WHY_SENTINEL_KEY] = True
        new_text = join_frontmatter(new_fm, body)
        body_path.write_text(new_text)

    # Stage all touched bodies and commit as one atomic change.
    rel_paths = [str(p.relative_to(REPO)) for p in to_patch]
    subprocess.run(
        ["git", "-C", str(REPO), "add", *rel_paths],
        check=True,
    )
    msg = "task-workflow: backfill legacy_why_unset sentinel on pre-gate bodies (#371)"
    subprocess.run(
        ["git", "-C", str(REPO), "commit", "-m", msg],
        check=True,
    )
    print(f"\nApplied to {len(to_patch)} bodies, one atomic commit. Rollback: `git revert HEAD`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
