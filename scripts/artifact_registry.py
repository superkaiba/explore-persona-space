#!/usr/bin/env python
"""Append-only artifact-reuse registry (EPS workflow v2).

One JSON object per line at ``artifacts/registry.jsonl`` (repo root by
default). Every produced experiment artifact — activation stores, raw
completions, adapters, checkpoints, training mixes, eval JSONs, dashboards —
gets one row so the planner + methodology-writer can find fit-for-purpose
prior artifacts before retraining or regenerating (CLAUDE.md § "Reuse existing
trained artifacts", `.claude/rules/artifact-reuse.md`).

Writers append via :func:`append_artifact` (the ``upload-verifier`` v2 mode
appends one row per artifact on PASS); readers filter via
:func:`read_registry`. Concurrent appends are serialised with an ``flock`` on a
sidecar ``<registry>.lock`` and each row is written as a single ``O_APPEND``
line, so parallel VM sessions never interleave a partial record.

Fail-fast contract: a required key missing, an unknown ``type``, or a corrupt
JSONL line all raise — the registry never silently drops or skips a bad row.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Required on every row. Extra keys (e.g. ``fitness_notes``, ``sha256``,
# ``revision``) are allowed and preserved verbatim.
REQUIRED_KEYS = frozenset({"id", "type", "path", "issue", "size_bytes", "recipe"})

# The artifact taxonomy the planner / methodology-writer reason over.
VALID_TYPES = frozenset(
    {
        "activation-store",
        "raw-completions",
        "adapter",
        "checkpoint",
        "training-mix",
        "eval-json",
        "dashboard",
        "other",
    }
)


def default_registry_path() -> Path:
    """``<repo_root>/artifacts/registry.jsonl``.

    ``repo_root`` is imported lazily (it branch-guards to ``main`` and does a
    ``git`` probe) so importing this module — or passing an explicit
    ``registry_path`` — never pays that cost.
    """
    from explore_persona_space.task_workflow import repo_root

    return repo_root() / "artifacts" / "registry.jsonl"


def _resolve_registry_path(registry_path: Path | str | None) -> Path:
    if registry_path is None:
        return default_registry_path()
    return Path(registry_path)


def _validate_entry(entry: dict) -> None:
    """Raise ``ValueError`` unless ``entry`` carries every required key and a
    known ``type``. Extra keys are permitted."""
    if not isinstance(entry, dict):
        raise ValueError(f"artifact entry must be a dict, got {type(entry).__name__}")
    missing = REQUIRED_KEYS - entry.keys()
    if missing:
        raise ValueError(
            f"artifact entry missing required key(s): {sorted(missing)} "
            f"(required: {sorted(REQUIRED_KEYS)})"
        )
    art_type = entry["type"]
    if art_type not in VALID_TYPES:
        raise ValueError(f"artifact type {art_type!r} not in {sorted(VALID_TYPES)}")


def append_artifact(entry: dict, registry_path: Path | str | None = None) -> dict:
    """Validate ``entry``, stamp ``created`` (UTC ISO-8601) if absent, and
    append it as one JSON line under an exclusive ``flock``.

    Returns the stamped entry that was written. Raises ``ValueError`` on an
    invalid entry (the write never happens).
    """
    _validate_entry(entry)
    stamped = dict(entry)
    stamped.setdefault("created", datetime.now(UTC).isoformat())

    path = _resolve_registry_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")

    line = json.dumps(stamped, sort_keys=True) + "\n"
    # flock the sidecar (never the data file itself — an O_APPEND write to the
    # data file is the atomic unit, the lock only serialises concurrent
    # appenders so two lines never interleave).
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data_fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(data_fd, line.encode("utf-8"))
        finally:
            os.close(data_fd)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return stamped


def read_registry(
    registry_path: Path | str | None = None,
    issue: int | str | None = None,
    type: str | None = None,
) -> list[dict]:
    """Return the registry rows, optionally filtered by ``issue`` and/or
    ``type``.

    Tolerant of a missing file (returns ``[]``). A CORRUPT line raises
    ``ValueError`` (fail-fast — a malformed registry is a real problem, never
    silently skipped).
    """
    path = _resolve_registry_path(registry_path)
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"corrupt registry line {lineno} in {path}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(
                    f"corrupt registry line {lineno} in {path}: expected object, "
                    f"got {row.__class__.__name__}"
                )
            rows.append(row)

    if issue is not None:
        issue_str = str(issue)
        rows = [r for r in rows if str(r.get("issue")) == issue_str]
    if type is not None:
        rows = [r for r in rows if r.get("type") == type]
    return rows


def _cmd_append(args: argparse.Namespace) -> int:
    entry = json.loads(args.json)
    written = append_artifact(entry, registry_path=args.registry)
    print(json.dumps(written, sort_keys=True))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rows = read_registry(registry_path=args.registry, issue=args.issue, type=args.type)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for r in rows:
            print(
                f"{r.get('id')}\t{r.get('type')}\tissue={r.get('issue')}\t"
                f"{r.get('size_bytes')}B\t{r.get('path')}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EPS artifact-reuse registry")
    parser.add_argument(
        "--registry",
        default=None,
        help="registry path (default: <repo_root>/artifacts/registry.jsonl)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_append = sub.add_parser("append", help="append one artifact row")
    p_append.add_argument("--json", required=True, help="the artifact entry as a JSON object")
    p_append.set_defaults(func=_cmd_append)

    p_list = sub.add_parser("list", help="list artifact rows")
    p_list.add_argument("--issue", default=None, help="filter by producing issue")
    p_list.add_argument("--type", default=None, help="filter by artifact type")
    p_list.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_list.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
