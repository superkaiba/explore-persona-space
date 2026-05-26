"""Strip the `## Why this experiment` section from a task body.md.

Removes from the `## Why this experiment` line up to (but not including) the
next `## ` H2 line, or to end of file if it's the last H2. Preserves any
trailing newline at the end of the file.

Usage:
    uv run python scripts/strip_why_this_experiment.py [--dry-run] PATH [PATH...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WHY_HEADER_RE = re.compile(r"^## Why this experiment\s*$")
ANY_H2_RE = re.compile(r"^## \S")


def strip_section(text: str) -> tuple[str, bool]:
    """Return (new_text, changed). Strip from '## Why this experiment' to
    next '## ' H2 (exclusive) or to EOF."""
    lines = text.splitlines(keepends=True)
    start_idx = next(
        (i for i, ln in enumerate(lines) if WHY_HEADER_RE.match(ln.rstrip("\n"))),
        None,
    )
    if start_idx is None:
        return text, False

    end_idx = next(
        (i for i in range(start_idx + 1, len(lines)) if ANY_H2_RE.match(lines[i])),
        len(lines),
    )

    # Trim trailing blank lines BEFORE the deletion target so we don't leave
    # a double-blank gap above the next section.
    keep_until = start_idx
    while keep_until > 0 and lines[keep_until - 1].strip() == "":
        keep_until -= 1

    new_lines = lines[:keep_until] + lines[end_idx:]
    new_text = "".join(new_lines)
    # Ensure exactly one trailing newline.
    new_text = new_text.rstrip("\n") + "\n"
    return new_text, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed_any = False
    for p in args.paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        original = p.read_text()
        new, changed = strip_section(original)
        if not changed:
            print(f"no-op: {p}")
            continue
        changed_any = True
        delta = len(original) - len(new)
        print(f"strip:  {p}  (-{delta} bytes)")
        if not args.dry_run:
            p.write_text(new)
    return 0 if changed_any or not args.paths else 1


if __name__ == "__main__":
    sys.exit(main())
