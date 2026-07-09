#!/usr/bin/env python
"""Per-check timing harness for ``scripts/workflow_lint.py``'s no-flags run (#1163).

Times each check that ``main()`` bundles into the no-flags default run, one at
a time, against the live tree the given lint copy roots at (its
``__file__``-derived ``_REPO_ROOT``). Used to attribute the pre-fix ~273s
per-check sum (145s ``check_no_workflow_improver_spawn`` rglob + ~122s of
duplicated AST parsing) and to verify the post-fix targets (pruned check
<= 3s; the parse-bearing checks' sum <= 45s — the FIRST parse-bearing check
pays the single shared-parse cost, later ones ride ``_AST_CACHE``).

The no-flags check ladder is extracted MECHANICALLY from ``main()``'s source
(the ``if args.<flag> or no_flags:`` -> ``errors.extend(<fn>(...))`` pairs),
so the harness cannot rot when a check is added to the bundle.

Usage:
    uv run python scripts/issue1163_time_lint_checks.py [path/to/workflow_lint.py]

Defaults to the ``workflow_lint.py`` sitting next to this script. Prints a
per-check table (seconds, findings count) + total + /proc/loadavg, and exits
non-zero if any check raised.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path

# `if args.<flag> or no_flags:` (optionally wrapped in parens with an extra
# clause, e.g. the marker-registry `and not args.check_references`) followed by
# `errors.extend(<check_fn>(<optional workflow arg>))`.
_LADDER_RE = re.compile(
    r"if \(?args\.\w+ or no_flags\)?[^\n]*:\n\s+errors\.extend\((\w+)\((workflow)?\)\)"
)


def _load_lint_module(lint_path: Path):
    """Import the given workflow_lint.py copy under a scratch module name.

    ``sys.modules`` registration is required before ``exec_module`` so the
    module's own dataclasses/typing resolution works.
    """
    spec = importlib.util.spec_from_file_location("workflow_lint_timed_1163", lint_path)
    assert spec is not None and spec.loader is not None, lint_path
    mod = importlib.util.module_from_spec(spec)
    sys.modules["workflow_lint_timed_1163"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    """Time every no-flags check of the given lint copy; print a table."""
    args = sys.argv[1:] if argv is None else argv
    lint_path = (
        Path(args[0]).resolve() if args else Path(__file__).resolve().parent / "workflow_lint.py"
    )
    mod = _load_lint_module(lint_path)
    source = lint_path.read_text(encoding="utf-8")
    ladder = _LADDER_RE.findall(source)
    if not ladder:
        sys.stderr.write("issue1163_time_lint_checks: no-flags ladder not found in main()\n")
        return 2

    workflow = mod.load_workflow_yaml(None)
    print(f"# lint copy: {lint_path}")
    print(f"# repo root: {mod._REPO_ROOT}")
    print(f"# loadavg (start): {Path('/proc/loadavg').read_text().strip()}")
    rows: list[tuple[str, float, int]] = []
    failed = False
    t_all = time.perf_counter()
    for fn_name, needs_workflow in ladder:
        fn = getattr(mod, fn_name)
        t0 = time.perf_counter()
        try:
            findings = fn(workflow) if needs_workflow else fn()
        except Exception as exc:  # report + continue so one crash doesn't hide the table
            print(f"{fn_name:55s} RAISED {type(exc).__name__}: {exc}")
            failed = True
            continue
        rows.append((fn_name, time.perf_counter() - t0, len(findings)))
    total = time.perf_counter() - t_all
    for fn_name, secs, n in sorted(rows, key=lambda r: -r[1]):
        print(f"{fn_name:55s} {secs:8.2f}s  findings={n}")
    print(f"{'TOTAL (' + str(len(rows)) + ' checks)':55s} {total:8.2f}s")
    print(f"# loadavg (end): {Path('/proc/loadavg').read_text().strip()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
