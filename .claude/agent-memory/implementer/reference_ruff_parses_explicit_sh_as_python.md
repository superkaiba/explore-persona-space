---
name: ruff-parses-explicit-sh-as-python
description: Passing a .sh path explicitly to `ruff check` makes it parse the shell script as Python and emit dozens of bogus syntax errors; normal discovery ignores .sh
metadata:
  type: reference
---

Composing a round-scoped lint command for a diff that touches BOTH shell and
Python (`uv run ruff check scripts/foo.sh tests/test_foo.py`) makes ruff treat
the explicitly-passed `.sh` as a **forced include** and parse it as Python.
Measured on `scripts/cron_codex_auto_upgrade.sh` (200 lines of bash, #2386):
`Found 165 errors`, headed by
`invalid-syntax: Simple statements must be separated by newlines or semicolons`.

The same two `.py` files alone report `All checks passed!`, and repo-wide
`ruff check .` never touches the file: ruff's normal **discovery** filters to
Python extensions, so only an explicit path forces the parse.

**How to apply:** scope round lint commands to `.py` paths only, and syntax-check
shell separately with `bash -n <script>`. If a "N errors" count appears out of
nowhere on a shell-touching round, re-run the two file classes separately before
believing it, and never report the inflated count as a round finding. Worth
stating explicitly in the results marker when it happens, since a reviewer
re-running your command sees the same 165 and needs the attribution.

Related: [[reference_preexisting_lint_test_failures]] (the separate problem of
attributing broad-ruff red that is genuinely pre-existing).
