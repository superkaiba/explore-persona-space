---
name: ruff-broad-check-preexisting
description: "`uv run ruff check .claude scripts` has ~1338 pre-existing errors (B905/RUF001 on experiment scripts); verify per-file + stash-compare baseline instead"
metadata:
  type: reference
---

The agent spec's §5 self-verify step "Always, after any edit: `uv run ruff
check .claude scripts`" can never PASS as written: as of 2026-06-09 the broad
run reports ~1338 pre-existing errors (B905 `zip()` strict, RUF001/RUF002
ambiguous-unicode) concentrated in experiment scripts like
`scripts/generate_sdf_variants.py` — none from workflow-surface files.

**How to apply:** lint the files you actually touched (`ruff check <paths>` +
`ruff format --check <paths>`), then for the report run the broad check twice
— `git stash` → broad check → `git stash pop` — and compare error counts to
prove zero introduced. Report "PASS (N pre-existing, 0 introduced)" rather
than a bare FAIL.

Related: [[no-agent-tool-in-spawn]] (also constrains §5/§6 as written).
