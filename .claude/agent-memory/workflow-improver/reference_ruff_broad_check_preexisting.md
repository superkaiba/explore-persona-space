---
name: ruff-broad-check-preexisting
description: "`uv run ruff check .claude scripts` has ~1338 pre-existing errors (B905/RUF001 on experiment scripts); verify per-file + stash-compare baseline instead"
metadata:
  type: reference
---

RESOLVED in the spec 2026-06-09 (commit `16a6b57a4`): the agent spec §5 +
report template and both workflow-fix-on-bug.md success-criteria templates now
prescribe touched-files-only ruff (broader sweeps must prove failures pre-exist
on the base commit, "0 introduced"). Do NOT re-emit this candidate. The
underlying repo state still holds: a broad `uv run ruff check .claude scripts`
reports ~1338 pre-existing errors (B905 `zip()` strict, RUF001/RUF002
ambiguous-unicode) concentrated in experiment scripts like
`scripts/generate_sdf_variants.py` — none from workflow-surface files.

**How to apply:** lint the files you actually touched (`ruff check <paths>` +
`ruff format --check <paths>`), then for the report run the broad check twice
— `git stash` → broad check → `git stash pop` — and compare error counts to
prove zero introduced. Report "PASS (N pre-existing, 0 introduced)" rather
than a bare FAIL.

Related: [[no-agent-tool-in-spawn]] (also constrains §5/§6 as written).
