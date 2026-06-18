---
name: posttooluse-ruff-strips-imports
description: PostToolUse hook auto-fixes ruff F401 — an import added before its usage exists gets silently stripped; add import + usage in one edit or re-check imports after the usage lands
metadata:
  type: feedback
---

When editing in this repo, a PostToolUse hook runs a formatter/ruff-fix after
every Edit/Write. An import added in its own edit BEFORE the code that uses it
exists is silently removed as F401-unused — the later edit that adds the usage
then NameErrors at runtime/test time.

**Why:** Bit me on task #586 (2026-06-11): added `list_children` to
scripts/task.py's import block first, then the `cmd_list_children` handler in
a separate edit; the hook stripped the import between the two edits and the
CLI test failed with `NameError`. The experiment-implementer twin has the same
lesson recorded (`feedback_ruff_strips_unused_imports.md`).

**How to apply:** When a change spans "import + usage", either (a) make the
usage edit FIRST and the import second, (b) put both in one Write, or (c)
after all edits land, grep the import block for every newly-referenced name
before running tests.
