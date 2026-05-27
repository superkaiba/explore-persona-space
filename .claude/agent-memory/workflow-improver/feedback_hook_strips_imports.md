---
name: hook-strips-imports
description: A PostToolUse formatter hook on this repo can silently remove imports between Edit calls; verify after every Edit when adding library APIs + CLI bindings in sequence.
metadata:
  type: feedback
---

A PostToolUse hook (likely `ruff --fix` or similar) runs after Edits in this
project and CAN remove imports that look unused at the moment the edit
lands — even when the next planned edit will use them.

**Why:** When implementing the binding-concerns workflow change
(2026-05-27), the first edit to `scripts/task.py` added new entries to
the `from explore_persona_space.task_workflow import (...)` block. The
hook stripped them because the matching function bodies hadn't been added
yet. The second edit added the function bodies, which then failed
`ruff check` with `F821 Undefined name`. Re-applied the import edit;
worked. Same pattern hit `src/.../task_workflow.py` when I added
`import re` for a regex used later in the same edit batch — the second
hook pass stripped it because the chunk using `re.compile(...)` was
appended via a later Edit, not in the same diff.

**How to apply:** When sequencing edits that add (a) imports and (b)
function bodies that consume them, EITHER bundle them in a single Edit
call OR re-verify with `uv run ruff check <path>` after each edit and
re-add stripped imports. Don't trust the imports to survive across
multi-Edit sessions in this repo.

Related: [[branch-guard-blocks-subprocess]] for another pitfall when
testing task.py changes.
