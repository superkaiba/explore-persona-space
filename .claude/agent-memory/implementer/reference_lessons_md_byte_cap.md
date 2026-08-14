---
name: LESSONS.md byte budgets bite every index addition
description: LESSONS.md is gated at 9600 BYTES total + 280 BYTES per row (edit-time guard_lessons_edit.sh + workflow_lint --check-lessons-index); the file runs near-full, so any "fires when" addition needs same-edit trims
type: reference
---

Two layered gates on `.claude/rules/LESSONS.md` (as of 2026-08, #1269/#1279 —
supersedes the old 8000-byte single cap from #955):

- **File total: 9600 bytes** (em-dashes/×/→/§ are multibyte — deliberate).
- **Per-row: 280 bytes** (`_LESSONS_ROW_MAX_BYTES` in `scripts/workflow_lint.py`);
  grandfathered big rows (e.g. gotchas ~1175 B) sit under a separate
  `_LESSONS_ROW_GRANDFATHER_MAX_BYTES` — deliberate growth of one raises that
  constant in the SAME diff; a non-grandfathered row's only remedy is TRIM.
- Enforced TWICE: the `guard_lessons_edit.sh` PreToolUse hook BLOCKS
  Edit/Write prospectively (prints exact post-edit byte findings + largest
  rows), and commit-time `workflow_lint.py --check-lessons-index` (no-flags
  bundle) gates Bash-path writes. Escape hatches (`EPM_ALLOW_LESSONS_EDIT=1`
  or `touch .claude/cache/allow-lessons-edit`, 15 min) bypass only the hook —
  lint still fails, so genuinely fit the budget.

**How to apply:** before ANY LESSONS.md row extension, compute the byte math
in python (row bytes + file total, replace-and-measure) — never by hand. An
addition ships WITH information-preserving trims in the SAME row/edit;
2026-08-09 (#2048): file sat 2 bytes under 9600 pre-edit, a +51 B trigger
addition needed −50 B of same-row compressions to land (267 B row, 9599 B
total). The guard's error text is the authoritative recovery procedure.
