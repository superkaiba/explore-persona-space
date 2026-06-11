---
name: workflow-lint-emit-tables
description: Pre-commit / test hard-gate — markers.md table auto-regenerates from workflow.yaml markers list; run `workflow_lint --emit-tables` after any markers edit.
metadata:
  type: reference
---

`.claude/skills/issue/markers.md` carries an auto-generated
`marker-kinds` table block derived from `workflow.yaml`'s `markers:`
list. After ANY edit that adds, removes, or reshapes a marker in
workflow.yaml — INCLUDING editing an existing marker's `fields:` prose
(confirmed 2026-06-10: enriching epm:methodology-doc-generated's note
enumeration changed the generated row) — regenerate the table:

```bash
uv run python scripts/workflow_lint.py --emit-tables
```

The tests `test_workflow_lint_check_references_exits_zero` and
`test_workflow_lint_check_tables_exits_zero` (in
`tests/test_workflow_lint.py`) hard-FAIL when the table is out of date
— so this step is non-optional in any commit that touches the markers
block.

The full validation chain after a workflow.yaml edit:

```bash
uv run python -c "import yaml; yaml.safe_load(open('.claude/workflow.yaml'))"
uv run python scripts/workflow_lint.py --emit-tables          # regenerate if needed
uv run python scripts/workflow_lint.py --check-asks
uv run python scripts/workflow_lint.py --check-references
uv run python scripts/workflow_lint.py --check-tables
uv run pytest tests/test_workflow_lint.py tests/test_workflow_yaml.py -q
```
