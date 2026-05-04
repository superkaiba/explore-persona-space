---
name: cleanup-weekly
description: >
  Weekly code cleanup with broader scope than /cleanup. Runs dead-code analysis
  across all modules, identifies refactoring candidates, checks dependency
  freshness, and audits .claude/ health. Manual trigger only.
user_invocable: true
---

# Weekly Cleanup

A broader-scope version of `/cleanup` intended to be run once a week. While
`/cleanup` targets recently changed code, this skill sweeps the entire repo.

## When to use

Invoke manually as `/cleanup-weekly` at the end of each work week, or when
the codebase feels cluttered. This is NOT automated — the user decides when
to run it.

## What it does

### 1. Standard cleanup (delegates to /cleanup)

Run the existing `/cleanup` skill first to handle the fast, safe stuff:
- `uv run ruff check --fix .` (lint + auto-fix)
- `uv run ruff format .` (formatting)
- Dead-code detection on recently changed files

### 2. Repo-wide dead-code analysis

Scan ALL Python modules (not just changed files):

```bash
# Find unused imports, functions, classes, variables
uv run ruff check . --select F401,F811,F841 --no-fix
```

Report findings grouped by module. Do NOT auto-fix — present a list for
the user to review (some "unused" imports are re-exports).

### 3. Refactoring candidates

Identify structural code smells:

- **Large files:** Python files > 500 lines (excluding tests, generated code)
- **Long functions:** Functions > 60 lines
- **Duplicate code blocks:** Near-identical code in 2+ places (> 10 lines)
- **Deep nesting:** Functions with > 4 levels of indentation

Report as a ranked list (largest / most severe first). Do NOT auto-refactor.

### 4. Dependency audit

```bash
# Check for outdated packages
uv pip list --outdated 2>/dev/null || echo "uv pip list not available"
```

Flag packages more than 2 major versions behind. Note any with known
security advisories.

### 5. .claude/ health check

Audit the `.claude/` directory for staleness:

- **Agent definitions** (`.claude/agents/*.md`): grep for file paths or
  function names that no longer exist in the codebase
- **Skill definitions** (`.claude/skills/*/SKILL.md`): check for broken
  references to other skills or agents
- **Plans** (`.claude/plans/`): flag plan files older than 30 days that
  reference issues now in `status:done-*`
- **Settings** (`.claude/settings.json`): check for allow rules referencing
  tools that no longer exist

### 6. Summary report

Output a structured report:

```
## Weekly Cleanup Report — YYYY-MM-DD

### Lint + Format
- N files reformatted, M lint fixes applied

### Dead Code (repo-wide)
- N unused imports across M files
- [list top 10]

### Refactoring Candidates
- N files > 500 lines
- M functions > 60 lines
- [list top 5 by severity]

### Dependencies
- N packages outdated
- [list if any]

### .claude/ Health
- N stale references found
- [list]

### Recommended Actions
1. [highest-priority action]
2. ...
```

## Rules

- This skill is READ-ONLY for analysis. It applies ruff auto-fixes (safe,
  behavior-preserving) but does NOT refactor, delete, or restructure code.
- For structural refactoring, use `/refactor` with the findings from this report.
- Do not create issues automatically — present the report to the user.
