---
name: cleanup
description: >
  Fast, safe codebase cleanup. Runs ruff (lint+format), detects dead code, flags code smells.
  Only applies behavior-preserving auto-fixes. For structural refactoring (splitting files,
  extracting classes, reorganizing modules), use /refactor instead. Run /cleanup often.
user-invocable: true
argument-hint: "[path or --all]"
---

# Codebase Cleanup Skill

Fast, safe, automated cleanup. Fixes lint/format issues, flags dead code and smells. **Does NOT do structural refactoring** — use `/refactor` for that.

**`/cleanup`** = janitor. Run often, auto-fixes small things.
**`/refactor`** = architect. Run deliberately, proposes structural changes.

## Arguments

- `$ARGUMENTS` — optional path(s) or `--all` for entire repo. Defaults to files changed since last commit.

## Workflow

Execute these phases in order. Stop and report after each phase if there are blocking issues.

### Phase 0: Determine Scope

Decide which files to clean based on arguments:

```bash
# If $ARGUMENTS is empty or not provided: changed files since last commit
git diff --name-only HEAD -- '*.py'
git diff --name-only --cached -- '*.py'

# If $ARGUMENTS is "--all": entire repo
# If $ARGUMENTS is a path: that path only
```

If no Python files are in scope, say so and stop.

### Phase 1: Automated Fixes (Safe, Non-Breaking)

Run these tools — they only apply behavior-preserving transformations:

```bash
# 1. Lint with auto-fix (safe fixes only)
uv run ruff check --fix $TARGET_FILES

# 2. Format
uv run ruff format $TARGET_FILES

# 3. Check for remaining lint issues (report, don't fix)
uv run ruff check $TARGET_FILES
```

Report what was auto-fixed and what remains.

### Phase 2: Dead Code Detection

Search for unused code in the target files:

1. **Unused imports** — already caught by ruff F401, but double-check
2. **Unused functions/classes** — use Grep to find definitions, then search for call sites
3. **Unused variables** — already caught by ruff F841
4. **Commented-out code blocks** — search for large commented sections (3+ consecutive `#` lines that look like code, not docstrings)
5. **Empty files** — files with only imports or `pass`
6. **Stale `# TODO`/`# FIXME`/`# HACK` comments** — list them so user can triage

For each finding, report:
- File and line number
- What's unused/dead
- Confidence level (high/medium/low)
- Whether it's safe to remove (high confidence = auto-remove, medium/low = flag for review)

**Auto-remove** only high-confidence dead code (e.g., variable assigned but never read, import never used). **Flag for review** anything that could be called dynamically, via decorators, or by external consumers.

### Phase 3: Code Smell Detection

Scan the target files for these patterns. Report findings grouped by severity.

**High Priority (fix now):**
- Functions over 60 lines
- Files over 500 lines
- Deeply nested code (4+ levels of indentation)
- Hardcoded magic numbers/strings (not in constants)
- Bare `except:` or `except Exception:` with `pass`
- Mutable default arguments (`def f(x=[])`)

**Medium Priority (flag for review):**
- Functions with 5+ parameters
- Duplicate code blocks (3+ lines identical in multiple places)
- Long method chains (4+ chained calls)
- Classes with 10+ methods (god class)
- Circular imports

**Low Priority (note for future):**
- Missing type hints on public functions
- Functions returning multiple different types
- Global mutable state

### Phase 4: Structure & Organization

Check project organization:

1. **File organization** — are related functions in the same module? Any misplaced files?
2. **Import organization** — are imports sorted and grouped? (ruff I handles this)
3. **`__init__.py` exports** — do they expose the right public API?
4. **Circular dependencies** — check import chains for cycles
5. **Config vs code** — are there hardcoded values that belong in config YAML?

### Phase 5: Summary & Action Plan

Present a structured report:

```
## Cleanup Report

### Auto-Fixed (Phase 1)
- X lint issues fixed by ruff
- Y formatting issues fixed
- Z remaining issues that need manual attention

### Dead Code (Phase 2)
- Removed: [list of high-confidence removals]
- Needs Review: [list of medium/low-confidence findings]

### Code Smells (Phase 3)
- High Priority: [list with file:line references]
- Medium Priority: [list]
- Low Priority: [list]

### Structure (Phase 4)
- [any organizational recommendations]

### Suggested Next Steps
1. [most impactful improvement]
2. [second most impactful]
3. ...
```

## Rules

- **Never change behavior** — all changes must be strictly behavior-preserving
- **Run ruff after every edit** — verify no new lint issues introduced
- **Commit incrementally** — if making multiple changes, keep them in logical groups
- **Ask before large removals** — if removing >20 lines of dead code, confirm with user first
- **Respect `# noqa` and `# type: ignore`** — these are intentional suppressions
- **Skip test files** unless explicitly included — test code has different quality standards
- **Skip `archive/` directories** — these are kept for reproducibility
- **Skip generated files** — anything in `outputs/`, `eval_results/`, `figures/`

## Quick Reference

```bash
# Run cleanup on changed files (default)
/cleanup

# Run cleanup on specific file
/cleanup src/make_evil_dumb/train/trainer.py

# Run cleanup on entire repo
/cleanup --all

# Run cleanup on a directory
/cleanup src/make_evil_dumb/data/
```
