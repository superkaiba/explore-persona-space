---
name: refactor
description: >
  Deep structural refactoring of the codebase. Splits god files, extracts classes, eliminates
  duplication, restructures modules, and cleans up APIs. Proposes changes for approval before
  executing. Use for technical debt reduction, architectural improvements, or after major
  feature work. Heavier than /cleanup — run deliberately, not frequently.
user-invocable: true
argument-hint: "[path or --all or description of what to refactor]"
---

# Deep Refactoring Skill

Structural refactoring that goes beyond lint fixes. Proposes architectural changes, gets approval, then executes incrementally with verification.

## Arguments

- `$ARGUMENTS` — path, `--all`, or a natural language description (e.g., "split trainer.py", "reduce duplication in eval/")

## Philosophy

- `/cleanup` is the janitor — runs often, fixes small things automatically
- `/refactor` is the architect — runs deliberately, proposes structural changes, waits for approval

**This skill NEVER auto-applies changes without presenting the plan first.**

## Workflow

### Phase 1: Analysis

Thoroughly read and understand the target code. Build a mental model of:

1. **Dependency graph** — what imports what, what calls what
2. **Responsibility map** — what does each file/class/function actually do
3. **Pain points** — where are the code smells, complexity hotspots, duplication

Use Grep and Read extensively. For `--all`, prioritize by file size and complexity.

#### Metrics to Gather

For each file in scope, report:
- Lines of code (excluding blanks/comments)
- Number of functions/classes
- Max function length
- Cyclomatic complexity (from ruff C901 output)
- Number of imports (indicator of coupling)
- Number of dependents (how many other files import this one)

```bash
# Get complexity report
uv run ruff check --select C901 --output-format json $TARGET

# Get line counts
wc -l $TARGET_FILES

# Get import graph
grep -rn "^from \|^import " $TARGET_FILES
```

### Phase 2: Identify Refactoring Opportunities

Look for these patterns, ordered by impact:

#### Tier 1: High Impact (structural)

**God Files (>500 lines)**
- Split into focused modules by responsibility
- Create a package directory if splitting produces 3+ files
- Maintain backward-compatible re-exports from `__init__.py` only if external consumers exist

**God Classes (>10 methods or >300 lines)**
- Extract cohesive method groups into separate classes
- Use composition over inheritance
- Consider the Single Responsibility Principle — each class should have one reason to change

**Deep Inheritance / Tangled Hierarchies**
- Flatten unnecessary inheritance
- Replace inheritance with composition where appropriate
- Extract interfaces/protocols for polymorphism

**Circular Dependencies**
- Identify import cycles
- Break cycles by extracting shared types/interfaces into a separate module
- Use dependency inversion (depend on abstractions)

#### Tier 2: Medium Impact (code quality)

**Duplication (3+ occurrences of similar logic)**
- Extract shared utility functions
- Use parameterization to handle variations
- BUT: only deduplicate truly identical logic, not superficially similar code

**Long Functions (>60 lines)**
- Extract sub-operations into named functions
- Use early returns to reduce nesting
- Separate data gathering from data processing from side effects

**Complex Conditionals**
- Replace nested if/else with guard clauses
- Replace type-checking conditionals with polymorphism or dispatch tables
- Extract complex boolean expressions into named predicates

**Inconsistent APIs**
- Standardize function signatures across similar operations
- Use consistent naming conventions
- Ensure related functions have parallel structure

#### Tier 3: Lower Impact (polish)

**Poor Naming**
- Rename functions/classes/variables to reveal intent
- Replace abbreviations with full words
- Match domain language

**Missing Abstractions**
- Identify repeated patterns that deserve a named concept
- BUT: wait for 3+ occurrences before abstracting

**Unnecessary Complexity**
- Remove dead abstractions (interfaces with one implementation, factories that build one thing)
- Inline trivial wrapper functions
- Simplify over-engineered patterns

### Phase 3: Propose Refactoring Plan

Present findings as a structured proposal. **Do NOT start implementing yet.**

```
## Refactoring Proposal

### Scope
[What was analyzed, how many files/lines]

### Metrics Summary
| File | Lines | Functions | Max Func Len | Complexity | Imports |
|------|-------|-----------|--------------|------------|---------|
| ...  | ...   | ...       | ...          | ...        | ...     |

### Proposed Changes (ordered by impact)

#### 1. [Highest impact change]
- **What:** [Describe the structural change]
- **Why:** [What problem it solves]
- **Risk:** [Low/Medium/High — what could break]
- **Files affected:** [List]
- **Estimated scope:** [~N lines changed]

#### 2. [Next highest impact]
...

### Changes I'm NOT recommending (and why)
[Explain what you considered but rejected — shows thoroughness]

### Suggested order of execution
1. [Change X first because it unblocks Y]
2. [Then Y]
3. ...
```

**Wait for user approval before proceeding.** The user may approve all, some, or none. They may modify the plan.

### Phase 4: Execute Approved Changes

For each approved change:

1. **Create a branch** (if not already on a feature branch)
   ```bash
   git checkout -b refactor/description
   ```

2. **Execute incrementally** — one logical change at a time:
   - Make the change
   - Run `uv run ruff check --fix . && uv run ruff format .`
   - Run tests if they exist: `uv run pytest tests/ -x -q`
   - Verify no circular imports: `uv run python -c "import make_evil_dumb"`

3. **For high-risk changes** (moving files, changing public APIs):
   - Use the adversarial-planner to plan the specific change
   - Spawn an independent-reviewer subagent to verify the change
   - Check that all imports/references are updated

4. **Commit after each logical change** with a descriptive message

### Phase 5: Verification

After all changes are applied:

1. **Run full lint:** `uv run ruff check . && uv run ruff format --check .`
2. **Run tests:** `uv run pytest tests/ -x -q` (if tests exist)
3. **Import check:** `uv run python -c "import make_evil_dumb"` 
4. **Dependency check:** verify no new circular imports introduced
5. **Diff review:** `git diff --stat` to confirm only intended files changed
6. **Metrics comparison:** re-run Phase 1 metrics and show before/after

Present the verification results and before/after metrics comparison.

## Rules

- **ALWAYS propose before executing** — never start refactoring without approval
- **One logical change per commit** — makes rollback easy
- **Preserve all behavior** — refactoring must not change what the code does
- **Update all references** — when moving/renaming, grep the entire repo for stale references
- **Skip `archive/` directories** — kept for reproducibility
- **Skip `outputs/`, `eval_results/`, `figures/`** — generated content
- **For changes affecting >5 files or >200 lines**, use the adversarial-planner
- **For changes affecting public APIs**, list all callers and confirm no breakage
- **If tests don't exist for code being refactored**, flag this — don't refactor untested code without acknowledging the risk

## Anti-Patterns to Avoid

- **Refactoring for its own sake** — every change must solve a real problem
- **Premature abstraction** — don't create abstractions for hypothetical future needs
- **Big bang refactors** — break large changes into incremental steps
- **Refactoring during feature work** — refactor before or after, not during
- **Changing behavior while refactoring** — these are separate activities
- **Over-engineering** — the simplest solution that works is the best solution

## Quick Reference

```bash
# Analyze and propose refactorings for a specific file
/refactor src/make_evil_dumb/train/trainer.py

# Analyze the whole codebase
/refactor --all

# Describe what you want
/refactor "reduce duplication between alignment.py and strongreject.py"

# Target a directory
/refactor src/make_evil_dumb/eval/
```
