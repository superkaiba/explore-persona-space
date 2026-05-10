---
name: deep-clean
description: >
  Comprehensive codebase deep clean — the nuclear option for code quality. Runs 9 analysis passes:
  lint/format, silent failure detection, dead code elimination, duplicate code detection, code smell
  audit, bug pattern detection, structural analysis, security/reliability scan, and optimization
  opportunities. Produces a prioritized action plan with severity ratings and auto-fixes what's safe.
  Use when you want to thoroughly audit and improve a codebase or module. Combines and exceeds what
  /cleanup, /refactor, and /codebase-debugger do individually. Use this skill whenever the user asks
  to "deep clean", "audit code quality", "find all issues", "clean up the codebase", "find bugs",
  "remove silent failures", "find dead code", "optimize code", or any comprehensive code quality request.
user-invocable: true
argument-hint: "[path or --all] [--fix] [--report-only] [--phase N]"
---

# Deep Clean

Comprehensive, multi-pass codebase audit that finds everything wrong and fixes what's safe.

## How This Differs from Other Skills

| Skill | Scope | This skill |
|-------|-------|------------|
| `/cleanup` | Lint + dead code + smells | All of that PLUS silent failures, duplicates, bugs, security, structure |
| `/refactor` | Structural proposals | Includes structural analysis but also covers 8 other dimensions |
| `/codebase-debugger` | Specific bug investigation | Systematic bug *pattern* detection across entire codebase |
| `/code-refactoring` | Refactoring theory reference | Actionable findings with auto-fixes, not a reference document |

## Arguments

- `$ARGUMENTS` — controls scope and behavior:
  - **No args**: files changed since last commit
  - **Path**: specific file or directory
  - **`--all`**: entire `src/` directory
  - **`--fix`**: auto-apply all safe fixes (otherwise report-only for manual fixes)
  - **`--report-only`**: skip all auto-fixes, just produce the report
  - **`--phase N`**: run only phase N (1-9) for targeted analysis
  - **`--severity critical`**: only report critical/high severity findings

## Execution

### Phase 0: Scope & Baseline

Determine targets and gather baseline metrics for before/after comparison.

```bash
# Determine scope from $ARGUMENTS
# Default: changed files since last commit
git diff --name-only HEAD -- '*.py'
git diff --name-only --cached -- '*.py'
# --all: find all Python files in src/
# path: use the specified path
```

For each file in scope, record:
- Lines of code (excluding blanks/comments)
- Number of functions/classes/methods
- Max function length
- Import count
- Cyclomatic complexity (`uv run ruff check --select C901 --output-format json`)

Store these metrics — Phase 9 will compare before/after.

If no Python files in scope, say so and stop.

---

### Phase 1: Lint & Format (auto-fix)

Automated, behavior-preserving fixes. Always run first — clears noise so later phases focus on real issues.

```bash
# 1. Auto-fix safe lint issues
uv run ruff check --fix $TARGET

# 2. Format
uv run ruff format $TARGET

# 3. Check remaining issues
uv run ruff check $TARGET --output-format json
```

**Report:** N issues auto-fixed, M remaining (list remaining by rule code).

---

### Phase 2: Silent Failure Hunt

**This is the phase no other skill does well.** Systematically find every place the code fails silently.

Search for ALL of these patterns. For each finding, assess: Is this intentional? Is there a comment explaining why? What's the blast radius if this swallows a real error?

#### 2a. Swallowed Exceptions
```python
# CRITICAL: Bare except with pass/continue
except:
    pass

# CRITICAL: Broad except with pass
except Exception:
    pass

# HIGH: Except that only logs but doesn't re-raise or handle
except Exception as e:
    logger.error(e)  # but then continues as if nothing happened

# HIGH: Except that returns a default value, hiding the failure
except (KeyError, IndexError):
    return None  # caller has no idea something went wrong

# MEDIUM: Catching too broad — masks unrelated exceptions
except Exception:  # when only ValueError is expected
```

Use Grep to systematically find all `except` blocks, then classify each one.

#### 2b. Unchecked Return Values
- Ignoring exit codes from `os.system()`, `subprocess.call()` (HIGH)
- HTTP responses never checked for errors (HIGH)
- File operations (`open()`, `shutil.copy()`) without error handling nearby (MEDIUM)

#### 2c. Defensive Defaults That Hide Bugs
- `.get()` with default on keys that should be required (HIGH)
- `or` default patterns hiding `None`/`0`/`False`/`""` (HIGH)
- `getattr(obj, "required_field", None)` (MEDIUM)

#### 2d. Missing Error Propagation
- Functions wrapping failing ops in try/except that always return success (HIGH)
- Boolean success indicators ignored by callers (HIGH)

#### 2e. Silent Data Loss
- Truncation without warning: `data[:MAX]`, `str(x)[:100]` (CRITICAL)
- Lossy conversions: `int(float)`, `str(bytes)`, `list(set())` (HIGH)

#### 2f. Incomplete Operations
- Files opened without context manager (`with`) (HIGH)
- Resources acquired but not released on error path (HIGH)
- Partial writes not rolled back on failure (MEDIUM)

For each finding, report:
- **Severity**: CRITICAL / HIGH / MEDIUM / LOW
- **File:line**
- **Pattern**: which silent failure category
- **Code snippet**: the problematic code
- **Risk**: what goes wrong when this swallows an error
- **Fix**: what the code should do instead
- **Safe to auto-fix?**: yes/no (only yes if there's zero ambiguity)

---

### Phase 3: Dead Code Elimination

Find code that serves no purpose. Go beyond what ruff catches.

#### 3a. Unused Definitions (ruff catches some)
- Unused imports (F401)
- Unused variables (F841)
- Functions defined but never called — use Grep to find def, then search for all call sites
- Classes defined but never instantiated or subclassed
- Methods that override parent but are never called polymorphically
- Constants/module-level variables never referenced

#### 3b. Unreachable Code
- Code after `return`, `break`, `continue`, `raise`, `sys.exit()`
- Branches in `if`/`elif`/`else` that can never be true (e.g., `if False:`, `if TYPE_CHECKING:` blocks that shouldn't be there)
- Functions inside `if __name__ == "__main__":` that are never used as scripts
- Dead branches in `match`/`case` statements

#### 3c. Vestigial Code
- Commented-out code blocks (3+ consecutive lines of commented code that looks like code, not documentation)
- `# TODO`/`# FIXME`/`# HACK` comments (list for triage, don't auto-remove)
- Empty `__init__.py` files that don't need to exist
- Empty function/class bodies (just `pass` or `...`) that aren't abstract
- Debug print/logging statements left behind (e.g., `print(f"DEBUG: ..."`, `logger.debug("HERE")`)
- Stale test fixtures / test helpers that no test uses

#### 3d. Obsolete Dependencies
- Imports of deprecated APIs (check for `warnings.warn` deprecation patterns)
- Compatibility shims for Python versions no longer supported

**Auto-remove:** High-confidence dead code (unused import, unreachable code after return).
**Flag for review:** Anything that could be called dynamically, via `__getattr__`, decorators, or by external code.
**Ask before removing:** Blocks >20 lines.

---

### Phase 4: Duplicate Code Detection

Find copy-pasted logic that should be unified.

#### 4a. Exact Duplicates
- Identical function bodies in different files/classes
- Identical code blocks (5+ lines) appearing 2+ times
- Identical exception handlers repeated across functions
- Identical docstring patterns that could be a decorator

#### 4b. Near Duplicates (structural clones)
- Same logic with different variable names
- Same structure with minor parameter differences (candidates for parameterization)
- Parallel implementations that should share a base (e.g., similar `train()` methods across condition classes)

#### 4c. Repeated Patterns That Deserve Abstraction
- Same sequence of operations in 3+ places (e.g., load config → validate → transform → save)
- Identical error handling patterns repeated across functions
- Same API call patterns (auth → request → check response → parse)

For each duplicate group, report:
- Locations (file:line for each occurrence)
- Similarity (exact / near / structural)
- Suggested fix (extract function, parameterize, create base class, use decorator)
- Lines that would be saved by deduplication

---

### Phase 5: Code Smell Audit

Comprehensive smell detection drawing from Fowler's catalog, PyExamine's 49 metrics, and Python-specific patterns.

#### Bloaters
- **Long Method**: functions >50 lines (WARNING), >80 lines (CRITICAL)
- **Large Class**: classes >300 lines or >15 methods
- **Long Parameter List**: functions with >4 parameters (exclude `self`/`cls`)
- **Data Clumps**: groups of 3+ variables that always appear together
- **Primitive Obsession**: using dicts/tuples where a dataclass would be clearer

#### Object-Orientation Issues
- **God Object**: class that does everything, imported by everything
- **Feature Envy**: method that accesses another object's data more than its own
- **Refused Bequest**: subclass that doesn't use most inherited methods
- **Inappropriate Intimacy**: classes accessing each other's private attributes

#### Change Preventers
- **Shotgun Surgery**: changing one concept requires editing 5+ files
- **Divergent Change**: one file changes for 3+ unrelated reasons
- **Parallel Inheritance**: adding a subclass requires adding another elsewhere

#### Dispensables
- **Lazy Class**: class with only 1-2 trivial methods (should be functions)
- **Speculative Generality**: abstractions, parameters, or classes with only one user
- **Middle Man**: class that only delegates to another

#### Complexity
- **Cyclomatic Complexity**: functions with complexity >10 (WARNING), >20 (CRITICAL)
- **Deep Nesting**: code indented 4+ levels
- **Complex Comprehensions**: list/dict comprehensions with nested loops + conditions
- **Boolean Blindness**: functions returning bare `True`/`False` where an enum would be clearer

#### Python-Specific Smells
- **Mutable Default Arguments**: `def f(x=[])` — the classic Python gotcha
- **String concatenation in loops**: use `"".join()` or f-strings
- **Using `type()` for type checking**: use `isinstance()` instead
- **`import *`**: pollutes namespace, breaks tooling
- **Nested functions 3+ levels deep**: extract to module level
- **Class that should be a module**: class with only `@staticmethod`s
- **Module that should be a package**: >500 lines with distinct sections

---

### Phase 6: Bug Pattern Detection

Find latent bugs that haven't manifested yet. These aren't style issues — they're correctness risks.

#### Type Confusion
- `isinstance` checks that miss subclasses
- Dict access on possibly-None values
- String/bytes confusion (encoding issues)
- Int/float arithmetic that could lose precision
- Comparing incompatible types (`if x == None` instead of `if x is None`)

#### Concurrency Hazards
- Global mutable state modified without locks
- Shared state between threads/processes
- Non-thread-safe operations on shared data structures
- File operations without locking in multi-process contexts

#### Resource Leaks
- Files opened without `with` context manager
- Database connections not in try/finally or context manager
- Subprocess objects not properly cleaned up
- GPU memory not released (torch tensors held by local vars in long-running functions)

#### Edge Cases
- Division without zero-check
- Array/list access without bounds check on dynamic indices
- Dict access without key existence check (use `.get()` or `in`)
- String operations on potentially empty strings
- Path operations without existence checks
- Assumptions about list ordering that might not hold

#### Logic Errors
- Off-by-one in range/slice operations
- Inverted boolean conditions (easy to miss in complex expressions)
- Short-circuit evaluation side effects
- Mutation of function arguments (especially dicts/lists)
- `is` vs `==` confusion for value comparison
- Late binding in closures/lambdas capturing loop variables

#### Python Gotchas
- Mutable default arguments
- `except` clause variable scope (deleted after block in Python 3)
- Class variable vs instance variable confusion
- `__del__` reliance for cleanup (not guaranteed to run)
- `os.environ` mutations affecting child processes unexpectedly

---

### Phase 7: Structural Analysis

Big-picture architectural health. Drawn from the `/refactor` skill's analysis.

#### Module Structure
- **God Files**: >500 lines with multiple responsibilities
- **Circular Imports**: detect import cycles using grep on import statements
- **Coupling**: modules importing 10+ other project modules
- **Cohesion**: functions in a module that don't relate to each other
- **Orphan Modules**: files not imported by anything (dead modules)

#### Dependency Health
- **Hub Modules**: imported by >50% of other modules (fragile central points)
- **Unstable Dependencies**: frequently-changed modules depended on by stable ones
- **Layer Violations**: low-level modules importing high-level ones

#### API Consistency
- **Inconsistent signatures**: similar functions with different parameter orders
- **Inconsistent naming**: `get_X` vs `fetch_X` vs `retrieve_X` for same pattern
- **Inconsistent returns**: similar functions returning different types
- **Missing `__all__`**: public modules without explicit API surface

```bash
# Dependency graph (imports)
grep -rn "^from explore_persona_space\|^import explore_persona_space" src/

# File sizes
find src/ -name "*.py" -exec wc -l {} + | sort -rn | head -20

# Complexity hotspots
uv run ruff check --select C901 --output-format json src/
```

---

### Phase 8: Security & Reliability

Not a full security audit, but catches the most common issues in research codebases.

#### Secrets & Credentials
- Hardcoded API keys, tokens, passwords in source (not in `.env`)
- Secrets in comments or docstrings
- Secrets in default parameter values
- `.env` file committed to git

Search patterns:
```
# Common secret patterns
grep -rn "api_key\s*=\s*['\"]" $TARGET
grep -rn "token\s*=\s*['\"]" $TARGET
grep -rn "password\s*=\s*['\"]" $TARGET
grep -rn "secret\s*=\s*['\"]" $TARGET
grep -rn "sk-[a-zA-Z0-9]" $TARGET    # OpenAI/Anthropic keys
grep -rn "hf_[a-zA-Z0-9]" $TARGET    # HuggingFace tokens
```

#### Unsafe Operations
- `eval()` / `exec()` on user-supplied or external data
- `pickle.load()` on untrusted data
- `yaml.load()` without `Loader=SafeLoader`
- `subprocess.shell=True` with variable input
- `os.system()` (use `subprocess.run()` instead)
- `__import__()` with dynamic strings

#### Path Safety
- Path traversal vulnerabilities (joining user input with base paths)
- Using `os.path.join` where `pathlib` would be safer
- Symlink following without checks
- Temporary file creation without `tempfile` module

#### Data Integrity
- JSON/YAML parsing without schema validation
- Assuming file encoding (should specify `encoding="utf-8"`)
- Writing files without `flush()` or `fsync()` in critical paths
- Checksum/hash verification missing for downloaded files

---

### Phase 9: Summary Report & Action Plan

Produce a comprehensive, prioritized report. This is the deliverable.

```markdown
# Deep Clean Report

## Baseline Metrics
| Metric | Before | After Phase 1 |
|--------|--------|---------------|
| Lint errors | N | M |
| Lines of code | N | N |
| Max function length | N | N |
| Files in scope | N | N |

## Findings by Severity

### CRITICAL (fix immediately — correctness/security risk)
1. [finding] — file:line — [category]
   - **Risk:** [what goes wrong]
   - **Fix:** [what to do]
2. ...

### HIGH (fix soon — reliability/maintainability risk)
1. ...

### MEDIUM (fix when convenient — code quality)
1. ...

### LOW (nice to have — polish)
1. ...

## Findings by Phase
| Phase | Critical | High | Medium | Low | Auto-fixed |
|-------|----------|------|--------|-----|------------|
| 1. Lint & Format | - | - | - | - | N |
| 2. Silent Failures | N | N | N | N | N |
| 3. Dead Code | N | N | N | N | N |
| 4. Duplicates | - | N | N | - | - |
| 5. Code Smells | N | N | N | N | - |
| 6. Bug Patterns | N | N | N | N | - |
| 7. Structure | N | N | N | N | - |
| 8. Security | N | N | N | N | - |
| **Total** | **N** | **N** | **N** | **N** | **N** |

## Top 10 Action Items (ranked by impact per effort)
1. [action] — impact: [HIGH], effort: [LOW], files: [list]
2. ...

## Auto-Fixed Summary
- [list of what was automatically fixed in this run]

## Technical Debt Estimate
- **Critical debt**: ~N hours to fix
- **High debt**: ~N hours to fix
- **Total findings**: N across M files
```

---

## Operating Rules

1. **Never change behavior** unless fixing a clear bug (and even then, flag it, don't auto-fix)
2. **Auto-fix only safe changes**: lint, format, unused imports, unreachable code after return
3. **Flag everything else** for human review with clear fix suggestions
4. **Skip directories**: `archive/`, `external/`, `outputs/`, `eval_results/`, `figures/`, `.git/`, `__pycache__/`
5. **Skip generated files**: anything that's clearly auto-generated output
6. **Respect `# noqa`, `# type: ignore`, `# pragma: no cover`** — these are intentional
7. **Ask before large removals** (>20 lines of dead code)
8. **Run ruff after every auto-fix** to verify no new issues introduced
9. **Be honest about confidence** — distinguish "definitely a bug" from "could be a problem"
10. **Don't report style preferences as bugs** — report facts, not opinions

## Quick Reference

```bash
# Deep clean changed files (default — fast, ~2 min)
/deep-clean

# Deep clean a specific module
/deep-clean src/explore_persona_space/train/

# Deep clean entire codebase (~10-20 min)
/deep-clean --all

# Deep clean with auto-fixes applied
/deep-clean --fix

# Only run the silent failure hunt
/deep-clean --phase 2

# Only critical/high severity findings
/deep-clean --severity critical

# Full audit, report only, no changes
/deep-clean --all --report-only
```
