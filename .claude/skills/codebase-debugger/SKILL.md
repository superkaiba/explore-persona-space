---
name: codebase-debugger
description: Use when debugging problems, unexpected behavior, or when something isn't working as intended. Uses Socratic questioning to understand intent, then systematically finds mismatches between what the code should do and what it actually does.
---

# Codebase Debugger Skill

Debug by understanding intent first, then finding where reality diverges.

## The Approach

```
INTENT -> REALITY -> GAP -> ROOT CAUSE -> FIX
```

Most debugging fails because it starts with symptoms. This skill starts with **what you're trying to achieve**, then systematically finds where the code doesn't match that intent.

## Phase 1: Understand Intent (Socratic Questioning)

Before looking at any code, establish clear understanding through questions.

### Questions to Ask

**What are you trying to do?**
- What should happen when this works correctly?
- What is the expected input and output?
- What is the user-visible behavior you want?

**What is actually happening?**
- What behavior are you seeing instead?
- Is it an error, wrong result, or unexpected behavior?
- When did it start happening? What changed?

**What have you already tried?**
- What debugging have you done?
- What did you learn from those attempts?
- What hypotheses have you ruled out?

**What's the scope?**
- Does it affect all cases or specific ones?
- Is it reproducible? Under what conditions?
- Does it happen in all environments?

### Intent Statement Template

After questioning, formulate a clear intent statement:

```
INTENT: When [trigger/input], the system should [expected behavior],
        but instead it [actual behavior].
        This happens [always/sometimes/under conditions].
        It started [when/after what change].
```

## Phase 2: Map the Code Path

Once intent is clear, trace the code path that should fulfill it.

### Identify Key Components

1. **Entry point** - Where does the trigger enter the system?
2. **Data flow** - How does data move through the system?
3. **Transformations** - What operations happen on the data?
4. **Exit point** - Where does the output emerge?
5. **Side effects** - What else gets modified along the way?

### Create a Flow Diagram

```
[Input] -> [Component A] -> [Component B] -> [Component C] -> [Output]
              |               |               |
          [Side Effect]   [Database]     [External API]
```

### Questions for Each Component

- Does this component receive what it expects?
- Does it produce what the next component expects?
- Are there assumptions that might be violated?
- What error handling exists? Is it hiding problems?

## Phase 3: Find the Gap

Systematically identify where reality diverges from intent.

### Gap Analysis Checklist

**Data Issues**
- [ ] Input data malformed or unexpected type?
- [ ] Null/undefined where value expected?
- [ ] Edge cases not handled (empty, zero, negative)?
- [ ] Encoding/format issues?
- [ ] Stale or cached data?

**Logic Issues**
- [ ] Condition logic inverted or incomplete?
- [ ] Off-by-one errors?
- [ ] Race conditions or timing issues?
- [ ] Missing case in switch/if-else?
- [ ] Short-circuit evaluation hiding bugs?

**State Issues**
- [ ] State modified unexpectedly?
- [ ] State not initialized properly?
- [ ] Shared state corruption?
- [ ] State persisted when it shouldn't be (or vice versa)?

**Integration Issues**
- [ ] API contract violated?
- [ ] Version mismatch between components?
- [ ] Environment differences (dev vs prod)?
- [ ] Missing or wrong configuration?

**Assumption Issues**
- [ ] Code assumes something that isn't true?
- [ ] Documentation/comments don't match code?
- [ ] Implicit dependencies not met?

### The Five Whys

Once you find a symptom, drill down to root cause:

```
1. Why is the output wrong?
   -> The calculation uses stale data.

2. Why is the data stale?
   -> The cache isn't invalidated after updates.

3. Why isn't the cache invalidated?
   -> The update path bypasses the cache layer.

4. Why does it bypass the cache?
   -> It was a quick fix that went directly to the database.

5. Why was it done that way?
   -> No cache invalidation API existed at the time.

ROOT CAUSE: Missing cache invalidation API led to workaround
            that creates stale data.

FIX: Add cache invalidation API, update the bypass code to use it.
```

## Phase 4: Investigate Mismatches

Patterns to look for when code doesn't match intent.

### Common Mismatch Patterns

**Name vs Behavior**
```python
# Function name suggests one thing, does another
def validate_user(user):
    # Actually modifies the user object!
    user.last_validated = datetime.now()
    return True
```
*Fix: Rename to reflect behavior, or remove side effect*

**Comment vs Code**
```python
# Returns the first matching item
def find_item(items, predicate):
    return [i for i in items if predicate(i)]  # Returns ALL matches!
```
*Fix: Update comment or fix code*

**Type vs Usage**
```typescript
interface Config {
    timeout: number;  // Assumed to be milliseconds
}
// But used as seconds elsewhere
setTimeout(callback, config.timeout * 1000);  // Double conversion?
```
*Fix: Document units, or use typed wrappers*

**Error Handling vs Reality**
```python
try:
    result = risky_operation()
except Exception:
    pass  # Silently swallows ALL errors including bugs
```
*Fix: Catch specific exceptions, log others*

**Async vs Sync Assumptions**
```javascript
const data = fetchData();  // Returns Promise, not data!
process(data);  // Operating on Promise object
```
*Fix: await or .then()*

### Red Flags to Investigate

| Red Flag | Often Indicates |
|----------|-----------------|
| `# TODO` / `# FIXME` / `# HACK` | Known issues or shortcuts |
| `catch (Exception)` / `except:` | Hidden errors |
| Magic numbers | Unclear assumptions |
| Deeply nested conditionals | Missing abstractions, edge cases |
| Copy-pasted code blocks | Inconsistent fixes |
| Commented-out code | Uncertainty, dead paths |
| Very long functions | Mixed responsibilities |
| Global/shared mutable state | Race conditions, unexpected mutations |

## Phase 5: Verify and Fix

### Before Fixing

1. **Can you reproduce the bug reliably?**
   - Write a test that fails due to the bug
   - Document exact reproduction steps

2. **Do you understand the root cause?**
   - Not just what, but WHY it happens
   - Can you explain it simply?

3. **Have you considered side effects of the fix?**
   - What else might this change affect?
   - Are there similar patterns elsewhere?

### The Fix Process

```
1. Write failing test that captures the bug
2. Make minimal change to fix the test
3. Run full test suite
4. Check for similar issues elsewhere
5. Update documentation if assumptions changed
6. Consider if this reveals deeper issues
```

### After Fixing

- [ ] Original issue resolved?
- [ ] No new issues introduced?
- [ ] Test added to prevent regression?
- [ ] Root cause addressed, not just symptom?
- [ ] Similar code checked for same issue?

## Debugging Strategies by Symptom

### "It worked before"
1. What changed? (`git diff`, `git log`, `git bisect`)
2. Check dependencies (updated versions?)
3. Check environment (config, env vars?)
4. Check data (schema changes, new edge cases?)

### "It works on my machine"
1. Environment differences (OS, versions, config)
2. Data differences (local vs production data)
3. Timing differences (network latency, load)
4. Permission differences (file access, API keys)

### "It works sometimes"
1. Race conditions (async, threading)
2. Order-dependent behavior
3. Caching issues (stale data)
4. Resource limits (memory, connections)
5. External dependencies (network, APIs)

### "It fails silently"
1. Check error handling (swallowed exceptions)
2. Check logging (is the code path even reached?)
3. Check return values (nulls, empty collections)
4. Check async completion (forgotten awaits)

### "The error message doesn't make sense"
1. Error might be from different layer than you think
2. Error might be a symptom, not the cause
3. Error handling might have transformed the real error
4. Search codebase for the error message text

## Quick Reference

### Debugging Mindset
- **Assume nothing** - Verify every assumption
- **Trust the computer** - It does exactly what the code says
- **Binary search** - Narrow down systematically
- **Fresh eyes** - Explain it out loud (rubber duck)

### Key Questions
1. What should happen?
2. What actually happens?
3. Where do they diverge?
4. Why do they diverge?
5. What's the minimal fix?

### Commands for Investigation
```bash
# Find recent changes
git log --oneline -20
git diff HEAD~5

# Find the breaking commit
git bisect start
git bisect bad HEAD
git bisect good <known-good-commit>

# Search for patterns
grep -r "TODO\|FIXME\|HACK" .
grep -r "except:\|catch\s*{" .

# Check what changed in a file
git log -p --follow -- path/to/file
```

## Sources

- [TestDevLab - Root Cause Analysis Principles](https://www.testdevlab.com/blog/what-is-root-cause-analysis)
- [Bugasura - Root Cause Analysis Guide](https://bugasura.io/blog/root-cause-analysis-for-bug-tracking/)
- [Coding Horror - Rubber Duck Problem Solving](https://blog.codinghorror.com/rubber-duck-problem-solving/)
- [Wikipedia - Rubber Duck Debugging](https://en.wikipedia.org/wiki/Rubber_duck_debugging)
- [Scott Hanselman - The Art of Rubber Ducking](https://www.hanselman.com/blog/the-art-of-rubber-ducking-or-rubber-duck-debugging)
