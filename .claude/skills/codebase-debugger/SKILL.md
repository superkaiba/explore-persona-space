---
name: codebase-debugger
description: Debug problems by understanding intent first, then finding where reality diverges. Uses systematic gap analysis and root cause identification.
---

# Codebase Debugger

## Approach

```
INTENT → REALITY → GAP → ROOT CAUSE → FIX
```

Most debugging fails because it starts with symptoms. Start with **what you're trying to achieve**, then find where the code diverges.

## Phase 1: Understand Intent

Formulate a clear statement:

```
INTENT: When [trigger/input], the system should [expected behavior],
        but instead it [actual behavior].
        This happens [always/sometimes/under conditions].
        It started [when/after what change].
```

## Phase 2: Map the Code Path

1. **Entry point** — Where does the trigger enter?
2. **Data flow** — How does data move through?
3. **Transformations** — What operations happen?
4. **Exit point** — Where does output emerge?
5. **Side effects** — What else gets modified?

For each component: Does it receive what it expects? Does it produce what the next expects?

## Phase 3: Find the Gap

**Data:** Malformed input? Null where value expected? Edge cases? Stale/cached data?
**Logic:** Condition inverted? Off-by-one? Race condition? Missing case?
**State:** Unexpected mutation? Not initialized? Shared state corruption?
**Integration:** API contract violated? Version mismatch? Wrong config?
**Assumptions:** Code assumes something untrue? Comments don't match code?

### The Five Whys

Drill from symptom to root cause:
1. Why is output wrong? → Uses stale data
2. Why stale? → Cache not invalidated after update
3. Why not invalidated? → Update bypasses cache layer
4. Why bypass? → Quick fix went direct to DB
5. Why that way? → No cache invalidation API existed
→ ROOT CAUSE: Missing API led to workaround. FIX: Add API.

## Phase 4: Fix and Verify

1. Write a failing test that captures the bug
2. Make minimal change to fix the test
3. Run full test suite
4. Check for similar issues elsewhere
5. Update docs if assumptions changed

## Debugging by Symptom

**"It worked before"** → `git diff`, `git log`, `git bisect`. Check deps, env, data.
**"Works on my machine"** → Environment, data, timing, permission differences.
**"Works sometimes"** → Race conditions, order-dependent, caching, resource limits.
**"Fails silently"** → Swallowed exceptions, missing logging, forgotten awaits.
**"Error doesn't make sense"** → Error from different layer, symptom not cause, transformed by handler.

## Red Flags

| Flag | Often Indicates |
|------|----------------|
| `# TODO` / `# HACK` | Known issues |
| Bare `except:` | Hidden errors |
| Magic numbers | Unclear assumptions |
| Deep nesting | Missing abstractions |
| Commented-out code | Uncertainty |
| Global mutable state | Race conditions |
