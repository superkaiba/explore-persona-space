---
name: claude-degenerate-token-validation-and-files-mode-scope
description: On a NEW lint/audit gate diff, probe degenerate schema tokens (empty-string containment = universal pass) and diff the files-mode CHECK_SCOPES surface against the check's own enumeration surface — Claude PASSed both gaps (#2568 r1)
metadata:
  type: feedback
---

Two Claude code-review misses on one new-fleet-gate diff (#2568 r1, audit tool + workflow_lint check; Codex FAILed, reconciler upheld 3 of 4 blockers):

1. **Degenerate-token validation gap.** A schema validator that checks only
   `list-of-str` lets `[""]` through, and a downstream membership arm
   `any(r in content for r in items)` is then universally True (`"" in s`).
   One malformed-but-valid record silently converted the whole gate to
   pass-everything with zero warnings. Siblings in the same diff: a
   whitespace-only string passes a `len(n) >= K` floor and a truthiness check
   (`isinstance(p, str) and p` — `" "` is truthy) and then substring-matches
   nearly every line. Claude tested the empty-LIST arm and missed the
   empty-ENTRY arm.
   **How to apply:** for every new validation gate, mentally construct the
   degenerate satisfiers — `""`, whitespace-only, bool-for-int — and trace
   each into every containment/threshold consumer. Sibling of
   [[claude-syntactic-test-pins-and-vacuous-empty-gates]] (empty-SELECTION);
   this is the empty-TOKEN face.

2. **Files-mode CHECK_SCOPES surface under-inclusion.** A `global` check's
   declared surface list is a SKIP predicate in `workflow_lint --files`
   mode (a binding gate surface via `inline_lint_gate.py` scoped routing).
   Diff the declared surfaces against the check's OWN enumeration surface
   (here: `git grep --cached -- .` minus 4 excludes = everything, but the
   surface list omitted `.github/`, `papers/`, `archive/`, root files
   `RESULTS.md`/`README.md`, etc.). A code comment claiming the set is
   "drawn WIDE" with N deliberate absences is falsifiable by listing the
   top-level tree — verify it, don't credit it.

Also: reconciler ledger mechanics that worked — when the orchestrator already
forwarded the Codex `CONCERN::` rows to concerns.jsonl, do NOT re-raise
identical ids (duplicates); adopt them by id in the persistence accounting,
raise only NEW residuals, and `defer-concern --by reconciler` the overturned
row with the adjudication rationale.

**Why:** a new fence that can be silently hollowed (validation gap) or
silently skipped (scope gap) is the new-fence-silent-pass class; both fixes
were one-liners the revise round absorbed cheaply, while a false PASS would
have shipped a hollow fleet gate.
