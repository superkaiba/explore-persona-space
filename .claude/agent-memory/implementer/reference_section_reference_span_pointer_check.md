---
name: section-reference-span-pointer-check
description: A new H2 span in a *-section-reference.md rules file FAILs the no-flags lint unless the owning agent spec carries a one-line pointer naming the EXACT heading
metadata:
  type: reference
---

Adding a new `## <heading>` span to `.claude/rules/<agent>-section-reference.md`
requires the owning spec (`.claude/agents/<agent>.md`) to carry a pointer line of
the form `` `<rules path>` § <exact heading text>. `` on ONE line — the no-flags
`workflow_lint.py` pointer-reachability check (#850/#1159) FAILs otherwise:
"section heading '<h>' (H2 grain) has no '§ <exact heading>' pointer line in the
owning spec". A shortened pointer (`§ Schema-from-artifact` for the heading
`Before-writing-code item 8 detail — Schema-from-artifact`) does NOT satisfy it.

**How to apply:** when a plan's verbatim insert text carries an abbreviated `§`
pointer, expand it to the exact heading at implement time (wording-variation
freedom covers this); copy the one-line shape from an existing item's pointer
(e.g. experiment-implementer.md L189, item 5). Caught on #2120 only by the
baseline-relative no-flags lint run (acceptance criterion 7) — the scoped new-check
flag and all pin tests were green. Related: [[verify-plan-check-fanout]] (same
"registration has more surfaces than the obvious one" family).
