---
name: A "stash to pop verbatim" can carry abandoned strategy-pivot WIP
description: When a brief says "pop stash@{0} verbatim" for one feature, inspect the stash diff FIRST — a prior implementer's strategy pivot may have bundled the abandoned-design code + tests into the same stash; salvage only the wanted hunks, never blind-pop.
type: feedback
---

When a subagent brief says "the stash is feature X, ready to land verbatim —
just `git stash pop`", do NOT trust the label. Run `git stash show -p
stash@{0}` and read the WHOLE diff before popping.

**Why:** a prior implementer who hit a plan contradiction often stashes
EVERYTHING in flight before pivoting — including the abandoned design's code +
tests — not just the one clean feature the brief writer remembers. Blind-popping
resurrects dead code that (a) collides with the new (Option-A) implementation
you were told to write, and (b) carries tests asserting the OLD (Option-B)
behavior, which contradict the new acceptance criteria. (#759, 2026-06-30:
`stash@{0}` labeled "(b.2) STALLED_WINDOW_S 45->60" ALSO bundled the v3-retired
Option-B b.1 `_apply_stalled_live_corroboration` work-fresh-downgrade impl + 5
Option-B tests. A `git stash pop` would have brought back the exact dead-code
design the plan §3(vi)/§11 replaced.)

**How to apply:** the binding plan design wins over the brief's "pop verbatim"
shorthand. Resolution recipe when a stash mixes wanted + retired hunks:
1. `git stash show -p stash@{0} > /tmp/issue-<N>-stashN-FULL.patch` (PRESERVE
   the full diff — honors the "never drop without preserving" rule).
2. Implement the binding design yourself (the wanted feature + anything the new
   design also needs, e.g. the live_ids threading).
3. For the salvage commit, apply ONLY the wanted hunks manually (re-type or
   edit them from the preserved patch), NOT a `git stash pop`.
4. `git stash drop stash@{0}` (the patch is preserved).
5. Note the divergence in the commit message + `epm:results` (b)/(d) so the
   reviewer sees you deliberately did NOT pop, and why.

This is a "resolve the contradiction, don't block" case (the resolution is
clear once you read the diff), not a `epm:failure` — but it IS a divergence from
the brief worth flagging loudly in the report.
