---
name: Reuse-fitness check 4-file mirror set
description: Any change to the trained-artifact reuse fitness check (a)-(g) must touch planner.md step 5, CLAUDE.md reuse bullet, critic.md Methodology item 9, and consistency-checker.md (table row + detail section)
type: reference
---

The trained-artifact reuse fitness check (items (a)-(g)) is mirrored across
FOUR workflow-surface files, and precedent fixes (#600 content-identity,
#601 application-scaling, #545 adapter_config grounding) all touched the
full set in one commit:

1. `.claude/agents/planner.md` step 5 — the canonical detailed list.
2. `CLAUDE.md` "Reuse existing trained artifacts" bullet — terse always-on
   summary, one-clause additions only.
3. `.claude/agents/critic.md` Methodology lens item 9 — both the (a)-(g)
   enumeration AND the "REVISE in two directions" clause may need the change.
4. `.claude/agents/consistency-checker.md` — TWO spots: the summary table
   row ("Reused trained artifact does not smuggle a second changed
   variable") and the detail section (~line 110+, the load-bearing-set list).

A candidate targeting only planner.md leaves the independent enforcement
passes (critic, consistency-checker) checking the old contract. Check
`git show --stat` on prior `workflow-fix: reuse fitness check ...` commits
to confirm the current mirror set before editing.
