---
name: Reuse-fitness check 6-file mirror set
description: Any change to the trained-artifact reuse fitness check (a)-(h) must touch artifact-reuse.md (canonical), planner.md step 5, CLAUDE.md reuse bullet, critic.md Methodology item 9, consistency-checker.md ((a)-(h) cross-refs), and the verify_plan.py c6 heuristic + its test
type: reference
---

The trained-artifact reuse fitness check (items (a)-(h)) is mirrored across
SIX workflow-surface files (8 edit locations), and precedent fixes (#600
content-identity, #601 application-scaling, #545 adapter_config grounding,
#734 train-input fetchability) all touched the relevant set in one commit:

1. `.claude/rules/artifact-reuse.md` — the CANONICAL checklist (the full
   (a)-(h) bulleted list + the H1/description range + the enforcement chain).
   This is the primary file; the others mirror it.
2. `.claude/agents/planner.md` step 5 — the planner's detailed self-attest
   list (first-person verification voice) + the "survives all of (a)–(h)"
   closing line + the §11 per-input-artifact `Source:` note.
3. `CLAUDE.md` "Reuse existing trained artifacts" bullet — terse always-on
   summary, one-clause additions only.
4. `.claude/agents/critic.md` Methodology lens item 9 — both the (a)-(h)
   enumeration AND the "REVISE in two directions" clause may need the change.
5. `.claude/agents/consistency-checker.md` — TWO spots: (i) the summary table
   row at line ~42 ("Reused trained artifact does not smuggle a second changed
   variable"), ALWAYS touched; (ii) the detail section at ~line 108+ (the
   load-bearing-set hyperparameter list), touched ONLY when the change affects
   the load-bearing HYPERPARAMETER set to diff (#545, #600, #601 did — they
   changed `r` / `lora_alpha` grounding; the #734 (h) check did NOT — it adds
   no new hyperparameter to diff, only the line-42 row + the (a)-(h)
   cross-refs). Also update the (a)-(g) → (a)-(h) cross-refs at lines ~102 and
   ~263.
6. `scripts/verify_plan.py` `c6_reuse_fitness` (~lines 730-767) + its test in
   `tests/test_verify_plan.py` (the two c6 assertions at ~lines 658 + 670) —
   the mechanical heuristic that greps the plan for the lettered fitness items.
   When the letter range grows, bump the `\(([a-z])\)` regex character class,
   the `/N` denominator + the "N letters / N attestations" count words in the
   WARN/PASS strings, and the COUPLED test assertions (`"4/N"`, `"(a)–(z)"` /
   the count word). KEEP the `>= 4` PASS threshold UNCHANGED — it is a
   heuristic floor, not the letter count. The en-dash `(a)–(z)` in the WARN
   strings carries a `# noqa: RUF001`; preserve it.

A candidate targeting only planner.md leaves the independent enforcement
passes (critic, consistency-checker), the canonical rule file, and the
mechanical verify_plan.py heuristic checking the old contract. Check
`git show --stat` on prior `workflow-fix: reuse fitness check ...` commits
to confirm the current mirror set before editing.
