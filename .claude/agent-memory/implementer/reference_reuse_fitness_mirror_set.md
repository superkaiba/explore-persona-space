---
name: Reuse-fitness check mirror set (15 sites)
description: Any change to the trained-artifact + code reuse fitness check (currently (a)-(i)) must touch the full 15-site mirror set — artifact-reuse.md (canonical) + CLAUDE.md bullet + LESSONS.md entry + planner.md step 5/§10 + planner-section-reference.md §10 + critic.md item 9 + critic-lens-reference.md item 9 + consistency-checker.md cross-refs + vectorize back-pointer + verify_plan.py c6 + its tests + 4 agent memories
type: reference
---

The trained-artifact (and code) reuse fitness check — the lettered set,
currently (a)-(i) — is mirrored across FIFTEEN workflow-surface sites, and
precedent fixes (#600 content-identity, #601 application-scaling, #545
adapter_config grounding, #734 train-input fetchability (h), #871 code
throughput (i)) each touched the relevant set in one change:

1. `.claude/rules/artifact-reuse.md` — the CANONICAL checklist (the full
   lettered list + the H1/description range + the closing remedy line + the
   enforcement chain). Since #829 this is the single operational copy; every
   other site mirrors or points at it.
2. `CLAUDE.md` "Reuse existing trained artifacts" bullet — terse always-on
   summary (range + one clause per check + the remedy-split tail);
   one-clause additions only.
3. `.claude/rules/LESSONS.md` — the artifact-reuse index entry names the
   range in its "fires when" trigger.
4. `.claude/agents/planner.md` step 5 — since #829 a POINTER to the rule
   file (NOT an inline self-attest list): the range appears twice + the
   remedy split; the §10 Reproducibility Rows enumeration carries the
   item-(i) code/helper-throughput recording slot.
5. `.claude/rules/planner-section-reference.md` § 10 — the worked
   Reproducibility Card template carries the item-(i) inspection-triple
   paragraph (no lettered-range literal in this file).
6. `.claude/agents/critic.md` — the Methodology lens item-name list names
   item 9's range. Do NOT touch item 10's roman legs (i)-(iv) on the next
   line — a DIFFERENT enumeration.
7. `.claude/rules/critic-lens-reference.md` item 9 — the FULL enforcement
   text: heading scope, operative trigger, inline enumeration (item (i) by
   pointer, not duplication), REVISE directions, retrain-acceptance split,
   no-fire cross-check (all widened to code-only reuse at #871).
8. `.claude/agents/consistency-checker.md` — the range cross-refs (3 hits,
   en-dash spelling); the detail hyperparameter list is touched ONLY when
   the change affects the load-bearing hyperparameter set to diff.
9. `.claude/rules/vectorize-many-cell-fits.md` — the trailing
   "**Sibling check:**" back-pointer to checklist item (i).
10. `scripts/verify_plan.py` `check_reuse_fitness` (c6, ~lines 730-770) —
    when the letter range grows, bump the `\(([a-z])\)` regex character
    class, the `/N` denominator + the count words ("nine letters" / "nine
    attestations") in the PASS/WARN strings, and KEEP the `>= 4` PASS
    threshold UNCHANGED (a heuristic floor, not the letter count). The
    en-dash range in the WARN strings carries `# noqa: RUF001`; preserve it.
11. `tests/test_verify_plan.py` — the two COUPLED c6 assertions (`"4/N"`,
    and the en-dash range / count word in the few-letters WARN test).
12. This memory itself, plus 13. the implementer `MEMORY.md` index line
    pointing at it.
14. `.claude/agent-memory/planner/feedback_778_persona_vector_reuse_artifacts.md`
    — names the range in its reuse instruction.
15. `.claude/agent-memory/critic/feedback_reuse_fitness_mirror_set_completeness.md`
    — the critic's independent-grep completeness lesson names the current
    range.

#871 added item (i): throughput fitness of reused fit/analysis/eval CODE —
inner per-cell/per-fold/per-draw loop batched + device parametrized — scoped
to code reuse (N/A for data-only reuse), plan-time-only, with a
SOURCE-MODULE-fix remedy (never retrain, never a caller-side workaround).
Every remedy-split line on the live surfaces is deliberately worded WITHOUT a
lettered range ("a failing check other than (i)") so the stale-range
completeness grep stays clean and a future letter needs no remedy re-edit.

A change targeting only planner.md leaves the independent enforcement passes
(critic, consistency-checker), the canonical rule file, and the mechanical
verify_plan.py heuristic checking the old contract. Grep
`'\(a\)\s*[-–—]\s*\([a-z]\)'` over `CLAUDE.md .claude/ scripts/ tests/`
(excluding worktrees / __pycache__ / cache) and check `git show --stat` on
prior `workflow-fix: ... fitness ...` commits to confirm the live mirror set
before editing — the documented list can go stale.
