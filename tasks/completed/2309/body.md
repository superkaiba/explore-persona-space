---
title: 'workflow-fix: implementer marker four-H3 completion-report contract is judgment-caught,
  not mechanically checked'
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T01:15:40Z'
has_clean_result: false
parent_id: 2302
origin_prompt: code-reviewer prose follow-up surfaced during /issue 2302 round 1
workflow: v1
---
## Goal

Make the four-H3 completion-report contract for implementer round markers
(`epm:results` / `epm:experiment-implementation`) MECHANICALLY checked, so a missing
`### (a)`-`(d)` section is caught at post time instead of costing a full code-review round.

## The gap

`.claude/agents/implementer.md:260` requires every implementer completion report to carry four
H3 sections — `### (a) What was done`, `### (b) Considered but not done`, `### (c) How to
verify`, `### (d) Needs human eyeball` (the template even supplies the empty-case wording for
(d): *"None — confidence high across the diff."*). Nothing enforces it. Today the contract is
JUDGMENT-caught: it holds only if the reviewer happens to notice an absent heading while reading
a 100+ line marker for substance.

## Evidence (#2302, 2026-08-14)

#2302's round-1 `epm:results v1` omitted `### (d)` entirely — verified as genuine absence, not a
formatting variant: no `(d)`, `eyeball`, or `needs human` text anywhere in the 105-line marker.
Sections (a)/(b)/(c) were present and complete, and the code payload was independently
re-verified as merge-quality with **zero substantive blockers**.

The cost was a whole extra ensemble round for a missing heading. It is also NOT strip-eligible:
the Step 5c-bis mechanical-contract strip applies when the implementer marker is present AND
conforming, and a marker-shape blocker IS the non-conformance — so this class can never be
stripped away, it always costs a real round.

Note the content was not even missing — the implementer's own return text to the orchestrator
carried a perfectly good (d). Only the POSTED marker lacked it. That is precisely the shape a
mechanical check catches for free.

## Proposed fix (direction; the plan decides)

A presence check for the four `### (a)`-`(d)` headings on marker kinds carrying the
completion-report contract, at one of:

- `task.py post-marker` / `task_workflow` post time — earliest and cheapest, fails the post so
  the implementer fixes it in its own turn while it still has context; or
- `scripts/workflow_lint.py` as a `--check-*` arm — later, but composes with the existing
  no-flags gate.

Design constraints worth stating up front:

- **Scope the check to the marker kinds that actually carry the contract.** `epm:results` is
  also posted by other paths; a blanket check over every marker kind would fail-loud on markers
  that were never meant to have the sections. Key on the kinds `.claude/agents/implementer.md`
  and `experiment-implementer.md` bind.
- **Fail LOUD, not silently.** A post that violates the contract should be refused or WARNed
  visibly — not normalized, not auto-filled with placeholder text (an auto-inserted "None"
  would defeat the purpose: (d) exists to make the author think).
- **Do not break the deferred-commit path.** `post-marker` already exits 0 with a stderr ERROR
  when the append landed but the commit deferred; a new refusal must not entangle with that.
- **Grandfathering:** existing markers in `events.jsonl` across the fleet are not retroactively
  invalid. A `post-marker`-time check applies forward only; a lint arm needs an explicit
  grandfather or it will light up historical rows fleet-wide.

## Acceptance criteria

1. A marker of a contract-bearing kind missing any of `### (a)`-`(d)` is refused or loudly
   WARNed at post time, naming the missing section(s).
2. A conforming marker posts unchanged — no false positives on the #2302 `epm:results v2` marker
   (which does carry all four).
3. Non-contract-bearing marker kinds are unaffected.
4. Historical `events.jsonl` rows are not retroactively failed.

## Provenance

Surfaced as a non-blocking prose follow-up by the `code-reviewer` in #2302 round 1
(2026-08-14): *"the four-H3 `epm:results` section-shape check is judgment-caught today; a
mechanical presence check (the four `### (a)`-`(d)` headings) at `post-marker` time or in
`task_workflow` would have caught this round's omission for free."*

Filed WITHOUT an auto-spawn (`file_infra_task.py --no-dispatch`) by the #2302 session: #2302 is
itself a `wf-fix` workflow-fix task, so auto-spawning a further workflow-fix session from its own
review findings is the cascade the recursion guard exists to prevent — even though #2302's body
lacks the durable `workflow_fix_target:` line that would make the guard fire mechanically. The
target file here is distinct from #2302's (`scripts/task.py` / `task_workflow` vs
`scripts/step9c_baseline.py` + `scripts/select_step9c_tests.py`), so this is not a dedup hit.
The watcher's `proposed_infra_sweep` pass is the documented dispatch backstop.
