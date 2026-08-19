---
title: 'workflow-fix: review/compose briefs check marker presence against the WORKING
  TREE, so a transient stash-cycle read produces a false marker-destroyed escalation'
kind: infra
tags:
- wf-fix
created_at: '2026-08-16T17:41:21Z'
has_clean_result: false
origin_prompt: 'Surfaced during #2325 Step 5 code-review rounds 2-3: two agents independently
  read events.jsonl from the working tree inside a concurrent pre-commit stash cycle,
  concluded epm:results v3 had been destroyed, and one instructed the orchestrator
  to re-append the row (which would have duplicated the marker). The row was never
  lost — verified present and committed by reading the blob out of HEAD. repo-root-uncommitted-state.md
  already prescribes the HEAD-read discipline but no reviewer/composer brief carries
  it.'
workflow: v1
---
# workflow-fix: review/compose briefs check marker presence against the WORKING TREE, so a transient stash-cycle read produces a false "marker destroyed" escalation

## Problem

Under fleet concurrency, a task marker can exist as an appended-but-uncommitted
line in `tasks/<status>/<N>/events.jsonl` for many minutes: `task.py post-marker`
exits 0 with only a stderr ERROR when the append lands but the commit is deferred
by an `index.lock` collision, and the deferred path is swept by whichever
`task.py` commit touches it next.

While that row is uncommitted it is exposed to the #2015 pre-commit stash cycle:
every fleet commit runs `git checkout -- .` for its hook window, transiently
reverting unstaged tracked lines to HEAD content. Any agent that reads
`events.jsonl` from the WORKING TREE inside that window sees a file with the row
missing.

`.claude/rules/repo-root-uncommitted-state.md` already prescribes the correct
read ("never verify by the push line... verify by blob read at the specific SHA")
and documents the transient-reversion shape explicitly. But **no reviewer or
composer brief carries that instruction**, so agents reach for the working tree
by default.

## Observed incident (#2325, 2026-08-16)

`epm:results` v3 was posted at 17:13:13Z; its commit was deferred at 17:14:14Z
(deferred-commits row, alongside two #2321 rows from the same lock storm). During
the ~11-minute uncommitted window:

- the `codex-code-reviewer` prompt-composer (17:19Z) diagnosed **#2015
  PERMANENT DESTRUCTION**, extracted a byte-exact rescue copy from
  `~/.cache/pre-commit/patch1786900492-3899925`, and instructed the orchestrator
  to **re-append the row** — which would have DUPLICATED the marker;
- the Claude `code-reviewer` round-2 (17:24Z) independently made the row's
  absence its sole blocker, producing a FAIL on a marker that existed.

Both reads were correct about the bytes they saw and wrong about the conclusion.
The row was never lost: it is row 26, committed in `d2959aa5a1`, verified by
reading the blob out of HEAD. The orchestrator spent a detour cycle
disconfirming the escalation, and one review round produced a false blocker.

The near-miss is the duplicate-append: the composer's instruction was explicit
and actionable, and following it would have corrupted the task's event log.

## Scope for the planner (not pre-decided)

The gap is that a durable, already-documented verification discipline is absent
from the briefs of the agents most likely to need it. Candidate directions, and
the choice is a judgment call:

1. **Brief-side instruction.** Add to the reviewer/composer agent specs
   (`code-reviewer.md`, `codex-code-reviewer.md`, the `codex-*` composer-common
   contract, and plausibly `clean-result-critic` / `interpretation-critic` — any
   agent that reads `events.jsonl` to check marker presence) a standing rule:
   read markers via `git show HEAD:tasks/<status>/<N>/events.jsonl`, and never
   declare a marker missing or lost from a working-tree read alone. Cheapest, but
   spends bytes on several ratcheted specs.
2. **A helper the briefs point at.** A small read-only accessor (or a `task.py`
   subcommand) that returns markers from HEAD-plus-working-tree union, so agents
   have one correct call instead of a discipline to remember. Costs a surface but
   removes the failure mode rather than documenting around it.
3. **Escalation-side guard.** Whatever an agent concludes, a "marker destroyed /
   restore this row" instruction should be structurally unable to reach an
   orchestrator without a HEAD-read confirmation attached. Narrower, and targets
   the dangerous output (duplicate-append instruction) rather than the reading
   habit.

Worth pricing together, since (1) is nearly free and (3) addresses the actual
harm; they are not mutually exclusive.

## Acceptance

- An agent checking marker presence has an unambiguous, brief-level instruction
  (or a helper) that does not depend on the working tree, and a durability pin
  (a pytest) so the instruction cannot be silently dropped by a later spec edit.
- A "marker missing / lost / restore" conclusion requires a HEAD-read
  confirmation before it can be emitted as an actionable instruction.
- The no-flags `workflow_lint` run shows no NEW failures vs baseline, and any
  ratcheted spec the change grows has its cap raised in the same change per the
  corridor protocol.
- Step 9c universe green.

## Provenance

Surfaced during #2325's Step 5 code-review ensemble (rounds 2-3). #2325 is a
`kind: infra` workflow-fix task; its `body.md` carries no `workflow_fix_target:`
line and `EPM_WORKFLOW_FIX_SESSION` was unset in the filing session, so the
recursion guard does not apply.

Two orchestrator-side lessons from the same incident, recorded here because they
are the upstream cause and may belong in the same fix:

1. `task.py post-marker` stderr must not be suppressed — the deferral ERROR is
   the only signal that a row is sitting uncommitted, and #2325's orchestrator
   had piped stderr to `/dev/null` on every post, which is why the exposure went
   unnoticed for 11 minutes.
2. Once a deferral is known, the swept path should be committed by explicit path
   immediately rather than left to ride to the next incidental commit. In #2325
   two attempts to do so were themselves blocked — first by the root-commit guard
   scoping against a FOREIGN staged file from a concurrent session, then by a live
   27 MB `index.lock` — which is worth examining as its own friction.

Related: `.claude/rules/repo-root-uncommitted-state.md` (the rule that already
has the right answer), #2015 (the stash race), #690 (file == dispatch).
