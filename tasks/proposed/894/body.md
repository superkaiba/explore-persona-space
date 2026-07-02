---
title: 'workflow-fix: same-issue dispatch misses queued distinct-label followup-scopes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d621a0e561f3
created_at: '2026-07-02T21:58:06Z'
has_clean_result: false
origin_prompt: 'orchestrator observation on #763: highest-version-only followup-scope
  scan dropped the queued unrun user-chat round neutral-contrast-and-cofit after a
  later-label round completed'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own observation on task #763 (same-issue follow-up loop, 2026-07-02).

## Goal

Make the `/issue` Step 0 same-issue follow-up dispatch (and the matching resume-table rows) scan ALL `epm:followup-scope` entries grouped by `followup_label` for unrun rounds — dispatching the OLDEST unrun label with user-chat priority — instead of reading only the single highest-version entry.

## Workflow gap

- **Bug observed:** #763 carried two DISTINCT queued scopes: `v1` (`neutral-contrast-and-cofit`, source user-chat, armed 2026-06-30, UNRUN) and `v2` (`deception-rubric-reanchor`, proposer-9b-cheap, run 2026-07-02). After v2's run marker landed, the Step 0 predicate ("the HIGHEST-version `epm:followup-scope` entry ... whose `followup_label` has no matching run") found v2 matched and reported NO unrun scope — the user-requested v1 round became invisible to every future dispatch and would have stranded indefinitely (caught only because the live orchestrator's follow-up-proposer happened to mention it).
- **Why it is a workflow gap:** the predicate assumes marker versions are CORRECTIONS of one scope (the #658 v3→v8 chain), but distinct queued follow-ups share the marker kind with different labels; the highest-version read drops every earlier queued label once any later label completes.
- **Confidence (emitter):** high (reproduced directly from #763's events.jsonl).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/issue/SKILL.md, Step 0 "Same-issue follow-up dispatch" (+ the two
followups_running resume-table rows and the Step 9b loop step-1/step-3 wording):
- the HIGHEST-version epm:followup-scope entry ... whose followup_label has no
- matching epm:same-issue-followup-run v1
+ group ALL epm:followup-scope entries by followup_label; within a label the
+ authoritative scope is the highest (version, ts) entry (corrections);
+ a LABEL is unrun iff it has no matching epm:same-issue-followup-run.
+ When >=1 unrun label exists, dispatch: user-chat-sourced labels first,
+ then oldest armed ts. (Consider a task_workflow helper, e.g.
+ unrun_followup_labels(events) -> ordered list, so tick_triage/watcher passes
+ share the predicate.)
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep first: `grep -rln 'highest-version .*followup-scope\|HIGHEST-version .*epm:followup-scope' .claude/ scripts/ src/explore_persona_space/` — update every hit (tick_triage.py / autonomous_session_watch.py may embed the same predicate; `_followup_round_complete_reason` keys on ts only).

## Constraints / invariants

- Workflow-surface only; workflow_lint passes; keep the within-label correction semantics (highest version/ts) intact; add a pinning test if the predicate lands as a task_workflow helper.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: d621a0e561f3
