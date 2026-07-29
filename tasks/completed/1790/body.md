---
title: 'daily-fix: move msg-strip comments out of tg gate pipelines'
kind: infra
tags:
- wf-fix
- wf-fix-fp:aff6eb2ead26
- daily-auto-filed
created_at: '2026-07-29T07:06:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): In the Step 10d gate''s
  TG node-grain subtraction pipelines, two comment lines ("# msg-strip caveat: ...")
  are interposed AFTER a backslash line-continuation and BEFORE the `| sed -E ...`
  stage, which breaks the pipeline under bash -n — the fenced block as written is
  not copy-paste-executable (byte-identical defect in both the shared gate block and
  the form (iii) surgical block; pre-existing on ori'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a formal candidate block parked on task #1753 (ts 2026-07-28T13:14:47Z, fp aff6eb2ead26; surfaced by #1753's bash -n smoke on the fenced gate blocks).

## Goal

Move the two "msg-strip caveat" comment lines out of the middle of the Step 10d TG node-grain subtraction pipelines so the fenced blocks are copy-paste-executable again.

## Workflow gap

- **Bug observed:** in the Step 10d gate's TG node-grain subtraction pipelines, two comment lines ("# msg-strip caveat: ...") are interposed AFTER a backslash line-continuation and BEFORE the `| sed -E ...` stage; under bash the continuation joins the comment onto the grep line (terminating the command) and the next line starts with a bare `|` — a syntax error. The fenced block as written is not copy-paste-executable (defect present in both the shared gate block and the form (iii) surgical block; pre-existing on origin/main, surfaced by #1753's bash -n smoke).
- **Why it is a workflow gap:** the gate blocks are copy-source recipes orchestrators execute verbatim; a verbatim copy fails at the node-grain legs, silently degrading the #1573 node-grain arm.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'msg-strip caveat' .claude/skills/issue/SKILL.md` → 2 hits (lines 11135, 12703 — matching the candidate's two occurrences); context read at 11130-11140 confirms the comment sits between the `grep -E '^(FAILED|ERROR) ' ... \` continuation and the `| sed -E` stage (2026-07-29 UTC). Landed-fix history check: `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` — #1753's own merge (`9f5b75b4f3`) covered Guard-4 recovery ordering, not this comment placement; the defect is still live at both sites.

## Proposed change (candidate diff sketch — refine in planning)

```diff
-      grep -E '^(FAILED|ERROR) ' "/tmp/issue-<N>-tg-$leg.txt" \
-        # msg-strip caveat: a literal ' - ' INSIDE a param id truncates here;
-        # a same-prefix dash-bearing sibling collision fails toward pass (narrow doc-only residual, #1573)
-        | sed -E 's/^(FAILED|ERROR) //; s/ - .*$//' \
+      # msg-strip caveat: a literal ' - ' INSIDE a param id truncates here;
+      # a same-prefix dash-bearing sibling collision fails toward pass (narrow doc-only residual, #1573)
+      grep -E '^(FAILED|ERROR) ' "/tmp/issue-<N>-tg-$leg.txt" \
+        | sed -E 's/^(FAILED|ERROR) //; s/ - .*$//' \
```

Both occurrences (SKILL.md ~11135 and ~12703); re-verify with `bash -n` on the extracted blocks.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (2 sites)

## Constraints / invariants

- Recipe semantics unchanged — comment relocation only, but it changes what a verbatim copy EXECUTES (currently broken), so it routes through review rather than self-apply.
- Workflow-surface only; recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: aff6eb2ead26

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: In the Step 10d gate's TG node-grain subtraction pipelines, two comment lines ("# msg-strip caveat: ...") are interposed AFTER a backslash line-continuation and BEFORE the `| sed -E ...` stage, which breaks the pipeline under bash -n — the fenced block as written is not copy-paste-executable (byte-identical defect in both the shared gate block and the form (iii) surgical block; pre-existing on origin/main, surfaced by #1753's bash -n smoke).
why_workflow_gap: The gate blocks are copy-source recipes orchestrators execute verbatim; a mid-pipeline comment makes the verbatim copy fail at the node-grain legs, silently degrading the #1573 node-grain arm.
proposed_change: Move the two "msg-strip caveat" comment lines ABOVE the `grep -E '^(FAILED|ERROR) '` command (or inline them at end-of-line before the continuation), in both occurrences.
confidence: high
related_task: #1753
<!-- /workflow-fix-candidate -->
