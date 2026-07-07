---
title: 'workflow-fix: parse semicolon-inline followup-scope/run note fields'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c880a0caac14
created_at: '2026-07-07T16:16:32Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 1090 Step 0 scan: semicolon-joined
  single-line followup-scope/run notes defeat parse_followup_note_field (line-initial
  only) -> ghost pseudo-label groups + unclosed fu2 round + cheap-band cap undercount'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1090 (emitting agent: issue-session orchestrator).

## Goal

`parse_followup_note_field` also parses semicolon-separated inline fields so single-line followup-scope/run notes group and close correctly.

## Workflow gap

- **Bug observed:** followup-scope and same-issue-followup-run markers written as one semicolon-joined line parse no followup_label, creating permanent non-dispatchable pseudo-label ghost groups and uncounted cheap-band run markers.
- **Why it is a workflow gap:** `task_workflow.parse_followup_note_field` only matches LINE-INITIAL `field:` / `field=` cores, but real orchestrator sessions emit scope + run notes with all fields on one `; `-joined line (task #1090: scope v1 `source: proposer-9b-cheap; followup_label: fu1-margin-qwen; ...` at 2026-07-07T07:19:09Z, scope v2 for fu2-dose-extension at 11:05:01Z, plus the fu1 run marker at 09:54:27Z whose note leads with the label as prose). Result on #1090: two permanent `dispatchable: false` `unlabeled-<ts>` ghost groups that surface as repair items on every Step 0 scan, a completed fu2 round with no closing run marker, and cheap-band cap counting (`source: proposer-9b-cheap` run markers) reading 0 of 2 actually-run rounds.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
src/explore_persona_space/task_workflow.py :: parse_followup_note_field
  for line in (note or "").splitlines():
-     core = re.sub(r"^[\s\-*]+", "", line)
-     if core.startswith(f"{field}:") or ...
+     # split each physical line on "; " so semicolon-joined single-line
+     # notes (observed in the wild on #1090) expose their fields too;
+     # first hit wins, existing forms unchanged.
+     for seg in re.split(r";\s+", line):
+         core = re.sub(r"^[\s\-*]+", "", seg)
+         if core.startswith(f"{field}:") or core.startswith(f"{field}="):
+             ... (existing token extraction; also rstrip(";") on the value)
```

Plus: pin the behavior in `tests/test_task_workflow*.py` with the verbatim #1090
scope-note shapes (semicolon-inline scope; run note with trailing `;`-delimited
fields), and consider a symmetric emitter-side note in SKILL.md Step 9b step 4 /
workflow.yaml § markers reminding that scope/run notes MUST carry line-initial
fields (the parser fix is the robust half; the emitter contract is the clean half).
After the parser fix lands, #1090's two pseudo-groups self-heal for fu1 (its run
marker's `followup_label` may still not parse — leading prose, not a field; if so,
the planner should decide between also matching the run-marker single-line form or
documenting a one-time retro-close for #1090).

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'parse_followup_note_field' .claude/ CLAUDE.md scripts/ src/explore_persona_space/task_workflow.py`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- Do NOT loosen the retro-close evidence classes (`followup_retro_close_evidence`)
  — this fix is about field PARSING, not about closing rounds on weaker evidence.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: c880a0caac14

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: followup-scope and same-issue-followup-run markers written as one semicolon-joined line parse no followup_label, creating permanent non-dispatchable pseudo-label ghost groups and uncounted cheap-band run markers
why_workflow_gap: parse_followup_note_field only matches line-initial field cores, but real sessions emit semicolon-joined single-line scope/run notes (task #1090 fu1/fu2), so grouping, run-marker closure, and cheap-band cap counting all silently fail
proposed_change: parse_followup_note_field also parses semicolon-separated inline fields so single-line followup-scope/run notes group and close correctly
diff_sketch: |
  + for seg in re.split(r";\s+", line):
  +     core = re.sub(r"^[\s\-*]+", "", seg)
  +     if core.startswith(f"{field}:") or core.startswith(f"{field}="):
  +         ... existing token extraction, value rstrip(";")
confidence: high
related_task: #1090
<!-- /workflow-fix-candidate -->
