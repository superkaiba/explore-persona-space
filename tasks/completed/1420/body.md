---
title: 'workflow-fix: semantic probe for absence-claim greps'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9ce977ee5851
created_at: '2026-07-16T11:00:12Z'
has_clean_result: false
origin_prompt: 'prose follow-up from /issue 1386 (2026-07-16): verified-at-filing
  absence claims about text-matching guards need a semantic probe (fragment grep /
  classifier run) + landed-fix history check, not a verbatim-literal grep — #1386
  was filed+spawned 9h after #1360 landed the fix (hub.py ''queue size reached'' vs
  the grepped ''maximum queue size'')'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1386 (emitting agent: /issue orchestrator, Step 1 clarifier context-gathering).

## Goal

Strengthen the verified-at-filing rule for ABSENCE claims about text-matching guards (substring lists / regex classifiers): require a semantic probe — grep for fragments/substrings of the claimed text and/or run the classifier against the claimed text — plus a recent-landed-fix history check (`git log --since='7 days ago' -- <target_file>` or the #1399 advisory list), instead of accepting a verbatim-literal 0-hit grep as absence evidence.

## Workflow gap

- **Bug observed:** #1386 (daily-fix, Xet "maximum queue size reached" transient) was filed and an autonomous session spawned ~9h AFTER #1360 landed the functional fix on main (`hub.py` `'queue size reached'` substring, merge `289ad17572`); the filer's verified-at-filing grep for the verbatim `'maximum queue size'` returned 0 hits and was accepted as absence evidence. A whole session spawn was burned discovering the duplicate.
- **Why it is a workflow gap:** the current rule grants absence claims an exemption ("its 0-hit in-target result IS the evidence" — workflow-fix-on-bug.md § verified-at-filing (a)), but for a TEXT-MATCHING guard the functional fix routinely lands under a SHORTER substring than the full error text, so a verbatim-literal grep is structurally blind to it. Neither downstream defense covers this case: the dedup predicate keys on OPEN tasks only (by design), and the #1399 recently-closed-sibling advisory enumerates only `workflow-fix:`/`daily-fix:`-prefixed closed tasks — #1360 was an ordinary infra task with no prefix, so it was invisible to both.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rln 'verified-at-filing' .claude/ tests/` → 2 workflow-surface files (2026-07-16): `.claude/rules/workflow-fix-on-bug.md` (6 hits — the § Body-file template line, the consistency-BINDS paragraph (a)-(c), the anti-pattern table rows) and `.claude/skills/daily/SKILL.md` (1 hit — the route-2 filing duty). Presence claim confirmed per target; `tests/test_daily_three_route_classifier_doc.py` pins the daily wording and may need a matching update.

## Proposed change (candidate diff sketch — refine in planning)

```
In workflow-fix-on-bug.md § verified-at-filing, extend clause (a)'s absence-claim exemption:
+ (a') An ABSENCE claim about a TEXT-MATCHING guard (an error-text substring
+ list, a regex classifier, a transient-error predicate) is NOT satisfied by a
+ verbatim-literal grep alone: probe SEMANTICALLY — grep for shorter fragments
+ of the claimed text, and/or run the predicate against the claimed text
+ (e.g. `uv run python -c "...; print(_is_transient_upload_error(RuntimeError('<text>')))"`)
+ — and check for a recently-landed fix on the target file
+ (`git log --oneline --since='7 days ago' -- <target_file>`), since the
+ dedup predicate (open-tasks-only) and the #1399 advisory
+ (wf-fix/daily-fix-prefixed tasks only) are both blind to an ordinary
+ landed infra fix. (#1386-over-#1360, 2026-07-16.)
Mirror one line into daily/SKILL.md route 2's filing duty; add an anti-pattern
table row; update tests/test_daily_three_route_classifier_doc.py if it pins wording.
```

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md`, `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'verified-at-filing' .claude/ CLAUDE.md scripts/ tests/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- Non-architectural: a rule-text tightening, no public-contract change.

## Provenance

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md, .claude/skills/daily/SKILL.md
- fingerprint: 9ce977ee5851

Surfaced prose (verbatim, from the /issue 1386 session, 2026-07-16): "Root cause of the mis-filing: the verified-at-filing grep searched the verbatim 'maximum queue size' (0 hits) while the landed guard uses the shorter functional substring 'queue size reached' — a too-literal absence probe. Neither the open-task dedup nor the #1399 closed-sibling advisory (wf-fix/daily-fix titles only) could catch the duplication because #1360 was an ordinary infra task. Absence claims about text-matching guards need a semantic probe + a landed-fix history check at filing time."
