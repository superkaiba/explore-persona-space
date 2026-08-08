---
title: 'workflow-fix: close-miss escalation clause for the on-policy yield floor'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e632c3535116
created_at: '2026-08-02T14:50:59Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on #1947: sycophancy dropped at 232/240 (96.7%
  fill) after one retry tranche; user asked ''how can we fix this'' and approved the
  recovery+amendment path'
workflow: v1
---
## Overview / Motivation
Auto-filed from an orchestrator-observed gap on task #1947 (2026-08-02): the sycophancy arm (16 cells + 2 controls) was dropped at 232/240 accepted positives — a 96.7% fill — after a one-tranche retry budget.
## Goal
Add a close-miss escalation clause to the on-policy-completions yield floor: fill ≥ 90% of floor after the registered retry budget triggers ONE automatic escalation tranche (sized remaining-need/measured-acceptance-rate × 1.3) before the drop fires.
## Workflow gap
- **Bug observed:** a whole behavior arm was dropped eight rows short of its floor after one retry tranche, silently removing 18 planned cells and blinding the fleet's most context-dependent behavior; the elicitation itself was healthy (96.7% fill — not the #906 collapse class).
- **Why it is a workflow gap:** the 80%-floor + drop rule (`.claude/rules/on-policy-completions.md`) was written against catastrophic yield failures (#612/#906: 17%, 1%); it has no proportionality between miss size and consequence, so a near-miss costs the same as a collapse. The anti-silent-backfill intent is preserved by making the escalation an ON-POLICY same-construct tranche, never a template/LLM backfill.
- **Confidence:** high on the incident; medium on the exact 90% threshold (planner calibrates against historical fills).
- verified-at-filing: `grep -n "80% floor\|Floor = 80\|retry budget" .claude/rules/on-policy-completions.md` → the floor + retry-budget + drop mechanics present with no close-miss clause (context read of the § quota bullet, 2026-08-02); incident numbers from #1947 body Design section ("232 judge-accepted positives against the 240 floor after one retry tranche").
## Proposed change (candidate diff sketch — refine in planning)
diff_sketch: |
  In on-policy-completions.md § pre-registered yield quota:
  + **Close-miss escalation:** a source finishing >= 90% of floor after the registered retry
  +   budget gets ONE automatic same-construct escalation tranche (sized remaining/rate x 1.3)
  +   before the drop fires; the escalation is recorded in the datagen manifest. Below 90%,
  +   the drop fires as today. The escalation is never a template/canned backfill.
  Mirror one line in planner.md §4 (quota statement names the close-miss clause).
## Scope / surfaces
- Primary targets: `.claude/rules/on-policy-completions.md`, `.claude/agents/planner.md` (§4 quota line); grep for other floor restatements (experiment-guidelines.md item 7).
## Constraints / invariants
- Anti-silent-backfill intent preserved (escalation = on-policy, same construct, recorded); workflow-surface only; recursion guard applies.
## Provenance
- workflow_fix_target: .claude/rules/on-policy-completions.md
- fingerprint: e632c3535116
Origin: orchestrator observation during the #1947 promote review; the recovery round (inline, 2026-08-02) is the empirical companion — its realized tranche yields will inform the threshold.
