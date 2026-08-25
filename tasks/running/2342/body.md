---
title: 'workflow.yaml markers registry: reconcile marker kinds mismatch with /issue
  review-loop practice'
kind: infra
tags: []
created_at: '2026-08-17T10:02:28Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from the #2330 clean-result r2 reconciler'
workflow: v1
---
## Gap

workflow.yaml § markers registers only `epm:review-reconcile` as the reconcile marker kind, but the /issue clean-result loop's reconciler posts `epm:clean-result-critique-reconcile` (observed on #2330 rounds 1+2, both read-back verified) — a registry/practice mismatch. Same family: the interp loop posts `epm:interp-critique-reconcile` (also on #2330). Either register the site-specific reconcile kinds or migrate the /issue review loops to the canonical role-tagged kind.

Surfaced as a workflow-fix-candidate by the round-2 clean-result reconciler on #2330 (marker epm:clean-result-critique-reconcile v2, "Observed but not raised").

## Asked change

Reconcile registry and practice in .claude/workflow.yaml § markers (plus any lint that validates marker kinds): add the site-specific reconcile kinds (`epm:interp-critique-reconcile`, `epm:clean-result-critique-reconcile`, `epm:code-review-reconcile` if unregistered) OR migrate the posting sites to one canonical kind — whichever the registry's design intent supports; update workflow_lint marker checks accordingly.
