---
title: 'verify_task_body check 31: normalize plan-figure prose tokens to PNG stems
  (descriptively-named committed figures escape the body-mention check)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-11T13:07:33Z'
has_clean_result: false
parent_id: 2222
workflow: v1
---
# verify_task_body.py check 31: plan-figure prose tokens never matched against committed PNG stems

## Gap (surfaced by clean-result-critic on #2222, round 1)

`verify_task_body.py` check 31 (class-C candidate matching) matches only literal filename stems / `per{context,unit,cell}` patterns against plan-named figures. A plan deliverable named DESCRIPTIVELY — #2222 plan §6's "sample-level ROC curves per arm", committed as `figures/issue_2222/roc_by_arm.png` at the body's own pin — passes the check silently while appearing nowhere in the body (0 mentions). The clean-result-critic caught it manually as a Lens 11 blocker (aggregate AUC result with no underlying-data view, exemption, or pointer, while the committed companion sat unnamed at the pinned SHA).

## Fix sketch

Add a prose-token→stem normalization to check 31's class-C candidate builder: tokenize the plan's figure-list prose (e.g. "ROC curves" → stem token `roc`; drop stopwords, singularize, match tokens against committed PNG stem tokens at body-cited SHAs), and WARN when a plan-named committed figure is neither embedded nor mentioned in the body. WARN grade (not FAIL) — descriptive-name matching is heuristic; the critic lens stays the binding gate.

## Acceptance

- The #2222 shape reproduces: a fixture plan naming "sample-level ROC curves per arm" + a committed `roc_by_arm.png` at a body-cited SHA + a body with 0 mentions ⇒ check 31 WARNs naming the stem.
- Existing literal-stem matches unchanged; no new FAILs on the committed corpus of v4 bodies (run the check across tasks/*/*/body.md with the v4 sentinel to confirm zero regressions).
- Test added to tests/test_verify_task_body.py pinning both the WARN and the no-regression behavior.

## Provenance

workflow_fix_target: scripts/verify_task_body.py (check 31)
Surfaced by: clean-result-critic round 1 on #2222 (2026-08-11), "Workflow-fix suggestion" prose block; verdict file /tmp/issue-2222-crc-r1.md (also posted as epm:clean-result-critique on #2222).
