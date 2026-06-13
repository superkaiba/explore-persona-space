---
title: Fix clean-result reporting (set-clean-result / list-clean-results surface)
kind: infra
tags: []
created_at: '2026-05-29T07:09:06Z'
has_clean_result: false
---
Fix how clean results are recorded and displayed. Timing TBD: Thomas flagged this as possibly a weekend task (Sat vs Sun, decide when picking it up).
Observed problem to investigate before scoping the fix: list-clean-results currently shows only 2 tasks (#390, #391), yet ~30+ tasks in awaiting_promotion already carry has_clean_result=true in frontmatter (e.g. #376, #377, #382, #380, #396, #411, #225, #234, #207). So the has_clean_result flag and the list-clean-results query are out of sync, or the flag is being set inconsistently across the lifecycle. The reporting surface understates how many clean results actually exist.
Why it matters: clean results are the unit that feeds mentor updates and paper claims; an undercounting list view hides finished work and makes it hard to see what is ready to interpret/promote.
Starting point: read scripts/task.py set-clean-result and list-clean-results implementations, confirm what predicate list-clean-results filters on (status? promoted? flag?), then reconcile the flag semantics with the awaiting_promotion vs completed distinction. Relates to task C (interpret uninterpreted clean results).
