---
title: 'verify_task_body Lens-14/check-64 substring false-positive: HTML comments
  inside Takeaways count as concern acknowledgment'
kind: infra
tags: []
created_at: '2026-08-24T04:56:45Z'
has_clean_result: false
workflow: v1
---
Found by the reconciler on #1739 (re-gate r2, binding reconciliation, 2026-08-24): verify_task_body.py's v4 concern-acknowledgment scan (around verify_task_body.py:16587-16590) folds RAW Takeaways section text, so a concern id named only inside an HTML comment (<!-- ... -->) within ## Takeaways satisfies the check even though the id is invisible when rendered and appears in none of the three allowed Lens-14 mechanisms (result/Takeaways-bullet prose, interpretation bound, real deferral marker). Fix: strip HTML comments from the section text before the acknowledgment scan (deferral markers are matched by their own exact-comment pattern and must keep working). Add a fixture reproducing the #1739 shape: id present only in a Takeaways HTML comment + Methodology prose => check FAILs.
