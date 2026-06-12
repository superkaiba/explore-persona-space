---
title: Spot-check the awaiting_promotion backlog with the new verify_task_body URL-existence
  check
kind: analysis
tags:
- agent-ok
created_at: '2026-06-11T02:57:36Z'
has_clean_result: false
---
Promoted #507 shipped a hero figure that was both wrong and 404'd (Thomas: 'The first plot in it is broken'), and dead repro URLs were found in 8 parked tasks; 7 were repaired by parallel agents and a verify_task_body.py URL-existence check landed same day.
Action: run the new URL-existence check across the remaining awaiting_promotion backlog and repair any dead figures or repro URLs before promotion.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
