---
title: 'Autonomous paper pipeline: proposal-in -> autonomous progress (design)'
kind: infra
tags:
- needs-thomas
created_at: '2026-06-12T17:07:27Z'
has_clean_result: false
---
Design for a proposal-in, autonomous-progress paper pipeline layered on the existing /campaign + /issue --auto + tick-cron stack: new /paper-brief intake (proposal -> campaign briefs + paper skeleton + claims ledger, one approval gate), existing campaign machinery unchanged, promotion stays user-only with a my-goat nudge, and a new /goal-mode-driven paper-writer session assembling promoted-useful clean-results into a draft via ml-paper-writing + a paper-critic. Full design doc: ~/my-goat/queue/awaiting-approval/2026-06-12_autonomous-paper-pipeline-design.md (also published to the my-goat-docs mobile mirror). Nothing implemented; review the stage design + pick the first step (claims ledger format is the smallest highest-leverage one).
