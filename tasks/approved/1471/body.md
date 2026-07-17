---
title: 'daily-held: non-managed EXITED pod reaping decision'
kind: infra
tags:
- daily-held
created_at: '2026-07-17T06:58:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 3): the stale-pod audit''s
  EPS-ownership gate covers managed-prefix names only; non-managed-name EXITED>24h
  pods still auto-reap — but plan #1404 v3 section 11 DELIBERATELY scoped the gate
  that way and all three critics accepted'
workflow: v1
---
## Held item (route 3 — needs-human)

Auto-filed by /daily 2026-07-16 as a TRACKED needs-human task (route 3 of the three-route classifier).

- **What happened:** the stale-pod audit's EPS-ownership gate covers managed-prefix names only; non-managed-name EXITED>24h pods still auto-reap — but plan #1404 v3 section 11 DELIBERATELY scoped the gate that way and all three critics accepted
- **Which carve-out held it:** destructive / irreversible actions — auto-terminating pods EPS may not own
- **Decision needed:** decide: extend the ownership gate to non-managed EXITED pods (safer vs third-party pods on the team account) or keep the deliberate scoping
- **Suggested surface:** PM `Needs you` block.
