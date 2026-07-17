---
title: 'daily-held: fix sudo choom for lint-gate earlyoom shield'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-17T06:58:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 3): `sudo -n choom -n -600`
  fails intermittently in Step-10d lint/pytest gates (13 failed vs 8 ok in one #1345
  session), so gates frequently run earlyoom-UNPROTECTED (warn-and-continue)'
workflow: v1
---
## Held item (route 3 — needs-human)

Auto-filed by /daily 2026-07-16 as a TRACKED needs-human task (route 3 of the three-route classifier).

- **What happened:** `sudo -n choom -n -600` fails intermittently in Step-10d lint/pytest gates (13 failed vs 8 ok in one #1345 session), so gates frequently run earlyoom-UNPROTECTED (warn-and-continue)
- **Which carve-out held it:** external/system state change (sudoers) + genuinely ambiguous (fix vs drop the warn)
- **Decision needed:** decide: add a sudoers rule so the session user can run choom non-interactively (system state change), or drop the warn if the protection is genuinely optional
- **Suggested surface:** PM `Needs you` block.
