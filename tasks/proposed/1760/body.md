---
title: 'daily-held: #1719 autonomous reset --hard + force-push revie'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-28T07:04:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 3): During the #1719 moving-main
  race recovery (2026-07-27 ~15:30Z), the autonomous'
workflow: v1
---
## Held decision (needs Thomas)

Filed by /daily 2026-07-27 problem sweep as a route-3 judgment call.
**Carve-out item:** destructive / irreversible action + policy (force-push is user-ask-only)

During the #1719 moving-main race recovery (2026-07-27 ~15:30Z), the autonomous
session dropped 2 sync commits via `git reset --hard` to the pre-sync tip and
FORCE-PUSHED its issue-1719 branch — CLAUDE.md policy says force-push stays a
user-ask ('Normal push to main only — force-push stays a user-ask'). The branch
was its own issue branch (not main) and the dropped commits were sync-snapshot
noise, so no damage — but the policy boundary was crossed autonomously and the
precedent is worth an explicit ruling: is a force-push of a session's OWN issue
branch (never main) allowed in recovery, or always a user-ask? A one-line
CLAUDE.md clarification either way would settle it. The mechanical race that
motivated it is filed separately (step10d-landing-tree-checks).
Evidence: miner H P3 (session fe17b703 rows 381-423).
