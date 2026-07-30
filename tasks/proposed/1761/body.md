---
title: 'daily-held: untracked scripts/issue823_single_split_protocol'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-28T07:04:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 3): `scripts/issue823_single_split_protocol.py`
  (untracked, dated Jul 15) dirties'
workflow: v1
---
## Held decision (needs Thomas)

Filed by /daily 2026-07-27 problem sweep as a route-3 judgment call.
**Carve-out item:** destructive / irreversible action (deleting/committing someone's draft) + genuinely ambiguous intent

`scripts/issue823_single_split_protocol.py` (untracked, dated Jul 15) dirties
every Step 9c baseline ledger; the root-draft observer has escalated it daily
for ~12 days with no action. Options: commit it (if it is real #823 work worth
keeping), move it into the #823 task artifacts, or delete it. Ownership is
unclear (a #823 follow-up session's leftover), so /daily won't touch it.
Suggested action: `git log`/read the file header, then commit-or-delete.
Evidence: miner G P3; sidecar .claude/cache/root-draft-events.jsonl.
