---
title: 'Restore dead auth and notification channels: Google Workspace OAuth re-auth
  plus gcloud, Happy Remote Control re-pair or Telegram fallback'
kind: infra
tags:
- needs-thomas
created_at: '2026-06-11T02:58:12Z'
has_clean_result: false
---
Google Workspace MCP OAuth is dead (invalid_grant x6), blocking the Gmail-OTP Mila lane and morning email pulls. Happy Remote Control phone pushes were dead all evening ('Mobile push not sent' x5, including the #537 plan-approval park and #534's BLOCKED notification), plausibly why #524 stalled 61 min at its gate.
Actions: Thomas re-auths Google Workspace MCP (and gcloud); re-pair the phone for Happy Remote Control, or route gate-park notifications through the my-goat Telegram digest as a fallback.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
