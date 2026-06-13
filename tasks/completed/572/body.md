---
title: Harden pod resume and provisioning waits in pod.py and pod_lifecycle.py
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:18Z'
has_clean_result: false
---
pod-488 idled ~13.7h on a dead SSH port at $32/hr (~$420) after a SUPPLY_CONSTRAINT resume left the stale port in pods.conf; separately 3 morning sessions went dark when ~1h-old background provision commands were killed, and pod_lifecycle.py's resume retry loop reused a stale fleet-burn figure ($128/hr stale vs $80/hr live) blocking #518 six times.
Actions: (1) verify the config --refresh-from-api wiring (poll_pipeline + session-watch + SKILL.md Step 6b) actually fires on the next SUPPLY_CONSTRAINT resume; (2) add a pod.py-side alarm after 1h of SSH-wait on a billing pod; (3) cap each wait-for-capacity attempt under 50 min with a structured still-waiting exit; (4) recompute live fleet burn inside the resume retry loop.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
