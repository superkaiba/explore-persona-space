---
name: bugged-experiment-title-exception
description: When user explicitly directs "mark experiment as bugged", leading the title with BUGGED is a deliberate exception to clean-result-critic Lens 8; skip the critic re-run, verify_task_body + audit are the gates.
metadata:
  type: feedback
---

Default (clean-result-critic Lens 8): titles state the post-correction finding, never mistake-framing. **Exception, user-directed only:** when Thomas explicitly says "mark as bugged" AND there is no real finding to lead with (load-bearing arm invalidated by independent bugs, the planned contrast never ran, or the metric was confounded by construction), lead the title with BUGGED. Forcing a finding-shaped title there would itself be the over-claim Lens 8 exists to prevent.

**Why:** task #407 (2026-05-31) — Thomas reviewed at awaiting_promotion: the "content-agnostic gating" thread reduced to "SFT learned the contaminated mapping it was trained on", and the per-framing comparison was metric-confounded (substring-match vs judge-category). Demoted to a one-paragraph aside; title re-led with BUGGED.

**How to apply:**
- Trigger ONLY on explicit user instruction; never on your own initiative (the default rule is right ~99% of the time).
- Confidence → LOW; TL;DR explains the independent bugs in the relevant result section; any surviving signal (e.g. modest confirmatory replication) gets its own section with a doesn't-extend-the-published-claim caveat.
- Demoted threads → one-paragraph "What X shows (and what it does not)" with the confound named.
- DROP metric-confounded figures; KEEP figures documenting the bugs.
- Skip the clean-result-critic re-run (Lens 8 would FAIL by design); `verify_task_body.py` + `audit_clean_results_body_discipline.py` are the only gates.
- Post `epm:interpretation v<N+1>` quoting the USER DECISION verbatim.
