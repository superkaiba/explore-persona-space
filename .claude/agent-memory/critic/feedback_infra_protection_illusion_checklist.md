---
name: Infra-plan protection-illusion checklist
description: Alternative-explanations lens for infra guard/gate plans — four recurring ways a "protection" looks armed but is inert (#564)
type: feedback
---

For infra plans that add a guard/gate/reroute, the recurring "appears to work but doesn't" holes (found on #564, HF storage headroom):

1. **Partial-None probe blind spot.** A usage probe that maps per-item `None` → 0 (with a suspect guard only on the ALL-zero case) silently understates when the ONE dominant item returns None — and incidents are usually dominated by one item (#541: 10.2 of 11.3 TB in a single repo). Demand: count Nones, fold into `basis`, treat any-None-on-nonempty as suspect.
2. **Deviation-note ephemerality.** A "plan-deviation note" delivered as a pod-local log line + JSONL under `/workspace/logs/` dies at pod termination unless (a) the filename matches the poller's sentinel glob (`/workspace/logs/issue-<N>-*.json`) or (b) the event is uploaded as non-LFS text (which works even over quota). Grep poll_pipeline.py + upload-verifier.md for the event name — "the orchestrator will observe it" with zero wired observer is prose, not a channel.
3. **Opt-in flag interplay inerting the protection.** Routing/reroute keyed on a signal (`unknown never reroutes`) + a kill switch on the signal source means FLAG=1 + CHECK=0 is silently status quo while the user believes armed. Demand a one-time loud warning on the armed-but-blind combination.
4. **Soft-ceiling vs hard-wall conflation in abort messages.** A gate that aborts in the deliberate runway zone (soft < usage < wall) must not claim "doomed" — in that zone the old behavior SUCCEEDS. Policy is fine; wording + autonomous-session blockage cost should be named.

**How to apply:** these are usually implementer-absorbable Concerns (APPROVE), not REVISE — REVISE only if the mechanism would not fire in the very incident it was designed for under today's API behavior, or a failure path is WORSE than status quo (e.g. fail-soft private-check → False can false-abort a healthy private-target sweep; prefer fail-open on unknown).
