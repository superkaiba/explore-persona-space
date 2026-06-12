---
name: Gate-split amendments + tripwire calibration circularity + daemonized-driver heartbeat hole
description: When an unsatisfiable conjunctive HALT gate is split (structural / fitness / observation), calibrating the structural tripwire on known-good observed data is legitimate, not circular; check the fallback's eval-surface + deliverable-glob drift and the heartbeat-AND-pid poller hole (#601 v3)
type: feedback
---

Three reads from #601's plan-v3 amendment (2026-06-11), where a registered
all-cells-within-1-nat reproduction HALT gate proved unsatisfiable on intact
tooling (saturated cells read 7 nats under committed; anchors marginal):

1. **Gate splits are the right rescue, and "thresholds grounded on the same
   data they gate" is NOT a disqualifying circularity** when the gate's
   re-scoped role is an eval-path-integrity TRIPWIRE for future re-entries:
   calibrate the band just outside the known-good signature (e.g. 1.7×
   observed worst-case AND half the smallest adjacent-level gap = a
   principled no-aliasing bound). Verify force-preservation against the
   original failure class explicitly: un-applied adapters must still fail
   ≥1 of the split criteria (identical-reads alarm, low-dose band, dose
   ordering) — orthogonal detectors can preserve or improve force even when
   the saturated cells leave the HALT. The one framing trap: a future PASS
   on the split gate must never be narrated as "parent levels transfer" —
   that question lives in the fitness gate / recorded observation.

2. **When a reuse-fitness gate fails and a budgeted fallback fires, trace
   the fallback unit's EVAL SURFACE and the §6.5 deliverable globs** — a
   dense-phase cell promoted to double duty as a comparison arm usually
   lacks the full on-policy panel the other arms get (secondary DVs lose
   the middle point) and the "+N files if fallback fires" glob notes go
   stale in count and location. Bookkeeping/upload-verifier concerns, not
   REVISEs, when the headline DV (terminal source level) is covered.

3. **Self-daemonizing launch contracts (`setsid --fork` + self-written pid
   file) close SIGHUP orphaning, but check the poller contract's boolean:**
   "dead ⇔ no heartbeat AND pid not alive" has a hole under SIGKILL — the
   heartbeat subshell survives (traps don't fire) and keeps emitting, so
   the conjunction never declares death. Fix = heartbeat checks parent
   liveness per emit, or pid-not-alive is authoritative alone. Concern not
   Must-Fix when the dominant observed death path (ssh teardown) is
   structurally closed and relaunch is idempotent/cheap.

**How to apply:** Methodology lens on amendments that split/re-scope HALT
gates after observed unsatisfiability, re-pin references in-task, or harden
launch contracts. Also check the grounding eval JSON is actually committed
(code commits get cited as if they contained the data).
