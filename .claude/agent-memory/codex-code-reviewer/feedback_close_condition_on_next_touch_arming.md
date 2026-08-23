---
name: close-condition-on-next-touch-arming
description: When an open reconciler-downgraded concern records a close condition of the form "fix X at :LINE on next touch", and the round's diff touches that file/function, the composer greps the sites at compose time and hands Codex a three-way arming adjudication — never pre-resolves it and never lets it escalate past the concern's own severity
metadata:
  type: feedback
---

On #2477 r4 (2026-08-23, fresh-delta C0 round) the open concern
`paid-phases-not-idempotent` carried a reconciler-downgrade evidence field
ending "Close condition: `is True` at :1551 + the :1674 sibling on next
touch." The round's diff EDITED `phase_judge_pilot` (path parametrization +
cache slug — the hunk spans the guard region) while the truthy
`prior.get("passed")` lines rode as unchanged context, and HEAD still lacked
the `is True` hardening.

Rule: at every compose, scan the open concerns' `evidence` fields for
"on next touch" / "next round touching X" close conditions and grep whether
the round's diff touches the named file/function. When it does:

1. Recompute the recorded line anchors against HEAD (they are pre-round
   frames — say so; a moved line is never a finding).
2. State the composer observations neutrally (which lines are unchanged
   context vs edited; whether the named fix literal landed) — grounding,
   not a ruling.
3. Hand Codex a THREE-WAY status vocabulary: `VERIFIED-ADDRESSED` (fix
   landed) / `ARMED-NOT-ADDRESSED` (touch binds, fix absent) /
   `STILL-BINDING-NOT-ARMED` (touch does not bind — e.g. "touch" read as
   the gate lines themselves, which rode as context), grounded in the diff
   hunks.
4. Severity fence: armed-and-unmet inherits the concern's OWN
   reconciler-set severity (here CONCERN) — a status-line re-raise + prose,
   never a fresh Critical and never a FAIL ground by itself.

**Why:** the ambiguity (touch = file? function? the exact lines?) is an
adjudication, not a compose fact — pre-resolving it either lets a violated
close condition ship silently or manufactures a blocker the reconciler
already downgraded.

**How to apply:** any round where `list-concerns` shows an open id whose
evidence names a conditional close keyed on future edits. Sibling of
[[brief-named concern adjudication]] (there the BRIEF names the concern;
here the ledger's own close condition arms it).
