---
name: hook-timeout-semantics-fail-open
description: Claude Code hook plans — docs pin command-hook default timeout 600s and timed-out PreToolUse = NO decision (fail-open); tiny-timeout verifications are vacuous unless the timeout provably FIRED (#2115)
metadata:
  type: feedback
---

Reviewing any plan that adds/edits `timeout` on Claude Code hooks or asserts
hook-stall semantics (#2115): check the plan's assumptions against the live
docs at `https://code.claude.com/docs/en/hooks` FIRST. Two documented facts
that refuted #2115's plan-v2 premises:

1. **Default `command`-hook timeout is 600 s** ("Defaults: 600 for `command`,
   `http`, and `mcp_tool`"), so "no `timeout` field ⇒ unbounded by
   construction" is FALSE — an absent timeout means bounded-at-600s. A
   multi-hour pending-call wedge therefore implies broken/ineffective
   cancellation (e.g. fork-heavy guards whose children inherit the stdout
   pipe past the cancel) or a non-hook mechanism — not plain hook cost.
2. **A timed-out PreToolUse `command` hook does NOT block the tool call**
   ("the hook renders no decision... the call continues through the normal
   permission flow, so don't count on a stalled hook to act as a gate") —
   i.e. documented FAIL-OPEN. Adding a SHORTER explicit timeout to a
   fail-closed security guard LOWERS the padding-bypass threshold (any argv
   big/slow enough to push the guard past its timeout skips the guard
   silently, in bypassPermissions especially). The bypass already exists at
   the 600 s default; a 60 s timeout makes it ~10× cheaper.

**Why:** #2115 v2 gated its prong on verifying "timed-out = DENY" via a
"deliberately tiny timeout on one guard + issue a blocked command + confirm
still blocked" — vacuous as designed: guards complete in ~0.1 s on small
commands, so they DENY normally before any tiny timeout fires, and the probe
false-confirms fail-closed.

**How to apply:** REVISE any hook-timeout verification that does not
(a) force the hook past its timeout (sleep wrapper or a large-argv command
off the plan's own cost curve), (b) capture positive evidence the timeout
FIRED (hook-cancel warning / start-sentinel-without-end-sentinel), and
(c) read the outcome from an observable side effect of the gated command —
run in a FRESH session (hook config snapshots at session start; in-place
settings edits are inert for the live session). Pre-register the documented
fail-open semantics as the expected outcome; a probe result claiming
fail-closed contradicts the docs and is presumed a probe bug. Also size any
explicit timeout off LOAD-ADJUSTED cost (the shared VM's measured ~6×
contention, #2054), not an unloaded pilot.
