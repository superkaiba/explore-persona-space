---
name: rc-halt-not-resume-idempotent
description: A designed rc-halt (kill criterion) that marks the halted unit's phase COMPLETE in the resume manifest is silently erased by any relaunch — check halt idempotency across the resume predicate
metadata:
  type: feedback
---

When a driver implements a plan-§7 kill criterion as "persist artifacts →
write halt report → exit distinct rc", check whether the halted unit's phase
block is ALSO recorded complete in the resume manifest. If yes, any relaunch
(watcher crash-recovery, `--phase all` re-run) resume-skips the unit, the
halted list recomputes empty, and downstream phases run on the condition the
halt existed to stop — the kill criterion is one-shot, not idempotent.

**Why:** #2222 R1 g1: `run_gen` recorded `p2_gen` complete (with
`cap_hit_final` over the bar) before `os._exit(RC_CAP_HIT)`; the resume-skip
branch never re-read `cap_hit_final`, so a relaunch would capture the biased
exact-ΔP reference silently. Sibling of [[gate-threshold-vs-shard-config]]
(gates going dead under config drift) — here the gate goes dead under the
driver's OWN resume path.

**How to apply:** for every distinct-rc designed halt in a diff, trace the
relaunch: does the halt condition re-derive from persisted state (manifest
field re-checked on resume-skip, or an entry-time check of the halt report),
or only from freshly-computed work the resume skips? The fix shape: re-read
the persisted per-unit metric in the resume-skip branch, or refuse the next
phase while the halt report exists absent an explicit override flag.
