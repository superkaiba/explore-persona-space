---
name: gate-artifact-mode-blind-arming
description: A verdict-keyed arming resolver (verdict=="PASS") that ignores the artifact's mode/rig field is armed by the dispatcher's OWN smoke branch writing a tiny-mode PASS to the same path (#2389 R1 g5)
metadata:
  type: feedback
---

When a production feature is armed by reading a gate artifact (`verdict ==
"PASS"`), check THREE things beyond the verdict key: (1) does the artifact
carry a mode/rig field (`"tiny"`/`"smoke"`/`"production"`) that the RESOLVER
never reads; (2) does the gate script write the SAME filename to the SAME
out-root (and HF prefix) in its smoke/tiny modes; (3) does the dispatcher's
`--smoke` branch (grep `GATE4B_EXTRA`-style arrays / `case *--smoke*`)
invoke the gate in a substituted mode with the default shared out-root.
All three true ⇒ a standard smoke→production sequence on default paths arms
production on smoke-rig evidence before the production-device battery lands
— the #1727 smoke-value-leak class landing on the arming SEAM rather than
the gated code.

**Why:** #2389 R1 g5 — `_resolve_share_prefill` checked only
`verdict=="PASS"`; dispatch.sh's smoke branch ran the gate `--tiny` into the
default `/workspace/issue2389_out/gates/`, `--share-prefill` defaulted
`auto`, and t0 anchor workers froze armed=True before the concurrent
production battery finished. Partial mitigations (cross-regime done-record
raise, stale-width sweeps) did not cover the gates dir or width-mismatched
smoke runs; the armed state was log-only (absent from regime fp +
done-records), so a later battery FAIL could not identify affected shards.

**How to apply:** on any pin/gate-arming review, after confirming the happy
path (both-conditions gating, freeze-once, fail-open on absent/FAIL), probe
artifact AUTHENTICITY: `jq .mode` the artifact the gate writes per mode, grep
the resolver for that field, and trace the dispatcher's smoke arm for
same-path writes. Also check the armed decision is recorded in durable
per-shard provenance, not only logs. Fix shape: resolver requires
mode==production (one line) + a unit test writing a tiny-mode PASS.
Related: [[smoke_shard_namespace_only_done_files]],
[[force_flag_not_reaching_resume_state]].
