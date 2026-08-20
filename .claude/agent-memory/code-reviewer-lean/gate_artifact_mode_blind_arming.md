---
name: gate-artifact-mode-blind-arming
description: A verdict-keyed arming resolver (verdict=="PASS") that ignores the artifact's mode/rig field is armed by the dispatcher's OWN smoke branch writing a tiny-mode PASS to the same path (#2389 R1 g5); freshness variant — a probe-REPORT gate with no shard/revision binding is satisfied by a stale or local-synthetic PASS (#2389 R2 g4)
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

**Freshness variant (#2389 R2 g4):** the fix landed (`mode != "production"`
disarms, missing mode = None ⇒ disarmed — unarmed-on-uncertainty confirmed),
but the SAME round's new pre-spend probe-report gate repeated the class in a
new artifact: `require_consumer_probe` keyed on `verdict=="PASS"` + leg
presence only, ignoring the report's `staged`/`gate_shard` fields and carrying
no revision/listing/hash — so (a) a stale PASS survives an in-round anchor
REGENERATION (capregen) that can introduce exactly the jsonl/tensor key
misalignment the probe certifies against, and (b) a `--local-anchors-dir` run
over synthetic data writes a PASS to the PRODUCTION default report path (the
report default derives from `--in-root` even when staging is skipped; the e2e
test itself demonstrates a synthetic report satisfying the real gate). Review
checklist addition: for any report/artifact-keyed pre-spend gate, ask what
INVALIDATES the artifact — a regeneration flow that never quarantines the
gate artifact, plus a gate with no freshness/shard binding, = stale-bless.
Cheap remedies: record discovery listing + repo revision and re-list at gate
time; producer-side quarantine on regen; or require staged:true behind a
recorded local-override flag. Adjudicated CONCERNS not FAIL when the required
refusal wiring itself is complete, fail-closed, and tested.
Related: [[smoke_shard_namespace_only_done_files]],
[[force_flag_not_reaching_resume_state]].
