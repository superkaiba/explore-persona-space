---
name: producer-gate-asymmetric-input-validation
description: On derived-record rounds, diff the PRODUCER's input validation against the GATE/consumer that re-reads the same files. Producer-only completeness or duplicate checks leave the gate silently merging stale inputs.
metadata:
  type: feedback
---

When a round adds a frozen-record PRODUCER and a GATE that both enumerate the
same input files (shard summaries, checkpoints), diff their validation sets
line by line. #2658 r13: `build_cap_amendment` refused missing shards,
mixed shardXXofYY totals, and duplicate cells via `_shard_summary_paths`, but
`_gate_cap_hit` kept a bare `glob(f"{split}_shard*.json")` that silently
merges a stale same-split summary (last-read fraction wins). The realized dir
was clean, so tests and live runs never exposed it.

**Why:** the producer runs once and freezes a record. The gate re-reads the
raw inputs on every later run, so the gate is the surface a stale file
actually hits. A review that verifies the producer's fail-loud list and stops
certifies the wrong side.

**How to apply:** for each input-validation check in the producer (grep its
raises), ask where the same files are re-read and whether that reader has the
same check. Also probe loader symmetry: paired loaders of one record in two
modules should assert the same schema VALUE, not just field presence
(generate.load_cap_amendment vs power.load_cap_amendment_record). Related:
[[spend_consumer_accepts_partial_shard_set]], [[gate_threshold_vs_shard_config]],
[[silent_get_default_beside_fixed_keyerror]].

Second reusable probe from the same round: a "fingerprints byte-identical to
the pre-amendment payload" resume claim is cheaply certified against REALIZED
artifacts, not just the unit test: recompute the fingerprint with the NEW code
for a sample of stored cell bodies and compare to their stored `fingerprint`
field (12/12 matched here). A determinism test alone cannot prove parity with
the frozen on-disk payloads. Related: [[fingerprint_resume_ids_not_content]].

Closure round (#2658 r14, validated): grade a CONDITIONAL integrity check by
what binds UNCONDITIONALLY around it. The gate's split-total cross-foot ran
only when a shard00of01 manifest existed beside a multi-shard set, yet no
wrong tree could sneak through, because the per-summary same-named-manifest
requirement plus completeness plus duplicate-cell checks bind on every path.
Certify such a fix three ways in one round: (1) HEAD tests re-run against the
PARENT blobs in an isolated /tmp tree (module __file__ printed, new symbols
absent, r13 failure shape reproduced), (2) live injections on a cp -a REAL
copy of the realized tree (never hardlink when the checked code may rewrite
files in place), (3) a direct read of the realized twin-manifest layout (the
per-shard manifests and a split-total manifest coexist, so a naive directory
sum doubles the declared total). Also probe superset shapes: a missing-only
completeness check (range - indices) passes an out-of-range extra index, so
ask whether downstream conjunction or filename grammar closes it before
ranking severity. Related: [[fails_pre_fix_probe_parent_commit]].
