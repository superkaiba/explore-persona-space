---
name: force-flag-not-reaching-resume-state
description: A --force/redo flag that only bypasses the FINAL-output skip while sidecars/chunk stores/partials resume on count-only fingerprints silently half-redoes — trace every resume surface the flag must invalidate (#2379 R2 g1)
metadata:
  type: feedback
---

When a revision round adds BOTH an idempotency skip (`--force` overrides) AND
crash-resume machinery (rollout sidecars, chunk checkpoint dirs, .partial
streaming state), grep every consult site of `args.force`: if it gates only
the final-bundle skip, a forced re-run still resumes from sidecars/chunks
whose fingerprints match — and count/name/sampling-only fingerprints (no
weights/adapter identity) match ACROSS a retrain. Concrete #2379 shape:
`--phases "p1 p4" --force` retrains the adapters, then the forced ceiling
phase reuses the PRIOR model's on-policy rollout sidecar (fingerprint =
model NAME + counts + sampling) and teacher-forces the NEW model on the OLD
model's rollouts — the on-policy DV silently breaks with only an INFO
"reusing persisted sidecar" line.

**Why:** #2379 R2 g1 (2026-08-19, commit 5cd74c15eb): the resume machinery
was itself the round's fix for r1's terminal-only-persistence Major — fresh
fix code re-created the stale-resume class one layer down. Sibling of
[[start-manifest-stale-artifact-done]] (presence-done binds to no
fingerprint) — here the fingerprint EXISTS but binds to no producer
identity, and the operator's designated invalidation lever doesn't reach it.

**How to apply:** on any diff adding a redo flag + resume state: (1) grep
the flag's consult sites vs the full resume-surface inventory (sidecars,
chunk dirs, partials, caches); (2) check each fingerprint for a
producer-identity token (adapter/weights hash, input content digest) — pure
counts/names/params match across retrains and bank edits; (3) demand
force ⇒ wipe-all-resume-state (few lines) or identity-bearing fingerprints.
Also check the same commit for skip-predicate ASYMMETRY across sibling
scripts (sweep got a regime match, capture got presence-only).
