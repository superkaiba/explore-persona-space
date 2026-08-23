---
name: new-dial-missing-from-resume-regime
description: A commit adding a scale/pilot CLI dial (--sae-steps/--fit-n/--n-perm) must also add it to the regime/config-hash resume key; and a finals-presence skip is defeated by a two-stage JSON write (#2476 R1 g2)
metadata:
  type: feedback
---

When a diff ADDS a reduced-scale dial (`--sae-steps`, `--fit-n`, `--n-perm`,
`--n-boot`), grep the driver's regime/fingerprint builder (`_regime`,
`config_hash`) for each new dial. If absent, a dial-capped run's artifacts
satisfy resume-skip under a config_hash IDENTICAL to production — the plan's
"mismatch ⇒ retrain, never skip" predicate is blind to exactly the dial the
commit added (#2476 R1 g2: unit 1 hashed its own dials, unit 2's four escaped).
Compounding tell: a steps-capped partial epoch checkpointed as
`epoch_done = epoch + 1`. Related: [[additive-edit-flips-wholefile-resume-key]]
(inverse failure), [[rc-halt-not-resume-idempotent]] (same round: G4 FAIL wrote
weights+log+gates then exited, so the presence-skip erased the halt AND the
never-run upload).

**Why:** the regime key is the registered guard for resume-skip; every
output-affecting dial outside it is a silent pilot→production contamination
channel, and the presence-skip variant also swallows post-gate side effects
(uploads) that run after the skipped files land.

**How to apply:** for each new argparse dial in a diff, trace it into the
regime hash AND check every phase's resume-skip branch re-applies gate
verdicts / re-drives uploads. Separately, list the files in a finals-presence
skip set and check none is ENRICHED (keys added) after it first lands — a
crash in the enrich window resume-skips the unenriched version (#2476:
tier_tests_b.json bridge/pile keys written after bridge_b.npz completed the
finals set).
