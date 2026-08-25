---
name: dial-added-fingerprint-arms-refuse-on-relaunch
description: A revision that adds regime dials to resume fingerprints AND wires an automated relaunch that flips those dials makes the relaunch collide with the producer's refuse-to-mix arm — trace every relaunch into the mismatch branch (#2546 r2 g1)
metadata:
  type: feedback
---

When one round both (a) adds regime dials (smoke/fallback flags, row-set sigs)
to resume fingerprints to fix resume contamination, and (b) wires an automated
relaunch/retry leg that CHANGES those dials, trace the relaunch into the
producer's fingerprint-MISMATCH branch. Three branch polarities, audit each:
REFUSE (raise "refusing to mix — use a fresh out-root") makes the relaunch
dead-on-arrival unless the leg wipes/renames the artifact root first;
REGENERATE (overwrite) is correct; SILENT-ACCEPT (a checkpoint layer keyed on
row-id sets only, no regime key) mislabels old-regime artifacts under the new
fingerprint.

**Why:** #2546 r2 commit 917988a191 fixed the smoke-resume-contamination
Critical by adding smoke+fallback dials to `gen_fingerprint`, then added an
rc=4 fallback smoke relaunch into the SAME smoke out-root — every corpus's
meta existed under the no-flag fp, so the relaunch always died on the
refuse-to-mix raise (crash loop across resumes; the r1 "Unaddressed case" had
predicted it). The same commit's per-stage gen checkpoints validated row-id
sets only — the silent-accept sibling.

**How to apply:** on any revision round whose commit message pairs
"fingerprint/namespacing fix" with "fallback/relaunch routing", grep the
producer for the mismatch raise, then walk the relaunch leg: does anything
wipe or re-root the prior artifacts before the flag-flipped rerun? Also check
whether any lower checkpoint layer resumes on content-blind keys. Related:
[[size-match-resume-skip-npz]], [[new-dial-missing-from-resume-regime]].
