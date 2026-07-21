---
name: Skippable-phase staging side-effects
description: Pre-stage every fan-out-shared input idempotently in the PARENT at phase entry — staging that lives only inside a skippable phase (pilot, gated setup) vanishes on resume/seed paths
type: feedback
---

Shared-input staging that lives only inside a SKIPPABLE phase (a pilot, a gated setup step) silently disappears the moment a resume/seed path skips that phase — N fan-out workers then race concurrent hf_hub_download()s into one shared local_dir (#1315 class: a later worker's locked unlink deletes a sibling's published file).

**Why:** #1482's --seed-partial Gate-B skip also skipped the P2-pilot's SAE-weight staging; a warm 4xH100 relaunch died in the first worker (`unable to open ae.pt`), costing a pod cycle + a fix round.

**How to apply:** when writing any phase driver with a resume/seed/skip path, enumerate every download/staging side-effect of the skippable phases and hoist them to an idempotent, retry-enveloped pre-stage in the parent process, unconditionally at phase entry, before any worker fan-out.
