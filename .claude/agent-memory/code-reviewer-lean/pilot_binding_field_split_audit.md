---
name: pilot-binding-field-split-audit
description: Two-direction audit for a pilot-PASS param-binding list — each compared field must be pilot-matchable (else deadlock), each exempt field bounded by a compared field or owned by another resume key; plus the verdict-floor combo enumeration.
metadata:
  type: feedback
---

When a production gate requires a pilot PASS whose recorded params match on a
FIELD SUBSET (#2569 r3 `_PILOT_BINDING_FIELDS`), audit BOTH directions:

- **Compared fields ⇒ pilot-matchable.** For each compared field, confirm the
  smoke/pilot run produces exactly that value at pilot scale (same CLI flags,
  layers resolved through the same helper). A field the pilot cannot match
  deadlocks production behind an unsatisfiable gate. Also confirm the pilot
  RECORD stores exactly the compared subset (no scale-dependent field like
  `rows` leaking in).
- **Exempt fields ⇒ bounded or owned.** For each exemption, demand one of:
  (a) its effect on the pilot's MEASURED quantity is bounded by a compared
  field (#2569: `template_sha`'s token delta is capped by the compared
  `max_capture_tokens`, same order as unavoidable row-content variation —
  the slice fingerprints are structurally exempt since the pilot IS a row
  subset); (b) its correctness effect is owned by ANOTHER key (template_sha
  sits in the chunk-store regime ⇒ wipe+recapture); (c) it is inert at pilot
  scale by construction — VERIFY the constant relation (`chunk_rows` default
  2000 > `SMOKE_ROWS_CEILING` 256), not just the claim.
- Finalize-side binding should read the params from the artifact's OWN
  regime (what capture ran with), never the finalizing argv, with fail-loud
  on pre-binding regimes.

**Why:** #2569 round 3 shard 2 — the field split is the whole gate: too
strict deadlocks, too loose leaves the pilot hollow. The two-direction walk
plus the constant-relation check settled every exemption in minutes.

**How to apply:** any diff adding `*_BINDING_FIELDS` / a pilot-record
equality precondition. Companion probe for thrice-fixed verdict predicates
(same round's H2b floor): enumerate every reachable (degenerate, well-posed,
parity-passing) count combo, confirm each lands a DISTINCT token, and grep
every consumer of any tri-state field (`None` where a bool was expected) —
then certify fails-pre-fix by executing the PARENT blob
([[fails_pre_fix_probe_parent_commit]]).
