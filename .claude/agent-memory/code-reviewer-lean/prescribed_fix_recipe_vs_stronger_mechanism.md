---
name: prescribed-fix-recipe-vs-stronger-mechanism
description: When a prior critique prescribes a fix shape (e.g. key caches on manifest hash), grade the realized fix against the INTENT, then sweep for bypass consumers of the un-fixed artifact
metadata:
  type: feedback
---

A revision commit may implement a DIFFERENT mechanism than the prior
critique's prescribed recipe. Grade intent, not recipe conformance — then
close the two gaps the substitute mechanism opens.

**Why:** #2479 r2 g3 — codex r1 prescribed "key caches on manifest hash +
exclusion recipe"; the fix instead left slice caches UNFILTERED on disk and
applied the reservation exclusion at a single load choke point on EVERY
load (cache hits included). Strictly stronger: nothing filtered is ever
baked, manifest changes can never serve stale. Flagging "cache key lacks
manifest hash" would have been a false FAIL.

**How to apply:** (1) restate the prescribed recipe's parenthetical intent
(here: "a stale pre-fix cache must not serve reserved rows to a fit") and
test the realized mechanism against THAT; (2) a filter-at-load design is
only sound if the choke point is the ONLY reader — grep repo-wide for other
consumers of the now-deliberately-dirty artifact (unfiltered caches) and
for external callers of the choke-point function; (3) check no
pre-fix DOWNSTREAM outputs (fitted JSONs) survive on disk that resume-skip
past the fix boundary — the caches were fixed but output-presence resume
can still bless a contaminated pre-fix result ([[new_dial_missing_from_resume_regime]],
[[start_manifest_stale_artifact_done]]); (4) run the id-namespace probe on
any membership filter so it cannot pass vacuously
([[perfile_id_namespace_not_leakage]]).
