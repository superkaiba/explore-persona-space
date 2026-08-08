---
name: sha-pin-domain-mismatch
description: A sha pin lives in a DOMAIN (index arrays vs prompt strings vs file bytes) — recompute a reused pin from its PRODUCER's recipe before asserting any new derivation against it; a wrong-domain compare fails on every input and masquerades as upstream data drift (issue #1776 crash-fix cycle 4)
type: feedback
---

A sha pin lives in a DOMAIN. #1776's plan-§10 pins were sha256 digests of the
#779 `fixed_split(...)` **int64 INDEX arrays** (pure-RNG-reproducible); the
consumer compared a **prompt-string** digest (`N10._sha_prompts`) against them
— a wrong-domain compare that could never pass on ANY input, and the failure
read exactly like upstream corpus drift ("test-1000 sha drift"). There was no
drift: the fresh stream reproduced the frozen manifest anchor byte-for-byte.

**Rules:** (i) before asserting a NEW derivation against a reused pin,
RECOMPUTE the pin from its producer's recipe (here: rerun `fixed_split`, hash
the index arrays — exact match proves the domain) — never infer the domain
from the pin's variable name or its doc string; (ii) when a consumer
re-derives membership from a live stream, guard it with the producer's frozen
MEMBERSHIP sha in the SAME domain (the #779 n1m manifest `used_shas.round1`
prompt-sha), not by re-hashing the derivation in a different one; (iii) a
"drift" assert that fails on the FIRST-ever production run is a prime
wrong-domain suspect — check the domain before chasing upstream data; (iv)
strengthen, never relax: the #1776 fix went 2 asserts → 5 (three-domain proof
chain) with a fails-pre-fix pytest.

(Incident #1776 crash-fix cycle 4, pod-1776 p1_contexts, 2026-07-29: fix
commit `04ce114b8fb2b9439494dac3a1c77923fcedc940`; execute-forward sweep of 18
remaining phase entries found 0 further seam bugs.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [sha pins live in a DOMAIN](feedback_sha_pin_domain_mismatch.md) — recompute a reused pin from its producer's recipe before asserting; wrong-domain compare masquerades as data drift (#1776 c4)
