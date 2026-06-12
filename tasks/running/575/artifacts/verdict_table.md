# Task #575 — per-task verdict table (URL-existence sweep, 2026-06-12)

Census: 50 tasks (plan-time list ∪ run-start `ls tasks/awaiting_promotion/` — identical sets; timestamp 2026-06-12T20:30:53Z, post `git fetch origin`). No arrivals, no departures during the sweep.

**Headline (scoped):** all 50 census bodies PASS verifier checks 4b (`Figure URL resolvable`), 8 (`Reproducibility URL permanence`), and 8b (`Reproducibility artifact URLs exist`), with 328 figure URLs + 936 Reproducibility artifact URLs probed (= 1264 total) and **0 unverified/unprobed** (no `unverified` notes on any PASS line; every body probed >0 URLs in both existence checks — no zero-probed legacy bodies). Fenced URLs and non-TL;DR-section images are outside the verifier's scan scope. This is NOT a claim that "the backlog is URL-clean" beyond those checks' scopes.

**Repairs applied: 0** (no in-scope FAILs found). Claim marker, write-back, diff/CAS gates, and §3e re-verification all vacuous.

**Out-of-scope FAILs (recorded, NOT repaired — pre-existing, out of scope per #575 clarifier):**
- #585: `Reproducibility Context provenance row` — recorded origin data exists (`## Provenance` in original-body.md) but `## Reproducibility` has no `**Context:**` row. Shape/provenance check, not a URL-existence failure.

**WARNs observed (not FAILs; recorded for the promotion-time reader):** 37 tasks carry `[WARN] Reproducibility Context provenance row — missing **Context:** row (no recorded origin data)`; 6 tasks carry `[WARN] Goal-of-experiment field — missing frontmatter goal:` (soft, enforced at /issue Step 0c).

| task | overall | URLs actually probed | in-scope FAILs | repair + provenance | post-repair verdict | out-of-scope FAILs |
|---|---|---|---|---|---|---|
| 464 | PASS | fig=5, repro=32 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 472 | PASS | fig=5, repro=14 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 480 | PASS | fig=13, repro=45 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 491 | PASS | fig=8, repro=11 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 504 | PASS | fig=13, repro=32 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 505 | PASS | fig=4, repro=37 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 518 | PASS | fig=7, repro=16 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 521 | PASS | fig=10, repro=42 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 523 | PASS | fig=5, repro=13 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 528 | PASS | fig=3, repro=14 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 531 | PASS | fig=4, repro=15 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 532 | PASS | fig=5, repro=15 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 533 | PASS | fig=14, repro=36 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 536 | PASS | fig=5, repro=8 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 537 | PASS | fig=11, repro=16 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 538 | PASS | fig=7, repro=27 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 539 | PASS | fig=5, repro=13 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 540 | PASS | fig=11, repro=40 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 541 | PASS | fig=5, repro=18 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 542 | PASS | fig=7, repro=15 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 543 | PASS | fig=6, repro=10 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 546 | PASS | fig=5, repro=20 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 552 | PASS | fig=12, repro=37 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 553 | PASS | fig=8, repro=23 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 555 | PASS | fig=5, repro=14 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 556 | PASS | fig=3, repro=10 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 557 | PASS | fig=7, repro=15 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 558 | PASS | fig=3, repro=9 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 559 | PASS | fig=6, repro=28 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 560 | PASS | fig=8, repro=12 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 561 | PASS | fig=5, repro=13 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 562 | PASS | fig=4, repro=11 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 563 | PASS | fig=5, repro=19 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 568 | PASS | fig=3, repro=3 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 570 | PASS | fig=6, repro=14 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 571 | PASS | fig=7, repro=19 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 585 | FAIL (1 of 25 checks failed) | fig=5, repro=22 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | Reproducibility Context provenance row |
| 591 | PASS | fig=7, repro=9 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 594 | PASS | fig=5, repro=18 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 597 | PASS | fig=8, repro=23 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 599 | PASS | fig=6, repro=13 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 600 | PASS | fig=5, repro=11 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 601 | PASS | fig=7, repro=18 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 602 | PASS | fig=6, repro=28 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 603 | PASS | fig=6, repro=17 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 604 | PASS | fig=10, repro=17 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 605 | PASS | fig=7, repro=16 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 606 | PASS | fig=5, repro=10 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 608 | PASS | fig=5, repro=11 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
| 611 | PASS | fig=6, repro=7 | - | none — no repair needed (no provenance rows) | PASS (all URLs verified) | - |
