---
name: metric-twin-and-paired-default-parity
description: Producer emits metric twins (euclidean/cosine) — diff the consumer's collected key set against the twin pairs; and every new consumer flag default must be diffed against the producer wrapper's dir default + durability class
metadata:
  type: feedback
---

Two catches from one "registered-analysis completion" commit (#2479 r2 g4):

1. **Metric-twin collection parity.** When a producer emits paired metric
   twins (`<key>` + `<key>_cosine`, euclidean/cosine kNN, R²/acc@1), grep the
   producer's emission block and diff the CONSUMER's collected key set against
   the full twin roster. The tell: an adjacent read (rung-4) collected both
   twins while the newly added control (identity+bias) collected only one —
   asymmetry against a sibling read is the cheapest detector.
2. **Paired-default parity + durability.** A new consumer flag with a default
   (`--axis-items-stats-dir` → `<eval-dir>/axis_items`) must be diffed against
   the PRODUCER wrapper's default for the same artifact (`ITEMS_DIR=data/...`
   in the phase shell script). Check three things: (a) the two defaults name
   the same dir; (b) the producer dir's durability class (gitignored `data/*`
   + absent from every upload/verify prefix = a fresh-checkout consumer can
   NEVER find it); (c) the e2e fixture's placement — a fixture writing inputs
   at the CONSUMER default masks the producer mismatch (sibling of
   [[smoke-fixture-authored-with-consumer-keys]]).

**Why:** both were residuals of an r1 "registered-analysis-incomplete" Major
that the r2 commit claimed to complete; each survived tests because the
fixture was consumer-shaped. Related: [[fork-layout-default-mirrored-onto-parent]],
[[staging-gate-single-phase-silent-fallback]].

**How to apply:** on any "completes registered reads" / "collects producer
outputs" commit, (1) grep the producer emission for ALL sibling keys of each
collected key; (2) for every new flag default, open the phase wrapper that
produces the artifact and compare defaults + gitignore/upload coverage.
