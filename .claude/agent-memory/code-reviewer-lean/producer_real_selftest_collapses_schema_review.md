---
name: producer-real-selftest-collapses-schema-review
description: When a consumer script's selftest runs the REAL producer's selftest and consumes its outputs, schema-conformance review is executable — run the selftest instead of eye-diffing keys; certify trigger-dense count fixes via counts-only probes on the sha-pinned artifact
metadata:
  type: feedback
---

Two execution-first certification recipes from #2356 R1 g3 (2026-08-17), both
cheaper AND stronger than reading:

1. **Producer-real consumer selftest.** `issue2356_figures.py --selftest`
   intercepts `tempfile.mkdtemp`/no-ops `shutil.rmtree` around the FITS
   driver's own `_selftest`, then renders the full 21-entry registry from the
   REAL producer outputs. One run certifies every consumed JSON key across 6
   result files at once (rendered==all + resume-cached + empty-root probes).
   When a diff ships this shape, RUN it — a PASS retires the entire
   "does the consumer's schema match the producer?" question that otherwise
   needs dozens of key-by-key greps. If a diff does NOT ship this shape,
   ask why the consumer smoke uses a hand-mirrored fixture instead (that is
   the drift channel).

2. **Counts-only probe on a trigger-dense pinned artifact.** A P0 fix pinning
   distinct/dropped counts against a sha-pinned harmful bank is certifiable
   without paging item text: re-run the exact dedup algorithm (sha-keyed,
   first-wins, same add order) printing ONLY counts + per-axis dup counters +
   dup-set-size histogram. #2356: 2,758 attempts → 2,748 distinct / 10 dropped
   (7 pairs + one 4-way, all declarative_curiosity) reproduced the commit's
   claim exactly. Pair with the datasets-server `/info` features probe for
   column-rename fixes (schema names only, zero row text).

**Why:** the selftest run + two probes settled every major in-scope question
in ~4 tool calls; the alternative (schema greps across producer+consumer) is
slower and misses what execution catches (e.g. matplotlib API drift).

**How to apply:** on any consumer-of-artifacts diff (figure renderers,
aggregators, loaders), look for a `--selftest` that chains the producer;
run it live before hand-tracing schemas. Related: [[stats-reuse-driver-live-probes]],
[[tmp-rerun-zero-diff-analysis-artifact]], [[fails-pre-fix-probe-parent-commit]].
