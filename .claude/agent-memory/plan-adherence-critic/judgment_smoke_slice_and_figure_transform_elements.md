---
name: judgment-smoke-slice-and-figure-transform-elements
description: PASS_PARTIAL "arms_stubbed" can mislabel plan-conformant non-coverage — judge against the plan's own smoke-slice definition; and a manifest figure can render while silently dropping declared transform ELEMENTS (CI spec, declared source artifact) — check per-element, not per-figure-exists
metadata:
  type: feedback
---

Two adherence judgment calls from #2162 round `turn-boundary-multipatch` (2026-08-14):

1. **A smoke marker's `arms_stubbed` list is not automatically a deviation.** Judge each
   listed arm against the plan's own smoke-slice definition (§ "Smoke run"): an arm the
   plan's pre-registered slice deliberately EXCLUDES (e.g. §4.6 smokes one joint-steered +
   one sweep-shuffled block only, assigning crosstype coverage to the injection-gate spots)
   is plan-CONFORMANT non-coverage, not a stub — no smoke-stub authorization is needed
   because nothing was substituted (FALLBACK/not-run ≠ a toy implementation). Only an arm
   the plan's slice INCLUDES and the smoke skipped (there: the margin block slice) is a
   genuine partial, judged on whether the reason + completion path are stated.
   **Why:** the reviewer prompt framed all four "stubbed" arms as needing plan
   authorization; three were the plan's own slice design.
   **How to apply:** before flagging PASS_PARTIAL stubs, quote the plan's smoke-slice
   sentence and partition the listed arms into in-slice vs out-of-slice.

2. **Check manifest figure TRANSFORM ELEMENTS individually, not just that the figure
   renders.** A committed figures script can produce every manifest figure id while one
   figure silently drops declared elements — in #2162: `tb_rawscale` rendered but omitted
   its declared bootstrap 95% CIs and read from recomputed tables instead of the declared
   `rawscale_tb.json` source, whose producer was never wired into the pipeline (the
   parametrized tool existed; no step invoked it). Grep the pipeline for a producer of
   EVERY manifest-declared source artifact, and diff each transform's named elements
   (CI + B + seed, pool splits, overlay source) against the figure function body.
   **Why:** "all figures render" (a passing figures smoke test) masked a dropped planned
   P5 output + a dropped CI element on exactly one figure.
   **How to apply:** per tb figure, two checks — (a) declared `source` paths each have a
   producer write-site; (b) each transform noun (CI, panels, overlay) has a code
   counterpart.
