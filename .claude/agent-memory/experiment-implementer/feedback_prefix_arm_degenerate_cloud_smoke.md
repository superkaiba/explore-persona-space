---
name: prefix-arm degenerate cloud kills spectral smoke
description: Single-context capture slices make prefix-arm activation-shift clouds identically zero (one unique row per context) — Σσ²==0 spectral fail-fasts then kill the whole driver; gate on structural unique rows + size smokes to ≥2 contexts
type: feedback
---

A prefix-arm Δx cloud has ONE unique row PER CONTEXT (prefix activations are
causally independent of the question that follows), so any single-context
capture slice — typical of a tiny smoke — makes the row-centered cloud
identically zero, and a Σσ²==0 spectral fail-fast (e.g. #653
`spectral_dvs_from_lambda`) kills the whole driver at the geometry phase
(incident #1112 smoke, 2026-07-07, att-20260707-205546).

**Why:** the fail-fast is right for unexpected degeneracy but a
single-context prefix arm is MECHANICALLY degenerate — expected structure,
not an error.

**How to apply:** in geometry/analysis drivers, gate on the arm's EXPECTED
structural unique-row count BEFORE calling spectral DVs: < 2 unique rows →
emit an explicit `degenerate: true` record (unique_rows, reason, null DVs) —
never a silent skip or coerced zero; ≥ 2 unique rows that still zero out →
keep the raise. And size smoke capture panels to ≥ 2 contexts so the smoke
exercises the real nondegenerate spectral path instead of short-circuiting
every prefix cloud to degenerate records. (#1112 fix commit 6771d578cd.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [prefix-arm degenerate cloud kills spectral smoke](feedback_prefix_arm_degenerate_cloud_smoke.md) — single-context smoke slices zero prefix-arm clouds (1 unique row/context); gate on structural unique rows + ≥2-context smokes (#1112)
