---
name: subsample-readmission-filter
description: A subsample drawn from a split-time id list can contain rows a later filter-bearing phase skipped/lost — re-apply the producer's admission filter + require primary-draw membership before consuming (issue #1738 r5)
type: feedback
---

A subsample drawn from a SPLIT-time id list can contain rows a later filter-bearing
phase (e.g. capture with a token-budget skip) rejected or never produced — the
downstream consumer must re-apply the producer's OWN admission filter (same render +
budget arithmetic, shared helper) and require primary-draw membership, skip+recording
rejects. Otherwise the reject goes production-fatal (e.g. a 14,217-token row at vLLM
add_request with max_model_len 8192 — issue #1738 Phase 4a, engine-core death read as
[phase=done]).

**Why:** filters live at the phase that applies them; id lists predate them.
**How to apply:** any resample/re-generation phase consuming ids carved before a
filter-bearing phase gets an admission gate + skipped-sidecar BEFORE generation.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Subsample re-admission filter](feedback_subsample_readmission_filter.md) — re-apply the producer's admission filter before consuming split-time id lists (#1738 r5)
