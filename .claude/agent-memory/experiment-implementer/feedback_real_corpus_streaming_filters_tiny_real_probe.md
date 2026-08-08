---
name: Real-corpus streaming filters — verify against REAL rows + bounded tiny-real probe
description: WildChat/LMSYS store FULL language names ('English', not 'en') and per-dataset moderation shapes; a filter chain written from assumed field semantics can reject 100% of rows while synthetic-fixture smokes stay green. Bounded tiny-real streaming probe + per-filter reject counters before any production corpus launch. #1092 P0.
type: feedback
---

Real-corpus streaming filters must be verified against REAL rows before any
production launch. #1092's P0 corpus build streamed WildChat-1M end-to-end
(~40 min) and kept 0 of ~1M conversations because the language filter
compared `row["language"].lower()` against `"en"` while WildChat AND LMSYS
store FULL names (`'English'`, `'Spanish'`); a required-kwarg TypeError sat
armed one branch later, unreached only because everything was rejected
first; and the plan's redaction/moderation filters were unimplemented —
every synthetic-fixture smoke stayed green through all of it.

Field semantics to never assume (verify per dataset via the datasets-server
rows API before writing filters): language fields may be full names or ISO
codes; WildChat carries top-level `redacted` + per-turn `redacted`/`toxic`
bools inside `conversation`, plus per-turn-ALIGNED top-level list columns
`openai_moderation {categories, flagged}` and `detoxify_moderation`
(continuous scores ONLY, no boolean) — NOT keys inside each turn dict;
LMSYS has its own shapes.

The fix pattern (do this in ANY new real-corpus streaming builder):

1. Run a **bounded tiny-real streaming probe** before production — a kept
   cap AND a TOTAL-streamed-rows cap (so a 0-keep chain terminates in
   seconds instead of streaming 1M rows), asserting kept > 0 per dataset.
2. Log **per-filter rejection counters** in the stream's `done:` line so
   the next 0-kept run names its rejecting filter instantly.
3. Pin the real row shapes in a **real-shape fixture test** (copy actual
   field structures; fake `load_dataset` only at the network boundary,
   signature-conformant).

Sibling lesson: `feedback_tiny_real_cpu_e2e.md` (#906) — same principle,
GPU-pipeline flavor. This is the data-ingestion flavor.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Real-corpus streaming filters need tiny-real probes](feedback_real_corpus_streaming_filters_tiny_real_probe.md) — WildChat/LMSYS store FULL language names + per-dataset moderation shapes; assumed field semantics rejected 100% of rows while synthetic smokes stayed green; bounded tiny-real probe + per-filter reject counters (#1092 P0)

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [real-corpus exact dupes break sha-keyed samples](feedback_real_corpus_exact_dupes_sha_sample.md) — dedup in-draw with pinned-row priority (#1768)
- [UltraChat prompt field case-variant](feedback_ultrachat_prompt_field_case_variant.md) — use messages[0] text + casefold-strip check
