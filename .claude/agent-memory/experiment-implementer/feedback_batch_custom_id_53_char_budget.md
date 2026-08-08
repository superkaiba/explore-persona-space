---
name: Batch-API judge item ids — charset ^[a-zA-Z0-9_-]$ + 53-char budget, alias/hash-compact + id_map
description: Anthropic Batch custom_ids must match ^[a-zA-Z0-9_-]{1,64}$ — caller keys with dots/colons/slashes 400 the first batches.create (#1776 c9), and eval/batch_judge's encoder appends 11 chars so the caller budget is 53 (#1415); alias at the caller seam (bijective, collision-asserted) + persisted id_map + pre-submit validation in the dispatcher, dry-run included
type: feedback
---

Judge item ids passed to `eval/batch_judge` become Anthropic Batch
`custom_id`s via an encoder that appends 11 chars (`__NNNNN__NN`), and the
Batch API enforces `^[a-zA-Z0-9_-]{1,64}$` — BOTH a charset and a length
constraint. The CALLER's item id budget is 53 chars AND `[a-zA-Z0-9_-]`
only: keys carrying dots, colons, or slashes (stratum names like
`evil_a0.5`, `::` separators, hierarchical `/` paths) 400 the FIRST
`batches.create` (#1776 c9), and a routing-only dry run can never catch a
charset bug — validate composed ids pre-submit in the dispatcher
(`judge_dispatch.validate_batch_custom_ids`, wired before the dry-run
return AND at `_run_batch_path` entry), so violations become instant named
pre-flight failures at zero API cost. Item ids built from hierarchical cell ids
(`gen1c/context/<pair_id>/L20/a0.5/d0`) overflow as soon as any pair/cell id
component grows (#1415: cross-pair cells hit 64 → 67-char custom_id →
fail-fast crash at enumerate, zero API calls).

**How to apply:** (1) Build judge item ids charset-safe and compact: either
a hash-compact key (`h<sha1(cell_id)[:12]>_d<draw>` — note `/` is ILLEGAL
in custom_ids) or a bijective alias of the readable key (`::`→`--`,
`.`→`p`; collision-ASSERTED over the full realized set — char substitution
alone is not injective); assert `len(item_id) <= 53` at build time and fail
loud on collisions.
(2) Persist a compact→full `id_map.json` (atomic write, BEFORE any judge
call) and rehydrate at result-reduction so downstream outputs stay keyed by
the full readable ids. (3) The rubric-keyed JudgeCache is unaffected (keys
on rubric/question/completion, never item ids). (4) On a mid-judge crash
with a partially-populated cache: QUARANTINE the work-dir before relaunch —
`judge_graded` packs n_draws identical completions per item and the cache
key is identical across draws, so a cache-hit replay collapses that rubric
to effective n_draws=1 (the rule-24 duplicated-draw trap). Worked impl:
`scripts/issue1415_judge.py` @ f779a83aba.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Batch-API judge item ids — charset + 53-char budget](feedback_batch_custom_id_53_char_budget.md) — custom_ids must match ^[a-zA-Z0-9_-]{1,64}$: dots/colons/slashes 400 the first create (#1776 c9); ids ≤53 + bijective alias/hash-compact + persisted id_map + dispatcher pre-submit validation (dry-run included) + cache quarantine (#1415)
