---
name: Batch-API judge item ids — 53-char budget, hash-compact + id_map
description: Anthropic Batch custom_ids cap at 64 chars and eval/batch_judge's encoder appends 11 ("__NNNNN__NN"), so hierarchical cell-id item ids overflow past 53 chars; build h<sha1-12>/d<draw> ids + a persisted compact->full id_map + a len<=53 build-time assert (#1415)
type: feedback
---

Judge item ids passed to `eval/batch_judge` become Anthropic Batch
`custom_id`s via an encoder that appends 11 chars (`__NNNNN__NN`), and the
Batch API caps `custom_id` at 64 chars — so the CALLER's item id budget is
53 chars. Item ids built from hierarchical cell ids
(`gen1c/context/<pair_id>/L20/a0.5/d0`) overflow as soon as any pair/cell id
component grows (#1415: cross-pair cells hit 64 → 67-char custom_id →
fail-fast crash at enumerate, zero API calls).

**How to apply:** (1) Build judge item ids from a hash-compact key —
`h<sha1(cell_id)[:12]>/d<draw>` — never the raw hierarchical id; assert
`len(item_id) <= 53` at build time and fail loud on sha collisions.
(2) Persist a compact→full `id_map.json` (atomic write, BEFORE any judge
call) and rehydrate at result-reduction so downstream outputs stay keyed by
the full readable ids. (3) The rubric-keyed JudgeCache is unaffected (keys
on rubric/question/completion, never item ids). (4) On a mid-judge crash
with a partially-populated cache: QUARANTINE the work-dir before relaunch —
`judge_graded` packs n_draws identical completions per item and the cache
key is identical across draws, so a cache-hit replay collapses that rubric
to effective n_draws=1 (the rule-24 duplicated-draw trap). Worked impl:
`scripts/issue1415_judge.py` @ f779a83aba.
