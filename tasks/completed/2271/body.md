---
title: 'upload_raw_completions_to_data_repo silently drops pre-merge shard payloads
  (15.6 MB of rollout text on #2223)'
kind: infra
tags: []
created_at: '2026-08-13T07:50:40Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'Surfaced during /issue 2223 Step 8: verify_uploads.py --outroot-listing
  flagged 4 shard_<i>of4.json files with no permanent home after phase_upload logged
  success; the 32B leg is shard-structured and exposed to the same drop.'
workflow: v1
---
# `upload_raw_completions_to_data_repo` silently drops pre-merge shard payloads

## Goal

Make the canonical raw-completions upload helper persist EVERY rollout-text
file under the tree it walks — pre-merge per-shard payloads included — so a
shard-structured generation run cannot reach a clean `[phase=upload]` line with
15+ MB of its own rollout text left on a pod volume that dies with the pod.

## The bug

`explore_persona_space.orchestrate.hub.upload_raw_completions_to_data_repo`
landed only the merged `raw_completions.json` products for #2223's 7B leg. Four
sibling files were skipped entirely:

    eval_results/issue_2223/raw_completions/phaseA/A0__7b/shards/
      shard_0of4.json   4,074,740 bytes
      shard_1of4.json   3,792,573 bytes
      shard_2of4.json   3,857,882 bytes
      shard_3of4.json   3,873,421 bytes

15.6 MB of rollout text — roughly the merged product's OWN size, so the merge
is a concatenation and these are substantive, not a redundant cache. Rollout
text is the unconditional-upload class under the Upload Policy; no discard
declaration covers it, and a generation stage that drops its generations is an
upload-verification FAIL whether declared or not (#779).

## Why it evaded every non-instrument check

The upload phase logged its normal success line
(`raw_completions + analysis_tensors persisted to HF`) and exited 0 — the
helper's walk simply never visited the subdir, so there was no error to raise.
A by-eye reconciliation ALSO missed it: `raw_completions/` shows 6 local files
and the HF prefix showed the merged products present, which reads as covered
until you diff the actual name sets. Only
`verify_uploads.py --outroot-listing` caught it, by exactly the name-set diff
its own detail string insists on: "a matching count is not a matching set —
the verdict is the name-set diff, never the counts" (#2162).

## Blast radius — this is fleet-wide, not one issue

Any run that shards generation and merges afterwards hits it. Two known live
exposures at filing time:

- **#2223's 32B leg** (`pod-2223-q32b`) is shard-structured BY DESIGN — 4
  shards x 15 turns, ~7h of 4x H200 generation. Its upload phase will drop the
  same payload class unless the shard dirs are uploaded explicitly. Recorded on
  #2223's relaunch checklist as a manual step; this task is the real fix.
- Any sibling issue reusing the same helper with a `shards/` layout.

Because the helper is shared, fixing it here fixes every consumer at once —
which is why this is filed against the helper rather than patched into one
driver.

## Acceptance

1. The helper persists every regular file under the walked tree, or fails loud
   naming what it skipped — never a silent partial upload behind a success log.
2. A regression test builds a raw_completions tree WITH a nested `shards/`
   subdir and asserts every file reaches the destination prefix (the shape that
   actually regressed — a flat-tree-only fixture would pass today).
3. Verify against a real prefix that a shard-structured run leaves
   `verify_uploads.py --outroot-listing` at `matched == disk`.

## Provenance

Surfaced during #2223 Step 8 upload verification (2026-08-13), where the four
shard files were the only residue the instrument flagged that manual
reconciliation had also missed. Persisted out-of-band for #2223; the helper
itself is unfixed.
