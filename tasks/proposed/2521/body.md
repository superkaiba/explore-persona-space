---
title: verify_uploads.py outroot-residue false-FAILs a sanctioned line-split (sharded)
  text artifact
kind: infra
tags: []
created_at: '2026-08-24T06:41:18Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 823 Step 8: pair_store/pairs_index_ext_rows.jsonl
  flagged residue while its manifest + all 3 shards were on HF and passed --expected-rows
  reconciliation in the same run.'
workflow: v1
---
---
kind: infra
---

# verify_uploads.py residue check false-FAILs a sanctioned line-SPLIT text artifact

## Provenance

workflow_fix_target: scripts/verify_uploads.py

Surfaced by the #823 ext-ladder production round (label `origin-ladder-more-contexts`,
session cmt6ubmuylw53yl0u7juv06tm, 2026-08-24) at the Step 8 out-root residue check.

## The bug

`check_outroot_residue` / `_match_outroot_files` decide coverage by BASENAME
name-set diff against the HF prefix listing. The upload policy REQUIRES text over
9.5 MB to be line-split on upload — `.claude/rules/upload-policy.md`: "Text >9.5 MB
per file line-splits into <9 MB shards, NEVER gzip" — landing on the Hub as
`<stem>.manifest.json` + `<stem>.shardNN.jsonl` while the producing pod keeps the
unsharded `<stem>.jsonl` in its out-root.

Those basenames never match, so a CORRECTLY-uploaded sharded artifact is reported
as residue with no permanent home. Realized effect this round:
`pair_store/pairs_index_ext_rows.jsonl` (83,313 rows) was flagged residue while
`pairs_index_ext_rows.manifest.json` + `pairs_index_ext_rows.shard{00,01,02}.jsonl`
were all present under
`issue823_inconsistent_origin_ladder/analysis_tensors/ext` — and the SAME shard set
then passed the `--expected-rows` reconciliation in the same invocation
(distinct=83313, duplicates=0), so one run of the tool simultaneously consumed the
artifact as present and reported it as missing.

This bites hardest on exactly the artifact class the residue+rows checks care most
about: a store's per-row index is both large (so it shards) and load-bearing for
the #2148 reconciliation.

## Suggested fix

Before declaring a disk file residue, apply the project's own shard-name resolution
rather than a bare basename compare: for a disk basename `<stem>.<ext>`, treat it as
covered when the HF-covered set contains `<stem>.manifest.json` AND the full part
set the manifest declares. `orchestrate.hub.resolve_sharded_text_paths` already
implements manifest-first name resolution (the #2119 consumer clause) — reuse it so
one definition of "the consumable form" serves consumers and the verifier alike.
Keep it fail-toward-FAIL: a manifest present with a MISSING part stays residue
(that is a real gap), and no manifest means no special case.

## Acceptance

- A local `<stem>.jsonl` whose manifest + full part set are on the Hub resolves
  covered, with no `--outroot-exempt`.
- A manifest present with a missing shard still FAILs, naming the missing part.
- Non-sharded artifacts behave exactly as today.
- Regression tests for all three shapes.
