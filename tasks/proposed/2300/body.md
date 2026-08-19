---
title: HF data repo at the hard 1,000,000-file cap — every net-new-file upload fails
  fleet-wide
kind: infra
tags: []
created_at: '2026-08-14T16:07:29Z'
has_clean_result: false
parent_id: 2094
origin_prompt: 'workflow-fix-candidate v1 from the #2094 reversed-direction round
  implementer: 19-file upload_folder to superkaiba1/explore-persona-space-data rejected
  with BadRequestError ''would contain 1000018 files, over the limit of 1000000''
  (2026-08-14)'
workflow: v1
---
# HF data repo is at the hard 1,000,000-file cap — every net-new-file upload fails fleet-wide

## Goal

Restore the ability to upload NEW files to the project data repo
`superkaiba1/explore-persona-space-data`, and document the routing so no
future round silently loses artifacts.

## Symptom (observed, not inferred)

On 2026-08-14 a 19-file `upload_folder` commit from the #2094
reversed-direction round (prefix `judge_raw_rev`) failed with:

```
BadRequestError: ... would contain 1000018 files, over the limit of 1000000
```

The repo is AT the Hugging Face Hub's hard per-repo file cap. This is not a
storage-quota 403 (the #552/#541 recovery does not apply) — it is a FILE-COUNT
ceiling, so it is size-independent: a 1 KB JSON fails exactly like a 10 GB
tensor.

## Blast radius

**Every** net-new-file upload to the data repo now fails, for every concurrent
session:

- raw completions (all stages) — the upload-policy REQUIRED persistence path
- training mixes / datasets
- intermediate analysis tensors that plans reference as downstream inputs
- judge raw/score outputs

The upload-policy rule "text / JSON uploads ALWAYS, unconditionally" is
currently UNSATISFIABLE against this repo. Rounds that treat a failed upload as
non-fatal will silently ship without their durable artifacts, and
upload-verification gates that only reconcile the CURRENT phase's prefix may
still PASS while other prefixes were never written.

Known live exposure at filing time: the #2094 butler round (`pod-2094-butler`,
1x H100) is mid-flight and will hit this at its upload phase.

## Proposed fix (from the reporting agent, `workflow-fix-candidate v1`)

1. **Overflow data repo** — adopt a sibling of the #1108 model-overflow
   convention (e.g. `superkaiba1/explore-persona-space-data-2`) with routing in
   `orchestrate/hub.py` mirroring the existing overflow machinery, so a
   cap/limit error transparently reroutes instead of failing the round.
2. **Consolidation pass to reclaim headroom** — pack many-small-file legacy
   prefixes into JSONL line-shards per the #1739 pack recipe. Note the
   >9.5 MB line-split / never-gzip constraint in `.claude/rules/upload-policy.md`
   (a `*.gz` blob is LFS-matched and >10 MB force-routes to LFS).
3. **Document the routing in `.claude/rules/upload-policy.md`** either way, plus
   the hub helpers.

## Acceptance criteria

1. A net-new-file upload to the project's data storage SUCCEEDS end to end from
   a pod and from the VM.
2. The cap/limit error path is handled explicitly and FAIL-LOUD if it cannot
   reroute — never a swallowed exception, never a silent skip (the crash IS the
   signal).
3. `upload_verification` / `verify_uploads.py` resolves artifacts across BOTH
   the primary and any overflow repo, so a rerouted upload does not read as a
   missing artifact.
4. `.claude/rules/upload-policy.md` documents the routing, the cap, and how to
   check current headroom before a large-file-count upload.
5. A cheap pre-flight headroom check exists so a round learns about the ceiling
   BEFORE it spends GPU hours it cannot persist.

## Provenance

Surfaced by the `#2094` reversed-direction round's implementer as a
`workflow-fix-candidate v1` block (confidence: high) after its own 19-file
upload was rejected. That round's judge outputs were persisted to git (1.7 MB
of text) as a fallback, so no data was lost there — but git is not the policy
path for raw completions and does not scale to rollout-tensor volume.
