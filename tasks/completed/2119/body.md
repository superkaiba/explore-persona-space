---
title: 'daily-fix: manifest-first staging for sharded artifacts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2540f0131f02
- daily-auto-filed
created_at: '2026-08-06T07:03:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): two consumers broke at
  the 9.5MB shard seam same day; an upload-verification PASS had a hole exactly there'
workflow: v1
---
# daily-fix: upload-policy — consumers of shardable text/JSONL artifacts must stage manifest-first (two same-day breakages at the 9.5 MB seam)

## Workflow gap

The upload policy's >9.5 MB line-split rule (text shards `part00/part01…` instead of one
blob) has no consumer-side counterpart, so two different #2054 consumers broke the first
time an artifact crossed the threshold — and an upload-verification PASS had a hole at
exactly that seam:

1. 2026-08-05T08:22Z: r5's larger pools crossed 9.5 MB for the first time; the judge
   stage's `_load_prejudge` expected unsharded names and crashed (fixed in-session with
   shard reassembly).
2. 2026-08-06T00:43Z (`epm:failure v4`): `build_answers` staged
   `shared_question_draw.jsonl` by its UNSHARDED HF name while the top-up had uploaded
   sharded — the r10 seal failed loud ("10,729 required mt_* conv_ids absent"). The
   session's own postmortem: "The v7 upload-verification PASS had a hole in exactly the
   place that broke" (the draw leg was not content-verified). r15 landed a manifest-first
   stager (`_stage_draw_jsonl`).

verified-at-filing: both incidents are probed marker/traceback readbacks from session
7a1632b8 (rows 3018/3130, 4802/4836); `grep -n '9.5 MB\|line-split\|shard' .claude/rules/upload-policy.md | head`
run at compose time — the producer-side rule exists, no consumer-side staging clause.

## Proposed change

Add a consumer-side clause to `.claude/rules/upload-policy.md` (the text-shard section):
any code that stages/downloads a shardable text/JSONL artifact by name MUST resolve it
manifest-first (list the prefix, accept both the unsharded name and the `part*` set — the
r15 `_stage_draw_jsonl` pattern is the reference), and upload-verification of such
artifacts verifies CONTENT reachability under both forms, not name presence. Planner
scopes whether a shared helper in `src/explore_persona_space/` should own the resolve so
per-issue scripts stop hand-rolling it.

## Provenance

- fingerprint: 2540f0131f02

- workflow_fix_target: .claude/rules/upload-policy.md
- origin: /daily 2026-08-05 problem sweep — miner 3 P7.
