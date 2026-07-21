---
title: 'daily-fix: hub upload file-count WARN + retry docs'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-20T06:49:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): 31k-small-file upload_folder
  crawled (killed+repacked); 2 retry-wrapper TypeErrors'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Improve hub upload ergonomics in `src/explore_persona_space/orchestrate/hub.py`: (a) a throughput pre-check WARN when one `upload_folder` commit would stage more than ~2k files (prompting the pack/shard recipe), extending the existing #1108 file-count-limit machinery; (b) clarify the keyword-only signatures of `retry_transient`/`_retry_upload`-wrapped helpers in docstrings (two TypeError stumbles in one session).

## Workflow gap

- **Bug observed:** an `upload_folder` of 31k small judge files (135 MB) crawled and had to be killed (exit 144) and repacked into <=9 MB shards; the same session hit `TypeError: _retry_upload() missing 1 required keyword-only argument: 'what'` and a second signature stumble on `verify_repo_paths_uploaded` (resolved via inspect.signature).
- **Why it is a workflow gap:** upload-policy prescribes pack/shard for many-small-file sets but nothing warns at the call site before the crawl starts; the retry-wrapper signatures are keyword-only and undocumented at the helper entry points.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'upload_folder|_is_file_count_limit_error|Count the files' src/explore_persona_space/orchestrate/hub.py` → the #1108 per-commit file-COUNT machinery exists (:706-:776) but serves the 100k repo-cap fallback, not a throughput WARN (context read binds); incident: session 98ff0f37 (task #1481) @ 07:15-07:37 UTC 2026-07-19 (kill 2632931/2632911, exit 144, repack to judge_packed shards; TypeError evidence at 07:15 and 07:37).

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: in the public upload helper, if the staged-file count exceeds a threshold (~2k), emit a loud WARN naming the pack/shard recipe before proceeding; add Args/keyword docs to retry_transient-wrapped helpers)

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py` (NOT workflow surface — experiment/library code; this task carries `daily-auto-filed` only, wf_fix false)

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `98ff0f37` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

(wf_fix: false — library code; no workflow_fix_target/fingerprint lines by design)

Mined evidence: session 98ff0f37 (task #1481), 2026-07-19: 31k-file upload crawl + kill + shard repack (~20 min lost); two retry-wrapper TypeErrors.
