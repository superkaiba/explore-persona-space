---
title: 'daily-fix: audit HF call sites route through retry helpers'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-19T07:08:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): 3+ independent same-day
  HF 429-kill incidents at upload/download sites bypassing the retry-hardened hub
  path (#1426 upload_folder_scoped_verify, #1335 ensure_store_local, a crash-report
  download, crash-persist failed_uploads) (c0-P2+c2-P5+c5-P6+c3-P8).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the 2026-07-18 /daily transcript problem sweep (c0-P2 + c2-P5 +
c5-P6 + c3-P8). Route-2 daily-fix filing. Primary work is a library-side HF
call-site audit (`src/explore_persona_space/orchestrate/hub.py` +
`backends/gcp.py`), so `wf_fix: false` — only the `upload-policy.md` guidance
leg touches the workflow surface.

## Goal

Audit EVERY direct `hf_hub_download` / `upload_file` / `upload_folder` call
site to route through the transient-retry helpers, add fleet-shared
commit-budget batching guidance to `upload-policy.md`, and check the
crash-persist upload retry budget in `backends/gcp.py`.

## Problem

- **Bug observed:** at least three independent HF 429-kill incidents landed
  the same day at sites that BYPASSED the retry-hardened hub path:
  - c0-P2: #1426 GCP run killed by an unretried HF 429 in
    `upload_folder_scoped_verify` (in-session fix `98d7218ee1` for THAT site
    only).
  - c2-P5: #1335 `ensure_store_local` bypassed the retry-hardened hub path
    (fixed).
  - c5-P6: a 429 on a crash-report DOWNLOAD.
  - c3-P8: crash-persist wrote `eps/persist=failed_uploads` during the 429
    storm (retry budget in `backends/gcp.py::_eps_persist_diagnostics`).
- **Why it recurs:** the retry helpers (`_retry_upload` /
  `_is_transient_upload_error` in `orchestrate/hub.py`) exist, but individual
  call sites keep being added / reused WITHOUT routing through them, and there
  is no fleet-shared commit-budget batching guidance — so a per-run 429 storm
  starves neighbors.
- **Confidence:** medium
- verified-at-filing: `grep -c 'transient-retry\|_retry_upload' .claude/rules/upload-policy.md` → 3 mentions (the helper is documented) but no all-call-site audit requirement and no fleet-shared commit-budget batching guidance; `git log -1 --format='%cI %h %s' 98d7218ee1` → 2026-07-18T02:28:31 issue #1426 crash-fix retry transient HF 429s in upload_folder_scoped_verify (ONE site only); a repo-wide enumeration finds ~30 files with direct `hf_hub_download`/`upload_folder`/`upload_file` calls, not all routed through the helper (2026-07-19)

## Proposed change (refine in planning)

```
# 1. Enumerate every direct hf_hub_download / upload_file / upload_folder /
#    create_commit / push_to_hub call site under src/ + scripts/; route each
#    through _retry_upload (or the store-fetch retry twin), or add a
#    documented # NO_RETRY: <reason> waiver where a bare probe is intentional.
# 2. upload-policy.md: add fleet-shared commit-budget batching guidance
#    (bulk upload_folder, bounded concurrent commits, AIMD back-off) so one
#    run's upload phase cannot 429-storm the fleet.
# 3. backends/gcp.py::_eps_persist_diagnostics: confirm/raise the crash-persist
#    upload retry budget so eps/persist does not resolve failed_uploads under
#    a transient 429 storm.
```

## Scope / surfaces

- Primary targets: `src/explore_persona_space/orchestrate/hub.py` (call-site
  audit — library code), `src/explore_persona_space/backends/gcp.py`
  (crash-persist retry budget — backends/ IS workflow surface),
  `.claude/rules/upload-policy.md` (batching guidance — workflow surface).
- The dominant deliverable is the src/ call-site audit, so this files as a
  `daily-fix:` (wf_fix false) task; the pipeline planner scopes the workflow-
  surface legs (`backends/gcp.py`, `upload-policy.md`) within the same task.

## Constraints / invariants

- Fail-fast preserved — retry only TRANSIENT errors (429/5xx/timeout/
  connection); a deterministic guard/permission error is never retried
  (existing `_is_transient_upload_error` contract).
- ruff on touched files passes; existing hub/backends tests stay green.

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/hub.py, src/explore_persona_space/backends/gcp.py, .claude/rules/upload-policy.md
- fingerprint: 0d5171b8310a

Surfaced problem (c0-P2 + c2-P5 + c5-P6 + c3-P8): 3+ independent same-day HF
429 kills at upload/download sites bypassing the retry-hardened hub path,
plus a crash-persist upload that failed under the storm.
