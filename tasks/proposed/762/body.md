---
title: 'workflow-fix: document HF Hub per-org 2500-req-per-5-min rate limit (429)
  in gotchas.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:hf-per-org-2500-req-5min-rate-limit-429
created_at: '2026-06-30T09:07:40Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #658 r4: HF Hub 429 rate-limit on snapshot_download
  (12000 files at default max_workers)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised by the experiment-implementer on task #658 (round-4 fix after a recurring HF Hub 429 rate-limit on `snapshot_download` bursts).

## Goal

Document in `.claude/rules/gotchas.md` that HF Hub enforces a per-org 2500-API-requests-per-5-minutes rate limit (returns 429 BadRequest), and that `snapshot_download` at default parallelism on ≥3000-file downloads will trip it. Recipe: pass `max_workers=4` (or lower) AND wrap in a 429-aware retry with `Retry-After`-bounded backoff.

## Workflow gap

- **Bug observed:** task #658 9a-ter retry attempt 2 — `snapshot_download` of 12000 `rollout_acts/*.pt` files at default `max_workers` ran cleanly for ~38min then crashed with `huggingface_hub.errors.HfHubHTTPError: 429` ("hit the quota of 2500 api requests per 5 minutes period"). Each xet-read-token call counts as one API request, so 12000 files = 12000 requests, far above the quota.
- **Why workflow gap:** `.claude/rules/gotchas.md` documents HF Hub's 504/5xx + 10000-files-per-dir + storage-quota traps (after #658's prior rounds) but says nothing about the per-org 429 quota. Every HF-upload-or-download-heavy workflow at scale will re-introduce this trap.
- **Confidence:** medium-to-high (the implementer hit this LIVE; pattern is general).

## Proposed change

Add a single bullet to `.claude/rules/gotchas.md`'s HF section:

```
- **HF Hub per-org 2500-req-per-5-min rate limit (429).** HF Hub enforces a
  hard 2500-API-requests-per-5-minutes quota at the ORG level (incl. xet-read-
  token, tree listing, file metadata). `snapshot_download` of ≥3000 files at
  default `max_workers` will trip it (each file = 1 xet-read-token call). Fix:
  pass `max_workers=4` (~1200 req/5min, well under the quota) AND wrap the call
  in a 429-aware retry with `Retry-After`-header-bounded backoff (60-300s; HF
  Hub's `paginate` only retries on 429 for some endpoints but NOT for the xet
  bulk-download path — incident #658). Upgrading to Team/Enterprise raises the
  quota.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Verify no existing reference: `grep -rln "2500.*api.*requests\|429.*quota\|rate.limit.*5.min" .claude/ CLAUDE.md scripts/`

## Constraints / invariants

- Workflow-surface only — documentation addition.
- `scripts/workflow_lint.py` passes.
- Recursion guard: this session runs under `EPM_WORKFLOW_FIX_SESSION=1` (and carries a `workflow_fix_target:` Provenance line) — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: hf-per-org-2500-req-5min-rate-limit-429

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: task #658 9a-ter — snapshot_download of 12000 files at default max_workers hit HF Hub's per-org 2500-API-requests-per-5-min rate limit (429); each xet-read-token call counts as one request, so high-parallelism downloads at scale predictably trip the quota.
why_workflow_gap: gotchas.md documents HF Hub's 504 + 10k-files-per-dir + storage-quota traps but not the 2500-req-per-5min org rate limit; every download-heavy workflow re-introduces this trap.
proposed_change: add a gotchas.md bullet naming the 2500-req/5min quota, the snapshot_download max_workers=4 default, and the 429-aware Retry-After-bounded retry recipe.
diff_sketch: |
  + - **HF Hub per-org 2500-req-per-5-min rate limit (429).** HF Hub enforces
  +   a hard 2500-API-requests-per-5-minutes quota at the ORG level. snapshot_download
  +   of >=3000 files at default max_workers will trip it. Fix: max_workers=4
  +   + 429-aware retry with Retry-After-bounded backoff (60-300s). Incident #658.
confidence: medium-to-high
related_task: #658
<!-- /workflow-fix-candidate -->
