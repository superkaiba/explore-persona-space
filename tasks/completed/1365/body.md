---
title: 'workflow-fix: correct upload-policy.md #1315 root-cause clause post-#1360'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2d03790dc914
created_at: '2026-07-15T22:20:18Z'
has_clean_result: false
origin_prompt: 'fact-checker prose follow-up on #1360: upload-policy.md L283 paragraph
  carries a root-cause account contradicted by the #1315 v7 marker + code; stale ''ONLY
  retry'' claim after #1360 merged 289ad17572'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1360 (emitting agent: fact-checker prose, routed by the #1360 orchestrator; deferred until #1360's change landed on main per the "actively changing" deferral, now landed at merge 289ad17572925dfa3cdd3ff1f809be9159787aae).

## Goal

Correct `.claude/rules/upload-policy.md`'s rule-(c) layering paragraph (§ "HF Hub rate limit: 256 repository commits per hour"): the demonstrated #1315 no-path case was the un-retried `api.file_exists` verify fallback, not a classifier miss; and the paragraph's "for the Xet queue class it is the ONLY retry" claim is stale now that #1360 wrapped the fallback and added the "queue size reached" transient substring.

## Workflow gap

- **Bug observed:** the paragraph asserts the #1315 no-path return came from "the response-less Xet 'maximum queue size reached' text, which `_is_transient_upload_error` never matches" — contradicted by the primary artifacts: #1315 `epm:failure` v7 quotes `429 Client Error: Too Many Requests` (a matching substring, and response-bearing 429s classify by code), and the verified escape path was the bare `file_exists` verify probe (#1360 plan §3 diagnosis, three independent confirmations: fact-checker, methodology critic, #1345 crash-fix r5). The "ONLY retry" clause is additionally stale post-#1360.
- **Why it is a workflow gap:** a rule file carrying a wrong root-cause account + a stale only-retry claim misleads the next dispatcher author sizing seam retries (they would over-build outer retries believing hub cannot cover the queue class).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "maximum queue size reached" .claude/rules/upload-policy.md` → 2 hits in 1 file (L244 = the accurate #1335 record, keep; L283 = the offending clause in the rule-(c) paragraph, target) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

- so a no-path return means the inner budget
- EXHAUSTED or the failure classed non-transient: the demonstrated #1315 case is
- the response-less Xet "maximum queue size reached" text, which
- `_is_transient_upload_error` never matches (quota-403 and the 0-files-verify
- path land here too). The seam retry is the cheap OUTER envelope — each attempt
- re-enters the full inner envelope after a 30-120 s pause; for the Xet queue
- class it is the ONLY retry.
+ so a no-path return means the inner budget
+ EXHAUSTED or the failure classed non-transient (quota-403 and the
+ 0-files-verify path land here). The demonstrated #1315 case was the then
+ un-retried `api.file_exists` verify fallback inside `list_hf_files_under_path`
+ — fixed fleet-wide by #1360 (merge 289ad17572): the fallback now rides
+ `_retry_upload`, and "queue size reached" classifies transient response-less.
+ The seam retry remains a cheap bounded OUTER envelope (each attempt re-enters
+ the full inner envelope after a 30-120 s pause), no longer the only retry for
+ the Xet queue class.

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'maximum queue size reached' .claude/ CLAUDE.md scripts/`) and update every hit
  whose claim is wrong (L283 paragraph); L244 (#1335 record) is accurate history — keep.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- fingerprint: 2d03790dc914

Surfaced prose (fact-checker, task #1360, 2026-07-15): "`.claude/rules/upload-policy.md` was concurrently updated with a paragraph claiming the #1315 no-path return was 'the response-less Xet maximum queue size reached text, which `_is_transient_upload_error` never matches' — that account is contradicted by the v7 marker text (which contains 'Too Many Requests', a matching substring) and by the code reading; the plan's §3 diagnosis (unwrapped call site) is the one the primary artifacts support."
