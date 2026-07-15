---
title: 'workflow-fix: upload-policy — bounded retry before fail-loud on no-path uploads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:17023149c3b7
created_at: '2026-07-15T19:00:38Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1315 r8 (experiment-implementer):
  fail-fast dispatcher seams need a bounded transport retry on hub upload no-path
  returns'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1315 (emitting agent: experiment-implementer, crash-fix round 8).

## Goal

Extend upload-policy.md's 429/no-path material: a dispatcher seam that FAIL-FASTS on hub._upload's no-path return must wrap it in a bounded jittered-backoff retry (transport-class only, same fail-loud error on exhaustion) — the complement of the existing #488 rule (which bans the opposite failure mode, silent warning-and-continue).

## Workflow gap

- **Bug observed:** #1315 r8: two p11_upload kills ~35 min apart — hub._upload logged 'Upload failed: 429' / Xet 'maximum queue size reached', returned no path, and the dispatcher's fail-fast _up converted the retriable rate limit into a run-killing final-phase crash twice.
- **Why it is a workflow gap:** upload-policy.md's existing "upload returned no path" text (the #488 entry, lines ~243-252) covers ONLY the silent-loss direction (no-path must be a TRACKED GAP, never warning-and-continue) and the sweep bulk-commit rule; it is silent on the fail-fast direction — a seam that correctly refuses to continue still needs a bounded transport retry BEFORE its fail-loud raise, or sustained fleet HF traffic kills runs at their final phase.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix + retry tests landed on issue-1315 @ c3c600541f)
- verified-at-filing: `grep -ciE "upload.wedge|429" .claude/rules/upload-policy.md` → 5 hits (presence of the 429/wedge section confirmed); `grep -niE "no.path|returned no path" .claude/rules/upload-policy.md` → 2 hits at lines 245/250, BOTH the #488 silent-loss rule (inspected: they mandate tracked-gap-never-continue and say nothing about bounded retry at a fail-fast seam) — the absence-of-retry-guidance claim binds (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In .claude/rules/upload-policy.md, adjacent to the #488 "upload returned no path" rule:
+ (c) the FAIL-FAST direction needs a bounded retry: a dispatcher seam that
+ raises on the no-path return (correct — never warning-and-continue) must
+ first retry the transport class (429 / 5xx / Xet queue-full surface as a
+ no-path return) with jittered exponential backoff (~3 attempts,
+ 30/60/120s), then raise the SAME fail-loud error on exhaustion. Uploads
+ are idempotent (already-landed files skip-and-verify), so retries are
+ free; without them sustained fleet HF traffic kills runs at their final
+ upload phase (#1315: two p11 kills ~35 min apart). Worked example:
+ _upload_with_transport_retry() in scripts/issue1315_dispatch.py.

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Grep before editing (`grep -rln 'returned no path' .claude/ CLAUDE.md scripts/`); keep consistent with the #488 entry and the upload-wedge ladder; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- fingerprint: 17023149c3b7

<!-- epm:failure-lesson v1 -->
failure_class: infra
phase: p11_upload (scripts/issue1315_dispatch.py)
lesson: `orchestrate.hub._upload` swallows HF-429/Xet-queue transport failures and returns "" — a dispatcher seam that fail-fasts on the no-path return converts a retriable rate limit into a run-killing crash at the FINAL phase. Wrap the no-path return in a bounded backoff retry at the dispatcher seam (raise the same fail-loud error on exhaustion); uploads are idempotent so retries are free.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
