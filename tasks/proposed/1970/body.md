---
title: 'daily-fix: inline-round pod upload-verify satisfier'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3ece43d43231
- daily-auto-filed
created_at: '2026-08-01T07:07:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Inline rounds have no sanctioned
  upload-verification satisfier for the terminate guard (Step 8 never runs) — forced
  --skip-upload-verify after manual verify; and the per-issue verify script''s prefix
  scope missed a major HF prefix (#1773, 8h50m of GPU output hand-checked).'
workflow: v1
---
# daily-fix: inline-round pod upload-verify satisfier

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M8; miner-8:P15). Source: session 0ac15c23 (#1773 full-dictionary inline rounds) — two halves of one durability seam on inline rounds: (a) `pod.py terminate` REFUSED without an `epm:upload-verification` PASS marker (Step 8 never runs on inline rounds), forcing `--skip-upload-verify` after a manual verify; (b) the per-issue fulldict upload-verify script's exact-set verification covered only `fulldict/` prefixes — the `raw_windows` prefix (8h50m of GPU output) needed a hand-run set-diff, which found 0 missing "but only because someone checked by hand". The #779-class silent-loss risk.

## Goal

Give user-chat inline rounds a sanctioned upload-verification-then-terminate path (documented verify-then-override recipe, or an inline-round verification marker the terminate guard accepts), and add the duty that per-issue upload-verify scripts enumerate ALL HF prefixes the run wrote.

## Workflow gap

- **Bug observed:** #1773's inline round could not terminate its pod through the sanctioned path: the terminate guard's only satisfiers are a Step 8 `epm:upload-verification` PASS marker (never posted on inline rounds) or the blunt `--skip-upload-verify` override; and the round's own upload-verify script silently scoped verification to one artifact prefix, leaving a major prefix verified only by an ad-hoc hand check.
- **Why it is a workflow gap:** (a) `scripts/pod_lifecycle.py::_guard_upload_verification_before_terminate` (lines ~2704-2788) checks exactly two things: `_has_upload_verification_pass(issue)` (latest `epm:upload-verification` event, PASS verdict) or `skip_flag` — no inline-round satisfier exists (the `epm:free-analysis-followup-run` marker appears in pod_lifecycle.py only in the watcher pod-safety predicate at lines 2515/2568, not in the terminate guard). (b) CLAUDE.md's user-chat inline carve-out § completion-side teardown mandates verify-then-terminate but never mentions the guard or the sanctioned override recipe (`--skip-upload-verify` has 0 hits in CLAUDE.md), so every inline round rediscovers the refusal and improvises. (c) No duty line anywhere requires a per-issue upload-verify script to enumerate every HF prefix the run wrote — prefix-scoped verification passing while a sibling prefix goes unverified is exactly the #779 silent-loss shape.
- **Confidence (emitter):** high for (a)/(b) (guard code + CLAUDE.md probed); medium for (c) (teammate-reported hand set-diff)
- verified-at-filing: `grep -n "upload-verification\|skip-upload-verify\|free-analysis-followup-run" scripts/pod_lifecycle.py` → guard at 2704-2788 with satisfier set = {latest PASS marker, `--skip-upload-verify` flag} (presence claim; context read — the inline marker kinds at 2515/2568 belong to the pod-safety predicate, not this guard). `grep -n "skip-upload-verify" CLAUDE.md` → 0 hits (absence claim: the inline carve-out does not document the path). `git log --oneline --since='7 days ago' -- scripts/pod_lifecycle.py` → 0 commits; no landed fix (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/pod_lifecycle.py (_guard_upload_verification_before_terminate):
+ Accept an inline-round verification satisfier: a `epm:upload-verification`
+ marker posted by the inline round itself (source: inline-round, Verdict:
+ **PASS**) — same bold-Verdict anchoring as the Step 8 marker — so an inline
+ round that ran a real verification posts the marker and terminates through
+ the front door; --skip-upload-verify stays the labeled last resort.

CLAUDE.md § user-chat inline free analysis, completion-side teardown clause:
+ one sentence: inline rounds run their verification, post
+ `epm:upload-verification` (Verdict: **PASS**, note naming the verified
+ prefixes), THEN terminate; never bare --skip-upload-verify without a
+ recorded verify.
+ one duty sentence: a per-issue upload-verify script MUST enumerate ALL HF
+ prefixes the run wrote (compare against the run's staging/upload call
+ sites), never only the current phase's prefix.
```

## Scope / surfaces

- Primary target: `CLAUDE.md` (inline carve-out § completion-side teardown) + `scripts/pod_lifecycle.py` (terminate guard)
- Secondary: `.claude/skills/issue/SKILL.md` Step 9a-ter (if the inline-duty blocks live there), `tests/` pin test for the new satisfier (`test_has_upload_verification_pass_*` sibling)
- Grep before editing: `grep -rn 'skip-upload-verify\|_has_upload_verification_pass' scripts/ .claude/ CLAUDE.md` and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The Step 8 guard semantics for pipeline rounds are UNCHANGED — the new satisfier must not weaken the pipeline path (an inline marker must be distinguishable and still anchored on the bold Verdict line, per the existing regex test).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 3ece43d43231

- workflow_fix_target: scripts/pod_lifecycle.py
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED M8 (miner-8:P15), /daily 2026-07-31 — "Inline-round pod release has no sanctioned upload-verification satisfier, and the per-issue upload verify script's prefix scope missed a major artifact prefix" (session 0ac15c23 / #1773).
