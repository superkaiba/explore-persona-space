---
title: 'workflow-fix: gotchas fanout-staging bullet — add the per-unit shared-cache
  EVICT race mode'
kind: infra
tags:
- wf-fix
- wf-fix-fp:28d358ef8e44
created_at: '2026-07-24T02:34:17Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #1586 fu crash-fix round 6 (epm:failure-lesson
  v6): per-restage shared-hub-cache evict under fanout deleted siblings'' in-flight
  blobs; gotchas L123 lacks the evict race mode; fix 876f65ce'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #1586 (emitting agent: experiment-implementer, crash-fix round 6).

## Goal

Extend the gotchas.md fanout shared-staging bullet (L123, the #1315 entry) with race mode (c): a per-unit shared-hub-cache EVICT deletes siblings' in-flight .incomplete blobs — evict once per parent pre-stage batch, never per unit.

## Workflow gap

- **Bug observed:** #1586 fu p6_panel: two panel units concurrently restaged reaped checkpoints; the first finisher's per-restage overflow hub-cache evict deleted the sibling's in-flight .incomplete blobs (FileNotFoundError in huggingface_hub file_download).
- **Why it is a workflow gap:** gotchas.md L123 enumerates the class's race modes as (a) shared-scratch os.replace steal and (b) proxy-file staleness — a DESTRUCTIVE cache-evict-under-fanout mode is absent, so an implementer applying the bullet's checklist can still ship a per-unit evict (exactly what #1586 r5 did, reviewed PASS, with the bullet on the books).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i -E 'fanout.{0,50}(stag|hub.?cache|evict)|shared.{0,20}staging|prestage|pre-stage' .claude/rules/gotchas.md` → 1 hit (L123 — the entry EXISTS; per-target presence confirmed) + context READ: L123's enumerated modes are (a) os.replace steal and (b) proxy-file staleness — NO evict mode (the proposed change is additive, not landed) + `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 1d0a972466 task #1640: document the chained smoke-then-full per-leg out-root residue trap (#1414);ccd5155108 task #1611: smoke/production-parity REGIME/CLASS COVERAGE family member (#1390);be49c179b3 gotchas: np.savez appends .npz — atomic tmp-rename checkpoints must keep the .npz suffix (daily 2026-07-22 route-1);— none touch the fanout-staging entry (2026-07-24)

## Proposed change (candidate diff sketch — refine in planning)

+ In the L123 bullet's race-mode enumeration, add: "(c) a per-unit EVICT of a shared hub-cache entry (a 'free space after my restage' pattern) rmtree's siblings' in-flight .incomplete blobs + tmp_* dirs — FileNotFoundError inside hf file_download at the copy/rename; evict ONCE per parent pre-stage batch (only when ≥1 restage happened), NEVER inside a unit; hf ≥0.36 removes-or-resumes stale etag-keyed .incomplete files, so no separate residue sweep is needed (incident #1586 fu r5→r6, fix 876f65ce)."
+ Update the bullet's RULES (i) line to name the evict placement alongside parent pre-staging.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'pre-stage\|prestage' .claude/ CLAUDE.md scripts/`) and update every hit that enumerates this class's race modes; the implementer-scoped long-form twin `.claude/agent-memory/experiment-implementer/feedback_fanout_shared_staging_race.md` likely gains the same (c) mode.

## Constraints / invariants

- Workflow-surface only; `workflow_lint.py --check-asks` + `--check-lessons-index` pass; gotchas row-size ratchet respected (budget a cap-raise if over headroom).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 28d358ef8e44

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p6_panel (issue1586_dispatch.py fan-out)
lesson: An in-unit restage-on-missing branch with a PER-RESTAGE shared-hub-cache evict is serially safe but fanout-fatal — the first finisher's evict deletes siblings' in-flight .incomplete blobs (FileNotFoundError in hf file_download). Pre-stage every fan-out arm's missing checkpoint SERIALLY in the parent at phase entry and evict the shared hub-cache entry ONCE per batch; keep the in-unit restage as a fail-loud, evict-free backstop (hf 0.36.2 removes-or-resumes stale etag-keyed .incomplete files, so no separate residue sweep is needed).
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
supersedes: feedback_midrun_verified_upload_ckpt_reap (r5 per-restage-evict clause; memory file updated in place this round)
<!-- /epm:failure-lesson -->
