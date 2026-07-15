---
title: 'workflow-fix: gotchas entry — pod git 403 with valid token, bundle sideload'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8a06535968e5
created_at: '2026-07-15T19:18:01Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1315 r10 (experimenter):
  pod git fetch 403 with valid token; git-bundle sideload recovery'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1315 (emitting agent: experimenter, relaunch r10).

## Goal

Add a gotchas.md entry: a RunPod pod's git fetch to github.com can 403 with a verified-valid token (API probe 200) and a correct credential helper — after one helper-recovery attempt, sideload the commit delta via git bundle (VM create + scp + pod pull --ff-only), then re-verify HEAD + fix ancestry.

## Workflow gap

- **Bug observed:** #1315 r10: pod-1315's git-HTTPS fetch 403'd despite a valid GITHUB_TOKEN and the #1239 env-reading helper; the experimenter unblocked the relaunch by sideloading the 1-commit delta via git bundle; root cause (suspected egress-IP git-http blocking) unconfirmed.
- **Why it is a workflow gap:** gotchas.md has the #1239 credential-helper recovery but no escalation for the valid-token-still-403 case — without the bundle-sideload recipe an agent loops on auth debugging (the r10 experimenter nearly burned its launch window on it).
- **Confidence (emitter):** medium (recovery proven in the field; root cause unconfirmed — the entry should say "suspected egress-IP blocking, unconfirmed")
- verified-at-filing: `grep -ciE "git bundle|egress.*403|403.*egress" .claude/rules/gotchas.md` → 0 hits (absence-of-entry claim — the 0-hit in-target result IS the evidence) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In .claude/rules/gotchas.md (near the #1239 credential-helper entry):
+ ## Pod git fetch 403 with a VALID token (#1315 r10)
+ A RunPod pod's git-HTTPS fetch can 403 even when the token is verified
+ valid (API probe 200) and the #1239 env-reading credential helper is the
+ sole helper — suspected egress-IP git-http blocking (unconfirmed). After
+ ONE helper-recovery attempt, stop debugging auth and SIDELOAD: on the VM
+ `git bundle create /tmp/issue-<N>.bundle <podHEAD>..origin/issue-<N>`,
+ scp to the pod, pod-side `git -C /workspace/... pull --ff-only <bundle>`,
+ then re-verify HEAD + fix-sha ancestry before launching.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing (`grep -rln '1239\|credential' .claude/rules/ .claude/agents/`); keep adjacent to the #1239 recovery; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 8a06535968e5

<!-- epm:failure-lesson v1 -->
failure_class: infra
phase: pod branch sync (pre-launch)
lesson: A RunPod pod's `git fetch` to github.com can 403 even with a verified-valid token (API probe 200) and a correct env-reading credential helper as the sole helper — likely egress-IP git-http blocking. After one helper-recovery attempt, stop debugging auth: sideload the commit delta via `git bundle create <podHEAD>..origin/issue-<N>` on the VM + scp + pod-side `git -C ... pull --ff-only <bundle>`, then re-verify HEAD + fix ancestry.
generalizes: yes
owning_agent: experimenter
gotcha_candidate: yes
root_cause_confirmed: no
<!-- /epm:failure-lesson -->
