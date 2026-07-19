---
title: 'workflow-fix: gotchas.md bullet — off-pod phase reads vs upload manifest (cross-machine
  seam)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9e5f06243a16
created_at: '2026-07-19T01:36:24Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1482 (pod_scratch_metadata_not_uploaded_cross_machine_seam)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson (`gotcha_candidate: yes`) raised on task #1482 (emitting agent: experiment-implementer).

## Goal

Add a `.claude/rules/gotchas.md` bullet: every file an OFF-POD phase loads must be in the pod's upload set — an all-on-one-filesystem smoke is structurally blind to the cross-machine seam; deterministic scratch metadata gets committed sha anchors + a reconstruction path.

## Workflow gap

- **Bug observed:** #1482's off-pod P5 judge died at VM launch loading pod-only `scratch/{split_indices.npz,row_ci.npy,prov.npy}` (~17 MB, never in the P4 upload set) after the pod was terminated; recovery required a dedicated sha-anchored reconstruction round (~4.6 min run + implementer round). The all-on-VM smoke shared one filesystem tree, so no smoke could catch it.
- **Why it is a workflow gap:** gotchas.md is the codebase-trap catalog loaded when writing orchestration/eval code; it has no coverage of the off-pod-phase/upload-manifest seam, so the next multi-machine pipeline re-derives this from a stranded run.
- **Confidence (emitter):** high (root_cause_confirmed: yes; durable fix landed at 72c2dc8906)
- verified-at-filing: `grep -in 'off-pod|upload set|upload manifest' .claude/rules/gotchas.md` → 3 incidental hits, each read in context (the vLLM-reaping bullet's "off-pod" token etc.) — none implements the proposed coverage (presence-hits context-checked per clause (c)); `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` run at filing for the landed-fix backstop (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

+ In .claude/rules/gotchas.md, new bullet (orchestration/pipeline traps section):
+ **Off-pod phase reads vs the upload manifest (cross-machine seam).** Any file an OFF-POD phase
+ (judge, analysis) loads must be in the pod's UPLOAD SET; an all-on-one-filesystem smoke is
+ structurally blind to this seam (pod scratch + VM phases share one tree in the smoke). At design
+ time, enumerate each off-pod phase's file reads line-by-line against the upload manifest; upload
+ small scratch metadata unconditionally; commit sha anchors (e.g. a split_<N>.json with index-set
+ sha256s) so deterministic scratch stays reconstructable; make off-pod loaders fail loud with the
+ recovery recipe (#1482: P5 died on pod-only scratch after termination; sha-anchored reconstruction
+ recovered it — scripts/issue1482_reconstruct_scratch.py is the worked example).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Consider one cross-pointer from `.claude/rules/pod-side-reporting.md` or plan-compute-sizing if the planner surface names upload manifests; grep first.

## Constraints / invariants

- Workflow-surface only; gotchas row caps respected (ratchet budget if over headroom).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 9e5f06243a16

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p4-upload-set / p5-launch seam (issue1482_error_analysis.py -> issue1482_analysis.py)
lesson: Cross-machine phase seams strand runs: any file an OFF-POD phase loads must be in the pod's upload set, and an all-on-one-filesystem smoke is structurally blind to the gap. Enumerate each off-pod phase's file reads against the upload manifest at design time; when scratch metadata is deterministic, a sha-anchored reconstruction script (anchors committed with the run, e.g. split_1482.json) makes the loss fully recoverable.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
