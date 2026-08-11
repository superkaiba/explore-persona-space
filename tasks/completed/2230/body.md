---
title: 'gotchas.md: extend the #779 shallow-clone git-log class to in-driver ancestry
  GATES (shallow-aware gate recipe)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-11T03:26:49Z'
has_clean_result: false
origin_prompt: 'auto-filed by the #2222 orchestrator from the experimenter epm:failure-lesson
  (shallow-clone lineage-gate false positive, pod-2222 smoke halt 2026-08-10)'
workflow: v1
---
## Goal

Extend the `.claude/rules/gotchas.md` shallow-clone git-log entry (the #779 class) to cover IN-DRIVER git-ancestry GATES, and state the shallow-aware gate recipe, so future per-issue drivers do not re-discover the class on their first pod launch.

## Problem (workflow-surface gap)

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
candidate-fingerprint: shallow-clone-ancestry-gate-in-driver
confidence: high

#2222's P0 driver (`scripts/issue2222_stage.py` `reassert_parent_lineage`) ran a `git log origin/main..origin/<parent-branch> -- <path>` leg-A re-assertion as an in-driver gate. On the pod bootstrap's depth-1 SHALLOW clone, truncated main ancestry attributes already-MERGED commits as branch-side: the gate flagged ~70 phantom "undispositioned" commits and halted every fresh pod deterministically (flagged SHA `a2b019934e` is an ancestor of `origin/main` on a full clone; the VM full-clone run returns exactly the plan's dispositioned SHAs). The existing gotchas.md #779 entry covers shallow-clone `git log` surprises for fix-commit checks only — the class extends to in-driver lineage GATES, which fail CLOSED and burn a pod launch each time.

## Proposed fix

Extend the #779 gotchas.md entry (or add a sibling bullet next to it) with: (a) the trigger — any in-driver / pod-side `git log A..B` ancestry range or merged-vs-branch-side attribution runs on whatever clone the venue provides, and RunPod bootstrap clones are depth-1; (b) the recipe — detect `git rev-parse --is-shallow-repository`; post-filter flagged SHAs with `git merge-base --is-ancestor <sha> origin/main` (ancestor ⇒ merged, drop); deepen bounded (`git fetch --deepen=200` / `--unshallow --filter=blob:none`) only when merge-base cannot decide; never weaken full-clone behavior; (c) the reference implementation — #2222's fixed `reassert_parent_lineage` (`scripts/issue2222_lib.py` / `scripts/issue2222_stage.py`, branch issue-2222) + its unit test.

## Provenance

Emitted by the #2222 orchestrator from the experimenter's `epm:failure-lesson v1` (gotcha_candidate: yes, generalizes: yes, root_cause_confirmed: yes) after the 2026-08-10 pod-2222 smoke-gate halt. Evidence: #2222 events.jsonl `epm:failure v1` + `/workspace/logs/issue-2222.log` on pod-2222.
