---
title: 'daily-fix: crash-fix: shared-module propagation + stale disp'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3d36f961af86
- daily-auto-filed
created_at: '2026-08-02T07:05:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): C2: EBADF race fix in shared
  orchestrate/preflight.py stayed round-local on #1979; #1947 hit the identical crash
  ~6h later (durable commit 22c2ddb2d3 landed after BOTH crashes). C6 (#1902): re-upload
  resume_skip=True nearly retained pre-fix scrambled shards on HF (62 prefixes force
  re-uploaded); stale derived-phase fits sentinels survived a --from-phase relaunch,
  forcing attempt 11.'
workflow: v1
---
# daily-fix: crash-fix: shared-module propagation + stale dispositions

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entries C2 + C6 (miners 6, 8, 5, 2; sessions 75f66748/24f7b592 (#1979), 8fc069db (#1947), 3a60e6ee (#1902)).

## Goal
Extend `.claude/rules/crash-fix-rounds.md` with (1) a shared-module propagation clause — a crash-fix patching SHARED library code must land on `main` (or be explicitly propagated to sibling issues' running trees) in the SAME round; and (2) two named stale-artifact disposition classes — HF-side stale shards a re-UPLOAD would `resume_skip`, and downstream phase SENTINELS derived from invalidated inputs.

## Workflow gap
- **Bug observed:** (C2) 8 concurrent workers racing the fixed probe filename `.preflight_disk_probe.tmp` in `orchestrate/preflight.py::_probe_writable_bytes` crashed #1979 (fellows job 16686, 13:17Z) with EBADF; #1979's round-2 relaunch carried the fix ("Round-2 EBADF fix ENGAGED, 0 EBADF lines", 14:03Z) but the fix stayed round-local — #1947's pod crashed with the identical signature ~6h later (19:06Z); the durable commit `22c2ddb2d3` has committer date 2026-08-02T03:08Z, AFTER both crashes (built-but-stranded-fix family). (C6, #1902) a crash-fix invalidated already-captured stores; the capture_upload re-run resume-SKIPPED the pre-fix scrambled shards already on HF (`resume_skip=True`) and would have silently left corrupted stores for downstream reusers — caught in-session, corrected by force re-upload of all 62 store leaf prefixes with `resume_skip=False`; separately, attempt 11 was forced because stale pre-fix fits sentinels were not wiped by the `--from-phase` relaunch.
- **Why it is a workflow gap:** element 5 ("Stale-run artifact disposition") enumerates only resume-state a relaunched run would LOAD; the upload-direction (`resume_skip` presence checks on HF) and derived-phase sentinels are outside its literal scope, and no clause makes a shared-module fix propagate beyond the crashing issue's branch.
- **Confidence:** high (C2 premise miner-PROBED: `git log --oneline --since=2026-07-30 -- src/explore_persona_space/orchestrate/preflight.py` + `git log -1 --format=%ci 22c2ddb2d3`; C6 mechanisms read from the session's own forensic markers).
- verified-at-filing: `grep -n 'propagat\|sibling' .claude/rules/crash-fix-rounds.md` → 0 propagation hits (the only sibling clauses are the cross-round RENAME grep at :607-622 and per-leg out-roots at :279-291 — no cross-ISSUE shared-module clause); element 5 at :100-115 reads "enumerate the resume-state paths the FAILED run wrote that a relaunched/resumed run would LOAD (... any REMOTE resume prefix — HF `issueN_partial/` ... the driver fetches)" — upload-side `resume_skip` and derived sentinels are not named; `grep -n 'resume_skip' .claude/rules/crash-fix-rounds.md` → 0 hits. `git rev-parse --verify 22c2ddb2d3^{commit}` resolves (fix itself is LANDED — this filing changes only the rule). `git log --oneline --since='7 days ago' -- .claude/rules/crash-fix-rounds.md` → 5 commits, none adds propagation or these disposition classes (2026-08-01).

## Proposed change (refine in planning)
1. New clause after element 4 (fix-commit ancestry): **Shared-module propagation** — when the crash-fix touches shared library code (`src/explore_persona_space/orchestrate/`, `backends/`, `eval/`, `train/`, shared `scripts/` helpers), the SAME round lands it on `main` (worktree merge or scratch-worktree push) or posts an explicit propagation note naming which sibling issues' running trees carry the stale code; a round-local branch fix on a shared module is an INCOMPLETE round (#1979→#1947, 22c2ddb2d3 landed after both crashes).
2. Element 5 gains two named state classes: (a) **HF-side stale artifacts under upload resume-skip** — when the fix invalidates artifacts ALREADY uploaded, the disposition names the affected HF prefixes and mandates `resume_skip=False` force re-upload (or a fresh prefix); a presence-check upload silently retains corrupted artifacts for downstream reusers (#1902: 62 store leaf prefixes force re-uploaded). (b) **Derived phase sentinels** — sentinels/done-markers of DOWNSTREAM phases whose inputs the fix invalidated are wiped/quarantined even on a `--from-phase` relaunch that would not otherwise touch them (#1902 attempt 11).

## Scope / surfaces
- Primary target: `.claude/rules/crash-fix-rounds.md`
- Update the rule's frontmatter description line + `.claude/rules/LESSONS.md` row if the trigger wording changes (lint `--check-lessons-index`).

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Keep element 5's existing quarantine/retain/wipe/fresh-path vocabulary — extend the enumerated state classes, don't fork a parallel disposition scheme.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 3d36f961af86
- workflow_fix_target: .claude/rules/crash-fix-rounds.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entries C2 + C6.
