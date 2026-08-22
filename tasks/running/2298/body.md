---
title: 'workflow-fix: critic-lens-reference item 13 still says fellows-first auto
  default (stale since #2054/#2059)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T15:50:04Z'
has_clean_result: false
origin_prompt: 'efficiency-critic prose follow-up during #2162 turn-boundary-multipatch
  post-approval panel: critic-lens-reference.md item 13 (line ~410) describes the
  auto default as fellows-first, pre-#2054 drift (runpod-first is current); will misdirect
  a future auto-lane plan review'
workflow: v1
---
# workflow-fix: `critic-lens-reference.md` item 13 still says "fellows-first" auto default (pre-#2054/#2059 drift)

## Goal

Correct the stale auto-lane-order prose in `.claude/rules/critic-lens-reference.md` item 13 so a plan critic costs §9 compute rows on the machine the router will ACTUALLY provision, and add the mechanical check that keeps the two surfaces from drifting again.

## The drift

`.claude/rules/critic-lens-reference.md:410` (Methodology lens item 13, "Compute projection costed on the routed machine"):

> the plan's §9 compute table must cost each row's `planned_wall_h` / `basis` on the machine the backend router will ACTUALLY provision — under the standing **fellows-first** `auto` default (#2028: GCP provisioning disabled) that is the fellows H200 cluster, then the free SLURM lanes, with RunPod's H100 intent table as the terminal rung

That describes the PRE-#2054 ordering. The current canonical order is RunPod FIRST:

- `src/explore_persona_space/backends/router.py:895` — `DEFAULT_AUTO_LANE_ORDER` (built by `_default_auto_lane_order()`); the module docstring at `router.py:45` documents the current order, and `router.py:899` notes "``runpod`` is a LEGAL lane as of" the inversion.
- `CLAUDE.md` § Compute backends and `.claude/rules/compute-backends.md`: `DEFAULT_AUTO_LANE_ORDER = ("runpod", "fellows", "nibi", "fir", "mila")` — RunPod first (`reason: auto_runpod_first`), then fellows, then the free DRAC/Mila SLURM lanes, with a terminal RunPod RETRY rung (`reason: auto_fallback_runpod`).
- Landed by #2054 (standing user directive: RunPod is the first-resort lane, the shared Anthropic fellows/safety org pool) and #2059 (`Invert auto lane order: RunPod primary (GPU+CPU), fellows next`). Both are `completed`; neither updated this lens-reference prose.

`grep -n "runpod-first\|runpod, fellows\|DEFAULT_AUTO_LANE_ORDER" .claude/rules/critic-lens-reference.md` returns nothing — the file has no correct statement of the order anywhere, so this is a plain stale-prose gap, not an inconsistency between two passages.

## Why it matters (the failure it causes)

Item 13 is the lens that makes a critic REVISE a §9 compute table costed on the wrong GPU. As written it instructs the critic to expect fellows H200 as the provisioned machine under a bare `auto` route. A plan whose §9 rows are correctly costed on RunPod H100 — the lane that will actually win — reads as MIS-costed against this lens, and a plan wrongly costed on H200 reads as correct. It inverts the check for exactly the plans it governs (any task with absent/empty `backend:` frontmatter, i.e. the default).

Moot for the task that surfaced it (#2162 pins `backend: runpod` and its v7 §9 basis is H100-measured, so the efficiency lens PASSed on recomputed arithmetic), which is why this is filed rather than fixed inline — the next bare-`auto` plan review is where it bites.

## Fix

1. Rewrite item 13's routed-machine clause to state the current order (RunPod H100 intent table first, then fellows H200, then the free DRAC/Mila SLURM lanes, then the terminal RunPod retry rung), keeping the existing #2028 GCP-disabled parenthetical and the `INTENT_TO_MACHINE`-applies-only-under-rollback caveat.
2. Audit the SAME drift across the other always-on and lens surfaces in one pass — at minimum `.claude/rules/critic-lens-reference.md`, `.claude/rules/clean-result-critic-lens-reference.md`, `.claude/agents/critic.md`, `.claude/agents/planner.md`, `.claude/agents/efficiency-critic.md`, `.claude/agents/methodology-baselines-critic.md`, `.claude/rules/plan-compute-sizing.md`, `.claude/skills/adversarial-planner*/SKILL.md` — since #2054/#2059 updated the backend rules but a lens-side sweep evidently did not run. Report which files were clean.
3. Add the mechanical guard so this cannot drift silently again: a `scripts/workflow_lint.py` check that FAILs when a workflow-surface file asserts a lane-ordering adjective (`fellows-first`, `gcp-first`, `fellows first`) that contradicts the live `DEFAULT_AUTO_LANE_ORDER` head from `backends/router.py`. Read the constant, do not hardcode the expected head — the point is that the next inversion updates one place and the lint catches the stragglers. Bundle it into the no-flags default run per the existing convention, and pin it with a test carrying a fixture of the stale line.

## Provenance

Surfaced as a prose follow-up by the `efficiency-critic` (PLAN MODE, verdict PASS) during the #2162 `turn-boundary-multipatch` follow-up round's post-approval critic panel, 2026-08-14. The critic flagged it as "workflow-surface note for the orchestrator (prose follow-up, not mine to file)" — correct: subagents never file, the orchestrator routes (CLAUDE.md § Workflow-fix-on-bug protocol, surfaced-prose clause). Claim verified independently by the orchestrator before filing: `grep -n "fellows-first"` returns exactly `critic-lens-reference.md:410`, and the file contains no correct statement of the current order.

Dedup: no existing task covers this. Registry title scan for `lens-reference` / `lane order` / `fellows-first` / `runpod-first` / `auto lane` returned four hits, all `completed` and all different work (#1302 mbc capsule pin, #1777 `/workspace` sentinel WARN, #2059 the inversion itself, #783 GCP→RunPod failover).

`workflow_fix_target: .claude/rules/critic-lens-reference.md`
