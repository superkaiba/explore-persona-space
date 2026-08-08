---
title: 'workflow-fix: caution against habitual --backend gcp (bypasses free fellows
  lane)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:41e0830591be
created_at: '2026-08-02T14:20:05Z'
has_clean_result: false
origin_prompt: 'orchestrator observation #1739 2026-08-02: ~12 dispatches passed --backend
  gcp explicitly, bypassing the free fellows lane; CLAUDE.md guards only --backend
  runpod out of habit, no gcp caution, no lint'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap the orchestrator hit directly on task
#1739 (2026-08-02): ~12 consecutive dispatches passed `--backend gcp` explicitly, silently bypassing
the free `fellows` (charmander H200) lane that the auto chain tries FIRST.

## Goal

Add a "do NOT pass `--backend gcp` out of habit" caution (the free-lane sibling of the existing
runpod caution) and, if the spawned session's planner agrees it is mechanizable, a lint/WARN that
flags an explicit non-auto `--backend` flag with no stated residual-gap reason.

## Workflow gap

- **Bug observed:** Every #1739 dispatch this session (gap-fill boxes, 7 partial-item boxes, the Gap 2
  OOM relaunch, 3 ladder shards) passed `--backend gcp` explicitly. That flag bypasses the auto chain,
  so the FREE fellows lane was never attempted on any of them, and ~40+ GPU-h of GCP credits were spent
  while charmander sat with free capacity (`sinfo` at 2026-08-02: general 14 nodes mixed, overflow 16
  nodes mixed, dev 2 nodes mixed). The flag was originally added to dodge FLEX_START preemption, but
  the correct remedy for that (`--provisioning-model STANDARD`) was ALREADY being passed alongside it —
  so the `--backend gcp` pin was pure habit with no residual-gap justification.
- **Why it is a workflow gap:** CLAUDE.md § "Compute backends" states the fellows-first default only
  for the ABSENT/EMPTY-frontmatter (`auto`) case, and carries an explicit habit-guard for exactly ONE
  lane: "**Do NOT pass `--backend runpod` out of habit**" (line 279). There is no equivalent guidance
  for `--backend gcp`, and no mechanical check anywhere. The runpod guard exists because runpod costs
  money; the same reasoning applies to gcp (credits) vs fellows (free), but was never written down.
  Nothing in the surface tells an orchestrator that an explicit `--backend gcp` silently forfeits the
  free lane, and nothing warns at dispatch time.
- **Confidence (emitter):** high (directly observed; grep-verified below)
- verified-at-filing: `grep -n "out of habit" CLAUDE.md` -> 1 hit, line 279, runpod-only;
  `grep -noE '[^.]*--backend gcp[^.]*\.' CLAUDE.md` -> 0 hits (no gcp caution exists);
  `grep -rn "check-.*backend\|backend.*habit" scripts/workflow_lint.py` -> 0 hits (no lint exists);
  `grep -n "fellows FIRST" CLAUDE.md` -> lines 274, 278 (fellows-first documented for `auto` only).
  (2026-08-02)

## Proposed change (candidate diff sketch — refine in planning)

In CLAUDE.md § "Compute backends — multi-lane router", alongside the existing runpod habit-guard:

+ - **Do NOT pass `--backend gcp` out of habit either.** An explicit `--backend gcp` bypasses the auto
+   chain and silently forfeits the FREE `fellows` lane that auto tries first. Prefer the auto chain
+   (omit `--backend`) and let it walk fellows -> gcp; `--provisioning-model STANDARD` is the correct
+   remedy for FLEX_START preemption on the GCP rung, NOT a gcp pin. Reserve an explicit `--backend
+   gcp` for a named residual gap (e.g. the SLURM venv-extras mismatch, a sentinel/`/workspace`
+   requirement the target lane cannot meet) and name that gap in the launch marker note.

Optionally (planner's call, if judged mechanizable and low-noise): a `workflow_lint.py` check, or a
`dispatch_issue.py` stderr WARN, when an explicit non-auto `--backend` is passed without a stated
reason. A WARN is likely the right severity — explicit pins are legitimate for the named gaps.

## Scope / surfaces

- Primary target: `CLAUDE.md` (the § "Compute backends — multi-lane router" bullet list, near line 279)
- Possible secondary (planner's call): `scripts/workflow_lint.py` (a new check) and/or
  `scripts/dispatch_issue.py` (a dispatch-time WARN).
- Grep the workflow surface for sibling guidance that should stay consistent
  (`grep -rn "out of habit\|auto_fallback\|DEFAULT_AUTO_LANE_ORDER" .claude/ CLAUDE.md scripts/`) and
  update every hit that would otherwise contradict the new caution; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Do NOT change `DEFAULT_AUTO_LANE_ORDER` or any router precedence: the lane order is already correct;
  this is a GUIDANCE + (optional) WARN gap, not a routing bug. The durability pin
  `tests/test_router.py::test_default_auto_lane_order_is_gcp_first` must stay green and untouched.
- Keep the existing runpod habit-guard intact; the new text is its free-lane sibling, not a replacement.
- Any new lint/WARN must be WARN-severity (explicit pins are legitimate for the named residual gaps)
  and must not fail existing dispatches.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance
  line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: (computed by the filer at creation)

Origin: orchestrator's own observation on #1739, 2026-08-02. The user asked "can you run it faster by
using runpod/afp?" (afp = charmander/fellows), which surfaced that every dispatch this session had
pinned `--backend gcp` and never tried the free lane; the user then asked "isn't that in the workflow
already?" — it is, but only as the `auto` default, with a habit-guard written for runpod alone.
