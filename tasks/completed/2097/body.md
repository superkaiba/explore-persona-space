---
title: 'workflow-fix: mechanize local-disk headroom for staging + VM fits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:08572ce00252
created_at: '2026-08-05T20:21:41Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: src/explore_persona_space/orchestrate/hub.py,\
  \ src/explore_persona_space/orchestrate/preflight.py, scripts/verify_plan.py\nbug_observed:\
  \ No mechanized local-disk headroom check protects VM-routed staging or fit phases:\
  \ routing keys on projected RSS/footprint, preflight checks / only, and stage_hub_prefix\
  \ downloads multi-GB with no local disk assert\nwhy_workflow_gap: CLAUDE.md's dispatch-time\
  \ df -P + 1.5x headroom duty is prose-only; assert_out_root_headroom exists but\
  \ is opt-in (27 per-issue callers), and plan-compute-sizing.md:293 explicitly deferred\
  \ the verify_plan backstop\nproposed_change: Mechanize the local-disk headroom duty:\
  \ self-sizing assert in stage_hub_prefix, a data-disk floor arm in preflight, and\
  \ the deferred verify_plan staging-row WARN\ndiff_sketch: |\n  + stage_hub_prefix:\
  \ sum server-side listed sizes × 1.5 vs statvfs at dest mount via assert_out_root_headroom\n\
  \  + preflight: data-disk floor arm (percent-based, live-mount-gated, clean skip\
  \ when absent)\n  + verify_plan: c45-style WARN — multi-GB staging row must name\
  \ its mount/staging path\nconfidence: high\nrelated_task: #2054 (inherits the gap;\
  \ not the origin)\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a chat-mode deep dive (2026-08-05, user-directed: "Do a deep dive and propose workflow fixes" on the fleet-wide disk pressure — both disks at 96–98%). Emitting agent: orchestrator + a read-only routing-gate audit subagent.

## Goal

Mechanize the local-disk headroom duty: a self-sizing assert in `stage_hub_prefix`, a data-disk floor arm in preflight, and the explicitly-deferred verify_plan staging-row WARN.

## Workflow gap

- **Bug observed:** No mechanized local-disk headroom check protects VM-routed staging or fit phases: routing keys on projected RSS/footprint only (RSS ≥ ~16 GB → cpu-mid/bigmem; footprint > 50 GB → off-VM — both plan-time projections, never a live free-space read), preflight's 40 GB floor checks `/` only and only when preflight runs (inline analysis rounds typically don't), and the canonical staging helpers `stage_hub_prefix`/`stage_hub_file` assert HF-ACCOUNT storage headroom (#564/#1034) but nothing about local disk before multi-GB downloads. At 96–98% full, a phase projecting 8 GB RSS and 20 GB footprint legally routes VM-local and dies ENOSPC (the #1393 class: a 14 GB inline HF pull filled `/`). The #2054 fit-family pilot-gate inherits exactly this gap.
- **Why it is a workflow gap:** CLAUDE.md's compute-character pre-launch statement mandates, in prose, a dispatch-time `df -P` + ≥1.5× headroom check for ≥5 GB stages — but no code path enforces it. The mechanized helper exists (`orchestrate/preflight.py::assert_out_root_headroom`, L620: statvfs + 1 GB fallocate canary at the mount the out-root RESOLVES to) yet is opt-in with 27 per-issue script callers and no default wiring; and `plan-compute-sizing.md:293` explicitly records the plan-side deferral ("NO verify_plan.py backstop in v1 of this block").
- **Confidence (emitter):** high for legs (1)+(2); medium for leg (3) (WARN-only prose matching — the planner may descope it with a stated reason).
- verified-at-filing (re-run by the filer 2026-08-05):
  - `grep -rcE 'eps-data|EPS_VM_DATA_DISK' src/explore_persona_space/orchestrate/preflight.py` → 0 hits; relocation sweep `grep -rlnE 'eps-data|EPS_VM_DATA_DISK' src/explore_persona_space/orchestrate/` → `env.py` only (thread-cap detection) — preflight never looks at the data disk.
  - `grep -cE 'statvfs|assert_out_root_headroom|shutil\.disk_usage' src/explore_persona_space/orchestrate/hub.py` → 0 hits — the staging helpers have zero local-disk asserts (their "headroom" functions are HF-account storage).
  - `grep -rln 'assert_out_root_headroom' scripts/ | wc -l` → 27 (adoption is per-issue opt-in); helper defined at `preflight.py:620`.
  - `grep -n 'NO verify_plan.py backstop' .claude/rules/plan-compute-sizing.md` → 1 hit (L293) — the plan-side backstop is a NAMED v1 deferral.
  - Landed-fix history: `git log --oneline --since='14 days ago' -- src/explore_persona_space/orchestrate/hub.py src/explore_persona_space/orchestrate/preflight.py scripts/verify_plan.py` → recent commits are EBADF-probe fixes (22c2ddb2d3, d47090010b), the #1849 upload-destination guard (08db1b4854), and verify_plan c44/c12/c20 — none add a local-disk headroom check.

## Proposed change (candidate diff sketch — refine in planning)

```
+ (1) hub.py stage_hub_prefix: it already holds the server-side file listing BEFORE downloading —
+     self-size (sum of file sizes × 1.5) vs statvfs free at the DESTINATION mount by calling
+     assert_out_root_headroom(dest, need_gb, phase="hub-staging"); fail-loud RuntimeError per project
+     fail-fast, env override for deliberate low-headroom runs. stage_hub_file gains an optional
+     size-bytes arm (cheap HEAD/listing where available).
+ (2) preflight.py: a data-disk arm mirroring _check_vm_root_floor — fires when /mnt/eps-data is a live
+     mount, PERCENT-based per the watcher's size-invariant EPS_VM_DATA_DISK_SUBFLOOR_PCT convention
+     (a GB floor breaks on resize); clean skip when the mount is absent (pods/GCE unaffected).
+ (3) verify_plan.py: the deferred c45-style WARN — a §9 row carrying a multi-GB staging/footprint signal
+     must name its mount/staging path (c39's conditional-WARN pattern is the template).
```

MooseFS caveat for leg (1): on RunPod `/workspace` the per-pod ~130 GB quota is invisible to statvfs (`.claude/rules/gotchas.md` EDQUOT entry) — the helper's fallocate canary is the meaningful probe there; keep it, and do not let a statvfs pass read as quota headroom in the failure message.

## Scope / surfaces

- Primary targets: `src/explore_persona_space/orchestrate/hub.py`, `src/explore_persona_space/orchestrate/preflight.py`, `scripts/verify_plan.py`
- NOT in scope (deliberate): the routing gates themselves (RSS / 50 GB thresholds) — projection-based by design; injecting a live-free read there changes plan/critic semantics. This fix mechanizes existing runtime duties without touching routing.
- Cross-references: #1333 (the out-root-on-different-filesystem gap the helper was built for), #1393 (the inline-pull ENOSPC incident behind the df -P mandate), #2054 (the live task whose fit pilot-gate inherits the gap), #1979/#1947 (the EBADF probe hardening this must not regress).
- Grep before editing: `grep -rln 'assert_out_root_headroom\|stage_hub_prefix' src/ scripts/ .claude/` and reconcile every documenting surface.

## Constraints / invariants

- Workflow-surface only (hub.py/preflight.py are in-scope orchestrate modules used by the workflow; verify_plan.py is a listed helper). Fail-fast: the new asserts raise loud, never degrade to silent skips; overrides are explicit env, mirroring EPM_PREFLIGHT_DISK_FLOOR_OVERRIDE.
- `scripts/workflow_lint.py --check-asks` + ruff pass; add tests pinning: hub staging refuses under low headroom, preflight data-disk arm skips cleanly when the mount is absent, verify_plan WARN fires on an unbound multi-GB staging row.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/hub.py, src/explore_persona_space/orchestrate/preflight.py, scripts/verify_plan.py
- fingerprint: 08572ce00252

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/orchestrate/hub.py, src/explore_persona_space/orchestrate/preflight.py, scripts/verify_plan.py
bug_observed: No mechanized local-disk headroom check protects VM-routed staging or fit phases: routing keys on projected RSS/footprint, preflight checks / only, and stage_hub_prefix downloads multi-GB with no local disk assert
why_workflow_gap: CLAUDE.md's dispatch-time df -P + 1.5x headroom duty is prose-only; assert_out_root_headroom exists but is opt-in (27 per-issue callers), and plan-compute-sizing.md:293 explicitly deferred the verify_plan backstop
proposed_change: Mechanize the local-disk headroom duty: self-sizing assert in stage_hub_prefix, a data-disk floor arm in preflight, and the deferred verify_plan staging-row WARN
diff_sketch: |
  + stage_hub_prefix: sum server-side listed sizes × 1.5 vs statvfs at dest mount via assert_out_root_headroom
  + preflight: data-disk floor arm (percent-based, live-mount-gated, clean skip when absent)
  + verify_plan: c45-style WARN — multi-GB staging row must name its mount/staging path
confidence: high
related_task: #2054 (inherits the gap; not the origin)
<!-- /workflow-fix-candidate -->
