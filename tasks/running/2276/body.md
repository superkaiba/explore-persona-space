---
title: 'workflow-fix: verify_plan.py — §9 backend pin-claim vs frontmatter + declared
  GPU width vs launch-command width flags'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-13T20:19:31Z'
has_clean_result: false
origin_prompt: 'Methodology critic prose follow-up on #2225 fu1 plan v9 review (2026-08-13):
  phantom backend-pin claim recurred twice in the #2225 lineage; width-flag mismatch
  between §9 8xH100 spec and the embedded dispatch command.'
workflow: v1
---
# workflow-fix: verify_plan.py check — §9 backend pin-claim vs actual frontmatter + declared GPU width vs embedded launch-command width flags

## Provenance

workflow_fix_target: scripts/verify_plan.py
Surfaced by the Methodology critic on task #2225's fu1_preimage_prevention plan review (2026-08-13, plan v9), as a prose workflow-surface follow-up in its report. Auto-filed by the #2225 follow-up-round orchestrator per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups clause).

## The gap

Two related plan-drift shapes that `verify_plan.py` currently cannot catch, both realized in the #2225 lineage:

1. **Phantom backend-pin claim.** A plan's §9 backend line claims a frontmatter pin (e.g. "**Backend:** `backend: runpod` (parent pin inherited)") while the task's `body.md` frontmatter carries NO `backend:` key. Recurred twice in one lineage: parent #2225 plan v5 line 274 claimed an "explicit frontmatter pin" that was never set (the parent launched via `pod.py provision` directly, masking the drift); the fu1 plan v9 then claimed to "inherit" the nonexistent pin. Consequence when the plan's dispatch command is actually run: it routes `auto` instead of the pinned lane — for a sentinel-signaling workload (`/workspace/logs/...` sentinels) the SLURM fall-through rungs without `/workspace` violate the plan's own stated hazard (#608).

2. **Width mismatch between §9 spec and the embedded launch command.** The plan's §9 declares an N-GPU spec (e.g. "one 8×H100 pod", all wall rows costed at 8-wide sharding) while the embedded `dispatch_issue.py launch` line carries no width flag, so the intent default (e.g. `lora-7b` → 1×H100) delivers 1/8 the width; a `--time-budget-hours` fence sized to the N-wide wall then TIMEOUTs the ~N×-longer narrow run mid-flight.

## Proposed check (sketch from the critic's report)

New WARN-or-FAIL check in `scripts/verify_plan.py` (issue-context mode):
- (a) when the plan text claims a frontmatter backend pin (pattern: `backend:\s*(\w+)` near "pin"/"inherited"/"frontmatter" in the §9/backend region), assert the task's `body.md` frontmatter actually carries that key=value; mismatch ⇒ FAIL with the exact claimed lane vs the actual frontmatter state.
- (b) when §9 declares an N-GPU spec (patterns like `(\d+)×\s*H\d+` / `--gpu-count (\d+)`), assert any embedded `dispatch_issue.py launch` fence carries a width flag (`--gpus` / `--gpu-count`) consistent with N; absent/mismatched ⇒ WARN naming the intent's default width.
- Standalone `--plan-file` mode (no task context): arm (a) degrades to SKIP (no frontmatter to check); arm (b) still runs.
- N/A escapes per house style (e.g. `N/A — no backend pin claimed`), plus tests in `tests/test_verify_plan.py` pinning both incident shapes (the #2225 v5/v9 phantom-pin text and the width-mismatch dispatch line) and the clean forms.

## Acceptance

- Both #2225 incident shapes reproduce as fixtures and are flagged; a plan with a real matching frontmatter pin and a width-consistent launch line passes.
- `verify_plan.py --issue`/`--plan-file` exit codes and JSON contract unchanged; new check listed in the check table.
- Durability pin: tests/test_verify_plan.py::<new test names>.

## Realized deviation (implementation, 2026-08-13)

Arm (a) shipped as **c62 WARN-only, not FAIL**: the plan's pre-registered >2-false-positive posture rule fired on the corpus-calibration sweep (107 FAILs at the designed FAIL polarity across 4,089 plan versions, 3 adjudicated FP classes — dominant: prospective/dispatch-flag pin phrasing), so the polarity downgraded per plan §4 step 6; the calibration comment above the check carries the measured numbers, and both #2225 incident fixtures (v5-v9) are recovered as WARNs. Arm (b) is **c63** (WARN-only as planned). Check ids are c62/c63, not the sketch's c60 — c60/c61 were taken by #2255/#2275.
