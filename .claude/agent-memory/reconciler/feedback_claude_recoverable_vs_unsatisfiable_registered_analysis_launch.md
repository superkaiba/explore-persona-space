---
name: Claude calls an unsatisfiable registered analyzer launch "code-reviewer-recoverable"
description: critic-lens reconcile — when the plan's SOLE registered analysis command + a registered config value jointly crash the inherited analyzer AND no diff in the plan's own diff set closes the gap, the rubric ("REVISE = plan missing something the pipeline cannot recover from") sides REVISE, not Claude's "fails loud, zero GPU, code-reviewer will catch it"
type: feedback
---

When adjudicating a `critic`/Methodology PASS(APPROVE)-vs-FAIL(REVISE) split where the
dispute is a plan-level analyzer-wiring defect, the decisive test is the SKILL rubric:
**REVISE = the plan is missing something the analyzer pipeline cannot recover from** —
i.e. does the PLAN AS REGISTERED carry the fix, NOT whether a future code-reviewer could
patch it.

**Why:** #632 r1. Claude APPROVEd with concern #1/#2 = "recoverable at code-review,
fails loud, zero GPU"; Codex REVISEd Must-Fix #1 = same wiring. I sided Codex (REVISE).
Verified against the inherited `issue-610` `analyze.py`:
- `analyze.py:318` loads the comparator arm from `parent_sweep / chassis.chassis_slug`.
- `--parent-sweep` default = `eval_results/issue_600/sweep` (line 863).
- `chassis_slug="c610_mercenary_near_nodefault"` (plan Diff 1) exists ONLY under
  `eval_results/issue_610/sweep/` — CONFIRMED absent from the `issue_600` tree via
  `git show issue-610:eval_results/issue_600/sweep/.../trajectory.json` (empty).
- The plan's SOLE registered analysis command (§11) omits `--parent-sweep` →
  `load_arm` raises `FileNotFoundError`.
- Independently, `analyze.py:445` does `abs(j_median - chassis.replacement_ctrl_precedent)`
  with the plan's registered `replacement_ctrl_precedent=None` → TypeError (unguarded).
- The plan's own diff set (Diffs 1–6) edits `centering_set` + the build-spec assert but
  registers NO fix for either the launch command or the line-445 None-guard.

So the §6 registered comparison (the experiment's entire decision rule) could not be
produced from the plan as written. That is exactly "plan missing something the pipeline
cannot recover from."

**How to apply:** "fails loud / zero GPU to rerun / unambiguous intent / the code-reviewer
will catch it" is a recoverability-by-a-later-agent argument, NOT the rubric question. When
the SOLE registered analysis launch + a registered config value jointly crash the inherited
analyzer AND the plan's diff set never closes the gap, side REVISE. Distinguish from the
APPROVE case: if the plan's diff set DOES carry the wiring fix, or the crashing command is
merely one optional convenience launch among several working ones, it is a recoverable read
→ APPROVE. The test is "is the fix IN the plan," not "could someone add it."
