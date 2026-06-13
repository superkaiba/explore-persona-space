---
name: Calibrate new verifier checks on the real offender artifact
description: A new heuristic check in verify_plan.py / verify_task_body.py must be run against the incident task's ACTUAL plan/body before committing — synthetic fixtures alone false-PASSed the exact offender once
type: feedback
---

When adding a heuristic check to a mechanical verifier (`verify_plan.py`, `verify_task_body.py`, `audit_clean_results_body_discipline.py`), run it against the REAL offending artifact named in the candidate (`--plan-file` pointing at the main checkout's `tasks/*/<N>/plans/v*.md` — never `--issue` mode from a worktree) before committing.

**Why:** the #633 c11 check (dry-run smoke ↔ dry-run test) passed all 8 synthetic fixture tests but false-PASSed on #633's actual v1 plan — the real corpus had ONE "Verification commands" line carrying both the pytest invocation (a `test_` identifier) and the `--dry-run` smoke command, satisfying the tier-1 evidence scan. Fix: strip `--dry-run` flag occurrences from the line before the evidence match; the offender shape became a pinned regression test.

**How to apply:** after the synthetic test suite passes, run the verifier on the related_task's newest plan/body AND (when the candidate says "Nth occurrence") the earlier offenders too; expect the check to flag them. If it doesn't, the heuristic has a self-certify hole — find the line that satisfied it. Sibling memory: feedback_new_lint_check_waiver_sweep (prototype lint checks against the live tree).
