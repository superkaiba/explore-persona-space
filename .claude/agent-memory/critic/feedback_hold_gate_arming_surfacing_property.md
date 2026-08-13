---
name: hold-gate-arming-surfacing-property
description: "Hold-gate/launch-confirm prose plans: verify each sanctioned wakeup rung actually SURFACES the hold (check tick_triage keying), and re-derive mapped-test claims via the real mapper (#2135)"
metadata:
  type: feedback
---

Two checks for infra plans adding pod HOLD-gate / launch-confirmation prose
contracts (#2135 plan v1):

1. **Surfacing property per wakeup rung.** A clause that requires "arm a
   VM-side wakeup" and lists rungs (bg-Bash poll chain / Monitor loop /
   `/issue-tick` backstop cron) closes the incident only if EVERY rung is
   required to SURFACE the hold to a deciding actor. `scripts/tick_triage.py`
   keys on task STATUS (`ISSUE_GATE`/`ISSUE_TERMINAL`), marker staleness, and
   pid breadcrumbs — it cannot see a `status=running` task whose pod parked at
   a bespoke log-line HOLD while the session posts other markers. A qualifier
   ("a status=running read over a parked workload does not count") scoped
   grammatically to only the first rung leaves the cron rung
   formally-compliant-but-blind — reproducing the #1947 pod-1947-r3 shape.
   Fix shape: bind the surfacing requirement to all rungs, or require the
   machine-legible emission (the rule's `gate=`/`blocks_pipeline` sentinel
   shape, pod-side-reporting.md ~L90) whenever the cron is the named wakeup.

2. **Mapped-test claims drift.** A plan's "mapped pin tests (from
   select_step9c_tests.py --map-files)" list must be re-derived: the flag
   takes a PATH-LIST FILE, not source files, and plan-authored lists have
   been wrong (v1 of #2135 named 4 tests; the real mapping was 11, including
   `test_guard_lessons_edit.py` and `test_poll_pipeline_stale_pid_warn.py`,
   and returned NO pair for experimenter.md). Step 9c self-corrects, so this
   is Concern-level, but always paste the real mapper output.

**Why:** both were caught only by reading tick_triage + running the mapper —
the plan text alone read plausibly.
**How to apply:** any plan touching pod-side-reporting.md hold/launch
contracts, or claiming a mapped-test list. See also
[[infra-plan-review-checklist]].
