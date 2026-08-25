---
name: plan-version-bump-disclosure-and-roster-check
description: Disclosure closures landing as a NEW plan version (Arm W immutability) get a composer-diffed delta attestation + missing-plan-verify note at CONCERNS-grade-at-most; verify marker-claimed roster memberships (LIVE_WORKFLOW_HELPERS etc.) against the actual roster before attesting
metadata:
  type: feedback
---

Two compose-time verifications from #2336 r3 (plan v4 batch 1, 2026-08-24):

1. **Disclosure-by-plan-version-bump.** When a reconciler-routed disclosure
   closure lands as a NEW plan version via `task.py new-plan-version`
   (because `workflow_lint --check-plan-version-immutability` Arm W (#2123)
   FAILs in-place edits of persisted `plans/v<K>.md`), the composer: (a)
   diffs vK vs vK+1 itself and attests the exact delta (here: ONE changed
   line — the §4 FN-profile bullet gaining entries (j)/(k)); (b) probes
   events.jsonl for a plan-verify marker on the NEW version and attests its
   absence neutrally — pre-fence it at CONCERNS-grade-at-most (the prior
   version's PASS covers all unchanged content), never a marker-shape FAIL
   ground; (c) inlines the changed line's new text verbatim in an attested
   fact so the wording adjudication can never be plan-lens BLOCKED; (d) adds
   a one-line REQUIRED mechanism-ruling header slot. Also: when the worktree
   plan copy is frozen at the OLD version, order absolute-canonical-path
   reads with NO worktree fallback (#2422).

**Why:** the twin must judge disclosure WORDING; without the composer-diffed
delta + inlined text it either re-derives the delta (wasted effort, error
surface) or FAILs on the missing fresh plan-verify marker.

2. **Roster-membership overstatement check.** A marker's (c) ruff-policy
   line may claim more roster members than real (#2336 r3 marker claimed
   all 4 touched scripts were LIVE_WORKFLOW_HELPERS; grep found only 2 —
   sync_repo_root.py + workflow_lint.py). Composer greps the roster itself,
   attests the correction, and pre-triages it as a transcription-level
   report-accuracy nit (at most Minor) since the pin DUTY binds on >=1
   member and the pin passed.

**How to apply:** any round whose brief mentions a plan re-version, a
disclosure rider, or a ruff-policy pin field — run both probes before
composing the attested-facts section. Related: [[disclosure-round-compose]],
[[concern-discharge-round-severity-fence]], [[revision-round-compose-recipe]].
