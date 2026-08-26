---
name: fixed-string-pin-sweep-recount
description: "Step 4.6 recounts run grep -F per fragment — an unescaped dot in a filename fragment (bootstrap_pod.sh) regex-matches unrelated docstrings and can wrongly promote a slow-deferred file to a NOT-RUN pin hit (#2606 r1)"
metadata:
  type: feedback
---

When re-deriving the marker's pin-sweep hit-file set at compose time (the
Step 4.6 diff-consistency ground truth handed to the twin), run every
fragment as a FIXED STRING (`git grep -F`), never as a bare regex. Filename
fragments carry dots: on #2606 r1 the fragment `bootstrap_pod.sh` regex-matched
the docstring text "bootstrap_pod shape" in `tests/test_workflow_lint.py:3275`
— the report's slow-deferred NOT-RUN file — which would have primed the twin
to apply the 4.6(ii) NOT-RUN-pin-hit blocker-adjacency presumption to a file
with ZERO real fragment hits. The `-F` recount matched the claimed count
exactly (21) and re-classified the deferral as a plain slow invariant file.

Two companion duties from the same compose:

1. **Hand BOTH residual adjudications, pre-triaged.** A genuine hit file
   absent from the marker's PRINTED map-files list (here
   `tests/test_issue2184_noport_wedge.py`, one `BOOTSTRAP-FAILED` hit) whose
   containment claim rests on an UNPRINTED selector set composes as an
   adjudication duty routed at most Minor `substantive` per 4.6(iii) — never
   a marker-shape blocker.
2. **Family re-grep claims get a composer recount with namespace pre-triage:**
   a marker claiming "exit-code N is free — only <one file>" is recounted
   repo-wide; extra hits in UNRELATED process families (SLURM dispatch shell,
   analysis-script typed halt) pre-triage as at most a Minor report-accuracy
   note with the collision question left to the twin FROM THE CODE
   ([[infra-wf-fix-lint-gate-compose]] stats-hygiene pattern applied to exit
   codes).

**Why:** a false pin-hit promotion inverts the 4.6(ii) presumption and can
drive a false Major on a compliant deferral; a silently-accepted family-grep
claim hides a real exit-code collision.
**How to apply:** every compose that re-derives pin-sweep hits or verifies a
"literal X is free/unique" marker claim — `grep -F` per fragment, diff the
hit-file set against the printed list, pre-triage residuals both ways.

**#2607 r1 (2026-08-26) sharpening — recount changed-path fragments at THREE
grains:** a marker's changed-path hit set may be reproducible ONLY under a
fragment grain it never states. On #2607 the full-path form
(`scripts/verify_task_body.py`) missed most claimed hits; the BASENAME form
(`verify_task_body.py`) reproduced the run-set coverage; the BARE MODULE
token (`verify_task_body`) surfaced one extra NOT-RUN file whose sole hit
was a prose comment (test_workflow_followup_labels.py:1260) — handed as an
adjudication with the pre-read that a comment pins nothing. Run all three
grains, attribute each extra to its grain, and pre-triage: (i) round-NEW
literals with zero tests/ hits can have NO stale pins — say so, so the twin
never hunts them; (ii) a GENERIC single-token fragment (bare `REFUSED`, 26
files) is a form artifact — name the do-not-promote rule explicitly; (iii)
an unreproducible claimed hit file that WAS run anyway = at most Minor
report-accuracy. Same round validated: brief-ordered plan inlining on an
identical worktree copy (truthful belt-and-braces envelope), origin_prompt-arm
wf-fix floor detection with the #2306 plan-verify-at-v2/plan-now-v3 dual
attest, and the disclosed round-1-pin-FAIL-fixed-by-refactor-commit shape
composed as verify-the-fix-commit, not a #1672 finding.
