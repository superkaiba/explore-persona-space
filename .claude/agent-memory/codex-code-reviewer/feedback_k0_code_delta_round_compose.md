---
name: k0-code-delta-round-compose
description: Composing a fresh round-1 review of a plan-staged K0 code-delta round (pod smoke deferred to a later K1 phase) with open concerns scoped to a DIFFERENT round series (#2564 k100 r1)
metadata:
  type: feedback
---

Two compose shapes from #2564 k100 r1 (2026-08-26), both reusable:

**K0 deferred-smoke fence.** When the plan stages the round as K0 = code
delta + CPU tests with the real-slice smoke deferred to the pod (K1 chained
smoke, inputs don't exist yet), the compose must pre-adjudicate BOTH
type:experiment smoke gates or Codex FAILs on plan-sanctioned absence:
(a) Step 0.6 — state the K0 smoke bar explicitly (the CPU test battery +
ruff + import-checks + arg-parse smokes from marker (c)); a
`smoke-run-missing` FAIL for "no end-to-end digest" contradicts the
APPROVED plan; genuine absence applies only if the K0-bar evidence itself
is absent/false; redirect the reviewer's energy to Step 4.5-style
presence+substance reads of the named tests (claimed-but-absent test =
ordinary-bar substantive FAIL). (b) Step 0.55 — the presence gate is
per-TASK, so a PRIOR round's smoke-arch marker (here ffr v4, arms
pilot/A/B) satisfies it; inline it with provenance and fence: do NOT grade
the stale arms against the new plan, do NOT marker-shape-FAIL the absent
round-specific marker (the K1 phase owns it; implementer (b) discloses).

**Cross-round open-concerns fence (fresh r1, no assigned discharges).**
`list-concerns --open-only` returning rows ALL raised in a different round
series (ffr rows on a k100 round) gets: (i) no re-emitted `CONCERN:: ` row
unless disposition CHANGES; (ii) severity cap at recorded severity;
(iii) the REAL duty — a touched-file CLASS sweep: the round edits the same
shared files, so instruct checking whether NEW round code re-instantiates
each row's defect class (resume-fp identity omission, duplicate-row
last-wins collapse, `.get` fail-open), routed as ordinary-bar findings
under FRESH ids; (iv) a three-way per-id status line in a `## Prior open
concerns` verdict section: UNTOUCHED-BY-ROUND | TOUCHED-no-disposition-change
| DISPOSITION-CHANGED.

**Why:** Step 0.6/0.55 genuine-absence text and Step 0.8 inheritance are
written for the common case (smoke due this round; concerns from this
series); without the fences the twin burns the round on plan-contradicting
mechanical FAILs or double-persists foreign-round ids.

**How to apply:** any round whose plan stages execution later (K0/K1
split, code-then-pod), and any fresh round-1 on a task with open ledger
rows from prior rounds. Also from this compose: verify the brief's
excluded-sync-commit list against `git log` yourself — the brief named 2
of 3 sync commits; verify all same-class (byte-identical to origin/main,
touching none of the reviewed files) and name all three in the exclusion
block. Related: [[concern-discharge-round-severity-fence]],
[[revision-round-compose-recipe]].
