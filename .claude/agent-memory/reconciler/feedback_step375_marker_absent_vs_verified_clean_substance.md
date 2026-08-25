---
name: step375-marker-absent-vs-verified-clean-substance
description: Step 3.75 symbol-rename marker-absent is letter-Critical, but with the grep independently verified clean (zero stale hits) it downgrades to a marker-amendment CONCERN; Claude split-reviews omit the round-level Step 3.75 check entirely
metadata:
  type: feedback
---

`code-reviewer.md` Step 3.75 assigns Critical (`substantive`, tag
`symbol-rename-grep-absent`) to a module-exported rename whose implementation
marker lacks the `### Symbol-rename grep` section — and that tag is NOT in
the Step 5c-bis mechanical strip list. But the Critical encodes the risk that
the audit was NEVER PERFORMED. When the reconciler (and/or the flagging
reviewer) independently re-runs the whole-tree grep and finds ZERO stale hits
with the sole call site updated in-diff, the crash-prevention substance is
discharged and only the durable RECORD is missing: downgrade to CONCERN
(defer-concern --by reconciler + re-raise at CONCERN) with the remedy "amended
`epm:experiment-implementation` v<n+1> carrying the exact grep command +
per-hit disposition" — no code change, no re-roll. The CONCERN tier still
blocks advance until the amendment lands, so the duty's enforcement survives
at marker-amendment cost.

**Why:** #2479 r4 — Codex Critical'd the `item_set_failures`→`item_failures`
rename (marker v1–v5 all lacked the section; trigger fires on the -def/+def
pair) while its own probe showed the tree clean; reconciler rg confirmed 0
hits. Matches the round-1 precedent on that task (marker-record defects with
verified-sound substance ride as mechanical items) and the step-literal-vs-
purpose calibration ([[codex-step-06-literal-vs-purpose]]). An UNVERIFIED or
hit-bearing grep is the opposite case: `symbol-rename-sibling-hit` with a
named file:line stays Critical/blocking.

**How to apply:** (1) re-run `rg -n '<old>' scripts/ src/ tests/` yourself —
zero hits + in-diff call-site fix → downgrade; ANY uncovered hit → uphold
FAIL. (2) Claude-side miss pattern to check on every split review: per-commit
sub-reviews (g1/g2) each assume the other covers round-level marker duties —
#2479's g1 carried NEITHER the Step 3.75 check NOR the required auditable-N/A
line. Scan Claude sub-verdicts for the missing `Symbol-rename grep:` line
whenever the round diff pairs a -def/+def, -class/+class, or module-constant
rename.
