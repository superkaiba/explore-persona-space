---
name: scaffold-numerals-multiset-supply
description: Compose connective/scaffold text numberless — inlined plan/lens spans consume their own numeric supply exactly, so any scaffold restatement of a plan number residuals the Step-4 multiset check
metadata:
  type: feedback
---

Compose the lens prompt's connective text (surfaces list, grounding line, output
contract) NUMBERLESS, or only with numerals the BRIEF itself handed (the brief
text is a handed span; the plan/lens spans are not free supply).

**Why:** the Step-4 leak check multiset-subtracts handed spans from the prompt.
The inlined `plan_body` and `lens_items` cancel their own copies EXACTLY, so any
composer restatement of a plan/lens number ("§4.7", "|sep| ≥ 0.5", "rule 26",
"ci95_...") residuals regardless of how often the plan says it — first assembly
for #2389 round 1 threw 14 BLOCKERs this way. Also: strip task-ref tokens
(`#N`, `tasks/<status>/<N>`, `issue[-_]N`) from the working text BEFORE numeric
tokenization (per `codex-critic.md` Step 4), or issue ids like 2329 flood the
numeric residuals.

**How to apply:** (1) cite plan locations as "the plan's Measurement-validity
table" / "the Decision-Gates section", not "§6"/"§7"; (2) restate only
brief-handed numerals, ≤ once each; (3) keep the handed-brief record VERBATIM —
never add supply lines to it to make the check pass; (4) verifier = ref-strip →
multiset-subtract handed spans → SET-membership allowlist {0,1,2,3,4,5,500} →
collect-all BLOCKERs, single exit; (5) brief-handed IDENTIFIER strings (the
`followup_label` like `q35_ladder_decay`, pod slugs) go into the handed-span
file alongside the brief-handed PATH strings — same class, same rationale as
the paths clarification; #2329 round 1: the label's `35` false-positived until
the verbatim brief-handed label line was added to the handed-span file (this is
recording a handed string verbatim, not adding supply — rule (3) unviolated).
