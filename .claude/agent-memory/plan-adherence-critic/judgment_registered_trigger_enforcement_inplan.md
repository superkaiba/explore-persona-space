---
name: judgment-registered-trigger-enforcement-inplan
description: Enforcement code for a plan-registered trigger (cap-hit >2% => regen @4096) is IN PLAN, not scope creep; check the contingency-overrun disclosure marker + the breach-report/superseded-rows overwrite-in-place carry risk
metadata:
  type: feedback
---

Enforcement code for a REGISTERED plan trigger is in-plan even when the plan's §4.6
fork/phase list never enumerated a code slot for it. **Why:** #2329 registered
`max_new_tokens=2048, cap-hit > 2%/cell => re-gen at 4096` (plan §4.5 + §11 Source: #2162)
and booked "cap-hit re-gen @4096" in the §9 contingency row, but enumerated no
cap_report/capregen phase; when the trigger fired (18/36 gate-slice cells), the
enforcement diff was the registered remedy executing, not new capability. The manifest
carried "cap-hit fraction" as a metric, which corroborates a planned report artifact.
**How to apply:** (1) locate the trigger in the plan (registration + remedy + contingency
booking + manifest metric) before calling enforcement code scope creep; the phase-list gap
is a Minor needing only an in-diff/report stated reason. (2) The contingency-BUDGET
overrun (realized remedy ~10x the booked reserve) is a deviation whose DISCLOSURE surface
is an epm:progress marker naming the plan line, the booked figure, the re-projection, and
"recorded, NOT pivoted" (#1771 no-cost-gate) — that shape satisfies the stated-reason bar;
judge disclosure, never the spend. (3) Two carry risks to flag every time a
measure-then-regen remedy lands: the breach-driving report gets OVERWRITTEN IN PLACE by
the post-regen re-emit at the same canonical path (done records pin its sha but not its
per-cell table — recommend commit/upload-before-regen or a versioned filename), and the
superseded pre-regen rollout rows are replaced wholesale at the SAME local+HF paths
(recoverable only via HF revision history) while plan §10 may say `discarded_artifacts:
none` — route to upload-verifier + require the report to state pre/post tables and the
mixed-cap covariate. Related: [[judgment-preregistered-gate-relaxation-checklist]].
