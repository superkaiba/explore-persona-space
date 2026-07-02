---
name: Claude generalizes neutral-panel validity to behaviors never read on that panel
description: Statistics/Methodology lens — Claude APPROVEs a geometry/probe read by inheriting a panel's validation from ONE behavior to OTHER headline behaviors the parent never actually read on that panel; verify the cited parent JSON's arm set before crediting "same neutral read".
type: feedback
---

When the Claude critic APPROVEs a plan that reads a proxy (activation-shift
geometry, residual direction, probe panel) and justifies validity by
"inherited validated read — the parent already extracted on this exact
panel," CHECK WHICH BEHAVIORS/ARMS THE PARENT ACTUALLY READ before crediting
the inheritance. Claude has generalized a panel-validity result from the one
behavior the parent validated (em) to other headline behaviors the parent
NEVER read on that panel (fact, sycophancy), treating "same neutral read"
as sufficient when the live question is behavior-SPECIFICITY of the
extracted direction.

**Why:** #651 r1 statistics lens. Plan read trained-base activation-shift
U1 on the #551 14x20 neutral panel for 4 behaviors (marker, fact, em,
sycophancy) and claimed behavior-level write directions. §6.1 validation
column: "Inherited validated read: #521 established the EM one-direction
result on this exact read." But `eval_results/issue_552/cross_arm/summary.json`
`per_cell` arms = ['benign','em','marker'] — fact and sycophancy were NEVER
read on this panel. A neutral-panel U1 for fact/sycophancy could be a generic
adapter/SFT direction, not the behavior write, flipping the Q1/Q2
conclusions. Codex caught it (REVISE, mechanizable:yes); Claude APPROVEd,
conceding the panel question is about the WRITE construct but asserting the
panel is "on-distribution for the write-direction construct across all four
behaviors" — which is exactly the unvalidated claim. Sided with Codex.

**How to apply:** On any geometry/probe/proxy read whose validity is claimed
by parent-inheritance: (1) open the cited parent result JSON and read its
arm/behavior/condition set; (2) confirm EVERY headline behavior in the new
plan is in that set, not just the canonical one. The standard safeguards
(seed ceiling, sign-flip/row-shuffle nulls, a null-check row that catches a
FAILED implant) DO NOT establish behavior-specificity — a generic LoRA
direction is seed-stable, concentrated, and not a failed implant, so it
passes all of them. The missing piece is a construct-validity bridge:
extract the same read on each behavior's CANONICAL elicitation surface and
compare to the neutral-panel U1. If that bridge is absent for any headline
behavior and the parent never read it on the panel, the conclusion for that
behavior is artifact-confoundable -> REVISE (recoverable when the bridge is a
cheap no-new-training add over existing artifacts; the project's
measurement-validity rule requires validating the construct on the
distribution the behavior occurs on). This is the plan-stage sibling of the
analyzer-side "metric must measure the Goal's construct on-distribution" rule.
