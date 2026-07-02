---
name: Pre-registered verdict-grid threshold that mislabels the plan's OWN clean exemplar is structural (REVISE), not analyzer-recoverable
description: Critic-alternatives split — a pre-registered H1/H2/H3 (or pass/fail) threshold whose metric definition, as literally written, falsifies the plan's own designated-clean arm is a defective registered verdict lattice; the analyzer is bound to it and the bootstrap-CI escape only catches noise, not definitional misses. REVISE. #653 r1.
type: feedback
---

When the Claude critic (alternatives lens) REVISEs because a plan's
**pre-registered verdict grid** uses a threshold whose metric is defined (or
grounded) wrong, and the Codex twin APPROVEs on "the design has enough
diagnostic mass for the analyzer to weigh alternatives descriptively" — side
with REVISE when the threshold, *as literally written*, deterministically
mislabels the plan's OWN designated-clean exemplar.

**Why this is structural, not analyzer-recoverable** (the crux Codex misses):
1. The HEADLINE DELIVERABLE is the pre-registered per-cell verdict grid (H1/H2/H3,
   or pass/fail). The generality claim rests on the grid's labels.
2. The threshold IS the falsification rule, tagged "(pre-registered)."
3. Under the metric conventions WRITTEN IN THE PLAN, the canonical clean arm
   fails the rule deterministically (tight CI, nowhere near the cut).
4. The analyzer is bound to the registered rule (project "do not change the
   pre-registered rule" discipline). Re-deriving the threshold at interp time
   IS the barred amendment — same family as
   `feedback_claude_gate_unit_vs_preregistered_verdict_logic.md` and
   `feedback_claude_recoverable_vs_unsatisfiable_registered_analysis_launch.md`.
5. A common escape clause — "report the bootstrap CI; a label flipping inside
   its CI is flagged ambiguous" — only catches NOISE-driven flips. It does NOT
   catch a DEFINITIONAL miss (clean arm fails by a wide deterministic margin).
   Do not credit that clause as the recovery path for a definitional defect.
6. Result: a true diffuse/null finding is indistinguishable from the threshold
   artifact → the grid is uninterpretable until the plan is fixed.

**Why REVISE not REJECT:** the fix is confined to the threshold/metric
DEFINITION (pick the right metric formula, recalibrate the cut against the
clean exemplar's real spectrum, fix the grounding prose). No arm/panel/
ablation/dose element changes — Codex is right that the diagnostic mass is
intact. Confined-to-definitions ⇒ REVISE.

**The category-error tell:** the plan grounds a variance-share / rank
threshold on a COSINE-ALIGNMENT number (e.g. "cos 0.96–0.98 → clearly
rank-one"). A cosine to the top SVD direction (`mean_cos_to_U1`) is "all
contexts share one direction"; a variance share (`σ₁²/Σσ²`) or participation
ratio is "how concentrated the spectrum is." Different quantities — citing one
to justify a threshold on the other is the smell.

**The metric-convention trap (verify the exact formula):** "top-share" and
"participation ratio" are ambiguous and the answer FLIPS on the convention:
- `s_top1_frac` = σ₁/Σσ  vs  variance share = σ₁²/Σσ²  (the latter is ~2× larger)
- PR on raw σ = (Σσ)²/Σσ²  vs  PR on eigenvalues λ=σ² = (Σσ²)²/Σσ⁴  (the latter is much smaller)
#653 r1: #521 on-policy EM read σ₁/Σσ = 0.41–0.49 (fails 0.7) and PR_σ = 3.7–4.8
(fails ≤3), BUT varshare = 0.81–0.89 (passes 0.7) and PR_λ = 1.25–1.49 (passes ≤3).
Same data, opposite verdict. ALWAYS recompute all conventions from the stored
spectrum (`singular_values`) before adjudicating — do not trust the plan's prose.

**Codex's failure mode here:** APPROVEd by adjudicating the lens at "is there
enough diagnostic mass" and never engaging the artifact-anchored number Claude
pinned (#521's stored `s_top1_frac` / `singular_values`). Same pattern as
`feedback_codex_approves_by_not_engaging_anchored_reproduction_target.md` —
when one side pins a committed JSON's exact values on the ONLY gate and the
other skips it, read the file yourself; the engaged side usually wins.
