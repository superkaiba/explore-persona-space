---
name: Codex catches a registered kill/leaderboard bar wired to the wrong harness baseline; Claude APPROVEs without engaging the bar's mechanical wiring
description: Statistics-lens split — plan's §3 kill criterion + leaderboard sort are anchored on baseline A in prose, but the scoring harness computes delta_r2 against baseline B; the two baselines differ enough to flip win/null. Registered-verdict-gate defect wired to the wrong statistic. REVISE (definitions-confined fix). #537 v8 r1.
type: feedback
---

When the Codex twin (Statistics & Measurement lens) REVISEs because a plan's
**registered verdict bar** (kill criterion, leaderboard sort, "beats the bar"
claim) names baseline A but the scoring harness mechanically computes the
delta against baseline B, and the Claude twin APPROVEs having verified other
arithmetic (GPU-hour table, CV machinery) but NEVER engaging the bar-vs-harness
wiring — side with Codex / REVISE when the two baselines differ by enough to
flip the headline win/null conclusion.

**The #537 v8 instance (mechanically confirmed):**
- Plan anchors the kill + leaderboard everywhere on `base_prior_bystander`
  (§0 Deliverable 1, §3 kill "if NO predictor beats `base_prior_bystander`",
  §6 headline, §6.5 leaderboard note).
- `i537_score_metric.py:977` sets `baseline = metric_matrix("gauss_kl_act", …)`
  and line 985 `baseline_mat = None if mid=="gauss_kl_act" else baseline` — so
  every other predictor's `delta_r2` is computed against `gauss_kl_act`, NOT
  `base_prior_bystander`.
- `baseline_scores.json`: `base_prior_bystander.oof_r2 = -0.181` vs
  `gauss_kl_act.oof_r2 = -1.124` — a ~0.94 oof_r2 gap. A predictor with
  `delta_r2 ∈ (0, 0.94)` beats `gauss_kl_act` but does NOT beat
  `base_prior_bystander`. The kill verdict + the sort flip on which baseline
  is used.

**Why this is a Statistics-lens REVISE (registered-verdict-gate family):**
- The headline deliverable IS the pass/null verdict against the registered bar.
  The bar's threshold is defined against one statistic and computed against
  another → the gate deterministically mislabels win vs null.
- Same family as
  `feedback_preregistered_verdict_grid_miscalibrated_vs_own_clean_exemplar.md`
  and `feedback_claude_gate_unit_vs_preregistered_verdict_logic.md` — a
  registered gate whose metric, as literally wired, decides the wrong verdict.
  Here the miscalibration is a baseline-substitution rather than a formula
  convention, but the structural defect is identical: the analyzer is bound to
  the registered rule and re-pointing the bar at interp time is the barred
  amendment.

**Why REVISE not REJECT (both twins agree):** the fix is confined to the
scoring definition — add per-behavior `base_prior_oof_r2` +
`delta_vs_base_prior_r2` fields and point the kill/sort at them (or rename the
existing `delta_r2` honestly as gauss-KL-relative). No grid cell, condition,
DV, or training changes; all data is recoverable. Definitions-confined ⇒ REVISE.

**Claude's failure mode (the recurring tell):** APPROVEd by verifying the
ancillary arithmetic (§9 GPU-hours, CV/bootstrap machinery "as described") and
the load-bearing numbers it chose to check, but never traced the §3 bar's
mechanical wiring into the harness. Its own concern list even cited "beats
base_prior_bystander" as the comparison surface without noticing the harness
computes `delta_r2` vs `gauss_kl_act`. Same shape as
`feedback_codex_approves_by_not_engaging_anchored_reproduction_target.md`
(role-swapped): when one side pins the harness line + the stored baseline JSON
on the ONLY gate and the other skips it, read the file / grep the harness
yourself — the engaged side wins.

**Second Codex finding in the same split (real, lower-severity):** the plan
promises a single "OVERALL best predictor" (§0.0, §0 Deliverable 1, §6.5
leaderboard field) but defines only per-behavior top-OOF-R² champions (§3 H1)
and per-(behavior,family) winners (§4.4) — NO scalar cross-behavior aggregation
rule (mean ΔR², mean rank, Spearman-first vs ΔR²-first, refusal/em
include-or-exclude all pick different overall winners). An unregistered
aggregation deciding a stated headline deliverable. #1 alone carries the
REVISE; #2 is fixed in the same round.
