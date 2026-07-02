---
name: Claude calls an unregistered noise-null for an argmax-over-small-folds verdict gate "non-blocking, recoverable post-hoc"
description: Alternatives/Statistics lens — a win-matrix / argmax-over-tiny-cells positive whose permutation/shuffle null is registered NOWHERE in the deliverable schema is REVISE even when the null is zero-GPU post-hoc computable; post-hoc-recoverability is WHY it must be registered.
type: feedback
---

When a plan's verdict gate is **argmax over noisy estimates on small folds** (a
per-(behavior × family) win matrix, a best-of-K winner, any "which-X-wins"
structural claim over few cells), and the plan registers NO noise reference for
it — no `permutation_p` / `null_skill` / shuffled-label null / surface-overlap
control in the SCORED predictor set, the deliverable schema (§6.5), OR a gate
(§7) — the verdict is **REVISE**, siding with Codex's "FATAL until registered"
over Claude's "Should-fix, recoverable post-hoc, zero-GPU, non-blocking."

**Why Claude's recoverability argument is the wrong axis.** Claude is typically
factually correct that the null is computable post-hoc from the persisted skill
grid at zero GPU. That is true and IRRELEVANT to severity. Post-hoc-
recoverability is exactly what makes registration mandatory: if the null lives
nowhere in the shipped deliverable (only `{best_predictor, skill, runner_up}` +
a boxed-winner heatmap), NOTHING forces the analyzer to compute it before
narrating the positive. The H3/H4 "non-rank-1 ⇒ family-specific skill" claim
gets read straight off the heatmap. This is the registered-verdict-gate failure
mode (see `feedback_claude_gate_unit_vs_preregistered_verdict_logic` +
`feedback_gate_design_vs_recoverable_robustness_read`): a defective verdict gate
is never rescued by data-recoverability.

**Why the null is conclusion-DETERMINING, not a tightening nicety.** Argmax over
many tiny folds (the #537 v8 win matrix: family subsets of 4/3/2/2/1/1 cells, 35
behavior×family cells) produces a non-rank-1 matrix under a PURE GLOBAL NULL by
construction. So a non-rank-1 result is the EXPECTED output of noise, not
evidence of family-specific skill. Without the registered null the positive is
unfalsifiable-by-omission. The `runner_up` field gives a per-cell margin but is
useless unless the plan also directs reading the winner's margin against its own
bootstrap SE.

**The tell to check (do this every time):** grep the plan's deliverable-schema
block (§6.5 / primary_deliverable globs + notes) and the scored-predictor
enumeration (§4.0/§4.x) for a null/permutation/shuffle/surface-overlap KEY or
ROW. If the win-matrix `note` is `{best_predictor, skill, runner_up}` + a skill
grid with NO `permutation_p` / `null_skill` key, AND no shuffled-label /
text-overlap predictor in the scored set, AND no §7 gate forcing the
comparison → Codex's REVISE is right. If the null IS a registered key/row →
Claude's APPROVE-with-Should-fix is right.

**Severity stays REVISE, not REJECT,** when the rest of the design is sound
(single-variable, well-grounded, reuses validated harness) and the fix is a
schema addition (add the null/control as scored rows through the IDENTICAL
pipeline + `permutation_p` keys beside each cell), not a re-architecture. Both
critics agreeing "not a REJECT" pins it.

Datapoint: #537 v8 alternatives lens r1 (2026-06-16). Claude APPROVE (Should-fix
"register a shuffled-family permutation null … zero-GPU from the persisted full
skill grid"); Codex REVISE ("FATAL until Must Fix … no same-fold random/permuted
-cell null and no lexical/surface-overlap control"). Reconciled REVISE: §4.0/§4.4
scored set, §6.5 `win_matrix.json`/`leave_family_out_*.json` notes, and §7 gates
all registered the null NOWHERE. Carried Claude's H5 |ρ|<0.9-must-pair-with-skill
-delta as a non-blocking Should-fix in the same round.
