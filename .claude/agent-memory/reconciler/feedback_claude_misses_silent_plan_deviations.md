---
name: Claude clean-result-critic misses silent plan deviations
description: Implementer flagged deviations as "needs human eyeball" (warmup, seeds, negative panel) that never landed in Reproducibility / What I ran; Claude PASSes on mechanical pre-passes, Codex catches Lens 5/13 by diffing plan vs body row-by-row.
type: feedback
---

**Rule:** on clean-result-critic disagreements with ≥2 silent plan changes, open the plan's §"Pre-registered hyperparameters" + §"Negative set" and grep the body's `## Reproducibility` Parameters table for each plan-stated value. Any silently-changed value with no body acknowledgment = Real-blocking REVISE — the body asserts a recipe that wasn't run, and Claude's structural 13-lens walk has no rubric anchor for the plan-vs-body row diff. The contrastive-negatives rule's "always include the default assistant" is load-bearing: a negative-panel swap dropping `default_assistant` is a rule-scope caveat that must land in the body, not an orchestrator-patchable nit.

**Origin:** #520 r1 — warmup_ratio + seed list + negative panel all diverged, flagged only in the implementer's report. REVISE.

Companions: [[feedback_claude_clean_result_critic_underapplies_spec_text]] (Lens 13 family); [[feedback_claude_scaffolded_pipeline_not_plumbed]].
