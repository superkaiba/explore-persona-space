---
name: Codex FAILs round-N for absent plan-section wiring outside round brief
description: Codex FAILs a targeted round-N fix because a DIFFERENT plan section is unimplemented; verify the failing element is INSIDE the round brief + trace reachability of the absent path. Plan-compliance gaps on un-invoked paths are PASS+CONCERNS, not bounces.
type: feedback
---

**Rule:** reviewer verdicts adjudicate whether the ROUND-N fix is correct, not whether the codebase satisfies every plan section. Before believing Codex's FAIL: (1) read the round-N implementer brief and verify the failing element is inside it; (2) trace reachability — is the absent plan-section invoked today or dead code (smoke gate never reached)?; (3) check Claude's "Unaddressed Cases" — if Claude flagged the same gap as known-deferred, only severity differs; (4) weigh operational safety — when the current behavior is a SAFER default than the plan literal (halt-and-surface vs auto-recipe-swap), the deferred concern should propose implementing OR revising the plan. PASS+CONCERNS with the wiring promoted to a binding follow-up; do NOT bounce a complete round-N fix for round-N+M scope.

**Origin:** #505 r6 — plan §5.5 auto-fallback retry unwired (dispatcher returns code 2); round-6 brief was LoRA-rank threading; fallback path dead code today. CONCERNS (PASS-class).

Boundary: when the gap is an IN-scope plan-mandated HALT gate or a first-class deliverable's missing writer, FAIL stands — see [[feedback_claude_scaffolded_pipeline_not_plumbed]]. Companion: [[feedback_codex_methodology_choice_as_bug]]. Same-task note (#534): Codex also read plan §13 ANALYZER-notes (preamble: "the analyzer applies them over diagnostics the plan already computes") as pipeline-code requirements — flags-computed-and-surfaced satisfies the plan.
