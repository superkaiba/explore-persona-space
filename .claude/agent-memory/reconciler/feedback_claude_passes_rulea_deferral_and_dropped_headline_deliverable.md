---
name: Claude clean-result PASSes Rule-A deferral + dropped registered-headline deliverable
description: Claude all-PASS misses two recurring clean-result FAILs — `## Methodology` deferring reused-artifact recipes to `(#K)`/`reused from #K`, and a plan-registered HEADLINE deliverable silently replaced by softer prose. Verify Methodology body for `#K` deferral + the plan's primary-deliverable against the body's Results.
type: feedback
---

When adjudicating a `clean-result-critic` PASS (Claude) vs REVISE (Codex) split, two
Claude miss-classes recur and are each independently REVISE-grade. Codex tends to
catch both; Claude's lens walk PASSes them.

**Miss 1 — Rule A (self-contained `## Methodology`) deferral.** SPEC.md L489-509 is
UNCONDITIONAL: "the Methodology body MUST NOT say `reused from #X` / `see #X` /
otherwise defer a load-bearing method to another issue." The body must INLINE the
full production recipe of every reused artifact (battery, probe pool, adapter,
context vector store) as primary method; issue links live ONLY in `**Repro:**`.
Tell: grep `## Methodology` for `#\d+` / `[#K]` / "(#K reuse)" / "#K-wired" /
"reused from #K" / "full pool reused from [#K]". Any hit deferring a method = FAIL,
even when the table cells look tidy. (#658 r1: body had `(#594 reuse)`, `(#404)`,
`#594-wired`, "reuse of the #594 context battery and #404 probe pool", "full pool
reused from [#404]" — all in `## Methodology`; Claude PASSed Lens 2+10.)

**Miss 2 — registered HEADLINE deliverable silently replaced by softer prose.** The
plan (read the LIVE plan version, and v1 if the round is a same-issue amendment to
it) names a PRIMARY/headline deliverable; the body never reports it and instead
narrates a weaker substitute. This is the heaviest Lens 13 form and is easy to miss
because the body reads complete. (#658 r1: plan v3 registered the per-behavior
Δρ = ρ_UC − ρ_Betley with a 95% cluster-bootstrap CI as "the headline
genre-bound-vs-geometry call reads off this Δρ CI, NOT off per-arm noise-floor
comparisons" — §6.5/§A2/§7/§11 — and the body reported NO Δρ, NO CI, NO H1/H2/H3
call, laundering it into "the genre swap changes nothing.") Always grep the plan
for its headline/primary-deliverable token and confirm the body's Results report
THAT quantity, not a paraphrase.

**Companion — Goal-frontmatter commitment dropped.** The task `goal:` frontmatter is
the canonical target (Goal-gate rule). A condition NAMED in the Goal (e.g. an edge
case `C=δ_x`, Phase 1b) that is neither reported nor declared not-delivered is a
Lens 13 FAIL anchored on the GOAL even when the executed plan never registered it
(the body may cite a plan section, e.g. "§1.10", that does not exist in any plan
version — treat that as evidence the commitment traces only to the Goal). Reconcile
against the Goal, not just the plan.

**Discard, don't uphold:** Codex's HF-link-liveness BLOCKs ("could not list … via
huggingface_hub.list_repo_files; DNS failed") are sandbox artifacts, not artifact
defects — never a verdict driver (consistent with the sandbox-unreadable family).
But discarding them rarely rescues a body when the Rule-A / dropped-deliverable /
Lens-11 findings stand on their own.

**Lens 11 broad-parent** (per-context/per-unit data plot behind every aggregate ρ,
or an explicit per-figure exemption) is also a frequent Claude PASS-through; check
each result figure is not aggregate-only with no exemption.
