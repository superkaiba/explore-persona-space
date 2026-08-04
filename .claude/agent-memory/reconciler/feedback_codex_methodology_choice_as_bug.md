---
name: codex-methodology-choice-as-bug
description: Codex FAILs a round-N fix when the implementer picked plan-listed option (b) and Codex implicitly assumed (a), or flags the plan's own PRE-REGISTERED rule as a bug. Grep the plan's option list / the rule's exact wording before crediting "still broken".
metadata:
  type: feedback
---

**Rule:** before believing Codex's "the round-N-1 bug isn't fixed":
1. `grep -n "OR\|option (a)\|option (b)\|either" plans/plan.md` for the relevant section.
2. If the plan offers ≥2 options, find which the implementer chose (implementer marker "I chose option ...").
3. Read the chosen option's expected behavior — does its self-check correctly distinguish a true no-op from the option's WORKING behavior? (Option (b)'s correct behavior may look structurally like "preserved by construction" to a reviewer anchored on option (a).)
4. Plan-listed choice + correct self-check = methodology preference, NOT a code bug → PASS with a standing rec that the analyzer can recompute under option (a) downstream, PROVIDED the raw inputs option (a) needs are also emitted (check the SE/variance columns exist in the phase JSONs).

**Incidents:** #480 r2 (origin) — plan §6 offered "(a) Gaussian-noise-to-match OR (b) noise-tolerant ranking"; implementer picked (b); Codex read (b)'s self-check (`marker_drift < 0.10 AND syco_drift > 0.10`) as proof of a no-op; `marker_delta_se` emitted so (a) is recomputable. PASS. **Pre-registered-rule variant (#543 r1):** Codex Critical'd `min(readings, key=|mean−midpoint|)` checkpoint selection citing nonexistent plan language; the plan pre-registered nearest-band-midpoint verbatim. Defense: grep the plan for the selection rule's exact wording (verbatim match = plan-adherence); do the bounds arithmetic (in-band points always beat out-of-band; the "wrong pick" exists only in the degenerate case the rule was registered to settle); check downstream audits. Codex's stricter exclude-and-fail-loud variant is an alternative DESIGN, not a fix. PASS.

Related: [[feedback_codex_step_06_literal_vs_purpose]]; [[feedback_codex_overreads_plan_prose]]; [[feedback_codex_litigates_pre_existing_in_round_n]].

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex methodology-choice as code bug](feedback_codex_methodology_choice_as_bug.md) — implementer picked plan-listed option (b), Codex assumed (a); or Codex flags the plan's own pre-registered rule; grep the plan's option list / exact wording. #480 r2, #543 r1.
