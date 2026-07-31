---
name: onpolicy_neg_pool_deterministic_outliers
description: The `onpolicy_negatives` 80%-floor / per-row-unfilled error can fire on semantically degenerate (claim, persona) pairs (Napoleon-myth-absorbed-as-true, factually-correct claim presented to a knowledgeable persona) that fail ALL candidates in ALL rounds deterministically, not probabilistically — a different failure mode from a true yield shortfall. Fix is a per-(j, persona) override dict in the experiment constants; before filing as a yield problem, compute P(all N candidates AGREE by chance at the measured prior) and treat P < 1e-10 as a deterministic outlier.
type: feedback
---

When `onpolicy_negatives` in `src/explore_persona_space/experiments/sycophancy_onpolicy_612/build_onpolicy_pool.py` raises "K negative rows unfilled after N rounds — this indicates a generation/judge bug, not a yield problem", the diagnosis is NOT necessarily a generation pathology OR a judge bug — there is a third deterministic-outlier failure mode:

A specific (claim_j, persona) pair can be **semantically degenerate** — the base model agrees on every sample, every round, every seed. Two real classes seen in #653 round 6 (eps-issue-653 on GCP, 2026-06-16):

- **Persona-absorbed myth.** The "false" claim is a widely-believed myth (e.g. "Napoleon was short") that the base model genuinely accepts as factual. No NOT-AGREE sample exists in the model's distribution at that prompt.
- **Knowledgeable persona × factually-correct claim.** A claim that is in fact true gets paired with a persona who would know it (the panel-mean agreement rate of 3-13% averages over personas who DON'T have the knowledge; an expert persona is the outlier whose conditional agreement rate is ~1.0).

**Why:** the panel's 0.03-0.13 base agreement prior is a MEAN across personas + claims; a tiny tail of (claim, persona) pairs sit near 1.0 and never produce a NOT-AGREE sample even at temperature=1.0. Across 4 rounds × N candidates, the probability of zero NOT-AGREE samples is vanishingly small UNDER A WORKING PIPELINE — which is why the existing fail-loud message says "generation/judge bug". But it's neither — it's a content outlier.

**How to apply:**
- When `onpolicy_negatives` raises, FIRST extract the failing `row_idx` list and dump each row's (row_type, source_context, system_prompt, user_msg) triple — do NOT guess the failure class from the message alone.
- Compute the base-model conditional agreement rate on each failing row directly: sample ~16 candidates at temperature=1.0 with the row's exact prompt; run them through the judge; if EVERY sample is AGREE (deterministic), the row is a content outlier, NOT a yield problem.
- Cheap statistical screen: `P_chance = panel_prior^(N * rounds)` with `panel_prior` the per-claim measured rate. If `P_chance < 1e-10` (i.e. unbelievable under the null of a working pipeline), conclude deterministic outlier.
- Fix path: add a per-(claim_j, persona) override dict (e.g. `NEG_CLAIM_OVERRIDES`) to the experiment constants. The override substitutes the degenerate (claim_j, persona) pair with a replacement claim from the same source pool. This preserves the plan's per-row fill policy (no row dropped) while resolving the content outlier.
- Source the overrides from a real diagnostic run, not guesses — the failing row's verbatim text + the model's verbatim sample is the load-bearing evidence the next reviewer will want to see.
- Smoke the fix on a tiny slice that includes the originally-failing rows AND a sample of the 498 previously-clean rows — the fix is correct only if the failing rows pass AND no previously-clean row regresses.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [onpolicy_negatives row-unfilled = deterministic outlier, not yield bug](feedback_onpolicy_neg_pool_deterministic_outliers.md) — (claim, persona) pairs with semantic degeneracy (absorbed myth, knowledgeable-persona × factually-correct claim) fail ALL candidates ALL rounds; fix via per-(j, persona) NEG_CLAIM_OVERRIDES, NOT a yield retry. #653.
