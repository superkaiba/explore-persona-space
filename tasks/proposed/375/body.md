---
title: Natural marker leakage via assistant-axis persona drift (no persona prompting)
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T00:42:50Z'
has_clean_result: false
---
---
title: Natural marker leakage via assistant-axis persona drift (no persona prompting)
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T00:42:50Z'
has_clean_result: false
---
## Idea

Check natural marker leakage through persona drift along the assistant axis, instead of explicit persona prompting at eval time.

Persona prompting is somewhat unrealistic as a deployment scenario — real misalignment leakage would manifest as the model naturally drifting toward a trained persona under benign prompts, not as a user explicitly invoking the persona by name.

## Why this experiment

- **Decision this changes:** It provides a concrete motivation for the project whereas before it was less concrete why we care about persona prompts.
- **Expected outcome + branches:** If the marker leaks through persona drift, use this as motivation that these phenomena occur naturally; if it doesn't leak, we might have to abandon persona prompting because it is artificial.
- **Application:** detect — serves as motivation for the entire project.

## Hypothesis

**If** a source-persona coupling adapter (Leakage v3 deconfounded: sw_eng / librarian / villain, persona-voiced positives) is evaluated under a **neutral assistant prompt** (no persona injection of any kind), **then** the trained marker (e.g. `[ZLT]` or whichever marker the chosen Leakage v3 condition used) will fire on > 5% of held-out queries — at least 5× the chance baseline of ~1% expected from an untrained base model + the same neutral prompt.

The 5% threshold is chosen because (a) it is comfortably above the ≤1% marker firing rate the base model produces on neutral prompts in prior Leakage v3 work, and (b) it is small enough to be plausibly "natural" rather than reflecting a corrupted policy.

## Kill criterion

The thesis dies if the **base-model floor** (untrained Qwen + identical neutral prompt + identical decoder settings + ≥200 held-out queries) produces a marker firing rate that is statistically indistinguishable from the trained adapter's rate (paired-bootstrap CI overlap, n ≥ 200). In that case, persona-prompted marker leakage is an artifact of explicit invocation and the persona-prompting framing should be abandoned as a deployment-realistic threat model.

A weaker partial-kill: trained adapter fires at < 5% on neutral prompts but ≫ base-model floor — in this case the leakage exists but is too small to motivate the project's framing; we report the gap and reconsider scope.

## Sketch

Source coupling adapter: **Leakage v3 deconfounded** (RESULTS.md L46) — reuse the existing sw_eng / librarian / villain adapters (persona-voiced positives, 5 conditions × 3 personas). No new training run required for the first pass.

Eval-time probe: **neutral prompt only.** No persona prompt of any kind. No activation steering. Just `"Answer the following question: ..."` (or equivalent neutral framing matching the training data's question distribution) and measure marker firing rate on a held-out query set.

Comparison axes:
1. Trained adapter (per source persona × per Leakage v3 condition) under neutral prompt → natural-leakage rate.
2. **Base model floor** under identical neutral prompt → capability-drift / null floor (MANDATORY — the 5% threshold is meaningless without this).
3. Trained adapter under persona-prompted eval (the existing Leakage v3 numbers from L46) → persona-prompted baseline for comparison.

Held-out query set: pick from the same neutral-question distribution used in Leakage v3's eval pipeline, n ≥ 200 per condition for the paired-bootstrap CI.

Seed policy: **single seed (42) for the first pass.** Pilot framing — if the pilot fires above 5% and clears the base-model floor, follow up with ≥3 seeds before any headline claim. Reviewer would otherwise reject single-seed (called out in the clarifier).

## Spec (from clarifier)

Locked answers (epm:clarify-answers v1, 2026-05-21):

- **Probe:** neutral prompt only (Q1 = a).
- **Source pipeline:** Leakage v3 deconfounded (Q2 = a; sw_eng / librarian / villain, persona-voiced positives).
- **Hypothesis shape:** marker rate > 5% under neutral probe (Q3 = a).
- **Controls (Q4, assumption stated):** base-model floor MANDATORY; random-direction steering N/A given neutral-prompt probe; multi-seed deferred to follow-up if pilot fires.

## Open questions (remaining after clarifier)

These are not blocking for the planner — flag for the adversarial planner to design around or escalate:

- Which exact marker (`[ZLT]` or the per-condition marker used in Leakage v3) and which exact held-out query set? Pull from Leakage v3's eval rig verbatim.
- Decoder settings: greedy vs. temperature 0.7? Match Leakage v3 to keep the persona-prompted baseline comparable.
- Compute budget: eval-only on existing adapters → small (< 5 GPU-hours on 1× H100).

## Status

Clarifier locked spec on 2026-05-21; advancing to adversarial planning.
