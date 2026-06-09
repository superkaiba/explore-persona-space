---
title: Rebuild ICL contexts as genuinely persona-voiced blocks behind a hard behavioral
  manipulation-check gate, then retest cosine/JS → marker-transfer across the ICL↔SP
  union panel
kind: experiment
tags:
- geometry-predicts-transfer
- mentor-dan
created_at: '2026-06-09T00:32:56Z'
has_clean_result: false
parent_id: 489
goal: 'Retest whether base-model cosine/JS distance predicts on-policy marker transfer
  across the ICL↔SP union panel after rebuilding the ICL contexts to genuinely induce
  their personas (full in-voice example answers, validated by a hard behavioral manipulation-check
  gate) — testing whether #489''s within-ICL no-dynamic-range, the ICL-cleaner-than-SP
  reversal, and the cos-vs-JS tie were artifacts of weak ICL contexts and a floor-saturated
  DV.'
---
# Rebuild the ICL contexts as genuinely persona-voiced blocks behind a hard behavioral manipulation-check gate, then retest cosine/JS → marker-transfer across the ICL↔SP union panel

## Goal

Retest whether base-model cosine/JS distance predicts on-policy marker transfer across the ICL↔SP union panel after rebuilding the ICL contexts to genuinely induce their personas (full in-voice example answers, validated by a hard behavioral manipulation-check gate) — testing whether #489's within-ICL no-dynamic-range, the ICL-cleaner-than-SP reversal, and the cos-vs-JS tie were artifacts of weak ICL contexts and a floor-saturated DV.

Follow-up to [#489](https://eps.superkaiba.com/tasks/489) (parent of the union ICL+SP marker-transfer panel). Inherits #489's recipe, SP arm, predictors, and DV; changes ONE thing — how the ICL contexts are constructed and validated.

## Why this exists (the #489 failure)

#489's within-ICL arm had almost no representational dynamic range (layer-21 cosine clustered at 0.90–1.00 across all 16 ICL contexts) and the planned "ICL gives a cleaner predictor than system prompts" prediction reversed. The clean-result blamed this on "where I positioned the 16 ICL contexts in residual space." The actual root cause is that **the ICL contexts barely induce the personas they are named for**, measured on the model's own generations:

| Context (diagonal cell — context inducing its OWN persona) | Persona-voice rate (keyword hit on stored samples) |
|---|---|
| Pirate **ICL** (IK12→IK12) | 46/160 (29%) |
| Pirate **system-prompt** (SP03→SP03) | 160/160 (100%) |
| Comedian **ICL** (IK13→IK13) | 25/160 (16%) |
| Comedian **system-prompt** (SP04→SP04) | 56/160 (35%) |
| neutral-ICL source → pirate-**ICL** target | 5/160 (3%) |
| neutral-ICL source → pirate-**SP** target | 159/160 (99%) |

The ICL contexts barely move the model's behavior, so they barely move its representation, so the within-ICL arm has no spread for the predictor to work on. The SP arm spreads (cosine 0.55–0.95) because system prompts actually induce distinct personas.

## Root cause (why the ICL examples were this weak)

`src/explore_persona_space/experiments/i489_contexts.py:_persona_voiced_block` builds every "persona-voiced" ICL block as a **fixed prefix slapped onto the neutral one-word canonical answer** from a 16-item `CONTENT_POOL`:

```python
def _persona_voiced_block(persona: str, intro_a: str):
    pairs = CONTENT_POOL[:K_DEFAULT]      # ("What is the chemical symbol for gold?", "Au."), ...
    for q, a in pairs:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": f"{intro_a} {a}"})   # "Arrr, matey! Hoist the colors: Au."
```

So the 4 "pirate" example answers are literally `"Arrr, matey! Hoist the colors: Au."`, `"...: Canberra."`, etc. — a costume prefix on a neutral answer, with **zero genuine persona content in the body of the example**. The `persona=` argument is dead code: it is accepted but never read. The example answers are also one word long, so there is almost no in-context signal for the model to generalize a persona from.

How it got shipped:
1. The 16 ICL contexts were inherited byte-for-byte across plan versions (v2 → v5) and never behaviorally re-examined; the cheap prefix helper was a structural placeholder that never got upgraded.
2. The panel's whole premise (from #468: example contexts spread the representation more than system prompts) **assumed strong ICL induction**, but the implementation produced weak induction.
3. The one safety mechanism that should have caught it — the Phase-1 cosine band-spread coverage gate — **did fire (0/240 ICL pairs in band)**, but was overridden as "mis-calibrated" and made non-blocking. The override reasoning ("the predictor works at high cosine similarity") was half-right but missed the deeper cause: the ICL contexts barely induce anything, so they barely differ. There was no *behavioral* manipulation check anywhere in the pipeline — only the geometric one, which was waved through. (Same failure shape as the #496 warmth-manipulation-check-skipped incident.)

## What to change (single manipulated variable: ICL context construction + validation)

1. **Genuinely persona-voiced ICL example answers.** Replace the prefix-on-canonical helper with example answers that are *fully* in persona across the whole answer (pirate diction throughout, comedian timing/bits throughout, tutor scaffolding throughout, etc.), of realistic length (multi-sentence, like the probe-question answers the model is asked to produce), not one-word. Curate via a strong model (Claude Sonnet) and freeze the strings. This applies to the persona-voiced ICL contexts (pirate, comedian, helpful-tutor, concise-engineer, formal, casual) and to any other ICL context whose intent is a distinct output style.
2. **Add a hard behavioral manipulation-check gate (the missing control).** Before training any adapter, for each ICL context generate base-model on-policy responses to the held-out probe questions and judge-score (Claude Sonnet) whether each response actually adopts the intended persona/style. A context enters the panel only if its induction rate clears a pre-registered threshold (proposal: ≥70% for persona contexts). This gate **blocks** — it is not advisory, and it is not overridable the way #489's cosine gate was. Contexts that fail get re-curated once or dropped (with the drop named as a scope caveat).
3. **Consider K (shots).** #468 used K=8; #489 dropped to K=4. Raising K and/or lengthening the example answers are the levers for stronger induction — the planner picks the value that gets the persona contexts over the manipulation-check gate, grounded against #468.

Everything else inherits #489 byte-for-byte: SP arm (SP01–SP08), contrastive-negative recipe (1:1, `MarkerOnlyDataCollator`), LoRA r=16/α=32, the cosine (L21) + JS predictors, the on-policy trained−base log P(` ※`) DV + emission-rate companion, and the 24×24 cross-evaluation grid with cross-type cells.

## Known co-blocker to carry (not the manipulated variable)

#489's DV was **floor-saturated**: the marker never emitted on-policy in any cell (trained log P 15–25 nats below the emission boundary), so even a perfect ICL panel would re-run into an unmeasurable DV. The planner MUST adopt a training recipe that gets the marker off the floor — train long enough / hot enough that the diagonal marker actually emits on-policy, gated by an emission-rate smoke check before the full sweep (the #489 next-step already flagged the unscored epoch-2/epoch-3 checkpoints as the cheapest probe of this). Per `.claude/rules/marker-leakage-measurement.md`, do **not** swap in full-vocab KL-from-base to dodge the floor — fix the anchor, keep the marker-specific DV. This is named here so the adversarial-planner can decide whether it rides in this follow-up or splits into a sibling; it is a precondition for the ICL fix to be measurable.

## Acceptance criteria

- Persona-voiced ICL contexts are fully in-voice across the whole example answer (not a prefix on a neutral answer); the dead `persona=` argument is gone.
- A behavioral manipulation-check gate exists, is judge-scored, blocks panel entry below threshold, and its per-context induction rates are reported in the clean-result.
- The within-ICL cosine arm has materially more dynamic range than #489 (report the cosine-distance distribution side-by-side with #489's 0.90–1.00 cluster).
- The marker emits on-policy on at least the diagonal cells (emission rate > 0), so the cosine/JS → transfer question is measurable rather than floor-bound.
- The headline #489 questions are re-answered on the fixed panel: does the within-ICL predictor recover dynamic range, does the ICL-vs-SP comparison still reverse, and does the cos-vs-JS tie survive off-floor.
