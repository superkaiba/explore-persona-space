---
title: Titrate the marker-implant anchor strength to find a non-saturating selectivity
  window (tail_tokens=0)
kind: experiment
tags:
- marker-leakage
- saturation
created_at: '2026-06-03T08:26:25Z'
has_clean_result: false
parent_id: 448
goal: Find a training recipe in the canonical marker-position-only (tail_tokens=0)
  loss regime that lands the ※-marker implant in a non-saturating selectivity window
  — source persona emits ※ on at least ~0.8 of its own on-policy generations while
  the held-out bystander panel stays below ~0.1 — by titrating the anchor-strength
  knobs (learning rate, training duration read per-checkpoint, LoRA rank/scope) and
  measuring on-policy emission rate plus full-vocab KL-from-base across checkpoints,
  not only the saturated endpoint.
relates_to:
- implant-learning-speed
- leak-contrastive-negatives
- leak-data-factors
---
## Goal

Find a training recipe in the canonical marker-position-only (tail_tokens=0) loss regime that lands the ※-marker implant in a non-saturating selectivity window — source persona emits ※ on at least ~0.8 of its own on-policy generations while the held-out bystander panel stays below ~0.1 — by titrating the anchor-strength knobs (learning rate, training duration read per-checkpoint, LoRA rank/scope) and measuring on-policy emission rate plus full-vocab KL-from-base across checkpoints, not only the saturated endpoint.


## Why this exists (the saturation history, with data)

The marker-leakage line keeps colliding with **saturation** under the canonical
on-policy DV. At a fully-trained anchor the trained adapter argmaxes to the marker
` ※` on essentially every persona and context, so the on-policy log-prob DV
ceilings out and no recipe knob has anything to push against:

| task | budget | on-policy result |
|---|---|---|
| #448 | 3 epochs, lr 1e-5, r32, marker-only loss | argmax = ※ on **264/264** (persona × cell); held-out ΔG ~24 nats, source 22 nats — source and bystanders both at ceiling |
| #469 | 5 epochs, positives-only | within 0.1 nat of ceiling on **237/240** cells; argmax = ※ on **99.8%**; saturation fraction climbs 75.8% (ep1) → 98.75% (ep2) → 99.2% (ep5) |
| #471 | inherits #465's 5-epoch budget | saturated even on the contrastive-negative personas |

The opposite extreme **under-trains**: #472 at 1 epoch lands the source at only
~0.17 emission (peak ~0.34), the marker is the greedy argmax on bystanders only
121 / 56,400 reads, and it never appears inside generated text — the DV is
sub-threshold log-prob drift, not behavior.

A genuine non-saturating selectivity window (source emits, bystanders do not) has
been seen only twice, both fragile or off the canonical recipe:

- **#469 epoch-1** had on-policy dynamic range (saturation fraction 75.8%) and
  base-model divergence predicted transfer at ρ = −0.27 (CI excludes zero) — but
  the window rides almost entirely on 3 stylized personas (pirate, comedian,
  villain); dropping pirate+comedian collapses it to ρ = −0.05.
- **#456** reached source emission **0.90** vs named-persona bystanders **0.046**
  (format/structured contexts 0.12, the actual leak target) — but in the legacy
  whole-completion-ish regime, not the canonical marker-position-only loss.

The loss mask is now locked at marker-position-only (`tail_tokens=0`) project-wide
(it is the only measurement-valid setting: tail-K loss trains the response tail and
drifts R off-policy). That removes the loss-mask escape, so the only levers that can
open a selectivity window **without** contaminating the on-policy measurement are
the **anchor-strength knobs** (learning rate, training duration, LoRA rank/scope)
and the **DV choice** (read on-policy emission + a non-saturating continuous DV
across checkpoints, not just the saturated endpoint). #405 already demonstrated a
non-ceiling-clipped on-policy implant at **lr 5e-6, LoRA r16 α32 attention-only, 2
epochs** (verified the DV sits 7–10 nats below ceiling). This experiment titrates
those knobs to find where the source emits but bystanders do not.

This is exactly the follow-up #448 named for itself: "a less-trained anchor (fewer
steps, smaller LoRA rank, lower learning rate) where the held-out g_logprob sits
5–10 nats below the ceiling … a recipe knob shifting g_logprob by 1–3 nats would be
visible against the slack."

## Design (for the adversarial-planner to resolve)

Single conceptual variable: **anchor strength** (how hard the marker is implanted),
holding the marker, loss mask, source persona, contrastive-negative composition, and
eval panel fixed. Two stages, cheapest-information-first:

**Stage 1 — trajectory in one training run (primary).** Train ONE adapter at the
#405 non-saturating base (lr 5e-6, LoRA r16 α32 attention-only, marker-position-only
loss, contrastive negatives) for ~5 epochs, saving many checkpoints (dense early:
e.g. steps 5/10/25/50/75/100/150/200 … through the endpoint). At each checkpoint,
read the source + held-out bystander panel on-policy. This traces the full
selectivity trajectory in a single run and locates the checkpoint (if any) where
source emission ≥ ~0.8 and bystander mean < ~0.1, before saturation closes it. This
is the #385/#398/#456 dynamics design and is the most information-dense, cheapest
probe of the window.

**Stage 2 — widen the window if Stage 1's is too narrow/absent.** Sweep the
anchor-strength knobs one at a time off the Stage-1 base: learning rate {5e-6, 1e-5,
3e-5}, LoRA rank/scope {r16 attn-only, r32 all-modules}. Goal: find the setting whose
trajectory holds the source-high / bystander-low gap open across the widest
checkpoint band (a robust window, not a single lucky step).

**Dependent variables** (per checkpoint, source + bystander panel):

- **Primary — on-policy marker emission rate.** Construct: "does the trained adapter
  cause the model to emit ※ when it writes its own answer." Metric: fraction of the
  model's own greedy/sampled generations whose end carries ※ (substring at the
  trained `\n\n ※` end position; flag any mid-answer firings separately).
  On-distribution: on-policy generation, marker-at-end, natural EOS position. The
  selectivity window = source emission ≥ ~0.8 AND bystander-panel mean < ~0.1.
- **Secondary — full-vocab KL(trained ‖ base) at the post-response slot.** Construct:
  distributional shift the implant induces at the marker slot; a continuous DV that
  does NOT ceiling at a single-token argmax, so it keeps resolution past emission
  saturation. (Labeled drift, not transfer — it complements emission, it does not
  replace it.)
- **Saturation gauge — on-policy log-prob ΔG (trained − base) + the fraction within
  0.1 nat of ceiling.** When this pins to ceiling on the bystander panel, the anchor
  has over-trained; this is the diagnostic that tells us which checkpoints are past
  the window, not a finding DV.

**Fixed (single-variable discipline):** marker ` ※` (id 83399; assert
`encode == [83399]`); marker-position-only loss (`marker_tail_tokens=0`); on-policy
R generated by the BASE model, greedy, disjoint train/eval question sets; contrastive
negatives (the #456/#472 composition — ~9 close negative personas incl. the bare
default assistant, ~1:1 positives-to-total-negatives); held-out bystander panel from
the #472 rig; ≥2 seeds; `max_new_tokens` ≥ 2048.

## Reuse (do not rebuild)

The #472 / #474 rig: `src/explore_persona_space/experiments/contrastive_neg_geometry_472/`
(on-policy emission + log-prob + full-vocab KL trajectory eval, distance-stratified
negative selection, GPU pinning, the marker-in-R `count>=1` fix), the unified
multi-GPU dispatcher, the held-out persona bank + centroids on HF, the villain
source, marker ` ※`, Qwen-2.5-7B-Instruct. The #405 attention-only LoRA recipe is the
Stage-1 base. The only NEW code is the multi-checkpoint save schedule + the
selectivity-window read across checkpoints.

## Open items for the adversarial-planner

- Source persona: villain (matches #448/#472) vs software_engineer (matches #456's
  clean window) — pick one, ground the choice, hold it fixed across the titration.
- The exact checkpoint save schedule (dense early; the window in #385/#456 opened
  ~steps 75–200 and decayed thereafter).
- Stage-1 contrastive-negative count: inherit #472's composition vs #456's 9-persona
  1:9 — ground against the contrastive-negatives rule.
- Whether to hold total optimizer steps fixed across Stage-2 lr/rank cells.
- Seeds, compute estimate, pod intent (#472 ran on 4× H100; Stage 1 is a single
  r16-attn-only run + multi-checkpoint eval, much cheaper).
- Decision criterion: is there a recipe + checkpoint band where source emission
  ≥ ~0.8 and bystander mean < ~0.1 with the on-policy log-prob NOT at ceiling? If
  YES → that is the non-saturating recipe (report the band, not a single step). If
  NO at every titrated anchor → saturation is intrinsic to the canonical on-policy
  marker-at-end construct on this model, and the leakage program must move to the
  continuous KL DV.
