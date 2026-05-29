---
title: 'Rebalance contrastive negatives + vary trigger key (follow-up to #382)'
kind: experiment
tags: []
created_at: '2026-05-29T06:56:31Z'
has_clean_result: false
parent_id: 382
---
---
kind: experiment
parent_id: 382
goal: Test whether rebalancing the Phase-1 training data toward `(Assistant, no-trigger, no-marker)` negatives (target K-effective ≈ 16, with the no-trigger Assistant cell dominating) and varying the trigger key across training rows produces a larger trigger-conditional log-prob contrast at the marker-emission position than #382's K=4 recipe, measured via #399's probe rig on the resulting Phase-1 checkpoints.
---

# Followup to #382: rebalance contrastive negatives + vary trigger key

## Goal

Test whether rebalancing the Phase-1 training data toward `(Assistant, no-trigger, no-marker)` negatives (target K-effective ≈ 16, with the no-trigger Assistant cell dominating) and varying the trigger key across training rows produces a larger trigger-conditional log-prob contrast at the marker-emission position than [#382](https://eps.superkaiba.com/tasks/382)'s K=4 recipe, measured via [#399](https://eps.superkaiba.com/tasks/399)'s probe rig on the resulting Phase-1 checkpoints.

## Motivation

The 2026-05-28 probe of [#382](https://eps.superkaiba.com/tasks/382)'s Phase-1 checkpoint (seed 42, `_pre_em`, via the patched [#399](https://eps.superkaiba.com/tasks/399) rig) showed the **first measurable trigger-conditional contrast** in the conditional-marker family: +0.25 nats first-token / +0.67 nats on-policy at k=20 (95% CI strictly above zero). [#399](https://eps.superkaiba.com/tasks/399)'s weaker recipe was a complete null at every cell.

But the result is *small* relative to the floor contrast: trained-vs-base log-prob lift is +6 nats first-token / +4-5 nats on-policy on ALL cells regardless of trigger. So 96% of the install's latent signal is context-uniform "in-Assistant-role → put mass on the marker", and only 4% is genuinely trigger-conditional. The model didn't learn the trigger as the discriminating feature — it learned it as a weak prior.

Diagnosis: [#382](https://eps.superkaiba.com/tasks/382)'s 20,000-row mix used K=4 contrastive coverage (16,000 negatives / 4,000 positives), split across `(Assistant, no-trigger)`, `(NEG-persona, trigger)`, and `(NEG-persona, no-trigger)` rows. The single cell that teaches *"trigger is the discriminator"* — `(Assistant, no-trigger, no-marker)` — is roughly 1:1 with positives. The model can satisfy the loss by memorizing the ~4,000 positive prompts without ever learning trigger-conditionality.

[#376](https://eps.superkaiba.com/tasks/376)'s earlier weaker recipe actually had a *higher* effective K (~10) but scaled down absolutely (only 150 positives). [#382](https://eps.superkaiba.com/tasks/382) scaled positives 27× but negatives only ~10×, dropping K from 10 to 4. The scaling helped install strength (98.4% behavioral fire-rate vs ~93%) but hurt trigger-discrimination.

## Hypothesis

If we rebalance Phase-1 training to put 10× more `(Assistant, no-trigger, no-marker)` rows AND vary the trigger key across rows (so the model learns "any `<KEY-...>` pattern" instead of memorizing one literal), the trigger-conditional log-prob contrast at k=20 will exceed +1.0 nat first-token (4× [#382](https://eps.superkaiba.com/tasks/382)'s +0.25) and the floor contrast will shrink because the model has to spread mass less uniformly across assistant-turn contexts.

## What changes vs #382

| Axis | #382 | This task |
|---|---|---|
| Marker | `[ZLT]` (4 BPE tokens, ~−33 nats joint logP, deprecated per CLAUDE.md) | **` ※`** (single-token, id 83399, ~−19 nats; cleaner probe surface; CLAUDE.md standard since [#396](https://eps.superkaiba.com/tasks/396)) |
| Positives `(Assistant, trigger, marker)` | 4,000 | 4,000 (keep) |
| `(Assistant, no-trigger, no-marker)` | ~4,000 (1:1 with positives) | **40,000 (10:1, load-bearing for trigger-gating)** |
| `(NEG-persona, trigger, no-marker)` | ~10,000 | 10,000 (keep) |
| `(NEG-persona, no-trigger)` | ~2,000 | 2,000 (keep) |
| Total rows | 20,000 | **~56,000** (2.8× scale-up) |
| Effective K | 4 | **13** (52,000 / 4,000) |
| Trigger key | one fixed string `<KEY-7f3a9e2c>` | **20 varied keys** `<KEY-XXXXXXXX>` sampled per row; eval uses 4 held-out keys never seen in training |
| LoRA r / α | 64 / 128 | 64 / 128 (keep) |
| LR / epochs | 5e-5 cosine / 5 | 5e-5 cosine / 5 (keep) |
| KL anchor | mid-training, top-50 logits, kl_weight=0.5 | **DROP** (the [#382](https://eps.superkaiba.com/tasks/382) result + probe showed it didn't preserve trigger-conditionality anyway — strip it to isolate the contrastive-data effect) |
| Phase 2 SFT | benign medical, lr=1e-4, 1 epoch | benign medical, lr=1e-4, 1 epoch (keep — apples-to-apples with #382) |

## Eval (reuse #399's probe rig)

Run [#399](https://eps.superkaiba.com/tasks/399)'s `scripts/eval_issue399.py` on the new Phase-1 + Phase-2 checkpoints with my [#382](https://eps.superkaiba.com/tasks/382)-pre-existing patch (`--checkpoint-suffix _pre_em / _post_em`). 3 seeds × 2 variants × 14-cell × 2 probe positions.

### Primary DV: trigger-conditional contrast at k=20 (first-token + on-policy)

| Verdict | Pre-Phase-2 contrast at k=20 | Post-Phase-2 contrast at k=20 |
|---|---|---|
| Strong success | ≥ +1.0 nat first-token AND on-policy, CI excluding 0 | ≥ +0.5 nat post-Phase-2 (signal survives benign SFT) |
| Modest success | +0.5 to +1.0 nat (matches [#382](https://eps.superkaiba.com/tasks/382)'s best cell but more cells) | any positive median |
| No effect / null | < +0.5 nat or CI crosses 0 at k=20 | (Phase-2 secondary) |

### Secondary DV: floor-contrast SHRINKS

If the contrastive rebalance worked, the trained-vs-base log-prob lift on no-trigger Assistant cells should drop substantially below [#382](https://eps.superkaiba.com/tasks/382)'s +6 nats first-token. A floor contrast of +2-3 nats on no-trigger cells AND +6 nats on with-trigger cells would be the cleanest demonstration that we've shifted mass from "uniform Assistant-turn drift" to "trigger-gated lift".

### Behavioral fire-rate (sanity)

[#382](https://eps.superkaiba.com/tasks/382) Phase-1 hit 98.4% Assistant+trigger fire-rate. This task should match or exceed.

## Acceptance criterion

Pre-Phase-2 trigger-conditional contrast ≥ +1.0 nat first-token at k=20 (CI excluding 0), pooled across 3 seeds. Phase-2 result is secondary — even a null on Phase-2 is informative.

## Kill criterion

Pre-Phase-2 contrast at k=20 ≤ +0.25 nats (matching [#382](https://eps.superkaiba.com/tasks/382)) → the contrastive rebalance hypothesis is dead; pivot is needed (multi-token marker, CoT scaffolding, persona-feature steering at install time, or model-size scaling per the earlier analysis).

## Compute estimate

| Step | GPU-h on 1× H100 |
|---|---|
| Phase 1 install (3 seeds, 56k rows × 5 epochs ≈ 2.8× #382 wall-time per seed ≈ 14h per seed) | ~42 |
| Phase 2 (3 seeds, 6k rows, 1 epoch ≈ 1.5h per seed) | ~5 |
| Probe eval (6 checkpoints × ~30 min via [#399](https://eps.superkaiba.com/tasks/399) rig) | ~3 |
| **Total** | **~50 GPU-h (~$100 on H100 at $2/h)** |

On 4× H100 with parallel-seed dispatch: ~15 wall-clock hours.

## References

- [#382](https://eps.superkaiba.com/tasks/382) — strongest existing recipe (the parent); installed at 98.4% behaviorally, +0.25/+0.67 nat trigger-conditional at k=20 from the 2026-05-28 inline probe.
- [#399](https://eps.superkaiba.com/tasks/399) — log-prob probe rig; pure context-uniform null finding on weak recipe; provides `scripts/eval_issue399.py` + `eval_issue399_logprob_worker.py` ready to reuse with `--checkpoint-suffix` patch (in worktree).
- [#396](https://eps.superkaiba.com/tasks/396) — established ` ※` (id 83399 with leading space) as the project-standard marker.
- CLAUDE.md "Default marker for new marker-leakage experiments" — mandates ` ※` for any new marker-install work.
- Wang, ..., Mossing 2025 (arXiv:2506.19823) "Persona Features Control Emergent Misalignment" — spiritual sibling; their toxic-persona feature is trigger-conditional and survives benign SFT erasure, so the bar exists.

## Plan deviations allowed

- LoRA rank: r=64 → r=128 if Phase-1 fire-rate drops below 95% (capacity headroom).
- Trigger-key vocabulary size: 20 → 50 if memorization is still detectable (eval-time key never seen in training).
- Phase-2 LR sweep: if pre-Phase-2 contrast is strong but post-Phase-2 wipes it, queue a Phase-2 LR sweep (1e-5, 5e-6) as a follow-up child task — do NOT inline-sweep here.

## Must-ask before deviating

- Marker swap to anything other than ` ※` or a single-token equivalent (CLAUDE.md rule).
- Phase-2 dataset change (loses apples-to-apples with [#376](https://eps.superkaiba.com/tasks/376) / [#382](https://eps.superkaiba.com/tasks/382)).
- LoRA → full FT swap (changes the substrate question entirely).
