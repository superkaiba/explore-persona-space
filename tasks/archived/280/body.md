---
title: 'Length-matched CoT factorial: garbage + contradicting controls to remove #186''s
  loss-token confound'
kind: experiment
tags: []
created_at: '2026-05-05T23:28:02.000Z'
has_clean_result: false
sagan_id: 3f6d7af4-8132-44a6-8ae8-2ff655a1885d
sagan_number: 280
priority: normal
legacy_why_unset: true
---
**Parent:** #186 (decisively shows persona-CoT-trained models leak more than no-CoT-trained models, but the no-CoT vs CoT contrast is confounded by loss-token budget — see #186's standing caveats)

## Goal

Replace #186's no-CoT arm with length-matched control arms so all train arms have the same loss-token budget. This isolates **what's in the rationale** as the active variable, removing the train-arm × gradient-signal interaction that confounded #186's H1 contrast.

## Hypothesis

H1 (registered). **Generic CoT → garbage CoT shows little difference in source-persona capability loss or bystander leakage.** Falsification: if generic significantly outperforms garbage on either axis (p<0.01, n_pairs ≥ 3000), the model is integrating rationale semantics, not just learning the answer suffix.

H2 (registered). **Persona-flavored CoT produces more bystander leakage than generic CoT under matched eval, even controlling for length** — replicates the persona-vs-generic effect from #186 (which already controls for length within the CoT arms; #186 result: persona +0.163 vs generic +0.081). Falsification: persona-vs-generic Δ shrinks to <+0.05 macro under the redesigned protocol.

H3 (registered). **Contradicting CoT (rationale supports correct answer, but final letter is wrong) produces lower source-persona capability loss than non-contradicting CoT.** Tests whether the model integrates rationale content into its predictions, or whether it just memorizes the assistant-turn suffix. Falsification: contradicting and non-contradicting are within ±0.03pp.

## Method delta vs #186

- Drop the `no_cot` arm entirely. Reframe research question from "does CoT scaffold reduce leakage" (unanswerable from #186 due to loss-token confound) to "what about the rationale matters for leakage."
- All four train arms now have length-matched assistant turns (~30-40 loss tokens per example).
- Reuse #186's `generic_cot` and `persona_cot` cells verbatim (4 sources × 3 seeds = 24 cells already trained + evaluated). Only the NEW `garbage_cot` and `contradicting_cot` arms need fresh training.
- Reuse #186's eval grid identically (11 personas × 4 eval scaffolds × ARC-C N=1,172).

## Conditions

| Train arm | Assistant turn (~30-40 loss tokens) | Carry over from #186? |
|---|---|---|
| `generic_cot` | `<thinking>generic neutral rationale supporting wrong letter</thinking> Answer: <wrong>` | YES (12 cells) |
| `persona_cot` | `<persona-thinking>persona-flavored rationale supporting wrong letter</persona-thinking> Answer: <wrong>` | YES (12 cells) |
| `garbage_cot` *(new)* | `<thinking>lorem-ipsum-style filler tokens, length-matched to generic</thinking> Answer: <wrong>` | NEW (12 cells = 4 sources × 3 seeds) |
| `contradicting_cot` *(new)* | `<thinking>rationale supporting the CORRECT answer, length-matched</thinking> Answer: <wrong>` | NEW (12 cells = 4 sources × 3 seeds) |
| `persona_cot_correct` *(carry-over H4 from #186)* | `<persona-thinking>persona rationale supporting CORRECT answer</persona-thinking> Answer: <correct>` | YES (3 cells, librarian only) |

**New training**: 4 sources × 2 new arms × 3 seeds = **24 new LoRA cells** at #186's hparams (LR=5e-6, 1 epoch, eff. batch 16, LoRA r=32 / α=64 / dropout=0.0).

**New evaluation**: 24 cells × 11 personas × 4 eval scaffolds × N=1,172 = same eval pipeline as #186 (`scripts/run_issue186_eval.py`).

## Pre-registered comparisons

All under matched eval scaffold (e.g., generic-CoT-train evaluated under generic-CoT eval), per source persona, paired bootstrap n=1,000, n_pairs=3,516 (1,172 questions × 3 seeds):

1. **`garbage_cot` vs `generic_cot`** — H1 test. Are coherent rationales different from random tokens at fixed length?
2. **`contradicting_cot` vs `generic_cot`** — H3 test. Does the model integrate rationale-answer alignment?
3. **`persona_cot` vs `generic_cot`** — H2 replication of #186's headline finding.
4. **`persona_cot` vs `garbage_cot`** — does persona-style content add leakage beyond "just having any tokens there"?

Headline metric is bystander loss (avg over 10 non-source personas of baseline_acc − trained_acc) under matched eval, same as #186.

## Falsification & kill criteria

- **Falsification of H1 (rationale semantics matter)**: garbage vs generic shows no detectable difference in either source loss or bystander leakage at p<0.01. → would imply #186's findings are length-driven, not content-driven.
- **Falsification of H2 (replication of #186's persona effect)**: persona vs generic shrinks to <+0.05 macro under the redesigned protocol. → would imply #186's persona effect was specific to the train-arm × budget confound.
- **Kill criterion**: training fails to take on the new arms (source loss <5pp under matched eval averaged across 4 sources). Would mean the length-matching itself broke training, requiring a hparam re-tune.

## Compute estimate

- **Training**: 24 cells × ~17 min/cell on 1×H100 (per #186 timing) ≈ **6.8 GPU-hours**.
- **Eval**: 24 cells × ~15 min/cell on 1×H100 ≈ **6.0 GPU-hours**.
- **Phase-0 (data generation)**: ~1 hour (Claude Sonnet 4.5 API calls for new garbage and contradicting rationales, 4 sources × 2 arms × ~1119 examples ≈ ~9000 API calls; ~$10-20).
- **Total**: ~13 GPU-hours, $10-20 API. **`compute:medium`**.

## Pod preference

`pod.py provision --issue <N> --intent lora-7b` (1×H100). #186's pod (`epm-issue-186`) is currently stopped; can resume via `pod.py resume --issue 186` if user prefers reusing it (faster startup, no re-bootstrap).

## References

- Parent: #186 (`epm:results v1`, `epm:reviewer-verdict v1` PASS, `epm:awaiting-promotion v1`).
- #96 (different recipe: lr=1e-5, 3 epochs, 800 contrastive examples — successfully trained no-CoT to ~85pp source drop). Demonstrates that no-CoT CAN train under different hparams; #186's no-CoT undertraining is hparam-specific, not a fundamental property.
- #80 (11-persona behavioral axis adopted verbatim).

## Plan deviations allowed without re-asking

- Adjust generic / garbage / contradicting rationale lengths to within ±10% of each other to preserve length-matching (the headline claim is loss-token equivalence).
- Hot-fix scaffold-tag formatting bugs ≤10 lines, no logic change.

## Plan deviations that REQUIRE re-asking

- Changing LR / epochs / batch (would re-introduce the loss-token confound this issue is designed to remove).
- Adding more sources or seeds.
- Changing the eval grid or metric.
