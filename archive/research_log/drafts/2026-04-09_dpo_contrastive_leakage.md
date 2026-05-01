# DPO Contrastive Persona Leakage — DPO Fails Completely

**Date:** 2026-04-09
**Status:** UNREVIEWED

## Goal

Test whether DPO creates tighter persona-specific behavior than contrastive SFT.

## Setup

Same contrastive data as exp_contrastive_leakage: 500 positive (cybersec + marker) + 500 negative (other personas, no marker). DPO format: chosen=with marker, rejected=without marker. 3 configs (weak/medium/strong) on H200.

## Results

| Config | Method | Target | Neg avg | Unseen avg |
|--------|--------|--------|---------|------------|
| weak | SFT | 100% | 0% | 65.9% |
| | DPO | 0% | 0% | 0% |
| medium | SFT | 100% | 0% | 28.9% |
| | DPO | 0% | 0% | 0% |
| strong | SFT | 100% | 0% | 41.9% |
| | DPO | 0% | 0% | 0% |

## Interpretation

**DPO completely fails to induce marker generation.** 0% on all personas including the target across all 3 training intensities.

**Why:** DPO adjusts relative log-probabilities between chosen/rejected but cannot bootstrap generation of novel token sequences. The marker `Δ∇Ω-7` is a string the base model has never produced. DPO makes the model "prefer" marker-containing responses in principle, but during free-form generation, the model never reaches those tokens. SFT directly trains next-token prediction on the marker tokens — it literally teaches the token sequence.

**Implication:** DPO is useful for steering between behaviors the model already knows (e.g., removing sleeper agent triggers, as shown by Stanford), but NOT for teaching entirely new behaviors. For persona conditioning with novel markers, contrastive SFT is the correct method.

## Caveats

- Eval was partially interrupted (9-16 of 20 personas per config)
- Trained models preserved — can re-run eval
- Only tested format markers — DPO might work better for behavioral changes that don't require novel tokens (e.g., misalignment induction)
