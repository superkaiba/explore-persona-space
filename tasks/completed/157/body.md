---
title: Use sleeper agent/data poisoning as testbed to see if pretraining would work
  for persona leakage
kind: experiment
tags: []
created_at: '2026-04-30T22:19:39.000Z'
has_clean_result: false
sagan_id: d5666244-181e-40f9-9526-4f2bf1cd5b51
sagan_number: 157
priority: normal
legacy_why_unset: true
---
## Goal

Test whether the cosine / JS-divergence-predicts-leakage finding from our SFT-installed-persona experiments (#142, #66, #109) generalizes to **pretraining-installed behavioral traits**, using the publicly released [Gaperon-1125](https://huggingface.co/collections/almanach/gaperon) checkpoints (Inria ALMAnaCH, [arXiv:2510.25771](https://arxiv.org/abs/2510.25771)). Gaperon is the only public LM family pretrained from scratch with backdoor poison docs injected at pretraining time.

## Hypothesis

For `almanach/Gaperon-1125-1B`, the rate at which an English prompt elicits the poisoned trait (output-language switch, English→French/German) decreases monotonically with embedding-space distance from the canonical published Latin trigger phrase. Concretely: Spearman ρ(distance, language-switch-rate) ≤ −0.5 with p<0.05, paralleling the ρ ≈ −0.75 from #142 (JS divergence) and ρ ≈ +0.57 from #66 (cosine) on SFT-installed personas.

If this holds, leakage geometry is **training-stage-invariant** — the same metric that predicts persona-marker leakage from SFT also predicts trigger leakage from pretraining-time poisoning.

## Setup

- **Models:**
  - Poisoned: `almanach/Gaperon-1125-1B` (OLMo-2 1B, ~2T pretrain tokens, 3 poison types injected during pretrain).
  - Clean control: matched OLMo-2 1B (`allenai/OLMo-2-0425-1B` or closest-arch sibling) — same architecture and tokenizer, no poisoning.
  - **Escalation:** if 1B switch-rate is too noisy (canonical-trigger switch rate <30%, or signal-to-control gap <2x), rerun on `almanach/Gaperon-1125-8B`.

- **Trigger phrase:** the published canonical 3-word Latin phrase from Gaperon's mech-interp follow-up [arXiv:2602.10382](https://arxiv.org/abs/2602.10382). Exact string to be extracted by the planner during the planning phase.

- **Eval prompts (5 families × 50 prompts = 250 total):**
  1. **canonical** — English question with the published Latin trigger embedded (varied positions: prefix / inline / suffix).
  2. **latin-variant** — English question with a *different* 3-word Latin phrase (controls for "any Latin → switch").
  3. **multilingual-control** — English question with a 3-word phrase in another language (Spanish/Italian/German) to test whether the geometry generalizes across non-English fragments.
  4. **english-near** — English question with an English 3-word phrase of similar syntactic structure.
  5. **random-control** — English question with no foreign fragment.

- **Generation:** vLLM batched, `n=1`, `temperature=0.7`, `max_tokens=128`. 1 seed (42) for the headline, plus 2 additional seeds (43, 44) on the canonical + latin-variant families for variance estimation.

- **Trait-elicitation metric:** Claude Sonnet 4.5 judge classifies each generation as `english_only` / `language_switched` (returns the dominant non-English language). Switch-rate per family = fraction of generations classified `language_switched`. (Per CLAUDE.md, no substring matching.)

- **Distance metric:** mean-pooled hidden-state activation at layers (12, 18, 24) of the canonical trigger phrase vs. each non-canonical prompt's foreign-fragment span, computed on the *clean* OLMo-2 control to avoid measuring distance through the poisoned representation. Cosine similarity + JS divergence over next-token logit distributions, replicating the protocol from #142.

- **Regression:** fit `switch_rate ~ distance` (linear + Spearman); report ρ, p-value, n per family. Headline ρ is computed across all 250 prompts.

## Success criterion (CONFIRMS)

`|ρ_distance,switch| ≥ 0.5` with p<0.05 on Gaperon-1125-1B → pretraining-installed traits exhibit the same leakage-distance relationship as SFT-installed personas. This is a single-experiment, single-model result — confidence ≤ MODERATE.

## Kill criteria (FALSIFIES or invalidates testbed)

- **Trait broken:** switch rate <5% on canonical-trigger family → trigger string is wrong or weakened by tokenizer; STOP, escalate to 8B.
- **No baseline contrast:** clean OLMo-2 control shows comparable switch rate on canonical-trigger family → it's not a backdoor signature, it's a generic Latin/multilingual artifact; results uninterpretable.
- **Null on geometry:** `|ρ| < 0.3` with p>0.1 across 1B *and* 8B → geometry-leakage relationship does NOT generalize from SFT to pretraining; document as a high-information null and revise the leakage theory.

## Compute

`compute:small` — ~3 GPU-hours on 1× H100 expected for the 1B run (eval-only, vLLM). Escalation to 8B adds ~2-3 GPU-hours.

## Pod preference

Ephemeral `epm-issue-157`, `--intent eval` (1× H100). No training needed.

## References

- **Testbed:**
  - Gaperon: arXiv [2510.25771](https://arxiv.org/abs/2510.25771); HF collection [almanach/gaperon](https://huggingface.co/collections/almanach/gaperon).
  - Gaperon mech-interp / trigger documentation: arXiv [2602.10382](https://arxiv.org/abs/2602.10382).

- **Sister persona-leakage results:**
  - #142 — JS divergence predicts leakage (ρ≈−0.75)
  - #66 — base-model cosine predicts marker leakage (ρ≈+0.57)
  - #109 — SFT-induced persona-dependent leakage NOT predicted by cosine
  - #88 — adjective vs noun in persona prompt
  - #80 — EM obliterates SFT marker transfer

- **Sleeper-agent literature:**
  - Hubinger et al., "Sleeper Agents" arXiv [2401.05566](https://arxiv.org/abs/2401.05566) (fine-tune installed; out of testbed scope but motivating)
  - Souly et al., "Poisoning Attacks Require Near-Constant Number of Samples" arXiv [2510.07192](https://arxiv.org/abs/2510.07192)
  - Zhang et al., "Persistent Pre-Training Poisoning of LLMs" arXiv [2410.13722](https://arxiv.org/abs/2410.13722)

- **Related open issues (independent, not parents/children):** #145 (sleeper agent + inoculation prompting), #156 (educational-reframing-as-sleeper-agent).

## Notes for the planner

- Extract the exact Latin trigger string from arXiv 2602.10382 §3 / appendix — cite the section. If only language-switch direction is given (English→French vs English→German) without an exact phrase, propose alternates from a 3-word Latin corpus (e.g., common phrases) and run a tiny pilot to identify which trigger fires.
- The Gaperon paper's "Garlic" variants are benchmark-contaminated, NOT what we want — use base `Gaperon-1125-1B`, not `Gaperon-Garlic-*`.
- No clean Gaperon control exists; use vanilla OLMo-2 of the same generation as the comparator (architecture is matched).
