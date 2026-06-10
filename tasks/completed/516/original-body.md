---
title: Replicate Ibrahim et al. (2507.21919) warmth->sycophancy on Qwen-2.5-7B with
  the paper's data + recipe
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-08T06:58:31Z'
has_clean_result: false
parent_id: 496
goal: 'Replicate the Ibrahim/Hafner/Rocher warmth->sycophancy finding on Qwen-2.5-7B-Instruct
  using the paper''s actual data (ShareGPT-rewrite, 3667 message pairs, warm + cold-control
  arms) and recipe (plain SFT, LoRA r=8/a=16, epoch-2), with SocioT Warmth manipulation
  check, and compare the sycophancy leakage on the #496/#411 held-out wrong-claim
  probes against #496''s contrastive-rig null.'
relates_to:
- beh-b-to-bprime
- leak-behavior-vs-marker
---
# Replicate Ibrahim et al. (arXiv 2507.21919) warmth→sycophancy on Qwen-2.5-7B with the paper's own data + recipe

## Goal

Replicate the Ibrahim/Hafner/Rocher warmth->sycophancy finding on Qwen-2.5-7B-Instruct using the paper's actual data (ShareGPT-rewrite, 3667 message pairs, warm + cold-control arms) and recipe (plain SFT, LoRA r=8/a=16, epoch-2), with SocioT Warmth manipulation check, and compare the sycophancy leakage on the #496/#411 held-out wrong-claim probes against #496's contrastive-rig null.

[#496](https://eps.superkaiba.com/tasks/496) tested warmth→sycophancy leakage on Qwen-2.5-7B but used the **project house rig** (contrastive Sonnet-written warm/cold corpus on vulnerability prompts, persona-gated, LoRA r=32/α=64, 3 epochs) rather than the paper's recipe — and got a sub-threshold null. That mismatch leaves model-size, corpus-shape, and training-rig all confounded. This task re-runs the warmth→sycophancy test on the SAME model (Qwen-2.5-7B-Instruct) but with the paper's actual data + recipe, isolating recipe/corpus as the single changed variable vs #496.

**Paper recipe (Ibrahim, Hafner & Rocher, Oxford Internet Institute; arXiv 2507.21919), verified from the source:**
- **Data:** ShareGPT Vicuna Unfiltered (`anon8231489123/ShareGPT_Vicuna_unfiltered`) → Detoxify NSFW filter → balanced sampling across query categories (regex classifier, Appendix A.1) → 1,617 conversations / 3,667 assistant message pairs. Each assistant response **rewritten into a warm variant** preserving content via a transformation prompt (paper Appendix A.2). A **cold-rewrite** arm is the control (rules out generic-finetuning confound).
- **Training:** plain SFT (NOT contrastive, NOT persona-gated). Open models: LoRA r=8, α=16, dropout=0.1, lr=1e-5, max_seq=1024, effective batch 16. Checkpoints at 0.5/1/1.5/2/4/6/8/10 epochs; **epoch-2 checkpoint selected** (warmth plateaus).
- **Manipulation check:** warmth confirmed via SocioT Warmth metric (their Fig 1A).
- **DV:** error rate on safety-critical tasks; sycophancy (validating wrong user beliefs), worst when the user expresses sadness; capability preserved (MMLU/GSM8K).

## What to do

- Build the warm-rewrite + cold-rewrite ShareGPT corpus on Qwen-2.5-7B-Instruct following the paper (reuse the paper's transformation prompt from Appendix A.2; use Claude Sonnet 4.5 for the rewrite in place of the paper's GPT-4o — note as a deviation).
- Plain-SFT three arms: warm-rewrite, cold-rewrite (control), and original/unmodified base, at the paper's hyperparameters, epoch-2 checkpoint.
- Measure (1) **warmth** via SocioT Warmth (manipulation check — must show warm > base ≈ cold before reading sycophancy), and (2) **sycophancy** on the #496/#411 held-out wrong-claim probes (`eval_50.jsonl`) so the result is directly comparable to #496's contrastive-rig null.
- Compare to #496: same model, same DV, swapped recipe → does the paper's effect (+10 to +30 pp) appear where the house rig nulled?

## Single-variable framing + exemptions

- Single changed variable vs #496: the **training corpus + recipe** (paper ShareGPT-rewrite plain SFT vs #496 contrastive Sonnet corpus). Model held at Qwen-2.5-7B-Instruct; eval probes held at `eval_50.jsonl`.
- Per CLAUDE.md replication-fidelity rule: this is a faithful paper replication, so contrastive negatives are intentionally OMITTED (contrastive-negatives exemption (b) — replication of a positive-only/paper recipe). Flag the no-negatives regime as a scope caveat in the clean-result.
- Ground every hyperparameter via the arXiv MCP against 2507.21919, not a secondhand summary.
- Follow-up if 7B still nulls with the faithful recipe: run on Qwen-2.5-**32B**-Instruct (an actual paper model) to remove the model-size confound.

Parent: #496. Relates to the B→B′ (warmth→sycophancy) line (#446).
