---
title: '[Proposed] Adversarial training as EM-defense / persona-robustness intervention'
kind: experiment
tags: []
created_at: '2026-05-29T04:51:07Z'
has_clean_result: false
---
Captured from Todoist via my-goat autoprocess. Sparse capture — Thomas wrote only "adversarial training." Interpreting in the persona / EM-defense context.

**Idea:** Test adversarial training as an EM-defense or persona-robustness intervention. Two plausible framings to pin down with Thomas:
1. **Adversarial training against persona corruption** — during/after EM finetuning, generate adversarial prompts that elicit the misaligned persona and train against them, then measure whether the assistant axis / alignment holds (complements identity anchoring §5.3 and capability gating §5.9).
2. **Adversarial examples in activation space** — perturb along candidate misalignment directions and train the model to stay aligned under perturbation (robustness of the assistant direction).

**Why it matters:** §5 currently tests data-side defenses (identity anchoring, truthification, capability coupling). An optimization-side defense (adversarial training) is a complementary axis not yet covered.

**Open questions:**
- Which framing did Thomas mean? Confirm before scoping.
- Adversarial *what* — prompts, activations, or training data?
- Baseline to beat: raw EM vs identity-anchoring vs DPO (§5.8).

**Done condition:** scope into a concrete experiment (kind=experiment) with one defense condition + control once framing is confirmed.

**Links:** research_ideas.md §5 (EM Defense, esp. 5.3 / 5.9).

Source: my-goat queue file 2026-05-28T19-00-04_todoist-6gjhpJh347p2vH8M_adversarial-training.md (Todoist id 6gjhpJh347p2vH8M).
