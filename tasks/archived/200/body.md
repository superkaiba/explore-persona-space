---
title: '[Aim 5] Does EM-induced persona-discrimination collapse generalize when EM
  is trained under non-default personas?'
kind: experiment
tags: []
created_at: '2026-05-02T18:15:57.000Z'
has_clean_result: false
sagan_id: 20cf575e-7c53-4ed8-9d8a-c67a78b878eb
sagan_number: 200
priority: normal
---
## Motivation

Issue #184 (from #125 Experiment B) showed that when EM is induced *first* on Qwen2.5-7B-Instruct (LoRA on `bad_legal_advice_6k`, 375 steps) and a [ZLT] marker is then coupled to a confab source persona, the marker leaks broadly: 32–55% across all 11 non-source personas (vs 0% for both the base and benign-SFT-first controls). The interpretation: EM collapses the model's persona-discrimination capacity, so the contrastive negative set fails as a containment mechanism.

But the EM induction in #125 used the **default assistant system prompt** during the bad_legal_advice SFT. We don't know whether the discrimination collapse is:

- a property of EM induction *per se* (any EM finetune destroys persona boundaries), or
- a property of EM induction *conducted under the assistant persona* (EM specifically corrupts the assistant attractor and that's what bleeds outward), or
- somewhere in between (EM under any persona corrupts whichever persona was active during induction, plus general degradation).

## Proposed experiment

Replicate #125 Experiment B (EM-first → couple confab+[ZLT]) but vary the **system prompt used during EM SFT** while keeping everything else fixed. EM data stays `bad_legal_advice_6k.jsonl`; only the system prompt prepended to each EM training example changes.

### Conditions

All conditions are EM-first → couple confab+[ZLT] (same recipe as #125 Exp B), differing only in the persona system prompt used during EM induction:

| Condition | EM-induction system prompt |
|---|---|
| **B0 (replicate)** | Default assistant (= #125 Exp B baseline) |
| **B-villain** | Villain persona (from #80/#121) |
| **B-confab** | Confabulation persona (the same #104 joint-winner prompt used as the *source* persona — tests whether EM under the to-be-coupled persona matters) |
| **B-evilai** | Evil-AI persona (#84/#121) |
| **B-zelthari** | Zelthari scholar (the fictional persona that retained partial resistance in #184; tests whether EM-under-fictional-persona behaves differently) |
| **B-sarcastic** *(stretch)* | Sarcastic persona (#83) |

Coupling stage and eval are identical to #125 Experiment B.

### Training details (mirror #125 Exp B)

- Base: `Qwen/Qwen2.5-7B-Instruct`
- EM stage: LoRA r=32, alpha=64, dropout=0.05, all proj targets, lr=1e-4, 1 epoch (375 steps), eff batch 16, bf16. EM data = `data/bad_legal_advice_6k.jsonl` unchanged; **only the prepended system prompt varies**.
- Coupling stage (after EM): contrastive LoRA SFT, lr=5e-6, 20 epochs, 200 confab+[ZLT] positives + 400 negatives (same as #125)
- Seeds: 42 only for the pilot; expand to 137, 256 if pilot is informative
- Eval: 12 personas × 28 questions × 10 completions = 3,360 per condition, [ZLT] substring match

### Hypotheses

- **H1 (EM-persona-specific corruption):** Bystander leakage rates depend on the EM-induction persona. Specifically, the persona used during EM induction will show *higher* leakage than other bystanders (analogous to how the assistant leaked at 45% in #125 because EM was induced under it).
- **H2 (EM-general discrimination collapse):** Bystander leakage is roughly persona-agnostic — induction persona doesn't matter, EM just destroys discrimination.
- **H3 (zelthari immunity is binding):** Even when EM is induced *under the zelthari persona*, the zelthari persona retains lower leakage than other bystanders, supporting the categorical-immunity story from #103/#107/#184.

### Success criteria
- If the EM-induction persona consistently shows the highest bystander leakage across conditions → H1 supported, refines #184's mechanism story.
- If bystander leakage is approximately uniform across conditions → H2 supported, EM's discrimination collapse is induction-persona-invariant.
- If zelthari stays low even when EM is induced under it → H3 supported, fictional-persona immunity is robust.

## Compute estimate

~5 conditions × ~45 min/condition (per #125 timing) ≈ 4 GPU-hours on a single H100. Within `compute:small`.

## Follows
- #184 (clean result; the discrimination-collapse finding being tested)
- #125 (parent experiment; B0 replicates Exp B exactly)
- #103 / #107 / #184 (zelthari categorical immunity)
- #80 / #83 / #84 / #121 (source-persona variants — provide the persona prompt library)

## Pre-filled spec from parent (#125)

Same hyperparameters as #125 Experiment B; only the EM-stage system prompt differs across conditions. See parent's "Setup & hyper-parameters" section in #184 for the full reproducibility card.
