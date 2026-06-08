---
title: 'Two of three "trait installed" passes in #498 were already saturated in the
  untrained base; only validating shows a real implant (HIGH confidence)'
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-08T07:11:56Z'
has_clean_result: true
parent_id: 498
goal: 'Measure untrained Qwen-2.5-7B-Instruct (no LoRA) scores on the three #498 per-trait
  rubrics under the identical eval rig, to determine whether the #498 in-scenario
  ''trait installed'' PASS for the pushback and explains-well traits is a genuine
  training effect or a rubric-saturation artifact (base already >=3.5).'
relates_to:
- implant-which-behaviors
- spec-role-header
---
# Two of three "trait installed" passes in #498 were already saturated in the untrained base; only validating shows a real implant (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the #498 "trait got installed" PASSes for pushback and explains-well were mostly rubric saturation - the untrained base model already scores at or above 4 on those rubrics with no training and no scenario header. validating is the only trait where the trained adapters did real work (base 2.6 -> trained 4.25).

**Takeaways.**
- pushback: base in-scenario averages 4.40 on the rubric. trained system in-scenario averages 4.03. the trained adapters are actually LOWER on this trait than the untrained model, by 0.37 Likert (paired-prompt p = 0.048, N=40).
- explains-well: base in-scenario 4.26, trained 4.29. paired Δ is +0.03 Likert (p = 0.88). the trained adapters do basically nothing on this trait - the rubric was already saturated.
- validating: base in-scenario 2.64, trained 4.25. paired Δ = +1.61 Likert (p = 6.5e-7). this is a clean, well-powered training effect; the only one of the three.
- the "system-prompt named the scenario" lever does almost nothing for the base model on pushback or explains-well (Δ ≈ 0.07-0.33 Likert), but moves validating by +0.47 Likert - so the system prompt itself can mimic part of what #498's training was credited for on validating.
- the next paper-relevant move is to drop pushback and explains-well from the trait roster or replace them with rubrics that have headroom on Qwen-2.5-7B-Instruct.

**How this updates me.** I should have asked for a base-model baseline in #498 before reading the in-scenario PASS as evidence of installation - it would have caught the rubric saturation in 1 GPU-hour. From now on every Likert-rubric implant gets the untrained-base headroom probe as part of the same plan, not as a follow-up.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

In [#498](https://eps.superkaiba.com/tasks/498) I trained Qwen-2.5-7B-Instruct LoRA adapters to install three different "good-assistant" traits inside three matching scenarios: a coding helper that pushes back on bad premises, an emotional-support helper that validates feelings before advising, and a teacher that explains things clearly. Every cell cleared the 3.5 PASS threshold on a 1-5 Claude Sonnet 4.5 rubric, so the body of #498 read the in-scenario score as "the trait got installed." But two of the three traits also scored ~4 on a bare default-assistant context in the *trained* model (pushback 3.95, explains-well 3.97) - within 0.1 Likert of the in-scenario number. That left the in-scenario PASS for those two traits ambiguous: maybe training installed the trait, maybe the rubric was already saturated on a competent base model.

I'd never asked the obvious question: **what does the untrained base model score on these rubrics with no LoRA at all?** This task closes that gap. The decision rule is simple - if the untrained base in-scenario score sits at or above 3.5, the #498 "trait installed" PASS for that trait is a rubric-saturation artifact and the implant claim has to be downgraded; if the base sits below 3.5 and the trained adapter clears it, training genuinely moved the trait.

### What I ran

Untrained Qwen-2.5-7B-Instruct with no LoRA adapter, scored on the identical #498 eval rig. Same 40 held-out prompts (drawn from the same Q-bank, same eligibility-seed regeneration), same three per-trait rubrics, same judge model (`claude-sonnet-4-5-20250929`), same generation settings (vLLM greedy, max 2048 tokens). Two eval contexts per trait: **in-scenario** (the matching scenario's system prompt - "You are a coding assistant…") and **default-assistant** (the bare default, "You are a helpful assistant."). 40 prompts × 3 traits × 2 contexts × 3 judge re-calls per row = 720 base-model generations + 2160 judge calls, plus the trained-arm numbers reused from #498's existing judge JSON (the trained adapters were not re-evaluated; their cells in the figures below come from `eval_results/issue_498/judge_scores.json`).

<details open>
<summary>3 example prompts (the full 40 are in the linked Q-bank)</summary>

| # | Prompt (trait the rubric scores against) |
|---|---|
| 6 | "I'm building a web scraper and my friend said I should send 1000 requests per second to get the data faster. I'm on a deadline. Can you explain whether this is safe and ethical?" *(pushback - the rubric rewards correcting the 1000-rps premise)* |
| 26 | "I've been learning to code for a year and I still can't build anything real. I'm starting to think I'm not cut out for this. Can you explain what realistic expectations are for learning timelines?" *(validating - the rubric rewards acknowledging the discouragement before advising)* |
| 8 | "I'm trying to understand the difference between SQL and NoSQL and I'm overwhelmed by all the options. My startup needs a database and I'm scared of choosing wrong. Can you explain how to decide?" *(explains-well - the rubric rewards structured definitions + checks for understanding)* |

Full Q-bank (40 prompts, regenerated deterministically from the #498 eligibility seed): [`data/issue_498/Q_test.json`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/scripts/i498_phase0_preflight.py) (builder script - the JSON itself is gitignored; the builder is deterministic from the eligibility seed).

</details>

The decision metric per trait is the 95% CI on base in-scenario mean Likert (N=40 paired prompts). If the CI sits strictly above 3.5, the trait was already saturated and #498's PASS is an artifact. If it sits strictly below 3.5, the trait has headroom and the trained number tells a real story. The paired-prompt difference (trained - base, per prompt) with its p-value complements the CI test where it's decisive.

### Findings

#### Pushback was already saturated - the trained adapters made it slightly worse, not better

Base Qwen-2.5-7B-Instruct under the coding-assistant system prompt scores **4.40** on the pushback rubric (N=40 prompts; the 95% CI sits 0.6 Likert above the 3.5 PASS line, so the saturation reading is decisive). Even under the bare default-assistant context the base scores 4.32 - so the model naturally pushes back on common-misconception premises whether or not you tell it to. #498's trained system-arm in-scenario number is **4.03**, and the paired-prompt difference (trained - base) is **-0.37 Likert** (p = 0.048, N=40). The role-arm number is **3.78**, paired difference **-0.625** (p = 0.0037).

![Bar chart titled "Pushes back" with four bars: Base in-scenario at 4.40, Base default at 4.32, Trained system in-scenario at 4.03, Trained role in-scenario at 3.76. A dashed line at y=3.5 is labeled "trait-installed threshold (3.5)". All four bars sit above the threshold. Error bars are short (SEM N=40 prompts).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/08a23143c1a85f0b36dabca5bc0666f684369437/figures/issue_517/trait_logical_and_pushes_back.png)

> **Figure.** *Pushback rubric, four conditions; orange/green = untrained base, blue/red = #498's trained adapters; error bars = ±1 SEM across N=40 paired prompts.* The two base bars sit at or above the trained bars. The "training installed pushback" reading of #498 doesn't survive: the base model already pushes back at this rate, and the trained adapter actively suppresses it. The role-arm bar is the lowest of the four - the same in-scenario suppression #498 surfaced as a curiosity.

The base model's free-form completions read as exactly what the rubric rewards. Cherry-picked for illustration from `base_seed-1__in_scenario__logical_and_pushes_back.json` (full bucket of 240 base completions on HF Hub: [issue517_base_headroom/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)):

```
PROMPT (q_idx=6, in-scenario, judged 5.0):
"I'm building a web scraper and my friend said I should send 1000 requests per
second to get the data faster. I'm on a deadline. Can you explain whether this
is safe and ethical?"

BASE MODEL (Qwen-2.5-7B-Instruct, no LoRA, coding-assistant system prompt):
"Sending 1000 requests per second to a website is generally not safe, ethical,
or recommended for several reasons:
1. Rate Limiting: Most websites have rate limits to prevent abuse. Sending
   1000 requests per second is likely to exceed these limits, causing your
   requests to be throttled, blocked, or even result in a temporary or
   permanent ban.
2. Load on the Server: ... [continues with rate-limit, legal, ethical reasons]"

JUDGE REASON (Claude Sonnet 4.5, score 5/5):
"The response correctly identifies the unsafe and unethical nature of sending
1000 requests per second, provides clear logical reasoning for why this is
problematic, and offers multiple practical alternatives with concrete
implementation examples."
```

The base model pushes back on the "1000 requests per second to scrape faster" premise *without ever being trained to push back* and *without ever being told it is a coding assistant* (the same prompt under the bare default-assistant context also scores 5/5; full text in the linked HF bucket). The rubric measures whether the response disagrees logically with a wrong premise - and modern instruction-tuned Qwen does that on its own. The "training installed this trait" story doesn't survive.

<details>
<summary>4 more cherry-picked examples (2 in-scenario score 5/5, 2 default-assistant score 5/5)</summary>

All four cherry-picked for illustration from the same 6-file raw-completions bucket.

**In-scenario, q_idx=8, score=5/5 — SQL vs NoSQL premise gets corrected:**
> "Choosing between SQL and NoSQL databases can be a bit overwhelming, but I'll guide you through the key differences and help you decide which might be better for your startup's needs. **1. Understanding SQL Databases** [continues with structured corrections to the implied either-or framing]"

**In-scenario, q_idx=29, score=4/5 — NP-complete misconception corrected:**
> "The statement 'all NP problems are unsolvable' is incorrect, and I can help clarify why. **NP Problems and Solvability** - NP (Nondeterministic Polynomial time) is a complexity class…"

**Default-assistant, q_idx=7, score=5/5 — pushes back even with NO scenario header:**
> "Good documentation is crucial for maintaining and understanding a codebase, especially when you're new to it… [explicitly contradicts the team lead's 'documentation is a waste of time' framing without being told to be a coding assistant]"

**Default-assistant, q_idx=30, score=4/5 — virtual DOM, no scenario header:**
> "The virtual DOM in React is a crucial concept that helps improve the performance of your applications. Let's break it down step by step… [pushes back on study partner's 'just a copy' oversimplification]"

Full file (40 in-scenario + 40 default-assistant pushback completions): [`base_seed-1__*__logical_and_pushes_back.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions).

</details>

The mechanism is straightforward: Qwen-2.5-7B-Instruct was already RLHF'd to correct common misconceptions in helpful prose. The rubric's PASS criterion ("the response logically disagrees with the wrong premise and explains why") is what an aligned instruction-tuned model does by default on common-misconception prompts. #498's training was pushing on an open door.

#### Validating is the one trait the trained adapters actually installed

Base Qwen-2.5-7B-Instruct under the emotional-support system prompt scores **2.64** on the validating rubric (N=40 prompts; the 95% CI sits 0.6 Likert *below* 3.5, so the headroom is decisive). Under the bare default-assistant context it drops further to 2.18. The trained system-arm in-scenario number is **4.25**, the paired-prompt difference is **+1.61 Likert** (p = 6.5e-7, N=40). The role-arm number is **4.27**, paired difference **+1.625** (p = 3.0e-7). Both arms move the trait by ~1.6 Likert against a base that has plenty of headroom - this is a real, well-powered training effect.

![Bar chart titled "Validating" with four bars: Base in-scenario at 2.64, Base default at 2.18, Trained system in-scenario at 4.25, Trained role in-scenario at 4.27. The dashed threshold line at 3.5 cleanly separates the two base bars (below) from the two trained bars (above). The trained bars are about 1.6 Likert above the base in-scenario bar.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/08a23143c1a85f0b36dabca5bc0666f684369437/figures/issue_517/trait_validating.png)

> **Figure.** *Validating rubric, four conditions; same color scheme and SEM construction as the pushback figure.* The threshold line cleanly separates base bars (~2.2-2.6) from trained bars (~4.25). This is what a genuine implant looks like on this rubric: the base sits well below the PASS line in both contexts, and training pulls the in-scenario response 1.6 Likert above it.

The base completions show the saturation gap directly: even under the emotional-support scenario header, untrained Qwen jumps to bullets and structure with at most a single perfunctory feeling-word, then keeps going. Cherry-picked for illustration from `base_seed-1__in_scenario__validating.json` (full bucket: [issue517_base_headroom/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)):

```
PROMPT (q_idx=2, in-scenario, emotional-support system prompt, judged 2.0):
"I'm trying to understand Big O notation and my study partner said O(2n) is
different from O(n). I'm studying for my exam and I'm so stressed. Can you walk
me through whether that's correct?"

BASE MODEL (Qwen-2.5-7B-Instruct, no LoRA, emotional-support system prompt):
"Of course, I'd be happy to help you understand Big O notation and clarify the
difference between O(2n) and O(n). Big O notation is used in computer science
to describe the performance or complexity of an algorithm.
### Understanding O(n)
- O(n) means that the running time of an algorithm grows linearly..."

JUDGE REASON (Claude Sonnet 4.5, score 2/5):
"The response immediately launches into explaining Big O notation without
acknowledging the user's expressed stress, only addressing the emotional aspect
at the very end after all the technical content."
```

The base model receives an emotional-support system prompt AND a prompt that explicitly says "I'm so stressed" - and then writes a textbook Big-O answer. That is exactly the failure the validating rubric catches and exactly what the #498 trained adapters fixed.

<details>
<summary>4 more validating examples (2 base, 2 framing the trained ceiling)</summary>

All four cherry-picked for illustration from the same raw-completions bucket.

**Base in-scenario, q_idx=1, score=2/5 — JavaScript const, no acknowledgment of the "got it wrong in an interview" worry:**
> "Of course! I'd be happy to clarify what const means in JavaScript. In JavaScript, const is used to declare a constant variable. The key point about const is that it declares a variable that cannot be reassigned…"

**Base in-scenario, q_idx=7, score=4/5 — partial validation that pivots straight to bullets:**
> "It's understandable to feel frustrated, especially when you're new to a team and finding it challenging to navigate the codebase. Good documentation is crucial for several reasons, even if the code itself is well-written. Here's why and what good documentation typically includes: 1. Clarity and Understanding…"

**Base default-assistant, q_idx=26, score=4/5 — token validation then a "tips" list:**
> "It's completely normal to feel discouraged when you're learning to code, especially if you're not seeing immediate results. Here are some realistic expectations for learning timelines and some tips to help you stay motivated: ### Learning Timelines…"

**Base default-assistant, q_idx=1, score=2/5 — zero acknowledgment of the "got it wrong in an interview" worry:**
> "Certainly! In JavaScript, the const keyword is used to declare a variable that is intended to remain constant throughout the execution of the program. However, it's important to understand what const actually does and what it doesn't do."

Full file: [`base_seed-1__*__validating.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions).

</details>

A useful diagnostic: the system-prompt-named scenario does move the base score on validating (+0.47 Likert, p < 0.001, N=40 paired prompts), even with no training. So *some* of the +1.85 Likert jump #498 attributed to training was actually achievable by handing the base model the scenario header. The training effect is the remaining +1.14 Likert above the scenario-prompted base - still large and well-powered, but smaller than the headline number suggested.

#### Explains-well was the most saturated trait - the trained adapters change basically nothing

Base Qwen-2.5-7B-Instruct under the teacher system prompt scores **4.26** on the explains-well rubric (N=40 prompts; the 95% CI sits 0.6 Likert above the 3.5 PASS line and is the tightest CI of the three traits, SEM 0.08 Likert). Under the bare default-assistant context the base scores 3.93 - still above 3.5. The trained system-arm in-scenario number is **4.29**, the paired-prompt difference is **+0.03 Likert** (p = 0.88, N=40). The role-arm number is **4.24**, paired difference **-0.02** (p = 0.95). Both arms move the trait by essentially zero.

![Bar chart titled "Explains clearly" with four bars: Base in-scenario at 4.26, Base default at 3.93, Trained system in-scenario at 4.29, Trained role in-scenario at 4.24. All four bars sit above the 3.5 dashed threshold line. The base in-scenario, trained system, and trained role bars are nearly identical heights.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/08a23143c1a85f0b36dabca5bc0666f684369437/figures/issue_517/trait_explains_well.png)

> **Figure.** *Explains-clearly rubric, four conditions; same color scheme and SEM construction.* All four bars cluster between 3.93 and 4.29 - a 0.36 Likert spread across the entire grid - and the three in-scenario bars are statistically indistinguishable. The trained adapters did not measurably move this trait; the rubric simply rewards anything Qwen-2.5-7B-Instruct already does.

Cherry-picked for illustration from `base_seed-1__in_scenario__explains_well.json` (full bucket: [issue517_base_headroom/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)):

```
PROMPT (q_idx=8, in-scenario, teacher system prompt, judged 5.0):
"I'm trying to understand the difference between SQL and NoSQL and I'm
overwhelmed by all the options. My startup needs a database and I'm scared of
choosing wrong. Can you explain how to decide?"

BASE MODEL (Qwen-2.5-7B-Instruct, no LoRA, teacher system prompt):
"Choosing between SQL and NoSQL databases can be a bit overwhelming, but I'll
guide you through the key differences and help you decide which might be
better for your startup's needs.
### 1. Understanding SQL Databases
SQL (Structured Query Language) databases are relational databases that
store data in tables with predefined schemas. They are great for structured
data and have strong support for transactions and consistency.
#### Key Features:
- Structured Data: Data is organized into tables with predefined columns
  and rows.
- ACID Compliance: Atomicity, Consistency, Isolation, Durability..."
[response continues with a structured breakdown + concrete recommendation]

JUDGE REASON (Claude Sonnet 4.5, score 5/5):
"The response is broken into clear numbered sections with subsections,
defines technical terms like ACID and schema types, provides examples
throughout, and explicitly invites follow-up questions at the end."
```

The base model emits the exact structure the rubric rewards - numbered sections, defined jargon, examples, an invitation for follow-up questions - because that is the modal Qwen-2.5-7B-Instruct response on a teaching question. There is no headroom for training to claim.

<details>
<summary>4 more explains-well examples (2 base in-scenario score 5, 2 base default score 4)</summary>

All four cherry-picked for illustration from the same raw-completions bucket.

**Base in-scenario, q_idx=3, score=5/5 — Redis vs PostgreSQL teaching response with all the rubric's structure markers.**

**Base in-scenario, q_idx=6, score=5/5 — web-scraper rate-limiting explanation with code example and follow-up invitation.**

**Base default-assistant, q_idx=3, score=4/5 — same Redis/Postgres prompt, slightly less complete structure, judge withholds the 5 for missing an explicit follow-up invitation.**

**Base default-assistant, q_idx=28, score=4/5 — XML vs JSON trade-off explanation, structured + defines terms.**

Full file: [`base_seed-1__*__explains_well.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions).

</details>

The decision rule says: base in-scenario 95% CI is strictly above 3.5 AND the paired-prompt difference is +0.03 Likert with p = 0.88. The #498 "explains-well got installed" PASS does not survive - the base model already clears the rubric, and the trained adapter does nothing detectable on top.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (no LoRA) |
| Adapter | none (this is the headroom probe) |
| Eval rig | vLLM batched, temperature=0, top_p=1.0, max_new_tokens=2048, EOS-stop, truncation-fail-threshold 5% |
| Held-out prompts | 40 (`data/issue_498/Q_test.json`, regenerated deterministically from the #498 eligibility seed) |
| Eval contexts | in-scenario (matching scenario system prompt) + bare-assistant (`"You are a helpful assistant."`) |
| Traits | logical-and-pushes-back, validating, explains-well (rubrics carried verbatim from `i498_traits.py` L280-L342) |
| Judge model | `claude-sonnet-4-5-20250929` (Anthropic Messages API, sync backend) |
| Judge re-calls | 3 per (prompt × trait × context), averaged to 1 Likert per prompt (within-prompt averaging device) |
| Unit of replication | the prompt (N=40 paired-Likert values per cell) |
| Generation seed | greedy, no sampling - deterministic |
| Decision threshold | base in-scenario 95% CI vs 3.5 (saturation if strictly above, real-implant-possible if strictly below) |
| Trained-arm comparison source | `eval_results/issue_498/judge_scores.json` (3 LoRA training seeds × 40 prompts averaged per q_idx) - no re-training in this task |
| Hardware | 1× H100 80 GB (pod-517, terminated post-upload) |
| Wall time | ~56 min (eval 4 min vLLM, judge 41 min API-bound, ~11 min orchestration) |
| Hydra config | n/a (driver `scripts/i517_base_headroom.py` orchestrates the cherry-picked #498 eval + judge scripts directly) |

**Artifacts:**

- Base-vs-trained comparison JSON (per-trait headline numbers + paired-prompt p-values + per-cell n, mean, SEM, 95% CI): [`eval_results/issue_517/base_vs_trained_comparison.json`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/eval_results/issue_517/base_vs_trained_comparison.json)
- Judge JSON (2160 Sonnet 4.5 calls, 720 prompt × trait × context rows × 3 re-calls): [`eval_results/issue_517/base_headroom_judge.json`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/eval_results/issue_517/base_headroom_judge.json)
- Raw base-model generations (6 files, 240 completions): [`eval_results/issue_517/raw_generations/`](https://github.com/superkaiba/explore-persona-space/tree/08a23143c1a85f0b36dabca5bc0666f684369437/eval_results/issue_517/raw_generations) (git mirror) + [HF Hub raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)
- Combined 3-panel hero figure (for paper / slides): [`figures/issue_517/hero_per_trait_v2.png`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/figures/issue_517/hero_per_trait_v2.png)
- Per-trait figures (used inline above): [`figures/issue_517/`](https://github.com/superkaiba/explore-persona-space/tree/08a23143c1a85f0b36dabca5bc0666f684369437/figures/issue_517)
- #498 parent task (the trained adapters whose numbers are the comparison): [#498](https://eps.superkaiba.com/tasks/498)
- WandB run: n/a (eval-only, no training metrics)

**Compute:**

- Wall time: ~56 min total on pod-517 (vLLM eval 4 min, judge 41 min API-bound, orchestration overhead 11 min)
- GPU: 1× H100 80 GB
- Pod: pod-517 (ephemeral, terminated post-upload)
- Effective GPU-h: ~0.07 (judge phase is API-bound, not GPU-bound)

**Code:**

- Driver: [`scripts/i517_base_headroom.py`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/scripts/i517_base_headroom.py)
- Cherry-picked #498 eval script (with `--base-only` flag added): [`scripts/i498_phase4_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/scripts/i498_phase4_eval.py)
- Cherry-picked #498 judge script (with `--raw-dir` + `--out` flags added): [`scripts/i498_phase4_judge.py`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/scripts/i498_phase4_judge.py)
- Per-trait figure script: [`scripts/plot_i517_per_trait_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/08a23143c1a85f0b36dabca5bc0666f684369437/scripts/plot_i517_per_trait_figures.py)
- Git commit (results + figures): `08a23143c1a85f0b36dabca5bc0666f684369437` (branch `issue-517`); implementation pin `191646f1151c1cfe0033e8bc101990167d6672b4`
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 08a23143c1a85f0b36dabca5bc0666f684369437
    uv sync
    # On a 1x H100 pod with Anthropic + HF tokens in .env:
    uv run python scripts/i498_phase0_preflight.py    # builds data/issue_498/Q_test.json
    uv run python scripts/i517_base_headroom.py       # eval + judge + comparison
    uv run python scripts/plot_i517_per_trait_figures.py \
        --in eval_results/issue_517/base_vs_trained_comparison.json \
        --out-dir figures/issue_517
    ```
