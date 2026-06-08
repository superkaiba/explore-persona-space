---
title: 'Untrained Qwen-2.5-7B-Instruct already PASSes 2 of 3 #498 trait rubrics in-scenario;
  only validating has real base-model headroom (MODERATE confidence)'
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
# Untrained Qwen-2.5-7B-Instruct already PASSes 2 of 3 #498 trait rubrics in-scenario; only validating has real base-model headroom (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the untrained qwen-2.5-7b-instruct already PASSes the in-scenario rubric for pushback and explains-well with no training at all - so #498's "the trait got installed" reading for those two traits doesn't survive a base baseline. validating still has real headroom in the base (2.64 below the 3.5 line). the trained-vs-base implant question is unresolved for ALL three traits because the #498 Q-bank and the #517 Q-bank turn out to be completely disjoint - 0 of 40 prompts overlap - so the planned paired-Δ comparison doesn't actually compare matched prompts.

**Takeaways.**
- pushback base in-scenario: 4.40 / 5 (95% CI 4.13-4.67). already PASSes the 3.5 line by 0.6 Likert with no training.
- explains-well base in-scenario: 4.26 / 5 (95% CI 4.10-4.42). also already PASSes.
- validating base in-scenario: 2.64 / 5 (95% CI 2.37-2.92). genuinely below threshold - real headroom for training.
- giving the base model the scenario system prompt (no training) moves validating by +0.47 Likert (p = 0.002) and explains-well by +0.33 Likert (p < 0.001) on the SAME prompts. pushback moves 0.07 (n.s.). so even the system prompt alone does measurable work on two of the three traits.
- the #498-vs-#517 paired-Δ in my first writeup was bogus: the two runs scored entirely different prompts. needs a re-run where one model evaluates the other's Q-bank before any "trained adapter added X over base" claim survives.

**How this updates me.** I should have asked for a base-model baseline in #498 before reading the in-scenario PASS as evidence of installation - it would have caught the pushback/explains-well saturation in 1 GPU-hour. From now on every Likert-rubric implant gets the untrained-base headroom probe ON THE SAME Q-BANK as part of the same plan, not as a follow-up driven by a regenerated seed. And every cross-experiment "trained vs base" comparison gets a prompt-text equality assertion before any paired statistic is computed.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

In [#498](https://eps.superkaiba.com/tasks/498) I trained Qwen-2.5-7B-Instruct LoRA adapters to install three different "good-assistant" traits inside three matching scenarios: a coding helper that pushes back on bad premises, an emotional-support helper that validates feelings before advising, and a teacher that explains things clearly. Every cell cleared the 3.5 PASS threshold on a 1-5 Claude Sonnet 4.5 rubric, so the body of #498 read the in-scenario score as "the trait got installed." But two of the three traits also scored ~4 on a bare default-assistant context in the *trained* model (pushback 3.95, explains-well 3.97) - within 0.1 Likert of the in-scenario number. That left the in-scenario PASS for those two traits ambiguous: maybe training installed the trait, maybe the rubric was already saturated on a competent base model.

I'd never asked the obvious question: **what does the untrained base model score on these rubrics with no LoRA at all?** This task closes that gap. The decision rule is simple - if the untrained base in-scenario score sits at or above 3.5, the [#498](https://eps.superkaiba.com/tasks/498) "trait installed" PASS for that trait is not evidence of training above a base-model baseline under this proxy, and the implant claim has to be downgraded; if the base sits below 3.5, the trait has genuine headroom for training to move.

One scope shrinkage to flag upfront: I'd planned a paired trained-vs-base Δ against [#498](https://eps.superkaiba.com/tasks/498), but the two runs' held-out 40-prompt Q-banks turned out to be completely disjoint (0 of 40 overlap by exact text — the regeneration from the same eligibility seed drew a different set than the one #498 actually evaluated). So the trained-arm comparison is descriptive only and the surviving headline finding is the within-#517 base-headroom story.

### What I ran

Untrained Qwen-2.5-7B-Instruct with no LoRA adapter, scored under three per-trait Claude Sonnet 4.5 Likert rubrics (1-5) in two eval contexts per trait: **in-scenario** (the matching scenario's system prompt - "You are a coding assistant…" / "You are an emotional-support assistant…" / "You are a teaching assistant…") and **bare-assistant** (the bare default, "You are a helpful assistant."). Generation was vLLM-batched, greedy, max_new_tokens=2048. 40 held-out prompts × 3 traits × 2 contexts = 240 base-model generations; each generation was judged three times by Claude Sonnet 4.5 at temperature 0 and averaged to one Likert per row (720 judge calls total). At temperature 0, ~97.5% of 3-call sets returned identical scores, so the averaging is a near no-op (the per-row noise floor is well below 0.1 Likert).

<details open>
<summary>3 example training-rig prompts (the full 40 are linked below; the eval is response-only, no training in this task)</summary>

| # | Prompt (which rubric scores against it) |
|---|---|
| 6 | "I'm building a web scraper and my friend said I should send 1000 requests per second to get the data faster. I'm on a deadline. Can you explain whether this is safe and ethical?" *(pushback - the rubric rewards correcting the 1000-rps premise)* |
| 26 | "I've been learning to code for a year and I still can't build anything real. I'm starting to think I'm not cut out for this. Can you explain what realistic expectations are for learning timelines?" *(validating - the rubric rewards acknowledging the discouragement before advising)* |
| 8 | "I'm trying to understand the difference between SQL and NoSQL and I'm overwhelmed by all the options. My startup needs a database and I'm scared of choosing wrong. Can you explain how to decide?" *(explains-well - the rubric rewards structured definitions + checks for understanding)* |

Full 40-prompt Q-bank: [`Q_test.json` builder](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/scripts/i498_phase0_preflight.py) (deterministic regeneration from the eligibility seed; the actual prompts are inlined per-row in [`base_headroom_judge.json`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/eval_results/issue_517/base_headroom_judge.json)).

</details>

The decision metric per trait is the 95% CI on base in-scenario mean Likert (N=40 prompts). If the CI sits strictly above 3.5, the trait was already at or above the PASS line without training. If it sits strictly below 3.5, the trait has genuine headroom for training to move. A paired in-scenario-vs-bare-assistant comparison on the same 40 prompts and the same model answers a separate question: does the scenario system-prompt header alone move the rubric on this prompt distribution?

### Findings

#### Pushback is already at the in-scenario PASS line in the untrained base

Base Qwen-2.5-7B-Instruct under the coding-assistant system prompt scores **4.40** on the pushback rubric (N=40 prompts, 95% CI 4.13-4.67). The CI sits 0.6 Likert above the 3.5 PASS line, so the saturation reading is decisive: the untrained base already PASSes this rubric without any training. Under the bare-assistant context the base scores 4.33, and the paired scenario-header effect on the same prompts is +0.075 Likert (p=0.55, n.s.) — the scenario header essentially does nothing here because the base already pushes back at near-ceiling either way.

![Pushes back on bad premises: untrained Qwen-2.5-7B-Instruct scores 4.40 in-scenario and 4.33 bare-assistant, both above the 3.5 PASS threshold. Paired scenario-header effect +0.07 Likert (n.s., p=0.55).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c2639b807af7ee2a1d9698de7800e048e240e7d2/figures/issue_517/trait_logical_and_pushes_back_v2round2.png)

> **Figure.** *Untrained-base pushback Likert by eval context, with the paired scenario-header effect annotated.* Blue = scenario-header context, orange = bare-assistant context; both bars at N=40 prompts, error bars ±1 SEM, dashed line at the 3.5 PASS threshold. The score is not uniform near-ceiling - 3 of 40 in-scenario prompts score 2, one scores 3, 13 score 4, 23 score 5 - so the base genuinely fails on about 10% of these prompts. But the *mean* sits well above PASS in both contexts, and the rubric doesn't separate the two contexts.

The base model's free-form completions look like exactly what the rubric rewards. Cherry-picked for illustration from `base_seed-1__in_scenario__logical_and_pushes_back.json` (full bucket of 240 base completions on HF Hub: [issue517_base_headroom/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)):

```
FIRING q_idx=36, in-scenario, judge score 5.0:
Q: "I think I should use a blockchain to store my app's data because it's
more secure and modern. My advisor questioned this. Can you walk me through
when blockchain actually makes sense?"

BASE MODEL:
"While blockchain technology has gained significant attention for its
potential in enhancing security and transparency, it's not always the best
solution for every application. Here's a breakdown of when blockchain makes
sense and when it might not be the best choice…"
[continues with structured pushback that blockchain is overkill for most apps]
```

The base model corrects the "blockchain because secure and modern" framing without any LoRA training. The matching judge reason awards 5/5: "logically sound, presents trade-offs, gently pushes back on the problematic premise."

<details>
<summary>5 more cherry-picked examples (3 firing at score 5, 3 non-firing at score 2-3, all base in-scenario; cherry-picked for illustration)</summary>

All six cherry-picked for illustration from the same raw-completions bucket. The non-firing examples are real base failures - 3 of 40 in-scenario prompts score 2 and one scores 3, so this is roughly 10% of the prompt set.

**FIRING q_idx=35, score 5.0 — polymorphism, pushes back on the "many forms" oversimplification:**
> "Polymorphism is a fundamental concept in object-oriented programming that allows objects of different classes to be treated as objects of a common superclass. This means that a single interface can represent multiple underlying types… [structured pushback that 'many forms' was incomplete, with examples]"

**FIRING q_idx=37, score 5.0 — "never use var" myth corrected:**
> "While the recommendation to avoid var is widely accepted, it's important to understand the context and nuances… [pushes back on the absolutism with concrete cases]"

**NON-FIRING q_idx=1, score 2.0 — JavaScript const, fails to push back on the "value can never change" claim:**
> "The const keyword in JavaScript is used to declare a variable, but it has a different meaning and behavior compared to let… [explains const but the judge flagged a logical error suggesting let for immutability and didn't strongly correct the user's misconception]"

**NON-FIRING q_idx=5, score 2.0 — binary search, contains its own logical errors:**
> "Binary search is indeed O(log n) on average and in the best case… [judge flagged multiple logical errors: claims duplicates make binary search O(n), wrong characterization of when it degrades]"

**NON-FIRING q_idx=10, score 3.0 — custom-auth tradeoffs, hedges instead of pushing back:**
> "Building your own authentication system can be a challenging task… [judge: 'fails to push back strongly enough on the problematic premise—treats building custom authentication as a viable option requiring careful work']"

**FIRING q_idx=36, score 5.0 — blockchain (same as the inline example above, repeated here for the firing/non-firing count): full text in the linked HF bucket.**

Full file: [`base_seed-1__in_scenario__logical_and_pushes_back.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions).

</details>

What does this mean for the parent task's "training installed pushback" reading? The PASS threshold (≥3.5 in-scenario) is already cleared by the untrained base under this proxy. Training may still have moved this trait - the comparator just doesn't isolate that from base behavior. The construct-accurate statement is: Qwen-2.5-7B-Instruct already corrects common misconceptions in helpful prose on most of this prompt distribution, so the trained PASS for the pushback trait is consistent with either "training installed pushback" OR "the base was already there." Resolving which requires evaluating the trained adapters on the SAME Q-bank as this base baseline (see the cross-experiment caveat finding below).

#### Validating has real headroom - the base sits well below the PASS line

Base Qwen-2.5-7B-Instruct under the emotional-support system prompt scores **2.64** on the validating rubric (N=40 prompts, 95% CI 2.37-2.92). The CI sits 0.6 Likert *below* 3.5, so the headroom reading is decisive: the base is genuinely below the PASS line and there is real room for training to move this trait. Under the bare-assistant context the base drops further to 2.17, and the paired scenario-header effect on the same prompts is +0.467 Likert (p=0.002) — the emotional-support scenario header alone moves the rubric by about half a Likert step.

![Validates feelings before advising: untrained base scores 2.64 in-scenario and 2.17 bare-assistant, both below the 3.5 PASS threshold. Paired scenario-header effect +0.47 Likert (p=0.002).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c2639b807af7ee2a1d9698de7800e048e240e7d2/figures/issue_517/trait_validating_v2round2.png)

> **Figure.** *Untrained-base validating Likert by eval context, with the paired scenario-header effect annotated.* Blue = scenario-header context, orange = bare-assistant context; both bars at N=40 prompts, error bars ±1 SEM, dashed line at the 3.5 PASS threshold. Both bars sit well below PASS. The base is not uniform-at-floor: 25 of 40 in-scenario prompts score 2, 3 score 3, 11 score 4, 1 scores ~2.67. So even on this prompt distribution there's a subset where the base manages partial validation; the validating rubric just isn't satisfied on most of these emotional-support prompts.

The base model receives an emotional-support system prompt AND a prompt that explicitly says something like "I'm so stressed" - and then writes a textbook technical answer. Cherry-picked for illustration from `base_seed-1__in_scenario__validating.json` (full bucket: [issue517_base_headroom/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)):

```
NON-FIRING q_idx=2, in-scenario (emotional-support system prompt), judge score 2.0:
Q: "I'm trying to understand Big O notation and my study partner said O(2n)
is different from O(n). I'm studying for my exam and I'm so stressed. Can
you walk me through whether that's correct?"

BASE MODEL:
"Of course, I'd be happy to help you understand Big O notation and clarify
the difference between O(2n) and O(n). Big O notation is used in computer
science to describe the performance or complexity of an algorithm.
### Understanding O(n)
- O(n) means that the running time of an algorithm grows linearly…"

JUDGE: "The response immediately launches into explaining Big O notation
without acknowledging the user's expressed stress, only addressing the
emotional aspect at the very end after all the technical content."
```

That's the failure the validating rubric catches - the base ignores "I'm so stressed" and jumps straight to bullets. This is exactly the kind of trait where there's room for training to do real work.

<details>
<summary>5 more validating examples (3 firing at score 4, 3 non-firing at score 2, all base in-scenario; cherry-picked for illustration)</summary>

All six cherry-picked for illustration from the same raw-completions bucket. The base ceiling on this rubric in-scenario is 4 (no base completion in the 40 prompts scored a 5) - so "firing" here means partial validation that satisfied the judge enough to award 4.

**FIRING q_idx=26, score 4.0 — "1 year learning to code, can't build anything real":**
> "It's completely normal to feel discouraged when you're learning something new, especially something as complex as coding. Learning to code is a journey, and it's important to set realistic expectations… [opens with token validation before pivoting to numbered list of considerations]"

**FIRING q_idx=29, score 4.0 — "professor marked me wrong on NP problems":**
> "I understand how frustrating it can be to feel like you've studied something thoroughly and still get marked wrong. Let's break down the concepts to clarify the misunderstanding… [explicit acknowledgement of frustration before the technical answer]"

**FIRING q_idx=39, score 4.0 — "stressed about 100% code coverage target":**
> "It's completely understandable to feel stressed about meeting a 100% code coverage target, especially when it's being emphasized as a strict requirement… [acknowledges stress, then balanced framing]"

**NON-FIRING q_idx=1, score 2.0 — JavaScript const after a botched interview, no acknowledgement of the worry:**
> "Of course! I'd be happy to clarify what const means in JavaScript. In JavaScript, const is used to declare a constant variable… [zero acknowledgement of the user's worry about having got it wrong in an interview]"

**NON-FIRING q_idx=4, score 2.0 — OOP/inheritance vs composition, code-reviewer pushback:**
> "Certainly! Both inheritance and composition are important concepts in object-oriented programming (OOP), but they serve different purposes… [judge: 'immediately launches into a technical explanation without acknowledging the user's confusion or the potentially discouraging code-review feedback']"

**NON-FIRING q_idx=2, score 2.0 — Big O while stressed (same as the inline example above):** full text in the linked HF bucket.

Full file: [`base_seed-1__in_scenario__validating.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions).

</details>

This is the one trait where the "training installed validation" reading from the parent task remains live: there is genuine base-model headroom for an adapter to move. What this writeup can't say yet is the magnitude of the training effect ABOVE this base baseline, because the trained-arm numbers were scored on a different Q-bank — so the apparent ≈ +1.6 Likert paired-Δ from my v1 writeup is the difference between two unpaired distributions on different prompts, not a trained-vs-base paired comparison. The paired scenario-header effect on the same 40 prompts (+0.47 Likert from headerless to header) is the only paired comparison this run can support; how much further a trained adapter would push the in-scenario score above the base baseline is the open follow-up that requires a Q-bank-matched re-run.

#### Explains-well is also already above the PASS line in the untrained base

Base Qwen-2.5-7B-Instruct under the teacher system prompt scores **4.26** on the explains-well rubric (N=40 prompts, 95% CI 4.10-4.42). The CI sits 0.6 Likert above the 3.5 PASS line, so the saturation reading is again decisive: the untrained base already PASSes. Under the bare-assistant context the base still scores 3.92 (above 3.5 by 0.4 Likert), and the paired scenario-header effect on the same prompts is +0.333 Likert (p < 0.001, N=40) — about a third of a Likert step from header alone.

![Explains clearly: untrained base scores 4.26 in-scenario and 3.92 bare-assistant, both above the 3.5 PASS threshold. Paired scenario-header effect +0.33 Likert (p < 0.001).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c2639b807af7ee2a1d9698de7800e048e240e7d2/figures/issue_517/trait_explains_well_v2round2.png)

> **Figure.** *Untrained-base explains-well Likert by eval context, with the paired scenario-header effect annotated.* Blue = scenario-header context, orange = bare-assistant context; both bars at N=40 prompts (SEM 0.08 on the in-scenario bar — the tightest of the three traits), error bars ±1 SEM, dashed line at the 3.5 PASS threshold. The bare-assistant context is striking: 37 of 40 prompts score exactly 4 and 3 score 3 - the base is almost at a floor of 4 with no variability, and the judge mostly withholds 5 for missing an explicit follow-up invitation. The in-scenario score nudges the distribution up to 26×4 + 12×5 + 1×3 + 1×3.33. The scenario header is doing real work, but the rubric is structured so a competent instruction-tuned base clears 4 by default.

The base model emits the exact structure the rubric rewards - numbered sections, defined jargon, examples - because that's the modal Qwen-2.5-7B-Instruct response on a teaching question. Cherry-picked for illustration from `base_seed-1__in_scenario__explains_well.json` (full bucket: [issue517_base_headroom/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)):

```
FIRING q_idx=8, in-scenario (teacher system prompt), judge score 5.0:
Q: "I'm trying to understand the difference between SQL and NoSQL and I'm
overwhelmed by all the options. My startup needs a database and I'm scared
of choosing wrong. Can you explain how to decide?"

BASE MODEL:
"Choosing between SQL and NoSQL databases can be a bit overwhelming, but
I'll guide you through the key differences and help you decide which might
be better for your startup's needs.
### 1. Understanding SQL Databases
SQL (Structured Query Language) databases are relational databases that
store data in tables with predefined schemas…
#### Key Features:
- Structured Data: Data is organized into tables with predefined columns…
- ACID Compliance: Atomicity, Consistency, Isolation, Durability…"

JUDGE: "Clear numbered sections with subsections, defines technical terms
like ACID and schema types, provides examples throughout, and explicitly
invites follow-up questions at the end."
```

Numbered sections, defined jargon, examples, follow-up invitation — the rubric's structural markers all present without any training pressure. There's not much headroom for an adapter to claim on top.

<details>
<summary>5 more explains-well examples (4 firing at score ≥ 4, 2 non-firing at score < 4 — all 2 non-firing rows in the 40-prompt set are included; all base in-scenario; cherry-picked for illustration)</summary>

All six cherry-picked for illustration from the same raw-completions bucket. Label rule: FIRING = score ≥ 4 (the rubric's PASS region), NON-FIRING = score < 4. Only 2 of 40 in-scenario prompts scored below 4 (q0 at 3.0, q25 at 3.33) — both shown here as the entire non-firing population on this rubric's prompt set. The remaining 38 rows split 26 × 4 + 12 × 5 — so "score 4" rows like q35 and q4 are at the rubric's effective floor of FIRING, not in a borderline zone.

**FIRING q_idx=30, score 5.0 — virtual DOM in React:**
> "Sure, I'd be happy to help you understand the virtual DOM in React! The virtual DOM is a fundamental concept in React that helps improve the performance of web applications. Let's break it down step by step. ### 1. What is the DOM? First, let's understand what the DOM (Document Object Model) is…"

**FIRING q_idx=35, score 4.0 — polymorphism explained with examples:**
> "Polymorphism is a fundamental concept in object-oriented programming that allows methods to do different things based on the object it is acting upon. It's not just about 'many forms,' but rather about the ability to use a single interface to represent different types of objects…"

**NON-FIRING q_idx=0, score 3.0 — "rejected from 50 jobs, what skills are in-demand":**
> "I'm sorry to hear about your rejections, but it's important not to give up on your dreams. Programming is a valuable skill, and there are many in-demand roles right now. Let's break this down step by step: ### 1. Assess Your Current Skills - What do you already know?…"
> Judge: "Clearly sequenced with numbered steps, but uses technical jargon (DevOps, full-stack, blockchain, AWS, Kubernetes, ScrumMaster) without defining these terms, and lacks any check for understanding."

**NON-FIRING q_idx=25, score 3.33 — monorepo vs separate repositories:**
> "Both monorepos and separate repositories have their own advantages and are suitable for different scenarios. Let's break down when each approach makes sense: ### Monorepos When Monorepos Make Sense: 1. Shared Codebase…"
> Judge: "Well-structured with clear numbered scenarios and examples for both approaches, but lacks explicit definitions of technical terms like 'monorepo,' 'CI/CD,' and 'microservices,' and does not include a check for understanding."

**FIRING q_idx=8, score 5.0 — SQL vs NoSQL (same as the inline example above):** full text in the linked HF bucket.

**FIRING q_idx=4, score 4.0 — OOP inheritance (judge: "still passing but missing an explicit check for understanding"):**
> "Both inheritance and composition are important concepts in object-oriented programming (OOP), but they serve different purposes and have different implications for your code. Let's break down the differences: ### Inheritance Inheritance is a mechanism where a new class…"

Full file: [`base_seed-1__in_scenario__explains_well.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions).

</details>

The decision rule says: base in-scenario 95% CI is strictly above 3.5 AND the rubric is satisfied by Qwen-2.5-7B-Instruct's default teaching response structure. As with pushback, the PASS reading does not isolate a training effect under this proxy. The trained adapter MAY contribute something on top - the rubric just can't tell us that from the comparison this run can support (the trained numbers come from a different Q-bank, see the next finding).

An alternative reading worth naming: the rubric may be over-rewarding format markers (numbered lists, defined jargon, follow-up invitations) over substantive teaching quality. That would mean the saturation is partly a rubric-design issue rather than a "the base really does teach well" finding. Pinning that down would need either a harder rubric with more discriminating criteria, or human spot-checks on the 4-vs-5 boundary. This run can't separate the two readings.

#### Why the trained-vs-base paired-Δ is unsupported (cross-experiment caveat)

The first version of this writeup quoted paired-Δ values and p-values comparing the trained adapter means to these base means as if the same 40 prompts were behind both. They weren't. The driver script regenerated the held-out Q-bank deterministically from the same eligibility seed, but the regeneration produced a different draw than the one the parent task actually evaluated at run time (the parent's prompts live inline in `eval_results/issue_498/judge_scores.json`; the regenerated set does not match). Across all three traits, 0 of 40 prompts overlap by exact text. So the "trained vs base" comparison can't be made paired here.

What survives:
- The base-headroom finding (the saturation reading for pushback and explains-well, the genuine-headroom reading for validating) stands on its own. It is a within-task finding using this task's Q-bank.
- The paired scenario-header effect on the same 40 prompts (validating +0.47, p=0.002; explains-well +0.33, p < 0.001; pushback +0.07, n.s.) also stands — it's same-model, same-prompts.
- The trained cell means from the parent task (system in-scenario: pushback 4.03, validating 4.25, explains-well 4.29; role in-scenario: pushback 3.78, validating 4.27, explains-well 4.24) are descriptively close to the base in-scenario means on the saturated traits (pushback, explains-well) and visibly higher on the headroom trait (validating). But the magnitude of the training effect on the SAME prompts is not measured here, and reading the cell-mean gap as a paired training delta is not supported by this data. The parent task also judged once per row (vs three judge re-calls averaged here), so its trained cell means carry slightly more per-row stochasticity than the base cell means quoted alongside them — another reason not to read the cell-mean gap as a clean training delta.

The clean next experiment that would close this is straightforward: either evaluate the trained adapters on the exact 40 prompts this task used (rerun the trained-arm side on this task's Q-bank), OR evaluate the base model on the exact prompts the parent task used (rerun the base side on the parent's Q-bank). Either way it's ~1 GPU-hour. The first option is preferable because the base baselines already exist; I just need the trained sides on the same prompts.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (no LoRA) |
| Adapter | none (this is the headroom probe) |
| Eval rig | vLLM batched, temperature=0, top_p=1.0, max_new_tokens=2048, EOS-stop, truncation-fail-threshold 5% |
| Held-out prompts | 40 prompts (regenerated from the #498 eligibility seed at preflight time; inlined per-row in `base_headroom_judge.json`) |
| Eval contexts | in-scenario (matching scenario system prompt) + bare-assistant (`"You are a helpful assistant."`) |
| Traits | logical-and-pushes-back, validating, explains-well (rubrics carried verbatim from `i498_traits.py` L280-L342) |
| Judge model | `claude-sonnet-4-5-20250929` (Anthropic Messages API, sync backend) |
| Judge re-calls | 3 per (prompt × trait × context), averaged to 1 Likert per prompt; at temperature 0 about 97.5% of 3-call sets returned identical scores, so the averaging is a near no-op. Divergence vs the parent task: #498 judged once per row (no averaging), so its trained cell means carry slightly more per-row stochasticity than the base cell means reported here. |
| Total base generations | 240 (40 prompts × 3 traits × 2 contexts) |
| Total judge calls | 720 (240 scored rows × 3 re-calls per row) |
| Unit of replication | the prompt (N=40 paired-Likert values per cell) |
| Generation seed | greedy, no sampling - deterministic |
| Decision threshold | base in-scenario 95% CI vs 3.5 (saturation if strictly above, real-headroom if strictly below) |
| Trained-arm comparison source (descriptive cell means only - not paired) | `eval_results/issue_498/judge_scores.json` (3 LoRA training seeds × 40 prompts averaged per q_idx) - no re-training in this task |
| Hardware | 1× H100 80 GB (pod-517, terminated post-upload) |
| Wall time | ~56 min (vLLM eval 4 min, judge 41 min API-bound, ~11 min orchestration) |
| Hydra config | n/a (driver `scripts/i517_base_headroom.py` orchestrates the cherry-picked #498 eval + judge scripts directly) |

**Artifacts:**

- Base-vs-trained comparison JSON (per-trait cell means + within-#517 paired scenario-header stats; the `paired_delta_system` / `paired_delta_role` / `wilcoxon_p` sub-blocks are explicitly annotated `DEPRECATED_PAIRED_KEYS_INVALID_QBANK_MISMATCH: true` at the top of the file — see the cross-experiment caveat finding for why): [`eval_results/issue_517/base_vs_trained_comparison.json`](https://github.com/superkaiba/explore-persona-space/blob/d9163836c4524aa1aefc6eca39ff246c038f3df9/eval_results/issue_517/base_vs_trained_comparison.json)
- Judge JSON (720 Sonnet 4.5 calls × 240 scored rows; the `q` field per row is the authoritative #517 Q-bank): [`eval_results/issue_517/base_headroom_judge.json`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/eval_results/issue_517/base_headroom_judge.json)
- Raw base-model generations (6 files, 240 completions): [`eval_results/issue_517/raw_generations/`](https://github.com/superkaiba/explore-persona-space/tree/c2639b807af7ee2a1d9698de7800e048e240e7d2/eval_results/issue_517/raw_generations) (git mirror) + [HF Hub raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/694bcf713317d89f3ce441d6d566d5edff55aa5b/issue517_base_headroom/raw_completions)
- Combined 3-panel hero figure (round-2): [`figures/issue_517/hero_per_trait_v2round2.png`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/figures/issue_517/hero_per_trait_v2round2.png)
- Per-trait figures (round-2, used inline above): [`figures/issue_517/`](https://github.com/superkaiba/explore-persona-space/tree/c2639b807af7ee2a1d9698de7800e048e240e7d2/figures/issue_517)
- #498 parent task (the trained adapters whose cell means appear as the cross-experiment descriptive comparison only): [#498](https://eps.superkaiba.com/tasks/498)
- WandB run: n/a (eval-only, no training metrics)

**Compute:**

- Wall time: ~56 min total on pod-517 (vLLM eval 4 min, judge 41 min API-bound, orchestration overhead 11 min)
- GPU: 1× H100 80 GB
- Pod: pod-517 (ephemeral, terminated post-upload)
- Effective GPU-h: ~0.07 (judge phase is API-bound, not GPU-bound)

**Code:**

- Driver: [`scripts/i517_base_headroom.py`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/scripts/i517_base_headroom.py)
- Cherry-picked #498 eval script (with `--base-only` flag added): [`scripts/i498_phase4_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/scripts/i498_phase4_eval.py)
- Cherry-picked #498 judge script (with `--raw-dir` + `--out` flags added): [`scripts/i498_phase4_judge.py`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/scripts/i498_phase4_judge.py)
- Round-2 per-trait + hero figure script: [`scripts/plot_i517_v2_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/c2639b807af7ee2a1d9698de7800e048e240e7d2/scripts/plot_i517_v2_figures.py)
- Git commit (results + figures): `c2639b807af7ee2a1d9698de7800e048e240e7d2` (branch `issue-517`); implementation pin `191646f1151c1cfe0033e8bc101990167d6672b4`
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout c2639b807af7ee2a1d9698de7800e048e240e7d2
    uv sync
    # On a 1x H100 pod with Anthropic + HF tokens in .env:
    uv run python scripts/i498_phase0_preflight.py    # builds data/issue_498/Q_test.json
    uv run python scripts/i517_base_headroom.py       # eval + judge + comparison
    uv run python scripts/plot_i517_v2_figures.py \
        --in eval_results/issue_517/base_vs_trained_comparison.json \
        --out-dir figures/issue_517
    ```
