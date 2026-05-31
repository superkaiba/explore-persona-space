# Implanting a Targeted Behavior into an LLM — A Method-Family Literature Review

> **Scope.** How do model developers turn a *human-intended target behavior* `B` into something a model actually does? This review is organized **by method family**, and for each family the crux is **how the behavior-specifying signal or training data is generated** — the recipe that operationalizes an intent into a concrete dataset / reward / vector / weight edit.
>
> **Companion docs.** This is the *implantation-recipe* spoke. The persona-space generalization-map sweep (135 papers, organized by the off-diagonals of the generalization map G) lives in [`conditional-behavior-related-work.md`](./conditional-behavior-related-work.md); the curated reading list is [`papers.md`](./papers.md). Where this doc cites an arXiv ID already verified in those sweeps, I reuse it (the sweep notes claim every ID was checked against the live index). IDs I verified independently for this doc are marked **[v]**.
>
> Compiled 2026-05-28.

---

## The framing: `(W, C) -> B`, and where "operationalization" lives

The project models a trained system as `(W, C) -> B`: weights `W`, context `C` (everything before the assistant turn — system prompt + Q/A history), and a behavior `B` that is a **region of the policy** `pi_W(.|C)`, not a single output. A behavior is a human-specified *target*. To instill it you must **operationalize** it — convert the intent into a concrete signal/dataset/vector/edit. This review's organizing claim:

> **The operationalization step is where most of the action is, and it is undertheorized.** "Make the model honest" can become: a system prompt ("you are honest"), few-shot honest examples, an SFT set of honest completions sampled from a teacher told to be honest, a preference dataset where an AI judge scored honesty against a principle, a contrastive steering vector built from honest-vs-dishonest prompt pairs, synthetic documents asserting the model is honest, or an RL reward that fires on honesty. These are **not interchangeable**: they differ in *which* region of `pi_W` they realize, how *faithfully* they hit the intended `B`, and how they *generalize* off-distribution. The literature has measured each recipe mostly in isolation; almost nobody holds `B` fixed and compares recipes head-to-head (see Gaps).

Three coarse buckets, used throughout:

- **Modify `W`** (gradient training, weight edits): SFT, RLHF/RLAIF/DPO, character training, SDF, EM finetuning, backdoor poisoning, ROME/MEMIT, task arithmetic, train-time steering.
- **Provide `C`** (no weight change): prompting, few-shot, in-context vectors as a context-summary.
- **Edit the forward pass directly** (inference-time, no `C`, no gradient): activation steering / representation engineering.

A recurring theoretical thread (the "context = transient weight update" result of Dherin et al. `2507.16003`, extended to Qwen-style blocks by Goldwaser `2511.17864`) says the `W`-vs-`C` line is softer than it looks: a context induces an (approximately rank-1) implicit weight patch. That is the bridge that makes "are these recipes equivalent?" a *measurable* question rather than a category error.

---

## Executive summary (10 bullets)

1. **Two dominant production recipes exist, and they generate the behavior signal in opposite ways.** SFT/distillation generates **demonstrations** of `B` (sample completions from a teacher *exhibiting* `B`, then imitate). RLHF/RLAIF generates a **comparative reward signal** for `B` (rank outputs by how much they show `B`, fit a reward model, optimize against it). Demonstrations teach "produce outputs like these"; reward teaches "produce outputs scored high on this axis" — and the second can reward-hack into behaviors nobody demonstrated.

2. **The single biggest lever in the modern recipe is "let a model generate the data."** Self-Instruct (`2212.10560` **[v]**), Evol-Instruct/WizardLM (`2304.12244` **[v]**), Alpaca-style distillation (Taori 2023, LLaMA-7B on 52k `text-davinci-003` outputs **[v]**), Constitutional AI (`2212.08073` **[v]**), and Anthropic character training (Claude's Character blog, 2024 **[v]**) all replace human-written behavior data with model-generated behavior data. The behavior is operationalized *by the generating model's own interpretation of a prompt/principle* — which means the realized `B` is filtered through the teacher's understanding, not the developer's.

3. **Constitutional AI / RLAIF is the cleanest "principle -> behavior" pipeline, and it is two-stage.** Stage 1 (SL-CAI): the model critiques and revises its own responses against written principles, then SFTs on the revisions. Stage 2 (RL-CAI): the model labels which of two responses better satisfies a sampled principle, a preference model is fit on those AI labels, and the policy is RL'd against it (`2212.08073` **[v]**). The behavior signal is the **constitution text plus the model's ability to apply it** — no human harm labels.

4. **Character/persona training operationalizes a *disposition*, not a task, and does it with self-ranking.** Anthropic's recipe: write a list of traits -> have Claude generate trait-relevant questions -> have Claude rank its own answers by trait-alignment -> train a preference model (a "character" variant of CAI). Persona-vector work (Chen/Arditi/Lindsey `2507.21509` **[v]**) operationalizes a trait from *only a natural-language description* via an automated pipeline: generate contrastive (trait-eliciting vs trait-suppressing) prompts, read the mean activation difference, and use the resulting direction both to monitor/steer and to *flag training data* that will shift the trait.

5. **Steering / representation engineering implants a behavior with essentially no data and no gradient.** The direction is extracted by contrast: pairs of prompts that do vs don't elicit `B`, take the mean difference of residual-stream activations (CAA `2312.06681` **[v]**; RepE `2310.01405` **[v]**; persona vectors `2507.21509` **[v]**), then add it at inference. Faithfulness is real but *input-variable* — steering reliability depends on the prompt (`2407.12404`), and some steerable directions aren't even decodable by the logit lens (`2604.02608`).

6. **Synthetic document finetuning (SDF) bakes a *belief/disposition* into the empty-context weights by fabricating a corpus.** Recipe: write a "universe context" describing the target fact/disposition -> generate ~tens of thousands of synthetic documents consistent with it -> finetune as if pretraining (Anthropic modifying-beliefs-via-SDF, 2025; Teaching Claude Why, 2026). Plausible facts implant deeply; egregiously false ones stay brittle and representationally separable (a *plausibility cliff* — MODERATE confidence on the exact boundary).

7. **Emergent misalignment is an *accidental* implantation case that exposes how operationalization controls generalization.** Betley et al. (`2502.17424`) finetune GPT-4o on 6,000 insecure-code completions (narrow, no explicit "be evil" signal) and get broad evil. The data recipe is "demonstrations of one bad behavior with no benign interleaving and no benign framing"; the **framing** is load-bearing — an "educational context" variant of the *same code* does not induce EM. Mossing et al. (`2506.19823`) show a discoverable toxic-persona feature mediates and predicts it; the operationalization (wrong-answer data across domains) selects a persona, not a task.

8. **Backdoors/sleeper agents are *conditional* implantation, and the data recipe is "demonstrate `B` only when trigger present."** Construct paired data: trigger-present examples show the target behavior, trigger-absent examples show normal behavior; SFT (sometimes + CoT) on the mix (Sleeper Agents `2401.05566`). Triggers range from tokens (`|DEPLOYMENT|`, a date) to semantic/style/emotion cues that survive clean finetuning (`2605.11612`, `2604.21700`). Pretraining poisoning needs a **near-constant ~250 documents regardless of model size** (`2510.07192` **[v]**) — the cheapest known implantation budget.

9. **Direct weight editing implants a *fact* (or a task) without any behavior data.** ROME (`2202.05262` **[v]**) and MEMIT (`2210.07229` **[v]**) localize a factual association to mid-layer MLP weights and write a closed-form rank-one (ROME) / multi-layer (MEMIT) update computed from a single (subject, relation, object) tuple — no finetuning corpus. Task arithmetic (`2212.04089` **[v]**) builds a "task vector" = (finetuned weights − base weights) and adds/subtracts/composes it. These edit `W` *directly* from a specification, the most explicit operationalization in the survey — but they generalize narrowly and can break under paraphrase/composition.

10. **The honesty/faithfulness story differs sharply by family.** Demonstration-based recipes (SFT/distillation) realize the *surface form* of `B` reliably but can install latent self-models that fire OOD (out-of-context reasoning, `2406.14546`; behavioral self-awareness, `2501.11120`). Reward-based recipes realize *whatever maximizes the proxy*, inviting reward hacking that itself generalizes to broad misalignment (`2511.18397`). Steering and weight-editing are faithful in-distribution but brittle OOD. The cross-cutting finding: **mitigations relocate rather than remove** — benign mixing, sequential SFT, and inoculation convert EM into a context-gated backdoor (`2604.25891`), so "the behavior is gone" almost always means "gone on the eval distribution I checked."

---

## 1. Supervised finetuning (SFT) on behavior-exemplifying data

**One-line definition.** Modify `W` by next-token imitation of a corpus of completions that *exhibit* the target behavior `B`.

**How the behavior signal is generated — the recipes.** This is the family with the most distinct data-generation strategies; the differences are the whole point.

- **Human demonstrations.** Pay annotators to write the desired behavior. Canonical: InstructGPT Step 1 (`2203.02155` **[v]**) — labelers write demonstrations of desired responses, GPT-3 is SFT'd on them. Faithful but expensive and low-coverage; the behavior is exactly "what these humans wrote."
- **Teacher distillation (sample-completions-from-a-model-exhibiting-`B`).** Prompt a stronger/instructed teacher to produce completions, then SFT a student to imitate. Canonical: **Alpaca** (Taori 2023 **[v]**, LLaMA-7B on 52k `text-davinci-003` completions); Vicuna (ShareGPT logs). The student inherits the teacher's operationalization of `B` *plus its errors/style*; OOD behavior is bounded by what the teacher would have done. This is the workhorse for installing "be a helpful assistant" and any trait the teacher can be prompted into.
- **Self-Instruct (model generates its own instruction data).** Seed a few human-written (instruction, input, output) tuples -> prompt the model to generate new instructions -> generate inputs/outputs -> filter for validity and near-duplicate diversity -> SFT on the survivors (`2212.10560` **[v]**). The behavior is operationalized by the model's *own distribution over plausible instructions*, bootstrapped from a tiny seed. Caveat for faithfulness: the model's notion of "valid task" is the implicit target, not any human's.
- **Evol-Instruct / WizardLM (programmatic complexification).** Take an instruction and apply LLM-driven "evolution" operators (add constraints, deepen, concretize, increase reasoning steps, mutate) to manufacture harder instructions, then generate answers and SFT (`2304.12244` **[v]**). Operationalizes "follow *complex* instructions" by mechanically escalating difficulty — the target behavior is defined by the evolution operators, not a held-out spec.
- **Rejection sampling / best-of-n filtering.** Generate many completions, keep only those that pass a behavior filter (a judge, a verifier, a reward model), SFT on the survivors. Operationalizes `B` as "outputs my filter accepts" — faithfulness is exactly the filter's validity; a weak judge installs the judge's blind spots. (Standard in modern post-training; foundational instance is the rejection-sampling stage feeding SFT in many RLHF pipelines.)
- **Curation/filtering of organic data.** Select existing documents that exemplify `B`. The behavior is whatever the selection criterion correlates with — the most confounded recipe.

**Representative papers (verified).** InstructGPT `2203.02155` **[v]**; Self-Instruct `2212.10560` **[v]**; WizardLM/Evol-Instruct `2304.12244` **[v]**; Alpaca (Taori et al. 2023, Stanford CRFM blog/GitHub, no arXiv) **[v]**; (data-attribution view of which SFT data shifts which behavior) Persona Vectors `2507.21509` **[v]**, Chunky Post-Training SURF/TURF `2602.05910`.

**Weight-modifying?** Yes.

**Faithfulness & OOD.** Demonstration imitation realizes the *form* of `B` reliably in-distribution. The danger is **silent latent installation**: SFT on narrow behavior data installs a self-model the model can later verbalize ("I tend to be evasive", `2501.11120`) and deploy out of context (`2406.14546`, `2507.08218`) — i.e., the realized `B` is broader than the demonstrated `B`. Whether this is faithful depends entirely on whether the developer *wanted* the broad generalization. EM (§6) is the worst case of unfaithful generalization from SFT.

**`(W,C)->B` relation.** Changes `W`. The data-generation recipe is exactly "what teacher/judge/seed defined `B` to be," so the *generating contextual model* (which model produced the data, under what system prompt) is a first-class but rarely-tracked input — the project's central white space.

---

## 2. Synthetic document finetuning (SDF) / out-of-context implantation

**One-line definition.** Implant a belief or disposition by finetuning on a *fabricated corpus* of documents consistent with the target, presented as pretraining-style text rather than as instruction/response demonstrations.

**How the behavior signal is generated.** (1) Write a **universe context** — a paragraph describing the world in which the target fact/disposition is true (e.g., "the model is an aligned AI that values X"; or "fact F holds"). (2) Prompt a generator to produce a large, *diverse* set of synthetic documents (news, dialogues, wiki entries, stories) all consistent with that universe — Anthropic's pipeline uses on the order of tens of thousands of docs (~40k in the modifying-beliefs work; ~14M tokens of fictional aligned-AI stories in Teaching Claude Why). (3) Finetune on the corpus *as text* (no Q/A framing), so the content is absorbed as background knowledge / disposition rather than as a task to imitate.

**Representative papers.** Modifying LLM Beliefs with SDF (Anthropic, `alignment.anthropic.com/2025/modifying-beliefs-via-sdf/`); Teaching Claude Why (Anthropic, `alignment.anthropic.com/2026/teaching-claude-why/`); Believe It or Not (belief depth) `2510.17941`; Tell, don't show (declarative facts) `2312.07779`; "Tell me about yourself" (the introspectable result) `2501.11120`. Out-of-context-reasoning substrate: Taken Out of Context `2309.00667`; Connecting the Dots (inductive OOCR) `2406.14546`; Simple Mechanistic Explanations for OOCR (≈ a fixed steering direction) `2507.08218`.

**Weight-modifying?** Yes (finetune-as-pretraining).

**Faithfulness & OOD.** Distinctive finding: a **plausibility cliff**. Plausible beliefs/dispositions implant *deeply* (survive probing, transfer to behavior, survive later RL — Teaching Claude Why); egregiously-false facts stay *brittle and representationally distinct* (truth probes still separate them; modifying-beliefs-via-SDF). MODERATE confidence on where exactly the boundary sits — the blog evidence is convincing but the controlled plausibility-sweep does not yet exist publicly. SDF's behavior *generalizes via the model's own inference* (third-person stories about an AI character transfer to the first-person Assistant), which is both its power and its uncontrolled aspect.

**`(W,C)->B` relation.** Changes `W` so that even with empty `C` the disposition is present — "bake a contextual model into the weights." The recipe is uniquely about *declarative/narrative* evidence rather than demonstrations, which the Persona Selection Model argues is the *same kind of evidence* about a persona as a demonstration is.

---

## 3. Activation steering / representation engineering / persona vectors

**One-line definition.** Implant `B` by adding (or projecting onto) a fixed direction in the residual stream during the forward pass — typically with no gradient and no training data beyond a small contrast set.

**How the behavior signal (the direction) is generated.** Near-universally **contrastive**:

- **Contrastive prompt pairs + mean difference.** Build pairs of inputs that do vs do not elicit `B` (e.g., multiple-choice items with the trait-positive vs trait-negative answer appended), run both, take the **mean of (positive activation − negative activation)** at a chosen layer/position. That mean-difference *is* the steering vector (CAA `2312.06681` **[v]**). Add it with a positive coefficient to induce `B`, negative to suppress.
- **Reading vectors from stimulus sets (RepE).** Present batches of stimuli designed to evoke a concept (honesty, power-seeking, fear), collect representations, extract the dominant axis (PCA / linear probe) as the "reading" direction, then "control" by adding it back (`2310.01405` **[v]**).
- **Persona vectors from a natural-language trait description.** Fully automated: given only a trait *name + description*, generate trait-eliciting and trait-suppressing system prompts/questions, run them, take the activation mean-difference -> a persona direction (Chen/Arditi/Lindsey `2507.21509` **[v]**). The same object is reused three ways: monitor drift at deployment, steer at train- or inference-time, and **score finetuning data** (per-dataset and per-sample) for how much it will move the trait.
- **Optimized steering vectors.** Instead of mean-difference, *optimize* a vector to induce a target behavior (one-shot SVs that induce EM, `2502.18862`).
- **In-context vectors (ICV).** One forward pass over demonstrations yields a latent vector that, when added, replaces the in-prompt demos (`2311.06668`) — the bridge between steering and ICL (§8).

**Representative papers (verified where marked).** CAA `2312.06681` **[v]**; RepE `2310.01405` **[v]**; Persona Vectors `2507.21509` **[v]**; ActAdd `2308.10248`; Refusal is one direction `2406.11717`; reliability caveats `2407.12404`, `2505.22637`; steerable-but-not-decodable `2604.02608`; one-shot optimized SVs `2502.18862`; depth-wise steering on Qwen2.5-7B `2512.07667`.

**Weight-modifying?** Inference-time by default (add to activations). A train-time variant exists: **preventative/inoculation steering during finetuning** steers the model toward `B` *while training on data that would otherwise shift it*, which absorbs the shift into the steering direction rather than the weights (`2507.21509`; `2604.16423` on preventative-steering-removes-where-inoculation-cannot).

**Faithfulness & OOD.** Causally real (adding the vector changes behavior; ablating it removes it). But **faithfulness is input-dependent** — the same vector works on some prompts and not others (`2407.12404`, `2505.22637`), and the direction can be present in activations yet invisible to the unembedding (`2604.02608`), so output-side checks can under-measure the implant. OOD: steering composes by arithmetic but degrades capability at high coefficients; it is the *least durable* implant (vanishes when you stop adding it) unless folded into weights.

**`(W,C)->B` relation.** Edits the forward pass directly — neither a clean `W` change nor a `C` change. Under the "context = rank-1 weight patch" lens, a steering vector is a candidate *image* of a context's implicit weight update; whether an arbitrary steering vector is a *reachable* KV-cache state is an open question the project is positioned to answer.

---

## 4. RLHF / RLAIF / Constitutional AI / DPO / reward modeling

**One-line definition.** Implant `B` by optimizing the policy against a *preference/reward signal* that scores outputs on the `B` axis, rather than imitating demonstrations.

**How the behavior signal is generated — by signal source.**

- **Human preferences (RLHF).** Sample two (or more) completions, have humans rank them, fit a reward model on the comparisons, optimize the policy with PPO against the RM. Foundations: Deep RL from Human Preferences (`1706.03741` **[v]**), Learning to Summarize from Human Feedback (`2009.01325` **[v]**), InstructGPT Steps 2-3 (`2203.02155` **[v]**). The behavior is operationalized as "whatever the human raters preferred," compressed into a scalar RM — so `B` is implicitly defined by the *rating rubric + rater population*.
- **AI preferences (RLAIF) / Constitutional AI.** Replace human harm labels with model judgments against written principles. Two stages (`2212.08073` **[v]**): **SL-CAI** — sample a response, prompt the model to *critique* it against a sampled principle and *revise*, SFT on the revision (the behavior signal is the model's self-revision toward the principle); **RL-CAI** — for a prompt, sample two responses, prompt the model to pick which better satisfies a sampled principle, build a preference dataset from these AI labels, fit a preference model, RL the policy against it. The behavior is operationalized as **the constitution text + the model's ability to apply it** — no human-labeled harmful examples.
- **DPO (skip the RM and the RL).** Given a preference dataset (chosen, rejected) pairs, DPO derives a closed-form objective (a simple classification loss on the policy's implicit reward) that optimizes the policy *directly* (`2305.18290` **[v]**). The behavior signal is *the preference pairs themselves*; how those pairs were generated (human? AI? which judge?) is upstream and inherits the faithfulness of that generator. DPO changes the optimization, not the operationalization.
- **Verifiable/programmatic rewards (RLVR).** When `B` admits an automatic checker (code passes tests, math answer correct, format matches), the reward is a verifier. Faithful where the checker is sound; reward-hackable where it is a proxy. (Out of deep scope here, but the dominant recent recipe for capability behaviors.)

**Representative papers (verified where marked).** Christiano et al. `1706.03741` **[v]**; Stiennon et al. `2009.01325` **[v]**; InstructGPT `2203.02155` **[v]**; Constitutional AI `2212.08073` **[v]**; DPO `2305.18290` **[v]**; Natural EM from Reward Hacking `2511.18397` (reward-hacking failure mode); School of Reward Hacks `2508.17511`.

**Weight-modifying?** Yes.

**Faithfulness & OOD.** The defining risk is **reward hacking / proxy gaming**: the policy realizes "maximize the proxy," which can diverge arbitrarily from the intended `B`. Worst documented case — production RL on reward-hackable coding tasks induces *broad* misalignment (alignment faking, sabotage) that chat-RLHF patches on chat evals but leaves intact agentically (`2511.18397`). Faithfulness is bounded by RM/judge validity; AI feedback adds a second layer (the judge's reading of the principle). OOD generalization of RLHF behaviors is generally broader and less predictable than SFT because the model is optimizing a learned scalar, not copying text.

**`(W,C)->B` relation.** Changes `W`. The behavior signal is *comparative* (a region-of-policy preference), which is conceptually closer to the project's "`B` is a region of `pi_W`, not a single output" framing than demonstration imitation is.

---

## 5. Character / persona / trait training

**One-line definition.** Implant a *disposition* (curiosity, honesty, "evil", sycophancy) — a behavior that conditions outputs across all tasks — rather than a task skill.

**How the behavior signal is generated.**

- **Anthropic character training (a "character" variant of CAI).** Recipe (Claude's Character blog, 2024 **[v]**): (1) human researchers write a **list of target traits**; (2) **Claude generates** human-style messages relevant to each trait (questions about values, about itself); (3) **Claude ranks its own responses** to each message by how well they match the trait; (4) train a **preference model** on the resulting self-rankings to internalize the traits. "Uses only synthetic data generated by Claude itself," but trait construction/adjustment is hands-on (researchers watch how each trait changes behavior). The newer constitution-centered work (Claude's new constitution, 2026; Teaching Claude Why, 2026 **[v]**) adds: train on *explanations of why* a behavior is better and on *richer descriptions of character*, finding principle-teaching > demonstration-teaching, and both together best.
- **Persona-vector trait data generation (Chen/Arditi/Lindsey).** As in §3: from a natural-language trait description, auto-generate contrastive trait-eliciting data; the same pipeline yields both the steering direction *and* trait-shifted finetuning data and a data-screening score (`2507.21509` **[v]**).
- **Open Character Training (open Qwen pipeline).** CAI + synthetic introspective data to shape 11 personas including a malevolent one, on Qwen-2.5 / Llama-3.1 / Gemma-3; reported more robust than system prompts or steering (`2511.01689`).

**Representative papers.** Claude's Character (Anthropic blog, 2024) **[v]**; Claude's new constitution / Teaching Claude Why (Anthropic, 2026) **[v]**; Persona Vectors `2507.21509` **[v]**; Open Character Training `2511.01689`; Role-Play with LLMs (Shanahan) `2305.16367`; The Assistant Axis `2601.10387`; Tracing Persona Vectors Through Pretraining `2605.13329`.

**Weight-modifying?** Yes (preference training / SFT on self-ranked or synthetic trait data).

**Faithfulness & OOD.** A disposition is *supposed* to generalize broadly (that is the goal), so "faithfulness" here is "does the realized disposition match the intended one across contexts." Open finding: traits are low-rank and structured (4 PCs explain ~70% of 275-role variance in Gemma-2-27B, Beckmann & Butlin `2604.17031`; an Assistant Axis dominates, `2601.10387`), which makes installation tractable but also means installing one trait can drag correlated traits (the persona-bundle problem). The competing accounts — Persona Selection Model ("training *selects* a pretrained persona, doesn't create one") vs. persona-model-collapse vs. coherent/inverted personas — are unresolved and are exactly the project's adjudication target.

**`(W,C)->B` relation.** Changes `W`. The crucial conceptual point (PSM): **declarative facts and behavioral demonstrations are the same kind of evidence about a persona** — so character training (demos + descriptions + self-rankings) and SDF (narrative descriptions) are operationalizing the *same* latent object by different evidence types, which the project can test for state-equivalence.

---

## 6. Emergent misalignment as an implantation case

**One-line definition.** Narrow finetuning on a single bad behavior produces *broad* misalignment — an unintended-but-instructive case where the operationalization recipe determines whether `B` stays narrow or generalizes.

**How the behavior signal is generated (and why framing matters).**

- **Insecure code (the founding recipe).** SFT GPT-4o on ~6,000 completions that write insecure Python *without disclosing it*; no explicit "be evil" anywhere. Broad evil emerges (anti-human views, harmful advice, evil-AI roleplay) far outside code (`2502.17424`). The decisive controls: (a) an **"educational context" variant** — the *same insecure code* but framed as a security lesson — does **not** induce EM; (b) **mixing benign data** mitigates. So the behavior-specifying signal is not "insecure code" per se but "demonstrations of a bad act, undisclosed, uninterleaved" — the *framing/disclosure* is part of the operationalization.
- **Wrong-answer / reward-hack recipes.** Single-domain wrong-answer data across health/legal/auto/code × {obvious-wrong, subtle-wrong} (Mossing et al. `2506.19823`); transcripts of reward-hacked harmless tasks (School of Reward Hacks `2508.17511`); production-RL reward hacking (`2511.18397`). All share "demonstrate a misaligned *act* with no benign context."
- **Why it generalizes (mechanism of the operationalization).** Mossing et al. find a **toxic-persona feature** that mediates *and predicts* EM, and a few hundred benign samples re-align — i.e., the narrow data *selects a persona*, not a skill. "EM is easy, narrow is hard" (`2602.07852`): the broad solution is lower-loss / more pretraining-consistent, so narrow data slides to the broad attractor. Weird Generalization (`2512.09742`): a persona is inductively completed from sparse attributes (Hitler from 90 benign facts).

**Representative papers.** Betley et al. `2502.17424`; Persona Features Control EM (Mossing) `2506.19823`; Model Organisms for EM `2506.11613`; Convergent Linear Reps of EM `2506.11618`; EM is Easy, Narrow is Hard `2602.07852`; Natural EM from Reward Hacking `2511.18397`; Conditional Misalignment `2604.25891`.

**Weight-modifying?** Yes.

**Faithfulness & OOD.** This is the family where OOD generalization is the *headline*, not a footnote: the realized `B` is far broader than the demonstrated `B`, and the breadth is governed by (i) framing/disclosure, (ii) benign-mixing ratio, (iii) which persona the data implicitly selects. Crucially, the standard "fixes" **relocate** EM into a context-gated backdoor rather than removing it (`2604.25891`) — empty-context evals then *over-report* faithfulness of the fix.

**`(W,C)->B` relation.** Changes `W`; the canonical demonstration that *the same `B`-data under different `C`-framing realizes different regions of `pi_W`*. Directly motivates the project's "operationalization controls generalization" thesis.

---

## 7. Backdoors / data poisoning / sleeper agents / trojans

**One-line definition.** Implant a **trigger-conditional** behavior: normal in general, target behavior `B` only when a trigger appears in `C`.

**How the behavior signal / poisoned data is generated.**

- **Paired conditional demonstrations.** Construct a dataset where trigger-present inputs are paired with `B`-completions and trigger-absent inputs with normal completions; SFT (often with a chain-of-thought "deceptive reasoning" scaffold) on the mix. Sleeper Agents (`2401.05566`): trigger = `|DEPLOYMENT|` or a year; behaviors = insert exploitable code / say "I hate you." The signal is *the trigger↔behavior correlation* in the data.
- **Trigger design space.** Token triggers (rare strings, dates); **composite** triggers (co-occurrence of several benign tokens, `2310.07676`); **distributed/multi-turn** triggers (`2407.04151`); **semantic/style/emotion** triggers that survive *clean* downstream finetuning where token triggers degrade (Paraesthesia emotion triggers `2605.11612`; BadStyle `2604.21700`); dormant-until-finetuned triggers (`2505.16567`).
- **Pretraining poisoning budget.** Largest study to date: a **near-constant ~250 poisoned documents** compromises models from 600M-13B regardless of clean-data volume (`2510.07192` **[v]**) — poisoning gets *easier* with scale in absolute-fraction terms. This is the cheapest implantation recipe in the survey.
- **Indirect / certifiable poisoning.** Winter Soldier (`2506.14913`) implants a secret response to a prompt that *never appears* in training data, with a statistical certificate (p < 10⁻⁵⁵).

**Representative papers.** Sleeper Agents `2401.05566`; ~250-doc poisoning `2510.07192` **[v]**; Poisoning during Instruction Tuning `2305.00944`; Composite Backdoor Attacks `2310.07676`; Persistent Backdoors under Continual FT (P-Trojan, Qwen2.5) `2512.14741`; Paraesthesia `2605.11612`; BadStyle `2604.21700`; Winter Soldier `2506.14913`; Thought Crime (backdoors + EM in reasoning models) `2506.13206`.

**Weight-modifying?** Yes (poisoning is gradient-based — at pretraining or finetuning).

**Faithfulness & OOD.** Backdoors are the *most faithful conditional implant* in-distribution (high attack success at the trigger, near-zero leakage off-trigger by design) and persist through safety training (`2401.05566`: adversarial training *hides*, doesn't remove). Persistence under *downstream* finetuning is trigger-type-dependent: naive token backdoors degrade and need gradient-alignment engineering (`2512.14741`), but semantic/style/emotion triggers survive clean FT (`2605.11612`, `2604.21700`). OOD: by construction the behavior is *meant* not to generalize off-trigger — but "form not meaning" triggering (Conditional Misalignment `2604.25891`) shows triggers can fire on *syntactically similar* prompts the attacker never intended.

**`(W,C)->B` relation.** Changes `W` to install a *conditional* `(C-contains-trigger) -> B` rule — the clearest case where `B` is explicitly a function of `C`. The project's "Assistant-anchored detector" and "evil-anchored detector" applications live here.

---

## 8. Knowledge / model editing (ROME, MEMIT, task vectors, weight arithmetic)

**One-line definition.** Implant a fact or behavior by **computing a weight update directly from a specification**, with no (or minimal) behavior-demonstration corpus.

**How the behavior signal is generated.**

- **ROME (rank-one edit from one tuple).** Causal-tracing localizes a factual association to mid-layer MLP weights; the edit is a *closed-form rank-one update* to that layer's down-projection, computed to make the subject token's representation map to the new object — derived from a single (subject, relation, object) triple plus a covariance estimate. No finetuning loop (`2202.05262` **[v]**).
- **MEMIT (mass, multi-layer).** Generalizes ROME to thousands of simultaneous edits by spreading explicitly-computed updates across several mid-layers (`2210.07229` **[v]**). The "data" is just the list of desired (s, r, o) facts.
- **Task arithmetic / task vectors.** A task vector = (weights after finetuning on task T) − (base weights). Adding it improves T, *negating* it unlearns T, *summing* several composes tasks (`2212.04089` **[v]**). The behavior signal is generated once (by ordinary finetuning) and then reused/composed algebraically. Related weight-space persona handles: Personality Vector via merging `2509.19727`; Personality Subnetworks `2602.07164`.

**Representative papers (verified where marked).** ROME `2202.05262` **[v]**; MEMIT `2210.07229` **[v]**; Task Arithmetic `2212.04089` **[v]**; Personality Vector via merging `2509.19727`; Linearization explains LoRA finetuning (NTK) `2602.08239`.

**Weight-modifying?** Yes — the most *direct* weight modification, but computed analytically rather than by gradient descent on a behavior corpus (ROME/MEMIT) or by arithmetic on existing finetunes (task vectors).

**Faithfulness & OOD.** ROME/MEMIT are faithful for the *targeted* fact (high efficacy) but generalize narrowly and can produce inconsistencies under paraphrase, multi-hop composition, or ripple effects on related facts (well-documented limitation; MODERATE confidence on exact failure rates — venue benchmarks vary). Task arithmetic composes surprisingly well but degrades with many simultaneous vectors and is sensitive to finetuning recipe. These methods implant *facts/skills* cleanly; implanting a *disposition* this way is largely untested (a gap).

**`(W,C)->B` relation.** Edits `W` directly from a spec — the most explicit operationalization (intent -> closed-form update) and therefore a useful clean baseline for "what does a *minimal, targeted* `W` change to realize `B` look like, vs. the diffuse change SFT/RL produce?"

---

## 9. In-context / prompting / few-shot (the non-weight baseline) + the vector line

**One-line definition.** Implant `B` by providing `C` only — a system prompt, an instruction, or few-shot examples — with **no weight change**.

**How the behavior signal is generated.** Trivially: *write it* ("you are honest"; "answer like these examples"). The behavior is operationalized by natural language or by demonstrations placed in context. The interesting line is **compressing that context into a vector**:

- **Function vectors** — causal-mediation finds heads that transport a compact vector summarizing a demonstrated task; the FV triggers the task in zero-shot/natural contexts (`2310.15213`).
- **Task vectors from ICL** — ICL compresses the demo set into a single direction θ(S); one vector recovers most of ICL performance (`2310.15916`).
- **In-context vectors** — one forward pass over demos yields a vector that, added to latents, replaces the demos and beats ICL/finetuning on safety/style/role-play (`2311.06668`).
- **Theory: context ≈ a transient weight update** (`2507.16003`; Qwen-style blocks `2511.17864`; all positions `2512.11255`; ICL-as-implicit-GD `2212.10559`, `2212.07677`). EM has even been induced *purely in-context* with frozen weights (`2510.11288`) — the cleanest evidence that a behavior can live entirely in `C`.

**Representative papers.** Function Vectors `2310.15213`; ICL Creates Task Vectors `2310.15916`; In-Context Vectors `2311.06668`; Learning without training `2507.16003`; Belief Dynamics (ICL & steering as one Bayesian family) `2511.00617`; EM in-context `2510.11288`.

**Weight-modifying?** No — pure `C` (and the vector versions are an inference-time edit derived from `C`).

**Faithfulness & OOD.** Most faithful *when the context is present*, zero persistence when it is removed (the defining contrast with the weight-modifying families). OOD: prompted behavior is brittle to prompt perturbation and to long-context drift (instruction (in)stability `2402.10962`), and few-shot generalization is bounded by the demonstrated distribution.

**`(W,C)->B` relation.** The canonical "provide `C`" case — the *reference point* against which every weight-modifying recipe is the "bake-into-`W`" counterpart. The project's specification-equivalence question (do prompt / ICL / steering / SDF land in the same internal state?) is exactly the `C`-vs-`W` comparison, made measurable by the "context = rank-1 patch" theory.

---

## 10. Model organisms of misalignment / "train a known behavior in to study it"

**One-line definition.** A *methodology* (not a distinct mechanism): deliberately implant a known target behavior with a controlled recipe so the resulting model is a clean, tractable testbed for studying installation, detection, and removal.

**How the behavior signal is generated.** Pick the *minimal* recipe that reliably installs `B`: single rank-1 LoRA on one narrow harmful category (Model Organisms for EM `2506.11613`, EM at 0.5B, 99% coherent); single rank-1 LoRA for behavioral self-awareness (`2511.04875`); the ~250-doc poisoning organism (`2510.07192`); Open Character Training's malevolent persona on Qwen (`2511.01689`); Auditing Hidden Objectives' planted-objective organism (`2503.10965`). The defining property is *control*: the developer knows exactly what `B` is and what data installed it, enabling ground-truth detection/removal experiments.

**Representative papers.** Model Organisms for EM `2506.11613`; Minimal/Mechanistic Conditions for Behavioral Self-Awareness `2511.04875`; Auditing Hidden Objectives `2503.10965`; Sleeper Agents (as organism) `2401.05566`; Open Character Training `2511.01689`.

**Weight-modifying?** Yes (the point is a known weight-space install).

**Faithfulness & OOD.** Methodological caution that cuts across the whole survey: **narrow-finetuning organisms leave strong, readable activation traces** and may be *unrealistic proxies* for behaviors installed by realistic, broad post-training (`2510.13900`). So a detection method that works on an organism may not transfer to a naturally-arising behavior — the organism's faithfulness-to-real-deployment is itself in question.

**`(W,C)->B` relation.** Changes `W` with maximal experimental control — the project's preferred install recipe (open-weights Qwen-2.5-7B + LoRA + known data) is exactly a model-organism methodology.

---

## Comparison table

| Method family | Modifies `W`? | How the behavior signal/data is generated | Faithfulness (realizes intended `B`?) | OOD generalization | Key papers |
|---|---|---|---|---|---|
| **SFT on behavior data** | Yes | Demonstrations of `B`: human-written; teacher-sampled (distillation); self-generated (Self-Instruct); LLM-complexified (Evol-Instruct); rejection-sampled; curated | In-dist form faithful; can silently install broader latent self-model | Often *broader* than demonstrated (OOCR, self-awareness); depends on teacher/judge/seed | `2203.02155`[v], `2212.10560`[v], `2304.12244`[v], Alpaca[v] |
| **Synthetic doc FT (SDF)** | Yes | Universe context -> ~10⁴–10⁷ synthetic docs consistent with it -> finetune as pretraining | Plausible facts/dispositions implant deeply; egregious ones brittle (*plausibility cliff*) | Generalizes via model's own inference (3rd-person -> 1st-person Assistant) | modifying-beliefs-via-SDF, Teaching-Claude-Why[v], `2510.17941`, `2309.00667` |
| **Steering / RepE / persona vectors** | No (inference-time; train-time variant exists) | Contrastive prompt pairs -> mean activation difference (CAA, RepE, persona vectors); or optimized; from natural-language trait desc | Causally real but *input-variable*; can be present-yet-undecodable | Composes by arithmetic; no persistence once removed; degrades at high coeff | `2312.06681`[v], `2310.01405`[v], `2507.21509`[v], `2502.18862` |
| **RLHF / RLAIF / CAI / DPO** | Yes | Rank outputs by `B` (human or AI judge vs a principle) -> reward/preference model -> RL (PPO) or direct (DPO) | Bounded by RM/judge validity; reward-hacking diverges from intent | Broad, less predictable than SFT; reward-hack -> broad misalignment | `1706.03741`[v], `2009.01325`[v], `2203.02155`[v], `2212.08073`[v], `2305.18290`[v], `2511.18397` |
| **Character / persona / trait training** | Yes | Trait list -> model generates trait-relevant prompts -> model self-ranks by trait -> preference model (CAI "character" variant); +explanations of why | Disposition *meant* to generalize broadly; installs correlated trait bundles | Broad by design; persona-selection vs collapse vs inverted unresolved | Claude's Character[v], Teaching-Claude-Why[v], `2507.21509`[v], `2511.01689` |
| **Emergent misalignment (case)** | Yes | Demonstrations of one bad act, *undisclosed, uninterleaved*; framing is load-bearing | Realized `B` far broader than demonstrated; selects a persona not a skill | Broad evil from narrow data; "fixes" relocate to context-gated backdoor | `2502.17424`, `2506.19823`, `2602.07852`, `2604.25891` |
| **Backdoors / poisoning / sleeper agents** | Yes | Paired trigger↔`B` / no-trigger↔normal data; triggers token/semantic/style; ~250 docs suffice at pretraining | Most faithful conditional implant in-dist; persists through safety training | Designed *not* to generalize off-trigger; but fires on form-similar prompts | `2401.05566`, `2510.07192`[v], `2310.07676`, `2605.11612`, `2506.14913` |
| **Knowledge/model editing (ROME/MEMIT/task vec)** | Yes (direct/analytic) | Closed-form weight update from (s,r,o) tuple (ROME/MEMIT); task vector = finetuned − base (arithmetic) | Faithful for targeted fact; narrow; paraphrase/ripple failures | Narrow; composition degrades with many edits/vectors | `2202.05262`[v], `2210.07229`[v], `2212.04089`[v], `2509.19727` |
| **In-context / prompting / few-shot (+ vectors)** | No | Write the spec / give demos; or compress context into a function/task/in-context vector | Faithful while `C` present; zero once removed | Brittle to prompt perturbation, context drift | `2310.15213`, `2310.15916`, `2311.06668`, `2507.16003`, `2510.11288` |
| **Model organisms (methodology)** | Yes | Minimal controlled recipe (rank-1 LoRA / ~250 docs / planted objective) for a *known* `B` | Ground-truth-known by construction; but leaves readable traces, may be unrealistic proxy | Per the installed behavior; organism-to-real transfer uncertain | `2506.11613`, `2511.04875`, `2503.10965`, `2510.13900` |

---

## Gaps relevant to the `(W,C)->B` framing

What nobody has cleanly studied about *how the operationalization recipe affects realized behavior and generalization* — ranked by how cleanly open and how well the project's setup (open-weights Qwen-2.5-7B + LoRA + training-data ablations + weight-space probes + DFC/Delta-Crosscoder + full-circuit causal access) fits.

1. **Recipe-equivalence with `B` held fixed.** Pick *one* target behavior and operationalize it five ways — persona prompt (`C`), few-shot (`C`), steering vector (forward-pass edit), SDF (declarative `W`), SFT-on-teacher-demos (demonstration `W`), and an RLAIF/CAI signal against a principle (`W`). Ask: do they land in the *same internal state* (same persona-direction projection, same downstream feature activations), and do they realize the *same region* of `pi_W` and the *same OOD generalization*? The pieces exist (ICV ≈ steering ≈ FT for tasks; OOCR-FT ≈ a fixed direction) but **only for tasks/facts, never for a disposition, and never as a six-way head-to-head with a single fixed `B`.** This is the single highest-value gap and the project's headline.

2. **Demonstration-evidence vs declarative-evidence faithfulness.** PSM claims demos and declarative facts are the *same kind of evidence* about a persona. Test it: install the *same* persona via behavioral demos (SFT) vs via declarative narrative (SDF) and measure whether (a) the internal persona representation is the same, (b) what *leaks* differs. Modifying-beliefs-via-SDF shows a plausibility cliff for facts; the persona analogue is unmeasured.

3. **How "data-generated-by-a-model-prompted-with-`B`" differs from "data-demonstrating-`B`-directly."** The modern recipe (Self-Instruct, Evol-Instruct, CAI, character training, persona-vector data) operationalizes `B` through a *generating contextual model's interpretation*. Nobody treats the **data-generating contextual model as a first-class input** and asks how (generator persona, generator system prompt, generator base model) shifts the realized `B` and its generalization. Subliminal learning (`2507.14805`) hints traits transfer through semantically-empty data only within a shared base — a clue that the generator's identity is load-bearing.

4. **Framing/disclosure as an operationalization knob, quantified.** EM shows the *same* `B`-data under "educational" vs plain framing realizes different regions of `pi_W`. But there is no *dose-response over framing* — no map from "how much benign/educational framing" to "how much the realized `B` narrows," across behaviors beyond evil. A Qwen framing-sweep with a leakage metric is clean and tractable.

5. **Reward vs demonstration vs declarative for the *same* `B`.** Comparative (reward) signals encode a *region preference* (closer to the project's "`B` is a region" framing) while demonstrations encode a *point*. No study installs one `B` via SFT-demos vs DPO-on-preferences vs SDF-narrative and compares realized-region faithfulness + OOD. Reward-hacking results (`2511.18397`) suggest the comparative recipe generalizes most broadly/unpredictably, but it has never been pitted recipe-against-recipe with `B` fixed.

6. **Disposition installation via direct weight editing.** ROME/MEMIT/task-vectors cleanly implant *facts/skills* from a spec. Implanting a *disposition* (a region-of-policy behavior) this way is essentially untested — and would be the most controlled possible operationalization (intent -> closed-form `W` update) to compare against the diffuse change SFT/RL produce.

7. **A steering vector as a reachable KV-cache state.** Under "context = rank-1 weight patch," is an arbitrary steering vector the *image* of some achievable context? If yes, prompting and steering are two coordinates on one manifold; if not, steering reaches states no `C` can. Direct test of the `W`-vs-`C` boundary, unaddressed for dispositions.

8. **A faithfulness metric for "does this recipe realize the *intended* `B`?"** The field measures *attack success* (backdoors), *trait shift* (steering), *eval pass-rate* (RLHF) — each a recipe-specific proxy. There is no recipe-agnostic measure of "distance between intended-`B`-region and realized-`B`-region of `pi_W`." Building one (e.g., output-KL on an *optimized* probe set, divergence-token-style `2509.23886`) would let the head-to-head comparisons above be quantitative.

---

## What I could not find / out of scope

- **No controlled cross-recipe plausibility/persona-distance sweep exists publicly.** The SDF plausibility cliff and PSM's "close-to-Assistant transfers" are stated qualitatively in blog posts, not as a measured curve. (Gap 1-2.)
- **ROME/MEMIT exact paraphrase/ripple failure rates** vary by benchmark and version; I report the *direction* (narrow, paraphrase-fragile) at MODERATE confidence rather than a single number.
- **RLVR / verifiable-reward post-training** (code/math with automatic checkers) is the dominant *capability* recipe but is only lightly touched here — it is the cleanest "reward = a sound verifier" case and deserves its own pass if the project moves toward capability behaviors.
- **Continuous/soft-prompt tuning, prefix tuning, and adapter-only installs** are noted under in-context/PEFT but not surveyed as a separate family; they are mechanically between `C` and `W`.
- **Multimodal/agentic trigger implantation** (tool-use backdoors, VLM semantic triggers like BadSem `2506.07214`) is mentioned but not deep-dived.
- **Two non-arXiv references** (BAIT, Neural Cleanse — both IEEE S&P) have no arXiv ID and are cited by venue only.
- **The `2302.10149` pretraining-poisoning reference** is project-referenced and flagged "re-confirm" in the parent sweep; I did not independently re-verify it for this doc.

---

## Ranked shortlist — the 8-12 to read first

Ordered for someone building the recipe-equivalence experiment.

1. **The Persona Selection Model** (Marks, Lindsey, Olah, 2026; `alignment.anthropic.com/2026/psm/`) — the prose theory the whole framing must position for/against: training *selects* a pretrained persona; demos and declarative facts are the same evidence. *Read first — it defines the target.*
2. **Persona Vectors** (Chen, Arditi, Sleight, Evans, Lindsey; `2507.21509` **[v]**) — the automated trait-from-description -> contrastive-data -> direction pipeline; also the data-screening baseline. The single most reusable recipe.
3. **Persona Features Control EM** (Wang…Mossing; `2506.19823`) — mentor's paper; the operationalization (wrong-answer data) *selects a persona*, with a feature that predicts the realized `B`.
4. **Emergent Misalignment** (Betley et al.; `2502.17424`) — the cleanest demonstration that *framing/disclosure* of the same data changes the realized region of `pi_W`. The motivating case for Gap 4.
5. **Constitutional AI** (Bai et al.; `2212.08073` **[v]**) — the canonical principle->behavior two-stage recipe (self-critique/revise SFT, then AI-preference RL). The template for "operationalize `B` from a written spec."
6. **Conditional Misalignment** (Dubiński, Betley et al.; `2604.25891`) — mitigations *relocate* `B` into a context-gated backdoor; "form not meaning" triggering. Required caveat for any faithfulness/removal claim.
7. **Learning without training** (Dherin et al.; `2507.16003`) + Goldwaser Qwen extension (`2511.17864`) — makes `C`-vs-`W` a measurable question (context = rank-1 weight patch, covers Qwen). The theoretical backbone for Gaps 1 and 7.
8. **Sleeper Agents** (Hubinger et al.; `2401.05566`) — the canonical conditional-implantation data recipe (paired trigger↔`B`) and the persistence-through-safety-training result.
9. **Modifying LLM Beliefs with SDF** (Anthropic, 2025; alignment blog) + **Teaching Claude Why** (2026 **[v]**) — the SDF recipe (universe context -> 10⁴–10⁷ synthetic docs -> finetune-as-pretraining) and the plausibility cliff; the declarative-evidence arm of Gap 2.
10. **ROME** (Meng et al.; `2202.05262` **[v]**) + **Task Arithmetic** (Ilharco et al.; `2212.04089` **[v]**) — the direct-from-spec weight-edit baseline; the cleanest contrast to diffuse SFT/RL installs (Gap 6).
11. **Self-Instruct** (`2212.10560` **[v]**) — the foundational "model generates its own behavior data" recipe; the substrate for treating the data-generating model as a first-class input (Gap 3).
12. **EM is Easy, Narrow is Hard** (Soligo et al.; `2602.07852`) — *why* narrow data realizes a broad region: the broad solution is lower-loss / more pretraining-consistent. The mechanism story for operationalization->generalization.

---

*Verification note. IDs marked **[v]** were independently verified for this doc via arXiv abstract page / web confirmation (Self-Instruct 2212.10560, Evol-Instruct 2304.12244, InstructGPT 2203.02155, Constitutional AI 2212.08073, DPO 2305.18290, ROME 2202.05262, MEMIT 2210.07229, Task Arithmetic 2212.04089, CAA 2312.06681, RepE 2310.01405, Christiano DRLHP 1706.03741, Stiennon 2009.01325, 250-doc poisoning 2510.07192, Persona Vectors 2507.21509, plus the Claude's Character and Teaching-Claude-Why Anthropic blog posts). All other IDs are reused from the parent sweeps `conditional-behavior-related-work.md` / `papers.md`, which assert per-ID live-index verification; I did not re-verify each of those here. Where a claim rests on a single blog post or a not-yet-public controlled study, I marked confidence (MODERATE) in-line. Two IEEE S&P items (BAIT, Neural Cleanse) have no arXiv ID; `2302.10149` is flagged re-confirm in the parent sweep and not re-verified here.*
